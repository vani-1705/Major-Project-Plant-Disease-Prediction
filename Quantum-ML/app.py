"""
Quantum ML - Plant Disease Prediction  (Flask backend)
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os, re, io, csv, json, uuid, pickle, sqlite3, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash

BASE = Path(__file__).parent
MODELS, DATA, STATIC = BASE / "models", BASE / "data", BASE / "static"
UPLOADS, SAMPLES = STATIC / "uploads", STATIC / "samples"
UPLOADS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ------------------------------------------------------------------ helpers
def find(name):
    """Look for a file in models/ , data/ or project root."""
    for d in (MODELS, DATA, BASE):
        if (d / name).exists():
            return d / name
    return None

def load_json(name, default):
    p = find(name) or (STATIC / name)
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default

def tokens(s):
    return set(re.findall(r"[a-z]+", s.lower()))

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0

TIPS = load_json("tips.json", {})
KEYWORDS = load_json("keywords.json", {"crops": {}, "diseases": {}})

def to_key(label):
    """Map any class label (e.g. Potato___Early_blight) to a key in tips.json."""
    if not TIPS:
        return label
    t = tokens(label)
    return max(TIPS, key=lambda k: jaccard(t, tokens(k)))

# ------------------------------------------------------------------ classes
def load_classes():
    c = load_json("classes.json", [])
    if isinstance(c, dict):
        if all(str(k).isdigit() for k in c):
            return [c[k] for k in sorted(c, key=int)]
        if all(isinstance(v, int) for v in c.values()):
            return [k for k, _ in sorted(c.items(), key=lambda x: x[1])]
        return list(c.values())
    return c

CLASSES = load_classes()

# ------------------------------------------------------------------ models
MODEL_FILES = {
    "SVM": "svm.pkl", "Random Forest": "rf.pkl", "MLP": "mlp.pkl",
    "QSVM": "qsvm.model", "VQC": "VQC.model", "QNN": "qnn.model",
}
_cache = {}

def get_model(name):
    if name in _cache:
        return _cache[name]
    path = find(MODEL_FILES[name])
    if path is None:
        raise FileNotFoundError(f"{MODEL_FILES[name]} not found in models/")
    if name in ("SVM", "Random Forest", "MLP"):
        try:
            with open(path, "rb") as f:
                m = pickle.load(f)
        except Exception:
            import joblib
            m = joblib.load(path)
    else:
        from qiskit_machine_learning.algorithms import QSVC, VQC, NeuralNetworkClassifier
        cls = {"QSVM": QSVC, "VQC": VQC, "QNN": NeuralNetworkClassifier}[name]
        m = cls.load(str(path))
    _cache[name] = m
    return m

_pp = None
def get_pp():
    global _pp
    if _pp is None:
        path = find("preprocess.pkl")
        if path is None:
            raise FileNotFoundError("preprocess.pkl not found in models/")
        try:
            with open(path, "rb") as f:
                _pp = pickle.load(f)
        except Exception:
            import joblib
            _pp = joblib.load(path)
    return _pp

def expected_dim(pp):
    if isinstance(pp, dict):
        objs = list(pp.values())
    elif hasattr(pp, "steps"):
        objs = [s[1] for s in pp.steps]
    else:
        objs = [pp]
    for o in objs:
        n = getattr(o, "n_features_in_", None)
        if n:
            return int(n)
    return None

def apply_pp(pp, X):
    if hasattr(pp, "transform"):
        return pp.transform(X)
    if isinstance(pp, dict):
        for v in pp.values():
            if hasattr(v, "transform"):
                X = v.transform(X)
    return X

# >>> EDIT HERE if your Colab feature extraction is different <<<
def feature_candidates(img):
    """Yield feature vectors. The one whose length matches preprocess.pkl is used."""
    rgb, gray = img.convert("RGB"), img.convert("L")
    for s in (64, 32, 128, 224, 100, 48, 16, 8, 256):
        yield np.asarray(rgb.resize((s, s)), dtype=np.float32).ravel() / 255.0
        yield np.asarray(gray.resize((s, s)), dtype=np.float32).ravel() / 255.0
    arr = np.asarray(rgb.resize((128, 128)))
    for b in (8, 16, 32):
        yield np.concatenate([np.histogram(arr[..., c], bins=b, range=(0, 255))[0] for c in range(3)]).astype(np.float32)

def extract(img):
    pp = get_pp()
    n = expected_dim(pp)
    vec = None
    for v in feature_candidates(img):
        if n is None or len(v) == n:
            vec = v
            break
    if vec is None:
        raise ValueError(f"preprocess.pkl expects {n} features. Edit feature_candidates() in app.py to match your Colab code.")
    return np.asarray(apply_pp(pp, vec.reshape(1, -1)))

def run_model(name, X):
    m = get_model(name)
    p = np.asarray(m.predict(X))
    if p.dtype.kind in "US":
        label = str(p.ravel()[0])
    else:
        idx = int(np.argmax(p[0])) if p.ndim > 1 and p.shape[-1] > 1 else int(p.ravel()[0])
        label = CLASSES[idx] if CLASSES and 0 <= idx < len(CLASSES) else str(idx)
    conf = None
    if hasattr(m, "predict_proba"):
        try:
            conf = float(np.max(m.predict_proba(X)))
        except Exception:
            pass
    return label, conf

# ------------------------------------------------------------------ samples
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def sample_images(key, limit=3):
    if not SAMPLES.exists():
        return []
    want = tokens(key)
    scored = []
    for p in SAMPLES.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            s = jaccard(want, tokens(p.parent.name + " " + p.stem))
            if s > 0:
                scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    return ["/static/" + p.relative_to(STATIC).as_posix() for _, p in scored[:limit]]

def build_result(label, key, conf, model, image_url, source):
    info = TIPS.get(key, {})
    imgs = [image_url] if image_url else sample_images(key)
    return {
        "disease": info.get("name", label.replace("_", " ").strip()),
        "class": label, "confidence": conf, "model": model, "source": source,
        "images": imgs,
        "tips_en": info.get("tips_en", []), "tips_te": info.get("tips_te", []),
    }

# ------------------------------------------------------------------ DB
DB = DATA / "app.db"

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE,
        password TEXT, login_count INTEGER DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, created TEXT,
        disease TEXT, image TEXT, source TEXT, model TEXT)""")

def owner():
    return session.get("email", "guest")

def save_history(res, image):
    with db() as c:
        c.execute("INSERT INTO history(owner,created,disease,image,source,model) VALUES(?,?,?,?,?,?)",
                  (owner(), datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   res["disease"], image, res["source"], res["model"]))

# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    model = request.form.get("model", "QSVM")
    if f is None or model not in MODEL_FILES:
        return jsonify(error="Image or model missing"), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        name = uuid.uuid4().hex + ".jpg"
        img.save(UPLOADS / name, quality=90)
        url = f"/static/uploads/{name}"
        label, conf = run_model(model, extract(img))
        res = build_result(label, to_key(label), conf, model, url, "image")
        save_history(res, url)
        return jsonify(res)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.post("/api/predict_text")
def api_predict_text():
    text = (request.get_json(silent=True) or {}).get("text", "").strip().lower()
    if not text:
        return jsonify(error="Please enter or speak the problem"), 400
    crops = KEYWORDS.get("crops", {})
    best, best_score = None, 0
    for key, words in KEYWORDS.get("diseases", {}).items():
        score = sum(3 for w in words if w.lower() in text)
        if score == 0:
            continue
        for crop, cw in crops.items():
            if key.startswith(crop) and any(w.lower() in text for w in cw):
                score += 2
        if score > best_score:
            best, best_score = key, score
    if not best:
        return jsonify(error="Could not match a disease. Mention the plant (tomato / potato / pepper) and the symptoms."), 404
    res = build_result(best, best, None, "Keyword match", None, "text")
    save_history(res, res["images"][0] if res["images"] else "")
    return jsonify(res)

@app.get("/api/history")
def api_history():
    with db() as c:
        rows = c.execute("SELECT created,disease,image,source,model FROM history WHERE owner=? ORDER BY id DESC LIMIT 100",
                         (owner(),)).fetchall()
    return jsonify([dict(r) for r in rows])

# ---- model performance
ORDER = ["SVM", "Random Forest", "MLP", "QSVM", "VQC", "QNN"]
CANON = {"svm": "SVM", "rf": "Random Forest", "randomforest": "Random Forest", "mlp": "MLP",
         "qsvm": "QSVM", "qsvc": "QSVM", "vqc": "VQC", "qnn": "QNN"}

def canon(n):
    return CANON.get(re.sub(r"[^a-z]", "", str(n).lower()))

def pick(d, *names):
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        for k, v in low.items():
            if n in k:
                return v
    return None

def frac(v):
    try:
        v = float(v)
        return v / 100 if v > 1 else v
    except Exception:
        return None

def norm_entry(d):
    return {
        "accuracy": frac(pick(d, "accuracy", "acc")),
        "precision": frac(pick(d, "precision")),
        "recall": frac(pick(d, "recall")),
        "f1": frac(pick(d, "f1")),
        "confusion_matrix": pick(d, "confusion", "conf_m", "cm"),
        "curve": pick(d, "curve", "history"),
    }

@app.get("/api/model_performance")
def api_perf():
    out = {}
    raw = load_json("model_results.json", None) or load_json("model_performance.json", {})
    items = raw.items() if isinstance(raw, dict) else [(r.get("model") or r.get("Model"), r) for r in raw]
    for name, d in items:
        cn = canon(name)
        if cn and isinstance(d, dict):
            out[cn] = norm_entry(d)
    p = find("comparison_results.csv")
    if p:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cn = canon(pick(row, "model", "name", "unnamed") or "")
                if cn and cn not in out:
                    out[cn] = norm_entry(row)
    models = {k: out[k] for k in ORDER if k in out}
    return jsonify(models=models, order=list(models))

# ---- auth
@app.post("/api/signup")
def signup():
    d = request.get_json(silent=True) or {}
    name, email, pw = d.get("name", "").strip(), d.get("email", "").strip().lower(), d.get("password", "")
    if not name or "@" not in email or len(pw) < 4:
        return jsonify(error="Enter name, valid email and password (min 4 characters)"), 400
    try:
        with db() as c:
            c.execute("INSERT INTO users(name,email,password,login_count) VALUES(?,?,?,1)",
                      (name, email, generate_password_hash(pw)))
    except sqlite3.IntegrityError:
        return jsonify(error="Email already registered. Please login."), 409
    session["email"], session["name"] = email, name
    return jsonify(name=name, email=email, login_count=1)

@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    email, pw = d.get("email", "").strip().lower(), d.get("password", "")
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not u or not check_password_hash(u["password"], pw):
            return jsonify(error="Wrong email or password"), 401
        c.execute("UPDATE users SET login_count=login_count+1 WHERE id=?", (u["id"],))
        count = u["login_count"] + 1
    session["email"], session["name"] = email, u["name"]
    return jsonify(name=u["name"], email=email, login_count=count)

@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)

@app.get("/api/me")
def me():
    if "email" not in session:
        return jsonify(logged_in=False)
    with db() as c:
        u = c.execute("SELECT name,email,login_count FROM users WHERE email=?", (session["email"],)).fetchone()
    return jsonify(logged_in=True, **dict(u)) if u else jsonify(logged_in=False)

if __name__ == "__main__":
    app.run(debug=True)
