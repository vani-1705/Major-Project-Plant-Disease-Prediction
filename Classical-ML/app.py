import os, json, sqlite3, joblib, numpy as np, cv2
from datetime import datetime
from flask import Flask, request, jsonify, session, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from skimage.feature import hog

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")
CORS(app)
MODEL = os.environ.get("MODEL", "SVM")            # SVM, RandomForest or MLP
model = joblib.load(f"models/app_{MODEL}.pkl")
scaler = joblib.load("models/app_scaler.pkl")
pca = joblib.load("models/app_pca.pkl")
classes = json.load(open("models/app_classes.json"))
KW = json.load(open("data/keywords.json"))
DB = "plantcare.db"

with sqlite3.connect(DB) as c:
    c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, pw TEXT);
    CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, uid INTEGER, source TEXT, input TEXT, cls TEXT, conf REAL, at TEXT);""")

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close(_):
    d = g.pop("db", None)
    if d: d.close()

def features(img):   # must match Colab features() exactly
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = hog(gray, pixels_per_cell=(16, 16))
    hist = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).flatten()
    return np.concatenate([h, hist])

def save(src, inp, top):
    if session.get("uid") and top:
        db().execute("INSERT INTO history(uid,source,input,cls,conf,at) VALUES(?,?,?,?,?,?)",
                     (session["uid"], src, inp, top["cls"], top["conf"], datetime.now().strftime("%d %b %Y, %I:%M %p")))
        db().commit()

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if not f: return jsonify(error="No image received"), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify(error="Could not read this image"), 400
    p = model.predict_proba(pca.transform(scaler.transform([features(img)])))[0]
    top = [{"cls": classes[i], "conf": round(float(p[i]) * 100, 1)} for i in np.argsort(p)[::-1][:3]]
    if top[0]["conf"] < 30:
        return jsonify(top=[], message="Not sure. Upload a clear leaf of tomato, potato or pepper.")
    save("image", f.filename, top[0])
    return jsonify(top=top)

@app.post("/api/text")
def text():
    d = request.get_json(force=True); t = (d.get("text") or "").lower(); src = d.get("source", "text")
    plants = [p for p, w in KW["_plants"].items() if any(x in t for x in w)]
    sc = {k: sum(x in t for x in w) for k, w in KW.items() if k != "_plants"}
    sc = {k: v for k, v in sc.items() if v}
    if not sc: return jsonify(top=[], message="No match. Describe the colour, spots and leaf changes.")
    tot, out = sum(sc.values()), []
    for k, v in sorted(sc.items(), key=lambda x: -x[1])[:3]:
        c = [x for x in classes if k in x.lower() and (not plants or any(x.lower().startswith(p) for p in plants))] \
            or [x for x in classes if k in x.lower()]
        if c: out.append({"cls": c[0], "conf": round(100 * v / tot, 1)})
    save(src, t[:200], out[0] if out else None)
    return jsonify(top=out)

@app.post("/api/register")
def register():
    d = request.get_json(force=True)
    u, e, p = (d.get(k, "").strip() for k in ("username", "email", "password"))
    if not (u and "@" in e and len(p) >= 6): return jsonify(error="Enter username, valid email and a password of 6+ characters"), 400
    try:
        cur = db().execute("INSERT INTO users(username,email,pw) VALUES(?,?,?)", (u, e, generate_password_hash(p)))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="Username or email already exists"), 400
    session["uid"] = cur.lastrowid
    return jsonify(user=u)

@app.post("/api/login")
def login():
    d = request.get_json(force=True)
    r = db().execute("SELECT * FROM users WHERE username=? OR email=?", (d.get("username", ""), d.get("email", ""))).fetchone()
    if not r or not check_password_hash(r["pw"], d.get("password", "")):
        return jsonify(error="Wrong username/email or password"), 401
    session["uid"] = r["id"]
    return jsonify(user=r["username"])

@app.post("/api/logout")
def logout():
    session.clear(); return jsonify(ok=True)

@app.get("/api/me")
def me():
    r = db().execute("SELECT username,email FROM users WHERE id=?", (session.get("uid"),)).fetchone()
    return jsonify(user=r["username"] if r else None, email=r["email"] if r else None)

@app.get("/api/history")
def history():
    if not session.get("uid"): return jsonify(error="Please login to see your history"), 401
    rows = db().execute("SELECT source,input,cls,conf,at FROM history WHERE uid=? ORDER BY id DESC LIMIT 100", (session["uid"],)).fetchall()
    return jsonify(items=[dict(r) for r in rows])

@app.get("/api/performance")
def performance():
    return app.response_class(open("data/model_performance.json").read(), mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)
