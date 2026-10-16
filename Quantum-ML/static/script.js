/* Quantum ML - Plant Disease Prediction (frontend) */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------- helpers ---------------- */
function toast(msg, ms = 2800) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

async function api(url, opts = {}) {
  let r;
  try { r = await fetch(url, opts); }
  catch (e) { throw new Error("Cannot reach the server. Please try again."); }
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `Server error (${r.status})`);
  return data;
}
const postJSON = (url, body) => api(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/* ---------------- navigation ---------------- */
function show(page) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === page));
  $$(".nav a").forEach(a => a.classList.toggle("active", a.dataset.page === page));
  window.scrollTo(0, 0);
  if (page === "graphs") loadGraphs();
  if (page === "history") loadHistory();
  if (page === "user") loadMe();
}
$$(".nav a").forEach(a => a.addEventListener("click", e => {
  e.preventDefault();
  show(a.dataset.page);
}));

/* ---------------- result ---------------- */
function showErr(msg) {
  const box = $("#result");
  box.innerHTML = `<p class="err">⚠️ ${esc(msg)}</p>`;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderResult(r) {
  const box = $("#result");
  const conf = r.confidence != null
    ? `<span class="badge ok">Confidence: ${(r.confidence * 100).toFixed(1)}%</span>` : "";
  const imgs = (r.images || []).map(u => `<img src="${esc(u)}" alt="">`).join("");
  const list = a => `<ul>${a.map(t => `<li>${esc(t)}</li>`).join("")}</ul>`;
  const en = r.tips_en && r.tips_en.length ? `<h5>Health Tips</h5>${list(r.tips_en)}` : "";
  const te = r.tips_te && r.tips_te.length ? `<h5>ఆరోగ్య సూచనలు</h5>${list(r.tips_te)}` : "";
  box.innerHTML = `
    <div class="r-grid">
      <div class="r-imgs">${imgs}</div>
      <div>
        <h3>${esc(r.disease)}</h3>
        <span class="badge">Model: ${esc(r.model)}</span>${conf}
        <div class="tips">${en || te ? en + te : "<p>No tips available for this disease.</p>"}</div>
      </div>
    </div>`;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function withBusy(btn, fn) {
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Predicting...";
  try { await fn(); }
  catch (e) { showErr(e.message); }
  finally { btn.disabled = false; btn.textContent = old; }
}

/* ---------------- image source ---------------- */
$("#chooseBtn").onclick = () => $("#imgFile").click();
$("#imgFile").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const p = $("#preview");
  p.src = URL.createObjectURL(f);
  p.hidden = false;
};
$("#predictImg").onclick = function () {
  const f = $("#imgFile").files[0];
  if (!f) return toast("Please choose an image first");
  withBusy(this, async () => {
    const fd = new FormData();
    fd.append("image", f);
    fd.append("model", $("#modelSel").value);
    renderResult(await api("/api/predict", { method: "POST", body: fd }));
  });
};

/* ---------------- text + voice source ---------------- */
function predictText(btn, area) {
  const text = area.value.trim();
  if (!text) return toast("Please enter or speak the problem");
  withBusy(btn, async () => renderResult(await postJSON("/api/predict_text", { text })));
}
$("#predictTxt").onclick = function () { predictText(this, $("#txt")); };
$("#predictVoice").onclick = function () { predictText(this, $("#voiceTxt")); };

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
$("#speakBtn").onclick = function () {
  const btn = this;
  if (!SR) return toast("Voice input is not supported in this browser. Use Chrome (https or localhost).");
  if (rec) { rec.stop(); return; }
  rec = new SR();
  rec.lang = $("#voiceLang").value;
  rec.interimResults = false;
  rec.continuous = false;
  btn.classList.add("rec");
  btn.textContent = "⏹ Stop";
  rec.onresult = e => {
    $("#voiceTxt").value = Array.from(e.results).map(r => r[0].transcript).join(" ");
  };
  rec.onerror = e => toast("Voice error: " + e.error);
  rec.onend = () => {
    rec = null;
    btn.classList.remove("rec");
    btn.textContent = "🎙️ Speak";
  };
  rec.start();
};

/* ---------------- model graphs ---------------- */
let chart = null;
const COLORS = ["#1d4ed8", "#059669", "#be185d", "#d97706"];

function modelCard(name, m) {
  let cmHtml = "";
  const cm = m.confusion_matrix;
  if (Array.isArray(cm) && cm.length && Array.isArray(cm[0])) {
    const max = Math.max(...cm.flat(), 1);
    const cells = cm.flat().map(v => {
      const a = 0.1 + 0.9 * v / max;
      return `<i title="${v}" style="--a:${a.toFixed(2)};display:flex;align-items:center;justify-content:center;` +
             `font-size:.75rem;font-style:normal;color:${a > 0.5 ? "#fff" : "#10203f"}">${v}</i>`;
    }).join("");
    cmHtml = `<div class="sub">Confusion matrix</div>` +
             `<div class="cm" style="grid-template-columns:repeat(${cm.length},1fr)">${cells}</div>`;
  }
  const bar = (label, v, color) => {
    const p = v == null ? 0 : Math.round(v * 1000) / 10;
    return `<div class="metric"><span>${label}</span>` +
           `<div class="bar"><b style="width:${p}%;background:${color}"></b></div>` +
           `<span>${v == null ? "-" : p + "%"}</span></div>`;
  };
  return `<div class="card"><h3 class="m-title">${esc(name)}</h3>${cmHtml}
    <div class="sub">Scores</div>
    ${bar("Accuracy", m.accuracy, COLORS[0])}
    ${bar("Precision", m.precision, COLORS[1])}
    ${bar("Recall", m.recall, COLORS[2])}
    ${bar("F1", m.f1, COLORS[3])}</div>`;
}

async function loadGraphs() {
  const grid = $("#graphGrid");
  try {
    const d = await api("/api/model_performance");
    const names = d.order || [];
    if (chart) { chart.destroy(); chart = null; }
    if (!names.length) {
      grid.innerHTML = `<div class="card"><p>No results found. Copy model_results.json and comparison_results.csv into the models/ folder.</p></div>`;
      return;
    }
    if (window.Chart) {
      const keys = ["accuracy", "precision", "recall", "f1"];
      const labels = ["Accuracy", "Precision", "Recall", "F1"];
      chart = new Chart($("#cmpChart"), {
        type: "bar",
        data: {
          labels: names,
          datasets: keys.map((k, i) => ({
            label: labels[i],
            backgroundColor: COLORS[i],
            data: names.map(n => d.models[n][k] ?? 0),
          })),
        },
        options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
      });
    }
    grid.innerHTML = names.map(n => modelCard(n, d.models[n])).join("");
  } catch (e) {
    toast(e.message);
  }
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const body = $("#histBody"), msg = $("#histMsg");
  try {
    const rows = await api("/api/history");
    msg.textContent = rows.length ? "" : "No predictions yet.";
    body.innerHTML = rows.map(r =>
      `<tr><td>${esc(r.created)}</td><td>${esc(r.disease)}</td>` +
      `<td>${r.image ? `<img src="${esc(r.image)}" alt="">` : "-"}</td></tr>`).join("");
  } catch (e) {
    msg.textContent = e.message;
  }
}

/* ---------------- user account ---------------- */
function authMsg(text, ok) {
  const m = $("#authMsg");
  m.textContent = text;
  m.className = "msg" + (ok ? " ok" : "");
}

function setUser(u) {
  const on = !!(u && u.logged_in !== false && u.email);
  $("#loggedBox").hidden = !on;
  $("#formBox").hidden = on;
  $("#authBtns").hidden = on;
  $("#logoutBtn").hidden = !on;
  $(".or").hidden = on;
  $("#loginCount").textContent = on ? (u.login_count ?? 0) : 0;
  if (on) {
    $("#uName").textContent = u.name;
    $("#uEmail").textContent = u.email;
  }
}

async function loadMe() {
  try { setUser(await api("/api/me")); } catch (e) { /* ignore */ }
}

async function auth(kind) {
  const body = { email: $("#fEmail").value, password: $("#fPass").value };
  if (kind === "signup") body.name = $("#fName").value;
  try {
    const u = await postJSON("/api/" + kind, body);
    setUser({ ...u, logged_in: true });
    authMsg(kind === "signup" ? "Account created" : "Login successful", true);
    $("#fPass").value = "";
  } catch (e) {
    authMsg(e.message, false);
  }
}

$("#signupBtn").onclick = () => auth("signup");
$("#loginBtn").onclick = () => auth("login");
$("#logoutBtn").onclick = async () => {
  try { await postJSON("/api/logout", {}); } catch (e) { /* ignore */ }
  setUser(null);
  authMsg("Logged out", true);
};
$("#eye").onclick = () => {
  const p = $("#fPass");
  p.type = p.type === "password" ? "text" : "password";
};

loadMe();
