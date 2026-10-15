const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 2600);
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  let d = {};
  try { d = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(d.error || `Request failed (HTTP ${r.status}) - server timeout/crash, check Render logs`);
  return d;
}
const postJSON = (url, body) =>
  api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ---------------- navigation ---------------- */
function show(id) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === id));
  $$(".nav a").forEach(a => a.classList.toggle("active", a.dataset.page === id));
  window.scrollTo(0, 0);
  if (id === "graphs") loadGraphs();
  if (id === "history") loadHistory();
  if (id === "user") loadMe();
}
$$(".nav a").forEach(a => a.addEventListener("click", e => { e.preventDefault(); show(a.dataset.page); }));

/* ---------------- result card ---------------- */
function renderResult(d) {
  const box = $("#result");
  box.hidden = false;
  const imgs = (d.images || []).map(u => `<img src="${esc(u)}" alt="${esc(d.disease)}">`).join("");
  const li = a => (a || []).map(t => `<li>${esc(t)}</li>`).join("");
  box.innerHTML = `
    <div class="r-grid">
      <div class="r-imgs">${imgs || "<p>No sample image found</p>"}</div>
      <div>
        <h3>🌿 ${esc(d.disease)}</h3>
        <span class="badge">Model: ${esc(d.model)}</span>
        ${d.confidence != null ? `<span class="badge ok">Confidence: ${(d.confidence * 100).toFixed(1)}%</span>` : ""}
        <div class="tips">
          <h5>Health Tips</h5><ul>${li(d.tips_en)}</ul>
          <h5>ఆరోగ్య సూచనలు</h5><ul>${li(d.tips_te)}</ul>
        </div>
      </div>
    </div>`;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}
function renderError(msg) {
  const box = $("#result");
  box.hidden = false;
  box.innerHTML = `<p class="err">⚠️ ${esc(msg)}</p>`;
}
function busy(btn, on, label) {
  btn.disabled = on;
  if (on) { btn.dataset.l = btn.textContent; btn.textContent = "Predicting..."; }
  else btn.textContent = btn.dataset.l || label;
}

/* ---------------- 1. image source ---------------- */
$("#chooseBtn").onclick = () => $("#imgFile").click();
$("#imgFile").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const p = $("#preview");
  p.src = URL.createObjectURL(f); p.hidden = false;
  $("#chooseBtn").textContent = f.name.length > 22 ? f.name.slice(0, 20) + "…" : f.name;
};
$("#predictImg").onclick = async function () {
  const f = $("#imgFile").files[0];
  if (!f) return toast("Choose an image first");
  const fd = new FormData();
  fd.append("image", f); fd.append("model", $("#modelSel").value);
  busy(this, true);
  try { renderResult(await api("/api/predict", { method: "POST", body: fd })); }
  catch (e) { renderError(e.message); }
  busy(this, false);
};

/* ---------------- 2. text source ---------------- */
async function predictText(text, btn) {
  if (!text.trim()) return toast("Type or speak the problem first");
  busy(btn, true);
  try { renderResult(await postJSON("/api/predict_text", { text })); }
  catch (e) { renderError(e.message); }
  busy(btn, false);
}
$("#predictTxt").onclick = function () { predictText($("#txt").value, this); };

/* ---------------- 3. voice source ---------------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
$("#speakBtn").onclick = function () {
  if (!SR) return toast("Voice input needs Chrome or Edge");
  if (listening) { rec.stop(); return; }
  rec = new SR();
  rec.lang = $("#voiceLang").value;
  rec.interimResults = true;
  rec.onstart = () => { listening = true; this.textContent = "⏹️ Stop"; this.classList.add("rec"); };
  rec.onend = () => { listening = false; this.textContent = "🎙️ Speak"; this.classList.remove("rec"); };
  rec.onerror = e => toast("Mic error: " + e.error);
  rec.onresult = e => {
    $("#voiceTxt").value = [...e.results].map(r => r[0].transcript).join(" ");
  };
  rec.start();
};
$("#predictVoice").onclick = function () { predictText($("#voiceTxt").value, this); };

/* ---------------- model graphs ---------------- */
let charts = [];
const COLORS = { SVM: "#6366f1", "Random Forest": "#10b981", MLP: "#f59e0b", QSVM: "#8b5cf6", VQC: "#ec4899", QNN: "#0ea5e9" };

function metricBar(label, v, c) {
  if (v == null) return "";
  return `<div class="metric"><span>${label}</span><div class="bar"><b style="width:${v * 100}%;background:${c}"></b></div><span>${v.toFixed(2)}</span></div>`;
}
function matrixHTML(m) {
  if (!Array.isArray(m) || !Array.isArray(m[0])) return "";
  const max = Math.max(...m.flat(), 1);
  const cells = m.flat().map(v => `<i title="${v}" style="--a:${(0.08 + 0.92 * v / max).toFixed(2)}"></i>`).join("");
  return `<div class="sub">Confusion Matrix</div><div class="cm" style="grid-template-columns:repeat(${m.length},1fr)">${cells}</div>`;
}
async function loadGraphs() {
  const grid = $("#graphGrid");
  charts.forEach(c => c.destroy()); charts = [];
  let d;
  try { d = await api("/api/model_performance"); }
  catch (e) { grid.innerHTML = `<div class="card err">${esc(e.message)}</div>`; return; }
  if (!d.order.length) { grid.innerHTML = `<div class="card"><p>No results found. Put model_results.json or comparison_results.csv in the data/ folder.</p></div>`; return; }

  // comparison bar chart
  const keys = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]];
  const palette = ["#1d4ed8", "#059669", "#f59e0b", "#be185d"];
  charts.push(new Chart($("#cmpChart"), {
    type: "bar",
    data: {
      labels: d.order,
      datasets: keys.map(([k, n], i) => ({ label: n, backgroundColor: palette[i], data: d.order.map(m => d.models[m][k]) }))
    },
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } } }
  }));

  // per-model cards
  grid.innerHTML = d.order.map((m, i) => {
    const x = d.models[m], c = COLORS[m];
    return `<div class="card">
      <h3 class="m-title">${esc(m)}</h3>
      <div class="sub">Accuracy</div><canvas id="ch${i}" height="130"></canvas>
      ${matrixHTML(x.confusion_matrix)}
      ${metricBar("Accuracy", x.accuracy, c)}${metricBar("Precision", x.precision, c)}
      ${metricBar("Recall", x.recall, c)}${metricBar("F1 Score", x.f1, c)}
    </div>`;
  }).join("");

  d.order.forEach((m, i) => {
    const x = d.models[m], c = COLORS[m];
    const hasCurve = Array.isArray(x.curve) && x.curve.length > 1 && typeof x.curve[0] === "number";
    const data = hasCurve ? x.curve : [x.accuracy, x.precision, x.recall, x.f1];
    const labels = hasCurve ? data.map((_, k) => k + 1) : ["Acc", "Prec", "Rec", "F1"];
    charts.push(new Chart($("#ch" + i), {
      type: "line",
      data: { labels, datasets: [{ data, borderColor: c, backgroundColor: c, tension: .3, pointRadius: 3 }] },
      options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } }
    }));
  });
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const body = $("#histBody");
  try {
    const rows = await api("/api/history");
    $("#histMsg").textContent = rows.length ? "" : "No predictions yet. Go to Home and predict a disease.";
    body.innerHTML = rows.map(r => `<tr>
      <td>${esc(r.created)}</td><td>${esc(r.disease)}</td>
      <td>${r.image ? `<img src="${esc(r.image)}" alt="">` : "-"}</td></tr>`).join("");
  } catch (e) { $("#histMsg").textContent = e.message; }
}

/* ---------------- user ---------------- */
function setUser(u) {
  const on = u && u.logged_in !== false && u.email;
  $("#formBox").hidden = !!on;
  $("#authBtns").hidden = !!on;
  $("#loggedBox").hidden = !on;
  $("#logoutBtn").hidden = !on;
  $("#loginCount").textContent = on ? u.login_count : 0;
  if (on) { $("#uName").textContent = u.name; $("#uEmail").textContent = u.email; }
}
async function loadMe() { try { setUser(await api("/api/me")); } catch (e) {} }

async function auth(kind) {
  const msg = $("#authMsg");
  msg.className = "msg"; msg.textContent = "";
  try {
    const u = await postJSON("/api/" + kind, {
      name: $("#fName").value, email: $("#fEmail").value, password: $("#fPass").value
    });
    setUser(u);
    msg.className = "msg ok"; msg.textContent = kind === "signup" ? "Account created!" : "Login successful!";
    $("#fPass").value = "";
  } catch (e) { msg.textContent = e.message; }
}
$("#signupBtn").onclick = () => auth("signup");
$("#loginBtn").onclick = () => auth("login");
$("#logoutBtn").onclick = async () => {
  await postJSON("/api/logout", {});
  setUser(null); $("#authMsg").textContent = "";
  toast("Logged out");
};
$("#eye").onclick = () => { const p = $("#fPass"); p.type = p.type === "password" ? "text" : "password"; };

loadMe();
