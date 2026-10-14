const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const KEYS = ["yellowleaf","mosaic","spider","target","septoria","leaf_mold","late_blight","early_blight","bacterial","healthy"];
let TIPS = {}, USER = null, PERF = null, perfLoaded = false, charts = [];
const lang = () => $("#lang").value;
const key = c => KEYS.find(k => c.toLowerCase().includes(k));
const plant = c => ["tomato","potato","pepper"].find(p => c.toLowerCase().startsWith(p)) || "";
const label = c => { const t = TIPS[key(c)], p = TIPS._plants[plant(c)];
  return (t ? (t.name[lang()] || t.name.en) : c) + (p ? " (" + (p[lang()] || p.en) + ")" : ""); };
const tips = c => { const t = TIPS[key(c)]; return t ? (t.tips[lang()] || t.tips.en) : []; };
const sample = c => `samples/${c}_1.jpg`;
async function api(u, o) { const r = await fetch(u, o); const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || "Request failed"); return d; }
const post = (u, b) => api(u, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(b)});

/* navigation */
function show(id) {
  $$(".sec").forEach(s => s.classList.toggle("on", s.id === id));
  $$("header a").forEach(a => a.classList.toggle("on", a.dataset.s === id));
  if (id === "hist") loadHist(); if (id === "perf") loadPerf(); if (id === "user") drawUser();
}
$$("header a").forEach(a => a.onclick = () => show(a.dataset.s));

/* results (shared by image, text, voice) */
function showResults(d, msgEl) {
  const box = $("#results");
  if (!d.top || !d.top.length) { msgEl.textContent = d.message || "No result."; box.style.display = "none"; return; }
  msgEl.textContent = "";
  const best = d.top[0].cls;
  box.style.display = "block";
  box.innerHTML = `<h2>Prediction Results</h2>` + d.top.map((r, i) => `<div class="res">
    <img src="${sample(r.cls)}" alt="" onerror="this.style.visibility='hidden'">
    <div><b>${i + 1}. ${label(r.cls)}</b><div class="bar"><i style="width:${r.conf}%"></i></div><span class="msg">${r.conf}%</span></div></div>`).join("") +
    `<h3>Health Tips: ${label(best)}</h3><ul>${tips(best).map(t => `<li>${t}</li>`).join("")}</ul>
     <button class="o" id="listen">🔊 Listen</button>`;
  $("#listen").onclick = () => { const u = new SpeechSynthesisUtterance(label(best) + ". " + tips(best).join(". "));
    u.lang = lang() === "te" ? "te-IN" : "en-IN"; speechSynthesis.speak(u); };
  box.scrollIntoView({behavior: "smooth"});
}

/* home: image, text, voice */
$("#file").onchange = e => { const f = e.target.files[0]; if (!f) return;
  $("#preview").src = URL.createObjectURL(f); $("#preview").style.display = "block"; };
$("#goImg").onclick = async () => {
  const f = $("#file").files[0]; if (!f) { $("#mImg").textContent = "Choose an image first."; return; }
  $("#mImg").textContent = "Analysing…"; const fd = new FormData(); fd.append("image", f);
  try { showResults(await api("/api/predict", {method: "POST", body: fd}), $("#mImg")); }
  catch (e) { $("#mImg").textContent = e.message + " (is the backend running?)"; }
};
async function textPredict(text, source, el) {
  if (!text.trim()) { el.textContent = "Enter or speak the problem first."; return; }
  try { showResults(await post("/api/text", {text, source}), el); } catch (e) { el.textContent = e.message; }
}
$("#goTxt").onclick = () => textPredict($("#txt").value, "text", $("#mTxt"));
$("#goVoice").onclick = () => textPredict($("#voiceTxt").value, "voice", $("#mVoice"));
$("#mic").onclick = () => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("#mVoice").textContent = "Voice needs Chrome or Edge."; return; }
  const r = new SR(); r.lang = lang() === "te" ? "te-IN" : "en-IN"; $("#mVoice").textContent = "Listening…";
  r.onresult = e => { $("#voiceTxt").value = e.results[0][0].transcript; $("#mVoice").textContent = "Recognised. Tap Predict Disease."; };
  r.onerror = e => $("#mVoice").textContent = "Voice error: " + e.error;
  r.start();
};

/* history */
let histTab = "image";
$$(".tabs button").forEach(b => b.onclick = () => { histTab = b.dataset.t; $$(".tabs button").forEach(x => x.classList.toggle("on", x === b)); loadHist(); });
async function loadHist() {
  const el = $("#histList");
  try { const {items} = await api("/api/history"); const rows = items.filter(x => x.source === histTab);
    el.innerHTML = rows.length ? rows.map(x => `<div class="row"><img src="${sample(x.cls)}" alt="" onerror="this.style.visibility='hidden'">
      <div style="flex:1"><b>${label(x.cls)}</b><br><span class="msg">${x.input}</span></div><div class="msg">${x.conf}%<br>${x.at}</div></div>`).join("")
      : "<p class='msg'>No predictions yet for this source.</p>";
  } catch (e) { el.innerHTML = `<p class="msg">${e.message}. Go to the User section to login.</p>`; }
}

/* user and header */
async function refreshUser() { const d = await api("/api/me"); USER = d.user; drawAcct(); drawUser(); }
function drawAcct() {
  $("#acct").innerHTML = USER ? `👤 ${USER} <button class="o" id="lo1">Logout</button>`
    : `👤 <a id="si">Sign up</a> <a id="li">Login</a>`;
  if (USER) $("#lo1").onclick = logout; else $("#si").onclick = $("#li").onclick = () => show("user");
}
async function logout() { await post("/api/logout", {}); USER = null; drawAcct(); drawUser(); }
function drawUser() {
  const b = $("#userBox");
  if (USER) { b.innerHTML = `<p>Welcome back, <b>${USER}</b>! Your predictions are saved in History.</p><button class="o" id="lo2">Logout</button>`; $("#lo2").onclick = logout; return; }
  b.innerHTML = `<label>Username<input id="u"></label><label>Email<input id="e" type="email"></label>
    <label>Password<input id="p" type="password"></label><button id="su">Sign up</button> <button class="o" id="lg">Login</button><p class="msg" id="um"></p>`;
  const go = (url) => async () => { try { await post(url, {username: $("#u").value, email: $("#e").value, password: $("#p").value}); await refreshUser(); show("home"); }
    catch (e) { $("#um").textContent = e.message; } };
  $("#su").onclick = go("/api/register"); $("#lg").onclick = go("/api/login");
}

/* model performance */
async function loadPerf() {
  if (perfLoaded) return;
  try { PERF = await api("/api/performance"); } catch (e) { $("#perf .card").insertAdjacentHTML("beforeend", "<p class='msg'>model_performance.json not found in data/ folder.</p>"); return; }
  perfLoaded = true; const names = Object.keys(PERF.models), mets = ["accuracy","precision","recall","f1"];
  new Chart($("#bar"), {type: "bar", data: {labels: ["Accuracy","Precision","Recall","F1"],
    datasets: names.map(n => ({label: n, data: mets.map(m => +(PERF.models[n][m] * 100).toFixed(1))}))},
    options: {scales: {y: {min: 0, max: 100}}, plugins: {title: {display: true, text: "Accuracy, Precision, Recall, F1 (%)"}}}});
  const lc = PERF.learning_curve;
  new Chart($("#curve"), {type: "line", data: {labels: lc.sizes, datasets: [
    {label: "Training accuracy", data: lc.train, borderColor: "#1d6fb8"}, {label: "Validation accuracy", data: lc.val, borderColor: "#2e9e5b"}]},
    options: {scales: {y: {min: 0, max: 1}}, plugins: {title: {display: true, text: "Model Learning Curve"}}}});
  $("#cmModel").innerHTML = names.map(n => `<option>${n}</option>`).join(""); $("#cmModel").onchange = drawCM; drawCM();
}
function drawCM() {
  const m = PERF.models[$("#cmModel").value].confusion_matrix, mx = Math.max(...m.flat());
  $("#cm").innerHTML = "<table><tr><th></th>" + m.map((_, i) => `<th>${i + 1}</th>`).join("") + "</tr>" +
    m.map((r, i) => `<tr><th>${i + 1}</th>` + r.map(v => `<td style="background:rgba(31,122,77,${(v / mx).toFixed(2)});color:${v / mx > .5 ? "#fff" : "#000"}">${v}</td>`).join("") + "</tr>").join("") + "</table>";
  $("#cmLegend").textContent = "Rows = actual, columns = predicted. " + PERF.classes.map((c, i) => `${i + 1}=${label(c)}`).join(", ");
}

fetch("tips.json").then(r => r.json()).then(t => { TIPS = t; refreshUser(); show("home"); });
