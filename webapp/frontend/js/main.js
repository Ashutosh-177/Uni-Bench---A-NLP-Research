import { createBarChart3D, seriesColorHex } from "./chart3d.js";

const API = "/api";

// ============================================================
// tiny fetch helpers
// ============================================================
async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `GET ${path} failed (${r.status})`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `POST ${path} failed (${r.status})`);
  return r.json();
}
async function apiPut(path, body) {
  const r = await fetch(API + path, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `PUT ${path} failed (${r.status})`);
  return r.json();
}

// ============================================================
// toast
// ============================================================
function toast(message, type = "") {
  const root = document.getElementById("toast-root");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  root.appendChild(el);
  gsap.fromTo(el, { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.3, ease: "power2.out" });
  setTimeout(() => {
    gsap.to(el, { opacity: 0, y: -8, duration: 0.25, onComplete: () => el.remove() });
  }, 4200);
}

// ============================================================
// app state
// ============================================================
const state = {
  config: null,
  registry: null,
  groqModels: null,
  colorOf: new Map(), // model name -> palette index (stable per config order)
};

function refreshColorMap() {
  state.colorOf.clear();
  (state.config?.models || []).forEach((m, i) => state.colorOf.set(m.name, i));
}

// ============================================================
// view routing
// ============================================================
const views = ["overview", "configure", "run", "calibrate", "results"];

function goTo(name) {
  views.forEach((v) => {
    const el = document.getElementById(`view-${v}`);
    if (v === name) {
      el.hidden = false;
      gsap.fromTo(el, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" });
    } else {
      el.hidden = true;
    }
  });
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === name);
  });
  if (name === "overview") loadOverview();
  if (name === "configure") loadConfigure();
  if (name === "results") loadResults();
  if (name === "calibrate") loadCalibrateHistory();
}

// Shows what calibration data already exists (from a previous CLI or
// dashboard session) so this view doesn't read as "nothing has been
// calibrated yet" right next to a Results view that clearly has been --
// the two surfaces read the exact same /results/summary corrections.
async function loadCalibrateHistory() {
  const el = document.getElementById("calib-history");
  try {
    const data = await apiGet("/results/summary");
    const calibrated = Object.entries(data.corrections).filter(([, c]) => c.method !== "uncalibrated");
    if (!calibrated.length) { el.innerHTML = ""; return; }
    const rows = calibrated.map(([name, c]) => `
      <div class="flex flex-gap-2" style="align-items:baseline; margin-bottom:6px">
        <span class="pill pill-good">${name}</span>
        <span class="text-sm muted">${c.detail}</span>
      </div>`).join("");
    el.innerHTML = `
      <div class="panel mt-5">
        <div class="panel-title">Already calibrated (from a previous session)</div>
        <div class="mt-4">${rows}</div>
        <p class="text-sm muted mt-4">New ratings below add to this sample and are folded into every score in Results immediately.</p>
      </div>`;
  } catch (e) {
    el.innerHTML = "";
  }
}

document.getElementById("nav-tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".nav-tab");
  if (tab) goTo(tab.dataset.view);
});
document.querySelectorAll("[data-nav]").forEach((btn) => {
  btn.addEventListener("click", () => goTo(btn.dataset.nav));
});

// ============================================================
// OVERVIEW
// ============================================================
async function loadOverview() {
  try {
    state.config = await apiGet("/config");
    refreshColorMap();
    document.getElementById("stat-models").textContent = state.config.models.length;
    document.getElementById("stat-tasks").textContent = state.config.tasks.length;
  } catch (e) { /* config missing is fine on first load */ }

  try {
    const summary = await apiGet("/results/summary");
    document.getElementById("stat-records").textContent = summary.n_records;
    const nCalibrated = Object.values(summary.corrections).filter(c => c.method !== "uncalibrated").length;
    document.getElementById("stat-calibrated").textContent = `${nCalibrated} / ${summary.judge_names.length}`;
  } catch (e) {
    document.getElementById("stat-records").textContent = "0";
    document.getElementById("stat-calibrated").textContent = "—";
  }
}

// ============================================================
// CONFIGURE
// ============================================================
async function loadConfigure() {
  if (!state.registry) state.registry = await apiGet("/registry");
  if (!state.config) state.config = await apiGet("/config");
  refreshColorMap();
  renderModelList();
  renderTaskList();
  renderJudgeList();

  // Live Groq model list is a real network round-trip (~0.5s+) to Groq's
  // own API -- it's a nice-to-have autocomplete enhancement, not required
  // to render the form, so it must NEVER block the initial paint above.
  // Fire-and-forget; just refresh the datalist + re-render when it lands.
  apiGet("/groq/models")
    .then((models) => { state.groqModels = models; renderModelList(); })
    .catch(() => { state.groqModels = null; }); // GROQ_API_KEY missing/invalid -- free-text input still works

  document.getElementById("cfg-temperature").value = state.config.run?.temperature ?? 0.0;
  document.getElementById("cfg-max-tokens").value = state.config.run?.max_tokens ?? 800;
}

const TASK_META = {
  fairness: { name: "Fairness probe", desc: "CFE-style contrastive pairs across gender, name/ethnicity, age, religion, disability" },
  summarization: { name: "Summarization", desc: "Hand-authored article set scored with ROUGE-1/2/L against reference summaries" },
};

function renderModelList() {
  const list = document.getElementById("model-list");
  list.innerHTML = "";
  document.getElementById("model-count-sub").textContent = `${state.config.models.length} model${state.config.models.length === 1 ? "" : "s"}`;

  state.config.models.forEach((m, i) => {
    const tpl = document.getElementById("tpl-model-row").content.cloneNode(true);
    const row = tpl.querySelector(".model-row");
    row.querySelector(".swatch").style.background = seriesColorHex(i);
    row.querySelector(".model-name").value = m.name;
    row.querySelector(".model-id").value = m.model_id;

    const providerSelect = row.querySelector(".model-provider");
    providerSelect.innerHTML = state.registry.providers.map(p => `<option value="${p}">${p}</option>`).join("");
    providerSelect.value = m.provider;

    row.querySelector(".model-name").addEventListener("input", (e) => { m.name = e.target.value; });
    row.querySelector(".model-id").addEventListener("input", (e) => { m.model_id = e.target.value; });
    providerSelect.addEventListener("change", (e) => { m.provider = e.target.value; });
    row.querySelector(".model-remove").addEventListener("click", () => {
      const removedName = m.name;
      state.config.models.splice(i, 1);
      state.config.judges = state.config.judges.filter(j => j !== removedName);
      renderModelList(); renderJudgeList(); refreshColorMap();
    });

    list.appendChild(tpl);
  });

  if (state.groqModels) {
    // Datalist for quick-pick of live, current Groq model ids -- exactly
    // the check that caught the stale qwen id during development.
    let dl = document.getElementById("groq-models-datalist");
    if (!dl) {
      dl = document.createElement("datalist");
      dl.id = "groq-models-datalist";
      document.body.appendChild(dl);
    }
    dl.innerHTML = state.groqModels.map(id => `<option value="${id}">`).join("");
    list.querySelectorAll(".model-id").forEach(inp => inp.setAttribute("list", "groq-models-datalist"));
  }
}

document.getElementById("btn-add-model").addEventListener("click", () => {
  state.config.models.push({ name: `model-${state.config.models.length + 1}`, provider: "groq", model_id: "" });
  renderModelList();
});

function renderTaskList() {
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  state.registry.tasks.forEach((taskName) => {
    const meta = TASK_META[taskName] || { name: taskName, desc: "" };
    const row = document.createElement("label");
    row.className = "task-row";
    row.innerHTML = `
      <div>
        <div class="task-row-name">${meta.name}</div>
        <div class="task-row-desc">${meta.desc}</div>
      </div>
      <input type="checkbox" ${state.config.tasks.includes(taskName) ? "checked" : ""}>
    `;
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked && !state.config.tasks.includes(taskName)) state.config.tasks.push(taskName);
      if (!e.target.checked) state.config.tasks = state.config.tasks.filter(t => t !== taskName);
    });
    list.appendChild(row);
  });
}

function renderJudgeList() {
  const list = document.getElementById("judge-list");
  list.innerHTML = "";
  state.config.models.forEach((m, i) => {
    const active = state.config.judges.includes(m.name);
    const chip = document.createElement("label");
    chip.className = `model-chip ${active ? "judge-active" : "judge-inactive"}`;
    chip.style.cursor = "pointer";
    chip.innerHTML = `
      <span class="swatch" style="background:${seriesColorHex(i)}"></span>
      <input type="checkbox" style="display:none" ${active ? "checked" : ""}>
      ${m.name}
      <span class="judge-state-label">${active ? "✓ judging" : "not judging"}</span>
    `;
    chip.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) { if (!state.config.judges.includes(m.name)) state.config.judges.push(m.name); }
      else { state.config.judges = state.config.judges.filter(j => j !== m.name); }
      chip.className = `model-chip ${e.target.checked ? "judge-active" : "judge-inactive"}`;
      chip.querySelector(".judge-state-label").textContent = e.target.checked ? "✓ judging" : "not judging";
    });
    list.appendChild(chip);
  });
}

document.getElementById("btn-save-config").addEventListener("click", async () => {
  state.config.run = {
    temperature: parseFloat(document.getElementById("cfg-temperature").value) || 0.0,
    max_tokens: parseInt(document.getElementById("cfg-max-tokens").value) || 800,
  };
  try {
    await apiPut("/config", state.config);
    toast("Configuration saved to config.yaml", "success");
    refreshColorMap();
  } catch (e) {
    toast(`Save failed: ${e.message}`, "error");
  }
});

// ============================================================
// RUN
// ============================================================
let pollTimer = null;
let runStartedAt = null;
let seenEvents = 0;

document.getElementById("btn-start-run").addEventListener("click", async () => {
  try {
    await apiPost("/run");
    document.getElementById("log-feed").innerHTML = "";
    seenEvents = 0;
    runStartedAt = Date.now();
    setRunStatus("running");
    toast("Run started", "success");
    startPolling();
  } catch (e) {
    toast(`Could not start run: ${e.message}`, "error");
  }
});

function setRunStatus(status) {
  const dot = document.getElementById("run-dot");
  const text = document.getElementById("run-status-text");
  dot.className = `dot ${status}`;
  text.textContent = status;
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollProgress, 600);
  pollProgress();
}

async function pollProgress() {
  let snap;
  try {
    snap = await apiGet(`/run/progress?since=${seenEvents}`);
  } catch (e) { return; }

  const total = snap.total || 0;
  const pct = total ? Math.round((snap.completed / total) * 100) : 0;
  document.getElementById("run-progress-count").textContent = `${snap.completed} / ${total}`;
  document.getElementById("run-progress-pct").textContent = `${pct}%`;
  gsap.to("#progress-fill", { width: `${pct}%`, duration: 0.4, ease: "power2.out" });

  if (runStartedAt) {
    const secs = Math.floor((Date.now() - runStartedAt) / 1000);
    document.getElementById("run-elapsed").textContent =
      `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
  }

  const feed = document.getElementById("log-feed");
  for (const ev of snap.events) {
    const row = document.createElement("div");
    row.className = `log-row ${ev.status}`;
    const icon = ev.status === "ok" ? "✓" : "✕";
    row.innerHTML = `
      <span class="status-icon">${icon}</span>
      <span class="lg-task">${ev.task}</span>
      <span class="lg-item">${ev.item_id}</span>
      <span class="lg-model">${ev.model}</span>
    `;
    feed.appendChild(row);
    gsap.fromTo(row, { opacity: 0, x: -8 }, { opacity: 1, x: 0, duration: 0.25, ease: "power2.out" });
    feed.scrollTop = feed.scrollHeight;
  }
  seenEvents = snap.completed;

  if (!snap.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    if (snap.error) {
      setRunStatus("error");
      toast(`Run failed: ${snap.error}`, "error");
    } else if (snap.total > 0) {
      setRunStatus("done");
      toast("Run complete", "success");
    }
  }
}

// Resume polling on page load if a run is already in progress (e.g. user
// switched tabs mid-run).
(async function checkRunOnLoad() {
  try {
    const snap = await apiGet("/run/progress?since=0");
    if (snap.running) {
      runStartedAt = Date.now() - ((snap.started_at ? (Date.now() / 1000 - snap.started_at) : 0) * 1000);
      setRunStatus("running");
      startPolling();
    }
  } catch (e) { /* backend not ready yet */ }
})();

// ============================================================
// CALIBRATE
// ============================================================
let calibQueue = [];
let calibRated = 0;

document.getElementById("btn-fetch-calibration").addEventListener("click", async () => {
  try {
    calibQueue = await apiGet("/calibration/sample?n=8");
    calibRated = 0;
    updateCalibPill();
    if (!calibQueue.length) {
      document.getElementById("calib-container").innerHTML =
        `<div class="empty-state"><p>No judge-scored records yet. Run a benchmark first.</p></div>`;
      return;
    }
    renderNextCalibCard();
  } catch (e) {
    toast(`Could not load samples: ${e.message}`, "error");
  }
});

function updateCalibPill() {
  document.getElementById("calib-progress-pill").textContent = `${calibRated} rated / ${calibQueue.length + calibRated} loaded`;
}

function renderNextCalibCard() {
  const container = document.getElementById("calib-container");
  if (!calibQueue.length) {
    container.innerHTML = `<div class="empty-state"><p>All loaded samples rated. Click &ldquo;Load samples&rdquo; for more.</p></div>`;
    return;
  }
  const cand = calibQueue[0];
  container.innerHTML = `
    <div class="calib-card">
      <div class="flex-between">
        <div class="flex flex-gap-2">
          <span class="pill">${cand.task}</span>
          <span class="pill">${cand.item_id}</span>
          <span class="pill">model: ${cand.model}</span>
        </div>
        <span class="muted text-sm mono">${calibQueue.length} remaining</span>
      </div>
      <div class="calib-context mt-4">${escapeHtml(cand.context)}</div>
      <div class="calib-ai-note">
        <span class="muted">AI judge (${cand.judge_name}):</span>
        <strong>${cand.ai_score}/10</strong>
        <span class="muted">&ldquo;${escapeHtml(cand.ai_reason)}&rdquo;</span>
      </div>
      <div class="score-slider-row">
        <span class="muted text-sm">Your score</span>
        <input type="range" class="score-slider" min="0" max="10" step="0.5" value="5" id="calib-slider">
        <span class="score-readout" id="calib-readout">5.0</span>
      </div>
      <div class="flex flex-gap-3 mt-5">
        <button class="btn btn-primary" id="calib-submit">Submit rating</button>
        <button class="btn btn-ghost" id="calib-skip">Skip</button>
      </div>
    </div>
  `;
  const slider = document.getElementById("calib-slider");
  const readout = document.getElementById("calib-readout");
  slider.addEventListener("input", () => { readout.textContent = parseFloat(slider.value).toFixed(1); });

  document.getElementById("calib-skip").addEventListener("click", () => {
    calibQueue.shift();
    renderNextCalibCard();
  });
  document.getElementById("calib-submit").addEventListener("click", async () => {
    try {
      await apiPost("/calibration/submit", {
        model: cand.model, task: cand.task, item_id: cand.item_id,
        judge_name: cand.judge_name, ai_score: cand.ai_score, ai_reason: cand.ai_reason,
        human_score: parseFloat(slider.value),
      });
      calibQueue.shift();
      calibRated++;
      updateCalibPill();
      renderNextCalibCard();
    } catch (e) {
      toast(`Submit failed: ${e.message}`, "error");
    }
  });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s ?? "";
  return div.innerHTML;
}

// ============================================================
// RESULTS
// ============================================================
let activeCharts = [];

async function loadResults() {
  const container = document.getElementById("results-container");
  let data;
  try {
    data = await apiGet("/results/summary");
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>No results yet. Run a benchmark to populate this view.</p></div>`;
    return;
  }

  activeCharts.forEach(c => c.dispose());
  activeCharts = [];

  if (!state.config) state.config = await apiGet("/config");
  refreshColorMap();

  const corrNote = Object.entries(data.corrections).map(([name, c]) => `
    <div class="flex flex-gap-2" style="align-items:baseline; margin-bottom:6px">
      <span class="pill ${c.method === "uncalibrated" ? "" : "pill-good"}">${name}</span>
      <span class="text-sm muted">${c.detail}</span>
    </div>`).join("");

  let html = `
    <div class="panel mt-5">
      <div class="panel-title">Judge bias calibration</div>
      <div class="mt-4">${corrNote}</div>
    </div>
  `;

  // Illustrative paid-tier cost: this run is entirely free-tier ($0 for
  // every model), which removes cost as a Pareto-discriminating axis. Groq's
  // published per-output-token rates (accessed while preparing the
  // accompanying paper) applied to each model's measured token count show
  // what WOULD discriminate on a paid tier -- not a real charge.
  const OUTPUT_PRICE_PER_1M = { "qwen-3.6-27b": 3.00, "gpt-oss-120b": 0.60, "gpt-oss-20b": 0.30 };
  function illustrativeCost(model, tokens) {
    const rate = OUTPUT_PRICE_PER_1M[model];
    if (rate == null || tokens == null) return null;
    return (tokens / 1e6) * rate;
  }

  data.tasks.forEach((task) => {
    const rows = data.summary.filter(r => r.task === task);
    const friedman = data.friedman.find(f => f.task === task);

    html += `
      <div class="panel mt-5" data-task-panel="${task}">
        <div class="panel-header">
          <div>
            <div class="panel-title">Task: ${task}</div>
            <div class="panel-title-sub">${rows.length} model(s) &middot; ${friedman ? `Friedman χ²=${friedman.statistic?.toFixed(3) ?? "—"}, p=${friedman.p_value?.toFixed(4) ?? "—"}${friedman.note ? " (" + friedman.note + ")" : friedman.low_power_warning ? " — low statistical power" : ""}` : "not enough data for a Friedman test yet"}</div>
          </div>
        </div>

        <div class="chart-3d-shell" id="chart-${task}"></div>
        <div class="chart-legend" id="legend-${task}"></div>

        <div style="overflow-x:auto; margin-top: var(--sp-5)">
          <table class="data-table">
            <thead><tr>
              <th>Model</th><th class="num">Judge score</th><th class="num">Composite</th>
              <th class="num">Cost/query</th><th class="num" title="Groq's published per-output-token rate applied to measured tokens -- not a real charge, this run used free-tier access throughout.">Illustrative paid cost&nbsp;†</th>
              <th class="num">Latency</th><th>Pareto-optimal</th>
            </tr></thead>
            <tbody>
              ${rows.map(r => `
                <tr>
                  <td><span class="model-chip"><span class="swatch" style="background:${seriesColorHex(state.colorOf.get(r.model) ?? 0)}"></span>${r.model}</span></td>
                  <td class="num">${fmt(r.avg_judge_score)}</td>
                  <td class="num">${fmt(r.composite_quick_glance)}</td>
                  <td class="num">$${fmt(r.cost_usd, 4)}</td>
                  <td class="num">$${fmt(illustrativeCost(r.model, r.tokens), 6)}</td>
                  <td class="num">${fmt(r.latency_s, 2)}s</td>
                  <td>${r.pareto_optimal ? `<span class="pill pill-good">✓ pareto</span>` : `<span class="pill">dominated</span>`}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
          <p class="text-sm muted mt-3">&dagger; Illustrative only, using Groq's published per-output-token rate applied to measured tokens for this run &mdash; every model here ran on a $0 free tier, so real cost cannot discriminate between them; this column shows what would.</p>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;

  // mount 3D charts after DOM insert
  data.tasks.forEach((task) => {
    const rows = data.summary.filter(r => r.task === task);
    const el = document.getElementById(`chart-${task}`);
    if (!el) return;
    const chart = createBarChart3D(el);
    const items = rows.map(r => ({
      label: r.model,
      value: Math.round((r.avg_judge_score ?? 0) * 100) / 100,
      colorIndex: state.colorOf.get(r.model) ?? 0,
    }));
    chart.update(items, { max: 10, valueSuffix: "/10" });
    activeCharts.push(chart);

    const legend = document.getElementById(`legend-${task}`);
    legend.innerHTML = rows.map(r => `
      <span class="legend-item"><span class="legend-swatch" style="background:${seriesColorHex(state.colorOf.get(r.model) ?? 0)}"></span>${r.model}</span>
    `).join("");
  });

  gsap.fromTo("[data-task-panel]", { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.45, stagger: 0.08, ease: "power2.out" });
}

function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(d);
}

document.getElementById("btn-refresh-results").addEventListener("click", loadResults);

// ============================================================
// boot
// ============================================================
gsap.fromTo("[data-animate]", { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.5, stagger: 0.06, ease: "power2.out", delay: 0.1 });
loadOverview();
