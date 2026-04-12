const statusEl = document.getElementById("status");
const weightsEl = document.getElementById("weights");
const learningEl = document.getElementById("learning");
const tradesEl = document.getElementById("trades");
const predictionsEl = document.getElementById("predictions");
const journalEl = document.getElementById("journal");
const signalSymbolEl = document.getElementById("signalSymbol");
const signalDirectionEl = document.getElementById("signalDirection");
const signalScoreEl = document.getElementById("signalScore");
const signalConfidenceEl = document.getElementById("signalConfidence");
const signalSourceEl = document.getElementById("signalSource");
const signalReasoningEl = document.getElementById("signalReasoning");
const sendSignalButtonEl = document.getElementById("sendSignalButton");
const signalFormStatusEl = document.getElementById("signalFormStatus");
const ctx = document.getElementById("equityChart").getContext("2d");

let equityChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{ label: "Equity", data: [] }]
  },
  options: {
    responsive: true,
    scales: { y: { beginAtZero: false } }
  }
});

function getApiToken() {
  const query = new URLSearchParams(window.location.search);
  const hash = new URLSearchParams(window.location.hash.startsWith("#") ? window.location.hash.slice(1) : window.location.hash);
  const token = query.get("token") || hash.get("token") || window.localStorage.getItem("dashboardApiToken");
  if (token) {
    window.localStorage.setItem("dashboardApiToken", token);
    if (query.get("token") || hash.get("token")) {
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    return token;
  }
  return "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function apiFetch(path) {
  const headers = {};
  const token = getApiToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(path, { headers });
  if (response.status === 401) {
    throw new Error("Unauthorized. Add ?token=YOUR_API_BEARER_TOKEN to the dashboard URL or store dashboardApiToken in localStorage.");
  }
  if (!response.ok) {
    throw new Error(`Request failed for ${path}: ${response.status}`);
  }
  return response.json();
}

function renderTable(rows) {
  if (!rows || rows.length === 0) {
    return "<p>No data yet.</p>";
  }
  const headers = Object.keys(rows[0]).slice(0, 8);
  const thead = "<tr>" + headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("") + "</tr>";
  const body = rows
    .map((row) => "<tr>" + headers.map((h) => `<td>${escapeHtml(JSON.stringify(row[h] ?? ""))}</td>`).join("") + "</tr>")
    .join("");
  return `<table><thead>${thead}</thead><tbody>${body}</tbody></table>`;
}

function renderJournal(entries) {
  if (!entries || entries.length === 0) {
    return "<p>No journal entries yet.</p>";
  }
  return entries.map((entry) => `
    <div class="journal-entry">
      <p><strong>${escapeHtml(entry.headline || "Journal entry")}</strong></p>
      <p>${escapeHtml(entry.created_at || "")} · ${escapeHtml(entry.symbol || "n/a")} · ${escapeHtml(entry.event_type || "n/a")}</p>
      <pre>${escapeHtml(entry.body || "")}</pre>
    </div>
  `).join("");
}

async function sendSignal() {
  signalFormStatusEl.textContent = "Sending signal...";
  try {
    const token = getApiToken();
    const headers = { "Content-Type": "application/json" };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const response = await fetch("/api/signals", {
      method: "POST",
      headers,
      body: JSON.stringify({
        symbol: signalSymbolEl.value.trim().toUpperCase(),
        direction: signalDirectionEl.value,
        score: Number(signalScoreEl.value),
        confidence: Number(signalConfidenceEl.value),
        source: signalSourceEl.value.trim() || "dashboard",
        reasoning: signalReasoningEl.value.trim(),
      }),
    });
    if (response.status === 401) {
      throw new Error("Unauthorized. Add ?token=YOUR_API_BEARER_TOKEN to the dashboard URL.");
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed with status ${response.status}`);
    }
    const payload = await response.json();
    signalFormStatusEl.textContent = `Signal accepted for ${payload.symbol} (${payload.direction}).`;
    await refreshDashboard();
  } catch (error) {
    signalFormStatusEl.textContent = error.message;
  }
}

async function refreshDashboard() {
  try {
    const [status, dashboard, equity, weights, journal] = await Promise.all([
      apiFetch("/api/status"),
      apiFetch("/api/dashboard"),
      apiFetch("/api/equity"),
      apiFetch("/api/model-weights"),
      apiFetch("/api/journal?limit=20"),
    ]);

    statusEl.innerHTML = `
      <p><span class="pill">Mode: ${escapeHtml(status.trading_mode)}</span></p>
      <p>Live Enabled: <strong>${escapeHtml(status.live_enabled)}</strong></p>
      <p>Kill Switch: <strong>${escapeHtml(status.kill_switch)}</strong></p>
      <p>Worker Healthy: <strong>${escapeHtml(status.worker_healthy)}</strong></p>
      <p>DSI Healthy: <strong>${escapeHtml(status.dsi_status?.configured ? (status.dsi_status.available && !(status.dsi_status.missing_models || []).length && !Object.keys(status.dsi_status.errors || {}).length) : "not configured")}</strong></p>
      <p>Broker Mode: <strong>${escapeHtml(status.broker_status?.mode || "n/a")}</strong></p>
      <p>Repository Backend: <strong>${escapeHtml(status.repository_status?.backend || "n/a")}</strong></p>
      <p>Last Cycle: <strong>${escapeHtml(status.last_cycle_at || "n/a")}</strong></p>
      <p>Current Strategy: <strong>${escapeHtml(status.current_strategy || "n/a")}</strong></p>
      <p>Most Influential Model: <strong>${escapeHtml(status.most_influential_model || "n/a")}</strong></p>
      <p>Market Regime: <strong>${escapeHtml(status.market_regime || "n/a")}</strong></p>
      <p>Weight Scope: <strong>${escapeHtml(status.weight_scope || "n/a")}</strong></p>
      <p>Account Equity: <strong>${escapeHtml(status.account_equity)}</strong></p>
      <p>Day PnL: <strong>${escapeHtml(status.day_pnl)}</strong></p>
      <p>Portfolio Heat: <strong>${escapeHtml(status.current_portfolio_heat ?? "n/a")}</strong></p>
      <p>Open Positions: <code>${escapeHtml(JSON.stringify(status.open_positions))}</code></p>
      <p>Last Error: <strong>${escapeHtml(status.last_error || status.worker_error || "none")}</strong></p>
    `;

    weightsEl.textContent = JSON.stringify(weights.payload || {}, null, 2);
    learningEl.innerHTML = renderTable(dashboard.learning_events || []);
    tradesEl.innerHTML = renderTable(dashboard.recent_trades || []);
    predictionsEl.innerHTML = renderTable(dashboard.recent_predictions || []);
    journalEl.innerHTML = renderJournal(journal || []);

    const points = (equity || []).slice().reverse();
    equityChart.data.labels = points.map((item) => item.created_at);
    equityChart.data.datasets[0].data = points.map((item) => item.equity);
    equityChart.update();
  } catch (error) {
    statusEl.innerHTML = `<p><strong>Dashboard error:</strong> ${error.message}</p>`;
  }
}

refreshDashboard();
setInterval(refreshDashboard, 10000);
sendSignalButtonEl.addEventListener("click", sendSignal);
