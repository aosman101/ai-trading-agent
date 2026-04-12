const statusEl = document.getElementById("status");
const weightsEl = document.getElementById("weights");
const learningEl = document.getElementById("learning");
const tradesEl = document.getElementById("trades");
const predictionsEl = document.getElementById("predictions");
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
    return token;
  }
  return "";
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
  const thead = "<tr>" + headers.map((h) => `<th>${h}</th>`).join("") + "</tr>";
  const body = rows
    .map((row) => "<tr>" + headers.map((h) => `<td>${JSON.stringify(row[h] ?? "")}</td>`).join("") + "</tr>")
    .join("");
  return `<table><thead>${thead}</thead><tbody>${body}</tbody></table>`;
}

async function refreshDashboard() {
  try {
    const [status, dashboard, equity, weights] = await Promise.all([
      apiFetch("/api/status"),
      apiFetch("/api/dashboard"),
      apiFetch("/api/equity"),
      apiFetch("/api/model-weights"),
    ]);

    statusEl.innerHTML = `
      <p><span class="pill">Mode: ${status.trading_mode}</span></p>
      <p>Live Enabled: <strong>${status.live_enabled}</strong></p>
      <p>Kill Switch: <strong>${status.kill_switch}</strong></p>
      <p>Worker Healthy: <strong>${status.worker_healthy}</strong></p>
      <p>DSI Healthy: <strong>${status.dsi_status?.configured ? (status.dsi_status.available && !(status.dsi_status.missing_models || []).length && !Object.keys(status.dsi_status.errors || {}).length) : "not configured"}</strong></p>
      <p>Broker Mode: <strong>${status.broker_status?.mode || "n/a"}</strong></p>
      <p>Repository Backend: <strong>${status.repository_status?.backend || "n/a"}</strong></p>
      <p>Last Cycle: <strong>${status.last_cycle_at || "n/a"}</strong></p>
      <p>Current Strategy: <strong>${status.current_strategy || "n/a"}</strong></p>
      <p>Most Influential Model: <strong>${status.most_influential_model || "n/a"}</strong></p>
      <p>Market Regime: <strong>${status.market_regime || "n/a"}</strong></p>
      <p>Weight Scope: <strong>${status.weight_scope || "n/a"}</strong></p>
      <p>Account Equity: <strong>${status.account_equity}</strong></p>
      <p>Day PnL: <strong>${status.day_pnl}</strong></p>
      <p>Portfolio Heat: <strong>${status.current_portfolio_heat ?? "n/a"}</strong></p>
      <p>Open Positions: <code>${JSON.stringify(status.open_positions)}</code></p>
      <p>Last Error: <strong>${status.last_error || status.worker_error || "none"}</strong></p>
    `;

    weightsEl.textContent = JSON.stringify(weights.payload || {}, null, 2);
    learningEl.innerHTML = renderTable(dashboard.learning_events || []);
    tradesEl.innerHTML = renderTable(dashboard.recent_trades || []);
    predictionsEl.innerHTML = renderTable(dashboard.recent_predictions || []);

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
