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
  const [status, dashboard, equity, weights] = await Promise.all([
    fetch("/api/status").then((r) => r.json()),
    fetch("/api/dashboard").then((r) => r.json()),
    fetch("/api/equity").then((r) => r.json()),
    fetch("/api/model-weights").then((r) => r.json()),
  ]);

  statusEl.innerHTML = `
    <p><span class="pill">Mode: ${status.trading_mode}</span></p>
    <p>Live Enabled: <strong>${status.live_enabled}</strong></p>
    <p>Kill Switch: <strong>${status.kill_switch}</strong></p>
    <p>Current Strategy: <strong>${status.current_strategy || "n/a"}</strong></p>
    <p>Most Influential Model: <strong>${status.most_influential_model || "n/a"}</strong></p>
    <p>Account Equity: <strong>${status.account_equity}</strong></p>
    <p>Day PnL: <strong>${status.day_pnl}</strong></p>
    <p>Open Positions: <code>${JSON.stringify(status.open_positions)}</code></p>
  `;

  weightsEl.textContent = JSON.stringify(weights.payload || {}, null, 2);
  learningEl.innerHTML = renderTable(dashboard.learning_events || []);
  tradesEl.innerHTML = renderTable(dashboard.recent_trades || []);
  predictionsEl.innerHTML = renderTable(dashboard.recent_predictions || []);

  const points = (equity || []).slice().reverse();
  equityChart.data.labels = points.map((item) => item.created_at);
  equityChart.data.datasets[0].data = points.map((item) => item.equity);
  equityChart.update();
}

refreshDashboard();
setInterval(refreshDashboard, 10000);
