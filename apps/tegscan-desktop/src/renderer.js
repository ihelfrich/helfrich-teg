const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const tiersEl = document.getElementById("tiers");
const checksEl = document.getElementById("checks");
const warningsEl = document.getElementById("warnings");
const generatedEl = document.getElementById("generated");
const refreshEl = document.getElementById("refresh");

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function metricCard(metric) {
  return `
    <article class="metric-card">
      <p>${metric.label}</p>
      <strong>${metric.display}</strong>
      <span>${metric.description}</span>
    </article>
  `;
}

function tierRow(tier) {
  const urban = Math.max(0, Math.min(1, tier.urban_share_of_tier));
  const rural = Math.max(0, Math.min(1, tier.rural_share_of_tier));
  const other = Math.max(0, 1 - urban - rural);
  return `
    <article class="tier-row">
      <div class="tier-title">
        <h3>${tier.label}</h3>
        <span>${tier.all_population_display}</span>
      </div>
      <div class="stack" aria-label="${tier.label} settlement split">
        <div class="urban" style="width: ${urban * 100}%"></div>
        <div class="rural" style="width: ${rural * 100}%"></div>
        <div class="other" style="width: ${other * 100}%"></div>
      </div>
      <div class="tier-stats">
        <span><b>Urban</b> ${tier.urban_population_display} &middot; ${pct(tier.urban_share_of_tier)} of tier &middot; ${pct(tier.urban_exposure_rate)} of urban pop.</span>
        <span><b>Rural</b> ${tier.rural_population_display} &middot; ${pct(tier.rural_share_of_tier)} of tier &middot; ${pct(tier.rural_exposure_rate)} of rural pop.</span>
      </div>
    </article>
  `;
}

function checkItem(check) {
  const cls = check.ok ? "ok" : "fail";
  const label = check.ok ? "OK" : "FAIL";
  return `
    <article class="check-item ${cls}">
      <span>${label}</span>
      <div>
        <strong>${check.name.replaceAll("_", " ")}</strong>
        <p>${check.details || ""}</p>
      </div>
    </article>
  `;
}

function inlineCheckItems(checks) {
  return Object.entries(checks).map(([name, ok]) => ({
    name,
    ok,
    details: ok ? "reconciled" : "failed"
  }));
}

async function load() {
  statusEl.textContent = "Loading";
  try {
    const [dashboard, doctor] = await Promise.all([
      window.tegscan.dashboard(),
      window.tegscan.doctor()
    ]);
    statusEl.textContent = doctor.status === "passed" ? "Validated" : "Needs attention";
    statusEl.dataset.state = doctor.status;
    generatedEl.textContent = dashboard.generated_at;
    metricsEl.innerHTML = dashboard.metrics.map(metricCard).join("");
    tiersEl.innerHTML = dashboard.tiers.map(tierRow).join("");

    const checks = [
      ...doctor.checks,
      ...inlineCheckItems(dashboard.checks)
    ];
    checksEl.innerHTML = checks.map(checkItem).join("");
    warningsEl.innerHTML = dashboard.warnings.map((warning) => `<p>${warning}</p>`).join("");
  } catch (error) {
    statusEl.textContent = "Error";
    statusEl.dataset.state = "failed";
    metricsEl.innerHTML = "";
    tiersEl.innerHTML = `<article class="error-box">${error.message}</article>`;
    checksEl.innerHTML = "";
    warningsEl.innerHTML = "";
  }
}

refreshEl.addEventListener("click", load);
load();
