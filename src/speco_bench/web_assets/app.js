const form = document.querySelector("#benchmarkForm");
const startButton = document.querySelector("#startButton");
const cancelButton = document.querySelector("#cancelButton");
const formError = document.querySelector("#formError");
const serverState = document.querySelector("#serverState");
const datasetOptions = document.querySelector("#datasetOptions");
const datasetFields = document.querySelector("#datasetFields");
const randomFields = document.querySelector("#randomFields");
const rangeRatio = document.querySelector("#rangeRatio");
const rangeRatioValue = document.querySelector("#rangeRatioValue");
const progressSection = document.querySelector("#progressSection");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const statusBadge = document.querySelector("#statusBadge");
const runTitle = document.querySelector("#runTitle");
const historyBody = document.querySelector("#historyBody");
const refreshButton = document.querySelector("#refreshButton");
const latencyCharts = {
  ttft_ms: document.querySelector("#ttftChart"),
  tpot_ms: document.querySelector("#tpotChart"),
  e2e_ms: document.querySelector("#e2eChart"),
};

let activeJobId = null;
let pollTimer = null;
let selectedJob = null;

const statusLabels = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已停止",
};

function setServerState(label, className) {
  serverState.className = `server-state ${className}`;
  serverState.querySelector("span:last-child").textContent = label;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  let data;
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function splitValues(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function numberValue(data, key, fallback = null) {
  const raw = data.get(key);
  if (raw === null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${key} 不是有效数字`);
  return value;
}

function selectedMode() {
  return form.elements.dataset_name.value;
}

function syncMode() {
  const random = selectedMode() === "random";
  datasetFields.hidden = random;
  randomFields.hidden = !random;
  form.elements.max_tokens.disabled = random;
  form.elements.tokenizer.disabled = !random;
  form.elements.random_input_len.disabled = !random;
  form.elements.random_output_len.disabled = !random;
  form.elements.random_range_ratio.disabled = !random;
  form.elements.trust_remote_code.disabled = !random;
}

function buildPayload() {
  const data = new FormData(form);
  const concurrencies = splitValues(String(data.get("concurrencies") || ""));
  const numPrompts = splitValues(String(data.get("num_prompts") || ""));
  if (!concurrencies.length) throw new Error("至少填写一个并发数");
  if (numPrompts.length && numPrompts.length !== concurrencies.length) {
    throw new Error("请求数必须与并发数一一对应");
  }

  const extraBodyText = String(data.get("extra_body") || "{}");
  let extraBody;
  try {
    extraBody = JSON.parse(extraBodyText);
  } catch {
    throw new Error("Extra Body 不是有效 JSON");
  }
  if (!extraBody || Array.isArray(extraBody) || typeof extraBody !== "object") {
    throw new Error("Extra Body 必须是 JSON 对象");
  }

  const mode = String(data.get("dataset_name"));
  const datasets = [
    ...form.querySelectorAll('input[name="datasets"]:checked'),
  ].map((input) => input.value);
  const customDataset = String(data.get("custom_dataset") || "").trim();
  if (customDataset) datasets.push(customDataset);
  if (mode === "custom" && !datasets.length) {
    throw new Error("至少选择一个数据集或填写自定义路径");
  }

  return {
    base_url: String(data.get("base_url") || "").trim(),
    model: String(data.get("model") || "").trim(),
    endpoint_type: String(data.get("endpoint_type")),
    dataset_name: mode,
    datasets,
    concurrencies,
    num_prompts: numPrompts,
    max_tokens: numberValue(data, "max_tokens"),
    tokenizer: String(data.get("tokenizer") || "").trim() || null,
    random_input_len: numberValue(data, "random_input_len", 1024),
    random_output_len: numberValue(data, "random_output_len", 128),
    random_range_ratio: numberValue(data, "random_range_ratio", 0),
    trust_remote_code: data.get("trust_remote_code") === "on",
    ignore_eos: data.get("ignore_eos") === "on",
    temperature: numberValue(data, "temperature", 0),
    top_p: numberValue(data, "top_p", 1),
    warmup_requests: numberValue(data, "warmup_requests", 1),
    seed: numberValue(data, "seed", 0),
    request_timeout_seconds: numberValue(
      data,
      "request_timeout_seconds",
      3600,
    ),
    metrics_url: String(data.get("metrics_url") || "").trim() || null,
    api_key: String(data.get("api_key") || "") || null,
    extra_body: extraBody,
  };
}

function setRunning(running) {
  startButton.disabled = running;
  cancelButton.hidden = !running;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number)
    ? number.toLocaleString("zh-CN", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "--";
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "--";
  const value = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function updateStatus(job) {
  const label = statusLabels[job.status] || job.status;
  statusBadge.textContent = label;
  statusBadge.className = `status-badge ${job.status}`;
  runTitle.textContent = `#${job.id}`;
  const running = job.status === "queued" || job.status === "running";
  setRunning(running);
  activeJobId = running ? job.id : null;

  if (job.progress) {
    const progress = job.progress;
    progressSection.hidden = false;
    const percent = Math.min(100, Number(progress.overall_percent || 0));
    document.querySelector("#progressPercent").textContent =
      `${formatNumber(percent, 1)}%`;
    document.querySelector("#progressBar").style.width = `${percent}%`;
    const phase =
      progress.phase === "warmup"
        ? "预热"
        : progress.phase === "completed"
          ? "全部完成"
          : "正式压测";
    const context = progress.dataset
      ? `${progress.dataset} · C${progress.concurrency}`
      : "";
    document.querySelector("#progressLabel").textContent =
      `${phase}${context ? ` · ${context}` : ""}`;
    document.querySelector("#progressCount").textContent =
      `${progress.completed} / ${progress.total}`;
    document.querySelector("#progressRate").textContent =
      `${formatNumber(progress.request_throughput)} req/s`;
    document.querySelector("#progressEta").textContent =
      `ETA ${formatDuration(progress.eta_seconds)}`;
  }
}

function latestCompletedRun(job) {
  return [...job.runs].reverse().find((run) => run.report);
}

function drawLatencyChart(canvas, distribution, color) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(170, Math.round(rect.width));
  const height = 210;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  const keys = ["p50", "p90", "p99"];
  const values = keys.map((key) => Number(distribution[key] || 0));
  const rawMaximum = Math.max(...values, 0);
  const maxValue = rawMaximum > 0 ? rawMaximum * 1.15 : 1;
  const top = 25;
  const bottom = 31;
  const left = 31;
  const right = 5;
  const plotHeight = height - top - bottom;
  const plotWidth = width - left - right;

  context.font = "9px system-ui";
  context.fillStyle = "#69747a";
  context.strokeStyle = "#e2e6e8";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 2; tick += 1) {
    const y = top + (plotHeight * tick) / 2;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(width - right, y);
    context.stroke();
    const label = maxValue * (1 - tick / 2);
    context.fillText(formatNumber(label, label < 10 ? 1 : 0), 1, y + 3);
  }

  const slotWidth = plotWidth / keys.length;
  const barWidth = Math.min(28, slotWidth * 0.55);
  keys.forEach((key, index) => {
    const value = values[index];
    const barHeight = (value / maxValue) * plotHeight;
    const center = left + slotWidth * (index + 0.5);
    const x = center - barWidth / 2;
    const y = top + plotHeight - barHeight;
    context.globalAlpha = 0.55 + index * 0.2;
    context.fillStyle = color;
    context.fillRect(x, y, barWidth, barHeight);
    context.globalAlpha = 1;
    context.fillStyle = "#344047";
    context.textAlign = "center";
    context.fillText(key.toUpperCase(), center, height - 10);
  });
  context.textAlign = "start";
}

function drawLatencyCharts(summary) {
  const colors = {
    ttft_ms: "#28739f",
    tpot_ms: "#b75345",
    e2e_ms: "#147d64",
  };
  for (const [metric, canvas] of Object.entries(latencyCharts)) {
    drawLatencyChart(canvas, summary[metric] || {}, colors[metric]);
  }
}

function renderRuns(job) {
  const runList = document.querySelector("#runList");
  runList.replaceChildren();
  for (const run of job.runs) {
    const item = document.createElement("div");
    item.className = "run-item";
    const throughput = run.report
      ? `${formatNumber(run.report.request_throughput)} req/s`
      : run.status === "failed"
        ? "失败"
        : statusLabels[run.status] || run.status;
    item.innerHTML = `
      <span class="run-index">${String(run.index).padStart(2, "0")}</span>
      <span class="run-name">
        <strong>${escapeHtml(run.dataset)} · C${run.concurrency}</strong>
        <small>${run.num_prompts ?? "全部"} 个请求</small>
      </span>
      <span class="run-value ${run.status === "failed" ? "failed" : ""}">
        ${throughput}
      </span>
    `;
    runList.append(item);
  }
}

function renderMessages(job) {
  const warnings = job.runs.flatMap((run) => run.report?.warnings || []);
  const failures = job.runs
    .filter((run) => run.error)
    .map((run) => `${run.dataset} / C${run.concurrency}: ${run.error}`);
  if (job.error) failures.push(job.error);
  const warningBox = document.querySelector("#warningBox");
  const messages = [...new Set([...warnings, ...failures])];
  warningBox.hidden = !messages.length;
  warningBox.textContent = messages.join(" · ");
}

function renderEmptyMetrics() {
  for (const id of [
    "requestThroughput",
    "outputThroughput",
    "meanTtft",
    "meanTpot",
    "successfulRequests",
    "acceptanceRate",
  ]) {
    document.querySelector(`#${id}`).textContent = "--";
  }
  document.querySelector("#requestTotal").textContent = "total";
  drawLatencyCharts({
    ttft_ms: {},
    tpot_ms: {},
    e2e_ms: {},
  });
}

function renderResult(job) {
  const completedRun = latestCompletedRun(job);
  renderRuns(job);
  const csvDownload = document.querySelector("#csvDownload");
  csvDownload.hidden = !job.runs.length;
  csvDownload.href = `/api/jobs/${job.id}/files/matrix.csv`;

  if (!completedRun) {
    const terminal = ["completed", "failed", "cancelled"].includes(job.status);
    resultContent.hidden = !terminal;
    emptyState.hidden = terminal || job.status === "running";
    if (terminal) {
      renderEmptyMetrics();
      renderMessages(job);
    }
    return;
  }

  const summary = completedRun.report;
  emptyState.hidden = true;
  resultContent.hidden = false;
  document.querySelector("#requestThroughput").textContent =
    formatNumber(summary.request_throughput);
  document.querySelector("#outputThroughput").textContent =
    formatNumber(summary.output_throughput);
  document.querySelector("#meanTtft").textContent =
    formatNumber(summary.ttft_ms.mean);
  document.querySelector("#meanTpot").textContent =
    formatNumber(summary.tpot_ms.mean);
  document.querySelector("#successfulRequests").textContent =
    summary.successful_requests;
  document.querySelector("#requestTotal").textContent =
    `/ ${summary.total_requests} total`;
  const acceptance = completedRun.report.spec_decode?.acceptance_rate;
  document.querySelector("#acceptanceRate").textContent =
    acceptance === null || acceptance === undefined
      ? "--"
      : `${formatNumber(acceptance * 100, 1)}%`;
  drawLatencyCharts(summary);
  renderMessages(job);
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value);
  return node.innerHTML;
}

function renderHistory(jobs) {
  historyBody.replaceChildren();
  if (!jobs.length) {
    historyBody.innerHTML =
      '<tr><td colspan="5" class="table-empty">暂无运行记录</td></tr>';
    return;
  }
  for (const job of jobs) {
    const row = document.createElement("tr");
    const datasets = job.configuration.datasets.join(", ");
    row.innerHTML = `
      <td>#${escapeHtml(job.id)}</td>
      <td>${escapeHtml(datasets)}</td>
      <td>${job.configuration.total_runs}</td>
      <td>${statusLabels[job.status] || escapeHtml(job.status)}</td>
      <td>${new Date(job.created_at).toLocaleString("zh-CN")}</td>
    `;
    row.addEventListener("click", () => selectJob(job));
    historyBody.append(row);
  }
}

function selectJob(job) {
  selectedJob = job;
  updateStatus(job);
  renderResult(job);
}

async function refreshJobs() {
  try {
    const data = await api("/api/jobs");
    setServerState("控制台已连接", "connected");
    renderHistory(data.jobs);
    const running = data.jobs.find(
      (job) => job.status === "queued" || job.status === "running",
    );
    if (running) {
      activeJobId = running.id;
      selectJob(running);
      startPolling();
    } else if (selectedJob) {
      const updated = data.jobs.find((job) => job.id === selectedJob.id);
      if (updated) selectJob(updated);
    } else if (data.jobs.length) {
      selectJob(data.jobs[0]);
    }
  } catch (error) {
    setServerState("连接失败", "disconnected");
    formError.textContent = error.message;
  }
}

async function pollActiveJob() {
  if (!activeJobId) return;
  try {
    const job = await api(`/api/jobs/${activeJobId}`);
    selectJob(job);
    if (!["queued", "running"].includes(job.status)) {
      stopPolling();
      await refreshJobs();
    }
  } catch (error) {
    stopPolling();
    formError.textContent = error.message;
  }
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = window.setInterval(pollActiveJob, 700);
}

function stopPolling() {
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = null;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  try {
    const payload = buildPayload();
    setRunning(true);
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    activeJobId = job.id;
    selectJob(job);
    progressSection.hidden = false;
    emptyState.hidden = true;
    startPolling();
    await refreshJobs();
  } catch (error) {
    setRunning(false);
    formError.textContent = error.message;
  }
});

cancelButton.addEventListener("click", async () => {
  if (!activeJobId) return;
  cancelButton.disabled = true;
  try {
    const job = await api(`/api/jobs/${activeJobId}/cancel`, {method: "POST"});
    selectJob(job);
    stopPolling();
    await refreshJobs();
  } catch (error) {
    formError.textContent = error.message;
  } finally {
    cancelButton.disabled = false;
  }
});

form.querySelectorAll('input[name="dataset_name"]').forEach((input) => {
  input.addEventListener("change", syncMode);
});

rangeRatio.addEventListener("input", () => {
  rangeRatioValue.value = `${Math.round(Number(rangeRatio.value) * 100)}%`;
});

refreshButton.addEventListener("click", refreshJobs);
window.addEventListener("resize", () => {
  const completedRun = selectedJob && latestCompletedRun(selectedJob);
  if (completedRun) drawLatencyCharts(completedRun.report);
});

async function initialize() {
  syncMode();
  try {
    const configuration = await api("/api/configuration");
    datasetOptions.replaceChildren();
    for (const [index, dataset] of configuration.datasets.entries()) {
      const label = document.createElement("label");
      label.className = "dataset-choice";
      label.title = dataset.path;
      label.innerHTML = `
        <input
          type="checkbox"
          name="datasets"
          value="${escapeHtml(dataset.name)}"
          ${index === 0 ? "checked" : ""}
        >
        <span>${escapeHtml(dataset.name)}</span>
      `;
      datasetOptions.append(label);
    }
    if (!configuration.datasets.length) {
      datasetOptions.innerHTML =
        '<span class="muted">未发现内置数据集</span>';
    }
    setServerState("控制台已连接", "connected");
    await refreshJobs();
  } catch (error) {
    setServerState("连接失败", "disconnected");
    formError.textContent = error.message;
  }
}

initialize();
