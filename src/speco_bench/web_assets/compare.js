(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const workspaceTabs = [...document.querySelectorAll("[data-workspace-target]")];
  const workspaces = {
    benchmark: $("benchmarkWorkspace"),
    compare: $("compareWorkspace"),
  };
  const form = $("compareForm");
  const startButton = $("compareStartButton");
  const stopButton = $("compareStopButton");
  const runState = $("compareRunState");
  const formError = $("compareFormError");
  const systemPromptEnabled = $("compareSystemPromptEnabled");
  const systemPromptField = $("compareSystemPromptField");
  const systemPrompt = $("compareSystemPrompt");
  const storageKey = "speco-bench-compare-config-v1";
  const savedFieldIds = [
    "compareBaseUrlA",
    "compareModelA",
    "compareBaseUrlB",
    "compareModelB",
    "compareMaxTokens",
    "compareTemperature",
    "compareTopP",
    "compareIgnoreEos",
    "compareSystemPromptEnabled",
    "compareSystemPrompt",
    "comparePrompt",
    "compareExtraBody",
    "compareTimeout",
  ];
  const sideElements = Object.fromEntries(
    ["A", "B"].map((label) => [
      label.toLowerCase(),
      {
        name: $(`compareOutputName${label}`),
        status: $(`compareStatus${label}`),
        ttft: $(`compareTtft${label}`),
        tpot: $(`compareTpot${label}`),
        e2e: $(`compareE2e${label}`),
        decodeRate: $(`compareDecodeRate${label}`),
        meta: $(`compareMeta${label}`),
        text: $(`compareText${label}`),
        error: $(`compareError${label}`),
      },
    ]),
  );
  const statusLabels = {
    idle: "等待",
    connecting: "连接中",
    streaming: "生成中",
    completed: "完成",
    failed: "失败",
    stopped: "已停止",
  };
  let controller = null;
  let activeRunId = 0;

  function selectWorkspace(name) {
    const target = workspaces[name] ? name : "benchmark";
    Object.entries(workspaces).forEach(([key, workspace]) => {
      workspace.hidden = key !== target;
    });
    workspaceTabs.forEach((tab) => {
      const selected = tab.dataset.workspaceTarget === target;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    try {
      localStorage.setItem("speco-bench-workspace", target);
    } catch (_error) {
      // Storage may be disabled; workspace switching still works.
    }
  }

  function saveConfig() {
    const config = {};
    savedFieldIds.forEach((id) => {
      const field = $(id);
      config[id] = field.type === "checkbox" ? field.checked : field.value;
    });
    try {
      localStorage.setItem(storageKey, JSON.stringify(config));
    } catch (_error) {
      // Configuration persistence is a convenience, not a requirement.
    }
  }

  function restoreConfig() {
    try {
      const config = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!config || typeof config !== "object") return;
      savedFieldIds.forEach((id) => {
        const field = $(id);
        if (!(id in config)) return;
        if (field.type === "checkbox") field.checked = Boolean(config[id]);
        else field.value = String(config[id]);
      });
    } catch (_error) {
      // Ignore malformed or inaccessible local storage.
    }
  }

  function formatMilliseconds(value) {
    if (!Number.isFinite(value)) return "--";
    if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
    return `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
  }

  function formatRate(value) {
    return Number.isFinite(value) ? `${value.toFixed(2)} tok/s` : "--";
  }

  function syncSystemPrompt() {
    const enabled = systemPromptEnabled.checked;
    systemPromptField.hidden = !enabled;
    systemPrompt.disabled = !enabled;
  }

  function setSideStatus(side, status) {
    const element = sideElements[side].status;
    element.className = `compare-status ${status}`;
    element.textContent = statusLabels[status] || status;
  }

  function renderStats(side, stats) {
    if (!stats || !sideElements[side]) return;
    const elements = sideElements[side];
    elements.ttft.textContent = formatMilliseconds(stats.ttft_ms);
    elements.tpot.textContent = formatMilliseconds(stats.tpot_ms);
    elements.e2e.textContent = formatMilliseconds(stats.e2e_ms);
    elements.decodeRate.textContent = formatRate(stats.decode_tokens_per_second);
    const sourceLabels = {
      usage: "API Usage",
      stream_chunks: "流数据块估算",
      pending: "等待 Token 统计",
    };
    elements.meta.textContent = [
      `Token ${stats.output_tokens ?? "--"}`,
      `字符 ${stats.char_count ?? "--"}`,
      `数据块 ${stats.chunk_count ?? "--"}`,
      sourceLabels[stats.token_source] || "等待 Token 统计",
    ].join(" · ");
    setSideStatus(side, stats.status || "streaming");
  }

  function resetSide(side, modelName) {
    const elements = sideElements[side];
    elements.name.textContent = modelName || `模型 ${side.toUpperCase()}`;
    elements.ttft.textContent = "--";
    elements.tpot.textContent = "--";
    elements.e2e.textContent = "--";
    elements.decodeRate.textContent = "--";
    elements.meta.textContent = "Token -- · 字符 -- · 数据块 --";
    elements.text.textContent = "";
    elements.text.classList.remove("empty");
    elements.error.textContent = "";
    setSideStatus(side, "connecting");
  }

  function handleEvent(event) {
    const side = event.side;
    if (!sideElements[side]) return;
    if (event.type === "delta") {
      sideElements[side].text.textContent += event.text || "";
      sideElements[side].text.scrollTop = sideElements[side].text.scrollHeight;
    }
    if (event.type === "error") {
      sideElements[side].error.textContent = event.error || "请求失败";
    }
    renderStats(side, event.stats);
  }

  async function consumeNdjson(response, signal) {
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        message = payload.error || message;
      } catch (_error) {
        // Keep the HTTP status when the response is not JSON.
      }
      throw new Error(message);
    }
    if (!response.body) throw new Error("浏览器未提供流式响应体");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const consumeLine = (line) => {
      if (!line.trim()) return;
      try {
        handleEvent(JSON.parse(line));
      } catch (_error) {
        throw new Error("服务端返回了无效的流式数据");
      }
    };

    while (true) {
      if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      lines.forEach(consumeLine);
      if (done) break;
    }
    consumeLine(buffer);
  }

  function payloadFromForm() {
    let extraBody;
    try {
      extraBody = JSON.parse($("compareExtraBody").value || "{}");
    } catch (_error) {
      throw new Error("Extra Body 不是有效的 JSON");
    }
    if (!extraBody || Array.isArray(extraBody) || typeof extraBody !== "object") {
      throw new Error("Extra Body 必须是 JSON 对象");
    }
    return {
      models: {
        a: {
          base_url: $("compareBaseUrlA").value.trim(),
          model: $("compareModelA").value.trim(),
          api_key: $("compareApiKeyA").value.trim() || null,
        },
        b: {
          base_url: $("compareBaseUrlB").value.trim(),
          model: $("compareModelB").value.trim(),
          api_key: $("compareApiKeyB").value.trim() || null,
        },
      },
      prompt: $("comparePrompt").value.trim(),
      system_prompt: systemPromptEnabled.checked
        ? systemPrompt.value.trim() || null
        : null,
      max_tokens: Number($("compareMaxTokens").value),
      temperature: Number($("compareTemperature").value),
      top_p: Number($("compareTopP").value),
      ignore_eos: $("compareIgnoreEos").checked,
      extra_body: extraBody,
      request_timeout_seconds: Number($("compareTimeout").value),
    };
  }

  async function startComparison(event) {
    event.preventDefault();
    formError.textContent = "";
    if (!form.reportValidity()) return;

    let payload;
    try {
      payload = payloadFromForm();
    } catch (error) {
      formError.textContent = error.message;
      return;
    }

    if (controller) controller.abort();
    controller = new AbortController();
    const runId = ++activeRunId;
    resetSide("a", payload.models.a.model);
    resetSide("b", payload.models.b.model);
    runState.className = "compare-run-state running";
    runState.textContent = "对比运行中";
    startButton.disabled = true;
    stopButton.disabled = false;
    saveConfig();

    try {
      const response = await fetch("/api/compare/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      await consumeNdjson(response, controller.signal);
      if (runId !== activeRunId) return;
      const hasFailure = Object.values(sideElements).some((elements) =>
        elements.status.classList.contains("failed")
      );
      runState.className = `compare-run-state ${hasFailure ? "failed" : "completed"}`;
      runState.textContent = hasFailure ? "部分请求失败" : "对比完成";
    } catch (error) {
      if (runId !== activeRunId) return;
      if (error.name === "AbortError") {
        Object.keys(sideElements).forEach((side) => {
          if (["connecting", "streaming"].some((name) => sideElements[side].status.classList.contains(name))) {
            setSideStatus(side, "stopped");
          }
        });
        runState.className = "compare-run-state";
        runState.textContent = "已停止";
      } else {
        formError.textContent = error.message || String(error);
        runState.className = "compare-run-state";
        runState.textContent = "请求失败";
      }
    } finally {
      if (runId === activeRunId) {
        controller = null;
        startButton.disabled = false;
        stopButton.disabled = true;
      }
    }
  }

  workspaceTabs.forEach((tab) => {
    tab.addEventListener("click", () => selectWorkspace(tab.dataset.workspaceTarget));
  });
  savedFieldIds.forEach((id) => {
    $(id).addEventListener("input", saveConfig);
  });
  systemPromptEnabled.addEventListener("input", syncSystemPrompt);
  form.addEventListener("submit", startComparison);
  stopButton.addEventListener("click", () => controller?.abort());
  restoreConfig();
  syncSystemPrompt();
  let initialWorkspace = "benchmark";
  try {
    initialWorkspace = localStorage.getItem("speco-bench-workspace") || "benchmark";
  } catch (_error) {
    // Use the benchmark workspace when storage is unavailable.
  }
  selectWorkspace(initialWorkspace);
})();
