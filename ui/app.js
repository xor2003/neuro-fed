const messages = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const stepsEl = document.getElementById("step-list");
const statusPill = document.getElementById("status-pill");
const lastUpdatedEl = document.getElementById("last-updated");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const tray = document.getElementById("tray");
const trayGrid = document.getElementById("tray-grid");
const trayToggle = document.getElementById("tray-toggle");

const mRequests = document.getElementById("m-requests");
const mLatency = document.getElementById("m-latency");
const mCache = document.getElementById("m-cache");
const mPc = document.getElementById("m-pc");
const mDb = document.getElementById("m-db");
const mMem = document.getElementById("m-mem");
const mCpu = document.getElementById("m-cpu");
const mSaved = document.getElementById("m-saved");
const mSource = document.getElementById("m-source");
let processingNode = null;

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let idx = 0;
  let val = bytes;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx++;
  }
  return `${val.toFixed(1)} ${units[idx]}`;
}

function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = content;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function appendProcessing() {
  const div = document.createElement("div");
  div.className = "msg assistant processing";
  div.innerHTML = `<span class="spinner"></span><span class="processing-text">Working…</span>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  processingNode = div;
  return div;
}

async function sendMessage() {
  const text = promptEl.value.trim();
  if (!text) return;
  appendMessage("user", text);
  promptEl.value = "";
  const node = appendProcessing();

  const payload = {
    model: "neurofed",
    messages: [{ role: "user", content: text }],
  };

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    const reply = data?.choices?.[0]?.message?.content ?? "(no response)";
    node.textContent = reply;
    node.classList.remove("processing");
    if (processingNode === node) processingNode = null;
  } catch (err) {
    node.textContent = `error: ${err}`;
    node.classList.remove("processing");
    if (processingNode === node) processingNode = null;
  }
}

sendBtn.addEventListener("click", sendMessage);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

trayToggle.addEventListener("click", () => {
  const hidden = trayGrid.classList.toggle("hidden");
  tray.classList.toggle("hidden", hidden);
  trayToggle.textContent = hidden ? "Show" : "Hide";
});

async function refreshState() {
  try {
    const res = await fetch("/ui/state");
    const data = await res.json();
    statusPill.textContent = data.status || "idle";
    statusPill.style.color = data.status === "error" ? "#ff5f8a" : "#b24cff";
    stepsEl.innerHTML = "";
    (data.steps || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      stepsEl.appendChild(li);
    });
    if (data.last_updated) {
      const ts = new Date(data.last_updated * 1000);
      lastUpdatedEl.textContent = ts.toLocaleTimeString();
    }
    if (typeof data.saved_total_usd === "number") {
      mSaved.textContent = `$${data.saved_total_usd.toFixed(4)}`;
    }
    if (data.last_source) {
      mSource.textContent = data.last_source;
    }
    if (typeof data.progress_percent === "number") {
      const pct = Math.max(0, Math.min(100, data.progress_percent));
      progressBar.style.width = `${pct}%`;
      const total = data.progress_total ?? 0;
      const current = data.progress_current ?? 0;
      progressText.textContent = total > 0 ? `${current}/${total} (${pct.toFixed(0)}%)` : `${pct.toFixed(0)}%`;
    }
    if (processingNode && processingNode.classList.contains("processing")) {
      const lastStep = (data.steps || []).slice(-1)[0];
      const label = lastStep || data.status || "Working…";
      const textEl = processingNode.querySelector(".processing-text");
      if (textEl) {
        textEl.textContent = `Working: ${label}`;
      }
    }
  } catch (_) {
  }
}

async function refreshMetrics() {
  try {
    const res = await fetch("/ui/metrics");
    const data = await res.json();
    mRequests.textContent = data.total_requests ?? 0;
    mLatency.textContent = data.total_processing_time_ms ?? 0;
    mCache.textContent = data.cache_hits ?? 0;
    mPc.textContent = data.pc_inference_calls ?? 0;
  } catch (_) {
  }
}

async function refreshStats() {
  try {
    const res = await fetch("/ui/stats");
    const data = await res.json();
    mDb.textContent = formatBytes(data.db_size_bytes);
    mMem.textContent = formatBytes(data.memory_bytes);
    mCpu.textContent = `${(data.cpu_usage ?? 0).toFixed(1)}%`;
  } catch (_) {
  }
}

setInterval(refreshState, 1000);
setInterval(refreshMetrics, 2000);
setInterval(refreshStats, 2000);
refreshState();
refreshMetrics();
refreshStats();
