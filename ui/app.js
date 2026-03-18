const MODE_CONFIG = {
  chat: {
    label: "chat",
    hint: "General conversation with local memory and telemetry.",
    placeholder: "Ask the node anything that does not require a special workflow.",
    quickPrompts: [
      "Summarize what this node can do right now.",
      "Explain the current architecture in plain English.",
      "What should I improve next in this project?",
    ],
    transform: (text) => text,
  },
  investigation: {
    label: "investigation",
    hint: "Evidence-first investigations with open questions and reusable prior evidence.",
    placeholder: "Investigate a bug, architecture choice, or system behavior.",
    quickPrompts: [
      "Investigate architecture drift between runtime and docs.",
      "Investigate why the learning benchmark regresses sometimes.",
      "Investigate the weakest part of the current assistant loop.",
    ],
    transform: (text) => `Investigate this carefully and separate findings from assumptions:\n${text}`,
  },
  code: {
    label: "code_task",
    hint: "Code-focused workflow with implementation, verification, and risk sections.",
    placeholder: "Describe the code task, repo area, or bug to fix.",
    quickPrompts: [
      "Implement a small UI improvement and verify it.",
      "Refactor the noisiest module without changing behavior.",
      "Add tests around the most brittle reasoning path.",
    ],
    transform: (text) => `Code task: inspect the relevant code path, implement the smallest coherent change, and verify it.\n${text}`,
  },
  text: {
    label: "text_task",
    hint: "Writing mode for rewrite, edit, summarization, and tone control.",
    placeholder: "Paste text to rewrite or describe the writing task.",
    quickPrompts: [
      "Rewrite a technical explanation to be shorter and clearer.",
      "Turn rough notes into a concise project update.",
      "Edit a paragraph to sound more direct and precise.",
    ],
    transform: (text) => `Text task: preserve meaning, improve clarity, and respect tone constraints.\n${text}`,
  },
};

const SECTION_ORDER = [
  "Goal",
  "Plan",
  "Findings",
  "Evidence",
  "Open Questions",
  "Relevant Evidence",
  "Deliverables",
  "Implementation",
  "Verification",
  "Risks",
  "Reusable Workflow",
  "Rewritten Text",
  "Quality Check",
];

const STORAGE_KEY = "neurofed_chat_v2";

const messages = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const askOnceBtn = document.getElementById("ask-once");
const clearHistoryBtn = document.getElementById("clear-history");
const showThoughtOps = document.getElementById("show-thoughtops");
const stepsEl = document.getElementById("step-list");
const statusPill = document.getElementById("status-pill");
const lastUpdatedEl = document.getElementById("last-updated");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const sectionCards = document.getElementById("section-cards");
const quickPromptsEl = document.getElementById("quick-prompts");
const activeModeBadge = document.getElementById("active-mode-badge");
const modeHint = document.getElementById("mode-hint");
const modeButtons = Array.from(document.querySelectorAll(".mode-chip"));

const mRequests = document.getElementById("m-requests");
const mLatency = document.getElementById("m-latency");
const mCache = document.getElementById("m-cache");
const mPc = document.getElementById("m-pc");
const mDb = document.getElementById("m-db");
const mMem = document.getElementById("m-mem");
const mCpu = document.getElementById("m-cpu");
const mSaved = document.getElementById("m-saved");
const mSource = document.getElementById("m-source");
const mIntent = document.getElementById("m-intent");
const mInvestigationHits = document.getElementById("m-investigation-hits");
const mWorkflowHits = document.getElementById("m-workflow-hits");

let processingNode = null;
let chatHistory = [];
let currentMode = "chat";
let lastStructuredContent = "";

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) {
      chatHistory = parsed.filter(
        (item) =>
          item &&
          typeof item.role === "string" &&
          typeof item.content === "string" &&
          typeof item.mode === "string"
      );
    }
  } catch (_) {
    chatHistory = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
  } catch (_) {
  }
}

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

function formatRelativeTime(timestampSeconds) {
  if (!timestampSeconds) return "Waiting for node state";
  const diff = Math.max(0, Math.round(Date.now() / 1000 - timestampSeconds));
  if (diff < 5) return "Updated just now";
  if (diff < 60) return `Updated ${diff}s ago`;
  if (diff < 3600) return `Updated ${Math.round(diff / 60)}m ago`;
  return `Updated ${Math.round(diff / 3600)}h ago`;
}

function parseStructuredSections(text) {
  const normalized = (text || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  const positions = [];
  for (const heading of SECTION_ORDER) {
    const marker = `${heading}:`;
    const index = normalized.indexOf(marker);
    if (index >= 0) {
      positions.push({ heading, index });
    }
  }
  positions.sort((a, b) => a.index - b.index);
  if (!positions.length) {
    return [];
  }
  return positions.map((item, idx) => {
    const start = item.index + item.heading.length + 1;
    const end = idx + 1 < positions.length ? positions[idx + 1].index : normalized.length;
    return {
      heading: item.heading,
      body: normalized.slice(start, end).trim(),
    };
  }).filter((section) => section.body);
}

function renderStructuredSections(text) {
  const sections = parseStructuredSections(text);
  lastStructuredContent = text || "";
  sectionCards.innerHTML = "";
  if (!sections.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = text ? "Latest answer is not structured." : "No structured answer yet.";
    sectionCards.appendChild(empty);
    return;
  }
  sections.forEach((section) => {
    const card = document.createElement("article");
    card.className = "section-card";
    const heading = document.createElement("h3");
    heading.textContent = section.heading;
    const body = document.createElement("pre");
    body.textContent = section.body;
    card.appendChild(heading);
    card.appendChild(body);
    sectionCards.appendChild(card);
  });
}

function appendMessage(role, content, persist = true, mode = currentMode) {
  const div = document.createElement("article");
  div.className = `msg ${role}`;

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.innerHTML = `<span>${role === "user" ? "You" : "Assistant"}</span><span>${MODE_CONFIG[mode]?.label || mode}</span>`;

  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = content;

  div.appendChild(meta);
  div.appendChild(body);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;

  if (role === "assistant") {
    renderStructuredSections(content);
  }

  if (persist) {
    chatHistory.push({ role, content, mode });
    saveHistory();
  }

  return div;
}

function appendThoughtOps(container, thoughtOps) {
  if (!thoughtOps) return;
  const div = document.createElement("div");
  div.className = "thoughtops";
  div.textContent = thoughtOps;
  container.appendChild(div);
}

function appendProcessing() {
  const div = document.createElement("article");
  div.className = "msg assistant processing";
  div.innerHTML = `
    <div class="msg-meta"><span>Assistant</span><span>${MODE_CONFIG[currentMode].label}</span></div>
    <div class="msg-body processing-row"><span class="spinner"></span><span class="processing-text">Working…</span></div>
  `;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  processingNode = div;
  return div;
}

function buildPrompt(text) {
  return MODE_CONFIG[currentMode].transform(text.trim());
}

async function executeRequest({ persistHistory }) {
  const text = promptEl.value.trim();
  if (!text) return;

  appendMessage("user", text, persistHistory, currentMode);
  promptEl.value = "";
  const node = appendProcessing();

  const payload = {
    model: "neurofed",
    messages: [{ role: "user", content: buildPrompt(text) }],
  };

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    const reply = data?.choices?.[0]?.message?.content ?? "(no response)";
    const body = node.querySelector(".msg-body");
    if (body) {
      body.textContent = reply;
      body.classList.remove("processing-row");
    } else {
      node.textContent = reply;
    }
    node.classList.remove("processing");
    if (processingNode === node) processingNode = null;

    if (persistHistory) {
      chatHistory.push({ role: "assistant", content: reply, mode: currentMode });
      saveHistory();
    }
    renderStructuredSections(reply);

    if (showThoughtOps?.checked) {
      const stateRes = await fetch("/ui/state");
      const uiState = await stateRes.json();
      const thoughtLine = (uiState.steps || []).find((s) => s.startsWith("ThoughtOps:"));
      if (thoughtLine) {
        appendThoughtOps(node, thoughtLine.trim());
      }
    }
  } catch (err) {
    const body = node.querySelector(".msg-body");
    if (body) {
      body.textContent = `error: ${err}`;
      body.classList.remove("processing-row");
    } else {
      node.textContent = `error: ${err}`;
    }
    node.classList.remove("processing");
    if (processingNode === node) processingNode = null;
    if (persistHistory) {
      chatHistory.push({ role: "assistant", content: `error: ${err}`, mode: currentMode });
      saveHistory();
    }
  }
}

function renderQuickPrompts() {
  quickPromptsEl.innerHTML = "";
  MODE_CONFIG[currentMode].quickPrompts.forEach((prompt) => {
    const btn = document.createElement("button");
    btn.className = "quick-prompt";
    btn.textContent = prompt;
    btn.addEventListener("click", () => {
      promptEl.value = prompt;
      promptEl.focus();
    });
    quickPromptsEl.appendChild(btn);
  });
}

function setMode(mode) {
  currentMode = mode;
  modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  activeModeBadge.textContent = MODE_CONFIG[mode].label;
  modeHint.textContent = MODE_CONFIG[mode].hint;
  promptEl.placeholder = MODE_CONFIG[mode].placeholder;
  renderQuickPrompts();
}

async function refreshState() {
  try {
    const res = await fetch("/ui/state");
    const data = await res.json();
    statusPill.textContent = data.status || "idle";
    statusPill.dataset.status = data.status || "idle";
    stepsEl.innerHTML = "";
    (data.steps || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      if (step.includes("Remote LLM request")) {
        li.classList.add("step-remote");
      } else if (step.includes("Local LLM request")) {
        li.classList.add("step-local");
      } else if (step.includes("Deterministic reasoning") || step.includes("PC reasoning") || step.includes("PC learning")) {
        li.classList.add("step-pc");
      } else if (step.includes("ThoughtOps")) {
        li.classList.add("step-thought");
      }
      stepsEl.appendChild(li);
    });
    lastUpdatedEl.textContent = formatRelativeTime(data.last_updated);

    if (typeof data.saved_total_usd === "number") {
      mSaved.textContent = `$${data.saved_total_usd.toFixed(4)}`;
    }
    if (data.last_source) {
      mSource.textContent = data.last_source;
    }
    if (data.last_intent) {
      mIntent.textContent = data.last_intent;
    }
    mInvestigationHits.textContent = data.investigation_memory_hits ?? 0;
    mWorkflowHits.textContent = data.workflow_memory_hits ?? 0;
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

sendBtn.addEventListener("click", () => executeRequest({ persistHistory: true }));
askOnceBtn.addEventListener("click", () => executeRequest({ persistHistory: false }));
clearHistoryBtn.addEventListener("click", () => {
  chatHistory = [];
  saveHistory();
  messages.innerHTML = "";
  renderStructuredSections(lastStructuredContent);
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    executeRequest({ persistHistory: true });
  }
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

setInterval(refreshState, 1000);
setInterval(refreshMetrics, 2000);
setInterval(refreshStats, 2000);

loadHistory();
setMode(currentMode);
chatHistory.forEach((msg) => appendMessage(msg.role, msg.content, false, msg.mode));
if (chatHistory.length) {
  const latestAssistant = [...chatHistory].reverse().find((item) => item.role === "assistant");
  if (latestAssistant) {
    renderStructuredSections(latestAssistant.content);
  }
}
refreshState();
refreshMetrics();
refreshStats();
