/* ============================================================
Frontend Logic - Adapted for serve.py (FastAPI Backend)
============================================================ */
const $ = (sel) => document.querySelector(sel);
const THINKING_AUTO_COLLAPSE_MS = 5000;
const STREAM_RENDER_INTERVAL_MS = 80;

const state = {
  sessionId: null,
  agents: [],
  selectedAgent: null,
  messages: [],
  events: [],
  workspaceFiles: [],
  pendingApprovals: [],
  pendingImages: [],
  runningTools: {},
  activityItems: [],
  currentAbortController: null,
  isRunning: false,
  isCancelling: false,
  showWorkspace: false,
  showEvents: false,
  workspaceModalOpen: false,
  thinkingPanels: {},
  thinkingAutoCloseTimers: {},
  contextWindowOpen: false,
  contextUsage: null,
  compactionRunning: false,
  currentPlan: null,
  pendingInputs: [],
};

let renderQueued = false;

function thinkingKeyForMessage(msg, index) {
  return msg.id || msg.created_at || `${msg.role}-${index}`;
}

function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  const run = () => {
    renderQueued = false;
    renderMessages();
    renderStatusArea();
  };
  const delay = state.isRunning ? STREAM_RENDER_INTERVAL_MS : 0;
  setTimeout(() => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(run);
    } else {
      run();
    }
  }, delay);
}

function findThinkingText(key) {
  const indexMatch = String(key || "").match(/^assistant-(\d+)$/);
  if (indexMatch) return state.messages[Number(indexMatch[1])]?.thinking || "";
  const message = state.messages.find((msg, index) => thinkingKeyForMessage(msg, index) === key);
  return message?.thinking || "";
}

function renderThinkingContent(details) {
  const key = details?.dataset?.thinkingKey;
  const content = details?.querySelector(".thinking-content");
  if (!key || !content) return;
  const text = findThinkingText(key);
  content.innerHTML = window.__markedReady ? marked.parse(text) : escapeHtml(text);
}

function scheduleThinkingAutoClose(key) {
  if (!key || state.thinkingAutoCloseTimers[key]) return;
  state.thinkingAutoCloseTimers[key] = setTimeout(() => {
    state.thinkingPanels[key] = false;
    delete state.thinkingAutoCloseTimers[key];
    const details = [...document.querySelectorAll(".thinking-block")].find((node) => node.dataset.thinkingKey === key);
    if (details) {
      details.open = false;
      const content = details.querySelector(".thinking-content");
      if (content) content.innerHTML = "";
    }
  }, THINKING_AUTO_COLLAPSE_MS);
}

function toggleThinkingPanel(details) {
  const key = details?.dataset?.thinkingKey;
  if (!details || !key) return;
  const nextOpen = !details.open;
  details.open = nextOpen;
  state.thinkingPanels[key] = nextOpen;
  const content = details.querySelector(".thinking-content");
  if (nextOpen) {
    renderThinkingContent(details);
  } else if (content) {
    content.innerHTML = "";
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function imageDataUrl(image) {
  if (image.preview_url) return image.preview_url;
  if (image.data_url) return image.data_url;
  if (image.data_base64) return `data:${image.mime_type || "image/png"};base64,${image.data_base64}`;
  return "";
}

function renderImageList(images) {
  if (!Array.isArray(images) || images.length === 0) return "";
  const items = images.map((image) => {
    const src = escapeHtml(imageDataUrl(image));
    const name = escapeHtml(image.name || "attached image");
    return `<img src="${src}" alt="${name}" loading="lazy" />`;
  }).join("");
  return `<div class="msg-images">${items}</div>`;
}

function runningToolsHtml() {
  const tools = Object.values(state.runningTools || {});
  if (tools.length === 0) return "";
  return tools.map((tool) => `
    <div class="tool-running-row">
      <div class="tool-running-pill">
        <span class="status-dot"></span>
        <span>Running tool: <strong>${escapeHtml(tool.tool_name || "tool")}</strong></span>
      </div>
    </div>
  `).join("");
}

function pushActivity(text, tone = "info") {
  const normalized = String(text || "").trim();
  if (!normalized) return;
  state.activityItems = [{
    id: Date.now() + Math.random().toString(36).slice(2),
    text: normalized,
    tone,
  }];
}

function clearActivity() {
  state.activityItems = [];
}

function activityHtml() {
  if (!state.isRunning || state.activityItems.length === 0) return "";
  const item = state.activityItems[state.activityItems.length - 1];
  return `
    <div class="msg-row bot-row activity-row">
      <div class="typing-bubble ${escapeHtml(item.tone)}">
        <span class="typing-text">${escapeHtml(item.text)}</span>
        <span class="typing-dots" aria-hidden="true">
          <span></span>
          <span></span>
          <span></span>
        </span>
      </div>
    </div>
  `;
}

// Split an option string into { label, description }. Supports the
// conventions "Label — description", "Label - description" and
// "Label | description"; a bare string becomes just a label.
function parseOption(raw) {
  const text = String(raw ?? "").trim();
  const match = text.match(/^(.*?)\s+(?:—|\||-{1,2})\s+(.+)$/s);
  if (match) return { label: match[1].trim(), description: match[2].trim() };
  return { label: text, description: "" };
}

function compactText(value, max = 180) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function fmtCount(value) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toLocaleString() : "0";
}

function fmtPercent(value) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? `${Math.round(n)}%` : "0%";
}

function usageChips(usage) {
  if (!usage) return "";
  return `
    <div class="usage-chips">
      <span>in <strong>${fmtCount(usage.tokens_input)}</strong></span>
      <span>out <strong>${fmtCount(usage.tokens_output)}</strong></span>
      <span>cached <strong>${fmtCount(usage.tokens_cached)}</strong></span>
      <span>calls <strong>${fmtCount(usage.llm_calls)}</strong></span>
    </div>
  `;
}

function selectedAgentConfig() {
  return state.agents.find((agent) => agent.name === state.selectedAgent) || null;
}

function updateAttachmentControls() {
  const btn = $("#attachImageBtn");
  if (!btn) return;
  const agent = selectedAgentConfig();
  const enabled = Boolean(agent?.supports_vision);
  btn.disabled = !enabled || state.isRunning || !state.sessionId;
  btn.title = enabled ? "Attach image" : "Selected model does not support image input";
}

function contextWindowPercent(usage = state.contextUsage) {
  const used = Number(usage?.used || 0);
  const max = Number(usage?.max || selectedAgentConfig()?.context_window || 0);
  if (!max) return 0;
  return Math.max(0, Math.min(100, (used / max) * 100));
}

function renderContextWindow() {
  const btn = $("#contextWindowBtn");
  const popover = $("#contextWindowPopover");
  if (!btn || !popover) return;

  btn.classList.toggle("active", state.contextWindowOpen);
  btn.classList.toggle("is-compacting", state.compactionRunning);
  const usage = state.contextUsage || {};
  const max = usage.max || selectedAgentConfig()?.context_window || 0;
  const percent = contextWindowPercent(usage);
  btn.title = state.compactionRunning
    ? "Compacting context"
    : `Context window ${fmtPercent(percent)}`;

  if (!state.contextWindowOpen) {
    popover.hidden = true;
    popover.innerHTML = "";
    return;
  }

  const live = Number(usage.live_message_tokens || 0);
  const prompt = Number(usage.prompt_tokens || 0);
  const reserved = Number(usage.reserved_output_tokens || 0);
  const safety = Number(usage.safety_margin_tokens || 0);
  const total = Number(usage.used || 0);
  const denom = Number(max || Math.max(total, 1));
  const segment = (value, cls, label) => {
    const width = Math.max(0, Math.min(100, (Number(value || 0) / denom) * 100));
    if (width <= 0) return "";
    return `<span class="cw-segment ${cls}" style="width:${width}%" title="${escapeHtml(label)}: ${fmtCount(value)} tokens"></span>`;
  };
  const layers = Array.isArray(usage.prompt_layers) ? usage.prompt_layers : [];
  const layerHtml = layers.length
    ? `<details class="cw-details"><summary>Prompt layers</summary>${layers.map((layer) => `
        <div class="cw-row"><span>${escapeHtml(layer.name || "Layer")}</span><strong>${fmtCount(layer.tokens)}</strong></div>
      `).join("")}</details>`
    : "";
  const summaryPreview = usage.summary
    ? `<details class="cw-details"><summary>Summary</summary><pre>${escapeHtml(JSON.stringify(usage.summary, null, 2))}</pre></details>`
    : "";

  popover.hidden = false;
  popover.innerHTML = `
    <div class="cw-head">
      <span>Context Window</span>
      <strong>${fmtPercent(percent)}</strong>
    </div>
    <div class="cw-counts"><span>${fmtCount(total)} / ${fmtCount(max || 0)} tokens</span></div>
    <div class="cw-bar" aria-hidden="true">
      ${segment(prompt, "prompt", "Prompt")}
      ${segment(live, "live", "Live messages")}
      ${segment(safety, "safety", "Safety margin")}
      ${segment(reserved, "reserved", "Reserved output")}
    </div>
    <div class="cw-legend">
      <span><i class="prompt"></i>Prompt ${fmtCount(prompt)}</span>
      <span><i class="live"></i>Live ${fmtCount(live)}</span>
      <span><i class="reserved"></i>Output ${fmtCount(reserved)}</span>
    </div>
    <div class="cw-row"><span>Messages</span><strong>${fmtCount(usage.message_count)}</strong></div>
    <div class="cw-row"><span>Threshold</span><strong>${fmtCount(usage.live_message_threshold_tokens)}</strong></div>
    <div class="cw-row"><span>Live budget</span><strong>${fmtCount(usage.live_message_budget_tokens)}</strong></div>
    ${state.compactionRunning ? `<div class="cw-status"><span class="status-dot"></span> Compacting context</div>` : ""}
    ${layerHtml}
    ${summaryPreview}
  `;
}

async function refreshContextWindow() {
  if (!state.sessionId || !state.selectedAgent) return;
  try {
    const params = new URLSearchParams({ session_id: state.sessionId, agent_name: state.selectedAgent });
    const usage = await fetch(`/api/chat/context?${params.toString()}`).then(r => r.json());
    state.contextUsage = usage;
  } catch (err) {
    logEvent("error", "Context Window Failed", { error: String(err) });
  }
  renderContextWindow();
}

function toggleContextWindow() {
  state.contextWindowOpen = !state.contextWindowOpen;
  renderContextWindow();
  if (state.contextWindowOpen) refreshContextWindow();
}

// --- Theme Management ---
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  renderThemeBtn();
}
function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  renderThemeBtn();
}
function renderThemeBtn() {
  const btn = $("#themeToggle");
  if (!btn) return;
  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  btn.innerHTML = isDark
    ? `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
    : `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
}

// --- Logfire KV Builder ---
function buildLogfireKV(obj) {
  if (obj === null) return `<span class="lf-v null">null</span>`;
  if (typeof obj === 'boolean') return `<span class="lf-v bool">${obj}</span>`;
  if (typeof obj === 'number') return `<span class="lf-v num">${obj}</span>`;
  if (typeof obj === 'string') {
    const escStr = escapeHtml(obj);
    if (obj.length > 180) {
      return `
        <details class="lf-dropdown lf-text-dropdown">
          <summary>${fmtCount(obj.length)} chars</summary>
          <pre>${escStr}</pre>
        </details>
      `;
    }
    return `<span class="lf-v str">"${escStr}"</span>`;
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) return `<span class="lf-v">[]</span>`;
    let html = `
      <details class="lf-dropdown">
        <summary>${fmtCount(obj.length)} item${obj.length === 1 ? "" : "s"}</summary>
        <div class="lf-kv-tree lf-nested">
    `;
    obj.forEach((item, index) => {
      html += `<div class="lf-kv-row"><span class="lf-k">${index}</span>${buildLogfireKV(item)}</div>`;
    });
    html += `</div></details>`;
    return html;
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return `<span class="lf-v">{}</span>`;
    let html = `<div class="lf-kv-tree lf-nested">`;
    keys.forEach(k => {
      html += `<div class="lf-kv-row"><span class="lf-k">${k}</span>${buildLogfireKV(obj[k])}</div>`;
    });
    html += `</div>`;
    return html;
  }
  return `<span class="lf-v">${obj}</span>`;
}

function compactMessage(msg) {
  return {
    role: msg?.role || "",
    source: msg?.source || "",
    tokens: msg?.token_count || 0,
    content: msg?.content || msg?.text || "",
    thinking: msg?.thinking || "",
    tool_calls: Array.isArray(msg?.tool_calls) ? msg.tool_calls.length : 0,
    created_at: msg?.created_at || "",
    name: msg?.name || "",
  };
}

function renderEventAttribute(key, value) {
  if (key === "input_messages" && Array.isArray(value)) {
    return renderInputMessages(value);
  }
  return buildLogfireKV(value);
}

function renderInputMessages(messages) {
  if (messages.length === 0) return `<span class="lf-v">[]</span>`;

  const rows = messages.map((msg, index) => {
    const role = escapeHtml(msg.role || "message");
    const source = msg.source ? ` · ${escapeHtml(msg.source)}` : "";
    const tokens = fmtCount(msg.tokens || 0);
    const content = escapeHtml(msg.content || "");
    const thinking = msg.thinking ? `
      <div class="lf-message-section">
        <span>thinking</span>
        <pre>${escapeHtml(msg.thinking)}</pre>
      </div>
    ` : "";
    const toolCallCount = Array.isArray(msg.tool_calls) ? msg.tool_calls.length : Number(msg.tool_calls || 0);
    const toolCalls = toolCallCount ? `<span class="lf-message-chip">${fmtCount(toolCallCount)} tool call${toolCallCount === 1 ? "" : "s"}</span>` : "";

    return `
      <details class="lf-message-card"${index === messages.length - 1 ? " open" : ""}>
        <summary>
          <span class="lf-message-index">${index}</span>
          <span class="lf-message-role">${role}${source}</span>
          <span class="lf-message-chip">${tokens} tokens</span>
          ${toolCalls}
        </summary>
        <div class="lf-message-body">
          <div class="lf-message-section">
            <span>content</span>
            <pre>${content || "(empty)"}</pre>
          </div>
          ${thinking}
        </div>
      </details>
    `;
  }).join("");

  return `<div class="lf-message-list">${rows}</div>`;
}

function summarizeEventForLog(agentName, ev) {
  const eventType = ev.event_type || ev.type || "event";
  if (eventType.startsWith("bash_")) {
    const detail = eventType === "bash_started"
      ? `${ev.description || ev.declared_action} (intención declarada)`
      : eventType === "bash_finished"
        ? (ev.timed_out ? "Tiempo agotado" : `Comando terminado · exit ${ev.exit_code}`)
        : (ev.error || ev.reason || "");
    return { type: eventType, title: `[${agentName}] ${eventType} · ${detail}`, attributes: ev };
  }
  if (eventType === "model_call") {
    const messages = Array.isArray(ev.input_messages) ? ev.input_messages : [];
    const inputTokens = messages.reduce((sum, msg) => sum + Number(msg.token_count || 0), 0);
    return {
      type: "model",
      title: `[${agentName}] model_call · ${ev.model || "unknown"} · ${messages.length} msg`,
      attributes: {
        event_type: eventType,
        model: ev.model,
        message_count: messages.length,
        input_tokens: inputTokens,
        last_message: compactMessage(messages[messages.length - 1] || {}),
        input_messages: messages.map(compactMessage),
      },
    };
  }
  if (eventType === "model_response") {
    const usage = ev.usage || {};
    return {
      type: "model",
      title: `[${agentName}] model_response · in ${fmtCount(usage.tokens_input)} / out ${fmtCount(usage.tokens_output)}`,
      attributes: {
        event_type: eventType,
        has_tool_calls: Boolean(ev.has_tool_calls),
        usage,
        response_preview: compactText(ev.text || ev.response || "", 260),
      },
    };
  }
  if (eventType === "tool_call") {
    return {
      type: "tool_call",
      title: `[${agentName}] tool_call · ${ev.tool_name || "unknown"}`,
      attributes: ev,
    };
  }
  if (eventType === "tool_result") {
    return {
      type: ev.success ? "tool_result" : "error",
      title: `[${agentName}] tool_result · ${ev.success ? "success" : "failed"}`,
      attributes: ev,
    };
  }
  if (eventType === "planning") {
    const steps = Array.isArray(ev.plan?.steps) ? ev.plan.steps : [];
    const done = steps.filter((s) => s.status === "done").length;
    return {
      type: "planning",
      title: `[${agentName}] planning · ${ev.phase}${steps.length ? ` · ${done}/${steps.length} done` : ""}`,
      attributes: {
        event_type: eventType,
        phase: ev.phase,
        rationale: ev.plan?.rationale,
        steps: steps.map((s) => `${s.id}. ${s.description} [${s.status}]`),
      },
    };
  }
  return {
    type: eventType,
    title: `[${agentName}] ${eventType}`,
    attributes: ev,
  };
}

// --- Event Logging System ---
function logEvent(type, title, data, durationMs = null) {
  const now = new Date();
  const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;

  state.events.push({
    id: Date.now() + Math.random().toString(36).substring(2),
    type, title, attributes: data, duration: durationMs, timestamp: timeStr
  });
  if (state.events.length > 500) state.events.shift();
  renderEvents();
}

function renderEvents() {
  const scroll = $("#eventsScroll");
  if (!scroll) return;

  if (state.events.length === 0) {
    scroll.innerHTML = `<div style="text-align:center; padding: 40px 20px; color: var(--text-muted); font-family:var(--mono); font-size:12px;">Waiting for traces...</div>`;
    return;
  }

  scroll.innerHTML = state.events.map((ev, idx) => {
    let rowClass = "info";
    if (ev.type === "tool" || ev.type === "tool_call") rowClass = "tool";
    if (ev.type === "model" || ev.type === "model_call" || ev.type === "model_response") rowClass = "model";
    if (ev.type === "success" || ev.type === "tool_result") rowClass = "success";
    if (ev.type === "error" || ev.type === "warning") rowClass = "error";

    const metricHtml = ev.duration !== null ? `<span class="lf-metric">${ev.duration >= 1000 ? (ev.duration / 1000).toFixed(2) + 's' : ev.duration + 'ms'}</span>` : '';

    const usage = ev.attributes?.usage;
    const summaryHtml = usage ? usageChips(usage) : "";
    let kvHtml = `${summaryHtml}<div class="lf-kv-tree">`;
    if (ev.attributes) {
      Object.keys(ev.attributes).forEach(k => {
        kvHtml += `<div class="lf-kv-row"><span class="lf-k">${k}</span>${renderEventAttribute(k, ev.attributes[k])}</div>`;
      });
    }
    kvHtml += `</div>`;

    return `
          <div class="lf-row ${rowClass}" id="lf-evt-${idx}">
            <div class="lf-summary" data-idx="${idx}">
              <span class="lf-time">${ev.timestamp}</span>
              <span class="lf-icon ${rowClass}"></span>
              <span class="lf-msg">${ev.title}</span>
              <div style="flex:1;"></div>
              ${metricHtml}
            </div>
            <div class="lf-details">${kvHtml}</div>
          </div>
        `;
  }).join('');

  scroll.querySelectorAll('.lf-summary').forEach(header => {
    header.addEventListener('click', () => {
      const idx = header.getAttribute('data-idx');
      const item = document.getElementById(`lf-evt-${idx}`);
      if (item) item.classList.toggle('expanded');
    });
  });

  scroll.scrollTop = scroll.scrollHeight;
}

// --- Workspace / Artifacts ---
async function refreshWorkspace(sync = false) {
  try {
    const params = new URLSearchParams();
    if (state.selectedAgent) params.set("agent_name", state.selectedAgent);
    const res = sync
      ? await fetch(`/api/workspace/sync?${params.toString()}`, { method: "POST" })
      : await fetch(`/api/workspace/files?${params.toString()}`);
    if (!res.ok) throw new Error("Workspace sync failed");
    const payload = await res.json();
    state.workspaceFiles = sync ? (payload.files || []) : payload;
  } catch (err) {
    console.warn("Could not fetch workspace:", err);
    if (sync) logEvent("error", "Workspace sync failed", { error: String(err) });
  }
  renderWorkspace();
}

async function uploadWorkspaceFile(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    logEvent("error", "Upload failed", { error: "Files must be 8 MiB or smaller." });
    return;
  }
  const params = new URLSearchParams({ path: file.name });
  if (state.sessionId) params.set("session_id", state.sessionId);
  if (state.selectedAgent) params.set("agent_name", state.selectedAgent);
  try {
    const response = await fetch(`/api/workspace/upload?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: await file.arrayBuffer(),
    });
    if (!response.ok) {
      const messages = { 400: "Invalid file path or upload.", 409: "A file with that name already exists.", 413: "Files must be 8 MiB or smaller." };
      throw new Error(messages[response.status] || "Upload failed.");
    }
    logEvent("info", "File uploaded", { file: file.name });
    await refreshWorkspace();
  } catch (err) {
    logEvent("error", "Upload failed", { error: String(err) });
  }
}

function renderWorkspace() {
  const panel = $("#workspacePanel");
  if (!panel) return;
  panel.classList.toggle("open", state.showWorkspace);
  $("#workspaceToggle").classList.toggle("active", state.showWorkspace);

  const list = $("#workspaceSidebarFiles");
  const files = Array.isArray(state.workspaceFiles) ? state.workspaceFiles : [];
  if (files.length === 0) {
    list.innerHTML = `<div style="text-align:center; padding: 40px 20px; color: var(--text-muted); font-size: 13px;">No files generated yet.</div>`;
  } else {
    const nameCounts = new Map();
    files.forEach((file) => {
      const name = String(file?.name ?? "");
      nameCounts.set(name, (nameCounts.get(name) || 0) + 1);
    });
    list.innerHTML = files.map((file, index) => {
      const name = String(file?.name ?? "");
      const path = String(file?.path ?? "");
      const syncState = String(file?.sync_status?.state ?? "error");
      const extension = name.split(".").pop() || "txt";
      const size = Number(file?.size_bytes);
      const duplicatePath = nameCounts.get(name) > 1
        ? `<span class="workspace-file-path">${escapeHtml(path)}</span>`
        : "";
      return `
          <button class="workspace-file" data-file-index="${index}">
            <span class="workspace-file-icon">${escapeHtml(extension)}</span>
            <div style="min-width:0; flex:1;">
              <span class="workspace-file-name">${escapeHtml(name)}</span>
              ${duplicatePath}
              <span class="workspace-file-meta">${Number.isFinite(size) ? `${(size / 1024).toFixed(1)} KB` : ""}</span>
              <span class="workspace-file-meta" aria-label="Sync status">${escapeHtml(syncState)}</span>
            </div>
          </button>
        `;
    }).join("");

    list.querySelectorAll('.workspace-file').forEach(btn => {
      btn.addEventListener('click', () => {
        const file = files[Number(btn.dataset.fileIndex)];
        if (file) openPreview(file.path, file.name);
      });
    });
  }
  $("#workspaceMeta").innerText = `${files.length} files`;
}

async function openPreview(path, name) {
  const selectedAgent = state.selectedAgent;
  const modalTitle = $("#modalTitle");
  const modalBody = $("#workspaceModalBody");
  modalTitle.textContent = String(name ?? "");
  modalBody.innerHTML = `<div style="text-align:center; padding: 40px;"><div class="spinner" style="margin: 0 auto;"></div></div>`;
  state.workspaceModalOpen = true;
  $("#workspaceModal").classList.add("open");

  try {
    const query = new URLSearchParams({ path: String(path ?? "") });
    if (selectedAgent) query.set("agent_name", selectedAgent);
    const res = await fetch(`/api/workspace/raw?${query.toString()}`);
    if (!res.ok) throw new Error("Failed to load file");
    const text = await res.text();

    const preview = document.createElement("div");
    preview.className = "msg-text";
    preview.style.cssText = "background: var(--bg-0); padding: 20px; border-radius: var(--radius-md); border: 1px solid var(--border-1); overflow-y: auto; max-height: 100%;";
    const pre = document.createElement("pre");
    pre.style.cssText = "white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--mono); font-size: 13px; color: var(--text);";
    pre.textContent = text;
    preview.appendChild(pre);
    modalBody.replaceChildren(preview);
    logEvent("info", `Viewed artifact: ${String(name ?? "")}`, {
      action: "Preview",
      file: String(path ?? ""),
      bytes: text.length,
    });
  } catch (err) {
    const error = document.createElement("div");
    error.style.cssText = "color: var(--danger); padding: 20px;";
    error.textContent = "Could not load preview.";
    modalBody.replaceChildren(error);
  }
}

// --- Plan / Todo List Panel ---
const PLAN_STATUS_ICON = {
  pending: "○",
  active: "◐",
  done: "✓",
  failed: "✗",
};

function renderPlan() {
  const panel = $("#planPanel");
  if (!panel) return;

  const plan = state.currentPlan;
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  if (!plan || steps.length === 0) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }

  const done = steps.filter((s) => s.status === "done").length;
  const active = steps.find((s) => s.status === "active");
  const open = state.planPanelOpen !== false; // default expanded
  const rows = steps
    .map((s, index) => {
      const status = s.status || "pending";
      const icon = PLAN_STATUS_ICON[status] || "○";
      return `
        <li class="plan-step is-${escapeHtml(status)}">
          <span class="plan-step-num">${index + 1}.</span>
          <span class="plan-step-icon">${icon}</span>
          <span class="plan-step-text">${escapeHtml(s.description || "")}</span>
          ${status === "active" ? `<span class="plan-step-badge">active</span>` : ""}
        </li>`;
    })
    .join("");

  panel.hidden = false;
  panel.classList.toggle("is-open", open);
  panel.innerHTML = `
    <div class="plan-header" onclick="togglePlanPanel()">
      <span class="plan-caret">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </span>
      <span class="plan-header-icon">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/>
          <line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
        </svg>
      </span>
      <span class="plan-title">Todos</span>
      <span class="plan-active-hint">${escapeHtml(active?.description || "")}</span>
      <span class="plan-progress">${done}/${steps.length} done</span>
      <button type="button" class="plan-dismiss" onclick="dismissPlan(event)" aria-label="Dismiss todo list" title="Dismiss">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <ul class="plan-steps">${rows}</ul>`;
  syncChatPadding();
}

// Keep the chat scroll area clear of the (variable-height) footer:
// plan panel + attachments + textarea growth all change its height.
function syncChatPadding() {
  const scroll = $("#chatScroll");
  const area = document.querySelector(".input-area");
  if (!scroll || !area) return;
  scroll.style.paddingBottom = `${area.offsetHeight + 24}px`;
}

window.togglePlanPanel = function () {
  state.planPanelOpen = state.planPanelOpen === false;
  renderPlan();
};

window.dismissPlan = function (event) {
  if (event) event.stopPropagation();
  state.currentPlan = null;
  renderPlan();
};

// --- Human-in-the-loop question card ---
// Renders an AskUserQuestion-style card: a full-width bordered panel whose
// options are numbered rows (label + optional description), with a
// "Type something else…" free-text row, keyboard shortcuts (1-9), a Skip
// button, and multi-select support when pending.multiSelect is set.
function inputRequestHtml(pending, cardIndex = 0, totalCards = 1) {
  const question = escapeHtml(pending.question || "The agent needs your input.");
  const opts = Array.isArray(pending.options) ? pending.options : [];
  const hasOptions = opts.length > 0;
  const multi = Boolean(pending.multiSelect);
  const selected = pending.selected instanceof Set ? pending.selected : new Set();
  const customOpen = Boolean(pending.customOpen) || !hasOptions;
  // Number-key shortcuts only apply to the first card of a batch.
  const showKeys = cardIndex === 0;

  const optionRows = opts.map((raw, index) => {
    const { label, description } = parseOption(raw);
    const isSelected = selected.has(index);
    const key = index + 1;
    return `
      <button type="button" class="input-req-option${isSelected ? " is-selected" : ""}"
        data-input-card="${cardIndex}" data-input-index="${index}" data-input-multi="${multi ? "1" : "0"}">
        <span class="input-req-option-body">
          <span class="input-req-option-label">${escapeHtml(label)}</span>
          ${description ? `<span class="input-req-option-desc">${escapeHtml(description)}</span>` : ""}
        </span>
        ${showKeys ? `<span class="input-req-key">${key <= 9 ? key : "•"}</span>` : ""}
      </button>`;
  }).join("");

  const customRow = hasOptions
    ? (customOpen
        ? `<form class="input-req-form" onsubmit="submitUserInputFromForm(event, ${cardIndex}); return false;">
             <input type="text" class="input-req-text" placeholder="Type something else…" autocomplete="off" ${cardIndex === 0 ? "autofocus" : ""} />
             <button type="submit" class="input-req-send">Send</button>
           </form>`
        : `<div class="input-req-custom">
             <button type="button" class="input-req-custom-trigger" onclick="openCustomInput(${cardIndex})">
               <span class="input-req-option-body"><span>Type something else…</span></span>
               ${showKeys ? `<span class="input-req-key">${Math.min(opts.length + 1, 9)}</span>` : ""}
             </button>
           </div>`)
    : `<form class="input-req-form" onsubmit="submitUserInputFromForm(event, ${cardIndex}); return false;">
         <input type="text" class="input-req-text" placeholder="Type your answer…" autocomplete="off" ${cardIndex === 0 ? "autofocus" : ""} />
         <button type="submit" class="input-req-send">Send</button>
       </form>`;

  const progress = totalCards > 1
    ? `<span class="input-req-count">${cardIndex + 1}/${totalCards}</span>`
    : (hasOptions ? `<span class="input-req-count">${opts.length}</span>` : "");
  const confirmBtn = multi
    ? `<button type="button" class="input-req-confirm" onclick="confirmMultiInput(${cardIndex})">Confirm${selected.size ? ` (${selected.size})` : ""}</button>`
    : "";

  return `
    <div class="input-req-card" data-card-index="${cardIndex}">
      <div class="input-req-head">
        <p class="input-req-question">${question}</p>
        ${progress}
      </div>
      <div class="input-req-options">${optionRows}</div>
      ${customRow}
      <div class="input-req-foot">
        <button type="button" class="input-req-skip" onclick="skipUserInput(${cardIndex})">Skip</button>
        <span class="input-req-foot-spacer"></span>
        ${confirmBtn}
      </div>
    </div>`;
}

// --- Chat Rendering ---
function renderMessages() {
  const container = $("#messagesContainer");
  const empty = $("#emptyState");
  const scroll = $("#chatScroll");
  const wasNearBottom = !scroll || (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight) < 96;

  container.querySelectorAll(".thinking-block").forEach((details) => {
    const key = details.dataset.thinkingKey;
    if (key) state.thinkingPanels[key] = details.open;
  });

  container.classList.toggle("is-streaming", state.isRunning);

  if (state.messages.length === 0 && state.pendingApprovals.length === 0) {
    empty.style.display = "flex"; container.style.display = "none";
    return;
  }

  empty.style.display = "none"; container.style.display = "block";

  let activityRendered = false;
  const activeThinkingMessage = state.messages.some((msg) => (
    msg.role === "assistant" && msg.streaming && String(msg.thinking || "").trim()
  ));
  let html = runningToolsHtml();
  html += state.messages.map((msg, index) => {
    if (msg.role === "approval_status") return ''; // Skip internal status renders for clean UI
    if (msg.role === "tool") return ''; // Tool outputs stay in traces, not the chat transcript.
    if (msg.interim) return ''; // Guard-vetoed draft answer — superseded by the next one.

    const isUser = msg.role === 'user';
    const bubbleCls = isUser ? 'msg-bubble user-bubble' : 'msg-bubble bot-bubble';
    const rowCls = isUser ? 'msg-row user-row' : 'msg-row bot-row';
    const hasContent = String(msg.content || "").trim().length > 0;
    const content = isUser ? escapeHtml(msg.content).replace(/\n/g, "<br>") : (window.__markedReady ? marked.parse(msg.content || "") : escapeHtml(msg.content));
    const imagesHtml = renderImageList(msg.images);

    let thinkingHtml = '';
    if (!isUser && msg.thinking) {
      const thinkingKey = thinkingKeyForMessage(msg, index);
      const shouldOpen = state.thinkingPanels[thinkingKey] ?? Boolean(msg.streaming);
      const open = shouldOpen ? " open" : "";
      const thinking = shouldOpen
        ? (window.__markedReady ? marked.parse(msg.thinking) : escapeHtml(msg.thinking))
        : "";
      thinkingHtml = `
        <details class="thinking-block" data-thinking-key="${escapeHtml(thinkingKey)}"${open}>
          <summary>
            <span class="thinking-title">Thinking</span>
            <small>${fmtCount(String(msg.thinking).length)} chars</small>
          </summary>
          <div class="thinking-content">${thinking}</div>
        </details>
      `;
    }

    let prefix = "";
    if (!activeThinkingMessage && !activityRendered && msg.role === "assistant" && msg.streaming) {
      prefix = activityHtml();
      activityRendered = true;
    }
    const textHtml = hasContent || isUser ? `<div class="msg-text">${content}</div>` : "";
    return `${prefix}<div class="${rowCls}"><div class="${bubbleCls}">${thinkingHtml}${textHtml}${imagesHtml}</div></div>`;
  }).join('');
  if (!activeThinkingMessage && !activityRendered) html += activityHtml();

  // Pending Approvals
  if (state.pendingApprovals.length > 0) {
    const disabled = state.isRunning ? " disabled" : "";
    html += state.pendingApprovals.map(ap => {
      const params = ap.parameters && Object.keys(ap.parameters).length
        ? `<details class="approval-params">
             <summary>Parameters</summary>
             <pre>${escapeHtml(JSON.stringify(ap.parameters, null, 2))}</pre>
           </details>`
        : "";
      return `
        <div class="approval-card">
          <div class="approval-head">
            <span class="approval-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
            </span>
            <span class="approval-title">Allow <code>${escapeHtml(ap.tool_name)}</code>?</span>
          </div>
          <p class="approval-reason">${escapeHtml(ap.reason || "The agent wants to run this tool.")}</p>
          ${params}
          <div class="approval-foot">
            <button class="btn-deny" onclick="submitApproval('${ap.tool_call_id}', false)"${disabled}>Deny</button>
            <button class="btn-approve" onclick="submitApproval('${ap.tool_call_id}', true)"${disabled}>Approve</button>
          </div>
        </div>`;
    }).join('');
  }

  // Pending human-input requests — a batch renders as stacked cards.
  if (state.pendingInputs.length > 0) {
    const total = state.pendingInputs.length;
    html += state.pendingInputs
      .map((pending, index) => inputRequestHtml(pending, index, total))
      .join("");
  }

  container.innerHTML = html;
  if (scroll && wasNearBottom) {
    scroll.scrollTop = scroll.scrollHeight;
  }
}

function renderStatusArea() {
  const area = $("#statusArea");
  const input = $("#chatInput");
  if (input) {
    input.readOnly = state.isRunning;
    input.classList.toggle("is-running", state.isRunning);
  }
  if (state.isRunning) {
    const text = state.isCancelling ? "Stopping..." : "Agent is thinking...";
    area.innerHTML = `<span class="status-dot"></span> <span>${text}</span>`;
  } else {
    area.innerHTML = ``;
  }
}

// --- API Communication (SSE) ---
async function readSseStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // Keep remainder

    for (const p of parts) {
      const line = p.split("\n").find(l => l.startsWith("data: "));
      if (!line) continue;

      try {
        const packet = JSON.parse(line.slice(6));
        handlePacket(packet);
      } catch (e) { console.warn("Parse error", e); }
    }
  }
}

function handlePacket(packet) {
  // Route events to Logfire Panel based on serve.py payload structure
  if (packet.type === "agent_event" && packet.event) {
    const ev = packet.event;
    const summarized = summarizeEventForLog(packet.agent_name, ev);
    logEvent(summarized.type, summarized.title, summarized.attributes);
    if (ev.type === "bash_started") {
      pushActivity(`${ev.description} (intención declarada: ${ev.declared_action})`, "tool");
    } else if (ev.type === "bash_finished") {
      pushActivity(ev.timed_out ? "Bash: tiempo agotado" : `Bash terminado · exit ${ev.exit_code}`, ev.timed_out || ev.exit_code !== 0 ? "error" : "tool");
    } else if (ev.type === "bash_failed" || ev.type === "bash_cancelled") {
      pushActivity(ev.error || ev.reason, "error");
    } else if (ev.type === "tool_call") {
      state.runningTools[ev.tool_call_id] = {
        tool_call_id: ev.tool_call_id,
        tool_name: ev.tool_name,
        started_at: Date.now(),
      };
      pushActivity(`Running tool: ${ev.tool_name || "tool"}`, "tool");
    } else if (ev.type === "tool_result") {
      delete state.runningTools[ev.tool_call_id];
      const denied = String(ev.error || "").toLowerCase().includes("denied")
        || String(ev.error || "").toLowerCase().includes("declined")
        || String(ev.error || "").toLowerCase().includes("rejected");
      pushActivity(
        ev.success ? "Tool completed" : (denied ? "Skipped denied tool" : "Tool failed"),
        ev.success ? "tool" : (denied ? "approval" : "error"),
      );
    } else if (ev.type === "approval_item") {
      delete state.runningTools[ev.item?.tool_call_id];
      pushActivity(`Waiting for approval: ${ev.item?.tool_name || "tool"}`, "approval");
    } else if (ev.type === "compaction") {
      state.compactionRunning = ev.phase === "start";
      if (ev.phase === "start") {
        pushActivity("Compacting context...", "info");
      } else if (ev.changed) {
        pushActivity("Context compacted", "info");
      }
      refreshContextWindow();
    } else if (ev.type === "planning") {
      if (ev.phase === "start") {
        pushActivity("Planning...", "info");
      }
      if (ev.plan) {
        const steps = Array.isArray(ev.plan.steps) ? ev.plan.steps : [];
        const allDone = steps.length > 0
          && steps.every((s) => s.status === "done" || s.status === "failed");
        // Fresh plan → expand; finished plan → collapse to a slim bar.
        state.planPanelOpen = !allDone;
        state.currentPlan = ev.plan;
        renderPlan();
      }
    } else if (ev.type === "user_input_request") {
      // Batched questions arrive as one event each — accumulate them.
      const exists = state.pendingInputs.some(
        (p) => p.tool_call_id && p.tool_call_id === ev.tool_call_id
      );
      if (!exists) {
        state.pendingInputs.push({
          question: ev.question || "",
          options: ev.options || null,
          tool_call_id: ev.tool_call_id || null,
          multiSelect: Boolean(ev.multi_select),
          selected: new Set(),
          customOpen: false,
        });
      }
      pushActivity("Waiting for your input...", "approval");
      renderMessages();
    }
  } else if (packet.type === "error") {
    state.compactionRunning = false;
    logEvent("error", "System Error", { message: packet.message });
    pushActivity(packet.message || "Something went wrong.", "error");
  }

  // Handle UI State
  if (packet.type === "status") {
    if (packet.status === "resuming") {
      pushActivity("Resuming with approved tools...", "info");
    } else if (packet.message) {
      pushActivity(packet.message, "approval");
    }
  }
  else if (packet.type === "thinking_delta") {
    let lastMsg = state.messages[state.messages.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant" || !lastMsg.streaming) {
      state.messages.push({ role: "assistant", streaming: true, content: "", thinking: packet.content });
      lastMsg = state.messages[state.messages.length - 1];
    } else {
      lastMsg.thinking = packet.content;
    }
    const thinkingKey = thinkingKeyForMessage(lastMsg, state.messages.length - 1);
    if (state.thinkingPanels[thinkingKey] === undefined) state.thinkingPanels[thinkingKey] = true;
    scheduleThinkingAutoClose(thinkingKey);
  }
  else if (packet.type === "assistant_delta") {
    let lastMsg = state.messages[state.messages.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant" || !lastMsg.streaming) {
      state.messages.push({ role: "assistant", streaming: true, content: packet.content, thinking: "" });
    } else {
      lastMsg.content = packet.content;
    }
  }
  else if (packet.type === "agent_complete") {
    state.runningTools = {};
    Object.values(state.thinkingAutoCloseTimers).forEach(clearTimeout);
    state.thinkingAutoCloseTimers = {};
    clearActivity();
    if (packet.assistant_message) {
      const streamingAssistant = [...state.messages].reverse().find((m) => m.role === "assistant" && m.streaming);
      const streamedThinking = streamingAssistant?.thinking || "";
      const finalMessage = { ...packet.assistant_message };
      if (!finalMessage.thinking && streamedThinking) finalMessage.thinking = streamedThinking;
      // Replace streaming msg with final msg
      state.messages = state.messages.filter(m => !m.streaming);
      state.messages.push(finalMessage);
    }
  }
  else if (packet.type === "session_state") {
    state.messages = packet.messages;
    state.pendingApprovals = packet.pending_approvals || [];
    state.contextUsage = packet.context_usage || state.contextUsage;
    state.runningTools = {};
    refreshWorkspace();
    renderContextWindow();
  }
  else if (packet.type === "approval_required") {
    state.pendingApprovals = packet.pending_approvals || [];
    state.isRunning = false;
    state.isCancelling = false;
    clearActivity();
  }
  else if (packet.type === "input_required") {
    // Authoritative list of every still-unanswered question — replaces
    // whatever the per-question stream events accumulated.
    state.pendingInputs = (packet.pending_questions || []).map((q) => ({
      question: q.question || "",
      options: q.options || null,
      tool_call_id: q.tool_call_id || null,
      multiSelect: Boolean(q.multi_select),
      selected: new Set(),
      customOpen: false,
    }));
    state.isRunning = false;
    state.isCancelling = false;
    clearActivity();
    pushActivity("Waiting for your input...", "approval");
  }
  else if (packet.type === "cancelled") {
    state.messages.forEach((msg) => {
      if (msg.role === "assistant" && msg.streaming) msg.streaming = false;
    });
    state.isRunning = false;
    state.isCancelling = false;
    state.currentAbortController = null;
    state.compactionRunning = false;
    clearActivity();
    pushActivity(packet.message || "Turn cancelled.", "approval");
  }
  else if (packet.type === "bash_started") {
    pushActivity(`${packet.description} (intención declarada: ${packet.declared_action})`, "tool");
  }
  else if (packet.type === "bash_finished") {
    pushActivity(packet.timed_out ? "Bash: tiempo agotado" : `Bash terminado · exit ${packet.exit_code}`, packet.timed_out || packet.exit_code !== 0 ? "error" : "tool");
  }
  else if (packet.type === "bash_failed" || packet.type === "bash_cancelled") {
    pushActivity(packet.error || packet.reason, "error");
  }
  else if (packet.type === "tool_call") {
    state.runningTools[packet.tool_call_id] = {
      tool_call_id: packet.tool_call_id,
      tool_name: packet.tool_name,
      started_at: Date.now(),
    };
    pushActivity(`Running tool: ${packet.tool_name || "tool"}`, "tool");
  }
  else if (packet.type === "tool_result") {
    delete state.runningTools[packet.tool_call_id];
    const denied = String(packet.error || "").toLowerCase().includes("denied")
      || String(packet.error || "").toLowerCase().includes("declined")
      || String(packet.error || "").toLowerCase().includes("rejected");
    pushActivity(
      packet.success ? "Tool completed" : (denied ? "Skipped denied tool" : "Tool failed"),
      packet.success ? "tool" : (denied ? "approval" : "error"),
    );
  }

  scheduleRender();
}

// --- Actions ---
function renderAttachmentPreview() {
  const container = $("#attachmentPreview");
  if (!container) return;
  if (state.pendingImages.length === 0) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = state.pendingImages.map((image, index) => `
    <div class="attachment-chip">
      <img src="${escapeHtml(image.preview_url)}" alt="" />
      <span title="${escapeHtml(image.name)}">${escapeHtml(image.name)}</span>
      <button type="button" data-remove-image="${index}" aria-label="Remove image">×</button>
    </div>
  `).join("");
  container.querySelectorAll("[data-remove-image]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const index = Number(btn.dataset.removeImage);
      state.pendingImages.splice(index, 1);
      renderAttachmentPreview();
      updateSendBtn();
    });
  });
  syncChatPadding();
}

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const [, dataBase64 = ""] = dataUrl.split(",", 2);
      resolve({
        name: file.name,
        mime_type: file.type || "image/png",
        data_base64: dataBase64,
        preview_url: dataUrl,
      });
    };
    reader.onerror = () => reject(reader.error || new Error("Could not read image"));
    reader.readAsDataURL(file);
  });
}

async function handleImageSelect(event) {
  if (!selectedAgentConfig()?.supports_vision) {
    logEvent("error", "Image Attach Blocked", { reason: "Selected model does not support image input" });
    event.target.value = "";
    return;
  }
  const files = Array.from(event.target.files || []).filter((file) => file.type.startsWith("image/"));
  if (files.length === 0) return;
  try {
    const images = await Promise.all(files.map(readImageFile));
    state.pendingImages.push(...images);
    renderAttachmentPreview();
    updateSendBtn();
  } catch (err) {
    logEvent("error", "Image Attach Failed", { error: String(err) });
  } finally {
    event.target.value = "";
  }
}

async function handleSend(e) {
  e.preventDefault();
  const input = $("#chatInput");
  const msg = input.value.trim();
  const images = state.pendingImages;
  if ((!msg && images.length === 0) || state.isRunning || !state.sessionId) return;

  state.messages.push({ role: "user", content: msg, images });
  input.value = "";
  state.pendingImages = [];
  renderAttachmentPreview();
  input.blur();
  input.style.height = 'auto';
  state.isRunning = true;
  state.isCancelling = false;
  state.currentAbortController = new AbortController();
  clearActivity();
  pushActivity("Preparing request...", "info");
  updateSendBtn();

  renderMessages();
  renderStatusArea();
  logEvent("info", "User Prompt", { input_length: msg.length, images: images.length });

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.currentAbortController.signal,
      body: JSON.stringify({
        session_id: state.sessionId,
        message: msg || "Please analyze the attached image.",
        agent_names: [state.selectedAgent],
        images: images.map(({ name, mime_type, data_base64 }) => ({ name, mime_type, data_base64 }))
      })
    });
    if (!res.ok) throw new Error(await res.text());
    await readSseStream(res);
  } catch (err) {
    if (err?.name !== "AbortError" || !state.isCancelling) {
      logEvent("error", "Request Failed", { error: String(err) });
    }
  } finally {
    state.isRunning = false;
    state.isCancelling = false;
    state.currentAbortController = null;
    state.compactionRunning = false;
    renderStatusArea();
    updateSendBtn();
  }
}

async function handleStop() {
  if (!state.isRunning || state.isCancelling || !state.sessionId) return;
  state.isCancelling = true;
  clearActivity();
  pushActivity("Stopping...", "approval");
  renderMessages();
  renderStatusArea();
  updateSendBtn();

  try {
    await fetch("/api/chat/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        agent_name: state.selectedAgent,
      }),
    });
  } catch (err) {
    logEvent("error", "Cancel Request Failed", { error: String(err) });
  } finally {
    state.currentAbortController?.abort();
  }
}

window.submitUserInputFromForm = function (event, cardIndex = 0) {
  const input = event.target.querySelector(".input-req-text");
  const value = input ? input.value.trim() : "";
  if (value) window.submitUserInput(value, cardIndex);
};

// Reveal the free-text field on one question card.
window.openCustomInput = function (cardIndex = 0) {
  const pending = state.pendingInputs[cardIndex];
  if (!pending) return;
  pending.customOpen = true;
  renderMessages();
  const field = document.querySelector(
    `.input-req-card[data-card-index="${cardIndex}"] .input-req-text`
  );
  if (field) field.focus();
};

// Skip one question — sends an empty answer so the run resumes.
window.skipUserInput = function (cardIndex = 0) {
  if (!state.pendingInputs[cardIndex]) return;
  window.submitUserInput("", cardIndex);
};

// Toggle one option in a multi-select question (no submit yet).
window.toggleMultiInput = function (cardIndex, index) {
  const pending = state.pendingInputs[cardIndex];
  if (!pending) return;
  if (!(pending.selected instanceof Set)) pending.selected = new Set();
  if (pending.selected.has(index)) pending.selected.delete(index);
  else pending.selected.add(index);
  renderMessages();
};

// Submit the accumulated selections of a multi-select question.
window.confirmMultiInput = function (cardIndex = 0) {
  const pending = state.pendingInputs[cardIndex];
  if (!pending) return;
  const opts = Array.isArray(pending.options) ? pending.options : [];
  const chosen = [...(pending.selected || [])]
    .sort((a, b) => a - b)
    .map((i) => parseOption(opts[i]).label);
  if (chosen.length === 0) return;
  window.submitUserInput(chosen.join(", "), cardIndex);
};

// Pick a single option by its list index.
window.pickInputOption = function (cardIndex, index) {
  const pending = state.pendingInputs[cardIndex];
  if (!pending) return;
  const opts = Array.isArray(pending.options) ? pending.options : [];
  const raw = opts[index];
  if (raw === undefined) return;
  window.submitUserInput(parseOption(raw).label, cardIndex);
};

window.submitUserInput = async function (answer, cardIndex = 0) {
  const pending = state.pendingInputs[cardIndex];
  if (!pending || !state.sessionId) return;
  if (state.isRunning) return;
  // The previous stream ended with finish_reason='input_needed'; answering
  // applies the answer to the run's tool state on the server and resumes it
  // as a new stream segment, which we consume here (same shape as approvals).
  // With a batch of questions the server only resumes the model once the
  // LAST one is answered — until then it replies with the remaining list.
  state.pendingInputs = state.pendingInputs.filter((p) => p !== pending);
  state.isRunning = true;
  state.isCancelling = false;
  state.currentAbortController = new AbortController();
  clearActivity();
  pushActivity("Input sent.", "approval");
  renderMessages();
  renderStatusArea();
  logEvent("info", "User input provided", { answer });

  try {
    const res = await fetch("/api/chat/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.currentAbortController.signal,
      body: JSON.stringify({
        session_id: state.sessionId,
        agent_name: state.selectedAgent,
        answer,
        tool_call_id: pending.tool_call_id || null,
      }),
    });
    if (!res.ok) throw new Error("Input failed");
    await readSseStream(res);
  } catch (err) {
    if (err?.name !== "AbortError" || !state.isCancelling) {
      state.pendingInputs = [pending, ...state.pendingInputs];
      logEvent("error", "User input failed", { error: String(err) });
      pushActivity("Failed to send input.", "error");
    }
  } finally {
    state.isRunning = false;
    state.isCancelling = false;
    state.currentAbortController = null;
    state.compactionRunning = false;
    renderStatusArea();
    updateSendBtn();
  }
};

window.submitApproval = async function (toolCallId, approved) {
  if (state.isRunning) return;
  const previousApprovals = [...state.pendingApprovals];
  state.isRunning = true;
  state.isCancelling = false;
  state.currentAbortController = new AbortController();
  state.pendingApprovals = state.pendingApprovals.filter(ap => ap.tool_call_id !== toolCallId);
  clearActivity();
  pushActivity(approved ? "Approval saved." : "Decision saved.", "approval");
  renderMessages();
  renderStatusArea();

  logEvent("info", `Approval Decision: ${approved ? 'Approve' : 'Deny'}`, { tool_call_id: toolCallId });

  try {
    const res = await fetch("/api/chat/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.currentAbortController.signal,
      body: JSON.stringify({
        session_id: state.sessionId,
        agent_name: state.selectedAgent,
        decisions: [{ tool_call_id: toolCallId, approved, reason: approved ? "Approved by user" : "Denied by user" }]
      })
    });
    if (!res.ok) throw new Error("Approval failed");
    await readSseStream(res);
  } catch (err) {
    if (err?.name !== "AbortError" || !state.isCancelling) {
      state.pendingApprovals = previousApprovals;
      logEvent("error", "Approval Request Failed", { error: String(err) });
      pushActivity("Approval request failed.", "error");
    }
  } finally {
    state.isRunning = false;
    state.isCancelling = false;
    state.currentAbortController = null;
    state.compactionRunning = false;
    renderStatusArea();
    updateSendBtn();
  }
}

async function handleNewChat() {
  try {
    const session = await fetch("/api/sessions", { method: "POST" }).then(r => r.json());
    state.sessionId = session.session_id;
    state.contextUsage = session.context_usage || null;
    state.contextWindowOpen = false;
    state.compactionRunning = false;
    state.messages = [];
    state.events = [];
    state.pendingApprovals = [];
    state.pendingImages = [];
    state.runningTools = {};
    state.isCancelling = false;
    state.currentAbortController = null;
    state.currentPlan = null;
    state.pendingInputs = [];
    clearActivity();
    $("#messagesContainer").innerHTML = "";
    renderAttachmentPreview();
    refreshWorkspace();
    renderPlan();
    renderMessages();
    renderEvents();
    updateSendBtn();
    renderContextWindow();
    logEvent("info", "New Session Initialized", { session_id: state.sessionId });
  } catch (err) {
    logEvent("error", "Failed to start session", { error: String(err) });
  }
}

function updateSendBtn() {
  const text = $("#chatInput").value.trim();
  const btn = $("#sendBtn");
  if (state.isRunning) {
    btn.disabled = state.isCancelling;
    btn.classList.add("is-stop");
    btn.setAttribute("aria-label", state.isCancelling ? "Stopping" : "Stop response");
    btn.innerHTML = `<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5" /></svg>`;
  } else {
    btn.disabled = (text.length === 0 && state.pendingImages.length === 0) || !state.sessionId;
    btn.classList.remove("is-stop");
    btn.setAttribute("aria-label", "Send message");
    btn.innerHTML = `<svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>`;
  }
  updateAttachmentControls();
  renderContextWindow();
}

// --- Init ---
async function initApp() {
  initTheme();

  try {
    // Fetch agents
    const agents = await fetch("/api/agents").then(r => r.json());
    state.agents = agents;
    const def = agents.find(a => a.default_selected) || agents[0];
    if (def) {
      state.selectedAgent = def.name;
      $("#activeAgentLabel").innerText = def.label;
      updateAttachmentControls();
    }

    // Create Session
    await handleNewChat();
  } catch (err) {
    console.error("Initialization error:", err);
    $("#activeAgentLabel").innerText = "API Disconnected";
  }

  // Listeners
  const messagesContainer = $("#messagesContainer");
  messagesContainer.addEventListener("pointerdown", (event) => {
    const summary = event.target.closest(".thinking-block summary");
    if (!summary) return;
    event.preventDefault();
    toggleThinkingPanel(summary.closest(".thinking-block"));
  });
  messagesContainer.addEventListener("click", (event) => {
    const summary = event.target.closest(".thinking-block summary");
    if (!summary) return;
    event.preventDefault();
  });
  messagesContainer.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const summary = event.target.closest(".thinking-block summary");
    if (!summary) return;
    event.preventDefault();
    toggleThinkingPanel(summary.closest(".thinking-block"));
  });
  // Question-card option rows: single-select submits, multi-select toggles.
  messagesContainer.addEventListener("click", (event) => {
    const option = event.target.closest(".input-req-option");
    if (!option) return;
    const cardIndex = Number(option.dataset.inputCard || 0);
    const index = Number(option.dataset.inputIndex);
    if (Number.isNaN(index)) return;
    if (option.dataset.inputMulti === "1") window.toggleMultiInput(cardIndex, index);
    else window.pickInputOption(cardIndex, index);
  });
  // Number keys 1-9 answer the FIRST pending question of a batch.
  document.addEventListener("keydown", (event) => {
    if (state.pendingInputs.length === 0 || state.isRunning) return;
    if (event.target.closest("input, textarea")) return;
    if (!/^[1-9]$/.test(event.key)) return;
    const pending = state.pendingInputs[0];
    const opts = Array.isArray(pending.options) ? pending.options : [];
    const index = Number(event.key) - 1;
    if (index === opts.length && opts.length) {
      event.preventDefault();
      window.openCustomInput(0);
      return;
    }
    if (index < 0 || index >= opts.length) return;
    event.preventDefault();
    if (pending.multiSelect) window.toggleMultiInput(0, index);
    else window.pickInputOption(0, index);
  });
  $("#themeToggle").addEventListener("click", toggleTheme);
  $("#workspaceToggle").addEventListener("click", () => {
    state.showWorkspace = !state.showWorkspace;
    renderWorkspace();
  });
  $("#workspaceRefresh").addEventListener("click", () => refreshWorkspace(true));
  $("#workspaceUpload").addEventListener("click", () => $("#workspaceUploadInput").click());
  $("#workspaceUploadInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    uploadWorkspaceFile(file);
  });
  $("#eventsToggle").addEventListener("click", () => {
    state.showEvents = !state.showEvents;
    $("#eventsPanel").classList.toggle("open", state.showEvents);
    $("#eventsToggle").classList.toggle("active", state.showEvents);
  });
  $("#eventsClose").addEventListener("click", () => {
    state.showEvents = false;
    $("#eventsPanel").classList.remove("open");
    $("#eventsToggle").classList.remove("active");
  });
  $("#lfClear").addEventListener("click", () => { state.events = []; renderEvents(); });
  $("#workspaceModalClose").addEventListener("click", () => {
    state.workspaceModalOpen = false;
    $("#workspaceModal").classList.remove("open");
  });

  $("#chatForm").addEventListener("submit", handleSend);
  $("#sendBtn").addEventListener("click", (event) => {
    event.preventDefault();
    if (state.isRunning) {
      handleStop();
    } else {
      handleSend(event);
    }
  });
  $("#attachImageBtn").addEventListener("click", () => $("#imageInput").click());
  $("#contextWindowBtn").addEventListener("click", toggleContextWindow);
  document.addEventListener("pointerdown", (event) => {
    if (!state.contextWindowOpen) return;
    if (event.target.closest("#contextWindowBtn") || event.target.closest("#contextWindowPopover")) return;
    state.contextWindowOpen = false;
    renderContextWindow();
  });
  $("#imageInput").addEventListener("change", handleImageSelect);
  $("#chatInput").addEventListener("input", (e) => {
    updateSendBtn();
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
    syncChatPadding();
  });
  window.addEventListener("resize", syncChatPadding);
  syncChatPadding();
  $("#chatInput").addEventListener("keydown", (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(e); }
  });
  $("#newChatBtn").addEventListener("click", handleNewChat);
}

window.addEventListener("DOMContentLoaded", initApp);
