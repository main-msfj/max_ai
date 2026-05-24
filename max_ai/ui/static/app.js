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
  const summary = Number(usage.summary_tokens || 0);
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
      ${segment(summary, "summary", "Summary")}
      ${segment(live, "live", "Live messages")}
      ${segment(safety, "safety", "Safety margin")}
      ${segment(reserved, "reserved", "Reserved output")}
    </div>
    <div class="cw-legend">
      <span><i class="prompt"></i>Prompt ${fmtCount(prompt)}</span>
      <span><i class="summary"></i>Summary ${fmtCount(summary)}</span>
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
    const toolCalls = msg.tool_calls ? `<span class="lf-message-chip">${fmtCount(msg.tool_calls)} tool call${msg.tool_calls === 1 ? "" : "s"}</span>` : "";

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
async function refreshWorkspace() {
  try {
    // Backend doesn't strictly need session_id query param for workspace_files
    const res = await fetch("/api/workspace/files");
    if (res.ok) state.workspaceFiles = await res.json();
  } catch (err) {
    console.warn("Could not fetch workspace:", err);
  }
  renderWorkspace();
}

function renderWorkspace() {
  const panel = $("#workspacePanel");
  if (!panel) return;
  panel.classList.toggle("open", state.showWorkspace);
  $("#workspaceToggle").classList.toggle("active", state.showWorkspace);

  const list = $("#workspaceSidebarFiles");
  if (state.workspaceFiles.length === 0) {
    list.innerHTML = `<div style="text-align:center; padding: 40px 20px; color: var(--text-muted); font-size: 13px;">No files generated yet.</div>`;
  } else {
    list.innerHTML = state.workspaceFiles.map(f => `
          <button class="workspace-file" data-url="${f.url}" data-name="${f.name}">
            <span class="workspace-file-icon">${f.name.split('.').pop() || 'txt'}</span>
            <div style="min-width:0; flex:1;">
              <span class="workspace-file-name">${f.name}</span>
              <span class="workspace-file-meta">${(f.size_bytes / 1024).toFixed(1)} KB</span>
            </div>
          </button>
        `).join('');

    list.querySelectorAll('.workspace-file').forEach(btn => {
      btn.addEventListener('click', () => openPreview(btn.dataset.url, btn.dataset.name));
    });
  }
  $("#workspaceMeta").innerText = `${state.workspaceFiles.length} files`;
}

async function openPreview(url, name) {
  $("#modalTitle").innerText = name;
  $("#workspaceModalBody").innerHTML = `<div style="text-align:center; padding: 40px;"><div class="spinner" style="margin: 0 auto;"></div></div>`;
  state.workspaceModalOpen = true;
  $("#workspaceModal").classList.add("open");

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load file");
    const text = await res.text();

    const isMd = name.endsWith('.md');
    const content = isMd && window.__markedReady ? marked.parse(text) : `<pre style="font-family: var(--mono); font-size: 13px; color: var(--text);">${text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>`;

    $("#workspaceModalBody").innerHTML = `<div class="msg-text" style="background: var(--bg-0); padding: 20px; border-radius: var(--radius-md); border: 1px solid var(--border-1); overflow-y:auto; max-height: 100%;">${content}</div>`;
    logEvent("info", `Viewed artifact: ${name}`, { action: "Preview", file: name, bytes: text.length });
  } catch (err) {
    $("#workspaceModalBody").innerHTML = `<div style="color: var(--danger); padding: 20px;">Could not load preview.</div>`;
  }
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
  let html = state.messages.map((msg, index) => {
    if (msg.role === "approval_status") return ''; // Skip internal status renders for clean UI
    if (msg.role === "tool") return ''; // Tool outputs stay in traces, not the chat transcript.

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
    html += state.pendingApprovals.map(ap => `
          <div class="approval-card">
            <div class="approval-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              Approval Required: ${ap.tool_name}
            </div>
            <p style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">${ap.reason || "The agent wants to execute this tool."}</p>
            <pre style="background:var(--bg-0); padding:10px; border-radius:6px; font-family:var(--mono); font-size:12px; overflow-x:auto;">${JSON.stringify(ap.parameters, null, 2)}</pre>
            <div class="approval-btns">
              <button class="btn-approve" onclick="submitApproval('${ap.tool_call_id}', true)"${disabled}>Approve</button>
              <button class="btn-deny" onclick="submitApproval('${ap.tool_call_id}', false)"${disabled}>Deny</button>
            </div>
          </div>
        `).join('');
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
    if (ev.type === "tool_call") {
      state.runningTools[ev.tool_call_id] = {
        tool_call_id: ev.tool_call_id,
        tool_name: ev.tool_name,
        started_at: Date.now(),
      };
    } else if (ev.type === "tool_result") {
      delete state.runningTools[ev.tool_call_id];
      const denied = String(ev.error || "").toLowerCase().includes("denied")
        || String(ev.error || "").toLowerCase().includes("declined")
        || String(ev.error || "").toLowerCase().includes("rejected");
      pushActivity(
        ev.success ? "Processing tool result" : (denied ? "Skipped denied tool" : "Tool failed"),
        ev.success ? "info" : (denied ? "approval" : "error"),
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
  else if (packet.type === "tool_call") {
    state.runningTools[packet.tool_call_id] = {
      tool_call_id: packet.tool_call_id,
      tool_name: packet.tool_name,
      started_at: Date.now(),
    };
  }
  else if (packet.type === "tool_result") {
    delete state.runningTools[packet.tool_call_id];
    const denied = String(packet.error || "").toLowerCase().includes("denied")
      || String(packet.error || "").toLowerCase().includes("declined")
      || String(packet.error || "").toLowerCase().includes("rejected");
    pushActivity(
      packet.success ? "Processing tool result" : (denied ? "Skipped denied tool" : "Tool failed"),
      packet.success ? "info" : (denied ? "approval" : "error"),
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
    clearActivity();
    $("#messagesContainer").innerHTML = "";
    renderAttachmentPreview();
    refreshWorkspace();
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
  $("#themeToggle").addEventListener("click", toggleTheme);
  $("#workspaceToggle").addEventListener("click", () => {
    state.showWorkspace = !state.showWorkspace;
    renderWorkspace();
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
  });
  $("#chatInput").addEventListener("keydown", (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(e); }
  });
  $("#newChatBtn").addEventListener("click", handleNewChat);
}

window.addEventListener("DOMContentLoaded", initApp);
