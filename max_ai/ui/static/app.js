/* ============================================================
Frontend Logic - Adapted for serve.py (FastAPI Backend)
============================================================ */
const $ = (sel) => document.querySelector(sel);

const state = {
  sessionId: null,
  agents: [],
  selectedAgent: null,
  messages: [],
  events: [],
  workspaceFiles: [],
  pendingApprovals: [],
  isRunning: false,
  showWorkspace: true,
  showEvents: false,
  workspaceModalOpen: false,
  thinkingPanels: {}
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function compactText(value, max = 180) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function fmtCount(value) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toLocaleString() : "0";
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
  container.classList.toggle("is-streaming", state.isRunning);

  if (state.messages.length === 0 && state.pendingApprovals.length === 0) {
    empty.style.display = "flex"; container.style.display = "none";
    return;
  }

  empty.style.display = "none"; container.style.display = "block";

  let html = state.messages.map((msg, index) => {
    if (msg.role === "approval_status") return ''; // Skip internal status renders for clean UI

    const isUser = msg.role === 'user';
    const bubbleCls = isUser ? 'msg-bubble user-bubble' : 'msg-bubble bot-bubble';
    const rowCls = isUser ? 'msg-row user-row' : 'msg-row bot-row';
    const content = isUser ? escapeHtml(msg.content).replace(/\n/g, "<br>") : (window.__markedReady ? marked.parse(msg.content || "") : escapeHtml(msg.content));

    let thinkingHtml = '';
    if (!isUser && msg.thinking) {
      const thinking = window.__markedReady ? marked.parse(msg.thinking) : escapeHtml(msg.thinking);
      const thinkingKey = msg.id || msg.created_at || `${msg.role}-${index}`;
      const shouldOpen = state.thinkingPanels[thinkingKey] ?? Boolean(msg.streaming);
      const open = shouldOpen ? " open" : "";
      thinkingHtml = `
        <details class="thinking-block" data-thinking-key="${escapeHtml(thinkingKey)}"${open}>
          <summary>
            <span class="thinking-title">Thinking process</span>
            <small>${fmtCount(String(msg.thinking).length)} chars</small>
          </summary>
          <div class="thinking-content">${thinking}</div>
        </details>
      `;
    }

    return `<div class="${rowCls}"><div class="${bubbleCls}">${thinkingHtml}<div class="msg-text">${content}</div></div></div>`;
  }).join('');

  // Pending Approvals
  if (state.pendingApprovals.length > 0) {
    html += state.pendingApprovals.map(ap => `
          <div class="approval-card">
            <div class="approval-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              Approval Required: ${ap.tool_name}
            </div>
            <p style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">${ap.reason || "The agent wants to execute this tool."}</p>
            <pre style="background:var(--bg-0); padding:10px; border-radius:6px; font-family:var(--mono); font-size:12px; overflow-x:auto;">${JSON.stringify(ap.parameters, null, 2)}</pre>
            <div class="approval-btns">
              <button class="btn-approve" onclick="submitApproval('${ap.tool_call_id}', true)">Approve</button>
              <button class="btn-deny" onclick="submitApproval('${ap.tool_call_id}', false)">Deny</button>
            </div>
          </div>
        `).join('');
  }

  container.innerHTML = html;
  container.querySelectorAll(".thinking-block").forEach((details) => {
    details.addEventListener("toggle", () => {
      const key = details.dataset.thinkingKey;
      if (key) state.thinkingPanels[key] = details.open;
    });
  });
  const scroll = $("#chatScroll");
  scroll.scrollTop = scroll.scrollHeight;
}

function renderStatusArea() {
  const area = $("#statusArea");
  const input = $("#chatInput");
  if (input) {
    input.readOnly = state.isRunning;
    input.classList.toggle("is-running", state.isRunning);
  }
  if (state.isRunning) {
    area.innerHTML = `<span class="status-dot"></span> <span>Agent is thinking...</span>`;
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
  } else if (packet.type === "error") {
    logEvent("error", "System Error", { message: packet.message });
  }

  // Handle UI State
  if (packet.type === "thinking_delta") {
    let lastMsg = state.messages[state.messages.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant" || !lastMsg.streaming) {
      state.messages.push({ role: "assistant", streaming: true, content: "", thinking: packet.content });
    } else {
      lastMsg.thinking = packet.content;
    }
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
    if (packet.assistant_message) {
      // Replace streaming msg with final msg
      state.messages = state.messages.filter(m => !m.streaming);
      state.messages.push(packet.assistant_message);
    }
  }
  else if (packet.type === "session_state") {
    state.messages = packet.messages;
    state.pendingApprovals = packet.pending_approvals || [];
    refreshWorkspace();
  }
  else if (packet.type === "approval_required") {
    state.pendingApprovals = packet.pending_approvals || [];
    state.isRunning = false;
  }

  renderMessages();
  renderStatusArea();
}

// --- Actions ---
async function handleSend(e) {
  e.preventDefault();
  const input = $("#chatInput");
  const msg = input.value.trim();
  if (!msg || state.isRunning || !state.sessionId) return;

  state.messages.push({ role: "user", content: msg });
  input.value = "";
  input.blur();
  input.style.height = 'auto';
  state.isRunning = true;
  $("#sendBtn").disabled = true;

  renderMessages();
  renderStatusArea();
  logEvent("info", "User Prompt", { input_length: msg.length });

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: msg, agent_names: [state.selectedAgent] })
    });
    if (!res.ok) throw new Error(await res.text());
    await readSseStream(res);
  } catch (err) {
    logEvent("error", "Request Failed", { error: String(err) });
  } finally {
    state.isRunning = false;
    renderStatusArea();
    updateSendBtn();
  }
}

window.submitApproval = async function (toolCallId, approved) {
  if (state.isRunning) return;
  state.isRunning = true;
  state.pendingApprovals = [];
  renderMessages();
  renderStatusArea();

  logEvent("info", `Approval Decision: ${approved ? 'Approve' : 'Deny'}`, { tool_call_id: toolCallId });

  try {
    const res = await fetch("/api/chat/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        agent_name: state.selectedAgent,
        decisions: [{ tool_call_id: toolCallId, approved, reason: approved ? "Approved by user" : "Denied by user" }]
      })
    });
    if (!res.ok) throw new Error("Approval failed");
    await readSseStream(res);
  } catch (err) {
    logEvent("error", "Approval Request Failed", { error: String(err) });
  } finally {
    state.isRunning = false;
    renderStatusArea();
  }
}

async function handleNewChat() {
  try {
    const session = await fetch("/api/sessions", { method: "POST" }).then(r => r.json());
    state.sessionId = session.session_id;
    state.messages = [];
    state.events = [];
    state.pendingApprovals = [];
    $("#messagesContainer").innerHTML = "";
    refreshWorkspace();
    renderMessages();
    renderEvents();
    logEvent("info", "New Session Initialized", { session_id: state.sessionId });
  } catch (err) {
    logEvent("error", "Failed to start session", { error: String(err) });
  }
}

function updateSendBtn() {
  const text = $("#chatInput").value.trim();
  $("#sendBtn").disabled = text.length === 0 || state.isRunning || !state.sessionId;
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
    }

    // Create Session
    await handleNewChat();
  } catch (err) {
    console.error("Initialization error:", err);
    $("#activeAgentLabel").innerText = "API Disconnected";
  }

  // Listeners
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
