const $ = (sel) => document.querySelector(sel);

const state = {
  sessionId: null,
  agents: [],
  selectedAgent: null,
  messages: [],
  workspaceFiles: [],
  selectedWorkspacePath: null,
  workspacePreview: { status: "empty", file: null, kind: "empty", content: "", error: null },
  workspaceModalOpen: false,
  showWorkspace: true,
  sessionUsage: { tokens_input: 0, tokens_output: 0, tokens_cached: 0, llm_calls: 0, tool_calls: 0 },
  contextUsage: { used: null, max: null },
  events: [],
  pendingApprovals: [],
  activeTools: {},
  isRunning: false,
  processingApproval: null,
  pendingDecisions: {},
  error: null,
  showEvents: false,
  knownSkills: [],
  thinkingBuffer: {},
  thinkingOpen: {},
};

/* ── Theme toggle (light / dark) ── */
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
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
  btn.title = isDark ? "Switch to light mode" : "Switch to dark mode";
  btn.innerHTML = isDark
    ? `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
    : `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
}

initTheme();

/* ── marked: configured in index.html before this module loads ── */
function renderMarkdown(text) {
  if (!text) return "";
  if (typeof marked === "undefined" || !window.__markedReady) {
    console.warn("[markdown] marked not ready, falling back to plain text");
    return `<pre>${esc(text)}</pre>`;
  }
  try {
    return marked.parse(text);
  } catch (err) {
    console.error("[markdown] parse error:", err);
    return `<pre>${esc(text)}</pre>`;
  }
}

/* ── Copy buttons for code & tables ── */
function attachCopyButtons(root) {
  if (!root) return;

  // Code blocks
  root.querySelectorAll(".code-block").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.title = "Copy code";
    btn.innerHTML = copyIconSvg();
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const code = pre.querySelector("code")?.innerText || "";
      copyToClipboard(code, btn);
    });
    pre.appendChild(btn);
  });

  // Tables
  root.querySelectorAll(".table-wrap").forEach((wrap) => {
    if (wrap.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn copy-btn-table";
    btn.title = "Copy table";
    btn.innerHTML = copyIconSvg();
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const table = wrap.querySelector("table");
      const text = tableToTsv(table);
      copyToClipboard(text, btn);
    });
    wrap.appendChild(btn);
  });
}

function copyIconSvg() {
  return `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
}

function checkIconSvg() {
  return `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
}

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Fallback for non-secure contexts
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch {}
    document.body.removeChild(ta);
  }
  if (btn) {
    const original = btn.innerHTML;
    btn.classList.add("copied");
    btn.innerHTML = checkIconSvg();
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = original;
    }, 1400);
  }
}

function tableToTsv(table) {
  if (!table) return "";
  const rows = [...table.querySelectorAll("tr")];
  return rows.map((row) =>
    [...row.querySelectorAll("th, td")].map((c) => (c.innerText || "").trim()).join("\t")
  ).join("\n");
}

/* ── Utilities ── */
async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function parseSseChunk(buffer) {
  const parts = buffer.split("\n\n");
  return {
    packets: parts.slice(0, -1).map((p) => {
      const line = p.split("\n").find((l) => l.startsWith("data: "));
      return line ? JSON.parse(line.slice(6)) : null;
    }).filter(Boolean),
    remainder: parts[parts.length - 1],
  };
}

function esc(str) {
  const d = document.createElement("div");
  d.textContent = str || "";
  return d.innerHTML;
}

function scrollBottom(el) { if (el) el.scrollTop = el.scrollHeight; }

function normalizeRenderableMessages(messages) {
  return (messages || []).filter((msg) => {
    if (!msg || typeof msg !== "object") return false;
    if (msg.role === "compaction_status") return true;
    if (msg.role === "approval_status") return true;
    if (msg.role === "tool" || msg.role === "system" || msg.role === "stop") return false;
    if (msg.role === "assistant" && !(msg.content || "").trim() && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      return false;
    }
    return true;
  });
}

function recordApprovalDecision(approval, approved) {
  if (!approval) return;
  state.messages.push({
    role: "approval_status",
    tool_call_id: approval.tool_call_id,
    tool_name: approval.tool_name,
    parameters: approval.parameters || null,
    approved,
    created_at: new Date().toISOString(),
  });
}

function mergeWithApprovals(serverMsgs, approvalEntries) {
  if (approvalEntries.length === 0) return serverMsgs;
  const merged = [];
  let ai = 0;
  for (const msg of serverMsgs) {
    const msgTime = new Date(msg.created_at || 0);
    while (ai < approvalEntries.length && new Date(approvalEntries[ai].created_at) <= msgTime) {
      merged.push(approvalEntries[ai++]);
    }
    merged.push(msg);
  }
  while (ai < approvalEntries.length) merged.push(approvalEntries[ai++]);
  return merged;
}

function mergeUsage(usage) {
  if (!usage) return;
  state.sessionUsage.tokens_input += usage.tokens_input || 0;
  state.sessionUsage.tokens_output += usage.tokens_output || 0;
  state.sessionUsage.tokens_cached += usage.tokens_cached || 0;
  state.sessionUsage.llm_calls += usage.llm_calls || 0;
  state.sessionUsage.tool_calls += usage.tool_calls || 0;
}

function totalContextTokens(usage) {
  if (state.contextUsage.used !== null && state.contextUsage.used !== undefined) {
    return state.contextUsage.used;
  }
  return (usage.tokens_input || 0) + (usage.tokens_output || 0) + (usage.tokens_cached || 0);
}

function updateContextUsage(contextUsage) {
  if (!contextUsage) return;
  if (contextUsage.used !== undefined && contextUsage.used !== null) {
    state.contextUsage.used = contextUsage.used;
  }
  if (contextUsage.max !== undefined && contextUsage.max !== null) {
    state.contextUsage.max = contextUsage.max;
  }
}

function recordCompactionEvent(packet) {
  const event = packet.event || packet;
  if ((event.event_type || event.type) !== "compaction") return;
  updateContextUsage({
    used: event.recent_token_count,
    max: event.max_history_tokens,
  });
  state.messages.push({
    role: "compaction_status",
    event_id: packet.event_id || event.event_id || `${Date.now()}`,
    strategy: event.strategy || "compaction",
    old_message_count: event.old_message_count || 0,
    recent_message_count: event.recent_message_count || 0,
    old_token_count: event.old_token_count || 0,
    recent_token_count: event.recent_token_count || 0,
    total_token_count: event.total_token_count || 0,
    max_history_tokens: event.max_history_tokens || null,
    created_at: new Date().toISOString(),
  });
}

function fmtCount(value) {
  const number = Number(value || 0);
  if (number >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
  if (number >= 1000) return `${(number / 1000).toFixed(1)}k`;
  return String(number);
}

function fmtBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function renderWorkspaceFilesMarkup(limit = null) {
  const files = limit == null ? state.workspaceFiles : state.workspaceFiles.slice(0, limit);
  if (!files.length) return '<div class="workspace-file-empty">No files yet</div>';
  return files.map((file) => `
    <a class="workspace-file" href="${esc(file.url)}" target="_blank" rel="noreferrer">
      <span class="workspace-file-name">${esc(file.name)}</span>
      <span class="workspace-file-meta">${esc(file.path)} · ${esc(fmtBytes(file.size_bytes))}</span>
    </a>`).join("");
}

function renderWorkspaceSidebarMarkup() {
  if (!state.workspaceFiles.length) return '<div class="workspace-file-empty">No files yet</div>';
  return state.workspaceFiles.map((file) => `
    <button type="button" class="workspace-file workspace-file-select${file.path === state.selectedWorkspacePath ? " active" : ""}" data-path="${esc(file.path)}">
      <span class="workspace-file-name">${esc(file.name)}</span>
      <span class="workspace-file-meta">${esc(file.path)} · ${esc(fmtBytes(file.size_bytes))}</span>
    </button>`).join("");
}

function getWorkspaceFile(path) {
  return state.workspaceFiles.find((file) => file.path === path) || null;
}

function workspaceFileKind(file) {
  if (!file) return "empty";
  const name = (file.name || file.path || "").toLowerCase();
  const ext = name.includes(".") ? name.split(".").pop() : "";
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "image";
  if (ext === "pdf") return "pdf";
  if (["md", "markdown"].includes(ext)) return "markdown";
  if (["txt", "json", "py", "js", "ts", "tsx", "jsx", "html", "css", "yaml", "yml", "xml", "csv", "log"].includes(ext)) return "text";
  return "unsupported";
}

async function loadWorkspacePreview(path, { force = false } = {}) {
  const file = getWorkspaceFile(path);
  if (!file) {
    state.selectedWorkspacePath = null;
    state.workspacePreview = { status: "empty", file: null, kind: "empty", content: "", error: null };
    return;
  }
  if (!force && state.selectedWorkspacePath === path && state.workspacePreview.file?.path === path) return;
  state.selectedWorkspacePath = path;
  const kind = workspaceFileKind(file);
  state.workspacePreview = { status: "loading", file, kind, content: "", error: null };
  renderStatus();
  if (kind === "image" || kind === "pdf" || kind === "unsupported") {
    state.workspacePreview = { status: "ready", file, kind, content: "", error: null };
    return;
  }
  const previewToken = `${path}:${Date.now()}`;
  state.workspacePreview.token = previewToken;
  try {
    const response = await fetch(file.url);
    if (!response.ok) throw new Error(`Could not load preview (${response.status})`);
    const content = await response.text();
    if (state.workspacePreview.token !== previewToken) return;
    state.workspacePreview = { status: "ready", file, kind, content, error: null };
  } catch (err) {
    if (state.workspacePreview.token !== previewToken) return;
    state.workspacePreview = { status: "error", file, kind, content: "", error: String(err) };
  }
}

function renderWorkspacePreviewMarkup() {
  const preview = state.workspacePreview;
  const file = preview.file;
  if (!file) return '<div class="workspace-preview-empty">Select a file to preview it here.</div>';
  const header = `
    <div class="workspace-preview-head">
      <div>
        <div class="workspace-preview-title">${esc(file.name)}</div>
        <div class="workspace-preview-meta">${esc(file.path)} · ${esc(fmtBytes(file.size_bytes))}</div>
      </div>
      <a class="workspace-preview-open" href="${esc(file.url)}" target="_blank" rel="noreferrer">Open</a>
    </div>`;
  if (preview.status === "loading") return `${header}<div class="workspace-preview-empty"><span class="tool-run-spinner"></span><span>Loading preview…</span></div>`;
  if (preview.status === "error") return `${header}<div class="workspace-preview-empty">${esc(preview.error || "Could not render preview")}</div>`;
  if (preview.kind === "image") return `${header}<div class="workspace-preview-body"><img class="workspace-preview-image" src="${esc(file.url)}" alt="${esc(file.name)}" /></div>`;
  if (preview.kind === "pdf") return `${header}<div class="workspace-preview-body"><iframe class="workspace-preview-frame" src="${esc(file.url)}"></iframe></div>`;
  if (preview.kind === "markdown") return `${header}<div class="workspace-preview-body workspace-preview-rich msg-text">${renderMarkdown(preview.content || "")}</div>`;
  if (preview.kind === "text") return `${header}<pre class="workspace-preview-text">${esc(preview.content || "")}</pre>`;
  return `${header}<div class="workspace-preview-empty">Preview is not available for this file type in the browser.</div>`;
}

function renderWorkspaceModal() {
  const modal = $("#workspaceModal");
  const body = $("#workspaceModalBody");
  if (!modal || !body) return;
  modal.style.display = state.workspaceModalOpen ? "flex" : "none";
  body.innerHTML = renderWorkspacePreviewMarkup();
}

function closeWorkspaceModal() {
  state.workspaceModalOpen = false;
  renderWorkspaceModal();
}

function renderWorkspacePanel() {
  const panel = $("#workspacePanel");
  if (!panel) return;
  panel.classList.toggle("open", state.showWorkspace);
}

/* ── Extract @skill-name patterns ── */
function extractSkillNames(agents) {
  const skills = new Set();
  for (const a of agents) {
    if (!a.invocation_hint) continue;
    const matches = a.invocation_hint.matchAll(/@([a-z0-9][a-z0-9_-]*)/gi);
    for (const m of matches) skills.add(m[1]);
  }
  return [...skills].sort();
}

/* ── Render: Agents ── */
function renderAgents() {
  const container = $("#agentDropdown");
  if (!container) return;
  const selected = state.agents.find((a) => a.name === state.selectedAgent);
  const label = selected ? (selected.label || selected.display_name || selected.name) : "Select agent";
  container.innerHTML = `
    <button type="button" class="agent-dd-trigger" id="agentDdBtn">
      <span class="agent-dd-label">${esc(label)}</span>
      <span class="agent-dd-caret">▾</span>
    </button>
    <div class="agent-dd-menu" id="agentDdMenu" style="display:none">
      ${state.agents.map((a) => `
        <div class="agent-dd-item${a.name === state.selectedAgent ? " active" : ""}" data-name="${esc(a.name)}">
          <div class="agent-dd-item-label">${esc(a.label || a.display_name || a.name)}</div>
          <div class="agent-dd-item-desc">${esc(a.description)}</div>
        </div>`).join("")}
    </div>`;
  const btn = $("#agentDdBtn");
  const menu = $("#agentDdMenu");
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.style.display = menu.style.display === "none" ? "" : "none";
  });
  container.querySelectorAll(".agent-dd-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      state.selectedAgent = item.dataset.name;
      menu.style.display = "none";
      refreshAgentTools().then(() => renderAll());
    });
  });
  document.addEventListener("click", () => { menu.style.display = "none"; }, { once: true });
}

function renderHints() {
  const section = $("#hintSection");
  if (section) section.style.display = "none";
}

function renderInputHint() {
  const agent = state.agents.find((a) => a.name === state.selectedAgent);
  const input = $("#chatInput");
  if (input) input.placeholder = agent ? "Message..." : "Select an agent...";
}

function renderStatus() {
  const selected = state.agents.find((agent) => agent.name === state.selectedAgent) || null;
  const tokenMeta = $("#tokenMeta");
  const tokenFill = $("#tokenFill");
  const popTools = $("#popoverTools");
  const workspaceSidebarFiles = $("#workspaceSidebarFiles");
  const workspacePanelMeta = $("#workspacePanelMeta");
  const toolRunStatus = $("#toolRunStatus");
  const used = totalContextTokens(state.sessionUsage);
  const contextWindow = state.contextUsage.max || (selected && selected.context_window ? selected.context_window : null);
  const ratio = contextWindow ? Math.min(1, used / contextWindow) : 0;

  if (tokenMeta) {
    tokenMeta.textContent = contextWindow ? `${fmtCount(used)} / ${fmtCount(contextWindow)}` : `${fmtCount(used)} / -`;
    tokenMeta.title = state.contextUsage.used !== null
      ? `current context ${state.contextUsage.used || 0} · input ${state.sessionUsage.tokens_input || 0} · output ${state.sessionUsage.tokens_output || 0} · cached ${state.sessionUsage.tokens_cached || 0}`
      : `input ${state.sessionUsage.tokens_input || 0} · output ${state.sessionUsage.tokens_output || 0} · cached ${state.sessionUsage.tokens_cached || 0}`;
    tokenMeta.classList.toggle("warn", Boolean(contextWindow && ratio >= 0.85));
  }
  if (tokenFill) {
    if (!contextWindow && used === 0) { tokenFill.style.width = "0%"; tokenFill.classList.add("empty"); }
    else if (!contextWindow) { tokenFill.style.width = "0%"; tokenFill.classList.remove("empty"); }
    else { tokenFill.style.width = `${Math.max(2, Math.round(ratio * 100))}%`; tokenFill.classList.remove("empty"); }
    tokenFill.classList.toggle("warn", Boolean(contextWindow && ratio >= 0.85));
  }
  if (popTools) {
    const agent = state.agents.find((item) => item.name === state.selectedAgent);
    const tools = agent && Array.isArray(agent.tools) ? agent.tools : [];
    popTools.innerHTML = tools.length === 0
      ? '<span class="tool-chip tool-chip-empty">No tools</span>'
      : tools.map((t) => `<span class="tool-chip">${esc(t)}</span>`).join("");
  }
  if (workspaceSidebarFiles) {
    workspaceSidebarFiles.innerHTML = renderWorkspaceSidebarMarkup();
    workspaceSidebarFiles.querySelectorAll(".workspace-file-select").forEach((button) => {
      button.addEventListener("click", () => {
        loadWorkspacePreview(button.dataset.path, { force: true }).then(() => {
          state.workspaceModalOpen = true;
          renderStatus();
          renderWorkspaceModal();
        });
      });
    });
  }
  if (workspacePanelMeta) {
    const count = state.workspaceFiles.length;
    workspacePanelMeta.textContent = `${count} file${count === 1 ? "" : "s"}`;
  }
  if (toolRunStatus) {
    const activeTools = Object.values(state.activeTools);
    if (!activeTools.length) {
      toolRunStatus.style.display = "none";
      toolRunStatus.innerHTML = "";
    } else {
      const label = activeTools.length === 1 ? activeTools[0].tool_name : `${activeTools.length} tools running`;
      toolRunStatus.style.display = "";
      toolRunStatus.title = activeTools.map((item) => item.tool_name).join(", ");
      toolRunStatus.innerHTML = `<span class="tool-run-spinner"></span><span>${esc(label)}</span>`;
    }
  }
}

function renderToolsList() {}

function renderError() {
  const bar = $("#errorBar");
  if (state.error) { bar.style.display = ""; bar.textContent = "⚠ " + state.error; }
  else { bar.style.display = "none"; }
}

function updateSendBtn() {
  const input = $("#chatInput").value.trim();
  $("#sendBtn").disabled = !state.sessionId || !state.selectedAgent || state.isRunning || !input;
}

/* ── Message rendering (no-blink DOM diffing) ── */
function msgDomId(msg, index) {
  if (msg.role === "approval_status") return `as-${msg.tool_call_id}`;
  if (msg.role === "compaction_status") return `cs-${msg.event_id || index}`;
  return `msg-${index}`;
}

function buildApprovalStatusHtml(msg) {
  const paramsHtml = msg.parameters
    ? `<details class="approval-status-details"><summary>Parameters</summary><pre class="approval-params">${esc(JSON.stringify(msg.parameters, null, 2))}</pre></details>`
    : "";
  return `<div class="approval-status-row" data-msgid="${esc(msgDomId(msg, 0))}">
    <span class="approval-status-chip ${msg.approved ? "approved" : "rejected"}">${msg.approved ? "Approved" : "Denied"}</span>
    <span class="approval-status-label">${esc(msg.tool_name)}</span>
    ${paramsHtml}
  </div>`;
}

function buildCompactionStatusHtml(msg) {
  const before = msg.total_token_count || 0;
  const after = msg.recent_token_count || 0;
  const max = msg.max_history_tokens || null;
  const tokenLabel = max
    ? `${fmtCount(before)} -> ${fmtCount(after)} / ${fmtCount(max)}`
    : `${fmtCount(before)} -> ${fmtCount(after)}`;
  return `<div class="compaction-row" data-msgid="${esc(msgDomId(msg, 0))}">
    <span class="compaction-chip">Compaction</span>
    <span class="compaction-label">${esc(msg.strategy || "context")}</span>
    <span class="compaction-meta">${esc(tokenLabel)} tokens · kept ${esc(msg.recent_message_count)} of ${esc((msg.old_message_count || 0) + (msg.recent_message_count || 0))} messages</span>
  </div>`;
}

function buildMessageHtml(msg, index) {
  const isUser = msg.role === "user";
  const rowClass = isUser ? "msg-row user-row" : "msg-row bot-row";
  const bubbleClass = isUser ? "msg-bubble user-bubble" : "msg-bubble bot-bubble";
  const streaming = msg.streaming ? '<span class="streaming-badge"></span>' : "";
  const thinkingHtml = msg.thinking
    ? thinkingDetailsHtml(msg.thinking, false, msg.source)
    : "";
  return `<div class="${rowClass}" data-msgid="${esc(msgDomId(msg, index))}">
    <div class="${bubbleClass}">
      ${thinkingHtml}
      <div class="msg-text">${isUser ? esc(msg.content) : renderMarkdown(msg.content)}</div>
      ${streaming}
    </div>
  </div>`;
}

function thinkingDetailsHtml(content, open = true, agentName = "") {
  return `<details class="thinking-block"${open ? " open" : ""} data-thinking-owner="${esc(agentName)}"><summary class="thinking-summary">Thinking</summary><div class="thinking-content">${renderMarkdown(content)}</div></details>`;
}

function updateThinkingDetails(details, content) {
  const contentEl = details.querySelector(".thinking-content");
  const nextHtml = renderMarkdown(content);
  if (contentEl && contentEl.innerHTML !== nextHtml) contentEl.innerHTML = nextHtml;
}

function buildApprovalRowHtml(ap) {
  const decision = state.pendingDecisions[ap.tool_call_id];
  const decided = decision !== undefined;
  return `<div class="approval-row" data-apid="${esc(ap.tool_call_id)}">
    <div class="approval-card${decided ? (decision ? ' decided-approve' : ' decided-deny') : ''}">
      <div class="approval-topline">
        <span class="approval-pill">${decided ? (decision ? 'Will approve' : 'Will deny') : 'Approval needed'}</span>
        <div class="approval-btns">
          <button class="btn-approve${decided && decision ? ' selected' : ''}" data-tcid="${esc(ap.tool_call_id)}" ${state.processingApproval ? "disabled" : ""}>Approve</button>
          <button class="btn-reject${decided && !decision ? ' selected' : ''}" data-tcid="${esc(ap.tool_call_id)}" ${state.processingApproval ? "disabled" : ""}>Deny</button>
        </div>
      </div>
      <div class="approval-tool-row">
        <div class="approval-tool">${esc(ap.tool_name)}</div>
        ${ap.reason ? `<div class="approval-reason">${esc(ap.reason)}</div>` : ""}
      </div>
      <details class="approval-details">
        <summary>Parameters</summary>
        <pre class="approval-params">${esc(JSON.stringify(ap.parameters, null, 2))}</pre>
      </details>
    </div>
  </div>`;
}

let _renderedMsgIds = [];
let _renderedStreamingAgents = new Set();

/* "Calling model..." indicator — shows immediately after send,
   removed on first thinking_delta / assistant_delta / agent_complete */
function showCallingIndicator() {
  const container = $("#messagesContainer");
  if (!container) return;
  if (container.querySelector("#callingIndicator")) return;
  $("#emptyState").style.display = "none";
  container.style.display = "";
  container.insertAdjacentHTML("beforeend", `
    <div class="msg-row bot-row" id="callingIndicator">
      <div class="msg-bubble bot-bubble">
        <div class="calling-indicator">
          <span class="calling-dot"></span>
          <span class="calling-dot"></span>
          <span class="calling-dot"></span>
          <span class="calling-label">Calling model</span>
        </div>
      </div>
    </div>`);
  scrollBottom($("#chatScroll"));
}

function hideCallingIndicator() {
  const el = document.getElementById("callingIndicator");
  if (el) el.remove();
}

function followOpenThinking(container) {
  container.querySelectorAll(".thinking-block[open] .thinking-content").forEach((el) => {
    el.scrollTop = el.scrollHeight;
  });
}

function renderMessages() {
  const container = $("#messagesContainer");
  const empty = $("#emptyState");

  if (state.messages.length === 0 && state.pendingApprovals.length === 0) {
    empty.style.display = "";
    container.style.display = "none";
    container.innerHTML = "";
    _renderedMsgIds = [];
    _renderedStreamingAgents = new Set();
    return;
  }
  empty.style.display = "none";
  container.style.display = "";

  const desiredIds = state.messages.map((m, i) => msgDomId(m, i));

  for (const oldId of _renderedMsgIds) {
    if (!desiredIds.includes(oldId)) {
      const el = container.querySelector(`[data-msgid="${CSS.escape(oldId)}"]`);
      if (el) el.remove();
    }
  }

  let refNode = null;
  for (let i = 0; i < state.messages.length; i++) {
    const msg = state.messages[i];
    const id = msgDomId(msg, i);
    let el = container.querySelector(`[data-msgid="${CSS.escape(id)}"]`);

    if (!el) {
      const tmp = document.createElement("div");
      tmp.innerHTML = msg.role === "approval_status"
        ? buildApprovalStatusHtml(msg)
        : msg.role === "compaction_status"
          ? buildCompactionStatusHtml(msg)
          : buildMessageHtml(msg, i);
      el = tmp.firstElementChild;
      if (refNode) refNode.after(el);
      else container.prepend(el);
    } else {
      if (msg.streaming) {
        const textEl = el.querySelector(".msg-text");
        if (textEl) {
          const newHtml = renderMarkdown(msg.content);
          if (textEl.innerHTML !== newHtml) textEl.innerHTML = newHtml;
        }
        const bubble = el.querySelector(".msg-bubble");
        if (bubble) {
          let thinkingEl = el.querySelector(".thinking-block");
          if (msg.thinking) {
            if (!thinkingEl) {
              const shouldOpen = state.thinkingOpen[msg.source] !== false;
              bubble.insertAdjacentHTML("afterbegin", thinkingDetailsHtml(msg.thinking, shouldOpen, msg.source));
              thinkingEl = el.querySelector(".thinking-block");
            } else {
              updateThinkingDetails(thinkingEl, msg.thinking);
            }
          } else if (thinkingEl) {
            thinkingEl.remove();
          }
        }
        if (!el.querySelector(".streaming-badge")) {
          const bubble = el.querySelector(".msg-bubble");
          if (bubble) bubble.insertAdjacentHTML("beforeend", '<span class="streaming-badge"></span>');
        }
      } else {
        if (el.querySelector(".streaming-badge")) {
          const tmp = document.createElement("div");
          tmp.innerHTML = buildMessageHtml(msg, i);
          const newEl = tmp.firstElementChild;
          el.replaceWith(newEl);
          el = newEl;
        }
      }
    }
    refNode = el;
  }

  _renderedMsgIds = desiredIds;

  // Thinking indicators
  const thinkingAgentsNow = new Set();
  for (const [agentName, buf] of Object.entries(state.thinkingBuffer)) {
    if (!buf.is_final && buf.content) {
      const alreadyHasStreaming = state.messages.some((m) => m.streaming && m.source === agentName);
      if (!alreadyHasStreaming) thinkingAgentsNow.add(agentName);
    }
  }
  for (const agentName of thinkingAgentsNow) {
    const buf = state.thinkingBuffer[agentName];
    const existing = container.querySelector(`[data-thinking-agent="${CSS.escape(agentName)}"]`);
    if (existing) {
      const details = existing.querySelector(".thinking-block");
      if (details) updateThinkingDetails(details, buf.content);
      continue;
    }
    if (!_renderedStreamingAgents.has(agentName)) {
      const shouldOpen = state.thinkingOpen[agentName] !== false;
      container.insertAdjacentHTML("beforeend", `
        <div class="msg-row bot-row" data-thinking-agent="${esc(agentName)}">
          <div class="msg-bubble bot-bubble">
            ${thinkingDetailsHtml(buf.content, shouldOpen, agentName)}
          </div>
        </div>`);
      _renderedStreamingAgents.add(agentName);
    }
  }
  for (const agentName of _renderedStreamingAgents) {
    if (!thinkingAgentsNow.has(agentName)) {
      const el = container.querySelector(`[data-thinking-agent="${CSS.escape(agentName)}"]`);
      if (el) el.remove();
      _renderedStreamingAgents.delete(agentName);
    }
  }

  // Pending approvals
  container.querySelectorAll(".approval-row[data-apid]").forEach((el) => {
    if (!state.pendingApprovals.find((a) => a.tool_call_id === el.dataset.apid)) el.remove();
  });
  for (const ap of state.pendingApprovals) {
    let el = container.querySelector(`.approval-row[data-apid="${CSS.escape(ap.tool_call_id)}"]`);
    const newHtml = buildApprovalRowHtml(ap);
    if (!el) {
      container.insertAdjacentHTML("beforeend", newHtml);
    } else {
      const tmp = document.createElement("div");
      tmp.innerHTML = newHtml;
      el.replaceWith(tmp.firstElementChild);
    }
  }

  // Submit decisions button
  const existingSubmit = container.querySelector(".btn-submit-decisions");
  const allDecided = state.pendingApprovals.length > 0 &&
    state.pendingApprovals.every((ap) => state.pendingDecisions[ap.tool_call_id] !== undefined);
  if (allDecided && !state.processingApproval) {
    if (!existingSubmit) container.insertAdjacentHTML("beforeend",
      `<div class="approval-row"><button class="btn-submit-decisions" id="submitDecisions">Submit decisions</button></div>`);
  } else {
    if (existingSubmit) existingSubmit.closest(".approval-row")?.remove();
  }

  // Event listeners
  container.querySelectorAll(".btn-approve").forEach((b) =>
    b.addEventListener("click", () => { state.pendingDecisions[b.dataset.tcid] = true; renderMessages(); })
  );
  container.querySelectorAll(".btn-reject").forEach((b) =>
    b.addEventListener("click", () => { state.pendingDecisions[b.dataset.tcid] = false; renderMessages(); })
  );
  const submitBtn = $("#submitDecisions");
  if (submitBtn) submitBtn.addEventListener("click", submitAllDecisions);
  container.querySelectorAll(".thinking-block[data-thinking-owner]").forEach((details) => {
    if (details.dataset.boundThinkingToggle === "1") return;
    details.dataset.boundThinkingToggle = "1";
    details.addEventListener("toggle", () => {
      const owner = details.dataset.thinkingOwner;
      if (owner) state.thinkingOpen[owner] = details.open;
    });
  });

  attachCopyButtons(container);
  followOpenThinking(container);
  scrollBottom($("#chatScroll"));
}

/* ── Events panel ── */
const VISIBLE_EVENT_TYPES = new Set([
  "agent_event", "agent_selected", "agent_complete", "approval_required", "run_complete", "session_state", "error",
]);

function renderEvents() {
  const panel = $("#eventsPanel");
  const badge = $("#eventsBadge");
  const toggle = $("#eventsToggle");
  const scroll = $("#eventsScroll");
  const visible = state.events.filter((e) => VISIBLE_EVENT_TYPES.has(e.type));
  panel.classList.toggle("open", state.showEvents);
  if (toggle) {
    toggle.classList.toggle("is-on", state.showEvents);
    toggle.setAttribute("aria-pressed", state.showEvents ? "true" : "false");
  }
  if (visible.length > 0) { badge.style.display = ""; badge.textContent = visible.length; }
  else { badge.style.display = "none"; }
  if (!state.showEvents) {
    if (scroll) scroll.innerHTML = "";
    return;
  }
  if (visible.length === 0) { scroll.innerHTML = '<div class="events-empty">No events yet.</div>'; return; }
  scroll.innerHTML = visible.map((p, i) => {
    const label = p.type === "agent_event"
      ? `${p.agent_name || ""} · ${(p.event || {}).event_type || "event"}`
      : p.type.replace(/_/g, " ");
    return `
      <div class="evt-card" data-idx="${i}">
        <div class="evt-head">
          <div class="evt-type"><span class="evt-dot ${eventDotClass(p.type)}"></span> <span class="evt-label">${esc(label)}</span></div>
          <span class="evt-expand">+</span>
        </div>
        <pre class="evt-pre" style="display:none">${esc(JSON.stringify(p, null, 2))}</pre>
      </div>`;
  }).join("");
  scrollBottom(scroll);
  scroll.querySelectorAll(".evt-card").forEach((card) => {
    card.addEventListener("click", () => {
      const pre = card.querySelector(".evt-pre");
      pre.style.display = pre.style.display === "none" ? "" : "none";
    });
  });
}

function eventDotClass(type) {
  if (type === "agent_event") return "ev-tool";
  if (type === "approval_required") return "ev-approval";
  if (type === "agent_complete" || type === "run_complete") return "ev-complete";
  if (type === "session_state" || type === "agent_selected") return "ev-session";
  if (type === "error") return "ev-approval";
  return "ev-default";
}

/* ── Skill Dropdown ── */
let dropdownIndex = -1;

function showSkillDropdown(filter) {
  const dd = $("#skillDropdown");
  const matches = state.knownSkills.filter((s) => s.startsWith(filter));
  if (matches.length === 0) { dd.style.display = "none"; dropdownIndex = -1; return; }
  dropdownIndex = 0;
  dd.style.display = "";
  dd.innerHTML = matches.map((s, i) =>
    `<div class="skill-option${i === 0 ? " active" : ""}" data-skill="${esc(s)}">@${esc(s)}</div>`
  ).join("");
  dd.querySelectorAll(".skill-option").forEach((opt) => {
    opt.addEventListener("mousedown", (e) => { e.preventDefault(); insertSkill(opt.dataset.skill); });
  });
}

function hideSkillDropdown() {
  $("#skillDropdown").style.display = "none";
  dropdownIndex = -1;
}

function insertSkill(skill) {
  const ta = $("#chatInput");
  const val = ta.value;
  const pos = ta.selectionStart;
  const before = val.slice(0, pos);
  const atIdx = before.lastIndexOf("@");
  if (atIdx === -1) return;
  const after = val.slice(pos);
  ta.value = before.slice(0, atIdx) + "@" + skill + " " + after;
  ta.selectionStart = ta.selectionEnd = atIdx + skill.length + 2;
  ta.focus();
  hideSkillDropdown();
  updateSendBtn();
}

function handleInputForDropdown() {
  const ta = $("#chatInput");
  const before = ta.value.slice(0, ta.selectionStart);
  const atMatch = before.match(/@([a-z0-9_-]*)$/i);
  if (atMatch) showSkillDropdown(atMatch[1]);
  else hideSkillDropdown();
}

/* ── Agent tools ── */
async function refreshAgentTools() {
  if (!state.selectedAgent) return;
  try {
    const detail = await fetchJson(`/api/agents/${encodeURIComponent(state.selectedAgent)}`);
    const idx = state.agents.findIndex((a) => a.name === state.selectedAgent);
    if (idx >= 0) {
      state.agents[idx].tools = detail.tools || [];
      state.agents[idx].model_name = detail.model_name || state.agents[idx].model_name;
      state.agents[idx].context_window = detail.context_window || state.agents[idx].context_window;
    }
  } catch (err) { console.warn("Could not fetch agent detail:", err); }
}

async function refreshWorkspaceFiles() {
  const previousTop = state.workspaceFiles[0]?.path || null;
  const previousSelected = state.selectedWorkspacePath;
  try {
    state.workspaceFiles = await fetchJson("/api/workspace/files");
  } catch (err) { console.warn("Could not fetch workspace files:", err); state.workspaceFiles = []; }
  const currentTop = state.workspaceFiles[0]?.path || null;
  const selectedStillExists = previousSelected && state.workspaceFiles.some((f) => f.path === previousSelected);
  const nextPath = currentTop && (currentTop !== previousTop || !selectedStillExists)
    ? currentTop : (selectedStillExists ? previousSelected : currentTop);
  if (nextPath) await loadWorkspacePreview(nextPath, { force: true });
  else state.workspacePreview = { status: "empty", file: null, kind: "empty", content: "", error: null };
}

function syncToolActivity(packet) {
  if (packet.type === "agent_complete" || packet.type === "run_complete" || packet.type === "error") {
    state.activeTools = {};
    return;
  }
  if (packet.type !== "agent_event" || !packet.event) return;
  const event = packet.event;
  if (event.event_type === "tool_call" && event.tool_call_id)
    state.activeTools[event.tool_call_id] = { tool_name: event.tool_name || "tool", parameters: event.parameters || {} };
  if ((event.event_type === "tool_call_response" || event.event_type === "tool_result") && event.tool_call_id)
    delete state.activeTools[event.tool_call_id];
}

function syncRuntimeActivity(packet) {
  if (packet.type === "agent_event" && packet.event?.event_type === "compaction") {
    recordCompactionEvent(packet);
  }
  if (packet.type === "session_state") {
    updateContextUsage(packet.context_usage);
  }
}

function renderAll() {
  renderAgents(); renderHints(); renderInputHint();
  renderWorkspacePanel(); renderStatus(); renderWorkspaceModal(); renderToolsList();
  renderError(); renderMessages(); renderEvents(); updateSendBtn(); renderThemeBtn();
}

/* ── SSE Stream Reader ── */
async function readSseStream(response, handler) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.remainder;
    for (const packet of parsed.packets) {
      state.events.push(packet);
      syncToolActivity(packet);
      syncRuntimeActivity(packet);
      handler(packet);
    }
  }
}

/* ── Send Message ── */
async function handleSend(e) {
  e.preventDefault();
  const input = $("#chatInput");
  const msg = input.value.trim();
  if (!msg || !state.sessionId || !state.selectedAgent || state.isRunning) return;

  state.messages.push({ role: "user", source: "user", content: msg.replace(/\s+$/, ""), created_at: new Date().toISOString() });
  input.value = ""; input.style.height = "auto";
  state.events = []; state.pendingApprovals = [];
  state.activeTools = {};
  state.isRunning = true; state.error = null;
  hideSkillDropdown();
  renderAll();
  showCallingIndicator();   // show "Calling model..." immediately

  const streamingDrafts = {};
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: msg, agent_names: [state.selectedAgent] }),
    });
    if (!res.ok || !res.body) throw new Error(await res.text());

    await readSseStream(res, (packet) => {
      // Hide calling indicator ONLY when actual content arrives or run finishes.
      // agent_event (model_call, tool_call, etc) is NOT enough — model may still be working silently.
      const hasContent =
        (packet.type === "thinking_delta" && packet.content) ||
        (packet.type === "assistant_delta" && packet.content) ||
        packet.type === "agent_complete" ||
        packet.type === "approval_required";
      if (hasContent) hideCallingIndicator();

      if (packet.type === "thinking_delta") {
        state.thinkingBuffer[packet.agent_name] = { content: packet.content, is_final: packet.is_final };
        renderMessages(); return;
      }
      if (packet.type === "assistant_delta") {
        streamingDrafts[packet.agent_name] = {
          role: "assistant", source: packet.agent_name, content: packet.content,
          thinking: state.thinkingBuffer[packet.agent_name]?.content || null,
          created_at: new Date().toISOString(), streaming: true,
        };
        state.messages = state.messages.filter((m) => !m.streaming);
        state.messages.push(...Object.values(streamingDrafts));
        renderMessages(); renderStatus(); return;
      }
      if (packet.type === "agent_complete" && packet.assistant_message) {
        mergeUsage(packet.usage);
        const thinking = state.thinkingBuffer[packet.agent_name]?.content || null;
        delete streamingDrafts[packet.agent_name];
        delete state.thinkingBuffer[packet.agent_name];
        state.messages = state.messages.filter((m) => !(m.streaming && m.source === packet.agent_name));
        const normalized = normalizeRenderableMessages([packet.assistant_message]);
        if (normalized[0]) { normalized[0].thinking = thinking; state.messages.push(normalized[0]); }
      }
      if (packet.type === "session_state") {
        const preservedEntries = state.messages.filter((m) => m.role === "approval_status" || m.role === "compaction_status");
        state.messages = mergeWithApprovals(normalizeRenderableMessages(packet.messages), preservedEntries);
        state.pendingApprovals = packet.pending_approvals || [];
        updateContextUsage(packet.context_usage);
        refreshWorkspaceFiles().then(() => renderStatus());
      }
      if (packet.type === "approval_required") {
        state.pendingApprovals = packet.pending_approvals || [];
        state.activeTools = {}; state.isRunning = false;
      }
      if (packet.type === "error") { state.activeTools = {}; state.error = packet.message; }
      renderMessages(); renderEvents(); renderStatus(); renderError();
    });
  } catch (err) { state.error = String(err); }
  finally { hideCallingIndicator(); state.isRunning = false; renderAll(); }
}

/* ── Submit All Approval Decisions ── */
async function submitAllDecisions() {
  if (!state.sessionId || state.processingApproval) return;
  const decisions = state.pendingApprovals.map((ap) => ({
    tool_call_id: ap.tool_call_id,
    approved: Boolean(state.pendingDecisions[ap.tool_call_id]),
    reason: state.pendingDecisions[ap.tool_call_id] ? "Approved" : "Rejected",
  }));
  if (decisions.length === 0) return;

  for (const ap of state.pendingApprovals) recordApprovalDecision(ap, Boolean(state.pendingDecisions[ap.tool_call_id]));
  const previousPending = state.pendingApprovals.slice();
  const previousApprovalMsgs = state.messages.filter((m) => m.role === "approval_status");
  state.activeTools = {}; state.processingApproval = true;
  state.pendingApprovals = []; state.pendingDecisions = {}; state.error = null;
  renderAll();
  showCallingIndicator();

  try {
    const res = await fetch("/api/chat/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, decisions }),
    });
    if (!res.ok || !res.body) throw new Error(await res.text());

    await readSseStream(res, (packet) => {
      const hasContent =
        (packet.type === "thinking_delta" && packet.content) ||
        (packet.type === "assistant_delta" && packet.content) ||
        packet.type === "agent_complete" ||
        packet.type === "approval_required";
      if (hasContent) hideCallingIndicator();

      if (packet.type === "thinking_delta") {
        state.thinkingBuffer[packet.agent_name] = { content: packet.content, is_final: packet.is_final };
        renderMessages(); return;
      }
      if (packet.type === "assistant_delta") {
        state.messages = state.messages.filter((m) => !(m.streaming && m.source === packet.agent_name));
        state.messages.push({
          role: "assistant", source: packet.agent_name, content: packet.content,
          thinking: state.thinkingBuffer[packet.agent_name]?.content || null,
          created_at: new Date().toISOString(), streaming: true,
        });
        renderMessages(); renderStatus(); return;
      }
      if (packet.type === "agent_complete" && packet.assistant_message) {
        mergeUsage(packet.usage);
        const thinking = state.thinkingBuffer[packet.agent_name]?.content || null;
        delete state.thinkingBuffer[packet.agent_name];
        state.messages = state.messages.filter((m) => !(m.streaming && m.source === packet.agent_name));
        const normalized = normalizeRenderableMessages([packet.assistant_message]);
        if (normalized[0]) { normalized[0].thinking = thinking; state.messages.push(normalized[0]); }
      }
      if (packet.type === "approval_required") {
        state.pendingApprovals = packet.pending_approvals || [];
        state.pendingDecisions = {}; state.activeTools = {}; state.processingApproval = null;
      }
      if (packet.type === "session_state") {
        const preservedEntries = state.messages.filter((m) => m.role === "approval_status" || m.role === "compaction_status");
        state.messages = mergeWithApprovals(normalizeRenderableMessages(packet.messages), preservedEntries);
        state.pendingApprovals = packet.pending_approvals || [];
        updateContextUsage(packet.context_usage);
        refreshWorkspaceFiles().then(() => renderStatus());
      }
      if (packet.type === "error") { state.activeTools = {}; state.error = packet.message; }
      renderMessages(); renderEvents(); renderStatus(); renderError();
    });
  } catch (err) {
    state.pendingApprovals = previousPending;
    const currentApprovals = state.messages.filter((m) => m.role === "approval_status");
    if (currentApprovals.length === 0 && previousApprovalMsgs.length > 0) state.messages.push(...previousApprovalMsgs);
    state.error = String(err);
  }
  finally { hideCallingIndicator(); state.processingApproval = null; state.pendingDecisions = {}; renderAll(); }
}

/* ── New Session ── */
async function handleNewSession() {
  try {
    const session = await fetchJson("/api/sessions", { method: "POST" });
    state.sessionId = session.session_id;
    state.messages = []; state.events = []; state.pendingApprovals = []; state.activeTools = {};
    state.selectedWorkspacePath = null;
    state.sessionUsage = { tokens_input: 0, tokens_output: 0, tokens_cached: 0, llm_calls: 0, tool_calls: 0 };
    state.contextUsage = { used: null, max: null };
    state.thinkingOpen = {};
    state.error = null;
    _renderedMsgIds = [];
    _renderedStreamingAgents = new Set();
    const container = $("#messagesContainer");
    if (container) container.innerHTML = "";
    await refreshWorkspaceFiles();
    renderAll();
  } catch (err) { state.error = String(err); renderAll(); }
}

/* ── Bootstrap ── */
async function init() {
  try {
    const [agentList, session] = await Promise.all([
      fetchJson("/api/agents"),
      fetchJson("/api/sessions", { method: "POST" }),
    ]);
    state.agents = agentList;
    const def = agentList.find((a) => a.default_selected);
    state.selectedAgent = def ? def.name : (agentList[0] ? agentList[0].name : null);
    await refreshAgentTools();
    await refreshWorkspaceFiles();
    state.sessionId = session.session_id;
    state.pendingApprovals = session.pending_approvals || [];
    updateContextUsage(session.context_usage);
    state.messages = normalizeRenderableMessages(session.messages || []);
    state.knownSkills = extractSkillNames(agentList);
    renderAll();
  } catch (err) { state.error = String(err); renderAll(); }
}

/* ── Event Bindings ── */
$("#newChatBtn").addEventListener("click", handleNewSession);
$("#contextToggle").addEventListener("click", (e) => {
  e.stopPropagation();
  const pop = $("#contextPopover");
  pop.style.display = pop.style.display === "none" ? "" : "none";
});
document.addEventListener("click", (e) => {
  const pop = $("#contextPopover");
  if (pop && !pop.contains(e.target) && e.target !== $("#contextToggle")) pop.style.display = "none";
});
$("#workspaceModal").addEventListener("click", (e) => { if (e.target === $("#workspaceModal")) closeWorkspaceModal(); });
$("#workspaceModalClose").addEventListener("click", closeWorkspaceModal);
$("#chatForm").addEventListener("submit", handleSend);
$("#chatInput").addEventListener("input", (e) => {
  updateSendBtn();
  handleInputForDropdown();
  const el = e.target;
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
});
$("#chatInput").addEventListener("keydown", (e) => {
  const dd = $("#skillDropdown");
  if (dd.style.display !== "none") {
    const opts = dd.querySelectorAll(".skill-option");
    if (e.key === "ArrowDown") { e.preventDefault(); dropdownIndex = Math.min(dropdownIndex + 1, opts.length - 1); opts.forEach((o, i) => o.classList.toggle("active", i === dropdownIndex)); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); dropdownIndex = Math.max(dropdownIndex - 1, 0); opts.forEach((o, i) => o.classList.toggle("active", i === dropdownIndex)); return; }
    if (e.key === "Tab" || e.key === "Enter") {
      if (dropdownIndex >= 0 && opts[dropdownIndex]) { e.preventDefault(); insertSkill(opts[dropdownIndex].dataset.skill); return; }
    }
    if (e.key === "Escape") { e.preventDefault(); hideSkillDropdown(); return; }
  }
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(e); }
});
$("#chatInput").addEventListener("blur", () => setTimeout(hideSkillDropdown, 150));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.workspaceModalOpen) closeWorkspaceModal(); });
$("#eventsToggle").addEventListener("click", () => { state.showEvents = !state.showEvents; renderEvents(); });
$("#eventsClose").addEventListener("click", () => { state.showEvents = false; renderEvents(); });
$("#workspaceToggle").addEventListener("click", () => { state.showWorkspace = !state.showWorkspace; renderWorkspacePanel(); });
$("#workspaceClose").addEventListener("click", () => { state.showWorkspace = false; renderWorkspacePanel(); });
$("#workspaceRefreshBtn").addEventListener("click", async () => { await refreshWorkspaceFiles(); renderStatus(); });
$("#themeToggle").addEventListener("click", toggleTheme);

init();
