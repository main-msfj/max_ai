const state = {
  busy: false,
  pendingApprovals: [],
  decisions: new Map(),
  assistantBubble: null,
  approvalNode: null,
};

const el = {
  messages: document.querySelector("#messages"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  status: document.querySelector("#status-pill"),
  clear: document.querySelector("#clear-button"),
  agentSubtitle: document.querySelector("#agent-subtitle"),
  agentName: document.querySelector("#agent-name"),
  agentModel: document.querySelector("#agent-model"),
  approvalsSection: document.querySelector("#approvals-section"),
  approvalCount: document.querySelector("#approval-count"),
  approvalsList: document.querySelector("#approvals-list"),
  resume: document.querySelector("#resume-button"),
  activity: document.querySelector("#activity-list"),
  clearActivity: document.querySelector("#clear-activity"),
};

boot();

function boot() {
  loadInfo();
  el.composer.addEventListener("submit", onSubmit);
  el.clear.addEventListener("click", clearSession);
  el.resume.addEventListener("click", resumeWithApprovals);
  el.clearActivity.addEventListener("click", () => {
    el.activity.innerHTML = "";
  });
  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.composer.requestSubmit();
    }
  });
  el.input.addEventListener("input", autoSizeInput);
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    el.agentName.textContent = info.name || "-";
    el.agentModel.textContent = info.model || "-";
    el.agentSubtitle.textContent = info.description || "Agent WebUI";
  } catch (error) {
    addActivity("error", "Info", error.message);
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const message = el.input.value.trim();
  if (!message || state.busy || state.pendingApprovals.length) return;

  removeEmptyState();
  addMessage("user", "You", message);
  el.input.value = "";
  autoSizeInput();
  state.assistantBubble = addMessage("assistant", "MaxAI", "");

  await streamPost("/api/chat", { message });
}

async function resumeWithApprovals() {
  if (state.busy || !state.pendingApprovals.length) return;

  const decisions = state.pendingApprovals.map((item) => ({
    tool_call_id: item.tool_call_id,
    approved: state.decisions.get(item.tool_call_id) === true,
  }));

  markApprovalGroupResolved();
  state.assistantBubble = addMessage("assistant", "MaxAI", "");
  await streamPost("/api/approve", { decisions });
}

async function clearSession() {
  if (state.busy) return;
  await fetch("/api/clear", { method: "POST" });
  state.pendingApprovals = [];
  state.decisions.clear();
  state.assistantBubble = null;
  state.approvalNode = null;
  el.messages.innerHTML = "";
  el.activity.innerHTML = "";
  clearApprovals();
  setStatus("Idle");
  addEmptyState();
}

async function streamPost(url, body) {
  state.busy = true;
  updateControls();
  setStatus("Running", "running");

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        handleSseBlock(part);
      }
    }

    if (buffer.trim()) handleSseBlock(buffer);
  } catch (error) {
    handleEvent({ type: "error", message: error.message });
  } finally {
    state.busy = false;
    if (!state.pendingApprovals.length) setStatus("Idle");
    updateControls();
    trimEmptyAssistantBubble();
  }
}

function handleSseBlock(block) {
  const lines = block.split("\n");
  for (const line of lines) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (!payload) continue;
    try {
      handleEvent(JSON.parse(payload));
    } catch (error) {
      addActivity("error", "Invalid stream event", error.message);
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "status":
      setStatus(event.status === "resuming" ? "Resuming" : "Running", "running");
      break;
    case "token":
      if (!event.is_final) appendAssistantText(event.text || "");
      break;
    case "assistant_text":
      appendAssistantText(event.text || "");
      break;
    case "tool_call":
      addActivity(
        "tool call",
        event.tool_name,
        formatParams(event.parameters)
      );
      break;
    case "tool_result":
      addActivity(
        event.success ? "tool result" : "tool failed",
        event.tool_call_id,
        event.success ? preview(event.result) : event.error
      );
      break;
    case "approval_item":
      addActivity(
        "approval",
        event.item.tool_name,
        formatParams(event.item.parameters)
      );
      break;
    case "compaction":
      addActivity(
        "compaction",
        event.strategy || "Context",
        `${event.old_message_count} old messages, ${event.old_token_count} tokens moved out; ${event.recent_message_count} recent messages kept`
      );
      break;
    case "reasoning":
      break;
    case "done":
      if (event.pending_approvals && event.pending_approvals.length) {
        showApprovals(event.pending_approvals);
        setStatus("Waiting", "waiting");
      } else {
        clearApprovals();
        setStatus("Idle");
      }
      break;
    case "error":
      addActivity("error", event.error_type || "Error", event.message);
      setStatus("Error", "waiting");
      break;
    case "event":
      addActivity("event", event.event_type, "");
      break;
    default:
      addActivity("event", event.type || "unknown", "");
  }
}

function appendAssistantText(text) {
  if (!text) return;
  if (!state.assistantBubble) {
    state.assistantBubble = addMessage("assistant", "MaxAI", "");
  }
  const raw = (state.assistantBubble.dataset.raw || "") + text;
  state.assistantBubble.dataset.raw = raw;
  state.assistantBubble.classList.add("markdown-body");
  state.assistantBubble.innerHTML = renderMarkdown(raw);
  scrollMessages();
}

function addMessage(role, label, text) {
  removeEmptyState();
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = label;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.dataset.raw = text;
    bubble.classList.add("markdown-body");
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  wrapper.append(header, bubble);
  el.messages.append(wrapper);
  scrollMessages();
  return bubble;
}

function trimEmptyAssistantBubble() {
  const bubble = state.assistantBubble;
  if (!bubble) return;
  if (bubble.textContent.trim()) return;
  const wrapper = bubble.closest(".message");
  if (wrapper) wrapper.remove();
  state.assistantBubble = null;
}

function showApprovals(items) {
  state.pendingApprovals = items;
  state.decisions.clear();
  el.approvalsSection.hidden = true;
  el.approvalCount.textContent = String(items.length);
  addInlineApprovals(items);
  updateControls();
}

function addInlineApprovals(items) {
  removeEmptyState();

  const wrapper = document.createElement("div");
  wrapper.className = "message assistant approval-message";

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = "MaxAI";

  const panel = document.createElement("div");
  panel.className = "approval-chat-panel";

  const title = document.createElement("div");
  title.className = "approval-chat-title";
  title.textContent =
    items.length === 1
      ? "Approval needed before I run this tool"
      : `${items.length} approvals needed before I continue`;

  const copy = document.createElement("div");
  copy.className = "approval-chat-copy";
  copy.textContent =
    "Choose approve or reject for each call. I will resume automatically.";

  const list = document.createElement("div");
  list.className = "approval-chat-list";

  panel.append(title, copy, list);
  wrapper.append(header, panel);
  el.messages.append(wrapper);
  state.approvalNode = wrapper;

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "approval-card";
    card.dataset.toolCallId = item.tool_call_id;

    const title = document.createElement("div");
    title.className = "approval-title";
    title.textContent = item.tool_name;

    const params = document.createElement("div");
    params.className = "approval-params";
    params.textContent = formatParams(item.parameters);

    const reason = document.createElement("div");
    reason.className = "approval-reason";
    reason.textContent = item.reason || "Approval required before execution.";

    const actions = document.createElement("div");
    actions.className = "approval-actions";

    const approve = document.createElement("button");
    approve.className = "approve-button";
    approve.type = "button";
    approve.textContent = "Approve";
    approve.addEventListener("click", () => setDecision(item.tool_call_id, true));

    const reject = document.createElement("button");
    reject.className = "reject-button";
    reject.type = "button";
    reject.textContent = "Reject";
    reject.addEventListener("click", () => setDecision(item.tool_call_id, false));

    actions.append(approve, reject);
    card.append(title, params, reason, actions);
    list.append(card);
  }

  scrollMessages();
}

function setDecision(toolCallId, approved) {
  if (state.busy || !state.pendingApprovals.length) return;
  state.decisions.set(toolCallId, approved);
  updateInlineApprovalCard(toolCallId, approved);
  updateControls();

  if (state.decisions.size === state.pendingApprovals.length) {
    window.setTimeout(() => {
      resumeWithApprovals();
    }, 180);
  }
}

function clearApprovals() {
  state.pendingApprovals = [];
  state.decisions.clear();
  el.approvalsSection.hidden = true;
  el.approvalsList.innerHTML = "";
  el.approvalCount.textContent = "0";
  state.approvalNode = null;
}

function updateInlineApprovalCard(toolCallId, approved) {
  if (!state.approvalNode) return;
  const card = state.approvalNode.querySelector(
    `[data-tool-call-id="${cssEscape(toolCallId)}"]`
  );
  if (!card) return;

  const approve = card.querySelector(".approve-button");
  const reject = card.querySelector(".reject-button");
  approve.classList.toggle("selected", approved);
  reject.classList.toggle("selected", !approved);
  card.classList.add("decided");
}

function markApprovalGroupResolved() {
  if (state.approvalNode) {
    state.approvalNode.classList.add("resolved");
    for (const button of state.approvalNode.querySelectorAll("button")) {
      button.disabled = true;
    }
    const copy = state.approvalNode.querySelector(".approval-chat-copy");
    if (copy) copy.textContent = "Decision received. Resuming the agent...";
  }
  state.pendingApprovals = [];
  state.decisions.clear();
  el.approvalsSection.hidden = true;
  el.approvalsList.innerHTML = "";
  el.approvalCount.textContent = "0";
}

function addActivity(kind, title, detail) {
  const item = document.createElement("div");
  item.className = kind === "error" ? "activity-item error" : "activity-item";

  const kindEl = document.createElement("span");
  kindEl.className = "kind";
  kindEl.textContent = kind;

  const body = document.createElement("div");
  body.textContent = detail ? `${title}: ${detail}` : title;

  item.append(kindEl, body);
  el.activity.append(item);
  el.activity.scrollTop = el.activity.scrollHeight;
}

function setStatus(label, modifier) {
  el.status.textContent = label;
  el.status.className = "status";
  if (modifier) el.status.classList.add(modifier);
}

function updateControls() {
  const waiting = state.pendingApprovals.length > 0;
  el.input.disabled = state.busy || waiting;
  el.send.disabled = state.busy || waiting;
  el.clear.disabled = state.busy;
  el.resume.disabled =
    state.busy ||
    !waiting ||
    state.decisions.size !== state.pendingApprovals.length;
}

function formatParams(params) {
  if (!params || Object.keys(params).length === 0) return "(no parameters)";
  return JSON.stringify(params, null, 2);
}

function preview(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 159)}...` : text;
}

function renderMarkdown(text) {
  if (!text) return "";

  const segments = [];
  const codeRegex = /```([\w.+-]*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push(renderInlineMarkdown(text.slice(lastIndex, match.index)));
    }

    const language = match[1] ? escapeHtml(match[1]) : "code";
    const code = escapeHtml(match[2].replace(/\n$/, ""));
    segments.push(
      `<div class="code-block"><div class="code-header">${language}</div><pre><code>${code}</code></pre></div>`
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    segments.push(renderInlineMarkdown(text.slice(lastIndex)));
  }

  return segments.join("");
}

function renderInlineMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const withBreaks = part.replace(/\n/g, "<br>");
      const withInlineCode = withBreaks.replace(
        /`([^`]+)`/g,
        "<code>$1</code>"
      );
      const withBold = withInlineCode.replace(
        /\*\*([^*]+)\*\*/g,
        "<strong>$1</strong>"
      );
      return `<p>${withBold}</p>`;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }
  return String(value).replace(/"/g, '\\"');
}

function autoSizeInput() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 140)}px`;
}

function scrollMessages() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

function removeEmptyState() {
  const empty = el.messages.querySelector(".empty-state");
  if (empty) empty.remove();
}

function addEmptyState() {
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.innerHTML = `
    <div class="empty-title">Start a session</div>
    <div class="empty-copy">
      Send a message to test this agent, stream output, and approve tool calls.
    </div>
  `;
  el.messages.append(empty);
}
