// The little the page does on its own: Enter sends, the chat follows the
// answer, the / and @ menus, the command panel, the Files and Trajectory
// pane, the turn's timer and the copy buttons.
(() => {
  const $ = (id) => document.getElementById(id);
  const sid = () => document.querySelector(".shell").dataset.sid;
  const box = () => document.querySelector(".composer textarea");
  const store = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
  const stored = (key) => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
  let follow = true;  // stay at the bottom unless the user scrolled up to read

  document.addEventListener("scroll", (e) => {
    const s = $("scroller");
    if (e.target === s) follow = s.scrollHeight - s.scrollTop - s.clientHeight < 80;
  }, true);

  // Every swap the server sends may grow the chat: keep the end in view.
  const settle = () => {
    const s = $("scroller");
    if (s && follow) s.scrollTop = s.scrollHeight;
    document.body.classList.toggle("busy", !!document.querySelector("#status .spin"));
  };
  document.addEventListener("htmx:oobAfterSwap", settle);

  // ---------- the side pane: Files or Trajectory ----------
  const openSide = (tab) => {
    window.maxaiTab = tab || "";
    $("side").hidden = !tab;
    document.body.classList.toggle("side-open", !!tab);
    for (const b of document.querySelectorAll("[data-tab]")) b.classList.toggle("active", b.dataset.tab === tab);
    for (const b of document.querySelectorAll("[data-body]")) b.hidden = b.dataset.body !== tab;
    for (const b of document.querySelectorAll("[data-open]")) b.classList.toggle("on", b.dataset.open === tab);
    store("maxai.side", tab || "");
    if (tab === "files") htmx.trigger($("tree"), "refresh");
    if (tab === "trace") htmx.ajax("GET", $("trace").getAttribute("hx-get"), { target: "#trace", swap: "outerHTML" });
  };
  const toggleSide = (tab) => openSide(window.maxaiTab === tab ? "" : tab);

  // Folders stay open when the tree is drawn again.
  const openDirs = new Set();
  const openRows = new Set();  // the same for the rows of the Trajectory
  document.addEventListener("toggle", (e) => {
    const el = e.target;
    if (el.matches?.("details.dir")) el.open ? openDirs.add(el.dataset.path) : openDirs.delete(el.dataset.path);
    if (el.matches?.(".ti details")) el.open ? openRows.add(el.dataset.id) : openRows.delete(el.dataset.id);
  }, true);
  document.addEventListener("htmx:afterSettle", () => {
    for (const row of document.querySelectorAll("#trace .ti details")) if (openRows.has(row.dataset.id)) row.open = true;
  });
  document.addEventListener("htmx:afterSwap", (e) => {
    if (e.target.id === "tree") {
      for (const dir of e.target.querySelectorAll("details.dir")) dir.open = openDirs.has(dir.dataset.path);
    }
    if (e.target.id === "file-view") e.target.closest(".files-tab").classList.add("viewing");
    if (e.target.id === "panel") { $("panel").hidden = false; }
  });

  // ---------- the composer ----------
  const grow = (b) => { b.style.height = "auto"; b.style.height = Math.min(b.scrollHeight, 240) + "px"; };
  const insert = (text) => {
    const b = box();
    const at = b.selectionStart ?? b.value.length;
    const before = b.value.slice(0, at), after = b.value.slice(b.selectionEnd ?? at);
    const pad = before && !/\s$/.test(before) && text.startsWith("@") ? " " : "";
    b.value = before + pad + text + after;
    b.focus();
    b.selectionStart = b.selectionEnd = (before + pad + text).length;
    grow(b);
    b.dispatchEvent(new Event("input", { bubbles: true }));
  };

  // ---------- the / and @ menus ----------
  const menu = { items: [], index: 0, kind: "", start: 0 };
  let commands = null;  // the / entries, fetched once
  let files = { at: 0, list: [] };  // the @ entries, fetched again after 5s

  const getCommands = async () => commands ??= await (await fetch(`/s/${sid()}/menu`)).json();
  const getFiles = async () => {
    if (Date.now() - files.at > 5000) {
      files = { at: Date.now(), list: await (await fetch(`/s/${sid()}/files?format=json`)).json() };
    }
    return files.list;
  };
  const closeMenu = () => { $("menu").hidden = true; menu.items = []; };
  const drawMenu = () => {
    const el = $("menu");
    if (!menu.items.length) return closeMenu();
    el.innerHTML = "";
    if (menu.kind === "@") {
      const head = document.createElement("div");
      head.className = "menu-group";
      head.textContent = "Workspace files";
      el.append(head);
    }
    menu.items.forEach((item, i) => {
      const row = document.createElement("div");
      row.className = "menu-item" + (i === menu.index ? " on" : "");
      const name = document.createElement("b");
      name.textContent = item.name;
      const text = document.createElement("span");
      text.textContent = item.description || "";
      row.append(name, text);
      if (item.kind === "skill") { const tag = document.createElement("i"); tag.textContent = "skill"; row.append(tag); }
      row.addEventListener("mousedown", (e) => { e.preventDefault(); menu.index = i; pick(); });
      el.append(row);
    });
    el.hidden = false;
    el.querySelector(".on")?.scrollIntoView({ block: "nearest" });
  };

  const updateMenu = async () => {
    const b = box();
    const before = b.value.slice(0, b.selectionStart);
    // 1. A command: "/" at the start, one word.
    if (/^\/\S*$/.test(b.value) && !b.value.includes("\n")) {
      const q = b.value.toLowerCase();
      const all = await getCommands();
      Object.assign(menu, { kind: "/", start: 0, index: 0, keep: false, items: all.filter((c) => c.name.startsWith(q)) });
      return drawMenu();
    }
    // 2. A file: the word being written starts with "@".
    const mention = before.match(/(?:^|\s)@([\w./-]*)$/);
    if (mention) {
      const q = mention[1].toLowerCase();
      const all = await getFiles();
      const found = all.filter((p) => p.toLowerCase().includes(q))
        .sort((a, b) => (a.toLowerCase().startsWith(q) ? 0 : 1) - (b.toLowerCase().startsWith(q) ? 0 : 1) || a.length - b.length)
        .slice(0, 50).map((p) => ({ name: p, description: "" }));
      // A path written whole needs no menu: Enter sends.
      if (found.length === 1 && found[0].name.toLowerCase() === q) return closeMenu();
      if (!found.length) found.push({ name: all.length ? "no file matches" : "the workspace is empty", description: "", none: true });
      Object.assign(menu, { kind: "@", keep: false, start: before.length - mention[1].length - 1, index: 0, items: found });
      return drawMenu();
    }
    closeMenu();
  };

  const pick = () => {
    const item = menu.items[menu.index];
    if (!item || item.none) return closeMenu();
    const b = box();
    closeMenu();
    if (menu.kind === "@") {
      const rest = b.value.slice(b.selectionStart);
      b.value = b.value.slice(0, menu.start) + "@" + item.name + " " + rest;
      b.selectionStart = b.selectionEnd = menu.start + item.name.length + 2;
      b.focus();
      return;
    }
    if (item.kind === "skill") {  // a skill waits for what it should do
      b.value = item.name + " " + (menu.keep ? b.value : "");
      b.focus();
      return;
    }
    if (!menu.keep) { b.value = ""; grow(b); }
    run(item.name);
  };

  // The Commands and Files buttons: open their menu, or close it if it is open.
  const toggleMenu = (trigger) => {
    const b = box();
    if (!$("menu").hidden && menu.kind === trigger) {
      closeMenu();
      // The trigger the button wrote, still alone, goes too.
      if (b.value.endsWith(trigger) && (b.value.length === 1 || /\s/.test(b.value.at(-2)))) b.value = b.value.slice(0, -1);
      b.focus();
      return;
    }
    if (trigger === "/" && b.value.trim() && !b.value.startsWith("/")) {
      // Text already written stays: the menu runs a command without touching it.
      getCommands().then((all) => {
        Object.assign(menu, { kind: "/", start: 0, index: 0, keep: true, items: all });
        drawMenu();
        b.focus();
      });
      return;
    }
    if (trigger === "/" && b.value.startsWith("/")) { b.focus(); updateMenu(); return; }
    insert(trigger);
  };

  // ---------- commands ----------
  const showPanel = (name) => {
    const panel = $("panel");
    panel.innerHTML = `<div class="panel-head"><b>${name.slice(1)}</b></div><div class="panel-body"><div class="loading"><span class="shimmer">Loading…</span></div></div>`;
    panel.hidden = false;
    htmx.ajax("GET", `/s/${sid()}/command?name=${encodeURIComponent(name)}`, { target: "#panel", swap: "innerHTML" });
  };
  const closePanel = () => { $("panel").hidden = true; $("panel").innerHTML = ""; };
  const page = {
    "/files": () => openSide("files"),
    "/trace": () => openSide("trace"),
    "/new": () => (location.href = "/new"),
    "/clear": () => (location.href = "/new"),
    "/thinking": () => {
      const hide = document.body.classList.toggle("hide-thinking");
      store("maxai.thinking", hide ? "hidden" : "");
    },
  };
  // A command of the page or of a panel; a skill or anything else is a message.
  const run = (text) => {
    const name = text.trim().split(/\s+/)[0].toLowerCase();
    if (page[name]) { page[name](); return true; }
    const known = (commands || []).find((c) => c.name === name);
    if (known?.kind === "skill") return false;
    if (known?.kind === "panel" || !commands || /^\/[a-z]+$/.test(name)) { showPanel(name); return true; }
    return false;
  };

  // The form is checked before htmx sends it: a command never reaches the chat.
  document.addEventListener("submit", (e) => {
    const b = e.target.querySelector?.(".composer textarea");
    if (!b) return;
    const text = b.value.trim();
    if (!text) { e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (text.startsWith("/") && run(text)) {
      e.preventDefault(); e.stopImmediatePropagation();
      b.value = ""; grow(b);
    }
    follow = true;
  }, true);

  document.addEventListener("keydown", (e) => {
    const b = e.target;
    const inBox = b.matches?.(".composer textarea");
    // 1. The menu takes the arrows, tab, enter and esc.
    if (inBox && !$("menu").hidden && menu.items.length) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        menu.index = (menu.index + (e.key === "ArrowDown" ? 1 : -1) + menu.items.length) % menu.items.length;
        return drawMenu();
      }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) { e.preventDefault(); return pick(); }
      if (e.key === "Escape") { e.preventDefault(); return closeMenu(); }
    }
    // 2. Enter sends, Shift+Enter is a new line.
    if (inBox && e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (b.value.trim()) b.form.requestSubmit();
      return;
    }
    // 3. Esc closes the panel, else stops the turn.
    if (e.key === "Escape") {
      if (!$("panel").hidden) return closePanel();
      if (document.body.classList.contains("busy")) htmx.ajax("POST", `/s/${sid()}/stop`, { swap: "none" });
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "b") { e.preventDefault(); toggleSide("files"); }
    if ((e.ctrlKey || e.metaKey) && e.key === ".") { e.preventDefault(); toggleSide("trace"); }
  });

  document.addEventListener("input", (e) => {
    if (e.target.matches?.(".composer textarea")) { grow(e.target); updateMenu(); }
  });
  // Leaving the box closes the menu, unless the focus went to a button of the composer.
  document.addEventListener("focusout", (e) => {
    if (!e.target.matches?.(".composer textarea")) return;
    if (e.relatedTarget?.closest?.(".composer")) return;
    setTimeout(() => { if (document.activeElement !== box()) closeMenu(); }, 150);
  });

  // The turn's timer.
  setInterval(() => {
    for (const el of document.querySelectorAll("[data-since]")) {
      const since = Number(el.dataset.since);
      if (since) el.textContent = Math.max(0, Math.floor((Date.now() - since) / 1000)) + "s";
    }
  }, 500);

  document.addEventListener("click", (e) => {
    const on = (selector) => e.target.closest?.(selector);
    if (on("[data-open]")) toggleSide(on("[data-open]").dataset.open);
    if (on("[data-tab]")) openSide(on("[data-tab]").dataset.tab);
    if (on("#side-close")) openSide("");
    if (on("#files-refresh")) { files.at = 0; htmx.trigger($("tree"), "refresh"); }
    if (on("[data-close-panel]")) closePanel();
    if (on("[data-insert]")) toggleMenu(on("[data-insert]").dataset.insert);
    if (on("[data-command]")) showPanel(on("[data-command]").dataset.command);
    if (on("[data-file-back]")) on(".files-tab").classList.remove("viewing");
    if (on("[data-mention]")) insert("@" + on("[data-mention]").dataset.mention + " ");
    if (on(".file")) {
      for (const f of document.querySelectorAll(".file.selected")) f.classList.remove("selected");
      on(".file").classList.add("selected");
    }

    // Copy a code block, without its line numbers.
    const copy = on(".copy");
    if (copy) {
      const code = copy.closest(".codeblock").querySelector("code").cloneNode(true);
      for (const n of code.querySelectorAll(".ln")) n.remove();
      navigator.clipboard?.writeText(code.textContent).then(() => {
        copy.textContent = "Copied";
        setTimeout(() => (copy.textContent = "Copy"), 1200);
      });
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    settle();
    if (stored("maxai.thinking") === "hidden") document.body.classList.add("hide-thinking");
    const side = stored("maxai.side");
    if (side === "files" || side === "trace") openSide(side);
    getCommands().catch(() => {});
    fetch(`/s/${sid()}/command?name=/permissions`).catch(() => {});  // so the shield opens at once
  });
})();
