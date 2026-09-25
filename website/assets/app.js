(() => {
  const root = document.documentElement;
  const sidebar = document.querySelector('#sidebar');
  const menuToggle = document.querySelector('#menu-toggle');
  const themeToggle = document.querySelector('#theme-toggle');
  const backdrop = document.querySelector('.nav-backdrop');
  const mobile = matchMedia('(max-width: 760px)');
  const systemTheme = matchMedia('(prefers-color-scheme: dark)');
  const normalizePath = (path) => path === '/index.html' ? '/' : path;

  function readTheme() {
    try { return localStorage.getItem('max-ai-theme'); } catch { return null; }
  }

  function setTheme(theme, persist = false) {
    const dark = theme === 'dark';
    root.dataset.theme = dark ? 'dark' : 'light';
    themeToggle.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    themeToggle.setAttribute('aria-pressed', String(dark));
    if (persist) {
      try { localStorage.setItem('max-ai-theme', theme); } catch { /* Storage is optional. */ }
    }
  }

  setTheme(readTheme() || (systemTheme.matches ? 'dark' : 'light'));
  themeToggle.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark', true));
  systemTheme.addEventListener('change', (event) => {
    if (!readTheme()) setTheme(event.matches ? 'dark' : 'light');
  });

  function setMenu(open, restoreFocus = false) {
    sidebar.classList.toggle('open', open);
    backdrop.classList.toggle('visible', open);
    document.body.classList.toggle('nav-open', open);
    sidebar.inert = mobile.matches && !open;
    menuToggle.setAttribute('aria-expanded', String(open));
    menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    if (open) (sidebar.querySelector('.active') || sidebar.querySelector('a')).focus();
    else if (restoreFocus) menuToggle.focus();
  }

  setMenu(false);
  menuToggle.addEventListener('click', () => setMenu(!sidebar.classList.contains('open')));
  backdrop.addEventListener('click', () => setMenu(false, true));
  mobile.addEventListener('change', () => setMenu(false));
  const rootDisclosures = '.nav-section, .nav-basecomponents, .nav-capability-root, .nav-examples, .nav-types, .nav-events, .nav-cli';
  sidebar.querySelectorAll('details > summary').forEach((summary) => {
    summary.addEventListener('click', () => {
      const disclosure = summary.parentElement;
      const rootDisclosure = disclosure.matches(rootDisclosures)
        ? disclosure
        : disclosure.closest(rootDisclosures);
      if (rootDisclosure) {
        sidebar.querySelectorAll(rootDisclosures).forEach((other) => {
          if (other !== rootDisclosure) other.open = false;
        });
      }
      const siblings = disclosure.parentElement.querySelectorAll(':scope > details');
      siblings.forEach((other) => {
        if (other !== disclosure) other.open = false;
      });
    });
  });
  sidebar.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => setMenu(false));
  });

  document.querySelectorAll('.copy-button').forEach((button) => {
    button.addEventListener('click', async () => {
      const code = button.closest('.code-card, .layer-code-panel')?.querySelector('pre code, pre');
      if (!code) return;
      const label = button.querySelector('span') || button;
      let copied = false;
      try {
        await navigator.clipboard.writeText(code.textContent);
        copied = true;
      } catch {
        // Leave the exact displayed example selected when clipboard access is blocked.
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        try { copied = document.execCommand('copy'); } catch { /* Manual copy remains available. */ }
        if (copied) selection.removeAllRanges();
      }
      label.textContent = copied ? 'Copied!' : 'Press Ctrl/Cmd+C';
      button.setAttribute('aria-label', copied ? 'Code copied' : 'Code selected; copy with your keyboard');
      setTimeout(() => {
        label.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy code');
      }, 1800);
    });
  });

  const pythonKeywords = new Set([
    'and', 'as', 'async', 'await', 'break', 'class', 'continue', 'def', 'del',
    'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global',
    'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or',
    'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield',
  ]);
  const pythonToken = /#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|@[A-Za-z_]\w*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b/g;
  document.querySelectorAll('pre code').forEach((code) => {
    const source = code.textContent;
    const pieces = [];
    let cursor = 0;
    for (const match of source.matchAll(pythonToken)) {
      const token = match[0];
      const start = match.index;
      if (start > cursor) pieces.push(document.createTextNode(source.slice(cursor, start)));
      let className = '';
      if (token.startsWith('#')) className = 'comment';
      else if (/^["']/.test(token)) className = 'str';
      else if (token.startsWith('@')) className = 'decor';
      else if (/^\d/.test(token)) className = 'num';
      else if (pythonKeywords.has(token)) className = 'kw';
      else if (/^[A-Z]/.test(token)) className = 'type';
      else if (/^\s*\(/.test(source.slice(start + token.length))) className = 'fn';
      if (className) {
        const span = document.createElement('span');
        span.className = className;
        span.textContent = token;
        pieces.push(span);
      } else {
        pieces.push(document.createTextNode(token));
      }
      cursor = start + token.length;
    }
    if (cursor < source.length) pieces.push(document.createTextNode(source.slice(cursor)));
    code.replaceChildren(...pieces);
  });

  function openAnchoredDetails() {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id ? document.getElementById(id) : null;
    if (target instanceof HTMLDetailsElement) target.open = true;
  }
  addEventListener('hashchange', openAnchoredDetails);
  openAnchoredDetails();

  document.querySelectorAll('[role="tablist"]').forEach((tablist) => {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    function activate(tab, focus = false) {
      tabs.forEach((candidate) => {
        const selected = candidate === tab;
        candidate.setAttribute('aria-selected', String(selected));
        candidate.tabIndex = selected ? 0 : -1;
        document.getElementById(candidate.getAttribute('aria-controls')).hidden = !selected;
      });
      if (focus) tab.focus();
      updateActiveSection();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', (event) => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next === undefined) return;
        event.preventDefault();
        activate(tabs[next], true);
      });
    });
  });

  const sections = [...document.querySelectorAll('main section[id]')];
  const tocLinks = [...document.querySelectorAll('.toc-link')];
  const navLinks = [...sidebar.querySelectorAll('.nav-link')];
  function updateActiveSection() {
    const offset = parseInt(getComputedStyle(root).getPropertyValue('--header-height'), 10) + 90;
    let active = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= offset) active = section;
    }
    if (!active) return;
    tocLinks.forEach((link) => {
      const selected = link.hash === '#' + active.id;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    const matched = navLinks.find((link) =>
      normalizePath(link.pathname) === normalizePath(location.pathname) && link.hash === '#' + active.id);
    if (matched) {
      navLinks.forEach((link) => {
        link.classList.toggle('active', link === matched);
        if (link === matched) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
  }

  let scrollPending = false;
  addEventListener('scroll', () => {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(() => {
      updateActiveSection();
      scrollPending = false;
    });
  }, { passive: true });
  addEventListener('resize', updateActiveSection);
  addEventListener('load', updateActiveSection);
  updateActiveSection();

  const dialog = document.querySelector('#search-dialog');
  const input = document.querySelector('#docs-search');
  const results = document.querySelector('#search-results');
  const status = document.querySelector('#search-status');
  let indexPromise;
  let documents = [];
  let searchVersion = 0;
  const normalized = (text) => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

  function message(text) {
    results.replaceChildren();
    const paragraph = document.createElement('p');
    paragraph.className = 'search-message';
    paragraph.textContent = text;
    results.append(paragraph);
    status.textContent = text;
  }

  function renderResults() {
    const query = normalized(input.value.trim());
    const terms = query.split(/\s+/).filter(Boolean);
    const matches = documents.map((doc) => {
      const title = normalized(doc.title);
      const page = normalized(doc.page);
      const body = normalized(doc.text);
      if (!terms.length) return { doc, score: doc.featured ? 1 : 0 };
      if (!terms.every((term) => (title + ' ' + page + ' ' + body).includes(term))) return { doc, score: 0 };
      const score = terms.reduce((sum, term) => sum + (title.includes(term) ? 12 : 0) + (page.includes(term) ? 4 : 0) + (body.includes(term) ? 1 : 0), 0);
      return { doc, score: score + (title === query ? 30 : 0) };
    }).filter((item) => item.score > 0).sort((a, b) => b.score - a.score).slice(0, 10);
    if (!matches.length) {
      message('No results. Try “tools”, “MCP”, “workspace”, or “output”.');
      return;
    }
    results.replaceChildren();
    for (const { doc } of matches) {
      const link = document.createElement('a');
      link.className = 'search-result';
      link.href = doc.url;
      const page = document.createElement('small');
      page.textContent = doc.page;
      const title = document.createElement('strong');
      title.textContent = doc.title;
      const description = document.createElement('p');
      description.textContent = doc.summary;
      link.append(page, title, description);
      link.addEventListener('click', () => dialog.close());
      results.append(link);
    }
    status.textContent = terms.length ? matches.length + ' results for ' + input.value : 'Suggested documentation pages';
    results.scrollTop = 0;
  }

  async function openSearch() {
    if (sidebar.classList.contains('open')) setMenu(false);
    dialog.showModal();
    input.value = '';
    input.focus();
    message('Loading documentation…');
    const version = ++searchVersion;
    try {
      indexPromise ||= fetch('/data/search-index.json').then((response) => {
        if (!response.ok) throw new Error('Search index unavailable');
        return response.json();
      });
      documents = await indexPromise;
      if (version === searchVersion && dialog.open) renderResults();
    } catch {
      indexPromise = null;
      if (dialog.open) message('Search is unavailable. Use the navigation to browse the documentation.');
    }
  }

  document.querySelectorAll('.search-trigger').forEach((button) => button.addEventListener('click', openSearch));
  document.querySelector('#search-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  input.addEventListener('input', () => { if (documents.length) renderResults(); });
  dialog.addEventListener('keydown', (event) => {
    const links = [...results.querySelectorAll('a')];
    if (!links.length) return;
    const current = links.indexOf(document.activeElement);
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = event.key === 'ArrowDown'
        ? (current + 1) % links.length
        : (current <= 0 ? links.length - 1 : current - 1);
      links[next].focus();
    }
    if (event.key === 'Enter' && document.activeElement === input) {
      event.preventDefault();
      links[0].click();
    }
  });
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      if (dialog.open) dialog.close();
      else openSearch();
    }
    if (!sidebar.classList.contains('open')) return;
    if (event.key === 'Escape') setMenu(false, true);
    if (event.key === 'Tab') {
      const focusable = [menuToggle, ...sidebar.querySelectorAll('a')];
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  const shortcut = document.querySelector('.search-trigger kbd');
  if (shortcut && !/Mac|iPhone|iPad/.test(navigator.platform)) shortcut.textContent = 'Ctrl K';

})();
