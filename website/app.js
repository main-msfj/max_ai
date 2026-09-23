const sidebar = document.querySelector('#sidebar');
const menuToggle = document.querySelector('#menu-toggle');
const themeToggle = document.querySelector('#theme-toggle');

menuToggle?.addEventListener('click', () => sidebar.classList.toggle('open'));
document.querySelectorAll('.nav-link').forEach((link) => {
  link.addEventListener('click', () => sidebar.classList.remove('open'));
});

themeToggle?.addEventListener('click', () => {
  document.body.classList.toggle('dark');
  themeToggle.textContent = document.body.classList.contains('dark') ? '☾' : '☼';
});

document.querySelectorAll('.copy-button').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy || '');
      const label = button.textContent;
      button.textContent = 'Copied ✓';
      setTimeout(() => { button.textContent = label; }, 1400);
    } catch {
      button.textContent = 'Select the code';
    }
  });
});

const sections = [...document.querySelectorAll('.content-section[id]')];
const links = [...document.querySelectorAll('.nav-link')];
const observer = new IntersectionObserver((entries) => {
  const visible = entries
    .filter((entry) => entry.isIntersecting)
    .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  const activeLink = links.find((link) => {
    const target = new URL(link.href, window.location.href);
    return target.pathname === window.location.pathname
      && target.hash === '#' + visible.target.id;
  });
  if (!activeLink) return;
  links.forEach((link) => {
    link.classList.toggle('active', link === activeLink);
  });
}, { rootMargin: '-18% 0px -65% 0px', threshold: [0, .2, .5] });
sections.forEach((section) => observer.observe(section));
