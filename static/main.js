// Auto-scroll training logs after each HTMX swap (including SSE swaps)
document.addEventListener('htmx:afterSwap', () => {
  const pre = document.querySelector('#training-logs');
  if (pre) pre.scrollTop = pre.scrollHeight;

  // Re-apply current filter to tools table after swap
  const filter = document.getElementById('tool-filter');
  if (filter && filter.value) {
    applyToolFilter(filter.value);
  }
});

// Copy-to-clipboard for plan code
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.copy-btn');
  if (!btn) return;
  const sel = btn.getAttribute('data-copy-target');
  const el = sel ? document.querySelector(sel) : null;
  if (!el) return;
  const text = el.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => (btn.textContent = original), 1200);
  }).catch(() => {
    alert('Copy failed');
  });
});

// Tool filter
function applyToolFilter(query) {
  const q = (query || '').toLowerCase();
  const rows = document.querySelectorAll('#tools-table tbody tr');
  rows.forEach((tr) => {
    const tds = tr.querySelectorAll('td');
    const txt = Array.from(tds).map(td => td.textContent.toLowerCase()).join(' ');
    tr.style.display = txt.includes(q) ? '' : 'none';
  });
}
document.addEventListener('input', (e) => {
  if (e.target && e.target.id === 'tool-filter') {
    applyToolFilter(e.target.value);
  }
});