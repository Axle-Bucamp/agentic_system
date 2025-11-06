const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  try { return JSON.parse(text); } catch (_) { return text; }
}

async function refreshHealth() {
  try {
    const res = await api('/health');
    el('health').textContent = `API Health: ${JSON.stringify(res)}`;
  } catch (e) {
    el('health').textContent = `API Health error: ${e}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  el('btn-health').addEventListener('click', refreshHealth);

  el('btn-openapi').addEventListener('click', async () => {
    const spec = await fetch('/api/openapi.json').then((r) => r.json());
    alert(`OpenAPI title: ${spec.info?.title || 'n/a'}`);
  });

  el('order-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.currentTarget).entries());
    data.quantity = parseFloat(data.quantity);

    // Try a conventional endpoint; if absent, surface the error body
    const res = await api('/trades/market', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    el('order-result').textContent = JSON.stringify(res, null, 2);
  });

  el('btn-portfolio-test').addEventListener('click', async () => {
    const res = await api('/portfolio');
    el('portfolio-test').textContent = JSON.stringify(res, null, 2);
  });

  el('btn-portfolio-prod').addEventListener('click', async () => {
    const res = await api('/portfolio?env=prod');
    el('portfolio-prod').textContent = JSON.stringify(res, null, 2);
  });

  refreshHealth();
});


