/* =========================================================================
   API client — wraps fetch() with sensible defaults and typed helpers.
   API base is detected automatically:
     - Vercel deployment:  /api/...
     - Local dev (run.py): http://127.0.0.1:8000/...
   ========================================================================= */

(function () {
  "use strict";

  const isLocalDev = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  const API_BASE = isLocalDev ? "http://127.0.0.1:8000" : "/api";

  async function request(path, opts = {}) {
    const url = `${API_BASE}${path}`;
    const init = {
      method: opts.method || "GET",
      headers: opts.headers || {},
      ...opts,
    };
    if (opts.json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    }
    let res;
    try {
      res = await fetch(url, init);
    } catch (err) {
      throw new Error(`Network error reaching ${url}: ${err.message}`);
    }
    const contentType = res.headers.get("content-type") || "";
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || body.error || JSON.stringify(body);
      } catch { /* ignore */ }
      throw new Error(`${res.status} — ${detail}`);
    }
    if (contentType.includes("application/json")) return res.json();
    return res;
  }

  const API = {
    base: API_BASE,
    health: () => request("/"),
    listModules: () => request("/modules"),
    getModule: (name) => request(`/modules/${encodeURIComponent(name)}`),
    sample: (module, limit = 20) => request(`/samples/${encodeURIComponent(module)}?limit=${limit}`),
    upload: async (file) => {
      const fd = new FormData();
      fd.append("file", file);
      return request("/upload-data", { method: "POST", body: fd });
    },
    train: (payload) => request("/train-model", { method: "POST", json: payload }),
    generate: (payload) => request("/generate-data", { method: "POST", json: payload }),
    metrics: (modelId, n = 500) => request(`/metrics/${encodeURIComponent(modelId)}?n_synth=${n}`),
    downloadUrl: (datasetId) => `${API_BASE}/download-data/${encodeURIComponent(datasetId)}`,
  };

  // Tiny session store so pages can pass model_id / dataset_id between them.
  const Session = {
    set: (k, v) => sessionStorage.setItem(`ntd_${k}`, typeof v === "string" ? v : JSON.stringify(v)),
    get: (k) => {
      const raw = sessionStorage.getItem(`ntd_${k}`);
      if (!raw) return null;
      try { return JSON.parse(raw); } catch { return raw; }
    },
    clear: (k) => sessionStorage.removeItem(`ntd_${k}`),
  };

  // Toast helper
  function toast(message, kind = "info") {
    const wrap = document.getElementById("alert-wrap");
    if (!wrap) { console.log(`[${kind}]`, message); return; }
    const el = document.createElement("div");
    el.className = `alert alert-${kind} reveal`;
    el.textContent = message;
    wrap.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 600); }, 5000);
  }

  function showLoader(parent, text = "working…") {
    parent.innerHTML = `<span class="spinner-pill">${text}</span>`;
  }

  // Tiny inline histogram (real vs synth overlay).
  function histogram(target, real, synth) {
    const all = [...real, ...synth];
    const min = Math.min(...all), max = Math.max(...all);
    const bins = 18;
    const w = (max - min) / bins || 1;
    const binFor = (v) => Math.min(bins - 1, Math.max(0, Math.floor((v - min) / w)));
    const r = new Array(bins).fill(0);
    const s = new Array(bins).fill(0);
    real.forEach(v => r[binFor(v)]++);
    synth.forEach(v => s[binFor(v)]++);
    const peak = Math.max(...r, ...s) || 1;
    target.innerHTML = `<div class="histogram">${
      r.map((rv, i) => `
        <div style="display:flex; flex-direction:column-reverse; flex:1; gap:1px;">
          <div class="bar"       style="height:${(rv / peak) * 100}%"></div>
          <div class="bar synth" style="height:${(s[i] / peak) * 100}%"></div>
        </div>`).join("")
    }</div>`;
  }

  // Correlation heatmap.
  function heatmap(target, matrix, labels) {
    const n = matrix.length;
    const colorFor = (v) => {
      // Map v in [-1, 1] to a warm-cool diverging palette.
      const t = (v + 1) / 2;
      const r = Math.round(139 * t + 44 * (1 - t));
      const g = Math.round(30 * t + 74 * (1 - t));
      const b = Math.round(30 * t + 107 * (1 - t));
      return `rgb(${r},${g},${b})`;
    };
    target.innerHTML = `<div class="heatmap" style="grid-template-columns:repeat(${n}, 1fr); max-width:340px;">${
      matrix.flatMap((row) =>
        row.map((v) => `<div class="cell" style="background:${colorFor(v)}" title="${v.toFixed(2)}">${v.toFixed(1)}</div>`)
      ).join("")
    }</div><div style="font-family:var(--f-mono); font-size:0.7rem; margin-top:0.4rem; color:var(--c-ink-soft);">${
      (labels || []).map(l => `<span style="margin-right:0.6rem">${l}</span>`).join("")
    }</div>`;
  }

  window.NTD = { API, Session, toast, showLoader, histogram, heatmap };
})();
