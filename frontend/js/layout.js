/* Reusable header + footer renderer for sub-pages.
   Avoids duplicating the nav across every HTML file. */

(function () {
  const PAGES = [
    ["../index.html", "Home"],
    ["upload.html", "Upload"],
    ["train.html", "Train"],
    ["generate.html", "Generate"],
    ["results.html", "Results"],
    ["download.html", "Download"],
  ];

  function renderHeader(active) {
    const nav = PAGES.map(([href, label]) => {
      const cls = label.toLowerCase() === active ? "active" : "";
      return `<a href="${href}" class="${cls}">${label}</a>`;
    }).join("");

    document.body.insertAdjacentHTML("afterbegin", `
      <header class="site-header">
        <div class="container d-flex align-items-center justify-content-between py-3">
          <a href="../index.html" class="brand"><span class="mark"></span>NTD<span style="opacity:.4;">·</span>SynthGen</a>
          <nav class="d-flex align-items-center flex-wrap">${nav}</nav>
        </div>
      </header>
    `);
  }

  function renderFooter() {
    document.body.insertAdjacentHTML("beforeend", `
      <footer class="site-footer">
        <div class="container d-flex flex-wrap justify-content-between gap-3">
          <span>NTD SynthGen v1.0 · Public health informatics research tool.</span>
          <span style="font-family: var(--f-mono); font-size: 0.78rem;">api.base = <span id="api-base">…</span></span>
        </div>
      </footer>
    `);
    const el = document.getElementById("api-base");
    if (el && window.NTD) el.textContent = NTD.API.base;
  }

  window.NTDLayout = { renderHeader, renderFooter };
})();
