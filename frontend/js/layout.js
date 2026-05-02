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
        <div class="container header-inner py-3">
          <a href="../index.html" class="brand"><span class="mark"></span>NTD<span style="opacity:.4;">·</span>SynthGen</a>
          <button class="nav-toggle" aria-label="Toggle menu" aria-expanded="false" onclick="this.setAttribute('aria-expanded', this.getAttribute('aria-expanded') === 'true' ? 'false' : 'true'); this.nextElementSibling.classList.toggle('open');">
            <span></span><span></span><span></span>
          </button>
          <nav class="site-nav">${nav}</nav>
        </div>
      </header>
    `);
  }

  function renderFooter() {
    document.body.insertAdjacentHTML("beforeend", `
      <footer class="site-footer">
        <div class="container">
          <span>NTD SynthGen v1.0 &middot; Public health informatics research tool</span>
        </div>
      </footer>
    `);
  }

  window.NTDLayout = { renderHeader, renderFooter };
})();
