const PER_PAGE = 48;
function byId(id) { return document.getElementById(id); }
function choices(select, values, label) {
  select.innerHTML = `<option value="">${label}</option>` + values.map(v => `<option>${v}</option>`).join("");
}
function gallery(items, fields, draw, grid) {
  let page = 0, shown = items;
  const bar = byId("bar"), out = byId("gallery"), status = byId("status");
  const selects = fields.map(f => {
    const s = document.createElement("select");
    const values = [...new Set(items.map(f.get).filter(v => v !== null && v !== undefined && v !== ""))].sort((a, b) => (a > b) - (a < b));
    choices(s, values, f.label);
    s.onchange = () => { page = 0; update(); };
    bar.insertBefore(s, status);
    return s;
  });
  function update() {
    shown = items.filter(i => fields.every((f, k) => !selects[k].value || String(f.get(i)) === selects[k].value));
    const pages = Math.max(1, Math.ceil(shown.length / PER_PAGE));
    page = Math.min(Math.max(page, 0), pages - 1);
    status.textContent = `${page + 1}/${pages}`;
    out.innerHTML = shown.slice(page * PER_PAGE, (page + 1) * PER_PAGE).map(draw).join("");
    window.scrollTo(0, 0);
  }
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "SELECT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowRight" || e.key.toLowerCase() === "n") { page++; update(); }  // N and P step as in the viewers
    if (e.key === "ArrowLeft" || e.key.toLowerCase() === "p") { page--; update(); }
  });
  byId("next").onclick = () => { page++; update(); };
  byId("back").onclick = () => { page--; update(); };
  update();
}
