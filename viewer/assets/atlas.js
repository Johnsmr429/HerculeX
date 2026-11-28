const PER_PAGE = 48;
const galleries = [];
function choices(select, values, label) {
    select.innerHTML = `<option value="">${label}</option>${values.map((v) => `<option>${v}</option>`).join("")}`;
}
function gallery(id, items, fields, draw) {
    let page = 0,
        shown = items;
    const root = document.getElementById(id);
    const bar = root.querySelector(".bar"),
        out = root.querySelector(".gallery"),
        status = root.querySelector(".status");
    const selects = fields.map((f) => {
        const s = document.createElement("select");
        const values = [
            ...new Set(items.map(f.get).filter((v) => v !== null && v !== undefined && v !== "")),
        ].sort((a, b) => (a > b) - (a < b));
        choices(s, values, f.label);
        s.onchange = () => {
            page = 0;
            update(true);
        };
        bar.insertBefore(s, status);
        return s;
    });
    function update(moved) {
        shown = items.filter((i) =>
            fields.every((f, k) => !selects[k].value || String(f.get(i)) === selects[k].value),
        );
        const pages = Math.max(1, Math.ceil(shown.length / PER_PAGE));
        page = Math.min(Math.max(page, 0), pages - 1);
        status.textContent = `${page + 1}/${pages}`;
        const above = moved && root.getBoundingClientRect().top < 0;
        out.innerHTML = shown
            .slice(page * PER_PAGE, (page + 1) * PER_PAGE)
            .map(draw)
            .join("");
        if (above) {
            root.scrollIntoView();
        }
    }
    function step(by) {
        page += by;
        update(true);
    }
    root.querySelector(".next").onclick = () => step(1);
    root.querySelector(".back").onclick = () => step(-1);
    galleries.push({ root, step });
    update(false);
}
function inView(g) {
    const r = g.root.getBoundingClientRect();
    return Math.min(r.bottom, innerHeight) - Math.max(r.top, 0);
}
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.metaKey || e.ctrlKey || e.altKey || !galleries.length) {
        return;
    }
    const by =
        e.key === "ArrowRight" || e.key.toLowerCase() === "n"
            ? 1
            : e.key === "ArrowLeft" || e.key.toLowerCase() === "p"
              ? -1
              : 0;
    const here = galleries.reduce((a, b) => (inView(b) > inView(a) ? b : a));
    if (by && inView(here) > 0) {
        here.step(by);
    }
});
