(() => {
  const NS = "http://www.w3.org/2000/svg";
  const SANS = '-apple-system, "SF Pro Text", system-ui, "Helvetica Neue", Arial, sans-serif';
  const MONO = '"KitMono", "SF Mono", Menlo, Consolas, monospace';
  const INK = "#1d1d1f";
  const MUTE = "#6e6e73";
  const FRAME = "#c7c7cc";
  const C = {
    violet: { s: "#974DFF", t: "#7a1fe0", f: "#eee3ff" },
    amber: { s: "#FDAE00", t: "#c27400", f: "#fff0cc" },
    blue: { s: "#3D85FF", t: "#1f5fd6", f: "#ddebff" },
    red: { s: "#F0656B", t: "#d23a43", f: "#fde1e4" },
    green: { s: "#22A06B", t: "#13804f", f: "#ddf3e7" },
    ink: { s: INK, t: INK, f: "#ffffff" },
    soft: { s: "#a1a1a6", t: MUTE, f: "#f5f5f7" },
  };
  const WIND = ["#8e6cff", "#ffb020", "#2fb8c6", "#ff6b6b", "#4e8bff", "#7bcb4c"];

  function el(parent, tag, attrs = {}, txt) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
    if (txt !== undefined) e.textContent = txt;
    parent.appendChild(e);
    return e;
  }

  function readable(size) {
    return size < 12.5 ? size + 1.2 : size;
  }

  function text(g, x, y, s, o = {}) {
    const t = el(
      g,
      "text",
      {
        x,
        y,
        "font-family": o.mono ? MONO : SANS,
        "font-size": readable(o.size || 13),
        "font-weight": o.weight || 400,
        "font-style": o.italic ? "italic" : "normal",
        fill: o.fill || INK,
        "text-anchor": o.anchor || "start",
        "dominant-baseline": o.base || "middle",
        opacity: o.opacity,
        "letter-spacing": o.spacing,
        stroke: o.halo ? "#ffffff" : undefined,
        "stroke-width": o.halo ? 4 : undefined,
        "stroke-linejoin": o.halo ? "round" : undefined,
        "paint-order": o.halo ? "stroke" : undefined,
      },
      s,
    );
    return t;
  }

  function chip(g, cx, cy, s, col = C.ink, o = {}) {
    const size = o.size || 12.5;
    const t = text(g, cx, cy + 0.5, s, {
      mono: !o.sans,
      size,
      fill: col.t,
      anchor: "middle",
      weight: o.weight || 500,
    });
    const w = Math.max(o.min || 0, t.getComputedTextLength() + 2 * (o.pad || 8));
    if (o.left) {
      cx += w / 2;
      t.setAttribute("x", cx);
    }
    const h = o.h || 24;
    const r = el(g, "rect", {
      x: cx - w / 2,
      y: cy - h / 2,
      width: w,
      height: h,
      rx: o.r ?? 6,
      fill: col.f,
      stroke: col.s,
      "stroke-width": o.sw || 1.4,
      "stroke-dasharray": o.dash,
      filter: o.glow ? `url(#glow-${o.glow})` : undefined,
    });
    g.insertBefore(r, t);
    return { l: cx - w / 2, r: cx + w / 2, t: cy - h / 2, b: cy + h / 2, cx, cy, w, h };
  }

  function panel(g, x, y, w, h, title, o = {}) {
    el(g, "rect", { x, y, width: w, height: h, fill: o.fill || "none", stroke: FRAME, "stroke-width": 1.2 });
    if (title) text(g, x + w / 2, y - 18, title, { size: 17, weight: 500, anchor: "middle" });
  }

  function arrow(g, d, col = INK, o = {}) {
    return el(g, "path", {
      d,
      fill: "none",
      stroke: col,
      "stroke-width": o.w || 2,
      "stroke-dasharray": o.dash,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "marker-end": o.head === false ? undefined : `url(#head-${o.head || headName(col)})`,
      "marker-start": o.both ? `url(#tail-${o.head || headName(col)})` : undefined,
      opacity: o.opacity,
    });
  }

  function headName(col) {
    for (const [k, v] of Object.entries(C)) if (v.s === col || v.t === col) return k;
    if (col === MUTE || col === "#a1a1a6") return "soft";
    return "ink";
  }

  function img(g, href, x, y, w, h, o = {}) {
    const id = `clip${Math.round(x)}_${Math.round(y)}_${Math.round(w)}`;
    const cp = el(document.querySelector("defs"), "clipPath", { id });
    el(cp, "rect", { x, y, width: w, height: h, rx: o.r ?? 4 });
    el(g, "image", {
      href,
      x,
      y,
      width: w,
      height: h,
      "clip-path": `url(#${id})`,
      preserveAspectRatio: "none",
      opacity: o.opacity,
    });
    if (o.frame !== false)
      el(g, "rect", {
        x,
        y,
        width: w,
        height: h,
        rx: o.r ?? 4,
        fill: "none",
        stroke: o.stroke || "rgba(0,0,0,0.25)",
        "stroke-width": 1,
      });
  }

  function tag(g, x, y, s, o = {}) {
    const t = text(g, x + 6, y + 10, s, { mono: true, size: 10.5, fill: "#fff", weight: 600 });
    const w = t.getComputedTextLength() + 12;
    g.insertBefore(el(g, "rect", { x, y, width: w, height: 20, rx: 4, fill: o.fill || "rgba(0,0,0,0.55)" }), t);
  }

  function crop(g, href, full, src, x, y, w, h, o = {}) {
    const id = `crop${Math.round(x)}_${Math.round(y)}_${Math.round(w)}`;
    const cp = el(document.querySelector("defs"), "clipPath", { id });
    el(cp, "rect", { x, y, width: w, height: h, rx: o.r ?? 4 });
    const box = el(g, "g", { "clip-path": `url(#${id})` });
    const view = el(box, "svg", {
      x,
      y,
      width: w,
      height: h,
      viewBox: src.join(" "),
      preserveAspectRatio: "none",
    });
    el(view, "image", { href, x: 0, y: 0, width: full[0], height: full[1], preserveAspectRatio: "none" });
    el(g, "rect", {
      x,
      y,
      width: w,
      height: h,
      rx: o.r ?? 4,
      fill: "none",
      stroke: "rgba(0,0,0,0.25)",
      "stroke-width": 1,
    });
  }

  function ramp(id, stops, x1, y1, x2, y2) {
    const lg = el(document.querySelector("defs"), "linearGradient", { id, x1, y1, x2, y2 });
    stops.forEach((c, i) => {
      el(lg, "stop", { offset: `${(100 * i) / (stops.length - 1)}%`, "stop-color": c });
    });
    return `url(#${id})`;
  }

  function figure(w, h) {
    const svg = el(document.body, "svg", { xmlns: NS, width: w, height: h, viewBox: `0 0 ${w} ${h}` });
    const defs = el(svg, "defs");
    for (const [k, v] of Object.entries(C)) {
      for (const [name, path] of [
        ["head", "M0,0 L10,5 L0,10 z"],
        ["tail", "M10,0 L0,5 L10,10 z"],
      ]) {
        const m = el(defs, "marker", {
          id: `${name}-${k}`,
          viewBox: "0 0 10 10",
          refX: name === "head" ? 8 : 2,
          refY: 5,
          markerWidth: 5,
          markerHeight: 5,
          orient: "auto",
        });
        el(m, "path", { d: path, fill: k === "soft" ? "#a1a1a6" : v.s });
      }
      const f = el(defs, "filter", { id: `glow-${k}`, x: "-40%", y: "-60%", width: "180%", height: "220%" });
      el(f, "feDropShadow", { dx: 0, dy: 0, stdDeviation: 5, "flood-color": v.s, "flood-opacity": 0.45 });
    }
    const dots = el(defs, "pattern", { id: "dots", width: 8, height: 8, patternUnits: "userSpaceOnUse" });
    el(dots, "circle", { cx: 1, cy: 1, r: 0.8, fill: "#d2d2d7" });
    const hatch = el(defs, "pattern", {
      id: "hatch",
      width: 8,
      height: 8,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    el(hatch, "line", { x1: 0, y1: 0, x2: 0, y2: 8, stroke: "#f6c3c6", "stroke-width": 3 });
    el(svg, "rect", { x: 0.5, y: 0.5, width: w - 1, height: h - 1, rx: 16, fill: "#ffffff", stroke: "#e3e3e8" });
    return el(svg, "g");
  }

  window.Kit = {
    NS,
    SANS,
    MONO,
    INK,
    MUTE,
    FRAME,
    C,
    WIND,
    el,
    readable,
    text,
    chip,
    panel,
    arrow,
    img,
    crop,
    ramp,
    tag,
    figure,
  };
})();
