const { INK, MUTE, C, el, text, chip, panel, arrow, img, crop, ramp, tag, figure } = window.Kit;

const TURN = ["#5e3cc8", "#4e8bff", "#2fb8c6", "#7bcb4c", "#ffb020", "#ff6b6b"];
const T = window.TILES;
const STRIP = [2400, 862];
const PASSAGE = [3200, 1612];

function note(g, x, y, lines, o = {}) {
  lines.forEach((s, i) => {
    text(g, x, y + i * 15, s, { size: 11.5, italic: true, fill: o.fill || MUTE, anchor: o.anchor || "start" });
  });
}

function turnBar(g, x, y, w, h, vertical) {
  const fill = ramp(
    `turns${Math.round(x)}`,
    vertical ? [...TURN].reverse() : TURN,
    0,
    0,
    vertical ? 0 : 1,
    vertical ? 1 : 0,
  );
  el(g, "rect", { x, y, width: w, height: h, rx: 3, fill });
  if (vertical) {
    text(g, x + w / 2, y - 12, `turn ${T.turns[1]}`, { size: 11, anchor: "middle", fill: MUTE });
    text(g, x + w / 2, y + h + 12, String(T.turns[0]), { size: 11, anchor: "middle", fill: MUTE });
  } else {
    text(g, x - 6, y + h / 2, `turn ${T.turns[0]}`, { size: 11, anchor: "end", fill: MUTE });
    text(g, x + w + 6, y + h / 2, String(T.turns[1]), { size: 11, fill: MUTE });
  }
}

function steps(g, x, y0, gap, items) {
  items.forEach(([name, col], i) => {
    const y = y0 + i * gap;
    chip(g, x, y, name, col, { glow: i === items.length - 1 ? headKey(col) : undefined });
    if (i < items.length - 1) arrow(g, `M${x},${y + 14} L${x},${y + gap - 15}`, INK, { w: 1.6 });
  });
}

function headKey(col) {
  for (const [k, v] of Object.entries(C)) if (v === col) return k;
  return undefined;
}

function hero() {
  const g = figure(1200, 704);
  panel(g, 24, 52, 380, 384, "Trace each slab");
  panel(g, 424, 52, 380, 384, "Stitch one scroll");
  panel(g, 824, 52, 352, 384, "Lay out and read");

  img(g, "tiles/zoom.bmp", 44, 72, 214, 214);
  tag(g, 49, 77, "z 4720 · 448 voxels");
  steps(g, 330, 98, 52, [
    ["crests", C.ink],
    ["surfels", C.ink],
    ["patches", C.ink],
    ["winding solve", C.violet],
  ]);
  note(g, 44, 312, [
    "surfels on an 8-voxel grid with their normals,",
    "turned out from the umbilicus; an exact min-cut",
    "gives every patch a turn, one slab at a time",
  ]);
  text(g, 44, 372, "37 slabs of 768 planes", { mono: true, size: 12 });
  text(g, 44, 392, "about 15 min a slab on an M1 Pro", { size: 11.5, italic: true, fill: MUTE });

  img(g, "tiles/axial-turns.bmp", 476, 72, 252, 252);
  tag(g, 481, 77, "z 4720");
  turnBar(g, 748, 88, 10, 220, true);
  const sv = chip(g, 504, 350, "seam votes", C.ink);
  const ts = chip(g, sv.r + 60, 350, "turn shift", C.ink);
  const pc = chip(g, ts.r + 46, 350, "pieces", C.blue, { glow: "blue" });
  arrow(g, `M${sv.r + 3},350 L${ts.l - 4},350`, INK, { w: 1.6 });
  arrow(g, `M${ts.r + 3},350 L${pc.l - 4},350`, INK, { w: 1.6 });
  note(g, 614, 384, ["every surfel gets one global turn;", "the slabs become 32,737 flat pieces"], {
    anchor: "middle",
  });

  crop(g, "tiles/strip.jpg", STRIP, [0, 0, 2400, 862], 842, 72, 316, 114);
  tag(g, 847, 77, "the page · 7.75 m in four rows");
  crop(g, "tiles/passage.png", PASSAGE, [1700, 830, 1500, 750], 842, 202, 316, 158);
  tag(g, 847, 207, "Reader-PU ink on the strip");
  const lo = chip(g, 960, 402, "layout", C.amber);
  const rd = chip(g, lo.r + 56, 402, "read", C.green, { glow: "green" });
  arrow(g, `M${lo.r + 3},402 L${rd.l - 4},402`, INK, { w: 1.6 });

  arrow(g, "M406,180 L421,180", INK, { w: 2 });
  arrow(g, "M806,180 L821,180", INK, { w: 2 });

  const X0 = 150,
    Y0 = 492,
    HY = 112;
  el(g, "rect", { x: X0, y: Y0, width: 1000, height: HY, fill: "url(#dots)" });
  el(g, "line", { x1: X0, x2: 1150, y1: Y0 + HY, y2: Y0 + HY, stroke: MUTE });
  const yl = el(g, "g", { transform: `translate(${X0 - 42},${Y0 + HY / 2}) rotate(-90)` });
  text(yl, 0, 0, "PHerc1545", { size: 12.5, anchor: "middle" });
  text(yl, 0, 16, "each pair to scale", { size: 11, anchor: "middle", fill: MUTE });
  const groups = [
    ["laid out", 9659, 8836, (v) => v.toLocaleString("en-US")],
    ["largest piece", 40.0, 2.64, (v) => v.toFixed(v < 10 ? 2 : 1)],
    ["largest joined area", 5681, 373, (v) => v.toLocaleString("en-US")],
  ];
  groups.forEach(([name, a, b, fmt], i) => {
    const gx = X0 + 170 + i * 330;
    const m = Math.max(a, b);
    [
      [a, C.violet, "trace 1.5", -44],
      [b, C.soft, "level 1.1", 44],
    ].forEach(([v, col, label, dx]) => {
      const h = Math.max((v / m) * HY, 2);
      el(g, "rect", {
        x: gx + dx - 34,
        y: Y0 + HY - h,
        width: 68,
        height: h,
        fill: col.f,
        stroke: col.s,
        "stroke-width": 1.4,
      });
      text(g, gx + dx, Y0 + HY - h - 12, fmt(v), { size: 15, anchor: "middle", fill: col.t, weight: 500 });
      chip(g, gx + dx, Y0 + HY + 22, label, col, { size: 11.5, h: 21 });
    });
    el(g, "line", { x1: gx - 110, x2: gx + 110, y1: Y0 + HY + 50, y2: Y0 + HY + 50, stroke: INK, "stroke-width": 1.2 });
    text(g, gx, Y0 + HY + 68, `${name} (cm²)`, { size: 15, anchor: "middle", weight: 500 });
  });
}

function slabs(g, x0, y0) {
  const s = 380 / (576 * 4 + 768);
  for (let k = 0; k < 5; k++) {
    const y = y0 + k * 576 * s;
    el(g, "rect", {
      x: x0 + (k % 2) * 12,
      y,
      width: 96,
      height: 768 * s,
      rx: 6,
      fill: k % 2 ? C.blue.f : C.violet.f,
      stroke: k % 2 ? C.blue.s : C.violet.s,
      "stroke-width": 1.3,
      opacity: 0.85,
    });
  }
  const bx = x0 + 120,
    u = 768 * s,
    st = 576 * s;
  el(g, "path", { d: `M${bx},${y0} l6,0 l0,${u} l-6,0`, fill: "none", stroke: INK, "stroke-width": 1.2 });
  text(g, bx + 12, y0 + u / 2, "768 planes", { size: 11.5 });
  el(g, "path", {
    d: `M${bx},${y0 + st + 2} l6,0 l0,${u - st - 4} l-6,0`,
    fill: "none",
    stroke: C.red.s,
    "stroke-width": 1.2,
  });
  text(g, bx + 12, y0 + st + (u - st) / 2 + 14, "192 shared", { size: 11.5, fill: C.red.t });
  el(g, "path", { d: `M${bx},${y0 + 2 * st} l6,0 l0,${st} l-6,0`, fill: "none", stroke: INK, "stroke-width": 1.2 });
  text(g, bx + 12, y0 + 2.5 * st, "step 576", { size: 11.5 });
}

function trace() {
  const g = figure(1200, 516);
  panel(g, 24, 52, 240, 440, "Slabs");
  panel(g, 284, 52, 300, 440, "Crests to patches");
  panel(g, 604, 52, 300, 440, "Winding solve");
  panel(g, 924, 52, 252, 440, "One slab of PHerc1545");

  slabs(g, 44, 72);
  note(g, 44, 462, ["37 slabs cover PHerc1545's", "20,961 planes, each traced alone"]);

  const gx0 = 300,
    gy0 = 70,
    cell = 24;
  for (let i = 0; i <= 11; i++)
    el(g, "line", { x1: gx0 + i * cell, x2: gx0 + i * cell, y1: gy0, y2: gy0 + 192, stroke: "#ececf0" });
  for (let j = 0; j <= 8; j++)
    el(g, "line", { x1: gx0, x2: gx0 + 264, y1: gy0 + j * cell, y2: gy0 + j * cell, stroke: "#ececf0" });
  const curve = (t) => [gx0 + 10 + t * 244, gy0 + 120 - 62 * Math.sin(t * 2.4) + 18 * t];
  const pts = [];
  for (let t = 0; t <= 1.0001; t += 0.02) pts.push(curve(t));
  const d = `M${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" L")}`;
  el(g, "path", { d, fill: "none", stroke: C.amber.f, "stroke-width": 24, "stroke-linecap": "round" });
  el(g, "path", { d, fill: "none", stroke: C.amber.s, "stroke-width": 1.6, "stroke-dasharray": "4 3" });
  const um = [gx0 + 22, gy0 + 176];
  el(g, "circle", { cx: um[0], cy: um[1], r: 5, fill: C.red.s });
  text(g, um[0] + 10, um[1] + 1, "umbilicus", { size: 11.5, italic: true, fill: C.red.t, halo: true });
  for (let t = 0.06; t < 1; t += 0.098) {
    const p = curve(t),
      q = curve(t + 0.01);
    let nx = -(q[1] - p[1]),
      ny = q[0] - p[0];
    const l = Math.hypot(nx, ny);
    nx /= l;
    ny /= l;
    if (nx * (p[0] - um[0]) + ny * (p[1] - um[1]) < 0) {
      nx = -nx;
      ny = -ny;
    }
    arrow(g, `M${p[0]},${p[1]} L${p[0] + nx * 22},${p[1] + ny * 22}`, C.violet.s, { w: 1.6 });
    el(g, "circle", { cx: p[0], cy: p[1], r: 4.2, fill: "#fff", stroke: INK, "stroke-width": 1.5 });
  }
  note(g, 300, 284, [
    "the crest of the surface prediction gives",
    "one surfel per 8-voxel cell, its normal",
    "turned away from the umbilicus",
  ]);
  const py = 376;
  const arc = (x1, x2, y, col) =>
    el(g, "path", {
      d: `M${x1},${y} Q${(x1 + x2) / 2},${y - 14} ${x2},${y}`,
      fill: "none",
      stroke: col,
      "stroke-width": 4,
      "stroke-linecap": "round",
    });
  arc(304, 400, py, C.violet.s);
  arc(424, 560, py, C.violet.s);
  el(g, "path", {
    d: `M400,${py} Q412,${py - 4} 424,${py}`,
    fill: "none",
    stroke: INK,
    "stroke-width": 1.6,
    "stroke-dasharray": "3 3",
  });
  text(g, 412, py + 16, "bridge", { size: 11, italic: true, fill: MUTE, anchor: "middle" });
  arc(304, 430, py + 38, "#2fb8c6");
  el(g, "path", {
    d: `M430,${py + 38} Q470,${py + 34} 560,${py + 28}`,
    fill: "none",
    stroke: "#2fb8c6",
    "stroke-width": 4,
    "stroke-linecap": "round",
  });
  el(g, "path", {
    d: `M430,${py + 38} Q470,${py + 46} 560,${py + 56}`,
    fill: "none",
    stroke: "#2fb8c6",
    "stroke-width": 4,
    "stroke-linecap": "round",
  });
  text(g, 430, py + 40, "✕", { size: 16, anchor: "middle", fill: C.red.s, weight: 700 });
  text(g, 470, py + 72, "cut at forks", { size: 11, italic: true, fill: C.red.t });
  note(g, 300, 462, ["linked surfels form single-sheet patches"]);

  const cx = 626,
    cy = 270;
  const radii = [70, 120, 170];
  const at = (r, a) => [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  const segs = [
    [0.22, 0.72],
    [0.86, 1.4],
  ];
  radii.forEach((r, k) => {
    for (const [a0, a1] of segs) {
      const pts = [];
      for (let a = a0; a <= a1 + 1e-6; a += 0.02)
        pts.push(
          at(r, a)
            .map((v) => v.toFixed(1))
            .join(","),
        );
      el(g, "path", {
        d: `M${pts.join(" L")}`,
        fill: "none",
        stroke: TURN[k * 2],
        "stroke-width": 5,
        "stroke-linecap": "round",
      });
    }
  });
  el(g, "circle", { cx, cy, r: 5, fill: C.red.s });
  [
    [0.47, 0],
    [0.47, 1],
    [1.13, 0],
    [1.13, 1],
  ].forEach(([a, i]) => {
    const p = at(radii[i] + 6, a),
      q = at(radii[i + 1] - 9, a);
    arrow(g, `M${p[0]},${p[1]} L${q[0]},${q[1]}`, INK, { w: 1.5, dash: "4 3" });
    const m = at((radii[i] + radii[i + 1]) / 2, a);
    text(g, m[0] + 9, m[1] + 4, "+1", { mono: true, size: 12, weight: 700, halo: true });
  });
  const s0 = at(radii[1] + 10, 0.72),
    s1 = at(radii[1] + 10, 0.86);
  el(g, "path", {
    d: `M${s0[0]},${s0[1]} Q${at(radii[1] + 22, 0.79)} ${s1[0]},${s1[1]}`,
    fill: "none",
    stroke: INK,
    "stroke-width": 1.5,
  });
  const m0 = at(radii[1] + 30, 0.79);
  text(g, m0[0], m0[1], "0", { mono: true, size: 12, weight: 700, anchor: "middle", halo: true });
  note(g, 806, 150, ["rays: the", "next layer", "out is +1"]);
  note(g, 806, 214, ["links: the", "same turn, 0"]);
  steps(g, 754, 330, 46, [
    ["exact min-cut", C.ink],
    ["8 reweighted draws", C.ink],
    ["merge if 6 of 8 agree", C.violet],
  ]);
  note(g, 620, 462, ["two passes at 40 and 84 voxels then join", "nearby pieces of the same turn"]);

  img(g, "tiles/axial-unit.bmp", 950, 72, 200, 200);
  tag(g, 955, 77, "this slab's own turns");
  img(g, "tiles/zoom.bmp", 950, 284, 200, 200);
  tag(g, 955, 289, "surfels and normals");
}

function stitch() {
  const g = figure(1200, 512);
  panel(g, 24, 52, 380, 440, "Before: each slab's own turns");
  panel(g, 424, 52, 352, 440, "Stitch");
  panel(g, 796, 52, 380, 440, "After: one turn for every surfel");
  const tw = 340,
    th = (tw * T.side.h) / T.side.w;
  img(g, "tiles/side-before.bmp", 44, 72, tw, th);
  img(g, "tiles/side-after.bmp", 816, 72, tw, th);
  const zy = (z) => 72 + ((z - T.side.z0) / (T.side.z1 - T.side.z0)) * th;
  T.side.units.slice(1).forEach(([z]) => {
    el(g, "line", {
      x1: 44 + tw - 26,
      x2: 44 + tw,
      y1: zy(z),
      y2: zy(z),
      stroke: "#fff",
      "stroke-width": 1.6,
      "stroke-dasharray": "3 2",
    });
  });
  tag(g, 49, 77, "z 2704–6160 · one colour per component");
  tag(g, 821, 77, "the same cut, global turn");
  note(g, 44, 430, [
    "a vertical cut through PHerc1545 from the",
    "umbilicus out: each slab solved its turns",
    "alone, so every slab is numbered differently",
  ]);
  turnBar(g, 860, 430, 252, 10, false);
  note(g, 986, 462, ["turns rise smoothly outward", "and carry on across the seams"], { anchor: "middle" });

  const rows = [
    [
      "seam votes",
      ["in each 192-plane overlap, surfels within", "4.5 voxels vote for the turn offset:", "97,102 matched pairs"],
    ],
    [
      "turn shift",
      ["one shift per slab component from", "the votes, refined by L1 against", "radial neighbours along the normals"],
    ],
    [
      "global turn",
      ["every surfel gets a turn and a spiral", "angle: its angle about the umbilicus", "plus 2π per turn"],
    ],
    [
      "pieces",
      [
        "joined within 1 rad of spiral angle, cut",
        "mid-overlap and flattened: 32,737 pieces,",
        "96.7% of grid steps within 10% of length",
      ],
    ],
  ];
  rows.forEach(([name, lines], i) => {
    const y = 92 + i * 110;
    chip(g, 444, y, `${i + 1}  ${name}`, i === 3 ? C.blue : C.ink, { left: true, glow: i === 3 ? "blue" : undefined });
    lines.forEach((s, k) => {
      text(g, 446, y + 24 + k * 16, s, { size: 12 });
    });
    if (i < 3) arrow(g, `M454,${y + 76} L454,${y + 96}`, INK, { w: 1.4 });
  });
  arrow(g, "M406,240 L421,240", INK, { w: 2 });
  arrow(g, "M778,240 L793,240", INK, { w: 2 });
}

function layout() {
  const g = figure(1200, 632);
  panel(g, 24, 52, 1152, 436, "The PHerc1545 strip, 7.75 m in four rows");
  crop(g, "tiles/strip.jpg", STRIP, [0, 0, 2400, 862], 44, 70, 1112, 400);
  tag(g, 934, 300, "the page's surface, no ink");
  const boxes = [
    ["arc length", ["spiral angle maps to length along", "the strip from the pieces' own", "steps in 0.05 rad bins"]],
    [
      "turn consensus",
      ["a piece 2.5 or more turns from the", "papyrus around it moves by whole", "turns: 2,549, then 526 pieces"],
    ],
    [
      "joints",
      ["edges within 48 voxels on the same", "turn at 10 or more points: 17,548", "joints, p90 residual 1.05 mm"],
    ],
    [
      "packing",
      ["joined groups go largest first and", "slide up to 60 mm: 7,978 cm² in", "53 groups of 4 cm² or more"],
    ],
  ];
  boxes.forEach(([name, lines], i) => {
    const x = 24 + i * 292;
    el(g, "rect", { x, y: 508, width: 276, height: 104, rx: 10, fill: "#fcfcfd", stroke: "#e3e3e8" });
    chip(g, x + 16, 532, name, C.amber, { left: true, glow: i === 3 ? "amber" : undefined });
    lines.forEach((s, k) => {
      text(g, x + 16, 562 + k * 17, s, { size: 12 });
    });
    if (i < 3) arrow(g, `M${x + 278},560 L${x + 290},560`, INK, { w: 1.6 });
  });
}

window.FIGS = { hero, trace, stitch, layout };
