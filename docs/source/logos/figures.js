const { INK, MUTE, C, WIND, el, text, chip, panel, arrow, img, tag, figure } = window.Kit;

function tokens(g, x, y, s, n = 4, o = {}) {
  const d = s / n;
  for (let k = 2; k >= 0; k--) {
    const ox = x + k * 6,
      oy = y - k * 6;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        el(g, "rect", {
          x: ox + i * d + 1,
          y: oy + j * d + 1,
          width: d - 2,
          height: d - 2,
          rx: 1.5,
          fill: k === 0 ? o.fill || "#ffffff" : "#f2f2f5",
          stroke: k === 0 ? INK : "#b9b9bf",
          "stroke-width": k === 0 ? 1 : 0.8,
        });
  }
}

function spiral(g, cx, cy, r0, b, turns, o = {}) {
  const seam = o.seam ?? Math.PI;
  const segs = [];
  for (let k = 0; k < Math.ceil(turns); k++) {
    const pts = [];
    const a0 = k * 2 * Math.PI,
      a1 = Math.min(turns, k + 1) * 2 * Math.PI;
    for (let a = a0; a <= a1 + 1e-6; a += 0.03) {
      const r = r0 + (b * a) / (2 * Math.PI);
      const th = seam + (o.curl || 1) * a;
      pts.push([cx + r * Math.cos(th), cy + r * Math.sin(th)]);
    }
    segs.push(pts);
    el(g, "path", {
      d: `M${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" L")}`,
      fill: "none",
      stroke: o.color ? o.color : WIND[k % 6],
      "stroke-width": o.w || 4,
      "stroke-linecap": "round",
      opacity: o.opacity,
    });
  }
  return segs;
}

function hero() {
  const g = figure(1200, 724);
  const T = window.TILES.hero;
  panel(g, 24, 52, 320, 404, "What it reads");
  panel(g, 364, 52, 472, 404, "How it thinks");
  panel(g, 856, 52, 320, 404, "What it predicts");

  const sp = el(g, "g");
  spiral(sp, 84, 112, 3, 7.5, 3.6, { w: 2.2, color: "#b9b9bf" });
  el(sp, "circle", { cx: 84, cy: 112, r: 3.6, fill: C.red.s });
  el(sp, "rect", { x: 98, y: 90, width: 11, height: 11, fill: "none", stroke: C.blue.s, "stroke-width": 1.6 });
  const pr = chip(g, 160, 104, "r", C.ink, { min: 26 });
  const pa = chip(g, pr.r + 38, 104, "curl·θ", C.ink);
  chip(g, pa.r + 24, 104, "z", C.ink, { min: 26 });
  text(g, 136, 134, "where the slab sits around", { size: 11.5, italic: true, fill: MUTE });
  text(g, 136, 149, "the umbilicus", { size: 11.5, italic: true, fill: MUTE });

  for (const [k, op] of [
    [2, 0.35],
    [1, 0.6],
  ])
    img(g, "tiles/hero-ct.bmp", 48 + k * 9, 186 - k * 9, 140, 140, { opacity: op });
  img(g, "tiles/hero-ct.bmp", 48, 186, 140, 140);
  text(g, 210, 232, "CT slab", { size: 13.5, weight: 600 });
  text(g, 210, 252, "64×128×128", { mono: true, size: 12 });
  text(g, 210, 271, "raw X-ray voxels", { size: 11.5, italic: true, fill: MUTE });

  img(g, "tiles/context.bmp", 48, 352, 92, 92);
  el(g, "rect", { x: 104, y: 384, width: 9, height: 9, fill: "none", stroke: C.blue.s, "stroke-width": 1.8 });
  arrow(g, "M113,384 C150,350 170,345 190,330", "#a1a1a6", { w: 1.6, dash: "4 4", head: false });
  text(g, 160, 382, "context", { size: 13.5, weight: 600 });
  text(g, 160, 401, "4 × 256²", { mono: true, size: 12 });
  text(g, 160, 420, "the whole scroll at", { size: 11.5, italic: true, fill: MUTE });
  text(g, 160, 435, "z ±8 and ±24", { size: 11.5, italic: true, fill: MUTE });

  const pos = chip(g, 470, 104, "position", C.ink);
  arrow(g, `M318,104 L${pos.l - 6},104`, INK, { w: 1.8 });
  text(g, pos.r + 10, 104, "Fourier, 128 values", { size: 11.5, italic: true, fill: MUTE });

  const stem = chip(g, 405, 256, "stem", C.ink);
  text(g, 405, 284, "4 × 3D conv", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  arrow(g, `M332,256 L${stem.l - 6},256`, INK, { w: 1.8 });
  tokens(g, 452, 240, 40, 4);
  text(g, 478, 300, "16³ tokens", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  arrow(g, `M${stem.r + 4},256 L446,256`, INK, { w: 1.8 });
  arrow(g, `M470,${pos.b + 4} L470,226`, INK, { w: 1.8 });
  text(g, 478, 170, "+", { size: 15, weight: 600, anchor: "start" });

  el(g, "rect", {
    x: 524,
    y: 166,
    width: 222,
    height: 180,
    rx: 18,
    fill: "#fbfaff",
    stroke: INK,
    "stroke-width": 1.6,
    "stroke-dasharray": "7 6",
  });
  text(g, 635, 188, "recurrent trunk", { size: 13, weight: 600, anchor: "middle" });
  arrow(g, "M500,256 L522,256", INK, { w: 1.8 });
  const bx = [546, 593, 640, 687];
  bx.forEach((x, i) => {
    chip(g, x + 18, 238, `b${i + 1}`, C.violet, { min: 38, h: 38, r: 8 });
    if (i < 3) arrow(g, `M${x + 38},238 L${x + 45},238`, C.violet.s, { w: 1.6 });
  });
  arrow(g, "M720,262 C720,300 552,300 552,262", C.violet.s, { w: 2.4 });
  text(g, 636, 312, "×8 loops", { mono: true, size: 12.5, fill: C.violet.t, weight: 600, anchor: "middle" });
  text(g, 636, 330, "same 4 blocks every loop", { size: 11, italic: true, fill: MUTE, anchor: "middle" });

  const lat = chip(g, 635, 404, "128 latents", C.blue);
  arrow(g, `M${lat.cx},${lat.t - 4} L${lat.cx},350`, C.blue.s, { w: 2, both: true });
  text(g, lat.cx + 10, 373, "exchange every block", { size: 11.5, italic: true, fill: C.blue.t });
  arrow(g, `M300,404 C380,404 470,404 ${lat.l - 6},404`, C.blue.s, { w: 2 });
  text(g, 440, 392, "attend once", { size: 11.5, italic: true, fill: C.blue.t, anchor: "middle" });

  const dec = chip(g, 790, 256, "decoder", C.ink);
  arrow(g, `M748,256 L${dec.l - 4},256`, INK, { w: 1.8 });
  text(g, 790, 282, "3 up-convs", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  text(g, 790, 297, "with skips", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  arrow(g, `M${dec.r + 4},256 L870,256`, INK, { w: 1.8 });

  const tx = [884, 1032],
    ty = [82, 262],
    ts = 120;
  img(g, "tiles/hero-sheet.bmp", tx[0], ty[0], ts, ts);
  img(g, "tiles/hero-winding.bmp", tx[1], ty[0], ts, ts);
  for (const [t, x, y] of T.turns) {
    const px = tx[1] + (x / 384) * ts,
      py = ty[0] + (y / 384) * ts;
    el(g, "circle", { cx: px, cy: py, r: 8.5, fill: "rgba(255,255,255,0.92)" });
    text(g, px, py + 0.5, String(t), { mono: true, size: 11, weight: 700, anchor: "middle", fill: INK });
  }
  el(g, "rect", {
    x: tx[0],
    y: ty[1],
    width: ts,
    height: ts,
    rx: 4,
    fill: "url(#hatch)",
    stroke: C.red.s,
    "stroke-width": 1.4,
    "stroke-dasharray": "5 4",
  });
  text(g, tx[0] + ts / 2, ty[1] + ts / 2 - 8, "not learned", {
    size: 12,
    italic: true,
    fill: C.red.t,
    anchor: "middle",
    weight: 500,
  });
  text(g, tx[0] + ts / 2, ty[1] + ts / 2 + 9, "in v0", {
    size: 12,
    italic: true,
    fill: C.red.t,
    anchor: "middle",
    weight: 500,
  });
  img(g, "tiles/hero-masked.bmp", tx[1], ty[1], ts, ts);
  const outs = [
    [tx[0], ty[0], "sheet", C.amber, "papyrus or air"],
    [tx[1], ty[0], "turn", C.violet, "which wrap"],
    [tx[0], ty[1], "contact", C.red, "two sheets touch"],
    [tx[1], ty[1], "recon", C.blue, "refill hidden CT"],
  ];
  for (const [x, y, name, col, cap] of outs) {
    chip(g, x + ts / 2, y + ts + 20, name, col, { glow: name === "turn" ? "violet" : undefined });
    text(g, x + ts / 2, y + ts + 44, cap, { size: 11.5, italic: true, fill: col.t, anchor: "middle" });
  }

  const X0 = 120,
    Y0 = 488,
    HY = 112,
    X1 = 1150;
  el(g, "rect", { x: X0, y: Y0, width: X1 - X0, height: HY, fill: "url(#dots)" });
  for (const v of [0, 0.5, 1]) {
    text(g, X0 - 10, Y0 + HY - v * HY, v.toFixed(1), { size: 12, anchor: "end", fill: MUTE });
    el(g, "line", { x1: X0 - 4, x2: X0, y1: Y0 + HY - v * HY, y2: Y0 + HY - v * HY, stroke: MUTE });
  }
  el(g, "line", { x1: X0, x2: X0, y1: Y0 - 4, y2: Y0 + HY, stroke: MUTE });
  el(g, "line", { x1: X0, x2: X1, y1: Y0 + HY, y2: Y0 + HY, stroke: MUTE });
  const yl = el(g, "g", { transform: `translate(${X0 - 52},${Y0 + HY / 2}) rotate(-90)` });
  text(yl, 0, 0, "v0 held-out", { size: 12.5, anchor: "middle", fill: INK });
  text(yl, 0, 16, "step 30,000", { size: 11, anchor: "middle", fill: MUTE });
  const bars = [
    ["Dice", "Scroll 1 cubes", 0.942, C.amber],
    ["recall", "PHerc0332 lines", 0.982, C.amber],
    ["Dice", "Kaggle surfaces", 0.311, C.amber],
    ["next turn", "Scroll 1 cubes", 0.771, C.violet],
    ["several wraps", "PHercParis4", 0.208, C.violet],
    ["F1", "Scroll 1 cubes", 0.0, C.red],
  ];
  const groups = [
    ["sheet", 0, 3, C.amber],
    ["turn", 3, 5, C.violet],
    ["contact", 5, 6, C.red],
  ];
  const slot = 150,
    gap = 40;
  let x = X0 + 60;
  const cx = [];
  for (let i = 0; i < bars.length; i++) {
    if (i === 3 || i === 5) x += gap;
    cx.push(x + slot / 2);
    x += slot;
  }
  bars.forEach(([m, d, v, col], i) => {
    const bh = Math.max(v * HY, 1.5);
    el(g, "rect", {
      x: cx[i] - 34,
      y: Y0 + HY - bh,
      width: 68,
      height: bh,
      fill: col.f,
      stroke: col.s,
      "stroke-width": 1.4,
    });
    text(g, cx[i], Y0 + HY - bh - 12, v.toFixed(v ? 3 : 0), { size: 15, anchor: "middle", fill: col.t, weight: 500 });
    chip(g, cx[i], Y0 + HY + 22, m, col, { size: 11.5, h: 21 });
    text(g, cx[i], Y0 + HY + 46, d, { size: 11.5, anchor: "middle", fill: MUTE });
  });
  for (const [name, a, b, col] of groups) {
    const l = cx[a] - slot / 2 + 12,
      r = cx[b - 1] + slot / 2 - 12;
    el(g, "line", { x1: l, x2: r, y1: Y0 + HY + 66, y2: Y0 + HY + 66, stroke: col.s, "stroke-width": 1.6 });
    text(g, (l + r) / 2, Y0 + HY + 84, name, { size: 15, anchor: "middle", fill: col.t, weight: 500 });
  }
}

function winding() {
  const g = figure(1200, 476);
  panel(g, 24, 52, 380, 404, "A number for every wrap");
  panel(g, 424, 52, 452, 404, "Four kinds of label, one loss");
  panel(g, 896, 52, 280, 404, "Held-out PHercParis4 cube");

  const cx = 214,
    cy = 204;
  el(g, "line", {
    x1: cx,
    y1: cy,
    x2: cx - 160,
    y2: cy,
    stroke: "#a1a1a6",
    "stroke-width": 1.5,
    "stroke-dasharray": "5 5",
  });
  text(g, cx - 150, cy - 12, "seam", { size: 11.5, italic: true, fill: MUTE });
  spiral(g, cx, cy, 14, 31, 4, { w: 5, seam: Math.PI });
  const ray = -0.62;
  const hits = [];
  for (let k = 0; k < 4; k++) {
    const a = ((ray - Math.PI + 4 * Math.PI) % (2 * Math.PI)) + 2 * Math.PI * k;
    const r = 14 + (31 * a) / (2 * Math.PI);
    hits.push([cx + r * Math.cos(ray), cy + r * Math.sin(ray), r]);
  }
  arrow(g, `M${cx},${cy} L${cx + 160 * Math.cos(ray)},${cy + 160 * Math.sin(ray)}`, INK, { w: 1.6 });
  hits.forEach(([x, y]) => {
    el(g, "circle", { cx: x, cy: y, r: 5, fill: "#fff", stroke: INK, "stroke-width": 1.6 });
  });
  for (let k = 0; k < 4; k++) {
    const a = 2 * Math.PI * k + Math.PI * 1.5;
    const r = 14 + (31 * a) / (2 * Math.PI) + 15.5;
    const x = cx + r * Math.cos(Math.PI + a),
      y = cy + r * Math.sin(Math.PI + a);
    el(g, "circle", { cx: x, cy: y, r: 9, fill: "#fff", stroke: WIND[k], "stroke-width": 1.6 });
    text(g, x, y + 0.5, String(k), { mono: true, size: 11, weight: 700, anchor: "middle", fill: INK });
  }
  el(g, "circle", { cx, cy, r: 4.5, fill: C.red.s });
  text(g, cx + 8, cy + 17, "umbilicus", { size: 11.5, italic: true, fill: C.red.t, halo: true, weight: 500 });
  text(g, 300, 76, "next turn out:", { size: 11.5, italic: true, fill: MUTE });
  text(g, 300, 91, "winding + 1", { size: 11.5, italic: true, fill: MUTE });
  chip(g, cx, 392, "winding = round(T − curl·θ/2π)", C.violet, { size: 13, h: 30, glow: "violet" });
  text(g, cx, 425, "the network predicts T at every voxel;", {
    size: 11.5,
    italic: true,
    fill: MUTE,
    anchor: "middle",
  });
  text(g, cx, 440, "the umbilicus and curl give θ", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });

  const cells = [
    [444, 70, "same sheet", "Δu = 0", "instance cubes · winding sets", "same"],
    [660, 70, "next turn", "Δu = +1", "instance cubes, within 480 µm", "next"],
    [444, 252, "relative", "Δu = wᵢ − wⱼ", "pairs across winding sets", "rel"],
    [660, 252, "absolute", "u = w", "off by default (weight 0)", "abs"],
  ];
  for (const [x, y, name, rule, src, kind] of cells) {
    const w = 200,
      h = 170;
    el(g, "rect", { x, y, width: w, height: h, rx: 10, fill: "#fcfcfd", stroke: "#e3e3e8" });
    chip(g, x + w / 2, y + 18, name, C.violet, { size: 12 });
    const bez = (k, t) => {
      const p0 = [x + 22, y + 64 + k * 26],
        p1 = [x + w / 2, y + 44 + k * 26],
        p2 = [x + w - 22, y + 64 + k * 26];
      return [0, 1].map((i) => (1 - t) * (1 - t) * p0[i] + 2 * (1 - t) * t * p1[i] + t * t * p2[i]);
    };
    for (let k = 0; k < 3; k++) {
      const a = bez(k, 0),
        m = [x + w / 2, y + 44 + k * 26],
        b = bez(k, 1);
      el(g, "path", {
        d: `M${a} Q${m} ${b}`,
        fill: "none",
        stroke: WIND[k],
        "stroke-width": 4,
        "stroke-linecap": "round",
      });
    }
    const pt = (p, lab) => {
      el(g, "circle", { cx: p[0], cy: p[1], r: 5, fill: "#fff", stroke: INK, "stroke-width": 1.6 });
      if (lab) text(g, p[0] + 9, p[1] - 9, lab, { mono: true, size: 10.5, fill: INK, weight: 600 });
    };
    if (kind === "same") {
      const a = bez(0, 0.2),
        b = bez(0, 0.78);
      arrow(g, `M${a[0] + 7},${a[1] - 2} Q${x + w / 2},${a[1] - 26} ${b[0] - 7},${b[1] - 2}`, INK, {
        w: 1.4,
        dash: "3 4",
        head: false,
      });
      pt(a, "a");
      pt(b, "b");
    } else if (kind === "next") {
      const a = bez(0, 0.5),
        b = bez(1, 0.5);
      arrow(g, `M${a[0]},${a[1] + 6} L${b[0]},${b[1] - 8}`, INK, { w: 1.6 });
      pt(a, "a");
      pt(b, "b");
      text(g, b[0] + 12, (a[1] + b[1]) / 2 + 30, "outward ray", { size: 10.5, italic: true, fill: MUTE });
    } else if (kind === "rel") {
      const a = bez(0, 0.25),
        b = bez(2, 0.72);
      arrow(g, `M${a[0] + 5},${a[1] + 5} L${b[0] - 5},${b[1] - 5}`, INK, { w: 1.4, dash: "3 4", head: false });
      pt(a, "wᵢ=0");
      pt(b, "wⱼ=2");
    } else {
      const a = bez(2, 0.5);
      pt(a, "w=2");
    }
    text(g, x + w / 2, y + 140, rule, { mono: true, size: 13, anchor: "middle", fill: C.violet.t, weight: 600 });
    text(g, x + w / 2, y + 158, src, { size: 11, italic: true, anchor: "middle", fill: MUTE });
  }
  el(g, "rect", { x: 660, y: 252, width: 200, height: 132, rx: 10, fill: "rgba(255,255,255,0.55)" });
  text(g, 650, 438, "a Huber loss on u at each labelled point, δ 0.25 (absolute 0.5)", {
    size: 11.5,
    italic: true,
    anchor: "middle",
    fill: MUTE,
  });

  const T = window.TILES.hero;
  img(g, "tiles/hero-ct.bmp", 914, 70, 118, 118);
  img(g, "tiles/hero-label.bmp", 1042, 70, 118, 118);
  tag(g, 918, 74, "CT");
  tag(g, 1046, 74, "label");
  img(g, "tiles/hero-winding.bmp", 936, 202, 200, 200);
  tag(g, 941, 207, "u");
  for (const [t, x, y] of T.turns) {
    const px = 936 + (x / 384) * 200,
      py = 202 + (y / 384) * 200;
    el(g, "circle", { cx: px, cy: py, r: 10, fill: "rgba(255,255,255,0.92)" });
    text(g, px, py + 0.5, String(t), { mono: true, size: 12, weight: 700, anchor: "middle", fill: INK });
  }
  text(g, 1036, 422, "sheets coloured by predicted u;", { size: 11.5, italic: true, anchor: "middle", fill: MUTE });
  text(g, 1036, 437, "dots mark places where u ≈ an integer", {
    size: 11.5,
    italic: true,
    anchor: "middle",
    fill: MUTE,
  });
}

function loop() {
  const g = figure(1200, 452);
  panel(g, 24, 52, 676, 380, "Depth by repetition");
  panel(g, 720, 52, 456, 380, "Inside one block");
  const L = 6,
    keep = 6,
    w = 62,
    gap = 10,
    x0 = 136,
    y0 = 176,
    h = 92;
  const sc = chip(g, 66, 222, "S", C.ink, { min: 34, h: 34 });
  text(g, 66, 252, "stem", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  text(g, 66, 266, "tokens", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  const lc = chip(g, 66, 362, "L", C.blue, { min: 34, h: 34 });
  text(g, 66, 392, "128", { size: 11.5, italic: true, fill: C.blue.t, anchor: "middle" });
  text(g, 66, 406, "latents", { size: 11.5, italic: true, fill: C.blue.t, anchor: "middle" });
  el(g, "line", {
    x1: sc.r + 2,
    y1: 132,
    x2: x0 + L * (w + gap) - gap - w / 2,
    y2: 132,
    stroke: INK,
    "stroke-width": 1.6,
  });
  el(g, "path", {
    d: `M${sc.cx},${sc.t - 2} L${sc.cx},132 L${sc.r + 2},132`,
    fill: "none",
    stroke: INK,
    "stroke-width": 1.6,
  });
  text(g, x0, 114, "S and a loop embedding eₗ are added at the start of every loop", {
    size: 11.5,
    italic: true,
    fill: MUTE,
  });
  el(g, "path", {
    d: `M${lc.cx},${lc.t - 2} L${lc.cx},316 L${x0 + L * (w + gap) - gap - w / 2},316`,
    fill: "none",
    stroke: C.blue.s,
    "stroke-width": 1.6,
  });
  text(g, x0, 340, "latents carry across loops and swap with the tokens inside every block", {
    size: 11.5,
    italic: true,
    fill: C.blue.t,
  });
  const fills = [C.violet.f, C.blue.f, C.amber.f, C.green.f];
  const strokes = [C.violet.s, C.blue.s, C.amber.s, C.green.s];
  for (let l = 0; l < L; l++) {
    const x = x0 + l * (w + gap);
    const on = l >= L - keep;
    const lg = el(g, "g", { opacity: on ? 1 : 0.38 });
    arrow(lg, `M${x + w / 2},133 L${x + w / 2},${y0 - 3}`, INK, { w: 1.4 });
    el(lg, "rect", { x, y: y0, width: w, height: h, rx: 7, fill: "#fff", stroke: INK, "stroke-width": 1.3 });
    for (let b = 0; b < 4; b++)
      el(lg, "rect", {
        x: x + 6,
        y: y0 + 7 + b * 20.5,
        width: w - 12,
        height: 16,
        rx: 3,
        fill: fills[b],
        stroke: strokes[b],
        "stroke-width": 1.1,
      });
    el(lg, "line", {
      x1: x + w / 2,
      y1: y0 + h + 2,
      x2: x + w / 2,
      y2: 314,
      stroke: C.blue.s,
      "stroke-width": 1.3,
      "stroke-dasharray": "3 3",
    });
    text(lg, x + w / 2, y0 + h + 14, String(l + 1), { mono: true, size: 11, anchor: "middle", fill: INK });
    if (l < L - 1) arrow(lg, `M${x + w + 1},${y0 + h / 2} L${x + w + gap - 1},${y0 + h / 2}`, INK, { w: 1.2 });
  }
  arrow(g, `M${sc.r + 2},${y0 + h / 2} L${x0 - 4},${y0 + h / 2}`, INK, { w: 1.6 });
  const xe = x0 + L * (w + gap) - gap;
  const dec = chip(g, xe + 52, y0 + h / 2, "decoder", C.ink);
  arrow(g, `M${xe + 2},${y0 + h / 2} L${dec.l - 4},${y0 + h / 2}`, INK, { w: 1.6 });
  el(g, "path", {
    d: `M${x0},372 L${x0},380 L${xe},380 L${xe},372`,
    fill: "none",
    stroke: C.violet.s,
    "stroke-width": 1.8,
  });
  text(g, (x0 + xe) / 2, 396, "a step draws 4 to 8 loops and backpropagates through all of them", {
    size: 11.5,
    italic: true,
    fill: C.violet.t,
    anchor: "middle",
    weight: 500,
  });
  text(g, (x0 + xe) / 2, 411, "evaluation and dumps use 8", {
    size: 11.5,
    italic: true,
    fill: C.violet.t,
    anchor: "middle",
  });
  let lx = x0;
  ["block 1", "block 2", "block 3", "block 4"].forEach((s, b) => {
    el(g, "rect", {
      x: lx,
      y: 76,
      width: 22,
      height: 15,
      rx: 3,
      fill: fills[b],
      stroke: strokes[b],
      "stroke-width": 1.1,
    });
    const bb = text(g, lx + 29, 84, s, { size: 11.5, fill: INK }).getBBox();
    lx = bb.x + bb.width + 18;
  });
  text(g, lx + 2, 84, "the same weights in every loop", { size: 11.5, italic: true, fill: MUTE });

  const rx = 760;
  el(g, "line", { x1: rx, y1: 78, x2: rx, y2: 416, stroke: INK, "stroke-width": 1.6 });
  arrow(g, `M${rx},404 L${rx},420`, INK, { w: 1.6 });
  const rows = [
    ["window attention", C.violet, "4×4×4 token windows; blocks 2 and 4", "shift the windows by 2 tokens"],
    ["latent exchange", C.blue, "128 latents read all 4,096 tokens,", "then every token reads the latents"],
    ["z recurrence", C.amber, "a gated linear recurrence (RG-LRU)", "runs up and down the slab's z axis"],
    ["SwiGLU", C.green, "a gated MLP, 256 → 683 → 256", "per token"],
  ];
  rows.forEach(([name, col, l1, l2], i) => {
    const y = 104 + i * 82;
    el(g, "circle", { cx: rx, cy: y, r: 9, fill: "#fff", stroke: INK, "stroke-width": 1.5 });
    text(g, rx, y + 0.5, "+", { size: 14, anchor: "middle", weight: 600 });
    el(g, "line", { x1: rx + 9, y1: y, x2: rx + 28, y2: y, stroke: INK, "stroke-width": 1.4 });
    const ix = rx + 34,
      iy = y - 30;
    el(g, "rect", { x: ix, y: iy, width: 92, height: 60, rx: 8, fill: col.f, stroke: col.s, "stroke-width": 1.3 });
    icon(g, i, ix, iy, col);
    chip(g, ix + 108, y - 14, name, col, { size: 12, left: true });
    text(g, ix + 108, y + 10, l1, { size: 11.5, fill: INK });
    text(g, ix + 108, y + 25, l2, { size: 11.5, fill: MUTE });
  });
}

function icon(g, kind, x, y, col) {
  if (kind === 0) {
    const n = 8,
      d = 6.2,
      ox = x + 21,
      oy = y + 5;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        el(g, "rect", {
          x: ox + i * d + 0.6,
          y: oy + j * d + 0.6,
          width: d - 1.2,
          height: d - 1.2,
          fill: "#fff",
          stroke: "#cbb8f5",
          "stroke-width": 0.6,
        });
    el(g, "rect", { x: ox, y: oy, width: 4 * d, height: 4 * d, fill: "none", stroke: col.s, "stroke-width": 1.8 });
    el(g, "rect", {
      x: ox + 2 * d,
      y: oy + 2 * d,
      width: 4 * d,
      height: 4 * d,
      fill: "none",
      stroke: col.s,
      "stroke-width": 1.4,
      "stroke-dasharray": "3 2",
    });
  } else if (kind === 1) {
    for (let i = 0; i < 7; i++)
      el(g, "rect", {
        x: x + 10 + i * 10.5,
        y: y + 41,
        width: 8,
        height: 8,
        rx: 1.5,
        fill: "#fff",
        stroke: INK,
        "stroke-width": 0.9,
      });
    for (let i = 0; i < 3; i++) {
      const cx = x + 28 + i * 18,
        cy = y + 15;
      el(g, "path", {
        d: `M${cx},${cy - 6} L${cx + 6},${cy} L${cx},${cy + 6} L${cx - 6},${cy} z`,
        fill: "#fff",
        stroke: col.s,
        "stroke-width": 1.4,
      });
    }
    arrow(g, `M${x + 30},${y + 38} L${x + 30},${y + 25}`, col.s, { w: 1.4 });
    arrow(g, `M${x + 62},${y + 24} L${x + 62},${y + 37}`, col.s, { w: 1.4 });
  } else if (kind === 2) {
    for (let i = 0; i < 5; i++)
      el(g, "rect", {
        x: x + 40,
        y: y + 5 + i * 10.4,
        width: 12,
        height: 8.5,
        rx: 1.5,
        fill: "#fff",
        stroke: INK,
        "stroke-width": 0.9,
      });
    arrow(g, `M${x + 30},${y + 52} L${x + 30},${y + 9}`, col.s, { w: 1.5 });
    arrow(g, `M${x + 62},${y + 8} L${x + 62},${y + 51}`, col.s, { w: 1.5 });
    text(g, x + 80, y + 30, "z", { mono: true, size: 11, fill: col.t, weight: 600 });
  } else {
    const hs = [16, 34, 16];
    const ws = [10, 10, 10];
    let xx = x + 22;
    hs.forEach((h, i) => {
      el(g, "rect", {
        x: xx,
        y: y + 30 - h / 2 + 0,
        width: ws[i],
        height: h,
        rx: 2,
        fill: "#fff",
        stroke: col.s,
        "stroke-width": 1.3,
      });
      if (i < 2) arrow(g, `M${xx + 12},${y + 30} L${xx + 20},${y + 30}`, col.s, { w: 1.3 });
      xx += 22;
    });
  }
}

function training() {
  const g = figure(1200, 492);
  panel(g, 24, 52, 360, 420, "1 · Pre-train: refill the CT");
  panel(g, 404, 52, 472, 420, "2 · Mid-train: name what is there");
  panel(g, 896, 52, 280, 420, "Mine the hard slabs");

  img(g, "tiles/hero-masked.bmp", 44, 72, 130, 130);
  img(g, "tiles/hero-ct.bmp", 234, 72, 130, 130);
  arrow(g, "M180,137 L228,137", C.blue.s, { w: 2.2 });
  chip(g, 109, 222, "75% hidden", C.ink, { size: 11.5 });
  chip(g, 299, 222, "recon", C.blue);
  text(g, 109, 246, "tokens of 4×8×8 voxels", { size: 11, italic: true, fill: MUTE, anchor: "middle" });
  text(g, 299, 246, "predict the hidden CT", { size: 11, italic: true, fill: C.blue.t, anchor: "middle" });
  const pre = [0.328, 0.305, 0.294, 0.285, 0.275, 0.268, 0.261, 0.256, 0.254, 0.253];
  const px0 = 84,
    px1 = 360,
    py0 = 290,
    py1 = 410,
    lo = 0.24,
    hi = 0.34;
  el(g, "rect", { x: px0, y: py0, width: px1 - px0, height: py1 - py0, fill: "url(#dots)" });
  const X = (s) => px0 + ((s - 5) / 45) * (px1 - px0),
    Y = (v) => py1 - ((v - lo) / (hi - lo)) * (py1 - py0);
  for (const v of [0.25, 0.3]) text(g, px0 - 8, Y(v), v.toFixed(2), { size: 11, anchor: "end", fill: MUTE });
  for (const s of [5, 25, 50]) text(g, X(s), py1 + 14, `${s}k`, { size: 11, anchor: "middle", fill: MUTE });
  el(g, "line", { x1: px0, x2: px1, y1: py1, y2: py1, stroke: MUTE });
  el(g, "line", { x1: px0, x2: px0, y1: py0, y2: py1, stroke: MUTE });
  el(g, "polyline", {
    points: pre.map((v, i) => `${X(5 + 5 * i)},${Y(v)}`).join(" "),
    fill: "none",
    stroke: C.blue.s,
    "stroke-width": 2.2,
  });
  pre.forEach((v, i) => {
    el(g, "circle", { cx: X(5 + 5 * i), cy: Y(v), r: 2.8, fill: C.blue.s });
  });
  text(g, X(5) + 8, Y(0.328) - 2, "0.328", { size: 11.5, fill: C.blue.t });
  text(g, X(50), Y(0.253) - 14, "0.253", { size: 11.5, fill: C.blue.t, anchor: "end" });
  text(g, px0, 276, "v0 held-out reconstruction MSE", { size: 12, weight: 500 });
  text(g, 204, 442, "50,000 steps on 86,817 slabs of 31 scans;", {
    size: 11.5,
    italic: true,
    fill: MUTE,
    anchor: "middle",
  });
  text(g, 204, 457, "predicting the mean scores about 1.0", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });

  const src = [
    [
      "instance cubes + bench",
      0.35,
      "1,152",
      [
        ["sheet", C.amber],
        ["contact", C.red],
        ["same", C.violet],
        ["next", C.violet],
      ],
    ],
    [
      "winding points + bench",
      0.25,
      "2,584",
      [
        ["same", C.violet],
        ["relative", C.violet],
      ],
    ],
    ["fibres", 0.1, "480", [["sheet ×0.7", C.amber]]],
    ["unlabelled CT", 0.15, "86,817", [["recon", C.blue]]],
    ["Kaggle surfaces", 0.15, "12,141", [["sheet", C.amber]]],
  ];
  const shades = ["#d9c7ff", "#ece2ff", "#ffe2a3", "#cfe2ff", "#fff1d1"];
  let bx = 424;
  const bw = 432;
  src.forEach(([, share], i) => {
    const w = share * bw;
    el(g, "rect", { x: bx, y: 74, width: w, height: 26, fill: shades[i], stroke: "#fff", "stroke-width": 2 });
    text(g, bx + w / 2, 87.5, `${Math.round(share * 100)}%`, { size: 11.5, anchor: "middle", weight: 600, fill: INK });
    bx += w;
  });
  src.forEach(([name, , , heads], i) => {
    const y = 128 + i * 31;
    el(g, "rect", { x: 424, y: y - 7, width: 14, height: 14, rx: 3, fill: shades[i] });
    text(g, 446, y, name, { size: 12.5, weight: 500 });
    let cx = 620;
    for (const [h, col] of heads) cx = chip(g, cx, y, h, col, { size: 11.5, h: 21, pad: 7, left: true }).r + 6;
  });
  text(g, 620, 112, "losses it feeds", { size: 10.5, italic: true, fill: MUTE });

  const mx0 = 452,
    mx1 = 766,
    my0 = 312,
    my1 = 430;
  el(g, "rect", { x: mx0, y: my0, width: mx1 - mx0, height: my1 - my0, fill: "url(#dots)" });
  const MX = (s) => mx0 + ((s - 5) / 30) * (mx1 - mx0),
    MY = (v) => my1 - v * (my1 - my0);
  for (const v of [0, 0.5, 1]) text(g, mx0 - 8, MY(v), v.toFixed(1), { size: 11, anchor: "end", fill: MUTE });
  for (const s of [5, 15, 25, 35]) text(g, MX(s), my1 + 14, `${s}k`, { size: 11, anchor: "middle", fill: MUTE });
  el(g, "line", { x1: mx0, x2: mx1, y1: my1, y2: my1, stroke: MUTE });
  el(g, "line", { x1: mx0, x2: mx0, y1: my0, y2: my1, stroke: MUTE });
  el(g, "line", {
    x1: MX(30),
    x2: MX(30),
    y1: my0 - 6,
    y2: my1,
    stroke: INK,
    "stroke-width": 1.2,
    "stroke-dasharray": "3 3",
  });
  text(g, MX(30) - 6, MY(0.36), "last saved", { size: 11, italic: true, anchor: "end", fill: INK });
  const series = [
    ["sheet Dice", C.amber, [0.914, 0.929, 0.926, 0.936, 0.936, 0.943, 0.947], "Scroll 1 cubes"],
    ["next turn", C.violet, [0.311, 0.543, 0.629, 0.661, 0.749, 0.772, 0.79], "Scroll 1 cubes"],
    ["several wraps", C.violet, [0, 0, 0, 0.011, 0.069, 0.184, 0.43], "PHercParis4"],
  ];
  series.forEach(([name, col, vs, data], k) => {
    const pts = vs.map((v, i) => [MX(5 + 5 * i), MY(v)]);
    el(g, "polyline", {
      points: pts
        .slice(0, 6)
        .map((p) => p.join(","))
        .join(" "),
      fill: "none",
      stroke: col.s,
      "stroke-width": 2.2,
      "stroke-dasharray": k === 2 ? "6 4" : undefined,
    });
    el(g, "polyline", {
      points: pts
        .slice(5)
        .map((p) => p.join(","))
        .join(" "),
      fill: "none",
      stroke: col.s,
      "stroke-width": 1.6,
      "stroke-dasharray": "2 3",
    });
    pts.forEach((p, i) => {
      el(g, "circle", {
        cx: p[0],
        cy: p[1],
        r: 2.8,
        fill: i === 6 ? "#fff" : col.s,
        stroke: col.s,
        "stroke-width": 1.3,
      });
    });
    const ly = [MY(0.947) - 2, MY(0.79) + 12, MY(0.43) + 2][k];
    text(g, mx1 + 10, ly, name, { size: 11.5, fill: col.t, weight: 600 });
    text(g, mx1 + 10, ly + 14, data, { size: 10.5, fill: MUTE, italic: true });
  });
  text(g, mx0, 296, "v0 held-out validation", { size: 12, weight: 500 });
  text(g, mx1 - 2, 296, "○ step 35k, not saved", { size: 10.5, italic: true, fill: MUTE, anchor: "end" });

  chip(g, 1036, 96, "confidence", C.green);
  text(g, 1036, 124, "a fifth head predicts each token's", {
    size: 11.5,
    italic: true,
    fill: C.green.t,
    anchor: "middle",
  });
  text(g, 1036, 139, "log(1 + loss)", { mono: true, size: 11.5, fill: C.green.t, anchor: "middle" });
  const hx0 = 924,
    hy1 = 310,
    n = 15;
  for (let i = 0; i < n; i++) {
    const h = 110 * Math.exp(-i / 6.5) + 14;
    const hot = i < 3;
    el(g, "rect", {
      x: hx0 + i * 15.5,
      y: hy1 - h,
      width: 12,
      height: h,
      rx: 2,
      fill: hot ? C.green.f : "#f2f2f5",
      stroke: hot ? C.green.s : "#c7c7cc",
      "stroke-width": 1.2,
    });
  }
  el(g, "line", { x1: hx0 - 4, x2: hx0 + n * 15.5, y1: hy1, y2: hy1, stroke: MUTE });
  text(g, hx0, hy1 + 14, "slabs of one source, hardest first", { size: 10.5, italic: true, fill: MUTE });
  el(g, "path", {
    d: `M${hx0},180 L${hx0},174 L${hx0 + 3 * 15.5 - 3.5},174 L${hx0 + 3 * 15.5 - 3.5},180`,
    fill: "none",
    stroke: C.green.s,
    "stroke-width": 1.6,
  });
  text(g, hx0 + 56, 177, "top 20% → drawn 3× as often", { size: 11.5, fill: C.green.t, weight: 600 });
  text(g, 1036, 360, "every 5,000 steps the model scores", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
  text(g, 1036, 375, "up to about 2,000 slabs per labelled", {
    size: 11.5,
    italic: true,
    fill: MUTE,
    anchor: "middle",
  });
  text(g, 1036, 390, "source and re-weights the sampler", { size: 11.5, italic: true, fill: MUTE, anchor: "middle" });
}

function kernel() {
  const g = figure(1200, 290);
  panel(g, 24, 52, 1152, 214, "One CUDA launch per micro-batch step · v0 time on one L4");
  const parts = [
    ["stem", 0.0572, "fwd"],
    ["loops", 0.2792, "fwd"],
    ["decoder", 0.0725, "fwd"],
    ["losses", 0.004, "fwd"],
    ["decoder", 0.1383, "bwd"],
    ["loops (recompute + backward)", 0.2621 + 0.0009 + 0.0004, "bwd"],
    ["context", 0.0004, "bwd"],
    ["stem", 0.182, "bwd"],
    ["AdamW", 0.0029, "opt"],
  ];
  const sum = parts.reduce((a, p) => a + p[1], 0);
  const x0 = 60,
    x1 = 1140,
    y = 126,
    h = 44;
  const col = { fwd: C.blue, bwd: C.violet, opt: C.amber };
  let x = x0;
  const spans = [];
  for (const [name, v, kind] of parts) {
    const w = ((x1 - x0) * v) / sum;
    el(g, "rect", {
      x,
      y,
      width: Math.max(w, 1.2),
      height: h,
      fill: col[kind].f,
      stroke: col[kind].s,
      "stroke-width": 1.2,
    });
    spans.push([x, w, name, v, kind]);
    x += w;
  }
  for (const [sx, w, name, v, kind] of spans) {
    if (w > 60) {
      text(g, sx + w / 2, y + h / 2 - 7, name, { size: 12, anchor: "middle", weight: 500, fill: col[kind].t });
      text(g, sx + w / 2, y + h / 2 + 10, `${Math.round(100 * v)}%`, {
        mono: true,
        size: 11.5,
        anchor: "middle",
        fill: INK,
      });
    }
  }
  const small = spans.filter((s) => s[1] <= 60);
  for (const [sx, w, name, v, kind] of small) {
    const cx = sx + w / 2;
    el(g, "line", { x1: cx, x2: cx, y1: y + h, y2: y + h + 14, stroke: col[kind].s, "stroke-width": 1.2 });
    const lab = v < 0.001 ? `${name} <0.1%` : `${name} ${(100 * v).toFixed(1)}%`;
    text(g, cx, y + h + 26, lab, { size: 11, anchor: name === "AdamW" ? "end" : "middle", fill: col[kind].t });
  }
  const fend = x0 + ((x1 - x0) * (parts[0][1] + parts[1][1] + parts[2][1] + parts[3][1])) / sum;
  arrow(g, `M${x0},${y - 18} L${fend - 4},${y - 18}`, C.blue.s, { w: 2 });
  text(g, x0, y - 32, "forward through L loops", { size: 12, fill: C.blue.t, weight: 500 });
  arrow(g, `M${x1 - 30},${y - 18} L${fend + 4},${y - 18}`, C.violet.s, { w: 2 });
  text(g, x1 - 30, y - 32, "backward, then the optimiser", { size: 12, fill: C.violet.t, weight: 500, anchor: "end" });
  const facts = [
    "≈760 operations between grid-wide barriers",
    "bf16 tensor cores, fp32 accumulation",
    "fused attention, implicit-GEMM convolutions",
    "fixed-order reductions, no float atomics",
  ];
  let fx = 60;
  for (const f of facts)
    fx = chip(g, fx, 232, f, C.soft, { sans: true, size: 11.5, h: 26, weight: 400, pad: 10, left: true }).r + 12;
}

function heldout() {
  const g = figure(1200, 822);
  const cols = [
    ["PHercParis4 instance cube", "hero", ["ct", "sheet", "winding"], "sheet Dice 0.942 · next turn 0.771"],
    ["PHercParis4 winding set", "points", ["ct", "sheet", "winding"], "several-wrap accuracy 0.208"],
    ["PHerc0332 slice lines", "lines", ["ct", "sheet", "winding"], "recall 0.982 · drift 1.86 turns"],
    ["Kaggle surface volume", "kaggle", ["ct", "sheet", "label"], "surface Dice 0.311"],
  ];
  const rows = ["CT", "sheet", "turn"];
  const ts = 222,
    gx = 22,
    x0 = 172,
    y0 = 78;
  rows.forEach((r, j) => {
    const cy = y0 + j * (ts + 16) + ts / 2;
    const col = [C.ink, C.amber, C.violet][j];
    chip(g, 96, cy, r, col, { size: 12.5, min: 64, glow: j === 2 ? "violet" : undefined });
  });
  text(g, 96, y0 + 2 * (ts + 16) + ts / 2 + 28, "label for", {
    size: 10.5,
    italic: true,
    fill: MUTE,
    anchor: "middle",
  });
  text(g, 96, y0 + 2 * (ts + 16) + ts / 2 + 42, "Kaggle", { size: 10.5, italic: true, fill: MUTE, anchor: "middle" });
  cols.forEach(([title, key, kinds, score], i) => {
    const x = x0 + i * (ts + gx);
    text(g, x + ts / 2, 46, title, { size: 15, weight: 500, anchor: "middle" });
    kinds.forEach((k, j) => {
      const y = y0 + j * (ts + 16);
      img(g, `tiles/${key}-${k}.bmp`, x, y, ts, ts);
      if (k === "winding")
        for (const [px, py] of window.TILES[key].points)
          el(g, "circle", {
            cx: x + (px / 384) * ts,
            cy: y + (py / 384) * ts,
            r: 2.4,
            fill: "#fff",
            stroke: INK,
            "stroke-width": 0.8,
          });
      if (k === "label") tag(g, x + 5, y + 5, "label");
    });
    text(g, x + ts / 2, y0 + 3 * (ts + 16) + 8, score, { size: 12.5, anchor: "middle", fill: INK, weight: 500 });
  });
}

window.FIGS = { hero, winding, loop, training, kernel, heldout };
