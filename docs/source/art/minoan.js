((root, factory) => {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Minoan = factory();
})(typeof self !== "undefined" ? self : this, () => {
  const TAU = Math.PI * 2;

  const CLAY = [25, 56, 62];
  const CLAY_LIGHT = [31, 58, 67];
  const CLAY_RED = [21, 56, 58];
  const KILN = [20, 22, 40];
  const SLIP = [22, 26, 7];
  const SLIP_THIN = [13, 58, 26];
  const WHITE = [40, 46, 91];
  const RED = [4, 66, 39];
  const PURPLE = [350, 42, 31];
  const SAFFRON = [33, 94, 56];

  const SCENES = {
    hero: {
      size: [2400, 900],
      seed: 79,
      word: { cx: 1204, base: 722, cap: 336 },
      band: 58,
    },
    social: {
      size: [1280, 640],
      seed: 79,
      word: { cx: 640, base: 532, cap: 178 },
      band: 40,
      tall: 1.32,
    },
    banner: {
      size: [3200, 620],
      seed: 79,
      word: { cx: 1600, base: 492, cap: 232 },
      band: 42,
      wide: true,
    },
  };

  function rng(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const hashString = (s) => [...s].reduce((h, ch) => Math.imul(h ^ ch.charCodeAt(0), 16777619), 2166136261) >>> 0;

  function noise(seed) {
    const h = (ix, iy) => {
      let v = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(seed, 1442695041);
      v = Math.imul(v ^ (v >>> 13), 1274126177);
      return (((v ^ (v >>> 16)) >>> 0) / 4294967296) * 2 - 1;
    };
    return (x, y = 0) => {
      const ix = Math.floor(x),
        iy = Math.floor(y);
      const fx = x - ix,
        fy = y - iy;
      const sx = fx * fx * (3 - 2 * fx),
        sy = fy * fy * (3 - 2 * fy);
      const a = h(ix, iy),
        b = h(ix + 1, iy),
        c = h(ix, iy + 1),
        d = h(ix + 1, iy + 1);
      return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
    };
  }
  const fbm = (n, x, y, octaves = 3) => {
    let s = 0,
      amp = 0.5,
      f = 1,
      norm = 0;
    for (let i = 0; i < octaves; i++, amp *= 0.5, f *= 2.03) {
      s += amp * n(x * f + i * 17.3, y * f - i * 9.1);
      norm += amp;
    }
    return s / norm;
  };

  const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const mix = (c, d, k) => [lerp(c[0], d[0], k), lerp(c[1], d[1], k), lerp(c[2], d[2], k)];
  const smooth = (a, b, x) => {
    const t = clamp((x - a) / (b - a));
    return t * t * (3 - 2 * t);
  };
  function rgb([h, s, l]) {
    s /= 100;
    l /= 100;
    const k = (n) => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return [f(0) * 255, f(8) * 255, f(4) * 255];
  }

  function catmull(points, step = 1.4) {
    const P = [points[0], ...points, points[points.length - 1]];
    const out = [];
    for (let i = 1; i < P.length - 2; i++) {
      const [p0, p1, p2, p3] = [P[i - 1], P[i], P[i + 1], P[i + 2]];
      const n = Math.max(2, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / step));
      for (let j = 0; j < n; j++) {
        const t = j / n,
          t2 = t * t,
          t3 = t2 * t;
        out.push(
          [0, 1].map(
            (k) =>
              0.5 *
              (2 * p1[k] +
                (p2[k] - p0[k]) * t +
                (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
                (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3),
          ),
        );
      }
    }
    out.push(points[points.length - 1]);
    return out;
  }
  function bend(p0, p1, bow = 0.1, sway = 0) {
    const dx = p1[0] - p0[0],
      dy = p1[1] - p0[1];
    const len = Math.hypot(dx, dy);
    const nx = -dy / len,
      ny = dx / len;
    return catmull([
      p0,
      [p0[0] + dx * 0.33 + nx * (bow + sway) * len, p0[1] + dy * 0.33 + ny * (bow + sway) * len],
      [p0[0] + dx * 0.68 + nx * (bow - sway) * len * 0.8, p0[1] + dy * 0.68 + ny * (bow - sway) * len * 0.8],
      p1,
    ]);
  }
  const lengths = (pts) => {
    const L = [0];
    for (let i = 1; i < pts.length; i++)
      L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    return L;
  };
  const tangent = (pts, i) => {
    const a = pts[Math.max(0, i - 1)],
      b = pts[Math.min(pts.length - 1, i + 1)];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    return [(b[0] - a[0]) / d, (b[1] - a[1]) / d];
  };
  const heading = (pts, back = 5) => {
    const a = pts[Math.max(0, pts.length - 1 - back)],
      b = pts[pts.length - 1];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  };
  const frame =
    ([x, y], ang) =>
    (along, across = 0) => [
      x + Math.cos(ang) * along - Math.sin(ang) * across,
      y + Math.sin(ang) * along + Math.cos(ang) * across,
    ];
  const along = (pts, u) => {
    const L = lengths(pts);
    const s = u * L[L.length - 1];
    const i = Math.max(
      0,
      Math.min(
        pts.length - 1,
        L.findIndex((v) => v >= s),
      ),
    );
    return { p: pts[i], ang: Math.atan2(tangent(pts, i)[1], tangent(pts, i)[0]), i };
  };

  const PROFILES = {
    even: () => 1,
    taper: (t) => Math.min(1, Math.sqrt(0.25 + t * 10)) * (1 - 0.92 * t) ** 0.65,
    rise: (t) => Math.min(1, Math.sqrt(0.4 + t * 12)) * (1 - 0.58 * t),
    leaf: (t) => Math.sin(Math.PI * t) ** 0.75,
    blade: (t) => Math.sin(Math.PI * t ** 0.62) ** 0.8,
    petal: (t) => Math.sin(Math.PI * t ** 1.35) ** 0.7,
    hair: (t) => Math.sin(Math.PI * Math.min(1, 0.1 + t)) ** 0.45,
    band: (t) => 0.92 + 0.1 * Math.sin(Math.PI * t),
  };

  function rasterize(ctx, m) {
    const style = (cov, load) => `rgb(${Math.round(255 * cov)},${Math.round(255 * clamp(load))},0)`;
    if (m.blob) {
      const P = m.blob;
      ctx.beginPath();
      ctx.moveTo((P[P.length - 1][0] + P[0][0]) / 2, (P[P.length - 1][1] + P[0][1]) / 2);
      for (let i = 0; i < P.length; i++) {
        const a = P[i],
          b = P[(i + 1) % P.length];
        ctx.quadraticCurveTo(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      }
      ctx.fillStyle = style(1, m.load);
      ctx.fill();
      return;
    }
    const { pts, wl, wr, load } = m;
    const n = pts.length;
    if (n < 2) return;
    const T = pts.map((_, i) => tangent(pts, i));
    const side = (i, u) => {
      const w = u > 0 ? wl[i] : wr[i];
      return [pts[i][0] - T[i][1] * w * u, pts[i][1] + T[i][0] * w * u];
    };
    const cap = (i, sign) => {
      const tilt = m.tilt ? m.tilt[sign > 0 ? 1 : 0] : 0;
      const [tx, ty] = [
        T[i][0] * Math.cos(tilt) - T[i][1] * Math.sin(tilt),
        T[i][0] * Math.sin(tilt) + T[i][1] * Math.cos(tilt),
      ];
      const [lx, ly] = side(i, 1),
        [rx, ry] = side(i, -1);
      const cx = (lx + rx) / 2,
        cy = (ly + ry) / 2;
      const w = (wl[i] + wr[i]) / 2;
      const q = Array.isArray(m.capk) ? m.capk[sign > 0 ? 1 : 0] : (m.capk ?? 0.9);
      for (let k = 1; k < 12; k++) {
        const f = (k / 12) * Math.PI;
        const c = Math.cos(f),
          s = Math.sin(f);
        const e = lerp(0.35, 1, q);
        const cc = Math.sign(c) * Math.abs(c) ** e,
          ss = s ** e * q;
        ctx.lineTo(cx + sign * (-ty * cc + tx * ss) * w, cy + sign * (tx * cc + ty * ss) * w);
      }
    };
    const end = m.dry ? Math.max(2, Math.round((n - 1) * (1 - m.dry))) : n - 1;
    const SEG = 12;
    for (let a = 0; a < end; a += SEG) {
      const i0 = Math.max(0, a - 3),
        i1 = Math.min(end, a + SEG + 3);
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) ctx.lineTo(...side(i, 1));
      if (i1 === n - 1 && m.caps[1]) cap(i1, 1);
      for (let i = i1; i >= i0; i--) ctx.lineTo(...side(i, -1));
      if (i0 === 0 && m.caps[0]) cap(0, -1);
      ctx.closePath();
      ctx.fillStyle = style(1, load[Math.min(n - 1, a + SEG / 2)]);
      ctx.fill();
    }
    const r = rng(m.seed);
    for (let k = 0; k < m.streaks; k++) {
      const u = (r() - 0.5) * 1.5;
      const t0 = r() * 0.7,
        t1 = Math.min(1, t0 + 0.2 + r() * 0.5);
      const i0 = Math.floor(t0 * end),
        i1 = Math.floor(t1 * end);
      if (i1 - i0 < 4) continue;
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) ctx.lineTo(...side(i, u));
      ctx.lineWidth = Math.max(0.7, (wl[i0] + wr[i0]) * (0.04 + r() * 0.08));
      ctx.strokeStyle = `rgb(0,${Math.round(255 * clamp(load[i0] + lerp(m.heavy[0], m.heavy[1], r())))},0)`;
      ctx.stroke();
    }
    if (m.dry) {
      const K = m.bristles;
      for (let j = 0; j < K; j++) {
        const u = lerp(-0.92, 0.92, (j + 0.2 + 0.6 * r()) / K);
        const stop = Math.round(lerp(end, n - 1, 0.15 + 0.85 * r() ** 0.8));
        ctx.beginPath();
        for (let i = Math.max(0, end - 4); i <= stop; i++) ctx.lineTo(...side(i, u));
        ctx.lineWidth = Math.max(0.6, ((wl[end] + wr[end]) / K) * (1.3 + 0.6 * r()));
        ctx.strokeStyle = style(1, load[end] * (0.6 + 0.3 * r()));
        ctx.stroke();
      }
    }
  }

  function paints(scene, ground) {
    const k = scene.size[0] / 2400;
    const W = scene.size[0];
    const mott = noise(scene.seed + 5),
      fire = noise(scene.seed + 6),
      fleck = noise(scene.seed + 7);
    const out = [0, 0, 0, 0];
    const thick = rgb(SLIP),
      thin = rgb(SLIP_THIN);
    const slip = (cov, ld, x, y) => {
      const m = mott(x / (26 * k), y / (26 * k)) * 0.6 + mott(x / (7 * k) + 50, y / (7 * k)) * 0.4;
      const red = smooth(0.3, 0.75, fbm(fire, x / (260 * k), y / (200 * k), 2));
      const th = clamp(ld * (0.94 + 0.08 * m) - 0.16 * red);
      const c = mix(thin, thick, smooth(0.25, 0.92, th));
      out[0] = c[0];
      out[1] = c[1];
      out[2] = c[2];
      out[3] = cov * lerp(0.55, 1, smooth(0.1, 0.7, th));
      return out;
    };
    const added = (hsl, lo, flake) => {
      const full = rgb(hsl),
        thin = rgb([hsl[0], hsl[1] * 0.9, hsl[2] + 6]);
      return (cov, ld, x, y) => {
        const m = mott(x / (14 * k) + 9, y / (14 * k));
        const c = mix(thin, full, smooth(0.3, 0.9, ld + 0.08 * m));
        const l = 1 + 0.035 * m;
        out[0] = c[0] * l;
        out[1] = c[1] * l;
        out[2] = c[2] * l;
        let a = cov * lerp(lo, 0.97, smooth(0.2, 0.85, ld));
        if (flake && fleck(x / (4 * k), y / (4 * k)) > 0.72) a *= 0.35;
        out[3] = a;
        return out;
      };
    };
    const scrape = (cov, _ld, x, y) => {
      const i = (y * W + x) * 3;
      out[0] = ground[i];
      out[1] = ground[i + 1];
      out[2] = ground[i + 2];
      out[3] = cov;
      return out;
    };
    return {
      scrape,
      slip,
      line: slip,
      white: added(WHITE, 0.5, true),
      red: added(RED, 0.6, false),
      purple: added(PURPLE, 0.6, false),
      saffron: added(SAFFRON, 0.7, false),
    };
  }

  function clay(scene) {
    const [W, H] = scene.size;
    const k = W / 2400;
    const n1 = noise(scene.seed + 11),
      n2 = noise(scene.seed + 23),
      n3 = noise(scene.seed + 37),
      n4 = noise(scene.seed + 41);
    const base = rgb(CLAY),
      light = rgb(CLAY_LIGHT),
      red = rgb(CLAY_RED),
      kiln = rgb(KILN);
    const blushes = [
      [0.035, 0.62, 150, 0.22],
      [0.975, 0.3, 170, 0.18],
      [0.3, 0.86, 110, 0.12],
      [0.68, 0.2, 170, 0.13],
    ].map(([x, y, r, a]) => [x * W, y * H, r * k, a]);
    const buf = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const zone = fbm(n1, x / (330 * k), y / (260 * k), 3);
        const mid = fbm(n2, x / (70 * k), y / (55 * k), 2);
        const burnish = n3(x / (300 * k), y / (2.4 * k)) * (0.5 + 0.5 * n3(x / (700 * k) + 30, y / (150 * k)));
        let c = zone > 0 ? mix(base, light, Math.min(1, zone * 1.0)) : mix(base, red, Math.min(1, -zone * 0.9));
        let b = 0;
        for (const [bx, by, br, ba] of blushes) {
          const d = ((x - bx) ** 2 + (y - by) ** 2 * 1.6) / (br * br);
          if (d < 6) b += ba * Math.exp(-d) * (0.55 + 0.45 * n4(x / (45 * k), y / (45 * k)));
        }
        c = mix(c, kiln, Math.min(0.4, b));
        const l = 1 + 0.022 * mid + 0.012 * burnish;
        const i = (y * W + x) * 3;
        buf[i] = c[0] * l;
        buf[i + 1] = c[1] * l;
        buf[i + 2] = c[2] * l;
      }
    }
    return buf;
  }

  function grain(buf, scene) {
    const [W, H] = scene.size;
    const k = W / 2400;
    const R = rng(scene.seed + 99);
    const tooth = noise(scene.seed + 98);
    const amp = scene.grain ?? 1.4;
    for (let y = 0, i = 0; y < H; y++) {
      for (let x = 0; x < W; x++, i += 3) {
        const lum = (buf[i] + buf[i + 1] + buf[i + 2]) / 3;
        let g = amp * tooth(x / (2.6 * k), y / (2.6 * k));
        const r = R();
        if (r < 0.0005) g -= 20 + R() * 26;
        else if (r < 0.0009) g += 14 + R() * 18;
        else if (lum < 60 && r < 0.0024) g += 16 + R() * 30;
        buf[i] += g;
        buf[i + 1] += g * 0.94;
        buf[i + 2] += g * 0.86;
      }
    }
  }

  const DEPTHS = ["band", "rock", "back", "word", "front"];
  const PAINTS = ["scrape", "slip", "red", "purple", "white", "saffron", "line"];

  function compose(scene) {
    const [W, H] = scene.size;
    const k = scene.word.cap / 336;
    const layers = new Map();
    const add = (name, mark) => {
      if (!layers.has(name)) layers.set(name, []);
      layers.get(name).push(mark);
    };
    const streams = {};
    const stream = (name) => {
      const R = (streams[name] ||= rng(scene.seed ^ hashString(name)));
      return {
        R,
        rand: (a, b) => a + (b - a) * R(),
        int: () => (R() * 4294967296) >>> 0,
        pick: (list) => list[Math.floor(R() * list.length)],
      };
    };

    function pull(layer, ctrl, o = {}) {
      const S = stream(o.stream || layer);
      const base = o.dense ? ctrl : catmull(ctrl, 1.4);
      const L = lengths(base);
      const total = L[L.length - 1] || 1;
      const prof = typeof o.profile === "function" ? o.profile : PROFILES[o.profile || "taper"];
      const wob = o.wobble ?? 0.7 * k;
      const nw = noise(S.int()),
        nl = noise(S.int()),
        nr = noise(S.int());
      const w0 = o.w ?? 4 * k;
      const rough = o.rough ?? 0.07;
      const grit = o.grit ?? 0.6;
      const press = o.press ?? 0.07;
      const [l0, l1] = o.load ?? [0.95, 0.75];
      const pts = [],
        wl = [],
        wr = [],
        load = [];
      for (let i = 0; i < base.length; i++) {
        const s = L[i],
          t = s / total;
        const [tx, ty] = tangent(base, i);
        const d = wob * (nw(s / 70) + 0.35 * nw(s / 17 + 40));
        pts.push([base[i][0] - ty * d, base[i][1] + tx * d]);
        let w = w0 * prof(t) * (1 + press * nw(s / 38 + 90));
        if (o.dir) w *= o.dir(Math.atan2(ty, tx), t);
        const edge = (n) => 1 + rough * n(s / (9 * k)) + rough * grit * n(s / (2.4 * k) + 7);
        wl.push(Math.max(0, (w / 2) * edge(nl)));
        wr.push(Math.max(0, (w / 2) * edge(nr)));
        load.push(clamp(lerp(l0, l1, t) + (o.loadNoise ?? 0.07) * nw(s / 50 + 300), 0.05, 1));
      }
      const mark = {
        pts,
        wl,
        wr,
        load,
        caps: o.caps ?? [true, true],
        capk: o.capk,
        tilt: o.tilt,
        dry: o.dry ?? 0,
        bristles: o.bristles ?? 7,
        streaks: o.streaks ?? 2,
        heavy: o.heavy ?? [0.1, 0.32],
        seed: S.int(),
      };
      add(layer, mark);
      if (o.halo) {
        const h = o.halo;
        add(`${layer.split(".")[0]}.scrape`, {
          ...mark,
          wl: wl.map((v) => v + h),
          wr: wr.map((v) => v + h),
          dry: 0,
          streaks: 0,
        });
      }
      return pts;
    }

    function dot(layer, x, y, r, o = {}) {
      const S = stream(o.stream || layer);
      const ang = S.rand(0, TAU);
      const e = o.round ?? S.rand(0.8, 1);
      const blob = [];
      for (let i = 0; i < 14; i++) {
        const t = (i / 14) * TAU;
        const q = r * (1 + S.rand(-0.08, 0.08));
        blob.push([x + Math.cos(t + ang) * q, y + Math.sin(t + ang) * q * e]);
      }
      add(layer, { blob, load: o.load ?? 0.95 });
      if (o.halo)
        add(`${layer.split(".")[0]}.scrape`, {
          blob: blob.map(([px, py]) => [x + (px - x) * (1 + o.halo / r), y + (py - y) * (1 + o.halo / r)]),
          load: 1,
        });
    }

    const P = { pull, dot, stream, scene, k, W, H };
    P.word = wordLayout(scene);
    P.ground = terrain(P);
    bands(P);
    rocks(P);
    letters(P);
    flora(P);
    return layers;
  }

  function rule(P, layer, y, w, o = {}) {
    const { k, W } = P;
    const S = P.stream(`rule-${layer}-${Math.round(y)}`);
    let x = -60 * k;
    while (x < W + 60 * k) {
      const x1 = Math.min(W + 80 * k, x + S.rand(520, 950) * k);
      const dy = S.rand(-0.7, 0.7) * k;
      const ph = S.rand(0, TAU);
      const pts = [];
      for (let px = x; px <= x1; px += 10 * k) pts.push([px, y + dy + 0.7 * k * Math.sin(px / (240 * k) + ph)]);
      P.pull(layer, pts, {
        w,
        profile: "band",
        wobble: 0.5 * k,
        rough: o.rough ?? 0.05,
        caps: [false, false],
        load: o.load ?? [0.96, 0.9],
        streaks: o.streaks ?? 3,
        stream: `rule-${layer}-${Math.round(y)}`,
      });
      x = x1 - S.rand(12, 30) * k;
    }
  }

  function bands(P) {
    const { k, W, scene } = P;
    const S = P.stream("bands");
    const top = scene.band;
    rule(P, "band.slip", top / 2 - 8 * k, top + 18 * k, { rough: 0.014, streaks: 6 });
    rule(P, "band.slip", top + 15 * k, 4.2 * k);
    rule(P, "band.red", top + 25 * k, 3 * k, { load: [0.9, 0.8] });

    const y = top / 2 - 2 * k;
    const R = top * 0.34;
    const D = R * 3.05;
    const eyes = [];
    for (let x = -D * 0.7 + S.rand(0, D * 0.3); x < W + D; x += D * S.rand(0.95, 1.05))
      eyes.push([x, y + S.rand(-0.8, 0.8) * k]);
    const r0 = R * 0.9;
    const coil = ([cx, cy], start, turns, r1) => {
      const pts = [];
      const n = Math.ceil(turns * 40);
      for (let i = 0; i <= n; i++) {
        const u = i / n;
        const a = start - u * turns * TAU;
        const r = lerp(r1, r0 * 0.16, u ** 0.85);
        pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
      }
      return pts;
    };
    for (let i = 0; i < eyes.length - 1; i++) {
      const a = eyes[i],
        b = eyes[i + 1];
      const out = coil(a, -Math.PI / 2, 1.05, r0 * S.rand(0.94, 1.04)).reverse();
      const into = coil(b, Math.PI / 2, 1.05, r0 * S.rand(0.94, 1.04));
      const t = out[out.length - 1],
        bt = into[0];
      const k1 = (bt[0] - t[0]) * 0.5;
      const link = catmull([t, [t[0] + k1 * 0.6, t[1] + 2 * k], [bt[0] - k1 * 0.6, bt[1] - 2 * k], bt], 2);
      P.pull("band.white", [...out, ...link.slice(1, -1), ...into], {
        dense: true,
        w: 4.4 * k,
        wobble: 0.5 * k,
        rough: 0.08,
        load: [1, 0.7],
        streaks: 1,
        profile: (t) => Math.min(1, Math.min(t, 1 - t) * 8) ** 0.7 * (0.85 + 0.25 * Math.sin(Math.PI * t)),
      });
    }
    for (const e of eyes) P.dot("band.red", e[0], e[1], R * 0.2);
    for (let i = 0; i < eyes.length - 1; i++) {
      const mx = (eyes[i][0] + eyes[i + 1][0]) / 2;
      P.dot("band.white", mx - D * 0.17, y - R * 0.66, R * 0.09);
      P.dot("band.white", mx + D * 0.17, y + R * 0.66, R * 0.09);
    }

    const g = P.word[0].base;
    const f = g + 90 * k;
    const q0 = g + 20 * k,
      q1 = f - 22 * k;
    rule(P, "band.slip", q0, 3.6 * k);
    rule(P, "band.slip", q1, 3.6 * k);
    rule(P, "band.red", q1 + 10 * k, 3 * k, { load: [0.9, 0.8] });
    const Rp = P.stream("ripple");
    const n = noise(scene.seed + 81);
    for (let x = -10 * k; x < W + 10 * k; x += Rp.rand(6.2, 8.4) * k) {
      const pts = [];
      const top = q0 + Rp.rand(5, 7.5) * k,
        bot = q1 - Rp.rand(5, 7.5) * k;
      const slant = (0.18 + 0.1 * n(x / (300 * k))) * (bot - top);
      for (let i = 0; i <= 8; i++) {
        const t = i / 8;
        pts.push([x + slant * (t - 0.5) + 1.3 * k * Math.sin(t * 9 + x * 0.05), lerp(top, bot, t)]);
      }
      P.pull("band.slip", pts, {
        w: Rp.rand(2.1, 3) * k,
        profile: "leaf",
        streaks: 0,
        rough: 0.12,
        load: [0.9, 0.75],
        wobble: 0.3 * k,
      });
    }
    rule(P, "band.slip", (f + P.H) / 2 + 8 * k, P.H - f + 18 * k, { rough: 0.012, streaks: 6 });
    rule(P, "band.white", f + (P.H - f) * 0.42, 2.4 * k, { load: [0.85, 0.6] });
    rule(P, "band.red", f + (P.H - f) * 0.62, 3 * k, { load: [0.9, 0.8] });
  }

  function terrain(P) {
    const { k, word } = P;
    const base = word[0].base + 3 * k;
    const left = word[0].x,
      right = word[word.length - 1].right;
    const n = noise(P.scene.seed + 61);
    const hills = [
      [left - 262 * k, 112 * k, 170 * k],
      [left - 150 * k, 74 * k, 110 * k],
      [left - 360 * k, 86 * k, 120 * k],
      [left - 40 * k, 22 * k, 80 * k],
      [left - 520 * k, 70 * k, 190 * k],
      [right + 246 * k, 128 * k, 165 * k],
      [right + 130 * k, 66 * k, 105 * k],
      [right + 352 * k, 92 * k, 120 * k],
      [right + 50 * k, 26 * k, 85 * k],
      [right + 520 * k, 64 * k, 170 * k],
    ];
    const bump = (t) => (Math.abs(t) < 1 ? ((Math.cos(Math.PI * t) + 1) / 2) ** 0.9 : 0);
    const ground = (x) => {
      let h = 0;
      for (const [c, hh, w] of hills) h = Math.max(h, hh * bump((x - c) / w) * (1 + 0.14 * n(x / (34 * k) + 9)));
      return base + 2 * k * n(x / (90 * k)) - h;
    };
    ground.base = base;
    return ground;
  }

  function rocks(P) {
    const { k, W, ground } = P;
    P.stream("rocks");
    const base = ground.base;
    const pts = [];
    for (let x = -40 * k; x <= W + 40 * k; x += 6 * k) pts.push([x, ground(x)]);
    P.pull("rock.slip", pts, {
      dense: false,
      w: 7 * k,
      profile: "band",
      wobble: 0.6 * k,
      rough: 0.08,
      caps: [false, false],
      load: [0.95, 0.9],
      streaks: 3,
    });
    const course = [
      ["red", 12, 7.5],
      ["white", 22, 2.6],
      ["slip", 31, 4.5],
      ["red", 43, 6],
      ["slip", 54, 3],
      ["white", 62, 2.4],
      ["slip", 72, 6],
      ["red", 85, 5],
      ["white", 94, 2.2],
      ["slip", 103, 4],
    ];
    const n = noise(P.scene.seed + 71);
    for (const [paint, d0, w] of course) {
      const d = d0 * k;
      let run = [];
      const flush = () => {
        if (run.length * 6 > 70) {
          P.pull(`rock.${paint}`, run, {
            dense: false,
            w: w * k,
            wobble: 0.5 * k,
            rough: 0.08,
            load: [0.95, 0.85],
            streaks: 1,
            profile: (t) => Math.min(1, Math.min(t, 1 - t) * 7) ** 0.6,
          });
        }
        run = [];
      };
      for (let x = -40 * k; x <= W + 40 * k; x += 6 * k) {
        const y = ground(x) + d + 1.6 * k * n(x / (50 * k), d0 / 9);
        if (y < base - 5 * k) run.push([x, y]);
        else flush();
      }
      flush();
    }
  }

  const XH = 0.62;
  const GLYPHS = {
    H: {
      adv: 0.95,
      strokes: [
        {
          kind: "stem",
          pts: [
            [0.165, 0.985],
            [0.158, 0.5],
            [0.165, 0.015],
          ],
          serif: [
            [0.1, 0.085],
            [0.095, 0.1],
          ],
        },
        {
          kind: "stem",
          pts: [
            [0.785, 0.99],
            [0.79, 0.5],
            [0.787, 0.015],
          ],
          serif: [
            [0.095, 0.09],
            [0.095, 0.105],
          ],
        },
        {
          kind: "bar",
          w: 0.44,
          pts: [
            [0.18, 0.505],
            [0.48, 0.522],
            [0.78, 0.533],
          ],
        },
      ],
    },
    e: {
      adv: 0.64,
      strokes: [
        {
          kind: "bar",
          w: 0.4,
          pts: [
            [0.085, 0.33],
            [0.31, 0.337],
            [0.545, 0.346],
          ],
        },
        {
          kind: "bowl",
          pts: [
            [0.55, 0.35],
            [0.515, 0.515],
            [0.33, XH],
            [0.14, 0.548],
            [0.06, 0.31],
            [0.13, 0.08],
            [0.33, -0.006],
            [0.49, 0.03],
            [0.585, 0.135],
          ],
        },
      ],
    },
    r: {
      adv: 0.5,
      strokes: [
        {
          kind: "stem",
          pts: [
            [0.15, 0.605],
            [0.146, 0.3],
            [0.15, 0.015],
          ],
          serif: [
            [0.09, 0.02],
            [0.085, 0.095],
          ],
        },
        {
          kind: "arm",
          w: 0.55,
          pts: [
            [0.16, 0.43],
            [0.22, 0.555],
            [0.33, XH],
            [0.45, 0.598],
          ],
          drop: 0.075,
        },
      ],
    },
    c: {
      adv: 0.58,
      strokes: [
        {
          kind: "bowl",
          pts: [
            [0.53, 0.505],
            [0.45, 0.592],
            [0.32, XH],
            [0.13, 0.548],
            [0.06, 0.31],
            [0.13, 0.08],
            [0.33, -0.006],
            [0.48, 0.03],
            [0.565, 0.135],
          ],
          drop: 0.072,
        },
      ],
    },
    u: {
      adv: 0.68,
      strokes: [
        {
          kind: "hook",
          pts: [
            [0.135, 0.605],
            [0.13, 0.3],
            [0.17, 0.08],
            [0.32, -0.006],
            [0.47, 0.04],
            [0.563, 0.17],
          ],
          serif: [[0.09, 0.02], null],
        },
        {
          kind: "stem",
          pts: [
            [0.572, 0.605],
            [0.574, 0.3],
            [0.578, 0.015],
          ],
          serif: [
            [0.09, 0.02],
            [0.02, 0.09],
          ],
        },
      ],
    },
    l: {
      adv: 0.36,
      strokes: [
        {
          kind: "stem",
          pts: [
            [0.165, 1.025],
            [0.161, 0.5],
            [0.168, 0.015],
          ],
          serif: [
            [0.095, 0.02],
            [0.09, 0.1],
          ],
        },
      ],
    },
    X: {
      adv: 0.88,
      strokes: [
        {
          kind: "diag",
          w: 1.04,
          pts: [
            [0.1, 0.985],
            [0.45, 0.5],
            [0.8, 0.015],
          ],
          serif: [
            [0.09, 0.08],
            [0.09, 0.1],
          ],
        },
        {
          kind: "diag",
          w: 0.46,
          pts: [
            [0.79, 0.99],
            [0.45, 0.5],
            [0.11, 0.015],
          ],
          serif: [
            [0.055, 0.065],
            [0.075, 0.05],
          ],
        },
      ],
    },
  };
  const WORD = "HerculeX";
  const STEM = 0.148;
  const DRY = { 6.1: 0.075 };

  function wordLayout(scene) {
    const { cx, base, cap } = scene.word;
    const R = rng(scene.seed ^ 0x5eed);
    const width = [...WORD].reduce((s, ch) => s + GLYPHS[ch].adv, 0) * cap;
    let x = cx - width / 2;
    return [...WORD].map((ch, i) => {
      const x0 = x;
      const adv = GLYPHS[ch].adv;
      x += adv * cap;
      const rot = (R() - 0.5) * 0.022;
      const dy = (R() - 0.5) * 0.012 * cap;
      const sc = 1 + (R() - 0.5) * 0.025;
      const mx = x0 + (adv * cap) / 2;
      const at = (u, v) => {
        const lx = (u - adv / 2) * cap * sc,
          ly = -v * cap * sc;
        return [mx + lx * Math.cos(rot) - ly * Math.sin(rot), base + dy + lx * Math.sin(rot) + ly * Math.cos(rot)];
      };
      return { ch, i, x: x0, right: x, base, cap, at };
    });
  }

  function letters(P) {
    const S = P.stream("letters");
    const contrast = (lo) => (ang) => lerp(lo, 1, Math.abs(Math.sin(ang)) ** 0.9);
    const profiles = {
      stem: (t) => 1 - 0.07 * Math.sin(Math.PI * t),
      diag: (t) => 1 - 0.06 * Math.sin(Math.PI * t),
      bar: (t) => Math.min(1, 0.8 + t * 3) * (1 - 0.12 * t) * (1 - 0.06 * Math.sin(Math.PI * t)),
      bowl: (t) => Math.min(1, 0.62 + t * 4) * (1 - 0.5 * smooth(0.8, 1, t)),
      hook: (t) => 1 - 0.38 * smooth(0.82, 1, t),
      arm: (t) => 0.42 + 0.58 * smooth(0, 0.45, t),
    };
    const style = {
      wobble: 0,
      press: 0.05,
      rough: 0.026,
      grit: 0.25,
      loadNoise: 0.06,
      streaks: 3,
      heavy: [0.04, 0.12],
    };
    for (const g of P.word) {
      const cap = g.cap;
      for (const [si, st] of GLYPHS[g.ch].strokes.entries()) {
        const uv = st.pts.map(([u, v]) => [u + S.rand(-0.004, 0.004), v + S.rand(-0.003, 0.003)]);
        const w = cap * STEM * (st.w ?? 1) * S.rand(0.97, 1.03);
        const pts = P.pull(
          "word.slip",
          uv.map(([u, v]) => g.at(u, v)),
          {
            ...style,
            w,
            wobble: cap * 0.004,
            profile: profiles[st.kind],
            dir: st.kind === "bowl" ? contrast(0.36) : st.kind === "hook" ? contrast(0.42) : undefined,
            capk: st.serif ? [st.serif[0] ? 0.4 : 0.85, st.serif[1] ? 0.4 : 0.85] : 0.85,
            load: [1, S.rand(0.8, 0.9)],
            dry: DRY[`${g.i}.${si}`],
            bristles: 16,
          },
        );
        (st.serif || []).forEach((reach, end) => {
          if (!reach) return;
          const [u, v] = end ? uv[uv.length - 1] : uv[0];
          const dir = end ? -1 : 1;
          const [l, r] = reach.map((q) => q * S.rand(0.88, 1.12));
          const lift = S.rand(-0.008, 0.012);
          const tick = [
            [u - l, v + dir * (0.012 + lift)],
            [u - l * 0.3, v + dir * 0.004],
            [u + r * 0.3, v + dir * 0.002],
            [u + r, v + dir * (0.013 - lift)],
          ];
          const tl = l + r;
          const mid = l / tl;
          P.pull(
            "word.slip",
            tick.map(([a, b]) => g.at(a, b)),
            {
              ...style,
              wobble: cap * 0.002,
              w: cap * S.rand(0.052, 0.062) * (st.w ? Math.sqrt(st.w) : 1),
              profile: (t) => Math.sin(Math.PI * clamp(t)) ** 0.55 * (1 - 0.25 * Math.abs(t - mid)),
              load: [0.95, 0.95],
            },
          );
        });
        if (st.drop) {
          const e = st.kind === "arm" ? pts[pts.length - 1] : pts[0];
          const [tx, ty] = st.kind === "arm" ? tangent(pts, pts.length - 1) : tangent(pts, 0).map((v) => -v);
          const r = st.drop * cap * S.rand(0.95, 1.05);
          P.dot("word.slip", e[0] - tx * r * 0.45 + ty * r * 0.25, e[1] - ty * r * 0.45 - tx * r * 0.25, r, {
            round: 0.92,
          });
        }
      }
    }
  }

  function blade(P, layer, p, ang, len, w, droop = 0.2, o = {}) {
    const pts = [];
    for (let i = 0; i <= 4; i++) {
      const t = i / 4;
      pts.push([p[0] + Math.cos(ang) * len * t, p[1] + Math.sin(ang) * len * t + droop * len * t * t]);
    }
    return P.pull(layer, pts, { w, profile: "blade", load: [0.95, 0.7], streaks: 1, ...o });
  }
  function leaf(P, layer, p, ang, len, w, curl = 0.15, o = {}) {
    const at = frame(p, ang);
    return P.pull(layer, [at(0), at(len * 0.5, len * curl * 0.45), at(len, len * curl)], {
      w,
      profile: "leaf",
      load: [0.95, 0.75],
      streaks: 1,
      ...o,
    });
  }

  function lily(P, depth, name, p, ang, s, o = {}) {
    const S = P.stream(name);
    const at = frame(p, ang);
    const paint = `${depth}.${o.paint || "red"}`;
    const line = `${depth}.line`;
    const flare = o.flare ?? 1;
    const halo = o.halo || 0;
    const lean = S.rand(-0.07, 0.07);
    const tips = [];
    for (const side of S.R() < 0.5 ? [-1, 1] : [1, -1]) {
      const c = flare * S.rand(0.8, 1.15);
      const reach = S.rand(0.9, 1.1);
      const curl = S.rand(0.85, 1.15);
      const pts = [
        at(0, side * 0.03 * s),
        at(0.3 * s * reach, side * 0.12 * s * c),
        at(0.6 * s * reach, side * 0.32 * s * c),
        at(0.74 * s * reach, side * (0.5 + 0.05 * curl) * s * c),
        at((0.74 - 0.12 * curl) * s * reach, side * (0.6 + 0.1 * curl) * s * c),
      ];
      tips.push(pts);
      P.pull(paint, pts, { w: 0.27 * s, profile: "petal", halo, load: [1, 0.72], streaks: 2 });
    }
    const mid = [at(0, 0), at(0.5 * s, lean * s), at(0.98 * s, lean * 2.4 * s)];
    P.pull(paint, mid, { w: 0.25 * s, profile: "petal", halo, load: [1, 0.78], streaks: 2 });
    P.pull(line, [at(-0.2 * s, 0), at(0.13 * s, 0)], { w: 0.15 * s, profile: "leaf", load: [1, 0.9], streaks: 0 });
    if (o.paint === "white") {
      for (const pts of tips) P.pull(line, pts.slice(0, 4), { w: 0.03 * s, profile: "hair", streaks: 0 });
      P.pull(line, [at(0.1 * s, 0), at(0.5 * s, lean * s * 0.9), at(0.8 * s, lean * 2 * s)], {
        w: 0.03 * s,
        profile: "hair",
        streaks: 0,
      });
    }
    const n = o.stamens ?? 4;
    for (let i = 0; i < n; i++) {
      const q = lerp(-1, 1, i / (n - 1)) + S.rand(-0.2, 0.2);
      const len = s * S.rand(1.02, 1.28);
      const end = at(len - Math.abs(q) * 0.12 * s, q * 0.34 * s + lean * 2 * s);
      P.pull(line, [at(0.12 * s, q * 0.02 * s), at(len * 0.6, q * 0.15 * s + lean * s), end], {
        w: 0.032 * s,
        profile: "hair",
        streaks: 0,
      });
      P.dot(o.anther ? `${depth}.${o.anther}` : line, end[0], end[1], 0.052 * s * S.rand(0.85, 1.15));
    }
  }

  function bud(P, depth, name, p, ang, s, paint = "red", halo = 0) {
    const S = P.stream(name);
    const at = frame(p, ang);
    const bow = S.rand(-0.06, 0.06);
    P.pull(`${depth}.${paint}`, [at(-0.04 * s), at(0.5 * s, bow * s), at(s, bow * 2 * s)], {
      w: 0.3 * s,
      profile: "petal",
      halo,
      load: [1, 0.8],
    });
    P.pull(`${depth}.line`, [at(0.15 * s, 0.03 * s), at(0.55 * s, bow * s + 0.02 * s), at(0.9 * s, bow * 1.8 * s)], {
      w: 0.028 * s,
      profile: "hair",
      streaks: 0,
    });
    P.pull(`${depth}.line`, [at(-0.2 * s), at(0.12 * s)], { w: 0.14 * s, profile: "leaf", streaks: 0 });
  }

  function crocus(P, depth, name, p, ang, s, o = {}) {
    const S = P.stream(name);
    const at = frame(p, ang);
    const paint = `${depth}.${o.paint || "white"}`;
    const halo = o.halo || 0;
    for (const side of [-1, 1]) {
      const c = S.rand(0.85, 1.2);
      P.pull(
        paint,
        [
          at(0, side * 0.03 * s),
          at(0.3 * s, side * 0.15 * s * c),
          at(0.62 * s, side * 0.27 * s * c),
          at(0.84 * s, side * 0.31 * s * c),
        ],
        { w: 0.25 * s, profile: "petal", halo, load: [1, 0.8] },
      );
    }
    P.pull(paint, [at(0, 0), at(0.45 * s, S.rand(-0.03, 0.03) * s), at(0.9 * s, S.rand(-0.06, 0.06) * s)], {
      w: 0.27 * s,
      profile: "petal",
      halo,
      load: [1, 0.85],
    });
    P.pull(`${depth}.line`, [at(-0.14 * s, 0), at(0.1 * s, 0)], { w: 0.13 * s, profile: "leaf", streaks: 0 });
    for (const side of [-1, 1]) {
      P.pull(
        `${depth}.line`,
        [at(0.08 * s, side * 0.06 * s), at(0.42 * s, side * 0.13 * s), at(0.7 * s, side * 0.16 * s)],
        { w: 0.026 * s, profile: "hair", streaks: 0 },
      );
    }
    const thread = o.thread || "red";
    for (let i = 0; i < 3; i++) {
      const q = i - 1 + S.rand(-0.2, 0.2);
      const end = at(s * S.rand(0.86, 1.04) - Math.abs(q) * 0.22 * s, q * S.rand(0.42, 0.56) * s);
      P.pull(`${depth}.${thread}`, [at(0.32 * s, q * 0.05 * s), at(0.72 * s, q * 0.2 * s), end], {
        w: 0.05 * s,
        profile: (t) => 0.55 + 0.45 * t,
        streaks: 0,
        load: [1, 1],
      });
      P.dot(`${depth}.${thread}`, end[0], end[1], 0.06 * s);
    }
  }

  function crocusTuft(P, depth, name, base, s, n, o = {}) {
    const S = P.stream(name);
    const lean = o.lean ?? 0;
    const halo = o.halo || 0;
    const blades = 4 + n * 2 + Math.floor(S.R() * 3);
    for (let i = 0; i < blades; i++) {
      const t = i / (blades - 1) - 0.5;
      const ang = -Math.PI / 2 + lean * 0.3 + t * S.rand(1.3, 1.9) + S.rand(-0.14, 0.14);
      blade(
        P,
        `${depth}.slip`,
        [base[0] + t * s * 0.22, base[1] + 2],
        ang,
        s * S.rand(0.55, 1.15) * (1 - Math.abs(t) * 0.55),
        s * S.rand(0.05, 0.075),
        Math.sign(t) * S.rand(0.1, 0.45) + 0.1,
        { halo },
      );
    }
    for (let j = 0; j < n; j++) {
      const dx = (j - (n - 1) / 2) * s * 0.34 + S.rand(-0.08, 0.08) * s;
      const h = s * S.rand(0.48, 0.82) * (j === Math.floor(n / 2) ? 1.12 : 1);
      const tip = [base[0] + dx + lean * h * 0.35 + dx * 0.3, base[1] - h];
      const stalk = bend([base[0] + dx * 0.3, base[1]], tip, S.rand(-0.12, 0.12));
      P.pull(`${depth}.slip`, stalk, { dense: true, w: s * 0.045, profile: "rise", halo });
      const paint = o.paints ? o.paints[j % o.paints.length] : "white";
      crocus(P, depth, `${name}-${j}`, tip, heading(stalk) + S.rand(-0.25, 0.25), s * S.rand(0.42, 0.55), {
        halo,
        paint,
      });
    }
  }

  function papyrus(P, depth, name, base, top, s, o = {}) {
    const S = P.stream(name);
    const slip = `${depth}.slip`;
    const stalk = bend(base, top, o.bow ?? S.rand(-0.07, 0.07), S.rand(-0.03, 0.03));
    P.pull(slip, stalk, { dense: true, w: s * 0.1, profile: "rise", load: [1, 0.75] });
    const ang = heading(stalk) + S.rand(-0.08, 0.08);
    const at = frame(top, ang);
    const pt = (r, a) => [top[0] + Math.cos(a) * r, top[1] + Math.sin(a) * r];
    for (const side of [-1, 1]) {
      P.pull(slip, [at(-0.36 * s, side * 0.02 * s), at(-0.1 * s, side * 0.13 * s), at(0.08 * s, side * 0.27 * s)], {
        w: s * 0.14,
        profile: "blade",
      });
    }
    const spread = S.rand(0.62, 0.8);
    const n = 7;
    for (let i = 0; i < n; i++) {
      const a = ang + lerp(-spread, spread, i / (n - 1)) + S.rand(-0.04, 0.04);
      P.pull(slip, [pt(s * 0.04, a), pt(s * S.rand(0.36, 0.42), a + S.rand(-0.03, 0.03))], {
        w: s * 0.17,
        profile: (t) => 0.3 + 0.7 * t ** 0.5,
        load: [0.85, 1],
        capk: 0.6,
      });
    }
    const m = 19 + Math.floor(S.R() * 5);
    for (let i = 0; i < m; i++) {
      const a = ang + lerp(-spread * 1.18, spread * 1.18, i / (m - 1)) + S.rand(-0.025, 0.025);
      const r1 = s * S.rand(0.9, 1.05) * (1 - 0.12 * (Math.abs(i / (m - 1) - 0.5) * 2) ** 2);
      P.pull(slip, [pt(s * 0.3, a), pt(r1 * 0.6, a + S.rand(-0.02, 0.02)), pt(r1, a + S.rand(-0.05, 0.05))], {
        w: s * 0.03,
        profile: "hair",
        streaks: 0,
      });
      P.dot(slip, ...pt(r1 + s * 0.05, a), s * 0.034);
    }
    for (let i = 0; i < 5; i++) {
      const a = ang + lerp(-spread * 0.75, spread * 0.75, (i + 0.5) / 5);
      P.dot(`${depth}.white`, ...pt(s * 0.3, a), s * 0.032);
    }
  }

  function reed(P, depth, name, base, top, s) {
    const S = P.stream(name);
    const stalk = bend(base, top, S.rand(-0.08, 0.08));
    P.pull(`${depth}.slip`, stalk, { dense: true, w: s * 0.075, profile: "rise" });
    let side = S.R() < 0.5 ? -1 : 1;
    for (let u = S.rand(0.08, 0.16); u < 0.9; u += S.rand(0.1, 0.17)) {
      const { p, ang } = along(stalk, u);
      const la = ang + side * S.rand(0.35, 0.6);
      blade(
        P,
        `${depth}.slip`,
        p,
        la,
        s * lerp(2.1, 1, u) * S.rand(0.85, 1.12),
        s * 0.1,
        S.rand(0.25, 0.55) * (Math.cos(la) * side > 0 ? 1 : 0.5),
      );
      side = -side;
    }
    blade(P, `${depth}.slip`, top, heading(stalk) + S.rand(-0.2, 0.2), s * 0.7, s * 0.07, 0.15);
  }

  function seaDaffodil(P, depth, name, p, ang, s) {
    const S = P.stream(name);
    const at = frame(p, ang);
    const white = `${depth}.white`,
      line = `${depth}.line`;
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * TAU + S.rand(-0.15, 0.15) + 0.3;
      const tip = at(0.28 * s + Math.cos(a) * s * 0.42, Math.sin(a) * s * S.rand(0.92, 1.08));
      P.pull(white, [at(0.2 * s, 0), [(at(0.24 * s)[0] + tip[0]) / 2, (at(0.24 * s)[1] + tip[1]) / 2], tip], {
        w: 0.13 * s,
        profile: "leaf",
        load: [1, 0.75],
      });
    }
    P.pull(white, [at(0), at(0.3 * s), at(0.55 * s)], {
      w: 0.36 * s,
      profile: (t) => 0.4 + 0.6 * t ** 0.8,
      caps: [true, false],
    });
    const rim = [];
    for (let i = 0; i <= 8; i++) rim.push(at(0.55 * s + (i % 2 ? 0.05 * s : -0.01 * s), lerp(-0.19, 0.19, i / 8) * s));
    P.pull(line, rim, { w: 0.026 * s, profile: "even", streaks: 0, wobble: 0.2 });
    for (const side of [-1, 1])
      P.pull(line, [at(0.03 * s, side * 0.07 * s), at(0.52 * s, side * 0.17 * s)], {
        w: 0.026 * s,
        profile: "hair",
        streaks: 0,
      });
    for (let i = 0; i < 3; i++) {
      const end = at(0.78 * s, (i - 1) * 0.12 * s);
      P.pull(line, [at(0.3 * s, (i - 1) * 0.03 * s), end], { w: 0.024 * s, profile: "hair", streaks: 0 });
      P.dot(`${depth}.saffron`, ...end, 0.04 * s);
    }
    P.pull(line, [at(-0.18 * s), at(0.04 * s)], { w: 0.12 * s, profile: "leaf", streaks: 0 });
  }

  function madonna(P, depth, name, base, top, s, flowers) {
    const S = P.stream(name);
    const stalk = bend(base, top, S.rand(-0.05, 0.05), 0.02);
    P.pull(`${depth}.slip`, stalk, { dense: true, w: s * 0.11, profile: "rise" });
    let side = S.R() < 0.5 ? -1 : 1;
    for (let u = 0.04; u < 0.7; u += S.rand(0.045, 0.07)) {
      const { p, ang } = along(stalk, u);
      leaf(
        P,
        `${depth}.slip`,
        p,
        ang + side * S.rand(0.42, 0.62),
        s * lerp(1.15, 0.55, u / 0.7) * S.rand(0.85, 1.1),
        s * 0.13,
        side * S.rand(0.1, 0.24),
      );
      side = -side;
    }
    flowers.forEach(([u, sd, droop, size], i) => {
      const { p, ang } = along(stalk, u);
      const out = ang + sd * (Math.PI / 2 - droop);
      const tip = [p[0] + Math.cos(out) * s * 0.42, p[1] + Math.sin(out) * s * 0.42];
      const ped = bend(p, tip, -sd * 0.12);
      P.pull(`${depth}.slip`, ped, { dense: true, w: s * 0.06, profile: "rise" });
      lily(P, depth, `${name}-${i}`, tip, heading(ped) + sd * 0.12, s * size, {
        paint: "white",
        flare: 0.62,
        anther: "saffron",
        stamens: 3,
      });
    });
    bud(P, depth, `${name}-bud`, top, heading(stalk) + S.rand(-0.15, 0.15), s * 0.62, "white");
    const { p, ang } = along(stalk, 0.93);
    const bt = [p[0] + Math.cos(ang - side * 0.7) * s * 0.3, p[1] + Math.sin(ang - side * 0.7) * s * 0.3];
    bud(P, depth, `${name}-bud2`, bt, ang - side * 0.55, s * 0.5, "white");
  }

  function akrotiri(P, depth, name, base, heads, s, o = {}) {
    const S = P.stream(name);
    const halo = o.halo || 0;
    for (let i = 0; i < 3; i++)
      blade(
        P,
        `${depth}.slip`,
        base,
        -Math.PI / 2 + (i - 1) * 0.55 + S.rand(-0.1, 0.1),
        s * S.rand(0.9, 1.3),
        s * 0.13,
        (i - 1) * 0.25,
        { halo },
      );
    heads.forEach(([x, y, kind, sway], i) => {
      const stem = bend(base, [x, y], sway ?? S.rand(-0.12, 0.12), S.rand(-0.05, 0.05));
      P.pull(`${depth}.slip`, stem, { dense: true, w: s * 0.085, profile: "rise", halo });
      const { p, ang } = along(stem, S.rand(0.35, 0.55));
      const sd = Math.cos(ang) > 0 ? -1 : 1;
      blade(P, `${depth}.slip`, p, ang + sd * S.rand(0.35, 0.5), s * S.rand(0.8, 1.1), s * 0.11, 0.15, { halo });
      const h = heading(stem);
      if (kind === "bud") bud(P, depth, `${name}-${i}`, [x, y], h + S.rand(-0.15, 0.15), s * 0.62, "red", halo);
      else lily(P, depth, `${name}-${i}`, [x, y], h + S.rand(-0.12, 0.12), s * S.rand(0.92, 1.08), { halo });
    });
  }

  function ribbon(P, depth, name, p, len, s, sway) {
    const S = P.stream(name);
    const pts = [];
    const ph = S.rand(0, TAU);
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      pts.push([p[0] + sway * len * t * 0.5 + Math.sin(ph + t * 5.5) * s * 0.9 * t, p[1] + len * t]);
    }
    P.pull(`${depth}.purple`, pts, {
      w: s,
      profile: (t) => 1 - 0.35 * t,
      halo: s * 0.3,
      load: [1, 0.8],
      caps: [false, true],
    });
    P.dot(`${depth}.white`, ...pts[8], s * 0.32);
  }

  function garland(P, name, a, b, sag, s, o = {}) {
    const S = P.stream(name);
    const pts = [];
    const n = 60;
    const shape = o.shape ?? 1;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      pts.push([
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t) + sag * Math.sin(Math.PI * t) ** shape * (1 + 0.12 * (t - 0.5)),
      ]);
    }
    const cord = catmull(pts, 1.5);
    P.pull("back.slip", cord, { dense: true, w: s * 0.09, profile: "even", streaks: 0 });
    const L = lengths(cord),
      total = L[L.length - 1];
    let side = 1;
    for (let d = s * 0.12; d < total - s * 0.1; d += s * S.rand(0.17, 0.36)) {
      const i = L.findIndex((v) => v >= d);
      const [tx, ty] = tangent(cord, i);
      const fwd = d < total * 0.5 ? 1 : -1;
      side = S.R() < 0.8 ? -side : side;
      const ang = Math.atan2(ty * fwd, tx * fwd) + side * S.rand(0.5, 0.8);
      leaf(P, "back.slip", cord[i], ang, s * S.rand(0.42, 0.62), s * 0.19, -side * S.rand(0.04, 0.16));
      if (S.R() < 0.22)
        P.dot("back.white", cord[i][0] + ty * s * 0.22 * side, cord[i][1] - tx * s * 0.22 * side, s * 0.055);
    }
    const kinds = o.kinds ?? ["crocus", "lily", "crocus", "bud"];
    let j = 0;
    const place = (d, size) => {
      const i = L.findIndex((v) => v >= d);
      if (i < 0) return;
      const [tx, ty] = tangent(cord, i);
      let ang = Math.atan2(ty, tx) + (j % 2 ? -1 : 1) * S.rand(1.1, 1.9);
      if (Math.sin(ang) > -0.15) ang = -Math.PI / 2 + S.rand(-0.9, 0.9);
      const p = cord[i];
      const kind = kinds[j % kinds.length];
      if (kind === "crocus")
        crocus(P, "back", `${name}-${j}`, p, ang, size * 1.05, { paint: S.R() < 0.3 ? "purple" : "white" });
      else if (kind === "bud") bud(P, "back", `${name}-${j}`, p, ang, size * 0.85, "red");
      else lily(P, "back", `${name}-${j}`, p, ang, size, { stamens: 3 });
      j++;
    };
    for (let d = s * S.rand(0.4, 0.9); d < total - s * 0.45; d += s * S.rand(0.6, 1.5)) {
      const size = s * (S.R() < 0.25 ? S.rand(1.1, 1.3) : S.rand(0.62, 0.95));
      place(d, size);
      if (S.R() < 0.22 && d + s * 0.4 < total - s * 0.45) {
        d += s * S.rand(0.3, 0.45);
        place(d, s * S.rand(0.55, 0.8));
      }
    }
    return cord;
  }

  function knot(P, _name, p, ang, w, s) {
    const at = frame(p, ang);
    const h = w * 0.62;
    for (const off of [-0.3, 0.32]) {
      P.pull("front.purple", [at(off * s, -h), at(off * s + 0.05 * s, 0), at(off * s, h)], {
        w: s * 0.32,
        profile: (t) => 0.8 + 0.2 * Math.sin(Math.PI * t),
        halo: s * 0.1,
        load: [1, 0.85],
      });
    }
  }

  function swallow(P, name, p, ang, s, flip = 1) {
    const S = P.stream(name);
    const at = frame(p, ang);
    const slip = "back.slip";
    P.pull(slip, [at(-0.5 * s, 0), at(-0.1 * s, 0.02 * s * flip), at(0.3 * s, 0)], {
      w: 0.24 * s,
      profile: (t) => Math.sin(Math.PI * t ** 0.75) ** 0.6,
      load: [0.9, 1],
    });
    P.dot(slip, ...at(0.32 * s, -0.01 * s * flip), 0.1 * s);
    P.pull(slip, [at(0.38 * s, -0.02 * s * flip), at(0.5 * s, 0)], { w: 0.05 * s, profile: "taper", streaks: 0 });
    for (const [side, spread, len] of [
      [1, 0.32, 0.6],
      [-1, 0.22, 0.5],
    ]) {
      P.pull(
        slip,
        [
          at(-0.42 * s, 0),
          at(-0.42 * s - len * s * 0.5, side * spread * s * 0.4 * flip),
          at(-0.42 * s - len * s, side * spread * s * flip),
        ],
        { w: 0.07 * s, profile: "taper", streaks: 0 },
      );
    }
    const beat = S.rand(0.8, 1.15);
    P.pull(
      slip,
      [
        at(0.05 * s, -0.04 * s * flip),
        at(-0.2 * s, -0.42 * s * flip * beat),
        at(-0.6 * s, -0.78 * s * flip * beat),
        at(-0.85 * s, -0.86 * s * flip * beat),
      ],
      { w: 0.2 * s, profile: "blade", load: [0.95, 0.75] },
    );
    P.pull(
      slip,
      [
        at(0.02 * s, 0.06 * s * flip),
        at(-0.25 * s, 0.32 * s * flip),
        at(-0.62 * s, 0.5 * s * flip),
        at(-0.9 * s, 0.52 * s * flip),
      ],
      { w: 0.18 * s, profile: "blade", load: [0.95, 0.75] },
    );
    P.pull(
      "back.white",
      [at(-0.3 * s, 0.06 * s * flip), at(0.05 * s, 0.07 * s * flip), at(0.22 * s, 0.05 * s * flip)],
      { w: 0.07 * s, profile: "leaf", streaks: 0 },
    );
    P.dot("back.red", ...at(0.3 * s, 0.06 * s * flip), 0.055 * s);
  }

  function flora(P) {
    const { k, word, ground } = P;
    const [H, e, r, c, u, l, e2, X] = word;
    const cap = H.cap;
    const left = H.x,
      right = X.right;
    const on = (x) => [x, ground(x) + 2 * k];
    const tall = P.scene.tall ?? 1;
    const up = (h) => H.base - h * k * tall;

    reed(P, "back", "reed-left", on(left - 352 * k), [left - 395 * k, up(610)], 60 * k);
    papyrus(P, "back", "papyrus-1", on(left - 262 * k), [left - 350 * k, up(420)], 60 * k);
    papyrus(P, "back", "papyrus-2", on(left - 240 * k), [left - 262 * k, up(505)], 76 * k);
    papyrus(P, "back", "papyrus-3", on(left - 222 * k), [left - 160 * k, up(470)], 64 * k);
    papyrus(P, "back", "papyrus-4", on(left - 205 * k), [left - 118 * k, up(318)], 48 * k);
    crocusTuft(P, "back", "tuft-left", on(left - 300 * k), 76 * k, 3, {
      lean: -0.3,
      paints: ["white", "purple", "white"],
    });
    akrotiri(
      P,
      "front",
      "akrotiri-left",
      on(left - 98 * k),
      [
        [left - 170 * k, up(300), "lily", 0.1],
        [left - 64 * k, up(395), "lily", -0.06],
        [left + 44 * k, up(250), "bud", -0.2],
      ],
      66 * k,
      { halo: 3 * k },
    );
    crocusTuft(P, "front", "tuft-h", on(left - 20 * k), 56 * k, 2, { lean: 0.4, halo: 3 * k });

    const hs = catmull([
      H.at(0.47, -0.01),
      H.at(0.42, 0.22),
      H.at(0.5, 0.5),
      H.at(0.43, 0.78),
      H.at(0.47, 1.05),
      H.at(0.56, 1.3),
    ]);
    P.pull("back.slip", hs, { dense: true, w: 7 * k, profile: "rise" });
    for (const [uu, sd, len] of [
      [0.12, -1, 0.5],
      [0.26, 1, 0.55],
      [0.55, -1, 0.5],
      [0.68, 1, 0.42],
    ]) {
      const { p, ang } = along(hs, uu);
      leaf(P, "back.slip", p, ang + sd * 0.62, len * 0.36 * cap, 11 * k, -sd * 0.12);
    }
    lily(P, "back", "lily-h", H.at(0.56, 1.3), heading(hs) + 0.22, 74 * k);
    const hb = along(hs, 0.8);
    const hbt = H.at(0.3, 1.2);
    const hbs = bend(hb.p, hbt, 0.12);
    P.pull("back.slip", hbs, { dense: true, w: 5 * k, profile: "rise" });
    bud(P, "back", "bud-h", hbt, heading(hbs) - 0.1, 46 * k);

    const xs = catmull([X.at(0.45, -0.01), X.at(0.4, 0.22), X.at(0.46, 0.5), X.at(0.42, 0.74), X.at(0.45, 0.9)]);
    P.pull("back.slip", xs, { dense: true, w: 6.5 * k, profile: "rise" });
    for (const [uu, sd, len] of [
      [0.1, -1, 0.4],
      [0.2, 1, 0.42],
      [0.62, -1, 0.3],
    ]) {
      const { p, ang } = along(xs, uu);
      leaf(P, "back.slip", p, ang + sd * 0.55, len * 0.36 * cap, 10 * k, -sd * 0.1);
    }
    const xt = along(xs, 1);
    const x1 = bend(xt.p, X.at(0.36, 1.12), 0.1);
    const x2 = bend(xt.p, X.at(0.6, 1.16), -0.1);
    P.pull("back.slip", x1, { dense: true, w: 5 * k, profile: "rise" });
    P.pull("back.slip", x2, { dense: true, w: 5 * k, profile: "rise" });
    lily(P, "back", "lily-x1", X.at(0.36, 1.12), heading(x1) - 0.2, 60 * k, {
      paint: "white",
      flare: 0.7,
      anther: "saffron",
      stamens: 3,
    });
    lily(P, "back", "lily-x2", X.at(0.6, 1.16), heading(x2) + 0.1, 66 * k, {
      paint: "white",
      flare: 0.7,
      anther: "saffron",
      stamens: 3,
    });
    bud(P, "back", "bud-x", X.at(0.47, 1.28), -Math.PI / 2 + 0.1, 44 * k, "white");
    P.pull("back.slip", bend(xt.p, X.at(0.47, 1.28), 0.04), { dense: true, w: 4 * k, profile: "rise" });

    const ga = H.at(0.8, 0.86),
      gb = l.at(0.16, 0.9),
      gc = X.at(0.27, 0.8);
    garland(P, "garland-1", ga, [gb[0] - 0.05 * cap, gb[1]], 0.19 * cap, 46 * k);
    garland(P, "garland-2", [gb[0] + 0.05 * cap, gb[1]], gc, 0.15 * cap, 46 * k, { kinds: ["lily", "crocus", "bud"] });
    knot(P, "knot-h", H.at(0.782, 0.86), Math.PI / 2, cap * STEM, 20 * k);
    ribbon(P, "front", "ribbon-h1", H.at(0.84, 0.84), 0.2 * cap, 9 * k, 0.3);
    ribbon(P, "front", "ribbon-h2", H.at(0.87, 0.85), 0.13 * cap, 8 * k, 0.9);

    const mx = (H.at(0.8, 0)[0] + l.at(0.16, 0)[0]) / 2;
    const sy = H.base - 1.52 * cap - (tall - 1) * 0.5 * cap;
    swallow(P, "swallow-1", [mx - 70 * k, sy + 12 * k], -0.32, 86 * k, 1);
    swallow(P, "swallow-2", [mx + 64 * k, sy - 4 * k], Math.PI + 0.42, 80 * k, -1);

    crocusTuft(P, "front", "tuft-er", on((e.right + r.x) / 2 + 6 * k), 42 * k, 1, { lean: -0.35, halo: 3 * k });
    crocusTuft(P, "front", "tuft-cu", on((c.right + u.x) / 2 + 4 * k), 54 * k, 2, {
      lean: 0.2,
      halo: 3 * k,
      paints: ["white", "purple"],
    });
    crocusTuft(P, "front", "tuft-le", on((l.right + e2.x) / 2 - 2 * k), 66 * k, 3, {
      lean: -0.15,
      halo: 3 * k,
      paints: ["purple", "white", "white"],
    });

    crocusTuft(P, "front", "tuft-x", on(right + 26 * k), 58 * k, 2, {
      lean: -0.3,
      halo: 3 * k,
      paints: ["white", "purple"],
    });
    madonna(P, "back", "madonna", on(right + 130 * k), [right + 108 * k, up(540)], 60 * k, [
      [0.66, 1, 0.25, 1.05],
      [0.74, -1, 0.15, 1.1],
      [0.83, 1, 0.05, 1],
    ]);
    const sb = on(right + 242 * k);
    for (const [dx, h, len] of [
      [-30, 250, 1],
      [40, 190, 0.8],
      [5, 300, 1],
    ]) {
      const tip = [sb[0] + dx * k, up(h)];
      const st = bend(sb, tip, dx > 0 ? -0.08 : 0.08);
      P.pull("back.slip", st, { dense: true, w: 6 * k, profile: "rise" });
      seaDaffodil(P, "back", `daffodil-${h}`, tip, heading(st) + (dx > 0 ? 0.5 : -0.4), 58 * k * len);
    }
    for (let i = 0; i < 4; i++)
      blade(P, "back.slip", sb, -Math.PI / 2 + (i - 1.5) * 0.38, 120 * k, 11 * k, (i - 1.5) * 0.12);
    papyrus(P, "back", "papyrus-5", on(right + 330 * k), [right + 300 * k, up(470)], 64 * k);
    papyrus(P, "back", "papyrus-6", on(right + 350 * k), [right + 392 * k, up(575)], 74 * k);
    reed(P, "back", "reed-right", on(right + 440 * k), [right + 470 * k, up(520)], 56 * k);
    crocusTuft(P, "back", "tuft-right", on(right + 300 * k), 70 * k, 3, {
      lean: 0.3,
      paints: ["white", "white", "purple"],
    });

    if (P.scene.wide) {
      const { W } = P;
      const farLeft = left - 520 * k;
      if (farLeft > 120 * k) {
        reed(P, "back", "wide-reed-l1", on(88 * k), [56 * k, up(560)], 56 * k);
        papyrus(P, "back", "wide-pap-l1", on(176 * k), [128 * k, up(440)], 62 * k);
        papyrus(P, "back", "wide-pap-l2", on(206 * k), [232 * k, up(535)], 74 * k);
        crocusTuft(P, "back", "wide-tuft-l1", on(270 * k), 68 * k, 3, {
          lean: -0.22,
          paints: ["white", "purple", "white"],
        });
        madonna(P, "back", "wide-mad-l1", on(420 * k), [404 * k, up(520)], 58 * k, [
          [0.64, 1, 0.24, 1.0],
          [0.74, -1, 0.16, 1.05],
          [0.84, 1, 0.06, 0.96],
        ]);
        akrotiri(
          P,
          "front",
          "wide-ak-l1",
          on(620 * k),
          [
            [560 * k, up(310), "lily", 0.14],
            [648 * k, up(410), "lily", -0.05],
            [720 * k, up(255), "bud", -0.18],
          ],
          64 * k,
          { halo: 3 * k },
        );
        const sbl = on(815 * k);
        for (const [dx, h, len] of [
          [-26, 235, 0.95],
          [34, 185, 0.8],
          [4, 295, 1.0],
        ]) {
          const tip = [sbl[0] + dx * k, up(h)];
          const st = bend(sbl, tip, dx > 0 ? -0.08 : 0.08);
          P.pull("back.slip", st, { dense: true, w: 6 * k, profile: "rise" });
          seaDaffodil(P, "back", `wide-daf-l-${h}`, tip, heading(st) + (dx > 0 ? 0.45 : -0.38), 54 * k * len);
        }
        for (let i = 0; i < 4; i++)
          blade(P, "back.slip", sbl, -Math.PI / 2 + (i - 1.5) * 0.36, 112 * k, 10 * k, (i - 1.5) * 0.12);
        crocusTuft(P, "front", "wide-tuft-l2", on(925 * k), 58 * k, 2, {
          lean: 0.25,
          halo: 3 * k,
          paints: ["purple", "white"],
        });
        swallow(P, "wide-swallow-l", [520 * k, sy + 28 * k], -0.24, 78 * k, 1);
      }

      const farRight = right + 540 * k;
      if (W - farRight > 120 * k) {
        const xr = W;
        crocusTuft(P, "front", "wide-tuft-r0", on(farRight + 70 * k), 58 * k, 2, {
          lean: -0.22,
          halo: 3 * k,
          paints: ["white", "purple"],
        });
        akrotiri(
          P,
          "front",
          "wide-ak-r1",
          on(farRight + 240 * k),
          [
            [farRight + 175 * k, up(275), "bud", 0.2],
            [farRight + 258 * k, up(415), "lily", -0.06],
            [farRight + 335 * k, up(310), "lily", -0.14],
          ],
          64 * k,
          { halo: 3 * k },
        );
        madonna(P, "back", "wide-mad-r1", on(xr - 470 * k), [xr - 452 * k, up(530)], 58 * k, [
          [0.64, -1, 0.24, 1.0],
          [0.74, 1, 0.15, 1.06],
          [0.84, -1, 0.06, 0.96],
        ]);
        const sbr = on(xr - 330 * k);
        for (const [dx, h, len] of [
          [-32, 195, 0.85],
          [28, 250, 0.95],
          [-2, 300, 1.0],
        ]) {
          const tip = [sbr[0] + dx * k, up(h)];
          const st = bend(sbr, tip, dx > 0 ? -0.08 : 0.08);
          P.pull("back.slip", st, { dense: true, w: 6 * k, profile: "rise" });
          seaDaffodil(P, "back", `wide-daf-r-${h}`, tip, heading(st) + (dx > 0 ? 0.42 : -0.4), 54 * k * len);
        }
        for (let i = 0; i < 4; i++)
          blade(P, "back.slip", sbr, -Math.PI / 2 + (i - 1.5) * 0.36, 112 * k, 10 * k, (i - 1.5) * 0.12);
        papyrus(P, "back", "wide-pap-r1", on(xr - 205 * k), [xr - 238 * k, up(475)], 64 * k);
        papyrus(P, "back", "wide-pap-r2", on(xr - 172 * k), [xr - 122 * k, up(560)], 74 * k);
        reed(P, "back", "wide-reed-r1", on(xr - 88 * k), [xr - 54 * k, up(520)], 56 * k);
        crocusTuft(P, "back", "wide-tuft-r2", on(xr - 235 * k), 68 * k, 3, {
          lean: 0.28,
          paints: ["white", "white", "purple"],
        });
        swallow(P, "wide-swallow-r", [xr - 610 * k, sy + 16 * k], Math.PI + 0.28, 76 * k, -1);
      }
    }
  }

  function bbox(marks, W, H) {
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const m of marks) {
      const pts = m.blob || m.pts;
      const pad = m.blob ? 3 : Math.max(...m.wl, ...m.wr) + 4;
      for (const [x, y] of pts) {
        if (x - pad < x0) x0 = x - pad;
        if (y - pad < y0) y0 = y - pad;
        if (x + pad > x1) x1 = x + pad;
        if (y + pad > y1) y1 = y + pad;
      }
    }
    x0 = Math.max(0, Math.floor(x0));
    y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(W, Math.ceil(x1));
    y1 = Math.min(H, Math.ceil(y1));
    return [x0, y0, x1 - x0, y1 - y0];
  }

  function paint(ctx, name = "hero", makeCanvas) {
    const scene = SCENES[name] ?? SCENES.hero;
    const [W, H] = scene.size;
    const make =
      makeCanvas ||
      ((w, h) =>
        typeof OffscreenCanvas !== "undefined"
          ? new OffscreenCanvas(w, h)
          : Object.assign(document.createElement("canvas"), { width: w, height: h }));
    const ground = clay(scene);
    const buf = Float32Array.from(ground);
    const layers = compose(scene);
    const pigment = paints(scene, ground);
    const lc = make(W, H).getContext("2d");
    for (const depth of DEPTHS) {
      for (const name of PAINTS) {
        const marks = layers.get(`${depth}.${name}`);
        if (!marks) continue;
        const [x0, y0, w, h] = bbox(marks, W, H);
        if (w <= 0 || h <= 0) continue;
        lc.globalCompositeOperation = "source-over";
        lc.fillStyle = "#000";
        lc.fillRect(x0, y0, w, h);
        lc.globalCompositeOperation = "lighten";
        lc.lineCap = "round";
        lc.lineJoin = "round";
        for (const m of marks) rasterize(lc, m);
        const data = lc.getImageData(x0, y0, w, h).data;
        const fn = pigment[name];
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const j = (y * w + x) * 4;
            const R = data[j];
            if (!R) continue;
            const X = x0 + x,
              Y = y0 + y;
            const c = fn(R / 255, Math.min(1, data[j + 1] / R), X, Y);
            const a = c[3];
            const i = (Y * W + X) * 3;
            buf[i] += (c[0] - buf[i]) * a;
            buf[i + 1] += (c[1] - buf[i + 1]) * a;
            buf[i + 2] += (c[2] - buf[i + 2]) * a;
          }
        }
      }
    }
    grain(buf, scene);
    const img = ctx.createImageData(W, H);
    for (let i = 0, j = 0; i < buf.length; i += 3, j += 4) {
      img.data[j] = buf[i];
      img.data[j + 1] = buf[i + 1];
      img.data[j + 2] = buf[i + 2];
      img.data[j + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return scene;
  }

  return { paint, SCENES, GLYPHS, wordLayout };
});
