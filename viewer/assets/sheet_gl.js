(() => {
    const PAN_SPEED = 0.9;
    const ZOOM_SPEED = 1.6;
    const BATCH = 8;
    const REQUESTS = 4;
    const PARENTS = 4;
    const PREVIEW = 2;
    const PREFETCH_AGE = 3000;
    const STILL_MS = 400;
    const MOVING_MS = 120;
    const MOVING_REQUESTS = 2;
    const SPOT_MM = 0.3;
    const COLOUR = { ink: "#ff3b30", blank: "#3a8dff", spot: "#39ff14" };
    const el = (id) => document.getElementById(id);
    const cv = el("view");
    const sheet = el("tiles");
    const label = el("label");
    const bar = el("bar");
    const selector = el("layers");
    const T = M.tile;
    const gl = sheet.getContext("webgl2", {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: true,
        premultipliedAlpha: true,
        powerPreference: "high-performance",
    });
    if (!gl) {
        label.textContent = "This browser has no WebGL2: the sheet cannot be drawn";
        return;
    }
    const cx = cv.getContext("2d");

    M.suggested ??= [];
    const shelves = M.shelves.length ? M.shelves : [{ x: 0, y: 0, w: 1e9, h: 1e9 }];
    const shelfAt = (x) => shelves.find((s) => x < s.x + s.w) || shelves[shelves.length - 1];
    const onShelf = (b) => {
        const s = shelfAt(b.x + b.w / 2);
        return { ...b, x: b.x - s.x, y: b.y + s.y };
    };
    M.boxes = M.boxes.map(onShelf);
    M.places = M.places.map(onShelf);
    M.suggested = M.suggested.map(onShelf);
    M.outlines = (M.outlines || []).map((o) => {
        const runs = [];
        let at = null;
        let run = null;
        for (const p of o.points) {
            const h = shelfAt(p[0]);
            if (h !== at) {
                run = [];
                runs.push(run);
                at = h;
            }
            run.push([p[0] - h.x, p[1] + h.y]);
        }
        return { ...o, runs };
    });
    const spots = M.boxes.length ? M.boxes : M.places;
    const under = M.layers.find((L) => L.path.endsWith("taught-scan"));
    const sheetLayer = M.layers.find((L) => L.path.endsWith("sheet"));
    M.layers = M.layers.filter((L) => L !== under);
    const taught = M.layers.findIndex((L) => L.path.endsWith("taught"));
    const stock = M.layers.findIndex((L) => L.path.endsWith("ink"));
    const isInk = (L) => L.path.endsWith("taught") || L.path.endsWith("ink");

    let layer =
        taught >= 0
            ? taught
            : Math.max(
                  0,
                  stock,
                  stock >= 0 ? 0 : M.layers.findIndex((L) => L.path.endsWith("layer0")),
              );
    let zoom = 0;
    let ox = 0;
    let oy = 0;
    let mirror = false;
    let turn = false;
    let boxes = true;
    let showMarks = true;
    let spot = -1;
    let pick = -1;
    let invert = false;
    let tool = null;
    let brush = 0.4;
    let marks = [];
    let drawing = null;
    let panning = false;
    let dn = null;
    let pendingFit = null;
    let yellow = kept.yellow ?? true;
    let heat = yellow && !!kept.heat;
    let heatFloor = kept.heatFloor ?? 0;
    let heatScan = kept.heatScan ?? 0.9;
    let tintFloor = floor;
    let tintScan = scan;
    const STOCK =
        M.tint === "purple"
            ? [
                  [0.33, 0.1, 0.62],
                  [0.82, 0.55, 1],
              ]
            : [
                  [0.69, 0.44, 0.06],
                  [1, 0.957, 0.376],
              ];
    const hex = (c) =>
        `#${c
            .map((v) =>
                Math.round(v * 255)
                    .toString(16)
                    .padStart(2, "0"),
            )
            .join("")}`;
    let colour =
        /^#[0-9a-f]{6}$/.test(kept.colour) && kept.colour !== hex(STOCK[1]) ? kept.colour : null;
    let shade = STOCK;
    let dirty = false;
    let tilesDirty = true;
    let overlayDrawn = true;
    let scheduled = 0;
    let frame = 0;
    let lastLabel = "";
    let lastBar = "";
    let lastView = "";
    let movedAt = 0;
    let stillTimer = 0;

    function invalidate(overlayOnly = false) {
        dirty = true;
        if (!overlayOnly) {
            tilesDirty = true;
        }
        if (!scheduled) {
            scheduled = requestAnimationFrame(loop);
        }
    }
    function loop(now) {
        scheduled = 0;
        move(now);
        if (dirty) {
            draw();
        }
        if (heldKeys.size) {
            invalidate();
        }
    }

    if (taught >= 0 && stock >= 0) {
        el("swap").disabled = false;
        el("swap").title = "Swap ink (T)";
        el("swap").onclick = () => setLayer(layer === taught ? stock : taught);
    }
    el("back").href = M.back;
    M.layers.forEach((L, i) => {
        const o = document.createElement("option");
        o.value = i;
        o.textContent = L.name;
        o.title = L.name;
        selector.append(o);
    });
    function setLayer(k) {
        layer = (Number(k) + M.layers.length) % M.layers.length;
        selector.value = layer;
        selector.title = M.layers[layer].name;
        updateLevels();
        invalidate();
    }
    setLayer(layer);
    function full() {
        document.fullscreenElement
            ? document.exitFullscreen()
            : document.documentElement.requestFullscreen().catch(() => {});
    }
    selector.onchange = () => {
        setLayer(selector.value);
        selector.blur();
    };
    el("fullscreen").onclick = full;
    el("paper").onclick = () => {
        invert = !invert;
        el("paper").setAttribute("aria-pressed", invert);
        invalidate();
    };
    el("mirror").onclick = () => {
        mirror = !mirror;
        el("mirror").setAttribute("aria-pressed", mirror);
        invalidate();
    };
    el("previous").disabled = el("next").disabled = spots.length < 2;
    el("previous").onclick = () => go(Math.max(spot, 0) - 1);
    el("next").onclick = () => go(spot + 1);
    el("fit").onclick = () => {
        spot = -1;
        fit(M.area);
    };
    el("native").onclick = voxel;

    const zoomLimit = () => 8 / (M.layers[layer].mm * devicePixelRatio);
    function zoomBy(f) {
        const x = ox + innerWidth / zoom / 2;
        const y = oy + innerHeight / zoom / 2;
        zoom = Math.min(zoomLimit(), Math.max(0.03, zoom * f));
        ox = x - innerWidth / zoom / 2;
        oy = y - innerHeight / zoom / 2;
        invalidate();
    }
    el("plus").onclick = () => zoomBy(1.6);
    el("minus").onclick = () => zoomBy(1 / 1.6);
    function baseOf(L) {
        if (!yellow) {
            return null;
        }
        if (L.path.endsWith("taught")) {
            return under || sheetLayer;
        }
        return L.path.endsWith("ink") ? sheetLayer : null;
    }
    function updateLevels() {
        const L = M.layers[layer];
        el("floor").disabled = !(baseOf(L) || (heat && isInk(L)));
        el("scan").disabled = !baseOf(L);
    }
    function look() {
        invalidate();
        if (colour) {
            const c = [1, 3, 5].map((i) => Number.parseInt(colour.slice(i, i + 2), 16) / 255);
            shade = [c.map((v) => 0.7 * v * v), c];
        } else {
            shade = STOCK;
        }
        el("colour").value = hex(shade[1]);
        el("yellow").textContent = colour ? "Colour" : M.tint === "purple" ? "Purple" : "Yellow";
        el("yellow").setAttribute("aria-pressed", yellow && !heat);
        el("heat").setAttribute("aria-pressed", heat);
        el("heatkey").hidden = !heat;
        updateLevels();
        el("floor").value = heat ? heatFloor : tintFloor;
        el("scan").value = (heat ? heatScan : tintScan) * 100;
        try {
            localStorage.inkLook = JSON.stringify({
                yellow,
                heat,
                floor: tintFloor,
                heatFloor,
                scan: tintScan,
                heatScan,
                ...(colour ? { colour } : {}),
            });
        } catch (_) {}
    }
    function tinted() {
        if (heat) {
            heat = false;
        } else {
            yellow = !yellow;
        }
        look();
    }
    function heated() {
        heat = !heat;
        if (heat) {
            yellow = true;
        }
        look();
    }
    function weakestBy(d) {
        const v = Math.min(240, Math.max(0, (heat ? heatFloor : tintFloor) + d));
        if (heat) {
            heatFloor = v;
        } else {
            tintFloor = v;
        }
        look();
    }
    function scanBy(d) {
        const v = Math.min(1, Math.max(0, (heat ? heatScan : tintScan) + d));
        if (heat) {
            heatScan = v;
        } else {
            tintScan = v;
        }
        look();
    }
    function coloured(picked) {
        colour = picked && picked !== hex(STOCK[1]) ? picked : null;
        yellow = true;
        heat = false;
        look();
    }
    el("yellow").onclick = tinted;
    el("heat").onclick = heated;
    el("colour").oninput = (e) => coloured(e.target.value);
    el("colour").onchange = (e) => e.target.blur();
    el("colour").oncontextmenu = (e) => {
        e.preventDefault();
        coloured(null);
    };
    el("floor").oninput = (e) => {
        const v = Number(e.target.value);
        if (heat) {
            heatFloor = v;
        } else {
            tintFloor = v;
        }
        look();
    };
    el("scan").oninput = (e) => {
        const v = e.target.value / 100;
        if (heat) {
            heatScan = v;
        } else {
            tintScan = v;
        }
        look();
    };
    look();
    for (const id of ["floor", "scan"]) {
        el(id).onchange = (e) => e.target.blur();
    }
    function letters() {
        const on = el("letters").hidden;
        el("letters").hidden = !on;
        el("greek").setAttribute("aria-pressed", on);
    }
    el("greek").onclick = letters;

    const allLayers = [...new Set([...M.layers, under, sheetLayer].filter(Boolean))];
    const layerIndex = new Map(allLayers.map((L, i) => [L, i]));
    const maps = new Map();
    const layerSource = new Map();
    for (const L of allLayers) {
        const k = L.path.lastIndexOf("/");
        const dir = new URL(L.path.slice(0, k + 1) || "./", location.href).pathname;
        if (!maps.has(dir)) {
            maps.set(dir, {
                dir,
                stamp: null,
                layers: null,
                present: null,
                loading: false,
                failed: false,
            });
        }
        layerSource.set(L, { map: maps.get(dir), name: L.path.slice(k + 1) });
    }
    async function loadMap(m) {
        m.loading = true;
        try {
            const reply = await fetch(`/tilemap?dir=${m.dir}`, { cache: "no-store" });
            if (!reply.ok) {
                throw new Error("no tile map");
            }
            const buffer = await reply.arrayBuffer();
            const n = new DataView(buffer).getUint32(0, true);
            const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
            for (const entry of Object.values(head.layers)) {
                entry.offsets = [0];
                for (const [c, r] of entry.grid) {
                    entry.offsets.push(entry.offsets[entry.offsets.length - 1] + c * r);
                }
            }
            m.layers = head.layers;
            m.present = new Uint8Array(buffer, 4 + n);
            m.stamp = head.stamp;
            m.failed = false;
        } catch (_) {
            m.failed = true;
        } finally {
            m.loading = false;
            invalidate();
        }
    }
    for (const m of maps.values()) {
        loadMap(m);
    }
    function entryOf(L) {
        const { map, name } = layerSource.get(L);
        return map.layers ? map.layers[name] || null : null;
    }
    function shownOf(L) {
        const e = entryOf(L);
        return e?.raw ? [e.low / 255, 255 / (e.high - e.low)] : [0, 1];
    }
    function cellOf(entry, level, c, r) {
        if (level < 0 || level >= entry.grid.length) {
            return -1;
        }
        const [c0, r0] = entry.origin ? entry.origin[level] : [0, 0];
        const [columns, rows] = entry.grid[level];
        if (c < c0 || r < r0 || c - c0 >= columns || r - r0 >= rows) {
            return -1;
        }
        return entry.first + entry.offsets[level] + (r - r0) * columns + (c - c0);
    }
    const stored = (map, entry, level, c, r) => {
        const k = cellOf(entry, level, c, r);
        return k >= 0 && (map.present[k >> 3] >> (k & 7)) & 1;
    };
    const keyOf = (li, level, c, r) => ((li * 16 + level) * 16384 + c) * 16384 + r;

    const CHUNK = 128;
    let atlases = [];
    let capacity = 0;
    let made = 0;
    let slotKey = new Float64Array(0);
    let slotUsed = new Uint32Array(0);
    let slotTime = new Float64Array(0);
    let slotKept = new Uint8Array(0);
    let free = [];
    const slotOf = new Map();
    const inflight = new Set();
    const retryAt = new Map();
    let lastAsked = new Set();
    let busy = 0;
    let queue = [];
    let programs = null;
    let instances = new Float32Array(9 * 1024);
    let chunkOf = new Uint16Array(1024);
    let sorted = new Float32Array(9 * 1024);
    const parallel = gl.getExtension("KHR_parallel_shader_compile");

    function compile(vertex, fragment) {
        const program = gl.createProgram();
        for (const [type, text] of [
            [gl.VERTEX_SHADER, vertex],
            [gl.FRAGMENT_SHADER, fragment],
        ]) {
            const s = gl.createShader(type);
            gl.shaderSource(s, text);
            gl.compileShader(s);
            gl.attachShader(program, s);
        }
        gl.linkProgram(program);
        return { program, uniforms: null };
    }
    function linked(p) {
        if (p.uniforms) {
            return true;
        }
        if (parallel && !gl.getProgramParameter(p.program, parallel.COMPLETION_STATUS_KHR)) {
            return false;
        }
        if (!gl.getProgramParameter(p.program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(p.program));
        }
        p.uniforms = {};
        const count = gl.getProgramParameter(p.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < count; i++) {
            const name = gl.getActiveUniform(p.program, i).name;
            p.uniforms[name] = gl.getUniformLocation(p.program, name);
        }
        return true;
    }

    const TILE_VERTEX = `#version 300 es
layout(location=0) in vec2 unit;
layout(location=1) in vec4 rect;
layout(location=2) in vec4 part;
layout(location=3) in float slot;
uniform vec2 screen;
out vec2 at;
flat out float layer;
void main(){vec2 p=rect.xy+unit*rect.zw;gl_Position=vec4(p.x/screen.x*2.-1.,1.-p.y/screen.y*2.,0,1);at=part.xy+unit*part.zw;layer=slot;}`;
    const TILE_FRAGMENT = `#version 300 es
precision mediump float;
precision mediump sampler2DArray;
in highp vec2 at;
flat in highp float layer;
uniform sampler2DArray tiles;
uniform float weakest,brightness,inverted;
uniform vec2 shown;
uniform vec3 dim,bright;
uniform int ink;
out vec4 colour;
const vec3 HEAT[9]=vec3[9](vec3(0.,0.,.5),vec3(0.,0.,1.),vec3(0.,.5,1.),vec3(.08,1.,.94),vec3(.49,1.,.48),vec3(.9,1.,.1),vec3(1.,.6,0.),vec3(1.,.15,0.),vec3(.7,0.,0.));
void main(){float raw=texture(tiles,vec3(at,layer)).r,v=clamp((raw-shown.x)*shown.y,0.,1.);vec3 c=vec3(v*brightness);float a=1.;
if(ink==1){float t=clamp((v-weakest)/(1.-weakest),0.,1.);c=mix(dim,bright,t);a=min(1.,t*3.);}
else if(ink==2){float x=v*8.;int i=min(int(x),7);c=mix(HEAT[i],HEAT[i+1],x-float(i));a=.45*clamp((v-weakest)*16.+step(weakest,0.),0.,1.);}
else if(ink==3){if(raw<.05)discard;c=vec3((.45+.55*v)*brightness);}
colour=vec4(mix(c,1.-c,inverted),a);}`;
    function setup() {
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        const tile = compile(TILE_VERTEX, TILE_FRAGMENT);
        const unit = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, unit);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
        const tileArray = gl.createVertexArray();
        gl.bindVertexArray(tileArray);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        const tileBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, tileBuffer);
        for (const index of [1, 2, 3]) {
            gl.enableVertexAttribArray(index);
            gl.vertexAttribDivisor(index, 1);
        }
        gl.bindVertexArray(null);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
        programs = { tile, tileArray, tileBuffer };
        atlases = [];
        made = 0;
        free = [];
        slotOf.clear();
        tilesDirty = true;
    }

    function slotsFor(width, height) {
        const across = (Math.ceil(width / (T / 2)) + 2) * (Math.ceil(height / (T / 2)) + 2);
        const wanted = Math.ceil((2 * across * 1.25 + 64) * 1.75);
        return Math.min(8 * CHUNK, Math.ceil(wanted / CHUNK) * CHUNK);
    }
    function grow(count) {
        if (slotKey.length >= count) {
            return;
        }
        const key = new Float64Array(count).fill(-1);
        const used = new Uint32Array(count);
        const time = new Float64Array(count);
        const kept = new Uint8Array(count);
        key.set(slotKey);
        used.set(slotUsed);
        time.set(slotTime);
        kept.set(slotKept);
        slotKey = key;
        slotUsed = used;
        slotTime = time;
        slotKept = kept;
    }
    function addChunk() {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
        gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R8, T, T, CHUNK);
        for (const [k, v] of [
            [gl.TEXTURE_MIN_FILTER, gl.LINEAR],
            [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
            [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
        ]) {
            gl.texParameteri(gl.TEXTURE_2D_ARRAY, k, v);
        }
        atlases.push(texture);
        grow(made + CHUNK);
        for (let s = made + CHUNK - 1; s >= made; s--) {
            free.push(s);
        }
        made += CHUNK;
    }
    function forget() {
        slotOf.clear();
        slotKey.fill(-1);
        free = [];
        for (let s = made - 1; s >= 0; s--) {
            free.push(s);
        }
    }
    function allocate(minAge) {
        if (free.length) {
            return free.pop();
        }
        if (made < capacity) {
            addChunk();
            return free.pop();
        }
        const now = performance.now();
        const old = [];
        for (let s = 0; s < made; s++) {
            if (slotUsed[s] < frame && now - slotTime[s] >= minAge) {
                old.push(s);
            }
        }
        if (!old.length) {
            return -1;
        }
        old.sort((a, b) => slotKept[a] - slotKept[b] || slotTime[a] - slotTime[b]);
        for (const s of old.slice(0, 32)) {
            slotOf.delete(slotKey[s]);
            slotKey[s] = -1;
            free.push(s);
        }
        return free.pop();
    }
    function upload(t, pixels) {
        if (!programs || gl.isContextLost() || t.map.stamp !== t.stamp || slotOf.has(t.key)) {
            return;
        }
        const slot = allocate(t.prefetch || !lastAsked.has(t.key) ? PREFETCH_AGE : 0);
        if (slot < 0) {
            return;
        }
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, atlases[(slot / CHUNK) | 0]);
        gl.texSubImage3D(
            gl.TEXTURE_2D_ARRAY,
            0,
            0,
            0,
            slot % CHUNK,
            T,
            T,
            1,
            gl.RED,
            gl.UNSIGNED_BYTE,
            pixels,
        );
        slotKey[slot] = t.key;
        slotUsed[slot] = frame;
        slotTime[slot] = performance.now();
        slotKept[slot] = 0;
        slotOf.set(t.key, slot);
        invalidate();
    }

    const replies = new Map();
    let requestId = 0;
    let fetcher = null;
    async function fetchTiles(url, body) {
        try {
            const reply = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
            });
            if (!reply.ok) {
                return { status: reply.status };
            }
            return { status: 200, buffer: await reply.arrayBuffer() };
        } catch {
            return { status: 0 };
        }
    }
    try {
        const receive = async ({ data: { id, url, body } }) => {
            const reply = await fetchTiles(url, body);
            self.postMessage({ id, ...reply }, reply.buffer ? [reply.buffer] : []);
        };
        const code = `${fetchTiles}\nself.onmessage = ${receive};`;
        fetcher = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
        fetcher.onmessage = ({ data }) => {
            const done = replies.get(data.id);
            replies.delete(data.id);
            done?.(data);
        };
        fetcher.onerror = (e) => {
            e.preventDefault();
            fetcher = null;
            for (const done of replies.values()) {
                done({ status: 0 });
            }
            replies.clear();
        };
    } catch (_) {
        fetcher = null;
    }
    function post(body) {
        if (fetcher) {
            return new Promise((resolve) => {
                const id = ++requestId;
                replies.set(id, resolve);
                fetcher.postMessage({ id, url: `${location.origin}/tiles`, body });
            });
        }
        return fetchTiles("/tiles", body);
    }
    async function fetchBatch(batch) {
        const map = batch[0].map;
        busy++;
        const body = JSON.stringify({
            dir: map.dir,
            stamp: map.stamp,
            size: T,
            tiles: batch.map((t) => [t.name, t.level, t.c, t.r]),
        });
        const reply = await post(body);
        try {
            if (reply.status === 409) {
                if (!map.loading) {
                    forget();
                    loadMap(map);
                }
            } else if (reply.status !== 200) {
                const later = performance.now() + 2000;
                for (const t of batch) {
                    retryAt.set(t.key, later);
                }
            } else {
                const size = T * T;
                for (
                    let k = 0;
                    k < batch.length && (k + 1) * size <= reply.buffer.byteLength;
                    k++
                ) {
                    upload(batch[k], new Uint8Array(reply.buffer, k * size, size));
                }
            }
        } finally {
            for (const t of batch) {
                inflight.delete(t.key);
            }
            busy--;
            pump();
            invalidate();
        }
    }
    function pump() {
        const limit = performance.now() - movedAt < MOVING_MS ? MOVING_REQUESTS : REQUESTS;
        while (busy < limit && queue.length) {
            const map = queue[0].map;
            const batch = [];
            const rest = [];
            for (const t of queue) {
                if (inflight.has(t.key)) {
                    continue;
                }
                if (batch.length < BATCH && t.map === map) {
                    batch.push(t);
                    inflight.add(t.key);
                } else {
                    rest.push(t);
                }
            }
            queue = rest;
            if (batch.length) {
                fetchBatch(batch);
            }
        }
    }

    function reserve(count) {
        if (chunkOf.length >= count) {
            return;
        }
        const more = Math.max(count, chunkOf.length * 2);
        const grown = new Float32Array(more * 9);
        grown.set(instances);
        instances = grown;
        const chunks = new Uint16Array(more);
        chunks.set(chunkOf);
        chunkOf = chunks;
        sorted = new Float32Array(more * 9);
    }

    function size() {
        const d = devicePixelRatio;
        for (const c of [cv, sheet]) {
            c.width = Math.max(1, Math.round(innerWidth * d));
            c.height = Math.max(1, Math.round(innerHeight * d));
        }
        if (pendingFit && innerWidth && innerHeight) {
            const [b, margin] = pendingFit;
            pendingFit = null;
            fit(b, margin);
        }
        invalidate();
    }

    function draw() {
        dirty = false;
        const d = devicePixelRatio;
        const W = cv.width;
        const H = cv.height;
        const s = zoom * d;
        const L = M.layers[layer];
        const fx = mirror !== turn;
        const fy = turn;
        if (tilesDirty) {
            tilesDirty = false;
            drawTiles();
        }
        overlay(W, H, s, fx, fy, d);
        if (zoom > 0) {
            chrome(L, fx, fy);
        }
    }

    function drawTiles() {
        const d = devicePixelRatio;
        const W = cv.width;
        const H = cv.height;
        const s = zoom * d;
        const L = M.layers[layer];
        const base = baseOf(L);
        const fx = mirror !== turn;
        const fy = turn;
        const vw = W / s;
        const vh = H / s;
        if (!programs || gl.isContextLost()) {
            return;
        }
        if (!linked(programs.tile)) {
            invalidate();
            tilesDirty = true;
            return;
        }
        frame++;
        const now = performance.now();
        const view = `${zoom} ${ox} ${oy} ${layer} ${heat} ${yellow}`;
        if (view !== lastView) {
            lastView = view;
            movedAt = now;
        }
        const still = now - movedAt >= STILL_MS;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, W, H);
        gl.disable(gl.SCISSOR_TEST);
        gl.clearColor(invert ? 1 : 0, invert ? 1 : 0, invert ? 1 : 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
        const ready = zoom > 0 && [...new Set([L, base].filter(Boolean))].every((x) => entryOf(x));
        let exact = 0;
        let coarse = 0;
        let empty = 0;
        const wanted = [];
        const prefetch = [];
        const asked = new Set();
        const ringed = new Set();
        if (ready) {
            capacity = Math.max(capacity, slotsFor(W, H));
            let lvl = 0;
            while (lvl < L.levels - 1 && L.mm * 2 ** (lvl + 1) * s <= 1) {
                lvl++;
            }
            const ts = T * L.mm * 2 ** lvl;
            const passes =
                base && heat
                    ? [
                          [base, 3, heatScan],
                          [L, 2, 1],
                      ]
                    : base
                      ? [
                            [base, 0, tintScan],
                            [L, 1, 1],
                        ]
                      : [[L, heat && isInk(L) ? 2 : 0, 1]];
            const ranges = [];
            let n = 0;
            const push = (x, y, w, h, u, v, du, dv, slot) => {
                reserve(n + 1);
                const o = n * 9;
                instances[o] = x;
                instances[o + 1] = y;
                instances[o + 2] = w;
                instances[o + 3] = h;
                instances[o + 4] = u;
                instances[o + 5] = v;
                instances[o + 6] = du;
                instances[o + 7] = dv;
                instances[o + 8] = slot % CHUNK;
                chunkOf[n] = (slot / CHUNK) | 0;
                slotUsed[slot] = frame;
                slotTime[slot] = now;
                if (still) {
                    slotKept[slot] = 1;
                }
                n++;
            };
            const want = (list, source, li, level, c, r, middle) => {
                const key = keyOf(li, level, c, r);
                const ring = list === prefetch;
                if (slotOf.has(key) || asked.has(key) || (ring && ringed.has(key))) {
                    return;
                }
                (ring ? ringed : asked).add(key);
                if (inflight.has(key) || (retryAt.get(key) ?? 0) > now) {
                    return;
                }
                const S = 2 ** (level - lvl);
                const dx = (c + 0.5) * S - middle[0];
                const dy = (r + 0.5) * S - middle[1];
                list.push({
                    key,
                    map: source.map,
                    stamp: source.map.stamp,
                    name: source.name,
                    level,
                    c,
                    r,
                    order: dx * dx + dy * dy,
                    prefetch: ring,
                });
            };
            for (const [P, mode, brightness] of passes) {
                const li = layerIndex.get(P);
                const source = layerSource.get(P);
                const entry = entryOf(P);
                const map = source.map;
                const levels = Math.min(P.levels, entry.grid.length);
                for (const h of shelves) {
                    const sx = ox + h.x;
                    const sy = oy - h.y;
                    if (sy + vh < 0 || sy > h.h || sx + vw < h.x || sx > h.x + h.w) {
                        continue;
                    }
                    const a = Math.max((h.x - sx) * s, 0);
                    const b = Math.max(-sy * s, 0);
                    const e = Math.min((h.x + h.w - sx) * s, W);
                    const f = Math.min((h.h - sy) * s, H);
                    const first = n;
                    const middle = [(sx + vw / 2) / ts, (sy + vh / 2) / ts];
                    const r0 = Math.max(0, Math.floor(sy / ts));
                    const r1 = Math.floor(Math.min(sy + vh, h.h) / ts);
                    const c0 = Math.max(0, Math.floor(Math.max(sx, h.x) / ts));
                    const c1 = Math.floor(Math.min(sx + vw, h.x + h.w) / ts);
                    for (let r = r0; r <= r1; r++) {
                        for (let c = c0; c <= c1; c++) {
                            const x0 = Math.round((c * ts - sx) * s);
                            const y0 = Math.round((r * ts - sy) * s);
                            let x = x0;
                            let y = y0;
                            let w = Math.round(((c + 1) * ts - sx) * s) - x0;
                            let hh = Math.round(((r + 1) * ts - sy) * s) - y0;
                            if (fx) {
                                x = W - x;
                                w = -w;
                            }
                            if (fy) {
                                y = H - y;
                                hh = -hh;
                            }
                            const slot = slotOf.get(keyOf(li, lvl, c, r));
                            if (slot !== undefined) {
                                push(x, y, w, hh, 0, 0, 1, 1, slot);
                                exact++;
                                continue;
                            }
                            if (!stored(map, entry, lvl, c, r)) {
                                continue;
                            }
                            want(wanted, source, li, lvl, c, r, middle);
                            let done = false;
                            if (lvl > 0) {
                                const children = [];
                                let have = 0;
                                for (let q = 0; q < 4; q++) {
                                    const cc = 2 * c + (q & 1);
                                    const rr = 2 * r + (q >> 1);
                                    if (!stored(map, entry, lvl - 1, cc, rr)) {
                                        continue;
                                    }
                                    have++;
                                    const child = slotOf.get(keyOf(li, lvl - 1, cc, rr));
                                    if (child !== undefined) {
                                        children.push([q, child]);
                                    }
                                }
                                if (have && children.length === have) {
                                    for (const [q, child] of children) {
                                        push(
                                            x + ((q & 1) * w) / 2,
                                            y + ((q >> 1) * hh) / 2,
                                            w / 2,
                                            hh / 2,
                                            0,
                                            0,
                                            1,
                                            1,
                                            child,
                                        );
                                    }
                                    coarse++;
                                    done = true;
                                }
                            }
                            for (let up = 1; !done && up <= PARENTS && lvl + up < levels; up++) {
                                const parent = slotOf.get(keyOf(li, lvl + up, c >> up, r >> up));
                                if (parent === undefined) {
                                    continue;
                                }
                                const k = 1 << up;
                                push(x, y, w, hh, (c % k) / k, (r % k) / k, 1 / k, 1 / k, parent);
                                coarse++;
                                done = true;
                            }
                            if (done) {
                                continue;
                            }
                            empty++;
                            const up = Math.min(PREVIEW, levels - 1 - lvl);
                            if (up > 0 && stored(map, entry, lvl + up, c >> up, r >> up)) {
                                want(wanted, source, li, lvl + up, c >> up, r >> up, middle);
                            }
                        }
                    }
                    for (let r = r0 - 1; r <= r1 + 1; r++) {
                        for (let c = c0 - 1; c <= c1 + 1; c++) {
                            if (
                                (r < r0 || r > r1 || c < c0 || c > c1) &&
                                r >= 0 &&
                                c >= 0 &&
                                stored(map, entry, lvl, c, r)
                            ) {
                                want(prefetch, source, li, lvl, c, r, middle);
                            }
                        }
                    }
                    const scissor = [
                        Math.round(fx ? W - e : a),
                        Math.round(fy ? b : H - f),
                        Math.round(e - a),
                        Math.round(f - b),
                    ];
                    const counts = new Uint32Array(atlases.length);
                    for (let i = first; i < n; i++) {
                        counts[chunkOf[i]]++;
                    }
                    const starts = new Uint32Array(atlases.length);
                    for (let k = 1; k < atlases.length; k++) {
                        starts[k] = starts[k - 1] + counts[k - 1];
                    }
                    const fill = starts.slice();
                    for (let i = first; i < n; i++) {
                        const to = (first + fill[chunkOf[i]]++) * 9;
                        sorted.set(instances.subarray(i * 9, i * 9 + 9), to);
                    }
                    if (n > first) {
                        instances.set(sorted.subarray(first * 9, n * 9), first * 9);
                    }
                    for (let k = 0; k < atlases.length; k++) {
                        if (counts[k]) {
                            ranges.push([
                                mode,
                                brightness,
                                P,
                                first + starts[k],
                                counts[k],
                                k,
                                ...scissor,
                            ]);
                        }
                    }
                }
            }
            const { tile, tileArray, tileBuffer } = programs;
            gl.useProgram(tile.program);
            gl.bindVertexArray(tileArray);
            gl.bindBuffer(gl.ARRAY_BUFFER, tileBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, instances.subarray(0, n * 9), gl.STREAM_DRAW);
            gl.activeTexture(gl.TEXTURE0);
            gl.uniform1i(tile.uniforms.tiles, 0);
            gl.uniform2f(tile.uniforms.screen, W, H);
            gl.uniform1f(tile.uniforms.inverted, invert ? 1 : 0);
            gl.uniform1f(tile.uniforms.weakest, (heat ? heatFloor : tintFloor) / 255);
            gl.uniform3fv(tile.uniforms.dim, shade[0]);
            gl.uniform3fv(tile.uniforms.bright, shade[1]);
            gl.enable(gl.SCISSOR_TEST);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            let stencilled = false;
            for (const [mode, brightness, P, first, count, chunk, sx, sy, sw, sh] of ranges) {
                const offset = first * 36;
                gl.bindTexture(gl.TEXTURE_2D_ARRAY, atlases[chunk]);
                gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 36, offset);
                gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 36, offset + 16);
                gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 36, offset + 32);
                gl.uniform1i(tile.uniforms.ink, mode);
                gl.uniform2f(tile.uniforms.shown, ...shownOf(P));
                gl.uniform1f(tile.uniforms.brightness, brightness);
                if (mode) {
                    gl.enable(gl.BLEND);
                } else {
                    gl.disable(gl.BLEND);
                }
                if (mode === 3 || mode === 2) {
                    if (base && heat) {
                        gl.enable(gl.STENCIL_TEST);
                        gl.stencilFunc(mode === 3 ? gl.ALWAYS : gl.EQUAL, 1, 255);
                        stencilled = true;
                    }
                } else if (stencilled) {
                    gl.disable(gl.STENCIL_TEST);
                    stencilled = false;
                }
                gl.scissor(sx, sy, sw, sh);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
            }
            gl.disable(gl.STENCIL_TEST);
            gl.disable(gl.SCISSOR_TEST);
            gl.bindVertexArray(null);
        }
        const tileOrder = (a, b) => b.level - a.level || a.order - b.order;
        wanted.sort(tileOrder);
        const ahead = prefetch.filter((t) => !asked.has(t.key)).sort(tileOrder);
        queue = wanted.length || busy || !still ? wanted : ahead;
        if (!still && !stillTimer) {
            stillTimer = setTimeout(
                () => {
                    stillTimer = 0;
                    invalidate();
                },
                STILL_MS - (now - movedAt) + 1,
            );
        }
        lastAsked = asked;
        pump();
        gl.disable(gl.BLEND);
        viewer.onframe?.(exact, coarse, empty);
    }

    function overlay(W, H, s, fx, fy, d) {
        const vw = W / s;
        const vh = H / s;
        const shown = (drawing ? [...marks, drawing] : marks).filter(
            (k) =>
                seen(k) &&
                k.box[0] < ox + vw &&
                k.box[2] > ox &&
                k.box[1] < oy + vh &&
                k.box[3] > oy,
        );
        const framed = boxes && (M.boxes.length || M.outlines.length || M.suggested.length);
        if (!overlayDrawn && !shown.length && !framed) {
            return;
        }
        cx.setTransform(1, 0, 0, 1, 0, 0);
        cx.clearRect(0, 0, W, H);
        overlayDrawn = shown.length > 0 || framed;
        if (!(zoom > 0)) {
            return;
        }
        cx.setTransform(
            fx ? -s : s,
            0,
            0,
            fy ? -s : s,
            fx ? W + ox * s : -ox * s,
            fy ? H + oy * s : -oy * s,
        );
        for (const k of shown) {
            ink(k, s);
        }
        if (!boxes) {
            return;
        }
        const visible = (b) => b.x < ox + vw && b.x + b.w > ox && b.y < oy + vh && b.y + b.h > oy;
        const strokeBoxes = (rects, colour, width) => {
            cx.shadowColor = colour;
            cx.shadowBlur = 6 * d;
            cx.strokeStyle = colour;
            cx.lineWidth = (width * d) / s;
            for (const b of rects) {
                cx.strokeRect(b.x, b.y, b.w, b.h);
            }
            cx.shadowBlur = 0;
        };
        if (M.boxes.length) {
            strokeBoxes(M.boxes.filter(visible), "#f00", 1);
        }
        if (M.outlines.length) {
            cx.shadowColor = "#ffd60a";
            cx.shadowBlur = 8 * d;
            cx.strokeStyle = "#ffd60a";
            cx.lineWidth = (2 * d) / s;
            cx.lineJoin = "round";
            cx.beginPath();
            for (const o of M.outlines) {
                for (const r of o.runs) {
                    cx.moveTo(r[0][0], r[0][1]);
                    for (let i = 1; i < r.length; i++) {
                        cx.lineTo(r[i][0], r[i][1]);
                    }
                }
            }
            cx.stroke();
            cx.shadowBlur = 0;
        }
        if (M.suggested.length) {
            const suggested = M.suggested.filter(visible);
            strokeBoxes(suggested, "#0ff", 2);
            cx.setTransform(1, 0, 0, 1, 0, 0);
            cx.fillStyle = "#0ff";
            cx.font = `${13 * d}px Calibri`;
            const at = (v, o, flip, size) => (flip ? size + (o - v) * s : (v - o) * s);
            for (const b of suggested) {
                const lx = Math.min(at(b.x, ox, fx, W), at(b.x + b.w, ox, fx, W));
                const ly = Math.min(at(b.y, oy, fy, H), at(b.y + b.h, oy, fy, H)) - 5 * d;
                cx.fillStyle = "#000";
                cx.fillRect(lx - 2 * d, ly - 13 * d, cx.measureText(b.label).width + 4 * d, 17 * d);
                cx.fillStyle = "#0ff";
                cx.fillText(b.label, lx, ly);
            }
        }
    }
    function ink(k, s) {
        cx.globalAlpha = 0.6;
        cx.strokeStyle = COLOUR[k.kind];
        cx.lineWidth = Math.max(1 / s, k.width_mm);
        cx.lineCap = cx.lineJoin = "round";
        cx.stroke(k.path);
        cx.globalAlpha = 1;
    }

    function chrome(L, fx, fy) {
        const nice =
            [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find(
                (v) => v >= 100 / zoom,
            ) || 1000;
        const width = `${nice * zoom}px`;
        if (width !== lastBar) {
            bar.style.width = lastBar = width;
        }
        const model = L.model || "post-trained Reader v3";
        const text =
            (nice >= 10 ? `${nice / 10} cm` : `${nice} mm`) +
            " · " +
            M.title +
            " · " +
            L.name +
            (M.tint === "purple" && L.path.endsWith("taught") && !M.title.includes(model)
                ? ` · ${model} model`
                : "") +
            (tool && tool !== "erase" ? ` · marker ${brush.toFixed(2)} mm` : "") +
            writes.status() +
            ([...maps.values()].some((m) => m.failed)
                ? " · tiles unavailable: restart the site server"
                : "");
        if (text !== lastLabel) {
            label.textContent = lastLabel = text;
        }
        updateSpotReadout(marks, seen, { ox, oy, zoom, flipX: fx, flipY: fy });
        const region =
            pick >= 0
                ? `Suggested ${pick + 1} / ${M.suggested.length}`
                : spot < 0
                  ? ""
                  : `${spot + 1} / ${spots.length}`;
        if (el("region").textContent !== region) {
            el("region").textContent = region;
        }
    }

    const heldKeys = Shortcuts.held;
    let then = performance.now();
    function move(now) {
        const dt = Math.min(0.05, (now - then) / 1000);
        then = now;
        if (!heldKeys.size) {
            return;
        }
        const m = mirror !== turn ? -1 : 1;
        const n = turn ? -1 : 1;
        const sx = ((PAN_SPEED * innerWidth) / zoom) * dt;
        const sy = ((PAN_SPEED * innerHeight) / zoom) * dt;
        if (heldKeys.has("a")) {
            ox -= m * sx;
        }
        if (heldKeys.has("d")) {
            ox += m * sx;
        }
        if (heldKeys.has("w")) {
            oy -= n * sy;
        }
        if (heldKeys.has("s")) {
            oy += n * sy;
        }
        if (heldKeys.has("e")) {
            zoomBy(Math.exp(-ZOOM_SPEED * dt));
        }
        if (heldKeys.has("r")) {
            zoomBy(Math.exp(ZOOM_SPEED * dt));
        }
        invalidate();
    }

    function fit(b, margin) {
        if (!innerWidth || !innerHeight) {
            pendingFit = [b, margin];
            return;
        }
        const top = document.querySelector("nav").offsetHeight || TOP;
        const m = margin || 1.02;
        zoom = Math.min(innerWidth / (b.w * m), Math.max(100, innerHeight - top) / (b.h * m));
        ox = b.x + b.w / 2 - innerWidth / zoom / 2;
        oy = b.y + b.h / 2 - (innerHeight + top) / zoom / 2;
        invalidate();
    }
    function go(k) {
        pick = -1;
        spot = (k + spots.length) % spots.length;
        fit(spots[spot], M.boxes.length ? 1.25 : 1.06);
    }
    function suggest() {
        spot = -1;
        pick = (pick + 1) % M.suggested.length;
        fit(M.suggested[pick], 1.25);
    }
    function voxel() {
        if (!zoom) {
            return;
        }
        const x = ox + innerWidth / zoom / 2;
        const y = oy + innerHeight / zoom / 2;
        zoom = 1 / (M.layers[layer].mm * devicePixelRatio);
        ox = x - innerWidth / zoom / 2;
        oy = y - innerHeight / zoom / 2;
        invalidate();
    }

    const page = decodeURIComponent(location.pathname.match(/\/sheets\/([^/]+)\//)?.[1] ?? "");
    const writes = markWrites(M.scan, page, invalidate);
    const discard = (k) => {
        marks = marks.filter((m) => m !== k);
        invalidate(true);
    };
    function seen(k) {
        return !k.removing && (showMarks || !k.old);
    }
    function sheetToScreen(p) {
        const s = shelfAt(p[0]);
        return [p[0] - s.x, p[1] + s.y];
    }
    const updateSpotReadout = spotReadout(el("spot-coordinates"), M.scan, sheetToScreen);
    function placed(k) {
        const shown = k.points.map(sheetToScreen);
        const path = new Path2D();
        const xs = shown.map((p) => p[0]);
        const ys = shown.map((p) => p[1]);
        const m = k.width_mm;
        shown.forEach(([x, y], i) => {
            if (i) {
                path.lineTo(x, y);
            } else {
                path.moveTo(x, y);
            }
        });
        if (shown.length === 1) {
            path.lineTo(shown[0][0] + 0.001, shown[0][1]);
        }
        k.path = path;
        k.box = [
            Math.min(...xs) - m,
            Math.min(...ys) - m,
            Math.max(...xs) + m,
            Math.max(...ys) + m,
        ];
        return k;
    }
    function sheetAt(e) {
        const px = mirror !== turn ? innerWidth - e.clientX : e.clientX;
        const py = turn ? innerHeight - e.clientY : e.clientY;
        const X = ox + px / zoom;
        const Y = oy + py / zoom;
        const s = shelves.find((s) => Y >= s.y && Y < s.y + s.h && X >= 0 && X < s.w);
        return s ? [+(X + s.x).toFixed(3), +(Y - s.y).toFixed(3)] : null;
    }
    function toggleMarks() {
        showMarks = !showMarks;
        el("showmarks").textContent = showMarks ? "Hide marks" : "Show marks";
        invalidate(true);
    }
    el("showmarks").onclick = toggleMarks;
    function wipe(p) {
        const k = marks.find(
            (k) =>
                seen(k) &&
                k.points.some(
                    (q) => Math.hypot(p[0] - q[0], p[1] - q[1]) <= k.width_mm / 2 + 3 / zoom,
                ),
        );
        if (k) {
            writes.remove(k, discard);
        }
    }
    function undo() {
        const k = marks.filter(seen).pop();
        if (k) {
            writes.remove(k, discard);
        }
    }
    function useTool(t) {
        tool = tool === t ? null : t;
        if (tool) {
            el("labelling").classList.add("open");
            el("labelopen").setAttribute("aria-pressed", true);
            el("labelopen").setAttribute("aria-expanded", true);
        }
        for (const [id, name] of [
            ["mark", "ink"],
            ["spotmark", "spot"],
            ["erase", "erase"],
        ]) {
            el(id).setAttribute("aria-pressed", tool === name);
        }
        cv.style.cursor = tool ? "crosshair" : "";
        invalidate(true);
    }
    el("mark").onclick = () => useTool("ink");
    el("spotmark").onclick = () => useTool("spot");
    el("erase").onclick = () => useTool("erase");
    el("undo").onclick = undo;

    cv.onpointerdown = (e) => {
        cv.setPointerCapture(e.pointerId);
        if (tool && !panning) {
            const p = sheetAt(e);
            if (tool === "erase") {
                if (p) {
                    wipe(p);
                }
            } else if (p) {
                drawing = placed({
                    kind: tool,
                    width_mm: tool === "spot" ? SPOT_MM : brush,
                    points: [p],
                });
                invalidate(true);
            }
            return;
        }
        dn = [e.clientX, e.clientY, ox, oy];
        cv.classList.add("drag");
    };
    cv.onpointermove = (e) => {
        if (drawing) {
            const p = sheetAt(e);
            const q = drawing.points[drawing.points.length - 1];
            if (p && Math.hypot(p[0] - q[0], p[1] - q[1]) * zoom >= 1) {
                drawing.points.push(p);
                placed(drawing);
                invalidate(true);
            }
            return;
        }
        if (tool === "erase" && e.buttons && !panning) {
            const p = sheetAt(e);
            if (p) {
                wipe(p);
            }
            return;
        }
        if (!dn) {
            return;
        }
        const m = mirror !== turn ? -1 : 1;
        const n = turn ? -1 : 1;
        ox = dn[2] - (m * (e.clientX - dn[0])) / zoom;
        oy = dn[3] - (n * (e.clientY - dn[1])) / zoom;
        invalidate();
    };
    cv.onpointerup = () => {
        dn = null;
        cv.classList.remove("drag");
        if (drawing) {
            const k = drawing;
            drawing = null;
            marks.push(k);
            writes.add(k, M.layers[layer].name);
            invalidate(true);
        }
    };
    cv.onpointercancel = cv.onpointerup;
    cv.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();
            if (!zoom) {
                return;
            }
            const px = mirror !== turn ? innerWidth - e.clientX : e.clientX;
            const py = turn ? innerHeight - e.clientY : e.clientY;
            const a = ox + px / zoom;
            const b = oy + py / zoom;
            zoom = Math.min(zoomLimit(), Math.max(0.03, zoom * Math.exp(-e.deltaY * 0.0022)));
            ox = a - px / zoom;
            oy = b - py / zoom;
            invalidate();
        },
        { passive: false },
    );
    const lab = (f) => () => {
        if (!el("marking").hidden) {
            f();
        }
    };
    const nudge = (dx, dy) => () => {
        ox += (dx * innerWidth * 0.6) / zoom;
        oy += (dy * innerHeight * 0.6) / zoom;
        invalidate();
    };
    Shortcuts.bind({
        wake: invalidate,
        start: () => {
            then = performance.now();
        },
        move: true,
        panLeft: nudge(-1, 0),
        panRight: nudge(1, 0),
        panUp: nudge(0, -1),
        panDown: nudge(0, 1),
        grab: () => {
            panning = true;
            cv.style.cursor = "grab";
        },
        grabEnd: () => {
            panning = false;
            cv.style.cursor = tool ? "crosshair" : "";
        },
        layerUp: () => setLayer(layer + 1),
        layerDown: () => setLayer(layer - 1),
        next: spots.length ? () => go(spot + 1) : null,
        previous: spots.length ? () => go(Math.max(spot, 0) - 1) : null,
        voxel,
        boxes: () => {
            boxes = !boxes;
            invalidate(true);
        },
        suggest: M.suggested.length ? suggest : null,
        mark: lab(() => useTool("ink")),
        erase: lab(() => useTool("erase")),
        spot: lab(() => useTool("spot")),
        undo: lab(undo),
        marks: lab(toggleMarks),
        greek: letters,
        brushDown: () => {
            brush = Math.max(0.02, brush / 1.25);
            invalidate(true);
        },
        brushUp: () => {
            brush = Math.min(20, brush * 1.25);
            invalidate(true);
        },
        swap: () => {
            if (taught >= 0 && stock >= 0) {
                setLayer(layer === taught ? stock : taught);
            }
        },
        yellow: tinted,
        heat: heated,
        floorDown: () => weakestBy(-8),
        floorUp: () => weakestBy(8),
        scanDown: () => scanBy(-0.05),
        scanUp: () => scanBy(0.05),
        mirror: () => el("mirror").click(),
        turn: () => {
            turn = !turn;
            invalidate();
        },
        full,
        back: () => {
            if (!document.fullscreenElement) {
                location.href = M.back;
            }
        },
    });
    el("keyhelp").onclick = () => Shortcuts.help(true);

    function closeMenus() {
        for (const o of document.querySelectorAll(".menu.open")) {
            o.classList.remove("open");
            o.querySelector(".opener").setAttribute("aria-expanded", "false");
        }
    }
    for (const m of document.querySelectorAll(".menu")) {
        m.querySelector(".opener").onclick = (e) => {
            e.stopPropagation();
            const open = !m.classList.contains("open");
            closeMenus();
            if (open && el("labelling").classList.contains("open")) {
                el("labelopen").click();
            }
            m.classList.toggle("open", open);
            m.querySelector(".opener").setAttribute("aria-expanded", open);
        };
    }
    addEventListener("pointerdown", (e) => {
        if (!e.target.closest(".menu")) {
            closeMenus();
        }
    });
    for (const b of document.querySelectorAll(".menu .items button")) {
        b.addEventListener("click", () => closeMenus());
    }
    el("labelopen").onclick = () => {
        closeMenus();
        const open = el("labelling").classList.toggle("open");
        el("labelopen").setAttribute("aria-pressed", open);
        el("labelopen").setAttribute("aria-expanded", open);
        if (!open && tool) {
            useTool(tool);
        }
    };
    if (M.scan && location.protocol.startsWith("http")) {
        fetch(`/marks?scan=${M.scan}${page ? `&page=${encodeURIComponent(page)}` : ""}`)
            .then((r) => (r.ok ? r.json() : Promise.reject()))
            .then((all) => {
                marks = all.map((k) => placed({ ...k, old: true }));
                el("marking").hidden = false;
                el("labelopen").disabled = false;
                invalidate(true);
            })
            .catch(() => {});
    }
    const nav = document.querySelector("nav");
    new ResizeObserver(() =>
        document.documentElement.style.setProperty("--toolbar-height", `${nav.offsetHeight}px`),
    ).observe(nav);
    for (const button of document.querySelectorAll(".menu>.opener")) {
        button.setAttribute("aria-expanded", "false");
    }
    addEventListener(
        "keydown",
        (e) => {
            if (e.key !== "Escape") {
                return;
            }
            const open = document.querySelector(".menu.open");
            if (open) {
                open.classList.remove("open");
                open.querySelector(".opener").setAttribute("aria-expanded", "false");
                open.querySelector(".opener").focus();
                e.stopImmediatePropagation();
            } else if (el("labelling").classList.contains("open")) {
                el("labelopen").click();
                el("labelopen").focus();
                e.stopImmediatePropagation();
            }
        },
        true,
    );

    sheet.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        programs = null;
    });
    sheet.addEventListener("webglcontextrestored", () => {
        setup();
        invalidate();
    });
    const viewer = {
        go,
        voxel,
        fit,
        view: () => ({
            zoom,
            ox,
            oy,
            layer,
            capacity,
            made,
            resident: slotOf.size,
            busy,
            queued: queue.length,
            inflight: inflight.size,
            retry: retryAt.size,
            free: free.length,
        }),
        onframe: null,
    };
    window.viewer = viewer;
    setup();
    addEventListener("resize", size);
    size();
    if (M.boxes.length) {
        go(0);
    } else {
        fit(M.area);
    }
    if (M.suggested.length) {
        el("suggested").disabled = false;
        el("suggested").title = "Next suggested area (J)";
        el("suggested").onclick = suggest;
    }
})();
