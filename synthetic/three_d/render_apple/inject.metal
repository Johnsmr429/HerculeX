struct Inject {
    int slabs, rows, cols, depth;
    int air_blocks, air_size;
    uint seed;
    float voxel_src, t_real, kollema_um, overlap_um, edge_voxels, air_level;
    float wave_um, wave_scale_um, swell_um, swell_scale_um, lump, lump_scale_um;
};

static uint mix_bits(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float reflect_into(float v, float n) {
    float period = 2.0f * (n - 1.0f);
    float r = v - period * floor(v / period);
    return r > n - 1.0f ? period - r : r;
}

static float slab_at(
    device const uchar *slabs, constant Inject &q, int k, float row, float col, float d) {
    float r = reflect_into(row, float(q.rows)), c = reflect_into(col, float(q.cols));
    float e = clamp(d + float(q.depth), 0.0f, float(2 * q.depth));
    int r0 = min(int(r), q.rows - 2), c0 = min(int(c), q.cols - 2),
        e0 = min(int(e), 2 * q.depth - 1);
    float fr = r - r0, fc = c - c0, fe = e - e0;
    int span = 2 * q.depth + 1;
    device const uchar *base = slabs + ((size_t(k) * q.rows + r0) * q.cols + c0) * span + e0;
    size_t row_step = size_t(q.cols) * span;
    float v = 0;
    for (int dr = 0; dr < 2; dr++)
        for (int dc = 0; dc < 2; dc++)
            for (int de = 0; de < 2; de++) {
                float w = (dr ? fr : 1 - fr) * (dc ? fc : 1 - fc) * (de ? fe : 1 - fe);
                v += w * float(base[dr * row_step + dc * span + de]);
            }
    return v;
}

static float cube_at(device const half *air, int k, int m, float3 local) {
    local = clamp(local, 0.0f, float(m) - 1.001f);
    int3 i0 = int3(local);
    float3 f = local - float3(i0);
    float v = 0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                float w = (dz ? f.x : 1 - f.x) * (dy ? f.y : 1 - f.y) * (dx ? f.z : 1 - f.z);
                v += w * float(air[((size_t(k) * m + i0.x + dz) * m + i0.y + dy) * m + i0.z + dx]);
            }
    return v;
}

static float air_at(device const half *air, constant Inject &q, float z, float y, float x) {
    int m = q.air_size;
    float spacing = 0.5f * float(m);
    float3 v = float3(z, y, x) / q.voxel_src / spacing - 0.5f;
    float3 base = floor(v);
    float total = 0, weights = 0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                float3 c = base + float3(dz, dy, dx);
                float3 t = 1.0f - fabs(v - c);
                float w = t.x * t.y * t.z;
                int3 ic = int3(c);
                uint h = mix_bits(uint(ic.x) * 73856093U ^ uint(ic.y) * 19349663U ^
                    uint(ic.z) * 83492791U ^ q.seed);
                float3 local = (v - c) * spacing + 0.5f * float(m);
                total += w * (cube_at(air, int(h % uint(q.air_blocks)), m, local) - q.air_level);
                weights += w * w;
            }
    return q.air_level + total / sqrt(max(weights, 1e-6f));
}

static float lattice(int2 c, uint seed) {
    uint h = mix_bits(uint(c.x) * 0x8da6b343U ^ uint(c.y) * 0xd8163841U ^ seed);
    return float(h & 0xffffU) / 32767.5f - 1.0f;
}

static float value_noise(float2 v, uint seed) {
    float2 b = floor(v), f = v - b;
    float2 w = f * f * (3.0f - 2.0f * f);
    int2 c = int2(b);
    float a = lattice(c, seed), bb = lattice(c + int2(1, 0), seed);
    float cc = lattice(c + int2(0, 1), seed), d = lattice(c + int2(1, 1), seed);
    return mix(mix(a, bb, w.x), mix(cc, d, w.x), w.y);
}

static float sheet_noise(
    constant Inject &q, int turn, float arc, float zeta, float scale_um, uint salt) {
    return value_noise(float2(arc, zeta) / scale_um, q.seed ^ salt ^ (uint(turn) * 0x27d4eb2dU));
}

struct Texel {
    int slab;
    float row, col;
};

static Texel texel_of(constant Inject &q, float arc, float zeta, int shift) {
    float along = arc + shift * q.kollema_um;
    int kollema = int(floor(along / q.kollema_um));
    float pos = along - kollema * q.kollema_um;
    uint h = mix_bits(uint(kollema) * 2654435761U ^ q.seed);
    Texel t;
    t.slab = int(h % uint(q.slabs));
    t.col = pos / q.voxel_src + float(mix_bits(h ^ 0x9e3779b9U) % uint(q.cols));
    t.row = zeta / q.voxel_src + float(mix_bits(h ^ 0x85ebca6bU) % uint(q.rows));
    return t;
}

static float sheet_add(constant Inject &q, device const uchar *slabs, thread const Sheets &o, int j,
    float d_in, int shift) {
    int turn = o.lo + j - 1;
    float arc = o.arc[j], zeta = o.f.zeta;
    d_in += q.wave_um * sheet_noise(q, turn, arc, zeta, q.wave_scale_um, 0x51ed270bU) +
        q.swell_um * sheet_noise(q, turn, arc, zeta, q.swell_scale_um, 0x3c6ef372U);
    float thickness = o.thickness[j] *
        max(1.0f + q.lump * sheet_noise(q, turn, arc, zeta, q.lump_scale_um, 0xa54ff53aU), 0.3f);
    float d = d_in * (q.t_real / max(thickness, 1.0f)) / q.voxel_src;
    float full = 0.5f * q.t_real / q.voxel_src + 1.0f, a = fabs(d);
    if (a >= full + q.edge_voxels)
        return 0.0f;
    float w = a <= full ? 1.0f : 0.5f + 0.5f * cos(M_PI_F * (a - full) / q.edge_voxels);
    Texel t = texel_of(q, o.arc[j], o.f.zeta, shift);
    return w * (slab_at(slabs, q, t.slab, t.row, t.col, d) - q.air_level);
}

kernel void inject(constant Params &p [[buffer(0)]], constant Inject &q [[buffer(1)]],
    device const float *map [[buffer(2)]], device const uchar *flags [[buffer(3)]],
    device const float *tables [[buffer(4)]], device const float *end [[buffer(5)]],
    device const float *tiles [[buffer(6)]], device const float *cracks [[buffer(7)]],
    device const uchar *slabs [[buffer(8)]], device const half *air [[buffer(9)]],
    device uchar *out [[buffer(10)]], uint3 g [[thread_position_in_grid]]) {
    if (int(g.x) >= p.bx || int(g.y) >= p.by || int(g.z) >= p.bz)
        return;
    int cells = p.nz * p.ny * p.nx, table = p.rows * p.per_turn * p.knots;
    Scene s;
    s.p = &p;
    s.a = map;
    s.b = map + cells;
    s.bad = flags;
    s.mask = flags + cells;
    s.radius = tables;
    s.thick = tables + table;
    s.delam = tables + 2 * table;
    s.arc = tables + 3 * table;
    s.arc_step = tables + 4 * table;
    s.end = end;
    s.tiles = tiles;
    s.cracks = cracks;
    float z = p.oz + g.z * p.voxel, y = p.oy + g.y * p.voxel, x = p.ox + g.x * p.voxel;
    uint index = ((g.z * uint(p.by) + g.y) * uint(p.bx) + g.x) * 2;
    float cy = y, cx = x;
    bool opening = uncrack(s, cy, cx);
    Sheets o;
    lookup(s, z, cy, cx, o.f);
    if (p.masked && !o.f.mask) {
        out[index] = 0;
        out[index + 1] = 0;
        return;
    }
    o.f.invalid = o.f.invalid || opening;
    sheets(s, o);
    float value = air_at(air, q, z, y, x);
    uchar label = 0;
    int nearest = -1;
    float best = 1e30f;
    for (int j = 0; j < 4; j++) {
        if (!o.exists[j])
            continue;
        float d_in = -o.distance[j];
        value += sheet_add(q, slabs, o, j, d_in, 0);
        float pos = o.arc[j] - q.kollema_um * floor(o.arc[j] / q.kollema_um);
        if (pos < q.overlap_um) {
            float extra = sheet_add(q, slabs, o, j, d_in + o.thickness[j], -1);
            value += extra;
            if (extra > 8.0f && fabs(d_in + o.thickness[j]) < fabs(d_in))
                label |= 64;
        }
        if (fabs(d_in) < best) {
            best = fabs(d_in);
            nearest = j;
        }
    }
    if (nearest >= 0) {
        label |= uchar(1 + (max(o.lo + nearest - 1, 0) % 63));
        if (best <= 0.5f * o.thickness[nearest])
            label |= 128;
    }
    out[index] = uchar(clamp(rint(value), 0.0f, 255.0f));
    out[index + 1] = label;
}
