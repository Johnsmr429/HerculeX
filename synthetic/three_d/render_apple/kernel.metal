#include <metal_stdlib>
using namespace metal;

constant int OFFSETS = 6;
constant int OFFSET[6] = {-2, -1, 0, 1, 2, 3};
constant float TWO_PI = 6.283185307179586f;
constant float SQRT2 = 1.4142135623730951f;
constant float SQRT_2PI = 2.5066282746310002f;

struct Params {
    int nz, ny, nx;
    float z0, dz, y0, x0, step;
    int rows, per_turn, knots;
    float z_first, knot_um, recto;
    float ply_dip, ply_dip_um, surface_dip, surface_dip_um, fibre, fray, compress_max,
        hole_threshold, hole_soft;
    int tile;
    float tile_um;
    float gap, papyrus, psf_um, phase, noise;
    int masked, n1, n2;
    float crinkle_um, specks, speck_threshold;
    float unsharp, unsharp_um, gain_sd;
    int n_cracks;
    float fog_um, debris, debris_um;
    int bz, by, bx;
    float oz, oy, ox, voxel;
    float noise1_z, noise1_y, noise1_x, noise2_z, noise2_y, noise2_x;
    float grain, grain_cell_um, noise_air;
    int grain_n;
    float rings, ring_um, ring_cy, ring_cx;
    int ring_n;
};

enum { RECTO, VERSO, HOLE, FRAY, CRINKLE, SPECK, GAIN, DEBRIS, DEBRIS_PLACE };

struct Scene {
    constant Params *p;
    device const float *a, *b;
    device const uchar *bad, *mask;
    device const float *radius, *thick, *delam, *arc, *arc_step, *end;
    device const float *tiles;
    device const float *noise1, *noise2, *grain, *ring;
    device const float *cracks;
};

struct Frame {
    float rho, theta, zeta, stretch;
    bool invalid, mask;
};

struct Where {
    int i0, k0;
    float w, v;
};

struct Sheets {
    Frame f;
    int lo;
    bool exists[4];
    float distance[4], thickness[4], rest_thickness[4], delamination[4], arc[4], tight[4],
        pitch_above[4];
};

static bool uncrack(thread const Scene &s, thread float &y, thread float &x) {
    bool gap = false;
    for (int k = s.p->n_cracks - 1; k >= 0; k--) {
        device const float *c = s.cracks + 12 * k;
        float ry = y - c[0], rx = x - c[1];
        float ty = c[3], tx = -c[2];
        float along = ry * ty + rx * tx, across = ry * c[2] + rx * c[3];
        float q = along / c[4];
        float f = (1.0f - q * q) * (1.0f + c[7] * q);
        float phase = TWO_PI * along / c[9];
        across -= c[8] * (sin(phase + 1.3f) + 0.5f * sin(2.05f * phase + 0.4f));
        if (f <= 0)
            continue;
        if (across > c[5] * f) {
            y -= f * (c[5] * c[2] + c[6] * ty);
            x -= f * (c[5] * c[3] + c[6] * tx);
        } else if (across >= 0) {
            gap = true;
        }
    }
    return gap;
}

static void lookup(thread const Scene &s, float z, float y, float x, thread Frame &f) {
    constant Params &p = *s.p;
    float cz = (z - p.z0) / p.dz, cy = (y - p.y0) / p.step, cx = (x - p.x0) / p.step;
    bool outside = cy < 0 || cy > p.ny - 1 || cx < 0 || cx > p.nx - 1;
    cz = clamp(cz, 0.0f, float(p.nz - 1));
    cy = clamp(cy, 0.0f, float(p.ny - 1));
    cx = clamp(cx, 0.0f, float(p.nx - 1));
    int iz = min(int(floor(cz)), p.nz - 2), iy = min(int(floor(cy)), p.ny - 2),
        ix = min(int(floor(cx)), p.nx - 2);
    float wz = cz - iz, wy = cy - iy, wx = cx - ix;
    float3 sc = float3(1.0f / p.dz, 1.0f / p.step, 1.0f / p.step);
    int k0 = (iz * p.ny + iy) * p.nx + ix;
    float a0 = s.a[k0], b0 = s.b[k0];
    float da = 0, db = 0, m = 0;
    float3 ga = 0, gb = 0;
    bool bad = false;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int k = k0 + (dz * p.ny + dy) * p.nx + dx;
                float ca = s.a[k] - a0, cb = s.b[k] - b0;
                float fz = dz ? wz : 1 - wz, fy = dy ? wy : 1 - wy, fx = dx ? wx : 1 - wx;
                float sz = dz ? 1 : -1, sy = dy ? 1 : -1, sx = dx ? 1 : -1;
                float3 dw = float3(sz * fy * fx, fz * sy * fx, fz * fy * sx) * sc;
                da += ca * fz * fy * fx;
                db += cb * fz * fy * fx;
                ga += ca * dw;
                gb += cb * dw;
                bad = bad || s.bad[k] > 0;
                m += s.mask[k] * fz * fy * fx;
            }
    float a = a0 + da, b = b0 + db;
    float rho = sqrt(a * a + b * b);
    float r = max(rho, 1e-9f);
    float3 g = (a * ga + b * gb) / r;
    float size = length(g);
    f.rho = rho;
    float t = fmod(atan2(b, a), TWO_PI);
    if (t < 0)
        t += TWO_PI;
    if (t >= TWO_PI)
        t -= TWO_PI;
    f.theta = t;
    f.zeta = z;
    f.stretch = 1.0f / max(size, 1e-6f);
    f.invalid = bad || outside;
    f.mask = (m >= 0.5f) && !outside;
}

static Where where_of(constant Params &p, float theta, float zeta) {
    Where q;
    float u = theta / TWO_PI * p.per_turn;
    q.i0 = int(floor(u));
    q.w = u - q.i0;
    float kz = clamp((zeta - p.z_first) / p.knot_um, 0.0f, float(p.knots - 1));
    q.k0 = min(int(floor(kz)), p.knots - 2);
    q.v = kz - q.k0;
    return q;
}

static float at(constant Params &p, device const float *data, int m, thread const Where &q) {
    int rows = p.rows * p.per_turn;
    int a = clamp(m * p.per_turn + q.i0, 0, rows - 1);
    int b = min(a + 1, rows - 1);
    device const float *ra = data + a * p.knots + q.k0, *rb = data + b * p.knots + q.k0;
    float w = q.w, v = q.v, r0 = ra[0];
    return r0 + (rb[0] - r0) * w * (1 - v) + (ra[1] - r0) * (1 - w) * v + (rb[1] - r0) * w * v;
}

static float wrapped(float d, float period) { return d - period * rint(d / period); }

static float at_arc(thread const Scene &s, int m, thread const Where &q) {
    constant Params &p = *s.p;
    int rows = p.rows * p.per_turn;
    int a = clamp(m * p.per_turn + q.i0, 0, rows - 1);
    int b = min(a + 1, rows - 1);
    int ia = a * p.knots + q.k0, ib = b * p.knots + q.k0;
    float w = q.w, v = q.v, r0 = s.arc[ia];
    float along = wrapped(s.arc[ib] - r0, p.tile * p.tile_um);
    float up = s.arc_step[ia], both = along + s.arc_step[ib];
    return r0 + along * w * (1 - v) + up * (1 - w) * v + both * w * v;
}

static void locate(thread const Scene &s, float rho, thread const Where &q, thread int &lo_out,
    thread float &frac) {
    constant Params &p = *s.p;
    int hi = p.rows - 1, lo = -1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (at(p, s.radius, max(mid, 0), q) <= rho)
            lo = mid;
        else
            hi = mid;
    }
    int base = max(lo, 0);
    float r0 = at(p, s.radius, base, q), r1 = at(p, s.radius, base + 1, q);
    frac = (rho - r0) / max(r1 - r0, 1e-6f);
    if (lo < 0)
        frac += 1.0f;
    lo_out = lo;
}

static float end_turn(thread const Scene &s, float theta, float zeta) {
    constant Params &p = *s.p;
    int i0 = int(floor(theta / TWO_PI * p.per_turn)) % p.per_turn;
    if (i0 < 0)
        i0 += p.per_turn;
    float kz = clamp((zeta - p.z_first) / p.knot_um, 0.0f, float(p.knots - 1));
    return s.end[i0 * p.knots + int(rint(kz))];
}

static float wrap(float x, float n) {
    float r = x - n * floor(x / n);
    return r >= n ? r - n : (r < 0 ? 0 : r);
}

static float tile_at(thread const Scene &s, int which, float arc, float z, float shift) {
    constant Params &p = *s.p;
    device const float *tile = s.tiles + which * p.tile * p.tile;
    float n = float(p.tile);
    float offset = shift * 0.6180339887f * n;
    float u = wrap(arc / p.tile_um + offset, n);
    float v = wrap(z / p.tile_um + 0.37f * offset, n);
    int i0 = int(floor(u)), j0 = int(floor(v));
    float fu = u - i0, fv = v - j0;
    i0 %= p.tile;
    j0 %= p.tile;
    int i1 = (i0 + 1) % p.tile, j1 = (j0 + 1) % p.tile;
    return tile[i0 * p.tile + j0] * (1 - fu) * (1 - fv) + tile[i1 * p.tile + j0] * fu * (1 - fv) +
        tile[i0 * p.tile + j1] * (1 - fu) * fv + tile[i1 * p.tile + j1] * fu * fv;
}

static void sheets(thread const Scene &s, thread Sheets &o) {
    constant Params &p = *s.p;
    thread Frame &f = o.f;
    Where q = where_of(p, f.theta, f.zeta);
    float frac;
    locate(s, f.rho, q, o.lo, frac);
    float end = end_turn(s, f.theta, f.zeta);
    float fraction = f.theta / TWO_PI;
    float radius[OFFSETS], thick[OFFSETS], delam[OFFSETS], arc[OFFSETS];
    bool exists[OFFSETS];
    for (int i = 0; i < OFFSETS; i++) {
        int m = o.lo + OFFSET[i];
        int safe = max(m, 0);
        radius[i] = at(p, s.radius, safe, q);
        delam[i] = at(p, s.delam, safe, q);
        thick[i] = at(p, s.thick, safe, q) + delam[i];
        arc[i] = (i >= 1 && i <= 4) ? at_arc(s, safe, q) : 0.0f;
        exists[i] = m >= 0 && (m + fraction <= end) && !f.invalid;
    }
    float squeeze[OFFSETS - 1], ratio[OFFSETS - 1];
    for (int i = 0; i < OFFSETS - 1; i++) {
        float pitch = (radius[i + 1] - radius[i]) * f.stretch;
        float mean = 0.5f * (thick[i + 1] + thick[i]);
        ratio[i] = pitch / max(mean, 1e-6f);
        squeeze[i] = min(ratio[i], 1.0f);
    }
    for (int j = 0; j < 4; j++) {
        int i = j + 1;
        float below = exists[i - 1] ? squeeze[i - 1] : 1.0f;
        float above = exists[i + 1] ? squeeze[i] : 1.0f;
        float factor = min(below, above);
        float tb = exists[i - 1] ? ratio[i - 1] : 1e30f, ta = exists[i + 1] ? ratio[i] : 1e30f;
        o.tight[j] = min(tb, ta);
        o.pitch_above[j] = (radius[i + 1] - radius[i]) * f.stretch;
        o.exists[j] = exists[i];
        o.distance[j] = (f.rho - radius[i]) * f.stretch -
            p.crinkle_um * tile_at(s, CRINKLE, arc[i], f.zeta, 11);
        o.thickness[j] = thick[i] * factor;
        o.rest_thickness[j] = thick[i];
        o.delamination[j] = delam[i] * factor;
        o.arc[j] = arc[i];
    }
}

static float erf_f(float x) {
    float z = fabs(x);
    float t = 1.0f / (1.0f + 0.5f * z);
    float tail = -0.82215223f + t * 0.17087277f;
    tail = -1.13520398f + t * (1.48851587f + t * tail);
    tail = -0.18628806f + t * (0.27886807f + t * tail);
    tail = 0.37409196f + t * (0.09678418f + t * tail);
    float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * tail));
    float e = 1.0f - r;
    return x < 0 ? -e : e;
}

static float erf_s(float z) { return z > 4.0f ? 1.0f : (z < -4.0f ? -1.0f : erf_f(z)); }

static float box(float u, float lo, float hi, float sigma) {
    return 0.5f * (erf_s((u - lo) / (SQRT2 * sigma)) - erf_s((u - hi) / (SQRT2 * sigma)));
}

static float edge_curvature(float x, float sigma) {
    float d = x / sigma;
    if (d > 7.0f || d < -7.0f)
        return 0.0f;
    return -x / (sigma * sigma * sigma * SQRT_2PI) * exp(-0.5f * d * d);
}

static float dip(float u, float centre, float width, float sigma) {
    float s = sqrt(width * width + sigma * sigma);
    float d = (u - centre) / s;
    if (d > 7.0f || d < -7.0f)
        return 0.0f;
    return (width / s) * exp(-0.5f * d * d);
}

static float profile(constant Params &p, float u, float lo, float split, float dl, float hi,
    float fr, float fv, float sigma) {
    float value = box(u, lo, split, sigma) * fr + box(u, split + dl, hi, sigma) * fv;
    value -= p.ply_dip * dip(u, split + dl / 2, p.ply_dip_um, sigma);
    return value -
        p.surface_dip * (dip(u, lo, p.surface_dip_um, sigma) + dip(u, hi, p.surface_dip_um, sigma));
}

static void density(thread const Scene &s, thread const Sheets &o, float sigma, float grain,
    thread float &total_out, thread float &curvature_out) {
    constant Params &p = *s.p;
    float total = 0, curvature = 0, r = p.recto, zeta = o.f.zeta;
    float wide = sqrt(sigma * sigma + p.unsharp_um * p.unsharp_um);
    float far = sqrt((p.unsharp > 0 ? wide * wide : sigma * sigma) + p.fog_um * p.fog_um);
    float dw = max(p.ply_dip_um, p.surface_dip_um);
    float reach = 6.0f * sqrt(far * far + dw * dw);
    for (int j = 0; j < 4; j++) {
        if (p.debris > 0 && o.exists[j]) {
            float gap = o.pitch_above[j] - o.thickness[j];
            if (gap > 20.0f) {
                float amount = clamp(tile_at(s, DEBRIS, o.arc[j], zeta, 19) - 1.0f, 0.0f, 1.0f);
                if (amount > 0) {
                    float place = clamp(
                        0.5f + 0.35f * tile_at(s, DEBRIS_PLACE, o.arc[j], zeta, 23), 0.15f, 0.85f);
                    float centre = o.thickness[j] / 2 + gap * place;
                    float line = dip(o.distance[j], centre, p.debris_um, sigma);
                    if (p.unsharp > 0)
                        line = (1 + p.unsharp) * line -
                            p.unsharp * dip(o.distance[j], centre, p.debris_um, wide);
                    total += p.debris * amount * line;
                }
            }
        }
        if (!o.exists[j] || fabs(o.distance[j]) > 0.75f * o.thickness[j] + reach)
            continue;
        float arc = o.arc[j];
        float t0 = o.thickness[j];
        float t = t0 * clamp(1.0f + p.fray * tile_at(s, FRAY, arc, zeta, 7), 0.5f, 1.5f);
        float u = o.distance[j], dl = o.delamination[j];
        float solid = max(t - dl, 1e-3f);
        float lo = -t / 2, hi = t / 2, split = lo + r * solid;
        float speck = tile_at(s, SPECK, arc, zeta, 13) - p.speck_threshold;
        speck = speck > 0 ? p.specks * speck : 0.0f;
        float fr = 1.0f + p.fibre * tile_at(s, RECTO, arc, zeta, 0) + speck;
        float fv = 1.0f + p.fibre * tile_at(s, VERSO, arc, zeta, 1) + speck;
        float haze = p.fog_um * clamp((1.6f - o.tight[j]) / 0.6f, 0.0f, 1.0f);
        float near = sqrt(sigma * sigma + haze * haze);
        float value = profile(p, u, lo, split, dl, hi, fr, fv, near);
        if (p.unsharp > 0)
            value = (1 + p.unsharp) * value -
                p.unsharp *
                    profile(p, u, lo, split, dl, hi, fr, fv, sqrt(wide * wide + haze * haze));
        else if (value < 0)
            value = 0;
        if (p.grain > 0)
            value *= grain;
        value *= exp(p.gain_sd * tile_at(s, GAIN, arc, zeta, 17));
        float squeeze = clamp(o.rest_thickness[j] / max(t0, 1e-6f), 1.0f, p.compress_max);
        float hole =
            clamp((tile_at(s, HOLE, arc, zeta, 3) - p.hole_threshold) / p.hole_soft, 0.0f, 1.0f);
        float weight = squeeze * hole;
        total += weight * value;
        if (p.phase != 0)
            curvature += weight *
                (edge_curvature(u - lo, sigma) - edge_curvature(u - split, sigma) +
                    edge_curvature(u - (split + dl), sigma) - edge_curvature(u - hi, sigma));
    }
    total_out = total;
    curvature_out = curvature;
}

static float noise_tile(device const float *tile, int n, float3 c) {
    int3 i = int3(floor(c));
    float3 fr = c - float3(i);
    float value = 0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                float w = (dz ? fr.x : 1 - fr.x) * (dy ? fr.y : 1 - fr.y) * (dx ? fr.z : 1 - fr.z);
                int a = (i.x + dz) % n, b = (i.y + dy) % n, cc = (i.z + dx) % n;
                value += w * tile[(a * n + b) * n + cc];
            }
    return value;
}

static float noise_at(thread const Scene &s, uint3 g) {
    constant Params &p = *s.p;
    float n1 = float(p.n1), n2 = float(p.n2);
    float3 c1 =
        float3(wrap(p.noise1_z + g.z, n1), wrap(p.noise1_y + g.y, n1), wrap(p.noise1_x + g.x, n1));
    float3 c2 =
        float3(wrap(p.noise2_z + g.z, n2), wrap(p.noise2_y + g.y, n2), wrap(p.noise2_x + g.x, n2));
    return (noise_tile(s.noise1, p.n1, c1) + noise_tile(s.noise2, p.n2, c2)) / SQRT2;
}

kernel void render(constant Params &p [[buffer(0)]], device const float *map [[buffer(1)]],
    device const uchar *flags [[buffer(2)]], device const float *tables [[buffer(3)]],
    device const float *end [[buffer(4)]], device const float *tiles [[buffer(5)]],
    device const float *noise [[buffer(6)]], device const float *cracks [[buffer(7)]],
    device uchar *out [[buffer(8)]], uint3 g [[thread_position_in_grid]]) {
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
    s.noise1 = noise;
    s.noise2 = noise + p.n1 * p.n1 * p.n1;
    s.grain = s.noise2 + p.n2 * p.n2 * p.n2;
    s.ring = s.grain + p.grain_n * p.grain_n * p.grain_n;
    s.cracks = cracks;
    float z = p.oz + g.z * p.voxel, y = p.oy + g.y * p.voxel, x = p.ox + g.x * p.voxel;
    uint index = (g.z * uint(p.by) + g.y) * uint(p.bx) + g.x;
    float sigma = sqrt(p.psf_um * p.psf_um + p.voxel * p.voxel / 12.0f);
    float cy = y, cx = x;
    bool gap = uncrack(s, cy, cx);
    Sheets o;
    lookup(s, z, cy, cx, o.f);
    if (p.masked && !o.f.mask) {
        out[index] = 0;
        return;
    }
    o.f.invalid = o.f.invalid || gap;
    sheets(s, o);
    float grain = 1.0f;
    if (p.grain > 0) {
        float n = float(p.grain_n), cell = p.grain_cell_um;
        float3 c = float3(wrap(z / cell, n), wrap(y / cell, n), wrap(x / cell, n));
        grain = max(1.0f + p.grain * noise_tile(s.grain, p.grain_n, c), 0.0f);
    }
    float total, curvature;
    density(s, o, sigma, grain, total, curvature);
    float contrast = p.papyrus - p.gap;
    float value = p.gap + contrast * total - p.phase * contrast * sigma * sigma * curvature;
    value +=
        p.noise * (p.noise_air + (1 - p.noise_air) * clamp(total, 0.0f, 1.0f)) * noise_at(s, g);
    if (p.rings > 0) {
        float u = wrap(
            sqrt((y - p.ring_cy) * (y - p.ring_cy) + (x - p.ring_cx) * (x - p.ring_cx)) / p.ring_um,
            float(p.ring_n));
        int i = int(floor(u));
        float f = u - i;
        value += p.rings * (s.ring[i % p.ring_n] * (1 - f) + s.ring[(i + 1) % p.ring_n] * f);
    }
    out[index] = uchar(clamp(rint(value), 0.0f, 255.0f));
}
