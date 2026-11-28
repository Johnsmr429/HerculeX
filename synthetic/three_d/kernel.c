#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t nz, ny, nx;
    double z0, dz, y0, x0, step;
    const float *a, *b;
    const uint8_t *bad, *mask;
    int64_t rows, per_turn, knots;
    double z_first, knot_um, core, recto;
    const double *radius, *thick, *delam, *arc, *end;
    double ply_dip, ply_dip_um, surface_dip, surface_dip_um, fibre, fray, compress_max,
        hole_threshold, hole_soft;
    int64_t tile;
    double tile_um;
    const float *recto_tile, *verso_tile, *hole_tile, *fray_tile;
    double gap, papyrus, psf_um, phase, noise;
    int64_t masked, n1, n2;
    const float *noise1, *noise2;
    double crinkle_um, specks, speck_threshold;
    const float *crinkle_tile, *speck_tile;
    double unsharp, unsharp_um, gain_sd;
    const float *gain_tile;
    int64_t n_cracks;
    const double *cracks;
    double fog_um;
    double debris, debris_um;
    const float *debris_tile, *debris_place_tile;
    double grain, grain_cell_um, noise_air;
    int64_t grain_n;
    const float *grain_tile;
    double rings, ring_um, ring_cy, ring_cx;
    int64_t ring_n;
    const float *ring_table;
} SynKernel;

static int syn_k_uncrack(const SynKernel *s, double *y, double *x) {
    int gap = 0;
    for (int64_t k = s->n_cracks - 1; k >= 0; k--) {
        const double *c = s->cracks + 12 * k;
        double ry = *y - c[0], rx = *x - c[1];
        double ty = c[3], tx = -c[2];
        double s_along = ry * ty + rx * tx, s_across = ry * c[2] + rx * c[3];
        double q = s_along / c[4];
        double f = (1.0 - q * q) * (1.0 + c[7] * q);
        double phase = 6.283185307179586 * s_along / c[9];
        s_across -= c[8] * (sin(phase + 1.3) + 0.5 * sin(2.05 * phase + 0.4));
        if (f <= 0)
            continue;
        if (s_across > c[5] * f) {
            *y -= f * (c[5] * c[2] + c[6] * ty);
            *x -= f * (c[5] * c[3] + c[6] * tx);
        } else if (s_across >= 0) {
            gap = 1;
        }
    }
    return gap;
}

#define SYN_OFFSETS 6
static const int SYN_OFFSET[SYN_OFFSETS] = {-2, -1, 0, 1, 2, 3};
static const double SYN_TAU = 6.283185307179586;
static const double SYN_SQRT2 = 1.4142135623730951;
static const double SYN_SQRT_TAU = 2.5066282746310002;

typedef struct {
    double rho, theta, zeta, stretch, normal[3];
    int invalid, mask;
} SynKernelFrame;

static void syn_k_lookup(const SynKernel *s, double z, double y, double x, SynKernelFrame *f) {
    double cz = (z - s->z0) / s->dz, cy = (y - s->y0) / s->step, cx = (x - s->x0) / s->step;
    int outside = cy < 0 || cy > s->ny - 1 || cx < 0 || cx > s->nx - 1;
    if (cz < 0)
        cz = 0;
    if (cz > s->nz - 1)
        cz = s->nz - 1;
    if (cy < 0)
        cy = 0;
    if (cy > s->ny - 1)
        cy = s->ny - 1;
    if (cx < 0)
        cx = 0;
    if (cx > s->nx - 1)
        cx = s->nx - 1;
    int64_t iz = (int64_t)floor(cz), iy = (int64_t)floor(cy), ix = (int64_t)floor(cx);
    if (iz > s->nz - 2)
        iz = s->nz - 2;
    if (iy > s->ny - 2)
        iy = s->ny - 2;
    if (ix > s->nx - 2)
        ix = s->nx - 2;
    double wz = cz - iz, wy = cy - iy, wx = cx - ix;
    double sc[3] = {1.0 / s->dz, 1.0 / s->step, 1.0 / s->step};
    double a = 0, b = 0, ga[3] = {0, 0, 0}, gb[3] = {0, 0, 0}, m = 0;
    int bad = 0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int64_t k = ((iz + dz) * s->ny + (iy + dy)) * s->nx + (ix + dx);
                double ca = s->a[k], cb = s->b[k];
                double fz = dz ? wz : 1 - wz, fy = dy ? wy : 1 - wy, fx = dx ? wx : 1 - wx;
                double sz = dz ? 1 : -1, sy = dy ? 1 : -1, sx = dx ? 1 : -1;
                a += ca * fz * fy * fx;
                b += cb * fz * fy * fx;
                ga[0] += ca * sz * fy * fx * sc[0];
                ga[1] += ca * fz * sy * fx * sc[1];
                ga[2] += ca * fz * fy * sx * sc[2];
                gb[0] += cb * sz * fy * fx * sc[0];
                gb[1] += cb * fz * sy * fx * sc[1];
                gb[2] += cb * fz * fy * sx * sc[2];
                bad |= s->bad[k] > 0;
                m += s->mask[k] * fz * fy * fx;
            }
    double rho = hypot(a, b);
    double r = rho > 1e-9 ? rho : 1e-9;
    double g[3], size = 0;
    for (int i = 0; i < 3; i++) {
        g[i] = (a * ga[i] + b * gb[i]) / r;
        size += g[i] * g[i];
    }
    size = sqrt(size);
    double n = size > 1e-12 ? size : 1e-12;
    for (int i = 0; i < 3; i++)
        f->normal[i] = g[i] / n;
    f->rho = rho;
    double t = fmod(atan2(b, a), SYN_TAU);
    if (t < 0)
        t += SYN_TAU;
    if (t >= SYN_TAU)
        t -= SYN_TAU;
    f->theta = t;
    f->zeta = z;
    f->stretch = 1.0 / (size > 1e-6 ? size : 1e-6);
    f->invalid = bad || outside;
    f->mask = (m >= 0.5) && !outside;
}

typedef struct {
    int64_t i0, k0;
    double w, v;
} SynKernelWhere;

static void syn_k_where(const SynKernel *s, double theta, double zeta, SynKernelWhere *q) {
    double u = theta / SYN_TAU * s->per_turn;
    q->i0 = (int64_t)floor(u);
    q->w = u - q->i0;
    double kz = (zeta - s->z_first) / s->knot_um;
    if (kz < 0)
        kz = 0;
    if (kz > s->knots - 1.000001)
        kz = s->knots - 1.000001;
    q->k0 = (int64_t)floor(kz);
    q->v = kz - q->k0;
}

static inline double syn_k_at(
    const SynKernel *s, const double *data, int64_t m, const SynKernelWhere *q) {
    int64_t rows = s->rows * s->per_turn;
    int64_t a = m * s->per_turn + q->i0;
    if (a < 0)
        a = 0;
    if (a > rows - 1)
        a = rows - 1;
    int64_t b = a + 1;
    if (b > rows - 1)
        b = rows - 1;
    const double *ra = data + a * s->knots + q->k0, *rb = data + b * s->knots + q->k0;
    double w = q->w, v = q->v;
    return ra[0] * (1 - w) * (1 - v) + rb[0] * w * (1 - v) + ra[1] * (1 - w) * v + rb[1] * w * v;
}

static void syn_k_locate(
    const SynKernel *s, double rho, const SynKernelWhere *q, int64_t *lo_out, double *frac) {
    int64_t lo = -2, g = *lo_out;
    if (g >= 0 && g <= s->rows - 3) {
        double r0 = syn_k_at(s, s->radius, g, q), r1 = syn_k_at(s, s->radius, g + 1, q);
        if (r0 <= rho && rho < r1)
            lo = g;
        else if (r1 <= rho && g + 1 <= s->rows - 3 && rho < syn_k_at(s, s->radius, g + 2, q))
            lo = g + 1;
        else if (rho < r0 && g >= 1 && syn_k_at(s, s->radius, g - 1, q) <= rho)
            lo = g - 1;
    }
    if (lo == -2) {
        int64_t hi = s->rows - 1;
        lo = -1;
        while (hi - lo > 1) {
            int64_t mid = (lo + hi) / 2;
            if (syn_k_at(s, s->radius, mid > 0 ? mid : 0, q) <= rho)
                lo = mid;
            else
                hi = mid;
        }
    }
    int64_t base = lo > 0 ? lo : 0;
    double r0 = syn_k_at(s, s->radius, base, q), r1 = syn_k_at(s, s->radius, base + 1, q);
    double d = r1 - r0;
    *frac = (rho - r0) / (d > 1e-6 ? d : 1e-6);
    if (lo < 0)
        *frac += 1.0;
    *lo_out = lo;
}

static double syn_k_end(const SynKernel *s, double theta, double zeta) {
    int64_t i0 = ((int64_t)floor(theta / SYN_TAU * s->per_turn)) % s->per_turn;
    if (i0 < 0)
        i0 += s->per_turn;
    double kz = (zeta - s->z_first) / s->knot_um;
    if (kz < 0)
        kz = 0;
    if (kz > s->knots - 1)
        kz = s->knots - 1;
    int64_t k = (int64_t)nearbyint(kz);
    return s->end[i0 * s->knots + k];
}

typedef struct {
    SynKernelFrame f;
    int64_t lo;
    double frac;
    int64_t m[4];
    int exists[4];
    double distance[4], thickness[4], rest_thickness[4], delamination[4], arc[4], tight[4],
        pitch_above[4];
} SynKernelSheets;

static double syn_k_tile(const SynKernel *s, const float *tile, double arc, double z, double shift);

static void syn_k_sheets(const SynKernel *s, double z, double y, double x, SynKernelSheets *o) {
    int gap = syn_k_uncrack(s, &y, &x);
    syn_k_lookup(s, z, y, x, &o->f);
    o->f.invalid |= gap;
    SynKernelFrame *f = &o->f;
    SynKernelWhere q;
    syn_k_where(s, f->theta, f->zeta, &q);
    syn_k_locate(s, f->rho, &q, &o->lo, &o->frac);
    double end = syn_k_end(s, f->theta, f->zeta);
    double fraction = f->theta / SYN_TAU;
    double radius[SYN_OFFSETS], thick[SYN_OFFSETS], delam[SYN_OFFSETS], arc[SYN_OFFSETS];
    int exists[SYN_OFFSETS];
    int64_t ms[SYN_OFFSETS];
    for (int i = 0; i < SYN_OFFSETS; i++) {
        int64_t m = o->lo + SYN_OFFSET[i];
        int64_t safe = m > 0 ? m : 0;
        ms[i] = m;
        radius[i] = syn_k_at(s, s->radius, safe, &q);
        delam[i] = syn_k_at(s, s->delam, safe, &q);
        thick[i] = syn_k_at(s, s->thick, safe, &q) + delam[i];
        arc[i] = (i >= 1 && i <= 4) ? syn_k_at(s, s->arc, safe, &q) : 0.0;
        exists[i] = m >= 0 && (m + fraction <= end) && !f->invalid;
    }
    double squeeze[SYN_OFFSETS - 1], ratio[SYN_OFFSETS - 1];
    for (int i = 0; i < SYN_OFFSETS - 1; i++) {
        double pitch = (radius[i + 1] - radius[i]) * f->stretch;
        double mean = 0.5 * (thick[i + 1] + thick[i]);
        ratio[i] = pitch / (mean > 1e-6 ? mean : 1e-6);
        squeeze[i] = ratio[i] < 1.0 ? ratio[i] : 1.0;
    }
    for (int j = 0; j < 4; j++) {
        int i = j + 1;
        double below = exists[i - 1] ? squeeze[i - 1] : 1.0;
        double above = exists[i + 1] ? squeeze[i] : 1.0;
        double factor = below < above ? below : above;
        double tb = exists[i - 1] ? ratio[i - 1] : INFINITY,
               ta = exists[i + 1] ? ratio[i] : INFINITY;
        o->tight[j] = tb < ta ? tb : ta;
        o->pitch_above[j] = (radius[i + 1] - radius[i]) * f->stretch;
        o->m[j] = ms[i];
        o->exists[j] = exists[i];
        o->distance[j] = (f->rho - radius[i]) * f->stretch -
            s->crinkle_um * syn_k_tile(s, s->crinkle_tile, arc[i], f->zeta, 11);
        o->thickness[j] = thick[i] * factor;
        o->rest_thickness[j] = thick[i];
        o->delamination[j] = delam[i] * factor;
        o->arc[j] = arc[i];
    }
}

static inline double syn_k_wrap(double x, double n) {
    double r = x - n * floor(x / n);
    return r >= n ? r - n : (r < 0 ? 0 : r);
}

static inline double syn_k_erf(double z) { return z > 4.0 ? 1.0 : (z < -4.0 ? -1.0 : erf(z)); }

static double syn_k_box(double u, double lo, double hi, double sigma) {
    return 0.5 *
        (syn_k_erf((u - lo) / (SYN_SQRT2 * sigma)) - syn_k_erf((u - hi) / (SYN_SQRT2 * sigma)));
}

static double syn_k_edge(double x, double sigma) {
    double d = x / sigma;
    if (d > 7.0 || d < -7.0)
        return 0.0;
    return -x / (sigma * sigma * sigma * SYN_SQRT_TAU) * exp(-0.5 * d * d);
}

static double syn_k_dip(double u, double centre, double width, double sigma) {
    double s = sqrt(width * width + sigma * sigma);
    double d = (u - centre) / s;
    if (d > 7.0 || d < -7.0)
        return 0.0;
    return (width / s) * exp(-0.5 * d * d);
}

static double syn_k_tile(
    const SynKernel *s, const float *tile, double arc, double z, double shift) {
    double n = (double)s->tile;
    double offset = shift * 0.6180339887 * n;
    double u = syn_k_wrap(arc / s->tile_um + offset, n);
    double v = syn_k_wrap(z / s->tile_um + 0.37 * offset, n);
    int64_t i0 = (int64_t)floor(u), j0 = (int64_t)floor(v);
    double fu = u - i0, fv = v - j0;
    i0 %= s->tile;
    j0 %= s->tile;
    int64_t i1 = (i0 + 1) % s->tile, j1 = (j0 + 1) % s->tile;
    return tile[i0 * s->tile + j0] * (1 - fu) * (1 - fv) + tile[i1 * s->tile + j0] * fu * (1 - fv) +
        tile[i0 * s->tile + j1] * (1 - fu) * fv + tile[i1 * s->tile + j1] * fu * fv;
}

static double syn_k_clip(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

static double syn_k_profile(const SynKernel *s, double u, double lo, double split, double dl,
    double hi, double fr, double fv, double sigma) {
    double value = syn_k_box(u, lo, split, sigma) * fr + syn_k_box(u, split + dl, hi, sigma) * fv;
    value -= s->ply_dip * syn_k_dip(u, split + dl / 2, s->ply_dip_um, sigma);
    return value -
        s->surface_dip *
        (syn_k_dip(u, lo, s->surface_dip_um, sigma) + syn_k_dip(u, hi, s->surface_dip_um, sigma));
}

static void syn_k_density(const SynKernel *s, const SynKernelSheets *o, double sigma, double grain,
    double *total_out, double *curvature_out) {
    double total = 0, curvature = 0, r = s->recto, zeta = o->f.zeta;
    double wide = sqrt(sigma * sigma + s->unsharp_um * s->unsharp_um);
    double far = sqrt((s->unsharp > 0 ? wide * wide : sigma * sigma) + s->fog_um * s->fog_um);
    double dw = s->ply_dip_um > s->surface_dip_um ? s->ply_dip_um : s->surface_dip_um;
    double reach = 6.0 * sqrt(far * far + dw * dw);
    for (int j = 0; j < 4; j++) {
        if (s->debris > 0 && o->exists[j]) {
            double gap = o->pitch_above[j] - o->thickness[j];
            if (gap > 20.0) {
                double amount =
                    syn_k_clip(syn_k_tile(s, s->debris_tile, o->arc[j], zeta, 19) - 1.0, 0.0, 1.0);
                if (amount > 0) {
                    double place = syn_k_clip(
                        0.5 + 0.35 * syn_k_tile(s, s->debris_place_tile, o->arc[j], zeta, 23), 0.15,
                        0.85);
                    double centre = o->thickness[j] / 2 + gap * place;
                    double line = syn_k_dip(o->distance[j], centre, s->debris_um, sigma);
                    if (s->unsharp > 0)
                        line = (1 + s->unsharp) * line -
                            s->unsharp * syn_k_dip(o->distance[j], centre, s->debris_um, wide);
                    total += s->debris * amount * line;
                }
            }
        }
        if (!o->exists[j] || fabs(o->distance[j]) > 0.75 * o->thickness[j] + reach)
            continue;
        double arc = o->arc[j];
        double t0 = o->thickness[j];
        double t =
            t0 * syn_k_clip(1.0 + s->fray * syn_k_tile(s, s->fray_tile, arc, zeta, 7), 0.5, 1.5);
        double u = o->distance[j], dl = o->delamination[j];
        double solid = t - dl > 1e-3 ? t - dl : 1e-3;
        double lo = -t / 2, hi = t / 2, split = lo + r * solid;
        double speck = syn_k_tile(s, s->speck_tile, arc, zeta, 13) - s->speck_threshold;
        speck = speck > 0 ? s->specks * speck : 0.0;
        double fr = 1.0 + s->fibre * syn_k_tile(s, s->recto_tile, arc, zeta, 0) + speck;
        double fv = 1.0 + s->fibre * syn_k_tile(s, s->verso_tile, arc, zeta, 1) + speck;
        double w = (1.6 - o->tight[j]) / 0.6;
        double haze = s->fog_um * (w < 0 ? 0 : (w > 1 ? 1 : w));
        double near = sqrt(sigma * sigma + haze * haze);
        double value = syn_k_profile(s, u, lo, split, dl, hi, fr, fv, near);
        if (s->unsharp > 0)
            value = (1 + s->unsharp) * value -
                s->unsharp *
                    syn_k_profile(s, u, lo, split, dl, hi, fr, fv, sqrt(wide * wide + haze * haze));
        else if (value < 0)
            value = 0;
        if (s->grain > 0)
            value *= grain;
        value *= exp(s->gain_sd * syn_k_tile(s, s->gain_tile, arc, zeta, 17));
        double squeeze =
            syn_k_clip(o->rest_thickness[j] / (t0 > 1e-6 ? t0 : 1e-6), 1.0, s->compress_max);
        double hole = syn_k_clip(
            (syn_k_tile(s, s->hole_tile, arc, zeta, 3) - s->hole_threshold) / s->hole_soft, 0.0,
            1.0);
        double weight = o->exists[j] ? squeeze * hole : 0.0;
        total += weight * value;
        curvature += weight *
            (syn_k_edge(u - lo, sigma) - syn_k_edge(u - split, sigma) +
                syn_k_edge(u - (split + dl), sigma) - syn_k_edge(u - hi, sigma));
    }
    *total_out = total;
    *curvature_out = curvature;
}

static double syn_k_noise(const SynKernel *s, double z, double y, double x, double voxel) {
    double p[3] = {z / voxel, y / voxel, x / voxel};
    double total = 0;
    const float *tiles[2] = {s->noise1, s->noise2};
    int64_t sizes[2] = {s->n1, s->n2};
    for (int t = 0; t < 2; t++) {
        int64_t n = sizes[t];
        const float *tile = tiles[t];
        double c[3], fr[3];
        int64_t i[3];
        for (int k = 0; k < 3; k++) {
            c[k] = syn_k_wrap(p[k], (double)n);
            i[k] = (int64_t)floor(c[k]);
            fr[k] = c[k] - i[k];
        }
        double value = 0;
        for (int dz = 0; dz < 2; dz++)
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    double w = (dz ? fr[0] : 1 - fr[0]) * (dy ? fr[1] : 1 - fr[1]) *
                        (dx ? fr[2] : 1 - fr[2]);
                    int64_t a = (i[0] + dz) % n, b = (i[1] + dy) % n, cc = (i[2] + dx) % n;
                    value += w * tile[(a * n + b) * n + cc];
                }
        total += value;
    }
    return total / SYN_SQRT2;
}

static double syn_k_tile3(const float *tile, int64_t n, double z, double y, double x) {
    double p[3] = {z, y, x}, c[3], fr[3];
    int64_t i[3];
    for (int k = 0; k < 3; k++) {
        c[k] = syn_k_wrap(p[k], (double)n);
        i[k] = (int64_t)floor(c[k]);
        fr[k] = c[k] - i[k];
    }
    double value = 0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                double w =
                    (dz ? fr[0] : 1 - fr[0]) * (dy ? fr[1] : 1 - fr[1]) * (dx ? fr[2] : 1 - fr[2]);
                int64_t a = (i[0] + dz) % n, b = (i[1] + dy) % n, cc = (i[2] + dx) % n;
                value += w * tile[(a * n + b) * n + cc];
            }
    return value;
}

static double syn_k_ct(
    const SynKernel *s, double z, double y, double x, double voxel, SynKernelSheets *o, int fast) {
    double sigma = sqrt(s->psf_um * s->psf_um + voxel * voxel / 12.0);
    if (s->masked && fast) {
        double cy = y, cx = x;
        syn_k_uncrack(s, &cy, &cx);
        syn_k_lookup(s, z, cy, cx, &o->f);
        if (!o->f.mask)
            return 0.0;
    }
    syn_k_sheets(s, z, y, x, o);
    double grain = 1.0;
    if (s->grain > 0) {
        double cell = s->grain_cell_um;
        grain =
            1.0 + s->grain * syn_k_tile3(s->grain_tile, s->grain_n, z / cell, y / cell, x / cell);
        if (grain < 0)
            grain = 0;
    }
    double total, curvature;
    syn_k_density(s, o, sigma, grain, &total, &curvature);
    double contrast = s->papyrus - s->gap;
    double value = s->gap + contrast * total - s->phase * contrast * sigma * sigma * curvature;
    value += s->noise * (s->noise_air + (1 - s->noise_air) * syn_k_clip(total, 0.0, 1.0)) *
        syn_k_noise(s, z, y, x, voxel);
    if (s->rings > 0) {
        double u =
            syn_k_wrap(hypot(y - s->ring_cy, x - s->ring_cx) / s->ring_um, (double)s->ring_n);
        int64_t i = (int64_t)floor(u);
        double f = u - i;
        value += s->rings *
            (s->ring_table[i % s->ring_n] * (1 - f) + s->ring_table[(i + 1) % s->ring_n] * f);
    }
    if (s->masked && !o->f.mask)
        value = 0.0;
    return value;
}

typedef struct {
    const SynKernel *s;
    int64_t nz, ny, nx;
    double origin[3], voxel;
    uint8_t *out;
    int64_t next;
    pthread_mutex_t lock;
} SynRenderJob;

static void *syn_k_worker(void *arg) {
    SynRenderJob *job = arg;
    SynKernelSheets o;
    o.lo = -1;
    for (;;) {
        pthread_mutex_lock(&job->lock);
        int64_t row = job->next++;
        pthread_mutex_unlock(&job->lock);
        if (row >= job->nz * job->ny)
            break;
        int64_t k = row / job->ny, j = row % job->ny;
        double z = job->origin[0] + k * job->voxel, y = job->origin[1] + j * job->voxel;
        uint8_t *line = job->out + row * job->nx;
        for (int64_t i = 0; i < job->nx; i++) {
            double x = job->origin[2] + i * job->voxel;
            double v = nearbyint(syn_k_ct(job->s, z, y, x, job->voxel, &o, 1));
            line[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return NULL;
}

void hx_syn_render(const SynKernel *s, int64_t nz, int64_t ny, int64_t nx, const double *origin,
    double voxel, uint8_t *out, int threads) {
    SynRenderJob job = {.s = s,
        .nz = nz,
        .ny = ny,
        .nx = nx,
        .origin = {origin[0], origin[1], origin[2]},
        .voxel = voxel,
        .out = out};
    pthread_mutex_init(&job.lock, NULL);
    if (threads < 1)
        threads = 1;
    pthread_t *pool = malloc(sizeof(pthread_t) * threads);
    for (int t = 0; t < threads; t++)
        pthread_create(&pool[t], NULL, syn_k_worker, &job);
    for (int t = 0; t < threads; t++)
        pthread_join(pool[t], NULL);
    free(pool);
    pthread_mutex_destroy(&job.lock);
}

void hx_syn_points(const SynKernel *s, int64_t n, const double *points, double voxel,
    double *out_ct, double *out_frame, double *out_sheet) {
    SynKernelSheets o;
    for (int64_t p = 0; p < n; p++) {
        o.lo = -1;
        out_ct[p] = syn_k_ct(s, points[3 * p], points[3 * p + 1], points[3 * p + 2], voxel, &o, 0);
        double *f = out_frame + 8 * p;
        f[0] = o.f.rho;
        f[1] = o.f.theta;
        f[2] = o.f.stretch;
        f[3] = o.f.normal[0];
        f[4] = o.f.normal[1];
        f[5] = o.f.normal[2];
        f[6] = (double)o.lo;
        f[7] = o.frac;
        double *q = out_sheet + 16 * p;
        for (int j = 0; j < 4; j++) {
            q[4 * j] = (double)o.m[j];
            q[4 * j + 1] = o.exists[j];
            q[4 * j + 2] = o.distance[j];
            q[4 * j + 3] = o.thickness[j];
        }
    }
}
