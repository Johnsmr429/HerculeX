#include "kernel.c"

#define SYN_TILE 512
#define SYN_TILE_UM 20.0
#define SYN_GRAIN 64
#define SYN_NOISE_1 61
#define SYN_NOISE_2 67
#define SYN_RING 4096

typedef struct {
    SynDeform deform;
    SynRoll roll;
    SynKernel kernel;
} SynScroll;

static void syn_mask(Arena *a, SynScroll *s, double reach) {
    SynDeform *d = &s->deform;
    size_t stride = (size_t)d->ny * d->nx, n = stride * d->nz;
    u8 *bad = pushz(a, n), *mask = pushz(a, n);
    s->kernel.bad = bad;
    s->kernel.mask = mask;
    size_t mark = a->used;
    int *queue = push(a, stride * sizeof(*queue));
    u8 *state = push(a, stride);
    double limit = reach * reach;
    int range = ceil(reach);
    for (int z = 0; z < d->nz; z++) {
        size_t at = (size_t)z * stride;
        for (int y = 0; y < d->ny; y++) {
            for (int x = 0; x < d->nx; x++, at++) {
                size_t next[] = {at + 1, at + d->nx, at + stride};
                int open[] = {x + 1 < d->nx, y + 1 < d->ny, z + 1 < d->nz};
                for (int k = 0; k < 3; k++) {
                    if (open[k] &&
                        hypot((double)d->a[next[k]] - d->a[at], (double)d->b[next[k]] - d->b[at]) /
                                (k == 2 ? d->dz : d->step) >
                            30) {
                        bad[at] = bad[next[k]] = 1;
                    }
                }
            }
        }
    }
    SynRoll r = s->roll;
    for (int z = 0; z < d->nz; z++) {
        const float *x = d->a + (size_t)z * stride, *y = d->b + (size_t)z * stride;
        const u8 *flags = bad + (size_t)z * stride;
        double height = d->z0 + z * d->dz;
        int knot = nearbyint(syn_clip((height - r.z_first) / r.knot_um, 0, r.knots - 1));
        memset(state, 0, stride);
        int inside = 0;
        for (size_t j = 0; j < stride; j++) {
            if (flags[j]) {
                continue;
            }
            double radius = hypot(x[j], y[j]), theta = syn_wrap(atan2(y[j], x[j]));
            int sample = (int)floor(theta / (2 * M_PI) * r.per_turn) % r.per_turn;
            double end = r.end[(size_t)sample * r.knots + knot];
            if (syn_winding(r, radius, theta, height) > end + .5) {
                continue;
            }
            inside++;
            int py = j / d->nx, px = j % d->nx;
            int first_y = fmax(0, py - range), last_y = fmin(d->ny - 1, py + range);
            int first_x = fmax(0, px - range), last_x = fmin(d->nx - 1, px + range);
            for (int dy = first_y; dy <= last_y; dy++) {
                for (int dx = first_x; dx <= last_x; dx++) {
                    double a = dy - py, b = dx - px;
                    if (a * a + b * b <= limit) {
                        state[(size_t)dy * d->nx + dx] = 1;
                    }
                }
            }
        }
        if (!inside) {
            for (int y = 0; y < d->ny; y++) {
                for (int x = 0; x < d->nx; x++) {
                    double dy = y + 1, dx = x;
                    state[(size_t)y * d->nx + x] = dy * dy + dx * dx <= limit;
                }
            }
        }
        size_t tail = 0;
        for (int y = 0; y < d->ny; y++) {
            for (int x = 0; x < d->nx; x++) {
                int i = y * d->nx + x;
                if (!state[i] && (x == 0 || y == 0 || x == d->nx - 1 || y == d->ny - 1)) {
                    state[i] = 2;
                    queue[tail++] = i;
                }
            }
        }
        for (size_t head = 0; head < tail; head++) {
            int at = queue[head], x = at % d->nx, y = at / d->nx;
            int next[] = {at - 1, at + 1, at - d->nx, at + d->nx};
            int open[] = {x > 0, x + 1 < d->nx, y > 0, y + 1 < d->ny};
            for (int k = 0; k < 4; k++) {
                if (open[k] && !state[next[k]]) {
                    state[next[k]] = 2;
                    queue[tail++] = next[k];
                }
            }
        }
        for (size_t j = 0; j < stride; j++) {
            mask[(size_t)z * stride + j] = state[j] != 2;
        }
    }
    a->used = mark;
}

static void syn_texture(Arena *a, json_object *spec, const char *name, float *out, int dims,
    const int *shape, const double *sigma, int fibre, int transpose) {
    size_t n = 1;
    for (int i = 0; i < dims; i++) {
        n *= shape[i];
    }
    size_t mark = a->used;
    double *data = push(a, n * sizeof(*data)), *scratch = push(a, n * sizeof(*scratch));
    Pcg64 rng = syn_rng(spec, name);
    syn_field(a, &rng, data, scratch, dims, shape, sigma, 4, 1);
    if (fibre) {
        double *second = push(a, n * sizeof(*second));
        const double fine[] = {sigma[0] / 4, sigma[1] / 2};
        syn_field(a, &rng, second, scratch, dims, shape, fine, 4, 1);
        for (size_t i = 0; i < n; i++) {
            data[i] += 0.5 * second[i];
        }
    }
    syn_center(data, scratch, n);
    for (size_t i = 0; i < n; i++) {
        size_t j = transpose ? i % shape[0] * shape[1] + i / shape[0] : i;
        out[i] = data[j];
    }
    a->used = mark;
}

static double syn_scroll_noise(
    Arena *a, json_object *spec, double voxel, float *noise1, float *noise2) {
    json_object *ct = jget(spec, "ct");
    double ny = jf64(ct, "noise_um"), nz = jf64(ct, "noise_z_um");
    double base_voxel = jf64(spec, "voxel_um"), noise = jf64(ct, "noise");
    if (ji64(spec, "version") >= 4) {
        double scale = nz <= 0 ? ny : pow(ny * ny * nz, 1.0 / 3);
        noise =
            noise * pow(fmin(1, scale / voxel), 1.5) / pow(fmin(1, scale / (2 * base_voxel)), 1.5);
    }
    double sig = ny / voxel / 2.355;
    double noise_sigma[] = {nz > 0 ? nz / voxel / 2.355 : sig, sig, sig};
    double key = nearbyint(voxel * 10000) / 10000;
    size_t mark = a->used;
    const char *text = python_float(a, key);
    syn_texture(a, spec, fmt(a, "noise.%d.%s", SYN_NOISE_1, text), noise1, 3,
        (int[]){SYN_NOISE_1, SYN_NOISE_1, SYN_NOISE_1}, noise_sigma, 0, 0);
    syn_texture(a, spec, fmt(a, "noise.%d.%s", SYN_NOISE_2, text), noise2, 3,
        (int[]){SYN_NOISE_2, SYN_NOISE_2, SYN_NOISE_2}, noise_sigma, 0, 0);
    a->used = mark;
    return noise;
}

static void syn_scroll(
    Arena *a, SynScroll *s, json_object *spec, const SynBox *window, double voxel, int workers) {
    s->roll = syn_roll(a, spec);
    syn_deform(a, &s->deform, spec, s->roll, window, workers);
    SynDeform *d = &s->deform;
    SynRoll r = s->roll;
    json_object *sheet = jget(spec, "sheet"), *ct = jget(spec, "ct"),
                *cracks = jget(spec, "cracks");
    int version = ji64(spec, "version"), n_cracks = json_object_array_length(cracks);
    double holes = jf64(sheet, "holes");
    float *tile = push(a, (size_t)9 * SYN_TILE * SYN_TILE * sizeof(*tile));
    size_t size1 = SYN_NOISE_1 * SYN_NOISE_1 * SYN_NOISE_1;
    size_t size2 = SYN_NOISE_2 * SYN_NOISE_2 * SYN_NOISE_2;
    float *noise1 =
        push(a, (size1 + size2 + SYN_GRAIN * SYN_GRAIN * SYN_GRAIN + SYN_RING) * sizeof(*noise1));
    float *noise2 = noise1 + size1, *grain = noise2 + size2;
    float *ring = grain + SYN_GRAIN * SYN_GRAIN * SYN_GRAIN;
    double *crack = pushz(a, (n_cracks ? n_cracks : 1) * 12 * sizeof(*crack));
    for (int i = 0; i < n_cracks; i++) {
        json_object *c = json_object_array_get_idx(cracks, i);
        SynPoint center = syn_pair(c, "centre_um");
        double angle = jf64(c, "angle_deg") * (M_PI / 180);
        double values[] = {center.y, center.x, sin(angle), cos(angle), jf64(c, "half_length_um"),
            jf64(c, "open_um"), jf64(c, "slip_um"), syn_optional(c, "wedge", 0),
            syn_optional(c, "wiggle_um", 0), syn_optional(c, "wiggle_wavelength_um", 1000)};
        memcpy(crack + i * 12, values, sizeof(values));
    }
    const int shape[] = {SYN_TILE, SYN_TILE};
    const char *names[] = {"tile.recto", "tile.verso", "tile.holes", "tile.fray", "tile.crinkle",
        "tile.specks", "tile.gain", "tile.debris", "tile.debris.place"};
    SynPoint fibre = syn_pair(sheet, "fibre_um");
    double blur[][2] = {{fibre.y, fibre.x}, {fibre.x, fibre.y},
        {jf64(sheet, "hole_um"), jf64(sheet, "hole_um")},
        {jf64(sheet, "fray_um"), jf64(sheet, "fray_um")},
        {jf64(sheet, "crinkle_scale_um"), jf64(sheet, "crinkle_scale_um")},
        {jf64(sheet, "speck_um"), jf64(sheet, "speck_um")},
        {jf64(sheet, "gain_um"), jf64(sheet, "gain_um")}, {300, 300}, {800, 800}};
    const float *tiles[9];
    for (int j = 0; j < 9; j++) {
        float *out = tile + (size_t)j * SYN_TILE * SYN_TILE;
        const double sig[] = {blur[j][0] / SYN_TILE_UM, blur[j][1] / SYN_TILE_UM};
        syn_texture(a, spec, names[j], out, 2, shape, sig, j < 2, j == 1 && version < 7);
        tiles[j] = out;
    }
    double noise = syn_scroll_noise(a, spec, voxel, noise1, noise2);
    syn_texture(a, spec, "tile.grain", grain, 3, (int[]){SYN_GRAIN, SYN_GRAIN, SYN_GRAIN},
        (double[]){1, 1, 1}, 0, 0);
    syn_texture(a, spec, "rings", ring, 1, (int[]){SYN_RING}, (double[]){0.7}, 0, 0);
    s->kernel = (SynKernel){.nz = d->nz,
        .ny = d->ny,
        .nx = d->nx,
        .z0 = d->z0,
        .dz = d->dz,
        .y0 = d->box.y0,
        .x0 = d->box.x0,
        .step = d->step,
        .a = d->a,
        .b = d->b,
        .rows = r.rows,
        .per_turn = r.per_turn,
        .knots = r.knots,
        .z_first = r.z_first,
        .knot_um = r.knot_um,
        .core = r.core,
        .recto = r.recto,
        .radius = r.radius,
        .thick = r.thickness,
        .delam = r.delamination,
        .arc = r.arc,
        .end = r.end,
        .ply_dip = jf64(sheet, "ply_dip"),
        .ply_dip_um = jf64(sheet, "ply_dip_um"),
        .surface_dip = jf64(sheet, "surface_dip"),
        .surface_dip_um = jf64(sheet, "surface_dip_um"),
        .fibre = jf64(sheet, "fibre"),
        .fray = jf64(sheet, "fray"),
        .compress_max = jf64(sheet, "compress_max"),
        .hole_soft = 0.4,
        .hole_threshold =
            holes > 0 ? syn_quantile(0.5 * (1 + syn_clip(2 * holes - 1, -.999999, .999999))) : -1e9,
        .tile = SYN_TILE,
        .tile_um = SYN_TILE_UM,
        .recto_tile = tiles[0],
        .verso_tile = tiles[1],
        .hole_tile = tiles[2],
        .fray_tile = tiles[3],
        .gap = jf64(ct, "gap"),
        .papyrus = jf64(ct, "papyrus"),
        .psf_um = jf64(ct, "psf_um"),
        .phase = jf64(ct, "phase"),
        .noise = noise,
        .masked = json_object_get_boolean(jget(ct, "mask")),
        .n1 = SYN_NOISE_1,
        .n2 = SYN_NOISE_2,
        .noise1 = noise1,
        .noise2 = noise2,
        .crinkle_um = jf64(sheet, "crinkle_um"),
        .specks = jf64(sheet, "specks"),
        .speck_threshold = jf64(sheet, "speck_threshold"),
        .crinkle_tile = tiles[4],
        .speck_tile = tiles[5],
        .unsharp = jf64(ct, "unsharp"),
        .unsharp_um = jf64(ct, "unsharp_um"),
        .gain_sd = jf64(sheet, "gain_sd"),
        .gain_tile = tiles[6],
        .n_cracks = n_cracks,
        .cracks = crack,
        .fog_um = jf64(sheet, "fog_um"),
        .debris = jf64(sheet, "debris"),
        .debris_um = jf64(sheet, "debris_um"),
        .debris_tile = tiles[7],
        .debris_place_tile = tiles[8],
        .grain = jf64(sheet, "grain"),
        .grain_cell_um = jf64(sheet, "grain_um") / 2,
        .noise_air = jf64(ct, "noise_air"),
        .grain_n = SYN_GRAIN,
        .grain_tile = grain,
        .rings = jf64(ct, "rings"),
        .ring_um = jf64(ct, "ring_um"),
        .ring_cy = d->center.y,
        .ring_cx = d->center.x,
        .ring_n = SYN_RING,
        .ring_table = ring};
    syn_mask(a, s, jf64(ct, "mask_um") / d->step);
}

typedef struct {
    double winding, nearest, sheet_winding, second, pitch, thickness, normal[3];
    i32 sheet;
    u8 valid, mask;
} SynTruthPoint;

static SynTruthPoint syn_truth_at(const SynKernel *kernel, double z, double y, double x) {
    SynKernelSheets at = {.lo = -1};
    syn_k_sheets(kernel, z, y, x, &at);
    SynTruthPoint t = {.winding = at.lo + at.frac,
        .nearest = NAN,
        .second = INFINITY,
        .pitch = INFINITY,
        .sheet = -1,
        .valid = !at.f.invalid,
        .mask = at.f.mask};
    memcpy(t.normal, at.f.normal, sizeof(t.normal));
    int nearest = -1;
    double distance = INFINITY;
    for (int j = 0; j < 4; j++) {
        if (!at.exists[j]) {
            continue;
        }
        double d = fabs(at.distance[j]);
        if (d < distance) {
            nearest = j;
            t.second = distance;
            distance = d;
        } else {
            t.second = fmin(t.second, d);
        }
    }
    for (int j = 0; j < 3; j++) {
        double a = at.distance[j], b = at.distance[j + 1];
        if (at.exists[j] && at.exists[j + 1] && a >= 0 && b < 0) {
            t.winding = at.m[j] + a / fmax(a - b, 1e-9);
        }
    }
    double fraction = at.f.theta / (2 * M_PI);
    t.winding = t.valid ? t.winding + fraction : NAN;
    if (nearest >= 0) {
        t.nearest = at.distance[nearest];
        t.sheet = at.m[nearest];
        t.sheet_winding = t.sheet + fraction;
        t.thickness = at.thickness[nearest];
        for (int j = 0; j < 4; j++) {
            if (j != nearest && at.exists[j]) {
                t.pitch = fmin(t.pitch, fabs(at.distance[j] - t.nearest));
            }
        }
    }
    return t;
}
