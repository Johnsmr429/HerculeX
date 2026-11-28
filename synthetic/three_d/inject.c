#include <glob.h>

typedef struct {
    int slabs, rows, cols, depth, air_blocks, air_size;
    u32 seed;
    float voxel_src, thickness, kollema_um, overlap_um, edge_voxels, air_level;
    float wave_um, wave_scale_um, swell_um, swell_scale_um, lump, lump_scale_um;
} SynInjectParams;

typedef struct {
    SynInjectParams params;
    const u8 *slabs;
    const _Float16 *air;
    double voxel;
} SynInjectLibrary;

static float syn_air_sum(const float *air, size_t n) {
    if (n < 8) {
        float sum = -0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += air[i];
        }
        return sum;
    }
    if (n > 128) {
        size_t half = n / 2 & ~(size_t)7;
        return syn_air_sum(air, half) + syn_air_sum(air + half, n - half);
    }
    float parts[8];
    for (int i = 0; i < 8; i++) {
        parts[i] = air[i];
    }
    size_t i = 8;
    for (; i + 8 <= n; i += 8) {
        for (int j = 0; j < 8; j++) {
            parts[j] += air[i + j];
        }
    }
    float sum = ((parts[0] + parts[1]) + (parts[2] + parts[3])) +
        ((parts[4] + parts[5]) + (parts[6] + parts[7]));
    for (; i < n; i++) {
        sum += air[i];
    }
    return sum;
}

static SynInjectLibrary syn_inject_library(Arena *a, const char *look) {
    const char *root = fmt(a, "%s/synthetic/library/%s", OUTPUTS, look);
    glob_t found;
    int code = glob(fmt(a, "%s/slab-*.npz", root), 0, NULL, &found);
    need(!code && found.gl_pathc > 0, "no slabs for %s: run synthetic.library build first", look);
    size_t mark = a->used;
    Array first = npz_get(a, found.gl_pathv[0], "ct");
    need(first.ndim == 3 && !strcmp(first.descr, "|u1") && first.shape[0] >= 2 &&
            first.shape[1] >= 2 && first.shape[2] == 25 && first.size == (size_t)elements(first),
        "%s has an invalid CT slab", found.gl_pathv[0]);
    a->used = mark;
    u8 *slabs = push(a, first.size * found.gl_pathc);
    mark = a->used;
    for (size_t i = 0; i < found.gl_pathc; i++) {
        Array slab = npz_get(a, found.gl_pathv[i], "ct");
        need(slab.ndim == first.ndim && !strcmp(slab.descr, first.descr) &&
                !memcmp(slab.shape, first.shape, 3 * sizeof(i64)) && slab.size == first.size,
            "%s has an incompatible CT slab", found.gl_pathv[i]);
        memcpy(slabs + i * slab.size, slab.data, slab.size);
        a->used = mark;
    }
    int slab_count = (int)found.gl_pathc;
    globfree(&found);
    code = glob(fmt(a, "%s/library-*.json", root), 0, NULL, &found);
    need(!code && found.gl_pathc > 0, "no texture summary for %s", look);
    json_object *summary = json_read(found.gl_pathv[0]);
    need(summary != NULL, "cannot read %s", found.gl_pathv[0]);
    double voxel = jf64(summary, "voxel_um");
    json_object_put(summary);
    globfree(&found);
    need(isfinite(voxel) && voxel > 0, "invalid voxel size for texture %s", look);
    const char *air_path = fmt(a, "%s/air.npz", root);
    Array level = npz_get(a, air_path, "level");
    need(level.ndim == 0 && (!strcmp(level.descr, "<f8") || !strcmp(level.descr, "<f4")),
        "invalid air level for %s", look);
    double mean = level.size == 8 ? *(double *)level.data : *(float *)level.data;
    Array air = npz_get(a, air_path, "blocks");
    int byte_air = !strcmp(air.descr, "|u1");
    need(air.ndim == 4 && (byte_air || !strcmp(air.descr, "<f2")) && air.shape[0] > 0 &&
            air.shape[1] >= 2 && air.shape[1] == air.shape[2] && air.shape[2] == air.shape[3] &&
            air.size == (size_t)elements(air) * (byte_air ? 1 : sizeof(_Float16)),
        "invalid air blocks for %s", look);
    size_t n = (size_t)air.shape[1] * air.shape[2] * air.shape[3];
    _Float16 *blocks = push(a, n * air.shape[0] * sizeof(*blocks));
    mark = a->used;
    float *block = push(a, n * sizeof(*block));
    for (i64 i = 0; i < air.shape[0]; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t at = i * n + j;
            block[j] = byte_air ? ((u8 *)air.data)[at] : ((const _Float16 *)air.data)[at];
        }
        float middle = syn_air_sum(block, n) / (float)n;
        for (size_t j = 0; j < n; j++) {
            float value = block[j] - middle;
            blocks[i * n + j] = (_Float16)(value + (float)mean);
        }
    }
    a->used = mark;
    SynInjectParams params = {.slabs = slab_count,
        .rows = first.shape[0],
        .cols = first.shape[1],
        .depth = 12,
        .air_blocks = air.shape[0],
        .air_size = air.shape[1],
        .voxel_src = voxel,
        .thickness = 6 * voxel,
        .kollema_um = 120000,
        .overlap_um = 15000,
        .edge_voxels = 2,
        .air_level = mean,
        .wave_um = 14,
        .wave_scale_um = 180,
        .swell_um = 30,
        .swell_scale_um = 900,
        .lump = .35,
        .lump_scale_um = 260};
    return (SynInjectLibrary){params, slabs, blocks, voxel};
}

#ifdef __APPLE__
void *hx_inj_open(const char *source, const char *function, char *message, int length);
int hx_inj_buffers(void *handle, const void *const *data, const i64 *lengths, int count);
int hx_inj_render(void *handle, const void *params, i64 params_length, const void *inject,
    i64 inject_length, int nz, int ny, int nx, u8 *out, int channels, double *seconds);
void hx_inj_close(void *handle);

typedef struct {
    void *gpu;
    SynGpuParams params;
    SynInjectParams texture;
    double origin[3], radius, voxel, built, seconds;
} SynInject;

static SynInject syn_inject_open(Arena *a, const SynInjectLibrary *lib, json_object *spec,
    const char *look, u64 seed, SynScroll *truth) {
    double begin = now();
    size_t mark = a->used;
    json_object *calibrated = syn_complete(a, spec);
    if (!strcmp(look, "PHerc0139")) {
        json_object *roll = jget(calibrated, "roll");
        double gap = fmax(112 - jf64(roll, "thickness_um"), 5);
        json_object_object_add(roll, "gap_um", jnum(gap));
    }
    int workers = (int)strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), NULL, 10);
    if (workers <= 0) {
        workers = fmax(1, cores() - 2);
    }
    SynScroll scroll;
    syn_scroll(a, &scroll, calibrated, NULL, lib->voxel, workers);
    if (truth) {
        *truth = scroll;
        mark = a->used;
    }
    const SynKernel *s = &scroll.kernel;
    Bytes kernel = read_file(a, "synthetic/three_d/render_apple/kernel.metal");
    Bytes addon = read_file(a, "synthetic/three_d/render_apple/inject.metal");
    need(kernel.data && addon.data, "cannot read the Metal injection kernels");
    char *code = fmt(a, "%.*s\n%.*s", (int)kernel.size, kernel.data, (int)addon.size, addon.data);
    char message[4096];
    void *gpu = hx_inj_open(code, "inject", message, (int)sizeof(message));
    need(gpu != NULL, "injection kernel: %s", message);
    i64 maps = (i64)s->nz * s->ny * s->nx, endings = s->per_turn * s->knots;
    i64 table = s->rows * endings, n_cracks = fmax(s->n_cracks, 1);
    float *map = push(a, 2 * maps * sizeof(float));
    float *roll = push(a, 5 * table * sizeof(float));
    float *end = push(a, endings * sizeof(float));
    float *cracks = push(a, 12 * n_cracks * sizeof(float));
    u8 *flags = push(a, 2 * maps);
    memcpy(map, s->a, maps * sizeof(float));
    memcpy(map + maps, s->b, maps * sizeof(float));
    memcpy(flags, s->bad, maps);
    memcpy(flags + maps, s->mask, maps);
    for (i64 i = 0; i < table; i++) {
        roll[i] = s->radius[i];
        roll[table + i] = s->thick[i];
        roll[2 * table + i] = s->delam[i];
        roll[3 * table + i] = s->arc[i];
        roll[4 * table + i] = i % s->knots + 1 < s->knots ? s->arc[i + 1] - s->arc[i] : 0;
    }
    for (i64 i = 0; i < endings; i++) {
        end[i] = s->end[i];
    }
    for (i64 i = 0; i < 12 * n_cracks; i++) {
        cracks[i] = s->cracks[i];
    }
    float dummy[16] = {0};
    SynInjectParams q = lib->params;
    q.seed = (u32)seed;
    i64 slab_bytes = (i64)q.slabs * q.rows * q.cols * (2 * q.depth + 1);
    i64 air_bytes = (i64)q.air_blocks * q.air_size * q.air_size * q.air_size * sizeof(_Float16);
    const void *buffers[] = {map, flags, roll, end, dummy, cracks, lib->slabs, lib->air};
    i64 lengths[] = {2 * maps * sizeof(float), 2 * maps, 5 * table * sizeof(float),
        endings * sizeof(float), sizeof(dummy), 12 * n_cracks * sizeof(float), slab_bytes,
        air_bytes};
    need(!hx_inj_buffers(gpu, buffers, lengths, 8), "cannot allocate the Metal injection scene");
    SynGpuParams p = {.nz = s->nz,
        .ny = s->ny,
        .nx = s->nx,
        .z0 = s->z0,
        .dz = s->dz,
        .y0 = s->y0,
        .x0 = s->x0,
        .step = s->step,
        .rows = s->rows,
        .per_turn = s->per_turn,
        .knots = s->knots,
        .z_first = s->z_first,
        .knot_um = s->knot_um,
        .recto = s->recto,
        .gap = s->gap,
        .tile = 1,
        .tile_um = 1e9,
        .masked = s->masked,
        .n_cracks = s->n_cracks,
        .voxel = lib->voxel};
    SynBox box = scroll.deform.box;
    json_object *band = jget(calibrated, "band_um");
    double z = (json_object_get_double(json_object_array_get_idx(band, 0)) +
                   json_object_get_double(json_object_array_get_idx(band, 1))) /
        2;
    SynInject result = {.gpu = gpu,
        .params = p,
        .texture = q,
        .voxel = lib->voxel,
        .origin = {z, (box.y0 + box.y1) / 2, (box.x0 + box.x1) / 2},
        .radius = (box.y1 - box.y0) / 2,
        .built = now() - begin};
    json_object_put(calibrated);
    a->used = mark;
    return result;
}

static void syn_inject_origin(const SynInject *scene, int side, double at, double out[3]) {
    double half = scene->voxel * side / 2;
    out[0] = scene->origin[0];
    out[1] = scene->origin[1] + at * scene->radius - half;
    out[2] = scene->origin[2] - half;
}

static void syn_inject_render(
    SynInject *scene, const double origin[3], int nz, int ny, int nx, u8 *out) {
    SynGpuParams p = scene->params;
    p.by = ny;
    p.bx = nx;
    p.oy = origin[1];
    p.ox = origin[2];
    i64 stride = (i64)ny * nx;
    i64 share = (1 << 23) / stride;
    int step = share > 1 ? (int)share : 1;
    for (int z = 0; z < nz; z += step) {
        double seconds;
        p.bz = fmin(step, nz - z);
        p.oz = origin[0] + z * scene->voxel;
        int code = hx_inj_render(scene->gpu, &p, sizeof(p), &scene->texture, sizeof(scene->texture),
            p.bz, ny, nx, out + 2 * stride * z, 2, &seconds);
        need(!code, "injection kernel: render failed (%d)", code);
        scene->seconds += seconds;
    }
}

static size_t syn_inject_crops(Arena *a, const SynInjectLibrary *library, json_object *spec,
    const char *look, int count, int seed, u8 *images) {
    size_t mark = a->used;
    SynScroll scroll;
    SynInject scene = syn_inject_open(a, library, spec, look, seed, &scroll);
    const size_t n = 20000;
    double *y = push(a, n * sizeof(*y)), *x = push(a, n * sizeof(*x));
    int side = 2 * REALISM_SIDE;
    u8 *ct = push(a, 16 * REALISM_N);
    double half = REALISM_SIDE * scene.voxel, mid = scene.origin[0];
    SynBox box = scroll.deform.box;
    need(box.y1 - box.y0 >= 2 * half && box.x1 - box.x0 >= 2 * half,
        "scroll is too small for injected crops");
    Pcg64 rng = pcg64(seed);
    for (size_t i = 0; i < n; i++) {
        y[i] = syn_u(&rng, box.y0 + half, box.y1 - half);
    }
    for (size_t i = 0; i < n; i++) {
        x[i] = syn_u(&rng, box.x0 + half, box.x1 - half);
    }
    size_t found = 0, tries = 0;
    for (size_t i = 0; i < n && found < (size_t)count && tries < 20 * (size_t)count; i++) {
        SynTruthPoint truth = syn_truth_at(&scroll.kernel, mid, y[i], x[i]);
        if (!truth.mask || fabs(truth.nearest) > truth.thickness / 2) {
            continue;
        }
        tries++;
        double corner[] = {mid - .5 * scene.voxel, y[i] - half, x[i] - half};
        syn_inject_render(&scene, corner, 2, side, side, ct);
        int filled = realism_average(ct, images + found * REALISM_N, 0, 2);
        found += (double)filled / REALISM_N >= .85;
    }
    hx_inj_close(scene.gpu);
    a->used = mark;
    return found;
}

static int syn_inject_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.inject render FAMILY SEED OUT.npz "
                        "[--look PHerc0139] [--size 1024] [--slices 1] [--at 0.5]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc >= 3, "%s", usage);
    const char *look = "PHerc0139";
    int width = 1024, depth = 1;
    double at = .5;
    for (int i = 3; i < argc; i++) {
        const char *option = argv[i];
        need(i + 1 < argc, "%s", usage);
        const char *value = argv[++i];
        char *end;
        if (!strcmp(option, "--look")) {
            look = value;
        } else if (!strcmp(option, "--size") || !strcmp(option, "--slices")) {
            long count = strtol(value, &end, 10);
            need(end != value && !*end && count > 0 && count <= INT_MAX, "%s", usage);
            if (!strcmp(option, "--size")) {
                width = (int)count;
            } else {
                depth = (int)count;
            }
        } else if (!strcmp(option, "--at")) {
            at = strtod(value, &end);
            need(end != value && !*end && isfinite(at), "%s", usage);
        } else {
            die("%s", usage);
        }
    }
    char *end;
    errno = 0;
    i64 seed = strtoll(argv[1], &end, 10);
    need(end != argv[1] && !*end && !errno && seed >= 0, "%s", usage);
    size_t total = (size_t)width * width;
    need(total <= SIZE_MAX / (size_t)depth / 8, "injection dimensions are too large");
    total *= depth;
    Arena a = arena((8ull << 30) + 8 * total);
    json_object *spec = syn_family_draw(&a, argv[0], seed, 0, look);
    need(spec != NULL, "unknown synthetic case %s", argv[0]);
    SynInjectLibrary library = syn_inject_library(&a, look);
    SynInject scene = syn_inject_open(&a, &library, spec, look, seed, NULL);
    double origin[3];
    syn_inject_origin(&scene, width, at, origin);
    u8 *samples = push(&a, 2 * total), *ct = push(&a, total), *label = push(&a, total);
    syn_inject_render(&scene, origin, depth, width, width, samples);
    hx_inj_close(scene.gpu);
    size_t on = 0, outside = 0;
    for (size_t i = 0; i < total; i++) {
        ct[i] = samples[2 * i];
        label[i] = samples[2 * i + 1];
        on += label[i] >= 128;
        outside += label[i] == 0;
    }
    const char *text = json_object_to_json_string_ext(
        spec, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
    size_t bytes = strlen(text);
    u32 *unicode = push(&a, bytes * sizeof(*unicode));
    size_t length = 0;
    for (const unsigned char *s = (const unsigned char *)text; *s;) {
        u32 cp = *s++;
        if (cp >= 0xc0) {
            int more = cp >= 0xf0 ? 3 : (cp >= 0xe0 ? 2 : 1);
            cp &= (1u << (6 - more)) - 1;
            for (int i = 0; i < more && *s; i++) {
                cp = cp << 6 | (*s++ & 63);
            }
        }
        unicode[length++] = cp;
    }
    Array arrays[] = {
        {.descr = "|u1", .ndim = 3, .shape = {depth, width, width}, .data = ct, .size = total},
        {.descr = "|u1", .ndim = 3, .shape = {depth, width, width}, .data = label, .size = total},
        {.descr = "<f8", .ndim = 1, .shape = {3}, .data = origin, .size = sizeof(origin)},
        {.descr = "<f8", .data = &scene.voxel, .size = sizeof(scene.voxel)},
        {.data = unicode, .size = length * sizeof(*unicode)}};
    need(length < 100000, "synthetic spec is too large");
    snprintf(arrays[4].descr, sizeof(arrays[4].descr), "<U%zu", length);
    const char *members[] = {"ct", "label", "origin_um", "voxel_um", "spec"};
    npz_write_codec(argv[2], 5, members, arrays, ZIP_CM_DEFLATE);
    json_object *report = json_object_new_object(), *shape = json_object_new_array_ext(3);
    json_object_array_add(shape, json_object_new_int(depth));
    json_object_array_add(shape, json_object_new_int(width));
    json_object_array_add(shape, json_object_new_int(width));
    json_object_object_add(report, "family", json_object_new_string(argv[0]));
    json_object_object_add(report, "seed", json_object_new_int64(seed));
    json_object_object_add(report, "shape", shape);
    json_object_object_add(report, "voxel_um", jnum(scene.voxel));
    json_object_object_add(report, "build_seconds", jnum(nearbyint(scene.built * 100) / 100));
    json_object_object_add(report, "gpu_seconds", jnum(nearbyint(scene.seconds * 1000) / 1000));
    json_object_object_add(report, "voxels_per_second",
        json_object_new_int64(nearbyint(total / fmax(scene.seconds, 1e-9))));
    json_object_object_add(report, "on_sheet", jnum(nearbyint(1000.0 * on / total) / 1000));
    json_object_object_add(report, "outside", jnum(nearbyint(1000.0 * outside / total) / 1000));
    puts(json_object_to_json_string_ext(
        report, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE));
    json_object_put(report);
    json_object_put(spec);
    munmap(a.base, a.size);
    return 0;
}
#endif
