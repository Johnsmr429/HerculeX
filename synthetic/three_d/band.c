#ifdef __APPLE__
int hx_syn_gpu_open(const char *source, char *message, int length);
void *hx_syn_gpu_scene(const void *const *data, const i64 *lengths);
int hx_syn_gpu_render(
    void *handle, const void *params, i64 length, int nz, int ny, int nx, u8 *out, double *seconds);
void hx_syn_gpu_close(void *handle);

typedef struct {
    int nz, ny, nx;
    float z0, dz, y0, x0, step;
    int rows, per_turn, knots;
    float z_first, knot_um, recto;
    float ply_dip, ply_dip_um, surface_dip, surface_dip_um, fibre, fray, compress_max;
    float hole_threshold, hole_soft;
    int tile;
    float tile_um, gap, papyrus, psf_um, phase, noise;
    int masked, n1, n2;
    float crinkle_um, specks, speck_threshold, unsharp, unsharp_um, gain_sd;
    int n_cracks;
    float fog_um, debris, debris_um;
    int bz, by, bx;
    float oz, oy, ox, voxel, noise1_z, noise1_y, noise1_x, noise2_z, noise2_y, noise2_x;
    float grain, grain_cell_um, noise_air;
    int grain_n;
    float rings, ring_um, ring_cy, ring_cx;
    int ring_n;
} SynGpuParams;

typedef struct {
    void *scene;
    SynGpuParams params;
} SynGpu;

static SynGpu syn_gpu_open(Arena *a, const SynKernel *s) {
    size_t mark = a->used;
    Bytes code = read_file(a, "synthetic/three_d/render_apple/kernel.metal");
    need(code.data != NULL, "cannot read synthetic Metal kernel");
    char *source = fmt(a, "%.*s", (int)code.size, code.data), message[4096];
    need(!hx_syn_gpu_open(source, message, sizeof(message)), "%s", message);
    i64 maps = s->nz * s->ny * s->nx, endings = s->per_turn * s->knots;
    i64 table = s->rows * endings, n_cracks = s->n_cracks ? s->n_cracks : 1;
    float *map = push(a, 2 * maps * sizeof(*map)), *roll = push(a, 5 * table * sizeof(*roll));
    float *end = push(a, endings * sizeof(*end)),
          *cracks = push(a, 12 * n_cracks * sizeof(*cracks));
    u8 *flags = push(a, 2 * maps);
    memcpy(map, s->a, maps * sizeof(*map));
    memcpy(map + maps, s->b, maps * sizeof(*map));
    memcpy(flags, s->bad, maps);
    memcpy(flags + maps, s->mask, maps);
    for (i64 i = 0; i < table; i++) {
        roll[i] = s->radius[i];
        roll[table + i] = s->thick[i];
        roll[2 * table + i] = s->delam[i];
        roll[3 * table + i] = fmod(s->arc[i], s->tile * s->tile_um);
        roll[4 * table + i] = i % s->knots + 1 < s->knots ? s->arc[i + 1] - s->arc[i] : 0;
    }
    for (i64 i = 0; i < endings; i++) {
        end[i] = s->end[i];
    }
    for (i64 i = 0; i < 12 * n_cracks; i++) {
        cracks[i] = s->cracks[i];
    }
    i64 noise = s->n1 * s->n1 * s->n1 + s->n2 * s->n2 * s->n2 +
        s->grain_n * s->grain_n * s->grain_n + s->ring_n;
    const void *buffers[] = {map, flags, roll, end, s->recto_tile, s->noise1, cracks};
    i64 lengths[] = {2 * maps * sizeof(float), 2 * maps, 5 * table * sizeof(float),
        endings * sizeof(float), 9 * s->tile * s->tile * sizeof(float), noise * sizeof(float),
        12 * n_cracks * sizeof(float)};
    void *scene = hx_syn_gpu_scene(buffers, lengths);
    need(scene != NULL, "cannot allocate synthetic Metal scene");
    a->used = mark;
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
        .ply_dip = s->ply_dip,
        .ply_dip_um = s->ply_dip_um,
        .surface_dip = s->surface_dip,
        .surface_dip_um = s->surface_dip_um,
        .fibre = s->fibre,
        .fray = s->fray,
        .compress_max = s->compress_max,
        .hole_threshold = s->hole_threshold,
        .hole_soft = s->hole_soft,
        .tile = s->tile,
        .tile_um = s->tile_um,
        .gap = s->gap,
        .papyrus = s->papyrus,
        .psf_um = s->psf_um,
        .phase = s->phase,
        .noise = s->noise,
        .masked = s->masked,
        .n1 = s->n1,
        .n2 = s->n2,
        .crinkle_um = s->crinkle_um,
        .specks = s->specks,
        .speck_threshold = s->speck_threshold,
        .unsharp = s->unsharp,
        .unsharp_um = s->unsharp_um,
        .gain_sd = s->gain_sd,
        .n_cracks = s->n_cracks,
        .fog_um = s->fog_um,
        .debris = s->debris,
        .debris_um = s->debris_um,
        .grain = s->grain,
        .grain_cell_um = s->grain_cell_um,
        .noise_air = s->noise_air,
        .grain_n = s->grain_n,
        .rings = s->rings,
        .ring_um = s->ring_um,
        .ring_cy = s->ring_cy,
        .ring_cx = s->ring_cx,
        .ring_n = s->ring_n};
    return (SynGpu){.scene = scene, .params = p};
}

static void syn_gpu_slab(SynGpu g, const double *origin, double voxel, int ny, int nx, u8 *out) {
    SynGpuParams p = g.params;
    p.by = ny;
    p.bx = nx;
    p.voxel = voxel;
    p.oy = origin[1];
    p.ox = origin[2];
    p.noise1_y = syn_k_wrap(origin[1] / voxel, p.n1);
    p.noise1_x = syn_k_wrap(origin[2] / voxel, p.n1);
    p.noise2_y = syn_k_wrap(origin[1] / voxel, p.n2);
    p.noise2_x = syn_k_wrap(origin[2] / voxel, p.n2);
    i64 stride = (i64)ny * nx;
    i64 share = (1 << 23) / stride;
    int step = share > 1 ? (int)share : 1;
    for (int z = 0; z < 128; z += step) {
        double corner = origin[0] + z * voxel, seconds;
        p.bz = fmin(step, 128 - z);
        p.oz = corner;
        p.noise1_z = syn_k_wrap(corner / voxel, p.n1);
        p.noise2_z = syn_k_wrap(corner / voxel, p.n2);
        int code =
            hx_syn_gpu_render(g.scene, &p, sizeof(p), p.bz, ny, nx, out + stride * z, &seconds);
        need(!code, "synthetic Metal render failed (%d)", code);
    }
}
#endif

static int syn_footprint(const SynScroll *s, const double *lo, const double *hi, double voxel) {
    const SynKernel *p = &s->kernel;
    i64 y0 = fmax(floor((lo[1] - voxel - p->y0) / p->step) - 1, 0);
    i64 x0 = fmax(floor((lo[2] - voxel - p->x0) / p->step) - 1, 0);
    i64 z0 = fmax(floor((lo[0] - voxel - p->z0) / p->dz), 0);
    i64 y1 = fmin(ceil((hi[1] + voxel - p->y0) / p->step) + 2, p->ny);
    i64 x1 = fmin(ceil((hi[2] + voxel - p->x0) / p->step) + 2, p->nx);
    i64 z1 = fmin(ceil((hi[0] + voxel - p->z0) / p->dz) + 1, p->nz);
    for (i64 z = z0; z < z1; z++) {
        for (i64 y = y0; y < y1; y++) {
            for (i64 x = x0; x < x1; x++) {
                if (p->mask[(z * p->ny + y) * p->nx + x]) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static json_object *syn_render_store(Arena *a, const SynScroll *scroll, json_object *spec,
    const char *out, const char *digest, int level, int gpu, int workers) {
    size_t mark = a->used;
    double base_voxel = jf64(spec, "voxel_um"), scale = ldexp(1, level), voxel = base_voxel * scale;
    SynBox box = scroll->deform.box;
    json_object *band = jget(spec, "band_um");
    double bottom = json_object_get_double(json_object_array_get_idx(band, 0));
    double top = json_object_get_double(json_object_array_get_idx(band, 1));
    double origin[] = {bottom, floor(box.y0 / voxel) * voxel, floor(box.x0 / voxel) * voxel};
    double extent[] = {top - bottom, box.y1 - origin[1], box.x1 - origin[2]};
    i64 grid[3], shape[3];
    const int brick = 128;
    for (int k = 0; k < 3; k++) {
        grid[k] = ceil(ceil(extent[k] / voxel) / brick);
        shape[k] = grid[k] * brick;
    }
    const char *meta = fmt(a, "%s/meta.json", out);
    if (!exists(meta)) {
        bricks_create(a, out, shape, (i64[]){brick, brick, brick}, fmt(a, "synthetic:%s", digest));
    }
    Bricks store;
    bricks_open(a, &store, out);
    need(!memcmp(shape, store.shape, sizeof(shape)),
        "synthetic brick store shape differs from spec");
    const char *path = fmt(a, "%s/bricks.bin", out);
    int fd = open(path, O_WRONLY | O_APPEND);
    need(fd >= 0, "cannot open %s", path);
    i64 offset = lseek(fd, 0, SEEK_END), wrote = 0, absent = 0;
    need(offset >= 0, "cannot seek in %s", path);
    size_t span = (size_t)brick * brick * shape[2], count = (size_t)brick * brick * brick;
    u8 *slab = push(a, span), *tile = push(a, count);
#ifdef __APPLE__
    SynGpu metal = {0};
    if (gpu) {
        metal = syn_gpu_open(a, &scroll->kernel);
    }
#endif
    double start = now(), shift = (scale - 1) * 0.5 * base_voxel;
    for (i64 z = 0; z < grid[0]; z++) {
        for (i64 y = 0; y < grid[1]; y++) {
            double lo[] = {origin[0] + shift + z * brick * voxel,
                origin[1] + shift + y * brick * voxel, origin[2] + shift};
            double hi[] = {lo[0] + brick * voxel, lo[1] + brick * voxel, lo[2] + shape[2] * voxel};
            int present = syn_footprint(scroll, lo, hi, voxel);
            if (present) {
#ifdef __APPLE__
                if (gpu) {
                    syn_gpu_slab(metal, lo, voxel, brick, shape[2], slab);
                }
#endif
                if (!gpu) {
                    hx_syn_render(
                        &scroll->kernel, brick, brick, shape[2], lo, voxel, slab, workers);
                }
            }
            for (i64 x = 0; x < grid[2]; x++) {
                i64 i = (z * grid[1] + y) * grid[2] + x;
                int stored = 0;
                if (present) {
                    for (int row = 0; row < brick * brick; row++) {
                        memcpy(
                            tile + (size_t)brick * row, slab + shape[2] * row + x * brick, brick);
                    }
                    stored = memcmp(tile, tile + 1, count - 1) != 0 || tile[0] != 0;
                }
                if (stored) {
                    write_all(fd, tile, count, path);
                    store.offsets[i] = offset;
                    offset += count;
                    wrote++;
                } else {
                    store.offsets[i] = ABSENT;
                    absent++;
                }
            }
        }
        bricks_save(&store);
        printf("render: brick plane %lld/%lld, %lld bricks, %lld absent, %.0f s\n",
            (long long)(z + 1), (long long)grid[0], (long long)wrote, (long long)absent,
            now() - start);
        fflush(stdout);
    }
    close(fd);
#ifdef __APPLE__
    hx_syn_gpu_close(metal.scene);
#endif
    json_object *result = json_object_new_object();
    json_object *xyz = json_object_new_array(), *sizes = json_object_new_array();
    for (int k = 0; k < 3; k++) {
        json_object_array_add(xyz, jnum(origin[k]));
        json_object_array_add(sizes, json_object_new_int64(shape[k]));
    }
    json_object_object_add(result, "origin_um", xyz);
    json_object_object_add(result, "voxel_um", jnum(base_voxel));
    json_object_object_add(result, "level", json_object_new_int(level));
    json_object_object_add(result, "shape", sizes);
    json_object_object_add(result, "bricks", json_object_new_int64(wrote));
    json_object_object_add(result, "absent", json_object_new_int64(absent));
    json_object_object_add(result, "render_seconds", jnum(nearbyint((now() - start) * 10) / 10));
    if (store.data) {
        munmap((void *)store.data, store.mapped);
    }
    pthread_mutex_destroy(&store.lock);
    a->used = mark;
    return result;
}

#include "labels.c"

int hx_synthetic_detector(
    const char *store, const char *out, i64 planes, json_object *record, int gpu);
json_object *hx_synthetic_level_fit(const char *unit, const char *out, const double *line,
    size_t knots, int curl, float **positions);

static int syn_band_command(int argc, char **argv) {
#ifdef __APPLE__
    int gpu = 1;
#else
    int gpu = 0;
#endif
    const char *usage = "Usage: ./run.sh synthetic.three_d.band render SPEC OUT "
                        "[--level N] [--device cpu|gpu]\n"
                        "       ./run.sh synthetic.three_d.band build SPEC [--name NAME] "
                        "[--device cpu|gpu] [--keep-ct]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    int build = argc >= 2 && !strcmp(argv[0], "build"), level = 1, keep = 0;
    need(build || (argc >= 3 && !strcmp(argv[0], "render")), "%s", usage);
    const char *name = NULL, *root = NULL;
    for (int i = build ? 2 : 3; i < argc; i++) {
        if (!build && !strcmp(argv[i], "--level") && i + 1 < argc) {
            char *end;
            long parse = strtol(argv[++i], &end, 10);
            need(!*end && parse >= 0 && parse < 16, "%s", usage);
            level = parse;
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            const char *device = argv[++i];
            need(!strcmp(device, "cpu") || !strcmp(device, "gpu"), "%s", usage);
            gpu = !strcmp(device, "gpu");
        } else if (build && !strcmp(argv[i], "--name") && i + 1 < argc) {
            name = argv[++i];
        } else if (build && !strcmp(argv[i], "--root") && i + 1 < argc) {
            root = argv[++i];
        } else if (build && !strcmp(argv[i], "--keep-ct")) {
            keep = 1;
        } else {
            die("%s", usage);
        }
    }
#ifndef __APPLE__
    need(!gpu, "the synthetic GPU renderer requires Metal");
#endif
    Arena a = arena(64 * GB);
    json_object *source = json_read(argv[1]);
    need(source != NULL, "cannot read %s", argv[1]);
    json_object *spec = syn_complete(&a, source);
    char digest[13];
    syn_spec_digest(&a, source, digest);
    json_object_put(source);
    int workers = (int)strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), NULL, 10);
    if (workers <= 0) {
        workers = fmax(1, cores() - 2);
    }
    double base_voxel = jf64(spec, "voxel_um"), voxel = ldexp(base_voxel, level);
    need(voxel > 0, "invalid synthetic voxel size");
    const char *out = build ? NULL : argv[2], *ct = out, *family = jstr(spec, "family");
    if (build) {
        if (!name) {
            name = fmt(&a, "%s-%s", family, digest);
        }
        out = root ? fmt(&a, "%s/%s", root, name) : fmt(&a, "%s/synthetic/bands/%s", OUTPUTS, name);
        ct = fmt(&a, "%s/ct-L1", out);
        json_write(fmt(&a, "%s/spec.json", out), spec);
    }
    double began = now();
    SynScroll scroll;
    syn_scroll(&a, &scroll, spec, NULL, voxel, workers);
    double times[6] = {began, now()};
    json_object *result = syn_render_store(&a, &scroll, spec, ct, digest, level, gpu, workers);
    if (build) {
        const char *record = fmt(&a, "%s/band.json", out), *control = fmt(&a, "%s/control", out);
        json_object_object_add(result, "family", json_object_new_string(family));
        json_object_object_add(result, "digest", json_object_new_string(digest));
        json_object_object_add(result, "name", json_object_new_string(name));
        json_object_object_add(
            result, "look", json_object_get(jget(jget(spec, "lineage"), "look")));
        json_write(record, result);
        times[2] = now();
        double origin[3];
        i64 depth = json_object_get_int64(json_object_array_get_idx(jget(result, "shape"), 0));
        for (int i = 0; i < 3; i++) {
            origin[i] =
                json_object_get_double(json_object_array_get_idx(jget(result, "origin_um"), i));
        }
        json_object *detection = json_object_new_object(), *synthetic = json_object_new_object();
        json_object_object_add(
            detection, "sample", json_object_new_string(fmt(&a, "SYN-%s", family)));
        json_object_object_add(detection, "volume", json_object_new_string(name));
        syn_add(detection, "scan_voxel_um", base_voxel);
        syn_add(detection, "voxel_um", voxel);
        json_object_object_add(detection, "level", json_object_new_int(level));
        json_object *rows = json_object_new_array_ext(2);
        json_object_array_add(rows, json_object_new_int(0));
        json_object_array_add(rows, json_object_new_int64(depth / 128));
        json_object_object_add(detection, "rows", rows);
        json_object_object_add(synthetic, "digest", json_object_new_string(digest));
        json_object_object_add(synthetic, "family", json_object_new_string(family));
        json_object_object_add(synthetic, "origin_um", json_object_get(jget(result, "origin_um")));
        json_object_object_add(detection, "synthetic", synthetic);
        need(!hx_synthetic_detector(ct, control, depth / 8, detection, gpu),
            "could not detect sheets in %s", ct);
        times[3] = now();
        if (!keep) {
            remove_tree(ct);
        }
        int knots = scroll.deform.nz;
        double *axis = push(&a, (size_t)knots * 3 * sizeof(*axis));
        for (int i = 0; i < knots; i++) {
            SynPoint center = syn_center_at(&scroll.deform, i);
            axis[3 * i] = (scroll.deform.z0 + i * scroll.deform.dz - origin[0]) / base_voxel;
            axis[3 * i + 1] = (center.x - origin[2]) / base_voxel;
            axis[3 * i + 2] = (center.y - origin[1]) / base_voxel;
        }
        int curl = scroll.deform.mirror < 0 ? -1 : 1;
        float *points;
        json_object *fit = hx_synthetic_level_fit(control, out, axis, knots, curl, &points);
        need(fit != NULL, "could not fit synthetic band %s", out);
        times[4] = now();
        json_object *truth =
            syn_band_labels(&a, &scroll, result, out, points, ji64(fit, "nodes"), origin, workers);
        free(points);
        times[5] = now();
        json_object_object_add(result, "nodes", json_object_get(jget(detection, "nodes")));
        json_object_object_add(result, "fit", fit);
        json_object_object_add(result, "truth", truth);
        const char *steps[] = {"tables", "render", "detect", "fit", "truth"};
        json_object *seconds = json_object_new_object();
        for (int i = 0; i < 5; i++) {
            syn_add(seconds, steps[i], nearbyint((times[i + 1] - times[i]) * 10) / 10);
        }
        json_object_object_add(result, "seconds", seconds);
        json_write(record, result);
        json_object_put(detection);
    }
    puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PRETTY));
    json_object_put(result);
    json_object_put(spec);
    return 0;
}
