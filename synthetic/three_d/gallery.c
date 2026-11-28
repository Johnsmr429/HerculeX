#ifdef __APPLE__
#include <blosc.h>
#include <sqlite3.h>

enum { GALLERY_SIDE = 1024, GALLERY_CUT = 512, GALLERY_CUBE = 256 };

typedef struct {
    const char *url;
    i64 shape[3], chunks[3], bytes;
    int separator, blosc, pack;
    sqlite3 *index;
    sqlite3_stmt *find;
} SynGalleryVolume;

typedef struct {
    SynGalleryVolume *volume;
    u8 *pixels, *chunk;
    i64 box[6], lo[3], count[3];
} SynGalleryBox;

static SynGalleryVolume syn_gallery_volume(
    Arena *a, const char *look, const char *control, double voxel) {
    json_object *catalog = json_read(fmt(a, "%s/catalog/metadata.json", LIBRARY));
    json_object *sample = jget(jget(catalog, "samples"), look);
    json_object *volumes = jget(sample, "volumes");
    json_object *spec = json_object_from_file(control);
    const char *best = site_string(json_object_object_get(spec, "scan"), "scan");
    if (best && !json_object_object_get(volumes, best)) {
        best = NULL;
    }
    int fixed = best != NULL;
    double distance = INFINITY, energy = 0;
    json_object_object_foreach(volumes, key, row) {
        if (fixed) {
            break;
        }
        json_object *properties = jget(row, "properties"), *data = jget(row, "data");
        double size = jf64(properties, "pixel_size_um");
        double power = jf64(properties, "energy_keV");
        if (size <= 0 || fabs(size - voxel) > distance ||
            (fabs(size - voxel) == distance && power < energy)) {
            continue;
        }
        for (size_t j = 0; j < json_object_array_length(data); j++) {
            json_object *product = json_object_array_get_idx(data, j);
            if (!strcmp(jstr(product, "type"), "ome-zarr")) {
                distance = fabs(size - voxel);
                energy = power;
                best = key;
                break;
            }
        }
    }
    need(best != NULL, "no level-0 volume for %s", look);
    char *source;
    const char *url = zarr_level(a, look, best, 0, &source);
    json_object_put(spec);
    json_object_put(catalog);
    const char *leaf = strrchr(source, '/');
    const char *start = leaf;
    while (start > source && start[-1] != '/') {
        start--;
    }
    size_t width = leaf - start;
    if (width > 5 && !memcmp(leaf - 5, ".zarr", 5)) {
        width -= 5;
    }
    const char *folder = env("HERCULEX_CHUNKS", fmt(a, "%s/cache/chunks", OUTPUTS));
    const char *cache = fmt(a, "%s/%s/%.*s/0", folder, look, (int)width, start);
    json_object *meta = json_object_from_file(fmt(a, "%s/.zarray", cache));
    if (!meta) {
        meta = http_json(fmt(a, "%s/.zarray", url));
    }
    need(!strcmp(jstr(meta, "dtype"), "|u1") && !strcmp(jstr(meta, "order"), "C") &&
            json_object_array_length(jget(meta, "shape")) == 3 &&
            json_object_array_length(jget(meta, "chunks")) == 3,
        "%s is not a three-dimensional uint8 volume", url);
    json_object *compressor = jget(meta, "compressor"),
                *sep = json_object_object_get(meta, "dimension_separator");
    const char *codec = compressor ? jstr(compressor, "id") : "";
    need(!compressor || !strcmp(codec, "blosc"), "%s uses unsupported compression", url);
    SynGalleryVolume v = {.url = url,
        .separator = sep ? *json_object_get_string(sep) : '.',
        .blosc = compressor != NULL,
        .pack = -1,
        .bytes = 1};
    for (int d = 0; d < 3; d++) {
        v.shape[d] = json_object_get_int64(json_object_array_get_idx(jget(meta, "shape"), d));
        v.chunks[d] = json_object_get_int64(json_object_array_get_idx(jget(meta, "chunks"), d));
        need(v.shape[d] > 0 && v.chunks[d] > 0 && v.chunks[d] <= INT_MAX / v.bytes,
            "invalid chunk shape from %s", url);
        v.bytes *= v.chunks[d];
    }
    json_object_put(meta);
    const char *index = fmt(a, "%s/index.sqlite", cache);
    if (exists(index)) {
        need(sqlite3_open_v2(index, &v.index, SQLITE_OPEN_READONLY, NULL) == SQLITE_OK,
            "cannot open %s", index);
        const char *query = "SELECT offset FROM chunks WHERE z=?1 AND y=?2 AND x=?3";
        need(sqlite3_prepare_v2(v.index, query, -1, &v.find, NULL) == SQLITE_OK, "cannot read %s",
            index);
        v.pack = open(fmt(a, "%s/chunks.pack", cache), O_RDONLY);
    }
    return v;
}

static void syn_gallery_copy(SynGalleryBox *box, i64 index, const u8 *chunk) {
    const i64 *step = box->volume->chunks;
    i64 found[3], width[3], lo[3], hi[3];
    for (int d = 2; d >= 0; d--) {
        found[d] = (box->lo[d] + index % box->count[d]) * step[d];
        index /= box->count[d];
        width[d] = box->box[2 * d + 1] - box->box[2 * d];
        lo[d] = found[d] > box->box[2 * d] ? found[d] : box->box[2 * d];
        hi[d] = fmin(found[d] + step[d], fmin(box->volume->shape[d], box->box[2 * d + 1]));
    }
    for (i64 z = lo[0]; z < hi[0]; z++) {
        for (i64 y = lo[1]; y < hi[1]; y++) {
            i64 from = ((z - found[0]) * step[1] + y - found[1]) * step[2] + lo[2] - found[2];
            i64 at =
                ((z - box->box[0]) * width[1] + y - box->box[2]) * width[2] + lo[2] - box->box[4];
            memcpy(box->pixels + at, chunk + from, hi[2] - lo[2]);
        }
    }
}

static void syn_gallery_chunk(void *context, Ask *ask, const u8 *data, size_t size) {
    if (!data) {
        return;
    }
    SynGalleryBox *box = context;
    i64 bytes = box->volume->bytes;
    if (box->volume->blosc) {
        size_t decoded = 0;
        need(!blosc_cbuffer_validate(data, size, &decoded) && decoded == (size_t)bytes &&
                blosc_decompress_ctx(data, box->chunk, bytes, 1) == bytes,
            "cannot decode CT chunk %s", ask->path);
        data = box->chunk;
    } else {
        need(size == (size_t)bytes, "wrong CT chunk size for %s", ask->path);
    }
    syn_gallery_copy(box, ask->index, data);
}

static void syn_gallery_read(Arena *a, SynGalleryVolume *v, const i64 bounds[6], u8 *pixels) {
    size_t mark = a->used;
    SynGalleryBox b = {.volume = v, .pixels = pixels, .chunk = push(a, v->bytes)};
    memcpy(b.box, bounds, sizeof(b.box));
    i64 count = 1, length = 1;
    for (int d = 0; d < 3; d++) {
        i64 low = fmax(bounds[2 * d], 0), high = fmin(bounds[2 * d + 1] - 1, v->shape[d] - 1);
        b.lo[d] = low / v->chunks[d];
        b.count[d] = high < low ? 0 : high / v->chunks[d] - b.lo[d] + 1;
        count *= b.count[d];
        length *= bounds[2 * d + 1] - bounds[2 * d];
    }
    memset(pixels, 0, length);
    Ask *pending = push(a, count * sizeof(*pending));
    i64 asked = 0;
    for (i64 i = 0; i < count; i++) {
        i64 z = b.lo[0] + i / b.count[2] / b.count[1];
        i64 y = b.lo[1] + i / b.count[2] % b.count[1];
        i64 x = b.lo[2] + i % b.count[2];
        if (v->find) {
            sqlite3_bind_int64(v->find, 1, z);
            sqlite3_bind_int64(v->find, 2, y);
            sqlite3_bind_int64(v->find, 3, x);
            int code = sqlite3_step(v->find);
            i64 offset = code == SQLITE_ROW ? sqlite3_column_int64(v->find, 0) : -1;
            sqlite3_reset(v->find);
            need(code == SQLITE_ROW || code == SQLITE_DONE, "cannot query the CT chunk index");
            if (offset >= 0 && v->pack >= 0 &&
                pread(v->pack, b.chunk, v->bytes, offset) == v->bytes) {
                syn_gallery_copy(&b, i, b.chunk);
                continue;
            }
            if (offset == -2) {
                continue;
            }
        }
        Ask *ask = pending + asked++;
        *ask = (Ask){.first = -1, .last = -1, .index = i};
        snprintf(ask->path, sizeof(ask->path), "%lld%c%lld%c%lld", (long long)z, v->separator,
            (long long)y, v->separator, (long long)x);
    }
    if (asked) {
        need(!fetch_ranges(v->url, pending, asked, syn_gallery_chunk, &b), "could not fetch CT");
    }
    a->used = mark;
}

static void syn_gallery_actual(
    Arena *a, SynGalleryVolume *volume, Array points, u64 seed, u8 *xy, u8 *zx) {
    Pcg64 rng = pcg64(seed);
    size_t total = (size_t)GALLERY_SIDE * GALLERY_SIDE;
    for (size_t i = 0; i < (size_t)points.shape[0]; i++) {
        const float *p = (const float *)points.data + 3 * draw(&rng, points.shape[0]);
        i64 x = nearbyint(p[0]), y = nearbyint(p[1]), z = nearbyint(p[2]);
        i64 across[] = {z, z + 1, y - GALLERY_SIDE / 2, y + GALLERY_SIDE / 2, x - GALLERY_SIDE / 2,
            x + GALLERY_SIDE / 2};
        syn_gallery_read(a, volume, across, xy);
        size_t missing = 0;
        for (size_t j = 0; j < total; j++) {
            missing += xy[j] == 0;
        }
        if (missing > .05 * total) {
            continue;
        }
        i64 upright[] = {z - GALLERY_CUT / 2, z + GALLERY_CUT / 2, y, y + 1, x - GALLERY_SIDE / 2,
            x + GALLERY_SIDE / 2};
        syn_gallery_read(a, volume, upright, zx);
        return;
    }
    die("no complete axial slice near the gallery truth points");
}

static u8 syn_gallery_grey(float value) {
    return (u8)fminf(fmaxf((value - 20) / 180 * 255, 0), 255);
}

static void syn_gallery_png(const char *path, const u8 *pixels, int width, int height, int rgba) {
    png_image image = {.version = PNG_IMAGE_VERSION,
        .width = width,
        .height = height,
        .format = rgba ? PNG_FORMAT_RGBA : PNG_FORMAT_GRAY};
    need(png_image_write_to_file(&image, path, 0, pixels, 0, NULL), "cannot write %s: %s", path,
        image.message);
}

static void syn_gallery_view(Arena *a, const char *folder, const char *name, const u8 *intensity,
    const u8 *label, int width, int height, int stride, const u8 palette[64][3]) {
    size_t save = a->used, count = (size_t)width * height;
    u8 *pixels = push(a, count * (label ? 4 : 1));
    for (size_t i = 0; i < count; i++) {
        pixels[i] = syn_gallery_grey(intensity[stride * i]);
    }
    syn_gallery_png(fmt(a, "%s/%s.png", folder, name), pixels, width, height, 0);
    if (label) {
        for (size_t i = 0; i < count; i++) {
            int bit = label[stride * i];
            u8 *out = pixels + 4 * i;
            memcpy(out, palette[bit & 63], 3);
            out[3] = bit >= 128 ? 200 : 0;
            if (bit & 64) {
                memset(out, 255, 3);
                out[3] = 230;
            }
        }
        const char *truth = !strcmp(name, "ct") ? "truth" : "cut_truth";
        syn_gallery_png(fmt(a, "%s/%s.png", folder, truth), pixels, width, height, 1);
    }
    a->used = save;
}

static json_object *syn_gallery_case(Arena *a, const char *folder, SynGalleryVolume *volume,
    Array points, SynInject *scene, const char *family, u64 seed, const char *look) {
    const int side = GALLERY_SIDE, cut = GALLERY_CUT, cube = GALLERY_CUBE, half = cube / 2;
    const int middle = side / 2;
    const i64 across_n = (i64)side * side, cut_n = (i64)cut * side,
              cube_n = (i64)cube * cube * cube;
    double origin[3], voxel = scene->voxel;
    syn_inject_origin(scene, side, .45, origin);
    const double low[] = {origin[0] - cut * voxel / 2, origin[1], origin[2]};
    const double centre[] = {origin[0] - half * voxel, origin[1] + (middle - half) * voxel,
        origin[2] + (middle - half) * voxel};
    u8 *xy = push(a, 2 * across_n), *zx = push(a, 2 * cut_n), *block = push(a, 2 * cube_n);
    syn_inject_render(scene, origin, 1, side, side, xy);
    syn_inject_render(scene, low, cut, 1, side, zx);
    syn_inject_render(scene, centre, cube, cube, cube, block);
    u8 palette[64][3] = {{0}};
    for (int i = 1; i < 64; i++) {
        double hue = fmod(i * .6180339887498949, 1);
        const int shift[] = {0, 4, 2};
        for (int channel = 0; channel < 3; channel++) {
            double x = fabs(fmod(hue * 6 + shift[channel], 6) - 3) - 1;
            palette[i][channel] = (u8)(40 + 200 * fmax(0, fmin(1, x)));
        }
    }
    int turns[64] = {0}, material = 0, overlap = 0;
    for (i64 i = 0; i < across_n; i++) {
        int label = xy[i * 2 + 1];
        turns[label & 63] = 1;
        material += label >= 128;
        overlap |= label & 64;
    }
    for (i64 i = 0; i < cut_n; i++) {
        overlap |= zx[i * 2 + 1] & 64;
    }
    int count = 0;
    for (int i = 1; i < 64; i++) {
        count += turns[i];
    }
    syn_gallery_view(a, folder, "ct", xy, xy + 1, side, side, 2, palette);
    syn_gallery_view(a, folder, "cut", zx, zx + 1, side, cut, 2, palette);
    syn_gallery_actual(a, volume, points, seed, xy, zx);
    syn_gallery_view(a, folder, "real", xy, NULL, side, side, 1, palette);
    syn_gallery_view(a, folder, "real_cut", zx, NULL, side, cut, 1, palette);
    size_t downsampled = cube_n / 8;
    u8 *small = push(a, downsampled), *labels = push(a, downsampled);
    for (int z = 0; z < half; z++) {
        for (int y = 0; y < half; y++) {
            for (int x = 0; x < half; x++) {
                i64 at = ((i64)z * half + y) * half + x;
                i64 from = 2 * (((i64)2 * z * cube + 2 * y) * cube + 2 * x);
                int sum = 0;
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            sum += block[from + 2 * ((dz * cube + dy) * cube + dx)];
                        }
                    }
                }
                small[at] = syn_gallery_grey(sum / 8.0f);
                labels[at] = block[from + 1];
            }
        }
    }
    write_file(fmt(a, "%s/cube.u8", folder), small, downsampled);
    write_file(fmt(a, "%s/cube_label.u8", folder), labels, downsampled);
    json_object *row = json_object_new_object();
    json_object_object_add(row, "family", json_object_new_string(family));
    json_object_object_add(row, "seed", json_object_new_uint64(seed));
    json_object_object_add(row, "look", json_object_new_string(look));
    json_object_object_add(row, "voxel_um", jnum(voxel));
    json_object_object_add(row, "side_mm", jnum(nearbyint(side * voxel / 10) / 100));
    json_object_object_add(row, "cube", json_object_new_int(half));
    json_object_object_add(row, "cube_mm", jnum(nearbyint(cube * voxel / 10) / 100));
    json_object_object_add(row, "turns_in_slice", json_object_new_int(count));
    json_object_object_add(row, "on_sheet", jnum(nearbyint(1000.0 * material / across_n) / 1000));
    json_object_object_add(row, "kollesis", json_object_new_boolean(overlap != 0));
    json_object_object_add(row, "build_seconds", jnum(nearbyint(scene->built * 10) / 10));
    json_object_object_add(row, "gpu_seconds", jnum(nearbyint(scene->seconds * 1000) / 1000));
    json_object_object_add(row, "voxels", json_object_new_int64(across_n + cut_n + cube_n));
    return row;
}
static int syn_gallery_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.gallery "
                        "[--look PHerc0139] [--cases FAMILY:SEED ...]";
    const char *look = "PHerc0139";
    const char *defaults[] = {"round/pressed:3", "oval/fused:5", "kidney/pressed:2",
        "crease/pressed:4", "u-fold/normal:6", "boxy/cracked:2"};
    const char **cases = defaults;
    int count = sizeof(defaults) / sizeof(*defaults);
    for (int j = 0; j < argc; j++) {
        if (!strcmp(argv[j], "--look") && j + 1 < argc) {
            look = argv[++j];
        } else if (!strcmp(argv[j], "--cases")) {
            int first = j + 1;
            while (j + 1 < argc && strncmp(argv[j + 1], "--", 2) != 0) {
                j++;
            }
            cases = (const char **)(argv + first);
            count = j + 1 - first;
        } else if (!strcmp(argv[j], "-h") || !strcmp(argv[j], "--help")) {
            puts(usage);
            return 0;
        } else {
            die("%s", usage);
        }
    }
    need(site_name(look), "%s", usage);
    Arena a = arena(8ull << 30);
    const char *sheets = fmt(&a, "%s/website/sheets", OUTPUTS);
    const char *page = fmt(&a, "%s/synthetic", sheets);
    const char *staging = fmt(&a, "%s/synthetic.building", sheets);
    int gallery_lock = lock_file(fmt(&a, "%s/.synthetic.lock", sheets), 1, 1);
    need(gallery_lock >= 0, "cannot lock the synthetic gallery");
    remove_tree(staging);
    mkdirs(staging);

    json_object *rows = json_object_new_array_ext(count);
    if (count) {
        SynInjectLibrary library = syn_inject_library(&a, look);
        const char *control = fmt(&a, "%s/bench/controls/%s-train", OUTPUTS, look);
        const char *path = fmt(&a, "%s/truth.npz", control);
        Array points = npz_get(&a, path, "points");
        need(points.ndim == 2 && points.shape[1] == 3 && points.shape[0] > 0 &&
                !strcmp(points.descr, "<f4"),
            "invalid truth points in %s", path);
        SynGalleryVolume volume =
            syn_gallery_volume(&a, look, fmt(&a, "%s/control.json", control), library.voxel);
        size_t mark = a.used;
        for (int i = 0; i < count; i++) {
            const char *colon = strrchr(cases[i], ':');
            need(colon && colon > cases[i] && isdigit((unsigned char)colon[1]), "%s", usage);
            const char *family = fmt(&a, "%.*s", (int)(colon - cases[i]), cases[i]);
            char *end;
            u64 seed = strtoull(colon + 1, &end, 10);
            need(!*end, "%s", usage);
            const char *subdir = fmt(&a, "case-%d", i);
            const char *folder = fmt(&a, "%s/%s", staging, subdir);
            mkdirs(folder);
            json_object *spec = syn_family_draw(&a, family, seed, 0, look);
            need(spec != NULL, "unknown synthetic case %s", family);
            SynInject scene = syn_inject_open(&a, &library, spec, look, seed, NULL);
            json_object_put(spec);
            json_object *row =
                syn_gallery_case(&a, folder, &volume, points, &scene, family, seed, look);
            hx_inj_close(scene.gpu);
            json_object_object_add(row, "folder", json_object_new_string(subdir));
            json_object_array_add(rows, row);
            puts(json_object_to_json_string_ext(row, JSON_C_TO_STRING_PLAIN));
            fflush(stdout);
            a.used = mark;
        }
        sqlite3_finalize(volume.find);
        sqlite3_close(volume.index);
        if (volume.pack >= 0) {
            close(volume.pack);
        }
    }
    glob_t files;
    int code = glob(fmt(&a, "%s/synthetic/library/%s/slab-*.npz", OUTPUTS, look), 0, NULL, &files);
    need(!code && files.gl_pathc > 0, "no slabs for %s: run synthetic.library build first", look);
    Array slab = npz_get(&a, files.gl_pathv[0], "ct");
    need(slab.ndim == 3 && !strcmp(slab.descr, "|u1") && slab.shape[0] > 0 && slab.shape[1] > 0 &&
            slab.shape[2] > 0 && slab.size == (size_t)elements(slab),
        "invalid CT slab %s", files.gl_pathv[0]);
    int slab_count = (int)files.gl_pathc;
    globfree(&files);
    int rows_slab = slab.shape[0], cols = slab.shape[1], span = slab.shape[2];
    size_t area = (size_t)rows_slab * cols;
    u8 *pixels = push(&a, area);
    int mid = span / 2;
    for (size_t i = 0; i < area; i++) {
        pixels[i] = syn_gallery_grey(((const u8 *)slab.data)[span * i + mid]);
    }
    syn_gallery_png(fmt(&a, "%s/slab.png", staging), pixels, cols, rows_slab, 0);
    time_t current = time(NULL);
    struct tm local;
    char made[80];
    need(localtime_r(&current, &local) && strftime(made, sizeof(made), "%d %B %Y", &local),
        "cannot format the gallery's date");
    json_object *manifest = json_object_new_object();
    json_object_object_add(manifest, "look", json_object_new_string(look));
    json_object_object_add(manifest, "slabs", json_object_new_int(slab_count));
    json_object_object_add(manifest, "made", json_object_new_string(made));
    json_object_object_add(manifest, "cases", rows);
    Bytes template = read_file(&a, "synthetic/three_d/gallery.html");
    const char *placeholder = "__MANIFEST__",
               *where = template.data ? strstr((const char *)template.data, placeholder) : NULL;
    need(where != NULL, "cannot find the synthetic gallery template");
    char *html = fmt(&a, "%.*s%s%s", (int)(where - (const char *)template.data), template.data,
        json_object_to_json_string_ext(
            manifest, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE),
        where + strlen(placeholder));
    write_file(fmt(&a, "%s/index.html", staging), html, strlen(html));
    remove_tree(page);
    need(!rename(staging, page), "cannot publish %s: %s", page, strerror(errno));
    json_object_put(manifest);
    char *home[] = {"home"};
    hx_site_pages_command(1, home);
    printf("{\"page\":\"%s/index.html\",\"cases\":%d}\n", page, count);
    close(gallery_lock);
    munmap(a.base, a.size);
    return 0;
}
#endif
