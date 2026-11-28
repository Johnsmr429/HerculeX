#ifndef HX_UNITY
const char hx_usage[] = "Read survey pieces forward and reversed with the native reader.";
#endif

typedef struct {
    const char *sample, *volume, *band, *pieces, *survey;
} SurveyBand;

static SurveyBand survey_band(Arena *a, const char *item) {
    const char *first = strchr(item, '/'), *last = first ? strchr(first + 1, '/') : NULL;
    need(first && last, "expected SAMPLE/SCAN/BAND, got %s", item);
    const char *sample = fmt(a, "%.*s", (int)(first - item), item);
    const char *volume = fmt(a, "%.*s", (int)(last - first - 1), first + 1);
    const char *band = last + 1;
    need(site_name(sample) && site_name(volume) && site_name(band), "invalid survey band %s", item);
    const char *pieces =
        fmt(a, "%s/unwrapping/level/%s/%s/pieces/%s", OUTPUTS, sample, volume, band);
    if (!exists(pieces)) {
        pieces =
            fmt(a, "%s/calibration/surface_ink/%s-%s-%s/pieces", LIBRARY, sample, volume, band);
    }
    return (SurveyBand){
        sample, volume, band, pieces, fmt(a, "%s/calibration/scroll_survey", LIBRARY)};
}

static const char *survey_stage(Arena *a, SurveyBand band, const char *name) {
    return fmt(a, "%s/%s%s%s", band.survey, *name ? name : "", *name ? "/" : "", band.sample);
}

static double survey_voxel(Arena *a, SurveyBand band, const char *piece) {
    const char *path = fmt(a, "%s/%s/meta.json", band.pieces, piece);
    json_object *meta = json_read(path);
    need(meta != NULL, "no survey piece metadata at %s", path);
    double voxel = jf64(meta, "voxel_um") / 1000;
    json_object_put(meta);
    need(voxel > 0, "invalid voxel size in %s", path);
    return voxel;
}

static void survey_write(const char *path, json_object *o) {
    const char *text = board_json(o);
    write_file(path, text, strlen(text));
}

static void survey_nearest(Arena *a, const u8 *image, int h, int w, i32 **fy, i32 **fx) {
    i64 n = (i64)h * w;
    *fy = push(a, n * sizeof(i32));
    *fx = push(a, n * sizeof(i32));
    size_t saved = a->used;
    u8 *outside = push(a, n);
    int found = 0;
    for (i64 i = 0; i < n; i++) {
        outside[i] = image[i] == 0;
        found |= outside[i];
    }
    if (found) {
        nearest_good(outside, h, w, *fy, *fx);
    } else {
        for (i64 i = 0; i < n; i++) {
            (*fy)[i] = -1;
            (*fx)[i] = 0;
        }
    }
    a->used = saved;
}

static double survey_distance(int y, int x, int py, int px) {
    double dy = (double)y - py, dx = (double)x - px;
    return sqrt(dy * dy + dx * dx);
}

static int survey_windows(
    Arena *a, const u8 *sheet, int h, int w, int spots[2][2], int *yh, int *xw) {
    *yh = h < 1024 ? h : 1024;
    *xw = w < 1024 ? w : 1024;
    int dy = *yh / 2, dx = *xw / 2;
    dy = dy ? dy : 1;
    dx = dx ? dx : 1;
    int rows = (h - *yh) / dy + 1, cols = (w - *xw) / dx + 1, count = rows * cols;
    size_t saved = a->used;
    i64(*mass)[2] = push(a, count * sizeof(*mass));
    for (int i = 0; i < count; i++) {
        int row = i / cols * dy, col = i % cols * dx;
        i64 total = 0;
        for (int y = row; y < row + *yh; y++) {
            for (int x = col; x < col + *xw; x++) {
                total += sheet[(i64)y * w + x] != 0;
            }
        }
        mass[i][0] = total;
        mass[i][1] = i;
    }
    qsort(mass, count, sizeof(*mass), by_mass);
    int n = 0;
    for (int i = 0; i < count && n < 2; i++) {
        int y = mass[i][1] / cols * dy, x = mass[i][1] % cols * dx;
        if (!n || abs(y - spots[0][0]) >= *yh || abs(x - spots[0][1]) >= *xw) {
            spots[n][0] = y;
            spots[n++][1] = x;
        }
    }
    a->used = saved;
    return n;
}

static const char *survey_piece_name(Arena *a, const char *path, int prefix) {
    const char *name = disk_base(path), *ext = strrchr(name, '.');
    need(ext && ext - name >= prefix, "invalid survey piece path %s", path);
    return fmt(a, "%.*s", (int)(ext - name) - prefix, name + prefix);
}

typedef struct {
    i64 key;
    double value;
    int used;
} SurveyEntry;

typedef struct {
    SurveyEntry *entries;
    size_t cap, count;
} SurveyMap;

static size_t survey_bucket(i64 key, size_t n) {
    u64 a = key;
    a ^= a >> 30;
    a *= UINT64_C(0xbf58476d1ce4e5b9);
    a ^= a >> 27;
    a *= UINT64_C(0x94d049bb133111eb);
    return (a ^ a >> 31) & (n - 1);
}

static SurveyEntry *survey_entry(SurveyMap *map, i64 key) {
    if (!map->cap) {
        return NULL;
    }
    size_t at = survey_bucket(key, map->cap);
    while (map->entries[at].used && map->entries[at].key != key) {
        at = (at + 1) & (map->cap - 1);
    }
    return map->entries + at;
}

static void survey_insert(SurveyMap *map, i64 key, double value) {
    if (map->count * 2 >= map->cap) {
        size_t cap = map->cap ? map->cap * 2 : 256;
        SurveyMap bigger = {
            .cap = cap, .count = map->count, .entries = calloc(cap, sizeof(*map->entries))};
        need(bigger.entries != NULL, "cannot allocate survey geometry keys");
        for (size_t i = 0; i < map->cap; i++) {
            SurveyEntry row = map->entries[i];
            if (row.used) {
                *survey_entry(&bigger, row.key) = row;
            }
        }
        free(map->entries);
        *map = bigger;
    }
    SurveyEntry *row = survey_entry(map, key);
    if (row->used) {
        row->value = fmax(row->value, value);
    } else {
        *row = (SurveyEntry){key, value, 1};
        map->count++;
    }
}

typedef struct {
    const char *name, *folder, *out;
    double area, side;
    int points, order;
    SurveyMap bricks;
} SurveyPiece;

static void survey_axis(Arena *a, SurveyBand band, Stage *stage) {
    int hx_umbilicus_keep(const char *, const char *);
    snprintf(stage->sample, sizeof(stage->sample), "%s", band.sample);
    snprintf(stage->scan, sizeof(stage->scan), "%s", band.volume);
    const char *path =
        fmt(a, "%s/unwrapping/level/%s/%s/sheet/axis.json", OUTPUTS, band.sample, band.volume);
    if (!exists(path)) {
        need(hx_umbilicus_keep(band.sample, band.volume) == 0, "no umbilicus for %s %s",
            band.sample, band.volume);
    }
    json_object *axis = json_read(path);
    stage->rows = board_count(axis);
    need(stage->rows, "no umbilicus in %s", path);
    stage->line = push(a, stage->rows * 3 * sizeof(*stage->line));
    for (int i = 0; i < stage->rows; i++) {
        json_object *row = json_object_array_get_idx(axis, i);
        need(board_count(row) == 3, "invalid umbilicus point in %s", path);
        for (int j = 0; j < 3; j++) {
            stage->line[3 * i + j] = json_object_get_double(json_object_array_get_idx(row, j));
        }
    }
    json_object_put(axis);
}

static int survey_area_order(const void *aa, const void *bb) {
    const SurveyPiece *a = aa, *b = bb;
    if (a->area != b->area) {
        return a->area > b->area ? -1 : 1;
    }
    return a->order - b->order;
}

static int survey_depth_order(const void *aa, const void *bb) {
    const double a = *(const double *)aa, b = *(const double *)bb;
    return (a > b) - (a < b);
}

static SurveyEntry survey_polar(const Stage *stage, Surface surface, i64 n, i64 at, double voxel) {
    double z = surface.xyz[2 * n + at];
    double x = surface.xyz[at] - interpolated(z, stage->line, stage->rows, 1);
    double y = surface.xyz[n + at] - interpolated(z, stage->line, stage->rows, 2);
    double angle = atan2(y, x);
    int low = 0, high = 73;
    while (low < high) {
        int mid = low + (high - low) / 2;
        double boundary = mid == 72 ? M_PI : mid * (M_PI / 36) - M_PI;
        if (angle < boundary) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return (SurveyEntry){(i64)floor((f32)z / 300.0f) * 100 + low, hypot(x, y) * voxel, 1};
}

static void survey_geometry(Arena *a, Stage *stage, SurveyPiece *pieces, int n) {
    SurveyMap outer = {0};
    double voxel = 0;
    for (int i = 0; i < n; i++) {
        size_t saved = a->used;
        Surface mesh = load_surface(a, pieces[i].folder);
        i64 size = (i64)mesh.h * mesh.w, count = 0;
        for (i64 j = 0; j < size; j++) {
            count += mesh.xyz[j] > 0;
        }
        if (count >= 30) {
            if (!voxel) {
                voxel = mesh.voxel / 1000;
            }
            pieces[i].points = (count + 2) / 3;
            pieces[i].side = fmin(mesh.h, mesh.w) * mesh.stride * mesh.voxel / 1000;
            int good = 0;
            for (i64 j = 0; j < size; j++) {
                if (mesh.xyz[j] > 0 && good++ % 3 == 0) {
                    SurveyEntry e = survey_polar(stage, mesh, size, j, voxel);
                    survey_insert(&outer, e.key, e.value);
                }
            }
        }
        a->used = saved;
    }
    for (int i = 0; i < n; i++) {
        SurveyPiece *piece = &pieces[i];
        if (!piece->points || piece->area < 100 || piece->area > 350) {
            piece->points = 0;
            continue;
        }
        size_t saved = a->used;
        Surface mesh = load_surface(a, piece->folder);
        i64 size = (i64)mesh.h * mesh.w;
        double *depths = push(a, piece->points * sizeof(*depths));
        int good = 0, found = 0;
        for (i64 j = 0; j < size; j++) {
            if (mesh.xyz[j] > 0 && good++ % 3 == 0) {
                SurveyEntry e = survey_polar(stage, mesh, size, j, voxel);
                SurveyEntry *limit = survey_entry(&outer, e.key);
                if (limit && limit->used) {
                    depths[found++] = limit->value - e.value;
                }
            }
        }
        qsort(depths, found, sizeof(*depths), survey_depth_order);
        if (!found || (depths[(found - 1) / 2] + depths[found / 2]) / 2 < 2) {
            piece->points = 0;
        }
        a->used = saved;
    }
    free(outer.entries);
}

static i64 survey_brick_key(int z, int y, int x) {
    const u64 mask = (UINT64_C(1) << 21) - 1;
    return ((z & mask) << 42) | ((y & mask) << 21) | (x & mask);
}

static void survey_bricks(Arena *a, SurveyPiece *piece) {
    size_t saved = a->used;
    Surface mesh = extended(a, load_surface(a, piece->folder));
    i64 size = (i64)mesh.h * mesh.w;
    for (i64 i = 0; i < size; i++) {
        if (mesh.xyz[i] <= 0) {
            continue;
        }
        int low[3], high[3];
        for (int axis = 0; axis < 3; axis++) {
            double v = mesh.xyz[axis * size + i];
            low[axis] = floor((v - 17) / 128);
            high[axis] = floor((v + 17) / 128);
        }
        for (int z = low[2]; z <= high[2]; z++) {
            for (int y = low[1]; y <= high[1]; y++) {
                for (int x = low[0]; x <= high[0]; x++) {
                    survey_insert(&piece->bricks, survey_brick_key(z, y, x), 0);
                }
            }
        }
    }
    a->used = saved;
}

static void survey_shuffle(SurveyPiece **pool, int n, int seed) {
    Pcg64 rng = pcg64(seed);
    for (int i = n - 1; i > 0; i--) {
        u32 mask = i, chosen;
        for (int shift = 1; shift < 32; shift *= 2) {
            mask |= mask >> shift;
        }
        do {
            chosen = pcg64_next32(&rng) & mask;
        } while (chosen > (u32)i);
        SurveyPiece *swap = pool[i];
        pool[i] = pool[chosen];
        pool[chosen] = swap;
    }
}

static int survey_look(Arena *a, SurveyPiece **pool, int count) {
    count = count < 60 ? count : 60;
    for (int i = 0; i < count; i++) {
        survey_bricks(a, pool[i]);
    }
    SurveyMap have = {0};
    int chosen = 0;
    while (chosen < count && have.count < 2500) {
        int best = chosen;
        double score = -1;
        for (int i = chosen; i < count; i++) {
            const SurveyMap map = pool[i]->bricks;
            i64 extra = 0;
            for (size_t k = 0; k < map.cap; k++) {
                SurveyEntry e = map.entries[k];
                SurveyEntry *done = e.used ? survey_entry(&have, e.key) : NULL;
                extra += e.used && (!done || !done->used);
            }
            double value = pool[i]->area / fmax(extra, 1);
            if (value > score) {
                best = i;
                score = value;
            }
        }
        SurveyPiece *piece = pool[best];
        memmove(pool + chosen + 1, pool + chosen, (best - chosen) * sizeof(*pool));
        pool[chosen++] = piece;
        for (size_t k = 0; k < piece->bricks.cap; k++) {
            SurveyEntry e = piece->bricks.entries[k];
            if (e.used) {
                survey_insert(&have, e.key, 0);
            }
        }
    }
    for (int i = 0; i < count; i++) {
        free(pool[i]->bricks.entries);
    }
    free(have.entries);
    return chosen;
}

enum { SURVEY_ALL, SURVEY_LOOK, SURVEY_REPLICATE, SURVEY_REREAD, SURVEY_ROW, SURVEY_SECOND };

static int survey_kind(const char *name) {
    const char *names[] = {"all", "look", "replicate", "reread", "row-sample", "second"};
    for (int i = 0; i < 6; i++) {
        if (!strcmp(name, names[i])) {
            return i;
        }
    }
    return SURVEY_ALL;
}

static Job *survey_old_pieces(Arena *a, SurveyBand band, int mode, int *count) {
    const char *where = mode == SURVEY_REREAD ? "" : "second-model";
    const char *out = survey_stage(a, band, where);
    if (mode == SURVEY_SECOND) {
        mkdirs(out);
    }
    glob_t paths[2] = {{0}};
    int n = 0;
    for (int s = 0; s < 2; s++) {
        if (mode == SURVEY_REREAD || s == 1) {
            const char *folder = survey_stage(a, band, s ? "replicate" : "");
            glob(fmt(a, "%s/piece-*.json", folder), 0, NULL, &paths[s]);
            n += paths[s].gl_pathc;
        }
    }
    Job *jobs = pushz(a, n * sizeof(*jobs));
    *count = 0;
    for (int s = 0; s < 2; s++) {
        for (size_t p = 0; p < paths[s].gl_pathc; p++) {
            const char *path = paths[s].gl_pathv[p], *name = survey_piece_name(a, path, 0);
            const char *target = out;
            if (mode == SURVEY_REREAD) {
                json_object *row = json_read(path);
                int current = !strcmp(board_text(row, "ruler"), "both directions read whole");
                json_object_put(row);
                if (current) {
                    continue;
                }
                target = survey_stage(a, band, s ? "replicate" : "");
            } else if (exists(fmt(a, "%s/%s.json", target, name))) {
                continue;
            }
            Job *job = jobs + (*count)++;
            snprintf(job->folder, sizeof(job->folder), "%s/%s", band.pieces, name);
            snprintf(job->out, sizeof(job->out), "%s/%s", target, name);
        }
        globfree(&paths[s]);
    }
    return jobs;
}

static Job *survey_plan(Arena *a, SurveyBand band, Stage *stage, int mode, int *count) {
    survey_axis(a, band, stage);
    if (mode == SURVEY_REREAD || mode == SURVEY_SECOND) {
        return survey_old_pieces(a, band, mode, count);
    }
    const char *first = survey_stage(a, band, ""), *second = survey_stage(a, band, "replicate");
    const char *out = mode == SURVEY_REPLICATE ? second
        : mode == SURVEY_ROW                   ? survey_stage(a, band, "row-sample")
                                               : first;
    mkdirs(out);
    glob_t paths = {0};
    glob(fmt(a, "%s/piece-*", band.pieces), 0, NULL, &paths);
    int n = paths.gl_pathc;
    SurveyPiece *pieces = pushz(a, n * sizeof(*pieces));
    SurveyPiece **pool = push(a, n * sizeof(*pool));
    for (int i = 0; i < n; i++) {
        const char *folder = fmt(a, "%s", paths.gl_pathv[i]), *name = disk_base(folder);
        json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
        need(meta != NULL, "no metadata for %s", folder);
        pieces[i] = (SurveyPiece){.name = name,
            .folder = folder,
            .out = fmt(a, "%s/%s", out, name),
            .area = board_number(meta, "area_mm2"),
            .order = i};
        json_object_put(meta);
    }
    globfree(&paths);
    if (mode == SURVEY_ALL) {
        qsort(pieces, n, sizeof(*pieces), survey_area_order);
    } else if (n) {
        survey_geometry(a, stage, pieces, n);
    }
    int pool_n = 0;
    for (int i = 0; i < n; i++) {
        SurveyPiece *p = &pieces[i];
        if (mode != SURVEY_ALL && !p->points) {
            continue;
        }
        if ((mode == SURVEY_REPLICATE || mode == SURVEY_ROW) &&
            exists(fmt(a, "%s/%s.json", first, p->name))) {
            continue;
        }
        if (mode == SURVEY_ROW && (p->side < 17 || exists(fmt(a, "%s/%s.json", second, p->name)))) {
            continue;
        }
        pool[pool_n++] = p;
    }
    if (mode == SURVEY_LOOK) {
        survey_shuffle(pool, pool_n, 3);
        pool_n = survey_look(a, pool, pool_n);
    } else if (mode == SURVEY_REPLICATE || mode == SURVEY_ROW) {
        survey_shuffle(pool, pool_n, mode == SURVEY_REPLICATE ? 7 : 8);
        pool_n = pool_n < 16 ? pool_n : 16;
    }
    Job *jobs = pushz(a, pool_n * sizeof(*jobs));
    *count = 0;
    for (int i = 0; i < pool_n; i++) {
        SurveyPiece *p = pool[i];
        if (!exists(fmt(a, "%s.json", p->out))) {
            Job *job = jobs + (*count)++;
            snprintf(job->folder, sizeof(job->folder), "%s", p->folder);
            snprintf(job->out, sizeof(job->out), "%s", p->out);
            job->area = p->area;
        }
    }
    return jobs;
}

static const char *survey_store(Arena *a, SurveyBand band, int mode) {
    const char *name = mode == SURVEY_REREAD ? "survey-reread"
        : mode == SURVEY_ROW                 ? "survey-row-sample"
                                             : "survey";
    const char *custom = env("HERCULEX_BRICKS", "");
    return *custom ? fmt(a, "%s/%s/ct_bricks/%s", custom, band.sample, name)
                   : fmt(a, "%s/cache/bricks/%s/%s", LIBRARY, band.sample, name);
}

static void survey_uniform(const f32 *in, f32 *out, int h, int w, int axis) {
    int lines = axis ? h : w, n = axis ? w : h, stride = axis ? 1 : w;
    for (int i = 0; i < lines; i++) {
        i64 base = axis ? (i64)i * w : i;
        double total = 0;
        for (int k = -16; k < 16; k++) {
            total += in[base + reflected(k, n) * stride];
        }
        for (int j = 0; j < n; j++) {
            out[base + j * (i64)stride] = total / 32;
            double next = in[base + reflected(j + 16, n) * stride];
            double old = in[base + reflected(j - 16, n) * stride];
            total += next - old;
        }
    }
}

static u8 *survey_open(Arena *a, Stack s, int y0, int x0, int sy, int sx, int reverse) {
    u8 *open = pushz(a, (i64)sy * sx);
    int start = s.layers / 2 - FACE_SEARCH, planes = 2 * FACE_SEARCH + 1;
    for (int y = 0; y < sy; y += 512) {
        for (int x = 0; x < sx; x += 512) {
            int h = fmin(512, sy - y), w = fmin(512, sx - x);
            if (h < 48 || w < 48) {
                continue;
            }
            size_t saved = a->used;
            i64 n = (i64)h * w;
            u8 *block = push(a, s.layers * n);
            f32 *smooth = push(a, planes * n * sizeof(*smooth));
            f32 *work = push(a, planes * n * sizeof(*work));
            int any = 0;
            for (int l = 0; l < s.layers; l++) {
                int at = reverse ? s.layers - l - 1 : l;
                for (int row = 0; row < h; row++) {
                    const u8 *source =
                        s.cube + ((i64)at * s.height + y0 + y + row) * s.width + x0 + x;
                    u8 *target = block + l * n + (i64)row * w;
                    memcpy(target, source, w);
                    for (int c = 0; c < w; c++) {
                        any |= source[c];
                    }
                }
            }
            if (any) {
                for (i64 i = 0; i < planes * n; i++) {
                    smooth[i] = block[start * n + i];
                }
                i64 dims[] = {planes, h, w};
                const double sigmas[] = {0.7, 2.5, 2.5};
                gaussian(smooth, work, dims, 3, sigmas);
                double *depth = push(a, n * sizeof(*depth));
                face_depth(smooth, n, planes, n, start, depth);
                f32 *low = work, *part = smooth;
                layer_at(block, n, s.layers, h, w, w, depth, 1, low);
                for (int l = 2; l <= 3; l++) {
                    layer_at(block, n, s.layers, h, w, w, depth, l, part);
                    for (i64 i = 0; i < n; i++) {
                        low[i] = fminf(low[i], part[i]);
                    }
                }
                survey_uniform(low, part, h, w, 0);
                survey_uniform(part, low, h, w, 1);
                for (int r = 0; r < h; r++) {
                    for (int c = 0; c < w; c++) {
                        open[(i64)(y + r) * sx + x + c] = low[(i64)r * w + c] < 65;
                    }
                }
            }
            a->used = saved;
        }
    }
    return open;
}

static u16 survey_half(f32 x) {
    if (x < 0x1p-14f) {
        return (u16)nearbyint(ldexp(x, 24));
    }
    int exponent;
    double significand = frexp(x, &exponent);
    return (u16)(((exponent + 14) << 10) + (int)nearbyint(ldexp(significand, 11)) - 1024);
}

static void survey_archive(
    Arena *a, const char *path, int count, char names[8][20], Array *arrays) {
    const char *part = fmt(a, "%s.part", path);
    int error = 0;
    zip_t *file = zip_open(part, ZIP_CREATE | ZIP_TRUNCATE, &error);
    need(file != NULL, "cannot make survey cells %s", path);
    for (int i = 0; i < count; i++) {
        Bytes b = npy_bytes(a, arrays[i]);
        char member[32];
        snprintf(member, sizeof(member), "%s.npy", names[i]);
        zip_source_t *src = zip_source_buffer(file, b.data, b.size, 0);
        zip_int64_t at = src ? zip_file_add(file, member, src, ZIP_FL_OVERWRITE) : -1;
        need(at >= 0 && !zip_set_file_compression(file, at, ZIP_CM_DEFLATE, 0),
            "cannot write %s to %s", member, path);
    }
    if (!count) {
        zip_source_t *empty = zip_source_buffer(file, "", 0, 0);
        need(empty != NULL, "cannot create empty survey cells");
        zip_int64_t at = zip_file_add(file, ".empty", empty, 0);
        need(at >= 0, "cannot create empty survey cells");
    }
    need(!zip_close(file) && !rename(part, path), "cannot publish survey cells %s", path);
}

static double survey_energy(Arena *a, const f32 *values, const f32 *sheet, const f32 *open, int h,
    int w, double pixel, int *cells) {
    if (!h || !w) {
        return 0;
    }
    size_t saved = a->used;
    i64 n = (i64)h * w;
    f32 *filtered[2];
    f32 *weights = push(a, n * sizeof(*weights)), *spare = push(a, n * sizeof(*spare));
    double scales[] = {0.4 / pixel, 3.0 / pixel};
    for (int j = 0; j < 2; j++) {
        f32 *out = filtered[j] = push(a, n * sizeof(*out));
        for (i64 i = 0; i < n; i++) {
            int seen = sheet[i] > 0.9f;
            weights[i] = seen;
            out[i] = seen ? values[i] : 0;
        }
        need(scales[j] < 63, "survey band smoothing exceeds supported scale");
        gaussian2(out, spare, h, w, scales[j]);
        gaussian2(weights, spare, h, w, scales[j]);
        for (i64 i = 0; i < n; i++) {
            out[i] /= fmaxf(weights[i], 0.001f);
        }
    }
    f32 *energy = spare;
    int count = 0;
    for (i64 i = 0; i < n; i++) {
        if (open[i] > 0.9f) {
            f32 delta = filtered[0][i] - filtered[1][i];
            energy[count++] = delta * delta;
        }
    }
    *cells = count;
    double sum = pairwise(energy, count);
    a->used = saved;
    return sum;
}

static void survey_map(Arena *a, const char *path, const u8 *sheet, const f32 *map, int h, int w) {
    size_t saved = a->used;
    i64 n = (i64)h * w;
    u8 *image = push(a, n);
    for (i64 i = 0; i < n; i++) {
        image[i] = sheet[i] ? nearbyintf(map[i] * 254) + 1 : 0;
    }
    write_png(a, path, image, w, h);
    a->used = saved;
}

static void survey_counts(json_object *row, i64 open, i64 total, f64 near, f64 far, f64 nband,
    f64 fband, int cells, i64 faces[4][3]) {
    json_object_object_add(row, "ruler", json_object_new_string("both directions read whole"));
    json_object_object_add(row, "pixels", json_object_new_int64(open));
    json_object_object_add(row, "sheet_pixels", json_object_new_int64(total));
    json_object_object_add(row, "forward", jnum(near));
    json_object_object_add(row, "reversed", jnum(far));
    json_object_object_add(row, "band_forward", jnum(nband));
    json_object_object_add(row, "band_reversed", jnum(fband));
    json_object_object_add(row, "cells", json_object_new_int(cells));
    json_object *count = json_object_new_object();
    const char *key[] = {"both", "near", "far", "neither"};
    for (int i = 0; i < 4; i++) {
        json_object *parts = json_object_new_array();
        for (int j = 0; j < 3; j++) {
            json_object_array_add(parts, json_object_new_int64(faces[i][j]));
        }
        json_object_object_add(count, key[i], parts);
    }
    json_object_object_add(row, "faces_open", count);
}

static void survey_ruler(Arena *a, Stage *stage, Job *job, Stack stack, int flip, const f32 *near,
    int total, double voxel, json_object *row) {
    i64 faces[4][3] = {{0}};
    if (total < (1 << 17)) {
        survey_counts(row, 0, 0, 0, 0, 0, 0, 0, faces);
        return;
    }
    int h = stack.height, w = stack.width, sy, sx, spots[2][2];
    i64 size = (i64)h * w;
    const u8 *sheet =
        stack.cube + (i64)(flip ? stack.layers - stack.layers / 2 - 1 : stack.layers / 2) * size;
    int windows = survey_windows(a, sheet, h, w, spots, &sy, &sx), nh = sy / 8, nw = sx / 8;
    u8 *back = push(a, DEPTH * size);
    f32 *far = push(a, size * sizeof(*far));
    window_of(stack, !flip, back);
    fix_stem(stage->reader, back, h, w);
    ink_read(stage->reader, back, h, w, far);
    const char *tail = disk_base(job->out);
    survey_map(
        a, fmt(a, "%.*sreversed-%s.png", (int)(tail - job->out), job->out, tail), sheet, far, h, w);
    i64 open = 0, pixels = 0;
    double counts[2] = {0}, bands[2] = {0};
    int cells = 0, kept = 0;
    char names[8][20];
    Array arrays[8];
    const char *keys[] = {"forward", "reversed", "sheet", "open"};
    for (int k = 0; k < windows; k++) {
        i64 m = (i64)nh * nw;
        u16 *packed = push(a, 4 * m * sizeof(*packed));
        size_t saved = a->used;
        f32 *maps = pushz(a, 4 * m * sizeof(*maps));
        int y0 = spots[k][0], x0 = spots[k][1];
        const u8 *front_gap = survey_open(a, stack, y0, x0, sy, sx, flip);
        const u8 *back_gap = survey_open(a, stack, y0, x0, sy, sx, !flip);
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                i64 at = (i64)(y0 + y) * w + x0 + x;
                i64 cell = (i64)(y / 8) * nw + x / 8, pixel = (i64)y * sx + x;
                int seen = sheet[at] != 0, front = seen && front_gap[pixel];
                int rear = seen && back_gap[pixel];
                int high = near[at] >= SURE, low = far[at] >= SURE;
                if (seen) {
                    int bucket = front ? (rear ? 0 : 1) : (rear ? 2 : 3);
                    faces[bucket][0]++;
                    faces[bucket][1] += high;
                    faces[bucket][2] += low;
                }
                pixels += seen;
                open += front;
                counts[0] += front && high;
                counts[1] += front && low;
                if (y < nh * 8 && x < nw * 8) {
                    maps[cell] += near[at] / 64;
                    maps[m + cell] += far[at] / 64;
                    maps[2 * m + cell] += seen / 64.0f;
                    maps[3 * m + cell] += front / 64.0f;
                }
            }
        }
        int hits = 0;
        for (int d = 0; d < 2; d++) {
            bands[d] += survey_energy(
                a, maps + d * m, maps + 2 * m, maps + 3 * m, nh, nw, voxel * 8, &hits);
        }
        cells += hits;
        for (int d = 0; d < 4; d++) {
            for (i64 i = 0; i < m; i++) {
                packed[d * m + i] = survey_half(maps[d * m + i]);
            }
            snprintf(names[kept], sizeof(names[kept]), "%s%d", keys[d], k);
            arrays[kept++] = (Array){.descr = "<f2",
                .ndim = 2,
                .shape = {nh, nw},
                .data = packed + d * m,
                .size = m * sizeof(*packed)};
        }
        a->used = saved;
    }
    survey_archive(a, fmt(a, "%s.npz", job->out), kept, names, arrays);
    survey_counts(row, open, pixels, counts[0], counts[1], bands[0], bands[1], cells, faces);
}

static void survey_read_piece(Arena *a, Stage *stage, Job *job, Bricks *bricks) {
    size_t saved = a->used;
    double start = now();
    const char *path = fmt(a, "%s/meta.json", job->folder);
    json_object *meta = json_read(path);
    need(meta != NULL, "no survey piece metadata at %s", path);
    Surface mesh = load_surface(a, job->folder);
    Readable surface = readable(a, mesh, stage->line, stage->rows);
    Stack stack = flatten(a, &surface, 1, bricks, LAYERS);
    Recto recto = recto_face(stack);
    int flip = !strcmp(recto.face, "far"), h = stack.height, w = stack.width;
    i64 size = (i64)h * w, seen = 0, sure = 0;
    u8 *window = push(a, DEPTH * size);
    f32 *near = push(a, size * sizeof(*near));
    window_of(stack, flip, window);
    fix_stem(stage->reader, window, h, w);
    ink_read(stage->reader, window, h, w, near);
    const u8 *sheet =
        stack.cube + (i64)(flip ? stack.layers - stack.layers / 2 - 1 : stack.layers / 2) * size;
    survey_map(a, fmt(a, "%s.png", job->out), sheet, near, h, w);
    for (i64 i = 0; i < size; i++) {
        seen += sheet[i] > 0;
        sure += sheet[i] > 0 && near[i] >= SURE;
    }
    json_object *row = json_object_new_object();
    json_object_object_add(row, "piece", json_object_new_string(disk_base(job->folder)));
    json_object_object_add(row, "face", json_object_new_string(recto.face));
    json_object_object_add(
        row, "polarity", recto.tiles >= FEWEST_FIBRE_TILES ? jnum(recto.polarity) : NULL);
    json_object_object_add(row, "area_mm2", jnum(board_number(meta, "area_mm2")));
    json_object_object_add(row, "sure", jnum(seen ? sure / (double)seen : 0));
    survey_ruler(a, stage, job, stack, flip, near, seen, mesh.voxel / 1000, row);
    json_object_object_add(row, "seconds", jnum(round((now() - start) * 10) / 10));
    survey_write(fmt(a, "%s.json", job->out), row);
    json_object_put(meta);
    json_object_put(row);
    a->used = saved;
}

static void survey_summary(Arena *a, const char *folder, const char *sample, double began) {
    glob_t paths = {0};
    glob(fmt(a, "%s/piece-*.json", folder), 0, NULL, &paths);
    double *ratios = push(a, paths.gl_pathc * sizeof(*ratios));
    i64 pixels = 0, sheet = 0;
    double totals[2] = {0}, bands[2] = {0}, excess = 0, square = 0;
    int n = 0, high = 0, low = 0;
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        json_object *row = json_read(paths.gl_pathv[i]);
        i64 count = board_integer(row, "pixels");
        if (count > 0) {
            double forward = board_number(row, "forward"), reversed = board_number(row, "reversed");
            double fband = board_number(row, "band_forward"),
                   rband = board_number(row, "band_reversed");
            double difference = (forward - reversed) / count;
            pixels += count;
            sheet += board_integer(row, "sheet_pixels");
            totals[0] += forward;
            totals[1] += reversed;
            bands[0] += fband;
            bands[1] += rband;
            excess += difference;
            square += difference * difference;
            double ratio = sqrt(fband / fmax(rband, 1e-9));
            ratios[n++] = ratio;
            high += ratio > 1.2;
            low += ratio < 1 / 1.2;
        }
        json_object_put(row);
    }
    globfree(&paths);
    json_object *summary = json_object_new_object();
    json_object_object_add(summary, "sample", json_object_new_string(sample));
    json_object_object_add(summary, "pieces", json_object_new_int(n));
    if (n) {
        qsort(ratios, n, sizeof(*ratios), survey_depth_order);
        const char *names[] = {"open_share", "forward", "reversed", "excess", "excess_error",
            "band_ratio", "median_piece"};
        double heat[] = {sheet ? pixels / (double)sheet : 0, totals[0] / pixels, totals[1] / pixels,
            excess / n, n > 1 ? sqrt(fmax(0, (square - excess * (excess / n)) / (n - 1) / n)) : NAN,
            sqrt(bands[0] / fmax(bands[1], 1e-9)), (ratios[n / 2] + ratios[(n - 1) / 2]) / 2};
        for (int i = 0; i < 7; i++) {
            json_object *value = i == 0 && !sheet ? NULL
                : isnan(heat[i])                  ? json_object_new_double(heat[i])
                                                  : jnum(heat[i]);
            json_object_object_add(summary, names[i], value);
        }
        json_object_object_add(summary, "high", json_object_new_int(high));
        json_object_object_add(summary, "low", json_object_new_int(low));
    }
    json_object_object_add(summary, "failed", json_object_new_int(0));
    json_object_object_add(summary, "minutes", jnum(round((now() - began) / 6) / 10));
    puts(board_json(summary));
    json_object_put(summary);
    fflush(stdout);
}

typedef struct {
    Arena a;
    SurveyBand band;
    Stage stage;
    Job *jobs;
    const char *store;
    int count;
    pid_t fetch;
} SurveyTask;

static SurveyTask survey_task(const char *item, int mode) {
    SurveyTask task = {.a = arena(256 * GB)};
    task.band = survey_band(&task.a, item);
    task.jobs = survey_plan(&task.a, task.band, &task.stage, mode, &task.count);
    task.store = survey_store(&task.a, task.band, mode);
    return task;
}

static int survey_wait(pid_t pid) {
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        need(errno == EINTR, "cannot wait for survey: %s", strerror(errno));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

static void survey_fetch_start(SurveyTask *t, int floor) {
    if (!t->count) {
        return;
    }
    fflush(NULL);
    t->fetch = fork();
    need(t->fetch >= 0, "cannot start survey fetch: %s", strerror(errno));
    if (!t->fetch) {
        if (floor) {
            const char *name = disk_base(t->store);
            const char *parent = fmt(&t->a, "%.*s", (int)(name - t->store), t->store);
            mkdirs(parent);
            if (board_disc(parent) < 40) {
                puts("disc full: try later");
                exit(75);
            }
        }
        fetch_store(&t->stage, t->jobs, t->count, t->store, &t->a);
        if (floor && board_disc(t->store) < 40) {
            puts("disc full: try later");
            exit(75);
        }
        exit(0);
    }
}

static int survey_finish_read(SurveyTask *t, int mode) {
    double began = now();
    fflush(NULL);
    pid_t reader = fork();
    need(reader >= 0, "cannot start survey reader: %s", strerror(errno));
    if (reader) {
        return survey_wait(reader);
    }
    if (mode == SURVEY_SECOND) {
        setenv("HERCULEX_INK_MODEL", "reader-v3", 1);
    }
    t->stage.model = chosen_model(&t->a);
    Bricks store = {0};
    bricks_open(&t->a, &store, t->store);
    i64 columns = 1;
    for (int i = 0; i < t->count; i++) {
        i64 width = columns_of(&t->a, t->jobs + i);
        columns = width > columns ? width : columns;
    }
    t->stage.reader = reader_open(&t->a, t->stage.model.weights, columns);
    for (int i = 0; i < t->count; i++) {
        survey_read_piece(&t->a, &t->stage, t->jobs + i, &store);
        if (i % 50 == 49) {
            printf("%s %d of %d %.0f s\n", t->band.sample, i + 1, t->count, now() - began);
            fflush(stdout);
        }
    }
    if (store.data) {
        munmap((void *)store.data, store.mapped);
    }
    remove_tree(t->store);
    const char *out = t->jobs[0].out, *last = strrchr(out, '/');
    survey_summary(&t->a, fmt(&t->a, "%.*s", (int)(last - out), out), t->band.sample, began);
    exit(0);
}

int survey_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.survey.scroll_survey all|look|replicate|reread|row-sample|second "
             "SAMPLE/SCAN/BAND ...");
        return !argc;
    }
    int mode = survey_kind(argv[0]);
    if (argc == 1) {
        return 0;
    }
    SurveyTask task = survey_task(argv[1], mode);
    survey_fetch_start(&task, 0);
    int status = 0;
    for (int i = 1; i < argc; i++) {
        status = task.fetch ? survey_wait(task.fetch) : 0;
        if (status) {
            munmap(task.a.base, task.a.size);
            return status;
        }
        SurveyTask next = {0};
        if (i + 1 < argc) {
            next = survey_task(argv[i + 1], mode);
            if (strcmp(task.store, next.store) != 0) {
                survey_fetch_start(&next, 0);
            }
        }
        if (task.count) {
            status = survey_finish_read(&task, mode);
        }
        munmap(task.a.base, task.a.size);
        if (status) {
            if (next.fetch) {
                survey_wait(next.fetch);
            }
            return status;
        }
        task = next;
        if (next.count && !next.fetch) {
            survey_fetch_start(&task, 0);
        }
    }
    return status;
}
