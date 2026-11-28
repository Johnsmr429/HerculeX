#ifndef HX_UNITY
const char hx_usage[] = "Measure published truth vertices against the papyrus ridges in cached CT.";
#endif

enum { RIDGE_POINTS = 49, RIDGE_MARGIN = 16, RIDGE_ZONE_COUNT = 256 };

typedef struct {
    const u8 *pack;
    i64 shape[3], chunk[3], grid[3], *offset;
    i64 length, bytes;
    u8 fill;
} RidgeVolume;

static RidgeVolume ridge_volume_open(Arena *a, const char *sample, const char *scan) {
    char *source;
    const char *url = zarr_level(a, sample, scan, 0, &source);
    const char *ending = strstr(source, ".zarr/");
    need(ending != NULL, "scan %s %s has no zarr path", sample, scan);
    const char *name = ending;
    while (name > source && name[-1] != '/') {
        name--;
    }
    const char *cache = env("HERCULEX_CHUNKS", fmt(a, "%s/cache/chunks", OUTPUTS));
    const char *folder = fmt(a, "%s/%s/%.*s/0", cache, sample, (int)(ending - name), name);
    json_object *meta = json_read(fmt(a, "%s/.zarray", folder));
    if (!meta) {
        meta = http_json(fmt(a, "%s/.zarray", url));
    }
    RidgeVolume v = {
        .bytes = 1, .fill = json_object_get_int(json_object_object_get(meta, "fill_value"))};
    i64 cells = 1;
    for (int i = 0; i < 3; i++) {
        v.shape[i] = json_object_get_int64(json_object_array_get_idx(jget(meta, "shape"), i));
        v.chunk[i] = json_object_get_int64(json_object_array_get_idx(jget(meta, "chunks"), i));
        need(v.shape[i] > 0 && v.chunk[i] > 0, "invalid CT shape or chunk at %s", folder);
        v.grid[i] = (v.shape[i] + v.chunk[i] - 1) / v.chunk[i];
        cells *= v.grid[i];
        v.bytes *= v.chunk[i];
    }
    json_object_put(meta);
    v.offset = push(a, cells * sizeof(*v.offset));
    for (i64 i = 0; i < cells; i++) {
        v.offset[i] = UNKNOWN;
    }
    const char *pack = fmt(a, "%s/chunks.pack", folder);
    int fd = open(pack, O_RDONLY | O_CLOEXEC);
    struct stat st = {0};
    if (fd >= 0) {
        need(!fstat(fd, &st), "cannot stat %s", pack);
        if (st.st_size) {
            v.pack = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            need(v.pack != MAP_FAILED, "cannot map %s", pack);
            v.length = st.st_size;
        }
        close(fd);
    }
    const char *index = fmt(a, "%s/index.sqlite", folder);
    if (exists(index)) {
        sqlite3 *db = NULL;
        need(sqlite3_open_v2(index, &db, SQLITE_OPEN_READONLY, NULL) == SQLITE_OK,
            "cannot read chunk index at %s", index);
        sqlite3_busy_timeout(db, 10000);
        sqlite3_stmt *q = NULL;
        need(sqlite3_prepare_v2(db, "SELECT z,y,x,offset FROM chunks", -1, &q, NULL) == SQLITE_OK,
            "cannot list chunks at %s", index);
        int rc;
        while ((rc = sqlite3_step(q)) == SQLITE_ROW) {
            i64 z = sqlite3_column_int64(q, 0), y = sqlite3_column_int64(q, 1),
                x = sqlite3_column_int64(q, 2), at = sqlite3_column_int64(q, 3);
            if (z >= 0 && z < v.grid[0] && y >= 0 && y < v.grid[1] && x >= 0 && x < v.grid[2]) {
                need(at < 0 || at <= v.length - v.bytes, "incomplete CT chunk at %s", pack);
                v.offset[(z * v.grid[1] + y) * v.grid[2] + x] = at < 0 ? ABSENT : at;
            }
        }
        need(rc == SQLITE_DONE, "cannot read chunk index at %s", index);
        sqlite3_finalize(q);
        sqlite3_close(db);
    }
    return v;
}

static void ridge_volume_close(const RidgeVolume *v) {
    if (v->pack) {
        munmap((void *)v->pack, v->length);
    }
}

static i64 ridge_offset(const RidgeVolume *v, i64 z, i64 y, i64 x) {
    if (z < 0 || z >= v->grid[0] || y < 0 || y >= v->grid[1] || x < 0 || x >= v->grid[2]) {
        return ABSENT;
    }
    return v->offset[(z * v->grid[1] + y) * v->grid[2] + x];
}

static int ridge_known(const RidgeVolume *v, const i64 lo[3], const i64 hi[3]) {
    i64 first[3], last[3];
    for (int d = 0; d < 3; d++) {
        first[d] = lo[d] < 0 ? -1 : lo[d] / v->chunk[d];
        last[d] = (hi[d] - 1) / v->chunk[d];
    }
    for (i64 z = first[0]; z <= last[0]; z++) {
        for (i64 y = first[1]; y <= last[1]; y++) {
            for (i64 x = first[2]; x <= last[2]; x++) {
                if (ridge_offset(v, z, y, x) == UNKNOWN) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static u8 ridge_voxel(const RidgeVolume *v, i64 z, i64 y, i64 x) {
    if (z < 0 || y < 0 || x < 0 || z >= v->shape[0] || y >= v->shape[1] || x >= v->shape[2]) {
        return v->fill;
    }
    i64 offset = ridge_offset(v, z / v->chunk[0], y / v->chunk[1], x / v->chunk[2]);
    if (offset < 0) {
        return v->fill;
    }
    i64 pos = ((z % v->chunk[0]) * v->chunk[1] + y % v->chunk[1]) * v->chunk[2] + x % v->chunk[2];
    return v->pack[offset + pos];
}

static f32 ridge_interp_at(const RidgeVolume *v, const double point[3]) {
    double frac[3];
    i64 lo[3];
    for (int d = 0; d < 3; d++) {
        double q = point[d];
        lo[d] = floor(q);
        frac[d] = q - lo[d];
    }
    double out = 0;
    for (int i = 0; i < 8; i++) {
        int z = i >> 2, y = (i >> 1) & 1, x = i & 1;
        double w =
            (z ? frac[0] : 1 - frac[0]) * (y ? frac[1] : 1 - frac[1]) * (x ? frac[2] : 1 - frac[2]);
        out += w * ridge_voxel(v, lo[0] + z, lo[1] + y, lo[2] + x);
    }
    return out;
}

static f32 ridge_interp(const RidgeVolume *v, const f32 *p, const f32 *n, double shift) {
    double point[3];
    for (int d = 0; d < 3; d++) {
        point[d] = (double)p[2 - d] + shift * n[2 - d];
    }
    return ridge_interp_at(v, point);
}

typedef struct {
    i32 z, y, x, index;
    u8 zone;
} RidgeVertex;

static int ridge_order(const void *left, const void *right) {
    const RidgeVertex *a = left, *b = right;
    if (a->zone != b->zone) {
        return a->zone - b->zone;
    }
    if (a->z != b->z) {
        return a->z - b->z;
    }
    if (a->y != b->y) {
        return a->y - b->y;
    }
    if (a->x != b->x) {
        return a->x - b->x;
    }
    return a->index - b->index;
}

static int ridge_same_cell(const RidgeVertex *a, const RidgeVertex *b) {
    return a->z == b->z && a->y == b->y && a->x == b->x && a->zone == b->zone;
}

static i64 ridge_collect(
    Arena *a, const RidgeVolume *v, const BenchTruth *t, int limit, i64 *index, f32 *profiles) {
    size_t saved = a->used;
    RidgeVertex *row = push(a, t->count * sizeof(*row));
    i64 count = 0, used = 0;
    for (i64 i = 0; i < t->count; i++) {
        const f32 *p = t->points + 3 * i, *n = t->normals + 3 * i;
        if (!(t->flags[i] & BENCH_DUPLICATE) && package_good_normal(n)) {
            row[count++] = (RidgeVertex){.z = floor(p[2] / v->chunk[0]),
                .y = floor(p[1] / v->chunk[1]),
                .x = floor(p[0] / v->chunk[2]),
                .index = i,
                .zone = t->zone[i]};
        }
    }
    qsort(row, count, sizeof(*row), ridge_order);
    i32 *pick = push(a, count * sizeof(*pick));
    i64 *bound = push(a, (count + 1) * sizeof(*bound));
    Pcg64 rng = pcg64(0);
    for (i64 first = 0, end; first < count; first = end) {
        int n = 0;
        for (end = first; end < count && row[end].zone == row[first].zone; end++) {
            if (end == first || !ridge_same_cell(row + end, row + end - 1)) {
                pick[n] = n;
                bound[n++] = end;
            }
        }
        bound[n] = end;
        i64 size = n < limit ? n : limit;
        package_choose(a, &rng, pick, n, size);
        for (i64 k = 0; k < size; k++) {
            int chosen = pick[k];
            RidgeVertex cell = row[bound[chosen]];
            i64 key[] = {cell.z, cell.y, cell.x}, lo[3], hi[3];
            for (int d = 0; d < 3; d++) {
                lo[d] = key[d] * v->chunk[d] - RIDGE_MARGIN;
                hi[d] = (key[d] + 1) * v->chunk[d] + RIDGE_MARGIN;
            }
            if (!ridge_known(v, lo, hi)) {
                continue;
            }
            for (i64 i = bound[chosen]; i < bound[chosen + 1]; i++) {
                i64 j = row[i].index;
                index[used] = j;
                for (int d = 0; d < RIDGE_POINTS; d++) {
                    profiles[used * RIDGE_POINTS + d] =
                        ridge_interp(v, t->points + 3 * j, t->normals + 3 * j, 0.5 * d - 12);
                }
                used++;
            }
        }
    }
    a->used = saved;
    return used;
}

static void ridge_smooth(const f32 *raw, int n, double sigma, f32 *smooth) {
    int radius = nearbyint(8 * sigma);
    double weights[49], sum = 0;
    for (int d = -radius; d <= radius; d++) {
        double x = d / (2 * sigma);
        weights[d + radius] = exp(-0.5 * x * x);
        sum += weights[d + radius];
    }
    for (int d = 0; d < 2 * radius + 1; d++) {
        weights[d] /= sum;
    }
    for (int i = 0; i < n; i++) {
        double v = 0;
        for (int d = -radius; d <= radius; d++) {
            int j = i + d;
            if (j < 0) {
                j = -1 - j;
            } else if (j >= n) {
                j = 2 * n - 1 - j;
            }
            v += raw[j] * weights[d + radius];
        }
        smooth[i] = v;
    }
}

static f32 ridge_peak(const f32 *p, int center) {
    int distance = RIDGE_POINTS, best = -1;
    for (int i = 1; i + 1 < RIDGE_POINTS; i++) {
        if (p[i] > p[i - 1] && p[i] >= p[i + 1] && abs(i - center) < distance) {
            distance = abs(i - center);
            best = i;
        }
    }
    return best < 0 ? NAN : 0.5 * (best - center);
}

typedef struct {
    const f32 *raw;
    f32 *found[2][3];
    i64 n;
} RidgePeaks;

static void ridge_find(void *payload, i64 block) {
    RidgePeaks *p = payload;
    for (i64 i = block * 128; i < (block + 1) * 128 && i < p->n; i++) {
        for (int s = 0; s < 2; s++) {
            f32 smooth[RIDGE_POINTS];
            ridge_smooth(p->raw + i * RIDGE_POINTS, RIDGE_POINTS, s ? 1 : 3, smooth);
            for (int k = 0; k < 3; k++) {
                p->found[s][k][i] = ridge_peak(smooth, k == 0 ? 24 : k == 1 ? 36 : 12);
            }
        }
    }
}

static u16 ridge_half(f32 x) {
    if (x < 0x1p-14f) {
        return nearbyint(ldexp(x, 24));
    }
    int exponent;
    double significand = frexp(x, &exponent);
    return ((exponent + 14) << 10) + (int)nearbyint(ldexp(significand, 11)) - 1024;
}

static json_object *ridge_abs_quantiles(double *v, i64 n) {
    if (!n) {
        return NULL;
    }
    qsort(v, n, sizeof(*v), bench_double_order);
    double q = 0.9 * (n - 1);
    i64 j = floor(q), k = ceil(q);
    return package_pair(package_round((v[(n - 1) / 2] + v[n / 2]) / 2, 2),
        package_round(v[j] + (v[k] - v[j]) * (q - j), 2));
}

static int ridge_group(const BenchTruth *t, i64 i, int group) {
    if (group == -1) {
        return 1;
    }
    if (group == RIDGE_ZONE_COUNT) {
        return (t->flags[i] & BENCH_PRESSED) != 0;
    }
    if (group == RIDGE_ZONE_COUNT + 1) {
        return (t->flags[i] & BENCH_PRESSED) == 0;
    }
    return t->zone[i] == group;
}

static json_object *ridge_summary(
    Arena *a, const BenchTruth *t, const i64 *index, const RidgePeaks *peaks, int group) {
    size_t saved = a->used;
    i32 *members = push(a, peaks->n * sizeof(*members));
    i64 n = 0;
    f32 mean[RIDGE_POINTS] = {0};
    for (i64 i = 0; i < peaks->n; i++) {
        if (ridge_group(t, index[i], group)) {
            members[n++] = i;
            for (int d = 0; d < RIDGE_POINTS; d++) {
                mean[d] += peaks->raw[i * RIDGE_POINTS + d];
            }
        }
    }
    if (!n) {
        a->used = saved;
        return NULL;
    }
    double *values = push(a, n * sizeof(*values));
    json_object *out = json_object_new_object();
    json_object_object_add(out, "vertices", json_object_new_int64(n));
    for (int s = 0; s < 2; s++) {
        json_object *row = json_object_new_object();
        i64 count = 0, two = 0, four = 0;
        for (i64 i = 0; i < n; i++) {
            f32 v = peaks->found[s][0][members[i]];
            if (isfinite(v)) {
                values[count++] = v;
                two += fabsf(v) <= 2;
                four += fabsf(v) <= 4;
            }
        }
        double median = bench_median(values, count, NAN);
        json_object_object_add(row, "signed_p50_vox", isfinite(median) ? jnum(median) : NULL);
        for (i64 i = 0; i < count; i++) {
            values[i] = fabs(values[i]);
        }
        json_object_object_add(row, "abs_p50_p90_vox", ridge_abs_quantiles(values, count));
        bench_ratio(row, "within_2_vox", two, count, 4);
        bench_ratio(row, "within_4_vox", four, count, 4);
        const char *nulls[] = {"null_+6_abs_p50_p90_vox", "null_-6_abs_p50_p90_vox"};
        for (int k = 0; k < 2; k++) {
            count = 0;
            for (i64 i = 0; i < n; i++) {
                f32 v = peaks->found[s][k + 1][members[i]];
                if (isfinite(v)) {
                    values[count++] = fabsf(v);
                }
            }
            json_object_object_add(row, nulls[k], ridge_abs_quantiles(values, count));
        }
        json_object_object_add(out, s ? "sigma_1" : "sigma_3", row);
    }
    int best = 0;
    json_object *profile = json_object_new_array_ext(25);
    for (int d = 0; d < RIDGE_POINTS; d++) {
        if (mean[d] > mean[best]) {
            best = d;
        }
        if (d % 2 == 0) {
            json_object_array_add(profile, jnum(package_round(mean[d] / (f32)n, 1)));
        }
    }
    json_object_object_add(out, "mean_profile_peak_vox", jnum(0.5 * best - 12));
    json_object_object_add(out, "mean_profile_every_vox", profile);
    a->used = saved;
    return out;
}

static json_object *ridge_check(Arena *a, const char *name, int limit) {
    size_t saved = a->used;
    const char *folder = fmt(a, "%s/bench/controls/%s", OUTPUTS, name);
    BenchTruth t = bench_truth(a, folder);
    json_object *entry = jget(t.record, "scan");
    RidgeVolume volume = ridge_volume_open(a, jstr(entry, "sample"), jstr(entry, "scan"));
    i64 *index = push(a, t.count * sizeof(*index));
    f32 *raw = push(a, t.count * RIDGE_POINTS * sizeof(*raw));
    i64 count = ridge_collect(a, &volume, &t, limit, index, raw);
    ridge_volume_close(&volume);
    need(count > 0, "%s has no complete cached CT chunks in the selected band", name);
    RidgePeaks peaks = {.raw = raw, .n = count};
    for (int s = 0; s < 2; s++) {
        for (int k = 0; k < 3; k++) {
            peaks.found[s][k] = push(a, count * sizeof(f32));
        }
    }
    parallel((count + 127) / 128, 4, ridge_find, &peaks);
    json_object *out = json_object_new_object(), *groups = json_object_new_object();
    json_object_object_add(out, "control", json_object_new_string(name));
    json_object_object_add(out, "scan", json_object_get(entry));
    json_object_object_add(out, "window_vox", package_pair(-12, 12));
    json_object_object_add(out, "step_vox", jnum(0.5));
    json_object_object_add(out, "smoothing_sigma_vox", package_pair(3, 1));
    json_object_object_add(out, "null_offset_vox", jnum(6));
    json_object_object_add(
        out, "sign", json_object_new_string("+ is towards the scroll axis along the truth normal"));
    json_object_object_add(out, "chunks_per_zone", json_object_new_int(limit));
    json_object_object_add(out, "by_group", groups);
    for (int group = -1; group <= RIDGE_ZONE_COUNT + 1; group++) {
        const char *label = group < 0                                   ? "all"
            : group == RIDGE_ZONE_COUNT                                 ? "pressed"
            : group == RIDGE_ZONE_COUNT + 1                             ? "loose"
            : group < (int)(sizeof(bench_zones) / sizeof(*bench_zones)) ? bench_zones[group]
                                                                        : NULL;
        if (label) {
            json_object *summary = ridge_summary(a, &t, index, &peaks, group);
            if (summary) {
                json_object_object_add(groups, label, summary);
            }
        }
    }
    u16 *halves = push(a, count * RIDGE_POINTS * sizeof(*halves));
    f32 offsets[RIDGE_POINTS];
    for (int d = 0; d < RIDGE_POINTS; d++) {
        offsets[d] = 0.5 * d - 12;
    }
    for (i64 i = 0; i < count * RIDGE_POINTS; i++) {
        halves[i] = ridge_half(raw[i]);
    }
    const char *keys[] = {"truth_index", "ridge_sigma3_vox", "profiles", "offsets_vox"};
    Array arrays[] = {
        {.descr = "<i8", .ndim = 1, .shape = {count}, .data = index, .size = count * 8},
        {.descr = "<f4", .ndim = 1, .shape = {count}, .data = peaks.found[0][0], .size = count * 4},
        {.descr = "<f2",
            .ndim = 2,
            .shape = {count, RIDGE_POINTS},
            .data = halves,
            .size = count * RIDGE_POINTS * 2},
        {.descr = "<f4",
            .ndim = 1,
            .shape = {RIDGE_POINTS},
            .data = offsets,
            .size = sizeof(offsets)},
    };
    const char *npz = fmt(a, "%s/bench/checks/%s-ridge.npz", OUTPUTS, name);
    const char *part = fmt(a, "%s.%d.part", npz, getpid());
    mkparent(npz);
    npz_write_codec(part, sizeof(keys) / sizeof(*keys), keys, arrays, ZIP_CM_DEFLATE);
    need(!rename(part, npz), "cannot publish ridge sample %s: %s", npz, strerror(errno));
    json_write(fmt(a, "%s/bench/checks/%s-ridge.json", OUTPUTS, name), out);
    json_object_put(t.record);
    a->used = saved;
    return out;
}

int bench_ridge_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.ridge CONTROL ... [--chunks N]");
        return !argc;
    }
    Arena a = arena(16ull << 30);
    int chunks = 240, count = 0;
    const char **names = push(&a, argc * sizeof(*names));
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--chunks")) {
            need(i + 1 < argc, "ridge --chunks needs a number");
            chunks = board_value(argv[++i]);
        } else {
            need(argv[i][0] != '-', "unknown ridge option %s", argv[i]);
            names[count++] = argv[i];
        }
    }
    need(chunks > 0 && count > 0, "ridge needs a control and a positive chunk count");
    for (int i = 0; i < count; i++) {
        json_object *report = ridge_check(&a, names[i], chunks);
        puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PRETTY));
        json_object_put(report);
    }
    munmap(a.base, a.size);
    return 0;
}
