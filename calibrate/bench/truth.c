#ifndef HX_UNITY
const char hx_usage[] = "Describe materialized v1 or wrap-name truth and compare their windings.";
#endif

typedef struct {
    const char *control, *source, *axis_source;
    f32 *points, *normals;
    const i32 *frame;
    const i64 *mesh_count;
    int *mesh, *wrap;
    double *winding, *knots;
    u8 *unresolved;
    i64 n;
    int meshes, nknots, sense;
    double voxel, spacing;
    json_object *names, *audit;
} BenchLegacyTruth;

static const char *legacy_group_path(Arena *a, const char *control) {
    return fmt(a, "%s/calibration/geometry_wrap/truth/%s.npz", LIBRARY, control);
}

static const char *legacy_names_path(Arena *a, const char *control) {
    return fmt(a, "%s/controls/sheets/%s/truth.npz", LIBRARY, control);
}

static json_object *legacy_available(Arena *a, const char *control) {
    json_object *array = json_object_new_array();
    if (exists(legacy_names_path(a, control))) {
        json_object_array_add(array, json_object_new_string("names"));
    }
    if (exists(legacy_group_path(a, control))) {
        json_object_array_add(array, json_object_new_string("groups"));
    }
    return array;
}

static int legacy_wrap(const char *name) {
    for (const char *at = name; (at = strstr(at, "-w")); at += 2) {
        char *end;
        long n = strtol(at + 2, &end, 10);
        if (isdigit((unsigned char)at[2]) && end > at + 2 && *end == '_') {
            return n;
        }
    }
    return -1;
}

static int legacy_utf8(char *s, u32 cp) {
    if (cp < 0x80) {
        s[0] = cp;
        return 1;
    }
    int bytes = cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
    u8 first[] = {0, 0, 0xc0, 0xe0, 0xf0};
    for (int i = bytes - 1; i > 0; i--) {
        s[i] = 0x80 | (cp & 0x3f);
        cp >>= 6;
    }
    s[0] = first[bytes] | cp;
    return bytes;
}

static json_object *legacy_names(Arena *a, const char *path) {
    size_t saved = a->used;
    Array names = npz_get(a, path, "names");
    int width = (int)strtol(names.descr + 2, NULL, 10);
    need(width > 0 && (names.descr[1] == 'U' || names.descr[1] == 'S'),
        "unexpected truth names in %s", path);
    char *name = push(a, 4 * width + 1);
    json_object *out = json_object_new_array();
    for (i64 i = 0; i < elements(names); i++) {
        int used = 0;
        for (int j = 0; j < width; j++) {
            i64 at = i * width + j;
            u32 cp = names.descr[1] == 'U' ? ((u32 *)names.data)[at] : ((u8 *)names.data)[at];
            if (!cp) {
                break;
            }
            used +=
                names.descr[1] == 'U' ? legacy_utf8(name + used, cp) : (name[used] = (char)cp, 1);
        }
        name[used] = 0;
        json_object_array_add(out, json_object_new_string(name));
    }
    a->used = saved;
    return out;
}

typedef struct {
    double spacing;
    i64 count;
} LegacySpacing;

static int legacy_spacing_order(const void *a, const void *b) {
    double x = ((const LegacySpacing *)a)->spacing, y = ((const LegacySpacing *)b)->spacing;
    return (x > y) - (x < y);
}

static double legacy_spacing(
    Arena *a, json_object *rec, const BenchLegacyTruth *t, double thinned) {
    size_t saved = a->used;
    const char *sample = jstr(rec, "sample"), *volume = jstr(rec, "volume");
    const char *pattern = fmt(
        a, "%s/published/open_data/%s/segments/*/mesh/*-on-%s-*um.tifxyz", LIBRARY, sample, volume);
    glob_t paths = {0};
    glob(pattern, 0, NULL, &paths);
    json_object *folders = json_object_new_object();
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        const char *path = paths.gl_pathv[i];
        if (!exists(fmt(a, "%s/z.tif", path))) {
            continue;
        }
        const char *tail = bench_basename(path);
        const char *parent = fmt(a, "%.*s", (int)(tail - path - 1), path);
        const char *mesh = bench_basename(parent);
        const char *segment = bench_basename(fmt(a, "%.*s", (int)(mesh - parent - 1), parent));
        json_object_object_add(folders, segment, json_object_new_string(path));
    }
    globfree(&paths);
    LegacySpacing *seen = push(a, t->meshes * sizeof(*seen));
    int found = 0;
    i64 total = 0;
    for (int i = 0; i < t->meshes; i++) {
        const char *name = board_item(t->names, i), *folder = board_text(folders, name);
        if (!*folder || !t->mesh_count[i]) {
            continue;
        }
        json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
        need(meta != NULL, "no published mesh metadata at %s", folder);
        double scale = json_object_get_double(json_object_array_get_idx(jget(meta, "scale"), 0));
        json_object_put(meta);
        seen[found++] = (LegacySpacing){
            fmax(1, nearbyint(thinned * scale / t->voxel)) / scale * t->voxel, t->mesh_count[i]};
        total += t->mesh_count[i];
    }
    json_object_put(folders);
    qsort(seen, found, sizeof(*seen), legacy_spacing_order);
    double result = thinned;
    for (int i = 0; i < found; i++) {
        total -= 2 * seen[i].count;
        if (total <= 0) {
            result = seen[i].spacing;
            break;
        }
    }
    a->used = saved;
    return result;
}

static void legacy_group_knots(Arena *a, BenchLegacyTruth *t, const char *path) {
    double scale = exp2(bench_number(npz_get(a, path, "level"), 0));
    Array knots = npz_get(a, path, "knots");
    need(knots.ndim == 2 && knots.shape[1] == 3, "invalid v1 axis in %s", path);
    t->knots = push(a, elements(knots) * sizeof(*t->knots));
    t->nknots = knots.shape[0];
    for (i64 i = 0; i < elements(knots); i++) {
        t->knots[i] = bench_number(knots, i) * scale + 0.5 * (scale - 1);
    }
    t->axis_source = "v1 wrap truth knots";
}

static void legacy_fit_axis(Arena *a, BenchLegacyTruth *t);
static json_object *legacy_name_audit(
    Arena *a, BenchLegacyTruth *t, json_object *rec, const char *cache);

static void legacy_cached_knots(Arena *a, BenchLegacyTruth *t, const char *cache) {
    const char *path = fmt(a, "%s/axis.json", cache);
    json_object *axis = json_read(path);
    int lock = -1;
    if (!axis) {
        lock = lock_file(fmt(a, "%s/.axis.lock", cache), 1, 1);
        need(lock >= 0, "cannot lock axis cache for %s", t->control);
        axis = json_read(path);
    }
    if (axis) {
        json_object *knots = jget(axis, "knots");
        t->nknots = board_count(knots);
        t->knots = push(a, 3 * t->nknots * sizeof(*t->knots));
        for (int i = 0; i < t->nknots; i++) {
            json_object *point = json_object_array_get_idx(knots, i);
            for (int d = 0; d < 3; d++) {
                t->knots[3 * i + d] = json_object_get_double(json_object_array_get_idx(point, d));
            }
        }
        t->axis_source = fmt(a, "%s", jstr(axis, "source"));
        json_object_put(axis);
        if (lock >= 0) {
            close(lock);
        }
        return;
    }
    const char *group = legacy_group_path(a, t->control);
    if (!exists(group)) {
        group = legacy_group_path(a, fmt(a, "%s-tier-a", t->control));
    }
    if (exists(group)) {
        legacy_group_knots(a, t, group);
    } else {
        legacy_fit_axis(a, t);
    }
    json_object *knots = json_object_new_array_ext(t->nknots);
    for (int i = 0; i < t->nknots; i++) {
        json_object *row = json_object_new_array_ext(3);
        for (int d = 0; d < 3; d++) {
            json_object_array_add(row, json_object_new_double(t->knots[3 * i + d]));
        }
        json_object_array_add(knots, row);
    }
    axis = json_object_new_object();
    json_object_object_add(axis, "knots", knots);
    json_object_object_add(axis, "source", json_object_new_string(t->axis_source));
    json_write(path, axis);
    json_object_put(axis);
    close(lock);
}

static BenchLegacyTruth legacy_truth(Arena *a, const char *control, const char *source) {
    const int groups = !strcmp(source, "groups");
    const char *path = groups ? legacy_group_path(a, control) : legacy_names_path(a, control);
    need(exists(path), "no %s truth for %s", source, control);
    const char *rec_path = fmt(a, "%s/controls/sheets/%s/record.json", LIBRARY, control);
    json_object *rec = json_read(rec_path);
    need(rec != NULL, "no control record at %s", rec_path);
    Array mesh = npz_get(a, path, "mesh");
    i64 n = elements(mesh);
    BenchLegacyTruth t = {.control = control,
        .source = source,
        .sense = 1,
        .n = n,
        .voxel = jf64(rec, "scan_voxel_um"),
        .names = legacy_names(a, path)};
    t.meshes = board_count(t.names);
    need(t.meshes && t.voxel > 0, "invalid legacy truth in %s", path);
    t.mesh = push(a, n * sizeof(*t.mesh));
    t.wrap = push(a, t.meshes * sizeof(*t.wrap));
    i64 *counts = pushz(a, t.meshes * sizeof(*counts));
    t.mesh_count = counts;
    int named = 0;
    for (int i = 0; i < t.meshes; i++) {
        t.wrap[i] = legacy_wrap(board_item(t.names, i));
        named += t.wrap[i] >= 0;
    }
    for (i64 i = 0; i < n; i++) {
        t.mesh[i] = bench_number(mesh, i);
        need(t.mesh[i] >= 0 && t.mesh[i] < t.meshes, "invalid truth mesh index in %s", path);
        counts[t.mesh[i]]++;
    }
    t.spacing = legacy_spacing(a, rec, &t, groups ? 150 : 100);
    i32 *frames = push(a, n * sizeof(*frames));
    t.frame = frames;
    if (groups) {
        const char *keys[] = {"x", "y", "z"}, *mask_path = "unresolved";
        t.points = push(a, 3 * n * sizeof(*t.points));
        t.winding = push(a, n * sizeof(*t.winding));
        for (int d = 0; d < 3; d++) {
            size_t saved = a->used;
            Array axis = npz_get(a, path, keys[d]);
            need(elements(axis) == n, "invalid legacy truth coordinate count in %s", path);
            for (i64 i = 0; i < n; i++) {
                t.points[3 * i + d] = bench_number(axis, i);
            }
            a->used = saved;
        }
        const Array wind = npz_get(a, path, "Wg"), frame = npz_get(a, path, "group");
        const Array mask = npz_get(a, path, mask_path);
        need(elements(wind) == n && elements(frame) == n && elements(mask) == n,
            "legacy truth arrays differ in %s", path);
        t.unresolved = mask.data;
        for (i64 i = 0; i < n; i++) {
            t.winding[i] = bench_number(wind, i);
            frames[i] = bench_number(frame, i);
        }
        legacy_group_knots(a, &t, path);
    } else {
        const char *cache = fmt(a, "%s/bench/truth/%s", OUTPUTS, control);
        double scale = exp2(ji64(rec, "level"));
        Array xyz = bench_array(a, path, "points", "<f4", n * 3);
        t.points = xyz.data;
        t.normals = bench_array(a, path, "normals", "<f4", n * 3).data;
        for (i64 i = 0; i < n * 3; i++) {
            t.points[i] *= (f32)scale;
        }
        legacy_cached_knots(a, &t, cache);
        t.winding = push(a, n * sizeof(*t.winding));
        u8 *duplicate = NULL;
        if (named) {
            const char *winding_path = fmt(a, "%s/names-winding.npy", cache);
            const char *duplicate_path = fmt(a, "%s/names-duplicate.npy", cache);
            const char *audit_path = fmt(a, "%s/names.json", cache);
            t.audit = json_read(audit_path);
            if (!t.audit || board_integer(t.audit, "version") != 2 || !exists(winding_path) ||
                !exists(duplicate_path)) {
                json_object_put(t.audit);
                int lock = lock_file(fmt(a, "%s/.names.lock", cache), 1, 1);
                need(lock >= 0, "cannot lock named winding cache for %s", control);
                t.audit = json_read(audit_path);
                if (!t.audit || board_integer(t.audit, "version") != 2 || !exists(winding_path) ||
                    !exists(duplicate_path)) {
                    json_object_put(t.audit);
                    t.audit = legacy_name_audit(a, &t, rec, cache);
                }
                close(lock);
            }
            t.sense = board_integer(t.audit, "sense");
            Array supplied = npy_read(a, winding_path);
            Array added = npy_read(a, duplicate_path);
            need(elements(supplied) == n && elements(added) == n,
                "legacy winding cache does not match %s", control);
            duplicate = added.data;
            for (i64 i = 0; i < n; i++) {
                t.winding[i] = bench_number(supplied, i);
            }
        } else {
            for (i64 i = 0; i < n; i++) {
                t.winding[i] = NAN;
            }
        }
        i64 lines = 0;
        for (i64 i = 0; i < n; i++) {
            if (duplicate && duplicate[i]) {
                continue;
            }
            t.mesh[lines] = t.mesh[i];
            t.winding[lines] = t.winding[i];
            frames[lines] = isfinite(t.winding[i]) ? 0 : -1;
            for (int d = 0; d < 3; d++) {
                t.points[3 * lines + d] = t.points[3 * i + d];
                t.normals[3 * lines + d] = t.normals[3 * i + d];
            }
            lines++;
        }
        t.n = lines;
        if (t.audit) {
            json_object_object_add(t.audit, "duplicates_dropped", json_object_new_int64(n - lines));
        }
    }
    json_object_put(rec);
    return t;
}

static f32 *legacy_group_normals(Arena *a, const BenchLegacyTruth *t) {
    f32 *out = push(a, t->n * 3 * sizeof(*out));
    size_t saved = a->used;
    const char *path = legacy_group_path(a, t->control);
    Array rows = npz_get(a, path, "row"), cols = npz_get(a, path, "col");
    need(elements(rows) == t->n && elements(cols) == t->n,
        "legacy truth grid does not match coordinates");
    i32 *row = push(a, t->n * sizeof(*row)), *col = push(a, t->n * sizeof(*col));
    i64 *begins = push(a, (t->meshes + 1LL) * sizeof(*begins));
    i64 *cursor = push(a, t->meshes * sizeof(*cursor));
    i32 *order = push(a, t->n * sizeof(*order));
    begins[0] = 0;
    for (int m = 0; m < t->meshes; m++) {
        cursor[m] = begins[m];
        begins[m + 1] = begins[m] + t->mesh_count[m];
    }
    for (i64 i = 0; i < t->n; i++) {
        row[i] = bench_number(rows, i);
        col[i] = bench_number(cols, i);
        order[cursor[t->mesh[i]]++] = i;
    }
    size_t scratch = a->used;
    for (int m = 0; m < t->meshes; m++) {
        if (begins[m] == begins[m + 1]) {
            continue;
        }
        int r0 = INT32_MAX, c0 = INT32_MAX, r1 = 0, c1 = 0;
        for (i64 j = begins[m]; j < begins[m + 1]; j++) {
            i32 i = order[j], r = row[i], c = col[i];
            r0 = r < r0 ? r : r0;
            c0 = c < c0 ? c : c0;
            r1 = r > r1 ? r : r1;
            c1 = c > c1 ? c : c1;
        }
        i64 w = c1 - c0 + 3LL, h = r1 - r0 + 3LL;
        i32 *grid = push(a, h * w * sizeof(*grid));
        memset(grid, 0xff, h * w * sizeof(*grid));
        for (i64 j = begins[m]; j < begins[m + 1]; j++) {
            i32 i = order[j];
            grid[(row[i] - r0 + 1) * w + col[i] - c0 + 1] = i;
        }
        for (i64 j = begins[m]; j < begins[m + 1]; j++) {
            i32 i = order[j];
            i64 cell = (row[i] - r0 + 1) * w + col[i] - c0 + 1;
            i32 near[] = {grid[cell - w], grid[cell + w], grid[cell - 1], grid[cell + 1]};
            double delta[2][3], normal[3];
            for (int d = 0; d < 2; d++) {
                i32 u = near[2 * d], v = near[2 * d + 1];
                for (int k = 0; k < 3; k++) {
                    delta[d][k] = u < 0 && v < 0 ? NAN
                                                 : (double)t->points[3LL * (v < 0 ? i : v) + k] -
                            t->points[3LL * (u < 0 ? i : u) + k];
                }
            }
            bench_cross3(delta[0], delta[1], normal);
            double scale = bench_norm3(normal);
            for (int d = 0; d < 3; d++) {
                out[3LL * i + d] = scale > 0 ? normal[d] / scale : NAN;
            }
        }
        a->used = scratch;
    }
    a->used = saved;
    return out;
}

static const char *legacy_default(Arena *a, const char *control) {
    int groups = exists(legacy_group_path(a, control));
    const char *path = legacy_names_path(a, control);
    if (!exists(path)) {
        return groups ? "groups" : NULL;
    }
    if (!groups) {
        return "names";
    }
    json_object *names = legacy_names(a, path);
    int named = 0;
    for (int i = 0; i < board_count(names) && !named; i++) {
        named = legacy_wrap(board_item(names, i)) >= 0;
    }
    json_object_put(names);
    return named ? "names" : "groups";
}

static void legacy_subset(BenchLegacyTruth *t, int every, const double (*box)[3]) {
    if (every == 1 && !box) {
        return;
    }
    i64 kept = 0, seen = 0;
    i32 *frame = (i32 *)t->frame;
    for (i64 i = 0; i < t->n; i++) {
        f32 *p = t->points + 3 * i;
        if (box &&
            (p[0] < box[0][0] || p[0] >= box[1][0] || p[1] < box[0][1] || p[1] >= box[1][1] ||
                p[2] < box[0][2] || p[2] >= box[1][2])) {
            continue;
        }
        if (seen++ % every) {
            continue;
        }
        memmove(t->points + 3 * kept, p, 3 * sizeof(*p));
        if (t->normals) {
            memmove(t->normals + 3 * kept, t->normals + 3 * i, 3 * sizeof(*p));
        }
        t->mesh[kept] = t->mesh[i];
        t->winding[kept] = t->winding[i];
        frame[kept] = frame[i];
        if (t->unresolved) {
            t->unresolved[kept] = t->unresolved[i];
        }
        kept++;
    }
    t->n = kept;
    t->spacing *= sqrt(every);
}

static void legacy_truth_close(BenchLegacyTruth *t) {
    json_object_put(t->names);
    json_object_put(t->audit);
}

static int legacy_frames_order(const void *a, const void *b) {
    i32 x = *(const i32 *)a, y = *(const i32 *)b;
    return (x > y) - (x < y);
}

static json_object *legacy_summary(Arena *a, json_object *manifest, const BenchLegacyTruth *t) {
    json_object *report = json_object_new_object();
    const char *keys[] = {"control", "split", "source", "axis"};
    const char *values[] = {
        t->control, bench_split(a, manifest, t->control), t->source, t->axis_source};
    for (int i = 0; i < 4; i++) {
        json_object_object_add(report, keys[i], json_object_new_string(values[i]));
    }
    json_object_object_add(report, "available", legacy_available(a, t->control));
    json_object_object_add(report, "vertices", json_object_new_int64(t->n));
    json_object_object_add(report, "meshes", json_object_new_int(t->meshes));
    int named = 0, frames = 0;
    i64 known = 0, unresolved = 0;
    double low = INFINITY, high = -INFINITY;
    i32 *frame = push(a, t->n * sizeof(*frame));
    for (int i = 0; i < t->meshes; i++) {
        named += t->wrap[i] >= 0;
    }
    i64 count = 0;
    for (i64 i = 0; i < t->n; i++) {
        double z = t->points[3 * i + 2];
        low = fmin(low, z);
        high = fmax(high, z);
        known += isfinite(t->winding[i]);
        unresolved += t->unresolved && t->unresolved[i] != 0;
        if (t->frame[i] >= 0) {
            frame[count++] = t->frame[i];
        }
    }
    qsort(frame, count, sizeof(*frame), legacy_frames_order);
    for (i64 i = 0; i < count; i++) {
        frames += !i || frame[i] != frame[i - 1];
    }
    json_object_object_add(report, "frames", json_object_new_int(frames));
    json_object_object_add(report, "named_meshes", json_object_new_int(named));
    bench_decimal(report, "with_winding", t->n ? known / (double)t->n : NAN, 4);
    json_object_object_add(report, "scan_voxel_um", jnum(t->voxel));
    bench_decimal(report, "spacing_um", t->spacing, 1);
    json_object *range = json_object_new_array();
    json_object_array_add(range, jnum(low));
    json_object_array_add(range, jnum(high));
    json_object_object_add(report, "z_range", range);
    if (t->audit) {
        const char *fields[] = {"fraction_arc", "consistency", "sense", "handedness",
            "handedness_votes", "handedness_source", "names_outward", "merged", "anchors",
            "missing_meshes", "matched", "duplicates_dropped"};
        for (size_t i = 0; i < sizeof(fields) / sizeof(*fields); i++) {
            json_object_object_add(
                report, fields[i], json_object_get(json_object_object_get(t->audit, fields[i])));
        }
    }
    if (t->unresolved) {
        bench_decimal(report, "unresolved", t->n ? unresolved / (double)t->n : NAN, 4);
    }
    return report;
}

typedef struct {
    const BenchLegacyTruth *names;
    BenchTree tree;
    double cutoff;
    i32 *nearest;
} BenchLegacyMatch;

static void legacy_bench_match(void *payload, i64 batch) {
    BenchLegacyMatch *m = payload;
    for (i64 i = batch * 128; i < (batch + 1) * 128 && i < m->names->n; i++) {
        m->nearest[i] = -1;
        if (!m->tree.count || !isfinite(m->names->winding[i])) {
            continue;
        }
        BenchCandidate near[BENCH_NEIGHBORS];
        int used = 0;
        bench_candidates(&m->tree, 0, m->names->points + 3 * i, m->cutoff, near, &used);
        if (used) {
            m->nearest[i] = near[0].index;
        }
    }
}

static json_object *legacy_agreement(
    Arena *a, const BenchLegacyTruth *name, const BenchLegacyTruth *group) {
    BenchLegacyMatch match = {.names = name,
        .tree = bench_tree(a, group->points, group->n),
        .cutoff = pow(40 / name->voxel, 2),
        .nearest = push(a, name->n * sizeof(i32))};
    parallel((name->n + 127) / 128, 4, legacy_bench_match, &match);
    BenchVote *frames = push(a, name->n * sizeof(*frames));
    BenchVote *meshes = push(a, name->n * sizeof(*meshes));
    i64 n = 0, known = 0;
    for (i64 i = 0; i < name->n; i++) {
        known += isfinite(name->winding[i]);
        i32 at = match.nearest[i];
        if (at >= 0) {
            i64 frame = group->frame[at];
            double value = nearbyint(name->winding[i] - name->sense * group->winding[at]);
            i64 offset = isfinite(value) ? (i64)value : INT64_MIN / 100000;
            frames[n] = (BenchVote){frame, offset};
            meshes[n++] = (BenchVote){name->mesh[i], offset * 100000 + frame};
        }
    }
    const char *fields[] = {"groups", "by_mesh"};
    json_object *out = json_object_new_object();
    i64 right = 0;
    for (int d = 0; d < 2; d++) {
        BenchVote *pairs = d ? meshes : frames;
        qsort(pairs, n, sizeof(*pairs), bench_vote_order);
        int count = 0;
        json_object *by = d ? json_object_new_object() : NULL;
        for (i64 j = 0; j < n;) {
            i64 key = pairs[j].key, start = j, most = 0;
            do {
                i64 value = pairs[j].turn, end = j + 1;
                while (end < n && pairs[end].key == key && pairs[end].turn == value) {
                    end++;
                }
                if (end - j > most) {
                    most = end - j;
                }
                j = end;
            } while (j < n && pairs[j].key == key);
            if (d) {
                const char *mesh = board_item(name->names, (int)key);
                const char *limit = mesh;
                for (int i = 0; *limit && i < 24; i++) {
                    bench_utf8(&limit);
                }
                bench_decimal(
                    by, fmt(a, "%.*s", (int)(limit - mesh), mesh), most / (double)(j - start), 4);
            } else {
                right += most;
                count++;
            }
        }
        json_object_object_add(out, fields[d], d ? by : json_object_new_int(count));
    }
    json_object_object_add(out, "control", json_object_new_string(name->control));
    json_object_object_add(out, "matched", json_object_new_int64(n));
    json_object_object_add(out, "named_with_winding", json_object_new_int64(known));
    bench_decimal(out, "share_on_group_offset", right / fmax(n, 1), 4);
    return out;
}

int bench_truth_command(int argc, char **argv) {
    Arena a = arena(32 * GB);
    const char *mode = "auto";
    int list = 0, agree = 0;
    json_object *controls = json_object_new_array();
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("calibrate.bench.truth [CONTROL ...] [--source auto|names|groups] "
                 "[--list|--agree]");
            json_object_put(controls);
            munmap(a.base, a.size);
            return 0;
        }
        if (!strcmp(argv[i], "--list")) {
            list = 1;
        } else if (!strcmp(argv[i], "--agree")) {
            agree = 1;
        } else if (!strcmp(argv[i], "--source")) {
            need(++i < argc, "--source needs auto, names or groups");
            mode = argv[i];
            need(!strcmp(mode, "auto") || !strcmp(mode, "names") || !strcmp(mode, "groups"),
                "invalid truth source %s", mode);
        } else {
            need(site_name(argv[i]), "invalid bench control %s", argv[i]);
            json_object_array_add(controls, json_object_new_string(argv[i]));
        }
    }
    if (!board_count(controls)) {
        glob_t g = {0};
        glob(fmt(&a, "%s/controls/sheets/*/record.json", LIBRARY), 0, NULL, &g);
        for (size_t i = 0; i < g.gl_pathc; i++) {
            const char *path = g.gl_pathv[i], *tail = bench_basename(path);
            const char *folder = fmt(&a, "%.*s", (int)(tail - path - 1), path);
            json_object_array_add(controls, json_object_new_string(bench_basename(folder)));
        }
        globfree(&g);
    }
    json_object *manifest = json_read(fmt(&a, "%s/calibrate/bench/controls.json", HOME));
    need(manifest != NULL, "cannot read bench controls");
    for (int i = 0; i < board_count(controls); i++) {
        size_t saved = a.used;
        const char *control = board_item(controls, i);
        if (list) {
            json_object *files = legacy_available(&a, control);
            printf("%-48s %-9s ", control, bench_split(&a, manifest, control));
            for (int k = 0; k < board_count(files); k++) {
                printf("%s%s", k ? "," : "", board_item(files, k));
            }
            puts("");
            json_object_put(files);
        } else if (agree) {
            BenchLegacyTruth names = legacy_truth(&a, control, "names");
            BenchLegacyTruth groups = legacy_truth(&a, control, "groups");
            json_object *out = legacy_agreement(&a, &names, &groups);
            puts(board_json(out));
            json_object_put(out);
            legacy_truth_close(&names);
            legacy_truth_close(&groups);
        } else {
            const char *source = !strcmp(mode, "auto") ? bench_truth_source(&a, control) : mode;
            BenchLegacyTruth t = legacy_truth(&a, control, source);
            json_object *out = legacy_summary(&a, manifest, &t);
            puts(board_json(out));
            json_object_put(out);
            legacy_truth_close(&t);
        }
        fflush(stdout);
        a.used = saved;
    }
    json_object_put(manifest);
    json_object_put(controls);
    munmap(a.base, a.size);
    return 0;
}

static void legacy_fit_axis(Arena *a, BenchLegacyTruth *t) {
    t->knots = push(a, 24 * sizeof(*t->knots));
    size_t saved = a->used;
    i32 *usable = push(a, t->n * sizeof(*usable)), *pick = push(a, t->n * sizeof(*pick));
    double *zs = push(a, t->n * sizeof(*zs));
    i64 n = 0;
    for (i64 i = 0; i < t->n; i++) {
        const f32 *normal = t->normals + 3 * i;
        if (hypotf(normal[0], normal[1]) > 0.7f) {
            zs[n] = t->points[3 * i + 2];
            usable[n++] = i;
        }
    }
    need(n >= 500, "%s has too few normals to fit its axis", t->control);
    qsort(zs, n, sizeof(*zs), bench_double_order);
    double edges[9];
    for (int i = 0; i < 9; i++) {
        double x = i * ((n - 1) / 8.0);
        i64 k = floor(x);
        edges[i] = zs[k] + (zs[(i64)ceil(x)] - zs[k]) * (x - k);
    }
    Pcg64 rng = pcg64(0);
    for (int slab = 0; slab < 8; slab++) {
        i64 size = 0;
        double lo = edges[slab], hi = edges[slab + 1];
        for (i64 j = 0; j < n; j++) {
            i32 i = usable[j];
            double z = t->points[3LL * i + 2];
            if (z >= lo && z <= hi) {
                pick[size++] = i;
            }
        }
        if (size < 500) {
            continue;
        }
        i64 count = size < 200000 ? size : 200000;
        package_choose(a, &rng, pick, size, count);
        size_t scratch = a->used;
        double *residual = push(a, count * sizeof(*residual));
        double *weight = push(a, count * sizeof(*weight));
        f32 *directions = push(a, count * 2 * sizeof(*directions));
        for (i64 i = 0; i < count; i++) {
            const f32 *normal = t->normals + 3LL * pick[i];
            f32 length = hypotf(normal[0], normal[1]);
            directions[2 * i] = normal[0] / length;
            directions[2 * i + 1] = normal[1] / length;
            weight[i] = 1;
        }
        double x = 0, y = 0;
        for (int round = 0; round < 12; round++) {
            double xx = 0, yy = 0, xy = 0, bx = 0, by = 0;
            for (i64 i = 0; i < count; i++) {
                const f32 *p = t->points + 3LL * pick[i];
                double nx = directions[2 * i], ny = directions[2 * i + 1], w = weight[i];
                double u = 1 - nx * nx, v = 1 - ny * ny, q = -nx * ny;
                xx += w * u;
                xy += w * q;
                yy += w * v;
                bx += w * u * p[0] + w * q * p[1];
                by += w * q * p[0] + w * v * p[1];
            }
            double determinant = xx * yy - xy * xy;
            need(determinant > 0, "cannot fit axis for %s", t->control);
            x = (yy * bx - xy * by) / determinant;
            y = (xx * by - xy * bx) / determinant;
            for (i64 i = 0; i < count; i++) {
                const f32 *p = t->points + 3LL * pick[i];
                residual[i] =
                    fabs((x - p[0]) * directions[2 * i + 1] - (y - p[1]) * directions[2 * i]);
            }
            double scale = fmax(2 * bench_median(residual, count, 0), 1);
            for (i64 i = 0; i < count; i++) {
                const f32 *p = t->points + 3LL * pick[i];
                double miss =
                    ((x - p[0]) * directions[2 * i + 1] - (y - p[1]) * directions[2 * i]) / scale;
                weight[i] = 1 / (1 + miss * miss);
            }
        }
        t->knots[3 * t->nknots] = (lo + hi) / 2;
        t->knots[3 * t->nknots + 1] = x;
        t->knots[3 * t->nknots + 2] = y;
        t->nknots++;
        a->used = scratch;
    }
    need(t->nknots > 0, "%s has no fitted axis knots", t->control);
    t->axis_source = "fitted to truth normals";
    a->used = saved;
}

typedef struct {
    double re, im;
    i64 n;
} LegacyDirection;

static void legacy_direction_add(LegacyDirection *d, double theta) {
    d->re += cos(theta);
    d->im += sin(theta);
    d->n++;
}

static i64 legacy_mesh_angles(Arena *a, const BenchLegacyTruth *t, const char *folder,
    const i32 *index, i64 first, i64 stop, double lo, double hi, double *angles) {
    size_t saved = a->used;
    BenchSurface s = bench_load_piece(a, folder, &(BenchOptions){.ignore_winding = 1});
    json_object *scale = jget(s.meta, "scale");
    int step = fmax(
        1, nearbyint(100 * json_object_get_double(json_object_array_get_idx(scale, 0)) / t->voxel));
    i64 w = (s.w + (i64)step - 1) / step, h = (s.h + (i64)step - 1) / step;
    f32 *points = push(a, w * h * 3 * sizeof(*points)), *part = push(a, w * h * 2 * sizeof(*part));
    i32 *row = push(a, w * h * sizeof(*row)), *col = push(a, w * h * sizeof(*col));
    LegacyDirection *rs = pushz(a, h * sizeof(*rs)), *cs = pushz(a, w * sizeof(*cs));
    i64 len = 0;
    for (i64 r = 0; r < h; r++) {
        for (i64 c = 0; c < w; c++) {
            i64 pos = r * step * s.w + c * step;
            const f32 *p = s.grid + 3 * pos;
            if (s.index[pos] < 0 || p[2] < lo || p[2] > hi) {
                continue;
            }
            double x = interpolated(p[2], t->knots, t->nknots, 1);
            double y = interpolated(p[2], t->knots, t->nknots, 2);
            f32 theta = atan2(p[1] - y, p[0] - x);
            f32 re = cosf(theta), im = sinf(theta);
            part[2 * len] = re;
            part[2 * len + 1] = im;
            cs[c].re += re;
            cs[c].im += im;
            cs[c].n++;
            rs[r].re += re;
            rs[r].im += im;
            rs[r].n++;
            row[len] = r;
            col[len] = c;
            memcpy(points + 3 * len++, p, 3 * sizeof(*points));
        }
    }
    double coherent[2] = {0};
    for (i64 c = 0; c < w; c++) {
        coherent[0] += hypot(cs[c].re, cs[c].im);
    }
    for (i64 r = 0; r < h; r++) {
        coherent[1] += hypot(rs[r].re, rs[r].im);
    }
    int use_row = coherent[1] > coherent[0];
    LegacyDirection *direction = use_row ? rs : cs;
    i64 count = use_row ? h : w;
    double *centre = push(a, count * sizeof(*centre));
    double previous = NAN, unwrapped = 0;
    for (i64 i = 0; i < count; i++) {
        if (!direction[i].n) {
            continue;
        }
        double theta = atan2(direction[i].im, direction[i].re);
        double change = theta - previous;
        if (isfinite(previous)) {
            unwrapped += change - 2 * M_PI * nearbyint(change / (2 * M_PI));
        } else {
            unwrapped = theta;
        }
        centre[i] = unwrapped;
        previous = theta;
    }
    double *values = push(a, len * sizeof(*values));
    for (i64 i = 0; i < len; i++) {
        double c = centre[use_row ? row[i] : col[i]];
        f32 x = cos(-c), y = sin(-c);
        f32 re = x * part[2 * i] - y * part[2 * i + 1];
        f32 im = y * part[2 * i] + x * part[2 * i + 1];
        values[i] = c + atan2f(im, re);
    }
    i64 matched = 0;
    if (len) {
        BenchTree tree = bench_tree(a, points, len);
        for (i64 j = first; j < stop; j++) {
            i32 i = index[j];
            const f32 *p = t->points + 3LL * i;
            BenchCandidate hit;
            int n = 0;
            bench_candidates_limit(&tree, 0, p, 2.25, &hit, &n, 1);
            if (n) {
                angles[i] = values[hit.index];
                matched++;
            }
        }
    }
    json_object_put(s.meta);
    a->used = saved;
    return matched;
}

typedef struct {
    i64 n;
    double x, y, xx, yy, xy;
} LegacyCorrelation;

static void legacy_correlation_add(LegacyCorrelation *r, double x, double y) {
    r->x += x;
    r->y += y;
    r->xx += x * x;
    r->yy += y * y;
    r->xy += x * y;
    r->n++;
}

static double legacy_correlation(LegacyCorrelation r) {
    double v = r.xx - r.x * r.x / r.n, w = r.yy - r.y * r.y / r.n;
    return (r.xy - r.x * r.y / r.n) / sqrt(v * w);
}

static int legacy_group_hand(Arena *a, const char *control) {
    size_t saved = a->used;
    BenchLegacyTruth t = legacy_truth(a, control, "groups");
    int max = -1;
    for (i64 i = 0; i < t.n; i += 4) {
        max = t.frame[i] > max ? t.frame[i] : max;
    }
    LegacyCorrelation *correlation = pushz(a, (max + 1LL) * sizeof(*correlation));
    for (i64 i = 0; i < t.n; i += 4) {
        const f32 *p = t.points + 3 * i;
        int frame = t.frame[i];
        if (frame < 0) {
            continue;
        }
        double x = p[0] - interpolated(p[2], t.knots, t.nknots, 1);
        double y = p[1] - interpolated(p[2], t.knots, t.nknots, 2);
        double layer = nearbyint(t.winding[i] - atan2(y, x) / (2 * M_PI));
        legacy_correlation_add(correlation + frame, layer, hypot(x, y));
    }
    double score = 0;
    for (int i = 0; i <= max; i++) {
        LegacyCorrelation r = correlation[i];
        if (r.n > 1000 && r.xx > r.x * r.x / r.n) {
            score += r.n * legacy_correlation(r);
        }
    }
    legacy_truth_close(&t);
    a->used = saved;
    return score >= 0 ? 1 : -1;
}

static double legacy_radius(const BenchLegacyTruth *t, i32 i, double *theta) {
    const f32 *p = t->points + 3LL * i;
    double x = p[0] - interpolated(p[2], t->knots, t->nknots, 1);
    double y = p[1] - interpolated(p[2], t->knots, t->nknots, 2);
    *theta = atan2(y, x);
    return hypot(x, y);
}

static double legacy_turn_median(const double *phi, const double *radius, const i32 *sample,
    i64 first, i64 stop, double lo, double hi, int rising, double *scratch, i64 *n) {
    *n = 0;
    for (i64 i = first; i < stop; i++) {
        double value = phi[sample[i]];
        if (rising ? value >= lo && value < hi : value > lo && value <= hi) {
            scratch[(*n)++] = radius[i];
        }
    }
    return bench_median(scratch, *n, NAN);
}

typedef struct {
    int mesh;
    i64 count;
} LegacyMerged;

static int legacy_merged_order(const void *a, const void *b) {
    const LegacyMerged *x = a, *y = b;
    return x->count != y->count ? x->count > y->count ? -1 : 1 : x->mesh - y->mesh;
}

static void legacy_anchor_meshes(Arena *a, BenchLegacyTruth *t, const i32 *order, const i64 *begin,
    const double *turns, json_object *anchors) {
    size_t saved = a->used;
    LegacyMerged *merged = push(a, t->meshes * sizeof(*merged));
    u8 *placed = pushz(a, t->meshes);
    int n = 0;
    for (int i = 0; i < t->meshes; i++) {
        if (turns[i] >= 1.6) {
            merged[n++] = (LegacyMerged){i, begin[i + 1] - begin[i]};
        } else if (isfinite(turns[i])) {
            placed[i] = 1;
        }
    }
    qsort(merged, n, sizeof(*merged), legacy_merged_order);
    double max_dist = 40 / t->voxel;
    for (int i = 0; i < n; i++) {
        int m = merged[i].mesh;
        size_t scratch = a->used;
        i64 len = 0, cap = 0, used = 0, seen = 0;
        for (int j = 0; j < t->meshes; j++) {
            if (placed[j]) {
                cap += begin[j + 1] - begin[j];
            }
        }
        f32 *points = push(a, cap * 3 * sizeof(*points));
        i32 *reference = push(a, cap * sizeof(*reference));
        for (i64 j = 0; j < t->n; j++) {
            if (placed[t->mesh[j]] && isfinite(t->winding[j])) {
                memcpy(points + 3 * len, t->points + 3 * j, 3 * sizeof(*points));
                reference[len++] = j;
            }
        }
        placed[m] = 1;
        i64 target = begin[m + 1] - begin[m];
        double *value = push(a, ((target + 3) / 4) * sizeof(*value));
        BenchTree tree = bench_tree(a, points, len);
        if (len) {
            for (i64 j = begin[m]; j < begin[m + 1]; j++) {
                i32 k = order[j];
                if (!isfinite(t->winding[k]) || seen++ % 4) {
                    continue;
                }
                BenchCandidate hit;
                int found = 0;
                bench_candidates_limit(
                    &tree, 0, t->points + 3LL * k, max_dist * max_dist, &hit, &found, 1);
                if (found) {
                    value[used++] = nearbyint(t->winding[reference[hit.index]] - t->winding[k]);
                }
            }
        }
        json_object *item = json_object_new_object();
        json_object_object_add(anchors, board_item(t->names, m), item);
        json_object_object_add(item, "pairs", json_object_new_int64(used));
        i64 sampled = (seen + 3) / 4;
        if (used < fmax(250, 0.01 * sampled)) {
            json_object_object_add(item, "offset", NULL);
            json_object_object_add(item, "agreement", NULL);
        } else {
            qsort(value, used, sizeof(*value), bench_double_order);
            i64 best = 0;
            double offset = 0;
            for (i64 start = 0, end; start < used; start = end) {
                for (end = start + 1; end < used && value[end] == value[start]; end++) {
                }
                if (end - start > best) {
                    best = end - start;
                    offset = value[start];
                }
            }
            json_object_object_add(item, "offset", json_object_new_int64(offset));
            package_ratio(item, "agreement", best, used);
            for (i64 j = begin[m]; j < begin[m + 1]; j++) {
                t->winding[order[j]] += offset;
            }
        }
        a->used = scratch;
    }
    a->used = saved;
}

static i64 legacy_mark_duplicates(Arena *a, const BenchLegacyTruth *t, u8 *duplicate) {
    size_t saved = a->used;
    f32 *points = push(a, t->n * 3 * sizeof(*points));
    i32 *index = push(a, t->n * sizeof(*index));
    i64 n = 0, repeats = 0;
    for (i64 i = 0; i < t->n; i++) {
        if (isfinite(t->winding[i])) {
            memcpy(points + 3 * n, t->points + 3 * i, 3 * sizeof(*points));
            index[n++] = i;
        }
    }
    BenchTree tree = bench_tree(a, points, n);
    double max = 40 / t->voxel;
    for (i64 i = 0; i < n; i++) {
        i32 j = index[i], mesh = t->mesh[j];
        BenchCandidate hit[8];
        int n = 0;
        bench_candidates_limit(&tree, 0, points + 3 * i, max * max, hit, &n, 8);
        for (int k = 0; k < n && !duplicate[j]; k++) {
            i32 other = index[hit[k].index], m = t->mesh[other];
            i64 a = t->mesh_count[m], b = t->mesh_count[mesh];
            if ((a > b || (a == b && m < mesh)) && fabs(t->winding[j] - t->winding[other]) < 0.5) {
                duplicate[j] = 1;
                repeats++;
            }
        }
    }
    a->used = saved;
    return repeats;
}

static json_object *legacy_name_consistency(
    Arena *a, const BenchLegacyTruth *t, const u8 *duplicate) {
    size_t saved = a->used;
    f32 *points = push(a, t->n * 3 * sizeof(*points));
    i32 *index = push(a, t->n * sizeof(*index)), *pick = push(a, t->n * sizeof(*pick));
    i64 n = 0, coincident = 0, same = 0, radial = 0, one = 0, integer = 0;
    for (i64 i = 0; i < t->n; i++) {
        if (!duplicate[i] && isfinite(t->winding[i]) && package_good_normal(t->normals + 3 * i)) {
            memcpy(points + 3 * n, t->points + 3 * i, 3 * sizeof(*points));
            pick[n] = n;
            index[n++] = i;
        }
    }
    BenchTree tree = bench_tree(a, points, n);
    i64 sampled = n < 50000 ? n : 50000;
    Pcg64 rng = pcg64(0);
    package_choose(a, &rng, pick, n, sampled);
    double reach = 400 / t->voxel, depth = 40 / t->voxel, spacing = t->spacing / t->voxel;
    for (i64 i = 0; i < sampled; i++) {
        i32 k = pick[i], j = index[k];
        BenchCandidate hit[48];
        int used = 0, outward = 0;
        const f32 *p = points + 3LL * k, *normal = t->normals + 3LL * j;
        bench_candidates_limit(&tree, 0, p, reach * reach, hit, &used, 48);
        for (int q = 0; q < used; q++) {
            i32 z = hit[q].index, other = index[z];
            double along = fabs(package_projection(p, points + 3LL * z, normal));
            double across = fmax(hit[q].squared - along * along, 0);
            double step = fabs(t->winding[other] - t->winding[j]);
            if (t->mesh[other] != t->mesh[j] && along <= depth && across <= spacing * spacing) {
                coincident++;
                same += step < 0.25;
            }
            if (!outward && along > 2 * depth && along >= 0.8 * sqrt(hit[q].squared)) {
                outward = 1;
                radial++;
                one += nearbyint(step) == 1;
                integer += fabs(step - nearbyint(step)) <= 0.15;
            }
        }
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "coincident_pairs", json_object_new_int64(coincident));
    package_ratio(out, "coincident_within_0.25", same, coincident);
    json_object_object_add(out, "radial_pairs", json_object_new_int64(radial));
    package_ratio(out, "radial_one", one, radial);
    package_ratio(out, "radial_within_0.15_of_whole", integer, radial);
    a->used = saved;
    return out;
}

static json_object *legacy_name_audit(
    Arena *a, BenchLegacyTruth *t, json_object *rec, const char *cache) {
    size_t saved = a->used;
    int meshes = t->meshes;
    i64 n = t->n;
    i64 *begin = pushz(a, (meshes + 1LL) * sizeof(*begin)),
        *cursor = push(a, meshes * sizeof(*cursor));
    i32 *order = push(a, n * sizeof(*order)), *sample = push(a, ((n + 3) / 4) * sizeof(*sample));
    double *phi = push(a, n * sizeof(*phi)), *scratch = push(a, n * sizeof(*scratch));
    double *turns = push(a, meshes * sizeof(*turns)), *shift = push(a, meshes * sizeof(*shift));
    double z0 = INFINITY, z1 = -INFINITY, margins = 2000 / t->voxel;
    for (int m = 0; m < meshes; m++) {
        cursor[m] = begin[m];
        begin[m + 1] = begin[m] + t->mesh_count[m];
        turns[m] = NAN;
        shift[m] = NAN;
    }
    for (i64 i = 0; i < n; i++) {
        order[cursor[t->mesh[i]]++] = i;
        double z = t->points[3 * i + 2];
        z0 = fmin(z0, z);
        z1 = fmax(z1, z);
        phi[i] = NAN;
        t->winding[i] = NAN;
    }
    json_object *out = json_object_new_object(), *missing = json_object_new_array(),
                *ranges = json_object_new_object(), *merged = json_object_new_array(),
                *shift_json = json_object_new_object(), *correction = json_object_new_object(),
                *anchors = json_object_new_object();
    const char *sample_name = jstr(rec, "sample"), *scan = jstr(rec, "volume");
    i64 matched = 0, total = 0;
    for (int m = 0; m < meshes; m++) {
        if (t->wrap[m] < 0 || begin[m] == begin[m + 1]) {
            continue;
        }
        const char *name = board_item(t->names, m);
        const char *pattern =
            fmt(a, "%s/published/open_data/%s/segments/%s/mesh/*-on-%s-*um.tifxyz", LIBRARY,
                sample_name, name, scan);
        glob_t folders = {0};
        glob(pattern, 0, NULL, &folders);
        i64 kept = 0;
        const char *folder = NULL;
        for (size_t j = 0; j < folders.gl_pathc; j++) {
            if (exists(fmt(a, "%s/z.tif", folders.gl_pathv[j]))) {
                folder = folders.gl_pathv[j];
            }
        }
        if (folder) {
            kept = legacy_mesh_angles(
                a, t, folder, order, begin[m], begin[m + 1], z0 - margins, z1 + margins, phi);
        } else {
            json_object_array_add(missing, json_object_new_string(name));
        }
        globfree(&folders);
        if (!folder) {
            continue;
        }
        total += begin[m + 1] - begin[m];
        matched += kept;
        double lo = INFINITY, hi = -INFINITY;
        for (i64 j = begin[m]; j < begin[m + 1]; j++) {
            lo = fmin(lo, phi[order[j]]);
            hi = fmax(hi, phi[order[j]]);
        }
        turns[m] = kept ? (hi - lo) / (2 * M_PI) : 0;
        if (turns[m] >= 1.6) {
            json_object_array_add(merged, json_object_new_string(name));
        }
        bench_decimal(ranges, name, turns[m], 3);
    }
    i64 known = 0;
    memset(cursor, 0, meshes * sizeof(*cursor));
    u8 *picked = pushz(a, n);
    for (i64 i = 0; i < n; i++) {
        if (isfinite(phi[i]) && !(known++ % 4)) {
            cursor[t->mesh[i]]++;
            picked[i] = 1;
        }
    }
    i64 count = (known + 3) / 4;
    double *radius = push(a, count * sizeof(*radius));
    i64 at = 0, hand_vote = 0;
    int votes = 0;
    LegacyCorrelation corr = {0};
    const double window = 0.05 * 2 * M_PI;
    for (int m = 0; m < meshes; m++) {
        i64 first = at;
        for (i64 j = begin[m]; j < begin[m + 1]; j++) {
            i32 i = order[j];
            if (picked[i]) {
                double angle;
                radius[at] = legacy_radius(t, i, &angle);
                sample[at++] = i;
            }
        }
        i64 size = at - first;
        if (!size) {
            continue;
        }
        for (i64 i = 0; i < size; i++) {
            scratch[i] = phi[sample[first + i]];
        }
        double low = package_quantile(scratch, size, 0.001);
        double high = package_quantile(scratch, size, 0.999);
        const double lo[] = {low, low + 2 * M_PI, high - 2 * M_PI - window, high - window};
        const double hi[] = {low + window, low + 2 * M_PI + window, high - 2 * M_PI, high};
        for (int i = 0; i < 4; i += 2) {
            i64 aa, bb;
            double x = legacy_turn_median(
                phi, radius, sample, first, at, lo[i], hi[i], i == 0, scratch, &aa);
            double y = legacy_turn_median(
                phi, radius, sample, first, at, lo[i + 1], hi[i + 1], i == 0, scratch, &bb);
            if (aa >= 50 && bb >= 50) {
                votes++;
                hand_vote += ((y > x) - (y < x)) * (aa < bb ? aa : bb);
            }
        }
        if (turns[m] < 1.6) {
            memcpy(scratch, radius + first, size * sizeof(*scratch));
            legacy_correlation_add(&corr, bench_median(scratch, size, NAN), t->wrap[m]);
        }
    }
    int hand = hand_vote >= 0 ? 1 : -1;
    const char *hand_source = votes ? "mesh overlaps past their start" : "default (no evidence)";
    if (!votes && exists(legacy_group_path(a, t->control))) {
        hand = legacy_group_hand(a, t->control);
        hand_source = "v1 wrap truth groups";
    }
    int outward = corr.n < 2 || legacy_correlation(corr) >= 0 ? 1 : -1;
    t->sense = hand * outward;
    double *fractions = push(a, meshes * sizeof(*fractions));
    int fractions_n = 0;
    for (int m = 0; m < meshes; m++) {
        i64 used = 0;
        for (i64 j = begin[m]; j < begin[m + 1]; j++) {
            double angle = phi[order[j]];
            if (isfinite(angle)) {
                scratch[used++] = t->sense * angle / (2 * M_PI);
            }
        }
        if (!used) {
            continue;
        }
        double start = package_quantile(scratch, used, 0.001);
        LegacyDirection unit = {0};
        for (i64 j = begin[m]; j < begin[m + 1]; j++) {
            i32 i = order[j];
            if (!isfinite(phi[i])) {
                continue;
            }
            double theta;
            legacy_radius(t, i, &theta);
            t->winding[i] = t->wrap[m] + t->sense * phi[i] / (2 * M_PI) - start;
            legacy_direction_add(&unit, 2 * M_PI * (t->winding[i] - t->sense * theta / (2 * M_PI)));
        }
        if (hypot(unit.re, unit.im) > 0) {
            double f = atan2(unit.im, unit.re) / (2 * M_PI);
            fractions[fractions_n++] = shift[m] = f - floor(f);
        }
    }
    qsort(fractions, fractions_n, sizeof(*fractions), bench_double_order);
    double arc = 0, best = -1, start = 0, top = 0;
    for (int j = 0; j < fractions_n; j++) {
        double next = j + 1 == fractions_n ? fractions[0] + 1 : fractions[j + 1];
        if (next - fractions[j] > best) {
            best = next - fractions[j];
            start = fractions[(j + 1) % fractions_n];
        }
    }
    if (fractions_n) {
        top = start;
        for (int m = 0; m < meshes; m++) {
            if (isfinite(shift[m])) {
                double f = shift[m] - start;
                shift[m] = start + (f - floor(f));
                top = fmax(top, shift[m]);
            }
        }
        arc = top - start;
    }
    for (int m = 0; m < meshes; m++) {
        if (!isfinite(shift[m])) {
            continue;
        }
        shift[m] = shift[m] - start - arc / 2;
        bench_decimal(shift_json, board_item(t->names, m), shift[m], 4);
        for (i64 j = begin[m]; j < begin[m + 1]; j++) {
            t->winding[order[j]] -= shift[m];
        }
    }
    legacy_anchor_meshes(a, t, order, begin, turns, anchors);
    if (!strcmp(t->control, "PHerc0139-20250728140407-rows-0007-0019") ||
        !strcmp(t->control, "PHerc0139-20250728140407-rows-0019-0031")) {
        for (int m = 0; m < meshes; m++) {
            if (t->wrap[m] >= 51 && t->wrap[m] <= 59) {
                json_object_object_add(
                    correction, board_item(t->names, m), json_object_new_int(-1));
                for (i64 j = begin[m]; j < begin[m + 1]; j++) {
                    t->winding[order[j]]--;
                }
            }
        }
    }
    u8 *duplicate = pushz(a, n);
    legacy_mark_duplicates(a, t, duplicate);
    json_object_object_add(out, "version", json_object_new_int(2));
    json_object_object_add(out, "fraction_shifts", shift_json);
    bench_decimal(out, "fraction_arc", arc, 4);
    json_object_object_add(out, "sense", json_object_new_int(t->sense));
    json_object_object_add(out, "handedness", json_object_new_int(hand));
    json_object_object_add(out, "handedness_votes", json_object_new_int(votes));
    json_object_object_add(out, "handedness_source", json_object_new_string(hand_source));
    json_object_object_add(out, "names_outward", json_object_new_int(outward));
    json_object_object_add(out, "turns", ranges);
    json_object_object_add(out, "merged", merged);
    json_object_object_add(out, "anchors", anchors);
    json_object_object_add(out, "name_corrections", correction);
    json_object_object_add(out, "missing_meshes", missing);
    package_ratio(out, "matched", matched, total ? total : 1);
    json_object_object_add(out, "consistency", legacy_name_consistency(a, t, duplicate));
    npy_write(fmt(a, "%s/names-winding.npy", cache),
        (Array){.descr = "<f8", .ndim = 1, .shape = {n}, .data = t->winding, .size = n * 8});
    npy_write(fmt(a, "%s/names-duplicate.npy", cache),
        (Array){.descr = "|b1", .ndim = 1, .shape = {n}, .data = duplicate, .size = n});
    json_write(fmt(a, "%s/names.json", cache), out);
    a->used = saved;
    return out;
}
