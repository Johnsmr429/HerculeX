#ifndef HX_UNITY
const char hx_usage[] = "Score predicted surfaces against a frozen truth control.";
#endif

#include "nearest.c"

enum {
    BENCH_DUPLICATE = 1,
    BENCH_PRESSED = 2,
    BENCH_OVERLAP = 4 | 8,
    BENCH_SKIP = 32 | 64,
    BENCH_TOUCHING = 128,
    BENCH_UNSCORED = 127,
};

static const char *const bench_zones[] = {
    "dev", "buffer", "holdout-A", "holdout-B", "train", "val"};

typedef struct {
    const f32 *points, *normals, *area;
    const double *winding, *knots;
    const i32 *frame;
    const i16 *mesh;
    const u8 *flags, *zone;
    i64 count;
    int sense, nknots;
    double voxel, spacing;
    json_object *record;
} BenchTruth;

typedef struct {
    const char *name;
    const f32 *grid, *winding;
    i32 *index;
    json_object *meta;
    double step, winding_floor, median_turn;
    i64 first, n;
    int h, w;
} BenchSurface;

typedef struct {
    BenchSurface *surfaces;
    int n;
    i64 count;
    f32 *xyz, *normals;
    double *turn, *area, *winding;
    i32 *surface, *vertex, *part;
    u8 *zone;
} BenchPieces;

typedef struct {
    const char *control, *label, *system, *folder;
    double low, high;
    int zones, locked, ignore_winding;
} BenchOptions;

static int bench_zone_of(const char *name) {
    for (int i = 0; i < 6; i++) {
        if (!strcmp(name, bench_zones[i])) {
            return i;
        }
    }
    return -1;
}

static int bench_in_zone(const BenchOptions *o, int zone, double z) {
    return zone >= 0 && zone < 6 && (o->zones & (1 << zone)) && z >= o->low && z < o->high;
}

static Array bench_array(
    Arena *a, const char *path, const char *name, const char *type, i64 expected) {
    Array x = npz_get(a, path, name);
    need(!strcmp(x.descr, type) && (expected < 0 || elements(x) == expected),
        "unexpected %s shape or type in %s", name, path);
    return x;
}

static double bench_number(Array x, i64 at) {
    if (!strcmp(x.descr, "<f8")) {
        return ((double *)x.data)[at];
    }
    if (!strcmp(x.descr, "<f4")) {
        return ((f32 *)x.data)[at];
    }
    if (!strcmp(x.descr, "<i8")) {
        return ((i64 *)x.data)[at];
    }
    if (!strcmp(x.descr, "<i4")) {
        return ((i32 *)x.data)[at];
    }
    if (!strcmp(x.descr, "<u4")) {
        return ((u32 *)x.data)[at];
    }
    if (!strcmp(x.descr, "<i2")) {
        return ((i16 *)x.data)[at];
    }
    if (!strcmp(x.descr, "<u2")) {
        return ((u16 *)x.data)[at];
    }
    if (!strcmp(x.descr, "|i1")) {
        return ((i8 *)x.data)[at];
    }
    need(!strcmp(x.descr, "|u1") || !strcmp(x.descr, "|b1"), "unknown bench type %s", x.descr);
    return ((u8 *)x.data)[at];
}

static BenchTruth bench_truth(Arena *a, const char *folder) {
    const char *path = fmt(a, "%s/truth.npz", folder);
    Array xyz = bench_array(a, path, "points", "<f4", -1);
    need(xyz.ndim == 2 && xyz.shape[1] == 3, "invalid truth points in %s", path);
    i64 n = xyz.shape[0];
    BenchTruth t = {.points = xyz.data,
        .count = n,
        .normals = bench_array(a, path, "normals", "<f4", n * 3).data,
        .winding = bench_array(a, path, "winding", "<f8", n).data,
        .area = bench_array(a, path, "area_mm2", "<f4", n).data,
        .frame = bench_array(a, path, "frame", "<i4", n).data,
        .mesh = bench_array(a, path, "mesh", "<i2", n).data,
        .flags = bench_array(a, path, "flags", "|u1", n).data,
        .zone = bench_array(a, path, "zone", "|u1", n).data,
        .voxel = bench_number(npz_get(a, path, "voxel_um"), 0),
        .spacing = bench_number(npz_get(a, path, "spacing_um"), 0),
        .sense = bench_number(npz_get(a, path, "sense"), 0)};
    Array knots = bench_array(a, path, "knots", "<f8", -1);
    need(knots.ndim == 2 && knots.shape[1] == 3 && knots.shape[0] > 0 && t.voxel > 0,
        "invalid truth spacing or axis in %s", path);
    t.knots = knots.data;
    t.nknots = knots.shape[0];
    t.record = json_read(fmt(a, "%s/control.json", folder));
    need(t.record != NULL, "no control record in %s", folder);
    return t;
}

static const char *bench_basename(const char *path) {
    const char *end = strrchr(path, '/');
    return end ? end + 1 : path;
}

static int bench_file(const char *path, const char *suffix) {
    size_t n = strlen(path), s = strlen(suffix);
    return n >= s && !strcmp(path + n - s, suffix);
}

static void bench_path_add(json_object *paths, const char *path) {
    for (int i = 0; i < board_count(paths); i++) {
        if (!strcmp(board_item(paths, i), path)) {
            return;
        }
    }
    json_object_array_add(paths, json_object_new_string(path));
}

static json_object *bench_paths(Arena *a, json_object *given) {
    json_object *paths = json_object_new_array();
    for (int i = 0; i < board_count(given); i++) {
        const char *path = board_item(given, i);
        if (bench_file(path, ".npz") || exists(fmt(a, "%s/x.tif", path))) {
            bench_path_add(paths, path);
            continue;
        }
        glob_t g = {0};
        glob(fmt(a, "%s/piece-*", path), 0, NULL, &g);
        int found = 0;
        for (size_t j = 0; j < g.gl_pathc; j++) {
            if (exists(fmt(a, "%s/x.tif", g.gl_pathv[j]))) {
                bench_path_add(paths, g.gl_pathv[j]);
                found++;
            }
        }
        globfree(&g);
        if (!found) {
            glob(fmt(a, "%s/*.npz", path), 0, NULL, &g);
            for (size_t j = 0; j < g.gl_pathc; j++) {
                bench_path_add(paths, g.gl_pathv[j]);
            }
            globfree(&g);
        }
    }
    for (int i = 0; i < board_count(given); i++) {
        glob_t g = {0};
        glob(fmt(a, "%s/*.tifxyz", board_item(given, i)), 0, NULL, &g);
        for (size_t j = 0; j < g.gl_pathc; j++) {
            if (exists(fmt(a, "%s/x.tif", g.gl_pathv[j]))) {
                bench_path_add(paths, g.gl_pathv[j]);
            }
        }
        globfree(&g);
    }
    return paths;
}

static BenchSurface bench_load_piece(Arena *a, const char *path, const BenchOptions *o) {
    const char *keys[] = {"x", "y", "z"};
    const char *name = bench_basename(path);
    BenchSurface s = {.name = name, .step = 1};
    Array mask = {0}, quality = {0}, data[3] = {{0}};
    if (bench_file(path, ".npz")) {
        s.name = fmt(a, "%.*s", (int)(strlen(name) - 4), name);
        zip_t *archive = zip_open(path, ZIP_RDONLY, NULL);
        need(archive != NULL, "cannot open %s", path);
        Array xyz = npz_member(a, archive, path, "xyz", 0);
        if (xyz.data) {
            need(xyz.ndim == 3 && xyz.shape[2] == 3, "invalid xyz shape in %s", path);
            s.h = xyz.shape[0];
            s.w = xyz.shape[1];
            i64 n = (i64)s.h * s.w * 3;
            if (!strcmp(xyz.descr, "<f4")) {
                s.grid = xyz.data;
            } else {
                f32 *flat = push(a, n * sizeof(*flat));
                for (i64 j = 0; j < n; j++) {
                    flat[j] = bench_number(xyz, j);
                }
                s.grid = flat;
            }
        } else {
            for (int k = 0; k < 3; k++) {
                data[k] = npz_member(a, archive, path, keys[k], 1);
            }
            s.h = data[0].shape[0];
            s.w = data[0].shape[1];
        }
        mask = npz_member(a, archive, path, "mask", 0);
        zip_close(archive);
    } else {
        for (int k = 0; k < 3; k++) {
            int w, h;
            f32 *value = tiff_read(a, fmt(a, "%s/%s.tif", path, keys[k]), &w, &h);
            need(!k || (w == s.w && h == s.h), "surface coordinate sizes differ in %s", path);
            s.w = w;
            s.h = h;
            data[k] = (Array){.descr = "<f4", .ndim = 2, .shape = {h, w}, .data = value};
        }
        const char *q = fmt(a, "%s/quality.npy", path), *wind = fmt(a, "%s/w.tif", path);
        if (exists(q)) {
            quality = npy_read(a, q);
        }
        if (!o->ignore_winding && exists(wind)) {
            int h, w;
            s.winding = tiff_read(a, wind, &w, &h);
            need(w == s.w && h == s.h, "surface winding has a different shape in %s", path);
        }
        s.meta = json_read(fmt(a, "%s/meta.json", path));
        if (s.meta && o->ignore_winding) {
            json_object_object_del(s.meta, "winding");
            json_object_object_del(s.meta, "chirality");
            json_object_object_del(s.meta, "cut_deg");
        }
    }
    i64 n = (i64)s.h * s.w;
    need(n > 0 && n <= INT32_MAX, "invalid bench surface size in %s", path);
    if (!s.grid) {
        f32 *grid = push(a, 3 * n * sizeof(*grid));
        for (int k = 0; k < 3; k++) {
            need(data[k].ndim == 2 && data[k].shape[0] == s.h && data[k].shape[1] == s.w,
                "surface coordinate sizes differ in %s", path);
            for (i64 j = 0; j < n; j++) {
                grid[3 * j + k] = bench_number(data[k], j);
            }
        }
        s.grid = grid;
    }
    need((!mask.data || elements(mask) == n) && (!quality.data || elements(quality) == n),
        "surface mask or quality does not match %s", path);
    s.index = push(a, n * sizeof(*s.index));
    for (i64 i = 0; i < n; i++) {
        const f32 *xyz = s.grid + 3 * i;
        int valid = bench_finite(xyz) && xyz[0] > 0 && xyz[1] > 0 && xyz[2] > 0;
        if (valid && mask.data) {
            valid &= bench_number(mask, i) != 0;
        }
        if (valid && quality.data) {
            double q = bench_number(quality, i);
            valid &= o->locked ? q == 1 : q > 0;
        }
        s.index[i] = valid ? s.n++ : -1;
    }
    return s;
}

static int bench_double_order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double bench_median(double *values, i64 n, double fallback) {
    if (!n) {
        return fallback;
    }
    qsort(values, n, sizeof(*values), bench_double_order);
    return (values[(n - 1) / 2] + values[n / 2]) * 0.5;
}

static double bench_step(const f32 *x, const f32 *y) {
    f32 dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static void bench_cross3(const double a[3], const double b[3], double n[3]) {
    for (int d = 0; d < 3; d++) {
        int p = (d + 1) % 3, q = (d + 2) % 3;
        n[d] = a[p] * b[q] - a[q] * b[p];
    }
}

static double bench_norm3(const double n[3]) {
    return sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

static void bench_piece_geometry(Arena *a, BenchPieces *p, BenchTruth *t, int index) {
    BenchSurface *s = p->surfaces + index;
    int h = s->h, w = s->w;
    i64 size = (i64)h * w, base = s->first;
    size_t saved = a->used;
    double *angles = push(a, s->n * sizeof(*angles));
    double *steps = push(a, s->n * sizeof(*steps));
    i32 *queue = push(a, s->n * sizeof(*queue));
    i64 count = 0;
    json_object *zones = jget(jget(t->record, "scan"), "zones");
    for (i64 i = 0; i < size; i++) {
        i32 at = s->index[i];
        if (at < 0) {
            continue;
        }
        i64 v = base + at;
        const f32 *xyz = s->grid + 3 * i;
        memcpy(p->xyz + 3 * v, xyz, 3 * sizeof(*xyz));
        p->surface[v] = index;
        p->vertex[v] = at;
        p->part[v] = -1;
        p->zone[v] = 255;
        p->winding[v] = s->winding ? s->winding[i] : NAN;
        for (int z = 0; z < board_count(zones); z++) {
            json_object *band = json_object_array_get_idx(zones, z);
            double low = json_object_get_double(json_object_array_get_idx(band, 0));
            double high = json_object_get_double(json_object_array_get_idx(band, 1));
            if (xyz[2] >= low && xyz[2] < high) {
                p->zone[v] = bench_zone_of(board_item(band, 2));
            }
        }
        double centre_x = interpolated(xyz[2], t->knots, t->nknots, 1);
        double centre_y = interpolated(xyz[2], t->knots, t->nknots, 2);
        angles[at] = atan2(xyz[1] - centre_y, xyz[0] - centre_x);
        int y = i / w, x = i % w;
        i64 near[] = {y ? i - w : i, y + 1 < h ? i + w : i, x ? i - 1 : i, x + 1 < w ? i + 1 : i};
        for (int k = 0; k < 4; k++) {
            if (s->index[near[k]] < 0) {
                near[k] = i;
            }
        }
        double tangent[2][3], n[3];
        for (int d = 0; d < 2; d++) {
            for (int k = 0; k < 3; k++) {
                tangent[d][k] = near[2 * d] == i && near[2 * d + 1] == i
                    ? NAN
                    : (double)s->grid[3 * near[2 * d + 1] + k] - s->grid[3 * near[2 * d] + k];
            }
        }
        bench_cross3(tangent[0], tangent[1], n);
        double length = bench_norm3(n);
        for (int d = 0; d < 3; d++) {
            p->normals[3 * v + d] = length > 0 ? n[d] / length : NAN;
        }
        if (near[3] != i) {
            steps[count++] = bench_step(xyz, s->grid + 3 * near[3]);
        }
        if (y + 1 < h && x + 1 < w && s->index[i + 1] >= 0 && s->index[i + w] >= 0 &&
            s->index[i + w + 1] >= 0) {
            double aa[3], bb[3], cross[3];
            for (int d = 0; d < 3; d++) {
                aa[d] = (double)s->grid[3 * (i + w + 1) + d] - xyz[d];
                bb[d] = (double)s->grid[3 * (i + w) + d] - s->grid[3 * (i + 1) + d];
            }
            bench_cross3(aa, bb, cross);
            double value = bench_norm3(cross) * 0.125 * pow(t->voxel * 1e-4, 2);
            p->area[v] += value;
            p->area[base + s->index[i + 1]] += value;
            p->area[base + s->index[i + w]] += value;
            p->area[base + s->index[i + w + 1]] += value;
        }
    }
    s->step = bench_median(steps, count, 1);
    int part = 0;
    for (i64 i = 0; i < size; i++) {
        int at = s->index[i];
        if (at < 0 || p->part[base + at] >= 0) {
            continue;
        }
        i64 n = 1;
        queue[0] = i;
        p->part[base + at] = part;
        p->turn[base + at] = angles[at];
        for (i64 j = 0; j < n; j++) {
            i64 pos = queue[j], current = base + s->index[pos];
            int y = pos / w, x = pos % w;
            i64 adjacent[] = {x + 1 < w ? pos + 1 : pos, y + 1 < h ? pos + w : pos,
                y ? pos - w : pos, x ? pos - 1 : pos};
            for (int k = 0; k < 4; k++) {
                i64 row = adjacent[k];
                int next = s->index[row];
                if (next < 0 || p->part[base + next] >= 0) {
                    continue;
                }
                p->part[base + next] = part;
                double delta = angles[next] - angles[s->index[pos]];
                p->turn[base + next] = p->turn[current] + atan2(sin(delta), cos(delta));
                queue[n++] = row;
            }
        }
        part++;
    }
    memcpy(steps, p->turn + base, s->n * sizeof(*steps));
    s->median_turn = bench_median(steps, s->n, 0);
    a->used = saved;
}

static BenchPieces bench_pieces(
    Arena *a, json_object *paths, BenchTruth *t, const BenchOptions *o) {
    BenchPieces p = {.surfaces = push(a, board_count(paths) * sizeof(BenchSurface))};
    for (int i = 0; i < board_count(paths); i++) {
        size_t saved = a->used;
        BenchSurface s = bench_load_piece(a, board_item(paths, i), o);
        if (s.n < 4) {
            json_object_put(s.meta);
            a->used = saved;
            continue;
        }
        s.first = p.count;
        p.count += s.n;
        p.surfaces[p.n++] = s;
    }
    need(p.count > 0 && p.count <= INT32_MAX, "no surface vertices or too many vertices");
    i64 n = p.count;
    p.xyz = push(a, 3 * n * sizeof(*p.xyz));
    p.normals = push(a, 3 * n * sizeof(*p.normals));
    p.area = pushz(a, n * sizeof(*p.area));
    p.turn = push(a, n * sizeof(*p.turn));
    p.winding = push(a, n * sizeof(*p.winding));
    p.surface = push(a, n * sizeof(*p.surface));
    p.vertex = push(a, n * sizeof(*p.vertex));
    p.part = push(a, n * sizeof(*p.part));
    p.zone = push(a, n * sizeof(*p.zone));
    for (int i = 0; i < p.n; i++) {
        bench_piece_geometry(a, &p, t, i);
    }
    return p;
}

typedef struct {
    i64 key, turn;
} BenchVote;

typedef struct {
    BenchVote *vote;
    i64 n;
} BenchModes;

typedef struct {
    int global, chirality;
    double cut, fraction, share;
} BenchAlignment;

typedef struct {
    const BenchMatches *matches;
    const u8 *eligible;
    const double *winding;
    const i32 *frame;
    i64 count;
    int surface_query;
} BenchPairs;

typedef struct {
    i64 vertex;
    double winding;
    int frame;
} BenchPair;

static BenchPair bench_pair(BenchPairs p, i64 row, int k) {
    i64 matched = p.matches[row].index[k];
    i64 truth = p.surface_query ? matched : row;
    return (BenchPair){p.surface_query ? row : matched, p.winding[truth], p.frame[truth]};
}

static double bench_phase(BenchPieces *s, BenchPairs p, int first, int chirality) {
    double re = 0, im = 0;
    for (i64 i = 0; i < p.count; i++) {
        int mask = p.eligible[i] & (first ? 1 : 7);
        for (int k = 0; mask && k < BENCH_KEEP; k++, mask >>= 1) {
            if (mask & 1) {
                BenchPair pair = bench_pair(p, i, k);
                double phase = 2 * M_PI * (pair.winding - chirality * s->winding[pair.vertex]);
                re += cos(phase);
                im += sin(phase);
            }
        }
    }
    return atan2(im, re) / (2 * M_PI);
}

static i64 bench_offset(BenchPieces *s, BenchPair p, double phase, int chirality) {
    double raw = nearbyint(p.winding - chirality * s->winding[p.vertex] - phase);
    return isfinite(raw) && raw > (double)INT64_MIN && raw < (double)INT64_MAX ? (i64)raw
                                                                               : INT64_MIN;
}

static i64 bench_frame(BenchPieces *s, BenchPair p, int global) {
    return global ? p.frame
                  : ((i64)p.frame * s->n + s->surface[p.vertex]) * 100000 + s->part[p.vertex];
}

static int bench_vote_order(const void *a, const void *b) {
    const BenchVote *x = a, *y = b;
    return x->key != y->key ? (x->key > y->key) - (x->key < y->key)
                            : (x->turn > y->turn) - (x->turn < y->turn);
}

static i64 bench_modes(BenchPieces *s, BenchPairs p, BenchAlignment align, BenchModes *m) {
    i64 count = 0, right = 0;
    for (i64 i = 0; i < p.count; i++) {
        if (p.eligible[i] & 1) {
            BenchPair pair = bench_pair(p, i, 0);
            m->vote[count++] = (BenchVote){bench_frame(s, pair, align.global),
                bench_offset(s, pair, align.fraction, align.chirality)};
        }
    }
    qsort(m->vote, count, sizeof(*m->vote), bench_vote_order);
    m->n = 0;
    for (i64 i = 0; i < count;) {
        i64 wanted = m->vote[i].key, best = m->vote[i].turn, most = 0;
        do {
            i64 turn = m->vote[i].turn, at = i + 1;
            while (at < count && m->vote[at].key == wanted && m->vote[at].turn == turn) {
                at++;
            }
            if (at - i > most) {
                most = at - i;
                best = turn;
            }
            i = at;
        } while (i < count && m->vote[i].key == wanted);
        m->vote[m->n++] = (BenchVote){wanted, best};
        right += most;
    }
    return right;
}

static int bench_mode(const BenchModes *m, i64 key, i64 *turn) {
    i64 lo = 0, hi = m->n;
    while (lo < hi) {
        i64 mid = lo + (hi - lo) / 2;
        if (m->vote[mid].key < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == m->n || m->vote[lo].key != key) {
        return 0;
    }
    *turn = m->vote[lo].turn;
    return 1;
}

static void bench_winding(BenchPieces *p, int sense, BenchAlignment align) {
    for (int i = 0; i < p->n; i++) {
        BenchSurface *s = &p->surfaces[i];
        double *values = p->winding + s->first, *turn = p->turn + s->first;
        if (align.global && s->winding) {
            for (i64 k = 0, j = 0; k < (i64)s->h * s->w; k++) {
                if (s->index[k] >= 0) {
                    values[j++] = s->winding[k];
                }
            }
            continue;
        }
        double scale = sense * (align.global ? align.chirality : 1) / (2 * M_PI);
        double base =
            align.global ? s->winding_floor - floor((s->median_turn - align.cut) * scale) : 0;
        for (i64 j = 0; j < s->n; j++) {
            values[j] = base + (turn[j] - align.cut) * scale;
        }
    }
}

static BenchAlignment bench_alignment(
    BenchPieces *p, BenchTruth *t, BenchPairs pairs, BenchModes *modes) {
    int global = 1, every_grid = 1;
    for (int i = 0; i < p->n; i++) {
        BenchSurface *s = p->surfaces + i;
        json_object *phase = json_object_object_get(s->meta, "winding");
        global &= s->winding || phase;
        every_grid &= s->winding != NULL;
        s->winding_floor = json_object_get_double(phase);
    }
    BenchAlignment best = {.chirality = 1};
    if (!global) {
        bench_winding(p, t->sense, best);
        double phase = bench_phase(p, pairs, 1, 1);
        bench_modes(p, pairs, (BenchAlignment){.chirality = 1, .fraction = phase}, modes);
        return best;
    }
    json_object *given = p->surfaces[0].meta;
    json_object *cut = json_object_object_get(given, "cut_deg");
    json_object *chirality = json_object_object_get(given, "chirality");
    int n = 0;
    for (i64 i = 0; i < pairs.count; i++) {
        n += !!(pairs.eligible[i] & 1);
    }
    i64 greatest = -1;
    for (int which = 0; which < (chirality ? 1 : 2); which++) {
        int sign = chirality ? json_object_get_int(chirality) : which ? -1 : 1;
        for (int angle = 0; angle < (cut || every_grid ? 1 : 360); angle += 5) {
            BenchAlignment next = {.global = 1,
                .chirality = sign,
                .cut = (cut ? json_object_get_double(cut) : angle) * (M_PI / 180)};
            bench_winding(p, t->sense, next);
            next.fraction = n ? bench_phase(p, pairs, 1, sign) : NAN;
            i64 best_right = bench_modes(p, pairs, next, modes);
            if (best_right > greatest) {
                greatest = best_right;
                next.share = n ? best_right / (double)n : 0;
                best = next;
            }
        }
    }
    bench_winding(p, t->sense, best);
    bench_modes(p, pairs, best, modes);
    return best;
}

static void bench_ranks(BenchPieces *s, BenchPairs p, BenchAlignment align, const BenchModes *modes,
    i8 *out, i8 *strict) {
    double phase = align.global ? align.fraction : bench_phase(s, p, 0, align.chirality);
    for (i64 i = 0; i < p.count; i++) {
        int mask = p.eligible[i], first = BENCH_UNSCORED, best = 10000;
        for (int k = 0; mask && k < BENCH_KEEP; k++, mask >>= 1) {
            if (mask & 1) {
                BenchPair pair = bench_pair(p, i, k);
                i64 baseline, key = bench_frame(s, pair, align.global);
                i64 turn = bench_offset(s, pair, phase, align.chirality);
                if (turn != INT64_MIN && bench_mode(modes, key, &baseline) &&
                    baseline != INT64_MIN) {
                    i64 delta = turn - baseline;
                    if (llabs(delta) < abs(best)) {
                        best = delta;
                    }
                    if (k == 0 && llabs(delta) < 10000) {
                        first = fmin(100, fmax(-100, delta));
                    }
                }
            }
        }
        out[i] = best == 10000 ? BENCH_UNSCORED : fmin(100, fmax(-100, best));
        if (strict) {
            strict[i] = first;
        }
    }
}

static u8 *bench_choose_surface(Arena *a, BenchPieces *p, BenchTruth *t, const BenchOptions *o,
    const BenchMatches *matched, double tolerance, u8 **touching) {
    *touching = pushz(a, p->count);
    u8 *mask = pushz(a, p->count);
    for (i64 i = 0; i < p->count; i++) {
        if (!bench_in_zone(o, p->zone[i], p->xyz[3 * i + 2])) {
            continue;
        }
        double min = INFINITY, max = -INFINITY;
        int skip = 0;
        for (int k = 0; k < BENCH_KEEP; k++) {
            i64 vertex = matched[i].index[k];
            if (vertex < 0 || matched[i].depth[k] > tolerance) {
                continue;
            }
            if (t->flags[vertex] & BENCH_SKIP) {
                skip = 1;
                break;
            }
            double w = t->winding[vertex];
            if (bench_in_zone(o, t->zone[vertex], t->points[3 * vertex + 2]) && isfinite(w) &&
                t->frame[vertex] >= 0) {
                mask[i] |= 1 << k;
                min = fmin(min, w);
                max = fmax(max, w);
            }
        }
        if (skip) {
            mask[i] = 0;
        } else if (max - min >= 0.5) {
            (*touching)[i] = 1;
        }
    }
    return mask;
}

static u8 *bench_choose_truth(
    Arena *a, BenchTruth *t, const BenchOptions *o, const BenchMatches *matched, double tolerance) {
    u8 *mask = pushz(a, t->count);
    for (i64 i = 0; i < t->count; i++) {
        if ((t->flags[i] & BENCH_SKIP) || !isfinite(t->winding[i]) || t->frame[i] < 0 ||
            !bench_in_zone(o, t->zone[i], t->points[3 * i + 2])) {
            continue;
        }
        for (int k = 0; k < BENCH_KEEP; k++) {
            if (matched[i].index[k] >= 0 && matched[i].depth[k] <= tolerance) {
                mask[i] |= 1 << k;
            }
        }
    }
    return mask;
}

typedef struct {
    i64 count;
    i8 *rank;
    const u8 *flags;
} BenchCrossings;

static BenchCrossings bench_crossings(Arena *a, const char *folder, BenchPieces *p, BenchTruth *t,
    const BenchOptions *o, BenchAlignment align, BenchModes *m, double tolerance, double lateral) {
    const char *path = fmt(a, "%s/crossings.npz", folder);
    const Array z = bench_array(a, path, "z", "<f4", -1);
    i64 n = z.shape[0];
    BenchCrossings c = {
        .count = n, .rank = push(a, n), .flags = bench_array(a, path, "flags", "|u1", n).data};
    memset(c.rank, BENCH_UNSCORED, n);
    size_t saved = a->used;
    const f32 *z_values = z.data;
    const u8 *zones = bench_array(a, path, "zone", "|u1", n).data;
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        count += !(c.flags[i] & BENCH_SKIP) && bench_in_zone(o, zones[i], z_values[i]);
    }
    if (!count) {
        a->used = saved;
        return c;
    }
    const f32 *xy = bench_array(a, path, "xy", "<f4", 2 * n).data;
    const f32 *normals = bench_array(a, path, "normal", "<f4", 3 * n).data;
    const i16 *mesh = bench_array(a, path, "mesh", "<i2", n).data;
    const double *winding = bench_array(a, path, "winding", "<f8", n).data;
    int capacity = 1 << 15;
    i32 *mesh_frame = push(a, capacity * sizeof(*mesh_frame));
    for (int i = 0; i < capacity; i++) {
        mesh_frame[i] = -1;
    }
    for (i64 i = 0; i < t->count; i++) {
        if (t->mesh[i] >= 0) {
            mesh_frame[t->mesh[i]] = t->frame[i];
        }
    }
    f32 *points = push(a, 3 * count * sizeof(*points));
    f32 *dirs = push(a, 3 * count * sizeof(*dirs));
    i32 *frames = push(a, count * sizeof(*frames));
    double *turns = push(a, count * sizeof(*turns));
    i64 *row = push(a, count * sizeof(*row));
    for (i64 i = 0, k = 0; i < n; i++) {
        if (!(c.flags[i] & BENCH_SKIP) && bench_in_zone(o, zones[i], z_values[i])) {
            row[k] = i;
            points[3 * k] = xy[2 * i];
            points[3 * k + 1] = xy[2 * i + 1];
            points[3 * k + 2] = z_values[i];
            memcpy(dirs + 3 * k, normals + 3 * i, 3 * sizeof(*dirs));
            frames[k] = mesh[i] >= 0 ? mesh_frame[mesh[i]] : -1;
            turns[k++] = winding[i];
        }
    }
    BenchMatches *matches =
        bench_on_sheet(a, p->xyz, p->count, points, count, dirs, tolerance, lateral, 1);
    u8 *eligible = pushz(a, count);
    for (i64 i = 0; i < count; i++) {
        for (int j = 0; j < BENCH_KEEP; j++) {
            if (matches[i].index[j] >= 0 && isfinite(matches[i].depth[j])) {
                eligible[i] |= 1 << j;
            }
        }
    }
    i8 *ranks = push(a, count);
    BenchPairs pairs = {matches, eligible, turns, frames, count, 0};
    bench_ranks(p, pairs, align, m, ranks, NULL);
    for (i64 i = 0; i < count; i++) {
        c.rank[row[i]] = ranks[i];
    }
    a->used = saved;
    return c;
}

typedef struct {
    i64 steps, off, edges, folded, judged, wrong, flipped, vertices;
    double area[5], largest;
} BenchArea;

static void bench_decimal(json_object *o, const char *key, double value, int digits) {
    double scale = pow(10, digits);
    json_object_object_add(o, key,
        isfinite(value) ? jnum(nearbyint(value * scale) / scale) : json_object_new_double(value));
}

static void bench_ratio(json_object *o, const char *name, double n, double d, int digits) {
    if (d > 0) {
        bench_decimal(o, name, n / d, digits);
    } else {
        json_object_object_add(o, name, NULL);
    }
}

static int bench_correct(const BenchPairs *p, const i8 *rank, i64 vertex) {
    return p->eligible[vertex] && rank[vertex] == 0;
}

static double bench_connected(
    Arena *a, BenchSurface *s, const BenchPieces *p, const BenchPairs *matches, const i8 *ranks) {
    size_t saved = a->used;
    i64 size = (i64)s->h * s->w;
    double best = 0;
    u8 *seen = pushz(a, size);
    i32 *queue = push(a, s->n * sizeof(*queue));
    for (i64 i = 0; i < size; i++) {
        i32 current = s->index[i];
        if (current < 0 || seen[i] || !bench_correct(matches, ranks, s->first + current)) {
            continue;
        }
        seen[i] = 1;
        queue[0] = i;
        i64 next = 1;
        double area = 0;
        for (i64 j = 0; j < next; j++) {
            i64 at = queue[j], y = at / s->w, x = at % s->w;
            area += p->area[s->first + s->index[at]];
            for (int r = -1; r <= 1; r++) {
                for (int c = -1; c <= 1; c++) {
                    if (y + r < 0 || y + r >= s->h || x + c < 0 || x + c >= s->w) {
                        continue;
                    }
                    i64 other = at + r * (i64)s->w + c;
                    i32 index = s->index[other];
                    if (index >= 0 && !seen[other] &&
                        bench_correct(matches, ranks, s->first + index)) {
                        queue[next++] = other;
                        seen[other] = 1;
                    }
                }
            }
        }
        best = fmax(best, area);
    }
    a->used = saved;
    return best;
}

static json_object *bench_surface_stats(Arena *a, BenchPieces *p, BenchTruth *t,
    const BenchOptions *o, BenchPairs pairs, const i8 *ranks, const u8 *ambiguous, double tolerance,
    BenchArea *stats) {
    json_object *surfaces = json_object_new_array();
    double total_area = 0;
    i64 surface_touching = 0, judged_count = 0;
    for (int s = 0; s < p->n; s++) {
        BenchSurface *piece = p->surfaces + s;
        size_t saved = a->used;
        i64 size = (i64)piece->h * piece->w, base = piece->first;
        double *steps = push(a, piece->n * sizeof(*steps));
        double areas[5] = {0};
        i64 normal_in = 0, normal_out = 0;
        stats->largest = fmax(stats->largest, bench_connected(a, piece, p, &pairs, ranks));
        for (i64 i = 0; i < size; i++) {
            if (piece->index[i] < 0) {
                continue;
            }
            i64 v = base + piece->index[i];
            const f32 *xyz = p->xyz + 3 * v;
            if (!bench_in_zone(o, p->zone[v], xyz[2])) {
                continue;
            }
            stats->vertices++;
            int judged = pairs.eligible[v], on_truth = pairs.matches[v].depth[0] <= tolerance;
            int group = judged                   ? (ranks[v] == 0 ? 0 : 1)
                : on_truth                       ? 2
                : pairs.matches[v].index[0] >= 0 ? 3
                                                 : 4;
            judged_count += judged != 0;
            surface_touching += ambiguous[v];
            areas[group] += p->area[v];
            const f32 *n = p->normals + 3 * v;
            if (bench_finite(n)) {
                double x = interpolated(xyz[2], t->knots, t->nknots, 1) - xyz[0];
                double y = interpolated(xyz[2], t->knots, t->nknots, 2) - xyz[1];
                double dot = x * n[0] + y * n[1];
                normal_in += dot > 0;
                normal_out += dot < 0;
            }
        }
        stats->flipped += normal_in < normal_out ? normal_in : normal_out;
        for (int axis = 0; axis < 2; axis++) {
            i64 count = 0;
            for (i64 i = 0; i < size; i++) {
                if (piece->index[i] < 0 ||
                    (axis == 0 ? i / piece->w == piece->h - 1 : i % piece->w == piece->w - 1)) {
                    continue;
                }
                i32 next = piece->index[i + (axis == 0 ? piece->w : 1)];
                if (next < 0) {
                    continue;
                }
                i64 v = base + piece->index[i], u = base + next;
                if (ranks[v] != BENCH_UNSCORED && ranks[u] != BENCH_UNSCORED) {
                    stats->judged++;
                    stats->wrong += ranks[v] != ranks[u];
                }
                const f32 *x = p->xyz + 3 * v;
                if (!bench_in_zone(o, p->zone[v], x[2])) {
                    continue;
                }
                steps[count++] = bench_step(x, p->xyz + 3 * u);
                const f32 *n = p->normals + 3 * v, *m = p->normals + 3 * u;
                double dot = (double)n[0] * m[0] + (double)n[1] * m[1] + (double)n[2] * m[2];
                if (isfinite(dot)) {
                    stats->edges++;
                    stats->folded += dot < 0;
                }
            }
            if (count) {
                double median = bench_median(steps, count, 0);
                stats->steps += count;
                for (i64 j = 0; j < count; j++) {
                    stats->off += fabs(steps[j] - median) > 0.10 * median;
                }
            }
        }
        double area = 0;
        for (int k = 0; k < 5; k++) {
            stats->area[k] += areas[k];
            area += areas[k];
        }
        total_area += area;
        json_object *row = json_object_new_object();
        json_object_object_add(row, "surface", json_object_new_string(piece->name));
        json_object_object_add(
            row, "winding", json_object_get(json_object_object_get(piece->meta, "winding")));
        bench_decimal(row, "area_cm2", area, 3);
        bench_decimal(row, "correct_cm2", areas[0], 3);
        bench_decimal(row, "wrong_cm2", areas[1], 3);
        json_object_array_add(surfaces, row);
        a->used = saved;
    }
    json_object *result = json_object_new_object();
    json_object_object_add(result, "per_surface", surfaces);
    bench_decimal(result, "surface_cm2", total_area, 3);
    bench_decimal(
        result, "surface_on_touching_truth_share", surface_touching / fmax(judged_count, 1), 4);
    bench_ratio(result, "wrong_join_share", stats->wrong, stats->judged, 5);
    bench_ratio(result, "non_isometric_share", stats->off, stats->steps, 4);
    bench_ratio(result, "flipped_share", stats->flipped, stats->vertices, 5);
    bench_ratio(result, "folded_edge_share", stats->folded, stats->edges, 5);
    return result;
}

static json_object *bench_distances(Arena *a, BenchPieces *p, BenchTruth *t, const BenchOptions *o,
    const BenchMatches *to_truth, const BenchMatches *to_surface, double tolerance) {
    const char *keys[] = {"truth_to_surface_covered_p50_p90",
        "truth_to_surface_within_300um_p50_p90", "surface_to_truth_p50_p90",
        "surface_near_truth_p50_p90"};
    json_object *out = json_object_new_object();
    size_t saved = a->used;
    i64 capacity = p->count > t->count ? p->count : t->count;
    double *values = push(a, capacity * sizeof(*values));
    for (int k = 0; k < 4; k++) {
        i64 n = 0, count = k < 2 ? t->count : p->count;
        const BenchMatches *matches = k < 2 ? to_surface : to_truth;
        for (i64 i = 0; i < count; i++) {
            double d = matches[i].depth[0];
            if (matches[i].index[0] < 0 || (!(k & 1) && d > tolerance)) {
                continue;
            }
            int in_scope = k < 2 ? !(t->flags[i] & BENCH_DUPLICATE) &&
                    bench_in_zone(o, t->zone[i], t->points[3 * i + 2])
                                 : bench_in_zone(o, p->zone[i], p->xyz[3 * i + 2]);
            if (in_scope) {
                values[n++] = d;
            }
        }
        json_object *row = n ? json_object_new_array() : NULL;
        if (n) {
            qsort(values, n, sizeof(*values), bench_double_order);
            for (int q = 0; q < 2; q++) {
                double at = (n - 1) * (q ? 0.9 : 0.5);
                i64 lo = floor(at), hi = ceil(at);
                double d = values[lo] + (at - lo) * (values[hi] - values[lo]);
                json_object_array_add(row, jnum(nearbyint(d * 100) / 100));
            }
        }
        json_object_object_add(out, keys[k], row);
    }
    a->used = saved;
    return out;
}

static void bench_result_areas(json_object *out, BenchTruth *t, const BenchOptions *o,
    const BenchArea *stats, const BenchMatches *matches, const i8 *ranks, double tolerance) {
    double truth = 0, covered = 0, correct = 0, by_press[2][2] = {{0}};
    for (i64 i = 0; i < t->count; i++) {
        if ((t->flags[i] & BENCH_DUPLICATE) ||
            !bench_in_zone(o, t->zone[i], t->points[3 * i + 2])) {
            continue;
        }
        double area = t->area[i] / 100.0f;
        int press = (t->flags[i] & BENCH_PRESSED) != 0;
        truth += area;
        by_press[press][0] += area;
        correct += ranks[i] == 0 ? area : 0;
        if (matches[i].depth[0] <= tolerance) {
            covered += area;
            by_press[press][1] += area;
        }
    }
    bench_decimal(out, "truth_cm2", truth, 3);
    json_object *area = json_object_new_object();
    const char *key[] = {
        "correct", "wrong", "on_truth_unjudged", "near_off_truth", "far_from_truth"};
    for (int i = 0; i < 5; i++) {
        bench_decimal(area, key[i], stats->area[i], 3);
    }
    bench_decimal(area, "largest_connected_correct", stats->largest, 3);
    bench_decimal(area, "truth_covered", covered, 3);
    bench_decimal(area, "truth_covered_correct", correct, 3);
    json_object_object_add(out, "area_cm2", area);
    bench_decimal(out, "truth_recall", covered / fmax(truth, 1e-12), 4);
    bench_decimal(out, "truth_recall_correct", correct / fmax(truth, 1e-12), 4);
    bench_decimal(out, "truth_recall_pressed", by_press[1][1] / fmax(by_press[1][0], 1e-12), 4);
    bench_decimal(out, "truth_recall_loose", by_press[0][1] / fmax(by_press[0][0], 1e-12), 4);
}

static json_object *bench_winding_stats(
    BenchTruth *t, const i8 *rank, const i8 *strict, BenchCrossings cross) {
    json_object *out = json_object_new_object();
    i64 matches[5][2] = {{0}}, errors[4][2] = {{0}}, hist[201] = {0};
    for (i64 i = 0; i < t->count; i++) {
        if (rank[i] == BENCH_UNSCORED) {
            continue;
        }
        int pressed = (t->flags[i] & BENCH_PRESSED) != 0;
        int touching = (t->flags[i] & BENCH_TOUCHING) != 0;
        matches[0][0]++;
        matches[0][1] += rank[i] == 0;
        matches[pressed ? 1 : 2][0]++;
        matches[pressed ? 1 : 2][1] += rank[i] == 0;
        matches[touching ? 4 : 3][0]++;
        matches[touching ? 4 : 3][1] += (touching ? rank[i] : strict[i]) == 0;
        hist[rank[i] + 100]++;
    }
    for (i64 i = 0; i < cross.count; i++) {
        if (cross.rank[i] == BENCH_UNSCORED) {
            continue;
        }
        int pressed = (cross.flags[i] & BENCH_PRESSED) != 0;
        int error = cross.rank[i] != 0;
        errors[0][0]++;
        errors[0][1] += error;
        errors[pressed ? 1 : 2][0]++;
        errors[pressed ? 1 : 2][1] += error;
        if (cross.flags[i] & BENCH_OVERLAP) {
            errors[3][0]++;
            errors[3][1] += error;
        }
    }
    const char *truth_keys[] = {"right_share", "right_share_pressed", "right_share_loose",
        "right_share_untouched_strict", "right_share_touching"};
    const char *cross_keys[] = {"rank_error_share_crossings", "rank_error_share_crossings_pressed",
        "rank_error_share_crossings_loose", "rank_error_share_crossings_overlap"};
    json_object_object_add(out, "judged_truth", json_object_new_int64(matches[0][0]));
    for (int i = 0; i < 5; i++) {
        bench_ratio(out, truth_keys[i], matches[i][1], matches[i][0], 4);
    }
    bench_decimal(out, "touching_share", matches[4][0] / fmax(matches[0][0], 1), 4);
    for (int i = 0; i < 4; i++) {
        bench_ratio(out, cross_keys[i], errors[i][1], errors[i][0], 4);
    }
    json_object_object_add(out, "judged_crossings", json_object_new_int64(errors[0][0]));
    json_object *histogram = json_object_new_object();
    for (int i = 0; i < 201; i++) {
        if (hist[i]) {
            char key[12];
            snprintf(key, sizeof(key), "%d", i - 100);
            json_object_object_add(histogram, key, json_object_new_int64(hist[i]));
        }
    }
    json_object_object_add(out, "rank_error_histogram", histogram);
    return out;
}

static u32 bench_utf8(const char **word) {
    const u8 *p = (const u8 *)*word;
    u32 c = *p++;
    int remain = (c & 0xf8) == 0xf0 ? 3 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xe0) == 0xc0 ? 1 : 0;
    if (remain) {
        c &= (1u << (6 - remain)) - 1;
        for (int i = 0; i < remain && (*p & 0xc0) == 0x80; i++) {
            c = c << 6 | (*p++ & 63);
        }
    }
    *word = (const char *)p;
    return c;
}

static void bench_save(Arena *a, const char *folder, BenchPieces *p, BenchTruth *t,
    BenchCrossings cross, const i8 *ranks, const i8 *truth_rank, const BenchMatches *to_truth,
    const BenchMatches *to_surface) {
    npy_write(fmt(a, "%s/truth_rank.npy", folder),
        (Array){.descr = "|i1",
            .ndim = 1,
            .shape = {t->count},
            .data = (void *)truth_rank,
            .size = t->count});
    npy_write(fmt(a, "%s/crossing_rank.npy", folder),
        (Array){.descr = "|i1",
            .ndim = 1,
            .shape = {cross.count},
            .data = cross.rank,
            .size = cross.count});
    size_t saved = a->used;
    f32 *depth = push(a, t->count * sizeof(*depth));
    for (i64 i = 0; i < t->count; i++) {
        depth[i] = to_surface[i].depth[0];
    }
    npy_write(fmt(a, "%s/truth_depth.npy", folder),
        (Array){.descr = "<f4",
            .ndim = 1,
            .shape = {t->count},
            .data = depth,
            .size = t->count * sizeof(*depth)});
    a->used = saved;
    i64 n = p->count;
    i64 *truth = push(a, n * sizeof(*truth)), *rank = push(a, n * sizeof(*rank));
    depth = push(a, n * sizeof(*depth));
    for (i64 i = 0; i < n; i++) {
        truth[i] = to_truth[i].index[0];
        depth[i] = to_truth[i].depth[0];
        rank[i] = ranks[i];
    }
    int width = 1;
    for (int i = 0; i < p->n; i++) {
        int len = 0;
        for (const char *name = p->surfaces[i].name; *name; len++) {
            bench_utf8(&name);
        }
        width = width < len ? len : width;
    }
    u32 *names = pushz(a, (i64)p->n * width * sizeof(*names));
    for (int i = 0; i < p->n; i++) {
        const char *name = p->surfaces[i].name;
        for (int k = 0; *name; k++) {
            names[i * width + k] = bench_utf8(&name);
        }
    }
    const char *keys[] = {"surface", "vertex", "truth_vertex", "depth_vox", "rank", "names"};
    Array arrays[] = {
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = p->surface, .size = n * 4},
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = p->vertex, .size = n * 4},
        {.descr = "<i8", .ndim = 1, .shape = {n}, .data = truth, .size = n * 8},
        {.descr = "<f4", .ndim = 1, .shape = {n}, .data = depth, .size = n * 4},
        {.descr = "<i8", .ndim = 1, .shape = {n}, .data = rank, .size = n * 8},
        {.ndim = 1, .shape = {p->n}, .data = names, .size = (i64)p->n * width * 4},
    };
    snprintf(arrays[5].descr, sizeof(arrays[5].descr), "<U%d", width);
    npz_write(fmt(a, "%s/surface_labels.npz", folder), 6, keys, arrays);
    a->used = saved;
}

static json_object *bench_score(
    Arena *a, const BenchOptions *o, json_object *paths, json_object *zones) {
    const char *input = fmt(a, "%s/bench/controls/%s", OUTPUTS, o->control);
    BenchTruth t = bench_truth(a, input);
    BenchPieces p = bench_pieces(a, paths, &t, o);
    double tolerance = 40 / t.voxel, depth = 300 / t.voxel, lateral = 0.75 * t.spacing / t.voxel;
    double *steps = push(a, p.n * sizeof(*steps));
    for (int i = 0; i < p.n; i++) {
        steps[i] = p.surfaces[i].step;
    }
    double step = bench_median(steps, p.n, 1);
    BenchMatches *to_truth =
        bench_on_sheet(a, t.points, t.count, p.xyz, p.count, t.normals, depth, lateral, 0);
    u8 *ambiguous;
    u8 *surface_mask = bench_choose_surface(a, &p, &t, o, to_truth, tolerance, &ambiguous);
    BenchPairs surface_pairs = {to_truth, surface_mask, t.winding, t.frame, p.count, 1};
    BenchMatches *to_surface =
        bench_on_sheet(a, p.xyz, p.count, t.points, t.count, t.normals, depth, 0.75 * step, 1);
    u8 *truth_mask = bench_choose_truth(a, &t, o, to_surface, tolerance);
    BenchPairs truth_pairs = {to_surface, truth_mask, t.winding, t.frame, t.count, 0};
    BenchModes modes = {.vote = push(a, p.count * sizeof(BenchVote))};
    BenchAlignment align = bench_alignment(&p, &t, surface_pairs, &modes);
    i8 *surface_rank = push(a, p.count), *truth_rank = push(a, t.count);
    i8 *truth_strict = push(a, t.count);
    bench_ranks(&p, surface_pairs, align, &modes, surface_rank, NULL);
    bench_ranks(&p, truth_pairs, align, &modes, truth_rank, truth_strict);
    BenchCrossings cross =
        bench_crossings(a, input, &p, &t, o, align, &modes, tolerance, 0.75 * step);
    BenchArea stats = {0};
    json_object *out = bench_surface_stats(
        a, &p, &t, o, surface_pairs, surface_rank, ambiguous, tolerance, &stats);
    json_object_object_add(out, "control", json_object_new_string(o->control));
    json_object_object_add(out, "zones", json_object_get(zones));
    json_object_object_add(out, "label", json_object_new_string(o->label));
    json_object_object_add(out, "mode", json_object_new_string(align.global ? "global" : "local"));
    json_object *alignment = NULL;
    if (align.global) {
        alignment = json_object_new_object();
        json_object_object_add(alignment, "chirality", json_object_new_int(align.chirality));
        bench_decimal(alignment, "cut_deg", align.cut * 180 / M_PI, 1);
        bench_decimal(alignment, "fraction", align.fraction, 4);
        bench_decimal(alignment, "fit_share", align.share, 4);
    }
    json_object_object_add(out, "alignment", alignment);
    json_object_object_add(out, "surfaces", json_object_new_int(p.n));
    json_object_object_add(
        out, "distance_vox", bench_distances(a, &p, &t, o, to_truth, to_surface, tolerance));
    json_object_object_add(
        out, "winding", bench_winding_stats(&t, truth_rank, truth_strict, cross));
    bench_result_areas(out, &t, o, &stats, to_surface, truth_rank, tolerance);
    mkdirs(o->folder);
    bench_save(a, o->folder, &p, &t, cross, surface_rank, truth_rank, to_truth, to_surface);
    json_object_object_add(out, "folder", json_object_new_string(o->folder));
    json_write(fmt(a, "%s/summary.json", o->folder), out);
    json_object_put(t.record);
    for (int i = 0; i < p.n; i++) {
        json_object_put(p.surfaces[i].meta);
    }
    return out;
}

static json_object *bench_headline(json_object *out) {
    json_object *flat = json_object_new_object(), *w = jget(out, "winding"),
                *a = jget(out, "area_cm2");
    const char *keys[] = {"truth_recall", "truth_recall_correct", "wrong_join_share",
        "non_isometric_share", "flipped_share", "folded_edge_share"};
    for (int i = 0; i < 6; i++) {
        json_object_object_add(flat, keys[i], json_object_get(jget(out, keys[i])));
    }
    const char *inside[] = {"right_share", "right_share_pressed", "right_share_loose",
        "rank_error_share_crossings", "correct", "wrong", "largest_connected_correct"};
    const char *names[] = {"winding_right_share", "winding_right_pressed", "winding_right_loose",
        "rank_error_crossings", "correct_cm2", "wrong_cm2", "largest_correct_cm2"};
    for (int i = 0; i < 7; i++) {
        json_object_object_add(flat, names[i], json_object_get(jget(i < 4 ? w : a, inside[i])));
    }
    json_object *d = jget(out, "distance_vox");
    const char *groups[] = {"truth_to_surface_covered_p50_p90", "surface_to_truth_p50_p90"};
    const char *prefix[] = {"truth_to_surface", "surface_to_truth"};
    for (int i = 0; i < 2; i++) {
        json_object *stats = jget(d, groups[i]);
        for (int k = 0; k < board_count(stats); k++) {
            char metric[48];
            snprintf(metric, sizeof(metric), "%s_p%d", prefix[i], k ? 90 : 50);
            json_object_object_add(
                flat, metric, json_object_get(json_object_array_get_idx(stats, k)));
        }
    }
    return flat;
}

static void bench_git(char out[128], int dirty) {
    *out = 0;
    int stream[2];
    need(!pipe(stream), "cannot read bench commit");
    pid_t pid = fork();
    need(pid >= 0, "cannot start git");
    if (!pid) {
        close(stream[0]);
        dup2(stream[1], STDOUT_FILENO);
        close(stream[1]);
        if (dirty) {
            execlp("git", "git", "-C", HOME, "status", "--porcelain", "--", "ml-experimental/logos",
                "level", "unroll", "calibrate/bench", "scans/chunks", "storage/data_roots.c",
                (char *)NULL);
        } else {
            execlp("git", "git", "-C", HOME, "rev-parse", "--short", "HEAD", (char *)NULL);
        }
        _exit(127);
    }
    close(stream[1]);
    FILE *f = fdopen(stream[0], "r");
    need(f != NULL, "cannot read git output");
    char line[128];
    if (fgets(out, 128, f)) {
        out[strcspn(out, "\r\n")] = 0;
    }
    while (fgets(line, sizeof(line), f)) {
    }
    fclose(f);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
    }
}

static const char *bench_json_array(Arena *a, json_object *items) {
    const char *text = json_object_to_json_string_ext(
        items, JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_NOSLASHESCAPE);
    size_t n = strlen(text);
    return n > 2 ? fmt(a, "[%.*s]", (int)n - 4, text + 2) : "[]";
}

static void bench_append(const char *folder, json_object *entry) {
    mkparent(folder);
    int fd = open(folder, O_APPEND | O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    need(fd >= 0 && !flock(fd, LOCK_EX), "cannot append bench record at %s", folder);
    const char *text = board_json(entry);
    write_all(fd, text, strlen(text), folder);
    write_all(fd, "\n", 1, folder);
    close(fd);
}

static Board bench_database(Arena *a) {
    const char *database = fmt(a, "%s/bench/results.sqlite", OUTPUTS);
    mkparent(database);
    Board db = {0};
    need(sqlite3_open(database, &db.db) == SQLITE_OK, "cannot open bench database %s", database);
    sqlite3_busy_timeout(db.db, 60000);
    board_exec(&db,
        "PRAGMA foreign_keys=ON; "
        "CREATE TABLE IF NOT EXISTS runs (id integer primary key,unit text unique,created text,"
        "control text,split text,system text,method text,kind text,commit_id text,parameters text,"
        "source text,detail text);"
        "CREATE TABLE IF NOT EXISTS metrics(run integer references runs(id) ON DELETE CASCADE,"
        "key text,value real,primary key(run,key));"
        "CREATE INDEX IF NOT EXISTS runs_by_control ON runs(control,system,kind,method)");
    return db;
}

static void bench_record(Arena *a, const BenchOptions *o, const char *when, const char *commit,
    json_object *given, json_object *zones, json_object *metrics, json_object *detail) {
    Board db = bench_database(a);
    const char *params = fmt(a, "{\"mask\": \"%s\", \"surfaces\": %s, \"zones\": %s}",
        o->locked ? "locked" : "known", bench_json_array(a, given), bench_json_array(a, zones));
    const char *split = board_item(zones, 0);
    for (int i = 1; i < board_count(zones); i++) {
        split = fmt(a, "%s,%s", split, board_item(zones, i));
    }
    char hex[65];
    sha256_hex(params, strlen(params), hex);
    const char *unit = fmt(a, "%s|%s|score|%s|%.12s", o->system, o->label, o->control, hex);
    board_exec(&db, "BEGIN IMMEDIATE; DELETE FROM runs WHERE unit=%Q", unit);
    board_exec(&db,
        "INSERT INTO runs(unit,created,control,split,system,method,kind,commit_id,"
        "parameters,detail) VALUES (%Q,%Q,%Q,%Q,%Q,%Q,'score',%Q,%Q,%Q)",
        unit, when, o->control, split, o->system, o->label, commit, params, board_json(detail));
    sqlite3_int64 run = sqlite3_last_insert_rowid(db.db);
    json_object_object_foreach(metrics, key, value) {
        double v = json_object_get_double(value);
        if (value && isfinite(v)) {
            board_exec(&db, "INSERT INTO metrics(run,key,value) VALUES (%lld,%Q,%!.17g)",
                (long long)run, key, v);
        }
    }
    board_exec(&db, "COMMIT");
    sqlite3_close(db.db);
    printf("recorded run %lld\n", (long long)run);
}

static void bench_time(char label[32], char when[32]) {
    time_t stamp = time(NULL);
    struct tm local;
    localtime_r(&stamp, &local);
    strftime(label, 32, "%Y%m%d-%H%M%S", &local);
    strftime(when, 32, "%Y-%m-%d %H:%M:%S", &local);
}

int bench_score_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.score CONTROL SURFACES ... [--holdout] [--zones LIST] "
             "[--mask known|locked] [--label NAME] [--system NAME] [--out FOLDER] "
             "[--no-record] [--ignore-winding] [--z Z0:Z1]");
        return !argc;
    }
    Arena a = arena(64 * GB);
    BenchOptions o = {.control = argv[0],
        .label = "surfaces",
        .system = "network",
        .low = -INFINITY,
        .high = INFINITY};
    need(site_name(o.control), "invalid bench control %s", o.control);
    json_object *given = json_object_new_array(), *zones = json_object_new_array();
    const char *zone_names = NULL;
    int holdout = 0, record = 1;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--holdout")) {
            holdout = 1;
        } else if (!strcmp(arg, "--no-record")) {
            record = 0;
        } else if (!strcmp(arg, "--ignore-winding")) {
            o.ignore_winding = 1;
        } else if (!strcmp(arg, "--")) {
            for (++i; i < argc; i++) {
                json_object_array_add(given, json_object_new_string(argv[i]));
            }
        } else if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            json_object_put(given);
            json_object_put(zones);
            munmap(a.base, a.size);
            return bench_score_command(0, NULL) - 1;
        } else if (!strncmp(arg, "--", 2)) {
            need(++i < argc, "%s needs a value", arg);
            const char *value = argv[i];
            if (!strcmp(arg, "--zones")) {
                zone_names = value;
            } else if (!strcmp(arg, "--mask")) {
                need(!strcmp(value, "known") || !strcmp(value, "locked"), "invalid mask %s", value);
                o.locked = !strcmp(value, "locked");
            } else if (!strcmp(arg, "--label")) {
                o.label = value;
            } else if (!strcmp(arg, "--system")) {
                o.system = value;
            } else if (!strcmp(arg, "--out")) {
                o.folder = value;
            } else if (!strcmp(arg, "--z")) {
                char *end;
                o.low = strtod(value, &end);
                need(end > value && *end == ':', "--z needs Z0:Z1");
                o.high = board_value(end + 1);
            } else {
                die("unknown score option %s", arg);
            }
        } else {
            json_object_array_add(given, json_object_new_string(arg));
        }
    }
    need(board_count(given), "bench score needs at least one surface");
    char *list = fmt(&a, "%s", zone_names ? zone_names : holdout ? "holdout-A,holdout-B" : "dev");
    for (char *item; (item = strsep(&list, ","));) {
        int zone = bench_zone_of(item);
        need(zone >= 0, "unknown bench zone %s", item);
        o.zones |= 1 << zone;
        json_object_array_add(zones, json_object_new_string(item));
    }
    int scored_holdout = o.zones & ((1 << 2) | (1 << 3));
    char dirty[128] = "", head[128] = "", day[32], when[32];
    if (scored_holdout || record) {
        bench_git(dirty, 1);
        bench_git(head, 0);
    }
    need(!scored_holdout || holdout, "scoring holdout zones needs --holdout");
    need(!scored_holdout || !*dirty,
        "holdout scoring needs a clean commit of the unwrap/ML code "
        "and the bench (git status shows changes)");
    const char *commit = fmt(&a, "%s%s", *head ? head : "unknown", *dirty ? "+dirty" : "");
    bench_time(day, when);
    if (!o.folder) {
        need(!strchr(o.label, '/'), "score labels must not contain slashes");
        o.folder = fmt(&a, "%s/bench/scores/%s/%s-%s", OUTPUTS, o.control, day, o.label);
    }
    json_object *paths = bench_paths(&a, given);
    json_object *out = bench_score(&a, &o, paths, zones);
    json_object *metrics = bench_headline(out);
    if (scored_holdout) {
        json_object *entry = json_object_new_object();
        const char *keys[] = {"time", "control", "label", "commit", "folder"};
        const char *values[] = {when, o.control, o.label, commit, o.folder};
        for (int i = 0; i < 5; i++) {
            json_object_object_add(entry, keys[i], json_object_new_string(values[i]));
        }
        json_object_object_add(entry, "surfaces", json_object_get(given));
        json_object_object_add(entry, "zones", json_object_get(zones));
        json_object_object_add(entry, "headline", json_object_get(metrics));
        bench_append(fmt(&a, "%s/bench/holdout-log.jsonl", OUTPUTS), entry);
        json_object_put(entry);
    }
    json_object_object_del(out, "per_surface");
    puts(json_object_to_json_string_ext(
        out, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_NOSLASHESCAPE));
    if (record) {
        bench_record(&a, &o, when, commit, given, zones, metrics, out);
    }
    json_object_put(metrics);
    json_object_put(given);
    json_object_put(zones);
    json_object_put(paths);
    json_object_put(out);
    munmap(a.base, a.size);
    return 0;
}
