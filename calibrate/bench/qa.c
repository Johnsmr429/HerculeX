#ifndef HX_UNITY
const char hx_usage[] = "Measure the CT coverage and profile of frozen truth vertices.";
#endif

typedef struct {
    double threshold, centre;
    i64 count;
} BenchPapyrus;

static f32 qa_half(u16 half) {
    int exponent = (half >> 10) & 31, fraction = half & 1023;
    f32 value = !exponent ? ldexpf(fraction, -24)
        : exponent == 31  ? fraction ? NAN : INFINITY
                          : ldexpf(fraction + 1024, exponent - 25);
    return half & 0x8000 ? -value : value;
}

static BenchPapyrus qa_papyrus(Arena *a, const char *path) {
    size_t saved = a->used;
    Array profiles = npz_get(a, path, "profiles"), offsets = npz_get(a, path, "offsets_vox");
    need(!strcmp(profiles.descr, "<f2") && profiles.ndim == 2 &&
            profiles.shape[1] == RIDGE_POINTS && !strcmp(offsets.descr, "<f4") &&
            elements(offsets) == RIDGE_POINTS,
        "unrecognized ridge sample at %s", path);
    i64 n = profiles.shape[0], positive = 0, known = 0;
    double *all = push(a, n * RIDGE_POINTS * sizeof(*all)), *peak = push(a, n * sizeof(*peak));
    f32 sum[RIDGE_POINTS] = {0};
    const u16 *input = profiles.data;
    const f32 *z = offsets.data;
    for (i64 i = 0; i < n; i++) {
        f32 raw[RIDGE_POINTS], smooth[RIDGE_POINTS];
        int empty = 1;
        for (int d = 0; d < RIDGE_POINTS; d++) {
            raw[d] = qa_half(input[i * RIDGE_POINTS + d]);
            if (fabsf(z[d]) <= 3 && raw[d] != 0) {
                empty = 0;
            }
        }
        if (empty) {
            continue;
        }
        ridge_smooth(raw, RIDGE_POINTS, 1, smooth);
        f32 best = -INFINITY;
        for (int d = 0; d < RIDGE_POINTS; d++) {
            sum[d] += raw[d];
            if (smooth[d] > 0) {
                all[positive++] = smooth[d];
            }
            best = fmaxf(best, smooth[d]);
        }
        peak[known++] = best;
    }
    int best = 0;
    for (int d = 1; d < RIDGE_POINTS; d++) {
        if (sum[d] > sum[best]) {
            best = d;
        }
    }
    BenchPapyrus p = {.centre = z[best], .count = known, .threshold = NAN};
    if (positive) {
        double air = package_quantile(all, positive, 0.05);
        p.threshold = air + 0.5 * (bench_median(peak, known, NAN) - air);
    }
    a->used = saved;
    return p;
}

typedef struct {
    const BenchTruth *truth;
    const RidgeVolume *volume;
    BenchPapyrus papyrus;
    u8 *qa, *peak;
} BenchVertexQA;

static void qa_scan_vertices(void *opaque, i64 block) {
    BenchVertexQA *q = opaque;
    const BenchTruth *t = q->truth;
    const RidgeVolume *v = q->volume;
    enum { POINTS = 37 };
    const int multiple[] = {2, 4, 4};
    for (i64 i = block * 256; i < t->count && i < (block + 1) * 256; i++) {
        const f32 *p = t->points + 3 * i, *normals = t->normals + 3 * i;
        if (!package_good_normal(normals)) {
            q->qa[i] = 255;
            q->peak[i] = 0;
            continue;
        }
        double lo[3], hi[3], first[3], last[3];
        for (int d = 0; d < 3; d++) {
            i64 stride = v->chunk[d] * multiple[d];
            double cell = floor((double)p[2 - d] / stride) * stride;
            lo[d] = cell - 8;
            hi[d] = cell + stride + 7;
            first[d] = floor(lo[d] / v->chunk[d]);
            last[d] = floor(hi[d] / v->chunk[d]);
        }
        f32 values[POINTS], smooth[POINTS];
        int unread = 0, empty = 1;
        for (int d = 0; d < POINTS; d++) {
            double at[3];
            i64 chunk[3];
            double shift = d * 0.5 - 9;
            for (int k = 0; k < 3; k++) {
                double voxel = (double)p[2 - k] + shift * normals[2 - k];
                double c = floor(voxel / v->chunk[k]);
                chunk[k] = fmax(first[k], fmin(last[k], c));
                at[k] = fmax(lo[k], fmin(hi[k], voxel));
            }
            unread |= ridge_offset(v, chunk[0], chunk[1], chunk[2]) == UNKNOWN;
            values[d] = ridge_interp_at(v, at);
        }
        ridge_smooth(values, POINTS, 1, smooth);
        f32 peak = 0;
        for (int d = 0; d < POINTS; d++) {
            double shift = d * 0.5 - 9;
            if (fabs(shift - q->papyrus.centre) <= 3) {
                peak = fmaxf(peak, smooth[d]);
            }
            if (fabs(shift) <= 3 && smooth[d] != 0) {
                empty = 0;
            }
        }
        q->qa[i] = unread ? 254 : empty ? 1 : peak < q->papyrus.threshold ? 2 : 0;
        q->peak[i] = fmax(0, fmin(255, peak));
    }
}

static json_object *qa_vertices(Arena *a, const char *name) {
    size_t saved = a->used;
    const char *folder = fmt(a, "%s/bench/controls/%s", OUTPUTS, name);
    BenchTruth truth = bench_truth(a, folder);
    const char *sample = fmt(a, "%s/bench/checks/%s-ridge.npz", OUTPUTS, name);
    BenchPapyrus papyrus = qa_papyrus(a, sample);
    need(papyrus.count > 0 && isfinite(papyrus.threshold), "%s has no sampled papyrus", name);
    json_object *entry = jget(truth.record, "scan");
    RidgeVolume volume = ridge_volume_open(a, jstr(entry, "sample"), jstr(entry, "scan"));
    i64 n = truth.count;
    BenchVertexQA run = {.truth = &truth,
        .volume = &volume,
        .papyrus = papyrus,
        .qa = push(a, n),
        .peak = push(a, n)};
    parallel((n + 255) / 256, 4, qa_scan_vertices, &run);
    ridge_volume_close(&volume);
    const char *path = fmt(a, "%s/bench/qa/%s/vertices.npz", OUTPUTS, name);
    const char *part = fmt(a, "%s.%d.part", path, getpid());
    const char *keys[] = {"qa", "peak", "threshold", "papyrus_offset_vox"};
    Array arrays[] = {
        {.descr = "|u1", .ndim = 1, .shape = {n}, .size = n, .data = run.qa},
        {.descr = "|u1", .ndim = 1, .shape = {n}, .size = n, .data = run.peak},
        {.descr = "<f8", .size = sizeof(double), .data = &papyrus.threshold},
        {.descr = "<f8", .size = sizeof(double), .data = &papyrus.centre},
    };
    mkparent(path);
    npz_write_codec(part, sizeof(keys) / sizeof(*keys), keys, arrays, ZIP_CM_DEFLATE);
    need(!rename(part, path), "cannot publish truth vertex QA at %s: %s", path, strerror(errno));
    i64 judged = 0, zero = 0, air = 0;
    for (i64 i = 0; i < n; i++) {
        judged += run.qa[i] < 254;
        zero += run.qa[i] == 1;
        air += run.qa[i] == 2;
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "name", json_object_new_string(name));
    json_object_object_add(out, "vertices", json_object_new_int64(n));
    json_object_object_add(out, "judged", json_object_new_int64(judged));
    bench_ratio(out, "zero_ct", zero, judged, 4);
    bench_ratio(out, "in_air", air, judged, 4);
    json_object_put(truth.record);
    a->used = saved;
    return out;
}

int bench_qa_vertices_command(int argc, char **argv) {
    need(argc > 0, "calibrate.bench.qa vertices CONTROL ...");
    Arena a = arena(16ull << 30);
    for (int i = 0; i < argc; i++) {
        json_object *out = qa_vertices(&a, argv[i]);
        puts(board_json(out));
        json_object_put(out);
    }
    munmap(a.base, a.size);
    return 0;
}

typedef struct {
    BenchTruth truth;
    json_object *names;
    const i64 *wrap;
    int meshes, main_frame, frames;
    int *frame;
} BenchQAMeshes;

static BenchQAMeshes qa_open(Arena *a, const char *folder) {
    const char *p = fmt(a, "%s/truth.npz", folder);
    BenchQAMeshes out = {.truth = bench_truth(a, folder), .names = legacy_names(a, p)};
    out.meshes = board_count(out.names);
    out.wrap = bench_array(a, p, "mesh_wrap", "<i8", out.meshes).data;
    out.frame = push(a, out.meshes * sizeof(*out.frame));
    const BenchTruth *t = &out.truth;
    size_t saved = a->used;
    int max = -1;
    for (i64 i = 0; i < t->count; i++) {
        max = t->frame[i] > max ? t->frame[i] : max;
    }
    int width = max + 1;
    i64 *seen = pushz(a, out.meshes * (i64)width * sizeof(*seen));
    i64 *used = pushz(a, width * sizeof(*used));
    for (i64 i = 0; i < t->count; i++) {
        int f = t->frame[i], m = t->mesh[i];
        if (f >= 0) {
            need(m >= 0 && m < out.meshes, "invalid QA truth mesh %d", m);
            seen[m * (i64)width + f]++;
            used[f]++;
        }
    }
    out.main_frame = -1;
    for (int f = 0; f < width; f++) {
        if (used[f] && (out.main_frame < 0 || used[f] > used[out.main_frame])) {
            out.main_frame = f;
        }
    }
    memset(used, 0, width * sizeof(*used));
    for (int m = 0; m < out.meshes; m++) {
        int frame = -1;
        i64 most = 0;
        for (int f = 0; f < width; f++) {
            i64 vote = seen[m * (i64)width + f];
            if (vote > most) {
                most = vote;
                frame = f;
            }
        }
        out.frame[m] = frame;
        if (frame >= 0 && !used[frame]++) {
            out.frames++;
        }
    }
    a->used = saved;
    return out;
}

typedef struct {
    i64 n;
    const f32 *z, *xy;
    const double *w;
    const i16 *mesh;
    const i32 *col;
    const u8 *flags;
} BenchQACrossings;

static BenchQACrossings qa_crossings(Arena *a, const char *folder) {
    const char *p = fmt(a, "%s/crossings.npz", folder);
    Array z = bench_array(a, p, "z", "<f4", -1);
    i64 n = elements(z);
    return (BenchQACrossings){.n = n,
        .z = z.data,
        .xy = bench_array(a, p, "xy", "<f4", n * 2).data,
        .w = bench_array(a, p, "winding", "<f8", n).data,
        .mesh = bench_array(a, p, "mesh", "<i2", n).data,
        .col = bench_array(a, p, "col", "<i4", n).data,
        .flags = bench_array(a, p, "flags", "|u1", n).data};
}

static inline double qa_theta(const BenchQACrossings *c, const BenchTruth *t, i64 i) {
    double x = interpolated(c->z[i], t->knots, t->nknots, 1);
    double y = interpolated(c->z[i], t->knots, t->nknots, 2);
    return atan2(c->xy[2 * i + 1] - y, c->xy[2 * i] - x);
}

typedef struct {
    int winding, first, last;
    i64 count;
} QAGap;

static int qa_gap_order(const void *left, const void *right) {
    const QAGap *a = left, *b = right;
    if (a->count != b->count) {
        return a->count > b->count ? -1 : 1;
    }
    return a->winding - b->winding;
}

static json_object *qa_gaps(Arena *a, const BenchQAMeshes *m, const BenchQACrossings *c) {
    size_t saved = a->used;
    json_object *out = json_object_new_object(), *frames = json_object_new_object();
    if (!c->n) {
        json_object_object_add(out, "by_frame", frames);
        bench_ratio(out, "missing_share", 0, 1, 4);
        return out;
    }
    i32 *plane = push(a, c->n * sizeof(*plane)), *z = push(a, c->n * sizeof(*z));
    double *theta = push(a, c->n * sizeof(*theta));
    int planes = 0, max_frame = -1;
    for (int i = 0; i < m->meshes; i++) {
        max_frame = m->frame[i] > max_frame ? m->frame[i] : max_frame;
    }
    int *low = push(a, (max_frame + 1) * sizeof(*low)),
        *high = push(a, (max_frame + 1) * sizeof(*high));
    for (int i = 0; i <= max_frame; i++) {
        low[i] = INT32_MAX;
        high[i] = INT32_MIN;
    }
    for (i64 i = 0; i < c->n; i++) {
        if (!i || c->z[i] != c->z[i - 1]) {
            z[planes++] = nearbyint(c->z[i]);
        }
        plane[i] = planes - 1;
        theta[i] = qa_theta(c, &m->truth, i);
        int f = m->frame[c->mesh[i]];
        if (f >= 0 && isfinite(c->w[i])) {
            int lo = floor(c->w[i] - 1.5), hi = ceil(c->w[i] + 1.5);
            low[f] = lo < low[f] ? lo : low[f];
            high[f] = hi > high[f] ? hi : high[f];
        }
    }
    size_t scratch = a->used;
    i64 span_total = 0, missing_total = 0, cells = planes * 360LL;
    for (int f = 0; f <= max_frame; f++) {
        if (low[f] > high[f]) {
            continue;
        }
        i64 width = high[f] - low[f] + 1LL;
        u8 *table = pushz(a, cells * width);
        for (i64 i = 0; i + 1 < c->n; i++) {
            if (plane[i] != plane[i + 1] || c->mesh[i] != c->mesh[i + 1] ||
                c->col[i + 1] != c->col[i] + 1 || m->frame[c->mesh[i]] != f ||
                !isfinite(c->w[i] + c->w[i + 1])) {
                continue;
            }
            double theta_a = theta[i];
            double theta_b = theta[i + 1];
            double left = (theta_a + M_PI) * (180 / M_PI);
            double span = legacy_relative(theta_b, theta_a) * (180 / M_PI);
            double right = left + span;
            int first = floor(fmin(left, right)), last = floor(fmax(left, right));
            for (int sector = first; sector <= last; sector++) {
                double part = fabs(span) > 1e-9 ? (sector + 0.5 - left) / span : 0;
                part = fmax(0, fmin(part, 1));
                double w = c->w[i] + part * (c->w[i + 1] - c->w[i]);
                int sheet = nearbyint(w - m->truth.sense * ((sector + 0.5) / 360 - 0.5));
                int s = (sector + 360) % 360;
                i64 pos = ((i64)plane[i] * 360 + s) * width + sheet - low[f];
                table[pos] = 1;
            }
        }
        QAGap *gap = pushz(a, width * sizeof(*gap));
        for (i64 j = 0; j < width; j++) {
            gap[j].winding = low[f] + j;
            gap[j].first = INT32_MAX;
            gap[j].last = INT32_MIN;
        }
        i64 cell_n = 0, spanned = 0, missing = 0;
        for (i64 i = 0; i < cells; i++) {
            u8 *row = table + i * width;
            i64 first = 0, last = width - 1;
            while (first < width && !row[first]) {
                first++;
            }
            if (first == width) {
                continue;
            }
            while (!row[last]) {
                last--;
            }
            cell_n++;
            spanned += last - first + 1;
            int layer = z[i / 360];
            for (i64 j = first + 1; j < last; j++) {
                if (!row[j]) {
                    missing++;
                    gap[j].count++;
                    gap[j].first = layer < gap[j].first ? layer : gap[j].first;
                    gap[j].last = layer > gap[j].last ? layer : gap[j].last;
                }
            }
        }
        if (cell_n) {
            json_object *entry = json_object_new_object(), *worst = json_object_new_array();
            json_object_object_add(frames, fmt(a, "%d", f), entry);
            json_object_object_add(entry, "cells", json_object_new_int64(cell_n));
            json_object_object_add(entry, "spanned", json_object_new_int64(spanned));
            json_object_object_add(entry, "missing", json_object_new_int64(missing));
            json_object_object_add(entry, "most_missing", worst);
            qsort(gap, width, sizeof(*gap), qa_gap_order);
            for (int i = 0; i < width && i < 6 && gap[i].count; i++) {
                QAGap g = gap[i];
                json_object *row = json_object_new_object();
                json_object_array_add(worst, row);
                json_object_object_add(row, "winding", json_object_new_int(g.winding));
                json_object_object_add(row, "cells", json_object_new_int64(g.count));
                json_object_object_add(row, "z", package_pair(g.first, g.last));
            }
        }
        span_total += spanned;
        missing_total += missing;
        a->used = scratch;
    }
    bench_ratio(out, "missing_share", missing_total, span_total ? span_total : 1, 4);
    json_object_object_add(out, "by_frame", frames);
    a->used = saved;
    return out;
}

typedef struct {
    i32 a, b, delta;
    i64 contact, differ, peak;
    double share;
} QAContact;

static int qa_contact_order(const void *left, const void *right) {
    const QAContact *a = left, *b = right;
    double x = a->share * a->contact, y = b->share * b->contact;
    if (x != y) {
        return x > y ? -1 : 1;
    }
    return a->a != b->a ? a->a - b->a : a->b - b->b;
}

static double qa_coincident(Arena *a, const BenchTruth *t, int first, int second) {
    size_t saved = a->used;
    i64 size = 0, n = 0, total = 0, contact = 0;
    for (i64 i = 0; i < t->count; i++) {
        size += t->mesh[i] == second;
    }
    f32 *points = push(a, size * 3 * sizeof(*points));
    i32 *index = push(a, size * sizeof(*index));
    for (i64 i = 0; i < t->count; i++) {
        if (t->mesh[i] == second) {
            memcpy(points + 3 * n, t->points + 3 * i, 3 * sizeof(*points));
            index[n++] = i;
        }
    }
    BenchTree tree = bench_tree(a, points, n);
    double depth = 16 / t->voxel, reach = depth + 0.75 * t->spacing / t->voxel;
    for (i64 i = 0; i < t->count; i++) {
        if (t->mesh[i] != first) {
            continue;
        }
        total++;
        BenchCandidate hit;
        int got = 0;
        bench_candidates_limit(&tree, 0, t->points + 3 * i, reach * reach, &hit, &got, 1);
        if (got) {
            int j = index[hit.index];
            double along =
                package_projection(t->points + 3 * i, t->points + 3LL * j, t->normals + 3 * i);
            contact += fabs(along) <= depth && nearbyint(t->winding[j] - t->winding[i]) != 0;
        }
    }
    a->used = saved;
    return package_round((double)contact / (total ? total : 1), 3);
}

static json_object *qa_contacts(
    Arena *a, const BenchQAMeshes *m, double *apart, int *other, i64 *contacts, int *partner) {
    const BenchTruth *t = &m->truth;
    size_t saved = a->used;
    i32 *pick = push(a, t->count * sizeof(*pick));
    i64 usable = 0;
    for (i64 i = 0; i < t->count; i++) {
        if (isfinite(t->winding[i]) && t->frame[i] >= 0 &&
            package_good_normal(t->normals + 3 * i)) {
            pick[usable++] = i;
        }
    }
    i64 count = usable < 300000 ? usable : 300000;
    Pcg64 rng = pcg64(0);
    package_choose(a, &rng, pick, usable, count);
    BenchTree tree = bench_tree(a, t->points, t->count);
    double depth = 16 / t->voxel, lateral = 0.75 * t->spacing / t->voxel;
    double reach = depth * depth + lateral * lateral;
    PackageVotes votes = {0};
    for (i64 at = 0; at < count; at++) {
        i32 i = pick[at];
        BenchCandidate hit[24];
        int n = 0, first = t->mesh[i];
        const f32 *p = t->points + 3LL * i, *normal = t->normals + 3LL * i;
        bench_candidates_limit(&tree, 0, p, reach, hit, &n, 24);
        for (int j = 0; j < n; j++) {
            i32 k = hit[j].index;
            int second = t->mesh[k];
            if (first == second || t->frame[k] != t->frame[i] || !isfinite(t->winding[k])) {
                continue;
            }
            double along = package_projection(p, t->points + 3LL * k, normal);
            if (fabs(along) <= depth && hit[j].squared - along * along <= lateral * lateral) {
                int delta = nearbyint(t->winding[k] - t->winding[i]);
                if (first < second) {
                    package_vote_add(&votes, first, second, delta, 0, 1);
                } else {
                    package_vote_add(&votes, second, first, -delta, 0, 1);
                }
            }
        }
    }
    package_votes_compact(&votes);
    QAContact *pairs = push(a, votes.count * sizeof(*pairs));
    i64 pairs_n = 0;
    for (i64 i = 0, j; i < votes.count; i = j) {
        PackageVote key = votes.slot[i];
        QAContact pair = {.a = key.a, .b = key.b};
        for (j = i; j < votes.count && votes.slot[j].a == key.a && votes.slot[j].b == key.b; j++) {
            PackageVote v = votes.slot[j];
            pair.contact += v.count;
            pair.differ += v.delta ? v.count : 0;
            if (v.count > pair.peak) {
                pair.peak = v.count;
                pair.delta = v.delta;
            }
        }
        if (pair.contact >= 100) {
            pair.share = package_round((double)pair.differ / pair.contact, 3);
            pairs[pairs_n++] = pair;
        }
    }
    free(votes.slot);
    qsort(pairs, pairs_n, sizeof(*pairs), qa_contact_order);
    json_object *out = json_object_new_array();
    for (i64 i = 0; i < pairs_n; i++) {
        QAContact p = pairs[i];
        int a0 = p.a, b0 = p.b;
        if (p.contact > contacts[a0]) {
            contacts[a0] = p.contact;
            partner[a0] = b0;
        }
        if (p.contact > contacts[b0]) {
            contacts[b0] = p.contact;
            partner[b0] = a0;
        }
        if (p.share < 0.5 || p.contact < 300) {
            continue;
        }
        double x = qa_coincident(a, t, a0, b0), y = qa_coincident(a, t, b0, a0);
        if (x > apart[a0]) {
            apart[a0] = x;
            other[a0] = b0;
        }
        if (y > apart[b0]) {
            apart[b0] = y;
            other[b0] = a0;
        }
        json_object *row = json_object_new_object(), *meshes = json_object_new_array_ext(2);
        json_object_array_add(out, row);
        json_object_object_add(row, "mesh_index", package_pair(a0, b0));
        json_object_array_add(meshes, json_object_new_string(board_item(m->names, a0)));
        json_object_array_add(meshes, json_object_new_string(board_item(m->names, b0)));
        json_object_object_add(row, "meshes", meshes);
        json_object_object_add(row, "wraps", package_pair(m->wrap[a0], m->wrap[b0]));
        json_object_object_add(row, "contacts", json_object_new_int64(p.contact));
        bench_decimal(row, "differ_share", p.share, 3);
        json_object_object_add(row, "mode_difference", json_object_new_int(p.delta));
        bench_ratio(row, "mode_share", p.peak, p.contact, 3);
        json_object_object_add(row, "coincident_share", package_pair(x, y));
    }
    a->used = saved;
    return out;
}

typedef struct {
    i64 n, kept, qa, zero, air, bits[5];
    double area, low, high;
    int z0, z1;
} QAMeasures;

static const char *const qa_bits[5] = {"pressed", "touching", "conflict", "order", "overlap"};

static void qa_count_vertex(QAMeasures *out, const BenchTruth *t, i64 i, const u8 *qa) {
    out->n++;
    int flag = t->flags[i], z = t->points[3 * i + 2];
    out->z0 = z < out->z0 ? z : out->z0;
    out->z1 = z > out->z1 ? z : out->z1;
    out->low = fmin(out->low, t->winding[i]);
    out->high = fmax(out->high, t->winding[i]);
    if (!(flag & BENCH_DUPLICATE)) {
        const int bits[] = {2, 128, 32, 64, 4 | 8 | 16};
        out->kept++;
        out->area += t->area[i];
        for (int j = 0; j < 5; j++) {
            out->bits[j] += (flag & bits[j]) != 0;
        }
    }
    if (qa && qa[i] < 254) {
        out->qa++;
        out->zero += qa[i] == 1;
        out->air += qa[i] == 2;
    }
}

static void qa_shares(json_object *out, const QAMeasures *m) {
    for (int i = 0; i < 5; i++) {
        bench_ratio(out, qa_bits[i], m->bits[i], m->kept, 4);
    }
}

typedef struct {
    json_object *link;
    double purity;
    int order;
} QAChain;

static int qa_chain_order(const void *left, const void *right) {
    const QAChain *a = left, *b = right;
    if (a->purity != b->purity) {
        return a->purity < b->purity ? -1 : 1;
    }
    return a->order - b->order;
}

static json_object *qa_chain(Arena *a, json_object *record) {
    size_t saved = a->used;
    json_object *chain = jget(record, "chain"), *links = jget(chain, "links");
    json_object *out = json_object_new_object(), *weakest = json_object_new_array(),
                *unlinked = json_object_new_array();
    QAChain *order = push(a, board_count(links) * sizeof(*order));
    int n = 0;
    for (int i = 0; i < board_count(links); i++) {
        json_object *r = json_object_array_get_idx(links, i);
        if (ji64(r, "votes")) {
            order[n++] = (QAChain){r, jf64(r, "purity"), i};
        } else {
            json_object_array_add(unlinked, package_pair(ji64(r, "from"), ji64(r, "to")));
        }
    }
    qsort(order, n, sizeof(*order), qa_chain_order);
    for (int i = 0; i < n && i < 3; i++) {
        json_object_array_add(weakest, json_object_get(order[i].link));
    }
    json_object_object_add(out, "runs", json_object_get(jget(chain, "runs")));
    json_object_object_add(out, "links", json_object_new_int(n));
    json_object_object_add(out, "unlinked", unlinked);
    json_object_object_add(out, "purity_min", n ? jnum(order[0].purity) : NULL);
    json_object_object_add(out, "purity_median",
        n ? jnum((order[n / 2].purity + order[(n - 1) / 2].purity) / 2) : NULL);
    json_object_object_add(out, "weakest", weakest);
    a->used = saved;
    return out;
}

static double qa_profile_contrast(const f32 *mean) {
    f32 low = INFINITY, high = -INFINITY, sum = 0;
    for (int d = 0; d < RIDGE_POINTS; d++) {
        low = fminf(low, mean[d]);
        high = fmaxf(high, mean[d]);
        sum += mean[d];
    }
    f32 level = sum / RIDGE_POINTS;
    return level > 1 ? package_round((double)(high - low) / level, 3) : NAN;
}

typedef struct {
    i32 mesh, z, y, x, index;
} QARidgeCell;

static int qa_ridge_cell_order(const void *left, const void *right) {
    const QARidgeCell *a = left, *b = right;
    if (a->mesh != b->mesh) {
        return a->mesh - b->mesh;
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

static int qa_ridge_cell_eq(QARidgeCell a, QARidgeCell b) {
    return a.mesh == b.mesh && a.z == b.z && a.y == b.y && a.x == b.x;
}

typedef struct {
    i64 in, empty, air, valid;
    f32 profile[RIDGE_POINTS];
} QARidgeMesh;

typedef struct {
    double contrast, ridge;
    i64 n;
    int mesh, position[3], order;
} QAWeakCell;

static int qa_weak_order(const void *left, const void *right) {
    const QAWeakCell *a = left, *b = right;
    if (a->contrast != b->contrast) {
        return a->contrast < b->contrast ? -1 : 1;
    }
    return a->order - b->order;
}

static void qa_ridge_numbers(
    Arena *a, const char *name, const BenchQAMeshes *m, json_object **rows, json_object *out) {
    size_t saved = a->used;
    const char *base = fmt(a, "%s/bench/checks/%s-ridge", OUTPUTS, name);
    const char *path = fmt(a, "%s.npz", base);
    if (!exists(path)) {
        return;
    }
    const BenchTruth *t = &m->truth;
    BenchPapyrus papyrus = qa_papyrus(a, path);
    Array profile = npz_get(a, path, "profiles"), steps = npz_get(a, path, "offsets_vox");
    i64 n = profile.shape[0];
    const u16 *raw = profile.data;
    const f32 *offset = steps.data;
    const i64 *index = bench_array(a, path, "truth_index", "<i8", n).data;
    const f32 *found = bench_array(a, path, "ridge_sigma3_vox", "<f4", n).data;
    f32 *ridges = push(a, n * 2 * sizeof(*ridges));
    QARidgeCell *cell = push(a, n * sizeof(*cell));
    QARidgeMesh *by_mesh = pushz(a, m->meshes * sizeof(*by_mesh));
    double *values = push(a, n * 2 * sizeof(*values)), *contrasts = push(a, n * sizeof(*contrasts));
    i64 count = 0, empty_total = 0, air_total = 0;
    for (i64 i = 0; i < n; i++) {
        i64 j = index[i];
        need(j >= 0 && j < t->count, "stale ridge QA sample for %s", name);
        int mesh = t->mesh[j], empty = 1;
        const f32 *p = t->points + 3 * j;
        QARidgeMesh *acc = by_mesh + mesh;
        f32 input[RIDGE_POINTS], smooth[RIDGE_POINTS];
        for (int d = 0; d < RIDGE_POINTS; d++) {
            input[d] = qa_half(raw[i * RIDGE_POINTS + d]);
            if (fabsf(offset[d]) <= 3 && input[d] != 0) {
                empty = 0;
            }
        }
        acc->in++;
        acc->empty += empty;
        empty_total += empty;
        if (empty) {
            continue;
        }
        cell[count++] =
            (QARidgeCell){mesh, floor(p[2] / 128.0), floor(p[1] / 128.0), floor(p[0] / 128.0), i};
        ridge_smooth(input, RIDGE_POINTS, 1, smooth);
        f32 best = -INFINITY;
        for (int d = 0; d < RIDGE_POINTS; d++) {
            acc->profile[d] += input[d];
            if (fabs(offset[d] - papyrus.centre) <= 3) {
                best = fmaxf(best, smooth[d]);
            }
        }
        int air = best < papyrus.threshold;
        acc->air += air;
        air_total += air;
        acc->valid += isfinite(found[i]);
        ridge_smooth(input, RIDGE_POINTS, 3, smooth);
        ridges[2 * i] = ridge_peak(smooth, 36);
        ridges[2 * i + 1] = ridge_peak(smooth, 12);
    }
    qsort(cell, count, sizeof(*cell), qa_ridge_cell_order);
    QAWeakCell *weak = push(a, count * sizeof(*weak));
    i64 weak_n = 0;
    for (i64 i = 0, end; i < count; i = end) {
        int mesh = cell[i].mesh;
        QARidgeMesh *acc = by_mesh + mesh;
        i64 n_cells = 0;
        for (end = i; end < count && cell[end].mesh == mesh;) {
            i64 stop = end + 1;
            while (stop < count && qa_ridge_cell_eq(cell[stop], cell[end])) {
                stop++;
            }
            i64 used = stop - end;
            if (used >= 20) {
                f32 mean[RIDGE_POINTS] = {0};
                for (i64 p = end; p < stop; p++) {
                    for (int d = 0; d < RIDGE_POINTS; d++) {
                        mean[d] += qa_half(raw[cell[p].index * RIDGE_POINTS + d]);
                    }
                }
                for (int d = 0; d < RIDGE_POINTS; d++) {
                    mean[d] /= (f32)used;
                }
                double contrast = qa_profile_contrast(mean);
                if (isfinite(contrast)) {
                    contrasts[n_cells++] = contrast;
                    if (used >= 40) {
                        QAWeakCell w = {
                            .contrast = contrast, .n = used, .mesh = mesh, .order = weak_n};
                        i64 valid = 0;
                        for (i64 p = end; p < stop; p++) {
                            f32 f = found[cell[p].index];
                            if (isfinite(f)) {
                                values[valid++] = fabsf(f);
                            }
                        }
                        w.ridge = bench_median(values, valid, NAN);
                        for (int d = 0; d < 3; d++) {
                            for (i64 k = 0; k < used; k++) {
                                values[k] = t->points[3 * index[cell[end + k].index] + d];
                            }
                            w.position[d] = bench_median(values, used, 0);
                        }
                        weak[weak_n++] = w;
                    }
                }
            }
            end = stop;
        }
        if (acc->valid < 20) {
            continue;
        }
        json_object *row = rows[mesh];
        i64 k = 0;
        for (i64 j = i; j < end; j++) {
            f32 f = found[cell[j].index];
            if (isfinite(f)) {
                values[k++] = f;
            }
        }
        bench_decimal(row, "ridge_signed_p50_vox", bench_median(values, k, NAN), 2);
        for (i64 j = 0; j < k; j++) {
            values[j] = fabs(values[j]);
        }
        json_object_object_add(row, "ridge_abs_p50_p90_vox", ridge_abs_quantiles(values, k));
        k = 0;
        for (i64 j = i; j < end; j++) {
            i64 p = cell[j].index;
            for (int d = 0; d < 2; d++) {
                f32 v = ridges[2 * p + d];
                if (isfinite(v)) {
                    values[k++] = fabsf(v);
                }
            }
        }
        double null = bench_median(values, k, NAN);
        json_object_object_add(row, "null_abs_p50_vox", isfinite(null) ? jnum(null) : NULL);
        int peak = 0;
        for (int d = 0; d < RIDGE_POINTS; d++) {
            acc->profile[d] /= (f32)(end - i);
            if (acc->profile[d] > acc->profile[peak]) {
                peak = d;
            }
        }
        double pooled = qa_profile_contrast(acc->profile);
        double contrast =
            n_cells ? package_round(bench_median(contrasts, n_cells, NAN), 3) : pooled;
        json_object_object_add(row, "profile_contrast", isfinite(contrast) ? jnum(contrast) : NULL);
        json_object_object_add(row, "pooled_contrast", isfinite(pooled) ? jnum(pooled) : NULL);
        bench_decimal(row, "profile_peak_vox", offset[peak], 1);
        bench_ratio(row, "ct_empty_share", acc->empty, acc->in, 4);
        bench_ratio(row, "air_share", acc->air, end - i, 4);
        json_object_object_add(row, "sampled", json_object_new_int64(end - i));
    }
    qsort(weak, weak_n, sizeof(*weak), qa_weak_order);
    json_object *summary = json_object_new_object(), *worst = json_object_new_array();
    i64 flat = 0;
    for (i64 j = 0; j < weak_n; j++) {
        flat += weak[j].contrast < 0.1;
        if (j < 12) {
            QAWeakCell w = weak[j];
            json_object *entry = json_object_new_object(), *position = json_object_new_array_ext(3);
            json_object_array_add(worst, entry);
            json_object_object_add(entry, "mesh", json_object_new_int(w.mesh));
            json_object_object_add(
                entry, "segment", json_object_new_string(board_item(m->names, w.mesh)));
            for (int d = 0; d < 3; d++) {
                json_object_array_add(position, json_object_new_int(w.position[d]));
            }
            json_object_object_add(entry, "centre_xyz", position);
            json_object_object_add(entry, "vertices", json_object_new_int64(w.n));
            bench_decimal(entry, "contrast", w.contrast, 3);
            json_object_object_add(
                entry, "ridge_abs_p50_vox", isfinite(w.ridge) ? jnum(w.ridge) : NULL);
        }
    }
    json_object_object_add(summary, "cells", json_object_new_int64(weak_n));
    json_object_object_add(summary, "under_0.1", json_object_new_int64(flat));
    json_object *quantiles = NULL;
    if (weak_n) {
        for (i64 i = 0; i < weak_n; i++) {
            contrasts[i] = weak[i].contrast;
        }
        quantiles = package_pair(package_round(package_quantile(contrasts, weak_n, 0.1), 3),
            package_round(package_quantile(contrasts, weak_n, 0.5), 3));
    }
    json_object_object_add(summary, "contrast_p10_p50", quantiles);
    json_object_object_add(out, "mesh_chunk_cells", summary);
    json_object_object_add(out, "weakest_cells", worst);
    json_object *record = json_read(fmt(a, "%s.json", base));
    if (record) {
        json_object *source = jget(jget(record, "by_group"), "all"),
                    *ridge = json_object_new_object(), *sigma = json_object_new_object();
        bench_ratio(ridge, "ct_empty_share", empty_total, n, 4);
        bench_ratio(ridge, "air_share", air_total, n - empty_total, 4);
        json_object_object_add(ridge, "sampled", json_object_get(jget(source, "vertices")));
        json_object_object_add(
            ridge, "mean_profile_peak_vox", json_object_get(jget(source, "mean_profile_peak_vox")));
        json_object_object_foreach(jget(source, "sigma_3"), key, value) {
            if (strstr(key, "abs") || strstr(key, "within")) {
                json_object_object_add(sigma, key, json_object_get(value));
            }
        }
        json_object_object_add(ridge, "sigma3", sigma);
        json_object_object_add(out, "ridge", ridge);
        json_object_put(record);
    }
    a->used = saved;
}

static json_object *qa_numbers(Arena *a, const char *name) {
    size_t saved = a->used;
    const char *folder = fmt(a, "%s/bench/controls/%s", OUTPUTS, name);
    BenchQAMeshes m = qa_open(a, folder);
    BenchQACrossings cross = qa_crossings(a, folder);
    const BenchTruth *t = &m.truth;
    const char *path = fmt(a, "%s/bench/qa/%s/vertices.npz", OUTPUTS, name);
    const u8 *vertex_qa = exists(path) ? bench_array(a, path, "qa", "|u1", t->count).data : NULL;
    QAMeasures total = {.low = INFINITY, .high = -INFINITY, .z0 = INT32_MAX, .z1 = INT32_MIN};
    QAMeasures *each = push(a, m.meshes * sizeof(*each));
    for (int i = 0; i < m.meshes; i++) {
        each[i] = total;
    }
    for (i64 i = 0; i < t->count; i++) {
        qa_count_vertex(&total, t, i, vertex_qa);
        qa_count_vertex(each + t->mesh[i], t, i, vertex_qa);
    }
    double *apart = pushz(a, m.meshes * sizeof(*apart));
    int *other = pushz(a, m.meshes * sizeof(*other)),
        *partner = pushz(a, m.meshes * sizeof(*partner));
    i64 *contacts = pushz(a, m.meshes * sizeof(*contacts));
    json_object *out = json_object_new_object(), *shares = json_object_new_object(),
                *detail = json_object_new_array_ext(m.meshes);
    json_object *pairs = qa_contacts(a, &m, apart, other, contacts, partner);
    json_object_object_add(out, "name", json_object_new_string(name));
    json_object_object_add(out, "scan", json_object_get(jget(t->record, "scan")));
    json_object_object_add(out, "z", json_object_get(jget(jget(t->record, "definition"), "z")));
    json_object_object_add(
        out, "package_version", json_object_get(json_object_object_get(t->record, "version")));
    json_object_object_add(out, "vertices", json_object_new_int64(t->count));
    bench_decimal(out, "area_cm2", total.area / 100, 1);
    json_object_object_add(out, "meshes", json_object_new_int(m.meshes));
    int named = 0;
    for (int j = 0; j < m.meshes; j++) {
        named += m.wrap[j] >= 0;
    }
    json_object_object_add(out, "named", json_object_new_int(named));
    json_object_object_add(out, "frames", json_object_new_int(m.frames));
    json_object_object_add(out, "sense", json_object_new_int(t->sense));
    json_object_object_add(out, "sense_votes", json_object_get(jget(t->record, "sense_votes")));
    bench_ratio(out, "duplicate_share", total.n - total.kept, total.n, 4);
    qa_shares(shares, &total);
    json_object_object_add(out, "shares", shares);
    json_object_object_add(out, "chain", qa_chain(a, t->record));
    json_object_object_add(out, "checks", json_object_get(jget(t->record, "checks")));
    json_object_object_add(out, "zones", json_object_get(jget(t->record, "zones")));
    json_object_object_add(out, "gaps", qa_gaps(a, &m, &cross));
    json_object_object_add(out, "same_surface_pairs", pairs);
    if (total.qa) {
        json_object *record = json_object_new_object();
        json_object_object_add(record, "judged", json_object_new_int64(total.qa));
        bench_ratio(record, "zero_ct", total.zero, total.qa, 4);
        bench_ratio(record, "in_air", total.air, total.qa, 4);
        json_object_object_add(out, "all_vertices", record);
    }
    json_object **rows = push(a, m.meshes * sizeof(*rows));
    for (int j = 0; j < m.meshes; j++) {
        QAMeasures v = each[j];
        json_object *row = rows[j] = json_object_new_object();
        json_object_array_add(detail, row);
        json_object_object_add(row, "mesh", json_object_new_int(j));
        json_object_object_add(row, "segment", json_object_new_string(board_item(m.names, j)));
        json_object_object_add(row, "wrap", json_object_new_int64(m.wrap[j]));
        json_object_object_add(row, "frame", json_object_new_int(m.frame[j]));
        json_object_object_add(row, "vertices", json_object_new_int64(v.n));
        bench_decimal(row, "area_cm2", v.area / 100, 2);
        json_object_object_add(row, "z", package_pair(v.z0, v.z1));
        json_object_object_add(
            row, "turns", isfinite(v.low) ? jnum(package_round(v.high - v.low, 2)) : NULL);
        bench_ratio(row, "duplicate_share", v.n - v.kept, v.n, 4);
        qa_shares(row, &v);
        if (apart[j] > 0) {
            json_object *p = json_object_new_object();
            json_object_object_add(
                p, "segment", json_object_new_string(board_item(m.names, other[j])));
            bench_decimal(p, "share", apart[j], 3);
            json_object_object_add(row, "numbered_apart_on", p);
        }
        if (contacts[j]) {
            json_object *p = json_object_new_object();
            json_object_object_add(
                p, "segment", json_object_new_string(board_item(m.names, partner[j])));
            json_object_object_add(p, "contacts", json_object_new_int64(contacts[j]));
            json_object_object_add(row, "shares_surface_most_with", p);
        }
        if (v.qa >= 50) {
            json_object_object_add(row, "all_vertices_judged", json_object_new_int64(v.qa));
            bench_ratio(row, "all_vertices_zero_ct", v.zero, v.qa, 4);
            bench_ratio(row, "all_vertices_in_air", v.air, v.qa, 4);
        }
    }
    qa_ridge_numbers(a, name, &m, rows, out);
    json_object_object_add(out, "meshes_detail", detail);
    json_write(fmt(a, "%s/bench/qa/%s/numbers.json", OUTPUTS, name), out);
    json_object_put(m.names);
    json_object_put(t->record);
    a->used = saved;
    return out;
}

int bench_qa_numbers_command(int argc, char **argv) {
    need(argc > 0, "calibrate.bench.qa numbers CONTROL ...");
    Arena a = arena(16ull << 30);
    for (int i = 0; i < argc; i++) {
        json_object *out = qa_numbers(&a, argv[i]);
        json_object_object_del(out, "meshes_detail");
        json_object_object_del(out, "zones");
        puts(board_json(out));
        json_object_put(out);
    }
    munmap(a.base, a.size);
    return 0;
}

#include <ft2build.h>
#include FT_FREETYPE_H

enum { QA_TILE = 640 };

static void qa_palette(u8 palette[256][3]) {
    memset(palette, 0, 256 * 3);
    for (int i = 0; i < 240; i++) {
        palette[i][0] = palette[i][1] = palette[i][2] = i * 255 / 239;
    }
    const u8 colors[][3] = {{235, 50, 50}, {40, 210, 40}, {60, 120, 255}, {250, 220, 30},
        {230, 60, 230}, {30, 225, 225}, {255, 255, 255}};
    memcpy(palette[240], colors, sizeof(colors));
}

static void qa_png(const char *path, int w, int h, const u8 *pixels, u8 (*palette)[3]) {
    png_image image = {.version = PNG_IMAGE_VERSION,
        .width = w,
        .height = h,
        .format = palette ? PNG_FORMAT_RGB_COLORMAP : PNG_FORMAT_RGB,
        .colormap_entries = palette ? 256 : 0};
    need(png_image_write_to_file(&image, path, 0, pixels, 0, palette), "cannot write %s: %s", path,
        image.message);
}

static void qa_text(u8 *pixel, int w, int h, int x, int y, const char *label) {
    FT_Library library;
    if (FT_Init_FreeType(&library)) {
        return;
    }
    FT_Face face;
    const char *fonts[] = {"/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"};
    int found = 0;
    for (size_t i = 0; i < sizeof(fonts) / sizeof(*fonts) && !found; i++) {
        found = !FT_New_Face(library, fonts[i], 0, &face);
    }
    if (found) {
        FT_Set_Pixel_Sizes(face, 0, 12);
        for (const unsigned char *p = (const unsigned char *)label; *p; p++) {
            if (FT_Load_Char(face, *p, FT_LOAD_RENDER)) {
                continue;
            }
            FT_GlyphSlot g = face->glyph;
            for (int r = 0; r < (int)g->bitmap.rows; r++) {
                int py = y + 10 - g->bitmap_top + r;
                if (py < 0 || py >= h) {
                    continue;
                }
                for (int c = 0; c < (int)g->bitmap.width; c++) {
                    int px = x + g->bitmap_left + c;
                    if (px >= 0 && px < w) {
                        int a = g->bitmap.buffer[r * g->bitmap.pitch + c];
                        u8 *at = pixel + 3 * ((i64)py * w + px);
                        for (int d = 0; d < 3; d++) {
                            at[d] += ((255 - at[d]) * a + 127) / 255;
                        }
                    }
                }
            }
            x += g->advance.x >> 6;
        }
        FT_Done_Face(face);
    }
    FT_Done_FreeType(library);
}

static double qa_percentile_hist(const i64 *hist, i64 n, double share) {
    double row = (n - 1) * share, mix = row - floor(row);
    i64 first = floor(row), last = ceil(row), count = 0;
    int a = 1, b = 255, found = 0;
    for (int i = 1; i < 256; i++) {
        count += hist[i];
        if (!found && count > first) {
            a = i;
            found = 1;
        }
        if (count > last) {
            b = i;
            break;
        }
    }
    return a + (b - a) * mix;
}

static double qa_lanczos(double d) {
    d = fabs(d);
    if (d == 0) {
        return 1;
    }
    return d >= 3 ? 0 : sin(d * M_PI) * sin(d * M_PI / 3) * 3 / (d * d * M_PI * M_PI);
}

static void qa_resample_row(const u8 *input, u8 *out, int w, int target, i64 stride) {
    double scale = (double)w / target;
    for (int x = 0; x < target; x++) {
        double centre = (x + 0.5) * scale - 0.5;
        int low = fmax(0, ceil(centre - 3 * scale));
        int high = fmin(w - 1, floor(centre + 3 * scale));
        double sum = 0, weight = 0;
        for (int i = low; i <= high; i++) {
            double z = qa_lanczos((i - centre) / scale);
            sum += z * input[i * stride];
            weight += z;
        }
        out[x * stride] = fmax(0, fmin(255, nearbyint(sum / weight)));
    }
}

typedef struct {
    int x, y;
    u8 color;
} QADot;

static void qa_rect(u8 *rgb, int w, int h, double xx, double yy, const u8 *color) {
    int x0 = trunc(xx - 1), y0 = trunc(yy - 1), x1 = trunc(xx + 1), y1 = trunc(yy + 1);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                memcpy(rgb + 3 * ((i64)y * w + x), color, 3);
            }
        }
    }
}

static void qa_overview(Arena *a, const char *path, const char *name, int z, const u8 *gray, int w,
    int h, u8 palette[256][3], const QADot *dot, i64 n) {
    size_t saved = a->used;
    int limit = w > h ? w : h;
    double scale = limit > 1600 ? (double)limit / 1600 : 1;
    int width = nearbyint(w / scale), height = nearbyint(h / scale);
    double shrink = (double)w / width;
    u8 *plain = push(a, w * (i64)h), *across = push(a, width * (i64)h);
    u8 *small = push(a, width * (i64)height), *rgb = push(a, 3 * width * (i64)height);
    for (i64 i = 0; i < w * (i64)h; i++) {
        plain[i] = palette[gray[i]][0];
    }
    for (int y = 0; y < h; y++) {
        qa_resample_row(plain + y * (i64)w, across + y * (i64)width, w, width, 1);
    }
    for (int x = 0; x < width; x++) {
        qa_resample_row(across + x, small + x, h, height, width);
    }
    for (i64 i = 0; i < (i64)width * height; i++) {
        memset(rgb + 3 * i, small[i], 3);
    }
    for (i64 i = 0; i < n; i++) {
        qa_rect(rgb, width, height, dot[i].x / shrink, dot[i].y / shrink, palette[dot[i].color]);
    }
    qa_text(rgb, width, height, 8, 8, fmt(a, "%s z %d (1:%.1f)", name, z, shrink));
    qa_png(path, width, height, rgb, NULL);
    a->used = saved;
}

typedef struct {
    i32 x, y, dot;
} QATileCell;

static int qa_tile_cell_order(const void *left, const void *right) {
    const QATileCell *a = left, *b = right;
    return a->x != b->x ? a->x - b->x : a->y != b->y ? a->y - b->y : a->dot - b->dot;
}

typedef struct {
    const char *why;
    double x, y;
} QATile;

static int qa_tiles(Arena *a, const QADot *dot, i64 n, double cx, double cy, QATile out[6]) {
    size_t saved = a->used;
    QATileCell *cell = push(a, n * sizeof(*cell));
    double *values = push(a, n * sizeof(*values));
    i64 count = 0, best[2] = {-1, -1}, size[2] = {0}, used = 0;
    for (i64 i = 0; i < n; i++) {
        if (dot[i].color == 246) {
            cell[count++] = (QATileCell){
                floor(dot[i].x / (double)QA_TILE), floor(dot[i].y / (double)QA_TILE), i};
        }
    }
    qsort(cell, count, sizeof(*cell), qa_tile_cell_order);
    for (i64 i = 0, end; i < count; i = end) {
        for (end = i + 1; end < count && cell[end].x == cell[i].x && cell[end].y == cell[i].y;
            end++) {
        }
        i64 n = end - i;
        if (n > size[0]) {
            size[1] = size[0];
            best[1] = best[0];
            size[0] = n;
            best[0] = i;
        } else if (n > size[1]) {
            size[1] = n;
            best[1] = i;
        }
    }
    for (int i = 0; i < 2 && size[i]; i++) {
        for (i64 j = 0; j < size[i]; j++) {
            values[j] = dot[cell[best[i] + j].dot].x;
        }
        double x = bench_median(values, size[i], 0);
        for (i64 j = 0; j < size[i]; j++) {
            values[j] = dot[cell[best[i] + j].dot].y;
        }
        out[used++] = (QATile){"flagged", x, bench_median(values, size[i], 0)};
    }
    const char *names[] = {"angle0", "angle90", "angle180", "angle270"};
    for (int d = 0; d < 4; d++) {
        double angle = d * M_PI / 2;
        count = 0;
        for (i64 i = 0; i < n; i++) {
            double x = dot[i].x - cx, y = dot[i].y - cy;
            if (fabs(legacy_relative(atan2(y, x), angle)) < M_PI / 18) {
                values[count++] = hypot(x, y);
            }
        }
        if (count) {
            double r = bench_median(values, count, 0);
            out[used++] = (QATile){names[d], cx + r * cos(angle), cy + r * sin(angle)};
        }
    }
    a->used = saved;
    return used;
}

static json_object *qa_plane(Arena *a, const char *name, const BenchQAMeshes *m,
    const BenchQACrossings *c, const RidgeVolume *volume, int z, int tiles) {
    size_t saved = a->used;
    const BenchTruth *t = &m->truth;
    need(c->n > 0, "%s has no crossing planes", name);
    i64 lo = 0, hi = c->n;
    while (lo < hi) {
        i64 at = (lo + hi) / 2;
        if (c->z[at] < z) {
            lo = at + 1;
        } else {
            hi = at;
        }
    }
    i64 nearest = lo == c->n ? c->n - 1 : lo > 0 && z - c->z[lo - 1] <= c->z[lo] - z ? lo - 1 : lo;
    int plane = c->z[nearest];
    i64 first = nearest, stop = nearest + 1;
    while (first > 0 && c->z[first - 1] == plane) {
        first--;
    }
    while (stop < c->n && c->z[stop] == plane) {
        stop++;
    }
    double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    for (i64 i = first; i < stop; i++) {
        const f32 *p = c->xy + 2 * i;
        xmin = fmin(xmin, p[0]);
        xmax = fmax(xmax, p[0]);
        ymin = fmin(ymin, p[1]);
        ymax = fmax(ymax, p[1]);
    }
    int x0 = fmax(0, floor(xmin - 64)), x1 = fmin(volume->shape[2], ceil(xmax + 64));
    int y0 = fmax(0, floor(ymin - 64)), y1 = fmin(volume->shape[1], ceil(ymax + 64));
    int w = x1 - x0, h = y1 - y0;
    need(w > 0 && h > 0, "truth crossing box is outside CT");
    i64 bytes = w * (i64)h;
    u8 *gray = push(a, bytes), *colored = push(a, bytes);
    i64 unknown = 0, histogram[256] = {0}, known = 0;
    i64 zc = plane / volume->chunk[0];
    for (i64 y = y0 / volume->chunk[1]; y <= (y1 - 1) / volume->chunk[1]; y++) {
        for (i64 x = x0 / volume->chunk[2]; x <= (x1 - 1) / volume->chunk[2]; x++) {
            unknown += ridge_offset(volume, zc, y, x) == UNKNOWN;
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            u8 v = ridge_voxel(volume, plane, y0 + y, x0 + x);
            gray[y * (i64)w + x] = v;
            histogram[v]++;
            known += v > 0;
        }
    }
    if (!known) {
        memset(gray, 0, bytes);
    } else {
        f32 low = qa_percentile_hist(histogram, known, 0.01);
        f32 width = fmax(qa_percentile_hist(histogram, known, 0.997) - low, 1);
        for (i64 i = 0; i < bytes; i++) {
            gray[i] = fmaxf(0, fminf((gray[i] - low) / width, 1)) * 239;
        }
    }
    memcpy(colored, gray, bytes);
    i64 n = stop - first, flagged = 0;
    QADot *dot = push(a, n * sizeof(*dot));
    double cx = interpolated(plane, t->knots, t->nknots, 1);
    double cy = interpolated(plane, t->knots, t->nknots, 2);
    for (i64 j = 0; j < n; j++) {
        i64 i = first + j;
        double theta = atan2(c->xy[2 * i + 1] - cy, c->xy[2 * i] - cx);
        int sheet = isfinite(c->w[i]) ? nearbyint(c->w[i] - t->sense * theta / (2 * M_PI)) : 0;
        int frame = m->frame[c->mesh[i]], bad = (c->flags[i] & BENCH_SKIP) != 0;
        flagged += bad;
        int color = bad ? 246 : (frame == m->main_frame ? 240 : 243) + (sheet % 3 + 3) % 3;
        dot[j] = (QADot){nearbyint(c->xy[2 * i] - x0), nearbyint(c->xy[2 * i + 1] - y0), color};
    }
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (i64 i = 0; i < n; i++) {
                int x = dot[i].x + dx, y = dot[i].y + dy;
                if (x >= 0 && x < w && y >= 0 && y < h) {
                    colored[y * (i64)w + x] = dot[i].color;
                }
            }
        }
    }
    const char *folder = fmt(a, "%s/bench/qa/%s", OUTPUTS, name);
    mkdirs(folder);
    const char *full = fmt(a, "%s/z%d.png", folder, plane);
    const char *overview = fmt(a, "%s/z%d-overview.png", folder, plane);
    json_object *out = json_object_new_object(), *files = json_object_new_array();
    u8 palette[256][3];
    qa_palette(palette);
    qa_png(full, w, h, colored, palette);
    json_object_array_add(files, json_object_new_string(full));
    qa_overview(a, overview, name, plane, gray, w, h, palette, dot, n);
    json_object_array_add(files, json_object_new_string(overview));
    if (tiles) {
        QATile choices[6];
        int tile_n = qa_tiles(a, dot, n, cx - x0, cy - y0, choices);
        u8 *rgb = push(a, QA_TILE * QA_TILE * 3);
        for (int i = 0; i < tile_n; i++) {
            QATile q = choices[i];
            int half = QA_TILE / 2;
            int a0 = fmin(fmax((int)q.x - half, 0), fmax(w - QA_TILE, 0));
            int b0 = fmin(fmax((int)q.y - half, 0), fmax(h - QA_TILE, 0));
            for (int y = 0; y < QA_TILE; y++) {
                for (int x = 0; x < QA_TILE; x++) {
                    u8 index = y + b0 < h && x + a0 < w ? colored[(y + b0) * (i64)w + x + a0] : 0;
                    memcpy(rgb + 3 * (y * QA_TILE + x), palette[index], 3);
                }
            }
            qa_text(rgb, QA_TILE, QA_TILE, 6, 6,
                fmt(a, "%s z %d x %d y %d (%s)", name, plane, a0 + x0, b0 + y0, q.why));
            const char *path = fmt(a, "%s/z%d-tile%d-%s.png", folder, plane, i, q.why);
            qa_png(path, QA_TILE, QA_TILE, rgb, NULL);
            json_object_array_add(files, json_object_new_string(path));
        }
    }
    json_object *bounds = json_object_new_array_ext(4);
    const int xy[] = {x0, x1, y0, y1};
    for (int d = 0; d < 4; d++) {
        json_object_array_add(bounds, json_object_new_int(xy[d]));
    }
    json_object_object_add(out, "plane", json_object_new_int(plane));
    json_object_object_add(out, "box_x_y", bounds);
    json_object_object_add(out, "crossings", json_object_new_int64(n));
    json_object_object_add(out, "flagged", json_object_new_int64(flagged));
    json_object_object_add(out, "unknown_chunks", json_object_new_int64(unknown));
    json_object_object_add(out, "files", files);
    a->used = saved;
    return out;
}

int bench_qa_planes_command(int argc, char **argv) {
    need(argc > 0, "calibrate.bench.qa planes CONTROL [--z Z ...] [--no-tiles]");
    Arena a = arena(16ull << 30);
    const char *name = argv[0], *folder = fmt(&a, "%s/bench/controls/%s", OUTPUTS, name);
    int n = 0, tiles = 1, *query = push(&a, (argc + 3) * sizeof(*query)), reading_z = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--z")) {
            reading_z = 1;
        } else if (!strcmp(argv[i], "--no-tiles")) {
            tiles = 0;
            reading_z = 0;
        } else {
            need(reading_z, "unknown planes option %s", argv[i]);
            query[n++] = board_value(argv[i]);
        }
    }
    BenchQAMeshes m = qa_open(&a, folder);
    BenchQACrossings c = qa_crossings(&a, folder);
    json_object *record = m.truth.record, *entry = jget(record, "scan");
    if (!n) {
        json_object *range = jget(jget(record, "definition"), "z");
        int lo = json_object_get_int(json_object_array_get_idx(range, 0));
        int hi = json_object_get_int(json_object_array_get_idx(range, 1));
        for (int i = 1; i <= 3; i++) {
            query[n++] = lo + (hi - lo) * i / 4;
        }
    }
    RidgeVolume volume = ridge_volume_open(&a, jstr(entry, "sample"), jstr(entry, "scan"));
    for (int i = 0; i < n; i++) {
        json_object *row = qa_plane(&a, name, &m, &c, &volume, query[i], tiles);
        puts(board_json(row));
        json_object_put(row);
    }
    ridge_volume_close(&volume);
    json_object_put(m.names);
    json_object_put(record);
    munmap(a.base, a.size);
    return 0;
}

static const char *const qa_rule_names[] = {"exclude", "lower weight (0.5)", "include"};
static const char *const qa_rules[][8] = {
    {"CT profile contrast under 0.08, or under 0.15 and half the package's median (not on papyrus)",
        "more than half of its sampled vertices lie in zero CT (outside the published volume's "
        "mask)",
        "more than 35% of its vertices have no papyrus within 3 voxels of the package's papyrus "
        "(PHerc0139: 2-17%)",
        "median distance to the CT ridge more than 3 voxels above the package's median",
        "conflict or order on more than 25% of its non-duplicate vertices",
        "more than 90% of its vertices repeat a larger mesh (a copy; the larger one carries the "
        "labels)"},
    {"CT profile contrast under 0.15, or under 0.2 and three quarters of the package's median (on "
     "a sheet 0.25-0.6)",
        "10-50% of its sampled vertices lie in zero CT",
        "20-35% of its vertices have no papyrus within 3 voxels",
        "median distance to the CT ridge 1.5 to 3 voxels above the package's median",
        "conflict or order on 5-25% of its non-duplicate vertices",
        "more than half of its vertices repeat a larger mesh",
        "30% or more of its vertices lie on another mesh's surface numbered another winding (one "
        "of the two is a copy under the wrong number, and the winding next to them is untraced "
        "there)"},
    {"everything else; ridge figures need at least 200 sampled vertices to count"},
};

static int qa_matches(const char *segment, json_object *filters) {
    for (int i = 0; i < board_count(filters); i++) {
        if (strstr(segment, board_item(filters, i))) {
            return 1;
        }
    }
    return 0;
}

static int qa_selected(const char *segment, json_object *sel) {
    if (qa_matches(segment, json_object_object_get(sel, "module_exclude")) ||
        qa_matches(segment, json_object_object_get(sel, "exclude"))) {
        return 0;
    }
    json_object *module = json_object_object_get(sel, "module_only");
    json_object *only = json_object_object_get(sel, "only");
    return (!board_count(module) || qa_matches(segment, module)) &&
        (!board_count(only) || qa_matches(segment, only));
}

static json_object *qa_selection(Arena *a, const char *name) {
    size_t saved = a->used;
    const char *label = fmt(a, "%s/unwrapping/network-training/labels/%s.npz", OUTPUTS, name);
    Array meta = {0};
    if (exists(label)) {
        zip_t *archive = zip_open(label, ZIP_RDONLY, NULL);
        need(archive != NULL, "cannot open %s", label);
        meta = npz_member(a, archive, label, "meta", 0);
        zip_close(archive);
    }
    if (!meta.data) {
        a->used = saved;
        return NULL;
    }
    int width = (int)strtol(meta.descr + 2, NULL, 10);
    need(width > 0 && (meta.descr[1] == 'U' || meta.descr[1] == 'S'), "invalid labels meta in %s",
        label);
    char *text = push(a, 4LL * width + 1);
    int n = 0;
    for (int i = 0; i < width; i++) {
        u32 cp = meta.descr[1] == 'U' ? ((const u32 *)meta.data)[i] : ((const u8 *)meta.data)[i];
        if (!cp) {
            break;
        }
        n += legacy_utf8(text + n, cp);
    }
    text[n] = 0;
    json_object *out = json_tokener_parse(text);
    need(out != NULL, "invalid label selection JSON in %s", label);
    a->used = saved;
    return out;
}

static json_object *qa_selection_shares(Arena *a, const char *name, json_object *selection) {
    size_t saved = a->used;
    BenchQAMeshes m = qa_open(a, fmt(a, "%s/bench/controls/%s", OUTPUTS, name));
    const BenchTruth *t = &m.truth;
    u8 *keep = push(a, m.meshes), zone[256] = {0};
    i64 *sizes = pushz(a, m.meshes * sizeof(*sizes));
    json_object *zones = json_object_object_get(selection, "zones");
    for (int k = 0; k < board_count(zones); k++) {
        int z = json_object_get_int(json_object_array_get_idx(zones, k));
        if (0 <= z && z < 256) {
            zone[z] = 1;
        }
    }
    json_object *range = json_object_object_get(selection, "z");
    double z0 = board_count(range) ? json_object_get_double(json_object_array_get_idx(range, 0))
                                   : -INFINITY;
    double z1 =
        board_count(range) ? json_object_get_double(json_object_array_get_idx(range, 1)) : INFINITY;
    for (int j = 0; j < m.meshes; j++) {
        keep[j] = qa_selected(board_item(m.names, j), selection);
    }
    i32 *index = push(a, t->count * sizeof(*index));
    i64 n = 0;
    for (i64 i = 0; i < t->count; i++) {
        int j = t->mesh[i];
        double z = t->points[3 * i + 2];
        sizes[j]++;
        if (keep[j] && zone[t->zone[i]] && z >= z0 && z < z1) {
            index[n++] = i;
        }
    }
    Package p = {.voxel = t->voxel,
        .spacing = t->spacing,
        .count = n,
        .grids_n = m.meshes,
        .grids = pushz(a, m.meshes * sizeof(PackageGrid))};
    PackageVertices v = {.points = push(a, n * 3 * sizeof(f32)),
        .normal = push(a, n * 3 * sizeof(f32)),
        .winding = push(a, n * sizeof(double)),
        .mesh = push(a, n * sizeof(i16)),
        .frame = push(a, n * sizeof(i32)),
        .flags = pushz(a, n)};
    int *rank = pushz(a, m.meshes * sizeof(*rank));
    for (int i = 0; i < m.meshes; i++) {
        p.grids[i].wrap = m.wrap[i];
        for (int j = 0; j < m.meshes; j++) {
            rank[i] += sizes[j] > sizes[i] || (sizes[j] == sizes[i] && j < i);
        }
    }
    for (i64 i = 0; i < n; i++) {
        i64 j = index[i];
        memcpy(v.points + 3 * i, t->points + 3 * j, 3 * sizeof(f32));
        memcpy(v.normal + 3 * i, t->normals + 3 * j, 3 * sizeof(f32));
        v.winding[i] = t->winding[j];
        v.mesh[i] = t->mesh[j];
        v.frame[i] = t->frame[j];
    }
    PackageNeighbors adjacent = {
        .package = &p, .vertices = &v, .tree = bench_tree(a, v.points, n), .rank = rank};
    parallel((n + 255) / 256, 4, package_flags_chunk, &adjacent);
    QAMeasures *acc = pushz(a, m.meshes * sizeof(*acc));
    i64 *released = pushz(a, m.meshes * sizeof(*released));
    for (i64 i = 0; i < n; i++) {
        int mesh = v.mesh[i], flag = v.flags[i];
        QAMeasures *x = acc + mesh;
        x->n++;
        released[mesh] +=
            (t->flags[index[i]] & BENCH_SKIP) != 0 && (flag & (BENCH_SKIP | BENCH_DUPLICATE)) == 0;
        if (!(flag & BENCH_DUPLICATE)) {
            x->kept++;
            x->area += t->area[index[i]];
            x->bits[2] += (flag & 32) != 0;
            x->bits[3] += (flag & 64) != 0;
        }
    }
    json_object *out = json_object_new_object();
    for (int i = 0; i < m.meshes; i++) {
        QAMeasures x = acc[i];
        if (!x.n) {
            continue;
        }
        json_object *row = json_object_new_object();
        json_object_object_add(out, board_item(m.names, i), row);
        json_object_object_add(row, "vertices", json_object_new_int64(x.n));
        bench_decimal(row, "area_cm2", x.area / 100, 2);
        bench_ratio(row, "duplicate_share", x.n - x.kept, x.n, 4);
        bench_ratio(row, "conflict", x.bits[2], x.kept, 4);
        bench_ratio(row, "order", x.bits[3], x.kept, 4);
        bench_ratio(row, "dropped_by_other_meshes", released[i], x.n, 4);
    }
    json_object_put(m.names);
    json_object_put(m.truth.record);
    a->used = saved;
    return out;
}

static const char *qa_spelling(json_object *value) { return value ? board_json(value) : "None"; }

static void qa_reason(json_object *reasons, const char *message) {
    json_object_array_add(reasons, json_object_new_string(message));
}

static int qa_judge(
    Arena *a, json_object *mesh, double ref_contrast, double ref_offset, json_object **reasons) {
    json_object *exclude = json_object_new_array(), *lower = json_object_new_array();
    json_object *value = json_object_object_get(mesh, "profile_contrast");
    json_object *offsets = json_object_object_get(mesh, "ridge_abs_p50_p90_vox");
    json_object *offset_value = offsets ? json_object_array_get_idx(offsets, 0) : NULL;
    i64 n = board_integer(mesh, "sampled");
    int sampled = n >= 200 && value != NULL;
    if (sampled) {
        double contrast = json_object_get_double(value),
               offset = json_object_get_double(offset_value);
        json_object *contrast_reference = jnum(ref_contrast), *offset_reference = jnum(ref_offset);
        if (contrast < 0.08 || (contrast < 0.15 && contrast < 0.5 * ref_contrast)) {
            qa_reason(exclude,
                fmt(a, "not on the papyrus: CT profile contrast %s (package median %s)",
                    qa_spelling(value), qa_spelling(contrast_reference)));
        } else if (contrast < 0.15 || (contrast < 0.2 && contrast < 0.75 * ref_contrast)) {
            qa_reason(lower,
                fmt(a, "weak CT profile contrast %s (package median %s)", qa_spelling(value),
                    qa_spelling(contrast_reference)));
        }
        if (offset > ref_offset + 3) {
            qa_reason(exclude,
                fmt(a, "off the sheet: median %s vox to the CT ridge (package median %s)",
                    qa_spelling(offset_value), qa_spelling(offset_reference)));
        } else if (offset > ref_offset + 1.5) {
            qa_reason(lower,
                fmt(a, "median %s vox to the CT ridge (package median %s)",
                    qa_spelling(offset_value), qa_spelling(offset_reference)));
        }
        json_object_put(contrast_reference);
        json_object_put(offset_reference);
    }
    json_object *full = json_object_object_get(mesh, "all_vertices_zero_ct");
    double empty =
        json_object_get_double(full ? full : json_object_object_get(mesh, "ct_empty_share"));
    if ((full || n >= 50) && empty > 0.5) {
        qa_reason(
            exclude, fmt(a, "CT masked out: %.0f%% of its vertices lie in zero CT", empty * 100));
    } else if ((full || n >= 50) && empty > 0.1) {
        qa_reason(lower,
            fmt(a, "%.0f%% of its vertices lie in zero CT (outside the volume's mask)",
                empty * 100));
    }
    value = json_object_object_get(mesh, full ? "all_vertices_in_air" : "air_share");
    double air = json_object_get_double(value);
    int measured = sampled || full != NULL;
    if (measured && value && air > 0.35) {
        qa_reason(exclude,
            fmt(a, "in air: %.0f%% of its vertices have no papyrus within 3 voxels", air * 100));
    } else if (measured && value && air > 0.2) {
        qa_reason(
            lower, fmt(a, "%.0f%% of its vertices have no papyrus within 3 voxels", air * 100));
    }
    json_object *apart = json_object_object_get(mesh, "numbered_apart_on");
    if (apart && jf64(apart, "share") >= 0.3) {
        qa_reason(lower,
            fmt(a, "%.0f%% of its vertices lie on the surface of %s, numbered another winding",
                100 * jf64(apart, "share"), jstr(apart, "segment")));
    }
    json_object *conflict = json_object_object_get(mesh, "conflict"),
                *order = json_object_object_get(mesh, "order");
    double bad = json_object_get_double(conflict) + json_object_get_double(order);
    if (bad > 0.25) {
        qa_reason(exclude,
            fmt(a, "contradicts other meshes: conflict %s, order %s", qa_spelling(conflict),
                qa_spelling(order)));
    } else if (bad > 0.05) {
        qa_reason(
            lower, fmt(a, "conflict %s, order %s", qa_spelling(conflict), qa_spelling(order)));
    }
    json_object *partner = json_object_object_get(mesh, "shares_surface_most_with");
    const char *segment = partner ? jstr(partner, "segment") : "a larger mesh";
    double duplicate = jf64(mesh, "duplicate_share");
    if (duplicate > 0.9) {
        qa_reason(
            exclude, fmt(a, "a copy: %.0f%% of its vertices repeat %s", duplicate * 100, segment));
    } else if (duplicate > 0.5) {
        qa_reason(lower, fmt(a, "%.0f%% of its vertices repeat %s", duplicate * 100, segment));
    }
    if (board_count(exclude)) {
        for (int i = 0; i < board_count(lower); i++) {
            qa_reason(exclude, board_item(lower, i));
        }
        json_object_put(lower);
        *reasons = exclude;
        return 2;
    }
    json_object_put(exclude);
    *reasons = lower;
    if (board_count(lower)) {
        return 1;
    }
    if (!measured) {
        qa_reason(lower, fmt(a, "CT unmeasured (%lld vertices sampled)", (long long)n));
    }
    return 0;
}

typedef struct {
    const char *segment;
    json_object *row;
    i64 wrap;
} QATableRow;

static int qa_table_order(const void *left, const void *right) {
    const QATableRow *a = left, *b = right;
    if (a->wrap != b->wrap) {
        return a->wrap < b->wrap ? -1 : 1;
    }
    return strcmp(a->segment, b->segment);
}

static const char *const qa_verdicts[] = {"include", "lower weight", "exclude", "left out"};

static void qa_python_repr(FILE *md, json_object *value) {
    if (!value) {
        fputs("None", md);
    } else if (json_object_is_type(value, json_type_string)) {
        const char *p = json_object_get_string(value);
        fputc('\'', md);
        for (; *p; p++) {
            if (*p == '\'' || *p == '\\') {
                fputc('\\', md);
            }
            fputc(*p, md);
        }
        fputc('\'', md);
    } else if (json_object_is_type(value, json_type_array)) {
        fputc('[', md);
        for (int i = 0; i < board_count(value); i++) {
            fputs(i ? ", " : "", md);
            qa_python_repr(md, json_object_array_get_idx(value, i));
        }
        fputc(']', md);
    } else {
        fputs(board_json(value), md);
    }
}

static void qa_table_markdown(FILE *md, const char *name, json_object *numbers,
    json_object *selection, int within, const i64 *count, const double *area, double contrast,
    double offset, QATableRow *rows, int n) {
    json_object *scan = jget(numbers, "scan"), *z = jget(numbers, "z");
    const char *lo = board_json(json_object_array_get_idx(z, 0));
    const char *hi = board_json(json_object_array_get_idx(z, 1));
    fprintf(md, "\n## %s\n\n%s %s, z %s-%s, %s cm2, %d meshes; package medians: contrast ", name,
        jstr(scan, "sample"), jstr(scan, "scan"), lo, hi, board_json(jget(numbers, "area_cm2")), n);
    json_object *ref_contrast = jnum(contrast), *ref_offset = jnum(offset);
    char a[4][32];
    for (int i = 0; i < 4; i++) {
        if (count[i]) {
            snprintf(a[i], sizeof(a[i]), "%.1f", area[i]);
        } else {
            snprintf(a[i], sizeof(a[i]), "0");
        }
    }
    fprintf(md,
        "%s, %s vox to the ridge. Include %lld (%s cm2), lower weight %lld (%s cm2), "
        "exclude %lld (%s cm2)",
        board_json(ref_contrast), board_json(ref_offset), (long long)count[0], a[0],
        (long long)count[1], a[1], (long long)count[2], a[2]);
    json_object_put(ref_contrast);
    json_object_put(ref_offset);
    if (selection) {
        fprintf(md, ", left out by the labels' selection %lld (%.1f cm2; only ",
            (long long)count[3], area[3]);
        qa_python_repr(md, json_object_object_get(selection, "only"));
        fputs(", exclude [", md);
        const char *keys[] = {"module_exclude", "exclude"};
        int n = 0;
        for (int k = 0; k < 2; k++) {
            json_object *part = json_object_object_get(selection, keys[k]);
            for (int i = 0; i < board_count(part); i++) {
                fputs(n++ ? ", " : "", md);
                qa_python_repr(md, json_object_array_get_idx(part, i));
            }
        }
        fputs("], z ", md);
        qa_python_repr(md, json_object_object_get(selection, "z"));
        if (within) {
            fputs("; for the selected meshes, area and the duplicate, conflict and order shares "
                  "are taken again among the selected meshes inside the window",
                md);
        }
        fputc(')', md);
    }
    fputs(".\n\n| Mesh | Wrap | cm2 | Recommendation | Why |\n|---|---|---|---|---|\n", md);
    qsort(rows, n, sizeof(*rows), qa_table_order);
    for (int i = 0; i < n; i++) {
        QATableRow row = rows[i];
        fprintf(md, "| %s | %lld | %s | %s | ", row.segment, (long long)row.wrap,
            board_json(jget(row.row, "area_cm2")), jstr(row.row, "recommendation"));
        json_object *reasons = jget(row.row, "reasons");
        if (!board_count(reasons)) {
            fputc('-', md);
        }
        for (int r = 0; r < board_count(reasons); r++) {
            fprintf(md, "%s%s", r ? "; " : "", board_item(reasons, r));
        }
        fputs(" |\n", md);
    }
}

static json_object *qa_table_package(Arena *a, const char *name, FILE *md, json_object *summary) {
    size_t saved = a->used;
    json_object *numbers = json_read(fmt(a, "%s/bench/qa/%s/numbers.json", OUTPUTS, name));
    need(numbers != NULL, "%s has no QA numbers", name);
    json_object *meshes = jget(numbers, "meshes_detail");
    int n = board_count(meshes), measured = 0;
    double *contrast = push(a, n * sizeof(*contrast)), *offsets = push(a, n * sizeof(*offsets));
    for (int i = 0; i < n; i++) {
        json_object *row = json_object_array_get_idx(meshes, i);
        json_object *value = json_object_object_get(row, "profile_contrast");
        if (board_integer(row, "sampled") >= 200 && value) {
            contrast[measured] = json_object_get_double(value);
            offsets[measured++] = json_object_get_double(
                json_object_array_get_idx(jget(row, "ridge_abs_p50_p90_vox"), 0));
        }
    }
    double ref_contrast = package_round(bench_median(contrast, measured, 0), 3);
    double ref_offset = package_round(bench_median(offsets, measured, 0), 2);
    json_object *selection = qa_selection(a, name);
    int within = !strcmp(board_text(selection, "duplicates"), "judged within the mesh selection");
    json_object *again = within ? qa_selection_shares(a, name, selection) : NULL;
    QATableRow *ordered = push(a, n * sizeof(*ordered));
    json_object *out = json_object_new_object(), *reference = json_object_new_object(),
                *rows = json_object_new_array_ext(n), *counts = json_object_new_object(),
                *areas = json_object_new_object();
    i64 count[4] = {0};
    double area[4] = {0};
    const char *metric_names[] = {"duplicate_share", "conflict", "order", "touching", "pressed",
        "sampled", "ridge_abs_p50_p90_vox", "ridge_signed_p50_vox", "null_abs_p50_vox",
        "profile_contrast", "ct_empty_share", "air_share", "all_vertices_zero_ct",
        "all_vertices_in_air", "dropped_by_other_meshes"};
    for (int i = 0; i < n; i++) {
        json_object *mesh = json_object_array_get_idx(meshes, i);
        const char *segment = jstr(mesh, "segment");
        int verdict;
        json_object *reasons;
        if (selection && !qa_selected(segment, selection)) {
            verdict = 3;
            reasons = json_object_new_array();
            qa_reason(reasons, "outside the labels' mesh selection");
        } else {
            if (within) {
                json_object *subset = json_object_object_get(again, segment);
                if (subset) {
                    json_object_object_foreach(subset, key, value) {
                        json_object_object_add(mesh, key, json_object_get(value));
                    }
                }
                json_object *apart = json_object_object_get(mesh, "numbered_apart_on");
                if (apart && !qa_selected(jstr(apart, "segment"), selection)) {
                    json_object_object_del(mesh, "numbered_apart_on");
                }
            }
            verdict = qa_judge(a, mesh, ref_contrast, ref_offset, &reasons);
        }
        const double weight[] = {1, 0.5, 0, 0};
        count[verdict]++;
        area[verdict] += jf64(mesh, "area_cm2");
        json_object *row = json_object_new_object(), *metrics = json_object_new_object();
        json_object_array_add(rows, row);
        const char *keys[] = {"segment", "wrap", "frame", "area_cm2"};
        for (int j = 0; j < 4; j++) {
            json_object_object_add(row, keys[j], json_object_get(jget(mesh, keys[j])));
        }
        json_object_object_add(row, "recommendation", json_object_new_string(qa_verdicts[verdict]));
        json_object_object_add(row, "weight", jnum(weight[verdict]));
        json_object_object_add(row, "reasons", reasons);
        for (size_t j = 0; j < sizeof(metric_names) / sizeof(*metric_names); j++) {
            const char *key = metric_names[j];
            json_object_object_add(
                metrics, key, json_object_get(json_object_object_get(mesh, key)));
        }
        json_object_object_add(row, "metrics", metrics);
        ordered[i] = (QATableRow){segment, row, ji64(mesh, "wrap")};
    }
    for (int i = 0; i < 4; i++) {
        json_object_object_add(counts, qa_verdicts[i], json_object_new_int64(count[i]));
        area[i] = package_round(area[i], 1);
        json_object_object_add(
            areas, qa_verdicts[i], count[i] ? jnum(area[i]) : json_object_new_int(0));
    }
    json_object_object_add(out, "scan", json_object_get(jget(numbers, "scan")));
    json_object_object_add(out, "z", json_object_get(jget(numbers, "z")));
    json_object_object_add(out, "area_cm2", json_object_get(jget(numbers, "area_cm2")));
    bench_decimal(reference, "contrast", ref_contrast, 3);
    bench_decimal(reference, "offset", ref_offset, 2);
    json_object_object_add(out, "reference", reference);
    json_object_object_add(out, "label_selection", json_object_get(selection));
    json_object_object_add(out, "counts", counts);
    json_object_object_add(summary, name, json_object_get(counts));
    json_object_object_add(out, "area_by_recommendation_cm2", areas);
    json_object_object_add(out, "meshes", rows);
    qa_table_markdown(
        md, name, numbers, selection, within, count, area, ref_contrast, ref_offset, ordered, n);
    json_object_put(again);
    json_object_put(selection);
    json_object_put(numbers);
    a->used = saved;
    return out;
}

int bench_qa_table_command(int argc, char **argv) {
    Arena a = arena(16ull << 30);
    int count = 0;
    const char **names = push(&a, argc * sizeof(*names)), *notes = NULL;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--notes")) {
            need(++i < argc, "--notes requires a markdown file");
            notes = argv[i];
        } else {
            need(argv[i][0] != '-', "unknown QA table option %s", argv[i]);
            names[count++] = argv[i];
        }
    }
    need(count > 0, "calibrate.bench.qa table CONTROL ... [--notes FILE]");
    char *text = NULL;
    size_t length = 0;
    FILE *md = open_memstream(&text, &length);
    need(md != NULL, "cannot create training truth Markdown");
    fputs("# Training truth\n\n", md);
    if (notes) {
        Bytes b = read_file(&a, notes);
        size_t start = 0, end = b.size;
        const u8 *values = b.data;
        while (start < end && isspace((unsigned char)values[start])) {
            start++;
        }
        while (end > start && isspace((unsigned char)values[end - 1])) {
            end--;
        }
        fwrite(values + start, 1, end - start, md);
        fputs("\n\n", md);
    }
    fputs("## Rules\n\nPer-mesh recommendation for training labels (bench.qa table).\n\n", md);
    json_object *record = json_object_new_object(), *rules = json_object_new_object(),
                *packages = json_object_new_object(), *summary = json_object_new_object();
    char day[32], built[32];
    bench_time(day, built);
    json_object_object_add(record, "built", json_object_new_string(built));
    json_object_object_add(record, "rules", rules);
    json_object_object_add(record, "packages", packages);
    for (int k = 0; k < 3; k++) {
        json_object *list = json_object_new_array();
        json_object_object_add(rules, qa_rule_names[k], list);
        fprintf(md, "- **%s**: ", qa_rule_names[k]);
        for (int j = 0; qa_rules[k][j]; j++) {
            fprintf(md, "%s%s", j ? "; " : "", qa_rules[k][j]);
            json_object_array_add(list, json_object_new_string(qa_rules[k][j]));
        }
        fputs(".\n", md);
    }
    for (int i = 0; i < count; i++) {
        json_object_object_add(packages, names[i], qa_table_package(&a, names[i], md, summary));
    }
    need(fclose(md) == 0, "cannot finish training truth Markdown");
    const char *root = fmt(&a, "%s/bench/qa", OUTPUTS);
    json_write(fmt(&a, "%s/training-truth.json", root), record);
    write_file(fmt(&a, "%s/training-truth.md", root), text, length);
    free(text);
    puts(board_json(summary));
    json_object_put(summary);
    json_object_put(record);
    munmap(a.base, a.size);
    return 0;
}

int bench_qa_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "-h") || !strcmp(argv[0], "--help")) {
        puts("calibrate.bench.qa vertices|numbers|planes|table ...");
        return !argc;
    }
    if (!strcmp(argv[0], "vertices")) {
        return bench_qa_vertices_command(argc - 1, argv + 1);
    }
    if (!strcmp(argv[0], "numbers")) {
        return bench_qa_numbers_command(argc - 1, argv + 1);
    }
    if (!strcmp(argv[0], "planes")) {
        return bench_qa_planes_command(argc - 1, argv + 1);
    }
    if (!strcmp(argv[0], "table")) {
        return bench_qa_table_command(argc - 1, argv + 1);
    }
    die("unknown bench QA command %s", argv[0]);
    return 1;
}
