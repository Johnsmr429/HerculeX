#ifndef HX_UNITY
const char hx_usage[] = "Score extracted surfaces against v1 mesh truth.";
#endif

typedef struct {
    i64 vertex, coarse;
    int mesh;
    double delta, phase;
} LegacyPhase;

static int legacy_phase_order(const void *left, const void *right) {
    const LegacyPhase *a = left, *b = right;
    if (a->coarse != b->coarse) {
        return a->coarse < b->coarse ? -1 : 1;
    }
    if (a->mesh != b->mesh) {
        return a->mesh - b->mesh;
    }
    return a->vertex < b->vertex ? -1 : a->vertex != b->vertex;
}

static double legacy_phase(double re, double im) { return atan2(im, re) / (2 * M_PI); }

static double legacy_phases(LegacyPhase *row, i64 n) {
    double strength = 0;
    qsort(row, n, sizeof(*row), legacy_phase_order);
    for (i64 at = 0, end; at < n; at = end) {
        double re = 0, im = 0;
        for (end = at; end < n && row[end].coarse == row[at].coarse; end++) {
            re += cos(2 * M_PI * row[end].delta);
            im += sin(2 * M_PI * row[end].delta);
        }
        double reference = legacy_phase(re, im);
        for (i64 i = at, j; i < end; i = j) {
            re = 0;
            im = 0;
            for (j = i; j < end && row[i].mesh == row[j].mesh; j++) {
                re += cos(2 * M_PI * row[j].delta);
                im += sin(2 * M_PI * row[j].delta);
            }
            strength += hypot(re, im);
            double phase = legacy_phase(re, im);
            phase -= nearbyint(phase - reference);
            for (i64 k = i; k < j; k++) {
                row[k].phase = phase;
            }
        }
    }
    return n ? strength / n : 0;
}

typedef struct {
    i32 *mesh;
    u8 *status;
    f32 *depth, *winding;
} LegacySurfaceLabels;

enum { LEGACY_NEAR = 1, LEGACY_ON = 2, LEGACY_PRESSED = 4, LEGACY_RIGHT = 8, LEGACY_WRONG = 16 };

static BenchModes legacy_modes(BenchVote *vote, i64 n) {
    qsort(vote, n, sizeof(*vote), bench_vote_order);
    i64 used = 0;
    for (i64 i = 0, j; i < n; i = j) {
        i64 peak = 0, mode = vote[i].turn;
        for (j = i; j < n && vote[j].key == vote[i].key;) {
            i64 end = j + 1;
            while (end < n && vote[j].key == vote[end].key && vote[j].turn == vote[end].turn) {
                end++;
            }
            if (end - j > peak) {
                peak = end - j;
                mode = vote[j].turn;
            }
            j = end;
        }
        vote[used++] = (BenchVote){vote[i].key, mode};
    }
    return (BenchModes){.vote = vote, .n = used};
}

static void legacy_surface_classify(Arena *a, const BenchSurface *s, const BenchPieces *p,
    const BenchLegacyTruth *truth, const BenchMatches *matches, LegacySurfaceLabels *labels) {
    size_t saved = a->used;
    LegacyPhase *phase = push(a, s->n * sizeof(*phase));
    BenchVote *vote = push(a, s->n * sizeof(*vote));
    i64 n = 0;
    for (i64 i = s->first; i < s->first + s->n; i++) {
        const BenchMatches *m = matches + i;
        i32 first = m->index[0], second = m->index[1];
        unsigned status = first >= 0 ? LEGACY_NEAR : 0;
        int on = m->depth[0] <= 40 / truth->voxel;
        labels->depth[i] = m->depth[0] * (f32)truth->voxel;
        labels->mesh[i] = on ? truth->mesh[first] : -1;
        labels->winding[i] = on ? truth->winding[first] : NAN;
        if (on) {
            status |= LEGACY_ON;
            if (second >= 0 && m->depth[1] <= 80 / truth->voxel &&
                fabs(truth->winding[first] - truth->winding[second]) >= 0.5) {
                status |= LEGACY_PRESSED;
            }
            if (isfinite(truth->winding[first])) {
                double delta = truth->winding[first] - truth->sense * p->turn[i] / (2 * M_PI);
                i64 key = ((i64)p->part[i] << 32) | (u32)truth->frame[first];
                phase[n++] = (LegacyPhase){
                    .vertex = i, .coarse = key, .mesh = truth->mesh[first], .delta = delta};
            }
        }
        labels->status[i] = status;
    }
    legacy_phases(phase, n);
    for (i64 i = 0; i < n; i++) {
        vote[i] = (BenchVote){phase[i].coarse, nearbyint(phase[i].delta - phase[i].phase)};
    }
    BenchModes modes = legacy_modes(vote, n);
    for (i64 i = 0; i < n; i++) {
        i64 now = nearbyint(phase[i].delta - phase[i].phase), mode = 0;
        bench_mode(&modes, phase[i].coarse, &mode);
        labels->status[phase[i].vertex] |= now == mode ? LEGACY_RIGHT : LEGACY_WRONG;
    }
    a->used = saved;
}

typedef struct {
    double best, over_one, over_four, wrong_area;
    i64 wrong_count;
} LegacyBlobs;

static LegacyBlobs legacy_surface_blobs(
    Arena *a, const BenchSurface *s, const BenchPieces *p, const LegacySurfaceLabels *labels) {
    size_t saved = a->used;
    i64 size = (i64)s->h * s->w;
    u8 *seen = pushz(a, size);
    i32 *queue = push(a, s->n * sizeof(*queue));
    LegacyBlobs out = {0};
    for (i64 i = 0; i < size; i++) {
        i32 first = s->index[i];
        if (first < 0 || seen[i]) {
            continue;
        }
        int group = labels->status[s->first + first] & (LEGACY_RIGHT | LEGACY_WRONG);
        if (!group) {
            continue;
        }
        i64 count = 1;
        double area = 0;
        queue[0] = i;
        seen[i] = 1;
        for (i64 q = 0; q < count; q++) {
            i32 at = queue[q], r = at / s->w, c = at % s->w;
            area += p->area[s->first + s->index[at]];
            for (int y = r - 1; y <= r + 1; y++) {
                for (int x = c - 1; x <= c + 1; x++) {
                    if (y < 0 || y >= s->h || x < 0 || x >= s->w) {
                        continue;
                    }
                    i64 k = (i64)y * s->w + x;
                    i32 next = s->index[k];
                    if (next >= 0 && !seen[k] && (labels->status[s->first + next] & group)) {
                        queue[count++] = k;
                        seen[k] = 1;
                    }
                }
            }
        }
        if (group == LEGACY_RIGHT) {
            out.best = fmax(out.best, area);
            out.over_one += area >= 1 ? area : 0;
            out.over_four += area >= 4 ? area : 0;
        } else if (area >= 0.16) {
            out.wrong_area += area;
            out.wrong_count++;
        }
    }
    a->used = saved;
    return out;
}

typedef struct {
    double area, on, off, no_truth, pressed, right, wrong, right_resolved, wrong_resolved;
    double right_big, wrong_big;
    i64 big;
    LegacyBlobs blobs;
} LegacySurfaceTotals;

typedef struct {
    int mesh;
    double area, correct;
} LegacyFollow;

static int legacy_follow_order(const void *left, const void *right) {
    const LegacyFollow *a = left, *b = right;
    if (a->area != b->area) {
        return a->area > b->area ? -1 : 1;
    }
    return a->mesh - b->mesh;
}

static json_object *legacy_surface_row(Arena *a, const BenchPieces *p, int mesh,
    const BenchLegacyTruth *truth, const LegacySurfaceLabels *labels, LegacySurfaceTotals *total) {
    const BenchSurface *s = p->surfaces + mesh;
    size_t saved = a->used;
    LegacySurfaceTotals acc = {0};
    LegacyFollow *follow = pushz(a, truth->meshes * sizeof(*follow));
    u8 *followed = pushz(a, truth->meshes);
    for (int m = 0; m < truth->meshes; m++) {
        follow[m].mesh = m;
    }
    int parts = 0;
    for (i64 i = s->first; i < s->first + s->n; i++) {
        double area = p->area[i];
        unsigned mark = labels->status[i];
        acc.area += area;
        if (p->part[i] + 1 > parts) {
            parts = p->part[i] + 1;
        }
        if (mark & LEGACY_ON) {
            acc.on += area;
            int m = labels->mesh[i];
            followed[m] = 1;
            follow[m].area += area;
            if (mark & LEGACY_RIGHT) {
                follow[m].correct += area;
            }
        } else if (mark & LEGACY_NEAR) {
            acc.off += area;
        } else {
            acc.no_truth += area;
        }
        if (mark & LEGACY_PRESSED) {
            acc.pressed += area;
        }
        if (mark & LEGACY_RIGHT) {
            acc.right += area;
            acc.right_resolved += mark & LEGACY_PRESSED ? 0 : area;
        }
        if (mark & LEGACY_WRONG) {
            acc.wrong += area;
            acc.wrong_resolved += mark & LEGACY_PRESSED ? 0 : area;
        }
    }
    acc.blobs = legacy_surface_blobs(a, s, p, labels);
    json_object *row = json_object_new_object();
    json_object_object_add(row, "surface", json_object_new_string(s->name));
    json_object_object_add(row, "vertices", json_object_new_int64(s->n));
    json_object_object_add(row, "parts", json_object_new_int(parts));
    bench_decimal(row, "area_cm2", acc.area, 4);
    bench_decimal(row, "on_truth_cm2", acc.on, 4);
    bench_decimal(row, "off_truth_cm2", acc.off, 4);
    bench_decimal(row, "no_truth_cm2", acc.no_truth, 4);
    bench_decimal(row, "pressed_cm2", acc.pressed, 4);
    bench_decimal(row, "correct_cm2", acc.right, 4);
    bench_decimal(row, "wrong_cm2", acc.wrong, 4);
    bench_decimal(row, "correct_resolved_cm2", acc.right_resolved, 4);
    bench_decimal(row, "wrong_resolved_cm2", acc.wrong_resolved, 4);
    bench_ratio(row, "wrong_rate", acc.wrong, acc.right + acc.wrong, 4);
    bench_decimal(row, "largest_correct_cm2", acc.blobs.best, 4);
    bench_decimal(row, "correct_in_1cm2_parts", acc.blobs.over_one, 4);
    bench_decimal(row, "correct_in_4cm2_parts", acc.blobs.over_four, 4);
    json_object_object_add(row, "wrong_blobs", json_object_new_int64(acc.blobs.wrong_count));
    bench_decimal(row, "wrong_in_blobs_cm2", acc.blobs.wrong_area, 4);
    int follow_n = 0;
    for (int m = 0; m < truth->meshes; m++) {
        if (followed[m]) {
            follow[follow_n++] = follow[m];
        }
    }
    for (int i = 0; i < follow_n; i++) {
        follow[i].area = package_round(follow[i].area, 4);
    }
    qsort(follow, follow_n, sizeof(*follow), legacy_follow_order);
    json_object *follows = json_object_new_array();
    for (int i = 0; i < follow_n && i < 4; i++) {
        LegacyFollow f = follow[i];
        json_object *item = json_object_new_object();
        json_object_array_add(follows, item);
        json_object_object_add(
            item, "mesh", json_object_new_string(board_item(truth->names, f.mesh)));
        json_object_object_add(item, "wrap", json_object_new_int(truth->wrap[f.mesh]));
        bench_decimal(item, "cm2", f.area, 4);
        double exact_area = 0;
        for (i64 j = s->first; j < s->first + s->n; j++) {
            if (labels->mesh[j] == f.mesh) {
                exact_area += p->area[j];
            }
        }
        bench_ratio(item, "correct_share", f.correct, fmax(exact_area, 1e-12), 4);
    }
    json_object_object_add(row, "follows", follows);
    double right = package_round(acc.right, 4), wrong = package_round(acc.wrong, 4);
    total->area += acc.area;
    total->on += acc.on;
    total->off += acc.off;
    total->no_truth += acc.no_truth;
    total->pressed += acc.pressed;
    total->right += right;
    total->wrong += wrong;
    total->right_resolved += package_round(acc.right_resolved, 4);
    total->wrong_resolved += package_round(acc.wrong_resolved, 4);
    if (package_round(acc.area, 4) >= 1) {
        total->big++;
        total->right_big += right;
        total->wrong_big += wrong;
    }
    total->blobs.best = fmax(total->blobs.best, package_round(acc.blobs.best, 4));
    total->blobs.over_one += package_round(acc.blobs.over_one, 4);
    total->blobs.over_four += package_round(acc.blobs.over_four, 4);
    total->blobs.wrong_count += acc.blobs.wrong_count;
    a->used = saved;
    return row;
}

static json_object *legacy_surface_depth(
    Arena *a, const BenchPieces *p, const LegacySurfaceLabels *labels, int bit) {
    size_t saved = a->used;
    double *sample = push(a, p->count * sizeof(*sample));
    i64 n = 0;
    for (i64 i = 0; i < p->count; i++) {
        if (labels->status[i] & bit) {
            sample[n++] = labels->depth[i];
        }
    }
    json_object *out = NULL;
    if (n) {
        qsort(sample, n, sizeof(*sample), bench_double_order);
        const double quantiles[] = {0.5, 0.9, 0.99};
        out = json_object_new_array();
        for (int i = 0; i < 3; i++) {
            double at = (n - 1) * quantiles[i];
            i64 j = floor(at), k = ceil(at);
            double value = sample[j] + (sample[k] - sample[j]) * (at - j);
            json_object_array_add(out, jnum(package_round(value, 2)));
        }
    }
    a->used = saved;
    return out;
}

static void legacy_surface_box(const BenchPieces *p, double box[2][3], double margin) {
    for (int d = 0; d < 3; d++) {
        box[0][d] = INFINITY;
        box[1][d] = -INFINITY;
    }
    for (i64 i = 0; i < p->count; i++) {
        for (int d = 0; d < 3; d++) {
            double v = p->xyz[3 * i + d];
            box[0][d] = fmin(box[0][d], v);
            box[1][d] = fmax(box[1][d], v);
        }
    }
    for (int d = 0; d < 3; d++) {
        box[0][d] = (f32)box[0][d] - (f32)margin;
        box[1][d] = (f32)box[1][d] + (f32)margin;
    }
}

static int legacy_in_box(const f32 *p, const double box[2][3]) {
    return p[0] >= box[0][0] && p[0] < box[1][0] && p[1] >= box[0][1] && p[1] < box[1][1] &&
        p[2] >= box[0][2] && p[2] < box[1][2];
}

static void legacy_surface_recall(Arena *a, const BenchLegacyTruth *t, const BenchPieces *p,
    const LegacySurfaceLabels *labels, json_object *out) {
    size_t saved = a->used;
    double box[2][3], depth = 40 / t->voxel;
    legacy_surface_box(p, box, depth);
    i32 *inside = push(a, t->n * sizeof(*inside));
    i64 count = 0;
    for (i64 i = 0; i < t->n; i++) {
        if (legacy_in_box(t->points + 3 * i, box)) {
            inside[count++] = i;
        }
    }
    i64 step = count >= 1000000 ? count / 500000 : 1;
    i64 size = (count + step - 1) / step;
    double *edges = push(a, p->count * sizeof(*edges));
    i64 n = 0;
    for (int m = 0; m < p->n; m++) {
        const BenchSurface *s = p->surfaces + m;
        for (int r = 0; r < s->h; r++) {
            for (int c = 1; c < s->w; c++) {
                i64 i = (i64)r * s->w + c;
                if (s->index[i] >= 0 && s->index[i - 1] >= 0) {
                    edges[n++] = bench_step(s->grid + 3 * i, s->grid + 3 * (i - 1));
                }
            }
        }
    }
    double lateral = 0.75 * bench_median(edges, n, 1);
    a->used = (char *)edges - (char *)a->base;
    f32 *query = push(a, 3 * size * sizeof(*query)), *normal = push(a, 3 * size * sizeof(*normal));
    json_object_object_add(out, "truth_recall_sample", json_object_new_int64(size));
    const char *keys[] = {"recall", "recall_null"};
    for (int pass = 0; pass < 2; pass++) {
        size_t scratch = a->used;
        f32 shift = pass * 80 / t->voxel;
        for (i64 i = 0; i < size; i++) {
            i32 source = inside[i * step];
            for (int d = 0; d < 3; d++) {
                normal[3 * i + d] = t->normals[3LL * source + d];
                query[3 * i + d] = t->points[3LL * source + d] + shift * normal[3 * i + d];
            }
        }
        BenchMatches *hit =
            bench_on_sheet_limit(a, p->xyz, p->count, query, size, normal, depth, lateral, 1, 8);
        i64 recall = 0, correct = 0;
        for (i64 i = 0; i < size; i++) {
            int j = hit[i].index[0];
            if (j >= 0) {
                recall++;
                correct += (labels->status[j] & LEGACY_RIGHT) != 0;
            }
        }
        bench_ratio(out, keys[pass], recall, size, 4);
        if (!pass) {
            bench_ratio(out, "recall_correct", correct, size ? size : 1, 4);
        }
        a->used = scratch;
    }
    a->used = saved;
}

static const char *legacy_surface_params(
    Arena *a, json_object *given, int locked, const char *source, int every) {
    return fmt(a, "{\"every\": %d, \"mask\": \"%s\", \"surfaces\": %s, \"truth\": \"%s\"}", every,
        locked ? "locked" : "known", bench_json_array(a, given), source);
}

static void legacy_record(Arena *a, const char *control, const char *system, const char *method,
    const char *kind, const char *split, const char *parameters, json_object *metrics,
    json_object *detail) {
    Board db = bench_database(a);
    char day[32], when[32], head[128], dirty[128], hex[65];
    bench_git(head, 0);
    bench_git(dirty, 1);
    bench_time(day, when);
    const char *commit = fmt(a, "%s%s", *head ? head : "unknown", *dirty ? "+dirty" : "");
    sha256_hex(parameters, strlen(parameters), hex);
    const char *unit = fmt(a, "%s|%s|%s|%s|%.12s", system, method, kind, control, hex);
    board_exec(&db, "BEGIN IMMEDIATE; DELETE FROM runs WHERE unit=%Q", unit);
    board_exec(&db,
        "INSERT INTO "
        "runs(unit,created,control,split,system,method,kind,commit_id,parameters,detail)"
        " VALUES (%Q,%Q,%Q,%Q,%Q,%Q,%Q,%Q,%Q,%Q)",
        unit, when, control, split, system, method, kind, commit, parameters, board_json(detail));
    sqlite3_int64 run = sqlite3_last_insert_rowid(db.db);
    json_object_object_foreach(metrics, key, value) {
        if (value &&
            (json_object_is_type(value, json_type_double) ||
                json_object_is_type(value, json_type_int))) {
            double number = json_object_get_double(value);
            if (isfinite(number)) {
                board_exec(&db, "INSERT INTO metrics(run,key,value) VALUES (%lld,%Q,%!.17g)",
                    (long long)run, key, number);
            }
        }
    }
    board_exec(&db, "COMMIT");
    sqlite3_close(db.db);
    printf("recorded run %lld\n", (long long)run);
}

static json_object *legacy_surface_headline(json_object *out) {
    const char *keys[] = {"area_cm2", "on_truth_share", "correct_cm2", "wrong_cm2", "wrong_rate",
        "wrong_rate_resolved", "largest_correct_cm2", "correct_in_1cm2_parts",
        "correct_in_4cm2_parts", "correct_cm2_in_1cm2_surfaces", "wrong_rate_in_1cm2_surfaces",
        "recall", "recall_correct", "recall_null", "surfaces"};
    json_object *metrics = json_object_new_object();
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) {
        json_object *value = json_object_object_get(out, keys[i]);
        if (value) {
            json_object_object_add(metrics, keys[i], json_object_get(value));
        }
    }
    json_object *depth = json_object_object_get(out, "depth_um_on_truth_p50_p90_p99");
    if (depth) {
        json_object_object_add(
            metrics, "depth_um_p50", json_object_get(json_object_array_get_idx(depth, 0)));
        json_object_object_add(
            metrics, "depth_um_p90", json_object_get(json_object_array_get_idx(depth, 1)));
    }
    return metrics;
}

static void legacy_surface_labels(
    Arena *a, const char *path, const BenchPieces *p, const LegacySurfaceLabels *labels) {
    size_t saved = a->used;
    i64 n = p->count;
    i64 *ids = push(a, n * sizeof(*ids)), *mesh = push(a, n * sizeof(*mesh));
    u8 *matched = push(a, n), *correct = push(a, n);
    for (i64 i = 0; i < n; i++) {
        ids[i] = p->surface[i];
        mesh[i] = labels->mesh[i];
        matched[i] = (labels->status[i] & LEGACY_ON) != 0;
        correct[i] = (labels->status[i] & LEGACY_RIGHT) != 0;
    }
    PackageGrid *grids = pushz(a, p->n * sizeof(*grids));
    for (int i = 0; i < p->n; i++) {
        grids[i].segment = p->surfaces[i].name;
    }
    Array names = package_names(a, &(Package){.grids_n = p->n, .grids = grids});
    const char *keys[] = {
        "xyz", "surface", "names", "depth_um", "matched", "mesh", "winding", "correct"};
    Array arrays[] = {
        {.descr = "<f4", .ndim = 2, .shape = {n, 3}, .data = p->xyz, .size = n * 12},
        {.descr = "<i8", .ndim = 1, .shape = {n}, .data = ids, .size = n * 8},
        names,
        {.descr = "<f4", .ndim = 1, .shape = {n}, .data = labels->depth, .size = n * 4},
        {.descr = "|b1", .ndim = 1, .shape = {n}, .data = matched, .size = n},
        {.descr = "<i8", .ndim = 1, .shape = {n}, .data = mesh, .size = n * 8},
        {.descr = "<f4", .ndim = 1, .shape = {n}, .data = labels->winding, .size = n * 4},
        {.descr = "|b1", .ndim = 1, .shape = {n}, .data = correct, .size = n},
    };
    npz_write_codec(path, sizeof(keys) / sizeof(*keys), keys, arrays, ZIP_CM_DEFLATE);
    a->used = saved;
}

static json_object *legacy_surface_score(Arena *a, BenchLegacyTruth *t, const BenchPieces *p,
    double lateral_um, int recall, int keep_rows, const char *labels_file) {
    i64 n = p->count;
    LegacySurfaceLabels labels = {.mesh = push(a, n * sizeof(i32)),
        .status = push(a, n),
        .depth = push(a, n * sizeof(f32)),
        .winding = push(a, n * sizeof(f32))};
    double lateral = (lateral_um > 0 ? lateral_um : 0.75 * t->spacing) / t->voxel;
    double reach = hypot(300 / t->voxel, lateral);
    double box[2][3];
    legacy_surface_box(p, box, reach);
    i64 in_box = 0;
    for (i64 i = 0; i < t->n; i++) {
        in_box += legacy_in_box(t->points + 3 * i, box);
    }
    if (in_box < 0.8 * t->n) {
        legacy_subset(t, 1, box);
    }
    size_t saved = a->used;
    BenchMatches *matches =
        bench_on_sheet(a, t->points, t->n, p->xyz, n, t->normals, 300 / t->voxel, lateral, 0);
    json_object *rows = json_object_new_array(), *out = json_object_new_object();
    LegacySurfaceTotals sum = {0};
    for (int m = 0; m < p->n; m++) {
        legacy_surface_classify(a, p->surfaces + m, p, t, matches, &labels);
        json_object *row = legacy_surface_row(a, p, m, t, &labels, &sum);
        if (keep_rows) {
            json_object_array_add(rows, row);
        } else {
            json_object_put(row);
        }
    }
    a->used = saved;
    double judged = sum.right + sum.wrong;
    json_object_object_add(out, "surfaces", json_object_new_int(p->n));
    json_object_object_add(out, "vertices", json_object_new_int64(n));
    bench_decimal(out, "area_cm2", sum.area, 3);
    bench_ratio(out, "on_truth_share", sum.on, sum.area, 4);
    bench_ratio(out, "off_truth_share", sum.off, sum.area, 4);
    bench_ratio(out, "no_truth_share", sum.no_truth, sum.area, 4);
    bench_ratio(out, "pressed_share_of_on_truth", sum.pressed, fmax(sum.on, 1e-12), 4);
    bench_decimal(out, "correct_cm2", sum.right, 3);
    bench_decimal(out, "wrong_cm2", sum.wrong, 3);
    bench_ratio(out, "wrong_rate", sum.wrong, judged, 4);
    bench_ratio(
        out, "wrong_rate_resolved", sum.wrong_resolved, sum.right_resolved + sum.wrong_resolved, 4);
    bench_ratio(out, "dominant_share", sum.right, judged, 4);
    bench_ratio(out, "largest_correct_cm2", sum.blobs.best, judged > 0, 3);
    bench_ratio(out, "correct_in_1cm2_parts", sum.blobs.over_one, judged > 0, 3);
    bench_ratio(out, "correct_in_4cm2_parts", sum.blobs.over_four, judged > 0, 3);
    json_object_object_add(out, "surfaces_1cm2", json_object_new_int64(sum.big));
    bench_decimal(out, "correct_cm2_in_1cm2_surfaces", sum.right_big, 3);
    bench_ratio(
        out, "wrong_rate_in_1cm2_surfaces", sum.wrong_big, sum.right_big + sum.wrong_big, 4);
    json_object_object_add(out, "wrong_blobs", json_object_new_int64(sum.blobs.wrong_count));
    json_object_object_add(
        out, "depth_um_on_truth_p50_p90_p99", legacy_surface_depth(a, p, &labels, LEGACY_ON));
    json_object_object_add(
        out, "depth_um_near_p50_p90_p99", legacy_surface_depth(a, p, &labels, LEGACY_NEAR));
    bench_decimal(out, "lateral_reach_um", lateral * t->voxel, 1);
    if (recall) {
        legacy_surface_recall(a, t, p, &labels, out);
    }
    if (labels_file) {
        legacy_surface_labels(a, labels_file, p, &labels);
    }
    if (keep_rows) {
        json_object_object_add(out, "per_surface", rows);
    } else {
        json_object_put(rows);
    }
    return out;
}

int bench_surface_score_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.surface_score CONTROL SURFACES... [--truth auto|names|groups]"
             " [--mask known|locked] [--every N] [--no-recall] [--lateral-um UM]"
             " [--labels FILE] [--rows] [--method NAME] [--system NAME] [--no-record]");
        return !argc;
    }
    Arena a = arena(16ull << 30);
    BenchOptions o = {
        .control = argv[0], .label = "surfaces", .system = "network", .ignore_winding = 1};
    const char *source = "auto", *labels = NULL;
    int every = 1, recall = 1, rows = 0, record = 1;
    double lateral = 0;
    json_object *given = json_object_new_array();
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--no-recall")) {
            recall = 0;
        } else if (!strcmp(arg, "--no-record")) {
            record = 0;
        } else if (!strcmp(arg, "--rows")) {
            rows = 1;
        } else if (arg[0] == '-') {
            need(i + 1 < argc, "missing value for %s", arg);
            const char *value = argv[++i];
            if (!strcmp(arg, "--truth")) {
                source = value;
            } else if (!strcmp(arg, "--mask")) {
                need(!strcmp(value, "known") || !strcmp(value, "locked"), "invalid mask %s", value);
                o.locked = !strcmp(value, "locked");
            } else if (!strcmp(arg, "--every")) {
                every = board_value(value);
                need(every > 0, "--every must be positive");
            } else if (!strcmp(arg, "--lateral-um")) {
                lateral = board_value(value);
                need(lateral >= 0, "--lateral-um must be nonnegative");
            } else if (!strcmp(arg, "--labels")) {
                labels = value;
            } else if (!strcmp(arg, "--method")) {
                o.label = value;
            } else if (!strcmp(arg, "--system")) {
                o.system = value;
            } else {
                die("unknown surface scorer option %s", arg);
            }
        } else {
            json_object_array_add(given, json_object_new_string(arg));
        }
    }
    need(board_count(given) > 0, "surface scorer needs at least one surface");
    if (!strcmp(source, "auto")) {
        source = legacy_default(&a, o.control);
    }
    need(source && (!strcmp(source, "names") || !strcmp(source, "groups")), "invalid truth source");
    BenchLegacyTruth t = legacy_truth(&a, o.control, source);
    if (!t.normals) {
        t.normals = legacy_group_normals(&a, &t);
    }
    legacy_subset(&t, every, NULL);
    json_object *paths = bench_paths(&a, given);
    BenchTruth geometry = {.knots = t.knots,
        .nknots = t.nknots,
        .voxel = t.voxel,
        .record = json_tokener_parse("{\"scan\":{\"zones\":[]}}")};
    BenchPieces pieces = bench_pieces(&a, paths, &geometry, &o);
    json_object *out = legacy_surface_score(&a, &t, &pieces, lateral, recall, rows, labels);
    json_object *manifest = json_read(fmt(&a, "%s/calibrate/bench/controls.json", HOME));
    need(manifest != NULL, "cannot read bench controls");
    const char *split = bench_split(&a, manifest, o.control);
    json_object_object_add(out, "control", json_object_new_string(o.control));
    json_object_object_add(out, "truth", json_object_new_string(source));
    json_object_object_add(out, "split", json_object_new_string(split));
    puts(board_json(out));
    if (record) {
        json_object *metrics = legacy_surface_headline(out);
        json_object_object_del(out, "per_surface");
        legacy_record(&a, o.control, o.system, o.label, "surface", split,
            legacy_surface_params(&a, given, o.locked, source, every), metrics, out);
        json_object_put(metrics);
    }
    for (int i = 0; i < pieces.n; i++) {
        json_object_put(pieces.surfaces[i].meta);
    }
    json_object_put(geometry.record);
    json_object_put(manifest);
    json_object_put(out);
    json_object_put(paths);
    json_object_put(given);
    legacy_truth_close(&t);
    munmap(a.base, a.size);
    return 0;
}
