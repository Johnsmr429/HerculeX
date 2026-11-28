#ifndef HX_UNITY
const char hx_usage[] = "Score winding fields against v1 mesh truth.";
#endif

typedef struct {
    Array values, valid, quality;
    double origin[3], spacing;
    int nearest;
} LegacyField;

typedef struct {
    i32 truth;
    double value, theta, index, field, delta, phase, residual;
    i64 offset, frame_mode, mesh_mode;
    f32 quality;
} LegacyWindingPoint;

static int legacy_field_sample(const LegacyField *f, const f32 *xyz, double *value, f32 *quality) {
    double position[3], frac[3];
    i64 base[3], stride[] = {f->values.shape[1] * f->values.shape[2], f->values.shape[2], 1};
    for (int d = 0; d < 3; d++) {
        i64 top = f->values.shape[d] - 1;
        position[d] = (xyz[2 - d] - f->origin[d]) / f->spacing;
        if (position[d] < 0 || position[d] > top) {
            return 0;
        }
        base[d] = f->nearest ? nearbyint(position[d]) : floor(position[d]);
        if (!f->nearest && top && base[d] == top) {
            base[d]--;
        }
        frac[d] = position[d] - base[d];
    }
    *value = 0;
    double confidence = 0;
    int known = 1;
    for (int corner = 0; corner < (f->nearest ? 1 : 8); corner++) {
        double weight = 1;
        i64 index = 0;
        for (int d = 0; d < 3; d++) {
            int high = (corner >> (2 - d)) & 1;
            index += (base[d] + (high && base[d] + 1 < f->values.shape[d])) * stride[d];
            if (!f->nearest) {
                weight *= high ? frac[d] : 1 - frac[d];
            }
        }
        double v = bench_number(f->values, index);
        if (weight > 0) {
            known &= isfinite(v) && (!f->valid.data || bench_number(f->valid, index) != 0);
            if (isfinite(v)) {
                *value += v * weight;
            }
        }
        if (f->quality.data) {
            confidence += weight * bench_number(f->quality, index);
        }
    }
    *quality = confidence;
    return known ? 2 : 1;
}

static double legacy_relative(double theta, double psi) {
    return atan2(sin(theta - psi), cos(theta - psi));
}

static i64 legacy_winding_trial(Arena *a, const BenchLegacyTruth *t, LegacyWindingPoint *point,
    const i32 *pick, i64 n, double psi, int is_angle, int chirality, int retain, double *strength) {
    size_t saved = a->used;
    LegacyPhase *phases = push(a, n * sizeof(*phases));
    BenchVote *vote = push(a, n * sizeof(*vote));
    for (i64 row = 0; row < n; row++) {
        i64 i = pick ? pick[row] : row;
        LegacyWindingPoint *v = point + i;
        i32 j = v->truth;
        double relative = legacy_relative(v->theta, psi) / (2 * M_PI);
        double field = v->value - (is_angle ? relative : 0);
        double index = t->winding[j] - t->sense * relative;
        phases[row] = (LegacyPhase){.vertex = i,
            .coarse = t->frame[j],
            .mesh = t->mesh[j],
            .delta = field - chirality * index};
        if (retain) {
            v->index = index;
            v->field = field;
        }
    }
    *strength = legacy_phases(phases, n);
    for (i64 i = 0; i < n; i++) {
        vote[i] = (BenchVote){phases[i].coarse, nearbyint(phases[i].delta - phases[i].phase)};
    }
    BenchModes modes = legacy_modes(vote, n);
    i64 right = 0;
    for (i64 i = 0; i < n; i++) {
        LegacyPhase p = phases[i];
        i64 offset = nearbyint(p.delta - p.phase), mode = 0;
        bench_mode(&modes, p.coarse, &mode);
        right += offset == mode;
        if (retain) {
            LegacyWindingPoint *v = point + p.vertex;
            v->delta = p.delta;
            v->phase = p.phase;
            v->offset = offset;
            v->frame_mode = mode;
            v->residual = p.delta - p.phase - offset;
        }
    }
    a->used = saved;
    return right;
}

static i32 *legacy_winding_choose(Arena *a, Pcg64 *rng, i64 n, i64 *size, i64 limit) {
    *size = n < limit ? n : limit;
    i32 *pick = push(a, n * sizeof(*pick));
    for (i64 i = 0; i < n; i++) {
        pick[i] = i;
    }
    package_choose(a, rng, pick, n, *size);
    return pick;
}

typedef struct {
    double psi, span, strength, other;
    int chirality, searched;
} LegacyWindingTrial;

static LegacyWindingTrial legacy_winding_align(Arena *a, const BenchLegacyTruth *t,
    LegacyWindingPoint *v, i64 n, int is_angle, const char *cut, Pcg64 *rng) {
    double re = 0, im = 0;
    for (i64 i = 0; i < n; i++) {
        re += cos(v[i].theta);
        im += sin(v[i].theta);
    }
    double psi = atan2(im, re), lo = INFINITY, hi = -INFINITY;
    for (i64 i = 0; i < n; i++) {
        double rel = legacy_relative(v[i].theta, psi);
        lo = fmin(lo, rel);
        hi = fmax(hi, rel);
    }
    double span = (hi - lo) * 180 / M_PI;
    int searched = cut ? !strcmp(cut, "search") : !is_angle && span > 300;
    double strength;
    if (searched) {
        size_t saved = a->used;
        i64 sample;
        i32 *pick = legacy_winding_choose(a, rng, n, &sample, 200000);
        i64 best = -1;
        for (int deg = 0; deg < 360; deg += 5) {
            double angle = deg * (M_PI / 180);
            i64 one = legacy_winding_trial(a, t, v, pick, sample, angle, 0, 1, 0, &strength);
            i64 reverse = legacy_winding_trial(a, t, v, pick, sample, angle, 0, -1, 0, &strength);
            i64 right = one > reverse ? one : reverse;
            if (right >= best) {
                best = right;
                psi = angle;
            }
        }
        a->used = saved;
    } else if (cut) {
        psi = board_value(cut) * (M_PI / 180) - M_PI;
    }
    i64 one = legacy_winding_trial(a, t, v, NULL, n, psi, is_angle, 1, 0, &strength);
    i64 reverse = legacy_winding_trial(a, t, v, NULL, n, psi, is_angle, -1, 0, &strength);
    int chirality = one >= reverse ? 1 : -1;
    legacy_winding_trial(a, t, v, NULL, n, psi, is_angle, chirality, 1, &strength);
    return (LegacyWindingTrial){.psi = psi,
        .span = span,
        .searched = searched,
        .chirality = chirality,
        .strength = strength,
        .other = (double)(chirality > 0 ? reverse : one) / n};
}

typedef struct {
    i64 count, right, mesh_right, global_right;
    int mesh, wrap;
    const char *name;
    i64 mesh_mode;
} LegacyWindingCensus;

static int legacy_winding_mesh_order(const void *left, const void *right) {
    const LegacyWindingCensus *a = left, *b = right;
    if (a->wrap != b->wrap) {
        return a->wrap - b->wrap;
    }
    return strcmp(a->name, b->name);
}

static json_object *legacy_winding_global(Arena *a, const BenchLegacyTruth *t,
    LegacyWindingPoint *v, i64 n, LegacyWindingTrial trial, int is_angle, json_object *out) {
    size_t saved = a->used;
    BenchVote *vote = push(a, n * sizeof(*vote));
    LegacyWindingCensus *meshes = pushz(a, t->meshes * sizeof(*meshes));
    i64 frame_right = 0, mesh_right = 0, global_right = 0, best_frame = -1;
    i64 min_frame = INT32_MAX, max_frame = INT32_MIN;
    double re = 0, im = 0;
    i64 bins[20] = {0}, offsets[7] = {0}, within_a = 0, within_b = 0;
    for (i64 i = 0; i < n; i++) {
        vote[i] = (BenchVote){0, v[i].offset};
        i64 frame = t->frame[v[i].truth];
        min_frame = min_frame < frame ? min_frame : frame;
        max_frame = max_frame > frame ? max_frame : frame;
        re += cos(2 * M_PI * v[i].delta);
        im += sin(2 * M_PI * v[i].delta);
        within_a += fabs(v[i].residual) <= 0.1;
        within_b += fabs(v[i].residual) <= 0.25;
        int bin = floor((v[i].residual + 0.5) * 20);
        if (bin >= 0 && bin <= 20) {
            bins[bin < 20 ? bin : 19]++;
        }
    }
    BenchModes global = legacy_modes(vote, n);
    i64 global_offset = global.vote[0].turn;
    for (i64 i = 0; i < n; i++) {
        vote[i] = (BenchVote){t->mesh[v[i].truth], v[i].offset};
    }
    BenchModes local = legacy_modes(vote, n);
    i64 *frame = pushz(a, (max_frame - min_frame + 1) * sizeof(*frame));
    i64 *correct = pushz(a, (max_frame - min_frame + 1) * sizeof(*correct));
    i64 frames = 0;
    for (i64 i = 0; i < n; i++) {
        LegacyWindingPoint *p = v + i;
        int mesh = t->mesh[p->truth];
        i64 f = t->frame[p->truth] - min_frame;
        bench_mode(&local, mesh, &p->mesh_mode);
        int right = p->offset == p->frame_mode, local_ok = p->offset == p->mesh_mode;
        int global_ok = p->offset == global_offset;
        frame_right += right;
        mesh_right += local_ok;
        global_right += global_ok;
        frame[f]++;
        correct[f] += right;
        LegacyWindingCensus *m = meshes + mesh;
        m->count++;
        m->mesh_right += local_ok;
        m->global_right += global_ok;
        m->mesh_mode = p->mesh_mode;
        i64 diff = p->offset - global_offset;
        offsets[diff < -3 ? 0 : diff > 3 ? 6 : diff + 3]++;
    }
    for (i64 f = 0; f <= max_frame - min_frame; f++) {
        if (frame[f]) {
            frames++;
            if (best_frame < 0 || frame[f] > frame[best_frame]) {
                best_frame = f;
            }
        }
    }
    json_object_object_add(out, "convention", json_object_new_string(is_angle ? "angle" : "level"));
    json_object_object_add(out, "chirality", json_object_new_int(trial.chirality));
    bench_decimal(out, "share_other_chirality", trial.other, 4);
    bench_decimal(out, "box_angle_span_deg", trial.span, 1);
    json_object_object_add(out, "encircles_axis", json_object_new_boolean(trial.span > 300));
    bench_decimal(out, "truth_cut_deg", legacy_relative(trial.psi + M_PI, 0) * 180 / M_PI, 1);
    json_object_object_add(out, "truth_cut_searched", json_object_new_boolean(trial.searched));
    bench_decimal(out, "phase", legacy_phase(re, im), 4);
    bench_decimal(out, "phase_strength", hypot(re, im) / n, 4);
    bench_decimal(out, "phase_strength_within_mesh", trial.strength, 4);
    bench_ratio(out, "phase_within_0.1", within_a, n, 4);
    bench_ratio(out, "phase_within_0.25", within_b, n, 4);
    json_object *histogram = json_object_new_array_ext(20);
    i64 used = 0;
    for (int i = 0; i < 20; i++) {
        used += bins[i];
    }
    for (int i = 0; i < 20; i++) {
        json_object_array_add(histogram, jnum(package_round((double)bins[i] / used, 4)));
    }
    json_object_object_add(out, "phase_histogram", histogram);
    bench_ratio(out, "share_global", global_right, n, 4);
    bench_ratio(out, "share_frames", frame_right, n, 4);
    bench_ratio(out, "share_mesh", mesh_right, n, 4);
    json_object_object_add(out, "frames", json_object_new_int64(frames));
    json_object *distribution = json_object_new_object();
    const char *offset_keys[] = {"other", "-2", "-1", "0", "1", "2", "other"};
    for (int i = 0; i < 7; i++) {
        if (offsets[i]) {
            bench_ratio(distribution, offset_keys[i], offsets[i], n, 4);
        }
    }
    json_object_object_add(out, "offsets_global", distribution);
    bench_ratio(out, "share_largest_frame", correct[best_frame], frame[best_frame], 4);
    int mesh_n = 0;
    for (int i = 0; i < t->meshes; i++) {
        if (meshes[i].count) {
            LegacyWindingCensus m = meshes[i];
            m.mesh = i;
            m.wrap = t->wrap[i];
            m.name = board_item(t->names, i);
            meshes[mesh_n++] = m;
        }
    }
    qsort(meshes, mesh_n, sizeof(*meshes), legacy_winding_mesh_order);
    json_object *per_mesh = json_object_new_array_ext(mesh_n);
    for (int i = 0; i < mesh_n; i++) {
        LegacyWindingCensus m = meshes[i];
        json_object *item = json_object_new_object();
        json_object_array_add(per_mesh, item);
        json_object_object_add(item, "mesh", json_object_new_string(m.name));
        json_object_object_add(item, "wrap", json_object_new_int(m.wrap));
        json_object_object_add(item, "vertices", json_object_new_int64(m.count));
        bench_ratio(item, "share_on_global", m.global_right, m.count, 4);
        bench_ratio(item, "share_on_own", m.mesh_right, m.count, 4);
        json_object_object_add(
            item, "own_minus_global", json_object_new_int64(m.mesh_mode - global_offset));
    }
    json_object_object_add(out, "per_mesh", per_mesh);
    a->used = saved;
    return out;
}

static void legacy_winding_footing(Arena *a, const BenchLegacyTruth *t, const LegacyWindingPoint *v,
    i64 n, i64 resolved, json_object *out) {
    size_t saved = a->used;
    BenchVote *vote = push(a, resolved * sizeof(*vote));
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        int j = v[i].truth;
        if (!t->unresolved[j]) {
            vote[count++] = (BenchVote){t->frame[j], v[i].offset};
        }
    }
    BenchModes modes = legacy_modes(vote, count);
    i64 *frames = pushz(a, modes.n * sizeof(*frames));
    i64 *correct = pushz(a, modes.n * sizeof(*correct));
    i64 all = 0;
    for (i64 i = 0; i < n; i++) {
        int j = v[i].truth;
        if (t->unresolved[j]) {
            continue;
        }
        i64 lo = 0, hi = modes.n, frame = t->frame[j];
        while (lo < hi) {
            i64 k = (lo + hi) / 2;
            if (modes.vote[k].key < frame) {
                lo = k + 1;
            } else {
                hi = k;
            }
        }
        int right = v[i].offset == modes.vote[lo].turn;
        frames[lo]++;
        correct[lo] += right;
        all += right;
    }
    i64 largest = 0;
    for (i64 i = 1; i < modes.n; i++) {
        if (frames[i] > frames[largest]) {
            largest = i;
        }
    }
    json_object *foot = json_object_new_object();
    bench_ratio(foot, "share_largest_frame_resolved", correct[largest], frames[largest], 4);
    bench_ratio(foot, "share_frames_resolved", all, resolved, 4);
    json_object_object_add(foot, "resolved_vertices", json_object_new_int64(resolved));
    json_object_object_add(out, "v1_footing", foot);
    a->used = saved;
}

static void legacy_winding_adjacency(Arena *a, const BenchLegacyTruth *t,
    const LegacyWindingPoint *v, i64 n, int chirality, Pcg64 *rng, json_object *out) {
    size_t saved = a->used;
    f32 *points = push(a, n * 3 * sizeof(*points));
    for (i64 i = 0; i < n; i++) {
        memcpy(points + 3 * i, t->points + 3LL * v[i].truth, 3 * sizeof(*points));
    }
    BenchTree tree = bench_tree(a, points, n);
    i64 size;
    i32 *pick = legacy_winding_choose(a, rng, n, &size, 100000);
    double reach = 400 / t->voxel, near = 80 / t->voxel, lateral = 160 / t->voxel;
    i64 pairs = 0, steps[5] = {0}, press[2] = {0}, right[2] = {0};
    for (i64 j = 0; j < size; j++) {
        i32 i = pick[j];
        const f32 *p = points + 3LL * i, *normal = t->normals + 3LL * v[i].truth;
        BenchCandidate hit[32];
        int used = 0, pressed = 0, radial = 0;
        bench_candidates_limit(&tree, 0, p, reach * reach, hit, &used, 32);
        for (int k = 0; k < used; k++) {
            i32 ix = hit[k].index;
            double step = v[ix].index - v[i].index;
            if (fabs(step) < 0.5) {
                continue;
            }
            double along = fabs(package_projection(p, points + 3LL * ix, normal));
            double across = fmax(hit[k].squared - along * along, 0);
            pressed |= along <= near && across <= lateral * lateral;
            double turn = nearbyint(step);
            if (!radial && fabs(turn) == 1 && along >= 0.8 * sqrt(along * along + across)) {
                radial = 1;
                pairs++;
                double got = nearbyint(v[ix].field - v[i].field) * chirality * turn;
                int bin = got < -1 ? 0 : got > 3 ? 4 : (int)got + 1;
                steps[bin]++;
            }
        }
        if (!t->unresolved) {
            press[pressed]++;
            right[pressed] += v[i].offset == v[i].frame_mode;
        }
    }
    json_object *adj = json_object_new_object(), *bins = json_object_new_object(),
                *pressed = json_object_new_object();
    json_object_object_add(adj, "pairs", json_object_new_int64(pairs));
    bench_ratio(adj, "exact", steps[2], pairs ? pairs : 1, 4);
    json_object_object_add(adj, "steps", bins);
    const char *labels[] = {"backward", "same sheet", "one", "two", "three or more"};
    for (int i = 0; i < 5; i++) {
        if (steps[i]) {
            bench_ratio(bins, labels[i], steps[i], pairs, 4);
        }
    }
    json_object_object_add(adj, "sampled", json_object_new_int64(size));
    json_object_object_add(out, "adjacency", adj);
    if (t->unresolved) {
        size = n;
        for (i64 i = 0; i < n; i++) {
            int p = t->unresolved[v[i].truth] != 0;
            press[p]++;
            right[p] += v[i].offset == v[i].frame_mode;
        }
        json_object_object_add(pressed, "source", json_object_new_string("v1 unresolved"));
        if (press[0]) {
            legacy_winding_footing(a, t, v, n, press[0], out);
        }
    } else {
        json_object_object_add(
            pressed, "source", json_object_new_string("another truth sheet within 80 µm, sampled"));
        json_object_object_add(pressed, "sampled", json_object_new_int64(size));
    }
    bench_ratio(pressed, "share_of_truth", press[1], size, 4);
    bench_ratio(pressed, "share_frames_pressed", right[1], press[1], 4);
    bench_ratio(pressed, "share_frames_resolved", right[0], press[0], 4);
    json_object_object_add(out, "pressed", pressed);
    a->used = saved;
}

typedef struct {
    f32 value;
    i32 i;
} LegacyQuality;

static int legacy_quality_order(const void *left, const void *right) {
    const LegacyQuality *a = left, *b = right;
    if (isnan(a->value) != isnan(b->value)) {
        return isnan(a->value) ? 1 : -1;
    }
    if (a->value != b->value && !isnan(a->value)) {
        return a->value > b->value ? -1 : 1;
    }
    return a->i - b->i;
}

static void legacy_winding_risk(
    Arena *a, const LegacyWindingPoint *v, i64 n, i64 boxed, json_object *out) {
    size_t saved = a->used;
    LegacyQuality *quality = push(a, n * sizeof(*quality));
    i64 *wrong = push(a, n * sizeof(*wrong));
    for (i64 i = 0; i < n; i++) {
        quality[i] = (LegacyQuality){v[i].quality, i};
    }
    qsort(quality, n, sizeof(*quality), legacy_quality_order);
    const double rates[] = {0.001, 0.01, 0.05};
    const char *rate_key[] = {"0.001", "0.01", "0.05"};
    for (int mesh = 0; mesh < 2; mesh++) {
        i64 count = 0, reaches[3] = {0};
        for (i64 i = 0; i < n; i++) {
            const LegacyWindingPoint *p = v + quality[i].i;
            count += p->offset != (mesh ? p->mesh_mode : p->frame_mode);
            wrong[i] = count;
            for (int j = 0; j < 3; j++) {
                if ((double)count / (i + 1) <= rates[j]) {
                    reaches[j] = i + 1;
                }
            }
        }
        json_object *result = json_object_new_object(), *curve = json_object_new_array_ext(20),
                    *at = json_object_new_object();
        for (int j = 0; j < 20; j++) {
            i64 k = nearbyint((1 - 0.05 * j) * n);
            k = k > 0 ? k : 1;
            json_object *item = json_object_new_object();
            bench_decimal(item, "quality_floor", quality[k - 1].value, 5);
            bench_ratio(item, "coverage", k, boxed, 4);
            bench_ratio(item, "wrong_rate", wrong[k - 1], k, 5);
            json_object_array_add(curve, item);
        }
        for (int j = 0; j < 3; j++) {
            bench_ratio(at, rate_key[j], reaches[j], boxed, 4);
        }
        json_object_object_add(result, "curve", curve);
        json_object_object_add(result, "coverage_at_wrong_rate", at);
        json_object_object_add(out, mesh ? "risk_coverage_mesh" : "risk_coverage_frames", result);
    }
    a->used = saved;
}

static void legacy_winding_errors(Arena *a, const char *path, const BenchLegacyTruth *t,
    const LegacyWindingPoint *v, i64 n, int quality) {
    size_t saved = a->used;
    f32 *points = push(a, n * 3 * sizeof(*points)), *d = push(a, n * sizeof(*d)),
        *q = push(a, quality ? n * sizeof(*q) : 0);
    i8 *error = push(a, n);
    i32 *mesh = push(a, n * sizeof(*mesh));
    for (i64 i = 0; i < n; i++) {
        memcpy(points + 3 * i, t->points + 3LL * v[i].truth, 3 * sizeof(*points));
        mesh[i] = t->mesh[v[i].truth];
        i64 fail = v[i].offset - v[i].frame_mode;
        error[i] = fail < -127 ? -127 : fail > 127 ? 127 : fail;
        d[i] = v[i].delta;
        if (quality) {
            q[i] = v[i].quality;
        }
    }
    const char *keys[] = {"points", "mesh", "error", "d", "quality"};
    Array arrays[] = {
        {.descr = "<f4", .ndim = 2, .shape = {n, 3}, .data = points, .size = n * 12},
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = mesh, .size = n * 4},
        {.descr = "|i1", .ndim = 1, .shape = {n}, .data = error, .size = n},
        {.descr = "<f4", .ndim = 1, .shape = {n}, .data = d, .size = n * 4},
        {.descr = "<f4",
            .ndim = 1,
            .shape = {quality ? n : 0},
            .data = q,
            .size = quality ? n * 4 : 0},
    };
    npz_write_codec(path, sizeof(keys) / sizeof(*keys), keys, arrays, ZIP_CM_DEFLATE);
    a->used = saved;
}

static json_object *legacy_winding_score(Arena *a, const BenchLegacyTruth *t, const LegacyField *f,
    int angle, const char *cut, const char *errors) {
    LegacyWindingPoint *v = push(a, t->n * sizeof(*v));
    i64 count = 0, boxed = 0;
    for (i64 i = 0; i < t->n; i++) {
        const f32 *xyz = t->points + 3 * i;
        if (!isfinite(t->winding[i])) {
            continue;
        }
        double value;
        f32 quality;
        int ok = legacy_field_sample(f, xyz, &value, &quality);
        boxed += ok != 0;
        if (ok == 2) {
            double x = interpolated(xyz[2], t->knots, t->nknots, 1);
            double y = interpolated(xyz[2], t->knots, t->nknots, 2);
            v[count++] = (LegacyWindingPoint){.truth = i,
                .value = value,
                .quality = quality,
                .theta = atan2(xyz[1] - y, xyz[0] - x)};
        }
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "truth_in_box", json_object_new_int64(boxed));
    json_object_object_add(out, "truth_valid", json_object_new_int64(count));
    bench_ratio(out, "coverage", count, boxed ? boxed : 1, 4);
    if (count < 100) {
        json_object_object_add(out, "error",
            json_object_new_string("fewer than 100 truth vertices with a valid field"));
        return out;
    }
    Pcg64 rng = pcg64(0);
    LegacyWindingTrial trial = legacy_winding_align(a, t, v, count, angle, cut, &rng);
    legacy_winding_global(a, t, v, count, trial, angle, out);
    legacy_winding_adjacency(a, t, v, count, trial.chirality, &rng, out);
    if (f->quality.data) {
        legacy_winding_risk(a, v, count, boxed, out);
    }
    if (errors) {
        legacy_winding_errors(a, errors, t, v, count, f->quality.data != NULL);
    }
    return out;
}

static json_object *legacy_winding_headline(Arena *a, json_object *out) {
    json_object *metrics = json_object_new_object();
    const char *keys[] = {"coverage", "truth_in_box", "truth_valid", "chirality", "phase_strength",
        "phase_strength_within_mesh", "share_global", "share_frames", "share_mesh",
        "share_largest_frame"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) {
        json_object *value = json_object_object_get(out, keys[i]);
        if (value) {
            json_object_object_add(metrics, keys[i], json_object_get(value));
        }
    }
    json_object *adjacency = json_object_object_get(out, "adjacency");
    if (adjacency) {
        json_object_object_add(
            metrics, "adjacent_exact", json_object_get(jget(adjacency, "exact")));
    }
    json_object *foot = json_object_object_get(out, "v1_footing");
    if (foot) {
        json_object_object_foreach(foot, key, value) {
            if (!strncmp(key, "share", 5)) {
                json_object_object_add(metrics, fmt(a, "v1_%s", key), json_object_get(value));
            }
        }
    }
    json_object *pressed = json_object_object_get(out, "pressed");
    if (pressed) {
        const char *names[] = {"share_of_truth", "share_frames_resolved", "share_frames_pressed"};
        for (int i = 0; i < 3; i++) {
            json_object *value = json_object_object_get(pressed, names[i]);
            if (value) {
                json_object_object_add(
                    metrics, fmt(a, "pressed_%s", names[i]), json_object_get(value));
            }
        }
    }
    json_object *coverage = json_object_object_get(out, "risk_coverage_frames");
    if (coverage) {
        json_object_object_foreach(jget(coverage, "coverage_at_wrong_rate"), rate, value) {
            json_object_object_add(
                metrics, fmt(a, "coverage_at_wrong_%s", rate), json_object_get(value));
        }
    }
    return metrics;
}

typedef struct {
    Bytes bytes;
    Array array;
} LegacyMapped;

static LegacyMapped legacy_map_npy(const char *path) {
    if (!path) {
        return (LegacyMapped){0};
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st;
    need(fd >= 0 && !fstat(fd, &st) && st.st_size > 0, "cannot read %s", path);
    void *memory = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    need(memory != MAP_FAILED, "cannot map %s", path);
    Bytes bytes = {.data = memory, .size = st.st_size};
    return (LegacyMapped){.bytes = bytes, .array = npy_parse(bytes, path)};
}

static void legacy_unmap(LegacyMapped m) {
    if (m.bytes.data) {
        munmap(m.bytes.data, m.bytes.size);
    }
}

static void legacy_npy_shape(Array values, Array other, const char *name) {
    if (other.data) {
        need(other.ndim == 3 && other.shape[0] == values.shape[0] &&
                other.shape[1] == values.shape[1] && other.shape[2] == values.shape[2],
            "%s array does not match the winding field", name);
    }
}

static json_object *legacy_string(const char *value) {
    return value ? json_object_new_string(value) : NULL;
}

static json_object *legacy_three(const double d[3]) {
    json_object *array = json_object_new_array_ext(3);
    for (int i = 0; i < 3; i++) {
        json_object_array_add(array, json_object_new_double(d[i]));
    }
    return array;
}

int bench_winding_score_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.winding_score CONTROL --field FILE --origin Z,Y,X "
             "[--level N] [--spacing VOX] [--cut DEG|search] [--valid FILE] [--quality FILE] "
             "[--convention level|angle] [--nearest] [--truth auto|names|groups] [--every N] "
             "[--errors FILE] [--method NAME] [--system NAME] [--no-record]");
        return !argc;
    }
    Arena a = arena(32ull << 30);
    const char *control = argv[0], *field = NULL, *origin = NULL, *source = "auto", *cut = NULL,
               *valid = NULL, *quality = NULL, *errors = NULL, *method = NULL, *system = "network";
    int level = 0, every = 1, record = 1, angle = 0, nearest = 0;
    double spacing = 0;
    for (int i = 1; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--no-record")) {
            record = 0;
        } else if (!strcmp(option, "--nearest")) {
            nearest = 1;
        } else {
            need(i + 1 < argc, "missing value for %s", option);
            const char *value = argv[++i];
            if (!strcmp(option, "--field")) {
                field = value;
            } else if (!strcmp(option, "--origin")) {
                origin = value;
            } else if (!strcmp(option, "--truth")) {
                source = value;
            } else if (!strcmp(option, "--cut")) {
                cut = value;
            } else if (!strcmp(option, "--level")) {
                level = board_value(value);
            } else if (!strcmp(option, "--spacing")) {
                spacing = board_value(value);
            } else if (!strcmp(option, "--valid")) {
                valid = value;
            } else if (!strcmp(option, "--quality")) {
                quality = value;
            } else if (!strcmp(option, "--errors")) {
                errors = value;
            } else if (!strcmp(option, "--every")) {
                every = board_value(value);
            } else if (!strcmp(option, "--method")) {
                method = value;
            } else if (!strcmp(option, "--system")) {
                system = value;
            } else if (!strcmp(option, "--convention")) {
                need(!strcmp(value, "angle") || !strcmp(value, "level"), "invalid convention %s",
                    value);
                angle = !strcmp(value, "angle");
            } else {
                die("unknown winding score option %s", option);
            }
        }
    }
    need(field && origin && every > 0,
        "winding scorer needs --field, --origin and positive --every");
    LegacyMapped values = legacy_map_npy(field), ok = legacy_map_npy(valid),
                 q = legacy_map_npy(quality);
    double step = spacing ? spacing : exp2(level);
    LegacyField f = {.values = values.array,
        .valid = ok.array,
        .quality = q.array,
        .nearest = nearest,
        .spacing = step};
    need(f.values.ndim == 3 && f.values.shape[0] > 0 && f.values.shape[1] > 0 &&
            f.values.shape[2] > 0 && isfinite(step) && step > 0,
        "invalid winding field or spacing");
    legacy_npy_shape(f.values, f.valid, "valid");
    legacy_npy_shape(f.values, f.quality, "quality");
    char *triple = fmt(&a, "%s", origin);
    double box[2][3];
    for (int i = 0; i < 3; i++) {
        const char *value = strsep(&triple, ",");
        need(value && *value, "--origin needs z,y,x");
        f.origin[i] = board_value(value);
        box[0][2 - i] = f.origin[i];
        box[1][2 - i] = f.origin[i] + step * (f.values.shape[i] - 1) + 1e-3;
    }
    need(!triple, "--origin needs z,y,x");
    const char *selected = !strcmp(source, "auto") ? legacy_default(&a, control) : source;
    need(selected && (!strcmp(selected, "names") || !strcmp(selected, "groups")),
        "invalid truth source");
    BenchLegacyTruth t = legacy_truth(&a, control, selected);
    if (!t.normals) {
        t.normals = legacy_group_normals(&a, &t);
    }
    legacy_subset(&t, every, box);
    json_object *out = legacy_winding_score(&a, &t, &f, angle, cut, errors);
    json_object *manifest = json_read(fmt(&a, "%s/calibrate/bench/controls.json", HOME));
    need(manifest != NULL, "cannot read bench controls");
    const char *split = bench_split(&a, manifest, control);
    json_object_object_add(out, "control", json_object_new_string(control));
    json_object_object_add(out, "truth", json_object_new_string(selected));
    json_object_object_add(out, "split", json_object_new_string(split));
    puts(board_json(out));
    if (record && !json_object_object_get(out, "error")) {
        json_object *params = json_object_new_object(), *box_json = json_object_new_array_ext(2);
        json_object_array_add(box_json, legacy_three(box[0]));
        json_object_array_add(box_json, legacy_three(box[1]));
        json_object_object_add(params, "box_xyz", box_json);
        json_object_object_add(params, "control", legacy_string(control));
        json_object_object_add(params, "convention", legacy_string(angle ? "angle" : "level"));
        json_object_object_add(params, "cut", legacy_string(cut));
        json_object_object_add(params, "errors", legacy_string(errors));
        json_object_object_add(params, "every", json_object_new_int(every));
        json_object_object_add(params, "field", legacy_string(field));
        json_object_object_add(params, "level", json_object_new_int(level));
        json_object_object_add(params, "method", legacy_string(method));
        json_object_object_add(params, "nearest", json_object_new_boolean(nearest));
        json_object_object_add(params, "no_record", json_object_new_boolean(0));
        json_object_object_add(params, "origin", legacy_three(f.origin));
        json_object_object_add(params, "quality", legacy_string(quality));
        json_object_object_add(params, "spacing", json_object_new_double(step));
        json_object_object_add(params, "system", legacy_string(system));
        json_object_object_add(params, "truth", legacy_string(source));
        json_object_object_add(params, "valid", legacy_string(valid));
        const char *basename = bench_basename(field), *dot = strrchr(basename, '.');
        const char *label = method ? method
            : dot                  ? fmt(&a, "%.*s", (int)(dot - basename), basename)
                                   : basename;
        json_object *metrics = legacy_winding_headline(&a, out);
        const char *json = json_object_to_json_string_ext(
            params, JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_NOSLASHESCAPE);
        legacy_record(&a, control, system, label, "winding", split, json, metrics, out);
        json_object_put(metrics);
        json_object_put(params);
    }
    legacy_truth_close(&t);
    json_object_put(manifest);
    json_object_put(out);
    legacy_unmap(values);
    legacy_unmap(ok);
    legacy_unmap(q);
    munmap(a.base, a.size);
    return 0;
}
