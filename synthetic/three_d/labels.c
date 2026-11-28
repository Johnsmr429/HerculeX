typedef struct {
    double winding, nearest, theta, z;
    i32 sheet, level;
    u8 matched, judged, pressed, outward, invalid, near;
} SynLabel;

typedef struct {
    int piece, value;
    size_t node;
    double weight;
} SynVote;
typedef struct {
    i32 root, count;
} SynRank;

static int syn_vote_order(const void *left, const void *right) {
    const SynVote *a = left, *b = right;
    if (a->piece != b->piece) {
        return (a->piece > b->piece) - (a->piece < b->piece);
    }
    if (a->value != b->value) {
        return (a->value > b->value) - (a->value < b->value);
    }
    return (a->node > b->node) - (a->node < b->node);
}

static int syn_rank_order(const void *left, const void *right) {
    const SynRank *a = left, *b = right;
    if (a->count != b->count) {
        return (a->count < b->count) - (a->count > b->count);
    }
    return (a->root > b->root) - (a->root < b->root);
}

static int syn_double_order(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static void syn_votes(SynVote *votes, size_t n, i32 *mode, double *best) {
    qsort(votes, n, sizeof(*votes), syn_vote_order);
    for (size_t i = 0; i < n;) {
        SynVote at = votes[i++];
        double weight = at.weight;
        while (i < n && votes[i].piece == at.piece && votes[i].value == at.value) {
            weight += votes[i++].weight;
        }
        if (weight > best[at.piece]) {
            mode[at.piece] = at.value;
            best[at.piece] = weight;
        }
    }
}

typedef struct {
    size_t n;
    const SynKernel *kernel;
    const float *xyz, *theta;
    const i8 *normal;
    const double *origin;
    double voxel, shift, scale;
    SynLabel *labels;
} SynLabelling;

static void syn_node_label(void *data, i64 block) {
    SynLabelling *w = data;
    size_t low = 2048 * (size_t)block, high = low + 2048;
    if (high > w->n) {
        high = w->n;
    }
    for (size_t i = low; i < high; i++) {
        double z = w->xyz[3 * i + 2] * w->scale + w->shift;
        double y = w->xyz[3 * i + 1] * w->scale + w->shift;
        double x = w->xyz[3 * i] * w->scale + w->shift;
        SynTruthPoint at = syn_truth_at(w->kernel, w->origin[0] + z * w->voxel,
            w->origin[1] + y * w->voxel, w->origin[2] + x * w->voxel);
        const double *normal = at.normal;
        double dot = w->normal[3 * i + 2] * normal[0] + w->normal[3 * i + 1] * normal[1] +
            w->normal[3 * i] * normal[2];
        double distance = fabs(at.nearest);
        int near = distance <= fmax(25, at.thickness / 2 + 10);
        SynLabel label = {.winding = at.winding,
            .nearest = at.nearest,
            .theta = at.sheet_winding,
            .z = z,
            .sheet = at.sheet,
            .outward = dot > 0,
            .invalid = !at.valid,
            .near = near,
            .matched = at.valid && near && at.second > distance + 10,
            .pressed = at.pitch <= 80};
        w->labels[i] = label;
    }
}

static json_object *syn_band_labels(Arena *a, const SynScroll *scroll, json_object *band,
    const char *out, const float *positions, size_t n, const double *origin, int workers) {
    const char *nodes = fmt(a, "%s/nodes.npz", out), *levels = fmt(a, "%s/levels.npz", out);
    const i32 *root = npz_get(a, nodes, "root").data,
              *potential = npz_get(a, nodes, "potential").data;
    const float *theta = npz_get(a, nodes, "theta").data, *area = positions + 3 * n;
    const i32 *node_level = npz_get(a, nodes, "node_level").data;
    const i8 *normal = npz_get(a, nodes, "normal").data;
    Array level = npz_get(a, levels, "level");
    const i32 *piece_level = level.data;
    int curl = *(const i64 *)npz_get(a, levels, "chirality").data;
    int pieces = level.shape[0];
    double start = now(), scale = ldexp(1, ji64(band, "level"));
    SynLabel *truth = push(a, n * sizeof(*truth));
    SynLabelling w = {.n = n,
        .kernel = &scroll->kernel,
        .xyz = positions,
        .theta = theta,
        .normal = normal,
        .origin = origin,
        .voxel = jf64(band, "voxel_um"),
        .shift = .5 * (scale - 1),
        .scale = scale,
        .labels = truth};
    parallel((n + 2047) / 2048, workers, syn_node_label, &w);
    double *errors = push(a, n * sizeof(*errors));
    int sense = 1;
    double phase = 0, best = INFINITY;
    for (int candidate = 1; candidate >= -1; candidate -= 2) {
        double re = 0, im = 0;
        size_t matched = 0;
        for (size_t i = 0; i < n; i++) {
            if (truth[i].matched) {
                double u = truth[i].theta - candidate * theta[i] / (2 * M_PI);
                re += cos(2 * M_PI * u);
                im += sin(2 * M_PI * u);
                errors[matched++] = u;
            }
        }
        double p = matched ? atan2(im, re) / (2 * M_PI) : 0;
        for (size_t i = 0; i < matched; i++) {
            double u = errors[i] - p;
            errors[i] = fabs(u - nearbyint(u));
        }
        qsort(errors, matched, sizeof(*errors), syn_double_order);
        double spread = matched ? (errors[matched / 2] + errors[(matched - 1) / 2]) * .5 : 1;
        if (spread < best) {
            sense = candidate;
            phase = p;
            best = spread;
        }
    }
    i32 *rank = pushz(a, n * sizeof(*rank));
    SynRank *ranks = push(a, (size_t)pieces * sizeof(*ranks));
    for (size_t i = 0; i < n; i++) {
        rank[root[i]]++;
    }
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        if (rank[i] >= 400) {
            need(count < pieces, "synthetic level ranks differ from the fit");
            ranks[count++] = (SynRank){.root = i, .count = rank[i]};
        }
        rank[i] = -1;
    }
    need(count == pieces, "synthetic level ranks differ from the fit");
    qsort(ranks, pieces, sizeof(*ranks), syn_rank_order);
    for (int i = 0; i < pieces; i++) {
        rank[ranks[i].root] = i;
    }
    u8 *judged = push(a, n), *pressed = push(a, n), *outward = push(a, n), *val = pushz(a, n);
    i32 *frame = push(a, n * sizeof(*frame)), *labels = push(a, n * sizeof(*labels));
    i32 *sheet = push(a, n * sizeof(*sheet)), *piece = push(a, n * sizeof(*piece));
    float *winding = push(a, n * sizeof(*winding)), *depth = push(a, n * sizeof(*depth));
    double *piece_area = pushz(a, (size_t)pieces * sizeof(*piece_area));
    double *labelled = pushz(a, (size_t)pieces * sizeof(*labelled));
    double *purity = pushz(a, (size_t)pieces * sizeof(*purity));
    double *piece_outward = pushz(a, (size_t)pieces * sizeof(*piece_outward));
    i32 *piece_frame = push(a, (size_t)pieces * sizeof(*piece_frame));
    i32 *mode = pushz(a, (size_t)pieces * sizeof(*mode));
    i64 *true_level = pushz(a, (size_t)pieces * sizeof(*true_level));
    u8 *piece_val = pushz(a, pieces);
    float *z = push(a, (size_t)pieces * 2 * sizeof(*z));
    for (int i = 0; i < pieces; i++) {
        piece_frame[i] = -1;
        z[2 * i] = INFINITY;
        z[2 * i + 1] = -INFINITY;
    }
    SynVote *votes = push(a, n * sizeof(*votes));
    size_t used = 0, seen = 0, matched = 0, off = 0, in_pieces = 0, agrees = 0;
    for (size_t i = 0; i < n; i++) {
        SynLabel t = truth[i];
        double u = t.theta - sense * theta[i] / (2 * M_PI) - phase;
        labels[i] = t.sheet < 0 ? 0 : (i32)nearbyint(u);
        judged[i] = t.matched && fabs(u - labels[i]) <= 0.2;
        frame[i] = judged[i] ? 0 : -1;
        pressed[i] = judged[i] && t.pressed;
        outward[i] = t.outward;
        winding[i] = t.winding;
        depth[i] = t.nearest;
        sheet[i] = t.sheet;
        seen += judged[i];
        matched += t.matched;
        off += !t.near && !t.invalid;
        int p = rank[root[i]];
        piece[i] = p;
        if (p < 0) {
            continue;
        }
        in_pieces++;
        agrees += node_level[i] - curl * potential[i] == piece_level[p];
        piece_area[p] += area[i];
        z[2 * p] = fmin(z[2 * p], t.z);
        z[2 * p + 1] = fmax(z[2 * p + 1], t.z);
        if (judged[i]) {
            piece_frame[p] = 0;
            votes[used++] = (SynVote){
                .piece = p, .value = node_level[i] - labels[i], .node = i, .weight = area[i]};
            labelled[p] += area[i];
            piece_outward[p] += area[i] * outward[i];
        }
    }
    syn_votes(votes, used, mode, purity);
    double sum_area = 0, sum_labelled = 0, judged_area = 0, pressed_area = 0;
    int pieces_with_truth = 0;
    used = 0;
    for (int p = 0; p < pieces; p++) {
        if (labelled[p] > 0) {
            purity[p] /= labelled[p];
            piece_outward[p] /= labelled[p];
            sum_area += piece_area[p];
            sum_labelled += labelled[p];
            votes[used++] =
                (SynVote){.piece = 0, .value = mode[p], .node = p, .weight = labelled[p]};
            pieces_with_truth++;
        } else {
            piece_outward[p] = NAN;
        }
    }
    int offset = 0;
    double weight = 0;
    syn_votes(votes, used, &offset, &weight);
    for (int p = 0; p < pieces; p++) {
        if (piece_frame[p] == 0) {
            true_level[p] = piece_level[p] - (mode[p] - offset);
        }
    }
    for (size_t i = 0; i < n; i++) {
        judged_area += area[i] * judged[i];
        pressed_area += area[i] * pressed[i];
    }
    const char *suffix = curl == 1 ? "plus" : "minus";
    Array left = npz_get(a, levels, fmt(a, "p_%s", suffix));
    const i32 *p = left.data, *q = npz_get(a, levels, fmt(a, "q_%s", suffix)).data;
    const i32 *k = npz_get(a, levels, fmt(a, "k_%s", suffix)).data;
    i64 terms = left.shape[0], judged_terms = 0, right = 0;
    u8 *term_judged = push(a, terms);
    i32 *term_k = push(a, terms * sizeof(*term_k));
    for (i64 i = 0; i < terms; i++) {
        term_judged[i] = piece_frame[p[i]] == 0 && piece_frame[q[i]] == 0;
        term_k[i] = term_judged[i] ? true_level[q[i]] - true_level[p[i]] : 0;
        judged_terms += term_judged[i];
        right += term_judged[i] && k[i] == term_k[i];
    }
    const char *names[] = {"node_judged", "node_frame", "node_label", "node_pressed", "node_val",
        "node_area", "node_W", "node_u", "node_sheet", "node_outward", "piece_outward",
        "node_piece", "piece_area", "piece_frame", "piece_offset", "piece_purity", "piece_labelled",
        "piece_true_level", "piece_val", "piece_z", "term_judged", "term_k_true"};
    const char *dtype[] = {"|b1", "<i4", "<i4", "|b1", "|b1", "<f4", "<f4", "<f4", "<i4", "|b1",
        "<f8", "<i4", "<f8", "<i4", "<i4", "<f8", "<f8", "<i8", "|b1", "<f4", "|b1", "<i4"};
    const void *values[] = {judged, frame, labels, pressed, val, area, winding, depth, sheet,
        outward, piece_outward, piece, piece_area, piece_frame, mode, purity, labelled, true_level,
        piece_val, z, term_judged, term_k};
    Array arrays[sizeof(names) / sizeof(*names)] = {0};
    for (int i = 0; i < (int)(sizeof(arrays) / sizeof(*arrays)); i++) {
        Array *v = arrays + i;
        v->data = (void *)values[i];
        v->ndim = i == 19 ? 2 : 1;
        v->shape[0] = i < 10 || i == 11 ? (i64)n : i >= 20 ? terms : pieces;
        v->shape[1] = 2;
        v->size = v->shape[0] * v->ndim * (dtype[i][2] - '0');
        memcpy(v->descr, dtype[i], 4);
    }
    npz_write(fmt(a, "%s/truth.npz", out), sizeof(arrays) / sizeof(*arrays), names, arrays);
    json_object *report = json_object_new_object();
    json_object_object_add(report, "band", json_object_get(jget(band, "name")));
    const char *copy[] = {"family", "look", "digest"};
    for (int i = 0; i < 3; i++) {
        json_object_object_add(report, copy[i], json_object_get(jget(band, copy[i])));
    }
    const char *metrics[] = {"nodes", "sense", "matched_nodes", "judged_nodes", "pieces", "terms",
        "chirality", "pieces_with_truth", "judged_terms"};
    i64 integers[] = {
        n, sense, matched, seen, pieces, terms, curl, pieces_with_truth, judged_terms};
    for (int i = 0; i < (int)(sizeof(metrics) / sizeof(*metrics)); i++) {
        json_object_object_add(report, metrics[i], json_object_new_int64(integers[i]));
    }
    syn_add(report, "phase", nearbyint(phase * 1e4) / 1e4);
    syn_add(report, "judged_area_cm2", nearbyint(judged_area) / 100);
    syn_add(report, "pressed_judged_area_cm2", nearbyint(pressed_area) / 100);
    syn_add(report, "nodes_off_sheet", n ? nearbyint((double)off * 1e4 / n) / 1e4 : 0);
    syn_add(report, "node_level_matches_piece_level",
        in_pieces ? nearbyint((double)agrees * 1e5 / in_pieces) / 1e5 : 0);
    syn_add(report, "labelled_piece_area_share",
        nearbyint(sum_labelled / fmax(sum_area, 1e-9) * 1e4) / 1e4);
    json_object_object_add(report, "judged_terms_right",
        judged_terms ? jnum(nearbyint((double)right * 1e4 / judged_terms) / 1e4) : NULL);
    json_object *offsets = json_object_new_object();
    if (pieces_with_truth) {
        json_object_object_add(offsets, "0", json_object_new_int(offset));
    }
    json_object_object_add(report, "frame_offsets", offsets);
    syn_add(report, "seconds", nearbyint((now() - start) * 10) / 10);
    json_write(fmt(a, "%s/truth.json", out), report);
    return report;
}
