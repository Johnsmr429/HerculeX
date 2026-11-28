#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t Index;

void hx_wrap_node_area(
    Index n, float cell, const float *nx, const float *ny, const float *nz, float *out) {
    for (Index i = 0; i < n; i++) {
        float a = nx[i], b = ny[i], c = nz[i];
        float norm = sqrtf(a * a + b * b + c * c);
        float spread = (fabsf(a) + fabsf(b) + fabsf(c)) / (norm < 1e-6f ? 1e-6f : norm);
        out[i] = cell / (spread < 1 ? 1 : spread);
    }
}

typedef struct {
    uint64_t group;
    int64_t delta;
    Index index;
} Vote;

static int vote_order(const void *left, const void *right) {
    const Vote *a = left, *b = right;
    if (a->group != b->group)
        return a->group < b->group ? -1 : 1;
    return (a->delta > b->delta) - (a->delta < b->delta);
}

static void assign(Vote *v, Index n, uint8_t *out, int yes, int no) {
    if (!n)
        return;
    qsort(v, (size_t)n, sizeof(*v), vote_order);
    for (Index start = 0; start < n;) {
        Index end = start, best = 0;
        int64_t mode = 0;
        while (end < n && v[end].group == v[start].group) {
            Index first = end++;
            while (end < n && v[end].group == v[start].group && v[end].delta == v[first].delta)
                end++;
            if (end - first > best) {
                best = end - first;
                mode = v[first].delta;
            }
        }
        for (Index i = start; i < end; i++)
            out[v[i].index] = v[i].delta == mode ? yes : no;
        start = end;
    }
}

static Vote *votes(Index n) {
    return n >= 0 && (uint64_t)n <= SIZE_MAX / sizeof(Vote)
        ? malloc((size_t)(n ? n : 1) * sizeof(Vote))
        : NULL;
}

static uint64_t group_stride(const int64_t *group, Index n) {
    int64_t largest = INT64_MIN;
    for (Index i = 0; i < n; i++) {
        if (group[i] > largest)
            largest = group[i];
    }
    return (uint64_t)largest + 1;
}

int hx_wrap_modal(Index n, const int64_t *piece, const int64_t *group, const void *levels, int kind,
    uint8_t *mask) {
    Vote *v = votes(n);
    if (!v) {
        return -1;
    }
    const int64_t *offset = levels;
    const double *real = levels;
    uint64_t stride = group_stride(group, n);
    Index used = 0;
    for (Index i = 0; i < n; i++) {
        int64_t key;
        if (kind == 1) {
            if (isnan(real[i])) {
                mask[i] = 0;
                continue;
            }
            double x = real[i] == 0 ? 0 : real[i];
            memcpy(&key, &x, sizeof(key));
            if (key < 0)
                key ^= INT64_MAX;
        } else {
            key = kind == 2 ? offset[i] ^ INT64_MIN : offset[i];
        }
        v[used++] =
            (Vote){.group = (uint64_t)piece[i] * stride + group[i], .delta = key, .index = i};
    }
    assign(v, used, mask, 1, 0);
    free(v);
    return 0;
}

static int64_t integer_at(const void *p, int width, Index i) {
    return width == 4 ? ((const int32_t *)p)[i] : ((const int64_t *)p)[i];
}

int hx_wrap_classify(Index n, Index k, const void *candidates, int cb, Index nodes,
    const void *membership, int pb, const double *angle, const double *reference,
    const int64_t *groups, const uint8_t *valid, Index possible, const uint8_t *unknown, int f32,
    uint8_t *labels, int64_t *out) {
    Vote *v = votes(n);
    if (!v) {
        return -1;
    }
    Index count = 0;
    int bad = 0;
    int64_t largest = INT64_MIN;
    for (Index i = 0; i < n && !bad; i++) {
        out[i] = -1;
        labels[i] = 2;
        for (Index j = 0; j < k; j++) {
            int64_t node = integer_at(candidates, cb, i * k + j);
            if (node < 0)
                continue;
            if (node >= nodes) {
                bad = 1;
                break;
            }
            int64_t p = integer_at(membership, pb, node);
            if (p >= possible) {
                bad = 1;
                break;
            }
            if (p < 0 || !valid[p] || out[i] >= 0)
                continue;
            out[i] = p;
            labels[i] = 3;
            if (!unknown[i]) {
                double diff = reference[i] - angle[node];
                double rounded = nearbyint(f32 ? (double)(float)diff : diff);
                int64_t d = rounded >= -0x1p63 && rounded < 0x1p63 ? (int64_t)rounded : INT64_MIN;
                if (groups[i] > largest)
                    largest = groups[i];
                v[count++] = (Vote){.group = (uint64_t)groups[i], .delta = d, .index = i};
            }
        }
    }
    if (!bad) {
        uint64_t stride = (uint64_t)largest + 1;
        for (Index i = 0; i < count; i++)
            v[i].group += (uint64_t)out[v[i].index] * stride;
        assign(v, count, labels, 0, 1);
    }
    free(v);
    return bad ? -2 : 0;
}

typedef struct {
    int64_t mesh;
    Index index;
} Position;

static int position_order(const void *l, const void *r) {
    const Position *a = l, *b = r;
    if (a->mesh != b->mesh)
        return a->mesh < b->mesh ? -1 : 1;
    return (a->index > b->index) - (a->index < b->index);
}

Index hx_wrap_wrong_blobs(Index n, const int64_t *ms, const int64_t *ys, const int64_t *xs,
    const int8_t *labels, double least) {
    if (!n) {
        return 0;
    }
    Position *order =
        (size_t)n <= SIZE_MAX / sizeof(*order) ? malloc((size_t)n * sizeof(*order)) : NULL;
    int ok = order != NULL, ordered = 1;
    Index total = 0;
    for (Index i = 0; ok && i < n; i++) {
        order[i] = (Position){.mesh = ms[i], .index = i};
        if (i && ms[i] < ms[i - 1])
            ordered = 0;
    }
    if (ok && !ordered)
        qsort(order, (size_t)n, sizeof(*order), position_order);
    for (Index a = 0; ok && a < n;) {
        Index end = a;
        int64_t mx = -1, my = -1;
        int wrong = 0;
        while (end < n && order[end].mesh == order[a].mesh) {
            Index i = order[end++].index;
            wrong |= labels[i] == 1;
            if (ys[i] > my)
                my = ys[i];
            if (xs[i] > mx)
                mx = xs[i];
        }
        if (!wrong) {
            a = end;
            continue;
        }
        if (mx > INT64_MAX - 4 || my > INT64_MAX - 4 ||
            (uint64_t)(my + 4) > SIZE_MAX / (uint64_t)(mx + 4)) {
            ok = 0;
            break;
        }
        Index w = mx + 4, h = my + 4;
        if (h > INT64_MAX / w) {
            ok = 0;
            break;
        }
        uint8_t *grid = calloc((size_t)h * w, 1);
        Index *queue = malloc((size_t)(end - a) * sizeof(*queue));
        if (!grid || !queue)
            ok = 0;
        for (Index j = a; ok && j < end; j++) {
            Index i = order[j].index;
            int64_t y = ys[i] < 0 ? ys[i] + my + 2 : ys[i];
            int64_t x = xs[i] < 0 ? xs[i] + mx + 2 : xs[i];
            if (x < 0 || y < 0) {
                ok = -1;
                break;
            }
            grid[(y + 1) * w + x + 1] = labels[i] == 1;
        }
        Index direction[] = {-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1};
        for (Index start = w; ok == 1 && start < (h - 1) * w; start++) {
            if (!grid[start])
                continue;
            grid[start] = 0;
            queue[0] = start;
            Index tail = 1;
            for (Index head = 0; head < tail; head++) {
                for (int k = 0; k < 8; k++) {
                    Index next = queue[head] + direction[k];
                    if (grid[next]) {
                        grid[next] = 0;
                        queue[tail++] = next;
                    }
                }
            }
            if (tail >= least)
                total += tail;
        }
        free(queue);
        free(grid);
        if (ok < 0)
            break;
        a = end;
    }
    free(order);
    return ok == 1 ? total : ok == 0 ? -1 : -2;
}

#ifdef HX_WRAP_STUDY

typedef struct {
    int64_t count, neighbours, *group, *mesh, *row, *column;
    int32_t *candidates;
    double *winding, vertex_mm2;
    uint8_t *unresolved;
} WrapTruth;

static void close_wrap_truth(WrapTruth *truth) {
    free(truth->column);
    free(truth->row);
    free(truth->mesh);
    free(truth->group);
    free(truth->winding);
    free(truth->unresolved);
    free(truth->candidates);
}

static double *wrap_column(NpzReader *file, const char *name, int64_t *count) {
    double *values = NULL;
    if (!npz_read(file, name, "<f8", 8, (void **)&values, count)) {
        return values;
    }
    float *narrow = NULL;
    if (!npz_read(file, name, "<f4", 4, (void **)&narrow, count)) {
        values = malloc((size_t)(*count ? *count : 1) * sizeof(*values));
        for (int64_t i = 0; values && i < *count; i++) {
            values[i] = narrow[i];
        }
    }
    free(narrow);
    return values;
}

static int wrap_ray_logit(NpzReader *file, const char *prefix, RayLogit *out) {
    const char *fields[] = {"mean", "scale", "weights", "intercept"};
    double *values[4];
    int64_t counts[4] = {0};
    char key[32];
    for (int i = 0; i < 4; i++) {
        snprintf(key, sizeof(key), "%s_%s", prefix, fields[i]);
        values[i] = wrap_column(file, key, &counts[i]);
    }
    uint32_t *names = NULL;
    int64_t codes = 0, n = counts[0];
    snprintf(key, sizeof(key), "%s_names", prefix);
    int ok = values[0] && values[1] && values[2] && values[3] && counts[1] == n && counts[2] == n &&
        counts[3] == 1 && isfinite(values[3][0]) &&
        !npz_read(file, key, "'<U", 4, (void **)&names, &codes) &&
        (n ? codes >= n && codes % n == 0 : codes == 0) &&
        (uint64_t)n <= SIZE_MAX / sizeof(RayTerm);
    RayTerm *terms = ok && n ? malloc((size_t)n * sizeof(*terms)) : NULL;
    out->terms = terms;
    ok = ok && (!n || terms);
    const char *features[RAY_FEATURES] = {[RAY_LENGTH] = "log_length_over_gap",
        [RAY_DEGREE] = "degree_min",
        [RAY_NORMAL] = "normal_dot",
        [RAY_COHERENCE] = "coherence_min",
        [RAY_PROM_A] = "prominence_a",
        [RAY_PROM_B] = "prominence_b",
        [RAY_LATERAL] = "lateral",
        [RAY_WIDTH_A] = "width_a",
        [RAY_WIDTH_B] = "width_b"};
    for (int64_t i = 0; ok && i < n; i++) {
        size_t width = (size_t)(codes / n), len = width;
        const uint32_t *name = names + (size_t)i * width;
        while (len && !name[len - 1]) {
            len--;
        }
        int feature = 0;
        for (; feature < RAY_FEATURES; feature++) {
            const char *candidate = features[feature];
            if (strlen(candidate) == len) {
                size_t j = 0;
                while (j < len && name[j] == (unsigned char)candidate[j]) {
                    j++;
                }
                if (j == len) {
                    break;
                }
            }
        }
        ok = feature < RAY_FEATURES && isfinite(values[0][i]) && isfinite(values[1][i]) &&
            values[1][i] != 0 && isfinite(values[2][i]);
        terms[i] = (RayTerm){.mean = values[0][i],
            .scale = values[1][i],
            .weight = values[2][i],
            .feature = feature};
    }
    if (ok) {
        out->count = n;
        out->intercept = values[3][0];
    }
    free(names);
    for (int i = 0; i < 4; i++) {
        free(values[i]);
    }
    return ok;
}

static int wrap_i64(NpzReader *file, const char *key, int64_t **out, int64_t count) {
    int64_t n = 0;
    if (!npz_read(file, key, "<i8", 8, (void **)out, &n)) {
        return n == count;
    }
    int32_t *narrow = NULL;
    if (npz_read(file, key, "<i4", 4, (void **)&narrow, &n)) {
        return 0;
    }
    int ok = n == count;
    if (ok) {
        *out = malloc((size_t)n * sizeof(**out));
        ok = *out != NULL;
        for (int64_t i = 0; ok && i < n; i++) {
            (*out)[i] = narrow[i];
        }
    }
    free(narrow);
    return ok;
}

static int wrap_truth(const char *path, WrapTruth *truth) {
    NpzReader file = {0};
    if (npz_open_reader(&file, path)) {
        return 0;
    }
    int64_t candidates = 0, masks = 0, scalar = 0;
    double *vertex = NULL;
    truth->winding = wrap_column(&file, "Wg", &truth->count);
    int64_t n = truth->count;
    int ok = truth->winding && n > 0 &&
        !npz_read(&file, "candidates", "<i4", 4, (void **)&truth->candidates, &candidates) &&
        candidates >= n && !(candidates % n) &&
        !npz_read(&file, "unresolved", "|b1", 1, (void **)&truth->unresolved, &masks) &&
        masks == n && !npz_read(&file, "vertex_mm2", "<f8", 8, (void **)&vertex, &scalar) &&
        scalar == 1 && vertex[0] > 0 && wrap_i64(&file, "group", &truth->group, n);
    if (ok) {
        truth->vertex_mm2 = vertex[0];
        truth->neighbours = candidates / n;
    }
    if (ok && zip_name_locate(file.file, "row.npy", 0) >= 0) {
        ok = wrap_i64(&file, "mesh", &truth->mesh, n) && wrap_i64(&file, "row", &truth->row, n) &&
            wrap_i64(&file, "col", &truth->column, n);
    }
    free(vertex);
    npz_close_reader(&file);
    return ok;
}

static double *wrap_areas(
    const Field *field, const Solution *solution, int32_t **piece, int32_t *groups) {
    if (dense_labels(solution->root, field->nodes_count, piece, groups)) {
        return NULL;
    }
    size_t n = (size_t)field->nodes_count;
    float *node_area = malloc(n * sizeof(*node_area));
    double *area = calloc((size_t)*groups, sizeof(*area));
    if (node_area && area) {
        double side = ldexp(HX_CELL * field->scan_voxel_um, field->level) / 1000;
        hx_node_area(field->nodes, field->nodes_count, (float)(side * side), node_area);
        for (size_t i = 0; i < n; i++) {
            area[(*piece)[i]] += node_area[i];
        }
    } else {
        free(area);
        area = NULL;
    }
    free(node_area);
    return area;
}

static json_object *wrap_statistics(
    const WrapTruth *truth, const Field *field, const Solution *solution) {
    int32_t *piece = NULL, groups = 0;
    double *area = wrap_areas(field, solution, &piece, &groups);
    size_t n = (size_t)field->nodes_count;
    double *angle = malloc(n * sizeof(*angle));
    uint8_t *valid = malloc((size_t)groups), *label = malloc((size_t)truth->count);
    int64_t *matched = malloc((size_t)truth->count * sizeof(*matched));
    int ok = angle && area && valid && label && matched;
    json_object *out = NULL;
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            angle[i] = solution->potential[i] + (double)(field->theta[i] / (float)(2 * M_PI));
        }
        out = json_object_new_object();
        double largest = 0;
        for (int32_t i = 0; i < groups; i++) {
            largest = fmax(largest, area[i]);
        }
        json_object_object_add(
            out, "largest_cm2", json_object_new_double(nearbyint(largest / 10) / 10));
    }
    const char *stages[] = {"all", "usable", "window"};
    const int floor_area[] = {0, 100, 400};
    const char *kinds[] = {"correct", "wrong", "missing", "unjudged"};
    for (int k = 0; ok && k < 3; k++) {
        int64_t visible = 0;
        double claimed = 0;
        for (int32_t i = 0; i < groups; i++) {
            valid[i] = area[i] >= floor_area[k];
            if (valid[i]) {
                visible++;
                claimed += area[i];
            }
        }
        int status = hx_wrap_classify(truth->count, truth->neighbours, truth->candidates, 4, n,
            piece, 4, angle, truth->winding, truth->group, valid, groups, truth->unresolved, 0,
            label, matched);
        if (status) {
            ok = 0;
            break;
        }
        int64_t hits[4] = {0};
        for (int64_t i = 0; i < truth->count; i++) {
            hits[label[i]]++;
        }
        double judged = fmax(1., hits[0] + hits[1]);
        json_object *row = json_object_new_object();
        for (int i = 0; i < 4; i++) {
            double share = nearbyint(10000. * hits[i] / truth->count) / 10000.;
            json_object_object_add(row, kinds[i], json_object_new_double(share));
        }
        json_object_object_add(row, "wrong_rate",
            json_object_new_double(nearbyint(10000. * hits[1] / judged) / 10000.));
        json_object_object_add(row, "correct_cm2",
            json_object_new_double(nearbyint(hits[0] * truth->vertex_mm2 / 10.) / 10.));
        json_object_object_add(row, "wrong_cm2",
            json_object_new_double(nearbyint(hits[1] * truth->vertex_mm2 / 10.) / 10.));
        json_object_object_add(
            row, "claimed_cm2", json_object_new_double(nearbyint(claimed / 10.) / 10.));
        json_object_object_add(row, "pieces", json_object_new_int64(visible));
        if (truth->row) {
            Index wrong = hx_wrap_wrong_blobs(truth->count, truth->mesh, truth->row, truth->column,
                (const int8_t *)label, 16. / truth->vertex_mm2);
            ok = wrong >= 0;
            json_object_object_add(row, "blob_wrong_rate",
                json_object_new_double(nearbyint(10000. * wrong / judged) / 10000.));
        }
        json_object_object_add(out, stages[k], row);
    }
    if (!ok) {
        json_object_put(out);
        out = NULL;
    }
    free(matched);
    free(label);
    free(valid);
    free(area);
    free(angle);
    free(piece);
    return out;
}

typedef struct {
    const char *folder;
    char *truth;
    int64_t nodes;
} WrapControl;

static int wrap_control_order(const void *left, const void *right) {
    const WrapControl *a = left, *b = right;
    if (a->nodes != b->nodes) {
        return (a->nodes > b->nodes) - (a->nodes < b->nodes);
    }
    return strcmp(a->folder, b->folder);
}

typedef struct {
    FitSettings fit;
    GuardSettings guard;
    RayModel ray_model;
    int learned;
} WrapTrial;

static json_object *score_wrap_control(const WrapControl *control, const WrapTrial *trial) {
    static pthread_mutex_t axis_lock = PTHREAD_MUTEX_INITIALIZER;
    Field field = {0};
    Solution solution = {0};
    WrapTruth truth = {0};
    json_object *result = NULL;
    const char *name = strrchr(control->folder, '/') + 1;
    json_object *record = read_record(control->folder, "record.json");
    const char *sample = json_object_get_string(json_object_object_get(record, "sample"));
    const char *volume = json_object_get_string(json_object_object_get(record, "volume"));
    int curl = 0;
    double began = level_clock();
    int ready = sample && volume && !read_curl_value(sample, volume, &curl);
    if (ready) {
        pthread_mutex_lock(&axis_lock);
        ready = !hx_umbilicus_keep(sample, volume);
        pthread_mutex_unlock(&axis_lock);
    }
    ready = ready && wrap_truth(control->truth, &truth) &&
        !prepare(&field, control->folder, trial ? &trial->fit : NULL);
    field.guard = trial ? &trial->guard : NULL;
    field.ray_model = trial && trial->learned ? &trial->ray_model : NULL;
    if (ready && !solve_field(&field, &solution, curl) && !guard_solution(&field, &solution)) {
        double spent = level_clock() - began;
        result = wrap_statistics(&truth, &field, &solution);
        if (result) {
            json_object_object_add(result, "control", json_object_new_string(name));
            json_object_object_add(result, "sample", json_object_new_string(sample));
            json_object_object_add(result, "frame_costs", NULL);
            json_object_object_add(
                result, "seconds", json_object_new_double(nearbyint(spent * 10) / 10));
        }
    }
    if (!result) {
        fprintf(stderr, "HerculeX: failed to score winding control %s\n", name);
    }
    json_object_put(record);
    close_wrap_truth(&truth);
    solution_close(&solution);
    field_close(&field);
    return result;
}

static int wrap_trial_setting(WrapTrial *trial, const char *arg) {
    const char *eq = strchr(arg, '=');
    if (!eq) {
        return 0;
    }
    char *end = NULL;
    double value = strtod(eq + 1, &end);
    if (end == eq + 1 || *end) {
        return -1;
    }
    size_t len = (size_t)(eq - arg);
    struct {
        const char *name;
        double *target;
    } settings[] = {{"PIECE_DEPTH", &trial->fit.piece_depth}, {"RAY_RIM", &trial->fit.ray_rim},
        {"FIELD_RAY_RIM", &trial->fit.field_rim}, {"RAY_SURE", &trial->fit.ray_sure},
        {"RAY_SCALE", &trial->fit.ray_scale}, {"NECK", &trial->fit.neck},
        {"NECK_CAP", &trial->fit.neck_cap}, {"CURVE_VOXELS", &trial->fit.curve},
        {"HIDDEN", &trial->fit.hidden}, {"HELD_TO_AXIS", &trial->fit.held},
        {"FLATTEN_TOLERANCE", &trial->fit.tolerance}, {"QUALITY_SPAN", &trial->fit.quality},
        {"PATCH_VOXELS", &trial->fit.patch}, {"DRAW_BYTES", &trial->fit.draw_bytes},
        {"RAY_MODEL_SCALE", &trial->ray_model.scale}, {"FLOOR_MM2", &trial->guard.floor_mm2},
        {"SPREAD", &trial->guard.spread}, {"MIN_PIECE_NODES", &trial->guard.min_nodes},
        {"CELL", &trial->guard.cell}, {"CONTACT_CELLS", &trial->guard.contact_cells},
        {"CONTACT_LAYERS", &trial->guard.contact_layers}};
    for (size_t i = 0; i < sizeof(settings) / sizeof(*settings); i++) {
        const char *name = settings[i].name;
        if (strlen(name) == len && !strncmp(arg, name, len)) {
            double *target = settings[i].target;
            if (!isfinite(value) ||
                ((target == &trial->fit.patch || target == &trial->guard.cell ||
                     target == &trial->guard.min_nodes) &&
                    value <= 0) ||
                (target == &trial->guard.contact_cells && (value < 1 || value > INT32_MAX))) {
                return -1;
            }
            *settings[i].target = value;
            return 1;
        }
    }
    if (len == 4 && !strncmp(arg, "BOOT", len)) {
        if (!isfinite(value) || value < INT32_MIN || value > INT32_MAX) {
            return -1;
        }
        trial->fit.boot = (int)value;
        return 1;
    }
    if (len == 9 && !strncmp(arg, "GUARD_CUT", len)) {
        if (!isfinite(value)) {
            return -1;
        }
        trial->guard.cut = value == 0 ? 0 : value < 2 ? 1 : 2;
        return 1;
    }
    const char *unsupported[] = {"GUARD_ROUNDS", "DRAWS", "THREADS", "UNITS", "COARSE_STEP",
        "JOIN_SHAPE", "SEED", "FLATTEN_STEPS", "FLATTEN_THREADS", "ATOM_PLANES"};
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(*unsupported); i++) {
        if (strlen(unsupported[i]) == len && !strncmp(arg, unsupported[i], len)) {
            return -1;
        }
    }
    if (len == 9 && !strncmp(arg, "RAY_MODEL", len)) {
        if (!isfinite(value)) {
            return -1;
        }
        trial->learned = value != 0;
        return 1;
    }
    return 0;
}

typedef struct {
    const WrapControl *controls;
    json_object **results;
    const WrapTrial *trial;
    size_t next;
    pthread_mutex_t lock;
} WrapStudy;

static void *score_wrap_study(void *context) {
    WrapStudy *study = context;
    for (;;) {
        pthread_mutex_lock(&study->lock);
        if (!study->next) {
            pthread_mutex_unlock(&study->lock);
            return NULL;
        }
        size_t i = --study->next;
        pthread_mutex_unlock(&study->lock);
        study->results[i] = score_wrap_control(study->controls + i, study->trial);
    }
}

int hx_wrap_study_main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        puts("Usage: ./run.sh calibrate.unwrap.sheet_wrap_study [CONTROL ...] [NAME=VALUE ...]");
        return 0;
    }
    WrapTrial trial = {.fit = default_fit, .guard = default_guard, .ray_model = {.scale = 1.}};
    int overridden = 0, names = 0;
    for (int i = 1; i < argc; i++) {
        int setting = wrap_trial_setting(&trial, argv[i]);
        if (setting < 0) {
            fprintf(stderr, "HerculeX: unsupported or invalid winding trial setting %s\n", argv[i]);
            return 2;
        }
        overridden |= setting != 0;
        names += strchr(argv[i], '=') == NULL;
    }
    if (trial.learned) {
        const char *path = getenv("HERCULEX_RAY_MODEL");
        NpzReader file = {0};
        int ok = path && *path && !npz_open_reader(&file, path) &&
            wrap_ray_logit(&file, "same", &trial.ray_model.same) &&
            wrap_ray_logit(&file, "skip", &trial.ray_model.skip);
        npz_close_reader(&file);
        if (!ok) {
            fprintf(stderr, "HerculeX: invalid learned ray model %s\n", path ? path : "(unset)");
            free(trial.ray_model.same.terms);
            free(trial.ray_model.skip.terms);
            return 2;
        }
    }
    char *library = local_root(library_root), *pattern = NULL;
    glob_t folders = {0};
    int listed = library && asprintf(&pattern, "%s/controls/sheets/*", library) >= 0 &&
        !glob(pattern, 0, NULL, &folders);
    WrapControl *controls = listed ? calloc(folders.gl_pathc, sizeof(*controls)) : NULL;
    size_t count = 0;
    for (size_t i = 0; controls && i < folders.gl_pathc; i++) {
        const char *folder = folders.gl_pathv[i], *name = strrchr(folder, '/') + 1;
        json_object *doc = read_record(folder, "record.json");
        const char *sample = json_object_get_string(json_object_object_get(doc, "sample"));
        int selected = !names;
        for (int j = 1; sample && j < argc; j++) {
            selected |= !strcmp(argv[j], name) || !strcmp(argv[j], sample);
        }
        char *truth = NULL;
        const char *suffix[] = {"-tier-a", ""};
        for (int j = 0; doc && selected && j < 2; j++) {
            int opened = asprintf(&truth, "%s/calibration/geometry_wrap/truth/%s%s.npz", library,
                             name, suffix[j]) >= 0 &&
                regular_file(truth);
            if (opened) {
                break;
            }
            free(truth);
            truth = NULL;
        }
        if (truth) {
            int64_t nodes = json_object_get_int64(json_object_object_get(doc, "nodes"));
            controls[count++] = (WrapControl){folder, truth, nodes};
        }
        json_object_put(doc);
    }
    json_object *rows = json_object_new_array_ext(count);
    json_object **results = count ? calloc(count, sizeof(*results)) : NULL;
    const char *setting = getenv("HERCULEX_STUDY_WORKERS");
    char *end = NULL;
    errno = 0;
    long workers = setting ? strtol(setting, &end, 10) : 3;
    int failed = !results || !rows || (setting && (end == setting || *end || errno || workers < 1));
    if (!failed) {
        qsort(controls, count, sizeof(*controls), wrap_control_order);
        size_t extra = (size_t)workers < count ? (size_t)workers - 1 : count - 1;
        pthread_t *threads = extra ? malloc(extra * sizeof(*threads)) : NULL;
        WrapStudy study = {
            controls, results, overridden ? &trial : NULL, count, PTHREAD_MUTEX_INITIALIZER};
        size_t started = 0;
        while (threads && started < extra &&
            !pthread_create(threads + started, NULL, score_wrap_study, &study)) {
            started++;
        }
        score_wrap_study(&study);
        for (size_t i = 0; i < started; i++) {
            pthread_join(threads[i], NULL);
        }
        pthread_mutex_destroy(&study.lock);
        free(threads);
    }
    for (size_t i = 0; i < count; i++) {
        json_object *result = results ? results[i] : NULL;
        if (result) {
            puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PLAIN));
            fflush(stdout);
            json_object_array_add(rows, result);
        } else {
            failed = 1;
        }
        free(controls[i].truth);
    }
    if (!failed && argc == 1) {
        failed = record_write_layout("records/surface/wrap-study-patches.json", rows);
    }
    if (!count) {
        fputs("HerculeX: no winding controls have truth\n", stderr);
    }
    free(results);
    json_object_put(rows);
    free(controls);
    globfree(&folders);
    free(pattern);
    free(library);
    free(trial.ray_model.same.terms);
    free(trial.ray_model.skip.terms);
    return failed != 0;
}

static double published_band_area(
    const char *library, const Field *field, double lower, double upper) {
    glob_t meshes = {0};
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/published/open_data/%s/segments/*/mesh/*-on-%s-*um.tifxyz", library,
            field->sample, field->volume) < 0) {
        return NAN;
    }
    int listed = glob(pattern, 0, NULL, &meshes);
    free(pattern);
    double area = 0;
    for (size_t i = 0; !listed && i < meshes.gl_pathc; i++) {
        const char *folder = meshes.gl_pathv[i];
        if (strstr(folder, "spiral-output")) {
            continue;
        }
        json_object *meta = read_record(folder, "meta.json");
        json_object *scale = json_object_object_get(meta, "scale");
        double spacing = json_object_get_double(json_object_array_get_idx(scale, 0));
        json_object_put(meta);
        if (!(spacing > 0)) {
            area = NAN;
            break;
        }
        TIFF *z = open_axis(folder, 2);
        if (!z) {
            continue;
        }
        uint32_t h = 0, w = 0;
        TIFFGetField(z, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(z, TIFFTAG_IMAGELENGTH, &h);
        TIFFClose(z);
        size_t cells = (size_t)h * w;
        double *depth = cells <= SIZE_MAX / sizeof(*depth) ? malloc(cells * sizeof(*depth)) : NULL;
        if (!depth || !read_axis(folder, w, h, 2, NULL, depth, NULL, NULL, NULL)) {
            free(depth);
            area = NAN;
            break;
        }
        size_t count = 0;
        for (size_t j = 0; j < cells; j++) {
            count += depth[j] >= lower && depth[j] < upper;
        }
        double side = field->scan_voxel_um / spacing / 10000;
        area += count * (side * side);
        free(depth);
    }
    globfree(&meshes);
    return area;
}

static int known_wrap_band(FILE *stream, const char *band, const char *scan) {
    if (fseek(stream, 0, SEEK_SET)) {
        return -1;
    }
    char *line = NULL;
    size_t capacity = 0;
    int found = 0;
    while (!found && getline(&line, &capacity, stream) >= 0) {
        json_object *row = json_tokener_parse(line);
        const char *name = json_object_get_string(json_object_object_get(row, "band"));
        const char *volume = json_object_get_string(json_object_object_get(row, "volume"));
        found = name && volume && !strcmp(name, band) && !strcmp(volume, scan);
        json_object_put(row);
    }
    free(line);
    return ferror(stream) ? -1 : found;
}

static json_object *survey_wrap_band(const char *path, const char *library) {
    Field field = {0};
    Solution solution = {0};
    int curl = 0, ok = 0;
    const char *name = strrchr(path, '/') + 1;
    json_object *record = read_record(path, "record.json"), *result = NULL;
    const char *sample = json_object_get_string(json_object_object_get(record, "sample"));
    const char *volume = json_object_get_string(json_object_object_get(record, "volume"));
    double start = level_clock();
    if (sample && volume && !read_curl_value(sample, volume, &curl) &&
        !hx_umbilicus_keep(sample, volume) && !prepare(&field, path, NULL) &&
        !solve_field(&field, &solution, curl) && !guard_solution(&field, &solution)) {
        int32_t *piece = NULL, groups = 0;
        double *area = wrap_areas(&field, &solution, &piece, &groups);
        free(piece);
        if (area) {
            double total = 0, one = 0, four = 0, max = 0;
            int one_count = 0, four_count = 0;
            for (int32_t i = 0; i < groups; i++) {
                double a = area[i];
                total += a;
                max = fmax(max, a);
                if (a >= 100) {
                    one += a;
                    one_count++;
                }
                if (a >= 400) {
                    four += a;
                    four_count++;
                }
            }
            free(area);
            double scale = ldexp(HX_CELL, field.level);
            double published = published_band_area(
                library, &field, field.plane0 * scale, (field.plane0 + field.grid[0]) * scale);
            ok = isfinite(published);
            if (ok) {
                result = json_object_new_object();
                json_object_object_add(result, "sample", json_object_new_string(sample));
                json_object_object_add(result, "volume", json_object_new_string(volume));
                json_object_object_add(result, "band", json_object_new_string(name));
                json_object_object_add(
                    result, "voxel_um", json_object_new_double(field.scan_voxel_um));
                json_object_object_add(result, "nodes", json_object_new_int64(field.nodes_count));
                json_object_object_add(result, "seconds",
                    json_object_new_double(nearbyint((level_clock() - start) * 10.) / 10));
                json_object_object_add(result, "turned",
                    json_object_new_double(nearbyint(field.turned_share * 10000.) / 10000));
                json_object_object_add(result, "frame_costs", NULL);
                const char *names[] = {
                    "area_cm2", "in_1cm2_pieces", "in_4cm2_pieces", "largest_cm2", "published_cm2"};
                double values[] = {total / 100, one / 100, four / 100, max / 100, published};
                for (int i = 0; i < 5; i++) {
                    json_object_object_add(
                        result, names[i], json_object_new_double(nearbyint(values[i] * 10.) / 10));
                }
                json_object_object_add(result, "pieces_1cm2", json_object_new_int(one_count));
                json_object_object_add(result, "pieces_4cm2", json_object_new_int(four_count));
            }
        }
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: failed to survey winding band %s\n", name);
    }
    json_object_put(record);
    solution_close(&solution);
    field_close(&field);
    return result;
}

int hx_wrap_survey_main(int argc, char **argv) {
    if (argc > 3 || (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")))) {
        puts("Usage: ./run.sh calibrate.unwrap.sheet_wrap_survey [PART [PARTS]]");
        return argc > 3 ? 2 : 0;
    }
    char *end;
    long part = 0, parts = 1;
    if (argc > 1) {
        part = strtol(argv[1], &end, 10);
        if (*end || part < 0) {
            return 2;
        }
    }
    if (argc == 3) {
        parts = strtol(argv[2], &end, 10);
        if (*end || parts <= 0) {
            return 2;
        }
    }
    if (part >= parts) {
        return 2;
    }
    char *outputs = local_root(outputs_root), *library = local_root(library_root), *pattern = NULL;
    glob_t list = {0};
    int ok = outputs && library &&
        asprintf(&pattern, "%s/unwrapping/nodes/*/*/nodes/band-*", outputs) >= 0;
    int listed = ok ? glob(pattern, 0, NULL, &list) : GLOB_ABORTED;
    WrapControl *bands = !listed ? calloc(list.gl_pathc, sizeof(*bands)) : NULL;
    size_t count = 0;
    for (size_t i = 0; bands && i < list.gl_pathc; i++) {
        const char *folder = list.gl_pathv[i];
        json_object *record = read_record(folder, "record.json");
        if (record) {
            bands[count++] = (WrapControl){.folder = folder,
                .nodes = json_object_get_int64(json_object_object_get(record, "nodes"))};
        }
        json_object_put(record);
    }
    ok = listed == GLOB_NOMATCH || bands;
    if (count) {
        qsort(bands, count, sizeof(*bands), wrap_control_order);
    }
    const char *home = getenv("HERCULEX_HOME");
    char *record = path_join(home && *home ? home : ".", "records/surface/wrap-survey.jsonl");
    FILE *out = count && record ? fopen(record, "a+") : NULL;
    ok = ok && (!count || out);
    for (size_t i = (size_t)part; ok && i < count; i += (size_t)parts) {
        if (bands[i].nodes > 32000000) {
            break;
        }
        const char *path = bands[i].folder, *name = strrchr(path, '/') + 1;
        json_object *doc = read_record(path, "record.json");
        const char *scan = json_object_get_string(json_object_object_get(doc, "volume"));
        int held = scan ? known_wrap_band(out, name, scan) : 0;
        ok = held >= 0;
        if (!held && scan) {
            json_object *result = survey_wrap_band(path, library);
            if (!result || flock(fileno(out), LOCK_EX)) {
                ok = 0;
            } else {
                held = known_wrap_band(out, name, scan);
                ok = held >= 0;
                if (!held) {
                    const char *line =
                        json_object_to_json_string_ext(result, JSON_C_TO_STRING_PLAIN);
                    ok = fprintf(out, "%s\n", line) >= 0 && !fflush(out);
                    if (ok) {
                        const char *sample =
                            json_object_get_string(json_object_object_get(result, "sample"));
                        printf("%s %s\n", sample, name);
                        fflush(stdout);
                    }
                }
                flock(fileno(out), LOCK_UN);
            }
            json_object_put(result);
        }
        json_object_put(doc);
    }
    if (out) {
        ok = !fclose(out) && ok;
    }
    free(record);
    free(bands);
    globfree(&list);
    free(pattern);
    free(library);
    free(outputs);
    return !ok;
}

static int controls_band(const char *library, const char *sample, const char *scan, int level,
    int width, int64_t *first, int64_t *vertices, int *mesh_count) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/published/open_data/%s/segments/*/mesh/*-on-%s-*um.tifxyz", library,
            sample, scan) < 0) {
        return 0;
    }
    glob_t list = {0};
    int listed = glob(pattern, 0, NULL, &list);
    free(pattern);
    int status = 1;
    size_t length = 0, capacity = 128;
    int64_t *rows = calloc(capacity, sizeof(*rows));
    for (size_t i = 0; rows && !listed && i < list.gl_pathc; i++) {
        const char *folder = list.gl_pathv[i];
        TIFF *tif = open_axis(folder, 2);
        if (!tif) {
            continue;
        }
        (*mesh_count)++;
        uint32_t w = 0, h = 0;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        TIFFClose(tif);
        size_t cells = (size_t)w * h;
        double *depth = cells <= SIZE_MAX / sizeof(*depth) ? malloc(cells * sizeof(*depth)) : NULL;
        if (!depth || !read_axis(folder, w, h, 2, NULL, depth, NULL, NULL, NULL)) {
            free(depth);
            status = 0;
            break;
        }
        for (size_t j = 0; status && j < cells; j++) {
            if (!(depth[j] > 0.)) {
                continue;
            }
            double at = ldexp(depth[j], -level - 7);
            if (!(at < INT_MAX)) {
                status = 0;
                break;
            }
            size_t row = (size_t)at;
            if (row >= capacity) {
                size_t more = row * 2 + 1;
                int64_t *bigger = realloc(rows, more * sizeof(*rows));
                if (!bigger) {
                    status = 0;
                    break;
                }
                rows = bigger;
                memset(rows + capacity, 0, (more - capacity) * sizeof(*rows));
                capacity = more;
            }
            rows[row]++;
            if (row >= length) {
                length = row + 1;
            }
        }
        free(depth);
    }
    if (status && rows && *mesh_count >= 3 && length) {
        size_t initial = length < (size_t)width ? length : (size_t)width;
        int64_t held = 0;
        for (size_t j = 0; j < initial; j++) {
            held += rows[j];
        }
        *first = 0;
        *vertices = held;
        for (size_t j = initial; j < length; j++) {
            held += rows[j] - rows[j - initial];
            if (held > *vertices) {
                *vertices = held;
                *first = j - initial + 1;
            }
        }
    }
    status = status && rows && (listed == 0 || listed == GLOB_NOMATCH);
    free(rows);
    globfree(&list);
    return status;
}

static int control_name_order(const void *one, const void *two) {
    return strcmp(*(const char *const *)one, *(const char *const *)two);
}

int hx_controls_plan_main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "plan") != 0) {
        fputs("Usage: ./run.sh calibrate.unwrap.sheet_controls plan\n", stderr);
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    char *library = local_root(library_root);
    char *meta = library ? path_join(library, "catalog/metadata.json") : NULL;
    char *out = library ? path_join(library, "controls/sheets") : NULL;
    char *plan = out ? path_join(out, "plan.json") : NULL;
    json_object *catalog = meta ? json_object_from_file(meta) : NULL;
    json_object *samples = json_object_object_get(catalog, "samples");
    json_object *rows = json_object_new_array();
    int ok = plan && json_object_is_type(samples, json_type_object);
    const char *setting = getenv("HERCULEX_UNIT_ROWS");
    int width = setting && *setting ? (int)strtol(setting, NULL, 10) : 12;
    size_t count = ok ? json_object_object_length(samples) : 0;
    const char **names = count ? calloc(count, sizeof(*names)) : NULL;
    if (ok && names) {
        size_t n = 0;
        json_object_object_foreach(samples, sample, owner) {
            (void)owner;
            names[n++] = sample;
        }
        qsort(names, count, sizeof(*names), control_name_order);
    }
    ok = ok && (!count || names) && width >= 2;
    for (size_t n = 0; ok && n < count; n++) {
        const char *sample = names[n];
        json_object *owner = json_object_object_get(samples, sample);
        json_object *volumes = json_object_object_get(owner, "volumes");
        if (!json_object_is_type(volumes, json_type_object)) {
            continue;
        }
        json_object_object_foreach(volumes, scan, detail) {
            json_object *properties = json_object_object_get(detail, "properties");
            double um = json_object_get_double(json_object_object_get(properties, "pixel_size_um"));
            const char *duplicates[] = {
                "20241024131838", "20250820105138", "20260319133554", "20260304131111"};
            int eligible = (um >= 7 && um <= 10) || (um >= 2.0 && um <= 2.6);
            for (int i = 0; eligible && i < 4; i++) {
                eligible = strcmp(scan, duplicates[i]) != 0;
            }
            json_object *products = json_object_object_get(detail, "data");
            int found = 0;
            for (size_t i = 0; eligible && !found && i < json_object_array_length(products); i++) {
                json_object *product = json_object_array_get_idx(products, i);
                const char *type = json_object_get_string(json_object_object_get(product, "type"));
                found = type && !strcmp(type, "ome-zarr");
            }
            if (!eligible || !found) {
                continue;
            }
            int level = (int)fmax(0, nearbyint(log2(18.7 / um)));
            int meshes = 0;
            int64_t first = 0, vertices = 0;
            if (!controls_band(library, sample, scan, level, width, &first, &vertices, &meshes)) {
                ok = 0;
                break;
            }
            if (meshes < 3 || !vertices) {
                continue;
            }
            json_object *row = json_object_new_object(), *range = json_object_new_array_ext(2);
            json_object_array_add(range, json_object_new_int64(first));
            json_object_array_add(range, json_object_new_int64(first + width));
            json_object_object_add(row, "sample", json_object_new_string(sample));
            json_object_object_add(row, "volume", json_object_new_string(scan));
            json_object_object_add(row, "voxel_um", json_object_new_double(um));
            json_object_object_add(row, "level", json_object_new_int(level));
            json_object_object_add(row, "rows", range);
            json_object_object_add(row, "meshes", json_object_new_int(meshes));
            json_object_object_add(row, "vertices", json_object_new_int64(vertices));
            json_object_array_add(rows, row);
        }
    }
    ok = ok && make_folders(out) && !record_write_layout(plan, rows);
    for (size_t i = 0; ok && i < json_object_array_length(rows); i++) {
        puts(json_object_to_json_string_ext(json_object_array_get_idx(rows, i), 0));
    }
    free(names);
    json_object_put(rows);
    json_object_put(catalog);
    free(plan);
    free(out);
    free(meta);
    free(library);
    return !ok;
}

typedef struct {
    float point[3], normal[3];
    int32_t mesh, wrap;
} ControlPoint;

typedef struct {
    ControlPoint *points;
    size_t count, capacity;
    char **names;
    size_t meshes;
} ControlTruth;

static int control_wrap(const char *name) {
    const char *at = name;
    while ((at = strstr(at, "-w"))) {
        at += 2;
        char *end;
        long value = strtol(at, &end, 10);
        if (end != at && *end == '_' && value >= 0 && value <= INT32_MAX) {
            return (int)value;
        }
    }
    return -1;
}

static int control_mesh(ControlTruth *truth, const char *folder, const char *name, double voxel,
    int level, int lower, int upper) {
    json_object *meta = read_record(folder, "meta.json");
    double spacing =
        json_object_get_double(json_object_array_get_idx(json_object_object_get(meta, "scale"), 0));
    json_object_put(meta);
    if (!(spacing > 0 && spacing * 100 / voxel < INT_MAX)) {
        return 0;
    }
    uint32_t w = 0, h = 0;
    int type = 0;
    double *grid = read_raw_mesh(folder, &w, &h, &type);
    if (!grid) {
        return 0;
    }
    int step = (int)fmax(1., nearbyint(100 * spacing / voxel));
    size_t cells = (size_t)w * h, beginning = truth->count;
    int64_t vertical = (int64_t)step * w;
    int wrap = control_wrap(name), ok = 1;
    double scale = ldexp(1., -level);
    for (int64_t r = step; ok && r + step < h; r += step) {
        for (int64_t c = step; c + step < w; c += step) {
            int64_t i = r * w + c;
            float z = (float)grid[2 * cells + i] * (float)scale;
            if (!(z >= lower && z < upper) || grid[i] <= 0 || grid[i - step] <= 0 ||
                grid[i + step] <= 0 || grid[i - vertical] <= 0 || grid[i + vertical] <= 0) {
                continue;
            }
            float u[3], v[3], normal[3];
            for (int k = 0; k < 3; k++) {
                const double *a = grid + k * cells;
                u[k] = (float)a[i + step] - (float)a[i - step];
                v[k] = (float)a[i + vertical] - (float)a[i - vertical];
            }
            normal[0] = u[1] * v[2] - u[2] * v[1];
            normal[1] = u[2] * v[0] - u[0] * v[2];
            normal[2] = u[0] * v[1] - u[1] * v[0];
            float length =
                sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (!(length > 0)) {
                continue;
            }
            if (truth->count == truth->capacity) {
                size_t total = truth->capacity ? truth->capacity * 2 : 65536;
                ControlPoint *larger = realloc(truth->points, total * sizeof(*larger));
                if (!larger) {
                    ok = 0;
                    break;
                }
                truth->points = larger;
                truth->capacity = total;
            }
            ControlPoint *p = truth->points + truth->count++;
            p->mesh = (int32_t)truth->meshes;
            p->wrap = wrap;
            for (int k = 0; k < 3; k++) {
                p->point[k] = (float)grid[k * cells + i] * (float)scale;
                p->normal[k] = normal[k] / length;
            }
        }
    }
    free(grid);
    if (truth->count - beginning < 50) {
        truth->count = beginning;
    } else if (ok) {
        char *copy = strdup(name);
        char **names = copy ? realloc(truth->names, (truth->meshes + 1) * sizeof(*names)) : NULL;
        ok = names != NULL;
        if (ok) {
            truth->names = names;
            truth->names[truth->meshes++] = copy;
        } else {
            free(copy);
        }
    }
    return ok;
}

static int control_truth_gather(ControlTruth *truth, const char *library, const char *sample,
    const char *scan, double voxel, int level, int lower, int upper) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/published/open_data/%s/segments/*/mesh/*-on-%s-*um.tifxyz", library,
            sample, scan) < 0) {
        return 0;
    }
    glob_t candidates = {0};
    int listed = glob(pattern, 0, NULL, &candidates), ok = !listed;
    free(pattern);
    for (size_t i = 0; ok && i < candidates.gl_pathc; i++) {
        const char *folder = candidates.gl_pathv[i];
        char *test = path_join(folder, "z.tif");
        int mesh = test && regular_file(test);
        free(test);
        if (!mesh) {
            continue;
        }
        const char *tail = strstr(folder, "/mesh/");
        if (!tail) {
            continue;
        }
        const char *at = tail;
        while (at > folder && at[-1] != '/') {
            at--;
        }
        char *name = strndup(at, (size_t)(tail - at));
        ok = name && control_mesh(truth, folder, name, voxel, level, lower, upper);
        free(name);
    }
    globfree(&candidates);
    return ok && truth->count;
}

static int control_nearest_xyz(
    const Search *search, const double p[3], size_t *result, int limit, double reach, int row) {
    int ix, iy, iz;
    if (!bin(p[0], search->bound, &ix) || !bin(p[1], search->bound, &iy) ||
        !bin(p[2], search->bound, &iz)) {
        return 0;
    }
    double distance[32], worst = reach * reach;
    int found = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                size_t bucket = bucket_index(search, ix + dx, iy + dy, iz + dz);
                for (size_t j = search->table[bucket].first; j; j = search->cells[j - 1].next) {
                    const Cell *cell = search->cells + j - 1;
                    if (row >= 0 && cell->row != row) {
                        continue;
                    }
                    double x = cell->x - p[0], y = cell->y - p[1], z = cell->z - p[2];
                    double d = x * x + y * y + z * z;
                    if (d > worst || (found < limit && d == worst)) {
                        continue;
                    }
                    int full = found == limit;
                    int at = full ? limit - 1 : found++;
                    if (full && d == distance[at] && j > result[at]) {
                        continue;
                    }
                    while (at &&
                        (d < distance[at - 1] || (d == distance[at - 1] && j < result[at - 1]))) {
                        distance[at] = distance[at - 1];
                        result[at] = result[at - 1];
                        at--;
                    }
                    distance[at] = d;
                    result[at] = j;
                    if (found == limit) {
                        worst = distance[limit - 1];
                    }
                }
            }
        }
    }
    return found;
}

static int control_nearest(
    const Search *search, const float point[3], size_t *result, int limit, double reach) {
    double p[] = {point[0], point[1], point[2]};
    return control_nearest_xyz(search, p, result, limit, reach, -1);
}

static int64_t *control_coincidence(
    const ControlPoint *points, size_t count, int64_t meshes, double tolerance, int64_t *length) {
    Search search = {.count = count, .bound = 4 * tolerance};
    search.cells = calloc(count, sizeof(*search.cells));
    if (!search.cells) {
        return NULL;
    }
    for (size_t i = 0; i < count; i++) {
        const float *p = points[i].point;
        search.cells[i] = (Cell){.x = p[0], .y = p[1], .z = p[2]};
    }
    int ok = index_cells(&search);
    size_t size = 0, capacity = 1024;
    int64_t *keys = ok ? malloc(capacity * sizeof(*keys)) : NULL;
    for (size_t i = 0; keys && i < count; i++) {
        size_t nearby[16] = {0};
        int32_t unique[16];
        int n = control_nearest(&search, points[i].point, nearby, 16, search.bound), used = 0;
        for (int j = 0; j < n; j++) {
            const ControlPoint *other = points + nearby[j] - 1;
            if (other->mesh == points[i].mesh) {
                continue;
            }
            double depth = 0;
            for (int axis = 0; axis < 3; axis++) {
                depth +=
                    ((double)other->point[axis] - points[i].point[axis]) * points[i].normal[axis];
            }
            if (fabs(depth) <= tolerance) {
                int p = 0;
                while (p < used && unique[p] < other->mesh) {
                    p++;
                }
                if (p == used || unique[p] != other->mesh) {
                    memmove(unique + p + 1, unique + p, (used - p) * sizeof(*unique));
                    unique[p] = other->mesh;
                    used++;
                }
            }
        }
        if (size + (size_t)used > capacity) {
            capacity *= 2;
            int64_t *more = realloc(keys, capacity * sizeof(*keys));
            if (!more) {
                free(keys);
                keys = NULL;
                break;
            }
            keys = more;
        }
        for (int j = 0; j < used; j++) {
            keys[size++] = (int64_t)i * meshes + unique[j];
        }
    }
    *length = (int64_t)size;
    free(search.table);
    free(search.cells);
    return keys;
}

static int calibration_names(Npz *archive, char *const *names, size_t count) {
    size_t stride = 0, width = 0;
    for (size_t i = 0; i < count; i++) {
        size_t bytes = strlen(names[i]);
        stride = bytes > stride ? bytes : stride;
    }
    uint32_t *values = calloc(stride * count, sizeof(*values));
    if (!values) {
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        const unsigned char *p = (const unsigned char *)names[i];
        size_t at = 0;
        while (*p) {
            unsigned lead = *p++;
            uint32_t code = lead;
            if (lead >= 128) {
                size_t tail = lead >= 0xf0 && lead <= 0xf4 ? 3
                    : lead >= 0xe0 && lead <= 0xef         ? 2
                    : lead >= 0xc2 && lead <= 0xdf         ? 1
                                                           : 0;
                code = lead & (0x7f >> (tail + 1));
                size_t read = 0;
                while (read < tail && (p[read] & 0xc0) == 0x80) {
                    code = (code << 6) | (p[read++] & 0x3f);
                }
                uint32_t minimum[] = {0, 0x80, 0x800, 0x10000};
                if (tail && read == tail && code >= minimum[tail] && code <= 0x10ffff &&
                    !(code >= 0xd800 && code <= 0xdfff)) {
                    p += tail;
                } else {
                    code = 0xdc00 + lead;
                }
            }
            values[i * stride + at++] = code;
        }
        width = at > width ? at : width;
    }
    for (size_t i = 0; width != stride && i < count; i++) {
        memmove(values + i * width, values + i * stride, width * sizeof(*values));
    }
    char descr[32];
    snprintf(descr, sizeof(descr), "<U%zu", width);
    int64_t shape[] = {count};
    int failed =
        npz_add(archive, "names", descr, shape, 1, values, width * count * sizeof(*values));
    free(values);
    return failed;
}

static int control_truth_save(const char *path, ControlTruth *truth, double tolerance) {
    size_t stride = (truth->count + 4000000 - 1) / 4000000;
    size_t n = (truth->count + stride - 1) / stride;
    for (size_t i = 0; i < n; i++) {
        truth->points[i] = truth->points[i * stride];
    }
    int64_t count = 0;
    int64_t *keys =
        control_coincidence(truth->points, n, (int64_t)truth->meshes, tolerance, &count);
    float *grid = malloc(n * 6 * sizeof(*grid));
    int32_t *part = malloc(n * 2 * sizeof(*part));
    int ok = keys && grid && part;
    for (size_t i = 0; ok && i < n; i++) {
        memcpy(grid + 3 * i, truth->points[i].point, 3 * sizeof(*grid));
        memcpy(grid + 3 * (n + i), truth->points[i].normal, 3 * sizeof(*grid));
        part[i] = truth->points[i].mesh;
        part[n + i] = truth->points[i].wrap;
    }
    char *part_path = NULL;
    int code = 0;
    Npz archive = {0};
    if (ok && asprintf(&part_path, "%s.part.%ld", path, (long)getpid()) >= 0) {
        archive.file = zip_open(part_path, ZIP_CREATE | ZIP_TRUNCATE, &code);
    }
    int64_t points[] = {n, 3}, vertices[] = {n}, coincidences[] = {count};
    int failed = !archive.file ||
        npz_add(&archive, "points", "<f4", points, 2, grid, 3 * n * sizeof(*grid)) ||
        npz_add(&archive, "normals", "<f4", points, 2, grid + 3 * n, 3 * n * sizeof(*grid)) ||
        npz_add(&archive, "mesh", "<i4", vertices, 1, part, n * sizeof(*part)) ||
        npz_add(&archive, "wrap", "<i4", vertices, 1, part + n, n * sizeof(*part)) ||
        calibration_names(&archive, truth->names, truth->meshes) ||
        npz_add(&archive, "coincide", "<i8", coincidences, 1, keys, count * sizeof(*keys));
    failed = npz_finish(&archive, part_path, failed);
    if (!failed) {
        failed = rename(part_path, path);
    }
    if (failed && part_path) {
        unlink(part_path);
    }
    free(part_path);
    free(part);
    free(grid);
    free(keys);
    return !failed;
}

static void control_truth_close(ControlTruth *truth) {
    for (size_t i = 0; i < truth->meshes; i++) {
        free(truth->names[i]);
    }
    free(truth->names);
    free(truth->points);
}

static int control_process(char *const argv[]) {
    pid_t pid = fork();
    if (!pid) {
        execv(argv[0], argv);
        _exit(127);
    }
    if (pid < 0) {
        return 0;
    }
    int status = 0;
    pid_t ended;
    do {
        ended = waitpid(pid, &status, 0);
    } while (ended < 0 && errno == EINTR);
    return ended == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static int build_control_nodes(const char *library, json_object *catalog, json_object *row,
    const char *target, const char *name, const char *sample, const char *scan, int level,
    int first, int last, double voxel) {
    json_object *volumes =
        json_object_object_get(json_object_object_get(catalog, sample), "volumes");
    json_object *products = json_object_object_get(json_object_object_get(volumes, scan), "data");
    json_object *product = NULL;
    for (size_t i = 0; !product && i < json_object_array_length(products); i++) {
        json_object *d = json_object_array_get_idx(products, i);
        const char *type = json_object_get_string(json_object_object_get(d, "type"));
        if (type && !strcmp(type, "ome-zarr")) {
            product = d;
        }
    }
    json_object *origin = json_object_array_get_idx(json_object_object_get(product, "origins"), 0);
    json_object *root =
        json_object_array_get_idx(json_object_object_get(origin, "access_roots"), 0);
    const char *path = json_object_get_string(json_object_object_get(origin, "path"));
    const char *host = json_object_get_string(json_object_object_get(root, "url"));
    char *url = NULL, *work = NULL;
    int ok = path && host;
    if (ok) {
        if (!strncmp(host, "s3://", 5)) {
            ok = asprintf(&url, "https://%s.s3.us-east-1.amazonaws.com", host + 5) >= 0;
        } else {
            url = strdup(host);
            ok = url != NULL;
        }
    }
    ok = ok && asprintf(&work, "%s/calibration/sheet_controls/%s", library, name) >= 0;
    char level_text[20], first_text[24], last_text[24];
    snprintf(level_text, sizeof(level_text), "%d", level);
    snprintf(first_text, sizeof(first_text), "%d", first);
    snprintf(last_text, sizeof(last_text), "%d", last);
    char *connections = getenv("HERCULEX_FETCH_CONNECTIONS");
    char *fetch[] = {"./run.sh", "scans.transfer.fetch_volume", (char *)path, level_text, work,
        "--root", url, "--rows", first_text, last_text, "--connections",
        connections && *connections ? connections : "16", NULL};
    if (ok) {
        ok = control_process(fetch);
    }
    if (ok) {
        json_object *meta = read_record(work, "meta.json");
        json_object *dims = json_object_object_get(meta, "shape");
        int height = json_object_get_int(json_object_array_get_idx(dims, 0));
        int start = first * (128 / HX_CELL) + 2;
        int stop = last * (128 / HX_CELL) - 2, top = (height + HX_CELL - 1) / HX_CELL - 2;
        if (stop > top) {
            stop = top;
        }
        json_object_put(meta);
        char low[24], high[24];
        snprintf(low, sizeof(low), "%d", start);
        snprintf(high, sizeof(high), "%d", stop);
        char *detect[] = {"./run.sh", "level.detection.detect_nodes", "store", work, (char *)target,
            low, high, NULL};
        ok = start < stop && control_process(detect);
    }
    if (ok) {
        char *record = path_join(target, "record.json");
        json_object *doc = record ? json_object_from_file(record) : NULL;
        if (doc) {
            json_object_object_add(doc, "sample", json_object_new_string(sample));
            json_object_object_add(doc, "volume", json_object_new_string(scan));
            json_object_object_add(doc, "scan_voxel_um", json_object_new_double(voxel));
            json_object_object_add(doc, "voxel_um", json_object_new_double(ldexp(voxel, level)));
            json_object_object_add(doc, "level", json_object_new_int(level));
            json_object_object_add(
                doc, "rows", json_object_get(json_object_object_get(row, "rows")));
            ok = !record_write_layout(record, doc);
        } else {
            ok = 0;
        }
        json_object_put(doc);
        free(record);
        if (ok) {
            remove_path(work);
        }
    }
    free(work);
    free(url);
    return ok;
}

static int build_control(
    const char *library, const char *root, json_object *catalog, json_object *row) {
    const char *sample = json_object_get_string(json_object_object_get(row, "sample"));
    const char *scan = json_object_get_string(json_object_object_get(row, "volume"));
    json_object *rows = json_object_object_get(row, "rows");
    int first = json_object_get_int(json_object_array_get_idx(rows, 0));
    int last = json_object_get_int(json_object_array_get_idx(rows, 1));
    int level = json_object_get_int(json_object_object_get(row, "level"));
    double voxel = json_object_get_double(json_object_object_get(row, "voxel_um"));
    char *name = NULL;
    int named = sample && scan && first >= 0 && last > first && level >= 0 && level < 30 &&
        voxel > 0 && asprintf(&name, "%s-%s-rows-%04d-%04d", sample, scan, first, last) >= 0;
    char *target = named ? path_join(root, name) : NULL;
    char *truth_path = target ? path_join(target, "truth.npz") : NULL;
    int ok = truth_path != NULL;
    if (ok && !regular_file(truth_path)) {
        json_object *record = read_record(target, "record.json");
        if (record && !json_object_object_get(record, "sample")) {
            json_object_put(record);
            record = NULL;
        }
        if (!record) {
            ok = build_control_nodes(
                library, catalog, row, target, name, sample, scan, level, first, last, voxel);
            record = ok ? read_record(target, "record.json") : NULL;
        }
        if (record) {
            json_object *planes = json_object_object_get(record, "cell_planes");
            int low = json_object_get_int(json_object_array_get_idx(planes, 0));
            int high = json_object_get_int(json_object_array_get_idx(planes, 1));
            ControlTruth truth = {0};
            ok = high > low &&
                control_truth_gather(
                    &truth, library, sample, scan, voxel, level, low * HX_CELL, high * HX_CELL) &&
                control_truth_save(truth_path, &truth, 40. / ldexp(voxel, level));
            if (ok) {
                printf("%s %zu meshes %zu vertices\n", name, truth.meshes, truth.count);
                fflush(stdout);
            }
            control_truth_close(&truth);
        } else {
            ok = 0;
        }
        json_object_put(record);
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: could not build sheet control %s\n", name ? name : "");
    }
    free(truth_path);
    free(target);
    free(name);
    return ok;
}

static int64_t control_name_count(zip_t *archive) {
    zip_file_t *file = zip_fopen(archive, "names.npy", 0);
    if (!file) {
        return 0;
    }
    char prefix[256] = {0};
    zip_int64_t n = zip_fread(file, prefix, sizeof(prefix) - 1);
    zip_fclose(file);
    if (n < 12 || memcmp(prefix, "\x93NUMPY", 6) != 0) {
        return 0;
    }
    const char *shape = strstr(prefix + 10, "'shape':");
    const char *opening = shape ? strchr(shape, '(') : NULL;
    return opening ? strtoll(opening + 1, NULL, 10) : 0;
}

static int update_control_truth(const char *path, const char *folder) {
    NpzReader file = {0};
    if (npz_open_reader(&file, path)) {
        return 0;
    }
    if (zip_name_locate(file.file, "coincide.npy", 0) >= 0) {
        npz_close_reader(&file);
        return 1;
    }
    float *xyz = NULL, *normals = NULL;
    int64_t axes = 0, direction = 0, length = 0;
    int64_t meshes = control_name_count(file.file);
    int64_t *mesh = NULL, *keys = NULL;
    ControlPoint *points = NULL;
    int ok = meshes > 0 && !npz_read(&file, "points", "<f4", 4, (void **)&xyz, &axes) &&
        axes >= 3 && !(axes % 3) &&
        !npz_read(&file, "normals", "<f4", 4, (void **)&normals, &direction) && direction == axes &&
        wrap_i64(&file, "mesh", &mesh, axes / 3);
    npz_close_reader(&file);
    json_object *record = read_record(folder, "record.json");
    double voxel = json_object_get_double(json_object_object_get(record, "voxel_um"));
    json_object_put(record);
    if (ok && voxel > 0) {
        points = malloc((size_t)(axes / 3) * sizeof(*points));
    }
    for (int64_t i = 0; points && i < axes / 3; i++) {
        memcpy(points[i].point, xyz + 3 * i, 3 * sizeof(float));
        memcpy(points[i].normal, normals + 3 * i, 3 * sizeof(float));
        points[i].mesh = (int32_t)mesh[i];
    }
    keys = points ? control_coincidence(points, (size_t)(axes / 3), meshes, 40. / voxel, &length)
                  : NULL;
    free(points);
    free(mesh);
    free(normals);
    free(xyz);
    Npz archive = {0};
    int error = 0;
    archive.file = keys ? zip_open(path, 0, &error) : NULL;
    if (archive.file) {
        zip_int64_t sheet = zip_name_locate(archive.file, "sheet.npy", 0);
        int64_t shape[] = {length};
        int failed = (sheet >= 0 && zip_delete(archive.file, sheet)) ||
            npz_add(&archive, "coincide", "<i8", shape, 1, keys, length * sizeof(*keys));
        ok = !npz_finish(&archive, NULL, failed);
        if (ok) {
            const char *name = strrchr(folder, '/') + 1;
            printf("%s %lld meshes, %lld coincidence keys\n", name, (long long)meshes,
                (long long)length);
        }
    } else {
        ok = 0;
    }
    free(keys);
    return ok;
}

static int truth_controls(const char *root) {
    char *pattern = path_join(root, "*/truth.npz");
    glob_t files = {0};
    int listed = pattern ? glob(pattern, 0, NULL, &files) : GLOB_ABORTED;
    int ok = !listed || listed == GLOB_NOMATCH;
    for (size_t i = 0; ok && i < files.gl_pathc; i++) {
        const char *path = files.gl_pathv[i], *ending = strrchr(path, '/');
        char *folder = strndup(path, (size_t)(ending - path));
        ok = folder && update_control_truth(path, folder);
        free(folder);
    }
    globfree(&files);
    free(pattern);
    return ok;
}

int hx_controls_main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "plan")) {
        return hx_controls_plan_main(argc, argv);
    }
    int build = argc >= 2 && !strcmp(argv[1], "build");
    int truth = argc == 2 && !strcmp(argv[1], "truth");
    int help = argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"));
    if (!build && !truth) {
        puts("Usage: ./run.sh calibrate.unwrap.sheet_controls plan|build [SCROLL ...]|truth");
        return help ? 0 : 2;
    }
    char *library = local_root(library_root);
    char *root = library ? path_join(library, "controls/sheets") : NULL;
    int ok = root != NULL;
    if (ok && truth) {
        ok = truth_controls(root);
    } else if (ok) {
        json_object *plan = read_record(root, "plan.json");
        json_object *catalog = read_record(library, "catalog/metadata.json");
        json_object *samples = json_object_object_get(catalog, "samples");
        ok = json_object_is_type(plan, json_type_array) &&
            json_object_is_type(samples, json_type_object);
        for (size_t i = 0; ok && i < json_object_array_length(plan); i++) {
            json_object *row = json_object_array_get_idx(plan, i);
            const char *sample = json_object_get_string(json_object_object_get(row, "sample"));
            int selected = argc == 2;
            for (int j = 2; sample && j < argc; j++) {
                selected |= !strcmp(argv[j], sample);
            }
            if (selected) {
                ok = build_control(library, root, samples, row);
            }
        }
        json_object_put(catalog);
        json_object_put(plan);
    }
    free(root);
    free(library);
    return !ok;
}

typedef struct {
    float xyz[3], normal[3];
    double winding;
    int32_t mesh, unit, row, column;
} RefVertex;

typedef struct {
    RefVertex *points;
    size_t count, capacity, meshes, units;
    char **names;
    int32_t *unit_mesh;
} RefGeometry;

void hx_reference_normals(const double *x, const double *y, const double *z, const uint8_t *valid,
    int64_t h, int64_t w, float *out, uint8_t *keep);
int hx_reference_unwrap(
    const double *theta, const uint8_t *mask, int64_t h, int64_t w, double *angle, int64_t *part);
int hx_fit_centre(const double *x, const double *y, const double *ux, const double *uy,
    const double *weight, size_t size, int robust, double centre[2]);

static int reference_axis(Field *field) {
    double low = INFINITY, high = -INFINITY;
    for (int64_t i = 0; i < field->nodes_count; i++) {
        float z = field->nodes[i].z;
        low = fmin(low, z);
        high = fmax(high, z);
    }
    size_t count = field->nodes_count, cap = (size_t)ceil((high + 1 - low) / 256);
    double *buf = malloc(count * 5 * sizeof(*buf));
    field->axis = calloc(cap, sizeof(*field->axis));
    if (!buf || !field->axis) {
        free(buf);
        return 0;
    }
    for (size_t k = 0; k < cap; k++) {
        double start = low + k * 256, stop = start + 256;
        size_t n = 0;
        for (size_t i = 0; i < count; i++) {
            const Node *p = field->nodes + i;
            if (p->z < start || p->z >= stop || p->prominence < 16) {
                continue;
            }
            float x = p->nx / 127.f, y = p->ny / 127.f, length = hypotf(x, y);
            if (length > .7f) {
                buf[n] = p->x;
                buf[count + n] = p->y;
                buf[2 * count + n] = x / length;
                buf[3 * count + n] = y / length;
                buf[4 * count + n++] = p->prominence;
            }
        }
        if (n >= 500) {
            double xy[2];
            if (!hx_fit_centre(buf, buf + count, buf + 2 * count, buf + 3 * count, buf + 4 * count,
                    n, 1, xy)) {
                free(buf);
                return 0;
            }
            field->axis[field->axis_count++] = (HxAxisKnot){start + 128, xy[0], xy[1]};
        }
    }
    free(buf);
    return field->axis_count > 0;
}

static int reference_valid(const double *grid, size_t n, size_t i) {
    double x = grid[i], y = grid[n + i], z = grid[2 * n + i];
    return x > 0 && y > 0 && z > 0 && isfinite(x) && isfinite(y) && isfinite(z);
}

static int reference_mesh(RefGeometry *ref, const Field *field, const char *folder,
    const char *name, double lower, double upper) {
    json_object *meta = read_record(folder, "meta.json");
    double spacing =
        json_object_get_double(json_object_array_get_idx(json_object_object_get(meta, "scale"), 0));
    json_object_put(meta);
    double voxel = field->scan_voxel_um, step_real = 150 * spacing / voxel;
    if (!(step_real > 0 && step_real < INT_MAX)) {
        return 0;
    }
    uint32_t w = 0, h = 0;
    int type = 0;
    double *grid = read_raw_mesh(folder, &w, &h, &type);
    if (!grid) {
        return 0;
    }
    int step = (int)fmax(1., nearbyint(step_real));
    int r0 = h, r1 = 0;
    size_t full = (size_t)w * h;
    for (size_t i = 0; i < full; i++) {
        double z = grid[2 * full + i];
        if (z >= lower && z < upper && reference_valid(grid, full, i)) {
            int r = (int)(i / w);
            r0 = r0 < r ? r0 : r;
            r1 = r;
        }
    }
    if (r1 < r0) {
        free(grid);
        return 1;
    }
    int rows = (r1 - r0) / step + 1, cols = (w - 1) / step + 1;
    size_t size = (size_t)rows * cols, start = ref->count;
    double level = ldexp(1., field->level), offset = .5 * (level - 1);
    double *planes = calloc(size * 5, sizeof(*planes));
    float *normal = calloc(size * 3, sizeof(*normal));
    uint8_t *valid = calloc(2 * size, 1);
    int64_t *part = malloc(size * sizeof(*part));
    int ok = planes && normal && valid && part;
    if (ok) {
        for (size_t i = 0; i < size; i++) {
            size_t src = ((i / cols) * step + r0) * w + i % cols * step;
            valid[i] = reference_valid(grid, full, src);
            for (int k = 0; k < 3; k++) {
                planes[k * size + i] = grid[k * full + src];
            }
            double cx, cy, z = (planes[2 * size + i] - offset) / level;
            hx_centre_at(field->axis, field->axis_count, z, &cx, &cy);
            planes[3 * size + i] =
                atan2(planes[size + i] - (cy * level + offset), planes[i] - (cx * level + offset));
        }
        hx_reference_normals(
            planes, planes + size, planes + 2 * size, valid, rows, cols, normal, valid + size);
        ok = hx_reference_unwrap(
                 planes + 3 * size, valid + size, rows, cols, planes + 4 * size, part) > 0;
    }
    free(grid);
    double margin = 500 / voxel;
    int64_t last_part = -1;
    for (size_t i = 0; ok && i < size; i++) {
        double z = planes[2 * size + i];
        if (!valid[size + i] || z < lower + margin || z >= upper - margin) {
            continue;
        }
        if (ref->count == ref->capacity) {
            size_t cap = ref->capacity ? ref->capacity * 2 : 65536;
            RefVertex *larger = realloc(ref->points, cap * sizeof(*larger));
            if (!larger) {
                ok = 0;
                break;
            }
            ref->points = larger;
            ref->capacity = cap;
        }
        RefVertex *p = ref->points + ref->count++;
        *p = (RefVertex){.xyz = {planes[i], planes[size + i], z},
            .normal = {normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]},
            .winding = planes[4 * size + i] / (2 * M_PI),
            .mesh = ref->meshes,
            .unit = part[i],
            .row = i / cols,
            .column = i % cols};
        if (part[i] > last_part) {
            last_part = part[i];
        }
    }
    free(part);
    free(valid);
    free(normal);
    free(planes);
    if (ref->count - start < 50) {
        ref->count = start;
        return ok;
    }
    int32_t *rank = calloc((size_t)last_part + 1, sizeof(*rank));
    if (ok && rank) {
        for (size_t i = start; i < ref->count; i++) {
            rank[ref->points[i].unit] = 1;
        }
        size_t units = ref->units;
        for (int64_t i = 0; i <= last_part; i++) {
            if (rank[i]) {
                rank[i] = (int32_t)units++;
            }
        }
        char **names = realloc(ref->names, (ref->meshes + 1) * sizeof(*names));
        if (names) {
            ref->names = names;
        }
        int32_t *map = realloc(ref->unit_mesh, units * sizeof(*map));
        if (map) {
            ref->unit_mesh = map;
        }
        char *copy = strdup(name);
        ok = names && map && copy;
        if (ok) {
            for (size_t i = ref->units; i < units; i++) {
                map[i] = ref->meshes;
            }
            for (size_t i = start; i < ref->count; i++) {
                ref->points[i].unit = rank[ref->points[i].unit];
            }
            ref->units = units;
            ref->names[ref->meshes++] = copy;
        } else {
            free(copy);
        }
    } else {
        ok = 0;
    }
    free(rank);
    return ok;
}

typedef struct {
    int32_t a, b, step;
} RefEdge;

typedef struct {
    int32_t unit;
    size_t size;
} RefRank;

static int reference_rank(const void *a, const void *b) {
    const RefRank *x = a, *y = b;
    if (x->size != y->size) {
        return x->size > y->size ? -1 : 1;
    }
    return (x->unit > y->unit) - (x->unit < y->unit);
}

static int reference_edge_order(const void *a, const void *b) {
    const RefEdge *x = a, *y = b;
    if (x->a != y->a) {
        return (x->a > y->a) - (x->a < y->a);
    }
    if (x->b != y->b) {
        return (x->b > y->b) - (x->b < y->b);
    }
    return (x->step > y->step) - (x->step < y->step);
}

static int reference_windings(const char *name, long value[2]) {
    const char *at = name;
    while ((at = strstr(at, "-w"))) {
        at += 2;
        char *end;
        value[0] = strtol(at, &end, 10);
        if (end != at && *end == '-') {
            const char *after = end + 1;
            value[1] = strtol(after, &end, 10);
            if (end != after && value[0] >= 0 && value[1] >= 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int reference_conflict(const RefGeometry *ref, int32_t a, int32_t b) {
    long x[2], y[2];
    return reference_windings(ref->names[ref->unit_mesh[a]], x) &&
        reference_windings(ref->names[ref->unit_mesh[b]], y) && (x[0] > y[1] || y[0] > x[1]);
}

static size_t reference_links(
    RefEdge *edges, size_t count, const RefGeometry *ref, const size_t *sizes) {
    qsort(edges, count, sizeof(*edges), reference_edge_order);
    size_t links = 0;
    for (size_t i = 0; i < count;) {
        RefEdge chosen = edges[i];
        size_t end = i, votes = 0;
        while (end < count && edges[end].a == chosen.a && edges[end].b == chosen.b) {
            size_t next = end + 1;
            while (next < count && edges[next].a == chosen.a && edges[next].b == chosen.b &&
                edges[next].step == edges[end].step) {
                next++;
            }
            if (next - end > votes) {
                chosen.step = edges[end].step;
                votes = next - end;
            }
            end = next;
        }
        size_t smaller = sizes[chosen.a] < sizes[chosen.b] ? sizes[chosen.a] : sizes[chosen.b];
        size_t total = end - i;
        if (total >= 1000 && total >= .2 * smaller && votes / (double)total >= .95 &&
            !reference_conflict(ref, chosen.a, chosen.b)) {
            edges[links++] = chosen;
        }
        i = end;
    }
    return links;
}

static double reference_depth(const RefVertex *p, const Cell *q, double lateral2) {
    double x = q->x - p->xyz[0], y = q->y - p->xyz[1], z = q->z - p->xyz[2];
    double depth = fabs(x * p->normal[0] + y * p->normal[1] + z * p->normal[2]);
    return fmax(0., x * x + y * y + z * z - depth * depth) <= lateral2 ? depth : INFINITY;
}

static int reference_frame(RefGeometry *ref, Search *search, double tolerance, double lateral,
    uint8_t *dropped, int32_t *group, uint8_t *unresolved) {
    size_t units = ref->units, count = ref->count;
    size_t *sizes = calloc(units, sizeof(*sizes));
    size_t *position = calloc(units, sizeof(*position));
    RefRank *rank = calloc(units, sizeof(*rank));
    int64_t *offset = calloc(units, sizeof(*offset));
    int32_t *queue = calloc(units, sizeof(*queue));
    size_t cap = 4096, length = 0;
    RefEdge *edges = malloc(cap * sizeof(*edges));
    int ok = sizes && position && rank && offset && queue && edges;
    for (size_t i = 0; ok && i < count; i++) {
        sizes[ref->points[i].unit]++;
    }
    for (size_t i = 0; ok && i < units; i++) {
        rank[i] = (RefRank){i, sizes[i]};
        group[i] = -1;
    }
    if (ok) {
        qsort(rank, units, sizeof(*rank), reference_rank);
        for (size_t i = 0; i < units; i++) {
            position[rank[i].unit] = i;
        }
    }
    double lateral2 = lateral * lateral;
    for (size_t i = 0; ok && i < count; i++) {
        RefVertex *p = ref->points + i;
        size_t near[12];
        int n = control_nearest(search, p->xyz, near, 12, hypot(tolerance, lateral));
        for (int j = 0; j < n; j++) {
            size_t ix = near[j] - 1;
            RefVertex *q = ref->points + ix;
            if (p->unit == q->unit ||
                reference_depth(p, search->cells + ix, lateral2) > tolerance) {
                continue;
            }
            if (position[q->unit] < position[p->unit]) {
                dropped[i] = 1;
            }
            if (length == cap) {
                cap *= 2;
                RefEdge *next = realloc(edges, cap * sizeof(*edges));
                if (!next) {
                    ok = 0;
                    break;
                }
                edges = next;
            }
            edges[length++] = (RefEdge){p->unit, q->unit, nearbyint(p->winding - q->winding)};
        }
    }
    size_t links = ok ? reference_links(edges, length, ref, sizes) : 0;
    size_t at = 0;
    for (size_t a = 0; ok && a < units; a++) {
        position[a] = at;
        while (at < links && (size_t)edges[at].a == a) {
            at++;
        }
    }
    int groups = 0;
    for (size_t i = 0; ok && i < units; i++) {
        int32_t start = rank[i].unit;
        if (group[start] >= 0) {
            continue;
        }
        size_t head = 0, tail = 0;
        group[start] = groups++;
        queue[tail++] = start;
        while (head < tail) {
            int32_t a = queue[head++];
            for (size_t j = position[a]; j < links && edges[j].a == a; j++) {
                int32_t b = edges[j].b;
                if (group[b] < 0) {
                    group[b] = group[a];
                    offset[b] = offset[a] + edges[j].step;
                    queue[tail++] = b;
                }
            }
        }
    }
    for (size_t i = 0; ok && i < count; i++) {
        ref->points[i].winding += offset[ref->points[i].unit];
    }
    double reach = hypot(2 * tolerance, lateral);
    for (size_t i = 0; ok && i < count; i++) {
        const RefVertex *p = ref->points + i;
        size_t near[16];
        int n = control_nearest(search, p->xyz, near, 16, reach);
        for (int j = 0; j < n; j++) {
            size_t other = near[j] - 1;
            const RefVertex *q = ref->points + other;
            double depth = reference_depth(p, search->cells + other, lateral2);
            if (depth <= 2 * tolerance &&
                (group[p->unit] == group[q->unit] ? fabs(p->winding - q->winding) > .5
                                                  : depth > tolerance)) {
                unresolved[i] = 1;
                break;
            }
        }
    }
    free(edges);
    free(queue);
    free(offset);
    free(rank);
    free(position);
    free(sizes);
    return ok ? groups : -1;
}

static uint16_t reference_half(float value) {
    if (!isfinite(value) || value >= 65520.f) {
        return 0x7c00;
    }
    if (value < 0x1p-14f) {
        return (uint16_t)nearbyintf(value * 0x1p24f);
    }
    int exponent;
    float fraction = frexpf(value, &exponent);
    return (uint16_t)(((exponent + 13) << 10) + (int)nearbyintf(fraction * 2048.f));
}

static int reference_candidates(const RefGeometry *ref, const Field *field, double tolerance,
    double lateral, const uint8_t *dropped, int32_t *out, uint16_t *depth) {
    Search search = {.bound = hypot(tolerance, lateral), .count = field->nodes_count};
    search.cells = calloc(search.count, sizeof(*search.cells));
    if (!search.cells) {
        return 0;
    }
    double scale = ldexp(1., field->level), shift = .5 * (scale - 1);
    for (size_t i = 0; i < search.count; i++) {
        const Node *p = field->nodes + i;
        search.cells[i] =
            (Cell){.x = scale * p->x + shift, .y = scale * p->y + shift, .z = scale * p->z + shift};
    }
    int ok = index_cells(&search);
    double lateral2 = lateral * lateral;
    for (size_t i = 0, row = 0; ok && i < ref->count; i++) {
        if (dropped[i]) {
            continue;
        }
        const RefVertex *p = ref->points + i;
        size_t near[32];
        int n = control_nearest(&search, p->xyz, near, 32, search.bound);
        int32_t picked[6] = {-1, -1, -1, -1, -1, -1};
        double best[6] = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
        for (int j = 0; j < n; j++) {
            size_t k = near[j] - 1;
            double score = reference_depth(p, search.cells + k, lateral2);
            if (score > tolerance || score >= best[5]) {
                continue;
            }
            int pos = 5;
            while (pos && score < best[pos - 1]) {
                best[pos] = best[pos - 1];
                picked[pos] = picked[pos - 1];
                pos--;
            }
            best[pos] = score;
            picked[pos] = (int32_t)k;
        }
        for (int j = 0; j < 6; j++, row++) {
            out[row] = picked[j];
            depth[row] = reference_half((float)best[j]);
        }
    }
    free(search.table);
    free(search.cells);
    return ok;
}

static int reference_archive(const char *path, const RefGeometry *ref, const Field *field,
    const uint8_t *dropped, const int32_t *group, const uint8_t *unresolved, size_t *kept,
    size_t *hit, size_t *pending) {
    for (size_t i = 0; i < ref->count; i++) {
        *kept += !dropped[i];
    }
    size_t n = *kept;
    int32_t *candidates = malloc(n * 6 * sizeof(*candidates));
    uint16_t *depth = malloc(n * 6 * sizeof(*depth));
    double *winding = malloc(n * sizeof(*winding));
    int32_t *indices = malloc(n * 5 * sizeof(*indices));
    float *coords = malloc(n * 3 * sizeof(*coords));
    uint8_t *mask = malloc(n);
    int ok = n && candidates && depth && winding && indices && coords && mask &&
        reference_candidates(ref, field, 40 / field->scan_voxel_um, 160 / field->scan_voxel_um,
            dropped, candidates, depth);
    for (size_t i = 0, at = 0; ok && i < ref->count; i++) {
        if (dropped[i]) {
            continue;
        }
        const RefVertex *p = ref->points + i;
        winding[at] = p->winding;
        indices[at] = group[p->unit];
        indices[n + at] = p->mesh;
        indices[2 * n + at] = p->unit;
        indices[3 * n + at] = p->row;
        indices[4 * n + at] = p->column;
        for (size_t k = 0; k < 3; k++) {
            coords[k * n + at] = p->xyz[k];
        }
        *hit += candidates[6 * at] >= 0;
        *pending += unresolved[i] != 0;
        mask[at++] = unresolved[i];
    }
    char *temp = NULL;
    int error = 0;
    Npz archive = {0};
    if (ok && asprintf(&temp, "%s.part.%ld", path, (long)getpid()) >= 0) {
        archive.file = zip_open(temp, ZIP_CREATE | ZIP_TRUNCATE, &error);
    }
    const char *column[] = {"group", "mesh", "unit", "row", "col"};
    const char *axis[] = {"x", "y", "z"};
    int64_t matrix[] = {n, 6}, points[] = {n};
    int64_t knots[] = {field->axis_count, 3}, level = field->level;
    double vertex = .15 * .15;
    int failed = !archive.file ||
        npz_add(
            &archive, "candidates", "<i4", matrix, 2, candidates, n * 6 * sizeof(*candidates)) ||
        npz_add(&archive, "depth", "<f2", matrix, 2, depth, n * 6 * sizeof(*depth)) ||
        npz_add(&archive, "Wg", "<f8", points, 1, winding, n * sizeof(*winding)) ||
        npz_add(&archive, "unresolved", "|b1", points, 1, mask, n) ||
        calibration_names(&archive, ref->names, ref->meshes) ||
        npz_add(&archive, "knots", "<f8", knots, 2, field->axis,
            field->axis_count * sizeof(*field->axis)) ||
        npz_add(&archive, "vertex_mm2", "<f8", NULL, 0, &vertex, sizeof(vertex)) ||
        npz_add(&archive, "voxel_um", "<f8", NULL, 0, &field->scan_voxel_um,
            sizeof(field->scan_voxel_um)) ||
        npz_add(&archive, "level", "<i8", NULL, 0, &level, sizeof(level));
    for (size_t k = 0; !failed && k < 5; k++) {
        failed =
            npz_add(&archive, column[k], "<i4", points, 1, indices + k * n, n * sizeof(*indices));
    }
    for (size_t k = 0; !failed && k < 3; k++) {
        failed = npz_add(&archive, axis[k], "<f4", points, 1, coords + k * n, n * sizeof(*coords));
    }
    failed = npz_finish(&archive, temp, failed);
    if (!failed && rename(temp, path)) {
        failed = 1;
        unlink(temp);
    }
    free(temp);
    free(mask);
    free(coords);
    free(indices);
    free(winding);
    free(depth);
    free(candidates);
    return !failed;
}

static int reference_truth(const char *control, const char *library, const char *out, int tier) {
    static const char *tier_a[] = {"20230702185753", "20230929220926", "20231005123336",
        "20231007101619", "20231012184424", "20231016151002", "20231022170901", "20231031143852",
        "20231106155351", "20231210121321", "20231221180251"};
    RefGeometry ref = {0};
    Field field = {0};
    int ok = !load_nodes_with_axis(&field, control, 0) && field.nodes_count > 0 &&
        field.nodes_count <= INT32_MAX && field.scan_voxel_um > 0 && field.level >= 0 &&
        field.level < 30 && field.sample && field.volume && reference_axis(&field);
    char *pattern = NULL;
    glob_t candidates = {0};
    ok = ok &&
        asprintf(&pattern, "%s/published/open_data/%s/segments/*/mesh/*-on-%s-*um.tifxyz", library,
            field.sample, field.volume) >= 0 &&
        !glob(pattern, 0, NULL, &candidates);
    double scale = ldexp(1., field.level);
    double lower = field.plane0 * HX_CELL * scale, upper = lower + field.grid[0] * HX_CELL * scale;
    for (size_t i = 0; ok && i < candidates.gl_pathc; i++) {
        const char *folder = candidates.gl_pathv[i];
        char *z = path_join(folder, "z.tif");
        int found = z && regular_file(z);
        free(z);
        if (!found) {
            continue;
        }
        const char *end = strstr(folder, "/mesh/"), *at = end;
        while (at && at > folder && at[-1] != '/') {
            at--;
        }
        char *name = at ? strndup(at, (size_t)(end - at)) : NULL;
        ok = name != NULL;
        if (ok && !strstr(name, "spiral-output")) {
            int selected = !tier;
            for (size_t j = 0; tier && j < sizeof(tier_a) / sizeof(*tier_a); j++) {
                selected |= !strcmp(name, tier_a[j]);
            }
            if (selected) {
                ok = reference_mesh(&ref, &field, folder, name, lower, upper);
            }
        }
        free(name);
    }
    globfree(&candidates);
    free(pattern);
    double tolerance = 40 / field.scan_voxel_um, lateral = 160 / field.scan_voxel_um;
    Search search = {.bound = hypot(2 * tolerance, lateral), .count = ref.count};
    if (ok && ref.count) {
        search.cells = calloc(ref.count, sizeof(*search.cells));
    }
    for (size_t i = 0; search.cells && i < ref.count; i++) {
        const float *p = ref.points[i].xyz;
        search.cells[i] = (Cell){.x = p[0], .y = p[1], .z = p[2]};
    }
    uint8_t *masks = search.cells ? calloc(ref.count, 2) : NULL;
    int32_t *group = masks ? malloc(ref.units * sizeof(*group)) : NULL;
    int groups = group && index_cells(&search)
        ? reference_frame(&ref, &search, tolerance, lateral, masks, group, masks + ref.count)
        : -1;
    free(search.table);
    free(search.cells);
    const char *name = strrchr(control, '/');
    name = name ? name + 1 : control;
    char *path = NULL;
    size_t kept = 0, hit = 0, pending = 0;
    ok = groups >= 0 && asprintf(&path, "%s/%s%s.npz", out, name, tier ? "-tier-a" : "") >= 0 &&
        reference_archive(
            path, &ref, &field, masks, group, masks + ref.count, &kept, &hit, &pending);
    if (ok) {
        printf("{\"control\":\"%s\",\"vertices\":%zu,\"kept\":%zu,\"meshes\":%zu,\"groups\":%d,"
               "\"unresolved\":%.4f,\"with_node\":%.4f,\"cm2\":%.1f}\n",
            name, ref.count, kept, ref.meshes, groups, pending / (double)kept, hit / (double)kept,
            kept * .15 * .15 / 100);
    } else {
        fprintf(stderr, "HerculeX: could not build winding truth for %s\n", control);
    }
    free(path);
    free(group);
    free(masks);
    free(ref.unit_mesh);
    for (size_t i = 0; i < ref.meshes; i++) {
        free(ref.names[i]);
    }
    free(ref.names);
    free(ref.points);
    field_close(&field);
    return ok;
}

int hx_wrap_truth_main(int argc, char **argv) {
    int tier = argc >= 2 && !strcmp(argv[1], "tier-a");
    int first = 1 + tier;
    if (argc <= first || !strcmp(argv[first], "--help") || !strcmp(argv[first], "-h")) {
        puts("Usage: ./run.sh calibrate.unwrap.sheet_wrap_truth [tier-a] CONTROL ...");
        return argc <= first;
    }
    char *library = local_root(library_root);
    char *out = library ? path_join(library, "calibration/geometry_wrap/truth") : NULL;
    char *root = library ? path_join(library, "controls/sheets") : NULL;
    int ok = root && out && make_folders(out);
    for (int i = first; ok && i < argc; i++) {
        char *folder = strchr(argv[i], '/') ? strdup(argv[i]) : path_join(root, argv[i]);
        char *end = folder ? folder + strlen(folder) : NULL;
        while (end && end > folder + 1 && end[-1] == '/') {
            *--end = 0;
        }
        ok = folder && reference_truth(folder, library, out, tier);
        free(folder);
    }
    free(root);
    free(out);
    free(library);
    return !ok;
}

typedef struct {
    float *xyz, *normal;
    int64_t *mesh, *wrap, *coincide, count, meshes, coincidences;
} LayoutTruth;

typedef struct {
    double area;
    int winding;
} LayoutPiece;

typedef struct {
    Search search;
    LayoutPiece *pieces;
    size_t span, capacity;
    double stride, gap, area, usable;
    int chirality;
} LayoutSurface;

static int layout_truth_open(LayoutTruth *t, const char *control) {
    NpzReader archive = {0};
    char *path = path_join(control, "truth.npz");
    int ok = path && !npz_open_reader(&archive, path);
    free(path);
    if (!ok) {
        return 0;
    }
    int64_t coordinates = 0, normals = 0;
    t->meshes = control_name_count(archive.file);
    ok = t->meshes > 0 && !npz_read(&archive, "points", "<f4", 4, (void **)&t->xyz, &coordinates) &&
        coordinates >= 3 && coordinates % 3 == 0 &&
        !npz_read(&archive, "normals", "<f4", 4, (void **)&t->normal, &normals) &&
        coordinates == normals;
    t->count = coordinates / 3;
    ok = ok && wrap_i64(&archive, "mesh", &t->mesh, t->count) &&
        wrap_i64(&archive, "wrap", &t->wrap, t->count);
    if (ok && zip_name_locate(archive.file, "coincide.npy", 0) >= 0) {
        ok = !npz_read(&archive, "coincide", "<i8", 8, (void **)&t->coincide, &t->coincidences);
    }
    npz_close_reader(&archive);
    for (int64_t i = 0; ok && i < t->count; i++) {
        ok = t->mesh[i] >= 0 && t->mesh[i] < t->meshes;
    }
    return ok;
}

static void layout_truth_close(LayoutTruth *t) {
    free(t->xyz);
    free(t->normal);
    free(t->mesh);
    free(t->wrap);
    free(t->coincide);
}

static int layout_points(LayoutSurface *s, const char *folder, int piece, double scale) {
    uint32_t w = 0, h = 0;
    int type;
    double *grid = read_raw_mesh(folder, &w, &h, &type);
    size_t size = (size_t)w * h;
    char *path = path_join(folder, "quality.npy");
    FILE *file = path ? fopen(path, "rb") : NULL;
    free(path);
    uint8_t *quality = file ? malloc(size) : NULL;
    uint8_t header[10];
    int ok = grid && quality && fread(header, 1, sizeof(header), file) == sizeof(header) &&
        !memcmp(header, "\x93NUMPY", 6) && header[6] == 1 &&
        !fseek(file, header[8] + 256 * header[9], SEEK_CUR) &&
        fread(quality, 1, size, file) == size;
    if (file) {
        fclose(file);
    }
    Search *points = &s->search;
    for (int kind = 0; ok && kind < 4; kind++) {
        int across = kind & 1, down = (kind >> 1) * w;
        for (uint32_t r = 0; ok && r + (kind >> 1) < h; r++) {
            for (uint32_t c = 0; c + across < w; c++) {
                size_t i = (size_t)r * w + c;
                if (quality[i] != 1 || quality[i + across] != 1 || quality[i + down] != 1 ||
                    quality[i + across + down] != 1) {
                    continue;
                }
                if (points->count == s->capacity) {
                    size_t cap = s->capacity ? 2 * s->capacity : 65536;
                    Cell *more = realloc(points->cells, cap * sizeof(*more));
                    if (!more) {
                        ok = 0;
                        break;
                    }
                    points->cells = more;
                    s->capacity = cap;
                }
                double xyz[3];
                for (size_t axis = 0; axis < 3; axis++) {
                    const double *g = grid + axis * size + i;
                    double value = g[0];
                    if (kind == 3) {
                        value = (value + g[across + down] + g[across] + g[down]) * .25;
                    } else if (kind) {
                        value = (value + g[across + down]) * .5;
                    }
                    xyz[axis] = (value - .5 * (scale - 1)) / scale;
                }
                points->cells[points->count++] =
                    (Cell){.x = xyz[0], .y = xyz[1], .z = xyz[2], .row = piece};
            }
        }
    }
    free(quality);
    free(grid);
    return ok;
}

static int layout_surface_open(LayoutSurface *s, const char *out, int level, int joint) {
    json_object *record = read_record(out, "record.json"), *gap = NULL;
    json_object *pieces = json_object_object_get(record, "pieces");
    int ok = json_object_is_type(pieces, json_type_array);
    double scale = ldexp(1., level);
    s->gap = json_object_object_get_ex(record, "gap_um", &gap) ? json_object_get_double(gap) : 160.;
    s->area = json_object_get_double(json_object_object_get(record, "area_cm2"));
    s->usable = json_object_get_double(json_object_object_get(record, "area_in_1cm2_pieces"));
    s->chirality = json_object_get_int(json_object_object_get(record, "chirality"));
    s->stride = HX_CELL;
    for (size_t i = 0; ok && i < json_object_array_length(pieces); i++) {
        json_object *piece = json_object_array_get_idx(pieces, i);
        size_t id = json_object_get_uint64(json_object_object_get(piece, "piece"));
        if (id >= INT32_MAX) {
            ok = 0;
            break;
        }
        s->span = id + 1 > s->span ? id + 1 : s->span;
    }
    s->pieces = ok ? calloc(s->span ? s->span : 1, sizeof(*s->pieces)) : NULL;
    ok = s->pieces != NULL;
    for (size_t i = 0; ok && i < json_object_array_length(pieces); i++) {
        json_object *piece = json_object_array_get_idx(pieces, i);
        int id = json_object_get_int(json_object_object_get(piece, "piece"));
        s->pieces[id] =
            (LayoutPiece){json_object_get_double(json_object_object_get(piece, "area_mm2")),
                json_object_get_int(json_object_object_get(piece, "winding"))};
        if (!json_object_get_boolean(json_object_object_get(piece, "written"))) {
            continue;
        }
        char *path = NULL;
        ok = asprintf(&path, "%s/piece-%04d", out, id) >= 0;
        if (ok && !joint && !s->search.count) {
            json_object *meta = read_record(path, "meta.json");
            double stride = json_object_get_double(json_object_object_get(meta, "stride"));
            if (stride > 0) {
                s->stride = stride / scale;
            }
            json_object_put(meta);
        }
        ok = ok && layout_points(s, path, id, scale);
        free(path);
    }
    json_object_put(record);
    s->search.bound = 1.5 * s->stride;
    return ok && index_cells(&s->search);
}

static void layout_surface_close(LayoutSurface *s) {
    free(s->search.table);
    free(s->search.cells);
    free(s->pieces);
}

static int64_t layout_hit(const Search *search, const double p[3], const float normal[3],
    const float *candidate_normals, double tolerance) {
    size_t near[8];
    int count = control_nearest_xyz(search, p, near, 8, search->bound, -1);
    double best = INFINITY;
    int64_t hit = -1;
    for (int k = 0; k < count; k++) {
        size_t j = near[k] - 1;
        const Cell *q = search->cells + j;
        const float *n = candidate_normals ? candidate_normals + 3 * q->row : normal;
        double d = fabs((q->x - p[0]) * n[0] + (q->y - p[1]) * n[1] + (q->z - p[2]) * n[2]);
        if (d <= tolerance && d < best) {
            hit = j;
            best = d;
        }
    }
    return hit;
}

static int layout_coincides(const LayoutTruth *truth, int64_t vertex, int64_t home) {
    int64_t key = vertex * truth->meshes + home, lo = 0, hi = truth->coincidences;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (truth->coincide[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < truth->coincidences && truth->coincide[lo] == key;
}

static double layout_round(double value, double scale) { return nearbyint(value * scale) / scale; }

static void layout_ratio(json_object *row, const char *key, double count, double total) {
    json_object_object_add(
        row, key, total > 0 ? json_object_new_double(layout_round(count / total, 1e4)) : NULL);
}

static int layout_int_order(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int layout_mesh_metrics(json_object *row, const char *control, const LayoutTruth *truth,
    const LayoutSurface *surface, double voxel) {
    Field field = {0};
    Search nodes = {.bound = 1.5 * HX_CELL};
    size_t span = surface->span, meshes = truth->meshes, count = truth->count;
    int *piece = malloc(count * sizeof(*piece));
    int *home = malloc((span ? span : 1) * sizeof(*home));
    int *named_offset = malloc(count * sizeof(*named_offset));
    int64_t *named = malloc(meshes * sizeof(*named));
    size_t *hits = calloc((span ? span : 1) * meshes, sizeof(*hits));
    int ok =
        piece && home && named && named_offset && hits && !load_nodes_with_axis(&field, control, 0);
    nodes.count = field.nodes_count;
    nodes.cells = ok ? calloc(nodes.count ? nodes.count : 1, sizeof(*nodes.cells)) : NULL;
    ok = nodes.cells != NULL;
    for (size_t i = 0; ok && i < nodes.count; i++) {
        const Node *n = field.nodes + i;
        nodes.cells[i] = (Cell){.x = n->x, .y = n->y, .z = n->z};
    }
    field_close(&field);
    ok = ok && index_cells(&nodes);
    for (size_t i = 0; ok && i < meshes; i++) {
        named[i] = INT64_MIN;
    }
    size_t covered = 0, shifted = 0, detected = 0, usable = 0;
    double tolerance = 40 / voxel, half_gap = surface->gap / (2 * voxel);
    for (size_t i = 0; ok && i < count; i++) {
        const float *xyz = truth->xyz + 3 * i, *normal = truth->normal + 3 * i;
        double p[] = {xyz[0], xyz[1], xyz[2]};
        size_t mesh = truth->mesh[i];
        if (named[mesh] == INT64_MIN) {
            named[mesh] = truth->wrap[i];
        }
        int64_t hit = layout_hit(&surface->search, p, normal, NULL, tolerance);
        piece[i] = hit < 0 ? -1 : surface->search.cells[hit].row;
        if (hit >= 0) {
            covered++;
            int id = piece[i];
            if (surface->pieces[id].area >= 100) {
                hits[(size_t)id * meshes + mesh]++;
                usable++;
            } else {
                piece[i] = -1;
            }
        }
        detected += layout_hit(&nodes, p, normal, NULL, tolerance) >= 0;
        for (size_t k = 0; k < 3; k++) {
            p[k] += normal[k] * half_gap;
        }
        shifted += layout_hit(&surface->search, p, normal, NULL, tolerance) >= 0;
    }
    for (size_t i = 0; ok && i < span; i++) {
        size_t best = 0;
        home[i] = -1;
        for (size_t j = 0; j < meshes; j++) {
            size_t votes = hits[i * meshes + j];
            if (votes > best) {
                home[i] = (int)j;
                best = votes;
            }
        }
    }
    size_t right = 0, wrong = 0, unjudged = 0, name_count = 0;
    for (size_t i = 0; ok && i < count; i++) {
        int id = piece[i];
        if (id < 0) {
            continue;
        }
        int64_t m = truth->mesh[i], h = home[id], known = truth->wrap[i];
        int correct = m == h || layout_coincides(truth, i, h);
        int conflict = named[h] >= 0 && named[m] >= 0 && named[h] != named[m];
        right += correct;
        wrong += !correct && conflict;
        unjudged += !correct && !conflict;
        if (known >= 0) {
            named_offset[name_count++] = surface->pieces[id].winding - surface->chirality * known;
        }
    }
    if (ok) {
        json_object_object_add(row, "vertices", json_object_new_int64(count));
        json_object_object_add(row, "meshes", json_object_new_int64(meshes));
        layout_ratio(row, "detected", detected, count);
        layout_ratio(row, "covered", covered, count);
        layout_ratio(row, "covered_null", shifted, count);
        layout_ratio(row, "in_usable_pieces", usable, count);
        layout_ratio(row, "recovery", right + unjudged, count);
        layout_ratio(row, "recovery_strict", right, count);
        layout_ratio(row, "purity", right, right + wrong ? right + wrong : 1);
        layout_ratio(row, "unjudged", unjudged, count);
        if (name_count > 100) {
            qsort(named_offset, name_count, sizeof(*named_offset), layout_int_order);
            size_t mode = 0;
            for (size_t i = 0, j = 0; i < name_count; i = j) {
                j = i + 1;
                while (j < name_count && named_offset[j] == named_offset[i]) {
                    j++;
                }
                mode = j - i > mode ? j - i : mode;
            }
            layout_ratio(row, "frame_agreement", mode, name_count);
        }
    }
    free(nodes.table);
    free(nodes.cells);
    free(hits);
    free(named);
    free(named_offset);
    free(home);
    free(piece);
    return ok;
}

static int layout_join_order(const void *a, const void *b) {
    const HxJoin *x = a, *y = b;
    if (x->lower != y->lower) {
        return (x->lower > y->lower) - (x->lower < y->lower);
    }
    return (x->upper > y->upper) - (x->upper < y->upper);
}

static int layout_truth_search(
    const LayoutTruth *truth, Search *search, double lower, double upper, int named) {
    size_t n = 0;
    for (int64_t i = 0; i < truth->count; i++) {
        float z = truth->xyz[3 * i + 2];
        n += z >= lower && z < upper && (!named || truth->wrap[i] >= 0);
    }
    search->cells = calloc(n ? n : 1, sizeof(*search->cells));
    if (!search->cells) {
        return 0;
    }
    for (int64_t i = 0; i < truth->count; i++) {
        const float *p = truth->xyz + 3 * i;
        if (p[2] >= lower && p[2] < upper && (!named || truth->wrap[i] >= 0)) {
            search->cells[search->count++] = (Cell){
                .x = p[0], .y = p[1], .z = p[2], .row = named ? i : truth->mesh[i], .column = i};
        }
    }
    return index_cells(search);
}

static int layout_joint_metrics(json_object *row, const LayoutTruth *truth,
    const LayoutSurface surface[2], const char *control, const char *const out[2], int seam,
    double voxel) {
    SeamField field = {0};
    int ok = !seam_field_open(&field, control, control, out[0], out[1], seam);
    int64_t count = ok
        ? hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges, field.edge_count,
              field.pairs, field.weights, field.pair_count, NULL, 0)
        : -1;
    HxJoin *joins = count >= 0 ? malloc((count ? count : 1) * sizeof(*joins)) : NULL;
    size_t *votes = count >= 0 ? calloc(2 * (count ? count : 1), sizeof(*votes)) : NULL;
    int64_t *wrap = count > 0 ? malloc(field.nodes_count * sizeof(*wrap)) : NULL;
    Boundary boundary[2] = {0};
    ok = joins && votes && !boundary_open(boundary, out[0]) &&
        !boundary_open(boundary + 1, out[1]) &&
        hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges, field.edge_count,
            field.pairs, field.weights, field.pair_count, joins, count) == count;
    free(boundary[0].rows);
    free(boundary[1].rows);
    int64_t span = boundary[0].pieces, pieces = span + boundary[1].pieces;
    int32_t *parents = ok && pieces >= 0 && pieces <= INT_MAX
        ? malloc((pieces ? pieces : 1) * sizeof(*parents))
        : NULL;
    ok = parents != NULL;
    for (int32_t i = 0; ok && i < pieces; i++) {
        parents[i] = i;
    }
    for (int64_t i = 0; ok && i < count; i++) {
        int32_t a = join_root(parents, joins[i].lower);
        int32_t b = join_root(parents, span + joins[i].upper);
        parents[b] = a;
    }
    if (ok) {
        qsort(joins, count, sizeof(*joins), layout_join_order);
    }
    double tolerance = 40 / voxel;
    Search named = {.bound = 1.5 * HX_CELL};
    if (ok && count) {
        ok = wrap && layout_truth_search(truth, &named, -INFINITY, INFINITY, 1);
        for (int64_t i = 0; ok && i < field.nodes_count; i++) {
            const Node *node = field.nodes + i;
            double p[] = {node->x, node->y, node->z};
            int64_t pick = layout_hit(&named, p, NULL, truth->normal, tolerance);
            wrap[i] = pick < 0 ? -1 : truth->wrap[named.cells[pick].row];
        }
        for (int64_t i = 0; ok && i < field.edge_count; i++) {
            int32_t a = field.edges[i].a, b = field.edges[i].b;
            if (field.side[a] > field.side[b]) {
                int32_t swap = a;
                a = b;
                b = swap;
            }
            if (field.side[a] == field.side[b] || field.piece[a] < 0 || field.piece[b] < 0 ||
                wrap[a] < 0 || wrap[b] < 0) {
                continue;
            }
            HxJoin key = {.lower = field.piece[a], .upper = field.piece[b]};
            HxJoin *found = bsearch(&key, joins, count, sizeof(*joins), layout_join_order);
            if (found) {
                size_t index = found - joins;
                votes[index]++;
                votes[count + index] += wrap[a] != wrap[b];
            }
        }
    }
    free(named.table);
    free(named.cells);
    int64_t judged = 0, false_joins = 0, all_votes = 0, false_votes = 0;
    for (int64_t i = 0; ok && i < count; i++) {
        if (votes[i]) {
            judged++;
            all_votes += joins[i].count;
            if (2 * votes[count + i] > votes[i]) {
                false_joins++;
                false_votes += joins[i].count;
            }
        }
    }
    free(wrap);
    free(votes);
    free(joins);
    seam_field_close(&field);
    Search above = {.bound = 2.5 * HX_CELL};
    double z = seam * HX_CELL, depth = 4 * HX_CELL;
    ok = ok && layout_truth_search(truth, &above, z, z + depth, 0);
    size_t seam_pairs = 0, covered = 0, correct = 0;
    for (int64_t i = 0; ok && i < truth->count; i++) {
        const float *a = truth->xyz + 3 * i;
        if (a[2] < z - depth || a[2] >= z) {
            continue;
        }
        double p[] = {a[0], a[1], a[2]};
        size_t neighbor;
        if (!control_nearest_xyz(&above, p, &neighbor, 1, above.bound, truth->mesh[i])) {
            continue;
        }
        const Cell *q = above.cells + neighbor - 1;
        double xyz[] = {q->x, q->y, q->z};
        int64_t lower = layout_hit(&surface[0].search, p, truth->normal + 3 * i, NULL, tolerance);
        int64_t upper =
            layout_hit(&surface[1].search, xyz, truth->normal + 3 * q->column, NULL, tolerance);
        seam_pairs++;
        if (lower >= 0 && upper >= 0) {
            covered++;
            correct += join_root(parents, surface[0].search.cells[lower].row) ==
                join_root(parents, span + surface[1].search.cells[upper].row);
        }
    }
    if (ok) {
        json_object_object_add(row, "seam_pairs", json_object_new_int64(seam_pairs));
        json_object_object_add(row, "covered_pairs", json_object_new_int64(covered));
        json_object_object_add(row, "joins", json_object_new_int64(count));
        layout_ratio(row, "joint_accuracy", correct, covered);
        json_object_object_add(row, "judged_joins", json_object_new_int64(judged));
        json_object_object_add(row, "false_joins", json_object_new_int64(false_joins));
        layout_ratio(row, "false_join_share", false_votes, all_votes);
    }
    free(above.table);
    free(above.cells);
    free(parents);
    return ok;
}

static json_object *layout_control_score(
    const char *control, json_object *record, const char *scratch, int joint, int chirality) {
    const char *name = strrchr(control, '/') + 1;
    const char *sample = json_object_get_string(json_object_object_get(record, "sample"));
    const char *scan = json_object_get_string(json_object_object_get(record, "volume"));
    int level = json_object_get_int(json_object_object_get(record, "level"));
    double voxel = json_object_get_double(json_object_object_get(record, "voxel_um"));
    json_object *planes = json_object_object_get(record, "cell_planes");
    int low = json_object_get_int(json_object_array_get_idx(planes, 0));
    int high = json_object_get_int(json_object_array_get_idx(planes, 1)), seam = (low + high) / 2;
    int clip[2][2] = {{low, seam}, {seam, high}};
    LayoutTruth truth = {0};
    LayoutSurface surfaces[2] = {0};
    char *base = NULL;
    int ok = sample && scan && level >= 0 && level < 30 && voxel > 0 && high > low &&
        (chirality || !read_curl_value(sample, scan, &chirality)) &&
        !hx_umbilicus_keep(sample, scan) && layout_truth_open(&truth, control) &&
        asprintf(&base, "%s/%s.XXXXXX", scratch, name) >= 0 && mkdtemp(base);
    if (!chirality) {
        fprintf(stderr, "HerculeX: %s requires the scan's recorded curl or --curl 1|-1\n", name);
    }
    double started = level_clock();
    const char *paths[2] = {NULL, NULL};
    char *owned[2] = {NULL, NULL};
    for (int i = 0; ok && i < (joint ? 2 : 1); i++) {
        owned[i] = path_join(base, i ? "upper" : "lower");
        paths[i] = owned[i];
        ok = paths[i] &&
            !write_unit_window(control, paths[i], chirality, 0, joint ? clip[i] : NULL, !joint);
    }
    double seconds = level_clock() - started;
    for (int i = 0; ok && i < (joint ? 2 : 1); i++) {
        ok = layout_surface_open(surfaces + i, paths[i], level, joint);
    }
    json_object *result = ok ? json_object_new_object() : NULL;
    if (result) {
        json_object_object_add(result, "control", json_object_new_string(name));
        if (joint) {
            ok = layout_joint_metrics(result, &truth, surfaces, control, paths, seam, voxel);
        } else {
            json_object_object_add(result, "sample", json_object_new_string(sample));
            json_object_object_add(result, "voxel_um", json_object_new_double(voxel));
            json_object_object_add(
                result, "seconds", json_object_new_double(layout_round(seconds, 10)));
            json_object_object_add(
                result, "area_cm2", json_object_new_double(layout_round(surfaces[0].area, 10)));
            json_object_object_add(result, "area_in_1cm2_pieces",
                json_object_new_double(layout_round(surfaces[0].usable, 10)));
            ok = layout_mesh_metrics(result, control, &truth, surfaces, voxel);
        }
    }
    if (!ok) {
        json_object_put(result);
        result = NULL;
    }
    layout_surface_close(surfaces);
    layout_surface_close(surfaces + 1);
    layout_truth_close(&truth);
    free(owned[0]);
    free(owned[1]);
    if (base) {
        remove_path(base);
    }
    free(base);
    return result;
}

int hx_sheet_scoring_main(int argc, char **argv) {
    int joint = argc >= 2 && !strcmp(argv[1], "joint"), curl = 0;
    int first = 1 + joint;
    if (argc > first && (!strcmp(argv[first], "--help") || !strcmp(argv[first], "-h"))) {
        puts("Usage: ./run.sh calibrate.unwrap.sheet_scoring [joint] [--curl 1|-1] [CONTROL ...]");
        return 0;
    }
    if (argc >= first + 2 && !strcmp(argv[first], "--curl") &&
        (!strcmp(argv[first + 1], "1") || !strcmp(argv[first + 1], "-1"))) {
        curl = argv[first + 1][0] == '-' ? -1 : 1;
        first += 2;
    }
    char *library = local_root(library_root);
    char *scratch = library ? path_join(library, "calibration/sheet_study") : NULL;
    char *pattern = library ? path_join(library, "controls/sheets/*/truth.npz") : NULL;
    glob_t controls = {0};
    int listed = pattern ? glob(pattern, 0, NULL, &controls) : GLOB_ABORTED;
    int ok = (listed == 0 || listed == GLOB_NOMATCH) && scratch && make_folders(scratch);
    json_object *rows = json_object_new_array(), *report = json_object_new_object();
    ok = ok && rows && report;
    const char *first_key = joint ? "joint_accuracy" : "recovery";
    const char *second_key = joint ? "false_join_share" : "real_mean_recovery";
    double totals[2] = {0};
    size_t accepted[2] = {0};
    for (size_t i = 0; ok && i < controls.gl_pathc; i++) {
        char *folder = strdup(controls.gl_pathv[i]);
        if (!folder) {
            ok = 0;
            break;
        }
        *strrchr(folder, '/') = 0;
        const char *name = strrchr(folder, '/') + 1;
        json_object *record = read_record(folder, "record.json");
        const char *sample = json_object_get_string(json_object_object_get(record, "sample"));
        int chosen = argc == first;
        for (int j = first; j < argc; j++) {
            chosen |= !strcmp(name, argv[j]) || (sample && !strcmp(sample, argv[j]));
        }
        if (chosen && record) {
            json_object *row = layout_control_score(folder, record, scratch, joint, curl);
            ok = row != NULL;
            if (ok) {
                json_object *a = json_object_object_get(row, first_key);
                json_object *b = joint ? json_object_object_get(row, second_key) : a;
                accepted[0] += a != NULL;
                accepted[1] += b != NULL;
                totals[0] += json_object_get_double(a);
                totals[1] += json_object_get_double(b);
                puts(json_object_to_json_string_ext(row, JSON_C_TO_STRING_PLAIN));
                fflush(stdout);
                json_object_array_add(rows, row);
            } else {
                fprintf(stderr, "HerculeX: could not score sheet control %s\n", name);
            }
        }
        json_object_put(record);
        free(folder);
    }
    if (ok) {
        layout_ratio(report, joint ? first_key : "mean_recovery", totals[0], accepted[0]);
        layout_ratio(report, second_key, totals[1], accepted[1]);
        puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PLAIN));
        if (argc == first) {
            json_object_object_add(report, "controls", json_object_get(rows));
            const char *path =
                joint ? "records/surface/joint-study.json" : "records/surface/sheet-study.json";
            ok = make_folders("records/surface") && !record_write_layout(path, report);
        }
    }
    json_object_put(report);
    json_object_put(rows);
    globfree(&controls);
    free(pattern);
    free(scratch);
    free(library);
    return !ok;
}

typedef struct {
    size_t unit, size;
    int tier_a;
} MeshRank;

typedef struct {
    int64_t q, f;
} MeshPair;

static int mesh_rank_order(const void *a, const void *b) {
    const MeshRank *x = a, *y = b;
    if (x->tier_a != y->tier_a) {
        return y->tier_a - x->tier_a;
    }
    if (x->size != y->size) {
        return x->size > y->size ? -1 : 1;
    }
    return (x->unit > y->unit) - (x->unit < y->unit);
}

static int mesh_tier(uint32_t code, const char *selection) {
    return code && code < 128 && strchr(selection, (int)code) != NULL;
}

static double mesh_depth(
    const Cell *p, const Cell *q, double *const normal[], size_t i, double lateral) {
    double x = q->x - p->x, y = q->y - p->y, z = q->z - p->z;
    double dep = fabs(x * normal[0][i] + y * normal[1][i] + z * normal[2][i]);
    return fmax(0., x * x + y * y + z * z - dep * dep) <= lateral * lateral ? dep : INFINITY;
}

static void mesh_audit_row(
    json_object *array, int64_t a, int64_t b, int64_t c, int64_t hits, double agreement) {
    json_object *row = json_object_new_array();
    json_object_array_add(row, json_object_new_int64(a));
    json_object_array_add(row, json_object_new_int64(b));
    json_object_array_add(row, json_object_new_int64(c));
    if (hits) {
        json_object_array_add(row, json_object_new_int64(hits));
        json_object_array_add(row, json_object_new_double(agreement));
    }
    json_object_array_add(array, row);
}

static int mesh_turn_order(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static int mesh_named_shift(const double *w, const int64_t *unit, size_t n, size_t u,
    const long window[2], size_t size, int64_t *result) {
    int64_t *turns = malloc(size * sizeof(*turns));
    if (!turns) {
        return 0;
    }
    for (size_t i = 0, at = 0; i < n; i++) {
        if ((size_t)unit[i] == u) {
            turns[at++] = (int64_t)floor(w[i]);
        }
    }
    qsort(turns, size, sizeof(*turns), mesh_turn_order);
    int64_t best = 0;
    size_t score = 0, end = 0;
    for (size_t i = 0; i < size; i++) {
        while (end < size && turns[end] - turns[i] <= window[1] - window[0]) {
            end++;
        }
        int64_t shift = window[0] - turns[i];
        size_t inside = end - i;
        if (inside > score || (inside == score && shift < best)) {
            best = shift;
            score = inside;
        }
    }
    free(turns);
    *result = best;
    return 1;
}

static int mesh_save_archive(NpzReader *src, const char *path, const char *selection, int64_t n,
    const uint8_t *keep, const uint8_t *duplicate, const uint8_t *unresolved, const int32_t *group,
    const double *w, const MeshPair *pairs, size_t count, json_object *audit) {
    char *part = NULL, *side = NULL;
    size_t stem = strlen(path);
    if (stem >= 4 && !strcmp(path + stem - 4, ".npz")) {
        stem -= 4;
    }
    int made = stem <= INT_MAX && asprintf(&part, "%s.part.%ld", path, (long)getpid()) >= 0 &&
        asprintf(&side, "%.*s.audit.json", (int)stem, path) >= 0;
    char *parent = made ? strdup(path) : NULL;
    char *slash = parent ? strrchr(parent, '/') : NULL;
    if (slash && slash != parent) {
        *slash = 0;
        made = make_folders(parent);
    }
    free(parent);
    Npz dst = {0};
    int error;
    if (made) {
        dst.file = zip_open(part, ZIP_CREATE | ZIP_TRUNCATE, &error);
    }
    int ok = dst.file != NULL;
    zip_int64_t files = zip_get_num_entries(src->file, 0);
    for (zip_int64_t i = 0; ok && i < files; i++) {
        const char *name = zip_get_name(src->file, i, 0);
#if LIBZIP_VERSION_MAJOR > 1 || (LIBZIP_VERSION_MAJOR == 1 && LIBZIP_VERSION_MINOR >= 10)
        zip_source_t *source =
            name ? zip_source_zip_file(dst.file, src->file, i, 0, 0, -1, NULL) : NULL;
#else
        zip_source_t *source = name ? zip_source_zip(dst.file, src->file, i, 0, 0, -1) : NULL;
#endif
        ok = source && zip_file_add(dst.file, name, source, ZIP_FL_ENC_UTF_8) >= 0;
        if (!ok && source) {
            zip_source_free(source);
        }
    }
    int64_t *indices = count ? malloc(count * sizeof(*indices)) : NULL;
    ok = ok && (!count || indices);
    int64_t length = count;
    for (int i = 0; ok && i < 2; i++) {
        for (size_t j = 0; j < count; j++) {
            indices[j] = i ? pairs[j].f : pairs[j].q;
        }
        ok = !npz_add(
            &dst, i ? "pair_f" : "pair_q", "<i8", &length, 1, indices, count * sizeof(*indices));
    }
    free(indices);
    const char *names[] = {"keep", "duplicate", "unresolved", "group", "Wg"};
    const void *arrays[] = {keep, duplicate, unresolved, group, w};
    for (int i = 0; ok && i < 5; i++) {
        char key[64];
        snprintf(key, sizeof(key), "%s_%s", names[i], selection);
        int width = i < 3 ? 1 : i == 3 ? 4 : 8;
        const char *type = i < 3 ? "|b1" : i == 3 ? "<i4" : "<f8";
        ok = !npz_add(&dst, key, type, &n, 1, arrays[i], (size_t)n * width);
    }
    ok = !npz_finish(&dst, part, !ok) && !rename(part, path) && !record_write_layout(side, audit);
    if (!ok && part) {
        unlink(part);
    }
    free(part);
    free(side);
    return ok;
}

int hx_mesh_reference_main(int argc, char **argv) {
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        puts("Usage: calibrate.unwrap.mesh_reference INPUT.npz OUT.npz TIERS [--by-name]");
        return 0;
    }
    if (argc < 4 || argc > 5 || !*argv[3] || strlen(argv[3]) > 32 ||
        strspn(argv[3], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") !=
            strlen(argv[3]) ||
        (argc == 5 && strcmp(argv[4], "--by-name") != 0)) {
        return 2;
    }
    NpzReader file = {0};
    if (npz_open_reader(&file, argv[1])) {
        fprintf(stderr, "cannot open reference archive %s\n", argv[1]);
        return 1;
    }
    int64_t n = 0, meshes = 0, units = 0, columns = 0;
    double *w = wrap_column(&file, "W", &n), *v[6] = {0};
    int64_t *unit = NULL, *mesh = NULL, *unit_mesh = NULL;
    uint32_t *tiers = NULL, *names = NULL;
    int ok = w && n > 0 && n <= INT32_MAX &&
        !npz_read(&file, "tiers", "'<U1'", 4, (void **)&tiers, &meshes) && meshes > 0 &&
        !npz_read(&file, "names", "'<U", 4, (void **)&names, &columns) && columns >= meshes &&
        columns % meshes == 0 && wrap_i64(&file, "unit", &unit, n) &&
        wrap_i64(&file, "mesh", &mesh, n);
    int32_t *narrow = NULL;
    if (ok && npz_read(&file, "unit_mesh", "<i8", 8, (void **)&unit_mesh, &units)) {
        ok = !npz_read(&file, "unit_mesh", "<i4", 4, (void **)&narrow, &units) &&
            wrap_i64(&file, "unit_mesh", &unit_mesh, units);
    }
    free(narrow);
    ok = ok && units > 0 && units < INT32_MAX;
    const char *keys[] = {"x", "y", "z", "nx", "ny", "nz"};
    for (int i = 0; ok && i < 6; i++) {
        int64_t length;
        v[i] = wrap_column(&file, keys[i], &length);
        ok = v[i] && length == n;
    }
    size_t m = ok ? (size_t)units + 1 : 1, vertices = ok ? (size_t)n : 1;
    size_t *position = calloc(m, sizeof(*position)), *sizes = calloc(m, sizeof(*sizes));
    uint8_t *inside = calloc(m, 1), *flags = calloc(vertices, 3), *keep = flags;
    uint8_t *duplicate = flags ? flags + vertices : NULL,
            *unknown = flags ? flags + 2 * vertices : NULL;
    MeshRank *rank = calloc(m, sizeof(*rank));
    int64_t *offset = calloc(m, sizeof(*offset));
    long (*ranges)[2] = calloc(m, sizeof(*ranges));
    int32_t *group = calloc(m, sizeof(*group)), *out_group = calloc(vertices, sizeof(*out_group));
    int32_t *queue = calloc(m, sizeof(*queue));
    size_t *mapping = calloc(vertices, sizeof(*mapping));
    const double tolerance = 40. / 7.91, lateral = 160. / 7.91;
    Search search = {.count = vertices, .bound = hypot(2 * tolerance, lateral)};
    search.cells = calloc(vertices, sizeof(*search.cells));
    ok = ok && position && sizes && inside && flags && rank && offset && ranges && group &&
        out_group && queue && mapping && search.cells;
    long width = ok ? columns / meshes : 0;
    char *name = ok ? malloc((size_t)width + 1) : NULL;
    ok = ok && name;
    for (size_t i = 0; ok && i < vertices; i++) {
        ok = unit[i] >= 0 && unit[i] < units && mesh[i] >= 0 && mesh[i] < meshes &&
            isfinite(w[i]) && fabs(w[i]) < 0x1p52;
        if (ok) {
            sizes[unit[i]]++;
            search.cells[i] = (Cell){.x = v[0][i], .y = v[1][i], .z = v[2][i]};
        }
    }
    for (size_t i = 0; ok && i + 1 < m; i++) {
        int64_t src = unit_mesh[i];
        ok = src >= 0 && src < meshes;
        if (ok) {
            uint32_t tier = tiers[src];
            inside[i] = mesh_tier(tier, argv[3]);
            rank[i] = (MeshRank){i, sizes[i], tier == 'A'};
            for (long j = 0; j < width; j++) {
                uint32_t code = names[src * width + j];
                name[j] = code < 128 ? (char)code : '?';
            }
            name[width] = 0;
            if (!reference_windings(name, ranges[i])) {
                ranges[i][0] = -1;
            }
        }
    }
    free(name);
    size_t count = 0, capacity = 0;
    MeshPair *pairs = NULL;
    if (ok) {
        qsort(rank, m - 1, sizeof(*rank), mesh_rank_order);
        for (size_t i = 0; i < m - 1; i++) {
            position[rank[i].unit] = i;
        }
        ok = index_cells(&search);
    }
    for (size_t i = 0; ok && i < vertices; i++) {
        const Cell *p = search.cells + i;
        double xyz[] = {p->x, p->y, p->z};
        size_t near[12];
        int hits = control_nearest_xyz(&search, xyz, near, 12, hypot(tolerance, lateral), -1);
        for (int j = 0; ok && j < hits; j++) {
            size_t k = near[j] - 1;
            int64_t a = unit[i], b = unit[k];
            if (a == b || mesh_depth(p, search.cells + k, v + 3, i, lateral) > tolerance) {
                continue;
            }
            if (count == capacity) {
                size_t cap = capacity ? 2 * capacity : 1024;
                MeshPair *next = cap > capacity && cap <= SIZE_MAX / sizeof(*pairs)
                    ? realloc(pairs, cap * sizeof(*pairs))
                    : NULL;
                if (!next) {
                    ok = 0;
                    break;
                }
                pairs = next;
                capacity = cap;
            }
            pairs[count++] = (MeshPair){i, k};
            duplicate[i] |= inside[a] && inside[b] && position[b] < position[a];
        }
    }
    free(search.table);
    search.table = NULL;
    RefEdge *edges = count ? malloc(count * sizeof(*edges)) : NULL;
    ok = ok && (!count || edges);
    for (size_t i = 0; ok && i < count; i++) {
        size_t q = pairs[i].q, f = pairs[i].f;
        edges[i] = (RefEdge){unit[q], unit[f], nearbyint(w[q] - w[f])};
    }
    json_object *audit = json_object_new_object(), *edge_log = json_object_new_array();
    json_object *flagged = json_object_new_array(), *conflicts = json_object_new_array();
    json_object_object_add(audit, "edges", edge_log);
    json_object_object_add(audit, "flagged", flagged);
    json_object_object_add(audit, "conflicts", conflicts);
    if (ok && count) {
        qsort(edges, count, sizeof(*edges), reference_edge_order);
    }
    size_t links = 0;
    for (size_t i = 0; ok && i < count;) {
        RefEdge best = edges[i];
        size_t end = i, votes = 0;
        while (end < count && edges[end].a == best.a && edges[end].b == best.b) {
            size_t next = end + 1;
            while (next < count && edges[next].a == best.a && edges[next].b == best.b &&
                edges[next].step == edges[end].step) {
                next++;
            }
            if (next - end > votes) {
                best.step = edges[end].step;
                votes = next - end;
            }
            end = next;
        }
        size_t a = best.a, b = best.b, total = end - i;
        double agreement = votes / (double)total;
        mesh_audit_row(edge_log, a, b, best.step, total, agreement);
        if (inside[a] && inside[b]) {
            int conflict = ranges[a][0] >= 0 && ranges[b][0] >= 0 &&
                (ranges[a][0] > ranges[b][1] || ranges[b][0] > ranges[a][1]);
            if (conflict && total >= 200) {
                mesh_audit_row(flagged, a, b, total, 0, 0);
            }
            size_t smaller = sizes[a] < sizes[b] ? sizes[a] : sizes[b];
            if (!conflict && total >= 1000 && total >= .2 * smaller && agreement >= .95) {
                edges[links++] = best;
            }
        }
        i = end;
    }
    size_t at = 0, anchored = 0;
    size_t hub = m - 1;
    for (size_t i = 0; ok && i < m; i++) {
        group[i] = -1;
        position[i] = at;
        while (at < links && (size_t)edges[at].a == i) {
            at++;
        }
        if (i < hub && argc == 5 && inside[i] && sizes[i] && ranges[i][0] >= 0) {
            anchored++;
        }
    }
    int groups = 0;
    for (size_t step = 0; ok && step < m; step++) {
        size_t a0 = step ? rank[step - 1].unit : hub;
        if ((step ? !inside[a0] : !anchored) || group[a0] >= 0) {
            continue;
        }
        size_t head = 0, tail = 0;
        group[a0] = groups++;
        queue[tail++] = (int32_t)a0;
        if (a0 == hub) {
            for (size_t i = 0; i < hub; i++) {
                if (inside[i] && sizes[i] && ranges[i][0] >= 0) {
                    group[i] = group[hub];
                    ok = mesh_named_shift(w, unit, vertices, i, ranges[i], sizes[i], offset + i);
                    if (!ok) {
                        break;
                    }
                    queue[tail++] = (int32_t)i;
                }
            }
        }
        while (ok && head < tail) {
            int32_t a = queue[head++];
            for (size_t j = position[a]; j < links && edges[j].a == a; j++) {
                int32_t b = edges[j].b;
                int64_t want = offset[a] + edges[j].step;
                if (group[b] < 0) {
                    group[b] = group[a];
                    offset[b] = want;
                    queue[tail++] = b;
                } else if (a < b && offset[b] != want) {
                    mesh_audit_row(conflicts, a, b, offset[b] - want, 0, 0);
                }
            }
        }
    }
    json_object_object_add(audit, "groups", json_object_new_int(groups));
    size_t subset = 0;
    for (size_t i = 0; ok && i < vertices; i++) {
        out_group[i] = group[unit[i]];
        w[i] += offset[unit[i]];
        keep[i] = inside[unit[i]] && !duplicate[i];
        if (mesh_tier(tiers[mesh[i]], argv[3])) {
            mapping[subset] = i;
            search.cells[subset++] = search.cells[i];
        }
    }
    search.count = subset;
    ok = ok && (!subset || index_cells(&search));
    for (size_t i = 0; ok && i < subset; i++) {
        size_t q = mapping[i], near[16];
        const Cell *p = search.cells + i;
        double xyz[] = {p->x, p->y, p->z};
        int hits = control_nearest_xyz(&search, xyz, near, 16, search.bound, -1);
        for (int j = 0; j < hits; j++) {
            size_t k = near[j] - 1, f = mapping[k];
            double depth = mesh_depth(p, search.cells + k, v + 3, q, lateral);
            if (depth <= 2 * tolerance &&
                (out_group[q] == out_group[f] ? fabs(w[q] - w[f]) > .5 : depth > tolerance)) {
                unknown[q] = 1;
                break;
            }
        }
    }
    ok = ok &&
        mesh_save_archive(&file, argv[2], argv[3], n, keep, duplicate, unknown, out_group, w, pairs,
            count, audit);
    if (!ok) {
        fprintf(stderr, "cannot frame reference archive %s\n", argv[1]);
    }
    json_object_put(audit);
    free(search.cells);
    free(search.table);
    free(mapping);
    free(queue);
    free(group);
    free(out_group);
    free(ranges);
    free(offset);
    free(rank);
    free(flags);
    free(inside);
    free(position);
    free(sizes);
    free(edges);
    free(pairs);
    free(unit_mesh);
    free(unit);
    free(mesh);
    free(tiers);
    free(names);
    free(w);
    for (int i = 0; i < 6; i++) {
        free(v[i]);
    }
    npz_close_reader(&file);
    return !ok;
}

#endif
