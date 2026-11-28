#include <fnmatch.h>

int hx_synthetic_bench_geometry(
    const char *unit, const char *out, const double *line, size_t knots, float *xyz, size_t count);

static const char *syn_bench_keys[] = {"correct", "wrong", "loose", "purity", "terms",
    "pressed_wrong", "modal_1cm2", "modal_4cm2", "bridges", "edge_depth", "bridge_depth"};

static int syn_score_order(const void *left, const void *right) {
    json_object *const *a = left, *const *b = right;
    double x = jf64(*a, "correct"), y = jf64(*b, "correct");
    return (x > y) - (x < y);
}

static void syn_bench_axis(
    Arena *a, json_object *spec, json_object *band, double **line, size_t *knots) {
    json_object *z = jget(spec, "band_um"), *place = jget(spec, "placement");
    json_object *flows = jget(spec, "flows"), *origin = jget(band, "origin_um");
    double o[3];
    for (int i = 0; i < 3; i++) {
        o[i] = json_object_get_double(json_object_array_get_idx(origin, i));
    }
    double lo = json_object_get_double(json_object_array_get_idx(z, 0));
    double hi = json_object_get_double(json_object_array_get_idx(z, 1));
    double margin = jf64(spec, "margin_um"), delta = jf64(jget(spec, "map"), "knot_um");
    int v = ji64(spec, "version");
    for (size_t i = 0; i < json_object_array_length(flows); i++) {
        json_object *f = json_object_array_get_idx(flows, i);
        const char *kind = json_object_get_string(json_object_object_get(f, "kind"));
        if (!kind || strcmp(kind, "swirl") != 0) {
            double scale = syn_optional(f, "z_scale_um", 4 * jf64(f, "scale_um"));
            delta = fmin(delta, v < 5 ? fmax(150, scale / 4) : fmax(100, scale / 6));
        }
    }
    SynDeform d = {.z0 = lo - margin,
        .dz = delta,
        .mid = (lo + hi) / 2,
        .center = syn_pair(place, "centre_um"),
        .tilt = syn_pair(place, "tilt"),
        .bend = syn_pair(place, "bend_um")};
    *knots = ceil((hi - lo + 2 * margin) / delta) + 2;
    *line = push(a, *knots * 3 * sizeof(**line));
    double voxel = jf64(band, "voxel_um");
    for (size_t k = 0; k < *knots; k++) {
        SynPoint c = syn_center_at(&d, k);
        (*line)[k * 3] = (d.z0 + k * d.dz - o[0]) / voxel;
        (*line)[k * 3 + 1] = (c.x - o[2]) / voxel;
        (*line)[k * 3 + 2] = (c.y - o[1]) / voxel;
    }
}

static json_object *syn_bench_band(Arena *a, const char *folder, json_object *band) {
    const char *truth = fmt(a, "%s/truth.npz", folder), *nodes = fmt(a, "%s/nodes.npz", folder);
    const char *levels = fmt(a, "%s/levels.npz", folder), *edges = fmt(a, "%s/edges.npz", folder);
    Array n = npz_get(a, truth, "node_judged");
    const u8 *judged = n.data, *pressed = npz_get(a, truth, "node_pressed").data;
    const i32 *piece = npz_get(a, truth, "node_piece").data,
              *label = npz_get(a, truth, "node_label").data;
    const i32 *level = npz_get(a, nodes, "node_level").data;
    const float *area = npz_get(a, truth, "node_area").data;
    size_t count = n.shape[0], at = 0;
    int max_piece = 0;
    double correct = 0, missed = 0, loose = 0, total = 0, wrong_pressed = 0, total_pressed = 0;
    SynVote *votes = push(a, count * sizeof(*votes));
    for (size_t i = 0; i < count; i++) {
        if (piece[i] > max_piece) {
            max_piece = piece[i];
        }
        if (judged[i]) {
            total += area[i];
            if (piece[i] >= 0) {
                votes[at++] = (SynVote){
                    .piece = 0, .value = level[i] - label[i], .node = i, .weight = area[i]};
            }
        }
    }
    if (!at || total <= 0) {
        return NULL;
    }
    int mode = 0;
    double weight = 0;
    syn_votes(votes, at, &mode, &weight);
    at = 0;
    for (size_t i = 0; i < count; i++) {
        if (!judged[i]) {
            continue;
        }
        if (piece[i] < 0) {
            loose += area[i];
            continue;
        }
        int offset = level[i] - label[i];
        if (offset == mode) {
            correct += area[i];
        } else {
            missed += area[i];
            wrong_pressed += area[i] * pressed[i];
        }
        total_pressed += area[i] * pressed[i];
        votes[at++] = (SynVote){.piece = piece[i], .value = offset, .node = i, .weight = area[i]};
    }
    int *modal_offset = pushz(a, (size_t)(max_piece + 1) * sizeof(*modal_offset));
    double *modal = pushz(a, (size_t)(max_piece + 1) * sizeof(*modal));
    syn_votes(votes, at, modal_offset, modal);
    double cm1 = 0, cm4 = 0;
    for (int i = 0; i <= max_piece; i++) {
        if (modal[i] >= 100) {
            cm1 += modal[i];
            if (modal[i] >= 400) {
                cm4 += modal[i];
            }
        }
    }
    Array supplied = npz_get(a, truth, "piece_labelled"), term = npz_get(a, truth, "term_judged");
    const double *purity = npz_get(a, truth, "piece_purity").data, *labelled = supplied.data;
    const u8 *term_judged = term.data;
    const i64 *curl = npz_get(a, levels, "chirality").data;
    const i32 *correct_term = npz_get(a, truth, "term_k_true").data;
    const i32 *term_k = npz_get(a, levels, *curl > 0 ? "k_plus" : "k_minus").data;
    double weighted = 0, known = 0;
    for (i64 i = 0; i < supplied.shape[0]; i++) {
        weighted += purity[i] * labelled[i];
        known += labelled[i];
    }
    i64 n_terms = 0, valid_terms = 0;
    for (i64 i = 0; i < term.shape[0]; i++) {
        n_terms += term_judged[i];
        valid_terms += term_judged[i] && term_k[i] == correct_term[i];
    }
    float *xyz = push(a, count * 3 * sizeof(*xyz));
    int needs_edges = !exists(edges);
    double *axis = NULL;
    size_t knots = 0;
    if (needs_edges) {
        json_object *input = json_read(fmt(a, "%s/spec.json", folder));
        if (input) {
            json_object *ready = syn_complete(a, input);
            syn_bench_axis(a, ready, band, &axis, &knots);
            json_object_put(ready);
            json_object_put(input);
        }
    }
    need(!hx_synthetic_bench_geometry(
             fmt(a, "%s/control", folder), needs_edges ? folder : NULL, axis, knots, xyz, count),
        "cannot read synthetic geometry in %s", folder);
    const i8 *normals = npz_get(a, nodes, "normal").data;
    const i32 *sheet = npz_get(a, truth, "node_sheet").data;
    double *normal = push(a, count * 3 * sizeof(*normal));
    for (size_t i = 0; i < count; i++) {
        const i8 *n = normals + 3 * i;
        double length =
            fmax(sqrt((double)n[0] * n[0] + (double)n[1] * n[1] + (double)n[2] * n[2]), 1e-9);
        for (int j = 0; j < 3; j++) {
            normal[3 * i + j] = n[j] / length;
        }
    }
    Array left = npz_get(a, edges, "a");
    const i32 *s = left.data, *t = npz_get(a, edges, "b").data;
    double *distances = push(a, (size_t)left.shape[0] * sizeof(*distances));
    size_t near = 0, bridges = 0;
    for (i64 i = 0; i < left.shape[0]; i++) {
        i32 p = s[i], q = t[i];
        if (sheet[p] < 0 || sheet[q] < 0) {
            continue;
        }
        double n[3], square = 0, d = 0;
        for (int j = 0; j < 3; j++) {
            n[j] = normal[3 * p + j] + normal[3 * q + j];
            square += n[j] * n[j];
        }
        double size = fmax(sqrt(square), 1e-9);
        for (int j = 0; j < 3; j++) {
            d += ((double)xyz[3 * q + j] - xyz[3 * p + j]) * (n[j] / size);
        }
        if (sheet[p] == sheet[q]) {
            distances[near++] = fabs(d);
        } else {
            distances[left.shape[0] - 1 - bridges++] = fabs(d);
        }
    }
    double *far = distances + left.shape[0] - bridges;
    qsort(distances, near, sizeof(*distances), syn_double_order);
    qsort(far, bridges, sizeof(*far), syn_double_order);
    double values[] = {correct / total, missed / total, loose / total,
        weighted / fmax(known, 1e-12), n_terms ? (double)valid_terms / n_terms : NAN,
        wrong_pressed / fmax(total_pressed, 1e-12), cm1 / total, cm4 / total,
        (double)bridges / fmax(bridges + near, 1),
        near ? (distances[near / 2] + distances[(near - 1) / 2]) * 0.5 : NAN,
        bridges ? (far[bridges / 2] + far[(bridges - 1) / 2]) * 0.5 : NAN};
    json_object *row = json_object_new_object();
    for (int i = 0; i < (int)(sizeof(values) / sizeof(*values)); i++) {
        json_object_object_add(row, syn_bench_keys[i], json_object_new_double(values[i]));
    }
    syn_add(row, "judged_cm2", total / 100);
    return row;
}

static int syn_bench_report(int argc, char **argv, const char *dir) {
    const char *usage =
        "Usage: ./run.sh synthetic.three_d.bench [BAND ...] [--family PATTERN] [--json]";
    Arena a = arena(64 * GB);
    const char *root = fmt(&a, "%s/synthetic/%s", OUTPUTS, dir), *family = "*";
    const char **folders = push(&a, (size_t)argc * sizeof(*folders));
    int count = 0, json = 0;
    for (int k = 0; k < argc; k++) {
        if (!strcmp(argv[k], "--help") || !strcmp(argv[k], "-h")) {
            puts(usage);
            return 0;
        }
        if (!strcmp(argv[k], "--json")) {
            json = 1;
        } else if (!strcmp(argv[k], "--family") && k + 1 < argc) {
            family = argv[++k];
        } else {
            need(argv[k][0] != '-', "%s", usage);
            folders[count++] = fmt(&a, "%s/%s", root, argv[k]);
        }
    }
    glob_t paths = {0};
    if (!count) {
        glob(fmt(&a, "%s/*/truth.npz", root), 0, NULL, &paths);
        count = paths.gl_pathc;
        folders = push(&a, (size_t)count * sizeof(*folders));
        for (int k = 0; k < count; k++) {
            const char *p = paths.gl_pathv[k];
            folders[k] = fmt(&a, "%.*s", (int)strlen(p) - 10, p);
        }
    }
    json_object *rows = json_object_new_array(), *groups = json_object_new_object();
    for (int k = 0; k < count; k++) {
        size_t mark = a.used;
        const char *folder = folders[k], *name = strrchr(folder, '/') + 1;
        json_object *band = json_read(fmt(&a, "%s/band.json", folder));
        need(band != NULL, "missing band in %s", folder);
        const char *f = jstr(band, "family");
        if (fnmatch(family, f, 0) == 0) {
            json_object *row = syn_bench_band(&a, folder, band);
            if (row) {
                json_object_object_add(row, "band", json_object_new_string(name));
                json_object_object_add(row, "family", json_object_new_string(f));
                json_object_array_add(rows, row);
                json_object *g = json_object_object_get(groups, f);
                if (!g) {
                    g = json_object_new_array();
                    json_object_object_add(groups, f, g);
                }
                json_object_array_add(g, json_object_get(row));
            }
        }
        json_object_put(band);
        a.used = mark;
    }
    globfree(&paths);
    json_object *summary = json_object_new_object();
    json_object_object_foreach(groups, f, g) {
        int n = json_object_array_length(g);
        json_object *out = json_object_new_object();
        for (int k = 0; k < (int)(sizeof(syn_bench_keys) / sizeof(*syn_bench_keys)); k++) {
            double sum = 0;
            int found = 0;
            for (int i = 0; i < n; i++) {
                double value = jf64(json_object_array_get_idx(g, i), syn_bench_keys[k]);
                if (!isnan(value)) {
                    sum += value;
                    found++;
                }
            }
            json_object_object_add(
                out, syn_bench_keys[k], json_object_new_double(found ? sum / found : NAN));
        }
        json_object_object_add(out, "bands", json_object_new_int(n));
        json_object_object_add(summary, f, out);
    }
    if (json) {
        json_object *out = json_object_new_object();
        json_object_object_add(out, "bands", json_object_get(rows));
        json_object_object_add(out, "families", json_object_get(summary));
        puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PRETTY));
        json_object_put(out);
    } else {
        count = json_object_object_length(summary);
        json_object **sorted = push(&a, (size_t)count * sizeof(*sorted));
        int i = 0;
        json_object_object_foreach(summary, f, row) {
            json_object_object_add(row, "family", json_object_new_string(f));
            sorted[i++] = row;
        }
        for (int k = 1; k < count; k++) {
            json_object *row = sorted[k];
            int j = k;
            while (j > 0 && syn_score_order(&row, &sorted[j - 1]) < 0) {
                sorted[j] = sorted[j - 1];
                j--;
            }
            sorted[j] = row;
        }
        printf("%-24s bands", "family");
        for (int k = 0; k < (int)(sizeof(syn_bench_keys) / sizeof(*syn_bench_keys)); k++) {
            printf(" %12s", syn_bench_keys[k]);
        }
        puts("");
        for (int k = 0; k < count; k++) {
            json_object *row = sorted[k];
            printf("%-24s %5d", jstr(row, "family"), (int)ji64(row, "bands"));
            for (int j = 0; j < (int)(sizeof(syn_bench_keys) / sizeof(*syn_bench_keys)); j++) {
                printf(" %12.4f", jf64(row, syn_bench_keys[j]));
            }
            puts("");
        }
    }
    json_object_put(groups);
    json_object_put(rows);
    json_object_put(summary);
    return 0;
}

static int syn_bench_command(int argc, char **argv) {
    return syn_bench_report(argc, argv, "bands");
}
