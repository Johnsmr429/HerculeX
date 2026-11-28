typedef struct {
    int a, b;
    double size, dx, dy;
} SpotEdge;

typedef struct {
    double x, y;
} SpotXY;

static int spot_length_order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double spot_edge_size(const f32 *xyz, int cells, int a, int b) {
    double x = (double)xyz[a] - xyz[b];
    double y = (double)xyz[cells + a] - xyz[cells + b];
    double z = (double)xyz[2 * cells + a] - xyz[2 * cells + b];
    return sqrt(x * x + y * y + z * z);
}

static void spot_stretch(
    Arena *a, json_object *report, const f32 *xyz, int h, int w, double step, const char *suffix) {
    size_t mark = a->used;
    int cells = h * w, count = 0, close = 0;
    double *lengths = push(a, 2 * (size_t)cells * sizeof(*lengths));
    for (int i = 0; i < cells; i++) {
        if (xyz[i] <= 0) {
            continue;
        }
        int next[] = {i + w, i % w + 1 < w ? i + 1 : cells};
        for (int k = 0; k < 2; k++) {
            if (next[k] < cells && xyz[next[k]] > 0) {
                double length = spot_edge_size(xyz, cells, i, next[k]) / step;
                close += fabs(length - 1) <= 0.1;
                lengths[count++] = length;
            }
        }
    }
    need(count > 0, "the %s spot mesh has no connected rows or columns", suffix);
    qsort(lengths, count, sizeof(*lengths), spot_length_order);
    json_object *values = json_object_new_array_ext(3);
    double fractions[] = {0.05, 0.5, 0.95};
    for (int k = 0; k < 3; k++) {
        double at = (count - 1) * fractions[k];
        int i = (int)at;
        double v = lengths[i];
        if (i + 1 < count) {
            v += (at - i) * (lengths[i + 1] - v);
        }
        json_object_array_add(values, jnum(nearbyint(v * 1000) / 1000));
    }
    json_object_object_add(report, fmt(a, "stretch_%s_p5_p50_p95", suffix), values);
    json_object_object_add(report, fmt(a, "within_10pct_%s", suffix),
        jnum(nearbyint((double)close * 1000 / count) / 1000));
    a->used = mark;
}

int spot_reflatten_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh spots.reflatten XYZ.npy STEP OUT.npz REPORT.json [ITERATIONS]");
        return 0;
    }
    need(argc == 4 || argc == 5,
        "usage: spots.reflatten XYZ.npy STEP OUT.npz REPORT.json [ITERATIONS]");
    char *end;
    double step = strtod(argv[1], &end);
    need(end != argv[1] && !*end && step > 0 && isfinite(step), "invalid mesh step");
    long iterations = argc == 5 ? strtol(argv[4], &end, 10) : 3000;
    need(iterations > 0 && (argc == 4 || (end != argv[4] && !*end)), "invalid iterations");
    Arena a = arena(24 * GB);
    Array mesh = npy_read(&a, argv[0]);
    need(mesh.ndim == 3 && mesh.shape[0] == 3 && !strcmp(mesh.descr, "<f4") && mesh.shape[1] > 0 &&
            mesh.shape[2] > 0 && elements(mesh) / 3 <= INT_MAX / 4,
        "the mesh must be a little-endian float32 (3, height, width) array");
    int h = (int)mesh.shape[1], w = (int)mesh.shape[2], cells = h * w;
    const f32 *xyz = mesh.data;
    SpotEdge *edges = push(&a, 4 * (size_t)cells * sizeof(*edges));
    int *degree = pushz(&a, (size_t)cells * sizeof(*degree));
    int count = 0, isolated = 0;
    const int steps[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int k = 0; k < 4; k++) {
        int dy = steps[k][0], dx = steps[k][1];
        for (int y = 0; y < h - dy; y++) {
            for (int x = dx < 0; x < w - (dx > 0); x++) {
                int i = y * w + x, j = i + dy * w + dx;
                if (xyz[i] > 0 && xyz[j] > 0) {
                    edges[count++] =
                        (SpotEdge){.a = i, .b = j, .size = spot_edge_size(xyz, cells, i, j)};
                    degree[i]++;
                    degree[j]++;
                }
            }
        }
    }
    need(count > 0, "the spot mesh has no edges");
    SpotXY *flat = pushz(&a, (size_t)cells * sizeof(*flat));
    SpotXY *pull = push(&a, (size_t)cells * sizeof(*pull));
    double *errors = push(&a, (size_t)count * sizeof(*errors));
    int linked = 0;
    double start_x = 0, start_y = 0;
    for (int i = 0; i < cells; i++) {
        if (degree[i]) {
            int row = i / w, col = i % w;
            flat[i] = (SpotXY){.x = col * step, .y = row * step};
            start_x += flat[i].x;
            start_y += flat[i].y;
            linked++;
        } else {
            isolated += xyz[i] > 0;
        }
    }
    start_x /= linked;
    start_y /= linked;
    json_object *history = json_object_new_array();
    for (long round = 0; round < iterations; round++) {
        int save = round % 500 == 0 || round + 1 == iterations;
        memset(pull, 0, (size_t)cells * sizeof(*pull));
        for (int i = 0; i < count; i++) {
            SpotEdge *e = edges + i;
            double dx = flat[e->a].x - flat[e->b].x, dy = flat[e->a].y - flat[e->b].y;
            double length = fmax(sqrt(dx * dx + dy * dy), 1e-9);
            e->dx = dx / length * e->size;
            e->dy = dy / length * e->size;
            pull[e->a].x += flat[e->b].x + e->dx;
            pull[e->a].y += flat[e->b].y + e->dy;
            if (save) {
                errors[i] = fabs(length / e->size - 1);
            }
        }
        for (int i = 0; i < count; i++) {
            SpotEdge e = edges[i];
            pull[e.b].x += flat[e.a].x - e.dx;
            pull[e.b].y += flat[e.a].y - e.dy;
        }
        for (int i = 0; i < cells; i++) {
            if (degree[i]) {
                flat[i] = (SpotXY){.x = pull[i].x / degree[i], .y = pull[i].y / degree[i]};
            }
        }
        if (save) {
            qsort(errors, count, sizeof(*errors), spot_length_order);
            double median = (errors[(count - 1) / 2] + errors[count / 2]) * 0.5;
            json_object_array_add(history, jnum(nearbyint(10000 * median) / 10000));
        }
    }
    double centre_x = 0, centre_y = 0;
    for (int i = 0; i < cells; i++) {
        centre_x += flat[i].x;
        centre_y += flat[i].y;
    }
    centre_x /= linked;
    centre_y /= linked;
    double trace = 0, skew = 0;
    for (int i = 0; i < cells; i++) {
        if (degree[i]) {
            double x = flat[i].x - centre_x, y = flat[i].y - centre_y;
            int row = i / w, col = i % w;
            double sx = col * step - start_x, sy = row * step - start_y;
            trace += x * sx + y * sy;
            skew += y * sx - x * sy;
        }
    }
    double angle = atan2(skew, trace), c = cos(angle), s = sin(angle);
    double low_x = INFINITY, low_y = INFINITY, high_x = -INFINITY, high_y = -INFINITY;
    for (int i = 0; i < cells; i++) {
        if (degree[i]) {
            double x = flat[i].x - centre_x, y = flat[i].y - centre_y;
            flat[i] = (SpotXY){.x = x * c + y * s + start_x, .y = y * c - x * s + start_y};
            low_x = fmin(low_x, flat[i].x);
            low_y = fmin(low_y, flat[i].y);
            high_x = fmax(high_x, flat[i].x);
            high_y = fmax(high_y, flat[i].y);
        }
    }
    double hh = floor((high_y - low_y) / step) + 1, ww = floor((high_x - low_x) / step) + 1;
    need(hh > 0 && ww > 0 && hh * ww < INT_MAX / 3, "the flattened mesh is too large");
    int out_h = (int)hh, out_w = (int)ww, output_cells = out_h * out_w;
    f32 *output = push(&a, 3 * (size_t)output_cells * sizeof(*output));
    f32 *position = push(&a, 2 * (size_t)cells * sizeof(*position));
    u64 *priority = pushz(&a, (size_t)output_cells * sizeof(*priority));
    for (int i = 0; i < 3 * output_cells; i++) {
        output[i] = -1;
    }
    for (int i = 0; i < cells; i++) {
        if (degree[i]) {
            flat[i].x = (flat[i].x - low_x) / step;
            flat[i].y = (flat[i].y - low_y) / step;
        }
        position[i] = degree[i] ? (f32)flat[i].x : NAN;
        position[cells + i] = degree[i] ? (f32)flat[i].y : NAN;
    }
    for (int i = 0; i < cells - w; i++) {
        if (i % w + 1 == w || !degree[i] || !degree[i + w + 1]) {
            continue;
        }
        for (int k = 0; k < 2; k++) {
            int j = k ? i + w + 1 : i + 1, m = k ? i + w : i + w + 1;
            if (!degree[j] || !degree[m]) {
                continue;
            }
            SpotXY p = flat[i], q = flat[j], r = flat[m];
            SpotXY e = {.x = q.x - p.x, .y = q.y - p.y};
            SpotXY f = {.x = r.x - p.x, .y = r.y - p.y};
            double det = e.x * f.y - e.y * f.x;
            if (det <= 1e-9) {
                continue;
            }
            int x0 = (int)floor(fmin(p.x, fmin(q.x, r.x)));
            int y0 = (int)floor(fmin(p.y, fmin(q.y, r.y)));
            int x1 = (int)fmin(out_w - 1, ceil(fmax(p.x, fmax(q.x, r.x))));
            int y1 = (int)fmin(out_h - 1, ceil(fmax(p.y, fmax(q.y, r.y))));
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    double l1 = ((x - p.x) * f.y - (y - p.y) * f.x) / det;
                    double l2 = (e.x * (y - p.y) - e.y * (x - p.x)) / det;
                    if (l1 < -1e-9 || l2 < -1e-9 || l1 + l2 > 1 + 1e-9) {
                        continue;
                    }
                    int at = y * out_w + x;
                    u64 rank = ((u64)(y - y0) * out_w + (x - x0)) * (2ull * cells) +
                        (u64)k * cells + i + 1;
                    if (rank > priority[at]) {
                        priority[at] = rank;
                        for (int d = 0; d < 3; d++) {
                            const f32 *plane = xyz + d * cells;
                            output[d * output_cells + at] =
                                (f32)((1 - l1 - l2) * plane[i] + l1 * plane[j] + l2 * plane[m]);
                        }
                    }
                }
            }
        }
    }
    int nodes = 0;
    for (int i = 0; i < output_cells; i++) {
        nodes += output[i] > 0;
    }
    json_object *report = json_object_new_object();
    json_object_object_add(report, "iterations", json_object_new_int64(iterations));
    json_object_object_add(report, "isolated_nodes_left_out", json_object_new_int(isolated));
    json_object_object_add(report, "median_edge_error_by_500", history);
    int dimensions[][2] = {{h, w}, {out_h, out_w}};
    const char *name[] = {"before", "after"};
    for (int i = 0; i < 2; i++) {
        json_object *size = json_object_new_array_ext(2);
        json_object_array_add(size, json_object_new_int(dimensions[i][0]));
        json_object_array_add(size, json_object_new_int(dimensions[i][1]));
        json_object_object_add(report, fmt(&a, "grid_%s", name[i]), size);
        spot_stretch(
            &a, report, i ? output : xyz, dimensions[i][0], dimensions[i][1], step, name[i]);
    }
    json_object_object_add(report, "nodes_after", json_object_new_int(nodes));
    Array arrays[] = {
        {.descr = "<f4",
            .ndim = 3,
            .shape = {3, out_h, out_w},
            .data = output,
            .size = 3 * (size_t)output_cells * sizeof(*output)},
        {.descr = "<f4",
            .ndim = 3,
            .shape = {2, h, w},
            .data = position,
            .size = 2 * (size_t)cells * sizeof(*position)},
    };
    npz_write(argv[2], 2, (const char *[]){"grid", "position"}, arrays);
    json_write(argv[3], report);
    json_object_put(report);
    return 0;
}
