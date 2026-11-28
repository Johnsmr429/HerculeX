#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(HX_RULER_FIELD) || defined(HX_RULER_CLI)
typedef struct {
    void *nodes;
    double *winding, voxel_um, refused;
    int64_t count;
    int level;
} RulerField;
#endif

#ifndef HX_RULER_FIELD
enum { FEWEST_SHEETS = 8 };

static int ruler_number_order(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double ruler_percentile(const double *sorted, size_t count, double q) {
    double at = q * (count - 1);
    size_t first = (size_t)at, last = first + (at != first);
    double part = at - first, width = sorted[last] - sorted[first];
    return part < .5 ? sorted[first] + part * width : sorted[last] - (1. - part) * width;
}

static size_t ruler_peaks(const double *profile, size_t length, size_t *peaks, double prominence) {
    size_t found = 0;
    for (size_t i = 1, next; i + 1 < length; i = next + 1) {
        double height = profile[i];
        next = i;
        if (!(height > profile[i - 1])) {
            continue;
        }
        while (next + 1 < length && profile[next + 1] == height) {
            next++;
        }
        if (next + 1 == length || !(height > profile[next + 1])) {
            continue;
        }
        double left = height, right = height;
        for (size_t j = i; j > 0 && profile[j - 1] <= height; j--) {
            left = fmin(left, profile[j - 1]);
        }
        for (size_t j = next + 1; j < length && profile[j] <= height; j++) {
            right = fmin(right, profile[j]);
        }
        if (height - fmax(left, right) >= prominence) {
            peaks[found++] = i + (next - i) / 2;
        }
    }
    return found;
}

int hx_ruler_line(const double *profile, size_t length, const double *cross, size_t intersections,
    double out[6]) {
    if (length < FEWEST_SHEETS || intersections < FEWEST_SHEETS) {
        return 0;
    }
    size_t *peaks = malloc(length * sizeof(*peaks));
    double *storage = malloc(length * 2 * sizeof(*storage));
    uint8_t *seen = calloc(length, 1);
    if (!peaks || !storage || !seen) {
        free(seen);
        free(storage);
        free(peaks);
        return -1;
    }
    double *sorted = storage, *winding = storage + length;
    memcpy(sorted, profile, length * sizeof(*sorted));
    qsort(sorted, length, sizeof(*sorted), ruler_number_order);
    double spread = ruler_percentile(sorted, length, .9) - ruler_percentile(sorted, length, .1);
    size_t count = ruler_peaks(profile, length, peaks, .15 * spread);
    int matched = 0;
    if (count >= FEWEST_SHEETS) {
        for (size_t i = 0; i < intersections; i++) {
            double at = cross[2 * i];
            size_t lo = 0, hi = count;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if ((double)peaks[mid] < at) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if (lo && (lo == count || at - peaks[lo - 1] <= peaks[lo] - at)) {
                lo--;
            }
            if (!seen[lo] && fabs((double)peaks[lo] - at) <= 3.) {
                seen[lo] = 1;
                winding[lo] = cross[2 * i + 1];
            }
        }
        size_t first = 0, last = count;
        while (first < count && !seen[first]) {
            first++;
        }
        while (last > first && !seen[last - 1]) {
            last--;
        }
        if (last > first && last - 1 - first >= FEWEST_SHEETS) {
            double span = last - 1 - first, continuous = 0, varied = 0;
            size_t turn = first, previous = first, hits = 0;
            for (size_t i = first; i < last; i++) {
                if (!seen[i]) {
                    continue;
                }
                double change =
                    fabs(winding[i] - winding[first]) + fabs(winding[last - 1] - winding[i]);
                if (change > continuous) {
                    continuous = change;
                    turn = i;
                }
                varied += fabs(winding[i] - winding[previous]);
                previous = i;
                hits++;
            }
            out[0] = span;
            out[1] = continuous / span;
            out[2] = (varied - continuous) / span;
            out[3] = fmin(fabs(winding[turn] - winding[first]),
                         fabs(winding[last - 1] - winding[turn])) >= 3.;
            out[4] = hits / (span + 1.);
            out[5] = (double)intersections / count;
            matched = 1;
        }
    }
    free(seen);
    free(storage);
    free(peaks);
    return matched;
}
#endif

#ifdef HX_RULER_FIELD
int hx_ruler_field(
    const char *sample, const char *scan, const char *unit, int lower, int upper, RulerField *out) {
    char *root = local_root(outputs_root), *path = NULL;
    int found = root &&
        asprintf(&path, "%s/unwrapping/nodes/%s/%s/nodes/%s", root, sample, scan, unit) >= 0;
    if (!found || access(path, R_OK)) {
        free(path);
        path = level_file(sample, scan, "");
        char *alias = NULL;
        found = path && asprintf(&alias, "%s/nodes/%s", path, unit) >= 0;
        free(path);
        path = alias;
    }
    free(root);
    int curl = 0;
    Field field = {0};
    Solution solution = {0};
    int ok = found && !read_curl_value(sample, scan, &curl) && !hx_umbilicus_keep(sample, scan) &&
        !load_nodes(&field, path);
    free(path);
    ok = ok && !field_clip(&field, lower, upper) && !build_edges(&field) && !turn_normals(&field) &&
        !cast_rays(&field) && !solve_field(&field, &solution, curl) &&
        !guard_solution(&field, &solution);
    double *wind = ok ? malloc((size_t)field.nodes_count * sizeof(*wind)) : NULL;
    if (wind) {
        for (int64_t i = 0; i < field.nodes_count; i++) {
            wind[i] = solution.level[i] + curl * (double)field.theta[i] / (2 * M_PI);
        }
        *out = (RulerField){.nodes = field.nodes,
            .winding = wind,
            .voxel_um = field.scan_voxel_um,
            .refused = solution.refused_share,
            .count = field.nodes_count,
            .level = field.level};
        field.nodes = NULL;
    }
    solution_close(&solution);
    field_close(&field);
    return wind != NULL;
}
#endif

#ifdef HX_RULER_CLI

int hx_ruler_field(
    const char *sample, const char *scan, const char *unit, int lower, int upper, RulerField *out);

enum { RULER_LINES = 150, RULER_SEEDS = 600, RULER_GAUSS = 4 };

typedef struct {
    float xyz[3];
    int8_t normal[3];
    uint8_t prominence, width, coherence;
    uint16_t pad;
} RulerNode;

typedef struct {
    double start[3], along[3], across[2][3];
} RulerLine;

typedef struct {
    uint64_t key;
    uint32_t node;
} RulerBin;

typedef struct {
    double at, turn;
    uint32_t node;
} RulerHit;

static RulerNode ruler_node(const RulerField *field, size_t idx) {
    RulerNode p;
    memcpy(&p, (const char *)field->nodes + idx * sizeof(p), sizeof(p));
    return p;
}

static double ruler_dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void ruler_cross(double *out, const double *a, const double *b) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void ruler_unit(double *n) {
    double size = fmax(sqrt(ruler_dot(n, n)), 1e-9);
    for (int i = 0; i < 3; i++) {
        n[i] /= size;
    }
}

static void ruler_seed_nodes(Arena *a, uint32_t count, uint32_t seeds[RULER_SEEDS]) {
    size_t mark = a->used;
    uint32_t *order = push(a, (size_t)count * sizeof(*order));
    for (uint32_t i = 0; i < count; i++) {
        order[i] = i;
    }
    Pcg64 rng = pcg64(7);
    for (uint32_t i = count - 1; i; i--) {
        uint32_t mask = UINT32_MAX >> __builtin_clz(i), choose;
        do {
            choose = pcg64_next32(&rng) & mask;
        } while (choose > i);
        uint32_t swap = order[i];
        order[i] = order[choose];
        order[choose] = swap;
    }
    memcpy(seeds, order, (count < RULER_SEEDS ? count : RULER_SEEDS) * sizeof(*seeds));
    a->used = mark;
}

static void ruler_voxel(const RulerLine *line, int ray, int step, int64_t box[3]) {
    for (int a = 0; a < 3; a++) {
        double shift = ray ? (ray % 2 ? 1 : -1) * line->across[(ray - 1) / 2][a] : 0.;
        box[2 - a] = (int64_t)nearbyint(line->start[a] + shift + step * line->along[a]);
    }
}

static int ruler_lines(Arena *a, const RulerField *field, const Bricks *store, double length,
    RulerLine lines[RULER_LINES]) {
    double low = INFINITY, high = -INFINITY;
    for (int64_t i = 0; i < field->count; i++) {
        float z = ruler_node(field, i).xyz[2];
        low = fmin(low, z);
        high = fmax(high, z);
    }
    low += 4.;
    high -= 4.;
    uint32_t seeds[RULER_SEEDS];
    ruler_seed_nodes(a, (uint32_t)field->count, seeds);
    int available = field->count < RULER_SEEDS ? (int)field->count : RULER_SEEDS;
    int count = 0;
    for (int i = 0; i < available && count < RULER_LINES; i++) {
        RulerNode node = ruler_node(field, seeds[i]);
        RulerLine line = {.along = {node.normal[0], node.normal[1], node.normal[2]}};
        ruler_unit(line.along);
        for (int d = 0; d < 3; d++) {
            line.start[d] = node.xyz[d] - .5 * length * line.along[d];
        }
        double end = line.start[2] + length * line.along[2];
        if (!(line.start[2] >= low && line.start[2] <= high && end >= low && end <= high)) {
            continue;
        }
        double zenith[] = {0, 0, 1}, horizon[] = {1, 0, 0};
        ruler_cross(line.across[0], line.along, fabs(line.along[2]) < .9 ? zenith : horizon);
        ruler_unit(line.across[0]);
        ruler_cross(line.across[1], line.along, line.across[0]);
        int inside = 1;
        for (int ray = 0; inside && ray < 5; ray++) {
            for (int ray_end = 0; inside && ray_end <= 1; ray_end++) {
                int64_t xyz[3];
                ruler_voxel(&line, ray, ray_end * (int)length, xyz);
                for (int d = 0; inside && d < 3; d++) {
                    inside = xyz[d] >= 0 && xyz[d] < store->shape[d];
                }
            }
        }
        if (inside) {
            lines[count++] = line;
        }
    }
    return count;
}

static void ruler_deliver(void *context, Ask *ask, const u8 *body, size_t size) {
    Bricks *store = context;
    need(!body || size == (size_t)store->size, "CT brick %s has invalid size %zu", ask->path, size);
    bricks_put(store, ask->index, body);
}

static void ruler_fetch(
    Arena *a, Bricks *store, const char *url, RulerLine *lines, int count, int steps) {
    size_t mark = a->used;
    uint8_t *seen = pushz(a, store->count);
    for (int k = 0; k < count; k++) {
        for (int ray = 0; ray < 5; ray++) {
            for (int i = 0; i < steps; i++) {
                int64_t zyx[3], index = 0;
                ruler_voxel(lines + k, ray, i, zyx);
                for (int axis = 0; axis < 3; axis++) {
                    index = index * store->grid[axis] + zyx[axis] / store->brick[axis];
                }
                seen[index] = 1;
            }
        }
    }
    size_t n = 0;
    for (int64_t i = 0; i < store->count; i++) {
        n += seen[i] && store->offsets[i] == UNKNOWN;
    }
    Ask *asks = push(a, n * sizeof(*asks));
    for (int64_t i = 0, at = 0; i < store->count; i++) {
        if (!seen[i] || store->offsets[i] != UNKNOWN) {
            continue;
        }
        Ask *ask = asks + at++;
        *ask = (Ask){.index = i, .first = -1};
        int64_t z = i / (store->grid[1] * store->grid[2]), y = i / store->grid[2] % store->grid[1];
        int64_t x = i % store->grid[2];
        snprintf(ask->path, sizeof(ask->path), "%lld/%lld/%lld", (long long)z, (long long)y,
            (long long)x);
    }
    int64_t failed = n ? fetch_ranges(url, asks, n, ruler_deliver, store) : 0;
    if (n) {
        bricks_save(store);
        bricks_map(store);
    }
    need(!failed, "%lld CT bricks failed; rerun to resume", (long long)failed);
    a->used = mark;
}

static int ruler_bin_order(const void *one, const void *two) {
    const RulerBin *a = one, *b = two;
    if (a->key != b->key) {
        return (a->key > b->key) - (a->key < b->key);
    }
    return (a->node > b->node) - (a->node < b->node);
}

static uint64_t ruler_key(int x, int y, int z) {
    const int bias = 1 << 20;
    return (uint64_t)(z + bias) << 42 | (uint64_t)(y + bias) << 21 | (uint64_t)(x + bias);
}

static int ruler_hit_order(const void *one, const void *two) {
    const RulerHit *a = one, *b = two;
    if (a->at != b->at) {
        return (a->at > b->at) - (a->at < b->at);
    }
    return (a->node > b->node) - (a->node < b->node);
}

static size_t ruler_hits(const RulerField *field, const RulerLine *line, double length,
    const RulerBin *bins, uint8_t *seen, int stamp, RulerHit *hits, double *out) {
    size_t count = 0;
    for (int64_t k = 0; 4. * k < length + 4.; k++) {
        double step = 4. * k, p[3];
        int low[3], high[3];
        for (int d = 0; d < 3; d++) {
            p[d] = line->start[d] + step * line->along[d];
            low[d] = (int)floor((p[d] - 6.) / 8.);
            high[d] = (int)floor((p[d] + 6.) / 8.);
        }
        for (int z = low[2]; z <= high[2]; z++) {
            for (int y = low[1]; y <= high[1]; y++) {
                for (int x = low[0]; x <= high[0]; x++) {
                    uint64_t key = ruler_key(x, y, z);
                    size_t lo = 0, hi = (size_t)field->count;
                    while (lo < hi) {
                        size_t mid = lo + (hi - lo) / 2;
                        if (bins[mid].key < key) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    for (size_t i = lo; i < (size_t)field->count && bins[i].key == key; i++) {
                        uint32_t id = bins[i].node;
                        if (seen[id] == stamp) {
                            continue;
                        }
                        RulerNode node = ruler_node(field, id);
                        double near[3] = {
                            node.xyz[0] - p[0], node.xyz[1] - p[1], node.xyz[2] - p[2]};
                        if (ruler_dot(near, near) > 36.) {
                            continue;
                        }
                        seen[id] = stamp;
                        double delta[3] = {node.xyz[0] - line->start[0],
                            node.xyz[1] - line->start[1], node.xyz[2] - line->start[2]};
                        double d = ruler_dot(delta, line->along);
                        for (int axis = 0; axis < 3; axis++) {
                            delta[axis] -= d * line->along[axis];
                        }
                        if (ruler_dot(delta, delta) > 16.) {
                            continue;
                        }
                        double normal[3] = {node.normal[0], node.normal[1], node.normal[2]};
                        ruler_unit(normal);
                        double cut = ruler_dot(normal, line->along);
                        if (fabs(cut) < .7) {
                            continue;
                        }
                        double where = d + ruler_dot(normal, delta) / cut;
                        if (fabs(where - d) <= 6. && where >= 0 && where <= length) {
                            hits[count++] = (RulerHit){where, field->winding[id], id};
                        }
                    }
                }
            }
        }
    }
    qsort(hits, count, sizeof(*hits), ruler_hit_order);
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        const RulerHit *hit = hits + i;
        if (used && hit->at - out[2 * used - 2] <= 2. && fabs(hit->turn - out[2 * used - 1]) < .5) {
            continue;
        }
        out[2 * used] = hit->at;
        out[2 * used + 1] = hit->turn;
        used++;
    }
    return used;
}

static json_object *ruler_rows(Arena *a, const RulerField *field, const Bricks *store,
    RulerLine *lines, int count, double length, json_object *report) {
    size_t n = (size_t)field->count;
    RulerBin *bins = push(a, n * sizeof(*bins));
    for (size_t i = 0; i < n; i++) {
        RulerNode node = ruler_node(field, i);
        int x = (int)floor(node.xyz[0] / 8.), y = (int)floor(node.xyz[1] / 8.);
        int z = (int)floor(node.xyz[2] / 8.);
        bins[i] = (RulerBin){ruler_key(x, y, z), (uint32_t)i};
    }
    qsort(bins, n, sizeof(*bins), ruler_bin_order);
    RulerHit *hits = push(a, n * sizeof(*hits));
    double *cross = push(a, n * 2 * sizeof(*cross));
    uint8_t *seen = pushz(a, n);
    int steps = (int)length + 1;
    double *profile = push(a, (size_t)steps * sizeof(*profile));
    double *smooth = push(a, (size_t)steps * sizeof(*smooth));
    double weight[2 * RULER_GAUSS + 1], total = 0;
    for (int k = -RULER_GAUSS; k <= RULER_GAUSS; k++) {
        weight[k + RULER_GAUSS] = exp(-.5 * k * k);
        total += weight[k + RULER_GAUSS];
    }
    for (int k = 0; k <= 2 * RULER_GAUSS; k++) {
        weight[k] /= total;
    }
    json_object *rows = json_object_new_array();
    double summaries[RULER_LINES][6], values[RULER_LINES];
    int scored = 0, right = 0, lost = 0, noisy = 0, folded = 0;
    const char *keys[] = {"raw_sheets", "levels_per_raw_sheet", "noise_per_raw_sheet", "folded",
        "raw_sheets_with_a_node", "nodes_per_raw_sheet"};
    for (int k = 0; k < count; k++) {
        for (int i = 0; i < steps; i++) {
            double intensity = 0;
            for (int ray = 0; ray < 5; ray++) {
                int64_t v[3];
                ruler_voxel(lines + k, ray, i, v);
                intensity += voxel_at(store, v[0], v[1], v[2]);
            }
            profile[i] = intensity / 5;
        }
        for (int i = 0; i < steps; i++) {
            double value = 0;
            for (int j = -RULER_GAUSS; j <= RULER_GAUSS; j++) {
                int at = i + j;
                at = at < 0 ? -at - 1 : at >= steps ? 2 * steps - at - 1 : at;
                value += profile[at] * weight[j + RULER_GAUSS];
            }
            smooth[i] = value;
        }
        size_t found = ruler_hits(field, lines + k, length, bins, seen, k + 1, hits, cross);
        double *summary = summaries[scored];
        int status = hx_ruler_line(smooth, (size_t)steps, cross, found, summary);
        need(status >= 0, "cannot allocate CT line scoring");
        if (!status) {
            continue;
        }
        json_object *row = json_object_new_object();
        for (int axis = 0; axis < 6; axis++) {
            json_object_object_add(row, keys[axis],
                axis == 3 ? json_object_new_boolean(summary[axis])
                          : json_object_new_double(summary[axis]));
        }
        json_object_array_add(rows, row);
        double gain = summary[1], noise = summary[2];
        right += gain >= .8 && gain <= 1.25 && noise <= .25;
        lost += gain < .5;
        noisy += noise > .25;
        folded += (int)summary[3];
        scored++;
    }
    need(scored > 0, "no lines in the CT slab contain enough sheets");
    json_object_object_add(report, "lines", json_object_new_int(scored));
    const char *labels[] = {"right", "lost", "noisy", "folded"};
    int counts[] = {right, lost, noisy, folded};
    for (int k = 0; k < 4; k++) {
        double share = nearbyint(1000. * counts[k] / scored) / 1000.;
        json_object_object_add(report, labels[k], json_object_new_double(share));
    }
    for (int k = 0; k < scored; k++) {
        values[k] = summaries[k][1];
    }
    qsort(values, scored, sizeof(*values), ruler_number_order);
    json_object *gains = json_object_new_array_ext(5);
    const double fractions[] = {.1, .25, .5, .75, .9};
    for (int i = 0; i < 5; i++) {
        double v = ruler_percentile(values, scored, fractions[i]);
        json_object_array_add(gains, json_object_new_double(nearbyint(v * 1000.) / 1000.));
    }
    json_object_object_add(report, "levels_per_raw_sheet", gains);
    const int columns[] = {0, 4, 5};
    const double unit[] = {3.2, 1., 1.}, scale[] = {100, 1000, 100};
    const char *name[] = {"raw_sheets_per_mm", "raw_sheets_with_a_node", "nodes_per_raw_sheet"};
    for (int axis = 0; axis < 3; axis++) {
        for (int k = 0; k < scored; k++) {
            values[k] = summaries[k][columns[axis]];
        }
        qsort(values, scored, sizeof(*values), ruler_number_order);
        double median = ruler_percentile(values, scored, .5) / unit[axis];
        json_object_object_add(report, name[axis],
            json_object_new_double(nearbyint(median * scale[axis]) / scale[axis]));
    }
    return rows;
}

int hx_raw_ruler_main(int argc, char **argv) {
    if (argc != 7) {
        puts("Usage: ./run.sh calibrate.unwrap.raw_sheet_ruler SCROLL SCAN UNIT PLANE0 PLANE1 "
             "OUT.json");
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    char *end = NULL;
    long from = strtol(argv[4], &end, 10);
    need(!*end && from >= 0 && from <= INT_MAX, "invalid first cell plane %s", argv[4]);
    long to = strtol(argv[5], &end, 10);
    need(!*end && to > from && to <= INT_MAX, "invalid last cell plane %s", argv[5]);
    const char *sample = argv[1], *scan = argv[2], *unit = argv[3];
    RulerField field = {0};
    need(sizeof(RulerNode) == 20 && hx_ruler_field(sample, scan, unit, (int)from, (int)to, &field),
        "cannot solve %s %s %s over cell planes %ld–%ld", sample, scan, unit, from, to);
    double node_um = ldexp(field.voxel_um, field.level), length = 3200. / node_um;
    need(isfinite(length) && length >= 2 * RULER_GAUSS && length < INT_MAX,
        "invalid CT voxel spacing %g", node_um);
    Arena a = arena(16 * GB);
    char *source = NULL;
    char *url = zarr_level(&a, sample, scan, field.level, &source);
    char *folder = fmt(&a, "%s/calibration/raw_sheet_ruler/%s/bricks", LIBRARY, sample);
    int lock = lock_file(fmt(&a, "%s.lock", folder), 1, 1);
    need(lock >= 0, "cannot claim CT cache for %s", sample);
    char *metadata = fmt(&a, "%s/meta.json", folder);
    if (!exists(metadata)) {
        json_object *meta = http_json(fmt(&a, "%s/.zarray", url));
        int64_t shape[3], brick[3];
        for (int i = 0; i < 3; i++) {
            shape[i] = json_object_get_int64(json_object_array_get_idx(jget(meta, "shape"), i));
            brick[i] = json_object_get_int64(json_object_array_get_idx(jget(meta, "chunks"), i));
            need(shape[i] > 0 && brick[i] > 0, "invalid CT array shape or brick size");
        }
        json_object_put(meta);
        bricks_create(&a, folder, shape, brick, source);
    } else {
        json_object *meta = json_read(metadata);
        const char *held = json_object_get_string(json_object_object_get(meta, "source"));
        need(!held || !strcmp(source, held), "%s holds a different scan or level", folder);
        json_object_put(meta);
    }
    Bricks store;
    bricks_open(&a, &store, folder);
    RulerLine lines[RULER_LINES];
    int count = ruler_lines(&a, &field, &store, length, lines);
    ruler_fetch(&a, &store, url, lines, count, (int)length + 1);
    json_object *report = json_object_new_object(), *planes = json_object_new_array_ext(2);
    json_object_array_add(planes, json_object_new_int64(from));
    json_object_array_add(planes, json_object_new_int64(to));
    json_object_object_add(report, "sample", json_object_new_string(sample));
    json_object_object_add(report, "unit", json_object_new_string(unit));
    json_object_object_add(report, "planes", planes);
    json_object_object_add(report, "nodes", json_object_new_int64(field.count));
    json_object_object_add(report, "node_um", jnum(nearbyint(node_um * 1000.) / 1000.));
    json_object_object_add(
        report, "refused_share", jnum(nearbyint(field.refused * 10000.) / 10000.));
    json_object *rows = ruler_rows(&a, &field, &store, lines, count, length, report);
    puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PLAIN));
    json_object_object_add(report, "rows", rows);
    json_write(argv[6], report);
    json_object_put(report);
    if (store.data) {
        munmap((void *)store.data, store.mapped);
    }
    pthread_mutex_destroy(&store.lock);
    close(lock);
    munmap(a.base, a.size);
    free(field.nodes);
    free(field.winding);
    return 0;
}

#endif
