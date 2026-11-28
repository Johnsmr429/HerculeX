#define _GNU_SOURCE
#include <errno.h>
#include <json-c/json.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../storage/data_roots.c"
#define HX_MESH_BRIDGE
#define hx_relax_mesh hx_area_relax_mesh
#include "../../sampling/column_sampling.c"
#undef hx_relax_mesh

#ifndef HX_UNITY
const char hx_usage[] = "Readable area of a scan in continuous areas of 4 cm² or more.";
#endif

typedef struct {
    double placed, mesh, bridged;
} Areas;

static int numeric(json_object *o, const char *key, double *v) {
    json_object *value = NULL;
    if (!json_object_object_get_ex(o, key, &value) ||
        !(json_object_is_type(value, json_type_double) ||
            json_object_is_type(value, json_type_int)))
        return 0;
    *v = json_object_get_double(value);
    return 1;
}

static int mesh_areas_of(const char *folder, Areas *areas) {
    char *metadata = path_join(folder, "meta.json");
    json_object *meta = metadata ? json_object_from_file(metadata) : NULL;
    double stride = 0, um = 0;
    int ok = meta && numeric(meta, "stride", &stride) && numeric(meta, "voxel_um", &um);
    free(metadata);
    json_object_put(meta);
    TIFF *header = ok ? open_axis(folder, 0) : NULL;
    uint32_t h = 0, w = 0;
    ok = header && TIFFGetField(header, TIFFTAG_IMAGEWIDTH, &w) &&
        TIFFGetField(header, TIFFTAG_IMAGELENGTH, &h) && w && h && w < INT_MAX - 2 * BRIDGE - 1 &&
        h < INT_MAX - 2 * BRIDGE - 1 &&
        (uint64_t)(w + 2 * BRIDGE + 1) * (h + 2 * BRIDGE + 1) <= SIZE_MAX / (3 * sizeof(double));
    if (header)
        TIFFClose(header);
    size_t n = (size_t)h * w;
    double *xyz = ok ? malloc(3 * n * sizeof(*xyz)) : NULL;
    uint8_t *mask = xyz ? malloc(2 * n) : NULL;
    ok = xyz && mask;
    MeshType type = {0};
    for (int a = 0; a < 3 && ok; a++) {
        MeshType next = {0};
        ok = read_axis(folder, w, h, a, NULL, xyz + a * n, NULL, &next.bits, &next.format);
        type = a ? common_type(type, next) : next;
    }
    size_t mesh = 0, bridged = 0, holes = 0;
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            mask[i] = xyz[i] > 0;
            mesh += mask[i];
        }
        bridged = mesh;
        if (mesh && mesh < n)
            ok = gap_mask(mask, h, w, mask + n, &holes, BRIDGE);
        if (ok && holes) {
            float *out = malloc(3 * n * sizeof(*out));
            ok = out &&
                bridge_grid(xyz, type, h, w, stride, mask, mask + n, holes, out, &default_blur,
                    RELAXATIONS, 3);
            if (ok) {
                bridged = 0;
                for (size_t i = 0; i < n; i++)
                    bridged += out[i] > 0;
            }
            free(out);
        }
    }
    free(mask);
    free(xyz);
    if (ok) {
        double side = stride * um / 1000, cell = side * side;
        areas->mesh = mesh * cell;
        areas->bridged = bridged * cell;
    }
    return ok;
}

typedef struct {
    char *folder;
    size_t group;
    Areas areas;
    int success;
} Surface;

typedef struct {
    Surface *pieces;
    size_t next, count;
} Jobs;

static void *measure(void *context) {
    Jobs *jobs = context;
    for (;;) {
        size_t i = __atomic_fetch_add(&jobs->next, 1, __ATOMIC_RELAXED);
        if (i >= jobs->count)
            return NULL;
        Surface *s = jobs->pieces + i;
        s->success = mesh_areas_of(s->folder, &s->areas);
    }
}

static char *group_key(json_object *group, double shift) {
    const char *id = json_object_to_json_string_ext(group, JSON_C_TO_STRING_PLAIN);
    char number[64];
    if (json_object_is_type(group, json_type_double) || json_object_is_type(group, json_type_int) ||
        json_object_is_type(group, json_type_boolean)) {
        snprintf(number, sizeof(number), "%.17g", json_object_get_double(group));
        id = number;
    }
    char *key = NULL;
    double position = nearbyint(shift);
    if (asprintf(&key, "%s/%.0f", id, position == 0 ? 0 : position) < 0)
        return NULL;
    return key;
}

static int layout(
    const char *base, json_object *rows, Surface *pieces, size_t count, size_t *groups) {
    json_object *index = json_object_new_object();
    int ok = index != NULL;
    for (size_t i = 0; ok && i < count; i++) {
        Surface *p = pieces + i;
        json_object *row = json_object_array_get_idx(rows, i), *group = NULL;
        const char *unit = json_object_get_string(json_object_object_get(row, "unit"));
        json_object *id = json_object_object_get(row, "piece");
        double shift;
        ok = unit && json_object_is_type(id, json_type_int) &&
            numeric(row, "area", &p->areas.placed) && numeric(row, "moved_from_joint_mm", &shift) &&
            isfinite(shift) && json_object_object_get_ex(row, "group", &group) &&
            asprintf(&p->folder, "%s/pieces/%s/piece-%04lld", base, unit,
                (long long)json_object_get_int64(id)) >= 0;
        char *key = ok ? group_key(group, shift) : NULL;
        json_object *known = NULL;
        if (!key) {
            ok = 0;
        } else if (json_object_object_get_ex(index, key, &known)) {
            p->group = (size_t)json_object_get_int64(known);
        } else {
            p->group = (*groups)++;
            ok = json_object_object_add(index, key, json_object_new_int64((int64_t)p->group)) == 0;
        }
        free(key);
    }
    json_object_put(index);
    return ok;
}

static double pairwise(const double *v, size_t n) {
    if (n < 8) {
        double total = -0.0;
        for (size_t i = 0; i < n; i++)
            total += v[i];
        return total;
    }
    if (n <= 128) {
        double a[8];
        memcpy(a, v, sizeof(a));
        size_t i = 8;
        for (; i + 7 < n; i += 8) {
            for (size_t k = 0; k < 8; k++)
                a[k] += v[i + k];
        }
        double sum = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        for (; i < n; i++)
            sum += v[i];
        return sum;
    }
    size_t half = n / 2 / 8 * 8;
    return pairwise(v, half) + pairwise(v + half, n - half);
}

static int totals(const Areas *groups, size_t n, char result[3][384]) {
    const char *names[] = {"locked", "mesh", "bridged"};
    if (!n)
        return 1;
    double *v = malloc(3 * n * sizeof(*v));
    if (!v)
        return 0;
    for (size_t k = 0; k < 3; k++) {
        size_t wide = 0, one = 0;
        double biggest = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            Areas g = groups[i];
            v[i] = (k == 0 ? g.placed : k == 1 ? g.mesh : g.bridged) / 100;
            biggest = fmax(biggest, v[i]);
            if (v[i] >= 4)
                v[n + wide++] = v[i];
            if (v[i] >= 1)
                v[2 * n + one++] = v[i];
        }
        snprintf(result[k], 384,
            "%s total %.0f {'in areas of 4 cm2 or more': %.0f, "
            "'in areas of 1 cm2 or more': %.0f} areas %zu largest %.1f",
            names[k], nearbyint(pairwise(v, n)) + 0., nearbyint(pairwise(v + n, wide)) + 0.,
            nearbyint(pairwise(v + 2 * n, one)) + 0., wide, biggest);
    }
    free(v);
    return 1;
}

static int report_at(const char *base, char result[3][384]) {
    char *path = path_join(base, "sheet/layout.json");
    json_object *doc = path ? json_object_from_file(path) : NULL;
    free(path);
    json_object *rows = json_object_object_get(doc, "pieces");
    if (!json_object_is_type(rows, json_type_array)) {
        json_object_put(doc);
        return 0;
    }
    size_t n = json_object_array_length(rows), ngroups = 0;
    Surface *pieces = calloc(n ? n : 1, sizeof(*pieces));
    int ok = pieces && layout(base, rows, pieces, n, &ngroups);
    json_object_put(doc);
    if (ok && n) {
        Jobs jobs = {.pieces = pieces, .count = n};
        pthread_t threads[3];
        size_t started = 0;
        while (started < 3 && started + 1 < n &&
            pthread_create(threads + started, NULL, measure, &jobs) == 0)
            started++;
        measure(&jobs);
        for (size_t i = 0; i < started; i++)
            pthread_join(threads[i], NULL);
    }
    Areas *sums = ok ? calloc(ngroups ? ngroups : 1, sizeof(*sums)) : NULL;
    ok = sums != NULL;
    for (size_t i = 0; ok && i < n; i++) {
        Surface p = pieces[i];
        if (!p.success) {
            fprintf(stderr, "cannot measure %s\n", p.folder);
            ok = 0;
            break;
        }
        sums[p.group].placed += p.areas.placed;
        sums[p.group].mesh += p.areas.mesh;
        sums[p.group].bridged += p.areas.bridged;
    }
    if (ok)
        ok = totals(sums, ngroups, result);
    free(sums);
    if (pieces) {
        for (size_t i = 0; i < n; i++)
            free(pieces[i].folder);
    }
    free(pieces);
    return ok;
}

int hx_continuous_area_main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ./run.sh calibrate.unwrap.continuous_area SCROLL SCAN\n");
        return 1;
    }
    char *root = local_root(outputs_root), *base = NULL;
    char result[3][384] = {{0}};
    int ok = root && asprintf(&base, "%s/unwrapping/level/%s/%s", root, argv[1], argv[2]) >= 0 &&
        report_at(base, result);
    if (!ok)
        fprintf(stderr, "area measurement failed for %s / %s\n", argv[1], argv[2]);
    for (int k = 0; ok && k < 3 && result[k][0]; k++)
        puts(result[k]);
    free(root);
    free(base);
    return ok ? 0 : 1;
}

#ifndef HX_UNITY
int main(int argc, char **argv) { return hx_continuous_area_main(argc, argv); }
#endif
