#define hx_usage scan_catalog_usage
#define main scan_catalog_command
#define bucket atlas_catalog_origin
#include "../catalog/scan_catalog.c"
#undef bucket
#undef main
#undef hx_usage

#define HX_FETCH_VOLUME_CLI
#define HX_FETCH_VOLUME_LIBRARY
#include "fetch_volume.c"

#include <math.h>

const char hx_usage[] = "Every scroll's coarsest useful level, about 75 µm per voxel.";

static const char *atlas_field(json_object *item, const char *name) {
    return json_object_get_string(json_object_object_get(item, name));
}

static void atlas_one(const char *out, json_object *row, double target) {
    json_object *voxel = json_object_object_get(row, "voxel_um");
    double pitch = json_object_get_double(voxel);
    double rounded = nearbyint(log2(target / pitch));
    int level = rounded > 0 ? (int)rounded : 0;
    const char *name = atlas_field(row, "sample");
    const char *type = atlas_field(row, "type");
    const char *pitch_text = json_object_get_string(voxel);
    char *folder = NULL;
    if (!type) {
        type = "None";
    }
    printf("%s %s %s level %d\n", name, type, pitch_text, level);
    fflush(stdout);
    json_object *products = json_object_object_get(row, "products");
    json_object *roots = json_object_object_get(row, "roots");
    const char *path = atlas_field(products, "ome-zarr"), *root = atlas_field(roots, "ome-zarr");
    int ok = root && asprintf(&folder, "%s/%s-%sum/L%d", out, name, pitch_text, level) >= 0 &&
        fetch_level(path, level, folder, root, 0, -1, 32, 0) == 0;
    if (!ok) {
        printf("failed %s cannot fetch %s level %d\n", name, path, level);
        fflush(stdout);
    }
    free(folder);
}

int main(int argc, char **argv) {
    const char usage[] = "Usage: ./run.sh scans.transfer.fetch_atlas OUT [VOXEL_UM]";
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts(usage);
        return 0;
    }
    if (argc < 2 || argc > 3 || !*argv[1]) {
        puts(usage);
        return 2;
    }
    char *tail = NULL;
    double target = argc == 3 ? strtod(argv[2], &tail) : 75.0;
    if ((tail && (tail == argv[2] || *tail)) || !isfinite(target) || target <= 0) {
        fprintf(stderr, "%s\n", usage);
        return 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        return 1;
    }
    const char *error = NULL;
    json_object *doc = cached_catalog(NULL);
    json_object *rows =
        doc ? catalog_rows(json_object_object_get(doc, "samples"), 0, &error) : NULL;
    if (!rows) {
        fprintf(stderr, "cannot load volume catalogue%s%s\n", error ? " near " : "",
            error ? error : "");
    }
    json_object *selected = json_object_new_array(), *best = NULL;
    const char *active = NULL;
    double score = 0;
    int ok = rows && selected;
    for (size_t i = 0; ok && i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *sample = atlas_field(row, "sample");
        if (active && strcmp(active, sample) != 0) {
            if (best) {
                json_object_get(best);
                ok = json_object_array_add(selected, best) == 0;
                if (!ok) {
                    json_object_put(best);
                    break;
                }
            }
            best = NULL;
        }
        active = sample;
        json_object *products = json_object_object_get(row, "products");
        json_object *voxel = json_object_object_get(row, "voxel_um");
        const char *path = atlas_field(products, "ome-zarr");
        if (!path || !voxel) {
            continue;
        }
        double pitch = json_object_get_double(voxel);
        double level = log2(target / pitch), rank = fabs(log(pitch / 9.0));
        if (!isfinite(rank) || !isfinite(level)) {
            fprintf(stderr, "invalid voxel size or target level for %s\n", sample);
            ok = 0;
            break;
        }
        if (!best || rank < score ||
            (rank == score &&
                strcmp(atlas_field(row, "volume"), atlas_field(best, "volume")) < 0)) {
            best = row;
            score = rank;
        }
    }
    if (ok && best) {
        json_object_get(best);
        ok = json_object_array_add(selected, best) == 0;
        if (!ok) {
            json_object_put(best);
        }
    }
    for (size_t i = 0; ok && i < json_object_array_length(selected); i++) {
        atlas_one(argv[1], json_object_array_get_idx(selected, i), target);
    }
    json_object_put(selected);
    json_object_put(rows);
    json_object_put(doc);
    curl_global_cleanup();
    return !ok;
}
