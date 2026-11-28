#define _GNU_SOURCE
#include "parallel_blocks.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <glob.h>
#include <json-c/json.h>
#include <limits.h>
#include <math.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>
#include <zip.h>

#define HX_NODE_STORE_LOCAL
#include "../../storage/node_store.c"

#include "../../storage/data_roots.c"

enum { HX_CELL = 8, MIN_PROMINENCE = 6, ATOM_PLANES = 24, RADIAL_RANKS = 1, UNITS = 64, DRAWS = 8 };

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} Node;

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} Edge;

typedef struct {
    int32_t a, b, rank;
} Pair;

typedef struct {
    int64_t lower, upper, count;
    double weight;
} HxJoin;

typedef struct {
    int64_t address, length, fd, offset;
} EdgeRun;

typedef struct {
    double z, x, y;
} HxAxisKnot;

typedef struct {
    Edge *edges;
    int64_t count;
} Run;

typedef struct {
    double floor_mm2, spread, min_nodes, cell, contact_cells, contact_layers;
    int cut;
} GuardSettings;

static const GuardSettings default_guard = {100., .30, 400., HX_CELL, 5., 2., 2};

typedef struct {
    double piece_depth, ray_rim, field_rim, ray_sure, ray_scale, neck, neck_cap;
    double curve, hidden, held, tolerance, quality, patch, draw_bytes;
    int boot;
} FitSettings;

static const FitSettings default_fit = {.piece_depth = 2,
    .ray_rim = .45,
    .field_rim = .6,
    .ray_sure = .75,
    .ray_scale = 1,
    .neck = .1f,
    .neck_cap = 1000,
    .curve = 8,
    .hidden = .5,
    .held = .01,
    .tolerance = 1e-3,
    .quality = .65,
    .patch = 24,
    .draw_bytes = 896LL << 20,
    .boot = 1};

enum {
    RAY_LENGTH,
    RAY_DEGREE,
    RAY_NORMAL,
    RAY_COHERENCE,
    RAY_PROM_A,
    RAY_PROM_B,
    RAY_LATERAL,
    RAY_WIDTH_A,
    RAY_WIDTH_B,
    RAY_FEATURES
};

typedef struct {
    double mean, scale, weight;
    int feature;
} RayTerm;

typedef struct {
    RayTerm *terms;
    int64_t count;
    double intercept;
} RayLogit;

typedef struct {
    RayLogit same, skip;
    double scale;
} RayModel;

typedef struct {
    Node *nodes;
    int64_t nodes_count;
    int32_t *first;
    int64_t cells;
    int grid[3];
    int shape[3];
    int plane0;
    int level;
    double scan_voxel_um;
    char *sample;
    char *volume;
    int64_t row0;
    HxAxisKnot *axis;
    int64_t axis_count;
    int64_t *ids;
    float *theta;
    float *quality;
    Run *runs;
    int run_count;
    Pair *pairs;
    int64_t pair_count;
    double turned_share;
    int64_t orientation_report[3];
    double orientation_ratios[2];
    uint64_t orientation_hash;
    const GuardSettings *guard;
    const RayModel *ray_model;
    const FitSettings *fit;
} Field;

static const FitSettings *fit_settings(const Field *field) {
    return field->fit ? field->fit : &default_fit;
}

typedef struct {
    Edge *attract;
    int64_t edge_count;
    float *rays, *ray_up;
    int32_t *root;
    int32_t *potential;
    int32_t *coarse_root;
    int32_t *level;
    uint8_t *kept;
    int64_t stats[4];
    int32_t patches;
    int64_t terms;
    int64_t energy, cuts;
    double kept_share, firm_share;
    double refused_share, timings[7];
    double guard[40][2];
    int guard_count;
} Solution;

typedef struct {
    int32_t *p, *q, *k, *element, *up, *down;
    int64_t count;
} Terms;

typedef struct {
    int32_t *members;
    int64_t *first;
    int8_t *sign;
    float *u, *v;
    int32_t pieces;
} Flattened;

typedef struct {
    double *grid, *xyz, *normal, *spread, *du, *dv;
    int64_t *cell;
    uint8_t *locked;
    int64_t nodes, height, width, occupied;
} Raster;

typedef struct {
    double *grid;
    uint8_t *known, *kept;
    int64_t height, width;
} Mesh;

void hx_cell_table(const int32_t *cell_of, int64_t count, int32_t *first, int64_t cells);
void hx_node_quality(const Node *nodes, int64_t count, double span, float *out);
int64_t hx_sheet_edges(const Node *nodes, const int32_t *first, int GZ, int GY, int GX, int plane,
    const float *theta, const float *quality, float sigma, float max_depth, float min_cos,
    float max_lateral, Edge *out, int64_t capacity);
void hx_sort_edges(Edge *edges, int64_t count, Edge *scratch);
void hx_forest_init(int64_t count, int32_t *data);
void hx_grow_turns(const Node *nodes, int64_t count, int32_t *data, const Edge *edges, int64_t n);
void hx_labels(int64_t count, int32_t *data, int32_t *root, int32_t *n);
int64_t hx_turn_roots(
    Node *nodes, int64_t count, const int32_t *root, const int32_t *parity, int32_t *balance);
int64_t hx_agreeing_edges(const Edge *edges, int64_t count, const Node *nodes, Edge *out);
int64_t hx_radial_pairs(const Node *nodes, const int32_t *first, int GZ, int GY, int GX, int plane,
    int cell, int plane0, const uint8_t *member, float reach, float tube, float same_sheet,
    int ranks, Pair *out, int64_t capacity);
void hx_centre_at(const HxAxisKnot *knots, int64_t count, double z, double *x, double *y);
void hx_merge_edges(const int64_t *runs, const int64_t *lengths, int count, Edge *out);
int hx_ray_weights(const Node *nodes, int64_t count, const Pair *pairs, size_t rays,
    const float *quality, double rim, double sure, float *out);
int hx_sampled_gap(
    const Node *nodes, int64_t count, const Pair *pairs, size_t rays, double *result);
void hx_ordered_rays(
    const Pair *pairs, const float *weight, const intptr_t *order, int64_t count, Edge *out);
void hx_same_label_wide(const Edge *edges, int64_t count, const int64_t *label, uint8_t *out);
int hx_mutex_sources(const Edge *attract, const uint8_t *use, int64_t n_attract, const Edge *repel,
    int64_t n_repel, int64_t count, int32_t *data, int64_t *stats, int a_fd, int64_t a_offset,
    int b_fd, int64_t b_offset);
int32_t hx_dense_roots(const int32_t *root, int64_t nodes, int32_t *rank, int32_t *label);
int hx_agglomerate(int64_t patches, const int32_t *patch, int64_t nodes, const Edge *edges,
    const uint8_t *use, int64_t edge_count, const Edge *rays, int64_t ray_count, float repel_scale,
    float floor_score, float neck, float neck_cap, int32_t *label, int64_t *stats);
int hx_settled_levels(int64_t count, const Edge *edges, int64_t length, const uint8_t *use,
    const void *labels, int wide, const uint8_t *members, const int32_t *potential, int fd,
    int64_t offset, int32_t *root, int32_t *levels);
int hx_curve_patches(
    const Node *nodes, const int32_t *patch, int32_t n, int32_t count, float size, int32_t *out);
int64_t hx_level_terms(int64_t count, const int32_t *patch, const int32_t *inner, const Edge *edges,
    int64_t edge_count, const Pair *rays, const float *ray_weight, int64_t ray_count,
    const float *theta, int32_t chirality, int32_t *element_term, uint64_t *handle);
void hx_level_terms_take(uint64_t handle, int32_t *p, int32_t *q, int32_t *k);
void hx_level_weights(int64_t terms, const Edge *edges, int64_t edge_count, const float *ray_weight,
    const float *ray_up, int64_t ray_count, float hidden, const int32_t *element_term,
    int32_t scale, int32_t *up, int32_t *down);
int64_t hx_level_minimize(int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const int32_t *up, const int32_t *down, int32_t *x, int32_t first_step,
    int64_t *log);
void hx_patch_start(const int32_t *x, const int32_t *piece, const int32_t *potential,
    const int32_t *patch, const int32_t *inner, int32_t chirality, int64_t nodes, int32_t *start);
int64_t hx_level_votes(int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const Edge *edges, int64_t edge_count, const float *ray_weight,
    const float *ray_up, int64_t ray_count, float hidden, const int32_t *element_term,
    const int32_t *x0, int32_t draws, int32_t threads, int32_t scale, int32_t boot,
    int32_t join_shape, uint64_t seed, int32_t *votes, int64_t *log);
void hx_firm_levels(const int32_t *x, const int32_t *p, const int32_t *q, const int32_t *k,
    int64_t terms, int32_t *votes, int draws, const int32_t *patch, const int32_t *inner,
    int64_t nodes, int chirality, int32_t *level);
void hx_firm_edges(const int32_t *votes, const int32_t *x, const int32_t *patch,
    const int32_t *inner, int chirality, const Edge *edges, int64_t count, const int32_t *term,
    uint8_t *kept, uint8_t *firm);
void hx_node_area(const Node *nodes, int64_t count, float cell_area, float *out);
void hx_order_members(int64_t count, const int32_t *label, int64_t *cursor, int32_t *order);
void hx_selected_piece_edges(const Edge *edges, int64_t length, const int32_t *label,
    const int32_t *potential, int64_t groups, int64_t *cursor, int32_t *ea, int32_t *eb);
void hx_level_flatten(const Node *nodes, const int32_t *order, const int64_t *piece_first,
    int32_t pieces, const int32_t *ea, const int32_t *eb, const int64_t *edge_first, int32_t *local,
    int8_t *sign, float *u, float *v, int32_t steps, double tolerance, double held,
    int32_t threads);
int64_t hx_raster_cells(int64_t height, int64_t width, int64_t nodes, const double *xyz,
    const double *normal, double *u, double *v, double step, double fold, int64_t *flat,
    double *grid, uint8_t *locked, double *spread);
int hx_centre_cells(int64_t height, int64_t width, int64_t nodes, const int64_t *flat,
    const double *du, const double *dv, const double *xyz, const uint8_t *take,
    const uint8_t *source, double step, int32_t slope_nodes, double slope_spread, double slope_low,
    double slope_high, double *grid);
int hx_mesh_isometry(const double *grid, int64_t height, int64_t width, const uint8_t *known,
    const uint8_t *locked, double stride, double *values);
int hx_relax_mesh(void *grid, int as_float, int64_t depth, int64_t height, int64_t width,
    const uint8_t *known, const uint8_t *free_cells, int64_t free_count, int32_t rounds);
int hx_orientation_repair(Node *nodes, int64_t count, const EdgeRun *runs, int64_t run_count,
    uint8_t *keep, int64_t report[3], double ratios[2]);
int hx_spread_piece_labels(
    int64_t *piece, const uint8_t *side, int64_t nodes, const Edge *edges, int64_t edge_count);
int64_t hx_piece_joins(const int64_t *piece, const uint8_t *side, int64_t nodes, const Edge *edges,
    int64_t edge_count, const Pair *pairs, const float *weights, int64_t pair_count, HxJoin *out,
    int64_t capacity);
void dgesv_(const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, double *b,
    const int *ldb, int *info);

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *values = data;
    for (size_t i = 0; i < bytes; i++)
        hash = (hash ^ values[i]) * UINT64_C(1099511628211);
    return hash;
}

static int digest_file(EVP_MD_CTX *context, const char *path) {
    unsigned char block[65536];
    FILE *file = fopen(path, "rb");
    int ok = file != NULL;
    while (ok) {
        size_t count = fread(block, 1, sizeof(block), file);
        if (count)
            ok = EVP_DigestUpdate(context, block, count);
        if (count < sizeof(block)) {
            ok = ok && !ferror(file);
            break;
        }
    }
    if (file)
        fclose(file);
    return ok ? 0 : -1;
}

static int finish_digest(EVP_MD_CTX *context, int ok, char output[13]) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    ok = ok && EVP_DigestFinal_ex(context, digest, &length) && length >= 6;
    EVP_MD_CTX_free(context);
    static const char hex[] = "0123456789abcdef";
    if (!ok)
        return -1;
    for (int i = 0; i < 6; i++) {
        output[2 * i] = hex[digest[i] >> 4];
        output[2 * i + 1] = hex[digest[i] & 15];
    }
    output[12] = 0;
    return 0;
}

static int method_id(char output[13]) {
    static const char *sources[] = {"level/level/run_levels.c", "storage/node_store.c",
        "level/level/piece_kernels.c", "level/level/network_simplex.c",
        "level/level/patch_agglomeration.c", "level/level/level_solver.c",
        "level/level/sheet_flattening.c", "level/level/piece_joiner.c", "level/level/scroll_axis.c",
        "level/level/orientation.c", "level/level/threads_apple/parallel_blocks.h",
        "level/level/threads_cuda/parallel_blocks.h"};
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    int ok = context && EVP_DigestInit_ex(context, EVP_sha256(), NULL);
    for (size_t i = 0; ok && i < sizeof(sources) / sizeof(*sources); i++)
        ok = digest_file(context, sources[i]) == 0;
    return finish_digest(context, ok, output);
}

static int node_rows(const char *folder, glob_t *chunks) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/rows-*", folder) < 0)
        return 0;
    int found = glob(pattern, 0, NULL, chunks) == 0 && chunks->gl_pathc >= 2;
    free(pattern);
    return found;
}

static int stage_method(
    char output[13], const char *tag, const char *node_folder, const char *piece_folder) {
    char code[13];
    if (method_id(code))
        return -1;
    glob_t chunks = {0};
    int found = node_rows(node_folder, &chunks);
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    int ok = found && context && EVP_DigestInit_ex(context, EVP_sha256(), NULL) &&
        EVP_DigestUpdate(context, tag, strlen(tag)) &&
        EVP_DigestUpdate(context, code, strlen(code));
    for (size_t i = 0; ok && i < chunks.gl_pathc; i++) {
        const char *name = strrchr(chunks.gl_pathv[i], '/');
        char *record = NULL;
        ok = asprintf(&record, "%s/%s/record.json", piece_folder,
                 name ? name + 1 : chunks.gl_pathv[i]) >= 0 &&
            digest_file(context, record) == 0;
        free(record);
    }
    globfree(&chunks);
    return finish_digest(context, ok, output);
}

static int record_is_current(const char *folder, const char *method) {
    char *path = NULL;
    if (asprintf(&path, "%s/record.json", folder) < 0)
        return 0;
    json_object *record = json_object_from_file(path), *value;
    free(path);
    int current = record && json_object_object_get_ex(record, "method", &value) &&
        strcmp(json_object_get_string(value), method) == 0;
    if (record)
        json_object_put(record);
    return current;
}

static int remove_path(const char *path) {
    struct stat info;
    if (lstat(path, &info))
        return errno == ENOENT ? 0 : -1;
    if (!S_ISDIR(info.st_mode))
        return unlink(path);
    DIR *folder = opendir(path);
    if (!folder)
        return -1;
    int failed = 0;
    for (struct dirent *entry = readdir(folder); entry; entry = readdir(folder)) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        char *child = NULL;
        if (asprintf(&child, "%s/%s", path, entry->d_name) < 0 || remove_path(child))
            failed = 1;
        free(child);
    }
    if (closedir(folder))
        failed = 1;
    int removed = rmdir(path);
    return failed || removed ? -1 : 0;
}

static int publish_folder(const char *part, const char *out) {
    struct stat info;
    if (lstat(out, &info))
        return errno == ENOENT ? rename(part, out) : -1;
    if (!S_ISDIR(info.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }
#ifdef __APPLE__
    int failed = renameatx_np(AT_FDCWD, part, AT_FDCWD, out, RENAME_SWAP);
#else
    int failed = renameat2(AT_FDCWD, part, AT_FDCWD, out, RENAME_EXCHANGE);
#endif
    return failed || remove_path(part) ? -1 : 0;
}

static char *json_text(json_object *object, const char *name) {
    json_object *value;
    return json_object_object_get_ex(object, name, &value) ? strdup(json_object_get_string(value))
                                                           : NULL;
}

static int json_triplet(json_object *object, const char *name, int *values) {
    json_object *array;
    if (!json_object_object_get_ex(object, name, &array) || json_object_array_length(array) != 3)
        return -1;
    for (int i = 0; i < 3; i++)
        values[i] = json_object_get_int(json_object_array_get_idx(array, (size_t)i));
    return 0;
}

static int load_axis(Field *field) {
    char *outputs = local_root(outputs_root);
    char *axis_path = NULL;
    int made = outputs &&
        asprintf(&axis_path, "%s/unwrapping/level/%s/%s/sheet/axis.json", outputs, field->sample,
            field->volume) >= 0;
    json_object *axis = made ? json_object_from_file(axis_path) : NULL;
    free(outputs);
    free(axis_path);
    if (!axis || !json_object_is_type(axis, json_type_array)) {
        if (axis)
            json_object_put(axis);
        return -1;
    }
    size_t count = json_object_array_length(axis);
    field->axis = calloc(count ? count : 1, sizeof(*field->axis));
    field->axis_count = (int64_t)count;
    double scale = (double)(UINT64_C(1) << field->level);
    for (size_t i = 0; field->axis && i < count; i++) {
        json_object *row = json_object_array_get_idx(axis, i);
        if (!json_object_is_type(row, json_type_array) || json_object_array_length(row) < 3) {
            free(field->axis);
            field->axis = NULL;
            break;
        }
        field->axis[i] =
            (HxAxisKnot){.z = json_object_get_double(json_object_array_get_idx(row, 0)) / scale,
                .x = json_object_get_double(json_object_array_get_idx(row, 1)) / scale,
                .y = json_object_get_double(json_object_array_get_idx(row, 2)) / scale};
    }
    json_object_put(axis);
    return field->axis && count >= 2 ? 0 : -1;
}

static int load_nodes_with_axis(Field *field, const char *unit, int axis) {
    char *record_path = NULL;
    if (asprintf(&record_path, "%s/record.json", unit) < 0)
        return -1;
    json_object *record = json_object_from_file(record_path);
    free(record_path);
    json_object *planes, *value, *rows;
    int valid = record && !json_triplet(record, "grid", field->grid) &&
        !json_triplet(record, "shape", field->shape) &&
        json_object_object_get_ex(record, "cell_planes", &planes) &&
        json_object_array_length(planes) == 2;
    if (!valid) {
        if (record)
            json_object_put(record);
        return -1;
    }
    field->plane0 = json_object_get_int(json_object_array_get_idx(planes, 0));
    field->grid[0] = json_object_get_int(json_object_array_get_idx(planes, 1)) - field->plane0;
    field->level =
        json_object_object_get_ex(record, "level", &value) ? json_object_get_int(value) : 0;
    field->scan_voxel_um = json_object_object_get_ex(record, "scan_voxel_um", &value)
        ? json_object_get_double(value)
        : 0;
    field->sample = json_text(record, "sample");
    field->volume = json_text(record, "volume");
    if (json_object_object_get_ex(record, "rows", &rows) && json_object_array_length(rows))
        field->row0 = json_object_get_int64(json_object_array_get_idx(rows, 0));
    int64_t expected =
        json_object_object_get_ex(record, "nodes", &value) ? json_object_get_int64(value) : -1;
    void *raw = NULL;
    size_t node_count = 0, cell_count = 0;
    int32_t *cell_of = NULL;
    int failed = hx_node_read(unit, &raw, &node_count) ||
        hx_cell_read(unit, &cell_of, &cell_count) || node_count != cell_count ||
        (expected >= 0 && (uint64_t)expected != node_count);
    json_object_put(record);
    if (failed) {
        free(cell_of);
        free(raw);
        return -1;
    }
    Node *input = raw;
    int64_t kept = 0;
    for (size_t i = 0; i < node_count; i++)
        kept += input[i].prominence >= MIN_PROMINENCE;
    field->nodes = malloc((size_t)(kept ? kept : 1) * sizeof(*field->nodes));
    field->ids = malloc((size_t)(kept ? kept : 1) * sizeof(*field->ids));
    int32_t *kept_cells = malloc((size_t)(kept ? kept : 1) * sizeof(*kept_cells));
    int64_t at = 0;
    for (size_t i = 0; field->nodes && field->ids && kept_cells && i < node_count; i++) {
        if (input[i].prominence < MIN_PROMINENCE)
            continue;
        field->nodes[at] = input[i];
        field->ids[at] = (int64_t)(((uint64_t)field->row0 << 32) + i);
        kept_cells[at++] = cell_of[i];
    }
    free(cell_of);
    free(raw);
    if (!field->nodes || !field->ids || !kept_cells) {
        free(kept_cells);
        return -1;
    }
    field->nodes_count = kept;
    field->cells = (int64_t)field->grid[0] * field->grid[1] * field->grid[2];
    if (field->cells < 0 || field->cells > INT32_MAX) {
        free(kept_cells);
        return -1;
    }
    field->first = calloc((size_t)field->cells + 1, sizeof(*field->first));
    if (!field->first) {
        free(kept_cells);
        return -1;
    }
    hx_cell_table(kept_cells, kept, field->first, field->cells);
    free(kept_cells);
    return axis ? load_axis(field) : 0;
}

static int load_nodes(Field *field, const char *unit) {
    return load_nodes_with_axis(field, unit, 1);
}

static int build_edges(Field *field) {
    const FitSettings *fit = fit_settings(field);
    field->theta = malloc((size_t)field->nodes_count * sizeof(*field->theta));
    field->quality = malloc((size_t)field->nodes_count * sizeof(*field->quality));
    if (!field->theta || !field->quality)
        return -1;
    for (int64_t i = 0; i < field->nodes_count; i++) {
        Node *node = &field->nodes[i];
        double cx, cy;
        hx_centre_at(field->axis, field->axis_count, node->z, &cx, &cy);
        double x = node->x - cx, y = node->y - cy;
        if (node->nx * x + node->ny * y < 0) {
            node->nx = -node->nx;
            node->ny = -node->ny;
            node->nz = -node->nz;
        }
        field->theta[i] = (float)atan2(y, x);
    }
    hx_node_quality(field->nodes, field->nodes_count, fit->quality, field->quality);
    field->run_count = (field->grid[0] + ATOM_PLANES - 1) / ATOM_PLANES;
    field->runs = calloc((size_t)field->run_count, sizeof(*field->runs));
    if (!field->runs)
        return -1;
    int layer = field->grid[1] * field->grid[2];
    for (int r = 0; r < field->run_count; r++) {
        int low = r * ATOM_PLANES;
        int high = low + ATOM_PLANES < field->grid[0] ? low + ATOM_PLANES : field->grid[0];
        int64_t capacity =
            13LL * (field->first[(int64_t)high * layer] - field->first[(int64_t)low * layer]);
        Edge *edges = malloc((size_t)(capacity ? capacity : 1) * sizeof(*edges));
        int64_t count = 0;
        for (int z = low; edges && z < high; z++) {
            int32_t begin = field->first[(int64_t)z * layer];
            int32_t end = field->first[(int64_t)(z + 1) * layer];
            if (begin == end)
                continue;
            int64_t found =
                hx_sheet_edges(field->nodes, field->first, field->grid[0], field->grid[1],
                    field->grid[2], z, field->theta, field->quality, 2.0f, fit->piece_depth,
                    0.8660254037844386f, 1.9f * HX_CELL, edges + count, capacity - count);
            if (found < 0) {
                free(edges);
                return -1;
            }
            count += found;
        }
        Edge *scratch = malloc((size_t)(count ? count : 1) * sizeof(*scratch));
        if (!edges || !scratch) {
            free(scratch);
            free(edges);
            return -1;
        }
        hx_sort_edges(edges, count, scratch);
        free(scratch);
        field->runs[r] = (Run){.edges = edges, .count = count};
    }
    return 0;
}

static int turn_normals(Field *field) {
    if (field->nodes_count > INT32_MAX)
        return -1;
    int32_t *forest = malloc((size_t)field->nodes_count * 3 * sizeof(*forest));
    if (!forest)
        return -1;
    hx_forest_init(field->nodes_count, forest);
    for (int i = 0; i < field->run_count; i++)
        hx_grow_turns(
            field->nodes, field->nodes_count, forest, field->runs[i].edges, field->runs[i].count);
    int32_t *root = forest, *parity = forest + field->nodes_count;
    int32_t *balance = forest + 2 * field->nodes_count;
    hx_labels(field->nodes_count, forest, root, parity);
    memset(balance, 0, (size_t)field->nodes_count * sizeof(*balance));
    int64_t changed = hx_turn_roots(field->nodes, field->nodes_count, root, parity, balance);
    field->turned_share = field->nodes_count ? (double)changed / field->nodes_count : NAN;
    free(forest);
    EdgeRun *sources = malloc((size_t)(field->run_count ? field->run_count : 1) * sizeof(*sources));
    uint8_t *repair = malloc((size_t)(field->nodes_count ? field->nodes_count : 1));
    for (int i = 0; sources && i < field->run_count; i++)
        sources[i] = (EdgeRun){.address = (int64_t)(intptr_t)field->runs[i].edges,
            .length = field->runs[i].count,
            .fd = -1};
    if (!sources || !repair ||
        hx_orientation_repair(field->nodes, field->nodes_count, sources, field->run_count, repair,
            field->orientation_report, field->orientation_ratios)) {
        free(repair);
        free(sources);
        return -1;
    }
    field->orientation_hash =
        hash_bytes(UINT64_C(14695981039346656037), repair, (size_t)field->nodes_count);
    for (int64_t i = 0; i < field->nodes_count; i++)
        if (repair[i]) {
            field->nodes[i].nx = -field->nodes[i].nx;
            field->nodes[i].ny = -field->nodes[i].ny;
            field->nodes[i].nz = -field->nodes[i].nz;
        }
    free(repair);
    free(sources);
    for (int r = 0; r < field->run_count; r++) {
        Run *run = &field->runs[r];
        Edge *kept = malloc((size_t)(run->count ? run->count : 1) * sizeof(*kept));
        if (!kept)
            return -1;
        int64_t count = hx_agreeing_edges(run->edges, run->count, field->nodes, kept);
        free(run->edges);
        run->edges = kept;
        run->count = count;
    }
    return 0;
}

static int cast_rays(Field *field) {
    field->pairs =
        malloc((size_t)(field->nodes_count ? field->nodes_count : 1) * sizeof(*field->pairs));
    uint8_t *member = malloc((size_t)(field->nodes_count ? field->nodes_count : 1));
    if (!field->pairs || !member) {
        free(member);
        return -1;
    }
    memset(member, 1, (size_t)field->nodes_count);
    int64_t count = 0;
    int layer = field->grid[1] * field->grid[2];
    for (int z = 0; z < field->grid[0]; z++) {
        int32_t begin = field->first[(int64_t)z * layer];
        int32_t end = field->first[(int64_t)(z + 1) * layer];
        if (begin == end)
            continue;
        int64_t found = hx_radial_pairs(field->nodes, field->first, field->grid[0], field->grid[1],
            field->grid[2], z, HX_CELL, field->plane0, member, 40.0f, 6.0f, 3.0f, RADIAL_RANKS,
            field->pairs + count, field->nodes_count - count);
        if (found < 0) {
            free(member);
            return -1;
        }
        count += found;
    }
    free(member);
    field->pair_count = count;
    return 0;
}

static int field_clip(Field *field, int lower, int upper) {
    int from = lower > field->plane0 ? lower : field->plane0;
    int top = field->plane0 + field->grid[0];
    int to = upper < top ? upper : top;
    if (to <= from) {
        return -1;
    }
    int64_t plane = (int64_t)field->grid[1] * field->grid[2];
    int64_t first = (from - field->plane0) * plane, cells = (to - from) * plane;
    int32_t before = field->first[first], kept = field->first[first + cells] - before;
    memmove(field->nodes, field->nodes + before, (size_t)kept * sizeof(*field->nodes));
    memmove(field->ids, field->ids + before, (size_t)kept * sizeof(*field->ids));
    for (int64_t i = 0; i <= cells; i++) {
        field->first[i] = field->first[first + i] - before;
    }
    field->plane0 = from;
    field->grid[0] = to - from;
    field->cells = cells;
    field->nodes_count = kept;
    return kept ? 0 : -1;
}

static int prepare(Field *field, const char *unit, const FitSettings *fit) {
    memset(field, 0, sizeof(*field));
    field->fit = fit;
    return load_nodes(field, unit) || build_edges(field) || turn_normals(field) || cast_rays(field)
        ? -1
        : 0;
}

static Edge *merged_edges(Field *field, int64_t *count) {
    *count = 0;
    int64_t *addresses = malloc((size_t)field->run_count * sizeof(*addresses));
    int64_t *lengths = malloc((size_t)field->run_count * sizeof(*lengths));
    for (int i = 0; addresses && lengths && i < field->run_count; i++) {
        addresses[i] = (int64_t)(intptr_t)field->runs[i].edges;
        lengths[i] = field->runs[i].count;
        *count += lengths[i];
    }
    Edge *out = addresses && lengths ? malloc((size_t)(*count ? *count : 1) * sizeof(*out)) : NULL;
    if (out)
        hx_merge_edges(addresses, lengths, field->run_count, out);
    free(lengths);
    free(addresses);
    return out;
}

static int double_before(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median_double(double *values, int64_t count) {
    qsort(values, (size_t)count, sizeof(*values), double_before);
    return count & 1 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) * 0.5;
}

static int learned_rays(const Field *field, Solution *solution) {
    int64_t count = field->pair_count;
    if (!count) {
        return 0;
    }
    if (field->nodes_count <= 0) {
        return -1;
    }
    uint32_t *degree = calloc((size_t)field->nodes_count, sizeof(*degree));
    double *length = malloc((size_t)count * sizeof(*length));
    if (!degree || !length) {
        free(length);
        free(degree);
        return -1;
    }
    for (int64_t i = 0; i < solution->edge_count; i++) {
        degree[solution->attract[i].a]++;
        degree[solution->attract[i].b]++;
    }
    for (int64_t i = 0; i < count; i++) {
        Pair p = field->pairs[i];
        Node a = field->nodes[p.a], b = field->nodes[p.b];
        double x = (double)b.x - a.x, y = (double)b.y - a.y, z = (double)b.z - a.z;
        length[i] = sqrt(x * x + y * y + z * z);
    }
    double cut = .6 * median_double(length, count);
    int64_t kept = 0;
    for (int64_t i = 0; i < count; i++) {
        if (length[i] > cut) {
            length[kept++] = length[i];
        }
    }
    double gap = kept ? median_double(length, kept) : 0;
    free(length);
    if (gap <= 0) {
        free(degree);
        return -1;
    }
    const RayModel *model = field->ray_model;
    const RayLogit *logits[] = {&model->same, &model->skip};
    float *output[] = {solution->rays, solution->ray_up};
    for (int64_t i = 0; i < count; i++) {
        Pair p = field->pairs[i];
        Node a = field->nodes[p.a], b = field->nodes[p.b];
        double dx = (double)b.x - a.x, dy = (double)b.y - a.y, dz = (double)b.z - a.z;
        double norm_a = fmax(sqrt(a.nx * a.nx + a.ny * a.ny + a.nz * a.nz), 1e-9);
        double norm_b = fmax(sqrt(b.nx * b.nx + b.ny * b.ny + b.nz * b.nz), 1e-9);
        double ax = a.nx / norm_a, ay = a.ny / norm_a, az = a.nz / norm_a;
        double along = dx * ax + dy * ay + dz * az;
        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        double v[RAY_FEATURES] = {[RAY_LENGTH] = log(fmax(distance / gap, 1e-3)),
            [RAY_DEGREE] = fmin(degree[p.a], degree[p.b]),
            [RAY_NORMAL] = fabs(ax * (b.nx / norm_b) + ay * (b.ny / norm_b) + az * (b.nz / norm_b)),
            [RAY_COHERENCE] = fmin(a.coherence, b.coherence) / 255.,
            [RAY_PROM_A] = a.prominence / 2.,
            [RAY_PROM_B] = b.prominence / 2.,
            [RAY_LATERAL] = sqrt(fmax(distance * distance - along * along, 0.)),
            [RAY_WIDTH_A] = a.width / 4.,
            [RAY_WIDTH_B] = b.width / 4.};
        float q = (float)model->scale * fminf(field->quality[p.a], field->quality[p.b]);
        for (int k = 0; k < 2; k++) {
            double z = logits[k]->intercept;
            for (int64_t j = 0; j < logits[k]->count; j++) {
                RayTerm term = logits[k]->terms[j];
                z += term.weight * (v[term.feature] - term.mean) / term.scale;
            }
            output[k][i] = (1. - 1. / (1. + exp(-z))) * q;
        }
    }
    free(degree);
    return 0;
}

typedef struct {
    float weight;
    intptr_t index;
} RayOrder;

static int ray_before(const void *left, const void *right) {
    const RayOrder *a = left, *b = right;
    if (a->weight != b->weight)
        return a->weight > b->weight ? -1 : 1;
    return (a->index > b->index) - (a->index < b->index);
}

static Edge *ordered_rays(Field *field, float *weight) {
    RayOrder *rows = malloc((size_t)(field->pair_count ? field->pair_count : 1) * sizeof(*rows));
    intptr_t *order = malloc((size_t)(field->pair_count ? field->pair_count : 1) * sizeof(*order));
    Edge *out = malloc((size_t)(field->pair_count ? field->pair_count : 1) * sizeof(*out));
    for (int64_t i = 0; rows && i < field->pair_count; i++)
        rows[i] = (RayOrder){.weight = weight[i], .index = (intptr_t)i};
    if (!rows || !order || !out) {
        free(out);
        free(order);
        free(rows);
        return NULL;
    }
    qsort(rows, (size_t)field->pair_count, sizeof(*rows), ray_before);
    for (int64_t i = 0; i < field->pair_count; i++)
        order[i] = rows[i].index;
    hx_ordered_rays(field->pairs, weight, order, field->pair_count, out);
    free(order);
    free(rows);
    return out;
}

static int dense_labels(const int32_t *root, int64_t count, int32_t **labels, int32_t *groups) {
    int32_t *rank = calloc((size_t)(count ? count : 1), sizeof(*rank));
    *labels = malloc((size_t)(count ? count : 1) * sizeof(**labels));
    if (!rank || !*labels) {
        free(*labels);
        *labels = NULL;
        free(rank);
        return -1;
    }
    *groups = hx_dense_roots(root, count, rank, *labels);
    free(rank);
    return *groups < 0 ? -1 : 0;
}

static int settled(int64_t count, const Edge *edges, int64_t edges_count, const uint8_t *use,
    const void *labels, const uint8_t *members, const int32_t *potential, int32_t *root,
    int32_t *levels) {
    return hx_settled_levels(
        count, edges, edges_count, use, labels, 0, members, potential, -1, 0, root, levels);
}

static void terms_close(Terms *terms) {
    free(terms->down);
    free(terms->up);
    free(terms->element);
    free(terms->k);
    free(terms->q);
    free(terms->p);
    memset(terms, 0, sizeof(*terms));
}

static int terms_open(Terms *terms, int32_t groups, const int32_t *patch, const int32_t *inner,
    const Solution *solution, Field *field, int chirality) {
    memset(terms, 0, sizeof(*terms));
    const Edge *edges = solution->attract;
    int64_t edge_count = solution->edge_count;
    int64_t elements = edge_count + field->pair_count;
    terms->element = malloc((size_t)(elements ? elements : 1) * sizeof(*terms->element));
    uint64_t handle = 0;
    terms->count = terms->element
        ? hx_level_terms(groups, patch, inner, edges, edge_count, field->pairs, solution->rays,
              field->pair_count, field->theta, chirality, terms->element, &handle)
        : -1;
    if (terms->count < 0)
        goto fail;
    size_t count = (size_t)(terms->count ? terms->count : 1);
    terms->p = malloc(count * sizeof(*terms->p));
    terms->q = malloc(count * sizeof(*terms->q));
    terms->k = malloc(count * sizeof(*terms->k));
    terms->up = malloc(count * sizeof(*terms->up));
    terms->down = malloc(count * sizeof(*terms->down));
    if (!terms->p || !terms->q || !terms->k || !terms->up || !terms->down)
        goto fail;
    hx_level_terms_take(handle, terms->p, terms->q, terms->k);
    hx_level_weights(terms->count, edges, edge_count, solution->rays, solution->ray_up,
        field->pair_count, fit_settings(field)->hidden, terms->element, UNITS, terms->up,
        terms->down);
    return 0;
fail:
    if (handle && terms->count >= 0) {
        int32_t *empty = malloc((size_t)(terms->count ? terms->count : 1) * sizeof(*empty));
        if (empty) {
            hx_level_terms_take(handle, empty, empty, empty);
            free(empty);
        }
    }
    terms_close(terms);
    return -1;
}

static int quick_pieces(Field *field, Solution *solution, int32_t **coarse_root,
    int32_t **coarse_potential, int32_t **patch, int32_t **inner, int chirality) {
    const FitSettings *fit = fit_settings(field);
    int32_t *patch_root = NULL, *rank = NULL, *patch_label = NULL, *node_label = NULL;
    solution->attract = merged_edges(field, &solution->edge_count);
    size_t rays = field->pair_count ? (size_t)field->pair_count : 1;
    float *weights = malloc(rays * (field->ray_model ? 2 : 1) * sizeof(*weights));
    int ray_status = -1;
    if (field->ray_model) {
        solution->rays = weights;
        solution->ray_up = weights ? weights + rays : NULL;
        if (weights && solution->attract) {
            ray_status = learned_rays(field, solution);
        }
    } else if (weights) {
        ray_status = hx_ray_weights(field->nodes, field->nodes_count, field->pairs,
            (size_t)field->pair_count, field->quality, fit->ray_rim, fit->ray_sure, weights);
    }
    Edge *repel = ray_status ? NULL : ordered_rays(field, weights);
    if (!field->ray_model) {
        free(weights);
    }
    uint8_t *inside = malloc((size_t)(solution->edge_count ? solution->edge_count : 1));
    int32_t *forest =
        malloc((size_t)(field->nodes_count ? field->nodes_count : 1) * 3 * sizeof(*forest));
    if (!solution->attract || !repel || !inside || !forest) {
        goto fail;
    }
    if (fit->patch == default_fit.patch) {
        int64_t *block = malloc((size_t)field->nodes_count * sizeof(*block));
        if (!block) {
            goto fail;
        }
        for (int64_t i = 0; i < field->nodes_count; i++) {
            Node node = field->nodes[i];
            int64_t z = (int64_t)floor(node.z / 24.0f);
            int64_t y = (int64_t)floor(node.y / 24.0f);
            int64_t x = (int64_t)floor(node.x / 24.0f);
            block[i] =
                (int64_t)(((uint64_t)z * (UINT64_C(1) << 20) + (uint64_t)y) * (UINT64_C(1) << 20) +
                    (uint64_t)x);
        }
        hx_same_label_wide(solution->attract, solution->edge_count, block, inside);
        free(block);
    } else {
        double width = fit->patch;
        for (int64_t i = 0; i < solution->edge_count; i++) {
            Edge edge = solution->attract[i];
            const Node *a = field->nodes + edge.a, *b = field->nodes + edge.b;
            inside[i] = floor(a->x / width) == floor(b->x / width) &&
                floor(a->y / width) == floor(b->y / width) &&
                floor(a->z / width) == floor(b->z / width);
        }
    }
    hx_forest_init(field->nodes_count, forest);
    int64_t forest_stats[4] = {0};
    if (hx_mutex_sources(solution->attract, inside, solution->edge_count, repel, field->pair_count,
            field->nodes_count, forest, forest_stats, -1, 0, -1, 0))
        goto fail;
    free(inside);
    inside = NULL;
    patch_root = malloc((size_t)field->nodes_count * sizeof(*patch_root));
    *inner = malloc((size_t)field->nodes_count * sizeof(**inner));
    *patch = malloc((size_t)field->nodes_count * sizeof(**patch));
    rank = calloc((size_t)field->nodes_count, sizeof(*rank));
    if (!patch_root || !*inner || !*patch || !rank)
        goto fail;
    hx_labels(field->nodes_count, forest, patch_root, *inner);
    solution->patches = hx_dense_roots(patch_root, field->nodes_count, rank, *patch);
    free(rank);
    rank = NULL;
    free(patch_root);
    patch_root = NULL;
    free(forest);
    forest = NULL;
    if (solution->patches < 0)
        goto fail;
    patch_label = calloc((size_t)(solution->patches ? solution->patches : 1), sizeof(*patch_label));
    if (!patch_label ||
        hx_agglomerate(solution->patches, *patch, field->nodes_count, solution->attract, NULL,
            solution->edge_count, repel, field->pair_count, fit->ray_scale, 0.0f, fit->neck,
            fit->neck_cap, patch_label, solution->stats)) {
        goto fail;
    }
    node_label = malloc((size_t)field->nodes_count * sizeof(*node_label));
    *coarse_root = malloc((size_t)field->nodes_count * sizeof(**coarse_root));
    *coarse_potential = malloc((size_t)field->nodes_count * sizeof(**coarse_potential));
    for (int64_t i = 0; node_label && i < field->nodes_count; i++)
        node_label[i] = patch_label[(*patch)[i]];
    free(patch_label);
    patch_label = NULL;
    if (!node_label || !*coarse_root || !*coarse_potential ||
        settled(field->nodes_count, solution->attract, solution->edge_count, NULL, node_label, NULL,
            NULL, *coarse_root, *coarse_potential)) {
        goto fail;
    }
    free(node_label);
    node_label = NULL;
    free(repel);
    (void)chirality;
    return 0;
fail:
    free(node_label);
    free(patch_label);
    free(rank);
    free(patch_root);
    free(forest);
    free(inside);
    free(repel);
    return -1;
}

static double level_clock(void);

static int level_draw_threads(double budget, int64_t terms, int64_t patches) {
    int64_t bytes = 20 * terms + 56 * patches;
    if (bytes <= 0)
        return DRAWS;
    int normal = (int)fmax(1., fmin(DRAWS, budget / bytes));
    const char *extra_text = getenv("HERCULEX_LEVEL_DRAW_EXTRA");
    const char *cpus_text = getenv("HERCULEX_LEVEL_DRAW_CPUS");
    if (!extra_text || !cpus_text)
        return normal;
    char *extra_end, *cpus_end;
    errno = 0;
    long long extra = strtoll(extra_text, &extra_end, 10);
    long cpus = strtol(cpus_text, &cpus_end, 10);
    if (errno || *extra_end || *cpus_end || extra <= 0 || cpus <= normal)
        return normal;
    int limit = cpus < DRAWS ? (int)cpus : DRAWS;
    int workers = (int)fmin(limit, (budget + extra) / bytes);
    return workers > normal ? workers : normal;
}

static int solve_field(Field *field, Solution *solution, int chirality) {
    const FitSettings *fit = fit_settings(field);
    memset(solution, 0, sizeof(*solution));
    double began = level_clock();
    int32_t *coarse_root = NULL, *coarse_potential = NULL, *patch = NULL, *inner = NULL;
    if (quick_pieces(field, solution, &coarse_root, &coarse_potential, &patch, &inner, chirality))
        goto fail;
    solution->timings[0] = level_clock() - began;
    began += solution->timings[0];
    if (!field->ray_model) {
        solution->rays =
            malloc((size_t)(field->pair_count ? field->pair_count : 1) * sizeof(*solution->rays));
        if (!solution->rays ||
            hx_ray_weights(field->nodes, field->nodes_count, field->pairs,
                (size_t)field->pair_count, field->quality, fit->field_rim, fit->ray_sure,
                solution->rays)) {
            goto fail;
        }
    }
    int32_t *curve = malloc((size_t)field->nodes_count * sizeof(*curve));
    if (!curve ||
        hx_curve_patches(field->nodes, patch, (int32_t)field->nodes_count, solution->patches,
            fit->curve, curve)) {
        free(curve);
        goto fail;
    }
    free(patch);
    patch = curve;
    int32_t *piece = NULL, piece_count = 0;
    if (dense_labels(coarse_root, field->nodes_count, &piece, &piece_count))
        goto fail;
    Terms coarse;
    if (terms_open(&coarse, piece_count, piece, coarse_potential, solution, field, chirality)) {
        free(piece);
        goto fail;
    }
    int32_t *coarse_x = calloc((size_t)(piece_count ? piece_count : 1), sizeof(*coarse_x));
    int64_t log[3] = {0};
    if (!coarse_x ||
        hx_level_minimize(piece_count, coarse.count, coarse.p, coarse.q, coarse.k, coarse.up,
            coarse.down, coarse_x, 32, log) < 0) {
        free(coarse_x);
        terms_close(&coarse);
        free(piece);
        goto fail;
    }
    int32_t *start = calloc((size_t)(solution->patches ? solution->patches : 1), sizeof(*start));
    if (start)
        hx_patch_start(
            coarse_x, piece, coarse_potential, patch, inner, chirality, field->nodes_count, start);
    free(coarse_x);
    terms_close(&coarse);
    free(piece);
    if (!start)
        goto fail;
    solution->timings[1] = level_clock() - began;
    began += solution->timings[1];
    Terms exact;
    if (terms_open(&exact, solution->patches, patch, inner, solution, field, chirality)) {
        free(start);
        goto fail;
    }
    solution->timings[2] = level_clock() - began;
    began += solution->timings[2];
    solution->terms = exact.count;
    int64_t solve_log[3] = {0};
    if (hx_level_minimize(solution->patches, exact.count, exact.p, exact.q, exact.k, exact.up,
            exact.down, start, 1, solve_log) < 0) {
        terms_close(&exact);
        free(start);
        goto fail;
    }
    solution->timings[3] = level_clock() - began;
    began += solution->timings[3];
    int32_t *votes = calloc((size_t)(exact.count ? exact.count : 1), sizeof(*votes));
    int64_t vote_log = 0;
    int threads = level_draw_threads(fit->draw_bytes, exact.count, solution->patches);
    if (!votes ||
        hx_level_votes(solution->patches, exact.count, exact.p, exact.q, exact.k, solution->attract,
            solution->edge_count, solution->rays, solution->ray_up, field->pair_count, fit->hidden,
            exact.element, start, DRAWS, threads, UNITS, fit->boot, 0, 1, votes, &vote_log)) {
        free(votes);
        terms_close(&exact);
        free(start);
        goto fail;
    }
    solution->level = malloc((size_t)field->nodes_count * sizeof(*solution->level));
    solution->kept = malloc((size_t)solution->edge_count);
    uint8_t *firm = malloc((size_t)solution->edge_count);
    if (!solution->level || !solution->kept || !firm) {
        free(firm);
        free(votes);
        terms_close(&exact);
        free(start);
        goto fail;
    }
    hx_firm_levels(start, exact.p, exact.q, exact.k, exact.count, votes, DRAWS, patch, inner,
        field->nodes_count, chirality, solution->level);
    hx_firm_edges(votes, start, patch, inner, chirality, solution->attract, solution->edge_count,
        exact.element, solution->kept, firm);
    int64_t kept_count = 0, firm_count = 0;
    for (int64_t i = 0; i < solution->edge_count; i++) {
        kept_count += solution->kept[i] != 0;
        firm_count += firm[i] != 0;
    }
    solution->energy = solve_log[2];
    solution->cuts = solve_log[0];
    solution->kept_share = solution->edge_count ? (double)kept_count / solution->edge_count : NAN;
    solution->firm_share = solution->edge_count ? (double)firm_count / solution->edge_count : NAN;
    free(votes);
    terms_close(&exact);
    free(start);
    solution->timings[4] = level_clock() - began;
    began += solution->timings[4];
    solution->root = malloc((size_t)field->nodes_count * sizeof(*solution->root));
    solution->potential = malloc((size_t)field->nodes_count * sizeof(*solution->potential));
    if (!solution->root || !solution->potential ||
        settled(field->nodes_count, solution->attract, solution->edge_count, firm, NULL, NULL, NULL,
            solution->root, solution->potential)) {
        free(firm);
        goto fail;
    }
    free(firm);
    float *area_node = malloc((size_t)field->nodes_count * sizeof(*area_node));
    const GuardSettings *guard = field->guard ? field->guard : &default_guard;
    double side = guard->cell * (UINT64_C(1) << field->level) * field->scan_voxel_um / 1000;
    if (!area_node)
        goto fail;
    hx_node_area(field->nodes, field->nodes_count, (float)(side * side), area_node);
    double *area = calloc((size_t)field->nodes_count, sizeof(*area));
    uint8_t *large = malloc((size_t)field->nodes_count);
    int32_t *groups = malloc((size_t)field->nodes_count * sizeof(*groups));
    for (int64_t i = 0; area && i < field->nodes_count; i++)
        area[solution->root[i]] += area_node[i];
    for (int64_t i = 0; area && large && groups && i < field->nodes_count; i++) {
        large[i] = area[solution->root[i]] >= guard->floor_mm2;
        groups[i] = large[i] ? solution->root[i] : -1;
    }
    free(area);
    free(area_node);
    int32_t *rest_root = malloc((size_t)field->nodes_count * sizeof(*rest_root));
    int32_t *rest_potential = malloc((size_t)field->nodes_count * sizeof(*rest_potential));
    if (!large || !groups || !rest_root || !rest_potential ||
        settled(field->nodes_count, solution->attract, solution->edge_count, solution->kept, groups,
            NULL, NULL, rest_root, rest_potential)) {
        free(rest_potential);
        free(rest_root);
        free(groups);
        free(large);
        goto fail;
    }
    for (int64_t i = 0; i < field->nodes_count; i++) {
        if (large[i]) {
            rest_root[i] = solution->root[i];
            rest_potential[i] = solution->potential[i];
        }
    }
    free(groups);
    free(large);
    free(solution->potential);
    free(solution->root);
    solution->root = rest_root;
    solution->potential = rest_potential;
    solution->coarse_root = coarse_root;
    solution->timings[5] = level_clock() - began;
    coarse_root = NULL;
    free(inner);
    free(patch);
    free(coarse_potential);
    free(coarse_root);
    return 0;
fail:
    free(inner);
    free(patch);
    free(coarse_potential);
    free(coarse_root);
    return -1;
}

static void solution_close(Solution *solution) {
    free(solution->kept);
    free(solution->level);
    free(solution->potential);
    free(solution->root);
    free(solution->coarse_root);
    free(solution->rays);
    free(solution->attract);
    memset(solution, 0, sizeof(*solution));
}

typedef struct {
    int32_t root, size;
} PieceRank;

static int piece_before(const void *left, const void *right) {
    const PieceRank *a = left, *b = right;
    if (a->size != b->size)
        return a->size > b->size ? -1 : 1;
    return (a->root > b->root) - (a->root < b->root);
}

static void flattened_close(Flattened *flat) {
    free(flat->v);
    free(flat->u);
    free(flat->sign);
    free(flat->first);
    free(flat->members);
    memset(flat, 0, sizeof(*flat));
}

static int piece_members(const Solution *solution, int64_t count, double minimum, Flattened *flat,
    int32_t *label, int64_t *cursor) {
    int32_t *sizes = calloc((size_t)(count ? count : 1), sizeof(*sizes));
    PieceRank *ranks = malloc((size_t)(count ? count : 1) * sizeof(*ranks));
    int failed = !sizes || !ranks || !label || !cursor;
    for (int64_t i = 0; !failed && i < count; i++) {
        sizes[solution->root[i]]++;
    }
    for (int64_t i = 0; !failed && i < count; i++) {
        if (sizes[i] >= minimum) {
            ranks[flat->pieces++] = (PieceRank){.root = (int32_t)i, .size = sizes[i]};
        }
        sizes[i] = -1;
    }
    int32_t pieces = flat->pieces;
    if (!failed) {
        qsort(ranks, (size_t)pieces, sizeof(*ranks), piece_before);
        flat->first = calloc((size_t)pieces + 1, sizeof(*flat->first));
        failed = !flat->first;
    }
    for (int32_t i = 0; !failed && i < pieces; i++) {
        sizes[ranks[i].root] = i;
        flat->first[i + 1] = flat->first[i] + ranks[i].size;
    }
    for (int64_t i = 0; !failed && i < count; i++) {
        label[i] = sizes[solution->root[i]];
    }
    if (!failed) {
        size_t members = flat->first[pieces] ? (size_t)flat->first[pieces] : 1;
        flat->members = malloc(members * sizeof(*flat->members));
        failed = !flat->members;
    }
    if (!failed) {
        memcpy(cursor, flat->first, (size_t)pieces * sizeof(*cursor));
        hx_order_members(count, label, cursor, flat->members);
    }
    free(ranks);
    free(sizes);
    return failed ? -1 : 0;
}

static int flatten_solution(Field *field, Solution *solution, Flattened *flat) {
    memset(flat, 0, sizeof(*flat));
    int64_t count = field->nodes_count;
    size_t bytes = (size_t)(count ? count : 1);
    int32_t *label = malloc(bytes * sizeof(*label));
    double minimum = field->guard ? field->guard->min_nodes : default_guard.min_nodes;
    size_t limit = minimum > count ? 0 : bytes / (size_t)ceil(minimum);
    int64_t *cursor = malloc((limit + 1) * sizeof(*cursor)), *edge_first = NULL;
    int32_t *ea = NULL, *eb = NULL, *local = NULL;
    if (piece_members(solution, count, minimum, flat, label, cursor)) {
        goto fail;
    }
    int32_t pieces = flat->pieces;
    edge_first = calloc((size_t)pieces + 1, sizeof(*edge_first));
    if (!edge_first)
        goto fail;
    hx_selected_piece_edges(solution->attract, solution->edge_count, label, solution->potential,
        pieces, edge_first + 1, NULL, NULL);
    for (int32_t i = 0; i < pieces; i++)
        edge_first[i + 1] += edge_first[i];
    int64_t edge_count = edge_first[pieces];
    ea = malloc((size_t)(edge_count ? edge_count : 1) * sizeof(*ea));
    eb = malloc((size_t)(edge_count ? edge_count : 1) * sizeof(*eb));
    if (!ea || !eb)
        goto fail;
    memcpy(cursor, edge_first, (size_t)pieces * sizeof(*cursor));
    hx_selected_piece_edges(solution->attract, solution->edge_count, label, solution->potential,
        pieces, cursor, ea, eb);
    local = malloc((size_t)(count ? count : 1) * sizeof(*local));
    flat->sign = calloc((size_t)(count ? count : 1), sizeof(*flat->sign));
    flat->u = calloc((size_t)(count ? count : 1), sizeof(*flat->u));
    flat->v = calloc((size_t)(count ? count : 1), sizeof(*flat->v));
    if (!local || !flat->sign || !flat->u || !flat->v)
        goto fail;
    const FitSettings *fit = fit_settings(field);
    hx_level_flatten(field->nodes, flat->members, flat->first, pieces, ea, eb, edge_first, local,
        flat->sign, flat->u, flat->v, 300, fit->tolerance, fit->held,
        (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
    free(local);
    free(eb);
    free(ea);
    free(edge_first);
    free(cursor);
    free(label);
    return 0;
fail:
    free(local);
    free(eb);
    free(ea);
    free(edge_first);
    free(cursor);
    free(label);
    flattened_close(flat);
    return -1;
}

static int solve_three(double matrix[3][3], double value[3], double out[3]) {
    int n = 3, nrhs = 1, pivots[3], info;
    dgesv_(&n, &nrhs, &matrix[0][0], &n, pivots, value, &n, &info);
    memcpy(out, value, sizeof(value[0]) * 3);
    return info ? -1 : 0;
}

#ifdef HX_UNITY
void hx_level_samples(uint32_t count, uint32_t size, uint32_t *out);
#endif

static int median_direction(Field *field, Solution *solution, Flattened *flat, int64_t begin,
    int64_t end, int roll, double *direction) {
    int64_t count = end - begin;
    if (count < 24)
        return 0;
    int samples = count < 2000 ? (int)count : 2000;
    double angles[2000];
#ifdef HX_UNITY
    uint32_t centres[2000];
    hx_level_samples((uint32_t)count, (uint32_t)samples, centres);
#endif
    int steep = 0;
    for (int sample = 0; sample < samples; sample++) {
#ifdef HX_UNITY
        int64_t local = centres[sample];
#else
        int64_t local = count <= 2000 ? sample : (int64_t)sample * count / samples;
#endif
        int32_t centre = flat->members[begin + local];
        double distance[24];
        int32_t nearest[24];
        int found = 0;
        for (int64_t at = begin; at < end; at++) {
            int32_t node = flat->members[at];
            double du = (double)flat->u[node] - flat->u[centre];
            double dv = (double)flat->v[node] - flat->v[centre];
            double d = du * du + dv * dv;
            int position;
            if (found < 24) {
                position = found++;
            } else {
                if (d >= distance[23])
                    continue;
                position = 23;
            }
            while (position > 0 && d < distance[position - 1]) {
                if (position < 24) {
                    distance[position] = distance[position - 1];
                    nearest[position] = nearest[position - 1];
                }
                position--;
            }
            distance[position] = d;
            nearest[position] = node;
        }
        double matrix[3][3] = {{0}};
        double target[3] = {0};
        double centre_value;
        if (!roll) {
            centre_value = field->nodes[centre].z;
        } else {
            double cx, cy;
            hx_centre_at(field->axis, field->axis_count, field->nodes[centre].z, &cx, &cy);
            double radius = hypot(field->nodes[centre].x - cx, field->nodes[centre].y - cy);
            double turn =
                solution->potential[centre] + (double)(field->theta[centre] / (float)(2 * M_PI));
            centre_value = 2 * M_PI * turn * radius;
        }
        for (int k = 0; k < 24; k++) {
            int32_t node = nearest[k];
            double row[3] = {(double)flat->u[node] - flat->u[centre],
                (double)flat->v[node] - flat->v[centre], 1};
            double value;
            if (!roll) {
                value = field->nodes[node].z;
            } else {
                double cx, cy;
                hx_centre_at(field->axis, field->axis_count, field->nodes[node].z, &cx, &cy);
                double radius = hypot(field->nodes[node].x - cx, field->nodes[node].y - cy);
                double turn =
                    solution->potential[node] + (double)(field->theta[node] / (float)(2 * M_PI));
                value = 2 * M_PI * turn * radius;
            }
            value -= centre_value;
            for (int y = 0; y < 3; y++) {
                target[y] += row[y] * value;
                for (int x = 0; x < 3; x++)
                    matrix[y][x] += row[y] * row[x];
            }
        }
        for (int axis = 0; axis < 3; axis++) {
            matrix[axis][axis] += 1e-9;
        }
        double slope[3] = {0};
        if (!solve_three(matrix, target, slope) && hypot(slope[0], slope[1]) >= 0.3)
            angles[steep++] = atan2(slope[0], slope[1]);
    }
    if (steep * 2 < samples) {
        return 0;
    }
    double sine = 0, cosine = 0;
    for (int i = 0; i < steep; i++) {
        sine += sin(angles[i]);
        cosine += cos(angles[i]);
    }
    double centre = atan2(sine / steep, cosine / steep);
    for (int i = 0; i < steep; i++)
        angles[i] = atan2(sin(angles[i] - centre), cos(angles[i] - centre));
    qsort(angles, (size_t)steep, sizeof(*angles), double_before);
    double median =
        steep & 1 ? angles[steep / 2] : (angles[steep / 2 - 1] + angles[steep / 2]) * 0.5;
    *direction = centre + median;
    return 1;
}

static int upright_piece(
    Field *field, Solution *solution, Flattened *flat, int32_t piece, double *direction) {
    int64_t begin = flat->first[piece], end = flat->first[piece + 1];
    int found = median_direction(field, solution, flat, begin, end, 0, direction);
    if (!found) {
        found = median_direction(field, solution, flat, begin, end, 1, direction);
        if (found > 0)
            *direction -= M_PI / 2;
    }
    if (found <= 0 || fabs(atan2(sin(*direction), cos(*direction))) <= 10 * M_PI / 180)
        return found < 0 ? -1 : 0;
    double cosine = cos(*direction), sine = sin(*direction);
    for (int64_t at = begin; at < end; at++) {
        int32_t i = flat->members[at];
        float u = flat->u[i], v = flat->v[i];
        flat->u[i] = (float)(u * cosine - v * sine);
        flat->v[i] = (float)(u * sine + v * cosine);
    }
    return 0;
}

static void raster_close(Raster *raster) {
    free(raster->locked);
    free(raster->cell);
    free(raster->spread);
    free(raster->grid);
    free(raster->dv);
    free(raster->du);
    free(raster->normal);
    free(raster->xyz);
    memset(raster, 0, sizeof(*raster));
}

static int raster_piece(Field *field, Flattened *flat, int32_t piece, Raster *raster) {
    memset(raster, 0, sizeof(*raster));
    if (piece < 0 || piece >= flat->pieces)
        return -1;
    int64_t begin = flat->first[piece], end = flat->first[piece + 1], count = end - begin;
    raster->nodes = count;
    raster->xyz = malloc((size_t)(count ? count : 1) * 3 * sizeof(*raster->xyz));
    raster->normal = malloc((size_t)(count ? count : 1) * 3 * sizeof(*raster->normal));
    raster->du = malloc((size_t)(count ? count : 1) * sizeof(*raster->du));
    raster->dv = malloc((size_t)(count ? count : 1) * sizeof(*raster->dv));
    if (!raster->xyz || !raster->normal || !raster->du || !raster->dv) {
        raster_close(raster);
        return -1;
    }
    float min_u = flat->u[flat->members[begin]], min_v = flat->v[flat->members[begin]];
    for (int64_t at = begin + 1; at < end; at++) {
        int32_t i = flat->members[at];
        min_u = fminf(min_u, flat->u[i]);
        min_v = fminf(min_v, flat->v[i]);
    }
    double max_u = 0, max_v = 0;
    for (int64_t at = begin; at < end; at++) {
        int64_t j = at - begin;
        int32_t i = flat->members[at];
        Node node = field->nodes[i];
        raster->xyz[3 * j] = node.x;
        raster->xyz[3 * j + 1] = node.y;
        raster->xyz[3 * j + 2] = node.z;
        float nx = node.nx * (float)flat->sign[i];
        float ny = node.ny * (float)flat->sign[i];
        float nz = node.nz * (float)flat->sign[i];
        float magnitude = fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), 1e-6f);
        raster->normal[3 * j] = nx / magnitude;
        raster->normal[3 * j + 1] = ny / magnitude;
        raster->normal[3 * j + 2] = nz / magnitude;
        raster->du[j] = (float)(flat->u[i] - min_u);
        raster->dv[j] = (float)(flat->v[i] - min_v);
        max_u = fmax(max_u, raster->du[j]);
        max_v = fmax(max_v, raster->dv[j]);
    }
    raster->height = (int64_t)(max_v / HX_CELL) + 1;
    raster->width = (int64_t)(max_u / HX_CELL) + 1;
    int64_t cells = raster->height * raster->width;
    raster->cell = malloc((size_t)(count ? count : 1) * sizeof(*raster->cell));
    raster->grid = calloc((size_t)(cells ? cells : 1) * 3, sizeof(*raster->grid));
    raster->spread = calloc((size_t)(cells ? cells : 1), sizeof(*raster->spread));
    raster->locked = malloc((size_t)(cells ? cells : 1));
    raster->occupied = raster->cell && raster->grid && raster->spread && raster->locked
        ? hx_raster_cells(raster->height, raster->width, count, raster->xyz, raster->normal,
              raster->du, raster->dv, HX_CELL, 2.0, raster->cell, raster->grid, raster->locked,
              raster->spread)
        : -1;
    if (raster->occupied < 1) {
        raster_close(raster);
        return -1;
    }
    int status = hx_centre_cells(raster->height, raster->width, count, raster->cell, raster->du,
        raster->dv, raster->xyz, NULL, raster->locked, HX_CELL, 4, 0.1 * HX_CELL * HX_CELL, 0.5,
        2.0, raster->grid);
    if (status) {
        raster_close(raster);
        return -1;
    }
    return 0;
}

static int contact_layers(Raster *raster) {
    int64_t cells = raster->height * raster->width;
    uint8_t *occupied = malloc((size_t)cells);
    double *near = calloc((size_t)cells * 3, sizeof(*near));
    int32_t *near_count = calloc((size_t)cells, sizeof(*near_count));
    double *depth = malloc((size_t)raster->nodes * sizeof(*depth));
    uint8_t *take = calloc((size_t)raster->nodes, 1);
    double *nearest = malloc((size_t)cells * sizeof(*nearest));
    if (!occupied || !near || !near_count || !depth || !take || !nearest)
        goto fail;
    for (int64_t at = 0; at < cells; at++) {
        occupied[at] = raster->grid[at] != 0;
        raster->locked[at] = occupied[at] && raster->spread[at] <= 3.0;
        nearest[at] = DBL_MAX;
    }
    for (int64_t row = 0; row < raster->height; row++)
        for (int64_t col = 0; col < raster->width; col++) {
            int64_t at = row * raster->width + col;
            int64_t low_row = row > 2 ? row - 2 : 0;
            int64_t high_row = row + 3 < raster->height ? row + 3 : raster->height;
            int64_t low_col = col > 2 ? col - 2 : 0;
            int64_t high_col = col + 3 < raster->width ? col + 3 : raster->width;
            for (int64_t y = low_row; y < high_row; y++)
                for (int64_t x = low_col; x < high_col; x++) {
                    int64_t source = y * raster->width + x;
                    if (!raster->locked[source])
                        continue;
                    near_count[at]++;
                    for (int axis = 0; axis < 3; axis++)
                        near[axis * cells + at] += raster->grid[axis * cells + source];
                }
            if (near_count[at])
                for (int axis = 0; axis < 3; axis++)
                    near[axis * cells + at] /= near_count[at];
        }
    int64_t selected = 0;
    for (int64_t i = 0; i < raster->nodes; i++) {
        int64_t at = raster->cell[i];
        depth[i] = DBL_MAX;
        if (!occupied[at] || raster->locked[at] || near_count[at] < 2)
            continue;
        double value = 0;
        for (int axis = 0; axis < 3; axis++)
            value += (raster->xyz[3 * i + axis] - near[axis * cells + at]) *
                raster->normal[3 * i + axis];
        depth[i] = value;
        nearest[at] = fmin(nearest[at], fabs(value));
    }
    for (int64_t i = 0; i < raster->nodes; i++) {
        int64_t at = raster->cell[i];
        take[i] = fabs(depth[i]) <= nearest[at] + 2.0;
        selected += take[i] != 0;
    }
    if (!selected) {
        free(nearest);
        free(take);
        free(depth);
        free(near_count);
        free(near);
        free(occupied);
        return 0;
    }
    int64_t *hits = calloc((size_t)cells, sizeof(*hits));
    double *mean = calloc((size_t)cells * 3, sizeof(*mean));
    double *offset = calloc((size_t)cells, sizeof(*offset));
    double *square = calloc((size_t)cells, sizeof(*square));
    uint8_t *source = malloc((size_t)raster->nodes);
    uint8_t *good = calloc((size_t)cells, 1);
    if (!hits || !mean || !offset || !square || !source || !good) {
        free(good);
        free(source);
        free(square);
        free(offset);
        free(mean);
        free(hits);
        goto fail;
    }
    for (int64_t i = 0; i < raster->nodes; i++) {
        source[i] = raster->locked[raster->cell[i]];
        if (!take[i])
            continue;
        int64_t at = raster->cell[i];
        hits[at]++;
        offset[at] += depth[i];
        square[at] += depth[i] * depth[i];
        for (int axis = 0; axis < 3; axis++)
            mean[axis * cells + at] += raster->xyz[3 * i + axis];
    }
    for (int64_t at = 0; at < cells; at++) {
        int64_t count = hits[at] ? hits[at] : 1;
        offset[at] /= count;
        square[at] /= count;
        for (int axis = 0; axis < 3; axis++)
            mean[axis * cells + at] /= count;
        double spread = sqrt(fmax(square[at] - offset[at] * offset[at], 0));
        good[at] = hits[at] && spread <= 3.0 && fabs(offset[at]) <= 4.0;
    }
    int status = hx_centre_cells(raster->height, raster->width, raster->nodes, raster->cell,
        raster->du, raster->dv, raster->xyz, take, source, HX_CELL, 4, 0.1 * HX_CELL * HX_CELL, 0.5,
        2.0, mean);
    if (!status)
        for (int64_t at = 0; at < cells; at++)
            if (good[at]) {
                raster->locked[at] = 1;
                for (int axis = 0; axis < 3; axis++)
                    raster->grid[axis * cells + at] = mean[axis * cells + at];
            }
    free(good);
    free(source);
    free(square);
    free(offset);
    free(mean);
    free(hits);
    free(nearest);
    free(take);
    free(depth);
    free(near_count);
    free(near);
    free(occupied);
    return status ? -1 : 0;
fail:
    free(nearest);
    free(take);
    free(depth);
    free(near_count);
    free(near);
    free(occupied);
    return -1;
}

static int second_layer(const Raster *raster, const GuardSettings *guard, uint8_t *doubled) {
    int64_t cells = raster->height * raster->width;
    int64_t *cells_of = NULL;
    const int64_t *flat = raster->cell;
    if (guard->cell != HX_CELL) {
        cells_of = malloc((size_t)raster->nodes * sizeof(*cells_of));
        if (!cells_of) {
            return -1;
        }
        for (int64_t i = 0; i < raster->nodes; i++) {
            int64_t at = flat[i], row = at / raster->width, column = at % raster->width;
            double x = raster->du[i] + (column + .5) * HX_CELL;
            double y = raster->dv[i] + (row + .5) * HX_CELL;
            x = fmax(0., x) / guard->cell;
            y = fmax(0., y) / guard->cell;
            if (x >= raster->width || y >= raster->height) {
                free(cells_of);
                return -1;
            }
            cells_of[i] = (int64_t)y * raster->width + (int64_t)x;
        }
        flat = cells_of;
    }
    if (guard->cut < 2) {
        for (int64_t i = 0; i < raster->nodes; i++) {
            doubled[i] = !raster->locked[flat[i]];
        }
        free(cells_of);
        return 0;
    }
    int64_t window = (int64_t)guard->contact_cells, before = window / 2, after = window - before;
    double ratio = window / guard->contact_cells;
    double threshold = 1.5 * ratio * ratio;
    double *near = calloc((size_t)cells * 3, sizeof(*near));
    int32_t *near_count = calloc((size_t)cells, sizeof(*near_count));
    double *depth = malloc((size_t)raster->nodes * sizeof(*depth));
    double *nearest = malloc((size_t)cells * sizeof(*nearest));
    uint8_t *eligible = malloc((size_t)raster->nodes);
    if (!near || !near_count || !depth || !nearest || !eligible)
        goto fail;
    for (int64_t at = 0; at < cells; at++)
        nearest[at] = DBL_MAX;
    for (int64_t row = 0; row < raster->height; row++)
        for (int64_t col = 0; col < raster->width; col++) {
            int64_t at = row * raster->width + col;
            int64_t low_row = row > before ? row - before : 0;
            int64_t high_row = row + after < raster->height ? row + after : raster->height;
            int64_t low_col = col > before ? col - before : 0;
            int64_t high_col = col + after < raster->width ? col + after : raster->width;
            for (int64_t y = low_row; y < high_row; y++)
                for (int64_t x = low_col; x < high_col; x++) {
                    int64_t source = y * raster->width + x;
                    if (!raster->locked[source])
                        continue;
                    near_count[at]++;
                    for (int axis = 0; axis < 3; axis++)
                        near[axis * cells + at] += raster->grid[axis * cells + source];
                }
            if (near_count[at])
                for (int axis = 0; axis < 3; axis++)
                    near[axis * cells + at] /= near_count[at];
        }
    for (int64_t i = 0; i < raster->nodes; i++) {
        int64_t at = flat[i];
        eligible[i] = !raster->locked[at];
        depth[i] = DBL_MAX;
        if (!eligible[i] || near_count[at] <= threshold)
            continue;
        double value = 0;
        for (int axis = 0; axis < 3; axis++)
            value += (raster->xyz[3 * i + axis] - near[axis * cells + at]) *
                raster->normal[3 * i + axis];
        depth[i] = value;
        nearest[at] = fmin(nearest[at], fabs(value));
    }
    for (int64_t i = 0; i < raster->nodes; i++) {
        int64_t at = flat[i];
        int keep = eligible[i] && near_count[at] > threshold &&
            fabs(depth[i]) <= nearest[at] + guard->contact_layers;
        doubled[i] = eligible[i] && !keep;
    }
    free(eligible);
    free(nearest);
    free(depth);
    free(near_count);
    free(cells_of);
    free(near);
    return 0;
fail:
    free(eligible);
    free(nearest);
    free(depth);
    free(near_count);
    free(cells_of);
    free(near);
    return -1;
}

static int refused_nodes(
    Field *field, Solution *solution, const float *area_node, uint8_t *refused, uint8_t *doubled) {
    const GuardSettings *guard = field->guard ? field->guard : &default_guard;
    int64_t count = field->nodes_count;
    double *area = calloc((size_t)(count ? count : 1), sizeof(*area));
    if (!area)
        return -1;
    for (int64_t i = 0; i < count; i++)
        area[solution->root[i]] += area_node[i];
    memset(refused, 0, (size_t)count);
    memset(doubled, 0, (size_t)count);
    Flattened flat;
    if (flatten_solution(field, solution, &flat)) {
        free(area);
        return -1;
    }
    for (int32_t piece = 0; piece < flat.pieces; piece++) {
        int64_t begin = flat.first[piece], end = flat.first[piece + 1];
        int32_t root = solution->root[flat.members[begin]];
        if (area[root] < guard->floor_mm2)
            break;
        Raster raster;
        if (raster_piece(field, &flat, piece, &raster)) {
            flattened_close(&flat);
            free(area);
            return -1;
        }
        int64_t occupied = 0, locked = 0;
        int64_t cells = raster.height * raster.width;
        for (int64_t at = 0; at < cells; at++) {
            occupied += raster.grid[at] != 0;
            locked += raster.locked[at] != 0;
        }
        double spread = 1.0 - (double)locked / (occupied ? occupied : 1);
        int reject = spread > guard->spread;
        if (solution->guard_count < 40) {
            double *row = solution->guard[solution->guard_count++];
            row[0] = nearbyint(area[root] / 10.) / 10.;
            row[1] = nearbyint(spread * 1000.) / 1000.;
        }
        uint8_t *local = reject && guard->cut ? malloc((size_t)raster.nodes) : NULL;
        int failed = reject && guard->cut && (!local || second_layer(&raster, guard, local));
        if (!failed && reject)
            for (int64_t at = begin; at < end; at++) {
                int32_t node = flat.members[at];
                refused[node] = 1;
                doubled[node] = local ? local[at - begin] : 0;
            }
        free(local);
        raster_close(&raster);
        if (failed) {
            flattened_close(&flat);
            free(area);
            return -1;
        }
    }
    flattened_close(&flat);
    free(area);
    return 0;
}

static int guard_solution(Field *field, Solution *solution) {
    double began = level_clock();
    const GuardSettings *guard = field->guard ? field->guard : &default_guard;
    int64_t count = field->nodes_count;
    float *area_node = malloc((size_t)(count ? count : 1) * sizeof(*area_node));
    uint8_t *refused = malloc((size_t)(count ? count : 1));
    uint8_t *doubled = malloc((size_t)(count ? count : 1));
    uint8_t *out = calloc((size_t)(count ? count : 1), 1);
    uint8_t *trial = malloc((size_t)(count ? count : 1));
    if (!area_node || !refused || !doubled || !out || !trial)
        goto fail;
    double side = guard->cell * (UINT64_C(1) << field->level) * field->scan_voxel_um / 1000;
    hx_node_area(field->nodes, count, (float)(side * side), area_node);
    if (refused_nodes(field, solution, area_node, refused, doubled))
        goto fail;
    for (int64_t i = 0; i < count; i++) {
        out[i] = refused[i] && (!guard->cut || doubled[i]);
        trial[i] = guard->cut && refused[i] && !doubled[i];
    }
    int32_t *cut_root = malloc((size_t)(count ? count : 1) * sizeof(*cut_root));
    int32_t *cut_potential = malloc((size_t)(count ? count : 1) * sizeof(*cut_potential));
    int32_t *probe = malloc((size_t)(count ? count : 1) * sizeof(*probe));
    uint8_t *again = malloc((size_t)(count ? count : 1));
    if (!cut_root || !cut_potential || !probe || !again) {
        free(again);
        free(probe);
        free(cut_potential);
        free(cut_root);
        goto fail;
    }
    for (int round = 0; round < 2; round++) {
        int any = 0;
        for (int64_t i = 0; i < count; i++)
            any |= trial[i];
        if (!any)
            break;
        if (settled(count, solution->attract, solution->edge_count, NULL, solution->root, trial,
                solution->potential, cut_root, cut_potential)) {
            free(again);
            free(probe);
            free(cut_potential);
            free(cut_root);
            goto fail;
        }
        for (int64_t i = 0; i < count; i++) {
            if (trial[i]) {
                solution->root[i] = cut_root[i];
                solution->potential[i] = cut_potential[i];
            }
            probe[i] = trial[i] ? solution->root[i] : (int32_t)i;
        }
        int32_t *held = solution->root;
        solution->root = probe;
        int failed = refused_nodes(field, solution, area_node, again, doubled);
        solution->root = held;
        if (failed) {
            free(again);
            free(probe);
            free(cut_potential);
            free(cut_root);
            goto fail;
        }
        for (int64_t i = 0; i < count; i++) {
            out[i] |= trial[i] && again[i] && doubled[i];
            trial[i] = trial[i] && again[i] && !doubled[i];
        }
    }
    int64_t refused_count = 0;
    for (int64_t i = 0; i < count; i++) {
        refused[i] = out[i] || trial[i];
        refused_count += refused[i] != 0;
    }
    if (refused_count) {
        if (settled(count, solution->attract, solution->edge_count, solution->kept,
                solution->coarse_root, refused, NULL, cut_root, cut_potential)) {
            free(again);
            free(probe);
            free(cut_potential);
            free(cut_root);
            goto fail;
        }
        for (int64_t i = 0; i < count; i++)
            if (refused[i]) {
                solution->root[i] = cut_root[i];
                solution->potential[i] = cut_potential[i];
            }
    }
    solution->refused_share = count ? (double)refused_count / count : NAN;
    solution->timings[6] = level_clock() - began;
    free(again);
    free(probe);
    free(cut_potential);
    free(cut_root);
    free(trial);
    free(out);
    free(doubled);
    free(refused);
    free(area_node);
    return 0;
fail:
    free(trial);
    free(out);
    free(doubled);
    free(refused);
    free(area_node);
    return -1;
}

static void mesh_close(Mesh *mesh) {
    free(mesh->kept);
    free(mesh->known);
    free(mesh->grid);
    memset(mesh, 0, sizeof(*mesh));
}

static uint8_t *closed_gaps(const uint8_t *mask, int64_t height, int64_t width) {
    int64_t padded_height = height + 6, padded_width = width + 6;
    size_t padded_cells = (size_t)padded_height * padded_width;
    uint8_t *padded = calloc(padded_cells, 1);
    uint8_t *grown = calloc(padded_cells, 1);
    uint8_t *closed = calloc((size_t)height * width, 1);
    if (!padded || !grown || !closed) {
        free(closed);
        free(grown);
        free(padded);
        return NULL;
    }
    for (int64_t row = 0; row < height; row++)
        memcpy(padded + (row + 3) * padded_width + 3, mask + row * width, (size_t)width);
    for (int64_t row = 0; row < padded_height; row++)
        for (int64_t col = 0; col < padded_width; col++) {
            int64_t low_row = row > 3 ? row - 3 : 0;
            int64_t high_row = row + 4 < padded_height ? row + 4 : padded_height;
            int64_t low_col = col > 3 ? col - 3 : 0;
            int64_t high_col = col + 4 < padded_width ? col + 4 : padded_width;
            for (int64_t y = low_row; y < high_row && !grown[row * padded_width + col]; y++)
                for (int64_t x = low_col; x < high_col; x++)
                    grown[row * padded_width + col] |= padded[y * padded_width + x];
        }
    for (int64_t row = 0; row < height; row++)
        for (int64_t col = 0; col < width; col++) {
            uint8_t value = 1;
            int64_t centre_row = row + 3, centre_col = col + 3;
            for (int64_t y = centre_row - 3; y <= centre_row + 3 && value; y++)
                for (int64_t x = centre_col - 3; x <= centre_col + 3; x++)
                    value &= grown[y * padded_width + x];
            closed[row * width + col] = value;
        }
    free(grown);
    free(padded);
    return closed;
}

static int bridge_holes(Raster *raster, Mesh *mesh) {
    memset(mesh, 0, sizeof(*mesh));
    mesh->height = raster->height;
    mesh->width = raster->width;
    int64_t cells = raster->height * raster->width;
    uint8_t *hole = closed_gaps(raster->locked, raster->height, raster->width);
    mesh->grid = malloc((size_t)cells * 3 * sizeof(*mesh->grid));
    mesh->known = malloc((size_t)cells);
    mesh->kept = malloc((size_t)cells);
    uint8_t *next = malloc((size_t)cells);
    if (!hole || !mesh->grid || !mesh->known || !mesh->kept || !next)
        goto fail;
    for (int64_t at = 0; at < cells; at++) {
        mesh->known[at] = raster->locked[at];
        for (int axis = 0; axis < 3; axis++)
            mesh->grid[axis * cells + at] =
                raster->locked[at] ? raster->grid[axis * cells + at] : 0;
    }
    for (int round = 0; round < 3; round++) {
        memcpy(next, mesh->known, (size_t)cells);
        for (int64_t row = 0; row < raster->height; row++)
            for (int64_t col = 0; col < raster->width; col++) {
                int64_t at = row * raster->width + col;
                if (mesh->known[at] || !hole[at])
                    continue;
                int count = 0;
                double total[3] = {0};
                for (int dy = -1; dy <= 1; dy++) {
                    if (row + dy < 0 || row + dy >= raster->height)
                        continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        if (col + dx < 0 || col + dx >= raster->width)
                            continue;
                        int64_t source = at + dy * raster->width + dx;
                        if (!mesh->known[source])
                            continue;
                        count++;
                        for (int axis = 0; axis < 3; axis++)
                            total[axis] += mesh->grid[axis * cells + source];
                    }
                }
                if (!count)
                    continue;
                next[at] = 1;
                for (int axis = 0; axis < 3; axis++)
                    mesh->grid[axis * cells + at] = total[axis] / count;
            }
        memcpy(mesh->known, next, (size_t)cells);
    }
    int64_t free_count = 0;
    for (int64_t at = 0; at < cells; at++) {
        next[at] = mesh->known[at] && !raster->locked[at];
        free_count += next[at] != 0;
    }
    if (hx_relax_mesh(
            mesh->grid, 0, 3, raster->height, raster->width, mesh->known, next, free_count, 150))
        goto fail;
    memcpy(mesh->kept, mesh->known, (size_t)cells);
    for (int axis = 0; axis < 2; axis++) {
        int64_t row_limit = raster->height - (axis == 0);
        int64_t col_limit = raster->width - (axis == 1);
        int64_t step = axis == 0 ? raster->width : 1;
        for (int64_t row = 0; row < row_limit; row++)
            for (int64_t col = 0; col < col_limit; col++) {
                int64_t at = row * raster->width + col, other = at + step;
                if (!mesh->known[at] || !mesh->known[other])
                    continue;
                double distance = 0;
                for (int k = 0; k < 3; k++) {
                    double delta = mesh->grid[k * cells + at] - mesh->grid[k * cells + other];
                    distance += delta * delta;
                }
                if (distance > 9.0 * HX_CELL * HX_CELL)
                    mesh->kept[at] = mesh->kept[other] = 0;
            }
    }
    free(next);
    free(hole);
    return 0;
fail:
    free(next);
    free(hole);
    mesh_close(mesh);
    return -1;
}

static int make_folders(const char *path) {
    char *copy = strdup(path);
    if (!copy)
        return 0;
    for (char *p = copy + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = 0;
        int ok = mkdir(copy, 0777) == 0 || errno == EEXIST;
        *p = '/';
        if (!ok) {
            free(copy);
            return 0;
        }
    }
    int ok = mkdir(copy, 0777) == 0 || errno == EEXIST;
    free(copy);
    return ok;
}

static int write_quality(const char *path, const uint8_t *quality, int64_t height, int64_t width) {
    FILE *file = fopen(path, "wb");
    if (!file)
        return -1;
    char shape[96];
    int body = snprintf(shape, sizeof(shape),
        "{'descr': '|u1', 'fortran_order': False, 'shape': (%lld, %lld), }", (long long)height,
        (long long)width);
    int header = body + 1;
    while ((10 + header) % 16)
        header++;
    unsigned char prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, header & 255, header >> 8};
    int ok = fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix) &&
        fwrite(shape, 1, (size_t)body, file) == (size_t)body;
    for (int i = body; ok && i + 1 < header; i++)
        ok = fputc(' ', file) != EOF;
    ok = ok && fputc('\n', file) != EOF &&
        fwrite(quality, 1, (size_t)(height * width), file) == (size_t)(height * width);
    ok = fclose(file) == 0 && ok;
    return ok ? 0 : -1;
}

static void *npy_blob(const char *type, const int64_t *shape, int dimensions, const void *values,
    size_t bytes, size_t *blob_size) {
    char dimensions_text[160];
    int used = dimensions == 0 ? snprintf(dimensions_text, sizeof(dimensions_text), "()")
        : dimensions == 1
        ? snprintf(dimensions_text, sizeof(dimensions_text), "(%lld,)", (long long)shape[0])
        : dimensions == 2 ? snprintf(dimensions_text, sizeof(dimensions_text), "(%lld, %lld)",
                                (long long)shape[0], (long long)shape[1])
                          : -1;
    char body[256];
    int body_size = used < 0 || (size_t)used >= sizeof(dimensions_text)
        ? -1
        : snprintf(body, sizeof(body), "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
              type, dimensions_text);
    if (body_size < 0 || (size_t)body_size >= sizeof(body))
        return NULL;
    size_t header = (size_t)body_size + 1;
    while ((10 + header) % 64)
        header++;
    if (header > UINT16_MAX || bytes > SIZE_MAX - 10 - header)
        return NULL;
    *blob_size = 10 + header + bytes;
    unsigned char *blob = malloc(*blob_size ? *blob_size : 1);
    if (!blob)
        return NULL;
    unsigned char prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, header & 255, header >> 8};
    memcpy(blob, prefix, sizeof(prefix));
    memcpy(blob + 10, body, (size_t)body_size);
    memset(blob + 10 + body_size, ' ', header - (size_t)body_size);
    blob[10 + header - 1] = '\n';
    if (bytes)
        memcpy(blob + 10 + header, values, bytes);
    return blob;
}

static int write_npy(const char *path, const char *type, const int64_t *shape, int dimensions,
    const void *values, size_t bytes) {
    size_t size = 0;
    void *blob = npy_blob(type, shape, dimensions, values, bytes, &size);
    FILE *file = blob ? fopen(path, "wb") : NULL;
    int ok = file && fwrite(blob, 1, size, file) == size;
    if (file && fclose(file))
        ok = 0;
    free(blob);
    return ok ? 0 : -1;
}

typedef struct {
    zip_t *file;
    void **blobs;
    size_t count, capacity;
} Npz;

static int npz_add(Npz *archive, const char *name, const char *type, const int64_t *shape,
    int dimensions, const void *values, size_t bytes) {
    char entry[96];
    int length = snprintf(entry, sizeof(entry), "%s.npy", name);
    if (length < 0 || (size_t)length >= sizeof(entry))
        return -1;
    if (archive->count == archive->capacity) {
        size_t capacity = archive->capacity ? archive->capacity * 2 : 16;
        void **blobs = realloc(archive->blobs, capacity * sizeof(*blobs));
        if (!blobs)
            return -1;
        archive->blobs = blobs;
        archive->capacity = capacity;
    }
    size_t blob_size = 0;
    void *blob = npy_blob(type, shape, dimensions, values, bytes, &blob_size);
    zip_source_t *source = blob ? zip_source_buffer(archive->file, blob, blob_size, 0) : NULL;
    if (!source) {
        free(blob);
        return -1;
    }
    zip_int64_t entry_id =
        zip_file_add(archive->file, entry, source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    if (entry_id < 0) {
        zip_source_free(source);
        free(blob);
        return -1;
    }
    archive->blobs[archive->count++] = blob;
    return zip_file_set_mtime(archive->file, (zip_uint64_t)entry_id, 315576000, 0);
}

static int npz_finish(Npz *archive, const char *path, int failed) {
    if (archive->file) {
        if (failed)
            zip_discard(archive->file);
        else if (zip_close(archive->file) < 0) {
            zip_discard(archive->file);
            failed = 1;
        }
    }
    for (size_t i = 0; i < archive->count; i++)
        free(archive->blobs[i]);
    free(archive->blobs);
    if (failed && path)
        unlink(path);
    return failed ? -1 : 0;
}

typedef struct {
    zip_t *file;
} NpzReader;

static int npz_open_reader(NpzReader *reader, const char *path) {
    int error = 0;
    reader->file = zip_open(path, ZIP_RDONLY, &error);
    return reader->file ? 0 : -1;
}

static int npz_read(NpzReader *reader, const char *name, const char *type, size_t item_size,
    void **values, int64_t *count) {
    char entry[96];
    int length = snprintf(entry, sizeof(entry), "%s.npy", name);
    zip_stat_t info;
    zip_stat_init(&info);
    if (length < 0 || (size_t)length >= sizeof(entry) ||
        zip_stat(reader->file, entry, ZIP_FL_ENC_UTF_8, &info) || info.size > SIZE_MAX)
        return -1;
    zip_file_t *file = zip_fopen(reader->file, entry, ZIP_FL_ENC_UTF_8);
    unsigned char *blob = file ? malloc((size_t)(info.size ? info.size : 1)) : NULL;
    size_t done = 0;
    while (blob && done < info.size) {
        zip_int64_t n = zip_fread(file, blob + done, info.size - done);
        if (n <= 0)
            break;
        done += (size_t)n;
    }
    if (file)
        zip_fclose(file);
    if (!blob || done != info.size || info.size < 10 || memcmp(blob, "\x93NUMPY", 6) != 0) {
        free(blob);
        return -1;
    }
    size_t prefix = blob[6] == 1 ? 10 : blob[6] == 2 || blob[6] == 3 ? 12 : 0;
    if (prefix == 12 && info.size < 12) {
        free(blob);
        return -1;
    }
    size_t header = prefix == 10 ? (size_t)blob[8] | (size_t)blob[9] << 8
        : prefix == 12
        ? (size_t)blob[8] | (size_t)blob[9] << 8 | (size_t)blob[10] << 16 | (size_t)blob[11] << 24
        : 0;
    size_t offset = prefix + header;
    if (!prefix || offset > info.size || !memmem(blob + prefix, header, type, strlen(type)) ||
        !item_size || (info.size - offset) % item_size) {
        free(blob);
        return -1;
    }
    size_t items = (info.size - offset) / item_size;
    if (items > INT64_MAX) {
        free(blob);
        return -1;
    }
    memmove(blob, blob + offset, info.size - offset);
    *values = blob;
    *count = (int64_t)items;
    return 0;
}

static void npz_close_reader(NpzReader *reader) {
    if (reader->file)
        zip_close(reader->file);
    reader->file = NULL;
}

typedef struct {
    int64_t id;
    int32_t piece;
    double turn;
} BoundaryRow;

typedef struct {
    BoundaryRow *rows;
    int64_t count, pieces;
} Boundary;

static int boundary_before(const void *left, const void *right) {
    const BoundaryRow *a = left, *b = right;
    return (a->id > b->id) - (a->id < b->id);
}

static int boundary_read(Boundary *boundary, const char *unit, int turns) {
    char *path = NULL;
    if (asprintf(&path, "%s/boundary.npz", unit) < 0)
        return -1;
    NpzReader reader = {0};
    int64_t *ids = NULL, *pieces = NULL;
    int32_t *labels = NULL, *potential = NULL;
    float *theta = NULL;
    int64_t id_count = 0, label_count = 0, potential_count = 0, theta_count = 0, piece_count = 0;
    int failed = npz_open_reader(&reader, path) ||
        npz_read(&reader, "ids", "<i8", sizeof(*ids), (void **)&ids, &id_count) ||
        npz_read(&reader, "piece", "<i4", sizeof(*labels), (void **)&labels, &label_count) ||
        id_count != label_count;
    if (!failed && turns)
        failed = npz_read(&reader, "potential", "<i4", sizeof(*potential), (void **)&potential,
                     &potential_count) ||
            npz_read(&reader, "theta", "<f4", sizeof(*theta), (void **)&theta, &theta_count) ||
            npz_read(&reader, "pieces", "<i8", sizeof(*pieces), (void **)&pieces, &piece_count) ||
            id_count != potential_count || id_count != theta_count || piece_count != 1;
    BoundaryRow *rows = !failed && (uint64_t)id_count <= SIZE_MAX / sizeof(*rows)
        ? malloc((size_t)(id_count ? id_count : 1) * sizeof(*rows))
        : NULL;
    int sorted = 1;
    for (int64_t i = 0; rows && i < id_count; i++) {
        rows[i] = (BoundaryRow){.id = ids[i],
            .piece = labels[i],
            .turn = turns ? potential[i] + theta[i] / (float)(2 * M_PI) : 0};
        sorted &= !i || ids[i - 1] <= ids[i];
    }
    if (rows && !sorted)
        qsort(rows, (size_t)id_count, sizeof(*rows), boundary_before);
    *boundary = (Boundary){
        .rows = rows, .count = rows ? id_count : 0, .pieces = rows && pieces ? *pieces : 0};
    npz_close_reader(&reader);
    free(pieces);
    free(theta);
    free(potential);
    free(labels);
    free(ids);
    free(path);
    return rows ? 0 : -1;
}

static int boundary_open(Boundary *boundary, const char *unit) {
    return boundary_read(boundary, unit, 1);
}

static const BoundaryRow *boundary_row(const Boundary *boundary, int64_t id) {
    int64_t first = 0, end = boundary->count;
    while (first < end) {
        int64_t middle = first + (end - first) / 2;
        if (boundary->rows[middle].id < id)
            first = middle + 1;
        else
            end = middle;
    }
    return first < boundary->count && boundary->rows[first].id == id ? boundary->rows + first
                                                                     : NULL;
}

static int32_t boundary_piece(const Boundary *boundary, int64_t id) {
    const BoundaryRow *row = boundary_row(boundary, id);
    return row ? row->piece : -1;
}

typedef struct {
    Node *nodes;
    int32_t *cells;
    size_t count, first;
    int grid[3], low, high, level;
    int64_t row0;
    char *sample, *volume;
} RawUnit;

static void raw_unit_close(RawUnit *unit) {
    free(unit->volume);
    free(unit->sample);
    free(unit->cells);
    free(unit->nodes);
    memset(unit, 0, sizeof(*unit));
}

static size_t raw_cell_at(const int32_t *cells, size_t count, int64_t target) {
    size_t first = 0;
    while (first < count) {
        size_t middle = first + (count - first) / 2;
        if (cells[middle] < target)
            first = middle + 1;
        else
            count = middle;
    }
    return first;
}

static int raw_unit_open(RawUnit *raw, const char *unit, int64_t begin, int64_t end) {
    memset(raw, 0, sizeof(*raw));
    char *path = NULL;
    if (asprintf(&path, "%s/record.json", unit) < 0)
        return -1;
    json_object *record = json_object_from_file(path);
    free(path);
    json_object *planes, *rows, *value;
    int valid = record && !json_triplet(record, "grid", raw->grid) &&
        json_object_object_get_ex(record, "cell_planes", &planes) &&
        json_object_array_length(planes) == 2;
    if (!valid) {
        if (record)
            json_object_put(record);
        return -1;
    }
    raw->low = json_object_get_int(json_object_array_get_idx(planes, 0));
    raw->high = json_object_get_int(json_object_array_get_idx(planes, 1));
    raw->level =
        json_object_object_get_ex(record, "level", &value) ? json_object_get_int(value) : 0;
    if (json_object_object_get_ex(record, "rows", &rows) && json_object_array_length(rows))
        raw->row0 = json_object_get_int64(json_object_array_get_idx(rows, 0));
    raw->sample = json_text(record, "sample");
    raw->volume = json_text(record, "volume");
    json_object_put(record);
    size_t cell_count = 0;
    if (!raw->sample || !raw->volume || raw->grid[1] <= 0 || raw->grid[2] <= 0 ||
        raw->low > raw->high || hx_cell_read(unit, &raw->cells, &cell_count)) {
        raw_unit_close(raw);
        return -1;
    }
    int64_t plane_size = (int64_t)raw->grid[1] * raw->grid[2];
    size_t first = begin <= raw->low ? 0
        : begin >= raw->high         ? cell_count
                             : raw_cell_at(raw->cells, cell_count, (begin - raw->low) * plane_size);
    size_t last = end >= raw->high ? cell_count
        : end <= raw->low          ? 0
                          : raw_cell_at(raw->cells, cell_count, (end - raw->low) * plane_size);
    void *nodes = NULL;
    if (hx_node_read_range(unit, first, last, &nodes, &raw->count)) {
        raw_unit_close(raw);
        return -1;
    }
    raw->nodes = nodes;
    raw->first = first;
    memmove(raw->cells, raw->cells + first, raw->count * sizeof(*raw->cells));
    return 0;
}

typedef struct {
    RawUnit *raw;
    Boundary *boundary;
    const char *chunk, *pieces;
    int64_t seam;
    int failed, half;
} SeamSide;

static void *seam_side_open(void *arg) {
    SeamSide *side = arg;
    side->failed = raw_unit_open(side->raw, side->chunk, side->seam - (side->half > 0 ? 0 : 4),
                       side->seam + (side->half < 0 ? 0 : 4)) ||
        boundary_read(side->boundary, side->pieces, 0);
    return NULL;
}

typedef struct {
    Node *nodes;
    int32_t *first;
    int64_t *ids, *piece;
    uint8_t *side;
    Edge *edges;
    Pair *pairs;
    float *weights;
    int64_t nodes_count, edge_count, pair_count;
} SeamField;

typedef struct {
    SeamField *field;
    const float *theta, *quality;
    const uint8_t *member;
    int y, x, start;
    int64_t edge_count[8], pair_count[8];
} SeamGraph;

static void seam_plane(void *arg, size_t z) {
    SeamGraph *graph = arg;
    SeamField *field = graph->field;
    int64_t plane = (int64_t)graph->y * graph->x;
    int64_t first = field->first[z * plane];
    int64_t count = field->first[(z + 1) * plane] - first;
    graph->edge_count[z] = hx_sheet_edges(field->nodes, field->first, 8, graph->y, graph->x, (int)z,
        graph->theta, graph->quality, 2.0f, 2.0f, 0.8660254037844386f, 1.9f * HX_CELL,
        field->edges + 13 * first, 13 * count);
    graph->pair_count[z] = hx_radial_pairs(field->nodes, field->first, 8, graph->y, graph->x,
        (int)z, HX_CELL, graph->start, graph->member, 40.0f, 6.0f, 3.0f, RADIAL_RANKS,
        field->pairs + first, count);
}

static int seam_graph(SeamGraph *graph) {
    SeamField *field = graph->field;
    size_t count = (size_t)field->nodes_count;
    field->edges = malloc((count ? 13 * count : 1) * sizeof(*field->edges));
    field->pairs = malloc((count ? count : 1) * sizeof(*field->pairs));
    uint8_t *member = malloc(count ? count : 1);
    int failed = !field->edges || !field->pairs || !member;
    if (!failed) {
        memset(member, 1, count);
        graph->member = member;
        each_block(8, graph, seam_plane);
    }
    free(member);
    int64_t plane = (int64_t)graph->y * graph->x;
    for (size_t z = 0; !failed && z < 8; z++) {
        int64_t edges = graph->edge_count[z], pairs = graph->pair_count[z];
        if (edges < 0 || pairs < 0)
            return -1;
        int64_t first = field->first[z * plane];
        memmove(field->edges + field->edge_count, field->edges + 13 * first,
            (size_t)edges * sizeof(*field->edges));
        memmove(field->pairs + field->pair_count, field->pairs + first,
            (size_t)pairs * sizeof(*field->pairs));
        field->edge_count += edges;
        field->pair_count += pairs;
    }
    return failed ? -1 : 0;
}

static void seam_field_close(SeamField *field) {
    free(field->weights);
    free(field->pairs);
    free(field->edges);
    free(field->side);
    free(field->piece);
    free(field->ids);
    free(field->first);
    free(field->nodes);
    memset(field, 0, sizeof(*field));
}

static int seam_field_open(SeamField *field, const char *lower, const char *upper,
    const char *lower_pieces, const char *upper_pieces, int seam) {
    memset(field, 0, sizeof(*field));
    RawUnit raw[2];
    memset(raw, 0, sizeof(raw));
    Boundary boundary[2];
    memset(boundary, 0, sizeof(boundary));
    int split = !strcmp(lower, upper);
    SeamSide sides[] = {
        {&raw[0], &boundary[0], lower, lower_pieces, seam, 0, -split},
        {&raw[1], &boundary[1], upper, upper_pieces, seam, 0, split},
    };
    pthread_t thread;
    int parallel = pthread_create(&thread, NULL, seam_side_open, &sides[0]) == 0;
    seam_side_open(&sides[1]);
    if (parallel)
        pthread_join(thread, NULL);
    else
        seam_side_open(&sides[0]);
    int failed = sides[0].failed || sides[1].failed || raw[0].grid[1] != raw[1].grid[1] ||
        raw[0].grid[2] != raw[1].grid[2] ||
        (split ? raw[0].low >= seam || raw[0].high <= seam
               : raw[0].high != seam || raw[1].low != seam) ||
        strcmp(raw[0].sample, raw[1].sample) != 0 || strcmp(raw[0].volume, raw[1].volume) != 0;
    int64_t plane_size = (int64_t)raw[0].grid[1] * raw[0].grid[2];
    int64_t count = 0;
    for (int side = 0; !failed && side < 2; side++)
        for (size_t i = 0; i < raw[side].count; i++) {
            int64_t plane = raw[side].low + raw[side].cells[i] / plane_size;
            count += raw[side].nodes[i].prominence >= MIN_PROMINENCE && plane >= seam - 4 &&
                plane < seam + 4;
        }
    int64_t cells = 8 * plane_size;
    field->nodes = malloc((size_t)(count ? count : 1) * sizeof(*field->nodes));
    field->ids = malloc((size_t)(count ? count : 1) * sizeof(*field->ids));
    field->piece = malloc((size_t)(count ? count : 1) * sizeof(*field->piece));
    field->side = malloc((size_t)(count ? count : 1));
    int32_t *cell_of = malloc((size_t)(count ? count : 1) * sizeof(*cell_of));
    field->first = calloc((size_t)(cells + 1), sizeof(*field->first));
    if (failed || !field->nodes || !field->ids || !field->piece || !field->side || !cell_of ||
        !field->first)
        failed = 1;
    int64_t at = 0;
    for (int side = 0; !failed && side < 2; side++)
        for (size_t i = 0; i < raw[side].count; i++) {
            int64_t plane = raw[side].low + raw[side].cells[i] / plane_size;
            if (raw[side].nodes[i].prominence < MIN_PROMINENCE || plane < seam - 4 ||
                plane >= seam + 4)
                continue;
            field->nodes[at] = raw[side].nodes[i];
            field->ids[at] = (int64_t)(((uint64_t)raw[side].row0 << 32) + raw[side].first + i);
            field->piece[at] = boundary_piece(&boundary[side], field->ids[at]);
            field->side[at] = side;
            cell_of[at] =
                (int32_t)((plane - seam + 4) * plane_size + raw[side].cells[i] % plane_size);
            at++;
        }
    field->nodes_count = at;
    if (!failed)
        hx_cell_table(cell_of, count, field->first, cells);
    free(cell_of);
    float *theta = malloc((size_t)(count ? count : 1) * sizeof(*theta));
    float *quality = malloc((size_t)(count ? count : 1) * sizeof(*quality));
    Field axis = {.sample = raw[0].sample ? strdup(raw[0].sample) : NULL,
        .volume = raw[0].volume ? strdup(raw[0].volume) : NULL,
        .level = raw[0].level};
    if (failed || !theta || !quality || !axis.sample || !axis.volume || load_axis(&axis))
        failed = 1;
    for (int64_t i = 0; !failed && i < count; i++) {
        double cx, cy;
        hx_centre_at(axis.axis, axis.axis_count, field->nodes[i].z, &cx, &cy);
        theta[i] = (float)atan2(field->nodes[i].y - cy, field->nodes[i].x - cx);
    }
    if (!failed)
        hx_node_quality(field->nodes, count, 0.65, quality);
    SeamGraph graph = {.field = field,
        .theta = theta,
        .quality = quality,
        .y = raw[0].grid[1],
        .x = raw[0].grid[2],
        .start = seam - 4};
    if (!failed && seam_graph(&graph))
        failed = 1;
    Edge *scratch = malloc((size_t)(field->edge_count ? field->edge_count : 1) * sizeof(*scratch));
    if (!failed && (!field->edges || !scratch))
        failed = 1;
    if (!failed)
        hx_sort_edges(field->edges, field->edge_count, scratch);
    free(scratch);
    field->weights =
        malloc((size_t)(field->pair_count ? field->pair_count : 1) * sizeof(*field->weights));
    if (!failed &&
        (!field->weights ||
            (field->pair_count &&
                hx_ray_weights(field->nodes, count, field->pairs, (size_t)field->pair_count,
                    quality, 0.45, 0.75, field->weights)) ||
            hx_spread_piece_labels(
                field->piece, field->side, count, field->edges, field->edge_count)))
        failed = 1;
    free(axis.axis);
    free(axis.volume);
    free(axis.sample);
    free(quality);
    free(theta);
    for (int i = 0; i < 2; i++) {
        free(boundary[i].rows);
        raw_unit_close(&raw[i]);
    }
    if (failed)
        seam_field_close(field);
    return failed ? -1 : 0;
}

static int write_boundary(
    const char *out, Field *field, Solution *solution, const Flattened *flat) {
    int32_t *label = malloc((size_t)(field->nodes_count ? field->nodes_count : 1) * sizeof(*label));
    int64_t count = 0;
    for (int64_t i = 0; label && i < field->nodes_count; i++) {
        label[i] = -1;
        int64_t plane = (int64_t)floor(field->nodes[i].z / HX_CELL);
        count += plane < field->plane0 + 4 || plane >= field->plane0 + field->grid[0] - 4;
    }
    for (int32_t piece = 0; label && piece < flat->pieces; piece++)
        for (int64_t at = flat->first[piece]; at < flat->first[piece + 1]; at++)
            label[flat->members[at]] = piece;
    int64_t *ids = malloc((size_t)(count ? count : 1) * sizeof(*ids));
    int32_t *pieces = malloc((size_t)(count ? count : 1) * sizeof(*pieces));
    int32_t *potential = malloc((size_t)(count ? count : 1) * sizeof(*potential));
    float *theta = malloc((size_t)(count ? count : 1) * sizeof(*theta));
    int64_t at = 0;
    for (int64_t i = 0; label && ids && pieces && potential && theta && i < field->nodes_count;
        i++) {
        int64_t plane = (int64_t)floor(field->nodes[i].z / HX_CELL);
        if (plane >= field->plane0 + 4 && plane < field->plane0 + field->grid[0] - 4)
            continue;
        ids[at] = field->ids[i];
        pieces[at] = label[i];
        potential[at] = solution->potential[i];
        theta[at++] = field->theta[i];
    }
    char *path = NULL;
    int error = 0;
    Npz archive = {0};
    archive.file = ids && pieces && potential && theta && at == count &&
            asprintf(&path, "%s/boundary.npz", out) >= 0
        ? zip_open(path, ZIP_CREATE | ZIP_TRUNCATE, &error)
        : NULL;
    int64_t shape[] = {count};
    int64_t piece_count = flat->pieces;
    int failed = !archive.file ||
        npz_add(&archive, "ids", "<i8", shape, 1, ids, (size_t)count * sizeof(*ids)) ||
        npz_add(&archive, "piece", "<i4", shape, 1, pieces, (size_t)count * sizeof(*pieces)) ||
        npz_add(&archive, "pieces", "<i8", NULL, 0, &piece_count, sizeof(piece_count)) ||
        npz_add(&archive, "potential", "<i4", shape, 1, potential,
            (size_t)count * sizeof(*potential)) ||
        npz_add(&archive, "theta", "<f4", shape, 1, theta, (size_t)count * sizeof(*theta));
    failed = npz_finish(&archive, path, failed);
    free(path);
    free(theta);
    free(potential);
    free(pieces);
    free(ids);
    free(label);
    return failed;
}

static int write_piece_terms(
    const char *out, Field *field, Solution *solution, const Flattened *flat, int chirality) {
    int32_t *label = malloc((size_t)(field->nodes_count ? field->nodes_count : 1) * sizeof(*label));
    int32_t *level = malloc((size_t)(flat->pieces ? flat->pieces : 1) * sizeof(*level));
    double *turns = calloc((size_t)(flat->pieces ? flat->pieces : 1), sizeof(*turns));
    for (int64_t i = 0; label && i < field->nodes_count; i++)
        label[i] = flat->pieces;
    for (int32_t piece = 0; label && level && turns && piece < flat->pieces; piece++) {
        int32_t node = flat->members[flat->first[piece]];
        int32_t root = solution->root[node];
        level[piece] = solution->level[root] - chirality * solution->potential[root];
        for (int64_t at = flat->first[piece]; at < flat->first[piece + 1]; at++) {
            node = flat->members[at];
            label[node] = piece;
            turns[piece] +=
                solution->potential[node] + (double)(field->theta[node] / (float)(2 * M_PI));
        }
        turns[piece] /= flat->first[piece + 1] - flat->first[piece];
    }
    char *path = NULL;
    int error = 0;
    Npz archive = {0};
    archive.file = label && level && turns && asprintf(&path, "%s/levels.npz", out) >= 0
        ? zip_open(path, ZIP_CREATE | ZIP_TRUNCATE, &error)
        : NULL;
    int64_t piece_shape[] = {flat->pieces};
    int64_t chirality_value = chirality;
    int failed = !archive.file ||
        npz_add(&archive, "level", "<i4", piece_shape, 1, level,
            (size_t)flat->pieces * sizeof(*level)) ||
        npz_add(&archive, "chirality", "<i8", NULL, 0, &chirality_value, sizeof(chirality_value)) ||
        npz_add(
            &archive, "turns", "<f8", piece_shape, 1, turns, (size_t)flat->pieces * sizeof(*turns));
    for (int curl = 1; !failed && curl >= -1; curl -= 2) {
        Terms terms;
        if (terms_open(
                &terms, flat->pieces + 1, label, solution->potential, solution, field, curl)) {
            failed = 1;
            break;
        }
        int64_t real = 0;
        for (int64_t i = 0; i < terms.count; i++)
            real += terms.p[i] >= 0 && terms.p[i] < flat->pieces && terms.q[i] >= 0 &&
                terms.q[i] < flat->pieces;
        int32_t *values = malloc((size_t)(real ? real : 1) * 5 * sizeof(*values));
        int32_t *p = values, *q = p + real, *k = q + real, *up = k + real, *down = up + real;
        int64_t at = 0;
        for (int64_t i = 0; values && i < terms.count; i++) {
            if (terms.p[i] < 0 || terms.p[i] >= flat->pieces || terms.q[i] < 0 ||
                terms.q[i] >= flat->pieces)
                continue;
            p[at] = terms.p[i];
            q[at] = terms.q[i];
            k[at] = terms.k[i];
            up[at] = terms.up[i];
            down[at++] = terms.down[i];
        }
        int64_t shape[] = {real};
        const char *suffix = curl > 0 ? "plus" : "minus";
        char names[5][24];
        snprintf(names[0], sizeof(names[0]), "p_%s", suffix);
        snprintf(names[1], sizeof(names[1]), "q_%s", suffix);
        snprintf(names[2], sizeof(names[2]), "k_%s", suffix);
        snprintf(names[3], sizeof(names[3]), "up_%s", suffix);
        snprintf(names[4], sizeof(names[4]), "dn_%s", suffix);
        int32_t *arrays[] = {p, q, k, up, down};
        failed = !values;
        for (int i = 0; !failed && i < 5; i++)
            failed = npz_add(
                &archive, names[i], "<i4", shape, 1, arrays[i], (size_t)real * sizeof(**arrays));
        free(values);
        terms_close(&terms);
    }
    failed = npz_finish(&archive, path, failed);
    free(path);
    free(turns);
    free(level);
    free(label);
    return failed;
}

static int write_tiff(const char *path, const float *values, int64_t height, int64_t width) {
    TIFF *file = TIFFOpen(path, "w");
    if (!file)
        return -1;
    int ok = TIFFSetField(file, TIFFTAG_IMAGEWIDTH, (uint32_t)width) &&
        TIFFSetField(file, TIFFTAG_IMAGELENGTH, (uint32_t)height) &&
        TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 1) &&
        TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 32) &&
        TIFFSetField(file, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP) &&
        TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
        TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
        TIFFSetField(file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(file, (uint32_t)width * 4));
    for (int64_t row = 0; ok && row < height; row++)
        ok = TIFFWriteScanline(file, (void *)(values + row * width), (uint32_t)row, 0) >= 0;
    TIFFClose(file);
    return ok ? 0 : -1;
}

static json_object *number_array(const double *values, int count) {
    json_object *array = json_object_new_array_ext(count);
    for (int i = 0; array && i < count; i++)
        json_object_array_add(array, json_object_new_double(values[i]));
    return array;
}

enum { READ_BLOCK_CELLS = 112, READ_OVERLAP_CELLS = 16, BLOCK_BASE = 1000000 };

typedef struct {
    int64_t top, left, bottom, right;
} PieceBox;

static json_object *piece_isometry(const Mesh *mesh, const Raster *raster, PieceBox box) {
    int64_t height = box.bottom - box.top, width = box.right - box.left;
    size_t size = (size_t)height * width, input = (size_t)mesh->height * mesh->width;
    double *grid = malloc(3 * size * sizeof(*grid));
    uint8_t *known = malloc(2 * size);
    if (!grid || !known) {
        free(grid);
        free(known);
        return NULL;
    }
    uint8_t *locked = known + size;
    for (int64_t i = 0; i < height; i++) {
        size_t source = (size_t)(box.top + i) * mesh->width + box.left;
        size_t dest = (size_t)i * width;
        memcpy(known + dest, mesh->kept + source, (size_t)width);
        memcpy(locked + dest, raster->locked + source, (size_t)width);
        for (int a = 0; a < 3; a++) {
            memcpy(grid + a * size + dest, mesh->grid + a * input + source,
                (size_t)width * sizeof(*grid));
        }
    }
    double info[2][7] = {{0}};
    int failed = hx_mesh_isometry(grid, height, width, known, locked, HX_CELL, &info[0][0]);
    free(known);
    free(grid);
    if (failed) {
        return NULL;
    }
    json_object *summary = json_object_new_object();
    int isometric = 0;
    for (int k = 0; k < 2; k++) {
        double *v = info[k];
        json_object *row = NULL;
        if (v[0]) {
            row = json_object_new_object();
            double percentiles[3];
            for (int i = 0; i < 3; i++) {
                percentiles[i] = nearbyint(v[i + 2] * 1000.) / 1000.;
            }
            double within = nearbyint(v[5] * 1000.) / 1000.;
            double jitter = nearbyint(v[6] * 1000.) / 1000.;
            json_object_object_add(row, "p5_p50_p95", number_array(percentiles, 3));
            json_object_object_add(row, "within_10pct", json_object_new_double(within));
            json_object_object_add(row, "jitter", v[1] ? json_object_new_double(jitter) : NULL);
            json_object_object_add(row, "pairs", json_object_new_int64((int64_t)v[0]));
            if (k) {
                isometric = within >= .7 && (!v[1] || jitter <= .07);
            }
        }
        json_object_object_add(summary, k ? "locked" : "all", row);
    }
    json_object_object_add(summary, "isometric", json_object_new_boolean(isometric));
    return summary;
}

static json_object *write_piece(Field *field, Solution *solution, Flattened *flat, int32_t piece,
    Raster *raster, Mesh *mesh, const char *out, PieceBox box, PieceBox core, PieceBox whole,
    int64_t id, int64_t core_locked, int block) {
    int64_t cells = mesh->height * mesh->width;
    int64_t top = box.top, left = box.left;
    int64_t height = box.bottom - top, width = box.right - left;
    size_t output_cells = (size_t)height * width;
    uint8_t *quality = calloc(output_cells, 1);
    float *values = malloc(output_cells * sizeof(*values));
    uint8_t *known = malloc(output_cells);
    if (!quality || !values || !known) {
        free(known);
        free(values);
        free(quality);
        return NULL;
    }
    for (int64_t row = 0; row < height; row++)
        for (int64_t col = 0; col < width; col++) {
            int64_t source = (row + top) * mesh->width + col + left;
            int64_t at = row * width + col;
            known[at] = mesh->kept[source];
            quality[at] = known[at] ? (raster->locked[source] ? 1 : 2) : 0;
        }
    double scale = (double)(UINT64_C(1) << field->level);
    double offset = 0.5 * (scale - 1),
           bbox[2][3] = {{DBL_MAX, DBL_MAX, DBL_MAX}, {-DBL_MAX, -DBL_MAX, -DBL_MAX}};
    for (int axis = 0; axis < 3; axis++) {
        double limit = field->shape[2 - axis] * scale;
        for (int64_t row = 0; row < height; row++)
            for (int64_t col = 0; col < width; col++) {
                int64_t source = (row + top) * mesh->width + col + left;
                int64_t at = row * width + col;
                double value = mesh->grid[axis * cells + source] * scale;
                if (known[at] && (value < -1 || value > limit)) {
                    if (value < -9 || value > limit + 8) {
                        free(known);
                        free(values);
                        free(quality);
                        return NULL;
                    }
                    known[at] = quality[at] = 0;
                }
            }
        for (size_t at = 0; at < output_cells; at++)
            if (!known[at])
                quality[at] = 0;
    }
    char *folder = NULL;
    if (asprintf(&folder, "%s/piece-%04lld", out, (long long)id) < 0 || !make_folders(folder)) {
        free(folder);
        free(known);
        free(values);
        free(quality);
        return NULL;
    }
    int failed = 0;
    for (int axis = 0; axis < 3 && !failed; axis++) {
        for (int64_t row = 0; row < height; row++)
            for (int64_t col = 0; col < width; col++) {
                int64_t source = (row + top) * mesh->width + col + left;
                int64_t at = row * width + col;
                values[at] =
                    known[at] ? (float)(mesh->grid[axis * cells + source] * scale + offset) : -1;
                if (known[at]) {
                    double point =
                        block ? mesh->grid[axis * cells + source] * scale + offset : values[at];
                    bbox[0][axis] = fmin(bbox[0][axis], point);
                    bbox[1][axis] = fmax(bbox[1][axis], point);
                }
            }
        char *path = NULL;
        const char names[] = "xyz";
        if (asprintf(&path, "%s/%c.tif", folder, names[axis]) < 0 ||
            write_tiff(path, values, height, width))
            failed = 1;
        free(path);
    }
    char *quality_path = NULL;
    if (asprintf(&quality_path, "%s/quality.npy", folder) < 0 ||
        write_quality(quality_path, quality, height, width))
        failed = 1;
    free(quality_path);
    int64_t begin = flat->first[piece], end = flat->first[piece + 1], locked = 0;
    double *winding = malloc((size_t)(end - begin) * 2 * sizeof(*winding));
    if (!winding) {
        free(folder);
        free(known);
        free(values);
        free(quality);
        return NULL;
    }
    double *radius = winding + (end - begin);
    double turn_min = DBL_MAX, turn_max = -DBL_MAX;
    for (size_t at = 0; at < output_cells; at++) {
        locked += quality[at] == 1;
    }
    if (core_locked < 0) {
        core_locked = locked;
    }
    for (int64_t at = begin; at < end; at++) {
        int64_t j = at - begin;
        int32_t i = flat->members[at];
        double arc =
            block ? (double)(field->theta[i] / (float)(2 * M_PI)) : field->theta[i] / (2 * M_PI);
        double turn = solution->potential[i] + arc;
        winding[j] = solution->level[i];
        double cx, cy;
        hx_centre_at(field->axis, field->axis_count, field->nodes[i].z, &cx, &cy);
        radius[j] = hypot(field->nodes[i].x - cx, field->nodes[i].y - cy);
        turn_min = fmin(turn_min, turn);
        turn_max = fmax(turn_max, turn);
    }
    double side = HX_CELL * scale * field->scan_voxel_um / 1000;
    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "sample", json_object_new_string(field->sample));
    json_object_object_add(meta, "scan", json_object_new_string(field->volume));
    json_object_object_add(meta, "voxel_um", json_object_new_double(field->scan_voxel_um));
    json_object_object_add(meta, "level_of_nodes", json_object_new_int(field->level));
    json_object *shape = json_object_new_array_ext(3);
    for (int i = 0; i < 3; i++)
        json_object_array_add(shape, json_object_new_int64((int64_t)field->shape[i] * scale));
    json_object_object_add(meta, "shape", shape);
    json_object_object_add(meta, "piece", json_object_new_int64(id));
    if (block &&
        (box.top != whole.top || box.left != whole.left || box.bottom != whole.bottom ||
            box.right != whole.right || core.bottom != whole.bottom || core.right != whole.right)) {
        double origin[] = {top - whole.top, left - whole.left};
        double margin[] = {
            core.top - top, box.bottom - core.bottom, core.left - left, box.right - core.right};
        json_object_object_add(meta, "source", json_object_new_int(piece));
        json_object_object_add(meta, "block", number_array(origin, 2));
        json_object_object_add(meta, "overlap", number_array(margin, 4));
    }
    json_object_object_add(
        meta, "winding", json_object_new_int((int)median_double(winding, end - begin)));
    json_object_object_add(meta, "frame", json_object_new_int(0));
    json_object_object_add(meta, "turns_spanned", json_object_new_double(turn_max - turn_min));
    json_object_object_add(meta, "cells", json_object_new_int64(core_locked));
    json_object_object_add(meta, "nodes", json_object_new_int64(end - begin));
    json_object_object_add(
        meta, "crowding", json_object_new_double((double)(end - begin) / raster->occupied));
    json_object_object_add(meta, "area_mm2",
        json_object_new_double(block ? core_locked * (side * side) : locked * side * side));
    json_object_object_add(meta, "width_mm", json_object_new_double(width * side));
    json_object_object_add(meta, "height_mm", json_object_new_double(height * side));
    json_object_object_add(meta, "median_radius_mm",
        json_object_new_double(
            median_double(radius, end - begin) * field->scan_voxel_um * scale / 1000));
    json_object_object_add(meta, "written", json_object_new_boolean(1));
    if (block) {
        json_object *isometry = piece_isometry(mesh, raster, box);
        failed |= !isometry;
        json_object_object_add(meta, "isometry", isometry);
    }
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "type", json_object_new_string("seg"));
    char name[32];
    snprintf(name, sizeof(name), "piece-%04lld", (long long)id);
    json_object_object_add(meta, "uuid", json_object_new_string(name));
    json_object *bounds = json_object_new_array_ext(2);
    json_object_array_add(bounds, number_array(bbox[0], 3));
    json_object_array_add(bounds, number_array(bbox[1], 3));
    json_object_object_add(meta, "bbox", bounds);
    double reciprocal[] = {1.0 / (HX_CELL * scale), 1.0 / (HX_CELL * scale)};
    json_object_object_add(meta, "scale", number_array(reciprocal, 2));
    json_object_object_add(meta, "stride", json_object_new_double(HX_CELL * scale));
    json_object_object_add(
        meta, "locked_share", json_object_new_double((double)locked / output_cells));
    char *meta_path = NULL;
    if (asprintf(&meta_path, "%s/meta.json", folder) < 0 ||
        json_object_to_file_ext(meta_path, meta, JSON_C_TO_STRING_PRETTY))
        failed = 1;
    free(meta_path);
    free(winding);
    free(folder);
    free(known);
    free(values);
    free(quality);
    if (failed) {
        json_object_put(meta);
        return NULL;
    }
    if (block) {
        const char *file_keys[] = {
            "format", "type", "uuid", "bbox", "scale", "stride", "locked_share"};
        for (size_t i = 0; i < sizeof(file_keys) / sizeof(*file_keys); i++) {
            json_object_object_del(meta, file_keys[i]);
        }
    }
    return meta;
}

static void field_close(Field *field) {
    for (int i = 0; i < field->run_count; i++)
        free(field->runs[i].edges);
    free(field->runs);
    free(field->pairs);
    free(field->quality);
    free(field->theta);
    free(field->axis);
    free(field->ids);
    free(field->first);
    free(field->nodes);
    free(field->volume);
    free(field->sample);
    memset(field, 0, sizeof(*field));
}

static int probe(const char *unit) {
    Field field;
    if (prepare(&field, unit, NULL)) {
        fprintf(stderr, "HerculeX: could not prepare native sheet graph %s: %s\n", unit,
            strerror(errno));
        field_close(&field);
        return 1;
    }
    int64_t edges = 0;
    uint64_t node_hash = UINT64_C(14695981039346656037);
    uint64_t edge_hash = node_hash, pair_hash = node_hash;
    node_hash =
        hash_bytes(node_hash, field.nodes, (size_t)field.nodes_count * sizeof(*field.nodes));
    for (int i = 0; i < field.run_count; i++) {
        edges += field.runs[i].count;
        edge_hash = hash_bytes(edge_hash, field.runs[i].edges,
            (size_t)field.runs[i].count * sizeof(*field.runs[i].edges));
    }
    pair_hash = hash_bytes(pair_hash, field.pairs, (size_t)field.pair_count * sizeof(*field.pairs));
    printf("{\"nodes\":%lld,\"edges\":%lld,\"rays\":%lld,\"turned_share\":%.9g,"
           "\"orientation\":{\"opposed_before\":%lld,\"turned_share\":%.9g,"
           "\"domains\":%lld,\"cut_share\":%.9g,\"blocks\":%lld,"
           "\"hash\":\"%016llx\"},"
           "\"nodes_hash\":\"%016llx\",\"edges_hash\":\"%016llx\","
           "\"rays_hash\":\"%016llx\"}\n",
        (long long)field.nodes_count, (long long)edges, (long long)field.pair_count,
        field.turned_share, (long long)field.orientation_report[0], field.orientation_ratios[0],
        (long long)field.orientation_report[1], field.orientation_ratios[1],
        (long long)field.orientation_report[2], (unsigned long long)field.orientation_hash,
        (unsigned long long)node_hash, (unsigned long long)edge_hash,
        (unsigned long long)pair_hash);
    field_close(&field);
    return 0;
}

static int solve_probe(const char *unit, int chirality) {
    Field field;
    Solution solution;
    if (prepare(&field, unit, NULL)) {
        fprintf(stderr, "HerculeX: could not prepare native sheet graph %s: %s\n", unit,
            strerror(errno));
        field_close(&field);
        return 1;
    }
    if (solve_field(&field, &solution, chirality)) {
        fprintf(stderr, "HerculeX: could not solve native sheet graph %s\n", unit);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    uint64_t initial = UINT64_C(14695981039346656037);
    uint64_t edge_hash = hash_bytes(
        initial, solution.attract, (size_t)solution.edge_count * sizeof(*solution.attract));
    uint64_t ray_hash =
        hash_bytes(initial, solution.rays, (size_t)field.pair_count * sizeof(*solution.rays));
    uint64_t root_hash =
        hash_bytes(initial, solution.root, (size_t)field.nodes_count * sizeof(*solution.root));
    uint64_t potential_hash = hash_bytes(
        initial, solution.potential, (size_t)field.nodes_count * sizeof(*solution.potential));
    uint64_t level_hash =
        hash_bytes(initial, solution.level, (size_t)field.nodes_count * sizeof(*solution.level));
    uint64_t kept_hash =
        hash_bytes(initial, solution.kept, (size_t)solution.edge_count * sizeof(*solution.kept));
    printf("{\"nodes\":%lld,\"edges\":%lld,\"rays\":%lld,\"patches\":%d,"
           "\"terms\":%lld,\"stats\":[%lld,%lld,%lld,%lld],"
           "\"edges_hash\":\"%016llx\",\"ray_weight_hash\":\"%016llx\","
           "\"root_hash\":\"%016llx\",\"potential_hash\":\"%016llx\","
           "\"level_hash\":\"%016llx\",\"kept_hash\":\"%016llx\"}\n",
        (long long)field.nodes_count, (long long)solution.edge_count, (long long)field.pair_count,
        solution.patches, (long long)solution.terms, (long long)solution.stats[0],
        (long long)solution.stats[1], (long long)solution.stats[2], (long long)solution.stats[3],
        (unsigned long long)edge_hash, (unsigned long long)ray_hash, (unsigned long long)root_hash,
        (unsigned long long)potential_hash, (unsigned long long)level_hash,
        (unsigned long long)kept_hash);
    solution_close(&solution);
    field_close(&field);
    return 0;
}

static int flatten_probe(const char *unit, int chirality) {
    Field field;
    Solution solution = {0};
    Flattened flat = {0};
    if (prepare(&field, unit, NULL)) {
        fprintf(stderr, "HerculeX: could not prepare native sheet graph %s\n", unit);
        field_close(&field);
        return 1;
    }
    if (solve_field(&field, &solution, chirality) || guard_solution(&field, &solution) ||
        flatten_solution(&field, &solution, &flat)) {
        fprintf(stderr, "HerculeX: could not flatten native sheet graph %s\n", unit);
        flattened_close(&flat);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    uint64_t initial = UINT64_C(14695981039346656037);
    printf("{\"pieces\":%d,\"placed\":%lld,\"refused_share\":%.9g,"
           "\"members_hash\":\"%016llx\","
           "\"first_hash\":\"%016llx\",\"sign_hash\":\"%016llx\","
           "\"u_hash\":\"%016llx\",\"v_hash\":\"%016llx\"}\n",
        flat.pieces, (long long)flat.first[flat.pieces], solution.refused_share,
        (unsigned long long)hash_bytes(
            initial, flat.members, (size_t)flat.first[flat.pieces] * sizeof(*flat.members)),
        (unsigned long long)hash_bytes(
            initial, flat.first, ((size_t)flat.pieces + 1) * sizeof(*flat.first)),
        (unsigned long long)hash_bytes(
            initial, flat.sign, (size_t)field.nodes_count * sizeof(*flat.sign)),
        (unsigned long long)hash_bytes(
            initial, flat.u, (size_t)field.nodes_count * sizeof(*flat.u)),
        (unsigned long long)hash_bytes(
            initial, flat.v, (size_t)field.nodes_count * sizeof(*flat.v)));
    flattened_close(&flat);
    solution_close(&solution);
    field_close(&field);
    return 0;
}

static int raster_probe(const char *unit, int chirality, int32_t piece, int mode) {
    Field field;
    Solution solution = {0};
    Flattened flat = {0};
    Raster raster = {0};
    Mesh mesh = {0};
    double direction = NAN;
    if (prepare(&field, unit, NULL)) {
        field_close(&field);
        return 1;
    }
    if (solve_field(&field, &solution, chirality) || guard_solution(&field, &solution) ||
        flatten_solution(&field, &solution, &flat) ||
        (mode > 2 && upright_piece(&field, &solution, &flat, piece, &direction)) ||
        raster_piece(&field, &flat, piece, &raster) || (mode && contact_layers(&raster)) ||
        (mode > 1 && bridge_holes(&raster, &mesh))) {
        fprintf(stderr, "HerculeX: could not raster native piece %d from %s\n", piece, unit);
        raster_close(&raster);
        flattened_close(&flat);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    uint64_t initial = UINT64_C(14695981039346656037);
    int64_t cells = raster.height * raster.width;
    printf("{\"piece\":%d,\"nodes\":%lld,\"shape\":[%lld,%lld],"
           "\"occupied\":%lld,\"grid_hash\":\"%016llx\","
           "\"locked_hash\":\"%016llx\",\"cell_hash\":\"%016llx\","
           "\"spread_hash\":\"%016llx\"",
        piece, (long long)raster.nodes, (long long)raster.height, (long long)raster.width,
        (long long)raster.occupied,
        (unsigned long long)hash_bytes(
            initial, raster.grid, (size_t)cells * 3 * sizeof(*raster.grid)),
        (unsigned long long)hash_bytes(initial, raster.locked, (size_t)cells),
        (unsigned long long)hash_bytes(
            initial, raster.cell, (size_t)raster.nodes * sizeof(*raster.cell)),
        (unsigned long long)hash_bytes(
            initial, raster.spread, (size_t)cells * sizeof(*raster.spread)));
    if (mode > 1)
        printf(",\"mesh_grid_hash\":\"%016llx\",\"known_hash\":\"%016llx\","
               "\"kept_hash\":\"%016llx\"",
            (unsigned long long)hash_bytes(
                initial, mesh.grid, (size_t)cells * 3 * sizeof(*mesh.grid)),
            (unsigned long long)hash_bytes(initial, mesh.known, (size_t)cells),
            (unsigned long long)hash_bytes(initial, mesh.kept, (size_t)cells));
    if (mode > 2)
        printf(",\"direction\":%.17g", direction);
    puts("}");
    mesh_close(&mesh);
    raster_close(&raster);
    flattened_close(&flat);
    solution_close(&solution);
    field_close(&field);
    return 0;
}

static double level_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static json_object *stretch_stats(
    const Field *field, const Solution *solution, const Flattened *flat) {
    int64_t capacity = solution->edge_count / 97 + 1;
    double *samples = malloc((size_t)capacity * sizeof(*samples));
    if (!samples) {
        return NULL;
    }
    int64_t held = 0, sampled = 0;
    for (int64_t i = 0; i < solution->edge_count; i++) {
        Edge e = solution->attract[i];
        if (solution->root[e.a] != solution->root[e.b] ||
            solution->potential[e.b] - solution->potential[e.a] != e.step) {
            continue;
        }
        if (held++ % 97 || !flat->sign[e.a]) {
            continue;
        }
        Node a = field->nodes[e.a], b = field->nodes[e.b];
        double x = (double)a.x - b.x, y = (double)a.y - b.y, z = (double)a.z - b.z;
        double step = hypotf(flat->u[e.a] - flat->u[e.b], flat->v[e.a] - flat->v[e.b]);
        samples[sampled++] = step / fmax(sqrt(x * x + y * y + z * z), 1e-6);
    }
    qsort(samples, (size_t)sampled, sizeof(*samples), double_before);
    const double q[] = {.05, .5, .95};
    double percentiles[3];
    for (int i = 0; i < 3; i++) {
        if (!sampled) {
            percentiles[i] = NAN;
        } else {
            double at = q[i] * (sampled - 1);
            int64_t lo = (int64_t)at, hi = lo + (lo + 1 < sampled);
            double fraction = at - lo, delta = samples[hi] - samples[lo];
            percentiles[i] = fraction < .5 ? samples[lo] + delta * fraction
                                           : samples[hi] - delta * (1 - fraction);
        }
    }
    free(samples);
    return number_array(percentiles, 3);
}

static json_object *unit_isometry(json_object *pieces) {
    size_t count = json_object_array_length(pieces), measured = 0, yes = 0;
    double *data = malloc((count ? count : 1) * 3 * sizeof(*data));
    if (!data) {
        return NULL;
    }
    for (size_t i = 0; i < count; i++) {
        json_object *item = json_object_array_get_idx(pieces, i);
        json_object *iso = json_object_object_get(item, "isometry");
        json_object *locked = json_object_object_get(iso, "locked");
        json_object *all = json_object_object_get(iso, "all");
        if (!locked) {
            continue;
        }
        yes += json_object_get_boolean(json_object_object_get(iso, "isometric"));
        data[measured] = json_object_get_double(json_object_object_get(locked, "within_10pct"));
        data[count + measured] =
            json_object_get_double(json_object_object_get(all, "within_10pct"));
        data[2 * count + measured++] =
            json_object_get_double(json_object_object_get(locked, "jitter"));
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "pieces", json_object_new_int64(measured));
    if (measured) {
        json_object_object_add(out, "isometric_share",
            json_object_new_double(nearbyint(1000. * yes / measured) / 1000.));
        const char *keys[] = {"median_within_10pct", "median_all_within_10pct", "median_jitter"};
        for (int i = 0; i < 3; i++) {
            double median = median_double(data + i * count, measured);
            json_object_object_add(
                out, keys[i], json_object_new_double(nearbyint(median * 1000) / 1000));
        }
    }
    free(data);
    return out;
}

#ifdef HX_UNITY
int hx_synthetic_raw_nodes(const char *unit, void **nodes, int32_t **cells, size_t *count) {
    size_t n = 0;
    if (hx_node_read(unit, nodes, count)) {
        return 1;
    }
    if (hx_cell_read(unit, cells, &n) || n != *count) {
        free(*nodes);
        return 1;
    }
    return 0;
}

static int write_synthetic_edges(const Field *field, const char *out) {
    int64_t edges = 0;
    for (int i = 0; i < field->run_count; i++) {
        edges += field->runs[i].count;
    }
    size_t entries = (size_t)(edges ? edges : 1);
    int32_t *a = malloc(2 * entries * sizeof(*a)), *b = a ? a + entries : NULL;
    for (int i = 0, at = 0; a && i < field->run_count; i++) {
        for (int64_t j = 0; j < field->runs[i].count; j++, at++) {
            a[at] = field->runs[i].edges[j].a;
            b[at] = field->runs[i].edges[j].b;
        }
    }
    Npz archive = {0};
    char *path = NULL;
    int error;
    if (a && asprintf(&path, "%s/edges.npz", out) >= 0) {
        archive.file = zip_open(path, ZIP_CREATE | ZIP_TRUNCATE, &error);
    }
    int failed = !archive.file || npz_add(&archive, "a", "<i4", &edges, 1, a, edges * sizeof(*a)) ||
        npz_add(&archive, "b", "<i4", &edges, 1, b, edges * sizeof(*b));
    failed = npz_finish(&archive, path, failed);
    free(a);
    free(path);
    return failed;
}

int hx_synthetic_bench_geometry(
    const char *unit, const char *out, const double *line, size_t knots, float *xyz, size_t count) {
    Field field = {0};
    int failed = load_nodes_with_axis(&field, unit, out && !knots);
    if (!failed && out && knots) {
        field.axis = calloc(knots, sizeof(*field.axis));
        field.axis_count = knots;
        double scale = ldexp(1, field.level);
        for (size_t k = 0; field.axis && k < knots; k++) {
            field.axis[k] = (HxAxisKnot){.z = line[3 * k] / scale,
                .x = line[3 * k + 1] / scale,
                .y = line[3 * k + 2] / scale};
        }
        failed = !field.axis;
    }
    if (!failed && out) {
        failed = build_edges(&field) || turn_normals(&field) || write_synthetic_edges(&field, out);
    }
    failed |= count != (size_t)field.nodes_count;
    for (size_t i = 0; !failed && i < count; i++) {
        xyz[3 * i] = field.nodes[i].x;
        xyz[3 * i + 1] = field.nodes[i].y;
        xyz[3 * i + 2] = field.nodes[i].z;
    }
    field_close(&field);
    return failed;
}

static int read_curl_value(const char *sample, const char *scan, int *curl);

static json_object *synthetic_level_fit(const char *unit, const char *out, const double *line,
    size_t knots, int curl, float **positions, const int *planes) {
    Field field = {0};
    Solution solution = {0};
    Flattened flat = {0};
    int failed = load_nodes_with_axis(&field, unit, line == NULL);
    double scale = ldexp(1, field.level);
    if (!failed && line) {
        field.axis = calloc(knots, sizeof(*field.axis));
        field.axis_count = knots;
        for (size_t k = 0; field.axis && k < knots; k++) {
            field.axis[k] = (HxAxisKnot){.z = line[3 * k] / scale,
                .x = line[3 * k + 1] / scale,
                .y = line[3 * k + 2] / scale};
        }
    }
    if (!failed && planes) {
        failed = field_clip(&field, planes[0], planes[1]) ||
            read_curl_value(field.sample, field.volume, &curl);
    }
    failed = failed || !field.axis || build_edges(&field) || turn_normals(&field) ||
        cast_rays(&field) || solve_field(&field, &solution, curl) ||
        guard_solution(&field, &solution);
    size_t n = (size_t)field.nodes_count;
    *positions = failed ? NULL : malloc((n ? n : 1) * 4 * sizeof(**positions));
    failed |= !*positions;
    if (!failed) {
        float *xyz = *positions;
        double cell = HX_CELL * scale * field.scan_voxel_um / 1000;
        hx_node_area(field.nodes, n, (float)(cell * cell), xyz + 3 * n);
        for (size_t i = 0; i < n; i++) {
            xyz[3 * i] = field.nodes[i].x;
            xyz[3 * i + 1] = field.nodes[i].y;
            xyz[3 * i + 2] = field.nodes[i].z;
        }
    }
    int64_t *cursor = malloc((n / 400 + 1) * sizeof(*cursor));
    int32_t *label = malloc((n ? n : 1) * sizeof(*label));
    failed = failed ||
        piece_members(
            &solution, field.nodes_count, default_guard.min_nodes, &flat, label, cursor) ||
        write_piece_terms(out, &field, &solution, &flat, curl);
    free(label);
    free(cursor);
    int8_t *normal = malloc((n ? n : 1) * 3);
    for (size_t i = 0; !failed && normal && i < n; i++) {
        normal[3 * i] = field.nodes[i].nx;
        normal[3 * i + 1] = field.nodes[i].ny;
        normal[3 * i + 2] = field.nodes[i].nz;
    }
    char *path = NULL;
    Npz archive = {0};
    int error;
    if (!failed && normal && asprintf(&path, "%s/nodes.npz", out) >= 0) {
        archive.file = zip_open(path, ZIP_CREATE | ZIP_TRUNCATE, &error);
    }
    int64_t shape[] = {(int64_t)n, 3};
    failed = failed || !archive.file ||
        npz_add(&archive, "ids", "<i8", shape, 1, field.ids, n * sizeof(*field.ids)) ||
        npz_add(&archive, "root", "<i4", shape, 1, solution.root, n * sizeof(*solution.root)) ||
        npz_add(&archive, "potential", "<i4", shape, 1, solution.potential,
            n * sizeof(*solution.potential)) ||
        npz_add(&archive, "theta", "<f4", shape, 1, field.theta, n * sizeof(*field.theta)) ||
        npz_add(
            &archive, "node_level", "<i4", shape, 1, solution.level, n * sizeof(*solution.level)) ||
        npz_add(&archive, "normal", "|i1", shape, 2, normal, n * 3);
    failed = npz_finish(&archive, path, failed);
    free(path);
    free(normal);
    failed = failed || write_synthetic_edges(&field, out);
    flattened_close(&flat);
    solution_close(&solution);
    field_close(&field);
    if (failed) {
        free(*positions);
        *positions = NULL;
        return NULL;
    }
    json_object *fit = json_object_new_object(), *centres = json_object_new_object();
    json_object_object_add(centres, "rule", json_object_new_string("umbilicus"));
    json_object_object_add(fit, "nodes", json_object_new_int64(n));
    json_object_object_add(fit, "chirality", json_object_new_int(curl));
    json_object_object_add(fit, "centres", centres);
    return fit;
}

json_object *hx_synthetic_level_fit(const char *unit, const char *out, const double *line,
    size_t knots, int curl, float **positions) {
    return synthetic_level_fit(unit, out, line, knots, curl, positions, NULL);
}

json_object *hx_solver_window_fit(const char *unit, const char *out, int first, int last) {
    const int planes[] = {first, last};
    float *positions;
    json_object *fit = synthetic_level_fit(unit, out, NULL, 0, 0, &positions, planes);
    free(positions);
    return fit;
}

#endif

static int write_unit_window(
    const char *unit, const char *out, int chirality, int blocks, const int *clip, int gap_report) {
    Field field = {0};
    Solution solution = {0};
    Flattened flat = {0};
    char method[13];
    double times[6];
    times[0] = level_clock();
    int failed = method_id(method) ||
        (clip ? load_nodes(&field, unit) || field_clip(&field, clip[0], clip[1]) ||
                    build_edges(&field) || turn_normals(&field) || cast_rays(&field)
              : prepare(&field, unit, NULL));
    times[1] = level_clock();
    failed =
        failed || solve_field(&field, &solution, chirality) || guard_solution(&field, &solution);
    times[2] = level_clock();
    failed = failed || flatten_solution(&field, &solution, &flat) || !make_folders(out);
    times[3] = level_clock();
    if (failed) {
        fprintf(stderr, "HerculeX: could not prepare native piece output %s\n", out);
        flattened_close(&flat);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    if (write_boundary(out, &field, &solution, &flat) ||
        write_piece_terms(out, &field, &solution, &flat, chirality)) {
        fprintf(stderr, "HerculeX: could not write native seam artifacts to %s\n", out);
        flattened_close(&flat);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    double gap = NAN;
    json_object *stretch = blocks ? stretch_stats(&field, &solution, &flat) : NULL;
    if (blocks || gap_report) {
        failed =
            hx_sampled_gap(field.nodes, field.nodes_count, field.pairs, field.pair_count, &gap);
        gap *= field.scan_voxel_um * (UINT64_C(1) << field.level);
    }
    times[4] = level_clock();
    json_object *pieces = json_object_new_array();
    double area = 0, area_1cm = 0, largest = 0;
    int pieces_1cm = 0, pieces_4cm = 0;
    for (int32_t piece = 0; !failed && piece < flat.pieces; piece++) {
        Raster raster = {0};
        Mesh mesh = {0};
        double direction;
        failed = upright_piece(&field, &solution, &flat, piece, &direction) ||
            raster_piece(&field, &flat, piece, &raster) || contact_layers(&raster) ||
            bridge_holes(&raster, &mesh);
        PieceBox kept = {mesh.height, mesh.width, 0, 0};
        for (int64_t r = 0; !failed && r < mesh.height; r++) {
            for (int64_t c = 0; c < mesh.width; c++) {
                if (mesh.kept[r * mesh.width + c]) {
                    kept.top = kept.top < r ? kept.top : r;
                    kept.left = kept.left < c ? kept.left : c;
                    kept.bottom = kept.bottom > r + 1 ? kept.bottom : r + 1;
                    kept.right = kept.right > c + 1 ? kept.right : c + 1;
                }
            }
        }
        int split = blocks &&
            (kept.bottom - kept.top > READ_BLOCK_CELLS ||
                kept.right - kept.left > READ_BLOCK_CELLS);
        int64_t step = split ? READ_BLOCK_CELLS : mesh.height + mesh.width;
        int64_t sequence = 0;
        for (int64_t r = kept.top; !failed && r < kept.bottom; r += step) {
            for (int64_t c = kept.left; !failed && c < kept.right; c += step) {
                PieceBox core = {r, c, r + step < kept.bottom ? r + step : kept.bottom,
                    c + step < kept.right ? c + step : kept.right};
                PieceBox box = kept;
                int64_t locked = blocks ? 0 : -1;
                if (blocks) {
                    for (int64_t i = core.top; i < core.bottom; i++) {
                        for (int64_t j = core.left; j < core.right; j++) {
                            locked += raster.locked[i * mesh.width + j];
                        }
                    }
                    if (!locked) {
                        continue;
                    }
                }
                if (split) {
                    int64_t pad = READ_OVERLAP_CELLS;
                    box = (PieceBox){r - pad > kept.top ? r - pad : kept.top,
                        c - pad > kept.left ? c - pad : kept.left,
                        core.bottom + pad < kept.bottom ? core.bottom + pad : kept.bottom,
                        core.right + pad < kept.right ? core.right + pad : kept.right};
                }
                int64_t id = sequence++ * BLOCK_BASE + piece;
                json_object *row = write_piece(&field, &solution, &flat, piece, &raster, &mesh, out,
                    box, core, kept, id, locked, blocks);
                if (!row) {
                    failed = 1;
                    break;
                }
                double size = json_object_get_double(json_object_object_get(row, "area_mm2"));
                area += size;
                largest = fmax(largest, size);
                if (size >= 100) {
                    area_1cm += size;
                    pieces_1cm++;
                }
                pieces_4cm += size >= 400;
                json_object_array_add(pieces, row);
            }
        }
        mesh_close(&mesh);
        raster_close(&raster);
        if (failed) {
            fprintf(stderr, "HerculeX: could not write native piece %d to %s\n", piece, out);
        }
    }
    if (failed) {
        json_object_put(stretch);
        json_object_put(pieces);
        flattened_close(&flat);
        solution_close(&solution);
        field_close(&field);
        return 1;
    }
    times[5] = level_clock();
    json_object *record = json_object_new_object();
    if (blocks) {
        json_object *seconds = json_object_new_object();
        const char *key[] = {"prepare", "grow", "flatten", "order", "write"};
        for (int i = 0; i < 5; i++) {
            double spent = nearbyint((times[i + 1] - times[i]) * 10) / 10;
            json_object_object_add(seconds, key[i], json_object_new_double(spent));
        }
        json_object_object_add(record, "seconds", seconds);
        json_object_object_add(record, "stretch_p5_p50_p95", stretch);
        json_object_object_add(record, "isometry", unit_isometry(pieces));
    }
    if (blocks || gap_report) {
        json_object_object_add(record, "gap_um", json_object_new_double(nearbyint(gap * 10) / 10));
    }
    json_object_object_add(record, "sample", json_object_new_string(field.sample));
    json_object_object_add(record, "volume", json_object_new_string(field.volume));
    json_object_object_add(record, "voxel_um", json_object_new_double(field.scan_voxel_um));
    json_object_object_add(record, "level", json_object_new_int(field.level));
    json_object *chunks = json_object_new_array();
    const char *name = strrchr(unit, '/');
    json_object_array_add(chunks, json_object_new_string(name ? name + 1 : unit));
    json_object_object_add(record, "chunks", chunks);
    double planes[] = {field.plane0, field.plane0 + field.grid[0]};
    json_object_object_add(record, "cell_planes", number_array(planes, 2));
    json_object *shared = json_object_new_array_ext(2);
    double lower[] = {field.plane0, field.plane0 + 4};
    double upper[] = {field.plane0 + field.grid[0] - 4, field.plane0 + field.grid[0]};
    json_object_array_add(shared, number_array(lower, 2));
    json_object_array_add(shared, number_array(upper, 2));
    json_object_object_add(record, "shared", shared);
    json_object_object_add(record, "chirality", json_object_new_int(chirality));
    json_object_object_add(record, "ordered_pieces", json_object_new_int(0));
    json_object_object_add(record, "order_terms", json_object_new_int(0));
    json_object_object_add(record, "frame_share", json_object_new_double(1));
    int32_t minimum = 0, maximum = 0;
    if (field.nodes_count) {
        minimum = solution.level[0];
        maximum = solution.level[0];
        for (int64_t i = 1; i < field.nodes_count; i++) {
            minimum = solution.level[i] < minimum ? solution.level[i] : minimum;
            maximum = solution.level[i] > maximum ? solution.level[i] : maximum;
        }
    }
    json_object_object_add(record, "turns_in_frame", json_object_new_int(maximum - minimum));
    json_object_object_add(record, "nodes", json_object_new_int64(field.nodes_count));
    json_object_object_add(record, "placed",
        json_object_new_double(
            field.nodes_count ? (double)flat.first[flat.pieces] / field.nodes_count : NAN));
    json_object_object_add(record, "pieces", pieces);
    json_object_object_add(record, "area_cm2", json_object_new_double(area / 100));
    json_object_object_add(record, "area_in_1cm2_pieces", json_object_new_double(area_1cm / 100));
    json_object_object_add(record, "pieces_1cm2", json_object_new_int(pieces_1cm));
    json_object_object_add(record, "pieces_4cm2", json_object_new_int(pieces_4cm));
    json_object_object_add(record, "largest_mm2", json_object_new_double(largest));
    json_object_object_add(record, "normals_turned", json_object_new_double(field.turned_share));
    json_object *stats = json_object_new_array_ext(4);
    for (int i = 0; i < 4; i++)
        json_object_array_add(stats, json_object_new_int64(solution.stats[i]));
    json_object_object_add(record, "stats", stats);
    json_object *route = json_object_new_object();
    json_object_object_add(route, "patches", json_object_new_int(solution.patches));
    json_object_object_add(route, "terms", json_object_new_int64(solution.terms));
    json_object_object_add(route, "energy", json_object_new_int64(solution.energy));
    json_object_object_add(route, "cuts", json_object_new_int64(solution.cuts));
    json_object_object_add(route, "kept", json_object_new_double(solution.kept_share));
    json_object_object_add(route, "firm", json_object_new_double(solution.firm_share));
    json_object_object_add(route, "refused_share", json_object_new_double(solution.refused_share));
    if (blocks) {
        json_object *sections = json_object_new_array_ext(solution.guard_count);
        for (int i = 0; i < solution.guard_count; i++) {
            json_object_array_add(sections, number_array(solution.guard[i], 2));
        }
        json_object_object_add(route, "pieces", sections);
        json_object *timing = json_object_new_object();
        const char *names[] = {
            "quick pieces", "start", "terms", "solve", "draws", "firm pieces", "guard"};
        for (int i = 0; i < 7; i++) {
            double seconds = nearbyint(solution.timings[i] * 10.) / 10.;
            json_object_object_add(timing, names[i], json_object_new_double(seconds));
        }
        json_object_object_add(route, "seconds", timing);
    }
    json_object_object_add(record, "level_route", route);
    json_object *orientation = json_object_new_object();
    json_object_object_add(
        orientation, "opposed_before", json_object_new_int64(field.orientation_report[0]));
    json_object_object_add(
        orientation, "turned_share", json_object_new_double(field.orientation_ratios[0]));
    json_object_object_add(
        orientation, "domains", json_object_new_int64(field.orientation_report[1]));
    json_object_object_add(
        orientation, "cut_share", json_object_new_double(field.orientation_ratios[1]));
    json_object_object_add(
        orientation, "blocks", json_object_new_int64(field.orientation_report[2]));
    json_object_object_add(record, "orientation", orientation);
    json_object *centres = json_object_new_object();
    json_object_object_add(centres, "rule", json_object_new_string("umbilicus"));
    json_object_object_add(record, "centres", centres);
    json_object_object_add(record, "method", json_object_new_string(method));
    char *path = NULL;
    failed = asprintf(&path, "%s/record.json", out) < 0 ||
        json_object_to_file_ext(path, record, JSON_C_TO_STRING_PRETTY);
    free(path);
    json_object_put(record);
    flattened_close(&flat);
    solution_close(&solution);
    field_close(&field);
    return failed ? 1 : 0;
}

static int write_unit(const char *unit, const char *out, int chirality, int blocks) {
    return write_unit_window(unit, out, chirality, blocks, NULL, 0);
}

static int join_probe(int argc, char **argv) {
    char *end;
    errno = 0;
    long seam = argc == 7 ? strtol(argv[6], &end, 10) : 0;
    if (argc != 7 || errno || *end || seam < INT_MIN || seam > INT_MAX)
        return 2;
    SeamField field;
    if (seam_field_open(&field, argv[2], argv[3], argv[4], argv[5], (int)seam)) {
        fputs("HerculeX: could not build the native seam field\n", stderr);
        return 1;
    }
    int64_t count = hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges,
        field.edge_count, field.pairs, field.weights, field.pair_count, NULL, 0);
    HxJoin *joins = count >= 0 ? malloc((size_t)(count ? count : 1) * sizeof(*joins)) : NULL;
    if (!joins ||
        hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges, field.edge_count,
            field.pairs, field.weights, field.pair_count, joins, count) != count) {
        free(joins);
        seam_field_close(&field);
        return 1;
    }
    printf("{\"nodes\":%lld,\"edges\":%lld,\"rays\":%lld,\"joins\":[", (long long)field.nodes_count,
        (long long)field.edge_count, (long long)field.pair_count);
    for (int64_t i = 0; i < count; i++)
        printf("%s[%lld,%lld,%lld]", i ? "," : "", (long long)joins[i].lower,
            (long long)joins[i].upper, (long long)joins[i].count);
    puts("]}");
    free(joins);
    seam_field_close(&field);
    return 0;
}

static int32_t join_root(int32_t *parent, int32_t node) {
    while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    return node;
}

static int seam_end_plane(const char *lower) {
    char *path = NULL;
    if (asprintf(&path, "%s/record.json", lower) < 0)
        return INT_MIN;
    json_object *record = json_object_from_file(path), *planes;
    free(path);
    int seam = record && json_object_object_get_ex(record, "cell_planes", &planes) &&
            json_object_array_length(planes) == 2
        ? json_object_get_int(json_object_array_get_idx(planes, 1))
        : INT_MIN;
    if (record)
        json_object_put(record);
    return seam;
}

static int write_joints(
    const char *node_folder, const char *piece_folder, const char *out, const char *method) {
    glob_t chunks = {0};
    if (!node_rows(node_folder, &chunks)) {
        globfree(&chunks);
        return 1;
    }
    size_t units = chunks.gl_pathc;
    char **pieces = calloc(units, sizeof(*pieces));
    int64_t *offset = calloc(units + 1, sizeof(*offset));
    int64_t *join_count = calloc(units - 1, sizeof(*join_count));
    if (!pieces || !offset || !join_count) {
        free(join_count);
        free(offset);
        free(pieces);
        globfree(&chunks);
        return 1;
    }
    int failed = 0;
    for (size_t i = 0; !failed && i < units; i++) {
        const char *name = strrchr(chunks.gl_pathv[i], '/');
        if (asprintf(&pieces[i], "%s/%s", piece_folder, name ? name + 1 : chunks.gl_pathv[i]) < 0)
            failed = 1;
        Boundary boundary = {0};
        if (!failed && boundary_open(&boundary, pieces[i]))
            failed = 1;
        if (!failed && (boundary.pieces < 0 || boundary.pieces > INT32_MAX - offset[i]))
            failed = 1;
        if (!failed)
            offset[i + 1] = offset[i] + boundary.pieces;
        free(boundary.rows);
    }
    int64_t total = offset[units];
    int32_t *parent = !failed && total <= INT32_MAX
        ? malloc((size_t)(total ? total : 1) * sizeof(*parent))
        : NULL;
    if (!parent)
        failed = 1;
    for (int32_t i = 0; !failed && i < total; i++)
        parent[i] = i;
    for (size_t seam_index = 0; !failed && seam_index + 1 < units; seam_index++) {
        int seam = seam_end_plane(chunks.gl_pathv[seam_index]);
        SeamField field = {0};
        if (seam == INT_MIN ||
            seam_field_open(&field, chunks.gl_pathv[seam_index], chunks.gl_pathv[seam_index + 1],
                pieces[seam_index], pieces[seam_index + 1], seam)) {
            failed = 1;
            break;
        }
        int64_t count = hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges,
            field.edge_count, field.pairs, field.weights, field.pair_count, NULL, 0);
        HxJoin *joins = count >= 0 ? malloc((size_t)(count ? count : 1) * sizeof(*joins)) : NULL;
        if (!joins ||
            hx_piece_joins(field.piece, field.side, field.nodes_count, field.edges,
                field.edge_count, field.pairs, field.weights, field.pair_count, joins,
                count) != count) {
            failed = 1;
        } else {
            join_count[seam_index] = count;
            for (int64_t i = 0; i < count; i++) {
                if (joins[i].lower < 0 || joins[i].upper < 0 ||
                    joins[i].lower >= offset[seam_index + 1] - offset[seam_index] ||
                    joins[i].upper >= offset[seam_index + 2] - offset[seam_index + 1]) {
                    failed = 1;
                    break;
                }
                int32_t a = join_root(parent, (int32_t)(offset[seam_index] + joins[i].lower));
                int32_t b = join_root(parent, (int32_t)(offset[seam_index + 1] + joins[i].upper));
                if (a < b)
                    parent[b] = a;
                else
                    parent[a] = b;
            }
        }
        free(joins);
        seam_field_close(&field);
    }
    int32_t sheets = 0;
    for (int32_t i = 0; !failed && i < total; i++)
        parent[i] = i == parent[i] ? sheets++ : parent[parent[i]];
    char *array_path = NULL, *record_path = NULL;
    if (!failed &&
        (!make_folders(out) || asprintf(&array_path, "%s/sheets.npy", out) < 0 ||
            asprintf(&record_path, "%s/record.json", out) < 0))
        failed = 1;
    int64_t shape[] = {total};
    if (!failed && write_npy(array_path, "<i4", shape, 1, parent, (size_t)total * sizeof(*parent)))
        failed = 1;
    json_object *summary = json_object_new_object();
    json_object *unit_rows = json_object_new_array_ext((int)units);
    json_object *piece_rows = json_object_new_array_ext((int)units);
    json_object *join_rows = json_object_new_array_ext((int)(units - 1));
    for (size_t i = 0; !failed && i < units; i++) {
        const char *name = strrchr(pieces[i], '/');
        json_object_array_add(unit_rows, json_object_new_string(name ? name + 1 : pieces[i]));
        json_object_array_add(piece_rows, json_object_new_int64(offset[i + 1] - offset[i]));
        if (i + 1 < units)
            json_object_array_add(join_rows, json_object_new_int64(join_count[i]));
    }
    json_object_object_add(summary, "units", unit_rows);
    json_object_object_add(summary, "pieces", piece_rows);
    json_object_object_add(summary, "joins", join_rows);
    json_object_object_add(summary, "sheets", json_object_new_int(sheets));
    json_object_object_add(summary, "method", json_object_new_string(failed ? "" : method));
    if (!failed && json_object_to_file_ext(record_path, summary, JSON_C_TO_STRING_PRETTY))
        failed = 1;
    json_object_put(summary);
    free(record_path);
    free(array_path);
    free(parent);
    for (size_t i = 0; i < units; i++)
        free(pieces[i]);
    free(join_count);
    free(offset);
    free(pieces);
    globfree(&chunks);
    return failed ? 1 : 0;
}

typedef struct {
    int32_t *level, *p, *q, *k, *up, *down;
    double *turns;
    int64_t count, terms;
    int chirality;
} FrameUnit;

static void frame_unit_close(FrameUnit *unit) {
    free(unit->down);
    free(unit->up);
    free(unit->k);
    free(unit->q);
    free(unit->p);
    free(unit->turns);
    free(unit->level);
    memset(unit, 0, sizeof(*unit));
}

static int frame_unit_open(FrameUnit *unit, const char *folder) {
    memset(unit, 0, sizeof(*unit));
    char *path = NULL;
    if (asprintf(&path, "%s/levels.npz", folder) < 0)
        return -1;
    NpzReader reader = {0};
    int64_t chirality_count = 0, turns_count = 0, p_count = 0, q_count = 0, k_count = 0,
            up_count = 0, down_count = 0;
    int64_t *chirality = NULL;
    int failed = npz_open_reader(&reader, path) ||
        npz_read(
            &reader, "level", "<i4", sizeof(*unit->level), (void **)&unit->level, &unit->count) ||
        npz_read(
            &reader, "turns", "<f8", sizeof(*unit->turns), (void **)&unit->turns, &turns_count) ||
        npz_read(&reader, "chirality", "<i8", sizeof(*chirality), (void **)&chirality,
            &chirality_count) ||
        chirality_count != 1 || turns_count != unit->count || abs((int)*chirality) != 1;
    unit->chirality = failed ? 0 : (int)*chirality;
    const char *suffix = unit->chirality > 0 ? "plus" : "minus";
    char names[5][24];
    snprintf(names[0], sizeof(names[0]), "p_%s", suffix);
    snprintf(names[1], sizeof(names[1]), "q_%s", suffix);
    snprintf(names[2], sizeof(names[2]), "k_%s", suffix);
    snprintf(names[3], sizeof(names[3]), "up_%s", suffix);
    snprintf(names[4], sizeof(names[4]), "dn_%s", suffix);
    if (!failed)
        failed =
            npz_read(&reader, names[0], "<i4", sizeof(*unit->p), (void **)&unit->p, &p_count) ||
            npz_read(&reader, names[1], "<i4", sizeof(*unit->q), (void **)&unit->q, &q_count) ||
            npz_read(&reader, names[2], "<i4", sizeof(*unit->k), (void **)&unit->k, &k_count) ||
            npz_read(&reader, names[3], "<i4", sizeof(*unit->up), (void **)&unit->up, &up_count) ||
            npz_read(
                &reader, names[4], "<i4", sizeof(*unit->down), (void **)&unit->down, &down_count) ||
            p_count != q_count || p_count != k_count || p_count != up_count ||
            p_count != down_count;
    unit->terms = failed ? 0 : p_count;
    free(chirality);
    npz_close_reader(&reader);
    free(path);
    if (failed)
        frame_unit_close(unit);
    return failed ? -1 : 0;
}

typedef struct {
    int32_t p, q, rise;
    double turn, up, down;
} SeamCross;

typedef struct {
    int32_t p, q, k, up, down;
    int64_t order;
} FrameTerm;

typedef struct {
    double cx, cy, dx, dy, half;
} Segment;

static int unit_segment(const char *unit, int last, Segment *segment) {
    char *path = NULL;
    if (asprintf(&path, "%s/record.json", unit) < 0)
        return -1;
    json_object *record = json_object_from_file(path), *centres, *rule, *segments;
    free(path);
    int found = record && json_object_object_get_ex(record, "centres", &centres) &&
        json_object_object_get_ex(centres, "rule", &rule) &&
        !strcmp(json_object_get_string(rule), "segment") &&
        json_object_object_get_ex(centres, "segments", &segments) &&
        json_object_array_length(segments);
    if (found) {
        size_t index = last ? json_object_array_length(segments) - 1 : 0;
        json_object *row = json_object_array_get_idx(segments, index);
        found = json_object_is_type(row, json_type_array) && json_object_array_length(row) >= 6;
        if (found) {
            double *values = &segment->cx;
            for (int i = 0; i < 5; i++)
                values[i] = json_object_get_double(json_object_array_get_idx(row, (size_t)i + 1));
        }
    }
    if (record)
        json_object_put(record);
    return found;
}

static int shared_segments(const char *lower, const char *upper, Segment sides[2], Segment *mean) {
    int a = unit_segment(lower, 1, &sides[0]);
    int b = unit_segment(upper, 0, &sides[1]);
    if (a < 0 || b < 0)
        return -1;
    if (!a || !b)
        return 0;
    if (sides[0].dx * sides[1].dx + sides[0].dy * sides[1].dy < 0) {
        sides[1].dx = -sides[1].dx;
        sides[1].dy = -sides[1].dy;
    }
    double *target = &mean->cx, *left = &sides[0].cx, *right = &sides[1].cx;
    for (int i = 0; i < 5; i++)
        target[i] = (left[i] + right[i]) * 0.5;
    double length = fmax(hypot(mean->dx, mean->dy), 1e-9);
    mean->dx /= length;
    mean->dy /= length;
    return 1;
}

static double segment_angle(double x, double y, const Segment *segment) {
    double t = (x - segment->cx) * segment->dx + (y - segment->cy) * segment->dy;
    t = fmax(-segment->half, fmin(segment->half, t));
    return atan2(y - (segment->cy + t * segment->dy), x - (segment->cx + t * segment->dx));
}

static double boundary_turn(const SeamField *field, int32_t node, const BoundaryRow *row,
    const Segment sides[2], const Segment *mean, int shared) {
    if (!shared)
        return row->turn;
    int side = field->side[node];
    double moved = segment_angle(field->nodes[node].x, field->nodes[node].y, mean) -
        segment_angle(field->nodes[node].x, field->nodes[node].y, sides + side);
    return row->turn + atan2(sin(moved), cos(moved)) / (2 * M_PI);
}

static int seam_crossings(const SeamField *field, const Boundary *lower, const Boundary *upper,
    int64_t lower_offset, int64_t upper_offset, const Segment sides[2], const Segment *mean,
    int shared, SeamCross **out, int64_t *out_count) {
    int64_t capacity = field->edge_count + field->pair_count;
    SeamCross *cross = malloc((size_t)(capacity ? capacity : 1) * sizeof(*cross));
    if (!cross)
        return -1;
    int64_t count = 0;
    for (int kind = 0; kind < 2; kind++) {
        int64_t links = kind ? field->pair_count : field->edge_count;
        for (int64_t i = 0; i < links; i++) {
            int32_t a = kind ? field->pairs[i].a : field->edges[i].a;
            int32_t b = kind ? field->pairs[i].b : field->edges[i].b;
            double weight = kind ? field->weights[i] : field->edges[i].weight;
            if (field->side[a] == field->side[b] || weight <= 0)
                continue;
            const BoundaryRow *row_a = boundary_row(field->side[a] ? upper : lower, field->ids[a]);
            const BoundaryRow *row_b = boundary_row(field->side[b] ? upper : lower, field->ids[b]);
            if (!row_a || !row_b || row_a->piece < 0 || row_b->piece < 0)
                continue;
            int swap = field->side[a] > field->side[b];
            const BoundaryRow *low = swap ? row_b : row_a;
            const BoundaryRow *high = swap ? row_a : row_b;
            double turn_a = boundary_turn(field, a, row_a, sides, mean, shared);
            double turn_b = boundary_turn(field, b, row_b, sides, mean, shared);
            cross[count++] = (SeamCross){.p = (int32_t)(lower_offset + low->piece),
                .q = (int32_t)(upper_offset + high->piece),
                .rise = kind ? (swap ? -1 : 1) : 0,
                .turn = swap ? turn_b - turn_a : turn_a - turn_b,
                .up = swap ? weight : weight * (kind ? 0.5 : 1.0),
                .down = swap ? weight * (kind ? 0.5 : 1.0) : weight};
        }
    }
    *out = cross;
    *out_count = count;
    return 0;
}

typedef struct {
    int64_t value, order;
    double weight;
} WeightedValue;

static int weighted_before(const void *left, const void *right) {
    const WeightedValue *a = left, *b = right;
    if (a->value != b->value)
        return a->value < b->value ? -1 : 1;
    return (a->order > b->order) - (a->order < b->order);
}

static int frame_term_before(const void *left, const void *right) {
    const FrameTerm *a = left, *b = right;
    if (a->p != b->p)
        return a->p < b->p ? -1 : 1;
    if (a->q != b->q)
        return a->q < b->q ? -1 : 1;
    if (a->k != b->k)
        return a->k < b->k ? -1 : 1;
    return (a->order > b->order) - (a->order < b->order);
}

static int seam_terms(const SeamCross *cross, int64_t count, int curl, int64_t *start,
    int64_t shift_from, int64_t total, FrameTerm **out, int64_t *out_count, int64_t *shift) {
    WeightedValue *values = malloc((size_t)(count ? count : 1) * sizeof(*values));
    FrameTerm *raw = malloc((size_t)(count ? count : 1) * sizeof(*raw));
    if (!values || !raw) {
        free(raw);
        free(values);
        return -1;
    }
    double weight = 0;
    for (int64_t i = 0; i < count; i++) {
        int64_t k = llrint(curl * cross[i].turn) + cross[i].rise;
        values[i] = (WeightedValue){.value = k + start[cross[i].p] - start[cross[i].q],
            .order = i,
            .weight = cross[i].down};
        raw[i] = (FrameTerm){.p = cross[i].p, .q = cross[i].q, .k = (int32_t)k, .order = i};
        weight += cross[i].down;
    }
    qsort(values, (size_t)count, sizeof(*values), weighted_before);
    double cumulative = 0;
    *shift = 0;
    for (int64_t i = 0; i < count; i++) {
        cumulative += values[i].weight;
        if (cumulative >= weight * 0.5) {
            *shift = values[i].value;
            break;
        }
    }
    for (int64_t i = shift_from; i < total; i++)
        start[i] += *shift;
    qsort(raw, (size_t)count, sizeof(*raw), frame_term_before);
    int64_t kept = 0;
    for (int64_t begin = 0; begin < count;) {
        int64_t end = begin + 1;
        double up = cross[raw[begin].order].up, down = cross[raw[begin].order].down;
        while (end < count && raw[end].p == raw[begin].p && raw[end].q == raw[begin].q &&
            raw[end].k == raw[begin].k) {
            up += cross[raw[end].order].up;
            down += cross[raw[end].order].down;
            end++;
        }
        int64_t up_cost = llrint(up * UNITS), down_cost = llrint(down * UNITS);
        if (up_cost < 1)
            up_cost = 1;
        if (down_cost < 1)
            down_cost = 1;
        if (up_cost > INT32_MAX || down_cost > INT32_MAX) {
            free(raw);
            free(values);
            return -1;
        }
        raw[kept] = raw[begin];
        raw[kept].up = (int32_t)up_cost;
        raw[kept].down = (int32_t)down_cost;
        kept++;
        begin = end;
    }
    free(values);
    *out = raw;
    *out_count = kept;
    return 0;
}

typedef struct {
    uint64_t key;
    int64_t agree, disagree;
} PairVote;

static int pair_vote_before(const void *left, const void *right) {
    const PairVote *a = left, *b = right;
    return (a->key > b->key) - (a->key < b->key);
}

static int int64_before(const void *left, const void *right) {
    int64_t a = *(const int64_t *)left, b = *(const int64_t *)right;
    return (a > b) - (a < b);
}

static int write_frame(
    const char *node_folder, const char *piece_folder, const char *out, const char *method) {
    glob_t chunks = {0};
    if (!node_rows(node_folder, &chunks)) {
        globfree(&chunks);
        return 1;
    }
    size_t units = chunks.gl_pathc;
    char **pieces = calloc(units, sizeof(*pieces));
    FrameUnit *data = calloc(units, sizeof(*data));
    int64_t *first = calloc(units + 1, sizeof(*first));
    int64_t *crossing_count = calloc(units - 1, sizeof(*crossing_count));
    int64_t *seam_term_count = calloc(units - 1, sizeof(*seam_term_count));
    int64_t *shifts = calloc(units - 1, sizeof(*shifts));
    if (!pieces || !data || !first || !crossing_count || !seam_term_count || !shifts) {
        free(shifts);
        free(seam_term_count);
        free(crossing_count);
        free(first);
        free(data);
        free(pieces);
        globfree(&chunks);
        return 1;
    }
    int failed = 0;
    int curl = 0;
    for (size_t i = 0; !failed && i < units; i++) {
        const char *name = strrchr(chunks.gl_pathv[i], '/');
        if (asprintf(&pieces[i], "%s/%s", piece_folder, name ? name + 1 : chunks.gl_pathv[i]) < 0 ||
            frame_unit_open(&data[i], pieces[i])) {
            failed = 1;
            break;
        }
        if (!i)
            curl = data[i].chirality;
        if (data[i].chirality != curl || data[i].count > INT32_MAX - first[i]) {
            failed = 1;
            break;
        }
        first[i + 1] = first[i] + data[i].count;
    }
    int64_t total = first[units];
    int64_t *start =
        !failed && total <= INT32_MAX ? malloc((size_t)(total ? total : 1) * sizeof(*start)) : NULL;
    if (!start)
        failed = 1;
    for (size_t i = 0; !failed && i < units; i++)
        for (int64_t j = 0; j < data[i].count; j++)
            start[first[i] + j] = data[i].level[j];
    FrameTerm *seam_all = NULL;
    int64_t seam_count = 0;
    for (size_t seam_index = 0; !failed && seam_index + 1 < units; seam_index++) {
        int seam = seam_end_plane(chunks.gl_pathv[seam_index]);
        if (seam == INT_MIN) {
            failed = 1;
            break;
        }
        Boundary boundary[2] = {0};
        SeamField field = {0};
        SeamCross *cross = NULL;
        int64_t crosses = 0;
        Segment sides[2] = {0}, mean = {0};
        int shared = shared_segments(pieces[seam_index], pieces[seam_index + 1], sides, &mean);
        if (shared < 0 || boundary_open(&boundary[0], pieces[seam_index]) ||
            boundary_open(&boundary[1], pieces[seam_index + 1]) ||
            seam_field_open(&field, chunks.gl_pathv[seam_index], chunks.gl_pathv[seam_index + 1],
                pieces[seam_index], pieces[seam_index + 1], seam) ||
            seam_crossings(&field, &boundary[0], &boundary[1], first[seam_index],
                first[seam_index + 1], sides, &mean, shared, &cross, &crosses)) {
            failed = 1;
        }
        FrameTerm *part = NULL;
        int64_t terms = 0;
        if (!failed &&
            seam_terms(cross, crosses, curl, start, first[seam_index + 1], total, &part, &terms,
                &shifts[seam_index]))
            failed = 1;
        FrameTerm *larger = !failed
            ? realloc(seam_all,
                  (size_t)(seam_count + terms ? seam_count + terms : 1) * sizeof(*seam_all))
            : NULL;
        if (!failed && !larger)
            failed = 1;
        if (!failed) {
            seam_all = larger;
            memcpy(seam_all + seam_count, part, (size_t)terms * sizeof(*part));
            seam_count += terms;
            crossing_count[seam_index] = crosses;
            seam_term_count[seam_index] = terms;
        }
        free(part);
        free(cross);
        seam_field_close(&field);
        free(boundary[1].rows);
        free(boundary[0].rows);
    }
    int64_t terms = seam_count;
    for (size_t i = 0; !failed && i < units; i++)
        terms += data[i].terms;
    int32_t *term_values =
        !failed ? malloc((size_t)(terms ? terms : 1) * 5 * sizeof(*term_values)) : NULL;
    int32_t *p = NULL, *q = NULL, *k = NULL, *up = NULL, *down = NULL;
    if (!term_values)
        failed = 1;
    else {
        p = term_values;
        q = p + terms;
        k = q + terms;
        up = k + terms;
        down = up + terms;
    }
    int64_t at = 0;
    for (size_t i = 0; !failed && i < units; i++)
        for (int64_t j = 0; j < data[i].terms; j++) {
            p[at] = (int32_t)(data[i].p[j] + first[i]);
            q[at] = (int32_t)(data[i].q[j] + first[i]);
            k[at] = data[i].k[j];
            up[at] = data[i].up[j];
            down[at++] = data[i].down[j];
        }
    for (int64_t i = 0; !failed && i < seam_count; i++) {
        p[at] = seam_all[i].p;
        q[at] = seam_all[i].q;
        k[at] = seam_all[i].k;
        up[at] = seam_all[i].up;
        down[at++] = seam_all[i].down;
    }
    int32_t *x = !failed ? malloc((size_t)(total ? total : 1) * sizeof(*x)) : NULL;
    for (int64_t i = 0; x && i < total; i++) {
        if (start[i] < INT32_MIN || start[i] > INT32_MAX)
            failed = 1;
        else
            x[i] = (int32_t)start[i];
    }
    int64_t log[3] = {0};
    if (!x || (!failed && hx_level_minimize(total, terms, p, q, k, up, down, x, 64, log) < 0))
        failed = 1;
    int64_t *level = !failed ? malloc((size_t)(total ? total : 1) * sizeof(*level)) : NULL;
    int64_t *moved = !failed ? malloc((size_t)(total ? total : 1) * sizeof(*moved)) : NULL;
    if (!level || !moved)
        failed = 1;
    for (int64_t i = 0; !failed && i < total; i++) {
        level[i] = x[i];
        moved[i] = level[i] - start[i];
    }
    if (!failed)
        qsort(moved, (size_t)total, sizeof(*moved), int64_before);
    int64_t middle = 0;
    if (!failed && total) {
        int64_t center = total >> 1;
        middle = total & 1 ? moved[center]
                           : (int64_t)(((double)moved[center - 1] + (double)moved[center]) * 0.5);
    }
    int64_t moved_count = 0;
    for (int64_t i = 0; !failed && i < total; i++)
        moved_count += level[i] - start[i] - middle != 0;
    PairVote *votes =
        !failed ? malloc((size_t)(seam_count ? seam_count : 1) * sizeof(*votes)) : NULL;
    double held_before = 0, held_after = 0, held_weight = 0;
    for (int64_t i = 0; votes && i < seam_count; i++) {
        int before = start[seam_all[i].q] - start[seam_all[i].p] == seam_all[i].k;
        int after = level[seam_all[i].q] - level[seam_all[i].p] == seam_all[i].k;
        held_before += seam_all[i].down * before;
        held_after += seam_all[i].down * after;
        held_weight += seam_all[i].down;
        votes[i] = (PairVote){.key = (uint64_t)seam_all[i].p * (uint64_t)total + seam_all[i].q,
            .agree = after ? seam_all[i].down : 0,
            .disagree = after ? 0 : seam_all[i].down};
    }
    if (!votes)
        failed = 1;
    if (!failed)
        qsort(votes, (size_t)seam_count, sizeof(*votes), pair_vote_before);
    int64_t standing = 0;
    for (int64_t begin = 0; !failed && begin < seam_count;) {
        int64_t end = begin + 1, agree = votes[begin].agree, disagree = votes[begin].disagree;
        while (end < seam_count && votes[end].key == votes[begin].key) {
            agree += votes[end].agree;
            disagree += votes[end].disagree;
            end++;
        }
        if (agree > disagree)
            votes[standing++] = votes[begin];
        begin = end;
    }
    int64_t *held = !failed ? malloc((size_t)(standing ? standing : 1) * 2 * sizeof(*held)) : NULL;
    int64_t *held_p = NULL, *held_q = NULL;
    if (!held || (!total && standing))
        failed = 1;
    else {
        held_p = held;
        held_q = held + standing;
    }
    for (int64_t i = 0; !failed && total && i < standing; i++) {
        held_p[i] = (int64_t)(votes[i].key / (uint64_t)total);
        held_q[i] = (int64_t)(votes[i].key % (uint64_t)total);
    }
    char *archive_path = NULL, *record_path = NULL;
    if (!failed &&
        (!make_folders(out) || asprintf(&archive_path, "%s/levels.npz", out) < 0 ||
            asprintf(&record_path, "%s/record.json", out) < 0))
        failed = 1;
    int error = 0;
    Npz archive = {0};
    archive.file = !failed ? zip_open(archive_path, ZIP_CREATE | ZIP_TRUNCATE, &error) : NULL;
    int64_t level_shape[] = {total}, first_shape[] = {(int64_t)units + 1},
            held_shape[] = {standing}, curl_value = curl;
    if (!failed)
        failed = !archive.file ||
            npz_add(
                &archive, "level", "<i8", level_shape, 1, level, (size_t)total * sizeof(*level)) ||
            npz_add(
                &archive, "first", "<i8", first_shape, 1, first, (units + 1) * sizeof(*first)) ||
            npz_add(&archive, "curl", "<i8", NULL, 0, &curl_value, sizeof(curl_value)) ||
            npz_add(&archive, "held_p", "<i8", held_shape, 1, held_p,
                (size_t)standing * sizeof(*held_p)) ||
            npz_add(&archive, "held_q", "<i8", held_shape, 1, held_q,
                (size_t)standing * sizeof(*held_q));
    failed = npz_finish(&archive, archive_path, failed);
    json_object *summary = json_object_new_object();
    json_object *unit_rows = json_object_new_array_ext((int)units);
    json_object *seam_rows = json_object_new_array_ext((int)(units - 1));
    for (size_t i = 0; !failed && i < units; i++) {
        const char *name = strrchr(pieces[i], '/');
        json_object_array_add(unit_rows, json_object_new_string(name ? name + 1 : pieces[i]));
        if (i + 1 < units) {
            json_object *row = json_object_new_object();
            json_object_object_add(row, "crossing", json_object_new_int64(crossing_count[i]));
            json_object_object_add(row, "terms", json_object_new_int64(seam_term_count[i]));
            json_object_object_add(row, "shift", json_object_new_int64(shifts[i]));
            json_object_array_add(seam_rows, row);
        }
    }
    json_object *grown = json_object_new_array_ext(1);
    json_object_array_add(grown, json_object_new_int(curl));
    json_object_object_add(summary, "units", unit_rows);
    json_object_object_add(summary, "pieces", json_object_new_int64(total));
    json_object_object_add(summary, "curl", json_object_new_int(curl));
    json_object_object_add(summary, "grown with", grown);
    json_object_object_add(summary, "energy", json_object_new_int64(log[2]));
    json_object_object_add(summary, "cuts", json_object_new_int64(log[0]));
    json_object_object_add(summary, "seams", seam_rows);
    json_object_object_add(summary, "terms", json_object_new_int64(terms));
    json_object_object_add(summary, "seam kept before",
        json_object_new_double(held_weight ? held_before / held_weight : NAN));
    json_object_object_add(
        summary, "seam kept", json_object_new_double(held_weight ? held_after / held_weight : NAN));
    json_object_object_add(
        summary, "pieces moved", json_object_new_double(total ? (double)moved_count / total : NAN));
    json_object_object_add(summary, "method", json_object_new_string(failed ? "" : method));
    if (!failed && json_object_to_file_ext(record_path, summary, JSON_C_TO_STRING_PRETTY))
        failed = 1;
    json_object_put(summary);
    free(record_path);
    free(archive_path);
    free(held);
    free(votes);
    free(moved);
    free(level);
    free(x);
    free(term_values);
    free(seam_all);
    free(start);
    for (size_t i = 0; i < units; i++) {
        frame_unit_close(&data[i]);
        free(pieces[i]);
    }
    free(shifts);
    free(seam_term_count);
    free(crossing_count);
    free(first);
    free(data);
    free(pieces);
    globfree(&chunks);
    return failed ? 1 : 0;
}

typedef int (*ScanWriter)(const char *, const char *, const char *, const char *);

static int write_scan_stage(
    const char *nodes, const char *pieces, const char *out, const char *tag, ScanWriter writer) {
    char method[13];
    if (stage_method(method, tag, nodes, pieces))
        return 1;
    if (record_is_current(out, method))
        return 0;
    char *part = NULL;
    if (asprintf(&part, "%s.building.%ld", out, (long)getpid()) < 0)
        return 1;
    int failed =
        remove_path(part) || writer(nodes, pieces, part, method) || publish_folder(part, out);
    if (failed) {
        remove_path(part);
        fprintf(stderr, "HerculeX: native %s stage failed: %s\n", tag, out);
    }
    free(part);
    return failed;
}

static int join_write_command(int argc, char **argv) {
    if (argc != 5)
        return 2;
    return write_scan_stage(argv[2], argv[3], argv[4], "joints", write_joints);
}

static int frame_write_command(int argc, char **argv) {
    if (argc != 5)
        return 2;
    return write_scan_stage(argv[2], argv[3], argv[4], "frame", write_frame);
}

static char *level_file(const char *sample, const char *scan, const char *name) {
    char *outputs = local_root(outputs_root), *path = NULL;
    if (outputs)
        asprintf(&path, "%s/unwrapping/level/%s/%s/%s", outputs, sample, scan, name);
    free(outputs);
    return path;
}

static int regular_file(const char *path) {
    struct stat info;
    return path && stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static int share_nodes_alias(const char *outputs, const char *sample, const char *scan) {
    char *link = NULL, *target = NULL, *folder = NULL;
    if (asprintf(&link, "%s/unwrapping/level/%s/%s/nodes", outputs, sample, scan) < 0 ||
        asprintf(&target, "../../../nodes/%s/%s/nodes", sample, scan) < 0)
        goto fail;
    folder = strdup(link);
    char *slash = folder ? strrchr(folder, '/') : NULL;
    if (!slash)
        goto fail;
    *slash = 0;
    if (!make_folders(folder))
        goto fail;
    struct stat info;
    if (lstat(link, &info) == 0) {
        if (!S_ISLNK(info.st_mode)) {
            free(folder);
            free(target);
            free(link);
            return 0;
        }
        char held[PATH_MAX];
        ssize_t size = readlink(link, held, sizeof(held) - 1);
        if (size >= 0) {
            held[size] = 0;
            if (!strcmp(held, target)) {
                free(folder);
                free(target);
                free(link);
                return 0;
            }
        }
        if (unlink(link))
            goto fail;
    } else if (errno != ENOENT) {
        goto fail;
    }
    int status = symlink(target, link);
    free(folder);
    free(target);
    free(link);
    return status;
fail:
    free(folder);
    free(target);
    free(link);
    return -1;
}

static int read_curl_value(const char *sample, const char *scan, int *curl) {
    char *path = level_file(sample, scan, "curl.json");
    json_object *document = path ? json_object_from_file(path) : NULL, *value, *rule;
    free(path);
    int valid = document && json_object_object_get_ex(document, "curl", &value) &&
        json_object_object_get_ex(document, "rule", &rule) &&
        json_object_is_type(rule, json_type_string) &&
        !strcmp(json_object_get_string(rule), "set by hand");
    int found = valid ? json_object_get_int(value) : 0;
    if (document)
        json_object_put(document);
    if (abs(found) != 1)
        return -1;
    *curl = found;
    return 0;
}

static int curl_command(int argc, char **argv) {
    if (argc < 4 || argc > 5 ||
        (argc == 5 && strcmp(argv[4], "1") != 0 && strcmp(argv[4], "-1") != 0)) {
        fputs("Usage: ./run.sh level.level.run_levels curl SCROLL SCAN [1|-1]\n", stderr);
        return 2;
    }
    if (argc == 5) {
        char *path = level_file(argv[2], argv[3], "curl.json");
        char *folder = path ? strdup(path) : NULL;
        char *slash = folder ? strrchr(folder, '/') : NULL;
        if (slash)
            *slash = 0;
        json_object *document = json_object_new_object();
        json_object_object_add(document, "curl", json_object_new_int(argv[4][0] == '-' ? -1 : 1));
        json_object_object_add(document, "rule", json_object_new_string("set by hand"));
        char stamp[32];
        time_t now = time(NULL);
        struct tm local;
        localtime_r(&now, &local);
        strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M", &local);
        json_object_object_add(document, "set", json_object_new_string(stamp));
        char *part = NULL;
        int failed = !path || !folder || !make_folders(folder) ||
            asprintf(&part, "%s.part.%ld", path, (long)getpid()) < 0 ||
            json_object_to_file_ext(part, document, JSON_C_TO_STRING_PLAIN) || rename(part, path);
        if (failed && part) {
            unlink(part);
        }
        free(part);
        json_object_put(document);
        free(folder);
        free(path);
        if (failed)
            return 1;
    }
    int curl;
    if (read_curl_value(argv[2], argv[3], &curl)) {
        fprintf(stderr, "%s %s: no curl set by hand\n", argv[2], argv[3]);
        return 1;
    }
    printf("%s %s curl %d\n", argv[2], argv[3], curl);
    return 0;
}

#ifdef HX_UNITY
int hx_umbilicus_keep(const char *sample, const char *scan);
#endif

static int ready_command(int argc, char **argv) {
    if (argc != 4) {
        fputs("Usage: ./run.sh level.level.run_levels ready SCROLL SCAN\n", stderr);
        return 2;
    }
    int curl;
    char *axis = level_file(argv[2], argv[3], "sheet/axis.json");
    int valid = axis && !read_curl_value(argv[2], argv[3], &curl);
#ifdef HX_UNITY
    valid = valid && !hx_umbilicus_keep(argv[2], argv[3]);
#else
    valid = valid && regular_file(axis);
#endif
    if (valid)
        printf("%s %s curl %d umbilicus %s\n", argv[2], argv[3], curl, axis);
    else
        fprintf(stderr, "%s %s is missing its hand-set curl or kept umbilicus\n", argv[2], argv[3]);
    free(axis);
    return valid ? 0 : 1;
}

static int write_claimed(const char *unit, const char *out, int curl) {
    char method[13];
    if (method_id(method))
        return 1;
    if (record_is_current(out, method))
        return 0;
#ifdef HX_UNITY
    char *path = path_join(unit, "record.json");
    json_object *record = path ? json_object_from_file(path) : NULL;
    free(path);
    const char *sample = json_object_get_string(json_object_object_get(record, "sample"));
    const char *scan = json_object_get_string(json_object_object_get(record, "volume"));
    int ready = sample && scan && !hx_umbilicus_keep(sample, scan);
    json_object_put(record);
    if (!ready) {
        return 1;
    }
#endif
    char *part = NULL;
    if (asprintf(&part, "%s.building.%ld", out, (long)getpid()) < 0)
        return 1;
    int failed = remove_path(part) || write_unit(unit, part, curl, 1) || publish_folder(part, out);
    if (failed) {
        remove_path(part);
        fprintf(stderr, "HerculeX: native level unit failed: %s\n", unit);
    }
    free(part);
    return failed;
}

static int unit_command(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fputs("Usage: ./run.sh level.level.run_levels unit SCROLL UNIT [SCAN]\n", stderr);
        return 2;
    }
    char *outputs = local_root(outputs_root), *pattern = NULL;
    const char *scan = argc == 5 ? argv[4] : "*";
    if (outputs)
        asprintf(&pattern, "%s/unwrapping/nodes/%s/%s/nodes/%s", outputs, argv[2], scan, argv[3]);
    glob_t paths = {0};
    int found = pattern && glob(pattern, 0, NULL, &paths) == 0 && paths.gl_pathc == 1;
    free(pattern);
    if (!found) {
        globfree(&paths);
        free(outputs);
        fprintf(stderr, "HerculeX: could not choose one node unit %s %s\n", argv[2], argv[3]);
        return 1;
    }
    char *record_path = NULL;
    asprintf(&record_path, "%s/record.json", paths.gl_pathv[0]);
    json_object *record = record_path ? json_object_from_file(record_path) : NULL;
    free(record_path);
    char *volume = record ? json_text(record, "volume") : NULL;
    int curl = 0;
    char *out = NULL;
    if (volume)
        asprintf(&out, "%s/unwrapping/level/%s/%s/pieces/%s", outputs, argv[2], volume, argv[3]);
    int failed = !record || !volume || !out || read_curl_value(argv[2], volume, &curl) ||
        share_nodes_alias(outputs, argv[2], volume) || write_claimed(paths.gl_pathv[0], out, curl);
    if (record)
        json_object_put(record);
    free(out);
    free(volume);
    globfree(&paths);
    free(outputs);
    return failed;
}

static int sample_selected(const char *sample, char **samples, int count) {
    if (!count)
        return 1;
    for (int i = 0; i < count; i++)
        if (!strcmp(sample, samples[i]))
            return 1;
    return 0;
}

static int finish_scan(const char *outputs, const char *sample, const char *scan) {
    char *nodes = NULL, *pieces = NULL, *pattern = NULL, *joints = NULL, *frame = NULL;
    int made = asprintf(&nodes, "%s/unwrapping/nodes/%s/%s/nodes", outputs, sample, scan) >= 0 &&
        asprintf(&pieces, "%s/unwrapping/level/%s/%s/pieces", outputs, sample, scan) >= 0 &&
        asprintf(&pattern, "%s/rows-*", nodes) >= 0 &&
        asprintf(&joints, "%s/joints", pieces) >= 0 && asprintf(&frame, "%s/frame", pieces) >= 0;
    glob_t chunks = {0};
    int enough = made && glob(pattern, 0, NULL, &chunks) == 0 && chunks.gl_pathc >= 2;
    globfree(&chunks);
    int failed = enough &&
        (write_scan_stage(nodes, pieces, joints, "joints", write_joints) ||
            write_scan_stage(nodes, pieces, frame, "frame", write_frame));
    free(frame);
    free(joints);
    free(pattern);
    free(pieces);
    free(nodes);
    return made && (!enough || !failed) ? 0 : 1;
}

static int finish_command(int argc, char **argv) {
    if (argc != 4)
        return 2;
    char *outputs = local_root(outputs_root);
    int failed = !outputs || finish_scan(outputs, argv[2], argv[3]);
    free(outputs);
    return failed;
}

static int run_command(int argc, char **argv) {
    int band = 0, sample_count = 0;
    const char *scan = "*";
    char **samples = calloc((size_t)argc, sizeof(*samples));
    for (int i = 2; samples && i < argc; i++) {
        if (!strcmp(argv[i], "--band")) {
            band = 1;
        } else if (!strcmp(argv[i], "--scan") && i + 1 < argc) {
            scan = argv[++i];
        } else if (!strcmp(argv[i], "--since") && i + 1 < argc) {
            i++;
        } else {
            samples[sample_count++] = argv[i];
        }
    }
    char *outputs = local_root(outputs_root), *pattern = NULL;
    if (outputs)
        asprintf(
            &pattern, "%s/unwrapping/nodes/*/%s/nodes/%s-*", outputs, scan, band ? "band" : "rows");
    glob_t units = {0};
    int listed = pattern && glob(pattern, 0, NULL, &units) == 0;
    free(pattern);
    char **finished = calloc(units.gl_pathc ? units.gl_pathc * 2 : 1, sizeof(*finished));
    size_t finished_count = 0;
    int failures = !samples || !outputs || !finished;
    for (size_t i = 0; listed && i < units.gl_pathc; i++) {
        char *record_path = NULL;
        asprintf(&record_path, "%s/record.json", units.gl_pathv[i]);
        json_object *record = record_path ? json_object_from_file(record_path) : NULL;
        free(record_path);
        char *sample = record ? json_text(record, "sample") : NULL;
        char *volume = record ? json_text(record, "volume") : NULL;
        const char *slash = strrchr(units.gl_pathv[i], '/');
        const char *name = slash ? slash + 1 : units.gl_pathv[i];
        int curl = 0;
        char *out = NULL;
        if (sample && volume)
            asprintf(&out, "%s/unwrapping/level/%s/%s/pieces/%s", outputs, sample, volume, name);
        if (sample && volume && sample_selected(sample, samples, sample_count)) {
            int failed = !out || read_curl_value(sample, volume, &curl) ||
                share_nodes_alias(outputs, sample, volume) ||
                write_claimed(units.gl_pathv[i], out, curl);
            failures += failed;
            int known = 0;
            for (size_t k = 0; !band && k < finished_count; k++)
                known |= !strcmp(finished[2 * k], sample) && !strcmp(finished[2 * k + 1], volume);
            if (!band && !known) {
                finished[2 * finished_count] = strdup(sample);
                finished[2 * finished_count + 1] = strdup(volume);
                if (!finished[2 * finished_count] || !finished[2 * finished_count + 1]) {
                    free(finished[2 * finished_count + 1]);
                    free(finished[2 * finished_count]);
                    finished[2 * finished_count] = NULL;
                    finished[2 * finished_count + 1] = NULL;
                    failures++;
                } else {
                    finished_count++;
                }
            }
        }
        free(out);
        free(volume);
        free(sample);
        if (record)
            json_object_put(record);
    }
    for (size_t i = 0; !failures && i < finished_count; i++)
        failures += finish_scan(outputs, finished[2 * i], finished[2 * i + 1]);
    for (size_t i = 0; i < finished_count; i++) {
        free(finished[2 * i + 1]);
        free(finished[2 * i]);
    }
    free(finished);
    globfree(&units);
    free(outputs);
    free(samples);
    return failures ? 1 : 0;
}

static void usage(FILE *out) {
    fputs("Usage: ./run.sh level.level.run_levels probe NODE_UNIT\n"
          "       ./run.sh level.level.run_levels solve NODE_UNIT CURL\n"
          "       ./run.sh level.level.run_levels flatten NODE_UNIT CURL\n"
          "       ./run.sh level.level.run_levels raster NODE_UNIT CURL PIECE\n"
          "       ./run.sh level.level.run_levels render NODE_UNIT CURL PIECE\n"
          "       ./run.sh level.level.run_levels mesh NODE_UNIT CURL PIECE\n"
          "       ./run.sh level.level.run_levels upright NODE_UNIT CURL PIECE\n"
          "       ./run.sh level.level.run_levels write NODE_UNIT OUT CURL\n"
          "       ./run.sh level.level.run_levels joinprobe LOWER UPPER PIECES0 PIECES1 SEAM\n"
          "       ./run.sh level.level.run_levels joinwrite NODES PIECES OUT\n"
          "       ./run.sh level.level.run_levels framewrite NODES PIECES OUT\n"
          "       ./run.sh level.level.run_levels {run,curl,ready,unit,finish} ...\n",
        out);
}

static int chirality_of(const char *text) {
    if (!strcmp(text, "1"))
        return 1;
    return !strcmp(text, "-1") ? -1 : 0;
}

static int piece_of(const char *text, int32_t *piece) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || *end || value < 0 || value > INT32_MAX)
        return -1;
    *piece = (int32_t)value;
    return 0;
}

int hx_levels_main(int argc, char **argv) {
    static const struct {
        const char *name;
        int (*run)(int, char **);
    } commands[] = {{"run", run_command}, {"curl", curl_command}, {"ready", ready_command},
        {"unit", unit_command}, {"finish", finish_command}, {"joinprobe", join_probe},
        {"joinwrite", join_write_command}, {"framewrite", frame_write_command}};
    static const char *raster_modes[] = {"raster", "render", "mesh", "upright"};
    const char *command = argc >= 2 ? argv[1] : "";
    for (size_t i = 0; i < sizeof(commands) / sizeof(*commands); i++) {
        if (!strcmp(command, commands[i].name))
            return commands[i].run(argc, argv);
    }
    if (argc == 2 && !strcmp(command, "method")) {
        char method[13];
        if (method_id(method)) {
            fputs("HerculeX: could not digest the native level sources\n", stderr);
            return 1;
        }
        puts(method);
        return 0;
    }
    if (argc == 3 && !strcmp(command, "probe"))
        return probe(argv[2]);
    int curl = argc >= 4 ? chirality_of(argv[3]) : 0;
    if (argc == 4 && curl) {
        if (!strcmp(command, "solve"))
            return solve_probe(argv[2], curl);
        if (!strcmp(command, "flatten"))
            return flatten_probe(argv[2], curl);
    }
    if (argc == 5 && curl) {
        for (int mode = 0; mode < (int)(sizeof(raster_modes) / sizeof(*raster_modes)); mode++) {
            if (!strcmp(command, raster_modes[mode])) {
                int32_t piece;
                return piece_of(argv[4], &piece) ? 2 : raster_probe(argv[2], curl, piece, mode);
            }
        }
    }
    int write_curl = argc == 5 ? chirality_of(argv[4]) : 0;
    if (write_curl && !strcmp(command, "write"))
        return write_unit(argv[2], argv[3], write_curl, 0);
    int help = argc == 2 && (!strcmp(command, "-h") || !strcmp(command, "--help"));
    usage(help ? stdout : stderr);
    return help ? 0 : 2;
}

#if !defined(HX_UNITY) && !defined(HX_LEVEL_EMBEDDED)
int main(int argc, char **argv) { return hx_levels_main(argc, argv); }
#endif
