#define HX_LEVEL_EMBEDDED
#include "../level/run_levels.c"

#ifdef HX_UNITY
#define HX_COLUMN_NATIVE
#define HX_MESH_BRIDGE
#endif
#define hx_relax_mesh hx_layout_relax_mesh
#define hx_readable_surface hx_layout_readable_surface
#include "../../sampling/column_sampling.c"
#undef hx_readable_surface
#undef hx_relax_mesh

#ifndef HX_UNITY
const char hx_usage[] = "Find and place the joints between neighbouring layout units.";
#endif

enum { MATCH_SCALE = 2 };

typedef struct {
    double x, y, z;
    int row, column;
    size_t next;
} Cell;

typedef struct {
    int x, y, z;
    size_t first;
} Bucket;

typedef struct {
    Cell *cells;
    Bucket *table;
    size_t count, slots;
    double bound;
} Search;

typedef struct {
    int64_t id, source;
    size_t order;
    int loaded;
    Search search;
} Block;

typedef struct {
    Block *blocks;
    size_t count;
    const char *folder;
} Side;

typedef struct {
    int64_t lower, upper, order;
    int32_t a, b;
} Crossing;

typedef struct {
    size_t hits;
    int64_t a, b;
    double x[2], y[2];
} Joint;

static void side_close(Side *side) {
    for (size_t i = 0; i < side->count; i++) {
        free(side->blocks[i].search.cells);
        free(side->blocks[i].search.table);
    }
    free(side->blocks);
}

static int block_order(const void *one, const void *two) {
    const Block *a = one, *b = two;
    if (a->source != b->source) {
        return a->source < b->source ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static json_object *read_record(const char *folder, const char *name) {
    char *path = path_join(folder, name);
    json_object *document = path ? json_object_from_file(path) : NULL;
    free(path);
    return document;
}

static int side_open(Side *side, const char *folder, int framed) {
    side->folder = folder;
    json_object *record = read_record(folder, "record.json");
    json_object *rows = json_object_object_get(record, "pieces");
    if (!json_object_is_type(rows, json_type_array)) {
        json_object_put(record);
        return 0;
    }
    size_t n = json_object_array_length(rows);
    side->blocks = calloc(n ? n : 1, sizeof(*side->blocks));
    if (!side->blocks) {
        json_object_put(record);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        json_object *piece = json_object_array_get_idx(rows, i);
        if (!json_object_get_boolean(json_object_object_get(piece, "written"))) {
            continue;
        }
        json_object *id = json_object_object_get(piece, "piece");
        json_object *source = framed ? json_object_object_get(piece, "source") : NULL;
        if (!json_object_is_type(id, json_type_int) ||
            (source && !json_object_is_type(source, json_type_int))) {
            json_object_put(record);
            return 0;
        }
        int64_t number = json_object_get_int64(id);
        side->blocks[side->count++] = (Block){
            .id = number, .source = source ? json_object_get_int64(source) : number, .order = i};
    }
    qsort(side->blocks, side->count, sizeof(*side->blocks), block_order);
    json_object_put(record);
    return 1;
}

static size_t block_at(const Side *side, int64_t key) {
    size_t lo = 0, hi = side->count;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (side->blocks[m].source < key) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    return lo;
}

static size_t bucket_index(const Search *search, int x, int y, int z) {
    uint64_t hash = (uint64_t)x * UINT64_C(0x9e3779b97f4a7c15) ^
        (uint64_t)y * UINT64_C(0xbf58476d1ce4e5b9) ^ (uint64_t)z * UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    size_t slot = (hash ^ (hash >> 27)) & (search->slots - 1);
    while (search->table[slot].first) {
        const Bucket *b = search->table + slot;
        if (b->x == x && b->y == y && b->z == z) {
            break;
        }
        slot = (slot + 1) & (search->slots - 1);
    }
    return slot;
}

static int bin(double x, double step, int *out) {
    double cell = floor(x / step);
    if (!(cell >= INT_MIN + 1 && cell <= INT_MAX - 1)) {
        return 0;
    }
    *out = (int)cell;
    return 1;
}

static int index_cells(Search *search) {
    size_t slots = 1;
    if (search->count > SIZE_MAX / 4) {
        return 0;
    }
    while (slots < 2 * search->count + 1) {
        slots *= 2;
    }
    search->table = calloc(slots, sizeof(*search->table));
    if (!search->table) {
        return 0;
    }
    search->slots = slots;
    for (size_t i = 0; i < search->count; i++) {
        Cell *point = search->cells + i;
        int x, y, z;
        if (!bin(point->x, search->bound, &x) || !bin(point->y, search->bound, &y) ||
            !bin(point->z, search->bound, &z)) {
            return 0;
        }
        size_t slot = bucket_index(search, x, y, z);
        point->next = search->table[slot].first;
        search->table[slot] = (Bucket){x, y, z, i + 1};
    }
    return 1;
}

static size_t nearest_cell(const Search *search, double x, double y, double z) {
    if (!search->count) {
        return 0;
    }
    int ix, iy, iz;
    if (!bin(x, search->bound, &ix) || !bin(y, search->bound, &iy) || !bin(z, search->bound, &iz)) {
        return 0;
    }
    size_t best = 0;
    double distance = search->bound * search->bound;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                size_t bucket = bucket_index(search, ix + dx, iy + dy, iz + dz);
                size_t i = search->table[bucket].first;
                while (i) {
                    const Cell *p = search->cells + i - 1;
                    double a = p->x - x, b = p->y - y, c = p->z - z;
                    double d = a * a + b * b + c * c;
                    if (d < distance || (best && d == distance && i < best)) {
                        best = i;
                        distance = d;
                    }
                    i = p->next;
                }
            }
        }
    }
    return best;
}

static size_t nearest(const Search *search, const Node *node) {
    return nearest_cell(search, MATCH_SCALE * (double)node->x, MATCH_SCALE * (double)node->y,
        MATCH_SCALE * (double)node->z);
}

static char *block_path(const Side *side, const Block *block) {
    char *path = NULL;
    return asprintf(&path, "%s/piece-%04lld", side->folder, (long long)block->id) >= 0 ? path
                                                                                       : NULL;
}

static double in_mm(double z, double voxel, int single) {
    return single ? (double)((float)z * (float)voxel) : z * voxel;
}

static int search_open(const Side *side, Block *block, double seam) {
    if (block->loaded) {
        return 1;
    }
    char *path = block_path(side, block);
    json_object *meta = path ? read_record(path, "meta.json") : NULL;
    double voxel = json_object_get_double(json_object_object_get(meta, "voxel_um")) / 1000;
    double stride = json_object_get_double(json_object_object_get(meta, "stride"));
    uint32_t width = 0, height = 0;
    int type = 0;
    double *mesh = voxel > 0 && isfinite(voxel) && stride > 0 && stride < INT_MAX
        ? read_raw_mesh(path, &width, &height, &type)
        : NULL;
    int ok = mesh != NULL, single = type == 32;
    Search *search = &block->search;
    search->bound = 3 * stride;
    if (ok) {
        size_t size = (size_t)width * height, n = 0;
        double start = seam - 1.5, stop = seam + 1.5;
        if (single) {
            start = (float)start;
            stop = (float)stop;
        }
        const double *z = mesh + 2 * size;
        for (size_t i = 0; i < size; i++) {
            double mm = in_mm(z[i], voxel, single);
            n += mesh[i] > 0 && mm >= start && mm <= stop;
        }
        search->cells = n ? calloc(n, sizeof(*search->cells)) : NULL;
        ok = !n || search->cells;
        for (size_t i = 0; ok && n && i < size; i++) {
            double mm = in_mm(z[i], voxel, single);
            if (mesh[i] > 0 && mm >= start && mm <= stop) {
                search->cells[search->count++] = (Cell){.x = mesh[i],
                    .y = mesh[size + i],
                    .z = z[i],
                    .row = i / width,
                    .column = i % width};
            }
        }
        ok = ok && (!n || index_cells(search));
    }
    free(mesh);
    json_object_put(meta);
    free(path);
    block->loaded = ok;
    return ok;
}

static int crossing_order(const void *one, const void *two) {
    const Crossing *a = one, *b = two;
    if (a->lower != b->lower) {
        return a->lower < b->lower ? -1 : 1;
    }
    if (a->upper != b->upper) {
        return a->upper < b->upper ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static size_t crossing_at(const Crossing *rows, size_t count, int64_t a, int64_t b) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (rows[m].lower < a || (rows[m].lower == a && rows[m].upper < b)) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    return lo;
}

static Crossing *crossings(const SeamField *field, size_t *used, double *seam) {
    size_t count = 0;
    for (int64_t i = 0; i < field->edge_count; i++) {
        Edge e = field->edges[i];
        count += field->side[e.a] != field->side[e.b] && field->piece[e.a] >= 0 &&
            field->piece[e.b] >= 0;
    }
    Crossing *out = calloc(count ? count : 1, sizeof(*out));
    double *heights = calloc(count ? count : 1, sizeof(*heights));
    if (!out || !heights) {
        free(out);
        free(heights);
        return NULL;
    }
    size_t n = 0;
    for (int64_t i = 0; i < field->edge_count; i++) {
        Edge e = field->edges[i];
        int a = field->side[e.a] ? e.b : e.a, b = field->side[e.a] ? e.a : e.b;
        if (field->side[a] != field->side[b] && field->piece[a] >= 0 && field->piece[b] >= 0) {
            out[n] = (Crossing){field->piece[a], field->piece[b], i, a, b};
            heights[n++] = MATCH_SCALE * (double)field->nodes[a].z;
        }
    }
    *used = n;
    *seam = n ? median_double(heights, n) : 0;
    free(heights);
    qsort(out, n, sizeof(*out), crossing_order);
    return out;
}

static void joint_cells(Joint *joint, double *values, size_t count, const Block *a, const Block *b,
    const size_t *lower, const size_t *upper, size_t hits) {
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        if (lower[i] && upper[i]) {
            const Cell *p = a->search.cells + lower[i] - 1;
            const Cell *q = b->search.cells + upper[i] - 1;
            values[used] = p->column;
            values[hits + used] = q->column;
            values[2 * hits + used] = p->row;
            values[3 * hits + used++] = q->row;
        }
    }
    *joint = (Joint){.hits = hits, .a = a->id, .b = b->id};
    for (int i = 0; i < 2; i++) {
        joint->x[i] = median_double(values + i * hits, hits);
        joint->y[i] = median_double(values + (i + 2) * hits, hits);
    }
}

static int find_joint(const SeamField *field, Side sides[2], HxJoin join, const Crossing *cross,
    size_t count, double seam, Joint *best) {
    size_t ia = block_at(sides, join.lower), ib = block_at(sides + 1, join.upper), end = ib;
    while (end < sides[1].count && sides[1].blocks[end].source == join.upper) {
        end++;
    }
    size_t blocks = end - ib;
    if (!count || !blocks || ia == sides[0].count || sides[0].blocks[ia].source != join.lower) {
        return 1;
    }
    if (count > SIZE_MAX / sizeof(size_t) / (blocks + 1) ||
        count > SIZE_MAX / (4 * sizeof(double))) {
        return 0;
    }
    size_t *indices = calloc(count * (blocks + 1), sizeof(*indices));
    uint8_t *loaded = calloc(blocks, 1);
    double *values = malloc(4 * count * sizeof(*values));
    int ok = indices && loaded && values;
    for (; ok && ia < sides[0].count && sides[0].blocks[ia].source == join.lower; ia++) {
        Block *a = sides[0].blocks + ia;
        ok = search_open(sides, a, seam);
        size_t found = 0;
        for (size_t k = 0; ok && k < count; k++) {
            indices[k] = nearest(&a->search, field->nodes + cross[k].a);
            found += indices[k] != 0;
        }
        if (found < 3 || found <= best->hits) {
            continue;
        }
        for (size_t j = ib; ok && j < end; j++) {
            Block *b = sides[1].blocks + j;
            size_t *upper = indices + (j - ib + 1) * count;
            if (!loaded[j - ib]) {
                ok = search_open(sides + 1, b, seam);
                for (size_t k = 0; ok && k < count; k++) {
                    upper[k] = nearest(&b->search, field->nodes + cross[k].b);
                }
                loaded[j - ib] = 1;
            }
            size_t both = 0;
            for (size_t k = 0; ok && k < count; k++) {
                both += indices[k] && upper[k];
            }
            if (both >= 3 && both > best->hits) {
                joint_cells(best, values, count, a, b, indices, upper, both);
            }
        }
    }
    free(values);
    free(loaded);
    free(indices);
    return ok;
}

static json_object *endpoint(int64_t unit, int64_t piece) {
    json_object *list = json_object_new_array_ext(2);
    if (list) {
        json_object_array_add(list, json_object_new_int64(unit));
        json_object_array_add(list, json_object_new_int64(piece));
    }
    return list;
}

static json_object *joint_json(Joint *joint, int unit) {
    json_object *row = json_object_new_object();
    if (row) {
        json_object_object_add(row, "a", endpoint(unit, joint->a));
        json_object_object_add(row, "b", endpoint(unit + 1LL, joint->b));
        json_object_object_add(row, "edges", json_object_new_int64(joint->hits));
        json_object_object_add(row, "columns", number_array(joint->x, 2));
        json_object_object_add(row, "rows", number_array(joint->y, 2));
    }
    return row;
}

static double first_voxel(const Side *side) {
    if (!side->count) {
        return 0;
    }
    const Block *first = side->blocks;
    for (size_t i = 1; i < side->count; i++) {
        if (side->blocks[i].order < first->order) {
            first = side->blocks + i;
        }
    }
    char *path = block_path(side, first);
    json_object *meta = path ? read_record(path, "meta.json") : NULL;
    double voxel = json_object_get_double(json_object_object_get(meta, "voxel_um")) / 1000;
    json_object_put(meta);
    free(path);
    return voxel;
}

static int layout_seam(json_object *result, const SeamField *field, Side sides[2], int unit) {
    int64_t n = hx_piece_joins(field->piece, field->side, field->nodes_count, field->edges,
        field->edge_count, field->pairs, field->weights, field->pair_count, NULL, 0);
    if (n <= 0 || n > INT_MAX) {
        return n == 0;
    }
    HxJoin *joins = calloc((size_t)n, sizeof(*joins));
    size_t count = 0;
    double seam = 0, voxel = first_voxel(sides);
    Crossing *cross = joins && voxel > 0 ? crossings(field, &count, &seam) : NULL;
    int ok = cross &&
        hx_piece_joins(field->piece, field->side, field->nodes_count, field->edges,
            field->edge_count, field->pairs, field->weights, field->pair_count, joins, n) == n;
    for (int64_t i = 0; ok && i < n; i++) {
        HxJoin j = joins[i];
        size_t a = crossing_at(cross, count, j.lower, j.upper);
        size_t size = (size_t)j.count;
        Joint best = {0};
        ok = a <= count && size <= count - a &&
            find_joint(field, sides, j, cross + a, size, seam * voxel, &best);
        if (ok && best.hits) {
            ok = !json_object_array_add(result, joint_json(&best, unit));
        }
    }
    free(cross);
    free(joins);
    return ok;
}

static int scan_seam(
    json_object *result, const char **nodes, const char *pieces, int unit, int framed) {
    json_object *record = read_record(nodes[0], "record.json");
    json_object *planes = json_object_object_get(record, "cell_planes");
    json_object *last =
        json_object_is_type(planes, json_type_array) ? json_object_array_get_idx(planes, 1) : NULL;
    int64_t plane = json_object_get_int64(last);
    int ok =
        json_object_is_type(last, json_type_int) && plane >= INT_MIN + 4 && plane <= INT_MAX - 4;
    json_object_put(record);
    if (!ok) {
        return 0;
    }
    char *lower = path_join(pieces, strrchr(nodes[0], '/') + 1);
    char *upper = path_join(pieces, strrchr(nodes[1], '/') + 1);
    Side sides[2] = {0};
    SeamField field = {0};
    ok = lower && upper && side_open(sides, lower, framed) && side_open(sides + 1, upper, framed) &&
        !seam_field_open(&field, nodes[0], nodes[1], lower, upper, (int)plane) &&
        layout_seam(result, &field, sides, unit);
    seam_field_close(&field);
    side_close(sides + 1);
    side_close(sides);
    free(upper);
    free(lower);
    return ok;
}

static json_object *scan_joints(const char *nodes, const char *pieces, int framed) {
    char *pattern = path_join(nodes, "rows-*");
    if (!pattern) {
        return NULL;
    }
    glob_t chunks = {0};
    int status = glob(pattern, 0, NULL, &chunks);
    free(pattern);
    json_object *result = status == 0 || status == GLOB_NOMATCH ? json_object_new_array() : NULL;
    for (size_t i = 1; result && i < chunks.gl_pathc; i++) {
        const char *neighbours[] = {chunks.gl_pathv[i - 1], chunks.gl_pathv[i]};
        if (i > INT_MAX || !scan_seam(result, neighbours, pieces, (int)(i - 1), framed)) {
            json_object_put(result);
            result = NULL;
        }
    }
    globfree(&chunks);
    return result;
}

int hx_seams_main(int argc, char **argv) {
    int framed = argc == 4 && strcmp(argv[3], "framed") == 0;
    if (argc != 4 || (!framed && strcmp(argv[3], "rows") != 0)) {
        fputs("Usage: ./run.sh level.layout.seams NODES PIECES rows|framed\n", stderr);
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    json_object *result = scan_joints(argv[1], argv[2], framed);
    int ok = result != NULL;
    if (result) {
        puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PLAIN));
    } else {
        fputs("HerculeX: could not find layout joints\n", stderr);
    }
    json_object_put(result);
    return ok ? 0 : 1;
}

#if !defined(HX_UNITY)
int main(int argc, char **argv) { return hx_seams_main(argc, argv); }
#endif

#ifdef HX_UNITY
#include "mosaic_layout.inc"
#include <sys/file.h>
#include <sys/wait.h>
#define Run ReferenceRun
#include "../../calibrate/unwrap/mesh_surfaces.c"
#undef Run
#define HX_WRAP_STUDY
#include "../../calibrate/unwrap/sheet_wrap_score.c"
#undef HX_WRAP_STUDY
#define HX_RULER_FIELD
#include "../../calibrate/unwrap/raw_sheet_ruler.c"
#undef HX_RULER_FIELD
#endif
