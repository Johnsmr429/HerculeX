#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { HX_JOIN_SPREAD = 3, HX_JOIN_MIN_EDGES = 3 };

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} HxJoinEdge;

typedef struct {
    int32_t a, b, rank;
} HxJoinPair;

typedef struct {
    int64_t lower, upper, count;
    double weight;
} HxJoin;

typedef struct {
    HxJoin *rows;
    size_t size, used;
} HxJoinTable;

static size_t slot_of(const HxJoinTable *table, int64_t lower, int64_t upper) {
    uint64_t key = (uint64_t)lower * UINT64_C(0x9e3779b97f4a7c15) + (uint64_t)upper;
    key ^= key >> 30;
    key *= UINT64_C(0xbf58476d1ce4e5b9);
    key ^= key >> 27;
    size_t slot = key & (table->size - 1);
    while (table->rows[slot].count &&
        (table->rows[slot].lower != lower || table->rows[slot].upper != upper))
        slot = (slot + 1) & (table->size - 1);
    return slot;
}

static int grow(HxJoinTable *table) {
    size_t size = table->size ? table->size * 2 : 64;
    if (size <= table->size || size > SIZE_MAX / sizeof(*table->rows))
        return -1;
    HxJoinTable next = {
        .rows = calloc(size, sizeof(*next.rows)), .size = size, .used = table->used};
    if (!next.rows)
        return -1;
    for (size_t i = 0; i < table->size; i++)
        if (table->rows[i].count)
            next.rows[slot_of(&next, table->rows[i].lower, table->rows[i].upper)] = table->rows[i];
    free(table->rows);
    *table = next;
    return 0;
}

static int add_edge(HxJoinTable *table, int64_t lower, int64_t upper) {
    if (!table->size && grow(table))
        return -1;
    size_t slot = slot_of(table, lower, upper);
    if (!table->rows[slot].count) {
        if (table->used >= table->size - table->size / 4) {
            if (grow(table))
                return -1;
            slot = slot_of(table, lower, upper);
        }
        table->rows[slot].lower = lower;
        table->rows[slot].upper = upper;
        table->used++;
    }
    table->rows[slot].count++;
    return 0;
}

static int join_before(const void *left, const void *right) {
    const HxJoin *a = left, *b = right;
    if (a->count != b->count)
        return a->count > b->count ? -1 : 1;
    if (a->lower != b->lower)
        return a->lower < b->lower ? -1 : 1;
    return (a->upper > b->upper) - (a->upper < b->upper);
}

int hx_spread_piece_labels(int64_t *piece, const uint8_t *side, int64_t nodes,
    const HxJoinEdge *edges, int64_t edge_count) {
    int64_t *next = malloc((size_t)(nodes ? nodes : 1) * sizeof(*next));
    if (!next)
        return -1;
    for (int round = 0; round < HX_JOIN_SPREAD; round++) {
        int changed = 0;
        for (int direction = 0; direction < 2; direction++) {
            memcpy(next, piece, (size_t)nodes * sizeof(*next));
            for (int64_t i = 0; i < edge_count; i++) {
                int32_t a = direction ? edges[i].b : edges[i].a;
                int32_t b = direction ? edges[i].a : edges[i].b;
                if (a < 0 || b < 0 || a >= nodes || b >= nodes) {
                    free(next);
                    return -1;
                }
                if (side[a] == side[b] && piece[a] < 0 && piece[b] >= 0) {
                    next[a] = piece[b];
                    changed = 1;
                }
            }
            memcpy(piece, next, (size_t)nodes * sizeof(*next));
        }
        if (!changed)
            break;
    }
    free(next);
    return 0;
}

int64_t hx_piece_joins(const int64_t *piece, const uint8_t *side, int64_t nodes,
    const HxJoinEdge *edges, int64_t edge_count, const HxJoinPair *pairs, const float *weights,
    int64_t pair_count, HxJoin *out, int64_t capacity) {
    HxJoinTable table = {0};
    for (int64_t i = 0; i < edge_count; i++) {
        int32_t a = edges[i].a, b = edges[i].b;
        if (a < 0 || b < 0 || a >= nodes || b >= nodes) {
            free(table.rows);
            return -1;
        }
        if (side[a] == side[b] || piece[a] < 0 || piece[b] < 0)
            continue;
        int64_t lower = side[a] == 0 ? piece[a] : piece[b];
        int64_t upper = side[a] == 0 ? piece[b] : piece[a];
        if (add_edge(&table, lower, upper)) {
            free(table.rows);
            return -1;
        }
    }
    for (int64_t i = 0; i < pair_count; i++) {
        int32_t a = pairs[i].a, b = pairs[i].b;
        if (a < 0 || b < 0 || a >= nodes || b >= nodes) {
            free(table.rows);
            return -1;
        }
        if (side[a] == side[b] || piece[a] < 0 || piece[b] < 0 || !table.size)
            continue;
        int64_t lower = side[a] == 0 ? piece[a] : piece[b];
        int64_t upper = side[a] == 0 ? piece[b] : piece[a];
        size_t slot = slot_of(&table, lower, upper);
        if (table.rows[slot].count)
            table.rows[slot].weight += weights[i];
    }
    int64_t kept = 0;
    for (size_t i = 0; i < table.size; i++)
        if (table.rows[i].count >= HX_JOIN_MIN_EDGES &&
            table.rows[i].weight <= 0.05 * (double)table.rows[i].count)
            table.rows[kept++] = table.rows[i];
    qsort(table.rows, (size_t)kept, sizeof(*table.rows), join_before);
    if (out && kept <= capacity)
        memcpy(out, table.rows, (size_t)kept * sizeof(*out));
    else if (out)
        kept = -1;
    free(table.rows);
    return kept;
}
