// Small sheet patches merged strongest first on the summed evidence between them: edges attract, rays repel.

#include "parallel_blocks.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)
#define GONE (-2)
#define OWNERS 8
#define HEAP_WAYS 4
#define AHEAD 16

typedef struct {
    int32_t key;
    float attract, repel;
} Entry;

typedef struct {
    Entry *slots;
    int32_t capacity, filled, live;
} Table;

typedef struct {
    float score;
    int32_t a, b;
} Item;

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} Pair;

typedef struct {
    Item *items;
    int64_t count, capacity;
} Heap;

static inline uint32_t slot_of(int32_t key, int32_t capacity) { return ((uint32_t)key * 2654435761u) & (uint32_t)(capacity - 1); }

static void table_init(Table *t, int32_t capacity) {
    t->slots = malloc(sizeof(Entry) * (size_t)capacity);
    t->capacity = capacity;
    t->filled = t->live = 0;
    for (int32_t i = 0; i < capacity; i++) t->slots[i].key = EMPTY;
}

static Entry *table_find(Table *t, int32_t key) {
    uint32_t i = slot_of(key, t->capacity);
    for (;;) {
        Entry *e = &t->slots[i];
        if (e->key == key) return e;
        if (e->key == EMPTY) return NULL;
        i = (i + 1) & (uint32_t)(t->capacity - 1);
    }
}

static void table_grow(Table *t, int32_t capacity) {
    Entry *old = t->slots;
    int32_t size = t->capacity;
    table_init(t, capacity);
    for (int32_t i = 0; i < size; i++) {
        if (old[i].key < 0) continue;
        uint32_t s = slot_of(old[i].key, capacity);
        while (t->slots[s].key != EMPTY) s = (s + 1) & (uint32_t)(capacity - 1);
        t->slots[s] = old[i];
        t->filled++;
        t->live++;
    }
    free(old);
}

static Entry *table_add(Table *t, int32_t key, float attract, float repel) {
    Entry *found = table_find(t, key);
    if (found) {
        found->attract += attract;
        found->repel += repel;
        return found;
    }
    if ((int64_t)(t->filled + 1) * 10 >= (int64_t)t->capacity * 7) table_grow(t, t->live * 4 >= t->capacity ? t->capacity * 2 : t->capacity);
    uint32_t i = slot_of(key, t->capacity);
    while (t->slots[i].key >= 0) i = (i + 1) & (uint32_t)(t->capacity - 1);
    if (t->slots[i].key == EMPTY) t->filled++;
    t->live++;
    t->slots[i] = (Entry){key, attract, repel};
    return &t->slots[i];
}

static void table_remove(Table *t, int32_t key) {
    Entry *e = table_find(t, key);
    if (e) {
        e->key = GONE;
        t->live--;
    }
}

static inline int before(const Item *p, const Item *q) {
    if (p->score != q->score) return p->score > q->score;
    if (p->a != q->a) return p->a < q->a;
    return p->b < q->b;
}

static void heap_push(Heap *h, float score, int32_t a, int32_t b) {
    if (h->count == h->capacity) {
        h->capacity *= 2;
        h->items = realloc(h->items, sizeof(Item) * (size_t)h->capacity);
    }
    int64_t i = h->count++;
    Item item = {score, a < b ? a : b, a < b ? b : a};
    while (i > 0) {
        int64_t up = (i - 1) / HEAP_WAYS;
        if (!before(&item, &h->items[up])) break;
        h->items[i] = h->items[up];
        i = up;
    }
    h->items[i] = item;
}

static Item heap_pop(Heap *h) {
    Item top = h->items[0], last = h->items[--h->count];
    int64_t i = 0;
    for (;;) {
        int64_t first = HEAP_WAYS * i + 1, c = first;
        if (c >= h->count) break;
        for (int64_t k = first + 1; k < first + HEAP_WAYS && k < h->count; k++)
            if (before(&h->items[k], &h->items[c])) c = k;
        if (!before(&h->items[c], &last)) break;
        h->items[i] = h->items[c];
        i = c;
    }
    if (h->count) h->items[i] = last;
    return top;
}

void hx_agglomerate(int64_t patches, const int32_t *patch, int64_t nodes, const Pair *edges, const uint8_t *use, int64_t edge_count, const Pair *rays, int64_t ray_count,
                    float repel_scale, float floor_score, float neck, float neck_cap, int32_t *label, int64_t *stats) {
    Table *tables = malloc(sizeof(Table) * (size_t)patches);
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)patches);
    float *mass = calloc((size_t)patches, sizeof(float));
    for (int64_t i = 0; i < nodes; i++) mass[patch[i]] += 1;
    EACH_BLOCK(owner, OWNERS) {
        int64_t lo = patches * (int64_t)owner / OWNERS, hi = patches * (int64_t)(owner + 1) / OWNERS;
        for (int64_t i = lo; i < hi; i++) {
            table_init(&tables[i], 8);
            parent[i] = (int32_t)i;
        }
        for (int64_t e = 0; e < edge_count; e++) {
            if (e + AHEAD < edge_count) {
                __builtin_prefetch(&patch[edges[e + AHEAD].a]);
                __builtin_prefetch(&patch[edges[e + AHEAD].b]);
            }
            if (!use[e]) continue;
            int32_t a = patch[edges[e].a], b = patch[edges[e].b];
            if (a == b) continue;
            if (a >= lo && a < hi) table_add(&tables[a], b, edges[e].weight, 0);
            if (b >= lo && b < hi) table_add(&tables[b], a, edges[e].weight, 0);
        }
        for (int64_t e = 0; e < ray_count; e++) {
            if (e + AHEAD < ray_count) {
                __builtin_prefetch(&patch[rays[e + AHEAD].a]);
                __builtin_prefetch(&patch[rays[e + AHEAD].b]);
            }
            int32_t a = patch[rays[e].a], b = patch[rays[e].b];
            if (a == b) continue;
            if (a >= lo && a < hi) table_add(&tables[a], b, 0, rays[e].weight);
            if (b >= lo && b < hi) table_add(&tables[b], a, 0, rays[e].weight);
        }
    } END_BLOCKS;
    int64_t pairs = 0;
    for (int64_t i = 0; i < patches; i++) pairs += tables[i].live;
    Heap heap = {malloc(sizeof(Item) * (size_t)(pairs / 2 + 16)), 0, pairs / 2 + 16};
    for (int64_t i = 0; i < patches; i++) {
        Table *t = &tables[i];
        for (int32_t s = 0; s < t->capacity; s++) {
            Entry *e = &t->slots[s];
            if (e->key <= i) continue;
            float score = e->attract - repel_scale * e->repel;
            if (score > floor_score) heap_push(&heap, score, (int32_t)i, e->key);
        }
    }
    while (heap.count) {
        Item top = heap_pop(&heap);
        stats[1]++;
        if (parent[top.a] != top.a || parent[top.b] != top.b) continue;
        Entry *e = table_find(&tables[top.a], top.b);
        if (!e) continue;
        float score = e->attract - repel_scale * e->repel;
        if (score < top.score) {
            stats[2]++;
            if (score > floor_score) heap_push(&heap, score, top.a, top.b);
            continue;
        }
        float smaller = mass[top.a] < mass[top.b] ? mass[top.a] : mass[top.b];
        float boundary = sqrtf(smaller);
        if (score < neck * (boundary < neck_cap ? boundary : neck_cap)) {
            stats[3]++;
            continue;
        }
        int32_t keep = tables[top.a].live >= tables[top.b].live ? top.a : top.b, gone = keep == top.a ? top.b : top.a;
        table_remove(&tables[keep], gone);
        Table *from = &tables[gone];
        for (int32_t i = 0; i < from->capacity; i++) {
            Entry *f = &from->slots[i];
            if (f->key < 0 || f->key == keep) continue;
            int32_t c = f->key;
            float at = f->attract, rp = f->repel;
            table_remove(&tables[c], gone);
            Entry *mine = table_add(&tables[keep], c, at, rp);
            table_add(&tables[c], keep, at, rp);
            float fresh = mine->attract - repel_scale * mine->repel;
            if (fresh > floor_score) heap_push(&heap, fresh, keep, c);
        }
        free(from->slots);
        from->slots = NULL;
        from->capacity = from->live = from->filled = 0;
        parent[gone] = keep;
        mass[keep] += mass[gone];
        stats[0]++;
    }
    for (int64_t i = 0; i < patches; i++) {
        int32_t r = (int32_t)i;
        while (parent[r] != r) r = parent[r];
        int32_t node = (int32_t)i;
        while (parent[node] != r) { int32_t up = parent[node]; parent[node] = r; node = up; }
        label[i] = r;
        free(tables[i].slots);
    }
    free(tables); free(parent); free(mass); free(heap.items);
}
