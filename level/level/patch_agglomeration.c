#include "parallel_blocks.h"
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)
#define GONE (-2)
#define OWNERS 8
#define HEAP_WAYS 4
#define AHEAD 16
#define COMPACT_SLOTS 32

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

enum { RUN_BLOCK = 8192 };

typedef struct {
    Item *items;
    size_t at, n, left;
    int64_t offset;
} Run;

typedef struct {
    Item *items;
    int64_t count, capacity, limit, written;
    Run *runs;
    size_t nruns, allocated;
    int fd;
} Heap;

static inline uint32_t slot_of(int32_t key, int32_t capacity) {
    return ((uint32_t)key * 2654435761u) & (uint32_t)(capacity - 1);
}

static Entry *table_find(Table *t, int32_t key) {
    if (t->capacity <= COMPACT_SLOTS) {
        for (int32_t i = 0; i < t->live; i++) {
            if (t->slots[i].key == key) {
                return t->slots + i;
            }
        }
        return NULL;
    }
    uint32_t i = slot_of(key, t->capacity);
    for (;;) {
        Entry *e = &t->slots[i];
        if (e->key == key)
            return e;
        if (e->key == EMPTY)
            return NULL;
        i = (i + 1) & (uint32_t)(t->capacity - 1);
    }
}

enum { SLOT_CLASSES = 10, SLOTS_LIMIT = 2 << (SLOT_CLASSES - 1) };

typedef struct SlotPage {
    struct SlotPage *older;
    size_t available, bytes;
    Entry entries[];
} SlotPage;

typedef struct {
    SlotPage *pages[SLOT_CLASSES];
    Entry *recycled[SLOT_CLASSES];
    size_t page_bytes;
} SlotPool;

static Entry *new_slots(SlotPool *pool, int32_t capacity) {
    size_t bytes = (size_t)capacity * sizeof(Entry);
    if (capacity > SLOTS_LIMIT || !pool->page_bytes) {
        return malloc(bytes);
    }
    int bin = __builtin_ctz((unsigned)capacity) - 1;
    Entry *slots = pool->recycled[bin];
    if (slots) {
        memcpy(&pool->recycled[bin], (const unsigned char *)slots, sizeof(Entry *));
        return slots;
    }
    SlotPage *page = pool->pages[bin];
    if (!page || !page->available) {
        size_t blocks = pool->page_bytes / bytes;
        size_t extent = sizeof(*page) + blocks * bytes;
        page = mmap(NULL, extent, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (page == MAP_FAILED) {
            return NULL;
        }
        page->bytes = extent;
        page->older = pool->pages[bin];
        page->available = blocks;
        pool->pages[bin] = page;
    }
    return page->entries + --page->available * (size_t)capacity;
}

static void return_slots(SlotPool *pool, Entry *slots, int32_t capacity) {
    if (capacity > SLOTS_LIMIT || !pool->page_bytes) {
        free(slots);
        return;
    }
    int bin = __builtin_ctz((unsigned)capacity) - 1;
    memcpy((unsigned char *)slots, &pool->recycled[bin], sizeof(Entry *));
    pool->recycled[bin] = slots;
}

static int table_grow(SlotPool *pool, Table *t, int32_t capacity) {
    Entry *old = t->slots;
    int32_t size = t->capacity;
    Entry *slots = new_slots(pool, capacity);
    if (!slots)
        return 0;
    if (capacity <= COMPACT_SLOTS) {
        if (t->live) {
            memcpy(slots, old, t->live * sizeof(*slots));
        }
    } else {
        memset(slots, 0xff, sizeof(Entry) * (size_t)capacity);
        int32_t end = size <= COMPACT_SLOTS ? t->live : size;
        for (int32_t i = 0; i < end; i++) {
            if (old[i].key < 0) {
                continue;
            }
            uint32_t at = slot_of(old[i].key, capacity);
            while (slots[at].key != EMPTY) {
                at = (at + 1) & (uint32_t)(capacity - 1);
            }
            slots[at] = old[i];
        }
    }
    t->slots = slots;
    t->capacity = capacity;
    t->filled = t->live;
    if (old) {
        return_slots(pool, old, size);
    }
    return 1;
}

static Entry *table_add(SlotPool *pool, Table *t, int32_t key, float attract, float repel) {
    Entry *found = table_find(t, key);
    if (found) {
        found->attract += attract;
        found->repel += repel;
        return found;
    }
    if (t->capacity <= COMPACT_SLOTS) {
        if (t->live == t->capacity && !table_grow(pool, t, t->capacity ? 2 * t->capacity : 2)) {
            return NULL;
        }
        if (t->capacity <= COMPACT_SLOTS) {
            Entry *slot = &t->slots[t->live++];
            *slot = (Entry){key, attract, repel};
            return slot;
        }
    }
    if ((int64_t)(t->filled + 1) * 10 >= (int64_t)t->capacity * 7) {
        int grow = (int64_t)t->live * 4 >= t->capacity;
        if ((grow && t->capacity > INT32_MAX / 2) ||
            !table_grow(pool, t, grow ? t->capacity * 2 : t->capacity))
            return NULL;
    }
    uint32_t i = slot_of(key, t->capacity);
    while (t->slots[i].key >= 0)
        i = (i + 1) & (uint32_t)(t->capacity - 1);
    if (t->slots[i].key == EMPTY)
        t->filled++;
    t->live++;
    t->slots[i] = (Entry){key, attract, repel};
    return &t->slots[i];
}

static void table_remove(Table *t, int32_t key) {
    Entry *e = table_find(t, key);
    if (e) {
        t->live--;
        if (t->capacity <= COMPACT_SLOTS) {
            *e = t->slots[t->live];
        } else {
            e->key = GONE;
        }
    }
}

static inline int before(const Item *p, const Item *q) {
    if (p->score != q->score)
        return p->score > q->score;
    if (p->a != q->a)
        return p->a < q->a;
    return p->b < q->b;
}

static int sorted_item(const void *a, const void *b) {
    if (before(a, b)) {
        return -1;
    }
    return before(b, a);
}

static int transfer_items(int fd, Item *data, size_t n, int64_t offset, int writing) {
    size_t bytes = n * sizeof(*data);
    char *p = (char *)data;
    while (bytes) {
        ssize_t got = writing ? pwrite(fd, p, bytes, offset) : pread(fd, p, bytes, offset);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return 0;
        }
        p += got;
        bytes -= (size_t)got;
        offset += got;
    }
    return 1;
}

static int refill_run(Heap *h, Run *r) {
    size_t n = r->left < RUN_BLOCK ? r->left : RUN_BLOCK;
    if (!transfer_items(h->fd, r->items, n, r->offset, 0)) {
        return 0;
    }
    r->at = 0;
    r->n = n;
    r->left -= n;
    r->offset += (int64_t)(n * sizeof(Item));
    return 1;
}

static int spill_heap(Heap *h) {
    if (h->nruns == h->allocated) {
        size_t n = h->allocated ? h->allocated * 2 : 8;
        Run *runs = realloc(h->runs, n * sizeof(*runs));
        if (!runs) {
            return 0;
        }
        h->allocated = n;
        h->runs = runs;
    }
    qsort(h->items, (size_t)h->count, sizeof(Item), sorted_item);
    size_t keep = (size_t)h->count / 2, send = (size_t)h->count - keep;
    size_t head = send < RUN_BLOCK ? send : RUN_BLOCK;
    Run run = {
        .items = malloc(head * sizeof(Item)), .left = send - head, .n = head, .offset = h->written};
    if (!run.items) {
        return 0;
    }
    memcpy(run.items, h->items + keep, head * sizeof(Item));
    if (!transfer_items(h->fd, h->items + keep + head, run.left, h->written, 1)) {
        free(run.items);
        return 0;
    }
    h->written += (int64_t)(run.left * sizeof(Item));
    h->count = keep;
    size_t i = h->nruns++;
    while (i) {
        size_t parent = (i - 1) / 2;
        Run *p = &h->runs[parent];
        if (!before(run.items, p->items + p->at)) {
            break;
        }
        h->runs[i] = *p;
        i = parent;
    }
    h->runs[i] = run;
    return 1;
}

static int pop_run(Heap *h, Item *item) {
    Run r = h->runs[0];
    *item = r.items[r.at++];
    if (r.at == r.n) {
        if (r.left) {
            if (!refill_run(h, &r)) {
                return 0;
            }
        } else {
            free(r.items);
            if (!--h->nruns) {
                return 1;
            }
            r = h->runs[h->nruns];
        }
    }
    size_t pos = 0;
    while (pos * 2 + 1 < h->nruns) {
        size_t child = pos * 2 + 1;
        Run *c = &h->runs[child];
        if (child + 1 < h->nruns && before(c[1].items + c[1].at, c->items + c->at)) {
            child++;
            c++;
        }
        if (!before(c->items + c->at, r.items + r.at)) {
            break;
        }
        h->runs[pos] = *c;
        pos = child;
    }
    h->runs[pos] = r;
    return 1;
}

static int heap_push(Heap *h, float score, int32_t a, int32_t b) {
    if (h->limit && h->count == h->limit) {
        if (!spill_heap(h)) {
            return 0;
        }
    } else if (h->count == h->capacity) {
        if ((uint64_t)h->capacity > SIZE_MAX / (2 * sizeof(Item)))
            return 0;
        int64_t cap = h->capacity * 2;
        if (h->limit && cap > h->limit) {
            cap = h->limit;
        }
        Item *grown = realloc(h->items, sizeof(Item) * (size_t)cap);
        if (!grown)
            return 0;
        h->capacity = cap;
        h->items = grown;
    }
    int64_t i = h->count++;
    Item item = {score, a < b ? a : b, a < b ? b : a};
    while (i > 0) {
        int64_t up = (i - 1) / HEAP_WAYS;
        if (!before(&item, &h->items[up]))
            break;
        h->items[i] = h->items[up];
        i = up;
    }
    h->items[i] = item;
    return 1;
}

static Item heap_pop(Heap *h) {
    Item top = h->items[0], last = h->items[--h->count];
    int64_t i = 0;
    for (;;) {
        int64_t first = HEAP_WAYS * i + 1, c = first;
        if (c >= h->count)
            break;
        for (int64_t k = first + 1; k < first + HEAP_WAYS && k < h->count; k++)
            if (before(&h->items[k], &h->items[c]))
                c = k;
        if (!before(&h->items[c], &last))
            break;
        h->items[i] = h->items[c];
        i = c;
    }
    if (h->count)
        h->items[i] = last;
    return top;
}

enum { PATCH_BATCH = 131072 };

typedef struct {
    int32_t a, b;
    float weight;
} Contact;

typedef struct {
    Table *tables;
    SlotPool *pools;
    Contact *contacts;
    int32_t *order;
    int32_t ends[OWNERS + 1], failed[OWNERS];
    int rays;
} PatchBatch;

static void patch_tables(void *context, size_t owner) {
    PatchBatch *batch = context;
    for (int32_t i = batch->ends[owner]; i < batch->ends[owner + 1]; i++) {
        Contact *pair = &batch->contacts[batch->order[i]];
        float attract = batch->rays ? 0 : pair->weight;
        float repel = batch->rays ? pair->weight : 0;
        if (!table_add(&batch->pools[owner], &batch->tables[pair->a], pair->b, attract, repel)) {
            batch->failed[owner] = 1;
            break;
        }
    }
}

static int read_pairs(int fd, Pair *buffer, size_t count, off_t offset) {
    size_t size = count * sizeof(*buffer), done = 0;
    while (done < size) {
        ssize_t got = pread(fd, (char *)buffer + done, size - done, offset + (off_t)done);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return 0;
        }
        done += (size_t)got;
    }
    return 1;
}

static int fill_tables(Table *tables, SlotPool *pools, const int32_t *patch, const Pair *sources[2],
    const int64_t lengths[2], const int fds[2], const int64_t offsets[2], const uint8_t *use) {
    size_t capacity = lengths[0] > lengths[1] ? lengths[0] : lengths[1];
    if (capacity > PATCH_BATCH) {
        capacity = PATCH_BATCH;
    }
    if (!capacity) {
        return 1;
    }
    Contact *contacts = malloc(2 * capacity * sizeof(*contacts));
    int32_t *order = malloc(2 * capacity * sizeof(*order));
    int file = fds[0] >= 0 || fds[1] >= 0;
    Pair *buffer = file ? malloc(capacity * sizeof(*buffer)) : NULL;
    int ok = contacts && order && (!file || buffer);
    PatchBatch batch = {.tables = tables, .pools = pools, .contacts = contacts, .order = order};
    for (int rays = 0; ok && rays < 2; rays++) {
        int64_t n = lengths[rays];
        batch.rays = rays;
        for (int64_t first = 0; ok && first < n; first += PATCH_BATCH) {
            size_t size = n - first > PATCH_BATCH ? PATCH_BATCH : n - first;
            const Pair *pairs;
            if (fds[rays] >= 0) {
                ok = read_pairs(fds[rays], buffer, size, offsets[rays] + first * sizeof(*buffer));
                if (!ok) {
                    break;
                }
                pairs = buffer;
            } else {
                pairs = sources[rays] + first;
            }
            int32_t fill[OWNERS] = {0}, used = 0;
            for (size_t i = 0; i < size; i++) {
                if (!rays && use && !use[first + i]) {
                    continue;
                }
                int32_t a = patch[pairs[i].a], b = patch[pairs[i].b];
                if (a != b) {
                    contacts[used++] = (Contact){a, b, pairs[i].weight};
                    contacts[used++] = (Contact){b, a, pairs[i].weight};
                    fill[(unsigned)a % OWNERS]++;
                    fill[(unsigned)b % OWNERS]++;
                }
            }
            for (int i = 0; i < OWNERS; i++) {
                batch.ends[i + 1] = batch.ends[i] + fill[i];
                fill[i] = batch.ends[i];
            }
            for (int32_t i = 0; i < used; i++) {
                order[fill[(unsigned)contacts[i].a % OWNERS]++] = i;
            }
            each_block(OWNERS, &batch, patch_tables);
            for (int i = 0; i < OWNERS; i++) {
                ok = ok && !batch.failed[i];
            }
        }
    }
    free(buffer);
    free(order);
    free(contacts);
    return ok;
}

int hx_agglomerate_sources(int64_t patches, const int32_t *patch, int64_t nodes, const Pair *edges,
    const uint8_t *use, int64_t edge_count, const Pair *rays, int64_t ray_count, float repel_scale,
    float floor_score, float neck, float neck_cap, int32_t *label, int64_t *stats, int edge_fd,
    int64_t edge_offset, int ray_fd, int64_t ray_offset, int heap_fd, int64_t heap_items) {
    if (patches < 0 || patches > INT32_MAX)
        return -1;
    if (!patches)
        return 0;
    SlotPool pools[OWNERS] = {0};
    for (int owner = 0; owner < OWNERS; owner++) {
        pools[owner].page_bytes = patches >= 65536 ? 256 * 1024 : 0;
    }
    if (heap_fd >= 0 && heap_items < 2) {
        return -1;
    }
    Heap heap = {.fd = heap_fd, .limit = heap_fd >= 0 ? heap_items : 0};
    int status = -1;
    Table *tables = calloc((size_t)patches, sizeof(Table));
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)patches);
    float *mass = calloc((size_t)patches, sizeof(float));
    if (!tables || !parent || !mass)
        goto finish;
    for (int32_t i = 0; i < patches; i++) {
        parent[i] = i;
    }
    for (int64_t i = 0; i < nodes; i++) {
        if (patch[i] < 0 || patch[i] >= patches)
            goto finish;
        mass[patch[i]] += 1;
    }
    const Pair *sources[] = {edges, rays};
    int64_t lengths[] = {edge_count, ray_count}, offsets[] = {edge_offset, ray_offset};
    int fds[] = {edge_fd, ray_fd};
    if (!fill_tables(tables, pools, patch, sources, lengths, fds, offsets, use)) {
        goto finish;
    }
    int64_t pairs = 0;
    for (int64_t i = 0; i < patches; i++)
        pairs += tables[i].live;
    heap.capacity = pairs / 2 + 16;
    if (heap.limit && heap.capacity > heap.limit) {
        heap.capacity = heap.limit;
    }
    heap.items = malloc(sizeof(Item) * (size_t)heap.capacity);
    if (!heap.items)
        goto finish;
    for (int64_t i = 0; i < patches; i++) {
        Table *t = &tables[i];
        int32_t limit = t->capacity <= COMPACT_SLOTS ? t->live : t->capacity;
        for (int32_t s = 0; s < limit; s++) {
            Entry *e = &t->slots[s];
            if (e->key <= i)
                continue;
            float score = e->attract - repel_scale * e->repel;
            if (score > floor_score && !heap_push(&heap, score, (int32_t)i, e->key))
                goto finish;
        }
    }
    while (heap.count || heap.nruns) {
        Item top;
        if (heap.nruns &&
            (!heap.count || before(heap.runs[0].items + heap.runs[0].at, heap.items))) {
            if (!pop_run(&heap, &top)) {
                goto finish;
            }
        } else {
            top = heap_pop(&heap);
        }
        stats[1]++;
        if (parent[top.a] != top.a || parent[top.b] != top.b)
            continue;
        Entry *e = table_find(&tables[top.a], top.b);
        if (!e)
            continue;
        float score = e->attract - repel_scale * e->repel;
        if (score < top.score) {
            stats[2]++;
            if (score > floor_score && !heap_push(&heap, score, top.a, top.b))
                goto finish;
            continue;
        }
        float smaller = mass[top.a] < mass[top.b] ? mass[top.a] : mass[top.b];
        float boundary = sqrtf(smaller);
        if (score < neck * (boundary < neck_cap ? boundary : neck_cap)) {
            stats[3]++;
            continue;
        }
        int32_t keep = tables[top.a].live >= tables[top.b].live ? top.a : top.b,
                gone = keep == top.a ? top.b : top.a;
        table_remove(&tables[keep], gone);
        Table *from = &tables[gone];
        int32_t limit = from->capacity <= COMPACT_SLOTS ? from->live : from->capacity;
        for (int32_t i = 0; i < limit; i++) {
            Entry *f = &from->slots[i];
            if (f->key < 0 || f->key == keep)
                continue;
            int32_t c = f->key;
            float at = f->attract, rp = f->repel;
            table_remove(&tables[c], gone);
            Entry *mine = table_add(&pools[(unsigned)keep % OWNERS], &tables[keep], c, at, rp);
            if (!mine || !table_add(&pools[(unsigned)c % OWNERS], &tables[c], keep, at, rp))
                goto finish;
            float fresh = mine->attract - repel_scale * mine->repel;
            if (fresh > floor_score && !heap_push(&heap, fresh, keep, c))
                goto finish;
        }
        if (from->slots)
            return_slots(&pools[(unsigned)gone % OWNERS], from->slots, from->capacity);
        *from = (Table){0};
        parent[gone] = keep;
        mass[keep] += mass[gone];
        stats[0]++;
    }
    for (int64_t i = 0; i < patches; i++) {
        int32_t r = (int32_t)i;
        while (parent[r] != r)
            r = parent[r];
        int32_t node = (int32_t)i;
        while (parent[node] != r) {
            int32_t up = parent[node];
            parent[node] = r;
            node = up;
        }
        label[i] = r;
    }
    status = 0;
finish:
    if (tables)
        for (int64_t i = 0; i < patches; i++)
            if (tables[i].slots)
                return_slots(&pools[(unsigned)i % OWNERS], tables[i].slots, tables[i].capacity);
    for (int owner = 0; owner < OWNERS; owner++) {
        for (int bin = 0; bin < SLOT_CLASSES; bin++) {
            SlotPage *page = pools[owner].pages[bin];
            while (page) {
                SlotPage *older = page->older;
                munmap(page, page->bytes);
                page = older;
            }
        }
    }
    free(tables);
    free(parent);
    free(mass);
    for (size_t j = 0; j < heap.nruns; j++) {
        free(heap.runs[j].items);
    }
    free(heap.runs);
    free(heap.items);
    return status;
}

int hx_agglomerate(int64_t patches, const int32_t *patch, int64_t nodes, const Pair *edges,
    const uint8_t *use, int64_t edge_count, const Pair *rays, int64_t ray_count, float repel_scale,
    float floor_score, float neck, float neck_cap, int32_t *label, int64_t *stats) {
    return hx_agglomerate_sources(patches, patch, nodes, edges, use, edge_count, rays, ray_count,
        repel_scale, floor_score, neck, neck_cap, label, stats, -1, 0, -1, 0, -1, 0);
}
