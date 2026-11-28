#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HX_SLAB_CELLS 8
#define HX_WRITE_BLOCK 8192

typedef struct {
    const uint8_t *bricks;
    const int64_t *offsets;
    int64_t bytes;
    int depth, height, width, bz, by, bx;
    int cell, margin, last;
} NodeVolume;

typedef struct {
    NodeVolume *volume;
    uint8_t *voxels;
    int plane, z0, z1, error;
} NodeSlab;

static int smaller(int a, int b) { return a < b ? a : b; }
static int grid_span(int length, int step) { return (length - 1) / step + 1; }

static void *read_node_slab(void *context) {
    NodeSlab *s = context;
    NodeVolume *v = s->volume;
    int begin = s->plane * v->cell - v->margin;
    int end = smaller(s->plane + HX_SLAB_CELLS, v->last);
    s->z0 = begin > 0 ? begin : 0;
    s->z1 = smaller(end * v->cell + v->margin, v->depth);
    int gy = grid_span(v->height, v->by), gx = grid_span(v->width, v->bx);
    int64_t brick_size = (int64_t)v->bz * v->by * v->bx;
    for (int z = s->z0; z < s->z1; z++) {
        int64_t iz = z / v->bz, slice = (int64_t)(z % v->bz) * v->by * v->bx;
        for (int iy = 0; iy < gy; iy++) {
            int rows = smaller(v->by, v->height - iy * v->by);
            for (int ix = 0; ix < gx; ix++) {
                int64_t offset = v->offsets[(iz * gy + iy) * gx + ix];
                int x = ix * v->bx, columns = smaller(v->bx, v->width - x);
                if (offset >= 0 && offset > v->bytes - brick_size) {
                    s->error = 1;
                    return NULL;
                }
                int64_t row0 = (int64_t)(z - s->z0) * v->height + iy * v->by;
                for (int y = 0; y < rows; y++) {
                    uint8_t *row = s->voxels + (row0 + y) * v->width + x;
                    if (offset < 0)
                        memset(row, 0, columns);
                    else
                        memcpy(row, v->bricks + offset + slice + (int64_t)y * v->bx, columns);
                }
            }
        }
    }
    return NULL;
}

typedef struct {
    uint8_t bytes[20];
} PackedNode;

typedef struct {
    PackedNode nodes[HX_WRITE_BLOCK];
    int32_t ids[HX_WRITE_BLOCK];
} NodeBatch;

typedef struct {
    uint32_t start, count;
} CellRange;

static int write_node_batch(FILE *nodes, FILE *cells, const NodeBatch *batch, size_t count) {
    return fwrite(batch->nodes, sizeof(*batch->nodes), count, nodes) == count &&
        fwrite(batch->ids, sizeof(*batch->ids), count, cells) == count;
}

static int write_slab_nodes(FILE *nodes, FILE *cells, const PackedNode *raw, const int32_t *ids,
    uint32_t count, int32_t first, int slab_cells, CellRange *range, NodeBatch *batch) {
    memset(range, 0, slab_cells * sizeof(*range));
    for (uint32_t i = 0; i < count; i++) {
        int32_t cell = ids[i];
        if (cell < 0 || cell >= slab_cells)
            return 5;
        CellRange *r = range + cell;
        if (!r->count)
            r->start = i;
        if (r->start + r->count != i)
            return 5;
        r->count++;
    }
    size_t filled = 0;
    for (int cell = 0; cell < slab_cells; cell++) {
        CellRange r = range[cell];
        for (uint32_t i = 0; i < r.count; i++) {
            batch->nodes[filled] = raw[r.start + i];
            batch->ids[filled++] = first + cell;
            if (filled == HX_WRITE_BLOCK) {
                if (!write_node_batch(nodes, cells, batch, filled))
                    return 6;
                filled = 0;
            }
        }
    }
    return write_node_batch(nodes, cells, batch, filled) ? 0 : 6;
}

typedef int (*NodeDetector)(const uint8_t *, int, int, int, int, int, int, int, int, int, void *,
    int32_t *, uint32_t, uint32_t *, double *);

int hx_detect_volume(NodeDetector detect, const uint8_t *bricks, int64_t bytes,
    const int64_t *offsets, const int32_t *shape, int cell, int margin, int air, int first,
    int last, const char *node_path, const char *cell_path, int64_t *total, double *gpu_time) {
    NodeVolume v = {.bricks = bricks,
        .offsets = offsets,
        .bytes = bytes,
        .depth = shape[0],
        .height = shape[1],
        .width = shape[2],
        .bz = shape[3],
        .by = shape[4],
        .bx = shape[5],
        .cell = cell,
        .margin = margin,
        .last = last};
    int cy = grid_span(v.height, cell), cx = grid_span(v.width, cell);
    int planes = smaller(HX_SLAB_CELLS, last - first);
    int64_t layer = (int64_t)cy * cx;
    if (layer > INT_MAX / (last - first))
        return 5;
    int slab_cells = planes * layer, nz = smaller(v.depth, planes * cell + 2 * margin);
    int64_t slab_bytes = (int64_t)nz * v.height * v.width;
    uint32_t capacity = slab_cells > INT_MAX / 60 ? INT_MAX / 20 : slab_cells * 3;
    NodeSlab slab[2] = {{.volume = &v, .voxels = malloc(slab_bytes), .plane = first},
        {.volume = &v, .voxels = last - first > planes ? malloc(slab_bytes) : NULL}};
    PackedNode *raw = malloc((size_t)capacity * sizeof(*raw));
    int32_t *ids = malloc((size_t)capacity * sizeof(*ids));
    CellRange *ranges = malloc(slab_cells * sizeof(*ranges));
    NodeBatch *batch = malloc(sizeof(*batch));
    FILE *files[] = {fopen(node_path, "wb"), fopen(cell_path, "wb")};
    int status = 6;
    if (!files[0] || !files[1])
        goto done;
    status = 2;
    if (!slab[0].voxels || !raw || !ids || !ranges || !batch)
        goto done;
    read_node_slab(&slab[0]);
    int slot = 0;
    for (int p = first; p < last; p += planes) {
        NodeSlab *s = &slab[slot];
        int end = smaller(p + planes, last);
        if (s->error) {
            status = 5;
            goto done;
        }
        NodeSlab *next = &slab[1 - slot];
        next->plane = end;
        pthread_t worker;
        int ahead =
            end < last && next->voxels && !pthread_create(&worker, NULL, read_node_slab, next);
        uint32_t found = 0;
        double seconds = 0;
        for (;;) {
            status = detect(s->voxels, s->z1 - s->z0, v.height, v.width, end - p, cy, cx,
                p * cell - s->z0, air, s->z0, raw, ids, capacity, &found, &seconds);
            if (status != 4 || found <= capacity)
                break;
            free(raw);
            free(ids);
            capacity = found;
            raw = malloc((size_t)capacity * sizeof(*raw));
            ids = malloc((size_t)capacity * sizeof(*ids));
            if (!raw || !ids) {
                status = 2;
                break;
            }
        }
        if (!status)
            status = write_slab_nodes(files[0], files[1], raw, ids, found, (p - first) * layer,
                (end - p) * layer, ranges, batch);
        if (ahead)
            pthread_join(worker, NULL);
        if (status)
            goto done;
        *total += found;
        *gpu_time += seconds;
        if (end < last) {
            if (next->voxels)
                slot = 1 - slot;
            slab[slot].plane = end;
            if (!ahead)
                read_node_slab(&slab[slot]);
        }
    }
done:
    for (size_t i = 0; i < sizeof(files) / sizeof(*files); i++)
        if (files[i] && fclose(files[i]))
            status = 6;
    free(batch);
    free(ranges);
    free(ids);
    free(raw);
    free(slab[1].voxels);
    free(slab[0].voxels);
    return status;
}
