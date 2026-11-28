#ifndef HX_UNITY
const char hx_usage[] = "Reassemble level-route tiles at their source grid offsets.";
#endif

typedef struct {
    const char *path;
    json_object *meta;
    i64 source, row, col, order;
    int h, w;
} V1Tile;

typedef struct {
    i64 source, tiles, cells;
} V1Source;

static const char *const v1_files[] = {"x.tif", "y.tif", "z.tif", "quality.npy"};

static int v1_tile_order(const void *left, const void *right) {
    const V1Tile *a = left, *b = right;
    if (a->source != b->source) {
        return a->source < b->source ? -1 : 1;
    }
    return a->order < b->order ? -1 : a->order != b->order;
}

static void v1_extent(Arena *a, V1Tile *tile) {
    const char *path = fmt(a, "%s/x.tif", tile->path);
    TIFF *f = TIFFOpen(path, "r");
    u32 w = 0, h = 0;
    need(f && TIFFGetField(f, TIFFTAG_IMAGEWIDTH, &w) && TIFFGetField(f, TIFFTAG_IMAGELENGTH, &h) &&
            w <= INT32_MAX && h <= INT32_MAX,
        "invalid tile shape in %s", path);
    TIFFClose(f);
    tile->w = w;
    tile->h = h;
    json_object *block = jget(tile->meta, "block");
    need(board_count(block) == 2, "%s has no row and column tile offsets", tile->path);
    tile->row = json_object_get_int64(json_object_array_get_idx(block, 0));
    tile->col = json_object_get_int64(json_object_array_get_idx(block, 1));
    need(tile->row >= 0 && tile->row <= INT32_MAX - h && tile->col >= 0 &&
            tile->col <= INT32_MAX - w,
        "invalid tile extent in %s", tile->path);
}

static void v1_paint(Arena *a, const V1Tile *tile, i64 width, f32 *xyz[3], u8 *quality) {
    size_t saved = a->used;
    const f32 *coord[3];
    for (int d = 0; d < 3; d++) {
        int w, h;
        coord[d] = tiff_read(a, fmt(a, "%s/%s", tile->path, v1_files[d]), &w, &h);
        need(w == tile->w && h == tile->h, "tile plane shapes disagree at %s", tile->path);
    }
    const char *qfile = fmt(a, "%s/%s", tile->path, v1_files[3]);
    Array q = exists(qfile) ? npy_read(a, qfile) : (Array){0};
    need(!q.data || (q.ndim == 2 && q.shape[0] == tile->h && q.shape[1] == tile->w),
        "tile quality shape disagrees at %s", tile->path);
    for (i64 r = 0; r < tile->h; r++) {
        for (i64 c = 0; c < tile->w; c++) {
            i64 at = r * tile->w + c;
            double value = q.data ? bench_number(q, at) : 1;
            if (!(coord[0][at] > 0 && coord[1][at] > 0 && coord[2][at] > 0 && value > 0)) {
                continue;
            }
            i64 to = (r + tile->row) * width + c + tile->col;
            for (int d = 0; d < 3; d++) {
                xyz[d][to] = coord[d][at];
            }
            quality[to] = isfinite(value) ? (u8)fmod(trunc(value), 256) : 0;
        }
    }
    a->used = saved;
}

static V1Source v1_reconstruct(Arena *a, const char *out, const V1Tile *tile, i64 tiles) {
    size_t saved = a->used;
    i64 w = 0, h = 0;
    for (i64 i = 0; i < tiles; i++) {
        i64 width = tile[i].col + tile[i].w, height = tile[i].row + tile[i].h;
        w = width > w ? width : w;
        h = height > h ? height : h;
    }
    i64 size = h * w;
    need(size > 0 && size <= INT64_MAX / 4, "invalid reconstructed tile extent");
    f32 *xyz[3];
    for (int d = 0; d < 3; d++) {
        xyz[d] = push(a, size * sizeof(f32));
        for (i64 i = 0; i < size; i++) {
            xyz[d][i] = -1;
        }
    }
    u8 *quality = pushz(a, size);
    for (i64 i = 0; i < tiles; i++) {
        v1_paint(a, tile + i, w, xyz, quality);
    }
    const char *target = fmt(a, "%s/source-%04lld.tifxyz", out, (long long)tile->source);
    mkdirs(target);
    for (int d = 0; d < 3; d++) {
        tiff_write(fmt(a, "%s/%s", target, v1_files[d]), xyz[d], w, h);
    }
    npy_write(fmt(a, "%s/%s", target, v1_files[3]),
        (Array){.descr = "|u1", .ndim = 2, .shape = {h, w}, .data = quality, .size = size});
    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "source", json_object_new_int64(tile->source));
    json_object_object_add(meta, "tiles", json_object_new_int64(tiles));
    json_object_object_add(
        meta, "scale", json_object_get(json_object_object_get(tile->meta, "scale")));
    json_object_object_add(
        meta, "level", json_object_get(json_object_object_get(tile->meta, "winding")));
    json_object_object_add(meta, "turns_spanned",
        json_object_get(json_object_object_get(tile->meta, "turns_spanned")));
    json_object_object_add(meta, "note",
        json_object_new_string("v1 level-route tiles reassembled at their block offsets; level is "
                               "the source's base level"));
    json_write(fmt(a, "%s/meta.json", target), meta);
    json_object_put(meta);
    i64 cells = 0;
    for (i64 i = 0; i < size; i++) {
        cells += xyz[2][i] > 0;
    }
    a->used = saved;
    return (V1Source){tile->source, tiles, cells};
}

static void v1_link(Arena *a, const char *out, const char *path) {
    size_t saved = a->used;
    const char *target = fmt(a, "%s/%s.tifxyz", out, bench_basename(path));
    mkdirs(target);
    for (int d = 0; d < 4; d++) {
        const char *from = fmt(a, "%s/%s", path, v1_files[d]);
        const char *to = fmt(a, "%s/%s", target, v1_files[d]);
        if (exists(from) && !exists(to)) {
            need(!symlink(from, to), "cannot link %s: %s", to, strerror(errno));
        }
    }
    write_file(fmt(a, "%s/meta.json", target), "{}", 2);
    a->used = saved;
}

int bench_v1_sources_command(int argc, char **argv) {
    if (argc && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("calibrate.bench.v1_sources PIECES OUT");
        return 0;
    }
    need(argc == 2, "calibrate.bench.v1_sources PIECES OUT");
    Arena a = arena(24ull << 30);
    char *pieces = resolved(argv[0]);
    const char *out = argv[1];
    need(pieces != NULL, "cannot resolve tiles folder %s", argv[0]);
    mkdirs(out);
    glob_t g = {0};
    glob(fmt(&a, "%s/piece-*", pieces), 0, NULL, &g);
    V1Tile *tiles = push(&a, g.gl_pathc * sizeof(*tiles));
    i64 n = 0, loose = 0;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        const char *path = g.gl_pathv[i];
        json_object *meta = json_read(fmt(&a, "%s/meta.json", path)), *source = NULL;
        need(meta != NULL && json_object_is_type(meta, json_type_object), "invalid metadata at %s",
            path);
        if (!json_object_object_get_ex(meta, "source", &source)) {
            v1_link(&a, out, path);
            json_object_put(meta);
            loose++;
        } else {
            need(json_object_is_type(source, json_type_int) ||
                    json_object_is_type(source, json_type_boolean),
                "tile source must be an integer");
            tiles[n] = (V1Tile){
                .path = path, .meta = meta, .source = json_object_get_int64(source), .order = i};
            v1_extent(&a, &tiles[n++]);
        }
    }
    qsort(tiles, n, sizeof(*tiles), v1_tile_order);
    i64 sources = 0;
    V1Source top[5] = {{0}};
    int best = 0;
    for (i64 first = 0, stop; first < n; first = stop) {
        for (stop = first + 1; stop < n && tiles[stop].source == tiles[first].source; stop++) {
        }
        V1Source row = v1_reconstruct(&a, out, tiles + first, stop - first);
        sources++;
        int rank = best;
        while (rank > 0 && row.cells > top[rank - 1].cells) {
            rank--;
        }
        if (rank < 5) {
            i64 end = best < 5 ? best++ : 4;
            memmove(top + rank + 1, top + rank, (end - rank) * sizeof(*top));
            top[rank] = row;
        }
        for (i64 i = first; i < stop; i++) {
            json_object_put(tiles[i].meta);
        }
    }
    globfree(&g);
    free(pieces);
    json_object *report = json_object_new_object(), *largest = json_object_new_array();
    json_object_object_add(report, "sources", json_object_new_int64(sources));
    json_object_object_add(report, "loose_tiles", json_object_new_int64(loose));
    json_object_object_add(report, "largest", largest);
    for (int i = 0; i < best; i++) {
        json_object *row = json_object_new_object();
        json_object_array_add(largest, row);
        json_object_object_add(row, "source", json_object_new_int64(top[i].source));
        json_object_object_add(row, "tiles", json_object_new_int64(top[i].tiles));
        json_object_object_add(row, "cells", json_object_new_int64(top[i].cells));
    }
    puts(board_json(report));
    json_object_put(report);
    munmap(a.base, a.size);
    return 0;
}
