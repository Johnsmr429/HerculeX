#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <json-c/json.h>
#include <math.h>
#include <openssl/evp.h>
#include <pwd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <unistd.h>

#if !defined(HX_ANATOMY_EXPORT) && !defined(HX_UNITY)
const char hx_usage[] = "A laid-out sheet as one tifxyz grid for VC3D.";
#endif
enum { STRIPE_BYTES = 8 * 1024 * 1024 };

typedef struct {
    const char *unit;
    char *folder;
    int64_t id, x, y;
    size_t order;
    double area;
    int flip_rows, flip_cols;
    uint32_t width, height;
} Piece;

typedef struct {
    uint32_t width, height, rows;
    double side, stride, um;
    float low[3], high[3];
    uint64_t known;
} Sheet;

#include "../../storage/data_roots.c"

static int mkdirs(char *path) {
    for (char *p = path + 1;; p++) {
        if (*p && *p != '/')
            continue;
        char end = *p;
        *p = 0;
        int ok = !mkdir(path, 0777) || errno == EEXIST;
        *p = end;
        if (!ok || !end)
            return ok;
    }
}

static int number(json_object *obj, const char *key, double *v) {
    json_object *value;
    if (!json_object_object_get_ex(obj, key, &value) ||
        (!json_object_is_type(value, json_type_int) &&
            !json_object_is_type(value, json_type_double))) {
        fprintf(stderr, "layout requires a number for %s\n", key);
        return 0;
    }
    *v = json_object_get_double(value);
    return isfinite(*v);
}

#include "../../sampling/column_sampling.c"

static int by_area(const void *one, const void *two) {
    const Piece *a = one, *b = two;
    if (a->area != b->area)
        return a->area < b->area ? -1 : 1;
    return (a->order > b->order) - (a->order < b->order);
}

static int layout(json_object *array, const char *base, Sheet *sheet, Piece *pieces, size_t count) {
    json_object *first = json_object_array_get_idx(array, 0);
    if (!number(first, "stride", &sheet->stride) || !number(first, "um", &sheet->um))
        return 0;
    sheet->side = sheet->stride * sheet->um;
    if (!(sheet->side > 0))
        return 0;
    double max_x = -INFINITY, max_y = -INFINITY;
    for (size_t i = 0; i < count; i++) {
        json_object *p = json_object_array_get_idx(array, i);
        Piece *piece = &pieces[i];
        double x, y, width, height;
        if (!number(p, "x_mm", &x) || !number(p, "y_mm", &y) || !number(p, "width_mm", &width) ||
            !number(p, "height_mm", &height) || !number(p, "area", &piece->area))
            return 0;
        max_x = fmax(max_x, x + width);
        max_y = fmax(max_y, y + height);
        double px = nearbyint(x / sheet->side), py = nearbyint(y / sheet->side);
        if (!(px >= INT32_MIN && px <= UINT32_MAX && py >= INT32_MIN && py <= UINT32_MAX))
            return 0;
        piece->x = (int64_t)px;
        piece->y = (int64_t)py;
        piece->order = i;
        json_object *unit, *id, *rows, *cols;
        if (!json_object_object_get_ex(p, "unit", &unit) ||
            !json_object_object_get_ex(p, "piece", &id) ||
            !json_object_object_get_ex(p, "flip_rows", &rows) ||
            !json_object_object_get_ex(p, "flip_cols", &cols))
            return 0;
        piece->unit = json_object_get_string(unit);
        piece->id = json_object_get_int64(id);
        piece->flip_rows = json_object_get_boolean(rows);
        piece->flip_cols = json_object_get_boolean(cols);
        if (asprintf(&piece->folder, "%s/pieces/%s/piece-%04lld", base, piece->unit,
                (long long)piece->id) < 0)
            return 0;
        TIFF *t = open_axis(piece->folder, 0);
        if (!t)
            return 0;
        int ok = TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &piece->width) &&
            TIFFGetField(t, TIFFTAG_IMAGELENGTH, &piece->height);
        TIFFClose(t);
        if (!ok || !piece->width || !piece->height)
            return 0;
    }
    double w = trunc(max_x / sheet->side) + 4, h = trunc(max_y / sheet->side) + 4;
    if (!(w > 0 && w <= UINT32_MAX && h > 0 && h <= UINT32_MAX))
        return 0;
    sheet->width = (uint32_t)w;
    sheet->height = (uint32_t)h;
    size_t cap = STRIPE_BYTES / (3 * sizeof(float) * sheet->width);
    sheet->rows = cap ? (uint32_t)fmin(cap, h) : 1;
    for (int i = 0; i < 3; i++) {
        sheet->low[i] = INFINITY;
        sheet->high[i] = -INFINITY;
    }
    qsort(pieces, count, sizeof(*pieces), by_area);
    return 1;
}

static int overlay(const Piece *p, const Sheet *s, uint32_t y0, uint32_t rows, float *plane) {
    int64_t lo = p->y + 1 > y0 ? p->y + 1 : y0;
    int64_t hi = p->y + p->height - 1;
    if (hi > (int64_t)y0 + rows)
        hi = (int64_t)y0 + rows;
    int64_t left = p->x + 1 > 0 ? p->x + 1 : 0;
    int64_t right = p->x + p->width - 1;
    if (right > s->width)
        right = s->width;
    if (hi <= lo || right <= left)
        return 1;
    size_t n = (size_t)p->height * p->width;
    if (n > SIZE_MAX / sizeof(float))
        return 0;
    float *xyz = malloc(n * sizeof(*xyz));
    uint8_t *valid = malloc(n);
    int ok = xyz && valid;
    for (int a = 0; ok && a < 3; a++) {
        ok = read_axis(
            p->folder, p->width, p->height, a, xyz, NULL, a == 0 ? valid : NULL, NULL, NULL);
        for (int64_t y = lo; ok && y < hi; y++) {
            size_t sy = p->flip_rows ? p->height - 1 - (y - p->y) : y - p->y;
            float *row = plane + ((size_t)a * rows + (size_t)(y - y0)) * s->width;
            for (int64_t x = left; x < right; x++) {
                size_t sx = p->flip_cols ? p->width - 1 - (x - p->x) : x - p->x;
                size_t i = sy * p->width + sx;
                if (valid[i] && valid[i - 1] && valid[i + 1] && valid[i - p->width] &&
                    valid[i + p->width])
                    row[x] = xyz[i];
            }
        }
    }
    free(valid);
    free(xyz);
    return ok;
}

static int file_uuid(const char *path, char uuid[37]) {
    const uint8_t url[] = {0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0,
        0x4f, 0xd4, 0x30, 0xc8};
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    int ok = ctx && EVP_DigestInit_ex(ctx, EVP_sha1(), NULL) &&
        EVP_DigestUpdate(ctx, url, sizeof(url)) && EVP_DigestUpdate(ctx, path, strlen(path)) &&
        EVP_DigestFinal_ex(ctx, hash, &n) && n >= 16;
    EVP_MD_CTX_free(ctx);
    if (!ok)
        return 0;
    hash[6] = (hash[6] & 15) | 0x50;
    hash[8] = (hash[8] & 63) | 0x80;
    char *c = uuid;
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            *c++ = '-';
        snprintf(c, 3, "%02x", hash[i]);
        c += 2;
    }
    return 1;
}

static json_object *vector(const float values[3]) {
    json_object *arr = json_object_new_array_ext(3);
    if (!arr)
        return NULL;
    for (int i = 0; i < 3; i++) {
        json_object *v = json_object_new_double(values[i]);
        if (!v || json_object_array_add(arr, v)) {
            json_object_put(v);
            json_object_put(arr);
            return NULL;
        }
    }
    return arr;
}

static int save_meta(
    const char *path, const char *folder, const Sheet *s, const char *sample, const char *scan) {
    char uuid[37];
    if (!file_uuid(folder, uuid))
        return 0;
    json_object *meta = json_object_new_object(), *bbox = json_object_new_array_ext(2);
    json_object *scale = json_object_new_array_ext(2);
    if (!meta || !bbox || !scale) {
        json_object_put(meta);
        json_object_put(bbox);
        json_object_put(scale);
        return 0;
    }
    json_object_array_add(bbox, vector(s->low));
    json_object_array_add(bbox, vector(s->high));
    json_object_array_add(scale, json_object_new_double(1 / s->stride));
    json_object_array_add(scale, json_object_new_double(1 / s->stride));
    json_object_object_add(meta, "sample", json_object_new_string(sample));
    json_object_object_add(meta, "scan", json_object_new_string(scan));
    json_object_object_add(meta, "voxel_um", json_object_new_double(s->um * 1000));
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "type", json_object_new_string("seg"));
    json_object_object_add(meta, "uuid", json_object_new_string(uuid));
    json_object_object_add(meta, "bbox", bbox);
    json_object_object_add(meta, "scale", scale);
    json_object_object_add(meta, "stride", json_object_new_double(s->stride));
    json_object_object_add(
        meta, "area_mm2", json_object_new_double(s->known * (s->side * s->side)));
    json_object_object_add(meta, "width_mm", json_object_new_double(s->width * s->side));
    json_object_object_add(meta, "height_mm", json_object_new_double(s->height * s->side));
    int ok = json_object_to_file_ext(path, meta, JSON_C_TO_STRING_PRETTY) == 0;
    json_object_put(meta);
    return ok;
}

static int set_tags(TIFF *t, const Sheet *s, const char *shape) {
    return TIFFSetField(t, TIFFTAG_IMAGEWIDTH, s->width) &&
        TIFFSetField(t, TIFFTAG_IMAGELENGTH, s->height) &&
        TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32) && TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1) &&
        TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP) &&
        TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
        TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
        TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, s->rows) &&
        TIFFSetField(t, TIFFTAG_XRESOLUTION, 1.0) && TIFFSetField(t, TIFFTAG_YRESOLUTION, 1.0) &&
        TIFFSetField(t, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE) &&
        TIFFSetField(t, TIFFTAG_IMAGEDESCRIPTION, shape);
}

static int install_files(const char *folder, const char *stage, char **tmp, char **dest) {
    int dir = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0)
        return 0;
    int lock;
    do {
        lock = flock(dir, LOCK_EX);
    } while (lock < 0 && errno == EINTR);
    int ok = lock == 0;
    char *saved[4] = {0};
    int exists[4] = {0};
    for (int i = 0; ok && i < 4; i++) {
        char prior[] = "old0";
        prior[3] += i;
        saved[i] = path_join(stage, prior);
        ok = saved[i] != NULL;
        if (ok) {
            exists[i] = link(dest[i], saved[i]) == 0;
            ok = exists[i] || errno == ENOENT;
        }
    }
    int installed = 0;
    while (ok && installed < 4) {
        ok = rename(tmp[installed], dest[installed]) == 0;
        if (ok)
            installed++;
    }
    if (!ok) {
        for (int i = installed - 1; i >= 0; i--) {
            if (exists[i]) {
                if (rename(saved[i], dest[i]) != 0) {
                    fprintf(stderr, "restore %s from %s\n", dest[i], saved[i]);
                    exists[i] = 0;
                }
            } else {
                unlink(dest[i]);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        if (exists[i])
            unlink(saved[i]);
        free(saved[i]);
    }
    close(dir);
    return ok;
}

static int compose(char *folder, const char *sample, const char *scan, Sheet *s,
    const Piece *pieces, size_t count) {
    if (!mkdirs(folder))
        return 0;
    char *stage = path_join(folder, ".export-XXXXXXXX");
    if (!stage || !mkdtemp(stage)) {
        free(stage);
        return 0;
    }
    const char *names[] = {"x.tif", "y.tif", "z.tif", "meta.json"};
    char *tmp[4] = {0}, *dest[4] = {0};
    TIFF *out[3] = {0};
    int ok = 1;
    char shape[96];
    snprintf(shape, sizeof(shape), "{\"shape\": [%u, %u]}", s->height, s->width);
    int large = (uint64_t)s->width * s->height > (UINT32_MAX - (32U << 20)) / 4;
    for (int a = 0; ok && a < 4; a++) {
        tmp[a] = path_join(stage, names[a]);
        dest[a] = path_join(folder, names[a]);
        ok = tmp[a] && dest[a];
        if (a < 3 && ok) {
            out[a] = TIFFOpen(tmp[a], large ? "w8l" : "wl");
            ok = out[a] && set_tags(out[a], s, shape);
        }
    }
    size_t capacity = (size_t)s->width * s->rows;
    float *plane = ok && capacity <= SIZE_MAX / (3 * sizeof(float))
        ? malloc(3 * capacity * sizeof(*plane))
        : NULL;
    ok = ok && plane;
    for (uint64_t y = 0; ok && y < s->height; y += s->rows) {
        uint32_t rows = (uint32_t)fmin(s->height - y, s->rows);
        size_t pixels = (size_t)s->width * rows;
        for (size_t k = 0; k < 3 * pixels; k++)
            plane[k] = -1;
        for (size_t i = 0; ok && i < count; i++)
            ok = overlay(&pieces[i], s, (uint32_t)y, rows, plane);
        for (size_t i = 0; ok && i < pixels; i++) {
            if (!(plane[i] >= 0))
                continue;
            s->known++;
            for (int a = 0; a < 3; a++) {
                float v = plane[(size_t)a * pixels + i];
                if (isnan(v)) {
                    s->low[a] = v;
                    s->high[a] = v;
                } else if (!isnan(s->low[a])) {
                    s->low[a] = fminf(s->low[a], v);
                    s->high[a] = fmaxf(s->high[a], v);
                }
            }
        }
        for (int a = 0; ok && a < 3; a++) {
            tmsize_t bytes = (tmsize_t)pixels * (tmsize_t)sizeof(float);
            ok = TIFFWriteEncodedStrip(
                     out[a], (uint32_t)(y / s->rows), plane + (size_t)a * pixels, bytes) == bytes;
        }
    }
    free(plane);
    for (int a = 0; a < 3; a++) {
        if (out[a]) {
            if (!TIFFFlush(out[a]))
                ok = 0;
            TIFFClose(out[a]);
        }
    }
    if (ok && !s->known) {
        fputs("no valid sheet cells in the export\n", stderr);
        ok = 0;
    }
    if (ok)
        ok = save_meta(tmp[3], folder, s, sample, scan);
    if (ok)
        ok = install_files(folder, stage, tmp, dest);
    for (int a = 0; a < 4; a++) {
        if (tmp[a])
            unlink(tmp[a]);
        free(tmp[a]);
        free(dest[a]);
    }
    rmdir(stage);
    free(stage);
    return ok;
}

static int export_sheet(const char *sample, const char *scan, const char *name, Sheet *sheet) {
    char *candidate = local_root(outputs_root);
    char *root = candidate ? realpath(candidate, NULL) : NULL, *base = NULL, *out = NULL;
    free(candidate);
    int ok = root && asprintf(&base, "%s/unwrapping/level/%s/%s", root, sample, scan) >= 0 &&
        asprintf(&out, "%s/website/sheets/%s/tifxyz/%s-%s-sheet", root, name, sample, scan) >= 0;
    char *path = ok ? path_join(base, "sheet/layout.json") : NULL;
    json_object *layout_doc = path ? json_object_from_file(path) : NULL;
    json_object *array = NULL;
    ok = layout_doc && json_object_object_get_ex(layout_doc, "pieces", &array) &&
        json_object_is_type(array, json_type_array);
    size_t count = ok ? json_object_array_length(array) : 0;
    Piece *pieces = count ? calloc(count, sizeof(*pieces)) : NULL;
    ok = pieces && layout(array, base, sheet, pieces, count) &&
        compose(out, sample, scan, sheet, pieces, count);
    if (pieces) {
        for (size_t i = 0; i < count; i++)
            free(pieces[i].folder);
    }
    free(pieces);
    json_object_put(layout_doc);
    free(path);
    free(out);
    free(base);
    free(root);
    return ok;
}

int hx_tifxyz_main(int argc, char **argv) {
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts("Usage: ./run.sh level.layout.tifxyz_export SCROLL SCAN NAME");
        return 0;
    }
#ifdef HX_ANATOMY_EXPORT
    if (!own_outputs())
        return 1;
#endif
    if (argc < 4) {
        fprintf(stderr, "usage: %s SCROLL SCAN NAME\n", argv[0]);
        return 1;
    }
    Sheet sheet = {0};
    if (!export_sheet(argv[1], argv[2], argv[3], &sheet)) {
        fprintf(stderr, "failed to export the sheet for %s/%s\n", argv[1], argv[2]);
        return 1;
    }
    printf("%s tifxyz (3, %u, %u) %.0f cm2\n", argv[1], sheet.height, sheet.width,
        sheet.known * (sheet.side * sheet.side) / 100);
    return 0;
}

#ifndef HX_UNITY
int main(int argc, char **argv) { return hx_tifxyz_main(argc, argv); }
#endif
