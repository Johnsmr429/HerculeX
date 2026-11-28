#define _GNU_SOURCE
#include <setjmp.h>
#include <stdio.h>
#include <jpeglib.h>
#include <png.h>
#include <limits.h>

#include "../storage/data_roots.c"
#include "../storage/tile_store.c"
#include "sheet_page.h"
#include "tile_decode.c"

#ifndef HX_UNITY
const char hx_usage[] = "Compose or verify a sheet's sparse tile pyramid.";
#endif

int hx_check_review(int argc, char **argv);

enum { TILE = 512, STRIP = 8, TILE_MARGIN = 2, LEVELS = 12 };

typedef struct {
    uint8_t *pixels;
    int w, h;
} Image;

typedef struct {
    json_object *record;
    const char *unit;
    int piece, mosaic, flip_rows, flip_cols, retired, sample, overlap[4];
    double x, y, width, height, area, mm, stride;
    Image image, mask;
} Piece;

typedef struct {
    int left, top, right, bottom;
} Box;

typedef struct {
    Piece *piece;
    Box bounds, interior;
    int x, y;
} Draw;

typedef uint64_t Histogram[256];

typedef struct {
    TilePack *pack;
    PngWriter *png;
    uint8_t *encoded, *half;
    size_t bound;
} Painter;

static int painter_open(Painter *p, TilePack *pack) {
    *p = (Painter){.pack = pack};
    size_t raw = (TILE + 1) * TILE;
    p->png = png_writer(raw);
    if (p->png) {
        p->bound = libdeflate_zlib_compress_bound(p->png->compressor, raw);
        p->encoded = malloc(p->bound + 57);
        p->half = malloc(TILE * TILE);
    }
    return p->encoded && p->half;
}

static void painter_close(Painter *p) {
    free(p->half);
    free(p->encoded);
}

int hx_site_folders(const char *path) {
    char *copy = strdup(path);
    if (!copy) {
        return 0;
    }
    int ok = 1;
    for (char *p = copy + 1; ok && *p; p++) {
        if (*p == '/') {
            *p = 0;
            ok = !mkdir(copy, 0777) || errno == EEXIST;
            *p = '/';
        }
    }
    ok = ok && (!mkdir(path, 0777) || errno == EEXIST);
    free(copy);
    return ok;
}

static json_object *document_at(const char *folder, const char *name) {
    char *path = path_join(folder, name);
    json_object *doc = path ? json_object_from_file(path) : NULL;
    free(path);
    return doc;
}

int hx_write_sheet_file(const char *path, void *data, size_t size) {
    char *part = NULL;
    struct stat prior;
    int kept = !lstat(path, &prior) && S_ISREG(prior.st_mode);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int fd;
    for (uint64_t stamp = (uint64_t)now.tv_sec * 1000000000 + now.tv_nsec;; stamp++) {
        if (asprintf(&part, "%s.%ld.%" PRIu64 ".part", path, (long)getpid(), stamp) < 0) {
            return 0;
        }
        fd = open(part, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (fd >= 0 || errno != EEXIST) {
            break;
        }
        free(part);
    }
    int ok = fd >= 0 && (!kept || !fchmod(fd, prior.st_mode & 0777)) &&
        transfer_at(fd, data, size, 0, 1);
    if (fd >= 0) {
        ok = !close(fd) && ok && !rename(part, path);
        if (!ok) {
            unlink(part);
        }
    }
    free(part);
    return ok;
}

static int write_document(const char *folder, const char *name, json_object *doc) {
    char *path = path_join(folder, name);
    const char *text = doc ? json_object_to_json_string_ext(doc, JSON_C_TO_STRING_PLAIN) : NULL;
    int ok = path && text && hx_write_sheet_file(path, (void *)text, strlen(text));
    free(path);
    return ok;
}

static int valid_path(const char *path, int nested) {
    if (!path || !*path) {
        return 0;
    }
    while (*path) {
        size_t part = strcspn(path, "/");
        if (!part || (part == 1 && path[0] == '.') || (part == 2 && !strncmp(path, "..", 2))) {
            return 0;
        }
        path += part;
        if (*path) {
            if (!nested || !*++path) {
                return 0;
            }
        }
    }
    return 1;
}

static double number(json_object *record, const char *key) {
    return json_object_get_double(json_object_object_get(record, key));
}

static int yes(json_object *record, const char *key) {
    return json_object_get_boolean(json_object_object_get(record, key));
}

static const char *text(json_object *record, const char *key) {
    json_object *value = json_object_object_get(record, key);
    return json_object_is_type(value, json_type_string) ? json_object_get_string(value) : NULL;
}

static int selected_unit(const char *units, const char *unit) {
    if (!units) {
        return 1;
    }
    size_t length = strlen(unit);
    while (*units) {
        size_t part = strcspn(units, ",");
        if (part == length && !memcmp(units, unit, length)) {
            return 1;
        }
        units += part;
        if (*units) {
            units++;
        }
    }
    return 0;
}

static char *piece_image(const char *folder, const Piece *piece, const char *layer) {
    char *path = NULL;
    int scan = !strcmp(layer, "sheet") || !strcmp(layer, "taught-scan");
    if (asprintf(&path, "%s/%s/piece-%04d%s%s.png", folder, piece->unit, piece->piece,
            scan ? "" : "-", scan ? "" : layer) < 0) {
        return NULL;
    }
    return path;
}

static int ink_layer(const char *layer) {
    return !strcmp(layer, "ink") || !strcmp(layer, "taught");
}

static int far_side(const char *folder, const Piece *piece, const char *layer) {
    char *name = NULL;
    if (asprintf(&name, "%s/piece-%04d-%s.json", piece->unit, piece->piece, layer) < 0) {
        return 0;
    }
    json_object *record = document_at(folder, name);
    const char *side = text(record, "written");
    int far = side && !strcmp(side, "far");
    json_object_put(record);
    free(name);
    return far;
}

typedef struct {
    double left, right, area, far;
} Mosaic;

int hx_mosaic_slot(json_object *groups, json_object *piece) {
    json_object *group = json_object_object_get(piece, "group");
    const char *label = json_object_to_json_string_ext(group, JSON_C_TO_STRING_PLAIN);
    if (json_object_is_type(group, json_type_boolean)) {
        label = json_object_get_boolean(group) ? "1" : "0";
    }
    double moved = nearbyint(number(piece, "moved_from_joint_mm"));
    char *offset = NULL;
    if (!isfinite(moved) || (moved != 0 && asprintf(&offset, "%s/%.0f", label, moved) < 0)) {
        return 0;
    }
    const char *key = offset ? offset : label;
    json_object *existing = json_object_object_get(groups, key);
    int slot = existing ? json_object_get_int(existing) : json_object_object_length(groups) + 1;
    if (!existing && json_object_object_add(groups, key, json_object_new_int(slot))) {
        slot = 0;
    }
    free(offset);
    return slot;
}

static int load_layout(json_object *rows, const char *folder, const char *side, Piece *pieces,
    size_t count, int largest) {
    json_object *groups = json_object_new_object();
    Mosaic *mosaics = calloc(count ? count : 1, sizeof(*mosaics));
    int *far = calloc(count ? count : 1, sizeof(*far));
    int ok = groups && mosaics && far;
    size_t used = 0;
    for (size_t i = 0; ok && i < count; i++) {
        json_object *record = json_object_array_get_idx(rows, i);
        Piece *p = pieces + i;
        *p = (Piece){.record = record,
            .unit = text(record, "unit"),
            .piece = json_object_get_int(json_object_object_get(record, "piece")),
            .x = number(record, "x_mm"),
            .y = number(record, "y_mm"),
            .width = number(record, "width_mm"),
            .height = number(record, "height_mm"),
            .area = number(record, "area"),
            .mm = number(record, "um"),
            .stride = number(record, "stride"),
            .flip_rows = yes(record, "flip_rows"),
            .flip_cols = yes(record, "flip_cols")};
        if (!valid_path(p->unit, 0) || p->piece < 0 || !(p->width >= 0 && p->height >= 0) ||
            !(p->mm > 0 && p->stride > 0) ||
            !isfinite(p->x + p->y + p->width + p->height + p->area + p->mm + p->stride)) {
            ok = 0;
            break;
        }
        json_object *overlap = json_object_object_get(record, "overlap");
        if (json_object_is_type(overlap, json_type_array)) {
            for (size_t j = 0; j < 4; j++) {
                p->overlap[j] = json_object_get_int(json_object_array_get_idx(overlap, j));
            }
        }
        p->mosaic = hx_mosaic_slot(groups, record);
        if (!p->mosaic) {
            ok = 0;
            break;
        }
        if ((size_t)p->mosaic > used) {
            mosaics[used++] = (Mosaic){.left = p->x, .right = p->x + p->width};
        }
        Mosaic *m = mosaics + p->mosaic - 1;
        m->left = fmin(m->left, p->x);
        m->right = fmax(m->right, p->x + p->width);
        m->area += p->area;
        far[i] = far_side(folder, p, side);
        if (far[i]) {
            m->far += p->area;
        }
    }
    int winner = largest && used ? 1 : 0;
    for (size_t i = 1; winner && i < used; i++) {
        if (mosaics[i].area > mosaics[winner - 1].area) {
            winner = (int)i + 1;
        }
    }
    for (size_t i = 0; ok && i < count; i++) {
        Piece *p = pieces + i;
        Mosaic *m = mosaics + p->mosaic - 1;
        int turn = m->far * 2 > m->area;
        if (turn) {
            p->x = m->left + m->right - p->x - p->width;
            json_object_object_add(p->record, "x_mm", json_object_new_double(p->x));
        }
        if (turn || far[i]) {
            p->flip_cols = !p->flip_cols;
            json_object_object_add(p->record, "flip_cols", json_object_new_boolean(p->flip_cols));
        }
    }
    if (winner) {
        for (size_t i = 0; ok && i < count; i++) {
            if (pieces[i].mosaic != winner) {
                pieces[i].mosaic = 0;
            }
        }
    }
    json_object_put(groups);
    free(mosaics);
    free(far);
    return ok;
}

int hx_sheet_orient(json_object *rows, const char *folder, const char *side) {
    if (!json_object_is_type(rows, json_type_array)) {
        return 0;
    }
    size_t count = json_object_array_length(rows);
    Piece *pieces = calloc(count ? count : 1, sizeof(*pieces));
    int ok = pieces && load_layout(rows, folder, side, pieces, count, 0);
    free(pieces);
    return ok;
}

static void free_images(Piece *p) {
    if (p->mask.pixels != p->image.pixels) {
        free(p->mask.pixels);
    }
    free(p->image.pixels);
    p->image = (Image){0};
    p->mask = (Image){0};
}

static Image finish_png(png_image *read) {
    read->format = PNG_FORMAT_GRAY;
    Image image = {0};
    if (read->width <= INT_MAX && read->height <= INT_MAX && read->width &&
        read->height <= SIZE_MAX / read->width) {
        image =
            (Image){.pixels = malloc(PNG_IMAGE_SIZE(*read)), .w = read->width, .h = read->height};
        png_color black = {0};
        if (image.pixels && !png_image_finish_read(read, &black, image.pixels, 0, NULL)) {
            free(image.pixels);
            image = (Image){0};
        }
    }
    png_image_free(read);
    return image;
}

static Image read_png(const char *path) {
    png_image read = {.version = PNG_IMAGE_VERSION};
    return png_image_begin_read_from_file(&read, path) ? finish_png(&read) : (Image){0};
}

uint8_t *hx_sheet_png(const char *path, int *w, int *h) {
    Image image = read_png(path);
    *w = image.w;
    *h = image.h;
    return image.pixels;
}

static int load_images(Piece *p, const char *folder, const char *layer) {
    if (p->image.pixels) {
        return 1;
    }
    int separate = ink_layer(layer);
    char *path = piece_image(folder, p, layer),
         *mask = separate ? piece_image(folder, p, "sheet") : NULL;
    int found = path && (!separate || mask) ? 0 : -1;
    if (found == 0 && !access(path, F_OK) && (!mask || !access(mask, F_OK))) {
        p->image = read_png(path);
        p->mask = mask ? read_png(mask) : p->image;
        found =
            p->image.pixels && p->mask.pixels && p->image.w == p->mask.w && p->image.h == p->mask.h
            ? 1
            : -1;
        if (found < 0) {
            fprintf(stderr, "HerculeX: cannot read piece image or mask %s\n", path);
            free_images(p);
        }
    }
    free(mask);
    free(path);
    return found;
}

static double sinc(double x) {
    if (x == 0) {
        return 1;
    }
    double angle = x * 0x1.921fb54442d18p+1;
    return sin(angle) / angle;
}

static double bicubic_weight(double x) {
    x = fabs(x);
    if (x < 1) {
        return ((1.5 * x - 2.5) * x) * x + 1;
    }
    if (x < 2) {
        return (((-.5 * x + 2.5) * x - 4) * x) + 2;
    }
    return 0;
}

static int resize_axis(const Image *in, Image *out, int horizontal, double in_end, int cubic) {
    int from = horizontal ? in->w : in->h;
    int to = horizontal ? out->w : out->h;
    int across = horizontal ? out->h : out->w;
    double scale = in_end / to, stretch = fmax(1, scale),
           reach = (cubic == 2     ? 1
                           : cubic ? 2
                                   : 3) *
        stretch;
    int limit = reach >= from ? from : (int)(2 * ceil(reach) + 1);
    double *weights = malloc((size_t)limit * sizeof(*weights));
    int *coeff = malloc((size_t)limit * sizeof(*coeff));
    if (!weights || !coeff) {
        free(coeff);
        free(weights);
        return 0;
    }
    for (int i = 0; i < to; i++) {
        double centre = (i + .5) * scale, total = 0;
        int lo = (int)fmax(0, centre - reach + .5);
        int hi = (int)fmin(from, centre + reach + .5);
        for (int x = lo; x < hi; x++) {
            double at = (x - centre + .5) / stretch;
            weights[x - lo] = cubic == 2 ? fmax(0, 1 - fabs(at))
                : cubic                  ? bicubic_weight(at)
                : at >= -3 && at < 3     ? sinc(at) * sinc(at / 3)
                                         : 0.;
            total += weights[x - lo];
        }
        for (int x = lo; x < hi; x++) {
            coeff[x - lo] = (int)lround(weights[x - lo] / total * (1 << FILTER_BITS));
        }
        for (int k = 0; k < across; k++) {
            int sum = FILTER_ROUND;
            for (int x = lo; x < hi; x++) {
                size_t cell = horizontal ? (size_t)k * in->w + x : (size_t)x * in->w + k;
                sum += in->pixels[cell] * coeff[x - lo];
            }
            out->pixels[horizontal ? (size_t)k * out->w + i : (size_t)i * out->w + k] =
                filtered_pixel(sum);
        }
    }
    free(coeff);
    free(weights);
    return 1;
}

static int scaled_image(Image in, Image *out, int w, int h, int nearest) {
    if ((size_t)h > SIZE_MAX / w) {
        return 0;
    }
    *out = (Image){.pixels = malloc((size_t)w * h), .w = w, .h = h};
    if (!out->pixels) {
        return 0;
    }
    if (nearest) {
        double sx = (double)in.w / w, sy = (double)in.h / h;
        double ay = .5 * sy;
        for (int y = 0; y < h; y++, ay += sy) {
            double ax = .5 * sx;
            size_t row = (size_t)(int)ay * in.w;
            for (int x = 0; x < w; x++, ax += sx) {
                out->pixels[(size_t)y * w + x] = in.pixels[row + (int)ax];
            }
        }
        return 1;
    }
    Image across = {.pixels = malloc((size_t)w * in.h), .w = w, .h = in.h};
    int ok = across.pixels && resize_axis(&in, &across, 1, in.w, 0) &&
        resize_axis(&across, out, 0, in.h, 0);
    free(across.pixels);
    return ok;
}

static int viewer_resize(const uint8_t *input, int w, int h, double x_end, double y_end,
    uint8_t *output, int cols, int rows, int cubic) {
    Image in = {.pixels = (uint8_t *)input, .w = w, .h = h};
    Image out = {.pixels = output, .w = cols, .h = rows};
    if (w == cols && x_end == w) {
        return resize_axis(&in, &out, 0, y_end, cubic);
    }
    if (h == rows && y_end == h) {
        return resize_axis(&in, &out, 1, x_end, cubic);
    }
    int vertical = h > rows && (int64_t)h - rows > 2 * ((int64_t)w - cols);
    Image pass = {.w = vertical ? w : cols, .h = vertical ? rows : h};
    pass.pixels = malloc((size_t)pass.w * pass.h);
    int ok = pass.pixels && resize_axis(&in, &pass, !vertical, vertical ? y_end : x_end, cubic) &&
        resize_axis(&pass, &out, vertical, vertical ? x_end : y_end, cubic);
    free(pass.pixels);
    return ok;
}

int hx_site_resize_grey_across_first(
    const uint8_t *input, int w, int h, uint8_t *output, int cols, int rows) {
    Image in = {.pixels = (uint8_t *)input, .w = w, .h = h};
    Image along = {.pixels = malloc((size_t)cols * h), .w = cols, .h = h};
    Image out = {.pixels = output, .w = cols, .h = rows};
    int ok =
        along.pixels && resize_axis(&in, &along, 1, w, 0) && resize_axis(&along, &out, 0, h, 0);
    free(along.pixels);
    return ok;
}

int hx_site_resize_grey(
    const uint8_t *in, int w, int h, uint8_t *out, int cols, int rows, int linear) {
    return viewer_resize(in, w, h, w, h, out, cols, rows, linear ? 2 : 0);
}

int hx_site_resize_cubic(const uint8_t *input, int w, int h, double x_end, double y_end,
    uint8_t *output, int cols, int rows) {
    return viewer_resize(input, w, h, (float)x_end, (float)y_end, output, cols, rows, 1);
}

int hx_site_resize_rgb(const uint8_t *input, int w, int h, uint8_t *output, int width, int height) {
    size_t old = (size_t)w * h, resized = (size_t)width * height;
    uint8_t *plane = malloc(old), *pass = malloc(resized);
    int ok = plane && pass;
    for (int c = 0; ok && c < 3; c++) {
        for (size_t i = 0; i < old; i++) {
            plane[i] = input[i * 3 + c];
        }
        ok = viewer_resize(plane, w, h, w, h, pass, width, height, 0);
        if (ok) {
            for (size_t i = 0; i < resized; i++) {
                output[i * 3 + c] = pass[i];
            }
        }
    }
    free(pass);
    free(plane);
    return ok;
}

static void flip(Image image, int rows, int cols) {
    if (rows) {
        for (int y = 0; y < image.h / 2; y++) {
            uint8_t *a = image.pixels + (size_t)y * image.w;
            uint8_t *b = image.pixels + (size_t)(image.h - 1 - y) * image.w;
            for (int x = 0; x < image.w; x++) {
                uint8_t value = a[x];
                a[x] = b[x];
                b[x] = value;
            }
        }
    }
    if (cols) {
        for (int y = 0; y < image.h; y++) {
            uint8_t *row = image.pixels + (size_t)y * image.w;
            for (int x = 0; x < image.w / 2; x++) {
                uint8_t value = row[x];
                row[x] = row[image.w - 1 - x];
                row[image.w - 1 - x] = value;
            }
        }
    }
}

static void sample_piece(Piece *p, Histogram histogram) {
    int until = 1, dy = p->flip_rows ? -1 : 1, dx = p->flip_cols ? -1 : 1;
    int y = p->flip_rows ? p->image.h - 1 : 0;
    int x0 = p->flip_cols ? p->image.w - 1 : 0;
    for (int row = 0; row < p->image.h; row++, y += dy) {
        const uint8_t *pixels = p->image.pixels + (size_t)y * p->image.w;
        int x = x0;
        for (int col = 0; col < p->image.w; col++, x += dx) {
            uint8_t value = pixels[x];
            if (value && --until == 0) {
                until = 97;
                histogram[value]++;
            }
        }
    }
    p->sample = 0;
}

static int prepare(
    Piece *p, const char *folder, const char *layer, double mm, Histogram histogram) {
    if (p->image.pixels) {
        return 1;
    }
    int status = load_images(p, folder, layer);
    if (status <= 0) {
        return p->sample ? -1 : status;
    }
    if (p->sample) {
        sample_piece(p, histogram);
    }
    int separate = p->image.pixels != p->mask.pixels;
    flip(p->image, p->flip_rows, p->flip_cols);
    if (separate) {
        flip(p->mask, p->flip_rows, p->flip_cols);
    }
    double ratio = p->mm / mm;
    if (fabs(ratio - 1) > .01) {
        double width = fmax(1, nearbyint(p->image.w * ratio));
        double height = fmax(1, nearbyint(p->image.h * ratio));
        Image image = {0}, mask = {0};
        int ok = width <= INT_MAX && height <= INT_MAX &&
            scaled_image(p->image, &image, (int)width, (int)height, 0) &&
            scaled_image(p->mask, &mask, (int)width, (int)height, 1);
        free_images(p);
        if (!ok) {
            free(image.pixels);
            free(mask.pixels);
            return -1;
        }
        p->image = image;
        p->mask = mask;
    }
    return 1;
}

static int piece_order(const void *a, const void *b) {
    const Piece *left = *(Piece *const *)a, *right = *(Piece *const *)b;
    if (left->area != right->area) {
        return left->area > right->area ? -1 : 1;
    }
    return left < right ? -1 : left > right;
}

static double percentile(const uint64_t *histogram, uint64_t count, double part) {
    double at = (count - 1) * part;
    uint64_t lower = (uint64_t)at, upper = (uint64_t)ceil(at), seen = 0;
    int low = -1;
    for (int i = 0; i < 256; i++) {
        seen += histogram[i];
        if (seen > lower && low < 0) {
            low = i;
        }
        if (seen > upper) {
            double fraction = at - lower;
            return fraction < .5 ? low + (i - low) * fraction : i - (i - low) * (1 - fraction);
        }
    }
    return 255;
}

static void shown_range(const Histogram histogram, double *low, double *high) {
    *low = 0;
    *high = 255;
    uint64_t count_pixels = 0;
    for (int i = 0; i < 256; i++) {
        count_pixels += histogram[i];
    }
    if (count_pixels) {
        double a = percentile(histogram, count_pixels, .01);
        double b = percentile(histogram, count_pixels, .99);
        if (a < b) {
            *low = a;
            *high = b;
        }
    }
}

typedef struct {
    struct jpeg_error_mgr handler;
    jmp_buf on_error;
} JpegError;

static void jpeg_failed(j_common_ptr codec) {
    JpegError *error = (JpegError *)codec->err;
    longjmp(error->on_error, 1);
}

typedef struct {
    struct jpeg_compress_struct codec;
    unsigned char *data;
    unsigned long length;
} JpegWriter;

static int encode_jpeg(
    JpegWriter *writer, uint8_t *pixels, int w, int h, int quality, int channels) {
    struct jpeg_compress_struct *codec = &writer->codec;
    JpegError error;
    codec->err = jpeg_std_error(&error.handler);
    error.handler.error_exit = jpeg_failed;
    if (setjmp(error.on_error)) {
        return 0;
    }
    jpeg_create_compress(codec);
    jpeg_mem_dest(codec, &writer->data, &writer->length);
    codec->image_width = w;
    codec->image_height = h;
    codec->input_components = channels;
    codec->in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(codec);
    jpeg_set_quality(codec, quality, TRUE);
    jpeg_start_compress(codec, TRUE);
    while (codec->next_scanline < (unsigned)h) {
        JSAMPROW row = pixels + (size_t)codec->next_scanline * w * channels;
        jpeg_write_scanlines(codec, &row, 1);
    }
    jpeg_finish_compress(codec);
    return 1;
}

int hx_site_save_jpeg_channels(
    const char *path, uint8_t *pixels, int w, int h, int quality, int channels) {
    JpegWriter writer = {0};
    int ok = encode_jpeg(&writer, pixels, w, h, quality, channels);
    writer.codec.err = NULL;
    jpeg_destroy_compress(&writer.codec);
    ok = ok && hx_write_sheet_file(path, writer.data, writer.length);
    free(writer.data);
    return ok;
}

int hx_site_save_jpeg(const char *path, uint8_t *pixels, int w, int h, int quality) {
    return hx_site_save_jpeg_channels(path, pixels, w, h, quality, 3);
}

static int put_jpeg(Painter *p, TileLayer *layer, size_t cell) {
    JpegWriter writer = {0};
    int ok = encode_jpeg(&writer, p->half, TILE, TILE, 90, 1);
    writer.codec.err = NULL;
    jpeg_destroy_compress(&writer.codec);
    ok = ok && tile_blob(p->pack, layer, cell, writer.data, writer.length);
    free(writer.data);
    return ok;
}

static int decode_jpeg(
    struct jpeg_decompress_struct *codec, const uint8_t *input, uint32_t length, uint8_t *dest) {
    JpegError error;
    codec->err = jpeg_std_error(&error.handler);
    error.handler.error_exit = jpeg_failed;
    if (setjmp(error.on_error)) {
        return 0;
    }
    jpeg_create_decompress(codec);
    jpeg_mem_src(codec, input, length);
    jpeg_read_header(codec, TRUE);
    if (codec->image_width != TILE || codec->image_height != TILE) {
        return 0;
    }
    codec->out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(codec);
    for (int y = 0; y < TILE; y++) {
        JSAMPROW row = dest + (size_t)y * TILE;
        if (jpeg_read_scanlines(codec, &row, 1) != 1) {
            return 0;
        }
    }
    return jpeg_finish_decompress(codec);
}

static int read_jpeg(const uint8_t *input, uint32_t length, uint8_t *dest) {
    struct jpeg_decompress_struct codec = {0};
    int ok = decode_jpeg(&codec, input, length, dest);
    codec.err = NULL;
    jpeg_destroy_decompress(&codec);
    return ok;
}

static Box clipped(int x, int y, int w, int h, int width, int height) {
    return (Box){.left = x > 0 ? x : 0,
        .top = y > 0 ? y : 0,
        .right = x + w < width ? x + w : width,
        .bottom = y + h < height ? y + h : height};
}

static int nonempty(Box box) { return box.right > box.left && box.bottom > box.top; }

#include "sheet_card.inc"

static int slice_at(double pos, int size) {
    if (pos < 0) {
        pos += size;
    }
    return (int)fmax(0, fmin(size, pos));
}

static int draw_of(Piece *p, double left, double mm, int w, int h, Draw *draw) {
    double step = p->stride * p->mm / mm, margin = nearbyint(TILE_MARGIN * step);
    double x = nearbyint((p->x - left) / mm) - margin, y = nearbyint(p->y / mm) - margin;
    if (!(fabs(x) < INT32_MAX / 2 && fabs(y) < INT32_MAX / 2 && p->image.w < INT32_MAX / 2 &&
            p->image.h < INT32_MAX / 2)) {
        return -1;
    }
    Box bounds = clipped(x, y, p->image.w, p->image.h, w, h);
    if (!nonempty(bounds)) {
        return 0;
    }
    int top = p->overlap[p->flip_rows ? 1 : 0], bottom = p->overlap[p->flip_rows ? 0 : 1];
    int left_cut = p->overlap[p->flip_cols ? 3 : 2], right_cut = p->overlap[p->flip_cols ? 2 : 3];
    int a = slice_at(margin + nearbyint(top * step), p->image.h);
    int b = slice_at(p->image.h - margin - nearbyint(bottom * step), p->image.h);
    int c = slice_at(margin + nearbyint(left_cut * step), p->image.w);
    int d = slice_at(p->image.w - margin - nearbyint(right_cut * step), p->image.w);
    *draw = (Draw){.piece = p,
        .bounds = bounds,
        .interior = clipped(x + c, y + a, d - c, b - a, w, h),
        .x = x,
        .y = y};
    return 1;
}

static int occupied_tile(const uint8_t *seen, int stride) {
    for (int y = 0; y < TILE; y++) {
        if (memchr(seen + (size_t)y * stride, 1, TILE)) {
            return 1;
        }
    }
    return 0;
}

static int paint_strip(
    Painter *p, TileLayer *layer, Draw *draw, size_t count, int col, int columns, double mm) {
    int first = INT_MAX, last = 0, multiple = 0;
    for (size_t i = 0; i < count; i++) {
        int top = draw[i].bounds.top / TILE, bottom = (draw[i].bounds.bottom + TILE - 1) / TILE;
        first = top < first ? top : first;
        last = bottom > last ? bottom : last;
        multiple |= draw[i].piece->mosaic != draw[0].piece->mosaic;
    }
    if (first >= last) {
        return 1;
    }
    int h = (last - first) * TILE, w = STRIP * TILE;
    size_t pixels = (size_t)h * w;
    if (pixels > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    uint8_t *canvas = calloc(pixels, 2), *covered = canvas ? canvas + pixels : NULL;
    int32_t *owner = multiple ? calloc(pixels, sizeof(*owner)) : NULL;
    int good = canvas && (!multiple || owner);
    for (int margins = 0; good && margins < 2; margins++) {
        for (size_t i = 0; i < count; i++) {
            Draw *item = draw + i;
            Box box = margins ? item->bounds : item->interior;
            if (!nonempty(box)) {
                continue;
            }
            Piece *piece = item->piece;
            size_t cell = (size_t)(box.top - item->y) * piece->image.w + box.left - item->x;
            size_t at = (size_t)(box.top - first * TILE) * w + box.left;
            int64_t layout[] = {
                box.bottom - box.top, box.right - box.left, piece->image.w, 1, piece->mask.w, 1};
            hx_paint_rows(canvas + at, covered + at, owner ? owner + at : NULL, w,
                piece->image.pixels + cell, piece->mask.pixels + cell, layout, piece->mosaic);
        }
    }
    if (good && owner) {
        int32_t expires[STRIP * TILE] = {0};
        int width = (int)fmax(3, fmin(INT32_MAX / 2, nearbyint(.25 / mm)));
        hx_part_mosaics(covered, canvas, owner, h, w, width, expires);
    }
    free(owner);
    int end = col + STRIP < columns ? col + STRIP : columns;
    for (int x = col; good && x < end; x++) {
        for (int y = first; good && y < last; y++) {
            size_t cell = tile_cell(layer, 0, x, y);
            size_t at = (size_t)(y - first) * TILE * w + (x - col) * TILE;
            if (cell == SIZE_MAX || !occupied_tile(covered + at, w)) {
                continue;
            }
            size_t length = encode_grey(p->png, canvas + at, TILE, TILE, w, p->encoded, p->bound);
            good = tile_blob(p->pack, layer, cell, p->encoded, length);
        }
    }
    const int64_t stride[] = {w, w, w, w};
    for (int x = col / 2; good && layer->levels > 1 && x < (end + 1) / 2; x++) {
        for (int y = first / 2; good && y < (last + 1) / 2; y++) {
            const uint8_t *children[4] = {0};
            int any = 0;
            for (int k = 0; k < 4; k++) {
                int a = 2 * x + k % 2, b = 2 * y + k / 2;
                size_t cell = tile_cell(layer, 0, a, b);
                if (cell != SIZE_MAX && layer->sizes[cell] && b >= first && b < last) {
                    children[k] = canvas + (size_t)(b - first) * TILE * w + (a - col) * TILE;
                    any = 1;
                }
            }
            if (any) {
                hx_half_tile(children, stride, p->half);
                good = put_jpeg(p, layer, tile_cell(layer, 1, x, y));
            }
        }
    }
    free(canvas);
    return good;
}

static int paint_range(Painter *p, TileLayer *layer, Piece **pieces, int count, const char *folder,
    double mm, int start, int finish, Histogram histogram) {
    Draw *draw = calloc(count ? (size_t)count : 1, sizeof(*draw));
    if (!draw) {
        return 0;
    }
    double tile_mm = TILE * mm;
    TileSpan first = layer->spans[0];
    int rows = first.y + first.h, columns = first.x + first.w, ok = 1;
    for (int col = start; ok && col < finish; col += STRIP) {
        double left = col * tile_mm, right = (col + STRIP) * tile_mm;
        size_t used = 0;
        for (int k = 0; ok && k < count; k++) {
            Piece *piece = pieces[k];
            if (piece->retired || piece->x >= right || piece->x + piece->width <= left) {
                continue;
            }
            int status = prepare(piece, folder, layer->name, mm, histogram);
            if (status > 0) {
                status = draw_of(piece, left, mm, STRIP * TILE, rows * TILE, draw + used);
                used += status > 0;
            }
            ok = status >= 0;
        }
        if (ok && used) {
            ok = paint_strip(p, layer, draw, used, col, columns, mm);
        }
        for (int k = 0; k < count; k++) {
            Piece *piece = pieces[k];
            if (!piece->retired && piece->x + piece->width <= right) {
                free_images(piece);
                piece->retired = 1;
            }
        }
    }
    for (int i = 0; i < count; i++) {
        Piece *piece = pieces[i];
        if (ok && piece->sample) {
            ok = load_images(piece, folder, layer->name) > 0;
            if (ok) {
                sample_piece(piece, histogram);
            }
            free_images(piece);
        } else if (!piece->retired) {
            free_images(piece);
        }
        piece->retired = 0;
    }
    free(draw);
    return ok;
}

typedef struct {
    TilePack *pack;
    TileLayer *layer;
    Piece **pieces;
    const char *folder;
    double mm, sample_left, sample_right;
    Histogram histogram;
    int count, start, finish, ok;
} PaintJob;

static void *paint_job(void *context) {
    PaintJob *job = context;
    Piece *pieces = malloc((size_t)job->count * sizeof(*pieces));
    Piece **order = malloc((size_t)job->count * sizeof(*order));
    Painter painter = {0};
    if (pieces && order && painter_open(&painter, job->pack)) {
        for (int i = 0; i < job->count; i++) {
            pieces[i] = *job->pieces[i];
            pieces[i].sample &= pieces[i].x >= job->sample_left && pieces[i].x < job->sample_right;
            order[i] = pieces + i;
        }
        job->ok = paint_range(&painter, job->layer, order, job->count, job->folder, job->mm,
            job->start, job->finish, job->histogram);
    }
    painter_close(&painter);
    free(order);
    free(pieces);
    return NULL;
}

static int paint_layer(Painter *p, TileLayer *layer, Piece **pieces, int count, const char *folder,
    double mm, Histogram histogram) {
    const char *configured = getenv("HERCULEX_COMPOSE_PROCESSES");
    char *end;
    long workers = configured ? strtol(configured, &end, 10) : sysconf(_SC_NPROCESSORS_ONLN);
    if (configured && (end == configured || *end || workers < 1)) {
        return 0;
    }
    TileSpan span = layer->spans[0];
    int start = span.x / STRIP * STRIP, stop = span.x + span.w;
    int strips = (stop - start + STRIP - 1) / STRIP;
    if (strips < 2 || workers == 1) {
        return paint_range(p, layer, pieces, count, folder, mm, start, stop, histogram);
    }
    enum { MAX_WORKERS = 32 };
    PaintJob jobs[MAX_WORKERS];
    pthread_t threads[MAX_WORKERS];
    int jobs_count = workers < MAX_WORKERS ? (int)workers : MAX_WORKERS;
    if (jobs_count > strips) {
        jobs_count = strips;
    }
    int started = 0, left = start;
    double tile_mm = TILE * mm;
    for (int i = 0; i < jobs_count; i++) {
        int right = start + STRIP * (strips * (i + 1) / jobs_count);
        jobs[i] = (PaintJob){.pack = p->pack,
            .layer = layer,
            .pieces = pieces,
            .folder = folder,
            .mm = mm,
            .sample_left = i ? left * tile_mm : -INFINITY,
            .sample_right = i + 1 < jobs_count ? right * tile_mm : INFINITY,
            .count = count,
            .start = left,
            .finish = right};
        left = right;
        if (started == i && !pthread_create(threads + i, NULL, paint_job, jobs + i)) {
            started++;
        } else {
            paint_job(jobs + i);
        }
    }
    int ok = 1;
    for (int i = 0; i < jobs_count; i++) {
        if (i < started) {
            pthread_join(threads[i], NULL);
        }
        ok &= jobs[i].ok;
        for (int j = 0; j < 256; j++) {
            histogram[j] += jobs[i].histogram[j];
        }
    }
    return ok;
}

static int pyramid(Painter *p, TileLayer *layer) {
    if (layer->levels <= 2) {
        return 1;
    }
    uint8_t *inputs = malloc(4 * TILE * TILE), *encoded = NULL;
    size_t capacity = 0;
    if (!inputs) {
        return 0;
    }
    int ok = 1;
    const int64_t stride[] = {TILE, TILE, TILE, TILE};
    for (size_t level = 2; ok && level < layer->levels; level++) {
        TileSpan span = layer->spans[level];
        for (int x = span.x; ok && x < span.x + span.w; x++) {
            for (int y = span.y; ok && y < span.y + span.h; y++) {
                const uint8_t *children[4] = {0};
                int any = 0;
                for (int k = 0; ok && k < 4; k++) {
                    size_t cell = tile_cell(layer, level - 1, x * 2 + k % 2, y * 2 + k / 2);
                    if (cell == SIZE_MAX || !layer->sizes[cell]) {
                        continue;
                    }
                    uint32_t bytes = layer->sizes[cell];
                    if (capacity < bytes) {
                        uint8_t *new = realloc(encoded, bytes);
                        if (!new) {
                            ok = 0;
                            break;
                        }
                        capacity = bytes;
                        encoded = new;
                    }
                    uint8_t *child = inputs + (size_t)k * TILE * TILE;
                    ok = transfer_at(
                             p->pack->pack, encoded, bytes, (off_t)layer->offsets[cell], 0) &&
                        read_jpeg(encoded, bytes, child);
                    children[k] = child;
                    any = 1;
                }
                if (ok && any) {
                    hx_half_tile(children, stride, p->half);
                    ok = put_jpeg(p, layer, tile_cell(layer, level, x, y));
                }
            }
        }
    }
    free(inputs);
    free(encoded);
    return ok;
}

static json_object *pair_of(int a, int b) {
    json_object *pair = json_object_new_array_ext(2);
    if (pair) {
        json_object_array_add(pair, json_object_new_int(a));
        json_object_array_add(pair, json_object_new_int(b));
    }
    return pair;
}

static TileLayer *plan_layer(
    TilePack *pack, Piece **pieces, int count, const char *name, double mm) {
    double left = INFINITY, top = INFINITY, right = -INFINITY, bottom = -INFINITY;
    for (int i = 0; i < count; i++) {
        Piece *p = pieces[i];
        left = fmin(left, p->x);
        top = fmin(top, p->y);
        right = fmax(right, p->x + p->width);
        bottom = fmax(bottom, p->y + p->height);
    }
    double tile = TILE * mm;
    double width = ceil((right + 2) / tile), height = ceil((bottom + 2) / tile);
    double x = fmax(0, floor((left - 2) / tile)), y = fmax(0, floor((top - 2) / tile));
    if (!(x <= width && y <= height && width < INT_MAX / TILE - 1 && height < INT_MAX / TILE - 1)) {
        return NULL;
    }
    json_object *meta = json_object_new_object(), *grid = json_object_new_array(),
                *origin = json_object_new_array();
    TileLayer *layer = NULL;
    if (meta && grid && origin) {
        int x0 = x, y0 = y, x1 = width, y1 = height;
        for (int i = 0; i < LEVELS; i++) {
            json_object_array_add(grid, pair_of(x1 - x0, y1 - y0));
            json_object_array_add(origin, pair_of(x0, y0));
            if (x1 <= 1 && y1 <= 1) {
                break;
            }
            x0 /= 2;
            y0 /= 2;
            x1 = (x1 + 1) / 2;
            y1 = (y1 + 1) / 2;
        }
        json_object_object_add(meta, "grid", json_object_get(grid));
        json_object_object_add(meta, "origin", json_object_get(origin));
        json_object_object_add(meta, "raw", json_object_new_boolean(1));
        layer = tile_layer_add(pack, name, meta);
    }
    json_object_put(meta);
    json_object_put(grid);
    json_object_put(origin);
    return layer;
}

static const char *display_layer(const char *layer, const char *ink, double mm, char *depth) {
    if (!strcmp(layer, "taught")) {
        return *ink ? ink : "ink";
    }
    if (!strcmp(layer, "ink")) {
        return "ink";
    }
    if (!strcmp(layer, "taught-scan")) {
        return "scan under the ink";
    }
    if (!strcmp(layer, "sheet")) {
        return "brightest across the sheet";
    }
    const char *offset = !strncmp(layer, "layer", 5) ? layer + 5 : layer;
    char *end;
    errno = 0;
    long layers = strtol(offset, &end, 10);
    if (errno || end == offset || *end) {
        return NULL;
    }
    int n = snprintf(depth, 64, "%+.0f µm", layers * mm * 1000);
    return n > 0 && n < 64 ? depth : NULL;
}

static int compose_layer(Painter *p, const char *name, Piece *pieces, size_t count,
    const char *folder, double mm, const char *model, const char *ink, json_object *records) {
    int used = 0;
    Piece **order = malloc((count ? count : 1) * sizeof(*order));
    if (!order) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        Piece *piece = pieces + i;
        if (piece->mosaic) {
            order[used++] = piece;
        }
    }
    qsort(order, used, sizeof(*order), piece_order);
    int sampled = 0, raw = !ink_layer(name), limit = raw ? 64 : 1;
    for (int i = 0; i < used && sampled < limit; i++) {
        char *path = piece_image(folder, order[i], name);
        if (!path) {
            free(order);
            return 0;
        }
        if (!access(path, R_OK)) {
            order[i]->sample = raw;
            sampled++;
        }
        free(path);
    }
    if (!sampled) {
        free(order);
        return 1;
    }
    double low, high;
    Histogram histogram = {0};
    char depth[64];
    const char *label = display_layer(name, ink, mm, depth);
    TileLayer *layer = label ? plan_layer(p->pack, order, used, name, mm) : NULL;
    int ok =
        layer && paint_layer(p, layer, order, used, folder, mm, histogram) && pyramid(p, layer);
    for (int i = 0; i < used; i++) {
        order[i]->sample = 0;
    }
    free(order);
    shown_range(histogram, &low, &high);
    if (ok) {
        json_object_object_add(layer->metadata, "low", json_object_new_double(low));
        json_object_object_add(layer->metadata, "high", json_object_new_double(high));
    }
    json_object *record = ok ? json_object_new_object() : NULL;
    if (record) {
        json_object_object_add(record, "path", json_object_new_string(name));
        json_object_object_add(record, "name", json_object_new_string(label));
        json_object_object_add(record, "mm", json_object_new_double(mm));
        json_object_object_add(record, "levels", json_object_new_int64(layer->levels));
        json_object_object_add(record, "low", json_object_new_double(low));
        json_object_object_add(record, "high", json_object_new_double(high));
        json_object_object_add(record, "stamp", json_object_new_int64(time(NULL)));
        json_object_object_add(record, "raw", json_object_new_boolean(1));
        if (*model && !strcmp(name, "taught")) {
            json_object_object_add(record, "model", json_object_new_string(model));
        }
        json_object_array_add(records, record);
    }
    return record != NULL;
}

typedef struct {
    const char *sample, *scan, *page, *layers, *units, *model, *ink, *surface;
    double mm;
    int written, largest;
} Sheet;

static json_object *filtered_layout(const char *folder, const char *units) {
    json_object *doc = document_at(folder, "layout.json");
    json_object *rows = json_object_object_get(doc, "pieces");
    json_object *selected =
        json_object_is_type(rows, json_type_array) ? json_object_new_array() : NULL;
    size_t count = selected ? json_object_array_length(rows) : 0;
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *unit = text(row, "unit");
        if (unit && selected_unit(units, unit)) {
            json_object_array_add(selected, json_object_get(row));
        }
    }
    json_object_put(doc);
    return selected;
}

static int save_placement(const char *folder, const char *name, json_object *pieces) {
    json_object *doc = json_object_new_object();
    if (!doc) {
        return 0;
    }
    json_object_object_add(doc, "pieces", json_object_get(pieces));
    int ok = write_document(folder, name, doc);
    json_object_put(doc);
    return ok;
}

static int save_layers(const char *folder, json_object *records) {
    json_object *kept = document_at(folder, "layers.json");
    size_t fresh = json_object_array_length(records);
    if (json_object_is_type(kept, json_type_array)) {
        for (size_t k = 0; k < json_object_array_length(kept); k++) {
            json_object *old = json_object_array_get_idx(kept, k);
            const char *path = text(old, "path");
            int replaced = 0;
            for (size_t i = 0; path && !replaced && i < fresh; i++) {
                const char *name = text(json_object_array_get_idx(records, i), "path");
                replaced = !strcmp(name, path);
            }
            if (!replaced) {
                json_object_array_add(records, json_object_get(old));
            }
        }
    }
    json_object_put(kept);
    return write_document(folder, "layers.json", records);
}

static int compose(const Sheet *sheet, const char *folder, const char *out, const char *placement,
    const char *layout_path) {
    json_object *pieces = filtered_layout(layout_path, sheet->units), *taught = NULL;
    json_object *records = json_object_new_array();
    size_t count = pieces ? json_object_array_length(pieces) : 0;
    int ok =
        pieces && records && count <= INT32_MAX && !json_object_deep_copy(pieces, &taught, NULL);
    Piece *near = ok ? calloc(count ? count : 1, sizeof(*near)) : NULL;
    Piece *far = ok ? calloc(count ? count : 1, sizeof(*far)) : NULL;
    char *layers = strdup(sheet->layers);
    Painter p = {0};
    TilePack pack = {.directory = -1, .pack = -1};
    ok = ok && near && far && layers &&
        load_layout(pieces, folder, "side", near, count, sheet->largest) &&
        load_layout(taught, folder, "taught-side", far, count, sheet->largest) &&
        hx_site_folders(out) && hx_site_folders(placement) && tile_pack_open(&pack, out);
    ok = ok && painter_open(&p, &pack);
    char *next = layers;
    while (ok && next) {
        char *name = strsep(&next, ",");
        Piece *group = sheet->written || !strncmp(name, "taught", 6) ? far : near;
        ok = valid_path(name, 0) &&
            compose_layer(
                &p, name, group, count, folder, sheet->mm, sheet->model, sheet->ink, records);
    }
    size_t written = records ? json_object_array_length(records) : 0;
    if (ok && written) {
        json_object *visible = sheet->written ? taught : pieces;
        ok = tile_pack_publish(&pack) && save_layers(out, records) &&
            save_placement(placement, "placed.json", visible) &&
            save_placement(placement, "placed-taught.json", taught) &&
            (!sheet->surface ||
                hx_sheet_review(visible, records, out, sheet->sample, sheet->scan, sheet->surface));
    } else if (ok) {
        fprintf(stderr, "HerculeX: no source images for %s: %s\n", sheet->page, sheet->layers);
        ok = 0;
    }
    if (ok) {
        printf("composed %s ", sheet->page);
        for (size_t i = 0; i < written; i++) {
            printf("%s%s", i ? ", " : "", text(json_object_array_get_idx(records, i), "path"));
        }
        putchar('\n');
    }
    tile_pack_close(&pack);
    painter_close(&p);
    free(layers);
    free(near);
    free(far);
    json_object_put(records);
    json_object_put(taught);
    json_object_put(pieces);
    return ok;
}

static int sheet_tiles(const Sheet *sheet) {
    const char *model = getenv("HERCULEX_INK_MODEL"), *faces = getenv("HERCULEX_FACES");
    char *produced = NULL, *base = NULL, *folder = NULL, *out = NULL;
    char *outputs = local_root(outputs_root);
    int stock = !model || !*model || !strcmp(model, "reader-v3");
    int ok =
        outputs && asprintf(&produced, "level-0%s%s", stock ? "" : "-", stock ? "" : model) >= 0;
    faces = faces ? faces : produced;
    ok = ok && (!*faces || valid_path(faces, 1)) &&
        asprintf(&base, "%s/unwrapping/level/%s/%s/sheet", outputs, sheet->sample, sheet->scan) >=
            0 &&
        asprintf(&folder, "%s/unwrapping/level/%s/%s/faces/%s", outputs, sheet->sample, sheet->scan,
            faces) >= 0 &&
        asprintf(&out, "%s/website/sheets/%s", outputs, sheet->page) >= 0;
    if (ok) {
        const char *placement = sheet->units || strcmp(faces, produced) != 0 ? out : base;
        ok = compose(sheet, folder, out, placement, base);
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: could not compose %s %s at %s\n", sheet->sample, sheet->scan,
            sheet->page);
    }
    free(out);
    free(folder);
    free(base);
    free(produced);
    free(outputs);
    return ok ? 0 : 1;
}

static int options(Sheet *sheet, int argc, char **argv) {
    const struct {
        const char *name;
        const char **value;
    } named[] = {{"--units", &sheet->units}, {"--model", &sheet->model},
        {"--ink-name", &sheet->ink}, {"--review-surface", &sheet->surface}};
    const char *args[6];
    int used = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--written-face")) {
            sheet->written = 1;
            continue;
        }
        const char **value = NULL;
        size_t length = strcspn(arg, "=");
        for (size_t k = 0; k < sizeof(named) / sizeof(*named); k++) {
            if (length == strlen(named[k].name) && !memcmp(arg, named[k].name, length)) {
                value = named[k].value;
                break;
            }
        }
        if (value) {
            if (arg[length] == '=') {
                *value = arg + length + 1;
            } else if (i + 1 < argc) {
                *value = argv[++i];
            } else {
                return 0;
            }
        } else if (used < 6 && strncmp(arg, "--", 2) != 0) {
            args[used++] = arg;
        } else {
            return 0;
        }
    }
    if (used < 4 || (used == 6 && strcmp(args[5], "largest") != 0)) {
        return 0;
    }
    sheet->sample = args[0];
    sheet->scan = args[1];
    sheet->page = args[3];
    sheet->layers = used >= 5 ? args[4] : "ink,layer0,sheet";
    sheet->largest = used == 6;
    char *end;
    sheet->mm = strtod(args[2], &end);
    return end != args[2] && !*end && isfinite(sheet->mm) && sheet->mm > 0 &&
        valid_path(sheet->sample, 0) && valid_path(sheet->scan, 0) && valid_path(sheet->page, 1);
}

#ifdef HX_UNITY
int hx_sheet_tiles_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    if (argc > 1 && strcmp(argv[1], "--verify") == 0) {
        return hx_check_review(argc - 1, argv + 1);
    }
    if (argc > 1 && strcmp(argv[1], "--card") == 0) {
        return card_command(argc - 1, argv + 1);
    }
    Sheet sheet = {.model = "", .ink = ""};
    if (!options(&sheet, argc, argv)) {
        puts("Usage: ./run.sh viewer.sheet_tiles SCROLL SCAN PIXEL_MM PAGE [LAYERS] [largest]\n"
             "       [--units LIST] [--model TITLE] [--ink-name TITLE] [--written-face]\n"
             "       [--review-surface SURFACE]\n"
             "       ./run.sh viewer.sheet_tiles --verify FOLDER SCAN URL\n"
             "       ./run.sh viewer.sheet_tiles --card FOLDER LEVEL X Y W H OUTPUT");
        return argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 2;
    }
    return sheet_tiles(&sheet);
}
