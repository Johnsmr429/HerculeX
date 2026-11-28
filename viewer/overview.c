enum { SITE_OVERVIEW_TILE = 512 };

typedef struct {
    int w, h;
    u8 *pixels;
} OverviewCut;

typedef struct {
    TileIndex index;
    u8 *blob, *tile;
    size_t capacity;
} OverviewStore;

static int overview_cut(Arena *a, OverviewStore *store, const char *name, int level, double step,
    double x0, double x1, double y1, OverviewCut *result) {
    double a0 = x0 / step, a1 = ceil(x1 / step), b1 = ceil(y1 / step);
    if (!(a0 >= 0 && a1 >= a0 && a1 < INT_MAX && b1 >= 0 && b1 < INT_MAX)) {
        return 0;
    }
    int left = (int)a0, right = (int)a1, bottom = (int)b1, tile = SITE_OVERVIEW_TILE;
    int w = right > left ? right - left : 1, h = bottom > 0 ? bottom : 1;
    u8 *pixels = pushz(a, (size_t)w * h);
    *result = (OverviewCut){w, h, pixels};
    json_object *layer = json_object_object_get(store->index.layers, name);
    TileSpan span;
    if (!tile_index_level(&store->index, layer, level, &span)) {
        return 1;
    }
    int rows = (h - 1) / tile + 1, first = left / tile, last = (left + w - 1) / tile + 1;
    int row_end = fmin(rows, (int64_t)span.y + span.h);
    int col_start = fmax(first, span.x);
    int col_end = fmin(last, (int64_t)span.x + span.w);
    for (int row = fmax(0, span.y); row < row_end; row++) {
        for (int col = col_start; col < col_end; col++) {
            size_t cell = span.first + (size_t)(row - span.y) * span.w + (col - span.x);
            uint32_t size;
            uint64_t off;
            if (!tile_index_blob(&store->index, cell, &off, &size)) {
                return 0;
            }
            if (!size) {
                continue;
            }
            if (store->capacity < size) {
                u8 *blob = realloc(store->blob, size);
                if (!blob) {
                    return 0;
                }
                store->blob = blob;
                store->capacity = size;
            }
            memset(store->tile, 0, (size_t)tile * tile);
            if (!transfer_at(store->index.pack, store->blob, size, off, 0) ||
                !site_raw_tile(store->blob, size, store->tile, tile)) {
                return 0;
            }
            int first_x = col * tile > left ? col * tile : left, first_y = row * tile;
            int end_x = (col + 1) * tile < right ? (col + 1) * tile : right;
            int end_y = (row + 1) * tile < bottom ? (row + 1) * tile : bottom;
            for (int y = first_y; end_x > first_x && y < end_y; y++) {
                memcpy(pixels + (size_t)y * w + first_x - left,
                    store->tile + (size_t)(y - row * tile) * tile + first_x - col * tile,
                    end_x - first_x);
            }
        }
    }
    return 1;
}

typedef struct {
    double floor, scan;
    int width, ink, compact, yellow;
} OverviewOptions;

static int overview_options(OverviewOptions *options, int argc, char **argv) {
    *options = (OverviewOptions){.width = 2400, .ink = 1, .compact = 1, .floor = 24, .scan = .6};
    for (int i = 2; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--no-ink")) {
            options->ink = 0;
        } else if (!strcmp(option, "--loose")) {
            options->compact = 0;
        } else if (++i >= argc) {
            return 0;
        } else if (!strcmp(option, "--tint")) {
            options->yellow = !strcmp(argv[i], "yellow");
            if (!options->yellow && strcmp(argv[i], "purple") != 0) {
                return 0;
            }
        } else {
            char *end;
            double value = strtod(argv[i], &end);
            if (end == argv[i] || *end || !isfinite(value)) {
                return 0;
            }
            if (!strcmp(option, "--width") && value == floor(value) && value > 0 &&
                value <= INT_MAX) {
                options->width = value;
            } else if (!strcmp(option, "--scan")) {
                options->scan = value;
            } else if (!strcmp(option, "--floor") && value < 255) {
                options->floor = value;
            } else {
                return 0;
            }
        }
    }
    return 1;
}

int hx_overview_command(int argc, char **argv) {
    OverviewOptions opt;
    need(argc >= 2 && site_name(argv[0]) && overview_options(&opt, argc, argv),
        "usage: ./run.sh viewer.overview PAGE OUT [--width PIXELS] [--floor VALUE] "
        "[--scan VALUE] [--tint purple|yellow] [--no-ink] [--loose]");
    Arena a = arena(4 * GB);
    const char *folder = fmt(&a, "%s/website/sheets/%s", OUTPUTS, argv[0]);
    const char *target = argv[1];
    json_object *meta = hx_viewer_metadata(fmt(&a, "%s/index.html", folder));
    json_object *layers = json_object_object_get(meta, "layers"), *sheet = NULL;
    const char *ink = NULL;
    for (size_t i = 0; i < site_items(layers); i++) {
        json_object *row = json_object_array_get_idx(layers, i);
        const char *path = site_string(row, "path");
        if (path && !strcmp(path, "sheet")) {
            sheet = row;
        } else if (path && opt.ink && (!strcmp(path, "taught") || (!ink && !strcmp(path, "ink")))) {
            ink = path;
        }
    }
    double mm = site_value(sheet, "mm"), low = site_value(sheet, "low");
    double high = site_value(sheet, "high");
    int levels = site_value(sheet, "levels");
    need(
        mm > 0 && high > low && levels > 0 && levels <= 64, "viewer %s has no scan layer", argv[0]);
    json_object *shelves = json_object_object_get(meta, "shelves");
    size_t count = site_items(shelves);
    double widest = 0, tallest = 0;
    for (size_t i = 0; i < (count ? count : 1); i++) {
        json_object *box =
            count ? json_object_array_get_idx(shelves, i) : json_object_object_get(meta, "area");
        widest = fmax(widest, site_value(box, "w"));
        tallest = fmax(tallest, (count ? site_value(box, "y") : 0) + site_value(box, "h"));
    }
    need(widest > 0 && tallest > 0, "viewer %s has an empty area", argv[0]);
    int level = fmax(0, fmin(levels - 1, floor(log2(widest / mm / opt.width))));
    double step = ldexp(mm, level);
    double nx = ceil(widest / step), ny = ceil(tallest / step);
    need(nx > 0 && nx <= INT_MAX && ny > 0 && ny <= INT_MAX && nx * ny * 3 < GB,
        "overview canvas is too large");
    int width = nx, height = ny;
    u8 *canvas = pushz(&a, (size_t)width * height * 3);
    int *bands = push(&a, (count ? count : 1) * sizeof(int) * 2);
    size_t shown = 0;
    OverviewStore store = {
        .index.pack = -1, .tile = push(&a, SITE_OVERVIEW_TILE * SITE_OVERVIEW_TILE)};
    int directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    need(directory >= 0 && tile_index_open(&store.index, directory) == 1, "cannot read tiles of %s",
        argv[0]);
    close(directory);
    const double dark[2][3] = {{.33, .10, .62}, {.69, .44, .06}};
    const double light[2][3] = {{.82, .55, 1}, {1, .957, .376}};
    float floor_255 = opt.floor / 255, scan_scale = opt.scan;
    float scale = high - low, low_level = low;
    size_t rewind = a.used;
    for (size_t i = 0; i < (count ? count : 1); i++) {
        json_object *box =
            count ? json_object_array_get_idx(shelves, i) : json_object_object_get(meta, "area");
        double x0 = count ? site_value(box, "x") : 0;
        double y0 = count ? nearbyint(site_value(box, "y") / step) : 0;
        double x1 = x0 + site_value(box, "w"), y1 = site_value(box, "h");
        need(y0 >= 0 && y0 <= height, "viewer %s has an invalid shelf", argv[0]);
        OverviewCut scan = {0}, read = {0};
        need(overview_cut(&a, &store, "sheet", level, step, x0, x1, y1, &scan) &&
                (!ink || overview_cut(&a, &store, ink, level, step, x0, x1, y1, &read)),
            "could not decode a shelf tile");
        int start = -1, end = -1, row0 = (int)y0;
        int h = scan.h < height - row0 ? scan.h : height - row0;
        int w = scan.w < width ? scan.w : width;
        for (int y = 0; y < h; y++) {
            int bright = 0;
            for (int x = 0; x < w; x++) {
                size_t at = (size_t)y * scan.w + x;
                float grey = scan_scale * fminf(1, fmaxf(0, (scan.pixels[at] - low_level) / scale));
                float t = ink
                    ? fminf(1, fmaxf(0, (read.pixels[at] / 255.f - floor_255) / (1.f - floor_255)))
                    : 0;
                float alpha = fminf(1, 3 * t);
                u8 *p = canvas + ((size_t)(row0 + y) * width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    double d = dark[opt.yellow][c], l = light[opt.yellow][c];
                    double value = grey * (1 - alpha) + (d + (l - d) * t) * alpha;
                    unsigned byte = (unsigned)(int)(value * 255);
                    p[c] = (u8)byte;
                    bright |= value > .02;
                }
            }
            if (bright) {
                if (start < 0) {
                    start = row0 + y;
                }
                end = row0 + y + 1;
            }
        }
        if (start >= 0) {
            bands[2 * shown] = start;
            bands[2 * shown + 1] = end;
            shown++;
        }
        a.used = rewind;
    }
    if (opt.compact && shown) {
        int gap = fmax(1, nearbyint(10.0 * width / opt.width));
        size_t n = (shown - 1) * gap;
        for (size_t i = 0; i < shown; i++) {
            n += bands[2 * i + 1] - bands[2 * i];
        }
        need(n <= INT_MAX && n * width * 3 < GB, "overview canvas is too large");
        u8 *compact = pushz(&a, n * width * 3), *dest = compact;
        for (size_t i = 0; i < shown; i++) {
            size_t rows = bands[2 * i + 1] - bands[2 * i];
            memcpy(dest, canvas + (size_t)bands[2 * i] * width * 3, rows * width * 3);
            if (i + 1 < shown) {
                dest += (rows + gap) * width * 3;
            }
        }
        canvas = compact;
        height = n;
    }
    double scaled = fmax(1, nearbyint((double)height * opt.width / width));
    need(scaled <= INT_MAX && scaled * opt.width * 3 < GB, "overview image is too large");
    int result_height = scaled;
    u8 *result = push(&a, (size_t)opt.width * result_height * 3);
    need(hx_site_resize_rgb(canvas, width, height, result, opt.width, result_height),
        "cannot resize overview image");
    const char *suffix = strrchr(target, '.');
    int ok;
    if (suffix && (!strcmp(suffix, ".jpg") || !strcmp(suffix, ".jpeg"))) {
        ok = hx_site_save_jpeg(target, result, opt.width, result_height, 90);
    } else {
        png_image png = {.version = PNG_IMAGE_VERSION,
            .width = opt.width,
            .height = result_height,
            .format = PNG_FORMAT_RGB};
        png_alloc_size_t n = 0;
        ok = png_image_write_to_memory(&png, NULL, &n, 0, result, 0, NULL);
        void *bytes = ok ? push(&a, n) : NULL;
        ok = ok && png_image_write_to_memory(&png, bytes, &n, 0, result, 0, NULL) &&
            hx_write_sheet_file(target, bytes, n);
    }
    need(ok, "cannot write %s", target);
    printf("%s (%d, %d)\n", target, opt.width, result_height);
    free(store.blob);
    tile_index_close(&store.index);
    json_object_put(meta);
    munmap(a.base, a.size);
    return 0;
}
