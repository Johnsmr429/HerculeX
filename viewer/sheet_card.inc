typedef struct {
    double x, y;
    int w, h;
    uint64_t filled;
} CardView;

typedef struct {
    TileIndex index;
    TileSpan span;
    double mm;
    uint8_t stretch[256], *compressed, *pixels;
    size_t capacity;
} Card;

static const char *card_layer(json_object *layers, json_object **selected) {
    const char *names[] = {"layer0", "taught-scan", "sheet"};
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
        for (size_t j = 0; j < json_object_array_length(layers); j++) {
            json_object *layer = json_object_array_get_idx(layers, j);
            const char *path = json_object_get_string(json_object_object_get(layer, "path"));
            if (path && strcmp(path, names[i]) == 0) {
                *selected = layer;
                return names[i];
            }
        }
    }
    return NULL;
}

static int card_open(Card *card, const char *folder, int level) {
    int directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int opened = directory >= 0 ? tile_index_open(&card->index, directory) : 0;
    if (directory >= 0) {
        close(directory);
    }
    json_object *layers = opened == 1 ? document_at(folder, "layers.json") : NULL;
    json_object *layer = NULL;
    const char *name =
        json_object_is_type(layers, json_type_array) ? card_layer(layers, &layer) : NULL;
    json_object *entry = name ? json_object_object_get(card->index.layers, name) : NULL;
    json_object *grid = json_object_object_get(entry, "grid");
    card->mm = ldexp(json_object_get_double(json_object_object_get(layer, "mm")), level);
    int ok = isfinite(card->mm) && card->mm > 0 && json_object_is_type(grid, json_type_array) &&
        ((size_t)level >= json_object_array_length(grid) ||
            tile_index_level(&card->index, entry, level, &card->span));
    int raw = json_object_get_boolean(json_object_object_get(layer, "raw"));
    double low = json_object_get_double(json_object_object_get(layer, "low"));
    double high = json_object_get_double(json_object_object_get(layer, "high"));
    if (raw && !(isfinite(low) && isfinite(high) && high > low)) {
        ok = 0;
    }
    for (int i = 0; ok && i < 256; i++) {
        float value = raw ? ((float)i - (float)low) * (float)(255.0 / (high - low)) : i;
        card->stretch[i] = fminf(255, fmaxf(0, value));
    }
    json_object_put(layers);
    card->pixels = ok ? malloc(TILE * TILE) : NULL;
    return card->pixels != NULL;
}

static int card_read(Card *card, uint64_t offset, uint32_t length) {
    if (length > card->capacity) {
        uint8_t *next = realloc(card->compressed, length);
        if (!next) {
            return 0;
        }
        card->compressed = next;
        card->capacity = length;
    }
    if (!transfer_at(card->index.pack, card->compressed, length, offset, 0)) {
        return 0;
    }
    if (length < 8 || png_sig_cmp(card->compressed, 0, 8) != 0) {
        return read_jpeg(card->compressed, length, card->pixels);
    }
    png_image png = {.version = PNG_IMAGE_VERSION};
    int ok = png_image_begin_read_from_memory(&png, card->compressed, length) &&
        png.width == TILE && png.height == TILE;
    png.format = PNG_FORMAT_GRAY;
    png_color black = {0};
    ok = ok && png_image_finish_read(&png, &black, card->pixels, 0, NULL);
    png_image_free(&png);
    return ok;
}

static Box card_clip(const CardView *view, double px, double py, double mm, int *x, int *y) {
    double dx = nearbyint((px - view->x) / mm), dy = nearbyint((py - view->y) / mm);
    if (dx <= -TILE || dy <= -TILE || dx >= view->w || dy >= view->h) {
        return (Box){0};
    }
    *x = dx;
    *y = dy;
    return clipped(*x, *y, TILE, TILE, view->w, view->h);
}

static int draw_cards(Card *card, CardView *views, size_t count, uint8_t *output) {
    const TileSpan *span = &card->span;
    for (int r = 0; r < span->h; r++) {
        for (int c = 0; c < span->w; c++) {
            int64_t tx = (int64_t)c + span->x, ty = (int64_t)r + span->y;
            if (tx < 0 || ty < 0) {
                continue;
            }
            uint64_t offset;
            uint32_t length;
            if (!tile_index_blob(
                    &card->index, span->first + (size_t)r * span->w + c, &offset, &length)) {
                return 0;
            }
            int decoded = 0;
            for (size_t i = 0; length && i < count; i++) {
                int x, y;
                CardView *view = views + i;
                Box box = card_clip(
                    view, tx * (TILE * card->mm), ty * (TILE * card->mm), card->mm, &x, &y);
                if (!nonempty(box)) {
                    continue;
                }
                if (!decoded && !card_read(card, offset, length)) {
                    return 0;
                }
                decoded = 1;
                for (int iy = box.top; iy < box.bottom; iy++) {
                    const uint8_t *row = card->pixels + (iy - y) * TILE;
                    for (int ix = box.left; ix < box.right; ix++) {
                        uint8_t value = card->stretch[row[ix - x]];
                        if (output) {
                            output[(size_t)iy * view->w + ix] = value;
                        } else {
                            view->filled += value > 0;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static int card_views(const double *extent, double mm, CardView views[25]) {
    double w = 1800 * mm, h = 1125 * mm;
    const double steps[] = {.2, .35, .5, .65, .8};
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            double x = extent[0] + extent[2] * steps[i] - w / 2;
            double y = extent[1] + extent[3] * steps[j] - h / 2;
            double cols = (x + w - x) / mm, rows = (y + h - y) / mm;
            if (!(cols >= 1 && cols <= 1801 && rows >= 1 && rows <= 1126)) {
                return 0;
            }
            views[5 * i + j] = (CardView){.x = x, .y = y, .w = cols, .h = rows};
        }
    }
    return 1;
}

static int make_card(Card *card, const double extent[4], const char *path) {
    CardView views[25];
    size_t count = sizeof(views) / sizeof(*views);
    if (!card_views(extent, card->mm, views) || !draw_cards(card, views, count, NULL)) {
        return 0;
    }
    CardView best = views[0];
    for (size_t i = 1; i < count; i++) {
        CardView view = views[i];
        if (view.filled * best.w * best.h > best.filled * view.w * view.h) {
            best = view;
        }
    }
    uint8_t *pixels = calloc((size_t)best.w, best.h);
    JpegWriter writer = {0};
    int ok = pixels && draw_cards(card, &best, 1, pixels) &&
        encode_jpeg(&writer, pixels, best.w, best.h, 92, 1);
    writer.codec.err = NULL;
    jpeg_destroy_compress(&writer.codec);
    ok = ok && hx_write_sheet_file(path, writer.data, writer.length);
    free(writer.data);
    free(pixels);
    return ok;
}

int hx_sheet_card(const char *folder, int level, const double extent[4], const char *path) {
    Card card = {.index.pack = -1};
    int ok = card_open(&card, folder, level) && make_card(&card, extent, path);
    if (!ok) {
        fprintf(stderr, "HerculeX: cannot make a review card for %s\n", folder);
    }
    free(card.pixels);
    free(card.compressed);
    tile_index_close(&card.index);
    return ok;
}

static int card_command(int argc, char **argv) {
    if (argc != 8) {
        puts("Usage: ./run.sh viewer.sheet_tiles --card FOLDER LEVEL X Y W H OUTPUT");
        return 2;
    }
    char *end;
    long level = strtol(argv[2], &end, 10);
    if (end == argv[2] || *end || level < 0 || level > LEVELS) {
        return 2;
    }
    double extent[4];
    for (size_t i = 0; i < 4; i++) {
        extent[i] = strtod(argv[3 + i], &end);
        if (end == argv[3 + i] || *end || !isfinite(extent[i])) {
            return 2;
        }
    }
    return hx_sheet_card(argv[1], level, extent, argv[7]) ? 0 : 1;
}
