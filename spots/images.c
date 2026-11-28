enum { SPOT_IMAGE_BAND = 2, SPOT_IMAGE_HOLE = 26, SPOT_IMAGE_STEP = 4 };

typedef struct {
    double mean, sd, low, high;
} SpotStats;

typedef struct {
    const char *name, *folder;
    const _Float16 **piece;
    SpotStats stats;
} SpotImageModel;

typedef struct {
    const u8 *volume;
    const char *name;
    int w, h, columns;
} SpotImagePiece;

typedef struct {
    int piece, pixel;
} SpotOwner;

typedef struct {
    f32 mean;
    double p95, p97, inky;
} SpotImageSummary;

static void spot_distance(i32 *distance, int h, int w) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = y * w + x, d = distance[i];
            if (y && distance[i - w] < d - 1) {
                d = distance[i - w] + 1;
            }
            if (x && distance[i - 1] < d - 1) {
                d = distance[i - 1] + 1;
            }
            distance[i] = d;
        }
    }
    for (int y = h - 1; y >= 0; y--) {
        for (int x = w - 1; x >= 0; x--) {
            int i = y * w + x, d = distance[i];
            if (y + 1 < h && distance[i + w] < d - 1) {
                d = distance[i + w] + 1;
            }
            if (x + 1 < w && distance[i + 1] < d - 1) {
                d = distance[i + 1] + 1;
            }
            distance[i] = d;
        }
    }
}

static void spot_place_image(Arena *a, SpotOwner *owner, int h, int w, int x0, int y0,
    const SpotImagePiece *p, int id, json_object *r, const char *surface, const char *work) {
    size_t saved = a->used;
    json_object *lay = jget(r, "layout"), *crops = json_object_object_get(r, "crops");
    json_object *crop = json_object_object_get(crops, surface);
    if (!crop) {
        crop = jget(r, "crop");
    }
    int r0 = json_object_get_int(json_object_array_get_idx(crop, 0));
    int c0 = json_object_get_int(json_object_array_get_idx(crop, 2));
    int flip_y = json_object_get_boolean(jget(lay, "flip_rows"));
    int flip_x = json_object_get_boolean(jget(lay, "flip_cols"));
    json_object *shape = jget(lay, "shape");
    int rows = json_object_get_int(json_object_array_get_idx(shape, 0));
    int cols = json_object_get_int(json_object_array_get_idx(shape, 1));
    int stride = (int)ji64(lay, "stride");
    double um = jf64(lay, "um");
    need(um > 0 && stride > 0, "spot piece %s has invalid placement", p->name);
    int top = (int)nearbyint(jf64(lay, "y_mm") / um +
        (flip_y ? (rows - 1 - r0 + MARGIN) * stride - (p->h - 1) : (r0 - MARGIN) * stride));
    int left = (int)nearbyint(jf64(lay, "x_mm") / um +
        (flip_x ? (cols - 1 - c0 + MARGIN) * stride - (p->w - 1) : (c0 - MARGIN) * stride));
    int a0 = top > y0 ? top : y0, b0 = left > x0 ? left : x0;
    int a1 = top + p->h < y0 + h ? top + p->h : y0 + h;
    int b1 = left + p->w < x0 + w ? left + p->w : x0 + w;
    if (a1 <= a0 || b1 <= b0) {
        return;
    }
    Surface mesh = load_surface(a, fmt(a, "%s/surfaces/%s/%s", work, surface, p->name));
    i64 nodes = (i64)mesh.h * mesh.w;
    u8 *good = push(a, nodes);
    f32 *mesh_spare = push(a, nodes * sizeof(*mesh_spare));
    f32 *mesh_work = push(a, 3 * nodes * sizeof(*mesh_work));
    i64 pad = (i64)(mesh.h + 2 * BRIDGE + 4) * (mesh.w + 2 * BRIDGE + 4);
    i32 *queue = push(a, (pad > p->columns ? pad : p->columns) * sizeof(*queue));
    bridged_mesh(&mesh, good, mesh_spare, mesh_work, queue);
    u8 *held = push(a, p->columns), *filled = push(a, p->columns);
    for (int i = 0; i < p->columns; i++) {
        u8 ok = 1;
        for (int z = SPOT_MIDDLE - SPOT_IMAGE_BAND; z <= SPOT_MIDDLE + SPOT_IMAGE_BAND; z++) {
            ok &= p->volume[(i64)z * p->columns + i] > 0;
        }
        held[i] = filled[i] = ok;
    }
    fill_holes(filled, p->h, p->w, queue);
    for (int i = 0; i < p->columns; i++) {
        queue[i] = filled[i] && !held[i] ? SPOT_IMAGE_HOLE + 1 : 0;
    }
    spot_distance(queue, p->h, p->w);
    for (int i = 0; i < p->columns; i++) {
        queue[i] = queue[i] > SPOT_IMAGE_HOLE ? 0 : SPOT_IMAGE_HOLE + 1;
    }
    spot_distance(queue, p->h, p->w);
    for (int y = a0; y < a1; y++) {
        int row = flip_y ? p->h - 1 - (y - top) : y - top;
        int mr = (int)nearbyint(row / (double)stride) - MARGIN;
        if (mr < 0 || mr >= mesh.h) {
            continue;
        }
        for (int x = b0; x < b1; x++) {
            int i = (y - y0) * w + x - x0;
            if (owner[i].piece >= 0) {
                continue;
            }
            int col = flip_x ? p->w - 1 - (x - left) : x - left;
            int mc = (int)nearbyint(col / (double)stride) - MARGIN;
            int pixel = row * p->w + col;
            if (mc >= 0 && mc < mesh.w && mesh.xyz[(i64)mr * mesh.w + mc] > 0 &&
                (held[pixel] || (filled[pixel] && queue[pixel] > SPOT_IMAGE_HOLE))) {
                owner[i] = (SpotOwner){.piece = id, .pixel = pixel};
            }
        }
    }
    a->used = saved;
}

static f32 spot_image_read(
    const SpotImageModel *model, const SpotImagePiece *pieces, SpotOwner at, int depth, int side) {
    if (at.piece < 0 || !model->piece[at.piece]) {
        return NAN;
    }
    i64 plane = (i64)(side * SPOT_DEPTHS + depth) * pieces[at.piece].columns;
    return model->piece[at.piece][plane + at.pixel];
}

static double spot_percentile(const f32 *sorted, i64 n, double fraction) {
    if (!n) {
        return NAN;
    }
    double at = (n - 1) * fraction;
    i64 low = (i64)at, high = low + (low + 1 < n);
    double t = at - low, delta = (double)sorted[high] - sorted[low];
    return t < 0.5 ? sorted[low] + t * delta : sorted[high] - (1 - t) * delta;
}

static SpotStats spot_image_stats(f32 *values, f32 *work, i64 count, double low, double high) {
    i64 n = 0;
    for (i64 i = 0; i < count; i++) {
        if (isfinite(values[i])) {
            values[n++] = values[i];
        }
    }
    need(n > 0, "the spot has no valid pixels for image statistics");
    f32 mean = mean_of(values, n);
    for (i64 i = 0; i < n; i++) {
        f32 delta = values[i] - mean;
        work[i] = delta * delta;
    }
    f32 sd = sqrtf(mean_of(work, n));
    qsort(values, n, sizeof(*values), by_float);
    return (SpotStats){.mean = mean,
        .sd = sd ? sd : 1,
        .low = spot_percentile(values, n, low),
        .high = spot_percentile(values, n, high)};
}

static SpotImageSummary spot_jpeg(Arena *a, const char *page, const char *surface,
    const char *stack, int side, int depth, f32 *frame, SpotOwner *owner, const int *inner,
    int ninner, f32 *small, u8 *grey, int h, int w, double low, double high, int dark) {
    size_t mark = a->used;
    int n = h * w, count = 0, inky = 0;
    f32 scale = (f32)fmax(high - low, 1e-9);
    for (int i = 0; i < n; i++) {
        if (owner[i].piece < 0) {
            grey[i] = 0;
            continue;
        }
        f32 value = frame[i],
            level = isnan(value) ? 0 : fminf(1, fmaxf(0, (value - (f32)low) / scale));
        grey[i] = (u8)(255 * (dark ? 1 - level : level));
    }
    for (int i = 0; i < ninner; i++) {
        f32 value = frame[inner[i]];
        if (!isnan(value)) {
            small[count++] = value;
        }
        inky += value > 2;
    }
    f32 mean = count ? mean_of(small, count) : NAN;
    qsort(small, count, sizeof(*small), by_float);
    SpotImageSummary stats = {.mean = mean,
        .p95 = spot_percentile(small, count, 0.95),
        .p97 = spot_percentile(small, count, 0.97),
        .inky = ninner ? (double)inky / ninner : NAN};
    Bytes jpeg = jpeg_encode(a, grey, w, h, 85);
    write_file(fmt(a, "%s/images/%s/%s/%s/d%+03d.jpg", page, surface, stack, spot_sides[side],
                   depth - SPOT_REACH),
        jpeg.data, jpeg.size);
    a->used = mark;
    return stats;
}

typedef struct {
    int id;
    double area;
} SpotImageOrder;

static int spot_image_order(const void *one, const void *two) {
    const SpotImageOrder *a = one, *b = two;
    return a->area == b->area ? (a->id > b->id) - (a->id < b->id) : a->area > b->area ? -1 : 1;
}

static json_object *spot_round(double v, double digits) {
    return isfinite(v) ? jnum(nearbyint(v * digits) / digits) : json_object_new_double(v);
}

static json_object *spot_image_surface(Arena *a, json_object *spot, json_object *records,
    const char *work, const char *page, const char *surface, json_object *names,
    json_object *manifest, json_object *reads) {
    int count = (int)site_items(records), nlisted = (int)site_items(names);
    if (!count || !nlisted) {
        return NULL;
    }
    SpotImagePiece *pieces = push(a, (size_t)count * sizeof(*pieces));
    SpotImageOrder *order = push(a, (size_t)count * sizeof(*order));
    int held = 0;
    Bytes *mapped = push(a, ((size_t)nlisted + 1) * count * sizeof(*mapped));
    for (int i = 0; i < count; i++) {
        json_object *r = json_object_array_get_idx(records, i);
        const char *name = spot_piece(r);
        char *path = fmt(a, "%s/stacks/%s/%s.npy", work, surface, name);
        Bytes bytes = spot_map(path);
        if (!bytes.data) {
            for (int k = 0; k < held; k++) {
                munmap(mapped[k].data, mapped[k].size);
            }
            return NULL;
        }
        mapped[held++] = bytes;
        Array stack = spot_stack(bytes, path);
        pieces[i] = (SpotImagePiece){.volume = stack.data,
            .name = name,
            .w = (int)stack.shape[2],
            .h = (int)stack.shape[1],
            .columns = (int)(stack.shape[1] * stack.shape[2])};
        order[i] = (SpotImageOrder){.id = i, .area = jf64(jget(r, "layout"), "area")};
    }
    SpotImageModel *models = pushz(a, ((size_t)nlisted + 2) * sizeof(*models));
    models[0] = (SpotImageModel){.name = "ensemble 5 depths", .folder = "ensemble-5-depths"};
    models[1] = (SpotImageModel){.name = "ensemble", .folder = "ensemble"};
    int nmodel = 2;
    for (int n = 0; n < nlisted; n++) {
        const char *name = json_object_get_string(json_object_array_get_idx(names, n));
        need(name && site_name(name), "invalid model in the spot recipe");
        const _Float16 **probability = pushz(a, (size_t)count * sizeof(*probability));
        int any = 0;
        for (int i = 0; i < count; i++) {
            SpotImagePiece p = pieces[i];
            const char *path = fmt(a, "%s/reads/%s/%s/%s.npy", work, surface, name, p.name);
            Bytes b = spot_map(path);
            if (b.data) {
                Array record = npy_parse(b, path);
                need(!strcmp(record.descr, "<f2") && record.ndim == 4 && record.shape[0] == 2 &&
                        record.shape[1] == SPOT_DEPTHS && record.shape[2] == p.h &&
                        record.shape[3] == p.w &&
                        record.size == (size_t)4 * SPOT_DEPTHS * p.columns,
                    "spot maps have the wrong shape at %s", path);
                mapped[held++] = b;
                probability[i] = record.data;
                any = 1;
            }
        }
        if (any) {
            models[nmodel++] = (SpotImageModel){.name = name, .folder = name, .piece = probability};
        }
    }
    if (nmodel == 2) {
        for (int k = 0; k < held; k++) {
            munmap(mapped[k].data, mapped[k].size);
        }
        return NULL;
    }
    SpotOutline outline = spot_outline_at(a, jget(spot, "outline_mm"), 0);
    double um = jf64(jget(json_object_array_get_idx(records, 0), "layout"), "um");
    need(um > 0 && isfinite(um), "invalid voxel size for the spot page");
    int x0 = (int)floor((outline.left - 3) / um), y0 = (int)floor((outline.top - 3) / um);
    int w = (int)ceil((outline.right + 3) / um) - x0;
    int h = (int)ceil((outline.bottom + 3) / um) - y0;
    need(h > 0 && w > 0 && h <= INT_MAX / w, "spot image dimensions are too large");
    int pixels = h * w;
    for (size_t i = 0; i < outline.count; i++) {
        outline.points[i].x = outline.points[i].x / um - x0;
        outline.points[i].y = outline.points[i].y / um - y0;
    }
    SpotOwner *owner = push(a, (size_t)pixels * sizeof(*owner));
    memset(owner, 0xff, (size_t)pixels * sizeof(*owner));
    qsort(order, count, sizeof(*order), spot_image_order);
    for (int j = 0; j < count; j++) {
        int i = order[j].id;
        spot_place_image(a, owner, h, w, x0, y0, pieces + i, i,
            json_object_array_get_idx(records, i), surface, work);
    }
    i64 samples = 0;
    int valid = 0, ninner = 0;
    size_t possible = ((size_t)h + SPOT_IMAGE_STEP - 1) / SPOT_IMAGE_STEP *
        (((size_t)w + SPOT_IMAGE_STEP - 1) / SPOT_IMAGE_STEP);
    int *sample = push(a, possible * sizeof(*sample));
    int *inner = push(a, possible * sizeof(*inner));
    for (int i = 0; i < pixels; i++) {
        if (owner[i].piece < 0) {
            continue;
        }
        valid++;
        int y = i / w, x = i % w;
        if (!(y % SPOT_IMAGE_STEP) && !(x % SPOT_IMAGE_STEP)) {
            sample[samples++] = i;
            if (spot_inside(&outline, x, y)) {
                inner[ninner++] = i;
            }
        }
    }
    need(samples > 0, "spot surface %s has no valid image tiles", surface);
    i64 total = samples * SPOT_DEPTHS;
    f32 *statistics = push(a, total * sizeof(*statistics));
    f32 *scratch = push(a, total * sizeof(*scratch));
    f32 *ensemble = pushz(a, total * sizeof(*ensemble));
    for (int m = 2; m < nmodel; m++) {
        for (int d = 0; d < SPOT_DEPTHS; d++) {
            for (i64 i = 0; i < samples; i++) {
                statistics[(i64)d * samples + i] =
                    spot_image_read(models + m, pieces, owner[sample[i]], d, 0);
            }
        }
        SpotStats scale = spot_image_stats(statistics, scratch, total, 0.5, 0.999);
        models[m].stats = scale;
        for (int d = 0; d < SPOT_DEPTHS; d++) {
            for (i64 i = 0; i < samples; i++) {
                f32 v = spot_image_read(models + m, pieces, owner[sample[i]], d, 0);
                ensemble[(i64)d * samples + i] += (v - (f32)scale.mean) / (f32)scale.sd;
            }
        }
    }
    for (i64 i = 0; i < total; i++) {
        statistics[i] = ensemble[i] /= nmodel - 2;
    }
    models[1].stats = spot_image_stats(statistics, scratch, total, 0.5, 0.999);
    for (int d = 0; d < SPOT_DEPTHS; d++) {
        int low = d > SPOT_IMAGE_BAND ? d - SPOT_IMAGE_BAND : 0;
        int high = d + SPOT_IMAGE_BAND + 1 < SPOT_DEPTHS ? d + SPOT_IMAGE_BAND + 1 : SPOT_DEPTHS;
        for (i64 i = 0; i < samples; i++) {
            f32 sum = 0;
            for (int k = low; k < high; k++) {
                sum += ensemble[(i64)k * samples + i];
            }
            statistics[(i64)d * samples + i] = sum / (high - low);
        }
    }
    models[0].stats = spot_image_stats(statistics, scratch, total, 0.5, 0.999);
    for (int d = 0; d < SPOT_DEPTHS; d++) {
        for (i64 i = 0; i < samples; i++) {
            SpotOwner at = owner[sample[i]];
            SpotImagePiece p = pieces[at.piece];
            statistics[(i64)d * samples + i] =
                p.volume[(i64)(SPOT_MIDDLE + d - SPOT_REACH) * p.columns + at.pixel];
        }
    }
    SpotStats ct = spot_image_stats(statistics, scratch, total, 0.01, 0.995);
    f32 *frame = push(a, (size_t)pixels * sizeof(*frame));
    f32 *average = push(a, (size_t)pixels * sizeof(*average));
    u8 *grey = push(a, pixels);
    const int band_count = 2 * SPOT_IMAGE_BAND + 1;
    _Float16 *ring = push(a, (size_t)band_count * pixels * sizeof(*ring));
    SpotImageSummary *summary = push(a, (size_t)nmodel * 2 * SPOT_DEPTHS * sizeof(*summary));
    for (int side = 0; side < 2; side++) {
        SpotImageSummary *row = summary + side * SPOT_DEPTHS * nmodel;
        for (int d = 0; d < SPOT_DEPTHS; d++) {
            memset(average, 0, (size_t)pixels * sizeof(*average));
            for (int m = 2; m < nmodel; m++) {
                SpotStats s = models[m].stats;
                for (int i = 0; i < pixels; i++) {
                    f32 value = spot_image_read(models + m, pieces, owner[i], d, side);
                    average[i] += frame[i] = (value - (f32)s.mean) / (f32)s.sd;
                }
                row[d * nmodel + m] = spot_jpeg(a, page, surface, models[m].folder, side, d, frame,
                    owner, inner, ninner, scratch, grey, h, w, (s.low - s.mean) / s.sd,
                    (s.high - s.mean) / s.sd, 1);
            }
            for (int i = 0; i < pixels; i++) {
                average[i] /= nmodel - 2;
                ring[(i64)(d % band_count) * pixels + i] = average[i];
            }
            SpotStats norm = models[1].stats;
            row[d * nmodel + 1] = spot_jpeg(a, page, surface, models[1].folder, side, d, average,
                owner, inner, ninner, scratch, grey, h, w, norm.low, norm.high, 1);
            int through = d + 1 == SPOT_DEPTHS ? d : d - SPOT_IMAGE_BAND;
            for (int mid = d - SPOT_IMAGE_BAND; mid >= 0 && mid <= through; mid++) {
                int low = mid > SPOT_IMAGE_BAND ? mid - SPOT_IMAGE_BAND : 0;
                int high = mid + SPOT_IMAGE_BAND + 1 < SPOT_DEPTHS ? mid + SPOT_IMAGE_BAND + 1
                                                                   : SPOT_DEPTHS;
                for (int i = 0; i < pixels; i++) {
                    f32 v = 0;
                    for (int k = low; k < high; k++) {
                        v += (f32)ring[(i64)(k % band_count) * pixels + i];
                    }
                    frame[i] = v / (high - low);
                }
                SpotStats band = models[0].stats;
                row[mid * nmodel] = spot_jpeg(a, page, surface, models[0].folder, side, mid, frame,
                    owner, inner, ninner, scratch, grey, h, w, band.low, band.high, 1);
            }
        }
    }
    for (int d = 0; d < SPOT_DEPTHS; d++) {
        for (int i = 0; i < pixels; i++) {
            SpotOwner at = owner[i];
            frame[i] = at.piece < 0
                ? NAN
                : pieces[at.piece]
                      .volume[(i64)(SPOT_MIDDLE + d - SPOT_REACH) * pieces[at.piece].columns +
                          at.pixel];
        }
        spot_jpeg(a, page, surface, "ct", 0, d, frame, owner, NULL, 0, scratch, grey, h, w, ct.low,
            ct.high, 0);
    }

    int best = 0;
    for (int d = 1; ninner && d < SPOT_DEPTHS; d++) {
        double value = summary[d * nmodel + 1].p97, old = summary[best * nmodel + 1].p97;
        if (!isnan(old) && (isnan(value) || value > old)) {
            best = d;
        }
    }
    for (int side = 0; side < 2; side++) {
        for (int d = 0; d < SPOT_DEPTHS; d++) {
            for (int m = 0; m < nmodel; m++) {
                SpotImageSummary s = summary[(side * SPOT_DEPTHS + d) * nmodel + m];
                SpotImageSummary other = summary[((1 - side) * SPOT_DEPTHS + d) * nmodel + m];
                json_object *r = json_object_new_object();
                json_object_object_add(r, "spot_id", json_object_get(jget(spot, "spot_id")));
                json_object_object_add(r, "surface", json_object_new_string(surface));
                json_object_object_add(r, "model", json_object_new_string(models[m].name));
                json_object_object_add(r, "side", json_object_new_string(spot_sides[side]));
                json_object_object_add(r, "depth", json_object_new_int(d - SPOT_REACH));
                json_object_object_add(r, "mean_z", ninner ? spot_round(s.mean, 10000) : NULL);
                json_object_object_add(r, "p95_z", ninner ? spot_round(s.p95, 10000) : NULL);
                json_object_object_add(r, "inky_share", ninner ? spot_round(s.inky, 100000) : NULL);
                json_object_object_add(
                    r, "other_side_mean_z", ninner ? spot_round(other.mean, 10000) : NULL);
                size_t mark = a->used;
                json_object_object_add(r, "image",
                    json_object_new_string(fmt(a, "%s/images/%s/%s/%s/d%+03d.jpg", page, surface,
                        models[m].folder, spot_sides[side], d - SPOT_REACH)));
                a->used = mark;
                json_object_array_add(reads, r);
            }
        }
    }
    json_object *report = json_object_new_object(), *layers = json_object_new_array();
    json_object *stats = json_object_new_object(), *line = json_object_new_array();
    for (int m = 0; m < nmodel; m++) {
        SpotStats s = models[m].stats;
        json_object *values = json_object_new_array_ext(4);
        json_object_array_add(layers, json_object_new_string(models[m].folder));
        json_object_array_add(values, spot_round(s.mean, 10000));
        json_object_array_add(values, spot_round(s.sd, 10000));
        json_object_array_add(values, spot_round(s.low, 10000));
        json_object_array_add(values, spot_round(s.high, 10000));
        json_object_object_add(stats, models[m].name, values);
    }
    json_object_array_add(layers, json_object_new_string("ct"));
    for (size_t i = 0; i < outline.count; i++) {
        json_object *point = json_object_new_array_ext(2);
        json_object_array_add(point, spot_round(outline.points[i].x, 10));
        json_object_array_add(point, spot_round(outline.points[i].y, 10));
        json_object_array_add(line, point);
    }
    json_object_object_add(report, "stacks", layers);
    json_object_object_add(report, "best_depth", json_object_new_int(best - SPOT_REACH));
    json_object_object_add(report, "stats", stats);
    json_object_object_add(report, "valid_share", spot_round(valid / (double)pixels, 10000));
    json_object_object_add(report, "within_cells", json_object_new_int(ninner));
    json_object_object_add(manifest, "width", json_object_new_int(w));
    json_object_object_add(manifest, "height", json_object_new_int(h));
    json_object_object_add(manifest, "outline", line);
    for (int k = 0; k < held; k++) {
        munmap(mapped[k].data, mapped[k].size);
    }
    printf("%s %s images %d stacks\n", jstr(spot, "spot_id"), surface, nmodel);
    return report;
}

static const char spot_page[] =
    "<!doctype html>\n"
    "<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,ini"
    "tial-scale=1\">\n"
    "<title>Spot depth sweep</title>\n"
    "<style>\n"
    ":root{--bg:#111;--fg:#ddd;--muted:#888;--accent:#39ff14}\n"
    "body{background:var(--bg);color:var(--fg);font:13px/1.4 system-ui;margin:0}\n"
    "header{position:sticky;top:0;background:#181818;padding:8px 12px;display:flex;flex-wrap:wrap"
    ";gap:8px 16px;align-items:center;z-index:2}\n"
    "label{color:var(--muted)}select,button{background:#222;color:var(--fg);border:1px solid #444"
    ";padding:2px 6px}\n"
    "#depth{width:240px}#d{font:600 15px ui-monospace,monospace;min-width:3ch;display:inline-bloc"
    "k}\n"
    "#wrap{position:relative;display:inline-block;margin:8px;transform-origin:center}#view{displa"
    "y:block;max-width:none}\n"
    "#ol{position:absolute;inset:0;pointer-events:none}\n"
    ".inv #view{filter:invert(1)}#info{color:var(--muted);padding:0 12px;white-space:pre-wrap;fon"
    "t:12px ui-monospace,monospace}\n"
    "</style></head><body>\n"
    "<header>\n"
    "<b id=\"title\"></b>\n"
    "<label>surface <select id=\"surface\"></select></label>\n"
    "<label>stack <select id=\"stack\"></select></label>\n"
    "<label>side <select id=\"side\"></select></label>\n"
    "<label>depth <input id=\"depth\" type=\"range\" min=\"-15\" max=\"15\" value=\"0\"> <span id"
    "=\"d\">0</span></label>\n"
    "<label>zoom <select id=\"zoom\"><option>0.25</option><option selected>0.5</option><option>1<"
    "/option><option>2</option></select></label>\n"
    "<button id=\"invert\">invert</button><button id=\"circle\">circle</button>\n"
    "<button id=\"mirror\">mirror</button><button id=\"turn\">turn 180°</button>\n"
    "<span style=\"color:var(--muted)\">← → depth · ↑ ↓ stack</span>\n"
    "</header>\n"
    "<div id=\"info\"></div>\n"
    "<div id=\"wrap\"><img id=\"view\" alt=\"\"><svg id=\"ol\"></svg></div>\n"
    "<script>\n"
    "const $=id=>document.getElementById(id);\n"
    "let m,showCircle=true;\n"
    "fetch('manifest.json').then(r=>r.json()).then(x=>{m=x;\n"
    " $('title').textContent=`${m.spot.spot_id} · ${m.spot.scroll} · ${m.spot.page} · ${m.spot.ar"
    "ea_cm2} cm²`;\n"
    " for(const s of Object.keys(m.surfaces)) $('surface').add(new Option(s));\n"
    " for(const s of m.sides) $('side').add(new Option(s));\n"
    " $('info').textContent='pieces: '+m.pieces.map(p=>`${p.name} share ${p.share} winding ${p.wi"
    "nding} production reads ${p.reading_side||'?'} · snap median ${p.snap.median_move_vox} vox`)"
    ".join('\\n');\n"
    " fillStacks();const b=m.surfaces[$('surface').value].best_depth;if(b!==undefined)$('depth')."
    "value=b;show();});\n"
    "function fillStacks(){const s=$('stack'),keep=s.value;s.innerHTML='';\n"
    " for(const k of m.surfaces[$('surface').value].stacks) s.add(new Option(k));\n"
    " if([...s.options].some(o=>o.value===keep)) s.value=keep;}\n"
    "function show(){const d=+$('depth').value,stack=$('stack').value,side=stack==='ct'?'inward':"
    "$('side').value;\n"
    " $('d').textContent=(d>0?'+':'')+d;\n"
    " $('view').src=`images/${$('surface').value}/${stack}/${side}/d${d<0?'-':'+'}${String(Math.a"
    "bs(d)).padStart(2,'0')}.jpg`;\n"
    " const z=+$('zoom').value;$('view').style.width=(m.width*z)+'px';\n"
    " const svg=$('ol');svg.setAttribute('viewBox',`0 0 ${m.width} ${m.height}`);svg.style.width="
    "(m.width*z)+'px';svg.style.height=(m.height*z)+'px';\n"
    " svg.innerHTML=showCircle?`<polygon points=\"${m.outline.map(p=>p.join(',')).join(' ')}\" fi"
    "ll=\"none\" stroke=\"#39ff14\" stroke-width=\"${3/z}\"/>`:'';}\n"
    "for(const id of ['surface']) $(id).onchange=()=>{fillStacks();show();};\n"
    "for(const id of ['stack','side','depth','zoom']) $(id).oninput=show;\n"
    "$('invert').onclick=()=>document.body.classList.toggle('inv');\n"
    "let mirror=false,turn=false;\n"
    "function orient(){$('wrap').style.transform=`${turn?'rotate(180deg) ':''}${mirror?'scaleX(-1"
    ")':''}`;\n"
    " try{localStorage.setItem('spot-orient',JSON.stringify({mirror,turn}));}catch(e){}}\n"
    "try{const o=JSON.parse(localStorage.getItem('spot-orient')||'{}');mirror=!!o.mirror;turn=!!o"
    ".turn;}catch(e){}\n"
    "$('mirror').onclick=()=>{mirror=!mirror;orient();};$('turn').onclick=()=>{turn=!turn;orient("
    ");};orient();\n"
    "$('circle').onclick=()=>{showCircle=!showCircle;show();};\n"
    "addEventListener('keydown',e=>{const r=$('depth'),s=$('stack');\n"
    " if(e.key==='ArrowRight'){r.value=Math.min(15,+r.value+1);show();e.preventDefault();}\n"
    " if(e.key==='ArrowLeft'){r.value=Math.max(-15,+r.value-1);show();e.preventDefault();}\n"
    " if(e.key==='ArrowDown'){s.selectedIndex=(s.selectedIndex+1)%s.options.length;show();e.preve"
    "ntDefault();}\n"
    " if(e.key==='ArrowUp'){s.selectedIndex=(s.selectedIndex+s.options.length-1)%s.options.length"
    ";show();e.preventDefault();}});\n"
    "</script></body></html>\n";

static int spot_index_order(const void *one, const void *two) {
    json_object *a = *(json_object *const *)one, *b = *(json_object *const *)two;
    const char *keys[] = {"scroll", "page", "place", "spot_id"};
    for (int i = 0; i < 4; i++) {
        int order = strcmp(jstr(a, keys[i]), jstr(b, keys[i]));
        if (order) {
            return order;
        }
    }
    return 0;
}

static void spot_images_index(Arena *a, const char *root) {
    size_t saved = a->used;
    json_object *catalogue = spot_rows(a, "spots"), *verdicts = spot_rows(a, "verdicts");
    size_t found = site_items(catalogue), judged = site_items(verdicts), count = 0;
    json_object **spots = push(a, found * sizeof(*spots));
    json_object *places = json_object_new_object();
    for (size_t i = 0; i < found; i++) {
        json_object *row = json_object_array_get_idx(catalogue, i);
        if (json_object_get_boolean(jget(row, "target"))) {
            spots[count++] = row;
            json_object_object_add(places, jstr(row, "place"), NULL);
        }
    }
    qsort(spots, count, sizeof(*spots), spot_index_order);
    const char *index = fmt(a, "%s/index.html", root);
    char *part = fmt(a, "%s.part.XXXXXXXX", index);
    mkparent(part);
    int fd = mkstemp(part);
    FILE *file = fd >= 0 ? fdopen(fd, "w") : NULL;
    need(file != NULL, "cannot write %s", index);
    fputs(
        "<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Spot candidates</title><style>body{background:#111;color:#ddd;font:13px system-ui;"
        "margin:16px}a{color:#7cf}table{border-collapse:collapse}"
        "td,th{padding:2px 8px;border-bottom:1px solid #333;text-align:left}"
        "th{position:sticky;top:0;background:#181818}</style></head><body>",
        file);
    fprintf(file, "<h1>Circled spots: %zu circles, %d places</h1>", count,
        json_object_object_length(places));
    fputs(
        "<p>A linked id has its surface rebuilt on m7 and read at every depth from -15 to +15.</p>"
        "<table><tr><th>spot</th><th>scroll</th><th>sheet</th><th>place</th><th>cm²</th>"
        "<th>covered</th><th>pieces</th><th>windings</th><th>main piece</th><th>status</th>"
        "<th>verdicts</th></tr>",
        file);
    for (size_t i = 0; i < count; i++) {
        size_t mark = a->used;
        json_object *row = spots[i];
        const char *id = jstr(row, "spot_id"), *status = site_string(row, "status");
        need(site_name(id), "invalid spot ID in index");
        fputs("<tr><td>", file);
        if (exists(fmt(a, "%s/%s/manifest.json", root, id))) {
            fprintf(file, "<a href=\"%s/index.html\">%s</a>", id, id);
        } else {
            fputs(id, file);
        }
        fprintf(file,
            "</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%lld</td><td>%zu</td><td>%s</td><td>%s</td><td>",
            site_html_escape(a, jstr(row, "scroll")), site_html_escape(a, jstr(row, "page")),
            site_html_escape(a, jstr(row, "place")), jstr(row, "area_cm2"),
            jstr(row, "covered_share"), (long long)ji64(row, "n_pieces"),
            site_items(jget(row, "windings")), jstr(row, "main_share"),
            site_html_escape(a, status ? status : ""));
        int seen = 0;
        for (size_t j = 0; j < judged; j++) {
            json_object *verdict = json_object_array_get_idx(verdicts, j);
            if (!strcmp(id, jstr(verdict, "spot_id"))) {
                fprintf(file, "%s%s: %s %lld", seen++ ? "; " : "",
                    site_html_escape(a, jstr(verdict, "by")), jstr(verdict, "readable"),
                    (long long)ji64(verdict, "letters"));
            }
        }
        fputs("</td></tr>\n", file);
        a->used = mark;
    }
    fputs("</table></body></html>", file);
    int ok = !ferror(file);
    need(!fclose(file) && ok && !rename(part, index), "cannot publish %s", index);
    json_object_put(places);
    json_object_put(catalogue);
    json_object_put(verdicts);
    a->used = saved;
}

static void spot_images(Arena *a, json_object *spot, json_object *document) {
    const char *work = spot_work(a, spot), *id = jstr(spot, "spot_id");
    const char *root = fmt(a, "%s/website/sheets/spot-candidates", OUTPUTS);
    const char *page = fmt(a, "%s/%s", root, id);
    json_object *records = jget(document, "pieces");
    json_object *recipe = json_read(fmt(a, "%s/reads/recipe.json", work));
    need(recipe != NULL, "the spot has no read recipe; run ./run.sh spots.pilot %s read", id);
    json_object *read_by = jget(recipe, "read_by");
    json_object *manifest = json_object_new_object(), *surfaces = json_object_new_object();
    json_object *reads = json_object_new_array(), *details = json_object_new_object();
    json_object_object_add(manifest, "surfaces", surfaces);
    for (int i = 0; i < 3; i++) {
        size_t mark = a->used;
        const char *name = spot_surfaces[i];
        json_object *layer = spot_image_surface(
            a, spot, records, work, page, name, jget(read_by, name), manifest, reads);
        if (layer) {
            json_object_object_add(surfaces, name, layer);
        }
        a->used = mark;
    }
    const char *keys[] = {
        "spot_id", "scroll", "scan", "route", "page", "place", "area_cm2", "x_mm", "y_mm"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) {
        json_object_object_add(details, keys[i], json_object_get(jget(spot, keys[i])));
    }
    json_object *pieces = json_object_new_array();
    const char *fields[] = {"name", "share", "winding", "reading_side", "snap"};
    for (size_t i = 0; i < site_items(records); i++) {
        json_object *r = json_object_array_get_idx(records, i), *shorter = json_object_new_object();
        for (size_t k = 0; k < sizeof(fields) / sizeof(*fields); k++) {
            json_object_object_add(shorter, fields[k], json_object_get(jget(r, fields[k])));
        }
        json_object_array_add(pieces, shorter);
    }
    json_object_object_add(manifest, "spot", details);
    json_object_object_add(manifest, "pieces", pieces);
    json_object_object_add(manifest, "depths", json_object_get(jget(recipe, "depths")));
    json_object_object_add(manifest, "sides", json_object_get(jget(recipe, "sides")));
    json_write(fmt(a, "%s/manifest.json", page), manifest);
    write_file(fmt(a, "%s/index.html", page), spot_page, sizeof(spot_page) - 1);
    spot_write(a, "reads", reads);
    spot_images_index(a, root);
    json_object_put(manifest);
    json_object_put(reads);
    json_object_put(recipe);
}
