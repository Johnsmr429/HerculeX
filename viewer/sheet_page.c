#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "page_template.h"
#include "sheet_page.h"

typedef struct {
    double left, right, area;
} Strip;

typedef struct {
    double left, top, right, bottom, area;
    int slot;
} Place;

static double number(json_object *entry, const char *field) {
    return json_object_get_double(json_object_object_get(entry, field));
}

static double tenth(double value) {
    char text[320];
    snprintf(text, sizeof(text), "%.1f", value);
    return strtod(text, NULL);
}

static json_object *rectangle(const double values[4]) {
    const char *names[] = {"x", "y", "w", "h"};
    json_object *box = json_object_new_object();
    for (int i = 0; box && i < 4; i++) {
        json_object_object_add(box, names[i], json_object_new_double(values[i]));
    }
    return box;
}

static int bigger(const void *a, const void *b) {
    const Place *left = a, *right = b;
    if (left->area != right->area) {
        return left->area > right->area ? -1 : 1;
    }
    return left->slot - right->slot;
}

static int read_layout(
    json_object *rows, Strip *strips, Place *places, size_t count, double *width, double *height) {
    json_object *groups = json_object_new_object();
    int used = 0;
    for (size_t i = 0; groups && i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        double x = number(row, "x_mm"), y = number(row, "y_mm"), w = number(row, "width_mm"),
               h = number(row, "height_mm"), area = number(row, "area");
        int slot = hx_mosaic_slot(groups, row);
        if (!slot || !(w >= 0 && h >= 0 && area >= 0) || !isfinite(x + y + w + h + area)) {
            used = 0;
            break;
        }
        strips[i] = (Strip){x, x + w, area};
        *width = fmax(*width, x + w);
        *height = fmax(*height, y + h);
        Place *place = places + slot - 1;
        if (slot > used) {
            *place = (Place){x, y, x + w, y + h, area, slot};
            used = slot;
        } else {
            place->left = fmin(place->left, x);
            place->top = fmin(place->top, y);
            place->right = fmax(place->right, x + w);
            place->bottom = fmax(place->bottom, y + h);
            place->area += area;
        }
    }
    json_object_put(groups);
    return used;
}

static double cut_sum(const double *areas, size_t count) {
    if (count > 128) {
        size_t half = (count / 2) & ~(size_t)7;
        return cut_sum(areas, half) + cut_sum(areas + half, count - half);
    }
    double total = 0;
    size_t rest = count & ~(size_t)7;
    if (rest) {
        double lanes[8] = {0};
        for (size_t i = 0; i < rest; i += 8) {
            for (size_t k = 0; k < 8; k++) {
                lanes[k] += areas[i + k];
            }
        }
        total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
            ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    }
    for (size_t i = rest; i < count; i++) {
        total += areas[i];
    }
    return total;
}

static double shelf_edge(
    const Strip *strips, size_t count, double *areas, double length, int edge, int shelves) {
    double winner = 0, least = INFINITY, step = (1.05 - .95) / 100;
    for (int i = 0; i <= 100; i++) {
        double factor = i == 100 ? 1.05 : .95 + i * step;
        double x = factor * edge * length / shelves;
        size_t crossings = 0;
        for (size_t j = 0; j < count; j++) {
            if (strips[j].left < x && strips[j].right > x) {
                areas[crossings++] = strips[j].area;
            }
        }
        double area = cut_sum(areas, crossings);
        if (area < least) {
            least = area;
            winner = x;
        }
        if (least == 0) {
            break;
        }
    }
    return winner;
}

static json_object *shelves_of(
    const Strip *strips, size_t count, double length, double height, double extent[4]) {
    double wanted = nearbyint(sqrt(length / (1.9 * (height + 20))));
    if (!(wanted >= 0 && wanted < INT_MAX)) {
        return NULL;
    }
    int shelves = wanted > 1 ? (int)wanted : 1;
    json_object *rows = json_object_new_array_ext(shelves);
    double *areas = shelves > 1 ? malloc(count * sizeof(*areas)) : NULL;
    if (!rows || (shelves > 1 && !areas)) {
        json_object_put(rows);
        free(areas);
        return NULL;
    }
    double left = 0;
    for (int i = 0; i < shelves; i++) {
        double right =
            i + 1 == shelves ? length : shelf_edge(strips, count, areas, length, i + 1, shelves);
        double row[] = {tenth(left), tenth(i * (height + 20)), tenth(right - left), tenth(height)};
        json_object_array_add(rows, rectangle(row));
        extent[2] = fmax(extent[2], row[2]);
        extent[3] = row[1] + row[3];
        left = right;
    }
    free(areas);
    return rows;
}

static json_object *places_of(Place *places, int used, double thumbnail[4]) {
    int most = used < 200 ? used : 200;
    json_object *rows = json_object_new_array_ext(most);
    qsort(places, used, sizeof(*places), bigger);
    for (int i = 0; rows && i < most; i++) {
        Place p = places[i];
        double box[] = {
            tenth(p.left), tenth(p.top), tenth(p.right - p.left), tenth(p.bottom - p.top)};
        json_object_array_add(rows, rectangle(box));
        if (!i) {
            memcpy(thumbnail, box, sizeof(box));
        }
    }
    return rows;
}

json_object *hx_sheet_metadata(json_object *rows, json_object *visible, json_object *layers,
    const char *title, const char *scan, const double *fixed, json_object *boxes,
    json_object *suggested) {
    size_t count = json_object_is_type(rows, json_type_array) ? json_object_array_length(rows) : 0;
    size_t seen =
        json_object_is_type(visible, json_type_array) ? json_object_array_length(visible) : 0;
    size_t most = count > seen ? count : seen;
    Strip *strips = most && most <= INT_MAX ? calloc(most, sizeof(*strips)) : NULL;
    Place *places = strips ? calloc(most, sizeof(*places)) : NULL;
    double width = -INFINITY, height = -INFINITY, area[4] = {0}, thumb[4] = {0};
    int used = !fixed && places ? read_layout(rows, strips, places, count, &width, &height) : 0;
    json_object *shelves = used ? shelves_of(strips, count, width, height, area) : NULL;
    if (fixed) {
        memcpy(area, fixed, sizeof(area));
        shelves = json_object_new_array();
    }
    if (places && visible && visible != rows) {
        used = read_layout(visible, strips, places, seen, &width, &height);
    } else if (!visible) {
        used = 0;
    }
    json_object *jumps = used ? places_of(places, used, thumb) : json_object_new_array();
    json_object *meta = shelves && jumps && json_object_is_type(layers, json_type_array)
        ? json_object_new_object()
        : NULL;
    if (meta) {
        json_object_object_add(meta, "title", json_object_new_string(title));
        json_object_object_add(meta, "layers", json_object_get(layers));
        json_object_object_add(meta, "tile", json_object_new_int(512));
        json_object_object_add(meta, "area", rectangle(area));
        json_object_object_add(meta, "back", json_object_new_string("../index.html"));
        json_object_object_add(meta, "scan", json_object_new_string(scan));
        json_object_object_add(
            meta, "boxes", boxes ? json_object_get(boxes) : json_object_new_array());
        json_object_object_add(meta, "places", json_object_get(jumps));
        json_object_object_add(meta, "shelves", json_object_get(shelves));
        json_object_object_add(
            meta, "suggested", suggested ? json_object_get(suggested) : json_object_new_array());
        json_object_object_add(meta, "tint", json_object_new_string(fixed ? "" : "purple"));
    }
    json_object_put(jumps);
    json_object_put(shelves);
    free(places);
    free(strips);
    return meta;
}

int hx_sheet_render(json_object *meta, const char *folder) {
    char *path = NULL, *html = NULL;
    int named = meta && asprintf(&path, "%s/index.html", folder) >= 0;
    char *page = named ? read_page_text(viewer_template) : NULL;
    size_t bytes = 0;
    FILE *out = page ? open_memstream(&html, &bytes) : NULL;
    int ok = 0;
    if (out) {
        json_object *title = json_object_object_get(meta, "title");
        const char *values[] = {json_object_get_string(title),
            json_object_to_json_string_ext(
                meta, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE),
            "160", "0.3"};
        int rendered = render_viewer(out, page, values);
        ok = !fclose(out) && rendered && hx_write_sheet_file(path, html, bytes);
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: cannot publish the review page in %s\n", folder);
    }
    free(html);
    free(page);
    free(path);
    return ok;
}

int hx_sheet_page(json_object *rows, json_object *layers, const char *folder, const char *title,
    const char *scan, int card_level) {
    json_object *meta = hx_sheet_metadata(rows, rows, layers, title, scan, NULL, NULL, NULL);
    char *path = NULL;
    int ok = meta != NULL;
    if (ok && card_level >= 0) {
        json_object *jumps = json_object_object_get(meta, "places");
        json_object *first = json_object_array_get_idx(jumps, 0);
        double extent[] = {
            number(first, "x"), number(first, "y"), number(first, "w"), number(first, "h")};
        ok = first && asprintf(&path, "%s/card.jpg", folder) >= 0 &&
            hx_sheet_card(folder, card_level, extent, path);
    }
    ok = ok && hx_sheet_render(meta, folder);
    json_object_put(meta);
    free(path);
    return ok;
}

int hx_sheet_review(json_object *rows, json_object *layers, const char *folder, const char *sample,
    const char *volume, const char *surface) {
    const char *inks[] = {"taught", "new-ink", "ink"}, *model = "";
    time_t stamp = 0;
    for (size_t i = 0; i < sizeof(inks) / sizeof(*inks); i++) {
        int found = 0;
        for (size_t j = 0; j < json_object_array_length(layers); j++) {
            json_object *layer = json_object_array_get_idx(layers, j);
            const char *path = json_object_get_string(json_object_object_get(layer, "path"));
            if (path && strcmp(path, inks[i]) == 0) {
                const char *name = json_object_get_string(json_object_object_get(layer, "model"));
                model = name ? name : "";
                stamp = json_object_get_int64(json_object_object_get(layer, "stamp"));
                found = 1;
                break;
            }
        }
        if (found) {
            break;
        }
    }
    struct tm utc;
    char date[64] = {0};
    if (stamp && gmtime_r(&stamp, &utc)) {
        strftime(date, sizeof(date), "%-d %B %Y", &utc);
    }
    char *title = NULL, *scan = NULL;
    int ok = asprintf(&title, "%s%s%s · %s%s%s", sample, *model ? " · " : "", model, surface,
                 *date ? " · " : "", date) >= 0 &&
        asprintf(&scan, "level/%s/%s", sample, volume) >= 0 &&
        hx_sheet_page(rows, layers, folder, title, scan, 3);
    free(scan);
    free(title);
    return ok;
}
