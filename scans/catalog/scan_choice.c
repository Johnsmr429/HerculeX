#define hx_usage scan_catalog_usage
#define main scan_catalog_command
#include "scan_catalog.c"
#undef main
#undef hx_usage

#include <math.h>

const char hx_scan_choice_usage[] = "Choose the survey CT scan for each scroll.";

typedef struct {
    json_object *row;
    double voxel;
    double energy;
} Candidate;

static int requested(const char *sample, char **samples, int count) {
    if (!count)
        return 1;
    for (int i = 0; i < count; i++) {
        if (!strcmp(sample, samples[i]))
            return 1;
    }
    return 0;
}

static int product(json_object *row, const char *name) {
    json_object *products, *value;
    return json_object_object_get_ex(row, "products", &products) &&
        json_object_object_get_ex(products, name, &value) && value &&
        !json_object_is_type(value, json_type_null);
}

static const char *text_field(json_object *row, const char *name) {
    json_object *value;
    return json_object_object_get_ex(row, name, &value) ? json_object_get_string(value) : NULL;
}

static double number_field(json_object *row, const char *name, double fallback) {
    json_object *value;
    return json_object_object_get_ex(row, name, &value) &&
            !json_object_is_type(value, json_type_null)
        ? json_object_get_double(value)
        : fallback;
}

static int fine(double voxel) { return voxel >= 2 && voxel <= 2.6; }

static double height_of(json_object *row) {
    json_object *products, *roots, *path_value, *root_value;
    if (!json_object_object_get_ex(row, "products", &products) ||
        !json_object_object_get_ex(row, "roots", &roots) ||
        !json_object_object_get_ex(products, "ome-zarr", &path_value) ||
        !json_object_object_get_ex(roots, "ome-zarr", &root_value))
        return NAN;
    const char *path = json_object_get_string(path_value);
    const char *root = json_object_get_string(root_value);
    while (*path == '/')
        path++;
    size_t n = strlen(path);
    while (n && path[n - 1] == '/')
        n--;
    char *url = NULL;
    int made = asprintf(&url, "%s/%.*s/0/.zarray", root, (int)n, path) >= 0;
    FILE *payload = made ? tmpfile() : NULL;
    CURL *curl = payload ? curl_easy_init() : NULL;
    int ok = curl != NULL;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, payload);
        ok = curl_easy_perform(curl) == CURLE_OK;
        curl_easy_cleanup(curl);
    }
    ok = ok && fseek(payload, 0, SEEK_SET) == 0;
    int fd = ok ? dup(fileno(payload)) : -1;
    json_object *document = fd >= 0 ? json_object_from_fd(fd) : NULL;
    if (fd >= 0)
        close(fd);
    if (payload)
        fclose(payload);
    free(url);
    json_object *shape;
    double height = NAN;
    if (document && json_object_object_get_ex(document, "shape", &shape) &&
        json_object_array_length(shape))
        height = json_object_get_double(json_object_array_get_idx(shape, 0)) *
            number_field(row, "voxel_um", NAN);
    if (document)
        json_object_put(document);
    return height;
}

static json_object *choose(Candidate *rows, size_t count) {
    if (count == 1)
        return rows[0].row;
    double band = INFINITY;
    for (size_t i = 0; i < count; i++) {
        if (fine(rows[i].voxel))
            band = fmin(band, nearbyint(rows[i].energy / 10));
    }
    int use_fine = isfinite(band);
    json_object *winner = NULL;
    double best = use_fine ? -INFINITY : INFINITY;
    const char *best_id = NULL;
    for (size_t i = 0; i < count; i++) {
        if (use_fine && (!fine(rows[i].voxel) || nearbyint(rows[i].energy / 10) != band))
            continue;
        double metric = use_fine ? height_of(rows[i].row) : rows[i].voxel;
        const char *id = text_field(rows[i].row, "volume");
        int better = !winner || (use_fine ? metric > best : metric < best) ||
            (metric == best && (use_fine ? strcmp(id, best_id) > 0 : strcmp(id, best_id) < 0));
        if (better) {
            winner = rows[i].row;
            best = metric;
            best_id = id;
        }
    }
    return winner;
}

static int contains_id(const char *values, const char *sample) {
    const char *short_name = !strncmp(sample, "PHerc", 5) ? sample + 5 : sample;
    size_t n = strlen(short_name);
    const char *at = values;
    while ((at = strstr(at, short_name))) {
        int left = at == values || at[-1] == ' ';
        int right = at[n] == 0 || at[n] == ' ';
        if (left && right)
            return 1;
        at += n;
    }
    return 0;
}

static int prize_order(const void *left, const void *right) {
    json_object *a = *(json_object *const *)left;
    json_object *b = *(json_object *const *)right;
    const char *as = text_field(a, "sample"), *bs = text_field(b, "sample");
    const char *grand = "0125 0191 0211 0257 0268 0358 0800 0813 0826 1203 1218 1447 1545";
    const char *letters = "0125 0175A 0175B 0191 0211 0257 0268 0358 0483A 0483B 0490A "
                          "0490B 0800 0813 0826 0846A 0846B 1203 1218 1545";
    int ag = contains_id(grand, as), bg = contains_id(grand, bs);
    if (ag != bg)
        return bg - ag;
    int al = contains_id(letters, as), bl = contains_id(letters, bs);
    return al != bl ? bl - al : strcmp(as, bs);
}

static int choose_rows(json_object *rows, char **samples, int count, int json) {
    size_t total = json_object_array_length(rows);
    json_object **chosen = calloc(total ? total : 1, sizeof(*chosen));
    Candidate *group = calloc(total ? total : 1, sizeof(*group));
    if (!chosen || !group) {
        free(chosen);
        free(group);
        return 0;
    }
    size_t used = 0, group_count = 0;
    const char *held = NULL;
    for (size_t i = 0; i <= total; i++) {
        json_object *row = i < total ? json_object_array_get_idx(rows, i) : NULL;
        const char *sample = row ? text_field(row, "sample") : NULL;
        if (held && (!sample || strcmp(sample, held) != 0)) {
            json_object *winner = choose(group, group_count);
            if (winner)
                chosen[used++] = winner;
            group_count = 0;
        }
        if (!row)
            break;
        held = sample;
        const char *type = text_field(row, "type");
        double voxel = number_field(row, "voxel_um", NAN);
        const char *volume = text_field(row, "volume");
        if (!sample || !volume || !requested(sample, samples, count) ||
            (type && !strcmp(type, "fragment")) || !product(row, "ome-zarr") || !isfinite(voxel) ||
            voxel > 12 || !strcmp(volume, "20241024131838"))
            continue;
        group[group_count++] =
            (Candidate){.row = row, .voxel = voxel, .energy = number_field(row, "energy_kev", 999)};
    }
    qsort(chosen, used, sizeof(*chosen), prize_order);
    if (json) {
        json_object *result = json_object_new_array();
        for (size_t i = 0; result && i < used; i++)
            json_object_array_add(result, json_object_get(chosen[i]));
        if (result) {
            puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PRETTY));
            json_object_put(result);
        }
    } else {
        for (size_t i = 0; i < used; i++)
            printf("%s %s\n", text_field(chosen[i], "sample"), text_field(chosen[i], "volume"));
    }
    free(group);
    free(chosen);
    return used > 0 || count == 0;
}

static json_object *volume_rows(const char **error) {
    json_object *document = cached_catalog(NULL);
    json_object *samples = json_object_object_get(document, "samples");
    json_object *rows = samples ? catalog_rows(samples, 0, error) : NULL;
    json_object_put(document);
    return rows;
}

static int report_volume(json_object *rows, const char *sample, const char *volume, int json) {
    json_object *result = json_object_new_array();
    size_t count = rows ? json_object_array_length(rows) : 0;
    for (size_t i = 0; result && i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *name = text_field(row, "sample"), *id = text_field(row, "volume");
        if (name && id && !strcmp(name, sample) && !strcmp(id, volume) && product(row, "ome-zarr"))
            json_object_array_add(result, json_object_get(row));
    }
    int ok = result && json_object_array_length(result) == 1;
    if (ok && json)
        puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PRETTY));
    else if (ok)
        printf("%s %s\n", sample, volume);
    json_object_put(result);
    return ok;
}

static int scan_query(int argc, char **argv, int exact) {
    int json = argc > 0 && !strcmp(argv[0], "--json");
    argv += json;
    argc -= json;
    if (exact && argc != 2)
        return 2;
    const char *error = NULL;
    json_object *rows = volume_rows(&error);
    int ok = exact ? report_volume(rows, argv[0], argv[1], json)
                   : rows && choose_rows(rows, argv, argc, json);
    if (!ok) {
        const char *separator = error ? ": " : "";
        const char *detail = error ? error : "";
        if (exact)
            fprintf(stderr, "HerculeX: no survey scan %s %s%s%s\n", argv[0], argv[1], separator,
                detail);
        else
            fprintf(stderr, "HerculeX: could not choose scans%s%s\n", separator, detail);
    }
    json_object_put(rows);
    return ok ? 0 : 1;
}

static int scan_choice_main(int argc, char **argv) {
    if (argc >= 2 && (!strcmp(argv[1], "chosen") || !strcmp(argv[1], "volume")))
        return scan_query(argc - 2, argv + 2, !strcmp(argv[1], "volume"));
    if (argc == 3 && !strcmp(argv[1], "sheet-level")) {
        double voxel = strtod(argv[2], NULL);
        if (voxel <= 0)
            return 2;
        double level = nearbyint(log2(18.7 / voxel));
        printf("%.0f\n", fmax(0, level));
        return 0;
    }
    puts("Usage: ./run.sh scans.catalog.scan_choice chosen [--json] [SCROLL ...]\n"
         "       ./run.sh scans.catalog.scan_choice volume [--json] SCROLL VOLUME\n"
         "       ./run.sh scans.catalog.scan_choice sheet-level VOXEL_UM");
    return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
}

#ifdef HX_UNITY
#include "inventory.c"
#include "../transfer/mirror.c"

int hx_scan_choice_main(int argc, char **argv) { return scan_choice_main(argc, argv); }

int hx_scan_catalog_main(int argc, char **argv) {
    return argc > 1 && !strcmp(argv[0], "catalog") ? scan_query(argc - 1, argv + 1, 0)
                                                   : scan_catalog_command();
}
#else
int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    return !strcmp(argv[1], "scans.catalog.scan_catalog") ? scan_catalog_command()
                                                          : scan_choice_main(argc - 1, argv + 1);
}
#endif
