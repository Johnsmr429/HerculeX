#include "../base.h"

static char fixture[] = "/tmp/hx-atlas.XXXXXX";

static void clean(void) { remove_tree(fixture); }

static void execute(char *binary, char *operation) {
    pid_t pid = fork();
    need(pid >= 0, "could not launch atlas check");
    if (!pid) {
        execl(binary, binary, "viewer.atlas_build", operation, (char *)NULL);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "atlas %s did not complete", operation);
}

static void text(const char *path, const char *value) { write_file(path, value, strlen(value)); }

static void atlas_scan(const char *path) {
    mkparent(path);
    FILE *file = fopen(path, "wb");
    need(file != NULL, "cannot create atlas JPEG");
    struct jpeg_error_mgr error;
    struct jpeg_compress_struct out = {.err = jpeg_std_error(&error)};
    jpeg_create_compress(&out);
    jpeg_stdio_dest(&out, file);
    out.image_width = 103;
    out.image_height = 2713;
    out.input_components = 1;
    out.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&out);
    jpeg_set_quality(&out, 89, 1);
    jpeg_start_compress(&out, 1);
    for (int y = 0; y < 2713; y++) {
        u8 row[103];
        for (int x = 0; x < 103; x++) {
            row[x] = y % 36 < 8 && (x + 3 * y) % 17 < 11 ? 220 : 70 + (x + y) % 39;
        }
        JSAMPROW source = row;
        jpeg_write_scanlines(&out, &source, 1);
    }
    jpeg_finish_compress(&out);
    jpeg_destroy_compress(&out);
    fclose(file);
}

int main(int argc, char **argv) {
    need(argc == 2 && mkdtemp(fixture) != NULL, "expected a native binary and a temporary folder");
    atexit(clean);
    Arena a = arena(16 << 20);
    const char *library = fmt(&a, "%s/library", fixture), *output = fmt(&a, "%s/outputs", fixture);
    const char *work = fmt(&a, "%s/atlas-work", fixture);
    setenv("HERCULEX_LIBRARY", library, 1);
    setenv("HERCULEX_OUTPUTS", output, 1);
    setenv("HERCULEX_ATLAS_WORK", work, 1);
    unsetenv("HERCULEX_ATLAS_SECONDS");
    const char *root = fmt(&a, "%s/datasets/cases", library),
               *folder = fmt(&a, "%s/Frag1-3.24um", root);
    text(fmt(&a, "%s/manifest.jsonl", root),
        "{\"case_id\":\"Frag1-3.24um\",\"sample\":\"PHerc0001\",\"segment\":\"Frag1\","
        "\"scan\":\"9um-77keV\",\"box\":[0,120,0,121],\"voxel_um\":9,\"truth\":\"infrared\","
        "\"supervised_pixels\":3856}"
        "\n");
    text(fmt(&a, "%s/registration.json", folder), "{\"offset_um\":[18,-36]}");
    int width = 121, height = 120;
    size_t packed_bytes = ((size_t)width * height + 7) / 8;
    u8 *packed = pushz(&a, packed_bytes);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int a = y >= 8 && y < 48 && x >= 10 && x < 50;
            int b = y >= 60 && y < 108 && x >= 63 && x < 110;
            if (a || b) {
                size_t i = (size_t)y * width + x;
                packed[i / 8] |= 0x80 >> (i % 8);
            }
        }
    }
    Array mask = {
        .descr = "|u1", .ndim = 1, .shape = {packed_bytes}, .data = packed, .size = packed_bytes};
    npy_write(fmt(&a, "%s/ink.npy", folder), mask);
    npy_write(fmt(&a, "%s/supervised.npy", folder), mask);
    execute(argv[1], "bodies");
    json_object *body = json_read(fmt(&a, "%s/bodies/Frag1-3.24um.json", work));
    json_object *rows = jget(body, "rows");
    need(ji64(body, "step") == 2 && json_object_array_length(rows) == 2,
        "connected ink body count differs");
    json_object *first = json_object_array_get_idx(rows, 0);
    need(ji64(first, "x0") == 3 && ji64(first, "x1") == 23 && ji64(first, "y0") == 5 &&
            ji64(first, "y1") == 25 && jf64(first, "mm2") == .130,
        "registered connected ink body differs");
    json_object_put(body);
    Array labels = npz_get(&a, fmt(&a, "%s/bodies/Frag1-3.24um.npz", work), "labels");
    need(labels.shape[0] == 60 && labels.shape[1] == 61 && ((u32 *)labels.data)[5 * 61 + 3] == 1,
        "registered body labels differ");

    const char *ct = fmt(&a, "%s/layers", folder);
    text(fmt(&a, "%s/meta.json", ct), "{\"shape\":[21,120,121],\"brick\":[21,120,121]}");
    i64 offset = 0;
    npy_write(fmt(&a, "%s/offsets.npy", ct),
        (Array){.descr = "<i8", .ndim = 3, .shape = {1, 1, 1}, .size = 8, .data = &offset});
    size_t bytes = (size_t)21 * height * width;
    u8 *voxels = push(&a, bytes);
    for (int z = 0; z < 21; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                voxels[((size_t)z * height + y) * width + x] = 1 + (7 * z + y + 3 * x) % 251;
            }
        }
    }
    write_file(fmt(&a, "%s/bricks.bin", ct), voxels, bytes);
    execute(argv[1], "renders");
    const char *faces = fmt(&a, "%s/faces/Frag1-3.24um", work);
    json_object *tiles = json_read(fmt(&a, "%s/tiles.json", faces));
    need(json_object_is_type(tiles, json_type_array) && json_object_array_length(tiles) == 1,
        "expected one labelled atlas CT crop");
    json_object *tile = json_object_array_get_idx(tiles, 0);
    int w, h;
    u8 *pixels = png_decode(&a, read_file(&a, fmt(&a, "%s/%s", faces, jstr(tile, "file"))), &w, &h);
    need(w == 128 && h == 128 && pixels[0] == 99 && pixels[127 * w + 127] == 0,
        "labelled CT max projection differs");
    json_object_put(tiles);

    text(fmt(&a, "%s/cards.json", work),
        "[{\"case\":\"fixture\",\"sample\":\"PHerc0001\",\"segment\":\"s\",\"id\":2,"
        "\"energy\":113,\"um\":3.3,\"h\":4,\"w\":5,\"x_mm\":1,\"y_mm\":2,"
        "\"file\":\"later.jpg\"},{\"case\":\"fixture\",\"sample\":\"PHerc0001\","
        "\"segment\":\"s\",\"id\":1,\"energy\":77,\"um\":3.3,\"h\":4,\"w\":5,"
        "\"x_mm\":1,\"y_mm\":2,\"file\":\"earlier.jpg\"}]");
    text(fmt(&a, "%s/alphabet.json", work),
        "{\"table\":[{\"capital\":\"Α\",\"small\":\"α\","
        "\"models\":[],\"examples\":[],\"matched\":1}]}");
    atlas_scan(fmt(&a,
        "%s/published/open_data/PHerc0001/segments/20250917182700/"
        "ink-detection/downsampled/20250917182700-L0-method-0-tile-5.000um-77keV-ds40.jpg",
        library));
    execute(argv[1], "segment_maps");
    execute(argv[1], "segment_scores");
    json_object *segments = json_read(fmt(&a, "%s/segments.json", work));
    need(json_object_is_type(segments, json_type_array) && json_object_array_length(segments) == 1,
        "atlas did not index the published fixture");
    json_object *map = json_object_array_get_idx(segments, 0);
    need(ji64(map, "w") == 91 && ji64(map, "h") == 2400 && jf64(map, "bright") == .1682 &&
            jf64(map, "rows") == 289870.7 && !strcmp(jstr(map, "recipe"), "method-0"),
        "published map size, brightness or row strength differs");
    json_object_put(segments);
    execute(argv[1], "pages");
    const char *atlas = fmt(&a, "%s/website/atlas", output);
    char *data = (char *)read_file(&a, fmt(&a, "%s/data.js", atlas)).data;
    const char *early = strstr(data, "earlier.jpg"), *late = strstr(data, "later.jpg");
    need(early && late && early < late, "atlas cards lost their priority order");
    char *page = (char *)read_file(&a, fmt(&a, "%s/index.html", atlas)).data;
    need(strstr(page, "Α<small>α</small>") && strstr(page, "4.0") && strstr(page, "5.0") &&
            strstr(page, "lines.html"),
        "atlas page lost alphabet or scale data");
    need(exists(fmt(&a, "%s/lines.html", atlas)) && exists(fmt(&a, "%s/published.html", atlas)),
        "atlas pages are incomplete");
    const char *ir = fmt(&a, "%s/datasets/fragments/Frag1", library);
    Bytes photo = png_encode(&a, voxels, width, height, 6);
    write_file(fmt(&a, "%s/ir.png", ir), photo.data, photo.size);
    u8 *paper = push(&a, (size_t)width * height);
    memset(paper, 255, (size_t)width * height);
    Bytes silhouette = png_encode(&a, paper, width, height, 6);
    write_file(fmt(&a, "%s/mask.png", ir), silhouette.data, silhouette.size);
    execute(argv[1], "everything");
    json_object *matches = json_read(fmt(&a, "%s/matches.json", work));
    need(ji64(matches, "checked") == 1 && json_object_array_length(jget(matches, "bodies")) == 2,
        "atlas lost a hand-labelled match or connected body");
    json_object_put(matches);

    const char *reports[] = {"cards", "windows", "fragments", "compare"};
    const size_t counts[] = {2, 1, 1, 0};
    for (size_t i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        json_object *report = json_read(fmt(&a, "%s/%s.json", work, reports[i]));
        need(json_object_is_type(report, json_type_array) &&
                json_object_array_length(report) == counts[i],
            "unexpected atlas %s count", reports[i]);
        json_object_put(report);
    }
    json_object *letters = json_read(fmt(&a, "%s/alphabet.json", work));
    need(json_object_array_length(jget(letters, "table")) == 24, "atlas alphabet is incomplete");
    json_object_put(letters);
    need(exists(fmt(&a, "%s/alphabet/strip.png", atlas)) &&
            exists(fmt(&a, "%s/fragments/Frag1-labels.jpg", atlas)),
        "atlas did not render its alphabet strip or infrared labels");
    munmap(a.base, a.size);
    puts("atlas e2e passed");
    return 0;
}
