#include "../base.h"

enum { PATCH_NODES = 240, GRID = 32, WINDOW = 32, NODES = 2 * PATCH_NODES };

typedef struct {
    f32 x, y, z;
    i8 nx, ny, nz;
    u8 prominence, width, coherence;
    u16 pad;
} FixtureNode;

static void make_control(Arena *a, const char *path, int windows, int prominence) {
    FixtureNode *nodes = push(a, windows * NODES * sizeof(*nodes));
    i32 *cells = push(a, windows * NODES * sizeof(*cells));
    for (int w = 0; w < windows; w++) {
        for (int y = 0; y < 15; y++) {
            for (int x = 0; x < GRID; x++) {
                int i = w * NODES + y * GRID + x;
                cells[i] = w * WINDOW * GRID * GRID + y * GRID + x;
                nodes[i] = (FixtureNode){.x = 8 * x + 4,
                    .y = 8 * y + 4,
                    .z = 8 * w * WINDOW + 4,
                    .nx = 32,
                    .width = 6,
                    .coherence = 80,
                    .prominence = prominence + y};
            }
        }
    }
    const char *record = fmt(a,
        "{\"grid\":[%d,32,32],\"shape\":[%d,256,256],\"cell_planes\":[4,%d],\"rows\":[3,4],"
        "\"nodes\":%d}",
        windows * WINDOW, windows * WINDOW * 8, 4 + windows * WINDOW, windows * NODES);
    write_file(fmt(a, "%s/record.json", path), record, strlen(record));
    write_file(fmt(a, "%s/nodes.bin", path), nodes, windows * NODES * sizeof(*nodes));
    write_file(fmt(a, "%s/cells.bin", path), cells, windows * NODES * sizeof(*cells));
}

static void make_fit(Arena *a, const char *folder, int window) {
    i64 ids[NODES];
    i32 root[NODES], sides[2][NODES - 2];
    i8 normals[NODES][3] = {{0}};
    int edges = 0, prior[2] = {-1, -1};
    for (int i = 0; i < NODES; i++) {
        int patch = i % GRID >= GRID / 2;
        ids[i] = (INT64_C(3) << 32) + window * NODES + i;
        root[i] = patch * (GRID / 2);
        normals[i][0] = 32;
        if (prior[patch] >= 0) {
            sides[0][edges] = prior[patch];
            sides[1][edges++] = i;
        }
        prior[patch] = i;
    }
    const char *node_keys[] = {"ids", "root", "normal"};
    Array packed[] = {
        {.descr = "<i8", .ndim = 1, .shape = {NODES}, .data = ids, .size = sizeof(ids)},
        {.descr = "<i4", .ndim = 1, .shape = {NODES}, .data = root, .size = sizeof(root)},
        {.descr = "|i1", .ndim = 2, .shape = {NODES, 3}, .data = normals, .size = sizeof(normals)},
    };
    npz_write(fmt(a, "%s/nodes.npz", folder), 3, node_keys, packed);
    const char *edge_keys[] = {"a", "b"};
    Array edges_npz[] = {
        {.descr = "<i4", .ndim = 1, .shape = {edges}, .data = sides[0], .size = sizeof(sides[0])},
        {.descr = "<i4", .ndim = 1, .shape = {edges}, .data = sides[1], .size = sizeof(sides[1])},
    };
    npz_write(fmt(a, "%s/edges.npz", folder), 2, edge_keys, edges_npz);
}

int main(int argc, char **argv) {
    need(argc == 2, "solver view check needs the native CLI path");
    Arena a = arena(2 << 20);
    char fixture[] = "/tmp/hx-solver-view-test.XXXXXX";
    need(mkdtemp(fixture) != NULL, "cannot make solver view fixture");
    const char *lib = fmt(&a, "%s/library", fixture), *outputs = fmt(&a, "%s/outputs", fixture);
    const char *name = "PHerc0139-20250728140407-rows-0007-0019";
    const char *control = fmt(&a, "%s/controls/sheets/%s", lib, name);
    const char *report = fmt(&a, "%s/synthetic/realism/solver", outputs);
    const char *bands = fmt(&a, "%s/bands", fixture);
    make_control(&a, control, 2, 100);
    for (int i = 0; i < 2; i++) {
        const char *window =
            fmt(&a, "%s/%s-p%d-%d", report, name, 4 + i * WINDOW, 4 + (i + 1) * WINDOW);
        make_fit(&a, window, i);
        const char *rec = fmt(&a, "{\"planes\":[%d,%d]}", 4 + i * WINDOW, 4 + (i + 1) * WINDOW);
        write_file(fmt(&a, "%s/window.json", window), rec, strlen(rec));
        const char *band = fmt(&a, "%s/band-%d", bands, i);
        make_control(&a, fmt(&a, "%s/control", band), 1, 40);
        make_fit(&a, band, 0);
        const char info[] = "{\"look\":\"PHerc0139\"}";
        write_file(fmt(&a, "%s/band.json", band), info, sizeof(info) - 1);
        write_file(fmt(&a, "%s/truth.json", band), "{}", 2);
    }
    setenv("HERCULEX_LIBRARY", lib, 1);
    setenv("HERCULEX_OUTPUTS", outputs, 1);
    pid_t pid = fork();
    need(pid >= 0, "cannot start solver view check");
    if (!pid) {
        execl(argv[1], argv[1], "synthetic.three_d.solver_view", "compare", "PHerc0139", "--bands",
            "2", "--root", bands, "--tag", "fixture", (char *)NULL);
        _exit(127);
    }
    int status;
    need(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "solver view comparison failed");
    json_object *record = json_read(fmt(&a, "%s/compare-PHerc0139-fixture.json", report));
    need(record != NULL, "solver view comparison did not save its report");
    need(ji64(record, "real_windows") == 2 && ji64(record, "bands") == 2 &&
            ji64(record, "real_patches") == 4 && ji64(record, "synthetic_patches") == 4,
        "wrong solver view patch census");
    json_object *features = jget(record, "features"),
                *prominence = jget(features, "prominence_p50");
    need(jf64(prominence, "real_p50") == 107 && jf64(prominence, "synthetic_p50") == 47 &&
            jf64(prominence, "ks") == 1 && jf64(prominence, "auc") == 1,
        "wrong solver view prominence report");
    need(jf64(record, "auc_logistic") == 1 && jf64(record, "auc_network") == 1 &&
            jf64(record, "auc_within_band") == 0.5 &&
            jf64(record, "auc_within_band_network") == 0.5,
        "solver view classifiers changed");
    json_object_put(record);
    remove_tree(fixture);
    munmap(a.base, a.size);
    puts("native solver view e2e passed");
    return 0;
}
