#include "../base.h"
#include <sqlite3.h>

enum { GRID = 8, PIXELS = GRID * GRID, TOTAL = 2 * PIXELS };

static void bench_fixture(Arena *a, const char *folder) {
    f32 points[TOTAL][3], normal[TOTAL][3] = {{0}}, area[TOTAL];
    f32 xy[TOTAL][2], z[TOTAL];
    i16 mesh[TOTAL];
    i32 frame[TOTAL] = {0};
    u8 flags[TOTAL] = {0}, zone[TOTAL] = {0};
    i64 sense = 1;
    double winding[TOTAL], knots[] = {0, 100, 100, 1000, 100, 100}, voxel = 10, spacing = 160;
    for (int s = 0; s < 2; s++) {
        for (int r = 0; r < GRID; r++) {
            for (int c = 0; c < GRID; c++) {
                int i = s * PIXELS + r * GRID + c;
                xy[i][0] = points[i][0] = 128 + 32 * s;
                xy[i][1] = points[i][1] = 100 + 16 * c;
                z[i] = points[i][2] = 400 + 16 * r;
                normal[i][0] = 1;
                area[i] = 0.0256f;
                mesh[i] = s;
                winding[i] = 3 + s;
            }
        }
    }
    const char *keys[] = {"points", "normals", "winding", "area_mm2", "frame", "mesh", "flags",
        "zone", "voxel_um", "spacing_um", "sense", "knots"};
    Array arrays[] = {
        {.descr = "<f4", .ndim = 2, .shape = {TOTAL, 3}, .data = points, .size = sizeof(points)},
        {.descr = "<f4", .ndim = 2, .shape = {TOTAL, 3}, .data = normal, .size = sizeof(normal)},
        {.descr = "<f8", .ndim = 1, .shape = {TOTAL}, .data = winding, .size = sizeof(winding)},
        {.descr = "<f4", .ndim = 1, .shape = {TOTAL}, .data = area, .size = sizeof(area)},
        {.descr = "<i4", .ndim = 1, .shape = {TOTAL}, .data = frame, .size = sizeof(frame)},
        {.descr = "<i2", .ndim = 1, .shape = {TOTAL}, .data = mesh, .size = sizeof(mesh)},
        {.descr = "|u1", .ndim = 1, .shape = {TOTAL}, .data = flags, .size = sizeof(flags)},
        {.descr = "|u1", .ndim = 1, .shape = {TOTAL}, .data = zone, .size = sizeof(zone)},
        {.descr = "<f8", .data = &voxel, .size = sizeof(voxel)},
        {.descr = "<f8", .data = &spacing, .size = sizeof(spacing)},
        {.descr = "<i8", .data = &sense, .size = sizeof(sense)},
        {.descr = "<f8", .ndim = 2, .shape = {2, 3}, .data = knots, .size = sizeof(knots)},
    };
    npz_write(fmt(a, "%s/truth.npz", folder), 12, keys, arrays);
    const char *c_keys[] = {"z", "xy", "normal", "mesh", "flags", "zone", "winding"};
    Array cross[] = {
        {.descr = "<f4", .ndim = 1, .shape = {TOTAL}, .data = z, .size = sizeof(z)},
        {.descr = "<f4", .ndim = 2, .shape = {TOTAL, 2}, .data = xy, .size = sizeof(xy)},
        arrays[1],
        arrays[5],
        arrays[6],
        arrays[7],
        arrays[2],
    };
    npz_write(fmt(a, "%s/crossings.npz", folder), 7, c_keys, cross);
    const char *record = "{\"scan\":{\"zones\":[[0,1000,\"dev\"]]}}";
    write_file(fmt(a, "%s/control.json", folder), record, strlen(record));
}

static void bench_grid(Arena *a, const char *folder, int s, int switched) {
    f32 plane[4][PIXELS];
    for (int r = 0; r < GRID; r++) {
        for (int c = 0; c < GRID; c++) {
            int i = r * GRID + c;
            plane[0][i] = 128 + 32 * (switched ? c >= GRID / 2 : s);
            plane[1][i] = 100 + 16 * c;
            plane[2][i] = 400 + 16 * r;
            plane[3][i] = 3 + s;
        }
    }
    const char axes[] = "xyzw";
    for (int i = 0; i < 4; i++) {
        tiff_write(fmt(a, "%s/%c.tif", folder, axes[i]), plane[i], GRID, GRID);
    }
}

static void run_score(const char *binary, const char *folder, const char *report, int record) {
    char *args[] = {(char *)binary, "calibrate.bench.score", "fixed", (char *)folder, "--out",
        (char *)report, record ? "--" : "--no-record", NULL};
    pid_t pid = fork();
    need(pid >= 0, "cannot start native scorer");
    if (!pid) {
        int out = open("/dev/null", O_WRONLY);
        dup2(out, STDOUT_FILENO);
        close(out);
        execv(binary, args);
        _exit(127);
    }
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        need(errno == EINTR, "cannot wait on native scorer");
    }
    need(WIFEXITED(status) && WEXITSTATUS(status) == 0, "native scorer returned an error");
}

int main(int argc, char **argv) {
    need(argc == 2, "usage: bench_score_golden PATH-TO-HERCULEX");
    char fixture[] = "/tmp/herculex-native-score.XXXXXXXX";
    need(mkdtemp(fixture) != NULL, "cannot create score fixture");
    setenv("HERCULEX_OUTPUTS", fixture, 1);
    Arena a = arena(8 << 20);
    bench_fixture(&a, fmt(&a, "%s/bench/controls/fixed", fixture));
    const char *piece = fmt(&a, "%s/perfect", fixture), *mixed = fmt(&a, "%s/switch", fixture);
    const char *out0 = fmt(&a, "%s/perfect-score", fixture),
               *out1 = fmt(&a, "%s/switch-score", fixture);
    bench_grid(&a, fmt(&a, "%s/piece-0", piece), 0, 0);
    bench_grid(&a, fmt(&a, "%s/piece-1", piece), 1, 0);
    bench_grid(&a, mixed, 0, 1);
    run_score(argv[1], piece, out0, 0);
    run_score(argv[1], mixed, out1, 1);
    json_object *perfect = json_read(fmt(&a, "%s/summary.json", out0));
    json_object *mixed_report = json_read(fmt(&a, "%s/summary.json", out1));
    need(jf64(perfect, "truth_recall") == 1 && jf64(perfect, "truth_recall_correct") == 1,
        "perfect fixture lost truth");
    need(jf64(jget(perfect, "winding"), "right_share") == 1 &&
            ji64(jget(perfect, "winding"), "judged_crossings") == TOTAL,
        "perfect crossing rank changed");
    need(jf64(jget(mixed_report, "winding"), "right_share") == 0.5 &&
            jf64(mixed_report, "wrong_join_share") > 0,
        "switch was not identified");
    Array ranks = npy_read(&a, fmt(&a, "%s/truth_rank.npy", out0));
    for (int i = 0; i < TOTAL; i++) {
        need(((i8 *)ranks.data)[i] == 0, "perfect truth vertex %d got the wrong rank", i);
    }
    sqlite3 *db;
    need(sqlite3_open_v2(fmt(&a, "%s/bench/results.sqlite", fixture), &db, SQLITE_OPEN_READONLY,
             NULL) == SQLITE_OK,
        "no bench result database");
    sqlite3_stmt *q;
    need(sqlite3_prepare_v2(db, "SELECT value FROM metrics WHERE key='winding_right_share'", -1, &q,
             NULL) == SQLITE_OK &&
            sqlite3_step(q) == SQLITE_ROW && sqlite3_column_double(q, 0) == 0.5,
        "result database did not persist the switch metric");
    sqlite3_finalize(q);
    sqlite3_close(db);
    json_object_put(perfect);
    json_object_put(mixed_report);
    remove_tree(fixture);
    munmap(a.base, a.size);
    puts("native bench score golden passed");
    return 0;
}
