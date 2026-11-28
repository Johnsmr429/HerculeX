#include "../base.h"

static int invoke(const char *binary, const char *module, const char *mode) {
    char *args[] = {(char *)binary, (char *)module, (char *)mode, "PHercTest/v1/rows-0-1", NULL};
    if (!mode) {
        args[2] = args[3];
        args[3] = NULL;
    }
    pid_t pid = fork();
    need(pid >= 0, "cannot start survey test");
    if (!pid) {
        execv(binary, args);
        _exit(127);
    }
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        need(errno == EINTR, "cannot wait for survey test");
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

int main(int argc, char **argv) {
    need(argc == 2, "usage: survey_e2e PATH-TO-HERCULEX");
    char temp[] = "/tmp/herculex-survey.XXXXXXXX";
    need(mkdtemp(temp) != NULL, "cannot create survey test fixture");
    Arena a = arena(16 << 20);
    const char *lib = fmt(&a, "%s/library", temp), *outputs = fmt(&a, "%s/outputs", temp);
    setenv("HERCULEX_LIBRARY", lib, 1);
    setenv("HERCULEX_OUTPUTS", outputs, 1);
    const char *results = fmt(&a, "%s/calibration/scroll_survey", lib);
    const char *first = fmt(&a, "%s/PHercTest", results);
    const char *pieces = fmt(&a, "%s/unwrapping/level/PHercTest/v1/pieces/rows-0-1", outputs);
    const char *meta = "{\"voxel_um\":40}";
    write_file(fmt(&a, "%s/piece-0001/meta.json", pieces), meta, strlen(meta));
    const char *axis = "[[0,0,0],[1000,0,0]]";
    write_file(
        fmt(&a, "%s/unwrapping/level/PHercTest/v1/sheet/axis.json", outputs), axis, strlen(axis));
    enum { H = 800, W = 925, NH = 100, NW = 115 };
    u8 *near = push(&a, H * W), *far = push(&a, H * W);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double dy = (y - 402.0) / 395, dx = (x - 463.0) / 456;
            int sheet = dy * dy + dx * dx < 1;
            double near_value = 120 + 75 * sin(y * 2 * M_PI / 125) + 30 * sin(x * 2 * M_PI / 146);
            double far_value = 125 + 78 * sin(x * 2 * M_PI / 125) + 25 * sin(y * 2 * M_PI / 152);
            near[y * W + x] = sheet ? nearbyint(near_value) : 0;
            far[y * W + x] = sheet ? nearbyint(far_value) : 0;
        }
    }
    png_write(fmt(&a, "%s/piece-0001.png", first), near, W, H);
    png_write(fmt(&a, "%s/reversed-piece-0001.png", first), far, W, H);
    const char *record = "{\"pixels\":5000}";
    write_file(fmt(&a, "%s/piece-0001.json", first), record, strlen(record));
    u16 *mask = push(&a, NH * NW * sizeof(*mask));
    for (int y = 0; y < NH; y++) {
        for (int x = 0; x < NW; x++) {
            mask[y * NW + x] = y > 4 ? 0x3a00 : 0x3400;
        }
    }
    Array cells = {.descr = "<f2",
        .ndim = 2,
        .shape = {NH, NW},
        .data = mask,
        .size = NH * NW * sizeof(*mask)};
    npz_write(fmt(&a, "%s/piece-0001.npz", first), 1, (const char *[]){"open0"}, &cells);
    need(!invoke(argv[1], "calibrate.survey.survey_rim_excess", NULL), "survey rim excess failed");
    need(!invoke(argv[1], "calibrate.survey.survey_row_power", NULL), "survey row power failed");
    need(!invoke(argv[1], "calibrate.survey.survey_fetch", "second"), "empty fetch plan failed");
    need(!invoke(argv[1], "calibrate.survey.scroll_survey", "all"), "completed survey plan failed");
    json_object *rim = json_read(fmt(&a, "%s/rim-excess.json", results));
    json_object *r = jget(jget(jget(rim, "PHercTest"), "first"), "piece-0001");
    const char *keys[] = {"0.0", "0.5", "1.0", "2.0", "3.0"};
    const i64 expected[][3] = {{557919, 142481, 175322}, {529743, 132844, 163579},
        {500795, 122599, 151911}, {440383, 107731, 134092}, {383099, 99401, 121233}};
    for (int i = 0; i < 5; i++) {
        json_object *counts = jget(r, keys[i]);
        for (int j = 0; j < 3; j++) {
            need(json_object_get_int64(json_object_array_get_idx(counts, j)) == expected[i][j],
                "survey rim count %s index %d changed", keys[i], j);
        }
    }
    json_object *power = json_read(fmt(&a, "%s/row-power.json", results));
    json_object *p = jget(jget(jget(power, "PHercTest"), "first"), "piece-0001");
    need(ji64(p, "windows") == 14 && fabs(jf64(p, "ratio") - 5.944715361165124) < 1e-6,
        "forward row power changed");
    need(fabs(jf64(jget(p, "reversed"), "ratio") - 0.10655109031296067) < 1e-6,
        "reversed row power changed");
    need(fabs(jf64(jget(p, "no_rim"), "ratio") - 6.044570996568109) < 1e-6,
        "trimmed row power changed");
    json_object_put(power);
    json_object_put(rim);
    remove_tree(temp);
    munmap(a.base, a.size);
    puts("survey e2e passed");
    return 0;
}
