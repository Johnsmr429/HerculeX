#include "../base.h"

static void execute(const char *binary, char *const args[]) {
    pid_t pid = fork();
    need(pid >= 0, "could not launch spot geometry check");
    if (!pid) {
        execv(binary, args);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "spot geometry command failed");
}

static char fixture[] = "/tmp/spot-geometry.XXXXXX";

static void clean(void) { remove_tree(fixture); }

int main(int argc, char **argv) {
    need(argc == 2 && mkdtemp(fixture), "expected a HerculeX binary and a temporary folder");
    atexit(clean);
    Arena a = arena(16 << 20);
    int h = 36, w = 44, cells = h * w;
    f32 *xyz = push(&a, 3 * cells * sizeof(*xyz));
    double px[44], py[36], x = 0, y = 0;
    for (int i = 0; i < w; i++) {
        px[i] = x += 16 * (1 + 0.25 * sin(i / 6.0));
    }
    for (int i = 0; i < h; i++) {
        py[i] = y += 16 * (1 + 0.15 * cos(i / 5.0));
    }
    for (int i = 0; i < cells; i++) {
        int y = i / w, x = i % w;
        int hole =
            (y >= 10 && y < 14 && x >= 10 && x < 20) || (y >= 20 && y < 23 && x >= 30 && x < 33);
        xyz[i] = hole ? -1 : 1000 + 400 * cos(px[x] / 400);
        xyz[cells + i] = hole ? -1 : 1000 + 400 * sin(px[x] / 400);
        xyz[2 * cells + i] = hole ? -1 : 1000 + py[y];
    }
    int lone = 21 * w + 31;
    xyz[lone] = xyz[cells + lone] = 1200;
    xyz[2 * cells + lone] = 1300;
    char *in = fmt(&a, "%s/input.npy", fixture), *out = fmt(&a, "%s/out.npz", fixture);
    char *report = fmt(&a, "%s/report.json", fixture);
    npy_write(in,
        (Array){.descr = "<f4",
            .ndim = 3,
            .shape = {3, h, w},
            .data = xyz,
            .size = 3 * cells * sizeof(*xyz)});
    execute(argv[1], (char *[]){argv[1], "spots.reflatten", in, "16", out, report, "1500", NULL});
    json_object *receipt = json_read(report);
    need(receipt && ji64(receipt, "isolated_nodes_left_out") == 1 &&
            ji64(receipt, "nodes_after") == 1434 && jf64(receipt, "within_10pct_after") == 1.0,
        "reflatten report differs from the reference");
    json_object_put(receipt);
    Array points = npz_get(&a, out, "position");
    need(isnan(((f32 *)points.data)[lone]) && isnan(((f32 *)points.data)[cells + lone]),
        "an isolated node entered the flattened mesh");
    Array grid = npz_get(&a, out, "grid");
    char sha[65];
    sha256_hex(grid.data, grid.size, sha);
    need(!strcmp(sha, "e172e58c13e22d349970b0af72bb2ad2c42c5b73a02be7e390d1105685bac990"),
        "the flattened mesh differs from the reference");

    int layers = 25;
    h = 15;
    w = 21;
    cells = h * w;
    u8 *stack = push(&a, layers * cells), *valid = push(&a, cells);
    for (int i = 0; i < cells; i++) {
        int y = i / w, x = i % w;
        valid[i] = !(y >= 2 && y < 5 && x >= 3 && x < 8);
        for (int z = 0; z < layers; z++) {
            stack[z * cells + i] = nearbyint(
                77 + 42 * sin((z + 2 * sin(x / 7.0) + cos(y / 9.0)) / 2) + 5 * cos((z + x) / 4.0));
        }
    }
    char *mask = fmt(&a, "%s/mask.npy", fixture);
    npy_write(in,
        (Array){.descr = "|u1",
            .ndim = 3,
            .shape = {layers, h, w},
            .data = stack,
            .size = layers * cells});
    npy_write(
        mask, (Array){.descr = "|b1", .ndim = 2, .shape = {h, w}, .data = valid, .size = cells});
    execute(argv[1],
        (char *[]){argv[1], "spots.flatten", in, out, "--mask", mask, "--rounds", "4", "--window",
            "9", "--reference-px", "17", "--smooth-px", "4", NULL});
    Array layers_out = npz_get(&a, out, "stack"), offset = npz_get(&a, out, "offset");
    sha256_hex(layers_out.data, layers_out.size, sha);
    need(!strcmp(sha, "bc0db048e8eecd2d38f912299b0a01f391011b7494e792834ac6e9ab31f5376e"),
        "the flattened stack differs from the reference");
    sha256_hex(offset.data, offset.size, sha);
    need(!strcmp(sha, "b86c3294ef7a8d39bf5fc1d67bbf39b0264da385f094cf8b41ebbd1dc7234736"),
        "the layer offsets differ from the reference");
    return 0;
}
