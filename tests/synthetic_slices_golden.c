#include "base.h"

int main(int argc, char **argv) {
    need(argc == 2, "native executable is required");
    char scratch[] = "/tmp/hx-slices-golden-XXXXXX";
    need(mkdtemp(scratch) != NULL, "cannot make synthetic slices fixture directory");
    Arena mem = arena(8 << 20);
    const char *out = fmt(&mem, "%s/out", scratch);
    pid_t pid = fork();
    if (!pid) {
        execl(argv[1], argv[1], "synthetic.two_d.layout", "--out", out, "--seed", "42", "--n", "2",
            "--batch", "1", NULL);
        _exit(127);
    }
    int status;
    need(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native synthetic slices fixture failed");
    const char *labels[] = {
        "5990635270e2798d0643af918f187cd42ac2c9ac88216ec3a3f9aa0159c404ca",
        "a8aee61af778e7249d1ca8742befec8c48608fb05245cbe86501da45b23049ff",
    };
    const char *distances[] = {
        "e99a03a5ceeff633be393daa5fa3d62a3754d3699f75f5044d7818daab39b68e",
        "6aab3adf453b0024f012c61151f99baaf4a57cccbb900821d2e2f11363ea3ed2",
    };
    const i64 styles[] = {8, 2};
    for (int i = 0; i < 2; i++) {
        const char *path = fmt(&mem, "%s/shard_%04d.npz", out, i);
        Array sheet = npz_get(&mem, path, "sheet_id");
        Array dist = npz_get(&mem, path, "dist");
        Array lines = npz_get(&mem, path, "lines");
        Array style = npz_get(&mem, path, "style");
        Array seed = npz_get(&mem, path, "seed");
        need(sheet.ndim == 3 && sheet.shape[0] == 1 && sheet.shape[1] == 256 &&
                sheet.shape[2] == 256 && !strcmp(sheet.descr, "<u2") && sheet.size == 256 * 256 * 2,
            "label shape or type changed");
        need(dist.ndim == 3 && lines.ndim == 3 &&
                !memcmp(dist.shape, sheet.shape, 3 * sizeof(i64)) &&
                !memcmp(lines.shape, sheet.shape, 3 * sizeof(i64)) && !strcmp(dist.descr, "|u1") &&
                !strcmp(lines.descr, "|u1") && lines.size == 256 * 256 && dist.size == lines.size,
            "condition shape or type changed");
        need(style.ndim == 1 && style.shape[0] == 1 && !strcmp(style.descr, "<i8") &&
                style.size == sizeof(i64) && !memcmp(style.data, &styles[i], sizeof(i64)),
            "synthetic slice style changed");
        const u64 expected_seed = 42;
        need(seed.ndim == 0 && !strcmp(seed.descr, "<u8") && seed.size == sizeof(u64) &&
                !memcmp(seed.data, &expected_seed, sizeof(u64)),
            "synthetic slice seed changed");
        for (size_t p = 0; p < lines.size; p++) {
            u16 label;
            memcpy(&label, (u8 *)sheet.data + p * sizeof(label), sizeof(label));
            need(((u8 *)lines.data)[p] == (label != 0), "lines and labels differ");
        }
        char hash[65];
        sha256_hex(sheet.data, sheet.size, hash);
        need(!strcmp(hash, labels[i]), "sheet labels differ from the Python fixture: %s", hash);
        sha256_hex(dist.data, dist.size, hash);
        need(!strcmp(hash, distances[i]), "distance map differs from the Python fixture: %s", hash);
    }
    remove_tree(scratch);
    munmap(mem.base, mem.size);
    return 0;
}
