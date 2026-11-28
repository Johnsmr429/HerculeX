#include "base.h"

static void write_crops(Arena *mem, const char *path, const char *name, int n, int side, int kind) {
    size_t saved = mem->used;
    size_t bytes = (size_t)n * side * side;
    u8 *out = push(mem, bytes);
    for (int i = 0; i < n; i++) {
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                int ridge = (x / (4 + kind) * 29 + y / (3 + 2 * kind) * 7) % 39;
                out[((size_t)i * side + y) * side + x] =
                    (u8)(x * 13 + y * 17 + i * 23 + ridge + kind * 3);
            }
        }
    }
    Array crops = {.descr = "|u1", .ndim = 3, .shape = {n, side, side}, .data = out, .size = bytes};
    npz_write_codec(path, 1, &name, &crops, ZIP_CM_DEFLATE);
    mem->used = saved;
}

int main(int argc, char **argv) {
    need(argc == 2, "native executable is required");
    char scratch[] = "/tmp/hx-slices-disc-XXXXXX";
    need(mkdtemp(scratch) != NULL, "cannot make discriminator fixture directory");
    Arena mem = arena(8 << 20);
    const char *real = fmt(&mem, "%s/real.npz", scratch);
    const char *first = fmt(&mem, "%s/first.npz", scratch);
    const char *second = fmt(&mem, "%s/second.npz", scratch);
    const char *stats = fmt(&mem, "%s/stats.json", scratch);
    write_crops(&mem, real, "crops", 11, 132, 0);
    write_crops(&mem, first, "image", 8, 144, 1);
    write_crops(&mem, second, "image", 7, 128, 2);
    pid_t pid = fork();
    if (!pid) {
        int fd = open(stats, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0) {
            _exit(126);
        }
        close(fd);
        execl(argv[1], argv[1], "synthetic.two_d.discriminator", real, first, second, "--steps",
            "3", "--seed", "7", NULL);
        _exit(127);
    }
    int status;
    need(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native synthetic discriminator fixture failed");
    json_object *report = json_read(stats);
    need(jf64(report, "real") == 11 && jf64(report, "synthetic") == 15 &&
            fabs(jf64(report, "auc") - .4) < 1e-9 &&
            fabs(jf64(report, "loss") - .6941392421722412) < 1e-6,
        "native discriminator AUC or class-balanced loss differs from the Python fixture");
    json_object_put(report);
    remove_tree(scratch);
    munmap(mem.base, mem.size);
    return 0;
}
