#include <fts.h>
#include <spawn.h>

extern char **environ;

static int port_path_order(const FTSENT **left, const FTSENT **right) {
    return strcmp((*left)->fts_name, (*right)->fts_name);
}

static void port_source_digest(const char *folder, const char *header, size_t prefix,
    const char *mlx, const char *cmake_prefix, char out[65]) {
    EVP_MD_CTX *hash = EVP_MD_CTX_new();
    need(hash && EVP_DigestInit_ex(hash, EVP_sha256(), NULL), "cannot initialize build digest");
    char *paths[] = {(char *)folder, (char *)header, NULL};
    FTS *walk = fts_open(paths, FTS_PHYSICAL | FTS_NOCHDIR, port_path_order);
    need(walk != NULL, "cannot walk %s", folder);
    FTSENT *entry;
    u8 bytes[65536];
    while ((entry = fts_read(walk))) {
        if (entry->fts_info == FTS_DNR || entry->fts_info == FTS_ERR) {
            die("cannot read %s", entry->fts_path);
        }
        struct stat info;
        if (entry->fts_info != FTS_F &&
            (entry->fts_info != FTS_SL || stat(entry->fts_accpath, &info) ||
                !S_ISREG(info.st_mode))) {
            continue;
        }
        int fd = open(entry->fts_accpath, O_RDONLY | O_CLOEXEC);
        need(fd >= 0, "cannot read %s", entry->fts_path);
        const char *relative = entry->fts_path + prefix;
        EVP_DigestUpdate(hash, relative, strlen(relative));
        ssize_t n;
        while ((n = read(fd, bytes, sizeof(bytes))) > 0 || (n < 0 && errno == EINTR)) {
            if (n > 0) {
                EVP_DigestUpdate(hash, bytes, (size_t)n);
            }
        }
        need(n == 0, "cannot hash %s", entry->fts_path);
        close(fd);
    }
    fts_close(walk);
    if (mlx) {
        EVP_DigestUpdate(hash, mlx, strlen(mlx) + 1);
        EVP_DigestUpdate(hash, cmake_prefix, strlen(cmake_prefix) + 1);
    }
    u8 digest[32];
    need(EVP_DigestFinal_ex(hash, digest, NULL), "cannot finish build digest");
    EVP_MD_CTX_free(hash);
    for (int i = 0; i < 32; i++) {
        snprintf(out + 2 * i, 3, "%02x", digest[i]);
    }
}

static void port_cmake(char **args) {
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    int error = posix_spawnp(&pid, args[0], &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    need(!error, "cannot run cmake: %s", strerror(error));
    int status;
    pid_t waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    need(waited == pid && WIFEXITED(status) && !WEXITSTATUS(status), "cmake %s failed", args[1]);
}

int port_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh port PORT TARGET");
        return 0;
    }
    need(argc == 2 && !strchr(argv[0], '/') && !strchr(argv[1], '/') && argv[0][0] != '.' &&
            argv[1][0] != '-' && argv[1][0] != '.',
        "usage: ./run.sh port PORT TARGET");
    Arena a = arena(1 << 20);
    char *base = fmt(&a, "%s/ports/", HOME);
    char *source = fmt(&a, "%s%s", base, argv[0]);
    need(exists(fmt(&a, "%s/CMakeLists.txt", source)), "unknown native port %s", argv[0]);
    char *build = fmt(&a, "%s/runtime/ports/%s", HOME, argv[0]);
    char *target = fmt(&a, "%s/%s", build, argv[1]);
    char *stamp = fmt(&a, "%s/.%s.source-digest", build, argv[1]);
    int lock = lock_file(fmt(&a, "%s/.build.lock", build), 1, 1);
    need(lock >= 0, "cannot lock native port %s", argv[0]);
    int surface = !strcmp(argv[0], "surface");
    const char *header = surface || !strcmp(argv[0], "lasagna") || !strcmp(argv[0], "spiral")
        ? fmt(&a, "%sjson.h", base)
        : NULL;
    const char *mlx = surface ? env("HERCULEX_MLX_ROOT", "") : NULL;
    char *resolved = mlx && *mlx ? realpath(mlx, NULL) : NULL;
    need(!mlx || !*mlx || resolved != NULL, "cannot resolve HERCULEX_MLX_ROOT: %s", mlx);
    if (resolved) {
        mlx = resolved;
    }
    char hash[65];
    port_source_digest(source, header, strlen(base), mlx, env("CMAKE_PREFIX_PATH", ""), hash);
    Bytes saved = slurp(&a, stamp);
    if (!exists(target) || saved.size != 64 || memcmp(hash, saved.data, 64) != 0) {
        char *configure[] = {"cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
            surface ? "-UMLX_DIR" : NULL, surface ? fmt(&a, "-DMLX_ROOT=%s", mlx) : NULL, NULL};
        char *compile[] = {"cmake", "--build", build, "-j", "8", "--target", argv[1], NULL};
        port_cmake(configure);
        port_cmake(compile);
        need(exists(target), "cmake did not produce %s", target);
        write_file(stamp, hash, 64);
    }
    puts(target);
    close(lock);
    free(resolved);
    munmap(a.base, a.size);
    return 0;
}
