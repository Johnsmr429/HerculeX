#include <limits.h>
#include <stdio.h>
#include <sys/stat.h>
#include "../storage/data_roots.c"

static char base[PATH_MAX];

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "data roots golden: %s\n", message);
        exit(1);
    }
}

static void clear(void) {
    const char *names[] = {"HERCULEX_DATA", "HERCULEX_LIBRARY", "HERCULEX_OUTPUTS",
        "HERCULEX_DISC_PCIE", "HERCULEX_DISC_PCLE3"};
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++)
        unsetenv(names[i]);
}

static void set(const char *name, const char *relative) {
    char *path = path_join(base, relative);
    setenv(name, path, 1);
    free(path);
}

static int is(RootSetting setting, const char *relative) {
    char *root = local_root(setting);
    char *expected = path_join(base, relative);
    int same = root && expected && !strcmp(root, expected);
    if (!same)
        fprintf(stderr, "%s is %s, not %s\n", setting.name, root ? root : "(none)", expected);
    free(root);
    free(expected);
    return same;
}

int main(void) {
    char made[] = "/tmp/hx-roots.XXXXXX";
    require(mkdtemp(made) && realpath(made, base), "cannot make a fixture folder");

    clear();
    setenv("HERCULEX_DATA", base, 1);
    require(is(library_root, "library") && is(outputs_root, "outputs"),
        "one data root holds the library and the outputs");

    clear();
    set("HERCULEX_DATA", "data");
    set("HERCULEX_OUTPUTS", "big");
    require(is(library_root, "data/library") && is(outputs_root, "big"),
        "the outputs can live apart from the data root");

    clear();
    set("HERCULEX_DISC_PCIE", "a");
    set("HERCULEX_DISC_PCLE3", "b");
    require(is(library_root, "a") && is(outputs_root, "b"), "the older names still work");

    clear();
    set("HERCULEX_LIBRARY", "lib");
    set("HERCULEX_DISC_PCIE", "a");
    require(is(library_root, "lib"), "the current name wins over the older one");

    clear();
    char *disc = path_join(base, "disc"), *data = path_join(base, "data");
    char *link = path_join(data, "outputs");
    require(!mkdir(disc, 0700) && !mkdir(data, 0700) && !symlink(disc, link),
        "cannot make the symlinked fixture");
    setenv("HERCULEX_DATA", data, 1);
    require(is(outputs_root, "disc"), "a symlinked data folder names its real place");

    unlink(link);
    rmdir(data);
    rmdir(disc);
    rmdir(base);
    free(link);
    free(data);
    free(disc);
    puts("data roots golden passed");
    return 0;
}
