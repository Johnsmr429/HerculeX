#ifndef HX_UNITY
const char hx_usage[] = "Report disc use and list reclaimable stage work.";
#endif

#include <fts.h>

typedef struct {
    const char *path, *kind;
    double gib;
} DiskEntry;

typedef struct {
    DiskEntry *entry;
    int count, cap;
} DiskEntries;

static double disk_size(const char *path) {
    char *paths[] = {(char *)path, NULL};
    FTS *tree = fts_open(paths, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
    if (!tree) {
        return 0;
    }
    typedef struct {
        dev_t dev;
        ino_t ino;
    } DiskInode;
    DiskInode *links = NULL;
    int count = 0, cap = 0;
    i64 blocks = 0;
    for (FTSENT *p; (p = fts_read(tree));) {
        if (p->fts_info == FTS_DP || p->fts_info == FTS_NS || p->fts_info == FTS_ERR ||
            !p->fts_statp) {
            continue;
        }
        struct stat *st = p->fts_statp;
        if (!S_ISDIR(st->st_mode) && st->st_nlink > 1) {
            int i = 0;
            while (i < count && (links[i].dev != st->st_dev || links[i].ino != st->st_ino)) {
                i++;
            }
            if (i < count) {
                continue;
            }
            if (count == cap) {
                cap = cap ? 2 * cap : 16;
                void *grown = realloc(links, cap * sizeof(*links));
                need(grown != NULL, "cannot record hard links in %s", path);
                links = grown;
            }
            links[count++] = (DiskInode){st->st_dev, st->st_ino};
        }
        blocks += st->st_blocks;
    }
    free(links);
    fts_close(tree);
    return blocks * 512.0 / GB;
}

static void disk_add(Arena *a, DiskEntries *d, const char *path, const char *kind) {
    if (d->count == d->cap) {
        d->cap = d->cap ? 2 * d->cap : 32;
        void *grown = realloc(d->entry, d->cap * sizeof(*d->entry));
        need(grown != NULL, "cannot allocate disk report");
        d->entry = grown;
    }
    d->entry[d->count++] = (DiskEntry){fmt(a, "%s", path), kind, 0};
}

static void disk_measure(void *context, i64 index) {
    DiskEntry *entry = (DiskEntry *)context + index;
    entry->gib = disk_size(entry->path);
}

static const char *disk_base(const char *path) {
    const char *base = strrchr(path, '/');
    return base ? base + 1 : path;
}

static int disk_order(const void *aa, const void *bb) {
    const DiskEntry *a = aa, *b = bb;
    if (a->gib != b->gib) {
        return a->gib > b->gib ? -1 : 1;
    }
    return strcmp(a->path, b->path);
}

static void disk_children(Arena *a, DiskEntries *d, const char *folder) {
    DIR *dir = opendir(folder);
    if (!dir) {
        return;
    }
    for (struct dirent *entry; (entry = readdir(dir));) {
        if (entry->d_name[0] == '.' && strcmp(entry->d_name, ".building") != 0) {
            continue;
        }
        char path[PATH_MAX];
        int n = snprintf(path, sizeof(path), "%s/%s", folder, entry->d_name);
        struct stat s;
        if (n > 0 && n < PATH_MAX && !lstat(path, &s) && !S_ISLNK(s.st_mode)) {
            disk_add(a, d, path, "");
        }
    }
    closedir(dir);
}

static void disk_tree(Arena *a, const char *folder, int depth, int indent) {
    size_t saved = a->used;
    DiskEntries d = {0};
    disk_children(a, &d, folder);
    if (!d.count) {
        free(d.entry);
        a->used = saved;
        return;
    }
    parallel(d.count, 8, disk_measure, d.entry);
    qsort(d.entry, d.count, sizeof(*d.entry), disk_order);
    double rest = 0;
    for (int i = 0; i < d.count; i++) {
        DiskEntry *entry = &d.entry[i];
        if (i < 8) {
            printf("%*s%7.1f GiB  %s\n", indent, "", entry->gib, disk_base(entry->path));
            if (depth > 1 && entry->gib > 1) {
                disk_tree(a, entry->path, depth - 1, indent + 4);
            }
        } else {
            rest += entry->gib;
        }
    }
    if (d.count > 8) {
        printf("%*s%7.1f GiB  (%d more)\n", indent, "", rest, d.count - 8);
    }
    free(d.entry);
    a->used = saved;
}

static int disk_has_faces(Arena *a, const char *store) {
    size_t saved = a->used;
    json_object *planned = json_read(fmt(a, "%s/planned.json", store));
    int count = board_count(planned), ready = count > 0;
    const char *model = getenv("HERCULEX_INK_MODEL");
    const char *production = model && *model && strcmp(model, "reader-v3") != 0
        ? fmt(a, "level-0-%s", model)
        : "level-0";
    const char *faces = env("HERCULEX_FACES", production);
    for (int i = 0; ready && i < count; i++) {
        char *piece = fmt(a, "%s", board_item(planned, i));
        const char *name = disk_base(piece);
        if (name == piece) {
            ready = 0;
            break;
        }
        *(char *)(name - 1) = 0;
        const char *unit = disk_base(piece);
        if (unit == piece) {
            ready = 0;
            break;
        }
        *(char *)(unit - 1) = 0;
        char *parent = strrchr(piece, '/');
        if (!parent) {
            ready = 0;
            break;
        }
        *parent = 0;
        ready = exists(fmt(a, "%s/faces/%s/%s/%s.png", piece, faces, unit, name));
    }
    json_object_put(planned);
    a->used = saved;
    return ready;
}

static void disk_detection(Arena *a, DiskEntries *d, const char *root) {
    glob_t paths = {0};
    glob(fmt(a, "%s/*/*/L*-*-*", root), 0, NULL, &paths);
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s", paths.gl_pathv[i]);
        char *chunk = strrchr(path, '/');
        if (!chunk) {
            continue;
        }
        *chunk++ = 0;
        char *scan = strrchr(path, '/');
        if (!scan) {
            continue;
        }
        *scan++ = 0;
        const char *sample = disk_base(path), *rows = strchr(chunk, '-');
        if (rows) {
            char unit[PATH_MAX];
            int n = snprintf(unit, sizeof(unit), "%s/unwrapping/nodes/%s/%s/nodes/rows%s", OUTPUTS,
                sample, scan, rows);
            if (n > 0 && n < PATH_MAX && runner_done(unit, NULL, NULL)) {
                disk_add(a, d, paths.gl_pathv[i], "detection store");
            }
        }
    }
    globfree(&paths);
}

static DiskEntries disk_reclaimable(Arena *a) {
    DiskEntries d = {0};
    double stamp = board_time();
    const char *routes[] = {"nodes", "level"};
    for (int i = 0; i < 2; i++) {
        glob_t paths = {0};
        glob(fmt(a, "%s/unwrapping/%s/*/*/*/*.building", OUTPUTS, routes[i]), 0, NULL, &paths);
        for (size_t j = 0; j < paths.gl_pathc; j++) {
            const char *path = paths.gl_pathv[j], *base = disk_base(path);
            struct stat st;
            char claim[PATH_MAX];
            int length = snprintf(claim, sizeof(claim), "%.*s.%.*s.claim", (int)(base - path), path,
                (int)strlen(base) - 9, base);
            if (length > 0 && length < PATH_MAX && !exists(claim) && !stat(path, &st) &&
                stamp - st.st_mtime > 3600) {
                disk_add(a, &d, path, "unfinished unit");
            }
        }
        globfree(&paths);
    }
    glob_t stores = {0};
    glob(fmt(a, "%s/cache/bricks/*/*", LIBRARY), 0, NULL, &stores);
    for (size_t i = 0; i < stores.gl_pathc; i++) {
        if (disk_has_faces(a, stores.gl_pathv[i])) {
            disk_add(a, &d, stores.gl_pathv[i], "read CT store");
        }
    }
    globfree(&stores);
    const char *work = fmt(a, "%s/unwrapping/detection-work", OUTPUTS);
    disk_detection(a, &d, work);
#ifdef __linux__
    char resolved[PATH_MAX];
    const char *key = realpath(work, resolved);
    if (key) {
        u64 hash = UINT64_C(14695981039346656037);
        for (const unsigned char *p = (const unsigned char *)key; *p; p++) {
            hash = (hash ^ *p) * UINT64_C(1099511628211);
        }
        const char *path = fmt(a, "/dev/shm/herculex-detection-%lu-%016llx",
            (unsigned long)getuid(), (unsigned long long)hash);
        struct stat st;
        if (!lstat(path, &st) && S_ISDIR(st.st_mode) && st.st_uid == getuid() &&
            (st.st_mode & 0777) == 0700) {
            disk_detection(a, &d, path);
        }
    }
#endif
    DiskEntries scratch = {0};
    disk_children(a, &scratch, env("HERCULEX_SCRATCH", fmt(a, "%s/runtime/scratch", HOME)));
    for (int i = 0; i < scratch.count; i++) {
        const char *path = scratch.entry[i].path;
        struct stat st;
        if (!stat(path, &st) && S_ISDIR(st.st_mode) && stamp - st.st_mtime > 86400) {
            disk_add(a, &d, path, "old scratch");
        }
    }
    free(scratch.entry);
    parallel(d.count, 8, disk_measure, d.entry);
    return d;
}

int disk_command(int argc, char **argv) {
    int reclaim = 0, depth = 2;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("runner.disk [report|reclaimable] [--depth N]");
            return 0;
        }
        if (!strcmp(argv[i], "reclaimable")) {
            reclaim = 1;
        } else if (!strcmp(argv[i], "--depth") && i + 1 < argc) {
            depth = board_value(argv[++i]);
        } else if (strcmp(argv[i], "report") != 0) {
            die("unknown disk argument %s", argv[i]);
        }
    }
    Arena a = arena(16 << 20);
    if (!reclaim) {
        const char *roots[] = {HOME, LIBRARY, OUTPUTS};
        const char *names[] = {"Mac", "library", "outputs"};
        for (int i = 0; i < 3; i++) {
            struct statvfs v;
            if (!statvfs(roots[i], &v)) {
                printf("%s: %.1f GiB free of %.1f GiB (%s)\n", names[i],
                    v.f_bavail * (double)v.f_frsize / GB, v.f_blocks * (double)v.f_frsize / GB,
                    roots[i]);
            }
        }
        const char *trees[] = {
            OUTPUTS, LIBRARY, fmt(&a, "%s/runtime", HOME), fmt(&a, "%s/studies", OUTPUTS)};
        const char *labels[] = {"outputs", "library", "runtime", "studies by owner"};
        for (int i = 0; i < 4; i++) {
            if (i != 3 || exists(trees[i])) {
                printf("\n%s\n", labels[i]);
                disk_tree(&a, trees[i], i == 3 ? 1 : depth, 2);
            }
        }
    }
    DiskEntries d = disk_reclaimable(&a);
    if (reclaim) {
        if (d.count > 1) {
            qsort(d.entry, d.count, sizeof(*d.entry), disk_order);
        }
        for (int i = 0; i < d.count; i++) {
            printf("%7.1f GiB  %-16s  rm -r '", d.entry[i].gib, d.entry[i].kind);
            for (const char *p = d.entry[i].path; *p; p++) {
                if (*p == '\'') {
                    fputs("'\\''", stdout);
                } else {
                    putchar(*p);
                }
            }
            puts("'");
        }
    } else {
        double gib = 0;
        for (int i = 0; i < d.count; i++) {
            gib += d.entry[i].gib;
        }
        printf(
            "\nreclaimable: %.1f GiB in %d folders; see `runner.disk reclaimable`\n", gib, d.count);
    }
    free(d.entry);
    munmap(a.base, a.size);
    return 0;
}
