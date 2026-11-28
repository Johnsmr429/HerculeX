#include <ctype.h>
#include <errno.h>
#include <fts.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char hx_usage[] =
    "Every module ./run.sh can run, by package, with the first line of what it does.";

static int ordered(const FTSENT **a, const FTSENT **b) {
    return strcmp((*a)->fts_name, (*b)->fts_name);
}

static int skipped(const FTSENT *entry) {
    const char *name = entry->fts_name;
    if (entry->fts_level == 2 && !strcmp(entry->fts_parent->fts_name, "data")) {
        return strcmp(name, "catalog") != 0;
    }
    const char *skip[] = {"runtime", "ports", "ideas", "backups", "tests", "node_modules", ".git"};
    for (size_t k = 0; k < sizeof(skip) / sizeof(*skip); k++) {
        if (strcmp(name, skip[k]) == 0)
            return 1;
    }
    return 0;
}

static const char *first_text(const char *line) {
    while (isspace((unsigned char)*line))
        line++;
    return line;
}

static char *heading(const char *start, const char *quote) {
    start = first_text(start);
    size_t n = strcspn(start, "\r\n");
    const char *end = strstr(start, quote);
    if (end && (size_t)(end - start) < n)
        n = (size_t)(end - start);
    while (n && isspace((unsigned char)start[n - 1]))
        n--;
    return strndup(start, n);
}

static int describe(const char *path, char **doc) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char *line = NULL;
    size_t cap = 0;
    int found = 0, pending = 0;
    while (getline(&line, &cap, f) >= 0) {
        const char *s = first_text(line);
        if (!*s)
            continue;
        const char *tag = "const char hx_usage";
        const char *value = strncmp(s, tag, strlen(tag)) == 0 ? strstr(s, "[] =") : NULL;
        if (value) {
            pending = 1;
            s = first_text(value + strlen("[] ="));
        }
        if (pending && *s == '"') {
            *doc = heading(s + 1, "\"");
            found = 1;
            break;
        }
    }
    int failed = ferror(f) || (found && !*doc);
    free(line);
    fclose(f);
    return failed ? -1 : found;
}

static void shown_native(char *registry, const char *name) {
    size_t length = strlen(name);
    for (char *entry = registry; entry && *entry;) {
        char *next = strchr(entry, '\n');
        size_t start = strcspn(entry, "\t\n");
        if (start == length && !strncmp(entry, name, length)) {
            *entry = '\t';
            break;
        }
        entry = next ? next + 1 : NULL;
    }
}

static void remaining_native(char *registry) {
    if (!registry) {
        return;
    }
    puts("\nNative commands and aliases");
    char *save;
    for (char *entry = strtok_r(registry, "\n", &save); entry;
        entry = strtok_r(NULL, "\n", &save)) {
        if (*entry == '\t') {
            continue;
        }
        char *usage = strchr(entry, '\t');
        if (usage) {
            *usage++ = 0;
        }
        printf("  %-42s %s\n", entry, usage ? usage : "");
    }
}

int main(void) {
    if (!access("unroll/pipeline.sh", X_OK)) {
        puts("Pipeline commands\n"
             "  unwrap [--queue] [--from STAGE] [--rotate180] SCROLL[:SCAN] ...\n"
             "  unroll SCROLL SCAN STAGE [ARGS]");
    }
    char root[] = ".";
    char *roots[] = {root, NULL};
    FTS *walk = fts_open(roots, FTS_PHYSICAL | FTS_NOCHDIR, ordered);
    if (!walk) {
        perror("list modules");
        return 1;
    }
    const char *installed = getenv("HERCULEX_LIST_NATIVE");
    char *registry = installed ? strdup(installed) : NULL;
    char *package = NULL;
    int failed = 0;
    for (;;) {
        errno = 0;
        FTSENT *p = fts_read(walk);
        if (!p) {
            failed = errno != 0;
            break;
        }
        if (p->fts_info == FTS_D && skipped(p)) {
            fts_set(walk, p, FTS_SKIP);
            continue;
        }
        if (p->fts_info != FTS_F)
            continue;
        const char *ext = strrchr(p->fts_name, '.');
        if (!ext || (strcmp(ext, ".c") != 0 && strcmp(ext, ".inc") != 0))
            continue;
        char *doc = NULL;
        int found = describe(p->fts_path, &doc);
        if (found > 0) {
            char *name = strndup(p->fts_path + 2, strlen(p->fts_path + 2) - strlen(ext));
            if (!name) {
                free(doc);
                failed = 1;
                break;
            }
            for (char *c = name; *c; c++) {
                if (*c == '/')
                    *c = '.';
            }
            char *last = strrchr(name, '.');
            size_t n = last ? (size_t)(last - name) : strlen(name);
            if (!package || strlen(package) != n || strncmp(name, package, n) != 0) {
                free(package);
                package = strndup(name, n);
                if (!package) {
                    free(name);
                    free(doc);
                    failed = 1;
                    break;
                }
                printf("\n%s\n", package);
            }
            shown_native(registry, name);
            printf("  %-42s %s\n", name, doc ? doc : "");
            free(name);
        }
        free(doc);
        if (found < 0) {
            failed = 1;
            break;
        }
    }
    free(package);
    fts_close(walk);
    if (!failed) {
        remaining_native(registry);
    }
    free(registry);
    if (failed) {
        perror("list modules");
        return 1;
    }
    puts("\n./run.sh MODULE --help for a module's arguments; ./run.sh check [cpu|all|lint]; "
         "./run.sh board --help");
    return 0;
}
