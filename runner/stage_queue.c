typedef struct {
    char final[PATH_MAX], part[PATH_MAX], lock[PATH_MAX];
    int fd, build;
} RunnerUnit;

json_object *runner_record(const char *folder) {
    char path[PATH_MAX];
    int n = snprintf(path, sizeof(path), "%s/record.json", folder);
    need(n > 0 && n < PATH_MAX, "unit path too long: %s", folder);
    return json_read(path);
}

int runner_done(const char *folder, const char *method, json_object *same_output) {
    size_t length = strlen(folder);
    if (length >= 9 && !strcmp(folder + length - 9, ".building")) {
        return 0;
    }
    json_object *record = runner_record(folder);
    if (!record) {
        return 0;
    }
    json_object *field = json_object_object_get(record, "method");
    const char *held = json_object_get_string(field);
    int done = !method || (held && !strcmp(held, method));
    if (!done && held && same_output) {
        for (size_t i = 0; i < json_object_array_length(same_output); i++) {
            json_object *alias = json_object_array_get_idx(same_output, i);
            if (json_object_equal(alias, field)) {
                done = 1;
                break;
            }
        }
    }
    json_object_put(record);
    return done;
}

void runner_method(char out[13], int n, const char **sources) {
    EVP_MD_CTX *hash = EVP_MD_CTX_new();
    need(hash && EVP_DigestInit_ex(hash, EVP_sha256(), NULL), "cannot start source hash");
    char buffer[65536];
    for (int i = 0; i < n; i++) {
        int fd = open(sources[i], O_RDONLY | O_CLOEXEC);
        need(fd >= 0, "cannot open %s: %s", sources[i], strerror(errno));
        for (;;) {
            ssize_t size = read(fd, buffer, sizeof(buffer));
            if (size < 0 && errno == EINTR) {
                continue;
            }
            need(size >= 0, "cannot read %s: %s", sources[i], strerror(errno));
            if (!size) {
                break;
            }
            need(EVP_DigestUpdate(hash, buffer, size), "cannot hash %s", sources[i]);
        }
        close(fd);
    }
    u8 sum[32];
    need(EVP_DigestFinal_ex(hash, sum, NULL), "cannot finish source hash");
    EVP_MD_CTX_free(hash);
    for (int i = 0; i < 6; i++) {
        snprintf(out + 2 * i, 3, "%02x", sum[i]);
    }
}

static int runner_same_inode(int fd, const char *path) {
    struct stat a, b;
    need(fstat(fd, &a) == 0, "cannot stat open unit lock: %s", strerror(errno));
    int rc = stat(path, &b);
    need(!rc || errno == ENOENT, "cannot stat %s: %s", path, strerror(errno));
    return !rc && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

int runner_unit_begin(RunnerUnit *u, const char *folder, int build) {
    const char *slash = strrchr(folder, '/'), *base = slash ? slash + 1 : folder;
    int parent = slash ? (int)(slash - folder + 1) : 0;
    int n = snprintf(u->final, sizeof(u->final), "%s", folder);
    need(n > 0 && n < PATH_MAX && *base, "invalid unit folder %s", folder);
    n = snprintf(u->lock, sizeof(u->lock), "%.*s.%s.%s", parent, folder, base,
        build ? "build-lock" : "claim");
    need(n > 0 && n < PATH_MAX, "unit lock path too long");
    u->build = build;
    u->fd = -1;
    mkparent(u->lock);
    for (;;) {
        int fd = open(u->lock, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC, 0666);
        need(fd >= 0, "cannot open lock %s: %s", u->lock, strerror(errno));
        int rc;
        do {
            rc = flock(fd, LOCK_EX | (build ? 0 : LOCK_NB));
        } while (rc && errno == EINTR);
        if (rc) {
            int busy = errno == EWOULDBLOCK || errno == EAGAIN;
            close(fd);
            need(busy, "cannot lock %s", u->lock);
            return 0;
        }
        if (runner_same_inode(fd, u->lock)) {
            u->fd = fd;
            break;
        }
        close(fd);
    }
    if (build) {
        n = snprintf(u->part, sizeof(u->part), "%s.building", folder);
        need(n > 0 && n < PATH_MAX, "unit building path too long");
        remove_tree(u->part);
        need(mkdir(u->part, 0755) == 0, "cannot make %s: %s", u->part, strerror(errno));
    }
    return 1;
}

void runner_unit_end(RunnerUnit *u, int publish) {
    if (u->fd < 0) {
        return;
    }
    if (u->build && publish) {
        struct stat st;
        int old = lstat(u->final, &st) == 0, rc;
        if (old) {
            need(S_ISDIR(st.st_mode), "unit path %s is not a directory", u->final);
#ifdef __APPLE__
            rc = renameatx_np(AT_FDCWD, u->part, AT_FDCWD, u->final, RENAME_SWAP);
#else
            rc = renameat2(AT_FDCWD, u->part, AT_FDCWD, u->final, RENAME_EXCHANGE);
#endif
        } else {
            rc = rename(u->part, u->final);
        }
        need(!rc, "cannot publish %s: %s", u->final, strerror(errno));
        if (old) {
            remove_tree(u->part);
        }
    }
    if (runner_same_inode(u->fd, u->lock)) {
        need(unlink(u->lock) == 0 || errno == ENOENT, "cannot remove lock %s", u->lock);
    }
    close(u->fd);
    u->fd = -1;
}

void runner_finish(const char *folder, json_object *record) {
    char path[PATH_MAX], temp[PATH_MAX];
    int n = snprintf(path, sizeof(path), "%s/record.json", folder);
    need(n > 0 && n < PATH_MAX, "unit path too long");
    n = snprintf(temp, sizeof(temp), "%s/.record-XXXXXXXX", folder);
    need(n > 0 && n < PATH_MAX, "unit temp path too long");
    int fd = mkstemp(temp);
    need(fd >= 0, "cannot open record file in %s: %s", folder, strerror(errno));
    const char *data = json_object_to_json_string_ext(record,
        JSON_C_TO_STRING_PRETTY_TAB | JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_SPACED |
            JSON_C_TO_STRING_NOSLASHESCAPE);
    char *copy = strdup(data);
    need(copy != NULL, "cannot allocate unit record");
    for (char *p = copy; *p; p++) {
        if (*p == '\t') {
            *p = ' ';
        }
    }
    write_all(fd, copy, strlen(copy), temp);
    free(copy);
    need(close(fd) == 0 && rename(temp, path) == 0, "cannot publish %s: %s", path, strerror(errno));
}
