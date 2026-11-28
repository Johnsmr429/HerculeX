static void keep_faces_file(const char *source, const char *target, const struct stat *info) {
    struct stat held;
    if (!stat(target, &held) && info->st_dev == held.st_dev && info->st_ino == held.st_ino) {
        return;
    }
    need(!unlink(target) || errno == ENOENT, "cannot refresh %s: %s", target, strerror(errno));
    if (!link(source, target)) {
        return;
    }
    int input = open(source, O_RDONLY | O_CLOEXEC);
    need(input >= 0, "cannot read %s: %s", source, strerror(errno));
    int output = open(target, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, info->st_mode & 0777);
    need(output >= 0, "cannot retain %s: %s", target, strerror(errno));
    u8 buffer[65536];
    for (;;) {
        ssize_t count = read(input, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        need(count >= 0, "cannot read %s: %s", source, strerror(errno));
        if (!count) {
            break;
        }
        write_all(output, buffer, (size_t)count, target);
    }
#ifdef __APPLE__
    struct timespec times[] = {info->st_atimespec, info->st_mtimespec};
#else
    struct timespec times[] = {info->st_atim, info->st_mtim};
#endif
    need(!fchmod(output, info->st_mode & 0777) && !futimens(output, times),
        "cannot retain file metadata: %s", target);
    need(!close(input) && !close(output), "cannot close retained face: %s", target);
}

static void keep_faces_tree(Arena *a, const char *source, const char *target) {
    DIR *directory = opendir(source);
    need(directory != NULL, "cannot open faces at %s: %s", source, strerror(errno));
    mkdirs(target);
    struct stat input, output;
    need(!stat(source, &input) && !stat(target, &output) &&
            (input.st_dev != output.st_dev || input.st_ino != output.st_ino),
        "source and retained faces must be different directories");
    for (;;) {
        errno = 0;
        struct dirent *entry = readdir(directory);
        need(entry || !errno, "cannot list %s: %s", source, strerror(errno));
        if (!entry) {
            break;
        }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
            continue;
        }
        size_t mark = a->used;
        const char *from = fmt(a, "%s/%s", source, entry->d_name);
        const char *to = fmt(a, "%s/%s", target, entry->d_name);
        struct stat info;
        need(!stat(from, &info), "cannot inspect %s: %s", from, strerror(errno));
        if (S_ISDIR(info.st_mode)) {
            keep_faces_tree(a, from, to);
        } else {
            need(S_ISREG(info.st_mode), "not a regular face file: %s", from);
            keep_faces_file(from, to, &info);
        }
        a->used = mark;
    }
    need(!closedir(directory), "cannot close %s", source);
}

int keep_faces_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "-h") || !strcmp(argv[0], "--help"))) {
        puts("Usage: ./run.sh runner.keep_faces SOURCE TARGET");
        return 0;
    }
    need(argc == 2, "usage: ./run.sh runner.keep_faces SOURCE TARGET");
    Arena a = arena(1 << 20);
    keep_faces_tree(&a, argv[0], argv[1]);
    munmap(a.base, a.size);
    return 0;
}
