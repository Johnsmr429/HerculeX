double runner_event_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void runner_event(const char *kind, double start, double end, json_object *fields) {
    const char *path = getenv("HERCULEX_EVENTS");
    if (!path || !*path) {
        return;
    }
    json_object *row = json_object_new_object();
    json_object_object_add(row, "kind", json_object_new_string(kind));
    json_object_object_add(row, "start", jnum(round(start * 1000) / 1000));
    json_object_object_add(row, "end", jnum(round(end * 1000) / 1000));
    json_object_object_add(row, "pid", json_object_new_int(getpid()));
    if (fields) {
        json_object_object_foreach(fields, key, value) {
            json_object_object_add(row, key, json_object_get(value));
        }
    }
    size_t size;
    const char *data = json_object_to_json_string_length(
        row, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE, &size);
    char *line = malloc(size + 1);
    need(line != NULL, "cannot allocate event record");
    memcpy(line, data, size);
    line[size] = '\n';
    int fd = open(path, O_APPEND | O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    need(fd >= 0, "cannot open events file %s: %s", path, strerror(errno));
    ssize_t n;
    do {
        n = write(fd, line, size + 1);
    } while (n < 0 && errno == EINTR);
    need(n >= 0 && (size_t)n == size + 1, "cannot append events to %s", path);
    need(close(fd) == 0, "cannot close events file %s", path);
    free(line);
    json_object_put(row);
}
