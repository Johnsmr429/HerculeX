typedef struct {
    FILE *input, *output;
    pid_t child;
    i64 queued;
} StudioReader;

static void studio_transfer(FILE *stream, void *data, size_t size, int writing) {
    size_t done = 0;
    while (done < size) {
        size_t moved = writing ? fwrite((u8 *)data + done, 1, size - done, stream)
                               : fread((u8 *)data + done, 1, size - done, stream);
        need(moved, "Studio worker stopped during inference");
        done += moved;
    }
}

static StudioReader *studio_open(Arena *a, const char *weights) {
    int request[2], response[2];
    need(!pipe(request) && !pipe(response), "cannot open Studio worker pipes");
    pid_t child = fork();
    need(child >= 0, "cannot start Studio worker");
    if (!child) {
        close(request[1]);
        close(response[0]);
        dup2(request[0], STDIN_FILENO);
        dup2(response[1], STDOUT_FILENO);
        close(request[0]);
        close(response[1]);
        const char *python = env("HERCULEX_PYTHON", "python3");
        execlp(python, python, "-I", "ink/model/studio/worker.py", weights, NULL);
        _exit(127);
    }
    close(request[0]);
    close(response[1]);
    StudioReader *reader = pushz(a, sizeof(*reader));
    reader->input = fdopen(request[1], "wb");
    reader->output = fdopen(response[0], "rb");
    reader->child = child;
    need(reader->input && reader->output, "cannot connect to Studio worker");
    char ready[4];
    studio_transfer(reader->output, ready, sizeof(ready), 0);
    need(!memcmp(ready, "HXOK", 4), "Studio worker did not initialize");
    return reader;
}

static i64 studio_read(StudioReader *reader, const u8 *stack, int h, int w, f32 *out) {
    u32 shape[3] = {DEPTH, (u32)h, (u32)w};
    studio_transfer(reader->input, shape, sizeof(shape), 1);
    studio_transfer(reader->input, (void *)stack, (size_t)DEPTH * h * w, 1);
    need(!fflush(reader->input), "cannot send a stack to Studio");
    studio_transfer(reader->output, out, (size_t)h * w * sizeof(*out), 0);
    return ++reader->queued;
}
