#include "data_roots.c"

const char *HOME, *DATA, *LIBRARY, *OUTPUTS;

void roots_init(void) {
    HOME = resolved(env("HERCULEX_HOME", "."));
    DATA = data_root();
    LIBRARY = local_root(library_root);
    OUTPUTS = local_root(outputs_root);
    need(HOME && DATA && LIBRARY && OUTPUTS, "cannot resolve HerculeX data roots");
}

#define UNKNOWN (-1)
#define ABSENT (-2)

typedef struct {
    char folder[PATH_MAX];
    i64 shape[3], brick[3], grid[3], size, count;
    i64 *offsets;
    const u8 *data;
    i64 mapped;
    pthread_mutex_t lock;
} Bricks;

void bricks_map(Bricks *b) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/bricks.bin", b->folder);
    int fd = open(path, O_RDONLY | O_CREAT, 0644);
    need(fd >= 0, "cannot open %s", path);
    struct stat st;
    fstat(fd, &st);
    if (b->data) {
        munmap((void *)b->data, b->mapped);
    }
    b->mapped = st.st_size;
    b->data = st.st_size ? mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0) : 0;
    need(b->data != MAP_FAILED, "cannot map %s", path);
    close(fd);
}

void bricks_open(Arena *a, Bricks *b, const char *folder) {
    snprintf(b->folder, sizeof(b->folder), "%s", folder);
    json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
    need(meta != 0, "%s holds no brick store", folder);
    for (int d = 0; d < 3; d++) {
        b->shape[d] = json_object_get_int64(json_object_array_get_idx(jget(meta, "shape"), d));
        b->brick[d] = json_object_get_int64(json_object_array_get_idx(jget(meta, "brick"), d));
        b->grid[d] = (b->shape[d] + b->brick[d] - 1) / b->brick[d];
    }
    json_object_put(meta);
    b->size = b->brick[0] * b->brick[1] * b->brick[2];
    b->count = b->grid[0] * b->grid[1] * b->grid[2];
    Array offsets = npy_read(a, fmt(a, "%s/offsets.npy", folder));
    need(elements(offsets) == b->count && !strcmp(offsets.descr, "<i8"),
        "%s/offsets.npy does not fit", folder);
    b->offsets = offsets.data;
    b->data = 0;
    pthread_mutex_init(&b->lock, 0);
    bricks_map(b);
}

void bricks_create(
    Arena *a, const char *folder, const i64 *shape, const i64 *brick, const char *source) {
    mkdirs(folder);
    json_object *meta = json_object_new_object();
    json_object *s = json_object_new_array(), *k = json_object_new_array();
    i64 count = 1, grid[3];
    for (int d = 0; d < 3; d++) {
        json_object_array_add(s, json_object_new_int64(shape[d]));
        json_object_array_add(k, json_object_new_int64(brick[d]));
        grid[d] = (shape[d] + brick[d] - 1) / brick[d];
        count *= grid[d];
    }
    json_object_object_add(meta, "shape", s);
    json_object_object_add(meta, "brick", k);
    json_object_object_add(meta, "source", source ? json_object_new_string(source) : 0);
    write_file(fmt(a, "%s/meta.json", folder),
        json_object_to_json_string_ext(meta, JSON_C_TO_STRING_SPACED),
        strlen(json_object_to_json_string_ext(meta, JSON_C_TO_STRING_SPACED)));
    json_object_put(meta);
    i64 *offsets = push(a, count * 8);
    for (i64 i = 0; i < count; i++) {
        offsets[i] = UNKNOWN;
    }
    npy_write(fmt(a, "%s/offsets.npy", folder),
        (Array){.descr = "<i8",
            .ndim = 3,
            .shape = {grid[0], grid[1], grid[2]},
            .data = offsets,
            .size = count * 8});
}

void bricks_put(Bricks *b, i64 index, const u8 *raw) {
    pthread_mutex_lock(&b->lock);
    if (!raw) {
        b->offsets[index] = ABSENT;
        pthread_mutex_unlock(&b->lock);
        return;
    }
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/bricks.bin", b->folder);
    int fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    need(fd >= 0, "cannot open %s", path);
    i64 offset = lseek(fd, 0, SEEK_END);
    write_all(fd, raw, b->size, path);
    close(fd);
    b->offsets[index] = offset;
    pthread_mutex_unlock(&b->lock);
}

void bricks_save(Bricks *b) {
    pthread_mutex_lock(&b->lock);
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/offsets.npy", b->folder);
    npy_write(path,
        (Array){.descr = "<i8",
            .ndim = 3,
            .shape = {b->grid[0], b->grid[1], b->grid[2]},
            .data = b->offsets,
            .size = b->count * 8});
    pthread_mutex_unlock(&b->lock);
}
