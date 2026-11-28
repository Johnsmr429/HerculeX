#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <json-c/json.h>
#include <openssl/evp.h>
#include <png.h>
#include <jpeglib.h>
#include <tiffio.h>
#include <zip.h>
#include <zstd.h>

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

#define GB (1ull << 30)

__attribute__((noreturn)) void die(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("herculex: ", stderr);
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
    exit(1);
}

void need(int ok, const char *format, ...) {
    if (ok) {
        return;
    }
    va_list args;
    va_start(args, format);
    fputs("herculex: ", stderr);
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
    exit(1);
}

#define need(ok, ...) ((ok) ? (void)0 : die(__VA_ARGS__))

extern char **environ;

FILE *spawn_read(char *const *args, int quiet, pid_t *pid) {
    int ends[2];
    if (pipe(ends)) {
        return NULL;
    }
    posix_spawn_file_actions_t ops;
    posix_spawn_file_actions_init(&ops);
    posix_spawn_file_actions_addclose(&ops, ends[0]);
    posix_spawn_file_actions_adddup2(&ops, ends[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&ops, ends[1]);
    if (quiet) {
        posix_spawn_file_actions_addopen(&ops, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    }
    int failed = posix_spawnp(pid, args[0], &ops, NULL, args, environ);
    posix_spawn_file_actions_destroy(&ops);
    close(ends[1]);
    FILE *out = failed ? NULL : fdopen(ends[0], "r");
    if (!out) {
        close(ends[0]);
        if (!failed) {
            waitpid(*pid, NULL, 0);
        }
    }
    return out;
}

int spawn_close(FILE *out, pid_t pid) {
    fclose(out);
    int status = 0;
    pid_t done;
    do {
        done = waitpid(pid, &status, 0);
    } while (done < 0 && errno == EINTR);
    return done == pid && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

typedef struct {
    u8 *base;
    size_t used, size;
} Arena;

Arena arena(size_t size) {
    void *base =
        mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    need(base != MAP_FAILED, "cannot reserve %zu bytes", size);
    return (Arena){.base = base, .size = size};
}

void *push(Arena *a, size_t n) {
    size_t at = (a->used + 63) & ~(size_t)63;
    need(at + n <= a->size, "arena of %zu bytes is full", a->size);
    a->used = at + n;
    return a->base + at;
}

void *pushz(Arena *a, size_t n) { return memset(push(a, n), 0, n); }

__attribute__((format(printf, 2, 3))) char *fmt(Arena *a, const char *format, ...) {
    va_list args, again;
    va_start(args, format);
    va_copy(again, args);
    int n = vsnprintf(0, 0, format, args);
    char *s = push(a, n + 1);
    vsnprintf(s, n + 1, format, again);
    va_end(again);
    va_end(args);
    return s;
}

typedef struct {
    u8 *data;
    size_t size;
} Bytes;

int exists(const char *path) {
    struct stat s;
    return stat(path, &s) == 0;
}

i64 mtime_ns(const char *path) {
    struct stat s;
    if (stat(path, &s)) {
        return 0;
    }
#ifdef __APPLE__
    return (i64)s.st_mtimespec.tv_sec * 1000000000 + s.st_mtimespec.tv_nsec;
#else
    return (i64)s.st_mtim.tv_sec * 1000000000 + s.st_mtim.tv_nsec;
#endif
}

void mkdirs(const char *path) {
    char p[PATH_MAX];
    snprintf(p, sizeof(p), "%s", path);
    for (char *c = p + 1; *c; c++) {
        if (*c != '/') {
            continue;
        }
        *c = 0;
        mkdir(p, 0755);
        *c = '/';
    }
    need(mkdir(p, 0755) == 0 || errno == EEXIST, "cannot make %s: %s", p, strerror(errno));
}

void mkparent(const char *path) {
    char p[PATH_MAX];
    snprintf(p, sizeof(p), "%s", path);
    char *slash = strrchr(p, '/');
    if (slash && slash != p) {
        *slash = 0;
        mkdirs(p);
    }
}

Bytes slurp(Arena *a, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return (Bytes){0};
    }
    struct stat s;
    fstat(fd, &s);
    Bytes b = {.data = push(a, s.st_size + 1), .size = s.st_size};
    for (size_t at = 0; at < b.size;) {
        ssize_t n = read(fd, b.data + at, b.size - at);
        need(n > 0, "cannot read %s: %s", path, strerror(errno));
        at += n;
    }
    b.data[b.size] = 0;
    close(fd);
    return b;
}

Bytes read_file(Arena *a, const char *path) {
    Bytes b = slurp(a, path);
    need(b.data != 0, "cannot open %s: %s", path, strerror(errno));
    return b;
}

void write_all(int fd, const void *data, size_t n, const char *path) {
    for (size_t at = 0; at < n;) {
        ssize_t w = write(fd, (const u8 *)data + at, n - at);
        need(w > 0, "cannot write %s: %s", path, strerror(errno));
        at += w;
    }
}

void write_file(const char *path, const void *data, size_t n) {
    char part[PATH_MAX];
    snprintf(part, sizeof(part), "%s.part", path);
    mkparent(path);
    int fd = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    need(fd >= 0, "cannot write %s: %s", part, strerror(errno));
    write_all(fd, data, n, part);
    close(fd);
    need(rename(part, path) == 0, "cannot rename %s: %s", part, strerror(errno));
}

int name_order(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

char **list_dir(Arena *a, const char *path, int *count) {
    *count = 0;
    DIR *d = opendir(path);
    if (!d) {
        return 0;
    }
    int cap = 256;
    char **names = malloc(cap * sizeof(char *));
    need(names != NULL, "cannot list %s: out of memory", path);
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') {
            continue;
        }
        if (*count == cap) {
            cap *= 2;
            char **larger = realloc(names, cap * sizeof(*names));
            need(larger != NULL, "cannot list %s: out of memory", path);
            names = larger;
        }
        names[(*count)++] = fmt(a, "%s", e->d_name);
    }
    closedir(d);
    char **sorted = push(a, (*count + 1) * sizeof(char *));
    memcpy(sorted, names, *count * sizeof(char *));
    free(names);
    qsort(sorted, *count, sizeof(char *), name_order);
    sorted[*count] = 0;
    return sorted;
}

void remove_tree(const char *path) {
    struct stat s;
    if (lstat(path, &s)) {
        return;
    }
    if (S_ISDIR(s.st_mode)) {
        DIR *d = opendir(path);
        if (d) {
            for (struct dirent *e = readdir(d); e; e = readdir(d)) {
                if (strcmp(e->d_name, ".") != 0 && strcmp(e->d_name, "..") != 0) {
                    char child[PATH_MAX];
                    snprintf(child, sizeof(child), "%s/%s", path, e->d_name);
                    remove_tree(child);
                }
            }
            closedir(d);
        }
        rmdir(path);
        return;
    }
    unlink(path);
}

int lock_file(const char *path, int exclusive, int wait) {
    mkparent(path);
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    need(fd >= 0, "cannot open lock %s: %s", path, strerror(errno));
    if (flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | (wait ? 0 : LOCK_NB))) {
        close(fd);
        return -1;
    }
    return fd;
}

f64 now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

char *utc_stamp(Arena *a, const char *format) {
    time_t t = time(0);
    struct tm u;
    gmtime_r(&t, &u);
    char s[64];
    strftime(s, sizeof(s), format, &u);
    return fmt(a, "%s", s);
}

const char *flag(int argc, char **argv, const char *name) {
    for (int i = 0; i + 1 < argc; i++) {
        if (!strcmp(argv[i], name)) {
            return argv[i + 1];
        }
    }
    return 0;
}

int has_flag(int argc, char **argv, const char *name) {
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], name)) {
            return 1;
        }
    }
    return 0;
}

const char *env(const char *name, const char *otherwise) {
    const char *value = getenv(name);
    return value && *value ? value : otherwise;
}

int cores(void) { return (int)sysconf(_SC_NPROCESSORS_ONLN); }

typedef struct {
    void (*fn)(void *, i64);
    void *context;
    i64 n;
    _Atomic i64 next;
} Work;

void *work_loop(void *w) {
    Work *work = w;
    for (i64 i; (i = atomic_fetch_add(&work->next, 1)) < work->n;) {
        work->fn(work->context, i);
    }
    return 0;
}

void parallel(i64 n, int threads, void (*fn)(void *, i64), void *context) {
    Work work = {.fn = fn, .context = context, .n = n};
    if (threads > n) {
        threads = (int)n;
    }
    pthread_t pool[256];
    if (threads > 256) {
        threads = 256;
    }
    for (int t = 1; t < threads; t++) {
        pthread_create(&pool[t], 0, work_loop, &work);
    }
    work_loop(&work);
    for (int t = 1; t < threads; t++) {
        pthread_join(pool[t], 0);
    }
}

int json_shortest(json_object *o, struct printbuf *pb, int level, int flags) {
    f64 v = json_object_get_double(o);
    char s[40];
    if (v == floor(v) && fabs(v) < 1e16) {
        snprintf(s, sizeof(s), "%.1f", v);
    } else {
        for (int digits = 15; digits <= 17; digits++) {
            snprintf(s, sizeof(s), "%.*g", digits, v);
            if (strtod(s, 0) == v) {
                break;
            }
        }
    }
    return printbuf_memappend(pb, s, (int)strlen(s));
}

double decimal_round(double value, int digits) {
    char rounded[96];
    snprintf(rounded, sizeof(rounded), "%.*f", digits, value);
    return strtod(rounded, NULL);
}

json_object *jnum(f64 v) {
    json_object *o = json_object_new_double(v);
    json_object_set_serializer(o, json_shortest, 0, 0);
    return o;
}

json_object *json_read(const char *path) {
    if (!exists(path)) {
        return 0;
    }
    json_object *o = json_object_from_file(path);
    need(o != 0, "cannot parse %s: %s", path, json_util_get_last_err());
    return o;
}

void json_write(const char *path, json_object *o) {
    size_t n;
    const char *text = json_object_to_json_string_length(
        o, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_NOSLASHESCAPE, &n);
    char *line = malloc(n + 1);
    memcpy(line, text, n);
    line[n] = '\n';
    write_file(path, line, n + 1);
    free(line);
}

json_object *jget(json_object *o, const char *key) {
    json_object *v = 0;
    need(o && json_object_object_get_ex(o, key, &v), "missing JSON field %s", key);
    return v;
}

f64 jf64(json_object *o, const char *key) { return json_object_get_double(jget(o, key)); }

i64 ji64(json_object *o, const char *key) { return json_object_get_int64(jget(o, key)); }

const char *jstr(json_object *o, const char *key) { return json_object_get_string(jget(o, key)); }

void digest(const char *name, const void *data, size_t n, u8 *out) {
    unsigned int size;
    need(EVP_Digest(data, n, out, &size, EVP_get_digestbyname(name), 0), "%s failed", name);
}

void sha256_hex(const void *data, size_t n, char *hex) {
    u8 d[32];
    digest("SHA256", data, n, d);
    for (int i = 0; i < 32; i++) {
        snprintf(hex + 2 * i, 3, "%02x", d[i]);
    }
}

char *python_float(Arena *a, f64 v) {
    if (v == 0) {
        return fmt(a, signbit(v) ? "-0.0" : "0.0");
    }
    char digits[40];
    int precision = 1;
    for (; precision <= 17; precision++) {
        snprintf(digits, sizeof(digits), "%.*e", precision - 1, v);
        if (strtod(digits, 0) == v) {
            break;
        }
    }
    char mantissa[40] = {0}, *e = strchr(digits, 'e');
    int exponent = (int)strtol(e + 1, NULL, 10), sign = digits[0] == '-', k = 0;
    for (char *c = digits + sign; c < e; c++) {
        if (*c != '.') {
            mantissa[k++] = *c;
        }
    }
    while (k > 1 && mantissa[k - 1] == '0') {
        mantissa[--k] = 0;
    }
    int point = exponent + 1;
    const char *minus = sign ? "-" : "";
    if (point > 16 || point <= -4) {
        return fmt(a, "%s%c%s%se%c%02d", minus, mantissa[0], k > 1 ? "." : "", mantissa + 1,
            exponent < 0 ? '-' : '+', abs(exponent));
    }
    char zeros[24] = {0};
    if (point <= 0) {
        memset(zeros, '0', -point);
        return fmt(a, "%s0.%s%s", minus, zeros, mantissa);
    }
    if (point >= k) {
        memset(zeros, '0', point - k);
        return fmt(a, "%s%s%s.0", minus, mantissa, zeros);
    }
    return fmt(a, "%s%.*s.%s", minus, point, mantissa, mantissa + point);
}

Bytes zstd_pack(Arena *a, const void *data, size_t n, int level) {
    size_t cap = ZSTD_compressBound(n);
    Bytes out = {.data = push(a, cap)};
    out.size = ZSTD_compress(out.data, cap, data, n, level);
    need(!ZSTD_isError(out.size), "zstd: %s", ZSTD_getErrorName(out.size));
    a->used -= cap - out.size;
    return out;
}

Bytes zstd_unpack(Arena *a, const void *data, size_t n) {
    u64 size = ZSTD_getFrameContentSize(data, n);
    need(size != ZSTD_CONTENTSIZE_ERROR, "not a zstd frame");
    if (size != ZSTD_CONTENTSIZE_UNKNOWN) {
        Bytes out = {.data = push(a, size), .size = size};
        size_t got = ZSTD_decompress(out.data, size, data, n);
        need(got == size, "zstd frame is damaged");
        return out;
    }
    ZSTD_DStream *z = ZSTD_createDStream();
    size_t cap = n * 4 + (1 << 20);
    Bytes out = {.data = malloc(cap)};
    need(z && out.data, "cannot unpack zstd frame: out of memory");
    ZSTD_inBuffer in = {.src = data, .size = n};
    while (in.pos < in.size) {
        if (out.size == cap) {
            cap *= 2;
            u8 *data = realloc(out.data, cap);
            need(data != NULL, "cannot unpack zstd frame: out of memory");
            out.data = data;
        }
        ZSTD_outBuffer o = {.dst = out.data, .size = cap, .pos = out.size};
        size_t r = ZSTD_decompressStream(z, &o, &in);
        need(!ZSTD_isError(r), "zstd: %s", ZSTD_getErrorName(r));
        out.size = o.pos;
    }
    ZSTD_freeDStream(z);
    u8 *kept = push(a, out.size);
    memcpy(kept, out.data, out.size);
    free(out.data);
    return (Bytes){.data = kept, .size = out.size};
}

typedef struct {
    char descr[8];
    int ndim;
    i64 shape[4];
    void *data;
    size_t size;
} Array;

i64 elements(Array x) {
    i64 n = 1;
    for (int d = 0; d < x.ndim; d++) {
        n *= x.shape[d];
    }
    return n;
}

Array npy_parse(Bytes b, const char *path) {
    need(b.size > 10 && !memcmp(b.data, "\x93NUMPY", 6), "%s is not an npy file", path);
    int major = b.data[6];
    size_t header = major == 1
        ? b.data[8] | b.data[9] << 8
        : b.data[8] | b.data[9] << 8 | b.data[10] << 16 | (size_t)b.data[11] << 24;
    size_t start = major == 1 ? 10 : 12;
    char *h = strndup((char *)b.data + start, header);
    Array x = {0};
    char *d = strstr(h, "'descr':");
    need(d && sscanf(d, "'descr': '%7[^']'", x.descr) == 1, "%s: no descr", path);
    need(!strstr(h, "'fortran_order': True"), "%s is Fortran ordered", path);
    char *s = strchr(strstr(h, "'shape':"), '(') + 1;
    while (*s && *s != ')') {
        char *end;
        i64 v = strtoll(s, &end, 10);
        if (end == s) {
            break;
        }
        x.shape[x.ndim++] = v;
        s = end;
        while (*s == ',' || *s == ' ') {
            s++;
        }
    }
    free(h);
    x.data = b.data + start + header;
    x.size = b.size - start - header;
    return x;
}

Array npy_read(Arena *a, const char *path) { return npy_parse(read_file(a, path), path); }

Bytes npy_bytes(Arena *a, Array x) {
    char header[256];
    char shape[128] = "";
    for (int d = 0; d < x.ndim; d++) {
        char one[32];
        snprintf(one, sizeof(one), "%s%lld%s", d ? " " : "", (long long)x.shape[d],
            x.ndim == 1 || d + 1 < x.ndim ? "," : "");
        strcat(shape, one);
    }
    int n = snprintf(header, sizeof(header),
        "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", x.descr, shape);
    int total = (10 + n + 1 + 63) / 64 * 64;
    Bytes b = {.data = push(a, total + x.size), .size = total + x.size};
    memcpy(b.data, "\x93NUMPY\x01\x00", 8);
    b.data[8] = (total - 10) & 255;
    b.data[9] = (total - 10) >> 8;
    memset(b.data + 10, ' ', total - 10);
    memcpy(b.data + 10, header, n);
    b.data[total - 1] = '\n';
    memcpy(b.data + total, x.data, x.size);
    return b;
}

void npy_write(const char *path, Array x) {
    Arena a = arena(x.size + (1 << 20));
    Bytes b = npy_bytes(&a, x);
    write_file(path, b.data, b.size);
    munmap(a.base, a.size);
}

Array npz_member(Arena *a, zip_t *z, const char *path, const char *name, int required) {
    char member[256];
    snprintf(member, sizeof(member), "%s.npy", name);
    zip_int64_t index = zip_name_locate(z, member, 0);
    if (!required && index < 0)
        return (Array){0};
    zip_stat_t st;
    need(index >= 0 && zip_stat_index(z, index, 0, &st) == 0, "%s has no %s", path, member);
    zip_file_t *f = zip_fopen_index(z, index, 0);
    Bytes b = {.data = push(a, st.size), .size = st.size};
    need(f && zip_fread(f, b.data, st.size) == (zip_int64_t)st.size, "cannot read %s:%s", path,
        member);
    zip_fclose(f);
    return npy_parse(b, path);
}

Array npz_get(Arena *a, const char *path, const char *name) {
    zip_t *z = zip_open(path, ZIP_RDONLY, NULL);
    need(z != 0, "cannot open %s", path);
    Array array = npz_member(a, z, path, name, 1);
    zip_close(z);
    return array;
}

void npz_write_codec(
    const char *path, int n, const char **names, Array *arrays, zip_int32_t codec) {
    char part[PATH_MAX];
    snprintf(part, sizeof(part), "%s.part", path);
    mkparent(path);
    unlink(part);
    int error;
    zip_t *z = zip_open(part, ZIP_CREATE | ZIP_TRUNCATE, &error);
    need(z != 0, "cannot write %s", part);
    size_t total = 1 << 20;
    for (int i = 0; i < n; i++) {
        total += arrays[i].size + 4096;
    }
    Arena a = arena(total);
    for (int i = 0; i < n; i++) {
        Bytes b = npy_bytes(&a, arrays[i]);
        zip_source_t *s = zip_source_buffer(z, b.data, b.size, 0);
        char member[256];
        snprintf(member, sizeof(member), "%s.npy", names[i]);
        zip_int64_t at = zip_file_add(z, member, s, ZIP_FL_OVERWRITE);
        need(at >= 0, "cannot add %s to %s", member, part);
        need(zip_set_file_compression(z, at, codec, 0) == 0, "cannot compress %s in %s", member,
            part);
    }
    need(zip_close(z) == 0, "cannot finish %s", part);
    munmap(a.base, a.size);
    need(rename(part, path) == 0, "cannot rename %s", part);
}

void npz_write(const char *path, int n, const char **names, Array *arrays) {
    npz_write_codec(path, n, names, arrays, ZIP_CM_STORE);
}

void tiff_quiet(const char *module, const char *format, va_list args) {}

static void quiet_tiff_warnings(void) { TIFFSetWarningHandler(tiff_quiet); }

TIFF *tiff_open_read(const char *path) {
    static pthread_once_t quiet = PTHREAD_ONCE_INIT;
    pthread_once(&quiet, quiet_tiff_warnings);
    return TIFFOpen(path, "r");
}

f32 *tiff_read(Arena *a, const char *path, int *w, int *h) {
    TIFF *t = tiff_open_read(path);
    need(t != 0, "cannot open %s", path);
    u32 width, height;
    u16 bits = 0, format = 0;
    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLEFORMAT, &format);
    need(bits == 32 && format == SAMPLEFORMAT_IEEEFP, "%s is not float32", path);
    f32 *data = push(a, (size_t)width * height * 4);
    for (u32 row = 0; row < height; row++) {
        need(TIFFReadScanline(t, data + (size_t)row * width, row, 0) == 1, "cannot read %s", path);
    }
    TIFFClose(t);
    *w = width;
    *h = height;
    return data;
}

void tiff_write(const char *path, const f32 *data, int w, int h) {
    char part[PATH_MAX];
    snprintf(part, sizeof(part), "%s.part", path);
    mkparent(path);
    TIFF *t = TIFFOpen(part, "w");
    need(t != 0, "cannot write %s", part);
    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(t, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, h);
    for (int row = 0; row < h; row++) {
        need(TIFFWriteScanline(t, (void *)(data + (size_t)row * w), row, 0) == 1, "cannot write %s",
            part);
    }
    TIFFClose(t);
    need(rename(part, path) == 0, "cannot rename %s", part);
}

typedef struct {
    Arena *arena;
    Bytes out;
    size_t cap;
    const u8 *in;
    size_t left;
} PngIo;

void png_put(png_structp p, png_bytep data, png_size_t n) {
    PngIo *io = png_get_io_ptr(p);
    if (io->out.size + n > io->cap) {
        io->cap = (io->out.size + n) * 2;
        u8 *larger = realloc(io->out.data, io->cap);
        if (!larger) {
            png_error(p, "cannot encode PNG: out of memory");
        }
        io->out.data = larger;
    }
    memcpy(io->out.data + io->out.size, data, n);
    io->out.size += n;
}

void png_get(png_structp p, png_bytep data, png_size_t n) {
    PngIo *io = png_get_io_ptr(p);
    need(n <= io->left, "PNG is truncated");
    memcpy(data, io->in, n);
    io->in += n;
    io->left -= n;
}

void png_flush_nothing(png_structp p) {}

Bytes png_encode(Arena *a, const u8 *grey, int w, int h, int level) {
    png_structp p = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info = png_create_info_struct(p);
    PngIo io = {.arena = a, .cap = (size_t)w * h / 2 + 4096};
    io.out.data = malloc(io.cap);
    need(!setjmp(png_jmpbuf(p)), "PNG encode failed");
    png_set_write_fn(p, &io, png_put, png_flush_nothing);
    png_set_compression_level(p, level);
    png_set_IHDR(p, info, w, h, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(p, info);
    for (int row = 0; row < h; row++) {
        png_write_row(p, grey + (size_t)row * w);
    }
    png_write_end(p, info);
    png_destroy_write_struct(&p, &info);
    Bytes out = {.data = push(a, io.out.size), .size = io.out.size};
    memcpy(out.data, io.out.data, out.size);
    free(io.out.data);
    return out;
}

u8 *png_decode(Arena *a, Bytes b, int *w, int *h) {
    png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info = png_create_info_struct(p);
    PngIo io = {.in = b.data, .left = b.size};
    need(!setjmp(png_jmpbuf(p)), "PNG decode failed");
    png_set_read_fn(p, &io, png_get);
    png_read_info(p, info);
    png_set_strip_16(p);
    png_set_palette_to_rgb(p);
    png_set_expand_gray_1_2_4_to_8(p);
    png_set_rgb_to_gray_fixed(p, 1, -1, -1);
    png_set_strip_alpha(p);
    png_read_update_info(p, info);
    *w = png_get_image_width(p, info);
    *h = png_get_image_height(p, info);
    u8 *grey = push(a, (size_t)*w * *h);
    for (int row = 0; row < *h; row++) {
        png_read_row(p, grey + (size_t)row * *w, 0);
    }
    png_destroy_read_struct(&p, &info, 0);
    return grey;
}

void png_write(const char *path, const u8 *grey, int w, int h) {
    Arena a = arena((size_t)w * h * 2 + (1 << 20));
    Bytes b = png_encode(&a, grey, w, h, 6);
    write_file(path, b.data, b.size);
    munmap(a.base, a.size);
}

u8 *png_read(Arena *a, const char *path, int *w, int *h) {
    return png_decode(a, read_file(a, path), w, h);
}

Bytes jpeg_encode(Arena *a, const u8 *grey, int w, int h, int quality) {
    struct jpeg_compress_struct c;
    struct jpeg_error_mgr e;
    c.err = jpeg_std_error(&e);
    jpeg_create_compress(&c);
    unsigned char *out = 0;
    unsigned long size = 0;
    jpeg_mem_dest(&c, &out, &size);
    c.image_width = w;
    c.image_height = h;
    c.input_components = 1;
    c.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&c);
    jpeg_set_quality(&c, quality, TRUE);
    jpeg_start_compress(&c, TRUE);
    while (c.next_scanline < c.image_height) {
        JSAMPROW row = (JSAMPROW)(grey + (size_t)c.next_scanline * w);
        jpeg_write_scanlines(&c, &row, 1);
    }
    jpeg_finish_compress(&c);
    jpeg_destroy_compress(&c);
    Bytes b = {.data = push(a, size), .size = size};
    memcpy(b.data, out, size);
    free(out);
    return b;
}

u8 *jpeg_decode(Arena *a, Bytes b, int *w, int *h) {
    struct jpeg_decompress_struct d;
    struct jpeg_error_mgr e;
    d.err = jpeg_std_error(&e);
    jpeg_create_decompress(&d);
    jpeg_mem_src(&d, b.data, b.size);
    jpeg_read_header(&d, TRUE);
    d.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&d);
    *w = d.output_width;
    *h = d.output_height;
    u8 *grey = push(a, (size_t)*w * *h);
    while (d.output_scanline < d.output_height) {
        JSAMPROW row = grey + (size_t)d.output_scanline * *w;
        jpeg_read_scanlines(&d, &row, 1);
    }
    jpeg_finish_decompress(&d);
    jpeg_destroy_decompress(&d);
    return grey;
}

typedef struct {
    unsigned __int128 state, inc;
    u32 spare;
    int has_spare;
} Pcg64;

u32 seed_hashmix(u32 value, u32 *hash) {
    value ^= *hash;
    *hash *= 0x931e8875u;
    value *= *hash;
    return value ^ value >> 16;
}

u32 seed_mix(u32 x, u32 y) {
    u32 r = 0xca01f9ddu * x - 0x4973f715u * y;
    return r ^ r >> 16;
}

Pcg64 pcg64(u64 seed) {
    u32 entropy[2] = {(u32)seed, (u32)(seed >> 32)};
    int words = seed >> 32 ? 2 : 1;
    u32 pool[4], hash = 0x43b0d7e5u;
    for (int i = 0; i < 4; i++) {
        pool[i] = seed_hashmix(i < words ? entropy[i] : 0, &hash);
    }
    for (int s = 0; s < 4; s++) {
        for (int d = 0; d < 4; d++) {
            if (s != d) {
                pool[d] = seed_mix(pool[d], seed_hashmix(pool[s], &hash));
            }
        }
    }
    u32 state[8], out_hash = 0x8b51f9ddu;
    for (int i = 0; i < 8; i++) {
        u32 v = pool[i % 4] ^ out_hash;
        out_hash *= 0x58f38dedu;
        v *= out_hash;
        state[i] = v ^ v >> 16;
    }
    u64 s[4];
    for (int i = 0; i < 4; i++) {
        s[i] = (u64)state[2 * i] | (u64)state[2 * i + 1] << 32;
    }
    unsigned __int128 mult = (unsigned __int128)0x2360ed051fc65da4ull << 64 | 0x4385df649fccf645ull;
    unsigned __int128 init = (unsigned __int128)s[0] << 64 | s[1];
    unsigned __int128 seq = (unsigned __int128)s[2] << 64 | s[3];
    Pcg64 r = {.inc = seq << 1 | 1};
    r.state = r.state * mult + r.inc;
    r.state += init;
    r.state = r.state * mult + r.inc;
    return r;
}

u64 pcg64_next(Pcg64 *r) {
    unsigned __int128 mult = (unsigned __int128)0x2360ed051fc65da4ull << 64 | 0x4385df649fccf645ull;
    r->state = r->state * mult + r->inc;
    u64 x = (u64)(r->state >> 64) ^ (u64)r->state;
    unsigned rot = (unsigned)(r->state >> 122);
    return x >> rot | x << ((64 - rot) & 63);
}

u32 pcg64_next32(Pcg64 *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    u64 x = pcg64_next(r);
    r->has_spare = 1;
    r->spare = (u32)(x >> 32);
    return (u32)x;
}

u64 draw(Pcg64 *r, u64 high) {
    u32 range = high - 1;
    if (!range) {
        return 0;
    }
    u32 span = range + 1;
    u64 m = (u64)pcg64_next32(r) * span;
    u32 leftover = m;
    if (leftover < span) {
        u32 threshold = (UINT32_MAX - range) % span;
        while (leftover < threshold) {
            m = (u64)pcg64_next32(r) * span;
            leftover = m;
        }
    }
    return m >> 32;
}

f64 pcg64_double(Pcg64 *r) { return (pcg64_next(r) >> 11) * (1.0 / 9007199254740992.0); }
