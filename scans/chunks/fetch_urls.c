#include <curl/curl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int64_t index;
    int64_t offset;
    int64_t length;
    int32_t status;
    int32_t attempts;
} lg_body;

typedef struct result {
    struct result *next;
    uint8_t *data;
    int64_t index;
    int64_t length;
    int32_t status;
    int32_t attempts;
} result;

typedef struct {
    CURL *easy;
    uint8_t *data;
    size_t length;
    size_t capacity;
    size_t max_body;
    int64_t index;
    int32_t attempt;
    int sized;
    char url[2048];
} slot;

typedef struct {
    double due;
    int64_t index;
    int32_t attempt;
} retry;

enum { DELIVERED, ABSENT, FAILED, BYTES, RETRIES, ACTIVE, QUEUED, CONNECTS, NEEDED, ABORTED, RETRY_CAUSE, STATS };

typedef struct {
    char *base;
    char *separator;
    int64_t *keys;
    int64_t *ranges;
    int64_t count;
    int64_t next;
    int32_t connections;
    int32_t retries;
    int64_t memory_cap;
    CURLM *multi;
    slot *slots;
    retry *pending;
    int64_t pending_count;
    int64_t pending_capacity;
    int64_t streak;
    int64_t streak_limit;
    pthread_t thread;
    int joinable;
    pthread_mutex_t lock;
    pthread_cond_t ready;
    result *head;
    result *tail;
    int64_t queued_count;
    int64_t queued_bytes;
    int64_t stats[STATS];
    int finished;
    int stop;
} lg_fetch;

enum { RAMP_START = 16, RAMP_PER_SECOND = 320 };

static pthread_once_t global_once = PTHREAD_ONCE_INIT;
static void global_init(void) { curl_global_init(CURL_GLOBAL_DEFAULT); }

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int reserve(slot *s, size_t needed) {
    if (needed <= s->capacity) return 1;
    size_t capacity = s->capacity ? s->capacity : 65536;
    while (capacity < needed) capacity *= 2;
    uint8_t *grown = realloc(s->data, capacity);
    if (!grown) return 0;
    s->data = grown;
    s->capacity = capacity;
    return 1;
}

static size_t on_body(char *data, size_t size, size_t count, void *user) {
    slot *s = user;
    size_t n = size * count;
    if (!s->sized) {
        s->sized = 1;
        curl_off_t announced = -1;
        curl_easy_getinfo(s->easy, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &announced);
        if (announced > 0 && (size_t)announced <= s->max_body && (size_t)announced > s->capacity) {
            uint8_t *exact = realloc(s->data, (size_t)announced);
            if (!exact) return 0;
            s->data = exact;
            s->capacity = (size_t)announced;
        }
    }
    if (s->length + n > s->max_body || !reserve(s, s->length + n)) return 0;
    memcpy(s->data + s->length, data, n);
    s->length += n;
    return n;
}

static int push(lg_fetch *f, slot *s, int32_t status, int keep_body) {
    result *r = calloc(1, sizeof(result));
    if (!r) return 0;
    r->index = s->index;
    r->status = status;
    r->attempts = s->attempt + 1;
    if (keep_body) {
        r->data = s->data;
        r->length = (int64_t)s->length;
        s->data = NULL;
        s->capacity = 0;
    }
    s->length = 0;
    pthread_mutex_lock(&f->lock);
    if (f->tail) f->tail->next = r;
    else f->head = r;
    f->tail = r;
    f->queued_count += 1;
    f->queued_bytes += r->length;
    if (status == 200) {
        f->stats[DELIVERED] += 1;
        f->stats[BYTES] += r->length;
    } else if (status == 403 || status == 404) {
        f->stats[ABSENT] += 1;
    } else {
        f->stats[FAILED] += 1;
    }
    f->stats[QUEUED] = f->queued_count;
    pthread_cond_broadcast(&f->ready);
    pthread_mutex_unlock(&f->lock);
    return 1;
}

static int retryable(CURLcode code, long http) {
    if (code != CURLE_OK)
        return code != CURLE_URL_MALFORMAT && code != CURLE_UNSUPPORTED_PROTOCOL && code != CURLE_WRITE_ERROR;
    return http >= 500 || http == 429 || http == 408 || http == 0;
}

static int32_t status_of(CURLcode code, long http) { return code != CURLE_OK ? -(int32_t)code : (int32_t)http; }

static int schedule_retry(lg_fetch *f, slot *s, double now, int32_t cause) {
    if (f->pending_count == f->pending_capacity) {
        retry *grown = realloc(f->pending, 2 * (size_t)f->pending_capacity * sizeof(retry));
        if (!grown) return 0;
        f->pending = grown;
        f->pending_capacity *= 2;
    }
    double backoff = 0.5 * (double)(1 << (s->attempt < 6 ? s->attempt : 6));
    if (backoff > 30.0) backoff = 30.0;
    backoff *= 0.5 + (double)arc4random_uniform(1000) / 1000.0;
    f->pending[f->pending_count++] = (retry){now + backoff, s->index, s->attempt + 1};
    s->length = 0;
    pthread_mutex_lock(&f->lock);
    f->stats[RETRIES] += 1;
    f->stats[RETRY_CAUSE] = cause;
    pthread_mutex_unlock(&f->lock);
    return 1;
}

static int next_due(lg_fetch *f, double now, int64_t *index, int32_t *attempt) {
    for (int64_t i = 0; i < f->pending_count; ++i) {
        if (f->pending[i].due <= now) {
            *index = f->pending[i].index;
            *attempt = f->pending[i].attempt;
            f->pending[i] = f->pending[--f->pending_count];
            return 1;
        }
    }
    return 0;
}

static void begin(lg_fetch *f, slot *s, int64_t index, int32_t attempt) {
    const int64_t *k = f->keys + 3 * index;
    snprintf(s->url, sizeof s->url, "%s/%lld%s%lld%s%lld", f->base, (long long)k[0], f->separator, (long long)k[1],
             f->separator, (long long)k[2]);
    s->index = index;
    s->attempt = attempt;
    s->length = 0;
    s->sized = 0;
    curl_easy_setopt(s->easy, CURLOPT_URL, s->url);
    if (f->ranges) {
        char range[48];
        snprintf(range, sizeof range, "%lld-%lld", (long long)f->ranges[2 * index],
                 (long long)f->ranges[2 * index + 1]);
        curl_easy_setopt(s->easy, CURLOPT_RANGE, range);
    }
    curl_multi_add_handle(f->multi, s->easy);
}

static void abort_fetch(lg_fetch *f) {
    pthread_mutex_lock(&f->lock);
    f->stats[ABORTED] = 1;
    pthread_mutex_unlock(&f->lock);
    __atomic_store_n(&f->stop, 1, __ATOMIC_RELEASE);
}

static void finish(lg_fetch *f, slot *s, CURLcode code, double now) {
    long http = 0;
    long connects = 0;
    curl_easy_getinfo(s->easy, CURLINFO_RESPONSE_CODE, &http);
    curl_easy_getinfo(s->easy, CURLINFO_NUM_CONNECTS, &connects);
    curl_multi_remove_handle(f->multi, s->easy);
    pthread_mutex_lock(&f->lock);
    f->stats[CONNECTS] += connects;
    pthread_mutex_unlock(&f->lock);
    int delivered = code == CURLE_OK && (http == 200 || http == 206);
    int answered = delivered || (code == CURLE_OK && (http == 403 || http == 404));
    f->streak = answered ? 0 : f->streak + 1;
    int kept;
    if (delivered) kept = push(f, s, 200, 1);
    else if (answered) kept = push(f, s, (int32_t)http, 0);
    else if (retryable(code, http) && s->attempt < f->retries) kept = schedule_retry(f, s, now, status_of(code, http));
    else kept = push(f, s, status_of(code, http), 0);
    s->index = -1;
    if (!kept || f->streak > f->streak_limit) abort_fetch(f);
}

static int held_over_cap(lg_fetch *f) {
    pthread_mutex_lock(&f->lock);
    int over = f->queued_bytes >= f->memory_cap;
    pthread_mutex_unlock(&f->lock);
    return over;
}

static void *run(void *argument) {
    lg_fetch *f = argument;
    int active = 0;
    double started = now_seconds();
    while (!__atomic_load_n(&f->stop, __ATOMIC_ACQUIRE)) {
        double now = now_seconds();
        int over = held_over_cap(f);
        double ramp = RAMP_START + (now - started) * RAMP_PER_SECOND;
        int allowed = ramp < f->connections ? (int)ramp : f->connections;
        for (int i = 0; i < f->connections && !over && active < allowed; ++i) {
            slot *s = &f->slots[i];
            if (s->index >= 0) continue;
            int64_t index;
            int32_t attempt;
            if (!next_due(f, now, &index, &attempt)) {
                if (f->next >= f->count) break;
                index = f->next++;
                attempt = 0;
            }
            begin(f, s, index, attempt);
            active += 1;
        }
        if (active == 0 && f->next >= f->count && f->pending_count == 0) break;
        int running = 0;
        curl_multi_perform(f->multi, &running);
        CURLMsg *message;
        int left;
        now = now_seconds();
        while ((message = curl_multi_info_read(f->multi, &left))) {
            if (message->msg != CURLMSG_DONE) continue;
            slot *s = NULL;
            curl_easy_getinfo(message->easy_handle, CURLINFO_PRIVATE, (char **)&s);
            finish(f, s, message->data.result, now);
            active -= 1;
        }
        pthread_mutex_lock(&f->lock);
        f->stats[ACTIVE] = active;
        pthread_mutex_unlock(&f->lock);
        if (active == 0 && f->next >= f->count && f->pending_count == 0) break;
        int wait_ms = 100;
        for (int64_t i = 0; i < f->pending_count; ++i) {
            int until = (int)((f->pending[i].due - now) * 1000.0) + 1;
            if (until < wait_ms) wait_ms = until < 1 ? 1 : until;
        }
        curl_multi_poll(f->multi, NULL, 0, wait_ms, NULL);
    }
    for (int i = 0; i < f->connections; ++i) {
        if (f->slots[i].index >= 0) curl_multi_remove_handle(f->multi, f->slots[i].easy);
        f->slots[i].index = -1;
    }
    pthread_mutex_lock(&f->lock);
    f->stats[ACTIVE] = 0;
    f->finished = 1;
    pthread_cond_broadcast(&f->ready);
    pthread_mutex_unlock(&f->lock);
    return NULL;
}

void lg_fetch_close(lg_fetch *f);

lg_fetch *lg_fetch_keys_start(const char *base, const char *separator, const int64_t *keys, int64_t count,
                              int32_t connections, int32_t timeout_ms, int32_t retries, int64_t memory_cap,
                              int64_t max_body, const int64_t *ranges) {
    pthread_once(&global_once, global_init);
    if (connections < 1 || count < 0 || strlen(base) + 3 * (strlen(separator) + 20) >= sizeof(((slot *)0)->url))
        return NULL;
    lg_fetch *f = calloc(1, sizeof(lg_fetch));
    if (!f) return NULL;
    f->base = strdup(base);
    f->separator = strdup(separator);
    f->keys = malloc((size_t)(count > 0 ? count : 1) * 3 * sizeof(int64_t));
    f->ranges = ranges ? malloc((size_t)(count > 0 ? count : 1) * 2 * sizeof(int64_t)) : NULL;
    f->pending = malloc((size_t)connections * sizeof(retry));
    f->slots = calloc((size_t)connections, sizeof(slot));
    f->multi = curl_multi_init();
    if (!f->base || !f->separator || !f->keys || (ranges && !f->ranges) || !f->pending || !f->slots || !f->multi) {
        free(f->base);
        free(f->separator);
        free(f->keys);
        free(f->ranges);
        free(f->pending);
        free(f->slots);
        if (f->multi) curl_multi_cleanup(f->multi);
        free(f);
        return NULL;
    }
    memcpy(f->keys, keys, (size_t)count * 3 * sizeof(int64_t));
    if (ranges) memcpy(f->ranges, ranges, (size_t)count * 2 * sizeof(int64_t));
    f->count = count;
    f->connections = connections;
    f->retries = retries;
    f->memory_cap = memory_cap;
    f->pending_capacity = connections;
    f->streak_limit = 4 * (int64_t)connections > 256 ? 4 * (int64_t)connections : 256;
    curl_multi_setopt(f->multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, (long)connections);
    curl_multi_setopt(f->multi, CURLMOPT_MAX_HOST_CONNECTIONS, (long)connections);
    curl_multi_setopt(f->multi, CURLMOPT_MAXCONNECTS, (long)connections);
    for (int i = 0; i < connections; ++i) {
        slot *s = &f->slots[i];
        s->index = -1;
        s->max_body = (size_t)max_body;
        s->easy = curl_easy_init();
        curl_easy_setopt(s->easy, CURLOPT_PRIVATE, (char *)s);
        curl_easy_setopt(s->easy, CURLOPT_WRITEFUNCTION, on_body);
        curl_easy_setopt(s->easy, CURLOPT_WRITEDATA, s);
        curl_easy_setopt(s->easy, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(s->easy, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(s->easy, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(s->easy, CURLOPT_CONNECTTIMEOUT_MS, 20000L);
        curl_easy_setopt(s->easy, CURLOPT_TIMEOUT_MS, (long)timeout_ms);
        curl_easy_setopt(s->easy, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(s->easy, CURLOPT_LOW_SPEED_TIME, 20L);
        curl_easy_setopt(s->easy, CURLOPT_USERAGENT, "herculex-chunks/1");
    }
    pthread_mutex_init(&f->lock, NULL);
    pthread_cond_init(&f->ready, NULL);
    f->joinable = pthread_create(&f->thread, NULL, run, f) == 0;
    if (!f->joinable) {
        lg_fetch_close(f);
        return NULL;
    }
    return f;
}

static int queue_ready(lg_fetch *f, int64_t capacity, int64_t max_bodies) {
    return f->finished || f->queued_count >= max_bodies || f->queued_bytes >= capacity;
}

int64_t lg_fetch_take(lg_fetch *f, uint8_t *buffer, int64_t capacity, lg_body *bodies, int64_t max_bodies,
                      int32_t wait_ms) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += wait_ms / 1000;
    deadline.tv_nsec += (long)(wait_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&f->lock);
    while (!queue_ready(f, capacity, max_bodies)) {
        if (pthread_cond_timedwait(&f->ready, &f->lock, &deadline) != 0) break;
    }
    result *taken = NULL;
    result *last = NULL;
    int64_t n = 0;
    int64_t used = 0;
    while (f->head && n < max_bodies && used + f->head->length <= capacity) {
        result *r = f->head;
        f->head = r->next;
        r->next = NULL;
        if (last) last->next = r;
        else taken = r;
        last = r;
        used += r->length;
        n += 1;
        f->queued_count -= 1;
        f->queued_bytes -= r->length;
    }
    if (!f->head) f->tail = NULL;
    if (n == 0 && f->head) f->stats[NEEDED] = f->head->length;
    int64_t outcome = n > 0 ? n : (f->head ? -2 : (f->finished ? -1 : 0));
    f->stats[QUEUED] = f->queued_count;
    int running = !f->finished;
    pthread_mutex_unlock(&f->lock);
    int64_t offset = 0;
    int64_t i = 0;
    for (result *r = taken; r;) {
        if (r->length) memcpy(buffer + offset, r->data, (size_t)r->length);
        bodies[i++] = (lg_body){r->index, offset, r->length, r->status, r->attempts};
        offset += r->length;
        result *next = r->next;
        free(r->data);
        free(r);
        r = next;
    }
    if (n > 0 && running) curl_multi_wakeup(f->multi);
    return outcome;
}

void lg_fetch_stats(lg_fetch *f, int64_t *stats) {
    pthread_mutex_lock(&f->lock);
    memcpy(stats, f->stats, sizeof(f->stats));
    pthread_mutex_unlock(&f->lock);
}

void lg_fetch_stop(lg_fetch *f) {
    __atomic_store_n(&f->stop, 1, __ATOMIC_RELEASE);
    curl_multi_wakeup(f->multi);
}

void lg_fetch_close(lg_fetch *f) {
    lg_fetch_stop(f);
    if (f->joinable) pthread_join(f->thread, NULL);
    for (result *r = f->head; r;) {
        result *next = r->next;
        free(r->data);
        free(r);
        r = next;
    }
    for (int i = 0; i < f->connections; ++i) {
        curl_easy_cleanup(f->slots[i].easy);
        free(f->slots[i].data);
    }
    curl_multi_cleanup(f->multi);
    pthread_mutex_destroy(&f->lock);
    pthread_cond_destroy(&f->ready);
    free(f->slots);
    free(f->pending);
    free(f->keys);
    free(f->ranges);
    free(f->separator);
    free(f->base);
    free(f);
}
