#ifndef HX_UNITY
const char hx_usage[] = "Show board job spans, sampled resources and slot usage.";
#endif

#include <fnmatch.h>

typedef struct {
    double start, end;
} TimelineSpan;

typedef struct {
    TimelineSpan *span;
    int n, cap;
} TimelineSpans;

static void timeline_push(TimelineSpans *a, double start, double end) {
    if (start >= end) {
        return;
    }
    if (a->n == a->cap) {
        a->cap = a->cap ? 2 * a->cap : 8;
        void *grown = realloc(a->span, a->cap * sizeof(*a->span));
        need(grown != NULL, "cannot allocate timeline spans");
        a->span = grown;
    }
    a->span[a->n++] = (TimelineSpan){start, end};
}

static int timeline_order(const void *a, const void *b) {
    double x = ((const TimelineSpan *)a)->start, y = ((const TimelineSpan *)b)->start;
    return (x > y) - (x < y);
}

static void timeline_merge(TimelineSpans *a) {
    if (a->n > 1) {
        qsort(a->span, a->n, sizeof(*a->span), timeline_order);
    }
    int n = 0;
    for (int i = 0; i < a->n; i++) {
        TimelineSpan span = a->span[i];
        if (n && span.start <= a->span[n - 1].end) {
            a->span[n - 1].end = fmax(a->span[n - 1].end, span.end);
        } else {
            a->span[n++] = span;
        }
    }
    a->n = n;
}

static void timeline_minus(TimelineSpans *a, const TimelineSpans *cut) {
    TimelineSpans out = {0};
    for (int i = 0, j = 0; i < a->n; i++) {
        double start = a->span[i].start, end = a->span[i].end;
        while (j < cut->n && cut->span[j].end <= start) {
            j++;
        }
        for (int k = j; k < cut->n && cut->span[k].start < end; k++) {
            timeline_push(&out, start, cut->span[k].start);
            start = fmax(start, cut->span[k].end);
        }
        timeline_push(&out, start, end);
    }
    free(a->span);
    *a = out;
}

static double timeline_length(const TimelineSpans *a, double start, double end) {
    double sum = 0;
    for (int i = 0; i < a->n; i++) {
        sum += fmax(0, fmin(end, a->span[i].end) - fmax(start, a->span[i].start));
    }
    return sum;
}

static json_object *timeline_json(const TimelineSpans *a) {
    json_object *out = json_object_new_array();
    for (int i = 0; i < a->n; i++) {
        json_object *pair = json_object_new_array();
        json_object_array_add(pair, jnum(a->span[i].start));
        json_object_array_add(pair, jnum(a->span[i].end));
        json_object_array_add(out, pair);
    }
    return out;
}

static void timeline_lane(const TimelineSpans *a, double start, double end, int width) {
    double step = (end - start) / width;
    for (int i = 0; i < width; i++) {
        double left = start + i * step;
        double covered = timeline_length(a, left, left + step) / step;
        fputs(covered > 0.75 ? "█" : covered > 0.05 ? "▌" : "·", stdout);
    }
}

static TimelineSpan timeline_run(const BoardJob *job, double stamp) {
    double start = job_number(job->row, "started");
    double end = board_active(job->state) ? stamp : job_number(job->row, "ended");
    return end > start && start > 0 ? (TimelineSpan){start, end} : (TimelineSpan){0};
}

static json_object *timeline_samples(Board *b, double start, double end, int width) {
    const char *labels[] = {
        "GPU %", "CPU %", "link MB/s", "internal MB/s", "library MB/s", "outputs MB/s", "room GB"};
    const char *cols[] = {
        "gpu", "cpu_pct", "net_mb_s", "internal_mb_s", "library_mb_s", "outputs_mb_s", "room_gb"};
    const char *blocks[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    double *values = calloc(width, sizeof(*values));
    int *counts = calloc(width, sizeof(*counts));
    need(values != NULL && counts != NULL, "cannot allocate timeline samples");
    json_object *data = json_object_new_object();
    for (int j = 0; j < 7; j++) {
        sqlite3_stmt *q = board_query(b,
            "SELECT time,%s FROM samples "
            "WHERE time>=%f AND time<%f AND %s IS NOT NULL",
            cols[j], start, end, cols[j]);
        memset(values, 0, width * sizeof(*values));
        memset(counts, 0, width * sizeof(*counts));
        while (sqlite3_step(q) == SQLITE_ROW) {
            int i = fmin(
                width - 1, fmax(0, (sqlite3_column_double(q, 0) - start) / (end - start) * width));
            values[i] += sqlite3_column_double(q, 1);
            counts[i]++;
        }
        sqlite3_finalize(q);
        double sum = 0, top = -INFINITY;
        int n = 0;
        for (int i = 0; i < width; i++) {
            if (counts[i]) {
                values[i] /= counts[i];
                sum += values[i];
                top = fmax(top, values[i]);
                n++;
            }
        }
        if (!n) {
            continue;
        }
        json_object *known = json_object_new_array();
        printf("%-14s", labels[j]);
        double scale = j < 2 ? 100 : top ? top : 1;
        for (int i = 0; i < width; i++) {
            int k = fmin(8, fmax(0, rint(values[i] / scale * 8)));
            fputs(counts[i] ? blocks[k] : " ", stdout);
            json_object_array_add(known, counts[i] ? jnum(values[i]) : NULL);
        }
        printf("  mean %.1f, top %.1f\n", sum / n, top);
        json_object_object_add(data, cols[j], known);
    }
    free(values);
    free(counts);
    return data;
}

static json_object *timeline_slots(
    BoardJob *jobs, int n, double start, double end, double stamp, int width) {
    json_object *data = json_object_new_object();
    for (int s = 0; s < 3; s++) {
        TimelineSpans sets[3] = {0};
        for (int i = 0; i < n; i++) {
            BoardJob *j = &jobs[i];
            if (!(j->slots & (1 << s))) {
                continue;
            }
            TimelineSpan ran = timeline_run(j, stamp);
            timeline_push(&sets[0], fmax(start, ran.start), fmin(end, ran.end));
            double began = job_number(j->row, "started");
            if (!began && board_waiting(j)) {
                began = stamp;
            }
            if (began <= j->added) {
                continue;
            }
            double ready = j->added;
            for (int d = 0; d < j->after.count; d++) {
                BoardJob *dep = board_find(jobs, n, j->after.item[d]);
                if (dep) {
                    double ended = job_number(dep->row, "ended");
                    ready = fmax(ready, ended ? ended : stamp);
                }
            }
            ready = fmin(ready, began);
            timeline_push(&sets[1], fmax(start, ready), fmin(end, began));
            timeline_push(&sets[2], fmax(start, j->added), fmin(end, ready));
        }
        for (int k = 0; k < 3; k++) {
            timeline_merge(sets + k);
        }
        timeline_minus(&sets[1], &sets[0]);
        timeline_minus(&sets[2], &sets[0]);
        timeline_minus(&sets[2], &sets[1]);
        printf("%-14s", board_slots[s]);
        timeline_lane(&sets[0], start, end, width);
        char a[64], b[64];
        board_duration(a, timeline_length(&sets[1], start, end));
        board_duration(b, timeline_length(&sets[2], start, end));
        double pct = 100 * timeline_length(&sets[0], start, end) / (end - start);
        printf("  held %.0f%%, idle while a job was ready %s, behind a stage %s\n", pct, a, b);
        json_object *row = json_object_new_object();
        const char *keys[] = {"held", "idle_ready", "idle_behind"};
        for (int k = 0; k < 3; k++) {
            json_object_object_add(row, keys[k], timeline_json(&sets[k]));
            free(sets[k].span);
        }
        json_object_object_add(data, board_slots[s], row);
    }
    return data;
}

static int timeline_events_order(const void *a, const void *b) {
    double x = board_number(*(json_object *const *)a, "start");
    double y = board_number(*(json_object *const *)b, "start");
    return (x > y) - (x < y);
}

static json_object *timeline_events(Board *b, const char *name, double start, double end) {
    char path[PATH_MAX];
    board_path(path, b, "events", name, ".jsonl");
    FILE *f = fopen(path, "r");
    json_object *phases = json_object_new_object();
    if (!f) {
        return phases;
    }
    json_object *rows = json_object_new_array();
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) >= 0) {
        json_object *row = json_tokener_parse(line);
        if (json_object_is_type(row, json_type_object) && *board_text(row, "kind")) {
            json_object_array_add(rows, row);
        } else {
            json_object_put(row);
        }
    }
    free(line);
    fclose(f);
    json_object_array_sort(rows, timeline_events_order);
    for (int i = 0; i < board_count(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *kind = board_text(row, "kind");
        json_object *spans = json_object_object_get(phases, kind);
        if (!spans) {
            spans = json_object_new_array();
            json_object_object_add(phases, kind, spans);
        }
        double a = fmax(start, board_number(row, "start")), z = fmin(end, board_number(row, "end"));
        int n = board_count(spans);
        if (a >= z) {
            continue;
        }
        json_object *last = n ? json_object_array_get_idx(spans, n - 1) : NULL;
        double reached = last ? json_object_get_double(json_object_array_get_idx(last, 1)) : 0;
        if (last && a <= reached) {
            json_object_array_put_idx(last, 1, jnum(fmax(z, reached)));
        } else {
            json_object *pair = json_object_new_array();
            json_object_array_add(pair, jnum(a));
            json_object_array_add(pair, jnum(z));
            json_object_array_add(spans, pair);
        }
    }
    json_object_put(rows);
    return phases;
}

static int timeline_job_order(const void *aa, const void *bb) {
    const BoardJob *a = aa, *b = bb;
    double start_a = job_number(a->row, "started"), start_b = job_number(b->row, "started");
    start_a = start_a ? start_a : a->added;
    start_b = start_b ? start_b : b->added;
    return (start_a > start_b) - (start_a < start_b);
}

static json_object *timeline_jobs(
    Board *b, BoardJob *jobs, int n, double start, double end, double stamp, int width) {
    json_object *data = json_object_new_array(), *phases[40];
    qsort(jobs, n, sizeof(*jobs), timeline_job_order);
    int named = 2, shown = n < 40 ? n : 40;
    for (int i = 0; i < shown; i++) {
        phases[i] = timeline_events(b, jobs[i].name, start, end);
        named = fmax(named, strlen(jobs[i].name) + 2);
        json_object_object_foreach(phases[i], kind, spans) {
            named = fmax(named, strlen(kind) + 4);
        }
    }
    for (int i = 0; i < shown; i++) {
        BoardJob *j = &jobs[i];
        TimelineSpan span = timeline_run(j, stamp);
        TimelineSpans ran = {.span = &span, .n = span.end > span.start};
        char slots[64] = "", took[64];
        for (int s = 0; s < 3; s++) {
            if (j->slots & (1 << s)) {
                size_t size = strlen(slots);
                snprintf(
                    slots + size, sizeof(slots) - size, "%s%s", size ? "+" : "", board_slots[s]);
            }
        }
        if (!*slots) {
            strcpy(slots, "cpu");
        }
        printf("%-*s", named, j->name);
        timeline_lane(&ran, start, end, width);
        board_duration(took, timeline_length(&ran, start, end));
        printf("  %s, %s, %s\n", slots, took, j->state);
        json_object *row = json_object_new_object();
        json_object_object_add(row, "name", json_object_new_string(j->name));
        json_object_object_add(row, "slots", json_object_new_string(slots));
        json_object_object_add(row, "state", json_object_new_string(j->state));
        json_object_object_add(row, "spans", timeline_json(&ran));
        json_object_object_add(row, "phases", phases[i]);
        json_object_array_add(data, row);
        json_object_object_foreach(phases[i], kind, spans) {
            TimelineSpans coverage = {0};
            for (int k = 0; k < board_count(spans); k++) {
                json_object *pair = json_object_array_get_idx(spans, k);
                double a = json_object_get_double(json_object_array_get_idx(pair, 0));
                double z = json_object_get_double(json_object_array_get_idx(pair, 1));
                timeline_push(&coverage, a, z);
            }
            printf("  %-*s", named - 2, kind);
            timeline_lane(&coverage, start, end, width);
            board_duration(took, timeline_length(&coverage, start, end));
            printf("  %s\n", took);
            free(coverage.span);
        }
    }
    if (n > shown) {
        printf("... and %d more jobs\n", n - shown);
    }
    return data;
}

int timeline_command(int argc, char **argv) {
    double hours = 6, stamp = board_time();
    int width = 96;
    const char *pattern = NULL, *out = NULL;
    for (int i = 0; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            puts("runner.timeline [PATTERN] [--hours H] [--width N] [--json OUT]");
            return 0;
        }
        if (!strcmp(arg, "--hours") || !strcmp(arg, "--width") || !strcmp(arg, "--json")) {
            need(i + 1 < argc, "%s needs a value", arg);
            const char *value = argv[++i];
            if (!strcmp(arg, "--hours")) {
                hours = board_value(value);
            } else if (!strcmp(arg, "--width")) {
                width = board_value(value);
            } else {
                out = value;
            }
        } else {
            need(!pattern && arg[0] != '-', "unexpected timeline argument %s", arg);
            pattern = arg;
        }
    }
    need(hours > 0 && width > 0, "timeline hours and width must be positive");
    char path[PATH_MAX];
    board_folder(path);
    Board b = {.path = path};
    board_open(&b);
    int n, count = 0;
    BoardJob *jobs = board_jobs(&b, &n);
    double start = pattern ? INFINITY : stamp - hours * 3600, end = pattern ? 0 : stamp;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        TimelineSpan ran = timeline_run(j, stamp);
        int chosen = pattern ? !fnmatch(pattern, j->name, 0) && j->added
                             : ran.end > start || board_waiting(j);
        if (chosen) {
            if (pattern) {
                double begin = job_number(j->row, "started"), ended = job_number(j->row, "ended");
                start = fmin(start, begin ? begin : j->added);
                end = fmax(end, ran.end ? ran.end : ended ? ended : stamp);
            }
            BoardJob swap = jobs[count];
            jobs[count++] = *j;
            *j = swap;
        }
    }
    need(count, "no board jobs match %s", pattern ? pattern : "None");
    if (pattern) {
        end = fmax(end, start + 60);
    }
    char from[64], to[64], duration[64];
    board_utc(from, start, "%H:%M");
    board_utc(to, end, "%H:%M");
    board_duration(duration, end - start);
    printf("Window %s-%s UTC, %s; one column is %.1f min\n", from, to, duration,
        (end - start) / width / 60);
    json_object *data = json_object_new_object();
    json_object_object_add(data, "start", jnum(start));
    json_object_object_add(data, "end", jnum(end));
    json_object_object_add(data, "width", json_object_new_int(width));
    json_object_object_add(data, "sampled", timeline_samples(&b, start, end, width));
    puts("");
    json_object_object_add(data, "slots", timeline_slots(jobs, count, start, end, stamp, width));
    puts("");
    json_object_object_add(data, "jobs", timeline_jobs(&b, jobs, count, start, end, stamp, width));
    if (out) {
        const char *text = board_json(data);
        write_file(out, text, strlen(text));
    }
    json_object_put(data);
    board_free(jobs, n);
    sqlite3_close(b.db);
    return 0;
}
