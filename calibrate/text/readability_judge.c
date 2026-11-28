#include <ctype.h>
#include <signal.h>
#include <spawn.h>

extern char **environ;

const char hx_usage_judge[] = "Compare two ink maps of the same passage in a seeded blind reading.";
static const char judge_prompt[] =
    "Two ink-detection maps, A (left) and B (right), of the same patch of a "
    "carbonised Herculaneum papyrus read from a CT scan; each is about 14 mm "
    "across, ink is dark, black at the borders is off the papyrus. Ancient Greek "
    "letters are 2 to 4 mm tall. Which map shows writing more clearly: answer "
    "exactly one of A, B or SAME on the first line. On the second line list any "
    "Greek letters you can name with confidence in the clearer map, or NONE. Do "
    "not guess.";

int hx_site_resize_grey_across_first(const u8 *input, int w, int h, u8 *output, int cols, int rows);

typedef struct {
    u32 x[624];
    int index;
} JudgeRandom;

static JudgeRandom judge_seed(u64 seed) {
    JudgeRandom rng = {.x = {19650218}, .index = 624};
    for (int i = 1; i < 624; i++) {
        rng.x[i] = 1812433253u * (rng.x[i - 1] ^ (rng.x[i - 1] >> 30)) + i;
    }
    u32 words[] = {(u32)seed, (u32)(seed >> 32)};
    int n = seed >> 32 ? 2 : 1, at = 1, word = 0;
    for (int k = 0; k < 624; k++) {
        rng.x[at] =
            (rng.x[at] ^ ((rng.x[at - 1] ^ (rng.x[at - 1] >> 30)) * 1664525u)) + words[word] + word;
        word = (word + 1) % n;
        if (++at >= 624) {
            rng.x[0] = rng.x[623];
            at = 1;
        }
    }
    for (int k = 0; k < 623; k++) {
        rng.x[at] = (rng.x[at] ^ ((rng.x[at - 1] ^ (rng.x[at - 1] >> 30)) * 1566083941u)) - at;
        if (++at >= 624) {
            rng.x[0] = rng.x[623];
            at = 1;
        }
    }
    rng.x[0] = 0x80000000u;
    return rng;
}

static u32 judge_next(JudgeRandom *rng) {
    if (rng->index == 624) {
        for (int i = 0; i < 624; i++) {
            u32 pair = (rng->x[i] & 0x80000000u) | (rng->x[(i + 1) % 624] & 0x7fffffffu);
            rng->x[i] = rng->x[(i + 397) % 624] ^ (pair >> 1) ^ (pair & 1 ? 0x9908b0dfu : 0);
        }
        rng->index = 0;
    }
    u32 value = rng->x[rng->index++];
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    return value ^ (value >> 18);
}

static json_object *judge_ask(Arena *a, const char *image) {
    char *settings =
        fmt(a, "%s", env("HERCULEX_JUDGE", "codex exec --skip-git-repo-check -s read-only -i"));
    size_t words = strlen(settings) / 2 + 3;
    char **args = push(a, words * sizeof(*args));
    int count = 0;
    char *word;
    while ((word = strsep(&settings, " \t\r\n"))) {
        if (*word) {
            args[count++] = word;
        }
    }
    need(count > 0, "HERCULEX_JUDGE holds no command");
    args[count++] = (char *)image;
    args[count] = NULL;
    FILE *input = tmpfile(), *output = tmpfile();
    need(input && output && fputs(judge_prompt, input) >= 0 && !fseek(input, 0, SEEK_SET),
        "cannot compose the judge request");
    posix_spawn_file_actions_t act;
    posix_spawnattr_t process;
    posix_spawn_file_actions_init(&act);
    posix_spawnattr_init(&process);
    posix_spawnattr_setflags(&process, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&process, 0);
    posix_spawn_file_actions_adddup2(&act, fileno(input), STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&act, fileno(output), STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&act, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    int error = posix_spawnp(&pid, args[0], &act, &process, args, environ);
    posix_spawn_file_actions_destroy(&act);
    posix_spawnattr_destroy(&process);
    fclose(input);
    need(!error, "cannot start %s: %s", args[0], strerror(error));
    double deadline = now() + 300;
    for (;;) {
        int status;
        pid_t seen = waitpid(pid, &status, WNOHANG);
        need(seen >= 0 || errno == EINTR, "could not wait for the judge");
        if (seen == pid) {
            break;
        }
        if (now() >= deadline) {
            kill(-pid, SIGKILL);
            waitpid(pid, NULL, 0);
            die("the judge took more than 300 seconds");
        }
        struct timespec short_wait = {.tv_nsec = 20000000};
        nanosleep(&short_wait, NULL);
    }
    need(!fseek(output, 0, SEEK_SET), "cannot read the judge verdict");
    char *line = NULL;
    size_t cap = 0;
    json_object *last[2] = {0};
    while (getline(&line, &cap, output) >= 0) {
        char *begin = line, *end = line + strlen(line);
        while (begin < end && isspace((unsigned char)*begin)) {
            begin++;
        }
        while (end > begin && isspace((unsigned char)end[-1])) {
            end--;
        }
        if (end > begin) {
            json_object_put(last[0]);
            last[0] = last[1];
            last[1] = json_object_new_string_len(begin, (int)(end - begin));
        }
    }
    free(line);
    fclose(output);
    json_object *verdict = json_object_new_array_ext(2);
    for (int i = 0; i < 2; i++) {
        if (last[i]) {
            json_object_array_add(verdict, last[i]);
        }
    }
    return verdict;
}

int readability_judge_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh calibrate.text.readability_judge PAGE FIRST_TITLE SECOND_TITLE "
             "[SEED]");
        return 0;
    }
    need((argc == 3 || argc == 4) && site_name(argv[0]) && !strchr(argv[2], '/'),
        "usage: calibrate.text.readability_judge PAGE FIRST_TITLE SECOND_TITLE [SEED]");
    char *end = NULL;
    u64 seed = 7;
    if (argc == 4) {
        const char *value = argv[3] + (*argv[3] == '-');
        errno = 0;
        seed = strtoull(value, &end, 10);
        need(!errno && end != value && !*end, "invalid judge seed %s", argv[3]);
    }
    JudgeRandom rng = judge_seed(seed);
    Arena a = arena(16 * GB);
    const char *titles[] = {argv[1], argv[2], "same", "unclear"};
    int tally[4] = {0};
    char *label = fmt(&a, "%s", argv[2]);
    for (char *p = label; *p; p++) {
        if (*p == ' ') {
            *p = '-';
        }
    }
    need(site_name(label), "invalid judge label %s", label);
    const char *page = fmt(&a, "%s/website/sheets/%s", OUTPUTS, argv[0]);
    const char *out = fmt(&a, "%s/calibration/readability_judge/%s/%s", LIBRARY, argv[0], label);
    mkdirs(out);
    int count;
    char **names = list_dir(&a, page, &count);
    json_object *rows = json_object_new_array();
    for (int i = 0; i < count; i++) {
        size_t mark = a.used;
        const char *folder = fmt(&a, "%s/%s", page, names[i]);
        struct stat status;
        if (stat(folder, &status) || !S_ISDIR(status.st_mode)) {
            a.used = mark;
            continue;
        }
        int files_n;
        char **files = list_dir(&a, folder, &files_n);
        const char *paths[2] = {0};
        for (int k = 0; k < files_n; k++) {
            const char *dash = strchr(files[k], '-'), *ext = strrchr(files[k], '.');
            if (!dash || !ext || dash >= ext || strcmp(ext, ".png") != 0) {
                continue;
            }
            char *title = fmt(&a, "%.*s", (int)(ext - dash - 1), dash + 1);
            for (char *p = title; *p; p++) {
                if (*p == '-') {
                    *p = ' ';
                }
            }
            for (int t = 0; t < 2; t++) {
                if (!strcmp(title, titles[t])) {
                    paths[t] = fmt(&a, "%s/%s", folder, files[k]);
                }
            }
        }
        if (!paths[0] || !paths[1]) {
            a.used = mark;
            continue;
        }
        int swap = judge_next(&rng) < 0x80000000u;
        judge_next(&rng);
        int w[2], h[2];
        u8 *left = png_read(&a, paths[swap], w, h);
        u8 *right = png_read(&a, paths[1 - swap], w + 1, h + 1);
        int height = h[0] > h[1] ? h[0] : h[1], width = w[0] + 16 + w[1];
        u8 *pair = pushz(&a, (i64)width * height);
        for (int y = 0; y < height; y++) {
            u8 *line = pair + (i64)y * width;
            if (y < h[0]) {
                memcpy(line, left + (i64)y * w[0], w[0]);
            }
            memset(line + w[0], 128, 16);
            if (y < h[1]) {
                memcpy(line + w[0] + 16, right + (i64)y * w[1], w[1]);
            }
        }
        double scale = 1024 / fmax(width, height);
        int ww = (int)nearbyint(width * scale), hh = (int)nearbyint(height * scale);
        u8 *resized = push(&a, (i64)ww * hh);
        need(ww > 0 && hh > 0 &&
                hx_site_resize_grey_across_first(pair, width, height, resized, ww, hh),
            "cannot resize judge panels for %s", names[i]);
        const char *image = fmt(&a, "%s/%s.png", out, names[i]);
        png_write(image, resized, ww, hh);
        json_object *verdict = judge_ask(&a, image);
        const char *line = json_object_get_string(json_object_array_get_idx(verdict, 0));
        int vote = line ? toupper((unsigned char)*line) : 0;
        int winner = vote == 'A' ? swap : vote == 'B' ? 1 - swap : vote == 'S' ? 2 : 3;
        tally[winner]++;
        json_object *row = json_object_new_object();
        json_object_object_add(row, "passage", json_object_new_string(names[i]));
        json_object_object_add(row, "swapped", json_object_new_boolean(swap));
        json_object_object_add(row, "verdict", verdict);
        json_object_object_add(row, "clearer", json_object_new_string(titles[winner]));
        json_object_array_add(rows, row);
        puts(json_object_to_json_string_ext(row, JSON_C_TO_STRING_PLAIN));
        fflush(stdout);
        a.used = mark;
    }
    json_write(fmt(&a, "%s/verdicts.json", out), rows);
    json_object *summary = json_object_new_object();
    for (int i = 0; i < 4; i++) {
        json_object_object_add(summary, titles[i], json_object_new_int(tally[i]));
    }
    puts(json_object_to_json_string_ext(summary, JSON_C_TO_STRING_PLAIN));
    json_object_put(rows);
    json_object_put(summary);
    return 0;
}
