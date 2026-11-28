#undef HX_UNITY
#define main unused_pipeline_main
#include "../level/pipeline.c"
#undef main

static void check(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "CUDA unwrap e2e: %s\n", message);
        exit(1);
    }
}

static void fixture_unit(const char *scroll, const char *unit, const char *data) {
    char *folder = NULL, *record = NULL, *store = NULL;
    check(asprintf(&folder, "nodes/%s/scan/nodes/%s", scroll, unit) >= 0 && make_folders(folder) &&
            asprintf(&record, "%s/record.json", folder) >= 0 &&
            asprintf(&store, "%s/nodes.bin", folder) >= 0,
        "cannot prepare unit");
    FILE *f = fopen(record, "w");
    check(f && fputs("{}", f) >= 0 && !fclose(f), "cannot write unit record");
    f = fopen(store, "w");
    check(f && fputs(data, f) >= 0 && !fclose(f), "cannot write node payload");
    free(folder);
    free(record);
    free(store);
}

static char *fixture_read(const char *path) {
    FILE *f = fopen(path, "r");
    check(f != NULL, "cannot read fixture log");
    char *out = calloc(8192, 1);
    check(out && fread(out, 1, 8191, f) && feof(f), "incomplete fixture log");
    fclose(f);
    return out;
}

static void before(const char *log, const char *first, const char *second) {
    const char *a = strstr(log, first), *b = strstr(log, second);
    check(a && b && a < b, second);
}

int main(void) {
    CudaPlace place = {.cpus = 64, .samplers = 6, .gpus = 2};
    cuda_place_limits(&place, 1, 240000000000ULL);
    check(place.reads == 2 && place.reserved == 16 && place.fits == 16 && place.threads == 12,
        "64-core resource plan");
    check(place.draw_cpus == 3 && place.draw_extra == 3500000000ULL,
        "spare fitting resources for winding draws");
    place.cpus = 12;
    cuda_place_limits(&place, 2, 240000000000ULL);
    check(place.reads == 4 && place.reserved == 6 && place.fits == 2 && place.threads == 6,
        "limited resource plan with two reads per GPU");
    cuda_place_limits(&place, 2, 16000000000ULL);
    check(place.draw_cpus == 3 && place.draw_extra == 0, "no extra memory with two fit slots");

    char folder[] = "/tmp/hx-cuda-unwrap-XXXXXX";
    check(mkdtemp(folder) && !chdir(folder), "cannot make fixture directory");
    fixture_unit("A", "rows-0000-0012", "small");
    fixture_unit("A", "rows-0011-0023", "this is a larger node payload");
    fixture_unit("B", "rows-0000-0012", "one");
    fixture_unit("C", "rows-0000-0012", "one");
    fixture_unit("D", "rows-0000-0012", "one");
    const char shim[] =
        "#!/bin/sh\n"
        "case $1 in\n"
        "  level.pipeline)\n"
        "    printf '%s-%s %s\\n' \"$4\" \"$3\" \"${CUDA_VISIBLE_DEVICES:-none}\" >> stages\n"
        "    case $3:$4 in\n"
        "      ink:A) n=0; until grep -q B-ink stages; do\n"
        "        n=$((n+1)); test \"$n\" -lt 100 || exit 8; sleep 0.02; done;;\n"
        "      ink:B) exit 1;;\n"
        "    esac;;\n"
        "  level.level.run_levels)\n"
        "    printf '%s-%s-%s\\n' \"$3\" \"$2\" \"$4\" >> stages\n"
        "    if [ \"$2\" = unit ]; then\n"
        "      printf '%s %s %s %s\\n' \"$3\" \"$4\" \"$HERCULEX_LEVEL_DRAW_CPUS\" "
        "\"$HERCULEX_LEVEL_DRAW_EXTRA\" >> draw-resources\n"
        "    fi\n"
        "    case $2:$3 in unit:C) exit 1;; esac;;\n"
        "  runner.review_page)\n"
        "    echo \"$2-publish\" >> stages\n"
        "    echo \"/sheets/sheet-$2-level/index.html: 7 pieces\";;\n"
        "  *) exit 9;;\n"
        "esac\n";
    FILE *f = fopen("run.sh", "w");
    check(f && fputs(shim, f) >= 0 && !fclose(f) && !chmod("run.sh", 0700),
        "cannot write fixture launcher");
    place = (CudaPlace){.fits = 1,
        .reads = 2,
        .threads = 4,
        .samplers = 6,
        .draw_cpus = 3,
        .draw_extra = 3500000000ULL};
    check(!setenv("CUDA_VISIBLE_DEVICES", "fixture-a,fixture-b", 1) && cuda_devices(&place) == 2,
        "visible device mapping");
#ifdef __linux__
    check(cuda_cpus(&place) > 0 && cuda_memory() > 0, "Linux affinity and memory discovery");
    place.reserved = place.cpus / 2;
#endif
    Scroll scrolls[] = {{"A", "scan"}, {"B", "scan"}, {"C", "scan"}, {"D", "scan"}, {"E", NULL}};
    int output = open("outcomes", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int terminal = dup(STDOUT_FILENO);
    check(output >= 0 && terminal >= 0 && dup2(output, STDOUT_FILENO) >= 0,
        "cannot capture scheduler output");
    close(output);
    int code = cuda_schedule(scrolls, 5, "nodes", "logs", &place);
    fflush(stdout);
    check(dup2(terminal, STDOUT_FILENO) >= 0, "cannot restore scheduler output");
    close(terminal);
    check(code == 1, "a failed scroll must fail the full invocation");
    char *stages = fixture_read("stages"), *outcome = fixture_read("outcomes");
    before(stages, "A-sheets", "B-sheets");
    before(stages, "B-sheets", "C-sheets");
    before(stages, "A-ready-scan", "A-unit-rows-0011-0023");
    before(stages, "A-unit-rows-0011-0023", "A-unit-rows-0000-0012");
    before(stages, "A-unit-rows-0000-0012", "A-finish-scan");
    before(stages, "A-finish-scan", "A-ink");
    before(stages, "A-ink", "A-publish");
    check(strstr(stages, "A-ink fixture-a") && strstr(stages, "B-ink fixture-b"),
        "parallel reads must map to their assigned GPUs");
    check(!strstr(stages, "C-finish") && !strstr(stages, "C-ink") && !strstr(stages, "B-publish"),
        "failed scroll entered a dependent stage");
    char *resources = fixture_read("draw-resources");
    before(resources, "C rows-0000-0012 3 3500000000", "C rows-0000-0012 3 0");
    free(resources);
    check(strstr(outcome, "A published sheet-A-level (7 pieces)") &&
            strstr(outcome, "D published sheet-D-level (7 pieces)") &&
            strstr(outcome, "B ink read failed") && strstr(outcome, "C level fits failed") &&
            strstr(outcome, "B-ink try 2") && strstr(outcome, "C-rows-0000-0012 try 2") &&
            strstr(outcome, "E no curl set or no umbilicus on hand"),
        "missing retry, publication or final outcomes");
    free(stages);
    free(outcome);
    for (int i = 0; i < place.gpus; i++) {
        free(place.gpu[i]);
    }
    free(place.gpu);
    check(!chdir("/"), "cannot leave fixture");
    char *rm[] = {"/bin/rm", "-rf", folder, NULL};
    check(run(rm) == 0, "cannot remove fixture directory");
    return 0;
}
