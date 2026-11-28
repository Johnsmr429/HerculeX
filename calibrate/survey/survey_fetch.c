#ifndef HX_UNITY
const char hx_usage[] = "Fetch the CT planes needed by the survey plan.";
#endif

int survey_fetch_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.survey.survey_fetch look|replicate|reread|row-sample|second "
             "SAMPLE/SCAN/BAND");
        return !argc;
    }
    need(argc == 2 && survey_kind(argv[0]) != SURVEY_ALL, "survey_fetch needs a mode and band");
    SurveyTask task = survey_task(argv[1], survey_kind(argv[0]));
    printf("%s %s %d pieces planned: ", task.band.sample, argv[0], task.count);
    for (int i = 0; i < task.count; i++) {
        const char *name = disk_base(task.jobs[i].folder);
        printf("%s%s", i ? "," : "", !strncmp(name, "piece-", 6) ? name + 6 : name);
    }
    puts("");
    fflush(stdout);
    survey_fetch_start(&task, 1);
    int status = task.fetch ? survey_wait(task.fetch) : 0;
    munmap(task.a.base, task.a.size);
    if (status && status != 75) {
        fprintf(stderr, "try later: survey fetch ended with code %d\n", status);
    }
    return status ? 75 : 0;
}
