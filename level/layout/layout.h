int hx_layout_main(int argc, char **argv);
int hx_seams_main(int argc, char **argv);
char *hx_chosen_scan(const char *sample);

int layout_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "-h") || !strcmp(argv[0], "--help"))) {
        puts("Usage: ./run.sh layout SCROLL [--scan ID]");
        return 0;
    }
    need(argc == 1 || argc == 2 || (argc == 3 && !strcmp(argv[1], "--scan")),
        "usage: layout SCROLL [--scan ID]");
    char *selected = argc == 1 ? hx_chosen_scan(argv[0]) : NULL;
    char *scan = argc == 1 ? selected : argv[argc - 1];
    need(scan != NULL, "no survey CT selected for %s", argv[0]);
    char *args[] = {"layout", argv[0], scan, NULL};
    int code = hx_layout_main(3, args);
    free(selected);
    return code;
}

int hx_layout_rows_main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        puts("Usage: ./run.sh level.layout.mosaic_layout SCROLL SCAN [framed|rows|band]");
        return argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 2;
    }
    char *args[] = {argv[0], argv[1], argv[2], argc == 4 ? argv[3] : "rows", NULL};
    return hx_layout_main(4, args);
}

int hx_layout_framed_main(int argc, char **argv) {
    if (argc != 3) {
        puts("Usage: ./run.sh level.level.scroll_layout SCROLL SCAN");
        return argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 2;
    }
    char *args[] = {argv[0], argv[1], argv[2], "framed", NULL};
    return hx_layout_main(4, args);
}
