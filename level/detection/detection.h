int hx_detection_main(int argc, char **argv);

int sheets_command(int argc, char **argv) {
    if (argc &&
        (!strcmp(argv[0], "run") || !strcmp(argv[0], "band") || !strcmp(argv[0], "store") ||
            !strcmp(argv[0], "-h") || !strcmp(argv[0], "--help"))) {
        char *args[argc + 2];
        args[0] = "sheets";
        memcpy(args + 1, argv, (argc + 1) * sizeof(*argv));
        return hx_detection_main(argc + 1, args);
    }
    char *args[argc + 3];
    args[0] = "sheets";
    args[1] = "run";
    memcpy(args + 2, argv, (argc + 1) * sizeof(*argv));
    return hx_detection_main(argc + 2, args);
}
