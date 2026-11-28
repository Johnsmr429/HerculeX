static int syn_controls_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.controls {list|build|bench} [NAME ...]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc >= 1, "%s", usage);
    if (!strcmp(argv[0], "bench")) {
        argv[0] = "--json";
        return syn_bench_report(argc, argv, "controls");
    }
    int build = !strcmp(argv[0], "build");
    need(build || !strcmp(argv[0], "list"), "%s", usage);
    Arena a = arena(GB);
    const char *root = fmt(&a, "%s/synthetic/controls", OUTPUTS);
    const char *source = fmt(&a, "%s/synthetic/three_d/controls.json", HOME);
    json_object *specs = json_read(source);
    if (!specs) {
        int lock = lock_file(fmt(&a, "%s.lock", source), 1, 1);
        need(lock >= 0, "cannot lock %s", source);
        specs = json_read(source);
        if (!specs) {
            const char *geometry[] = {"fused", "round", "flipped", "crease", "cracked"};
            const char *families[] = {"round/fused", "round/normal", "u-fold/pressed",
                "crease/pressed", "flattened/cracked"};
            const char *looks[] = {"PHerc0139", "PHerc1667"};
            specs = json_object_new_object();
            for (int i = 0; i < 5; i++) {
                for (int l = 0; l < 2; l++) {
                    for (int n = 1; n <= 3; n++) {
                        size_t mark = a.used;
                        json_object *spec =
                            syn_family_draw(&a, families[i], 9000 + n, 256, looks[l]);
                        json_object *roll = jget(spec, "roll"), *sheet = jget(spec, "sheet");
                        double pitch = jf64(roll, "thickness_um") +
                            jf64(roll, "gap_um") * (1 - 0.5 * jf64(roll, "contact"));
                        double outer = 12000 + 1000 * n + 1000 * l;
                        double turns = syn_clip((outer - jf64(roll, "core_um")) / pitch, 30, 110);
                        syn_add(roll, "turns", nearbyint(turns * 100) / 100);
                        if (i == 0) {
                            syn_add(roll, "contact", 0.6);
                            syn_add(roll, "contact_soft", 0.08);
                            syn_add_pair(roll, "contact_corr_um", 30000, 8000);
                            syn_add(roll, "contact_corr_turns", 2);
                            syn_add(sheet, "surface_dip", 0);
                            syn_add(sheet, "ply_dip", fmin(jf64(sheet, "ply_dip"), 0.1));
                        }
                        json_object_object_add(spec, "family",
                            json_object_new_string(fmt(&a, "control/%s", geometry[i])));
                        json_object_object_add(
                            specs, fmt(&a, "%s-%s-s%d", geometry[i], looks[l], n), spec);
                        a.used = mark;
                    }
                }
            }
            json_write(source, specs);
        }
        close(lock);
    }
    json_object_object_foreach(specs, name, spec) {
        size_t mark = a.used;
        int selected = argc == 1;
        for (int i = 1; i < argc; i++) {
            selected |= !strcmp(argv[i], name);
        }
        const char *folder = fmt(&a, "%s/%s", root, name);
        int exists_truth = exists(fmt(&a, "%s/truth.json", folder));
        if (!build) {
            printf("%s %s\n", name, exists_truth ? "built" : "");
        } else if (selected && exists_truth) {
            printf("%s: frozen\n", name);
        } else if (selected) {
            const char *spec_path = fmt(&a, "%s/spec.json", folder);
            json_write(spec_path, spec);
            fflush(NULL);
            pid_t pid = fork();
            if (pid == 0) {
                int quiet = open("/dev/null", O_WRONLY);
                if (quiet >= 0) {
                    dup2(quiet, STDOUT_FILENO);
                    close(quiet);
                }
                execl("./run.sh", "./run.sh", "synthetic.three_d.band", "build", spec_path,
                    "--name", name, "--root", root, NULL);
                _exit(127);
            }
            int status;
            need(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
                    !WEXITSTATUS(status),
                "could not build control %s", name);
            json_object *band = json_read(fmt(&a, "%s/band.json", folder));
            json_object *report = jget(band, "seconds");
            double seconds = 0;
            json_object_object_foreach(report, task, t) {
                if (*task) {
                    seconds += json_object_get_double(t);
                }
            }
            printf("%s: %lld nodes, %.0f s\n", name, (long long)ji64(band, "nodes"), seconds);
            json_object_put(band);
        }
        fflush(stdout);
        a.used = mark;
    }
    json_object_put(specs);
    return 0;
}
