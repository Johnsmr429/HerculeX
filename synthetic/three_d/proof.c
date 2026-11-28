static int syn_proof_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.proof TAG";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc == 1 && *argv[0] && !strchr(argv[0], '/'), "%s", usage);
    Arena a = arena(64 << 20);
    const char *tag = argv[0], *root = fmt(&a, "%s/synthetic/realism", OUTPUTS);
    FILE *file = fopen(fmt(&a, "%s/discriminator.jsonl", root), "r");
    json_object *cnn = json_object_new_object();
    char *line = NULL;
    size_t cap = 0;
    while (file && getline(&line, &cap, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        const char *mark = json_object_get_string(json_object_object_get(row, "tag"));
        if (mark && !strcmp(mark, tag)) {
            json_object_object_add(cnn, jstr(row, "look"), json_object_get(jget(row, "auc")));
        }
        json_object_put(row);
    }
    free(line);
    if (file) {
        fclose(file);
    }
    json_object *out = json_object_new_object();
    const char *looks[] = {"PHerc0139", "PHerc1667", "PHerc0172", "PHercParis4"};
    for (int i = 0; i < 4; i++) {
        const char *look = looks[i];
        json_object *row = json_object_new_object(), *measured = json_object_object_get(cnn, look);
        int cnn_ok = measured && json_object_get_double(measured) <= 0.75;
        json_object_object_add(row, "cnn_auc", json_object_get(measured));
        json_object_object_add(row, "cnn_pass", measured ? json_object_new_boolean(cnn_ok) : NULL);
        json_object *actual = json_read(fmt(&a, "%s/solver/compare-%s-%s.json", root, look, tag));
        json_object *solver = NULL;
        int solver_ok = 0;
        const char *description = "solver view missing";
        if (actual) {
            solver = json_object_new_object();
            const char *keys[] = {"auc_logistic", "auc_network", "auc_within_band_network"};
            double auc[3];
            for (int k = 0; k < 3; k++) {
                auc[k] = jf64(actual, keys[k]);
                json_object_object_add(solver, keys[k], json_object_get(jget(actual, keys[k])));
            }
            double worst = -INFINITY;
            const char *name = NULL;
            json_object *features = jget(actual, "features");
            json_object_object_foreach(features, key, feature) {
                double ks = jf64(feature, "ks");
                if (ks > worst) {
                    worst = ks;
                    name = key;
                }
            }
            need(name != NULL, "no solver features for %s", look);
            json_object_object_add(solver, "worst_feature", json_object_new_string(name));
            syn_add(solver, "worst_ks", worst);
            json_object_object_add(solver, "bands", json_object_get(jget(actual, "bands")));
            solver_ok = fmax(auc[0], fmax(auc[1], auc[2])) <= 0.70 && worst <= 0.25;
            description = fmt(&a, "solver AUC %.2f/%.2f/%.2f, worst %s KS %.2f (%lld bands)",
                auc[0], auc[1], auc[2], name, worst, (long long)ji64(actual, "bands"));
        }
        json_object_object_add(row, "solver", solver);
        json_object_object_add(
            row, "solver_pass", actual ? json_object_new_boolean(solver_ok) : NULL);
        json_object_object_add(row, "passes", json_object_new_boolean(cnn_ok && solver_ok));
        json_object_object_add(out, look, row);
        const char *note =
            measured ? fmt(&a, "CNN AUC %.3f", json_object_get_double(measured)) : "CNN missing";
        printf(
            "%-12s %s  %s; %s\n", look, cnn_ok && solver_ok ? "PASS" : "fail", note, description);
        json_object_put(actual);
    }
    json_write(fmt(&a, "%s/proof-%s.json", root, tag), out);
    json_object_put(cnn);
    json_object_put(out);
    return 0;
}
