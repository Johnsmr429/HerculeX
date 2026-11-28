#include "../base.h"

static void frame(const char *bin, const char *input, const char *out, const char *tiers) {
    pid_t pid = fork();
    need(pid >= 0, "cannot fork mesh reference test");
    if (!pid) {
        execl(bin, bin, "calibrate.unwrap.mesh_reference", input, out, tiers, "--by-name", NULL);
        _exit(127);
    }
    int status;
    need(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "mesh reference test failed");
}

int main(int argc, char **argv) {
    need(argc == 2, "native executable required");
    char scratch[] = "/tmp/hx-reference-e2e-XXXXXX";
    need(mkdtemp(scratch) != NULL, "cannot create mesh reference fixture");
    Arena a = arena(8 << 20);
    enum { LENGTH = 1007, N = 3 * LENGTH };
    double *x = push(&a, N * sizeof(*x)), *w = push(&a, N * sizeof(*w));
    double *zero = pushz(&a, N * sizeof(*zero)), *one = push(&a, N * sizeof(*one));
    int32_t *unit = push(&a, N * sizeof(*unit)), mesh[] = {0, 1, 2};
    uint32_t tiers[] = {'A', 'B', 'A'};
    uint32_t names[][8] = {{'a', '-', 'w', '1', '0', '-', '1', '1'},
        {'b', '-', 'w', '1', '2', '-', '1', '3'}, {'c', '-', 'w', '1', '0', '-', '1', '1'}};
    for (int i = 0; i < N; i++) {
        int u = i / LENGTH;
        x[i] = 70. * (i % LENGTH) + (u + 1) * (u + 1) * .135;
        unit[i] = u;
        w[i] = (double[]){.1, .23, .72}[u];
        one[i] = 1;
    }
    const char *keys[] = {
        "x", "y", "z", "nx", "ny", "nz", "W", "unit", "mesh", "tiers", "names", "unit_mesh"};
    Array fields[] = {{"<f8", 1, {N}, x, N * sizeof(*x)}, {"<f8", 1, {N}, zero, N * sizeof(*zero)},
        {"<f8", 1, {N}, zero, N * sizeof(*zero)}, {"<f8", 1, {N}, zero, N * sizeof(*zero)},
        {"<f8", 1, {N}, zero, N * sizeof(*zero)}, {"<f8", 1, {N}, one, N * sizeof(*one)},
        {"<f8", 1, {N}, w, N * sizeof(*w)}, {"<i4", 1, {N}, unit, N * sizeof(*unit)},
        {"<i4", 1, {N}, unit, N * sizeof(*unit)}, {"<U1", 1, {3}, tiers, sizeof(tiers)},
        {"<U8", 1, {3}, names, sizeof(names)}, {"<i4", 1, {3}, mesh, sizeof(mesh)}};
    const char *input = fmt(&a, "%s/input.npz", scratch);
    npz_write_codec(input, sizeof(fields) / sizeof(*fields), keys, fields, ZIP_CM_STORE);
    const char *choices[] = {"A", "AB"};
    for (int i = 0; i < 2; i++) {
        const char *tier = choices[i], *out = fmt(&a, "%s/out-%s.npz", scratch, tier);
        frame(argv[1], input, out, tier);
        Array shifted = npz_get(&a, out, fmt(&a, "Wg_%s", tier));
        Array grouped = npz_get(&a, out, fmt(&a, "group_%s", tier));
        Array keep = npz_get(&a, out, fmt(&a, "keep_%s", tier));
        double *turn = shifted.data;
        int32_t *group = grouped.data;
        uint8_t *held = keep.data;
        need(turn[0] == 10.1 && turn[LENGTH] == (i ? 12.23 : .23) && turn[2 * LENGTH] == 10.72 &&
                group[0] == 0 && group[LENGTH] == (i ? 0 : -1) && group[2 * LENGTH] == 0 &&
                held[0] && !held[LENGTH] && !held[2 * LENGTH],
            "tier priority or name anchor differs for %s", tier);
        json_object *audit = json_object_from_file(fmt(&a, "%s/out-%s.audit.json", scratch, tier));
        json_object *cycles = json_object_object_get(audit, "conflicts");
        json_object *first = cycles ? json_object_array_get_idx(cycles, 0) : NULL;
        json_object *flagged = json_object_object_get(audit, "flagged");
        need(first && flagged && json_object_array_length(cycles) == 1 &&
                json_object_array_length(flagged) == (i ? 4 : 0) &&
                json_object_get_int(json_object_array_get_idx(first, 0)) == 0 &&
                json_object_get_int(json_object_array_get_idx(first, 1)) == 2 &&
                json_object_get_int(json_object_array_get_idx(first, 2)) == 1,
            "conflict or cycle audit differs for %s", tier);
        json_object_put(audit);
    }
    remove_tree(scratch);
    return 0;
}
