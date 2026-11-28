#include "base.h"

int main(int argc, char **argv) {
    need(argc == 2, "native executable is required");
    char folder[] = "/tmp/hx-syn-golden-XXXXXX";
    need(mkdtemp(folder) != NULL, "cannot make synthetic fixture directory");
    Arena a = arena(8 << 20);
    char *spec = fmt(&a, "%s/spec.json", folder), *out = fmt(&a, "%s/band", folder);
    const char setup[] =
        "{\"seed\":5,\"band_um\":[0.0,620.0],\"margin_um\":300.0,"
        "\"roll\":{\"turns\":3.25,\"per_turn\":67,\"core_um\":500,\"contact\":0.25,"
        "\"delamination\":0.2,\"peel_turns\":0.8,\"peel_extent_deg\":100},"
        "\"sheet\":{\"holes\":0.1,\"fog_um\":12.0,\"debris\":0.6,\"grain\":0.4,\"grain_um\":25.0},"
        "\"ct\":{\"noise_air\":0.3,\"rings\":3.0,\"ring_um\":15.0},"
        "\"shape\":{\"crush\":{\"kappa\":0.45,\"kappa_core\":0.2},\"harmonics\":[[3,0.1,20]]},"
        "\"folds\":[{\"angle_deg\":125,\"direction_deg\":15,\"position_um\":650,\"offset_um\":850}]"
        ","
        "\"flows\":[{\"scale_um\":3000,\"amp_um\":100},"
        "{\"kind\":\"swirl\",\"angle_deg\":45,\"radius_um\":600}],"
        "\"cracks\":[{\"centre_um\":[0,700],\"angle_deg\":80,\"half_length_um\":600,"
        "\"open_um\":45,\"slip_um\":80,\"wiggle_um\":10}]}";
    write_file(spec, setup, sizeof(setup) - 1);
    setenv("HERCULEX_SYNTHETIC_THREADS", "2", 1);
    pid_t pid = fork();
    if (!pid) {
        execl(argv[1], argv[1], "synthetic.three_d.band", "render", spec, out, "--level", "2",
            "--device", "cpu", NULL);
        _exit(127);
    }
    int status;
    need(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native synthetic fixture did not render");
    Bytes raw = read_file(&a, fmt(&a, "%s/bricks.bin", out));
    Array offsets = npy_read(&a, fmt(&a, "%s/offsets.npy", out));
    const i64 expected[] = {0, -2};
    char hash[65];
    sha256_hex(raw.data, raw.size, hash);
    int matched = raw.size == 128 * 128 * 128 && offsets.size == sizeof(expected) &&
        !memcmp(offsets.data, expected, sizeof(expected)) &&
        !strcmp(hash, "ee8893dfcf124e6f2ecf0e0b291679220dc83fd8c3f34b135cb47e1f401f6011");
    remove_tree(folder);
    need(
        matched, "synthetic CT or brick offsets differed from the folded-band reference: %s", hash);
    return 0;
}
