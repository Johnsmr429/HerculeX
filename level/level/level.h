int hx_levels_main(int argc, char **argv);
int hx_umbilicus_main(int argc, char **argv);

int pieces_command(int argc, char **argv) {
    char *args[argc + 3];
    args[0] = "pieces";
    args[1] = "run";
    memcpy(args + 2, argv, (argc + 1) * sizeof(*argv));
    return hx_levels_main(argc + 2, args);
}

int fit_command(int argc, char **argv) {
    need(argc == 3, "usage: fit SCROLL SCAN UNIT");
    char *args[] = {"fit", "unit", argv[0], argv[2], argv[1], 0};
    return hx_levels_main(5, args);
}

int curl_command(int argc, char **argv) {
    need(argc == 2 || argc == 3, "usage: curl SCROLL SCAN [1|-1]");
    char *args[] = {"curl", "curl", argv[0], argv[1], argc == 3 ? argv[2] : 0, 0};
    return hx_levels_main(argc + 2, args);
}

int umbilicus_command(int argc, char **argv) {
    char *args[argc + 2];
    args[0] = "umbilicus";
    memcpy(args + 1, argv, (argc + 1) * sizeof(*argv));
    return hx_umbilicus_main(argc + 1, args);
}

void hx_level_samples(uint32_t count, uint32_t size, uint32_t *out) {
    Pcg64 rng = pcg64(0);
    if (count > 10000 && size > count / 50) {
        uint32_t *tail = malloc(count * sizeof(*tail));
        need(tail != NULL, "cannot sample upright nodes");
        for (uint32_t i = 0; i < count; i++) {
            tail[i] = i;
        }
        uint32_t first = count - size;
        for (uint32_t i = count - 1; i > 0 && i >= first; i--) {
            uint32_t j = draw(&rng, i + 1), swap = tail[j];
            tail[j] = tail[i];
            tail[i] = swap;
        }
        memcpy(out, tail + first, size * sizeof(*out));
        free(tail);
        return;
    }
    uint32_t chosen[4096];
    memset(chosen, 0xff, sizeof(chosen));
    const uint32_t mask = sizeof(chosen) / sizeof(*chosen) - 1;
    for (uint32_t i = 0, end = count - size; i < size; i++, end++) {
        uint32_t pick = draw(&rng, end + 1);
        uint32_t slot = pick & mask;
        while (chosen[slot] != UINT32_MAX && chosen[slot] != pick) {
            slot = (slot + 1) & mask;
        }
        if (chosen[slot] == pick) {
            pick = end;
            slot = pick & mask;
            while (chosen[slot] != UINT32_MAX) {
                slot = (slot + 1) & mask;
            }
        }
        chosen[slot] = out[i] = pick;
    }
    for (uint32_t i = size - 1; i > 0; i--) {
        uint32_t j = draw(&rng, i + 1), swap = out[j];
        out[j] = out[i];
        out[i] = swap;
    }
}
