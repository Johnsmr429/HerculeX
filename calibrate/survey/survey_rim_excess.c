#ifndef HX_UNITY
const char hx_usage[] = "Count forward and reversed ink at increasing rim cuts.";
#endif

static json_object *survey_rim_piece(Arena *a, const char *folder, const char *name, double voxel) {
    const char *near_path = fmt(a, "%s/%s.png", folder, name);
    const char *far_path = fmt(a, "%s/reversed-%s.png", folder, name);
    const char *cells_path = fmt(a, "%s/%s.npz", folder, name);
    if (!exists(far_path) || !exists(cells_path)) {
        return NULL;
    }
    int w, h, far_w, far_h, sy, sx, spots[2][2];
    const u8 *near = png_read(a, near_path, &w, &h);
    const u8 *far = png_read(a, far_path, &far_w, &far_h);
    need(w == far_w && h == far_h, "survey maps differ in shape for %s", name);
    i32 *fy, *fx;
    survey_nearest(a, near, h, w, &fy, &fx);
    int count = survey_windows(a, near, h, w, spots, &sy, &sx);
    const double rims[] = {0, 0.5, 1, 2, 3};
    i64 counts[5][3] = {{0}};
    zip_t *archive = zip_open(cells_path, ZIP_RDONLY, NULL);
    need(archive != NULL, "cannot open %s", cells_path);
    for (int k = 0; k < count; k++) {
        char member[24], key[16];
        snprintf(key, sizeof(key), "open%d", k);
        snprintf(member, sizeof(member), "%s.npy", key);
        if (zip_name_locate(archive, member, 0) < 0) {
            continue;
        }
        Array cells = npz_get(a, cells_path, key);
        need(cells.ndim == 2 && !strcmp(cells.descr, "<f2") && cells.shape[0] * 8 <= sy &&
                cells.shape[1] * 8 <= sx,
            "invalid open-column cells in %s", cells_path);
        const u16 *mask = cells.data;
        for (int r = 0; r < cells.shape[0] * 8; r++) {
            int y = spots[k][0] + r;
            for (int c = 0; c < cells.shape[1] * 8; c++) {
                int x = spots[k][1] + c;
                i64 at = (i64)y * w + x;
                if (mask[r / 8 * cells.shape[1] + c / 8] > 0x3800 && near[at]) {
                    double distance = survey_distance(y, x, fy[at], fx[at]) * voxel;
                    for (int j = 0; j < 5 && distance >= rims[j]; j++) {
                        counts[j][0]++;
                        counts[j][1] += near[at] >= 167;
                        counts[j][2] += far[at] >= 167;
                    }
                }
            }
        }
    }
    zip_close(archive);
    json_object *row = json_object_new_object();
    for (int i = 0; i < 5; i++) {
        char key[16];
        snprintf(key, sizeof(key), "%.1f", rims[i]);
        json_object *one = json_object_new_array();
        for (int k = 0; k < 3; k++) {
            json_object_array_add(one, json_object_new_int64(counts[i][k]));
        }
        json_object_object_add(row, key, one);
    }
    return row;
}

int survey_rim_command(int argc, char **argv) {
    if (argc && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("calibrate.survey.survey_rim_excess SAMPLE/SCAN/BAND ...");
        return 0;
    }
    Arena a = arena(4ull * GB);
    json_object *record = json_object_new_object();
    const char *stages[] = {"", "replicate"}, *keys[] = {"first", "replicate"};
    for (int i = 0; i < argc; i++) {
        size_t band_base = a.used;
        SurveyBand band = survey_band(&a, argv[i]);
        json_object *result = json_object_new_object();
        json_object_object_add(record, band.sample, result);
        printf("%s {", band.sample);
        for (int s = 0; s < 2; s++) {
            const char *folder = survey_stage(&a, band, stages[s]);
            glob_t paths = {0};
            glob(fmt(&a, "%s/piece-*.json", folder), 0, NULL, &paths);
            json_object *stage = json_object_new_object();
            json_object_object_add(result, keys[s], stage);
            for (size_t p = 0; p < paths.gl_pathc; p++) {
                size_t saved = a.used;
                const char *name = survey_piece_name(&a, paths.gl_pathv[p], 0);
                double voxel = survey_voxel(&a, band, name);
                json_object *prior = json_read(paths.gl_pathv[p]);
                if (board_integer(prior, "pixels")) {
                    json_object *row = survey_rim_piece(&a, folder, name, voxel);
                    if (row) {
                        json_object_object_add(stage, name, row);
                    }
                }
                json_object_put(prior);
                a.used = saved;
            }
            globfree(&paths);
            printf("%s'%s': %d", s ? ", " : "", keys[s], json_object_object_length(stage));
        }
        survey_write(fmt(&a, "%s/rim-excess.json", band.survey), record);
        puts("}");
        fflush(stdout);
        a.used = band_base;
    }
    json_object_put(record);
    munmap(a.base, a.size);
    return 0;
}
