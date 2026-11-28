#include "../row_measurement.h"

#ifndef HX_UNITY
const char hx_usage[] = "Measure row spacings on the forward and reversed survey maps.";
#endif

#define SURVEY_PERIODS 25

static void survey_blur(Arena *a, double *data, int h, int w, double sigma, int f32_output) {
    size_t saved = a->used;
    int radius = (int)lround(4 * sigma);
    double norm = 0, *weight = push(a, (radius + 1) * sizeof(*weight));
    for (int j = 0; j <= radius; j++) {
        weight[j] = exp(-0.5 * ((double)j / sigma) * ((double)j / sigma));
        norm += weight[j] * (j ? 2 : 1);
    }
    for (int j = 0; j <= radius; j++) {
        weight[j] /= norm;
    }
    double *temp = push(a, (i64)h * w * sizeof(*temp));
    for (int axis = 0; axis < 2; axis++) {
        const double *in = axis ? temp : data;
        double *out = axis ? data : temp;
        int lines = axis ? h : w, n = axis ? w : h, stride = axis ? 1 : w;
        size_t used = a->used;
        double *line = (double *)push(a, (n + 2 * radius) * sizeof(*line)) + radius;
        for (int i = 0; i < lines; i++) {
            i64 offset = axis ? (i64)i * w : i;
            for (int k = -radius; k < n + radius; k++) {
                line[k] = in[offset + reflected(k, n) * stride];
            }
            for (int j = 0; j < n; j++) {
                double total = weight[0] * line[j];
                for (int k = radius; k > 0; k--) {
                    total += (line[j - k] + line[j + k]) * weight[k];
                }
                out[offset + (i64)j * stride] = f32_output ? (f32)total : total;
            }
        }
        a->used = used;
    }
    a->used = saved;
}

static int survey_power(Arena *a, const double *values, const u8 *good, int h, int w,
    int transposed, double pitch, int sy, int sx, double out[SURVEY_PERIODS]) {
    int image_h = transposed ? w : h, image_w = transposed ? h : w;
    int stride_y = transposed ? 1 : w, stride_x = transposed ? w : 1;
    if (image_h < sy || image_w < sx) {
        return 0;
    }
    int step = 6.0 / pitch, found = 0;
    need(step > 0 && sy > 0 && sx > 0, "invalid row-power window");
    size_t saved = a->used;
    double *split = push(a, sy * sizeof(*split)), *hits = push(a, sy * sizeof(*hits));
    double *hann = push(a, sy * sizeof(*hann));
    double (*re)[SURVEY_PERIODS] = push(a, sy * sizeof(*re));
    double (*im)[SURVEY_PERIODS] = push(a, sy * sizeof(*im));
    for (int y = 0; y < sy; y++) {
        hann[y] = sy > 1 ? 0.5 - 0.5 * cos(2 * M_PI * y / (sy - 1)) : 1;
        for (int p = 0; p < SURVEY_PERIODS; p++) {
            double phase = -2 * M_PI * (y * pitch / (3 + p * 0.25));
            re[y][p] = cos(phase);
            im[y][p] = sin(phase);
        }
    }
    for (int y = 0; y <= image_h - sy; y += step) {
        for (int x = 0; x <= image_w - sx; x += step) {
            i64 window = (i64)y * stride_y + (i64)x * stride_x;
            if (row_coverage(good + window, sy, sx, stride_y, stride_x) < 0.7) {
                continue;
            }
            double best[SURVEY_PERIODS] = {0};
            for (int j = 0; j < 17; j++) {
                double tilt = j * (0.16 / 16) - 0.08;
                row_projection(values + window, good + window, sy, sx, stride_y, stride_x, pitch,
                    tilt, hann, split, hits);
                for (int p = 0; p < SURVEY_PERIODS; p++) {
                    double real = 0, imag = 0;
                    for (int r = 0; r < sy; r++) {
                        real += split[r] * re[r][p];
                        imag += split[r] * im[r][p];
                    }
                    best[p] = fmax(best[p], (real * real + imag * imag) / sy);
                }
            }
            for (int p = 0; p < SURVEY_PERIODS; p++) {
                out[p] += best[p];
            }
            found++;
        }
    }
    a->used = saved;
    return found;
}

static json_object *survey_row_ratio(
    Arena *a, const u8 *image, int h, int w, double voxel, double rim) {
    size_t saved = a->used;
    int nh = h / 8, nw = w / 8;
    double pitch = voxel * 8;
    int sy = (rim ? 14 : 16) / pitch, sx = 10 / pitch;
    if (nh < sy || nw < sy || nh < sx || nw < sx || sy < 1 || sx < 1) {
        return NULL;
    }
    i32 *fy = NULL, *fx = NULL;
    if (rim) {
        survey_nearest(a, image, h, w, &fy, &fx);
    }
    i64 n = (i64)nh * nw;
    double *cells = push(a, n * sizeof(*cells)), *blur = push(a, n * sizeof(*blur));
    double *weights = push(a, n * sizeof(*weights));
    u8 *good = push(a, n);
    for (int r = 0; r < nh; r++) {
        for (int c = 0; c < nw; c++) {
            int hit = 0;
            double sum = 0;
            for (int dy = 0; dy < 8; dy++) {
                int y = r * 8 + dy;
                for (int dx = 0; dx < 8; dx++) {
                    int x = c * 8 + dx;
                    i64 at = y * (i64)w + x;
                    int seen = image[at] > 0;
                    if (rim && seen) {
                        seen = survey_distance(y, x, fy[at], fx[at]) * voxel >= rim;
                    }
                    if (seen) {
                        hit++;
                        sum += ((f32)image[at] - 1) / 254;
                    }
                }
            }
            i64 at = r * (i64)nw + c;
            good[at] = hit > 32;
            cells[at] = hit ? sum / hit : 0;
            blur[at] = good[at] ? cells[at] : 0;
            weights[at] = good[at];
        }
    }
    double spread = 4 / pitch;
    survey_blur(a, blur, nh, nw, spread, 0);
    survey_blur(a, weights, nh, nw, spread, 1);
    double sum = 0, squared = 0;
    i64 count = 0;
    for (i64 at = 0; at < n; at++) {
        cells[at] = good[at] ? cells[at] - blur[at] / fmax(weights[at], (f32)0.001) : 0;
        sum += cells[at];
        squared += cells[at] * cells[at];
        count += good[at];
    }
    if (!count) {
        a->used = saved;
        return NULL;
    }
    double scale = fmax(1e-6, sqrt(fmax(0, squared / count - (sum / count) * (sum / count))));
    for (i64 at = 0; at < n; at++) {
        cells[at] /= scale;
    }
    double powers[2][SURVEY_PERIODS] = {{0}};
    int across = survey_power(a, cells, good, nh, nw, 0, pitch, sy, sx, powers[0]);
    int along = survey_power(a, cells, good, nh, nw, 1, pitch, sy, sx, powers[1]);
    a->used = saved;
    if (!across || !along) {
        return NULL;
    }
    json_object *row = json_object_new_object();
    double band[2] = {0};
    for (int d = 0; d < 2; d++) {
        json_object *spectrum = json_object_new_array();
        for (int p = 0; p < SURVEY_PERIODS; p++) {
            double power = powers[d][p] / (d ? along : across);
            if (p >= 4 && p <= 16) {
                band[d] += power;
            }
            json_object_array_add(spectrum, jnum(rint(power * 10000) / 10000));
        }
        json_object_object_add(row, d ? "along" : "across", spectrum);
    }
    json_object_object_add(row, "ratio",
        band[1] ? jnum(band[0] / band[1]) : json_object_new_double(band[0] ? INFINITY : NAN));
    json_object_object_add(row, "windows", json_object_new_int(across));
    return row;
}

static json_object *survey_row_stage(Arena *a, SurveyBand band, const char *folder) {
    json_object *rows = json_object_new_object();
    glob_t paths = {0};
    glob(fmt(a, "%s/piece-*.png", folder), 0, NULL, &paths);
    for (size_t p = 0; p < paths.gl_pathc; p++) {
        size_t saved = a->used;
        const char *name = survey_piece_name(a, paths.gl_pathv[p], 0);
        double voxel = survey_voxel(a, band, name);
        int h, w, h2 = 0, w2 = 0;
        const u8 *map = png_read(a, paths.gl_pathv[p], &w, &h);
        const char *back = fmt(a, "%s/reversed-%s.png", folder, name);
        const u8 *rear = exists(back) ? png_read(a, back, &w2, &h2) : NULL;
        json_object *found = survey_row_ratio(a, map, h, w, voxel, 0);
        json_object *trim = survey_row_ratio(a, map, h, w, voxel, 1);
        if (found || trim) {
            if (found && rear) {
                json_object_object_add(
                    found, "reversed", survey_row_ratio(a, rear, h2, w2, voxel, 0));
            }
            if (!found) {
                found = json_object_new_object();
            }
            if (trim) {
                json_object_object_add(found, "no_rim", trim);
                if (rear) {
                    json_object_object_add(
                        trim, "reversed", survey_row_ratio(a, rear, h2, w2, voxel, 1));
                }
            }
            json_object_object_add(rows, name, found);
        }
        a->used = saved;
    }
    globfree(&paths);
    return rows;
}

static int survey_double_order(const void *aa, const void *bb) {
    const double a = *(const double *)aa, b = *(const double *)bb;
    return (a > b) - (a < b);
}

int survey_row_power_command(int argc, char **argv) {
    if (argc && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("calibrate.survey.survey_row_power SAMPLE/SCAN/BAND ...");
        return 0;
    }
    Arena a = arena(4ull * GB);
    const char *record_path = fmt(&a, "%s/calibration/scroll_survey/row-power.json", LIBRARY);
    json_object *record = json_read(record_path);
    record = record ? record : json_object_new_object();
    const char *stages[] = {"", "replicate", "row-sample"};
    const char *keys[] = {"first", "replicate", "row_sample"};
    for (int i = 0; i < argc; i++) {
        size_t saved = a.used;
        SurveyBand band = survey_band(&a, argv[i]);
        json_object *result = json_object_new_object();
        json_object *parts[3];
        int total = 0, count = 0;
        for (int s = 0; s < 3; s++) {
            parts[s] = survey_row_stage(&a, band, survey_stage(&a, band, stages[s]));
            json_object_object_add(result, keys[s], parts[s]);
            total += json_object_object_length(parts[s]);
        }
        double *ratios = push(&a, total * sizeof(*ratios));
        for (int s = 0; s < 3; s++) {
            json_object_object_foreach(parts[s], key, value) {
                (void)key;
                json_object *ratio = json_object_object_get(value, "ratio");
                if (ratio) {
                    ratios[count++] = json_object_get_double(ratio);
                }
            }
        }
        json_object_object_add(record, band.sample, result);
        json_write(record_path, record);
        if (count) {
            qsort(ratios, count, sizeof(*ratios), survey_double_order);
            printf("%s %d pieces %.3f\n", band.sample, count,
                (ratios[(count - 1) / 2] + ratios[count / 2]) / 2);
        } else {
            printf("%s 0 pieces None\n", band.sample);
        }
        fflush(stdout);
        a.used = saved;
    }
    json_object_put(record);
    munmap(a.base, a.size);
    return 0;
}
