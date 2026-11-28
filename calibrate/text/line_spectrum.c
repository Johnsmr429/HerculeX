#include "../row_measurement.h"

const char hx_usage_line_spectrum[] = "Measure line-spacing power across and along the papyrus.";

typedef struct {
    int cell, h, w, step, terms, angles;
    double pitch, coverage, *period, *tilt;
} LineSpectrum;

static double line_option(json_object *settings, const char *key, double fallback) {
    json_object *v = json_object_object_get(settings, key);
    return v ? json_object_get_double(v) : fallback;
}

static LineSpectrum line_settings(Arena *a, json_object *settings) {
    int cell = (int)line_option(settings, "cell", 8);
    double pitch = line_option(settings, "um", 0.009362) * cell;
    double coverage = line_option(settings, "min_cover", 0.7);
    need(cell > 0 && pitch > 0 && isfinite(pitch), "invalid line-spectrum cell size or pitch");
    json_object *window = json_object_object_get(settings, "window_mm");
    double mm_h = window ? json_object_get_double(json_object_array_get_idx(window, 0)) : 24;
    double mm_w = window ? json_object_get_double(json_object_array_get_idx(window, 1)) : 12;
    double mm_step = line_option(settings, "step_mm", 6);
    need(mm_h >= pitch && mm_w >= pitch && mm_step >= pitch && mm_h / pitch < INT_MAX &&
            mm_w / pitch < INT_MAX && mm_step / pitch < INT_MAX,
        "invalid line-spectrum window or step");
    json_object *periods = json_object_object_get(settings, "periods");
    json_object *tilts = json_object_object_get(settings, "tilts");
    int terms = periods ? (int)json_object_array_length(periods) : 25;
    int angles = tilts ? (int)json_object_array_length(tilts) : 17;
    need(terms > 0 && angles > 0, "line spectrum needs periods and tilts");
    double *period = push(a, (size_t)terms * sizeof(*period));
    double *tilt = push(a, (size_t)angles * sizeof(*tilt));
    for (int i = 0; i < terms; i++) {
        period[i] =
            periods ? json_object_get_double(json_object_array_get_idx(periods, i)) : 3 + i * 0.25;
        need(period[i] > 0 && isfinite(period[i]), "invalid line period");
    }
    for (int i = 0; i < angles; i++) {
        tilt[i] =
            tilts ? json_object_get_double(json_object_array_get_idx(tilts, i)) : -0.08 + i * 0.01;
        need(isfinite(tilt[i]), "invalid line tilt");
    }
    return (LineSpectrum){.cell = cell,
        .h = (int)(mm_h / pitch),
        .w = (int)(mm_w / pitch),
        .step = (int)(mm_step / pitch),
        .pitch = pitch,
        .coverage = coverage,
        .terms = terms,
        .angles = angles,
        .period = period,
        .tilt = tilt};
}

static json_object *line_windows(
    Arena *a, const double *white, const u8 *good, int h, int w, int reversed, LineSpectrum s) {
    json_object *windows = json_object_new_array();
    int height = reversed ? w : h, width = reversed ? h : w;
    int y_stride = reversed ? 1 : w, x_stride = reversed ? w : 1;
    if (height < s.h || width < s.w) {
        return windows;
    }
    size_t mark = a->used;
    double *row = push(a, (size_t)s.h * sizeof(*row));
    double *hits = push(a, (size_t)s.h * sizeof(*hits));
    double *hann = push(a, (size_t)s.h * sizeof(*hann));
    double *best = push(a, (size_t)s.terms * sizeof(*best));
    double *re = push(a, (i64)s.h * s.terms * sizeof(*re));
    double *im = push(a, (i64)s.h * s.terms * sizeof(*im));
    for (int y = 0; y < s.h; y++) {
        hann[y] = s.h == 1 ? 1 : 0.5 - 0.5 * cos(2 * M_PI * y / (s.h - 1));
        for (int k = 0; k < s.terms; k++) {
            double phase = -2 * M_PI * (y * s.pitch / s.period[k]);
            re[(i64)k * s.h + y] = cos(phase);
            im[(i64)k * s.h + y] = sin(phase);
        }
    }
    for (int y = 0; y <= height - s.h; y += s.step) {
        for (int x = 0; x <= width - s.w; x += s.step) {
            i64 start = (i64)y * y_stride + (i64)x * x_stride;
            if (row_coverage(good + start, s.h, s.w, y_stride, x_stride) < s.coverage) {
                continue;
            }
            memset(best, 0, (size_t)s.terms * sizeof(*best));
            for (int t = 0; t < s.angles; t++) {
                row_projection(white + start, good + start, s.h, s.w, y_stride, x_stride, s.pitch,
                    s.tilt[t], hann, row, hits);
                for (int k = 0; k < s.terms; k++) {
                    double u = 0, v = 0;
                    for (int r = 0; r < s.h; r++) {
                        u += row[r] * re[(i64)k * s.h + r];
                        v += row[r] * im[(i64)k * s.h + r];
                    }
                    double amp = hypot(u, v), power = amp * amp / s.h;
                    best[k] = isnan(power) ? power : fmax(best[k], power);
                }
            }
            json_object *powers = json_object_new_array_ext(s.terms);
            for (int k = 0; k < s.terms; k++) {
                json_object_array_add(
                    powers, isfinite(best[k]) ? jnum(best[k]) : json_object_new_double(best[k]));
            }
            json_object_array_add(windows, powers);
        }
    }
    a->used = mark;
    return windows;
}

int line_spectrum_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh calibrate.text.line_spectrum IMAGE.npy MASK.npy "
                        "OUT.json [SETTINGS.json]";
    if (argc == 1 && (!strcmp(argv[0], "-h") || !strcmp(argv[0], "--help"))) {
        puts(usage);
        return 0;
    }
    need(argc == 3 || argc == 4, "%s", usage);
    Arena a = arena(24 * GB);
    json_object *config = argc == 4 ? json_read(argv[3]) : NULL;
    need(argc == 3 || config, "cannot read line-spectrum settings");
    LineSpectrum s = line_settings(&a, config);
    json_object_put(config);
    Array image = npy_read(&a, argv[0]), mask = npy_read(&a, argv[1]);
    int kind = (unsigned char)image.descr[1], bytes = (unsigned char)image.descr[2] - '0';
    need(image.ndim == 2 && mask.ndim == 2 && image.shape[0] == mask.shape[0] &&
            image.shape[1] == mask.shape[1] && image.shape[0] <= INT_MAX &&
            image.shape[1] <= INT_MAX &&
            (!strcmp(mask.descr, "|b1") || !strcmp(mask.descr, "|u1")) &&
            ((kind == 'f' && (bytes == 2 || bytes == 4 || bytes == 8)) ||
                ((kind == 'i' || kind == 'u' || kind == 'b') &&
                    (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8))) &&
            (image.descr[0] == '<' || image.descr[0] == '|'),
        "invalid line-spectrum image or mask");
    int h = (int)image.shape[0] / s.cell, w = (int)image.shape[1] / s.cell;
    i64 n = (i64)h * w, full_w = image.shape[1], block = (i64)s.cell * s.cell;
    const u8 *seen = mask.data;
    double *values = push(&a, n * sizeof(*values)), *trend = push(&a, n * sizeof(*trend));
    double *weight = push(&a, n * sizeof(*weight)), *work = push(&a, n * sizeof(*work));
    u8 *good = push(&a, n);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double sum = 0;
            i64 covered = 0;
            for (int r = 0; r < s.cell; r++) {
                for (int c = 0; c < s.cell; c++) {
                    i64 at = ((i64)y * s.cell + r) * full_w + (i64)x * s.cell + c;
                    if (seen[at]) {
                        covered++;
                        sum += kind == 'f' && bytes == 8 ? ((const f64 *)image.data)[at]
                                                         : spot_input(image, kind, bytes, at);
                    }
                }
            }
            i64 i = (i64)y * w + x;
            good[i] = covered > block / 2;
            values[i] = (sum / block) / fmax(0.001, (double)covered / block);
            trend[i] = good[i] ? values[i] : 0;
            weight[i] = good[i];
        }
    }
    if (n) {
        double sigma = 4 / s.pitch;
        need(sigma < INT_MAX / 8, "line-spectrum smoothing radius is too large");
        spot_gaussian(&a, trend, work, h, w, sigma, 0);
        spot_gaussian(&a, weight, work, h, w, sigma, 1);
    }
    i64 chosen = 0;
    for (i64 i = 0; i < n; i++) {
        values[i] = good[i] ? values[i] - trend[i] / fmax(weight[i], (f32)0.001) : 0;
        if (good[i]) {
            work[chosen++] = values[i];
        }
    }
    double mean = chosen ? pairwise64(work, chosen) / chosen : 0;
    for (i64 i = 0; i < chosen; i++) {
        double d = work[i] - mean;
        work[i] = d * d;
    }
    double sd = chosen ? fmax(sqrt(pairwise64(work, chosen) / chosen), 1e-6) : NAN;
    for (i64 i = 0; i < n; i++) {
        values[i] /= sd;
    }
    json_object *result = json_object_new_object();
    json_object_object_add(result, "across", line_windows(&a, values, good, h, w, 0, s));
    json_object_object_add(result, "along", line_windows(&a, values, good, h, w, 1, s));
    json_write(argv[2], result);
    json_object_put(result);
    return 0;
}
