static json_object *syn_percentiles(
    double *values, size_t n, const double *quantiles, int count, double round_by) {
    if (!n) {
        return NULL;
    }
    qsort(values, n, sizeof(*values), syn_double_order);
    json_object *out = json_object_new_array_ext(count);
    for (int i = 0; i < count; i++) {
        double position = quantiles[i] * (n - 1);
        size_t lo = position, hi = lo + (lo + 1 < n);
        double w = position - lo;
        double v = syn_mix(values[lo], values[hi], w);
        json_object_array_add(out, jnum(nearbyint(v * round_by) / round_by));
    }
    return out;
}

static void syn_sheet_checks(
    Arena *a, const SynScroll *s, json_object *spec, int samples, json_object *out) {
    size_t mark = a->used;
    SynDeform d = s->deform;
    double *points = push(a, (size_t)samples * 3 * sizeof(*points));
    double *pitch = push(a, (size_t)samples * sizeof(*pitch));
    double *radial_pitch = push(a, (size_t)samples * sizeof(*radial_pitch));
    json_object *band = jget(spec, "band_um");
    double low[] = {json_object_get_double(json_object_array_get_idx(band, 0)), d.box.y0, d.box.x0};
    double high[] = {
        json_object_get_double(json_object_array_get_idx(band, 1)), d.box.y1, d.box.x1};
    Pcg64 rng = pcg64(0);
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < samples; i++) {
            points[3 * i + j] = syn_u(&rng, low[j], high[j]);
        }
    }
    size_t inside = 0, on = 0, pressed = 0, finite_pitch = 0, finite_radial = 0;
    for (int i = 0; i < samples; i++) {
        double z = points[3 * i], y = points[3 * i + 1], x = points[3 * i + 2];
        SynTruthPoint t = syn_truth_at(&s->kernel, z, y, x);
        inside += t.valid && t.mask;
        if (!(t.valid && t.mask && fabs(t.nearest) <= t.thickness / 2)) {
            continue;
        }
        on++;
        if (isfinite(t.pitch)) {
            pitch[finite_pitch++] = t.pitch;
            pressed += t.pitch <= 80;
        }
        double dz = z - d.mid, mm = dz / 1000;
        double dy = y - (d.center.y + d.tilt.y * dz + d.bend.y * (mm * mm));
        double dx = x - (d.center.x + d.tilt.x * dz + d.bend.x * (mm * mm));
        double r = fmax(hypot(dy, dx), 1e-9);
        SynTruthPoint ahead = syn_truth_at(&s->kernel, z, y + 5 * (dy / r), x + 5 * (dx / r));
        double rate = fabs(ahead.winding - t.winding) / 5;
        if (rate > 0 && isfinite(rate)) {
            radial_pitch[finite_radial++] = 1 / rate;
        }
    }
    double q[] = {0.05, 0.25, 0.5, 0.75, 0.95};
    json_object_object_add(out, "pitch_um", syn_percentiles(pitch, finite_pitch, q, 5, 10));
    if (finite_pitch) {
        json_object_object_add(
            out, "radial_pitch_um", syn_percentiles(radial_pitch, finite_radial, q, 5, 10));
    }
    json_object_object_add(out, "pressed",
        finite_pitch ? jnum(nearbyint((double)pressed * 1000 / finite_pitch) / 1000) : NULL);
    syn_add(out, "sheet_share", finite_pitch ? nearbyint(on * 1000 / fmax(inside, 1)) / 1000 : 0);
    a->used = mark;
}

static int syn_verify_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.verify FAMILY SEED";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc == 2, "%s", usage);
    char *end;
    i64 seed = strtoll(argv[1], &end, 10);
    need(*argv[1] && !*end, "%s", usage);
    Arena a = arena(64 * GB);
    json_object *spec = syn_family_draw(&a, argv[0], seed, 0, NULL);
    long threads = strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), &end, 10);
    int workers = threads > 0 && threads <= INT_MAX && !*end ? threads : fmax(1, cores() - 2);
    SynScroll s;
    double voxel = jf64(spec, "voxel_um");
    syn_scroll(&a, &s, spec, NULL, 2 * voxel, workers);
    SynDeform d = s.deform;
    size_t total = (size_t)d.nz * d.ny * d.nx;
    double *stretch = push(&a, total * sizeof(*stretch));
    size_t kept = 0, bad = 0, positive = 0, negative = 0, zero = 0;
    for (int iz = 0; iz < d.nz; iz++) {
        for (int y = 0; y < d.ny; y++) {
            for (int x = 0; x < d.nx; x++) {
                size_t i = ((size_t)iz * d.ny + y) * d.nx + x;
                bad += s.kernel.bad[i] != 0;
                if (s.kernel.bad[i] || !s.kernel.mask[i]) {
                    continue;
                }
                size_t yl = y ? i - d.nx : i, yh = y + 1 < d.ny ? i + d.nx : i;
                size_t xl = x ? i - 1 : i, xh = x + 1 < d.nx ? i + 1 : i;
                double dy = d.step * (y && y + 1 < d.ny ? 2 : 1);
                double dx = d.step * (x && x + 1 < d.nx ? 2 : 1);
                double ay = ((double)d.a[yh] - d.a[yl]) / dy, ax = ((double)d.a[xh] - d.a[xl]) / dx;
                double by = ((double)d.b[yh] - d.b[yl]) / dy, bx = ((double)d.b[xh] - d.b[xl]) / dx;
                double determinant = ay * bx - ax * by;
                positive += determinant > 0;
                negative += determinant < 0;
                zero += determinant == 0;
                double rho = fmax(hypot(d.a[i], d.b[i]), 1e-9);
                if (rho > s.roll.core) {
                    double gy = (d.a[i] * ay + d.b[i] * by) / rho;
                    double gx = (d.a[i] * ax + d.b[i] * bx) / rho;
                    double size = fmax(hypot(gy, gx), 1e-12);
                    double ty = -gx / size, tx = gy / size;
                    stretch[kept++] = 1 / fmax(hypot(ay * ty + ax * tx, by * ty + bx * tx), 1e-12);
                }
            }
        }
    }
    double map_count = positive + negative + zero;
    double flipped = positive * 2 >= map_count ? negative + zero : positive + zero;
    json_object *out = json_object_new_object();
    json_object_object_add(out, "family", json_object_get(jget(spec, "family")));
    syn_add(out, "turns", s.roll.turns);
    syn_add_pair(out, "box_mm", nearbyint((d.box.y1 - d.box.y0) / 100) / 10,
        nearbyint((d.box.x1 - d.box.x0) / 100) / 10);
    syn_add(out, "jump_cells", nearbyint((double)bad / total * 1e5) / 1e5);
    syn_add(out, "flipped", map_count ? nearbyint(flipped / map_count * 1e6) / 1e6 : 0);
    json_object_object_add(
        out, "stretch", syn_percentiles(stretch, kept, (double[]){0.05, 0.5, 0.95}, 3, 1000));
    syn_sheet_checks(&a, &s, spec, 40000, out);
    puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PRETTY));
    json_object_put(out);
    json_object_put(spec);
    return 0;
}
