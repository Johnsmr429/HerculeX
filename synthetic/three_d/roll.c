typedef struct {
    int rows, per_turn, knots;
    double turns, knot_um, z_first, core, recto;
    double *radius, *thickness, *delamination, *gap, *pitch, *arc, *end, *outer;
} SynRoll;

static double syn_clip(double x, double low, double high) {
    if (x < low) {
        return low;
    }
    return x > high ? high : x;
}

static void syn_roll_field(Arena *a, json_object *spec, SynRoll r, const char *name, double along,
    double between, double height, double *out, double *tmp) {
    size_t n = (size_t)r.rows * r.per_turn * r.knots;
    Pcg64 rng = syn_rng(spec, name);
    for (size_t i = 0; i < n; i++) {
        tmp[i] = syn_normal(&rng);
    }
    syn_filter(a, tmp, out, n, 1, r.knots, height, 3, 0);
    syn_filter(a, out, tmp, n, (size_t)r.knots * r.per_turn, r.rows, between, 3, 0);
    syn_filter(a, tmp, out, n, r.knots, r.rows * r.per_turn, along, 3, 0);
    syn_center(out, tmp, n);
}

static SynRoll syn_roll(Arena *a, json_object *spec) {
    json_object *roll = jget(spec, "roll"), *band = jget(spec, "band_um");
    double bottom = json_object_get_double(json_object_array_get_idx(band, 0));
    double top = json_object_get_double(json_object_array_get_idx(band, 1));
    double margin = jf64(spec, "margin_um"), thickness = jf64(roll, "thickness_um");
    double gap = jf64(roll, "gap_um");
    SynRoll r = {.turns = jf64(roll, "turns"),
        .per_turn = ji64(roll, "per_turn"),
        .knot_um = jf64(roll, "knot_um"),
        .z_first = bottom - margin,
        .core = jf64(roll, "core_um"),
        .recto = jf64(roll, "recto_share")};
    const char *correlation[] = {"thickness_corr_um", "gap_corr_um", "wander_corr_um",
        "contact_corr_um", "delamination_corr_um"};
    const char *turns[] = {
        "thickness_corr_turns", "gap_corr_turns", "wander_corr_turns", "contact_corr_turns"};
    const char *names[] = {
        "roll.thickness", "roll.gap", "roll.wander", "roll.contact", "roll.delamination"};
    double rates[5][2];
    double per_um = r.per_turn / (2 * M_PI * (r.core + 0.5 * r.turns * (thickness + gap)));
    double finest = INFINITY;
    for (int k = 0; k < 5; k++) {
        json_object *pair = jget(roll, correlation[k]);
        rates[k][0] = json_object_get_double(json_object_array_get_idx(pair, 0)) * per_um;
        rates[k][1] = json_object_get_double(json_object_array_get_idx(pair, 1));
        finest = fmin(finest, rates[k][1]);
    }
    if (ji64(spec, "version") >= 5) {
        r.knot_um = fmax(fmin(r.knot_um, finest / 5), 60);
    }
    need(r.knot_um > 0 && r.per_turn > 0 && r.turns >= 0 && top >= bottom,
        "invalid synthetic roll extent");
    r.rows = ceil(r.turns) + 3;
    r.knots = ceil((top - bottom + 2 * margin) / r.knot_um) + 2;
    need(r.knots > 1, "invalid synthetic roll knot count");
    size_t slab = (size_t)r.per_turn * r.knots, n = slab * r.rows, bytes = n * sizeof(double);
    r.radius = push(a, bytes);
    r.thickness = push(a, bytes);
    r.delamination = pushz(a, bytes);
    r.gap = push(a, bytes);
    r.pitch = push(a, bytes);
    r.arc = push(a, bytes);
    r.end = push(a, slab * sizeof(double));
    r.outer = push(a, (size_t)r.knots * sizeof(double));
    size_t mark = a->used;
    double *field = push(a, bytes), *scratch = push(a, bytes);
    double probability = jf64(roll, "delamination");
    double contact = jf64(roll, "contact"), wander = jf64(roll, "wander");
    double variation[] = {jf64(roll, "thickness_spread"), jf64(roll, "gap_spread"), wander};
    for (int k = 0; k < 5; k++) {
        if ((k == 2 && wander <= 0) || (k == 3 && contact <= 0) || (k == 4 && probability <= 0)) {
            continue;
        }
        double cross = k == 4 ? 0 : jf64(roll, turns[k]);
        syn_roll_field(a, spec, r, names[k], rates[k][0], cross, rates[k][1] * (1 / r.knot_um),
            field, scratch);
        if (k == 4) {
            Pcg64 rng = syn_rng(spec, "roll.delamination.depth");
            json_object *depth = jget(roll, "delamination_um");
            double lo = json_object_get_double(json_object_array_get_idx(depth, 0));
            double hi = json_object_get_double(json_object_array_get_idx(depth, 1));
            double threshold = syn_quantile(1 - probability);
            for (int row = 0; row < r.rows; row++) {
                double value = lo + (hi - lo) * pcg64_double(&rng);
                for (size_t j = 0; j < slab; j++) {
                    size_t at = row * slab + j;
                    r.delamination[at] = syn_clip((field[at] - threshold) / 0.3, 0, 1) * value;
                }
            }
        } else if (k == 3) {
            double threshold = syn_quantile(contact), soft = fmax(jf64(roll, "contact_soft"), 1e-3);
            for (size_t i = 0; i < n; i++) {
                r.gap[i] *= syn_clip((field[i] - threshold) / soft, 0, 1);
            }
        } else {
            double spread = variation[k], bias = spread * spread / 2;
            double trend = jf64(roll, "gap_trend");
            for (int row = 0; row < r.rows; row++) {
                double fraction = row / fmax(r.turns, 1) - 0.5;
                double base_gap = gap * syn_clip(1 + trend * fraction * 2, 0.2, 3);
                for (size_t j = 0; j < slab; j++) {
                    size_t at = row * slab + j;
                    double value = exp(spread * field[at] - bias);
                    if (!k) {
                        r.thickness[at] =
                            syn_clip(thickness * value, .35 * thickness, 2.5 * thickness);
                    } else if (k == 1) {
                        r.gap[at] = base_gap * value;
                    } else {
                        r.gap[at] *= value;
                    }
                }
            }
        }
    }
    double packets = jf64(roll, "packets");
    if (packets > 0) {
        Pcg64 rng = syn_rng(spec, "roll.packets");
        json_object *range = jget(roll, "packet_gap_um");
        double lo = json_object_get_double(json_object_array_get_idx(range, 0));
        double hi = json_object_get_double(json_object_array_get_idx(range, 1));
        for (int i = 0; i < r.rows; i++) {
            field[i] = pcg64_double(&rng);
        }
        for (int i = 0; i < r.rows; i++) {
            if (field[i] >= packets) {
                continue;
            }
            double centre = pcg64_double(&rng), width = 0.15 + (0.6 - 0.15) * pcg64_double(&rng);
            double amplitude = lo + (hi - lo) * pcg64_double(&rng);
            for (int j = 0; j < r.per_turn; j++) {
                double u = j / (double)r.per_turn - centre + 0.5;
                double distance = (u - floor(u) - 0.5) / (width / 2.5);
                double bump = amplitude * exp(-0.5 * (distance * distance));
                size_t at = i * slab + (size_t)j * r.knots;
                for (int z = 0; z < r.knots; z++) {
                    r.gap[at + z] += bump;
                }
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        size_t next = i < n - slab ? i + slab : i;
        double current = r.thickness[i] + r.delamination[i];
        double following = r.thickness[next] + r.delamination[next];
        r.pitch[i] = 0.5 * (current + following) + r.gap[i];
    }
    memset(scratch, 0, slab * sizeof(double));
    for (int j = 0; j < r.per_turn; j++) {
        for (int z = 0; z < r.knots; z++) {
            size_t at = (size_t)j * r.knots + z;
            double first = r.core + j / (double)r.per_turn * r.pitch[z];
            double distance = 0;
            for (int i = 0; i < r.rows; i++) {
                size_t k = i * slab + at;
                r.radius[k] = first + distance;
                distance += r.pitch[k];
            }
        }
    }
    double arc_step = 2 * M_PI / r.per_turn;
    double *arc = scratch;
    for (size_t i = 0; i < n; i += r.knots) {
        for (int z = 0; z < r.knots; z++) {
            arc[z] += r.radius[i + z];
            r.arc[i + z] = arc[z] * arc_step;
        }
    }
    size_t outside = (size_t)ceil(r.turns) * slab;
    for (int z = 0; z < r.knots; z++) {
        double outer = r.radius[outside + z];
        for (int j = 1; j < r.per_turn; j++) {
            outer = fmax(outer, r.radius[outside + (size_t)j * r.knots + z]);
        }
        r.outer[z] = outer;
    }
    double peel = jf64(roll, "peel_turns"), extent = jf64(roll, "peel_extent_deg");
    if (peel > 0 && extent > 0) {
        Pcg64 rng = syn_rng(spec, "roll.peel");
        double centre = pcg64_double(&rng), width = extent / 360;
        for (size_t i = 0; i < slab; i++) {
            r.end[i] = syn_normal(&rng);
        }
        syn_filter(a, r.end, scratch, slab, r.knots, r.per_turn, 4, 3, 1);
        syn_filter(a, scratch, r.end, slab, 1, r.knots, 3, 3, 0);
        syn_center(r.end, scratch, slab);
        for (int j = 0; j < r.per_turn; j++) {
            double u = j / (double)r.per_turn - centre + 0.5;
            double distance = fabs(u - floor(u) - 0.5);
            for (int z = 0; z < r.knots; z++) {
                size_t i = (size_t)j * r.knots + z;
                double edge = 1 + 0.15 * r.end[i];
                r.end[i] = r.turns -
                    peel * syn_clip((width / 2 * edge - distance) / (0.05 * width + 1e-3), 0, 1);
            }
        }
    } else {
        for (size_t i = 0; i < slab; i++) {
            r.end[i] = r.turns;
        }
    }
    a->used = mark;
    return r;
}

static double syn_roll_at(SynRoll r, const double *table, int turn, double theta, double z) {
    double t = theta / (2 * M_PI) * r.per_turn;
    int sample = floor(t);
    double w = t - sample;
    double kz = syn_clip((z - r.z_first) / r.knot_um, 0, r.knots - 1.000001);
    int k = floor(kz), rows = r.rows * r.per_turn;
    double v = kz - k;
    int a = syn_clip((double)turn * r.per_turn + sample, 0, rows - 1);
    int b = a + 1 < rows ? a + 1 : a;
    const double *x = table + (size_t)a * r.knots + k, *y = table + (size_t)b * r.knots + k;
    return x[0] * (1 - w) * (1 - v) + y[0] * w * (1 - v) + x[1] * (1 - w) * v + y[1] * w * v;
}

static double syn_winding(SynRoll r, double rho, double theta, double z) {
    int lo = -1, hi = r.rows - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (syn_roll_at(r, r.radius, mid, theta, z) <= rho) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int k = lo < 0 ? 0 : lo;
    double r0 = syn_roll_at(r, r.radius, k, theta, z);
    double r1 = syn_roll_at(r, r.radius, k + 1, theta, z);
    return k + (rho - r0) / fmax(r1 - r0, 1e-6) + theta / (2 * M_PI);
}
