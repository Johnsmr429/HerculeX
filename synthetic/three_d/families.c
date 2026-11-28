#define SYN_SCAN_BRICK 128

enum {
    SYN_ROUND,
    SYN_OVAL,
    SYN_FLAT,
    SYN_PANCAKE,
    SYN_LENS,
    SYN_BOXY,
    SYN_DIAMOND,
    SYN_POLYGON,
    SYN_ECCENTRIC,
    SYN_D_SHAPE,
    SYN_KIDNEY,
    SYN_U_FOLD,
    SYN_CREASE,
    SYN_Z_FOLD,
    SYN_S_CORE,
    SYN_TWISTED,
    SYN_TAPERED,
    SYN_TILTED,
    SYN_WAVY,
    SYN_WRINKLED,
    SYN_DENTED
};
enum { SYN_LOOSE, SYN_NORMAL, SYN_PRESSED, SYN_FUSED, SYN_DELAMINATED, SYN_DAMAGED, SYN_CRACKED };

static const char *syn_shapes[] = {"round", "oval", "flattened", "pancake", "lens", "boxy",
    "diamond", "polygon", "eccentric", "d-shape", "kidney", "u-fold", "crease", "z-fold", "s-core",
    "twisted", "tapered", "tilted", "wavy", "wrinkled", "dented"};
static const char *syn_conditions[] = {
    "loose", "normal", "pressed", "fused", "delaminated", "damaged", "cracked"};

typedef struct {
    const char *name, *container, *key;
    int pos;
    double lo, hi;
    int log_scale;
} SynTuning;

static const SynTuning syn_tuning[] = {
    {"gap", "ct", "gap", -1, 25, 75, 0},
    {"papyrus", "ct", "papyrus", -1, 80, 200, 0},
    {"noise", "ct", "noise", -1, 1.5, 18, 1},
    {"noise_um", "ct", "noise_um", -1, 6, 45, 1},
    {"noise_z_um", "ct", "noise_z_um", -1, 6, 120, 1},
    {"psf_um", "ct", "psf_um", -1, 3, 18, 1},
    {"unsharp", "ct", "unsharp", -1, 0.0, 5.0, 0},
    {"unsharp_um", "ct", "unsharp_um", -1, 4, 35, 1},
    {"fibre", "sheet", "fibre", -1, 0.0, 0.9, 0},
    {"fibre_along", "sheet", "fibre_um", 0, 150, 3000, 1},
    {"fibre_across", "sheet", "fibre_um", 1, 15, 200, 1},
    {"fray", "sheet", "fray", -1, 0.0, 0.9, 0},
    {"fray_um", "sheet", "fray_um", -1, 25, 250, 1},
    {"crinkle_um", "sheet", "crinkle_um", -1, 0.0, 50.0, 0},
    {"crinkle_scale_um", "sheet", "crinkle_scale_um", -1, 50, 1000, 1},
    {"specks", "sheet", "specks", -1, 0.0, 3.0, 0},
    {"speck_threshold", "sheet", "speck_threshold", -1, 1.6, 3.3, 0},
    {"speck_um", "sheet", "speck_um", -1, 12, 70, 1},
    {"ply_dip", "sheet", "ply_dip", -1, 0.0, 0.95, 0},
    {"surface_dip", "sheet", "surface_dip", -1, -0.6, 0.5, 0},
    {"gain_sd", "sheet", "gain_sd", -1, 0.0, 0.6, 0},
    {"thickness_um", "roll", "thickness_um", -1, 30, 130, 1},
    {"thickness_spread", "roll", "thickness_spread", -1, 0.05, 0.6, 0},
    {"gap_um", "roll", "gap_um", -1, 30, 320, 1},
    {"gap_spread", "roll", "gap_spread", -1, 0.2, 1.1, 0},
    {"wander", "roll", "wander", -1, 0.05, 1.1, 0},
    {"contact", "roll", "contact", -1, 0.0, 0.45, 0},
    {"delamination", "roll", "delamination", -1, 0.0, 0.8, 0},
    {"delamination_hi_um", "roll", "delamination_um", 1, 8, 70, 1},
    {"wander_s_um", "roll", "wander_corr_um", 0, 300, 4000, 1},
    {"wander_z_um", "roll", "wander_corr_um", 1, 300, 3000, 1},
    {"corrugate_amp_um", "flows", "amp_um", 1, 0, 250, 0},
    {"corrugate_z_um", "flows", "z_scale_um", 1, 400, 3000, 1},
    {"fog_um", "sheet", "fog_um", -1, 0.0, 30.0, 0},
    {"grain", "sheet", "grain", -1, 0.0, 0.8, 0},
    {"grain_um", "sheet", "grain_um", -1, 10, 80, 1},
    {"noise_air", "ct", "noise_air", -1, 0.1, 1.0, 0},
    {"gap_s_um", "roll", "gap_corr_um", 0, 300, 4000, 1},
    {"contact_s_um", "roll", "contact_corr_um", 0, 300, 6000, 1},
    {"thickness_s_um", "roll", "thickness_corr_um", 0, 300, 4000, 1},
    {"delamination_s_um", "roll", "delamination_corr_um", 0, 300, 4000, 1},
    {"debris", "sheet", "debris", -1, 0.0, 1.0, 0},
    {"debris_um", "sheet", "debris_um", -1, 3, 15, 1},
    {"rings", "ct", "rings", -1, 0.0, 8.0, 0},
    {"ring_um", "ct", "ring_um", -1, 6, 60, 1},
};

static double syn_u(Pcg64 *g, double lo, double hi) {
#ifdef __APPLE__
    return fma(hi - lo, pcg64_double(g), lo);
#else
    return lo + (hi - lo) * pcg64_double(g);
#endif
}

static double syn_log_u(Pcg64 *g, double lo, double hi) { return exp(syn_u(g, log(lo), log(hi))); }

static void syn_add(json_object *o, const char *name, double value) {
    json_object_object_add(o, name, jnum(value));
}

static json_object *syn_values(int n, const double *v) {
    json_object *a = json_object_new_array_ext(n);
    for (int i = 0; i < n; i++) {
        json_object_array_add(a, jnum(v[i]));
    }
    return a;
}

static void syn_add_pair(json_object *o, const char *name, double first, double second) {
    json_object_object_add(o, name, syn_values(2, (double[]){first, second}));
}

static json_object *syn_look(
    Arena *a, Pcg64 *g, json_object *base, int scanner, const char *look_name, json_object *rows) {
    const char *group = scanner ? "113keV" : "54keV";
    int n = json_object_object_length(rows);
    size_t mark = a->used;
    const char **names = push(a, n * sizeof(*names));
    int count = 0;
    if (rows) {
        json_object_object_foreach(rows, key, value) {
            const char *category = json_object_get_string(json_object_object_get(value, "group"));
            if (look_name ? !strcmp(look_name, key) : category && !strcmp(group, category)) {
                names[count++] = key;
            }
        }
    }
    qsort(names, count, sizeof(*names), name_order);
    json_object *selection = NULL;
    if (count) {
        const char *name = names[draw(g, count)];
        selection = json_object_new_string(name);
        json_object *look = jget(rows, name), *settings = jget(look, "settings"),
                    *spread = jget(look, "spread");
        json_object *defaults = json_tokener_parse(syn_defaults);
        json_object_object_foreach(settings, key, value) {
            const SynTuning *t = syn_tuning;
            while (t < syn_tuning + sizeof(syn_tuning) / sizeof(*syn_tuning) &&
                strcmp(t->name, key) != 0) {
                t++;
            }
            need(t != syn_tuning + sizeof(syn_tuning) / sizeof(*syn_tuning),
                "unknown look setting %s", key);
            double center = json_object_get_double(value),
                   sigma = 1.5 * syn_optional(spread, key, 0);
            if (t->log_scale) {
                sigma = fmax(sigma / fmax(center, 1e-9), 0.1);
                center *= exp(sigma * syn_normal(g));
            } else {
                sigma = fmax(sigma, fmax(0.1 * fabs(center), 0.02 * (t->hi - t->lo)));
                center += sigma * syn_normal(g);
            }
            double drawn = syn_clip(center, t->lo, t->hi);
            json_object *section = jget(base, t->container);
            if (!strcmp(t->container, "flows")) {
                syn_add(json_object_array_get_idx(section, t->pos), t->key, drawn);
            } else if (t->pos < 0) {
                syn_add(section, t->key, drawn);
            } else {
                json_object *values = json_object_object_get(section, t->key);
                if (!values) {
                    if (!strcmp(t->key, "delamination_um")) {
                        values = syn_values(2, (double[]){5, 30});
                    } else {
                        json_object *old = jget(jget(defaults, t->container), t->key);
                        need(!json_object_deep_copy(old, &values, NULL),
                            "cannot copy look defaults");
                    }
                    json_object_object_add(section, t->key, values);
                }
                json_object_array_put_idx(values, t->pos, jnum(drawn));
            }
        }
        json_object_put(defaults);
        json_object *delam = json_object_object_get(jget(base, "roll"), "delamination_um");
        if (delam) {
            double lo = json_object_get_double(json_object_array_get_idx(delam, 0));
            double hi = json_object_get_double(json_object_array_get_idx(delam, 1));
            json_object_array_put_idx(delam, 0, jnum(fmin(lo, hi)));
        }
    }
    a->used = mark;
    return selection;
}

static void syn_condition(Pcg64 *g, int kind, json_object *roll, json_object *sheet, int tuned) {
    if (!tuned) {
        syn_add(roll, "thickness_um", syn_log_u(g, 50, 100));
        syn_add(roll, "thickness_spread", syn_u(g, 0.2, 0.4));
        syn_add(roll, "wander", syn_u(g, 0.3, 0.7));
        syn_add(roll, "gap_um", syn_log_u(g, 140, 250));
        syn_add(roll, "gap_spread", syn_u(g, 0.4, 0.8));
        syn_add(roll, "contact", syn_u(g, 0.05, 0.15));
        syn_add(sheet, "fibre", syn_u(g, 0.15, 0.45));
        syn_add(sheet, "fray", syn_u(g, 0.15, 0.4));
        syn_add(sheet, "ply_dip", syn_u(g, 0.1, 0.45));
        syn_add(sheet, "surface_dip", syn_u(g, 0.1, 0.4));
    }
    syn_add(sheet, "debris", syn_u(g, 0, 0.4));
    syn_add(sheet, "debris_um", syn_u(g, 4, 10));
    syn_add(roll, "gap_trend", syn_u(g, -0.5, 0.3));
    syn_add(roll, "packets", syn_u(g, 0, 0.03));
    double packet = syn_u(g, 150, 300);
    syn_add_pair(roll, "packet_gap_um", packet, syn_u(g, 400, 1200));
    syn_add(roll, "peel_turns", pcg64_double(g) < 0.4 ? syn_u(g, 0, 4) : 0);
    syn_add(roll, "peel_extent_deg", syn_u(g, 30, 160));
    double contact = syn_optional(roll, "contact", 0.1), gap = jf64(roll, "gap_um");
    if (kind == SYN_LOOSE) {
        syn_add(roll, "gap_um", gap * syn_log_u(g, 1.6, 2.8));
        syn_add(roll, "contact", 0.2 * contact);
    } else if (kind == SYN_NORMAL || kind == SYN_CRACKED) {
        syn_add(roll, "gap_um", gap * syn_log_u(g, 0.8, 1.3));
        syn_add(roll, "contact", contact * syn_u(g, 0.6, 1.2));
    } else if (kind == SYN_PRESSED || kind == SYN_FUSED) {
        syn_add(roll, "gap_um", gap * syn_log_u(g, 0.5, 0.85));
        syn_add(roll, "contact", fmax(contact, syn_u(g, 0.25, 0.5)));
        syn_add(roll, "contact_soft", syn_u(g, 0.15, 0.5));
        double along = syn_log_u(g, 3000, 20000);
        syn_add_pair(roll, "contact_corr_um", along, syn_log_u(g, 2000, 8000));
        syn_add(roll, "contact_corr_turns", syn_u(g, 0.3, 1.2));
        if (kind == SYN_FUSED) {
            syn_add(sheet, "surface_dip", 0.2 * syn_optional(sheet, "surface_dip", 0.2));
            syn_add(sheet, "ply_dip", 0.3 * syn_optional(sheet, "ply_dip", 0.3));
            syn_add(sheet, "fibre", 0.5 * syn_optional(sheet, "fibre", 0.3));
        }
    } else if (kind == SYN_DELAMINATED) {
        json_object *delam = json_object_object_get(roll, "delamination_um");
        double hi = delam ? json_object_get_double(json_object_array_get_idx(delam, 1)) : 60;
        syn_add(
            roll, "delamination", fmax(syn_optional(roll, "delamination", 0), syn_u(g, 0.4, 0.8)));
        syn_add_pair(roll, "delamination_um", syn_u(g, 5, 15), fmin(1.5 * hi, 120));
        syn_add(sheet, "ply_dip", fmax(syn_optional(sheet, "ply_dip", 0.3), syn_u(g, 0.3, 0.6)));
    } else if (kind == SYN_DAMAGED) {
        syn_add(roll, "gap_um", gap * syn_log_u(g, 0.8, 1.6));
        syn_add(roll, "packets", syn_u(g, 0.02, 0.08));
        syn_add(roll, "peel_turns", syn_u(g, 0, 12));
        syn_add(roll, "peel_extent_deg", syn_u(g, 0, 200));
        syn_add(sheet, "holes", syn_u(g, 0.03, 0.2));
        syn_add(sheet, "hole_um", syn_log_u(g, 300, 1500));
        syn_add(sheet, "fray", fmin(1.5 * syn_optional(sheet, "fray", 0.3), 0.6));
        syn_add(sheet, "debris", syn_u(g, 0.3, 0.8));
    }
}

static void syn_add_harmonic(json_object *rows, int m, double amplitude, double phase) {
    json_object *item = json_object_new_array_ext(3);
    json_object_array_add(item, json_object_new_int(m));
    json_object_array_add(item, jnum(amplitude));
    json_object_array_add(item, jnum(phase));
    json_object_array_add(rows, item);
}

static void syn_family_crush(
    Arena *a, Pcg64 *g, int kind, json_object *shape, json_object *roll, json_object *spec) {
    double k = syn_u(g, 0.85, 1), core = syn_u(g, 0.6, 1), reach = syn_log_u(g, 1500, 8000);
    json_object *crush = json_object_new_object(), *h = json_object_new_array();
    for (int m = 2; m <= 3; m++) {
        double amp = syn_u(g, 0, .02);
        syn_add_harmonic(h, m, amp, syn_u(g, 0, 360));
    }
    switch (kind) {
    case SYN_OVAL:
        k = syn_u(g, 0.55, 0.85);
        core = syn_u(g, 0.3, 0.7);
        break;
    case SYN_FLAT:
    case SYN_KIDNEY:
    case SYN_U_FOLD:
    case SYN_CREASE:
    case SYN_Z_FOLD:
    case SYN_TWISTED:
    case SYN_TAPERED:
    case SYN_DENTED:
        k = syn_u(g, 0.3, 0.55);
        core = syn_u(g, 0.08, 0.35);
        break;
    case SYN_PANCAKE:
    case SYN_S_CORE:
        k = syn_u(g, 0.22, 0.4);
        core = syn_u(g, 0.03, 0.12);
        break;
    case SYN_LENS:
        k = syn_u(g, 0.25, 0.5);
        core = syn_u(g, 0.05, 0.2);
        syn_add(crush, "ny", syn_u(g, 1, 1.35));
        break;
    case SYN_BOXY:
    case SYN_DIAMOND:
        k = syn_u(g, 0.5, 0.9);
        syn_add(crush, "nx", kind == SYN_BOXY ? syn_u(g, 3, 5) : syn_u(g, 1.15, 1.6));
        syn_add(crush, "ny", kind == SYN_BOXY ? syn_u(g, 3, 5) : syn_u(g, 1.15, 1.6));
        break;
    case SYN_POLYGON:
    case SYN_ECCENTRIC:
    case SYN_D_SHAPE:
        k = kind == SYN_D_SHAPE ? syn_u(g, 0.5, 0.8) : syn_u(g, 0.6, 0.95);
        json_object_array_del_idx(h, 0, 2);
        if (kind == SYN_POLYGON) {
            for (int m = 3; m <= 6; m++) {
                double amp = syn_u(g, 0.03, 0.08) / (m - 2);
                syn_add_harmonic(h, m, amp, syn_u(g, 0, 360));
            }
        } else if (kind == SYN_ECCENTRIC) {
            core = syn_u(g, 0.2, 0.6);
            double a1 = syn_u(g, .05, .15);
            syn_add_harmonic(h, 1, a1, syn_u(g, 0, 360));
            double a2 = syn_u(g, 0, .05);
            syn_add_harmonic(h, 2, a2, syn_u(g, 0, 360));
        } else {
            syn_add_harmonic(h, 1, syn_u(g, .06, .12), 0);
            syn_add_harmonic(h, 2, syn_u(g, .04, .08), 0);
            syn_add_harmonic(h, 3, syn_u(g, .02, .05), 180);
        }
        break;
    case SYN_WAVY:
    case SYN_WRINKLED:
    case SYN_TILTED:
        k = syn_u(g, 0.45, 0.95);
        core = syn_u(g, 0.1, 0.6);
        break;
    default: break;
    }
    if (k < 0.6) {
        for (int m = 2; m <= 5; m++) {
            double amp = syn_u(g, 0, 0.04) / (m - 1);
            syn_add_harmonic(h, m, amp, syn_u(g, 0, 360));
        }
        syn_add(roll, "contact", jf64(roll, "contact") * 0.3);
    }
    json_object *ready = syn_complete(a, spec), *r = jget(ready, "roll");
    double thick = jf64(r, "thickness_um"), spread = jf64(r, "gap_spread");
    double wander = jf64(r, "wander");
    double gap = jf64(r, "gap_um") * exp(-(spread * spread + wander * wander) / 2) *
        (1 - jf64(r, "contact"));
    syn_add(crush, "kappa", (thick + k * gap) / (thick + gap));
    syn_add(crush, "kappa_core", (thick + fmin(core, k) * gap) / (thick + gap));
    syn_add(crush, "core_reach_um", reach);
    json_object_put(ready);
    json_object_object_add(shape, "crush", crush);
    json_object_object_add(shape, "harmonics", h);
}

static SynPoint syn_family_axes(Arena *a, json_object *spec) {
    size_t mark = a->used;
    json_object *roll = jget(spec, "roll");
    double pitch =
        jf64(roll, "thickness_um") + jf64(roll, "gap_um") * (1 - 0.5 * jf64(roll, "contact"));
    double core = jf64(roll, "core_um"), rho = core + jf64(roll, "turns") * pitch;
    SynCrush *crush = syn_crush(a, jget(jget(spec, "shape"), "crush"), core, rho * 1.1, 1);
    int i = (int)(rho / crush->step);
    double left = i * crush->step, right = (i + 1) * crush->step;
    double x = rho - left, width = right - left;
    SynPoint axes = {.x = (crush->a[i + 1] - crush->a[i]) / width * x + crush->a[i],
        .y = (crush->b[i + 1] - crush->b[i]) / width * x + crush->b[i]};
    a->used = mark;
    return axes;
}

static json_object *syn_fold_record(double angle, double direction, double pos, double offset) {
    json_object *row = json_object_new_object();
    syn_add(row, "angle_deg", angle);
    syn_add(row, "direction_deg", direction);
    syn_add(row, "position_um", pos);
    syn_add(row, "offset_um", offset);
    return row;
}

static json_object *syn_family_folds(Arena *a, Pcg64 *g, int kind, json_object *spec) {
    json_object *out = json_object_new_array();
    if (kind < SYN_KIDNEY || kind > SYN_Z_FOLD) {
        return out;
    }
    SynPoint axis = syn_family_axes(a, spec);
    double b = 1.1 * axis.y, side = pcg64_double(g) < .5 ? 1 : -1, pos = 0;
    if (kind == SYN_Z_FOLD) {
        pos = syn_u(g, 0.15, 0.4) * axis.x;
    }
    const double bounds[][6] = {{30, 75, -0.2, 0.2, 1, 3}, {140, 180, 0.1, 0.5, 300, 1500},
        {165, 180, 0.2, 0.6, 40, 250}, {150, 180, 0.15, 0.4, 100, 800}};
    const double *r = bounds[kind - SYN_KIDNEY];
    int count = kind == SYN_Z_FOLD ? 2 : 1;
    for (int i = 0; i < count; i++) {
        double angle = syn_u(g, r[0], r[1]);
        if (kind != SYN_Z_FOLD || i) {
            pos = syn_u(g, r[2], r[3]) * axis.x;
        }
        double offset = syn_u(g, r[4], r[5]);
        if (kind == SYN_KIDNEY) {
            offset *= b;
        }
        json_object_array_add(out, syn_fold_record(angle, 180.0 * i, pos, side * (b + offset)));
    }
    return out;
}

static json_object *syn_family_cracks(Arena *a, Pcg64 *g, json_object *spec) {
    json_object *out = json_object_new_array();
    SynPoint axis = syn_family_axes(a, spec);
    double angle = jf64(jget(spec, "shape"), "angle_deg") * (M_PI / 180);
    double mirror = json_object_get_boolean(jget(jget(spec, "placement"), "mirror")) ? -1 : 1;
    int count = 2 + draw(g, 9);
    for (int i = 0; i < count; i++) {
        double r = sqrt(pcg64_double(g)), phi = syn_u(g, 0, 2 * M_PI);
        SynPoint p =
            syn_rotate((SynPoint){.x = axis.x * r * cos(phi), .y = axis.y * r * sin(phi)}, angle);
        p.x *= mirror;
        double radial = atan2(p.y, p.x) * (180 / M_PI) + 90;
        json_object *row = json_object_new_object();
        syn_add_pair(row, "centre_um", p.y, p.x);
        syn_add(row, "angle_deg", radial + 25 * syn_normal(g));
        syn_add(row, "half_length_um", syn_log_u(g, 1000, 8000));
        syn_add(row, "open_um", syn_log_u(g, 20, 300));
        syn_add(row, "slip_um", syn_u(g, -400, 400));
        syn_add(row, "wedge", syn_u(g, -0.9, 0.9));
        syn_add(row, "wiggle_um", syn_log_u(g, 5, 60));
        syn_add(row, "wiggle_wavelength_um", syn_log_u(g, 300, 2500));
        json_object_array_add(out, row);
    }
    return out;
}

static json_object *syn_flow_record(Pcg64 *g, const double *range) {
    json_object *row = json_object_new_object();
    syn_add(row, "scale_um", syn_log_u(g, range[0], range[1]));
    syn_add(row, "amp_um", syn_u(g, range[2], range[3]));
    if (range[4]) {
        syn_add(row, "z_scale_um", syn_log_u(g, range[4], range[5]));
    }
    return row;
}

static json_object *syn_family_draw(
    Arena *a, const char *family, i64 seed, int slices, const char *look_name) {
    size_t mark = a->used;
    const char *slash = strchr(family, '/');
    need(slash && !strchr(slash + 1, '/'), "expected a shape/condition family");
    int kind = -1, condition = -1;
    for (int k = 0; k <= SYN_DENTED; k++) {
        if (strlen(syn_shapes[k]) == (size_t)(slash - family) &&
            !strncmp(syn_shapes[k], family, slash - family)) {
            kind = k;
        }
    }
    for (int k = 0; k <= SYN_CRACKED; k++) {
        if (!strcmp(syn_conditions[k], slash + 1)) {
            condition = k;
        }
    }
    json_object *spec = json_object_new_object();
    json_object_object_add(spec, "family", json_object_new_string(family));
    json_object_object_add(spec, "seed", json_object_new_int64(seed));
    Pcg64 g = syn_rng(spec, fmt(a, "family.%s", family));
    const double voxels[] = {7.91, 8.64, 9.362};
    int scanner = draw(&g, 3);
    const char *path =
        env("HERCULEX_SYNTHETIC_LOOKS", fmt(a, "%s/synthetic/three_d/looks.json", HOME));
    json_object *looks = json_read(path), *profiles = looks ? jget(looks, "profiles") : NULL;
    if (look_name) {
        double v = profiles ? jf64(jget(profiles, look_name), "voxel_um") : 9.362;
        scanner = 0;
        while (scanner < 3 && fabs(voxels[scanner] - v) >= 1e-3) {
            scanner++;
        }
        need(scanner < 3, "look %s has unsupported voxel size", look_name);
    }
    double voxel = voxels[scanner], v1 = 2 * voxel;
    if (!slices) {
        slices = draw(&g, 4) == 3 ? 384 : 256;
    }
    need(slices > 0 && !(slices % SYN_SCAN_BRICK),
        "a band is a whole number of 128-slice brick planes");
    double start = (double)draw(&g, 40) * SYN_SCAN_BRICK * v1;
    syn_add(spec, "voxel_um", voxel);
    syn_add_pair(spec, "band_um", start, start + slices * v1);
    json_object *roll = json_object_new_object(), *sheet = json_object_new_object();
    json_object *ct = json_object_new_object(), *tuned_flows = json_object_new_array_ext(2);
    json_object *shape = json_object_new_object(), *placement = json_object_new_object();
    json_object_object_add(spec, "roll", roll);
    json_object_object_add(spec, "sheet", sheet);
    json_object_object_add(spec, "ct", ct);
    json_object_object_add(spec, "shape", shape);
    json_object_object_add(spec, "placement", placement);
    for (int k = 0; k < 2; k++) {
        json_object_array_add(tuned_flows, json_object_new_object());
    }
    json_object_object_add(spec, "flows", tuned_flows);
    json_object *look = syn_look(a, &g, spec, scanner, look_name, profiles);
    json_object_put(looks);
    json_object *lineage = json_object_new_object();
    json_object_object_add(lineage, "look", look);
    json_object_object_add(spec, "lineage", lineage);
    syn_condition(&g, condition, roll, sheet, look != NULL);
    double core = syn_log_u(&g, 500, 3000);
    syn_add(roll, "core_um", core);
    syn_add(roll, "start_deg", syn_u(&g, 0, 360));
    double pitch = jf64(roll, "thickness_um") +
        jf64(roll, "gap_um") * (1 - 0.5 * jf64(roll, "contact")) + 600 * jf64(roll, "packets");
    double outer = syn_log_u(&g, 8000, 22000);
    syn_add(roll, "turns", nearbyint(100 * syn_clip((outer - core) / pitch, 30, 160)) / 100);
    syn_family_crush(a, &g, kind, shape, roll, spec);
    syn_add(shape, "angle_deg", syn_u(&g, 0, 180));
    json_object_object_add(placement, "mirror", json_object_new_boolean(pcg64_double(&g) < 0.5));
    if (!look) {
        const double options[][8] = {{38, 55, 95, 135, 10, 18, 6, 10},
            {40, 58, 95, 135, 10, 18, 7, 11}, {42, 58, 100, 140, 11, 18, 7, 12}};
        const char *names[] = {"gap", "papyrus", "noise", "psf_um"};
        for (int k = 0; k < 4; k++) {
            syn_add(ct, names[k], syn_u(&g, options[scanner][2 * k], options[scanner][2 * k + 1]));
        }
        syn_add(ct, "phase", syn_u(&g, 0.1, 0.6));
    }
    if (kind == SYN_TWISTED || kind == SYN_TAPERED) {
        int twist = kind == SYN_TWISTED;
        double magnitude = twist ? syn_u(&g, 1, 6) : syn_u(&g, 0.01, 0.06);
        double sign = pcg64_double(&g) < 0.5 ? 1 : -1;
        syn_add(shape, twist ? "twist_deg_per_mm" : "kappa_slope_per_mm", magnitude * sign);
    }
    if (kind == SYN_TILTED) {
        double tilt = syn_u(&g, -0.15, 0.15);
        syn_add_pair(placement, "tilt", tilt, syn_u(&g, -0.15, 0.15));
        double bend = syn_u(&g, -100, 100);
        syn_add_pair(placement, "bend_um", bend, syn_u(&g, -100, 100));
    }
    json_object *full = syn_complete(a, spec);
    json_object_object_add(spec, "folds", syn_family_folds(a, &g, kind, full));
    int crushed = kind == SYN_FLAT || kind == SYN_PANCAKE || kind == SYN_LENS ||
        (kind >= SYN_KIDNEY && kind <= SYN_S_CORE) || kind == SYN_DENTED;
    json_object *flows = json_object_new_array();
    json_object_array_add(
        flows, syn_flow_record(&g, (double[]){4000, 12000, 50, crushed ? 700 : 400, 0, 0}));
    json_object_array_add(flows, syn_flow_record(&g, (double[]){1500, 4000, 20, 120, 500, 1500}));
    if (kind == SYN_WAVY) {
        json_object_array_add(flows, syn_flow_record(&g, (double[]){1200, 3500, 100, 300, 0, 0}));
    } else if (kind == SYN_WRINKLED) {
        json_object_array_add(flows, syn_flow_record(&g, (double[]){400, 1000, 20, 70, 500, 3000}));
    } else if (kind == SYN_DENTED) {
        json_object_array_add(flows, syn_flow_record(&g, (double[]){2500, 6000, 400, 1200, 0, 0}));
    } else if (kind == SYN_S_CORE) {
        json_object *swirl = json_object_new_object();
        json_object_object_add(swirl, "kind", json_object_new_string("swirl"));
        double angle = syn_u(&g, 90, 220);
        syn_add(swirl, "angle_deg", pcg64_double(&g) < 0.5 ? angle : -angle);
        syn_add(swirl, "radius_um", syn_log_u(&g, 1500, 4000));
        json_object_array_add(flows, swirl);
    }
    for (int k = 0; look && k < 2; k++) {
        syn_merge(json_object_array_get_idx(flows, k), json_object_array_get_idx(tuned_flows, k));
    }
    json_object_object_add(spec, "flows", flows);
    if (condition == SYN_CRACKED) {
        json_object_object_add(spec, "cracks", syn_family_cracks(a, &g, full));
    }
    json_object_put(full);
    full = syn_complete(a, spec);
    json_object_put(spec);
    a->used = mark;
    return full;
}

static int syn_held_out(int i, int j) {
    return i == SYN_Z_FOLD || (i == SYN_KIDNEY && j == SYN_FUSED) ||
        (i == SYN_POLYGON && j == SYN_DELAMINATED) || (i == SYN_TAPERED && j == SYN_PRESSED) ||
        (i == SYN_WRINKLED && j == SYN_DAMAGED) || (i == SYN_BOXY && j == SYN_LOOSE) ||
        (i == SYN_LENS && (j == SYN_FUSED || j == SYN_CRACKED)) ||
        (i == SYN_TWISTED && j == SYN_CRACKED);
}

static int syn_families_command(int argc, char **argv) {
    const char *usage =
        "Usage: ./run.sh synthetic.three_d.families {list|spec FAMILY SEED [OUT]} [--slices N]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    if (argc == 1 && !strcmp(argv[0], "list")) {
        int count = 0, held = 0;
        for (int i = 0; i <= SYN_DENTED; i++) {
            for (int j = 0; j <= SYN_CRACKED; j++) {
                int reserved = syn_held_out(i, j);
                printf("%s/%s %s\n", syn_shapes[i], syn_conditions[j], reserved ? "held-out" : "");
                count++;
                held += reserved;
            }
        }
        fprintf(stderr, "%d families, %d held out\n", count, held);
        return 0;
    }
    need(argc >= 3 && !strcmp(argv[0], "spec"), "%s", usage);
    char *end;
    i64 seed = strtoll(argv[2], &end, 10);
    need(*argv[2] && !*end, "invalid synthetic seed %s", argv[2]);
    const char *out = NULL;
    int slices = 0;
    for (int k = 3; k < argc; k++) {
        if (!strcmp(argv[k], "--slices") && k + 1 < argc) {
            slices = strtol(argv[++k], &end, 10);
            need(*argv[k] && !*end, "invalid synthetic slice count %s", argv[k]);
        } else {
            need(!out && argv[k][0] != '-', "%s", usage);
            out = argv[k];
        }
    }
    Arena a = arena(GB);
    json_object *spec = syn_family_draw(&a, argv[1], seed, slices, NULL);
    if (out) {
        json_write(out, spec);
    } else {
        puts(json_object_to_json_string_ext(spec, JSON_C_TO_STRING_PRETTY));
    }
    json_object_put(spec);
    return 0;
}
