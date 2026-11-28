static const char syn_defaults[] =
    "{\"version\":7,\"family\":\"custom\",\"seed\":0,\"lineage\":{\"parent\":null,\"edits\""
    ":[],\"look\":null},\"voxel_um\":9.362,\"band_um\":[0.0,4800.0],\"margin_um\":600.0,\"r"
    "oll\":{\"turns\":60.0,\"per_turn\":512,\"knot_um\":250.0,\"core_um\":1200.0,\"start_de"
    "g\":0.0,\"thickness_um\":65.0,\"thickness_spread\":0.3,\"thickness_corr_um\":[3000.0,2"
    "000.0],\"thickness_corr_turns\":0.5,\"recto_share\":0.5,\"gap_um\":120.0,\"gap_spread"
    "\":0.6,\"gap_trend\":0.0,\"gap_corr_um\":[3000.0,2500.0],\"gap_corr_turns\":0.4,\"wand"
    "er\":0.5,\"wander_corr_um\":[1000.0,800.0],\"wander_corr_turns\":0.15,\"contact\":0.12"
    ",\"contact_soft\":0.5,\"contact_corr_um\":[5000.0,3000.0],\"contact_corr_turns\":0.3,"
    "\"packets\":0.0,\"packet_gap_um\":[300.0,900.0],\"delamination\":0.0,\"delamination_um"
    "\":[15.0,60.0],\"delamination_corr_um\":[3000.0,2000.0],\"peel_turns\":0.0,\"peel_exte"
    "nt_deg\":0.0},\"sheet\":{\"ply_dip\":0.3,\"ply_dip_um\":6.0,\"surface_dip\":0.25,\"sur"
    "face_dip_um\":5.0,\"fibre\":0.3,\"fibre_um\":[1200.0,40.0],\"holes\":0.0,\"hole_um\":5"
    "00.0,\"fray\":0.3,\"crinkle_um\":8.0,\"crinkle_scale_um\":150.0,\"specks\":0.6,\"speck"
    "_threshold\":2.6,\"speck_um\":30.0,\"fray_um\":80.0,\"gain_sd\":0.2,\"gain_um\":2000.0"
    ",\"fog_um\":0.0,\"debris\":0.0,\"debris_um\":6.0,\"compress_max\":1.25,\"grain\":0.0,"
    "\"grain_um\":30.0},\"shape\":{\"crush\":{\"kappa\":1.0,\"kappa_core\":1.0,\"core_reach"
    "_um\":3000.0,\"nx\":2.0,\"ny\":2.0,\"keep_length\":true},\"harmonics\":[],\"angle_deg"
    "\":0.0,\"twist_deg_per_mm\":0.0,\"kappa_slope_per_mm\":0.0},\"folds\":[],\"flows\":[],"
    "\"cracks\":[],\"placement\":{\"centre_um\":[0.0,0.0],\"tilt\":[0.0,0.0],\"bend_um\":[0"
    ".0,0.0],\"mirror\":false},\"map\":{\"step_um\":100.0,\"knot_um\":400.0},\"ct\":{\"gap"
    "\":55.0,\"papyrus\":115.0,\"psf_um\":9.0,\"phase\":0.0,\"unsharp\":2.0,\"unsharp_um\":"
    "12.0,\"noise\":13.0,\"noise_um\":12.0,\"noise_z_um\":0.0,\"noise_air\":1.0,\"rings\":0"
    ".0,\"ring_um\":12.0,\"cupping\":0.0,\"drift\":0.0,\"mask\":true,\"mask_um\":150.0}}";

static void syn_merge(json_object *base, json_object *extra) {
    json_object_object_foreach(extra, name, value) {
        json_object *here = json_object_object_get(base, name);
        if (json_object_is_type(here, json_type_object) &&
            json_object_is_type(value, json_type_object)) {
            syn_merge(here, value);
        } else {
            json_object *copy = NULL;
            if (value) {
                need(!json_object_deep_copy(value, &copy, NULL), "cannot copy spec field %s", name);
            }
            json_object_object_add(base, name, copy);
        }
    }
}

static const char *syn_unknown(
    Arena *a, json_object *known, json_object *input, const char *prefix, const char *found) {
    json_object_object_foreach(input, key, value) {
        json_object *base;
        if (!json_object_object_get_ex(known, key, &base)) {
            found = fmt(a, "%s%s%s%s", found, *found ? ", " : "", prefix, key);
        } else if (json_object_is_type(base, json_type_object) && json_object_object_length(base) &&
            json_object_is_type(value, json_type_object)) {
            found = syn_unknown(a, base, value, fmt(a, "%s%s.", prefix, key), found);
        }
    }
    return found;
}

static json_object *syn_complete(Arena *a, json_object *input) {
    need(json_object_is_type(input, json_type_object), "a synthetic spec must be a JSON object");
    json_object *spec = json_tokener_parse(syn_defaults);
    need(spec != NULL, "synthetic defaults could not be read");
    json_object *version = json_object_object_get(input, "version");
    double v = version ? json_object_get_double(version) : 7;
    if (v < 2) {
        json_object *sheet = jget(spec, "sheet");
        json_object_object_add(sheet, "crinkle_um", jnum(0));
        json_object_object_add(sheet, "specks", jnum(0));
    }
    if (v < 3) {
        json_object *sheet = jget(spec, "sheet"), *ct = jget(spec, "ct");
        json_object_object_add(sheet, "fray_um", jnum(150));
        json_object_object_add(sheet, "speck_um", jnum(14));
        json_object_object_add(sheet, "gain_sd", jnum(0));
        json_object_object_add(ct, "unsharp", jnum(0));
        json_object_object_add(ct, "phase", jnum(0.35));
    }
    size_t mark = a->used;
    const char *unknown = syn_unknown(a, spec, input, "", "");
    need(!*unknown, "unknown spec keys: %s", unknown);
    a->used = mark;
    syn_merge(spec, input);
    return spec;
}

static void syn_spec_string(FILE *out, json_object *value) {
    const char *quoted = json_object_to_json_string_ext(value, JSON_C_TO_STRING_NOSLASHESCAPE);
    const u8 *p = (const u8 *)quoted;
    while (*p) {
        if (*p < 0x7f) {
            fputc(*p++, out);
            continue;
        }
        if (*p == 0x7f) {
            fputs("\\u007f", out);
            p++;
            continue;
        }
        int tails = *p < 0xe0 ? 1 : *p < 0xf0 ? 2 : 3;
        u32 code = *p++ & (0x3fu >> tails);
        for (int i = 0; i < tails; i++) {
            need((*p & 0xc0) == 0x80, "invalid UTF-8 in synthetic spec");
            code = (code << 6) | (*p++ & 0x3f);
        }
        if (code <= 0xffff) {
            fprintf(out, "\\u%04x", code);
        } else {
            code -= 0x10000;
            fprintf(out, "\\u%04x\\u%04x", 0xd800u + (code >> 10), 0xdc00u + (code & 1023));
        }
    }
}

static void syn_spec_sorted(Arena *a, FILE *out, json_object *value) {
    if (json_object_is_type(value, json_type_object)) {
        size_t mark = a->used;
        int n = json_object_object_length(value), count = 0;
        const char **names = push(a, (size_t)n * sizeof(*names));
        struct json_object_iterator itr = json_object_iter_begin(value);
        struct json_object_iterator end = json_object_iter_end(value);
        for (; !json_object_iter_equal(&itr, &end); json_object_iter_next(&itr)) {
            names[count++] = json_object_iter_peek_name(&itr);
        }
        qsort(names, count, sizeof(*names), name_order);
        fputc('{', out);
        for (int k = 0; k < count; k++) {
            if (k) {
                fputs(", ", out);
            }
            json_object *name = json_object_new_string(names[k]);
            syn_spec_string(out, name);
            json_object_put(name);
            fputs(": ", out);
            syn_spec_sorted(a, out, json_object_object_get(value, names[k]));
        }
        fputc('}', out);
        a->used = mark;
    } else if (json_object_is_type(value, json_type_array)) {
        size_t n = json_object_array_length(value);
        fputc('[', out);
        for (size_t i = 0; i < n; i++) {
            if (i) {
                fputs(", ", out);
            }
            syn_spec_sorted(a, out, json_object_array_get_idx(value, i));
        }
        fputc(']', out);
    } else if (json_object_is_type(value, json_type_double)) {
        double v = json_object_get_double(value);
        size_t mark = a->used;
        fputs(isfinite(v)  ? python_float(a, v)
                : isnan(v) ? "NaN"
                : v < 0    ? "-Infinity"
                           : "Infinity",
            out);
        a->used = mark;
    } else if (json_object_is_type(value, json_type_string)) {
        syn_spec_string(out, value);
    } else {
        fputs(json_object_to_json_string_ext(value, JSON_C_TO_STRING_PLAIN), out);
    }
}

static void syn_spec_digest(Arena *a, json_object *input, char name[13]) {
    json_object *spec = syn_complete(a, input);
    json_object_object_del(spec, "lineage");
    char *data = NULL;
    size_t bytes = 0;
    FILE *out = open_memstream(&data, &bytes);
    need(out != NULL, "cannot build synthetic spec digest");
    syn_spec_sorted(a, out, spec);
    need(!fclose(out), "cannot finalize synthetic spec digest");
    char hash[65];
    sha256_hex(data, bytes, hash);
    memcpy(name, hash, 12);
    name[12] = 0;
    free(data);
    json_object_put(spec);
}

int syn_spec_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.spec {complete INPUT OUT|digest INPUT}";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    int complete = argc == 3 && !strcmp(argv[0], "complete");
    need(complete || (argc == 2 && !strcmp(argv[0], "digest")), "%s", usage);
    Arena a = arena(16 << 20);
    json_object *input = json_read(argv[1]);
    need(input != NULL, "cannot read %s", argv[1]);
    if (complete) {
        json_object *ready = syn_complete(&a, input);
        json_write(argv[2], ready);
        json_object_put(ready);
    } else {
        char digest[13];
        syn_spec_digest(&a, input, digest);
        puts(digest);
    }
    json_object_put(input);
    return 0;
}
