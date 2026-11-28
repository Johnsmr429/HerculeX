#define SYN_RHO 192
#define SYN_ANGLE 1024

typedef struct {
    double x, y;
} SynPoint;
typedef struct {
    double y0, x0, y1, x1;
} SynBox;
typedef struct {
    double nx, ny, step;
    double a[SYN_RHO], b[SYN_RHO], fraction[SYN_RHO][SYN_ANGLE + 1];
} SynCrush;
typedef struct {
    double a, s, c, sa, ca, offset, position, side, arc;
    int sliding;
} SynFold;
typedef struct {
    double y0, x0, step, rate, radius;
    int steps, ny, nx, swirl;
    float *vx, *vy;
} SynFlow;
typedef struct {
    int nz, ny, nx, folds, flows;
    double z0, dz, step, mid, start, mirror, harmonic_norm;
    SynBox box;
    SynPoint center, tilt, bend;
    json_object *harmonics;
    SynCrush **crush;
    SynFold *fold;
    SynFlow *flow;
    double *angle;
    float *a, *b;
} SynDeform;

static double syn_wrap(double angle) {
    double wrapped = fmod(angle, 2 * M_PI);
    return wrapped < 0 ? wrapped + 2 * M_PI : wrapped;
}

static double syn_optional(json_object *o, const char *name, double fallback) {
    json_object *value = json_object_object_get(o, name);
    return value ? json_object_get_double(value) : fallback;
}

static SynPoint syn_pair(json_object *o, const char *name) {
    json_object *value = jget(o, name);
    return (SynPoint){.y = json_object_get_double(json_object_array_get_idx(value, 0)),
        .x = json_object_get_double(json_object_array_get_idx(value, 1))};
}

static SynPoint syn_rotate(SynPoint p, double a) {
    double c = cos(a), s = sin(a);
    return (SynPoint){.x = p.x * c - p.y * s, .y = p.x * s + p.y * c};
}

static double syn_lame_length(const SynPoint *curve, double a, double b, double *fraction) {
    double distances[SYN_ANGLE];
    for (int j = 0; j < SYN_ANGLE; j++) {
        SynPoint p = curve[j], q = curve[(j + 1) % SYN_ANGLE];
        distances[j] = hypot(a * q.x - a * p.x, b * q.y - b * p.y);
    }
    double length = syn_sum(distances, SYN_ANGLE);
    if (fraction) {
        double acc = 0;
        for (int j = 0; j < SYN_ANGLE; j++) {
            fraction[j] = acc / fmax(length, 1e-9);
            acc += distances[j];
        }
        fraction[SYN_ANGLE] = 1;
    }
    return length;
}

static SynCrush *syn_crush(Arena *a, json_object *shape, double core, double top, double height) {
    SynCrush *c = push(a, sizeof(*c));
    c->nx = jf64(shape, "nx");
    c->ny = jf64(shape, "ny");
    c->step = top / (SYN_RHO - 1);
    const double da = 2 * M_PI / SYN_ANGLE;
    SynPoint curve[SYN_ANGLE];
    for (int j = 0; j < SYN_ANGLE; j++) {
        double x = cos(j * da), y = sin(j * da);
        curve[j] = (SynPoint){
            .x = copysign(pow(fabs(x), 2 / c->nx), x), .y = copysign(pow(fabs(y), 2 / c->ny), y)};
    }
    double inner = 1 - (1 - jf64(shape, "kappa_core")) * height;
    double outer = 1 - (1 - jf64(shape, "kappa")) * height;
    double reach = fmax(jf64(shape, "core_reach_um"), 1);
    int keep = json_object_get_boolean(jget(shape, "keep_length"));
    double last = inner, last_rho = 0, integral = 0;
    c->a[0] = c->b[0] = 0;
    for (int j = 0; j <= SYN_ANGLE; j++) {
        c->fraction[0][j] = (double)j / SYN_ANGLE;
    }
    for (int i = 1; i < SYN_RHO; i++) {
        double rho = i == SYN_RHO - 1 ? top : i * c->step;
        double rate = outer + (inner - outer) * exp(-fmax(rho - core, 0) / reach);
        if (rho > core) {
            integral += (rho - last_rho) * 0.5 * (rate + last);
        }
        last = rate;
        last_rho = rho;
        double b = rho <= core ? inner * rho : inner * core + integral;
        double x = fmax(rho, b);
        if (keep) {
            double lo = b, hi = fmax(3.5 * rho, b + 1e-6);
            for (int j = 0; j < 40; j++) {
                double mid = 0.5 * (lo + hi);
                if (syn_lame_length(curve, mid, b, NULL) < 2 * M_PI * rho) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            x = 0.5 * (lo + hi);
        }
        c->a[i] = fmax(x, c->a[i - 1]);
        c->b[i] = fmax(b, c->b[i - 1]);
        syn_lame_length(curve, fmax(c->a[i], 1e-6), fmax(c->b[i], 1e-6), c->fraction[i]);
    }
    return c;
}

static double syn_mix(double x, double y, double w) { return x + (y - x) * w; }

static SynPoint syn_axes(const SynCrush *c, double rho, int *i, double *w) {
    double j = syn_clip(rho / c->step, 0, SYN_RHO - 1);
    *i = fmin(floor(j), SYN_RHO - 2);
    *w = j - *i;
    return (SynPoint){
        .x = syn_mix(c->a[*i], c->a[*i + 1], *w), .y = syn_mix(c->b[*i], c->b[*i + 1], *w)};
}

static SynPoint syn_crush_forward(const SynCrush *c, double rho, double theta) {
    int i;
    double w;
    SynPoint axes = syn_axes(c, rho, &i, &w);
    const double *a = c->fraction[i], *b = c->fraction[i + 1];
    double target = theta / (2 * M_PI);
    int lo = 0, hi = SYN_ANGLE;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if ((1 - w) * a[m] + w * b[m] <= target) {
            lo = m;
        } else {
            hi = m;
        }
    }
    double begin = (1 - w) * a[lo] + w * b[lo], end = (1 - w) * a[hi] + w * b[hi];
    double t = (lo + (target - begin) / (end - begin)) * (2 * M_PI / SYN_ANGLE);
    double x = cos(t), y = sin(t);
    return (SynPoint){.x = axes.x * copysign(pow(fabs(x), 2 / c->nx), x),
        .y = axes.y * copysign(pow(fabs(y), 2 / c->ny), y)};
}

static SynPoint syn_crush_inverse(const SynCrush *c, SynPoint p) {
    double x = fabs(p.x), y = fabs(p.y);
    double lo = 0, hi = c->step * (SYN_RHO - 1), w, level = 0;
    int i;
    for (int k = 0; k < 48; k++) {
        double rho = 0.5 * (lo + hi);
        SynPoint axes = syn_axes(c, rho, &i, &w);
        level = pow(x / fmax(axes.x, 1e-9), c->nx) + pow(y / fmax(axes.y, 1e-9), c->ny);
        if (level > 1) {
            lo = rho;
        } else {
            hi = rho;
        }
    }
    double rho = 0.5 * (lo + hi);
    SynPoint axes = syn_axes(c, rho, &i, &w);
    double xx = x / fmax(axes.x, 1e-9), yy = y / fmax(axes.y, 1e-9);
    double cs = copysign(pow(syn_clip(xx, 0, 1), c->nx / 2), p.x);
    double sn = copysign(pow(syn_clip(yy, 0, 1), c->ny / 2), p.y);
    double u = syn_wrap(atan2(sn, cs)) / (2 * M_PI) * SYN_ANGLE;
    int j = (int)floor(u) % SYN_ANGLE;
    double f = u - floor(u);
    double theta = 2 * M_PI *
        ((1 - w) * syn_mix(c->fraction[i][j], c->fraction[i][j + 1], f) +
            w * syn_mix(c->fraction[i + 1][j], c->fraction[i + 1][j + 1], f));
    if (hi >= c->step * (SYN_RHO - 1) && level > 1) {
        level = pow(xx, c->nx) + pow(yy, c->ny);
        rho *= pow(level, 1 / fmax(c->nx, c->ny));
    }
    return (SynPoint){.x = rho, .y = syn_wrap(theta)};
}

static double syn_harmonic(json_object *rows, double phi) {
    double total = 0;
    for (size_t i = 0; i < json_object_array_length(rows); i++) {
        json_object *item = json_object_array_get_idx(rows, i);
        double k = json_object_get_double(json_object_array_get_idx(item, 0));
        double a = json_object_get_double(json_object_array_get_idx(item, 1));
        double p = json_object_get_double(json_object_array_get_idx(item, 2));
        total += a * cos(k * phi + p * (M_PI / 180));
    }
    return exp(total);
}

static SynPoint syn_fold_apply(const SynFold *f, SynPoint p, int inverse) {
    double s = p.x * f->c + p.y * f->s - f->position;
    double t = (-p.x * f->s + p.y * f->c) * f->side;
    double o = f->offset;
    if (inverse) {
        if (s < 0 && t < o) {
            return p;
        }
        double d = t - o, radius = hypot(s, d), beta = atan2(s, -d);
        if (beta >= 0 && beta <= f->a) {
            s = beta * (f->sliding ? radius : o);
            t = o - radius;
        } else {
            double along = s * f->ca + d * f->sa, across = -s * f->sa + d * f->ca;
            s = along < 0 ? 1e9 : along + (f->sliding ? -across * f->a : f->arc);
            t = o + across;
        }
    } else {
        if (s < 0) {
            return p;
        }
        double r = o - t, width = fmax(f->sliding ? r : o, 1e-6);
        double span = f->sliding ? width * f->a : f->arc;
        if (s <= span) {
            double beta = syn_clip(s / width, 0, f->a);
            s = r * sin(beta);
            t = o - r * cos(beta);
        } else {
            double along = s - span, across = t - o;
            s = along * f->ca - across * f->sa;
            t = o + along * f->sa + across * f->ca;
        }
    }
    t *= f->side;
    s += f->position;
    return (SynPoint){.x = s * f->c - t * f->s, .y = s * f->s + t * f->c};
}

static SynPoint syn_velocity(const SynFlow *f, int z, SynPoint p) {
    if (f->swirl) {
        double w = f->rate * exp(-0.5 * (p.x * p.x + p.y * p.y) / (f->radius * f->radius));
        return (SynPoint){.x = -w * p.y, .y = w * p.x};
    }
    double x = syn_clip((p.x - f->x0) / f->step, 0, f->nx - 1);
    double y = syn_clip((p.y - f->y0) / f->step, 0, f->ny - 1);
    int x0 = floor(x), y0 = floor(y);
    int x1 = x0 + (x0 < f->nx - 1), y1 = y0 + (y0 < f->ny - 1);
    double u = x - x0, v = y - y0;
    const float *vx = f->vx + (size_t)z * f->ny * f->nx, *vy = f->vy + (size_t)z * f->ny * f->nx;
    size_t ids[] = {(size_t)y0 * f->nx + x0, (size_t)y0 * f->nx + x1, (size_t)y1 * f->nx + x0,
        (size_t)y1 * f->nx + x1};
    double weights[] = {(1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u};
    SynPoint result = {0};
    for (int k = 0; k < 4; k++) {
        result.x += vx[ids[k]] * weights[k];
        result.y += vy[ids[k]] * weights[k];
    }
    return (SynPoint){.x = (float)result.x, .y = (float)result.y};
}

static SynPoint syn_move(const SynFlow *f, int z, SynPoint p, double sign) {
    double dt = sign / f->steps;
    for (int i = 0; i < f->steps; i++) {
        SynPoint k[4];
        k[0] = syn_velocity(f, z, p);
        for (int j = 1; j < 4; j++) {
            double step = j == 3 ? dt : 0.5 * dt;
            double dx = k[j - 1].x * step, dy = k[j - 1].y * step;
            if (!f->swirl) {
                dx = (float)k[j - 1].x * (float)step;
                dy = (float)k[j - 1].y * (float)step;
            }
            k[j] = syn_velocity(f, z, (SynPoint){.x = p.x + dx, .y = p.y + dy});
        }
        double dx = dt / 6 * (k[0].x + 2 * k[1].x + 2 * k[2].x + k[3].x);
        double dy = dt / 6 * (k[0].y + 2 * k[1].y + 2 * k[2].y + k[3].y);
        if (!f->swirl) {
            dx = (float)(dt / 6) *
                ((float)k[0].x + 2 * (float)k[1].x + 2 * (float)k[2].x + (float)k[3].x);
            dy = (float)(dt / 6) *
                ((float)k[0].y + 2 * (float)k[1].y + 2 * (float)k[2].y + (float)k[3].y);
        }
        p.x += dx;
        p.y += dy;
    }
    return p;
}

static SynPoint syn_center_at(const SynDeform *d, int z) {
    double dz = d->z0 + z * d->dz - d->mid, mm = dz / 1000;
    return (SynPoint){.x = d->center.x + d->tilt.x * dz + d->bend.x * (mm * mm),
        .y = d->center.y + d->tilt.y * dz + d->bend.y * (mm * mm)};
}

static SynPoint syn_forward(const SynDeform *d, int z, double rho, double theta, int placed) {
    SynPoint p = syn_crush_forward(d->crush[z], rho, syn_wrap(theta + d->start));
    double scale = syn_harmonic(d->harmonics, atan2(p.y, p.x)) / d->harmonic_norm;
    p.x *= scale;
    p.y *= scale;
    for (int i = 0; i < d->folds; i++) {
        p = syn_fold_apply(d->fold + i, p, 0);
    }
    p = syn_rotate(p, d->angle[z]);
    if (placed) {
        for (int i = 0; i < d->flows; i++) {
            p = syn_move(d->flow + i, z, p, 1);
        }
        SynPoint c = syn_center_at(d, z);
        p.x = p.x * d->mirror + c.x;
        p.y += c.y;
    }
    return p;
}

static SynBox syn_extent(const SynDeform *d, double rho, int count, int placed, double pad) {
    SynBox box = {.x0 = INFINITY, .y0 = INFINITY, .x1 = -INFINITY, .y1 = -INFINITY};
    double step = 2 * M_PI / count;
    for (int z = 0; z < d->nz; z++) {
        for (int i = 0; i < count; i++) {
            SynPoint p = syn_forward(d, z, rho, i * step, placed);
            box.x0 = fmin(box.x0, p.x);
            box.y0 = fmin(box.y0, p.y);
            box.x1 = fmax(box.x1, p.x);
            box.y1 = fmax(box.y1, p.y);
        }
    }
    return (SynBox){.x0 = box.x0 - pad, .y0 = box.y0 - pad, .x1 = box.x1 + pad, .y1 = box.y1 + pad};
}

static void syn_deform_plane(void *data, i64 iz) {
    SynDeform *d = data;
    SynPoint c = syn_center_at(d, iz);
    for (int y = 0; y < d->ny; y++) {
        for (int x = 0; x < d->nx; x++) {
            SynPoint p = {.x = (d->box.x0 + x * d->step - c.x) * d->mirror,
                .y = d->box.y0 + y * d->step - c.y};
            for (int i = d->flows - 1; i >= 0; i--) {
                p = syn_move(d->flow + i, iz, p, -1);
            }
            p = syn_rotate(p, -d->angle[iz]);
            for (int i = d->folds - 1; i >= 0; i--) {
                p = syn_fold_apply(d->fold + i, p, 1);
            }
            double scale = syn_harmonic(d->harmonics, atan2(p.y, p.x)) / d->harmonic_norm;
            SynPoint rest =
                syn_crush_inverse(d->crush[iz], (SynPoint){.x = p.x / scale, .y = p.y / scale});
            double theta = syn_wrap(rest.y - d->start);
            size_t i = ((size_t)iz * d->ny + y) * d->nx + x;
            d->a[i] = rest.x * cos(theta);
            d->b[i] = rest.x * sin(theta);
        }
    }
}

static void syn_deform(
    Arena *a, SynDeform *d, json_object *spec, SynRoll roll, const SynBox *wanted, int workers) {
    json_object *map = jget(spec, "map"), *shape = jget(spec, "shape"), *ct = jget(spec, "ct");
    json_object *flows = jget(spec, "flows"), *folds = jget(spec, "folds");
    json_object *place = jget(spec, "placement"), *band = jget(spec, "band_um");
    int version = ji64(spec, "version");
    double bottom = json_object_get_double(json_object_array_get_idx(band, 0));
    double top = json_object_get_double(json_object_array_get_idx(band, 1));
    double margin = jf64(spec, "margin_um"), dz = jf64(map, "knot_um"), outer = 0;
    *d = (SynDeform){.mid = 0.5 * (bottom + top),
        .z0 = bottom - margin,
        .step = jf64(map, "step_um"),
        .flows = json_object_array_length(flows),
        .folds = json_object_array_length(folds),
        .center = syn_pair(place, "centre_um"),
        .tilt = syn_pair(place, "tilt"),
        .bend = syn_pair(place, "bend_um"),
        .mirror = json_object_get_boolean(jget(place, "mirror")) ? -1 : 1,
        .start = jf64(jget(spec, "roll"), "start_deg") * (M_PI / 180),
        .harmonics = jget(shape, "harmonics"),
        .harmonic_norm = 1};
    double amplitude = 0;
    d->flow = pushz(a, d->flows * sizeof(*d->flow));
    for (int i = 0; i < d->flows; i++) {
        json_object *f = json_object_array_get_idx(flows, i);
        const char *kind = json_object_get_string(json_object_object_get(f, "kind"));
        d->flow[i].swirl = kind && !strcmp(kind, "swirl");
        if (!d->flow[i].swirl) {
            double scale = syn_optional(f, "z_scale_um", 4 * jf64(f, "scale_um"));
            dz = fmin(dz, version < 5 ? fmax(150, scale / 4) : fmax(100, scale / 6));
        }
        amplitude = fmax(amplitude, syn_optional(f, "amp_um", 0));
    }
    need(dz > 0 && d->step > 0, "invalid synthetic map spacing");
    d->dz = dz;
    d->nz = ceil((top - bottom + 2 * margin) / dz) + 2;
    for (int k = 0; k < roll.knots; k++) {
        outer = fmax(outer, roll.outer[k]);
    }
    d->crush = push(a, d->nz * sizeof(*d->crush));
    d->angle = push(a, d->nz * sizeof(*d->angle));
    double *height = malloc(d->nz * sizeof(*height));
    need(height != NULL, "cannot allocate synthetic map heights");
    for (int j = 0; j < d->nz; j++) {
        double z = d->z0 + dz * j - d->mid;
        height[j] = syn_clip(1 + jf64(shape, "kappa_slope_per_mm") * z / 1000, 0, 2);
        d->angle[j] =
            (jf64(shape, "angle_deg") + jf64(shape, "twist_deg_per_mm") * z / 1000) * (M_PI / 180);
        int old = 0;
        while (old < j && height[old] != height[j]) {
            old++;
        }
        d->crush[j] = old < j
            ? d->crush[old]
            : syn_crush(a, jget(shape, "crush"), roll.core, outer * 1.15 + 2000, height[j]);
    }
    free(height);
    if (version >= 6 && json_object_array_length(d->harmonics)) {
        double angles[720];
        for (int i = 0; i < 720; i++) {
            angles[i] = syn_harmonic(d->harmonics, i * (2 * M_PI / 720));
        }
        d->harmonic_norm = syn_sum(angles, 720) / 720;
    }
    d->fold = push(a, d->folds * sizeof(*d->fold));
    for (int i = 0; i < d->folds; i++) {
        json_object *f = json_object_array_get_idx(folds, i);
        double radians = fabs(jf64(f, "angle_deg")) * (M_PI / 180);
        double direction = jf64(f, "direction_deg") * (M_PI / 180), offset = jf64(f, "offset_um");
        d->fold[i] = (SynFold){.a = radians,
            .s = sin(direction),
            .c = cos(direction),
            .sa = sin(radians),
            .ca = cos(radians),
            .offset = fabs(offset),
            .position = jf64(f, "position_um"),
            .side = offset < 0 ? -1 : 1,
            .arc = fabs(offset) * radians,
            .sliding = version >= 6};
    }
    if (d->flows) {
        SynBox box = syn_extent(d, outer + 500, 720, 0, 4 * amplitude + 1000);
        for (int i = 0; i < d->flows; i++) {
            json_object *f = json_object_array_get_idx(flows, i);
            SynFlow *flow = d->flow + i;
            if (flow->swirl) {
                flow->steps = syn_optional(f, "steps", 24);
                flow->rate = jf64(f, "angle_deg") * (M_PI / 180);
                flow->radius = jf64(f, "radius_um");
                continue;
            }
            double scale = jf64(f, "scale_um"), amp = jf64(f, "amp_um");
            flow->steps = syn_optional(f, "steps", version < 4 ? 8 : 4);
            flow->x0 = box.x0;
            flow->y0 = box.y0;
            flow->step = scale / 4;
            flow->ny = ceil((box.y1 - box.y0) / flow->step) + 1;
            flow->nx = ceil((box.x1 - box.x0) / flow->step) + 1;
            size_t n = (size_t)d->nz * flow->ny * flow->nx;
            flow->vx = push(a, n * sizeof(*flow->vx));
            flow->vy = push(a, n * sizeof(*flow->vy));
            size_t before = a->used;
            double *tmp = push(a, n * sizeof(*tmp)), *value = push(a, n * sizeof(*value));
            Pcg64 rng = syn_rng(spec, fmt(a, "flow.%d", i));
            double sigma = syn_optional(f, "z_scale_um", 4 * scale) / fmax(dz, 1e-6);
            const double sig[] = {sigma, 4, 4};
            const int dims[] = {d->nz, flow->ny, flow->nx};
            float *components[] = {flow->vy, flow->vx};
            for (int c = 0; c < 2; c++) {
                syn_field(a, &rng, value, tmp, 3, dims, sig, 3, 0);
                syn_center(value, tmp, n);
                for (size_t j = 0; j < n; j++) {
                    components[c][j] = amp * value[j];
                }
            }
            a->used = before;
        }
    }
    d->box =
        wanted ? *wanted : syn_extent(d, outer + 300, 1440, 1, 4 * d->step + jf64(ct, "mask_um"));
    d->ny = ceil((d->box.y1 - d->box.y0) / d->step) + 1;
    d->nx = ceil((d->box.x1 - d->box.x0) / d->step) + 1;
    size_t n = (size_t)d->nx * d->ny * d->nz;
    d->a = push(a, n * sizeof(*d->a));
    d->b = push(a, n * sizeof(*d->b));
    parallel(d->nz, workers, syn_deform_plane, d);
}
