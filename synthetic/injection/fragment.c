enum {
    FRAGMENT_Z = 224,
    FRAGMENT_Y = 288,
    FRAGMENT_X = 288,
    FRAGMENT_CORE = 3,
    FRAGMENT_EXTENT = 8,
    FRAGMENT_NEED = 9
};

static const i64 fragment_dims[] = {FRAGMENT_Z, FRAGMENT_Y, FRAGMENT_X};

typedef struct {
    const u8 *ct, *clean;
    const float *inward, *outward;
    int rows, cols, depth, gap_rows, gap_cols;
} FragmentSlab;

typedef struct {
    SlabLibrary scan;
    SlabPoint *axis;
    int knots;
    size_t *keep, kept;
    InjectionNoise noise;
    const float *relief;
    InjectionShape relief_shape;
    double voxel, sigma;
    json_object *summary;
    glob_t slabs;
    FragmentSlab *loaded;
    Arena assets;
    void *gpu;
} InjectionFragments;

typedef struct {
    double half_u, half_v, tear, scale, reach, clean, voxel;
    int slab, row, col;
    u32 seed;
    const double *relief;
    int relief_side;
} InjectionFragment;

typedef struct {
    SlabPoint centre, frame[3];
    double least, *height;
    int side;
} InjectionSite;

typedef struct {
    InjectionFragment frag;
    InjectionSite site;
    json_object *record;
} FragmentPlacement;

typedef struct {
    u8 *ct, *index, *core;
    int count;
    FragmentPlacement *placed;
} FragmentBlock;

static InjectionFragments fragment_open(Arena *a, const char *scroll, const char *package) {
    InjectionFragments library = {.scan = slab_open(a, package, scroll), .assets = arena(2 * GB)};
    library.axis = slab_axis_read(a, &library.scan, &library.knots);
    library.keep = injection_truth_keep(a, &library.scan, &library.kept);
    const char *folder = injection_profile_folder(a, scroll);
    library.summary = json_read(fmt(a, "%s/profile.json", folder));
    const char *npz = fmt(a, "%s/profile.npz", folder);
    library.noise = injection_noise_open(a, npz);
    Array relief = npz_get(a, npz, "relief_spectrum");
    need(relief.ndim == 2 && relief.shape[0] == relief.shape[1] && !strcmp(relief.descr, "<f4"),
        "%s: invalid relief spectrum", npz);
    library.relief_shape = injection_shape((i64[]){1, relief.shape[0], relief.shape[1]});
    need(relief.size == library.relief_shape.n * sizeof(float), "%s: truncated relief spectrum",
        npz);
    library.relief = relief.data;
    json_object *response = jget(library.summary, "response");
    library.sigma =
        (jf64(response, "sigma_x") + jf64(response, "sigma_y") + jf64(response, "sigma_z")) / 3;
    glob_t meta = {0};
    need(!glob(fmt(a, "%s/library-*.json", folder), GLOB_NOSORT, NULL, &meta) && meta.gl_pathc,
        "%s: no slab library metadata", folder);
    json_object *slab = json_read(meta.gl_pathv[0]);
    library.voxel = jf64(slab, "voxel_um");
    json_object_put(slab);
    globfree(&meta);
    need(!glob(fmt(a, "%s/slab-*.npz", folder), 0, NULL, &library.slabs) &&
            library.slabs.gl_pathc && library.slabs.gl_pathc <= INT_MAX,
        "%s: no fragment material slabs", folder);
    qsort(library.slabs.gl_pathv, library.slabs.gl_pathc, sizeof(*library.slabs.gl_pathv),
        slab_mesh_order);
    library.loaded = pushz(&library.assets, library.slabs.gl_pathc * sizeof(*library.loaded));
    library.gpu = injection_kernel(a, "fragment", "inject_fragment");
    return library;
}

static void fragment_close(InjectionFragments *library) {
    hx_inj_close(library->gpu);
    hx_cached_ct_close(&library->scan.ct);
    json_object_put(library->summary);
    globfree(&library->slabs);
    munmap(library->assets.base, library->assets.size);
}

static u8 *fragment_real_block(
    Arena *a, InjectionFragments *library, Pcg64 *rng, int *tries_left, SlabPoint *origin_xyz) {
    size_t mark = a->used;
    const float *points = library->scan.points.data;
    while (*tries_left > 0) {
        (*tries_left)--;
        const float *p = points + 3 * library->keep[draw(rng, library->kept)];
        i64 z = (i64)nearbyint(p[2]), y = (i64)nearbyint(p[1]), x = (i64)nearbyint(p[0]);
        if (z >= library->scan.low - FRAGMENT_Z && z <= library->scan.high + FRAGMENT_Z) {
            continue;
        }
        i64 box[] =
            {
                z - FRAGMENT_Z / 2,
                z + FRAGMENT_Z / 2,
                y - FRAGMENT_Y / 2,
                y + FRAGMENT_Y / 2,
                x - FRAGMENT_X / 2,
                x + FRAGMENT_X / 2,
            },
            dims[3];
        u8 *out = slab_read(a, &library->scan, box, dims);
        size_t n = (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X, zeros = 0;
        for (size_t i = 0; i < n; i++) {
            zeros += out[i] == 0;
        }
        if ((double)zeros / n <= .001) {
            *origin_xyz = (SlabPoint){{box[4], box[2], box[0]}};
            return out;
        }
        a->used = mark;
    }
    return NULL;
}

static int fragment_sheet_rule(
    const double *column, int length, double step, double *inward, double *outward) {
    need(length > 0 && length % 2 && step > 0 && 4.0 / step < 500, "invalid sheet-rule grid");
    int radius = (int)(4.0 / step + .5), mid = length / 2;
    double sigma = 1 / step, weights[501], norm = 0;
    for (int i = -radius; i <= radius; i++) {
        double w = exp(-0.5 / (sigma * sigma) * i * i);
        norm += w;
        weights[abs(i)] = w;
    }
    for (int i = 0; i <= radius; i++) {
        weights[i] /= norm;
    }
    int second = 0;
    for (int k = 0; k < 2; k++) {
        int sign = k ? -1 : 1, below = 0, found = 0;
        double low = INFINITY, room = mid * step;
        for (int i = 0; i <= mid; i++) {
            double d = i * step, at = mid + i * sign;
            double val = column[(int)at] * weights[0];
            for (int j = radius; j > 0; j--) {
                val += (column[reflected((i64)at - j, length)] +
                           column[reflected((i64)at + j, length)]) *
                    weights[j];
            }
            if (val <= 80) {
                below = 1;
            } else if (below && !found) {
                room = d;
                found = 1;
            }
            low = fmin(low, val);
            second |= d >= 1.5 && d <= 8 && val - low >= 15;
        }
        *(k ? outward : inward) = room;
    }
    return second;
}

static FragmentSlab *fragment_slab(InjectionFragments *lib, int index) {
    FragmentSlab *slab = lib->loaded + index;
    if (slab->ct) {
        return slab;
    }
    Arena *a = &lib->assets;
    const char *path = lib->slabs.gl_pathv[index];
    Array ct = npz_get(a, path, "ct");
    Array in = npz_get(a, path, "half_in"), out = npz_get(a, path, "half_out");
    need(ct.ndim == 3 && ct.shape[0] == SLAB_SIDE && ct.shape[1] == SLAB_SIDE &&
            ct.shape[2] == 25 && !strcmp(ct.descr, "|u1") && ct.size == (size_t)elements(ct) &&
            in.ndim == 2 && out.ndim == 2 && !strcmp(in.descr, "<f2") &&
            !strcmp(out.descr, "<f2") && in.shape[0] == SLAB_SIDE / SLAB_GAP_STEP &&
            in.shape[1] == in.shape[0] && !memcmp(in.shape, out.shape, sizeof(i64) * 2) &&
            in.size == (size_t)elements(in) * 2 && out.size == in.size,
        "%s: invalid fragment slab or half gaps", path);
    int side = (int)in.shape[0];
    size_t count = (size_t)side * side;
    u8 *clean = push(a, count);
    float *inner = push(a, count * sizeof(*inner)), *outer = push(a, count * sizeof(*outer));
    for (int r = 0; r < side; r++) {
        for (int c = 0; c < side; c++) {
            int y = r * SLAB_GAP_STEP + SLAB_GAP_STEP / 2,
                x = c * SLAB_GAP_STEP + SLAB_GAP_STEP / 2;
            const u8 *p = (const u8 *)ct.data + ((size_t)y * SLAB_SIDE + x) * 25;
            double col[25], room_in, room_out;
            for (int i = 0; i < 25; i++) {
                col[i] = p[i];
            }
            int second = fragment_sheet_rule(col, 25, 1.0, &room_in, &room_out);
            size_t at = (size_t)r * side + c;
            int single = !second && room_in >= 10 && room_out >= 10;
            clean[at] = single;
            inner[at] =
                single ? FRAGMENT_EXTENT : fminf(FRAGMENT_EXTENT, ((const _Float16 *)in.data)[at]);
            outer[at] =
                single ? FRAGMENT_EXTENT : fminf(FRAGMENT_EXTENT, ((const _Float16 *)out.data)[at]);
        }
    }
    *slab = (FragmentSlab){ct.data, clean, inner, outer, SLAB_SIDE, SLAB_SIDE, 25, side, side};
    return slab;
}

static InjectionFragment fragment_draw(Arena *a, InjectionFragments *lib, Pcg64 *rng) {
    double radius = 30 + 26 * pcg64_double(rng), aspect = .6 + .4 * pcg64_double(rng);
    int wide = pcg64_double(rng) < .5;
    InjectionFragment f = {
        .half_u = wide ? radius : radius * aspect,
        .half_v = wide ? radius * aspect : radius,
        .clean = -1,
        .reach = radius * (1 + 1.5 * .2) + 2,
        .voxel = lib->voxel,
    };
    int low = (int)f.reach, span = SLAB_SIDE - low * 2;
    for (int i = 0; i < 300; i++) {
        int k = (int)draw(rng, lib->slabs.gl_pathc);
        int row = low + (int)draw(rng, span), col = low + (int)draw(rng, span);
        FragmentSlab *slab = fragment_slab(lib, k);
        int n = 0, yes = 0;
        for (int y = 0; y < slab->gap_rows; y++) {
            double v = (y * SLAB_GAP_STEP - row) / f.half_v;
            if (fabs(v) > 1) {
                continue;
            }
            for (int x = 0; x < slab->gap_cols; x++) {
                double u = (x * SLAB_GAP_STEP - col) / f.half_u;
                if (u * u + v * v <= 1) {
                    n++;
                    yes += slab->clean[(size_t)y * slab->gap_cols + x];
                }
            }
        }
        double share = n ? (double)yes / n : 0;
        if (share > f.clean) {
            f.clean = share;
            f.slab = k;
            f.row = row;
            f.col = col;
        }
        if (share >= .9) {
            break;
        }
    }
    f.tear = (.05 + (.2 - .05) * pcg64_double(rng)) * radius * f.voxel;
    f.scale = 80 + (300 - 80) * pcg64_double(rng);
    f.seed = pcg64_next32(rng);
    InjectionShape shape = lib->relief_shape;
    double *relief = push(a, shape.n * sizeof(*relief));
    size_t mark = a->used;
    InjectionComplex *fspace = push(a, shape.n * sizeof(*fspace));
    for (size_t i = 0; i < shape.n; i++) {
        fspace[i] = (InjectionComplex){syn_normal(rng), 0};
    }
    injection_fft(a, fspace, shape, 0);
    for (size_t i = 0; i < shape.n; i++) {
        double amplitude = sqrt(fmax(0, lib->relief[i]));
        fspace[i].re *= amplitude;
        fspace[i].im *= amplitude;
    }
    injection_fft(a, fspace, shape, 1);
    for (size_t i = 0; i < shape.n; i++) {
        relief[i] = fspace[i].re;
    }
    a->used = mark;
    f.relief = relief;
    f.relief_side = shape.dim[1];
    return f;
}

static double fragment_dot(SlabPoint a, SlabPoint b) {
    return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
}

static SlabPoint fragment_cross(SlabPoint a, SlabPoint b) {
    return (SlabPoint){{a.v[1] * b.v[2] - a.v[2] * b.v[1], a.v[2] * b.v[0] - a.v[0] * b.v[2],
        a.v[0] * b.v[1] - a.v[1] * b.v[0]}};
}

static SlabPoint fragment_normalise(SlabPoint p, double minimum) {
    double n = fmax(minimum, sqrt(fragment_dot(p, p)));
    for (int k = 0; k < 3; k++) {
        p.v[k] /= n;
    }
    return p;
}

static void fragment_frame(SlabPoint normal, SlabPoint inward, SlabPoint frame[3]) {
    SlabPoint ew = fragment_normalise(normal, 1e-30);
    if (fragment_dot(ew, inward) < 0) {
        for (int k = 0; k < 3; k++) {
            ew.v[k] = -ew.v[k];
        }
    }
    SlabPoint eu = fragment_normalise(fragment_cross((SlabPoint){{1, 0, 0}}, ew), 1e-9);
    frame[0] = eu;
    frame[1] = fragment_cross(ew, eu);
    frame[2] = ew;
}

static SlabPoint fragment_point(
    SlabPoint centre, const SlabPoint frame[3], double u, double v, double h) {
    SlabPoint out;
    for (int k = 0; k < 3; k++) {
        out.v[k] = centre.v[k] + u * frame[0].v[k] + v * frame[1].v[k] + h * frame[2].v[k];
    }
    return out;
}

static size_t fragment_index(int z, int y, int x) {
    return ((size_t)z * FRAGMENT_Y + y) * FRAGMENT_X + x;
}

static float fragment_ct_linear(const u8 *raw, SlabPoint p) {
    int lo[3], hi[3];
    double t[3];
    for (int k = 0; k < 3; k++) {
        double x = p.v[k];
        if (x < 0 || x > fragment_dims[k] - 1) {
            return 0;
        }
        lo[k] = (int)x;
        hi[k] = (int)slab_nearest(lo[k] + 1, fragment_dims[k]);
        t[k] = x - lo[k];
    }
    double total = 0;
    for (int dz = 0; dz < 2; dz++) {
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                size_t at =
                    fragment_index(dz ? hi[0] : lo[0], dy ? hi[1] : lo[1], dx ? hi[2] : lo[2]);
                total += raw[at] * (dz ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) *
                    (dx ? t[2] : 1 - t[2]);
            }
        }
    }
    return (float)total;
}

static SlabPoint fragment_local_normal(Arena *a, const u8 *block, SlabPoint centre) {
    size_t mark = a->used;
    enum { WINDOW = 20, WIDTH = WINDOW * 2 + 1, N = WIDTH * WIDTH * WIDTH };
    int lo[3];
    for (int k = 0; k < 3; k++) {
        lo[k] = (int)centre.v[k] - WINDOW;
    }
    float *ct = push(a, N * sizeof(*ct)), *spare = push(a, N * sizeof(*spare));
    for (int z = 0; z < WIDTH; z++) {
        for (int y = 0; y < WIDTH; y++) {
            const u8 *in = block + fragment_index(lo[0] + z, lo[1] + y, lo[2]);
            for (int x = 0; x < WIDTH; x++) {
                ct[(z * WIDTH + y) * WIDTH + x] = in[x];
            }
        }
    }
    gaussian(ct, spare, (i64[]){WIDTH, WIDTH, WIDTH}, 3, (double[]){1.5, 1.5, 1.5});
    const int stride[] = {WIDTH * WIDTH, WIDTH, 1};
    double tensor[9] = {0};
    for (int z = 0; z < WIDTH; z++) {
        for (int y = 0; y < WIDTH; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int place[] = {z, y, x}, at = (z * WIDTH + y) * WIDTH + x;
                float derivative[3];
                double d2 = 0;
                for (int k = 0; k < 3; k++) {
                    int low = place[k] > 0, high = place[k] + 1 < WIDTH;
                    derivative[k] =
                        (ct[at + high * stride[k]] - ct[at - low * stride[k]]) / (low + high);
                    d2 += (place[k] - WINDOW) * (place[k] - WINDOW);
                }
                double weight = exp(-d2 / 200);
                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 3; i++) {
                        tensor[j * 3 + i] += (derivative[j] * derivative[i]) * weight;
                    }
                }
            }
        }
    }
    char vectors = 'V', lower = 'L';
    int side = 3, count = 32, status;
    double values[3], work[32];
    dsyev_(&vectors, &lower, &side, tensor, &side, values, work, &count, &status);
    need(!status, "cannot fit the local fragment normal");
    a->used = mark;
    return (SlabPoint){{tensor[6], tensor[7], tensor[8]}};
}

static int fragment_mod(int n, int d) {
    int r = n % d;
    return r < 0 ? r + d : r;
}

static u32 fragment_mix(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

static double fragment_lattice(i64 x, i64 y, u32 seed) {
    u32 key = (u32)x * 0x8da6b343U ^ (u32)y * 0xd8163841U ^ seed;
    return (fragment_mix(key) & 0xffff) / 32767.5 - 1;
}

static double fragment_value_noise(double x, double y, u32 seed) {
    i64 a = (i64)floor(x), b = (i64)floor(y);
    double tx = x - a, ty = y - b;
    tx = tx * tx * (3 - 2 * tx);
    ty = ty * ty * (3 - 2 * ty);
    double a0 = fragment_lattice(a, b, seed), a1 = fragment_lattice(a + 1, b, seed);
    double b0 = fragment_lattice(a, b + 1, seed), b1 = fragment_lattice(a + 1, b + 1, seed);
    double low = a0 + (a1 - a0) * tx, high = b0 + (b1 - b0) * tx;
    return low + (high - low) * ty;
}

static double fragment_rim(const InjectionFragment *f, double u, double v) {
    double x = u / f->half_u, y = v / f->half_v;
    double radius = hypot(x, y), theta = atan2(y, x), c = cos(theta), s = sin(theta);
    double reach = f->half_u * f->half_v / fmax(hypot(f->half_v * c, f->half_u * s), 1e-3);
    double scale = f->scale / f->voxel;
    double rx = c * reach / scale, ry = s * reach / scale;
    double tear = fragment_value_noise(rx, ry, f->seed ^ 0x3c6ef372U) +
        .5 * fragment_value_noise(2 * rx + 17, 2 * ry + 17, f->seed ^ 0x9e3779b9U);
    return (1 - radius) * reach + f->tear / f->voxel * tear;
}

static double fragment_height(const InjectionSite *site, double u, double v) {
    int n = site->side;
    double x = u + (n - 1) * .5, y = v + (n - 1) * .5;
    i64 x0 = (i64)floor(x), y0 = (i64)floor(y);
    double tx = x - x0, ty = y - y0;
    i64 xa = slab_nearest(x0, n), xb = slab_nearest(x0 + 1, n);
    i64 ya = slab_nearest(y0, n), yb = slab_nearest(y0 + 1, n);
    const double *h = site->height;
    return (1 - ty) * ((1 - tx) * h[ya * n + xa] + tx * h[ya * n + xb]) +
        ty * ((1 - tx) * h[yb * n + xa] + tx * h[yb * n + xb]);
}

static size_t fragment_mid_surface(Arena *a, const FragmentPlacement *placed, double step,
    double inner, SlabPoint **points, SlabPoint **normals) {
    const InjectionFragment *f = &placed->frag;
    const InjectionSite *site = &placed->site;
    size_t side = (size_t)ceil((2 * f->reach + step) / step), count = 0;
    *points = push(a, side * side * sizeof(**points));
    if (normals) {
        *normals = push(a, side * side * sizeof(**normals));
    }
    double increment = (-f->reach + step) - (-f->reach);
    for (size_t y = 0; y < side; y++) {
        double u = -f->reach + y * increment;
        for (size_t x = 0; x < side; x++) {
            double v = -f->reach + x * increment;
            if (fragment_rim(f, u, v) < inner) {
                continue;
            }
            (*points)[count] =
                fragment_point(site->centre, site->frame, u, v, fragment_height(site, u, v));
            if (normals) {
                double du = fragment_height(site, u + .5, v) - fragment_height(site, u - .5, v);
                double dv = fragment_height(site, u, v + .5) - fragment_height(site, u, v - .5);
                SlabPoint n = {{0, 0, 0}};
                for (int k = 0; k < 3; k++) {
                    n.v[k] =
                        site->frame[2].v[k] - du * site->frame[0].v[k] - dv * site->frame[1].v[k];
                }
                (*normals)[count] = fragment_normalise(n, 1e-30);
            }
            count++;
        }
    }
    return count;
}

static void fragment_gaussian_nearest(double *data, double *spare, int side) {
    double weights[13], sum = 0;
    for (int i = -12; i <= 12; i++) {
        double w = exp(-.5 / 9.0 * i * i);
        sum += w;
        weights[abs(i)] = w;
    }
    for (int i = 0; i <= 12; i++) {
        weights[i] /= sum;
    }
    for (int axis = 0; axis < 2; axis++) {
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                const double *from = axis ? spare : data;
                double *to = axis ? data : spare;
                int p = axis ? x : y, stride = axis ? 1 : side;
                const double *line = from + (axis ? y * side : x);
                double value = weights[0] * line[p * stride];
                for (int r = 12; r > 0; r--) {
                    value += weights[r] *
                        (line[slab_nearest(p - r, side) * stride] +
                            line[slab_nearest(p + r, side) * stride]);
                }
                to[y * side + x] = value;
            }
        }
    }
}

static void fragment_pulls(Arena *a, const u8 *papyrus, SlabPoint centre, const SlabPoint frame[3],
    int radius, double *pull, double *weight) {
    int side = radius * 2 + 1;
    size_t mark = a->used, pixels = (size_t)side * side;
    double *first = push(a, pixels * sizeof(*first));
    double *spare = push(a, pixels * sizeof(*spare));
    memset(pull, 0, pixels * sizeof(*pull));
    memset(weight, 0, pixels * sizeof(*weight));
    for (int sign = 1; sign >= -1; sign -= 2) {
        double edges[9];
        int found = 0;
        for (int row = 0; row < side; row++) {
            for (int col = 0; col < side; col++) {
                size_t at = (size_t)row * side + col;
                first[at] = 0;
                double u = (row - radius) * 3.0, v = (col - radius) * 3.0;
                SlabPoint base = fragment_point(centre, frame, u, v, 0);
                for (int step = 1; step <= 48; step++) {
                    double h = .5 * step;
                    SlabPoint p = base;
                    for (int k = 0; k < 3; k++) {
                        p.v[k] += sign * h * frame[2].v[k];
                    }
                    if (fragment_ct_linear(papyrus, p) > .5f) {
                        first[at] = h;
                        if (abs(row - radius) <= 1 && abs(col - radius) <= 1) {
                            edges[found++] = h;
                        }
                        break;
                    }
                }
            }
        }
        if (!found) {
            continue;
        }
        qsort(edges, found, sizeof(*edges), slab_order);
        double reference = slab_quantile(edges, found, .5);
        for (size_t i = 0; i < pixels; i++) {
            if (first[i] > 0) {
                double w = 1 / first[i];
                pull[i] += w * sign * (first[i] - reference);
                weight[i] += w;
            }
        }
    }
    fragment_gaussian_nearest(pull, spare, side);
    fragment_gaussian_nearest(weight, spare, side);
    for (size_t i = 0; i < pixels; i++) {
        pull[i] = weight[i] > 1e-6 ? pull[i] / weight[i] : 0;
    }
    a->used = mark;
}

static int fragment_weighted_plane(
    const double *pull, const double *weight, int radius, double *fit) {
    int side = radius * 2 + 1;
    double table[3][4] = {{0}};
    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            size_t p = (size_t)y * side + x;
            double row[] = {1, (y - radius) * 3.0, (x - radius) * 3.0, pull[p]};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    table[i][j] += weight[p] * row[i] * row[j];
                }
            }
        }
    }
    if (table[0][0] <= 0) {
        return 0;
    }
    for (int basis = 0; basis < 3; basis++) {
        int row = basis;
        for (int i = basis + 1; i < 3; i++) {
            if (fabs(table[i][basis]) > fabs(table[row][basis])) {
                row = i;
            }
        }
        if (fabs(table[row][basis]) < 1e-20) {
            return 0;
        }
        for (int j = basis; j < 4; j++) {
            double t = table[basis][j];
            table[basis][j] = table[row][j];
            table[row][j] = t;
        }
        double divisor = table[basis][basis];
        for (int j = basis; j < 4; j++) {
            table[basis][j] /= divisor;
        }
        for (int i = 0; i < 3; i++) {
            if (i == basis) {
                continue;
            }
            double v = table[i][basis];
            for (int j = basis; j < 4; j++) {
                table[i][j] -= v * table[basis][j];
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        fit[i] = table[i][3];
    }
    return 1;
}

static double fragment_spline_2d(SlabSpline spline, double y, double x) {
    SlabTaps ty = slab_taps(y, spline.dims[1], spline.stride[1]);
    SlabTaps tx = slab_taps(x, spline.dims[2], spline.stride[2]);
    double sum = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sum += spline.data[ty.index[i] + tx.index[j]] * ty.weight[i] * tx.weight[j];
        }
    }
    return sum;
}

static void fragment_lie(Arena *a, const u8 *papyrus, InjectionSite *site,
    const InjectionFragment *f, SlabPoint inward) {
    enum { PAD = 12 };
    int half = (int)ceil(f->reach), radius = (int)ceil(half / 3.0), side = radius * 2 + 1;
    site->side = 2 * half + 1;
    site->height = push(a, (size_t)site->side * site->side * sizeof(*site->height));
    size_t mark = a->used;
    double *pull = push(a, (size_t)side * side * sizeof(*pull));
    double *weight = push(a, (size_t)side * side * sizeof(*weight));
    fragment_pulls(a, papyrus, site->centre, site->frame, radius, pull, weight);
    double fit[3];
    if (fragment_weighted_plane(pull, weight, radius, fit)) {
        SlabPoint normal = site->frame[2];
        for (int k = 0; k < 3; k++) {
            normal.v[k] -= fit[1] * site->frame[0].v[k] + fit[2] * site->frame[1].v[k];
        }
        fragment_frame(normal, inward, site->frame);
        fragment_pulls(a, papyrus, site->centre, site->frame, radius, pull, weight);
    }
    int wider = side + 2 * PAD;
    SlabSpline spline = slab_spline(a, 1, wider, wider);
    for (int y = 0; y < wider; y++) {
        for (int x = 0; x < wider; x++) {
            i64 sy = slab_nearest(y - PAD, side), sx = slab_nearest(x - PAD, side);
            spline.data[(size_t)y * wider + x] = pull[sy * side + sx];
        }
    }
    slab_prefilter(spline);
    double base = fragment_spline_2d(spline, radius + PAD, radius + PAD);
    int tile = f->relief_side;
    double relief_base = f->relief[(size_t)(tile / 2) * tile + tile / 2];
    for (int y = 0; y < site->side; y++) {
        int ry = fragment_mod(y - half + tile / 2, tile);
        for (int x = 0; x < site->side; x++) {
            int rx = fragment_mod(x - half + tile / 2, tile);
            double h = fragment_spline_2d(
                spline, (x - half) / 3.0 + radius + PAD, (y - half) / 3.0 + radius + PAD);
            site->height[(size_t)y * site->side + x] =
                (h - base) + .5 * (f->relief[(size_t)ry * tile + rx] - relief_base);
        }
    }
    a->used = mark;
}

static void fragment_distance_line(i32 *line, int n, int *sources, double *limits) {
    enum { MISSING = INT_MAX / 4 };
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (line[i] >= MISSING) {
            continue;
        }
        double begin = -INFINITY;
        while (count) {
            int a = sources[count - 1];
            begin = ((double)line[i] - line[a] + i * i - a * a) / (2 * (i - a));
            if (begin > limits[count - 1]) {
                break;
            }
            count--;
        }
        sources[count] = i;
        limits[count++] = begin;
    }
    if (!count) {
        return;
    }
    int copy[FRAGMENT_X];
    memcpy(copy, line, (size_t)n * sizeof(*line));
    int k = 0;
    for (int i = 0; i < n; i++) {
        while (k + 1 < count && i > limits[k + 1]) {
            k++;
        }
        int at = sources[k], delta = i - at;
        line[i] = copy[at] + delta * delta;
    }
}

static void fragment_air(Arena *a, const u8 *block, u8 *mask, i32 *distance) {
    size_t mark = a->used, pixels = (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X;
    float *smooth = push(a, pixels * sizeof(*smooth)),
          *scratch = push(a, pixels * sizeof(*scratch));
    for (size_t i = 0; i < pixels; i++) {
        smooth[i] = block[i];
    }
    gaussian(smooth, scratch, fragment_dims, 3, (double[]){1, 1, 1});
    size_t covered = 0;
    for (size_t i = 0; i < pixels; i++) {
        mask[i] = smooth[i] > 80;
        covered += mask[i];
        distance[i] = mask[i] ? 0 : INT_MAX / 4;
    }
    a->used = mark;
    if (!covered) {
        for (int z = 0; z < FRAGMENT_Z; z++) {
            for (int y = 0; y < FRAGMENT_Y; y++) {
                for (int x = 0; x < FRAGMENT_X; x++) {
                    distance[fragment_index(z, y, x)] = (z + 1) * (z + 1) + y * y + x * x;
                }
            }
        }
        return;
    }
    i32 line[FRAGMENT_X];
    int sources[FRAGMENT_X];
    double limits[FRAGMENT_X];
    size_t stride = 1;
    for (int axis = 2; axis >= 0; axis--) {
        int n = (int)fragment_dims[axis];
        for (size_t offset = 0; offset < pixels; offset += stride * n) {
            for (size_t k = 0; k < stride; k++) {
                size_t base = offset + k;
                for (int i = 0; i < n; i++) {
                    line[i] = distance[base + i * stride];
                }
                fragment_distance_line(line, n, sources, limits);
                for (int i = 0; i < n; i++) {
                    distance[base + i * stride] = line[i];
                }
            }
        }
        stride *= n;
    }
}

static double fragment_room(Arena *a, const i32 *distance, const FragmentPlacement *placed) {
    size_t mark = a->used;
    SlabPoint *points;
    size_t count = fragment_mid_surface(a, placed, 3, -1, &points, NULL);
    i32 least = INT_MAX;
    for (size_t i = 0; i < count; i++) {
        SlabPoint p = points[i];
        int z = (int)nearbyint(p.v[0]), y = (int)nearbyint(p.v[1]), x = (int)nearbyint(p.v[2]);
        if (z < 0 || z >= FRAGMENT_Z || y < 0 || y >= FRAGMENT_Y || x < 0 || x >= FRAGMENT_X) {
            least = 0;
            break;
        }
        i32 d = distance[fragment_index(z, y, x)];
        if (d < least) {
            least = d;
        }
        if (least < FRAGMENT_NEED * FRAGMENT_NEED) {
            break;
        }
    }
    a->used = mark;
    return count && least >= FRAGMENT_NEED * FRAGMENT_NEED ? sqrt(least) : 0;
}

static int fragment_place(Arena *a, InjectionFragments *library, const u8 *block, const u8 *mask,
    const i32 *distance, const int *candidates, int n, SlabPoint origin_xyz,
    FragmentPlacement *placed, const FragmentPlacement *previous, int count, Pcg64 *rng) {
    if (!n) {
        return 0;
    }
    size_t mark = a->used;
    int *order = push(a, (size_t)n * sizeof(*order));
    memcpy(order, candidates, (size_t)n * sizeof(*order));
    for (int i = n - 1; i > 0; i--) {
        int k = disc_interval(rng, i), t = order[i];
        order[i] = order[k];
        order[k] = t;
    }
    size_t candidate_boundary = a->used;
    const int trials = n < 400 ? n : 400;
    for (int i = 0; i < trials; i++) {
        int j = order[i];
        int plane = j / (FRAGMENT_Y * FRAGMENT_X), row = j / FRAGMENT_X % FRAGMENT_Y;
        SlabPoint centre = {{plane, row, j % FRAGMENT_X}};
        int overlap = 0;
        for (int k = 0; k < count; k++) {
            SlabPoint d;
            for (int axis = 0; axis < 3; axis++) {
                d.v[axis] = centre.v[axis] - previous[k].site.centre.v[axis];
            }
            double reach = previous[k].frag.reach + placed->frag.reach + 4;
            overlap |= fragment_dot(d, d) < reach * reach;
        }
        if (overlap) {
            continue;
        }
        SlabPoint world;
        for (int k = 0; k < 3; k++) {
            world.v[k] = origin_xyz.v[k] + centre.v[2 - k];
        }
        SlabPoint axis = slab_axis(library->axis, library->knots, world.v[2]);
        SlabPoint inward = {{0, axis.v[2] - world.v[1], axis.v[1] - world.v[0]}};
        InjectionSite *site = &placed->site;
        site->centre = centre;
        fragment_frame(fragment_local_normal(a, block, centre), inward, site->frame);
        fragment_lie(a, mask, site, &placed->frag, inward);
        site->least = fragment_room(a, distance, placed);
        if (site->least) {
            size_t bytes = (size_t)site->side * site->side * sizeof(*site->height);
            a->used = mark;
            double *height = push(a, bytes);
            memmove(height, site->height, bytes);
            site->height = height;
            return 1;
        }
        a->used = candidate_boundary;
    }
    a->used = mark;
    return 0;
}

static float *fragment_noise(Arena *a, InjectionNoise profile, Pcg64 *rng) {
    size_t count = (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X;
    float *noise = push(a, count * sizeof(*noise));
    size_t mark = a->used;
    const int *shape = profile.shape.dim;
    int bz = (FRAGMENT_Z - 1) / shape[0] + 1, by = (FRAGMENT_Y - 1) / shape[1] + 1;
    int bx = (FRAGMENT_X - 1) / shape[2] + 1;
    size_t cubes = (size_t)bz * by * bx;
    double *drawn = push(a, cubes * profile.shape.n * sizeof(*drawn));
    injection_draw_noise(a, profile, 0, cubes, rng, drawn);
    for (int z = 0; z < FRAGMENT_Z; z++) {
        for (int y = 0; y < FRAGMENT_Y; y++) {
            for (int x = 0; x < FRAGMENT_X; x++) {
                size_t cube = ((size_t)(z / shape[0]) * by + y / shape[1]) * bx + x / shape[2];
                size_t p =
                    ((size_t)(z % shape[0]) * shape[1] + y % shape[1]) * shape[2] + x % shape[2];
                noise[fragment_index(z, y, x)] = (float)drawn[cube * profile.shape.n + p];
            }
        }
    }
    a->used = mark;
    return noise;
}

typedef struct {
    float centre[4], eu[4], ev[4], ew[4];
    float half_u, half_v, tear_um, tear_scale_um, sigma_um, core_um, voxel;
    i32 slab_rows, slab_cols, slab_depth, row0, col0;
    i32 gap_step, gap_rows, gap_cols, height_size;
    u32 seed;
    i32 nz, ny, nx;
} InjectionFragmentParams;

_Static_assert(sizeof(InjectionFragmentParams) == 144, "fragment kernel ABI");

static json_object *fragment_values(const double *values, int count, double scale) {
    json_object *out = json_object_new_array();
    for (int i = 0; i < count; i++) {
        json_object_array_add(out, injection_number(injection_round(values[i], scale)));
    }
    return out;
}

static json_object *fragment_record(Arena *a, const InjectionFragments *library,
    const FragmentPlacement *placed, SlabPoint origin_xyz, double cap, const float *height,
    i64 core, i64 changed, int index) {
    const InjectionFragment *f = &placed->frag;
    const InjectionSite *s = &placed->site;
    json_object *out = json_object_new_object(), *frame = json_object_new_array();
    double centre[] = {s->centre.v[2] + origin_xyz.v[0], s->centre.v[1] + origin_xyz.v[1],
        s->centre.v[0] + origin_xyz.v[2]};
    for (int k = 0; k < 3; k++) {
        const double *v = s->frame[k].v;
        json_object_array_add(frame, fragment_values((double[]){v[2], v[1], v[0]}, 3, 1e6));
    }
    const char *path = library->slabs.gl_pathv[f->slab], *leaf = strrchr(path, '/') + 1;
    const char *filename = fmt(a, "%.*s", (int)(strlen(leaf) - 4), leaf);
    char hash[65];
    size_t n = (size_t)s->side * s->side;
    sha256_hex(height, n * sizeof(*height), hash);
    double spread = sqrt(injection_moments(s->height, n).variance);
    json_object_object_add(out, "kind", json_object_new_string("fragment"));
    json_object_object_add(out, "index", json_object_new_int(index));
    json_object_object_add(out, "centre_xyz", fragment_values(centre, 3, 1000));
    json_object_object_add(out, "frame_xyz", frame);
    json_object_object_add(
        out, "half_voxels", fragment_values((double[]){f->half_u, f->half_v}, 2, 1000));
    json_object_object_add(out, "tear_um", jnum(injection_round(f->tear, 1000)));
    json_object_object_add(out, "tear_scale_um", jnum(injection_round(f->scale, 1000)));
    json_object_object_add(out, "relief_share", jnum(.5));
    json_object_object_add(out, "height_spread_voxels", jnum(injection_round(spread, 10000)));
    json_object_object_add(out, "height_sha256", json_object_new_string(hash));
    json_object_object_add(out, "slab", json_object_new_string(filename));
    json_object *place = json_object_new_array();
    json_object_array_add(place, json_object_new_int(f->row));
    json_object_array_add(place, json_object_new_int(f->col));
    json_object_object_add(out, "slab_place", place);
    json_object_object_add(out, "slab_clean_share", jnum(injection_round(f->clean, 1000)));
    json_object_object_add(out, "seed", json_object_new_uint64(f->seed));
    json_object_object_add(out, "least_air_voxels", jnum(injection_round(s->least, 1000)));
    json_object_object_add(out, "cap_voxels", jnum(injection_round(cap, 1000)));
    json_object_object_add(out, "core_voxels", json_object_new_int64(core));
    json_object_object_add(out, "changed_voxels", json_object_new_int64(changed));
    return out;
}

static void fragment_render(Arena *a, InjectionFragments *library, FragmentBlock *out,
    FragmentPlacement *placed, const float *noise, SlabPoint origin_xyz) {
    size_t mark = a->used, pixels = (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X;
    const InjectionFragment *f = &placed->frag;
    const InjectionSite *s = &placed->site;
    const FragmentSlab *slab = fragment_slab(library, f->slab);
    double cap = fmax(FRAGMENT_CORE, fmin(FRAGMENT_EXTENT, s->least - 1 - library->sigma));
    size_t gaps = (size_t)slab->gap_rows * slab->gap_cols, n = (size_t)s->side * s->side;
    float *in = push(a, gaps * sizeof(*in)), *ex = push(a, gaps * sizeof(*ex));
    float *height = push(a, n * sizeof(*height));
    for (size_t i = 0; i < gaps; i++) {
        in[i] = fminf((float)cap, slab->inward[i]);
        ex[i] = fminf((float)cap, slab->outward[i]);
    }
    for (size_t i = 0; i < n; i++) {
        height[i] = (float)s->height[i];
    }
    InjectionFragmentParams params = {
        .half_u = f->half_u * f->voxel,
        .half_v = f->half_v * f->voxel,
        .tear_um = f->tear,
        .tear_scale_um = f->scale,
        .sigma_um = library->sigma * f->voxel,
        .core_um = FRAGMENT_CORE * f->voxel,
        .voxel = f->voxel,
        .slab_rows = slab->rows,
        .slab_cols = slab->cols,
        .slab_depth = slab->depth,
        .row0 = f->row,
        .col0 = f->col,
        .gap_step = SLAB_GAP_STEP,
        .gap_rows = slab->gap_rows,
        .gap_cols = slab->gap_cols,
        .height_size = s->side,
        .seed = f->seed,
        .nz = FRAGMENT_Z,
        .ny = FRAGMENT_Y,
        .nx = FRAGMENT_X,
    };
    for (int k = 0; k < 3; k++) {
        params.centre[k] = s->centre.v[k] * f->voxel;
        params.eu[k] = s->frame[0].v[k];
        params.ev[k] = s->frame[1].v[k];
        params.ew[k] = s->frame[2].v[k];
    }
    const void *buffers[] = {out->ct, noise, slab->ct, in, ex, height};
    i64 sizes[] = {
        pixels, pixels * 4, (i64)slab->rows * slab->cols * slab->depth, gaps * 4, gaps * 4, n * 4};
    need(!hx_inj_buffers(library->gpu, buffers, sizes, 6), "cannot prepare fragment Metal buffers");
    u8 *raw = push(a, pixels * 2);
    i32 dummy = 0;
    double seconds;
    int code = hx_inj_render(library->gpu, &params, sizeof(params), &dummy, sizeof(dummy),
        FRAGMENT_Z, FRAGMENT_Y, FRAGMENT_X, raw, 2, &seconds);
    need(!code, "fragment render failed (%d)", code);
    i64 core = 0, changed = 0;
    for (size_t i = 0; i < pixels; i++) {
        u8 label = raw[i * 2 + 1];
        out->ct[i] = raw[i * 2];
        if (label) {
            out->index[i] = out->count + 1;
            changed++;
        }
        if (label == 2) {
            out->core[i] = 1;
            core++;
        }
    }
    placed->record =
        fragment_record(a, library, placed, origin_xyz, cap, height, core, changed, out->count + 1);
    a->used = mark;
}

static FragmentBlock fragment_inject_block(Arena *a, InjectionFragments *library, const u8 *block,
    SlabPoint origin_xyz, int count, Pcg64 *rng) {
    need(count > 0 && count <= 255, "invalid fragment count");
    size_t pixels = (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X;
    FragmentBlock out = {
        .ct = push(a, pixels),
        .index = pushz(a, pixels),
        .core = pushz(a, pixels),
        .placed = pushz(a, (size_t)count * sizeof(*out.placed)),
    };
    memcpy(out.ct, block, pixels);
    u8 *mask = push(a, pixels);
    i32 *distance = push(a, pixels * sizeof(*distance));
    fragment_air(a, block, mask, distance);
    enum { FRAGMENT_MARGIN = 78 };
    size_t max = (size_t)(FRAGMENT_Z - 2 * FRAGMENT_MARGIN) * (FRAGMENT_Y - 2 * FRAGMENT_MARGIN) *
        (FRAGMENT_X - 2 * FRAGMENT_MARGIN);
    int *candidates = push(a, max * sizeof(*candidates)), n = 0;
    for (int z = FRAGMENT_MARGIN; z < FRAGMENT_Z - FRAGMENT_MARGIN; z++) {
        for (int y = FRAGMENT_MARGIN; y < FRAGMENT_Y - FRAGMENT_MARGIN; y++) {
            for (int x = FRAGMENT_MARGIN; x < FRAGMENT_X - FRAGMENT_MARGIN; x++) {
                size_t at = fragment_index(z, y, x);
                if (distance[at] >= (FRAGMENT_NEED + 1) * (FRAGMENT_NEED + 1)) {
                    candidates[n++] = (int)at;
                }
            }
        }
    }
    const float *noise = fragment_noise(a, library->noise, rng);
    for (int i = 0; i < count; i++) {
        size_t before = a->used;
        FragmentPlacement next = {.frag = fragment_draw(a, library, rng)};
        if (fragment_place(a, library, block, mask, distance, candidates, n, origin_xyz, &next,
                out.placed, out.count, rng)) {
            fragment_render(a, library, &out, &next, noise, origin_xyz);
            out.placed[out.count++] = next;
        } else {
            a->used = before;
        }
    }
    return out;
}

static void fragment_dispose(FragmentBlock *block) {
    for (int i = 0; i < block->count; i++) {
        json_object_put(block->placed[i].record);
    }
    block->count = 0;
}

static u8 fragment_grey(float value) { return (u8)fmaxf(0, fminf(255, (value - 20) / 180 * 255)); }

static json_object *fragment_demo(Arena *a, const char *path, u64 seed) {
    const char *scroll = "PHerc0139", *package = "PHerc0139-train";
    InjectionFragments library = fragment_open(a, scroll, package);
    Pcg64 rng = pcg64(seed);
    size_t mark = a->used;
    int tries = 40 * 20;
    FragmentBlock result = {0};
    u8 *original = NULL;
    for (int i = 0; i < 20; i++) {
        SlabPoint origin;
        original = fragment_real_block(a, &library, &rng, &tries, &origin);
        if (!original) {
            break;
        }
        result = fragment_inject_block(a, &library, original, origin, 3, &rng);
        if (result.count) {
            break;
        }
        a->used = mark;
    }
    need(result.count, "no fragment site found");
    const FragmentPlacement *p = result.placed;
    int z = (int)nearbyint(p->site.centre.v[0]), side = (int)(2 * p->frag.reach + 20);
    enum {
        GAP = 6,
        CUT_DEPTH = 49,
        REPEAT = 3,
        TOP_WIDTH = FRAGMENT_X * 2 + GAP,
        HEIGHT = FRAGMENT_Y + GAP + 2 * CUT_DEPTH * REPEAT + GAP
    };
    int width = TOP_WIDTH > side ? TOP_WIDTH : side;
    u8 *image = pushz(a, (size_t)HEIGHT * width);
    size_t plane = fragment_index(z, 0, 0);
    for (int y = 0; y < FRAGMENT_Y; y++) {
        u8 *row = image + (size_t)y * width;
        for (int x = 0; x < FRAGMENT_X; x++) {
            size_t i = plane + (size_t)y * FRAGMENT_X + x;
            row[x] = fragment_grey(original[i]);
            row[FRAGMENT_X + GAP + x] = fragment_grey(result.ct[i]);
        }
        memset(row + FRAGMENT_X, 255, GAP);
    }
    memset(image + (size_t)FRAGMENT_Y * width, 255, (size_t)GAP * width);
    for (int mode = 0; mode < 2; mode++) {
        const u8 *raw = mode ? result.ct : original;
        int y0 = FRAGMENT_Y + GAP + mode * (CUT_DEPTH * REPEAT + GAP);
        for (int depth = 0; depth < CUT_DEPTH; depth++) {
            for (int x = 0; x < side; x++) {
                SlabPoint sample =
                    fragment_point(p->site.centre, p->site.frame, x - side / 2.0, 0, depth - 24);
                u8 value = fragment_grey(fragment_ct_linear(raw, sample));
                for (int r = 0; r < REPEAT; r++) {
                    image[(size_t)(y0 + depth * REPEAT + r) * width + x] = value;
                }
            }
        }
    }
    for (int i = 0; i < GAP; i++) {
        size_t row = FRAGMENT_Y + GAP + CUT_DEPTH * REPEAT + i;
        memset(image + row * width, 255, side);
    }
    png_write(path, image, width, HEIGHT);
    i64 n = 0, sum = 0;
    for (size_t i = 0; i < (size_t)FRAGMENT_Z * FRAGMENT_Y * FRAGMENT_X; i++) {
        if (result.core[i]) {
            sum += result.ct[i];
            n++;
        }
    }
    json_object *summary = json_object_new_object(), *placed = json_object_new_array();
    for (int i = 0; i < result.count; i++) {
        json_object_array_add(placed, json_object_get(result.placed[i].record));
    }
    json_object_object_add(summary, "fragments", placed);
    json_object_object_add(
        summary, "core_mean", injection_number(n ? injection_round((double)sum / n, 10) : NAN));
    fragment_dispose(&result);
    fragment_close(&library);
    return summary;
}

static int injection_fragment_command(int argc, char **argv) {
    const char *usage = "synthetic.injection.fragment demo OUT.png [--seed N]";
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts(usage);
        return 0;
    }
    need(argc >= 2 && !strcmp(argv[0], "demo"), "%s", usage);
    u64 seed = 0;
    for (int i = 2; i < argc; i++) {
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        need(!strcmp(key, "--seed"), "unknown fragment option: %s", key);
        if (!value) {
            need(i + 1 < argc, "%s needs a value", key);
            value = argv[++i];
        }
        seed = slices_unsigned(value, key);
    }
    Arena a = arena(3 * GB);
    json_object *out = fragment_demo(&a, argv[1], seed);
    puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PRETTY));
    json_object_put(out);
    munmap(a.base, a.size);
    return 0;
}
