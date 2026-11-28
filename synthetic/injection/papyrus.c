void *hx_inj_open(const char *source, const char *function, char *message, int length);
int hx_inj_buffers(void *handle, const void *const *data, const i64 *lengths, int count);
int hx_inj_render(void *handle, const void *params, i64 params_length, const void *inject,
    i64 inject_length, int nz, int ny, int nx, u8 *out, int channels, double *seconds);
void hx_inj_close(void *handle);

typedef struct {
    float origin[4], centre[4], eu[4], ev[4], ew[4];
    float voxel, bend_u, bend_v, half_u, half_v, edge;
    float t_recto, t_verso, dip, dip_um, parenchyma, lump, lump_um;
    float fibre_recto, spacing_recto, radius_recto, fibre_verso, spacing_verso, radius_verso;
    float jitter, wave_um, wave_scale_um, fray_um, sigma, contrast;
    u32 seed;
    i32 nz, ny, nx, pad[3];
} InjectionPapyrus;

_Static_assert(sizeof(InjectionPapyrus) == 208, "papyrus kernel ABI");

static void *injection_kernel(Arena *a, const char *module, const char *entry) {
    size_t mark = a->used;
    Bytes source = read_file(a, fmt(a, "synthetic/injection/render_apple/%s.metal", module));
    char message[4096];
    void *gpu = hx_inj_open((const char *)source.data, entry, message, sizeof(message));
    need(gpu != NULL, "%s kernel: %s", module, message);
    a->used = mark;
    return gpu;
}

static u8 injection_preview_grey(double value) {
    return (u8)fmax(0, fmin(255, (value - 20) / 180 * 255));
}

static json_object *injection_papyrus_preview(Arena *a, const char *path, u64 seed) {
    enum { N = 256, D = 25, G = 6, WIDTH = 2 * N + G, HEIGHT = N + G + G * D };
    const char *folder = injection_profile_folder(a, "PHerc0139");
    json_object *profile = json_read(fmt(a, "%s/profile.json", folder));
    double voxel = jf64(profile, "voxel_um"), level = jf64(jget(profile, "air"), "level");
    json_object_put(profile);
    InjectionNoise noise = injection_noise_open(a, fmt(a, "%s/profile.npz", folder));
    need(noise.shape.dim[0] == 32 && noise.shape.dim[1] == 32 && noise.shape.dim[2] == 32,
        "%s: papyrus preview needs 32-voxel air cubes", folder);
    float *signal = push(a, (size_t)N * D * N * sizeof(*signal));
    int half = N / 2, half_depth = D / 2;
    InjectionPapyrus params = {
        .origin = {-half * voxel, -half_depth * voxel, -half * voxel},
        .eu = {0, 0, 1},
        .ev = {1, 0, 0},
        .ew = {0, 1, 0},
        .voxel = (float)voxel,
        .half_u = 1e6f,
        .half_v = 1e6f,
        .edge = 8,
        .t_recto = 28,
        .t_verso = 28,
        .dip = .15f,
        .dip_um = 4,
        .parenchyma = .6f,
        .lump = .25f,
        .lump_um = 200,
        .fibre_recto = 1,
        .spacing_recto = 100,
        .radius_recto = 10,
        .fibre_verso = 1,
        .spacing_verso = 100,
        .radius_verso = 10,
        .jitter = .8f,
        .wave_um = 10,
        .wave_scale_um = 300,
        .fray_um = 50,
        .sigma = 11,
        .contrast = 63.5f,
        .seed = (u32)seed,
        .nz = N,
        .ny = D,
        .nx = N,
    };
    void *gpu = injection_kernel(a, "papyrus", "papyrus");
    need(!hx_inj_buffers(gpu, NULL, NULL, 0), "cannot set papyrus kernel buffers");
    i32 dummy = 0;
    double seconds;
    int status = hx_inj_render(gpu, &params, sizeof(params), &dummy, sizeof(dummy), N, D, N,
        (u8 *)signal, sizeof(*signal), &seconds);
    hx_inj_close(gpu);
    need(!status, "papyrus kernel failed (%d)", status);
    glob_t paths = {0};
    need(!glob(fmt(a, "%s/slab-*.npz", folder), 0, NULL, &paths) && paths.gl_pathc,
        "%s: no papyrus preview slabs", folder);
    qsort(paths.gl_pathv, paths.gl_pathc, sizeof(*paths.gl_pathv), slab_mesh_order);
    Array slab = npz_get(a, paths.gl_pathv[0], "ct");
    globfree(&paths);
    need(slab.ndim == 3 && !strcmp(slab.descr, "|u1") && slab.shape[0] >= N && slab.shape[1] >= N &&
            slab.shape[2] == D && slab.size == (size_t)elements(slab),
        "%s: invalid papyrus preview slab", folder);
    const u8 *real = slab.data;
    const int side = noise.shape.dim[0];
    Pcg64 rng = pcg64(seed);
    double *cube = push(a, noise.shape.n * sizeof(*cube));
    double predicted[D] = {0};
    float measured[D] = {0};
    u8 *image = push(a, (size_t)WIDTH * HEIGHT);
    memset(image, 255, (size_t)WIDTH * HEIGHT);
    for (int bz = 0; bz < N; bz += side) {
        for (int bx = 0; bx < N; bx += side) {
            injection_draw_noise(a, noise, level, 1, &rng, cube);
            for (int z = 0; z < side; z++) {
                for (int y = 0; y < D; y++) {
                    for (int x = 0; x < side; x++) {
                        size_t out = ((size_t)(bz + z) * D + y) * N + bx + x;
                        size_t in = ((size_t)z * side + y) * side + x;
                        double ct = fmax(0, fmin(255, cube[in] + signal[out]));
                        predicted[y] += ct;
                        u8 grey = injection_preview_grey(ct);
                        if (y == D / 2) {
                            image[(size_t)(bz + z) * WIDTH + bx + x] = grey;
                        }
                        if (bz + z == N / 2) {
                            for (int j = 0; j < G; j++) {
                                image[((size_t)N + G + G * y + j) * WIDTH + bx + x] = grey;
                            }
                        }
                    }
                }
            }
        }
    }
    size_t w = (size_t)slab.shape[1], pixels = slab.size / D;
    for (size_t i = 0; i < pixels; i++) {
        for (int y = 0; y < D; y++) {
            u8 value = real[D * i + y];
            measured[y] += value;
            u8 grey = (u8)fmaxf(0, fminf(255, ((float)value - 20) / 180 * 255));
            size_t row = i / w, col = i % w;
            if (col < N && row < N && y == D / 2) {
                image[row * WIDTH + N + G + col] = grey;
            }
            if (col < N && row == N / 2) {
                for (int j = 0; j < G; j++) {
                    image[((size_t)N + G + G * y + j) * WIDTH + N + G + col] = grey;
                }
            }
        }
    }
    png_write(path, image, WIDTH, HEIGHT);
    json_object *out = json_object_new_object(), *synthetic = json_object_new_array();
    json_object *reference = json_object_new_array();
    for (int y = 0; y < D; y++) {
        json_object_array_add(synthetic, jnum(injection_round(predicted[y] / (N * N), 10)));
        json_object_array_add(reference, jnum(injection_round(measured[y] / (float)pixels, 10)));
    }
    json_object_object_add(out, "seconds", jnum(injection_round(seconds, 10000)));
    json_object_object_add(out, "profile_synthetic", synthetic);
    json_object_object_add(out, "profile_real", reference);
    return out;
}

static int injection_papyrus_command(int argc, char **argv) {
    const char *usage = "synthetic.injection.papyrus show OUT.png [--seed N]";
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts(usage);
        return 0;
    }
    need(argc >= 2 && !strcmp(argv[0], "show"), "%s", usage);
    u64 seed = 0;
    for (int i = 2; i < argc; i++) {
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        need(!strcmp(key, "--seed"), "unknown papyrus option: %s", key);
        if (!value) {
            need(i + 1 < argc, "%s needs a value", key);
            value = argv[++i];
        }
        seed = slices_unsigned(value, key);
    }
    Arena a = arena(GB);
    json_object *out = injection_papyrus_preview(&a, argv[1], seed);
    puts(json_object_to_json_string(out));
    json_object_put(out);
    munmap(a.base, a.size);
    return 0;
}
