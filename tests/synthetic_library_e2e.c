#include "base.h"
#include <sqlite3.h>

static void sql(const char *path, const char *script) {
    sqlite3 *db = NULL;
    need(sqlite3_open(path, &db) == SQLITE_OK, "cannot create %s", path);
    need(sqlite3_exec(db, script, NULL, NULL, NULL) == SQLITE_OK, "%s: %s", path,
        sqlite3_errmsg(db));
    sqlite3_close(db);
}

static const char *fixture(Arena *a, const char *scratch, int air) {
    const char *scroll = "PHerc0139", *package = "PHerc0139-fixture";
    const char *scan = "20250728140407-9.362um-1.2m-113keV-masked";
    const char *out = fmt(a, "%s/%s", scratch, air ? "air" : "slab");
    const char *control = fmt(a, "%s/bench/controls/%s", out, package);
    const char *cache = fmt(a, "%s/cache/chunks", out);
    const char *pack = fmt(a, "%s/%s/%s/0", cache, scroll, scan);
    mkdirs(pack);
    int nz = air ? 32 : 580, ny = air ? 320 : 580, nx = air ? 320 : 80;
    const char *dims = fmt(a, "[%d,%d,%d]", nz, ny, nx);
    sql(fmt(a, "%s/arrays.sqlite", cache),
        fmt(a,
            "CREATE TABLE arrays(sample,name,shape,chunks,fill_value,dtype,root,url,level);"
            "CREATE TABLE resolved(sample,scan,kind,url);"
            "INSERT INTO arrays VALUES('%s','%s','%s','%s',0,'|u1','local','local/0',0);",
            scroll, scan, dims, dims));
    sql(fmt(a, "%s/index.sqlite", pack),
        "CREATE TABLE chunks(z,y,x,offset); INSERT INTO chunks VALUES (0,0,0,0);");
    FILE *file = fopen(fmt(a, "%s/chunks.pack", pack), "wb");
    need(file != NULL, "cannot create cached CT fixture");
    size_t mark = a->used, plane = (size_t)nx * ny;
    u8 *pixels = push(a, plane);
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            double centre = 28 + .018 * (y - 42) + .015 * (z - 42);
            double crest = centre + sin(y / 27.0) + cos(z / 17.0);
            for (int x = 0; x < nx; x++) {
                if (air) {
                    pixels[y * nx + x] = y >= 25 && y < 35 && x >= 40 && x < 60
                        ? 140
                        : 40 + (2 * z + 3 * y + x) % 27;
                } else {
                    double near = (x - crest) / 2, far = (x - centre - 10) / 2.4;
                    double value =
                        55 + 130 * exp(-near * near) + 50 * exp(-far * far) + (z + 2 * y + x) % 7;
                    pixels[y * nx + x] = (u8)fmin(255, value);
                }
            }
        }
        need(fwrite(pixels, 1, plane, file) == plane, "cannot write cached CT fixture");
    }
    need(fclose(file) == 0, "cannot finish cached CT fixture");
    a->used = mark;
    f32 build_points[] = {40, 260, 260};
    f32 air_points[] = {160, 160, 16, 160, 160, 16};
    double voxel = 9.362;
    u8 zones[] = {0, 0};
    const char *names[] = {"points", "zone", "voxel_um"};
    Array arrays[] = {
        {.descr = "<f4",
            .ndim = 2,
            .shape = {air ? 2 : 1, 3},
            .data = air ? air_points : build_points,
            .size = air ? sizeof(air_points) : sizeof(build_points)},
        {.descr = "|u1", .ndim = 1, .shape = {air ? 2 : 1}, .data = zones, .size = air ? 2 : 1},
        {.descr = "<f8", .data = &voxel, .size = sizeof(voxel)},
    };
    npz_write(fmt(a, "%s/truth.npz", control), 3, names, arrays);
    if (air) {
        return out;
    }
    const char *mesh = fmt(a, "%s/meshes/plane.tifxyz", control);
    const char axis[] = "{\"knots_z_x_y\":[[0,0,300],[600,0,300]]}";
    const char scale[] = "{\"scale\":[0.03125,0.03125]}";
    write_file(fmt(a, "%s/axis.json", control), axis, sizeof(axis) - 1);
    write_file(fmt(a, "%s/meta.json", mesh), scale, sizeof(scale) - 1);
    f32 grids[4][23 * 23];
    for (int r = 0; r < 23; r++) {
        for (int c = 0; c < 23; c++) {
            int i = r * 23 + c;
            grids[1][i] = 10 + c * 32;
            grids[2][i] = 10 + r * 32;
            grids[0][i] = 28.0f + .018f * (grids[1][i] - 42.0f) + .015f * (grids[2][i] - 42.0f);
            grids[3][i] = 3.5f;
        }
    }
    for (int i = 0; i < 4; i++) {
        tiff_write(fmt(a, "%s/%c.tif", mesh, "xyzw"[i]), grids[i], 23, 23);
    }
    return out;
}

static void launch(const char *exe, const char *outputs, char *const *args) {
    pid_t pid = fork();
    if (!pid) {
        setenv("HERCULEX_OUTPUTS", outputs, 1);
        unsetenv("HERCULEX_CHUNKS");
        execv(exe, args);
        _exit(127);
    }
    int status;
    need(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native slab library step failed (%s)", args[2]);
}

static void assert_digest(const void *data, size_t size, const char *expected, const char *name) {
    char hash[65];
    sha256_hex(data, size, hash);
    need(!strcmp(hash, expected), "native slab %s differs from the Python fixture: %s", name, hash);
}

int main(int argc, char **argv) {
    need(argc == 2, "native executable is required");
    char scratch[] = "/tmp/hx-library-fixture-XXXXXX";
    need(mkdtemp(scratch) != NULL, "cannot make the slab fixture directory");
    Arena a = arena(64 << 20);
    const char *build = fixture(&a, scratch, 0), *air = fixture(&a, scratch, 1);
    char *archive = fmt(&a, "%s/synthetic/library/PHerc0139/slab-042-000.npz", build);
    char *image = fmt(&a, "%s/slab.png", scratch);
    launch(argv[1], build,
        (char *[]){argv[1], "synthetic.library.native", "build", "PHerc0139-fixture", "--count",
            "1", "--seed", "42", NULL});
    launch(argv[1], air,
        (char *[]){argv[1], "synthetic.library.native", "air", "PHerc0139-fixture", "--count", "2",
            "--seed", "42", NULL});
    launch(argv[1], build,
        (char *[]){argv[1], "synthetic.library.native", "show", archive, image, NULL});
    const char *names[] = {"ct", "half_in", "half_out", "points", "normals", "ridge"};
    const char *hashes[] = {
        "6f149f4b95e40c204a02195730da49022091dda5eedab889d2b0b5a530655f66",
        "24b60b38965ce4b05e2eeb448e0c0d46439717e740cdae08f2b01c4235b9c643",
        "a1e690e6de53679b50e04360a1b0de30bdf8e185bb7aa67d8a2c7969f30e5552",
        "381ad0c3e037e12363d994ee6661e801062dbd9bb386672d8d75913292bc80a7",
        "c3f81f8766ed623b0fd42ad5e7210180466d0eeb194633d3a87536a3a47fbb89",
        "fdfaa4b08070dfd1272ce4e472007695a0fab6c4b5ec0ce95b584ea61d0e0ccd",
    };
    for (int i = 0; i < 6; i++) {
        Array member = npz_get(&a, archive, names[i]);
        assert_digest(member.data, member.size, hashes[i], names[i]);
    }
    const char *air_file = fmt(&a, "%s/synthetic/library/PHerc0139/air.npz", air);
    Array cubes = npz_get(&a, air_file, "blocks"), level = npz_get(&a, air_file, "level");
    assert_digest(cubes.data, cubes.size,
        "71fcf828bbb08045f10f474fbdb562d3fa3cf0a9c8c0248f97c70ea23fbcb3f3", "air blocks");
    assert_digest(level.data, level.size,
        "75620c4d6de605cd62ad109ba867457bee568fabedbb6f9146d9397b2dbb8c29", "air mean");
    int w, h;
    u8 *preview = png_read(&a, image, &w, &h);
    need(w == 512 && h == 670, "slab preview dimensions changed");
    assert_digest(preview, (size_t)w * h,
        "54176c442a0abd4e759ac5808594a9cc9396133645c6122b56110bd10ec54b08", "preview");
    remove_tree(scratch);
    munmap(a.base, a.size);
    return 0;
}
