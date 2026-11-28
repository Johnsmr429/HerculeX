#include "../base.h"

#ifdef __APPLE__
#include <sqlite3.h>

static char gallery_fixture[] = "/tmp/hx-gallery.XXXXXX";

static void clear_gallery(void) { remove_tree(gallery_fixture); }

static void fixture_text(const char *path, const char *value) {
    write_file(path, value, strlen(value));
}

static void same_hash(Bytes b, const char *expected) {
    u8 sum[32];
    char hex[65];
    digest("sha256", b.data, b.size, sum);
    for (size_t j = 0; j < sizeof(sum); j++) {
        snprintf(hex + 2 * j, 3, "%02x", sum[j]);
    }
    need(!strcmp(hex, expected), "gallery image differs: expected %s, found %s", expected, hex);
}
#endif

int main(int argc, char **argv) {
#ifndef __APPLE__
    (void)argc;
    (void)argv;
    return 77;
#else
    need(argc == 2 && mkdtemp(gallery_fixture), "expected native binary and a temporary folder");
    atexit(clear_gallery);
    Arena a = arena(4 << 20);
    const char *library = fmt(&a, "%s/library", gallery_fixture);
    const char *outputs = fmt(&a, "%s/outputs", gallery_fixture);
    const char *texture = fmt(&a, "%s/synthetic/library/PHerc0139", outputs);
    setenv("HERCULEX_LIBRARY", library, 1);
    setenv("HERCULEX_OUTPUTS", outputs, 1);
    unsetenv("HERCULEX_SYNTHETIC_LOOKS");
    unsetenv("HERCULEX_CHUNKS");
    setenv("HERCULEX_SYNTHETIC_THREADS", "2", 1);
    fixture_text(fmt(&a, "%s/catalog/metadata.json", library),
        "{\"samples\":{\"PHerc0139\":{\"volumes\":{\"ct\":{\"properties\":{\"pixel_size_um\":9.362,"
        "\"energy_keV\":113},\"data\":[{\"type\":\"ome-zarr\",\"origins\":[{\"path\":\"ct.zarr\","
        "\"access_roots\":[{\"url\":\"http://127.0.0.1:1\"}]}]}]}}}}}");
    u8 slab[9 * 8 * 25], blocks[16];
    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 25; z++) {
                slab[(y * 8 + x) * 25 + z] = 60 + (y + 3 * x + 7 * z) % 48;
            }
        }
    }
    for (size_t i = 0; i < sizeof(blocks); i++) {
        blocks[i] = 52 + i;
    }
    Array ct = {.descr = "|u1", .ndim = 3, .shape = {9, 8, 25}, .data = slab, .size = sizeof(slab)};
    npz_write(fmt(&a, "%s/slab-001.npz", texture), 1, (const char *[]){"ct"}, &ct);
    double level = 60.5;
    Array air[] = {
        {.descr = "|u1", .ndim = 4, .shape = {2, 2, 2, 2}, .data = blocks, .size = sizeof(blocks)},
        {.descr = "<f8", .data = &level, .size = sizeof(level)}};
    npz_write(fmt(&a, "%s/air.npz", texture), 2, (const char *[]){"blocks", "level"}, air);
    fixture_text(fmt(&a, "%s/library-fixture.json", texture), "{\"voxel_um\":9.362}");
    float point[] = {512, 512, 8};
    Array points = {
        .descr = "<f4", .ndim = 2, .shape = {1, 3}, .data = point, .size = sizeof(point)};
    npz_write(fmt(&a, "%s/bench/controls/PHerc0139-train/truth.npz", outputs), 1,
        (const char *[]){"points"}, &points);

    const char *pack = fmt(&a, "%s/cache/chunks/PHerc0139/ct/0", outputs);
    fixture_text(fmt(&a, "%s/.zarray", pack),
        "{\"shape\":[16,1024,1024],\"chunks\":[1,1024,1024],\"dtype\":\"|u1\",\"order\":\"C\","
        "\"compressor\":null}");
    sqlite3 *db;
    sqlite3_stmt *stmt;
    const char *query = "CREATE TABLE chunks(z INTEGER,y INTEGER,x INTEGER,offset INTEGER);";
    need(sqlite3_open(fmt(&a, "%s/index.sqlite", pack), &db) == SQLITE_OK &&
            sqlite3_exec(db, query, NULL, NULL, NULL) == SQLITE_OK &&
            sqlite3_prepare_v2(db, "INSERT INTO chunks VALUES(?1,0,0,?2)", -1, &stmt, NULL) ==
                SQLITE_OK,
        "cannot create the fixture CT index");
    FILE *volume = fopen(fmt(&a, "%s/chunks.pack", pack), "wb");
    need(volume != NULL, "cannot create the fixture CT pack");
    const size_t area = 1024 * 1024, mark = a.used;
    u8 *slice = push(&a, area);
    for (int z = 0; z < 16; z++) {
        sqlite3_bind_int(stmt, 1, z);
        sqlite3_bind_int64(stmt, 2, area * z);
        need(sqlite3_step(stmt) == SQLITE_DONE && sqlite3_reset(stmt) == SQLITE_OK,
            "cannot index the fixture CT slice");
        for (int y = 0; y < 1024; y++) {
            for (int x = 0; x < 1024; x++) {
                slice[(size_t)y * 1024 + x] = 60 + (z + 2 * y + 3 * x) % 80;
            }
        }
        need(fwrite(slice, 1, area, volume) == area, "cannot write the fixture CT slice");
    }
    fclose(volume);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    a.used = mark;
    pid_t child = fork();
    need(child >= 0, "cannot launch the gallery");
    if (!child) {
        execl(argv[1], argv[1], "synthetic.three_d.gallery", "--cases", "round/pressed:3",
            (char *)NULL);
        _exit(127);
    }
    int status;
    need(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
        "the gallery did not publish");
    const char *sheets = fmt(&a, "%s/website/sheets", outputs);
    const char *gallery = fmt(&a, "%s/synthetic", sheets);
    int w, h;
    const u8 *image = png_read(&a, fmt(&a, "%s/slab.png", gallery), &w, &h);
    need(w == 8 && h == 9 && image[0] == 107 && image[7] == 69 && image[71] == 80,
        "the gallery slab differs from the golden image");
    Bytes page = read_file(&a, fmt(&a, "%s/index.html", gallery));
    Bytes start = read_file(&a, fmt(&a, "%s/index.html", sheets));
    need(page.data && strstr((char *)page.data, "\"slabs\":1") &&
            strstr((char *)page.data, "\"turns_in_slice\":63") &&
            strstr((char *)page.data, "\"on_sheet\":0.585") &&
            strstr((char *)page.data, "\"kollesis\":true") && start.data &&
            strstr((char *)start.data, "<a href=\"synthetic/index.html\">Synthetic</a>"),
        "the gallery page or its link differs");
    const size_t saved = a.used;
    same_hash(read_file(&a, fmt(&a, "%s/case-0/cube.u8", gallery)),
        "9d998c8ebcfebc117876ffbb3374ac7024af5c509ac0c36fff0e809c6d3826a8");
    a.used = saved;
    same_hash(read_file(&a, fmt(&a, "%s/case-0/cube_label.u8", gallery)),
        "b95dcf9244c676f16af76dc2c4a49581c078ad54dbfc59ed3ca27cd357be0781");
    a.used = saved;
    w = h = 0;
    u8 *real = png_read(&a, fmt(&a, "%s/case-0/real.png", gallery), &w, &h);
    need(w == 1024 && h == 1024, "wrong size of the real slice");
    same_hash((Bytes){.data = real, .size = area},
        "ca7c26792b26ffda7abd86af9112f40a88fb58aa353e06ba95b258afdd586d35");
    a.used = saved;
    const char *archive = fmt(&a, "%s/crop.npz", gallery_fixture);
    child = fork();
    need(child >= 0, "cannot launch injected crop rendering");
    if (!child) {
        execl(argv[1], argv[1], "synthetic.three_d.inject.render", "round/pressed", "3", archive,
            "--size", "32", "--slices", "3", "--at", ".31", (char *)NULL);
        _exit(127);
    }
    need(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
        "the injected crop did not render");
    Array ink = npz_get(&a, archive, "ct"), label = npz_get(&a, archive, "label");
    need(ink.ndim == 3 && ink.shape[0] == 3 && ink.shape[1] == 32 && ink.shape[2] == 32,
        "the injected crop has the wrong shape");
    same_hash((Bytes){.data = ink.data, .size = ink.size},
        "46598e0fdd21cbc587e80f8cb514a9f45558da5f7f9698f1851236acc007dec0");
    same_hash((Bytes){.data = label.data, .size = label.size},
        "106cb8f77eaadeb372e0aa8d79bcd657dffc4de6ab82ba9673c99e97058de846");
    const double *origin = npz_get(&a, archive, "origin_um").data;
    need(origin[0] == 26363.392 && fabs(origin[1] - 4051.898854531554) < 1e-9 &&
            fabs(origin[2] + 132.60231509578) < 1e-9,
        "the injected crop has the wrong origin");
    int error;
    zip_t *zip = zip_open(archive, ZIP_RDONLY, &error);
    need(zip != NULL && zip_get_num_entries(zip, 0) == 5, "invalid injected crop archive");
    for (int i = 0; i < 5; i++) {
        zip_stat_t item;
        need(!zip_stat_index(zip, i, 0, &item) && item.comp_method == ZIP_CM_DEFLATE,
            "injected crop archive lost ZIP compression");
    }
    zip_close(zip);
    munmap(a.base, a.size);
    puts("synthetic gallery e2e passed");
    return 0;
#endif
}
