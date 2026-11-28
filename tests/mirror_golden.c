#include "../scans/catalog/scan_choice.c"

static void mirror_require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "mirror golden failed: %s\n", message);
        exit(1);
    }
}

static void mirror_test_plane(const char *path) {
    TIFF *file = TIFFOpen(path, "w");
    mirror_require(file && TIFFSetField(file, TIFFTAG_IMAGEWIDTH, 128) &&
            TIFFSetField(file, TIFFTAG_IMAGELENGTH, 96) &&
            TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 32) &&
            TIFFSetField(file, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP) &&
            TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 1) &&
            TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
            TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK),
        "published TIFF tags");
    float row[128];
    for (uint32_t r = 0; r < 96; r++) {
        for (uint32_t c = 0; c < 128; c++) {
            row[c] = c < 16 || (r >= 24 && r <= 32) ? -1.f : 8.f + .25f * c + 6.f * r;
        }
        mirror_require(TIFFWriteScanline(file, row, r, 0) >= 0, "published TIFF pixels");
    }
    TIFFClose(file);
}

int main(void) {
    char scratch[] = "/tmp/hx-mirror-golden-XXXXXX";
    const char *root = mkdtemp(scratch);
    mirror_require(root != NULL, "temporary folder");
    char raw[128], packed[128], object[128], text[128], invalid[128];
    snprintf(raw, sizeof(raw), "%s/x-original.tif", root);
    snprintf(packed, sizeof(packed), "%s/x.tif", root);
    snprintf(object, sizeof(object), "%s/model.obj.zst", root);
    snprintf(text, sizeof(text), "%s/model.obj", root);
    snprintf(invalid, sizeof(invalid), "%s/invalid.tif", root);
    mirror_test_plane(raw);
    mirror_test_plane(packed);
    struct stat before, after;
    mirror_require(!stat(packed, &before), "size of original TIFF");
    mirror_pack_tiff(packed, before.st_size);
    mirror_require(!stat(packed, &after) && after.st_size < before.st_size / 2 &&
            mirror_tiff_equal(packed, raw),
        "smaller TIFF with exactly the published floats");

    FILE *value = fopen(invalid, "wb");
    mirror_require(
        value && fputs("not a TIFF", value) >= 0 && !fclose(value), "malformed TIFF fixture");
    mirror_pack_tiff(invalid, 10);
    value = fopen(invalid, "rb");
    char bytes[12] = {0};
    mirror_require(value && fread(bytes, 1, sizeof(bytes), value) == 10 &&
            !memcmp(bytes, "not a TIFF", 10) && !fclose(value),
        "retain unreadable TIFF byte for byte");

    value = fopen(text, "wb");
    MirrorPayload obj = {
        .file = fopen(object, "wb"), .zstd = ZSTD_createCCtx(), .capacity = ZSTD_CStreamOutSize()};
    obj.block = malloc(obj.capacity);
    const char line[] = "v 4894.498047 2601.994628 12.44665\nvt 0.5 0.25\nf 1/1 2/2 3/3\n";
    mirror_require(value && obj.file && obj.zstd && obj.block, "OBJ stream");
    for (int i = 0; i < 5000; i++) {
        mirror_require(
            fputs(line, value) >= 0 && mirror_zstd(&obj, line, sizeof(line) - 1, ZSTD_e_continue),
            "write OBJ text to both streams");
    }
    mirror_require(mirror_zstd(&obj, NULL, 0, ZSTD_e_end) && !fclose(obj.file) && !fclose(value) &&
            mirror_zstd_equal(object, text) && !stat(object, &after) && !stat(text, &before) &&
            after.st_size < before.st_size / 4,
        "smaller OBJ with exactly the published text");
    ZSTD_freeCCtx(obj.zstd);
    free(obj.block);

    Mirror mirror = {.root = scratch};
    char *name = mirror_held(&mirror, "model.obj");
    mirror_require(name && !strcmp(name, object) && !mirror_held(&mirror, "../elsewhere") &&
            mirror_rank("/full-scrolls/Scroll1/paths/mesh.obj") == 1 &&
            mirror_rank("/full-scrolls/Scroll1/paths/mesh.tif") == 3 &&
            mirror_rank("/full-scrolls/Scroll1/paths/mesh_mask.png") == 0 &&
            mirror_rank("/datasets/fiber-skeletons/truth.tif") == 2 &&
            mirror_rank("/datasets/spiral_datasets/PHercParis4/outer_shell/mesh") == 4 &&
            mirror_rank("/full-scrolls/Scroll1/paths/mesh_points.obj") < 0 &&
            mirror_rank("/full-scrolls/Scroll5/working/mesh.obj") < 0 &&
            mirror_rank("/community-uploads/user/ink.zip") < 0,
        "mirror names and priority for files that cannot be rebuilt");
    free(name);
    mirror_require(!unlink(raw) && !unlink(packed) && !unlink(object) && !unlink(text) &&
            !unlink(invalid) && !rmdir(root),
        "remove temporary mirror fixture");
    puts("mirror golden passed");
    return 0;
}
