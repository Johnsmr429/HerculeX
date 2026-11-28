#include "base.h"
#include "synthetic/three_d/fields.c"
#include "synthetic/two_d/layout.c"
#include "synthetic/two_d/sampler.c"
#include "synthetic/two_d/model.c"

static void fixture_network(void *context, int n, const float *input, const float *condition,
    const i64 *style, float log_sigma, float *output) {
    for (int i = 0; i < n; i++) {
        const float *d = condition + (size_t)i * 3 * SLICES_PIXELS;
        for (int p = 0; p < SLICES_PIXELS; p++) {
            int at = i * SLICES_PIXELS + p;
            output[at] = .1f * input[at] + .2f * d[SLICES_PIXELS + p] +
                .08f * d[SLICES_PIXELS * 2 + p] + .02f * style[i];
        }
    }
}

static void checkpoint_member(zip_t *zip, const char *name, const void *data, size_t size) {
    zip_source_t *source = zip_source_buffer(zip, data, size, 0);
    need(source && zip_file_add(zip, name, source, 0) >= 0,
        "cannot write checkpoint fixture member %s", name);
}

static void checkpoint_fixture(void) {
    static const char plain[] = "\x80\x02}X\x0a\0\0\0"
                                "f.0.weightctorch._utils\n_rebuild_tensor_v2\n"
                                "((X\x07\0\0\0storagectorch\nFloatStorage\nX\x01\0\0\0"
                                "0X\x03\0\0\0"
                                "cpuK\x06tQK\x01K\x02K\x02\x86K\x01K\x03\x86\x89"
                                "ccollections\nOrderedDict\n)RtRs.";
    static const char wrapped[] =
        "\x80\x05\x95\x94\0\0\0\0\0\0\0}\x8c\x05model}\x8c\x0a"
        "f.0.weight\x8c\x0ctorch._utils\x8c\x12_rebuild_tensor_v2\x93((\x8c\x07storage"
        "\x8c\x05torch\x8c\x0c"
        "FloatStorage\x93\x8c\x01"
        "0\x8c\x03"
        "cpuK\x06tQK\x01"
        "K\x02K\x02\x86K\x01K\x03\x86\x89\x8c\x0b"
        "collections\x8c\x0bOrderedDict\x93)RtRss.";
    char folder[] = "/tmp/hx-slices-checkpoint-XXXXXX";
    need(mkdtemp(folder) != NULL, "cannot make checkpoint fixture directory");
    Arena mem = arena(4096);
    const float storage[] = {1, 2, 3, 4, 5, 6}, expected[] = {2, 5, 3, 6};
    const char *paths[] = {fmt(&mem, "%s/plain.pt", folder), fmt(&mem, "%s/wrapped.pth", folder)};
    const char *pickles[] = {plain, wrapped};
    const size_t sizes[] = {sizeof(plain) - 1, sizeof(wrapped) - 1};
    for (int i = 0; i < 2; i++) {
        int error;
        zip_t *zip = zip_open(paths[i], ZIP_CREATE | ZIP_EXCL, &error);
        need(zip != NULL, "cannot create checkpoint fixture");
        checkpoint_member(zip, "weights/data.pkl", pickles[i], sizes[i]);
        checkpoint_member(zip, "weights/byteorder", "little", 6);
        checkpoint_member(zip, "weights/data/0", storage, sizeof(storage));
        need(!zip_close(zip), "cannot finish checkpoint fixture");
        SliceModel model = slices_model_open(paths[i]);
        const float *actual = slices_parameter(&model, "f.0", "weight", (int[]){2, 2}, 2);
        need(!memcmp(actual, expected, sizeof(expected)), "incorrect checkpoint strides");
        json_object_put(model.header);
        munmap(model.memory.base, model.memory.size);
    }
    remove_tree(folder);
    munmap(mem.base, mem.size);
}

int main(void) {
    checkpoint_fixture();
    const int size = 384, n = 2;
    const size_t count = (size_t)size * size * n;
    Arena mem = arena(8 << 20);
    u8 *lines = push(&mem, count);
    u8 *dist = push(&mem, count);
    u8 *result = push(&mem, count);
    for (size_t i = 0; i < count; i++) {
        lines[i] = i % 11 == 0;
        dist[i] = i % 64;
    }
    i64 style[] = {3, 5};
    SliceNoise rng = slices_noise_seed(42);
    slices_sample(&rng, fixture_network, NULL, size, size, n, 3, 32, 0, lines, dist, style, result);
    char hash[65];
    sha256_hex(result, count, hash);
    need(!strcmp(hash, "47d62ee9ae57b335868ae72710abd4894df2118b2f011e7e427c2ea5cb4f688c"),
        "synthetic overlapping-window sample differs from the Python fixture: %s", hash);
    munmap(mem.base, mem.size);
    return 0;
}
