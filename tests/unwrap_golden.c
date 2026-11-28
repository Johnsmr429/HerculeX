#define _GNU_SOURCE
#include "../storage/node_store.c"
#include "../calibrate/unwrap/mesh_surfaces.c"
#define Index WrapIndex
#include "../calibrate/unwrap/sheet_wrap_score.c"
#undef Index
#include "../calibrate/unwrap/raw_sheet_ruler.c"
#include <pthread.h>

static void require(int pass, const char *what) {
    if (!pass) {
        fprintf(stderr, "node-store golden failed: %s\n", what);
        exit(1);
    }
}

static void *pack_thread(void *path) { return (void *)(intptr_t)hx_node_pack(path); }

static void reference_golden(void) {
    double x[] = {0, 1, 0, 1}, y[] = {0, 0, 1, 1}, z[] = {1, 1, 1, 1};
    uint8_t valid[] = {1, 1, 1, 1}, keep[4] = {0};
    float normal[12] = {0};
    hx_reference_normals(x, y, z, valid, 2, 2, normal, keep);
    for (int i = 0; i < 4; i++) {
        require(keep[i] && normal[3 * i + 2] == 1, "reference plane normals");
    }
    double theta[] = {3., -3., 3.1, -2.9}, phi[4];
    int64_t parts[4];
    require(hx_reference_unwrap(theta, valid, 2, 2, phi, parts) == 1 && phi[0] == theta[0] &&
            phi[1] > 3.2 && phi[1] < 3.3 && parts[0] == 0 && parts[3] == 0,
        "unwrap connected reference rows");

    int32_t candidates[] = {0, 1, 2, 3}, membership[] = {0, 0, 0, 0};
    double angle[] = {0, 0, 1, 0}, reference[] = {0, 0, 0, 0};
    int64_t groups[] = {0, 0, 0, 0}, chosen[4];
    uint8_t selected[] = {1}, unknown[] = {0, 0, 0, 1}, labels[4];
    int status = hx_wrap_classify(4, 1, candidates, 4, 4, membership, 4, angle, reference, groups,
        selected, 1, unknown, 0, labels, chosen);
    require(!status && labels[0] == 0 && labels[1] == 0 && labels[2] == 1 && labels[3] == 3,
        "reference winding classification");
}

static void ruler_golden(void) {
    enum { SHEETS = 43, PITCH = 10, LENGTH = SHEETS * PITCH + 5 };
    double profile[LENGTH], crossings[2 * SHEETS], row[6];
    for (int i = 0; i < LENGTH; i++) {
        profile[i] = 40;
    }
    for (int i = 0; i < SHEETS; i++) {
        int position = 5 + i * PITCH;
        for (int j = -4; j <= 4; j++) {
            profile[position + j] += 180 * exp(-j * j / 4.5);
        }
        crossings[2 * i] = position;
    }
    for (int run = 0; run < 5; run++) {
        for (int i = 0; i < SHEETS; i++) {
            int winding[] = {i, abs(i - SHEETS / 2), i / 2, i + i % 2, i % 2};
            crossings[2 * i + 1] = winding[run];
        }
        require(hx_ruler_line(profile, LENGTH, crossings, SHEETS, row) == 1 &&
                row[0] == SHEETS - 1 && row[4] == 1,
            "ruler finds every known sheet");
        if (run == 4) {
            require(row[1] < .1 && row[2] > .9, "ruler finds winding noise");
        } else {
            require(row[1] == (run == 2 ? .5 : 1.) && row[2] == 0 && row[3] == (run == 1),
                "ruler separates folds and missing levels");
        }
    }
}

int main(void) {
    char scratch[] = "/tmp/hx-node-golden-XXXXXX";
    const char *folder = mkdtemp(scratch);
    require(folder != NULL, "temporary folder");
    char nodes[128], cells[128], packed_nodes[128], packed_cells[128];
    snprintf(nodes, sizeof(nodes), "%s/nodes.bin", folder);
    snprintf(cells, sizeof(cells), "%s/cells.bin", folder);
    snprintf(packed_nodes, sizeof(packed_nodes), "%s/nodes.zst", folder);
    snprintf(packed_cells, sizeof(packed_cells), "%s/cells.zst", folder);

    const size_t count = 9000, first = 127, last = 400;
    size_t node_bytes = count * HX_NODE_BYTES, cell_bytes = count * sizeof(int32_t);
    uint8_t *raw = malloc(node_bytes);
    int32_t *ids = malloc(cell_bytes);
    require(raw && ids, "fixture allocations");
    for (size_t i = 0; i < node_bytes; i++) {
        raw[i] = (uint8_t)(i ^ i >> 8 ^ i >> 16);
    }
    for (size_t i = 0; i < count; i++) {
        ids[i] = (int32_t)(i / 3) * 17;
    }
    require(!node_write_file(nodes, raw, node_bytes), "write raw nodes");
    require(!node_write_file(cells, ids, cell_bytes - 1), "write truncated cells");
    require(hx_node_pack(folder) && !access(nodes, R_OK) && !access(cells, R_OK),
        "preserve originals after partial packing failure");
    require(!node_write_file(cells, ids, cell_bytes), "repair raw cells");
    require(!hx_node_pack(folder) && access(nodes, F_OK) && access(cells, F_OK),
        "retry publishes both packed files");
    require(!hx_node_pack(folder), "resuming a packed unit is safe");

    void *loaded = NULL;
    size_t n = 0;
    require(!hx_node_read_range(folder, first, last, &loaded, &n) && n == last - first &&
            !memcmp(loaded, raw + first * HX_NODE_BYTES, n * HX_NODE_BYTES),
        "streamed subrange matches raw nodes");
    hx_node_free(loaded);
    require(!hx_node_read(folder, &loaded, &n) && n == count && !memcmp(loaded, raw, node_bytes),
        "packed nodes match all raw records");
    hx_node_free(loaded);

    int32_t *loaded_ids = NULL;
    require(!hx_cell_read(folder, &loaded_ids, &n) && n == count &&
            !memcmp(loaded_ids, ids, cell_bytes),
        "delta-coded cells match raw indices");
    free(loaded_ids);
    free(ids);
    free(raw);
    unlink(packed_nodes);
    unlink(packed_cells);
    require(!node_write_file(nodes, "", 0) && !node_write_file(cells, "", 0), "write empty unit");
    pthread_t worker;
    require(!pthread_create(&worker, NULL, pack_thread, (void *)folder), "spawn concurrent pack");
    require(!hx_node_pack(folder), "pack empty unit");
    void *status = NULL;
    require(!pthread_join(worker, &status) && !status, "concurrent empty unit pack");
    require(!hx_node_count(folder, &n) && !n, "read empty unit count");
    require(!hx_node_read(folder, &loaded, &n) && !n, "read empty node unit");
    hx_node_free(loaded);
    require(!hx_cell_read(folder, &loaded_ids, &n) && !n, "read empty cell unit");
    free(loaded_ids);
    unlink(packed_nodes);
    unlink(packed_cells);
    require(!rmdir(folder), "remove fixture");
    reference_golden();
    ruler_golden();
    puts("node-store golden passed");
    return 0;
}
