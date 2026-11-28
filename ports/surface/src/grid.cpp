#include "grid.h"

#include "json.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace hx {

namespace {

constexpr int64_t CUBE = 128;
constexpr uint32_t PLANE = (uint32_t)(CUBE * CUBE), DATA = 256,
                   FIRST = DATA + (uint32_t)CUBE * PLANE;

void put16(std::string &b, uint16_t v) {
    b += (char)(v & 0xFF);
    b += (char)(v >> 8);
}

void put32(std::string &b, uint32_t v) {
    for (int k = 0; k < 4; ++k)
        b += (char)((v >> (8 * k)) & 0xFF);
}

void entry(std::string &b, uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    put16(b, tag);
    put16(b, type);
    put32(b, count);
    if (type == 3) {
        put16(b, (uint16_t)value);
        put16(b, 0);
    } else {
        put32(b, value);
    }
}

void plain_tags(std::string &b) {
    entry(b, 256, 4, 1, CUBE);
    entry(b, 257, 4, 1, CUBE);
    entry(b, 258, 3, 1, 8);
    entry(b, 259, 3, 1, 1);
    entry(b, 262, 3, 1, 1);
}

std::string cube_head() {
    std::string h;
    h += "II*";
    h += '\0';
    put32(h, 8);
    put16(h, 14);
    plain_tags(h);
    entry(h, 270, 2, 27, 182);
    entry(h, 273, 4, 1, DATA);
    entry(h, 277, 3, 1, 1);
    entry(h, 278, 4, 1, CUBE);
    entry(h, 279, 4, 1, PLANE);
    entry(h, 282, 5, 1, 224);
    entry(h, 283, 5, 1, 232);
    entry(h, 296, 3, 1, 1);
    entry(h, 305, 2, 12, 240);
    put32(h, FIRST);
    h += "{\"shape\": [128, 128, 128]}";
    h += '\0';
    h.resize(224, '\0');
    for (int k = 0; k < 4; ++k)
        put32(h, 1);
    h += "tifffile.py";
    h += '\0';
    h.resize(DATA, '\0');
    return h;
}

std::string cube_tail() {
    std::string t;
    const uint32_t shared = FIRST + 150;
    for (int64_t k = 1; k < CUBE; ++k) {
        const uint32_t at = FIRST + (uint32_t)(k - 1) * 166;
        put16(t, 12);
        plain_tags(t);
        entry(t, 273, 4, 1, DATA + (uint32_t)k * PLANE);
        entry(t, 277, 3, 1, 1);
        entry(t, 278, 4, 1, CUBE);
        entry(t, 279, 4, 1, PLANE);
        entry(t, 282, 5, 1, shared);
        entry(t, 283, 5, 1, shared + 8);
        entry(t, 296, 3, 1, 1);
        put32(t, k + 1 < CUBE ? at + 166 : 0);
        for (int r = 0; r < 4; ++r)
            put32(t, 1);
    }
    return t;
}

void write_cube(const std::string &path, const std::string &head, const std::string &tail,
    const uint8_t *cube) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    out.write(head.data(), (std::streamsize)head.size());
    out.write((const char *)cube, (std::streamsize)(CUBE * PLANE));
    out.write(tail.data(), (std::streamsize)tail.size());
}

}

int64_t write_fiesta_grid(const std::string &out, const uint8_t *surface, const Box &box,
    const Reader &read, const std::string &raw_source, const std::string &pred_dir,
    const std::string &net, int threshold) {
    for (const auto &span : box)
        if (span[0] % CUBE || span[1] % CUBE)
            throw std::runtime_error("--fiesta-grid needs a box on multiples of 128");
    const int64_t z0 = box[0][0], z1 = box[0][1], y0 = box[1][0], y1 = box[1][1], x0 = box[2][0],
                  x1 = box[2][1];
    const int64_t ny = y1 - y0, nx = x1 - x0;
    const std::string pred = out + "/cubes_PRED", raw_dir = out + "/cubes_RAW";
    for (const std::string &dir : {out, pred, raw_dir})
        mkdir(dir.c_str(), 0755);
    const std::string head = cube_head(), tail = cube_tail();
    std::vector<uint8_t> band((size_t)(CUBE * ny * nx)), cube((size_t)(CUBE * PLANE));
    std::vector<std::string> present;
    for (int64_t bz = z0; bz < z1; bz += CUBE) {
        read(Box{{{bz, bz + CUBE}, {y0, y1}, {x0, x1}}}, band.data());
        for (int64_t by = 0; by < ny; by += CUBE)
            for (int64_t bx = 0; bx < nx; bx += CUBE) {
                char id[40];
                std::snprintf(id, sizeof id, "z%05lld_y%05lld_x%05lld", (long long)bz,
                    (long long)(y0 + by), (long long)(x0 + bx));
                bool any = false;
                for (int64_t z = 0; z < CUBE; ++z)
                    for (int64_t y = 0; y < CUBE; ++y) {
                        const uint8_t *row = surface + ((bz - z0 + z) * ny + by + y) * nx + bx;
                        uint8_t *dst = cube.data() + (z * CUBE + y) * CUBE;
                        for (int64_t x = 0; x < CUBE; ++x) {
                            dst[x] = row[x] >= threshold ? 255 : 0;
                            any = any || dst[x];
                        }
                    }
                if (any) {
                    write_cube(pred + "/" + id + ".tif", head, tail, cube.data());
                    present.push_back(id);
                }
                for (int64_t z = 0; z < CUBE; ++z)
                    for (int64_t y = 0; y < CUBE; ++y)
                        std::copy_n(band.data() + (z * ny + by + y) * nx + bx, CUBE,
                            cube.data() + (z * CUBE + y) * CUBE);
                write_cube(raw_dir + "/" + id + ".tif", head, tail, cube.data());
            }
    }
    std::sort(present.begin(), present.end());
    std::string list = "[";
    for (size_t i = 0; i < present.size(); ++i)
        list += (i ? ",\n" : "\n") + quote(present[i]);
    list += present.empty() ? "]" : "\n]";
    std::ofstream(pred + "/present.json", std::ios::binary | std::ios::trunc) << list;
    std::string m = "{\n \"chunk_size\": 128,\n \"bbox_l0_zyx\": [\n";
    const int64_t bounds[6] = {z0, z1, y0, y1, x0, x1};
    for (int i = 0; i < 6; ++i)
        m += "  " + std::to_string(bounds[i]) + (i < 5 ? ",\n" : "\n");
    m += " ],\n \"n_chunks\": [\n  " + std::to_string((z1 - z0) / CUBE) + ",\n  " +
        std::to_string(ny / CUBE) + ",\n  " + std::to_string(nx / CUBE) +
        "\n ],\n \"sources\": {\n  \"pred\": " + quote(pred_dir) +
        ",\n  \"raw\": " + quote(raw_source) +
        "\n },\n \"umbilicus_yx\": null,\n \"created_by\": \"HerculeX ports/surface\",\n" +
        " \"threshold\": " + std::to_string(threshold) + ",\n \"net\": " + quote(net) + "\n}";
    std::ofstream(out + "/manifest.json", std::ios::binary | std::ios::trunc) << m;
    return (int64_t)present.size();
}

}
