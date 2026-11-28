#include "export.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace sp {

static void put16(std::vector<uint8_t> &b, uint16_t v) {
    b.push_back((uint8_t)(v & 0xFF));
    b.push_back((uint8_t)(v >> 8));
}

static void put32(std::vector<uint8_t> &b, uint32_t v) {
    for (int k = 0; k < 4; ++k)
        b.push_back((uint8_t)((v >> (8 * k)) & 0xFF));
}

static void write_tiff(const std::string &path, const std::vector<float> &v, int rows, int cols) {
    std::vector<uint8_t> b;
    const uint32_t bytes = (uint32_t)(v.size() * 4);
    const uint32_t data = 8, ifd = data + bytes + (bytes & 1);
    b.insert(b.end(), {'I', 'I', 42, 0});
    put32(b, ifd);
    const uint8_t *raw = (const uint8_t *)v.data();
    b.insert(b.end(), raw, raw + bytes);
    if (bytes & 1)
        b.push_back(0);
    struct Tag {
        uint16_t id, type;
        uint32_t value;
    };
    const Tag tags[] = {{256, 4, (uint32_t)cols}, {257, 4, (uint32_t)rows}, {258, 3, 32},
        {259, 3, 1}, {262, 3, 1}, {273, 4, data}, {277, 3, 1}, {278, 4, (uint32_t)rows},
        {279, 4, bytes}, {284, 3, 1}, {339, 3, 3}};
    put16(b, (uint16_t)(sizeof(tags) / sizeof(tags[0])));
    for (const Tag &t : tags) {
        put16(b, t.id);
        put16(b, t.type);
        put32(b, 1);
        if (t.type == 3) {
            put16(b, (uint16_t)t.value);
            put16(b, 0);
        } else {
            put32(b, t.value);
        }
    }
    put32(b, 0);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    out.write((const char *)b.data(), (std::streamsize)b.size());
}

static std::string uuid4(std::mt19937_64 &rng) {
    uint64_t a = rng(), b = rng();
    a = (a & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
    b = (b & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
    char s[40];
    std::snprintf(s, sizeof s, "%08x-%04x-%04x-%04x-%012llx", (unsigned)(a >> 32),
        (unsigned)((a >> 16) & 0xFFFF), (unsigned)(a & 0xFFFF), (unsigned)(b >> 48),
        (unsigned long long)(b & 0xFFFFFFFFFFFFULL));
    return s;
}

int export_windings(
    const Model &m, const Shared &sh, const std::string &dir, const ExportSettings &e) {
    mkdir(dir.c_str(), 0755);
    std::vector<float> zs;
    for (double z = (double)(m.s.z_begin - m.s.flow_z_margin);
        z < (double)(m.s.z_end + m.s.flow_z_margin); z += e.step)
        zs.push_back((float)z);
    std::vector<std::vector<std::array<float, 2>>> windings((size_t)e.outer_winding);
    size_t columns = 0;
    for (int k = 0; k < e.outer_winding; ++k) {
        const double step = (double)e.step / ((double)k + 0.5) / (double)sh.dr;
        const size_t count = (size_t)std::ceil(2.0 * M_PI / step);
        for (size_t i = 0; i < count; ++i) {
            const double theta = (double)k * 2.0 * M_PI + (double)i * step;
            const double r = (double)sh.dr * theta / (2.0 * M_PI);
            windings[(size_t)k].push_back(
                {(float)(std::sin(theta) * r), (float)(std::cos(theta) * r)});
        }
        columns += count;
    }
    const size_t rows = zs.size();
    std::vector<float> spiral(rows * columns * 3), scroll(rows * columns * 3);
    for (size_t r = 0; r < rows; ++r) {
        size_t c = 0;
        for (const auto &w : windings)
            for (const auto &yx : w) {
                const size_t at = (r * columns + c++) * 3;
                spiral[at] = zs[r];
                spiral[at + 1] = yx[0];
                spiral[at + 2] = yx[1];
            }
    }
    spiral_to_slice(m, sh, spiral.data(), rows * columns, scroll.data());
    std::mt19937_64 rng(std::random_device{}());
    int written = 0;
    size_t offset = 0;
    for (int k = 0; k < e.outer_winding; ++k) {
        const size_t n = windings[(size_t)k].size();
        if (n >= 2 && k >= e.first_winding) {
            std::vector<float> x(rows * n), y(rows * n), z(rows * n);
            double lo[3] = {INFINITY, INFINITY, INFINITY},
                   hi[3] = {-INFINITY, -INFINITY, -INFINITY};
            std::vector<uint8_t> valid(rows * n, 0);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < n; ++c) {
                    const size_t src = (r * columns + offset + c) * 3, dst = r * n + (n - 1 - c);
                    float pz = scroll[src], py = scroll[src + 1], px = scroll[src + 2];
                    if (pz < (float)m.s.z_begin || pz >= (float)m.s.z_end)
                        pz = py = px = -1.0f;
                    x[dst] = px;
                    y[dst] = py;
                    z[dst] = pz;
                    valid[dst] = !(px == -1.0f && py == -1.0f && pz == -1.0f);
                    if (valid[dst]) {
                        const double v[3] = {px, py, pz};
                        for (int a = 0; a < 3; ++a) {
                            lo[a] = std::min(lo[a], v[a]);
                            hi[a] = std::max(hi[a], v[a]);
                        }
                    }
                }
            size_t quads = 0;
            for (size_t r = 0; r + 1 < rows; ++r)
                for (size_t c = 0; c + 1 < n; ++c)
                    quads += valid[r * n + c] && valid[(r + 1) * n + c] && valid[r * n + c + 1] &&
                        valid[(r + 1) * n + c + 1];
            char name[16];
            std::snprintf(name, sizeof name, "w%03d", k);
            const std::string out = dir + "/" + name;
            mkdir(out.c_str(), 0755);
            write_tiff(out + "/x.tif", x, (int)rows, (int)n);
            write_tiff(out + "/y.tif", y, (int)rows, (int)n);
            write_tiff(out + "/z.tif", z, (int)rows, (int)n);
            const double area = (double)quads * e.step * e.step;
            const bool any = std::isfinite(lo[0]);
            char meta[1024];
            std::snprintf(meta, sizeof meta,
                "{\n    \"scale\": [%.17g, %.17g],\n    \"bbox\": [[%.9g, %.9g, %.9g], [%.9g, "
                "%.9g, %.9g]],\n"
                "    \"area_vx2\": %.17g,\n    \"area_cm2\": %.12g,\n    \"format\": \"tifxyz\",\n "
                "   "
                "\"type\": \"seg\",\n"
                "    \"uuid\": \"%s\",\n    \"source\": \"ports/spiral fitted\",\n"
                "    \"grid_orientation\": {\"z_direction_is_top_to_bottom\": null}\n}\n",
                1.0 / e.step, 1.0 / e.step, any ? lo[0] : -1.0, any ? lo[1] : -1.0,
                any ? lo[2] : -1.0, any ? hi[0] : -1.0, any ? hi[1] : -1.0, any ? hi[2] : -1.0,
                area, area * e.voxel_um * e.voxel_um / 1e8, uuid4(rng).c_str());
            std::ofstream(out + "/meta.json") << meta;
            ++written;
        }
        offset += n;
    }
    return written;
}

}
