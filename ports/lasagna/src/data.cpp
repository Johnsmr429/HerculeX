#include "data.h"

#include "io.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

namespace la {

static constexpr int64_t CHUNK = 32, PADDED = 34;

static bool read_file(const std::string &path, std::vector<uint8_t> &out) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return false;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    out.resize((size_t)st.st_size);
    size_t got = 0;
    while (got < out.size()) {
        const ssize_t r = ::read(fd, out.data() + got, out.size() - got);
        if (r <= 0)
            break;
        got += (size_t)r;
    }
    ::close(fd);
    out.resize(got);
    return true;
}

Volume Volume::open(const std::string &manifest, const std::vector<std::string> &names) {
    Volume v;
    const std::string base = manifest.substr(0, manifest.find_last_of('/'));
    Json m = Json::parse(read_text(manifest));
    std::array<double, 3> full{};
    for (int i = 0; i < 3; ++i)
        full[i] = m["base_shape_zyx"][i].num();
    const double factor = m.has("grad_mag_factor") ? m["grad_mag_factor"].num() : 1.0;
    v.grad_mag_scale = (float)(m["grad_mag_encode_scale"].num() * factor);
    for (const std::string &name : names) {
        const Json &g = m["groups"][name];
        Channel c;
        c.name = name;
        c.array = base + "/" + g["zarr"].str();
        Json z = Json::parse(read_text(c.array + "/.zarray"));
        if (z["dtype"].str() != "|u1" || z["compressor"]["id"].str() != "blosc")
            throw std::runtime_error("unexpected zarr array " + c.array);
        for (int i = 0; i < 3; ++i) {
            c.shape[i] = (int64_t)z["shape"][i].num();
            c.chunks[i] = (int64_t)z["chunks"][i].num();
            if (c.chunks[i] != CHUNK)
                throw std::runtime_error("chunks other than 32 in " + c.array);
            c.grid[i] = (c.shape[i] + CHUNK - 1) / CHUNK;
        }
        for (int a = 0; a < 3; ++a)
            c.inv_scale[a] = (float)(1.0 / (double)(full[2 - a] / (double)c.shape[2 - a]));
        c.table.assign((size_t)(c.grid[0] * c.grid[1] * c.grid[2]), nullptr);
        c.absent.assign(c.table.size(), 0);
        v.channels.push_back(std::move(c));
    }
    return v;
}

Channel &Volume::channel(const std::string &name) {
    for (auto &c : channels)
        if (c.name == name)
            return c;
    throw std::runtime_error("no channel " + name);
}

const uint8_t *Channel::block(int64_t cz, int64_t cy, int64_t cx) const {
    if (cz < 0 || cy < 0 || cx < 0 || cz >= grid[0] || cy >= grid[1] || cx >= grid[2])
        return nullptr;
    return table[(size_t)((cz * grid[1] + cy) * grid[2] + cx)];
}

void Channel::load(const std::vector<int64_t> &cells) {
    std::vector<int64_t> fresh;
    for (int64_t cell : cells)
        if (!table[(size_t)cell])
            fresh.push_back(cell);
    std::sort(fresh.begin(), fresh.end());
    fresh.erase(std::unique(fresh.begin(), fresh.end()), fresh.end());
    const int64_t G0 = grid[0], G1 = grid[1], G2 = grid[2];
    for (size_t first = 0; first < fresh.size(); first += 4096) {
        const size_t count = std::min<size_t>(4096, fresh.size() - first);
        std::vector<int64_t> sources;
        for (size_t k = first; k < first + count; ++k) {
            const int64_t cz = fresh[k] / (G1 * G2), cy = (fresh[k] / G2) % G1, cx = fresh[k] % G2;
            for (int64_t nz = cz - 1; nz <= cz + 1; ++nz)
                for (int64_t ny = cy - 1; ny <= cy + 1; ++ny)
                    for (int64_t nx = cx - 1; nx <= cx + 1; ++nx)
                        if (nz >= 0 && ny >= 0 && nx >= 0 && nz < G0 && ny < G1 && nx < G2)
                            sources.push_back((nz * G1 + ny) * G2 + nx);
        }
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        struct Source {
            std::shared_ptr<const std::vector<uint8_t>> keep;
            const uint8_t *base = nullptr;
            int64_t sz = CHUNK * CHUNK, sy = CHUNK;
        };
        std::vector<Source> decoded(sources.size());
        std::vector<int64_t> fetch;
        for (size_t k = 0; k < sources.size(); ++k) {
            const int64_t cell = sources[k];
            if (const uint8_t *block = table[(size_t)cell]) {
                decoded[k] = {nullptr, block + (PADDED + 1) * PADDED + 1, PADDED * PADDED, PADDED};
                continue;
            }
            if (absent[(size_t)cell])
                continue;
            const auto hit = recent.find(cell);
            if (hit != recent.end())
                decoded[k] = {hit->second, hit->second->data()};
            else
                fetch.push_back((int64_t)k);
        }
        Source *out = decoded.data();
        const int64_t *todo = fetch.data(), *src = sources.data();
        const std::string folder = array;
        dispatch_apply(
            fetch.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
              const int64_t k = todo[j], cell = src[k];
              const std::string path = folder + "/" + std::to_string(cell / (G1 * G2)) + "/" +
                  std::to_string((cell / G2) % G1) + "/" + std::to_string(cell % G2);
              std::vector<uint8_t> frame;
              if (!read_file(path, frame))
                  return;
              auto raw = std::make_shared<std::vector<uint8_t>>((size_t)(CHUNK * CHUNK * CHUNK));
              blosc_decode(frame.data(), frame.size(), raw->data(), raw->size());
              out[k] = {raw, raw->data()};
            });
        for (int64_t k : fetch) {
            const int64_t cell = sources[(size_t)k];
            if (!decoded[(size_t)k].base) {
                absent[(size_t)cell] = 1;
                continue;
            }
            if (recent_order.size() < 2048)
                recent_order.push_back(cell);
            else {
                recent.erase(recent_order[recent_next]);
                recent_order[recent_next] = cell;
                recent_next = (recent_next + 1) % recent_order.size();
            }
            recent[cell] = decoded[(size_t)k].keep;
        }
        std::vector<std::unique_ptr<uint8_t[]>> made(count);
        std::unique_ptr<uint8_t[]> *slots = made.data();
        const int64_t *wanted = fresh.data() + first;
        const std::array<int64_t, 3> extent = shape;
        const std::vector<int64_t> *keys = &sources;
        const Source *chunks = decoded.data();
        dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
          const int64_t cell = wanted[k];
          const int64_t cz = cell / (G1 * G2), cy = (cell / G2) % G1, cx = cell % G2;
          auto block = std::make_unique<uint8_t[]>((size_t)(PADDED * PADDED * PADDED));
          std::memset(block.get(), 0, (size_t)(PADDED * PADDED * PADDED));
          const int64_t start[3] = {cz * CHUNK - 1, cy * CHUNK - 1, cx * CHUNK - 1};
          for (int64_t nz = cz - 1; nz <= cz + 1; ++nz)
              for (int64_t ny = cy - 1; ny <= cy + 1; ++ny)
                  for (int64_t nx = cx - 1; nx <= cx + 1; ++nx) {
                      if (nz < 0 || ny < 0 || nx < 0 || nz >= G0 || ny >= G1 || nx >= G2)
                          continue;
                      const int64_t key = (nz * G1 + ny) * G2 + nx;
                      const auto at = std::lower_bound(keys->begin(), keys->end(), key);
                      const Source &raw = chunks[(size_t)(at - keys->begin())];
                      if (!raw.base)
                          continue;
                      const int64_t lo[3] = {nz * CHUNK, ny * CHUNK, nx * CHUNK};
                      int64_t a[3], b[3];
                      for (int d = 0; d < 3; ++d) {
                          a[d] = std::max({lo[d], start[d], (int64_t)0});
                          b[d] = std::min({lo[d] + CHUNK, start[d] + PADDED, extent[d]});
                      }
                      for (int64_t z = a[0]; z < b[0]; ++z)
                          for (int64_t y = a[1]; y < b[1]; ++y)
                              std::memcpy(block.get() +
                                      ((z - start[0]) * PADDED + (y - start[1])) * PADDED +
                                      (a[2] - start[2]),
                                  raw.base + (z - lo[0]) * raw.sz + (y - lo[1]) * raw.sy +
                                      (a[2] - lo[2]),
                                  (size_t)(b[2] - a[2]));
                  }
          slots[k] = std::move(block);
        });
        for (size_t k = 0; k < count; ++k) {
            table[(size_t)fresh[first + k]] = made[k].get();
            blocks.push_back(std::move(made[k]));
        }
    }
}

void Volume::prefetch(Channel &c, const float *xyz, size_t n) {
    std::vector<int64_t> cells;
    cells.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float p[3];
        bool finite = true;
        for (int a = 0; a < 3; ++a) {
            p[a] = (xyz[i * 3 + a] - origin[a]) * c.inv_scale[a];
            finite = finite && std::isfinite(p[a]);
        }
        if (!finite)
            continue;
        const int64_t ci[3] = {(int64_t)std::floor(p[2] / 32.0f), (int64_t)std::floor(p[1] / 32.0f),
            (int64_t)std::floor(p[0] / 32.0f)};
        if (ci[0] < 0 || ci[1] < 0 || ci[2] < 0 || ci[0] >= c.grid[0] || ci[1] >= c.grid[1] ||
            ci[2] >= c.grid[2])
            continue;
        cells.push_back((ci[0] * c.grid[1] + ci[1]) * c.grid[2] + ci[2]);
    }
    c.load(cells);
}

float decode(const Volume &v, const std::string &name, float raw) {
    if (name == "grad_mag")
        return raw / v.grad_mag_scale;
    if (name == "nx" || name == "ny")
        return (raw - 128.0f) / 127.0f;
    if (name == "cos")
        return raw / 255.0f;
    return raw;
}

float decode_backward(const Volume &v, const std::string &name, float grad) {
    if (name == "grad_mag")
        return grad / v.grad_mag_scale;
    if (name == "nx" || name == "ny")
        return grad / 127.0f;
    if (name == "cos")
        return grad / 255.0f;
    return grad;
}

struct Cell {
    const uint8_t *chunk;
    float fx, fy, fz;
    int i000, i100, i010, i110, i001, i101, i011, i111;
};

static bool locate(const Volume &v, const Channel &c, const float *q, Cell &cell) {
    const float px = (q[0] - v.origin[0]) * c.inv_scale[0];
    const float py = (q[1] - v.origin[1]) * c.inv_scale[1];
    const float pz = (q[2] - v.origin[2]) * c.inv_scale[2];
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
        return false;
    const int ci_x = (int)std::floor(px / 32.0f), ci_y = (int)std::floor(py / 32.0f),
              ci_z = (int)std::floor(pz / 32.0f);
    const uint8_t *chunk = c.block(ci_z, ci_y, ci_x);
    if (!chunk)
        return false;
    cell.chunk = chunk;
    const float lx = px - (float)ci_x * 32.0f + 1.0f, ly = py - (float)ci_y * 32.0f + 1.0f,
                lz = pz - (float)ci_z * 32.0f + 1.0f;
    const float lx0 = std::floor(lx), ly0 = std::floor(ly), lz0 = std::floor(lz);
    cell.fx = lx - lx0;
    cell.fy = ly - ly0;
    cell.fz = lz - lz0;
    const int ix0 = std::clamp((int)lx0, 0, 33), iy0 = std::clamp((int)ly0, 0, 33),
              iz0 = std::clamp((int)lz0, 0, 33);
    const int ix1 = std::min(ix0 + 1, 33), iy1 = std::min(iy0 + 1, 33), iz1 = std::min(iz0 + 1, 33);
    const int sz = 34 * 34, sy = 34;
    cell.i000 = iz0 * sz + iy0 * sy + ix0;
    cell.i100 = iz0 * sz + iy0 * sy + ix1;
    cell.i010 = iz0 * sz + iy1 * sy + ix0;
    cell.i110 = iz0 * sz + iy1 * sy + ix1;
    cell.i001 = iz1 * sz + iy0 * sy + ix0;
    cell.i101 = iz1 * sz + iy0 * sy + ix1;
    cell.i011 = iz1 * sz + iy1 * sy + ix0;
    cell.i111 = iz1 * sz + iy1 * sy + ix1;
    return true;
}

static float interpolate(const Cell &q) {
    const float fx = q.fx, fy = q.fy, fz = q.fz;
    const uint8_t *ch = q.chunk;
    float val = 0.0f;
    val = std::fma((1.0f - fx) * (1.0f - fy) * (1.0f - fz), (float)ch[q.i000], val);
    val = std::fma(fx * (1.0f - fy) * (1.0f - fz), (float)ch[q.i100], val);
    val = std::fma((1.0f - fx) * fy * (1.0f - fz), (float)ch[q.i010], val);
    val = std::fma(fx * fy * (1.0f - fz), (float)ch[q.i110], val);
    val = std::fma((1.0f - fx) * (1.0f - fy) * fz, (float)ch[q.i001], val);
    val = std::fma(fx * (1.0f - fy) * fz, (float)ch[q.i101], val);
    val = std::fma((1.0f - fx) * fy * fz, (float)ch[q.i011], val);
    val = std::fma(fx * fy * fz, (float)ch[q.i111], val);
    return val;
}

template <class F> static void parallel(size_t n, F body) {
    if (n <= 4096) {
        for (size_t i = 0; i < n; ++i)
            body(i);
        return;
    }
    const size_t blocks = (n + 2047) / 2048;
    dispatch_apply(blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
      for (size_t i = b * 2048; i < std::min(n, (b + 1) * 2048); ++i)
          body(i);
    });
}

void sample_u8(const Volume &v, const Channel &c, const float *xyz, size_t n, uint8_t *out) {
    parallel(n, [&](size_t i) {
        Cell q;
        if (!locate(v, c, xyz + i * 3, q)) {
            out[i] = 0;
            return;
        }
        out[i] = (uint8_t)std::clamp((int)std::round(interpolate(q)), 0, 255);
    });
}

void sample_raw(const Volume &v, const Channel &c, const float *xyz, size_t n, float *out) {
    parallel(n, [&](size_t i) {
        Cell q;
        out[i] = locate(v, c, xyz + i * 3, q) ? interpolate(q) : 0.0f;
    });
}

void sample_raw_backward(const Volume &v, const Channel &c, const float *xyz, size_t n,
    const float *grad_out, float *grad_xyz) {
    parallel(n, [&](size_t i) {
        Cell q;
        float g[3] = {0.0f, 0.0f, 0.0f};
        if (locate(v, c, xyz + i * 3, q)) {
            const float fx = q.fx, fy = q.fy, fz = q.fz, go = grad_out[i];
            const uint8_t *ch = q.chunk;
            const float v000 = ch[q.i000], v100 = ch[q.i100], v010 = ch[q.i010], v110 = ch[q.i110];
            const float v001 = ch[q.i001], v101 = ch[q.i101], v011 = ch[q.i011], v111 = ch[q.i111];
            float dfx = (1.0f - fy) * (1.0f - fz) * (v100 - v000);
            dfx = std::fma(fy * (1.0f - fz), v110 - v010, dfx);
            dfx = std::fma((1.0f - fy) * fz, v101 - v001, dfx);
            dfx = std::fma(fy * fz, v111 - v011, dfx);
            float dfy = (1.0f - fx) * (1.0f - fz) * (v010 - v000);
            dfy = std::fma(fx * (1.0f - fz), v110 - v100, dfy);
            dfy = std::fma((1.0f - fx) * fz, v011 - v001, dfy);
            dfy = std::fma(fx * fz, v111 - v101, dfy);
            float dfz = (1.0f - fx) * (1.0f - fy) * (v001 - v000);
            dfz = std::fma(fx * (1.0f - fy), v101 - v100, dfz);
            dfz = std::fma((1.0f - fx) * fy, v011 - v010, dfz);
            dfz = std::fma(fx * fy, v111 - v110, dfz);
            g[0] = std::fma(go * dfx, c.inv_scale[0], 0.0f);
            g[1] = std::fma(go * dfy, c.inv_scale[1], 0.0f);
            g[2] = std::fma(go * dfz, c.inv_scale[2], 0.0f);
        }
        grad_xyz[i * 3] = g[0];
        grad_xyz[i * 3 + 1] = g[1];
        grad_xyz[i * 3 + 2] = g[2];
    });
}

}
