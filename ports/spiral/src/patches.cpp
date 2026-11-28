#include "patches.h"

#include "io.h"
#include "json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace sp {

namespace {

constexpr float TWO_PI = 6.2831855f, PI = 3.1415927f;

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int crossing(float delta) { return (delta > PI ? 1 : 0) - (delta < -PI ? 1 : 0); }

void build_masks(Patch &p) {
    const int H = p.H, W = p.W;
    p.vertex_valid.assign((size_t)H * W, 0);
    for (size_t i = 0; i < (size_t)H * W; ++i)
        p.vertex_valid[i] =
            !(p.zyx[3 * i] == -1.0f && p.zyx[3 * i + 1] == -1.0f && p.zyx[3 * i + 2] == -1.0f);
    p.quad_valid.assign((size_t)(H - 1) * (W - 1), 0);
    for (int r = 0; r + 1 < H; ++r)
        for (int c = 0; c + 1 < W; ++c) {
            const auto v = [&](int rr, int cc) { return p.vertex_valid[(size_t)rr * W + cc] != 0; };
            p.quad_valid[(size_t)r * (W - 1) + c] =
                v(r, c) && v(r + 1, c) && v(r, c + 1) && v(r + 1, c + 1);
        }
}

bool any(const std::vector<uint8_t> &v) {
    return std::any_of(v.begin(), v.end(), [](uint8_t x) { return x != 0; });
}

void erode(Patch &p, int cells) {
    const int H = p.H, W = p.W;
    std::vector<uint8_t> current = p.vertex_valid;
    for (int it = 0; it < cells; ++it) {
        std::vector<uint8_t> next((size_t)H * W, 0);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                const auto at = [&](int rr, int cc) {
                    return rr >= 0 && rr < H && cc >= 0 && cc < W && current[(size_t)rr * W + cc];
                };
                next[(size_t)r * W + c] =
                    at(r, c) && at(r - 1, c) && at(r + 1, c) && at(r, c - 1) && at(r, c + 1);
            }
        current.swap(next);
    }
    for (size_t i = 0; i < (size_t)H * W; ++i)
        if (p.vertex_valid[i] && !current[i])
            p.zyx[3 * i] = p.zyx[3 * i + 1] = p.zyx[3 * i + 2] = -1.0f;
    build_masks(p);
}

std::vector<uint8_t> largest_component(const std::vector<uint8_t> &mask, int rows, int cols) {
    std::vector<int32_t> label(mask.size(), 0);
    std::vector<int64_t> sizes{0};
    std::vector<size_t> stack;
    for (size_t start = 0; start < mask.size(); ++start) {
        if (!mask[start] || label[start])
            continue;
        const int32_t id = (int32_t)sizes.size();
        int64_t size = 0;
        label[start] = id;
        stack.push_back(start);
        while (!stack.empty()) {
            const size_t at = stack.back();
            stack.pop_back();
            ++size;
            const int r = (int)(at / (size_t)cols), c = (int)(at % (size_t)cols);
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    const int rr = r + dr, cc = c + dc;
                    if (rr < 0 || rr >= rows || cc < 0 || cc >= cols)
                        continue;
                    const size_t n = (size_t)rr * cols + cc;
                    if (mask[n] && !label[n]) {
                        label[n] = id;
                        stack.push_back(n);
                    }
                }
        }
        sizes.push_back(size);
    }
    if (sizes.size() <= 2)
        return mask;
    const int32_t best =
        (int32_t)(std::max_element(sizes.begin() + 1, sizes.end()) - sizes.begin());
    std::vector<uint8_t> out(mask.size(), 0);
    for (size_t i = 0; i < mask.size(); ++i)
        out[i] = label[i] == best;
    return out;
}

void sampling_mask(Patch &p, int64_t z_begin, int64_t z_end, double margin) {
    const int W = p.W, cols = W - 1, rows = p.H - 1;
    std::vector<uint8_t> roi((size_t)rows * cols, 0);
    const float lo = (float)((double)z_begin - margin), hi = (float)((double)z_end + margin);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const auto z = [&](int rr, int cc) { return p.zyx[((size_t)rr * W + cc) * 3]; };
            const float qz = (z(r, c) + z(r + 1, c) + z(r, c + 1) + z(r + 1, c + 1)) / 4.0f;
            roi[(size_t)r * cols + c] = p.quad_valid[(size_t)r * cols + c] && qz >= lo && qz < hi;
        }
    if (!any(roi))
        roi = p.quad_valid;
    p.sampling = largest_component(roi, rows, cols);
    int64_t count = 0;
    for (uint8_t v : p.sampling)
        count += v;
    p.area = (double)count * ((1.0 / p.scale[0]) * (1.0 / p.scale[1]));
}

void build_tree(Patch &p) {
    const int64_t cols = p.W - 1;
    p.cells.clear();
    int64_t row_hi = 0, col_hi = 0;
    p.row_lo = p.H - 1;
    p.col_lo = cols;
    for (int64_t r = 0; r < p.H - 1; ++r)
        for (int64_t c = 0; c < cols; ++c)
            if (p.sampling[(size_t)(r * cols + c)]) {
                p.cells.push_back(r * cols + c);
                p.row_lo = std::min(p.row_lo, r);
                p.col_lo = std::min(p.col_lo, c);
                row_hi = std::max(row_hi, r + 1);
                col_hi = std::max(col_hi, c + 1);
            }
    p.rect_w = col_hi - p.col_lo;
    const int64_t count = (int64_t)p.cells.size();
    p.rectangular = count == (row_hi - p.row_lo) * p.rect_w;
    p.preorder.assign((size_t)count, 0);
    p.parent.assign((size_t)count, -1);
    if (p.rectangular) {
        for (int64_t position = 0; position < count; ++position) {
            const int64_t row = position / p.rect_w, offset = position % p.rect_w;
            const int64_t column = row % 2 == 0 ? offset : p.rect_w - 1 - offset;
            p.preorder[(size_t)position] = row * p.rect_w + column;
            p.parent[(size_t)position] = position == 0 ? -1 : p.preorder[(size_t)position - 1];
        }
        return;
    }
    std::vector<int64_t> dense((size_t)((p.H - 1) * cols), -1);
    for (int64_t o = 0; o < count; ++o)
        dense[(size_t)p.cells[(size_t)o]] = o;
    struct Frame {
        int64_t ordinal;
        std::array<int64_t, 8> neighbors{};
        int count = 0, next = 0;
    };
    const int offsets[8][2] = {
        {0, 1}, {1, -1}, {1, 0}, {1, 1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};
    auto frame_of = [&](int64_t ordinal) {
        Frame f;
        f.ordinal = ordinal;
        const int64_t linear = p.cells[(size_t)ordinal], row = linear / cols,
                      column = linear % cols;
        for (const auto &o : offsets) {
            const int64_t r = row + o[0], c = column + o[1];
            if (r < 0 || c < 0 || r >= p.H - 1 || c >= cols)
                continue;
            const int64_t n = dense[(size_t)(r * cols + c)];
            if (n >= 0)
                f.neighbors[(size_t)f.count++] = n;
        }
        return f;
    };
    std::vector<uint8_t> visited((size_t)count, 0);
    std::vector<Frame> stack;
    int64_t position = 0;
    for (int64_t seed = 0; seed < count; ++seed) {
        if (visited[(size_t)seed])
            continue;
        visited[(size_t)seed] = 1;
        p.preorder[(size_t)position] = seed;
        p.parent[(size_t)position] = -1;
        ++position;
        stack.push_back(frame_of(seed));
        while (!stack.empty()) {
            Frame &f = stack.back();
            if (f.next >= f.count) {
                stack.pop_back();
                continue;
            }
            const int64_t child = f.neighbors[(size_t)f.next++];
            if (visited[(size_t)child])
                continue;
            visited[(size_t)child] = 1;
            p.preorder[(size_t)position] = child;
            p.parent[(size_t)position] = f.ordinal;
            ++position;
            stack.push_back(frame_of(child));
        }
    }
}

void lookup(const Patch &p, float fi, float fj, float *out) {
    int64_t i0 = (int64_t)std::floor(fi), j0 = (int64_t)std::floor(fj);
    i0 = std::min<int64_t>(std::max<int64_t>(i0, 0), p.H - 2);
    j0 = std::min<int64_t>(std::max<int64_t>(j0, 0), p.W - 2);
    const float di = std::clamp(fi - (float)i0, 0.0f, 1.0f),
                dj = std::clamp(fj - (float)j0, 0.0f, 1.0f);
    const float *tl = p.zyx.data() + ((size_t)i0 * p.W + (size_t)j0) * 3, *tr = tl + 3,
                *bl = tl + (size_t)p.W * 3, *br = bl + 3;
    for (int c = 0; c < 3; ++c) {
        const float top = tl[c] + (tr[c] - tl[c]) * dj, bottom = bl[c] + (br[c] - bl[c]) * dj;
        out[c] = top + (bottom - top) * di;
    }
}

std::vector<std::string> list_dir(const std::string &dir) {
    std::vector<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d)
        throw std::runtime_error("cannot list " + dir);
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n != "." && n != "..")
            names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    return names;
}

}

PatchSet load_patches(
    const std::string &dir, int64_t z_begin, int64_t z_end, const PatchSettings &ps) {
    PatchSet set;
    size_t prefiltered = 0, eroded = 0, outside = 0;
    for (const std::string &entry : list_dir(dir)) {
        const std::string path = dir + "/" + entry;
        try {
            Patch p;
            p.name = entry;
            const Json meta = Json::parse(read_text(path + "/meta.json"));
            const Json &scale = meta["scale"];
            if (scale.kind != Json::Array || scale.items.size() != 2)
                throw std::runtime_error("meta.json scale must contain two numbers");
            for (int k = 0; k < 2; ++k) {
                p.scale[k] = scale.items[(size_t)k].num();
                if (!(p.scale[k] > 0) || !std::isfinite(p.scale[k]))
                    throw std::runtime_error("meta.json scale must be positive");
            }
            int cells = ps.erode;
            if (meta.has("spiral_patch_erode_cells") &&
                meta["spiral_patch_erode_cells"].kind == Json::Number)
                cells = (int)meta["spiral_patch_erode_cells"].num();
            const Image z = read_tiff(path + "/z.tif");
            std::vector<float> mask;
            if (exists(path + "/mask.tif")) {
                const Image mk = read_tiff(path + "/mask.tif");
                mask.resize((size_t)mk.H * mk.W);
                for (size_t i = 0; i < mask.size(); ++i)
                    mask[i] = mk.data[i * (size_t)mk.C];
            }
            bool in_range = false;
            for (size_t i = 0; i < z.data.size() && !in_range; ++i)
                in_range = z.data[i] >= (float)z_begin && z.data[i] < (float)z_end &&
                    (mask.empty() || mask[i] != 0.0f);
            if (!in_range) {
                ++prefiltered;
                continue;
            }
            const Image y = read_tiff(path + "/y.tif"), x = read_tiff(path + "/x.tif");
            if (y.H != z.H || y.W != z.W || x.H != z.H || x.W != z.W)
                throw std::runtime_error("x, y and z disagree in shape");
            p.H = z.H;
            p.W = z.W;
            p.zyx.resize((size_t)p.H * p.W * 3);
            for (size_t i = 0; i < (size_t)p.H * p.W; ++i) {
                p.zyx[3 * i] = z.data[i];
                p.zyx[3 * i + 1] = y.data[i];
                p.zyx[3 * i + 2] = x.data[i];
                if (!mask.empty() && mask[i] == 0.0f)
                    p.zyx[3 * i] = p.zyx[3 * i + 1] = p.zyx[3 * i + 2] = -1.0f;
            }
            build_masks(p);
            if (!any(p.quad_valid))
                throw std::runtime_error("patch has no valid quad");
            if (cells > 0) {
                erode(p, cells);
                if (!any(p.quad_valid)) {
                    ++eroded;
                    continue;
                }
            }
            bool intersects = false;
            for (size_t i = 0; i < (size_t)p.H * p.W && !intersects; ++i)
                intersects = p.vertex_valid[i] && p.zyx[3 * i] >= (float)z_begin &&
                    p.zyx[3 * i] < (float)z_end;
            if (!intersects) {
                ++outside;
                continue;
            }
            sampling_mask(p, z_begin, z_end, ps.z_margin);
            build_tree(p);
            set.patches.push_back(std::move(p));
        } catch (const std::exception &e) {
            std::printf("Failed to load segment %s: %s\n", entry.c_str(), e.what());
        }
    }
    if (prefiltered)
        std::printf("z ROI prefilter skipped %zu patches\n", prefiltered);
    if (eroded)
        std::printf("erosion removed %zu patches\n", eroded);
    if (outside)
        std::printf("erosion moved %zu patches outside the z ROI\n", outside);
    for (const Patch &p : set.patches)
        for (size_t i = 0; i < (size_t)p.H * p.W; ++i)
            if (p.vertex_valid[i] && p.zyx[3 * i] >= (float)z_begin && p.zyx[3 * i] < (float)z_end)
                set.trusted.insert(set.trusted.end(), p.zyx.begin() + (long)(3 * i),
                    p.zyx.begin() + (long)(3 * i + 3));
    reindex_patches(set, ps);
    return set;
}

void reindex_patches(PatchSet &set, const PatchSettings &ps) {
    set.start.assign(set.patches.size(), 0);
    set.nodes = 0;
    for (size_t k = 0; k < set.patches.size(); ++k) {
        set.start[k] = set.nodes;
        set.nodes += (int64_t)set.patches[k].cells.size();
    }
    std::vector<float> weights(set.patches.size());
    float total = 0;
    for (size_t k = 0; k < weights.size(); ++k) {
        weights[k] = std::pow((float)set.patches[k].area, (float)ps.area_exponent);
        total += weights[k];
    }
    set.probability.assign(weights.size(), 0.0);
    for (size_t k = 0; k < weights.size(); ++k)
        set.probability[k] = (double)(weights[k] / total);
    set.theta.assign((size_t)set.nodes, 0.0f);
    set.potential.assign((size_t)set.nodes, 0);
}

void refresh_theta(PatchSet &set, const Model &m, const Shared &sh) {
    std::vector<float> centres((size_t)set.nodes * 3), spiral((size_t)set.nodes * 3);
    for (size_t k = 0; k < set.patches.size(); ++k) {
        const Patch &p = set.patches[k];
        const int64_t cols = p.W - 1;
        for (size_t o = 0; o < p.cells.size(); ++o) {
            const int64_t linear = p.cells[o];
            lookup(p, (float)(linear / cols) + 0.5f, (float)(linear % cols) + 0.5f,
                centres.data() + (size_t)(set.start[k] + (int64_t)o) * 3);
        }
    }
    slice_to_spiral(m, sh, centres.data(), (size_t)set.nodes, spiral.data());
    for (int64_t n = 0; n < set.nodes; ++n) {
        const float t = std::atan2(spiral[(size_t)n * 3 + 1], spiral[(size_t)n * 3 + 2]);
        set.theta[(size_t)n] = t < 0.0f ? t + TWO_PI : t;
    }
    for (size_t k = 0; k < set.patches.size(); ++k) {
        const Patch &p = set.patches[k];
        const int64_t base = set.start[k];
        for (size_t position = 0; position < p.preorder.size(); ++position) {
            const int64_t node = p.preorder[position], parent = p.parent[position];
            set.potential[(size_t)(base + node)] = parent < 0
                ? 0
                : set.potential[(size_t)(base + parent)] +
                    crossing(set.theta[(size_t)(base + node)] - set.theta[(size_t)(base + parent)]);
        }
    }
}

std::vector<size_t> non_liftable_patches(const PatchSet &set) {
    std::vector<size_t> bad;
    const int offsets[4][2] = {{0, 1}, {1, -1}, {1, 0}, {1, 1}};
    for (size_t k = 0; k < set.patches.size(); ++k) {
        const Patch &p = set.patches[k];
        const int64_t cols = p.W - 1, base = set.start[k];
        std::vector<int64_t> dense((size_t)((p.H - 1) * cols), -1);
        for (size_t o = 0; o < p.cells.size(); ++o)
            dense[(size_t)p.cells[o]] = (int64_t)o;
        bool consistent = true;
        for (size_t o = 0; o < p.cells.size() && consistent; ++o) {
            const int64_t row = p.cells[o] / cols, column = p.cells[o] % cols;
            for (const auto &d : offsets) {
                const int64_t r = row + d[0], c = column + d[1];
                if (r < 0 || c < 0 || r >= p.H - 1 || c >= cols)
                    continue;
                const int64_t n = dense[(size_t)(r * cols + c)];
                if (n < 0)
                    continue;
                const size_t a = (size_t)(base + (int64_t)o), b = (size_t)(base + n);
                if (set.potential[b] - set.potential[a] - crossing(set.theta[b] - set.theta[a]) !=
                    0) {
                    consistent = false;
                    break;
                }
            }
        }
        if (!consistent)
            bad.push_back(k);
    }
    return bad;
}

size_t enforce_liftability(
    PatchSet &set, const Model &m, const Shared &sh, const PatchSettings &ps) {
    size_t dropped = 0;
    for (;;) {
        const std::vector<size_t> bad = non_liftable_patches(set);
        if (bad.empty())
            return dropped;
        for (size_t i = bad.size(); i-- > 0;) {
            std::printf("non-liftable patch %s has theta cycle inconsistencies; excluding it from "
                        "this fit\n",
                set.patches[bad[i]].name.c_str());
            set.rejected.push_back(set.patches[bad[i]].name);
            set.patches.erase(set.patches.begin() + (long)bad[i]);
        }
        dropped += bad.size();
        reindex_patches(set, ps);
        refresh_theta(set, m, sh);
        set.fresh = true;
        set.refreshed = -1;
    }
}

bool refresh_if_due(
    PatchSet &set, long iteration, const Model &m, const Shared &sh, const PatchSettings &ps) {
    if (set.empty())
        return false;
    if (set.fresh) {
        set.refreshed = iteration;
        set.fresh = false;
        return false;
    }
    if (set.refreshed >= 0 && iteration - set.refreshed < ps.refresh_interval)
        return false;
    refresh_theta(set, m, sh);
    set.refreshed = iteration;
    enforce_liftability(set, m, sh, ps);
    return true;
}

PatchSample draw_patches(const PatchSet &set, int rows, int cap, std::mt19937_64 &rng) {
    PatchSample s;
    s.rows = rows;
    s.cap = cap;
    s.counts.assign((size_t)rows, 0);
    s.points.assign((size_t)rows * cap * 3, 0.0f);
    s.nodes.assign((size_t)rows * cap, 0);
    if (set.empty() || rows <= 0)
        return s;
    std::discrete_distribution<int> pick(set.probability.begin(), set.probability.end());
    std::vector<int> chosen((size_t)rows);
    for (int r = 0; r < rows; ++r)
        chosen[(size_t)r] = pick(rng);
    const uint64_t seed = rng() >> 1;
    for (int r = 0; r < rows; ++r) {
        const Patch &p = set.patches[(size_t)chosen[(size_t)r]];
        const int64_t cols = p.W - 1;
        std::mt19937_64 gen(splitmix64(seed + (uint64_t)r));
        const int valid = (int)p.cells.size(), count = std::min(cap, valid);
        s.counts[(size_t)r] = count;
        std::vector<int> selected;
        selected.reserve((size_t)count);
        if (count == valid) {
            selected.resize((size_t)valid);
            std::iota(selected.begin(), selected.end(), 0);
        } else {
            std::unordered_set<int> taken;
            taken.reserve((size_t)count * 2);
            for (int candidate = valid - count; candidate < valid; ++candidate) {
                const int draw = std::uniform_int_distribution<int>(0, candidate)(gen);
                if (taken.insert(draw).second)
                    selected.push_back(draw);
                else {
                    taken.insert(candidate);
                    selected.push_back(candidate);
                }
            }
        }
        for (int end = count; end > 1; --end)
            std::swap(selected[(size_t)end - 1],
                selected[(size_t)std::uniform_int_distribution<int>(0, end - 1)(gen)]);
        for (int k = 0; k < count; ++k) {
            const int64_t linear = p.cells[(size_t)selected[(size_t)k]];
            const float fi = (float)(linear / cols) + std::generate_canonical<float, 24>(gen);
            const float fj = (float)(linear % cols) + std::generate_canonical<float, 24>(gen);
            lookup(p, fi, fj, s.points.data() + ((size_t)r * cap + (size_t)k) * 3);
            s.nodes[(size_t)r * cap + (size_t)k] =
                set.start[(size_t)chosen[(size_t)r]] + selected[(size_t)k];
        }
    }
    return s;
}

double patch_loss(const Model &m, const Shared &sh, const PatchSet &set, const PatchSample &s,
    const PatchSettings &ps, float weight, Grads &g) {
    if (s.rows <= 0)
        return 0;
    std::vector<float> points;
    std::vector<int64_t> nodes;
    std::vector<int32_t> row;
    for (int r = 0; r < s.rows; ++r)
        for (int k = 0; k < s.counts[(size_t)r]; ++k) {
            const size_t at = (size_t)r * s.cap + (size_t)k;
            points.insert(points.end(), s.points.begin() + (long)(3 * at),
                s.points.begin() + (long)(3 * at + 3));
            nodes.push_back(s.nodes[at]);
            row.push_back(r);
        }
    const size_t n = nodes.size();
    std::vector<float> out(n * 3), grad(n * 3, 0.0f);
    slice_to_spiral(m, sh, points.data(), n, out.data());
    std::vector<float> theta(n), radius(n), raw(n), unwrapped(n);
    std::vector<uint8_t> guarded(n);
    for (size_t i = 0; i < n; ++i) {
        const float y = out[i * 3 + 1], x0 = out[i * 3 + 2];
        guarded[i] = std::fabs(x0) < 1e-10f;
        const float x = guarded[i] ? 1e-10f : x0;
        const float t = std::atan2(y, x);
        theta[i] = t - TWO_PI * std::floor(t / TWO_PI);
        radius[i] = std::sqrt(y * y + x * x);
        raw[i] = radius[i] - theta[i] / TWO_PI * sh.dr;
        const int steps =
            set.potential[(size_t)nodes[i]] + crossing(theta[i] - set.theta[(size_t)nodes[i]]);
        unwrapped[i] = std::max(raw[i], 0.0f) + (float)steps * sh.dr;
    }
    const int R = s.rows;
    std::vector<double> mean((size_t)R, 0.0);
    for (size_t i = 0; i < n; ++i)
        mean[(size_t)row[i]] += unwrapped[i];
    for (int r = 0; r < R; ++r)
        mean[(size_t)r] /= std::max(s.counts[(size_t)r], 1);
    const float margin = sh.dr * ps.margin, p = ps.p;
    std::vector<double> sums((size_t)R, 0.0), pull((size_t)R, 0.0);
    std::vector<float> hinged(n), sign(n);
    for (size_t i = 0; i < n; ++i) {
        const float d = unwrapped[i] - (float)mean[(size_t)row[i]];
        sign[i] = d > 0 ? 1.0f : (d < 0 ? -1.0f : 0.0f);
        hinged[i] = std::max(std::fabs(d) - margin, 0.0f);
        sums[(size_t)row[i]] += std::pow((double)hinged[i] + 1e-5, (double)p);
    }
    double loss = 0;
    std::vector<double> scale((size_t)R, 0.0);
    for (int r = 0; r < R; ++r) {
        const double count = std::max(s.counts[(size_t)r], 1);
        const double value = sums[(size_t)r] / count;
        loss += std::pow(value, 1.0 / p);
        scale[(size_t)r] = std::pow(value, 1.0 / p - 1.0) / p / count / R * weight;
    }
    loss /= R;
    std::vector<double> local(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (hinged[i] <= 0)
            continue;
        local[i] = p * std::pow((double)hinged[i] + 1e-5, (double)p - 1.0) * sign[i];
        pull[(size_t)row[i]] += local[i];
    }
    double gdr = 0;
    for (size_t i = 0; i < n; ++i) {
        const int r = row[i];
        const double gu =
            scale[(size_t)r] * (local[i] - pull[(size_t)r] / std::max(s.counts[(size_t)r], 1));
        if (raw[i] < 0.0f || gu == 0.0)
            continue;
        const float gs = (float)gu, gtheta = -gs * sh.dr / TWO_PI;
        gdr += -gu * theta[i] / TWO_PI;
        const float y = out[i * 3 + 1], x = guarded[i] ? 1e-10f : out[i * 3 + 2];
        const float r2 = y * y + x * x;
        grad[i * 3 + 1] = gtheta * (x / r2) + gs * (y / radius[i]);
        if (!guarded[i])
            grad[i * 3 + 2] = gtheta * (-y / r2) + gs * (x / radius[i]);
    }
    g.dr += gdr;
    slice_to_spiral_backward(m, sh, points.data(), n, grad.data(), nullptr, g);
    return loss;
}

}
