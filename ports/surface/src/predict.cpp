#include "predict.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace hx {

static double pairwise(const double *a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i)
            res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j)
            r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j)
                r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i)
            res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

std::vector<std::vector<int64_t>> window_steps(
    const std::array<int64_t, 3> &size, int patch, double step, Rule rule) {
    std::vector<std::vector<int64_t>> steps(3);
    for (int d = 0; d < 3; ++d) {
        const int64_t s = size[d], tile = patch;
        if (s < tile)
            throw std::runtime_error("the volume must be at least one patch in every axis");
        const double target = rule == Rule::NNUNET
            ? tile * step
            : (double)std::max<int64_t>(1, step != 0.0 ? (int64_t)(tile * step) : tile);
        const int64_t count = (int64_t)std::ceil((double)(s - tile) / target) + 1;
        const double actual = count > 1 ? (double)(s - tile) / (double)(count - 1) : 99999999999.0;
        for (int64_t i = 0; i < count; ++i) {
            int64_t v = (int64_t)std::nearbyint(actual * (double)i);
            if (rule == Rule::VESUVIUS)
                v = std::min(v, s - tile);
            steps[d].push_back(v);
        }
    }
    return steps;
}

std::vector<float> gaussian_importance(int patch, Rule rule) {
    const double sigma = patch / 8.0;
    const int radius = (int)(4.0 * sigma + 0.5);
    std::vector<double> k(2 * radius + 1);
    const double scale = -0.5 / (sigma * sigma);
    for (int i = -radius; i <= radius; ++i)
        k[i + radius] = std::exp(scale * (double)((int64_t)i * i));
    const double total = pairwise(k.data(), k.size());
    for (double &v : k)
        v = v / total;
    const int c = patch / 2;
    const size_t n = (size_t)patch * patch * patch;
    std::vector<double> g(n);
    for (int z = 0; z < patch; ++z)
        for (int y = 0; y < patch; ++y) {
            const double zy = k[radius + z - c] * k[radius + y - c];
            for (int x = 0; x < patch; ++x)
                g[((size_t)z * patch + y) * patch + x] = zy * k[radius + x - c];
        }
    const double peak = *std::max_element(g.begin(), g.end());
    std::vector<float> out(n);
    if (rule == Rule::NNUNET) {
        const double divisor = peak / 10;
        std::vector<_Float16> h(n);
        float least = INFINITY;
        for (size_t i = 0; i < n; ++i) {
            h[i] = (_Float16)(g[i] / divisor);
            if ((float)h[i] != 0.0f)
                least = std::min(least, (float)h[i]);
        }
        for (size_t i = 0; i < n; ++i)
            out[i] = (float)h[i] == 0.0f ? least : (float)h[i];
        return out;
    }
    const double divisor = std::max(peak, 1e-12);
    for (size_t i = 0; i < n; ++i)
        out[i] = (float)std::max(g[i] / divisor, 0.0);
    return out;
}

static mx::array mirrored(const mx::array &a, const std::vector<int> &axes) {
    mx::array r = a;
    for (int axis : axes) {
        const int n = r.shape(axis);
        r = mx::take(r, mx::arange(n - 1, -1, -1, mx::int32), axis);
    }
    return r;
}

static double setting(const char *name, double fallback) {
    const char *v = std::getenv(name);
    return v && *v ? std::atof(v) : fallback;
}

class Spill {
  public:
    Spill() {
        const char *dir = std::getenv("TMPDIR");
        std::string path = dir && *dir ? dir : "/tmp";
        if (path.back() != '/')
            path += '/';
        path += "hx-surface-XXXXXX";
        fd_ = mkstemp(path.data());
        if (fd_ < 0)
            throw std::runtime_error("cannot create a scratch file at " + path);
        unlink(path.c_str());
    }
    ~Spill() {
        if (fd_ >= 0)
            close(fd_);
    }
    Spill(const Spill &) = delete;
    Spill &operator=(const Spill &) = delete;

    void write(const char *p, size_t left, off_t at) {
        while (left) {
            const ssize_t w = pwrite(fd_, p, std::min(left, (size_t)1 << 30), at);
            if (w < 0 && errno == EINTR)
                continue;
            if (w <= 0)
                throw std::runtime_error("the scratch file could not be written");
            p += w;
            at += w;
            left -= (size_t)w;
        }
    }

    void read(char *p, size_t left, off_t at) {
        while (left) {
            const ssize_t r = pread(fd_, p, std::min(left, (size_t)1 << 30), at);
            if (r < 0 && errno == EINTR)
                continue;
            if (r <= 0)
                throw std::runtime_error("the scratch file could not be read");
            p += r;
            at += r;
            left -= (size_t)r;
        }
    }

  private:
    int fd_ = -1;
};

static mx::array spilled_mean(
    SurfaceNet &net, const mx::array &x, const std::vector<std::vector<int>> &order) {
    Spill spill;
    mx::Shape shape;
    size_t plane = 0, volume = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        auto flipped = mirrored(x, order[i]);
        mx::eval(flipped);
        auto out = mx::contiguous(mirrored(net.forward(mx::contiguous(flipped)), order[i]));
        mx::eval(out);
        if (i == 0) {
            shape = out.shape();
            volume = out.nbytes();
            plane = volume / (size_t)shape[0];
        }
        spill.write(out.data<char>(), volume, (off_t)(i * volume));
    }
    const int Z = shape[0];
    const int step = std::max(
        1, (int)(setting("HX_SURFACE_SLAB_MB", 128) * 1e6 / (double)(plane * order.size())));
    std::vector<mx::array> parts;
    for (int a = 0; a < Z; a += step) {
        const int b = std::min(Z, a + step);
        mx::Shape part_shape = shape;
        part_shape[0] = b - a;
        std::vector<mx::array> outs;
        for (size_t i = 0; i < order.size(); ++i) {
            auto data = mx::allocator::malloc((size_t)(b - a) * plane);
            auto buffer = data.buffer();
            spill.read(static_cast<char *>(buffer.raw_ptr()), (size_t)(b - a) * plane,
                (off_t)(i * volume + (size_t)a * plane));
            outs.push_back(mx::array(std::move(data), part_shape, mx::float32));
        }
        auto part = mx::mean(mx::stack(outs, 0), 0);
        mx::eval(part);
        parts.push_back(part);
    }
    auto total = parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
    mx::eval(total);
    return total;
}

mx::array mirrored_forward(SurfaceNet &net, const mx::array &x) {
    if (net.rule() == Rule::NNUNET) {
        const std::vector<std::vector<int>> combos = {
            {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}};
        mx::array total = net.forward(x);
        for (const auto &axes : combos) {
            auto flipped = mirrored(x, axes);
            mx::eval(flipped);
            total = total + mirrored(net.forward(mx::contiguous(flipped)), axes);
            mx::eval(total);
        }
        return total / 8.0f;
    }
    const std::vector<std::vector<int>> order = {
        {}, {2}, {1}, {0}, {2, 1}, {2, 0}, {1, 0}, {2, 1, 0}};
    const double spill = setting("HX_SURFACE_MIRROR_MB", 256) * 1e6;
    if ((double)x.size() * 2 * sizeof(float) * order.size() > spill)
        return spilled_mean(net, x, order);
    std::vector<mx::array> outs;
    for (const auto &axes : order) {
        auto flipped = mirrored(x, axes);
        mx::eval(flipped);
        auto out = mirrored(net.forward(mx::contiguous(flipped)), axes);
        mx::eval(out);
        outs.push_back(out);
    }
    return mx::mean(mx::stack(outs, 0), 0);
}

static std::vector<float> predict_patch(
    SurfaceNet &net, const std::vector<float> &x, int P, bool mirror, const mx::array &weight) {
    auto xm = mx::array(x.data(), {P, P, P, 1}, mx::float32);
    mx::array total = mirror ? mirrored_forward(net, xm) : net.forward(xm);
    total = mx::transpose(mx::astype(total, mx::float32), {3, 0, 1, 2}) * weight;
    total = mx::contiguous(total);
    mx::eval(total);
    const float *p = total.data<float>();
    return std::vector<float>(p, p + total.size());
}

void predict_volume(SurfaceNet &net, const Reader &read, const std::array<int64_t, 3> &shape,
    double step, bool mirror, const Writer &write, bool verbose) {
    const Rule rule = net.rule();
    const int P = net.patch();
    const bool half = rule == Rule::NNUNET;
    const auto steps = window_steps(shape, P, step, rule);
    const auto gauss = gaussian_importance(P, rule);
    std::vector<_Float16> gauss_half(gauss.size());
    for (size_t i = 0; i < gauss.size(); ++i)
        gauss_half[i] = (_Float16)gauss[i];
    const auto weight = mx::array(gauss.data(), {P, P, P}, mx::float32);
    const int64_t Y = shape[1], X = shape[2], plane = Y * X, cube = (int64_t)P * P * P;
    std::vector<_Float16> hl, hc;
    std::vector<float> fl, fc;
    int64_t window_z = 0, depth = 0;
    bool have = false;
    auto finish = [&](int64_t rows) {
        std::vector<float> out((size_t)(2 * rows * plane));
        for (int c = 0; c < 2; ++c)
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t p = 0; p < plane; ++p) {
                    const size_t li = (size_t)((c * depth + r) * plane + p),
                                 ci = (size_t)(r * plane + p);
                    const size_t oi = (size_t)((c * rows + r) * plane + p);
                    if (half) {
                        out[oi] = (float)(_Float16)((float)hl[li] / (float)hc[ci]);
                    } else {
                        const float w = fc[ci];
                        out[oi] = w > 0.0f ? fl[li] / (w + 1e-8f) : 0.0f;
                    }
                }
        return out;
    };
    const size_t total = steps[0].size() * steps[1].size() * steps[2].size();
    size_t n = 0;
    const auto started = std::chrono::steady_clock::now();
    std::vector<uint8_t> ct((size_t)cube);
    std::vector<float> x((size_t)cube);
    for (int64_t sz : steps[0]) {
        if (!have || sz + P > window_z + depth) {
            const int64_t fresh = std::min(shape[0], sz + 2 * P) - sz;
            std::vector<_Float16> nhl, nhc;
            std::vector<float> nfl, nfc;
            if (half) {
                nhl.assign((size_t)(2 * fresh * plane), (_Float16)0.0f);
                nhc.assign((size_t)(fresh * plane), (_Float16)0.0f);
            } else {
                nfl.assign((size_t)(2 * fresh * plane), 0.0f);
                nfc.assign((size_t)(fresh * plane), 0.0f);
            }
            if (have) {
                const int64_t keep = sz - window_z;
                write(window_z, keep, finish(keep));
                const int64_t carry = depth - keep;
                for (int c = 0; c < 2; ++c)
                    for (int64_t r = 0; r < carry; ++r)
                        for (int64_t p = 0; p < plane; ++p) {
                            const size_t to = (size_t)((c * fresh + r) * plane + p);
                            const size_t from = (size_t)((c * depth + keep + r) * plane + p);
                            if (half)
                                nhl[to] = hl[from];
                            else
                                nfl[to] = fl[from];
                        }
                for (int64_t r = 0; r < carry; ++r)
                    for (int64_t p = 0; p < plane; ++p) {
                        if (half)
                            nhc[(size_t)(r * plane + p)] = hc[(size_t)((keep + r) * plane + p)];
                        else
                            nfc[(size_t)(r * plane + p)] = fc[(size_t)((keep + r) * plane + p)];
                    }
            }
            hl.swap(nhl);
            hc.swap(nhc);
            fl.swap(nfl);
            fc.swap(nfc);
            window_z = sz;
            depth = fresh;
            have = true;
        }
        for (int64_t sy : steps[1])
            for (int64_t sx : steps[2]) {
                read(Box{{{sz, sz + P}, {sy, sy + P}, {sx, sx + P}}}, ct.data());
                ++n;
                if (rule == Rule::VESUVIUS) {
                    const auto range = std::minmax_element(ct.begin(), ct.end());
                    if (*range.first == *range.second)
                        continue;
                }
                net.normalise(ct.data(), ct.size(), x.data());
                const auto pred = predict_patch(net, x, P, mirror, weight);
                const int64_t z0 = sz - window_z;
                for (int c = 0; c < 2; ++c)
                    for (int64_t z = 0; z < P; ++z)
                        for (int64_t y = 0; y < P; ++y) {
                            const size_t li =
                                (size_t)(((c * depth + z0 + z) * Y + sy + y) * X + sx);
                            const size_t pi = (size_t)(((c * P + z) * P + y) * P);
                            if (half) {
                                for (int64_t xx = 0; xx < P; ++xx)
                                    hl[li + xx] = (_Float16)((float)hl[li + xx] + pred[pi + xx]);
                            } else {
                                for (int64_t xx = 0; xx < P; ++xx)
                                    fl[li + xx] += pred[pi + xx];
                            }
                        }
                for (int64_t z = 0; z < P; ++z)
                    for (int64_t y = 0; y < P; ++y) {
                        const size_t ci = (size_t)(((z0 + z) * Y + sy + y) * X + sx);
                        const size_t gi = (size_t)((z * P + y) * P);
                        if (half) {
                            for (int64_t xx = 0; xx < P; ++xx)
                                hc[ci + xx] =
                                    (_Float16)((float)hc[ci + xx] + (float)gauss_half[gi + xx]);
                        } else {
                            for (int64_t xx = 0; xx < P; ++xx)
                                fc[ci + xx] += gauss[gi + xx];
                        }
                    }
                if (verbose && (n % 8 == 0 || n == total)) {
                    const double each =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                            .count() /
                        (double)n;
                    std::fprintf(stderr, "patch %zu/%zu, %.2f s each, %.1f min left\n", n, total,
                        each, each * (double)(total - n) / 60.0);
                }
            }
    }
    write(window_z, shape[0] - window_z, finish(shape[0] - window_z));
}

uint8_t surface_probability(float background, float surface) {
    const float p = 1.0f / (1.0f + std::exp(background - surface));
    return (uint8_t)std::nearbyint(p * 255.0f);
}

}
