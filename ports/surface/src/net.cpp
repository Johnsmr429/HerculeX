#include "net.h"

#include "json.h"
#include "volume.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace hx {

using Arrays = std::vector<mx::array>;

static const char *STATS_PARTIAL = R"(
    const uint c = thread_position_in_threadgroup.x;
    const uint lane = thread_position_in_threadgroup.y;
    const uint group = threadgroup_position_in_grid.x;
    const uint groups = threadgroups_per_grid.x;
    const uint voxels = dims[0];
    const uint per = (voxels + groups - 1) / groups;
    const uint start = group * per;
    const uint stop = min(voxels, start + per);
    float n = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (uint v = start + lane; v < stop; v += R) {
        const float value = static_cast<float>(x[(size_t)v * C + c]);
        n += 1.0f;
        const float delta = value - mean;
        mean += delta / n;
        m2 += delta * (value - mean);
    }
    threadgroup float tn[R * C];
    threadgroup float tmean[R * C];
    threadgroup float tm2[R * C];
    tn[lane * C + c] = n; tmean[lane * C + c] = mean; tm2[lane * C + c] = m2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float an = tn[c], amean = tmean[c], am2 = tm2[c];
        for (uint l = 1; l < R; ++l) {
            const float bn = tn[l * C + c];
            if (bn == 0.0f) continue;
            const float bmean = tmean[l * C + c], bm2 = tm2[l * C + c];
            const float total = an + bn;
            const float delta = bmean - amean;
            amean += delta * (bn / total);
            am2 += bm2 + delta * delta * (an * bn / total);
            an = total;
        }
        const size_t at = ((size_t)group * C + c) * 3;
        partial[at] = an; partial[at + 1] = amean; partial[at + 2] = am2;
    }
    )";

static const char *STATS_MERGE = R"(
    const uint c = thread_position_in_grid.x;
    if (c >= C) return;
    float an = 0.0f, amean = 0.0f, am2 = 0.0f;
    for (uint g = 0; g < G; ++g) {
        const size_t at = ((size_t)g * C + c) * 3;
        const float bn = partial[at];
        if (bn == 0.0f) continue;
        const float bmean = partial[at + 1], bm2 = partial[at + 2];
        const float total = an + bn;
        const float delta = bmean - amean;
        amean += delta * (bn / total);
        am2 += bm2 + delta * delta * (an * bn / total);
        an = total;
    }
    mean[c] = amean;
    var[c] = am2 / an;
    )";

static const char *STATS_STEP = R"(
    const uint c = thread_position_in_threadgroup.x;
    const uint lane = thread_position_in_threadgroup.y;
    const uint group = threadgroup_position_in_grid.x;
    const uint groups = threadgroups_per_grid.x;
    const uint voxels = dims[0];
    const uint first = dims[1];
    const uint last = dims[2];
    const uint per = (voxels + groups - 1) / groups;
    const uint start = group * per;
    const uint stop = min(min(voxels, start + per), last);
    const size_t at = (((size_t)group * R + lane) * C + c) * 3;
    float n = state[at], mean = state[at + 1], m2 = state[at + 2];
    uint v = start + lane;
    if (v < first) v += (first - v + R - 1) / R * R;
    for (; v < stop; v += R) {
        const float value = static_cast<float>(x[(size_t)(v - first) * C + c]);
        n += 1.0f;
        const float delta = value - mean;
        mean += delta / n;
        m2 += delta * (value - mean);
    }
    next[at] = n; next[at + 1] = mean; next[at + 2] = m2;
    )";

static const char *STATS_LANES = R"(
    const uint c = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const size_t base = (size_t)group * R;
    size_t at = (base * C + c) * 3;
    float an = state[at], amean = state[at + 1], am2 = state[at + 2];
    for (uint l = 1; l < R; ++l) {
        at = ((base + l) * C + c) * 3;
        const float bn = state[at];
        if (bn == 0.0f) continue;
        const float bmean = state[at + 1], bm2 = state[at + 2];
        const float total = an + bn;
        const float delta = bmean - amean;
        amean += delta * (bn / total);
        am2 += bm2 + delta * delta * (an * bn / total);
        an = total;
    }
    const size_t out = ((size_t)group * C + c) * 3;
    partial[out] = an; partial[out + 1] = amean; partial[out + 2] = am2;
    )";

static const char *NORM_SCSE_RESIDUAL = R"(
    const uint lane = thread_index_in_simdgroup;
    const uint v = thread_position_in_grid.x / 32;
    if (v >= dims[0]) return;
    const size_t base = (size_t)v * C;
    float h[C / 32];
    float dot = 0.0f;
    for (uint k = 0; k < C / 32; ++k) {
        const uint c = lane + 32 * k;
        h[k] = (y[base + c] - mean[c]) * rsqrt(var[c] + 1e-5f) * gamma[c] + beta[c];
        dot = fma(h[k], sse[c], dot);
    }
    const float s = 1.0f / (1.0f + exp(-(simd_sum(dot) + sse_b[0])));
    for (uint k = 0; k < C / 32; ++k) {
        const uint c = lane + 32 * k;
        const float o = h[k] * (gate[c] + s) + r[base + c];
        out[base + c] = max(o, 0.01f * o);
    }
    )";

static const char *UPSAMPLE = R"(
    const uint t = thread_position_in_grid.x;
    const uint Z = dims[0], Y = dims[1], X = dims[2];
    const uint quads = C / 4;
    const uint q = t % quads;
    uint v = t / quads;
    const uint ox = v % (2 * X); v /= 2 * X;
    const uint oy = v % (2 * Y); const uint oz = v / (2 * Y);
    if (oz >= 2 * Z) return;
    const uint src = ((oz / 2) * Y + oy / 2) * X + ox / 2;
    const uint sub = ((oz % 2) * 2 + oy % 2) * 2 + ox % 2;
    const float4 value = reinterpret_cast<const device float4 *>(m)[((size_t)src * 8 + sub) * quads + q];
    reinterpret_cast<device float4 *>(out)[t] = value + reinterpret_cast<const device float4 *>(bias)[q];
    )";

static const char *STEM = R"(
    const uint v = thread_position_in_grid.x;
    const int Z = dims[0], Y = dims[1], X = dims[2];
    threadgroup float tw[C * 27];
    for (uint i = thread_position_in_threadgroup.x; i < C * 27; i += threads_per_threadgroup.x) tw[i] = w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (v >= (uint)(Z * Y * X)) return;
    const int z = v / (Y * X), yy = (v / X) % Y, xx = v % X;
    float taps[27];
    for (int dz = 0; dz < 3; ++dz)
        for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) {
                const int iz = z + dz - 1, iy = yy + dy - 1, ix = xx + dx - 1;
                const bool inside = iz >= 0 && iz < Z && iy >= 0 && iy < Y && ix >= 0 && ix < X;
                taps[(dz * 3 + dy) * 3 + dx] = inside ? x[((size_t)iz * Y + iy) * X + ix] : 0.0f;
            }
    device float4 *out = reinterpret_cast<device float4 *>(y) + (size_t)v * (C / 4);
    for (int q = 0; q < C / 4; ++q) {
        float4 acc = 0.0f;
        for (int t = 0; t < 27; ++t) {
            acc.x = fma(taps[t], tw[(q * 4 + 0) * 27 + t], acc.x);
            acc.y = fma(taps[t], tw[(q * 4 + 1) * 27 + t], acc.y);
            acc.z = fma(taps[t], tw[(q * 4 + 2) * 27 + t], acc.z);
            acc.w = fma(taps[t], tw[(q * 4 + 3) * 27 + t], acc.w);
        }
        out[q] = acc + float4(b[q * 4], b[q * 4 + 1], b[q * 4 + 2], b[q * 4 + 3]);
    }
    )";

static mx::array leaky(const mx::array &x) { return mx::maximum(x, 0.01f * x); }

static Arrays norm_act_impl(const Arrays &a) {
    return {leaky((a[0] - a[1]) * mx::rsqrt(a[2] + 1e-5f) * a[3] + a[4])};
}

static Arrays norm_impl(const Arrays &a) {
    return {(a[0] - a[1]) * mx::rsqrt(a[2] + 1e-5f) * a[3] + a[4]};
}

static Arrays norm_residual_impl(const Arrays &a) {
    return {leaky((a[0] - a[1]) * mx::rsqrt(a[2] + 1e-5f) * a[3] + a[4] + a[5])};
}

struct Block {
    int stride = 1;
    bool pool = false, scse = false;
    std::string c1, g1, b1, c2, g2, b2, sc, sg, sb, sse, sseb;
    mx::array gate = mx::array(0.0f);
};

struct Decoder {
    std::string up, upb, g, b;
    mx::array wa = mx::array(0.0f), wb = mx::array(0.0f);
};

struct Op {
    bool endstage = false;
    Block block;
};

struct NetState {
    std::map<std::string, mx::array> w;
    std::string stem_w, stem_g, stem_b, head_w, head_b;
    std::vector<Op> ops;
    std::vector<Decoder> decoders;
    mx::fast::CustomKernelFunction partial, merge, scse, upsample, stem, step, lanes;
    std::function<Arrays(const Arrays &)> norm_act, norm, norm_residual;
    std::vector<int> stage_channels;
    size_t cache = 0;
};

static const mx::array &W(NetState &n, const std::string &k) { return n.w.at(k); }

static std::pair<mx::array, mx::array> stats(NetState &n, const mx::array &x) {
    const int channels = x.shape().back();
    const int voxels = (int)(x.size() / channels);
    const int lanes = std::max(1, std::min(32, 1024 / channels));
    const int groups = std::min(1024, std::max(1, (voxels + lanes * 64 - 1) / (lanes * 64)));
    auto dims = mx::array({(uint32_t)voxels}, mx::uint32);
    auto p = n.partial({x, dims}, {{groups * channels * 3}}, {mx::float32},
        {groups * channels, lanes, 1}, {channels, lanes, 1}, {{"C", channels}, {"R", lanes}},
        std::nullopt, false, {})[0];
    auto mv = n.merge({p}, {{channels}, {channels}}, {mx::float32, mx::float32}, {channels, 1, 1},
        {std::min(channels, 256), 1, 1}, {{"C", channels}, {"G", groups}}, std::nullopt, false, {});
    return {mx::reshape(mv[0], {1, 1, 1, channels}), mx::reshape(mv[1], {1, 1, 1, channels})};
}

static mx::array norm_of(
    NetState &n, const mx::array &y, const std::string &g, const std::string &b, bool act) {
    auto [mean, var] = stats(n, y);
    return (act ? n.norm_act : n.norm)({y, mean, var, W(n, g), W(n, b)})[0];
}

static mx::array conv(const mx::array &x, const mx::array &w, int stride) {
    const int k = w.shape()[1];
    auto y = mx::conv3d(mx::expand_dims(x, 0), w, {stride, stride, stride}, {k / 2, k / 2, k / 2});
    return mx::squeeze(y, 0);
}

static mx::array stem_conv(NetState &n, const mx::array &x, const mx::array &w) {
    const int z = x.shape()[0], y = x.shape()[1], xx = x.shape()[2], channels = w.shape()[0];
    auto dims = mx::array({z, y, xx}, mx::int32);
    auto zero = mx::zeros({channels}, mx::float32);
    return n.stem({mx::reshape(x, {z, y, xx}), mx::reshape(w, {channels, 27}), zero, dims},
        {{z, y, xx, channels}}, {mx::float32}, {z * y * xx, 1, 1}, {256, 1, 1}, {{"C", channels}},
        std::nullopt, false, {})[0];
}

static mx::array block_forward(NetState &n, const mx::array &x, Block &b) {
    auto y = conv(x, W(n, b.c1), b.stride);
    mx::eval(y);
    auto h = norm_of(n, y, b.g1, b.b1, true);
    y = mx::array(0.0f);
    mx::eval(h);
    y = conv(h, W(n, b.c2), 1);
    mx::eval(y);
    h = mx::array(0.0f);
    auto r = x;
    if (b.pool) {
        const auto &s = r.shape();
        r = mx::mean(mx::reshape(r, {s[0] / 2, 2, s[1] / 2, 2, s[2] / 2, 2, s[3]}), {1, 3, 5});
    }
    if (!b.sc.empty()) {
        const auto &wsc = W(n, b.sc);
        r = mx::matmul(r, mx::transpose(mx::reshape(wsc, {wsc.shape()[0], wsc.shape()[4]})));
        mx::eval(r);
        r = norm_of(n, r, b.sg, b.sb, false);
    }
    mx::eval(r);
    auto [mean, var] = stats(n, y);
    mx::array out(0.0f);
    if (b.scse) {
        const int channels = y.shape().back();
        const int voxels = (int)(y.size() / channels);
        auto dims = mx::array({(uint32_t)voxels}, mx::uint32);
        out = n.scse({y, mx::reshape(mean, {-1}), mx::reshape(var, {-1}), W(n, b.g2), W(n, b.b2),
                         mx::reshape(b.gate, {-1}), mx::reshape(W(n, b.sse), {-1}),
                         mx::reshape(W(n, b.sseb), {-1}), r, dims},
            {y.shape()}, {mx::float32}, {voxels * 32, 1, 1}, {256, 1, 1}, {{"C", channels}},
            std::nullopt, false, {})[0];
    } else {
        out = n.norm_residual({y, mean, var, W(n, b.g2), W(n, b.b2), r})[0];
    }
    mx::eval(out);
    return out;
}

static mx::array transposed(
    NetState &n, const mx::array &x, const mx::array &w, const mx::array &bias) {
    const auto &s = x.shape();
    const int channels = w.shape()[1] / 8;
    auto m = mx::matmul(mx::reshape(x, {-1, s[3]}), w);
    auto dims = mx::array({(uint32_t)s[0], (uint32_t)s[1], (uint32_t)s[2]}, mx::uint32);
    return n.upsample({m, bias, dims}, {{2 * s[0], 2 * s[1], 2 * s[2], channels}}, {mx::float32},
        {8 * s[0] * s[1] * s[2] * channels / 4, 1, 1}, {256, 1, 1}, {{"C", channels}}, std::nullopt,
        false, {})[0];
}

static double setting(const char *name, double fallback) {
    const char *v = std::getenv(name);
    return v && *v ? std::atof(v) : fallback;
}

static void trace(const char *phase, int level = -1) {
    static const bool on = setting("HX_SURFACE_TRACE", 0) > 0;
    static const auto started = std::chrono::steady_clock::now();
    if (on)
        std::fprintf(stderr, "surface %8.3f s %s %d active %.3f cache %.3f peak %.3f GB\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
            phase, level, (double)mx::get_active_memory() / 1e9,
            (double)mx::get_cache_memory() / 1e9, (double)mx::get_peak_memory() / 1e9);
}

static const char *COPY = R"(
    const uint i = thread_position_in_grid.x;
    if (i < dims[0]) y[i] = x[i];
    )";

static mx::array tight(const mx::array &a) {
    if (a.buffer_size() <= a.nbytes() + ((size_t)1 << 20))
        return a;
    static auto kernel = mx::fast::metal_kernel("hx_tight_copy", {"x", "dims"}, {"y"}, COPY);
    const int count = (int)a.size();
    auto dims = mx::array({(uint32_t)count}, mx::uint32);
    auto out = kernel({a, dims}, {a.shape()}, {a.dtype()}, {count, 1, 1}, {256, 1, 1}, {},
        std::nullopt, false, {})[0];
    mx::eval(out);
    return out;
}

class Store {
  public:
    Store(int z, int y, int x, int c, bool disk)
        : Z(z), Y(y), X(x), C(c), plane_((size_t)y * x * c * sizeof(float)) {
        if (!disk)
            return;
        const char *dir = std::getenv("TMPDIR");
        std::string path = dir && *dir ? dir : "/tmp";
        if (path.back() != '/')
            path += '/';
        path += "hx-surface-XXXXXX";
        fd_ = mkstemp(path.data());
        if (fd_ < 0)
            throw std::runtime_error("cannot create a scratch file at " + path);
        unlink(path.c_str());
        worker_ = std::thread([this] { work(); });
    }
    ~Store() {
        if (fd_ < 0)
            return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        changed_.notify_all();
        worker_.join();
        close(fd_);
    }
    Store(const Store &) = delete;
    Store &operator=(const Store &) = delete;

    void put(int a, const mx::array &planes) {
        mx::eval(planes);
        const int count = planes.shape(0);
        if (fd_ < 0) {
            pieces_.push_back({a, a + count, tight(planes)});
            return;
        }
        mx::array flat = planes.flags().row_contiguous ? planes : mx::contiguous(planes);
        mx::eval(flat);
        wait(submitted_);
        submit({true, a, a + count, flat});
    }

    mx::array get(int a, int b) {
        if (fd_ >= 0) {
            if (!(ahead_ && ahead_a_ <= a && b <= ahead_b_))
                read_ahead(a, b);
            wait(ahead_id_ + 1);
            ahead_ = false;
            mx::array got = ahead_buffer_;
            ahead_buffer_ = mx::array(0.0f);
            if (ahead_a_ != a || ahead_b_ != b)
                got = mx::slice(got, {a - ahead_a_, 0, 0, 0}, {b - ahead_a_, Y, X, C});
            const int next_b = std::min(Z, b + (b - a) + 1);
            if (next_b > b) {
                read_ahead(std::max(0, b - 2), next_b);
                ahead_ = true;
            }
            return got;
        }
        std::vector<mx::array> parts;
        int covered = 0;
        for (const Piece &piece : pieces_) {
            const int lo = std::max(a, piece.a), hi = std::min(b, piece.b);
            if (lo >= hi)
                continue;
            covered += hi - lo;
            parts.push_back(lo == piece.a && hi == piece.b
                    ? piece.data
                    : mx::slice(piece.data, {lo - piece.a, 0, 0, 0}, {hi - piece.a, Y, X, C}));
        }
        if (covered != b - a)
            throw std::runtime_error("planes were asked for after their release");
        auto out = parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
        mx::eval(out);
        return out;
    }

    void release(int below) {
        pieces_.erase(std::remove_if(pieces_.begin(), pieces_.end(),
                          [&](const Piece &p) { return p.b <= below; }),
            pieces_.end());
    }

    const int Z, Y, X, C;

  private:
    struct Piece {
        int a, b;
        mx::array data;
    };

    struct Job {
        bool write;
        int a, b;
        mx::array data;
    };

    mx::array buffer(int a, int b) {
        return mx::array(
            mx::allocator::malloc((size_t)(b - a) * plane_), {b - a, Y, X, C}, mx::float32);
    }

    uint64_t submit(Job job) {
        uint64_t id = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
            id = submitted_++;
        }
        changed_.notify_all();
        return id;
    }

    void wait(uint64_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return done_ >= count || failed_; });
        if (failed_)
            throw std::runtime_error("the scratch file could not be written or read");
    }

    void read_ahead(int a, int b) {
        ahead_a_ = a;
        ahead_b_ = b;
        ahead_buffer_ = buffer(a, b);
        ahead_id_ = submit({false, a, b, ahead_buffer_});
    }

    bool transfer(Job &job) {
        char *p = job.data.data<char>();
        size_t left = (size_t)(job.b - job.a) * plane_;
        off_t at = (off_t)job.a * (off_t)plane_;
        while (left) {
            const ssize_t n = job.write ? pwrite(fd_, p, std::min(left, (size_t)1 << 30), at)
                                        : pread(fd_, p, std::min(left, (size_t)1 << 30), at);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                return false;
            p += n;
            at += n;
            left -= (size_t)n;
        }
        return true;
    }

    void work() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            changed_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty())
                return;
            bool ok = false;
            {
                Job job = std::move(jobs_.front());
                jobs_.pop_front();
                lock.unlock();
                ok = transfer(job);
            }
            lock.lock();
            ++done_;
            failed_ = failed_ || !ok;
            changed_.notify_all();
        }
    }

    std::vector<Piece> pieces_;
    size_t plane_;
    int fd_ = -1;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Job> jobs_;
    uint64_t submitted_ = 0, done_ = 0, ahead_id_ = 0;
    bool stop_ = false, failed_ = false, ahead_ = false;
    int ahead_a_ = 0, ahead_b_ = 0;
    mx::array ahead_buffer_ = mx::array(0.0f);
};

struct Planes {
    int Z = 0, Y = 0, X = 0, C = 0;
    bool cheap = false;
    std::function<mx::array(int, int)> get;
    std::function<void(int)> release = [](int) {};
};

struct Act {
    bool chunked = false;
    mx::array full = mx::array(0.0f);
    Planes planes;
};

struct Plan {
    std::vector<bool> chunked, disk;
    std::vector<int> planes;
};

static Planes planes_of(const mx::array &a) {
    Planes p;
    p.Z = a.shape(0);
    p.Y = a.shape(1);
    p.X = a.shape(2);
    p.C = a.shape(3);
    const int Y = p.Y, X = p.X, C = p.C;
    p.get = [a, Y, X, C](int z0, int z1) { return mx::slice(a, {z0, 0, 0, 0}, {z1, Y, X, C}); };
    return p;
}

static Planes planes_of(const std::shared_ptr<Store> &s) {
    Planes p;
    p.Z = s->Z;
    p.Y = s->Y;
    p.X = s->X;
    p.C = s->C;
    p.get = [s](int a, int b) { return s->get(a, b); };
    p.release = [s](int z) { s->release(z); };
    return p;
}

static Planes source_of(const Act &a) { return a.chunked ? a.planes : planes_of(a.full); }

static std::vector<std::pair<int, int>> ranges(int z, int step) {
    const int count = (z + step - 1) / step;
    std::vector<std::pair<int, int>> r;
    for (int i = 0; i < count; ++i)
        r.push_back({(int)((int64_t)i * z / count), (int)((int64_t)(i + 1) * z / count)});
    return r;
}

struct Welford {
    int channels = 0, voxels = 0, lanes = 0, groups = 0;
    mx::array state = mx::array(0.0f);
};

static Welford welford_begin(int channels, int voxels) {
    Welford w;
    w.channels = channels;
    w.voxels = voxels;
    w.lanes = std::max(1, std::min(32, 1024 / channels));
    w.groups = std::min(1024, std::max(1, (voxels + w.lanes * 64 - 1) / (w.lanes * 64)));
    w.state = mx::zeros({w.groups * w.lanes * channels * 3}, mx::float32);
    return w;
}

static void welford_add(NetState &n, Welford &w, const mx::array &x, int first) {
    const int count = (int)(x.size() / w.channels);
    auto dims =
        mx::array({(uint32_t)w.voxels, (uint32_t)first, (uint32_t)(first + count)}, mx::uint32);
    w.state = n.step({x, w.state, dims}, {w.state.shape()}, {mx::float32},
        {w.groups * w.channels, w.lanes, 1}, {w.channels, w.lanes, 1},
        {{"C", w.channels}, {"R", w.lanes}}, std::nullopt, false, {})[0];
    mx::eval(w.state);
}

static std::pair<mx::array, mx::array> welford_end(NetState &n, Welford &w) {
    auto p = n.lanes({w.state}, {{w.groups * w.channels * 3}}, {mx::float32},
        {w.groups * w.channels, 1, 1}, {w.channels, 1, 1}, {{"C", w.channels}, {"R", w.lanes}},
        std::nullopt, false, {})[0];
    auto mv = n.merge({p}, {{w.channels}, {w.channels}}, {mx::float32, mx::float32},
        {w.channels, 1, 1}, {std::min(w.channels, 256), 1, 1}, {{"C", w.channels}, {"G", w.groups}},
        std::nullopt, false, {});
    auto mean = mx::reshape(mv[0], {1, 1, 1, w.channels});
    auto var = mx::reshape(mv[1], {1, 1, 1, w.channels});
    mx::eval(mean, var);
    w.state = mx::array(0.0f);
    return {mean, var};
}

static mx::array conv_rows(const mx::array &slab, const mx::array &w, int skip, int rows) {
    auto y = mx::squeeze(mx::conv3d(mx::expand_dims(slab, 0), w, {1, 1, 1}, {1, 1, 1}), 0);
    auto out = mx::slice(y, {skip, 0, 0, 0}, {skip + rows, y.shape(1), y.shape(2), y.shape(3)});
    mx::eval(out);
    return out;
}

static mx::array conv_planes(const Planes &src, const mx::array &w, int z0, int z1) {
    const int lo = std::max(0, z0 - 1), hi = std::min(src.Z, z1 + 1);
    return conv_rows(src.get(lo, hi), w, z0 - lo, z1 - z0);
}

static mx::array stem_planes(NetState &n, const mx::array &x, int a, int b) {
    const int lo = std::max(0, a - 1), hi = std::min(x.shape(0), b + 1);
    auto y = stem_conv(
        n, mx::slice(x, {lo, 0, 0, 0}, {hi, x.shape(1), x.shape(2), x.shape(3)}), W(n, n.stem_w));
    return mx::slice(y, {a - lo, 0, 0, 0}, {b - lo, y.shape(1), y.shape(2), y.shape(3)});
}

static mx::array block_out(NetState &n, const Block &b, const mx::array &y, const mx::array &mean,
    const mx::array &var, const mx::array &r) {
    if (!b.scse)
        return n.norm_residual({y, mean, var, W(n, b.g2), W(n, b.b2), r})[0];
    const int channels = y.shape().back();
    const int voxels = (int)(y.size() / channels);
    auto dims = mx::array({(uint32_t)voxels}, mx::uint32);
    return n.scse({y, mx::reshape(mean, {-1}), mx::reshape(var, {-1}), W(n, b.g2), W(n, b.b2),
                      mx::reshape(b.gate, {-1}), mx::reshape(W(n, b.sse), {-1}),
                      mx::reshape(W(n, b.sseb), {-1}), r, dims},
        {y.shape()}, {mx::float32}, {voxels * 32, 1, 1}, {256, 1, 1}, {{"C", channels}},
        std::nullopt, false, {})[0];
}

static mx::array skip_of(NetState &n, const mx::array &x, const Block &b) {
    auto r = x;
    if (b.pool) {
        const auto &s = r.shape();
        r = mx::mean(mx::reshape(r, {s[0] / 2, 2, s[1] / 2, 2, s[2] / 2, 2, s[3]}), {1, 3, 5});
    }
    if (!b.sc.empty()) {
        const auto &wsc = W(n, b.sc);
        r = mx::matmul(r, mx::transpose(mx::reshape(wsc, {wsc.shape()[0], wsc.shape()[4]})));
    }
    return r;
}

static mx::array block_tail(NetState &n, mx::array y, mx::array r, Block &b) {
    mx::eval(y);
    auto h = norm_of(n, y, b.g1, b.b1, true);
    y = mx::array(0.0f);
    mx::eval(h);
    y = conv(h, W(n, b.c2), 1);
    mx::eval(y);
    h = mx::array(0.0f);
    if (!b.sc.empty()) {
        mx::eval(r);
        r = norm_of(n, r, b.sg, b.sb, false);
    }
    mx::eval(r);
    auto [mean, var] = stats(n, y);
    auto out = block_out(n, b, y, mean, var, r);
    mx::eval(out);
    return out;
}

static Act chunked_block(NetState &n, Act &in, Block &b, int level, const Plan &plan, bool owned) {
    Planes src = source_of(in);
    const bool down = b.stride == 2, keep = plan.chunked[level];
    const bool projected = b.pool || !b.sc.empty(), normed = !b.sc.empty();
    const auto &w1 = W(n, b.c1);
    const int C = w1.shape(0), Z = down ? src.Z / 2 : src.Z, Y = down ? src.Y / 2 : src.Y,
              X = down ? src.X / 2 : src.X, plane = Y * X, voxels = Z * plane;
    const int step = down ? std::max(1, plan.planes[level - 1] / 2) : plan.planes[level];
    const bool disk = keep && plan.disk[level];
    auto y1 = std::make_shared<Store>(Z, Y, X, C, disk);
    std::shared_ptr<Store> r1;
    if (projected)
        r1 = std::make_shared<Store>(Z, Y, X, normed ? C : src.C, disk);
    Welford s1 = welford_begin(C, voxels), sr = welford_begin(r1 ? r1->C : C, voxels);
    for (auto [a, e] : ranges(Z, step)) {
        mx::array y(0.0f), r(0.0f);
        if (down) {
            const int lo = std::max(0, 2 * a - 1), hi = std::min(src.Z, 2 * e);
            auto slab = src.get(lo, hi);
            auto padded = a == 0
                ? mx::pad(slab, std::vector<std::pair<int, int>>{{1, 0}, {0, 0}, {0, 0}, {0, 0}})
                : slab;
            y = mx::squeeze(mx::conv3d(mx::expand_dims(padded, 0), w1, {2, 2, 2}, {0, 1, 1}), 0);
            mx::eval(y);
            if (projected) {
                r = skip_of(n,
                    mx::slice(slab, {2 * a - lo, 0, 0, 0}, {2 * e - lo, src.Y, src.X, src.C}), b);
                mx::eval(r);
            }
        } else {
            y = conv_planes(src, w1, a, e);
            if (projected) {
                r = skip_of(n, src.get(a, e), b);
                mx::eval(r);
            }
        }
        if (keep)
            welford_add(n, s1, y, a * plane);
        y1->put(a, y);
        if (projected) {
            if (keep && normed)
                welford_add(n, sr, r, a * plane);
            r1->put(a, r);
        }
    }
    trace("block conv1", level);
    if (!keep) {
        auto y = y1->get(0, Z);
        y1.reset();
        mx::array r = in.chunked ? mx::array(0.0f) : in.full;
        if (projected) {
            r = r1->get(0, Z);
            r1.reset();
        } else if (in.chunked) {
            r = src.get(0, Z);
        }
        Act out;
        out.full = block_tail(n, y, r, b);
        trace("block out", level);
        return out;
    }
    auto [m1, v1] = welford_end(n, s1);
    Planes h;
    h.Z = Z;
    h.Y = Y;
    h.X = X;
    h.C = C;
    h.get = [&n, &b, y1, m1, v1](int a, int e) {
        auto out = n.norm_act({y1->get(a, e), m1, v1, W(n, b.g1), W(n, b.b1)})[0];
        mx::eval(out);
        return out;
    };
    auto y2 = std::make_shared<Store>(Z, Y, X, C, disk);
    Welford s2 = welford_begin(C, voxels);
    for (auto [a, e] : ranges(Z, plan.planes[level])) {
        auto y = conv_planes(h, W(n, b.c2), a, e);
        welford_add(n, s2, y, a * plane);
        y2->put(a, y);
        y1->release(e - 1);
    }
    h = Planes();
    y1.reset();
    auto [m2, v2] = welford_end(n, s2);
    trace("block conv2", level);
    Act out;
    out.chunked = true;
    if (!projected && src.cheap) {
        out.planes.Z = Z;
        out.planes.Y = Y;
        out.planes.X = X;
        out.planes.C = C;
        out.planes.get = [&n, &b, y2, m2, v2, src](int a, int e) {
            auto o = block_out(n, b, y2->get(a, e), m2, v2, src.get(a, e));
            mx::eval(o);
            return o;
        };
        out.planes.release = [y2](int z) { y2->release(z); };
        return out;
    }
    mx::array mr(0.0f), vr(0.0f);
    if (projected && normed)
        std::tie(mr, vr) = welford_end(n, sr);
    auto result = std::make_shared<Store>(Z, Y, X, C, disk);
    for (auto [a, e] : ranges(Z, plan.planes[level])) {
        mx::array r = projected ? r1->get(a, e) : src.get(a, e);
        if (projected && normed)
            r = n.norm({r, mr, vr, W(n, b.sg), W(n, b.sb)})[0];
        auto o = block_out(n, b, y2->get(a, e), m2, v2, r);
        mx::eval(o);
        result->put(a, o);
        y2->release(e);
        if (r1)
            r1->release(e);
        if (owned && !projected)
            src.release(e);
    }
    trace("block out", level);
    out.planes = planes_of(result);
    return out;
}

static Act chunked_decoder(
    NetState &n, Act &x, Act &skip, Decoder &d, int level, const Plan &plan) {
    Planes xs = source_of(x), sk = source_of(skip);
    const int Z = sk.Z, Y = sk.Y, X = sk.X, C = d.wa.shape(0);
    auto h = std::make_shared<Store>(Z, Y, X, C, plan.disk[level]);
    Welford s = welford_begin(C, Z * Y * X);
    for (auto [z0, z1] : ranges(Z, plan.planes[level])) {
        const int lo = std::max(0, z0 - 1), hi = std::min(Z, z1 + 1);
        const int a = lo / 2, e = (hi + 1) / 2;
        auto up = transposed(n, xs.get(a, e), W(n, d.up), W(n, d.upb));
        mx::eval(up);
        up = mx::slice(up, {lo - 2 * a, 0, 0, 0}, {hi - 2 * a, Y, X, up.shape(3)});
        auto ha = conv_rows(up, d.wa, z0 - lo, z1 - z0);
        up = mx::array(0.0f);
        auto hb = conv_rows(sk.get(lo, hi), d.wb, z0 - lo, z1 - z0);
        auto sum = ha + hb;
        mx::eval(sum);
        ha = mx::array(0.0f);
        hb = mx::array(0.0f);
        welford_add(n, s, sum, z0 * Y * X);
        h->put(z0, sum);
        sk.release(z1 - 1);
        xs.release((z1 - 1) / 2);
    }
    auto [m, v] = welford_end(n, s);
    trace("decoder", level);
    Act out;
    out.chunked = true;
    out.planes.Z = Z;
    out.planes.Y = Y;
    out.planes.X = X;
    out.planes.C = C;
    out.planes.get = [&n, &d, h, m, v](int a, int e) {
        auto o = n.norm_act({h->get(a, e), m, v, W(n, d.g), W(n, d.b)})[0];
        mx::eval(o);
        return o;
    };
    out.planes.release = [h](int z) { h->release(z); };
    return out;
}

static Plan plan_for(NetState &n, const mx::array &x) {
    Plan plan;
    const double whole = setting("HX_SURFACE_WHOLE_MB", 320) * 1e6,
                 chunk = setting("HX_SURFACE_CHUNK_MB", 192) * 1e6,
                 disk = setting("HX_SURFACE_DISK_MB", 512) * 1e6,
                 slab = setting("HX_SURFACE_SLAB_MB", 128) * 1e6;
    const int Z = x.shape(0), Y = x.shape(1), X = x.shape(2);
    for (size_t s = 0; s < n.stage_channels.size(); ++s) {
        const int z = Z >> s, y = Y >> s, xx = X >> s;
        const double plane = (double)y * xx * n.stage_channels[s] * sizeof(float),
                     bytes = plane * z;
        const bool chunked = s == 0 ? bytes > whole : plan.chunked[s - 1] && bytes >= chunk;
        plan.chunked.push_back(chunked);
        plan.disk.push_back(chunked && bytes >= disk);
        plan.planes.push_back(std::max(1, std::min(z, (int)(slab / plane))));
    }
    return plan;
}

static mx::array chunked_forward(NetState &n, const mx::array &x, const Plan &plan) {
    const int Z = x.shape(0), Y = x.shape(1), X = x.shape(2);
    mx::set_cache_limit(std::min(n.cache, (size_t)(setting("HX_SURFACE_CACHE_MB", 256) * 1e6)));
    trace("chunked forward");
    const int C0 = W(n, n.stem_w).shape(0);
    Welford ws = welford_begin(C0, Z * Y * X);
    for (auto [a, e] : ranges(Z, plan.planes[0])) {
        auto s = stem_planes(n, x, a, e);
        mx::eval(s);
        welford_add(n, ws, s, a * Y * X);
    }
    auto [sm, sv] = welford_end(n, ws);
    Act cur;
    cur.chunked = true;
    cur.planes.Z = Z;
    cur.planes.Y = Y;
    cur.planes.X = X;
    cur.planes.C = C0;
    cur.planes.cheap = true;
    cur.planes.get = [&n, x, sm, sv](int a, int e) {
        auto o = n.norm_act({stem_planes(n, x, a, e), sm, sv, W(n, n.stem_g), W(n, n.stem_b)})[0];
        mx::eval(o);
        return o;
    };
    trace("stem");
    std::vector<Act> skips;
    int level = 0;
    bool first = true;
    for (auto &op : n.ops) {
        if (op.endstage) {
            skips.push_back(cur);
            ++level;
            first = true;
            continue;
        }
        if (!plan.chunked[level] && !cur.chunked) {
            Act next;
            next.full = block_forward(n, cur.full, op.block);
            cur = next;
        } else {
            cur = chunked_block(n, cur, op.block, level, plan, !first);
        }
        first = false;
    }
    cur = Act();
    Act x_act = skips.back();
    skips.pop_back();
    for (auto &d : n.decoders) {
        Act skip = skips.back();
        skips.pop_back();
        const int lv = (int)skips.size();
        if (plan.chunked[lv]) {
            x_act = chunked_decoder(n, x_act, skip, d, lv, plan);
            continue;
        }
        auto up = transposed(n, x_act.full, W(n, d.up), W(n, d.upb));
        x_act.full = mx::array(0.0f);
        mx::eval(up);
        auto h = conv(up, d.wa, 1);
        mx::eval(h);
        up = mx::array(0.0f);
        auto hb = conv(skip.full, d.wb, 1);
        mx::eval(hb);
        skip.full = mx::array(0.0f);
        h = h + hb;
        hb = mx::array(0.0f);
        mx::eval(h);
        x_act.full = norm_of(n, h, d.g, d.b, true);
        h = mx::array(0.0f);
        mx::eval(x_act.full);
    }
    Planes top = source_of(x_act);
    x_act = Act();
    std::vector<mx::array> parts;
    for (auto [a, e] : ranges(Z, plan.planes[0])) {
        auto l = mx::matmul(top.get(a, e), W(n, n.head_w)) + W(n, n.head_b);
        mx::eval(l);
        parts.push_back(l);
        top.release(e);
    }
    top = Planes();
    auto logits = parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
    mx::eval(logits);
    trace("head");
    return logits;
}

static std::string name_of(const Json &entry, const std::string &a, const std::string &b = "") {
    if (!entry.has(a))
        return "";
    const Json &e = entry[a];
    if (b.empty())
        return e.kind == Json::String ? e.str() : "";
    return e.has(b) ? e[b].str() : "";
}

SurfaceNet::SurfaceNet(
    const std::string &spec_path, const std::string &weights_path, double cache_gb)
    : state_(std::make_unique<NetState>()) {
    mx::set_cache_limit((size_t)(cache_gb * (1ull << 30)));
    state_->cache = (size_t)(cache_gb * (1ull << 30));
    Json spec = Json::parse(read_text(spec_path));
    name_ = spec["name"].str();
    patch_ = (int)spec["patch"].num();
    const Json &input = spec["input"];
    if (input["scheme"].str() == "ct") {
        rule_ = Rule::NNUNET;
        lower_ = (float)input["lower"].num();
        upper_ = (float)input["upper"].num();
        mean_ = (float)input["mean"].num();
        std_ = (float)std::max(input["std"].num(), 1e-8);
    } else {
        rule_ = Rule::VESUVIUS;
    }
    NetState &n = *state_;
    auto loaded = mx::load_safetensors(weights_path);
    for (auto &[k, v] : loaded.first)
        n.w.insert_or_assign(k, mx::astype(v, mx::float32));
    n.partial = mx::fast::metal_kernel(
        "hx_channel_stats_partial", {"x", "dims"}, {"partial"}, STATS_PARTIAL);
    n.merge =
        mx::fast::metal_kernel("hx_channel_stats_merge", {"partial"}, {"mean", "var"}, STATS_MERGE);
    n.scse = mx::fast::metal_kernel("hx_norm_scse_residual",
        {"y", "mean", "var", "gamma", "beta", "gate", "sse", "sse_b", "r", "dims"}, {"out"},
        NORM_SCSE_RESIDUAL);
    n.upsample =
        mx::fast::metal_kernel("hx_transposed_shuffle", {"m", "bias", "dims"}, {"out"}, UPSAMPLE);
    n.stem = mx::fast::metal_kernel("hx_stem_conv", {"x", "w", "b", "dims"}, {"y"}, STEM);
    n.step = mx::fast::metal_kernel(
        "hx_channel_stats_step", {"x", "state", "dims"}, {"next"}, STATS_STEP);
    n.lanes = mx::fast::metal_kernel("hx_channel_stats_lanes", {"state"}, {"partial"}, STATS_LANES);
    n.norm_act = mx::compile(norm_act_impl, true);
    n.norm = mx::compile(norm_impl, true);
    n.norm_residual = mx::compile(norm_residual_impl, true);
    n.stem_w = spec["stem"]["conv"]["weight"].str();
    n.stem_g = spec["stem"]["norm"]["weight"].str();
    n.stem_b = spec["stem"]["norm"]["bias"].str();
    std::vector<mx::array> keep;
    for (const Json &stage : spec["stages"].items) {
        n.stage_channels.push_back(
            (int)W(n, stage.items.at(0)["conv1"]["conv"]["weight"].str()).shape(0));
        for (const Json &blk : stage.items) {
            Op op;
            Block &b = op.block;
            b.stride = (int)blk["conv1"]["conv"]["stride"].num();
            b.c1 = blk["conv1"]["conv"]["weight"].str();
            b.g1 = blk["conv1"]["norm"]["weight"].str();
            b.b1 = blk["conv1"]["norm"]["bias"].str();
            b.c2 = blk["conv2"]["conv"]["weight"].str();
            b.g2 = blk["conv2"]["norm"]["weight"].str();
            b.b2 = blk["conv2"]["norm"]["bias"].str();
            const Json &skip = blk["skip"];
            b.pool = skip["pool"].truth();
            b.sc = name_of(skip, "conv", "weight");
            b.sg = name_of(skip, "norm", "weight");
            b.sb = name_of(skip, "norm", "bias");
            if (blk.has("scse") && blk["scse"].kind == Json::Object) {
                const Json &s = blk["scse"];
                b.scse = true;
                b.sse = s["sse"].str();
                b.sseb = s["sse_bias"].str();
                auto hidden = mx::maximum(
                    mx::matmul(W(n, b.b2), W(n, s["fc1"].str())) + W(n, s["fc1_bias"].str()),
                    mx::array(0.0f));
                b.gate = mx::sigmoid(
                    mx::matmul(hidden, W(n, s["fc2"].str())) + W(n, s["fc2_bias"].str()));
                keep.push_back(b.gate);
            }
            n.ops.push_back(op);
        }
        Op end;
        end.endstage = true;
        n.ops.push_back(end);
    }
    for (const Json &dec : spec["decoder"].items) {
        Decoder d;
        d.up = dec["up"]["weight"].str();
        d.upb = dec["up"]["bias"].str();
        d.g = dec["block"]["norm"]["weight"].str();
        d.b = dec["block"]["norm"]["bias"].str();
        auto upw = W(n, d.up);
        const int up_out = upw.shape().back(), cin = upw.shape()[0];
        auto cw = W(n, dec["block"]["conv"]["weight"].str());
        const auto &s = cw.shape();
        d.wa = mx::contiguous(mx::slice(cw, {0, 0, 0, 0, 0}, {s[0], s[1], s[2], s[3], up_out}));
        d.wb = mx::contiguous(mx::slice(cw, {0, 0, 0, 0, up_out}, {s[0], s[1], s[2], s[3], s[4]}));
        n.w.insert_or_assign(d.up, mx::reshape(upw, {cin, -1}));
        keep.push_back(d.wa);
        keep.push_back(d.wb);
        n.decoders.push_back(d);
    }
    n.head_w = spec["head"]["weight"].str();
    n.head_b = spec["head"]["bias"].str();
    for (auto &[k, v] : n.w)
        keep.push_back(v);
    mx::eval(keep);
}

SurfaceNet::~SurfaceNet() = default;

static double pairwise_sum(const double *a, size_t n) {
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
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

void SurfaceNet::normalise(const uint8_t *ct, size_t n, float *out) const {
    if (rule_ == Rule::NNUNET) {
        for (size_t i = 0; i < n; ++i) {
            const float v = std::min(std::max((float)ct[i], lower_), upper_);
            out[i] = (v - mean_) / std_;
        }
        return;
    }
    std::vector<double> values(n);
    for (size_t i = 0; i < n; ++i)
        values[i] = (double)ct[i];
    const double mean = pairwise_sum(values.data(), n) / (double)n;
    std::vector<double> sq(n);
    for (size_t i = 0; i < n; ++i) {
        const double d = values[i] - mean;
        sq[i] = d * d;
    }
    const double sd = std::max(std::sqrt(pairwise_sum(sq.data(), n) / (double)n), 1e-8);
    for (size_t i = 0; i < n; ++i)
        out[i] = (float)((values[i] - mean) / sd);
}

mx::array SurfaceNet::forward(const mx::array &input) {
    NetState &n = *state_;
    auto x = mx::astype(input, mx::float32);
    const Plan plan = plan_for(n, x);
    if (plan.chunked.at(0)) {
        mx::eval(x);
        return chunked_forward(n, x, plan);
    }
    auto y = stem_conv(n, x, W(n, n.stem_w));
    mx::eval(y);
    x = norm_of(n, y, n.stem_g, n.stem_b, true);
    y = mx::array(0.0f);
    mx::eval(x);
    std::vector<mx::array> skips;
    for (auto &op : n.ops) {
        if (!op.endstage)
            x = block_forward(n, x, op.block);
        else
            skips.push_back(x);
    }
    x = skips.back();
    skips.pop_back();
    for (auto &d : n.decoders) {
        auto skip = skips.back();
        skips.pop_back();
        auto up = transposed(n, x, W(n, d.up), W(n, d.upb));
        x = mx::array(0.0f);
        mx::eval(up);
        auto h = conv(up, d.wa, 1);
        mx::eval(h);
        up = mx::array(0.0f);
        auto hb = conv(skip, d.wb, 1);
        mx::eval(hb);
        skip = mx::array(0.0f);
        h = h + hb;
        hb = mx::array(0.0f);
        mx::eval(h);
        x = norm_of(n, h, d.g, d.b, true);
        h = mx::array(0.0f);
        mx::eval(x);
    }
    auto logits = mx::matmul(x, W(n, n.head_w)) + W(n, n.head_b);
    mx::eval(logits);
    return logits;
}

}
