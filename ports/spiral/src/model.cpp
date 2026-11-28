#include "model.h"

#include "gpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <dispatch/dispatch.h>
#include <stdexcept>

namespace sp {

static constexpr float TWO_PI = 6.2831855f;

template <class F> static void parallel(size_t n, size_t grain, F &&body) {
    const size_t chunks = (n + grain - 1) / grain;
    if (chunks <= 1) {
        body(0, n);
        return;
    }
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
      body(c * grain, std::min(n, (c + 1) * grain));
    });
}

static inline void atomic_add(float *at, float v) {
    std::atomic_ref<float> ref(*at);
    float old = ref.load(std::memory_order_relaxed);
    while (!ref.compare_exchange_weak(old, old + v, std::memory_order_relaxed)) {
    }
}

static inline void atomic_add(double *at, double v) {
    std::atomic_ref<double> ref(*at);
    double old = ref.load(std::memory_order_relaxed);
    while (!ref.compare_exchange_weak(old, old + v, std::memory_order_relaxed)) {
    }
}

static inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float lerp(float a, float b, float w) {
    return w < 0.5f ? a + w * (b - a) : b - (b - a) * (1.0f - w);
}
static inline void lerp_grad(float a, float b, float w, float g, float &ga, float &gb, float &gw) {
    ga = g * (1.0f - w);
    gb = g * w;
    gw = g * (b - a);
}

void Model::build(const Settings &settings, const std::vector<std::array<float, 3>> &points) {
    s = settings;
    flow_min = {s.z_begin - s.flow_z_margin, -s.flow_radius, -s.flow_radius};
    flow_max = {s.z_end + s.flow_z_margin, s.flow_radius, s.flow_radius};
    for (int i = 0; i < 3; ++i) {
        range[i] = (float)(flow_max[i] - flow_min[i]);
        hr[i] = (int)((flow_max[i] - flow_min[i]) / s.flow_voxel_resolution);
        lr[i] = (int)(hr[i] / s.flow_scale_factor);
    }
    low.assign((size_t)s.stages * 3 * lr_cells(), 0.0f);
    high.assign((size_t)s.stages * 3 * hr_cells(), 0.0f);
    const float residual = (float)(s.initial_dr - s.min_gap);
    dr_logit = std::log(std::expm1(residual)) / (float)s.dr_parameter_scale;
    linear_z = (int)((flow_max[0] - flow_min[0]) / s.linear_z_resolution);
    linear.assign((size_t)linear_z * 4, 0.0f);
    windings = s.gap_capacity_windings - 1;
    first_index.assign((size_t)windings + 1, 0.0f);
    int64_t total = 0;
    for (int k = 1; k <= windings; ++k) {
        const float v = (float)(2 * M_PI) * ((float)k + 0.5f) * (float)s.initial_dr /
                (float)s.gap_logit_resolution +
            0.5f;
        total += (int64_t)v;
        first_index[(size_t)k] = (float)total;
    }
    gap_width = (int)total;
    gap_num_z = (int)((double)(flow_max[0] - flow_min[0]) / (double)s.gap_logit_resolution);
    gap.assign((size_t)gap_num_z * gap_width, 0.0f);
    const double bias = s.softplus_bias;
    const double softplus_bias = std::log1p(std::exp(-std::fabs(bias))) + std::max(bias, 0.0);
    const double sigmoid_bias = 1.0 / (1.0 + std::exp(-bias));
    softplus_scale =
        (float)(200.0 * s.initial_dr / (s.initial_dr - s.min_gap) * softplus_bias / sigmoid_bias);
    softplus_denominator = softplus((float)bias);
    const int64_t n = s.z_end - s.z_begin;
    std::vector<double> raw_y((size_t)n), raw_x((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        const double z = (double)(s.z_begin + i);
        size_t k = (size_t)(std::lower_bound(points.begin(), points.end(), (float)z,
                                [](const std::array<float, 3> &p, float v) { return p[0] < v; }) -
            points.begin());
        k = std::min(std::max<size_t>(k, 1), points.size() - 1);
        const auto &a = points[k - 1], &b = points[k];
        const double t = (z - a[0]) / (double)(b[0] - a[0]);
        raw_y[(size_t)i] = (double)(float)(a[1] + t * (b[1] - a[1]));
        raw_x[(size_t)i] = (double)(float)(a[2] + t * (b[2] - a[2]));
    }
    const int radius = (int)(4.0 * 75.0 + 0.5);
    std::vector<double> w((size_t)(2 * radius + 1));
    double sum = 0;
    for (int k = -radius; k <= radius; ++k)
        sum += (w[(size_t)(k + radius)] = std::exp(-0.5 / (75.0 * 75.0) * k * k));
    for (double &v : w)
        v /= sum;
    umbilicus_z.resize((size_t)n);
    umbilicus_yx.resize((size_t)n);
    umbilicus_raw.resize((size_t)n * 3);
    for (int64_t i = 0; i < n; ++i) {
        umbilicus_raw[(size_t)i * 3] = (float)(s.z_begin + i);
        umbilicus_raw[(size_t)i * 3 + 1] = (float)raw_y[(size_t)i];
        umbilicus_raw[(size_t)i * 3 + 2] = (float)raw_x[(size_t)i];
    }
    for (int64_t i = 0; i < n; ++i) {
        double y = 0, x = 0;
        for (int k = -radius; k <= radius; ++k) {
            const int64_t j = std::min<int64_t>(std::max<int64_t>(i + k, 0), n - 1);
            y += w[(size_t)(k + radius)] * raw_y[(size_t)j];
            x += w[(size_t)(k + radius)] * raw_x[(size_t)j];
        }
        umbilicus_z[(size_t)i] = (float)(s.z_begin + i);
        umbilicus_yx[(size_t)i] = {(float)y, (float)x};
    }
}

static inline void axis_weights(int in, int out, int o, int &i0, int &i1, float &l0, float &l1) {
    const float scale = (float)in / (float)out;
    float src = scale * ((float)o + 0.5f) - 0.5f;
    if (src < 0)
        src = 0;
    i0 = (int)src;
    i1 = i0 + (i0 < in - 1 ? 1 : 0);
    l1 = src - (float)i0;
    l0 = 1.0f - l1;
}

void upsample(const Model &m, const Field &low, Field &up) {
    up.resize((size_t)m.s.stages * 3 * m.hr_cells());
    const int Z = m.hr[0], Y = m.hr[1], X = m.hr[2], lz = m.lr[0], ly = m.lr[1], lx = m.lr[2];
    const size_t planes = (size_t)m.s.stages * 3;
    parallel(planes * Z, 1, [&](size_t a, size_t b) {
        for (size_t pz = a; pz < b; ++pz) {
            const size_t plane = pz / Z;
            const int z = (int)(pz % Z);
            int z0, z1, y0, y1, x0, x1;
            float tz0, tz1, ty0, ty1, tx0, tx1;
            axis_weights(lz, Z, z, z0, z1, tz0, tz1);
            const float *src = low.data() + plane * m.lr_cells();
            float *dst = up.data() + plane * m.hr_cells() + (size_t)z * Y * X;
            for (int y = 0; y < Y; ++y) {
                axis_weights(ly, Y, y, y0, y1, ty0, ty1);
                for (int x = 0; x < X; ++x) {
                    axis_weights(lx, X, x, x0, x1, tx0, tx1);
                    auto at = [&](int k, int j, int i) {
                        return src[((size_t)k * ly + j) * lx + i];
                    };
                    dst[(size_t)y * X + x] = tz0 *
                            (ty0 * (tx0 * at(z0, y0, x0) + tx1 * at(z0, y0, x1)) +
                                ty1 * (tx0 * at(z0, y1, x0) + tx1 * at(z0, y1, x1))) +
                        tz1 *
                            (ty0 * (tx0 * at(z1, y0, x0) + tx1 * at(z1, y0, x1)) +
                                ty1 * (tx0 * at(z1, y1, x0) + tx1 * at(z1, y1, x1)));
                }
            }
        }
    });
}

void upsample_adjoint(const Model &m, const Field &grad_up, Field &grad_low) {
    grad_low.assign((size_t)m.s.stages * 3 * m.lr_cells(), 0.0f);
    const int Z = m.hr[0], Y = m.hr[1], X = m.hr[2], lz = m.lr[0], ly = m.lr[1], lx = m.lr[2];
    const size_t planes = (size_t)m.s.stages * 3;
    parallel(planes, 1, [&](size_t a, size_t b) {
        for (size_t plane = a; plane < b; ++plane) {
            const float *g = grad_up.data() + plane * m.hr_cells();
            std::vector<double> acc(m.lr_cells(), 0.0);
            for (int z = 0; z < Z; ++z) {
                int z0, z1, y0, y1, x0, x1;
                float tz0, tz1, ty0, ty1, tx0, tx1;
                axis_weights(lz, Z, z, z0, z1, tz0, tz1);
                for (int y = 0; y < Y; ++y) {
                    axis_weights(ly, Y, y, y0, y1, ty0, ty1);
                    for (int x = 0; x < X; ++x) {
                        axis_weights(lx, X, x, x0, x1, tx0, tx1);
                        const double v = g[((size_t)z * Y + y) * X + x];
                        auto add = [&](int k, int j, int i, double w) {
                            acc[((size_t)k * ly + j) * lx + i] += w * v;
                        };
                        add(z0, y0, x0, (double)tz0 * ty0 * tx0);
                        add(z0, y0, x1, (double)tz0 * ty0 * tx1);
                        add(z0, y1, x0, (double)tz0 * ty1 * tx0);
                        add(z0, y1, x1, (double)tz0 * ty1 * tx1);
                        add(z1, y0, x0, (double)tz1 * ty0 * tx0);
                        add(z1, y0, x1, (double)tz1 * ty0 * tx1);
                        add(z1, y1, x0, (double)tz1 * ty1 * tx0);
                        add(z1, y1, x1, (double)tz1 * ty1 * tx1);
                    }
                }
            }
            float *out = grad_low.data() + plane * m.lr_cells();
            for (size_t i = 0; i < acc.size(); ++i)
                out[i] = (float)acc[i];
        }
    });
}

void make_shared(const Model &m, Shared &sh) {
    sh.dr = (float)m.s.min_gap + softplus(m.dr_logit * (float)m.s.dr_parameter_scale);
    sh.linear.resize(m.linear.size());
    for (size_t i = 0; i < m.linear.size(); ++i)
        sh.linear[i] = m.linear[i] * (float)m.s.linear_logits_scale;
    sh.gap.resize(m.gap.size());
    for (int z = 0; z < m.gap_num_z; ++z)
        for (int w = 0; w < m.gap_width; ++w) {
            const size_t i = (size_t)z * m.gap_width + w;
            sh.gap[i] = w == 0 ? 0.0f : m.gap[i] * (float)m.s.gap_lr_scale;
        }
    if (gpu::enabled())
        gpu::upsample(m, m.low, sh.low_up);
    else
        upsample(m, m.low, sh.low_up);
}

void Grads::reset(const Model &m) {
    dr = 0;
    linear.assign(m.linear.size(), 0.0);
    gap.assign(m.gap.size(), 0.0);
    const size_t n = (size_t)m.s.stages * 3 * m.hr_cells();
    if (field.size() != n || !gpu::enabled())
        field.assign(n, 0.0f);
    else
        gpu::zero(field);
}

struct Sample {
    const float *low, *high;
    float *acc;
    int Z, Y, X;
    size_t cells;
};

static inline void trilinear_setup(
    const Sample &f, const float p[3], int lo[3], float fr[3], bool inside[3]) {
    const int sizes[3] = {f.Z, f.Y, f.X};
    for (int d = 0; d < 3; ++d) {
        const float raw = p[d] * (float)(sizes[d] - 1);
        inside[d] = raw >= 0.0f && raw <= (float)(sizes[d] - 1);
        float c = std::min(std::max(raw, 0.0f), (float)(sizes[d] - 1));
        if (std::isnan(c))
            c = 0.0f;
        int l = (int)std::floor(c);
        l = std::min(std::max(l, 0), sizes[d] - 2);
        lo[d] = l;
        fr[d] = c - (float)l;
    }
}

static inline void velocity(const Sample &f, const float p[3], float v[3]) {
    int lo[3];
    float fr[3];
    bool inside[3];
    trilinear_setup(f, p, lo, fr, inside);
    const size_t base = ((size_t)lo[0] * f.Y + lo[1]) * f.X + lo[2];
    const size_t offs[8] = {0, 1, (size_t)f.X, (size_t)f.X + 1, (size_t)f.Y * f.X,
        (size_t)f.Y * f.X + 1, (size_t)f.Y * f.X + f.X, (size_t)f.Y * f.X + f.X + 1};
    for (int c = 0; c < 3; ++c) {
        float lv = 0, hv = 0;
        for (int k = 0; k < 8; ++k) {
            const float w = (k & 4 ? fr[0] : 1 - fr[0]) * (k & 2 ? fr[1] : 1 - fr[1]) *
                (k & 1 ? fr[2] : 1 - fr[2]);
            lv += w * f.low[c * f.cells + base + offs[k]];
            hv += w * f.high[c * f.cells + base + offs[k]];
        }
        v[c] = lv + hv;
    }
}

static inline void velocity_backward(
    const Sample &f, const float p[3], const float g[3], float gp[3]) {
    int lo[3];
    float fr[3];
    bool inside[3];
    trilinear_setup(f, p, lo, fr, inside);
    const size_t base = ((size_t)lo[0] * f.Y + lo[1]) * f.X + lo[2];
    const size_t offs[8] = {0, 1, (size_t)f.X, (size_t)f.X + 1, (size_t)f.Y * f.X,
        (size_t)f.Y * f.X + 1, (size_t)f.Y * f.X + f.X, (size_t)f.Y * f.X + f.X + 1};
    float dz = 0, dy = 0, dx = 0;
    const int sizes[3] = {f.Z, f.Y, f.X};
    for (int k = 0; k < 8; ++k) {
        const float wz = k & 4 ? fr[0] : 1 - fr[0], wy = k & 2 ? fr[1] : 1 - fr[1],
                    wx = k & 1 ? fr[2] : 1 - fr[2];
        const float w = wz * wy * wx;
        float vg = 0;
        for (int c = 0; c < 3; ++c) {
            const size_t at = c * f.cells + base + offs[k];
            if (f.acc)
                atomic_add(&f.acc[at], g[c] * w);
            vg += (f.low[at] + f.high[at]) * g[c];
        }
        dz += vg * (k & 4 ? 1.0f : -1.0f) * wy * wx;
        dy += vg * (k & 2 ? 1.0f : -1.0f) * wz * wx;
        dx += vg * (k & 1 ? 1.0f : -1.0f) * wz * wy;
    }
    const float d[3] = {dz, dy, dx};
    for (int i = 0; i < 3; ++i)
        gp[i] = inside[i] ? d[i] * (float)(sizes[i] - 1) : 0.0f;
}

static Sample slab(const Model &m, const Shared &sh, Grads *g, int s) {
    Sample f;
    f.Z = m.hr[0];
    f.Y = m.hr[1];
    f.X = m.hr[2];
    f.cells = m.hr_cells();
    f.low = sh.low_up.data() + (size_t)s * 3 * f.cells;
    f.high = m.high.data() + (size_t)s * 3 * f.cells;
    f.acc = g ? g->field.data() + (size_t)s * 3 * f.cells : nullptr;
    return f;
}

static void rk4(const Sample &f, float y[3], float h, int steps) {
    for (int i = 0; i < steps; ++i) {
        float k1[3], k2[3], k3[3], k4[3], x2[3], x3[3], x4[3];
        velocity(f, y, k1);
        for (int c = 0; c < 3; ++c)
            x2[c] = y[c] + (h / 2) * k1[c];
        velocity(f, x2, k2);
        for (int c = 0; c < 3; ++c)
            x3[c] = y[c] + (h / 2) * k2[c];
        velocity(f, x3, k3);
        for (int c = 0; c < 3; ++c)
            x4[c] = y[c] + h * k3[c];
        velocity(f, x4, k4);
        for (int c = 0; c < 3; ++c)
            y[c] = y[c] + (h / 6) * (k1[c] + 2 * k2[c] + 2 * k3[c] + k4[c]);
    }
}

static void rk4_backward(const Sample &f, const float y0[3], float h, int steps, float g[3]) {
    std::vector<std::array<float, 3>> stages((size_t)steps * 4);
    float y[3] = {y0[0], y0[1], y0[2]};
    for (int i = 0; i < steps; ++i) {
        float k1[3], k2[3], k3[3], k4[3], x2[3], x3[3], x4[3];
        velocity(f, y, k1);
        for (int c = 0; c < 3; ++c)
            x2[c] = y[c] + (h / 2) * k1[c];
        velocity(f, x2, k2);
        for (int c = 0; c < 3; ++c)
            x3[c] = y[c] + (h / 2) * k2[c];
        velocity(f, x3, k3);
        for (int c = 0; c < 3; ++c)
            x4[c] = y[c] + h * k3[c];
        velocity(f, x4, k4);
        for (int c = 0; c < 3; ++c) {
            stages[(size_t)i * 4][c] = y[c];
            stages[(size_t)i * 4 + 1][c] = x2[c];
            stages[(size_t)i * 4 + 2][c] = x3[c];
            stages[(size_t)i * 4 + 3][c] = x4[c];
        }
        for (int c = 0; c < 3; ++c)
            y[c] = y[c] + (h / 6) * (k1[c] + 2 * k2[c] + 2 * k3[c] + k4[c]);
    }
    for (int i = steps - 1; i >= 0; --i) {
        float g6[3], in[3], b4[3], b3[3], b2[3], b1[3];
        for (int c = 0; c < 3; ++c)
            g6[c] = g[c] * (h / 6);
        velocity_backward(f, stages[(size_t)i * 4 + 3].data(), g6, b4);
        for (int c = 0; c < 3; ++c)
            in[c] = g6[c] * 2 + b4[c] * h;
        velocity_backward(f, stages[(size_t)i * 4 + 2].data(), in, b3);
        for (int c = 0; c < 3; ++c)
            in[c] = g6[c] * 2 + b3[c] * (h / 2);
        velocity_backward(f, stages[(size_t)i * 4 + 1].data(), in, b2);
        for (int c = 0; c < 3; ++c)
            in[c] = g6[c] + b2[c] * (h / 2);
        velocity_backward(f, stages[(size_t)i * 4].data(), in, b1);
        for (int c = 0; c < 3; ++c)
            g[c] = (((g[c] + b4[c]) + b3[c]) + b2[c]) + b1[c];
    }
}

static void flow(const Model &m, const Shared &sh, float p[3], bool inverse) {
    float y[3];
    for (int c = 0; c < 3; ++c)
        y[c] = (p[c] - (float)m.flow_min[c]) / m.range[c];
    const float h = (inverse ? -1.0f : 1.0f) / (float)m.s.rk4_steps;
    for (int k = 0; k < m.s.stages; ++k) {
        const int s = inverse ? m.s.stages - 1 - k : k;
        rk4(slab(m, sh, nullptr, s), y, h, m.s.rk4_steps);
    }
    for (int c = 0; c < 3; ++c)
        p[c] = y[c] * m.range[c] + (float)m.flow_min[c];
}

static void flow_backward(
    const Model &m, const Shared &sh, const float p[3], bool inverse, float g[3], Grads *gr) {
    float y[3];
    for (int c = 0; c < 3; ++c)
        y[c] = (p[c] - (float)m.flow_min[c]) / m.range[c];
    const float h = (inverse ? -1.0f : 1.0f) / (float)m.s.rk4_steps;
    std::vector<std::array<float, 3>> inputs((size_t)m.s.stages);
    for (int k = 0; k < m.s.stages; ++k) {
        const int s = inverse ? m.s.stages - 1 - k : k;
        inputs[(size_t)k] = {y[0], y[1], y[2]};
        rk4(slab(m, sh, nullptr, s), y, h, m.s.rk4_steps);
    }
    float gy[3];
    for (int c = 0; c < 3; ++c)
        gy[c] = g[c] * m.range[c];
    for (int k = m.s.stages - 1; k >= 0; --k) {
        const int s = inverse ? m.s.stages - 1 - k : k;
        rk4_backward(slab(m, sh, gr, s), inputs[(size_t)k].data(), h, m.s.rk4_steps, gy);
    }
    for (int c = 0; c < 3; ++c)
        g[c] = gy[c] / m.range[c];
}

static inline void expm2(const float L[4], float M[4]) {
    const float a = L[0], b = L[1], c = L[2], d = L[3];
    const float mm = 0.5f * (a + d);
    const float s2 = (0.5f * (a - d)) * (0.5f * (a - d)) + b * c;
    const bool small = std::fabs(s2) < 1e-8f;
    const float s = std::sqrt(std::fabs(small ? 1.0f : s2));
    const float ch = small ? 1.0f + s2 / 2.0f : (s2 >= 0 ? std::cosh(s) : std::cos(s));
    const float sn = small ? 1.0f + s2 / 6.0f : (s2 >= 0 ? std::sinh(s) : std::sin(s)) / s;
    const float em = std::exp(mm);
    const float fd = em * ch, fo = em * sn;
    M[0] = fd + fo * (a - mm);
    M[1] = fo * b;
    M[2] = fo * c;
    M[3] = fd + fo * (d - mm);
}

static inline void expm2_backward(const float L[4], const float G[4], float gL[4]) {
    const double a = L[0], b = L[1], c = L[2], d = L[3];
    const double mm = 0.5 * (a + d), h = 0.5 * (a - d);
    const double s2 = h * h + b * c;
    const bool small = std::fabs((float)s2) < 1e-8;
    double ch, sn, dch, dsn;
    if (small) {
        ch = 1 + s2 / 2;
        sn = 1 + s2 / 6;
        dch = 0.5;
        dsn = 1.0 / 6.0;
    } else {
        const double s = std::sqrt(std::fabs(s2));
        ch = s2 >= 0 ? std::cosh(s) : std::cos(s);
        sn = (s2 >= 0 ? std::sinh(s) : std::sin(s)) / s;
        dch = sn / 2;
        dsn = (ch - sn) / (2 * s2);
    }
    const double em = std::exp(mm);
    const double fd = em * ch, fo = em * sn;
    const double gfd = G[0] + G[3];
    const double gfo = G[0] * (a - mm) + G[1] * b + G[2] * c + G[3] * (d - mm);
    double ga = G[0] * fo * 0.5 - G[3] * fo * 0.5, gb = G[1] * fo, gc = G[2] * fo,
           gd = -G[0] * fo * 0.5 + G[3] * fo * 0.5;
    const double gm = gfd * fd + gfo * fo;
    const double gs2 = gfd * em * dch + gfo * em * dsn;
    ga += gm * 0.5 + gs2 * h;
    gd += gm * 0.5 - gs2 * h;
    gb += gs2 * c;
    gc += gs2 * b;
    gL[0] = (float)ga;
    gL[1] = (float)gb;
    gL[2] = (float)gc;
    gL[3] = (float)gd;
}

struct LinearAt {
    int lo;
    float frac, dcoord;
};

static inline LinearAt linear_at(const Model &m, float z) {
    const int Z = m.linear_z;
    const float zn = (z - (float)m.flow_min[0]) / (float)(m.flow_max[0] - m.flow_min[0]);
    const float raw = (zn * 2 - 1 + 1) / 2 * (float)(Z - 1);
    const float coord = std::min(std::max(raw, 0.0f), (float)(Z - 1));
    float lo = std::floor(coord);
    if (Z > 1)
        lo = std::min(lo, (float)(Z - 2));
    LinearAt at;
    at.lo = (int)lo;
    at.frac = coord - lo;
    at.dcoord = (raw >= 0.0f && raw <= (float)(Z - 1))
        ? (float)(Z - 1) / (float)(m.flow_max[0] - m.flow_min[0])
        : 0.0f;
    return at;
}

static void linear_map(const Model &m, const Shared &sh, float p[3], bool inverse) {
    const LinearAt at = linear_at(m, p[0]);
    float L[4], M[4];
    for (int k = 0; k < 4; ++k) {
        const float a = sh.linear[(size_t)at.lo * 4 + k];
        const float b = m.linear_z > 1 ? sh.linear[(size_t)(at.lo + 1) * 4 + k] : a;
        L[k] = m.linear_z > 1 ? lerp(a, b, at.frac) : a;
        if (inverse)
            L[k] = -L[k];
    }
    expm2(L, M);
    const float y = p[1], x = p[2];
    p[1] = M[0] * y + M[1] * x;
    p[2] = M[2] * y + M[3] * x;
}

static void linear_backward(
    const Model &m, const Shared &sh, const float p[3], bool inverse, float g[3], Grads *gr) {
    const LinearAt at = linear_at(m, p[0]);
    float L[4], M[4], A[4], B[4];
    for (int k = 0; k < 4; ++k) {
        A[k] = sh.linear[(size_t)at.lo * 4 + k];
        B[k] = m.linear_z > 1 ? sh.linear[(size_t)(at.lo + 1) * 4 + k] : A[k];
        L[k] = m.linear_z > 1 ? lerp(A[k], B[k], at.frac) : A[k];
        if (inverse)
            L[k] = -L[k];
    }
    expm2(L, M);
    const float y = p[1], x = p[2];
    const float gy = g[1], gx = g[2];
    const float GM[4] = {gy * y, gy * x, gx * y, gx * x};
    float gL[4];
    expm2_backward(L, GM, gL);
    float gz = g[0];
    for (int k = 0; k < 4; ++k) {
        const float gl = inverse ? -gL[k] : gL[k];
        float ga, gb, gw;
        lerp_grad(A[k], B[k], at.frac, gl, ga, gb, gw);
        if (gr) {
            if (m.linear_z > 1) {
                atomic_add(&gr->linear[(size_t)at.lo * 4 + k], (double)ga);
                atomic_add(&gr->linear[(size_t)(at.lo + 1) * 4 + k], (double)gb);
            } else {
                atomic_add(&gr->linear[(size_t)k], (double)gl);
            }
        }
        gz += gw * at.dcoord;
    }
    g[0] = gz;
    g[1] = M[0] * gy + M[2] * gx;
    g[2] = M[1] * gy + M[3] * gx;
}

static inline float centre_segment(const Model &m, float z, int &idx) {
    const auto &xp = m.umbilicus_z;
    const int n = (int)xp.size();
    int i = (int)(std::lower_bound(xp.begin(), xp.end(), z) - xp.begin()) - 1;
    idx = std::min(std::max(i, 0), n - 2);
    return 0;
}

static inline void centre(const Model &m, float z, float c[2], float dc[2]) {
    int i;
    centre_segment(m, z, i);
    const float x0 = m.umbilicus_z[(size_t)i], x1 = m.umbilicus_z[(size_t)i + 1];
    for (int k = 0; k < 2; ++k) {
        const float f0 = m.umbilicus_yx[(size_t)i][k], f1 = m.umbilicus_yx[(size_t)i + 1][k];
        const float slope = (f1 - f0) / (x1 - x0);
        const float b = f0 - slope * x0;
        c[k] = slope * z + b;
        dc[k] = slope;
    }
}

static void umbilicus_map(const Model &m, float p[3], bool inverse) {
    float c[2], dc[2];
    centre(m, p[0], c, dc);
    p[1] = inverse ? p[1] - c[0] : p[1] + c[0];
    p[2] = inverse ? p[2] - c[1] : p[2] + c[1];
}

static void umbilicus_backward(const Model &m, const float p[3], bool inverse, float g[3]) {
    float c[2], dc[2];
    centre(m, p[0], c, dc);
    const float sign = inverse ? -1.0f : 1.0f;
    g[0] += sign * (dc[0] * g[1] + dc[1] * g[2]);
}

struct Polar {
    float theta, radius, xg;
    bool guard;
};

static inline Polar polar(float y, float x) {
    Polar q;
    q.guard = std::fabs(x) < 1e-10f;
    q.xg = q.guard ? 1e-10f : x;
    const float t = std::atan2(y, q.xg);
    q.theta = t - TWO_PI * std::floor(t / TWO_PI);
    q.radius = std::sqrt(y * y + q.xg * q.xg);
    return q;
}

struct GapLookup {
    float zpix;
    int z0, z1;
    float wz;
    bool zinside;
};

static inline GapLookup gap_z(const Model &m, float z) {
    GapLookup l;
    const float gz = (z - (float)m.flow_min[0]) / (float)(m.flow_max[0] - m.flow_min[0]) * 2 - 1;
    float pix = (gz + 1) * 0.5f * (float)(m.gap_num_z - 1);
    l.zinside = !(pix < 0 || pix > (float)(m.gap_num_z - 1));
    pix = std::min(std::max(pix, 0.0f), (float)(m.gap_num_z - 1));
    l.zpix = pix;
    l.z0 = (int)std::floor(pix);
    l.wz = pix - (float)l.z0;
    l.z1 = std::min(l.z0 + 1, m.gap_num_z - 1);
    return l;
}

static inline float logit_at(const Model &m, const Shared &sh, const GapLookup &lz, int j, float tn,
    float &dl_dtn, float &dl_dz, int &x0, int &x1, float &wx) {
    const float first = m.first_index[(size_t)j], next = m.first_index[(size_t)j + 1];
    const float total = m.first_index[(size_t)m.windings];
    const float coord = lerp(first, next, tn);
    const float gx = coord / total * 2 - 1;
    float pix = (gx + 1) * 0.5f * (float)(m.gap_width - 1);
    const bool xinside = !(pix < 0 || pix > (float)(m.gap_width - 1));
    pix = std::min(std::max(pix, 0.0f), (float)(m.gap_width - 1));
    x0 = (int)std::floor(pix);
    wx = pix - (float)x0;
    x1 = std::min(x0 + 1, m.gap_width - 1);
    const float *g = sh.gap.data();
    const size_t r0 = (size_t)lz.z0 * m.gap_width, r1 = (size_t)lz.z1 * m.gap_width;
    const float v00 = g[r0 + x0], v01 = g[r0 + x1], v10 = g[r1 + x0], v11 = g[r1 + x1];
    const float value = ((1 - wx) * (1 - lz.wz)) * v00 + (wx * (1 - lz.wz)) * v01 +
        ((1 - wx) * lz.wz) * v10 + (wx * lz.wz) * v11;
    const float dpix = xinside ? ((1 - lz.wz) * (v01 - v00) + lz.wz * (v11 - v10)) : 0.0f;
    dl_dtn = dpix * (float)(m.gap_width - 1) / total * (next - first);
    dl_dz = lz.zinside ? ((1 - wx) * (v10 - v00) + wx * (v11 - v01)) * (float)(m.gap_num_z - 1) /
            (float)(m.flow_max[0] - m.flow_min[0])
                       : 0.0f;
    return value;
}

static inline void logit_backward(
    const Model &m, const GapLookup &lz, int x0, int x1, float wx, float g, Grads *gr) {
    if (!gr || g == 0.0f)
        return;
    const size_t r0 = (size_t)lz.z0 * m.gap_width, r1 = (size_t)lz.z1 * m.gap_width;
    atomic_add(&gr->gap[r0 + x0], (double)(g * (1 - wx) * (1 - lz.wz)));
    atomic_add(&gr->gap[r0 + x1], (double)(g * wx * (1 - lz.wz)));
    atomic_add(&gr->gap[r1 + x0], (double)(g * (1 - wx) * lz.wz));
    atomic_add(&gr->gap[r1 + x1], (double)(g * wx * lz.wz));
}

struct Bracket {
    float r_in, r_out;
    int index;
};

static void winding_radii(
    const Model &m, const Shared &sh, float theta, float z, int count, std::vector<float> &R) {
    const float tn = theta / TWO_PI;
    const GapLookup lz = gap_z(m, z);
    R.resize((size_t)count);
    float sum = 0, dt, dz;
    int x0, x1;
    float wx;
    const float zero = sh.dr * tn;
    R[0] = zero + 0.0f;
    for (int j = 0; j + 1 < count; ++j) {
        const float l = logit_at(m, sh, lz, j, tn, dt, dz, x0, x1, wx);
        const float gap = (float)m.s.min_gap +
            (sh.dr - (float)m.s.min_gap) *
                (softplus((float)m.s.softplus_bias + m.softplus_scale * l) /
                    m.softplus_denominator);
        sum += gap;
        R[(size_t)j + 1] = zero + sum;
    }
}

static void radii_backward(const Model &m, const Shared &sh, float theta, float z, int index,
    float g_in, float g_out, float &g_theta, float &g_z, double &g_dr, Grads *gr) {
    const float tn = theta / TWO_PI;
    const GapLookup lz = gap_z(m, z);
    g_theta += (g_in + g_out) * sh.dr / TWO_PI;
    g_dr += (double)((g_in + g_out) * tn);
    for (int j = 0; j <= index; ++j) {
        const float gg = (j < index ? g_in : 0.0f) + g_out;
        if (gg == 0.0f)
            continue;
        float dt, dz, wx;
        int x0, x1;
        const float l = logit_at(m, sh, lz, j, tn, dt, dz, x0, x1, wx);
        const float u = (float)m.s.softplus_bias + m.softplus_scale * l;
        const float ratio = softplus(u) / m.softplus_denominator;
        const float dgap_dl =
            (sh.dr - (float)m.s.min_gap) * sigmoid(u) * m.softplus_scale / m.softplus_denominator;
        g_dr += (double)(gg * ratio);
        const float gl = gg * dgap_dl;
        logit_backward(m, lz, x0, x1, wx, gl, gr);
        g_theta += gl * dt / TWO_PI;
        g_z += gl * dz;
    }
}

static void gap_forward_map(const Model &m, const Shared &sh, float p[3], std::vector<float> &R) {
    const Polar q = polar(p[1], p[2]);
    const float shifted = std::max(q.radius - q.theta / TWO_PI * sh.dr, 0.0f);
    int inner = (int)std::floor(shifted / sh.dr);
    inner = std::min(std::max(inner, 0), m.windings - 2);
    winding_radii(m, sh, q.theta, p[0], inner + 2, R);
    const float oi = ((float)inner + q.theta / TWO_PI) * sh.dr;
    const float oo = oi + sh.dr;
    const float frac = (q.radius - oi) / (oo - oi);
    const float rt = lerp(R[(size_t)inner], R[(size_t)inner + 1], frac);
    const float delta = rt - q.radius;
    const float norm = std::max(std::sqrt(p[1] * p[1] + p[2] * p[2]), 1e-12f);
    p[1] = p[1] + p[1] / norm * delta;
    p[2] = p[2] + p[2] / norm * delta;
}

static void gap_inverse_map(const Model &m, const Shared &sh, float p[3], std::vector<float> &R) {
    const Polar q = polar(p[1], p[2]);
    winding_radii(m, sh, q.theta, p[0], m.windings, R);
    int idx = (int)(std::lower_bound(R.begin(), R.end(), q.radius) - R.begin()) - 1;
    idx = std::min(std::max(idx, 0), m.windings - 2);
    const float oi = ((float)idx + q.theta / TWO_PI) * sh.dr;
    const float oo = oi + sh.dr;
    const float frac = (q.radius - R[(size_t)idx]) / (R[(size_t)idx + 1] - R[(size_t)idx]);
    const float ro = lerp(oi, oo, frac);
    const float delta = ro - q.radius;
    const float norm = std::max(std::sqrt(p[1] * p[1] + p[2] * p[2]), 1e-12f);
    p[1] = p[1] + p[1] / norm * delta;
    p[2] = p[2] + p[2] / norm * delta;
}

static void outward_backward(const float y, const float x, float delta, const float g[3], float &gy,
    float &gx, float &gdelta) {
    const float n = std::sqrt(y * y + x * x);
    const float norm = std::max(n, 1e-12f);
    const float uy = y / norm, ux = x / norm;
    gdelta = g[1] * uy + g[2] * ux;
    if (n > 1e-12f) {
        const float dot = g[1] * uy + g[2] * ux;
        gy = g[1] + delta * (g[1] - uy * dot) / norm;
        gx = g[2] + delta * (g[2] - ux * dot) / norm;
    } else {
        gy = g[1] + delta * g[1] / norm;
        gx = g[2] + delta * g[2] / norm;
    }
}

static void polar_backward(
    const Polar &q, float y, float gtheta, float gradius, float &gy, float &gx) {
    const float r2 = y * y + q.xg * q.xg;
    gy += gtheta * (q.xg / r2) + gradius * (y / q.radius);
    if (!q.guard)
        gx += gtheta * (-y / r2) + gradius * (q.xg / q.radius);
}

static void gap_forward_backward(
    const Model &m, const Shared &sh, const float p[3], float g[3], Grads *gr, double &g_dr) {
    const Polar q = polar(p[1], p[2]);
    const float shifted = std::max(q.radius - q.theta / TWO_PI * sh.dr, 0.0f);
    int inner = (int)std::floor(shifted / sh.dr);
    inner = std::min(std::max(inner, 0), m.windings - 2);
    std::vector<float> R;
    winding_radii(m, sh, q.theta, p[0], inner + 2, R);
    const float tn = q.theta / TWO_PI;
    const float oi = ((float)inner + tn) * sh.dr;
    const float oo = oi + sh.dr;
    const float den = oo - oi;
    const float frac = (q.radius - oi) / den;
    const float Ri = R[(size_t)inner], Ro = R[(size_t)inner + 1];
    const float rt = lerp(Ri, Ro, frac);
    const float delta = rt - q.radius;
    float gy, gx, gdelta;
    outward_backward(p[1], p[2], delta, g, gy, gx, gdelta);
    float gRi, gRo, gfrac;
    lerp_grad(Ri, Ro, frac, gdelta, gRi, gRo, gfrac);
    float gradius = -gdelta + gfrac / den;
    const float goi = -gfrac / den;
    float gtheta = goi * sh.dr / TWO_PI;
    g_dr += (double)(goi * ((float)inner + tn)) - (double)(gfrac * (q.radius - oi) / (den * den));
    float gz = g[0];
    radii_backward(m, sh, q.theta, p[0], inner, gRi, gRo, gtheta, gz, g_dr, gr);
    polar_backward(q, p[1], gtheta, gradius, gy, gx);
    g[0] = gz;
    g[1] = gy;
    g[2] = gx;
}

static void gap_inverse_backward(
    const Model &m, const Shared &sh, const float p[3], float g[3], Grads *gr, double &g_dr) {
    const Polar q = polar(p[1], p[2]);
    std::vector<float> R;
    winding_radii(m, sh, q.theta, p[0], m.windings, R);
    int idx = (int)(std::lower_bound(R.begin(), R.end(), q.radius) - R.begin()) - 1;
    idx = std::min(std::max(idx, 0), m.windings - 2);
    const float tn = q.theta / TWO_PI;
    const float oi = ((float)idx + tn) * sh.dr;
    const float oo = oi + sh.dr;
    const float Ri = R[(size_t)idx], Ro = R[(size_t)idx + 1];
    const float den = Ro - Ri;
    const float frac = (q.radius - Ri) / den;
    const float ro = lerp(oi, oo, frac);
    const float delta = ro - q.radius;
    float gy, gx, gdelta;
    outward_backward(p[1], p[2], delta, g, gy, gx, gdelta);
    float goi, goo, gfrac;
    lerp_grad(oi, oo, frac, gdelta, goi, goo, gfrac);
    float gradius = -gdelta + gfrac / den;
    const float gRi = gfrac * (-(den) + (q.radius - Ri)) / (den * den) * 1.0f;
    const float gRo = -gfrac * (q.radius - Ri) / (den * den);
    const float goi_total = goi + goo;
    float gtheta = goi_total * sh.dr / TWO_PI;
    g_dr += (double)(goi_total * ((float)idx + tn) + goo);
    float gz = g[0];
    radii_backward(m, sh, q.theta, p[0], idx, gRi, gRo, gtheta, gz, g_dr, gr);
    polar_backward(q, p[1], gtheta, gradius, gy, gx);
    g[0] = gz;
    g[1] = gy;
    g[2] = gx;
}

static inline void flip_x(float p[3]) { p[2] = -p[2]; }

static void to_flow(const Model &m, const float q[3], float *y) {
    for (int c = 0; c < 3; ++c)
        y[c] = (q[c] - (float)m.flow_min[c]) / m.range[c];
}

static void from_flow(const Model &m, const float *y, float q[3]) {
    for (int c = 0; c < 3; ++c)
        q[c] = y[c] * m.range[c] + (float)m.flow_min[c];
}

static void slice_to_spiral_gpu(
    const Model &m, const Shared &sh, const float *p, size_t n, float *out) {
    std::vector<float> y(n * 3);
    parallel(n, 1024, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            umbilicus_map(m, q, true);
            linear_map(m, sh, q, true);
            to_flow(m, q, &y[i * 3]);
        }
    });
    gpu::flow(m, sh, y.data(), n, true, nullptr);
    parallel(n, 256, [&](size_t a, size_t b) {
        std::vector<float> R;
        for (size_t i = a; i < b; ++i) {
            float q[3];
            from_flow(m, &y[i * 3], q);
            if (m.s.anticlockwise)
                flip_x(q);
            gap_inverse_map(m, sh, q, R);
            for (int c = 0; c < 3; ++c)
                out[i * 3 + c] = q[c];
        }
    });
}

static void spiral_to_slice_gpu(
    const Model &m, const Shared &sh, const float *p, size_t n, float *out) {
    std::vector<float> y(n * 3);
    parallel(n, 256, [&](size_t a, size_t b) {
        std::vector<float> R;
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            gap_forward_map(m, sh, q, R);
            if (m.s.anticlockwise)
                flip_x(q);
            to_flow(m, q, &y[i * 3]);
        }
    });
    gpu::flow(m, sh, y.data(), n, false, nullptr);
    parallel(n, 1024, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            float q[3];
            from_flow(m, &y[i * 3], q);
            linear_map(m, sh, q, false);
            umbilicus_map(m, q, false);
            for (int c = 0; c < 3; ++c)
                out[i * 3 + c] = q[c];
        }
    });
}

static void add_dr(std::atomic<double> &total, double v) {
    double cur = total.load();
    while (!total.compare_exchange_weak(cur, cur + v)) {
    }
}

static void slice_to_spiral_backward_gpu(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters) {
    const size_t S = (size_t)m.s.stages;
    std::vector<float> s1(n * 3), y(n * 3), inputs(S * n * 3), gy(n * 3);
    parallel(n, 1024, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            umbilicus_map(m, q, true);
            for (int c = 0; c < 3; ++c)
                s1[i * 3 + c] = q[c];
            linear_map(m, sh, q, true);
            to_flow(m, q, &y[i * 3]);
        }
    });
    gpu::flow(m, sh, y.data(), n, true, inputs.data());
    std::atomic<double> dr_total{0.0};
    Grads *gr = parameters ? &g : nullptr;
    parallel(n, 128, [&](size_t a, size_t b) {
        double dr_local = 0;
        for (size_t i = a; i < b; ++i) {
            float q[3];
            from_flow(m, &y[i * 3], q);
            if (m.s.anticlockwise)
                flip_x(q);
            float gq[3] = {grad_out[i * 3], grad_out[i * 3 + 1], grad_out[i * 3 + 2]};
            gap_inverse_backward(m, sh, q, gq, gr, dr_local);
            if (m.s.anticlockwise)
                gq[2] = -gq[2];
            for (int c = 0; c < 3; ++c)
                gy[i * 3 + c] = gq[c] * m.range[c];
        }
        add_dr(dr_total, dr_local);
    });
    Field scratch;
    if (!parameters)
        scratch.assign((size_t)m.s.stages * 3 * m.hr_cells(), 0.0f);
    gpu::flow_backward(m, sh, inputs.data(), gy.data(), n, true, parameters ? g.field : scratch);
    parallel(n, 1024, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            float gq[3];
            for (int c = 0; c < 3; ++c)
                gq[c] = gy[i * 3 + c] / m.range[c];
            linear_backward(m, sh, &s1[i * 3], true, gq, gr);
            umbilicus_backward(m, &p[i * 3], true, gq);
            if (grad_p)
                for (int c = 0; c < 3; ++c)
                    grad_p[i * 3 + c] = gq[c];
        }
    });
    if (parameters)
        g.dr += dr_total.load();
}

static void spiral_to_slice_backward_gpu(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters) {
    const size_t S = (size_t)m.s.stages;
    std::vector<float> y(n * 3), inputs(S * n * 3), gy(n * 3);
    parallel(n, 256, [&](size_t a, size_t b) {
        std::vector<float> R;
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            gap_forward_map(m, sh, q, R);
            if (m.s.anticlockwise)
                flip_x(q);
            to_flow(m, q, &y[i * 3]);
        }
    });
    gpu::flow(m, sh, y.data(), n, false, inputs.data());
    Grads *gr = parameters ? &g : nullptr;
    parallel(n, 1024, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            float s2[3];
            from_flow(m, &y[i * 3], s2);
            float s3[3] = {s2[0], s2[1], s2[2]};
            linear_map(m, sh, s3, false);
            float gq[3] = {grad_out[i * 3], grad_out[i * 3 + 1], grad_out[i * 3 + 2]};
            umbilicus_backward(m, s3, false, gq);
            linear_backward(m, sh, s2, false, gq, gr);
            for (int c = 0; c < 3; ++c)
                gy[i * 3 + c] = gq[c] * m.range[c];
        }
    });
    Field scratch;
    if (!parameters)
        scratch.assign((size_t)m.s.stages * 3 * m.hr_cells(), 0.0f);
    gpu::flow_backward(m, sh, inputs.data(), gy.data(), n, false, parameters ? g.field : scratch);
    std::atomic<double> dr_total{0.0};
    parallel(n, 128, [&](size_t a, size_t b) {
        double dr_local = 0;
        for (size_t i = a; i < b; ++i) {
            float gq[3];
            for (int c = 0; c < 3; ++c)
                gq[c] = gy[i * 3 + c] / m.range[c];
            if (m.s.anticlockwise)
                gq[2] = -gq[2];
            gap_forward_backward(m, sh, &p[i * 3], gq, gr, dr_local);
            if (grad_p)
                for (int c = 0; c < 3; ++c)
                    grad_p[i * 3 + c] = gq[c];
        }
        add_dr(dr_total, dr_local);
    });
    if (parameters)
        g.dr += dr_total.load();
}

void slice_to_spiral(const Model &m, const Shared &sh, const float *p, size_t n, float *out) {
    if (gpu::points())
        return slice_to_spiral_gpu(m, sh, p, n, out);
    parallel(n, 256, [&](size_t a, size_t b) {
        std::vector<float> R;
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            umbilicus_map(m, q, true);
            linear_map(m, sh, q, true);
            flow(m, sh, q, true);
            if (m.s.anticlockwise)
                flip_x(q);
            gap_inverse_map(m, sh, q, R);
            for (int c = 0; c < 3; ++c)
                out[i * 3 + c] = q[c];
        }
    });
}

void spiral_to_slice(const Model &m, const Shared &sh, const float *p, size_t n, float *out) {
    if (gpu::points())
        return spiral_to_slice_gpu(m, sh, p, n, out);
    parallel(n, 256, [&](size_t a, size_t b) {
        std::vector<float> R;
        for (size_t i = a; i < b; ++i) {
            float q[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            gap_forward_map(m, sh, q, R);
            if (m.s.anticlockwise)
                flip_x(q);
            flow(m, sh, q, false);
            linear_map(m, sh, q, false);
            umbilicus_map(m, q, false);
            for (int c = 0; c < 3; ++c)
                out[i * 3 + c] = q[c];
        }
    });
}

void slice_to_spiral_backward(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters) {
    if (gpu::points())
        return slice_to_spiral_backward_gpu(m, sh, p, n, grad_out, grad_p, g, parameters);
    std::atomic<double> dr_total{0.0};
    parallel(n, 128, [&](size_t a, size_t b) {
        std::vector<float> R;
        double dr_local = 0;
        Grads *gr = parameters ? &g : nullptr;
        for (size_t i = a; i < b; ++i) {
            float s0[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            float s1[3] = {s0[0], s0[1], s0[2]};
            umbilicus_map(m, s1, true);
            float s2[3] = {s1[0], s1[1], s1[2]};
            linear_map(m, sh, s2, true);
            float s3[3] = {s2[0], s2[1], s2[2]};
            flow(m, sh, s3, true);
            if (m.s.anticlockwise)
                flip_x(s3);
            float gq[3] = {grad_out[i * 3], grad_out[i * 3 + 1], grad_out[i * 3 + 2]};
            gap_inverse_backward(m, sh, s3, gq, gr, dr_local);
            if (m.s.anticlockwise)
                gq[2] = -gq[2];
            flow_backward(m, sh, s2, true, gq, gr);
            linear_backward(m, sh, s1, true, gq, gr);
            umbilicus_backward(m, s0, true, gq);
            if (grad_p)
                for (int c = 0; c < 3; ++c)
                    grad_p[i * 3 + c] = gq[c];
        }
        double cur = dr_total.load();
        while (!dr_total.compare_exchange_weak(cur, cur + dr_local)) {
        }
    });
    if (parameters)
        g.dr += dr_total.load();
}

void spiral_to_slice_backward(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters) {
    if (gpu::points())
        return spiral_to_slice_backward_gpu(m, sh, p, n, grad_out, grad_p, g, parameters);
    std::atomic<double> dr_total{0.0};
    parallel(n, 128, [&](size_t a, size_t b) {
        std::vector<float> R;
        double dr_local = 0;
        Grads *gr = parameters ? &g : nullptr;
        for (size_t i = a; i < b; ++i) {
            float s0[3] = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            float s1[3] = {s0[0], s0[1], s0[2]};
            gap_forward_map(m, sh, s1, R);
            if (m.s.anticlockwise)
                flip_x(s1);
            float s2[3] = {s1[0], s1[1], s1[2]};
            flow(m, sh, s2, false);
            float s3[3] = {s2[0], s2[1], s2[2]};
            linear_map(m, sh, s3, false);
            float gq[3] = {grad_out[i * 3], grad_out[i * 3 + 1], grad_out[i * 3 + 2]};
            umbilicus_backward(m, s3, false, gq);
            linear_backward(m, sh, s2, false, gq, gr);
            flow_backward(m, sh, s1, false, gq, gr);
            if (m.s.anticlockwise)
                gq[2] = -gq[2];
            gap_forward_backward(m, sh, s0, gq, gr, dr_local);
            if (grad_p)
                for (int c = 0; c < 3; ++c)
                    grad_p[i * 3 + c] = gq[c];
        }
        double cur = dr_total.load();
        while (!dr_total.compare_exchange_weak(cur, cur + dr_local)) {
        }
    });
    if (parameters)
        g.dr += dr_total.load();
}

float native_log_gap(const Model &m, const Shared &sh, int winding, float theta, float z) {
    const GapLookup lz = gap_z(m, z);
    const int j = std::min(std::max(winding, 0), m.windings - 1);
    float dt, dz, wx;
    int x0, x1;
    const float l = logit_at(m, sh, lz, j, theta / TWO_PI, dt, dz, x0, x1, wx);
    const float gap = (float)m.s.min_gap +
        (sh.dr - (float)m.s.min_gap) *
            (softplus((float)m.s.softplus_bias + m.softplus_scale * l) / m.softplus_denominator);
    return std::log(gap);
}

void native_log_gap_backward(
    const Model &m, const Shared &sh, int winding, float theta, float z, float grad, Grads &g) {
    const GapLookup lz = gap_z(m, z);
    const int j = std::min(std::max(winding, 0), m.windings - 1);
    float dt, dz, wx;
    int x0, x1;
    const float l = logit_at(m, sh, lz, j, theta / TWO_PI, dt, dz, x0, x1, wx);
    const float u = (float)m.s.softplus_bias + m.softplus_scale * l;
    const float gap =
        (float)m.s.min_gap + (sh.dr - (float)m.s.min_gap) * (softplus(u) / m.softplus_denominator);
    const float dgap =
        (sh.dr - (float)m.s.min_gap) * sigmoid(u) * m.softplus_scale / m.softplus_denominator;
    logit_backward(m, lz, x0, x1, wx, grad / gap * dgap, &g);
}

void finish_gradients(const Model &m, const Grads &g, double &dr_logit, std::vector<double> &linear,
    std::vector<double> &gap, Field &grad_low, Field &grad_high) {
    const float x = m.dr_logit * (float)m.s.dr_parameter_scale;
    dr_logit = g.dr * (double)sigmoid(x) * m.s.dr_parameter_scale;
    linear.resize(g.linear.size());
    for (size_t i = 0; i < g.linear.size(); ++i)
        linear[i] = g.linear[i] * m.s.linear_logits_scale;
    gap.assign(g.gap.size(), 0.0);
    for (int z = 0; z < m.gap_num_z; ++z)
        for (int w = 1; w < m.gap_width; ++w) {
            const size_t i = (size_t)z * m.gap_width + w;
            gap[i] = g.gap[i] * m.s.gap_lr_scale;
        }
    if (gpu::enabled()) {
        gpu::upsample_adjoint(m, g.field, grad_low);
    } else {
        grad_high = g.field;
        upsample_adjoint(m, g.field, grad_low);
    }
}

}
