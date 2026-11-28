#include "mesh.h"

#include "io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sys/stat.h>

namespace la {

namespace {

struct F3 {
    float x, y, z;
};

F3 ld(const float *p) { return {p[0], p[1], p[2]}; }
F3 operator+(F3 a, F3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
F3 operator-(F3 a, F3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
F3 operator*(F3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
F3 operator/(F3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
float dot(F3 a, F3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float norm(F3 a) { return std::sqrt(dot(a, a)); }
F3 cross(F3 a, F3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
bool finite(F3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }
float comp(F3 a, int i) { return i == 0 ? a.x : i == 1 ? a.y : a.z; }

void push(std::vector<float> &g, const Mesh &m, int d, int h, int w, F3 v) {
    float *p = g.data() + (((size_t)d * m.H + h) * m.W + w) * 3;
    p[0] += v.x;
    p[1] += v.y;
    p[2] += v.z;
}

F3 X(const Mesh &m, int d, int h, int w) { return ld(m.at(d, h, w)); }

F3 unit_backward(F3 e, float r, F3 g) { return g * r - e * (r * r * r * dot(e, g)); }

struct Penalty {
    float value;
    F3 grad;
};

Penalty step_penalty(F3 diff, float t, F3 dir, bool valid) {
    const float length = std::sqrt(dot(diff, diff) + 1.0e-8f), t2 = t * t;
    const F3 toward = diff / length;
    if (valid) {
        const float signed_proj = dot(diff, dir), proj = std::fabs(signed_proj);
        const float s = std::max(t - proj, 0.0f), l = std::max(length - t, 0.0f);
        const float sign = signed_proj > 0.0f ? 1.0f : signed_proj < 0.0f ? -1.0f : 0.0f;
        const F3 g = dir * (-2.0f * s / t2 * sign) + toward * (2.0f * l / t2);
        return {s * s / t2 + l * l / t2, g};
    }
    const float rel = (length - t) / t;
    return {rel * rel, toward * (2.0f * rel / t)};
}

std::pair<F3, bool> unit_dir(F3 diff) {
    const float length = norm(diff);
    const bool valid = length > 1.0e-6f;
    return {valid ? diff / std::max(length, 1.0e-12f) : F3{0, 0, 0}, valid};
}

std::pair<F3, bool> normalised(F3 sum) {
    const float length = norm(sum);
    const bool valid = length > 1.0e-6f;
    return {valid ? sum / std::max(length, 1.0e-12f) : F3{0, 0, 0}, valid};
}

float huber(float r) {
    const float a = std::fabs(r), q = std::min(a, 1.0f), lin = a - q;
    return q * q + 2.0f * lin;
}

float huber_grad(float r) {
    return std::fabs(r) < 1.0f ? 2.0f * r : (r > 0.0f ? 2.0f : r < 0.0f ? -2.0f : 0.0f);
}

F3 quad_normal(F3 p00, F3 p10, F3 p01, F3 p11, double fh, double fw) {
    const float u = (float)fh, v = (float)fw;
    const F3 dh = (p10 - p00) * (1.0f - v) + (p11 - p01) * v;
    const F3 dw = (p01 - p00) * (1.0f - u) + (p11 - p10) * u;
    F3 n = cross(dh, dw);
    float nn = norm(n);
    if ((double)nn < 1.0e-8) {
        n = {0.0f, 0.0f, 1.0f};
        nn = norm(n);
    }
    return n / std::max(nn, 1.0e-8f);
}

F3 bilinear(F3 p00, F3 p10, F3 p01, F3 p11, double fh, double fw) {
    const float u = (float)fh, v = (float)fw;
    return p00 + (p10 - p00) * u + (p01 - p00) * v + (p11 - p10 - p01 + p00) * (u * v);
}

struct Surf {
    const Mesh &m;
    int d;
    F3 p(int h, int w) const { return X(m, d, h, w); }
};

F3 seed_ray(F3 seed, F3 ref, F3 fallback_normal) {
    const F3 ray = ref - seed;
    const float rn = norm(ray);
    const F3 fallback = fallback_normal / std::max(norm(fallback_normal), 1.0e-8f);
    if ((double)rn <= 1.0e-6)
        return fallback;
    const F3 unit = ray / std::max(rn, 1.0e-8f);
    if ((double)std::fabs(dot(unit, fallback)) <= 1.0e-3)
        return fallback;
    return unit;
}

bool ray_at(const Surf &s, F3 seed, double h_ref, double w_ref, F3 &ray) {
    const int H = s.m.H, W = s.m.W;
    if (H < 2 || W < 2)
        return false;
    const int row = std::max(0, std::min((int)h_ref, H - 2)),
              col = std::max(0, std::min((int)w_ref, W - 2));
    const double fh = std::max(0.0, std::min(1.0, h_ref - row)),
                 fw = std::max(0.0, std::min(1.0, w_ref - col));
    const F3 a = s.p(row, col), b = s.p(row + 1, col), c = s.p(row, col + 1),
             e = s.p(row + 1, col + 1);
    ray = seed_ray(seed, bilinear(a, b, c, e, fh, fw), quad_normal(a, b, c, e, fh, fw));
    return true;
}

struct Anchor {
    F3 point, normal;
    double h, w;
};

float cross2(F3 v, F3 n, int i, int j) { return comp(v, i) * comp(n, j) - comp(v, j) * comp(n, i); }

bool intersect_quad(F3 o, F3 n, F3 p00, F3 p10, F3 p01, F3 p11, double frac_h, double frac_w,
    double &u, double &v) {
    const F3 a = p10 - p00, b = p01 - p00, c = p11 - p10 - p01 + p00, g = p00 - o;
    const int I[3] = {0, 0, 1}, J[3] = {1, 2, 2};
    float A[3], B[3], C[3], G[3];
    for (int k = 0; k < 3; ++k) {
        A[k] = cross2(a, n, I[k], J[k]);
        B[k] = cross2(b, n, I[k], J[k]);
        C[k] = cross2(c, n, I[k], J[k]);
        G[k] = cross2(g, n, I[k], J[k]);
    }
    const int P[3] = {0, 0, 1}, Q[3] = {1, 2, 2};
    float alpha[3], beta[3], gamma[3];
    for (int k = 0; k < 3; ++k) {
        const int p = P[k], q = Q[k];
        alpha[k] = A[p] * C[q] - A[q] * C[p];
        beta[k] = A[p] * B[q] - A[q] * B[p] + G[p] * C[q] - G[q] * C[p];
        gamma[k] = G[p] * B[q] - G[q] * B[p];
    }
    int best = 0;
    for (int k = 1; k < 3; ++k)
        if (std::fabs((double)alpha[k]) > std::fabs((double)alpha[best]))
            best = k;
    const double af = alpha[best], bf = beta[best], gf = gamma[best], eps = 1.0e-12;
    if (std::fabs(af) < eps)
        u = std::fabs(bf) < eps ? frac_h : -gf / bf;
    else {
        double disc = bf * bf - 4.0 * af * gf;
        if (disc < 0)
            disc = 0.0;
        const double root = std::sqrt(disc), u1 = (-bf + root) / (2.0 * af),
                     u2 = (-bf - root) / (2.0 * af);
        u = std::fabs(u1 - frac_h) <= std::fabs(u2 - frac_h) ? u1 : u2;
    }
    double den[3], num[3];
    int bv = 0;
    for (int k = 0; k < 3; ++k) {
        den[k] = (double)B[k] + u * (double)C[k];
        num[k] = -((double)G[k] + u * (double)A[k]);
    }
    for (int k = 1; k < 3; ++k)
        if (std::fabs(den[k]) > std::fabs(den[bv]))
            bv = k;
    v = std::fabs(den[bv]) < eps ? frac_w : num[bv] / den[bv];
    return true;
}

bool local_update(const Surf &s, F3 seed, double h_ref, double w_ref, F3 ray, Anchor &out) {
    const int H = s.m.H, W = s.m.W;
    if (H < 2 || W < 2 || !finite(seed) || !finite(ray))
        return false;
    const F3 n = ray / std::max(norm(ray), 1.0e-8f);
    int row = std::max(0, std::min((int)h_ref, H - 2)),
        col = std::max(0, std::min((int)w_ref, W - 2));
    double fh = std::max(0.0, std::min(1.0, h_ref - row)),
           fw = std::max(0.0, std::min(1.0, w_ref - col));
    for (int it = 1; it <= 20; ++it) {
        double u, v;
        const F3 a = s.p(row, col), b = s.p(row + 1, col), c = s.p(row, col + 1),
                 e = s.p(row + 1, col + 1);
        intersect_quad(seed, n, a, b, c, e, fh, fw, u, v);
        if (!std::isfinite((float)u) || !std::isfinite((float)v))
            return false;
        const double eps = 1.0e-2;
        if (-eps <= u && u <= 1.0 + eps && -eps <= v && v <= 1.0 + eps) {
            const double uc = std::max(0.0, std::min(1.0, u)), vc = std::max(0.0, std::min(1.0, v));
            out.h = std::max(0.0, std::min((double)(H - 1), row + uc));
            out.w = std::max(0.0, std::min((double)(W - 1), col + vc));
            out.point = bilinear(a, b, c, e, uc, vc);
            out.normal = quad_normal(a, b, c, e, uc, vc);
            return true;
        }
        const int sh = u < 0.0 ? -1 : (u > 1.0 ? 1 : 0), sw = v < 0.0 ? -1 : (v > 1.0 ? 1 : 0);
        const int nr = std::max(0, std::min(H - 2, row + sh)),
                  nc = std::max(0, std::min(W - 2, col + sw));
        if (nr == row && nc == col)
            return false;
        const double th = row + u, tw = col + v;
        row = nr;
        col = nc;
        fh = std::max(0.0, std::min(1.0, th - row));
        fw = std::max(0.0, std::min(1.0, tw - col));
    }
    return false;
}

bool brute_force(const Surf &s, F3 seed, double h_ref, double w_ref, Anchor &out, F3 &ray) {
    const int H = s.m.H, W = s.m.W;
    if (H < 2 || W < 2 || !ray_at(s, seed, h_ref, w_ref, ray))
        return false;
    const float eps = 1.0e-12f, href = (float)h_ref, wref = (float)w_ref;
    const int I[3] = {0, 0, 1}, J[3] = {1, 2, 2};
    float best = std::numeric_limits<float>::infinity();
    int best_r = -1, best_c = -1;
    float best_u = 0, best_v = 0, best_h = 0, best_w = 0;
    for (int r = 0; r + 1 < H; ++r)
        for (int c = 0; c + 1 < W; ++c) {
            const F3 p00 = s.p(r, c), p10 = s.p(r + 1, c), p01 = s.p(r, c + 1),
                     p11 = s.p(r + 1, c + 1);
            const F3 a = p10 - p00, b = p01 - p00, cc = p11 - p10 - p01 + p00, g = p00 - seed;
            float A[3], B[3], C[3], G[3];
            for (int k = 0; k < 3; ++k) {
                A[k] = cross2(a, ray, I[k], J[k]);
                B[k] = cross2(b, ray, I[k], J[k]);
                C[k] = cross2(cc, ray, I[k], J[k]);
                G[k] = cross2(g, ray, I[k], J[k]);
            }
            float al[3], be[3], ga[3];
            for (int k = 0; k < 3; ++k) {
                const int p = I[k], q = J[k];
                al[k] = A[p] * C[q] - A[q] * C[p];
                be[k] = A[p] * B[q] - A[q] * B[p] + G[p] * C[q] - G[q] * C[p];
                ga[k] = G[p] * B[q] - G[q] * B[p];
            }
            const bool q0 =
                std::fabs(al[0]) >= std::fabs(al[1]) && std::fabs(al[0]) >= std::fabs(al[2]);
            const bool q1 = !q0 && std::fabs(al[1]) >= std::fabs(al[2]);
            const int k = q0 ? 0 : q1 ? 1 : 2;
            const float alpha = al[k], beta = be[k], gamma = ga[k];
            const float disc = std::max(beta * beta - 4.0f * alpha * gamma, 0.0f),
                        root = std::sqrt(disc + eps);
            const bool linear = std::fabs(alpha) < eps;
            const float den = 2.0f * alpha + eps * (linear ? 1.0f : 0.0f);
            float u1 = (-beta + root) / den, u2 = (-beta - root) / den;
            const float ulin = -gamma / (beta + eps * (std::fabs(beta) < eps ? 1.0f : 0.0f));
            if (linear)
                u1 = u2 = ulin;
            const float fh0 = std::min(1.0f, std::max(0.0f, href - (float)r)),
                        fw0 = std::min(1.0f, std::max(0.0f, wref - (float)c));
            (void)fw0;
            const float u = std::fabs(u1 - fh0) <= std::fabs(u2 - fh0) ? u1 : u2;
            float dv[3], nv[3];
            for (int j = 0; j < 3; ++j) {
                dv[j] = B[j] + u * C[j];
                nv[j] = -(G[j] + u * A[j]);
            }
            const bool v0 =
                std::fabs(dv[0]) >= std::fabs(dv[1]) && std::fabs(dv[0]) >= std::fabs(dv[2]);
            const bool v1 = !v0 && std::fabs(dv[1]) >= std::fabs(dv[2]);
            const int j = v0 ? 0 : v1 ? 1 : 2;
            const float v = nv[j] / (dv[j] + eps * (std::fabs(dv[j]) < eps ? 1.0f : 0.0f));
            if (!(std::isfinite(u) && std::isfinite(v) && u >= 0.0f && u <= 1.0f && v >= 0.0f &&
                    v <= 1.0f))
                continue;
            const float h_all = (float)r + u, w_all = (float)c + v;
            const float dh = h_all - href, dw = w_all - wref, score = dh * dh + dw * dw;
            if (score < best) {
                best = score;
                best_r = r;
                best_c = c;
                best_u = u;
                best_v = v;
                best_h = h_all;
                best_w = w_all;
            }
        }
    if (best_r < 0)
        return false;
    const F3 a = s.p(best_r, best_c), b = s.p(best_r + 1, best_c), c = s.p(best_r, best_c + 1),
             e = s.p(best_r + 1, best_c + 1);
    out.h = best_h;
    out.w = best_w;
    out.point = bilinear(a, b, c, e, best_u, best_v);
    out.normal = quad_normal(a, b, c, e, best_u, best_v);
    return true;
}

}

std::vector<float> normal_targets(const std::vector<float> &nx, const std::vector<float> &ny) {
    std::vector<float> t(nx.size() * 3);
    for (size_t i = 0; i < nx.size(); ++i) {
        const float z = std::sqrt(std::max(1.0f - nx[i] * nx[i] - ny[i] * ny[i], 1.0e-8f));
        const F3 g{nx[i], ny[i], z};
        const F3 u = g / (norm(g) + 1.0e-8f);
        t[i * 3] = u.x;
        t[i * 3 + 1] = u.y;
        t[i * 3 + 2] = u.z;
    }
    return t;
}

LossOut mesh_normal(
    const Mesh &m, const std::vector<float> &target, const std::vector<float> &mask) {
    const int D = m.D, H = m.H, W = m.W;
    const size_t n = m.points();
    LossOut o;
    o.grad.assign(m.xyz.size(), 0.0f);
    std::vector<float> lm(n);
    std::vector<F3> a(n), b(n), c(n), N(n);
    std::vector<float> r(n);
    double wsum = 0;
    for (float k : mask)
        wsum += k;
    for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const size_t i = ((size_t)d * H + h) * W + w;
                a[i] = h == 0    ? X(m, d, 1, w) - X(m, d, 0, w)
                    : h == H - 1 ? X(m, d, H - 1, w) - X(m, d, H - 2, w)
                                 : X(m, d, h + 1, w) - X(m, d, h - 1, w);
                b[i] = w == 0    ? X(m, d, h, 1) - X(m, d, h, 0)
                    : w == W - 1 ? X(m, d, h, W - 1) - X(m, d, h, W - 2)
                                 : X(m, d, h, w + 1) - X(m, d, h, w - 1);
                c[i] = cross(a[i], b[i]);
                r[i] = 1.0f / std::sqrt(dot(c[i], c[i]) + 1.0e-12f);
                N[i] = c[i] * r[i];
                const float dt = dot(N[i], ld(target.data() + i * 3));
                lm[i] = 1.0f - dt * dt;
            }
    double total = 0;
    for (size_t i = 0; i < n; ++i)
        total += (double)(lm[i] * (wsum > 0 ? mask[i] : 1.0f));
    o.value = wsum > 0 ? total / wsum : total / (double)n;
    for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const size_t i = ((size_t)d * H + h) * W + w;
                const float gl = wsum > 0 ? (float)(mask[i] / wsum) : (float)(1.0 / (double)n);
                if (gl == 0.0f)
                    continue;
                const F3 t = ld(target.data() + i * 3);
                const float dt = dot(N[i], t);
                const F3 gc = unit_backward(c[i], r[i], t * (-2.0f * dt * gl));
                const F3 ga = cross(b[i], gc), gb = cross(gc, a[i]);
                const int h1 = h == 0 ? 1
                    : h == H - 1      ? H - 1
                                      : h + 1,
                          h0 = h == 0 ? 0
                    : h == H - 1      ? H - 2
                                      : h - 1;
                const int w1 = w == 0 ? 1
                    : w == W - 1      ? W - 1
                                      : w + 1,
                          w0 = w == 0 ? 0
                    : w == W - 1      ? W - 2
                                      : w - 1;
                push(o.grad, m, d, h1, w, ga);
                push(o.grad, m, d, h0, w, ga * -1.0f);
                push(o.grad, m, d, h, w1, gb);
                push(o.grad, m, d, h, w0, gb * -1.0f);
            }
    return o;
}

LossOut mesh_step(const Mesh &m, double step) {
    const int D = m.D, H = m.H, W = m.W;
    const float t = std::max((float)step, 1.0e-12f),
                td = std::max((float)(step * std::sqrt(2.0)), 1.0e-12f);
    LossOut o;
    o.grad.assign(m.xyz.size(), 0.0f);
    const size_t cells = (size_t)D * (H - 1) * (W - 1);
    const float gk = 0.25f / (float)cells;
    double total = 0;
    for (int d = 0; d < D; ++d) {
        auto dh = [&](int h, int w) { return X(m, d, h + 1, w) - X(m, d, h, w); };
        auto dw = [&](int h, int w) { return X(m, d, h, w + 1) - X(m, d, h, w); };
        for (int h = 0; h + 1 < H; ++h)
            for (int w = 0; w + 1 < W; ++w) {
                auto [u0, v0] = unit_dir(dh(h, w));
                auto [u1, v1] = unit_dir(dh(h, w + 1));
                F3 sum = u0 + u1;
                float count = (float)v0 + (float)v1;
                if (W > 2 && w >= 1) {
                    auto [u2, v2] = unit_dir(dh(h, w - 1));
                    sum = sum + u2;
                    count = count + (float)v2;
                }
                if (!(count > 0.0f))
                    sum = {0, 0, 0};
                auto [dir_h, valid_h] = normalised(sum);
                auto [x0, y0] = unit_dir(dw(h, w));
                auto [x1, y1] = unit_dir(dw(h + 1, w));
                F3 wsum = x0 + x1;
                float wcount = (float)y0 + (float)y1;
                if (H > 2 && h >= 1) {
                    auto [x2, y2] = unit_dir(dw(h - 1, w));
                    wsum = wsum + x2;
                    wcount = wcount + (float)y2;
                }
                if (!(wcount > 0.0f))
                    wsum = {0, 0, 0};
                auto [dir_w, valid_w] = normalised(wsum);
                const F3 d1 = X(m, d, h + 1, w + 1) - X(m, d, h, w),
                         d2 = X(m, d, h + 1, w) - X(m, d, h, w + 1);
                auto [dir1, valid1] = unit_dir(d1);
                auto [dir2, valid2] = unit_dir(d2);
                const Penalty ph = step_penalty(dh(h, w), t, dir_h, valid_h),
                              pw = step_penalty(dw(h, w), t, dir_w, valid_w);
                const Penalty p1 = step_penalty(d1, td, dir1, valid1),
                              p2 = step_penalty(d2, td, dir2, valid2);
                total += (double)((ph.value + pw.value + p1.value + p2.value) * 0.25f);
                push(o.grad, m, d, h + 1, w, ph.grad * gk);
                push(o.grad, m, d, h, w, ph.grad * -gk);
                push(o.grad, m, d, h, w + 1, pw.grad * gk);
                push(o.grad, m, d, h, w, pw.grad * -gk);
                push(o.grad, m, d, h + 1, w + 1, p1.grad * gk);
                push(o.grad, m, d, h, w, p1.grad * -gk);
                push(o.grad, m, d, h + 1, w, p2.grad * gk);
                push(o.grad, m, d, h, w + 1, p2.grad * -gk);
            }
    }
    o.value = total / (double)cells;
    return o;
}

LossOut mesh_smooth(const Mesh &m, double step) {
    const int D = m.D, H = m.H, W = m.W;
    const float sq = (float)(step * step);
    LossOut o;
    o.grad.assign(m.xyz.size(), 0.0f);
    const size_t cells = (size_t)D * (H - 2) * (W - 2);
    double total = 0;
    for (int d = 0; d < D; ++d)
        for (int h = 1; h + 1 < H; ++h)
            for (int w = 1; w + 1 < W; ++w) {
                const F3 avg = (X(m, d, h - 1, w) + X(m, d, h + 1, w) + X(m, d, h, w - 1) +
                                   X(m, d, h, w + 1)) *
                    0.25f;
                const F3 diff = X(m, d, h, w) - avg;
                total += (double)(dot(diff, diff) / sq);
                const F3 g = diff * (2.0f / sq / (float)cells);
                push(o.grad, m, d, h, w, g);
                const F3 q = g * -0.25f;
                push(o.grad, m, d, h - 1, w, q);
                push(o.grad, m, d, h + 1, w, q);
                push(o.grad, m, d, h, w - 1, q);
                push(o.grad, m, d, h, w + 1, q);
            }
    o.value = total / (double)cells;
    return o;
}

LossOut mesh_bend(const Mesh &m) {
    const int D = m.D, H = m.H, W = m.W;
    LossOut o;
    o.grad.assign(m.xyz.size(), 0.0f);
    const size_t cells = (size_t)D * (H - 2) * (W - 2);
    const float gk = 0.25f / (float)cells;
    double total = 0;
    struct Triplet {
        int h0, w0, h2, w2;
    };
    for (int d = 0; d < D; ++d)
        for (int h = 1; h + 1 < H; ++h)
            for (int w = 1; w + 1 < W; ++w) {
                const Triplet ts[4] = {{h - 1, w, h + 1, w}, {h, w - 1, h, w + 1},
                    {h - 1, w - 1, h + 1, w + 1}, {h - 1, w + 1, h + 1, w - 1}};
                float pens[4];
                for (int k = 0; k < 4; ++k) {
                    const F3 mid = X(m, d, h, w), e1 = mid - X(m, d, ts[k].h0, ts[k].w0),
                             e2 = X(m, d, ts[k].h2, ts[k].w2) - mid;
                    const float r1 = 1.0f / std::sqrt(dot(e1, e1) + 1.0e-12f),
                                r2 = 1.0f / std::sqrt(dot(e2, e2) + 1.0e-12f);
                    const F3 n1 = e1 * r1, n2 = e2 * r2;
                    const float cosv = dot(n1 * -1.0f, n2), excess = std::max(cosv - -0.5f, 0.0f);
                    pens[k] = excess * excess;
                    if (excess <= 0.0f)
                        continue;
                    const float gc = 2.0f * excess * gk;
                    const F3 g1 = unit_backward(e1, r1, n2 * -gc),
                             g2 = unit_backward(e2, r2, n1 * -gc);
                    push(o.grad, m, d, h, w, g1 - g2);
                    push(o.grad, m, d, ts[k].h0, ts[k].w0, g1 * -1.0f);
                    push(o.grad, m, d, ts[k].h2, ts[k].w2, g2);
                }
                total += (double)(0.25f * (pens[0] + pens[1] + pens[2] + pens[3]));
            }
    o.value = total / (double)cells;
    return o;
}

void Station::reset(std::array<float, 3> s, int H, int W, int D) {
    seed = s;
    h_frac.assign((size_t)D, (H - 1) / 2.0);
    w_frac.assign((size_t)D, (W - 1) / 2.0);
    initialized.assign((size_t)D, 0);
    has_ray.assign((size_t)D, 0);
    ray.assign((size_t)D, {0, 0, 0});
}

std::array<float, 3> grid_centre(const Mesh &m) {
    const double hm = (m.H - 1) / 2.0, wm = (m.W - 1) / 2.0;
    const int h0 = (int)hm, w0 = (int)wm, h1 = std::min(h0 + 1, m.H - 1),
              w1 = std::min(w0 + 1, m.W - 1);
    const double fh = hm - h0, fw = wm - w0;
    const float k00 = (float)((1 - fh) * (1 - fw)), k10 = (float)(fh * (1 - fw)),
                k01 = (float)((1 - fh) * fw), k11 = (float)(fh * fw);
    const F3 c = X(m, 0, h0, w0) * k00 + X(m, 0, h1, w0) * k10 + X(m, 0, h0, w1) * k01 +
        X(m, 0, h1, w1) * k11;
    return {c.x, c.y, c.z};
}

void station_loss(
    const Mesh &m, Station &s, double mesh_step, int anchor_d, LossOut &normal, LossOut &tangent) {
    const int D = m.D, H = m.H, W = m.W;
    const F3 seed{s.seed[0], s.seed[1], s.seed[2]};
    const float scale = (float)std::max(1.0e-6, mesh_step);
    normal = LossOut{0.0, std::vector<float>(m.xyz.size(), 0.0f)};
    tangent = LossOut{0.0, std::vector<float>(m.xyz.size(), 0.0f)};
    struct Hit {
        int d;
        Anchor a;
    };
    std::vector<Hit> hits;
    for (int d = 0; d < D; ++d) {
        const Surf surf{m, d};
        Anchor a{};
        bool found = false;
        if (s.initialized[(size_t)d]) {
            F3 ray{s.ray[(size_t)d][0], s.ray[(size_t)d][1], s.ray[(size_t)d][2]};
            bool have = s.has_ray[(size_t)d];
            if (!have) {
                have = ray_at(surf, seed, s.h_frac[(size_t)d], s.w_frac[(size_t)d], ray);
                s.has_ray[(size_t)d] = have;
                s.ray[(size_t)d] = {ray.x, ray.y, ray.z};
            }
            if (have)
                found = local_update(surf, seed, s.h_frac[(size_t)d], s.w_frac[(size_t)d], ray, a);
            if (!found) {
                F3 r2;
                found = brute_force(surf, seed, s.h_frac[(size_t)d], s.w_frac[(size_t)d], a, r2);
                if (found) {
                    s.ray[(size_t)d] = {r2.x, r2.y, r2.z};
                    s.has_ray[(size_t)d] = 1;
                }
            }
        } else {
            F3 r2;
            found = brute_force(surf, seed, s.h_frac[(size_t)d], s.w_frac[(size_t)d], a, r2);
            if (found) {
                s.ray[(size_t)d] = {r2.x, r2.y, r2.z};
                s.has_ray[(size_t)d] = 1;
                s.initialized[(size_t)d] = 1;
            }
        }
        if (!found)
            continue;
        s.h_frac[(size_t)d] = a.h;
        s.w_frac[(size_t)d] = a.w;
        hits.push_back({d, a});
    }
    const size_t n = m.points();
    for (const Hit &hit : hits) {
        if (hit.d != anchor_d)
            continue;
        const F3 na = hit.a.normal, diff = hit.a.point - seed;
        const float offset = diff.x * na.x + diff.y * na.y + diff.z * na.z;
        double total = 0;
        for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    const F3 p = X(m, d, h, w), target = p - na * offset;
                    const F3 e = p - target;
                    const float dist = e.x * na.x + e.y * na.y + e.z * na.z, r = dist / scale;
                    const float hh = (float)h - (float)hit.a.h, ww = (float)w - (float)hit.a.w;
                    const float idx = std::sqrt(hh * hh + ww * ww), z = idx / 20.0f;
                    const float weight = 0.1f + 0.9f * std::exp(-0.5f * (z * z));
                    total += (double)(huber(r) * weight);
                    push(normal.grad, m, d, h, w, na * (huber_grad(r) * weight / scale / (float)n));
                }
        normal.value = total / (double)n;
        break;
    }
    if (hits.empty())
        return;
    const double hm = (H - 1) / 2.0, wm = (W - 1) / 2.0;
    double sum = 0;
    const float k = 1.0f / (float)hits.size();
    for (const Hit &hit : hits) {
        const int d = hit.d;
        const float dh = (float)(hit.a.h - hm), dw = (float)(hit.a.w - wm);
        double part = 0;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const F3 th = h + 1 < H ? X(m, d, h + 1, w) - X(m, d, h, w)
                                        : X(m, d, H - 1, w) - X(m, d, H - 2, w);
                const F3 tw = w + 1 < W ? X(m, d, h, w + 1) - X(m, d, h, w)
                                        : X(m, d, h, W - 1) - X(m, d, h, W - 2);
                const F3 p = X(m, d, h, w), target = p + th * dh + tw * dw, e = p - target;
                const float dist = norm(e), r = dist / scale;
                part += (double)huber(r);
                if (dist > 0.0f)
                    push(tangent.grad, m, d, h, w,
                        e * (k * huber_grad(r) / scale / dist / (float)(H * W)));
            }
        sum += part / (double)(H * W);
    }
    tangent.value = sum / (double)hits.size();
}

namespace {

double eff_weight(const Json &config, const Json &stage, const std::string &term) {
    const Json &base = config["base"];
    const double b = base.has(term) && base[term].kind == Json::Number ? base[term].num() : 0.0;
    double v = b;
    if (stage.has("default_mul") && stage["default_mul"].kind == Json::Number)
        v = b * stage["default_mul"].num();
    if (stage.has("w_fac")) {
        const Json &f = stage["w_fac"];
        if (f.kind == Json::Number)
            v = b * f.num();
        else if (f.kind == Json::Object && f.has(term) && f[term].kind == Json::Number)
            v = b * f[term].num();
    }
    return v;
}

Grid to_grid(const std::vector<float> &flat, int D, int H, int W) {
    Grid g{H, W, 3 * D, std::vector<float>((size_t)3 * D * H * W)};
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    g.at(h, w)[d * 3 + c] = flat[(((size_t)c * D + d) * H + h) * W + w];
    return g;
}

}

MeshFit::MeshFit(const std::string &manifest, const Json &config_, const std::vector<float> &flat,
    int D_, int H_, int W_, double mesh_step_, std::vector<int> windings)
    : volume(Volume::open(manifest, {"grad_mag", "nx", "ny"})), config(config_), D(D_), H(H_),
      W(W_), mesh_step(mesh_step_), depth_windings(std::move(windings)) {
    levels = build_pyramid(to_grid(flat, D, H, W), 5);
}

Mesh MeshFit::current() const {
    const Grid g = integrate_pyramid(levels);
    Mesh m{D, H, W, std::vector<float>((size_t)D * H * W * 3)};
    for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < 3; ++c)
                    m.at(d, h, w)[c] = g.at(h, w)[d * 3 + c];
    return m;
}

std::vector<float> MeshFit::flat() const {
    const Mesh m = current();
    std::vector<float> out((size_t)3 * D * H * W);
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    out[(((size_t)c * D + d) * H + h) * W + w] = m.at(d, h, w)[c];
    return out;
}

double MeshFit::evaluate(const MeshWeights &w, std::vector<float> &grad) {
    const Mesh m = current();
    grad.assign(m.xyz.size(), 0.0f);
    double total = 0;
    auto add = [&](double weight, const LossOut &o) {
        if (weight == 0.0)
            return;
        total += weight * o.value;
        for (size_t i = 0; i < grad.size(); ++i)
            grad[i] += (float)weight * o.grad[i];
    };
    add(w.step, la::mesh_step(m, mesh_step));
    add(w.smooth, mesh_smooth(m, mesh_step));
    if (w.normal != 0.0) {
        const size_t n = m.points();
        std::vector<float> values[3];
        const char *names[3] = {"grad_mag", "nx", "ny"};
        for (int k = 0; k < 3; ++k) {
            Channel &ch = volume.channel(names[k]);
            volume.prefetch(ch, m.xyz.data(), n);
            std::vector<uint8_t> u8(n);
            sample_u8(volume, ch, m.xyz.data(), n, u8.data());
            values[k].resize(n);
            for (size_t i = 0; i < n; ++i)
                values[k][i] = decode(volume, names[k], (float)u8[i]);
        }
        std::vector<float> mask(n);
        for (size_t i = 0; i < n; ++i)
            mask[i] = values[0][i] > 0.0f ? 1.0f : 0.0f;
        add(w.normal, mesh_normal(m, normal_targets(values[1], values[2]), mask));
    }
    if (w.station_n != 0.0 || w.station_t != 0.0) {
        int anchor = std::max(0, (D - 1) / 2);
        if ((int)depth_windings.size() == D)
            for (int d = 0; d < D; ++d)
                if (depth_windings[(size_t)d] == 0) {
                    anchor = d;
                    break;
                }
        LossOut sn, st;
        station_loss(m, station, mesh_step, anchor, sn, st);
        add(w.station_n, sn);
        add(w.station_t, st);
    }
    add(w.bend, mesh_bend(m));
    return total;
}

void MeshFit::run() {
    seed = grid_centre(current());
    static const char *ported[] = {"normal", "step", "smooth", "station_n", "station_t", "bend"};
    for (const Json &stage : config["stages"].items) {
        const int steps = stage.has("steps") && stage["steps"].kind == Json::Number
            ? (int)stage["steps"].num()
            : 0;
        if (steps <= 0)
            continue;
        bool mesh_only = stage.has("params") && stage["params"].size() == 1 &&
            stage["params"][0].str() == "mesh_ms";
        if (!mesh_only)
            throw std::runtime_error(
                "mesh stage " + stage["name"].str() + " optimises more than mesh_ms");
        for (const auto &[term, value] : config["base"].fields) {
            (void)value;
            if (std::find(std::begin(ported), std::end(ported), term) == std::end(ported) &&
                eff_weight(config, stage, term) != 0.0)
                throw std::runtime_error("mesh term " + term + " is not ported");
        }
        const MeshWeights w{eff_weight(config, stage, "normal"), eff_weight(config, stage, "step"),
            eff_weight(config, stage, "smooth"), eff_weight(config, stage, "station_n"),
            eff_weight(config, stage, "station_t"), eff_weight(config, stage, "bend")};
        if (w.station_n > 0.0 || w.station_t > 0.0)
            station.reset(seed, H, W, D);
        const int frozen = stage.has("min_scaledown") && stage["min_scaledown"].kind == Json::Number
            ? (int)stage["min_scaledown"].num()
            : 0;
        std::vector<double> lrs(levels.size(), 1.0e-3);
        for (size_t i = 0; i < levels.size(); ++i) {
            if (stage.has("lr")) {
                const Json &lr = stage["lr"];
                if (lr.kind == Json::Number)
                    lrs[i] = lr.num();
                else if (lr.kind == Json::Array) {
                    const long n = (long)lr.items.size(), idx = -1 - (long)i;
                    lrs[i] = n == 0
                        ? 0.0
                        : (-n <= idx ? lr.items[(size_t)(n + idx)].num() : lr.items[0].num());
                }
            }
            if ((int)i < frozen)
                lrs[i] = 0.0;
        }
        const Json &args =
            stage.has("args") && stage["args"].kind == Json::Object ? stage["args"] : Json();
        const int warmup =
            args.has("lr_warmup_steps") && args["lr_warmup_steps"].kind == Json::Number
            ? (int)args["lr_warmup_steps"].num()
            : 0;
        Adam adam;
        std::vector<float> grad;
        MeshStageReport report{stage.has("name") ? stage["name"].str() : "stage", steps};
        report.first_loss = report.last_loss = evaluate(w, grad);
        for (int step = 0; step < steps; ++step) {
            report.last_loss = evaluate(w, grad);
            Grid g{H, W, 3 * D, std::vector<float>((size_t)3 * D * H * W)};
            for (int d = 0; d < D; ++d)
                for (int h = 0; h < H; ++h)
                    for (int x = 0; x < W; ++x)
                        for (int c = 0; c < 3; ++c)
                            g.at(h, x)[d * 3 + c] = grad[(((size_t)d * H + h) * W + x) * 3 + c];
            std::vector<double> lr = lrs;
            if (warmup > 0)
                for (double &v : lr)
                    v *= std::min(1.0, std::max(0.0, (double)(step + 1) / (double)warmup));
            adam.update(levels, integrate_pyramid_backward(levels, g), lr);
        }
        if (log)
            std::fprintf(log, "%s: %d steps loss %.6g -> %.6g\n", report.name.c_str(), steps,
                report.first_loss, report.last_loss);
        stages.push_back(report);
    }
}

void write_tiff_f32(const std::string &path, const float *data, int H, int W) {
    const size_t row = (size_t)W * 4, total = row * (size_t)H;
    const int rows =
        (int)std::max<size_t>(1, std::min<size_t>((size_t)H, 262144 / std::max<size_t>(row, 1)));
    const int strips = (H + rows - 1) / rows;
    std::string out;
    auto u16 = [&](uint32_t v) {
        out.push_back((char)(v & 0xFF)), out.push_back((char)((v >> 8) & 0xFF));
    };
    auto u32 = [&](uint32_t v) { u16(v & 0xFFFF), u16(v >> 16); };
    out += "II";
    u16(42);
    const uint32_t ifd = (uint32_t)(8 + total);
    u32(ifd);
    out.append((const char *)data, total);
    const uint32_t arrays = ifd + 2 + 11 * 12 + 4;
    auto entry = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
        u16(tag), u16(type), u32(count), u32(value);
    };
    u16(11);
    entry(256, 3, 1, (uint32_t)W);
    entry(257, 3, 1, (uint32_t)H);
    entry(258, 3, 1, 32);
    entry(259, 3, 1, 1);
    entry(262, 3, 1, 1);
    entry(273, 4, (uint32_t)strips, strips == 1 ? 8u : arrays + 4u * (uint32_t)strips);
    entry(277, 3, 1, 1);
    entry(278, 3, 1, (uint32_t)rows);
    entry(279, 4, (uint32_t)strips, strips == 1 ? (uint32_t)total : arrays);
    entry(284, 3, 1, 1);
    entry(339, 3, 1, 3);
    u32(0);
    if (strips > 1) {
        for (int k = 0; k < strips; ++k)
            u32((uint32_t)(row * (size_t)std::min(rows, H - k * rows)));
        for (int k = 0; k < strips; ++k)
            u32((uint32_t)(8 + row * (size_t)rows * (size_t)k));
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f.write(out.data(), (std::streamsize)out.size());
}

void write_tifxyz(const std::string &dir, const Mesh &m, int d, double winding, double mesh_step,
    const std::string &meta_extra) {
    ::mkdir(dir.c_str(), 0755);
    const int H = m.H, W = m.W;
    std::vector<float> x((size_t)H * W), y(x.size()), z(x.size()), dl(x.size());
    std::vector<uint8_t> valid(x.size());
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    bool any = false;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const size_t i = (size_t)h * W + w;
            const float *p = m.at(d, h, w);
            x[i] = p[0];
            y[i] = p[1];
            z[i] = p[2];
            valid[i] = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
                !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f);
            dl[i] = valid[i] ? (float)winding : -1.0f;
            if (valid[i]) {
                any = true;
                for (int a = 0; a < 3; ++a) {
                    lo[a] = std::min(lo[a], (double)p[a]);
                    hi[a] = std::max(hi[a], (double)p[a]);
                }
            }
        }
    long quads = 0;
    for (int h = 0; h + 1 < H; ++h)
        for (int w = 0; w + 1 < W; ++w)
            quads += valid[(size_t)h * W + w] && valid[(size_t)h * W + w + 1] &&
                valid[(size_t)(h + 1) * W + w] && valid[(size_t)(h + 1) * W + w + 1];
    if (!any)
        for (int a = 0; a < 3; ++a)
            lo[a] = hi[a] = -1.0;
    std::mt19937_64 rng(std::random_device{}());
    char uuid[37];
    const uint64_t r0 = rng(), r1 = rng();
    std::snprintf(uuid, sizeof uuid, "%08x-%04x-4%03x-%04x-%012llx", (unsigned)(r0 >> 32),
        (unsigned)((r0 >> 16) & 0xFFFF), (unsigned)(r0 & 0xFFF),
        (unsigned)(0x8000 | ((r1 >> 48) & 0x3FFF)), (unsigned long long)(r1 & 0xFFFFFFFFFFFFULL));
    const std::string name = dir.substr(dir.find_last_of('/') + 1);
    const double scale = 1.0 / mesh_step;
    char head[1024];
    std::snprintf(head, sizeof head,
        "{\n  \"uuid\": \"%s\",\n  \"name\": \"%s\",\n  \"type\": \"seg\",\n  \"format\": "
        "\"tifxyz\",\n  "
        "\"scale\": [%.17g, %.17g],\n"
        "  \"bbox\": [[%.17g, %.17g, %.17g], [%.17g, %.17g, %.17g]],\n  \"area_vx2\": %.17g",
        uuid, name.c_str(), scale, scale, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
        (double)quads * mesh_step * mesh_step);
    std::ofstream meta(dir + "/meta.json", std::ios::trunc);
    meta << head << meta_extra << "\n}\n";
    write_tiff_f32(dir + "/x.tif", x.data(), H, W);
    write_tiff_f32(dir + "/y.tif", y.data(), H, W);
    write_tiff_f32(dir + "/z.tif", z.data(), H, W);
    write_tiff_f32(dir + "/d.tif", dl.data(), H, W);
}

}
