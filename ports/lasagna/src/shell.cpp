#include "shell.h"

#include <algorithm>
#include <cmath>

namespace la {

namespace {

struct V3 {
    float x, y, z;
};

V3 load(const float *p) { return {p[0], p[1], p[2]}; }
V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 scale(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float norm(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

void push(std::vector<float> &g, const Shell &s, int h, int w, V3 d) {
    float *p = g.data() + ((size_t)h * s.W + (size_t)((w % s.W + s.W) % s.W)) * 3;
    p[0] += d.x;
    p[1] += d.y;
    p[2] += d.z;
}

std::pair<float, float> spacing_scales(const Shell &s) {
    float h_scale = 1.0f;
    if (s.H > 1) {
        double sum = 0;
        for (int h = 0; h + 1 < s.H; ++h)
            for (int w = 0; w < s.W; ++w)
                sum += norm(sub(load(s.at(h + 1, w)), load(s.at(h, w))));
        h_scale = std::max(1.0f, (float)(sum / ((double)(s.H - 1) * s.W)));
    }
    double sum = 0;
    for (int h = 0; h < s.H; ++h)
        for (int w = 0; w < s.W; ++w)
            sum += norm(sub(load(s.at(h, w + 1)), load(s.at(h, w))));
    const float w_scale = std::max(1.0f, (float)(sum / ((double)s.H * s.W)));
    return {h_scale, w_scale};
}

struct Laplace {
    int h, w;
    V3 residual;
};

void laplace_backward(
    std::vector<float> &g, const Shell &s, int h, int w, int dh, int dw, float k) {
    const V3 r = sub(
        load(s.at(h, w)), scale(add(load(s.at(h - dh, w - dw)), load(s.at(h + dh, w + dw))), 0.5f));
    const V3 d = scale(r, k);
    push(g, s, h, w, d);
    push(g, s, h - dh, w - dw, scale(d, -0.5f));
    push(g, s, h + dh, w + dw, scale(d, -0.5f));
}

float laplace_lm(const Shell &s, int h, int w, int dh, int dw, float sc) {
    const V3 r = sub(
        load(s.at(h, w)), scale(add(load(s.at(h - dh, w - dw)), load(s.at(h + dh, w + dw))), 0.5f));
    const float t = norm(r) / sc;
    return t * t;
}

}

LossOut cyl_smooth(const Shell &s) {
    LossOut o;
    o.grad.assign(s.xyz.size(), 0.0f);
    const auto [h_scale, w_scale] = spacing_scales(s);
    const size_t n = (s.H > 2 ? (size_t)(s.H - 2) * s.W : 0) + (size_t)s.H * s.W;
    const float k_h = 2.0f / (h_scale * h_scale) / (float)n,
                k_w = 2.0f / (w_scale * w_scale) / (float)n;
    double sum = 0;
    if (s.H > 2)
        for (int h = 1; h + 1 < s.H; ++h)
            for (int w = 0; w < s.W; ++w) {
                sum += laplace_lm(s, h, w, 1, 0, h_scale);
                laplace_backward(o.grad, s, h, w, 1, 0, k_h);
            }
    for (int h = 0; h < s.H; ++h)
        for (int w = 0; w < s.W; ++w) {
            sum += laplace_lm(s, h, w, 0, 1, w_scale);
            laplace_backward(o.grad, s, h, w, 0, 1, k_w);
        }
    o.value = sum / (double)n;
    return o;
}

LossOut cyl_z_smooth(const Shell &s) {
    LossOut o;
    o.grad.assign(s.xyz.size(), 0.0f);
    if (s.H <= 2)
        return o;
    const float h_scale = spacing_scales(s).first;
    const size_t n = (size_t)(s.H - 2) * s.W;
    const float k = 2.0f / (h_scale * h_scale) / (float)n;
    double sum = 0;
    for (int h = 1; h + 1 < s.H; ++h)
        for (int w = 0; w < s.W; ++w) {
            sum += laplace_lm(s, h, w, 1, 0, h_scale);
            laplace_backward(o.grad, s, h, w, 1, 0, k);
        }
    o.value = sum / (double)n;
    return o;
}

LossOut cyl_z_center(const Shell &s, double target, double height_step) {
    LossOut o;
    o.grad.assign(s.xyz.size(), 0.0f);
    const float sc = (float)std::max(1.0, height_step);
    double zsum = 0;
    for (int i = 0; i < s.H * s.W; ++i)
        zsum += s.xyz[(size_t)i * 3 + 2];
    const float z_error = (float)(zsum / (double)(s.H * s.W)) - (float)target;
    if (s.H <= 1) {
        const float t = z_error / sc;
        o.value = (double)t * t;
        const float d = 2.0f * t / sc / (float)(s.H * s.W);
        for (int i = 0; i < s.H * s.W; ++i)
            o.grad[(size_t)i * 3 + 2] = d;
        return o;
    }
    std::vector<V3> dir((size_t)s.H * s.W);
    double zdir = 0;
    for (int h = 0; h < s.H; ++h)
        for (int w = 0; w < s.W; ++w) {
            V3 v;
            if (s.H == 2)
                v = sub(load(s.at(1, w)), load(s.at(0, w)));
            else if (h == 0)
                v = sub(load(s.at(1, w)), load(s.at(0, w)));
            else if (h == s.H - 1)
                v = sub(load(s.at(h, w)), load(s.at(h - 1, w)));
            else
                v = scale(sub(load(s.at(h + 1, w)), load(s.at(h - 1, w))), 0.5f);
            dir[(size_t)h * s.W + w] = v;
            zdir += v.z;
        }
    const float orient = zdir / (double)(s.H * s.W) < 0.0 ? -1.0f : 1.0f;
    double sum = 0;
    const size_t n = (size_t)s.H * s.W;
    for (size_t i = 0; i < n; ++i) {
        const V3 v = dir[i];
        const float len = std::max(norm(v), 1.0e-8f);
        const V3 u = scale(scale(v, 1.0f / len), orient);
        const V3 p = load(s.xyz.data() + i * 3);
        const V3 r = sub(p, sub(p, scale(u, z_error)));
        const float t = norm(r) / sc;
        sum += (double)t * t;
        const V3 d = scale(r, 2.0f / (sc * sc) / (float)n);
        o.grad[i * 3] = d.x;
        o.grad[i * 3 + 1] = d.y;
        o.grad[i * 3 + 2] = d.z;
    }
    o.value = sum / (double)n;
    return o;
}

LossOut cyl_step(
    const Shell &s, const std::vector<uint8_t> &valid, double width_step, double height_step) {
    LossOut o;
    o.grad.assign(s.xyz.size(), 0.0f);
    const int H = s.H, W = s.W;
    double wsum = 0, hsum = 0;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            wsum += norm(sub(load(s.at(h, w + 1)), load(s.at(h, w))));
    for (int h = 0; h + 1 < H; ++h)
        for (int w = 0; w < W; ++w)
            hsum += norm(sub(load(s.at(h + 1, w)), load(s.at(h, w))));
    float wt, ht;
    if (width_step > 0)
        wt = std::max((float)width_step, 1.0e-6f);
    else
        wt = std::max((float)((wsum + hsum) / ((double)H * W + (double)(H - 1) * W)), 1.0e-6f);
    if (height_step > 0)
        ht = std::max((float)height_step, 1.0e-6f);
    else if (H > 1)
        ht = std::max((float)(hsum / ((double)(H - 1) * W)), 1.0e-6f);
    else
        ht = wt;
    const float dt = std::max(std::sqrt(ht * ht + wt * wt), 1.0e-6f);
    const bool masked = !valid.empty();
    auto ok = [&](int h, int w) {
        return !masked || valid[(size_t)h * W + (size_t)((w % W + W) % W)] != 0;
    };
    const auto [h_scale, w_scale] = spacing_scales(s);
    std::vector<float> smooth((size_t)H * W, 0.0f);
    if (masked)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                float v = laplace_lm(s, h, w, 0, 1, w_scale);
                if (H > 2 && h > 0 && h + 1 < H)
                    v = v + laplace_lm(s, h, w, 1, 0, h_scale);
                smooth[(size_t)h * W + w] = v;
            }
    const size_t n = (size_t)H * W + (H > 1 ? 3 * (size_t)(H - 1) * W : 0);
    const float inv_n = 1.0f / (float)n;
    double sum = 0;
    auto smooth_backward = [&](int h, int w, float k) {
        laplace_backward(o.grad, s, h, ((w % W) + W) % W, 0, 1, 2.0f * k / (w_scale * w_scale));
        if (H > 2 && h > 0 && h + 1 < H)
            laplace_backward(o.grad, s, h, ((w % W) + W) % W, 1, 0, 2.0f * k / (h_scale * h_scale));
    };
    auto edge = [&](int h0, int w0, int h1, int w1, float target) {
        const V3 d = sub(load(s.at(h1, w1)), load(s.at(h0, w0)));
        const float len = norm(d);
        if (ok(h0, w0) && ok(h1, w1)) {
            const float t = (len - target) / target;
            sum += (double)t * t;
            const float k = 2.0f * t / target * inv_n;
            const V3 u = len > 0.0f ? scale(d, 1.0f / len) : V3{0, 0, 0};
            push(o.grad, s, h1, w1, scale(u, k));
            push(o.grad, s, h0, w0, scale(u, -k));
        } else {
            sum += (double)smooth[(size_t)h0 * W + (size_t)((w0 % W + W) % W)] +
                (double)smooth[(size_t)h1 * W + (size_t)((w1 % W + W) % W)];
            smooth_backward(h0, w0, inv_n);
            smooth_backward(h1, w1, inv_n);
        }
    };
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            edge(h, w, h, w + 1, wt);
    if (H > 1) {
        for (int h = 0; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                edge(h, w, h + 1, w, ht);
        for (int h = 0; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                edge(h, w, h + 1, w + 1, dt);
        for (int h = 0; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                edge(h, w + 1, h + 1, w, dt);
    }
    o.value = sum / (double)n;
    return o;
}

LossOut cyl_bend(const Shell &s) {
    LossOut o;
    o.grad.assign(s.xyz.size(), 0.0f);
    const float threshold = (float)std::cos(M_PI - M_PI / 3.0);
    const int H = s.H, W = s.W;
    size_t n = 0;
    if (H > 2)
        n += (size_t)(H - 2) * W;
    if (W > 2)
        n += (size_t)H * W;
    if (H > 2 && W > 2)
        n += 2 * (size_t)(H - 2) * W;
    if (!n)
        return o;
    double sum = 0;
    auto bend = [&](int ha, int wa, int hb, int wb, int hc, int wc) {
        const V3 e1 = sub(load(s.at(hb, wb)), load(s.at(ha, wa))),
                 e2 = sub(load(s.at(hc, wc)), load(s.at(hb, wb)));
        const float l1 = std::max(norm(e1), 1.0e-12f), l2 = std::max(norm(e2), 1.0e-12f);
        const V3 n1 = scale(e1, 1.0f / l1), n2 = scale(e2, 1.0f / l2);
        const float c = -dot(n1, n2);
        const float excess = std::max(c - threshold, 0.0f);
        sum += (double)excess * excess;
        if (excess <= 0.0f)
            return;
        const float k = -2.0f * excess / (float)n;
        const V3 g1 = scale(sub(n2, scale(n1, dot(n1, n2))), k / l1);
        const V3 g2 = scale(sub(n1, scale(n2, dot(n1, n2))), k / l2);
        push(o.grad, s, hb, wb, g1);
        push(o.grad, s, ha, wa, scale(g1, -1.0f));
        push(o.grad, s, hc, wc, g2);
        push(o.grad, s, hb, wb, scale(g2, -1.0f));
    };
    if (H > 2)
        for (int h = 1; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                bend(h - 1, w, h, w, h + 1, w);
    if (W > 2)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                bend(h, w - 1, h, w, h, w + 1);
    if (H > 2 && W > 2) {
        for (int h = 1; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                bend(h - 1, w - 1, h, w, h + 1, w + 1);
        for (int h = 1; h + 1 < H; ++h)
            for (int w = 0; w < W; ++w)
                bend(h - 1, w + 1, h, w, h + 1, w - 1);
    }
    o.value = sum / (double)n;
    return o;
}

Grid supersample(const Grid &f, int factor) {
    if (factor <= 1)
        return f;
    const int th = std::max(2, (f.H - 1) * factor + 1), tw = std::max(3, f.W * factor);
    Grid rows{th, f.W, f.C, std::vector<float>((size_t)th * f.W * f.C)};
    const float hs = th > 1 ? (float)(f.H - 1) / (float)(th - 1) : 0.0f;
    for (int y = 0; y < th; ++y) {
        const float src = hs * (float)y;
        const int i0 = std::min((int)src, f.H - 1), i1 = std::min(i0 + 1, f.H - 1);
        const float l1 = src - (float)i0, l0 = 1.0f - l1;
        for (int w = 0; w < f.W; ++w)
            for (int c = 0; c < f.C; ++c)
                rows.at(y, w)[c] = l0 * f.at(i0, w)[c] + l1 * f.at(i1, w)[c];
    }
    if (tw == f.W)
        return rows;
    Grid out{th, tw, f.C, std::vector<float>((size_t)th * tw * f.C)};
    const float step = (float)f.W / (float)tw;
    for (int x = 0; x < tw; ++x) {
        const float phase = (float)x * step;
        const int i0 = (int)std::floor(phase);
        const float frac = phase - (float)i0;
        const int a = ((i0 % f.W) + f.W) % f.W, b = (a + 1) % f.W;
        for (int y = 0; y < th; ++y)
            for (int c = 0; c < f.C; ++c) {
                const float p0 = rows.at(y, a)[c], p1 = rows.at(y, b)[c];
                out.at(y, x)[c] = p0 + frac * (p1 - p0);
            }
    }
    return out;
}

Grid supersample_backward(const Grid &g, int H, int W, int factor) {
    Grid rows{g.H, W, g.C, std::vector<float>((size_t)g.H * W * g.C, 0.0f)};
    if (g.W == W)
        rows = g;
    else {
        const float step = (float)W / (float)g.W;
        for (int x = 0; x < g.W; ++x) {
            const float phase = (float)x * step;
            const int i0 = (int)std::floor(phase);
            const float frac = phase - (float)i0;
            const int a = ((i0 % W) + W) % W, b = (a + 1) % W;
            for (int y = 0; y < g.H; ++y)
                for (int c = 0; c < g.C; ++c) {
                    const float v = g.at(y, x)[c];
                    rows.at(y, a)[c] += v - frac * v;
                    rows.at(y, b)[c] += frac * v;
                }
        }
    }
    Grid out{H, W, g.C, std::vector<float>((size_t)H * W * g.C, 0.0f)};
    if (factor <= 1)
        return rows;
    const int th = g.H;
    const float hs = th > 1 ? (float)(H - 1) / (float)(th - 1) : 0.0f;
    for (int y = 0; y < th; ++y) {
        const float src = hs * (float)y;
        const int i0 = std::min((int)src, H - 1), i1 = std::min(i0 + 1, H - 1);
        const float l1 = src - (float)i0, l0 = 1.0f - l1;
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < g.C; ++c) {
                out.at(i0, w)[c] += l0 * rows.at(y, w)[c];
                out.at(i1, w)[c] += l1 * rows.at(y, w)[c];
            }
    }
    return out;
}

Grid interp_width_at_offsets(const Grid &f, const Grid &offsets, float scale) {
    Grid out{f.H, f.W, f.C, std::vector<float>(f.v.size())};
    for (int h = 0; h < f.H; ++h)
        for (int w = 0; w < f.W; ++w) {
            const float phase = (float)w + offsets.at(h, w)[0] * scale;
            const float i0f = std::floor(phase);
            const float frac = phase - i0f;
            const int a = (((int)i0f % f.W) + f.W) % f.W, b = (a + 1) % f.W;
            for (int c = 0; c < f.C; ++c) {
                const float p0 = f.at(h, a)[c], p1 = f.at(h, b)[c];
                out.at(h, w)[c] = p0 + frac * (p1 - p0);
            }
        }
    return out;
}

Shell shell_geometry(
    const Grid &base, const Grid &offsets, const Grid &delta, bool has_base, int factor) {
    const Grid base_s = supersample(base, factor), offsets_s = supersample(offsets, factor),
               delta_s = supersample(delta, factor);
    const Grid conn = has_base ? interp_width_at_offsets(base_s, offsets_s, (float)factor) : base_s;
    Shell s;
    s.H = delta_s.H;
    s.W = delta_s.W;
    s.xyz.resize(delta_s.v.size());
    for (size_t i = 0; i < s.xyz.size(); ++i)
        s.xyz[i] = conn.v[i] + delta_s.v[i];
    return s;
}

std::vector<float> unit_normals(const Shell &s) {
    std::vector<float> n(s.xyz.size());
    for (int h = 0; h < s.H; ++h)
        for (int w = 0; w < s.W; ++w) {
            V3 dh;
            if (s.H == 1)
                dh = {0, 0, 0};
            else if (h == 0)
                dh = sub(load(s.at(1, w)), load(s.at(0, w)));
            else if (h == s.H - 1)
                dh = sub(load(s.at(h, w)), load(s.at(h - 1, w)));
            else
                dh = sub(load(s.at(h + 1, w)), load(s.at(h - 1, w)));
            const V3 dw = sub(load(s.at(h, w + 1)), load(s.at(h, w - 1)));
            const V3 c = {
                dh.y * dw.z - dh.z * dw.y, dh.z * dw.x - dh.x * dw.z, dh.x * dw.y - dh.y * dw.x};
            const float len = std::max(norm(c), 1.0e-8f);
            float *p = n.data() + ((size_t)h * s.W + w) * 3;
            p[0] = c.x / len;
            p[1] = c.y / len;
            p[2] = c.z / len;
        }
    return n;
}

LossOut cyl_step_push(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<uint8_t> &valid, double width_step, double mesh_step, int factor) {
    LossOut o;
    o.grad.assign(delta.v.size(), 0.0f);
    if (width_step <= 0)
        return o;
    const Shell s = shell_geometry(base, offsets, delta, has_base, factor);
    double esum = 0;
    size_t ecount = 0;
    for (int h = 0; h < s.H; ++h)
        for (int w = 0; w < s.W; ++w)
            if (valid[(size_t)h * s.W + w] && valid[(size_t)h * s.W + (size_t)((w + 1) % s.W)]) {
                esum += norm(sub(load(s.at(h, w + 1)), load(s.at(h, w))));
                ++ecount;
            }
    if (!ecount)
        return o;
    const float observed = (float)(esum / (double)ecount) * (float)factor;
    const float push = (float)width_step - observed;
    if (!std::isfinite(push) || std::fabs(push) <= 1.0e-6f)
        return o;
    const std::vector<float> normals = unit_normals(s);
    const float sc = (float)std::max(1.0, mesh_step);
    double wsum = 0, lsum = 0;
    for (int i = 0; i < s.H * s.W; ++i)
        wsum += valid[(size_t)i] ? 1.0 : 0.0;
    std::vector<float> gs(s.xyz.size(), 0.0f);
    const float inv = 1.0f / (float)std::max(wsum, 1.0);
    for (int i = 0; i < s.H * s.W; ++i) {
        const V3 p = load(s.xyz.data() + (size_t)i * 3);
        const V3 nn = scale(load(normals.data() + (size_t)i * 3), -1.0f);
        const V3 proxy = add(p, scale(nn, push));
        const V3 d = sub(p, proxy);
        const float len = norm(d), r = len / sc;
        const float lm = r <= 1.0f ? 0.5f * r * r : r - 0.5f;
        if (!valid[(size_t)i])
            continue;
        lsum += lm;
        const float dr = r <= 1.0f ? r : 1.0f;
        const float k = len > 0.0f ? dr / (len * sc) * inv : 0.0f;
        gs[(size_t)i * 3] = d.x * k;
        gs[(size_t)i * 3 + 1] = d.y * k;
        gs[(size_t)i * 3 + 2] = d.z * k;
    }
    o.value = wsum > 0 ? lsum / std::max(wsum, 1.0) : INFINITY;
    o.grad = supersample_backward(Grid{s.H, s.W, 3, gs}, delta.H, delta.W, factor).v;
    return o;
}

static V3 unit(V3 v, float eps) { return scale(v, 1.0f / std::max(norm(v), eps)); }

void shell_normal_targets(const Shell &xyz, const std::vector<float> &nx,
    const std::vector<float> &ny, const std::vector<uint8_t> &valid, std::vector<float> &target,
    std::vector<float> &mask) {
    const int H = xyz.H, W = xyz.W;
    const size_t n = (size_t)H * W;
    const std::vector<float> mesh = unit_normals(xyz);
    std::vector<V3> t(n);
    for (size_t i = 0; i < n; ++i) {
        const float z = std::sqrt(std::max(1.0f - nx[i] * nx[i] - ny[i] * ny[i], 1.0e-8f));
        V3 g = {nx[i], ny[i], z};
        g = scale(g, 1.0f / (norm(g) + 1.0e-8f));
        g = unit(g, 1.0e-8f);
        const V3 m = scale(load(mesh.data() + i * 3), -1.0f);
        t[i] = dot(g, m) < 0.0f ? scale(g, -1.0f) : g;
    }
    std::vector<V3> fill(n, V3{0, 0, 0});
    std::vector<float> conf(n, 0.0f);
    auto ok = [&](int h, int w) { return valid[(size_t)h * W + w] != 0; };
    if (H > 1)
        for (int w = 0; w < W; ++w) {
            std::vector<int> prev(H), next(H);
            int last = -1;
            for (int h = 0; h < H; ++h)
                prev[h] = last = ok(h, w) ? h : last;
            last = H;
            for (int h = H - 1; h >= 0; --h)
                next[h] = last = ok(h, w) ? h : last;
            for (int h = 0; h < H; ++h) {
                if (ok(h, w) || prev[h] < 0 || next[h] >= H)
                    continue;
                const float total = std::max((float)(next[h] - prev[h]), 1.0f);
                const float u = std::max((float)(h - prev[h]), 0.0f) / total;
                const V3 a = t[(size_t)prev[h] * W + w], b = t[(size_t)next[h] * W + w];
                const V3 g = unit(add(scale(a, 1.0f - u), scale(b, u)), 1.0e-8f);
                const float k = 1.0f / total;
                fill[(size_t)h * W + w] = add(fill[(size_t)h * W + w], scale(g, k));
                conf[(size_t)h * W + w] += k;
            }
        }
    if (W > 1)
        for (int h = 0; h < H; ++h) {
            bool any = false;
            for (int w = 0; w < W; ++w)
                any = any || ok(h, w);
            if (!any)
                continue;
            const int L = 3 * W;
            std::vector<int> prev(L), next(L);
            int last = -L;
            for (int i = 0; i < L; ++i)
                prev[i] = last = ok(h, i % W) ? i : last;
            last = L;
            for (int i = L - 1; i >= 0; --i)
                next[i] = last = ok(h, i % W) ? i : last;
            for (int w = 0; w < W; ++w) {
                if (ok(h, w))
                    continue;
                const int center = W + w, pw = prev[center], nw = next[center];
                const float dp = std::max((float)(center - pw), 0.0f),
                            dn = std::max((float)(nw - center), 0.0f);
                const float total = std::max(dp + dn, 1.0f);
                const float u = dp / total;
                const V3 a = t[(size_t)h * W + (size_t)(((pw % W) + W) % W)],
                         b = t[(size_t)h * W + (size_t)(((nw % W) + W) % W)];
                const V3 g = unit(add(scale(a, 1.0f - u), scale(b, u)), 1.0e-8f);
                const float k = 1.0f / total;
                fill[(size_t)h * W + w] = add(fill[(size_t)h * W + w], scale(g, k));
                conf[(size_t)h * W + w] += k;
            }
        }
    target.assign(n * 3, 0.0f);
    mask.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        V3 out;
        if (valid[i]) {
            out = t[i];
            mask[i] = 1.0f;
        } else {
            out = unit(scale(fill[i], 1.0f / std::max(conf[i], 1.0e-8f)), 1.0e-8f);
            mask[i] = conf[i] > 0.0f ? 0.5f : 0.0f;
        }
        target[i * 3] = out.x;
        target[i * 3 + 1] = out.y;
        target[i * 3 + 2] = out.z;
    }
}

LossOut cyl_normal(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<float> &target, const std::vector<float> &mask, int factor) {
    LossOut o;
    const Shell s = shell_geometry(base, offsets, delta, has_base, factor);
    const int H = s.H, W = s.W;
    const size_t n = (size_t)H * W;
    double wsum = 0, lsum = 0;
    for (size_t i = 0; i < n; ++i)
        wsum += mask[i];
    std::vector<float> gx(n * 3, 0.0f);
    const float inv = 1.0f / (float)std::max(wsum, 1.0);
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const size_t i = (size_t)h * W + w;
            V3 dh;
            int ha = h, hb = h;
            if (H == 1)
                dh = {0, 0, 0};
            else if (h == 0) {
                ha = 0;
                hb = 1;
            } else if (h == H - 1) {
                ha = h - 1;
                hb = h;
            } else {
                ha = h - 1;
                hb = h + 1;
            }
            if (H > 1)
                dh = sub(load(s.at(hb, w)), load(s.at(ha, w)));
            const V3 dw = sub(load(s.at(h, w + 1)), load(s.at(h, w - 1)));
            const V3 c = {
                dh.y * dw.z - dh.z * dw.y, dh.z * dw.x - dh.x * dw.z, dh.x * dw.y - dh.y * dw.x};
            const float lc = norm(c), dc = std::max(lc, 1.0e-8f);
            const V3 n1 = scale(c, 1.0f / dc);
            const float l1 = norm(n1), d1 = std::max(l1, 1.0e-8f);
            const V3 n2 = scale(n1, 1.0f / d1);
            const V3 tg = load(target.data() + i * 3);
            const float raw = dot(n2, tg);
            const float d = std::clamp(raw, -1.0f, 1.0f);
            const float lm = 1.0f - std::fabs(d) * std::fabs(d);
            lsum += (double)lm * mask[i];
            if (mask[i] == 0.0f || raw < -1.0f || raw > 1.0f)
                continue;
            const float gd = -2.0f * d * mask[i] * inv;
            V3 g2 = scale(tg, gd);
            V3 g1 = l1 > 1.0e-8f ? scale(sub(g2, scale(n2, dot(n2, g2))), 1.0f / d1)
                                 : scale(g2, 1.0f / d1);
            V3 gc = lc > 1.0e-8f ? scale(sub(g1, scale(n1, dot(n1, g1))), 1.0f / dc)
                                 : scale(g1, 1.0f / dc);
            const V3 gdh = {
                dw.y * gc.z - dw.z * gc.y, dw.z * gc.x - dw.x * gc.z, dw.x * gc.y - dw.y * gc.x};
            const V3 gdw = {
                gc.y * dh.z - gc.z * dh.y, gc.z * dh.x - gc.x * dh.z, gc.x * dh.y - gc.y * dh.x};
            auto addg = [&](int hh, int ww, V3 v) {
                float *p = gx.data() + ((size_t)hh * W + (size_t)(((ww % W) + W) % W)) * 3;
                p[0] += v.x;
                p[1] += v.y;
                p[2] += v.z;
            };
            if (H > 1) {
                addg(hb, w, gdh);
                addg(ha, w, scale(gdh, -1.0f));
            }
            addg(h, w + 1, gdw);
            addg(h, w - 1, scale(gdw, -1.0f));
        }
    o.value = wsum > 0 ? lsum / std::max(wsum, 1.0) : INFINITY;
    o.grad = supersample_backward(Grid{H, W, 3, gx}, delta.H, delta.W, factor).v;
    return o;
}

LossOut cyl_outside(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<uint8_t> &valid, const DepthVolume &f, double depth_max, double model_step,
    int factor) {
    LossOut o;
    o.grad.assign(delta.v.size(), 0.0f);
    if (depth_max <= 0)
        return o;
    const Shell s = shell_geometry(base, offsets, delta, has_base, factor);
    const size_t n = (size_t)s.H * s.W;
    double wsum = 0;
    for (size_t i = 0; i < n; ++i)
        wsum += valid[i] ? 1.0 : 0.0;
    if (wsum <= 0)
        return o;
    const float inv[3] = {1.0f / f.spacing[0], 1.0f / f.spacing[1], 1.0f / f.spacing[2]};
    const int size[3] = {f.X, f.Y, f.Z};
    const float ms = (float)std::max(1.0e-6, model_step), dm = (float)depth_max;
    std::vector<float> gs(n * 3, 0.0f);
    double lsum = 0;
    const float k_mean = 1.0f / (float)std::max(wsum, 1.0);
    auto voxel = [&](int z, int y, int x) -> float {
        if (x < 0 || y < 0 || z < 0 || x >= f.X || y >= f.Y || z >= f.Z)
            return 0.0f;
        return (float)f.v[((size_t)z * f.Y + y) * f.X + x];
    };
    for (size_t i = 0; i < n; ++i) {
        const float *p = s.xyz.data() + i * 3;
        float idx[3], g[3], src[3];
        bool out = false;
        for (int a = 0; a < 3; ++a) {
            idx[a] = (p[a] - f.origin[a]) * inv[a];
            out = out || idx[a] < 0.0f || idx[a] > (float)(size[a] - 1);
            g[a] = (p[a] - f.origin[a]) * inv[a] / (float)std::max(1, size[a] - 1) * 2.0f - 1.0f;
            src[a] = ((g[a] + 1.0f) / 2.0f) * (float)(size[a] - 1);
        }
        float q = 255.0f;
        float dq[3] = {0, 0, 0};
        if (!out) {
            const float ix = src[0], iy = src[1], iz = src[2];
            const int x0 = (int)std::floor(ix), y0 = (int)std::floor(iy), z0 = (int)std::floor(iz);
            const int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
            const float fx0 = (float)x0, fy0 = (float)y0, fz0 = (float)z0, fx1 = (float)x1,
                        fy1 = (float)y1, fz1 = (float)z1;
            const float tnw = (fx1 - ix) * (fy1 - iy) * (fz1 - iz),
                        tne = (ix - fx0) * (fy1 - iy) * (fz1 - iz);
            const float tsw = (fx1 - ix) * (iy - fy0) * (fz1 - iz),
                        tse = (ix - fx0) * (iy - fy0) * (fz1 - iz);
            const float bnw = (fx1 - ix) * (fy1 - iy) * (iz - fz0),
                        bne = (ix - fx0) * (fy1 - iy) * (iz - fz0);
            const float bsw = (fx1 - ix) * (iy - fy0) * (iz - fz0),
                        bse = (ix - fx0) * (iy - fy0) * (iz - fz0);
            const float v_tnw = voxel(z0, y0, x0), v_tne = voxel(z0, y0, x1),
                        v_tsw = voxel(z0, y1, x0), v_tse = voxel(z0, y1, x1);
            const float v_bnw = voxel(z1, y0, x0), v_bne = voxel(z1, y0, x1),
                        v_bsw = voxel(z1, y1, x0), v_bse = voxel(z1, y1, x1);
            q = 0.0f;
            q += v_tnw * tnw;
            q += v_tne * tne;
            q += v_tsw * tsw;
            q += v_tse * tse;
            q += v_bnw * bnw;
            q += v_bne * bne;
            q += v_bsw * bsw;
            q += v_bse * bse;
            float gx = 0, gy = 0, gz = 0;
            gx -= v_tnw * (fy1 - iy) * (fz1 - iz);
            gy -= v_tnw * (fx1 - ix) * (fz1 - iz);
            gz -= v_tnw * (fx1 - ix) * (fy1 - iy);
            gx += v_tne * (fy1 - iy) * (fz1 - iz);
            gy -= v_tne * (ix - fx0) * (fz1 - iz);
            gz -= v_tne * (ix - fx0) * (fy1 - iy);
            gx -= v_tsw * (iy - fy0) * (fz1 - iz);
            gy += v_tsw * (fx1 - ix) * (fz1 - iz);
            gz -= v_tsw * (fx1 - ix) * (iy - fy0);
            gx += v_tse * (iy - fy0) * (fz1 - iz);
            gy += v_tse * (ix - fx0) * (fz1 - iz);
            gz -= v_tse * (ix - fx0) * (iy - fy0);
            gx -= v_bnw * (fy1 - iy) * (iz - fz0);
            gy -= v_bnw * (fx1 - ix) * (iz - fz0);
            gz += v_bnw * (fx1 - ix) * (fy1 - iy);
            gx += v_bne * (fy1 - iy) * (iz - fz0);
            gy -= v_bne * (ix - fx0) * (iz - fz0);
            gz += v_bne * (ix - fx0) * (fy1 - iy);
            gx -= v_bsw * (iy - fy0) * (iz - fz0);
            gy += v_bsw * (fx1 - ix) * (iz - fz0);
            gz += v_bsw * (fx1 - ix) * (iy - fy0);
            gx += v_bse * (iy - fy0) * (iz - fz0);
            gy += v_bse * (ix - fx0) * (iz - fz0);
            gz += v_bse * (ix - fx0) * (iy - fy0);
            const float gsrc[3] = {gx, gy, gz};
            for (int a = 0; a < 3; ++a) {
                const float to_norm = (float)(size[a] - 1) / 2.0f;
                const float from_xyz = inv[a] / (float)std::max(1, size[a] - 1) * 2.0f;
                dq[a] = gsrc[a] * to_norm * from_xyz;
            }
        }
        const float u = q / 255.0f;
        const float vd = u * u * dm;
        const float r = vd / ms;
        const float lm = r * r;
        if (!valid[i])
            continue;
        lsum += lm;
        if (out)
            continue;
        const float dlm_dq = 2.0f * r / ms * (2.0f * u / 255.0f * dm) * k_mean;
        for (int a = 0; a < 3; ++a)
            gs[i * 3 + (size_t)a] = dlm_dq * dq[a];
    }
    o.value = lsum / std::max(wsum, 1.0);
    o.grad = supersample_backward(Grid{s.H, s.W, 3, gs}, delta.H, delta.W, factor).v;
    return o;
}

}
