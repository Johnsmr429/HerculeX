#include "pyramid.h"

#include <algorithm>
#include <cmath>

namespace la {

namespace {

struct Taps {
    int i0, i1;
    float l0, l1;
};

std::vector<Taps> taps(int in, int out) {
    std::vector<Taps> t((size_t)out);
    const float r = (in > 1 && out > 1) ? (float)(in - 1) / (float)(out - 1) : 0.0f;
    for (int o = 0; o < out; ++o) {
        const float src = r * (float)o;
        const int i0 = std::min((int)src, in - 1);
        const int p = i0 < in - 1 ? 1 : 0;
        const float l1 = src - (float)i0;
        t[(size_t)o] = {i0, i0 + p, 1.0f - l1, l1};
    }
    return t;
}

}

Grid resize_bilinear(const Grid &in, int out_h, int out_w) {
    Grid out{out_h, out_w, in.C, std::vector<float>((size_t)out_h * out_w * in.C)};
    const auto th = taps(in.H, out_h), tw = taps(in.W, out_w);
    for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x)
            for (int c = 0; c < in.C; ++c) {
                const Taps &a = th[(size_t)y], &b = tw[(size_t)x];
                const float top = b.l0 * in.at(a.i0, b.i0)[c] + b.l1 * in.at(a.i0, b.i1)[c];
                const float bottom = b.l0 * in.at(a.i1, b.i0)[c] + b.l1 * in.at(a.i1, b.i1)[c];
                out.at(y, x)[c] = a.l0 * top + a.l1 * bottom;
            }
    return out;
}

Grid resize_bilinear_backward(const Grid &g, int in_h, int in_w) {
    Grid out{in_h, in_w, g.C, std::vector<float>((size_t)in_h * in_w * g.C, 0.0f)};
    const auto th = taps(in_h, g.H), tw = taps(in_w, g.W);
    for (int y = 0; y < g.H; ++y)
        for (int x = 0; x < g.W; ++x)
            for (int c = 0; c < g.C; ++c) {
                const Taps &a = th[(size_t)y], &b = tw[(size_t)x];
                const float v = g.at(y, x)[c];
                out.at(a.i0, b.i0)[c] += a.l0 * b.l0 * v;
                out.at(a.i0, b.i1)[c] += a.l0 * b.l1 * v;
                out.at(a.i1, b.i0)[c] += a.l1 * b.l0 * v;
                out.at(a.i1, b.i1)[c] += a.l1 * b.l1 * v;
            }
    return out;
}

std::vector<Grid> build_pyramid(const Grid &delta, int scales) {
    std::vector<Grid> targets = {delta};
    for (int k = 1; k < scales; ++k) {
        const Grid &prev = targets.back();
        targets.push_back(
            resize_bilinear(prev, std::max(2, (prev.H + 1) / 2), std::max(2, (prev.W + 1) / 2)));
    }
    std::vector<Grid> residuals(targets.size());
    residuals.back() = targets.back();
    Grid recon = targets.back();
    for (int i = (int)targets.size() - 2; i >= 0; --i) {
        const Grid up = resize_bilinear(recon, targets[(size_t)i].H, targets[(size_t)i].W);
        Grid r = targets[(size_t)i];
        for (size_t j = 0; j < r.v.size(); ++j)
            r.v[j] = targets[(size_t)i].v[j] - up.v[j];
        recon = up;
        for (size_t j = 0; j < recon.v.size(); ++j)
            recon.v[j] = up.v[j] + r.v[j];
        residuals[(size_t)i] = std::move(r);
    }
    return residuals;
}

Grid integrate_pyramid(const std::vector<Grid> &levels) {
    Grid v = levels.back();
    for (int i = (int)levels.size() - 2; i >= 0; --i) {
        Grid up = resize_bilinear(v, levels[(size_t)i].H, levels[(size_t)i].W);
        for (size_t j = 0; j < up.v.size(); ++j)
            up.v[j] = up.v[j] + levels[(size_t)i].v[j];
        v = std::move(up);
    }
    return v;
}

std::vector<Grid> integrate_pyramid_backward(const std::vector<Grid> &levels, const Grid &grad) {
    std::vector<Grid> out(levels.size());
    Grid g = grad;
    for (size_t i = 0; i + 1 < levels.size(); ++i) {
        out[i] = g;
        g = resize_bilinear_backward(g, levels[i + 1].H, levels[i + 1].W);
    }
    out.back() = g;
    return out;
}

void Adam::update(
    std::vector<Grid> &params, const std::vector<Grid> &grads, const std::vector<double> &lr) {
    if (first.empty()) {
        for (const Grid &p : params) {
            first.emplace_back(p.v.size(), 0.0f);
            second.emplace_back(p.v.size(), 0.0f);
        }
    }
    ++step;
    const double bc1 = 1.0 - std::pow(beta1, (double)step),
                 bc2 = 1.0 - std::pow(beta2, (double)step);
    const float b1 = (float)(1.0 - beta1), b2 = (float)beta2, c2 = (float)(1.0 - beta2);
    const float root = (float)std::sqrt(bc2), e = (float)eps;
    for (size_t k = 0; k < params.size(); ++k) {
        const float step_size = (float)(lr[k] / bc1);
        std::vector<float> &m = first[k], &v = second[k];
        for (size_t j = 0; j < params[k].v.size(); ++j) {
            const float g = grads[k].v[j];
            m[j] = b1 < 0.5f ? m[j] + b1 * (g - m[j]) : g - (g - m[j]) * (1.0f - b1);
            v[j] = v[j] * b2 + c2 * g * g;
            const float denom = std::sqrt(v[j]) / root + e;
            params[k].v[j] = params[k].v[j] + (-step_size) * (m[j] / denom);
        }
    }
}

}
