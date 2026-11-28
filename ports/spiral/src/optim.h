#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace sp {

struct AdamW {
    double lr = 0, weight_decay = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    std::vector<float> m, v;
    long step = 0;

    void update(float *p, const float *g, size_t n, double lr_now) {
        if (m.size() != n) {
            m.assign(n, 0.0f);
            v.assign(n, 0.0f);
        }
        ++step;
        const float decay = (float)(1.0 - lr_now * weight_decay);
        const double bc1 = 1.0 - std::pow(beta1, (double)step),
                     bc2 = 1.0 - std::pow(beta2, (double)step);
        const float step_size = (float)(lr_now / bc1), root = (float)std::sqrt(bc2);
        const float b1 = (float)beta1, b2 = (float)beta2, e = (float)eps;
        for (size_t i = 0; i < n; ++i) {
            const float gi = std::isfinite(g[i]) ? g[i] : 0.0f;
            if (weight_decay != 0.0)
                p[i] *= decay;
            m[i] = m[i] + (gi - m[i]) * (1.0f - b1);
            v[i] = v[i] * b2 + gi * gi * (1.0f - b2);
            p[i] -= step_size * m[i] / (std::sqrt(v[i]) / root + e);
        }
    }
};

}
