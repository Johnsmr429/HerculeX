#include "cylinder.h"

#include "io.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace la {

static double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
    if (x <= xp.front())
        return fp.front();
    if (x >= xp.back())
        return fp.back();
    const size_t j = (size_t)(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin());
    const double t = (x - xp[j - 1]) / (xp[j] - xp[j - 1]);
    return fp[j - 1] + t * (fp[j] - fp[j - 1]);
}

Umbilicus load_umbilicus(const std::string &path, double z_step) {
    Json data = Json::parse(read_text(path));
    std::vector<std::array<double, 3>> rows;
    for (const Json &p : data["control_points"].items)
        rows.push_back({p["x"].num(), p["y"].num(), p["z"].num()});
    std::stable_sort(
        rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a[2] < b[2]; });
    std::vector<double> zs, xs, ys;
    for (size_t i = 0; i < rows.size();) {
        size_t j = i;
        double sx = 0, sy = 0;
        while (j < rows.size() && rows[j][2] == rows[i][2]) {
            sx += rows[j][0];
            sy += rows[j][1];
            ++j;
        }
        zs.push_back(rows[i][2]);
        xs.push_back(sx / (double)(j - i));
        ys.push_back(sy / (double)(j - i));
        i = j;
    }
    Umbilicus u;
    u.step = z_step;
    u.z0 = std::floor(zs.front() / z_step) * z_step;
    const double z1 = std::ceil(zs.back() / z_step) * z_step;
    const int count = std::max(1, (int)std::llround((z1 - u.z0) / z_step) + 1);
    for (int k = 0; k < count; ++k) {
        const double z = u.z0 + (double)k * z_step;
        u.table.push_back((float)interp(z, zs, xs));
        u.table.push_back((float)interp(z, zs, ys));
    }
    return u;
}

void Umbilicus::xy(float z, float &x, float &y) const {
    const int M = (int)(table.size() / 2);
    if (M == 1) {
        x = table[0];
        y = table[1];
        return;
    }
    const float t = std::clamp((z - (float)z0) / (float)step, 0.0f, (float)(M - 1));
    const int i0 = std::clamp((int)std::floor(t), 0, M - 1), i1 = std::min(i0 + 1, M - 1);
    const float frac = t - (float)i0;
    x = table[(size_t)i0 * 2] + frac * (table[(size_t)i1 * 2] - table[(size_t)i0 * 2]);
    y = table[(size_t)i0 * 2 + 1] + frac * (table[(size_t)i1 * 2 + 1] - table[(size_t)i0 * 2 + 1]);
}

std::vector<float> linspace(float start, float end, int steps) {
    std::vector<float> out((size_t)steps);
    if (steps == 1) {
        out[0] = start;
        return out;
    }
    const float step = (end - start) / (float)(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i)
        out[(size_t)i] =
            i < halfway ? start + step * (float)i : end - step * (float)(steps - i - 1);
    return out;
}

int shell_width_for_radius(double radius, double width_step) {
    radius = std::max(1.0, radius);
    const double target = std::max(1.0, width_step);
    const double max_step = 2.0 * radius * std::sin(M_PI / 3.0);
    if (target >= max_step)
        return 3;
    const double ratio = std::min(std::max(target / (2.0 * radius), 1.0e-12), std::sin(M_PI / 3.0));
    const int near = (int)std::nearbyint(M_PI / std::asin(ratio));
    int best = 3;
    double best_err = std::fabs(2.0 * radius * std::sin(M_PI / 3.0) - target);
    for (int w = std::max(3, near - 3); w <= std::max(3, near + 3); ++w) {
        const double err = std::fabs(2.0 * radius * std::sin(M_PI / (double)w) - target);
        if (err < best_err || (err == best_err && w < best)) {
            best = w;
            best_err = err;
        }
    }
    return best;
}

TubeStart tube_start(const Umbilicus &u, double z_center, double model_h, double z_step,
    double radius, double width_step) {
    TubeStart t;
    const double step = std::max(1.0, z_step), height = std::max(1.0, model_h);
    const double z0 = z_center - 0.5 * height, z1 = z_center + 0.5 * height;
    const double dense_step = std::max(1.0, step * 0.25);
    const int n_dense = std::max(2, (int)std::ceil(std::fabs(z1 - z0) / dense_step) + 1);
    const std::vector<float> zd = linspace((float)z0, (float)z1, n_dense);
    std::vector<float> px(zd.size()), py(zd.size());
    for (size_t i = 0; i < zd.size(); ++i)
        u.xy(zd[i], px[i], py[i]);
    std::vector<float> cum(zd.size(), 0.0f);
    double acc = 0;
    for (size_t i = 1; i < zd.size(); ++i) {
        const float dx = px[i] - px[i - 1], dy = py[i] - py[i - 1], dz = zd[i] - zd[i - 1];
        acc += (double)std::sqrt(dx * dx + dy * dy + dz * dz);
        cum[i] = (float)acc;
    }
    const double total = (double)cum.back();
    const int H = std::max(2, (int)std::ceil(total / step) + 1);
    const std::vector<float> s = linspace(0.0f, (float)total, H);
    t.z.resize((size_t)H);
    for (int k = 0; k < H; ++k) {
        int idx = (int)(std::lower_bound(cum.begin(), cum.end(), s[(size_t)k]) - cum.begin());
        idx = std::clamp(idx, 1, n_dense - 1);
        const float s0 = cum[(size_t)idx - 1], s1 = cum[(size_t)idx];
        const float alpha = (s[(size_t)k] - s0) / std::max(s1 - s0, 1.0e-8f);
        t.z[(size_t)k] = zd[(size_t)idx - 1] + alpha * (zd[(size_t)idx] - zd[(size_t)idx - 1]);
    }
    t.z.front() = zd.front();
    t.z.back() = zd.back();
    const int W = shell_width_for_radius(radius, width_step);
    t.base = Grid{H, W, 3, std::vector<float>((size_t)H * W * 3)};
    t.dirs = Grid{H, W, 3, std::vector<float>((size_t)H * W * 3)};
    t.delta = Grid{H, W, 3, std::vector<float>((size_t)H * W * 3)};
    const float da = (float)(2.0 * M_PI / (double)W);
    for (int h = 0; h < H; ++h) {
        float ux, uy;
        u.xy(t.z[(size_t)h], ux, uy);
        for (int w = 0; w < W; ++w) {
            const float a = (float)w * da;
            const float c = (float)std::cos((double)a), sn = (float)std::sin((double)a);
            float *b = t.base.at(h, w), *d = t.dirs.at(h, w), *e = t.delta.at(h, w);
            b[0] = ux;
            b[1] = uy;
            b[2] = t.z[(size_t)h];
            d[0] = c;
            d[1] = sn;
            d[2] = 0.0f;
            e[0] = c * (float)radius;
            e[1] = sn * (float)radius;
            e[2] = 0.0f * (float)radius;
        }
    }
    if (H > 1) {
        double hs = 0;
        for (int h = 0; h + 1 < H; ++h) {
            const float *a = t.base.at(h, 0), *b = t.base.at(h + 1, 0);
            const float dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
            hs += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        t.height_step = hs / (double)(H - 1);
    }
    return t;
}

}
