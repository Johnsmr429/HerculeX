#include "losses.h"

#include <algorithm>
#include <cmath>

namespace sp {

static constexpr float TWO_PI = 6.2831855f;

struct Angle {
    float theta, radius, xg;
    bool guard;
};

static inline Angle angle_of(float y, float x) {
    Angle a;
    a.guard = std::fabs(x) < 1e-10f;
    a.xg = a.guard ? 1e-10f : x;
    const float t = std::atan2(y, a.xg);
    a.theta = t - TWO_PI * std::floor(t / TWO_PI);
    a.radius = std::sqrt(y * y + a.xg * a.xg);
    return a;
}

static inline void angle_backward(
    const Angle &a, float y, float gtheta, float gradius, float &gy, float &gx) {
    const float r2 = y * y + a.xg * a.xg;
    gy += gtheta * (a.xg / r2) + gradius * (y / a.radius);
    if (!a.guard)
        gx += gtheta * (-y / r2) + gradius * (a.xg / a.radius);
}

static inline float sgn(float v) { return v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f); }

void draw_samples(const Model &m, const Shared &sh, const TrackPool &pool, const LossSettings &ls,
    std::mt19937_64 &rng, Samples &s) {
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    const float zb = (float)m.s.z_begin, ze = (float)(m.s.z_end - 1);
    s.tracks = draw_tracks(pool, ls.tracks, ls.track_points, ls.max_crossings, rng);
    std::vector<double> cumulative;
    double total = 0;
    for (int k = 1; k < ls.outer_winding; ++k)
        cumulative.push_back(total += (double)((float)k + 0.5f));
    s.sym_points.resize((size_t)ls.regularisation_points * 3);
    s.sym_tangent.resize((size_t)ls.regularisation_points * 3);
    for (int i = 0; i < ls.regularisation_points; ++i) {
        const double u = (double)unit(rng) * total;
        const size_t j =
            std::min((size_t)(std::upper_bound(cumulative.begin(), cumulative.end(), u) -
                         cumulative.begin()),
                cumulative.size() - 1);
        const float winding = (float)j + 1.0f;
        const float theta = unit(rng) * TWO_PI;
        const float radius = (winding + theta / TWO_PI) * sh.dr;
        const float z = zb + unit(rng) * (ze - zb);
        const float dr_dtheta = sh.dr / TWO_PI;
        const float ty = std::cos(theta) * radius + std::sin(theta) * dr_dtheta;
        const float tx = -std::sin(theta) * radius + std::cos(theta) * dr_dtheta;
        const float tn = std::max(std::sqrt(ty * ty + tx * tx), 1e-12f);
        const size_t at = (size_t)i * 3;
        s.sym_points[at] = z;
        s.sym_points[at + 1] = std::sin(theta) * radius;
        s.sym_points[at + 2] = std::cos(theta) * radius;
        s.sym_tangent[at] = 0.0f;
        s.sym_tangent[at + 1] = ty / tn;
        s.sym_tangent[at + 2] = tx / tn;
    }
    const float rmax = sh.dr * (float)ls.outer_winding, rmin = sh.dr;
    s.normal_points.resize((size_t)ls.normal_points * 3);
    for (int i = 0; i < ls.normal_points; ++i) {
        const float theta = unit(rng) * TWO_PI;
        const float radius = std::sqrt(unit(rng) * (rmax * rmax - rmin * rmin) + rmin * rmin);
        const float z = zb + unit(rng) * (ze - zb);
        const size_t at = (size_t)i * 3;
        s.normal_points[at] = z;
        s.normal_points[at + 1] = std::sin(theta) * radius;
        s.normal_points[at + 2] = std::cos(theta) * radius;
    }
    const float a = 1.0f, b = (float)(ls.outer_winding - 2);
    const float mass = ((b + 1.0f) * (b + 1.0f) - a * a) / 2.0f;
    s.spacing_winding.resize((size_t)ls.spacing_points);
    s.spacing_theta.resize((size_t)ls.spacing_points);
    s.spacing_z.resize((size_t)ls.spacing_points);
    for (int i = 0; i < ls.spacing_points; ++i) {
        const float k = std::ceil(std::sqrt(2.0f * unit(rng) * mass + a * a) - 1.0f);
        s.spacing_winding[(size_t)i] = (int32_t)std::min(std::max(k, a), b);
        s.spacing_theta[(size_t)i] = unit(rng) * TWO_PI;
        s.spacing_z[(size_t)i] = zb + unit(rng) * (ze - zb);
    }
}

double umbilicus_loss(
    const Model &m, const Shared &sh, const std::vector<float> &pts, float weight, Grads &g) {
    const size_t n = pts.size() / 3;
    std::vector<float> out(pts.size()), grad(pts.size(), 0.0f);
    slice_to_spiral(m, sh, pts.data(), n, out.data());
    double sum = 0;
    const float scale = weight / (float)(2 * n);
    for (size_t i = 0; i < n; ++i) {
        sum += std::fabs(out[i * 3 + 1]) + std::fabs(out[i * 3 + 2]);
        grad[i * 3 + 1] = sgn(out[i * 3 + 1]) * scale;
        grad[i * 3 + 2] = sgn(out[i * 3 + 2]) * scale;
    }
    slice_to_spiral_backward(m, sh, pts.data(), n, grad.data(), nullptr, g);
    return sum / (double)(2 * n);
}

double sym_loss(
    const Model &m, const Shared &sh, const Samples &s, const LossSettings &ls, Grads &g) {
    const size_t n = s.sym_points.size() / 3;
    if (!n)
        return 0;
    const float eps = ls.sym_epsilon;
    std::vector<float> pts(n * 9), out(n * 9), grad(n * 9, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            const float p = s.sym_points[i * 3 + c];
            pts[i * 3 + c] = p;
            pts[(n + i) * 3 + c] = p + eps * (c == 0 ? 1.0f : 0.0f);
            pts[(2 * n + i) * 3 + c] = p + eps * s.sym_tangent[i * 3 + c];
        }
    spiral_to_slice(m, sh, pts.data(), 3 * n, out.data());
    double sum = 0;
    const float scale = ls.weight_sym / (float)n;
    for (size_t i = 0; i < n; ++i) {
        float a[3], b[3];
        for (int c = 0; c < 3; ++c) {
            a[c] = (out[(n + i) * 3 + c] - out[i * 3 + c]) / eps;
            b[c] = (out[(2 * n + i) * 3 + c] - out[i * 3 + c]) / eps;
        }
        const float g11 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        const float g22 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        const float g12 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        const float tr = g11 + g22, det = g11 * g22 - g12 * g12;
        const float num = tr + 2.0f * 1e-3f, den = det + 1e-3f * tr + 1e-3f * 1e-3f;
        const float raw = tr + num / den - 4.0f;
        const float energy = std::min(std::max(raw, 0.0f), 100.0f);
        sum += energy;
        if (!(raw >= 0.0f && raw <= 100.0f))
            continue;
        const float dtr = 1.0f + (den - num * 1e-3f) / (den * den);
        const float ddet = -num / (den * den);
        const float d11 = dtr + ddet * g22, d22 = dtr + ddet * g11, d12 = ddet * (-2.0f * g12);
        for (int c = 0; c < 3; ++c) {
            const float ga = (2.0f * a[c] * d11 + b[c] * d12) * scale,
                        gb = (2.0f * b[c] * d22 + a[c] * d12) * scale;
            grad[(n + i) * 3 + c] += ga / eps;
            grad[(2 * n + i) * 3 + c] += gb / eps;
            grad[i * 3 + c] += -(ga + gb) / eps;
        }
    }
    spiral_to_slice_backward(m, sh, pts.data(), 3 * n, grad.data(), nullptr, g);
    return sum / (double)n;
}

double normals_loss(const Model &m, const Shared &sh, const Lasagna &L, const Samples &s,
    const LossSettings &ls, Grads &g) {
    const size_t n = s.normal_points.size() / 3;
    if (!n || L.nx.empty())
        return 0;
    const float half = sh.dr / 2;
    std::vector<float> pts(n * 9), out(n * 9);
    for (size_t i = 0; i < n; ++i) {
        const float y = s.normal_points[i * 3 + 1], x = s.normal_points[i * 3 + 2];
        const float r = std::sqrt(y * y + x * x);
        const float sy = r > 0 ? y / r : 0.0f, cx = r > 0 ? x / r : 1.0f;
        const float z = s.normal_points[i * 3];
        const float inner = r - half, outer = r + half;
        pts[i * 3] = z;
        pts[i * 3 + 1] = sy * inner;
        pts[i * 3 + 2] = cx * inner;
        pts[(n + i) * 3] = z;
        pts[(n + i) * 3 + 1] = sy * outer;
        pts[(n + i) * 3 + 2] = cx * outer;
        for (int c = 0; c < 3; ++c)
            pts[(2 * n + i) * 3 + c] = s.normal_points[i * 3 + c];
    }
    spiral_to_slice(m, sh, pts.data(), 3 * n, out.data());
    const float eps = ls.normals_epsilon;
    std::vector<float> fd(n * 18), fd_out(n * 18), grad(n * 18, 0.0f);
    for (int k = 0; k < 3; ++k)
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                const float base = out[(2 * n + i) * 3 + c];
                fd[((size_t)k * n + i) * 3 + c] = base + (c == k ? eps : 0.0f);
                fd[((size_t)(3 + k) * n + i) * 3 + c] = base - (c == k ? eps : 0.0f);
            }
    slice_to_spiral(m, sh, fd.data(), 6 * n, fd_out.data());
    std::vector<float> target(n * 3), weight(n), cov(n * 3), outward(n * 3);
    double weight_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t idx[3];
        bool inside = true;
        const int64_t origin[3] = {L.z_origin, 0, 0};
        for (int c = 0; c < 3; ++c) {
            const float mid = (out[i * 3 + c] + out[(n + i) * 3 + c]) / 2;
            idx[c] = (int64_t)std::nearbyint(mid / (float)L.scale) - origin[c];
            inside &= idx[c] >= 0 && idx[c] < L.shape[(size_t)c];
            idx[c] = std::min(std::max<int64_t>(idx[c], 0), L.shape[(size_t)c] - 1);
        }
        const size_t at = L.at(idx[0], idx[1], idx[2]);
        const uint8_t ux = L.nx[at], uy = L.ny[at];
        const float vx = ((float)ux - 128.0f) / 127.0f, vy = ((float)uy - 128.0f) / 127.0f;
        const float vz = std::sqrt(std::max(1.0f - vx * vx - vy * vy, 0.0f));
        const float vn = std::max(std::sqrt(vz * vz + vy * vy + vx * vx), 1e-12f);
        target[i * 3] = vz / vn;
        target[i * 3 + 1] = vy / vn;
        target[i * 3 + 2] = vx / vn;
        weight[i] = ((ux != 0 || uy != 0) && inside) ? 1.0f : 0.0f;
        weight_sum += weight[i];
        const float sy = s.normal_points[i * 3 + 1], sx = s.normal_points[i * 3 + 2];
        const float sn = std::max(std::sqrt(sy * sy + sx * sx), 1e-12f);
        outward[i * 3] = 0.0f;
        outward[i * 3 + 1] = sy / sn;
        outward[i * 3 + 2] = sx / sn;
        for (int k = 0; k < 3; ++k) {
            float v = 0;
            for (int c = 0; c < 3; ++c)
                v += (fd_out[((size_t)k * n + i) * 3 + c] -
                         fd_out[((size_t)(3 + k) * n + i) * 3 + c]) /
                    (2.0f * eps) * outward[i * 3 + c];
            cov[i * 3 + k] = v;
        }
    }
    const float denom = (float)std::max(weight_sum, 1.0);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const float cn = std::sqrt(cov[i * 3] * cov[i * 3] + cov[i * 3 + 1] * cov[i * 3 + 1] +
            cov[i * 3 + 2] * cov[i * 3 + 2]);
        const float norm = std::max(cn, 1e-12f);
        float nrm[3], dot = 0;
        for (int c = 0; c < 3; ++c)
            dot += (nrm[c] = cov[i * 3 + c] / norm) * target[i * 3 + c];
        sum += (1.0f - std::fabs(dot)) * weight[i];
        if (weight[i] == 0.0f)
            continue;
        const float gscale = -sgn(dot) * weight[i] / denom * ls.weight_normals;
        float gn[3], gdot = 0;
        for (int c = 0; c < 3; ++c)
            gdot += (gn[c] = gscale * target[i * 3 + c]) * nrm[c];
        float gc[3];
        for (int c = 0; c < 3; ++c)
            gc[c] = cn > 1e-12f ? (gn[c] - nrm[c] * gdot) / norm : gn[c] / norm;
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                const float v = gc[k] * outward[i * 3 + c] / (2.0f * eps);
                grad[((size_t)k * n + i) * 3 + c] += v;
                grad[((size_t)(3 + k) * n + i) * 3 + c] -= v;
            }
    }
    slice_to_spiral_backward(m, sh, fd.data(), 6 * n, grad.data(), nullptr, g);
    return sum / (double)denom;
}

double spacing_loss(
    const Model &m, const Shared &sh, const Samples &s, const LossSettings &ls, Grads &g) {
    const size_t n = s.spacing_winding.size();
    if (!n)
        return 0;
    const float floor_log = (float)std::log((double)ls.min_spacing);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const float ell =
            native_log_gap(m, sh, s.spacing_winding[i], s.spacing_theta[i], s.spacing_z[i]);
        const float deficiency = std::max(floor_log - ell, 0.0f);
        sum += deficiency * deficiency;
        if (deficiency > 0)
            native_log_gap_backward(m, sh, s.spacing_winding[i], s.spacing_theta[i], s.spacing_z[i],
                -2.0f * deficiency / (float)n * ls.weight_spacing, g);
    }
    return sum / (double)n;
}

double track_loss(
    const Model &m, const Shared &sh, const TrackSample &t, const LossSettings &ls, Grads &g) {
    const size_t n = t.points.size() / 3, rows = t.rows();
    if (!n)
        return 0;
    std::vector<float> out(n * 3), grad(n * 3, 0.0f);
    slice_to_spiral(m, sh, t.points.data(), n, out.data());
    std::vector<Angle> ang(n);
    std::vector<float> shifted(n), raw(n);
    for (size_t i = 0; i < n; ++i) {
        ang[i] = angle_of(out[i * 3 + 1], out[i * 3 + 2]);
        raw[i] = ang[i].radius - ang[i].theta / TWO_PI * sh.dr;
        shifted[i] = std::max(raw[i], 0.0f);
    }
    std::vector<float> adjust(n, 0.0f);
    float running = 0;
    std::vector<float> cumulative(n, 0.0f);
    for (size_t i = 1; i < n; ++i) {
        float step = 0;
        if (t.row_id[i] == t.row_id[i - 1]) {
            const float d = ang[i].theta - ang[i - 1].theta;
            step = ((d > (float)M_PI) ? 1.0f : 0.0f) - ((d < -(float)M_PI) ? 1.0f : 0.0f);
            step *= sh.dr;
        }
        running += step;
        cumulative[i] = running;
    }
    for (size_t i = 0; i < n; ++i)
        adjust[i] = cumulative[i] - cumulative[(size_t)t.row_starts[(size_t)t.row_id[i]]];
    std::vector<float> unwrapped(n);
    for (size_t i = 0; i < n; ++i)
        unwrapped[i] = shifted[i] + adjust[i];
    std::vector<float> alignment(rows, 0.0f);
    for (size_t p = 0; p < t.partner_rows.size(); ++p)
        alignment[(size_t)t.partner_rows[p]] =
            unwrapped[(size_t)t.primary_cross[p]] - unwrapped[(size_t)t.partner_cross[p]];
    for (size_t i = 0; i < n; ++i)
        unwrapped[i] += alignment[(size_t)t.row_id[i]];
    const int G = t.num_groups, T = t.target_points;
    std::vector<double> target_sum((size_t)G, 0.0);
    std::vector<int> target_count((size_t)G, 0);
    for (size_t r = 0; r < rows; ++r)
        for (int k = 0; k < T; ++k) {
            target_sum[(size_t)t.group_id[r]] +=
                unwrapped[(size_t)t.target_flat[r * (size_t)T + (size_t)k]];
            ++target_count[(size_t)t.group_id[r]];
        }
    std::vector<float> target((size_t)G);
    for (int q = 0; q < G; ++q)
        target[(size_t)q] = (float)(target_sum[(size_t)q] / std::max(target_count[(size_t)q], 1));
    const float margin = sh.dr * ls.track_margin, p = ls.track_p;
    std::vector<double> sums((size_t)G, 0.0);
    std::vector<int> counts((size_t)G, 0);
    std::vector<float> hinged(n);
    for (size_t i = 0; i < n; ++i) {
        const int q = t.group_id[(size_t)t.row_id[i]];
        hinged[i] = std::max(std::fabs(unwrapped[i] - target[(size_t)q]) - margin, 0.0f);
        sums[(size_t)q] += std::pow((double)hinged[i] + 1e-5, (double)p);
        ++counts[(size_t)q];
    }
    double loss = 0;
    std::vector<double> dgroup((size_t)G, 0.0);
    for (int q = 0; q < G; ++q) {
        const double mean = sums[(size_t)q] / std::max(counts[(size_t)q], 1);
        loss += std::pow(mean, 1.0 / p);
        dgroup[(size_t)q] =
            mean > 0 ? std::pow(mean, 1.0 / p - 1.0) / p / std::max(counts[(size_t)q], 1) / G : 0.0;
    }
    loss /= G;
    std::vector<float> gu(n, 0.0f);
    std::vector<double> gtarget((size_t)G, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const int q = t.group_id[(size_t)t.row_id[i]];
        if (hinged[i] <= 0)
            continue;
        const double d = dgroup[(size_t)q] * p *
            std::pow((double)hinged[i] + 1e-5, (double)p - 1.0) * ls.weight_track;
        const float s = sgn(unwrapped[i] - target[(size_t)q]);
        gu[i] += (float)(d * s);
        gtarget[(size_t)q] -= d * s;
    }
    for (size_t r = 0; r < rows; ++r) {
        const int q = t.group_id[r];
        const double share = gtarget[(size_t)q] / std::max(target_count[(size_t)q], 1);
        for (int k = 0; k < T; ++k)
            gu[(size_t)t.target_flat[r * (size_t)T + (size_t)k]] += (float)share;
    }
    double gdr = 0;
    for (size_t i = 0; i < n; ++i) {
        if (raw[i] < 0.0f || gu[i] == 0.0f)
            continue;
        const float gtheta = -gu[i] * sh.dr / TWO_PI;
        gdr += -(double)gu[i] * ang[i].theta / TWO_PI;
        float gy = 0, gx = 0;
        angle_backward(ang[i], out[i * 3 + 1], gtheta, gu[i], gy, gx);
        grad[i * 3 + 1] = gy;
        grad[i * 3 + 2] = gx;
    }
    g.dr += gdr;
    slice_to_spiral_backward(m, sh, t.points.data(), n, grad.data(), nullptr, g);
    return loss;
}

LossValues evaluate(const Model &m, const Shared &sh, const Lasagna &lasagna,
    const std::vector<float> &umbilicus_points, const Samples &s, const LossSettings &ls,
    Grads &g) {
    LossValues v;
    v.umbilicus = umbilicus_loss(m, sh, umbilicus_points, ls.weight_umbilicus, g);
    v.sym = sym_loss(m, sh, s, ls, g);
    v.normals = normals_loss(m, sh, lasagna, s, ls, g);
    v.spacing = spacing_loss(m, sh, s, ls, g);
    v.track = track_loss(m, sh, s.tracks, ls, g);
    return v;
}

}
