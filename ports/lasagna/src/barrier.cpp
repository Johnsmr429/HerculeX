#include "barrier.h"

#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/WindingNumberAABB.h>
#include <igl/signed_distance.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace la {

namespace {

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXiR = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVector3d = Eigen::Matrix<double, 1, 3>;

struct Mesh {
    MatrixXdR V;
    MatrixXiR F;
};

Mesh capped_mesh(const Shell &s) {
    const int H = s.H, W = s.W;
    if (H < 2 || W < 3)
        throw std::runtime_error("capped shell mesh needs H>=2 and W>=3");
    Mesh m;
    m.V.resize((Eigen::Index)H * W + 2, 3);
    for (int i = 0; i < H * W; ++i)
        for (int c = 0; c < 3; ++c)
            m.V(i, c) = (double)s.xyz[(size_t)i * 3 + c];
    for (int c = 0; c < 3; ++c) {
        double bottom = 0, top = 0;
        for (int w = 0; w < W; ++w) {
            bottom += (double)s.at(0, w)[c];
            top += (double)s.at(H - 1, w)[c];
        }
        m.V(H * W, c) = bottom / (double)W;
        m.V(H * W + 1, c) = top / (double)W;
    }
    std::vector<std::array<int, 3>> faces;
    for (int h = 0; h + 1 < H; ++h)
        for (int w = 0; w < W; ++w) {
            const int p00 = h * W + w, p01 = h * W + (w + 1) % W, p10 = (h + 1) * W + w,
                      p11 = (h + 1) * W + (w + 1) % W;
            faces.push_back({p00, p11, p10});
            faces.push_back({p00, p01, p11});
        }
    for (int w = 0; w < W; ++w) {
        const int w1 = (w + 1) % W;
        faces.push_back({H * W, w1, w});
        faces.push_back({H * W + 1, (H - 1) * W + w, (H - 1) * W + w1});
    }
    m.F.resize((Eigen::Index)faces.size(), 3);
    for (size_t f = 0; f < faces.size(); ++f)
        for (int c = 0; c < 3; ++c)
            m.F((Eigen::Index)f, c) = faces[f][(size_t)c];
    return m;
}

struct Chunk {
    int64_t x0, x1, y0, y1, z0, z1;
};

uint8_t encode_depth(double depth, double depth_max) {
    if (!(depth > 0.0) || !(depth_max > 0.0) || !std::isfinite(depth_max))
        return 0;
    const double normalized = std::min(1.0, std::max(0.0, depth / depth_max));
    const long q = std::lround(255.0 * std::sqrt(normalized));
    return (uint8_t)std::min<long>(255, std::max<long>(0, q));
}

double positive(double depth) { return (!(depth > 0.0) || !std::isfinite(depth)) ? 0.0 : depth; }

double capped(double depth, double depth_max) {
    const double p = positive(depth);
    return (!(depth_max > 0.0) || !std::isfinite(depth_max)) ? p : std::min(p, depth_max);
}

double smoothstep(double e0, double e1, double x) {
    if (!(e1 > e0))
        return x >= e1 ? 1.0 : 0.0;
    const double t = std::min(1.0, std::max(0.0, (x - e0) / (e1 - e0)));
    return t * t * (3.0 - 2.0 * t);
}

double trilinear(const std::array<double, 8> &v, double tx, double ty, double tz) {
    tx = std::min(1.0, std::max(0.0, tx));
    ty = std::min(1.0, std::max(0.0, ty));
    tz = std::min(1.0, std::max(0.0, tz));
    const double c00 = v[0] * (1.0 - tx) + v[1] * tx, c10 = v[2] * (1.0 - tx) + v[3] * tx;
    const double c01 = v[4] * (1.0 - tx) + v[5] * tx, c11 = v[6] * (1.0 - tx) + v[7] * tx;
    const double c0 = c00 * (1.0 - ty) + c10 * ty, c1 = c01 * (1.0 - ty) + c11 * ty;
    return c0 * (1.0 - tz) + c1 * tz;
}

double bubble(double tx, double ty, double tz) {
    auto axis = [](double t) {
        t = std::min(1.0, std::max(0.0, t));
        return 4.0 * t * (1.0 - t);
    };
    return axis(tx) * axis(ty) * axis(tz);
}

template <typename Fn> void parallel(int64_t total, int threads, int64_t grain, Fn fn) {
    const int64_t chunks = std::max<int64_t>(1, (total + grain - 1) / grain);
    threads = (int)std::max<int64_t>(1, std::min<int64_t>(threads, chunks));
    std::atomic<int64_t> next(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t)
        workers.emplace_back([&, t] {
            for (;;) {
                const int64_t begin = next.fetch_add(grain);
                if (begin >= total)
                    break;
                fn(begin, std::min(total, begin + grain), t);
            }
        });
    for (std::thread &w : workers)
        w.join();
}

bool voxel_range(
    double lo, double hi, double origin, double spacing, int64_t dim, int64_t &o0, int64_t &o1) {
    int64_t i0 = (int64_t)std::floor((lo - origin) / spacing),
            i1 = (int64_t)std::ceil((hi - origin) / spacing);
    if (i1 < 0 || i0 >= dim)
        return false;
    i0 = std::max<int64_t>(0, std::min<int64_t>(dim - 1, i0));
    i1 = std::max<int64_t>(0, std::min<int64_t>(dim - 1, i1));
    if (i1 < i0)
        return false;
    o0 = i0;
    o1 = i1;
    return true;
}

struct Scan {
    const igl::WindingNumberAABB<double, int> &hier;
    const igl::AABB<MatrixXdR, 3> &tree;
    const Mesh &m;
    const double *origin, *spacing;
    int64_t Z, Y, X;
    bool inside;
    double depth_max;
    int threads;

    RowVector3d point(double x, double y, double z) const {
        return RowVector3d(
            origin[0] + x * spacing[0], origin[1] + y * spacing[1], origin[2] + z * spacing[2]);
    }
    bool active(const RowVector3d &q) const {
        return (hier.winding_number(q.transpose()) > 0.5) == inside;
    }
    double distance(const RowVector3d &q) const {
        int face = -1;
        RowVector3d closest;
        return std::sqrt(std::max(0.0, tree.squared_distance(m.V, m.F, q, face, closest)));
    }

    int64_t full(float *depth) const {
        std::atomic<int64_t> count(0);
        parallel(Z * Y * X, threads, 2048, [&](int64_t begin, int64_t end, int) {
            int64_t local = 0;
            for (int64_t n = begin; n < end; ++n) {
                const RowVector3d q =
                    point((double)(n % X), (double)((n / X) % Y), (double)(n / X / Y));
                const float d = (float)capped(active(q) ? distance(q) : 0.0, depth_max);
                depth[n] = d;
                local += d > 0.0f;
            }
            count += local;
        });
        return count;
    }

    int64_t chunked(
        float *depth, int chunk_size, double deep_interp_chunks, double deep_blend_chunks) const {
        const int64_t total = Z * Y * X, cs = std::max(1, chunk_size);
        const int64_t CX = (X + cs - 1) / cs, CY = (Y + cs - 1) / cs, CZ = (Z + cs - 1) / cs,
                      chunks = CX * CY * CZ;
        const double max_spacing = std::max({spacing[0], spacing[1], spacing[2]});
        const double deep = std::max(0.0, deep_interp_chunks) * (double)cs * max_spacing;
        const double blend = std::max(0.0, deep_blend_chunks) * (double)cs * max_spacing;
        const bool coarse = deep_interp_chunks > 0.0;
        std::fill(depth, depth + total, 0.0f);
        auto bounds = [&](int64_t ci) {
            const int64_t cx = ci % CX, cy = (ci / CX) % CY, cz = ci / CX / CY;
            return Chunk{cx * cs, std::min(X, cx * cs + cs), cy * cs, std::min(Y, cy * cs + cs),
                cz * cs, std::min(Z, cz * cs + cs)};
        };
        auto centre = [&](const Chunk &b) {
            return point(0.5 * (double)(b.x0 + b.x1), 0.5 * (double)(b.y0 + b.y1),
                0.5 * (double)(b.z0 + b.z1));
        };
        std::vector<uint8_t> centre_active((size_t)chunks, 0);
        parallel(chunks, threads, 64, [&](int64_t begin, int64_t end, int) {
            for (int64_t ci = begin; ci < end; ++ci)
                centre_active[(size_t)ci] = active(centre(bounds(ci)));
        });
        std::vector<uint8_t> seed((size_t)chunks, 0);
        for (Eigen::Index f = 0; f < m.F.rows(); ++f) {
            const int i0 = m.F(f, 0), i1 = m.F(f, 1), i2 = m.F(f, 2);
            int64_t r[3][2];
            bool inside_box = true;
            for (int a = 0; a < 3 && inside_box; ++a) {
                const double lo = std::min({m.V(i0, a), m.V(i1, a), m.V(i2, a)}),
                             hi = std::max({m.V(i0, a), m.V(i1, a), m.V(i2, a)});
                inside_box = voxel_range(lo, hi, origin[a], spacing[a],
                    a == 0       ? X
                        : a == 1 ? Y
                                 : Z,
                    r[a][0], r[a][1]);
            }
            if (!inside_box)
                continue;
            for (int64_t cz = r[2][0] / cs; cz <= r[2][1] / cs; ++cz)
                for (int64_t cy = r[1][0] / cs; cy <= r[1][1] / cs; ++cy)
                    for (int64_t cx = r[0][0] / cs; cx <= r[0][1] / cs; ++cx)
                        seed[(size_t)((cz * CY + cy) * CX + cx)] = 1;
        }
        std::vector<uint8_t> candidate((size_t)chunks, 0), processed((size_t)chunks, 0);
        auto dilate = [&](int64_t ci) {
            const int64_t cx = ci % CX, cy = (ci / CX) % CY, cz = ci / CX / CY;
            for (int64_t z = std::max<int64_t>(0, cz - 1); z <= std::min(CZ - 1, cz + 1); ++z)
                for (int64_t y = std::max<int64_t>(0, cy - 1); y <= std::min(CY - 1, cy + 1); ++y)
                    for (int64_t x = std::max<int64_t>(0, cx - 1); x <= std::min(CX - 1, cx + 1);
                        ++x)
                        candidate[(size_t)((z * CY + y) * CX + x)] = 1;
        };
        for (int64_t ci = 0; ci < chunks; ++ci)
            if (centre_active[(size_t)ci] || seed[(size_t)ci])
                dilate(ci);
        std::atomic<int64_t> count(0);
        std::vector<std::vector<int64_t>> grown((size_t)threads);
        for (;;) {
            std::vector<int64_t> scan;
            for (int64_t ci = 0; ci < chunks; ++ci)
                if (candidate[(size_t)ci] && !processed[(size_t)ci]) {
                    processed[(size_t)ci] = 1;
                    scan.push_back(ci);
                }
            if (scan.empty())
                break;
            for (auto &g : grown)
                g.clear();
            parallel((int64_t)scan.size(), threads, 1, [&](int64_t begin, int64_t end, int t) {
                int64_t local = 0;
                for (int64_t si = begin; si < end; ++si) {
                    const int64_t ci = scan[(size_t)si];
                    const Chunk b = bounds(ci);
                    int64_t chunk_active = 0;
                    auto write = [&](int64_t x, int64_t y, int64_t z, double d) {
                        const float f = (float)capped(d, depth_max);
                        depth[(z * Y + y) * X + x] = f;
                        chunk_active += f > 0.0f;
                    };
                    bool certified = false, blended = false, coarse_only = false;
                    std::array<double, 8> corners{};
                    double correction = 0.0;
                    if (centre_active[(size_t)ci]) {
                        const double cd = positive(distance(centre(b)));
                        const double hx = 0.5 * (double)(b.x1 - b.x0) * spacing[0],
                                     hy = 0.5 * (double)(b.y1 - b.y0) * spacing[1],
                                     hz = 0.5 * (double)(b.z1 - b.z0) * spacing[2];
                        const double half = std::sqrt(hx * hx + hy * hy + hz * hz);
                        if (cd > half) {
                            certified = true;
                            coarse_only = coarse && cd - half >= deep + blend;
                            blended = coarse && !coarse_only && cd + half > deep;
                            if (coarse_only || blended) {
                                for (int zb = 0; zb <= 1; ++zb)
                                    for (int yb = 0; yb <= 1; ++yb)
                                        for (int xb = 0; xb <= 1; ++xb)
                                            corners[(size_t)(xb + 2 * yb + 4 * zb)] =
                                                positive(distance(point((double)(xb ? b.x1 : b.x0),
                                                    (double)(yb ? b.y1 : b.y0),
                                                    (double)(zb ? b.z1 : b.z0))));
                                correction = cd - trilinear(corners, 0.5, 0.5, 0.5);
                            }
                        }
                    }
                    auto coarse_at = [&](int64_t x, int64_t y, int64_t z) {
                        const double tx = (double)(x - b.x0) / (double)(b.x1 - b.x0),
                                     ty = (double)(y - b.y0) / (double)(b.y1 - b.y0),
                                     tz = (double)(z - b.z0) / (double)(b.z1 - b.z0);
                        return positive(
                            trilinear(corners, tx, ty, tz) + bubble(tx, ty, tz) * correction);
                    };
                    for (int64_t z = b.z0; z < b.z1; ++z)
                        for (int64_t y = b.y0; y < b.y1; ++y)
                            for (int64_t x = b.x0; x < b.x1; ++x) {
                                const RowVector3d q = point((double)x, (double)y, (double)z);
                                if (certified && coarse_only)
                                    write(x, y, z, coarse_at(x, y, z));
                                else if (certified && blended) {
                                    const double exact = positive(distance(q));
                                    const double w = smoothstep(deep, deep + blend, exact);
                                    write(x, y, z, exact * (1.0 - w) + coarse_at(x, y, z) * w);
                                } else if (certified)
                                    write(x, y, z, distance(q));
                                else
                                    write(x, y, z, active(q) ? distance(q) : 0.0);
                            }
                    local += chunk_active;
                    if (chunk_active > 0)
                        grown[(size_t)t].push_back(ci);
                }
                count += local;
            });
            for (const auto &g : grown)
                for (int64_t ci : g)
                    dilate(ci);
        }
        return count;
    }
};

}

DepthVolume previous_shell_barrier(const Shell &shell, bool inside, double grid_step, int threads,
    int chunk_size, double deep_interp_chunks, double deep_blend_chunks) {
    const double step = std::max(1.0e-6, grid_step);
    double lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = INFINITY;
        hi[a] = -INFINITY;
    }
    for (size_t i = 0; i < shell.xyz.size(); i += 3)
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], (double)shell.xyz[i + (size_t)a]);
            hi[a] = std::max(hi[a], (double)shell.xyz[i + (size_t)a]);
        }
    double origin[3], spacing[3] = {step, step, step};
    int64_t dims[3];
    for (int a = 0; a < 3; ++a) {
        origin[a] = lo[a] - kBarrierDepthMax;
        dims[a] = std::max<int64_t>(
            1, (int64_t)std::ceil(((hi[a] + kBarrierDepthMax) - origin[a]) / step) + 1);
    }
    const Mesh m = capped_mesh(shell);
    igl::AABB<MatrixXdR, 3> tree;
    tree.init(m.V, m.F);
    igl::WindingNumberAABB<double, int> hier(m.V, m.F);
    hier.grow();
    if (threads <= 0)
        threads = (int)std::max(1u, std::thread::hardware_concurrency());
    const Scan scan{hier, tree, m, origin, spacing, dims[2], dims[1], dims[0], inside,
        kBarrierDepthMax, threads};
    const int64_t total = dims[0] * dims[1] * dims[2];
    std::vector<float> depth((size_t)total);
    int64_t active = chunk_size <= 0
        ? scan.full(depth.data())
        : scan.chunked(depth.data(), chunk_size, deep_interp_chunks, deep_blend_chunks);
    if (chunk_size > 0 && active == 0)
        scan.full(depth.data());
    DepthVolume out;
    out.Z = (int)dims[2];
    out.Y = (int)dims[1];
    out.X = (int)dims[0];
    out.v.resize((size_t)total);
    for (int64_t n = 0; n < total; ++n)
        out.v[(size_t)n] = encode_depth((double)depth[(size_t)n], kBarrierDepthMax);
    for (int a = 0; a < 3; ++a) {
        out.origin[a] = (float)origin[a];
        out.spacing[a] = (float)step;
    }
    return out;
}

}
