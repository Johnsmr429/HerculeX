#pragma once

#include <cstdint>
#include <vector>

namespace la {

struct Shell {
    int H = 0, W = 0;
    std::vector<float> xyz;
    const float *at(int h, int w) const {
        return xyz.data() + ((size_t)h * W + (size_t)((w % W + W) % W)) * 3;
    }
};

struct LossOut {
    double value = 0;
    std::vector<float> grad;
};

LossOut cyl_smooth(const Shell &s);
LossOut cyl_z_smooth(const Shell &s);
LossOut cyl_z_center(const Shell &s, double target, double height_step);
LossOut cyl_step(
    const Shell &s, const std::vector<uint8_t> &valid, double width_step, double height_step);
LossOut cyl_bend(const Shell &s);

struct Grid {
    int H = 0, W = 0, C = 0;
    std::vector<float> v;
    float *at(int h, int w) { return v.data() + ((size_t)h * W + (size_t)w) * C; }
    const float *at(int h, int w) const { return v.data() + ((size_t)h * W + (size_t)w) * C; }
};

Grid supersample(const Grid &field, int factor);
Grid supersample_backward(const Grid &grad, int H, int W, int factor);
Grid interp_width_at_offsets(const Grid &field, const Grid &offsets, float scale);
Shell shell_geometry(
    const Grid &base, const Grid &offsets, const Grid &delta, bool has_base, int factor);
std::vector<float> unit_normals(const Shell &s);
LossOut cyl_step_push(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<uint8_t> &valid, double width_step, double mesh_step, int factor = 4);
void shell_normal_targets(const Shell &xyz, const std::vector<float> &nx,
    const std::vector<float> &ny, const std::vector<uint8_t> &valid, std::vector<float> &target,
    std::vector<float> &mask);
LossOut cyl_normal(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<float> &target, const std::vector<float> &mask, int factor = 4);

struct DepthVolume {
    int Z = 0, Y = 0, X = 0;
    std::vector<uint8_t> v;
    float origin[3] = {0, 0, 0}, spacing[3] = {1, 1, 1};
};

LossOut cyl_outside(const Grid &base, const Grid &offsets, const Grid &delta, bool has_base,
    const std::vector<uint8_t> &valid, const DepthVolume &field, double depth_max,
    double model_step, int factor);

}
