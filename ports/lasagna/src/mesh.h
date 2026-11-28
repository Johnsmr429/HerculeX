#pragma once

#include "data.h"
#include "json.h"
#include "pyramid.h"
#include "shell.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace la {

struct Mesh {
    int D = 0, H = 0, W = 0;
    std::vector<float> xyz;
    float *at(int d, int h, int w) { return xyz.data() + (((size_t)d * H + h) * W + w) * 3; }
    const float *at(int d, int h, int w) const {
        return xyz.data() + (((size_t)d * H + h) * W + w) * 3;
    }
    size_t points() const { return (size_t)D * H * W; }
};

LossOut mesh_normal(
    const Mesh &m, const std::vector<float> &target, const std::vector<float> &mask);
LossOut mesh_step(const Mesh &m, double step);
LossOut mesh_smooth(const Mesh &m, double step);
LossOut mesh_bend(const Mesh &m);

struct Station {
    std::array<float, 3> seed{};
    std::vector<double> h_frac, w_frac;
    std::vector<uint8_t> initialized, has_ray;
    std::vector<std::array<float, 3>> ray;
    void reset(std::array<float, 3> seed, int H, int W, int D);
};

void station_loss(
    const Mesh &m, Station &s, double mesh_step, int anchor_d, LossOut &normal, LossOut &tangent);
std::array<float, 3> grid_centre(const Mesh &m);
std::vector<float> normal_targets(const std::vector<float> &nx, const std::vector<float> &ny);

struct MeshWeights {
    double normal = 0, step = 0, smooth = 0, station_n = 0, station_t = 0, bend = 0;
};

struct MeshStageReport {
    std::string name;
    int steps = 0;
    double first_loss = 0, last_loss = 0;
};

struct MeshFit {
    Volume volume;
    Json config;
    int D = 0, H = 0, W = 0;
    std::vector<Grid> levels;
    double mesh_step = 100;
    std::vector<int> depth_windings;
    std::array<float, 3> seed{};
    Station station;
    std::vector<MeshStageReport> stages;
    std::FILE *log = nullptr;

    MeshFit(const std::string &manifest, const Json &config, const std::vector<float> &flat, int D,
        int H, int W, double mesh_step, std::vector<int> depth_windings);
    Mesh current() const;
    std::vector<float> flat() const;
    double evaluate(const MeshWeights &w, std::vector<float> &grad);
    void run();
};

void write_tifxyz(const std::string &dir, const Mesh &m, int d, double winding, double mesh_step,
    const std::string &meta_extra);
void write_tiff_f32(const std::string &path, const float *data, int H, int W);

}
