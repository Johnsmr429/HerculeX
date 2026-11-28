#pragma once

#include "model.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sp {

struct PatchSettings {
    int erode = 1, per_step = 360, points = 800;
    double z_margin = 0, area_exponent = 0.5, exclusion_radius = 16.0;
    float weight = 8.0f, margin = 0.025f, p = 3.0f;
    long refresh_interval = 100, dt_start = 25000;
};

struct Patch {
    std::string name;
    int H = 0, W = 0;
    std::vector<float> zyx;
    std::vector<uint8_t> vertex_valid, quad_valid, sampling;
    double scale[2] = {1, 1}, area = 0;
    std::vector<int64_t> cells;
    bool rectangular = false;
    int64_t row_lo = 0, col_lo = 0, rect_w = 0;
    std::vector<int64_t> preorder, parent;
};

struct PatchSet {
    std::vector<Patch> patches;
    std::vector<double> probability;
    std::vector<int64_t> start;
    int64_t nodes = 0;
    std::vector<float> theta;
    std::vector<int32_t> potential;
    std::vector<float> trusted;
    std::vector<std::string> rejected;
    long refreshed = -1;
    bool fresh = false;
    bool empty() const { return patches.empty(); }
};

PatchSet load_patches(
    const std::string &dir, int64_t z_begin, int64_t z_end, const PatchSettings &ps);
void reindex_patches(PatchSet &set, const PatchSettings &ps);
void refresh_theta(PatchSet &set, const Model &m, const Shared &sh);
std::vector<size_t> non_liftable_patches(const PatchSet &set);
size_t enforce_liftability(
    PatchSet &set, const Model &m, const Shared &sh, const PatchSettings &ps);
bool refresh_if_due(
    PatchSet &set, long iteration, const Model &m, const Shared &sh, const PatchSettings &ps);

struct PatchSample {
    int rows = 0, cap = 0;
    std::vector<int32_t> counts;
    std::vector<float> points;
    std::vector<int64_t> nodes;
};

PatchSample draw_patches(const PatchSet &set, int rows, int cap, std::mt19937_64 &rng);
double patch_loss(const Model &m, const Shared &sh, const PatchSet &set, const PatchSample &s,
    const PatchSettings &ps, float weight, Grads &g);

}
