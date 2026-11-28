#pragma once

#include "net.h"
#include "volume.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace hx {

using Reader = std::function<void(const Box &box, uint8_t *out)>;
using Writer = std::function<void(int64_t z0, int64_t rows, const std::vector<float> &logits)>;

std::vector<std::vector<int64_t>> window_steps(
    const std::array<int64_t, 3> &size, int patch, double step, Rule rule);
std::vector<float> gaussian_importance(int patch, Rule rule);
void predict_volume(SurfaceNet &net, const Reader &read, const std::array<int64_t, 3> &shape,
    double step, bool mirror, const Writer &write, bool verbose);
mx::array mirrored_forward(SurfaceNet &net, const mx::array &x);
uint8_t surface_probability(float background, float surface);

}
