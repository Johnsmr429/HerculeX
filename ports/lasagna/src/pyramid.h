#pragma once

#include "shell.h"

#include <vector>

namespace la {

Grid resize_bilinear(const Grid &in, int out_h, int out_w);
Grid resize_bilinear_backward(const Grid &grad, int in_h, int in_w);
std::vector<Grid> build_pyramid(const Grid &delta, int scales);
Grid integrate_pyramid(const std::vector<Grid> &levels);
std::vector<Grid> integrate_pyramid_backward(const std::vector<Grid> &levels, const Grid &grad);

struct Adam {
    double beta1 = 0.9, beta2 = 0.999, eps = 1.0e-8;
    long step = 0;
    std::vector<std::vector<float>> first, second;
    void update(
        std::vector<Grid> &params, const std::vector<Grid> &grads, const std::vector<double> &lr);
};

}
