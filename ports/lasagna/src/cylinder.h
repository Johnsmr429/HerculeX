#pragma once

#include "shell.h"

#include <string>
#include <vector>

namespace la {

struct Umbilicus {
    std::vector<float> table;
    double z0 = 0, step = 1000;
    void xy(float z, float &x, float &y) const;
};

Umbilicus load_umbilicus(const std::string &path, double z_step = 1000.0);
std::vector<float> linspace(float start, float end, int steps);
int shell_width_for_radius(double radius, double width_step);

struct TubeStart {
    Grid base, dirs, delta;
    std::vector<float> z;
    double height_step = 0;
};

TubeStart tube_start(const Umbilicus &u, double z_center, double model_h, double z_step,
    double radius, double width_step);

}
