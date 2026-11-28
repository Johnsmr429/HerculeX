#pragma once

#include "cylinder.h"
#include "data.h"
#include "json.h"
#include "pyramid.h"
#include "shell.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace la {

struct CylinderWeights {
    double normal = 0, smooth = 0, z_smooth = 0, z_center = 0, step = 0, step_push = 0, bend = 0,
           outside = 0;
};

struct PassReport {
    std::string label;
    int steps = 0, width = 0;
    double first_loss = 0, last_loss = 0, invalid_frac = 0;
};

struct CylinderFit {
    Volume volume;
    Umbilicus umbilicus;
    Json config;
    std::array<double, 3> seed{};
    std::array<double, 3> extent{};
    double model_h = 0, model_step = 0, z_target = 0, mesh_step = 100;
    int index = 0;
    Grid base, offsets;
    std::vector<Grid> levels;
    double width_step = 0, width_target_step = 0;
    std::vector<Shell> completed;
    DepthVolume field;
    bool has_field = false;
    double outside_model_step = 0;
    int outside_factor = 2;
    std::vector<PassReport> passes;
    std::FILE *log = nullptr;

    CylinderFit(const std::string &manifest, const Json &config, std::array<double, 3> seed,
        double model_h);
    void run();
    Shell current() const;
    std::vector<float> mesh_flat(int &shells, int &H, int &W) const;
    void begin(const Shell &from, int idx);
    void resample_width(int target_w, double target_step);
    double evaluate(const CylinderWeights &w, Grid &grad, std::vector<uint8_t> &valid);

  private:
    void start_tube();
    PassReport pass(const std::string &label, const CylinderWeights &w, const Json &stage,
        double wstep_start, double wstep_end, int steps);
};

}
