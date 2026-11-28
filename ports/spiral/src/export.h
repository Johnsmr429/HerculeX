#pragma once

#include "model.h"

#include <string>

namespace sp {

struct ExportSettings {
    int first_winding = 10, outer_winding = 130;
    float step = 20.0f;
    double voxel_um = 0.0;
};

int export_windings(
    const Model &m, const Shared &sh, const std::string &dir, const ExportSettings &e);

}
