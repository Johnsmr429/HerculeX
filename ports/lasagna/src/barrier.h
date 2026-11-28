#pragma once

#include "shell.h"

namespace la {

constexpr double kBarrierDepthMax = 2000.0;

DepthVolume previous_shell_barrier(const Shell &shell, bool inside, double grid_step = 64.0,
    int threads = 0, int chunk_size = 8, double deep_interp_chunks = 10.0,
    double deep_blend_chunks = 2.0);

}
