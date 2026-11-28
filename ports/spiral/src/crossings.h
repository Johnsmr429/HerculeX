#pragma once

#include "io.h"

#include <cstdint>
#include <vector>

namespace sp {

struct CrossingCsr {
    std::vector<int64_t> offsets;
    std::vector<int32_t> partners, self_local, partner_local;
    uint64_t accepted_events = 0, paired_tracks = 0;
};

struct Clipping {
    std::vector<int64_t> point_map;
    std::vector<int32_t> track_map;
};

CrossingCsr find_crossings(const Tracks &t);
CrossingCsr sidecar_crossings(const Npz &sidecar, const Tracks &t);
Tracks clip_tracks(
    const Tracks &t, const std::vector<float> &anchors, double radius, Clipping &clipping);
CrossingCsr clip_crossings(
    const CrossingCsr &csr, const Tracks &before, const Tracks &after, const Clipping &clipping);

}
