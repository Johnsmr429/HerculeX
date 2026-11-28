#pragma once

#include "crossings.h"
#include "io.h"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace sp {

struct TrackPool {
    size_t count = 0;
    std::vector<double> arclengths;
    std::vector<double> cumulative_probability;
    std::array<double, 2> length_edges{};
    std::array<int64_t, 3> bin_counts{};
    std::vector<int64_t> cross_offsets;
    std::vector<int32_t> cross_partners, cross_self, cross_partner;
    std::vector<float> zyx;
    std::vector<int64_t> offsets, lengths;
    std::vector<int32_t> record_sample;
    size_t connected = 0, directed = 0, undersized = 0;
};

TrackPool prepare_tracks(const Tracks &tracks, const CrossingCsr *crossings, double min_spacing,
    double max_spacing, const std::array<double, 3> &bin_weights);

struct TrackSample {
    std::vector<float> points;
    std::vector<int32_t> row_id;
    std::vector<int64_t> row_starts;
    std::vector<int32_t> group_id, row_slot;
    int num_groups = 0, group_width = 1, target_points = 0;
    std::vector<int64_t> target_flat;
    std::vector<int64_t> primary_cross, partner_cross;
    std::vector<int32_t> partner_rows;
    size_t rows() const { return group_id.size(); }
};

TrackSample draw_tracks(
    const TrackPool &pool, int k, int target_points, int max_crossings, std::mt19937_64 &rng);

}
