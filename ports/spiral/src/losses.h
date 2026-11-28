#pragma once

#include "io.h"
#include "model.h"
#include "tracks.h"

#include <random>
#include <vector>

namespace sp {

struct LossSettings {
    int outer_winding = 130;
    int tracks = 1011, track_points = 96, max_crossings = 2;
    int normal_points = 1263, regularisation_points = 95, spacing_points = 2000;
    float weight_umbilicus = 1.25f, weight_sym = 10.0f, weight_normals = 100.0f,
          weight_spacing = 2.0f, weight_track = 50.0f, weight_patch = 8.0f;
    float normals_epsilon = 8.0f, sym_epsilon = 4.0f, min_spacing = 6.0f;
    float track_margin = 0.025f, track_p = 6.0f;
};

struct Samples {
    TrackSample tracks;
    std::vector<float> sym_points, sym_tangent;
    std::vector<float> normal_points;
    std::vector<int32_t> spacing_winding;
    std::vector<float> spacing_theta, spacing_z;
};

struct LossValues {
    double umbilicus = 0, sym = 0, normals = 0, spacing = 0, track = 0, patch = 0;
    double total(const LossSettings &w) const {
        return umbilicus * w.weight_umbilicus + sym * w.weight_sym + normals * w.weight_normals +
            spacing * w.weight_spacing + track * w.weight_track + patch * w.weight_patch;
    }
};

void draw_samples(const Model &m, const Shared &sh, const TrackPool &pool, const LossSettings &ls,
    std::mt19937_64 &rng, Samples &s);
double umbilicus_loss(
    const Model &m, const Shared &sh, const std::vector<float> &pts, float weight, Grads &g);
double sym_loss(
    const Model &m, const Shared &sh, const Samples &s, const LossSettings &ls, Grads &g);
double normals_loss(const Model &m, const Shared &sh, const Lasagna &L, const Samples &s,
    const LossSettings &ls, Grads &g);
double spacing_loss(
    const Model &m, const Shared &sh, const Samples &s, const LossSettings &ls, Grads &g);
double track_loss(
    const Model &m, const Shared &sh, const TrackSample &t, const LossSettings &ls, Grads &g);
LossValues evaluate(const Model &m, const Shared &sh, const Lasagna &lasagna,
    const std::vector<float> &umbilicus_points, const Samples &s, const LossSettings &ls, Grads &g);

}
