#include "tracks.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sp {

static double quantile(std::vector<double> sorted, double q) {
    std::sort(sorted.begin(), sorted.end());
    const double at = q * (double)(sorted.size() - 1);
    const size_t lo = (size_t)std::floor(at), hi = (size_t)std::ceil(at);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - (double)lo);
}

static std::vector<double> cumulative_arclengths(const float *p, int64_t begin, int64_t end) {
    std::vector<double> c((size_t)(end - begin), 0.0);
    for (int64_t i = begin + 1; i < end; ++i) {
        const double dz = (double)p[3 * i] - p[3 * (i - 1)],
                     dy = (double)p[3 * i + 1] - p[3 * (i - 1) + 1],
                     dx = (double)p[3 * i + 2] - p[3 * (i - 1) + 2];
        c[(size_t)(i - begin)] =
            c[(size_t)(i - begin - 1)] + std::sqrt(dz * dz + dy * dy + dx * dx);
    }
    return c;
}

TrackPool prepare_tracks(const Tracks &t, const CrossingCsr *crossings, double min_spacing,
    double max_spacing, const std::array<double, 3> &w) {
    TrackPool pool;
    pool.count = t.size();
    pool.arclengths = t.arclengths;
    pool.length_edges = {quantile(t.arclengths, 1.0 / 3.0), quantile(t.arclengths, 2.0 / 3.0)};
    std::vector<int> bins(pool.count);
    for (size_t i = 0; i < pool.count; ++i) {
        bins[i] =
            (t.arclengths[i] > pool.length_edges[0]) + (t.arclengths[i] > pool.length_edges[1]);
        ++pool.bin_counts[(size_t)bins[i]];
    }
    double weights[3], total = 0;
    for (int b = 0; b < 3; ++b)
        total += (weights[b] = pool.bin_counts[(size_t)b] > 0 ? w[(size_t)b] : 0.0);
    if (total <= 0)
        throw std::runtime_error("track length bin weights leave no non-empty bin");
    pool.cumulative_probability.resize(pool.count);
    double running = 0;
    for (size_t i = 0; i < pool.count; ++i) {
        running +=
            (double)(float)(weights[bins[i]] / total / (double)pool.bin_counts[(size_t)bins[i]]);
        pool.cumulative_probability[i] = running;
    }
    pool.cross_offsets.assign(pool.count + 1, 0);
    if (crossings) {
        if (crossings->offsets.size() != pool.count + 1)
            throw std::runtime_error("crossings do not match the tracks");
        pool.cross_offsets = crossings->offsets;
        pool.cross_partners = crossings->partners;
        pool.cross_self = crossings->self_local;
        pool.cross_partner = crossings->partner_local;
        for (size_t i = 0; i < pool.count; ++i)
            pool.connected += pool.cross_offsets[i + 1] > pool.cross_offsets[i];
        pool.directed = pool.cross_partners.size();
    }
    std::vector<int64_t> anchor_offsets(pool.count + 1, 0);
    std::vector<int32_t> anchors(pool.cross_self);
    for (size_t i = 0; i < pool.count; ++i) {
        anchor_offsets[i + 1] = pool.cross_offsets[i + 1];
        std::sort(anchors.begin() + anchor_offsets[i], anchors.begin() + anchor_offsets[i + 1]);
    }
    std::vector<int32_t> anchor_samples(anchors.size(), -1);
    pool.offsets.assign(pool.count + 1, 0);
    pool.lengths.assign(pool.count, 0);
    std::vector<std::vector<float>> parts(pool.count);
    for (size_t i = 0; i < pool.count; ++i) {
        const int64_t begin = t.offsets[i], end = t.offsets[i + 1];
        const auto cumulative = cumulative_arclengths(t.zyx.data(), begin, end);
        std::vector<float> &out = parts[i];
        if (cumulative.back() <= 0.0) {
            out.assign(t.zyx.begin() + 3 * begin, t.zyx.begin() + 3 * begin + 3);
            pool.lengths[i] = 1;
            continue;
        }
        std::vector<int32_t> local{0};
        for (int64_t a = anchor_offsets[i]; a < anchor_offsets[i + 1]; ++a)
            if (anchors[(size_t)a] >= 0 && anchors[(size_t)a] < end - begin)
                local.push_back(anchors[(size_t)a]);
        local.push_back((int32_t)(end - begin - 1));
        std::sort(local.begin(), local.end());
        local.erase(std::unique(local.begin(), local.end()), local.end());
        std::vector<double> marks;
        for (int32_t l : local)
            if (marks.empty() || cumulative[(size_t)l] != marks.back())
                marks.push_back(cumulative[(size_t)l]);
        std::vector<double> positions;
        for (size_t s = 1; s < marks.size(); ++s) {
            const double left = marks[s - 1], span = marks[s] - left;
            const int64_t intervals = std::max<int64_t>(1, (int64_t)std::ceil(span / max_spacing));
            const bool feasible = intervals <= (int64_t)std::floor(span / min_spacing);
            const double step = feasible ? span / (double)intervals : max_spacing;
            for (int64_t k = 0; k < intervals; ++k)
                positions.push_back(left + (double)k * step);
        }
        positions.push_back(marks.back());
        for (size_t o = 0; o < positions.size(); ++o) {
            const double position = positions[o];
            size_t right =
                (size_t)(std::upper_bound(cumulative.begin(), cumulative.end(), position) -
                    cumulative.begin());
            right = std::clamp<size_t>(right, 1, cumulative.size() - 1);
            const size_t left = right - 1;
            const double den = cumulative[right] - cumulative[left];
            const double alpha = den > 0.0 ? (position - cumulative[left]) / den : 0.0;
            for (int c = 0; c < 3; ++c)
                out.push_back((float)(t.zyx[3 * (begin + (int64_t)left) + c] * (1.0 - alpha) +
                    t.zyx[3 * (begin + (int64_t)right) + c] * alpha));
            if (o > 0 && position - positions[o - 1] < min_spacing - 1e-9)
                ++pool.undersized;
        }
        for (int64_t a = anchor_offsets[i]; a < anchor_offsets[i + 1]; ++a) {
            const int32_t l = anchors[(size_t)a];
            if (l < 0 || (size_t)l >= cumulative.size())
                continue;
            const double position = cumulative[(size_t)l];
            size_t sample =
                (size_t)(std::lower_bound(positions.begin(), positions.end(), position) -
                    positions.begin());
            if (sample == positions.size())
                sample = positions.size() - 1;
            else if (sample > 0 &&
                std::fabs(positions[sample - 1] - position) <=
                    std::fabs(positions[sample] - position))
                --sample;
            anchor_samples[(size_t)a] = (int32_t)sample;
        }
        pool.lengths[i] = (int64_t)positions.size();
    }
    for (size_t i = 0; i < pool.count; ++i)
        pool.offsets[i + 1] = pool.offsets[i] + pool.lengths[i];
    pool.zyx.reserve((size_t)pool.offsets.back() * 3);
    for (auto &p : parts)
        pool.zyx.insert(pool.zyx.end(), p.begin(), p.end());
    pool.record_sample.assign(pool.directed, -1);
    for (size_t i = 0; i < pool.count; ++i)
        for (int64_t r = pool.cross_offsets[i]; r < pool.cross_offsets[i + 1]; ++r) {
            const int32_t l = pool.cross_self[(size_t)r];
            auto first = anchors.begin() + anchor_offsets[i],
                 last = anchors.begin() + anchor_offsets[i + 1];
            auto found = std::lower_bound(first, last, l);
            if (found != last && *found == l)
                pool.record_sample[(size_t)r] = anchor_samples[(size_t)(found - anchors.begin())];
        }
    return pool;
}

TrackSample draw_tracks(
    const TrackPool &pool, int k, int target_points, int max_crossings, std::mt19937_64 &rng) {
    TrackSample s;
    s.num_groups = k;
    s.group_width = max_crossings + 1;
    s.target_points = target_points;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double total = pool.cumulative_probability.back();
    std::vector<int32_t> track;
    for (int i = 0; i < k; ++i) {
        const double u = unit(rng) * total;
        size_t at = (size_t)(std::upper_bound(pool.cumulative_probability.begin(),
                                 pool.cumulative_probability.end(), u) -
            pool.cumulative_probability.begin());
        track.push_back((int32_t)std::min(at, pool.count - 1));
    }
    for (int i = 0; i < k; ++i) {
        s.group_id.push_back(i);
        s.row_slot.push_back(0);
    }
    std::vector<int32_t> self_samples, partner_samples, partner_groups;
    if (max_crossings > 0 && pool.directed > 0) {
        const uint64_t seed = rng() & ((1ull << 63) - 1);
        for (int row = 0; row < k; ++row) {
            const int32_t primary = track[(size_t)row];
            const int64_t begin = pool.cross_offsets[(size_t)primary],
                          end = pool.cross_offsets[(size_t)primary + 1];
            const size_t degree = (size_t)(end - begin),
                         count = std::min((size_t)max_crossings, degree);
            if (!count)
                continue;
            std::vector<int32_t> candidates(degree);
            std::iota(candidates.begin(), candidates.end(), (int32_t)begin);
            std::mt19937_64 random(seed + 0x9e3779b97f4a7c15ULL * ((uint64_t)row + 1));
            for (size_t slot = 0; slot < count; ++slot) {
                std::uniform_int_distribution<size_t> draw(slot, degree - 1);
                std::swap(candidates[slot], candidates[draw(random)]);
                const int32_t record = candidates[slot];
                const int32_t partner = pool.cross_partners[(size_t)record];
                int32_t reciprocal = -1;
                for (int64_t c = pool.cross_offsets[(size_t)partner];
                    c < pool.cross_offsets[(size_t)partner + 1]; ++c)
                    if (pool.cross_partners[(size_t)c] == primary &&
                        pool.cross_self[(size_t)c] == pool.cross_partner[(size_t)record] &&
                        pool.cross_partner[(size_t)c] == pool.cross_self[(size_t)record]) {
                        reciprocal = (int32_t)c;
                        break;
                    }
                if (reciprocal < 0)
                    throw std::runtime_error("crossing index requires reciprocal directed records");
                track.push_back(partner);
                s.group_id.push_back(row);
                s.row_slot.push_back((int32_t)slot + 1);
                partner_groups.push_back(row);
                self_samples.push_back(pool.record_sample[(size_t)record]);
                partner_samples.push_back(pool.record_sample[(size_t)reciprocal]);
            }
        }
    }
    const size_t rows = track.size();
    s.row_starts.assign(rows + 1, 0);
    for (size_t r = 0; r < rows; ++r)
        s.row_starts[r + 1] = s.row_starts[r] + pool.lengths[(size_t)track[r]];
    s.points.resize((size_t)s.row_starts.back() * 3);
    s.row_id.resize((size_t)s.row_starts.back());
    for (size_t r = 0; r < rows; ++r) {
        const int64_t source = pool.offsets[(size_t)track[r]];
        for (int64_t j = 0; j < pool.lengths[(size_t)track[r]]; ++j) {
            const size_t at = (size_t)(s.row_starts[r] + j);
            s.row_id[at] = (int32_t)r;
            for (int c = 0; c < 3; ++c)
                s.points[at * 3 + c] = pool.zyx[(size_t)(source + j) * 3 + c];
        }
    }
    const float step = target_points > 1 ? 1.0f / (float)(target_points - 1) : 0.0f;
    s.target_flat.resize(rows * (size_t)target_points);
    for (size_t r = 0; r < rows; ++r) {
        const float span = (float)(pool.lengths[(size_t)track[r]] - 1);
        for (int i = 0; i < target_points; ++i) {
            float fraction = 0.0f;
            if (target_points > 1)
                fraction = i < target_points / 2 ? (float)i * step
                                                 : 1.0f - (float)(target_points - 1 - i) * step;
            const int64_t local = target_points == 1 ? 0 : (int64_t)std::nearbyint(fraction * span);
            s.target_flat[r * (size_t)target_points + (size_t)i] = s.row_starts[r] + local;
        }
    }
    for (size_t p = 0; p < partner_groups.size(); ++p) {
        const int32_t row = (int32_t)((size_t)k + p);
        s.partner_rows.push_back(row);
        s.primary_cross.push_back(s.row_starts[(size_t)partner_groups[p]] + self_samples[p]);
        s.partner_cross.push_back(s.row_starts[(size_t)row] + partner_samples[p]);
    }
    return s;
}

}
