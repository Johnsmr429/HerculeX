#include "crossings.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace sp {

namespace {

struct Event {
    int32_t first, second, first_local, second_local;
};

struct Tangent {
    double z = 0, y = 0, x = 0;
    bool valid = false;
};

Tangent track_tangent(const std::vector<int32_t> &c, const std::vector<int64_t> &offsets,
    int32_t track, int32_t local, double radius) {
    const int64_t begin = offsets[(size_t)track], end = offsets[(size_t)track + 1],
                  center = begin + local;
    int64_t left = center, right = center;
    auto distance = [&](int64_t i) {
        const double dz = (double)c[(size_t)(3 * i)] - c[(size_t)(3 * center)];
        const double dy = (double)c[(size_t)(3 * i + 1)] - c[(size_t)(3 * center + 1)];
        const double dx = (double)c[(size_t)(3 * i + 2)] - c[(size_t)(3 * center + 2)];
        return std::sqrt(dz * dz + dy * dy + dx * dx);
    };
    while (left > begin && distance(left) < radius)
        --left;
    while (right + 1 < end && distance(right) < radius)
        ++right;
    if (left == right)
        return {};
    const double dz = (double)c[(size_t)(3 * right)] - c[(size_t)(3 * left)];
    const double dy = (double)c[(size_t)(3 * right + 1)] - c[(size_t)(3 * left + 1)];
    const double dx = (double)c[(size_t)(3 * right + 2)] - c[(size_t)(3 * left + 2)];
    const double norm = std::sqrt(dz * dz + dy * dy + dx * dx);
    if (norm == 0.0)
        return {};
    return {dz / norm, dy / norm, dx / norm, true};
}

}

CrossingCsr find_crossings(const Tracks &t) {
    const size_t tracks = t.size(), points = (size_t)t.offsets.back();
    std::vector<int32_t> c(points * 3);
    for (size_t i = 0; i < points * 3; ++i)
        c[i] = (int32_t)t.zyx[i];
    std::vector<uint64_t> key(points);
    for (size_t i = 0; i < points; ++i) {
        for (int a = 0; a < 3; ++a)
            if (c[3 * i + (size_t)a] < 0 || c[3 * i + (size_t)a] >= (1 << 20))
                throw std::runtime_error("track coordinates outside the 20-bit crossing key range");
        key[i] =
            ((uint64_t)c[3 * i] << 40) | ((uint64_t)c[3 * i + 1] << 20) | (uint64_t)c[3 * i + 2];
    }
    std::vector<uint32_t> order(points);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(
        order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    const double cutoff = std::cos(30.0 * std::acos(-1.0) / 180.0);
    std::vector<Event> events;
    std::vector<std::pair<int32_t, int32_t>> unique;
    std::vector<Tangent> tangents;
    for (size_t position = 0; position < points;) {
        size_t end = position + 1;
        const uint64_t k = key[order[position]];
        while (end < points && key[order[end]] == k)
            ++end;
        if (end - position < 2) {
            position = end;
            continue;
        }
        unique.clear();
        for (size_t item = position; item < end; ++item) {
            const int64_t flat = order[item];
            const int32_t track =
                (int32_t)(std::upper_bound(t.offsets.begin() + 1, t.offsets.end(), flat) -
                    (t.offsets.begin() + 1));
            if (t.families[(size_t)track] < 0)
                continue;
            unique.emplace_back(track, (int32_t)(flat - t.offsets[(size_t)track]));
        }
        std::sort(unique.begin(), unique.end());
        auto out = unique.begin();
        for (auto in = unique.begin(); in != unique.end();) {
            const int32_t track = in->first;
            int32_t local = in->second;
            do {
                local = std::min(local, in->second);
                ++in;
            } while (in != unique.end() && in->first == track);
            *out++ = {track, local};
        }
        unique.erase(out, unique.end());
        if (unique.size() < 2) {
            position = end;
            continue;
        }
        tangents.resize(unique.size());
        for (size_t i = 0; i < unique.size(); ++i)
            tangents[i] = track_tangent(c, t.offsets, unique[i].first, unique[i].second, 12.0);
        for (size_t a = 0; a < unique.size(); ++a) {
            if (!tangents[a].valid)
                continue;
            for (size_t b = a + 1; b < unique.size(); ++b) {
                if (t.families[(size_t)unique[a].first] == t.families[(size_t)unique[b].first] ||
                    !tangents[b].valid)
                    continue;
                const double dot = tangents[a].z * tangents[b].z + tangents[a].y * tangents[b].y +
                    tangents[a].x * tangents[b].x;
                if (std::abs(dot) > cutoff)
                    continue;
                events.push_back(
                    {unique[a].first, unique[b].first, unique[a].second, unique[b].second});
            }
        }
        position = end;
    }
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        return std::tie(a.first, a.second, a.first_local, a.second_local) <
            std::tie(b.first, b.second, b.first_local, b.second_local);
    });
    std::vector<double> arclength(points, 0.0);
    for (size_t track = 0; track < tracks; ++track)
        for (int64_t p = t.offsets[track] + 1; p < t.offsets[track + 1]; ++p) {
            const double dz = (double)c[(size_t)(3 * p)] - c[(size_t)(3 * (p - 1))];
            const double dy = (double)c[(size_t)(3 * p + 1)] - c[(size_t)(3 * (p - 1) + 1)];
            const double dx = (double)c[(size_t)(3 * p + 2)] - c[(size_t)(3 * (p - 1) + 2)];
            arclength[(size_t)p] =
                arclength[(size_t)p - 1] + std::sqrt(dz * dz + dy * dy + dx * dx);
        }
    CrossingCsr csr;
    std::vector<Event> edges;
    for (size_t begin = 0; begin < events.size();) {
        size_t end = begin + 1;
        while (end < events.size() && events[end].first == events[begin].first &&
            events[end].second == events[begin].second)
            ++end;
        Event best{};
        std::tuple<double, double, double, int32_t, int32_t> best_key;
        bool have = false;
        for (size_t cluster = begin; cluster < end;) {
            size_t cluster_end = cluster + 1;
            while (cluster_end < end &&
                std::abs(events[cluster_end].first_local - events[cluster_end - 1].first_local) <=
                    4 &&
                std::abs(events[cluster_end].second_local - events[cluster_end - 1].second_local) <=
                    4)
                ++cluster_end;
            const Event &candidate = events[cluster + (cluster_end - cluster) / 2];
            const double first_position =
                arclength[(size_t)(t.offsets[(size_t)candidate.first] + candidate.first_local)];
            const double second_position =
                arclength[(size_t)(t.offsets[(size_t)candidate.second] + candidate.second_local)];
            const double clearance = std::min({first_position,
                arclength[(size_t)t.offsets[(size_t)candidate.first + 1] - 1] - first_position,
                second_position,
                arclength[(size_t)t.offsets[(size_t)candidate.second + 1] - 1] - second_position});
            const auto k = std::make_tuple(clearance, first_position, second_position,
                candidate.first_local, candidate.second_local);
            if (!have || k > best_key) {
                have = true;
                best = candidate;
                best_key = k;
            }
            ++csr.accepted_events;
            cluster = cluster_end;
        }
        edges.push_back(best);
        begin = end;
    }
    csr.offsets.assign(tracks + 1, 0);
    for (const Event &e : edges) {
        ++csr.offsets[(size_t)e.first + 1];
        ++csr.offsets[(size_t)e.second + 1];
    }
    for (size_t i = 0; i < tracks; ++i)
        csr.offsets[i + 1] += csr.offsets[i];
    const size_t count = (size_t)csr.offsets.back();
    csr.partners.resize(count);
    csr.self_local.resize(count);
    csr.partner_local.resize(count);
    std::vector<int64_t> cursor(csr.offsets.begin(), csr.offsets.end() - 1);
    for (const Event &e : edges) {
        const size_t a = (size_t)cursor[(size_t)e.first]++;
        csr.partners[a] = e.second;
        csr.self_local[a] = e.first_local;
        csr.partner_local[a] = e.second_local;
        const size_t b = (size_t)cursor[(size_t)e.second]++;
        csr.partners[b] = e.first;
        csr.self_local[b] = e.second_local;
        csr.partner_local[b] = e.first_local;
    }
    for (size_t i = 0; i < tracks; ++i)
        csr.paired_tracks += csr.offsets[i + 1] != csr.offsets[i];
    return csr;
}

CrossingCsr sidecar_crossings(const Npz &sidecar, const Tracks &t) {
    const ArrayView &ids = sidecar.arrays.at("source_ids"), &off = sidecar.arrays.at("offsets"),
                    &par = sidecar.arrays.at("partners"), &self = sidecar.arrays.at("self_local"),
                    &other = sidecar.arrays.at("partner_local");
    const uint64_t *cache_ids = ids.as<uint64_t>();
    const size_t cached = ids.count(), count = t.size();
    std::vector<int32_t> rows(count), local_of(cached, -1);
    for (size_t i = 0; i < count; ++i) {
        const uint64_t *found = std::lower_bound(cache_ids, cache_ids + cached, t.source_ids[i]);
        if (found == cache_ids + cached || *found != t.source_ids[i])
            throw std::runtime_error("crossing cache lacks a selected track");
        rows[i] = (int32_t)(found - cache_ids);
        local_of[(size_t)rows[i]] = (int32_t)i;
    }
    CrossingCsr csr;
    csr.offsets.assign(count + 1, 0);
    for (size_t i = 0; i < count; ++i) {
        for (int64_t r = off.as<int64_t>()[rows[i]]; r < off.as<int64_t>()[rows[i] + 1]; ++r) {
            const int32_t global = par.as<int32_t>()[r];
            if (global < 0 || (size_t)global >= cached)
                continue;
            const int32_t partner = local_of[(size_t)global];
            if (partner < 0)
                continue;
            csr.partners.push_back(partner);
            csr.self_local.push_back(self.as<int32_t>()[r]);
            csr.partner_local.push_back(other.as<int32_t>()[r]);
        }
        csr.offsets[i + 1] = (int64_t)csr.partners.size();
        csr.paired_tracks += csr.offsets[i + 1] > csr.offsets[i];
    }
    return csr;
}

Tracks clip_tracks(
    const Tracks &t, const std::vector<float> &anchors, double radius, Clipping &clipping) {
    clipping.point_map.assign((size_t)t.offsets.back(), -1);
    clipping.track_map.assign(t.size(), -1);
    if (anchors.empty() || radius <= 0) {
        std::iota(clipping.point_map.begin(), clipping.point_map.end(), (int64_t)0);
        std::iota(clipping.track_map.begin(), clipping.track_map.end(), 0);
        return t;
    }
    const double cell = radius, limit = radius * radius;
    auto cell_of = [&](double v) { return (int64_t)std::floor(v / cell); };
    auto key_of = [](int64_t z, int64_t y, int64_t x) {
        return ((uint64_t)(z + (1 << 20)) << 42) | ((uint64_t)(y + (1 << 20)) << 21) |
            (uint64_t)(x + (1 << 20));
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
    for (size_t i = 0; i < anchors.size() / 3; ++i)
        grid[key_of(
                 cell_of(anchors[3 * i]), cell_of(anchors[3 * i + 1]), cell_of(anchors[3 * i + 2]))]
            .push_back((uint32_t)i);
    auto near = [&](const float *p) {
        const int64_t cz = cell_of(p[0]), cy = cell_of(p[1]), cx = cell_of(p[2]);
        for (int64_t z = cz - 1; z <= cz + 1; ++z)
            for (int64_t y = cy - 1; y <= cy + 1; ++y)
                for (int64_t x = cx - 1; x <= cx + 1; ++x) {
                    const auto found = grid.find(key_of(z, y, x));
                    if (found == grid.end())
                        continue;
                    for (uint32_t a : found->second) {
                        const double dz = (double)p[0] - (double)anchors[3 * (size_t)a],
                                     dy = (double)p[1] - (double)anchors[3 * (size_t)a + 1],
                                     dx = (double)p[2] - (double)anchors[3 * (size_t)a + 2];
                        if (dz * dz + dy * dy + dx * dx < limit)
                            return true;
                    }
                }
        return false;
    };
    Tracks out;
    std::vector<float> kept;
    std::vector<int64_t> kept_points;
    for (size_t i = 0; i < t.size(); ++i) {
        kept.clear();
        kept_points.clear();
        for (int64_t p = t.offsets[i]; p < t.offsets[i + 1]; ++p) {
            const float *q = t.zyx.data() + 3 * p;
            if (near(q))
                continue;
            kept.insert(kept.end(), q, q + 3);
            kept_points.push_back(p);
        }
        if (kept.size() < 6)
            continue;
        clipping.track_map[i] = (int32_t)out.size();
        for (size_t k = 0; k < kept_points.size(); ++k)
            clipping.point_map[(size_t)kept_points[k]] = (int64_t)(out.zyx.size() / 3 + k);
        out.zyx.insert(out.zyx.end(), kept.begin(), kept.end());
        out.offsets.push_back((int64_t)(out.zyx.size() / 3));
        out.source_ids.push_back(t.source_ids[i]);
        out.families.push_back(t.families[i]);
        out.arclengths.push_back(t.arclengths[i]);
        out.tortuosities.push_back(t.tortuosities[i]);
    }
    return out;
}

CrossingCsr clip_crossings(
    const CrossingCsr &csr, const Tracks &before, const Tracks &after, const Clipping &clipping) {
    CrossingCsr out;
    out.offsets.assign(after.size() + 1, 0);
    std::vector<int32_t> rows;
    for (size_t i = 0; i < before.size(); ++i) {
        const int32_t row = clipping.track_map[i];
        if (row < 0)
            continue;
        for (int64_t r = csr.offsets[i]; r < csr.offsets[i + 1]; ++r) {
            const int32_t partner_before = csr.partners[(size_t)r],
                          partner = clipping.track_map[(size_t)partner_before];
            if (partner < 0)
                continue;
            const int64_t self_point = before.offsets[i] + csr.self_local[(size_t)r];
            const int64_t partner_point =
                before.offsets[(size_t)partner_before] + csr.partner_local[(size_t)r];
            if (self_point >= before.offsets[i + 1] ||
                partner_point >= before.offsets[(size_t)partner_before + 1])
                continue;
            const int64_t self_after = clipping.point_map[(size_t)self_point],
                          partner_after = clipping.point_map[(size_t)partner_point];
            if (self_after < 0 || partner_after < 0)
                continue;
            rows.push_back(row);
            out.partners.push_back(partner);
            out.self_local.push_back((int32_t)(self_after - after.offsets[(size_t)row]));
            out.partner_local.push_back((int32_t)(partner_after - after.offsets[(size_t)partner]));
        }
    }
    for (int32_t row : rows)
        ++out.offsets[(size_t)row + 1];
    for (size_t i = 0; i < after.size(); ++i)
        out.offsets[i + 1] += out.offsets[i];
    for (size_t i = 0; i < after.size(); ++i)
        out.paired_tracks += out.offsets[i + 1] > out.offsets[i];
    return out;
}

}
