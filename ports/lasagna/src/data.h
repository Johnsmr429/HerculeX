#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace la {

struct Channel {
    std::string name, array;
    std::array<int64_t, 3> shape{}, chunks{}, grid{};
    std::array<float, 3> inv_scale{};
    std::vector<const uint8_t *> table;
    std::vector<std::unique_ptr<uint8_t[]>> blocks;
    std::vector<uint8_t> absent;
    std::unordered_map<int64_t, std::shared_ptr<const std::vector<uint8_t>>> recent;
    std::vector<int64_t> recent_order;
    size_t recent_next = 0;
    void load(const std::vector<int64_t> &cells);
    const uint8_t *block(int64_t cz, int64_t cy, int64_t cx) const;
};

struct Volume {
    std::array<float, 3> origin{};
    float grad_mag_scale = 1.0f;
    std::vector<Channel> channels;
    static Volume open(const std::string &manifest, const std::vector<std::string> &names);
    Channel &channel(const std::string &name);
    void prefetch(Channel &c, const float *xyz, size_t n);
};

float decode(const Volume &v, const std::string &name, float raw);
float decode_backward(const Volume &v, const std::string &name, float grad);
void sample_u8(const Volume &v, const Channel &c, const float *xyz, size_t n, uint8_t *out);
void sample_raw(const Volume &v, const Channel &c, const float *xyz, size_t n, float *out);
void sample_raw_backward(const Volume &v, const Channel &c, const float *xyz, size_t n,
    const float *grad_out, float *grad_xyz);

}
