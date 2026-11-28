#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sp {

std::string read_text(const std::string &path);

struct Mapping {
    const uint8_t *data = nullptr;
    size_t size = 0;
    Mapping() = default;
    explicit Mapping(const std::string &path);
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    Mapping(Mapping &&other) noexcept;
    Mapping &operator=(Mapping &&other) noexcept;
    ~Mapping();
};

struct ArrayView {
    const uint8_t *data = nullptr;
    std::string descr;
    std::vector<int64_t> shape;
    size_t count() const;
    template <class T> const T *as() const { return reinterpret_cast<const T *>(data); }
};

ArrayView parse_npy(const uint8_t *bytes, size_t size);

struct Npz {
    Mapping file;
    std::map<std::string, ArrayView> arrays;
    explicit Npz(const std::string &path);
};

struct Tracks {
    std::vector<float> zyx;
    std::vector<int64_t> offsets{0};
    std::vector<uint64_t> source_ids;
    std::vector<int8_t> families;
    std::vector<double> arclengths, tortuosities;
    size_t size() const { return source_ids.size(); }
};

Tracks load_tracks(const std::string &store, int64_t z_lo, int64_t z_hi);

struct Lasagna {
    int64_t z_origin = 0, scale = 4;
    std::array<int64_t, 3> shape{};
    std::vector<uint8_t> nx, ny, grad_mag;
    size_t at(int64_t z, int64_t y, int64_t x) const {
        return ((size_t)z * shape[1] + y) * shape[2] + x;
    }
};

Lasagna load_lasagna(const std::string &nx, const std::string &ny, const std::string &grad_mag,
    const std::string &group, int64_t scale, int64_t z_begin, int64_t z_end);

std::vector<std::array<float, 3>> load_umbilicus(const std::string &path);

struct Image {
    int H = 0, W = 0, C = 0;
    std::vector<float> data;
};

Image read_tiff(const std::string &path);
bool exists(const std::string &path);

void save_npy(const std::string &path, const std::string &descr, const std::vector<int64_t> &shape,
    const void *data);
std::vector<float> load_npy_float(const std::string &path, std::vector<int64_t> *shape = nullptr);
std::vector<int64_t> load_npy_int(const std::string &path, std::vector<int64_t> *shape = nullptr);

}
