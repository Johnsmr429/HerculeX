#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace la {

std::string read_text(const std::string &path);
bool exists(const std::string &path);

struct Mapping {
    const uint8_t *data = nullptr;
    size_t size = 0;
    explicit Mapping(const std::string &path);
    Mapping(Mapping &&other) noexcept;
    Mapping &operator=(Mapping &&other) noexcept;
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    ~Mapping();
};

struct ArrayView {
    const uint8_t *data = nullptr;
    std::string descr;
    std::vector<int64_t> shape;
    size_t count() const;
};

ArrayView parse_npy(const uint8_t *bytes, size_t size);
std::vector<float> load_npy_float(const std::string &path, std::vector<int64_t> *shape = nullptr);
std::vector<uint8_t> load_npy_u8(const std::string &path, std::vector<int64_t> *shape = nullptr);
void save_npy(const std::string &path, const std::string &descr, const std::vector<int64_t> &shape,
    const void *data);
void blosc_decode(const uint8_t *src, size_t length, uint8_t *dst, size_t expected);

}
