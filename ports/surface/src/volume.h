#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hx {

using Box = std::array<std::array<int64_t, 2>, 3>;

std::string read_text(const std::string &path);
void write_text(const std::string &path, const std::string &text);
bool exists(const std::string &path);

struct Mapped {
    uint8_t *base = nullptr;
    size_t length = 0;
    size_t header = 0;
    std::vector<int64_t> shape;
    std::string descr;
    Mapped() = default;
    Mapped(const Mapped &) = delete;
    Mapped &operator=(const Mapped &) = delete;
    ~Mapped();
    uint8_t *data() const { return base + header; }
    void flush() const;
};

void map_npy(Mapped &m, const std::string &path, bool writable);
void create_npy(Mapped &m, const std::string &path, const std::string &descr,
    const std::vector<int64_t> &shape);
void save_npy(const std::string &path, const std::string &descr, const std::vector<int64_t> &shape,
    const void *data, size_t bytes);

struct BrickStore {
    std::array<int64_t, 3> shape{}, brick{}, grid{};
    std::vector<int64_t> offsets;
    std::string source;
    Mapped bricks_map;
    const uint8_t *bricks = nullptr;
    size_t bricks_length = 0;
    void open(const std::string &folder);
    void read(const Box &box, uint8_t *out) const;
};

}
