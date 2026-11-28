#include "volume.h"

#include "json.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hx {

std::string read_text(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

void write_text(const std::string &path, const std::string &text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write " + tmp);
        out << text;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot replace " + path);
}

bool exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

Mapped::~Mapped() {
    if (base)
        munmap(base, length);
}

void Mapped::flush() const {
    if (base)
        msync(base, length, MS_SYNC);
}

static void parse_header(Mapped &m) {
    if (m.length < 10 || std::memcmp(m.base, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("not an npy file");
    const int major = m.base[6];
    size_t len = 0, start = 0;
    if (major == 1) {
        len = m.base[8] | (m.base[9] << 8);
        start = 10;
    } else {
        len =
            m.base[8] | (m.base[9] << 8) | ((size_t)m.base[10] << 16) | ((size_t)m.base[11] << 24);
        start = 12;
    }
    std::string h((const char *)m.base + start, len);
    m.header = start + len;
    auto field = [&](const std::string &key) {
        size_t at = h.find("'" + key + "'");
        if (at == std::string::npos)
            throw std::runtime_error("npy header lacks " + key);
        return h.substr(h.find(':', at) + 1);
    };
    std::string d = field("descr");
    size_t q0 = d.find('\''), q1 = d.find('\'', q0 + 1);
    m.descr = d.substr(q0 + 1, q1 - q0 - 1);
    if (field("fortran_order").find("True") < field("fortran_order").find(','))
        throw std::runtime_error("fortran npy");
    std::string s = field("shape");
    s = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
    m.shape.clear();
    std::stringstream parts(s);
    std::string item;
    while (std::getline(parts, item, ','))
        if (item.find_first_not_of(' ') != std::string::npos)
            m.shape.push_back(std::stoll(item));
}

static std::string header_bytes(const std::string &descr, const std::vector<int64_t> &shape) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i)
        dims += (i ? ", " : "") + std::to_string(shape[i]);
    if (shape.size() == 1)
        dims += ",";
    std::string dict =
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    size_t total = 10 + dict.size() + 1;
    dict += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::string out = "\x93NUMPY";
    out += (char)1;
    out += (char)0;
    out += (char)(dict.size() & 0xFF);
    out += (char)((dict.size() >> 8) & 0xFF);
    return out + dict;
}

static size_t item_size(const std::string &descr) { return (size_t)std::stoi(descr.substr(2)); }

static size_t count(const std::vector<int64_t> &shape) {
    size_t n = 1;
    for (auto s : shape)
        n *= (size_t)s;
    return n;
}

void map_npy(Mapped &m, const std::string &path, bool writable) {
    int fd = ::open(path.c_str(), writable ? O_RDWR : O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    m.length = (size_t)st.st_size;
    void *p =
        mmap(nullptr, m.length, writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED)
        throw std::runtime_error("cannot map " + path);
    m.base = (uint8_t *)p;
    parse_header(m);
}

void create_npy(Mapped &m, const std::string &path, const std::string &descr,
    const std::vector<int64_t> &shape) {
    std::string h = header_bytes(descr, shape);
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("cannot create " + path);
    m.length = h.size() + count(shape) * item_size(descr);
    if (ftruncate(fd, (off_t)m.length) != 0 ||
        pwrite(fd, h.data(), h.size(), 0) != (ssize_t)h.size()) {
        ::close(fd);
        throw std::runtime_error("cannot size " + path);
    }
    void *p = mmap(nullptr, m.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED)
        throw std::runtime_error("cannot map " + path);
    m.base = (uint8_t *)p;
    m.header = h.size();
    m.shape = shape;
    m.descr = descr;
}

void save_npy(const std::string &path, const std::string &descr, const std::vector<int64_t> &shape,
    const void *data, size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    std::string h = header_bytes(descr, shape);
    out.write(h.data(), (std::streamsize)h.size());
    out.write((const char *)data, (std::streamsize)bytes);
}

void BrickStore::open(const std::string &folder) {
    Json meta = Json::parse(read_text(folder + "/meta.json"));
    for (int i = 0; i < 3; ++i) {
        shape[i] = (int64_t)meta["shape"][i].num();
        brick[i] = (int64_t)meta["brick"][i].num();
        grid[i] = (shape[i] + brick[i] - 1) / brick[i];
    }
    if (meta.has("source") && meta["source"].kind == Json::String)
        source = meta["source"].str();
    Mapped table;
    map_npy(table, folder + "/offsets.npy", false);
    if (table.descr != "<i8" || count(table.shape) != (size_t)(grid[0] * grid[1] * grid[2]))
        throw std::runtime_error("unexpected offsets table in " + folder);
    offsets.assign(
        (const int64_t *)table.data(), (const int64_t *)table.data() + count(table.shape));
    const std::string path = folder + "/bricks.bin";
    if (exists(path)) {
        int fd = ::open(path.c_str(), O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        bricks_length = (size_t)st.st_size;
        if (bricks_length) {
            void *p = mmap(nullptr, bricks_length, PROT_READ, MAP_SHARED, fd, 0);
            if (p == MAP_FAILED)
                throw std::runtime_error("cannot map " + path);
            bricks_map.base = (uint8_t *)p;
            bricks_map.length = bricks_length;
            bricks = (const uint8_t *)p;
        }
        ::close(fd);
    }
}

void BrickStore::read(const Box &box, uint8_t *out) const {
    const int64_t dz = box[0][1] - box[0][0], dy = box[1][1] - box[1][0],
                  dx = box[2][1] - box[2][0];
    std::memset(out, 0, (size_t)(dz * dy * dx));
    const int64_t size = brick[0] * brick[1] * brick[2];
    for (int64_t i = box[0][0] / brick[0]; i < (box[0][1] + brick[0] - 1) / brick[0]; ++i)
        for (int64_t j = box[1][0] / brick[1]; j < (box[1][1] + brick[1] - 1) / brick[1]; ++j)
            for (int64_t k = box[2][0] / brick[2]; k < (box[2][1] + brick[2] - 1) / brick[2]; ++k) {
                if (i >= grid[0] || j >= grid[1] || k >= grid[2])
                    continue;
                const int64_t offset = offsets[(size_t)((i * grid[1] + j) * grid[2] + k)];
                if (offset < 0 || !bricks || (size_t)(offset + size) > bricks_length)
                    continue;
                const uint8_t *cube = bricks + offset;
                const int64_t z0 = std::max(box[0][0], i * brick[0]),
                              z1 = std::min(box[0][1], (i + 1) * brick[0]);
                const int64_t y0 = std::max(box[1][0], j * brick[1]),
                              y1 = std::min(box[1][1], (j + 1) * brick[1]);
                const int64_t x0 = std::max(box[2][0], k * brick[2]),
                              x1 = std::min(box[2][1], (k + 1) * brick[2]);
                for (int64_t z = z0; z < z1; ++z)
                    for (int64_t y = y0; y < y1; ++y) {
                        const uint8_t *src = cube +
                            ((z - i * brick[0]) * brick[1] + (y - j * brick[1])) * brick[2] +
                            (x0 - k * brick[2]);
                        uint8_t *dst =
                            out + ((z - box[0][0]) * dy + (y - box[1][0])) * dx + (x0 - box[2][0]);
                        std::memcpy(dst, src, (size_t)(x1 - x0));
                    }
            }
}

}
