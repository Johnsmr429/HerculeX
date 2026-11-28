#include "io.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <lz4.h>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace la {

std::string read_text(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

bool exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

Mapping::Mapping(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    size = (size_t)st.st_size;
    if (size) {
        void *p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("cannot map " + path);
        }
        data = (const uint8_t *)p;
    }
    ::close(fd);
}

Mapping::Mapping(Mapping &&other) noexcept : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
}

Mapping &Mapping::operator=(Mapping &&other) noexcept {
    std::swap(data, other.data);
    std::swap(size, other.size);
    return *this;
}

Mapping::~Mapping() {
    if (data)
        munmap((void *)data, size);
}

size_t ArrayView::count() const {
    size_t n = 1;
    for (auto s : shape)
        n *= (size_t)s;
    return n;
}

ArrayView parse_npy(const uint8_t *b, size_t size) {
    if (size < 10 || std::memcmp(b, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("not an npy array");
    size_t len, start;
    if (b[6] == 1) {
        len = b[8] | (b[9] << 8);
        start = 10;
    } else {
        len = b[8] | (b[9] << 8) | ((size_t)b[10] << 16) | ((size_t)b[11] << 24);
        start = 12;
    }
    std::string h((const char *)b + start, len);
    ArrayView v;
    v.data = b + start + len;
    size_t d = h.find("'descr'");
    size_t q0 = h.find('\'', h.find(':', d) + 1), q1 = h.find('\'', q0 + 1);
    v.descr = h.substr(q0 + 1, q1 - q0 - 1);
    size_t f = h.find("'fortran_order'");
    if (h.compare(h.find(':', f) + 2, 4, "True") == 0)
        throw std::runtime_error("fortran-ordered npy");
    size_t s = h.find('(', h.find("'shape'")), e = h.find(')', s);
    std::stringstream parts(h.substr(s + 1, e - s - 1));
    std::string item;
    while (std::getline(parts, item, ','))
        if (item.find_first_not_of(' ') != std::string::npos)
            v.shape.push_back(std::stoll(item));
    return v;
}

std::vector<float> load_npy_float(const std::string &path, std::vector<int64_t> *shape) {
    Mapping m(path);
    ArrayView v = parse_npy(m.data, m.size);
    if (shape)
        *shape = v.shape;
    std::vector<float> out(v.count());
    if (v.descr == "<f4")
        std::memcpy(out.data(), v.data, out.size() * 4);
    else if (v.descr == "<f8")
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = (float)((const double *)v.data)[i];
    else
        throw std::runtime_error("unsupported dtype " + v.descr + " in " + path);
    return out;
}

std::vector<uint8_t> load_npy_u8(const std::string &path, std::vector<int64_t> *shape) {
    Mapping m(path);
    ArrayView v = parse_npy(m.data, m.size);
    if (shape)
        *shape = v.shape;
    if (v.descr != "|u1")
        throw std::runtime_error("unsupported dtype " + v.descr + " in " + path);
    return std::vector<uint8_t>(v.data, v.data + v.count());
}

static size_t item_size(const std::string &descr) { return (size_t)std::stoi(descr.substr(2)); }

void save_npy(const std::string &path, const std::string &descr, const std::vector<int64_t> &shape,
    const void *data) {
    std::string dims;
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += (i ? ", " : "") + std::to_string(shape[i]);
        n *= (size_t)shape[i];
    }
    if (shape.size() == 1)
        dims += ",";
    std::string dict =
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const size_t total = 10 + dict.size() + 1;
    dict += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::string head = "\x93NUMPY";
    head += (char)1;
    head += (char)0;
    head += (char)(dict.size() & 0xFF);
    head += (char)((dict.size() >> 8) & 0xFF);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    out << head << dict;
    out.write((const char *)data, (std::streamsize)(n * item_size(descr)));
}

static uint32_t u32(const uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}

void blosc_decode(const uint8_t *src, size_t length, uint8_t *dst, size_t expected) {
    if (length < 16)
        throw std::runtime_error("short blosc frame");
    const uint8_t flags = src[2];
    const size_t typesize = src[3], nbytes = u32(src + 4), blocksize = u32(src + 8);
    if (nbytes != expected)
        throw std::runtime_error("blosc frame of unexpected size");
    if (flags & 0x02) {
        std::memcpy(dst, src + 16, nbytes);
        return;
    }
    if ((flags >> 5) != 1)
        throw std::runtime_error("blosc frame not lz4");
    if (typesize != 1)
        throw std::runtime_error("blosc typesize other than 1");
    const size_t blocks = (nbytes + blocksize - 1) / blocksize;
    for (size_t k = 0; k < blocks; ++k) {
        const size_t size = std::min(blocksize, nbytes - k * blocksize);
        const uint8_t *block = src + u32(src + 16 + 4 * k);
        const size_t csize = u32(block);
        uint8_t *out = dst + k * blocksize;
        if (csize == size)
            std::memcpy(out, block + 4, size);
        else if (LZ4_decompress_safe((const char *)block + 4, (char *)out, (int)csize, (int)size) !=
            (int)size)
            throw std::runtime_error("lz4 decode failed");
    }
}

}
