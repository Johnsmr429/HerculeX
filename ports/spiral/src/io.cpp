#include "io.h"

#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <fstream>
#include <lz4.h>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sp {

std::string read_text(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
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

static uint32_t u32(const uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

Npz::Npz(const std::string &path) : file(path) {
    const uint8_t *b = file.data;
    size_t eocd = file.size >= 22 ? file.size - 22 : 0;
    while (eocd > 0 && u32(b + eocd) != 0x06054b50)
        --eocd;
    if (u32(b + eocd) != 0x06054b50)
        throw std::runtime_error("no zip directory in " + path);
    const size_t entries = u16(b + eocd + 10);
    size_t at = u32(b + eocd + 16);
    for (size_t i = 0; i < entries; ++i) {
        if (u32(b + at) != 0x02014b50)
            throw std::runtime_error("bad zip directory in " + path);
        const uint16_t method = u16(b + at + 10), name_len = u16(b + at + 28),
                       extra_len = u16(b + at + 30), comment_len = u16(b + at + 32);
        uint64_t compressed = u32(b + at + 20), local = u32(b + at + 42);
        std::string name((const char *)b + at + 46, name_len);
        const uint8_t *extra = b + at + 46 + name_len;
        for (size_t x = 0; x + 4 <= extra_len;) {
            const uint16_t tag = u16(extra + x), n = u16(extra + x + 2);
            if (tag == 1) {
                size_t k = x + 4;
                if (u32(b + at + 24) == 0xFFFFFFFF)
                    k += 8;
                if (compressed == 0xFFFFFFFF) {
                    std::memcpy(&compressed, extra + k, 8);
                    k += 8;
                }
                if (local == 0xFFFFFFFF)
                    std::memcpy(&local, extra + k, 8);
            }
            x += 4 + n;
        }
        if (method != 0)
            throw std::runtime_error("compressed npz member " + name);
        const uint8_t *header = b + local;
        const size_t data = local + 30 + u16(header + 26) + u16(header + 28);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0)
            name = name.substr(0, name.size() - 4);
        arrays[name] = parse_npy(b + data, (size_t)compressed);
        at += 46 + name_len + extra_len + comment_len;
    }
}

Tracks load_tracks(const std::string &store, int64_t z_lo, int64_t z_hi) {
    Mapping header(store + "/header.bin");
    if (header.size < 64 || std::memcmp(header.data, "VCTRK01", 8) != 0 ||
        u32(header.data + 8) != 1)
        throw std::runtime_error("unsupported packed track store " + store);
    uint64_t tracks, points;
    std::memcpy(&tracks, header.data + 16, 8);
    std::memcpy(&points, header.data + 24, 8);
    Mapping coordinates(store + "/coordinates.i32"), offsets(store + "/offsets.i64"),
        sources(store + "/source_ids.u64"), families(store + "/family_codes.i8"),
        bounds(store + "/z_bounds.i32"), arcs(store + "/arclengths.f64"),
        tortuous(store + "/tortuosities.f64");
    if (coordinates.size != points * 12 || offsets.size != (tracks + 1) * 8 ||
        bounds.size != tracks * 8)
        throw std::runtime_error("packed track store sizes disagree in " + store);
    const int32_t *xyz = (const int32_t *)coordinates.data;
    const int64_t *off = (const int64_t *)offsets.data;
    const int32_t *zb = (const int32_t *)bounds.data;
    Tracks t;
    for (uint64_t i = 0; i < tracks; ++i) {
        if (off[i + 1] == off[i] || zb[2 * i] < z_lo || zb[2 * i + 1] >= z_hi)
            continue;
        for (int64_t p = off[i]; p < off[i + 1]; ++p)
            for (int c = 0; c < 3; ++c)
                t.zyx.push_back((float)xyz[p * 3 + c]);
        t.offsets.push_back((int64_t)(t.zyx.size() / 3));
        t.source_ids.push_back(((const uint64_t *)sources.data)[i]);
        t.families.push_back(((const int8_t *)families.data)[i]);
        t.arclengths.push_back(((const double *)arcs.data)[i]);
        t.tortuosities.push_back(((const double *)tortuous.data)[i]);
    }
    return t;
}

static void blosc_decode(const uint8_t *src, size_t length, uint8_t *dst, size_t expected) {
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

static void read_rows(const std::string &array, const std::string &group, int64_t z_lo,
    int64_t z_hi, std::array<int64_t, 3> &shape, std::vector<uint8_t> &out) {
    Json meta = Json::parse(read_text(array + "/" + group + "/.zarray"));
    std::array<int64_t, 3> full{}, chunk{};
    for (int i = 0; i < 3; ++i) {
        full[i] = (int64_t)meta["shape"][i].num();
        chunk[i] = (int64_t)meta["chunks"][i].num();
    }
    if (meta["dtype"].str() != "|u1" || meta["compressor"]["id"].str() != "blosc")
        throw std::runtime_error("unexpected zarr array " + array);
    shape = {z_hi - z_lo, full[1], full[2]};
    out.assign((size_t)(shape[0] * shape[1] * shape[2]), 0);
    const int64_t cz0 = z_lo / chunk[0], cz1 = (z_hi - 1) / chunk[0];
    const int64_t cy = (full[1] + chunk[1] - 1) / chunk[1],
                  cx = (full[2] + chunk[2] - 1) / chunk[2];
    const int64_t jobs = (cz1 - cz0 + 1) * cy * cx;
    __block std::string failure;
    dispatch_apply(
        (size_t)jobs, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t job) {
          const int64_t kz = cz0 + (int64_t)job / (cy * cx), ky = ((int64_t)job / cx) % cy,
                        kx = (int64_t)job % cx;
          const std::string path = array + "/" + group + "/" + std::to_string(kz) + "/" +
              std::to_string(ky) + "/" + std::to_string(kx);
          struct stat st;
          if (stat(path.c_str(), &st) != 0)
              return;
          try {
              Mapping m(path);
              std::vector<uint8_t> cube((size_t)(chunk[0] * chunk[1] * chunk[2]));
              blosc_decode(m.data, m.size, cube.data(), cube.size());
              for (int64_t z = std::max(z_lo, kz * chunk[0]);
                  z < std::min(z_hi, (kz + 1) * chunk[0]); ++z)
                  for (int64_t y = ky * chunk[1]; y < std::min(full[1], (ky + 1) * chunk[1]); ++y) {
                      const int64_t x0 = kx * chunk[2], x1 = std::min(full[2], (kx + 1) * chunk[2]);
                      std::memcpy(&out[(size_t)(((z - z_lo) * full[1] + y) * full[2] + x0)],
                          &cube[(size_t)(((z - kz * chunk[0]) * chunk[1] + (y - ky * chunk[1])) *
                              chunk[2])],
                          (size_t)(x1 - x0));
                  }
          } catch (const std::exception &e) {
              failure = e.what();
          }
        });
    if (!failure.empty())
        throw std::runtime_error(failure);
}

Lasagna load_lasagna(const std::string &nx, const std::string &ny, const std::string &grad_mag,
    const std::string &group, int64_t scale, int64_t z_begin, int64_t z_end) {
    Json meta = Json::parse(read_text(nx + "/" + group + "/.zarray"));
    const int64_t z_size = (int64_t)meta["shape"][0].num();
    Lasagna l;
    l.scale = scale;
    l.z_origin = std::max<int64_t>(0, (int64_t)std::floor((double)z_begin / (double)scale));
    const int64_t z_hi =
        std::min<int64_t>(z_size, (int64_t)std::ceil((double)z_end / (double)scale));
    if (z_hi <= l.z_origin)
        throw std::runtime_error("lasagna z range is empty");
    std::array<int64_t, 3> a, b, c;
    read_rows(nx, group, l.z_origin, z_hi, a, l.nx);
    read_rows(ny, group, l.z_origin, z_hi, b, l.ny);
    read_rows(grad_mag, group, l.z_origin, z_hi, c, l.grad_mag);
    if (a != b || a != c)
        throw std::runtime_error("lasagna stores disagree in shape");
    l.shape = a;
    return l;
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
            out[i] = (float)v.as<double>()[i];
    else
        throw std::runtime_error("unsupported dtype " + v.descr + " in " + path);
    return out;
}

std::vector<int64_t> load_npy_int(const std::string &path, std::vector<int64_t> *shape) {
    Mapping m(path);
    ArrayView v = parse_npy(m.data, m.size);
    if (shape)
        *shape = v.shape;
    std::vector<int64_t> out(v.count());
    if (v.descr == "<i8")
        std::memcpy(out.data(), v.data, out.size() * 8);
    else if (v.descr == "<i4")
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = v.as<int32_t>()[i];
    else if (v.descr == "|b1" || v.descr == "|u1")
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = v.as<uint8_t>()[i];
    else
        throw std::runtime_error("unsupported dtype " + v.descr + " in " + path);
    return out;
}

bool exists(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

Image read_tiff(const std::string &path) {
    Mapping file(path);
    const uint8_t *b = file.data;
    if (file.size < 8 || b[0] != 'I' || b[1] != 'I' || b[2] != 42 || b[3] != 0)
        throw std::runtime_error(path + ": not a little-endian TIFF");
    auto u16 = [&](size_t at) { return (uint32_t)b[at] | ((uint32_t)b[at + 1] << 8); };
    auto u32v = [&](size_t at) {
        return (uint32_t)b[at] | ((uint32_t)b[at + 1] << 8) | ((uint32_t)b[at + 2] << 16) |
            ((uint32_t)b[at + 3] << 24);
    };
    const size_t ifd = u32v(4);
    const uint32_t entries = u16(ifd);
    uint32_t width = 0, height = 0, bits = 8, compression = 1, samples = 1, rows = 0, format = 1,
             planar = 1;
    std::vector<uint32_t> offsets, counts;
    auto values = [&](size_t entry, std::vector<uint32_t> &out) {
        const uint32_t type = u16(entry + 2), count = u32v(entry + 4);
        const size_t size = type == 3 ? 2 : 4;
        const size_t at = count * size <= 4 ? entry + 8 : u32v(entry + 8);
        out.resize(count);
        for (uint32_t k = 0; k < count; ++k)
            out[k] = type == 3 ? u16(at + k * 2) : u32v(at + k * 4);
    };
    for (uint32_t e = 0; e < entries; ++e) {
        const size_t entry = ifd + 2 + 12 * (size_t)e;
        const uint32_t tag = u16(entry);
        std::vector<uint32_t> v;
        values(entry, v);
        if (v.empty())
            continue;
        if (tag == 256)
            width = v[0];
        else if (tag == 257)
            height = v[0];
        else if (tag == 258)
            bits = v[0];
        else if (tag == 259)
            compression = v[0];
        else if (tag == 273)
            offsets = v;
        else if (tag == 277)
            samples = v[0];
        else if (tag == 278)
            rows = v[0];
        else if (tag == 279)
            counts = v;
        else if (tag == 284)
            planar = v[0];
        else if (tag == 339)
            format = v[0];
        else if (tag == 322 || tag == 323)
            throw std::runtime_error(path + ": tiled TIFFs are not supported");
    }
    if (compression != 1)
        throw std::runtime_error(path + ": compressed TIFFs are not supported");
    if (planar != 1 && samples > 1)
        throw std::runtime_error(path + ": planar TIFFs are not supported");
    if (offsets.empty() || offsets.size() != counts.size())
        throw std::runtime_error(path + ": TIFF without strips");
    Image image;
    image.H = (int)height;
    image.W = (int)width;
    image.C = (int)samples;
    const size_t n = (size_t)width * height * samples, item = bits / 8;
    std::vector<uint8_t> raw;
    raw.reserve(n * item);
    for (size_t k = 0; k < offsets.size(); ++k) {
        if ((size_t)offsets[k] + counts[k] > file.size)
            throw std::runtime_error(path + ": truncated TIFF strip");
        raw.insert(raw.end(), b + offsets[k], b + offsets[k] + counts[k]);
    }
    if (raw.size() < n * item)
        throw std::runtime_error(path + ": TIFF strips shorter than the image");
    image.data.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t *p = raw.data() + i * item;
        if (format == 3 && bits == 32)
            std::memcpy(&image.data[i], p, 4);
        else if (format == 3 && bits == 64) {
            double d;
            std::memcpy(&d, p, 8);
            image.data[i] = (float)d;
        } else if (bits == 8)
            image.data[i] = format == 2 ? (float)(int8_t)p[0] : (float)p[0];
        else if (bits == 16) {
            uint16_t v;
            std::memcpy(&v, p, 2);
            image.data[i] = format == 2 ? (float)(int16_t)v : (float)v;
        } else if (bits == 32) {
            uint32_t v;
            std::memcpy(&v, p, 4);
            image.data[i] = format == 2 ? (float)(int32_t)v : (float)v;
        } else
            throw std::runtime_error(path + ": unsupported TIFF sample format");
    }
    return image;
}

std::vector<std::array<float, 3>> load_umbilicus(const std::string &path) {
    Json data = Json::parse(read_text(path));
    std::vector<std::array<float, 3>> points;
    for (const Json &p : data["control_points"].items)
        points.push_back({(float)p["z"].num(), (float)p["y"].num(), (float)p["x"].num()});
    std::stable_sort(
        points.begin(), points.end(), [](const auto &a, const auto &b) { return a[0] < b[0]; });
    return points;
}

}
