#include "fetch.h"
#include "grid.h"
#include "json.h"
#include "net.h"
#include "predict.h"
#include "volume.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace hx;

static const char *OPEN_DATA = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

struct Span {
    int64_t keep0, keep1, read0, read1;
};

static std::vector<Span> crop_spans(
    int64_t start, int64_t stop, int64_t size, int64_t margin, int64_t patch) {
    std::vector<Span> spans;
    for (int64_t a = start; a < stop;) {
        const int64_t b = std::min(stop, a + size);
        int64_t lo = std::max(start, a - margin), hi = std::min(stop, b + margin);
        if (hi - lo < patch) {
            const int64_t nlo = std::max(start, std::min(lo, hi - patch)),
                          nhi = std::min(stop, std::max(hi, lo + patch));
            lo = nlo;
            hi = nhi;
        }
        spans.push_back({a, b, lo, hi});
        a = b;
    }
    return spans;
}

static Box parse_box(const std::string &text) {
    Box box{};
    int i = 0;
    size_t at = 0;
    while (i < 3) {
        size_t comma = text.find(',', at);
        std::string part =
            text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        size_t colon = part.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error("box must be z0:z1,y0:y1,x0:x1");
        box[i][0] = std::stoll(part.substr(0, colon));
        box[i][1] = std::stoll(part.substr(colon + 1));
        if (box[i][1] <= box[i][0])
            throw std::runtime_error("empty box span");
        ++i;
        if (comma == std::string::npos)
            break;
        at = comma + 1;
    }
    if (i != 3)
        throw std::runtime_error("box must have three spans");
    return box;
}

static std::string fmt(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

static std::string absolute(const std::string &path) {
    char resolved[PATH_MAX];
    return realpath(path.c_str(), resolved) ? std::string(resolved) : path;
}

static std::string box_json(const Box &box) {
    std::string s = "[";
    for (int d = 0; d < 3; ++d)
        s += (d ? ", [" : "[") + std::to_string(box[d][0]) + ", " + std::to_string(box[d][1]) + "]";
    return s + "]";
}

static void usage() {
    std::fprintf(stderr,
        "surface_predict --spec NET.json --weights NET.safetensors (--store DIR --box "
        "z0:z1,y0:y1,x0:x1 | "
        "--npy CT.npy)\n"
        "                [--out DIR] [--step 0.5] [--no-mirror] [--crop N] [--logits FILE.npy] "
        "[--forward | "
        "--forward-mirrored]\n"
        "                [--source PATH//LEVEL] [--no-fetch | --fetch-only] [--bucket URL] "
        "[--connections 32]\n"
        "                [--fiesta-grid DIR] [--threshold 128] [--cache-gb 1] [--quiet]\n");
}

int main(int argc, char **argv) {
    std::string spec, weights, store_dir, box_text, npy, out_dir, logits_path, grid_dir,
        source_name, bucket = OPEN_DATA;
    double step = 0.5, cache_gb = 1.0;
    bool mirror = true, forward_only = false, forward_mirrored = false, verbose = true,
         fetch = true, fetch_only = false;
    int64_t crop = 0;
    int repeat = 1, connections = 32, threshold = 128;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        try {
            if (a == "--spec")
                spec = next();
            else if (a == "--weights")
                weights = next();
            else if (a == "--store")
                store_dir = next();
            else if (a == "--box")
                box_text = next();
            else if (a == "--npy")
                npy = next();
            else if (a == "--out")
                out_dir = next();
            else if (a == "--logits")
                logits_path = next();
            else if (a == "--fiesta-grid")
                grid_dir = next();
            else if (a == "--bucket")
                bucket = next();
            else if (a == "--source")
                source_name = next();
            else if (a == "--step")
                step = std::stod(next());
            else if (a == "--crop")
                crop = std::stoll(next());
            else if (a == "--cache-gb")
                cache_gb = std::stod(next());
            else if (a == "--repeat")
                repeat = std::stoi(next());
            else if (a == "--connections")
                connections = std::stoi(next());
            else if (a == "--threshold")
                threshold = std::stoi(next());
            else if (a == "--no-mirror")
                mirror = false;
            else if (a == "--forward")
                forward_only = true;
            else if (a == "--forward-mirrored")
                forward_only = forward_mirrored = true;
            else if (a == "--no-fetch")
                fetch = false;
            else if (a == "--fetch-only")
                fetch_only = true;
            else if (a == "--quiet")
                verbose = false;
            else {
                usage();
                return 2;
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 2;
        }
    }
    if (!store_dir.empty()) {
        try {
            if (partial_store(store_dir))
                throw std::runtime_error(store_dir +
                    " holds only the planes a read fetched; use a full-brick store, "
                    "for example a new folder with --source PATH//LEVEL");
            if (!exists(store_dir + "/meta.json")) {
                if (source_name.empty())
                    throw std::runtime_error(
                        store_dir + " is not a brick store; --source creates one");
                create_store(store_dir, source_name, bucket);
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "surface_predict: %s\n", e.what());
            return 1;
        }
    }
    if (fetch_only && !store_dir.empty() && !box_text.empty()) {
        try {
            const int64_t got = fetch_missing(store_dir, parse_box(box_text), bucket, connections);
            if (verbose)
                std::fprintf(stderr, "fetched %lld bricks of the box\n", (long long)got);
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "surface_predict: %s\n", e.what());
            return 1;
        }
    }
    if (fetch_only || spec.empty() || weights.empty() || (store_dir.empty() == npy.empty())) {
        usage();
        return 2;
    }
    try {
        Json description = Json::parse(read_text(spec));
        SurfaceNet net(spec, weights, cache_gb);
        BrickStore store;
        Mapped source;
        Box box{};
        if (!store_dir.empty()) {
            box = parse_box(box_text);
            if (fetch) {
                const int64_t got = fetch_missing(store_dir, box, bucket, connections);
                if (got && verbose)
                    std::fprintf(stderr, "fetched %lld bricks of the box\n", (long long)got);
            }
            store.open(store_dir);
        } else {
            map_npy(source, npy, false);
            if (source.descr != "|u1" || source.shape.size() != 3)
                throw std::runtime_error("--npy must be uint8 (z, y, x)");
            for (int d = 0; d < 3; ++d)
                box[d] = {0, source.shape[d]};
        }
        const std::array<int64_t, 3> shape = {
            box[0][1] - box[0][0], box[1][1] - box[1][0], box[2][1] - box[2][0]};
        auto read_box = [&](const Box &b, uint8_t *out) {
            if (!store_dir.empty()) {
                store.read(b, out);
                return;
            }
            const int64_t dy = b[1][1] - b[1][0], dx = b[2][1] - b[2][0];
            for (int64_t z = b[0][0]; z < b[0][1]; ++z)
                for (int64_t y = b[1][0]; y < b[1][1]; ++y)
                    std::memcpy(out + ((z - b[0][0]) * dy + (y - b[1][0])) * dx,
                        source.data() + (z * shape[1] + y) * shape[2] + b[2][0], (size_t)dx);
        };
        if (forward_only) {
            const size_t n = (size_t)(shape[0] * shape[1] * shape[2]);
            std::vector<uint8_t> ct(n);
            read_box(box, ct.data());
            std::vector<float> x(n);
            net.normalise(ct.data(), n, x.data());
            auto input =
                mx::array(x.data(), {(int)shape[0], (int)shape[1], (int)shape[2], 1}, mx::float32);
            mx::array out(0.0f);
            double best = 1e30;
            for (int r = 0; r < std::max(1, repeat); ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                out = mx::contiguous(mx::transpose(
                    forward_mirrored ? mirrored_forward(net, input) : net.forward(input),
                    {3, 0, 1, 2}));
                mx::eval(out);
                best = std::min(best,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::fprintf(stderr, "forward %.3f s\n", best);
            if (logits_path.empty())
                throw std::runtime_error("--forward needs --logits");
            save_npy(logits_path, "<f4", {2, shape[0], shape[1], shape[2]}, out.data<float>(),
                out.nbytes());
            return 0;
        }
        if (out_dir.empty())
            throw std::runtime_error("--out is required");
        mkdir(out_dir.c_str(), 0755);
        const int P = net.patch();
        if (std::min({shape[0], shape[1], shape[2]}) < P)
            throw std::runtime_error(
                "the box must be at least " + std::to_string(P) + " voxels in every axis");
        std::string settings = "{\"box\": " + box_json(box) + ", \"net\": \"" + net.name() +
            "\", \"step\": " + fmt(step) + ", \"mirror\": " + (mirror ? "true" : "false") +
            ", \"crop\": " + std::to_string(crop) + "}";
        const std::string progress = out_dir + "/crops.json", surface = out_dir + "/surface.npy";
        std::vector<std::array<int64_t, 4>> finished;
        if (exists(progress) && exists(surface)) {
            Json kept = Json::parse(read_text(progress));
            if (kept["settings"].kind == Json::String && kept["settings"].str() == settings)
                for (const Json &f : kept["finished"].items)
                    finished.push_back({(int64_t)f[0].num(), (int64_t)f[1].num(),
                        (int64_t)f[2].num(), (int64_t)f[3].num()});
        }
        Mapped probability;
        if (!finished.empty())
            map_npy(probability, surface, true);
        else
            create_npy(probability, surface, "|u1", {shape[0], shape[1], shape[2]});
        Mapped logits;
        const bool half = net.rule() == Rule::NNUNET;
        if (!logits_path.empty())
            create_npy(
                logits, logits_path, half ? "<f2" : "<f4", {2, shape[0], shape[1], shape[2]});
        const int64_t size_y = crop ? crop : shape[1], size_x = crop ? crop : shape[2],
                      margin = crop ? P / 2 : 0;
        const auto started = std::chrono::steady_clock::now();
        for (const Span &ky : crop_spans(0, shape[1], size_y, margin, P))
            for (const Span &kx : crop_spans(0, shape[2], size_x, margin, P)) {
                const std::array<int64_t, 4> id = {ky.keep0, ky.keep1, kx.keep0, kx.keep1};
                if (std::find(finished.begin(), finished.end(), id) != finished.end())
                    continue;
                const int64_t ry = ky.read1 - ky.read0, rx = kx.read1 - kx.read0;
                auto read = [&](const Box &b, uint8_t *out) {
                    read_box(Box{{{box[0][0] + b[0][0], box[0][0] + b[0][1]},
                                 {box[1][0] + ky.read0 + b[1][0], box[1][0] + ky.read0 + b[1][1]},
                                 {box[2][0] + kx.read0 + b[2][0], box[2][0] + kx.read0 + b[2][1]}}},
                        out);
                };
                auto write = [&](int64_t z, int64_t rows, const std::vector<float> &l) {
                    const int64_t plane = ry * rx;
                    uint8_t *p = probability.data();
                    for (int64_t r = 0; r < rows; ++r)
                        for (int64_t y = ky.keep0; y < ky.keep1; ++y)
                            for (int64_t x = kx.keep0; x < kx.keep1; ++x) {
                                const int64_t at = r * plane + (y - ky.read0) * rx + (x - kx.read0);
                                const float b = l[(size_t)at], s = l[(size_t)(rows * plane + at)];
                                const size_t to = (size_t)(((z + r) * shape[1] + y) * shape[2] + x);
                                p[to] = surface_probability(b, s);
                                if (logits.base) {
                                    const size_t c1 = (size_t)(shape[0] * shape[1] * shape[2]) + to;
                                    if (half) {
                                        ((_Float16 *)logits.data())[to] = (_Float16)b;
                                        ((_Float16 *)logits.data())[c1] = (_Float16)s;
                                    } else {
                                        ((float *)logits.data())[to] = b;
                                        ((float *)logits.data())[c1] = s;
                                    }
                                }
                            }
                    probability.flush();
                };
                predict_volume(net, read, {shape[0], ry, rx}, step, mirror, write, verbose);
                finished.push_back(id);
                std::string record = "{\"settings\": \"";
                for (char c : settings)
                    record += c == '"' ? std::string("\\\"") : std::string(1, c);
                record += "\", \"finished\": [";
                for (size_t f = 0; f < finished.size(); ++f)
                    record += (f ? ", [" : "[") + std::to_string(finished[f][0]) + ", " +
                        std::to_string(finished[f][1]) + ", " + std::to_string(finished[f][2]) +
                        ", " + std::to_string(finished[f][3]) + "]";
                write_text(progress, record + "]}");
                if (crop && verbose) {
                    const double s =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                            .count();
                    std::fprintf(stderr, "crop y [%lld, %lld] x [%lld, %lld] done, %.0f s\n",
                        (long long)ky.keep0, (long long)ky.keep1, (long long)kx.keep0,
                        (long long)kx.keep1, s);
                }
            }
        if (logits.base)
            logits.flush();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const size_t n = (size_t)(shape[0] * shape[1] * shape[2]);
        size_t above = 0;
        for (size_t i = 0; i < n; ++i)
            above += probability.data()[i] > 127;
        const std::string blend = half ? "nnunet" : "vesuvius";
        char rate[64];
        std::snprintf(
            rate, sizeof rate, "%.1f,\n \"voxels_per_second\": %.0f", seconds, (double)n / seconds);
        std::string weights_json = "null";
        if (description.has("source") && description["source"].has("path") &&
            description["source"].has("sha256"))
            weights_json = "{\"path\": " + quote(description["source"]["path"].str()) +
                ", \"sha256\": " + quote(description["source"]["sha256"].str()) + "}";
        const std::string record =
            "{\n \"store\": " + (store_dir.empty() ? std::string("null") : quote(store_dir)) +
            ",\n \"source\": " +
            (store.source.empty() ? std::string("null") : quote(store.source)) +
            ",\n \"npy\": " + (npy.empty() ? std::string("null") : quote(npy)) +
            ",\n \"box\": " + box_json(box) + ",\n \"net\": " + quote(net.name()) +
            ",\n \"weights\": " + weights_json + ",\n \"blend\": \"" + blend +
            "\",\n \"step\": " + fmt(step) + ",\n \"mirror\": " + (mirror ? "true" : "false") +
            ",\n \"crop\": " + std::to_string(crop) + ",\n \"seconds\": " + rate +
            ",\n \"surface_fraction\": " + fmt((double)above / (double)n) + "\n}\n";
        write_text(out_dir + "/surface.json", record);
        std::printf("{\"net\": \"%s\", \"blend\": \"%s\", \"step\": %s, \"mirror\": %s, \"crop\": "
                    "%lld, \"seconds\": %.1f, "
                    "\"voxels_per_second\": %.0f, \"surface_fraction\": %s}\n",
            net.name().c_str(), blend.c_str(), fmt(step).c_str(), mirror ? "true" : "false",
            (long long)crop, seconds, (double)n / seconds, fmt((double)above / (double)n).c_str());
        if (!grid_dir.empty()) {
            const int64_t cubes = write_fiesta_grid(grid_dir, probability.data(), box, read_box,
                store_dir.empty() ? npy : store_dir, absolute(out_dir), net.name(), threshold);
            if (verbose)
                std::fprintf(
                    stderr, "%lld surface cubes -> %s\n", (long long)cubes, grid_dir.c_str());
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "surface_predict: %s\n", e.what());
        return 1;
    }
}
