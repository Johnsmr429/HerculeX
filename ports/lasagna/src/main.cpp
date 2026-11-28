#include "barrier.h"
#include "cylinder.h"
#include "data.h"
#include "fit.h"
#include "pyramid.h"
#include "io.h"
#include "json.h"
#include "mesh.h"
#include "shell.h"

#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace la;

static int sample(const std::string &manifest, const std::string &points,
    const std::string &weights, const std::string &out) {
    const std::vector<std::string> names = {"grad_mag", "nx", "ny"};
    Volume v = Volume::open(manifest, names);
    std::vector<int64_t> shape;
    const std::vector<float> xyz = load_npy_float(points, &shape);
    const size_t n = xyz.size() / 3;
    std::vector<float> grad(n * 3, 0.0f), part(n * 3);
    for (const std::string &name : names) {
        Channel &c = v.channel(name);
        v.prefetch(c, xyz.data(), n);
        std::vector<uint8_t> u8(n);
        std::vector<float> raw(n), plain(n), smooth(n), go(n);
        sample_u8(v, c, xyz.data(), n, u8.data());
        sample_raw(v, c, xyz.data(), n, raw.data());
        for (size_t i = 0; i < n; ++i) {
            plain[i] = decode(v, name, (float)u8[i]);
            smooth[i] = decode(v, name, raw[i]);
        }
        const std::vector<float> w = load_npy_float(weights + "/" + name + "_weight.npy");
        for (size_t i = 0; i < n; ++i)
            go[i] = decode_backward(v, name, w[i]);
        sample_raw_backward(v, c, xyz.data(), n, go.data(), part.data());
        for (size_t i = 0; i < n * 3; ++i)
            grad[i] += part[i];
        save_npy(out + "/" + name + ".npy", "<f4", {(int64_t)n}, plain.data());
        save_npy(out + "/" + name + "_diff.npy", "<f4", {(int64_t)n}, smooth.data());
        std::printf("%s: %zu blocks\n", name.c_str(), c.blocks.size());
    }
    save_npy(out + "/grad_points.npy", "<f4", {(int64_t)n, 3}, grad.data());
    return 0;
}

static DepthVolume load_barrier(const std::string &path, const Json &c) {
    DepthVolume field;
    std::vector<int64_t> vs;
    field.v = load_npy_u8(path, &vs);
    field.Z = (int)vs[1];
    field.Y = (int)vs[2];
    field.X = (int)vs[3];
    for (int a = 0; a < 3; ++a) {
        field.origin[a] = (float)c["cyl_outside_origin"][(size_t)a].num();
        field.spacing[a] = (float)c["cyl_outside_spacing"][(size_t)a].num();
    }
    return field;
}

static int cyl_loss(const std::string &name, const std::string &manifest, const std::string &dir,
    const std::string &out) {
    std::vector<int64_t> shape;
    Shell s;
    s.xyz = load_npy_float(dir + "/cyl_xyz.npy", &shape);
    s.H = (int)shape[shape.size() - 3];
    s.W = (int)shape[shape.size() - 2];
    Json c = Json::parse(read_text(dir + "/case.json"));
    auto num = [&](const char *key) {
        return c.has(key) && c[key].kind == Json::Number ? c[key].num() : 0.0;
    };
    LossOut o;
    if (name == "cyl_smooth")
        o = cyl_smooth(s);
    else if (name == "cyl_z_smooth")
        o = cyl_z_smooth(s);
    else if (name == "cyl_z_center")
        o = cyl_z_center(s,
            c.has("cyl_z_center_target") ? num("cyl_z_center_target") : num("cyl_seed_z"),
            num("cyl_shell_height_step"));
    else if (name == "cyl_bend")
        o = cyl_bend(s);
    else if (name == "cyl_step") {
        Volume v = Volume::open(manifest, {"grad_mag"});
        Channel &g = v.channel("grad_mag");
        const size_t n = (size_t)s.H * s.W;
        v.prefetch(g, s.xyz.data(), n);
        std::vector<uint8_t> u8(n), valid(n);
        sample_u8(v, g, s.xyz.data(), n, u8.data());
        for (size_t i = 0; i < n; ++i)
            valid[i] = decode(v, "grad_mag", (float)u8[i]) > 0.0f;
        o = cyl_step(s, valid, num("cyl_shell_width_step"), num("cyl_shell_height_step"));
    } else if (name == "cyl_step_push") {
        auto grid = [&](const std::string &file, int channels) {
            std::vector<int64_t> sh;
            Grid g;
            g.v = load_npy_float(dir + "/" + file, &sh);
            g.H = (int)sh[0];
            g.W = (int)sh[1];
            g.C = channels;
            return g;
        };
        const Grid base = grid("cyl_shell_base_xyz.npy", 3),
                   offsets = grid("cyl_shell_w_offsets.npy", 1),
                   delta = grid("cyl_shell_delta_xyz.npy", 3);
        const bool has_base = num("cyl_shell_index") > 0;
        const Shell fine = shell_geometry(base, offsets, delta, has_base, 4);
        Volume v = Volume::open(manifest, {"grad_mag"});
        Channel &g = v.channel("grad_mag");
        const size_t n = (size_t)fine.H * fine.W;
        v.prefetch(g, fine.xyz.data(), n);
        std::vector<uint8_t> u8(n), valid(n);
        sample_u8(v, g, fine.xyz.data(), n, u8.data());
        for (size_t i = 0; i < n; ++i)
            valid[i] = decode(v, "grad_mag", (float)u8[i]) > 0.0f;
        o = cyl_step_push(
            base, offsets, delta, has_base, valid, num("cyl_shell_width_step"), num("mesh_step"));
        shape = {delta.H, delta.W, 3};
        save_npy(out + "/grad_delta.npy", "<f4", shape, o.grad.data());
    } else if (name == "cyl_normal") {
        auto grid = [&](const std::string &file, int channels) {
            std::vector<int64_t> sh;
            Grid g;
            g.v = load_npy_float(dir + "/" + file, &sh);
            g.H = (int)sh[0];
            g.W = (int)sh[1];
            g.C = channels;
            return g;
        };
        const Grid base = grid("cyl_shell_base_xyz.npy", 3),
                   offsets = grid("cyl_shell_w_offsets.npy", 1),
                   delta = grid("cyl_shell_delta_xyz.npy", 3);
        const bool has_base = num("cyl_shell_index") > 0;
        const Shell fine = shell_geometry(base, offsets, delta, has_base, 4);
        Volume v = Volume::open(manifest, {"grad_mag", "nx", "ny"});
        const size_t n = (size_t)fine.H * fine.W;
        std::vector<float> values[3];
        const char *names[3] = {"grad_mag", "nx", "ny"};
        for (int k = 0; k < 3; ++k) {
            Channel &ch = v.channel(names[k]);
            v.prefetch(ch, fine.xyz.data(), n);
            std::vector<uint8_t> u8(n);
            sample_u8(v, ch, fine.xyz.data(), n, u8.data());
            values[k].resize(n);
            for (size_t i = 0; i < n; ++i)
                values[k][i] = decode(v, names[k], (float)u8[i]);
        }
        std::vector<uint8_t> valid(n);
        for (size_t i = 0; i < n; ++i)
            valid[i] = values[0][i] > 0.0f;
        std::vector<float> target, mask;
        shell_normal_targets(fine, values[1], values[2], valid, target, mask);
        o = cyl_normal(base, offsets, delta, has_base, target, mask);
        shape = {delta.H, delta.W, 3};
        save_npy(out + "/grad_delta.npy", "<f4", shape, o.grad.data());
    } else if (name == "cyl_outside") {
        auto grid = [&](const std::string &file, int channels) {
            std::vector<int64_t> sh;
            Grid g;
            g.v = load_npy_float(dir + "/" + file, &sh);
            g.H = (int)sh[0];
            g.W = (int)sh[1];
            g.C = channels;
            return g;
        };
        const Grid base = grid("cyl_shell_base_xyz.npy", 3),
                   offsets = grid("cyl_shell_w_offsets.npy", 1),
                   delta = grid("cyl_shell_delta_xyz.npy", 3);
        const bool has_base = num("cyl_shell_index") > 0;
        const int factor = std::max(1, (int)num("cyl_outside_sample_factor"));
        const Shell fine = shell_geometry(base, offsets, delta, has_base, factor);
        Volume v = Volume::open(manifest, {"grad_mag"});
        Channel &gm = v.channel("grad_mag");
        const size_t n = (size_t)fine.H * fine.W;
        v.prefetch(gm, fine.xyz.data(), n);
        std::vector<uint8_t> u8(n), valid(n);
        sample_u8(v, gm, fine.xyz.data(), n, u8.data());
        for (size_t i = 0; i < n; ++i)
            valid[i] = decode(v, "grad_mag", (float)u8[i]) > 0.0f;
        const DepthVolume field = load_barrier(dir + "/cyl_outside_volume.npy", c);
        const double model_step =
            num("cyl_outside_model_step") > 0 ? num("cyl_outside_model_step") : num("mesh_step");
        o = cyl_outside(base, offsets, delta, has_base, valid, field, num("cyl_outside_depth_max"),
            model_step, factor);
        shape = {delta.H, delta.W, 3};
        save_npy(out + "/grad_delta.npy", "<f4", shape, o.grad.data());
    } else
        throw std::runtime_error("no cylinder loss " + name);
    std::FILE *f = std::fopen((out + "/value.txt").c_str(), "w");
    std::fprintf(f, "%.17g\n", o.value);
    std::fclose(f);
    if (name != "cyl_step_push" && name != "cyl_normal" && name != "cyl_outside")
        save_npy(out + "/grad_xyz.npy", "<f4", shape, o.grad.data());
    return 0;
}

static int cyl_init(char **argv) {
    const Umbilicus u = load_umbilicus(argv[2]);
    const TubeStart t = tube_start(u, std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5]),
        std::stod(argv[6]), std::stod(argv[7]));
    const std::string out = argv[8];
    const std::vector<int64_t> shape = {t.base.H, t.base.W, 3};
    save_npy(out + "/base.npy", "<f4", shape, t.base.v.data());
    save_npy(out + "/dirs.npy", "<f4", shape, t.dirs.v.data());
    save_npy(out + "/delta.npy", "<f4", shape, t.delta.v.data());
    std::printf("H=%d W=%d h_step=%.1f\n", t.base.H, t.base.W, t.height_step);
    return 0;
}

static Grid load_grid(const std::string &path, int channels) {
    std::vector<int64_t> sh;
    Grid g;
    g.v = load_npy_float(path, &sh);
    g.H = (int)sh[0];
    g.W = (int)sh[1];
    g.C = channels;
    return g;
}

static Grid load_level(const std::string &path) {
    std::vector<int64_t> sh;
    const std::vector<float> raw = load_npy_float(path, &sh);
    Grid g{(int)sh[2], (int)sh[3], 3, std::vector<float>((size_t)sh[2] * sh[3] * 3)};
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < g.H; ++y)
            for (int x = 0; x < g.W; ++x)
                g.at(y, x)[c] = raw[((size_t)c * g.H + y) * g.W + x];
    return g;
}

static void save_level(const std::string &path, const Grid &g) {
    std::vector<float> raw((size_t)g.H * g.W * 3);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < g.H; ++y)
            for (int x = 0; x < g.W; ++x)
                raw[((size_t)c * g.H + y) * g.W + x] = g.at(y, x)[c];
    save_npy(path, "<f4", {3, 1, g.H, g.W}, raw.data());
}

static int cyl_step(const std::string &manifest, const std::string &dir, const std::string &out) {
    const Json c = Json::parse(read_text(dir + "/case.json"));
    auto num = [&](const char *key, double fallback) {
        return c.has(key) && c[key].kind == Json::Number ? c[key].num() : fallback;
    };
    CylinderFit fit(manifest, Json(), {0.0, 0.0, num("cyl_z_center_target", 0.0)}, 1.0);
    for (int k = 0; exists(dir + "/level" + std::to_string(k) + ".npy"); ++k)
        fit.levels.push_back(load_level(dir + "/level" + std::to_string(k) + ".npy"));
    fit.base = load_grid(dir + "/cyl_shell_base_xyz.npy", 3);
    fit.offsets = load_grid(dir + "/cyl_shell_w_offsets.npy", 1);
    fit.index = (int)num("cyl_shell_index", 0.0);
    fit.z_target = num("cyl_z_center_target", 0.0);
    fit.width_step = num("cyl_shell_width_step", 0.0);
    fit.model_step = num("cyl_shell_height_step", 0.0);
    fit.mesh_step = num("mesh_step", 100.0);
    if (exists(dir + "/barrier.npy")) {
        fit.field = load_barrier(dir + "/barrier.npy", c);
        fit.has_field = true;
        fit.outside_model_step = num("cyl_outside_model_step", 0.0);
        fit.outside_factor = std::max(1, (int)num("cyl_outside_sample_factor", 2.0));
    }
    const Json &w = c["weights"];
    auto weight = [&](const char *k) { return w.has(k) ? w[k].num() : 0.0; };
    const CylinderWeights weights{weight("cyl_normal"), weight("cyl_smooth"),
        weight("cyl_z_smooth"), weight("cyl_z_center"), weight("cyl_step"), weight("cyl_step_push"),
        weight("cyl_bend"), weight("cyl_outside")};
    Grid grad;
    std::vector<uint8_t> valid;
    const double loss = fit.evaluate(weights, grad, valid);
    Adam adam;
    adam.update(fit.levels, integrate_pyramid_backward(fit.levels, grad),
        std::vector<double>(fit.levels.size(), c["lr"].num()));
    for (size_t k = 0; k < fit.levels.size(); ++k)
        save_level(out + "/level" + std::to_string(k) + ".npy", fit.levels[k]);
    std::printf("loss %.9g\n", loss);
    return 0;
}

static int cyl_shell(const std::string &manifest, const std::string &shell_path, int index,
    int width, const std::string &out) {
    std::vector<int64_t> shape;
    Shell s;
    s.xyz = load_npy_float(shell_path, &shape);
    s.H = (int)shape[0];
    s.W = (int)shape[1];
    CylinderFit fit(manifest, Json(), {0.0, 0.0, 0.0}, 1.0);
    fit.begin(s, index);
    if (width != s.W)
        fit.resample_width(width, fit.width_target_step);
    const Shell now = fit.current();
    save_npy(out + "/xyz.npy", "<f4", {now.H, now.W, 3}, now.xyz.data());
    save_npy(out + "/base.npy", "<f4", {fit.base.H, fit.base.W, 3}, fit.base.v.data());
    return 0;
}

static int barrier(const std::string &shell_path, const std::string &out, const std::string &mode) {
    std::vector<int64_t> shape;
    Shell s;
    s.xyz = load_npy_float(shell_path, &shape);
    s.H = (int)shape[0];
    s.W = (int)shape[1];
    const DepthVolume field = previous_shell_barrier(s, mode != "outside");
    save_npy(out + "/volume.npy", "|u1", {1, field.Z, field.Y, field.X}, field.v.data());
    std::FILE *f = std::fopen((out + "/barrier.json").c_str(), "w");
    std::fprintf(f,
        "{\"origin\": [%.17g, %.17g, %.17g], \"spacing\": [%.17g, %.17g, %.17g], \"depth_max\": "
        "%.17g}\n",
        field.origin[0], field.origin[1], field.origin[2], field.spacing[0], field.spacing[1],
        field.spacing[2], kBarrierDepthMax);
    std::fclose(f);
    return 0;
}

static int cyl_fit(const std::string &manifest, const std::string &config_path,
    const std::string &seed_text, double model_h, const std::string &out) {
    std::array<double, 3> seed{};
    if (std::sscanf(seed_text.c_str(), "%lf %lf %lf", &seed[0], &seed[1], &seed[2]) != 3)
        throw std::runtime_error("seed must be \"X Y Z\"");
    CylinderFit fit(manifest, Json::parse(read_text(config_path)), seed, model_h);
    fit.log = stdout;
    fit.run();
    for (size_t n = 0; n < fit.completed.size(); ++n) {
        const Shell &s = fit.completed[n];
        save_npy(out + "/shell" + std::to_string(n) + ".npy", "<f4", {s.H, s.W, 3}, s.xyz.data());
    }
    int shells = 0, H = 0, W = 0;
    const std::vector<float> flat = fit.mesh_flat(shells, H, W);
    save_npy(out + "/mesh_flat.npy", "<f4", {3, shells, H, W}, flat.data());
    std::FILE *f = std::fopen((out + "/cylinder.json").c_str(), "w");
    std::fprintf(f,
        "{\"shells\": %d, \"height\": %d, \"width\": %d, \"z_center\": %.17g, \"model_h\": %.17g, "
        "\"passes\": [",
        shells, H, W, fit.z_target, fit.model_h);
    for (size_t k = 0; k < fit.passes.size(); ++k) {
        const PassReport &r = fit.passes[k];
        std::fprintf(f,
            "%s\n  {\"label\": \"%s\", \"steps\": %d, \"width\": %d, \"first_loss\": %.9g, "
            "\"last_loss\": %.9g, "
            "\"invalid_frac\": %.6g}",
            k ? "," : "", r.label.c_str(), r.steps, r.width, r.first_loss, r.last_loss,
            r.invalid_frac);
    }
    std::fprintf(f, "\n]}\n");
    std::fclose(f);
    return 0;
}

static std::string station_json(const Station &s) {
    std::string out = "{\"h_frac\": [";
    char buf[64];
    for (size_t d = 0; d < s.h_frac.size(); ++d) {
        std::snprintf(buf, sizeof buf, "%s%.17g", d ? ", " : "", s.h_frac[d]);
        out += buf;
    }
    out += "], \"w_frac\": [";
    for (size_t d = 0; d < s.w_frac.size(); ++d) {
        std::snprintf(buf, sizeof buf, "%s%.17g", d ? ", " : "", s.w_frac[d]);
        out += buf;
    }
    out += "], \"ray\": [";
    for (size_t d = 0; d < s.ray.size(); ++d) {
        if (s.has_ray[d])
            std::snprintf(buf, sizeof buf, "%s[%.9g, %.9g, %.9g]", d ? ", " : "", s.ray[d][0],
                s.ray[d][1], s.ray[d][2]);
        else
            std::snprintf(buf, sizeof buf, "%snull", d ? ", " : "");
        out += buf;
    }
    return out + "]}";
}

static Mesh load_mesh(const std::string &path) {
    std::vector<int64_t> shape;
    Mesh m;
    m.xyz = load_npy_float(path, &shape);
    m.D = (int)shape[0];
    m.H = (int)shape[1];
    m.W = (int)shape[2];
    return m;
}

static int mesh_loss(const std::string &name, const std::string &manifest, const std::string &dir,
    const std::string &out) {
    const Mesh m = load_mesh(dir + "/xyz_lr.npy");
    const Json c = Json::parse(read_text(dir + "/case.json"));
    const double step = c["mesh_step"].num();
    std::vector<std::pair<std::string, LossOut>> parts;
    if (name == "normal") {
        Volume v = Volume::open(manifest, {"grad_mag", "nx", "ny"});
        const size_t n = m.points();
        std::vector<float> values[3];
        const char *names[3] = {"grad_mag", "nx", "ny"};
        for (int k = 0; k < 3; ++k) {
            Channel &ch = v.channel(names[k]);
            v.prefetch(ch, m.xyz.data(), n);
            std::vector<uint8_t> u8(n);
            sample_u8(v, ch, m.xyz.data(), n, u8.data());
            values[k].resize(n);
            for (size_t i = 0; i < n; ++i)
                values[k][i] = decode(v, names[k], (float)u8[i]);
        }
        std::vector<float> mask(n);
        for (size_t i = 0; i < n; ++i)
            mask[i] = values[0][i] > 0.0f ? 1.0f : 0.0f;
        parts.push_back({"normal", mesh_normal(m, normal_targets(values[1], values[2]), mask)});
    } else if (name == "step")
        parts.push_back({"step", mesh_step(m, step)});
    else if (name == "smooth")
        parts.push_back({"smooth", mesh_smooth(m, step)});
    else if (name == "bend")
        parts.push_back({"bend", mesh_bend(m)});
    else if (name == "station") {
        const Json &b = c["station_before"];
        Station st;
        st.reset({(float)b["seed"][0].num(), (float)b["seed"][1].num(), (float)b["seed"][2].num()},
            m.H, m.W, m.D);
        for (int d = 0; d < m.D; ++d) {
            st.h_frac[(size_t)d] = b["h_frac"][(size_t)d].num();
            st.w_frac[(size_t)d] = b["w_frac"][(size_t)d].num();
            st.initialized[(size_t)d] = b["initialized"][(size_t)d].truth();
            const Json &r = b["ray"][(size_t)d];
            if (r.kind == Json::Array) {
                st.has_ray[(size_t)d] = 1;
                st.ray[(size_t)d] = {(float)r[0].num(), (float)r[1].num(), (float)r[2].num()};
            }
        }
        int anchor = std::max(0, (m.D - 1) / 2);
        const Json &dw = c["depth_windings"];
        if ((int)dw.size() == m.D)
            for (int d = 0; d < m.D; ++d)
                if ((int)dw[(size_t)d].num() == 0) {
                    anchor = d;
                    break;
                }
        LossOut n, t;
        station_loss(m, st, step, anchor, n, t);
        parts.push_back({"station_n", n});
        parts.push_back({"station_t", t});
        std::ofstream(out + "/station.json") << station_json(st) << "\n";
    } else
        throw std::runtime_error("no mesh loss " + name);
    for (const auto &[part, o] : parts) {
        std::FILE *f = std::fopen((out + "/value_" + part + ".txt").c_str(), "w");
        std::fprintf(f, "%.17g\n", o.value);
        std::fclose(f);
        save_npy(out + "/grad_" + part + ".npy", "<f4", {m.D, m.H, m.W, 3}, o.grad.data());
    }
    return 0;
}

static void export_mesh(const MeshFit &fit, const std::string &out, const std::string &config_text,
    const Json &manifest) {
    const Mesh m = fit.current();
    const std::vector<float> flat = fit.flat();
    save_npy(out + "/mesh_flat.npy", "<f4", {3, m.D, m.H, m.W}, flat.data());
    const std::string tifxyz = out + "/tifxyz";
    ::mkdir(tifxyz.c_str(), 0755);
    const Json &shape = manifest["base_shape_zyx"];
    char dims[128];
    std::snprintf(dims, sizeof dims, "[%d, %d, %d]", (int)shape[0].num(), (int)shape[1].num(),
        (int)shape[2].num());
    const std::string extra = std::string(",\n  \"base_shape_zyx\": ") + dims +
        ",\n  \"lasagna_base_shape_zyx\": " + dims +
        ",\n  \"created_by\": \"HerculeX ports/lasagna\",\n  \"fit_config\": " + config_text;
    for (int d = 0; d < m.D; ++d) {
        char name[64];
        std::snprintf(name, sizeof name, "/winding_%04d.tifxyz", d);
        write_tifxyz(
            tifxyz + name, m, d, (double)fit.depth_windings[(size_t)d], fit.mesh_step, extra);
    }
    std::FILE *f = std::fopen((out + "/mesh.json").c_str(), "w");
    std::fprintf(f,
        "{\"depth\": %d, \"height\": %d, \"width\": %d, \"mesh_step\": %.17g, \"stages\": [", m.D,
        m.H, m.W, fit.mesh_step);
    for (size_t k = 0; k < fit.stages.size(); ++k)
        std::fprintf(f,
            "%s\n  {\"name\": \"%s\", \"steps\": %d, \"first_loss\": %.9g, \"last_loss\": %.9g}",
            k ? "," : "", fit.stages[k].name.c_str(), fit.stages[k].steps, fit.stages[k].first_loss,
            fit.stages[k].last_loss);
    std::fprintf(f, "\n]}\n");
    std::fclose(f);
}

static int mesh_fit(const std::string &manifest, const std::string &config_path,
    const std::string &flat_path, double step, const std::string &out) {
    std::vector<int64_t> shape;
    const std::vector<float> flat = load_npy_float(flat_path, &shape);
    const int D = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
    std::vector<int> windings(D);
    for (int d = 0; d < D; ++d)
        windings[(size_t)d] = d;
    const std::string text = read_text(config_path);
    MeshFit fit(manifest, Json::parse(text), flat, D, H, W, step, windings);
    fit.log = stdout;
    fit.run();
    export_mesh(fit, out, text, Json::parse(read_text(manifest)));
    return 0;
}

static int full_fit(const std::string &manifest, const std::string &cyl_path,
    const std::string &mesh_path, const std::string &seed_text, double model_h,
    const std::string &out) {
    std::array<double, 3> seed{};
    if (std::sscanf(seed_text.c_str(), "%lf %lf %lf", &seed[0], &seed[1], &seed[2]) != 3)
        throw std::runtime_error("seed must be \"X Y Z\"");
    const Json cyl_config = Json::parse(read_text(cyl_path));
    CylinderFit cyl(manifest, cyl_config, seed, model_h);
    cyl.log = stdout;
    cyl.run();
    int shells = 0, H = 0, W = 0;
    const std::vector<float> flat = cyl.mesh_flat(shells, H, W);
    save_npy(out + "/cylinder_mesh_flat.npy", "<f4", {3, shells, H, W}, flat.data());
    std::vector<int> windings(shells);
    for (int d = 0; d < shells; ++d)
        windings[(size_t)d] = d;
    const std::string text = read_text(mesh_path);
    MeshFit fit(manifest, Json::parse(text), flat, shells, H, W, cyl.mesh_step, windings);
    fit.log = stdout;
    fit.run();
    export_mesh(fit, out, text, Json::parse(read_text(manifest)));
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "sample")
            return sample(argv[2], argv[3], argv[4], argv[5]);
        if (argc == 6 && std::string(argv[1]) == "cyl-loss")
            return cyl_loss(argv[2], argv[3], argv[4], argv[5]);
        if (argc == 9 && std::string(argv[1]) == "cyl-init")
            return cyl_init(argv);
        if (argc == 5 && std::string(argv[1]) == "cyl-step")
            return cyl_step(argv[2], argv[3], argv[4]);
        if (argc == 5 && std::string(argv[1]) == "barrier")
            return barrier(argv[2], argv[3], argv[4]);
        if (argc == 7 && std::string(argv[1]) == "cyl-shell")
            return cyl_shell(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), argv[6]);
        if (argc == 6 && std::string(argv[1]) == "mesh-loss")
            return mesh_loss(argv[2], argv[3], argv[4], argv[5]);
        if (argc == 7 && std::string(argv[1]) == "mesh-fit")
            return mesh_fit(argv[2], argv[3], argv[4], std::stod(argv[5]), argv[6]);
        if (argc == 8 && std::string(argv[1]) == "fit")
            return full_fit(argv[2], argv[3], argv[4], argv[5], std::stod(argv[6]), argv[7]);
        if (argc == 7 && std::string(argv[1]) == "cyl-fit")
            return cyl_fit(argv[2], argv[3], argv[4], std::stod(argv[5]), argv[6]);
        std::fprintf(stderr,
            "lasagna_fit sample MANIFEST POINTS.npy WEIGHTS_DIR OUT_DIR\n"
            "lasagna_fit cyl-loss NAME MANIFEST CASE_DIR OUT_DIR\n"
            "lasagna_fit cyl-init UMBILICUS Z_CENTER MODEL_H Z_STEP RADIUS WIDTH_STEP OUT_DIR\n"
            "lasagna_fit cyl-step MANIFEST CASE_DIR OUT_DIR\n"
            "lasagna_fit cyl-shell MANIFEST SHELL.npy INDEX WIDTH OUT_DIR\n"
            "lasagna_fit barrier SHELL.npy OUT_DIR inside|outside\n"
            "lasagna_fit cyl-fit MANIFEST STAGES.json \"X Y Z\" MODEL_H OUT_DIR\n"
            "lasagna_fit mesh-loss NAME MANIFEST CASE_DIR OUT_DIR\n"
            "lasagna_fit mesh-fit MANIFEST STAGES.json MESH_FLAT.npy MESH_STEP OUT_DIR\n"
            "lasagna_fit fit MANIFEST CYL_STAGES.json MESH_STAGES.json \"X Y Z\" MODEL_H "
            "OUT_DIR\n");
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "lasagna_fit: %s\n", e.what());
        return 1;
    }
}
