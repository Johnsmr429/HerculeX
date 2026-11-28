#include "fit.h"

#include "barrier.h"
#include "io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace la {

namespace {

constexpr double kInitialRadius = 2000.0;
constexpr int kScales = 5;

const Json &empty_json() {
    static const Json empty;
    return empty;
}

double number(const Json &j, const char *key, double fallback) {
    return j.has(key) && j[key].kind == Json::Number ? j[key].num() : fallback;
}

const Json &args_of(const Json &stage) {
    return stage.has("args") && stage["args"].kind == Json::Object ? stage["args"] : empty_json();
}

bool truthy(const Json &j) {
    if (j.kind == Json::Bool)
        return j.boolean;
    if (j.kind == Json::Number)
        return j.number != 0.0;
    if (j.kind == Json::String) {
        std::string s;
        for (char c : j.text)
            if (!std::isspace((unsigned char)c))
                s += (char)std::tolower((unsigned char)c);
        return s != "" && s != "0" && s != "false" && s != "no" && s != "off";
    }
    return false;
}

CylinderWeights stage_weights(const Json &config, const Json &stage) {
    const Json &base = config["base"];
    auto weight = [&](const char *term) {
        const double b = number(base, term, 0.0);
        double v = b;
        if (stage.has("default_mul") && stage["default_mul"].kind == Json::Number)
            v = b * stage["default_mul"].num();
        if (stage.has("w_fac")) {
            const Json &f = stage["w_fac"];
            if (f.kind == Json::Number)
                v = b * f.num();
            else if (f.kind == Json::Object && f.has(term) && f[term].kind == Json::Number)
                v = b * f[term].num();
        }
        return v;
    };
    CylinderWeights w;
    w.normal = weight("cyl_normal");
    w.smooth = weight("cyl_smooth");
    w.z_smooth = weight("cyl_z_smooth");
    w.z_center = weight("cyl_z_center");
    w.step = weight("cyl_step");
    w.step_push = stage["name"].str() == "cyl_grow" ? weight("cyl_step_push") : 0.0;
    w.bend = weight("cyl_bend");
    w.outside = weight("cyl_outside");
    return w;
}

std::vector<double> stage_lrs(const Json &stage, size_t levels) {
    std::vector<double> out(levels, 1.0e-3);
    const int frozen = std::max(0, (int)number(stage, "min_scaledown", 0.0));
    for (size_t i = 0; i < levels; ++i) {
        if (stage.has("lr")) {
            const Json &lr = stage["lr"];
            if (lr.kind == Json::Number)
                out[i] = lr.num();
            else if (lr.kind == Json::Array) {
                const long n = (long)lr.items.size(), idx = -1 - (long)i;
                out[i] = n == 0
                    ? 0.0
                    : (-n <= idx ? lr.items[(size_t)(n + idx)].num() : lr.items[0].num());
            }
        }
        if ((int)i < frozen)
            out[i] = 0.0;
    }
    return out;
}

int grow_direction(const Json &args) {
    if (!args.has("cyl_grow_direction"))
        return 1;
    const Json &d = args["cyl_grow_direction"];
    if (d.kind == Json::Number && (d.num() == 1.0 || d.num() == -1.0))
        return (int)d.num();
    if (d.kind == Json::String) {
        std::string s;
        for (char c : d.text)
            if (!std::isspace((unsigned char)c))
                s += (char)std::tolower((unsigned char)c);
        for (const char *v : {"outward", "outwards", "grow", "expand", "+", "+1", "1"})
            if (s == v)
                return 1;
        for (const char *v : {"inward", "inwards", "shrink", "contract", "-", "-1"})
            if (s == v)
                return -1;
    }
    throw std::runtime_error("cyl_grow_direction must be 'outward' or 'inward'");
}

bool cylinder_stage(const Json &stage) {
    if (!stage.has("params"))
        return false;
    for (const Json &p : stage["params"].items)
        if (p.kind == Json::String && p.text == "cyl_params")
            return true;
    return false;
}

struct WidthTarget {
    int count;
    double circumference, step;
};

WidthTarget grow_target(int ref_w, double ref_circ, int shell, double factor, int direction) {
    ref_w = std::max(3, ref_w);
    ref_circ = std::max(1.0e-6, ref_circ);
    double scale = std::pow(std::max(1.0, factor), (double)std::max(0, shell));
    if (direction < 0)
        scale = 1.0 / scale;
    const double circ = ref_circ * scale;
    const int w = std::max(3, (int)std::floor((double)ref_w * scale + 0.5));
    return {w, circ, circ / (double)w};
}

Grid umbilicus_base(const Umbilicus &u, const std::vector<float> &z, int W) {
    Grid b{(int)z.size(), W, 3, std::vector<float>(z.size() * (size_t)W * 3)};
    for (int h = 0; h < b.H; ++h) {
        float ux, uy;
        u.xy(z[(size_t)h], ux, uy);
        for (int w = 0; w < W; ++w) {
            float *p = b.at(h, w);
            p[0] = ux;
            p[1] = uy;
            p[2] = z[(size_t)h];
        }
    }
    return b;
}

Shell resample_shell_width(const Shell &s, int target_w) {
    target_w = std::max(3, target_w);
    if (s.W == target_w)
        return s;
    Shell out{s.H, target_w, std::vector<float>((size_t)s.H * target_w * 3)};
    const float ratio = (float)((double)s.W / (double)target_w);
    for (int k = 0; k < target_w; ++k) {
        const float phase = (float)k * ratio;
        const long i0 = (long)std::floor(phase);
        const float frac = phase - (float)i0;
        const int a = (int)(((i0 % s.W) + s.W) % s.W), b = (a + 1) % s.W;
        for (int h = 0; h < s.H; ++h)
            for (int c = 0; c < 3; ++c) {
                const float p0 = s.at(h, a)[c], p1 = s.at(h, b)[c];
                out.xyz[((size_t)h * target_w + k) * 3 + c] = p0 + frac * (p1 - p0);
            }
    }
    return out;
}

}

CylinderFit::CylinderFit(const std::string &manifest, const Json &config_,
    std::array<double, 3> seed_, double model_h_req)
    : volume(Volume::open(manifest, {"grad_mag", "nx", "ny"})), config(config_), seed(seed_) {
    const Json m = Json::parse(read_text(manifest));
    const std::string dir = manifest.substr(0, manifest.find_last_of('/') + 1);
    umbilicus = load_umbilicus(dir + m["umbilicus_json"].str());
    const double to_base = number(m, "source_to_base", 1.0);
    for (int a = 0; a < 3; ++a)
        extent[(size_t)a] = (double)(long)(m["base_shape_zyx"][(size_t)(2 - a)].num() * to_base);
    const double half = 2.0 * std::max(1.0, model_h_req);
    double z0 = seed[2] - half, z1 = seed[2] + half;
    z0 = std::max(0.0, z0);
    z1 = std::min(std::max(0.0, extent[2] - 1.0), z1);
    if (z1 <= z0)
        throw std::runtime_error("invalid cylinder seed z extent");
    z_target = 0.5 * (z0 + z1);
    model_h = z1 - z0;
    mesh_step = number(args_of(config), "mesh-step", 100.0);
}

Shell CylinderFit::current() const {
    const Grid delta = integrate_pyramid(levels);
    const Grid conn = index > 0 ? interp_width_at_offsets(base, offsets, 1.0f) : base;
    Shell s{delta.H, delta.W, std::vector<float>(delta.v.size())};
    for (size_t i = 0; i < s.xyz.size(); ++i)
        s.xyz[i] = conn.v[i] + delta.v[i];
    return s;
}

void CylinderFit::start_tube() {
    const TubeStart t =
        tube_start(umbilicus, z_target, model_h, model_step, kInitialRadius, width_target_step);
    base = t.base;
    levels = build_pyramid(t.delta, kScales);
    offsets = Grid{t.base.H, t.base.W, 1, std::vector<float>((size_t)t.base.H * t.base.W, 0.0f)};
    index = 0;
    width_step = width_target_step;
}

void CylinderFit::begin(const Shell &from, int idx) {
    std::vector<float> z((size_t)from.H);
    for (int h = 0; h < from.H; ++h)
        z[(size_t)h] = from.at(h, 0)[2];
    base = umbilicus_base(umbilicus, z, from.W);
    Grid delta{from.H, from.W, 3, std::vector<float>(from.xyz.size())};
    for (size_t i = 0; i < delta.v.size(); ++i)
        delta.v[i] = from.xyz[i] - base.v[i];
    levels = build_pyramid(delta, kScales);
    offsets = Grid{from.H, from.W, 1, std::vector<float>((size_t)from.H * from.W, 0.0f)};
    index = idx;
    width_step = width_target_step;
}

void CylinderFit::resample_width(int target_w, double target_step) {
    begin(resample_shell_width(current(), target_w), index);
    width_target_step = width_step = std::max(1.0e-6, target_step);
}

double CylinderFit::evaluate(const CylinderWeights &w, Grid &grad, std::vector<uint8_t> &valid) {
    const Grid delta = integrate_pyramid(levels);
    const bool has_base = index > 0;
    const Grid conn = has_base ? interp_width_at_offsets(base, offsets, 1.0f) : base;
    Shell s{delta.H, delta.W, std::vector<float>(delta.v.size())};
    for (size_t i = 0; i < s.xyz.size(); ++i)
        s.xyz[i] = conn.v[i] + delta.v[i];
    auto sample = [&](const Shell &at, const char *name) {
        Channel &ch = volume.channel(name);
        const size_t n = (size_t)at.H * at.W;
        volume.prefetch(ch, at.xyz.data(), n);
        std::vector<uint8_t> u8(n);
        sample_u8(volume, ch, at.xyz.data(), n, u8.data());
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i)
            out[i] = decode(volume, name, (float)u8[i]);
        return out;
    };
    auto mask = [&](const Shell &at) {
        const std::vector<float> g = sample(at, "grad_mag");
        std::vector<uint8_t> m(g.size());
        for (size_t i = 0; i < g.size(); ++i)
            m[i] = g[i] > 0.0f;
        return m;
    };
    valid = mask(s);
    grad = Grid{delta.H, delta.W, 3, std::vector<float>(delta.v.size(), 0.0f)};
    double total = 0;
    auto add = [&](double weight, const LossOut &o) {
        if (weight == 0.0)
            return;
        total += weight * o.value;
        for (size_t i = 0; i < grad.v.size(); ++i)
            grad.v[i] += (float)weight * o.grad[i];
    };
    Shell fine;
    std::vector<uint8_t> fine_valid;
    if (w.normal != 0.0 || w.step_push != 0.0) {
        fine = shell_geometry(base, offsets, delta, has_base, 4);
        fine_valid = mask(fine);
    }
    if (w.normal != 0.0) {
        std::vector<float> target, weights;
        shell_normal_targets(
            fine, sample(fine, "nx"), sample(fine, "ny"), fine_valid, target, weights);
        add(w.normal, cyl_normal(base, offsets, delta, has_base, target, weights));
    }
    add(w.smooth, cyl_smooth(s));
    add(w.z_smooth, cyl_z_smooth(s));
    add(w.z_center, cyl_z_center(s, z_target, model_step));
    if (w.step_push != 0.0)
        add(w.step_push,
            cyl_step_push(base, offsets, delta, has_base, fine_valid, width_step, mesh_step));
    add(w.step, cyl_step(s, valid, width_step, model_step));
    add(w.bend, cyl_bend(s));
    if (w.outside != 0.0 && has_field) {
        const Shell fine = shell_geometry(base, offsets, delta, has_base, outside_factor);
        add(w.outside,
            cyl_outside(base, offsets, delta, has_base, mask(fine), field, kBarrierDepthMax,
                outside_model_step, outside_factor));
    }
    return total;
}

PassReport CylinderFit::pass(const std::string &label, const CylinderWeights &w, const Json &stage,
    double wstep_start, double wstep_end, int steps) {
    const Json &args = args_of(stage);
    outside_factor = std::max(1, (int)number(args, "cyl_outside_sample_factor", 2.0));
    width_step = wstep_start;
    const std::vector<double> lrs = stage_lrs(stage, levels.size());
    const int warmup =
        std::max(0, (int)number(args, "lr_warmup_steps", number(args, "warmup_steps", 0.0)));
    Adam adam;
    PassReport r{label, steps, levels[0].W};
    Grid grad;
    std::vector<uint8_t> valid;
    r.first_loss = r.last_loss = evaluate(w, grad, valid);
    for (int step = 0; step < steps; ++step) {
        width_step = wstep_start + ((double)(step + 1) / (double)steps) * (wstep_end - wstep_start);
        r.last_loss = evaluate(w, grad, valid);
        const std::vector<Grid> grads = integrate_pyramid_backward(levels, grad);
        std::vector<double> lr = lrs;
        if (warmup > 0)
            for (double &x : lr)
                x *= std::min(1.0, std::max(0.0, (double)(step + 1) / (double)warmup));
        adam.update(levels, grads, lr);
    }
    const int H = levels[0].H, W = levels[0].W;
    size_t bad = 0;
    for (int h = 0; h < H; ++h)
        for (int x = 0; x < W; ++x)
            if (!(valid[(size_t)h * W + x] && valid[(size_t)h * W + (x + 1) % W]))
                ++bad;
    r.invalid_frac = (double)bad / (double)std::max(1, H * W);
    if (log)
        std::fprintf(log, "%s: %d steps width %d loss %.6g -> %.6g ifrac %.4f\n", label.c_str(),
            steps, W, r.first_loss, r.last_loss, r.invalid_frac);
    passes.push_back(r);
    return r;
}

void CylinderFit::run() {
    const Json &stages = config["stages"];
    int max_shells = 16;
    bool output_all = false, found_max = false;
    for (const Json &stage : stages.items) {
        const Json &args = args_of(stage);
        for (const char *key :
            {"cyl_max_shells", "cyl_shell_max_shells", "cyl_shell_search_max_shells"})
            if (!found_max && args.has(key) && args[key].kind == Json::Number) {
                max_shells = (int)args[key].num();
                found_max = true;
            }
        for (const char *key : {"cyl_output_all_shells", "cyl_shell_output_all"})
            if (args.has(key) && truthy(args[key]))
                output_all = true;
    }
    if (max_shells <= 0)
        throw std::runtime_error("cylinder max shells must be > 0");
    bool done = false, aborted = false;
    int ref_w = 0;
    double ref_circ = 0;
    auto complete = [&] {
        if ((int)completed.size() > index)
            completed[(size_t)index] = current();
        else
            completed.push_back(current());
    };
    auto collapse = [&] {
        if (output_all || completed.empty())
            return;
        completed = {completed.back()};
        index = 0;
    };
    for (size_t si = 0; si < stages.items.size() && !aborted; ++si) {
        const Json &stage = stages[si];
        const std::string role = stage["name"].str();
        const int steps = std::max(0, (int)number(stage, "steps", 0.0));
        if (!cylinder_stage(stage)) {
            if (steps > 0)
                throw std::runtime_error("stage " + role + " is not ported");
            continue;
        }
        const Json &args = args_of(stage);
        const double previous_step = width_target_step;
        if (args.has("model-step") && args["model-step"].kind == Json::Number)
            width_target_step = args["model-step"].num();
        const double base_step = width_target_step;
        if (base_step > 0.0)
            model_step = base_step;
        if (role == "cyl_init") {
            if (done)
                continue;
            start_tube();
            ref_w = base.W;
            ref_circ = 2.0 * M_PI * kInitialRadius;
            has_field = false;
            outside_model_step = base_step;
            pass("cyl_init", stage_weights(config, stage), stage, base_step, base_step, steps);
            complete();
            continue;
        }
        if (role == "cyl_grow") {
            if (done)
                continue;
            if (completed.empty())
                throw std::runtime_error("cyl_grow needs a completed shell");
            const int direction = grow_direction(args);
            double factor = 1.5;
            for (const char *key : {"cyl_shell_growth_factor", "grow_factor", "cyl_grow_factor"})
                if (args.has(key) && args[key].kind == Json::Number)
                    factor = std::max(1.0, args[key].num());
            const Json *refine =
                si + 1 < stages.items.size() && stages[si + 1]["name"].str() == "cyl_grow_refine"
                ? &stages[si + 1]
                : nullptr;
            const CylinderWeights grow_w = stage_weights(config, stage);
            for (;;) {
                const int shell = (int)completed.size();
                if (shell >= max_shells)
                    break;
                const WidthTarget prev = grow_target(ref_w, ref_circ, shell - 1, factor, direction);
                const WidthTarget grow = grow_target(ref_w, ref_circ, shell, factor, direction);
                if (grow_w.outside > 0.0) {
                    field = previous_shell_barrier(completed.back(), direction > 0,
                        number(args, "cyl_outside_grid_step", 64.0));
                    has_field = true;
                }
                begin(completed[(size_t)shell - 1], shell);
                outside_model_step = base_step;
                pass("cyl_grow_shell" + std::to_string(shell + 1), grow_w, stage, prev.step,
                    grow.step, steps);
                if (steps > 0)
                    resample_width(grow.count, grow.step);
                complete();
                if (refine && number(*refine, "steps", 0.0) > 0) {
                    begin(completed.back(), (int)completed.size() - 1);
                    const double refine_step = width_target_step;
                    outside_model_step = refine_step;
                    const PassReport r = pass("cyl_grow_refine_shell" + std::to_string(shell + 1),
                        stage_weights(config, *refine), *refine, refine_step, refine_step,
                        (int)number(*refine, "steps", 0.0));
                    complete();
                    const Json &rargs = args_of(*refine);
                    double max_ifrac = 0.5;
                    bool limited = true;
                    for (const char *key : {"cyl_refine_max_ifrac", "cyl_grow_refine_max_ifrac"})
                        if (rargs.has(key)) {
                            limited = rargs[key].kind == Json::Number;
                            if (limited)
                                max_ifrac = rargs[key].num();
                            break;
                        }
                    if (limited && std::isfinite(r.invalid_frac) && r.invalid_frac > max_ifrac) {
                        if (log)
                            std::fprintf(
                                log, "stop: ifrac %.4f exceeds %.4f\n", r.invalid_frac, max_ifrac);
                        done = aborted = true;
                    }
                    width_target_step = base_step;
                }
                if (aborted)
                    break;
            }
            collapse();
            done = true;
            continue;
        }
        if (role == "cyl_grow_refine")
            continue;
        if (completed.empty())
            throw std::runtime_error("stage " + role + " needs a completed shell");
        if (std::fabs(base_step - (previous_step > 0.0 ? previous_step : base_step)) > 1.0e-6)
            throw std::runtime_error(
                "stage " + role + " changes the model step, which is not ported");
        begin(completed.back(), (int)completed.size() - 1);
        CylinderWeights w = stage_weights(config, stage);
        w.step_push = 0.0;
        outside_model_step = base_step;
        pass(role, w, stage, base_step, base_step, steps);
        complete();
        collapse();
    }
}

std::vector<float> CylinderFit::mesh_flat(int &shells, int &H, int &W) const {
    if (completed.empty())
        throw std::runtime_error("no completed cylinder shells");
    int max_w = 0;
    for (const Shell &s : completed)
        max_w = std::max(max_w, s.W);
    shells = (int)completed.size();
    H = completed[0].H;
    W = max_w + 1;
    std::vector<float> out((size_t)3 * shells * H * W);
    for (int n = 0; n < shells; ++n) {
        const Shell s = resample_shell_width(completed[(size_t)n], max_w);
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    out[(((size_t)c * shells + n) * H + h) * W + w] = s.at(h, w % max_w)[c];
    }
    return out;
}

}
