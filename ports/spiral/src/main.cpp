#include "export.h"
#include "gpu.h"
#include "io.h"
#include "json.h"
#include "losses.h"
#include "model.h"
#include "optim.h"
#include "patches.h"
#include "tracks.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace sp;

static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

template <class V> static void load_into(V &target, const std::string &path) {
    std::vector<float> v = load_npy_float(path);
    if (v.size() != target.size())
        throw std::runtime_error(path + ": " + std::to_string(v.size()) + " values, model has " +
            std::to_string(target.size()));
    target.assign(v.begin(), v.end());
}

static void save_grads(
    const std::string &dir, const std::string &name, const Model &m, const Grads &g) {
    double dr;
    std::vector<double> linear, gap;
    Field low, high;
    finish_gradients(m, g, dr, linear, gap, low, high);
    if (high.empty())
        high = g.field;
    std::vector<float> lf(linear.begin(), linear.end()), gf(gap.begin(), gap.end());
    float d = (float)dr;
    save_npy(dir + "/" + name + "_grad_dr_logit.npy", "<f4", {}, &d);
    save_npy(dir + "/" + name + "_grad_linear.npy", "<f4", {(int64_t)m.linear_z, 2, 2}, lf.data());
    save_npy(dir + "/" + name + "_grad_gap.npy", "<f4", {m.gap_num_z, m.gap_width}, gf.data());
    save_npy(dir + "/" + name + "_grad_low.npy", "<f4", {m.s.stages, 3, m.lr[0], m.lr[1], m.lr[2]},
        low.data());
    save_npy(dir + "/" + name + "_grad_high.npy", "<f4", {m.s.stages, 3, m.hr[0], m.hr[1], m.hr[2]},
        high.data());
}

static std::string path_in(const std::string &dataset, const std::string &rel) {
    return rel.size() && rel[0] == '/' ? rel : dataset + "/" + rel;
}

static std::string patches_dir(const std::string &dataset, const Json &spec) {
    const Json &paths = spec["paths"];
    if (paths.has("verified_patches"))
        return path_in(dataset, paths["verified_patches"].str());
    return exists(dataset + "/verified_patches") ? dataset + "/verified_patches" : "";
}

static Tracks prepared_input(const Tracks &tracks, const PatchSet &patches, const PatchSettings &ps,
    CrossingCsr &csr, const std::string &tracks_path) {
    const bool sidecar = exists(tracks_path + ".crossings.npz");
    CrossingCsr full;
    if (sidecar)
        full = sidecar_crossings(Npz(tracks_path + ".crossings.npz"), tracks);
    if (patches.trusted.empty()) {
        csr = sidecar ? std::move(full) : find_crossings(tracks);
        return tracks;
    }
    Clipping clipping;
    Tracks clipped = clip_tracks(tracks, patches.trusted, ps.exclusion_radius, clipping);
    std::printf(
        "track radius loss: %zu/%zu tracks survive exclusion (radius %.1f); %lld points retained\n",
        clipped.size(), tracks.size(), ps.exclusion_radius, (long long)clipped.offsets.back());
    csr = sidecar ? clip_crossings(full, tracks, clipped, clipping) : find_crossings(clipped);
    std::printf("track crossings: %s after point exclusion (%zu directed crossings)\n",
        sidecar ? "remapped the sidecar" : "rebuilt", csr.partners.size());
    return clipped;
}

static void patch_replay(
    const std::string &dir, const std::string &dataset, const Model &m, const Shared &sh) {
    const std::string out = dir + "/port";
    const Json spec = Json::parse(read_text(dataset + "/spiral-scroll.json"));
    PatchSettings ps;
    PatchSet set = load_patches(patches_dir(dataset, spec), m.s.z_begin, m.s.z_end, ps);
    for (size_t k = 0; k < set.patches.size(); ++k) {
        const Patch &p = set.patches[k];
        save_npy(
            out + "/patch" + std::to_string(k) + "_zyx.npy", "<f4", {p.H, p.W, 3}, p.zyx.data());
        save_npy(out + "/patch" + std::to_string(k) + "_sampling_mask.npy", "|u1",
            {p.H - 1, p.W - 1}, p.sampling.data());
    }
    save_npy(out + "/patch_probabilities.npy", "<f8", {(int64_t)set.probability.size()},
        set.probability.data());
    const std::string tracks_path = path_in(dataset, spec["paths"]["tracks_dbm"].str());
    CrossingCsr csr;
    const Tracks tracks = prepared_input(
        load_tracks(tracks_path + ".vctracks", m.s.z_begin, m.s.z_end), set, ps, csr, tracks_path);
    save_npy(out + "/prepared_flat_zyx.npy", "<f4", {tracks.offsets.back(), 3}, tracks.zyx.data());
    save_npy(out + "/prepared_offsets.npy", "<i8", {(int64_t)tracks.offsets.size()},
        tracks.offsets.data());
    save_npy(
        out + "/crossings_offsets.npy", "<i8", {(int64_t)csr.offsets.size()}, csr.offsets.data());
    for (const auto &[name, values] :
        {std::pair<const char *, const std::vector<int32_t> *>{"partners", &csr.partners},
            {"self_local", &csr.self_local}, {"partner_local", &csr.partner_local}})
        save_npy(
            out + "/crossings_" + name + ".npy", "<i4", {(int64_t)values->size()}, values->data());
    refresh_theta(set, m, sh);
    save_npy(out + "/theta_node_theta.npy", "<f4", {set.nodes}, set.theta.data());
    save_npy(out + "/theta_potential.npy", "<i4", {set.nodes}, set.potential.data());
    std::vector<int64_t> shape;
    PatchSample smp;
    smp.points = load_npy_float(dir + "/loss_patch_slice_zyxs.npy", &shape);
    smp.rows = (int)shape[0];
    smp.cap = (int)shape[1];
    smp.nodes = load_npy_int(dir + "/loss_patch_node_ids.npy");
    const std::vector<int64_t> mask = load_npy_int(dir + "/loss_patch_sample_mask.npy");
    smp.counts.assign((size_t)smp.rows, 0);
    for (int r = 0; r < smp.rows; ++r)
        for (int k = 0; k < smp.cap; ++k)
            smp.counts[(size_t)r] += mask[(size_t)r * smp.cap + (size_t)k] != 0;
    Grads g;
    g.reset(m);
    const double value = patch_loss(m, sh, set, smp, ps, 1.0f, g);
    const float original = load_npy_float(dir + "/loss_patch_value.npy")[0];
    std::printf("patch     port %.6g original %.6g (%.2e relative)\n", value, (double)original,
        std::fabs(value - original) / std::max(1e-30, std::fabs((double)original)));
    save_grads(out, "loss_patch", m, g);
}

static void family_replay(
    const std::string &dir, const std::string &dataset, const Model &m, const Shared &sh) {
    const std::string out = dir + "/port";
    LossSettings ls;
    ls.weight_umbilicus = ls.weight_sym = ls.weight_normals = ls.weight_spacing = ls.weight_track =
        1.0f;
    Json spec = Json::parse(read_text(dataset + "/spiral-scroll.json"));
    const Json &paths = spec["paths"];
    Lasagna lasagna = load_lasagna(path_in(dataset, paths["normal_x"].str()),
        path_in(dataset, paths["normal_y"].str()),
        path_in(dataset, paths["gradient_magnitude"].str()), spec["normal_zarr_group"].str(),
        (int64_t)spec["lasagna_scale"].num(), m.s.z_begin, m.s.z_end);
    auto f = [&](const std::string &k) { return load_npy_float(dir + "/loss_" + k + ".npy"); };
    auto ints = [&](const std::string &k) { return load_npy_int(dir + "/loss_" + k + ".npy"); };
    for (const std::string name : {"umbilicus", "sym", "normals", "spacing", "track"}) {
        Samples smp;
        Grads g;
        g.reset(m);
        double value = 0;
        auto t = std::chrono::steady_clock::now();
        if (name == "umbilicus")
            value = umbilicus_loss(m, sh, f("umbilicus_points"), 1.0f, g);
        if (name == "sym") {
            std::vector<float> c = f("sym_combined");
            const size_t n = c.size() / 9;
            smp.sym_points.assign(c.begin(), c.begin() + (long)(n * 3));
            smp.sym_tangent.resize(n * 3);
            for (size_t i = 0; i < n * 3; ++i)
                smp.sym_tangent[i] = (c[2 * n * 3 + i] - c[i]) / ls.sym_epsilon;
            value = sym_loss(m, sh, smp, ls, g);
        }
        if (name == "normals") {
            std::vector<float> c = f("normals_inv");
            const size_t n = c.size() / 9;
            smp.normal_points.assign(c.begin() + (long)(2 * n * 3), c.end());
            value = normals_loss(m, sh, lasagna, smp, ls, g);
        }
        if (name == "spacing") {
            auto w = ints("spacing_winding");
            smp.spacing_winding.assign(w.begin(), w.end());
            smp.spacing_theta = f("spacing_theta");
            smp.spacing_z = f("spacing_z");
            value = spacing_loss(m, sh, smp, ls, g);
        }
        if (name == "track") {
            TrackSample &tr = smp.tracks;
            tr.points = f("track_sampled_scroll");
            auto row_id = ints("track_row_id");
            tr.row_id.assign(row_id.begin(), row_id.end());
            tr.row_starts = ints("track_row_starts");
            auto gid = ints("track_group_id"), slot = ints("track_row_slot");
            tr.group_id.assign(gid.begin(), gid.end());
            tr.row_slot.assign(slot.begin(), slot.end());
            tr.num_groups = (int)ints("track_num_groups")[0];
            tr.group_width = (int)ints("track_group_width")[0];
            std::vector<int64_t> shape;
            tr.target_flat = load_npy_int(dir + "/loss_track_target_flat_idx.npy", &shape);
            tr.target_points = (int)shape[1];
            tr.primary_cross = ints("track_primary_cross_flat");
            tr.partner_cross = ints("track_partner_cross_flat");
            auto rows = ints("track_partner_rows");
            tr.partner_rows.assign(rows.begin(), rows.end());
            value = track_loss(m, sh, tr, ls, g);
        }
        const float original = f(name + "_value")[0];
        std::printf("%-9s port %.6g original %.6g (%.2e relative) %.3f s\n", name.c_str(), value,
            (double)original,
            std::fabs(value - original) / std::max(1e-30, std::fabs((double)original)),
            seconds_since(t));
        save_grads(out, "loss_" + name, m, g);
    }
}

static int golden(const std::string &dir, const std::string &dataset, bool metal) {
    std::string message;
    if (metal && !gpu::init(message))
        std::printf("Metal unavailable (%s)\n", message.c_str());
    std::printf("device: %s\n", gpu::enabled() ? "Metal" : "CPU");
    Json meta = Json::parse(read_text(dir + "/meta.json"));
    Settings s;
    s.z_begin = (int64_t)meta["config"]["z_begin"].num();
    s.z_end = (int64_t)meta["config"]["z_end"].num();
    s.anticlockwise = meta["outward_sense"].str() == "ACW";
    Model m;
    m.build(s, load_umbilicus(dataset + "/umbilicus.json"));
    std::vector<float> uz = load_npy_float(dir + "/umbilicus_z.npy"),
                       uyx = load_npy_float(dir + "/umbilicus_yx.npy");
    double table = 0;
    for (size_t i = 0; i < m.umbilicus_yx.size(); ++i)
        for (int k = 0; k < 2; ++k)
            table = std::max(table, (double)std::fabs(m.umbilicus_yx[i][k] - uyx[i * 2 + k]));
    std::printf("umbilicus table: %zu rows, max |ours - original| %.3g voxels\n",
        m.umbilicus_yx.size(), table);
    for (size_t i = 0; i < m.umbilicus_yx.size(); ++i) {
        m.umbilicus_z[i] = uz[i];
        m.umbilicus_yx[i] = {uyx[i * 2], uyx[i * 2 + 1]};
    }
    std::vector<float> dr = load_npy_float(dir + "/dr_logit.npy");
    m.dr_logit = dr[0];
    load_into(m.linear, dir + "/linear.npy");
    load_into(m.gap, dir + "/gap.npy");
    load_into(m.low, dir + "/low.npy");
    load_into(m.high, dir + "/high.npy");
    Shared sh;
    make_shared(m, sh);
    const std::string out = dir + "/port";
    mkdir(out.c_str(), 0755);
    for (const std::string name : {"s2s", "inv"}) {
        std::vector<int64_t> shape;
        std::vector<float> points = load_npy_float(dir + "/" + name + "_points.npy", &shape);
        std::vector<float> weights = load_npy_float(dir + "/" + name + "_weights.npy");
        const size_t n = points.size() / 3;
        std::vector<float> result(points.size()), grad_points(points.size());
        auto t = std::chrono::steady_clock::now();
        if (name == "s2s")
            slice_to_spiral(m, sh, points.data(), n, result.data());
        else
            spiral_to_slice(m, sh, points.data(), n, result.data());
        const double forward = seconds_since(t);
        Grads g;
        g.reset(m);
        t = std::chrono::steady_clock::now();
        if (name == "s2s")
            slice_to_spiral_backward(
                m, sh, points.data(), n, weights.data(), grad_points.data(), g);
        else
            spiral_to_slice_backward(
                m, sh, points.data(), n, weights.data(), grad_points.data(), g);
        const double backward = seconds_since(t);
        save_npy(out + "/" + name + "_result.npy", "<f4", shape, result.data());
        save_npy(out + "/" + name + "_grad_points.npy", "<f4", shape, grad_points.data());
        save_grads(out, name, m, g);
        std::printf("%s: %zu points, forward %.3f s, backward %.3f s\n", name.c_str(), n, forward,
            backward);
    }
    family_replay(dir, dataset, m, sh);
    if (exists(dir + "/patch0_zyx.npy"))
        patch_replay(dir, dataset, m, sh);
    return 0;
}

static void save_params(const std::string &dir, const Model &m) {
    mkdir(dir.c_str(), 0755);
    save_npy(dir + "/dr_logit.npy", "<f4", {}, &m.dr_logit);
    save_npy(dir + "/linear.npy", "<f4", {(int64_t)m.linear_z, 2, 2}, m.linear.data());
    save_npy(dir + "/gap.npy", "<f4", {m.gap_num_z, m.gap_width}, m.gap.data());
    save_npy(dir + "/low.npy", "<f4", {m.s.stages, 3, m.lr[0], m.lr[1], m.lr[2]}, m.low.data());
    save_npy(dir + "/high.npy", "<f4", {m.s.stages, 3, m.hr[0], m.hr[1], m.hr[2]}, m.high.data());
}

static int fit(int argc, char **argv) {
    std::string dataset, out;
    bool metal = false, use_patches = true;
    int64_t z0 = 6900, z1 = 7100;
    long steps = 1500, schedule = 0;
    uint64_t seed = 1;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--dataset")
            dataset = next();
        else if (a == "--out")
            out = next();
        else if (a == "--steps")
            steps = std::stol(next());
        else if (a == "--seed")
            seed = std::stoull(next());
        else if (a == "--schedule")
            schedule = std::stol(next());
        else if (a == "--metal")
            metal = true;
        else if (a == "--no-patches")
            use_patches = false;
        else if (a == "--points")
            gpu::set_points(next() == "gpu");
        else if (a == "--z") {
            std::string v = next();
            z0 = std::stoll(v.substr(0, v.find(':')));
            z1 = std::stoll(v.substr(v.find(':') + 1));
        }
    }
    if (dataset.empty() || out.empty())
        throw std::runtime_error("fit needs --dataset and --out");
    mkdir(out.c_str(), 0755);
    std::string message;
    if (metal && !gpu::init(message))
        std::printf("Metal unavailable (%s), running on the CPU\n", message.c_str());
    std::printf("device: %s\n", gpu::enabled() ? "Metal" : "CPU");
    const auto started = std::chrono::steady_clock::now();
    Json spec = Json::parse(read_text(dataset + "/spiral-scroll.json"));
    const Json &paths = spec["paths"];
    Settings s;
    s.z_begin = z0;
    s.z_end = z1;
    std::string sense = spec["spiral_outward_sense"].str();
    for (auto &c : sense)
        c = (char)toupper(c);
    s.anticlockwise = sense == "ACW";
    const std::string tracks_path = path_in(dataset, paths["tracks_dbm"].str());
    PatchSettings ps;
    ps.per_step = (int)std::max(1.0, std::nearbyint(360.0 * (double)(z1 - z0) / 9500.0));
    PatchSet patches;
    const std::string patch_dir = use_patches ? patches_dir(dataset, spec) : "";
    if (!patch_dir.empty()) {
        patches = load_patches(patch_dir, z0, z1, ps);
        std::printf("loaded %zu patches from %s: %lld sampling quads, %zu trusted vertices\n",
            patches.patches.size(), patch_dir.c_str(), (long long)patches.nodes,
            patches.trusted.size() / 3);
        if (!patches.empty() && steps > ps.dt_start)
            throw std::runtime_error(
                "the patch distance-transform loss (from step 25000) is not ported");
    }
    CrossingCsr crossings;
    Tracks tracks = prepared_input(
        load_tracks(tracks_path + ".vctracks", z0, z1), patches, ps, crossings, tracks_path);
    TrackPool pool = prepare_tracks(tracks, &crossings, 20.0, 60.0, {0.0, 0.15, 0.85});
    const std::string group = spec.has("normal_zarr_group") ? spec["normal_zarr_group"].str() : "4";
    const int64_t scale = spec.has("lasagna_scale") ? (int64_t)spec["lasagna_scale"].num() : 4;
    Lasagna lasagna = load_lasagna(path_in(dataset, paths["normal_x"].str()),
        path_in(dataset, paths["normal_y"].str()),
        path_in(dataset, paths["gradient_magnitude"].str()), group, scale, z0, z1);
    Model m;
    m.build(s, load_umbilicus(dataset + "/umbilicus.json"));
    std::printf("loaded %zu tracks within z-roi [%lld, %lld), %lld resampled points, %zu/%zu "
                "connected, %zu directed "
                "crossings\n",
        tracks.size(), (long long)z0, (long long)z1, (long long)pool.offsets.back(), pool.connected,
        pool.count, pool.directed);
    std::printf("track length bins: short <= %.1f (%lld), medium <= %.1f (%lld), long (%lld)\n",
        pool.length_edges[0], (long long)pool.bin_counts[0], pool.length_edges[1],
        (long long)pool.bin_counts[1], (long long)pool.bin_counts[2]);
    std::printf("setup %.1f s, flow %d x 3 x %d x %d x %d, %zu parameters\n",
        seconds_since(started), m.s.stages, m.hr[0], m.hr[1], m.hr[2],
        m.low.size() + m.high.size() + m.linear.size() + m.gap.size() + 1);
    LossSettings ls;
    const double slices = (double)(z1 - z0);
    ls.tracks = (int)std::max(1L, std::lround(48000 * slices / 9500));
    ls.normal_points = (int)std::max(1L, std::lround(60000 * slices / 9500));
    ls.regularisation_points = (int)std::max(1L, std::lround(4500 * slices / 9500));
    const double lr = 3e-5,
                 gamma = std::pow(0.3, 1.0 / (double)std::max(1L, schedule ? schedule : steps));
    AdamW o_dr, o_linear, o_gap, o_low, o_high;
    Field low_first, low_second, high_first, high_second;
    o_gap.weight_decay = 0.01;
    std::mt19937_64 rng(seed);
    Shared sh;
    Samples samples;
    Grads g;
    if (!patches.empty()) {
        make_shared(m, sh);
        refresh_theta(patches, m, sh);
        enforce_liftability(patches, m, sh, ps);
        patches.fresh = true;
        std::printf("theta topology ready: %zu patches, %lld nodes, %zu rejected\n",
            patches.patches.size(), (long long)patches.nodes, patches.rejected.size());
    }
    double step_time = 0, phases[9] = {0};
    for (long it = 0; it < steps; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        auto mark = [&](int k, std::chrono::steady_clock::time_point &t) {
            phases[k] += seconds_since(t);
            t = std::chrono::steady_clock::now();
        };
        auto tp = t0;
        make_shared(m, sh);
        refresh_if_due(patches, it, m, sh, ps);
        mark(0, tp);
        draw_samples(m, sh, pool, ls, rng, samples);
        const PatchSample patch_sample = draw_patches(patches, ps.per_step, ps.points, rng);
        mark(1, tp);
        g.reset(m);
        LossValues v;
        v.umbilicus = umbilicus_loss(m, sh, m.umbilicus_raw, ls.weight_umbilicus, g);
        if (!patches.empty())
            v.patch = patch_loss(m, sh, patches, patch_sample, ps, ls.weight_patch, g);
        mark(2, tp);
        v.sym = sym_loss(m, sh, samples, ls, g);
        mark(3, tp);
        v.normals = normals_loss(m, sh, lasagna, samples, ls, g);
        mark(4, tp);
        v.spacing = spacing_loss(m, sh, samples, ls, g);
        mark(5, tp);
        v.track = track_loss(m, sh, samples.tracks, ls, g);
        mark(6, tp);
        double gdr;
        std::vector<double> glin, ggap;
        Field glow, ghigh;
        finish_gradients(m, g, gdr, glin, ggap, glow, ghigh);
        mark(7, tp);
        const double lr_now = lr * std::pow(gamma, (double)it);
        float fdr = (float)gdr;
        std::vector<float> flin(glin.begin(), glin.end()), fgap(ggap.begin(), ggap.end());
        o_dr.update(&m.dr_logit, &fdr, 1, lr_now);
        o_linear.update(m.linear.data(), flin.data(), flin.size(), lr_now);
        o_gap.update(m.gap.data(), fgap.data(), fgap.size(), lr_now);
        if (gpu::enabled()) {
            gpu::adamw(m.low, glow, low_first, low_second, lr_now, 0.0, it + 1);
            gpu::adamw(m.high, g.field, high_first, high_second, lr_now * 0.2, 0.0, it + 1);
        } else {
            o_low.update(m.low.data(), glow.data(), glow.size(), lr_now);
            o_high.update(m.high.data(), ghigh.data(), ghigh.size(), lr_now * 0.2);
        }
        mark(8, tp);
        step_time += seconds_since(t0);
        if (it == 0 || (it + 1) % 200 == 0 || it + 1 == steps)
            std::printf("step %ld: loss = %.1f, umbilicus = %.1f, patch_radius = %.1f, "
                        "sym_dirichlet = %.1f, "
                        "dense_normals = %.1f, "
                        "min_spacing = %.1f, track_radius = %.1f, track_dt = 0.0 (%.3f s/step)\n",
                it, v.total(ls), v.umbilicus * ls.weight_umbilicus, v.patch * ls.weight_patch,
                v.sym * ls.weight_sym, v.normals * ls.weight_normals, v.spacing * ls.weight_spacing,
                v.track * ls.weight_track, step_time / (double)(it + 1));
        std::fflush(stdout);
    }
    const char *names[9] = {
        "upsample", "draw", "umbilicus", "sym", "normals", "spacing", "track", "finish", "adamw"};
    std::printf("per step:");
    for (int k = 0; k < 9; ++k)
        std::printf(" %s %.1f ms", names[k], 1000.0 * phases[k] / (double)std::max(1L, steps));
    std::printf("\n");
    save_params(out + "/params", m);
    const auto exporting = std::chrono::steady_clock::now();
    make_shared(m, sh);
    ExportSettings es;
    es.voxel_um = spec.has("voxel_size_um") ? spec["voxel_size_um"].num() : 0.0;
    const int written = export_windings(m, sh, out + "/meshes", es);
    std::printf("exported %d windings in %.1f s\n", written, seconds_since(exporting));
    std::printf("{\"steps\": %ld, \"seconds\": %.1f, \"seconds_per_step\": %.4f}\n", steps,
        seconds_since(started), step_time / (double)std::max(1L, steps));
    return 0;
}

static int export_command(int argc, char **argv) {
    std::string dataset, params, out;
    bool metal = false;
    int64_t z0 = 6900, z1 = 7100;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--dataset")
            dataset = next();
        else if (a == "--params")
            params = next();
        else if (a == "--out")
            out = next();
        else if (a == "--metal")
            metal = true;
        else if (a == "--z") {
            std::string v = next();
            z0 = std::stoll(v.substr(0, v.find(':')));
            z1 = std::stoll(v.substr(v.find(':') + 1));
        }
    }
    if (dataset.empty() || params.empty() || out.empty())
        throw std::runtime_error("export needs --dataset, --params and --out");
    std::string message;
    if (metal && !gpu::init(message))
        std::printf("Metal unavailable (%s), running on the CPU\n", message.c_str());
    Json spec = Json::parse(read_text(dataset + "/spiral-scroll.json"));
    Settings s;
    s.z_begin = z0;
    s.z_end = z1;
    std::string sense = spec["spiral_outward_sense"].str();
    for (auto &c : sense)
        c = (char)toupper(c);
    s.anticlockwise = sense == "ACW";
    Model m;
    m.build(s, load_umbilicus(dataset + "/umbilicus.json"));
    m.dr_logit = load_npy_float(params + "/dr_logit.npy")[0];
    load_into(m.linear, params + "/linear.npy");
    load_into(m.gap, params + "/gap.npy");
    load_into(m.low, params + "/low.npy");
    load_into(m.high, params + "/high.npy");
    Shared sh;
    make_shared(m, sh);
    ExportSettings es;
    es.voxel_um = spec.has("voxel_size_um") ? spec["voxel_size_um"].num() : 0.0;
    const auto t = std::chrono::steady_clock::now();
    const int written = export_windings(m, sh, out, es);
    std::printf("exported %d windings in %.1f s\n", written, seconds_since(t));
    return 0;
}

static int crossings_command(int argc, char **argv) {
    std::string dataset, out;
    int64_t z0 = 6900, z1 = 7100;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--dataset")
            dataset = next();
        else if (a == "--out")
            out = next();
        else if (a == "--z") {
            std::string v = next();
            z0 = std::stoll(v.substr(0, v.find(':')));
            z1 = std::stoll(v.substr(v.find(':') + 1));
        }
    }
    if (dataset.empty() || out.empty())
        throw std::runtime_error("crossings needs --dataset and --out");
    mkdir(out.c_str(), 0755);
    const Json spec = Json::parse(read_text(dataset + "/spiral-scroll.json"));
    const std::string tracks_path = path_in(dataset, spec["paths"]["tracks_dbm"].str());
    const Tracks tracks = load_tracks(tracks_path + ".vctracks", z0, z1);
    const auto t = std::chrono::steady_clock::now();
    const CrossingCsr rebuilt = find_crossings(tracks);
    const double seconds = seconds_since(t);
    Npz sidecar(tracks_path + ".crossings.npz");
    const CrossingCsr cached = sidecar_crossings(sidecar, tracks);
    for (const auto &[name, csr] :
        {std::pair<const char *, const CrossingCsr *>{"rebuilt", &rebuilt}, {"sidecar", &cached}}) {
        save_npy(out + "/" + name + "_offsets.npy", "<i8", {(int64_t)csr->offsets.size()},
            csr->offsets.data());
        save_npy(out + "/" + name + "_partners.npy", "<i4", {(int64_t)csr->partners.size()},
            csr->partners.data());
        save_npy(out + "/" + name + "_self_local.npy", "<i4", {(int64_t)csr->self_local.size()},
            csr->self_local.data());
        save_npy(out + "/" + name + "_partner_local.npy", "<i4",
            {(int64_t)csr->partner_local.size()}, csr->partner_local.data());
    }
    std::printf("%zu tracks: %zu directed crossings rebuilt in %.2f s, %zu in the sidecar\n",
        tracks.size(), rebuilt.partners.size(), seconds, cached.partners.size());
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc >= 4 && std::string(argv[1]) == "golden") {
            bool metal = false;
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--metal")
                    metal = true;
                else if (a == "--points" && i + 1 < argc)
                    gpu::set_points(std::string(argv[++i]) == "gpu");
            }
            return golden(argv[2], argv[3], metal);
        }
        if (argc >= 2 && std::string(argv[1]) == "fit")
            return fit(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "export")
            return export_command(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "crossings")
            return crossings_command(argc, argv);
        std::fprintf(stderr,
            "spiral_fit fit --dataset DIR --out DIR [--z 6900:7100] [--steps 1500] [--schedule N] "
            "[--seed 1] "
            "[--no-patches] [--metal [--points cpu|gpu]]\n       spiral_fit export --dataset DIR "
            "--params DIR "
            "--out DIR [--z 6900:7100] [--metal]\n       spiral_fit golden GOLDEN_DIR DATASET "
            "[--metal "
            "[--points cpu|gpu]]\n       spiral_fit crossings --dataset DIR --out DIR [--z "
            "6900:7100]\n");
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "spiral_fit: %s\n", e.what());
        return 1;
    }
}
