#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

namespace sp {

template <class T> struct PageAllocator {
    using value_type = T;
    static constexpr size_t page = 16384;
    PageAllocator() = default;
    template <class U> PageAllocator(const PageAllocator<U> &) {}
    static size_t rounded(size_t n) { return (n * sizeof(T) + page - 1) / page * page; }
    T *allocate(size_t n) {
        void *p = nullptr;
        if (posix_memalign(&p, page, rounded(n)) != 0)
            throw std::bad_alloc();
        return (T *)p;
    }
    void deallocate(T *p, size_t) { std::free(p); }
    template <class U> bool operator==(const PageAllocator<U> &) const { return true; }
    template <class U> bool operator!=(const PageAllocator<U> &) const { return false; }
};

using Field = std::vector<float, PageAllocator<float>>;

struct Settings {
    int64_t z_begin = 4000, z_end = 17000;
    int64_t flow_z_margin = 160, flow_radius = 3200, flow_voxel_resolution = 16,
            flow_scale_factor = 6;
    int stages = 2, rk4_steps = 3;
    int64_t linear_z_resolution = 48, gap_logit_resolution = 24;
    int gap_capacity_windings = 144;
    double gap_lr_scale = 0.3, min_gap = 1.0, softplus_bias = 4.0, initial_dr = 16.0;
    double dr_parameter_scale = 12.0, linear_logits_scale = 40.0;
    bool anticlockwise = false;
};

struct Model {
    Settings s;
    std::array<int64_t, 3> flow_min{}, flow_max{};
    std::array<float, 3> range{};
    std::array<int, 3> hr{}, lr{};
    Field low, high;
    float dr_logit = 0;
    int linear_z = 0;
    std::vector<float> linear;
    int gap_num_z = 0, gap_width = 0, windings = 0;
    std::vector<float> first_index;
    std::vector<float> gap;
    float softplus_scale = 0, softplus_denominator = 0;
    std::vector<float> umbilicus_z;
    std::vector<std::array<float, 2>> umbilicus_yx;
    std::vector<float> umbilicus_raw;

    void build(const Settings &settings, const std::vector<std::array<float, 3>> &umbilicus_points);
    size_t hr_cells() const { return (size_t)hr[0] * hr[1] * hr[2]; }
    size_t lr_cells() const { return (size_t)lr[0] * lr[1] * lr[2]; }
};

struct Shared {
    float dr = 0;
    std::vector<float> linear;
    std::vector<float> gap;
    Field low_up;
};

void make_shared(const Model &m, Shared &sh);
void upsample(const Model &m, const Field &low, Field &up);
void upsample_adjoint(const Model &m, const Field &grad_up, Field &grad_low);

struct Grads {
    double dr = 0;
    std::vector<double> linear, gap;
    Field field;
    void reset(const Model &m);
};

void slice_to_spiral(const Model &m, const Shared &sh, const float *p, size_t n, float *out);
void spiral_to_slice(const Model &m, const Shared &sh, const float *p, size_t n, float *out);
void slice_to_spiral_backward(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters = true);
void spiral_to_slice_backward(const Model &m, const Shared &sh, const float *p, size_t n,
    const float *grad_out, float *grad_p, Grads &g, bool parameters = true);
float native_log_gap(const Model &m, const Shared &sh, int winding, float theta, float z);
void native_log_gap_backward(
    const Model &m, const Shared &sh, int winding, float theta, float z, float grad, Grads &g);
void finish_gradients(const Model &m, const Grads &g, double &dr_logit, std::vector<double> &linear,
    std::vector<double> &gap, Field &grad_low, Field &grad_high);

}
