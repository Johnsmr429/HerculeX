R"(
#include <metal_stdlib>
using namespace metal;

struct FlowArgs {
    int3 shape;
    int steps;
    uint count;
    uint plane;
    uint slab;
    int keep;
    float h;
};

struct Cell {
    uint base;
    float3 near;
    float3 far;
    float3 last;
    float3 raw;
};

inline Cell cell_of(int3 shape, float3 p) {
    float3 last = float3(shape - 1);
    float3 raw = p * last;
    float3 c = clamp(raw, 0.0f, last);
    c = select(c, float3(0.0f), isnan(c));
    int3 lo = min(int3(floor(c)), shape - 2);
    float3 far = c - float3(lo);
    return {uint((lo.x * shape.y + lo.y) * shape.z + lo.z), 1.0f - far, far, last, raw};
}

inline float3 corner(device const float* low, device const float* high, uint plane, uint i) {
    return float3(low[i], low[i + plane], low[i + 2 * plane]) + float3(high[i], high[i + plane], high[i + 2 * plane]);
}

inline float3 velocity(device const float* low, device const float* high, int3 shape, uint plane, float3 p) {
    Cell c = cell_of(shape, p);
    uint dz = uint(shape.y * shape.z), dy = uint(shape.z);
    float3 v = 0.0f;
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            uint row = c.base + a * dz + b * dy;
            float w = (a ? c.far.x : c.near.x) * (b ? c.far.y : c.near.y);
            v += w * (c.near.z * corner(low, high, plane, row) + c.far.z * corner(low, high, plane, row + 1));
        }
    }
    return v;
}

inline float3 adjoint(device const float* low, device const float* high, device atomic_float* acc, int3 shape, uint plane,
                      bool keep, float3 g, float3 p) {
    Cell c = cell_of(shape, p);
    uint dz = uint(shape.y * shape.z), dy = uint(shape.z);
    float3 out = 0.0f;
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            for (int d = 0; d < 2; d++) {
                uint i = c.base + a * dz + b * dy + d;
                float wz = a ? c.far.x : c.near.x, wy = b ? c.far.y : c.near.y, wx = d ? c.far.z : c.near.z;
                if (keep) {
                    float3 share = g * (wz * wy * wx);
                    atomic_fetch_add_explicit(acc + i, share.x, memory_order_relaxed);
                    atomic_fetch_add_explicit(acc + i + plane, share.y, memory_order_relaxed);
                    atomic_fetch_add_explicit(acc + i + 2 * plane, share.z, memory_order_relaxed);
                }
                float along = dot(corner(low, high, plane, i), g);
                out += along * float3((a ? 1.0f : -1.0f) * wy * wx, (b ? 1.0f : -1.0f) * wz * wx, (d ? 1.0f : -1.0f) * wz * wy);
            }
        }
    }
    float3 inside = select(float3(0.0f), float3(1.0f), (c.raw >= 0.0f) && (c.raw <= c.last));
    return out * inside * c.last;
}

kernel void flow_forward(device const float* start [[buffer(0)]], device float* end [[buffer(1)]],
                         device const float* low_all [[buffer(2)]], device const float* high_all [[buffer(3)]],
                         constant FlowArgs& a [[buffer(4)]], uint n [[thread_position_in_grid]]) {
    if (n >= a.count) return;
    device const float* low = low_all + a.slab * 3 * a.plane;
    device const float* high = high_all + a.slab * 3 * a.plane;
    float h = a.h;
    float3 y = float3(start[3 * n], start[3 * n + 1], start[3 * n + 2]);
    for (int s = 0; s < a.steps; s++) {
        float3 k1 = velocity(low, high, a.shape, a.plane, y);
        float3 k2 = velocity(low, high, a.shape, a.plane, y + (h / 2) * k1);
        float3 k3 = velocity(low, high, a.shape, a.plane, y + (h / 2) * k2);
        float3 k4 = velocity(low, high, a.shape, a.plane, y + h * k3);
        y = y + (h / 6) * (((k1 + 2 * k2) + 2 * k3) + k4);
    }
    end[3 * n] = y.x;
    end[3 * n + 1] = y.y;
    end[3 * n + 2] = y.z;
}

kernel void flow_backward(device const float* start [[buffer(0)]], device const float* gradient [[buffer(1)]],
                          device float* start_gradient [[buffer(2)]], device const float* low_all [[buffer(3)]],
                          device const float* high_all [[buffer(4)]], device atomic_float* acc_all [[buffer(5)]],
                          constant FlowArgs& a [[buffer(6)]], uint n [[thread_position_in_grid]]) {
    if (n >= a.count) return;
    device const float* low = low_all + a.slab * 3 * a.plane;
    device const float* high = high_all + a.slab * 3 * a.plane;
    device atomic_float* acc = acc_all + a.slab * 3 * a.plane;
    float h = a.h;
    bool keep = a.keep != 0;
    float3 stages[32];
    float3 y = float3(start[3 * n], start[3 * n + 1], start[3 * n + 2]);
    for (int s = 0; s < a.steps; s++) {
        stages[4 * s] = y;
        float3 k1 = velocity(low, high, a.shape, a.plane, y);
        stages[4 * s + 1] = y + (h / 2) * k1;
        float3 k2 = velocity(low, high, a.shape, a.plane, stages[4 * s + 1]);
        stages[4 * s + 2] = y + (h / 2) * k2;
        float3 k3 = velocity(low, high, a.shape, a.plane, stages[4 * s + 2]);
        stages[4 * s + 3] = y + h * k3;
        float3 k4 = velocity(low, high, a.shape, a.plane, stages[4 * s + 3]);
        y = y + (h / 6) * (((k1 + 2 * k2) + 2 * k3) + k4);
    }
    float3 g = float3(gradient[3 * n], gradient[3 * n + 1], gradient[3 * n + 2]);
    for (int s = a.steps - 1; s >= 0; s--) {
        float3 sixth = g * (h / 6);
        float3 b4 = adjoint(low, high, acc, a.shape, a.plane, keep, sixth, stages[4 * s + 3]);
        float3 b3 = adjoint(low, high, acc, a.shape, a.plane, keep, sixth * 2 + b4 * h, stages[4 * s + 2]);
        float3 b2 = adjoint(low, high, acc, a.shape, a.plane, keep, sixth * 2 + b3 * (h / 2), stages[4 * s + 1]);
        float3 b1 = adjoint(low, high, acc, a.shape, a.plane, keep, sixth + b2 * (h / 2), stages[4 * s]);
        g = (((g + b4) + b3) + b2) + b1;
    }
    start_gradient[3 * n] = g.x;
    start_gradient[3 * n + 1] = g.y;
    start_gradient[3 * n + 2] = g.z;
}

struct UpArgs {
    int3 hr;
    int3 lr;
    uint planes;
};

inline float lr_at(device const float* src, int3 lr, int k, int j, int i) { return src[(k * lr.y + j) * lr.z + i]; }

kernel void upsample(device const float* low [[buffer(0)]], device float* up [[buffer(1)]], device const int* i0 [[buffer(2)]],
                     device const int* i1 [[buffer(3)]], device const float* l0 [[buffer(4)]], device const float* l1 [[buffer(5)]],
                     constant UpArgs& a [[buffer(6)]], uint t [[thread_position_in_grid]]) {
    uint per = uint(a.hr.x * a.hr.y * a.hr.z);
    if (t >= per * a.planes) return;
    uint plane = t / per, r = t % per;
    int x = int(r % uint(a.hr.z));
    int y = int((r / uint(a.hr.z)) % uint(a.hr.y));
    int z = int(r / uint(a.hr.y * a.hr.z));
    int zo = 0, yo = a.hr.x, xo = a.hr.x + a.hr.y;
    int z0 = i0[zo + z], z1 = i1[zo + z], y0 = i0[yo + y], y1 = i1[yo + y], x0 = i0[xo + x], x1 = i1[xo + x];
    float tz0 = l0[zo + z], tz1 = l1[zo + z], ty0 = l0[yo + y], ty1 = l1[yo + y], tx0 = l0[xo + x], tx1 = l1[xo + x];
    device const float* src = low + plane * uint(a.lr.x * a.lr.y * a.lr.z);
    up[t] = tz0 * (ty0 * (tx0 * lr_at(src, a.lr, z0, y0, x0) + tx1 * lr_at(src, a.lr, z0, y0, x1)) +
                   ty1 * (tx0 * lr_at(src, a.lr, z0, y1, x0) + tx1 * lr_at(src, a.lr, z0, y1, x1))) +
            tz1 * (ty0 * (tx0 * lr_at(src, a.lr, z1, y0, x0) + tx1 * lr_at(src, a.lr, z1, y0, x1)) +
                   ty1 * (tx0 * lr_at(src, a.lr, z1, y1, x0) + tx1 * lr_at(src, a.lr, z1, y1, x1)));
}

struct AxisArgs {
    uint outer;
    uint in_len;
    uint out_len;
    uint inner;
};

kernel void adjoint_axis(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], device const int* starts [[buffer(2)]],
                         device const int* sources [[buffer(3)]], device const float* weights [[buffer(4)]],
                         constant AxisArgs& a [[buffer(5)]], uint t [[thread_position_in_grid]]) {
    uint total = a.outer * a.out_len * a.inner;
    if (t >= total) return;
    uint inner = t % a.inner;
    uint j = (t / a.inner) % a.out_len;
    uint outer = t / (a.inner * a.out_len);
    float sum = 0.0f;
    for (int k = starts[j]; k < starts[j + 1]; k++) sum += weights[k] * in[(outer * a.in_len + uint(sources[k])) * a.inner + inner];
    out[t] = sum;
}

struct AdamArgs {
    uint count;
    float decay;
    float step_size;
    float root;
    float beta1;
    float beta2;
    float eps;
    int weight_decay;
};

kernel void adamw(device float* p [[buffer(0)]], device const float* g [[buffer(1)]], device float* m [[buffer(2)]],
                  device float* v [[buffer(3)]], constant AdamArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    float gi = isfinite(g[i]) ? g[i] : 0.0f;
    float pi = p[i];
    if (a.weight_decay) pi *= a.decay;
    float mi = m[i] + (gi - m[i]) * (1.0f - a.beta1);
    float vi = v[i] * a.beta2 + gi * gi * (1.0f - a.beta2);
    m[i] = mi;
    v[i] = vi;
    p[i] = pi - a.step_size * mi / (sqrt(vi) / a.root + a.eps);
}
)"
