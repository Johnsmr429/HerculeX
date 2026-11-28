#include "gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <map>
#include <vector>

namespace sp::gpu {

static const char *SOURCE =
#include "kernels.inc"
    ;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static std::map<std::string, id<MTLComputePipelineState>> pipelines;
static bool ready = false;

struct FlowArgs {
    int shape[4];
    int steps;
    uint32_t count, plane, slab;
    int keep;
    float h;
};

struct UpArgs {
    int hr[4];
    int lr[4];
    uint32_t planes;
};

struct AxisArgs {
    uint32_t outer, in_len, out_len, inner;
};

struct AdamArgs {
    uint32_t count;
    float decay, step_size, root, beta1, beta2, eps;
    int weight_decay;
};

bool init(std::string &message) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            message = "no Metal device";
            return false;
        }
        queue = [device newCommandQueue];
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                                      options:options
                                                        error:&error];
        if (!library) {
            message = error ? [[error localizedDescription] UTF8String] : "kernel compile failed";
            return false;
        }
        for (const char *name :
            {"flow_forward", "flow_backward", "upsample", "adjoint_axis", "adamw"}) {
            id<MTLFunction> f = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f
                                                                                  error:&error];
            if (!p) {
                message = std::string("pipeline ") + name + " failed";
                return false;
            }
            pipelines[name] = p;
        }
        ready = true;
        return true;
    }
}

bool enabled() { return ready; }

static bool points_on_gpu = true;

bool points() { return ready && points_on_gpu; }

void set_points(bool on) { points_on_gpu = on; }

static id<MTLBuffer> wrap(const Field &a) {
    return [device newBufferWithBytesNoCopy:(void *)a.data()
                                     length:PageAllocator<float>::rounded(a.capacity())
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static id<MTLBuffer> copy_in(const void *data, size_t bytes) {
    return [device newBufferWithBytes:data
                               length:std::max<size_t>(bytes, 16)
                              options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> scratch(size_t bytes) {
    return [device newBufferWithLength:std::max<size_t>(bytes, 16)
                               options:MTLResourceStorageModeShared];
}

static void dispatch(id<MTLComputeCommandEncoder> enc, const std::string &name, size_t threads) {
    id<MTLComputePipelineState> p = pipelines[name];
    [enc setComputePipelineState:p];
    const NSUInteger group = std::min<NSUInteger>(256, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(std::max<size_t>(threads, 1), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

struct Tables {
    const Model *model = nullptr;
    id<MTLBuffer> i0, i1, l0, l1;
    id<MTLBuffer> starts[3], sources[3], weights[3];
};

static Tables tables;

static void axis(int in, int out, int o, int &a, int &b, float &wa, float &wb) {
    const float scale = (float)in / (float)out;
    float src = scale * ((float)o + 0.5f) - 0.5f;
    if (src < 0)
        src = 0;
    a = (int)src;
    b = a + (a < in - 1 ? 1 : 0);
    wb = src - (float)a;
    wa = 1.0f - wb;
}

static void build_tables(const Model &m) {
    if (tables.model == &m)
        return;
    std::vector<int> i0, i1;
    std::vector<float> l0, l1;
    for (int d = 0; d < 3; ++d) {
        std::vector<std::vector<std::pair<int, float>>> back((size_t)m.lr[d]);
        for (int o = 0; o < m.hr[d]; ++o) {
            int a, b;
            float wa, wb;
            axis(m.lr[d], m.hr[d], o, a, b, wa, wb);
            i0.push_back(a);
            i1.push_back(b);
            l0.push_back(wa);
            l1.push_back(wb);
            back[(size_t)a].push_back({o, wa});
            back[(size_t)b].push_back({o, wb});
        }
        std::vector<int> starts{0}, sources;
        std::vector<float> weights;
        for (auto &row : back) {
            for (auto &[o, w] : row) {
                sources.push_back(o);
                weights.push_back(w);
            }
            starts.push_back((int)sources.size());
        }
        tables.starts[d] = copy_in(starts.data(), starts.size() * 4);
        tables.sources[d] = copy_in(sources.data(), sources.size() * 4);
        tables.weights[d] = copy_in(weights.data(), weights.size() * 4);
    }
    tables.i0 = copy_in(i0.data(), i0.size() * 4);
    tables.i1 = copy_in(i1.data(), i1.size() * 4);
    tables.l0 = copy_in(l0.data(), l0.size() * 4);
    tables.l1 = copy_in(l1.data(), l1.size() * 4);
    tables.model = &m;
}

void upsample(const Model &m, const Field &low, Field &up) {
    @autoreleasepool {
        build_tables(m);
        up.resize((size_t)m.s.stages * 3 * m.hr_cells());
        UpArgs a{{m.hr[0], m.hr[1], m.hr[2], 0}, {m.lr[0], m.lr[1], m.lr[2], 0},
            (uint32_t)(m.s.stages * 3)};
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:wrap(low) offset:0 atIndex:0];
        [enc setBuffer:wrap(up) offset:0 atIndex:1];
        [enc setBuffer:tables.i0 offset:0 atIndex:2];
        [enc setBuffer:tables.i1 offset:0 atIndex:3];
        [enc setBuffer:tables.l0 offset:0 atIndex:4];
        [enc setBuffer:tables.l1 offset:0 atIndex:5];
        [enc setBytes:&a length:sizeof(a) atIndex:6];
        dispatch(enc, "upsample", up.size());
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void upsample_adjoint(const Model &m, const Field &grad_up, Field &grad_low) {
    @autoreleasepool {
        build_tables(m);
        const uint32_t P = (uint32_t)(m.s.stages * 3);
        const uint32_t Z = m.hr[0], Y = m.hr[1], X = m.hr[2], lz = m.lr[0], ly = m.lr[1],
                       lx = m.lr[2];
        grad_low.resize((size_t)P * lz * ly * lx);
        id<MTLBuffer> t1 = scratch((size_t)P * Z * Y * lx * 4),
                      t2 = scratch((size_t)P * Z * ly * lx * 4);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const AxisArgs passes[3] = {{P * Z * Y, X, lx, 1}, {P * Z, Y, ly, lx}, {P, Z, lz, ly * lx}};
        id<MTLBuffer> ins[3] = {wrap(grad_up), t1, t2}, outs[3] = {t1, t2, wrap(grad_low)};
        const int axes[3] = {2, 1, 0};
        for (int k = 0; k < 3; ++k) {
            [enc setBuffer:ins[k] offset:0 atIndex:0];
            [enc setBuffer:outs[k] offset:0 atIndex:1];
            [enc setBuffer:tables.starts[axes[k]] offset:0 atIndex:2];
            [enc setBuffer:tables.sources[axes[k]] offset:0 atIndex:3];
            [enc setBuffer:tables.weights[axes[k]] offset:0 atIndex:4];
            [enc setBytes:&passes[k] length:sizeof(AxisArgs) atIndex:5];
            dispatch(
                enc, "adjoint_axis", (size_t)passes[k].outer * passes[k].out_len * passes[k].inner);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void adamw(Field &p, const Field &g, Field &first, Field &second, double lr, double weight_decay,
    long step) {
    @autoreleasepool {
        if (first.size() != p.size()) {
            first.assign(p.size(), 0.0f);
            second.assign(p.size(), 0.0f);
        }
        const double bc1 = 1.0 - std::pow(0.9, (double)step),
                     bc2 = 1.0 - std::pow(0.999, (double)step);
        AdamArgs a{(uint32_t)p.size(), (float)(1.0 - lr * weight_decay), (float)(lr / bc1),
            (float)std::sqrt(bc2), 0.9f, 0.999f, 1e-8f, weight_decay != 0.0};
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:wrap(p) offset:0 atIndex:0];
        [enc setBuffer:wrap(g) offset:0 atIndex:1];
        [enc setBuffer:wrap(first) offset:0 atIndex:2];
        [enc setBuffer:wrap(second) offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        dispatch(enc, "adamw", p.size());
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void zero(Field &a) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:wrap(a) range:NSMakeRange(0, a.size() * sizeof(float)) value:0];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

static FlowArgs flow_args(const Model &m, size_t n, int slab, bool inverse, bool keep) {
    FlowArgs a{};
    a.shape[0] = m.hr[0];
    a.shape[1] = m.hr[1];
    a.shape[2] = m.hr[2];
    a.steps = m.s.rk4_steps;
    a.count = (uint32_t)n;
    a.plane = (uint32_t)m.hr_cells();
    a.slab = (uint32_t)slab;
    a.keep = keep;
    a.h = (inverse ? -1.0f : 1.0f) / (float)m.s.rk4_steps;
    return a;
}

void flow(const Model &m, const Shared &sh, float *y, size_t n, bool inverse, float *inputs) {
    if (!n)
        return;
    @autoreleasepool {
        const int S = m.s.stages;
        const size_t bytes = n * 3 * sizeof(float);
        id<MTLBuffer> stages = scratch(bytes * (size_t)(S + 1));
        std::memcpy([stages contents], y, bytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLBuffer> low = wrap(sh.low_up), high = wrap(m.high);
        for (int k = 0; k < S; ++k) {
            const FlowArgs a = flow_args(m, n, inverse ? S - 1 - k : k, inverse, false);
            [enc setBuffer:stages offset:bytes * (size_t)k atIndex:0];
            [enc setBuffer:stages offset:bytes * (size_t)(k + 1) atIndex:1];
            [enc setBuffer:low offset:0 atIndex:2];
            [enc setBuffer:high offset:0 atIndex:3];
            [enc setBytes:&a length:sizeof(a) atIndex:4];
            dispatch(enc, "flow_forward", n);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float *all = (const float *)[stages contents];
        std::memcpy(y, all + n * 3 * (size_t)S, bytes);
        if (inputs)
            std::memcpy(inputs, all, bytes * (size_t)S);
    }
}

void flow_backward(const Model &m, const Shared &sh, const float *inputs, float *grad, size_t n,
    bool inverse, Field &acc) {
    if (!n)
        return;
    @autoreleasepool {
        const int S = m.s.stages;
        const size_t bytes = n * 3 * sizeof(float);
        id<MTLBuffer> starts = copy_in(inputs, bytes * (size_t)S);
        id<MTLBuffer> g0 = copy_in(grad, bytes), g1 = scratch(bytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLBuffer> low = wrap(sh.low_up), high = wrap(m.high), field = wrap(acc);
        for (int k = S - 1; k >= 0; --k) {
            const FlowArgs a = flow_args(m, n, inverse ? S - 1 - k : k, inverse, true);
            [enc setBuffer:starts offset:bytes * (size_t)k atIndex:0];
            [enc setBuffer:g0 offset:0 atIndex:1];
            [enc setBuffer:g1 offset:0 atIndex:2];
            [enc setBuffer:low offset:0 atIndex:3];
            [enc setBuffer:high offset:0 atIndex:4];
            [enc setBuffer:field offset:0 atIndex:5];
            [enc setBytes:&a length:sizeof(a) atIndex:6];
            dispatch(enc, "flow_backward", n);
            std::swap(g0, g1);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        std::memcpy(grad, [g0 contents], bytes);
    }
}

}
