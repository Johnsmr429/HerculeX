#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;

typedef struct {
    int32_t Z, Y, X, cells_x, cells_y, first_z, air, slab_z0;
    uint32_t capacity;
} Params;

int hx_gpu_open(const char *source, char *message, int length) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            snprintf(message, length, "no Metal device");
            return 1;
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (!library) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return 2;
        }
        pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"cell_nodes"]
                                                  error:&error];
        if (!pipeline) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return 3;
        }
        queue = [device newCommandQueue];
        snprintf(message, length, "%s", [[device name] UTF8String]);
        return 0;
    }
}

int hx_gpu_cells(const uint8_t *slab, int Z, int Y, int X, int cells_z, int cells_y, int cells_x,
    int first_z, int air, int slab_z0, void *nodes, int32_t *ids, uint32_t capacity,
    uint32_t *count, double *seconds) {
    @autoreleasepool {
        id<MTLBuffer> volume = [device newBufferWithBytes:slab
                                                   length:(size_t)Z * Y * X
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:(size_t)capacity * 20
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> index = [device newBufferWithLength:(size_t)capacity * 4
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> counter = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        if (!volume || !out || !index || !counter)
            return 2;
        memset([counter contents], 0, 4);
        Params params = {.Z = Z,
            .Y = Y,
            .X = X,
            .cells_x = cells_x,
            .cells_y = cells_y,
            .first_z = first_z,
            .air = air,
            .slab_z0 = slab_z0,
            .capacity = capacity};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:volume offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBuffer:index offset:0 atIndex:2];
        [encoder setBuffer:counter offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)cells_z * cells_y * cells_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.error)
            return 3;
        *seconds = command.GPUEndTime - command.GPUStartTime;
        uint32_t found = *(uint32_t *)[counter contents];
        *count = found;
        if (found > capacity)
            return 4;
        memcpy(nodes, [out contents], (size_t)found * 20);
        memcpy(ids, [index contents], (size_t)found * 4);
        return 0;
    }
}

#include "../node_stream.h"

static id<MTLDevice> stack_gpu;
static id<MTLCommandQueue> stack_queue;
static id<MTLComputePipelineState> stack_kernel, fibre_stages[4];

int hx_stack_open(const char *source, char *message, int length) {
    @autoreleasepool {
        stack_gpu = MTLCreateSystemDefaultDevice();
        if (!stack_gpu)
            return 1;
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *))
            options.mathMode = MTLMathModeSafe;
        else
            options.fastMathEnabled = NO;
        NSError *error = nil;
        id<MTLLibrary> library =
            [stack_gpu newLibraryWithSource:[NSString stringWithUTF8String:source]
                                    options:options
                                      error:&error];
        if (library)
            stack_kernel = [stack_gpu
                newComputePipelineStateWithFunction:[library newFunctionWithName:@"flatten"]
                                              error:&error];
        NSArray *names = @[ @"fibre_window", @"smooth_axis", @"ply_layers", @"grain_energy" ];
        for (int k = 0; k < 4; k++) {
            id<MTLFunction> entry = [library newFunctionWithName:names[k]];
            if (!entry ||
                !(fibre_stages[k] = [stack_gpu newComputePipelineStateWithFunction:entry
                                                                             error:&error])) {
                snprintf(message, length, "%s",
                    error.localizedDescription.UTF8String ?: "missing fibre kernel");
                return 2;
            }
        }
        stack_queue = [stack_gpu newCommandQueue];
        if (!stack_kernel || !stack_queue) {
            snprintf(message, length, "%s", error.localizedDescription.UTF8String ?: "Metal queue");
            return 2;
        }
        return 0;
    }
}

int hx_stack_sample(const uint8_t *volume, const int64_t *offset, const int32_t *range,
    const int32_t *g, const float *grid, const int32_t *cy, const int32_t *cx, const float *fy,
    const float *fx, uint8_t *result, const float *rays) {
    @autoreleasepool {
        int64_t cells = (int64_t)g[0] * g[1] * g[2];
        size_t height = g[5], width = g[6], layers = g[7];
        size_t bytes = height * width * layers;
        MTLResourceOptions shared = MTLResourceStorageModeShared;
        if (bytes > stack_gpu.maxBufferLength || !stack_gpu.hasUnifiedMemory)
            return 1;
        id<MTLBuffer> map = [stack_gpu newBufferWithLength:cells * 8 options:shared];
        if (!map)
            return 1;
        uint64_t *addresses = map.contents;
        NSMutableArray<id<MTLBuffer>> *held = [NSMutableArray array];
        for (int64_t i = 0; i < cells; i++) {
            addresses[i] = 0;
            if (offset[i] < 0 || range[2 * i + 1] < range[2 * i])
                continue;
            id<MTLBuffer> page = [stack_gpu newBufferWithBytesNoCopy:(void *)(volume + offset[i])
                                                              length:128 * 128 * 128
                                                             options:shared
                                                         deallocator:nil];
            if (!page)
                return 1;
            addresses[i] = page.gpuAddress;
            [held addObject:page];
        }
        const void *source[] = {grid, cy, cx, fy, fx};
        size_t sizes[] = {(size_t)7 * g[3] * g[4] * sizeof(float), height * sizeof(*cy),
            width * sizeof(*cx), height * sizeof(*fy), width * sizeof(*fx)};
        id<MTLBuffer> inputs[6] = {map};
        for (int i = 0; i < 5; i++) {
            inputs[i + 1] = [stack_gpu newBufferWithBytes:source[i] length:sizes[i] options:shared];
            if (!inputs[i + 1])
                return 1;
        }
        id<MTLBuffer> out = [stack_gpu newBufferWithLength:bytes options:shared];
        id<MTLCommandBuffer> cmd = [stack_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!out || !enc)
            return 1;
        [enc setComputePipelineState:stack_kernel];
        for (int i = 0; i < 6; i++)
            [enc setBuffer:inputs[i] offset:0 atIndex:i];
        [enc setBytes:g length:8 * sizeof(int32_t) atIndex:6];
        [enc setBuffer:out offset:0 atIndex:7];
        for (id<MTLBuffer> buffer in held)
            [enc useResource:buffer usage:MTLResourceUsageRead];
        [enc dispatchThreads:MTLSizeMake(height * width, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error)
            return 2;
        memcpy(result, out.contents, bytes);
        return 0;
    }
}

typedef struct {
    float high, low;
    uint64_t bits;
} FibreWeight;

static pthread_mutex_t fibre_lock = PTHREAD_MUTEX_INITIALIZER;
static id<MTLBuffer> fibre_buffer;

static void fibre_dispatch(id<MTLComputeCommandEncoder> enc, int k, int n, int lines,
    const size_t *buffers, int count, const int *params, int words) {
    [enc setComputePipelineState:fibre_stages[k]];
    for (int i = 0; i < count; i++)
        [enc setBuffer:fibre_buffer offset:buffers[i] atIndex:i];
    [enc setBytes:params length:words * sizeof(*params) atIndex:count];
    [enc dispatchThreads:MTLSizeMake(n, lines, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static void smooth_fibre(id<MTLComputeCommandEncoder> enc, size_t src, size_t out, int scale,
    int count, int length, int inner) {
    int radii[] = {3, 10, 6, 12, 24}, r = radii[scale];
    int p[] = {count, length, inner, r, r >= 10 ? r / 8 * 8 : 0};
    size_t buf[] = {src, out, scale * 49 * sizeof(FibreWeight)};
    fibre_dispatch(enc, 1, count, 1, buf, 3, p, 5);
}
static size_t aligned_fibre(size_t size) { return (size + 255) & ~(size_t)255; }
int hx_fibre_energy(const uint8_t *raw, const int *idx, int n, int depth, int height, int width,
    const double *weights, float *energy) {
    @autoreleasepool {
        int pixels = height * width, planes = 17 * pixels, plies = 6 * pixels;
        size_t volume = aligned_fibre(5 * 49 * sizeof(FibreWeight));
        size_t indices = volume + aligned_fibre((size_t)depth * pixels);
        size_t a = indices + aligned_fibre(n * sizeof(int)),
               b = a + aligned_fibre(planes * sizeof(float));
        size_t layers = b + aligned_fibre(planes * sizeof(float)),
               stride = aligned_fibre(plies * sizeof(float));
        size_t fields[] = {layers + stride, layers + 2 * stride, layers + 3 * stride},
               out = layers + 4 * stride;
        size_t out_bytes = (size_t)24 * n * sizeof(float), need = out + out_bytes;
        pthread_mutex_lock(&fibre_lock);
        if (!fibre_buffer || fibre_buffer.length < need) {
            id<MTLBuffer> bigger = [stack_gpu newBufferWithLength:need
                                                          options:MTLResourceStorageModeShared];
            if (!bigger) {
                pthread_mutex_unlock(&fibre_lock);
                return 1;
            }
            fibre_buffer = bigger;
        }
        uint8_t *data = fibre_buffer.contents;
        FibreWeight *table = (FibreWeight *)data;
        for (int k = 0; k < 5 * 49; k++) {
            table[k].high = weights[k];
            table[k].low = weights[k] - (double)table[k].high;
            memcpy(&table[k].bits, weights + k, sizeof(double));
        }
        memcpy(data + volume, raw, (size_t)depth * pixels);
        memcpy(data + indices, idx, (size_t)n * sizeof(int));
        id<MTLCommandBuffer> command = [stack_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) {
            pthread_mutex_unlock(&fibre_lock);
            return 1;
        }
        for (int back = 0; back < 2; back++) {
            int reverse = back * (depth % 2), p[] = {depth, pixels, back, reverse};
            if (!reverse) {
                size_t convert[] = {volume, a};
                fibre_dispatch(enc, 0, planes, 1, convert, 2, p, 3);
                smooth_fibre(enc, a, b, 0, planes, 17, pixels);
                smooth_fibre(enc, b, a, 1, planes, height, width);
                smooth_fibre(enc, a, b, 1, planes, width, 1);
            }
            size_t interpolate[] = {volume, b, layers};
            fibre_dispatch(enc, 2, pixels, 1, interpolate, 3, p, 4);
        }
        for (int scale = 0; scale < 3; scale++) {
            smooth_fibre(enc, layers, a, scale + 2, plies, height, width);
            smooth_fibre(enc, a, fields[scale], scale + 2, plies, width, 1);
        }
        for (int band = 0; band < 2; band++) {
            size_t measure[] = {fields[band], fields[band + 1], indices, out};
            int params[] = {n, height, width, band};
            fibre_dispatch(enc, 3, n, 6, measure, 4, params, 4);
        }
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        int failed = command.status != MTLCommandBufferStatusCompleted;
        if (!failed)
            memcpy(energy, data + out, out_bytes);
        pthread_mutex_unlock(&fibre_lock);
        return failed;
    }
}
