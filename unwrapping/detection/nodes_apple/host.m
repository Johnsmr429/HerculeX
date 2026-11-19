// Metal host for kernel.metal: one blocking call per slab.

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
        pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"cell_nodes"]
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

// Runs every cell of a slab; nodes and their cell numbers arrive in completion order, sorted by the caller.
int hx_gpu_cells(const uint8_t *slab, int Z, int Y, int X, int cells_z, int cells_y, int cells_x, int first_z, int air,
                 int slab_z0, void *nodes, int32_t *ids, uint32_t capacity, uint32_t *count, double *seconds) {
    @autoreleasepool {
        id<MTLBuffer> volume = [device newBufferWithBytes:slab
                                                   length:(size_t)Z * Y * X
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:(size_t)capacity * 20 options:MTLResourceStorageModeShared];
        id<MTLBuffer> index = [device newBufferWithLength:(size_t)capacity * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> counter = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        if (!volume || !out || !index || !counter) return 2;
        memset([counter contents], 0, 4);
        Params params = {Z, Y, X, cells_x, cells_y, first_z, air, slab_z0, capacity};
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
        if (command.error) return 3;
        *seconds = command.GPUEndTime - command.GPUStartTime;
        uint32_t found = *(uint32_t *)[counter contents];
        *count = found;
        if (found > capacity) return 4;
        memcpy(nodes, [out contents], (size_t)found * 20);
        memcpy(ids, [index contents], (size_t)found * 4);
        return 0;
    }
}
