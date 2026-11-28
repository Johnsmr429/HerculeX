#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../render_apple/metal_render.h"

#define BUFFERS 7

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;

typedef struct {
    void *buffers[BUFFERS];
} Scene;

int hx_syn_gpu_open(const char *source, char *message, int length) {
    @autoreleasepool {
        if (pipeline)
            return 0;
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            snprintf(message, length, "no Metal device");
            return 1;
        }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:options
                                                        error:&error];
        if (!library) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return 2;
        }
        pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"render"]
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

void *hx_syn_gpu_scene(const void *const *data, const int64_t *lengths) {
    @autoreleasepool {
        Scene *scene = calloc(1, sizeof(Scene));
        for (int i = 0; i < BUFFERS; i++) {
            id<MTLBuffer> buffer = [device newBufferWithBytes:data[i]
                                                       length:(NSUInteger)lengths[i]
                                                      options:MTLResourceStorageModeShared];
            if (!buffer) {
                for (int k = 0; k < i; k++)
                    CFBridgingRelease(scene->buffers[k]);
                free(scene);
                return NULL;
            }
            scene->buffers[i] = (void *)CFBridgingRetain(buffer);
        }
        return scene;
    }
}

int hx_syn_gpu_replace(void *handle, int which, const void *data, int64_t length) {
    @autoreleasepool {
        Scene *scene = handle;
        id<MTLBuffer> buffer = [device newBufferWithBytes:data
                                                   length:(NSUInteger)length
                                                  options:MTLResourceStorageModeShared];
        if (!buffer)
            return 1;
        CFBridgingRelease(scene->buffers[which]);
        scene->buffers[which] = (void *)CFBridgingRetain(buffer);
        return 0;
    }
}

int hx_syn_gpu_render(void *handle, const void *params, int64_t params_length, int nz, int ny,
    int nx, uint8_t *out, double *seconds) {
    @autoreleasepool {
        Scene *scene = handle;
        size_t size = (size_t)nz * ny * nx;
        id<MTLBuffer> result = [device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        if (!result)
            return 2;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:params length:(NSUInteger)params_length atIndex:0];
        for (int i = 0; i < BUFFERS; i++)
            [encoder setBuffer:(__bridge id<MTLBuffer>)scene->buffers[i] offset:0 atIndex:i + 1];
        [encoder setBuffer:result offset:0 atIndex:BUFFERS + 1];
        return finish_render(
            command, encoder, pipeline, result, MTLSizeMake(nx, ny, nz), out, size, seconds);
    }
}

void hx_syn_gpu_close(void *handle) {
    Scene *scene = handle;
    if (!scene)
        return;
    for (int i = 0; i < BUFFERS; i++)
        CFBridgingRelease(scene->buffers[i]);
    free(scene);
}
