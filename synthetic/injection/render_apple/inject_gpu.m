#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../render_apple/metal_render.h"

#define MAX_BUFFERS 16

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;

typedef struct {
    void *pipeline;
    int count;
    void *buffers[MAX_BUFFERS];
} Scene;

void *hx_inj_open(const char *source, const char *function, char *message, int length) {
    @autoreleasepool {
        if (!device) {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                snprintf(message, length, "no Metal device");
                return NULL;
            }
            queue = [device newCommandQueue];
        }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:options
                                                        error:&error];
        if (!library) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return NULL;
        }
        id<MTLFunction> entry =
            [library newFunctionWithName:[NSString stringWithUTF8String:function]];
        if (!entry) {
            snprintf(message, length, "no function %s", function);
            return NULL;
        }
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:entry
                                                                                  error:&error];
        if (!state) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return NULL;
        }
        Scene *scene = calloc(1, sizeof(Scene));
        scene->pipeline = (void *)CFBridgingRetain(state);
        snprintf(message, length, "%s", [[device name] UTF8String]);
        return scene;
    }
}

int hx_inj_buffers(void *handle, const void *const *data, const int64_t *lengths, int count) {
    @autoreleasepool {
        Scene *scene = handle;
        if (count > MAX_BUFFERS)
            return 1;
        for (int i = 0; i < scene->count; i++)
            CFBridgingRelease(scene->buffers[i]);
        scene->count = 0;
        for (int i = 0; i < count; i++) {
            id<MTLBuffer> buffer = [device newBufferWithBytes:data[i]
                                                       length:(NSUInteger)lengths[i]
                                                      options:MTLResourceStorageModeShared];
            if (!buffer)
                return 2;
            scene->buffers[i] = (void *)CFBridgingRetain(buffer);
            scene->count = i + 1;
        }
        return 0;
    }
}

int hx_inj_render(void *handle, const void *params, int64_t params_length, const void *inject,
    int64_t inject_length, int nz, int ny, int nx, uint8_t *out, int channels, double *seconds) {
    @autoreleasepool {
        Scene *scene = handle;
        id<MTLComputePipelineState> state = (__bridge id<MTLComputePipelineState>)scene->pipeline;
        size_t size = (size_t)nz * ny * nx * channels;
        id<MTLBuffer> result = [device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        if (!result)
            return 2;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state];
        [encoder setBytes:params length:(NSUInteger)params_length atIndex:0];
        [encoder setBytes:inject length:(NSUInteger)inject_length atIndex:1];
        for (int i = 0; i < scene->count; i++)
            [encoder setBuffer:(__bridge id<MTLBuffer>)scene->buffers[i] offset:0 atIndex:i + 2];
        [encoder setBuffer:result offset:0 atIndex:scene->count + 2];
        return finish_render(
            command, encoder, state, result, MTLSizeMake(nx, ny, nz), out, size, seconds);
    }
}

void hx_inj_close(void *handle) {
    Scene *scene = handle;
    if (!scene)
        return;
    for (int i = 0; i < scene->count; i++)
        CFBridgingRelease(scene->buffers[i]);
    CFBridgingRelease(scene->pipeline);
    free(scene);
}
