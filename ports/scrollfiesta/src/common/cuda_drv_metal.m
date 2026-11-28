#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cuda_drv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_KERNELS 16
#define MAX_ALLOCS 256

typedef struct { uintptr_t base; size_t bytes; id<MTLBuffer> buffer; } Alloc;

struct CudaDrv {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> kernels[MAX_KERNELS];
    const char *signature[MAX_KERNELS];
    int n_kernels;
    Alloc allocs[MAX_ALLOCS];
    int n_allocs;
    char name[256];
    char error[1024];
};

static const struct { const char *name, *kinds; } SIGNATURES[] = {
    {"vcr_mark", "PPP4444444444P"},
    {"vcr_layers", "PPP4444PP44444P4P"},
    {"vcr_ctmax", "PPP444444PP44444P"},
    {"vcr_mark_rect", "PPP444444444444444PPP4"},
    {"vcr_layers_rect", "PPP4444444PP44444P4P"},
    {"vcr_ctmax_rect", "PPP44444444444PP44444P"},
    {"vcr_patch_table", "PP4"},
};

static void set_error(CudaDrv cu, const char *what) {
    snprintf(cu->error, sizeof cu->error, "%s", what);
}

CudaDrv CudaDrv_open(const char *source, const char *const *kernel_names, int n_kernels, char *why, size_t why_len) {
    @autoreleasepool {
        if (n_kernels > MAX_KERNELS) { snprintf(why, why_len, "too many kernels"); return NULL; }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { snprintf(why, why_len, "no Metal device"); return NULL; }
        MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
        options.mathMode = MTLMathModeSafe;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
        if (!library) { snprintf(why, why_len, "Metal compile: %s", [[error description] UTF8String]); [device release]; return NULL; }
        CudaDrv cu = calloc(1, sizeof *cu);
        cu->device = device;
        cu->queue = [device newCommandQueue];
        cu->n_kernels = n_kernels;
        for (int k = 0; k < n_kernels; k++) {
            const char *kinds = NULL;
            for (size_t s = 0; s < sizeof SIGNATURES / sizeof SIGNATURES[0]; s++)
                if (!strcmp(SIGNATURES[s].name, kernel_names[k])) kinds = SIGNATURES[s].kinds;
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_names[k]]];
            if (!kinds || !function) {
                snprintf(why, why_len, "no Metal kernel or signature for %s", kernel_names[k]);
                [function release];
                [library release];
                CudaDrv_close(cu);
                return NULL;
            }
            cu->kernels[k] = [device newComputePipelineStateWithFunction:function error:&error];
            [function release];
            if (!cu->kernels[k]) {
                snprintf(why, why_len, "Metal pipeline %s: %s", kernel_names[k], [[error description] UTF8String]);
                [library release];
                CudaDrv_close(cu);
                return NULL;
            }
            cu->signature[k] = kinds;
        }
        [library release];
        snprintf(cu->name, sizeof cu->name, "%s (Metal)", [[device name] UTF8String]);
        return cu;
    }
}

void CudaDrv_close(CudaDrv cu) {
    if (!cu) return;
    for (int i = 0; i < cu->n_allocs; i++) [cu->allocs[i].buffer release];
    for (int k = 0; k < cu->n_kernels; k++) [cu->kernels[k] release];
    [cu->queue release];
    [cu->device release];
    free(cu);
}

const char *CudaDrv_device(CudaDrv cu) { return cu ? cu->name : "none"; }

int CudaDrv_alloc(CudaDrv cu, CudaDrv_ptr *p, size_t bytes) {
    if (cu->n_allocs == MAX_ALLOCS) { set_error(cu, "too many allocations"); return 1; }
    id<MTLBuffer> buffer = [cu->device newBufferWithLength:(bytes ? bytes : 1) options:MTLResourceStorageModeShared];
    if (!buffer) { set_error(cu, "out of GPU memory"); return 1; }
    Alloc *a = &cu->allocs[cu->n_allocs++];
    a->buffer = buffer;
    a->base = (uintptr_t)[buffer contents];
    a->bytes = bytes ? bytes : 1;
    *p = (CudaDrv_ptr)a->base;
    return 0;
}

static Alloc *find(CudaDrv cu, CudaDrv_ptr p) {
    for (int i = 0; i < cu->n_allocs; i++)
        if ((uintptr_t)p >= cu->allocs[i].base && (uintptr_t)p < cu->allocs[i].base + cu->allocs[i].bytes) return &cu->allocs[i];
    return NULL;
}

void CudaDrv_free(CudaDrv cu, CudaDrv_ptr p) {
    for (int i = 0; i < cu->n_allocs; i++)
        if (cu->allocs[i].base == (uintptr_t)p) {
            [cu->allocs[i].buffer release];
            cu->allocs[i] = cu->allocs[--cu->n_allocs];
            return;
        }
}

int CudaDrv_upload(CudaDrv cu, CudaDrv_ptr dst, const void *src, size_t bytes) {
    if (bytes && !find(cu, dst)) { set_error(cu, "upload to unknown memory"); return 1; }
    memcpy((void *)(uintptr_t)dst, src, bytes);
    return 0;
}

int CudaDrv_download(CudaDrv cu, void *dst, CudaDrv_ptr src, size_t bytes) {
    if (bytes && !find(cu, src)) { set_error(cu, "download from unknown memory"); return 1; }
    memcpy(dst, (const void *)(uintptr_t)src, bytes);
    return 0;
}

int CudaDrv_memset8(CudaDrv cu, CudaDrv_ptr dst, uint8_t value, size_t bytes) {
    if (bytes && !find(cu, dst)) { set_error(cu, "memset of unknown memory"); return 1; }
    memset((void *)(uintptr_t)dst, value, bytes);
    return 0;
}

int CudaDrv_launch(CudaDrv cu, int k, unsigned gx, unsigned gy, unsigned bx, unsigned by, void **params) {
    @autoreleasepool {
        if (k < 0 || k >= cu->n_kernels) { set_error(cu, "no such kernel"); return 1; }
        if (!gx || !gy) return 0;
        id<MTLCommandBuffer> command = [cu->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:cu->kernels[k]];
        const char *kinds = cu->signature[k];
        for (int i = 0; kinds[i]; i++) {
            if (kinds[i] == 'P') {
                CudaDrv_ptr p = *(const CudaDrv_ptr *)params[i];
                Alloc *a = find(cu, p);
                if (!a) { [encoder endEncoding]; set_error(cu, "kernel argument outside any allocation"); return 1; }
                [encoder setBuffer:a->buffer offset:(NSUInteger)((uintptr_t)p - a->base) atIndex:(NSUInteger)i];
            } else {
                [encoder setBytes:params[i] length:4 atIndex:(NSUInteger)i];
            }
        }
        [encoder dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(bx, by, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.error) { set_error(cu, [[command.error description] UTF8String]); return 1; }
        return 0;
    }
}

const char *CudaDrv_error(CudaDrv cu) { return cu ? cu->error : "no device"; }

int CudaDrv_selftest(void) {
    char why[512];
    static const char *names[] = {"vcr_patch_table"};
    static const char *source =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void vcr_patch_table(device int *tab [[buffer(0)]], device const int *changes [[buffer(1)]],\n"
        "    constant int &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
        "    const int i = (int)gid; if (i >= n) return;\n"
        "    if (changes[3*i] >= 0) tab[changes[3*i]] = -1; tab[changes[3*i+1]] = changes[3*i+2]; }\n";
    CudaDrv cu = CudaDrv_open(source, names, 1, why, sizeof why);
    if (!cu) { fprintf(stderr, "CudaDrv_selftest (Metal): %s\n", why); return 1; }
    CudaDrv_ptr tab, changes;
    int host_tab[8] = {0, 1, 2, 3, 4, 5, 6, 7}, host_changes[6] = {2, 5, 9, -1, 6, 4}, n = 2;
    int fails = CudaDrv_alloc(cu, &tab, sizeof host_tab) || CudaDrv_alloc(cu, &changes, sizeof host_changes)
             || CudaDrv_upload(cu, tab, host_tab, sizeof host_tab) || CudaDrv_upload(cu, changes, host_changes, sizeof host_changes);
    void *args[] = {&tab, &changes, &n};
    fails = fails || CudaDrv_launch(cu, 0, 1, 1, 128, 1, args) || CudaDrv_download(cu, host_tab, tab, sizeof host_tab);
    fails = fails || host_tab[2] != -1 || host_tab[5] != 9 || host_tab[6] != 4;
    fprintf(stderr, "CudaDrv_selftest (Metal, %s): %s\n", CudaDrv_device(cu), fails ? "FAILED" : "OK");
    CudaDrv_close(cu);
    return fails;
}
