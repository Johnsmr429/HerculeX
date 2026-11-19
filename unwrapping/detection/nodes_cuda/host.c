// CUDA host for kernel.cu: one blocking call per slab, the twin of nodes_apple/host.m with the same two calls.
// It builds without the CUDA toolkit: the driver API (libcuda, part of the NVIDIA driver) and the run-time compiler
// (libnvrtc, part of PyTorch's CUDA wheels) are declared here, and library.py loads both before this library.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct _nvrtcProgram *nvrtcProgram;
typedef unsigned long long CUdeviceptr;

int cuInit(unsigned int flags);
int cuDeviceGet(int *device, int ordinal);
int cuDeviceGetName(char *name, int length, int device);
int cuDevicePrimaryCtxRetain(CUcontext *context, int device);
int cuCtxPushCurrent_v2(CUcontext context);
int cuCtxPopCurrent_v2(CUcontext *context);
int cuCtxSynchronize(void);
int cuModuleLoadData(CUmodule *module, const void *image);
int cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
int cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes);
int cuMemFree_v2(CUdeviceptr pointer);
int cuMemcpyHtoD_v2(CUdeviceptr to, const void *from, size_t bytes);
int cuMemcpyDtoH_v2(void *to, CUdeviceptr from, size_t bytes);
int cuMemsetD32_v2(CUdeviceptr to, unsigned int value, size_t count);
int cuLaunchKernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                   unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
                   void *stream, void **arguments, void **extra);
int nvrtcCreateProgram(nvrtcProgram *program, const char *source, const char *name, int headers,
                       const char *const *header_sources, const char *const *header_names);
int nvrtcCompileProgram(nvrtcProgram program, int count, const char *const *options);
int nvrtcGetPTXSize(nvrtcProgram program, size_t *size);
int nvrtcGetPTX(nvrtcProgram program, char *ptx);
int nvrtcGetProgramLogSize(nvrtcProgram program, size_t *size);
int nvrtcGetProgramLog(nvrtcProgram program, char *log);
int nvrtcDestroyProgram(nvrtcProgram *program);

static CUcontext context;
static CUfunction kernel;

typedef struct {
    int32_t Z, Y, X, cells_x, cells_y, first_z, air, slab_z0;
    uint32_t capacity;
} Params;

int hx_gpu_open(const char *source, char *message, int length) {
    int device;
    if (cuInit(0) || cuDeviceGet(&device, 0) || cuDevicePrimaryCtxRetain(&context, device)) {
        snprintf(message, length, "no CUDA device");
        return 1;
    }
    nvrtcProgram program = NULL;
    size_t size = 0;
    if (nvrtcCreateProgram(&program, source, "kernel.cu", 0, NULL, NULL) || nvrtcCompileProgram(program, 0, NULL)) {
        snprintf(message, length, "NVRTC could not compile kernel.cu");
        if (program && !nvrtcGetProgramLogSize(program, &size) && size > 1 && size <= (size_t)length)
            nvrtcGetProgramLog(program, message);
        if (program) nvrtcDestroyProgram(&program);
        return 2;
    }
    nvrtcGetPTXSize(program, &size);
    char *ptx = malloc(size);
    nvrtcGetPTX(program, ptx);
    nvrtcDestroyProgram(&program);
    CUmodule module;
    CUcontext popped;
    cuCtxPushCurrent_v2(context);
    int refused = cuModuleLoadData(&module, ptx) || cuModuleGetFunction(&kernel, module, "cell_nodes");
    cuCtxPopCurrent_v2(&popped);
    free(ptx);
    if (refused) {
        snprintf(message, length, "the CUDA driver refused the compiled kernel");
        return 3;
    }
    cuDeviceGetName(message, length, device);
    return 0;
}

// Runs every cell of a slab; nodes and their cell numbers arrive in completion order, sorted by the caller.
int hx_gpu_cells(const uint8_t *slab, int Z, int Y, int X, int cells_z, int cells_y, int cells_x, int first_z, int air,
                 int slab_z0, void *nodes, int32_t *ids, uint32_t capacity, uint32_t *count, double *seconds) {
    Params params = {Z, Y, X, cells_x, cells_y, first_z, air, slab_z0, capacity};
    CUdeviceptr volume = 0, out = 0, index = 0, counter = 0;
    CUcontext popped;
    size_t voxels = (size_t)Z * Y * X;
    uint32_t found = 0;
    int status = 2;
    cuCtxPushCurrent_v2(context);
    if (!(cuMemAlloc_v2(&volume, voxels) || cuMemAlloc_v2(&out, (size_t)capacity * 20) ||
          cuMemAlloc_v2(&index, (size_t)capacity * 4) || cuMemAlloc_v2(&counter, 4) ||
          cuMemcpyHtoD_v2(volume, slab, voxels) || cuMemsetD32_v2(counter, 0, 1))) {
        void *arguments[] = {&volume, &out, &index, &counter, &params};
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        status = cuLaunchKernel(kernel, (unsigned int)cells_z * cells_y * cells_x, 1, 1, 256, 1, 1, 0, NULL, arguments,
                                NULL) ||
                         cuCtxSynchronize() || cuMemcpyDtoH_v2(&found, counter, 4)
                     ? 3
                     : 0;
        clock_gettime(CLOCK_MONOTONIC, &end);
        if (!status) {
            *seconds = (double)(end.tv_sec - start.tv_sec) + 1e-9 * (double)(end.tv_nsec - start.tv_nsec);
            *count = found;
            if (found > capacity) status = 4;
            else if (found && (cuMemcpyDtoH_v2(nodes, out, (size_t)found * 20) ||
                               cuMemcpyDtoH_v2(ids, index, (size_t)found * 4)))
                status = 3;
        }
    }
    if (volume) cuMemFree_v2(volume);
    if (out) cuMemFree_v2(out);
    if (index) cuMemFree_v2(index);
    if (counter) cuMemFree_v2(counter);
    cuCtxPopCurrent_v2(&popped);
    return status;
}
