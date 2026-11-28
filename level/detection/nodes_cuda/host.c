#include <pthread.h>
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
int cuLaunchKernel(CUfunction function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_bytes, void *stream, void **arguments, void **extra);
int nvrtcCreateProgram(nvrtcProgram *program, const char *source, const char *name, int headers,
    const char *const *header_sources, const char *const *header_names);
int nvrtcCompileProgram(nvrtcProgram program, int count, const char *const *options);
int nvrtcGetPTXSize(nvrtcProgram program, size_t *size);
int nvrtcGetPTX(nvrtcProgram program, char *ptx);
int nvrtcGetProgramLogSize(nvrtcProgram program, size_t *size);
int nvrtcGetProgramLog(nvrtcProgram program, char *log);
int nvrtcDestroyProgram(nvrtcProgram *program);

int cuStreamCreate(void **stream, unsigned int flags);
int cuStreamSynchronize(void *stream);
int cuMemcpyHtoDAsync_v2(CUdeviceptr dest, const void *src, size_t bytes, void *stream);
int cuMemcpyDtoHAsync_v2(void *dest, CUdeviceptr src, size_t bytes, void *stream);

static CUcontext context;
static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static int gpu_device;

static void initialize_device(void) {
    if (!cuInit(0) && !cuDeviceGet(&gpu_device, 0))
        cuDevicePrimaryCtxRetain(&context, gpu_device);
}
static CUfunction kernel;

typedef struct {
    int32_t Z, Y, X, cells_x, cells_y, first_z, air, slab_z0;
    uint32_t capacity;
} Params;

static int compile_kernel(const char *source, const char **names, int count, CUfunction *functions,
    char *message, int length) {
    const char *name = names[0];
    pthread_once(&initialized, initialize_device);
    if (!context) {
        snprintf(message, length, "no CUDA device");
        return 1;
    }
    nvrtcProgram program = NULL;
    size_t size = 0;
    const char *options[] = {"--std=c++20"};
    int result = 2;
    char *ptx = NULL;
    if (nvrtcCreateProgram(&program, source, name, 0, NULL, NULL) ||
        nvrtcCompileProgram(program, 1, options)) {
        snprintf(message, length, "NVRTC could not compile %s", name);
        if (program && !nvrtcGetProgramLogSize(program, &size) && size > 1 &&
            size <= (size_t)length)
            nvrtcGetProgramLog(program, message);
        goto finish;
    }
    if (nvrtcGetPTXSize(program, &size))
        goto finish;
    ptx = malloc(size);
    if (!ptx || nvrtcGetPTX(program, ptx))
        goto finish;
    CUmodule module;
    CUcontext popped;
    if (cuCtxPushCurrent_v2(context))
        goto finish;
    int refused = cuModuleLoadData(&module, ptx);
    for (int i = 0; i < count && !refused; i++) {
        name = names[i];
        refused = cuModuleGetFunction(&functions[i], module, name);
    }
    cuCtxPopCurrent_v2(&popped);
    if (refused) {
        snprintf(message, length, "the CUDA driver refused %s", name);
        result = 3;
        goto finish;
    }
    cuDeviceGetName(message, length, gpu_device);
    result = 0;
finish:
    if (program)
        nvrtcDestroyProgram(&program);
    free(ptx);
    return result;
}

int hx_gpu_open(const char *source, char *message, int length) {
    const char *names[] = {"cell_nodes"};
    return compile_kernel(source, names, 1, &kernel, message, length);
}

static void free_device_buffers(const CUdeviceptr *buffers, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (buffers[i])
            cuMemFree_v2(buffers[i]);
}

int hx_gpu_cells(const uint8_t *slab, int Z, int Y, int X, int cells_z, int cells_y, int cells_x,
    int first_z, int air, int slab_z0, void *nodes, int32_t *ids, uint32_t capacity,
    uint32_t *count, double *seconds) {
    Params params = {.Z = Z,
        .Y = Y,
        .X = X,
        .cells_x = cells_x,
        .cells_y = cells_y,
        .first_z = first_z,
        .air = air,
        .slab_z0 = slab_z0,
        .capacity = capacity};
    CUdeviceptr volume = 0, out = 0, index = 0, counter = 0;
    CUcontext popped;
    size_t voxels = (size_t)Z * Y * X;
    uint32_t found = 0;
    int status = 2;
    if (cuCtxPushCurrent_v2(context))
        return 2;
    if (cuMemAlloc_v2(&volume, voxels) || cuMemAlloc_v2(&out, (size_t)capacity * 20) ||
        cuMemAlloc_v2(&index, (size_t)capacity * 4) || cuMemAlloc_v2(&counter, 4) ||
        cuMemcpyHtoD_v2(volume, slab, voxels) || cuMemsetD32_v2(counter, 0, 1))
        goto cleanup;
    void *arguments[] = {&volume, &out, &index, &counter, &params};
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    status = 3;
    if (cuLaunchKernel(kernel, (unsigned)cells_z * cells_y * cells_x, 1, 1, 256, 1, 1, 0, NULL,
            arguments, NULL) ||
        cuCtxSynchronize() || cuMemcpyDtoH_v2(&found, counter, 4))
        goto cleanup;
    clock_gettime(CLOCK_MONOTONIC, &end);
    *seconds = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    *count = found;
    if (found > capacity)
        status = 4;
    else if (!found ||
        !(cuMemcpyDtoH_v2(nodes, out, (size_t)found * 20) ||
            cuMemcpyDtoH_v2(ids, index, (size_t)found * 4)))
        status = 0;
cleanup:
    free_device_buffers((CUdeviceptr[]){volume, out, index, counter}, 4);
    cuCtxPopCurrent_v2(&popped);
    return status;
}

#include "../node_stream.h"

static CUfunction stack_functions[5];

int hx_stack_open(const char *source, char *message, int length) {
    const char *names[] = {"flatten", "fibre_window", "smooth_axis", "ply_layers", "grain_energy"};
    return compile_kernel(source, names, 5, stack_functions, message, length);
}

typedef struct {
    int64_t offset;
    int32_t low, high;
} StackBrick;

int hx_stack_sample(const uint8_t *volume, const int64_t *offset, const int32_t *range,
    const int32_t *g, const float *grid, const int32_t *cy, const int32_t *cx, const float *fy,
    const float *fx, uint8_t *result, const float *rays) {
    size_t cells = (size_t)g[0] * g[1] * g[2], bytes = 0;
    size_t plane = 128 * 128, columns = (size_t)g[5] * g[6], out_bytes = columns * g[7];
    size_t index_bytes = cells * sizeof(StackBrick), ray_bytes = columns * 6 * sizeof(float);
    CUdeviceptr data = 0, index = 0, field = 0, out = 0;
    CUcontext popped;
    if (cuCtxPushCurrent_v2(context))
        return 1;
    int status = 1;
    StackBrick *table = malloc(index_bytes);
    if (!table)
        goto finish;
    for (size_t i = 0; i < cells; i++) {
        int lo = range[2 * i], hi = offset[i] < 0 ? -1 : range[2 * i + 1];
        table[i] = (StackBrick){.offset = bytes, .low = lo, .high = hi};
        if (hi >= lo)
            bytes += (hi + 1 - lo) * plane;
    }
    if (cuMemAlloc_v2(&data, bytes ? bytes : 1) || cuMemAlloc_v2(&index, index_bytes) ||
        cuMemAlloc_v2(&field, ray_bytes) || cuMemAlloc_v2(&out, out_bytes))
        goto finish;
    status = 2;
    for (size_t i = 0; i < cells; i++) {
        StackBrick brick = table[i];
        if (brick.high >= brick.low &&
            cuMemcpyHtoD_v2(data + brick.offset, volume + offset[i] + brick.low * plane,
                (brick.high + 1 - brick.low) * plane))
            goto finish;
    }
    if (cuMemcpyHtoD_v2(index, table, index_bytes) || cuMemcpyHtoD_v2(field, rays, ray_bytes))
        goto finish;
    int32_t shape[] = {g[0], g[1], g[2], columns, g[7], g[8]};
    void *args[] = {&data, &index, &field, shape, &out};
    if (cuLaunchKernel(
            stack_functions[0], (columns + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL) ||
        cuMemcpyDtoH_v2(result, out, out_bytes))
        goto finish;
    status = 0;
finish:
    free_device_buffers((CUdeviceptr[]){out, field, index, data}, 4);
    free(table);
    cuCtxPopCurrent_v2(&popped);
    return status;
}

typedef struct {
    CUdeviceptr device;
    size_t capacity;
    void *stream;
    pthread_mutex_t lock;
} FibreArena;

static FibreArena fibre_arena = {.lock = PTHREAD_MUTEX_INITIALIZER};

static int run_fibre(int function, int count, int lines, void **args) {
    return cuLaunchKernel(stack_functions[function], (count + 255) / 256, lines, 1, 256, 1, 1, 0,
        fibre_arena.stream, args, NULL);
}

static int smooth_fibre(CUdeviceptr src, CUdeviceptr out, CUdeviceptr table, int scale, int count,
    int length, int inner) {
    int radii[] = {3, 10, 6, 12, 24};
    int radius = radii[scale], plain = radius >= 10 ? (radius / 8) * 8 : 0;
    CUdeviceptr weights = table + (size_t)scale * 49 * sizeof(double);
    void *args[] = {&src, &out, &weights, &count, &length, &inner, &radius, &plain};
    return run_fibre(2, count, 1, args);
}

int hx_fibre_energy(const uint8_t *raw, const int *indices, int count, int depth, int height,
    int width, const double *weights, float *energy) {
    int pixels = height * width, samples = 17, planes = samples * pixels, plies = 6 * pixels;
    size_t raw_bytes = ((size_t)depth * pixels + 7) & ~(size_t)7;
    size_t index_bytes = count * sizeof(int), weights_bytes = 5 * 49 * sizeof(double);
    size_t field_bytes = (size_t)planes * sizeof(float), ply_bytes = (size_t)plies * sizeof(float);
    size_t out_bytes = (size_t)24 * count * sizeof(float);
    size_t need =
        weights_bytes + raw_bytes + index_bytes + field_bytes * 2 + ply_bytes * 4 + out_bytes;
    CUcontext popped;
    int failed = 1;
    if (cuCtxPushCurrent_v2(context))
        return 1;
    pthread_mutex_lock(&fibre_arena.lock);
    if (!fibre_arena.stream && cuStreamCreate(&fibre_arena.stream, 1))
        goto done;
    if (need > fibre_arena.capacity) {
        CUdeviceptr replacement = 0;
        if (cuMemAlloc_v2(&replacement, need))
            goto done;
        if (fibre_arena.device)
            cuMemFree_v2(fibre_arena.device);
        fibre_arena.device = replacement;
        fibre_arena.capacity = need;
    }
    CUdeviceptr table = fibre_arena.device, volume = table + weights_bytes;
    CUdeviceptr idx = volume + raw_bytes, a = idx + index_bytes, b = a + field_bytes;
    CUdeviceptr layers = b + field_bytes,
                fields[3] = {layers + ply_bytes, layers + 2 * ply_bytes, layers + 3 * ply_bytes};
    CUdeviceptr out = layers + 4 * ply_bytes;
    if (cuMemcpyHtoDAsync_v2(table, weights, weights_bytes, fibre_arena.stream) ||
        cuMemcpyHtoDAsync_v2(volume, raw, (size_t)depth * pixels, fibre_arena.stream) ||
        cuMemcpyHtoDAsync_v2(idx, indices, index_bytes, fibre_arena.stream))
        goto done;
    for (int back = 0; back < 2; back++) {
        int reverse = back * (depth % 2);
        if (!reverse) {
            void *convert[] = {&volume, &a, &depth, &pixels, &back};
            if (run_fibre(1, planes, 1, convert) ||
                smooth_fibre(a, b, table, 0, planes, samples, pixels) ||
                smooth_fibre(b, a, table, 1, planes, height, width) ||
                smooth_fibre(a, b, table, 1, planes, width, 1))
                goto done;
        }
        void *args[] = {&volume, &b, &layers, &depth, &samples, &pixels, &back, &reverse};
        if (run_fibre(3, pixels, 1, args))
            goto done;
    }
    for (int band = 0; band < 3; band++) {
        if (smooth_fibre(layers, a, table, band + 2, plies, height, width) ||
            smooth_fibre(a, fields[band], table, band + 2, plies, width, 1))
            goto done;
    }
    for (int band = 0; band < 2; band++) {
        void *args[] = {
            &fields[band], &fields[band + 1], &idx, &count, &height, &width, &band, &out};
        if (run_fibre(4, count, 6, args))
            goto done;
    }
    failed = cuMemcpyDtoHAsync_v2(energy, out, out_bytes, fibre_arena.stream);
done:
    if (fibre_arena.stream && cuStreamSynchronize(fibre_arena.stream))
        failed = 1;
    pthread_mutex_unlock(&fibre_arena.lock);
    cuCtxPopCurrent_v2(&popped);
    return failed;
}
