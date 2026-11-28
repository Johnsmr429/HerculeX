#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STEM 16
#define DEPTH 17
#define THREADS 256

typedef unsigned char u8;
typedef unsigned short u16;

cublasHandle_t blas;
cudnnHandle_t dnn;

void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "herculex: %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

__device__ float value(u16 h) { return __half2float(__ushort_as_half(h)); }

__device__ u16 bits(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ float leaky(float x) { return x > 0 ? x : __fmul_rn(x, 0.01f); }

int blocks(long n) { return (n + THREADS - 1) / THREADS; }

extern "C" void *gpu_alloc(size_t n) {
    void *p;
    if (!blas) {
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "herculex: cuBLAS does not start\n");
            exit(1);
        }
        cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
        if (cudnnCreate(&dnn) != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "herculex: cuDNN does not start\n");
            exit(1);
        }
    }
    check(cudaMalloc(&p, n), "cudaMalloc");
    return p;
}

extern "C" void gpu_put(void *to, const void *from, size_t n) {
    check(cudaMemcpy(to, from, n, cudaMemcpyHostToDevice), "upload");
}

extern "C" void gpu_get(void *to, const void *from, size_t n) {
    check(cudaMemcpy(to, from, n, cudaMemcpyDeviceToHost), "download");
}

__global__ void tile_grey(const u8 *stack, int height, int width, int y0, int x0, int rows,
    int cols, const u16 *table, u16 *image, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    long z = blockIdx.z;
    if (x < w && y < h) {
        u8 grey = x < cols && y < rows ? stack[(z * height + y0 + y) * width + x0 + x] : 0;
        image[(z * h + y) * w + x] = table[grey];
    }
}

__global__ void bias_leaky(u16 *y, const u16 *bias, long n, long total) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = bits(leaky(value(bits(value(y[i]) + value(bias[i / n])))));
    }
}

void dnn_check(cudnnStatus_t status, const char *what) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "herculex: %s: %s\n", what, cudnnGetErrorString(status));
        exit(1);
    }
}

typedef struct {
    int key[10];
    size_t room;
    uint64_t last_use;
    cudnnTensorDescriptor_t x, y;
    cudnnFilterDescriptor_t w;
    cudnnConvolutionDescriptor_t c;
    cudnnConvolutionFwdAlgo_t algo;
} Plan;

enum { PLAN_BUCKETS = 512, PLAN_WAYS = 4 };
static Plan plans[PLAN_BUCKETS][PLAN_WAYS];
static uint64_t plan_clock;

static Plan *plan_slot(const int key[10], size_t room) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < 10; i++) {
        hash = (hash ^ (uint32_t)key[i]) * 16777619u;
    }
    Plan *bucket = plans[(hash ^ (hash >> 16)) % PLAN_BUCKETS], *oldest = bucket;
    for (int p = 0; p < PLAN_WAYS; p++) {
        Plan *plan = bucket + p;
        if (plan->last_use && plan->room == room && !memcmp(plan->key, key, sizeof(plan->key))) {
            return plan;
        }
        if (plan->last_use < oldest->last_use) {
            oldest = plan;
        }
    }
    return oldest;
}

Plan *plan_of(int batch, int in, const int *size, int dims, int out, int k, int stride,
    cudnnDataType_t type, size_t room) {
    int key[10] = {
        batch, in, dims, out, k, stride, size[0], size[1], dims == 3 ? size[2] : 0, type};
    Plan *plan = plan_slot(key, room);
    if (plan->last_use) {
        if (plan->room == room && !memcmp(plan->key, key, sizeof(key))) {
            plan->last_use = ++plan_clock;
            return plan;
        }
        cudnnDestroyTensorDescriptor(plan->x);
        cudnnDestroyTensorDescriptor(plan->y);
        cudnnDestroyFilterDescriptor(plan->w);
        cudnnDestroyConvolutionDescriptor(plan->c);
    }
    memcpy(plan->key, key, sizeof(key));
    plan->room = room;
    plan->last_use = ++plan_clock;
    dnn_check(cudnnCreateTensorDescriptor(&plan->x), "cuDNN input descriptor");
    dnn_check(cudnnCreateTensorDescriptor(&plan->y), "cuDNN output descriptor");
    dnn_check(cudnnCreateFilterDescriptor(&plan->w), "cuDNN filter descriptor");
    dnn_check(cudnnCreateConvolutionDescriptor(&plan->c), "cuDNN convolution descriptor");
    int xs[5] = {batch, in}, ys[5] = {batch, out}, ws[5] = {out, in}, pads[3], strides[3], ones[3];
    int xst[5], yst[5];
    for (int d = 0; d < dims; d++) {
        xs[2 + d] = size[d];
        ys[2 + d] = size[d] / (d < dims - 2 ? 1 : stride);
        ws[2 + d] = k;
        pads[d] = k / 2;
        strides[d] = d < dims - 2 ? 1 : stride;
        ones[d] = 1;
    }
    xst[dims + 1] = yst[dims + 1] = 1;
    for (int d = dims; d >= 0; d--) {
        xst[d] = xst[d + 1] * xs[d + 1];
        yst[d] = yst[d + 1] * ys[d + 1];
    }
    cudnnSetTensorNdDescriptor(plan->x, type, dims + 2, xs, xst);
    cudnnSetTensorNdDescriptor(plan->y, type, dims + 2, ys, yst);
    cudnnSetFilterNdDescriptor(plan->w, type, CUDNN_TENSOR_NCHW, dims + 2, ws);
    cudnnSetConvolutionNdDescriptor(
        plan->c, dims, pads, strides, ones, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionMathType(
        plan->c, type == CUDNN_DATA_HALF ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH);
    cudnnConvolutionFwdAlgoPerf_t found[8];
    int count = 0;
    dnn_check(cudnnGetConvolutionForwardAlgorithm_v7(
                  dnn, plan->x, plan->w, plan->c, plan->y, 8, &count, found),
        "cuDNN algorithm");
    int pick = 0;
    while (
        pick < count && (found[pick].status != CUDNN_STATUS_SUCCESS || found[pick].memory > room)) {
        pick++;
    }
    if (pick == count) {
        fprintf(stderr,
            "herculex: no cuDNN convolution for %d input and %d output channels "
            "at %dx%d with %zu workspace bytes\n",
            in, out, size[dims - 2], size[dims - 1], room);
        exit(1);
    }
    plan->algo = found[pick].algo;
    return plan;
}

void convolve(const void *x, int batch, int in, const int *size, int dims, const void *weight,
    int out, int k, int stride, void *y, cudnnDataType_t type, void *workspace, size_t room) {
    Plan *p = plan_of(batch, in, size, dims, out, k, stride, type, room);
    float one = 1, zero = 0;
    dnn_check(cudnnConvolutionForward(
                  dnn, &one, p->x, x, p->w, weight, p->c, p->algo, workspace, room, &zero, p->y, y),
        "cuDNN convolution");
}

__device__ u16 activate(u16 x, u16 bias) {
    return bits(leaky(value(bits(value(x) + value(bias)))));
}

enum { STEM_PIXELS = 32, STEM_WARPS = THREADS / STEM_PIXELS };

__global__ void stem_pool(const u16 *second, const u16 *bias, const u16 *attention,
    u16 attention_bias, u16 *features, long n) {
    __shared__ u16 samples[DEPTH][STEM][STEM_PIXELS];
    __shared__ float logits[DEPTH][STEM_PIXELS];
    int lane = threadIdx.x % STEM_PIXELS, warp = threadIdx.x / STEM_PIXELS;
    long i = blockIdx.x * (long)STEM_PIXELS + lane, plane = DEPTH * n;
    if (i < n) {
        for (int z = warp; z < DEPTH; z += STEM_WARPS) {
            float acc = 0;
            for (int c = 0; c < STEM; c++) {
                u16 a = activate(second[c * plane + z * n + i], bias[c]);
                samples[z][c][lane] = a;
                acc = __fmaf_rn(value(a), value(attention[c]), acc);
            }
            logits[z][lane] = value(bits(acc + value(attention_bias)));
        }
    }
    __syncthreads();
    if (!warp && i < n) {
        float top = -INFINITY, total = 0;
        for (int z = 0; z < DEPTH; z++) {
            top = fmaxf(top, logits[z][lane]);
        }
        for (int z = 0; z < DEPTH; z++) {
            logits[z][lane] = expf(logits[z][lane] - top);
            total += logits[z][lane];
        }
        for (int z = 0; z < DEPTH; z++) {
            logits[z][lane] = value(bits(logits[z][lane] / total));
        }
    }
    __syncthreads();
    if (i >= n) {
        return;
    }
    for (int c = warp; c < STEM; c += STEM_WARPS) {
        float pooled = 0, peak = -INFINITY;
        for (int z = 0; z < DEPTH; z++) {
            float b = value(samples[z][c][lane]);
            pooled += value(bits(b * logits[z][lane]));
            peak = fmaxf(peak, b);
        }
        features[c * n + i] = bits(pooled);
        features[(STEM + c) * n + i] = bits(peak);
    }
}

extern "C" void gpu_stem(const u8 *stack, int height, int width, int y0, int x0, int rows, int cols,
    const u16 *table, const u16 *w1, const u16 *b1, const u16 *w2, const u16 *b2,
    const u16 *attention, u16 attention_bias, u16 *image, u16 *first, u16 *second, u16 *features,
    int h, int w, void *workspace, size_t room) {
    long pixels = (long)h * w, n = DEPTH * pixels;
    int size[3] = {DEPTH, h, w};
    tile_grey<<<dim3((w + 31) / 32, (h + 7) / 8, DEPTH), dim3(32, 8)>>>(
        stack, height, width, y0, x0, rows, cols, table, image, h, w);
    convolve(image, 1, 1, size, 3, w1, STEM, 3, 1, first, CUDNN_DATA_HALF, workspace, room);
    bias_leaky<<<blocks(STEM * n), THREADS>>>(first, b1, n, STEM * n);
    convolve(first, 1, STEM, size, 3, w2, STEM, 3, 1, second, CUDNN_DATA_HALF, workspace, room);
    stem_pool<<<(pixels + STEM_PIXELS - 1) / STEM_PIXELS, THREADS>>>(
        second, b2, attention, attention_bias, features, pixels);
}

void gemm(const u16 *a, const u16 *b, float *c, int m, int n, int k, int transpose_a) {
    float one = 1, zero = 0;
    cublasStatus_t s = cublasGemmEx(blas, CUBLAS_OP_N, transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N, n,
        m, k, &one, b, CUDA_R_16F, n, a, CUDA_R_16F, transpose_a ? m : k, &zero, c, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "herculex: cublasGemmEx failed: %d\n", s);
        exit(1);
    }
}

extern "C" void gpu_conv(const u16 *x, int batch, int in, int h, int w, const u16 *weight, int out,
    int k, int stride, u16 *y, void *workspace, size_t room) {
    int size[2] = {h, w};
    convolve(x, batch, in, size, 2, weight, out, k, stride, y, CUDNN_DATA_HALF, workspace, room);
}

__global__ void up_scatter(
    const float *acc, const u16 *bias, const u16 *skip, u16 *y, int out, int h, int w) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long n = (long)h * w;
    if (i >= out * 4 * n) {
        return;
    }
    long row = i / n, at = i % n;
    int o = row / 4, ky = row % 4 / 2, kx = row % 2;
    int yy = at / w * 2 + ky, xx = at % w * 2 + kx;
    y[((long)o * 2 * h + yy) * 2 * w + xx] = bits(value(bits(acc[i])) + value(bias[o]));
    y[out * 4 * n + i] = skip[i];
}

extern "C" void gpu_up(const u16 *x, int in, int h, int w, const u16 *weight, const u16 *bias,
    int out, const u16 *skip, u16 *y, float *acc) {
    long n = (long)h * w;
    gemm(weight, x, acc, out * 4, n, in, 1);
    up_scatter<<<blocks(out * 4 * n), THREADS>>>(acc, bias, skip, y, out, h, w);
}

#define SPLITS 64

__device__ void combine(float *count, float *mean, float *m2, float n, float m, float q) {
    float total = *count + n;
    if (total == 0) {
        return;
    }
    float delta = m - *mean;
    *mean += delta * (n / total);
    *m2 += q + delta * delta * (*count * n / total);
    *count = total;
}

__device__ float2 norm_part(
    const u16 *row, float shift, long first, long last, float shared[3][THREADS / 32]) {
    float count = 0, mean = 0, m2 = 0;
    for (long i = first + threadIdx.x; i < last; i += THREADS) {
        float v = value(bits(value(row[i]) + shift));
        count += 1;
        float delta = v - mean;
        mean += delta / count;
        m2 += delta * (v - mean);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        float n2 = __shfl_down_sync(0xffffffff, count, offset);
        float m = __shfl_down_sync(0xffffffff, mean, offset);
        float q = __shfl_down_sync(0xffffffff, m2, offset);
        combine(&count, &mean, &m2, n2, m, q);
    }
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (!lane) {
        shared[0][warp] = count;
        shared[1][warp] = mean;
        shared[2][warp] = m2;
    }
    __syncthreads();
    if (threadIdx.x) {
        return {};
    }
    count = mean = m2 = 0;
    for (int k = 0; k < THREADS / 32; k++) {
        combine(&count, &mean, &m2, shared[0][k], shared[1][k], shared[2][k]);
    }
    return {mean, m2};
}

__global__ void norm_parts(
    const u16 *x, const u16 *bias, int channels, long n, int splits, float *parts) {
    __shared__ float shared[3][THREADS / 32];
    long c = blockIdx.x, part = blockIdx.y;
    float2 result = norm_part(x + c * n, bias ? value(bias[c % channels]) : 0, n * part / splits,
        n * (part + 1) / splits, shared);
    if (!threadIdx.x) {
        parts[(c * SPLITS + part) * 2] = result.x;
        parts[(c * SPLITS + part) * 2 + 1] = result.y;
    }
}

__device__ void merge_norm(float size, float2 part, float &count, float &mean, float &m2) {
    float delta = part.x - mean;
    float total = count + size;
    mean += delta * (size / total);
    m2 += part.y + delta * delta * (count * size / total);
    count = total;
}

__global__ void norm_merge(const float *parts, long n, int splits, int channels, float *stats) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    float count = 0, mean = 0, m2 = 0;
    for (int p = 0; p < splits; p++) {
        float size = n * (p + 1) / splits - n * p / splits;
        const float *part = parts + (c * SPLITS + p) * 2;
        merge_norm(size, {part[0], part[1]}, count, mean, m2);
    }
    stats[2 * c] = mean;
    stats[2 * c + 1] = rsqrtf((float)(m2 / n) + 1e-5f);
}

__device__ u16 norm_value(u16 x, float mean, float scale, float gamma, float beta,
    const u16 *residual, long i, int nonlinear) {
    float v = value(bits(__fadd_rn(__fmul_rn(__fmul_rn(value(x) - mean, scale), gamma), beta)));
    if (residual) {
        v = value(bits(v + value(residual[i])));
    }
    return bits(nonlinear ? leaky(v) : v);
}

__global__ void norm_apply(const u16 *x, const float *stats, const u16 *gamma, const u16 *beta,
    const u16 *residual, int nonlinear, u16 *y, int channels, long n) {
    long c = blockIdx.y;
    int k = c % channels;
    float m = stats[2 * c], scale = stats[2 * c + 1];
    float g = value(gamma[k]), b = value(beta[k]);
    for (long i = blockIdx.x * (long)THREADS + threadIdx.x; i < n; i += (long)gridDim.x * THREADS) {
        long at = c * n + i;
        y[at] = norm_value(x[at], m, scale, g, b, residual, at, nonlinear);
    }
}

__global__ void norm_small(const u16 *x, const u16 *bias, const u16 *gamma, const u16 *beta,
    const u16 *residual, int nonlinear, u16 *y, int channels, long n) {
    __shared__ float warps[3][THREADS / 32];
    __shared__ float stats[2];
    long c = blockIdx.x, i = threadIdx.x;
    int k = c % channels;
    float2 part = norm_part(x + c * n, bias ? value(bias[k]) : 0, 0, n, warps);
    if (!i) {
        float count = 0, mean = 0, m2 = 0;
        merge_norm(n, part, count, mean, m2);
        stats[0] = mean;
        stats[1] = rsqrtf((float)(m2 / n) + 1e-5f);
    }
    __syncthreads();
    if (i < n) {
        long at = c * n + i;
        y[at] = norm_value(
            x[at], stats[0], stats[1], value(gamma[k]), value(beta[k]), residual, at, nonlinear);
    }
}

float *norm_scratch;

extern "C" void gpu_norm(const u16 *x, const u16 *bias, int channels, int batch, long n,
    const u16 *gamma, const u16 *beta, const u16 *residual, int nonlinear, u16 *y) {
    int rows = channels * batch;
    if (n <= THREADS) {
        norm_small<<<rows, THREADS>>>(x, bias, gamma, beta, residual, nonlinear, y, channels, n);
        return;
    }
    if (!norm_scratch) {
        check(cudaMalloc(&norm_scratch, 4096 * (SPLITS + 1) * 2 * sizeof(float)), "cudaMalloc");
    }
    float *stats = norm_scratch + 4096 * SPLITS * 2;
    int splits = n / 4096 < 1 ? 1 : n / 4096 > SPLITS ? SPLITS : n / 4096;
    norm_parts<<<dim3(rows, splits), THREADS>>>(x, bias, channels, n, splits, norm_scratch);
    norm_merge<<<blocks(rows), THREADS>>>(norm_scratch, n, splits, rows, stats);
    int spread = blocks(n) < 64 ? blocks(n) : 64;
    norm_apply<<<dim3(spread, rows), THREADS>>>(
        x, stats, gamma, beta, residual, nonlinear, y, channels, n);
}

__global__ void pool(const u16 *x, int channels, int h, int w, u16 *y) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    int ho = h / 2, wo = w / 2;
    if (i >= (long)channels * ho * wo) {
        return;
    }
    long c = i / ((long)ho * wo);
    int yy = i / wo % ho * 2, xx = i % wo * 2;
    const u16 *p = x + (c * h + yy) * w + xx;
    y[i] = bits((value(p[0]) + value(p[1]) + value(p[w]) + value(p[w + 1])) / 4);
}

extern "C" void gpu_pool(const u16 *x, int channels, int h, int w, u16 *y) {
    pool<<<blocks((long)channels * h * w / 4), THREADS>>>(x, channels, h, w, y);
}

__global__ void sigmoid(const u16 *logits, const u16 *bias, float *read, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float biased = value(bits(value(logits[i]) + value(*bias)));
        read[i] = 1 / (1 + expf(-biased));
    }
}

extern "C" void gpu_ink(const u16 *logits, const u16 *bias, float *read, long n) {
    sigmoid<<<blocks(n), THREADS>>>(logits, bias, read, n);
}

__global__ void moment_parts(const float *x, long n, long volume, double *parts) {
    __shared__ double sums[THREADS];
    int c = blockIdx.x, part = blockIdx.y;
    long first = n * part / SPLITS, last = n * (part + 1) / SPLITS;
    double sum = 0;
    for (long i = first + threadIdx.x; i < last; i += THREADS) {
        sum += x[(i / volume * STEM + c) * volume + i % volume];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            sums[threadIdx.x] += sums[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        parts[c * SPLITS + part] = sums[0];
    }
}

__global__ void square_parts(
    const float *x, long n, long volume, const double *means, double *parts) {
    __shared__ double sums[THREADS];
    int c = blockIdx.x, part = blockIdx.y;
    long first = n * part / SPLITS, last = n * (part + 1) / SPLITS;
    double sum = 0, m = means[c];
    for (long i = first + threadIdx.x; i < last; i += THREADS) {
        double d = x[(i / volume * STEM + c) * volume + i % volume] - m;
        sum += d * d;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            sums[threadIdx.x] += sums[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        parts[c * SPLITS + part] = sums[0];
    }
}

double *moment_scratch;

__global__ void moment_merge(const double *parts, long n, double *means, float *out) {
    int c = threadIdx.x;
    double sum = 0;
    for (int p = 0; p < SPLITS; p++) {
        sum += parts[c * SPLITS + p];
    }
    double result = __ddiv_rn(sum, (double)n);
    if (means) {
        means[c] = result;
    }
    out[c] = (float)result;
}

void moments(const float *x, long n, long volume, float *mean, float *variance) {
    if (!moment_scratch) {
        check(cudaMalloc(&moment_scratch,
                  (STEM * SPLITS + STEM) * sizeof(double) + 2 * STEM * sizeof(float)),
            "cudaMalloc");
    }
    double *parts = moment_scratch, *means = parts + STEM * SPLITS;
    float *stats = (float *)(means + STEM), host[2 * STEM];
    moment_parts<<<dim3(STEM, SPLITS), THREADS>>>(x, n, volume, parts);
    moment_merge<<<1, STEM>>>(parts, n, means, stats);
    square_parts<<<dim3(STEM, SPLITS), THREADS>>>(x, n, volume, means, parts);
    moment_merge<<<1, STEM>>>(parts, n, NULL, stats + STEM);
    check(cudaMemcpy(host, stats, sizeof(host), cudaMemcpyDeviceToHost), "moments");
    memcpy(mean, host, STEM * sizeof(float));
    memcpy(variance, host + STEM, STEM * sizeof(float));
}

__global__ void blend(const float *read, const float *window, int y0, int x0, int th, int tw,
    int tx, float *total, float *weight, int width) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)th * tw) {
        return;
    }
    int y = i / tw, x = i % tw;
    float b = window[y * tx + x];
    long at = (long)(y0 + y) * width + x0 + x;
    total[at] = __fadd_rn(total[at], __fmul_rn(read[y * tx + x], b));
    weight[at] = __fadd_rn(weight[at], b);
}

extern "C" void gpu_blend(const float *read, const float *window, int y0, int x0, int th, int tw,
    int tx, float *total, float *weight, int width) {
    blend<<<blocks((long)th * tw), THREADS>>>(
        read, window, y0, x0, th, tw, tx, total, weight, width);
}

__global__ void finish(float *total, const float *weight, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        total[i] = total[i] / fmaxf(weight[i], 1e-6f);
    }
}

extern "C" void gpu_finish(float *total, const float *weight, long n, float *host) {
    finish<<<blocks(n), THREADS>>>(total, weight, n);
    check(cudaMemcpy(host, total, n * sizeof(float), cudaMemcpyDeviceToHost), "ink map");
}

extern "C" void gpu_zero(void *p, size_t n) { check(cudaMemset(p, 0, n), "memset"); }

__global__ void grey_float(const u8 *tiles, const float *table, float *image, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        image[i] = table[tiles[i]];
    }
}

__global__ void norm_leaky(float *x, const float *norm, long volume, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        int c = i / volume % STEM;
        x[i] = leaky(__fadd_rn(__fmul_rn(x[i], norm[c]), norm[STEM + c]));
    }
}

extern "C" void gpu_measure(const u8 *tiles, const float *table, const float *conv1,
    const float *conv2, float *image, float *first, float *second, int count, int h, int w,
    const float *norm, float *mean, float *variance, int stage, void *workspace, size_t room) {
    long volume = (long)DEPTH * h * w, n = count * volume;
    int size[3] = {DEPTH, h, w};
    if (stage == 1) {
        grey_float<<<blocks(n), THREADS>>>(tiles, table, image, n);
        convolve(
            image, count, 1, size, 3, conv1, STEM, 3, 1, first, CUDNN_DATA_FLOAT, workspace, room);
        moments(first, n, volume, mean, variance);
        return;
    }
    norm_leaky<<<blocks(n * STEM), THREADS>>>(first, norm, volume, n * STEM);
    convolve(
        first, count, STEM, size, 3, conv2, STEM, 3, 1, second, CUDNN_DATA_FLOAT, workspace, room);
    moments(second, n, volume, mean, variance);
}
