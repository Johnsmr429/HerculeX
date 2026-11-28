#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const float *key;
    float *weight;
    cudnnTensorDescriptor_t x, y;
    cudnnFilterDescriptor_t w;
    cudnnConvolutionDescriptor_t conv;
    cudnnConvolutionFwdAlgo_t algorithm;
} SliceCudaPlan;

typedef struct {
    SliceCudaPlan plans[160];
    cudnnHandle_t dnn;
    void *workspace;
    float *input, *output;
    double *noise;
    size_t input_bytes, output_bytes, noise_bytes;
    uint64_t seed, offset;
    int count, grid_limit;
} SliceCudaModel;

enum { workspace_bytes = 128u << 20 };

static void slice_cuda_check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "herculex: Synthetic-Slices CUDA %s: %s\n", operation,
            cudaGetErrorString(status));
        exit(1);
    }
}

static void slice_cudnn_check(cudnnStatus_t status, const char *operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "herculex: Synthetic-Slices cuDNN %s: %s\n", operation,
            cudnnGetErrorString(status));
        exit(1);
    }
}

extern "C" int slices_accelerator_available(void) {
    int devices;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

extern "C" void *slices_accelerator_open(uint64_t seed) {
    SliceCudaModel *model = (SliceCudaModel *)calloc(1, sizeof(*model));
    if (!model) {
        return NULL;
    }
    int device;
    cudaDeviceProp props;
    slice_cuda_check(cudaGetDevice(&device), "device");
    slice_cuda_check(cudaGetDeviceProperties(&props, device), "device properties");
    model->seed = seed;
    model->grid_limit = props.multiProcessorCount * (props.maxThreadsPerMultiProcessor / 256);
    slice_cudnn_check(cudnnCreate(&model->dnn), "create handle");
    slice_cuda_check(cudaMalloc(&model->workspace, workspace_bytes), "allocate workspace");
    return model;
}

extern "C" void slices_accelerator_close(void *ptr) {
    SliceCudaModel *model = (SliceCudaModel *)ptr;
    for (int i = 0; i < model->count; i++) {
        SliceCudaPlan *plan = &model->plans[i];
        cudaFree(plan->weight);
        cudnnDestroyTensorDescriptor(plan->x);
        cudnnDestroyTensorDescriptor(plan->y);
        cudnnDestroyFilterDescriptor(plan->w);
        cudnnDestroyConvolutionDescriptor(plan->conv);
    }
    cudaFree(model->input);
    cudaFree(model->output);
    cudaFree(model->noise);
    cudaFree(model->workspace);
    cudnnDestroy(model->dnn);
    free(model);
}

static void *slices_cuda_buffer(void *data, size_t *bytes, size_t next) {
    if (next > *bytes) {
        slice_cuda_check(cudaFree(data), "free buffer");
        slice_cuda_check(cudaMalloc(&data, next), "allocate buffer");
        *bytes = next;
    }
    return data;
}

static SliceCudaPlan *slices_cuda_plan(
    SliceCudaModel *model, const float *weights, int ci, int co, int side, int kernel, int stride) {
    for (int i = 0; i < model->count; i++) {
        if (model->plans[i].key == weights) {
            return &model->plans[i];
        }
    }
    if (model->count == (int)(sizeof(model->plans) / sizeof(model->plans[0]))) {
        fprintf(stderr, "herculex: too many Synthetic-Slices CUDA convolution plans\n");
        exit(1);
    }
    SliceCudaPlan *plan = &model->plans[model->count++];
    plan->key = weights;
    size_t bytes = (size_t)ci * co * kernel * kernel * sizeof(float);
    slice_cuda_check(cudaMalloc(&plan->weight, bytes), "allocate weights");
    slice_cuda_check(
        cudaMemcpy(plan->weight, weights, bytes, cudaMemcpyHostToDevice), "copy weights");
    slice_cudnn_check(cudnnCreateTensorDescriptor(&plan->x), "create input descriptor");
    slice_cudnn_check(cudnnCreateTensorDescriptor(&plan->y), "create output descriptor");
    slice_cudnn_check(cudnnCreateFilterDescriptor(&plan->w), "create filter descriptor");
    slice_cudnn_check(
        cudnnCreateConvolutionDescriptor(&plan->conv), "create convolution descriptor");
    int out_side = side / stride;
    slice_cudnn_check(
        cudnnSetTensor4dDescriptor(plan->x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, ci, side, side),
        "set input descriptor");
    slice_cudnn_check(cudnnSetTensor4dDescriptor(
                          plan->y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, co, out_side, out_side),
        "set output descriptor");
    slice_cudnn_check(cudnnSetFilter4dDescriptor(
                          plan->w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, co, ci, kernel, kernel),
        "set filter descriptor");
    int pad = kernel / 2;
    slice_cudnn_check(cudnnSetConvolution2dDescriptor(plan->conv, pad, pad, stride, stride, 1, 1,
                          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "set convolution descriptor");
    slice_cudnn_check(cudnnSetConvolutionMathType(plan->conv, CUDNN_FMA_MATH), "FP32 precision");
    cudnnConvolutionFwdAlgoPerf_t algorithms[8];
    int n;
    slice_cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(
                          model->dnn, plan->x, plan->w, plan->conv, plan->y, 8, &n, algorithms),
        "choose convolution algorithm");
    for (int i = 0; i < n; i++) {
        if (algorithms[i].status == CUDNN_STATUS_SUCCESS &&
            algorithms[i].memory <= workspace_bytes) {
            plan->algorithm = algorithms[i].algo;
            return plan;
        }
    }
    fprintf(stderr, "herculex: no Synthetic-Slices CUDA convolution algorithm fits\n");
    exit(1);
}

extern "C" void slices_accelerator_conv(void *ptr, const float *weights, const float *input,
    float *out, int ci, int co, int side, int kernel, int stride) {
    SliceCudaModel *model = (SliceCudaModel *)ptr;
    SliceCudaPlan *plan = slices_cuda_plan(model, weights, ci, co, side, kernel, stride);
    int out_side = side / stride;
    size_t in_bytes = (size_t)ci * side * side * sizeof(float);
    size_t out_bytes = (size_t)co * out_side * out_side * sizeof(float);
    model->input = (float *)slices_cuda_buffer(model->input, &model->input_bytes, in_bytes);
    model->output = (float *)slices_cuda_buffer(model->output, &model->output_bytes, out_bytes);
    slice_cuda_check(
        cudaMemcpy(model->input, input, in_bytes, cudaMemcpyHostToDevice), "copy input");
    float alpha = 1, beta = 0;
    slice_cudnn_check(cudnnConvolutionForward(model->dnn, &alpha, plan->x, model->input, plan->w,
                          plan->weight, plan->conv, plan->algorithm, model->workspace,
                          workspace_bytes, &beta, plan->y, model->output),
        "convolution");
    slice_cuda_check(
        cudaMemcpy(out, model->output, out_bytes, cudaMemcpyDeviceToHost), "copy output");
}

__global__ void slices_noise_kernel(double *out, size_t n, uint64_t seed, uint64_t offset) {
    size_t thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread, offset, &state);
    for (size_t i = thread; i < n; i += 2 * stride) {
        double2 sample = curand_normal2_double(&state);
        out[i] = sample.x;
        if (i + stride < n) {
            out[i + stride] = sample.y;
        }
    }
}

extern "C" void slices_accelerator_noise(void *ptr, double *out, size_t n) {
    SliceCudaModel *model = (SliceCudaModel *)ptr;
    size_t blocks = std::min((n + 255) / 256, (size_t)model->grid_limit);
    size_t bytes = n * sizeof(*out);
    model->noise = (double *)slices_cuda_buffer(model->noise, &model->noise_bytes, bytes);
    slices_noise_kernel<<<blocks, 256>>>(model->noise, n, model->seed, model->offset);
    slice_cuda_check(cudaGetLastError(), "draw sampler noise");
    model->offset += 4 * ((n - 1) / (2 * blocks * 256) + 1);
    slice_cuda_check(
        cudaMemcpy(out, model->noise, bytes, cudaMemcpyDeviceToHost), "copy sampler noise");
}
