#include "studio_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "studio_cuda: %s: %s\n", #x, cudaGetErrorString(e_));                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

__device__ __constant__ float c_BT4[6][6] = {
    {0.249230769f, 0.000000000f, -0.553846154f, 0.000000000f, 0.196923077f, 0.000000000f},
    {0.000000000f, -0.296703297f, -0.395604396f, 0.131868132f, 0.175824176f, 0.000000000f},
    {0.000000000f, 0.296703297f, -0.395604396f, -0.131868132f, 0.175824176f, 0.000000000f},
    {0.000000000f, -0.216000000f, -0.144000000f, 0.384000000f, 0.256000000f, 0.000000000f},
    {0.000000000f, 0.216000000f, -0.144000000f, -0.384000000f, 0.256000000f, 0.000000000f},
    {0.000000000f, 0.249230769f, 0.000000000f, -0.553846154f, 0.000000000f, 0.196923077f},
};

__device__ __constant__ float c_AT4[4][6] = {
    {5.078125000f, 5.687500000f, 5.687500000f, 3.906250000f, 3.906250000f, 0.000000000f},
    {0.000000000f, 4.265625000f, -4.265625000f, 5.859375000f, -5.859375000f, 0.000000000f},
    {0.000000000f, 3.199218750f, 3.199218750f, 8.789062500f, 8.789062500f, 0.000000000f},
    {0.000000000f, 2.399414062f, -2.399414062f, 13.183593750f, -13.183593750f, 5.078125000f},
};

__device__ __constant__ float c_G4[6][3] = {
    {0.790123457f, 0.000000000f, 0.000000000f},
    {-0.526748971f, -0.395061728f, -0.296296296f},
    {-0.526748971f, 0.395061728f, -0.296296296f},
    {0.131687243f, 0.197530864f, 0.296296296f},
    {0.131687243f, -0.197530864f, 0.296296296f},
    {0.000000000f, 0.000000000f, 1.000000000f},
};

__device__ __forceinline__ unsigned su32(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void st_smem16(unsigned s, uint4 v) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(s), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void st_cs_u32(void *p, unsigned v) {
    asm volatile("st.global.cs.u32 [%0], %1;\n" ::"l"(p), "r"(v));
}

__device__ __forceinline__ void st_cs_u4(void *p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y),
        "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void cp16(unsigned s, const void *g, bool ok) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 16 : 0));
#else
    uint4 v = ok ? *reinterpret_cast<const uint4 *>(g) : make_uint4(0, 0, 0, 0);
    st_smem16(s, v);
#endif
}

__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_wait0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}
__device__ __forceinline__ void cp_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void ldsm(unsigned *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}

__device__ __forceinline__ void mma_f16(float *c, const unsigned *a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

__device__ __forceinline__ unsigned sad64(unsigned base, int row, int chunk) {
    return base + row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ unsigned pack_h2(float a, float b) {
    __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<unsigned *>(&v);
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ int reflect_idx(int i, int n) {
    if (n <= 1)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return i;
}

__device__ __forceinline__ void bt4_1d(const float d[6], float o[6]) {
    o[0] = c_BT4[0][0] * d[0] + c_BT4[0][2] * d[2] + c_BT4[0][4] * d[4];
    o[1] = c_BT4[1][1] * d[1] + c_BT4[1][2] * d[2] + c_BT4[1][3] * d[3] + c_BT4[1][4] * d[4];
    o[2] = c_BT4[2][1] * d[1] + c_BT4[2][2] * d[2] + c_BT4[2][3] * d[3] + c_BT4[2][4] * d[4];
    o[3] = c_BT4[3][1] * d[1] + c_BT4[3][2] * d[2] + c_BT4[3][3] * d[3] + c_BT4[3][4] * d[4];
    o[4] = c_BT4[4][1] * d[1] + c_BT4[4][2] * d[2] + c_BT4[4][3] * d[3] + c_BT4[4][4] * d[4];
    o[5] = c_BT4[5][1] * d[1] + c_BT4[5][3] * d[3] + c_BT4[5][5] * d[5];
}

__device__ __forceinline__ void at4_1d(const float m[6], float o[4]) {
    float s12 = m[1] + m[2], d12 = m[1] - m[2];
    float s34 = m[3] + m[4], d34 = m[3] - m[4];
    o[0] = c_AT4[0][0] * m[0] + c_AT4[0][1] * s12 + c_AT4[0][3] * s34;
    o[1] = c_AT4[1][1] * d12 + c_AT4[1][3] * d34;
    o[2] = c_AT4[2][1] * s12 + c_AT4[2][3] * s34;
    o[3] = c_AT4[3][1] * d12 + c_AT4[3][3] * d34 + c_AT4[3][5] * m[5];
}

__device__ __forceinline__ void bt2_1d(const float d[4], float o[4]) {
    o[0] = d[0] - d[2];
    o[1] = d[1] + d[2];
    o[2] = d[2] - d[1];
    o[3] = d[1] - d[3];
}

__device__ __forceinline__ void at2_1d(const float m[4], float o[2]) {
    o[0] = m[0] + m[1] + m[2];
    o[1] = m[1] - m[2] - m[3];
}

__global__ void k_transform_wino_w(
    const float *w_oizyx, int Cout, int Cin, int alpha, float *u_f32, float *plane_max) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)Cout * Cin;
    if (idx >= total)
        return;
    int co = (int)(idx / Cin), ci = (int)(idx % Cin);
    const float *w = w_oizyx + idx * 27;
    if (alpha == 4) {
        float t1[4][3][3];
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                float g0 = w[0 * 9 + ky * 3 + kx], g1 = w[1 * 9 + ky * 3 + kx],
                      g2 = w[2 * 9 + ky * 3 + kx];
                t1[0][ky][kx] = g0;
                t1[1][ky][kx] = 0.5f * (g0 + g1 + g2);
                t1[2][ky][kx] = 0.5f * (g0 - g1 + g2);
                t1[3][ky][kx] = g2;
            }
        float t2[4][4][3];
        for (int pz = 0; pz < 4; ++pz)
            for (int kx = 0; kx < 3; ++kx) {
                float g0 = t1[pz][0][kx], g1 = t1[pz][1][kx], g2 = t1[pz][2][kx];
                t2[pz][0][kx] = g0;
                t2[pz][1][kx] = 0.5f * (g0 + g1 + g2);
                t2[pz][2][kx] = 0.5f * (g0 - g1 + g2);
                t2[pz][3][kx] = g2;
            }
        for (int pz = 0; pz < 4; ++pz)
            for (int py = 0; py < 4; ++py) {
                float g0 = t2[pz][py][0], g1 = t2[pz][py][1], g2 = t2[pz][py][2];
                float v[4] = {g0, 0.5f * (g0 + g1 + g2), 0.5f * (g0 - g1 + g2), g2};
                for (int px = 0; px < 4; ++px) {
                    int p = (pz * 4 + py) * 4 + px;
                    u_f32[(long)p * total + (long)co * Cin + ci] = v[px];
                    unsigned int bits = __float_as_uint(fabsf(v[px]));
                    atomicMax((unsigned int *)&plane_max[p], bits);
                }
            }
    } else {
        float t1[6][3][3];
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                float g0 = w[0 * 9 + ky * 3 + kx], g1 = w[1 * 9 + ky * 3 + kx],
                      g2 = w[2 * 9 + ky * 3 + kx];
                for (int pz = 0; pz < 6; ++pz)
                    t1[pz][ky][kx] = c_G4[pz][0] * g0 + c_G4[pz][1] * g1 + c_G4[pz][2] * g2;
            }
        float t2[6][6][3];
        for (int pz = 0; pz < 6; ++pz)
            for (int kx = 0; kx < 3; ++kx) {
                float g0 = t1[pz][0][kx], g1 = t1[pz][1][kx], g2 = t1[pz][2][kx];
                for (int py = 0; py < 6; ++py)
                    t2[pz][py][kx] = c_G4[py][0] * g0 + c_G4[py][1] * g1 + c_G4[py][2] * g2;
            }
        for (int pz = 0; pz < 6; ++pz)
            for (int py = 0; py < 6; ++py) {
                float g0 = t2[pz][py][0], g1 = t2[pz][py][1], g2 = t2[pz][py][2];
                for (int px = 0; px < 6; ++px) {
                    float v = c_G4[px][0] * g0 + c_G4[px][1] * g1 + c_G4[px][2] * g2;
                    int p = (pz * 6 + py) * 6 + px;
                    u_f32[(long)p * total + (long)co * Cin + ci] = v;
                    unsigned int bits = __float_as_uint(fabsf(v));
                    atomicMax((unsigned int *)&plane_max[p], bits);
                }
            }
    }
}

__global__ void k_transform_wino2d_w(
    const float *w_oizyx, int Cout, int Cin, float *u_f32, float *plane_max) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)Cout * Cin;
    if (idx >= total)
        return;
    int co = (int)(idx / Cin), ci = (int)(idx % Cin);
    const float *w = w_oizyx + idx * 27;
    long p_stride = 3 * total;
    for (int dz = 0; dz < 3; ++dz) {
        float t1[6][3];
        for (int kx = 0; kx < 3; ++kx) {
            float g0 = w[dz * 9 + 0 * 3 + kx];
            float g1 = w[dz * 9 + 1 * 3 + kx];
            float g2 = w[dz * 9 + 2 * 3 + kx];
            for (int py = 0; py < 6; ++py)
                t1[py][kx] = c_G4[py][0] * g0 + c_G4[py][1] * g1 + c_G4[py][2] * g2;
        }
        for (int py = 0; py < 6; ++py) {
            float g0 = t1[py][0], g1 = t1[py][1], g2 = t1[py][2];
            for (int px = 0; px < 6; ++px) {
                float v = c_G4[px][0] * g0 + c_G4[px][1] * g1 + c_G4[px][2] * g2;
                int p = py * 6 + px;
                u_f32[(long)p * p_stride + (long)dz * total + (long)co * Cin + ci] = v;
                unsigned int bits = __float_as_uint(fabsf(v));
                atomicMax((unsigned int *)&plane_max[p], bits);
            }
        }
    }
}

__global__ void k_quantize_wino_w(const float *u_f32, const float *plane_max, int planes,
    long stride, __half *u_f16, float *u_scale) {
    (void)planes;
    int p = blockIdx.y;
    float amax = plane_max[p];
    float sc = fmaxf(amax / 16.0f, 1e-8f);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        u_scale[p] = sc;
    float inv = 1.0f / sc;
    long base = (long)p * stride;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < stride;
        i += (long)gridDim.x * blockDim.x)
        u_f16[base + i] = __float2half_rn(u_f32[base + i] * inv);
}

__global__ void k_reorder_direct_w(const float *w_oizyx, int Cout, int Cin, __half *w_taps) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)Cout * Cin * 27;
    if (idx >= total)
        return;
    int tap = (int)(idx % 27);
    long t = idx / 27;
    int ci = (int)(t % Cin);
    int co = (int)(t / Cin);
    w_taps[((long)tap * Cout + co) * Cin + ci] = __float2half_rn(w_oizyx[idx]);
}

__global__ void k_f32_to_f16(const float *src, __half *dst, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half_rn(src[i]);
}

__global__ void k_gn_finalize_f32(const float *acc, float *params, int BG, float count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BG)
        return;
    float mean = acc[2 * i] / count;
    float var = acc[2 * i + 1] / count - mean * mean;
    if (var < 0.0f)
        var = 0.0f;
    params[2 * i] = mean;
    params[2 * i + 1] = rsqrtf(var + 1e-5f);
}

__global__ void k_gn_stats_f16(const __half *x, float *gn_acc, int B, int DHW, int C, int G) {
    __shared__ float sm_s0[192], sm_q0[192], sm_s1[192], sm_q1[192];
    __shared__ int sm_g0[192], sm_g1[192];
    int b = blockIdx.y;
    int chunk = blockIdx.x;
    int nchunks = gridDim.x;
    int tid = threadIdx.x;
    int cpg = C / G;
    int half_c = C >> 1;
    const __half2 *xb = reinterpret_cast<const __half2 *>(x + (long)b * DHW * C);

    int pix_per_chunk = (DHW + nchunks - 1) / nchunks;
    int p0 = chunk * pix_per_chunk;
    int p1 = min(DHW, p0 + pix_per_chunk);

    if (half_c <= 192) {
        int pl = tid / half_c;
        int c2 = tid % half_c;
        int stride_p = 192 / half_c;
        int c0 = c2 * 2;
        float fs0 = 0.f, fq0 = 0.f, fs1 = 0.f, fq1 = 0.f;
        if (pl < stride_p) {
            for (int p = p0 + pl; p < p1; p += stride_p) {
                float2 f = __half22float2(xb[(long)p * half_c + c2]);
                fs0 += f.x;
                fq0 = fmaf(f.x, f.x, fq0);
                fs1 += f.y;
                fq1 = fmaf(f.y, f.y, fq1);
            }
        }
        sm_s0[tid] = fs0;
        sm_q0[tid] = fq0;
        sm_s1[tid] = fs1;
        sm_q1[tid] = fq1;
        sm_g0[tid] = pl < stride_p ? (c0 / cpg) : -1;
        sm_g1[tid] = pl < stride_p ? ((c0 + 1) / cpg) : -1;
        __syncthreads();
        if (tid < G) {
            float gs = 0.f, gq = 0.f;
            for (int i = 0; i < 192; ++i) {
                if (sm_g0[i] == tid) {
                    gs += sm_s0[i];
                    gq += sm_q0[i];
                }
                if (sm_g1[i] == tid) {
                    gs += sm_s1[i];
                    gq += sm_q1[i];
                }
            }
            atomicAdd(&gn_acc[(b * G + tid) * 2], gs);
            atomicAdd(&gn_acc[(b * G + tid) * 2 + 1], gq);
        }
    } else {
        int reps = (half_c + 191) / 192;
        for (int r = 0; r < reps; ++r) {
            int c2 = r * 192 + tid;
            float fs0 = 0.f, fq0 = 0.f, fs1 = 0.f, fq1 = 0.f;
            int g0 = -1, g1 = -1;
            if (c2 < half_c) {
                int ch = c2 * 2;
                g0 = ch / cpg;
                g1 = (ch + 1) / cpg;
                for (int p = p0; p < p1; ++p) {
                    float2 f = __half22float2(xb[(long)p * half_c + c2]);
                    fs0 += f.x;
                    fq0 = fmaf(f.x, f.x, fq0);
                    fs1 += f.y;
                    fq1 = fmaf(f.y, f.y, fq1);
                }
            }
            sm_s0[tid] = fs0;
            sm_q0[tid] = fq0;
            sm_s1[tid] = fs1;
            sm_q1[tid] = fq1;
            sm_g0[tid] = g0;
            sm_g1[tid] = g1;
            __syncthreads();
            if (tid < G) {
                float gs = 0.f, gq = 0.f;
                for (int i = 0; i < 192; ++i) {
                    if (sm_g0[i] == tid) {
                        gs += sm_s0[i];
                        gq += sm_q0[i];
                    }
                    if (sm_g1[i] == tid) {
                        gs += sm_s1[i];
                        gq += sm_q1[i];
                    }
                }
                if (gs != 0.f || gq != 0.f) {
                    atomicAdd(&gn_acc[(b * G + tid) * 2], gs);
                    atomicAdd(&gn_acc[(b * G + tid) * 2 + 1], gq);
                }
            }
            __syncthreads();
        }
    }
}

__global__ void k_gn_silu_f16(const __half *x, const float *gn_params, const float *gamma,
    const float *beta, __half *y, int B, int DHW, int C, int G) {
    __shared__ float s_sc[768], s_sh[768];
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int cpg = C / G;
    for (int c = tid; c < C; c += 192) {
        int g = c / cpg;
        float m = gn_params[(b * G + g) * 2];
        float is = gn_params[(b * G + g) * 2 + 1];
        float sc = is * gamma[c];
        s_sc[c] = sc;
        s_sh[c] = beta[c] - m * sc;
    }
    __syncthreads();

    int half_c = C >> 1;
    const __half2 *xb = reinterpret_cast<const __half2 *>(x + (long)b * DHW * C);
    __half2 *yb = reinterpret_cast<__half2 *>(y + (long)b * DHW * C);

    int chunks = gridDim.x;
    int pix_per_chunk = (DHW + chunks - 1) / chunks;
    int p0 = blockIdx.x * pix_per_chunk;
    int p1 = min(DHW, p0 + pix_per_chunk);

    if (half_c <= 192) {
        int stride_p = 192 / half_c;
        int pl = tid / half_c;
        int c2 = tid % half_c;
        if (pl >= stride_p)
            return;
        int c0 = c2 << 1;
        float sc0 = s_sc[c0], sh0 = s_sh[c0];
        float sc1 = s_sc[c0 + 1], sh1 = s_sh[c0 + 1];
        for (int p = p0 + pl; p < p1; p += stride_p) {
            float2 f = __half22float2(xb[(long)p * half_c + c2]);
            float v0 = silu_f(fmaf(f.x, sc0, sh0));
            float v1 = silu_f(fmaf(f.y, sc1, sh1));
            yb[(long)p * half_c + c2] = __floats2half2_rn(v0, v1);
        }
    } else {
        for (int p = p0; p < p1; ++p) {
            for (int c2 = tid; c2 < half_c; c2 += 192) {
                int c0 = c2 << 1;
                float2 f = __half22float2(xb[(long)p * half_c + c2]);
                float v0 = silu_f(fmaf(f.x, s_sc[c0], s_sh[c0]));
                float v1 = silu_f(fmaf(f.y, s_sc[c0 + 1], s_sh[c0 + 1]));
                yb[(long)p * half_c + c2] = __floats2half2_rn(v0, v1);
            }
        }
    }
}

__global__ void k_crop_tiles_u8(
    const uint8_t *vol, int H, int W, const int *coords, int B, float *out_bdhw) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * 16 * 128 * 128;
    if (idx >= total)
        return;
    int x = (int)(idx & 127);
    int y = (int)((idx >> 7) & 127);
    int z = (int)((idx >> 14) & 15);
    int b = (int)(idx >> 18);
    int sy = reflect_idx(coords[2 * b] - 16 + y, H);
    int sx = reflect_idx(coords[2 * b + 1] - 16 + x, W);
    uint8_t raw = vol[((long)z * H + sy) * W + sx];
    float v = raw > 200 ? 200.0f : (float)raw;
    out_bdhw[idx] = v * (1.0f / 255.0f);
}

__global__ __launch_bounds__(256, 4) void k_stem_conv3d_gn(
    const float *in_bdhw, const float *w_48x27, int B, __half *out, float *gn_acc) {
    __shared__ float sm[3][18][18];
    __shared__ float sw[48][27];
    __shared__ float s_gs[16];
    __shared__ float s_gq[16];
    int tid = threadIdx.x;
    int bz = blockIdx.z;
    int b = bz >> 4;
    int z = bz & 15;
    int ty = blockIdx.y * 16;
    int tx = blockIdx.x * 16;
    int lx = tid & 15;
    int ly = tid >> 4;
    int lane = tid & 31;

    if (tid < 16) {
        s_gs[tid] = 0.f;
        s_gq[tid] = 0.f;
    }
    for (int i = tid; i < 48 * 27; i += 256)
        sw[i / 27][i % 27] = w_48x27[i];

    const float *in_b = in_bdhw + (long)b * 16 * 128 * 128;
    for (int i = tid; i < 3 * 18 * 18; i += 256) {
        int dz = i / 324;
        int rem = i % 324;
        int sy = rem / 18;
        int sx = rem % 18;
        int iz = z + dz - 1;
        int gy = ty + sy - 1;
        int gx = tx + sx - 1;
        sm[dz][sy][sx] = (iz >= 0 && iz < 16 && gy >= 0 && gy < 128 && gx >= 0 && gx < 128)
            ? in_b[(iz * 128 + gy) * 128 + gx]
            : 0.f;
    }
    __syncthreads();

    float patch[27];
#pragma unroll
    for (int dz = 0; dz < 3; ++dz)
#pragma unroll
        for (int dy = 0; dy < 3; ++dy)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx)
                patch[dz * 9 + dy * 3 + dx] = sm[dz][ly + dy][lx + dx];

    int gy = ty + ly, gx = tx + lx;
    __half2 *dst =
        reinterpret_cast<__half2 *>(out + ((((long)b * 16 + z) * 128 + gy) * 128 + gx) * 48);
    for (int co = 0; co < 48; co += 2) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int k = 0; k < 27; ++k) {
            a0 = fmaf(patch[k], sw[co][k], a0);
            a1 = fmaf(patch[k], sw[co + 1][k], a1);
        }
        dst[co >> 1] = __floats2half2_rn(a0, a1);
        float s0 = a0, q0 = a0 * a0, s1 = a1, q1 = a1 * a1;
        for (int off = 16; off > 0; off >>= 1) {
            s0 += __shfl_down_sync(0xffffffff, s0, off);
            q0 += __shfl_down_sync(0xffffffff, q0, off);
            s1 += __shfl_down_sync(0xffffffff, s1, off);
            q1 += __shfl_down_sync(0xffffffff, q1, off);
        }
        if (lane == 0) {
            int g0 = co / 3, g1 = (co + 1) / 3;
            if (g0 == g1) {
                atomicAdd(&s_gs[g0], s0 + s1);
                atomicAdd(&s_gq[g0], q0 + q1);
            } else {
                atomicAdd(&s_gs[g0], s0);
                atomicAdd(&s_gq[g0], q0);
                atomicAdd(&s_gs[g1], s1);
                atomicAdd(&s_gq[g1], q1);
            }
        }
    }
    __syncthreads();
    if (tid < 16) {
        atomicAdd(&gn_acc[(b * 16 + tid) * 2], s_gs[tid]);
        atomicAdd(&gn_acc[(b * 16 + tid) * 2 + 1], s_gq[tid]);
    }
}

template <int C, int TP, int G>
__global__ __launch_bounds__(192, 4) void k_wino2d44_in_fused(const __half *X,
    const float *gn_params, const float *gamma, const float *beta, __half *V, int B, int D, int H,
    int W) {
    constexpr int W_IN = TP * 4 + 2;
    constexpr int C_H2 = C / 2;
    constexpr int ROW_H2 = TP * C_H2;
    constexpr int C_U4 = C / 8;
    constexpr int TOTAL_U4 = 6 * W_IN * C_U4;
    __shared__ float s_sc[C], s_sh[C];
    __shared__ __half sm_x[6][W_IN][C];

    int tid = threadIdx.x;
    int tx0 = blockIdx.x * TP;
    int ty = blockIdx.y;
    int bz = blockIdx.z;
    int b = bz / D;
    int tw = W >> 2, th = H >> 2;
    int tiles_yx = th * tw;

    if (tid < C) {
        int g = tid / (C / G);
        float m = gn_params[(b * G + g) * 2];
        float is = gn_params[(b * G + g) * 2 + 1];
        float sc = is * gamma[tid];
        s_sc[tid] = sc;
        s_sh[tid] = beta[tid] - m * sc;
    }
    __syncthreads();

    int y0 = ty * 4 - 1;
    int x0 = tx0 * 4 - 1;
    const __half *Xz = X + (long)bz * H * W * C;
    for (int i = tid; i < TOTAL_U4; i += 192) {
        int cu = i % C_U4;
        int t = i / C_U4;
        int dx = t % W_IN;
        int dy = t / W_IN;
        int iy = y0 + dy, ix = x0 + dx;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            v = *reinterpret_cast<const uint4 *>(Xz + ((long)iy * W + ix) * C + cu * 8);
            __half2 *h = reinterpret_cast<__half2 *>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __half22float2(h[k]);
                int c0 = cu * 8 + k * 2;
                float v0 = silu_f(fmaf(f.x, s_sc[c0], s_sh[c0]));
                float v1 = silu_f(fmaf(f.y, s_sc[c0 + 1], s_sh[c0 + 1]));
                h[k] = __floats2half2_rn(v0, v1);
            }
        }
        *reinterpret_cast<uint4 *>(&sm_x[dy][dx][cu * 8]) = v;
    }
    __syncthreads();

    constexpr int Y_WORK = W_IN * C_H2;
    for (int i = tid; i < Y_WORK; i += 192) {
        int c2 = i % C_H2;
        int dx = i / C_H2;
        float d0[6], d1[6], o0[6], o1[6];
#pragma unroll
        for (int dy = 0; dy < 6; ++dy) {
            float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&sm_x[dy][dx][c2 * 2]));
            d0[dy] = f.x;
            d1[dy] = f.y;
        }
        bt4_1d(d0, o0);
        bt4_1d(d1, o1);
#pragma unroll
        for (int py = 0; py < 6; ++py)
            *reinterpret_cast<__half2 *>(&sm_x[py][dx][c2 * 2]) = __floats2half2_rn(o0[py], o1[py]);
    }
    __syncthreads();

    int grp = tid / ROW_H2;
    int col = tid % ROW_H2;
    int tl = col / C_H2;
    int c2 = col % C_H2;
    long plane_stride = (long)B * D * tiles_yx * C;
    long base_off = ((long)bz * tiles_yx + ty * tw + tx0) * C + col * 2;
    for (int py = grp; py < 6; py += 2) {
        float d0[6], d1[6], o0[6], o1[6];
#pragma unroll
        for (int dx = 0; dx < 6; ++dx) {
            float2 f =
                __half22float2(*reinterpret_cast<const __half2 *>(&sm_x[py][tl * 4 + dx][c2 * 2]));
            d0[dx] = f.x;
            d1[dx] = f.y;
        }
        bt4_1d(d0, o0);
        bt4_1d(d1, o1);
#pragma unroll
        for (int px = 0; px < 6; ++px) {
            int plane = py * 6 + px;
            __half2 hv = __floats2half2_rn(o0[px], o1[px]);
            st_cs_u32(
                V + (long)plane * plane_stride + base_off, *reinterpret_cast<unsigned *>(&hv));
        }
    }
}

__global__ __launch_bounds__(128, 4) void k_gemm_wino2d_c48(
    const __half *V, const __half *U, __half *M_out, int B, int D, int tiles_yx) {
    __shared__ __align__(128) unsigned char smem[(64 + 48) * 32 * 2];
    int z = blockIdx.x;
    int m0 = blockIdx.y * 64;
    int bp = blockIdx.z;
    int b = bp % B;
    int plane = bp / B;
    int bz = b * D + z;
    long p_stride = (long)B * D * tiles_yx * 48;
    const __half *Vp = V + (long)plane * p_stride;
    const __half *Up = U + (long)plane * 3 * 48 * 48;
    __half *Mp = M_out + (long)plane * p_stride + ((long)bz * tiles_yx + m0) * 48;

    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int wm = warp * 16;
    int q = lane >> 3, r = lane & 7;
    unsigned as = su32(smem);
    unsigned bs = as + 64 * 64;

    float acc[6][4];
#pragma unroll
    for (int j = 0; j < 6; ++j)
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

#pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
        int iz = z + dz;
        if (iz < 0 || iz >= D)
            continue;
        const __half *Az = Vp + ((long)(b * D + iz) * tiles_yx + m0) * 48;
        const __half *Uz = Up + (long)(dz + 1) * 48 * 48;

#pragma unroll
        for (int c = tid; c < 256; c += 128) {
            int row = c >> 2, col = c & 3;
            st_smem16(sad64(as, row, col),
                *reinterpret_cast<const uint4 *>(Az + (long)row * 48 + col * 8));
        }
        for (int c = tid; c < 192; c += 128) {
            int row = c >> 2, col = c & 3;
            st_smem16(
                sad64(bs, row, col), *reinterpret_cast<const uint4 *>(Uz + row * 48 + col * 8));
        }
        __syncthreads();

        uint4 a_next =
            *reinterpret_cast<const uint4 *>(Az + (long)(tid >> 1) * 48 + 32 + (tid & 1) * 8);
        uint4 b_next = tid < 96
            ? *reinterpret_cast<const uint4 *>(Uz + (tid >> 1) * 48 + 32 + (tid & 1) * 8)
            : make_uint4(0, 0, 0, 0);

#pragma unroll
        for (int kb = 0; kb < 32; kb += 16) {
            unsigned af[4], bf[3][4];
            ldsm(af, sad64(as, wm + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < 3; ++j)
                ldsm(bf[j], sad64(bs, j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int j = 0; j < 6; ++j)
                mma_f16(acc[j], af, bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();

        st_smem16(sad64(as, tid >> 1, tid & 1), a_next);
        if (tid < 96)
            st_smem16(sad64(bs, tid >> 1, tid & 1), b_next);
        __syncthreads();

        {
            unsigned af[4], bf[3][4];
            ldsm(af, sad64(as, wm + (lane & 15), lane >> 4));
#pragma unroll
            for (int j = 0; j < 3; ++j)
                ldsm(bf[j], sad64(bs, j * 16 + r + ((q >> 1) << 3), q & 1));
#pragma unroll
            for (int j = 0; j < 6; ++j)
                mma_f16(acc[j], af, bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }

    __half *sout = reinterpret_cast<__half *>(smem);
    int gq = lane >> 2, tq = lane & 3;
    int m = wm + gq;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
        *reinterpret_cast<unsigned *>(sout + m * 48 + j * 8 + tq * 2) =
            pack_h2(acc[j][0], acc[j][1]);
        *reinterpret_cast<unsigned *>(sout + (m + 8) * 48 + j * 8 + tq * 2) =
            pack_h2(acc[j][2], acc[j][3]);
    }
    __syncthreads();

#pragma unroll
    for (int c = tid; c < 384; c += 128) {
        int row = c / 6, col = c % 6;
        st_cs_u4(Mp + (long)row * 48 + col * 8,
            *reinterpret_cast<const uint4 *>(sout + row * 48 + col * 8));
    }
}

__global__ __launch_bounds__(256, 2) void k_gemm_wino2d_c96(
    const __half *V, const __half *U, __half *M_out, int B, int D, int tiles_yx) {
    constexpr int BM = 128, BN = 96, BK = 32;
    constexpr int ABYTES = BM * BK * 2, BBYTES = BN * BK * 2;
    extern __shared__ __align__(128) unsigned char smem[];

    int z = blockIdx.x;
    int m0 = blockIdx.y * BM;
    int bp = blockIdx.z;
    int b = bp % B;
    int plane = bp / B;
    int bz = b * D + z;
    long p_stride = (long)B * D * tiles_yx * 96;
    const __half *Vp = V + (long)plane * p_stride;
    const __half *Up = U + (long)plane * 3 * 96 * 96;
    __half *Mp = M_out + (long)plane * p_stride + ((long)bz * tiles_yx + m0) * 96;

    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int wm = (warp >> 1) * 32;
    int wn = (warp & 1) * 48;
    int q = lane >> 3, r = lane & 7;
    unsigned sbase = su32(smem);

    float acc[2][6][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 6; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    int dz_list[3], ndz = 0;
    for (int dz = -1; dz <= 1; ++dz)
        if (z + dz >= 0 && z + dz < D)
            dz_list[ndz++] = dz;
    int nsteps = ndz * 3;

    uint4 ra[2], rb[2];
    auto load_g = [&](int step) {
        int dz = dz_list[step / 3];
        int kc = (step % 3) * 32;
        const __half *Az = Vp + ((long)(b * D + z + dz) * tiles_yx + m0) * 96 + kc;
        const __half *Uz = Up + (long)(dz + 1) * 96 * 96 + kc;
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int c = tid + it * 256;
            ra[it] = *reinterpret_cast<const uint4 *>(Az + (long)(c >> 2) * 96 + (c & 3) * 8);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int c = tid + it * 256;
            if (c < 384)
                rb[it] = *reinterpret_cast<const uint4 *>(Uz + (long)(c >> 2) * 96 + (c & 3) * 8);
        }
    };
    auto store_s = [&](int stage) {
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int c = tid + it * 256;
            st_smem16(sad64(as, c >> 2, c & 3), ra[it]);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int c = tid + it * 256;
            if (c < 384)
                st_smem16(sad64(bs, c >> 2, c & 3), rb[it]);
        }
    };

    load_g(0);
    store_s(0);
    __syncthreads();
    for (int s = 0; s < nsteps; ++s) {
        if (s + 1 < nsteps)
            load_g(s + 1);
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < 32; kb += 16) {
            unsigned af[2][4], bf[3][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < 3; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 6; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        if (s + 1 < nsteps) {
            store_s((s + 1) & 1);
            __syncthreads();
        }
    }
    __syncthreads();

    __half *sout = reinterpret_cast<__half *>(smem);
    int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int m = wm + i * 16 + gq;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            int n = wn + j * 8 + tq * 2;
            *reinterpret_cast<unsigned *>(sout + m * 96 + n) = pack_h2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<unsigned *>(sout + (m + 8) * 96 + n) =
                pack_h2(acc[i][j][2], acc[i][j][3]);
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = tid; c < 1536; c += 256) {
        int row = c / 12, col = c % 12;
        st_cs_u4(Mp + (long)row * 96 + col * 8,
            *reinterpret_cast<const uint4 *>(sout + row * 96 + col * 8));
    }
}

template <int C, int TP, int G>
__global__ __launch_bounds__(192, 3) void k_wino2d44_out_fused(const __half *M,
    const float *u_scale, const __half *res, __half *out, float *gn_acc, int B, int D, int H,
    int W) {
    constexpr int C_H2 = C / 2;
    constexpr int ROW_H2 = TP * C_H2;
    __shared__ float2 sm[6][4][ROW_H2];

    int tid = threadIdx.x;
    int tx0 = blockIdx.x * TP;
    int ty = blockIdx.y;
    int bz = blockIdx.z;
    int b = bz / D;
    int tw = W >> 2, th = H >> 2;
    int tiles_yx = th * tw;
    long plane_stride = (long)B * D * tiles_yx * C;
    long base_off = ((long)bz * tiles_yx + ty * tw + tx0) * C;

    int grp = tid / ROW_H2;
    int col = tid % ROW_H2;
    int tl = col / C_H2;
    int c2 = col % C_H2;

    for (int py = grp; py < 6; py += 2) {
        float m0[6], m1[6], o0[4], o1[4];
#pragma unroll
        for (int px = 0; px < 6; ++px) {
            int plane = py * 6 + px;
            float sc = u_scale[plane];
            float2 f = __half22float2(*reinterpret_cast<const __half2 *>(
                M + (long)plane * plane_stride + base_off + col * 2));
            m0[px] = f.x * sc;
            m1[px] = f.y * sc;
        }
        at4_1d(m0, o0);
        at4_1d(m1, o1);
#pragma unroll
        for (int ox = 0; ox < 4; ++ox)
            sm[py][ox][col] = make_float2(o0[ox], o1[ox]);
    }
    __syncthreads();

    __half *out_z = out + (long)bz * H * W * C;
    const __half *res_z = res ? res + (long)bz * H * W * C : nullptr;
    float s0 = 0.f, q0 = 0.f, s1 = 0.f, q1 = 0.f;
    for (int ox = grp; ox < 4; ox += 2) {
        float m0[6], m1[6], o0[4], o1[4];
#pragma unroll
        for (int py = 0; py < 6; ++py) {
            float2 f = sm[py][ox][col];
            m0[py] = f.x;
            m1[py] = f.y;
        }
        at4_1d(m0, o0);
        at4_1d(m1, o1);
        int gx = (tx0 + tl) * 4 + ox;
#pragma unroll
        for (int oy = 0; oy < 4; ++oy) {
            int gy = ty * 4 + oy;
            long off = ((long)gy * W + gx) * C + c2 * 2;
            float v0 = o0[oy], v1 = o1[oy];
            if (res_z) {
                float2 rf = __half22float2(*reinterpret_cast<const __half2 *>(res_z + off));
                v0 += rf.x;
                v1 += rf.y;
            }
            *reinterpret_cast<__half2 *>(out_z + off) = __floats2half2_rn(v0, v1);
            if (gn_acc) {
                s0 += v0;
                q0 = fmaf(v0, v0, q0);
                s1 += v1;
                q1 = fmaf(v1, v1, q1);
            }
        }
    }
    if (gn_acc) {
        __syncthreads();
        float *sf = reinterpret_cast<float *>(&sm[0][0][0]);
        sf[tid] = s0;
        sf[192 + tid] = q0;
        sf[384 + tid] = s1;
        sf[576 + tid] = q1;
        __syncthreads();
        if (tid < C_H2) {
            float ts0 = 0.f, tq0 = 0.f, ts1 = 0.f, tq1 = 0.f;
#pragma unroll
            for (int g2 = 0; g2 < 2; ++g2)
#pragma unroll
                for (int t = 0; t < TP; ++t) {
                    int idx = g2 * ROW_H2 + t * C_H2 + tid;
                    ts0 += sf[idx];
                    tq0 += sf[192 + idx];
                    ts1 += sf[384 + idx];
                    tq1 += sf[576 + idx];
                }
            int c0 = tid * 2;
            int g0 = c0 / (C / G), g1 = (c0 + 1) / (C / G);
            if (g0 == g1) {
                atomicAdd(&gn_acc[(b * G + g0) * 2], ts0 + ts1);
                atomicAdd(&gn_acc[(b * G + g0) * 2 + 1], tq0 + tq1);
            } else {
                atomicAdd(&gn_acc[(b * G + g0) * 2], ts0);
                atomicAdd(&gn_acc[(b * G + g0) * 2 + 1], tq0);
                atomicAdd(&gn_acc[(b * G + g1) * 2], ts1);
                atomicAdd(&gn_acc[(b * G + g1) * 2 + 1], tq1);
            }
        }
    }
}

__global__ __launch_bounds__(192, 2) void k_wino444_in_ch64_fused(const __half *x,
    const float *gn_params, const float *gamma, const float *beta, __half *V, int B, int D, int H,
    int W, int C) {
    __shared__ __half2 sm[6][6][6][33];
    int tid = threadIdx.x;
    int cl2 = tid & 31;
    int wid = tid >> 5;
    int c = blockIdx.y * 64 + (cl2 << 1);
    int td = D >> 2, th = H >> 2, tw = W >> 2;
    int P = B * td * th * tw;
    int p = blockIdx.x;
    if (p >= P)
        return;
    int tx = p % tw;
    int t1 = p / tw;
    int ty = t1 % th;
    int t2 = t1 / th;
    int tz = t2 % td;
    int b = t2 / td;
    int z0 = tz * 4 - 1, y0 = ty * 4 - 1, x0 = tx * 4 - 1;

    int cpg = C / 32;
    int g0 = c / cpg, g1 = (c + 1) / cpg;
    float m0 = gn_params[(b * 32 + g0) * 2], is0 = gn_params[(b * 32 + g0) * 2 + 1];
    float m1 = gn_params[(b * 32 + g1) * 2], is1 = gn_params[(b * 32 + g1) * 2 + 1];
    float sc0 = is0 * gamma[c], sh0 = beta[c] - m0 * sc0;
    float sc1 = is1 * gamma[c + 1], sh1 = beta[c + 1] - m1 * sc1;

    for (int yx = wid; yx < 36; yx += 6) {
        int dy = yx / 6, dx = yx % 6;
        int iy = y0 + dy, ix = x0 + dx;
        float d0[6], d1[6];
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            long yx_off = ((long)iy * W + ix) * C + c;
            long z_stride = (long)H * W * C;
            long b_off = (long)b * D * z_stride + yx_off;
#pragma unroll
            for (int dz = 0; dz < 6; ++dz) {
                int iz = z0 + dz;
                if (iz >= 0 && iz < D) {
                    float2 f = __half22float2(
                        *reinterpret_cast<const __half2 *>(x + b_off + iz * z_stride));
                    d0[dz] = silu_f(fmaf(f.x, sc0, sh0));
                    d1[dz] = silu_f(fmaf(f.y, sc1, sh1));
                } else {
                    d0[dz] = 0.f;
                    d1[dz] = 0.f;
                }
            }
        } else {
#pragma unroll
            for (int dz = 0; dz < 6; ++dz) {
                d0[dz] = 0.f;
                d1[dz] = 0.f;
            }
        }
        float o0[6], o1[6];
        bt4_1d(d0, o0);
        bt4_1d(d1, o1);
#pragma unroll
        for (int pz = 0; pz < 6; ++pz)
            sm[pz][dy][dx][cl2] = __floats2half2_rn(o0[pz], o1[pz]);
    }
    __syncthreads();

    for (int zx = wid; zx < 36; zx += 6) {
        int pz = zx / 6, dx = zx % 6;
        float d0[6], d1[6], o0[6], o1[6];
#pragma unroll
        for (int dy = 0; dy < 6; ++dy) {
            float2 f = __half22float2(sm[pz][dy][dx][cl2]);
            d0[dy] = f.x;
            d1[dy] = f.y;
        }
        bt4_1d(d0, o0);
        bt4_1d(d1, o1);
#pragma unroll
        for (int py = 0; py < 6; ++py)
            sm[pz][py][dx][cl2] = __floats2half2_rn(o0[py], o1[py]);
    }
    __syncthreads();

    long plane_stride = (long)P * C;
    long tile_off = (long)p * C + c;
    for (int zy = wid; zy < 36; zy += 6) {
        int pz = zy / 6, py = zy % 6;
        float d0[6], d1[6], o0[6], o1[6];
#pragma unroll
        for (int dx = 0; dx < 6; ++dx) {
            float2 f = __half22float2(sm[pz][py][dx][cl2]);
            d0[dx] = f.x;
            d1[dx] = f.y;
        }
        bt4_1d(d0, o0);
        bt4_1d(d1, o1);
        int base_plane = (pz * 6 + py) * 6;
#pragma unroll
        for (int px = 0; px < 6; ++px) {
            __half2 hv = __floats2half2_rn(o0[px], o1[px]);
            st_cs_u32(V + (long)(base_plane + px) * plane_stride + tile_off,
                *reinterpret_cast<unsigned *>(&hv));
        }
    }
}

__global__ __launch_bounds__(192, 2) void k_wino444_out_ch64_fused(const __half *M,
    const float *u_scale, const __half *res, __half *out, float *gn_acc, int B, int D, int H, int W,
    int C) {
    __shared__ __half2 sm[6][6][4][33];
    int tid = threadIdx.x;
    int cl2 = tid & 31;
    int wid = tid >> 5;
    int c = blockIdx.y * 64 + (cl2 << 1);
    int td = D >> 2, th = H >> 2, tw = W >> 2;
    int P = B * td * th * tw;
    int p = blockIdx.x;
    if (p >= P)
        return;
    int tx = p % tw;
    int t1 = p / tw;
    int ty = t1 % th;
    int t2 = t1 / th;
    int tz = t2 % td;
    int b = t2 / td;
    long plane_stride = (long)P * C;
    long tile_off = (long)p * C + c;

    for (int zy = wid; zy < 36; zy += 6) {
        int pz = zy / 6, py = zy % 6;
        int base_plane = (pz * 6 + py) * 6;
        float m0[6], m1[6], o0[4], o1[4];
#pragma unroll
        for (int px = 0; px < 6; ++px) {
            int plane = base_plane + px;
            float sc = u_scale[plane];
            float2 f = __half22float2(
                *reinterpret_cast<const __half2 *>(M + (long)plane * plane_stride + tile_off));
            m0[px] = f.x * sc;
            m1[px] = f.y * sc;
        }
        at4_1d(m0, o0);
        at4_1d(m1, o1);
#pragma unroll
        for (int ox = 0; ox < 4; ++ox)
            sm[pz][py][ox][cl2] = __floats2half2_rn(o0[ox], o1[ox]);
    }
    __syncthreads();

    for (int zx = wid; zx < 24; zx += 6) {
        int pz = zx >> 2, ox = zx & 3;
        float m0[6], m1[6], o0[4], o1[4];
#pragma unroll
        for (int py = 0; py < 6; ++py) {
            float2 f = __half22float2(sm[pz][py][ox][cl2]);
            m0[py] = f.x;
            m1[py] = f.y;
        }
        at4_1d(m0, o0);
        at4_1d(m1, o1);
#pragma unroll
        for (int oy = 0; oy < 4; ++oy)
            sm[pz][oy][ox][cl2] = __floats2half2_rn(o0[oy], o1[oy]);
    }
    __syncthreads();

    float s0 = 0.f, q0 = 0.f, s1 = 0.f, q1 = 0.f;
    for (int yx = wid; yx < 16; yx += 6) {
        int oy = yx >> 2, ox = yx & 3;
        float m0[6], m1[6], o0[4], o1[4];
#pragma unroll
        for (int pz = 0; pz < 6; ++pz) {
            float2 f = __half22float2(sm[pz][oy][ox][cl2]);
            m0[pz] = f.x;
            m1[pz] = f.y;
        }
        at4_1d(m0, o0);
        at4_1d(m1, o1);
        long yx_off = ((long)(ty * 4 + oy) * W + tx * 4 + ox) * C + c;
        long z_stride = (long)H * W * C;
        long base = (long)b * D * z_stride + (long)(tz * 4) * z_stride + yx_off;
#pragma unroll
        for (int oz = 0; oz < 4; ++oz) {
            long off = base + (long)oz * z_stride;
            float v0 = o0[oz], v1 = o1[oz];
            if (res) {
                float2 r = __half22float2(*reinterpret_cast<const __half2 *>(res + off));
                v0 += r.x;
                v1 += r.y;
            }
            *reinterpret_cast<__half2 *>(out + off) = __floats2half2_rn(v0, v1);
            if (gn_acc) {
                s0 += v0;
                q0 = fmaf(v0, v0, q0);
                s1 += v1;
                q1 = fmaf(v1, v1, q1);
            }
        }
    }
    if (gn_acc) {
        __syncthreads();
        float *sf = reinterpret_cast<float *>(&sm[0][0][0][0]);
        sf[tid] = s0;
        sf[192 + tid] = q0;
        sf[384 + tid] = s1;
        sf[576 + tid] = q1;
        __syncthreads();
        if (tid < 32) {
            float ts0 = 0.f, tq0 = 0.f, ts1 = 0.f, tq1 = 0.f;
#pragma unroll
            for (int w = 0; w < 6; ++w) {
                int idx = w * 32 + tid;
                ts0 += sf[idx];
                tq0 += sf[192 + idx];
                ts1 += sf[384 + idx];
                tq1 += sf[576 + idx];
            }
            int cpg = C / 32;
            int cc = blockIdx.y * 64 + tid * 2;
            int g0 = cc / cpg, g1 = (cc + 1) / cpg;
            if (g0 == g1) {
                atomicAdd(&gn_acc[(b * 32 + g0) * 2], ts0 + ts1);
                atomicAdd(&gn_acc[(b * 32 + g0) * 2 + 1], tq0 + tq1);
            } else {
                atomicAdd(&gn_acc[(b * 32 + g0) * 2], ts0);
                atomicAdd(&gn_acc[(b * 32 + g0) * 2 + 1], tq0);
                atomicAdd(&gn_acc[(b * 32 + g1) * 2], ts1);
                atomicAdd(&gn_acc[(b * 32 + g1) * 2 + 1], tq1);
            }
        }
    }
}

__global__ __launch_bounds__(256, 3) void k_wino222_in_ch64_fused(const __half *x,
    const float *gn_params, const float *gamma, const float *beta, __half *V, int B, int D, int H,
    int W, int C) {
    __shared__ float2 sm[4][4][4][33];
    int tid = threadIdx.x;
    int cl2 = tid & 31;
    int wid = tid >> 5;
    int c = blockIdx.y * 64 + (cl2 << 1);
    int td = D >> 1, th = H >> 1, tw = W >> 1;
    int P = B * td * th * tw;
    int p = blockIdx.x;
    if (p >= P)
        return;
    int tx = p % tw;
    int t1 = p / tw;
    int ty = t1 % th;
    int t2 = t1 / th;
    int tz = t2 % td;
    int b = t2 / td;
    int z0 = tz * 2 - 1, y0 = ty * 2 - 1, x0 = tx * 2 - 1;

    int cpg = C / 32;
    int g0 = c / cpg, g1 = (c + 1) / cpg;
    float m0 = gn_params[(b * 32 + g0) * 2], is0 = gn_params[(b * 32 + g0) * 2 + 1];
    float m1 = gn_params[(b * 32 + g1) * 2], is1 = gn_params[(b * 32 + g1) * 2 + 1];
    float sc0 = is0 * gamma[c], sh0 = beta[c] - m0 * sc0;
    float sc1 = is1 * gamma[c + 1], sh1 = beta[c + 1] - m1 * sc1;

    for (int yx = wid; yx < 16; yx += 8) {
        int dy = yx >> 2, dx = yx & 3;
        int iy = y0 + dy, ix = x0 + dx;
        float d0[4], d1[4];
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            long yx_off = ((long)iy * W + ix) * C + c;
            long z_stride = (long)H * W * C;
            long b_off = (long)b * D * z_stride + yx_off;
#pragma unroll
            for (int dz = 0; dz < 4; ++dz) {
                int iz = z0 + dz;
                if (iz >= 0 && iz < D) {
                    float2 f = __half22float2(
                        *reinterpret_cast<const __half2 *>(x + b_off + iz * z_stride));
                    d0[dz] = silu_f(fmaf(f.x, sc0, sh0));
                    d1[dz] = silu_f(fmaf(f.y, sc1, sh1));
                } else {
                    d0[dz] = 0.f;
                    d1[dz] = 0.f;
                }
            }
        } else {
#pragma unroll
            for (int dz = 0; dz < 4; ++dz) {
                d0[dz] = 0.f;
                d1[dz] = 0.f;
            }
        }
        float o0[4], o1[4];
        bt2_1d(d0, o0);
        bt2_1d(d1, o1);
#pragma unroll
        for (int pz = 0; pz < 4; ++pz)
            sm[pz][dy][dx][cl2] = make_float2(o0[pz], o1[pz]);
    }
    __syncthreads();

    for (int zx = wid; zx < 16; zx += 8) {
        int pz = zx >> 2, dx = zx & 3;
        float d0[4], d1[4], o0[4], o1[4];
#pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
            float2 f = sm[pz][dy][dx][cl2];
            d0[dy] = f.x;
            d1[dy] = f.y;
        }
        bt2_1d(d0, o0);
        bt2_1d(d1, o1);
#pragma unroll
        for (int py = 0; py < 4; ++py)
            sm[pz][py][dx][cl2] = make_float2(o0[py], o1[py]);
    }
    __syncthreads();

    long plane_stride = (long)P * C;
    long tile_off = (long)p * C + c;
    for (int zy = wid; zy < 16; zy += 8) {
        int pz = zy >> 2, py = zy & 3;
        float d0[4], d1[4], o0[4], o1[4];
#pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            float2 f = sm[pz][py][dx][cl2];
            d0[dx] = f.x;
            d1[dx] = f.y;
        }
        bt2_1d(d0, o0);
        bt2_1d(d1, o1);
        int base_plane = (pz * 4 + py) * 4;
#pragma unroll
        for (int px = 0; px < 4; ++px) {
            __half2 hv = __floats2half2_rn(o0[px], o1[px]);
            st_cs_u32(V + (long)(base_plane + px) * plane_stride + tile_off,
                *reinterpret_cast<unsigned *>(&hv));
        }
    }
}

__global__ __launch_bounds__(256, 4) void k_wino222_out_ch64_fused(const __half *M,
    const float *u_scale, const __half *res, __half *out, float *gn_acc, int B, int D, int H, int W,
    int C) {
    __shared__ float2 sm[4][4][2][33];
    int tid = threadIdx.x;
    int cl2 = tid & 31;
    int wid = tid >> 5;
    int c = blockIdx.y * 64 + (cl2 << 1);
    int td = D >> 1, th = H >> 1, tw = W >> 1;
    int P = B * td * th * tw;
    int p = blockIdx.x;
    if (p >= P)
        return;
    int tx = p % tw;
    int t1 = p / tw;
    int ty = t1 % th;
    int t2 = t1 / th;
    int tz = t2 % td;
    int b = t2 / td;
    long plane_stride = (long)P * C;
    long tile_off = (long)p * C + c;

    for (int zy = wid; zy < 16; zy += 8) {
        int pz = zy >> 2, py = zy & 3;
        int base_plane = (pz * 4 + py) * 4;
        float m0[4], m1[4], o0[2], o1[2];
#pragma unroll
        for (int px = 0; px < 4; ++px) {
            int plane = base_plane + px;
            float sc = u_scale[plane];
            float2 f = __half22float2(
                *reinterpret_cast<const __half2 *>(M + (long)plane * plane_stride + tile_off));
            m0[px] = f.x * sc;
            m1[px] = f.y * sc;
        }
        at2_1d(m0, o0);
        at2_1d(m1, o1);
        sm[pz][py][0][cl2] = make_float2(o0[0], o1[0]);
        sm[pz][py][1][cl2] = make_float2(o0[1], o1[1]);
    }
    __syncthreads();

    {
        int pz = wid >> 1, ox = wid & 1;
        float m0[4], m1[4], o0[2], o1[2];
#pragma unroll
        for (int py = 0; py < 4; ++py) {
            float2 f = sm[pz][py][ox][cl2];
            m0[py] = f.x;
            m1[py] = f.y;
        }
        at2_1d(m0, o0);
        at2_1d(m1, o1);
        sm[pz][0][ox][cl2] = make_float2(o0[0], o1[0]);
        sm[pz][1][ox][cl2] = make_float2(o0[1], o1[1]);
    }
    __syncthreads();

    float s0 = 0.f, q0 = 0.f, s1 = 0.f, q1 = 0.f;
    if (wid < 4) {
        int oy = wid >> 1, ox = wid & 1;
        float m0[4], m1[4], o0[2], o1[2];
#pragma unroll
        for (int pz = 0; pz < 4; ++pz) {
            float2 f = sm[pz][oy][ox][cl2];
            m0[pz] = f.x;
            m1[pz] = f.y;
        }
        at2_1d(m0, o0);
        at2_1d(m1, o1);
        long yx_off = ((long)(ty * 2 + oy) * W + tx * 2 + ox) * C + c;
        long z_stride = (long)H * W * C;
        long base = (long)b * D * z_stride + (long)(tz * 2) * z_stride + yx_off;
#pragma unroll
        for (int oz = 0; oz < 2; ++oz) {
            long off = base + (long)oz * z_stride;
            float v0 = o0[oz], v1 = o1[oz];
            if (res) {
                float2 r = __half22float2(*reinterpret_cast<const __half2 *>(res + off));
                v0 += r.x;
                v1 += r.y;
            }
            *reinterpret_cast<__half2 *>(out + off) = __floats2half2_rn(v0, v1);
            if (gn_acc) {
                s0 += v0;
                q0 = fmaf(v0, v0, q0);
                s1 += v1;
                q1 = fmaf(v1, v1, q1);
            }
        }
    }
    if (gn_acc) {
        __syncthreads();
        float *sf = reinterpret_cast<float *>(&sm[0][0][0][0]);
        if (wid < 4) {
            sf[tid] = s0;
            sf[128 + tid] = q0;
            sf[256 + tid] = s1;
            sf[384 + tid] = q1;
        }
        __syncthreads();
        if (tid < 32) {
            float ts0 = 0.f, tq0 = 0.f, ts1 = 0.f, tq1 = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                int idx = w * 32 + tid;
                ts0 += sf[idx];
                tq0 += sf[128 + idx];
                ts1 += sf[256 + idx];
                tq1 += sf[384 + idx];
            }
            int cpg = C / 32;
            int cc = blockIdx.y * 64 + tid * 2;
            int g0 = cc / cpg, g1 = (cc + 1) / cpg;
            if (g0 == g1) {
                atomicAdd(&gn_acc[(b * 32 + g0) * 2], ts0 + ts1);
                atomicAdd(&gn_acc[(b * 32 + g0) * 2 + 1], tq0 + tq1);
            } else {
                atomicAdd(&gn_acc[(b * 32 + g0) * 2], ts0);
                atomicAdd(&gn_acc[(b * 32 + g0) * 2 + 1], tq0);
                atomicAdd(&gn_acc[(b * 32 + g1) * 2], ts1);
                atomicAdd(&gn_acc[(b * 32 + g1) * 2 + 1], tq1);
            }
        }
    }
}

template <int BM, int BN, int WGM, int WGN, int MIN_BLKS>
__global__ __launch_bounds__(WGM * WGN * 32, MIN_BLKS) void k_gemm_batched_f16(
    const __half *A, const __half *B, __half *C, int M, int N, int K) {
    constexpr int BK = 32;
    constexpr int THREADS = WGM * WGN * 32;
    constexpr int MT = (BM / WGM) / 16;
    constexpr int NT = (BN / WGN) / 8;
    constexpr int ABYTES = BM * BK * 2;
    constexpr int BBYTES = BN * BK * 2;
    constexpr int A_ITERS = (BM * BK / 8 + THREADS - 1) / THREADS;
    constexpr int B_ITERS = (BN * BK / 8 + THREADS - 1) / THREADS;
    extern __shared__ __align__(128) unsigned char smem[];

    int m0 = blockIdx.x * BM;
    int n0 = blockIdx.y * BN;
    int plane = blockIdx.z;
    const __half *Ap = A + (long)plane * M * K;
    const __half *Bp = B + (long)plane * N * K;
    __half *Cp = C + (long)plane * M * N;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wm = (warp / WGN) * MT * 16;
    int wn = (warp % WGN) * NT * 8;
    unsigned sbase = su32(smem);

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    int nk = (K + BK - 1) / BK;
    int q = lane >> 3, r = lane & 7;

#if __CUDA_ARCH__ >= 800
    auto load_stage = [&](int stage, int k0) {
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
        for (int c = tid; c < BM * BK / 8; c += THREADS) {
            int row = c >> 2, col = c & 3;
            bool ok = (m0 + row < M) && (k0 + col * 8 < K);
            const __half *p = ok ? Ap + (long)(m0 + row) * K + k0 + col * 8 : Ap;
            cp16(sad64(as, row, col), p, ok);
        }
        for (int c = tid; c < BN * BK / 8; c += THREADS) {
            int row = c >> 2, col = c & 3;
            bool ok = (n0 + row < N) && (k0 + col * 8 < K);
            const __half *p = ok ? Bp + (long)(n0 + row) * K + k0 + col * 8 : Bp;
            cp16(sad64(bs, row, col), p, ok);
        }
    };
    if (nk > 0) {
        load_stage(0, 0);
        cp_commit();
    }
    for (int s = 0; s < nk; ++s) {
        if (s + 1 < nk)
            load_stage((s + 1) & 1, (s + 1) * BK);
        cp_commit();
        cp_wait1();
        __syncthreads();
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }
    cp_wait0();
#else
    uint4 ra[A_ITERS], rb[B_ITERS];
    auto load_g = [&](int k0) {
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8) {
                int row = c >> 2, col = c & 3;
                bool ok = (m0 + row < M) && (k0 + col * 8 < K);
                ra[it] = ok
                    ? *reinterpret_cast<const uint4 *>(Ap + (long)(m0 + row) * K + k0 + col * 8)
                    : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8) {
                int row = c >> 2, col = c & 3;
                bool ok = (n0 + row < N) && (k0 + col * 8 < K);
                rb[it] = ok
                    ? *reinterpret_cast<const uint4 *>(Bp + (long)(n0 + row) * K + k0 + col * 8)
                    : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto store_s = [&](int stage) {
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8)
                st_smem16(sad64(as, c >> 2, c & 3), ra[it]);
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8)
                st_smem16(sad64(bs, c >> 2, c & 3), rb[it]);
        }
    };
    if (nk > 0) {
        load_g(0);
        store_s(0);
        __syncthreads();
    }
    for (int s = 0; s < nk; ++s) {
        if (s + 1 < nk)
            load_g((s + 1) * BK);
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        if (s + 1 < nk) {
            store_s((s + 1) & 1);
            __syncthreads();
        }
    }
    __syncthreads();
#endif

    __half *sout = reinterpret_cast<__half *>(smem);
    int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        int m = wm + i * 16 + gq;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int n = wn + j * 8 + tq * 2;
            *reinterpret_cast<unsigned *>(sout + m * BN + n) = pack_h2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<unsigned *>(sout + (m + 8) * BN + n) =
                pack_h2(acc[i][j][2], acc[i][j][3]);
        }
    }
    __syncthreads();

    constexpr int BN_U4 = BN / 8;
    constexpr int OUT_U4 = BM * BN_U4;
    for (int c = tid; c < OUT_U4; c += THREADS) {
        int row = c / BN_U4;
        int col = c % BN_U4;
        int gm = m0 + row;
        int gn = n0 + col * 8;
        if (gm < M && gn < N) {
            uint4 v = *reinterpret_cast<const uint4 *>(sout + row * BN + col * 8);
            if (gridDim.z > 1)
                st_cs_u4(Cp + (long)gm * N + gn, v);
            else
                *reinterpret_cast<uint4 *>(Cp + (long)gm * N + gn) = v;
        }
    }
}

template <int BM, int BN, int WGM, int WGN, int MIN_BLKS>
__global__ __launch_bounds__(WGM * WGN * 32, MIN_BLKS) void k_gemm_fuse_f16(const __half *A,
    const float *gn_params, const float *gamma, const float *beta, const __half *B, __half *C,
    float *gn_acc, int DHW, int M, int N, int K, int G_in, int G_out) {
    constexpr int BK = 32;
    constexpr int THREADS = WGM * WGN * 32;
    constexpr int MT = (BM / WGM) / 16;
    constexpr int NT = (BN / WGN) / 8;
    constexpr int ABYTES = BM * BK * 2;
    constexpr int BBYTES = BN * BK * 2;
    constexpr int A_ITERS = (BM * BK / 8 + THREADS - 1) / THREADS;
    constexpr int B_ITERS = (BN * BK / 8 + THREADS - 1) / THREADS;
    extern __shared__ __align__(128) unsigned char smem[];

    int m0 = blockIdx.x * BM;
    int n0 = blockIdx.y * BN;
    int b = m0 / DHW;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wm = (warp / WGN) * MT * 16;
    int wn = (warp % WGN) * NT * 8;

    float *s_sc = reinterpret_cast<float *>(smem);
    float *s_sh = s_sc + K;
    size_t param_bytes = ((size_t)2 * K * sizeof(float) + 127) & ~(size_t)127;
    unsigned char *tile_smem = smem + param_bytes;
    unsigned sbase = su32(tile_smem);

    int cpg_in = K / G_in;
    for (int k = tid; k < K; k += THREADS) {
        int g = k / cpg_in;
        float m = gn_params[(b * G_in + g) * 2];
        float is = gn_params[(b * G_in + g) * 2 + 1];
        float sc = is * gamma[k];
        s_sc[k] = sc;
        s_sh[k] = beta[k] - m * sc;
    }
    __syncthreads();

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    int nk = (K + BK - 1) / BK;
    int q = lane >> 3, r = lane & 7;

    auto apply_gn_silu = [&](uint4 raw, int kbase) {
        __half2 *h = reinterpret_cast<__half2 *>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __half22float2(h[i]);
            int c0 = kbase + 2 * i;
            float v0 = silu_f(fmaf(f.x, s_sc[c0], s_sh[c0]));
            float v1 = silu_f(fmaf(f.y, s_sc[c0 + 1], s_sh[c0 + 1]));
            h[i] = __floats2half2_rn(v0, v1);
        }
        return raw;
    };

    uint4 ra[A_ITERS], rb[B_ITERS];
    auto load_g = [&](int k0) {
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8) {
                int row = c >> 2, col = c & 3;
                bool ok = (m0 + row < M) && (k0 + col * 8 < K);
                ra[it] = ok ? apply_gn_silu(*reinterpret_cast<const uint4 *>(
                                                A + (long)(m0 + row) * K + k0 + col * 8),
                                  k0 + col * 8)
                            : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8) {
                int row = c >> 2, col = c & 3;
                bool ok = (n0 + row < N) && (k0 + col * 8 < K);
                rb[it] = ok
                    ? *reinterpret_cast<const uint4 *>(B + (long)(n0 + row) * K + k0 + col * 8)
                    : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto store_s = [&](int stage) {
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8)
                st_smem16(sad64(as, c >> 2, c & 3), ra[it]);
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8)
                st_smem16(sad64(bs, c >> 2, c & 3), rb[it]);
        }
    };
    if (nk > 0) {
        load_g(0);
        store_s(0);
        __syncthreads();
    }
    for (int s = 0; s < nk; ++s) {
        if (s + 1 < nk)
            load_g((s + 1) * BK);
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        if (s + 1 < nk) {
            store_s((s + 1) & 1);
            __syncthreads();
        }
    }
    __syncthreads();

    __half *sout = reinterpret_cast<__half *>(tile_smem);
    int gq = lane >> 2, tq = lane & 3;
    int cpg_out = N / G_out;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        int m = wm + i * 16 + gq;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int n = wn + j * 8 + tq * 2;
            *reinterpret_cast<unsigned *>(sout + m * BN + n) = pack_h2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<unsigned *>(sout + (m + 8) * BN + n) =
                pack_h2(acc[i][j][2], acc[i][j][3]);
        }
    }
    __syncthreads();

    constexpr int BN_U4 = BN / 8;
    constexpr int OUT_U4 = BM * BN_U4;
    for (int c = tid; c < OUT_U4; c += THREADS) {
        int row = c / BN_U4;
        int col = c % BN_U4;
        int gm = m0 + row;
        int gn = n0 + col * 8;
        if (gm < M && gn < N)
            *reinterpret_cast<uint4 *>(C + (long)gm * N + gn) =
                *reinterpret_cast<const uint4 *>(sout + row * BN + col * 8);
    }

    int bn_h2 = BN >> 1;
    if (tid < bn_h2) {
        int gn = n0 + tid * 2;
        if (gn < N) {
            float ts0 = 0.f, tq0 = 0.f, ts1 = 0.f, tq1 = 0.f;
            int rows = min(BM, M - m0);
            for (int row = 0; row < rows; ++row) {
                float2 f =
                    __half22float2(*reinterpret_cast<const __half2 *>(sout + row * BN + tid * 2));
                ts0 += f.x;
                tq0 = fmaf(f.x, f.x, tq0);
                ts1 += f.y;
                tq1 = fmaf(f.y, f.y, tq1);
            }
            int g0 = gn / cpg_out, g1 = (gn + 1) / cpg_out;
            if (g0 == g1) {
                atomicAdd(&gn_acc[(b * G_out + g0) * 2], ts0 + ts1);
                atomicAdd(&gn_acc[(b * G_out + g0) * 2 + 1], tq0 + tq1);
            } else {
                atomicAdd(&gn_acc[(b * G_out + g0) * 2], ts0);
                atomicAdd(&gn_acc[(b * G_out + g0) * 2 + 1], tq0);
                atomicAdd(&gn_acc[(b * G_out + g1) * 2], ts1);
                atomicAdd(&gn_acc[(b * G_out + g1) * 2 + 1], tq1);
            }
        }
    }
}

template <int BM, int BN, int WGM, int WGN, int MIN_BLKS>
__global__ __launch_bounds__(WGM * WGN * 32, MIN_BLKS) void k_conv3d_direct_mma(const __half *X,
    const __half *W_taps, const __half *res, __half *Y, int B, int Di, int Hi, int Wi, int Do,
    int Ho, int Wo, int sz, int sy, int sx, int Cin, int Cout) {
    constexpr int BK = 32;
    constexpr int THREADS = WGM * WGN * 32;
    constexpr int MT = (BM / WGM) / 16;
    constexpr int NT = (BN / WGN) / 8;
    constexpr int ABYTES = BM * BK * 2;
    constexpr int BBYTES = BN * BK * 2;
    constexpr int A_ITERS = (BM * BK / 8 + THREADS - 1) / THREADS;
    constexpr int B_ITERS = (BN * BK / 8 + THREADS - 1) / THREADS;
    extern __shared__ __align__(128) unsigned char smem[];

    int DHWo = Do * Ho * Wo;
    int M = B * DHWo;
    int m0 = blockIdx.x * BM;
    int n0 = blockIdx.y * BN;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wm = (warp / WGN) * MT * 16;
    int wn = (warp % WGN) * NT * 8;
    unsigned sbase = su32(smem);

    int logW = __ffs(Wo) - 1;
    int logH = __ffs(Ho) - 1;
    int logD = __ffs(Do) - 1;
    int maskW = Wo - 1, maskH = Ho - 1, maskD = Do - 1;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    int kc_steps = (Cin + BK - 1) / BK;
    int total_steps = 27 * kc_steps;
    int q = lane >> 3, r = lane & 7;

#if __CUDA_ARCH__ >= 800
    auto load_step = [&](int stage, int step) {
        int tap = step / kc_steps;
        int kc = (step % kc_steps) * BK;
        int dz = tap / 9 - 1;
        int dy = (tap / 3) % 3 - 1;
        int dx = tap % 3 - 1;
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
        for (int c = tid; c < BM * BK / 8; c += THREADS) {
            int row = c >> 2, col = c & 3;
            int m = m0 + row;
            bool ok = false;
            const __half *p = X;
            if (m < M && kc + col * 8 < Cin) {
                int ox = m & maskW;
                int oy = (m >> logW) & maskH;
                int oz = (m >> (logW + logH)) & maskD;
                int b = m >> (logW + logH + logD);
                int iz = oz * sz + dz;
                int iy = oy * sy + dy;
                int ix = ox * sx + dx;
                if (iz >= 0 && iz < Di && iy >= 0 && iy < Hi && ix >= 0 && ix < Wi) {
                    ok = true;
                    p = X + ((((long)b * Di + iz) * Hi + iy) * Wi + ix) * Cin + kc + col * 8;
                }
            }
            cp16(sad64(as, row, col), p, ok);
        }
        const __half *Wt = W_taps + (long)tap * Cout * Cin;
        for (int c = tid; c < BN * BK / 8; c += THREADS) {
            int row = c >> 2, col = c & 3;
            bool ok = (n0 + row < Cout) && (kc + col * 8 < Cin);
            const __half *p = ok ? Wt + (long)(n0 + row) * Cin + kc + col * 8 : Wt;
            cp16(sad64(bs, row, col), p, ok);
        }
    };
    if (total_steps > 0) {
        load_step(0, 0);
        cp_commit();
    }
    for (int s = 0; s < total_steps; ++s) {
        if (s + 1 < total_steps)
            load_step((s + 1) & 1, s + 1);
        cp_commit();
        cp_wait1();
        __syncthreads();
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }
    cp_wait0();
#else
    uint4 ra[A_ITERS], rb[B_ITERS];
    auto load_g = [&](int step) {
        int tap = step / kc_steps;
        int kc = (step % kc_steps) * BK;
        int dz = tap / 9 - 1;
        int dy = (tap / 3) % 3 - 1;
        int dx = tap % 3 - 1;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8) {
                int row = c >> 2, col = c & 3;
                int m = m0 + row;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < M && kc + col * 8 < Cin) {
                    int ox = m & maskW;
                    int oy = (m >> logW) & maskH;
                    int oz = (m >> (logW + logH)) & maskD;
                    int b = m >> (logW + logH + logD);
                    int iz = oz * sz + dz;
                    int iy = oy * sy + dy;
                    int ix = ox * sx + dx;
                    if (iz >= 0 && iz < Di && iy >= 0 && iy < Hi && ix >= 0 && ix < Wi)
                        v = *reinterpret_cast<const uint4 *>(
                            X + ((((long)b * Di + iz) * Hi + iy) * Wi + ix) * Cin + kc + col * 8);
                }
                ra[it] = v;
            }
        }
        const __half *Wt = W_taps + (long)tap * Cout * Cin;
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8) {
                int row = c >> 2, col = c & 3;
                bool ok = (n0 + row < Cout) && (kc + col * 8 < Cin);
                rb[it] = ok
                    ? *reinterpret_cast<const uint4 *>(Wt + (long)(n0 + row) * Cin + kc + col * 8)
                    : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto store_s = [&](int stage) {
        unsigned as = sbase + stage * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BM * BK / 8)
                st_smem16(sad64(as, c >> 2, c & 3), ra[it]);
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int c = tid + it * THREADS;
            if (c < BN * BK / 8)
                st_smem16(sad64(bs, c >> 2, c & 3), rb[it]);
        }
    };
    if (total_steps > 0) {
        load_g(0);
        store_s(0);
        __syncthreads();
    }
    for (int s = 0; s < total_steps; ++s) {
        if (s + 1 < total_steps)
            load_g(s + 1);
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES);
        unsigned bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm(af[i], sad64(as, wm + i * 16 + (lane & 15), (kb >> 3) + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                ldsm(bf[j], sad64(bs, wn + j * 16 + r + ((q >> 1) << 3), (kb >> 3) + (q & 1)));
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
        if (s + 1 < total_steps) {
            store_s((s + 1) & 1);
            __syncthreads();
        }
    }
    __syncthreads();
#endif

    __half *sout = reinterpret_cast<__half *>(smem);
    int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        int m = wm + i * 16 + gq;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int n = wn + j * 8 + tq * 2;
            *reinterpret_cast<unsigned *>(sout + m * BN + n) = pack_h2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<unsigned *>(sout + (m + 8) * BN + n) =
                pack_h2(acc[i][j][2], acc[i][j][3]);
        }
    }
    __syncthreads();

    constexpr int BN_U4 = BN / 8;
    constexpr int OUT_U4 = BM * BN_U4;
    for (int c = tid; c < OUT_U4; c += THREADS) {
        int row = c / BN_U4;
        int col = c % BN_U4;
        int gm = m0 + row;
        int gn = n0 + col * 8;
        if (gm < M && gn < Cout) {
            uint4 v = *reinterpret_cast<const uint4 *>(sout + row * BN + col * 8);
            long off = (long)gm * Cout + gn;
            if (res) {
                uint4 r_u4 = *reinterpret_cast<const uint4 *>(res + off);
                __half2 *vh = reinterpret_cast<__half2 *>(&v);
                const __half2 *rh = reinterpret_cast<const __half2 *>(&r_u4);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 f1 = __half22float2(vh[k]);
                    float2 f2 = __half22float2(rh[k]);
                    vh[k] = __floats2half2_rn(f1.x + f2.x, f1.y + f2.y);
                }
            }
            *reinterpret_cast<uint4 *>(Y + off) = v;
        }
    }
}

__global__ void k_trilinear_concat_gn_stats(const __half *low, const __half *skip, __half *cat,
    float *gn_acc, int B, int Di, int Hi, int Wi, int Do, int Ho, int Wo, int C) {
    __shared__ float sm_su[8][48], sm_qu[8][48], sm_ss[8][48], sm_qs[8][48];
    int b = blockIdx.z;
    int c0 = blockIdx.y * 48;
    int chunk = blockIdx.x;
    int nchunks = gridDim.x;
    int tid = threadIdx.x;
    int pl = tid / 24;
    int cl2 = tid % 24;
    int cl = cl2 << 1;
    int c2 = (c0 >> 1) + cl2;
    int half_c = C >> 1;
    int C2 = C * 2;
    int half_c2 = C;
    int cpg = C2 / 32;
    int DHWo = Do * Ho * Wo;
    int pix_per_chunk = (DHWo + nchunks - 1) / nchunks;
    int p0 = chunk * pix_per_chunk;
    int p1 = min(DHWo, p0 + pix_per_chunk);

    const __half2 *lb = reinterpret_cast<const __half2 *>(low + (long)b * Di * Hi * Wi * C);
    const __half2 *sb = reinterpret_cast<const __half2 *>(skip + (long)b * DHWo * C);
    __half2 *cb = reinterpret_cast<__half2 *>(cat + (long)b * DHWo * C2);

    int logW = __ffs(Wo) - 1, logH = __ffs(Ho) - 1;
    int maskW = Wo - 1, maskH = Ho - 1;

    float su0 = 0.f, qu0 = 0.f, su1 = 0.f, qu1 = 0.f;
    float ss0 = 0.f, qs0 = 0.f, ss1 = 0.f, qs1 = 0.f;
    for (int pix = p0 + pl; pix < p1; pix += 8) {
        int ox = pix & maskW;
        int oy = (pix >> logW) & maskH;
        int oz = pix >> (logW + logH);

        float fy = 0.5f * (float)oy - 0.25f;
        int iy0 = fy <= 0.0f ? 0 : (fy >= (float)(Hi - 1) ? Hi - 1 : (int)floorf(fy));
        int iy1 = (fy <= 0.0f || fy >= (float)(Hi - 1)) ? iy0 : iy0 + 1;
        float wy1 = (fy <= 0.0f || fy >= (float)(Hi - 1)) ? 0.0f : fy - (float)iy0;

        float fx = 0.5f * (float)ox - 0.25f;
        int ix0 = fx <= 0.0f ? 0 : (fx >= (float)(Wi - 1) ? Wi - 1 : (int)floorf(fx));
        int ix1 = (fx <= 0.0f || fx >= (float)(Wi - 1)) ? ix0 : ix0 + 1;
        float wx1 = (fx <= 0.0f || fx >= (float)(Wi - 1)) ? 0.0f : fx - (float)ix0;

        float wy0 = 1.0f - wy1, wx0 = 1.0f - wx1;
        float vu0, vu1;
        if (Do == Di) {
            long zoff = (long)oz * Hi * Wi * half_c + c2;
            float2 v00 = __half22float2(lb[zoff + ((long)iy0 * Wi + ix0) * half_c]);
            float2 v01 = __half22float2(lb[zoff + ((long)iy0 * Wi + ix1) * half_c]);
            float2 v10 = __half22float2(lb[zoff + ((long)iy1 * Wi + ix0) * half_c]);
            float2 v11 = __half22float2(lb[zoff + ((long)iy1 * Wi + ix1) * half_c]);
            vu0 = wy0 * (wx0 * v00.x + wx1 * v01.x) + wy1 * (wx0 * v10.x + wx1 * v11.x);
            vu1 = wy0 * (wx0 * v00.y + wx1 * v01.y) + wy1 * (wx0 * v10.y + wx1 * v11.y);
        } else {
            float fz = 0.5f * (float)oz - 0.25f;
            int iz0 = fz <= 0.0f ? 0 : (fz >= (float)(Di - 1) ? Di - 1 : (int)floorf(fz));
            int iz1 = (fz <= 0.0f || fz >= (float)(Di - 1)) ? iz0 : iz0 + 1;
            float wz1 = (fz <= 0.0f || fz >= (float)(Di - 1)) ? 0.0f : fz - (float)iz0;
            float wz0 = 1.0f - wz1;
            long z0off = (long)iz0 * Hi * Wi * half_c + c2;
            long z1off = (long)iz1 * Hi * Wi * half_c + c2;
            long o00 = ((long)iy0 * Wi + ix0) * half_c, o01 = ((long)iy0 * Wi + ix1) * half_c;
            long o10 = ((long)iy1 * Wi + ix0) * half_c, o11 = ((long)iy1 * Wi + ix1) * half_c;
            float2 p00 = __half22float2(lb[z0off + o00]);
            float2 p01 = __half22float2(lb[z0off + o01]);
            float2 p10 = __half22float2(lb[z0off + o10]);
            float2 p11 = __half22float2(lb[z0off + o11]);
            float a0x = wy0 * (wx0 * p00.x + wx1 * p01.x) + wy1 * (wx0 * p10.x + wx1 * p11.x);
            float a0y = wy0 * (wx0 * p00.y + wx1 * p01.y) + wy1 * (wx0 * p10.y + wx1 * p11.y);
            float2 q00 = __half22float2(lb[z1off + o00]);
            float2 q01 = __half22float2(lb[z1off + o01]);
            float2 q10 = __half22float2(lb[z1off + o10]);
            float2 q11 = __half22float2(lb[z1off + o11]);
            float a1x = wy0 * (wx0 * q00.x + wx1 * q01.x) + wy1 * (wx0 * q10.x + wx1 * q11.x);
            float a1y = wy0 * (wx0 * q00.y + wx1 * q01.y) + wy1 * (wx0 * q10.y + wx1 * q11.y);
            vu0 = wz0 * a0x + wz1 * a1x;
            vu1 = wz0 * a0y + wz1 * a1y;
        }
        __half2 hs = sb[(long)pix * half_c + c2];
        float2 vs = __half22float2(hs);
        cb[(long)pix * half_c2 + c2] = __floats2half2_rn(vu0, vu1);
        cb[(long)pix * half_c2 + half_c + c2] = hs;
        su0 += vu0;
        qu0 = fmaf(vu0, vu0, qu0);
        su1 += vu1;
        qu1 = fmaf(vu1, vu1, qu1);
        ss0 += vs.x;
        qs0 = fmaf(vs.x, vs.x, qs0);
        ss1 += vs.y;
        qs1 = fmaf(vs.y, vs.y, qs1);
    }

    sm_su[pl][cl] = su0;
    sm_su[pl][cl + 1] = su1;
    sm_qu[pl][cl] = qu0;
    sm_qu[pl][cl + 1] = qu1;
    sm_ss[pl][cl] = ss0;
    sm_ss[pl][cl + 1] = ss1;
    sm_qs[pl][cl] = qs0;
    sm_qs[pl][cl + 1] = qs1;
    __syncthreads();
    if (tid < 48) {
        float a = 0.f, bq = 0.f, cs = 0.f, dq = 0.f;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            a += sm_su[p][tid];
            bq += sm_qu[p][tid];
            cs += sm_ss[p][tid];
            dq += sm_qs[p][tid];
        }
        sm_su[0][tid] = a;
        sm_qu[0][tid] = bq;
        sm_ss[0][tid] = cs;
        sm_qs[0][tid] = dq;
    }
    __syncthreads();
    if (tid < 32) {
        float gs = 0.f, gq = 0.f;
        for (int i = 0; i < 48; ++i) {
            if ((c0 + i) / cpg == tid) {
                gs += sm_su[0][i];
                gq += sm_qu[0][i];
            }
            if ((C + c0 + i) / cpg == tid) {
                gs += sm_ss[0][i];
                gq += sm_qs[0][i];
            }
        }
        if (gs != 0.f || gq != 0.f) {
            atomicAdd(&gn_acc[(b * 32 + tid) * 2], gs);
            atomicAdd(&gn_acc[(b * 32 + tid) * 2 + 1], gq);
        }
    }
}

__global__ void k_head_logits(const __half *x, const float *gn_params, const float *gn_gamma,
    const float *gn_beta, const float *head_w, float head_b, const float *depth_w, int B,
    float *out_logits) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * 128 * 128;
    if (idx >= total)
        return;
    int ox = (int)(idx & 127);
    int oy = (int)((idx >> 7) & 127);
    int b = (int)(idx >> 14);
    float acc = 0.0f;
    for (int z = 0; z < 16; ++z) {
        const __half *p = x + ((((long)b * 16 + z) * 128 + oy) * 128 + ox) * 48;
        float dot = head_b;
#pragma unroll
        for (int c = 0; c < 48; ++c) {
            int g = c / 3;
            float mean = gn_params[(b * 16 + g) * 2];
            float istd = gn_params[(b * 16 + g) * 2 + 1];
            float v = (__half2float(p[c]) - mean) * istd * gn_gamma[c] + gn_beta[c];
            dot = fmaf(silu_f(v), head_w[c], dot);
        }
        acc = fmaf(dot, depth_w[z], acc);
    }
    out_logits[idx] = acc;
}

__global__ void k_head_blend(const __half *x, const float *gn_params, const float *gn_gamma,
    const float *gn_beta, const float *head_w, float head_b, const float *depth_w,
    const float *hann96, const int *coords, int B, int H, int W, float *out_sum, float *out_wgt) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * 96 * 96;
    if (idx >= total)
        return;
    int ox = (int)(idx % 96);
    int t = (int)(idx / 96);
    int oy = t % 96;
    int b = t / 96;
    int gy = coords[2 * b] + oy;
    int gx = coords[2 * b + 1] + ox;
    if (gy >= H || gx >= W)
        return;
    int ty = oy + 16, tx = ox + 16;
    float acc = 0.0f;
    for (int z = 0; z < 16; ++z) {
        const __half *p = x + ((((long)b * 16 + z) * 128 + ty) * 128 + tx) * 48;
        float dot = head_b;
#pragma unroll
        for (int c = 0; c < 48; ++c) {
            int g = c / 3;
            float mean = gn_params[(b * 16 + g) * 2];
            float istd = gn_params[(b * 16 + g) * 2 + 1];
            float v = (__half2float(p[c]) - mean) * istd * gn_gamma[c] + gn_beta[c];
            dot = fmaf(silu_f(v), head_w[c], dot);
        }
        acc = fmaf(dot, depth_w[z], acc);
    }
    float prob = 1.0f / (1.0f + __expf(-acc));
    float hw = hann96[oy] * hann96[ox];
    long goff = (long)gy * W + gx;
    atomicAdd(&out_sum[goff], prob * hw);
    atomicAdd(&out_wgt[goff], hw);
}

__global__ void k_empty_mask(const uint8_t *vol, int H, int W, uint8_t *empty) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long HW = (long)H * W;
    if (idx >= HW)
        return;
    uint8_t any = 0;
    for (int z = 0; z < 16; ++z)
        any |= vol[(long)z * HW + idx];
    empty[idx] = any == 0 ? 1 : 0;
}

__global__ void k_finalize_map(float *sum, const float *wgt, const uint8_t *empty, long HW) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW)
        return;
    if (empty[i]) {
        sum[i] = 0.0f;
        return;
    }
    float w = wgt[i];
    sum[i] = w > 0.0f ? sum[i] / fmaxf(w, 1e-12f) : 0.0f;
}

struct GNWeights {
    float *gamma;
    float *beta;
};

struct WinoConv {
    __half *U;
    float *scale;
    int alpha;
};

struct ResBlockW {
    GNWeights gn0, gn1;
    WinoConv w0, w1;
    __half *d0, *d1;
};

struct StudioEngine {
    StudioConfig cfg;
    cudaStream_t stream;
    std::vector<void *> allocs;

    float *stem_w;
    ResBlockW enc[5][6];
    GNWeights down_gn[4];
    __half *down_w[4];
    __half *up_w[4];
    GNWeights fuse_gn[4];
    __half *fuse_w[4];
    ResBlockW dec[4][3];
    GNWeights head_gn;
    float *head_w;
    float head_b;
    float *depth_w;
    float *hann96;

    float *d_in_f32;
    __half *d_a, *d_b, *d_cat, *d_aux;
    __half *d_skip[4];
    __half *d_V, *d_M;
    float *d_gn_acc;
    float *d_gn_params;
    float *d_logits;
    int *d_coords;

    uint8_t *d_vol;
    uint8_t *d_empty;
    float *d_out_sum;
    float *d_out_wgt;
    size_t vol_cap_hw;

    cudaEvent_t ev[24];
    double prof_ms[12];
    int prof_tiles;

    void *gpu_alloc(size_t bytes) {
        void *p = nullptr;
        CK(cudaMalloc(&p, bytes ? bytes : 16));
        allocs.push_back(p);
        return p;
    }
};

static int gcd_int(int a, int b) {
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static float *upload_f32(StudioEngine *e, const StudioTensor *t) {
    size_t bytes = (size_t)t->elements * sizeof(float);
    float *d = (float *)e->gpu_alloc(bytes);
    CK(cudaMemcpy(d, t->data, bytes, cudaMemcpyHostToDevice));
    return d;
}

static __half *upload_f16(StudioEngine *e, const StudioTensor *t) {
    float *tmp = nullptr;
    CK(cudaMalloc(&tmp, (size_t)t->elements * sizeof(float)));
    CK(cudaMemcpy(tmp, t->data, (size_t)t->elements * sizeof(float), cudaMemcpyHostToDevice));
    __half *d = (__half *)e->gpu_alloc((size_t)t->elements * sizeof(__half));
    int grd = (int)((t->elements + 255) / 256);
    k_f32_to_f16<<<grd, 256>>>(tmp, d, t->elements);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(tmp));
    return d;
}

static __half *upload_direct_w(StudioEngine *e, const StudioTensor *t, int Cout, int Cin) {
    float *tmp = nullptr;
    size_t bytes = (size_t)t->elements * sizeof(float);
    CK(cudaMalloc(&tmp, bytes));
    CK(cudaMemcpy(tmp, t->data, bytes, cudaMemcpyHostToDevice));
    __half *d = (__half *)e->gpu_alloc((size_t)t->elements * sizeof(__half));
    int grd = (int)((t->elements + 255) / 256);
    k_reorder_direct_w<<<grd, 256>>>(tmp, Cout, Cin, d);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(tmp));
    return d;
}

static WinoConv upload_wino_w(
    StudioEngine *e, const StudioTensor *t, int Cout, int Cin, int alpha) {
    int planes = alpha * alpha * alpha;
    long stride = (long)Cout * Cin;
    float *d_src = nullptr, *d_uf32 = nullptr, *d_pmax = nullptr;
    CK(cudaMalloc(&d_src, (size_t)t->elements * sizeof(float)));
    CK(cudaMemcpy(d_src, t->data, (size_t)t->elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_uf32, (size_t)planes * stride * sizeof(float)));
    CK(cudaMalloc(&d_pmax, (size_t)planes * sizeof(float)));
    CK(cudaMemset(d_pmax, 0, (size_t)planes * sizeof(float)));
    int grd = (int)((stride + 255) / 256);
    k_transform_wino_w<<<grd, 256>>>(d_src, Cout, Cin, alpha, d_uf32, d_pmax);
    __half *d_u = (__half *)e->gpu_alloc((size_t)planes * stride * sizeof(__half));
    float *d_sc = (float *)e->gpu_alloc((size_t)planes * sizeof(float));
    int qg = (int)std::min<long>((stride + 255) / 256, 256);
    k_quantize_wino_w<<<dim3(qg, planes), 256>>>(d_uf32, d_pmax, planes, stride, d_u, d_sc);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_src));
    CK(cudaFree(d_uf32));
    CK(cudaFree(d_pmax));
    return WinoConv{d_u, d_sc, alpha};
}

static WinoConv upload_wino2d_w(StudioEngine *e, const StudioTensor *t, int Cout, int Cin) {
    int planes = 36;
    long stride = (long)3 * Cout * Cin;
    float *d_src = nullptr, *d_uf32 = nullptr, *d_pmax = nullptr;
    CK(cudaMalloc(&d_src, (size_t)t->elements * sizeof(float)));
    CK(cudaMemcpy(d_src, t->data, (size_t)t->elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_uf32, (size_t)planes * stride * sizeof(float)));
    CK(cudaMalloc(&d_pmax, (size_t)planes * sizeof(float)));
    CK(cudaMemset(d_pmax, 0, (size_t)planes * sizeof(float)));
    int grd = (int)(((long)Cout * Cin + 255) / 256);
    k_transform_wino2d_w<<<grd, 256>>>(d_src, Cout, Cin, d_uf32, d_pmax);
    __half *d_u = (__half *)e->gpu_alloc((size_t)planes * stride * sizeof(__half));
    float *d_sc = (float *)e->gpu_alloc((size_t)planes * sizeof(float));
    int qg = (int)std::min<long>((stride + 255) / 256, 256);
    k_quantize_wino_w<<<dim3(qg, planes), 256>>>(d_uf32, d_pmax, planes, stride, d_u, d_sc);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_src));
    CK(cudaFree(d_uf32));
    CK(cudaFree(d_pmax));
    return WinoConv{d_u, d_sc, 24};
}

static GNWeights load_gn(StudioEngine *e, const StudioWeights *w, const char *prefix) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    float *g = upload_f32(e, studio_weights_find(w, name));
    snprintf(name, sizeof(name), "%s.bias", prefix);
    float *b = upload_f32(e, studio_weights_find(w, name));
    return GNWeights{g, b};
}

static void launch_gemm(
    StudioEngine *e, const __half *A, const __half *B, __half *C, int M, int N, int K, int planes) {
    if (N == 48) {
        size_t sm = 2 * (128 + 48) * 32 * 2;
        dim3 grd((M + 127) / 128, 1, planes);
        k_gemm_batched_f16<128, 48, 4, 1, 2><<<grd, 128, sm, e->stream>>>(A, B, C, M, N, K);
    } else if (N % 96 == 0) {
        size_t sm = 2 * (128 + 96) * 32 * 2;
        dim3 grd((M + 127) / 128, N / 96, planes);
        k_gemm_batched_f16<128, 96, 4, 2, 2><<<grd, 256, sm, e->stream>>>(A, B, C, M, N, K);
    } else {
        size_t sm = 2 * (128 + 128) * 32 * 2;
        dim3 grd((M + 127) / 128, (N + 127) / 128, planes);
        k_gemm_batched_f16<128, 128, 4, 2, 2><<<grd, 256, sm, e->stream>>>(A, B, C, M, N, K);
    }
}

static void launch_fuse_gemm(StudioEngine *e, const __half *A, const GNWeights &gn, const __half *B,
    __half *C, int B_batch, int DHW, int N, int K) {
    int M = B_batch * DHW;
    int G_in = gcd_int(32, K);
    int G_out = gcd_int(32, N);
    size_t param_bytes = ((size_t)2 * K * sizeof(float) + 127) & ~(size_t)127;
    CK(cudaMemsetAsync(e->d_gn_acc, 0, (size_t)B_batch * G_out * 2 * sizeof(float), e->stream));
    if (N == 48) {
        size_t sm = param_bytes + 2 * (128 + 48) * 32 * 2;
        dim3 grd((M + 127) / 128, 1);
        k_gemm_fuse_f16<128, 48, 4, 1, 2><<<grd, 128, sm, e->stream>>>(
            A, e->d_gn_params, gn.gamma, gn.beta, B, C, e->d_gn_acc, DHW, M, N, K, G_in, G_out);
    } else if (N % 96 == 0) {
        size_t sm = param_bytes + 2 * (128 + 96) * 32 * 2;
        dim3 grd((M + 127) / 128, N / 96);
        k_gemm_fuse_f16<128, 96, 4, 2, 1><<<grd, 256, sm, e->stream>>>(
            A, e->d_gn_params, gn.gamma, gn.beta, B, C, e->d_gn_acc, DHW, M, N, K, G_in, G_out);
    } else {
        size_t sm = param_bytes + 2 * (128 + 128) * 32 * 2;
        dim3 grd((M + 127) / 128, (N + 127) / 128);
        k_gemm_fuse_f16<128, 128, 4, 2, 1><<<grd, 256, sm, e->stream>>>(
            A, e->d_gn_params, gn.gamma, gn.beta, B, C, e->d_gn_acc, DHW, M, N, K, G_in, G_out);
    }
}

static void launch_direct_conv(StudioEngine *e, const __half *X, const __half *Wt,
    const __half *res, __half *Y, int B, int Di, int Hi, int Wi, int Do, int Ho, int Wo, int sz,
    int sy, int sx, int Cin, int Cout) {
    int M = B * Do * Ho * Wo;
    if (Cout % 96 == 0) {
        size_t sm = 2 * (128 + 96) * 32 * 2;
        dim3 grd((M + 127) / 128, Cout / 96);
        k_conv3d_direct_mma<128, 96, 4, 2, 2><<<grd, 256, sm, e->stream>>>(
            X, Wt, res, Y, B, Di, Hi, Wi, Do, Ho, Wo, sz, sy, sx, Cin, Cout);
    } else {
        size_t sm = 2 * (128 + 128) * 32 * 2;
        dim3 grd((M + 127) / 128, (Cout + 127) / 128);
        k_conv3d_direct_mma<128, 128, 4, 2, 2><<<grd, 256, sm, e->stream>>>(
            X, Wt, res, Y, B, Di, Hi, Wi, Do, Ho, Wo, sz, sy, sx, Cin, Cout);
    }
}

static void zero_gn_acc(StudioEngine *e, int B, int G) {
    CK(cudaMemsetAsync(e->d_gn_acc, 0, (size_t)B * G * 2 * sizeof(float), e->stream));
}

static void run_gn_finalize(StudioEngine *e, int B, int D, int H, int W, int C) {
    int G = gcd_int(32, C);
    float cnt = (float)((long)D * H * W * (C / G));
    int BG = B * G;
    k_gn_finalize_f32<<<(BG + 63) / 64, 64, 0, e->stream>>>(e->d_gn_acc, e->d_gn_params, BG, cnt);
}

static void run_gn_stats(StudioEngine *e, const __half *x, int B, int D, int H, int W, int C) {
    int G = gcd_int(32, C);
    int DHW = D * H * W;
    zero_gn_acc(e, B, G);
    int chunks = DHW >= 65536 ? 32 : (DHW >= 8192 ? 16 : 4);
    k_gn_stats_f16<<<dim3(chunks, B), 192, 0, e->stream>>>(x, e->d_gn_acc, B, DHW, C, G);
    run_gn_finalize(e, B, D, H, W, C);
}

static void run_gn_silu(StudioEngine *e, const __half *x, const GNWeights &gn, __half *y, int B,
    int D, int H, int W, int C) {
    int G = gcd_int(32, C);
    int DHW = D * H * W;
    int chunks = DHW >= 65536 ? 128 : (DHW >= 8192 ? 64 : 16);
    k_gn_silu_f16<<<dim3(chunks, B), 192, 0, e->stream>>>(
        x, e->d_gn_params, gn.gamma, gn.beta, y, B, DHW, C, G);
}

static void run_wino2d_conv(StudioEngine *e, const __half *x, const GNWeights &gn,
    const WinoConv &wc, const __half *res, __half *out, int B, int stage, int need_gn_out) {
    if (stage == 0) {
        int D = 16, H = 128, W = 128, tw = 32, th = 32, tiles_yx = 1024;
        dim3 g_io(tw / 4, th, B * D);
        k_wino2d44_in_fused<48, 4, 16>
            <<<g_io, 192, 0, e->stream>>>(x, e->d_gn_params, gn.gamma, gn.beta, e->d_V, B, D, H, W);
        dim3 g_m(D, tiles_yx / 64, B * 36);
        k_gemm_wino2d_c48<<<g_m, 128, 0, e->stream>>>(e->d_V, wc.U, e->d_M, B, D, tiles_yx);
        if (need_gn_out)
            zero_gn_acc(e, B, 16);
        k_wino2d44_out_fused<48, 4, 16><<<g_io, 192, 0, e->stream>>>(
            e->d_M, wc.scale, res, out, need_gn_out ? e->d_gn_acc : nullptr, B, D, H, W);
        if (need_gn_out)
            run_gn_finalize(e, B, D, H, W, 48);
    } else {
        int D = 16, H = 64, W = 64, tw = 16, th = 16, tiles_yx = 256;
        dim3 g_io(tw / 2, th, B * D);
        k_wino2d44_in_fused<96, 2, 32>
            <<<g_io, 192, 0, e->stream>>>(x, e->d_gn_params, gn.gamma, gn.beta, e->d_V, B, D, H, W);
        dim3 g_m(D, tiles_yx / 128, B * 36);
        size_t sm = 2 * (128 + 96) * 32 * 2;
        k_gemm_wino2d_c96<<<g_m, 256, sm, e->stream>>>(e->d_V, wc.U, e->d_M, B, D, tiles_yx);
        if (need_gn_out)
            zero_gn_acc(e, B, 32);
        k_wino2d44_out_fused<96, 2, 32><<<g_io, 192, 0, e->stream>>>(
            e->d_M, wc.scale, res, out, need_gn_out ? e->d_gn_acc : nullptr, B, D, H, W);
        if (need_gn_out)
            run_gn_finalize(e, B, D, H, W, 96);
    }
}

static void run_wino3d_conv(StudioEngine *e, const __half *x, const GNWeights &gn,
    const WinoConv &wc, const __half *res, __half *out, int B, int D, int H, int W, int C,
    int need_gn_out) {
    if (wc.alpha == 6) {
        int P = B * (D >> 2) * (H >> 2) * (W >> 2);
        dim3 grd(P, C / 64);
        k_wino444_in_ch64_fused<<<grd, 192, 0, e->stream>>>(
            x, e->d_gn_params, gn.gamma, gn.beta, e->d_V, B, D, H, W, C);
        launch_gemm(e, e->d_V, wc.U, e->d_M, P, C, C, 216);
        if (need_gn_out)
            zero_gn_acc(e, B, 32);
        k_wino444_out_ch64_fused<<<grd, 192, 0, e->stream>>>(
            e->d_M, wc.scale, res, out, need_gn_out ? e->d_gn_acc : nullptr, B, D, H, W, C);
        if (need_gn_out)
            run_gn_finalize(e, B, D, H, W, C);
    } else {
        int P = B * (D >> 1) * (H >> 1) * (W >> 1);
        dim3 grd(P, C / 64);
        k_wino222_in_ch64_fused<<<grd, 256, 0, e->stream>>>(
            x, e->d_gn_params, gn.gamma, gn.beta, e->d_V, B, D, H, W, C);
        launch_gemm(e, e->d_V, wc.U, e->d_M, P, C, C, 64);
        if (need_gn_out)
            zero_gn_acc(e, B, 32);
        k_wino222_out_ch64_fused<<<grd, 256, 0, e->stream>>>(
            e->d_M, wc.scale, res, out, need_gn_out ? e->d_gn_acc : nullptr, B, D, H, W, C);
        if (need_gn_out)
            run_gn_finalize(e, B, D, H, W, C);
    }
}

static void run_direct_resblock(StudioEngine *e, __half *x, __half *tmp, const ResBlockW &rb, int B,
    int D, int H, int W, int C, int need_gn_out) {
    run_gn_silu(e, x, rb.gn0, e->d_aux, B, D, H, W, C);
    launch_direct_conv(e, e->d_aux, rb.d0, nullptr, tmp, B, D, H, W, D, H, W, 1, 1, 1, C, C);
    run_gn_stats(e, tmp, B, D, H, W, C);
    run_gn_silu(e, tmp, rb.gn1, e->d_aux, B, D, H, W, C);
    launch_direct_conv(e, e->d_aux, rb.d1, x, tmp, B, D, H, W, D, H, W, 1, 1, 1, C, C);
    if (need_gn_out)
        run_gn_stats(e, tmp, B, D, H, W, C);
}

static void run_backbone(StudioEngine *e, int B) {
    const int widths[5] = {48, 96, 192, 384, 640};
    const int ecounts[5] = {2, 3, 4, 5, 6};
    const int dcounts[4] = {3, 3, 2, 2};
    const int ds[5] = {16, 16, 8, 4, 2};
    const int hs[5] = {128, 64, 32, 16, 8};

    auto mark = [&](int idx) {
        if (e->cfg.profile)
            CK(cudaEventRecord(e->ev[idx], e->stream));
    };

    mark(0);
    zero_gn_acc(e, B, 16);
    k_stem_conv3d_gn<<<dim3(8, 8, B * 16), 256, 0, e->stream>>>(
        e->d_in_f32, e->stem_w, B, e->d_a, e->d_gn_acc);
    run_gn_finalize(e, B, 16, 128, 128, 48);
    mark(1);

    __half *cur = e->d_a, *alt = e->d_b;
    for (int s = 0; s < 5; ++s) {
        int C = widths[s], D = ds[s], H = hs[s], W = hs[s];
        for (int blk = 0; blk < ecounts[s]; ++blk) {
            int need_out_gn = !(s == 4 && blk == ecounts[s] - 1);
            if (s < 2) {
                run_wino2d_conv(
                    e, cur, e->enc[s][blk].gn0, e->enc[s][blk].w0, nullptr, alt, B, s, 1);
                run_wino2d_conv(
                    e, alt, e->enc[s][blk].gn1, e->enc[s][blk].w1, cur, cur, B, s, need_out_gn);
            } else if (s < 4) {
                run_wino3d_conv(
                    e, cur, e->enc[s][blk].gn0, e->enc[s][blk].w0, nullptr, alt, B, D, H, W, C, 1);
                run_wino3d_conv(e, alt, e->enc[s][blk].gn1, e->enc[s][blk].w1, cur, cur, B, D, H, W,
                    C, need_out_gn);
            } else {
                run_direct_resblock(e, cur, alt, e->enc[s][blk], B, D, H, W, C, need_out_gn);
                std::swap(cur, alt);
            }
        }
        if (s < 4) {
            size_t skip_bytes = (size_t)B * D * H * W * C * sizeof(__half);
            CK(cudaMemcpyAsync(e->d_skip[s], cur, skip_bytes, cudaMemcpyDeviceToDevice, e->stream));
            run_gn_silu(e, cur, e->down_gn[s], e->d_aux, B, D, H, W, C);
            int nextC = widths[s + 1];
            int sz = s == 0 ? 1 : 2;
            launch_direct_conv(e, e->d_aux, e->down_w[s], nullptr, alt, B, D, H, W, ds[s + 1],
                hs[s + 1], hs[s + 1], sz, 2, 2, C, nextC);
            run_gn_stats(e, alt, B, ds[s + 1], hs[s + 1], hs[s + 1], nextC);
            std::swap(cur, alt);
        }
        mark(2 + s);
    }

    for (int u = 0; u < 4; ++u) {
        int s = 3 - u;
        int Cin = widths[s + 1], Cout = widths[s];
        int Di = ds[s + 1], Hi = hs[s + 1], Wi = hs[s + 1];
        int Do = ds[s], Ho = hs[s], Wo = hs[s];
        int Mi = B * Di * Hi * Wi;
        launch_gemm(e, cur, e->up_w[u], alt, Mi, Cout, Cin, 1);

        zero_gn_acc(e, B, 32);
        int DHWo = Do * Ho * Wo;
        int chunks = DHWo >= 65536 ? 32 : (DHWo >= 8192 ? 16 : 8);
        k_trilinear_concat_gn_stats<<<dim3(chunks, Cout / 48, B), 192, 0, e->stream>>>(
            alt, e->d_skip[s], e->d_cat, e->d_gn_acc, B, Di, Hi, Wi, Do, Ho, Wo, Cout);
        run_gn_finalize(e, B, Do, Ho, Wo, Cout * 2);

        launch_fuse_gemm(e, e->d_cat, e->fuse_gn[u], e->fuse_w[u], cur, B, DHWo, Cout, Cout * 2);
        run_gn_finalize(e, B, Do, Ho, Wo, Cout);

        for (int blk = 0; blk < dcounts[u]; ++blk) {
            if (s < 2) {
                run_wino2d_conv(
                    e, cur, e->dec[u][blk].gn0, e->dec[u][blk].w0, nullptr, alt, B, s, 1);
                run_wino2d_conv(e, alt, e->dec[u][blk].gn1, e->dec[u][blk].w1, cur, cur, B, s, 1);
            } else {
                run_wino3d_conv(e, cur, e->dec[u][blk].gn0, e->dec[u][blk].w0, nullptr, alt, B, Do,
                    Ho, Wo, Cout, 1);
                run_wino3d_conv(e, alt, e->dec[u][blk].gn1, e->dec[u][blk].w1, cur, cur, B, Do, Ho,
                    Wo, Cout, 1);
            }
        }
        mark(7 + u);
    }

    if (cur != e->d_a) {
        size_t bytes = (size_t)B * 16 * 128 * 128 * 48 * sizeof(__half);
        CK(cudaMemcpyAsync(e->d_a, cur, bytes, cudaMemcpyDeviceToDevice, e->stream));
    }
}

static void accum_profile(StudioEngine *e, int B) {
    if (!e->cfg.profile)
        return;
    CK(cudaStreamSynchronize(e->stream));
    for (int i = 0; i < 11; ++i) {
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e->ev[i], e->ev[i + 1]));
        e->prof_ms[i] += ms;
    }
    e->prof_tiles += B;
}

extern "C" StudioEngine *studio_engine_create(const StudioWeights *w, StudioConfig cfg) {
    StudioEngine *e = new StudioEngine();
    memset(e->prof_ms, 0, sizeof(e->prof_ms));
    e->prof_tiles = 0;
    e->d_vol = nullptr;
    e->d_empty = nullptr;
    e->d_out_sum = nullptr;
    e->d_out_wgt = nullptr;
    e->vol_cap_hw = 0;
    if (cfg.batch_size < 1)
        cfg.batch_size = 8;
    e->cfg = cfg;
    CK(cudaStreamCreate(&e->stream));
    for (int i = 0; i < 24; ++i)
        CK(cudaEventCreate(&e->ev[i]));

    e->stem_w = upload_f32(e, studio_weights_find(w, "stem.weight"));

    const int widths[5] = {48, 96, 192, 384, 640};
    const int ecounts[5] = {2, 3, 4, 5, 6};
    const int dcounts[4] = {3, 3, 2, 2};
    char name[128];

    for (int s = 0; s < 5; ++s) {
        int C = widths[s];
        int alpha = (s == 2 && cfg.use_f444) ? 6 : 4;
        for (int b = 0; b < ecounts[s]; ++b) {
            snprintf(name, sizeof(name), "enc.%d.%d.branch.0", s, b);
            e->enc[s][b].gn0 = load_gn(e, w, name);
            snprintf(name, sizeof(name), "enc.%d.%d.branch.3", s, b);
            e->enc[s][b].gn1 = load_gn(e, w, name);
            snprintf(name, sizeof(name), "enc.%d.%d.branch.2.weight", s, b);
            const StudioTensor *t0 = studio_weights_find(w, name);
            snprintf(name, sizeof(name), "enc.%d.%d.branch.6.weight", s, b);
            const StudioTensor *t1 = studio_weights_find(w, name);
            if (s < 2) {
                e->enc[s][b].w0 = upload_wino2d_w(e, t0, C, C);
                e->enc[s][b].w1 = upload_wino2d_w(e, t1, C, C);
                e->enc[s][b].d0 = nullptr;
                e->enc[s][b].d1 = nullptr;
            } else if (s < 4) {
                e->enc[s][b].w0 = upload_wino_w(e, t0, C, C, alpha);
                e->enc[s][b].w1 = upload_wino_w(e, t1, C, C, alpha);
                e->enc[s][b].d0 = nullptr;
                e->enc[s][b].d1 = nullptr;
            } else {
                e->enc[s][b].d0 = upload_direct_w(e, t0, C, C);
                e->enc[s][b].d1 = upload_direct_w(e, t1, C, C);
            }
        }
        if (s < 4) {
            snprintf(name, sizeof(name), "down.%d.0", s);
            e->down_gn[s] = load_gn(e, w, name);
            snprintf(name, sizeof(name), "down.%d.2.weight", s);
            e->down_w[s] = upload_direct_w(e, studio_weights_find(w, name), widths[s + 1], C);
        }
    }

    for (int u = 0; u < 4; ++u) {
        int s = 3 - u;
        int C = widths[s];
        int alpha = (s == 2 && cfg.use_f444) ? 6 : 4;
        snprintf(name, sizeof(name), "up.%d.weight", u);
        e->up_w[u] = upload_f16(e, studio_weights_find(w, name));
        snprintf(name, sizeof(name), "fuse.%d.0", u);
        e->fuse_gn[u] = load_gn(e, w, name);
        snprintf(name, sizeof(name), "fuse.%d.2.weight", u);
        e->fuse_w[u] = upload_f16(e, studio_weights_find(w, name));
        for (int b = 0; b < dcounts[u]; ++b) {
            snprintf(name, sizeof(name), "dec.%d.%d.branch.0", u, b);
            e->dec[u][b].gn0 = load_gn(e, w, name);
            snprintf(name, sizeof(name), "dec.%d.%d.branch.3", u, b);
            e->dec[u][b].gn1 = load_gn(e, w, name);
            snprintf(name, sizeof(name), "dec.%d.%d.branch.2.weight", u, b);
            const StudioTensor *t0 = studio_weights_find(w, name);
            snprintf(name, sizeof(name), "dec.%d.%d.branch.6.weight", u, b);
            const StudioTensor *t1 = studio_weights_find(w, name);
            if (s < 2) {
                e->dec[u][b].w0 = upload_wino2d_w(e, t0, C, C);
                e->dec[u][b].w1 = upload_wino2d_w(e, t1, C, C);
                e->dec[u][b].d0 = nullptr;
                e->dec[u][b].d1 = nullptr;
            } else {
                e->dec[u][b].w0 = upload_wino_w(e, t0, C, C, alpha);
                e->dec[u][b].w1 = upload_wino_w(e, t1, C, C, alpha);
                e->dec[u][b].d0 = nullptr;
                e->dec[u][b].d1 = nullptr;
            }
        }
    }

    e->head_gn = load_gn(e, w, "head.0");
    e->head_w = upload_f32(e, studio_weights_find(w, "head.2.weight"));
    e->head_b = studio_weights_find(w, "head.2.bias")->data[0];

    const StudioTensor *dl = studio_weights_find(w, "depth_logits");
    float dw[16], mx = dl->data[0], sum = 0.0f;
    for (int i = 1; i < 16; ++i)
        if (dl->data[i] > mx)
            mx = dl->data[i];
    for (int i = 0; i < 16; ++i) {
        dw[i] = expf(dl->data[i] - mx);
        sum += dw[i];
    }
    for (int i = 0; i < 16; ++i)
        dw[i] /= sum;
    e->depth_w = (float *)e->gpu_alloc(16 * sizeof(float));
    CK(cudaMemcpy(e->depth_w, dw, 16 * sizeof(float), cudaMemcpyHostToDevice));

    float hann[96];
    for (int i = 0; i < 96; ++i)
        hann[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979323846f * ((float)i + 0.5f) / 96.0f);
    e->hann96 = (float *)e->gpu_alloc(96 * sizeof(float));
    CK(cudaMemcpy(e->hann96, hann, 96 * sizeof(float), cudaMemcpyHostToDevice));

    int B = cfg.batch_size;
    size_t s0 = (size_t)B * 16 * 128 * 128 * 48;
    size_t s1 = (size_t)B * 16 * 64 * 64 * 96;
    size_t s2 = (size_t)B * 8 * 32 * 32 * 192;
    size_t s3 = (size_t)B * 4 * 16 * 16 * 384;
    e->d_in_f32 = (float *)e->gpu_alloc((size_t)B * 16 * 128 * 128 * sizeof(float));
    e->d_a = (__half *)e->gpu_alloc(s0 * sizeof(__half));
    e->d_b = (__half *)e->gpu_alloc(s0 * sizeof(__half));
    e->d_aux = (__half *)e->gpu_alloc(s0 * sizeof(__half));
    e->d_cat = (__half *)e->gpu_alloc(s0 * 2 * sizeof(__half));
    e->d_skip[0] = (__half *)e->gpu_alloc(s0 * sizeof(__half));
    e->d_skip[1] = (__half *)e->gpu_alloc(s1 * sizeof(__half));
    e->d_skip[2] = (__half *)e->gpu_alloc(s2 * sizeof(__half));
    e->d_skip[3] = (__half *)e->gpu_alloc(s3 * sizeof(__half));

    size_t wino_elems = (size_t)36 * B * 16 * 1024 * 48;
    e->d_V = (__half *)e->gpu_alloc(wino_elems * sizeof(__half));
    e->d_M = (__half *)e->gpu_alloc(wino_elems * sizeof(__half));
    e->d_gn_acc = (float *)e->gpu_alloc((size_t)B * 32 * 2 * sizeof(float));
    e->d_gn_params = (float *)e->gpu_alloc((size_t)B * 32 * 2 * sizeof(float));
    e->d_logits = (float *)e->gpu_alloc((size_t)B * 128 * 128 * sizeof(float));
    e->d_coords = (int *)e->gpu_alloc((size_t)B * 2 * sizeof(int));
    return e;
}

extern "C" void studio_engine_destroy(StudioEngine *e) {
    if (!e)
        return;
    for (int i = 0; i < 24; ++i)
        cudaEventDestroy(e->ev[i]);
    cudaStreamDestroy(e->stream);
    for (void *p : e->allocs)
        cudaFree(p);
    if (e->d_vol)
        cudaFree(e->d_vol);
    if (e->d_empty)
        cudaFree(e->d_empty);
    if (e->d_out_sum)
        cudaFree(e->d_out_sum);
    if (e->d_out_wgt)
        cudaFree(e->d_out_wgt);
    delete e;
}

extern "C" void studio_engine_forward_tiles(
    StudioEngine *e, const float *input_ncdhw, int batch, float *output_n1hw) {
    memset(e->prof_ms, 0, sizeof(e->prof_ms));
    e->prof_tiles = 0;
    for (int off = 0; off < batch; off += e->cfg.batch_size) {
        int b = std::min(e->cfg.batch_size, batch - off);
        size_t in_bytes = (size_t)b * 16 * 128 * 128 * sizeof(float);
        CK(cudaMemcpyAsync(e->d_in_f32, input_ncdhw + (size_t)off * 16 * 128 * 128, in_bytes,
            cudaMemcpyHostToDevice, e->stream));
        run_backbone(e, b);
        long total = (long)b * 128 * 128;
        k_head_logits<<<(total + 255) / 256, 256, 0, e->stream>>>(e->d_a, e->d_gn_params,
            e->head_gn.gamma, e->head_gn.beta, e->head_w, e->head_b, e->depth_w, b, e->d_logits);
        if (e->cfg.profile)
            CK(cudaEventRecord(e->ev[11], e->stream));
        accum_profile(e, b);
        CK(cudaMemcpyAsync(output_n1hw + (size_t)off * 128 * 128, e->d_logits,
            (size_t)total * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    }
    CK(cudaStreamSynchronize(e->stream));
}

extern "C" void studio_engine_predict(
    StudioEngine *e, const uint8_t *volume_dhw, int depth, int height, int width, float *out_hw) {
    (void)depth;
    memset(e->prof_ms, 0, sizeof(e->prof_ms));
    e->prof_tiles = 0;
    size_t HW = (size_t)height * (size_t)width;
    if (HW > e->vol_cap_hw) {
        if (e->d_vol)
            cudaFree(e->d_vol);
        if (e->d_empty)
            cudaFree(e->d_empty);
        if (e->d_out_sum)
            cudaFree(e->d_out_sum);
        if (e->d_out_wgt)
            cudaFree(e->d_out_wgt);
        CK(cudaMalloc(&e->d_vol, 16 * HW));
        CK(cudaMalloc(&e->d_empty, HW));
        CK(cudaMalloc(&e->d_out_sum, HW * sizeof(float)));
        CK(cudaMalloc(&e->d_out_wgt, HW * sizeof(float)));
        e->vol_cap_hw = HW;
    }
    CK(cudaMemcpyAsync(e->d_vol, volume_dhw, 16 * HW, cudaMemcpyHostToDevice, e->stream));
    CK(cudaMemsetAsync(e->d_out_sum, 0, HW * sizeof(float), e->stream));
    CK(cudaMemsetAsync(e->d_out_wgt, 0, HW * sizeof(float), e->stream));
    k_empty_mask<<<(HW + 255) / 256, 256, 0, e->stream>>>(e->d_vol, height, width, e->d_empty);

    std::vector<uint8_t> h_empty(HW);
    CK(cudaMemcpyAsync(h_empty.data(), e->d_empty, HW, cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));

    std::vector<int> ys, xs;
    for (int y = 0; y < height; y += 64)
        ys.push_back(y);
    int last_y = height > 96 ? height - 96 : 0;
    if (std::find(ys.begin(), ys.end(), last_y) == ys.end())
        ys.push_back(last_y);
    std::sort(ys.begin(), ys.end());

    for (int x = 0; x < width; x += 64)
        xs.push_back(x);
    int last_x = width > 96 ? width - 96 : 0;
    if (std::find(xs.begin(), xs.end(), last_x) == xs.end())
        xs.push_back(last_x);
    std::sort(xs.begin(), xs.end());

    std::vector<int> pts;
    for (int y : ys) {
        for (int x : xs) {
            int y1 = std::min(height, y + 96), x1 = std::min(width, x + 96);
            bool any = false;
            for (int yy = y; yy < y1 && !any; ++yy)
                for (int xx = x; xx < x1; ++xx)
                    if (!h_empty[(size_t)yy * width + xx]) {
                        any = true;
                        break;
                    }
            if (any) {
                pts.push_back(y);
                pts.push_back(x);
            }
        }
    }

    int npts = (int)(pts.size() / 2);
    for (int off = 0; off < npts; off += e->cfg.batch_size) {
        int b = std::min(e->cfg.batch_size, npts - off);
        CK(cudaMemcpyAsync(e->d_coords, pts.data() + off * 2, (size_t)b * 2 * sizeof(int),
            cudaMemcpyHostToDevice, e->stream));
        long crop_n = (long)b * 16 * 128 * 128;
        k_crop_tiles_u8<<<(crop_n + 255) / 256, 256, 0, e->stream>>>(
            e->d_vol, height, width, e->d_coords, b, e->d_in_f32);
        run_backbone(e, b);
        long head_n = (long)b * 96 * 96;
        k_head_blend<<<(head_n + 255) / 256, 256, 0, e->stream>>>(e->d_a, e->d_gn_params,
            e->head_gn.gamma, e->head_gn.beta, e->head_w, e->head_b, e->depth_w, e->hann96,
            e->d_coords, b, height, width, e->d_out_sum, e->d_out_wgt);
        if (e->cfg.profile)
            CK(cudaEventRecord(e->ev[11], e->stream));
        accum_profile(e, b);
    }

    k_finalize_map<<<(HW + 255) / 256, 256, 0, e->stream>>>(
        e->d_out_sum, e->d_out_wgt, e->d_empty, (long)HW);
    CK(cudaMemcpyAsync(
        out_hw, e->d_out_sum, HW * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));
}

extern "C" void studio_engine_print_profile(StudioEngine *e) {
    if (!e || e->prof_tiles == 0)
        return;
    const char *names[11] = {"stem", "enc0+d0", "enc1+d1", "enc2+d2", "enc3+d3", "enc4", "up0+dec0",
        "up1+dec1", "up2+dec2", "up3+dec3", "head"};
    double total = 0.0;
    for (int i = 0; i < 11; ++i)
        total += e->prof_ms[i];
    printf("profile (%d tiles, %.3f ms/tile total):\n", e->prof_tiles, total / e->prof_tiles);
    for (int i = 0; i < 11; ++i)
        printf("  %-10s %7.3f ms/tile (%5.1f%%)\n", names[i], e->prof_ms[i] / e->prof_tiles,
            100.0 * e->prof_ms[i] / (total > 0 ? total : 1.0));
}
