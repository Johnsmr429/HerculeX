#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <signal.h>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;
typedef uint64_t u64;

constexpr int Threads = 256;
constexpr int Warps = 8;
constexpr int SmemBytes = 96 * 1024;

static void fail(const std::string &m) {
    fprintf(stderr, "model: %s\n", m.c_str());
    exit(1);
}
#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess)                                                                     \
            fail(std::string(#x) + ": " + cudaGetErrorString(e_));                                 \
    } while (0)

__device__ __forceinline__ unsigned su32(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(unsigned s, const void *g, bool ok) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ void ldsm(unsigned *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldsmt(unsigned *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void mma(float *c, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int swz(int row, int rb) {
    return rb == 64 ? ((row >> 1) & 3) : (row & 7);
}
__device__ __forceinline__ unsigned sad(unsigned base, int row, int chunk, int rb) {
    return base + row * rb + ((chunk ^ swz(row, rb)) << 4);
}
__device__ __forceinline__ unsigned pack2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned *>(&v);
}
__host__ __device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__host__ __device__ __forceinline__ float gelu_d(float x) {
    float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608f * (1.f + 0.134145f * x * x);
}
__host__ __device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__host__ __device__ __forceinline__ float softplusf(float x) {
    return x > 20.f ? x : log1pf(expf(x));
}
__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ uint4 gelu8(uint4 v) {
    __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&v);
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(p[i]);
        p[i] = __floats2bfloat162_rn(gelu(f.x), gelu(f.y));
    }
    return v;
}
__device__ __forceinline__ int gtid() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ __forceinline__ int gstride() { return gridDim.x * blockDim.x; }

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
    u64 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return (next() >> 40) * (1.f / 16777216.f); }
    float sym() { return 2.f * uni() - 1.f; }
};

enum { S_DENSE = 0, S_CONVT = 1, S_CTX1 = 2, S_CTX2 = 3 };
struct Src {
    int mode = S_DENSE;
    const bf16 *p = nullptr;
    long ld = 0;
    int a = 0, b = 0, c = 0, d = 1, e = 1, f = 1;
};

__device__ __forceinline__ const bf16 *src_ptr(const Src &s, int r, int c) {
    if (s.mode == S_DENSE)
        return s.p + (long)r * s.ld + c;
    if (s.mode == S_CONVT) {
        int cout = (int)s.ld, tap = c / cout, co = c - tap * cout;
        int ix = r % s.c, t1 = r / s.c, iy = t1 % s.b, t2 = t1 / s.b, iz = t2 % s.a, bb = t2 / s.a;
        int tx = tap % s.f, ty = (tap / s.f) % s.e, tz = tap / (s.f * s.e);
        long o = (((long)bb * s.a * s.d + iz * s.d + tz) * (s.b * s.e) + iy * s.e + ty) *
                (long)(s.c * s.f) +
            ix * s.f + tx;
        return s.p + o * cout + co;
    }
    if (s.mode == S_CTX1) {
        int g = s.a / 4, ox = r % g, oy = (r / g) % g, bb = r / (g * g), ky = c / 16,
            kx = (c / 4) % 4;
        return s.p + (((long)bb * s.a + 4 * oy + ky) * s.a + 4 * ox + kx) * 4;
    }
    int g1 = s.a / 4, g2 = s.a / 16, ox = r % g2, oy = (r / g2) % g2, bb = r / (g2 * g2);
    int tap = c / 128, ci = c % 128, ky = tap / 4, kx = tap % 4;
    return s.p + (((long)bb * g1 + 4 * oy + ky) * g1 + 4 * ox + kx) * 128 + ci;
}

enum {
    E_F32 = 0,
    E_F32_ADD,
    E_BF16,
    E_F32_BF16,
    E_SWIGLU,
    E_CONVT,
    E_GELU_Z,
    E_GELU_GRAD,
    E_PART,
    E_CTX,
    E_CTX_DX
};
struct Epi {
    int mode = E_F32;
    float *f = nullptr;
    bf16 *h = nullptr, *h2 = nullptr;
    const float *bias = nullptr;
    const bf16 *z = nullptr;
    long ld = 0, ld2 = 0;
    int a = 0, b = 0, c = 0, d = 0, fz = 1, fy = 1, fx = 1;
    const float *box = nullptr;
    const float *marker = nullptr;
};

__device__ __forceinline__ void epi2(
    const Epi &e, int M, int N, int split, int m, int n, float v0, float v1) {
    if (m >= M || n >= N)
        return;
    if (e.bias && e.mode != E_PART && e.mode != E_CONVT && e.mode != E_CTX_DX) {
        v0 += e.bias[n];
        v1 += e.bias[n + 1];
    }
    long o = (long)m * e.ld + n;
    switch (e.mode) {
    case E_F32: *reinterpret_cast<float2 *>(e.f + o) = make_float2(v0, v1); break;
    case E_F32_ADD: {
        float2 *p = reinterpret_cast<float2 *>(e.f + o);
        float2 q = *p;
        *p = make_float2(q.x + v0, q.y + v1);
        break;
    }
    case E_BF16: *reinterpret_cast<unsigned *>(e.h + o) = pack2(v0, v1); break;
    case E_F32_BF16:
        *reinterpret_cast<float2 *>(e.f + o) = make_float2(v0, v1);
        *reinterpret_cast<unsigned *>(e.h + o) = pack2(v0, v1);
        break;
    case E_SWIGLU: {
        *reinterpret_cast<unsigned *>(e.h + o) = pack2(v0, v1);
        float s = v0 * sigm(v0);
        e.h2[(long)m * e.ld2 + (n >> 1)] = __float2bfloat16_rn(s * v1);
        break;
    }
    case E_CONVT: {
        int cout = e.a, tap = n / cout, co = n - tap * cout;
        int Z = e.b, Y = e.c, X = e.d;
        int ix = m % X, t1 = m / X, iy = t1 % Y, t2 = t1 / Y, iz = t2 % Z, bb = t2 / Z;
        int fx = e.fx, fy = e.fy, fz = e.fz;
        int tx = tap % fx, ty = (tap / fx) % fy, tz = tap / (fx * fy);
        long os = (((long)bb * Z * fz + iz * fz + tz) * (Y * fy) + iy * fy + ty) * (long)(X * fx) +
            ix * fx + tx;
        float b0 = e.bias ? e.bias[co] : 0.f, b1 = e.bias ? e.bias[co + 1] : 0.f;
        *reinterpret_cast<unsigned *>(e.h + os * cout + co) = pack2(v0 + b0, v1 + b1);
        break;
    }
    case E_GELU_Z:
        *reinterpret_cast<unsigned *>(e.h + o) = pack2(v0, v1);
        *reinterpret_cast<unsigned *>(e.h2 + o) = pack2(gelu(v0), gelu(v1));
        break;
    case E_GELU_GRAD: {
        __nv_bfloat162 zz = *reinterpret_cast<const __nv_bfloat162 *>(e.z + o);
        float2 zf = __bfloat1622float2(zz);
        *reinterpret_cast<unsigned *>(e.h + o) = pack2(v0 * gelu_d(zf.x), v1 * gelu_d(zf.y));
        break;
    }
    case E_PART:
        *reinterpret_cast<float2 *>(e.f + (long)split * M * N + (long)m * N + n) =
            make_float2(v0, v1);
        break;
    case E_CTX: {
        int g = e.a / 16, ox = m % g, oy = (m / g) % g, bb = m / (g * g);
        const float *bx = e.box + bb * 4;
        bool in = 16.f * ox < bx[3] && 16.f * (ox + 1) > bx[1] && 16.f * oy < bx[2] &&
            16.f * (oy + 1) > bx[0];
        if (in) {
            v0 += e.marker[n];
            v1 += e.marker[n + 1];
        }
        *reinterpret_cast<float2 *>(e.f + o) = make_float2(v0, v1);
        break;
    }
    case E_CTX_DX: {
        int g2 = e.a / 16, g1 = e.a / 4, ox = m % g2, oy = (m / g2) % g2, bb = m / (g2 * g2);
        int tap = n / 128, ci = n % 128, ky = tap / 4, kx = tap % 4;
        long pix = ((long)bb * g1 + 4 * oy + ky) * g1 + 4 * ox + kx;
        __nv_bfloat162 zz = *reinterpret_cast<const __nv_bfloat162 *>(e.z + pix * 128 + ci);
        float2 zf = __bfloat1622float2(zz);
        *reinterpret_cast<unsigned *>(e.h + pix * 128 + ci) =
            pack2(v0 * gelu_d(zf.x), v1 * gelu_d(zf.y));
        break;
    }
    }
}

struct Gemm {
    int M = 0, N = 0, K = 0;
    Src a;
    int akm = 0;
    Src b;
    int bnm = 0;
    int act_a = 0;
    int ksplit = 1;
    Epi e;
};

template <int BN> __device__ void gemm_tile(const Gemm &g, int tile, unsigned char *smem) {
    constexpr int BM = 128, BK = 32;
    constexpr int WGN = BN == 128 ? 4 : 2, MT = BN == 128 ? 4 : 2, NT = 4;
    constexpr int ABYTES = BM * BK * 2, BBYTES = BN * BK * 2;
    const int mt = (g.M + BM - 1) / BM, nt = (g.N + BN - 1) / BN;
    const int split = tile / (mt * nt), rem = tile % (mt * nt);
    const int m0 = (rem / nt) * BM, n0 = (rem % nt) * BN;
    int kc = (g.K + g.ksplit - 1) / g.ksplit;
    kc = (kc + BK - 1) / BK * BK;
    const int kbeg = split * kc, kend = min(g.K, kbeg + kc);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = (warp / WGN) * MT * 16, wn = (warp % WGN) * NT * 8;
    const unsigned sbase = su32(smem);
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[i][j][q] = 0.f;
    const int arb = g.akm ? BM * 2 : 64, brb = g.bnm ? 64 : BN * 2;
    auto load = [&](int stage, int k0) {
        unsigned as = sbase + stage * (ABYTES + BBYTES), bs = as + ABYTES;
        for (int c = threadIdx.x; c < BM * BK / 8; c += Threads) {
            int row, col;
            bool ok;
            const bf16 *p;
            if (g.akm) {
                row = c / (BM / 8);
                col = c % (BM / 8);
                ok = k0 + row < kend && m0 + col * 8 < g.M;
                p = ok ? src_ptr(g.a, k0 + row, m0 + col * 8) : g.a.p;
            } else {
                row = c / 4;
                col = c % 4;
                ok = m0 + row < g.M && k0 + col * 8 < kend;
                p = ok ? src_ptr(g.a, m0 + row, k0 + col * 8) : g.a.p;
            }
            unsigned d = sad(as, row, col, arb);
            if (g.act_a) {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (ok)
                    v = gelu8(*reinterpret_cast<const uint4 *>(p));
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(d), "r"(v.x), "r"(v.y),
                    "r"(v.z), "r"(v.w));
            } else
                cp16(d, p, ok);
        }
        for (int c = threadIdx.x; c < BN * BK / 8; c += Threads) {
            int row, col;
            bool ok;
            const bf16 *p;
            if (g.bnm) {
                row = c / 4;
                col = c % 4;
                ok = n0 + row < g.N && k0 + col * 8 < kend;
                p = ok ? src_ptr(g.b, n0 + row, k0 + col * 8) : g.b.p;
            } else {
                row = c / (BN / 8);
                col = c % (BN / 8);
                ok = k0 + row < kend && n0 + col * 8 < g.N;
                p = ok ? src_ptr(g.b, k0 + row, n0 + col * 8) : g.b.p;
            }
            cp16(sad(bs, row, col, brb), p, ok);
        }
    };
    const int nk = (kend - kbeg + BK - 1) / BK;
    if (nk > 0) {
        load(0, kbeg);
        cp_commit();
    }
    for (int s = 0; s < nk; ++s) {
        if (s + 1 < nk)
            load((s + 1) & 1, kbeg + (s + 1) * BK);
        cp_commit();
        cp_wait1();
        __syncthreads();
        unsigned as = sbase + (s & 1) * (ABYTES + BBYTES), bs = as + ABYTES;
#pragma unroll
        for (int kb = 0; kb < BK; kb += 16) {
            unsigned af[MT][4], bfr[NT / 2][4];
            const int q = lane >> 3, r = lane & 7;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int mb = wm + i * 16;
                if (g.akm)
                    ldsmt(af[i], sad(as, kb + r + ((q >> 1) << 3), (mb >> 3) + (q & 1), arb));
                else
                    ldsm(af[i], sad(as, mb + (lane & 15), (kb >> 3) + (lane >> 4), arb));
            }
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
                int nb = wn + j * 16;
                if (g.bnm)
                    ldsm(bfr[j], sad(bs, nb + r + ((q >> 1) << 3), (kb >> 3) + (q & 1), brb));
                else
                    ldsmt(bfr[j], sad(bs, kb + r + ((q & 1) << 3), (nb >> 3) + (q >> 1), brb));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma(acc[i][j], af[i], bfr[j >> 1][(j & 1) * 2], bfr[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }
    cp_wait0();
    const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int m = m0 + wm + i * 16 + gq, n = n0 + wn + j * 8 + tq * 2;
            epi2(g.e, g.M, g.N, split, m, n, acc[i][j][0], acc[i][j][1]);
            epi2(g.e, g.M, g.N, split, m + 8, n, acc[i][j][2], acc[i][j][3]);
        }
}

__device__ void gemm_run(const Gemm &g, unsigned char *smem) {
    int wide = ((g.M + 127) / 128) * ((g.N + 127) / 128) * g.ksplit;
    if (g.N >= 128 && wide >= 4 * (int)gridDim.x) {
        for (int t = blockIdx.x; t < wide; t += gridDim.x)
            gemm_tile<128>(g, t, smem);
    } else {
        int tiles = ((g.M + 127) / 128) * ((g.N + 63) / 64) * g.ksplit;
        for (int t = blockIdx.x; t < tiles; t += gridDim.x)
            gemm_tile<64>(g, t, smem);
    }
}

__device__ void reduce_parts(float *dst, const float *parts, long count, int splits, int accumulate,
    long stride = 0, long off = 0) {
    if (!stride)
        stride = count;
    for (long i = gtid(); i < count; i += gstride()) {
        float s = 0.f;
        for (int k = 0; k < splits; ++k)
            s += parts[(long)k * stride + off + i];
        dst[i] = accumulate ? dst[i] + s : s;
    }
}

__host__ __device__ __forceinline__ int fdiv(int a, int s) {
    return a >= 0 ? a / s : -((-a + s - 1) / s);
}
__host__ __device__ __forceinline__ int cdiv(int a, int s) { return -fdiv(-a, s); }

struct Conv {
    const bf16 *s0 = nullptr, *s1 = nullptr;
    int c0 = 0, c1 = 0, act0 = 0, act1 = 0;
    const uint8_t *mask = nullptr;
    int msh[3] = {0, 0, 0};
    int TD = 1, TH = 1, TW = 1;
    int B = 1, SZ = 1, SY = 1, SX = 1, OZ = 1, OY = 1, OX = 1;
    int sz = 1, sy = 1, sx = 1;
    int dx = 0;
    int TZ = 4, TY = 8, TX = 8;
    const bf16 *w = nullptr;
    int cin = 0, cout = 0;
    const float *bias = nullptr;
    bf16 *d0 = nullptr;
    float *f0 = nullptr;
    const bf16 *z0 = nullptr;
    int acc0 = 0;
    bf16 *d1 = nullptr;
    const bf16 *z1 = nullptr;
    int acc1 = 0;
    const uint8_t *m1 = nullptr;
    int osh[3] = {0, 0, 0};
    int split = 0;
};

struct Halo {
    int b, oz, oy, ox, ez, ey, ex, rows;
};

__device__ __forceinline__ Halo halo_box(const Conv &c, int b, int z0, int y0, int x0) {
    Halo h;
    h.b = b;
    if (!c.dx) {
        h.oz = z0 * c.sz - 1;
        h.oy = y0 * c.sy - 1;
        h.ox = x0 * c.sx - 1;
        h.ez = (c.TZ - 1) * c.sz + 3;
        h.ey = (c.TY - 1) * c.sy + 3;
        h.ex = (c.TX - 1) * c.sx + 3;
    } else {
        h.oz = cdiv(z0 - 1, c.sz);
        h.oy = cdiv(y0 - 1, c.sy);
        h.ox = cdiv(x0 - 1, c.sx);
        h.ez = fdiv(z0 + c.TZ, c.sz) - h.oz + 1;
        h.ey = fdiv(y0 + c.TY, c.sy) - h.oy + 1;
        h.ex = fdiv(x0 + c.TX, c.sx) - h.ox + 1;
    }
    h.rows = h.ez * h.ey * h.ex;
    return h;
}

__device__ __forceinline__ uint4 halo_item(const Conv &c, const Halo &h, int chunk, int item) {
    int row = item >> 2, part = item & 3;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row >= h.rows)
        return v;
    int hx = row % h.ex, hy = (row / h.ex) % h.ey, hz = row / (h.ex * h.ey);
    int z = h.oz + hz, y = h.oy + hy, x = h.ox + hx;
    if (z < 0 || y < 0 || x < 0 || z >= c.SZ || y >= c.SY || x >= c.SX)
        return v;
    int ch = chunk * 32;
    bool second = c.s1 && ch >= c.c0;
    const bf16 *base = second ? c.s1 : c.s0;
    int cc = second ? c.c1 : c.c0, off = second ? ch - c.c0 : ch;
    if (second && c.mask) {
        int t =
            (((h.b * c.TD + (z >> c.msh[0])) * c.TH + (y >> c.msh[1])) * c.TW + (x >> c.msh[2]));
        if (c.mask[t])
            return v;
    }
    long site = (((long)h.b * c.SZ + z) * c.SY + y) * c.SX + x;
    v = *reinterpret_cast<const uint4 *>(base + site * cc + off + part * 8);
    if (second ? c.act1 : c.act0)
        v = gelu8(v);
    return v;
}

__device__ __forceinline__ void st_shared(unsigned a, uint4 v) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

struct Lane {
    int mz, my, mx;
};

__device__ __forceinline__ Lane lane_site(const Conv &c, int m) {
    int lx = __ffs(c.TX) - 1, ly = __ffs(c.TY) - 1;
    return Lane{m >> (lx + ly), (m >> lx) & (c.TY - 1), m & (c.TX - 1)};
}

__device__ __forceinline__ int conv_row(const Conv &c, const Halo &h, int z0, int y0, int x0,
    Lane L, int kz, int ky, int kx, bool &ok) {
    ok = true;
    if (!c.dx)
        return ((L.mz * c.sz + kz) * h.ey + L.my * c.sy + ky) * h.ex + L.mx * c.sx + kx;
    int vz = z0 + L.mz + 1 - kz, vy = y0 + L.my + 1 - ky, vx = x0 + L.mx + 1 - kx;
    if ((c.sz == 2 && (vz & 1)) || (c.sy == 2 && (vy & 1)) || (c.sx == 2 && (vx & 1))) {
        ok = false;
        return 0;
    }
    int oz = c.sz == 2 ? vz >> 1 : vz, oy = c.sy == 2 ? vy >> 1 : vy, ox = c.sx == 2 ? vx >> 1 : vx;
    return ((oz - h.oz) * h.ey + oy - h.oy) * h.ex + ox - h.ox;
}

__device__ __forceinline__ void conv_store(
    const Conv &c, long site, int N, int n, float v0, float v1, int b, int z, int y, int x) {
    if (c.f0) {
        float b0 = c.bias ? c.bias[n] : 0.f, b1 = c.bias ? c.bias[n + 1] : 0.f;
        *reinterpret_cast<float2 *>(c.f0 + site * N + n) = make_float2(v0 + b0, v1 + b1);
        return;
    }
    if (c.bias) {
        v0 += c.bias[n];
        v1 += c.bias[n + 1];
    }
    bool second = c.d1 && n >= c.split;
    int C = c.d1 ? (second ? N - c.split : c.split) : N;
    int nn = second ? n - c.split : n;
    bf16 *d = second ? c.d1 : c.d0;
    const bf16 *z_ = second ? c.z1 : c.z0;
    long o = site * C + nn;
    if (second && c.m1) {
        int t = (((b * c.TD + (z >> c.osh[0])) * c.TH + (y >> c.osh[1])) * c.TW + (x >> c.osh[2]));
        if (c.m1[t])
            v0 = v1 = 0.f;
    }
    if (second ? c.acc1 : c.acc0) {
        float2 q = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(d + o));
        v0 += q.x;
        v1 += q.y;
    }
    if (z_) {
        float2 q = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(z_ + o));
        v0 *= gelu_d(q.x);
        v1 *= gelu_d(q.y);
    }
    *reinterpret_cast<unsigned *>(d + o) = pack2(v0, v1);
}

template <int WGM, int MT, int NT, int PF>
__device__ void conv_run(const Conv &c, unsigned char *smem) {
    constexpr int WGN = Warps / WGM, NTILE = WGN * NT * 8;
    const int K = c.dx ? c.cout : c.c0 + c.c1, N = c.dx ? c.cin : c.cout;
    const int nchunks = K / 32, ntiles = N / NTILE;
    const int gz = c.OZ / c.TZ, gy = c.OY / c.TY, gx = c.OX / c.TX;
    const int items = c.B * gz * gy * gx * ntiles;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = lane >> 3, r = lane & 7;
    const int wm = (warp / WGN) * MT * 16, wn = (warp % WGN) * NT * 8;
    const unsigned hbase = su32(smem), bbase = hbase + 64 * 1024, zrow = bbase + 2 * 4096;
    if (threadIdx.x < 4)
        st_shared(zrow + threadIdx.x * 16, make_uint4(0, 0, 0, 0));
    const int brb = c.dx ? 64 : NTILE * 2;
    auto decode = [&](int it, int &b, int &z0, int &y0, int &x0, int &n0) {
        n0 = (it % ntiles) * NTILE;
        int t = it / ntiles;
        x0 = (t % gx) * c.TX;
        t /= gx;
        y0 = (t % gy) * c.TY;
        t /= gy;
        z0 = (t % gz) * c.TZ;
        b = t / gz;
    };
    auto loadB = [&](int stage, int chunk, int tap, int n0) {
        unsigned bs = bbase + stage * 4096;
        for (int k = threadIdx.x; k < NTILE * 4; k += Threads) {
            int row, col;
            const bf16 *p;
            if (!c.dx) {
                row = k / (NTILE / 8);
                col = k % (NTILE / 8);
                p = c.w + ((long)(tap * c.cin + chunk * 32 + row) * c.cout + n0 + col * 8);
            } else {
                row = k / 4;
                col = k % 4;
                p = c.w + ((long)(tap * c.cin + n0 + row) * c.cout + chunk * 32 + col * 8);
            }
            cp16(sad(bs, row, col, brb), p, true);
        }
    };
    uint4 pre[PF];
    Lane lanes[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
        lanes[i] = lane_site(c, wm + i * 16 + (lane & 15));
    int it = blockIdx.x;
    if (it >= items)
        return;
    int b, z0, y0, x0, n0;
    decode(it, b, z0, y0, x0, n0);
    Halo h = halo_box(c, b, z0, y0, x0);
    for (int i = 0; i < PF; ++i)
        pre[i] = halo_item(c, h, 0, threadIdx.x + i * Threads);
    int chunk = 0;
    float acc[MT][NT][4];
    while (true) {
        __syncthreads();
        for (int i = 0; i < PF; ++i) {
            int item = threadIdx.x + i * Threads;
            if ((item >> 2) < h.rows)
                st_shared(sad(hbase, item >> 2, item & 3, 64), pre[i]);
        }
        __syncthreads();
        int nit = it, nchunk = chunk + 1, nb_, nz, ny, nx, nn;
        if (nchunk == nchunks) {
            nchunk = 0;
            nit = it + gridDim.x;
        }
        Halo nh = h;
        if (nit < items) {
            decode(nit, nb_, nz, ny, nx, nn);
            nh = halo_box(c, nb_, nz, ny, nx);
            for (int i = 0; i < PF; ++i)
                pre[i] = halo_item(c, nh, nchunk, threadIdx.x + i * Threads);
        }
        if (chunk == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        acc[i][j][k] = 0.f;
        }
        loadB(0, chunk, 0, n0);
        cp_commit();
        for (int tap = 0; tap < 27; ++tap) {
            if (tap + 1 < 27)
                loadB((tap + 1) & 1, chunk, tap + 1, n0);
            cp_commit();
            const int kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
            unsigned rbase[MT];
            int rsw[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                bool ok;
                int row = conv_row(c, h, z0, y0, x0, lanes[i], kz, ky, kx, ok);
                rbase[i] = ok ? hbase + row * 64 : zrow;
                rsw[i] = ok ? swz(row, 64) : 0;
            }
            cp_wait1();
            __syncthreads();
            unsigned bs = bbase + (tap & 1) * 4096;
#pragma unroll
            for (int kb = 0; kb < 32; kb += 16) {
                unsigned af[MT][4], bfr[NT / 2][4];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    int ch = (kb >> 3) + (lane >> 4);
                    ldsm(af[i], rbase[i] + ((ch ^ rsw[i]) << 4));
                }
#pragma unroll
                for (int j = 0; j < NT / 2; ++j) {
                    int nb = wn + j * 16;
                    if (!c.dx)
                        ldsmt(bfr[j], sad(bs, kb + r + ((q & 1) << 3), (nb >> 3) + (q >> 1), brb));
                    else
                        ldsm(bfr[j], sad(bs, nb + r + ((q >> 1) << 3), (kb >> 3) + (q & 1), brb));
                }
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma(acc[i][j], af[i], bfr[j >> 1][(j & 1) * 2],
                            bfr[j >> 1][(j & 1) * 2 + 1]);
            }
            __syncthreads();
        }
        cp_wait0();
        if (chunk + 1 == nchunks) {
            const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int m = wm + i * 16 + gq + half * 8;
                    int mx = m % c.TX, my = (m / c.TX) % c.TY, mz = m / (c.TX * c.TY);
                    int z = z0 + mz, y = y0 + my, x = x0 + mx;
                    long site = (((long)b * c.OZ + z) * c.OY + y) * c.OX + x;
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        conv_store(c, site, N, n0 + wn + j * 8 + tq * 2, acc[i][j][half * 2],
                            acc[i][j][half * 2 + 1], b, z, y, x);
                }
        }
        if (nit >= items)
            break;
        it = nit;
        chunk = nchunk;
        if (chunk == 0) {
            b = nb_;
            z0 = nz;
            y0 = ny;
            x0 = nx;
            n0 = nn;
        }
        h = nh;
    }
}

struct ConvW {
    const bf16 *s0 = nullptr, *s1 = nullptr;
    int c0 = 0, c1 = 0, act0 = 0, act1 = 0;
    const uint8_t *mask = nullptr;
    int msh[3] = {0, 0, 0};
    int TD = 1, TH = 1, TW = 1;
    int B = 1, SZ = 1, SY = 1, SX = 1, OZ = 1, OY = 1, OX = 1;
    int sz = 1, sy = 1, sx = 1;
    int TZ = 4, TY = 8, TX = 8;
    const bf16 *dy = nullptr;
    int cout = 0;
    float *part = nullptr, *bpart = nullptr;
    int splits = 1;
};

template <int PF> __device__ void convw_run(const ConvW &w, unsigned char *smem) {
    Conv c;
    c.s0 = w.s0;
    c.s1 = w.s1;
    c.c0 = w.c0;
    c.c1 = w.c1;
    c.act0 = w.act0;
    c.act1 = w.act1;
    c.mask = w.mask;
    for (int i = 0; i < 3; ++i)
        c.msh[i] = w.msh[i];
    c.TD = w.TD;
    c.TH = w.TH;
    c.TW = w.TW;
    c.B = w.B;
    c.SZ = w.SZ;
    c.SY = w.SY;
    c.SX = w.SX;
    c.OZ = w.OZ;
    c.OY = w.OY;
    c.OX = w.OX;
    c.sz = w.sz;
    c.sy = w.sy;
    c.sx = w.sx;
    c.TZ = w.TZ;
    c.TY = w.TY;
    c.TX = w.TX;
    const int cin = w.c0 + w.c1, nci = cin / 32, nco = w.cout / 32;
    const int items = nci * nco * 2, units = items * w.splits;
    const int gz = w.OZ / w.TZ, gy = w.OY / w.TY, gx = w.OX / w.TX, tiles = w.B * gz * gy * gx;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = lane >> 3, r = lane & 7;
    const unsigned hbase = su32(smem), dbase = hbase + 64 * 1024;
    const int M = w.TZ * w.TY * w.TX;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        const int item = u / w.splits, s = u % w.splits;
        const int tg = item % 2, coj = (item / 2) % nco, cj = item / (2 * nco);
        const int tap0 = tg * 16 + warp, tap1 = tg * 16 + warp + 8;
        const bool v0 = tap0 < 27 && (tg == 1 || tap0 < 16),
                   v1 = tap1 < 27 && (tg == 1 || tap1 < 16);
        float acc[2][2][4][4];
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        acc[a][i][j][k] = 0.f;
        float bsum = 0.f;
        const bool dobias = cj == 0 && tg == 0 && w.bpart;
        for (int t = s; t < tiles; t += w.splits) {
            int x0 = (t % gx) * w.TX, y0 = ((t / gx) % gy) * w.TY,
                z0 = ((t / (gx * gy)) % gz) * w.TZ, b = t / (gx * gy * gz);
            Halo h = halo_box(c, b, z0, y0, x0);
            __syncthreads();
            for (int i = 0; i < PF; ++i) {
                int it = threadIdx.x + i * Threads;
                if ((it >> 2) < h.rows)
                    st_shared(sad(hbase, it >> 2, it & 3, 64), halo_item(c, h, cj, it));
            }
            for (int it = threadIdx.x; it < M * 4; it += Threads) {
                int m = it >> 2, part = it & 3;
                int mx = m % w.TX, my = (m / w.TX) % w.TY, mz = m / (w.TX * w.TY);
                long site = (((long)b * w.OZ + z0 + mz) * w.OY + y0 + my) * w.OX + x0 + mx;
                st_shared(sad(dbase, m, part, 64),
                    *reinterpret_cast<const uint4 *>(w.dy + site * w.cout + coj * 32 + part * 8));
            }
            __syncthreads();
            if (dobias && warp == 0) {
                const bf16 *ds = reinterpret_cast<const bf16 *>(smem + 64 * 1024);
                for (int m = 0; m < M; ++m) {
                    int chunk = lane >> 3, within = lane & 7;
                    bsum += b2f(ds[m * 32 + ((chunk ^ swz(m, 64)) << 3) + within]);
                }
            }
            for (int kb = 0; kb < M; kb += 16) {
                unsigned bfr[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    ldsmt(bfr[j], sad(dbase, kb + r + ((q & 1) << 3), j * 2 + (q >> 1), 64));
                Lane L = lane_site(c, kb + r + ((q >> 1) << 3));
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    if (!(a ? v1 : v0))
                        continue;
                    bool ok;
                    int tp = a ? tap1 : tap0;
                    int row = conv_row(c, h, z0, y0, x0, L, tp / 9, (tp / 3) % 3, tp % 3, ok);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        unsigned af[4];
                        ldsmt(af, sad(hbase, row, i * 2 + (q & 1), 64));
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            mma(acc[a][i][j], af, bfr[j >> 1][(j & 1) * 2],
                                bfr[j >> 1][(j & 1) * 2 + 1]);
                    }
                }
            }
        }
        const int gq = lane >> 2, tq = lane & 3;
        const long wsize = 27L * cin * w.cout;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            if (!(a ? v1 : v0))
                continue;
            int tap = a ? tap1 : tap0;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        int ci = cj * 32 + i * 16 + gq + half * 8, co = coj * 32 + j * 8 + tq * 2;
                        float *p = w.part + s * wsize + ((long)tap * cin + ci) * w.cout + co;
                        p[0] = acc[a][i][j][half * 2];
                        p[1] = acc[a][i][j][half * 2 + 1];
                    }
        }
        if (dobias && warp == 0)
            w.bpart[(long)s * w.cout + coj * 32 + lane] = bsum;
    }
}

__device__ void conv1_fwd(const bf16 *ct, int B, int Z, int Y, int X, const float *wt,
    const float *bias, bf16 *z, unsigned char *smem) {
    float *hal = reinterpret_cast<float *>(smem);
    float *ws = hal + 640;
    const int gz = Z / 4, gy = Y / 8, gx = X / 8, tiles = B * gz * gy * gx;
    for (int i = threadIdx.x; i < 27 * 32 + 32; i += Threads)
        ws[i] = i < 864 ? wt[i] : bias[i - 864];
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int x0 = (t % gx) * 8, y0 = ((t / gx) % gy) * 8, z0 = ((t / (gx * gy)) % gz) * 4,
            b = t / (gx * gy * gz);
        __syncthreads();
        for (int i = threadIdx.x; i < 600; i += Threads) {
            int hx = i % 10, hy = (i / 10) % 10, hz = i / 100;
            int zz = z0 + hz - 1, yy = y0 + hy - 1, xx = x0 + hx - 1;
            bool ok = zz >= 0 && yy >= 0 && xx >= 0 && zz < Z && yy < Y && xx < X;
            hal[i] = ok ? b2f(ct[(((long)b * Z + zz) * Y + yy) * X + xx]) : 0.f;
        }
        __syncthreads();
        int m = threadIdx.x, mx = m % 8, my = (m / 8) % 8, mz = m / 64;
        float o[32];
        for (int k = 0; k < 32; ++k)
            o[k] = ws[864 + k];
        for (int tap = 0; tap < 27; ++tap) {
            int kx = tap % 3, ky = (tap / 3) % 3, kz = tap / 9;
            float v = hal[((mz + kz) * 10 + my + ky) * 10 + mx + kx];
#pragma unroll
            for (int k = 0; k < 32; ++k)
                o[k] += v * ws[tap * 32 + k];
        }
        long site = (((long)b * Z + z0 + mz) * Y + y0 + my) * X + x0 + mx;
        uint4 *dst = reinterpret_cast<uint4 *>(z + site * 32);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            dst[k] = make_uint4(pack2(o[k * 8], o[k * 8 + 1]), pack2(o[k * 8 + 2], o[k * 8 + 3]),
                pack2(o[k * 8 + 4], o[k * 8 + 5]), pack2(o[k * 8 + 6], o[k * 8 + 7]));
    }
}

__device__ void conv1_dw(
    const bf16 *ct, const bf16 *dz, int B, int Z, int Y, int X, float *part, unsigned char *smem) {
    float *hal = reinterpret_cast<float *>(smem);
    float *ds = hal + 640;
    const int gz = Z / 4, gy = Y / 8, gx = X / 8, tiles = B * gz * gy * gx;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int x0 = (t % gx) * 8, y0 = ((t / gx) % gy) * 8, z0 = ((t / (gx * gy)) % gz) * 4,
            b = t / (gx * gy * gz);
        __syncthreads();
        for (int i = threadIdx.x; i < 600; i += Threads) {
            int hx = i % 10, hy = (i / 10) % 10, hz = i / 100;
            int zz = z0 + hz - 1, yy = y0 + hy - 1, xx = x0 + hx - 1;
            bool ok = zz >= 0 && yy >= 0 && xx >= 0 && zz < Z && yy < Y && xx < X;
            hal[i] = ok ? b2f(ct[(((long)b * Z + zz) * Y + yy) * X + xx]) : 0.f;
        }
        for (int i = threadIdx.x; i < 256 * 32; i += Threads) {
            int m = i / 32, k = i % 32, mx = m % 8, my = (m / 8) % 8, mz = m / 64;
            long site = (((long)b * Z + z0 + mz) * Y + y0 + my) * X + x0 + mx;
            ds[i] = b2f(dz[site * 32 + k]);
        }
        __syncthreads();
        for (int a = 0; a < 4; ++a) {
            int p = threadIdx.x + a * Threads;
            if (p >= 896)
                break;
            float s = 0.f;
            if (p < 864) {
                int tap = p / 32, k = p % 32, kx = tap % 3, ky = (tap / 3) % 3, kz = tap / 9;
                for (int m = 0; m < 256; ++m) {
                    int mx = m % 8, my = (m / 8) % 8, mz = m / 64;
                    s += hal[((mz + kz) * 10 + my + ky) * 10 + mx + kx] * ds[m * 32 + k];
                }
            } else
                for (int m = 0; m < 256; ++m)
                    s += ds[m * 32 + p - 864];
            acc[a] += s;
        }
    }
    for (int a = 0; a < 4; ++a) {
        int p = threadIdx.x + a * Threads;
        if (p < 896)
            part[(long)blockIdx.x * 896 + p] = acc[a];
    }
}

__device__ void ln_fwd(
    const float *x, int rows, const float *g, const float *bt, bf16 *yh, float *yf, float *stat) {
    const int lane = threadIdx.x & 31, wg = (blockIdx.x * Threads + threadIdx.x) >> 5,
              nw = gstride() >> 5;
    for (int r = wg; r < rows; r += nw) {
        const float4 *p = reinterpret_cast<const float4 *>(x + (long)r * 256 + lane * 8);
        float4 a = p[0], b = p[1];
        float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float s = 0.f;
        for (int i = 0; i < 8; ++i)
            s += v[i];
        for (int o = 16; o; o >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, o);
        float mu = s / 256.f, q = 0.f;
        for (int i = 0; i < 8; ++i)
            q += (v[i] - mu) * (v[i] - mu);
        for (int o = 16; o; o >>= 1)
            q += __shfl_xor_sync(0xffffffffu, q, o);
        float rs = rsqrtf(q / 256.f + 1e-5f);
        float y[8];
        for (int i = 0; i < 8; ++i)
            y[i] = (v[i] - mu) * rs * g[lane * 8 + i] + bt[lane * 8 + i];
        if (yh)
            *reinterpret_cast<uint4 *>(yh + (long)r * 256 + lane * 8) = make_uint4(
                pack2(y[0], y[1]), pack2(y[2], y[3]), pack2(y[4], y[5]), pack2(y[6], y[7]));
        if (yf) {
            float4 *o = reinterpret_cast<float4 *>(yf + (long)r * 256 + lane * 8);
            o[0] = make_float4(y[0], y[1], y[2], y[3]);
            o[1] = make_float4(y[4], y[5], y[6], y[7]);
        }
        if (lane == 0) {
            stat[2 * r] = mu;
            stat[2 * r + 1] = rs;
        }
    }
}

__device__ void ln_bwd(const float *x, const float *stat, const float *dy, int rows, const float *g,
    float *dx, int dx_acc, float *part, unsigned char *smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float pg[8] = {0, 0, 0, 0, 0, 0, 0, 0}, pb[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int r = blockIdx.x * Warps + warp; r < rows; r += gridDim.x * Warps) {
        const float4 *px = reinterpret_cast<const float4 *>(x + (long)r * 256 + lane * 8);
        const float4 *pd = reinterpret_cast<const float4 *>(dy + (long)r * 256 + lane * 8);
        float4 a = px[0], b = px[1], c = pd[0], d = pd[1];
        float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w},
              gv[8] = {c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
        float mu = stat[2 * r], rs = stat[2 * r + 1], s1 = 0.f, s2 = 0.f, xh[8], gg[8];
        for (int i = 0; i < 8; ++i) {
            xh[i] = (xv[i] - mu) * rs;
            gg[i] = gv[i] * g[lane * 8 + i];
            s1 += gg[i];
            s2 += gg[i] * xh[i];
            pg[i] += gv[i] * xh[i];
            pb[i] += gv[i];
        }
        for (int o = 16; o; o >>= 1) {
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            s2 += __shfl_xor_sync(0xffffffffu, s2, o);
        }
        s1 /= 256.f;
        s2 /= 256.f;
        float4 *po = reinterpret_cast<float4 *>(dx + (long)r * 256 + lane * 8);
        float o[8];
        for (int i = 0; i < 8; ++i)
            o[i] = rs * (gg[i] - s1 - xh[i] * s2);
        if (dx_acc) {
            float4 e = po[0], f = po[1];
            o[0] += e.x;
            o[1] += e.y;
            o[2] += e.z;
            o[3] += e.w;
            o[4] += f.x;
            o[5] += f.y;
            o[6] += f.z;
            o[7] += f.w;
        }
        po[0] = make_float4(o[0], o[1], o[2], o[3]);
        po[1] = make_float4(o[4], o[5], o[6], o[7]);
    }
    float *red = reinterpret_cast<float *>(smem);
    __syncthreads();
    for (int i = 0; i < 8; ++i) {
        red[warp * 512 + lane * 8 + i] = pg[i];
        red[warp * 512 + 256 + lane * 8 + i] = pb[i];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < 512; k += Threads) {
        float s = 0.f;
        for (int w = 0; w < Warps; ++w)
            s += red[w * 512 + k];
        part[(long)blockIdx.x * 512 + k] = s;
    }
    __syncthreads();
}

struct Attn {
    int kind = 0;
    const bf16 *q = nullptr, *k = nullptr, *v = nullptr;
    long ldq = 0, ldk = 0;
    int B = 1, nq = 0, nk = 0;
    int TD = 1, TH = 1, TW = 1, shift = 0;
    bf16 *o = nullptr;
    long ldo = 256;
    float *lse = nullptr;
    float *po = nullptr, *pm = nullptr, *pl = nullptr;
    const bf16 *dO = nullptr;
    const float *dsum = nullptr;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr;
    long lddq = 256, lddk = 256;
    float *pdq = nullptr, *pdk = nullptr, *pdv = nullptr;
};

struct AItem {
    int b, qt, hp, kc, nq, nk;
    int qrow[64];
};

__device__ __forceinline__ int attn_items(const Attn &a) {
    if (a.kind == 0) {
        int wz = (a.TD + a.shift + 3) / 4, wy = (a.TH + a.shift + 3) / 4,
            wx = (a.TW + a.shift + 3) / 4;
        return a.B * wz * wy * wx * 4;
    }
    return a.B * ((a.nq + 63) / 64) * 4 * ((a.nk + 127) / 128);
}

__device__ __forceinline__ int attn_chunks(const Attn &a) {
    return a.kind == 0 ? 1 : (a.nk + 127) / 128;
}

__device__ void attn_decode(const Attn &a, int it, AItem &w) {
    if (a.kind == 0) {
        int wz = (a.TD + a.shift + 3) / 4, wy = (a.TH + a.shift + 3) / 4,
            wx = (a.TW + a.shift + 3) / 4;
        w.hp = it % 4;
        int t = it / 4;
        int ix = t % wx, iy = (t / wx) % wy, iz = (t / (wx * wy)) % wz;
        w.b = t / (wx * wy * wz);
        w.qt = t % (wx * wy * wz);
        w.kc = 0;
        int z0 = max(0, iz * 4 - a.shift), z1 = min(a.TD, iz * 4 - a.shift + 4);
        int y0 = max(0, iy * 4 - a.shift), y1 = min(a.TH, iy * 4 - a.shift + 4);
        int x0 = max(0, ix * 4 - a.shift), x1 = min(a.TW, ix * 4 - a.shift + 4);
        int n = 0, T = a.TD * a.TH * a.TW;
        for (int z = z0; z < z1; ++z)
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x)
                    w.qrow[n++] = w.b * T + (z * a.TH + y) * a.TW + x;
        for (int i = n; i < 64; ++i)
            w.qrow[i] = w.qrow[n - 1];
        w.nq = w.nk = n;
        return;
    }
    int nqt = (a.nq + 63) / 64, nkc = (a.nk + 127) / 128;
    w.kc = it % nkc;
    int t = it / nkc;
    w.hp = t % 4;
    t /= 4;
    w.qt = t % nqt;
    w.b = t / nqt;
    w.nq = min(64, a.nq - w.qt * 64);
    w.nk = min(128, a.nk - w.kc * 128);
    for (int i = 0; i < 64; ++i)
        w.qrow[i] = w.b * a.nq + w.qt * 64 + min(i, w.nq - 1);
}

__device__ __forceinline__ int attn_krow(const Attn &a, const AItem &w, int j) {
    if (a.kind == 0)
        return w.qrow[min(j, w.nk - 1)];
    return w.b * a.nk + w.kc * 128 + min(j, w.nk - 1);
}

__device__ void attn_load(
    const Attn &a, const AItem &w, unsigned qs, unsigned ks, unsigned vs, unsigned dos, bool bwd) {
    for (int c = threadIdx.x; c < 2 * 64 * 4; c += Threads) {
        int h = c / 256, row = (c / 4) % 64, part = c % 4, head = w.hp * 2 + h;
        int gr = w.qrow[row];
        cp16(
            sad(qs + h * 4096, row, part, 64), a.q + (long)gr * a.ldq + head * 32 + part * 8, true);
        if (bwd)
            cp16(sad(dos + h * 4096, row, part, 64), a.dO + (long)gr * a.ldo + head * 32 + part * 8,
                true);
    }
    for (int c = threadIdx.x; c < 2 * 128 * 4; c += Threads) {
        int h = c / 512, row = (c / 4) % 128, part = c % 4, head = w.hp * 2 + h;
        long kr = attn_krow(a, w, row);
        cp16(sad(ks + h * 8192, row, part, 64), a.k + kr * a.ldk + head * 32 + part * 8, true);
        cp16(sad(vs + h * 8192, row, part, 64), a.v + kr * a.ldk + head * 32 + part * 8, true);
    }
    cp_commit();
    cp_wait0();
    __syncthreads();
}

__device__ __forceinline__ void attn_scores(
    unsigned qs, unsigned ks, int rq0, int nk, float S[16][4]) {
    const int lane = threadIdx.x & 31, q = lane >> 3, r = lane & 7, tq = lane & 3;
    unsigned af[2][4];
    for (int kk = 0; kk < 2; ++kk)
        ldsm(af[kk], sad(qs, rq0 + (lane & 15), kk * 2 + (lane >> 4), 64));
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            S[j][e] = 0.f;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        if (jj * 16 >= nk)
            break;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned bf[4];
            ldsm(bf, sad(ks, jj * 16 + r + ((q >> 1) << 3), kk * 2 + (q & 1), 64));
            mma(S[2 * jj], af[kk], bf[0], bf[1]);
            mma(S[2 * jj + 1], af[kk], bf[2], bf[3]);
        }
    }
    const float sc = 0.17677669529f;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int key = j * 8 + tq * 2 + (e & 1);
            S[j][e] = key < nk ? S[j][e] * sc : -INFINITY;
        }
}

__device__ void attn_fwd(const Attn &a, unsigned char *smem) {
    const unsigned qs = su32(smem), ks = qs + 8192, vs = ks + 16384;
    AItem &w = *reinterpret_cast<AItem *>(smem + 48 * 1024);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = lane >> 3, r = lane & 7;
    const int gq = lane >> 2, tq = lane & 3, hh = warp >> 2, rq0 = (warp & 3) * 16;
    const int items = attn_items(a), nch = attn_chunks(a);
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        __syncthreads();
        if (threadIdx.x == 0)
            attn_decode(a, it, w);
        __syncthreads();
        attn_load(a, w, qs, ks, vs, 0, false);
        float S[16][4];
        attn_scores(qs + hh * 4096, ks + hh * 8192, rq0, w.nk, S);
        float m0 = -INFINITY, m1 = -INFINITY;
        for (int j = 0; j < 16; ++j) {
            m0 = fmaxf(m0, fmaxf(S[j][0], S[j][1]));
            m1 = fmaxf(m1, fmaxf(S[j][2], S[j][3]));
        }
        for (int o = 1; o < 4; o <<= 1) {
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
        }
        float l0 = 0.f, l1 = 0.f;
        for (int j = 0; j < 16; ++j) {
            S[j][0] = expf(S[j][0] - m0);
            S[j][1] = expf(S[j][1] - m0);
            S[j][2] = expf(S[j][2] - m1);
            S[j][3] = expf(S[j][3] - m1);
            l0 += S[j][0] + S[j][1];
            l1 += S[j][2] + S[j][3];
        }
        for (int o = 1; o < 4; o <<= 1) {
            l0 += __shfl_xor_sync(0xffffffffu, l0, o);
            l1 += __shfl_xor_sync(0xffffffffu, l1, o);
        }
        float O[4][4];
        for (int j = 0; j < 4; ++j)
            for (int e = 0; e < 4; ++e)
                O[j][e] = 0.f;
        for (int kk = 0; kk < 8; ++kk) {
            if (kk * 16 >= w.nk)
                break;
            unsigned pa[4] = {pack2(S[2 * kk][0], S[2 * kk][1]), pack2(S[2 * kk][2], S[2 * kk][3]),
                pack2(S[2 * kk + 1][0], S[2 * kk + 1][1]),
                pack2(S[2 * kk + 1][2], S[2 * kk + 1][3])};
            for (int nb = 0; nb < 2; ++nb) {
                unsigned bf[4];
                ldsmt(bf, sad(vs + hh * 8192, kk * 16 + r + ((q & 1) << 3), nb * 2 + (q >> 1), 64));
                mma(O[nb * 2], pa, bf[0], bf[1]);
                mma(O[nb * 2 + 1], pa, bf[2], bf[3]);
            }
        }
        const int head = w.hp * 2 + hh;
        for (int half = 0; half < 2; ++half) {
            int row = rq0 + gq + half * 8;
            if (row >= w.nq)
                continue;
            int gr = w.qrow[row];
            float mm = half ? m1 : m0, ll = half ? l1 : l0;
            if (nch == 1) {
                float inv = 1.f / ll;
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<unsigned *>(a.o + (long)gr * a.ldo + head * 32 + j * 8 +
                        tq * 2) = pack2(O[j][half * 2] * inv, O[j][half * 2 + 1] * inv);
                if (tq == 0)
                    a.lse[(long)gr * 8 + head] = mm + logf(ll);
            } else {
                long base = ((long)w.kc * a.B * a.nq + gr) * 8 + head;
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<float2 *>(a.po + base * 32 + j * 8 + tq * 2) =
                        make_float2(O[j][half * 2], O[j][half * 2 + 1]);
                if (tq == 0) {
                    a.pm[base] = mm;
                    a.pl[base] = ll;
                }
            }
        }
    }
}

__device__ void attn_merge(const Attn &a) {
    const int nch = attn_chunks(a);
    if (nch == 1)
        return;
    const long rows = (long)a.B * a.nq;
    for (long i = gtid(); i < rows * 8; i += gstride()) {
        float M = -INFINITY;
        for (int c = 0; c < nch; ++c)
            M = fmaxf(M, a.pm[c * rows * 8 + i]);
        float L = 0.f, o[32];
        for (int d = 0; d < 32; ++d)
            o[d] = 0.f;
        for (int c = 0; c < nch; ++c) {
            float s = expf(a.pm[c * rows * 8 + i] - M);
            L += a.pl[c * rows * 8 + i] * s;
            const float *p = a.po + (c * rows * 8 + i) * 32;
            for (int d = 0; d < 32; ++d)
                o[d] += p[d] * s;
        }
        long row = i / 8, head = i % 8;
        for (int d = 0; d < 32; d += 2)
            *reinterpret_cast<unsigned *>(a.o + row * a.ldo + head * 32 + d) =
                pack2(o[d] / L, o[d + 1] / L);
        a.lse[i] = M + logf(L);
    }
}

__device__ void attn_dsum(const Attn &a, const bf16 *o, int rows) {
    for (long i = gtid(); i < (long)rows * 8; i += gstride()) {
        long row = i / 8, head = i % 8;
        float s = 0.f;
        for (int d = 0; d < 32; ++d)
            s += b2f(a.dO[row * a.ldo + head * 32 + d]) * b2f(o[row * a.ldo + head * 32 + d]);
        const_cast<float *>(a.dsum)[i] = s;
    }
}

__device__ void attn_bwd(const Attn &a, unsigned char *smem) {
    const unsigned qs = su32(smem), ks = qs + 8192, vs = ks + 16384, dos = vs + 16384,
                   ps = dos + 8192;
    AItem &w = *reinterpret_cast<AItem *>(smem + 90 * 1024);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = lane >> 3, r = lane & 7;
    const int gq = lane >> 2, tq = lane & 3, hh = warp >> 2, rq0 = (warp & 3) * 16;
    const int items = attn_items(a), nch = attn_chunks(a), nqt = a.kind == 0 ? 1 : (a.nq + 63) / 64;
    const float sc = 0.17677669529f;
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        __syncthreads();
        if (threadIdx.x == 0)
            attn_decode(a, it, w);
        __syncthreads();
        attn_load(a, w, qs, ks, vs, dos, true);
        const int head = w.hp * 2 + hh;
        float S[16][4];
        attn_scores(qs + hh * 4096, ks + hh * 8192, rq0, w.nk, S);
        float L[2], Dd[2];
        for (int half = 0; half < 2; ++half) {
            int row = rq0 + gq + half * 8;
            int gr = w.qrow[min(row, w.nq - 1)];
            L[half] = a.lse[(long)gr * 8 + head];
            Dd[half] = a.dsum[(long)gr * 8 + head];
        }
        for (int j = 0; j < 16; ++j)
            for (int e = 0; e < 4; ++e) {
                int row = rq0 + gq + (e >> 1) * 8;
                S[j][e] = row < w.nq ? expf(S[j][e] - L[e >> 1]) : 0.f;
            }
        float dP[16][4];
        for (int j = 0; j < 16; ++j)
            for (int e = 0; e < 4; ++e)
                dP[j][e] = 0.f;
        unsigned af[2][4];
        for (int kk = 0; kk < 2; ++kk)
            ldsm(af[kk], sad(dos + hh * 4096, rq0 + (lane & 15), kk * 2 + (lane >> 4), 64));
        for (int jj = 0; jj < 8; ++jj) {
            if (jj * 16 >= w.nk)
                break;
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bf[4];
                ldsm(bf, sad(vs + hh * 8192, jj * 16 + r + ((q >> 1) << 3), kk * 2 + (q & 1), 64));
                mma(dP[2 * jj], af[kk], bf[0], bf[1]);
                mma(dP[2 * jj + 1], af[kk], bf[2], bf[3]);
            }
        }
        for (int j = 0; j < 16; ++j)
            for (int e = 0; e < 4; ++e)
                dP[j][e] = S[j][e] * (dP[j][e] - Dd[e >> 1]) * sc;
        float dQ[4][4];
        for (int j = 0; j < 4; ++j)
            for (int e = 0; e < 4; ++e)
                dQ[j][e] = 0.f;
        for (int kk = 0; kk < 8; ++kk) {
            if (kk * 16 >= w.nk)
                break;
            unsigned pa[4] = {pack2(dP[2 * kk][0], dP[2 * kk][1]),
                pack2(dP[2 * kk][2], dP[2 * kk][3]), pack2(dP[2 * kk + 1][0], dP[2 * kk + 1][1]),
                pack2(dP[2 * kk + 1][2], dP[2 * kk + 1][3])};
            for (int nb = 0; nb < 2; ++nb) {
                unsigned bf[4];
                ldsmt(bf, sad(ks + hh * 8192, kk * 16 + r + ((q & 1) << 3), nb * 2 + (q >> 1), 64));
                mma(dQ[nb * 2], pa, bf[0], bf[1]);
                mma(dQ[nb * 2 + 1], pa, bf[2], bf[3]);
            }
        }
        for (int half = 0; half < 2; ++half) {
            int row = rq0 + gq + half * 8;
            if (row >= w.nq)
                continue;
            long gr = w.qrow[row];
            float *dst = a.kind == 0 || nch == 1
                ? a.dq + gr * a.lddq + head * 32
                : a.pdq + ((long)w.kc * a.B * a.nq + gr) * 256 + head * 32;
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<float2 *>(dst + j * 8 + tq * 2) =
                    make_float2(dQ[j][half * 2], dQ[j][half * 2 + 1]);
        }
        for (int pass = 0; pass < 2; ++pass) {
            __syncthreads();
            for (int j = 0; j < 16; ++j)
                for (int half = 0; half < 2; ++half) {
                    int row = rq0 + gq + half * 8, key = j * 8 + tq * 2;
                    float v0 = pass ? dP[j][half * 2] : S[j][half * 2],
                          v1 = pass ? dP[j][half * 2 + 1] : S[j][half * 2 + 1];
                    unsigned addr = sad(ps + hh * 16384, row, key >> 3, 256) + (key & 7) * 2;
                    asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(addr), "r"(pack2(v0, v1)));
                }
            __syncthreads();
            const unsigned bsrc = pass ? qs + hh * 4096 : dos + hh * 4096;
            const int kw = warp & 3;
            float G[2][4][4];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int e = 0; e < 4; ++e)
                        G[i][j][e] = 0.f;
            for (int kb = 0; kb < 64; kb += 16) {
                unsigned bf[2][4];
                for (int nb = 0; nb < 2; ++nb)
                    ldsmt(bf[nb], sad(bsrc, kb + r + ((q & 1) << 3), nb * 2 + (q >> 1), 64));
                for (int i = 0; i < 2; ++i) {
                    int mb = kw * 32 + i * 16;
                    unsigned pa[4];
                    ldsmt(pa,
                        sad(ps + hh * 16384, kb + r + ((q >> 1) << 3), (mb >> 3) + (q & 1), 256));
                    for (int j = 0; j < 4; ++j)
                        mma(G[i][j], pa, bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
                }
            }
            for (int i = 0; i < 2; ++i)
                for (int half = 0; half < 2; ++half) {
                    int key = kw * 32 + i * 16 + gq + half * 8;
                    if (key >= w.nk)
                        continue;
                    long kr = attn_krow(a, w, key);
                    float *base = pass ? a.dk : a.dv;
                    float *dst = a.kind == 0 || nqt == 1
                        ? base + kr * a.lddk + head * 32
                        : (pass ? a.pdk : a.pdv) + ((long)w.qt * a.B * a.nk + kr) * 256 + head * 32;
                    for (int j = 0; j < 4; ++j)
                        *reinterpret_cast<float2 *>(dst + j * 8 + tq * 2) =
                            make_float2(G[i][j][half * 2], G[i][j][half * 2 + 1]);
                }
        }
    }
}

__device__ void attn_bwd_reduce(const Attn &a) {
    const int nch = attn_chunks(a), nqt = a.kind == 0 ? 1 : (a.nq + 63) / 64;
    if (a.kind == 0)
        return;
    if (nch > 1) {
        long n = (long)a.B * a.nq * 256;
        for (long i = gtid(); i < n; i += gstride()) {
            float s = 0.f;
            for (int c = 0; c < nch; ++c)
                s += a.pdq[c * n + i];
            a.dq[(i / 256) * a.lddq + i % 256] = s;
        }
    }
    if (nqt > 1) {
        long n = (long)a.B * a.nk * 256;
        for (long i = gtid(); i < n; i += gstride()) {
            float s = 0.f, t = 0.f;
            for (int c = 0; c < nqt; ++c) {
                s += a.pdk[c * n + i];
                t += a.pdv[c * n + i];
            }
            long row = i / 256, col = i % 256;
            a.dk[row * a.lddk + col] = s;
            a.dv[row * a.lddk + col] = t;
        }
    }
}

struct Recur {
    const bf16 *x = nullptr, *gates = nullptr;
    const float *lam = nullptr;
    float *h = nullptr;
    bf16 *y = nullptr;
    int B = 1, TD = 1, TH = 1, TW = 1;
    const float *dy = nullptr;
    float *dx = nullptr;
    bf16 *dg = nullptr;
    float *dlam = nullptr;
};

__device__ void recur_fwd(const Recur &R) {
    const int cols = R.B * R.TH * R.TW, plane = R.TH * R.TW;
    for (long i = gtid(); i < (long)cols * 256; i += gstride()) {
        int ch = i % 256, col = i / 256, b = col / plane, yx = col % plane;
        float acc[16];
        for (int t = 0; t < R.TD; ++t)
            acc[t] = 0.f;
        for (int dir = 0; dir < 2; ++dir) {
            float ls = logf(sigm(R.lam[dir * 256 + ch])), hp = 0.f;
            for (int s = 0; s < R.TD; ++s) {
                int t = dir ? R.TD - 1 - s : s;
                long row = (long)b * R.TD * plane + t * plane + yx;
                float rr = sigm(b2f(R.gates[row * 1024 + dir * 512 + ch]));
                float ii = sigm(b2f(R.gates[row * 1024 + dir * 512 + 256 + ch]));
                float a = expf(8.f * rr * ls), bb = sqrtf(fmaxf(1.f - a * a, 1e-12f));
                hp = a * hp + bb * ii * b2f(R.x[row * 256 + ch]);
                R.h[(dir * (long)R.B * R.TD * plane + row) * 256 + ch] = hp;
                acc[t] += hp;
            }
        }
        for (int t = 0; t < R.TD; ++t) {
            long row = (long)b * R.TD * plane + t * plane + yx;
            R.y[row * 256 + ch] = __float2bfloat16_rn(acc[t]);
        }
    }
}

__device__ void recur_bwd(const Recur &R) {
    const int cols = R.B * R.TH * R.TW, plane = R.TH * R.TW;
    for (long i = gtid(); i < (long)cols * 256; i += gstride()) {
        int ch = i % 256, col = i / 256, b = col / plane, yx = col % plane;
        float dxs[16];
        for (int t = 0; t < R.TD; ++t)
            dxs[t] = 0.f;
        float dl[2] = {0.f, 0.f};
        for (int dir = 0; dir < 2; ++dir) {
            float sl = sigm(R.lam[dir * 256 + ch]), ls = logf(sl), carry = 0.f;
            for (int s = R.TD - 1; s >= 0; --s) {
                int t = dir ? R.TD - 1 - s : s;
                long row = (long)b * R.TD * plane + t * plane + yx;
                float gr = b2f(R.gates[row * 1024 + dir * 512 + ch]),
                      gi = b2f(R.gates[row * 1024 + dir * 512 + 256 + ch]);
                float rr = sigm(gr), ii = sigm(gi), xv = b2f(R.x[row * 256 + ch]);
                float a = expf(8.f * rr * ls), bb = sqrtf(fmaxf(1.f - a * a, 1e-12f));
                float hprev = 0.f;
                if (s > 0) {
                    int tp = dir ? R.TD - s : s - 1;
                    long rp = (long)b * R.TD * plane + tp * plane + yx;
                    hprev = R.h[(dir * (long)R.B * R.TD * plane + rp) * 256 + ch];
                }
                float g = R.dy[row * 256 + ch] + carry;
                carry = g * a;
                float da = g * (hprev - (a / bb) * ii * xv);
                float dix = g * bb;
                dxs[t] += dix * ii;
                float dgi = dix * xv * ii * (1.f - ii);
                float dgr = da * a * 8.f * ls * rr * (1.f - rr);
                dl[dir] += da * a * 8.f * rr * (1.f - sl);
                R.dg[row * 1024 + dir * 512 + ch] = __float2bfloat16_rn(dgr);
                R.dg[row * 1024 + dir * 512 + 256 + ch] = __float2bfloat16_rn(dgi);
            }
        }
        for (int t = 0; t < R.TD; ++t) {
            long row = (long)b * R.TD * plane + t * plane + yx;
            R.dx[row * 256 + ch] = dxs[t];
        }
        R.dlam[(long)col * 512 + ch] = dl[0];
        R.dlam[(long)col * 512 + 256 + ch] = dl[1];
    }
}

template <int C>
__device__ void lnc_fwd(
    const bf16 *x, long sites, const float *g, const float *b, bf16 *y, float *stat) {
    for (long s = gtid(); s < sites; s += gstride()) {
        const uint4 *p = reinterpret_cast<const uint4 *>(x + s * C);
        float sum = 0.f;
        for (int k = 0; k < C / 8; ++k) {
            uint4 u = p[k];
            const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(h[j]);
                sum += f.x + f.y;
            }
        }
        float mu = sum / C, sq = 0.f;
        for (int k = 0; k < C / 8; ++k) {
            uint4 u = p[k];
            const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(h[j]);
                float dx0 = f.x - mu, dx1 = f.y - mu;
                sq += dx0 * dx0 + dx1 * dx1;
            }
        }
        float rs = rsqrtf(sq / C + 0.1f);
        uint4 *o = reinterpret_cast<uint4 *>(y + s * C);
        for (int k = 0; k < C / 8; ++k) {
            uint4 u = p[k];
            const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
            unsigned r[4];
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(h[j]);
                int c = k * 8 + j * 2;
                r[j] = pack2((f.x - mu) * rs * g[c] + b[c], (f.y - mu) * rs * g[c + 1] + b[c + 1]);
            }
            o[k] = make_uint4(r[0], r[1], r[2], r[3]);
        }
        stat[2 * s] = mu;
        stat[2 * s + 1] = rs;
    }
}

template <int C>
__device__ void lnc_bg(const bf16 *x, const float *stat, const bf16 *dy, long sites, float *part,
    unsigned char *smem) {
    constexpr int NG = Threads / C;
    const int c = threadIdx.x % C, grp = threadIdx.x / C;
    const long s0 = sites * blockIdx.x / gridDim.x, s1 = sites * (blockIdx.x + 1) / gridDim.x;
    float ag = 0.f, ab = 0.f;
    for (long s = s0 + grp; s < s1; s += NG) {
        float d = b2f(dy[s * C + c]);
        ag += d * (b2f(x[s * C + c]) - stat[2 * s]) * stat[2 * s + 1];
        ab += d;
    }
    float *red = reinterpret_cast<float *>(smem);
    __syncthreads();
    red[grp * 2 * C + c] = ag;
    red[grp * 2 * C + C + c] = ab;
    __syncthreads();
    for (int k = threadIdx.x; k < 2 * C; k += Threads) {
        float t = 0.f;
        for (int q = 0; q < NG; ++q)
            t += red[q * 2 * C + k];
        part[(long)blockIdx.x * 2 * C + k] = t;
    }
    __syncthreads();
}

template <int C>
__device__ void lnc_bx(const bf16 *x, const float *stat, bf16 *dy, long sites, const float *g) {
    for (long s = gtid(); s < sites; s += gstride()) {
        const float mu = stat[2 * s], rs = stat[2 * s + 1];
        const uint4 *px = reinterpret_cast<const uint4 *>(x + s * C);
        uint4 *pd = reinterpret_cast<uint4 *>(dy + s * C);
        float s1 = 0.f, s2 = 0.f;
        for (int k = 0; k < C / 8; ++k) {
            uint4 u = px[k], v = pd[k];
            const __nv_bfloat162 *hx = reinterpret_cast<const __nv_bfloat162 *>(&u),
                                 *hd = reinterpret_cast<const __nv_bfloat162 *>(&v);
            for (int j = 0; j < 4; ++j) {
                float2 fx = __bfloat1622float2(hx[j]), fd = __bfloat1622float2(hd[j]);
                int c = k * 8 + j * 2;
                float g0 = fd.x * g[c], g1 = fd.y * g[c + 1];
                s1 += g0 + g1;
                s2 += g0 * (fx.x - mu) * rs + g1 * (fx.y - mu) * rs;
            }
        }
        s1 /= C;
        s2 /= C;
        for (int k = 0; k < C / 8; ++k) {
            uint4 u = px[k], v = pd[k];
            const __nv_bfloat162 *hx = reinterpret_cast<const __nv_bfloat162 *>(&u),
                                 *hd = reinterpret_cast<const __nv_bfloat162 *>(&v);
            unsigned r[4];
            for (int j = 0; j < 4; ++j) {
                float2 fx = __bfloat1622float2(hx[j]), fd = __bfloat1622float2(hd[j]);
                int c = k * 8 + j * 2;
                r[j] = pack2(rs * (fd.x * g[c] - s1 - (fx.x - mu) * rs * s2),
                    rs * (fd.y * g[c + 1] - s1 - (fx.y - mu) * rs * s2));
            }
            pd[k] = make_uint4(r[0], r[1], r[2], r[3]);
        }
    }
}

struct LnA {
    const float *x = nullptr;
    int rows = 0;
    const float *g = nullptr, *b = nullptr;
    bf16 *yh = nullptr;
    float *yf = nullptr;
    float *stat = nullptr;
    const float *dy = nullptr;
    float *dx = nullptr;
    int acc = 0;
    float *part = nullptr;
};

struct Pair {
    float a[3], b[3];
    float anga, angb, target, weight;
    int kind, token;
};

struct Meta {
    int curl, recon, labelled, group;
    float conf, w_recon, w_pos, w_neg, w_con_pos, w_con_neg;
    float box[4];
    float origin_z, voxel_scale;
    int npairs, ncsr, nogeom, pad_;
};

struct Misc {
    const void *p0 = nullptr, *p1 = nullptr, *p2 = nullptr, *p3 = nullptr, *p4 = nullptr;
    void *q0 = nullptr, *q1 = nullptr, *q2 = nullptr, *q3 = nullptr;
    long n = 0, m = 0;
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0;
    float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
};

enum OpKind {
    O_GEMM = 0,
    O_CONV_A,
    O_CONV_B,
    O_CONV_C,
    O_CONV_D,
    O_CONVW10,
    O_CONVW12,
    O_CONVW15,
    O_CONV1F,
    O_CONV1W,
    O_REDUCE,
    O_LNF,
    O_LNB,
    O_ATTF,
    O_ATTM,
    O_ATTD,
    O_ATTB,
    O_ATTR,
    O_RECF,
    O_RECB,
    O_CVT,
    O_LOOPIN,
    O_CKSAVE,
    O_CKREST,
    O_FOURIER,
    O_MASKTOK,
    O_HEADS,
    O_PAIRS,
    O_SCATTER,
    O_VOXLOSS,
    O_CONF,
    O_COLSUM,
    O_ZERO,
    O_EMBGRAD,
    O_MASKGRAD,
    O_LATB,
    O_LATBG,
    O_ADDF,
    O_OPTNORM,
    O_OPTSTEP,
    O_COPYF,
    O_SWIGLUB,
    O_MARKERG,
    O_LNCF,
    O_LNCBG,
    O_LNCBX
};

struct Op {
    int kind;
    int pad[3];
    alignas(16) unsigned char args[480];
};

struct StepIO {
    const bf16 *ct, *ctt;
    const uint8_t *lab, *tokmask;
    const float *centre;
    const Meta *meta;
    const Pair *pairs;
    const int *ptok;
    const int *csr_v, *csr_o, *csr_e;
    const float *csr_w;
    float *metrics;
    const float *scal;
    int B, D, H, W, T, maxpairs, maxcsr, maxent;
};

__device__ __forceinline__ float trilin(
    const float *h4, int D, int H, int W, const float *p, int b) {
    float z = fminf(fmaxf(p[0], 0.f), D - 1.f), y = fminf(fmaxf(p[1], 0.f), H - 1.f),
          x = fminf(fmaxf(p[2], 0.f), W - 1.f);
    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - z0, fy = y - y0, fx = x - x0, s = 0.f;
    for (int c = 0; c < 8; ++c) {
        int zz = c & 4 ? z1 : z0, yy = c & 2 ? y1 : y0, xx = c & 1 ? x1 : x0;
        float w = (c & 4 ? fz : 1 - fz) * (c & 2 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx);
        s += w * h4[((((long)b * D + zz) * H + yy) * W + xx) * 4 + 1];
    }
    return s;
}

__device__ __forceinline__ float huber(float e, float d) {
    float a = fabsf(e);
    return a <= d ? 0.5f * e * e : d * (a - 0.5f * d);
}
__device__ __forceinline__ float huber_d(float e, float d) { return fminf(d, fmaxf(-d, e)); }

__device__ void block_reduce_store(float *vals, int n, float *dst, unsigned char *smem) {
    float *red = reinterpret_cast<float *>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        float v = vals[i];
        for (int o = 16; o; o >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0)
            red[i * Warps + warp] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += Threads) {
        float s = 0.f;
        for (int w = 0; w < Warps; ++w)
            s += red[i * Warps + w];
        dst[(long)blockIdx.x * n + i] = s;
    }
    __syncthreads();
}

__device__ void op_misc(
    int kind, const Misc &m, const StepIO &io, int loop, int slot, unsigned char *smem) {
    switch (kind) {
    case O_CVT: {
        const float *s = (const float *)m.p0;
        bf16 *d = (bf16 *)m.q0;
        for (long i = gtid(); i < m.n; i += gstride())
            d[i] = __float2bfloat16_rn(s[i]);
        break;
    }
    case O_COPYF: {
        const float *s = (const float *)m.p0;
        float *d = (float *)m.q0;
        for (long i = gtid(); i < m.n; i += gstride())
            d[i] = s[i];
        break;
    }
    case O_ZERO: {
        float *d = (float *)m.q0;
        for (long i = gtid(); i < m.n; i += gstride())
            d[i] = 0.f;
        break;
    }
    case O_ADDF: {
        const float *s = (const float *)m.p0;
        float *d = (float *)m.q0;
        for (long i = gtid(); i < m.n; i += gstride())
            d[i] += s[i];
        break;
    }
    case O_LOOPIN: {
        const float *P = (const float *)m.p0, *S = (const float *)m.p1,
                    *E = (const float *)m.p2 + loop * 256, *Lt = (const float *)m.p3;
        float *Pin = (float *)m.q0, *Lin = (float *)m.q1;
        for (long i = gtid(); i < m.n; i += gstride())
            Pin[i] = P[i] + S[i] + E[i & 255];
        for (long i = gtid(); i < m.m; i += gstride())
            Lin[i] = Lt[i] + E[i & 255];
        break;
    }
    case O_CKSAVE:
    case O_CKREST: {
        float *ck = (float *)m.q2 + (long)slot * (m.n + m.m);
        float *P = (float *)m.q0, *Lt = (float *)m.q1;
        for (long i = gtid(); i < m.n; i += gstride()) {
            if (kind == O_CKSAVE)
                ck[i] = P[i];
            else
                P[i] = ck[i];
        }
        for (long i = gtid(); i < m.m; i += gstride()) {
            if (kind == O_CKSAVE)
                ck[m.n + i] = Lt[i];
            else
                Lt[i] = ck[m.n + i];
        }
        break;
    }
    case O_FOURIER: {
        bf16 *F = (bf16 *)m.q0;
        const int TD = io.D / 4, TH = io.H / 8, TW = io.W / 8, T = TD * TH * TW;
        for (long i = gtid(); i < (long)io.B * T * 128; i += gstride()) {
            int f = i % 128, row = i / 128, b = row / T, t = row % T;
            int tx = t % TW, ty = (t / TW) % TH, tz = t / (TW * TH);
            float z = 4 * tz + 1.5f, y = 8 * ty + 3.5f, x = 8 * tx + 3.5f;
            int zi = 4 * tz + 1;
            float cy = 0.5f *
                (io.centre[((long)b * io.D + zi) * 2] + io.centre[((long)b * io.D + zi + 1) * 2]);
            float cx = 0.5f *
                (io.centre[((long)b * io.D + zi) * 2 + 1] +
                    io.centre[((long)b * io.D + zi + 1) * 2 + 1]);
            float c = (float)io.meta[b].curl;
            float vs = io.meta[b].voxel_scale > 0.f ? io.meta[b].voxel_scale : 1.f;
            float th = atan2f(y - cy, x - cx);
            float sc[4] = {hypotf(y - cy, x - cx) * vs / 1000.f, c * sinf(th), cosf(th),
                (io.meta[b].origin_z + z) * vs / 1000.f};
            int which = f / 32, k = (f % 32) / 2;
            float fr = 3.14159265f * exp2f(0.5f * k), v = fr * sc[which];
            F[i] = __float2bfloat16_rn(io.meta[b].nogeom ? 0.f : (f & 1) ? cosf(v) : sinf(v));
        }
        break;
    }
    case O_MASKTOK: {
        float *S = (float *)m.q0;
        const float *mk = (const float *)m.p0;
        for (long i = gtid(); i < (long)io.B * io.T * 256; i += gstride())
            if (io.tokmask[i / 256])
                S[i] = mk[i & 255];
        break;
    }
    case O_MASKGRAD: {
        float *dS = (float *)m.q0;
        float acc = 0.f;
        int c = threadIdx.x;
        for (long r = blockIdx.x; r < (long)io.B * io.T; r += gridDim.x)
            if (io.tokmask[r]) {
                acc += dS[r * 256 + c];
                dS[r * 256 + c] = 0.f;
            }
        ((float *)m.q1)[(long)blockIdx.x * 256 + c] = acc;
        break;
    }
    case O_COLSUM: {
        const bf16 *s = (const bf16 *)m.p0;
        const float *sf = (const float *)m.p1;
        long rows = m.n;
        int N = m.i0;
        long r0 = rows * blockIdx.x / gridDim.x, r1 = rows * (blockIdx.x + 1) / gridDim.x;
        for (int c = threadIdx.x; c < N; c += Threads) {
            float acc = 0.f;
            for (long r = r0; r < r1; ++r)
                acc += s ? b2f(s[r * N + c]) : sf[r * N + c];
            ((float *)m.q0)[(long)blockIdx.x * N + c] = acc;
        }
        break;
    }
    case O_EMBGRAD: {
        const float *a = (const float *)m.p0, *b = (const float *)m.p1;
        int c = threadIdx.x;
        float acc = 0.f;
        for (long r = blockIdx.x; r < m.n; r += gridDim.x)
            acc += a[r * 256 + c];
        for (long r = blockIdx.x; r < m.m; r += gridDim.x)
            acc += b[r * 256 + c];
        ((float *)m.q0)[(long)blockIdx.x * 256 + c] = acc;
        break;
    }
    case O_SWIGLUB: {
        const float *dh = (const float *)m.p0;
        const bf16 *pre = (const bf16 *)m.p1;
        bf16 *dp = (bf16 *)m.q0;
        const int hd = m.i0;
        for (long i = gtid(); i < m.n * hd; i += gstride()) {
            long r = i / hd;
            int j = i % hd;
            float2 ab = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162 *>(pre + r * 2 * hd + 2 * j));
            float sg = sigm(ab.x), si = ab.x * sg, g = dh[i];
            *reinterpret_cast<unsigned *>(dp + r * 2 * hd + 2 * j) =
                pack2(g * ab.y * sg * (1.f + ab.x * (1.f - sg)), g * si);
        }
        break;
    }
    case O_MARKERG: {
        const float *d = (const float *)m.p0, *bx = (const float *)m.p1;
        const int g = m.i0 / 16, per = g * g;
        int c = threadIdx.x;
        float acc = 0.f;
        for (long r = blockIdx.x; r < m.n; r += gridDim.x) {
            int ox = r % g, oy = (r / g) % g, bb = r / per;
            const float *b4 = bx + bb * 4;
            if (16.f * ox < b4[3] && 16.f * (ox + 1) > b4[1] && 16.f * oy < b4[2] &&
                16.f * (oy + 1) > b4[0])
                acc += d[r * 256 + c];
        }
        ((float *)m.q0)[(long)blockIdx.x * 256 + c] = acc;
        break;
    }
    case O_LATB: {
        const float *lat = (const float *)m.p0;
        float *d = (float *)m.q0;
        for (long i = gtid(); i < m.n; i += gstride())
            d[i] = lat[i % m.m];
        break;
    }
    case O_LATBG: {
        const float *d = (const float *)m.p0;
        float *g = (float *)m.q0;
        for (long i = gtid(); i < m.m; i += gstride()) {
            float s = 0.f;
            for (int b = 0; b < m.i0; ++b)
                s += d[b * m.m + i];
            g[i] += s;
        }
        break;
    }
    case O_HEADS: {
        const bf16 *z = (const bf16 *)m.p0;
        const float *w = (const float *)m.p1, *bias = (const float *)m.p2;
        float *out = (float *)m.q0;
        for (long v = gtid(); v < m.n; v += gstride()) {
            float o[4] = {bias[0], bias[1], bias[2], bias[3]};
            const uint4 *zp = reinterpret_cast<const uint4 *>(z + v * 32);
            for (int k = 0; k < 4; ++k) {
                uint4 u = zp[k];
                const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
                for (int j = 0; j < 4; ++j) {
                    float2 f = __bfloat1622float2(h[j]);
                    int c = k * 8 + j * 2;
                    float a0 = gelu(f.x), a1 = gelu(f.y);
                    for (int q = 0; q < 4; ++q)
                        o[q] += a0 * w[c * 4 + q] + a1 * w[(c + 1) * 4 + q];
                }
            }
            *reinterpret_cast<float4 *>(out + v * 4) = make_float4(o[0], o[1], o[2], o[3]);
        }
        break;
    }
    case O_PAIRS: {
        const float *h4 = (const float *)m.p0;
        float *pl = (float *)m.q0;
        float2 *pg = (float2 *)m.q1;
        for (long i = gtid(); i < (long)io.B * io.maxpairs; i += gstride()) {
            int b = i / io.maxpairs, p = i % io.maxpairs;
            if (p >= io.meta[b].npairs)
                continue;
            const Pair &q = io.pairs[i];
            float ua = trilin(h4, io.D, io.H, io.W, q.a, b) - q.anga, ub = 0.f;
            if (q.kind != 3)
                ub = trilin(h4, io.D, io.H, io.W, q.b, b) - q.angb;
            float e = q.kind == 0 ? ua - ub
                : q.kind == 1     ? ub - ua - 1.f
                : q.kind == 2     ? ua - ub - q.target
                                  : ua - q.target;
            float d = q.kind == 3 ? 0.5f : 0.25f;
            float g = q.weight * huber_d(e, d);
            pl[i] = q.weight * huber(e, d);
            pg[i] = q.kind == 1 ? make_float2(-g, g) : make_float2(g, q.kind == 3 ? 0.f : -g);
        }
        break;
    }
    case O_SCATTER: {
        const float2 *pg = (const float2 *)m.p0;
        float *dt = (float *)m.q0;
        for (long i = gtid(); i < (long)io.B * io.maxcsr; i += gstride()) {
            int b = i / io.maxcsr, k = i % io.maxcsr;
            if (k >= io.meta[b].ncsr)
                continue;
            const int *off = io.csr_o + (long)b * (io.maxcsr + 1);
            float s = 0.f;
            for (int e = off[k]; e < off[k + 1]; ++e) {
                int ent = io.csr_e[(long)b * io.maxent + e];
                float2 g = pg[(long)b * io.maxpairs + (ent >> 1)];
                s += io.csr_w[(long)b * io.maxent + e] * ((ent & 1) ? g.y : g.x);
            }
            dt[(long)b * io.D * io.H * io.W + io.csr_v[(long)b * io.maxcsr + k]] = s;
        }
        break;
    }
    case O_VOXLOSS: {
        const float *h4 = (const float *)m.p0, *dt = (const float *)m.p1, *hw = (const float *)m.p3;
        const bf16 *z = (const bf16 *)m.p2;
        const float *pl = (const float *)m.p4;
        bf16 *dz = (bf16 *)m.q0;
        float *tokl = (float *)m.q1;
        float *part = (float *)m.q2;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int TD = io.D / 4, TH = io.H / 8, TW = io.W / 8, T = TD * TH * TW;
        float acc[132 + 8];
        for (int i = 0; i < 140; ++i)
            acc[i] = 0.f;
        for (long tk = blockIdx.x * Warps + warp; tk < (long)io.B * T; tk += gridDim.x * Warps) {
            int b = tk / T, t = tk % T;
            int tx = t % TW, ty = (t / TW) % TH, tz = t / (TW * TH);
            const Meta &mt = io.meta[b];
            bool masked = io.tokmask[tk] && mt.recon;
            float tl = 0.f;
            for (int k = 0; k < 8; ++k) {
                int vi = lane + 32 * k, vx = vi % 8, vy = (vi / 8) % 8, vz = vi / 64;
                long v = (((long)b * io.D + 4 * tz + vz) * io.H + 8 * ty + vy) * io.W + 8 * tx + vx;
                float4 o = *reinterpret_cast<const float4 *>(h4 + v * 4);
                float d[4] = {0.f, dt[v], 0.f, 0.f};
                if (masked) {
                    float e = o.w - b2f(io.ctt[v]);
                    tl += mt.w_recon * e * e;
                    acc[132] += mt.w_recon * e * e;
                    d[3] = 2.f * mt.w_recon * e;
                }
                uint8_t lb = io.lab[v];
                if (lb & 1) {
                    float y = (lb & 2) ? 1.f : 0.f, w = y > 0 ? mt.w_pos : mt.w_neg;
                    float l = w * (softplusf(o.x) - y * o.x);
                    tl += l;
                    acc[133] += l;
                    d[0] = w * (sigm(o.x) - y);
                    float pr = sigm(o.x);
                    acc[137] += (pr > 0.5f && y > 0) ? 1.f : 0.f;
                    acc[138] += (pr > 0.5f && y == 0) ? 1.f : 0.f;
                    acc[139] += (pr <= 0.5f && y > 0) ? 1.f : 0.f;
                }
                if (lb & 4) {
                    float y = (lb & 8) ? 1.f : 0.f, w = y > 0 ? mt.w_con_pos : mt.w_con_neg;
                    float l = w * (softplusf(o.z) - y * o.z);
                    tl += l;
                    acc[134] += l;
                    d[2] = w * (sigm(o.z) - y);
                }
                const uint4 *zp = reinterpret_cast<const uint4 *>(z + v * 32);
                uint4 outv[4];
                for (int kk = 0; kk < 4; ++kk) {
                    uint4 u = zp[kk];
                    const __nv_bfloat162 *hh = reinterpret_cast<const __nv_bfloat162 *>(&u);
                    unsigned res[4];
                    for (int j = 0; j < 4; ++j) {
                        float2 f = __bfloat1622float2(hh[j]);
                        int c = kk * 8 + j * 2;
                        float a0 = gelu(f.x), a1 = gelu(f.y), s0 = 0.f, s1 = 0.f;
                        for (int q = 0; q < 4; ++q) {
                            s0 += hw[c * 4 + q] * d[q];
                            s1 += hw[(c + 1) * 4 + q] * d[q];
                            acc[c * 4 + q] += a0 * d[q];
                            acc[(c + 1) * 4 + q] += a1 * d[q];
                        }
                        res[j] = pack2(s0 * gelu_d(f.x), s1 * gelu_d(f.y));
                    }
                    outv[kk] = make_uint4(res[0], res[1], res[2], res[3]);
                }
                uint4 *dp = reinterpret_cast<uint4 *>(dz + v * 32);
                for (int kk = 0; kk < 4; ++kk)
                    dp[kk] = outv[kk];
                for (int q = 0; q < 4; ++q)
                    acc[128 + q] += d[q];
            }
            for (int o = 16; o; o >>= 1)
                tl += __shfl_xor_sync(0xffffffffu, tl, o);
            if (lane == 0) {
                int p0 = io.ptok[(long)b * (T + 1) + t], p1 = io.ptok[(long)b * (T + 1) + t + 1];
                float ps = 0.f;
                for (int p = p0; p < p1; ++p)
                    ps += pl[(long)b * io.maxpairs + p];
                acc[135] += ps;
                tokl[tk] = (tl + ps) * io.B * T;
            }
        }
        block_reduce_store(acc, 140, part, smem);
        break;
    }
    case O_CONF: {
        const float *P = (const float *)m.p0, *w = (const float *)m.p1, *bias = (const float *)m.p2,
                    *tokl = (const float *)m.p3;
        float *dP = (float *)m.q0, *conf = (float *)m.q1, *part = (float *)m.q2;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const float lw = m.f0 / (float)(io.B * io.T);
        float acc[8 + 2];
        for (int i = 0; i < 10; ++i)
            acc[i] = 0.f;
        float gw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const float psc = 0.0625f;
        for (long tk = blockIdx.x * Warps + warp; tk < (long)io.B * io.T; tk += gridDim.x * Warps) {
            const float *p = P + tk * 256 + lane * 8;
            float s = 0.f;
            for (int i = 0; i < 8; ++i)
                s += (p[i] * psc) * w[lane * 8 + i];
            for (int o = 16; o; o >>= 1)
                s += __shfl_xor_sync(0xffffffffu, s, o);
            s += bias[0];
            float c = softplusf(s), e = c - log1pf(fmaxf(tokl[tk], 0.f));
            if (lane == 0) {
                conf[tk] = c;
                acc[8] += lw * e * e;
            }
            float g = m.i0 ? 2.f * lw * e * sigm(s) : 0.f;
            if (lane == 0)
                acc[9] += g;
            if (m.i0) {
                float *dp = dP + tk * 256 + lane * 8;
                for (int i = 0; i < 8; ++i) {
                    dp[i] += g * psc * w[lane * 8 + i];
                    gw[i] += g * (p[i] * psc);
                }
            }
        }
        float *red = reinterpret_cast<float *>(smem);
        __syncthreads();
        for (int i = 0; i < 8; ++i)
            red[warp * 256 + lane * 8 + i] = gw[i];
        if (lane == 0) {
            red[2048 + warp] = acc[8];
            red[2048 + 8 + warp] = acc[9];
        }
        __syncthreads();
        for (int c = threadIdx.x; c < 258; c += Threads) {
            float s = 0.f;
            for (int ww = 0; ww < Warps; ++ww)
                s += c < 256 ? red[ww * 256 + c] : red[2048 + (c - 256) * 8 + ww];
            part[(long)blockIdx.x * 258 + c] = s;
        }
        __syncthreads();
        break;
    }
    case O_OPTNORM: {
        const float *g = (const float *)m.p0;
        float *part = (float *)m.q0;
        long n = m.n, r0 = n * blockIdx.x / gridDim.x, r1 = n * (blockIdx.x + 1) / gridDim.x;
        float s = 0.f;
        for (long i = r0 + threadIdx.x; i < r1; i += Threads)
            s += g[i] * g[i];
        block_reduce_store(&s, 1, part, smem);
        break;
    }
    case O_OPTSTEP: {
        float *master = (float *)m.q0, *m1 = (float *)m.q1, *m2 = (float *)m.q2;
        bf16 *wb = (bf16 *)m.q3;
        const float *g = (const float *)m.p0, *part = (const float *)m.p1;
        const uint8_t *flags = (const uint8_t *)m.p2;
        float *metrics = io.metrics;
        float tot = 0.f;
        for (int k = 0; k < (int)gridDim.x; ++k)
            tot += part[k];
        float norm = sqrtf(tot), clip = norm > m.f1 ? m.f1 / (norm + 1e-6f) : 1.f;
        if (gtid() == 0) {
            metrics[40] = norm;
            metrics[41] = clip;
        }
        float lr = io.scal[0], b1 = 0.9f, b2 = 0.95f;
        float c1 = 1.f - powf(b1, io.scal[1]), c2 = 1.f - powf(b2, io.scal[1]);
        bool skip_ctx = io.scal[2] < 0.f;
        bool finite = isfinite(norm);
        if (gtid() == 0)
            metrics[42] = finite ? 0.f : 1.f;
        if (!finite)
            break;
        for (long i = gtid(); i < m.n; i += gstride()) {
            uint8_t fl = flags[i];
            if ((fl & 2) || (skip_ctx && (fl & 4)))
                continue;
            float gg = g[i] * clip;
            float a = b1 * m1[i] + (1.f - b1) * gg, v = b2 * m2[i] + (1.f - b2) * gg * gg;
            m1[i] = a;
            m2[i] = v;
            float w = master[i];
            w -= lr * (a / c1 / (sqrtf(v / c2) + 1e-8f) + ((fl & 1) ? m.f2 * w : 0.f));
            master[i] = w;
            wb[i] = __float2bfloat16_rn(w);
        }
        break;
    }
    }
}

__device__ void exec_op(const Op &op, const StepIO &io, int loop, int slot, unsigned char *smem) {
    switch (op.kind) {
    case O_GEMM: {
        Gemm g = *reinterpret_cast<const Gemm *>(op.args);
        gemm_run(g, smem);
        break;
    }
    case O_CONV_A: {
        Conv c = *reinterpret_cast<const Conv *>(op.args);
        conv_run<8, 2, 4, 10>(c, smem);
        break;
    }
    case O_CONV_B: {
        Conv c = *reinterpret_cast<const Conv *>(op.args);
        conv_run<4, 4, 4, 10>(c, smem);
        break;
    }
    case O_CONV_C: {
        Conv c = *reinterpret_cast<const Conv *>(op.args);
        conv_run<4, 2, 4, 15>(c, smem);
        break;
    }
    case O_CONV_D: {
        Conv c = *reinterpret_cast<const Conv *>(op.args);
        conv_run<2, 2, 2, 12>(c, smem);
        break;
    }
    case O_CONVW10: {
        ConvW w = *reinterpret_cast<const ConvW *>(op.args);
        convw_run<10>(w, smem);
        break;
    }
    case O_CONVW12: {
        ConvW w = *reinterpret_cast<const ConvW *>(op.args);
        convw_run<12>(w, smem);
        break;
    }
    case O_CONVW15: {
        ConvW w = *reinterpret_cast<const ConvW *>(op.args);
        convw_run<15>(w, smem);
        break;
    }
    case O_CONV1F: {
        const Misc &m = *reinterpret_cast<const Misc *>(op.args);
        conv1_fwd((const bf16 *)m.p0, io.B, io.D, io.H, io.W, (const float *)m.p1,
            (const float *)m.p2, (bf16 *)m.q0, smem);
        break;
    }
    case O_CONV1W: {
        const Misc &m = *reinterpret_cast<const Misc *>(op.args);
        conv1_dw(
            (const bf16 *)m.p0, (const bf16 *)m.p1, io.B, io.D, io.H, io.W, (float *)m.q0, smem);
        break;
    }
    case O_REDUCE: {
        const Misc &m = *reinterpret_cast<const Misc *>(op.args);
        reduce_parts(
            (float *)m.q0 + (long)loop * m.i3, (const float *)m.p0, m.n, m.i0, m.i1, m.m, m.i2);
        break;
    }
    case O_LNF: {
        LnA l = *reinterpret_cast<const LnA *>(op.args);
        ln_fwd(l.x, l.rows, l.g, l.b, l.yh, l.yf, l.stat);
        break;
    }
    case O_LNB: {
        LnA l = *reinterpret_cast<const LnA *>(op.args);
        ln_bwd(l.x, l.stat, l.dy, l.rows, l.g, l.dx, l.acc, l.part, smem);
        break;
    }
    case O_ATTF: {
        Attn a = *reinterpret_cast<const Attn *>(op.args);
        attn_fwd(a, smem);
        break;
    }
    case O_ATTM: {
        Attn a = *reinterpret_cast<const Attn *>(op.args);
        attn_merge(a);
        break;
    }
    case O_ATTD: {
        Attn a = *reinterpret_cast<const Attn *>(op.args);
        attn_dsum(a, a.o, a.B * (a.kind == 0 ? a.TD * a.TH * a.TW : a.nq));
        break;
    }
    case O_ATTB: {
        Attn a = *reinterpret_cast<const Attn *>(op.args);
        attn_bwd(a, smem);
        break;
    }
    case O_ATTR: {
        Attn a = *reinterpret_cast<const Attn *>(op.args);
        attn_bwd_reduce(a);
        break;
    }
    case O_RECF: {
        Recur r = *reinterpret_cast<const Recur *>(op.args);
        recur_fwd(r);
        break;
    }
    case O_RECB: {
        Recur r = *reinterpret_cast<const Recur *>(op.args);
        recur_bwd(r);
        break;
    }
    case O_LNCF:
    case O_LNCBG:
    case O_LNCBX: {
        const Misc &m = *reinterpret_cast<const Misc *>(op.args);
        const bf16 *x = (const bf16 *)m.p0;
        const float *g = (const float *)m.p1, *st = (const float *)m.p3;
        if (op.kind == O_LNCF) {
            if (m.i0 == 32)
                lnc_fwd<32>(x, m.n, g, (const float *)m.p2, (bf16 *)m.q0, (float *)m.q1);
            if (m.i0 == 64)
                lnc_fwd<64>(x, m.n, g, (const float *)m.p2, (bf16 *)m.q0, (float *)m.q1);
            if (m.i0 == 128)
                lnc_fwd<128>(x, m.n, g, (const float *)m.p2, (bf16 *)m.q0, (float *)m.q1);
        } else if (op.kind == O_LNCBG) {
            if (m.i0 == 32)
                lnc_bg<32>(x, st, (const bf16 *)m.q0, m.n, (float *)m.q1, smem);
            if (m.i0 == 64)
                lnc_bg<64>(x, st, (const bf16 *)m.q0, m.n, (float *)m.q1, smem);
            if (m.i0 == 128)
                lnc_bg<128>(x, st, (const bf16 *)m.q0, m.n, (float *)m.q1, smem);
        } else {
            if (m.i0 == 32)
                lnc_bx<32>(x, st, (bf16 *)m.q0, m.n, g);
            if (m.i0 == 64)
                lnc_bx<64>(x, st, (bf16 *)m.q0, m.n, g);
            if (m.i0 == 128)
                lnc_bx<128>(x, st, (bf16 *)m.q0, m.n, g);
        }
        break;
    }
    default: {
        Misc m = *reinterpret_cast<const Misc *>(op.args);
        op_misc(op.kind, m, io, loop, slot, smem);
    }
    }
}

struct Plan {
    const Op *ops;
    int seg[16][2];
    int L, train, update, retain, zero_grad;
};
enum {
    SEG_STEM = 0,
    SEG_LOOP,
    SEG_DEC,
    SEG_LOSS,
    SEG_DBWD,
    SEG_LBWD,
    SEG_CTXB,
    SEG_SBWD,
    SEG_OPT,
    SEG_SAVE,
    SEG_REST,
    SEG_ZGRAD
};

__global__ void __launch_bounds__(Threads, 1)
    k_step(Plan p, StepIO io, unsigned long long *clocks) {
    extern __shared__ __align__(128) unsigned char smem[];
    cg::grid_group grid = cg::this_grid();
    auto run = [&](int s, int loop, int slot) {
        long long t0 = clock64();
        for (int i = p.seg[s][0]; i < p.seg[s][1]; ++i) {
            exec_op(p.ops[i], io, loop, slot, smem);
            grid.sync();
        }
        if (clocks && blockIdx.x == 0 && threadIdx.x == 0)
            clocks[s] += clock64() - t0;
    };
    if (p.zero_grad)
        run(SEG_ZGRAD, 0, 0);
    run(SEG_STEM, 0, 0);
    const int first = p.train ? max(0, p.L - p.retain) : p.L;
    for (int l = 0; l < p.L; ++l) {
        if (p.train && l >= first)
            run(SEG_SAVE, l, l - first);
        run(SEG_LOOP, l, 0);
    }
    run(SEG_DEC, 0, 0);
    run(SEG_LOSS, 0, 0);
    if (!p.train)
        return;
    run(SEG_DBWD, 0, 0);
    for (int l = p.L - 1; l >= first; --l) {
        run(SEG_REST, l, l - first);
        run(SEG_LOOP, l, 0);
        run(SEG_LBWD, l, 0);
    }
    if (first == 0)
        run(SEG_CTXB, 0, 0);
    run(SEG_SBWD, 0, 0);
    if (p.update)
        run(SEG_OPT, 0, 0);
}

struct Cfg {
    int B = 2, D = 64, H = 128, W = 128, C = 256, latents = 128, hidden = 688, hid = 683,
        retain = 6;
    int TD() const { return D / 4; }
    int TH() const { return H / 8; }
    int TW() const { return W / 8; }
    long T() const { return (long)TD() * TH() * TW(); }
    long V() const { return (long)D * H * W; }
    long V1() const { return V() / 4; }
    long V2() const { return V() / 32; }
    long Cn() const { return (long)(C / 16) * (C / 16); }
    long Cp() const { return (long)(C / 4) * (C / 4); }
};

struct ParamInfo {
    std::string name;
    long off, n;
    int decay;
};

struct Blk {
    float *xin, *p1, *p2, *p3, *stw, *stl, *stg, *stp, *sts, *str_, *stm, *lsew, *lseg, *lses,
        *hrec;
    bf16 *xw, *qkv, *ow, *xl, *xg, *qg, *kvg, *og, *xp, *xs, *qs, *kvs, *os, *xr, *gates, *yrec,
        *xm, *mpre, *hm;
    float *lin, *l1;
};

static int device_coop_blocks() {
    int dev = 0, sm = 0, bps = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_step, Threads, SmemBytes));
    if (sm < 1)
        sm = 1;
    if (bps < 1)
        bps = 1;
    return sm * bps;
}

struct Model {
    Cfg c;
    int G = 58;
    std::vector<ParamInfo> params;
    std::map<std::string, int> pidx;
    long ptotal = 0;
    std::vector<float> host_master;
    std::vector<uint8_t> host_flags;
    float *master = nullptr, *grad = nullptr, *m1 = nullptr, *m2 = nullptr;
    bf16 *wb = nullptr;
    uint8_t *flags = nullptr;
    std::vector<void *> allocs;
    long scratch_n = 0;
    float *scratch = nullptr;
    bf16 *ct = nullptr, *ctt = nullptr, *ctx = nullptr;
    uint8_t *lab = nullptr, *tokmask = nullptr;
    float *centre = nullptr, *boxes = nullptr;
    Meta *meta = nullptr;
    Pair *pairs = nullptr;
    int *ptok = nullptr, *csr_v = nullptr, *csr_o = nullptr, *csr_e = nullptr;
    float *csr_w = nullptr;
    float *metrics = nullptr, *scal = nullptr;
    int maxpairs = 12288, maxcsr = 16 * 12288, maxent = 16 * 12288;
    bf16 *z0a, *z0b, *z1a, *z1b, *z2a, *z2b, *F, *c1z, *c1a, *xlq, *xck, *qc, *kvc, *oc, *Pb, *u2u,
        *z2c, *u1u, *z1c, *u0u, *z0c;
    float *y3, *st3, *S, *ctxt, *stq, *stk, *L0, *Lt, *P, *ck, *lsec, *h4, *dturn, *pl, *tokl,
        *conf;
    float2 *pg;
    Blk blk[4];
    float *Pin, *Lin, *dres, *dlat, *dS, *dP, *dxa, *dxb2, *dxl_, *dog, *dqs, *dkvs, *dqg, *dkvg,
        *dqkv, *dy3, *dsum, *pdq, *pdk, *pdv, *apo, *apm, *apl, *dqc, *dkvc, *doc, *dctx, *dl0;
    bf16 *dresb, *dlatb, *dpre, *dgates, *dosb, *dqsb, *dkvsb, *dogb, *dqgb, *dkvgb, *dowb, *dqkvb,
        *dz0c, *du0u, *ds0, *dz1c, *du1u, *ds1, *dz2c, *du2u, *ds2, *dz1a, *dz2a, *dz0a, *dy3b,
        *dSb, *docb, *dqcb, *dkvcb, *dctxb, *dc1z;
    float *dhm;
    bf16 *r0a, *r0b, *r1a, *r1b, *r2a, *r2b, *r2c, *r1c, *r0c, *Pn;
    float *st0a, *st0b, *st1a, *st1b, *st2a, *st2b, *st2c, *st1c, *st0c, *stdec, *dPn;
    std::vector<Op> ops;
    int seg[16][2] = {};
    Op *dops = nullptr;

    template <class T> T *alloc(long n) {
        void *p = nullptr;
        CK(cudaMalloc(&p, std::max<long>(n, 1) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<long>(n, 1) * sizeof(T)));
        allocs.push_back(p);
        return (T *)p;
    }
    ~Model() {
        for (void *p : allocs)
            cudaFree(p);
    }

    void param(const std::string &name, long n, int decay) {
        pidx[name] = params.size();
        params.push_back({name, ptotal, n, decay});
        ptotal += n;
    }
    long off(const std::string &n) const {
        auto it = pidx.find(n);
        if (it == pidx.end())
            fail("no parameter " + n);
        return params[it->second].off;
    }
    float *Wf(const std::string &n) { return master + off(n); }
    bf16 *Wb(const std::string &n) { return wb + off(n); }
    float *Gr(const std::string &n) { return grad + off(n); }

    void declare() {
        auto conv = [&](const std::string &n, int cin, int cout) {
            param(n + ".w", 27L * cin * cout, 1);
            param(n + ".b", cout, 0);
        };
        auto lin = [&](const std::string &n, int k, int nn) {
            param(n + ".w", (long)k * nn, 1);
            param(n + ".b", nn, 0);
        };
        auto ln = [&](const std::string &n) {
            param(n + ".g", 256, 0);
            param(n + ".b", 256, 0);
        };
        auto lnc = [&](const std::string &n, int cc) {
            param(n + ".ln.g", cc, 0);
            param(n + ".ln.b", cc, 0);
        };
        param("s0a.w", 27 * 32, 1);
        param("s0a.b", 32, 0);
        lnc("s0a", 32);
        conv("s0b", 32, 32);
        lnc("s0b", 32);
        conv("s1a", 32, 64);
        lnc("s1a", 64);
        conv("s1b", 64, 64);
        lnc("s1b", 64);
        conv("s2a", 64, 128);
        lnc("s2a", 128);
        conv("s2b", 128, 128);
        lnc("s2b", 128);
        conv("s3", 128, 256);
        ln("stem.ln");
        lin("pos", 128, 256);
        param("mask", 256, 0);
        lin("ctx1", 64, 128);
        lin("ctx2", 2048, 256);
        param("marker", 256, 0);
        param("latents", (long)c.latents * 256, 0);
        ln("cx.lnq");
        ln("cx.lnk");
        lin("cx.q", 256, 256);
        lin("cx.kv", 256, 512);
        lin("cx.o", 256, 256);
        param("loopemb", 32 * 256, 0);
        for (int k = 0; k < 4; ++k) {
            std::string p = "b" + std::to_string(k) + ".";
            ln(p + "wn");
            lin(p + "qkv", 256, 768);
            lin(p + "wo", 256, 256);
            ln(p + "lnl");
            ln(p + "lng");
            lin(p + "gq", 256, 256);
            lin(p + "gkv", 256, 512);
            lin(p + "go", 256, 256);
            ln(p + "lnp");
            ln(p + "lns");
            lin(p + "sq", 256, 256);
            lin(p + "skv", 256, 512);
            lin(p + "so", 256, 256);
            ln(p + "lnr");
            lin(p + "rg", 256, 1024);
            param(p + "lam", 512, 0);
            lin(p + "ro", 256, 256);
            ln(p + "lnm");
            lin(p + "mi", 256, 2 * c.hidden);
            lin(p + "mo", c.hidden, 256);
        }
        ln("dec.ln");
        param("u2up.w", 256L * 8 * 128, 1);
        param("u2up.b", 128, 0);
        conv("u2c", 256, 128);
        lnc("u2c", 128);
        param("u1up.w", 128L * 8 * 64, 1);
        param("u1up.b", 64, 0);
        conv("u1c", 128, 64);
        lnc("u1c", 64);
        param("u0up.w", 64L * 4 * 32, 1);
        param("u0up.b", 32, 0);
        conv("u0c", 64, 32);
        lnc("u0c", 32);
        param("heads.w", 32 * 4, 1);
        param("heads.b", 4, 0);
        param("conf.w", 256, 1);
        param("conf.b", 1, 0);
    }

    void init(u64 seed) {
        host_master.assign(ptotal, 0.f);
        host_flags.assign(ptotal, 0);
        Rng r(seed);
        auto normal = [&]() {
            float u = std::max(r.uni(), 1e-7f), v = r.uni();
            return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v);
        };
        for (auto &p : params) {
            float *w = host_master.data() + p.off;
            std::string n = p.name;
            auto ends = [&](const char *s) {
                return n.size() >= strlen(s) && n.compare(n.size() - strlen(s), strlen(s), s) == 0;
            };
            float sd = 0.f;
            if (ends(".g")) {
                for (long i = 0; i < p.n; ++i)
                    w[i] = 1.f;
            } else if (ends(".b") || n == "conf.w") {
            } else if (n == "mask" || n == "marker" || n == "latents" || n == "loopemb") {
                sd = 0.02f;
            } else if (ends(".lam")) {
                for (long i = 0; i < p.n; ++i) {
                    float a = 0.9f + 0.099f * r.uni(), s = std::pow(a, 1.f / 8.f);
                    w[i] = std::log(s / (1.f - s));
                }
            } else {
                long fan = 1;
                if (n == "s0a.w")
                    fan = 27;
                else if (ends(".w") &&
                    (n[0] == 's' || n.rfind("u2c", 0) == 0 || n.rfind("u1c", 0) == 0 ||
                        n.rfind("u0c", 0) == 0) &&
                    n.find('.') < 4 && n != "s3.w")
                    fan = p.n / ([&] {
                        for (auto &q : params)
                            if (q.name == n.substr(0, n.find('.')) + ".b")
                                return q.n;
                        return 1L;
                    }());
                else if (n == "s3.w")
                    fan = 27 * 128;
                else if (n.rfind("u2up", 0) == 0)
                    fan = 256;
                else if (n.rfind("u1up", 0) == 0)
                    fan = 128;
                else if (n.rfind("u0up", 0) == 0)
                    fan = 64;
                else if (n == "heads.w")
                    fan = 32;
                else if (n == "ctx1.w")
                    fan = 64;
                else if (n == "ctx2.w")
                    fan = 2048;
                else if (n == "pos.w")
                    fan = 128;
                else if (ends("mo.w"))
                    fan = c.hidden;
                else
                    fan = 256;
                sd = std::sqrt(2.f / fan);
                if (ends("wo.w") || ends("go.w") || ends("so.w") || ends("ro.w") || ends("mo.w") ||
                    n == "cx.o.w")
                    sd = 0.1f / std::sqrt((float)fan);
                if (n == "heads.w")
                    sd = 1.f / std::sqrt((float)fan);
            }
            if (sd > 0)
                for (long i = 0; i < p.n; ++i)
                    w[i] = sd * normal();
            uint8_t base_flag = (uint8_t)(p.decay ? 1 : 0);
            if (n.rfind("ctx1.", 0) == 0 || n.rfind("ctx2.", 0) == 0 || n == "marker" ||
                n == "latents" || n.rfind("cx.", 0) == 0)
                base_flag |= 4;
            for (long i = 0; i < p.n; ++i)
                host_flags[p.off + i] = base_flag;
            if (ends("mi.w") || ends("mi.b")) {
                long cols = 2L * c.hidden, rows = p.n / cols;
                for (long rr = 0; rr < rows; ++rr)
                    for (long cc = 2L * c.hid; cc < cols; ++cc) {
                        w[rr * cols + cc] = 0.f;
                        host_flags[p.off + rr * cols + cc] = 2;
                    }
            }
            if (ends("mo.w"))
                for (long rr = c.hid; rr < c.hidden; ++rr)
                    for (long cc = 0; cc < 256; ++cc) {
                        w[rr * 256 + cc] = 0.f;
                        host_flags[p.off + rr * 256 + cc] = 2;
                    }
        }
    }

    void upload_params() {
        CK(cudaMemcpy(master, host_master.data(), ptotal * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(flags, host_flags.data(), ptotal, cudaMemcpyHostToDevice));
        std::vector<bf16> hb(ptotal);
        for (long i = 0; i < ptotal; ++i)
            hb[i] = __float2bfloat16(host_master[i]);
        CK(cudaMemcpy(wb, hb.data(), ptotal * 2, cudaMemcpyHostToDevice));
    }

    template <class T> void push(int kind, const T &a) {
        static_assert(sizeof(T) <= sizeof(Op::args), "op args too large");
        Op o{};
        o.kind = kind;
        memcpy(o.args, &a, sizeof(T));
        ops.push_back(o);
    }
    Misc mz() { return Misc(); }
    void zero(float *p, long n) {
        Misc m;
        m.q0 = p;
        m.n = n;
        push(O_ZERO, m);
    }
    void cvt(const float *s, bf16 *d, long n) {
        Misc m;
        m.p0 = s;
        m.q0 = d;
        m.n = n;
        push(O_CVT, m);
    }
    void copyf(const float *s, float *d, long n) {
        Misc m;
        m.p0 = s;
        m.q0 = d;
        m.n = n;
        push(O_COPYF, m);
    }
    void addf(const float *s, float *d, long n) {
        Misc m;
        m.p0 = s;
        m.q0 = d;
        m.n = n;
        push(O_ADDF, m);
    }
    void reduce(float *dst, const float *parts, long count, int splits, int acc, long stride = 0,
        int offset = 0, int loopstride = 0) {
        Misc m;
        m.q0 = dst;
        m.p0 = parts;
        m.n = count;
        m.i0 = splits;
        m.i1 = acc;
        m.m = stride;
        m.i2 = offset;
        m.i3 = loopstride;
        push(O_REDUCE, m);
    }
    void gemm(const Gemm &g) { push(O_GEMM, g); }
    int ksplit_for(long M, long N, long K) {
        long tiles = ((M + 127) / 128) * ((N + 63) / 64);
        long s = std::max<long>(1, (2L * G + tiles - 1) / tiles);
        s = std::min<long>(s, std::max<long>(1, K / 64));
        s = std::min<long>(s, scratch_n / std::max<long>(1, M * N));
        return (int)std::max<long>(1, s);
    }
    void lin(const bf16 *x, long rows, int K, const std::string &name, int N, Epi e, int act = 0) {
        Gemm g;
        g.M = rows;
        g.N = N;
        g.K = K;
        g.a.p = x;
        g.a.ld = K;
        g.act_a = act;
        g.b.p = Wb(name + ".w");
        g.b.ld = N;
        e.bias = Wf(name + ".b");
        g.e = e;
        gemm(g);
    }
    void lin_bwd(const bf16 *x, const bf16 *dc, long rows, int K, int N, const std::string &name,
        float *dx, int dxmode) {
        if (dx) {
            Gemm g;
            g.M = rows;
            g.N = K;
            g.K = N;
            g.a.p = dc;
            g.a.ld = N;
            g.b.p = Wb(name + ".w");
            g.b.ld = N;
            g.bnm = 1;
            g.e.mode = dxmode;
            g.e.f = dx;
            g.e.ld = K;
            gemm(g);
        }
        dweight(x, 0, K, rows, dc, N, Gr(name + ".w"));
        colsum(dc, nullptr, rows, N, Gr(name + ".b"));
    }
    void dweight(const bf16 *x, int act, int K, long rows, const bf16 *dc, int N, float *gw) {
        Gemm g;
        g.M = K;
        g.N = N;
        g.K = rows;
        g.a.p = x;
        g.a.ld = K;
        g.akm = 1;
        g.act_a = act;
        g.b.p = dc;
        g.b.ld = N;
        g.ksplit = ksplit_for(K, N, rows);
        g.e.mode = E_PART;
        g.e.f = scratch;
        gemm(g);
        reduce(gw, scratch, (long)K * N, g.ksplit, 1);
    }
    void colsum(const bf16 *s, const float *sf, long rows, int N, float *gb) {
        Misc m;
        m.p0 = s;
        m.p1 = sf;
        m.n = rows;
        m.i0 = N;
        m.q0 = scratch;
        push(O_COLSUM, m);
        reduce(gb, scratch, N, G, 1);
    }
    void lnf(const float *x, long rows, const std::string &name, bf16 *yh, float *yf, float *stat) {
        LnA l;
        l.x = x;
        l.rows = rows;
        l.g = Wf(name + ".g");
        l.b = Wf(name + ".b");
        l.yh = yh;
        l.yf = yf;
        l.stat = stat;
        push(O_LNF, l);
    }
    void lnb(const float *x, const float *stat, const float *dy, long rows, const std::string &name,
        float *dx, int acc) {
        LnA l;
        l.x = x;
        l.rows = rows;
        l.stat = const_cast<float *>(stat);
        l.dy = dy;
        l.g = Wf(name + ".g");
        l.dx = dx;
        l.acc = acc;
        l.part = scratch;
        push(O_LNB, l);
        reduce(Gr(name + ".g"), scratch, 256, G, 1, 512, 0);
        reduce(Gr(name + ".b"), scratch, 256, G, 1, 512, 256);
    }
    Conv convd(const bf16 *s0, int c0, int act0, const bf16 *s1, int c1, int act1, int sz, int sy,
        int sx, int oz, int oy, int ox, int st0, int st1, int st2, int tz, int ty, int tx,
        const std::string &name, int cin, int cout) {
        Conv k;
        k.s0 = s0;
        k.c0 = c0;
        k.act0 = act0;
        k.s1 = s1;
        k.c1 = c1;
        k.act1 = act1;
        k.TD = c.TD();
        k.TH = c.TH();
        k.TW = c.TW();
        k.B = c.B;
        k.SZ = sz;
        k.SY = sy;
        k.SX = sx;
        k.OZ = oz;
        k.OY = oy;
        k.OX = ox;
        k.sz = st0;
        k.sy = st1;
        k.sx = st2;
        k.TZ = tz;
        k.TY = ty;
        k.TX = tx;
        k.w = Wb(name + ".w");
        k.cin = cin;
        k.cout = cout;
        return k;
    }
    void convw(const bf16 *s0, int c0, int act0, const bf16 *s1, int c1, int act1,
        const uint8_t *mask, int ms0, int ms1, int ms2, int sz, int sy, int sx, int oz, int oy,
        int ox, int st0, int st1, int st2, int tz, int ty, int tx, const bf16 *dy, int cout,
        const std::string &name, int kind) {
        ConvW w;
        w.s0 = s0;
        w.c0 = c0;
        w.act0 = act0;
        w.s1 = s1;
        w.c1 = c1;
        w.act1 = act1;
        w.mask = mask;
        w.msh[0] = ms0;
        w.msh[1] = ms1;
        w.msh[2] = ms2;
        w.TD = c.TD();
        w.TH = c.TH();
        w.TW = c.TW();
        w.B = c.B;
        w.SZ = sz;
        w.SY = sy;
        w.SX = sx;
        w.OZ = oz;
        w.OY = oy;
        w.OX = ox;
        w.sz = st0;
        w.sy = st1;
        w.sx = st2;
        w.TZ = tz;
        w.TY = ty;
        w.TX = tx;
        w.dy = dy;
        w.cout = cout;
        int cin = c0 + c1, items = (cin / 32) * (cout / 32) * 2;
        long wsize = 27L * cin * cout;
        int splits = std::max(1, (2 * G + items - 1) / items);
        splits = (int)std::min<long>(splits, (scratch_n - 4096) / (wsize + cout));
        w.splits = std::max(1, splits);
        w.part = scratch;
        w.bpart = scratch + (long)w.splits * wsize;
        push(kind, w);
        reduce(Gr(name + ".w"), scratch, wsize, w.splits, 1);
        reduce(Gr(name + ".b"), w.bpart, cout, w.splits, 1);
    }
    Attn attn(int kind, const bf16 *q, long ldq, const bf16 *k, const bf16 *v, long ldk, int nq,
        int nk, bf16 *o, float *lse, int shift) {
        Attn a;
        a.kind = kind;
        a.q = q;
        a.ldq = ldq;
        a.k = k;
        a.v = v;
        a.ldk = ldk;
        a.B = c.B;
        a.nq = nq;
        a.nk = nk;
        a.TD = c.TD();
        a.TH = c.TH();
        a.TW = c.TW();
        a.shift = shift;
        a.o = o;
        a.lse = lse;
        a.po = apo;
        a.pm = apm;
        a.pl = apl;
        a.dsum = dsum;
        a.pdq = pdq;
        a.pdk = pdk;
        a.pdv = pdv;
        return a;
    }
    void attn_bwd_ops(
        Attn a, const bf16 *dO, float *dq, long lddq, float *dk, float *dv, long lddk) {
        a.dO = dO;
        a.dq = dq;
        a.lddq = lddq;
        a.dk = dk;
        a.dv = dv;
        a.lddk = lddk;
        push(O_ATTD, a);
        push(O_ATTB, a);
        if (a.kind != 0)
            push(O_ATTR, a);
    }

    void allocate() {
        G = device_coop_blocks();
        declare();
        master = alloc<float>(ptotal);
        grad = alloc<float>(ptotal);
        m1 = alloc<float>(ptotal);
        m2 = alloc<float>(ptotal);
        wb = alloc<bf16>(ptotal);
        flags = alloc<uint8_t>(ptotal);
        const long B = c.B, V = c.V(), V1 = c.V1(), V2 = c.V2(), T = c.T(), BT = B * T,
                   Lc = c.latents, BL = B * Lc;
        const long Cn = c.Cn(), Cp = c.Cp(), Hd = c.hidden;
        scratch_n = std::max<long>(
            24L << 20, std::max<long>((long)G * 256L * 256L, B * c.TH() * c.TW() * 512L + 4096L));
        scratch = alloc<float>(scratch_n);
        ct = alloc<bf16>(B * V);
        ctt = alloc<bf16>(B * V);
        lab = alloc<uint8_t>(B * V);
        tokmask = alloc<uint8_t>(BT);
        ctx = alloc<bf16>(B * c.C * c.C * 4);
        centre = alloc<float>(B * c.D * 2);
        boxes = alloc<float>(B * 4);
        meta = alloc<Meta>(B);
        pairs = alloc<Pair>(B * maxpairs);
        ptok = alloc<int>(B * (T + 1));
        csr_v = alloc<int>(B * maxcsr);
        csr_o = alloc<int>(B * (maxcsr + 1));
        csr_e = alloc<int>(B * maxent);
        csr_w = alloc<float>(B * maxent);
        metrics = alloc<float>(64);
        scal = alloc<float>(8);
        z0a = alloc<bf16>(B * V * 32);
        z0b = alloc<bf16>(B * V * 32);
        z1a = alloc<bf16>(B * V1 * 64);
        z1b = alloc<bf16>(B * V1 * 64);
        z2a = alloc<bf16>(B * V2 * 128);
        z2b = alloc<bf16>(B * V2 * 128);
        y3 = alloc<float>(BT * 256);
        st3 = alloc<float>(BT * 2);
        S = alloc<float>(BT * 256);
        F = alloc<bf16>(BT * 128);
        c1z = alloc<bf16>(B * Cp * 128);
        c1a = alloc<bf16>(B * Cp * 128);
        ctxt = alloc<float>(B * Cn * 256);
        L0 = alloc<float>(BL * 256);
        stq = alloc<float>(BL * 2);
        stk = alloc<float>(B * Cn * 2);
        xlq = alloc<bf16>(BL * 256);
        xck = alloc<bf16>(B * Cn * 256);
        qc = alloc<bf16>(BL * 256);
        kvc = alloc<bf16>(B * Cn * 512);
        oc = alloc<bf16>(BL * 256);
        lsec = alloc<float>(BL * 8);
        Lt = alloc<float>(BL * 256);
        P = alloc<float>(BT * 256);
        ck = alloc<float>((long)c.retain * (BT + BL) * 256);
        Pin = alloc<float>(BT * 256);
        Lin = alloc<float>(BL * 256);
        for (int k = 0; k < 4; ++k) {
            Blk &b = blk[k];
            b.xin = k == 0 ? Pin : alloc<float>(BT * 256);
            b.p1 = alloc<float>(BT * 256);
            b.p2 = alloc<float>(BT * 256);
            b.p3 = alloc<float>(BT * 256);
            b.stw = alloc<float>(BT * 2);
            b.stl = alloc<float>(BL * 2);
            b.stg = alloc<float>(BT * 2);
            b.stp = alloc<float>(BT * 2);
            b.sts = alloc<float>(BL * 2);
            b.str_ = alloc<float>(BT * 2);
            b.stm = alloc<float>(BT * 2);
            b.lsew = alloc<float>(BT * 8);
            b.lseg = alloc<float>(BL * 8);
            b.lses = alloc<float>(BT * 8);
            b.hrec = alloc<float>(2 * BT * 256);
            b.xw = alloc<bf16>(BT * 256);
            b.qkv = alloc<bf16>(BT * 768);
            b.ow = alloc<bf16>(BT * 256);
            b.xl = alloc<bf16>(BL * 256);
            b.xg = alloc<bf16>(BT * 256);
            b.qg = alloc<bf16>(BL * 256);
            b.kvg = alloc<bf16>(BT * 512);
            b.og = alloc<bf16>(BL * 256);
            b.xp = alloc<bf16>(BT * 256);
            b.xs = alloc<bf16>(BL * 256);
            b.qs = alloc<bf16>(BT * 256);
            b.kvs = alloc<bf16>(BL * 512);
            b.os = alloc<bf16>(BT * 256);
            b.xr = alloc<bf16>(BT * 256);
            b.gates = alloc<bf16>(BT * 1024);
            b.yrec = alloc<bf16>(BT * 256);
            b.xm = alloc<bf16>(BT * 256);
            b.mpre = alloc<bf16>(BT * 2 * Hd);
            b.hm = alloc<bf16>(BT * Hd);
            b.lin = k == 0 ? Lin : blk[k - 1].l1;
            b.l1 = alloc<float>(BL * 256);
        }
        Pb = alloc<bf16>(BT * 256);
        u2u = alloc<bf16>(B * V2 * 128);
        z2c = alloc<bf16>(B * V2 * 128);
        u1u = alloc<bf16>(B * V1 * 64);
        z1c = alloc<bf16>(B * V1 * 64);
        u0u = alloc<bf16>(B * V * 32);
        z0c = alloc<bf16>(B * V * 32);
        h4 = alloc<float>(B * V * 4);
        dturn = alloc<float>(B * V);
        pl = alloc<float>(B * maxpairs);
        pg = alloc<float2>(B * maxpairs);
        tokl = alloc<float>(BT);
        conf = alloc<float>(BT);
        dres = alloc<float>(BT * 256);
        dlat = alloc<float>(BL * 256);
        dS = alloc<float>(BT * 256);
        dP = dres;
        dxa = alloc<float>(BT * 256);
        dxb2 = alloc<float>(BT * 256);
        dxl_ = alloc<float>(BL * 256);
        dog = alloc<float>(BL * 256);
        dqs = alloc<float>(BT * 256);
        dkvs = alloc<float>(BL * 512);
        dqg = alloc<float>(BL * 256);
        dkvg = alloc<float>(BT * 512);
        dqkv = alloc<float>(BT * 768);
        dy3 = alloc<float>(BT * 256);
        dsum = alloc<float>(std::max(BT, B * Cn) * 8);
        dhm = alloc<float>(BT * Hd);
        long nchmax = std::max<long>((T + 127) / 128, (Cn + 127) / 128), nqtmax = (T + 63) / 64;
        pdq = alloc<float>(nchmax * BL * 256);
        pdk = alloc<float>(nqtmax * BL * 256);
        pdv = alloc<float>(nqtmax * BL * 256);
        apo = alloc<float>(nchmax * BL * 8 * 32);
        apm = alloc<float>(nchmax * BL * 8);
        apl = alloc<float>(nchmax * BL * 8);
        dqc = alloc<float>(BL * 256);
        dkvc = alloc<float>(B * Cn * 512);
        doc = alloc<float>(BL * 256);
        dctx = alloc<float>(B * Cn * 256);
        dl0 = alloc<float>(std::max(BL, B * Cn) * 256);
        dresb = alloc<bf16>(BT * 256);
        dlatb = alloc<bf16>(BL * 256);
        dpre = alloc<bf16>(BT * 2 * Hd);
        dgates = alloc<bf16>(BT * 1024);
        dosb = alloc<bf16>(BT * 256);
        dqsb = alloc<bf16>(BT * 256);
        dkvsb = alloc<bf16>(BL * 512);
        dogb = alloc<bf16>(BL * 256);
        dqgb = alloc<bf16>(BL * 256);
        dkvgb = alloc<bf16>(BT * 512);
        dowb = alloc<bf16>(BT * 256);
        dqkvb = alloc<bf16>(BT * 768);
        dz0c = alloc<bf16>(B * V * 32);
        du0u = alloc<bf16>(B * V * 32);
        ds0 = alloc<bf16>(B * V * 32);
        dz1c = alloc<bf16>(B * V1 * 64);
        du1u = alloc<bf16>(B * V1 * 64);
        ds1 = alloc<bf16>(B * V1 * 64);
        dz2c = alloc<bf16>(B * V2 * 128);
        du2u = alloc<bf16>(B * V2 * 128);
        ds2 = alloc<bf16>(B * V2 * 128);
        dz1a = alloc<bf16>(B * V1 * 64);
        dz2a = alloc<bf16>(B * V2 * 128);
        dz0a = alloc<bf16>(B * V * 32);
        dy3b = alloc<bf16>(BT * 256);
        dSb = alloc<bf16>(BT * 256);
        docb = alloc<bf16>(BL * 256);
        dqcb = alloc<bf16>(BL * 256);
        dkvcb = alloc<bf16>(B * Cn * 512);
        dctxb = alloc<bf16>(B * Cn * 256);
        dc1z = alloc<bf16>(B * Cp * 128);
        r0a = alloc<bf16>(B * V * 32);
        r0b = alloc<bf16>(B * V * 32);
        r0c = alloc<bf16>(B * V * 32);
        r1a = alloc<bf16>(B * V1 * 64);
        r1b = alloc<bf16>(B * V1 * 64);
        r1c = alloc<bf16>(B * V1 * 64);
        r2a = alloc<bf16>(B * V2 * 128);
        r2b = alloc<bf16>(B * V2 * 128);
        r2c = alloc<bf16>(B * V2 * 128);
        st0a = alloc<float>(B * V * 2);
        st0b = alloc<float>(B * V * 2);
        st0c = alloc<float>(B * V * 2);
        st1a = alloc<float>(B * V1 * 2);
        st1b = alloc<float>(B * V1 * 2);
        st1c = alloc<float>(B * V1 * 2);
        st2a = alloc<float>(B * V2 * 2);
        st2b = alloc<float>(B * V2 * 2);
        st2c = alloc<float>(B * V2 * 2);
        Pn = alloc<bf16>(BT * 256);
        stdec = alloc<float>(BT * 2);
        dPn = alloc<float>(BT * 256);
    }
    void lncf(const bf16 *r, bf16 *z, float *st, long sites, int C, const std::string &name) {
        Misc m;
        m.p0 = r;
        m.p1 = Wf(name + ".ln.g");
        m.p2 = Wf(name + ".ln.b");
        m.q0 = z;
        m.q1 = st;
        m.n = sites;
        m.i0 = C;
        push(O_LNCF, m);
    }
    void lncb(
        const bf16 *r, const float *st, bf16 *dz, long sites, int C, const std::string &name) {
        Misc m;
        m.p0 = r;
        m.p1 = Wf(name + ".ln.g");
        m.p3 = st;
        m.q0 = dz;
        m.q1 = scratch;
        m.n = sites;
        m.i0 = C;
        push(O_LNCBG, m);
        reduce(Gr(name + ".ln.g"), scratch, C, G, 1, 2 * C, 0);
        reduce(Gr(name + ".ln.b"), scratch, C, G, 1, 2 * C, C);
        push(O_LNCBX, m);
    }

    void begin(int s) { seg[s][0] = ops.size(); }
    void end(int s) { seg[s][1] = ops.size(); }

    void build(float conf_weight) {
        const long B = c.B, D = c.D, H = c.H, W = c.W, V = c.V(), V1 = c.V1(), V2 = c.V2(),
                   T = c.T(), BT = B * T;
        const long Lc = c.latents, BL = B * Lc, Cn = c.Cn(), Cp = c.Cp(), Hd = c.hidden;
        const int TD = c.TD(), TH = c.TH(), TW = c.TW();
        ops.clear();
        begin(SEG_ZGRAD);
        zero(grad, ptotal);
        end(SEG_ZGRAD);
        begin(SEG_STEM);
        zero(dS, BT * 256);
        zero(P, BT * 256);
        {
            Misc m;
            m.p0 = ct;
            m.p1 = Wf("s0a.w");
            m.p2 = Wf("s0a.b");
            m.q0 = r0a;
            push(O_CONV1F, m);
        }
        lncf(r0a, z0a, st0a, B * V, 32, "s0a");
        {
            Conv k =
                convd(z0a, 32, 1, nullptr, 0, 0, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, "s0b", 32, 32);
            k.bias = Wf("s0b.b");
            k.d0 = r0b;
            push(O_CONV_A, k);
        }
        lncf(r0b, z0b, st0b, B * V, 32, "s0b");
        {
            Conv k = convd(z0b, 32, 1, nullptr, 0, 0, D, H, W, D, H / 2, W / 2, 1, 2, 2, 4, 4, 8,
                "s1a", 32, 64);
            k.bias = Wf("s1a.b");
            k.d0 = r1a;
            push(O_CONV_C, k);
        }
        lncf(r1a, z1a, st1a, B * V1, 64, "s1a");
        {
            Conv k = convd(z1a, 64, 1, nullptr, 0, 0, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1, 1, 4,
                8, 8, "s1b", 64, 64);
            k.bias = Wf("s1b.b");
            k.d0 = r1b;
            push(O_CONV_B, k);
        }
        lncf(r1b, z1b, st1b, B * V1, 64, "s1b");
        {
            Conv k = convd(z1b, 64, 1, nullptr, 0, 0, D, H / 2, W / 2, D / 2, H / 4, W / 4, 2, 2, 2,
                2, 4, 8, "s2a", 64, 128);
            k.bias = Wf("s2a.b");
            k.d0 = r2a;
            push(O_CONV_D, k);
        }
        lncf(r2a, z2a, st2a, B * V2, 128, "s2a");
        {
            Conv k = convd(z2a, 128, 1, nullptr, 0, 0, D / 2, H / 4, W / 4, D / 2, H / 4, W / 4, 1,
                1, 1, 4, 8, 8, "s2b", 128, 128);
            k.bias = Wf("s2b.b");
            k.d0 = r2b;
            push(O_CONV_B, k);
        }
        lncf(r2b, z2b, st2b, B * V2, 128, "s2b");
        {
            Conv k = convd(z2b, 128, 1, nullptr, 0, 0, D / 2, H / 4, W / 4, TD, TH, TW, 2, 2, 2, 2,
                4, 8, "s3", 128, 256);
            k.bias = Wf("s3.b");
            k.f0 = y3;
            push(O_CONV_D, k);
        }
        lnf(y3, BT, "stem.ln", nullptr, S, st3);
        {
            Misc m;
            m.q0 = F;
            push(O_FOURIER, m);
        }
        {
            Epi e;
            e.mode = E_F32_ADD;
            e.f = S;
            e.ld = 256;
            lin(F, BT, 128, "pos", 256, e);
        }
        {
            Misc m;
            m.q0 = S;
            m.p0 = Wf("mask");
            push(O_MASKTOK, m);
        }
        {
            Gemm g;
            g.M = B * Cp;
            g.N = 128;
            g.K = 64;
            g.a.mode = S_CTX1;
            g.a.p = ctx;
            g.a.a = c.C;
            g.b.p = Wb("ctx1.w");
            g.b.ld = 128;
            g.e.mode = E_GELU_Z;
            g.e.h = c1z;
            g.e.h2 = c1a;
            g.e.ld = 128;
            g.e.bias = Wf("ctx1.b");
            gemm(g);
        }
        {
            Gemm g;
            g.M = B * Cn;
            g.N = 256;
            g.K = 2048;
            g.a.mode = S_CTX2;
            g.a.p = c1a;
            g.a.a = c.C;
            g.b.p = Wb("ctx2.w");
            g.b.ld = 256;
            g.e.mode = E_CTX;
            g.e.f = ctxt;
            g.e.ld = 256;
            g.e.a = c.C;
            g.e.bias = Wf("ctx2.b");
            g.e.box = boxes;
            g.e.marker = Wf("marker");
            gemm(g);
        }
        {
            Misc m;
            m.p0 = Wf("latents");
            m.q0 = L0;
            m.n = BL * 256;
            m.m = Lc * 256;
            push(O_LATB, m);
        }
        lnf(L0, BL, "cx.lnq", xlq, nullptr, stq);
        lnf(ctxt, B * Cn, "cx.lnk", xck, nullptr, stk);
        {
            Epi e;
            e.mode = E_BF16;
            e.h = qc;
            e.ld = 256;
            lin(xlq, BL, 256, "cx.q", 256, e);
            Epi f;
            f.mode = E_BF16;
            f.h = kvc;
            f.ld = 512;
            lin(xck, B * Cn, 256, "cx.kv", 512, f);
        }
        Attn actx = attn(1, qc, 256, kvc, kvc + 256, 512, Lc, Cn, oc, lsec, 0);
        push(O_ATTF, actx);
        push(O_ATTM, actx);
        copyf(L0, Lt, BL * 256);
        {
            Epi e;
            e.mode = E_F32_ADD;
            e.f = Lt;
            e.ld = 256;
            lin(oc, BL, 256, "cx.o", 256, e);
        }
        end(SEG_STEM);

        begin(SEG_SAVE);
        {
            Misc m;
            m.q0 = P;
            m.q1 = Lt;
            m.q2 = ck;
            m.n = BT * 256;
            m.m = BL * 256;
            push(O_CKSAVE, m);
        }
        end(SEG_SAVE);
        begin(SEG_REST);
        {
            Misc m;
            m.q0 = P;
            m.q1 = Lt;
            m.q2 = ck;
            m.n = BT * 256;
            m.m = BL * 256;
            push(O_CKREST, m);
        }
        end(SEG_REST);

        begin(SEG_LOOP);
        {
            Misc m;
            m.p0 = P;
            m.p1 = S;
            m.p2 = Wf("loopemb");
            m.p3 = Lt;
            m.q0 = Pin;
            m.q1 = Lin;
            m.n = BT * 256;
            m.m = BL * 256;
            push(O_LOOPIN, m);
        }
        Attn aw[4], ag[4], as_[4];
        for (int k = 0; k < 4; ++k) {
            Blk &b = blk[k];
            std::string p = "b" + std::to_string(k) + ".";
            float *xout = k == 3 ? P : blk[k + 1].xin;
            lnf(b.xin, BT, p + "wn", b.xw, nullptr, b.stw);
            {
                Epi e;
                e.mode = E_BF16;
                e.h = b.qkv;
                e.ld = 768;
                lin(b.xw, BT, 256, p + "qkv", 768, e);
            }
            aw[k] = attn(
                0, b.qkv, 768, b.qkv + 256, b.qkv + 512, 768, 0, 0, b.ow, b.lsew, (k & 1) ? 2 : 0);
            push(O_ATTF, aw[k]);
            copyf(b.xin, b.p1, BT * 256);
            {
                Epi e;
                e.mode = E_F32_ADD;
                e.f = b.p1;
                e.ld = 256;
                lin(b.ow, BT, 256, p + "wo", 256, e);
            }
            lnf(b.lin, BL, p + "lnl", b.xl, nullptr, b.stl);
            lnf(b.p1, BT, p + "lng", b.xg, nullptr, b.stg);
            {
                Epi e;
                e.mode = E_BF16;
                e.h = b.qg;
                e.ld = 256;
                lin(b.xl, BL, 256, p + "gq", 256, e);
                Epi f;
                f.mode = E_BF16;
                f.h = b.kvg;
                f.ld = 512;
                lin(b.xg, BT, 256, p + "gkv", 512, f);
            }
            ag[k] = attn(1, b.qg, 256, b.kvg, b.kvg + 256, 512, Lc, T, b.og, b.lseg, 0);
            push(O_ATTF, ag[k]);
            push(O_ATTM, ag[k]);
            copyf(b.lin, b.l1, BL * 256);
            {
                Epi e;
                e.mode = E_F32_ADD;
                e.f = b.l1;
                e.ld = 256;
                lin(b.og, BL, 256, p + "go", 256, e);
            }
            lnf(b.p1, BT, p + "lnp", b.xp, nullptr, b.stp);
            lnf(b.l1, BL, p + "lns", b.xs, nullptr, b.sts);
            {
                Epi e;
                e.mode = E_BF16;
                e.h = b.qs;
                e.ld = 256;
                lin(b.xp, BT, 256, p + "sq", 256, e);
                Epi f;
                f.mode = E_BF16;
                f.h = b.kvs;
                f.ld = 512;
                lin(b.xs, BL, 256, p + "skv", 512, f);
            }
            as_[k] = attn(1, b.qs, 256, b.kvs, b.kvs + 256, 512, T, Lc, b.os, b.lses, 0);
            push(O_ATTF, as_[k]);
            copyf(b.p1, b.p2, BT * 256);
            {
                Epi e;
                e.mode = E_F32_ADD;
                e.f = b.p2;
                e.ld = 256;
                lin(b.os, BT, 256, p + "so", 256, e);
            }
            lnf(b.p2, BT, p + "lnr", b.xr, nullptr, b.str_);
            {
                Epi e;
                e.mode = E_BF16;
                e.h = b.gates;
                e.ld = 1024;
                lin(b.xr, BT, 256, p + "rg", 1024, e);
            }
            {
                Recur r;
                r.x = b.xr;
                r.gates = b.gates;
                r.lam = Wf(p + "lam");
                r.h = b.hrec;
                r.y = b.yrec;
                r.B = c.B;
                r.TD = TD;
                r.TH = TH;
                r.TW = TW;
                push(O_RECF, r);
            }
            copyf(b.p2, b.p3, BT * 256);
            {
                Epi e;
                e.mode = E_F32_ADD;
                e.f = b.p3;
                e.ld = 256;
                lin(b.yrec, BT, 256, p + "ro", 256, e);
            }
            lnf(b.p3, BT, p + "lnm", b.xm, nullptr, b.stm);
            {
                Epi e;
                e.mode = E_SWIGLU;
                e.h = b.mpre;
                e.ld = 2 * Hd;
                e.h2 = b.hm;
                e.ld2 = Hd;
                lin(b.xm, BT, 256, p + "mi", 2 * Hd, e);
            }
            copyf(b.p3, xout, BT * 256);
            {
                Epi e;
                e.mode = E_F32_ADD;
                e.f = xout;
                e.ld = 256;
                lin(b.hm, BT, Hd, p + "mo", 256, e);
            }
        }
        copyf(blk[3].l1, Lt, BL * 256);
        end(SEG_LOOP);

        begin(SEG_DEC);
        lnf(P, BT, "dec.ln", Pn, nullptr, stdec);
        auto convt = [&](const bf16 *x, int act, long rows, int cin, int taps, int cout,
                         const std::string &name, bf16 *out, int Z, int Y, int X, int fz, int fy,
                         int fx) {
            Gemm g;
            g.M = rows;
            g.N = taps * cout;
            g.K = cin;
            g.a.p = x;
            g.a.ld = cin;
            g.act_a = act;
            g.b.p = Wb(name + ".w");
            g.b.ld = taps * cout;
            g.e.mode = E_CONVT;
            g.e.h = out;
            g.e.a = cout;
            g.e.b = Z;
            g.e.c = Y;
            g.e.d = X;
            g.e.fz = fz;
            g.e.fy = fy;
            g.e.fx = fx;
            g.e.bias = Wf(name + ".b");
            gemm(g);
        };
        convt(Pn, 0, BT, 256, 8, 128, "u2up", u2u, TD, TH, TW, 2, 2, 2);
        {
            Conv k = convd(u2u, 128, 0, z2b, 128, 1, D / 2, H / 4, W / 4, D / 2, H / 4, W / 4, 1, 1,
                1, 4, 8, 8, "u2c", 256, 128);
            k.mask = tokmask;
            k.msh[0] = k.msh[1] = k.msh[2] = 1;
            k.bias = Wf("u2c.b");
            k.d0 = r2c;
            push(O_CONV_B, k);
        }
        lncf(r2c, z2c, st2c, B * V2, 128, "u2c");
        convt(z2c, 1, B * V2, 128, 8, 64, "u1up", u1u, D / 2, H / 4, W / 4, 2, 2, 2);
        {
            Conv k = convd(u1u, 64, 0, z1b, 64, 1, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1, 1, 4, 8,
                8, "u1c", 128, 64);
            k.mask = tokmask;
            k.msh[0] = k.msh[1] = k.msh[2] = 2;
            k.bias = Wf("u1c.b");
            k.d0 = r1c;
            push(O_CONV_B, k);
        }
        lncf(r1c, z1c, st1c, B * V1, 64, "u1c");
        convt(z1c, 1, B * V1, 64, 4, 32, "u0up", u0u, D, H / 2, W / 2, 1, 2, 2);
        {
            Conv k =
                convd(u0u, 32, 0, z0b, 32, 1, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, "u0c", 64, 32);
            k.mask = tokmask;
            k.msh[0] = 2;
            k.msh[1] = k.msh[2] = 3;
            k.bias = Wf("u0c.b");
            k.d0 = r0c;
            push(O_CONV_A, k);
        }
        lncf(r0c, z0c, st0c, B * V, 32, "u0c");
        {
            Misc m;
            m.p0 = z0c;
            m.p1 = Wf("heads.w");
            m.p2 = Wf("heads.b");
            m.q0 = h4;
            m.n = B * V;
            push(O_HEADS, m);
        }
        end(SEG_DEC);

        begin(SEG_LOSS);
        zero(dturn, B * V);
        zero(dres, BT * 256);
        zero(dlat, BL * 256);
        {
            Misc m;
            m.p0 = h4;
            m.q0 = pl;
            m.q1 = pg;
            push(O_PAIRS, m);
        }
        {
            Misc m;
            m.p0 = pg;
            m.q0 = dturn;
            push(O_SCATTER, m);
        }
        {
            Misc m;
            m.p0 = h4;
            m.p1 = dturn;
            m.p2 = z0c;
            m.p3 = Wf("heads.w");
            m.p4 = pl;
            m.q0 = dz0c;
            m.q1 = tokl;
            m.q2 = scratch;
            push(O_VOXLOSS, m);
            reduce(Gr("heads.w"), scratch, 132, G, 1, 140, 0);
            reduce(metrics, scratch, 8, G, 0, 140, 132);
        }
        {
            Misc m;
            m.p0 = P;
            m.p1 = Wf("conf.w");
            m.p2 = Wf("conf.b");
            m.p3 = tokl;
            m.q0 = dres;
            m.q1 = conf;
            m.q2 = scratch;
            m.f0 = conf_weight;
            m.i0 = 1;
            push(O_CONF, m);
            reduce(Gr("conf.w"), scratch, 256, G, 1, 258, 0);
            reduce(metrics + 8, scratch, 1, G, 0, 258, 256);
            reduce(Gr("conf.b"), scratch, 1, G, 1, 258, 257);
        }
        end(SEG_LOSS);

        begin(SEG_DBWD);
        lncb(r0c, st0c, dz0c, B * V, 32, "u0c");
        {
            Conv k = convd(
                dz0c, 32, 0, nullptr, 0, 0, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, "u0c", 64, 32);
            k.dx = 1;
            k.d0 = du0u;
            k.d1 = ds0;
            k.m1 = tokmask;
            k.osh[0] = 2;
            k.osh[1] = k.osh[2] = 3;
            k.split = 32;
            push(O_CONV_B, k);
        }
        convw(u0u, 32, 0, z0b, 32, 1, tokmask, 2, 3, 3, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, dz0c,
            32, "u0c", O_CONVW10);
        auto convt_bwd = [&](const bf16 *dout, long rows_in, int cin, int taps, int cout,
                             const std::string &name, int Z, int Y, int X, int fz, int fy, int fx,
                             const bf16 *xin, int act, int dxmode, bf16 *dzh, const bf16 *zin,
                             float *dzf, long out_rows) {
            Src gs;
            gs.mode = S_CONVT;
            gs.p = dout;
            gs.ld = cout;
            gs.a = Z;
            gs.b = Y;
            gs.c = X;
            gs.d = fz;
            gs.e = fy;
            gs.f = fx;
            Gemm g;
            g.M = rows_in;
            g.N = cin;
            g.K = taps * cout;
            g.a = gs;
            g.b.p = Wb(name + ".w");
            g.b.ld = taps * cout;
            g.bnm = 1;
            g.e.mode = dxmode;
            g.e.h = dzh;
            g.e.z = zin;
            g.e.f = dzf;
            g.e.ld = cin;
            gemm(g);
            Gemm w;
            w.M = cin;
            w.N = taps * cout;
            w.K = rows_in;
            w.a.p = xin;
            w.a.ld = cin;
            w.akm = 1;
            w.act_a = act;
            w.b = gs;
            w.ksplit = ksplit_for(cin, taps * cout, rows_in);
            w.e.mode = E_PART;
            w.e.f = scratch;
            gemm(w);
            reduce(Gr(name + ".w"), scratch, (long)cin * taps * cout, w.ksplit, 1);
            colsum(dout, nullptr, out_rows, cout, Gr(name + ".b"));
        };
        convt_bwd(du0u, B * V1, 64, 4, 32, "u0up", D, H / 2, W / 2, 1, 2, 2, z1c, 1, E_GELU_GRAD,
            dz1c, z1c, nullptr, B * V);
        lncb(r1c, st1c, dz1c, B * V1, 64, "u1c");
        {
            Conv k = convd(dz1c, 64, 0, nullptr, 0, 0, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1, 1, 4,
                8, 8, "u1c", 128, 64);
            k.dx = 1;
            k.d0 = du1u;
            k.d1 = ds1;
            k.m1 = tokmask;
            k.osh[0] = k.osh[1] = k.osh[2] = 2;
            k.split = 64;
            push(O_CONV_B, k);
        }
        convw(u1u, 64, 0, z1b, 64, 1, tokmask, 2, 2, 2, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1, 1,
            4, 8, 8, dz1c, 64, "u1c", O_CONVW10);
        convt_bwd(du1u, B * V2, 128, 8, 64, "u1up", D / 2, H / 4, W / 4, 2, 2, 2, z2c, 1,
            E_GELU_GRAD, dz2c, z2c, nullptr, B * V1);
        lncb(r2c, st2c, dz2c, B * V2, 128, "u2c");
        {
            Conv k = convd(dz2c, 128, 0, nullptr, 0, 0, D / 2, H / 4, W / 4, D / 2, H / 4, W / 4, 1,
                1, 1, 4, 8, 8, "u2c", 256, 128);
            k.dx = 1;
            k.d0 = du2u;
            k.d1 = ds2;
            k.m1 = tokmask;
            k.osh[0] = k.osh[1] = k.osh[2] = 1;
            k.split = 128;
            push(O_CONV_B, k);
        }
        convw(u2u, 128, 0, z2b, 128, 1, tokmask, 1, 1, 1, D / 2, H / 4, W / 4, D / 2, H / 4, W / 4,
            1, 1, 1, 4, 8, 8, dz2c, 128, "u2c", O_CONVW10);
        convt_bwd(du2u, BT, 256, 8, 128, "u2up", TD, TH, TW, 2, 2, 2, Pn, 0, E_F32, nullptr,
            nullptr, dPn, B * V2);
        lnb(P, stdec, dPn, BT, "dec.ln", dres, 1);
        end(SEG_DBWD);

        begin(SEG_LBWD);
        for (int k = 3; k >= 0; --k) {
            Blk &b = blk[k];
            std::string p = "b" + std::to_string(k) + ".";
            cvt(dres, dresb, BT * 256);
            {
                Gemm g;
                g.M = BT;
                g.N = Hd;
                g.K = 256;
                g.a.p = dresb;
                g.a.ld = 256;
                g.b.p = Wb(p + "mo.w");
                g.b.ld = 256;
                g.bnm = 1;
                g.e.mode = E_F32;
                g.e.f = dhm;
                g.e.ld = Hd;
                gemm(g);
            }
            dweight(b.hm, 0, Hd, BT, dresb, 256, Gr(p + "mo.w"));
            colsum(dresb, nullptr, BT, 256, Gr(p + "mo.b"));
            {
                Misc m;
                m.p0 = dhm;
                m.p1 = b.mpre;
                m.q0 = dpre;
                m.n = BT;
                m.i0 = Hd;
                push(O_SWIGLUB, m);
            }
            lin_bwd(b.xm, dpre, BT, 256, 2 * Hd, p + "mi", dxa, E_F32);
            lnb(b.p3, b.stm, dxa, BT, p + "lnm", dres, 1);
            cvt(dres, dresb, BT * 256);
            lin_bwd(b.yrec, dresb, BT, 256, 256, p + "ro", dxb2, E_F32);
            {
                Recur r;
                r.x = b.xr;
                r.gates = b.gates;
                r.lam = Wf(p + "lam");
                r.h = b.hrec;
                r.B = c.B;
                r.TD = TD;
                r.TH = TH;
                r.TW = TW;
                r.dy = dxb2;
                r.dx = dxa;
                r.dg = dgates;
                r.dlam = scratch;
                push(O_RECB, r);
                reduce(Gr(p + "lam"), scratch, 512, B * TH * TW, 1);
            }
            lin_bwd(b.xr, dgates, BT, 256, 1024, p + "rg", dxa, E_F32_ADD);
            lnb(b.p2, b.str_, dxa, BT, p + "lnr", dres, 1);
            cvt(dres, dresb, BT * 256);
            lin_bwd(b.os, dresb, BT, 256, 256, p + "so", dxa, E_F32);
            cvt(dxa, dosb, BT * 256);
            attn_bwd_ops(as_[k], dosb, dqs, 256, dkvs, dkvs + 256, 512);
            cvt(dqs, dqsb, BT * 256);
            cvt(dkvs, dkvsb, BL * 512);
            lin_bwd(b.xp, dqsb, BT, 256, 256, p + "sq", dxa, E_F32);
            lin_bwd(b.xs, dkvsb, BL, 256, 512, p + "skv", dxl_, E_F32);
            lnb(b.p1, b.stp, dxa, BT, p + "lnp", dres, 1);
            lnb(b.l1, b.sts, dxl_, BL, p + "lns", dlat, 1);
            cvt(dlat, dlatb, BL * 256);
            lin_bwd(b.og, dlatb, BL, 256, 256, p + "go", dog, E_F32);
            cvt(dog, dogb, BL * 256);
            attn_bwd_ops(ag[k], dogb, dqg, 256, dkvg, dkvg + 256, 512);
            cvt(dqg, dqgb, BL * 256);
            cvt(dkvg, dkvgb, BT * 512);
            lin_bwd(b.xl, dqgb, BL, 256, 256, p + "gq", dxl_, E_F32);
            lin_bwd(b.xg, dkvgb, BT, 256, 512, p + "gkv", dxa, E_F32);
            lnb(b.lin, b.stl, dxl_, BL, p + "lnl", dlat, 1);
            lnb(b.p1, b.stg, dxa, BT, p + "lng", dres, 1);
            cvt(dres, dresb, BT * 256);
            lin_bwd(b.ow, dresb, BT, 256, 256, p + "wo", dxa, E_F32);
            cvt(dxa, dowb, BT * 256);
            attn_bwd_ops(aw[k], dowb, dqkv, 768, dqkv + 256, dqkv + 512, 768);
            cvt(dqkv, dqkvb, BT * 768);
            lin_bwd(b.xw, dqkvb, BT, 256, 768, p + "qkv", dxa, E_F32);
            lnb(b.xin, b.stw, dxa, BT, p + "wn", dres, 1);
        }
        addf(dres, dS, BT * 256);
        {
            Misc m;
            m.p0 = dres;
            m.p1 = dlat;
            m.n = BT;
            m.m = BL;
            m.q0 = scratch;
            push(O_EMBGRAD, m);
            reduce(Gr("loopemb"), scratch, 256, G, 1, 0, 0, 256);
        }
        end(SEG_LBWD);

        begin(SEG_CTXB);
        cvt(dlat, dlatb, BL * 256);
        lin_bwd(oc, dlatb, BL, 256, 256, "cx.o", doc, E_F32);
        cvt(doc, docb, BL * 256);
        attn_bwd_ops(actx, docb, dqc, 256, dkvc, dkvc + 256, 512);
        cvt(dqc, dqcb, BL * 256);
        cvt(dkvc, dkvcb, B * Cn * 512);
        lin_bwd(xlq, dqcb, BL, 256, 256, "cx.q", dxl_, E_F32);
        lin_bwd(xck, dkvcb, B * Cn, 256, 512, "cx.kv", dctx, E_F32);
        lnb(L0, stq, dxl_, BL, "cx.lnq", dlat, 1);
        {
            Misc m;
            m.p0 = dlat;
            m.q0 = Gr("latents");
            m.m = Lc * 256;
            m.i0 = c.B;
            push(O_LATBG, m);
        }
        copyf(dctx, dl0, B * Cn * 256);
        lnb(ctxt, stk, dl0, B * Cn, "cx.lnk", dctx, 0);
        {
            Misc m;
            m.p0 = dctx;
            m.p1 = boxes;
            m.q0 = scratch;
            m.i0 = c.C;
            m.n = B * Cn;
            push(O_MARKERG, m);
            reduce(Gr("marker"), scratch, 256, G, 1);
        }
        cvt(dctx, dctxb, B * Cn * 256);
        {
            Src s;
            s.mode = S_CTX2;
            s.p = c1a;
            s.a = c.C;
            Gemm w;
            w.M = 2048;
            w.N = 256;
            w.K = B * Cn;
            w.a = s;
            w.akm = 1;
            w.b.p = dctxb;
            w.b.ld = 256;
            w.ksplit = ksplit_for(2048, 256, B * Cn);
            w.e.mode = E_PART;
            w.e.f = scratch;
            gemm(w);
            reduce(Gr("ctx2.w"), scratch, 2048L * 256, w.ksplit, 1);
            colsum(dctxb, nullptr, B * Cn, 256, Gr("ctx2.b"));
            Gemm g;
            g.M = B * Cn;
            g.N = 2048;
            g.K = 256;
            g.a.p = dctxb;
            g.a.ld = 256;
            g.b.p = Wb("ctx2.w");
            g.b.ld = 256;
            g.bnm = 1;
            g.e.mode = E_CTX_DX;
            g.e.h = dc1z;
            g.e.z = c1z;
            g.e.a = c.C;
            gemm(g);
            Src s1;
            s1.mode = S_CTX1;
            s1.p = ctx;
            s1.a = c.C;
            Gemm w1;
            w1.M = 64;
            w1.N = 128;
            w1.K = B * Cp;
            w1.a = s1;
            w1.akm = 1;
            w1.b.p = dc1z;
            w1.b.ld = 128;
            w1.ksplit = ksplit_for(64, 128, B * Cp);
            w1.e.mode = E_PART;
            w1.e.f = scratch;
            gemm(w1);
            reduce(Gr("ctx1.w"), scratch, 64L * 128, w1.ksplit, 1);
            colsum(dc1z, nullptr, B * Cp, 128, Gr("ctx1.b"));
        }
        end(SEG_CTXB);

        begin(SEG_SBWD);
        {
            Misc m;
            m.q0 = dS;
            m.q1 = scratch;
            push(O_MASKGRAD, m);
            reduce(Gr("mask"), scratch, 256, G, 1);
        }
        cvt(dS, dSb, BT * 256);
        dweight(F, 0, 128, BT, dSb, 256, Gr("pos.w"));
        colsum(dSb, nullptr, BT, 256, Gr("pos.b"));
        lnb(y3, st3, dS, BT, "stem.ln", dy3, 0);
        cvt(dy3, dy3b, BT * 256);
        {
            Conv k = convd(dy3b, 256, 0, nullptr, 0, 0, TD, TH, TW, D / 2, H / 4, W / 4, 2, 2, 2, 4,
                8, 8, "s3", 128, 256);
            k.dx = 1;
            k.d0 = ds2;
            k.acc0 = 1;
            k.z0 = z2b;
            push(O_CONV_B, k);
        }
        convw(z2b, 128, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D / 2, H / 4, W / 4, TD, TH, TW, 2, 2,
            2, 2, 4, 8, dy3b, 256, "s3", O_CONVW12);
        lncb(r2b, st2b, ds2, B * V2, 128, "s2b");
        {
            Conv k = convd(ds2, 128, 0, nullptr, 0, 0, D / 2, H / 4, W / 4, D / 2, H / 4, W / 4, 1,
                1, 1, 4, 8, 8, "s2b", 128, 128);
            k.dx = 1;
            k.d0 = dz2a;
            k.z0 = z2a;
            push(O_CONV_B, k);
        }
        convw(z2a, 128, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D / 2, H / 4, W / 4, D / 2, H / 4,
            W / 4, 1, 1, 1, 4, 8, 8, ds2, 128, "s2b", O_CONVW10);
        lncb(r2a, st2a, dz2a, B * V2, 128, "s2a");
        {
            Conv k = convd(dz2a, 128, 0, nullptr, 0, 0, D / 2, H / 4, W / 4, D, H / 2, W / 2, 2, 2,
                2, 4, 8, 8, "s2a", 64, 128);
            k.dx = 1;
            k.d0 = ds1;
            k.acc0 = 1;
            k.z0 = z1b;
            push(O_CONV_B, k);
        }
        convw(z1b, 64, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D, H / 2, W / 2, D / 2, H / 4, W / 4, 2,
            2, 2, 2, 4, 8, dz2a, 128, "s2a", O_CONVW12);
        lncb(r1b, st1b, ds1, B * V1, 64, "s1b");
        {
            Conv k = convd(ds1, 64, 0, nullptr, 0, 0, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1, 1, 4,
                8, 8, "s1b", 64, 64);
            k.dx = 1;
            k.d0 = dz1a;
            k.z0 = z1a;
            push(O_CONV_B, k);
        }
        convw(z1a, 64, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D, H / 2, W / 2, D, H / 2, W / 2, 1, 1,
            1, 4, 8, 8, ds1, 64, "s1b", O_CONVW10);
        lncb(r1a, st1a, dz1a, B * V1, 64, "s1a");
        {
            Conv k = convd(dz1a, 64, 0, nullptr, 0, 0, D, H / 2, W / 2, D, H, W, 1, 2, 2, 4, 8, 8,
                "s1a", 32, 64);
            k.dx = 1;
            k.d0 = ds0;
            k.acc0 = 1;
            k.z0 = z0b;
            push(O_CONV_A, k);
        }
        convw(z0b, 32, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D, H, W, D, H / 2, W / 2, 1, 2, 2, 4, 4,
            8, dz1a, 64, "s1a", O_CONVW15);
        lncb(r0b, st0b, ds0, B * V, 32, "s0b");
        {
            Conv k =
                convd(ds0, 32, 0, nullptr, 0, 0, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, "s0b", 32, 32);
            k.dx = 1;
            k.d0 = dz0a;
            k.z0 = z0a;
            push(O_CONV_A, k);
        }
        convw(z0a, 32, 1, nullptr, 0, 0, nullptr, 0, 0, 0, D, H, W, D, H, W, 1, 1, 1, 4, 8, 8, ds0,
            32, "s0b", O_CONVW10);
        lncb(r0a, st0a, dz0a, B * V, 32, "s0a");
        {
            Misc m;
            m.p0 = ct;
            m.p1 = dz0a;
            m.q0 = scratch;
            push(O_CONV1W, m);
            reduce(Gr("s0a.w"), scratch, 896, G, 1);
        }
        end(SEG_SBWD);

        begin(SEG_OPT);
        {
            Misc m;
            m.p0 = grad;
            m.n = ptotal;
            m.q0 = scratch;
            push(O_OPTNORM, m);
            Misc s;
            s.p0 = grad;
            s.p1 = scratch;
            s.p2 = flags;
            s.q0 = master;
            s.q1 = m1;
            s.q2 = m2;
            s.q3 = wb;
            s.n = ptotal;
            s.f1 = 1.f;
            s.f2 = 0.05f;
            push(O_OPTSTEP, s);
        }
        end(SEG_OPT);
        if (dops)
            cudaFree(dops);
        CK(cudaMalloc(&dops, ops.size() * sizeof(Op)));
        CK(cudaMemcpy(dops, ops.data(), ops.size() * sizeof(Op), cudaMemcpyHostToDevice));
    }
};

struct HostBatch {
    std::vector<bf16> ct, ctt, ctx;
    std::vector<uint8_t> lab, tokmask;
    std::vector<float> centre, boxes;
    std::vector<Meta> meta;
    std::vector<Pair> pairs;
    std::vector<int> ptok, csr_v, csr_o, csr_e;
    std::vector<float> csr_w;
    void resize(const Cfg &c, int maxpairs, int maxcsr, int maxent) {
        ct.assign(c.B * c.V(), __float2bfloat16(0.f));
        ctt = ct;
        ctx.assign((long)c.B * c.C * c.C * 4, __float2bfloat16(0.f));
        lab.assign(c.B * c.V(), 0);
        tokmask.assign(c.B * c.T(), 0);
        centre.assign((long)c.B * c.D * 2, 0.f);
        boxes.assign(c.B * 4, 0.f);
        meta.assign(c.B, Meta{});
        pairs.assign((long)c.B * maxpairs, Pair{});
        ptok.assign(c.B * (c.T() + 1), 0);
        csr_v.assign((long)c.B * maxcsr, 0);
        csr_o.assign((long)c.B * (maxcsr + 1), 0);
        csr_e.assign((long)c.B * maxent, 0);
        csr_w.assign((long)c.B * maxent, 0.f);
    }
};

static int token_of(const Cfg &c, const float *p) {
    int z = std::min(c.D - 1, std::max(0, (int)p[0])),
        y = std::min(c.H - 1, std::max(0, (int)p[1])),
        x = std::min(c.W - 1, std::max(0, (int)p[2]));
    return ((z / 4) * c.TH() + y / 8) * c.TW() + x / 8;
}

static void corners(const Cfg &c, const float *p, long *idx, float *w) {
    float z = std::min(std::max(p[0], 0.f), c.D - 1.f),
          y = std::min(std::max(p[1], 0.f), c.H - 1.f),
          x = std::min(std::max(p[2], 0.f), c.W - 1.f);
    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    int z1 = std::min(z0 + 1, c.D - 1), y1 = std::min(y0 + 1, c.H - 1),
        x1 = std::min(x0 + 1, c.W - 1);
    float fz = z - z0, fy = y - y0, fx = x - x0;
    for (int k = 0; k < 8; ++k) {
        int zz = k & 4 ? z1 : z0, yy = k & 2 ? y1 : y0, xx = k & 1 ? x1 : x0;
        idx[k] = ((long)zz * c.H + yy) * c.W + xx;
        w[k] = (k & 4 ? fz : 1 - fz) * (k & 2 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx);
    }
}

static void pack_pairs(const Cfg &c, HostBatch &hb, int b, std::vector<Pair> &ps, int maxpairs,
    int maxcsr, int maxent) {
    if ((int)ps.size() > maxpairs)
        ps.resize(maxpairs);
    for (auto &p : ps)
        p.token = token_of(c, p.a);
    std::stable_sort(
        ps.begin(), ps.end(), [](const Pair &x, const Pair &y) { return x.token < y.token; });
    const long T = c.T();
    int *pt = hb.ptok.data() + b * (T + 1);
    for (long t = 0, k = 0; t <= T; ++t) {
        while (k < (long)ps.size() && ps[k].token < t)
            ++k;
        pt[t] = k;
    }
    std::copy(ps.begin(), ps.end(), hb.pairs.begin() + (long)b * maxpairs);
    struct Ent {
        long v;
        int e;
        float w;
    };
    std::vector<Ent> ents;
    ents.reserve(ps.size() * 16);
    for (size_t i = 0; i < ps.size(); ++i)
        for (int side = 0; side < (ps[i].kind == 3 ? 1 : 2); ++side) {
            long idx[8];
            float w[8];
            corners(c, side ? ps[i].b : ps[i].a, idx, w);
            for (int k = 0; k < 8; ++k)
                if (w[k] != 0.f)
                    ents.push_back({idx[k], (int)(2 * i + side), w[k]});
        }
    std::stable_sort(
        ents.begin(), ents.end(), [](const Ent &x, const Ent &y) { return x.v < y.v; });
    int nv = 0;
    int *cv = hb.csr_v.data() + (long)b * maxcsr, *co = hb.csr_o.data() + (long)b * (maxcsr + 1);
    int *ce = hb.csr_e.data() + (long)b * maxent;
    float *cw = hb.csr_w.data() + (long)b * maxent;
    for (size_t k = 0; k < ents.size() && (long)k < maxent; ++k) {
        if (k == 0 || ents[k].v != ents[k - 1].v) {
            if (nv >= maxcsr)
                break;
            cv[nv] = ents[k].v;
            co[nv] = k;
            ++nv;
        }
        ce[k] = ents[k].e;
        cw[k] = ents[k].w;
    }
    co[nv] = std::min<long>(ents.size(), maxent);
    hb.meta[b].npairs = ps.size();
    hb.meta[b].ncsr = nv;
}

static void upload_batch(Model &m, const HostBatch &hb, cudaStream_t s) {
    auto up = [&](void *d, const void *h, size_t bytes) {
        CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s));
    };
    up(m.ct, hb.ct.data(), hb.ct.size() * 2);
    up(m.ctt, hb.ctt.data(), hb.ctt.size() * 2);
    up(m.ctx, hb.ctx.data(), hb.ctx.size() * 2);
    up(m.lab, hb.lab.data(), hb.lab.size());
    up(m.tokmask, hb.tokmask.data(), hb.tokmask.size());
    up(m.centre, hb.centre.data(), hb.centre.size() * 4);
    up(m.boxes, hb.boxes.data(), hb.boxes.size() * 4);
    up(m.meta, hb.meta.data(), hb.meta.size() * sizeof(Meta));
    up(m.pairs, hb.pairs.data(), hb.pairs.size() * sizeof(Pair));
    up(m.ptok, hb.ptok.data(), hb.ptok.size() * 4);
    up(m.csr_v, hb.csr_v.data(), hb.csr_v.size() * 4);
    up(m.csr_o, hb.csr_o.data(), hb.csr_o.size() * 4);
    up(m.csr_e, hb.csr_e.data(), hb.csr_e.size() * 4);
    up(m.csr_w, hb.csr_w.data(), hb.csr_w.size() * 4);
}

static StepIO step_io(Model &m) {
    StepIO io;
    io.ct = m.ct;
    io.ctt = m.ctt;
    io.lab = m.lab;
    io.tokmask = m.tokmask;
    io.centre = m.centre;
    io.meta = m.meta;
    io.pairs = m.pairs;
    io.ptok = m.ptok;
    io.csr_v = m.csr_v;
    io.csr_o = m.csr_o;
    io.csr_e = m.csr_e;
    io.csr_w = m.csr_w;
    io.metrics = m.metrics;
    io.scal = m.scal;
    io.B = m.c.B;
    io.D = m.c.D;
    io.H = m.c.H;
    io.W = m.c.W;
    io.T = m.c.T();
    io.maxpairs = m.maxpairs;
    io.maxcsr = m.maxcsr;
    io.maxent = m.maxent;
    return io;
}

static void launch_step(Model &m, int L, int train, int update, cudaStream_t s,
    unsigned long long *clocks, int zero_grad = 1) {
    Plan p;
    p.ops = m.dops;
    for (int i = 0; i < 16; ++i) {
        p.seg[i][0] = m.seg[i][0];
        p.seg[i][1] = m.seg[i][1];
    }
    p.L = L;
    p.train = train;
    p.update = update;
    p.retain = m.c.retain;
    p.zero_grad = zero_grad;
    StepIO io = step_io(m);
    void *args[] = {&p, &io, &clocks};
    CK(cudaLaunchCooperativeKernel((void *)k_step, dim3(m.G), dim3(Threads), args, SmemBytes, s));
}

__global__ void __launch_bounds__(Threads, 1) k_gemm(Gemm g) {
    extern __shared__ __align__(128) unsigned char sm[];
    gemm_run(g, sm);
}
__global__ void __launch_bounds__(Threads, 1)
    k_reduce(float *d, const float *p, long n, int s, int acc) {
    reduce_parts(d, p, n, s, acc);
}
template <int WGM, int MT, int NT, int PF>
__global__ void __launch_bounds__(Threads, 1) k_conv(Conv c) {
    extern __shared__ __align__(128) unsigned char sm[];
    conv_run<WGM, MT, NT, PF>(c, sm);
}
template <int PF> __global__ void __launch_bounds__(Threads, 1) k_convw(ConvW w) {
    extern __shared__ __align__(128) unsigned char sm[];
    convw_run<PF>(w, sm);
}
__global__ void __launch_bounds__(Threads, 1)
    k_conv1(const bf16 *ct, int B, int Z, int Y, int X, const float *w, const float *b, bf16 *z) {
    extern __shared__ __align__(128) unsigned char sm[];
    conv1_fwd(ct, B, Z, Y, X, w, b, z, sm);
}
__global__ void __launch_bounds__(Threads, 1)
    k_conv1w(const bf16 *ct, const bf16 *dz, int B, int Z, int Y, int X, float *part) {
    extern __shared__ __align__(128) unsigned char sm[];
    conv1_dw(ct, dz, B, Z, Y, X, part, sm);
}
__global__ void __launch_bounds__(Threads, 1) k_attn(Attn a, int stage, const bf16 *o, int rows) {
    extern __shared__ __align__(128) unsigned char sm[];
    if (stage == 0)
        attn_fwd(a, sm);
    if (stage == 1)
        attn_merge(a);
    if (stage == 2)
        attn_dsum(a, o, rows);
    if (stage == 3)
        attn_bwd(a, sm);
    if (stage == 4)
        attn_bwd_reduce(a);
}
__global__ void __launch_bounds__(Threads, 1)
    k_ln(int stage, const float *x, int rows, const float *g, const float *b, bf16 *yh, float *yf,
        float *stat, const float *dy, float *dx, float *part) {
    extern __shared__ __align__(128) unsigned char sm[];
    if (stage == 0)
        ln_fwd(x, rows, g, b, yh, yf, stat);
    else
        ln_bwd(x, stat, dy, rows, g, dx, 0, part, sm);
}
__global__ void __launch_bounds__(Threads, 1) k_recur(Recur r, int stage) {
    if (stage == 0)
        recur_fwd(r);
    else
        recur_bwd(r);
}

template <class T> struct Dev {
    T *p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        CK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T)));
    }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T> &v) {
        CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

static float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }
static std::vector<bf16> tobf(const std::vector<float> &v) {
    std::vector<bf16> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = __float2bfloat16(v[i]);
    return o;
}
static std::vector<float> fromb(const std::vector<bf16> &v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = __bfloat162float(v[i]);
    return o;
}
static std::vector<float> rnd(size_t n, Rng &r, float s = 1.f) {
    std::vector<float> v(n);
    for (auto &x : v)
        x = rb(r.sym() * s);
    return v;
}
static double nrms(const std::vector<float> &a, const std::vector<double> &b) {
    double e = 0, s = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        e += (a[i] - b[i]) * (a[i] - b[i]);
        s += b[i] * b[i];
    }
    return std::sqrt(e / std::max(s, 1e-30));
}
static int report(const char *name, double err, double tol) {
    printf("unit %-28s nrms=%.3e tol=%.1e %s\n", name, err, tol, err <= tol ? "pass" : "FAIL");
    return err <= tol ? 0 : 1;
}
static int G = 58;
static void launch_setup() {
    static bool done = false;
    if (done)
        return;
    done = true;
    G = device_coop_blocks();
    CK(cudaFuncSetAttribute(k_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(
        k_conv<8, 2, 4, 10>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(
        k_conv<4, 4, 4, 10>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(
        k_conv<4, 2, 4, 15>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(
        k_conv<2, 2, 2, 12>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_convw<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_convw<15>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_convw<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_conv1, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_conv1w, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    CK(cudaFuncSetAttribute(k_ln, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
}

static int test_gemm() {
    Rng r(1);
    int fails = 0;
    for (int variant = 0; variant < 4; ++variant) {
        int M = 1000, N = variant == 3 ? 136 : 256, K = 200;
        auto A = rnd((size_t)M * K, r), B = rnd((size_t)K * N, r);
        std::vector<double> ref((size_t)M * N, 0.0);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                for (int n = 0; n < N; ++n)
                    ref[(size_t)m * N + n] += (double)A[(size_t)m * K + k] * B[(size_t)k * N + n];
        std::vector<float> At(A.size()), Bt(B.size());
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                At[(size_t)k * M + m] = A[(size_t)m * K + k];
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                Bt[(size_t)n * K + k] = B[(size_t)k * N + n];
        Dev<bf16> dA(A.size()), dB(B.size());
        Dev<float> dC((size_t)M * N), part((size_t)M * N * 4);
        Gemm g;
        g.M = M;
        g.N = N;
        g.K = K;
        g.akm = variant == 1;
        g.bnm = variant == 2;
        dA.put(tobf(g.akm ? At : A));
        dB.put(tobf(g.bnm ? Bt : B));
        g.a.p = dA.p;
        g.a.ld = g.akm ? M : K;
        g.b.p = dB.p;
        g.b.ld = g.bnm ? K : N;
        g.e.mode = E_F32;
        g.e.f = dC.p;
        g.e.ld = N;
        if (variant == 3) {
            g.ksplit = 3;
            g.e.mode = E_PART;
            g.e.f = part.p;
        }
        k_gemm<<<G, Threads, SmemBytes>>>(g);
        if (variant == 3)
            k_reduce<<<G, Threads>>>(dC.p, part.p, (long)M * N, 3, 0);
        CK(cudaDeviceSynchronize());
        char name[64];
        snprintf(name, sizeof(name), "gemm variant %d", variant);
        fails += report(name, nrms(dC.get(), ref), 1e-5);
    }
    return fails;
}

struct CpuConv {
    int B, SZ, SY, SX, OZ, OY, OX, sz, sy, sx, cin, cout;
};
static std::vector<double> cpu_conv(
    const CpuConv &c, const std::vector<float> &x, const std::vector<float> &w) {
    std::vector<double> y((size_t)c.B * c.OZ * c.OY * c.OX * c.cout, 0.0);
    for (int b = 0; b < c.B; ++b)
        for (int z = 0; z < c.OZ; ++z)
            for (int yy = 0; yy < c.OY; ++yy)
                for (int xx = 0; xx < c.OX; ++xx) {
                    double *o = &y[((((size_t)b * c.OZ + z) * c.OY + yy) * c.OX + xx) * c.cout];
                    for (int t = 0; t < 27; ++t) {
                        int iz = z * c.sz - 1 + t / 9, iy = yy * c.sy - 1 + (t / 3) % 3,
                            ix = xx * c.sx - 1 + t % 3;
                        if (iz < 0 || iy < 0 || ix < 0 || iz >= c.SZ || iy >= c.SY || ix >= c.SX)
                            continue;
                        const float *in =
                            &x[((((size_t)b * c.SZ + iz) * c.SY + iy) * c.SX + ix) * c.cin];
                        for (int ci = 0; ci < c.cin; ++ci)
                            for (int co = 0; co < c.cout; ++co)
                                o[co] += (double)in[ci] * w[((size_t)t * c.cin + ci) * c.cout + co];
                    }
                }
    return y;
}

template <int WGM, int MT, int NT, int PF>
static int test_conv_case(const char *name, int SZ, int SY, int SX, int st[3], int cin, int cout,
    int TZ, int TY, int TX, int dxTZ, int dxTY, int dxTX) {
    Rng r(7 + cin + cout + st[1]);
    int B = 2, OZ = SZ / st[0], OY = SY / st[1], OX = SX / st[2];
    CpuConv cc{B, SZ, SY, SX, OZ, OY, OX, st[0], st[1], st[2], cin, cout};
    auto x = rnd((size_t)B * SZ * SY * SX * cin, r), w = rnd(27L * cin * cout, r, 0.2f);
    auto y = cpu_conv(cc, x, w);
    Dev<bf16> dx_(x.size()), dw(w.size()), dy_(y.size()), out(y.size()), gx(x.size());
    dx_.put(tobf(x));
    dw.put(tobf(w));
    Conv c;
    c.s0 = dx_.p;
    c.c0 = cin;
    c.B = B;
    c.SZ = SZ;
    c.SY = SY;
    c.SX = SX;
    c.OZ = OZ;
    c.OY = OY;
    c.OX = OX;
    c.sz = st[0];
    c.sy = st[1];
    c.sx = st[2];
    c.TZ = TZ;
    c.TY = TY;
    c.TX = TX;
    c.w = dw.p;
    c.cin = cin;
    c.cout = cout;
    c.d0 = out.p;
    k_conv<WGM, MT, NT, PF><<<G, Threads, SmemBytes>>>(c);
    CK(cudaDeviceSynchronize());
    int fails = 0;
    char nm[96];
    snprintf(nm, sizeof(nm), "%s forward", name);
    fails += report(nm, nrms(fromb(out.get()), y), 1e-2);
    auto g = rnd(y.size(), r);
    dy_.put(tobf(g));
    Conv d = c;
    d.dx = 1;
    d.s0 = dy_.p;
    d.c0 = cout;
    d.SZ = OZ;
    d.SY = OY;
    d.SX = OX;
    d.OZ = SZ;
    d.OY = SY;
    d.OX = SX;
    d.TZ = dxTZ;
    d.TY = dxTY;
    d.TX = dxTX;
    d.d0 = gx.p;
    if (cin >= 64)
        k_conv<4, 4, 4, 10><<<G, Threads, SmemBytes>>>(d);
    else
        k_conv<8, 2, 4, 10><<<G, Threads, SmemBytes>>>(d);
    CK(cudaDeviceSynchronize());
    auto gxv = fromb(gx.get());
    double lhs = 0, rhs = 0, sc = 0;
    for (size_t i = 0; i < y.size(); ++i)
        lhs += (double)g[i] * y[i];
    for (size_t i = 0; i < x.size(); ++i) {
        rhs += (double)gxv[i] * x[i];
        sc += std::fabs((double)gxv[i] * x[i]);
    }
    snprintf(nm, sizeof(nm), "%s data gradient", name);
    fails += report(nm, std::fabs(lhs - rhs) / std::max(sc, 1e-30) * 10, 1e-2);
    std::vector<double> gxref(x.size(), 0.0);
    for (int b = 0; b < B; ++b)
        for (int z = 0; z < OZ; ++z)
            for (int yy = 0; yy < OY; ++yy)
                for (int xx = 0; xx < OX; ++xx) {
                    const float *gy = &g[((((size_t)b * OZ + z) * OY + yy) * OX + xx) * cout];
                    for (int t = 0; t < 27; ++t) {
                        int iz = z * st[0] - 1 + t / 9, iy = yy * st[1] - 1 + (t / 3) % 3,
                            ix = xx * st[2] - 1 + t % 3;
                        if (iz < 0 || iy < 0 || ix < 0 || iz >= SZ || iy >= SY || ix >= SX)
                            continue;
                        double *o = &gxref[((((size_t)b * SZ + iz) * SY + iy) * SX + ix) * cin];
                        for (int ci = 0; ci < cin; ++ci)
                            for (int co = 0; co < cout; ++co)
                                o[ci] += (double)gy[co] * w[((size_t)t * cin + ci) * cout + co];
                    }
                }
    snprintf(nm, sizeof(nm), "%s data gradient exact", name);
    fails += report(nm, nrms(gxv, gxref), 1e-2);
    int splits = 3;
    Dev<float> part((size_t)splits * w.size()), bpart((size_t)splits * cout), gw(w.size()),
        gb(cout);
    ConvW cw;
    cw.s0 = dx_.p;
    cw.c0 = cin;
    cw.B = B;
    cw.SZ = SZ;
    cw.SY = SY;
    cw.SX = SX;
    cw.OZ = OZ;
    cw.OY = OY;
    cw.OX = OX;
    cw.sz = st[0];
    cw.sy = st[1];
    cw.sx = st[2];
    cw.TZ = TZ;
    cw.TY = TY;
    cw.TX = TX;
    cw.dy = dy_.p;
    cw.cout = cout;
    cw.part = part.p;
    cw.bpart = bpart.p;
    cw.splits = splits;
    k_convw<PF><<<G, Threads, SmemBytes>>>(cw);
    k_reduce<<<G, Threads>>>(gw.p, part.p, (long)w.size(), splits, 0);
    k_reduce<<<G, Threads>>>(gb.p, bpart.p, cout, splits, 0);
    CK(cudaDeviceSynchronize());
    std::vector<double> gwref(w.size(), 0.0), gbref(cout, 0.0);
    for (int b = 0; b < B; ++b)
        for (int z = 0; z < OZ; ++z)
            for (int yy = 0; yy < OY; ++yy)
                for (int xx = 0; xx < OX; ++xx) {
                    const float *gy = &g[((((size_t)b * OZ + z) * OY + yy) * OX + xx) * cout];
                    for (int co = 0; co < cout; ++co)
                        gbref[co] += gy[co];
                    for (int t = 0; t < 27; ++t) {
                        int iz = z * st[0] - 1 + t / 9, iy = yy * st[1] - 1 + (t / 3) % 3,
                            ix = xx * st[2] - 1 + t % 3;
                        if (iz < 0 || iy < 0 || ix < 0 || iz >= SZ || iy >= SY || ix >= SX)
                            continue;
                        const float *in = &x[((((size_t)b * SZ + iz) * SY + iy) * SX + ix) * cin];
                        for (int ci = 0; ci < cin; ++ci)
                            for (int co = 0; co < cout; ++co)
                                gwref[((size_t)t * cin + ci) * cout + co] +=
                                    (double)in[ci] * gy[co];
                    }
                }
    snprintf(nm, sizeof(nm), "%s weight gradient", name);
    fails += report(nm, nrms(gw.get(), gwref), 1e-4);
    snprintf(nm, sizeof(nm), "%s bias gradient", name);
    fails += report(nm, nrms(gb.get(), gbref), 1e-4);
    return fails;
}

static int test_conv1() {
    Rng r(3);
    int B = 2, Z = 8, Y = 16, X = 16;
    auto x = rnd((size_t)B * Z * Y * X, r), w = rnd(27 * 32, r, 0.3f), b = rnd(32, r, 0.1f);
    CpuConv cc{B, Z, Y, X, Z, Y, X, 1, 1, 1, 1, 32};
    auto y = cpu_conv(cc, x, w);
    for (size_t i = 0; i < y.size(); ++i)
        y[i] += b[i % 32];
    Dev<bf16> dx_(x.size()), z(y.size()), dz(y.size());
    Dev<float> dw(w.size()), db(32), part((size_t)G * 896), gw(896);
    dx_.put(tobf(x));
    dw.put(w);
    db.put(b);
    k_conv1<<<G, Threads, SmemBytes>>>(dx_.p, B, Z, Y, X, dw.p, db.p, z.p);
    CK(cudaDeviceSynchronize());
    int fails = report("conv1 forward", nrms(fromb(z.get()), y), 1e-2);
    auto g = rnd(y.size(), r);
    dz.put(tobf(g));
    k_conv1w<<<G, Threads, SmemBytes>>>(dx_.p, dz.p, B, Z, Y, X, part.p);
    k_reduce<<<G, Threads>>>(gw.p, part.p, 896, G, 0);
    CK(cudaDeviceSynchronize());
    std::vector<double> ref(896, 0.0);
    for (int bb = 0; bb < B; ++bb)
        for (int zz = 0; zz < Z; ++zz)
            for (int yy = 0; yy < Y; ++yy)
                for (int xx = 0; xx < X; ++xx) {
                    const float *gy = &g[((((size_t)bb * Z + zz) * Y + yy) * X + xx) * 32];
                    for (int co = 0; co < 32; ++co)
                        ref[864 + co] += gy[co];
                    for (int t = 0; t < 27; ++t) {
                        int iz = zz - 1 + t / 9, iy = yy - 1 + (t / 3) % 3, ix = xx - 1 + t % 3;
                        if (iz < 0 || iy < 0 || ix < 0 || iz >= Z || iy >= Y || ix >= X)
                            continue;
                        double v = x[(((size_t)bb * Z + iz) * Y + iy) * X + ix];
                        for (int co = 0; co < 32; ++co)
                            ref[t * 32 + co] += v * gy[co];
                    }
                }
    fails += report("conv1 weight gradient", nrms(gw.get(), ref), 1e-4);
    return fails;
}

static void cpu_attn(int nq, int nk, const std::vector<float> &Q, const std::vector<float> &K,
    const std::vector<float> &V, const std::vector<float> &dO, std::vector<double> &O,
    std::vector<double> &dQ, std::vector<double> &dK, std::vector<double> &dV, int head, long ldq,
    long ldk, long qoff, long koff, long voff) {
    const double sc = 1.0 / std::sqrt(32.0);
    for (int i = 0; i < nq; ++i) {
        std::vector<double> s(nk), p(nk);
        double m = -1e300;
        for (int j = 0; j < nk; ++j) {
            double v = 0;
            for (int d = 0; d < 32; ++d)
                v += (double)Q[i * ldq + qoff + head * 32 + d] * K[j * ldk + koff + head * 32 + d];
            s[j] = v * sc;
            m = std::max(m, s[j]);
        }
        double l = 0;
        for (int j = 0; j < nk; ++j)
            l += (p[j] = std::exp(s[j] - m));
        for (int j = 0; j < nk; ++j)
            p[j] /= l;
        double o[32] = {0};
        for (int j = 0; j < nk; ++j)
            for (int d = 0; d < 32; ++d)
                o[d] += p[j] * V[j * ldk + voff + head * 32 + d];
        for (int d = 0; d < 32; ++d)
            O[(size_t)i * 256 + head * 32 + d] = o[d];
        double D = 0;
        for (int d = 0; d < 32; ++d)
            D += dO[(size_t)i * 256 + head * 32 + d] * o[d];
        for (int j = 0; j < nk; ++j) {
            double dp = 0;
            for (int d = 0; d < 32; ++d) {
                dp += dO[(size_t)i * 256 + head * 32 + d] * V[j * ldk + voff + head * 32 + d];
                dV[(size_t)j * 256 + head * 32 + d] += p[j] * dO[(size_t)i * 256 + head * 32 + d];
            }
            double ds = p[j] * (dp - D) * sc;
            for (int d = 0; d < 32; ++d) {
                dQ[(size_t)i * 256 + head * 32 + d] += ds * K[j * ldk + koff + head * 32 + d];
                dK[(size_t)j * 256 + head * 32 + d] += ds * Q[i * ldq + qoff + head * 32 + d];
            }
        }
    }
}

static int test_attn_dense(int nq, int nk) {
    Rng r(11 + nq + nk);
    int B = 2;
    auto Q = rnd((size_t)B * nq * 256, r), KV = rnd((size_t)B * nk * 512, r),
         dO = rnd((size_t)B * nq * 256, r);
    std::vector<double> O((size_t)B * nq * 256), dQ(O.size(), 0.0), dK((size_t)B * nk * 256, 0.0),
        dV(dK.size(), 0.0);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < 8; ++h) {
            std::vector<float> q(
                Q.begin() + (size_t)b * nq * 256, Q.begin() + (size_t)(b + 1) * nq * 256);
            std::vector<float> kv(
                KV.begin() + (size_t)b * nk * 512, KV.begin() + (size_t)(b + 1) * nk * 512);
            std::vector<float> d(
                dO.begin() + (size_t)b * nq * 256, dO.begin() + (size_t)(b + 1) * nq * 256);
            std::vector<double> o((size_t)nq * 256), dq(o.size(), 0.0), dk((size_t)nk * 256, 0.0),
                dv(dk.size(), 0.0);
            cpu_attn(nq, nk, q, kv, kv, d, o, dq, dk, dv, h, 256, 512, 0, 0, 256);
            for (int i = 0; i < nq; ++i)
                for (int c = 0; c < 32; ++c) {
                    size_t gi = ((size_t)b * nq + i) * 256 + h * 32 + c,
                           li = (size_t)i * 256 + h * 32 + c;
                    O[gi] = o[li];
                    dQ[gi] = dq[li];
                }
            for (int j = 0; j < nk; ++j)
                for (int c = 0; c < 32; ++c) {
                    size_t gi = ((size_t)b * nk + j) * 256 + h * 32 + c,
                           li = (size_t)j * 256 + h * 32 + c;
                    dK[gi] = dk[li];
                    dV[gi] = dv[li];
                }
        }
    int nch = (nk + 127) / 128, nqt = (nq + 63) / 64;
    Dev<bf16> dq_(Q.size()), dkv(KV.size()), o(Q.size()), ddo(dO.size());
    Dev<float> lse((size_t)B * nq * 8), po((size_t)nch * B * nq * 8 * 32),
        pm((size_t)nch * B * nq * 8), pl((size_t)nch * B * nq * 8), dsum((size_t)B * nq * 8),
        gq((size_t)B * nq * 256), gk((size_t)B * nk * 256), gv((size_t)B * nk * 256),
        pdq((size_t)nch * B * nq * 256), pdk((size_t)nqt * B * nk * 256),
        pdv((size_t)nqt * B * nk * 256);
    dq_.put(tobf(Q));
    dkv.put(tobf(KV));
    ddo.put(tobf(dO));
    Attn a;
    a.kind = 1;
    a.q = dq_.p;
    a.ldq = 256;
    a.k = dkv.p;
    a.v = dkv.p + 256;
    a.ldk = 512;
    a.B = B;
    a.nq = nq;
    a.nk = nk;
    a.o = o.p;
    a.lse = lse.p;
    a.po = po.p;
    a.pm = pm.p;
    a.pl = pl.p;
    a.dO = ddo.p;
    a.dsum = dsum.p;
    a.dq = gq.p;
    a.dk = gk.p;
    a.dv = gv.p;
    a.pdq = pdq.p;
    a.pdk = pdk.p;
    a.pdv = pdv.p;
    for (int st = 0; st < 5; ++st)
        k_attn<<<G, Threads, SmemBytes>>>(a, st, o.p, B * nq);
    CK(cudaDeviceSynchronize());
    char nm[64];
    int fails = 0;
    snprintf(nm, sizeof(nm), "attention %dx%d forward", nq, nk);
    fails += report(nm, nrms(fromb(o.get()), O), 1e-2);
    snprintf(nm, sizeof(nm), "attention %dx%d dq", nq, nk);
    fails += report(nm, nrms(gq.get(), dQ), 3e-2);
    snprintf(nm, sizeof(nm), "attention %dx%d dk", nq, nk);
    fails += report(nm, nrms(gk.get(), dK), 3e-2);
    snprintf(nm, sizeof(nm), "attention %dx%d dv", nq, nk);
    fails += report(nm, nrms(gv.get(), dV), 3e-2);
    return fails;
}

static int test_attn_window(int TD, int TH, int TW, int shift) {
    Rng r(21 + shift);
    int B = 2, T = TD * TH * TW;
    auto QKV = rnd((size_t)B * T * 768, r), dO = rnd((size_t)B * T * 256, r);
    std::vector<double> O((size_t)B * T * 256, 0.0), dQKV((size_t)B * T * 768, 0.0);
    int wz = (TD + shift + 3) / 4, wy = (TH + shift + 3) / 4, wx = (TW + shift + 3) / 4;
    for (int b = 0; b < B; ++b)
        for (int iz = 0; iz < wz; ++iz)
            for (int iy = 0; iy < wy; ++iy)
                for (int ix = 0; ix < wx; ++ix) {
                    std::vector<int> rows;
                    for (int z = std::max(0, iz * 4 - shift); z < std::min(TD, iz * 4 - shift + 4);
                        ++z)
                        for (int y = std::max(0, iy * 4 - shift);
                            y < std::min(TH, iy * 4 - shift + 4); ++y)
                            for (int x = std::max(0, ix * 4 - shift);
                                x < std::min(TW, ix * 4 - shift + 4); ++x)
                                rows.push_back(b * T + (z * TH + y) * TW + x);
                    int n = rows.size();
                    std::vector<float> q((size_t)n * 768), d((size_t)n * 256);
                    for (int i = 0; i < n; ++i) {
                        std::copy(QKV.begin() + (size_t)rows[i] * 768,
                            QKV.begin() + (size_t)rows[i] * 768 + 768, q.begin() + (size_t)i * 768);
                        std::copy(dO.begin() + (size_t)rows[i] * 256,
                            dO.begin() + (size_t)rows[i] * 256 + 256, d.begin() + (size_t)i * 256);
                    }
                    for (int h = 0; h < 8; ++h) {
                        std::vector<double> o((size_t)n * 256), dq(o.size(), 0.0),
                            dk(o.size(), 0.0), dv(o.size(), 0.0);
                        cpu_attn(n, n, q, q, q, d, o, dq, dk, dv, h, 768, 768, 0, 256, 512);
                        for (int i = 0; i < n; ++i)
                            for (int c = 0; c < 32; ++c) {
                                O[(size_t)rows[i] * 256 + h * 32 + c] =
                                    o[(size_t)i * 256 + h * 32 + c];
                                dQKV[(size_t)rows[i] * 768 + h * 32 + c] =
                                    dq[(size_t)i * 256 + h * 32 + c];
                                dQKV[(size_t)rows[i] * 768 + 256 + h * 32 + c] =
                                    dk[(size_t)i * 256 + h * 32 + c];
                                dQKV[(size_t)rows[i] * 768 + 512 + h * 32 + c] =
                                    dv[(size_t)i * 256 + h * 32 + c];
                            }
                    }
                }
    Dev<bf16> dqkv(QKV.size()), o(O.size()), ddo(dO.size());
    Dev<float> lse((size_t)B * T * 8), dsum((size_t)B * T * 8), g(QKV.size());
    dqkv.put(tobf(QKV));
    ddo.put(tobf(dO));
    Attn a;
    a.kind = 0;
    a.q = dqkv.p;
    a.k = dqkv.p + 256;
    a.v = dqkv.p + 512;
    a.ldq = 768;
    a.ldk = 768;
    a.B = B;
    a.TD = TD;
    a.TH = TH;
    a.TW = TW;
    a.shift = shift;
    a.o = o.p;
    a.lse = lse.p;
    a.dO = ddo.p;
    a.dsum = dsum.p;
    a.dq = g.p;
    a.dk = g.p + 256;
    a.dv = g.p + 512;
    a.lddq = 768;
    a.lddk = 768;
    for (int st = 0; st < 5; ++st)
        k_attn<<<G, Threads, SmemBytes>>>(a, st, o.p, B * T);
    CK(cudaDeviceSynchronize());
    char nm[64];
    int fails = 0;
    snprintf(nm, sizeof(nm), "window %d shift %d forward", TD, shift);
    fails += report(nm, nrms(fromb(o.get()), O), 1e-2);
    snprintf(nm, sizeof(nm), "window %d shift %d backward", TD, shift);
    fails += report(nm, nrms(g.get(), dQKV), 3e-2);
    return fails;
}

static int test_ln() {
    Rng r(5);
    int rows = 3000;
    auto x = rnd((size_t)rows * 256, r, 3.f), g = rnd(256, r), b = rnd(256, r),
         dy = rnd((size_t)rows * 256, r);
    for (auto &v : x)
        v += 0.5f;
    std::vector<double> y(x.size()), dx(x.size()), dg(256, 0.0), db(256, 0.0);
    for (int i = 0; i < rows; ++i) {
        double mu = 0, var = 0;
        for (int c = 0; c < 256; ++c)
            mu += x[(size_t)i * 256 + c];
        mu /= 256;
        for (int c = 0; c < 256; ++c)
            var += std::pow(x[(size_t)i * 256 + c] - mu, 2);
        double rs = 1.0 / std::sqrt(var / 256 + 1e-5), s1 = 0, s2 = 0;
        std::vector<double> xh(256), gg(256);
        for (int c = 0; c < 256; ++c) {
            xh[c] = (x[(size_t)i * 256 + c] - mu) * rs;
            y[(size_t)i * 256 + c] = xh[c] * g[c] + b[c];
            gg[c] = dy[(size_t)i * 256 + c] * g[c];
            s1 += gg[c];
            s2 += gg[c] * xh[c];
            dg[c] += dy[(size_t)i * 256 + c] * xh[c];
            db[c] += dy[(size_t)i * 256 + c];
        }
        for (int c = 0; c < 256; ++c)
            dx[(size_t)i * 256 + c] = rs * (gg[c] - s1 / 256 - xh[c] * s2 / 256);
    }
    Dev<float> dX(x.size()), dG(256), dB(256), yf(x.size()), st((size_t)rows * 2), ddy(x.size()),
        gx(x.size()), part((size_t)G * 512), red(512);
    Dev<bf16> yh(x.size());
    dX.put(x);
    dG.put(g);
    dB.put(b);
    ddy.put(dy);
    k_ln<<<G, Threads, SmemBytes>>>(
        0, dX.p, rows, dG.p, dB.p, yh.p, yf.p, st.p, nullptr, nullptr, nullptr);
    k_ln<<<G, Threads, SmemBytes>>>(
        1, dX.p, rows, dG.p, dB.p, nullptr, nullptr, st.p, ddy.p, gx.p, part.p);
    k_reduce<<<G, Threads>>>(red.p, part.p, 512, G, 0);
    CK(cudaDeviceSynchronize());
    int fails = report("layernorm forward", nrms(yf.get(), y), 1e-5);
    fails += report("layernorm dx", nrms(gx.get(), dx), 1e-4);
    auto rv = red.get();
    std::vector<float> rg(rv.begin(), rv.begin() + 256), rbv(rv.begin() + 256, rv.end());
    fails += report("layernorm dgamma", nrms(rg, dg), 1e-4);
    fails += report("layernorm dbeta", nrms(rbv, db), 1e-4);
    return fails;
}

static int test_recur() {
    Rng r(9);
    int B = 2, TD = 16, TH = 4, TW = 4, rows = B * TD * TH * TW;
    auto x = rnd((size_t)rows * 256, r), gates = rnd((size_t)rows * 1024, r, 2.f),
         lam = rnd(512, r, 2.f), dy = rnd((size_t)rows * 256, r);
    for (auto &v : lam)
        v += 3.f;
    auto f = [&](const std::vector<double> &xx, const std::vector<double> &gg,
                 const std::vector<double> &ll, std::vector<double> &y) {
        y.assign((size_t)rows * 256, 0.0);
        for (int b = 0; b < B; ++b)
            for (int yx = 0; yx < TH * TW; ++yx)
                for (int ch = 0; ch < 256; ++ch)
                    for (int dir = 0; dir < 2; ++dir) {
                        double ls = std::log(1.0 / (1.0 + std::exp(-ll[dir * 256 + ch]))), h = 0;
                        for (int s = 0; s < TD; ++s) {
                            int t = dir ? TD - 1 - s : s;
                            size_t row = (size_t)b * TD * TH * TW + t * TH * TW + yx;
                            double rr = 1 / (1 + std::exp(-gg[row * 1024 + dir * 512 + ch]));
                            double ii = 1 / (1 + std::exp(-gg[row * 1024 + dir * 512 + 256 + ch]));
                            double a = std::exp(8 * rr * ls),
                                   bb = std::sqrt(std::max(1 - a * a, 1e-12));
                            h = a * h + bb * ii * xx[row * 256 + ch];
                            y[row * 256 + ch] += h;
                        }
                    }
    };
    std::vector<double> X(x.begin(), x.end()), Gt(gates.begin(), gates.end()),
        L(lam.begin(), lam.end()), y0;
    f(X, Gt, L, y0);
    Dev<bf16> dxb(x.size()), dgb(gates.size()), yb(x.size()), dg(gates.size());
    Dev<float> dl(512), h((size_t)2 * rows * 256), ddy(dy.size()), gx(x.size()),
        dlam((size_t)B * TH * TW * 512), red(512);
    dxb.put(tobf(x));
    dgb.put(tobf(gates));
    dl.put(lam);
    ddy.put(dy);
    Recur R;
    R.x = dxb.p;
    R.gates = dgb.p;
    R.lam = dl.p;
    R.h = h.p;
    R.y = yb.p;
    R.B = B;
    R.TD = TD;
    R.TH = TH;
    R.TW = TW;
    R.dy = ddy.p;
    R.dx = gx.p;
    R.dg = dg.p;
    R.dlam = dlam.p;
    k_recur<<<G, Threads>>>(R, 0);
    k_recur<<<G, Threads>>>(R, 1);
    k_reduce<<<G, Threads>>>(red.p, dlam.p, 512, B * TH * TW, 0);
    CK(cudaDeviceSynchronize());
    int fails = report("recurrence forward", nrms(fromb(yb.get()), y0), 1e-2);
    auto dir_deriv = [&](int which) {
        Rng q(77 + which);
        std::vector<double> vx(X.size(), 0.0), vg(Gt.size(), 0.0), vl(L.size(), 0.0);
        if (which == 0)
            for (auto &v : vx)
                v = q.sym();
        if (which == 1)
            for (auto &v : vg)
                v = q.sym();
        if (which == 2)
            for (auto &v : vl)
                v = q.sym();
        double e = 1e-4;
        std::vector<double> Xp = X, Xm = X, Gp = Gt, Gm = Gt, Lp = L, Lm = L, yp, ym;
        for (size_t i = 0; i < X.size(); ++i) {
            Xp[i] += e * vx[i];
            Xm[i] -= e * vx[i];
        }
        for (size_t i = 0; i < Gt.size(); ++i) {
            Gp[i] += e * vg[i];
            Gm[i] -= e * vg[i];
        }
        for (size_t i = 0; i < L.size(); ++i) {
            Lp[i] += e * vl[i];
            Lm[i] -= e * vl[i];
        }
        f(Xp, Gp, Lp, yp);
        f(Xm, Gm, Lm, ym);
        double fd = 0;
        for (size_t i = 0; i < yp.size(); ++i)
            fd += dy[i] * (yp[i] - ym[i]) / (2 * e);
        double an = 0;
        if (which == 0) {
            auto g = gx.get();
            for (size_t i = 0; i < g.size(); ++i)
                an += g[i] * vx[i];
        }
        if (which == 1) {
            auto g = fromb(dg.get());
            for (size_t i = 0; i < g.size(); ++i)
                an += g[i] * vg[i];
        }
        if (which == 2) {
            auto g = red.get();
            for (size_t i = 0; i < g.size(); ++i)
                an += g[i] * vl[i];
        }
        return std::fabs(an - fd) / std::max(std::fabs(fd), 1e-12);
    };
    fails += report("recurrence dx direct", dir_deriv(0), 2e-2);
    fails += report("recurrence dgates", dir_deriv(1), 2e-2);
    fails += report("recurrence dlambda", dir_deriv(2), 2e-2);
    return fails;
}

template <int C>
__global__ void __launch_bounds__(Threads, 1) k_lnc(int stage, const bf16 *x, long n,
    const float *g, const float *b, bf16 *y, float *st, bf16 *dy, float *part) {
    extern __shared__ __align__(128) unsigned char sm[];
    if (stage == 0)
        lnc_fwd<C>(x, n, g, b, y, st);
    if (stage == 1)
        lnc_bg<C>(x, st, dy, n, part, sm);
    if (stage == 2)
        lnc_bx<C>(x, st, dy, n, g);
}

template <int C> static int test_lnc() {
    CK(cudaFuncSetAttribute(k_lnc<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    Rng r(31 + C);
    long n = 5000;
    auto x = rnd(n * C, r, 2.f), g = rnd(C, r), bb = rnd(C, r), dy = rnd(n * C, r);
    for (long i = 0; i < 300 * C; ++i)
        x[i] = rb(0.01f * (i % C));
    std::vector<double> y(n * C), dx(n * C), dg(C, 0.0), db(C, 0.0);
    for (long s = 0; s < n; ++s) {
        double mu = 0, sq = 0;
        for (int c = 0; c < C; ++c)
            mu += x[s * C + c];
        mu /= C;
        for (int c = 0; c < C; ++c)
            sq += (x[s * C + c] - mu) * (x[s * C + c] - mu);
        double rs = 1.0 / std::sqrt(sq / C + 0.1), s1 = 0, s2 = 0;
        for (int c = 0; c < C; ++c) {
            double xh = (x[s * C + c] - mu) * rs;
            y[s * C + c] = xh * g[c] + bb[c];
            dg[c] += dy[s * C + c] * xh;
            db[c] += dy[s * C + c];
            s1 += dy[s * C + c] * g[c];
            s2 += dy[s * C + c] * g[c] * xh;
        }
        for (int c = 0; c < C; ++c)
            dx[s * C + c] =
                rs * (dy[s * C + c] * g[c] - s1 / C - (x[s * C + c] - mu) * rs * s2 / C);
    }
    Dev<bf16> dX(n * C), dY(n * C), dD(n * C);
    Dev<float> dG(C), dB(C), st(n * 2), part((size_t)G * 2 * C), red(2 * C);
    dX.put(tobf(x));
    dG.put(g);
    dB.put(bb);
    dD.put(tobf(dy));
    k_lnc<C><<<G, Threads, SmemBytes>>>(0, dX.p, n, dG.p, dB.p, dY.p, st.p, nullptr, nullptr);
    k_lnc<C><<<G, Threads, SmemBytes>>>(1, dX.p, n, dG.p, dB.p, nullptr, st.p, dD.p, part.p);
    k_reduce<<<G, Threads>>>(red.p, part.p, 2 * C, G, 0);
    k_lnc<C><<<G, Threads, SmemBytes>>>(2, dX.p, n, dG.p, dB.p, nullptr, st.p, dD.p, nullptr);
    CK(cudaDeviceSynchronize());
    char nm[64];
    int f = 0;
    snprintf(nm, sizeof(nm), "channel norm %d forward", C);
    f += report(nm, nrms(fromb(dY.get()), y), 1e-2);
    snprintf(nm, sizeof(nm), "channel norm %d dx", C);
    f += report(nm, nrms(fromb(dD.get()), dx), 1e-2);
    auto rv = red.get();
    std::vector<float> rg(rv.begin(), rv.begin() + C), rbv(rv.begin() + C, rv.end());
    snprintf(nm, sizeof(nm), "channel norm %d dgamma", C);
    f += report(nm, nrms(rg, dg), 1e-3);
    snprintf(nm, sizeof(nm), "channel norm %d dbeta", C);
    f += report(nm, nrms(rbv, db), 1e-3);
    return f;
}

static int unit_tests() {
    launch_setup();
    int fails = 0;
    fails += test_gemm();
    int s1[3] = {1, 1, 1}, s122[3] = {1, 2, 2}, s2[3] = {2, 2, 2};
    fails += test_conv_case<8, 2, 4, 10>("conv 32->32", 8, 16, 16, s1, 32, 32, 4, 8, 8, 4, 8, 8);
    fails += test_conv_case<4, 4, 4, 10>("conv 64->64", 8, 16, 16, s1, 64, 64, 4, 8, 8, 4, 8, 8);
    fails +=
        test_conv_case<4, 2, 4, 15>("conv 32->64 s122", 8, 16, 16, s122, 32, 64, 4, 4, 8, 4, 8, 8);
    fails +=
        test_conv_case<2, 2, 2, 12>("conv 64->128 s2", 8, 16, 16, s2, 64, 128, 2, 4, 8, 4, 8, 8);
    fails += test_conv1();
    fails += test_attn_dense(128, 256);
    fails += test_attn_dense(256, 128);
    fails += test_attn_window(8, 8, 8, 0);
    fails += test_attn_window(8, 8, 8, 2);
    fails += test_ln();
    fails += test_recur();
    fails += test_lnc<32>();
    fails += test_lnc<64>();
    fails += test_lnc<128>();
    printf("unit tests: %s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails;
}

static void synth_batch(const Cfg &c, HostBatch &hb, Model &m, u64 seed) {
    Rng r(seed);
    hb.resize(c, m.maxpairs, m.maxcsr, m.maxent);
    const long V = c.V(), T = c.T();
    for (int b = 0; b < c.B; ++b) {
        for (long i = 0; i < V; ++i) {
            float v = r.sym() * 1.5f;
            hb.ctt[b * V + i] = __float2bfloat16(v);
            hb.ct[b * V + i] = __float2bfloat16(v);
            uint8_t lb = 0;
            if (r.uni() < 0.8f)
                lb |= 1 | (r.uni() < 0.3f ? 2 : 0);
            if (b == 0 && r.uni() < 0.5f)
                lb |= 4 | (r.uni() < 0.2f ? 8 : 0);
            hb.lab[b * V + i] = lb;
        }
        long masked = 0;
        for (long t = 0; t < T; ++t) {
            uint8_t mk = b == 0 && r.uni() < 0.75f;
            hb.tokmask[b * T + t] = mk;
            masked += mk;
        }
        for (long i = 0; i < V; ++i) {
            int x = i % c.W, y = (i / c.W) % c.H, z = i / (c.W * c.H);
            long t = ((z / 4) * c.TH() + y / 8) * c.TW() + x / 8;
            if (hb.tokmask[b * T + t])
                hb.ct[b * V + i] = __float2bfloat16(0.f);
        }
        for (long i = 0; i < (long)c.C * c.C * 4; ++i)
            hb.ctx[(long)b * c.C * c.C * 4 + i] = __float2bfloat16(r.sym());
        for (int z = 0; z < c.D; ++z) {
            hb.centre[(b * c.D + z) * 2] = -40.f + 0.3f * z;
            hb.centre[(b * c.D + z) * 2 + 1] = -60.f + 0.2f * z;
        }
        hb.boxes[b * 4 + 0] = 8;
        hb.boxes[b * 4 + 1] = 8;
        hb.boxes[b * 4 + 2] = 40;
        hb.boxes[b * 4 + 3] = 30;
        Meta &mt = hb.meta[b];
        mt.curl = b ? -1 : 1;
        mt.recon = b == 0;
        mt.labelled = 1;
        mt.conf = b ? 0.7f : 1.f;
        mt.w_recon = masked ? 1.f / (masked * 256.f) : 0.f;
        mt.w_pos = mt.conf * 0.5f / (V * 0.24f) / c.B;
        mt.w_neg = mt.conf * 0.5f / (V * 0.56f) / c.B;
        mt.w_con_pos = b == 0 ? 0.5f / (V * 0.5f) / c.B : 0.f;
        mt.w_con_neg = b == 0 ? 0.5f / (V * 0.5f) / c.B : 0.f;
        for (int k = 0; k < 4; ++k)
            mt.box[k] = hb.boxes[b * 4 + k];
        mt.origin_z = 1000.f * b;
        mt.voxel_scale = 1.f;
        std::vector<Pair> ps;
        for (int i = 0; i < 400; ++i) {
            Pair p{};
            p.kind = i % 4;
            for (int k = 0; k < 3; ++k) {
                float lim = k == 0 ? c.D - 1 : k == 1 ? c.H - 1 : c.W - 1;
                p.a[k] = r.uni() * lim;
                p.b[k] = r.uni() * lim;
            }
            p.anga = 0.1f * r.sym();
            p.angb = 0.1f * r.sym();
            p.target = p.kind == 2 ? (float)(int)(3 * r.sym()) : p.kind == 3 ? 2.f * r.sym() : 0.f;
            p.weight = 1.f / 100.f / c.B;
            ps.push_back(p);
        }
        pack_pairs(c, hb, b, ps, m.maxpairs, m.maxcsr, m.maxent);
    }
}

static double objective(Model &m) {
    std::vector<float> mt(64);
    CK(cudaMemcpy(mt.data(), m.metrics, 64 * 4, cudaMemcpyDeviceToHost));
    return (double)mt[0] + mt[1] + mt[2] + mt[3] + mt[8];
}

static int test_model() {
    launch_setup();
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    Model m;
    m.c.B = 2;
    m.c.D = 16;
    m.c.H = 32;
    m.c.W = 64;
    m.c.C = 64;
    m.allocate();
    m.init(5);
    m.upload_params();
    m.build(0.f);
    printf("model ops=%zu parameters=%ld\n", m.ops.size(), m.ptotal);
    HostBatch hb;
    synth_batch(m.c, hb, m, 9);
    upload_batch(m, hb, 0);
    float sc[8] = {1e-3f, 1.f, 0, 0, 0, 0, 0, 0};
    CK(cudaMemcpy(m.scal, sc, 32, cudaMemcpyHostToDevice));
    const int L = 3;
    launch_step(m, L, 1, 0, 0, nullptr);
    CK(cudaDeviceSynchronize());
    double f0 = objective(m);
    std::vector<float> g(m.ptotal);
    CK(cudaMemcpy(g.data(), m.grad, m.ptotal * 4, cudaMemcpyDeviceToHost));
    printf("model objective=%.6f\n", f0);
    std::vector<float> metr(64);
    CK(cudaMemcpy(metr.data(), m.metrics, 256, cudaMemcpyDeviceToHost));
    printf("model losses recon=%.5f sheet=%.5f contact=%.5f pairs=%.5f\n", metr[0], metr[1],
        metr[2], metr[3]);
    int fails = 0, checked = 0;
    std::vector<float> base = m.host_master;
    for (auto &p : m.params) {
        Rng r(1000 + checked);
        std::vector<float> v(p.n);
        double rms = 0;
        for (long i = 0; i < p.n; ++i)
            rms += (double)base[p.off + i] * base[p.off + i];
        rms = std::sqrt(rms / p.n);
        double eps = std::max(2e-2 * rms, 2e-3);
        double gv = 0;
        for (long i = 0; i < p.n; ++i) {
            v[i] = m.host_flags[p.off + i] & 2 ? 0.f : r.sym();
            gv += (double)g[p.off + i] * v[i];
        }
        double fp = 0, fm = 0;
        for (int grow = 0; grow < 4; ++grow) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                for (long i = 0; i < p.n; ++i)
                    m.host_master[p.off + i] = base[p.off + i] + (sgn ? -1 : 1) * eps * v[i];
                m.upload_params();
                launch_step(m, L, 0, 0, 0, nullptr);
                CK(cudaDeviceSynchronize());
                (sgn ? fm : fp) = objective(m);
            }
            if (std::fabs(fp - fm) > 4e-3 || grow == 3)
                break;
            eps *= 4;
        }
        for (long i = 0; i < p.n; ++i)
            m.host_master[p.off + i] = base[p.off + i];
        double fd = (fp - fm) / (2 * eps);
        double err = std::fabs(fd - gv) / std::max(std::fabs(fd) + std::fabs(gv), 1e-9);
        bool ok = err < 0.15 || std::fabs(fp - fm) < 2e-3;
        printf("grad %-14s n=%-8ld eps=%.2e analytic=% .5e numeric=% .5e rel=%.3f %s\n",
            p.name.c_str(), p.n, eps, gv, fd, err,
            std::fabs(fp - fm) < 2e-3 ? "below-noise"
                : ok                  ? "ok"
                                      : "BAD");
        fails += !ok;
        ++checked;
    }
    m.upload_params();
    std::vector<float> w1(m.ptotal), w2(m.ptotal);
    for (int rep = 0; rep < 2; ++rep) {
        m.upload_params();
        CK(cudaMemset(m.m1, 0, m.ptotal * 4));
        CK(cudaMemset(m.m2, 0, m.ptotal * 4));
        for (int s = 0; s < 2; ++s) {
            launch_step(m, 5, 1, 1, 0, nullptr);
            CK(cudaDeviceSynchronize());
        }
        CK(cudaMemcpy(rep ? w2.data() : w1.data(), m.master, m.ptotal * 4, cudaMemcpyDeviceToHost));
    }
    long diff = 0;
    for (long i = 0; i < m.ptotal; ++i)
        diff += memcmp(&w1[i], &w2[i], 4) != 0;
    printf("determinism two runs of two steps: %ld differing weights of %ld %s\n", diff, m.ptotal,
        diff ? "FAIL" : "ok");
    fails += diff != 0;
    printf("model test: %s (%d failing of %d tensors)\n", fails ? "FAIL" : "PASS", fails, checked);
    return fails;
}

template <class F> static double timeit(F f, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    f();
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i)
        f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

template <int WGM, int MT, int NT, int PF>
static void bench_conv(
    const char *name, int Z, int Y, int X, int cin, int cout, int TZ, int TY, int TX) {
    int B = 2;
    size_t nin = (size_t)B * Z * Y * X * cin, nout = (size_t)B * Z * Y * X * cout;
    Dev<bf16> x(nin), y(nout), w(27L * cin * cout), gx(nin);
    Conv c;
    c.s0 = x.p;
    c.c0 = cin;
    c.B = B;
    c.SZ = Z;
    c.SY = Y;
    c.SX = X;
    c.OZ = Z;
    c.OY = Y;
    c.OX = X;
    c.TZ = TZ;
    c.TY = TY;
    c.TX = TX;
    c.w = w.p;
    c.cin = cin;
    c.cout = cout;
    c.d0 = y.p;
    double fl = 2.0 * B * Z * Y * X * 27.0 * cin * cout;
    double t = timeit([&] { k_conv<WGM, MT, NT, PF><<<G, Threads, SmemBytes>>>(c); }, 5);
    printf("bench %-22s forward %.3f ms %.1f TFLOP/s\n", name, t, fl / t / 1e9);
    Conv d = c;
    d.dx = 1;
    d.s0 = y.p;
    d.c0 = cout;
    d.d0 = gx.p;
    double t2 = cin >= 64 ? timeit([&] { k_conv<4, 4, 4, 10><<<G, Threads, SmemBytes>>>(d); }, 5)
                          : timeit([&] { k_conv<8, 2, 4, 10><<<G, Threads, SmemBytes>>>(d); }, 5);
    printf("bench %-22s data grad %.3f ms %.1f TFLOP/s\n", name, t2, fl / t2 / 1e9);
    int items = (cin / 32) * (cout / 32) * 2, splits = std::max(1, (2 * G + items - 1) / items);
    Dev<float> part((size_t)splits * 27 * cin * cout), bpart((size_t)splits * cout);
    ConvW cw;
    cw.s0 = x.p;
    cw.c0 = cin;
    cw.B = B;
    cw.SZ = Z;
    cw.SY = Y;
    cw.SX = X;
    cw.OZ = Z;
    cw.OY = Y;
    cw.OX = X;
    cw.TZ = TZ;
    cw.TY = TY;
    cw.TX = TX;
    cw.dy = y.p;
    cw.cout = cout;
    cw.part = part.p;
    cw.bpart = bpart.p;
    cw.splits = splits;
    double t3 = timeit([&] { k_convw<10><<<G, Threads, SmemBytes>>>(cw); }, 5);
    printf("bench %-22s weight grad %.3f ms %.1f TFLOP/s (splits %d)\n", name, t3, fl / t3 / 1e9,
        splits);
}

static void bench_gemm(int M, int N, int K) {
    Dev<bf16> a((size_t)M * K), b((size_t)K * N);
    Dev<float> c((size_t)M * N);
    Gemm g;
    g.M = M;
    g.N = N;
    g.K = K;
    g.a.p = a.p;
    g.a.ld = K;
    g.b.p = b.p;
    g.b.ld = N;
    g.e.mode = E_F32;
    g.e.f = c.p;
    g.e.ld = N;
    double t = timeit([&] { k_gemm<<<G, Threads, SmemBytes>>>(g); }, 10);
    printf("bench gemm %dx%dx%d %.3f ms %.1f TFLOP/s\n", M, N, K, t, 2.0 * M * N * K / t / 1e9);
}

static void bench_all() {
    launch_setup();
    bench_conv<8, 2, 4, 10>("s0b 32->32 full", 64, 128, 128, 32, 32, 4, 8, 8);
    bench_conv<4, 4, 4, 10>("s1b 64->64 64^3", 64, 64, 64, 64, 64, 4, 8, 8);
    bench_conv<4, 4, 4, 10>("u1 128->64 64^3", 64, 64, 64, 128, 64, 4, 8, 8);
    bench_conv<4, 4, 4, 10>("s2b 128->128 32^3", 32, 32, 32, 128, 128, 4, 8, 8);
    bench_gemm(8192, 768, 256);
    bench_gemm(8192, 256, 256);
    bench_gemm(8192, 1376, 256);
    bench_gemm(8192, 256, 688);
}

struct Record {
    int file;
    long offset;
    int layout, source, split, curl, bits, npoints, group;
    float voxel, conf, origin[3], box[4], scale;
};

struct Dataset {
    std::vector<std::string> paths;
    std::vector<int> fds;
    std::vector<Record> recs;
    std::vector<int> train[5], held;
    u64 fingerprint = 1469598103934665603ULL;
    static constexpr long V = 64L * 128 * 128;
    static constexpr long LabBytes = 128 + 5 * V + 262144 + 512 + 24 * 4096;
    static constexpr long UnlBytes = 128 + V + 262144 + 512;
    static constexpr long SurfBytes = 128 + 2 * V;
    static long bytes_for(int layout) {
        return layout == 0 ? LabBytes : layout == 1 ? UnlBytes : SurfBytes;
    }
    void mix(u64 v) { fingerprint = (fingerprint ^ v) * 1099511628211ULL; }
    explicit Dataset(const std::string &dir, bool all = false) {
        if (!std::filesystem::is_directory(dir))
            fail("no shard directory " + dir);
        for (auto &e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".shard")
                paths.push_back(e.path().string());
        std::sort(paths.begin(), paths.end());
        for (auto &p : paths) {
            int fd = open(p.c_str(), O_RDONLY);
            if (fd < 0)
                fail("cannot open " + p);
            unsigned char h[64];
            if (pread(fd, h, 64, 0) != 64 || memcmp(h, "LGSF", 4))
                fail("not a shard: " + p);
            uint32_t layout;
            uint64_t count;
            memcpy(&layout, h + 4, 4);
            memcpy(&count, h + 8, 8);
            if (layout > 2)
                fail("unknown shard layout in " + p);
            long stride = bytes_for(layout);
            struct stat st;
            fstat(fd, &st);
            if ((long)st.st_size != 64 + (long)count * stride)
                fail("truncated shard " + p);
            mix(std::filesystem::path(p).filename().string().size());
            mix(st.st_size);
            int fi = fds.size();
            fds.push_back(fd);
            for (uint64_t i = 0; i < count; ++i) {
                unsigned char r[128];
                long off = 64 + (long)i * stride;
                if (pread(fd, r, 128, off) != 128 || memcmp(r, "LGS1", 4))
                    fail("bad record in " + p);
                Record R{};
                R.file = fi;
                R.offset = off;
                R.layout = layout;
                uint32_t u[5];
                memcpy(u, r + 4, 20);
                R.source = u[1];
                R.split = u[2];
                R.curl = (int32_t)u[3];
                R.bits = u[4];
                memcpy(&R.voxel, r + 24, 4);
                memcpy(&R.conf, r + 28, 4);
                int32_t o[3];
                memcpy(o, r + 32, 12);
                for (int k = 0; k < 3; ++k)
                    R.origin[k] = o[k];
                memcpy(R.box, r + 44, 16);
                memcpy(&R.scale, r + 60, 4);
                uint32_t np;
                memcpy(&np, r + 64, 4);
                R.npoints = np;
                R.group = layout == 1 ? 3
                    : layout == 2     ? 4
                    : R.source == 1   ? 0
                    : R.source == 3   ? 1
                    : R.source == 2   ? 2
                                      : 5;
                if (all && R.group == 5 && R.source >= 4 && R.source <= 6)
                    R.group = 1;
                for (int k = 0; k < 16; ++k)
                    mix(((u64 *)r)[k]);
                int id = recs.size();
                recs.push_back(R);
                if (R.split)
                    held.push_back(id);
                if ((!R.split || all) && R.group < 5)
                    train[R.group].push_back(id);
            }
        }
        if (recs.empty())
            fail("no records in " + dir);
        printf("shards=%zu records=%zu train=%zu/%zu/%zu/%zu/%zu held_out=%zu%s fingerprint=%llu\n",
            paths.size(), recs.size(), train[0].size(), train[1].size(), train[2].size(),
            train[3].size(), train[4].size(), held.size(), all ? " (trained on too)" : "",
            (unsigned long long)fingerprint);
    }
    void read(int id, std::vector<unsigned char> &buf) const {
        const Record &R = recs[id];
        long n = bytes_for(R.layout);
        buf.resize(n);
        long got = pread(fds[R.file], buf.data(), n, R.offset);
        if (got != n)
            fail("short read");
    }
};

struct Weights {
    float recon = 1, sheet = 1, contact = 0.5f, same = 1, next = 1, relative = 1, absolute = 0.f;
    float nogeom_drop = 0.15f;
    int accum = 1;
};

struct Slab {
    std::vector<float> ct;
    std::vector<uint8_t> sheet;
    std::vector<uint16_t> inst;
    std::vector<uint8_t> context;
    std::vector<float> centre;
    struct Pt {
        float z, y, x;
        int set, kind;
        float w;
    };
    std::vector<Pt> pts;
};

static void decode(const Record &R, const std::vector<unsigned char> &buf, Slab &s) {
    const long V = Dataset::V;
    s.ct.resize(V);
    s.sheet.assign(V, 255);
    s.inst.assign(V, 0);
    const unsigned char *p = buf.data() + 128;
    if (R.layout == 2) {
        for (long i = 0; i < V; ++i)
            s.ct[i] = p[i];
        memcpy(s.sheet.data(), p + V, V);
        s.context.assign(262144, 0);
        s.centre.assign(128, -10000.f);
        s.pts.clear();
        return;
    }
    if (R.layout == 1) {
        for (long i = 0; i < V; ++i)
            s.ct[i] = p[i];
        p += V;
    } else {
        const uint16_t *c = reinterpret_cast<const uint16_t *>(p);
        for (long i = 0; i < V; ++i)
            s.ct[i] = c[i];
        p += 2 * V;
        memcpy(s.sheet.data(), p, V);
        p += V;
        memcpy(s.inst.data(), p, 2 * V);
        p += 2 * V;
    }
    s.context.assign(p, p + 262144);
    p += 262144;
    s.centre.resize(128);
    memcpy(s.centre.data(), p, 512);
    p += 512;
    s.pts.clear();
    for (int i = 0; i < R.npoints; ++i) {
        Slab::Pt q;
        memcpy(&q.z, p + i * 24, 4);
        memcpy(&q.y, p + i * 24 + 4, 4);
        memcpy(&q.x, p + i * 24 + 8, 4);
        memcpy(&q.set, p + i * 24 + 12, 4);
        memcpy(&q.kind, p + i * 24 + 16, 4);
        memcpy(&q.w, p + i * 24 + 20, 4);
        s.pts.push_back(q);
    }
}

static void rot90(std::vector<float> &a, int k, int D, int H, int W) {
    if (!k)
        return;
    std::vector<float> b(a.size());
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int ny = y, nx = x;
                for (int r = 0; r < k; ++r) {
                    int t = ny;
                    ny = nx;
                    nx = H - 1 - t;
                }
                b[((long)z * H + ny) * W + nx] = a[((long)z * H + y) * W + x];
            }
    a.swap(b);
}
template <class T> static void rot90t(std::vector<T> &a, int k, int D, int H, int W) {
    if (!k)
        return;
    std::vector<T> b(a.size());
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int ny = y, nx = x;
                for (int r = 0; r < k; ++r) {
                    int t = ny;
                    ny = nx;
                    nx = H - 1 - t;
                }
                b[((long)z * H + ny) * W + nx] = a[((long)z * H + y) * W + x];
            }
    a.swap(b);
}
static void rotpt(float &y, float &x, int k, float H) {
    for (int r = 0; r < k; ++r) {
        float t = y;
        y = x;
        x = H - 1 - t;
    }
}
template <class T> static void zflip(std::vector<T> &a, int D, long plane) {
    for (int z = 0; z < D / 2; ++z)
        std::swap_ranges(
            a.begin() + z * plane, a.begin() + (z + 1) * plane, a.begin() + (D - 1 - z) * plane);
}

static void blur(std::vector<float> &a, int D, int H, int W, float sigma) {
    int rad = std::min(3, (int)std::ceil(3 * sigma));
    float k[7], s = 0;
    for (int i = -rad; i <= rad; ++i)
        s += (k[i + 3] = std::exp(-0.5f * i * i / (sigma * sigma)));
    for (int i = -rad; i <= rad; ++i)
        k[i + 3] /= s;
    std::vector<float> b(a.size());
    long dims[3] = {D, H, W}, strides[3] = {(long)H * W, W, 1};
    for (int ax = 0; ax < 3; ++ax) {
        for (long i = 0; i < (long)a.size(); ++i) {
            long c = (i / strides[ax]) % dims[ax];
            float v = 0;
            for (int j = -rad; j <= rad; ++j) {
                long cc = std::min<long>(dims[ax] - 1, std::max<long>(0, c + j));
                v += k[j + 3] * a[i + (cc - c) * strides[ax]];
            }
            b[i] = v;
        }
        a.swap(b);
    }
}

struct Prepared {
    HostBatch hb;
    std::vector<int> ids;
    int slabs_per_group[5] = {0, 0, 0, 0, 0}, hard[5] = {0, 0, 0, 0, 0};
    double read_s = 0, aug_s = 0;
};

static float angle_term(const Slab &s, float z, float y, float x, int curl, float ref) {
    int zi = std::min(63, std::max(0, (int)(z + 0.5f)));
    float th = std::atan2(y - s.centre[zi * 2], x - s.centre[zi * 2 + 1]);
    float d = th - ref;
    d -= 6.2831853f * std::floor((d + 3.14159265f) / 6.2831853f);
    return curl * (ref + d) / 6.2831853f;
}

static void prepare_slab(const Cfg &c, const Dataset &data, int id, bool recon, u64 key,
    const Weights &lw, HostBatch &hb, int b, Model &m, long &masked_total, bool augment) {
    const long V = c.V(), T = c.T();
    std::vector<unsigned char> buf;
    data.read(id, buf);
    const Record &R = data.recs[id];
    Slab s;
    decode(R, buf, s);
    float ref_orig = angle_term(s, 31.5f, 63.5f, 63.5f, 1, 0.f) * 6.2831853f;
    Rng r(key);
    int k = augment ? r.next() % 4 : 0;
    bool fz = augment && (r.next() & 1);
    rot90(s.ct, k, c.D, c.H, c.W);
    rot90t(s.sheet, k, c.D, c.H, c.W);
    rot90t(s.inst, k, c.D, c.H, c.W);
    float box[4] = {R.box[0], R.box[1], R.box[2], R.box[3]};
    for (int z = 0; z < c.D; ++z)
        rotpt(s.centre[z * 2], s.centre[z * 2 + 1], k, c.H);
    for (auto &p : s.pts)
        rotpt(p.y, p.x, k, c.H);
    if (k) {
        std::vector<uint8_t> cx(s.context.size());
        for (int sl = 0; sl < 4; ++sl)
            for (int y = 0; y < 256; ++y)
                for (int x = 0; x < 256; ++x) {
                    int ny = y, nx = x;
                    for (int q = 0; q < k; ++q) {
                        int t = ny;
                        ny = nx;
                        nx = 255 - t;
                    }
                    cx[(sl * 256 + ny) * 256 + nx] = s.context[(sl * 256 + y) * 256 + x];
                }
        s.context.swap(cx);
        for (int q = 0; q < k; ++q) {
            float y0 = box[0], x0 = box[1], y1 = box[2], x1 = box[3];
            box[0] = x0;
            box[2] = x1;
            box[1] = 256 - y1;
            box[3] = 256 - y0;
        }
    }
    if (fz) {
        zflip(s.ct, c.D, (long)c.H * c.W);
        zflip(s.sheet, c.D, (long)c.H * c.W);
        zflip(s.inst, c.D, (long)c.H * c.W);
        for (int z = 0; z < c.D / 2; ++z) {
            std::swap(s.centre[z * 2], s.centre[(c.D - 1 - z) * 2]);
            std::swap(s.centre[z * 2 + 1], s.centre[(c.D - 1 - z) * 2 + 1]);
        }
        for (auto &p : s.pts)
            p.z = c.D - 1 - p.z;
        for (int sl = 0; sl < 2; ++sl)
            std::swap_ranges(s.context.begin() + sl * 65536, s.context.begin() + (sl + 1) * 65536,
                s.context.begin() + (3 - sl) * 65536);
    }
    int bins = R.bits == 16 ? 65536 : 256;
    std::vector<long> hist(bins, 0);
    for (long i = 0; i < V; ++i)
        hist[std::min(bins - 1, std::max(0, (int)s.ct[i]))]++;
    long lo_n = (long)(0.005 * V), hi_n = (long)(0.995 * V), acc = 0;
    float lo = 0, hi = bins - 1;
    bool lo_set = false;
    for (int i = 0; i < bins; ++i) {
        acc += hist[i];
        if (!lo_set && acc > lo_n) {
            lo = i;
            lo_set = true;
        }
        if (acc >= hi_n) {
            hi = i;
            break;
        }
    }
    if (hi <= lo)
        hi = lo + 1;
    float gam =
        augment ? std::exp(std::log(0.7f) + r.uni() * (std::log(1.5f) - std::log(0.7f))) : 1.f;
    float con = augment ? 0.75f + 0.5f * r.uni() : 1.f, noise = augment ? 0.03f * r.uni() : 0.f;
    constexpr int LutN = 1024;
    float lut[LutN + 1];
    for (int i = 0; i <= LutN; ++i)
        lut[i] = (std::pow((float)i / LutN, gam) - 0.5f) * con + 0.5f;
    float inv_span = (float)LutN / (hi - lo);
    for (long i = 0; i < V; ++i) {
        float x = (s.ct[i] - lo) * inv_span;
        float v;
        if (x <= 0.f)
            v = lut[0];
        else if (x >= (float)LutN)
            v = lut[LutN];
        else {
            int idx = (int)x;
            float f = x - idx;
            v = lut[idx] + f * (lut[idx + 1] - lut[idx]);
        }
        if (noise > 0)
            v += noise * (r.uni() + r.uni() + r.uni() - 1.5f) * 2.f;
        s.ct[i] = v;
    }
    if (augment && r.uni() < 0.3f)
        blur(s.ct, c.D, c.H, c.W, 0.2f + 0.8f * r.uni());
    double mu = 0, var = 0;
    for (long i = 0; i < V; ++i)
        mu += s.ct[i];
    mu /= V;
    for (long i = 0; i < V; ++i)
        var += (s.ct[i] - mu) * (s.ct[i] - mu);
    float inv = 1.f / std::sqrt(std::max(var / V, 1e-8));
    std::vector<uint8_t> tm(T, 0);
    if (recon) {
        std::vector<std::pair<u64, int>> order(T);
        for (long t = 0; t < T; ++t)
            order[t] = {r.next(), (int)t};
        std::sort(order.begin(), order.end());
        for (long t = 0; t < T * 3 / 4; ++t)
            tm[order[t].second] = 1;
    }
    for (long i = 0; i < V; ++i) {
        float v = (s.ct[i] - mu) * inv;
        hb.ctt[b * V + i] = __float2bfloat16(v);
        int x = i % c.W, y = (i / c.W) % c.H, z = i / ((long)c.W * c.H);
        long t = ((z / 4) * c.TH() + y / 8) * c.TW() + x / 8;
        hb.ct[b * V + i] = __float2bfloat16(tm[t] ? 0.f : v);
    }
    for (long t = 0; t < T; ++t)
        hb.tokmask[b * T + t] = tm[t];
    bool drop_geom = augment && R.layout != 2 && lw.nogeom_drop > 0.f && r.uni() < lw.nogeom_drop;
    long ctxpix = 256L * 256;
    if (drop_geom) {
        for (long i = 0; i < ctxpix * 4; ++i)
            hb.ctx[(long)b * ctxpix * 4 + i] = __float2bfloat16(0.f);
    } else {
        for (int sl = 0; sl < 4; ++sl) {
            double cm = 0, cv = 0;
            for (long i = 0; i < ctxpix; ++i)
                cm += s.context[sl * ctxpix + i];
            cm /= ctxpix;
            for (long i = 0; i < ctxpix; ++i) {
                double d = s.context[sl * ctxpix + i] - cm;
                cv += d * d;
            }
            float ci = 1.f / std::sqrt(std::max(cv / ctxpix, 1e-6));
            for (long i = 0; i < ctxpix; ++i)
                hb.ctx[((long)b * ctxpix + i) * 4 + sl] =
                    __float2bfloat16((s.context[sl * ctxpix + i] - cm) * ci);
        }
    }
    for (int z = 0; z < c.D; ++z) {
        hb.centre[(b * c.D + z) * 2] = s.centre[z * 2];
        hb.centre[(b * c.D + z) * 2 + 1] = s.centre[z * 2 + 1];
    }
    for (int q = 0; q < 4; ++q)
        hb.boxes[b * 4 + q] = drop_geom ? 0.f : box[q];
    Meta &mt = hb.meta[b];
    mt = Meta{};
    mt.curl = R.curl;
    mt.recon = recon;
    mt.labelled = !R.layout;
    mt.group = R.group;
    mt.conf = R.conf;
    for (int q = 0; q < 4; ++q)
        mt.box[q] = drop_geom ? 0.f : box[q];
    mt.origin_z = (R.layout == 2 || drop_geom) ? 0.f : R.origin[0];
    mt.voxel_scale = R.voxel > 0.f ? R.voxel / 8.f : 1.f;
    mt.nogeom = R.layout == 2 || drop_geom;
    long npos = 0, nneg = 0, ncpos = 0, ncneg = 0;
    for (long i = 0; i < V; ++i) {
        uint8_t lb = 0;
        if (s.sheet[i] < 2) {
            lb |= 1;
            if (s.sheet[i]) {
                lb |= 2;
                ++npos;
            } else
                ++nneg;
        }
        if (R.group == 0) {
            lb |= 4;
            uint16_t id0 = s.inst[i];
            bool hit = false;
            if (id0) {
                int x = i % c.W, y = (i / c.W) % c.H, z = i / ((long)c.W * c.H);
                const int dx[6] = {1, -1, 0, 0, 0, 0}, dy[6] = {0, 0, 1, -1, 0, 0},
                          dz[6] = {0, 0, 0, 0, 1, -1};
                for (int n = 0; n < 6; ++n) {
                    int xx = x + dx[n], yy = y + dy[n], zz = z + dz[n];
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= c.W || yy >= c.H || zz >= c.D)
                        continue;
                    uint16_t o = s.inst[((long)zz * c.H + yy) * c.W + xx];
                    if (o && o != id0) {
                        lb |= 8;
                        hit = true;
                        break;
                    }
                }
            }
            if (hit)
                ++ncpos;
            else
                ++ncneg;
        }
        hb.lab[b * V + i] = lb;
    }
    const float denom_b = (float)c.B * (float)std::max(1, lw.accum);
    if (npos && nneg) {
        float w = lw.sheet * R.conf / ((float)(npos + nneg) * denom_b);
        mt.w_pos = w;
        mt.w_neg = w;
    } else if (npos) {
        mt.w_pos = 0.1f * lw.sheet * R.conf / ((float)npos * denom_b);
        mt.w_neg = 0.f;
    } else if (nneg) {
        mt.w_pos = 0.f;
        mt.w_neg = 0.1f * lw.sheet * R.conf / ((float)nneg * denom_b);
    }
    if (R.group == 0 && (ncpos + ncneg) > 0) {
        float pw =
            ncpos > 0 ? std::min(15.f, std::max(1.f, std::sqrt((float)ncneg / (float)ncpos))) : 1.f;
        float cd = ((float)ncneg + pw * (float)ncpos) * denom_b;
        mt.w_con_pos = lw.contact * R.conf * pw / cd;
        mt.w_con_neg = lw.contact * R.conf / cd;
    }
    if (recon)
        for (long t = 0; t < T; ++t)
            masked_total += tm[t] * 256;
    float ref = angle_term(s, 31.5f, 63.5f, 63.5f, 1, 0.f) * 6.2831853f;
    float rot_shift = R.curl * (ref - ref_orig) / 6.2831853f;
    std::vector<Pair> ps;
    auto add = [&](float za, float ya, float xa, float zb, float yb, float xb, int kind,
                   float target) {
        Pair p{};
        p.a[0] = za;
        p.a[1] = ya;
        p.a[2] = xa;
        p.b[0] = zb;
        p.b[1] = yb;
        p.b[2] = xb;
        p.kind = kind;
        p.target = target;
        p.anga = angle_term(s, za, ya, xa, R.curl, ref);
        p.angb = kind == 3 ? 0.f : angle_term(s, zb, yb, xb, R.curl, ref);
        ps.push_back(p);
    };
    if (R.group == 0) {
        std::vector<long> sites;
        sites.reserve(V / 2);
        std::unordered_map<int, std::vector<long>> by;
        by.reserve(256);
        for (long i = 0; i < V; ++i)
            if (s.inst[i]) {
                sites.push_back(i);
                by[s.inst[i]].push_back(i);
            }
        auto zyx = [&](long i, float *o) {
            o[2] = i % c.W;
            o[1] = (i / c.W) % c.H;
            o[0] = i / ((long)c.W * c.H);
        };
        for (int q = 0; q < 4096 && !sites.empty(); ++q) {
            long a = sites[r.next() % sites.size()];
            auto &lst = by[s.inst[a]];
            long bb = lst[r.next() % lst.size()];
            float pa[3], pb[3];
            zyx(a, pa);
            zyx(bb, pb);
            add(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], 0, 0.f);
        }
        float max_t = std::min(80.f, 480.f / std::max(R.voxel > 0.f ? R.voxel : 8.f, 1.f));
        for (int q = 0; q < 4096 && !sites.empty(); ++q) {
            long a = sites[r.next() % sites.size()];
            float pa[3];
            zyx(a, pa);
            int z = pa[0];
            float cy = s.centre[z * 2], cx = s.centre[z * 2 + 1], dy = pa[1] - cy, dx = pa[2] - cx,
                  rr = std::hypot(dy, dx);
            if (rr < 1e-3f)
                continue;
            dy /= rr;
            dx /= rr;
            uint16_t id0 = s.inst[a];
            bool left_id0 = false;
            for (float t = 1.f; t < max_t; t += 0.5f) {
                int y = (int)std::lround(pa[1] + dy * t), x = (int)std::lround(pa[2] + dx * t);
                if (y < 0 || x < 0 || y >= c.H || x >= c.W)
                    break;
                uint16_t o = s.inst[((long)z * c.H + y) * c.W + x];
                if (o == 0) {
                    left_id0 = true;
                    continue;
                }
                if (o == id0) {
                    if (left_id0)
                        break;
                    continue;
                }
                add(pa[0], pa[1], pa[2], z, y, x, 1, 1.f);
                break;
            }
        }
    }
    if (!s.pts.empty()) {
        std::map<int, std::vector<int>> sets;
        for (int i = 0; i < (int)s.pts.size(); ++i)
            sets[s.pts[i].set].push_back(i);
        std::vector<int> same, rel, abs_;
        for (auto &kv : sets)
            if (kv.second.size() > 1)
                (s.pts[kv.second[0]].kind == 0 ? same : rel).push_back(kv.first);
        for (int i = 0; i < (int)s.pts.size(); ++i)
            if (s.pts[i].kind == 2)
                abs_.push_back(i);
        for (int q = 0; q < 4096 && !same.empty(); ++q) {
            auto &l = sets[same[r.next() % same.size()]];
            int a = l[r.next() % l.size()], bb = l[r.next() % l.size()];
            if (a == bb)
                continue;
            add(s.pts[a].z, s.pts[a].y, s.pts[a].x, s.pts[bb].z, s.pts[bb].y, s.pts[bb].x, 0, 0.f);
        }
        for (int q = 0; q < 4096 && !rel.empty(); ++q) {
            auto &l = sets[rel[r.next() % rel.size()]];
            int a = l[r.next() % l.size()], bb = l[r.next() % l.size()];
            if (a == bb)
                continue;
            add(s.pts[a].z, s.pts[a].y, s.pts[a].x, s.pts[bb].z, s.pts[bb].y, s.pts[bb].x, 2,
                s.pts[a].w - s.pts[bb].w);
        }
        if (abs_.size() > 1) {
            for (int q = 0; q < 4096; ++q) {
                int a = abs_[r.next() % abs_.size()], bb = abs_[r.next() % abs_.size()];
                if (a == bb)
                    continue;
                add(s.pts[a].z, s.pts[a].y, s.pts[a].x, s.pts[bb].z, s.pts[bb].y, s.pts[bb].x, 2,
                    s.pts[a].w - s.pts[bb].w);
            }
        }
        if (lw.absolute > 0.f)
            for (int a : abs_)
                add(s.pts[a].z, s.pts[a].y, s.pts[a].x, 0, 0, 0, 3, s.pts[a].w - rot_shift);
    }
    int cnt[4] = {0, 0, 0, 0};
    for (auto &p : ps)
        cnt[p.kind]++;
    const float kw[4] = {lw.same, lw.next, lw.relative, lw.absolute};
    for (auto &p : ps)
        p.weight = kw[p.kind] * R.conf / (cnt[p.kind] * denom_b);
    pack_pairs(c, hb, b, ps, m.maxpairs, m.maxcsr, m.maxent);
}

struct Sampler {
    const Dataset &d;
    std::vector<uint8_t> hard;
    std::vector<double> cum[5];
    double share[5] = {0.4, 0.25, 0.15, 0.2, 0.0};
    explicit Sampler(const Dataset &ds) : d(ds), hard(ds.recs.size(), 0) { rebuild(); }
    void rebuild() {
        for (int g = 0; g < 5; ++g) {
            cum[g].resize(d.train[g].size());
            double s = 0;
            for (size_t i = 0; i < cum[g].size(); ++i)
                cum[g][i] = (s += hard[d.train[g][i]] ? 3 : 1);
        }
    }
    int draw(bool pre, Rng &r, int &group) const {
        group = 3;
        if (!pre) {
            double u = r.uni(), acc = 0;
            group = 3;
            for (int g = 0; g < 5; ++g) {
                acc += share[g];
                if (u < acc) {
                    group = g;
                    break;
                }
            }
        }
        if (d.train[group].empty())
            fail("a training source is empty");
        double x = r.uni() * cum[group].back();
        size_t i = std::upper_bound(cum[group].begin(), cum[group].end(), x) - cum[group].begin();
        return d.train[group][std::min(i, d.train[group].size() - 1)];
    }
};

struct Pool {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::function<void()>> q;
    bool stop = false;
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i)
            th.emplace_back([this] {
                for (;;) {
                    std::function<void()> f;
                    {
                        std::unique_lock<std::mutex> l(mu);
                        cv.wait(l, [this] { return stop || !q.empty(); });
                        if (stop && q.empty())
                            return;
                        f = std::move(q.front());
                        q.pop_front();
                    }
                    f();
                }
            });
    }
    void push(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> l(mu);
            q.push_back(std::move(f));
        }
        cv.notify_one();
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> l(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto &t : th)
            t.join();
    }
};

static bool g_unmasked = false;

static void prepare_batch(const Cfg &c, const Dataset &data, const Sampler &smp, bool pre, u64 seed,
    u64 step, const Weights &lw, Model &m, Prepared &out, Pool &pool, bool augment,
    const std::vector<int> *fixed) {
    out.hb.resize(c, m.maxpairs, m.maxcsr, m.maxent);
    out.ids.assign(c.B, 0);
    std::vector<int> groups(c.B);
    for (int b = 0; b < c.B; ++b) {
        Rng r(seed * 1000003ULL + step * 97ULL + b);
        if (fixed) {
            out.ids[b] = (*fixed)[b];
            groups[b] = data.recs[out.ids[b]].group;
        } else
            out.ids[b] = smp.draw(pre, r, groups[b]);
    }
    std::vector<long> masked(c.B, 0);
    std::atomic<int> left(c.B);
    std::mutex dm;
    std::condition_variable dc;
    auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < c.B; ++b)
        pool.push([&, b] {
            bool recon = !g_unmasked && (pre || data.recs[out.ids[b]].layout == 1);
            prepare_slab(c, data, out.ids[b], recon, seed * 7919ULL + step * 131ULL + b * 17ULL, lw,
                out.hb, b, m, masked[b], augment);
            if (--left == 0) {
                std::lock_guard<std::mutex> l(dm);
                dc.notify_all();
            }
        });
    {
        std::unique_lock<std::mutex> l(dm);
        dc.wait(l, [&] { return left.load() == 0; });
    }
    long mt = 0;
    for (long v : masked)
        mt += v;
    for (int b = 0; b < c.B; ++b)
        out.hb.meta[b].w_recon = mt ? lw.recon / ((float)mt * std::max(1, lw.accum)) : 0.f;
    for (int g = 0; g < 5; ++g)
        out.slabs_per_group[g] = out.hard[g] = 0;
    for (int b = 0; b < c.B; ++b) {
        int g = std::min(4, groups[b]);
        out.slabs_per_group[g]++;
        out.hard[g] += smp.hard[out.ids[b]];
    }
    out.aug_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double step_flops(const Cfg &c, int L, int retain) {
    double B = c.B, V = c.V(), V1 = c.V1(), V2 = c.V2(), T = c.T(), Lc = c.latents, Cn = c.Cn(),
           Cp = c.Cp(), Hd = c.hid;
    auto conv = [&](double sites, double cin, double cout) {
        return 2.0 * sites * 27 * cin * cout;
    };
    double enc = conv(V, 1, 32) + conv(V, 32, 32) + conv(V1, 32, 64) + conv(V1, 64, 64) +
        conv(V2, 64, 128) + conv(V2, 128, 128) + conv(T, 128, 256) + 2.0 * T * 128 * 256 +
        2.0 * Cp * 64 * 128 + 2.0 * Cn * 2048 * 256;
    double dec = 2.0 * T * 256 * 8 * 128 + conv(V2, 256, 128) + 2.0 * V2 * 128 * 8 * 64 +
        conv(V1, 128, 64) + 2.0 * V1 * 64 * 4 * 32 + conv(V, 64, 32) + 2.0 * V * 32 * 4;
    double blk = 2.0 * T * 256 * 768 + 4.0 * T * 64 * 256 + 2.0 * T * 256 * 256 +
        2.0 * Lc * 256 * 256 + 2.0 * T * 256 * 512 + 4.0 * Lc * T * 256 + 2.0 * Lc * 256 * 256 +
        2.0 * T * 256 * 256 + 2.0 * Lc * 256 * 512 + 4.0 * T * Lc * 256 + 2.0 * T * 256 * 256 +
        2.0 * T * 256 * 1024 + 2.0 * T * 256 * 256 + 2.0 * T * 256 * 2 * Hd + 2.0 * T * Hd * 256;
    double loop = 4 * blk;
    int R = std::min(L, retain);
    return B * (enc + dec + L * loop + R * loop + 2 * (enc + dec + R * loop));
}

static double lr_at(long step, long total, bool pre, double peak_override = 0.0) {
    long warm = pre ? 2000 : 1000;
    double peak = peak_override > 0.0 ? peak_override : (pre ? 1e-3 : 3e-4), lo = pre ? 1e-5 : 1e-6;
    if (step < warm)
        return peak * (step + 1) / warm;
    double p = std::min(1.0, (double)(step - warm) / std::max<long>(1, total - warm));
    return lo + (peak - lo) * 0.5 * (1 + std::cos(3.14159265358979 * p));
}

struct Ckpt {
    char magic[8] = {'l', 'o', 'g', 'o', 's', 0, 0, 0};
    int stage = 0;
    long step = 0, total = 0, adam_t = 0;
    u64 seed = 1;
    long ptotal = 0;
};

static void save_ckpt(
    Model &m, const std::string &path, const Ckpt &h, const std::vector<uint8_t> &hard) {
    std::vector<float> a(m.ptotal), b(m.ptotal), cc(m.ptotal);
    CK(cudaMemcpy(a.data(), m.master, m.ptotal * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), m.m1, m.ptotal * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cc.data(), m.m2, m.ptotal * 4, cudaMemcpyDeviceToHost));
    std::string tmp = path + ".part";
    gzFile f = gzopen(tmp.c_str(), "wb1");
    if (!f)
        fail("cannot write " + path);
    gzwrite(f, &h, sizeof(h));
    gzwrite(f, a.data(), m.ptotal * 4);
    gzwrite(f, b.data(), m.ptotal * 4);
    gzwrite(f, cc.data(), m.ptotal * 4);
    long nh = hard.size();
    gzwrite(f, &nh, 8);
    if (nh)
        gzwrite(f, hard.data(), nh);
    gzclose(f);
    std::filesystem::rename(tmp, path);
}

static Ckpt load_ckpt(Model &m, const std::string &path, bool moments, std::vector<uint8_t> *hard) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f)
        fail("cannot read " + path);
    Ckpt h;
    gzread(f, &h, sizeof(h));
    bool ok_magic = memcmp(h.magic, "logos\0\0\0", 8) == 0 || memcmp(h.magic, "LOGOSCK1", 8) == 0;
    if (!ok_magic || h.ptotal != m.ptotal)
        fail("checkpoint does not match this model: " + path);
    memcpy(h.magic, "logos\0\0\0", 8);
    std::vector<float> a(m.ptotal), b(m.ptotal), cc(m.ptotal);
    gzread(f, a.data(), m.ptotal * 4);
    gzread(f, b.data(), m.ptotal * 4);
    gzread(f, cc.data(), m.ptotal * 4);
    long nh = 0;
    gzread(f, &nh, 8);
    std::vector<uint8_t> hd(std::max<long>(nh, 0));
    if (nh > 0)
        gzread(f, hd.data(), nh);
    gzclose(f);
    m.host_master = a;
    m.upload_params();
    if (moments) {
        CK(cudaMemcpy(m.m1, b.data(), m.ptotal * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(m.m2, cc.data(), m.ptotal * 4, cudaMemcpyHostToDevice));
        if (hard && hd.size() == hard->size())
            *hard = hd;
    }
    return h;
}

struct Args {
    std::map<std::string, std::string> kv;
    std::string cmd;
    Args(int argc, char **argv) {
        if (argc > 1)
            cmd = argv[1];
        for (int i = 2; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0)
                fail("unexpected argument " + k);
            if (i + 1 >= argc)
                fail("missing value for " + k);
            kv[k.substr(2)] = argv[++i];
        }
    }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    std::string need(const std::string &k) const {
        auto it = kv.find(k);
        if (it == kv.end())
            fail("--" + k + " is required");
        return it->second;
    }
};

struct EvalTotals {
    double tp = 0, fp = 0, fn = 0, ctp = 0, cfp = 0, cfn = 0, next_ok = 0, next_n = 0, rel_ok = 0,
           rel_n = 0;
    double jumps = 0, length = 0, clean = 0, lines = 0, recall = 0, recall_n = 0, recon = 0,
           recon_n = 0;
    void json(FILE *f) const {
        auto field = [&](const char *k, double a, double b, bool comma) {
            if (comma)
                fputc(',', f);
            if (b > 0 && std::isfinite(a / b))
                fprintf(f, "\"%s\":%.6g", k, a / b);
            else
                fprintf(f, "\"%s\":null", k);
        };
        fputc('{', f);
        field("sheet_dice", 2 * tp, 2 * tp + fp + fn, false);
        field("contact_f1", 2 * ctp, 2 * ctp + cfp + cfn, true);
        field("next_turn_accuracy", next_ok, next_n, true);
        fprintf(f, ",\"next_turn_pairs\":%.0f", next_n);
        field("relative_accuracy", rel_ok, rel_n, true);
        fprintf(f, ",\"relative_pairs\":%.0f", rel_n);
        field("winding_jumps_per_1000_voxels", 1000 * jumps, length, true);
        field("lines_without_jumps", clean, lines, true);
        fprintf(f, ",\"lines\":%.0f", lines);
        field("sheet_recall_on_lines", recall, recall_n, true);
        field("recon_mse", recon, recon_n, true);
        fputc('}', f);
    }
};

static float sample_turn(
    const std::vector<float> &h4, const Cfg &c, int b, const float *p, int ch) {
    long idx[8];
    float w[8];
    corners(c, p, idx, w);
    float s = 0;
    for (int k = 0; k < 8; ++k)
        s += w[k] * h4[((long)b * c.V() + idx[k]) * 4 + ch];
    return s;
}

static void eval_batch(const Cfg &c, int maxpairs, const Dataset &data, const std::vector<int> &ids,
    const HostBatch &hb, const std::vector<float> &h4, const std::vector<float> &metrics,
    std::map<int, EvalTotals> &bysrc, EvalTotals &all) {
    const long V = c.V();
    bool any_recon = false;
    for (int b = 0; b < (int)ids.size(); ++b)
        any_recon |= hb.meta[b].recon != 0;
    if (any_recon) {
        all.recon += metrics[0];
        all.recon_n += 1;
        bysrc[0].recon += metrics[0];
        bysrc[0].recon_n += 1;
    }
    for (int b = 0; b < (int)ids.size(); ++b) {
        const Record &R = data.recs[ids[b]];
        EvalTotals &e = bysrc[R.source];
        EvalTotals local;
        if (R.group == 0 || R.group == 2 || R.group == 4) {
            for (long i = 0; i < V; ++i) {
                uint8_t lb = hb.lab[b * V + i];
                const float *o = &h4[((long)b * V + i) * 4];
                if (lb & 1) {
                    bool y = lb & 2, p = o[0] > 0;
                    local.tp += y && p;
                    local.fp += !y && p;
                    local.fn += y && !p;
                }
                if (lb & 4) {
                    bool y = lb & 8, p = o[2] > 0;
                    local.ctp += y && p;
                    local.cfp += !y && p;
                    local.cfn += y && !p;
                }
            }
        }
        const Pair *ps = hb.pairs.data() + (long)b * maxpairs;
        for (int k = 0; k < hb.meta[b].npairs; ++k) {
            const Pair &p = ps[k];
            float ua = sample_turn(h4, c, b, p.a, 1) - p.anga,
                  ub = sample_turn(h4, c, b, p.b, 1) - p.angb;
            if (p.kind == 1) {
                local.next_ok += std::lround(ub - ua) == 1;
                local.next_n += 1;
            }
            if (p.kind == 2) {
                local.rel_ok += std::lround(ua - ub) == std::lround(p.target);
                local.rel_n += 1;
            }
        }
        if ((R.source >= 4 && R.source <= 6) || (R.group == 1 && R.split)) {
            std::vector<unsigned char> buf;
            data.read(ids[b], buf);
            Slab s;
            decode(R, buf, s);
            float ref = angle_term(s, 31.5f, 63.5f, 63.5f, 1, 0.f) * 6.2831853f;
            std::map<int, std::vector<int>> sets;
            for (int i = 0; i < (int)s.pts.size(); ++i)
                if (s.pts[i].kind == 0)
                    sets[s.pts[i].set].push_back(i);
            for (auto &kv : sets) {
                int jumps = 0;
                double len = 0;
                long prev = 0;
                for (size_t j = 0; j < kv.second.size(); ++j) {
                    const auto &q = s.pts[kv.second[j]];
                    float p3[3] = {q.z, q.y, q.x};
                    float u =
                        sample_turn(h4, c, b, p3, 1) - angle_term(s, q.z, q.y, q.x, R.curl, ref);
                    long ru = std::lround(u);
                    float sh = sample_turn(h4, c, b, p3, 0);
                    local.recall += sh > 0;
                    local.recall_n += 1;
                    if (j) {
                        const auto &pq = s.pts[kv.second[j - 1]];
                        len += std::sqrt((q.z - pq.z) * (q.z - pq.z) + (q.y - pq.y) * (q.y - pq.y) +
                            (q.x - pq.x) * (q.x - pq.x));
                        jumps += ru != prev;
                    }
                    prev = ru;
                }
                if (kv.second.size() > 1) {
                    local.jumps += jumps;
                    local.length += len;
                    local.clean += jumps == 0;
                    local.lines += 1;
                }
            }
        }
        for (EvalTotals *t : {&e, &all}) {
            t->tp += local.tp;
            t->fp += local.fp;
            t->fn += local.fn;
            t->ctp += local.ctp;
            t->cfp += local.cfp;
            t->cfn += local.cfn;
            t->next_ok += local.next_ok;
            t->next_n += local.next_n;
            t->rel_ok += local.rel_ok;
            t->rel_n += local.rel_n;
            t->jumps += local.jumps;
            t->length += local.length;
            t->clean += local.clean;
            t->lines += local.lines;
            t->recall += local.recall;
            t->recall_n += local.recall_n;
            t->recon += local.recon;
            t->recon_n += local.recon_n;
        }
    }
}

static void run_eval(Model &m, const Dataset &data, const std::vector<int> &ids, int L, u64 seed,
    Pool &pool, FILE *out, long step, const char *what) {
    Sampler dummy(data);
    Weights lw;
    std::map<int, EvalTotals> bysrc;
    EvalTotals all;
    Prepared pr;
    std::vector<float> h4((long)m.c.B * m.c.V() * 4), metrics(64);
    for (size_t i = 0; i < ids.size(); i += m.c.B) {
        std::vector<int> batch;
        for (int b = 0; b < m.c.B; ++b)
            batch.push_back(ids[std::min(i + b, ids.size() - 1)]);
        prepare_batch(m.c, data, dummy, false, seed, i, lw, m, pr, pool, false, &batch);
        for (int b = 0; b < m.c.B; ++b)
            if (data.recs[batch[b]].layout == 1) {
                pr.hb.meta[b].recon = 1;
            }
        upload_batch(m, pr.hb, 0);
        launch_step(m, L, 0, 0, 0, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h4.data(), m.h4, h4.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(metrics.data(), m.metrics, 256, cudaMemcpyDeviceToHost));
        std::vector<int> used(
            batch.begin(), batch.begin() + std::min<size_t>(m.c.B, ids.size() - i));
        eval_batch(m.c, m.maxpairs, data, used, pr.hb, h4, metrics, bysrc, all);
    }
    fprintf(out, "{\"step\":%ld,\"data\":\"%s\",\"slabs\":%zu,\"loops\":%d,\"all\":", step, what,
        ids.size(), L);
    all.json(out);
    fprintf(out, ",\"by_source\":{");
    bool first = true;
    for (auto &kv : bysrc) {
        fprintf(out, "%s\"%d\":", first ? "" : ",", kv.first);
        kv.second.json(out);
        first = false;
    }
    fprintf(out, "}}\n");
    fflush(out);
}

static std::vector<int> fixed_heldout(const Dataset &d, int per_source) {
    std::map<int, std::vector<int>> by;
    for (int id : d.held)
        by[d.recs[id].source].push_back(id);
    std::vector<int> out;
    for (auto &kv : by) {
        auto &v = kv.second;
        int n = std::min<int>(per_source, v.size());
        for (int i = 0; i < n; ++i)
            out.push_back(v[(long)i * v.size() / n]);
    }
    return out;
}

static void tensor_stats(Model &m, FILE *f, const std::vector<float> &before) {
    std::vector<float> w(m.ptotal), g(m.ptotal), a(m.ptotal), b(m.ptotal);
    CK(cudaMemcpy(w.data(), m.master, m.ptotal * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g.data(), m.grad, m.ptotal * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(a.data(), m.m1, m.ptotal * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), m.m2, m.ptotal * 4, cudaMemcpyDeviceToHost));
    fprintf(f, ",\"parameters\":{");
    for (size_t k = 0; k < m.params.size(); ++k) {
        auto &p = m.params[k];
        double sw = 0, sg = 0, su = 0, s1 = 0, s2 = 0;
        for (long i = p.off; i < p.off + p.n; ++i) {
            sw += (double)w[i] * w[i];
            sg += (double)g[i] * g[i];
            double d = w[i] - before[i];
            su += d * d;
            s1 += (double)a[i] * a[i];
            s2 += (double)b[i] * b[i];
        }
        double rw = std::sqrt(sw / p.n);
        fprintf(f,
            "%s\"%s\":{\"weight_rms\":%.4g,\"gradient_rms\":%.4g,\"update_to_weight\":%.4g,\"m1_"
            "rms\":%.4g,\"m2_rms\":%.4g}",
            k ? "," : "", p.name.c_str(), rw, std::sqrt(sg / p.n),
            std::sqrt(su / p.n) / std::max(rw, 1e-12), std::sqrt(s1 / p.n), std::sqrt(s2 / p.n));
    }
    fprintf(f, "}");
    auto act = [&](const char *name, const void *ptr, long n, bool isbf) {
        long k = std::min<long>(n, 1L << 20);
        std::vector<float> v(k);
        if (isbf) {
            std::vector<bf16> h(k);
            CK(cudaMemcpy(h.data(), ptr, k * 2, cudaMemcpyDeviceToHost));
            for (long i = 0; i < k; ++i)
                v[i] = __bfloat162float(h[i]);
        } else
            CK(cudaMemcpy(v.data(), ptr, k * 4, cudaMemcpyDeviceToHost));
        double s = 0, mx = 0, dead = 0;
        for (float x : v) {
            s += (double)x * x;
            mx = std::max(mx, (double)std::fabs(x));
            dead += x <= 0;
        }
        fprintf(f, ",\"%s\":{\"rms\":%.4g,\"max\":%.4g,\"nonpositive_share\":%.4g}", name,
            std::sqrt(s / k), mx, dead / k);
    };
    fprintf(f, ",\"activations\":{\"_\":0");
    const long BT = m.c.B * m.c.T();
    act("stem_z0b", m.z0b, m.c.B * m.c.V() * 32, true);
    act("stem_z2b", m.z2b, m.c.B * m.c.V2() * 128, true);
    act("stem_tokens", m.S, BT * 256, false);
    act("trunk_out", m.P, BT * 256, false);
    act("latents", m.Lt, m.c.B * m.c.latents * 256, false);
    act("dec_z2c", m.z2c, m.c.B * m.c.V2() * 128, true);
    act("dec_z0c", m.z0c, m.c.B * m.c.V() * 32, true);
    act("heads", m.h4, m.c.B * m.c.V() * 4, false);
    act("confidence", m.conf, BT, false);
    for (int k = 0; k < 4; ++k) {
        std::string n = "block" + std::to_string(k) + "_gates";
        act(n.c_str(), m.blk[k].gates, BT * 1024, true);
    }
    fprintf(f, "}");
}

static volatile sig_atomic_t g_stop = 0;

static void on_stop(int) { g_stop = 1; }

static size_t estimate_model_bytes(
    const Cfg &c, int G, long ptotal, int maxpairs, int maxcsr, int maxent) {
    const size_t B = (size_t)c.B, V = (size_t)c.V(), V1 = (size_t)c.V1(), V2 = (size_t)c.V2(),
                 T = (size_t)c.T(), BT = B * T, Lc = (size_t)c.latents, BL = B * Lc,
                 Cn = (size_t)c.Cn(), Cp = (size_t)c.Cp(), Hd = (size_t)c.hidden;
    const size_t nchmax = std::max<size_t>((T + 127) / 128, (Cn + 127) / 128),
                 nqtmax = (T + 63) / 64;
    const size_t scratch_n = std::max<size_t>(24UL << 20,
        std::max<size_t>((size_t)G * 256UL * 256UL, B * c.TH() * c.TW() * 512UL + 4096UL));
    size_t bytes = (size_t)ptotal * 19 + scratch_n * 4 + (1UL << 20);
    bytes += B *
        (V * 5 + T + (size_t)c.C * c.C * 8 + (size_t)c.D * 8 + 16 + sizeof(Meta) +
            (size_t)maxpairs * (sizeof(Pair) + 12) + (T + 1) * 4 + (size_t)maxcsr * 8 + 4 +
            (size_t)maxent * 8);
    bytes += B * (V * 748 + V1 * 1432 + V2 * 2840);
    bytes += BT * (256UL * 4 * 27 + 4 * 56 + 4 * 96 + 4 * 512 * 4 + 4 * Hd + 4 * 13);
    bytes += BT *
        (2 *
            (128 + 256 + 4 * (7 * 256 + 768 + 512 + 1024 + 3 * Hd) + 256 + 4 * 256 + 768 + 512 +
                1024 + 2 * Hd + 256));
    bytes += BL *
        (256UL * 4 * 11 + 4 * 24 + (3 * nchmax + 2 * nqtmax) * 256 * 4 + nchmax * 64 +
            2 * (4 * (3 * 256 + 512) + 4 * 256 + 512));
    bytes += (size_t)c.retain * (BT + BL) * 256 * 4;
    bytes += B * Cn * (256 * 12 + 512 * 4 + 8 + 2 * (2 * 256 + 1024)) + B * Cp * 3 * 128 * 2;
    return bytes;
}

static int auto_select_batch(int retain, int max_cap = 60) {
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    int G = device_coop_blocks();
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const size_t budget = (size_t)((double)free_b * 0.80);
    Model probe;
    probe.declare();
    int best = 1;
    for (int b = 1; b <= max_cap; ++b) {
        if (b > 2 && (b & 1))
            continue;
        Cfg cfg;
        cfg.B = b;
        cfg.retain = retain;
        size_t need =
            estimate_model_bytes(cfg, G, probe.ptotal, probe.maxpairs, probe.maxcsr, probe.maxent);
        if (need <= budget)
            best = b;
        else
            break;
    }
    printf("gpu %s (sm_%d%d, %d SMs, %d coop blocks, %.1f/%.1f GiB free) -> auto batch=%d\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, G,
        (double)free_b / (1024.0 * 1024.0 * 1024.0), (double)total_b / (1024.0 * 1024.0 * 1024.0),
        best);
    return best;
}

static int cmd_train(const Args &a) {
    const std::string stage = a.need("stage");
    const bool pre = stage == "pre";
    if (!pre && stage != "mid")
        fail("--stage must be pre or mid");
    const std::string out = a.need("out");
    std::filesystem::create_directories(out);
    const long total = std::stol(a.get("steps", pre ? "600000" : "400000"));
    const u64 seed = std::stoull(a.get("seed", "1"));
    const int retain = std::max(1, std::stoi(a.get("retain", "8")));
    const int loops_min = std::max(1, std::stoi(a.get("loops-min", "4")));
    const int loops_max = std::max(loops_min, std::stoi(a.get("loops-max", "8")));
    const float w_abs = std::stof(a.get("w-absolute", "0"));
    const float nogeom_drop = std::stof(a.get("nogeom-drop", "0.15"));
    const long mine_every = std::stol(a.get("mine-every", "5000"));
    const long mine_cap = std::max<long>(1, std::stol(a.get("mine-cap", "500")));
    launch_setup();
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    const std::string batch_arg = a.get("batch", "auto");
    const int batch_sz =
        batch_arg == "auto" ? auto_select_batch(retain) : std::max(1, std::stoi(batch_arg));
    const std::string accum_arg = a.get("accum", "auto");
    const int accum = accum_arg == "auto" ? std::max(1, (16 + batch_sz - 1) / batch_sz)
                                          : std::max(1, std::stoi(accum_arg));
    double peak_lr = std::stod(a.get("lr", "0"));
    if (peak_lr <= 0.0) {
        double base = pre ? 1e-3 : 3e-4;
        double scale = std::min(2.5, std::max(1.0, std::sqrt((double)(batch_sz * accum) / 8.0)));
        peak_lr = base * scale;
    }
    Model m;
    m.c.B = batch_sz;
    m.c.retain = retain;
    m.allocate();
    m.init(seed);
    m.upload_params();
    m.build(pre ? 0.f : 0.1f / (float)accum);
    printf("config batch=%d accum=%d effective_batch=%d lr=%.2e retain=%d loops=%d..%d "
           "w_absolute=%.3f nogeom_drop=%.2f\n",
        batch_sz, accum, batch_sz * accum, peak_lr, retain, loops_min, loops_max, w_abs,
        nogeom_drop);
    Dataset data(a.need("shards"), a.get("all-labels", "0") == "1");
    std::unique_ptr<Dataset> val;
    if (!a.get("val").empty())
        val = std::make_unique<Dataset>(a.get("val"));
    Sampler smp(data);
    {
        std::string sh = a.get("shares", "0.4,0.25,0.15,0.2,0");
        double v[5] = {0, 0, 0, 0, 0}, t = 0;
        if (sscanf(sh.c_str(), "%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4]) < 4)
            fail("--shares needs 4 or 5 numbers");
        for (int g = 0; g < 5; ++g)
            t += v[g];
        for (int g = 0; g < 5; ++g) {
            smp.share[g] = v[g] / t;
            if (!pre && smp.share[g] > 0 && data.train[g].empty())
                fail("a source with a share has no training records");
        }
        printf("shares instances=%.3f points=%.3f fibres=%.3f unlabelled=%.3f kaggle=%.3f\n",
            smp.share[0], smp.share[1], smp.share[2], smp.share[3], smp.share[4]);
    }
    Ckpt h;
    h.stage = pre ? 0 : 1;
    h.total = total;
    h.seed = seed;
    h.ptotal = m.ptotal;
    if (!a.get("resume").empty()) {
        Ckpt peek;
        {
            gzFile f = gzopen(a.get("resume").c_str(), "rb");
            if (!f)
                fail("cannot read " + a.get("resume"));
            gzread(f, &peek, sizeof(peek));
            gzclose(f);
        }
        Ckpt r = load_ckpt(m, a.get("resume"), peek.stage == h.stage, &smp.hard);
        if (r.stage == h.stage) {
            h = r;
            smp.rebuild();
        } else {
            CK(cudaMemset(m.m1, 0, m.ptotal * 4));
            CK(cudaMemset(m.m2, 0, m.ptotal * 4));
        }
    }
    FILE *tj = fopen((out + "/train.jsonl").c_str(), "a"),
         *vj = fopen((out + "/val.jsonl").c_str(), "a");
    std::string gcmd =
        "nvidia-smi "
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,"
        "clocks.sm,clocks.mem --format=csv,noheader,nounits -l 60 >> '" +
        out + "/gpu.csv' 2>/dev/null & echo $!";
    FILE *gp = popen(gcmd.c_str(), "r");
    int gpid = 0;
    if (gp) {
        if (fscanf(gp, "%d", &gpid) != 1)
            gpid = 0;
        pclose(gp);
    }
    Pool pool(std::max(4, (int)std::thread::hardware_concurrency() - 2));
    Weights lw;
    lw.absolute = w_abs;
    lw.nogeom_drop = nogeom_drop;
    lw.accum = accum;
    if (pre) {
        lw = Weights{1, 0, 0, 0, 0, 0, 0, nogeom_drop, accum};
    }
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    float *ring = m.alloc<float>((long)128 * accum * 64);
    unsigned long long *clocks = m.alloc<unsigned long long>(16);
    constexpr int Slots = 6;
    constexpr int Prefetch = 4;
    std::vector<Prepared> slots(Slots);
    std::vector<std::future<void>> fut(Slots);
    auto start_prep = [&](long micro) {
        int sl = micro % Slots;
        fut[sl] = std::async(std::launch::async, [&, micro, sl] {
            prepare_batch(
                m.c, data, smp, pre, h.seed, micro, lw, m, slots[sl], pool, true, nullptr);
        });
    };
    const long first = h.step;
    long log_start = first;
    const long total_micro = total * accum;
    for (long mi = first * accum; mi < std::min(total_micro, first * accum + Prefetch); ++mi)
        start_prep(mi);
    std::vector<int> valids;
    if (val)
        valids = fixed_heldout(*val, 64);
    std::vector<int> Ls(100), groups(5, 0), hardc(5, 0);
    double host_s = 0, wait_s = 0;
    auto tw = std::chrono::steady_clock::now();
    std::vector<float> before;
    signal(SIGTERM, on_stop);
    signal(SIGINT, on_stop);
    for (long it = first; it < total; ++it) {
        Rng lr_(h.seed * 31 + it);
        int L = loops_min + (int)(lr_.next() % (u64)(loops_max - loops_min + 1));
        Ls[it % 100] = L;
        bool stats = (it + 1) % 1000 == 0;
        if (stats) {
            CK(cudaStreamSynchronize(st));
            before.resize(m.ptotal);
            CK(cudaMemcpy(before.data(), m.master, m.ptotal * 4, cudaMemcpyDeviceToHost));
        }
        float sc[8] = {(float)lr_at(it, total, pre, peak_lr), (float)(h.adam_t + 1),
            L <= m.c.retain ? 1.f : -1.f, 0, 0, 0, 0, 0};
        CK(cudaMemcpyAsync(m.scal, sc, 32, cudaMemcpyHostToDevice, st));
        for (int mb = 0; mb < accum; ++mb) {
            long mi = it * accum + mb;
            int sl = mi % Slots;
            auto w0 = std::chrono::steady_clock::now();
            fut[sl].get();
            wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            host_s += slots[sl].aug_s;
            for (int g = 0; g < 5; ++g) {
                groups[g] += slots[sl].slabs_per_group[g];
                hardc[g] += slots[sl].hard[g];
            }
            upload_batch(m, slots[sl].hb, st);
            launch_step(m, L, 1, mb == accum - 1 ? 1 : 0, st, clocks, mb == 0 ? 1 : 0);
            CK(cudaMemcpyAsync(ring + ((it % 128) * accum + mb) * 64, m.metrics, 256,
                cudaMemcpyDeviceToDevice, st));
            if (mi + Prefetch < total_micro)
                start_prep(mi + Prefetch);
        }
        h.adam_t += 1;
        h.step = it + 1;
        if ((it + 1) % 100 == 0 || it + 1 == total) {
            CK(cudaStreamSynchronize(st));
            double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
            tw = std::chrono::steady_clock::now();
            std::vector<float> rg((size_t)128 * accum * 64);
            CK(cudaMemcpy(rg.data(), ring, rg.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> ck(16);
            CK(cudaMemcpy(ck.data(), clocks, 128, cudaMemcpyDeviceToHost));
            CK(cudaMemset(clocks, 0, 128));
            long n = it + 1 - log_start;
            log_start = it + 1;
            double sum[64] = {0};
            int nonfinite = 0, clipped = 0;
            for (long k = it + 1 - n; k <= it; ++k) {
                for (int mb = 0; mb < accum; ++mb) {
                    const float *row = &rg[((k % 128) * accum + mb) * 64];
                    for (int j = 0; j < 16; ++j)
                        sum[j] += row[j];
                }
                const float *last_row = &rg[((k % 128) * accum + accum - 1) * 64];
                sum[40] += last_row[40];
                sum[41] += last_row[41];
                nonfinite += last_row[42] > 0;
                clipped += last_row[41] < 1.f;
            }
            double flops = 0;
            int lmin = 99, lmax = 0;
            double lsum = 0;
            for (long k = it + 1 - n; k <= it; ++k) {
                int L_ = Ls[k % 100];
                flops += (double)accum * step_flops(m.c, L_, m.c.retain);
                lmin = std::min(lmin, L_);
                lmax = std::max(lmax, L_);
                lsum += L_;
            }
            double obj = (sum[0] + sum[1] + sum[2] + sum[3] + sum[8]) / n;
            fprintf(tj,
                "{\"step\":%ld,\"stage\":\"%s\",\"data_fingerprint\":%llu,\"objective\":%.6g,"
                "\"losses\":{\"recon\":%.6g,"
                "\"sheet\":%.6g,\"contact\":%.6g,\"pairs\":%.6g,\"confidence\":%.6g},\"sheet_tp\":%"
                ".0f,\"sheet_fp\":%.0f,"
                "\"sheet_fn\":%.0f,\"gradient_norm\":%.6g,\"clip_factor\":%.6g,\"clipped_steps\":%"
                "d,\"nonfinite_steps\":%d,"
                "\"learning_rate\":%.6g,\"loops_mean\":%.3f,\"loops_min\":%d,\"loops_max\":%d,"
                "\"steps_per_second\":%.4f,"
                "\"algorithmic_tflops\":%.3f,\"host_seconds_per_batch\":%.4f,\"data_wait_seconds_"
                "per_step\":%.4f,"
                "\"slabs\":{\"instances\":%d,\"winding_points\":%d,\"fibres\":%d,\"unlabelled\":%d,"
                "\"kaggle\":%d},\"hard_slabs\":%d",
                it + 1, stage.c_str(), (unsigned long long)data.fingerprint, obj, sum[0] / n,
                sum[1] / n, sum[2] / n, sum[3] / n, sum[8] / n, sum[5], sum[6], sum[7], sum[40] / n,
                sum[41] / n, clipped, nonfinite, lr_at(it, total, pre, peak_lr), lsum / n, lmin,
                lmax, n / secs, flops / secs / 1e12, host_s / (n * accum), wait_s / n, groups[0],
                groups[1], groups[2], groups[3], groups[4],
                hardc[0] + hardc[1] + hardc[2] + hardc[3] + hardc[4]);
            double ctot = 0;
            for (int s = 0; s < 11; ++s)
                ctot += ck[s];
            const char *segn[11] = {"stem", "loop_forward", "decoder", "losses", "decoder_backward",
                "loop_backward", "context_backward", "stem_backward", "optimizer", "save",
                "restore"};
            fprintf(tj, ",\"time_share\":{");
            for (int s = 0; s < 11; ++s)
                fprintf(tj, "%s\"%s\":%.4f", s ? "," : "", segn[s], ctot ? ck[s] / ctot : 0.0);
            fprintf(tj, "}");
            if (stats)
                tensor_stats(m, tj, before);
            fprintf(tj, "}\n");
            fflush(tj);
            printf("step %ld obj %.5f recon %.5f sheet %.5f pairs %.5f gnorm %.3f lr %.2e %.2f "
                   "steps/s %.1f TFLOP/s wait %.3f\n",
                it + 1, obj, sum[0] / n, sum[1] / n, sum[3] / n, sum[40] / n,
                lr_at(it, total, pre, peak_lr), n / secs, flops / secs / 1e12, wait_s / n);
            fflush(stdout);
            host_s = wait_s = 0;
            std::fill(groups.begin(), groups.end(), 0);
            std::fill(hardc.begin(), hardc.end(), 0);
            if (nonfinite) {
                save_ckpt(m, out + "/last-good.ckpt", h, smp.hard);
                fail("non-finite gradient norm in the last 100 steps; the update was skipped and "
                     "last-good.ckpt written");
            }
        }
        if (!pre && mine_every > 0 && (it + 1) % mine_every == 0) {
            CK(cudaStreamSynchronize(st));
            for (int k = 0; k < Slots; ++k)
                if (fut[k].valid())
                    fut[k].wait();
            std::vector<std::pair<float, int>> score[5];
            std::vector<float> cf(m.c.B * m.c.T());
            Prepared pr;
            Sampler dummy(data);
            Weights eval_lw = lw;
            eval_lw.accum = 1;
            for (int g = 0; g < 5; ++g) {
                std::vector<int> pool_ids;
                const auto &all = data.train[g];
                long stride = std::max<long>(1, (long)all.size() / mine_cap),
                     start = ((it + 1) / mine_every) % stride;
                for (long k = start; k < (long)all.size(); k += stride)
                    pool_ids.push_back(all[k]);
                for (size_t i = 0; g != 3 && smp.share[g] > 0 && i < pool_ids.size(); i += m.c.B) {
                    std::vector<int> batch;
                    for (int b = 0; b < m.c.B; ++b)
                        batch.push_back(pool_ids[std::min(i + b, pool_ids.size() - 1)]);
                    prepare_batch(
                        m.c, data, dummy, false, h.seed, i, eval_lw, m, pr, pool, false, &batch);
                    upload_batch(m, pr.hb, 0);
                    launch_step(m, std::min(6, loops_max), 0, 0, 0, nullptr, 0);
                    CK(cudaDeviceSynchronize());
                    CK(cudaMemcpy(cf.data(), m.conf, cf.size() * 4, cudaMemcpyDeviceToHost));
                    for (int b = 0; b < m.c.B && i + b < pool_ids.size(); ++b) {
                        double s = 0;
                        for (long t = 0; t < m.c.T(); ++t)
                            s += cf[b * m.c.T() + t];
                        score[g].push_back({(float)(s / m.c.T()), batch[b]});
                    }
                }
            }
            std::fill(smp.hard.begin(), smp.hard.end(), 0);
            for (int g = 0; g < 5; ++g) {
                std::sort(score[g].begin(), score[g].end(), [](auto &x, auto &y) {
                    return x.first > y.first || (x.first == y.first && x.second < y.second);
                });
                for (size_t i = 0; i < (score[g].size() + 4) / 5; ++i)
                    smp.hard[score[g][i].second] = 1;
            }
            smp.rebuild();
            if (!(val && ((it + 1) % 5000 == 0 || it + 1 == total)))
                for (long mi = (it + 1) * accum;
                    mi < std::min(total_micro, (it + 1) * accum + Prefetch); ++mi)
                    start_prep(mi);
            fprintf(tj,
                "{\"step\":%ld,\"mining\":{\"labelled_slabs\":%zu,\"top_share\":0.2,\"multiplier\":"
                "3}}\n",
                it + 1, score[0].size() + score[1].size() + score[2].size() + score[4].size());
            fflush(tj);
        }
        if (val && ((it + 1) % 5000 == 0 || it + 1 == total)) {
            CK(cudaStreamSynchronize(st));
            for (int k = 0; k < Slots; ++k)
                if (fut[k].valid())
                    fut[k].wait();
            run_eval(
                m, *val, valids, 8, 12345, pool, vj, it + 1, "fixed held-out validation records");
            for (long mi = (it + 1) * accum;
                mi < std::min(total_micro, (it + 1) * accum + Prefetch); ++mi)
                start_prep(mi);
        }
        const bool stop = g_stop != 0;
        if ((it + 1) % 5000 == 0 || it + 1 == total || stop) {
            CK(cudaStreamSynchronize(st));
            save_ckpt(m, out + "/step-" + std::to_string(it + 1) + ".ckpt", h, smp.hard);
        }
        if (stop) {
            fprintf(stderr, "stopped at step %ld; step-%ld.ckpt written\n", it + 1, it + 1);
            break;
        }
    }
    CK(cudaStreamSynchronize(st));
    for (int k = 0; k < Slots; ++k)
        if (fut[k].valid())
            fut[k].wait();
    if (gpid > 0)
        kill(gpid, SIGTERM);
    fclose(tj);
    fclose(vj);
    return 0;
}

static int cmd_eval(const Args &a) {
    launch_setup();
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    Model m;
    m.allocate();
    m.init(1);
    m.upload_params();
    m.build(0.1f);
    load_ckpt(m, a.need("ckpt"), false, nullptr);
    Dataset data(a.need("shards"));
    Pool pool(std::max(4, (int)std::thread::hardware_concurrency() - 2));
    FILE *f = fopen(a.need("out").c_str(), "w");
    int per = std::stoi(a.get("per-source", "100000"));
    run_eval(m, data, fixed_heldout(data, per), std::stoi(a.get("loops", "8")), 12345, pool, f, 0,
        "held-out records");
    fclose(f);
    return 0;
}

static int cmd_dump(const Args &a) {
    g_unmasked = true;
    launch_setup();
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    Model m;
    m.allocate();
    m.init(1);
    m.upload_params();
    m.build(0.1f);
    if (!a.get("ckpt").empty())
        load_ckpt(m, a.get("ckpt"), false, nullptr);
    Dataset data(a.need("shards"));
    Pool pool(8);
    std::string out = a.need("out");
    std::filesystem::create_directories(out);
    int per = std::stoi(a.get("per-source", "4"));
    auto ids = fixed_heldout(data, per);
    Sampler dummy(data);
    Weights lw;
    Prepared pr;
    std::vector<float> h4((long)m.c.B * m.c.V() * 4);
    for (size_t i = 0; i < ids.size(); i += m.c.B) {
        std::vector<int> batch;
        for (int b = 0; b < m.c.B; ++b)
            batch.push_back(ids[std::min(i + b, ids.size() - 1)]);
        prepare_batch(m.c, data, dummy, false, 7, i, lw, m, pr, pool, false, &batch);
        upload_batch(m, pr.hb, 0);
        launch_step(m, std::stoi(a.get("loops", "8")), 0, 0, 0, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h4.data(), m.h4, h4.size() * 4, cudaMemcpyDeviceToHost));
        for (int b = 0; b < m.c.B && i + b < ids.size(); ++b) {
            const Record &R = data.recs[batch[b]];
            std::vector<unsigned char> buf;
            data.read(batch[b], buf);
            Slab s;
            decode(R, buf, s);
            float ref = angle_term(s, 31.5f, 63.5f, 63.5f, 1, 0.f) * 6.2831853f;
            char name[256];
            snprintf(
                name, sizeof(name), "%s/slab-%04zu-source%d.bin", out.c_str(), i + b, R.source);
            FILE *f = fopen(name, "wb");
            int hdr[4] = {64, 128, 128, R.source};
            fwrite(hdr, 4, 4, f);
            const long V = m.c.V();
            std::vector<float> ct(V), sh(V), u(V);
            for (long v = 0; v < V; ++v) {
                ct[v] = __bfloat162float(pr.hb.ctt[b * V + v]);
                sh[v] = 1.f / (1.f + std::exp(-h4[((long)b * V + v) * 4]));
                int x = v % 128, y = (v / 128) % 128, z = v / (128 * 128);
                u[v] = h4[((long)b * V + v) * 4 + 1] - angle_term(s, z, y, x, R.curl, ref);
            }
            fwrite(ct.data(), 4, V, f);
            fwrite(sh.data(), 4, V, f);
            fwrite(u.data(), 4, V, f);
            int np = s.pts.size();
            fwrite(&np, 4, 1, f);
            for (auto &p : s.pts)
                fwrite(&p, sizeof(p), 1, f);
            std::vector<uint8_t> lab(V, 255);
            if ((long)s.sheet.size() == V)
                lab = s.sheet;
            fwrite(lab.data(), 1, V, f);
            float tail[6] = {R.origin[0], R.origin[1], R.origin[2], R.voxel, (float)R.curl, ref};
            fwrite(tail, 4, 6, f);
            std::vector<float> ctr(128, 0.f);
            if (s.centre.size() == 128)
                ctr = s.centre;
            fwrite(ctr.data(), 4, 128, f);
            fclose(f);
        }
    }
    printf("dumped %zu slabs to %s\n", ids.size(), out.c_str());
    return 0;
}

static int cmd_bench(const Args &a) {
    launch_setup();
    CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    const int retain = std::max(1, std::stoi(a.get("retain", "8")));
    const std::string batch_arg = a.get("batch", "auto");
    Model m;
    m.c.B = batch_arg == "auto" ? auto_select_batch(retain) : std::max(1, std::stoi(batch_arg));
    m.c.retain = retain;
    m.allocate();
    m.init(1);
    m.upload_params();
    const bool pre = a.get("stage", "pre") == "pre";
    m.build(pre ? 0.f : 0.1f);
    Dataset data(a.need("shards"));
    Sampler smp(data);
    Pool pool(std::max(4, (int)std::thread::hardware_concurrency() - 2));
    Weights lw;
    if (pre)
        lw = Weights{1, 0, 0, 0, 0, 0, 0, 0.15f, 1};
    int steps = std::stoi(a.get("steps", "40"));
    Prepared pr;
    unsigned long long *clocks = m.alloc<unsigned long long>(16);
    double tot = 0, fl = 0;
    for (int it = 0; it < steps; ++it) {
        prepare_batch(m.c, data, smp, pre, 3, it, lw, m, pr, pool, true, nullptr);
        upload_batch(m, pr.hb, 0);
        float sc[8] = {1e-4f, (float)(it + 1), 1.f, 0, 0, 0, 0, 0};
        CK(cudaMemcpy(m.scal, sc, 32, cudaMemcpyHostToDevice));
        int L = std::stoi(a.get("loops", "8"));
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        launch_step(m, L, 1, 1, 0, it >= 3 ? clocks : nullptr);
        CK(cudaDeviceSynchronize());
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (it >= 3) {
            tot += s;
            fl += step_flops(m.c, L, m.c.retain);
        }
        if (it < 3 || it % 10 == 0)
            printf("bench step %d B=%d L=%d %.3f s host %.3f s\n", it, m.c.B, L, s, pr.aug_s);
    }
    std::vector<unsigned long long> ck(16);
    CK(cudaMemcpy(ck.data(), clocks, 128, cudaMemcpyDeviceToHost));
    double ctot = 0;
    for (int s = 0; s < 11; ++s)
        ctot += ck[s];
    printf("bench %s B=%d: %.3f steps/s (%.2f slabs/s) at L=%s, %.1f algorithmic TFLOP/s\n",
        pre ? "pre" : "mid", m.c.B, (steps - 3) / tot, (steps - 3) * m.c.B / tot,
        a.get("loops", "8").c_str(), fl / tot / 1e12);
    const char *segn[11] = {"stem", "loop_forward", "decoder", "losses", "decoder_backward",
        "loop_backward", "context_backward", "stem_backward", "optimizer", "save", "restore"};
    for (int s = 0; s < 11; ++s)
        printf("  %-18s %.3f\n", segn[s], ctot ? ck[s] / ctot : 0.0);
    return 0;
}

int main(int argc, char **argv) {
    Args a(argc, argv);
    if (a.cmd == "test")
        return unit_tests();
    if (a.cmd == "model")
        return test_model();
    if (a.cmd == "train")
        return cmd_train(a);
    if (a.cmd == "eval")
        return cmd_eval(a);
    if (a.cmd == "dump")
        return cmd_dump(a);
    if (a.cmd == "bench")
        return cmd_bench(a);
    if (a.cmd == "kernels") {
        bench_all();
        return 0;
    }
    fprintf(stderr,
        "usage: model test | model | kernels\n"
        "       model train --stage pre|mid --shards DIR [--val DIR] [--resume CKPT] --out DIR "
        "[--steps N] [--seed S]\n"
        "                   [--batch auto|N] [--accum auto|N] [--retain 8] [--loops-min 4] "
        "[--loops-max 8] [--lr F]\n"
        "                   [--w-absolute 0] [--nogeom-drop 0.15] [--mine-every 5000] "
        "[--mine-cap 500]\n"
        "                   [--shares instances,points,fibres,unlabelled,kaggle] [--all-labels "
        "0|1]\n"
        "       model eval --ckpt CKPT --shards DIR --out FILE.json [--loops 8] [--per-source N]\n"
        "       model dump --ckpt CKPT --shards DIR --out DIR [--per-source 4] [--loops 8]\n"
        "       model bench --shards DIR [--stage pre|mid] [--batch auto|N] [--loops 8] [--steps "
        "40]\n");
    return 2;
}
