#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// The depth stem with its norms folded in, in one pass: two 3x3x3 convolutions, attention over depth and max over
// depth.

constant constexpr int TX = 16, TY = 8;
constant constexpr int HX = TX + 2, HY = TY + 2;
constant constexpr int IX = TX + 4, IY = TY + 4;
constant constexpr int C = 16, Z = 17;
constant constexpr int SLICE = HX * HY * C;

#ifdef RAW_INPUT
using Input = uchar;
#else
using Input = half;
#endif

struct Dims {
    uint batch, height, width;
    float attention_bias;
#ifdef TILE_EDGE
    uint full_width, offset_x, right;
#endif
};

kernel void depth_stem(device const Input *image [[buffer(0)]], device const half *first [[buffer(1)]],
                       device const half *first_bias [[buffer(2)]], device const half *second [[buffer(3)]],
                       device const half *second_bias [[buffer(4)]], device const half *attention [[buffer(5)]],
                       device half *out [[buffer(6)]], constant Dims &d [[buffer(7)]],
#ifdef RAW_INPUT
                       device const half *values [[buffer(8)]],
#endif
                       uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half ring[3 * SLICE];
    threadgroup half input[3][IY * IX];

    const int H = d.height, W = d.width;
    const int y0 = group.y * TY;
#ifdef TILE_EDGE
    const int x0 = d.right ? W - TX : 0, pitch = d.full_width, offset_x = d.offset_x;
#else
    const int x0 = group.x * TX, pitch = W, offset_x = 0;
#endif
    device const Input *volume = image + size_t(group.z) * Z * H * pitch;

    auto load_input = [&](int z) {
        threadgroup half *slot = input[(z + 3) % 3];
        for (int i = tid; i < IY * IX; i += 128) {
            int y = y0 - 2 + i / IX, x = x0 - 2 + i % IX;
            if (z >= 0 && z < Z && y >= 0 && y < H && x >= 0 && x < W) {
                Input v = volume[(size_t(z) * H + y) * pitch + x + offset_x];
#ifdef RAW_INPUT
                slot[i] = values[v];
#else
                slot[i] = v;
#endif
            } else {
                slot[i] = half(0);
            }
        }
    };
    auto first_conv = [&](int z) {
        threadgroup half *slot = ring + ((z + 3) % 3) * SLICE;
        for (int p = tid; p < HY * HX; p += 128) {
            int py = p / HX, px = p % HX;
            int y = y0 - 1 + py, x = x0 - 1 + px;
            bool inside = z >= 0 && z < Z && y >= 0 && y < H && x >= 0 && x < W;
            float acc[C];
            for (int c = 0; c < C; c++) acc[c] = float(first_bias[c]);
            if (inside) {
                for (int dz = 0; dz < 3; dz++) {
                    threadgroup half *s = input[(z + dz + 2) % 3];
                    for (int dy = 0; dy < 3; dy++)
                        for (int dx = 0; dx < 3; dx++) {
                            float v = s[(py + dy) * IX + px + dx];
                            device const half *w = first + ((dz * 3 + dy) * 3 + dx) * C;
                            for (int c = 0; c < C; c++) acc[c] += v * float(w[c]);
                        }
                }
            }
            for (int c = 0; c < C; c++) {
                float a = acc[c];
                slot[p * C + c] = inside ? half(a > 0 ? a : 0.01f * a) : half(0);
            }
        }
    };

    load_input(-1);
    load_input(0);
    load_input(1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    first_conv(-1);
    first_conv(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int own_y = tid / TX, own_x = tid % TX;
    float best = -INFINITY, total = 0;
    float pooled[C], peak[C];
    for (int c = 0; c < C; c++) {
        pooled[c] = 0;
        peak[c] = -INFINITY;
    }

    for (int z = 0; z < Z; z++) {
        load_input(z + 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        first_conv(z + 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc[2][2][2];
        for (int r = 0; r < 2; r++)
            for (int b = 0; b < 2; b++)
                for (int o = 0; o < 2; o++) acc[r][b][o] = simdgroup_float8x8(0);
        for (int dz = 0; dz < 3; dz++) {
            threadgroup half *slot = ring + ((z + dz + 2) % 3) * SLICE;
            for (int dy = 0; dy < 3; dy++)
                for (int dx = 0; dx < 3; dx++) {
                    int tap = (dz * 3 + dy) * 3 + dx;
                    for (int k = 0; k < 2; k++) {
                        simdgroup_half8x8 w0, w1;
                        simdgroup_load(w0, second + tap * C * C + k * 8 * C, C);
                        simdgroup_load(w1, second + tap * C * C + k * 8 * C + 8, C);
                        for (int r = 0; r < 2; r++)
                            for (int b = 0; b < 2; b++) {
                                simdgroup_half8x8 a;
                                simdgroup_load(a, slot + ((simd * 2 + r + dy) * HX + b * 8 + dx) * C + k * 8, C);
                                simdgroup_multiply_accumulate(acc[r][b][0], a, w0, acc[r][b][0]);
                                simdgroup_multiply_accumulate(acc[r][b][1], a, w1, acc[r][b][1]);
                            }
                    }
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *result = ring + ((z + 2) % 3) * SLICE;
        for (int r = 0; r < 2; r++)
            for (int b = 0; b < 2; b++)
                for (int o = 0; o < 2; o++) {
                    simdgroup_half8x8 narrow;
                    narrow.thread_elements()[0] = half(acc[r][b][o].thread_elements()[0]);
                    narrow.thread_elements()[1] = half(acc[r][b][o].thread_elements()[1]);
                    simdgroup_store(narrow, result + ((simd * 2 + r) * TX + b * 8) * C + o * 8, C);
                }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float logit = d.attention_bias;
        float value[C];
        for (int c = 0; c < C; c++) {
            float a = float(result[tid * C + c]) + float(second_bias[c]);
            a = a > 0 ? a : 0.01f * a;
            value[c] = a;
            logit += a * float(attention[c]);
            peak[c] = max(peak[c], a);
        }
        float top = max(best, logit), fade = exp(best - top), weight = exp(logit - top);
        total = total * fade + weight;
        for (int c = 0; c < C; c++) pooled[c] = pooled[c] * fade + weight * value[c];
        best = top;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int y = y0 + own_y, x = x0 + own_x;
#ifdef TILE_EDGE
    // The two 3x3 convolutions depend on tile padding only within two columns of the cut.
    if (d.right ? x < W - 2 : x >= 2) return;
#endif
    if (y < H && x < W) {
        device half *o = out + size_t(group.z) * 2 * C * H * W + size_t(y) * W + x;
        for (int c = 0; c < C; c++) {
            o[size_t(c) * H * W] = half(pooled[c] / total);
            o[size_t(C + c) * H * W] = half(peak[c]);
        }
    }
}
