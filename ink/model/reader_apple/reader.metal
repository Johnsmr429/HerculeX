#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr int TX = 16, TY = 8;
constant constexpr int HX = TX + 2, HY = TY + 2;
constant constexpr int IX = TX + 4, IY = TY + 4;
constant constexpr int C = 16, Z = 17;
constant constexpr int SLICE = HX * HY * C;

struct Dims {
    uint height, width;
    float attention_bias;
};

using StemResult = simdgroup_float8x8[2][2][2];

inline void stem_clear(thread StemResult &acc) {
    for (int r = 0; r < 2; r++) {
        for (int x = 0; x < 2; x++) {
            acc[r][x][0] = simdgroup_float8x8(0);
            acc[r][x][1] = simdgroup_float8x8(0);
        }
    }
}

inline void stem_plane(
    threadgroup const half *slice, device const half *weight, uint simd, thread StemResult &acc) {
    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            device const half *tap = weight + (dy * 3 + dx) * C * C;
            for (int k = 0; k < 2; k++) {
                simdgroup_half8x8 w0, w1;
                simdgroup_load(w0, tap + k * 8 * C, C);
                simdgroup_load(w1, tap + k * 8 * C + 8, C);
                for (int r = 0; r < 2; r++) {
                    for (int x = 0; x < 2; x++) {
                        simdgroup_half8x8 a;
                        simdgroup_load(
                            a, slice + ((simd * 2 + r + dy) * HX + x * 8 + dx) * C + k * 8, C);
                        simdgroup_multiply_accumulate(acc[r][x][0], a, w0, acc[r][x][0]);
                        simdgroup_multiply_accumulate(acc[r][x][1], a, w1, acc[r][x][1]);
                    }
                }
            }
        }
    }
}

inline void stem_store(threadgroup half *result, uint simd, thread StemResult &acc) {
    for (int r = 0; r < 2; r++) {
        for (int x = 0; x < 2; x++) {
            for (int c = 0; c < 2; c++) {
                simdgroup_half8x8 narrow;
                narrow.thread_elements()[0] = half(acc[r][x][c].thread_elements()[0]);
                narrow.thread_elements()[1] = half(acc[r][x][c].thread_elements()[1]);
                simdgroup_store(narrow, result + ((simd * 2 + r) * TX + x * 8) * C + c * 8, C);
            }
        }
    }
}

kernel void depth_stem(device const uchar *image [[buffer(0)]], constant half4 *first [[buffer(1)]],
    constant half4 *first_bias [[buffer(2)]], device const half *second [[buffer(3)]],
    device const half *second_bias [[buffer(4)]], device const half *attention [[buffer(5)]],
    device half *out [[buffer(6)]], constant Dims &d [[buffer(7)]],
    device const half *values [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half ring[3 * SLICE];
    threadgroup half input[3][IY * IX];
    const int H = d.height, W = d.width;
    const int y0 = group.y * TY, x0 = group.x * TX;

    auto load_input = [&](int z) {
        threadgroup half *slot = input[(z + 3) % 3];
        for (int i = tid; i < IY * IX; i += 128) {
            int y = y0 - 2 + i / IX, x = x0 - 2 + i % IX;
            bool inside = z >= 0 && z < Z && y >= 0 && y < H && x >= 0 && x < W;
            slot[i] = inside ? values[image[(size_t(z) * H + y) * W + x]] : half(0);
        }
    };
    auto first_conv = [&](int z) {
        threadgroup half *slot = ring + ((z + 3) % 3) * SLICE;
        for (int p = tid; p < HY * HX; p += 128) {
            int py = p / HX, px = p % HX;
            int y = y0 - 1 + py, x = x0 - 1 + px;
            bool inside = z >= 0 && z < Z && y >= 0 && y < H && x >= 0 && x < W;
            float4 acc[C / 4];
            for (int c = 0; c < C / 4; c++) {
                acc[c] = float4(first_bias[c]);
            }
            for (int dz = 0; inside && dz < 3; dz++) {
                threadgroup half *s = input[(z + dz + 2) % 3];
                for (int dy = 0; dy < 3; dy++) {
                    for (int dx = 0; dx < 3; dx++) {
                        float v = s[(py + dy) * IX + px + dx];
                        int tap = ((dz * 3 + dy) * 3 + dx) * (C / 4);
                        for (int c = 0; c < C / 4; c++) {
                            acc[c] += v * float4(first[tap + c]);
                        }
                    }
                }
            }
            for (int c = 0; c < C / 4; c++) {
                for (int k = 0; k < 4; k++) {
                    float a = acc[c][k];
                    slot[p * C + c * 4 + k] = inside ? half(a > 0 ? a : 0.01f * a) : half(0);
                }
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
        StemResult acc;
        stem_clear(acc);
        for (int dz = 0; dz < 3; dz++) {
            threadgroup const half *slot = ring + ((z + dz + 2) % 3) * SLICE;
            stem_plane(slot, second + dz * 9 * C * C, simd, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *result = ring + ((z + 2) % 3) * SLICE;
        stem_store(result, simd, acc);
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
        for (int c = 0; c < C; c++) {
            pooled[c] = pooled[c] * fade + weight * value[c];
        }
        best = top;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int y = y0 + own_y, x = x0 + own_x;
    if (y < H && x < W) {
        device half *o = out + size_t(y) * W + x;
        for (int c = 0; c < C; c++) {
            o[size_t(c) * H * W] = half(pooled[c] / total);
            o[size_t(C + c) * H * W] = half(peak[c]);
        }
    }
}

struct Box {
    int tiles, height, width;
    uint count;
};

kernel void stem_first(device const uchar *tiles [[buffer(0)]],
    device const half *values [[buffer(1)]], device const half *weight [[buffer(2)]],
    device half *out [[buffer(3)]], constant Box &b [[buffer(4)]],
    uint3 at [[thread_position_in_grid]]) {
    int x = at.x, y = at.y, z = at.z % Z, t = at.z / Z;
    if (x >= b.width || y >= b.height) {
        return;
    }
    float acc[C] = {0};
    for (int dz = 0; dz < 3; dz++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                int zz = z + dz - 1, yy = y + dy - 1, xx = x + dx - 1;
                if (zz < 0 || zz >= Z || yy < 0 || yy >= b.height || xx < 0 || xx >= b.width) {
                    continue;
                }
                float v = values[tiles[((size_t(t) * Z + zz) * b.height + yy) * b.width + xx]];
                for (int c = 0; c < C; c++) {
                    acc[c] += v * float(weight[c * 27 + (dz * 3 + dy) * 3 + dx]);
                }
            }
        }
    }
    device half *o = out + ((size_t(t) * Z + z) * b.height + y) * b.width + x;
    for (int c = 0; c < C; c++) {
        o[c * b.count] = half(acc[c]);
    }
}

kernel void stem_second(device const half *first [[buffer(0)]],
    device const half *weight [[buffer(1)]], device const half *norm [[buffer(2)]],
    device half *out [[buffer(3)]], constant Box &b [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half act[3][HY][HX][C];
    int t = group.z / Z, z = group.z % Z, x0 = group.x * TX, y0 = group.y * TY;
    for (int i = tid; i < 3 * HY * HX; i += TX * TY) {
        int dz = i / (HY * HX), py = i / HX % HY, px = i % HX;
        int zz = z + dz - 1, yy = y0 + py - 1, xx = x0 + px - 1;
        bool inside = zz >= 0 && zz < Z && yy >= 0 && yy < b.height && xx >= 0 && xx < b.width;
        if (!inside) {
            for (int k = 0; k < C; k++) {
                act[dz][py][px][k] = half(0);
            }
            continue;
        }
        device const half *v = first + ((size_t(t) * Z + zz) * b.height + yy) * b.width + xx;
        for (int k = 0; k < C; k++) {
            half scaled = v[k * b.count] * norm[k];
            half a = scaled + norm[C + k];
            act[dz][py][px][k] = a > 0 ? a : a * 0.01h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    StemResult acc;
    stem_clear(acc);
    for (int dz = (z == 0); dz < 3 - (z + 1 == Z); dz++) {
        stem_plane(&act[dz][0][0][0], weight + dz * 9 * C * C, simd, acc);
    }
    threadgroup half result[TY * TX * C];
    stem_store(result, simd, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int x = x0 + tid % TX, y = y0 + tid / TX;
    if (x < b.width && y < b.height) {
        device half *o = out + ((size_t(t) * Z + z) * b.height + y) * b.width + x;
        for (int c = 0; c < C; c++) {
            o[c * b.count] = result[tid * C + c];
        }
    }
}
