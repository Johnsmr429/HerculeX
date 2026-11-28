#include <metal_stdlib>
using namespace metal;

struct Fragment {
    float3 centre, eu, ev, ew;
    float half_u, half_v, tear_um, tear_scale_um, sigma_um, core_um, voxel;
    int slab_rows, slab_cols, slab_depth, row0, col0;
    int gap_step, gap_rows, gap_cols, height_size;
    uint seed;
    int nz, ny, nx;
};

struct Dummy {
    int unused;
};

static uint mix_bits(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float lattice(int2 c, uint seed) {
    uint h = mix_bits(uint(c.x) * 0x8da6b343U ^ uint(c.y) * 0xd8163841U ^ seed);
    return float(h & 0xffffU) / 32767.5f - 1.0f;
}

static float value_noise(float2 v, uint seed) {
    float2 b = floor(v), f = v - b;
    float2 w = f * f * (3.0f - 2.0f * f);
    int2 c = int2(b);
    return mix(mix(lattice(c, seed), lattice(c + int2(1, 0), seed), w.x),
        mix(lattice(c + int2(0, 1), seed), lattice(c + int2(1, 1), seed), w.x), w.y);
}

static float erf_c(float x) {
    float z = fabs(x);
    if (z > 4.0f)
        return x < 0 ? -1.0f : 1.0f;
    float t = 1.0f / (1.0f + 0.5f * z);
    float tail = -0.82215223f + t * 0.17087277f;
    tail = -1.13520398f + t * (1.48851587f + t * tail);
    tail = -0.18628806f + t * (0.27886807f + t * tail);
    tail = 0.37409196f + t * (0.09678418f + t * tail);
    float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * tail));
    float e = 1.0f - r;
    return x < 0 ? -e : e;
}

static float phi(float x) { return 0.5f * (1.0f + erf_c(x * M_SQRT1_2_F)); }

static float bilinear(device const float *a, int rows, int cols, float r, float c) {
    r = clamp(r, 0.0f, float(rows) - 1.001f);
    c = clamp(c, 0.0f, float(cols) - 1.001f);
    int r0 = int(r), c0 = int(c);
    float fr = r - r0, fc = c - c0;
    float top = mix(a[r0 * cols + c0], a[r0 * cols + c0 + 1], fc);
    float bottom = mix(a[(r0 + 1) * cols + c0], a[(r0 + 1) * cols + c0 + 1], fc);
    return mix(top, bottom, fr);
}

static float4 catmull_rom(float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5f *
        float4(
            -t3 + 2.0f * t2 - t, 3.0f * t3 - 5.0f * t2 + 2.0f, -3.0f * t3 + 4.0f * t2 + t, t3 - t2);
}

static float slab_at(
    device const uchar *slab, constant Fragment &f, float row, float col, float depth) {
    float r = clamp(row, 0.0f, float(f.slab_rows) - 1.001f),
          c = clamp(col, 0.0f, float(f.slab_cols) - 1.001f);
    float e = clamp(depth + 0.5f * float(f.slab_depth - 1), 0.0f, float(f.slab_depth) - 1.001f);
    int r0 = int(r), c0 = int(c), e0 = int(e);
    float4 wr = catmull_rom(r - r0), wc = catmull_rom(c - c0), we = catmull_rom(e - e0);
    float v = 0;
    for (int dr = 0; dr < 4; dr++) {
        size_t rr = size_t(clamp(r0 + dr - 1, 0, f.slab_rows - 1));
        for (int dc = 0; dc < 4; dc++) {
            size_t cc = size_t(clamp(c0 + dc - 1, 0, f.slab_cols - 1));
            float line = 0;
            for (int de = 0; de < 4; de++) {
                int ee = clamp(e0 + de - 1, 0, f.slab_depth - 1);
                line += we[de] * float(slab[(rr * f.slab_cols + cc) * f.slab_depth + ee]);
            }
            v += wr[dr] * wc[dc] * line;
        }
    }
    return v;
}

kernel void inject_fragment(constant Fragment &f [[buffer(0)]],
    constant Dummy &unused [[buffer(1)]], device const uchar *real [[buffer(2)]],
    device const float *noise [[buffer(3)]], device const uchar *slab [[buffer(4)]],
    device const float *half_in [[buffer(5)]], device const float *half_out [[buffer(6)]],
    device const float *height [[buffer(7)]], device uchar *out [[buffer(8)]],
    uint3 g [[thread_position_in_grid]]) {
    if (int(g.x) >= f.nx || int(g.y) >= f.ny || int(g.z) >= f.nz)
        return;
    uint index = (g.z * uint(f.ny) + g.y) * uint(f.nx) + g.x;
    float r = float(real[index]);
    out[index * 2] = real[index];
    out[index * 2 + 1] = 0;
    float3 p = float3(g.z, g.y, g.x) * f.voxel - f.centre;
    float u0 = dot(p, f.eu), v0 = dot(p, f.ev), w = dot(p, f.ew);
    float middle = 0.5f * float(f.height_size - 1);
    float hr = v0 / f.voxel + middle, hc = u0 / f.voxel + middle;
    int n = f.height_size;
    float h = bilinear(height, n, n, hr, hc);
    float hu =
        0.5f * (bilinear(height, n, n, hr, hc + 1.0f) - bilinear(height, n, n, hr, hc - 1.0f));
    float hv =
        0.5f * (bilinear(height, n, n, hr + 1.0f, hc) - bilinear(height, n, n, hr - 1.0f, hc));
    float slope = sqrt(1.0f + hu * hu + hv * hv);
    float wp = (w - f.voxel * h) / slope;
    float u = u0 + wp * hu / slope, v = v0 + wp * hv / slope;
    float row = v / f.voxel + float(f.row0), col = u / f.voxel + float(f.col0);
    float e_in =
        f.voxel * bilinear(half_in, f.gap_rows, f.gap_cols, row / f.gap_step, col / f.gap_step);
    float e_out =
        f.voxel * bilinear(half_out, f.gap_rows, f.gap_cols, row / f.gap_step, col / f.gap_step);
    if (wp > e_in + 3.0f * f.sigma_um || wp < -e_out - 3.0f * f.sigma_um)
        return;
    float su = u / f.half_u, sv = v / f.half_v;
    float rho = length(float2(su, sv));
    float theta = atan2(sv, su);
    float reach = f.half_u * f.half_v /
        max(length(float2(f.half_v * cos(theta), f.half_u * sin(theta))), 1e-3f);
    float2 ring = float2(cos(theta), sin(theta)) * reach / f.tear_scale_um;
    float tear = f.tear_um *
        (value_noise(ring, f.seed ^ 0x3c6ef372U) +
            0.5f * value_noise(2.0f * ring + 17.0f, f.seed ^ 0x9e3779b9U));
    float rim = phi(((1.0f - rho) * reach + tear) / f.sigma_um);
    float across = phi((e_in - wp) / f.sigma_um) * phi((wp + e_out) / f.sigma_um);
    float alpha = rim * across;
    if (alpha <= 1e-4f)
        return;
    float s = slab_at(slab, f, row, col, wp / f.voxel);
    float keep = sqrt(max(2.0f * alpha * (1.0f - alpha), 0.0f));
    float value = (1.0f - alpha) * r + alpha * s + keep * noise[index];
    out[index * 2] = uchar(clamp(rint(value), 0.0f, 255.0f));
    out[index * 2 + 1] = rim >= 0.5f && fabs(wp) <= f.core_um ? 2 : (alpha >= 0.5f ? 1 : 0);
}
