#include <metal_stdlib>
using namespace metal;

struct Fragment {
    float3 origin, centre, eu, ev, ew;
    float voxel, bend_u, bend_v;
    float half_u, half_v, edge;
    float t_recto, t_verso, dip, dip_um;
    float parenchyma, lump, lump_um;
    float fibre_recto, spacing_recto, radius_recto;
    float fibre_verso, spacing_verso, radius_verso;
    float jitter, wave_um, wave_scale_um, fray_um;
    float sigma, contrast;
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

static float unit_hash(int k, uint salt) {
    return float(mix_bits(uint(k) * 0x9e3779b1U ^ salt) & 0xffffffU) / 16777216.0f;
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

static float blurred_box(float x, float a, float b, float sigma) {
    float s = 1.0f / (M_SQRT2_F * max(sigma, 1e-3f));
    return 0.5f * (erf_c((b - x) * s) - erf_c((a - x) * s));
}

static float fibre_family(float along, float across, float w, float w0, float w1, float spacing,
    float radius, float density, constant Fragment &f, uint salt, float reach_along) {
    if (density <= 0.0f || spacing <= 0.0f)
        return 0.0f;
    float total = 0.0f;
    int k0 = int(floor(across / spacing));
    for (int k = k0 - 2; k <= k0 + 2; k++) {
        float position = (k + 0.5f + f.jitter * (unit_hash(k, salt) - 0.5f)) * spacing;
        float r = radius * (0.6f + 0.8f * unit_hash(k, salt ^ 0x5bd1e995U));
        float depth = w0 + r + (w1 - w0 - 2 * r) * unit_hash(k, salt ^ 0x1b873593U);
        float wave =
            f.wave_um * value_noise(float2(along / f.wave_scale_um, float(k)), salt ^ 0xcc9e2d51U);
        float da = across - position - wave, dw = w - depth;
        float s2 = f.sigma * f.sigma + 0.5f * r * r;
        float end = f.fray_um * unit_hash(k, salt ^ 0xe6546b64U);
        float extent = blurred_box(along, -reach_along - end, reach_along + end, f.sigma);
        total += density * (r * r / (2.0f * s2)) * exp(-(da * da + dw * dw) / (2.0f * s2)) * extent;
    }
    return total;
}

kernel void papyrus(constant Fragment &f [[buffer(0)]], constant Dummy &unused [[buffer(1)]],
    device float *out [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
    if (int(g.x) >= f.nx || int(g.y) >= f.ny || int(g.z) >= f.nz)
        return;
    float3 p = f.origin + float3(g.z, g.y, g.x) * f.voxel - f.centre;
    float u = dot(p, f.eu), v = dot(p, f.ev);
    float w = dot(p, f.ew) - 0.5f * (f.bend_u * u * u + f.bend_v * v * v);
    float lump = 1.0f + f.lump * value_noise(float2(u, v) / f.lump_um, f.seed ^ 0x27d4eb2fU);
    float edge_sigma = sqrt(f.sigma * f.sigma + f.edge * f.edge);
    float inside = blurred_box(u, -f.half_u, f.half_u, edge_sigma) *
        blurred_box(v, -f.half_v, f.half_v, edge_sigma);
    float recto = blurred_box(w, 0.0f, f.t_recto, f.sigma),
          verso = blurred_box(w, -f.t_verso, 0.0f, f.sigma);
    float dip = f.dip * exp(-(w * w) / (2.0f * (f.dip_um * f.dip_um + f.sigma * f.sigma)));
    float density = f.parenchyma * lump * (recto + verso - dip) * inside;
    density += fibre_family(u, v, w, 0.0f, f.t_recto, f.spacing_recto, f.radius_recto,
                   f.fibre_recto, f, f.seed ^ 0x85ebca6bU, f.half_u) *
        blurred_box(v, -f.half_v, f.half_v, edge_sigma);
    density += fibre_family(v, u, w, -f.t_verso, 0.0f, f.spacing_verso, f.radius_verso,
                   f.fibre_verso, f, f.seed ^ 0xc2b2ae35U, f.half_v) *
        blurred_box(u, -f.half_u, f.half_u, edge_sigma);
    uint index = (g.z * uint(f.ny) + g.y) * uint(f.nx) + g.x;
    out[index] = f.contrast * density;
}
