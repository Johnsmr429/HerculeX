#define WIN (CELL + 2 * MARGIN)
#define VOXELS (WIN * WIN * WIN)
#define BPV 2
#define SMAX 16
#define BINS (2 * SMAX * BPV + 1)
#define MAXN 8
#define LATERAL 12.0f
#define MIN_FILL 0.05f
#define MAX_LAG 40
#define THREADS 256
#define UNIT 256

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

typedef struct {
    float x, y, z;
    signed char nx, ny, nz;
    uchar prominence, width, coherence;
    ushort pad;
} Node;

typedef struct {
    int Z, Y, X, cells_x, cells_y, first_z, air, slab_z0;
    uint capacity;
} Params;

__device__ int clamped(int v) { return v < 0 ? 0 : (v >= WIN ? WIN - 1 : v); }

__device__ ushort blur_xy(const uchar *plane, int i) {
    int x = i % WIN, y = i / WIN;
    int left = clamped(x - 1), right = clamped(x + 1);
    const uchar *upper = plane + clamped(y - 1) * WIN;
    const uchar *middle = plane + y * WIN;
    const uchar *lower = plane + clamped(y + 1) * WIN;
    return upper[left] + 2 * upper[x] + upper[right] +
        2 * (middle[left] + 2 * middle[x] + middle[right]) + lower[left] + 2 * lower[x] +
        lower[right];
}

extern "C" __global__ void cell_nodes(
    const uchar *vol, Node *out, int *ids, uint *counter, Params p) {
    __shared__ uchar win[VOXELS];
    __shared__ ushort ring[3][WIN * WIN];
    __shared__ float gauss[WIN];
    __shared__ float partial[THREADS / 32][6];
    __shared__ float axis[4];
    __shared__ uint acc[BINS], cnt[BINS], inner;
    __shared__ int proceed;

    uint cell = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, group = tid >> 5;
    int kx = cell % p.cells_x, ky = (cell / p.cells_x) % p.cells_y,
        kz = cell / (p.cells_x * p.cells_y);
    int oz = p.first_z + kz * CELL, oy = ky * CELL, ox = kx * CELL;

    if (tid == 0)
        inner = 0u;
    __syncthreads();
    uint occupied = 0;
    for (int i = tid; i < CELL * CELL * CELL; i += THREADS) {
        int z = oz + i / (CELL * CELL), y = oy + (i / CELL) % CELL, x = ox + i % CELL;
        if (z < p.Z && y < p.Y && x < p.X && vol[((long long)z * p.Y + y) * p.X + x] > p.air)
            occupied++;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        occupied += __shfl_down_sync(0xffffffffu, occupied, offset);
    if (lane == 0 && occupied)
        atomicAdd(&inner, occupied);
    __syncthreads();
    if (inner < MIN_FILL * CELL * CELL * CELL)
        return;

    int bz = oz - MARGIN, by = oy - MARGIN, bx = ox - MARGIN;
    for (int i = tid; i < VOXELS; i += THREADS) {
        int z = bz + i / (WIN * WIN), y = by + (i / WIN) % WIN, x = bx + i % WIN;
        win[i] = (z >= 0 && z < p.Z && y >= 0 && y < p.Y && x >= 0 && x < p.X)
            ? vol[((long long)z * p.Y + y) * p.X + x]
            : 0;
    }
    float centre = (WIN - 1) * 0.5f;
    if (tid < WIN) {
        float d = (tid - centre) / (7.0f * WIN / 24);
        gauss[tid] = expf(-0.5f * d * d);
    }
    for (int b = tid; b < BINS; b += THREADS) {
        acc[b] = 0u;
        cnt[b] = 0u;
    }
    __syncthreads();

    float tensor[6] = {0};
    ushort previous[(WIN * WIN + THREADS - 1) / THREADS];
    ushort current[(WIN * WIN + THREADS - 1) / THREADS];
    for (int i = tid; i < WIN * WIN; i += THREADS) {
        int item = i / THREADS;
        current[item] = blur_xy(win, i);
        previous[item] = current[item];
    }
    for (int plane = 0; plane < WIN; plane++) {
        ushort *slot = ring[plane % 3];
        for (int i = tid; i < WIN * WIN; i += THREADS) {
            int item = i / THREADS;
            ushort next = blur_xy(win + clamped(plane + 1) * WIN * WIN, i);
            slot[i] = previous[item] + 2 * current[item] + next;
            previous[item] = current[item];
            current[item] = next;
        }
        __syncthreads();
        int z = plane - 1;
        if (z >= 1) {
            ushort *below = ring[(z + 2) % 3], *here = ring[z % 3], *above = ring[(z + 1) % 3];
            for (int i = tid; i < WIN * WIN; i += THREADS) {
                int y = i / WIN, x = i % WIN, v = z * WIN * WIN + i;
                if (y < 1 || y >= WIN - 1 || x < 1 || x >= WIN - 1)
                    continue;
                if (!win[v] || !win[v - WIN * WIN] || !win[v + WIN * WIN] || !win[v - WIN] ||
                    !win[v + WIN] || !win[v - 1] || !win[v + 1])
                    continue;
                float gz = ((float)above[i] - (float)below[i]) * (1.0f / 64);
                float gy = ((float)here[i + WIN] - (float)here[i - WIN]) * (1.0f / 64);
                float gx = ((float)here[i + 1] - (float)here[i - 1]) * (1.0f / 64);
                float w = gauss[z] * gauss[y] * gauss[x];
                tensor[0] += w * gz * gz;
                tensor[1] += w * gz * gy;
                tensor[2] += w * gz * gx;
                tensor[3] += w * gy * gy;
                tensor[4] += w * gy * gx;
                tensor[5] += w * gx * gx;
            }
        }
        __syncthreads();
    }
    for (int k = 0; k < 6; k++) {
        float sum = tensor[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            partial[group][k] = sum;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float t[6] = {0};
        for (int g = 0; g < THREADS / 32; g++)
            for (int k = 0; k < 6; k++)
                t[k] += partial[g][k];
        float trace = t[0] + t[3] + t[5];
        proceed = trace > 0;
        if (proceed) {
            float a[3][3] = {{t[0] / trace, t[1] / trace, t[2] / trace},
                {t[1] / trace, t[3] / trace, t[4] / trace},
                {t[2] / trace, t[4] / trace, t[5] / trace}},
                  v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            for (int sweep = 0; sweep < 8; sweep++) {
                float off = fabsf(a[0][1]) + fabsf(a[0][2]) + fabsf(a[1][2]);
                if (off < 1e-9f)
                    break;
                for (int p = 0; p < 2; p++)
                    for (int q = p + 1; q < 3; q++) {
                        if (fabsf(a[p][q]) < 1e-12f)
                            continue;
                        float theta = (a[q][q] - a[p][p]) / (2.0f * a[p][q]);
                        float t = (theta >= 0 ? 1.0f : -1.0f) /
                            (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                        float c = rsqrtf(t * t + 1.0f), s = t * c;
                        for (int k = 0; k < 3; k++) {
                            float akp = a[k][p], akq = a[k][q];
                            a[k][p] = c * akp - s * akq;
                            a[k][q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++) {
                            float apk = a[p][k], aqk = a[q][k];
                            a[p][k] = c * apk - s * aqk;
                            a[q][k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++) {
                            float vkp = v[k][p], vkq = v[k][q];
                            v[k][p] = c * vkp - s * vkq;
                            v[k][q] = s * vkp + c * vkq;
                        }
                    }
            }
            int order[3] = {0, 1, 2};
            for (int i = 0; i < 2; i++)
                for (int j = i + 1; j < 3; j++)
                    if (a[order[j]][order[j]] > a[order[i]][order[i]]) {
                        int q = order[i];
                        order[i] = order[j];
                        order[j] = q;
                    }
            float l1 = a[order[0]][order[0]], l2 = a[order[1]][order[1]];
            axis[0] = v[0][order[0]];
            axis[1] = v[1][order[0]];
            axis[2] = v[2][order[0]];
            axis[3] = (l1 - l2) / (l1 + l2 + 1e-9f);
        }
    }
    __syncthreads();
    if (!proceed)
        return;
    float nz = axis[0], ny = axis[1], nx = axis[2];

    for (int i = tid; i < VOXELS; i += THREADS) {
        uint value = win[i];
        if (!value)
            continue;
        float dz = (i / (WIN * WIN)) - centre, dy = ((i / WIN) % WIN) - centre,
              dx = (i % WIN) - centre;
        float s = dz * nz + dy * ny + dx * nx;
        float lateral = dz * dz + dy * dy + dx * dx - s * s;
        if (lateral > LATERAL * LATERAL || fabsf(s) >= SMAX - 0.5f)
            continue;
        float position = (s + SMAX) * BPV;
        int bin = position;
        uint high = (uint)((position - bin) * UNIT + 0.5f), low = UNIT - high;
        atomicAdd(&acc[bin], value * low);
        atomicAdd(&cnt[bin], low);
        atomicAdd(&acc[bin + 1], value * high);
        atomicAdd(&cnt[bin + 1], high);
    }
    __syncthreads();
    if (tid != 0)
        return;

    float profile[BINS], smoothed[BINS];
    bool valid[BINS];
    int used = 0;
    for (int b = 0; b < BINS; b++) {
        uint c = cnt[b];
        valid[b] = c >= 8 * UNIT;
        used += valid[b];
        profile[b] = valid[b] ? (float)acc[b] / c : 0;
    }
    for (int b = 1; b < BINS - 1; b++)
        smoothed[b] = valid[b - 1] && valid[b + 1]
            ? 0.25f * profile[b - 1] + 0.5f * profile[b] + 0.25f * profile[b + 1]
            : profile[b];
    smoothed[0] = profile[0];
    smoothed[BINS - 1] = profile[BINS - 1];
    if (used < MAX_LAG)
        return;

    Node found[MAXN];
    int n = 0;
    float cz = oz + (CELL - 1) * 0.5f, cy = oy + (CELL - 1) * 0.5f, cx = ox + (CELL - 1) * 0.5f;
    for (int b = 2; b < BINS - 2 && n < MAXN; b++) {
        if (!valid[b] || !(smoothed[b] > smoothed[b - 1] && smoothed[b] >= smoothed[b + 1]))
            continue;
        float left = smoothed[b], right = smoothed[b];
        for (int k = b - 1; k >= 0 && valid[k] && smoothed[k] <= smoothed[b]; k--)
            left = fminf(left, smoothed[k]);
        for (int k = b + 1; k < BINS && valid[k] && smoothed[k] <= smoothed[b]; k++)
            right = fminf(right, smoothed[k]);
        float prominence = smoothed[b] - fmaxf(left, right);
        if (prominence < 1.0f)
            continue;
        float denom = smoothed[b - 1] - 2 * smoothed[b] + smoothed[b + 1];
        float shift = denom < 0 ? 0.5f * (smoothed[b - 1] - smoothed[b + 1]) / denom : 0;
        float s = (b + shift) / BPV - SMAX;
        float pz = cz + s * nz, py = cy + s * ny, px = cx + s * nx;
        if (pz < oz - 1 || pz > oz + CELL || py < oy - 1 || py > oy + CELL || px < ox - 1 ||
            px > ox + CELL)
            continue;
        float half_level = smoothed[b] - 0.5f * prominence;
        int lo = b, hi = b;
        while (lo > 0 && valid[lo - 1] && smoothed[lo] > half_level)
            lo--;
        while (hi < BINS - 1 && valid[hi + 1] && smoothed[hi] > half_level)
            hi++;
        Node node = {
            .x = px,
            .y = py,
            .z = pz + p.slab_z0,
            .nx = (signed char)rintf(nx * 127),
            .ny = (signed char)rintf(ny * 127),
            .nz = (signed char)rintf(nz * 127),
            .prominence = (uchar)fminf(255.0f, prominence * 2.0f),
            .width = (uchar)fminf(255.0f, (float)(hi - lo) * 4 / BPV),
            .coherence = (uchar)rintf(axis[3] * 255.0f),
        };
        found[n++] = node;
    }
    if (n == 0)
        return;
    uint base = atomicAdd(counter, (uint)n);
    for (int k = 0; k < n; k++) {
        if (base + k >= p.capacity)
            return;
        out[base + k] = found[k];
        ids[base + k] = cell;
    }
}
