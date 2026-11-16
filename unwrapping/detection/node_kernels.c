
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef CELL
#define CELL 8
#endif
#ifndef MARGIN
#define MARGIN 8
#endif
#define WIN (CELL + 2 * MARGIN)
#define BPV 2
#ifndef SMAX
#define SMAX 16
#endif
#define BINS (2 * SMAX * BPV + 1)
#define MAXN 8
#ifndef LATERAL
#define LATERAL 12.0f
#endif
#define MIN_FILL 0.05f
#define MIN_LAG 4
#define MAX_LAG 40

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} Node;

typedef struct {
    uint8_t mean, coherence, wavelength, contrast;
} CellStats;

static void jacobi3(double a[3][3], double v[3][3]) {
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) v[i][j] = i == j;
    for (int sweep = 0; sweep < 8; sweep++) {
        double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
        if (off < 1e-12) return;
        for (int p = 0; p < 2; p++) for (int q = p + 1; q < 3; q++) {
            if (fabs(a[p][q]) < 1e-15) continue;
            double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
            double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; k++) {
                double akp = a[k][p], akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; k++) {
                double apk = a[p][k], aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; k++) {
                double vkp = v[k][p], vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
}

static int analyse(const uint8_t *vol, int Z, int Y, int X, int oz, int oy, int ox, int air,
                   int z_shift, Node *out, CellStats *stats) {
    static const float kernel[3] = {0.25f, 0.5f, 0.25f};
    static __thread float raw[WIN][WIN][WIN], tmp[WIN][WIN][WIN], smooth[WIN][WIN][WIN];
    static __thread uint8_t filled[WIN][WIN][WIN];
    memset(stats, 0, sizeof(*stats));
    int bz = oz - MARGIN, by = oy - MARGIN, bx = ox - MARGIN;
    long total = 0, sum = 0, inner = 0;
    for (int z = 0; z < WIN; z++) for (int y = 0; y < WIN; y++) for (int x = 0; x < WIN; x++) {
        int gz = bz + z, gy = by + y, gx = bx + x;
        int value = 0;
        if (gz >= 0 && gz < Z && gy >= 0 && gy < Y && gx >= 0 && gx < X)
            value = vol[((size_t)gz * Y + gy) * X + gx];
        raw[z][y][x] = value;
        filled[z][y][x] = value > 0;
        if (value > air) {
            total++;
            sum += value;
            if (z >= MARGIN && z < MARGIN + CELL && y >= MARGIN && y < MARGIN + CELL &&
                x >= MARGIN && x < MARGIN + CELL) inner++;
        }
    }
    if (inner < MIN_FILL * CELL * CELL * CELL) return 0;
    stats->mean = (uint8_t)(sum / total);

    for (int z = 0; z < WIN; z++) for (int y = 0; y < WIN; y++) for (int x = 0; x < WIN; x++) {
        int a = x > 0 ? x - 1 : x, b = x < WIN - 1 ? x + 1 : x;
        tmp[z][y][x] = kernel[0] * raw[z][y][a] + kernel[1] * raw[z][y][x] + kernel[2] * raw[z][y][b];
    }
    for (int z = 0; z < WIN; z++) for (int y = 0; y < WIN; y++) for (int x = 0; x < WIN; x++) {
        int a = y > 0 ? y - 1 : y, b = y < WIN - 1 ? y + 1 : y;
        smooth[z][y][x] = kernel[0] * tmp[z][a][x] + kernel[1] * tmp[z][y][x] + kernel[2] * tmp[z][b][x];
    }
    for (int z = 0; z < WIN; z++) for (int y = 0; y < WIN; y++) for (int x = 0; x < WIN; x++) {
        int a = z > 0 ? z - 1 : z, b = z < WIN - 1 ? z + 1 : z;
        tmp[z][y][x] = kernel[0] * smooth[a][y][x] + kernel[1] * smooth[z][y][x] + kernel[2] * smooth[b][y][x];
    }

    float weight1[WIN];
    float centre = (WIN - 1) * 0.5f;
    for (int i = 0; i < WIN; i++) {
        float d = (i - centre) / (7.0f * WIN / 24);
        weight1[i] = expf(-0.5f * d * d);
    }
    double j00 = 0, j01 = 0, j02 = 0, j11 = 0, j12 = 0, j22 = 0;
    for (int z = 1; z < WIN - 1; z++) for (int y = 1; y < WIN - 1; y++) {
        float wzy = weight1[z] * weight1[y];
        for (int x = 1; x < WIN - 1; x++) {
            if (!(filled[z][y][x] & filled[z - 1][y][x] & filled[z + 1][y][x] & filled[z][y - 1][x] &
                  filled[z][y + 1][x] & filled[z][y][x - 1] & filled[z][y][x + 1])) continue;
            float gz = tmp[z + 1][y][x] - tmp[z - 1][y][x];
            float gy = tmp[z][y + 1][x] - tmp[z][y - 1][x];
            float gx = tmp[z][y][x + 1] - tmp[z][y][x - 1];
            float w = wzy * weight1[x];
            j00 += w * gz * gz; j01 += w * gz * gy; j02 += w * gz * gx;
            j11 += w * gy * gy; j12 += w * gy * gx; j22 += w * gx * gx;
        }
    }
    double a[3][3] = {{j00, j01, j02}, {j01, j11, j12}, {j02, j12, j22}}, v[3][3];
    if (j00 + j11 + j22 <= 0) return 0;
    jacobi3(a, v);
    int order[3] = {0, 1, 2};
    for (int i = 0; i < 2; i++) for (int j = i + 1; j < 3; j++)
        if (a[order[j]][order[j]] > a[order[i]][order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    double l1 = a[order[0]][order[0]], l2 = a[order[1]][order[1]];
    float nz = (float)v[0][order[0]], ny = (float)v[1][order[0]], nx = (float)v[2][order[0]];
    float coherence = (float)((l1 - l2) / (l1 + l2 + 1e-9));
    stats->coherence = (uint8_t)lrintf(coherence * 255.0f);

    float acc[BINS] = {0}, cnt[BINS] = {0};
    for (int z = 0; z < WIN; z++) for (int y = 0; y < WIN; y++) for (int x = 0; x < WIN; x++) {
        if (!filled[z][y][x]) continue;
        float dz = z - centre, dy = y - centre, dx = x - centre;
        float s = dz * nz + dy * ny + dx * nx;
        float lateral = dz * dz + dy * dy + dx * dx - s * s;
        if (lateral > LATERAL * LATERAL || fabsf(s) >= SMAX - 0.5f) continue;
        float position = (s + SMAX) * BPV;
        int bin = (int)position;
        float f = position - bin;
        acc[bin] += raw[z][y][x] * (1 - f);
        cnt[bin] += 1 - f;
        acc[bin + 1] += raw[z][y][x] * f;
        cnt[bin + 1] += f;
    }
    float profile[BINS], smoothed[BINS];
    int valid[BINS];
    for (int b = 0; b < BINS; b++) {
        valid[b] = cnt[b] >= 8.0f;
        profile[b] = valid[b] ? acc[b] / cnt[b] : 0;
    }
    for (int b = 1; b < BINS - 1; b++)
        smoothed[b] = valid[b - 1] && valid[b + 1]
                          ? 0.25f * profile[b - 1] + 0.5f * profile[b] + 0.25f * profile[b + 1]
                          : profile[b];
    smoothed[0] = profile[0];
    smoothed[BINS - 1] = profile[BINS - 1];

    float mean = 0;
    int used = 0;
    for (int b = 0; b < BINS; b++) if (valid[b]) { mean += smoothed[b]; used++; }
    if (used < MAX_LAG) return 0;
    mean /= used;
    float variance = 0;
    for (int b = 0; b < BINS; b++) if (valid[b]) variance += (smoothed[b] - mean) * (smoothed[b] - mean);
    variance /= used;
    stats->contrast = (uint8_t)fminf(255.0f, sqrtf(variance) * 8.0f);
    float previous = 1, best = 0;
    int best_lag = 0;
    float corr[MAX_LAG + 2];
    for (int lag = 1; lag <= MAX_LAG + 1; lag++) {
        float c = 0;
        int n = 0;
        for (int b = 0; b + lag < BINS; b++)
            if (valid[b] && valid[b + lag]) { c += (smoothed[b] - mean) * (smoothed[b + lag] - mean); n++; }
        corr[lag] = n ? c / n / (variance + 1e-6f) : 0;
    }
    for (int lag = MIN_LAG; lag <= MAX_LAG; lag++) {
        if (corr[lag] > corr[lag - 1] && corr[lag] >= corr[lag + 1] && corr[lag] > best) {
            best = corr[lag];
            best_lag = lag;
        }
    }
    (void)previous;
    stats->wavelength = best_lag ? (uint8_t)(best_lag * 4 / BPV) : 0;

    int n = 0;
    float cz = oz + (CELL - 1) * 0.5f, cy = oy + (CELL - 1) * 0.5f, cx = ox + (CELL - 1) * 0.5f;
    for (int b = 2; b < BINS - 2 && n < MAXN; b++) {
        if (!valid[b] || !(smoothed[b] > smoothed[b - 1] && smoothed[b] >= smoothed[b + 1])) continue;
        float left = smoothed[b], right = smoothed[b];
        for (int k = b - 1; k >= 0 && valid[k] && smoothed[k] <= smoothed[b]; k--) if (smoothed[k] < left) left = smoothed[k];
        for (int k = b + 1; k < BINS && valid[k] && smoothed[k] <= smoothed[b]; k++) if (smoothed[k] < right) right = smoothed[k];
        float prominence = smoothed[b] - fmaxf(left, right);
        if (prominence < 1.0f) continue;
        float denom = smoothed[b - 1] - 2 * smoothed[b] + smoothed[b + 1];
        float shift = denom < 0 ? 0.5f * (smoothed[b - 1] - smoothed[b + 1]) / denom : 0;
        float s = (b + shift) / BPV - SMAX;
        float pz = cz + s * nz, py = cy + s * ny, px = cx + s * nx;
        if (pz < oz - 1 || pz > oz + CELL || py < oy - 1 || py > oy + CELL || px < ox - 1 || px > ox + CELL) continue;
        float half = smoothed[b] - 0.5f * prominence;
        int lo = b, hi = b;
        while (lo > 0 && valid[lo - 1] && smoothed[lo] > half) lo--;
        while (hi < BINS - 1 && valid[hi + 1] && smoothed[hi] > half) hi++;
        Node *node = &out[n++];
        node->x = px;
        node->y = py;
        node->z = pz + z_shift;
        node->nx = (int8_t)lrintf(nx * 127);
        node->ny = (int8_t)lrintf(ny * 127);
        node->nz = (int8_t)lrintf(nz * 127);
        node->prominence = (uint8_t)fminf(255.0f, prominence * 2.0f);
        node->width = (uint8_t)fminf(255.0f, (float)(hi - lo) * 4 / BPV);
        node->coherence = stats->coherence;
        node->pad = 0;
    }
    return n;
}

// All cells of one row (fixed cell z and y) of a slab; `slab_z0` is the slab's first plane in the volume.
void hx_cell_row(const uint8_t *vol, int Z, int Y, int X, int oz, int oy, int cells_x, int air,
                 int slab_z0, Node *nodes, uint8_t *counts, CellStats *stats) {
    for (int c = 0; c < cells_x; c++)
        counts[c] = (uint8_t)analyse(vol, Z, Y, X, oz, oy, c * CELL, air, slab_z0,
                                     nodes + (size_t)c * MAXN, stats + c);
}
