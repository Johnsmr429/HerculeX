#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { CELL_VOXELS = 16, BLOCK_VOXELS = 128, ORIENTATION_UNITS = 4096 };

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} OrientationNode;

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} OrientationEdge;

typedef struct {
    int64_t address, length, fd, offset;
} OrientationRun;

typedef struct {
    double x0, y0;
    int height, width;
} Section;

typedef struct {
    int32_t band, index;
} Member;

int64_t hx_level_blocks(int64_t count, const OrientationNode *nodes, const OrientationRun *runs,
    int64_t run_count, float side, int32_t *block);
int64_t hx_level_block_pairs(int64_t blocks, const int32_t *block, const OrientationNode *nodes,
    const OrientationRun *runs, int64_t run_count, int32_t units, uint64_t *handle);
void hx_level_terms_take(uint64_t handle, int32_t *p, int32_t *q, int32_t *k);
int64_t hx_level_two_labels(int64_t count, int64_t terms, const int32_t *a, const int32_t *b,
    const int32_t *weight, const int64_t *unary, uint8_t *label);
int64_t hx_level_domains(int64_t count, const OrientationNode *nodes, const OrientationRun *runs,
    int64_t run_count, const uint8_t *flip, uint8_t *keep, int64_t *log);

static int member_before(const void *left, const void *right) {
    const Member *a = left, *b = right;
    if (a->band != b->band)
        return (a->band > b->band) - (a->band < b->band);
    return (a->index > b->index) - (a->index < b->index);
}

static int section_bounds(Section *section, double xmin, double ymin, double xmax, double ymax) {
    section->x0 = xmin - 8 * CELL_VOXELS;
    section->y0 = ymin - 8 * CELL_VOXELS;
    double height = (ymax - section->y0) / CELL_VOXELS;
    double width = (xmax - section->x0) / CELL_VOXELS;
    if (!(height >= 0 && height < INT32_MAX - 10 && width >= 0 && width < INT32_MAX - 10))
        return -1;
    section->height = (int)height + 10;
    section->width = (int)width + 10;
    return 0;
}

static int section_cell(const Section *section, double x, double y) {
    int col = (int)((x - section->x0) / CELL_VOXELS);
    int row = (int)((y - section->y0) / CELL_VOXELS);
    return row * section->width + col;
}

static void dilate_line(const uint8_t *input, uint8_t *output, int count, int stride, int radius) {
    for (int i = 0; i < count; i++) {
        int low = i > radius ? i - radius : 0;
        int high = i + radius + 1 < count ? i + radius + 1 : count;
        uint8_t value = 0;
        for (int j = low; j < high; j++)
            value |= input[j * stride];
        output[i * stride] = value;
    }
}

static void erode_line(const uint8_t *input, uint8_t *output, int count, int stride, int radius) {
    for (int i = 0; i < count; i++) {
        uint8_t value = i >= radius && i + radius < count;
        int low = i > radius ? i - radius : 0;
        int high = i + radius + 1 < count ? i + radius + 1 : count;
        for (int j = low; value && j < high; j++)
            value &= input[j * stride];
        output[i * stride] = value;
    }
}

static int close_mask(uint8_t *mask, int height, int width) {
    size_t cells = (size_t)height * width;
    uint8_t *a = malloc(cells), *b = malloc(cells);
    if (!a || !b) {
        free(b);
        free(a);
        return -1;
    }
    for (int row = 0; row < height; row++)
        dilate_line(mask + (size_t)row * width, a + (size_t)row * width, width, 1, 5);
    for (int col = 0; col < width; col++)
        dilate_line(a + col, b + col, height, width, 5);
    for (int row = 0; row < height; row++)
        erode_line(b + (size_t)row * width, a + (size_t)row * width, width, 1, 5);
    for (int col = 0; col < width; col++)
        erode_line(a + col, mask + col, height, width, 5);
    free(b);
    free(a);
    return 0;
}

static int fill_mask(uint8_t *mask, int height, int width) {
    size_t cells = (size_t)height * width;
    uint8_t *outside = calloc(cells, 1);
    int32_t *queue = malloc(cells * sizeof(*queue));
    if (!outside || !queue) {
        free(queue);
        free(outside);
        return -1;
    }
    size_t begin = 0, end = 0;
    for (int row = 0; row < height; row++) {
        int at[] = {row * width, row * width + width - 1};
        for (int side = 0; side < 2; side++)
            if (!mask[at[side]] && !outside[at[side]]) {
                outside[at[side]] = 1;
                queue[end++] = at[side];
            }
    }
    for (int col = 0; col < width; col++) {
        int at[] = {col, (height - 1) * width + col};
        for (int side = 0; side < 2; side++)
            if (!mask[at[side]] && !outside[at[side]]) {
                outside[at[side]] = 1;
                queue[end++] = at[side];
            }
    }
    while (begin < end) {
        int at = queue[begin++], row = at / width, col = at % width;
        int next[4] = {at - width, at + width, at - 1, at + 1};
        uint8_t valid[4] = {row > 0, row + 1 < height, col > 0, col + 1 < width};
        for (int k = 0; k < 4; k++)
            if (valid[k] && !mask[next[k]] && !outside[next[k]]) {
                outside[next[k]] = 1;
                queue[end++] = next[k];
            }
    }
    for (size_t i = 0; i < cells; i++)
        mask[i] = !outside[i];
    free(queue);
    free(outside);
    return 0;
}

static void distance_line(const double *input, double *output, int count, int stride) {
    int *site = malloc((size_t)count * sizeof(*site));
    double *cut = malloc(((size_t)count + 1) * sizeof(*cut));
    if (!site || !cut) {
        free(cut);
        free(site);
        for (int i = 0; i < count; i++)
            output[i * stride] = 0;
        return;
    }
    int top = 0;
    site[0] = 0;
    cut[0] = -DBL_MAX;
    cut[1] = DBL_MAX;
    for (int q = 1; q < count; q++) {
        int p = site[top];
        double split = ((input[q * stride] + (double)q * q) - (input[p * stride] + (double)p * p)) /
            (2.0 * (q - p));
        while (split <= cut[top]) {
            top--;
            p = site[top];
            split = ((input[q * stride] + (double)q * q) - (input[p * stride] + (double)p * p)) /
                (2.0 * (q - p));
        }
        site[++top] = q;
        cut[top] = split;
        cut[top + 1] = DBL_MAX;
    }
    top = 0;
    for (int q = 0; q < count; q++) {
        while (cut[top + 1] < q)
            top++;
        double delta = q - site[top];
        output[q * stride] = delta * delta + input[site[top] * stride];
    }
    free(cut);
    free(site);
}

static int distance_mask(const uint8_t *mask, int height, int width, double *distance) {
    size_t cells = (size_t)height * width;
    double *input = malloc(cells * sizeof(*input));
    double *middle = malloc(cells * sizeof(*middle));
    if (!input || !middle) {
        free(middle);
        free(input);
        return -1;
    }
    for (size_t i = 0; i < cells; i++)
        input[i] = mask[i] ? 1e12 : 0;
    for (int row = 0; row < height; row++)
        distance_line(input + (size_t)row * width, middle + (size_t)row * width, width, 1);
    for (int col = 0; col < width; col++)
        distance_line(middle + col, distance + col, height, width);
    for (size_t i = 0; i < cells; i++)
        distance[i] = sqrt(distance[i]);
    free(middle);
    free(input);
    return 0;
}

static int reflected(int i, int count) {
    while (i < 0 || i >= count) {
        if (i < 0)
            i = -i - 1;
        if (i >= count)
            i = 2 * count - i - 1;
    }
    return i;
}

static int blur_distance(double *distance, int height, int width) {
    size_t cells = (size_t)height * width;
    double *middle = malloc(cells * sizeof(*middle));
    double *output = malloc(cells * sizeof(*output));
    if (!middle || !output) {
        free(output);
        free(middle);
        return -1;
    }
    double kernel[17], total = 0;
    for (int k = -8; k <= 8; k++) {
        kernel[k + 8] = exp(-(double)(k * k) / 8.0);
        total += kernel[k + 8];
    }
    for (int k = 0; k < 17; k++)
        kernel[k] /= total;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            double value = 0;
            for (int k = -8; k <= 8; k++)
                value += kernel[k + 8] * distance[(size_t)row * width + reflected(col + k, width)];
            middle[(size_t)row * width + col] = value;
        }
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            double value = 0;
            for (int k = -8; k <= 8; k++)
                value += kernel[k + 8] * middle[(size_t)reflected(row + k, height) * width + col];
            output[(size_t)row * width + col] = value;
        }
    memcpy(distance, output, cells * sizeof(*distance));
    free(output);
    free(middle);
    return 0;
}

static int face_section(
    const OrientationNode *nodes, const int32_t *ids, int32_t count, float *score) {
    const OrientationNode *first = nodes + ids[0];
    double xmin = first->x, xmax = first->x, ymin = first->y, ymax = first->y;
    for (int32_t j = 1; j < count; j++) {
        const OrientationNode *node = nodes + ids[j];
        xmin = fmin(xmin, node->x);
        xmax = fmax(xmax, node->x);
        ymin = fmin(ymin, node->y);
        ymax = fmax(ymax, node->y);
    }
    Section section;
    if (section_bounds(&section, xmin, ymin, xmax, ymax))
        return -1;
    size_t cells = (size_t)section.height * section.width;
    uint8_t *mask = calloc(cells, 1);
    double *distance = malloc(cells * sizeof(*distance));
    if (!mask || !distance) {
        free(distance);
        free(mask);
        return -1;
    }
    for (int32_t j = 0; j < count; j++) {
        const OrientationNode *node = nodes + ids[j];
        mask[section_cell(&section, node->x, node->y)] = 1;
    }
    int failed = close_mask(mask, section.height, section.width) ||
        fill_mask(mask, section.height, section.width) ||
        distance_mask(mask, section.height, section.width, distance) ||
        blur_distance(distance, section.height, section.width);
    if (!failed)
        for (int32_t j = 0; j < count; j++) {
            int i = ids[j];
            const OrientationNode *node = nodes + i;
            int at = section_cell(&section, node->x, node->y);
            double ox = -(distance[at + 1] - distance[at - 1]) * 0.5;
            double oy = -(distance[at + section.width] - distance[at - section.width]) * 0.5;
            float nx = node->nx, ny = node->ny, nz = node->nz;
            float magnitude = fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), 1e-6f);
            double size = hypot(ox, oy);
            score[i] = (float)(((nx / magnitude) * ox + (ny / magnitude) * oy) / fmax(size, 1e-6) *
                fmin(size * 2, 1));
        }
    free(distance);
    free(mask);
    return failed ? -1 : 0;
}

static int facing(const OrientationNode *nodes, int64_t count, float *score) {
    Member *members = malloc((size_t)(count ? count : 1) * sizeof(*members));
    int32_t *ids = malloc((size_t)(count ? count : 1) * sizeof(*ids));
    if (!members || !ids) {
        free(ids);
        free(members);
        return -1;
    }
    for (int64_t i = 0; i < count; i++) {
        double band = floor(nodes[i].z / BLOCK_VOXELS);
        if (!(band >= INT32_MIN && band <= INT32_MAX && isfinite(nodes[i].x) &&
                isfinite(nodes[i].y))) {
            free(ids);
            free(members);
            return -1;
        }
        members[i] = (Member){.band = (int32_t)band, .index = (int32_t)i};
    }
    qsort(members, (size_t)count, sizeof(*members), member_before);
    for (int64_t i = 0; i < count; i++)
        ids[i] = members[i].index;
    int failed = 0;
    for (int64_t begin = 0; begin < count && !failed;) {
        int64_t end = begin + 1;
        while (end < count && members[end].band == members[begin].band)
            end++;
        failed = face_section(nodes, ids + begin, (int32_t)(end - begin), score);
        begin = end;
    }
    free(ids);
    free(members);
    return failed;
}

int hx_orientation_repair(OrientationNode *nodes, int64_t count, const OrientationRun *runs,
    int64_t run_count, uint8_t *keep, int64_t report[3], double ratios[2]) {
    float *score = malloc((size_t)(count ? count : 1) * sizeof(*score));
    int32_t *block = malloc((size_t)(count ? count : 1) * sizeof(*block));
    if (!score || !block || facing(nodes, count, score)) {
        free(block);
        free(score);
        return -1;
    }
    int64_t groups = hx_level_blocks(count, nodes, runs, run_count, 16.0f, block);
    double *sum = calloc((size_t)(groups > 0 ? groups : 1), sizeof(*sum));
    int64_t *unary = malloc((size_t)(groups > 0 ? groups : 1) * sizeof(*unary));
    if (groups < 0 || !sum || !unary) {
        free(unary);
        free(sum);
        free(block);
        free(score);
        return -1;
    }
    for (int64_t i = 0; i < count; i++)
        sum[block[i]] += score[i];
    for (int64_t i = 0; i < groups; i++)
        unary[i] = (int64_t)nearbyint(sum[i] * (-0.01 * ORIENTATION_UNITS));
    free(sum);
    free(score);
    uint64_t handle = 0;
    int64_t pairs =
        hx_level_block_pairs(groups, block, nodes, runs, run_count, ORIENTATION_UNITS, &handle);
    size_t pair_count = (size_t)(pairs > 0 ? pairs : 1);
    int32_t *a = malloc(pair_count * sizeof(*a));
    int32_t *b = malloc(pair_count * sizeof(*b));
    int32_t *weight = malloc(pair_count * sizeof(*weight));
    uint8_t *cut = calloc((size_t)(groups > 0 ? groups : 1), 1);
    uint8_t *flip = malloc((size_t)(count ? count : 1));
    if (pairs < 0 || !a || !b || !weight || !cut || !flip) {
        free(flip);
        free(cut);
        free(weight);
        free(b);
        free(a);
        free(unary);
        free(block);
        return -1;
    }
    hx_level_terms_take(handle, a, b, weight);
    int64_t flow = hx_level_two_labels(groups, pairs, a, b, weight, unary, cut);
    free(weight);
    free(b);
    free(a);
    free(unary);
    int64_t cut_count = 0;
    for (int64_t i = 0; i < count; i++) {
        flip[i] = cut[block[i]];
        cut_count += flip[i] != 0;
    }
    free(cut);
    free(block);
    int64_t log = 0;
    int64_t domains =
        flow < 0 ? -1 : hx_level_domains(count, nodes, runs, run_count, flip, keep, &log);
    free(flip);
    if (domains < 0)
        return -1;
    int64_t kept = 0;
    for (int64_t i = 0; i < count; i++)
        kept += keep[i] != 0;
    report[0] = log;
    report[1] = domains;
    report[2] = groups;
    ratios[0] = count ? (double)kept / count : NAN;
    ratios[1] = count ? (double)cut_count / count : NAN;
    return 0;
}

void hx_orientation_block(
    OrientationNode *nodes, int64_t count, const double *cx, const double *cy, float *theta) {
    for (int64_t i = 0; i < count; i++) {
        OrientationNode *n = nodes + i;
        double x = n->x - cx[i], y = n->y - cy[i];
        if (n->nx * x + n->ny * y < 0) {
            n->nx = -n->nx;
            n->ny = -n->ny;
            n->nz = -n->nz;
        }
        theta[i] = (float)atan2(y, x);
    }
}
