#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

typedef unsigned long long Count;
typedef struct {
    int n, size, active, candidates, frontier, distance, turn;
    Count flow;
} Counters;

__device__ void queued(int id, int *marks, int stamp, int *vertices, int *size) {
    if (atomicAdd(marks + id, 0) != stamp && atomicExch(marks + id, stamp) != stamp)
        vertices[atomicAdd(size, 1)] = id;
}

__global__ void roots(const long long *terminal, int *label, int *front, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->n; i += blockDim.x * gridDim.x) {
        int h = terminal[i] < 0 ? 0 : c->n;
        label[i] = h;
        if (!h)
            front[atomicAdd(&c->size, 1)] = i;
    }
}

__global__ void begin_level(Counters *c) {
    c->frontier = c->size;
    c->size = 0;
    c->distance++;
}

__global__ void visit(const int *front, int *out, const int *first, const int2 *arcs,
    const int *cap, int *label, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->frontier;
        i += blockDim.x * gridDim.x) {
        int v = front[i];
        for (int e = first[v]; e < first[v + 1]; e++) {
            int2 a = arcs[e];
            if (cap[a.y] > 0 && atomicAdd(label + a.x, 0) == c->n &&
                atomicCAS(label + a.x, c->n, c->distance) == c->n)
                out[atomicAdd(&c->size, 1)] = a.x;
        }
    }
}

__global__ void start_active(
    const long long *terminal, const int *label, int *active, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->n; i += blockDim.x * gridDim.x)
        if (terminal[i] > 0 && label[i] < c->n)
            active[atomicAdd(&c->active, 1)] = i;
}

__global__ void begin_pass(Counters *c) {
    c->candidates = 0;
    c->turn++;
}

__global__ void discharge(const int *active, int *candidate, int *marks, const int *first,
    const int2 *arcs, int *cap, long long *terminal, const int *label, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->active;
        i += blockDim.x * gridDim.x) {
        int v = active[i], h = label[v];
        long long amount = (long long)atomicAdd((Count *)(terminal + v), 0ULL);
        if (amount <= 0 || h == c->n)
            continue;
        long long before = amount;
        int stamp = c->turn;
        for (int e = first[v]; amount > 0 && e < first[v + 1]; e++) {
            int2 a = arcs[e];
            if (h != label[a.x] + 1)
                continue;
            int allowed = cap[e];
            if (allowed <= 0)
                continue;
            int sent = amount < allowed ? (int)amount : allowed;
            cap[e] -= sent;
            cap[a.y] += sent;
            long long old = (long long)atomicAdd((Count *)(terminal + a.x), (Count)sent);
            if (old < 0) {
                long long take = -old < sent ? -old : sent;
                atomicAdd(&c->flow, (Count)take);
            }
            if (old + sent > 0)
                queued(a.x, marks, stamp, candidate, &c->candidates);
            amount -= sent;
        }
        long long now = (long long)atomicAdd((Count *)(terminal + v), (Count)(amount - before)) +
            amount - before;
        if (now > 0)
            queued(v, marks, stamp, candidate, &c->candidates);
    }
}

__global__ void new_heights(const int *candidate, int *pending, const int *first, const int2 *arcs,
    const int *cap, const long long *terminal, const int *old, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->candidates;
        i += blockDim.x * gridDim.x) {
        int v = candidate[i], low = c->n;
        if (terminal[v] <= 0) {
            pending[i] = -1;
            continue;
        }
        for (int e = first[v]; e < first[v + 1]; e++)
            if (cap[e] > 0)
                low = min(low, old[arcs[e].x] + 1);
        pending[i] = low;
    }
}

__global__ void activate(
    const int *candidate, const int *pending, int *active, int *label, Counters *c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c->candidates;
        i += blockDim.x * gridDim.x) {
        int h = pending[i];
        if (h < 0)
            continue;
        int v = candidate[i];
        label[v] = h;
        if (h < c->n)
            active[atomicAdd(&c->active, 1)] = v;
    }
}

typedef struct {
    int n, m;
    int *first, *cap, *label, *one, *two, *marks, *pending;
    int2 *arcs;
    long long *terminal;
    Counters *c;
    void *arena;
    cudaStream_t stream;
    int blocks;
    int *saved_cap;
    long long *saved_terminal;
} Flow;

extern "C" void *hx_flow_open(int n, int m, const int *first, const int2 *arcs) {
    Flow *f = (Flow *)calloc(1, sizeof(*f));
    if (!f)
        return NULL;
    cudaDeviceProp prop;
    f->n = n;
    f->m = m;
    size_t bytes =
        256 + sizeof(int2) * m + sizeof(long long) * n + sizeof(int) * (m + 6ULL * n + 1);
    size_t available = 0, total = 0;
    if (cudaMemGetInfo(&available, &total) || bytes > available ||
        available - bytes < 1024ULL * 1024 * 1024) {
        free(f);
        return NULL;
    }
    f->saved_cap = (int *)malloc((size_t)m * sizeof(int));
    f->saved_terminal = (long long *)malloc((size_t)n * sizeof(long long));
    if (!f->saved_cap || !f->saved_terminal || cudaGetDeviceProperties(&prop, 0) ||
        cudaStreamCreateWithFlags(&f->stream, cudaStreamNonBlocking) ||
        cudaMalloc(&f->arena, bytes)) {
        if (f->stream)
            cudaStreamDestroy(f->stream);
        free(f->saved_cap);
        free(f->saved_terminal);
        free(f);
        return NULL;
    }
    f->blocks = prop.multiProcessorCount * 4;
    f->c = (Counters *)f->arena;
    f->arcs = (int2 *)((char *)f->arena + 256);
    f->terminal = (long long *)(f->arcs + m);
    f->cap = (int *)(f->terminal + n);
    f->first = f->cap + m;
    f->label = f->first + n + 1;
    f->one = f->label + n;
    f->two = f->one + n;
    f->marks = f->two + n;
    f->pending = f->marks + n;
    if (cudaMemcpyAsync(
            f->first, first, sizeof(int) * (n + 1), cudaMemcpyHostToDevice, f->stream) ||
        cudaMemcpyAsync(f->arcs, arcs, sizeof(int2) * m, cudaMemcpyHostToDevice, f->stream) ||
        cudaStreamSynchronize(f->stream)) {
        cudaFree(f->arena);
        cudaStreamDestroy(f->stream);
        free(f->saved_cap);
        free(f->saved_terminal);
        free(f);
        return NULL;
    }
    return f;
}

static int copied(Flow *f, int *dest, const int *src) {
    return cudaMemcpyAsync(dest, src, sizeof(int), cudaMemcpyDeviceToHost, f->stream) ||
        cudaStreamSynchronize(f->stream);
}

static int blocks_for(Flow *f, int jobs) {
    int groups = (jobs + 127) / 128;
    return groups < 1 ? 1 : groups > f->blocks ? f->blocks : groups;
}

static int global_labels(Flow *f, int *active) {
    Counters *c = f->c;
    if (cudaMemsetAsync(&c->size, 0, sizeof(int), f->stream) ||
        cudaMemsetAsync(&c->distance, 0, sizeof(int), f->stream))
        return 1;
    roots<<<f->blocks, 128, 0, f->stream>>>(f->terminal, f->label, f->one, c);
    int count = f->n, *front = f->one, *out = f->two;
    while (count) {
        int groups = blocks_for(f, count);
        for (int i = 0; i < 8; i++) {
            begin_level<<<1, 1, 0, f->stream>>>(c);
            visit<<<groups, 128, 0, f->stream>>>(
                front, out, f->first, f->arcs, f->cap, f->label, c);
            std::swap(front, out);
        }
        if (copied(f, &count, &c->size))
            return 1;
    }
    if (cudaMemsetAsync(&c->active, 0, sizeof(int), f->stream))
        return 1;
    start_active<<<f->blocks, 128, 0, f->stream>>>(f->terminal, f->label, f->one, c);
    return copied(f, active, &c->active);
}

extern "C" long long hx_flow_run(void *opaque, int *cap, int64_t *term) {
    Flow *f = (Flow *)opaque;
    Counters data = {.n = f->n};
    Counters *c = f->c;
    if (cudaMemcpyAsync(c, &data, sizeof(data), cudaMemcpyHostToDevice, f->stream) ||
        cudaMemcpyAsync(f->cap, cap, sizeof(int) * f->m, cudaMemcpyHostToDevice, f->stream) ||
        cudaMemcpyAsync(
            f->terminal, term, sizeof(long long) * f->n, cudaMemcpyHostToDevice, f->stream) ||
        cudaMemsetAsync(f->marks, 0, sizeof(int) * f->n, f->stream))
        return -1;
    int passes = 0, active = 0;
    for (;;) {
        if (global_labels(f, &active))
            return -1;
        if (!active)
            break;
        for (int batch = 0; batch < 12 && active; batch++) {
            int groups = blocks_for(f, active);
            for (int i = 0; i < 8; i++) {
                begin_pass<<<1, 1, 0, f->stream>>>(c);
                discharge<<<groups, 128, 0, f->stream>>>(
                    f->one, f->two, f->marks, f->first, f->arcs, f->cap, f->terminal, f->label, c);
                new_heights<<<groups, 128, 0, f->stream>>>(
                    f->two, f->pending, f->first, f->arcs, f->cap, f->terminal, f->label, c);
                if (cudaMemsetAsync(&c->active, 0, sizeof(int), f->stream))
                    return -1;
                activate<<<groups, 128, 0, f->stream>>>(f->two, f->pending, f->one, f->label, c);
            }
            passes += 8;
            if (copied(f, &active, &c->active) || passes > 20000)
                return -1;
        }
    }
    if (cudaMemcpyAsync(&data, c, sizeof(data), cudaMemcpyDeviceToHost, f->stream) ||
        cudaMemcpyAsync(
            f->saved_cap, f->cap, sizeof(int) * f->m, cudaMemcpyDeviceToHost, f->stream) ||
        cudaMemcpyAsync(f->saved_terminal, f->terminal, sizeof(long long) * f->n,
            cudaMemcpyDeviceToHost, f->stream) ||
        cudaStreamSynchronize(f->stream))
        return -1;
    memcpy(cap, f->saved_cap, (size_t)f->m * sizeof(int));
    memcpy(term, f->saved_terminal, (size_t)f->n * sizeof(long long));
    return data.flow;
}

extern "C" void hx_flow_close(void *opaque) {
    Flow *f = (Flow *)opaque;
    if (!f)
        return;
    cudaStreamSynchronize(f->stream);
    cudaFree(f->arena);
    cudaStreamDestroy(f->stream);
    free(f->saved_terminal);
    free(f->saved_cap);
    free(f);
}
