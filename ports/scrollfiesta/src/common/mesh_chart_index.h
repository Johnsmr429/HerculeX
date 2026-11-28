#ifndef MESH_CHART_INDEX_INCLUDED
#define MESH_CHART_INDEX_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "mesh_bin.h"

typedef struct MeshChartRec {
    int32_t chart;
    uint64_t v0, v1;
    uint64_t f0, f1;
    double lo[3], hi[3];
    double uvlo[2], uvhi[2];
} MeshChartRec;

typedef struct MeshChartIndex {
    MeshChartRec *rec;
    size_t n;
    MeshBinHeader mesh;
    uint64_t chart_bytes;
} MeshChartIndex;

int MeshChartIndex_build(const char *vmesh, const char *chart_i32, const char *out_idx);

typedef struct MeshChartStamp {
    MeshBinHeader mesh;
    uint64_t chart_bytes;
    int64_t mtime;
} MeshChartStamp;
int MeshChartIndex_stamp(const char *vmesh, const char *chart_i32, MeshChartStamp *out);

int MeshChartIndex_build_mem(const MeshChartStamp *stamp, const char *vmesh, const char *chart_i32,
                             const MeshBinData *m, const int32_t *vchart, const char *out_idx);

int MeshChartIndex_load(const char *idx, const char *vmesh, MeshChartIndex *out);
void MeshChartIndex_free(MeshChartIndex *ix);

int MeshChartIndex_read(const MeshChartIndex *ix, const char *vmesh, const size_t *sel, size_t nsel,
                        MeshBinData *out, int32_t **vchart);

int MeshChartIndex_read_side(const MeshChartIndex *ix, const char *path, size_t es, const size_t *sel, size_t nsel,
                             void **out);

int MeshChartIndex_path(const char *vmesh, char *out, size_t cap);

int MeshChartIndex_selftest(void);

#endif
