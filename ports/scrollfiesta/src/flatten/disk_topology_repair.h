#ifndef DISK_TOPOLOGY_REPAIR_INCLUDED
#define DISK_TOPOLOGY_REPAIR_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    size_t components;
    size_t already_disks;
    size_t repaired_components;
    size_t failed_components;
    size_t vertices_in, vertices_out;
    size_t faces_in, faces_out;
    size_t boundary_loops_before, boundary_loops_after;
    size_t handles_opened;
    size_t seam_edges;
} DiskTopologyRepairStats;

int DiskTopologyRepair_process(Arena_T arena,
                               const float *verts, size_t nv,
                               const int32_t *faces, size_t nf,
                               float **out_verts, size_t *out_nv,
                               int32_t **out_faces, size_t *out_nf,
                               int32_t **out_vertex_source,
                               DiskTopologyRepairStats *stats);

int DiskTopologyRepair_selftest(void);

#endif
