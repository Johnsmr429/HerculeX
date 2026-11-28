#ifndef ASM_CLEAN_INCLUDED
#define ASM_CLEAN_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../common/mesh_bin.h"
#include "asm_types.h"

typedef struct AsmCleanOpts {
    double blob_area_per_face;
    double blob_double_sided;
    double min_chart_area;
    double stress_band;
    int    flatten_iters;
} AsmCleanOpts;

typedef struct AsmCleanStats {
    size_t components;
    size_t blobs, tiny, nonmanifold, handles, flat_failed, suspect, kept;
    size_t verts_in, faces_in;
    size_t verts_kept, faces_kept, verts_blob, faces_blob;
    double area_in, area_kept, area_blob;
    double flatten_sec;
    size_t degenerate_split;
    size_t degenerate_faces;
    size_t crumple_split;
    size_t crumple_pieces;
    size_t crumple_faces;
    double crumple_area_excluded;
} AsmCleanStats;

void AsmClean_default_opts(AsmCleanOpts *o);

int AsmClean_cube(Arena_T scratch, Arena_T persist,
                  const MeshBinData *mesh, int32_t cube,
                  const AsmCleanOpts *opts,
                  AsmChart **out_charts, size_t *out_n,
                  AsmCleanStats *stats);

void AsmClean_boundary_mask(Arena_T scratch, size_t nv,
                            const int32_t *faces, size_t nf, uint8_t *mark);

int AsmClean_selftest(void);

#endif
