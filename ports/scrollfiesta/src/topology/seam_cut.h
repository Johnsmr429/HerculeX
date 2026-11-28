#ifndef SEAM_CUT_INCLUDED
#define SEAM_CUT_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct {
    double accept_frac;
    int    r_max;
} SeamCutOpts;

typedef struct {
    float   *verts;
    int32_t *faces;
    size_t   nv, nf;
    int32_t *vmap;
    long     genus_before, genus_after;
    long     loops_before, loops_after;
    long     n_terminals;
    long     seam_edges;
    int      is_disk_after;
} SeamCutResult;

int SeamCut_run(Arena_T arena, const float *verts, size_t nv,
                const int32_t *faces, size_t nf,
                const SeamCutOpts *opts, SeamCutResult *out);

int SeamCut_repair_manifold(Arena_T arena, const float *verts, size_t nv,
                            const int32_t *faces, size_t nf,
                            float **out_verts, size_t *out_nv,
                            int32_t **out_faces, size_t *out_nf,
                            int32_t **out_vmap);

int SeamCut_to_boundary(Arena_T arena, const float *verts, size_t nv,
                         const int32_t *faces, size_t nf,
                         const uint8_t *terminals, SeamCutResult *out);

typedef int (*SeamCutFanCheck)(void *context, int32_t vertex,
                               const int32_t *corners, const int32_t *rim,
                               size_t nf);
int SeamCut_to_boundary_checked(Arena_T arena, const float *verts, size_t nv,
                                 const int32_t *faces, size_t nf,
                                 const uint8_t *terminals,
                                 SeamCutFanCheck fan_check, void *context,
                                 SeamCutResult *out);

int SeamCut_to_boundary_eligible(Arena_T arena, const float *verts, size_t nv,
                                  const int32_t *faces, size_t nf,
                                  const uint8_t *terminals,
                                  const uint8_t *eligible_boundary,
                                  const uint8_t *blocked_vertices,
                                  SeamCutResult *out);

typedef enum SeamCutReason {
    SEAM_CUT_NOT_RUN=0, SEAM_CUT_COMPLETE, SEAM_CUT_INVALID_INPUT,
    SEAM_CUT_NONMANIFOLD, SEAM_CUT_RESERVED_TERMINAL,
    SEAM_CUT_INELIGIBLE_TERMINAL, SEAM_CUT_UNREACHABLE_TERMINAL,
    SEAM_CUT_FAN_CHECK_FAILED, SEAM_CUT_RAY_ALREADY_CUT,
    SEAM_CUT_RAY_BLOCKED, SEAM_CUT_RAY_UNREACHABLE,
    SEAM_CUT_OVERFULL_UNRESOLVED, SEAM_CUT_SURGERY_FAILED
} SeamCutReason;

typedef struct SeamCutDiagnostic {
    SeamCutReason reason;
    size_t requested_terminals, overfull_terminals, interior_terminals;
    size_t reserved_vertices, reserved_terminals;
    size_t boundary_vertices, eligible_boundary_vertices, blocked_boundary_vertices;
    size_t unreachable_terminals, additional_rays, seam_edges;
    int32_t first_reserved_terminal, first_unreachable_terminal;
    int32_t terminal, ray;
    int input_counts_complete, boundary_counts_complete;
    int reachability_complete, routes_complete, surgery_complete;
} SeamCutDiagnostic;

const char *SeamCutReason_name(SeamCutReason reason);
int SeamCut_to_boundary_eligible_report(Arena_T arena, const float *verts,
                                         size_t nv,const int32_t *faces,size_t nf,
                                         const uint8_t *terminals,
                                         const uint8_t *eligible_boundary,
                                         const uint8_t *blocked_vertices,
                                         SeamCutResult *out,
                                         SeamCutDiagnostic *diagnostic);

struct SlimRefineDirection;
int SeamCut_direction_terminals(Arena_T arena,const float *verts,size_t nv,
                                 const int32_t *faces,size_t nf,
                                 const struct SlimRefineDirection *directions,
                                 uint8_t *terminals,size_t *count);

typedef struct {
    int32_t *verts;
    size_t   n;
    double   length;
} HandleLoop;

int SeamCut_handle_loops(Arena_T arena, const float *verts, size_t nv,
                         const int32_t *faces, size_t nf,
                         HandleLoop **out_loops, size_t *out_n);

int SeamCut_sever_short_handles(Arena_T arena, const float *verts, size_t nv,
                                const int32_t *faces, size_t nf, double max_loop_len,
                                float **out_verts, size_t *out_nv,
                                int32_t **out_faces, size_t *out_nf,
                                long *out_loops_cut);

int SeamCut_selftest(void);

#endif
