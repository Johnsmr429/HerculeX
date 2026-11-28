#ifndef INTERSECTION_CLEANUP_INCLUDED
#define INTERSECTION_CLEANUP_INCLUDED

#include <stddef.h>
#include <stdint.h>

enum {
    INTERSECTION_HIT_OVERLAP = 1,
    INTERSECTION_HIT_STAB = 2,
    INTERSECTION_HIT_FOLD = 3
};

enum {
    INTERSECTION_HIT_MASK_OVERLAP = 1u << (INTERSECTION_HIT_OVERLAP - 1),
    INTERSECTION_HIT_MASK_STAB = 1u << (INTERSECTION_HIT_STAB - 1),
    INTERSECTION_HIT_MASK_FOLD = 1u << (INTERSECTION_HIT_FOLD - 1),
    INTERSECTION_HIT_MASK_ALL = INTERSECTION_HIT_MASK_OVERLAP |
                                INTERSECTION_HIT_MASK_STAB |
                                INTERSECTION_HIT_MASK_FOLD
};

typedef struct {
    double gap_max;
    double parallel_angle_deg;
    double max_delete_fraction;
    double max_delete_fraction_hard;
    size_t min_delete_budget_faces;
    size_t max_conflicts;
    int include_hinges;
    unsigned hit_kind_mask;
} IntersectionCleanupParams;

typedef struct {
    size_t candidate_pairs;
    size_t overlap_pairs;
    size_t stab_pairs;
    size_t fold_pairs;
    size_t conflicts;
    size_t faces_deleted;
    size_t delete_budget;
    size_t components_touched;
    size_t max_conflict_degree;
    int budget_rejected;
} IntersectionCleanupStats;

typedef int (*IntersectionCleanupConflictVisitor)(size_t face_a,
                                                  size_t face_b,
                                                  int hit_kind,
                                                  void *context);

typedef int (*IntersectionCleanupCandidateVisitor)(size_t face_a,size_t face_b,void *context);
typedef struct IntersectionCleanupCandidateStats {
    size_t overlapping_pairs,selected_pairs;
    int complete;
} IntersectionCleanupCandidateStats;

int IntersectionCleanup_visit_candidates(const float *verts,size_t nv,
    const int32_t *faces,size_t nf,const uint8_t *selected,double gap,
    IntersectionCleanupCandidateVisitor visitor,void *context,
    IntersectionCleanupCandidateStats *stats);

void IntersectionCleanup_default_params(IntersectionCleanupParams *params);

int IntersectionCleanup_audit(const float *verts, size_t nv,
                              const int32_t *faces, size_t nf,
                              const uint8_t *face_mask,
                              const IntersectionCleanupParams *params,
                              size_t *face_conflict_degree,
                              IntersectionCleanupStats *stats);

int IntersectionCleanup_audit_visit(
                              const float *verts, size_t nv,
                              const int32_t *faces, size_t nf,
                              const uint8_t *face_mask,
                              const IntersectionCleanupParams *params,
                              size_t *face_conflict_degree,
                              IntersectionCleanupConflictVisitor visitor,
                              void *visitor_context,
                              IntersectionCleanupStats *stats);

int IntersectionCleanup_audit_visit_parallel(
                              const float *verts, size_t nv,
                              const int32_t *faces, size_t nf,
                              const uint8_t *face_mask,
                              const IntersectionCleanupParams *params,
                              size_t *face_conflict_degree,
                              IntersectionCleanupConflictVisitor visitor,
                              void *visitor_context,
                              IntersectionCleanupStats *stats);

int IntersectionCleanup_pair_test(const float *verts, size_t nv,
                                  const int32_t face_a[3],
                                  const int32_t face_b[3],
                                  const IntersectionCleanupParams *params,
                                  int *shared_out);

int IntersectionCleanup_cross_budget(
    const float *verts_a, size_t nv_a, const int32_t *faces_a, size_t nf_a,
    const float *verts_b, size_t nv_b, const int32_t *faces_b, size_t nf_b,
    double max_error, double *budget_a, double *budget_b,
    size_t *tested_pairs);

int IntersectionCleanup_partition_budget(const float *verts,size_t nv,
    const int32_t *faces,size_t nf,const int32_t *owner,double max_error,
    double *budget,size_t *tested_pairs);

int IntersectionCleanup_process(const float *verts, size_t nv,
                                int32_t *faces, size_t *pnf,
                                const uint8_t *face_mask,
                                const IntersectionCleanupParams *params,
                                IntersectionCleanupStats *stats);

int IntersectionCleanup_selftest(void);

#endif
