#ifndef ASM_AUDIT_INCLUDED
#define ASM_AUDIT_INCLUDED
#include "asm_field.h"

typedef struct AsmAuditStats {
    size_t source_faces, accounted_faces, represented_faces, absent_faces, multiply_owned_faces, altered_vertices, foreign_faces;
    size_t placed_charts, unplaced_charts, excluded_charts, faces, invalid_faces, strict_bad_faces, bad_metric_charts;
    size_t obligations, passing_seams, unresolved_seams, continuity_components, overlapping_pairs, self_overlapping_pairs;
    size_t repair_obligations, passing_repair_seams, unresolved_repair_seams;
    size_t source_regions, represented_source_regions, fractured_source_regions;
    size_t unresolved_present_repair_seams, unresolved_missing_repair_seams;
    double source_area, represented_area, excluded_area, unplaced_area, missing_area;
    double within10_area, within25_area, sigma_min, sigma_max, overlap_area, largest_connected_area;
    double largest_possible_connected_area, source_region_connected_area;
    size_t torn_seams, torn_cuts, torn_between_pieces;
    size_t untorn_source_regions, untorn_fractured_source_regions;
    double torn_length, untorn_source_region_connected_area;
    double boundary_length, stitched_length, border_length;
    int complete, source_preserved, geometry_qualified;
    int legacy_geometry_qualified;
    int contacts_complete;
    size_t contact_leaf_pairs, contact_leaf_pair_budget;
    char source_digest[65], field_digest[65], obligations_digest[65];
} AsmAuditStats;

int AsmAudit_run(const AsmRun *run, const char *out_dir, AsmAuditStats *out);
int AsmAudit_checkpoint(const AsmRun *run, const char *out_dir, AsmAuditStats *out);
int AsmAudit_encoded(Arena_T arena, const char *out_dir, const char *stem,
                      const float *xyz, const double *uv, size_t nv,
                      const int32_t *faces, size_t nf, const int32_t *chart,
                      const int32_t *component, const int32_t *cube);
int AsmAudit_selftest(void);
#endif
