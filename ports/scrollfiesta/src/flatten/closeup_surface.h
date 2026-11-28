#ifndef CLOSEUP_SURFACE_H
#define CLOSEUP_SURFACE_H

#include <stddef.h>
#include <stdint.h>

typedef struct CloseupObs {
    size_t   n, cap;
    float   *uv;
    float   *p;
    float   *w;
    int32_t *chart;
} CloseupObs;

typedef struct CloseupObsGate {
    double stretch_ratio, stretch_floor;
    double max_edge3d;
    double min_scale, max_scale;
} CloseupObsGate;

void CloseupObsGate_default(CloseupObsGate *g);
void CloseupObs_free(CloseupObs *o);

int CloseupObs_add_faces(CloseupObs *o, const float *verts, const double *uv, const int32_t *faces, size_t nf,
                         const uint8_t *face_use, const int32_t *face_chart, const CloseupObsGate *gate,
                         size_t *n_used, size_t *n_gated);

enum { CLOSEUP_NODE_NONE = 0, CLOSEUP_NODE_DATA = 1, CLOSEUP_NODE_FILL = 2 };

typedef struct CloseupGrid {
    size_t   ni, nj;
    double   u0, v0, h;
    float   *pos;
    float   *tu, *tv;
    float   *nrm;
    float   *dist;
    float   *support;
    int32_t *chart;
    int32_t *component;
    uint8_t *state;
} CloseupGrid;

int  CloseupGrid_alloc(CloseupGrid *g, double umin, double umax, double vmin, double vmax, double h);
void CloseupGrid_free(CloseupGrid *g);

int  CloseupGrid_save(const char *path, const CloseupGrid *g, const int32_t *meta, size_t nmeta);
int  CloseupGrid_load(const char *path, CloseupGrid *g, int32_t *meta, size_t cap, size_t *nmeta);

int  CloseupGrid_taylor(const CloseupGrid *g, size_t i, size_t j, double u, double v, double p[3]);
int  CloseupGrid_eval(const CloseupGrid *g, double u, double v, double p[3], double n[3]);

typedef struct CloseupFitOpts {
    double fit_px;
    int    min_obs;
    double data_reach_px;
    double fill_px;
    double fill_fit_min_px, fill_fit_max_px;
    double robust_vox;
    int    robust_rounds;
    double tear_ratio;
    double tear_float_vox;
    const int32_t *chart_wrap;
    int32_t nchart_wrap;
    int    bridge;
    double bridge_px, bridge_agree;
    int    variational_holes;
} CloseupFitOpts;

void CloseupFitOpts_default(CloseupFitOpts *o);
void CloseupFitOpts_complete(CloseupFitOpts *o);

typedef struct CloseupFitStats {
    size_t nodes, data_nodes, fill_nodes, fill_rejected, data_failed;
    size_t obs, obs_downweighted;
    double resid_p50, resid_p90, resid_p99;
    double seconds;
    size_t torn_stretch, torn_float;
    size_t blobs, blobs_with_wrap, bridges;
    size_t variational_holes, variational_filled, variational_refused, variational_nodes;
} CloseupFitStats;

int CloseupSurface_fill_holes(CloseupGrid *g, double max_distance_px, CloseupFitStats *st);

int CloseupSurface_complete_fixed(CloseupGrid *g, double max_distance_px, CloseupFitStats *st);

int CloseupSurface_complete_fixed_uv(CloseupGrid *g, double *uv_nodes, double max_distance_px, CloseupFitStats *st);

typedef struct CloseupProposalStats {
    size_t holes, proposed, refused, contact_refused, nodes;
    int metric_qualification_pending;
} CloseupProposalStats;

int CloseupSurface_propose_fixed(CloseupGrid *g, double max_distance_px, int32_t **patch_ids,
                                 CloseupProposalStats *st);

typedef struct CloseupProposalMetricStats {
    size_t triangles, patches, passed_patches, refused_patches;
    size_t metric_failures, orientation_failures, invalid_faces, empty_patches;
    double min_squared_scale, max_squared_scale;
    int metric_qualified;
} CloseupProposalMetricStats;

int CloseupSurface_audit_proposal_metric(const float *verts, size_t nv, const double *uv, const double *frame_uv,
                                        const int32_t *faces, size_t nf, const int32_t *face_patch,
                                        size_t npatches, uint8_t *patch_pass, CloseupProposalMetricStats *st);

typedef struct CloseupRebuildResult {
    size_t nodes, faces, patches;
    int32_t *node_patch;
    int32_t *component_group;
    int32_t *face_nodes;
    int32_t *face_patch;
} CloseupRebuildResult;

typedef struct CloseupRebuildWorldStats {
    size_t pairs, observed_observed, observed_inferred, inferred_inferred;
    size_t overlaps, stabs, folds;
    int complete;
} CloseupRebuildWorldStats;

typedef struct CloseupRebuildStats {
    size_t data_nodes, old_fill_nodes, domain_nodes, enclosed_gap_nodes;
    size_t source_components, supported_groups, certified_joins, refused_joins;
    size_t patches, proposed_patches, refused_patches, proposed_nodes, proposed_faces;
    size_t reach_refused, orientation_refused, contact_refused, unsupported_nodes;
    size_t separated_cells;
    size_t alternate_diagonals;
    size_t screened_nodes;
    double tangent_screen_energy;
    size_t old_fill_start_nodes, target_start_patches, motion_sweeps, motion_vertices;
    size_t coupled_passes, coupled_blocks, coupled_vertices, coupled_solves;
    size_t orientation_before_motion, orientation_after_motion;
    size_t transport_components, transport_anchored_components, transport_unanchored_components;
    size_t transport_unanchored_faces, transport_degenerate_faces, transport_bad_edges;
    size_t transport_winding_edges, transport_nonmanifold_edges;
    CloseupRebuildWorldStats world_exact, world_proximity;
    double target_error_before_motion, target_error_after_motion;
    int metric_qualification_pending;
} CloseupRebuildStats;

typedef struct CloseupRebuildDomainStats {
    size_t required_nodes, required_enclosed_gap_nodes, added_nodes, nodes, required_beyond_reach, data_donors;
    double reach_px, source_frame_bounds[4];
} CloseupRebuildDomainStats;

int CloseupSurface_rebuild_regrowth_domain(const CloseupGrid *g, uint8_t *domain,
                                           double max_distance_px, CloseupRebuildDomainStats *st);

int CloseupSurface_rebuild_generated(CloseupGrid *g, const uint8_t *domain,
                                     const int32_t *observed_faces, size_t observed_nf,
                                     double max_distance_px, CloseupRebuildResult *result,
                                     CloseupRebuildStats *st);
void CloseupRebuildResult_free(CloseupRebuildResult *result);

int CloseupSurface_fit(CloseupGrid *g, CloseupObs *obs, const CloseupFitOpts *o, CloseupFitStats *st);

typedef struct CloseupFitCache CloseupFitCache;
CloseupFitCache *CloseupFitCache_new(void);
void CloseupFitCache_free(CloseupFitCache **cache);
int CloseupSurface_fit_cached(CloseupGrid *g, CloseupObs *obs, const CloseupFitOpts *o, CloseupFitStats *st,
                              CloseupFitCache *cache);

int CloseupSurface_orient(CloseupGrid *g, const float *inward, double *agree);

typedef struct CloseupFlatStats {
    size_t coarse_nodes, coarse_tris, components, iterations;
    size_t coarse_fringe;
    double energy_first, energy_last;
    double energy_hist[64];
    double logs_p50_before, logs_p99_before;
    double logs_p50_after, logs_p99_after;
    double area_ratio;
    double seconds;
} CloseupFlatStats;

int CloseupSurface_flatten(const CloseupGrid *g, double coarse_px, int iters, double tol, double *uv_out,
                           CloseupFlatStats *st);

int CloseupSurface_flatten_slim(const CloseupGrid *g, double coarse_px, int iters, double *uv_out, CloseupFlatStats *st);

typedef struct CloseupDedupStats {
    size_t pairs;
    size_t data_pairs;
    size_t dropped_fill;
    double seconds;
} CloseupDedupStats;

int CloseupSurface_dedup_fill(CloseupGrid *g, double r_vox, double frame_px, double cos_min, CloseupDedupStats *st);

typedef struct CloseupApronStats {
    size_t fill_before;
    size_t dropped;
    double seconds;
} CloseupApronStats;

int CloseupSurface_trim_apron(CloseupGrid *g, double reach_px, CloseupApronStats *st);

int CloseupSurface_mesh(const CloseupGrid *g, const double *uv_nodes, float **verts, double **uv, float **nrm,
                        int32_t **faces, int32_t **node_of_vert, size_t *nv, size_t *nf);

typedef struct CloseupFrame {
    int    reflect;
    double theta, cu, cv;
    double hand_vote;
    double hand_agree;
    double axial_gx, axial_gy;
} CloseupFrame;

int  CloseupSurface_frame(const float *verts, const double *uv, const int32_t *faces, size_t nf,
                          const uint8_t *face_use, const double *axial, const float *inward, double cu, double cv,
                          CloseupFrame *fr);
void CloseupFrame_apply(const CloseupFrame *fr, double u, double v, double *fu, double *fv);

int  CloseupSurface_raster_attr(const double *uv, const int32_t *faces, size_t nf, const uint8_t *attr,
                                double umin, double vmin, size_t W, size_t H, uint8_t *out);

double CloseupSurface_rot_angle(double m00, double m01, double m10, double m11);

int CloseupSurface_selftest(void);
int CloseupSurface_rebuild_selftest(void);

#endif
