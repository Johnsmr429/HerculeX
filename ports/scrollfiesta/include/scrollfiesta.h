#ifndef SCROLLFIESTA_H_INCLUDED
#define SCROLLFIESTA_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SCROLLFIESTA_VERSION_MAJOR 0
#define SCROLLFIESTA_VERSION_MINOR 10
#define SCROLLFIESTA_VERSION_PATCH 0

#define SCROLLFIESTA_ABI_VERSION   2u

#if defined(_WIN32) && defined(SCROLLFIESTA_SHARED)
  #if defined(SCROLLFIESTA_BUILD)
    #define SF_API __declspec(dllexport)
  #else
    #define SF_API __declspec(dllimport)
  #endif
#else
  #define SF_API
#endif

typedef enum sf_status {
    SF_OK              = 0,
    SF_ERROR           = -1,
    SF_ERROR_BAD_ARG   = -2,
    SF_ERROR_OOM       = -3,
    SF_ERROR_IO        = -4,
    SF_ERROR_TIMEOUT   = -5,
    SF_CANCELLED       = -6,
    SF_ERROR_NO_TIFF   = -7,
    SF_ERROR_ABI       = -8,
    SF_ERROR_UNSUPPORTED = -9
} sf_status;

typedef struct sf_mesh {
    float   *vertices;
    int32_t *faces;
    float   *vertex_normals;
    uint8_t *pin_mask;
    int32_t *vmap;
    size_t   n_vertices;
    size_t   n_faces;
} sf_mesh;

typedef struct sf_mesh_list {
    sf_mesh *items;
    size_t   count;
} sf_mesh_list;

typedef int  (*sf_progress_fn)(void *user, const char *stage, double fraction);
typedef void (*sf_log_fn)(void *user, const char *line);

typedef struct sf_kv { const char *key; const char *value; } sf_kv;

typedef struct sf_common_opts {
    int            n_threads;
    double         timeout_sec;
    sf_progress_fn progress;
    void          *progress_user;
    sf_log_fn      log;
    void          *log_user;
    const sf_kv   *tuning;
    size_t         n_tuning;
} sf_common_opts;

typedef struct sf_topology_report {
    long   n_verts, n_faces, n_edges;
    long   n_boundary_edges;
    long   n_manifold_edges;
    long   n_nonmanifold_edges;
    long   n_boundary_loops;
    long   euler;
    double genus;
    long   n_unref_verts;
    int    is_disk;
    long   n_components;
} sf_topology_report;

typedef struct sf_developability_report {
    double nondev_fraction;
    double mean_lambda;
} sf_developability_report;

typedef struct sf_cleanup_config {
    sf_common_opts common;
    int    manifold_repair;
    int    reorient;
    int    fill_pinholes;
    int    sliver_cleanup;
    double sliver_min_alt;
    double degen_area;
    double max_collapse_len;
    float  cull_min_area_frac;
    int    respect_pins;
} sf_cleanup_config;

typedef struct sf_cleanup_report {
    size_t nm_edges_resolved, faces_deleted, pinch_splits, orient_flips;
    size_t pinholes_filled, tris_added, loops_skipped;
    size_t n_flips, n_collapses;
    size_t components_culled;
} sf_cleanup_report;

typedef struct sf_holefill_config {
    sf_common_opts common;
    int interior_only;
} sf_holefill_config;

typedef struct sf_holefill_report {
    size_t n_loops, n_interior, n_filled;
} sf_holefill_report;

typedef struct sf_decimate_config {
    sf_common_opts common;
    float  target_ratio;
    size_t target_faces;
    int    respect_pins;
} sf_decimate_config;

typedef struct sf_decimate_report {
    size_t faces_in, faces_out, verts_out;
} sf_decimate_report;

typedef struct sf_fair_config {
    sf_common_opts common;
    int    max_iters;
    double tol_grad;
    double tol_e;
    double max_disp_vox;
    int    pin_boundary;
} sf_fair_config;

typedef struct sf_fair_report {
    double e_initial, e_final, max_disp;
    int    iters, converged;
} sf_fair_report;

typedef struct sf_orient_config {
    sf_common_opts common;
    int   align_components;
    float align_radius;
} sf_orient_config;

typedef struct sf_orient_report {
    size_t flips, components, residual_same_dir, components_flipped;
} sf_orient_report;

typedef struct sf_split_report {
    size_t n_pieces;
    int    self_gated;
    double nondev_before;
    double nondev_after_max;
} sf_split_report;

typedef struct sf_split_peel_config {
    sf_common_opts common;
    double min_gap;
    double gap_depth;
    size_t min_comp_verts;
} sf_split_peel_config;

typedef struct sf_split_dev_config {
    sf_common_opts common;
    double eps;
    double gap_depth;
    size_t min_comp_verts;
    int    apply_gate;
    double gate_margin;
    double gate_abs_ceil;
} sf_split_dev_config;

typedef struct sf_split_bridge_config {
    sf_common_opts common;
    int    oracle_grid_size;
    double component_timeout_sec;
} sf_split_bridge_config;

typedef struct sf_split_overlap_config {
    sf_common_opts common;
    int    sheet_count;
    double component_timeout_sec;
} sf_split_overlap_config;

typedef struct sf_detangle_config {
    sf_common_opts common;
    int    enable_peel, enable_dev, enable_bridge, enable_overlap;
    size_t min_comp_verts;
    double component_timeout_sec;
} sf_detangle_config;

typedef struct sf_detangle_report {
    size_t n_pieces;
    size_t peel_splits, dev_splits, bridge_splits, overlap_splits;
} sf_detangle_report;

typedef struct sf_seamcut_config {
    sf_common_opts common;
    double accept_frac;
    int    r_max;
} sf_seamcut_config;

typedef struct sf_seamcut_report {
    long genus_before, genus_after;
    long loops_before, loops_after;
    long n_terminals, seam_edges;
    int  is_disk_after;
    long n_components;
} sf_seamcut_report;

typedef struct sf_handle_loop {
    int32_t *vertex_indices;
    size_t   n;
    double   length;
} sf_handle_loop;

typedef struct sf_bpa_config {
    sf_common_opts common;
    float rho;
} sf_bpa_config;

typedef struct sf_bpa_report {
    size_t n_faces;
} sf_bpa_report;

typedef struct sf_volume {
    const uint8_t *data;
    int nx, ny, nz;
} sf_volume;

typedef struct sf_pipeline_config {
    sf_common_opts common;
    int     cube_size;
    int     halo_voxels;
    int64_t origin_xyz[3];
    float   cvt_target_ratio;
    int     skip_simplify;
    int     enable_depth_peel;
    int     enable_dev_cut;
    int     enable_sever;
    const char *dump_dir;
} sf_pipeline_config;

typedef struct sf_pipeline_report {
    double t_extract, t_cvt, t_trim, t_dump;
    size_t n_bad_sheets;
} sf_pipeline_report;

typedef struct sf_weld_config {
    sf_common_opts common;
    float cube_size;
    float rho, rho_max;
    float weld_eps;
    int   fill_holes;
    int   cleanup;
    float band;
} sf_weld_config;

typedef struct sf_weld_report {
    size_t verts_welded, bridge_faces, holes_filled;
} sf_weld_report;

SF_API void        sf_version(int *major, int *minor, int *patch);
SF_API const char *sf_version_string(void);
SF_API const char *sf_status_str(sf_status s);

SF_API void *sf_malloc(size_t n);
SF_API void  sf_free(void *p);
SF_API void  sf_mesh_free(sf_mesh *m);
SF_API void  sf_mesh_list_free(sf_mesh_list *l);
SF_API void  sf_handle_loops_free(sf_handle_loop *loops, size_t n);
SF_API sf_status sf_mesh_validate(const sf_mesh *m, char *err, size_t err_len);

SF_API sf_status sf_mesh_load_obj(const char *path, sf_mesh *out);
SF_API sf_status sf_mesh_save_obj(const char *path, const sf_mesh *m);

SF_API sf_common_opts          sf_common_opts_default(void);
SF_API sf_cleanup_config       sf_cleanup_config_default(void);
SF_API sf_holefill_config      sf_holefill_config_default(void);
SF_API sf_decimate_config      sf_decimate_config_default(void);
SF_API sf_fair_config          sf_fair_config_default(void);
SF_API sf_orient_config        sf_orient_config_default(void);
SF_API sf_split_peel_config    sf_split_peel_config_default(void);
SF_API sf_split_dev_config     sf_split_dev_config_default(void);
SF_API sf_split_bridge_config  sf_split_bridge_config_default(void);
SF_API sf_split_overlap_config sf_split_overlap_config_default(void);
SF_API sf_detangle_config      sf_detangle_config_default(void);
SF_API sf_seamcut_config       sf_seamcut_config_default(void);
SF_API sf_bpa_config           sf_bpa_config_default(void);
SF_API sf_pipeline_config      sf_pipeline_config_default(void);
SF_API sf_weld_config          sf_weld_config_default(void);

SF_API sf_status sf_topology_audit(const sf_mesh *in, const sf_common_opts *opts,
                                   sf_topology_report *out);
SF_API sf_status sf_developability_energy(const sf_mesh *in, double eps,
                                          const sf_common_opts *opts,
                                          double **out_lambda  ,
                                          sf_developability_report *rep);
SF_API sf_status sf_count_sheets(const sf_mesh *in, int grid_size  ,
                                 const sf_common_opts *opts, int *out_sheets);

SF_API sf_status sf_cleanup(const sf_mesh *in, const sf_cleanup_config *cfg,
                            sf_mesh *out, sf_cleanup_report *rep);
SF_API sf_status sf_fill_holes(const sf_mesh *in, const sf_holefill_config *cfg,
                               sf_mesh *out, sf_holefill_report *rep);
SF_API sf_status sf_decimate(const sf_mesh *in, const sf_decimate_config *cfg,
                             sf_mesh *out, sf_decimate_report *rep);
SF_API sf_status sf_fair_developable(const sf_mesh *in, const sf_fair_config *cfg,
                                     sf_mesh *out, sf_fair_report *rep);
SF_API sf_status sf_orient(const sf_mesh *in, const sf_orient_config *cfg,
                           sf_mesh *out, sf_orient_report *rep);

SF_API sf_status sf_split_depth_peel(const sf_mesh *in, const sf_split_peel_config *cfg,
                                     sf_mesh_list *out, sf_split_report *rep);
SF_API sf_status sf_split_developability(const sf_mesh *in, const sf_split_dev_config *cfg,
                                         sf_mesh_list *out, sf_split_report *rep);
SF_API sf_status sf_split_bridges(const sf_mesh *in, const sf_split_bridge_config *cfg,
                                  sf_mesh_list *out, sf_split_report *rep);
SF_API sf_status sf_split_overlap(const sf_mesh *in, const sf_split_overlap_config *cfg,
                                  sf_mesh_list *out, sf_split_report *rep);
SF_API sf_status sf_detangle(const sf_mesh *in, const sf_detangle_config *cfg,
                             sf_mesh_list *out, sf_detangle_report *rep);

SF_API sf_status sf_cut_to_disk(const sf_mesh *in, const sf_seamcut_config *cfg,
                                sf_mesh *out, sf_seamcut_report *rep);
SF_API sf_status sf_sever_short_handles(const sf_mesh *in,
                                        double max_loop_len  ,
                                        const sf_common_opts *opts,
                                        sf_mesh *out, long *out_loops_cut);
SF_API sf_status sf_find_handle_loops(const sf_mesh *in, const sf_common_opts *opts,
                                      sf_handle_loop **out_loops, size_t *out_n);

SF_API sf_status sf_reconstruct_bpa(const float *points_xyz,
                                    const float *normals_xyz  ,
                                    size_t n_points, const sf_bpa_config *cfg,
                                    sf_mesh *out, sf_bpa_report *rep);
SF_API sf_status sf_mls_project(const float *points_xyz, size_t n_points,
                                float radius_vox, const sf_common_opts *opts,
                                float **out_points_xyz  ,
                                float **out_normals_xyz  );

SF_API sf_status sf_pipeline_run(const sf_volume *vol, const sf_pipeline_config *cfg,
                                 sf_mesh_list *out_full,
                                 sf_mesh_list *out_trimmed,
                                 sf_pipeline_report *rep);

SF_API sf_status sf_weld(const sf_mesh *meshes, size_t n_meshes,
                         const sf_weld_config *cfg,
                         sf_mesh *out, sf_weld_report *rep);

SF_API int sf_selftest(void);

typedef struct sf_api {
    uint32_t abi_version;
    uint32_t struct_size;

    void        (*version)(int *major, int *minor, int *patch);
    const char *(*version_string)(void);
    const char *(*status_str)(sf_status s);

    void      *(*malloc_)(size_t n);
    void       (*free_)(void *p);
    void       (*mesh_free)(sf_mesh *m);
    void       (*mesh_list_free)(sf_mesh_list *l);
    void       (*handle_loops_free)(sf_handle_loop *loops, size_t n);
    sf_status  (*mesh_validate)(const sf_mesh *m, char *err, size_t err_len);

    sf_status  (*mesh_load_obj)(const char *path, sf_mesh *out);
    sf_status  (*mesh_save_obj)(const char *path, const sf_mesh *m);

    sf_common_opts          (*common_opts_default)(void);
    sf_cleanup_config       (*cleanup_config_default)(void);
    sf_holefill_config      (*holefill_config_default)(void);
    sf_decimate_config      (*decimate_config_default)(void);
    sf_fair_config          (*fair_config_default)(void);
    sf_orient_config        (*orient_config_default)(void);
    sf_split_peel_config    (*split_peel_config_default)(void);
    sf_split_dev_config     (*split_dev_config_default)(void);
    sf_split_bridge_config  (*split_bridge_config_default)(void);
    sf_split_overlap_config (*split_overlap_config_default)(void);
    sf_detangle_config      (*detangle_config_default)(void);
    sf_seamcut_config       (*seamcut_config_default)(void);
    sf_bpa_config           (*bpa_config_default)(void);
    sf_pipeline_config      (*pipeline_config_default)(void);
    sf_weld_config          (*weld_config_default)(void);

    sf_status  (*topology_audit)(const sf_mesh *in, const sf_common_opts *opts,
                                 sf_topology_report *out);
    sf_status  (*developability_energy)(const sf_mesh *in, double eps,
                                        const sf_common_opts *opts,
                                        double **out_lambda,
                                        sf_developability_report *rep);
    sf_status  (*count_sheets)(const sf_mesh *in, int grid_size,
                               const sf_common_opts *opts, int *out_sheets);

    sf_status  (*cleanup)(const sf_mesh *in, const sf_cleanup_config *cfg,
                          sf_mesh *out, sf_cleanup_report *rep);
    sf_status  (*fill_holes)(const sf_mesh *in, const sf_holefill_config *cfg,
                             sf_mesh *out, sf_holefill_report *rep);
    sf_status  (*decimate)(const sf_mesh *in, const sf_decimate_config *cfg,
                           sf_mesh *out, sf_decimate_report *rep);
    sf_status  (*fair_developable)(const sf_mesh *in, const sf_fair_config *cfg,
                                   sf_mesh *out, sf_fair_report *rep);
    sf_status  (*orient)(const sf_mesh *in, const sf_orient_config *cfg,
                         sf_mesh *out, sf_orient_report *rep);

    sf_status  (*split_depth_peel)(const sf_mesh *in, const sf_split_peel_config *cfg,
                                   sf_mesh_list *out, sf_split_report *rep);
    sf_status  (*split_developability)(const sf_mesh *in, const sf_split_dev_config *cfg,
                                       sf_mesh_list *out, sf_split_report *rep);
    sf_status  (*split_bridges)(const sf_mesh *in, const sf_split_bridge_config *cfg,
                                sf_mesh_list *out, sf_split_report *rep);
    sf_status  (*split_overlap)(const sf_mesh *in, const sf_split_overlap_config *cfg,
                                sf_mesh_list *out, sf_split_report *rep);
    sf_status  (*detangle)(const sf_mesh *in, const sf_detangle_config *cfg,
                           sf_mesh_list *out, sf_detangle_report *rep);

    sf_status  (*cut_to_disk)(const sf_mesh *in, const sf_seamcut_config *cfg,
                              sf_mesh *out, sf_seamcut_report *rep);
    sf_status  (*sever_short_handles)(const sf_mesh *in, double max_loop_len,
                                      const sf_common_opts *opts,
                                      sf_mesh *out, long *out_loops_cut);
    sf_status  (*find_handle_loops)(const sf_mesh *in, const sf_common_opts *opts,
                                    sf_handle_loop **out_loops, size_t *out_n);

    sf_status  (*reconstruct_bpa)(const float *points_xyz, const float *normals_xyz,
                                  size_t n_points, const sf_bpa_config *cfg,
                                  sf_mesh *out, sf_bpa_report *rep);
    sf_status  (*mls_project)(const float *points_xyz, size_t n_points,
                              float radius_vox, const sf_common_opts *opts,
                              float **out_points_xyz, float **out_normals_xyz);

    sf_status  (*pipeline_run)(const sf_volume *vol, const sf_pipeline_config *cfg,
                               sf_mesh_list *out_full, sf_mesh_list *out_trimmed,
                               sf_pipeline_report *rep);

    sf_status  (*weld)(const sf_mesh *meshes, size_t n_meshes,
                       const sf_weld_config *cfg,
                       sf_mesh *out, sf_weld_report *rep);

    int        (*selftest)(void);
} sf_api;

SF_API const sf_api *sf_get_api(uint32_t requested_abi);
typedef const sf_api *(*sf_get_api_fn)(uint32_t requested_abi);

#ifdef __cplusplus
}
#endif

#endif
