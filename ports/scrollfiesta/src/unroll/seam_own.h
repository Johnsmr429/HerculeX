#ifndef SEAM_OWN_INCLUDED
#define SEAM_OWN_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "../common/raw_sample.h"
#include "piece_set.h"

typedef struct SeamOwnOpts {
    float  axis_point[3];
    float  axis_dir[3];
    const char *raw_dir;
    long   raw_chunk;
    CubeTable *ct;
    double pitch;
    double radius_gate;
    double cell;
    double seam_gate3d;
    double spiral_a, spiral_b;
    double tex_du, tex_dv;
    double normal_range;
    int    normal_samples;
    size_t region_cap;
    int    grid_cap;
    int    seam_cut;
    int    seam_halfband;
    int    min_overlap_px;
    int    rehome;
    int    rehome_min_faces;
    double rehome_frac_tol;
    int    verbose;
} SeamOwnOpts;

void SeamOwnOpts_default(SeamOwnOpts *o);

int SeamOwn_read_index(const char *placed_dir, SeamOwnOpts *o);

typedef struct SeamOwnResult {
    uint8_t *face_keep;
    uint8_t *face_dec;
    int32_t *face_region;

    size_t n_true_pairs, n_seam_pairs;
    size_t n_mystery_pairs;
    size_t n_regions, n_layers_total, max_region_faces;
    size_t n_vote_picks, n_energy_picks, n_largest_picks;
    size_t n_multicut_fallbacks, n_regions_skipped_cap, n_regions_skipped_grid;
    size_t n_kept, n_dropped;
    size_t n_seam_regions, n_seams_abut_only, n_seams_skipped;
    size_t n_seam_dropped, n_seam_restored;
    size_t n_rehome_layers, n_rehomed_faces;
    size_t n_rehome_dup, n_rehome_blocked, n_rehome_adjacent;
    size_t n_rehome_small, n_rehome_incoherent;
    size_t multi_cells_before, multi_cells_after;
    size_t prior_agree, prior_disagree;
    double energy_mean;
    int    have_texture;
    double seconds;
} SeamOwnResult;

int SeamOwn_run(Arena_T arena, PieceSet *ps,
                const SeamOwnOpts *opts, SeamOwnResult *out);

int SeamOwn_selftest(void);

#endif
