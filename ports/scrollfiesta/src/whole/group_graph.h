#ifndef GROUP_GRAPH_INCLUDED
#define GROUP_GRAPH_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "cube_schedule.h"
#include "placed_cube.h"

typedef struct {
    int32_t cube;
    int32_t gid;
    int32_t k;
    double  du;
    double  prior;
    double  du_prior;
    int32_t n_prior;
    int32_t comp;
} GGNode;

typedef struct {
    int32_t a, b;
    int32_t obs;
    double  du;
    double  conf;
    double  mad;
    int32_t n;
    uint8_t in_tree;
} GGEdge;

typedef struct {
    GGNode  *nodes;
    size_t   n_nodes;
    GGEdge  *edges;
    size_t   n_edges;
    int32_t *cube_node0;
    size_t   n_cubes;
    int32_t  n_comp;
    double   energy;
    size_t   n_frustrated;
    int32_t  moves_applied;
    double   edge_redundancy;
    double   anchor_weight_effective;
    int32_t  du_gauge_components;
    double   du_prior_resid_median, du_prior_resid_p95, du_prior_resid_max;
    size_t   pairs_used, pairs_rej_radius, pairs_rej_frac;
    size_t   edges_rej_prior, edges_rej_dr;
} GroupGraph;

typedef struct {
    double pair_gate;
    double radius_gate;
    double frac_gate;
    double prior_gate;
    double edge_dr_gate;
    int    intra_prior;
    double intra_frac_max;
    double intra_conf;
    int    min_edge_pairs;
    int    conf_n_cap;
    double conf_mad0;
    int    prior_min_verts;
    int    raw_component_gauge;
    int    raw_du_gauge;
    int    physical_du_gauge;
    double anchor_weight;
    double anchor_redundancy_ref;
    int    do_moves;
    int    max_moves;
    int    du_passes;
    int    verbose;
} GroupGraphOpts;

void GroupGraphOpts_default(GroupGraphOpts *o);

int GroupGraph_build(Arena_T arena, const CubeNode *cnodes, size_t n_cubes,
                     SkinVert *const *skins, const size_t *nskin,
                     const float axis_point[3],
                     double spiral_a, double spiral_b, double pitch,
                     const GroupGraphOpts *opts, GroupGraph *out);

int GroupGraph_solve(Arena_T arena, GroupGraph *g, const GroupGraphOpts *opts);

int GroupGraph_cube_reg(Arena_T arena, const GroupGraph *g, size_t cube,
                        PlacedReg *out);

int32_t GroupGraph_node_of(const GroupGraph *g, size_t cube, int32_t gid);

int GroupGraph_selftest(void);

#endif
