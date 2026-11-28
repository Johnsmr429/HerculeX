#ifndef ASM_TYPES_INCLUDED
#define ASM_TYPES_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "../common/arena.h"
#include "../common/mesh_pile.h"

enum {
    ASM_CHART_BLOB          = 1u << 0,
    ASM_CHART_SUSPECT       = 1u << 1,
    ASM_CHART_EXTRAS        = 1u << 2,
    ASM_CHART_CUT_CHILD     = 1u << 3,
    ASM_CHART_HANDLE        = 1u << 4,
    ASM_CHART_NONMANIFOLD   = 1u << 5,
    ASM_CHART_FLAT_FAILED   = 1u << 6,
    ASM_CHART_TINY          = 1u << 7,
    ASM_CHART_MIRROR        = 1u << 8
};

#define ASM_CHART_EXCLUDED_MASK \
    (ASM_CHART_BLOB | ASM_CHART_EXTRAS | ASM_CHART_NONMANIFOLD | \
     ASM_CHART_FLAT_FAILED | ASM_CHART_TINY)

enum { ASM_PLACE_NONE, ASM_PLACE_ROOT, ASM_PLACE_SEAM, ASM_PLACE_LAYER };

typedef struct AsmChart {
    int32_t  id;
    int32_t  cube;
    int32_t  comp;
    int32_t  parent;
    uint32_t flags;
    int32_t  component;
    int32_t  placed;
    int32_t  placement_group, placement_state, placement_parent;
    size_t   placement_support;
    double   placement_rms;

    size_t   nv, nf;
    int32_t *vid;
    float   *xyz;
    float   *nrm;
    float   *uv;
    double  *placed_uv;
    int32_t *faces;
    uint8_t *boundary;
    uint8_t *repaired;

    double   area3d;
    double   area_uv;
    double   sigma_lo;
    double   sigma_hi;
    double   stress_frac;
    size_t   n_flipped;
    double   centroid[3];
    float    bbox_lo[3], bbox_hi[3];

    double   pose_x, pose_y, pose_theta;
} AsmChart;

enum {
    ASM_REL_DROPPED   = 1u << 0,
    ASM_REL_REPAIRED  = 1u << 1,
    ASM_REL_SUSPECT   = 1u << 2,
    ASM_REL_CONTACT   = 1u << 3,
    ASM_REL_PARITY    = 1u << 4,
    ASM_REL_SWITCHED  = 1u << 5,
    ASM_REL_WEAK      = 1u << 6,
    ASM_REL_LAYOUT    = 1u << 7,
    ASM_REL_SHORT     = 1u << 8,
    ASM_REL_READMIT   = 1u << 9,
    ASM_REL_CROSSWRAP = 1u << 10,
    ASM_REL_TORN      = 1u << 11
};

#define ASM_REL_PLACEMENT_ONLY (ASM_REL_WEAK | ASM_REL_SHORT)

typedef struct AsmRelation {
    int32_t  a, b;
    uint32_t flags;
    uint32_t pad0;
    double   theta, tx, ty;
    double   w_xy;
    double   w_theta;
    double   rms;
    double   seam_len;
    double   normal_gap;
    double   mean_gap;
    size_t   n_corr;
    int32_t  corr_first;
    int32_t  corr_count;
    double   robust_w;
    double   residual;
    double   flag_mass;
    uint32_t continuity;
    uint32_t pad1;
    double   continuity_rms;
} AsmRelation;
typedef char AsmRelation_has_no_padding[sizeof(AsmRelation) == 8*sizeof(int32_t) + 13*sizeof(double) + sizeof(size_t) ? 1 : -1];

typedef struct AsmTrim {
    int32_t face;
    uint32_t valid;
    double inverse[4];
} AsmTrim;

typedef struct AsmCorr {
    int32_t va, vb;
    float gap_a[2], gap_b[2];
    int32_t run;
    uint32_t valid;
    AsmTrim trim_a, trim_b;
} AsmCorr;

typedef struct AsmLayerPair {
    int32_t a, b;
    int32_t count;
    int8_t  side;
    double  d_median;
    int32_t hit_first, hit_count;
    int8_t  k;
    int8_t  nsign;
} AsmLayerPair;

typedef struct AsmLayerHit {
    int32_t a, va, b, fb;
    float   l0, l1;
    float   dist;
    int8_t  side;
    int8_t  order;
    int8_t  nsign;
} AsmLayerHit;

typedef struct AsmRun {
    Arena_T        arena;
    int            continuity_ready;

    MeshPileEntry *pile;
    size_t         n_cubes;

    AsmChart      *charts;
    size_t         n_charts, cap_charts;

    AsmRelation   *rels;
    size_t         n_rels, cap_rels;

    AsmCorr       *corr;
    size_t         n_corr, cap_corr;

    AsmLayerPair  *layers;
    size_t         n_layers, cap_layers;

    AsmLayerHit   *layer_hits;
    size_t         n_layer_hits, cap_layer_hits;

    double        *chart_layer_d;
    double         layer_d_global;
    double         cube_size;

    uint8_t       *repair_resume;
    size_t         repair_resume_size;
} AsmRun;

size_t AsmRun_push_chart(AsmRun *run, const AsmChart *c);
size_t AsmRun_push_rel(AsmRun *run, const AsmRelation *r);
size_t AsmRun_push_corr(AsmRun *run, const AsmCorr *c);
size_t AsmRun_push_layer(AsmRun *run, const AsmLayerPair *l);
size_t AsmRun_push_layer_hit(AsmRun *run, const AsmLayerHit *h);

static inline int AsmChart_in_layout(const AsmChart *c)
{
    return (c->flags & ASM_CHART_EXCLUDED_MASK) == 0u && c->uv != NULL;
}

static inline int AsmChart_registered(const AsmChart *c)
{
    return AsmChart_in_layout(c) && c->placed && c->placement_state != ASM_PLACE_NONE;
}

static inline void AsmChart_point(const AsmChart *c, size_t v, double p[2])
{
    if (c->placed_uv) { p[0] = c->placed_uv[2*v]; p[1] = c->placed_uv[2*v+1]; return; }
    double u = c->uv[2*v], w = c->uv[2*v+1];
    if (c->flags & ASM_CHART_MIRROR) u = -u;
    double ct = cos(c->pose_theta), sn = sin(c->pose_theta);
    p[0] = ct*u-sn*w+c->pose_x; p[1] = sn*u+ct*w+c->pose_y;
}

static inline int AsmRel_is_join(const AsmRelation *r)
{
    return (r->flags & (ASM_REL_DROPPED | ASM_REL_CONTACT | ASM_REL_SWITCHED | ASM_REL_PLACEMENT_ONLY)) == 0u;
}

#endif
