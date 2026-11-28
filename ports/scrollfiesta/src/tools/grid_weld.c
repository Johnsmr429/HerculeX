#include "../common/ves_platform.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../common/arena.h"
#include "../common/except.h"
#include "../common/mesh_bin.h"
#include "../common/obj_io.h"
#include "../common/cc_color.h"
#include "../common/mesh_types.h"
#include "../common/mesh_manifold.h"
#include "../common/pipeline_constants.h"
#include "../remesh/seam_weld.h"
#include "../remesh/seam_planes.h"
#include "../remesh/seam_refine.h"
#include "../remesh/seam_band_cvt.h"
#include "../remesh/sheet_reweld.h"
#include "../remesh/seam_hole_fill.h"
#include "../remesh/fold_cleanup.h"
#include "../remesh/pinhole_fill.h"
#include "../remesh/manifold_guard.h"
#include "../remesh/orient_mesh.h"
#include "../remesh/orient_weld.h"
#include "../remesh/weld_cleanup.h"
#include "../remesh/intersection_cleanup.h"
#include "../remesh/boundary_arc_weld.h"
#include "../remesh/chart_lineage.h"
#include "../remesh/chart_bridge_forest.h"
#include "../remesh/chart_zipper.h"
#include "../remesh/stream_weld_format.h"
#include "../common/pitch_table.h"
#include "../holefill/hole_fill.h"
#include "../common/vert_weld.h"
#include "../flatten/developability.h"
#include "../flatten/disk_topology_repair.h"
#include "../flatten/topology_invariants.h"

#define CUBE_SIZE_VOX 128.0f
#define SEAM_AXIS_EPS 0.6f

static int gw_parse_placed_calibration(const char *json,
                                       double *pitch,
                                       double *umb_y, double *umb_x);

static int near_cube_boundary(const float *v, float zone)
{
    for (int k = 0; k < 3; k++) {
        float m = v[k] - CUBE_SIZE_VOX * floorf(v[k] / CUBE_SIZE_VOX);
        if (m < zone || m > CUBE_SIZE_VOX - zone) return 1;
    }
    return 0;
}

static const float CUBE_PALETTE[16][3] = {
    { 0.894f, 0.102f, 0.110f },
    { 0.215f, 0.494f, 0.722f },
    { 1.000f, 0.498f, 0.000f },
    { 0.596f, 0.306f, 0.639f },
    { 0.651f, 0.337f, 0.157f },
    { 0.969f, 0.506f, 0.749f },
    { 0.301f, 0.686f, 0.290f },
    { 0.498f, 0.498f, 0.498f },
    { 0.984f, 0.890f, 0.435f },
    { 0.400f, 0.761f, 0.647f },
    { 0.988f, 0.553f, 0.384f },
    { 0.553f, 0.627f, 0.796f },
    { 0.906f, 0.541f, 0.765f },
    { 0.706f, 0.871f, 0.412f },
    { 0.553f, 0.310f, 0.310f },
    { 0.165f, 0.631f, 0.596f }
};
static const float SEAM_COLOR[3] = { 0.000f, 1.000f, 0.000f };
static const float PINHOLE_COLOR[3] = { 0.100f, 0.450f, 1.000f };

static int parse_cube_origin(const char *cube_id,
                             int64_t *vz, int64_t *vy, int64_t *vx)
{
    int iz = 0, iy = 0, ix = 0;
    if (sscanf(cube_id, "z%d_y%d_x%d", &iz, &iy, &ix) != 3) return -1;
    *vz = iz;
    *vy = iy;
    *vx = ix;
    return 0;
}

typedef struct {
    double *z, *y, *x;
    size_t n;
} GwAxisTable;

static void gw_axis_table_free(GwAxisTable *t)
{
    free(t->z); free(t->y); free(t->x);
    memset(t, 0, sizeof(*t));
}

static int gw_axis_parse_line(const char *line,
                              double *z, double *y, double *x)
{
    char *end = NULL;
    const char *p = line;
    *z = strtod(p, &end);
    if (end == p) return 0;
    p = end; while (*p == ' ' || *p == '\t') p++;
    if (*p++ != ',') return 0;
    *y = strtod(p, &end);
    if (end == p) return 0;
    p = end; while (*p == ' ' || *p == '\t') p++;
    if (*p++ != ',') return 0;
    *x = strtod(p, &end);
    return end != p && isfinite(*z) && isfinite(*y) && isfinite(*x);
}

static int gw_axis_table_load(const char *path, GwAxisTable *t)
{
    FILE *f = NULL;
    size_t cap = 32;
    char line[1024];
    memset(t, 0, sizeof(*t));
    f = fopen(path, "r");
    if (f == NULL) return -1;
    t->z = (double *)malloc(cap * sizeof(double));
    t->y = (double *)malloc(cap * sizeof(double));
    t->x = (double *)malloc(cap * sizeof(double));
    if (t->z == NULL || t->y == NULL || t->x == NULL) goto fail;
    while (fgets(line, sizeof(line), f) != NULL) {
        double z, y, x;
        if (!gw_axis_parse_line(line, &z, &y, &x)) continue;
        if (t->n == cap) {
            double *p;
            cap *= 2;
            p = (double *)realloc(t->z, cap * sizeof(double));
            if (p == NULL) goto fail; t->z = p;
            p = (double *)realloc(t->y, cap * sizeof(double));
            if (p == NULL) goto fail; t->y = p;
            p = (double *)realloc(t->x, cap * sizeof(double));
            if (p == NULL) goto fail; t->x = p;
        }
        t->z[t->n] = z; t->y[t->n] = y; t->x[t->n] = x; t->n++;
    }
    fclose(f); f = NULL;
    if (t->n < 2) goto fail;
    for (size_t i = 1; i < t->n; i++) {
        double z = t->z[i], y = t->y[i], x = t->x[i];
        size_t j = i;
        while (j > 0 && t->z[j-1] > z) {
            t->z[j] = t->z[j-1]; t->y[j] = t->y[j-1];
            t->x[j] = t->x[j-1]; j--;
        }
        t->z[j] = z; t->y[j] = y; t->x[j] = x;
    }
    for (size_t i = 1; i < t->n; i++)
        if (!(t->z[i] > t->z[i-1])) goto fail;
    return 0;
fail:
    if (f != NULL) fclose(f);
    gw_axis_table_free(t);
    return -1;
}

static void gw_axis_table_eval(const GwAxisTable *t, double z,
                               double *out_y, double *out_x)
{
    BpaBridgeGate g;
    memset(&g, 0, sizeof(g));
    g.axis_z = t->z; g.axis_y = t->y; g.axis_x = t->x; g.axis_n = t->n;
    BpaBridgeGate_axis_at(&g, z, out_y, out_x);
}

static int gw_file_exists(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static int gw_placed_pair_presence(int has_obj, int has_facekeep)
{
    if (has_obj && has_facekeep) return 1;
    if (!has_obj && !has_facekeep) return 0;
    return -1;
}

static int gw_read_facekeep(Arena_T arena, const char *path,
                            size_t expected_nf, uint8_t **out_keep,
                            size_t *out_kept)
{
    FILE *f = NULL;
    long bytes = 0;
    uint8_t *keep = NULL;
    size_t kept = 0;
    *out_keep = NULL;
    *out_kept = 0;
    f = fopen(path, "rb");
    if (f == NULL || fseek(f, 0, SEEK_END) != 0 ||
        (bytes = ftell(f)) < 0 || (size_t)bytes != expected_nf ||
        fseek(f, 0, SEEK_SET) != 0)
        goto fail;
    keep = (uint8_t *)ARENA_ALLOC(arena, (expected_nf ? expected_nf : 1));
    if (expected_nf > 0 && fread(keep, 1, expected_nf, f) != expected_nf)
        goto fail;
    if (fgetc(f) != EOF) goto fail;
    fclose(f);
    for (size_t i = 0; i < expected_nf; i++) {
        if (keep[i] > 1) return -1;
        kept += keep[i] != 0;
    }
    *out_keep = keep;
    *out_kept = kept;
    return 0;
fail:
    if (f != NULL) fclose(f);
    return -1;
}

typedef struct {
    char **ids;
    size_t n;
    size_t cap;
} CubeList;

static void cubelist_push(Arena_T arena, CubeList *cl, const char *id)
{
    if (cl->n >= cl->cap) {
        size_t new_cap = cl->cap == 0 ? 16 : cl->cap * 2;
        char **new_ids = (char **)ARENA_ALLOC(arena,
                            (new_cap * sizeof(char *)));
        if (cl->ids) {
            memcpy(new_ids, cl->ids, cl->n * sizeof(char *));
        }
        cl->ids = new_ids;
        cl->cap = new_cap;
    }
    size_t len = strlen(id);
    char *copy = (char *)ARENA_ALLOC(arena, (len + 1));
    memcpy(copy, id, len + 1);
    cl->ids[cl->n++] = copy;
}

#ifdef _MSC_VER
#include <windows.h>

static int enumerate_cube_dirs(Arena_T arena, const char *base_dir,
                               CubeList *out)
{
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s/*", base_dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (strcmp(fd.cFileName, ".") == 0) continue;
        if (strcmp(fd.cFileName, "..") == 0) continue;
        if (fd.cFileName[0] != 'z') continue;
        cubelist_push(arena, out, fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return 0;
}
#else
#include <dirent.h>

static int enumerate_cube_dirs(Arena_T arena, const char *base_dir,
                               CubeList *out)
{
    DIR *d = opendir(base_dir);
    if (!d) return -1;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] != 'z') continue;
        if (strchr(ent->d_name, '_') == NULL) continue;
        cubelist_push(arena, out, ent->d_name);
    }
    closedir(d);
    return 0;
}
#endif

static void cubelist_filter_bbox(CubeList *cl,
                                 int64_t z0, int64_t z1, int64_t y0, int64_t y1,
                                 int64_t x0, int64_t x1)
{
    size_t w = 0;
    for (size_t i = 0; i < cl->n; i++) {
        int64_t vz = 0, vy = 0, vx = 0;
        if (parse_cube_origin(cl->ids[i], &vz, &vy, &vx) != 0) continue;
        if (vz >= z0 && vz <= z1 && vy >= y0 && vy <= y1 &&
            vx >= x0 && vx <= x1) {
            cl->ids[w++] = cl->ids[i];
        }
    }
    cl->n = w;
}

static int topology_safe_micro_weld(Arena_T arena,
                                    const float *verts, size_t nv,
                                    int32_t *faces, size_t nf,
                                    const ChartLineage *lineage,
                                    float **out_verts, size_t *out_nv,
                                    size_t *out_nf,
                                    size_t *components_before,
                                    size_t *components_after)
{
    int32_t *group = NULL;
    uint8_t *chart_is_disk = NULL;
    int32_t *candidate_faces = NULL;
    size_t before = 0, after = 0;
    BoundaryArcWeldParams p;
    BoundaryArcWeldStats s;
    BoundaryArcWeldPolicy policy;
    ChartLineageStats lineage_stats;
    TopologyAuditOptions audit_options;
    TopologyAuditReport before_topology, after_topology;
    float *candidate_verts = NULL;
    size_t candidate_nv = 0, candidate_nf = 0;
    int reject = 0, rc = -1;

    memset(&before_topology, 0, sizeof before_topology);
    memset(&after_topology, 0, sizeof after_topology);
    TopologyAudit_options_default(&audit_options);
    if (TopologyAudit_analyze(verts, nv, faces, nf, &audit_options,
                              &before_topology) != 0)
        goto cleanup;
    before = before_topology.face_components;
    memset(&policy, 0, sizeof policy);
    memset(&lineage_stats, 0, sizeof lineage_stats);
    if (lineage != NULL) {
        if (ChartLineage_assign(arena, lineage, verts, nv, faces, nf,
                                &group, &lineage_stats) != 0)
            goto cleanup;
        chart_is_disk = (uint8_t *)ARENA_CALLOC(
            arena, before_topology.face_components, 1L);
        for (size_t c = 0; c < before_topology.face_components; c++)
            chart_is_disk[c] = before_topology.component[c].homeomorphic_to_disk
                             ? 1u : 0u;
        policy.chart_component = before_topology.vertex_component;
        policy.chart_is_disk = chart_is_disk;
        policy.n_chart_components = before_topology.face_components;
        policy.cross_chart_forest = 1;
        policy.require_edge_zipper = 1;
    } else if (SheetReweld_label(arena, faces, nf, nv, &group, &before) != 0) {
        goto cleanup;
    }
    candidate_faces = (int32_t *)malloc(nf*3*sizeof(int32_t));
    if (candidate_faces == NULL) goto cleanup;
    memcpy(candidate_faces, faces, nf*3*sizeof(int32_t));
    BoundaryArcWeld_default_params(&p);
    { const char *e;
      if ((e=getenv("SEAM_MICROWELD_NORMAL_MIN"))) {
          double v=atof(e); if (v >= -1.0 && v <= 1.0) p.normal_dot_min=v; }
      if ((e=getenv("SEAM_MICROWELD_TANGENT_MAX"))) {
          double v=atof(e); if (v >= -1.0 && v <= 1.0) p.tangent_dot_max=v; }
      if ((e=getenv("SEAM_MICROWELD_ARC_GAP"))) {
          double v=atof(e); if (v > 0.0) p.max_anchor_gap=v; }
      if ((e=getenv("SEAM_MICROWELD_ARC_SUPPORT"))) {
          double v=atof(e); if (v > 0.0) p.min_support_length=v; }
      if ((e=getenv("SEAM_MICROWELD_ARC_ANCHORS"))) {
          long v=atol(e); if (v >= 2 && v <= 64) p.min_anchors=(size_t)v; }
      if ((e=getenv("SEAM_MICROWELD_AMBIG_MARGIN"))) {
          double v=atof(e); if (v >= 0.0) p.ambiguity_margin=v; } }
    if (BoundaryArcWeld_process_with_policy(
            arena, verts, nv, candidate_faces, nf, 1e-3f,
            group, &p, lineage != NULL ? &policy : NULL,
            &candidate_verts, &candidate_nv, &candidate_nf, &s) != 0)
        goto cleanup;
    fprintf(stderr,
            "  boundary-arc weld: %zu/%zu reciprocal pair(s) accepted "
            "(%zu supported cluster(s), %zu local-link; %zu ambiguous; "
            "%zu arc links, %.1f/%.1f vox gap/support)\n",
            s.accepted_pairs, s.reciprocal_pairs, s.accepted_clusters,
            s.local_link_pairs, s.ambiguous_vertices, s.arc_links,
            p.max_anchor_gap, p.min_support_length);
    if (lineage != NULL) {
        fprintf(stderr,
                "  chart transactions: lineage %zu/%zu current chart(s) "
                "(%zu ambiguous), accepted %zu/%zu complete zipper(s); "
                "reject[self=%zu nondisk=%zu nonpath=%zu sparse=%zu cycle=%zu]\n",
                lineage_stats.components_with_lineage,
                lineage_stats.current_components,
                lineage_stats.ambiguous_components,
                s.chart_clusters_accepted, s.chart_clusters_considered,
                s.chart_clusters_rejected_self,
                s.chart_clusters_rejected_nondisk,
                s.chart_clusters_rejected_nonpath,
                s.chart_clusters_rejected_sparse,
                s.chart_clusters_rejected_cycle);
    }
    if (TopologyAudit_analyze(candidate_verts, candidate_nv,
                              candidate_faces, candidate_nf, &audit_options,
                              &after_topology) != 0)
        goto cleanup;
    reject =
        after_topology.face_components > before_topology.face_components ||
        after_topology.nondisk_components > before_topology.nondisk_components ||
        after_topology.invalid_surface_components >
            before_topology.invalid_surface_components ||
        after_topology.nonorientable_components >
            before_topology.nonorientable_components ||
        after_topology.beta_1 > before_topology.beta_1 ||
        after_topology.beta_2 > before_topology.beta_2;
    fprintf(stderr,
            "  exact micro-weld topology: cc %zu->%zu, nondisk %zu->%zu, "
            "invalid %zu->%zu, beta1 %lld->%lld: %s\n",
            before_topology.face_components, after_topology.face_components,
            before_topology.nondisk_components,
            after_topology.nondisk_components,
            before_topology.invalid_surface_components,
            after_topology.invalid_surface_components,
            (long long)before_topology.beta_1,
            (long long)after_topology.beta_1,
            reject ? "REJECT (kept input)" : "accept");
    if (reject) {
        *out_verts = (float *)verts;
        *out_nv = nv;
        *out_nf = nf;
        after = before_topology.face_components;
    } else {
        memcpy(faces, candidate_faces, candidate_nf*3*sizeof(int32_t));
        *out_verts = candidate_verts;
        *out_nv = candidate_nv;
        *out_nf = candidate_nf;
        after = after_topology.face_components;
    }
    if (components_before) *components_before = before;
    if (components_after) *components_after = after;
    rc = 0;

cleanup:
    TopologyAudit_dispose(&after_topology);
    TopologyAudit_dispose(&before_topology);
    free(candidate_faces);
    return rc;
}

static int source_chart_smaller_support_is_unambiguous(
    size_t small_faces,double small_area,double small_diag,
    size_t large_faces,double large_area,double large_diag)
{
    (void)small_faces;
    (void)large_faces;
    if(!(small_area>0.0&&large_area>small_area&&large_diag>small_diag))
        return 0;
    return small_diag*4.0<=large_diag&&small_area*16.0<=large_area;
}

typedef struct {
    size_t a, b;
    size_t face_pairs;
    size_t face_a, face_b;
    size_t overlap_pairs, stab_pairs, fold_pairs;
} SourceChartConflictEdge;

typedef struct {
    const int32_t *faces;
    const TopologyAuditReport *topology;
    SourceChartConflictEdge *edge;
    size_t n, cap;
} SourceChartConflictCollector;

typedef struct {
    size_t loser, winner;
    size_t face_pairs;
} SourceChartOwnershipEdge;

typedef struct {
    size_t component;
    size_t faces;
    double area, diag;
} SourceChartOrder;

typedef struct {
    int32_t lo, hi;
    size_t face;
    uint8_t slot;
} SourceChartPeelHalfEdge;

static int source_chart_conflict_edge_cmp(const void *va, const void *vb)
{
    const SourceChartConflictEdge *a=(const SourceChartConflictEdge*)va;
    const SourceChartConflictEdge *b=(const SourceChartConflictEdge*)vb;
    if(a->a<b->a)return -1; if(a->a>b->a)return 1;
    if(a->b<b->b)return -1; if(a->b>b->b)return 1;
    return 0;
}

static int source_chart_order_cmp(const void *va, const void *vb)
{
    const SourceChartOrder *a=(const SourceChartOrder*)va;
    const SourceChartOrder *b=(const SourceChartOrder*)vb;
    if(a->area>b->area)return -1; if(a->area<b->area)return 1;
    if(a->diag>b->diag)return -1; if(a->diag<b->diag)return 1;
    if(a->faces>b->faces)return -1; if(a->faces<b->faces)return 1;
    if(a->component<b->component)return -1;
    if(a->component>b->component)return 1;
    return 0;
}

static int source_chart_peel_halfedge_cmp(const void *va,const void *vb)
{
    const SourceChartPeelHalfEdge *a=(const SourceChartPeelHalfEdge*)va;
    const SourceChartPeelHalfEdge *b=(const SourceChartPeelHalfEdge*)vb;
    if(a->lo<b->lo)return -1;if(a->lo>b->lo)return 1;
    if(a->hi<b->hi)return -1;if(a->hi>b->hi)return 1;
    if(a->face<b->face)return -1;if(a->face>b->face)return 1;
    return (int)a->slot-(int)b->slot;
}

static int source_chart_collect_conflict(size_t face_a,size_t face_b,
                                         int hit_kind,void *context)
{
    SourceChartConflictCollector *cc=(SourceChartConflictCollector*)context;
    int32_t ca,cb;
    (void)hit_kind;
    ca=cc->topology->vertex_component[cc->faces[face_a*3]];
    cb=cc->topology->vertex_component[cc->faces[face_b*3]];
    if(ca<0||cb<0)return -1;
    if(cc->n==cc->cap){
        size_t next=cc->cap?cc->cap*2:256;
        SourceChartConflictEdge *grown;
        if(next<cc->cap||next>SIZE_MAX/sizeof(*grown))return -1;
        grown=(SourceChartConflictEdge*)realloc(cc->edge,next*sizeof(*grown));
        if(!grown)return -1;
        cc->edge=grown;cc->cap=next;
    }
    cc->edge[cc->n].a=(size_t)ca<(size_t)cb?(size_t)ca:(size_t)cb;
    cc->edge[cc->n].b=(size_t)ca<(size_t)cb?(size_t)cb:(size_t)ca;
    cc->edge[cc->n].face_pairs=1;
    cc->edge[cc->n].face_a=face_a;
    cc->edge[cc->n].face_b=face_b;
    cc->edge[cc->n].overlap_pairs=
        hit_kind==INTERSECTION_HIT_OVERLAP?1:0;
    cc->edge[cc->n].stab_pairs=hit_kind==INTERSECTION_HIT_STAB?1:0;
    cc->edge[cc->n].fold_pairs=hit_kind==INTERSECTION_HIT_FOLD?1:0;
    cc->n++;
    return 0;
}

static void source_chart_dedupe_conflicts(SourceChartConflictCollector *cc)
{
    size_t w=0;
    if(cc->n==0)return;
    qsort(cc->edge,cc->n,sizeof(*cc->edge),source_chart_conflict_edge_cmp);
    for(size_t i=0;i<cc->n;i++){
        if(w>0&&cc->edge[w-1].a==cc->edge[i].a&&
           cc->edge[w-1].b==cc->edge[i].b){
            cc->edge[w-1].face_pairs+=cc->edge[i].face_pairs;
            cc->edge[w-1].overlap_pairs+=cc->edge[i].overlap_pairs;
            cc->edge[w-1].stab_pairs+=cc->edge[i].stab_pairs;
            cc->edge[w-1].fold_pairs+=cc->edge[i].fold_pairs;
        }else{
            cc->edge[w++]=cc->edge[i];
        }
    }
    cc->n=w;
}

static double source_chart_face_area(const float *verts,const int32_t *face)
{
    double ab[3],ac[3],cr[3];
    for(int d=0;d<3;d++){
        ab[d]=(double)verts[(size_t)face[1]*3+(size_t)d]-
              (double)verts[(size_t)face[0]*3+(size_t)d];
        ac[d]=(double)verts[(size_t)face[2]*3+(size_t)d]-
              (double)verts[(size_t)face[0]*3+(size_t)d];
    }
    cr[0]=ab[1]*ac[2]-ab[2]*ac[1];
    cr[1]=ab[2]*ac[0]-ab[0]*ac[2];
    cr[2]=ab[0]*ac[1]-ab[1]*ac[0];
    return 0.5*sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
}

static uint32_t source_chart_peel_next_stamp(uint32_t *visited,size_t nf,
                                             uint32_t *stamp)
{
    (*stamp)++;
    if(*stamp==0){memset(visited,0,nf*sizeof(*visited));*stamp=1;}
    return *stamp;
}

static int source_chart_boundary_cap_plan(
    const float *verts,const int32_t *faces,size_t nf,int32_t comp,
    size_t candidate,const int32_t *component_head,
    const int32_t *component_next,const size_t *component_remaining,
    const int32_t *neighbor,const uint8_t *face_remove,size_t *queue,
    uint32_t *visited,uint32_t *stamp,size_t *out_keep_seed,
    size_t *out_keep_faces,size_t *out_remove_faces,double *out_remove_area)
{
    uint32_t mark;
    size_t remaining,seen_total=0,best_count=0,best_seed=SIZE_MAX;
    size_t best_min=SIZE_MAX;
    double total_area=0.0,best_area=-1.0;
    int32_t seed;
    if(!verts||!faces||!component_head||!component_next||
       !component_remaining||!neighbor||!face_remove||!queue||!visited||
       !stamp||comp<0||candidate>=nf)return -1;
    remaining=component_remaining[(size_t)comp];
    if(remaining<=1)return 0;
    mark=source_chart_peel_next_stamp(visited,nf,stamp);
    for(seed=component_head[(size_t)comp];seed>=0;
        seed=component_next[(size_t)seed]){
        size_t qh=0,qt=0,count=0,min_face=(size_t)seed;
        double area=0.0;
        if((size_t)seed==candidate||face_remove[(size_t)seed]||
           visited[(size_t)seed]==mark)continue;
        queue[qt++]=(size_t)seed;visited[(size_t)seed]=mark;
        while(qh<qt){
            size_t cur=queue[qh++];
            count++;area+=source_chart_face_area(verts,&faces[cur*3]);
            if(cur<min_face)min_face=cur;
            for(int k=0;k<3;k++){
                int32_t nb=neighbor[cur*3+(size_t)k];
                if(nb<0||(size_t)nb==candidate||face_remove[(size_t)nb]||
                   visited[(size_t)nb]==mark)continue;
                visited[(size_t)nb]=mark;queue[qt++]=(size_t)nb;
            }
        }
        seen_total+=count;total_area+=area;
        if(best_seed==SIZE_MAX||area>best_area||
           (area==best_area&&count>best_count)||
           (area==best_area&&count==best_count&&min_face<best_min)){
            best_seed=(size_t)seed;best_count=count;best_area=area;
            best_min=min_face;
        }
    }
    if(seen_total+1!=remaining||best_seed==SIZE_MAX||best_count==0)
        return -1;
    if(out_keep_seed)*out_keep_seed=best_seed;
    if(out_keep_faces)*out_keep_faces=best_count;
    if(out_remove_faces)*out_remove_faces=remaining-best_count;
    if(out_remove_area){
        double removed=total_area+
            source_chart_face_area(verts,&faces[candidate*3])-best_area;
        *out_remove_area=removed>0.0?removed:0.0;
    }
    return 1;
}

static int source_chart_boundary_peel(
    const float *verts,size_t nv,const int32_t *faces,size_t nf,
    const TopologyAuditReport *topology,const double *component_area,
    const SourceChartConflictEdge *pair,size_t npair,
    uint8_t *face_remove,int verbose,
    size_t *out_peeled,size_t *out_touched)
{
    SourceChartPeelHalfEdge *he=NULL;
    size_t *face_edge=NULL,*edge_count=NULL,*degree=NULL;
    int32_t *neighbor=NULL,*component_head=NULL,*component_next=NULL;
    size_t *component_remaining=NULL,*queue=NULL;
    uint32_t *visited=NULL,stamp=0;
    uint8_t *active=NULL,*touched=NULL;
    size_t nhe=0,nedge=0,unresolved=0,peeled=0,ntouched=0;
    size_t nc=topology->face_components;
    int rc=-1;
    if(out_peeled)*out_peeled=0;if(out_touched)*out_touched=0;
    if(!verts||!faces||!topology||!component_area||!face_remove)return -1;
    if(nf>(size_t)INT32_MAX||nc>(size_t)INT32_MAX||nf>SIZE_MAX/3||
       nf*3>SIZE_MAX/sizeof(*he)||nf*3>SIZE_MAX/sizeof(*face_edge)||
       nf*3>SIZE_MAX/sizeof(*neighbor))return -1;
    he=(SourceChartPeelHalfEdge*)malloc((nf?nf*3:1)*sizeof(*he));
    face_edge=(size_t*)malloc((nf?nf*3:1)*sizeof(*face_edge));
    edge_count=(size_t*)calloc(nf?nf*3:1,sizeof(*edge_count));
    degree=(size_t*)calloc(nf?nf:1,sizeof(*degree));
    neighbor=(int32_t*)malloc((nf?nf*3:1)*sizeof(*neighbor));
    component_head=(int32_t*)malloc((nc?nc:1)*sizeof(*component_head));
    component_next=(int32_t*)malloc((nf?nf:1)*sizeof(*component_next));
    component_remaining=(size_t*)calloc(nc?nc:1,sizeof(*component_remaining));
    queue=(size_t*)malloc((nf?nf:1)*sizeof(*queue));
    visited=(uint32_t*)calloc(nf?nf:1,sizeof(*visited));
    active=(uint8_t*)calloc(npair?npair:1,1);
    touched=(uint8_t*)calloc(nc?nc:1,1);
    if(!he||!face_edge||!edge_count||!degree||!neighbor||!component_head||
       !component_next||!component_remaining||!queue||!visited||!active||
       !touched)goto done;
    for(size_t i=0;i<nf*3;i++){face_edge[i]=SIZE_MAX;neighbor[i]=-1;}
    for(size_t c=0;c<nc;c++)component_head[c]=-1;
    for(size_t f=0;f<nf;f++){
        int32_t c=topology->vertex_component[faces[f*3]];
        component_next[f]=-1;
        if(face_remove[f]||c<0||(size_t)c>=nc)continue;
        for(int k=0;k<3;k++)
            if(faces[f*3+(size_t)k]<0||
               (size_t)faces[f*3+(size_t)k]>=nv)
                goto done;
        component_next[f]=component_head[(size_t)c];
        component_head[(size_t)c]=(int32_t)f;
        component_remaining[(size_t)c]++;
        for(int k=0;k<3;k++){
            int32_t a=faces[f*3+(size_t)k];
            int32_t b=faces[f*3+(size_t)((k+1)%3)];
            he[nhe].lo=a<b?a:b;he[nhe].hi=a<b?b:a;
            he[nhe].face=f;he[nhe].slot=(uint8_t)k;nhe++;
        }
    }
    qsort(he,nhe,sizeof(*he),source_chart_peel_halfedge_cmp);
    for(size_t i=0;i<nhe;){
        size_t j=i+1;
        while(j<nhe&&he[j].lo==he[i].lo&&he[j].hi==he[i].hi)j++;
        edge_count[nedge]=j-i;
        for(size_t q=i;q<j;q++)
            face_edge[he[q].face*3+(size_t)he[q].slot]=nedge;
        if(j-i==2){
            neighbor[he[i].face*3+(size_t)he[i].slot]=(int32_t)he[i+1].face;
            neighbor[he[i+1].face*3+(size_t)he[i+1].slot]=(int32_t)he[i].face;
        }
        nedge++;i=j;
    }
    for(size_t i=0;i<npair;i++){
        size_t a=pair[i].face_a,b=pair[i].face_b;
        int32_t ca,cb;
        if(a>=nf||b>=nf)goto done;
        if(face_remove[a]||face_remove[b])continue;
        ca=topology->vertex_component[faces[a*3]];
        cb=topology->vertex_component[faces[b*3]];
        if(ca<0||cb<0){
            if(verbose)fprintf(stderr,
                    "ERROR: source chart boundary peel cannot resolve "
                    "invalid conflict faces %zu/%zu (charts=%d/%d)\n",
                    a,b,ca,cb);
            goto done;
        }
        active[i]=1;degree[a]++;degree[b]++;unresolved++;
    }
    while(unresolved>0){
        size_t best=SIZE_MAX,best_degree=0,best_boundary=0;
        size_t cap_best=SIZE_MAX,cap_keep_seed=SIZE_MAX;
        size_t cap_keep_faces=0,cap_remove_faces=0,cap_degree=0;
        double best_relative=HUGE_VAL,best_support=HUGE_VAL;
        double cap_score=HUGE_VAL,cap_fraction=HUGE_VAL;
        for(size_t i=0;i<npair;i++)if(active[i]){
            size_t candidate[2]={pair[i].face_a,pair[i].face_b};
            for(int side=0;side<2;side++){
                size_t f=candidate[side],boundary=0,seen=0;
                int32_t comp;
                double relative,support;
                if(f>=nf||face_remove[f]||degree[f]==0)continue;
                comp=topology->vertex_component[faces[f*3]];
                if(comp<0||(size_t)comp>=nc||component_remaining[(size_t)comp]<=1)
                    continue;
                for(int k=0;k<3;k++){
                    size_t e=face_edge[f*3+(size_t)k];
                    if(e!=SIZE_MAX&&edge_count[e]==1)boundary++;
                }
                {
                    int32_t start=component_head[(size_t)comp];
                    while(start>=0&&((size_t)start==f||face_remove[(size_t)start]))
                        start=component_next[(size_t)start];
                    if(start<0)continue;
                    source_chart_peel_next_stamp(visited,nf,&stamp);
                    size_t qh=0,qt=0;queue[qt++]=(size_t)start;
                    visited[(size_t)start]=stamp;
                    while(qh<qt){
                        size_t cur=queue[qh++];seen++;
                        for(int k=0;k<3;k++){
                            int32_t nb=neighbor[cur*3+(size_t)k];
                            if(nb<0||(size_t)nb==f||face_remove[(size_t)nb]||
                               visited[(size_t)nb]==stamp)continue;
                            visited[(size_t)nb]=stamp;queue[qt++]=(size_t)nb;
                        }
                    }
                }
                if(seen+1!=component_remaining[(size_t)comp]){
                    size_t keep_seed=SIZE_MAX,keep_faces=0,remove_faces=0;
                    double remove_area=0.0,fraction,score;
                    int plan=source_chart_boundary_cap_plan(
                        verts,faces,nf,comp,f,component_head,component_next,
                        component_remaining,neighbor,face_remove,queue,visited,
                        &stamp,&keep_seed,&keep_faces,&remove_faces,
                        &remove_area);
                    if(plan<0)goto done;
                    if(plan==0||remove_faces==0||keep_faces==0)continue;
                    support=component_area[(size_t)comp];
                    fraction=support>0.0?remove_area/support:
                        (double)remove_faces/
                        (double)component_remaining[(size_t)comp];
                    score=fraction/(double)degree[f];
                    if(cap_best==SIZE_MAX||score<cap_score||
                       (score==cap_score&&degree[f]>cap_degree)||
                       (score==cap_score&&degree[f]==cap_degree&&
                        remove_faces<cap_remove_faces)||
                       (score==cap_score&&degree[f]==cap_degree&&
                        remove_faces==cap_remove_faces&&f<cap_best)){
                        cap_best=f;cap_keep_seed=keep_seed;
                        cap_keep_faces=keep_faces;
                        cap_remove_faces=remove_faces;cap_degree=degree[f];
                        cap_score=score;cap_fraction=fraction;
                    }
                    continue;
                }
                support=component_area[(size_t)comp];
                relative=support>0.0?
                    source_chart_face_area(verts,&faces[f*3])*
                    (double)topology->component[(size_t)comp].faces/support:
                    HUGE_VAL;
                if(best==SIZE_MAX||
                   ((boundary>0)!=(best_boundary>0)&&boundary>0)||
                   ((boundary>0)==(best_boundary>0)&&
                    (degree[f]>best_degree||
                     (degree[f]==best_degree&&boundary>best_boundary)||
                     (degree[f]==best_degree&&boundary==best_boundary&&
                      relative<best_relative)||
                     (degree[f]==best_degree&&boundary==best_boundary&&
                      relative==best_relative&&support<best_support)||
                     (degree[f]==best_degree&&boundary==best_boundary&&
                      relative==best_relative&&support==best_support&&f<best)))){
                    best=f;best_degree=degree[f];best_boundary=boundary;
                    best_relative=relative;best_support=support;
                }
            }
        }
        if(best==SIZE_MAX&&cap_best!=SIZE_MAX){
            int32_t comp=topology->vertex_component[faces[cap_best*3]];
            uint32_t keep_mark;
            size_t qh=0,qt=0,removed_now=0,covered=0;
            if(comp<0||(size_t)comp>=nc||cap_keep_seed>=nf||
               face_remove[cap_keep_seed])goto done;
            keep_mark=source_chart_peel_next_stamp(visited,nf,&stamp);
            queue[qt++]=cap_keep_seed;visited[cap_keep_seed]=keep_mark;
            while(qh<qt){
                size_t cur=queue[qh++];
                for(int k=0;k<3;k++){
                    int32_t nb=neighbor[cur*3+(size_t)k];
                    if(nb<0||(size_t)nb==cap_best||face_remove[(size_t)nb]||
                       visited[(size_t)nb]==keep_mark)continue;
                    visited[(size_t)nb]=keep_mark;queue[qt++]=(size_t)nb;
                }
            }
            for(int32_t cur=component_head[(size_t)comp];cur>=0;
                cur=component_next[(size_t)cur]){
                size_t f=(size_t)cur;
                if(face_remove[f]||visited[f]==keep_mark)continue;
                face_remove[f]=1;removed_now++;peeled++;
                if(component_remaining[(size_t)comp]==0)goto done;
                component_remaining[(size_t)comp]--;
                for(int k=0;k<3;k++){
                    size_t e=face_edge[f*3+(size_t)k];
                    if(e!=SIZE_MAX&&edge_count[e]>0)edge_count[e]--;
                }
            }
            if(removed_now!=cap_remove_faces||
               component_remaining[(size_t)comp]!=cap_keep_faces)goto done;
            if(!touched[(size_t)comp]){touched[(size_t)comp]=1;ntouched++;}
            for(size_t i=0;i<npair;i++)if(active[i]&&
                (face_remove[pair[i].face_a]||face_remove[pair[i].face_b])){
                size_t a=pair[i].face_a,b=pair[i].face_b;
                active[i]=0;unresolved--;covered++;
                if(a<nf&&degree[a]>0)degree[a]--;
                if(b<nf&&degree[b]>0)degree[b]--;
            }
            if(covered==0)goto done;
            if(verbose)fprintf(stderr,
                "  source chart boundary-cap peel: chart %d removed %zu "
                "face(s) at neck %zu, retained %zu-face disk; resolved %zu "
                "exact pair(s), support fraction %.6g\n",
                comp,removed_now,cap_best,cap_keep_faces,covered,cap_fraction);
            continue;
        }
        if(best==SIZE_MAX){
            if(verbose){
                size_t shown=0;
                fprintf(stderr,
                        "ERROR: %zu exact source conflict(s) cannot be removed "
                        "without splitting a retained chart atom\n",
                        unresolved);
                for(size_t i=0;i<npair&&shown<32;i++)if(active[i]){
                    size_t fa=pair[i].face_a,fb=pair[i].face_b;
                    size_t candidate[2]={fa,fb},boundary[2]={0,0};
                    int32_t comp[2]={-1,-1};
                    for(int side=0;side<2;side++)if(candidate[side]<nf){
                        size_t f=candidate[side];
                        comp[side]=topology->vertex_component[faces[f*3]];
                        for(int k=0;k<3;k++){
                            size_t e=face_edge[f*3+(size_t)k];
                            if(e!=SIZE_MAX&&edge_count[e]==1)boundary[side]++;
                        }
                    }
                    fprintf(stderr,
                            "  unresolved pair faces=%zu/%zu charts=%d/%d "
                            "boundary_edges=%zu/%zu degree=%zu/%zu "
                            "remaining=%zu/%zu kind=%s\n",
                            fa,fb,comp[0],comp[1],boundary[0],boundary[1],
                            fa<nf?degree[fa]:0,fb<nf?degree[fb]:0,
                            comp[0]>=0?component_remaining[(size_t)comp[0]]:0,
                            comp[1]>=0?component_remaining[(size_t)comp[1]]:0,
                            pair[i].overlap_pairs?"overlap":
                            pair[i].stab_pairs?"stab":"fold");
                    for(int side=0;side<2;side++)if(candidate[side]<nf){
                        size_t f=candidate[side];
                        fprintf(stderr,"    f%zu",f);
                        for(int k=0;k<3;k++){
                            int32_t v=faces[f*3+(size_t)k];
                            fprintf(stderr," (%.6f %.6f %.6f)",
                                    verts[(size_t)v*3],verts[(size_t)v*3+1],
                                    verts[(size_t)v*3+2]);
                        }
                        fputc('\n',stderr);
                    }
                    shown++;
                }
            }
            goto done;
        }
        {
            int32_t comp=topology->vertex_component[faces[best*3]];
            face_remove[best]=1;peeled++;
            component_remaining[(size_t)comp]--;
            if(!touched[(size_t)comp]){touched[(size_t)comp]=1;ntouched++;}
            for(int k=0;k<3;k++){
                size_t e=face_edge[best*3+(size_t)k];
                if(e!=SIZE_MAX&&edge_count[e]>0)edge_count[e]--;
            }
            for(size_t i=0;i<npair;i++)if(active[i]&&
                (pair[i].face_a==best||pair[i].face_b==best)){
                size_t other=pair[i].face_a==best?pair[i].face_b:pair[i].face_a;
                active[i]=0;unresolved--;
                if(degree[best]>0)degree[best]--;
                if(other<nf&&degree[other]>0)degree[other]--;
            }
        }
    }
    if(out_peeled)*out_peeled=peeled;
    if(out_touched)*out_touched=ntouched;
    rc=0;
done:
    free(touched);free(active);free(visited);free(queue);
    free(component_remaining);free(component_next);free(component_head);
    free(neighbor);free(degree);free(edge_count);free(face_edge);free(he);
    return rc;
}

static const char *source_chart_cube_name(size_t component,
                                          const int32_t *component_cube,
                                          const CubeList *cubes)
{
    int32_t ci;
    if(!component_cube)return "unknown";
    ci=component_cube[component];
    if(ci==-2)return "mixed";
    if(ci>=0&&cubes&&(size_t)ci<cubes->n)return cubes->ids[(size_t)ci];
    return "unknown";
}

static int source_chart_resolve_conflicts(
    const TopologyAuditReport *topology,const double *component_area,
    const double *component_diag,const SourceChartConflictEdge *edge,
    size_t nedge,const int32_t *component_cube,const CubeList *cubes,
    uint8_t *drop,int verbose,size_t *out_removed,size_t *out_deferred)
{
    size_t nc=topology->face_components,removed=0;
    size_t deferred=0,hard_errors=0;
    SourceChartOwnershipEdge *owner=NULL;
    SourceChartOrder *order=NULL;
    int rc=-1;
    if(out_removed)*out_removed=0;
    if(out_deferred)*out_deferred=0;
    owner=(SourceChartOwnershipEdge*)malloc((nedge?nedge:1)*sizeof(*owner));
    order=(SourceChartOrder*)malloc((nc?nc:1)*sizeof(*order));
    if(!owner||!order)goto done;
    for(size_t i=0;i<nedge;i++){
        size_t a=edge[i].a,b=edge[i].b;
        int a_loses,b_loses;
        owner[i].loser=owner[i].winner=SIZE_MAX;
        owner[i].face_pairs=0;
        if(a>=nc||b>=nc){
            fprintf(stderr,
                    "ERROR: invalid source-chart conflict edge %zu/%zu "
                    "for %zu component(s)\n",a,b,nc);
            hard_errors++;
            continue;
        }
        if(a==b){
            if(deferred<32&&verbose)fprintf(stderr,
                "  source chart boundary-shell candidate: chart %zu has an "
                "exact self-contact "
                "(%zu face pair(s): %zu overlap/%zu stab/%zu fold; "
                "representative faces %zu/%zu, cube=%s)\n",
                a,edge[i].face_pairs,edge[i].overlap_pairs,
                edge[i].stab_pairs,edge[i].fold_pairs,
                edge[i].face_a,edge[i].face_b,
                a<nc?source_chart_cube_name(a,component_cube,cubes):"invalid");
            deferred++;
            continue;
        }
        a_loses=source_chart_smaller_support_is_unambiguous(
            topology->component[a].faces,component_area[a],component_diag[a],
            topology->component[b].faces,component_area[b],component_diag[b]);
        b_loses=source_chart_smaller_support_is_unambiguous(
            topology->component[b].faces,component_area[b],component_diag[b],
            topology->component[a].faces,component_area[a],component_diag[a]);
        if(a_loses==b_loses){
            if(deferred<32&&verbose)fprintf(stderr,
                    "  source chart boundary-shell candidate: near-peer "
                    "geometry between chart %zu (%zuf/%.1farea/%.1fdiag) "
                    "and chart %zu "
                    "(%zuf/%.1farea/%.1fdiag), %zu exact face pair(s) "
                    "[%zu overlap/%zu stab/%zu fold], "
                    "cubes=%s/%s representative faces=%zu/%zu\n",
                    a,topology->component[a].faces,component_area[a],
                    component_diag[a],b,topology->component[b].faces,
                    component_area[b],component_diag[b],edge[i].face_pairs,
                    edge[i].overlap_pairs,edge[i].stab_pairs,
                    edge[i].fold_pairs,
                    source_chart_cube_name(a,component_cube,cubes),
                    source_chart_cube_name(b,component_cube,cubes),
                    edge[i].face_a,edge[i].face_b);
            deferred++;
            continue;
        }
        owner[i].loser=a_loses?a:b;
        owner[i].winner=a_loses?b:a;
        owner[i].face_pairs=edge[i].face_pairs;
    }
    if(hard_errors>0){
        if(hard_errors>32)fprintf(stderr,
                "ERROR: %zu invalid/self source conflict edge(s) "
                "(first 32 shown)\n",hard_errors);
        goto done;
    }
    for(size_t c=0;c<nc;c++){
        order[c].component=c;
        order[c].faces=topology->component[c].faces;
        order[c].area=component_area[c];
        order[c].diag=component_diag[c];
    }
    qsort(order,nc,sizeof(*order),source_chart_order_cmp);
    for(size_t oi=0;oi<nc;oi++){
        size_t c=order[oi].component,winner=SIZE_MAX,pairs=0;
        if(drop[c])continue;
        for(size_t e=0;e<nedge;e++){
            if(owner[e].loser!=SIZE_MAX&&owner[e].loser==c&&
               !drop[owner[e].winner]){
                winner=owner[e].winner;pairs=owner[e].face_pairs;break;
            }
        }
        if(winner==SIZE_MAX)continue;
        drop[c]=1;removed++;
        if(verbose)fprintf(stderr,
            "  source chart ownership: removed whole %zu-face chart %zu "
            "(area %.1f, diag %.1f) from %zu exact pair(s) with retained "
            "chart %zu (%zu faces, area %.1f, diag %.1f)\n",
            topology->component[c].faces,c,component_area[c],component_diag[c],
            pairs,winner,topology->component[winner].faces,
            component_area[winner],component_diag[winner]);
    }
    for(size_t e=0;e<nedge;e++){
        if(owner[e].loser==SIZE_MAX)continue;
        if(!drop[owner[e].loser]&&!drop[owner[e].winner]){
            fprintf(stderr,"ERROR: unresolved source ownership edge %zu->%zu\n",
                    owner[e].loser,owner[e].winner);
            goto done;
        }
    }
    if(out_removed)*out_removed=removed;
    if(out_deferred)*out_deferred=deferred;
    rc=0;
done:
    free(order);free(owner);
    return rc;
}

static int drop_closed_source_bubbles(Arena_T arena,
                                      float *verts, size_t *nv,
                                      int32_t *faces, size_t *nf,
                                      int16_t *vert_cube_idx,
                                      uint8_t *vert_is_weld,
                                      int32_t *lineage_parent0,
                                      int32_t *lineage_parent1,
                                      size_t *out_closed,
                                      size_t *out_dropped_faces,
                                      size_t *out_dropped_verts)
{
    Arena_Mark mark = Arena_save(arena);
    TopologyAuditReport topology;
    uint8_t *drop = NULL, *used = NULL;
    int32_t *map = NULL;
    size_t old_nv = *nv, old_nf = *nf;
    size_t closed = 0, write_f = 0, write_v = 0;
    int rc = -1;

    memset(&topology, 0, sizeof topology);
    if (TopologyAudit_analyze(verts, *nv, faces, *nf, NULL,
                              &topology) != 0)
        goto done;
    drop = (uint8_t *)ARENA_CALLOC(
        arena, topology.face_components ? topology.face_components : 1, 1);
    for (size_t c = 0; c < topology.face_components; c++) {
        const TopologyComponentInvariant *q = &topology.component[c];
        if (q->surface_valid && q->orientable && q->beta_0 == 1 &&
            q->beta_1 == 0 && q->beta_2 == 1 && q->boundary_edges == 0) {
            drop[c] = 1;
            closed++;
        }
    }
    if (closed == 0) {
        rc = 0;
        goto done;
    }
    used = (uint8_t *)ARENA_CALLOC(arena, *nv ? *nv : 1, 1);
    map = (int32_t *)ARENA_ALLOC(
        arena, (*nv ? *nv : 1) * sizeof(*map));
    for (size_t f = 0; f < *nf; f++) {
        int32_t a = faces[f*3+0], b = faces[f*3+1], c = faces[f*3+2];
        int32_t component = topology.vertex_component[a];
        if (component < 0 ||
            (size_t)component >= topology.face_components ||
            component != topology.vertex_component[b] ||
            component != topology.vertex_component[c])
            goto done;
        if (drop[(size_t)component]) continue;
        faces[write_f*3+0] = a;
        faces[write_f*3+1] = b;
        faces[write_f*3+2] = c;
        used[a] = used[b] = used[c] = 1;
        write_f++;
    }
    for (size_t v = 0; v < *nv; v++) {
        if (!used[v]) {
            map[v] = -1;
            continue;
        }
        map[v] = (int32_t)write_v;
        if (write_v != v) {
            memmove(&verts[write_v*3], &verts[v*3], 3*sizeof(float));
            if (vert_cube_idx)
                vert_cube_idx[write_v] = vert_cube_idx[v];
            if (vert_is_weld)
                vert_is_weld[write_v] = vert_is_weld[v];
            if (lineage_parent0)
                lineage_parent0[write_v] = lineage_parent0[v];
            if (lineage_parent1)
                lineage_parent1[write_v] = lineage_parent1[v];
        }
        write_v++;
    }
    for (size_t f = 0; f < write_f; f++) {
        faces[f*3+0] = map[faces[f*3+0]];
        faces[f*3+1] = map[faces[f*3+1]];
        faces[f*3+2] = map[faces[f*3+2]];
    }
    *nv = write_v;
    *nf = write_f;
    rc = 0;
done:
    if (rc == 0) {
        if (out_closed) *out_closed = closed;
        if (out_dropped_faces) *out_dropped_faces = old_nf - *nf;
        if (out_dropped_verts) *out_dropped_verts = old_nv - *nv;
    }
    TopologyAudit_dispose(&topology);
    Arena_restore(arena, mark);
    return rc;
}

static int source_disk_repair_with_metadata(
    Arena_T arena, float **verts, size_t *nv,
    int32_t **faces, size_t *nf, size_t vert_capacity,
    int16_t *vert_cube_idx, uint8_t *vert_is_weld,
    int32_t *lineage_parent0, int32_t *lineage_parent1,
    DiskTopologyRepairStats *stats)
{
    const size_t old_nv = *nv, old_nf = *nf;
    float *cut_verts = NULL;
    int32_t *cut_faces = NULL, *cut_source = NULL;
    size_t cut_nv = 0, cut_nf = 0;
    int16_t *old_cube = NULL;
    uint8_t *old_weld = NULL;
    int32_t *old_parent0 = NULL, *old_parent1 = NULL;
    DiskTopologyRepairStats local_stats;
    int rc = -1;

    memset(&local_stats, 0, sizeof local_stats);
    old_cube = (int16_t *)malloc(old_nv * sizeof(*old_cube));
    old_weld = (uint8_t *)malloc(old_nv * sizeof(*old_weld));
    if (lineage_parent0 && lineage_parent1) {
        old_parent0 = (int32_t *)malloc(old_nv * sizeof(*old_parent0));
        old_parent1 = (int32_t *)malloc(old_nv * sizeof(*old_parent1));
    }
    if (!old_cube || !old_weld ||
        ((lineage_parent0 && lineage_parent1) &&
         (!old_parent0 || !old_parent1)))
        goto done;
    memcpy(old_cube, vert_cube_idx, old_nv * sizeof(*old_cube));
    memcpy(old_weld, vert_is_weld, old_nv * sizeof(*old_weld));
    if (old_parent0 && old_parent1) {
        memcpy(old_parent0, lineage_parent0,
               old_nv * sizeof(*old_parent0));
        memcpy(old_parent1, lineage_parent1,
               old_nv * sizeof(*old_parent1));
    }
    if (DiskTopologyRepair_process(
            arena, *verts, *nv, *faces, *nf,
            &cut_verts, &cut_nv, &cut_faces, &cut_nf,
            &cut_source, &local_stats) != 0 ||
        cut_nf != old_nf || cut_nv > vert_capacity)
        goto done;
    for (size_t v = 0; v < cut_nv; v++) {
        int32_t source = cut_source[v];
        if (source < 0 || (size_t)source >= old_nv)
            goto done;
    }
    for (size_t v = 0; v < cut_nv; v++) {
        size_t source = (size_t)cut_source[v];
        vert_cube_idx[v] = old_cube[source];
        vert_is_weld[v] = old_weld[source];
        if (old_parent0 && old_parent1) {
            lineage_parent0[v] = old_parent0[source];
            lineage_parent1[v] = old_parent1[source];
        }
    }
    *verts = cut_verts; *nv = cut_nv;
    *faces = cut_faces; *nf = cut_nf;
    rc = 0;
done:
    if (stats) *stats = local_stats;
    free(old_parent1); free(old_parent0);
    free(old_weld); free(old_cube);
    return rc;
}

static int source_depinch_with_metadata(
    Arena_T arena, float **verts, size_t *nv,
    int32_t *faces, size_t nf, size_t vert_capacity,
    int16_t *vert_cube_idx, uint8_t *vert_is_weld,
    int32_t *lineage_parent0, int32_t *lineage_parent1,
    size_t *out_splits)
{
    size_t old_nv;
    ComponentMesh mesh;
    int32_t *vertex_source = NULL;
    size_t splits = 0;

    if (out_splits) *out_splits = 0;
    if (!arena || !verts || !nv || !*verts || !faces || nf == 0 ||
        !vert_cube_idx || !vert_is_weld || *nv == 0)
        return -1;
    old_nv = *nv;
    memset(&mesh, 0, sizeof mesh);
    mesh.verts = *verts;
    mesh.faces = faces;
    mesh.nv = old_nv;
    mesh.nf = nf;
    mesh.comp_id = 1;
    mesh.self = &mesh;
    if (PinholeFill_split_pinches_mapped(
            arena, &mesh, &splits, &vertex_source) != 0 ||
        !vertex_source || mesh.nv > vert_capacity)
        return -1;
    for (size_t v = 0; v < mesh.nv; v++) {
        int32_t source = vertex_source[v];
        if (source < 0 || (size_t)source >= old_nv ||
            (v < old_nv && (size_t)source != v))
            return -1;
    }
    for (size_t v = old_nv; v < mesh.nv; v++) {
        size_t source = (size_t)vertex_source[v];
        vert_cube_idx[v] = vert_cube_idx[source];
        vert_is_weld[v] = vert_is_weld[source];
        if (lineage_parent0 && lineage_parent1) {
            lineage_parent0[v] = lineage_parent0[source];
            lineage_parent1[v] = lineage_parent1[source];
        }
    }
    *verts = mesh.verts;
    *nv = mesh.nv;
    if (out_splits) *out_splits = splits;
    return 0;
}

static int source_self_conflict_mask(const float *verts, size_t nv,
                                     const int32_t *faces, size_t nf,
                                     uint8_t **out_face_mask,
                                     size_t *out_components,
                                     size_t *out_pairs)
{
    TopologyAuditReport topology;
    SourceChartConflictCollector conflicts;
    IntersectionCleanupParams params;
    IntersectionCleanupStats audit;
    uint8_t *self_component = NULL, *mask = NULL;
    size_t components = 0, pairs = 0;
    int rc = -1;

    *out_face_mask = NULL;
    if (out_components) *out_components = 0;
    if (out_pairs) *out_pairs = 0;
    memset(&topology, 0, sizeof topology);
    memset(&conflicts, 0, sizeof conflicts);
    memset(&audit, 0, sizeof audit);
    if (TopologyAudit_analyze(verts, nv, faces, nf, NULL, &topology) != 0)
        goto done;
    self_component = (uint8_t *)calloc(
        topology.face_components ? topology.face_components : 1, 1);
    if (!self_component) goto done;
    IntersectionCleanup_default_params(&params);
    params.gap_max = 0.0;
    params.include_hinges = 0;
    conflicts.faces = faces;
    conflicts.topology = &topology;
    if (IntersectionCleanup_audit_visit(
            verts, nv, faces, nf, NULL, &params, NULL,
            source_chart_collect_conflict, &conflicts, &audit) != 0)
        goto done;
    for (size_t i = 0; i < conflicts.n; i++) {
        if (conflicts.edge[i].a != conflicts.edge[i].b) continue;
        pairs++;
        if (!self_component[conflicts.edge[i].a]) {
            self_component[conflicts.edge[i].a] = 1;
            components++;
        }
    }
    if (components > 0) {
        mask = (uint8_t *)calloc(nf ? nf : 1, 1);
        if (!mask) goto done;
        for (size_t f = 0; f < nf; f++) {
            int32_t component = topology.vertex_component[faces[f*3]];
            if (component < 0 ||
                (size_t)component >= topology.face_components)
                goto done;
            mask[f] = self_component[(size_t)component];
        }
    }
    *out_face_mask = mask;
    mask = NULL;
    if (out_components) *out_components = components;
    if (out_pairs) *out_pairs = pairs;
    rc = 0;
done:
    free(mask);
    free(self_component);
    free(conflicts.edge);
    TopologyAudit_dispose(&topology);
    return rc;
}

static int certify_source_charts(Arena_T arena,
                                 float *verts, size_t *nv,
                                 int32_t *faces, size_t *nf,
                                 size_t vert_capacity,
                                 int16_t *vert_cube_idx,
                                 const CubeList *cubes,
                                 uint8_t *vert_is_weld,
                                 int32_t *lineage_parent0,
                                 int32_t *lineage_parent1,
                                 size_t *out_closed_charts,
                                 size_t *out_overlap_charts,
                                 size_t *out_dropped_faces,
                                 size_t *out_dropped_verts)
{
    Arena_Mark mark = Arena_save(arena);
    TopologyAuditReport before, after;
    uint8_t *drop = NULL, *used = NULL, *face_mask = NULL;
    uint8_t *face_remove = NULL;
    SourceChartConflictCollector conflicts;
    SourceChartConflictEdge *raw_conflicts = NULL;
    double *component_area = NULL, *component_diag = NULL;
    double *component_min = NULL, *component_max = NULL;
    int32_t *component_cube = NULL, *map = NULL;
    size_t closed = 0, overlap = 0, write_f = 0, write_v = 0;
    size_t raw_conflict_count = 0, dropped_components = 0;
    size_t old_nf = *nf, old_nv = *nv;
    size_t compacted_nv = 0;
    int rc = -1;
    memset(&before, 0, sizeof before);
    memset(&after, 0, sizeof after);
    memset(&conflicts, 0, sizeof conflicts);
    if (TopologyAudit_analyze(verts, *nv, faces, *nf, NULL, &before) != 0)
        goto done;
    drop = (uint8_t *)ARENA_CALLOC(
        arena, (before.face_components ? before.face_components : 1), 1L);
    component_area=(double*)ARENA_CALLOC(
        arena,(before.face_components?before.face_components:1),
        sizeof(*component_area));
    component_diag=(double*)ARENA_CALLOC(
        arena,(before.face_components?before.face_components:1),
        sizeof(*component_diag));
    component_min=(double*)ARENA_ALLOC(
        arena,((before.face_components?before.face_components:1)*3*
                     sizeof(*component_min)));
    component_max=(double*)ARENA_ALLOC(
        arena,((before.face_components?before.face_components:1)*3*
                     sizeof(*component_max)));
    component_cube=(int32_t*)ARENA_ALLOC(
        arena,((before.face_components?before.face_components:1)*
                     sizeof(*component_cube)));
    for(size_t c=0;c<before.face_components;c++)component_cube[c]=-1;
    for(size_t c=0;c<before.face_components;c++)for(int d=0;d<3;d++){
        component_min[c*3+(size_t)d]=HUGE_VAL;
        component_max[c*3+(size_t)d]=-HUGE_VAL;
    }
    for(size_t v=0;v<*nv;v++){
        int32_t c=before.vertex_component[v];
        if(c<0)continue;
        if(vert_cube_idx&&vert_cube_idx[v]>=0){
            int32_t ci=vert_cube_idx[v];
            if(component_cube[(size_t)c]==-1)component_cube[(size_t)c]=ci;
            else if(component_cube[(size_t)c]!=ci)
                component_cube[(size_t)c]=-2;
        }
        for(int d=0;d<3;d++){
            double q=(double)verts[v*3+(size_t)d];
            if(q<component_min[(size_t)c*3+(size_t)d])
                component_min[(size_t)c*3+(size_t)d]=q;
            if(q>component_max[(size_t)c*3+(size_t)d])
                component_max[(size_t)c*3+(size_t)d]=q;
        }
    }
    for(size_t f=0;f<*nf;f++){
        int32_t a=faces[f*3],b=faces[f*3+1],c=faces[f*3+2];
        int32_t component=before.vertex_component[a];
        double ab[3],ac[3],cross[3];
        for(int d=0;d<3;d++){
            ab[d]=(double)verts[(size_t)b*3+(size_t)d]-
                  (double)verts[(size_t)a*3+(size_t)d];
            ac[d]=(double)verts[(size_t)c*3+(size_t)d]-
                  (double)verts[(size_t)a*3+(size_t)d];
        }
        cross[0]=ab[1]*ac[2]-ab[2]*ac[1];
        cross[1]=ab[2]*ac[0]-ab[0]*ac[2];
        cross[2]=ab[0]*ac[1]-ab[1]*ac[0];
        component_area[(size_t)component]+=
            0.5*sqrt(cross[0]*cross[0]+cross[1]*cross[1]+
                     cross[2]*cross[2]);
    }
    for(size_t c=0;c<before.face_components;c++){
        double sum=0.0;
        for(int d=0;d<3;d++){
            double q=component_max[c*3+(size_t)d]-
                     component_min[c*3+(size_t)d];
            sum+=q*q;
        }
        component_diag[c]=sqrt(sum);
    }
    for (size_t c = 0; c < before.face_components; c++) {
        const TopologyComponentInvariant *q = &before.component[c];
        int32_t source_cube = -1;
        int mixed_source_cubes = 0;
        if (q->homeomorphic_to_disk) continue;
        if (q->surface_valid && q->orientable && q->beta_0 == 1 &&
            q->beta_1 == 0 && q->beta_2 == 1 && q->boundary_edges == 0) {
            drop[c] = 1;
            closed++;
            continue;
        }
        if (vert_cube_idx != NULL) {
            for (size_t v = 0; v < *nv; v++) {
                int32_t ci;
                if (before.vertex_component[v] != (int32_t)c) continue;
                ci = vert_cube_idx[v];
                if (ci < 0) continue;
                if (source_cube < 0) source_cube = ci;
                else if (ci != source_cube) mixed_source_cubes = 1;
            }
        }
        fprintf(stderr,
                "ERROR: source chart %zu is neither a disk nor a closed "
                "bubble (faces=%zu boundary=%zu/%zu beta=(%lld,%lld,%lld), "
                "chi=%lld valid=%d orientable=%d winding=%d, nonmanifold="
                "%zuE/%zuV irregular-boundary=%zu same-direction=%zu, "
                "defect=0x%08x)\n",
                c, q->faces, q->boundary_loops, q->boundary_edges,
                (long long)q->beta_0, (long long)q->beta_1,
                (long long)q->beta_2, (long long)q->euler_characteristic,
                q->surface_valid, q->orientable,
                q->input_winding_consistent, q->nonmanifold_edges,
                q->nonmanifold_vertices, q->boundary_irregular_vertices,
                q->same_direction_edges, (unsigned)q->defect_mask);
        fprintf(stderr,
                "  source chart %zu provenance: cube=%d%s%s%s, "
                "bbox=[%.6g %.6g %.6g]-[%.6g %.6g %.6g]\n",
                c, (int)source_cube,
                source_cube >= 0 && cubes != NULL &&
                    (size_t)source_cube < cubes->n ? " (" : "",
                source_cube >= 0 && cubes != NULL &&
                    (size_t)source_cube < cubes->n ?
                    cubes->ids[(size_t)source_cube] : "",
                source_cube >= 0 && cubes != NULL &&
                    (size_t)source_cube < cubes->n ? ")" : "",
                q->bbox_min[0], q->bbox_min[1], q->bbox_min[2],
                q->bbox_max[0], q->bbox_max[1], q->bbox_max[2]);
        if (mixed_source_cubes)
            fprintf(stderr, "  source chart %zu spans multiple input cubes\n", c);
        if (getenv("GW_DROP_NONDISK_CHARTS") != NULL) {
            fprintf(stderr,
                    "  GW_DROP_NONDISK_CHARTS: quarantining chart %zu "
                    "(%zu faces)\n", c, q->faces);
            drop[c] = 1;
            continue;
        }
        goto done;
    }

    face_mask = (uint8_t *)ARENA_CALLOC(arena, (*nf ? *nf : 1), 1L);
    {
        IntersectionCleanupParams ip;
        IntersectionCleanupStats ia;
        size_t resolved=0,deferred=0,peeled=0,touched=0;
        for (size_t f = 0; f < *nf; f++) {
            int32_t component = before.vertex_component[faces[f*3]];
            face_mask[f] = component >= 0 &&
                           !drop[(size_t)component] ? 1 : 0;
        }
        IntersectionCleanup_default_params(&ip);
        ip.gap_max = 0.0;
        ip.include_hinges = 0;
        conflicts.faces=faces;
        conflicts.topology=&before;
        if (IntersectionCleanup_audit_visit(
                verts, *nv, faces, *nf, face_mask, &ip,
                NULL,source_chart_collect_conflict,&conflicts,&ia) != 0)
            goto done;
        if(conflicts.n>0){
            if(conflicts.n>SIZE_MAX/sizeof(*raw_conflicts))goto done;
            raw_conflicts=(SourceChartConflictEdge*)malloc(
                conflicts.n*sizeof(*raw_conflicts));
            if(!raw_conflicts)goto done;
            memcpy(raw_conflicts,conflicts.edge,
                   conflicts.n*sizeof(*raw_conflicts));
            raw_conflict_count=conflicts.n;
        }
        source_chart_dedupe_conflicts(&conflicts);
        if(source_chart_resolve_conflicts(
                &before,component_area,component_diag,conflicts.edge,
                conflicts.n,component_cube,cubes,drop,1,&resolved,
                &deferred)!=0)
            goto done;
        overlap+=resolved;
        face_remove=(uint8_t*)ARENA_CALLOC(
            arena,(*nf?*nf:1),1L);
        for(size_t f=0;f<*nf;f++){
            int32_t component=before.vertex_component[faces[f*3]];
            if(component<0||(size_t)component>=before.face_components)
                goto done;
            face_remove[f]=drop[(size_t)component]?1:0;
        }
        if(deferred>0&&source_chart_boundary_peel(
                verts,*nv,faces,*nf,&before,component_area,
                raw_conflicts,raw_conflict_count,face_remove,
                1,&peeled,&touched)!=0)
            goto done;
        if(ia.conflicts>0)fprintf(stderr,
            "  source chart embedded repair: %zu face conflict(s), %zu "
            "component edge(s), %zu whole chart(s) removed; "
            "%zu boundary face(s) shelled from %zu chart(s) across %zu "
            "near-peer edge(s)\n",
            ia.conflicts,conflicts.n,resolved,peeled,touched,deferred);
    }

    for(size_t c=0;c<before.face_components;c++)
        if(drop[c])dropped_components++;
    used = (uint8_t *)ARENA_CALLOC(arena, (*nv ? *nv : 1), 1L);
    map = (int32_t *)ARENA_ALLOC(
        arena, ((*nv ? *nv : 1) * sizeof(*map)));
    for (size_t f = 0; f < *nf; f++) {
        int32_t a = faces[f*3+0], b = faces[f*3+1], c = faces[f*3+2];
        int32_t component = before.vertex_component[a];
        if (component < 0 || (size_t)component >= before.face_components)
            goto done;
        if (face_remove[f]) continue;
        faces[write_f*3+0] = a;
        faces[write_f*3+1] = b;
        faces[write_f*3+2] = c;
        used[a] = used[b] = used[c] = 1;
        write_f++;
    }
    for (size_t v = 0; v < *nv; v++) {
        if (!used[v]) { map[v] = -1; continue; }
        map[v] = (int32_t)write_v;
        if (write_v != v) {
            memmove(&verts[write_v*3], &verts[v*3], 3*sizeof(float));
            if (vert_cube_idx) vert_cube_idx[write_v] = vert_cube_idx[v];
            if (vert_is_weld) vert_is_weld[write_v] = vert_is_weld[v];
            if (lineage_parent0)
                lineage_parent0[write_v] = lineage_parent0[v];
            if (lineage_parent1)
                lineage_parent1[write_v] = lineage_parent1[v];
        }
        write_v++;
    }
    for (size_t f = 0; f < write_f; f++) {
        faces[f*3+0] = map[faces[f*3+0]];
        faces[f*3+1] = map[faces[f*3+1]];
        faces[f*3+2] = map[faces[f*3+2]];
    }
    *nv = write_v;
    *nf = write_f;
    compacted_nv = write_v;
    if (*nf == 0 || TopologyAudit_analyze(verts, *nv, faces, *nf,
                                           NULL, &after) != 0 ||
        dropped_components>before.face_components ||
        after.face_components!=before.face_components-dropped_components) {
        fprintf(stderr,
                "ERROR: source chart certificate changed chart ownership "
                "during embedded repair (%zu/%zu disks, components=%zu expected=%zu, "
                "isolated=%zu, beta1=%lld)\n",
                after.disk_components, after.face_components,
                after.face_components,
                dropped_components<=before.face_components?
                    before.face_components-dropped_components:0,
                after.isolated_vertices, (long long)after.beta_1);
        goto done;
    }
    if (!after.all_components_are_disks || after.isolated_vertices != 0) {
        float *repair_verts = verts;
        int32_t *repair_faces = faces;
        size_t pinch_splits = 0;
        DiskTopologyRepairStats recut;
        memset(&recut, 0, sizeof recut);
        TopologyAudit_dispose(&after);
        memset(&after, 0, sizeof after);
        if (source_depinch_with_metadata(
                arena, &repair_verts, nv, repair_faces, *nf,
                vert_capacity, vert_cube_idx, vert_is_weld,
                lineage_parent0, lineage_parent1, &pinch_splits) != 0 ||
            source_disk_repair_with_metadata(
                arena, &repair_verts, nv, &repair_faces, nf,
                vert_capacity, vert_cube_idx, vert_is_weld,
                lineage_parent0, lineage_parent1, &recut) != 0 ||
            *nv > vert_capacity || *nf > old_nf) {
            fprintf(stderr,
                    "ERROR: source chart interior-conflict disk recut failed "
                    "(%zu pinch split(s), %zu/%zu component(s) repaired, "
                    "%zu failure(s), capacity=%zu)\n",
                    pinch_splits, recut.repaired_components,
                    recut.components, recut.failed_components,
                    vert_capacity);
            goto done;
        }
        if (repair_verts != verts)
            memmove(verts, repair_verts, *nv*3*sizeof(*verts));
        if (repair_faces != faces)
            memmove(faces, repair_faces, *nf*3*sizeof(*faces));
        if (TopologyAudit_analyze(verts, *nv, faces, *nf,
                                  NULL, &after) != 0) {
            fprintf(stderr,
                    "ERROR: source chart topology audit failed after "
                    "interior-conflict disk recut\n");
            goto done;
        }
        fprintf(stderr,
                "  source chart interior-conflict repair: %zu pinch split(s), "
                "%zu component(s) recut with %zu seam edge(s); %zu -> %zu "
                "vertices, %zu faces preserved\n",
                pinch_splits, recut.repaired_components, recut.seam_edges,
                compacted_nv, *nv, *nf);
    }
    if (!after.all_components_are_disks || after.isolated_vertices != 0 ||
        after.face_components!=before.face_components-dropped_components) {
        fprintf(stderr,
                "ERROR: source chart certificate failed after topology-safe "
                "embedded repair (%zu/%zu disks, components=%zu expected=%zu, "
                "isolated=%zu, beta1=%lld)\n",
                after.disk_components, after.face_components,
                after.face_components,
                dropped_components<=before.face_components?
                    before.face_components-dropped_components:0,
                after.isolated_vertices, (long long)after.beta_1);
        goto done;
    }
    {
        IntersectionCleanupParams ip;
        IntersectionCleanupStats ia;
        IntersectionCleanup_default_params(&ip);
        ip.gap_max = 0.0;
        ip.include_hinges = 0;
        if (IntersectionCleanup_audit(
                verts, *nv, faces, *nf, NULL, &ip, NULL, &ia) != 0 ||
            ia.conflicts != 0) {
            fprintf(stderr,
                    "ERROR: source chart embedded-geometry certificate "
                    "failed after ownership/boundary-shell selection "
                    "(%zu conflicts)\n",
                    ia.conflicts);
            goto done;
        }
    }
    if (out_closed_charts) *out_closed_charts = closed;
    if (out_overlap_charts) *out_overlap_charts = overlap;
    if (out_dropped_faces) *out_dropped_faces = old_nf - write_f;
    if (out_dropped_verts) *out_dropped_verts = old_nv - compacted_nv;
    rc = 0;
done:
    free(raw_conflicts);
    free(conflicts.edge);
    TopologyAudit_dispose(&after);
    TopologyAudit_dispose(&before);
    Arena_restore(arena, mark);
    return rc;
}

static size_t cull_tiny_components(Arena_T arena, int32_t *faces, size_t nf,
                                   size_t min_verts, size_t *out_nf,
                                   size_t *out_removed_comps);

static int source_depinch_recut_fixture(Arena_T arena,
                                        const int *removed_fan,
                                        size_t nremoved,
                                        size_t expected_splits,
                                        const char *label)
{
    enum { RING = 8, INPUT_NV = 17, INPUT_NF = 24, CAPACITY = 256 };
    float *verts = (float *)ARENA_ALLOC(
        arena, CAPACITY * 3 * sizeof(*verts));
    int32_t *faces = (int32_t *)ARENA_ALLOC(
        arena, INPUT_NF * 3 * sizeof(*faces));
    int16_t *cube = (int16_t *)ARENA_ALLOC(
        arena, CAPACITY * sizeof(*cube));
    uint8_t *weld = (uint8_t *)ARENA_CALLOC(arena, CAPACITY, 1L);
    int32_t *parent0 = (int32_t *)ARENA_ALLOC(
        arena, CAPACITY * sizeof(*parent0));
    int32_t *parent1 = (int32_t *)ARENA_ALLOC(
        arena, CAPACITY * sizeof(*parent1));
    TopologyAuditReport before, depinched, final;
    DiskTopologyRepairStats recut;
    size_t nv = INPUT_NV, nf = 0, splits = 0, nf_before;
    int local = 0;

    memset(&before, 0, sizeof before);
    memset(&depinched, 0, sizeof depinched);
    memset(&final, 0, sizeof final);
    memset(&recut, 0, sizeof recut);
    memset(verts, 0, CAPACITY * 3 * sizeof(*verts));
    for (int i = 0; i < RING; i++) {
        double a = 2.0*M_PI*(double)i/(double)RING;
        verts[(1+i)*3+1] = (float)cos(a);
        verts[(1+i)*3+2] = (float)sin(a);
        verts[(1+RING+i)*3+1] = (float)(2.0*cos(a));
        verts[(1+RING+i)*3+2] = (float)(2.0*sin(a));
    }
    for (int i = 0; i < RING; i++) {
        int removed = 0;
        int j = (i+1)%RING;
        for (size_t q = 0; q < nremoved; q++)
            if (removed_fan[q] == i) removed = 1;
        if (!removed) {
            faces[nf*3] = 0;
            faces[nf*3+1] = 1+i;
            faces[nf*3+2] = 1+j;
            nf++;
        }
        faces[nf*3] = 1+i;
        faces[nf*3+1] = 1+RING+i;
        faces[nf*3+2] = 1+RING+j;
        nf++;
        faces[nf*3] = 1+i;
        faces[nf*3+1] = 1+RING+j;
        faces[nf*3+2] = 1+j;
        nf++;
    }
    nf_before = nf;
    for (size_t v = 0; v < nv; v++) {
        cube[v] = (int16_t)(v%3);
        weld[v] = v == 0 ? 1u : 0u;
        parent0[v] = -2;
        parent1[v] = (int32_t)v;
    }
    local = TopologyAudit_analyze(
                verts, nv, faces, nf, NULL, &before) != 0 ||
            before.face_components != 1 ||
            (expected_splits == 0 ?
                before.invalid_surface_components != 0 :
                before.invalid_surface_components != 1);
    if (!local)
        local = source_depinch_with_metadata(
                    arena, &verts, &nv, faces, nf, CAPACITY,
                    cube, weld, parent0, parent1, &splits) != 0 ||
                splits != expected_splits ||
                nv != INPUT_NV + expected_splits;
    for (size_t v = INPUT_NV; !local && v < nv; v++) {
        local = cube[v] != cube[0] || weld[v] != weld[0] ||
                parent0[v] != parent0[0] || parent1[v] != parent1[0] ||
                memcmp(&verts[v*3], &verts[0], 3*sizeof(*verts)) != 0;
    }
    if (!local)
        local = TopologyAudit_analyze(
                    verts, nv, faces, nf, NULL, &depinched) != 0 ||
                depinched.face_components != 1 ||
                depinched.invalid_surface_components != 0 ||
                depinched.nonorientable_components != 0 ||
                depinched.inconsistent_winding_components != 0;
    if (!local)
        local = source_disk_repair_with_metadata(
                    arena, &verts, &nv, &faces, &nf, CAPACITY,
                    cube, weld, parent0, parent1, &recut) != 0 ||
                nf != nf_before || recut.failed_components != 0 ||
                recut.repaired_components != 1;
    if (!local)
        local = TopologyAudit_analyze(
                    verts, nv, faces, nf, NULL, &final) != 0 ||
                !final.all_components_are_disks ||
                final.invalid_surface_components != 0 ||
                final.inconsistent_winding_components != 0;
    fprintf(stderr,
            "[selftest] source cleanup %s -> %s "
            "(removed=%zu splits=%zu boundaries=%zu repaired=%zu)\n",
            label, local ? "FAIL" : "ok", nremoved, splits,
            depinched.boundary_components, recut.repaired_components);
    TopologyAudit_dispose(&final);
    TopologyAudit_dispose(&depinched);
    TopologyAudit_dispose(&before);
    return local ? 1 : 0;
}

static int source_chart_selftest(void)
{
    Arena_T arena = Arena_new();
    int fail = 0;
    {
        const int ordinary_hole[] = {0};
        const int two_touching[] = {0,4};
        const int three_touching[] = {0,3,6};
        fail += source_depinch_recut_fixture(
            arena, ordinary_hole, 1, 0, "ordinary-hole control");
        fail += source_depinch_recut_fixture(
            arena, two_touching, 2, 1, "two-face bowtie");
        fail += source_depinch_recut_fixture(
            arena, three_touching, 3, 2, "three-face bowtie");
    }
    {
        float *v = (float *)ARENA_ALLOC(arena, 6*3*sizeof(float));
        int32_t *f = (int32_t *)ARENA_ALLOC(arena, 6*3*sizeof(int32_t));
        const float vv[18] = {
            0,0,0, 0,10,0, 0,0,10,
            0,2,2, 0,6,2, 0,2,6
        };
        const int32_t ff[18] = {
            0,1,4, 0,4,3, 1,2,5,
            1,5,4, 2,0,3, 2,3,5
        };
        size_t nv = 6, nf = 6, loops = 0, holes = 0, filled = 0;
        TopologyAuditReport tr;
        memset(&tr, 0, sizeof tr);
        memcpy(v, vv, sizeof vv); memcpy(f, ff, sizeof ff);
        fail |= HoleFill_process_ex(arena, &v, &f, &nv, &nf, NULL, 2,
                                    &loops, &holes, &filled) != 0;
        fail |= TopologyAudit_analyze(v, nv, f, nf, NULL, &tr) != 0 ||
                !tr.all_components_are_disks || filled != 1;
        fprintf(stderr,
                "[selftest] source chart exact triangular hole -> %s "
                "(loops=%zu holes=%zu filled=%zu)\n",
                fail ? "FAIL" : "ok", loops, holes, filled);
        TopologyAudit_dispose(&tr);
    }
    {
        float v[5*3] = {
             0.0f,       0.0f,       0.0f,
             1.0f,       0.0f,       0.0f,
             0.5f,       0.8660254f,  0.0f,
            -0.5f,       0.8660254f,  0.0f,
            -1.0f,       0.0f,       0.0f
        };
        int32_t f[2*3] = {0,1,2, 0,3,4};
        ComponentMesh cm;
        TopologyAuditReport tr;
        size_t closed = 0;
        int local;
        memset(&cm, 0, sizeof cm);
        memset(&tr, 0, sizeof tr);
        cm.verts = v;
        cm.faces = f;
        cm.nv = 5;
        cm.nf = 2;
        cm.self = &cm;
        local = PinholeFill_close_bowties(arena, &cm, 1, &closed) != 0 ||
                closed != 1 || cm.nf != 3 ||
                TopologyAudit_analyze(cm.verts, cm.nv, cm.faces, cm.nf,
                                      NULL, &tr) != 0 ||
                !tr.all_components_are_disks || tr.beta_1 != 0 ||
                tr.disk_components != 1;
        fail |= local;
        fprintf(stderr,
                "[selftest] two-fan pinch spanning closure -> %s "
                "(closed=%zu faces=%zu disks=%zu beta1=%lld)\n",
                local ? "FAIL" : "ok", closed, cm.nf,
                tr.disk_components, (long long)tr.beta_1);
        TopologyAudit_dispose(&tr);
    }
    {
        int dense_micro=source_chart_smaller_support_is_unambiguous(
            165,32.0,9.0,530,5000.0,142.0);
        int scaled_dense=source_chart_smaller_support_is_unambiguous(
            289,40.8,24.1,527,17407.6,194.0);
        int denser_micro=source_chart_smaller_support_is_unambiguous(
            392,81.2,20.8,75,3885.6,120.8);
        int near_peer=source_chart_smaller_support_is_unambiguous(
            165,800.0,30.0,530,1000.0,60.0);
        int local=!dense_micro||!scaled_dense||!denser_micro||near_peer;
        fail|=local;
        fprintf(stderr,
                "[selftest] source physical-support ownership -> %s "
                "(dense-micro=%s scaled-dense=%s denser-micro=%s "
                "near-peer=%s)\n",
                local?"FAIL":"ok",dense_micro?"drop":"keep",
                scaled_dense?"drop":"keep",denser_micro?"drop":"keep",
                near_peer?"drop":"keep");
    }
    {
        TopologyAuditReport tr;
        TopologyComponentInvariant comp[3];
        SourceChartConflictEdge edge[2]={{0,1,1},{1,2,1}};
        double area[3]={1.0,20.0,400.0};
        double diag[3]={1.0,5.0,25.0};
        uint8_t drop[3]={0,0,0};
        size_t removed=0;
        memset(&tr,0,sizeof tr);memset(comp,0,sizeof comp);
        tr.face_components=3;tr.component=comp;
        comp[0].faces=10;comp[1].faces=50;comp[2].faces=200;
        int local=source_chart_resolve_conflicts(
            &tr,area,diag,edge,2,NULL,NULL,drop,0,&removed,NULL)!=0||
            removed!=1||drop[0]||!drop[1]||drop[2];
        fail|=local;
        fprintf(stderr,
                "[selftest] source ownership graph non-cascade -> %s\n",
                local?"FAIL":"ok");
    }
    {
        float v[8*3] = {
            0,0,0, 1,0,0, 1,1,0, 0,1,0,
            3,0,0, 4,0,0, 4,1,0, 3,1,0
        };
        int32_t f[4*3] = {0,1,2, 0,2,3, 4,5,6, 4,6,7};
        int32_t kept[4*3];
        SourceChartConflictEdge pair;
        TopologyAuditReport before,after;
        double area[2]={1.0,1.0};
        uint8_t remove[4]={0,0,0,0};
        size_t peeled=0,touched=0,nkeep=0;
        int local;
        memset(&pair,0,sizeof pair);
        memset(&before,0,sizeof before);
        memset(&after,0,sizeof after);
        local=TopologyAudit_analyze(v,8,f,4,NULL,&before)!=0||
              before.face_components!=2||!before.all_components_are_disks;
        if(!local){
            pair.a=(size_t)before.vertex_component[0];
            pair.b=(size_t)before.vertex_component[4];
            pair.face_pairs=1;pair.face_a=0;pair.face_b=2;
            local=source_chart_boundary_peel(
                v,8,f,4,&before,area,&pair,1,remove,0,
                &peeled,&touched)!=0||
                peeled!=1||touched!=1||!remove[0]||
                remove[1]||remove[2]||remove[3];
        }
        if(!local){
            for(size_t q=0;q<4;q++)if(!remove[q]){
                memcpy(&kept[nkeep*3],&f[q*3],3*sizeof(*kept));
                nkeep++;
            }
            local=TopologyAudit_analyze(v,8,kept,nkeep,NULL,&after)!=0||
                  after.face_components!=2||after.disk_components!=2;
        }
        fail|=local;
        fprintf(stderr,
                "[selftest] near-peer boundary-shell conflict cover -> %s "
                "(peeled=%zu touched=%zu components=%zu)\n",
                local?"FAIL":"ok",peeled,touched,after.face_components);
        TopologyAudit_dispose(&after);
        TopologyAudit_dispose(&before);
    }
    {
        float v[8*3] = {
             0, 0,0, -1, 1,0,  1, 1,0, -1,-1,0, 1,-1,0,
             3, 0,0,  4, 0,0,  3, 1,0
        };
        int32_t f[4*3] = {0,1,2, 1,0,3, 2,0,4, 5,6,7};
        SourceChartConflictEdge pair;
        TopologyAuditReport before,after;
        double area[2]={1.0,1.0};
        uint8_t remove[4]={0,0,0,0};
        size_t peeled=99,touched=99;
        int local;
        memset(&pair,0,sizeof pair);
        memset(&before,0,sizeof before);memset(&after,0,sizeof after);
        local=TopologyAudit_analyze(v,8,f,4,NULL,&before)!=0||
              before.face_components!=2||!before.all_components_are_disks;
        if(!local){
            pair.a=(size_t)before.vertex_component[0];
            pair.b=(size_t)before.vertex_component[5];
            pair.face_pairs=1;pair.face_a=0;pair.face_b=3;
            local=source_chart_boundary_peel(
                v,8,f,4,&before,area,&pair,1,remove,0,
                &peeled,&touched)!=0||
                peeled!=2||touched!=1||
                !remove[0]||remove[1]||!remove[2]||remove[3];
        }
        if(!local){
            int32_t kept[2*3];size_t nkeep=0;
            for(size_t q=0;q<4;q++)if(!remove[q]){
                memcpy(&kept[nkeep*3],&f[q*3],3*sizeof(*kept));nkeep++;
            }
            local=nkeep!=2||
                  TopologyAudit_analyze(v,8,kept,nkeep,NULL,&after)!=0||
                  after.face_components!=2||after.disk_components!=2;
        }
        fail|=local;
        fprintf(stderr,
                "[selftest] boundary-cap articulation closure -> %s "
                "(peeled=%zu components=%zu)\n",
                local?"FAIL":"ok",peeled,after.face_components);
        TopologyAudit_dispose(&after);
        TopologyAudit_dispose(&before);
    }
    {
        enum { INPUT_NV = 12, INPUT_NF = 14, CAPACITY = 64 };
        const float base_v[6*3] = {
             0, 0,0,  2, 0,0,  1, 1,0,
             1,-1,0,  2.5f,1,0, -0.5f,1,0
        };
        const int32_t base_f[7*3] = {
            0,1,2, 1,0,3, 2,1,4, 0,2,5,
            1,3,4, 2,4,5, 0,5,3
        };
        float *v=(float*)ARENA_CALLOC(arena,CAPACITY*3,sizeof(*v));
        int32_t *f=(int32_t*)ARENA_ALLOC(
            arena,INPUT_NF*3*sizeof(*f));
        int16_t *cube=(int16_t*)ARENA_CALLOC(
            arena,CAPACITY,sizeof(*cube));
        uint8_t *weld=(uint8_t*)ARENA_CALLOC(arena,CAPACITY,1L);
        int32_t *parent0=(int32_t*)ARENA_ALLOC(
            arena,CAPACITY*sizeof(*parent0));
        int32_t *parent1=(int32_t*)ARENA_ALLOC(
            arena,CAPACITY*sizeof(*parent1));
        TopologyAuditReport before,after;
        SourceChartConflictEdge pair;
        DiskTopologyRepairStats recut;
        double area[2]={10.0,10.0};
        uint8_t remove[INPUT_NF]={0};
        size_t nv=INPUT_NV,nf=INPUT_NF,peeled=0,touched=0,splits=0;
        int local=0;
        memset(&before,0,sizeof before);memset(&after,0,sizeof after);
        memset(&pair,0,sizeof pair);memset(&recut,0,sizeof recut);
        for(int component=0;component<2;component++){
            size_t vo=(size_t)component*6,fo=(size_t)component*7;
            for(size_t q=0;q<6;q++){
                memcpy(&v[(vo+q)*3],&base_v[q*3],3*sizeof(*v));
                v[(vo+q)*3]+=(float)(component*10);
            }
            for(size_t q=0;q<7*3;q++)
                f[fo*3+q]=base_f[q]+(int32_t)vo;
        }
        for(size_t q=0;q<CAPACITY;q++){
            cube[q]=(int16_t)(q<6?0:1);parent0[q]=-1;parent1[q]=-1;
        }
        local=TopologyAudit_analyze(v,nv,f,nf,NULL,&before)!=0||
              before.face_components!=2||before.disk_components!=2;
        if(!local){
            pair.a=(size_t)before.vertex_component[0];
            pair.b=(size_t)before.vertex_component[6];
            pair.face_pairs=1;pair.stab_pairs=1;
            pair.face_a=0;pair.face_b=7;
            local=source_chart_boundary_peel(
                v,nv,f,nf,&before,area,&pair,1,remove,0,
                &peeled,&touched)!=0||peeled!=1||touched!=1||
                !remove[0]||remove[7];
        }
        if(!local){
            size_t w=0;
            for(size_t q=0;q<nf;q++)if(!remove[q]){
                if(w!=q)memmove(&f[w*3],&f[q*3],3*sizeof(*f));
                w++;
            }
            nf=w;
            local=source_depinch_with_metadata(
                        arena,&v,&nv,f,nf,CAPACITY,cube,weld,
                        parent0,parent1,&splits)!=0||splits!=0||
                  source_disk_repair_with_metadata(
                        arena,&v,&nv,&f,&nf,CAPACITY,cube,weld,
                        parent0,parent1,&recut)!=0||
                  recut.failed_components!=0||
                  recut.repaired_components!=1||nf!=INPUT_NF-1;
        }
        if(!local)
            local=TopologyAudit_analyze(v,nv,f,nf,NULL,&after)!=0||
                  after.face_components!=2||after.disk_components!=2||
                  !after.all_components_are_disks;
        fail|=local;
        fprintf(stderr,
                "[selftest] interior conflict punch/recut -> %s "
                "(peeled=%zu splits=%zu repaired=%zu seam=%zu)\n",
                local?"FAIL":"ok",peeled,splits,
                recut.repaired_components,recut.seam_edges);
        TopologyAudit_dispose(&after);
        TopologyAudit_dispose(&before);
    }
    {
        float v[7*3] = {
            -1,0,0, 1,0,0, 0,1,1, 0,-1,1,
            -0.1f,0.2f,0.1f, 0.1f,0.2f,0.1f, 0,0.2f,0.3f
        };
        int32_t f[3*3] = {0,1,2, 1,0,3, 4,5,6};
        int16_t cube[7] = {0,0,0,0,1,1,1};
        uint8_t weld[7] = {0};
        size_t nv=7,nf=3,closed=0,overlap=0,df=0,dv=0;
        int local=certify_source_charts(
            arena,v,&nv,f,&nf,7,cube,NULL,weld,NULL,NULL,
            &closed,&overlap,&df,&dv)!=0||
            nv!=4||nf!=2||closed!=0||overlap!=1||df!=1||dv!=3;
        fail|=local;
        fprintf(stderr,
                "[selftest] source exact-overlap chart ownership -> %s "
                "(overlap=%zu faces=%zu verts=%zu)\n",
                local?"FAIL":"ok",overlap,df,dv);
    }
    {
        float v[7*3] = {
            0,0,0, 0,1,0, 0,0,1,
            3,0,0, 4,0,0, 3,1,0, 3,0,1
        };
        int32_t f[5*3] = {
            0,1,2,
            3,5,4, 3,4,6, 4,5,6, 5,3,6
        };
        int16_t cube[7] = {0,0,0,1,1,1,1};
        uint8_t weld[7] = {0};
        size_t nv = 7, nf = 5, closed = 0, overlap = 0, df = 0, dv = 0;
        int local = certify_source_charts(arena, v, &nv, f, &nf, 7,
                                          cube, NULL, weld, NULL, NULL,
                                          &closed, &overlap,
                                          &df, &dv) != 0 ||
                    nv != 3 || nf != 1 || closed != 1 || df != 4 || dv != 4;
        fail |= local;
        fprintf(stderr,
                "[selftest] source closed-bubble filter -> %s "
                "(closed=%zu faces=%zu verts=%zu)\n",
                local ? "FAIL" : "ok", closed, df, dv);
    }
    {
        float v[7*3] = {
            0,0,0, 0,1,0, 0,0,1,
            3,0,0, 4,0,0, 3,1,0, 3,0,1
        };
        int32_t f[5*3] = {
            0,1,2,
            3,5,4, 3,4,6, 4,5,6, 5,3,6
        };
        int16_t cube[7] = {0,0,0,1,1,1,1};
        uint8_t weld[7] = {0};
        size_t nv = 7, nf = 5, closed = 0, df = 0, dv = 0;
        int local = drop_closed_source_bubbles(
                        arena, v, &nv, f, &nf, cube, weld, NULL, NULL,
                        &closed, &df, &dv) != 0 ||
                    nv != 3 || nf != 1 || closed != 1 || df != 4 || dv != 4;
        fail |= local;
        fprintf(stderr,
                "[selftest] source pre-cut bubble filter -> %s "
                "(closed=%zu faces=%zu verts=%zu)\n",
                local ? "FAIL" : "ok", closed, df, dv);
    }
    Arena_dispose(&arena);
    return fail ? 1 : 0;
}

static int unassimilated_triangle_selftest(void)
{
    Arena_T arena = Arena_new();
    int32_t faces[9] = {
        0,1,2,
        3,4,5, 3,5,6
    };
    size_t nf = 0, comps = 0;
    size_t removed = cull_tiny_components(
        arena, faces, 3, 4, &nf, &comps);
    int fail = removed != 1 || comps != 1 || nf != 2 ||
               faces[0] != 3 || faces[1] != 4 || faces[2] != 5 ||
               faces[3] != 3 || faces[4] != 5 || faces[5] != 6;
    fprintf(stderr,
            "[selftest] unassimilated one-triangle chart cull -> %s\n",
            fail ? "FAIL" : "ok");
    Arena_dispose(&arena);
    return fail;
}

static int topology_safe_micro_weld_selftest(void)
{
    int failures = BoundaryArcWeld_selftest();
    failures += ChartLineage_selftest();
    failures += ChartBridgeForest_selftest();
    failures += ChartZipper_selftest();
    failures += StreamWeldFormat_selftest();
    failures += TopologyAudit_selftest();
    failures += DiskTopologyRepair_selftest();
    return failures;
}

static int grid_weld_selftest(void)
{
    Arena_T a = Arena_new();
    CubeList cl = {0, 0, 0};
    cubelist_push(a, &cl, "z04352_y03328_x02816");
    cubelist_push(a, &cl, "z04352_y02048_x01536");
    cubelist_push(a, &cl, "z04736_y03840_x03328");
    cubelist_push(a, &cl, "z04352_y03328_x03456");
    cubelist_push(a, &cl, "notacube");
    cubelist_filter_bbox(&cl, 4352, 4736, 3328, 3840, 2816, 3328);
    int fails = 0;
    if (cl.n != 2) {
        fprintf(stderr, "[selftest] survivors: got %zu expect 2 -> FAIL\n", cl.n);
        fails++;
    } else {
        if (strcmp(cl.ids[0], "z04352_y03328_x02816") != 0) {
            fprintf(stderr, "[selftest] survivor[0]=%s -> FAIL\n", cl.ids[0]); fails++;
        }
        if (strcmp(cl.ids[1], "z04736_y03840_x03328") != 0) {
            fprintf(stderr, "[selftest] survivor[1]=%s -> FAIL\n", cl.ids[1]); fails++;
        }
    }
    if (!fails) fprintf(stderr, "[selftest] subgrid-bbox-filter -> ok (2 survivors)\n");
    {
        int ok = gw_placed_pair_presence(1, 1) == 1 &&
                 gw_placed_pair_presence(0, 0) == 0 &&
                 gw_placed_pair_presence(1, 0) == -1 &&
                 gw_placed_pair_presence(0, 1) == -1;
        fprintf(stderr,
                "[selftest] placed mesh/facekeep pair policy -> %s\n",
                ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    {
        const char *json =
            "{\"axis_point_zyx\":[0,3405,2878],\"pitch\":9.5,"
            "\"calibration\":{\"spiral_b\":-9.5}}";
        double pitch = 0.0, umb_y = 0.0, umb_x = 0.0;
        int ok = gw_parse_placed_calibration(
            json, &pitch, &umb_y, &umb_x) == 0 &&
            fabs(pitch - 9.5) < 1.0e-12 &&
            fabs(umb_y - 3405.0) < 1.0e-12 &&
            fabs(umb_x - 2878.0) < 1.0e-12;
        fprintf(stderr,
                "[selftest] placed winding calibration handoff -> %s\n",
                ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    fails += source_chart_selftest();
    fails += unassimilated_triangle_selftest();
    fails += topology_safe_micro_weld_selftest();
    fails += BallPivot_winding_selftest();
    fails += IntersectionCleanup_selftest();
    Arena_dispose(&a);
    fprintf(stderr, "=== grid_weld selftest %s (%d failure%s) ===\n",
            fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}

typedef struct {
    int32_t a, b, c;
    int32_t orig0, orig1, orig2;
} SortedFace;

static void sort3i(int32_t v[3])
{
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { int32_t t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
}

static int cmp_sorted_face(const void *pa, const void *pb)
{
    const SortedFace *fa = (const SortedFace *)pa;
    const SortedFace *fb = (const SortedFace *)pb;
    if (fa->a != fb->a) return (fa->a < fb->a) ? -1 : 1;
    if (fa->b != fb->b) return (fa->b < fb->b) ? -1 : 1;
    if (fa->c != fb->c) return (fa->c < fb->c) ? -1 : 1;
    return 0;
}

typedef struct {
    int32_t src, dst;
} DHE;

static int cmp_dhe_undirected(const void *pa, const void *pb)
{
    const DHE *a = (const DHE *)pa;
    const DHE *b = (const DHE *)pb;
    int32_t a0 = (a->src < a->dst) ? a->src : a->dst;
    int32_t a1 = (a->src < a->dst) ? a->dst : a->src;
    int32_t b0 = (b->src < b->dst) ? b->src : b->dst;
    int32_t b1 = (b->src < b->dst) ? b->dst : b->src;
    if (a0 != b0) return (a0 < b0) ? -1 : 1;
    if (a1 != b1) return (a1 < b1) ? -1 : 1;
    return 0;
}

static int cmp_int64(const void *pa, const void *pb)
{
    int64_t a = *(const int64_t *)pa;
    int64_t b = *(const int64_t *)pb;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

typedef struct {
    size_t unpaired;
    size_t non_manifold;
    size_t same_dir_pairs;
    size_t manifold_pairs;
} ManifoldStats;

static ManifoldStats manifold_audit(Arena_T arena,
                                    const int32_t *faces, size_t nf,
                                    const float *verts_for_diag,
                                    FILE *diag_fp)
{
    ManifoldStats s = {0, 0, 0, 0};
    if (nf == 0) return s;
    size_t hn = nf * 3;
    DHE *he = (DHE *)ARENA_ALLOC(arena, hn * sizeof(DHE));
    for (size_t f = 0; f < nf; f++) {
        int32_t v0 = faces[f * 3 + 0];
        int32_t v1 = faces[f * 3 + 1];
        int32_t v2 = faces[f * 3 + 2];
        he[f * 3 + 0].src = v0; he[f * 3 + 0].dst = v1;
        he[f * 3 + 1].src = v1; he[f * 3 + 1].dst = v2;
        he[f * 3 + 2].src = v2; he[f * 3 + 2].dst = v0;
    }
    qsort(he, hn, sizeof(DHE), cmp_dhe_undirected);
    size_t diag_vidx = 1;
    if (diag_fp) {
        fprintf(diag_fp, "# manifold-audit problem edges\n");
        fprintf(diag_fp, "# unpaired = red, non_manifold = magenta, same_dir = yellow\n");
    }
    size_t i = 0;
    while (i < hn) {
        size_t j = i + 1;
        int32_t a0 = (he[i].src < he[i].dst) ? he[i].src : he[i].dst;
        int32_t a1 = (he[i].src < he[i].dst) ? he[i].dst : he[i].src;
        while (j < hn) {
            int32_t b0 = (he[j].src < he[j].dst) ? he[j].src : he[j].dst;
            int32_t b1 = (he[j].src < he[j].dst) ? he[j].dst : he[j].src;
            if (b0 != a0 || b1 != a1) break;
            j++;
        }
        size_t run = j - i;
        int kind = -1;
        if (run == 1) { s.unpaired++; kind = 0; }
        else if (run > 2) { s.non_manifold++; kind = 1; }
        else {
            int dir0 = (he[i].src < he[i].dst) ? 0 : 1;
            int dir1 = (he[i + 1].src < he[i + 1].dst) ? 0 : 1;
            if (dir0 == dir1) { s.same_dir_pairs++; kind = 2; }
            else s.manifold_pairs++;
        }
        if (diag_fp && verts_for_diag && (kind == 1 || kind == 2)) {
            float r = (kind == 1) ? 1.0f : 1.0f;
            float g = (kind == 1) ? 0.0f : 1.0f;
            float b = (kind == 1) ? 1.0f : 0.0f;
            int32_t p0 = a0, p1 = a1;
            fprintf(diag_fp, "v %.4f %.4f %.4f %.3f %.3f %.3f\n",
                (double)verts_for_diag[p0 * 3 + 0],
                (double)verts_for_diag[p0 * 3 + 1],
                (double)verts_for_diag[p0 * 3 + 2],
                (double)r, (double)g, (double)b);
            fprintf(diag_fp, "v %.4f %.4f %.4f %.3f %.3f %.3f\n",
                (double)verts_for_diag[p1 * 3 + 0],
                (double)verts_for_diag[p1 * 3 + 1],
                (double)verts_for_diag[p1 * 3 + 2],
                (double)r, (double)g, (double)b);
            fprintf(diag_fp, "l %zu %zu\n", diag_vidx, diag_vidx + 1);
            diag_vidx += 2;
        }
        i = j;
    }
    return s;
}

static void emit_nonmanifold_neighborhood(Arena_T arena,
                                          const int32_t *faces, size_t nf,
                                          const float *verts, size_t nv,
                                          const char *out_path)
{
    if (nf == 0 || nv == 0) return;

    typedef struct { int32_t a0, a1; int32_t face; } DHE2;
    size_t hn = nf * 3;
    DHE2 *he = (DHE2 *)ARENA_ALLOC(arena, hn * sizeof(DHE2));
    for (size_t f = 0; f < nf; f++) {
        int32_t v[3] = { faces[f*3+0], faces[f*3+1], faces[f*3+2] };
        for (int e = 0; e < 3; e++) {
            int32_t s = v[e], d = v[(e+1)%3];
            he[f*3 + e].a0 = (s < d) ? s : d;
            he[f*3 + e].a1 = (s < d) ? d : s;
            he[f*3 + e].face = (int32_t)f;
        }
    }
    qsort(he, hn, sizeof(DHE2), cmp_dhe_undirected);

    uint8_t *nm_face = (uint8_t *)ARENA_CALLOC(arena, nf, 1L);
    size_t i = 0;
    while (i < hn) {
        size_t j = i + 1;
        while (j < hn && he[j].a0 == he[i].a0 && he[j].a1 == he[i].a1) j++;
        size_t run = j - i;
        if (run > 2) {
            for (size_t k = i; k < j; k++) nm_face[he[k].face] = 1;
        }
        i = j;
    }

    uint8_t *vert_use = (uint8_t *)ARENA_CALLOC(arena, nv, 1L);
    for (size_t f = 0; f < nf; f++) {
        if (!nm_face[f]) continue;
        for (int k = 0; k < 3; k++) vert_use[faces[f*3+k]] = 1;
    }
    uint8_t *ring_face = (uint8_t *)ARENA_CALLOC(arena, nf, 1L);
    for (size_t f = 0; f < nf; f++) {
        if (nm_face[f]) continue;
        for (int k = 0; k < 3; k++) {
            if (vert_use[faces[f*3+k]] == 1) { ring_face[f] = 1; break; }
        }
    }
    for (size_t f = 0; f < nf; f++) {
        if (!ring_face[f]) continue;
        for (int k = 0; k < 3; k++) {
            int32_t v = faces[f*3+k];
            if (vert_use[v] == 0) vert_use[v] = 2;
        }
    }

    int32_t *remap = (int32_t *)ARENA_ALLOC(arena, nv * sizeof(int32_t));
    size_t out_nv = 0;
    for (size_t v = 0; v < nv; v++) {
        if (vert_use[v]) { remap[v] = (int32_t)(out_nv++); }
        else             { remap[v] = -1; }
    }
    if (out_nv == 0) return;

    FILE *fp = fopen(out_path, "w");
    if (!fp) return;
    fprintf(fp, "# non-manifold neighbourhood\n");
    fprintf(fp, "# magenta = faces on a non-manifold edge\n");
    fprintf(fp, "# grey    = 1-ring context faces\n");

    for (size_t v = 0; v < nv; v++) {
        if (!vert_use[v]) continue;
        float r, g, b;
        if (vert_use[v] == 1) { r = 1.0f; g = 0.0f; b = 1.0f; }
        else                  { r = 0.5f; g = 0.5f; b = 0.5f; }
        fprintf(fp, "v %.4f %.4f %.4f %.3f %.3f %.3f\n",
                (double)verts[v*3+0], (double)verts[v*3+1], (double)verts[v*3+2],
                (double)r, (double)g, (double)b);
    }
    for (size_t f = 0; f < nf; f++) {
        if (!nm_face[f] && !ring_face[f]) continue;
        int32_t a = remap[faces[f*3+0]];
        int32_t b2 = remap[faces[f*3+1]];
        int32_t c = remap[faces[f*3+2]];
        fprintf(fp, "f %d %d %d\n", a + 1, b2 + 1, c + 1);
    }
    fclose(fp);
}

static void emit_embedded_conflict_neighborhood(
        Arena_T arena,
        const int32_t *faces, size_t nf,
        const float *verts, size_t nv,
        const size_t *face_conflict_degree,
        const char *out_path)
{
    if (nf == 0 || nv == 0 || !face_conflict_degree) return;

    uint8_t *conflict_face = (uint8_t *)ARENA_CALLOC(arena, nf, 1L);
    uint8_t *ring_face = (uint8_t *)ARENA_CALLOC(arena, nf, 1L);
    uint8_t *vert_use = (uint8_t *)ARENA_CALLOC(arena, nv, 1L);
    int32_t *remap = (int32_t *)ARENA_ALLOC(arena, nv * sizeof(int32_t));
    size_t conflict_faces = 0, out_nv = 0;

    for (size_t f = 0; f < nf; f++) {
        if (face_conflict_degree[f] == 0) continue;
        conflict_face[f] = 1;
        conflict_faces++;
        for (int k = 0; k < 3; k++)
            vert_use[faces[f*3+k]] = 1;
    }
    if (conflict_faces == 0) return;

    for (size_t f = 0; f < nf; f++) {
        if (conflict_face[f]) continue;
        for (int k = 0; k < 3; k++) {
            if (vert_use[faces[f*3+k]] == 1) {
                ring_face[f] = 1;
                break;
            }
        }
    }
    for (size_t f = 0; f < nf; f++) {
        if (!ring_face[f]) continue;
        for (int k = 0; k < 3; k++) {
            int32_t v = faces[f*3+k];
            if (vert_use[v] == 0) vert_use[v] = 2;
        }
    }
    for (size_t v = 0; v < nv; v++) {
        if (vert_use[v]) remap[v] = (int32_t)(out_nv++);
        else remap[v] = -1;
    }

    FILE *fp = fopen(out_path, "w");
    if (!fp) return;
    fprintf(fp, "# exact embedded-geometry conflict neighbourhood\n");
    fprintf(fp, "# red  = face participating in an exact conflict\n");
    fprintf(fp, "# grey = vertex-adjacent 1-ring context\n");
    fprintf(fp, "# conflict_faces=%zu\n", conflict_faces);
    for (size_t v = 0; v < nv; v++) {
        if (!vert_use[v]) continue;
        double r = vert_use[v] == 1 ? 1.0 : 0.5;
        double g = vert_use[v] == 1 ? 0.0 : 0.5;
        double b = vert_use[v] == 1 ? 0.0 : 0.5;
        fprintf(fp, "v %.4f %.4f %.4f %.3f %.3f %.3f\n",
                (double)verts[v*3+0], (double)verts[v*3+1],
                (double)verts[v*3+2], r, g, b);
    }
    for (size_t f = 0; f < nf; f++) {
        if (!conflict_face[f] && !ring_face[f]) continue;
        fprintf(fp, "f %d %d %d\n",
                remap[faces[f*3+0]] + 1,
                remap[faces[f*3+1]] + 1,
                remap[faces[f*3+2]] + 1);
    }
    fclose(fp);
}

typedef struct {
    int32_t v0, v1;
    int32_t face;
} EdgeEntry;

static int cmp_edge_undirected(const void *pa, const void *pb)
{
    const EdgeEntry *a = (const EdgeEntry *)pa;
    const EdgeEntry *b = (const EdgeEntry *)pb;
    int32_t au = (a->v0 < a->v1) ? a->v0 : a->v1;
    int32_t av = (a->v0 < a->v1) ? a->v1 : a->v0;
    int32_t bu = (b->v0 < b->v1) ? b->v0 : b->v1;
    int32_t bv = (b->v0 < b->v1) ? b->v1 : b->v0;
    if (au != bu) return (au < bu) ? -1 : 1;
    if (av != bv) return (av < bv) ? -1 : 1;
    return 0;
}

static size_t repair_winding(Arena_T arena,
                             int32_t *faces, size_t nf,
                             size_t *out_components)
{
    if (nf == 0) {
        *out_components = 0;
        return 0;
    }
    size_t hn = nf * 3;
    EdgeEntry *edges = (EdgeEntry *)ARENA_ALLOC(arena,
                          hn * sizeof(EdgeEntry));
    for (size_t f = 0; f < nf; f++) {
        int32_t v0 = faces[f * 3 + 0];
        int32_t v1 = faces[f * 3 + 1];
        int32_t v2 = faces[f * 3 + 2];
        edges[f * 3 + 0].v0 = v0; edges[f * 3 + 0].v1 = v1; edges[f * 3 + 0].face = (int32_t)f;
        edges[f * 3 + 1].v0 = v1; edges[f * 3 + 1].v1 = v2; edges[f * 3 + 1].face = (int32_t)f;
        edges[f * 3 + 2].v0 = v2; edges[f * 3 + 2].v1 = v0; edges[f * 3 + 2].face = (int32_t)f;
    }
    qsort(edges, hn, sizeof(EdgeEntry), cmp_edge_undirected);

    int32_t *adj = (int32_t *)ARENA_ALLOC(arena,
                       nf * 3L * sizeof(int32_t));
    int32_t *adj_edge_u = (int32_t *)ARENA_ALLOC(arena,
                              nf * 3L * sizeof(int32_t));
    int32_t *adj_edge_v = (int32_t *)ARENA_ALLOC(arena,
                              nf * 3L * sizeof(int32_t));
    int32_t *adj_count = (int32_t *)ARENA_CALLOC(arena,
                             nf, sizeof(int32_t));
    for (size_t i = 0; i < hn; ) {
        size_t j = i + 1;
        int32_t au = (edges[i].v0 < edges[i].v1) ? edges[i].v0 : edges[i].v1;
        int32_t av = (edges[i].v0 < edges[i].v1) ? edges[i].v1 : edges[i].v0;
        while (j < hn) {
            int32_t bu = (edges[j].v0 < edges[j].v1) ? edges[j].v0 : edges[j].v1;
            int32_t bv = (edges[j].v0 < edges[j].v1) ? edges[j].v1 : edges[j].v0;
            if (bu != au || bv != av) break;
            j++;
        }
        size_t run = j - i;
        if (run == 2) {
            int32_t fA = edges[i].face;
            int32_t fB = edges[i + 1].face;
            if (adj_count[fA] < 3 && adj_count[fB] < 3) {
                adj[fA * 3 + adj_count[fA]] = fB;
                adj_edge_u[fA * 3 + adj_count[fA]] = edges[i].v0;
                adj_edge_v[fA * 3 + adj_count[fA]] = edges[i].v1;
                adj_count[fA]++;
                adj[fB * 3 + adj_count[fB]] = fA;
                adj_edge_u[fB * 3 + adj_count[fB]] = edges[i + 1].v0;
                adj_edge_v[fB * 3 + adj_count[fB]] = edges[i + 1].v1;
                adj_count[fB]++;
            }
        }
        i = j;
    }

    uint8_t *visited = (uint8_t *)ARENA_CALLOC(arena, nf, 1L);
    int32_t *queue = (int32_t *)ARENA_ALLOC(arena,
                         nf * sizeof(int32_t));
    size_t n_flipped = 0;
    size_t n_components = 0;

    for (size_t seed = 0; seed < nf; seed++) {
        if (visited[seed]) continue;
        n_components++;
        size_t qh = 0, qt = 0;
        queue[qt++] = (int32_t)seed;
        visited[seed] = 1;
        while (qh < qt) {
            int32_t fcur = queue[qh++];
            int32_t cv0 = faces[fcur * 3 + 0];
            int32_t cv1 = faces[fcur * 3 + 1];
            int32_t cv2 = faces[fcur * 3 + 2];
            for (int k = 0; k < adj_count[fcur]; k++) {
                int32_t fnext = adj[fcur * 3 + k];
                if (visited[fnext]) continue;
                int32_t ea = adj_edge_u[fcur * 3 + k];
                int32_t eb = adj_edge_v[fcur * 3 + k];
                int parent_ab = ((cv0 == ea && cv1 == eb) ||
                                 (cv1 == ea && cv2 == eb) ||
                                 (cv2 == ea && cv0 == eb));
                int32_t nv0 = faces[fnext * 3 + 0];
                int32_t nv1 = faces[fnext * 3 + 1];
                int32_t nv2 = faces[fnext * 3 + 2];
                int neigh_ab = ((nv0 == ea && nv1 == eb) ||
                                (nv1 == ea && nv2 == eb) ||
                                (nv2 == ea && nv0 == eb));
                if (parent_ab == neigh_ab) {
                    int32_t tmp = faces[fnext * 3 + 1];
                    faces[fnext * 3 + 1] = faces[fnext * 3 + 2];
                    faces[fnext * 3 + 2] = tmp;
                    n_flipped++;
                }
                visited[fnext] = 1;
                queue[qt++] = fnext;
            }
        }
    }

    *out_components = n_components;
    return n_flipped;
}

static size_t cull_tiny_components(Arena_T arena, int32_t *faces, size_t nf,
                                   size_t min_verts, size_t *out_nf,
                                   size_t *out_removed_comps) {
    *out_removed_comps = 0;
    if (nf == 0) { *out_nf = 0; return 0; }

    size_t hn = nf * 3;
    EdgeEntry *edges = (EdgeEntry *)ARENA_ALLOC(arena,
                          hn * sizeof(EdgeEntry));
    for (size_t f = 0; f < nf; f++) {
        int32_t v0 = faces[f*3+0], v1 = faces[f*3+1], v2 = faces[f*3+2];
        edges[f*3+0].v0=v0; edges[f*3+0].v1=v1; edges[f*3+0].face=(int32_t)f;
        edges[f*3+1].v0=v1; edges[f*3+1].v1=v2; edges[f*3+1].face=(int32_t)f;
        edges[f*3+2].v0=v2; edges[f*3+2].v1=v0; edges[f*3+2].face=(int32_t)f;
    }
    qsort(edges, hn, sizeof(EdgeEntry), cmp_edge_undirected);

    int32_t *adj = (int32_t *)ARENA_ALLOC(arena,
                       nf * 3L * sizeof(int32_t));
    int32_t *adj_count = (int32_t *)ARENA_CALLOC(arena,
                             nf, sizeof(int32_t));
    for (size_t i = 0; i < hn; ) {
        size_t j = i + 1;
        int32_t au = (edges[i].v0 < edges[i].v1) ? edges[i].v0 : edges[i].v1;
        int32_t av = (edges[i].v0 < edges[i].v1) ? edges[i].v1 : edges[i].v0;
        while (j < hn) {
            int32_t bu = (edges[j].v0 < edges[j].v1) ? edges[j].v0 : edges[j].v1;
            int32_t bv = (edges[j].v0 < edges[j].v1) ? edges[j].v1 : edges[j].v0;
            if (bu != au || bv != av) break;
            j++;
        }
        if (j - i == 2) {
            int32_t fA = edges[i].face, fB = edges[i + 1].face;
            if (adj_count[fA] < 3 && adj_count[fB] < 3) {
                adj[fA * 3 + adj_count[fA]++] = fB;
                adj[fB * 3 + adj_count[fB]++] = fA;
            }
        }
        i = j;
    }

    int32_t *comp = (int32_t *)ARENA_ALLOC(arena,
                        nf * sizeof(int32_t));
    for (size_t f = 0; f < nf; f++) { comp[f] = -1; }
    int32_t *queue = (int32_t *)ARENA_ALLOC(arena,
                         nf * sizeof(int32_t));
    size_t max_v = 0;
    for (size_t k = 0; k < nf * 3; k++) {
        if ((size_t)faces[k] + 1 > max_v) { max_v = (size_t)faces[k] + 1; }
    }
    int32_t *vseen = (int32_t *)ARENA_ALLOC(arena,
                         max_v * sizeof(int32_t));
    for (size_t v = 0; v < max_v; v++) { vseen[v] = -1; }
    size_t *comp_verts = (size_t *)ARENA_ALLOC(arena,
                             nf * sizeof(size_t));

    size_t n_comp = 0;
    for (size_t seed = 0; seed < nf; seed++) {
        if (comp[seed] != -1) { continue; }
        int32_t cid = (int32_t)n_comp;
        size_t vcount = 0, qh = 0, qt = 0;
        queue[qt++] = (int32_t)seed; comp[seed] = cid;
        while (qh < qt) {
            int32_t fcur = queue[qh++];
            for (int e = 0; e < 3; e++) {
                int32_t v = faces[fcur*3+e];
                if (vseen[v] != cid) { vseen[v] = cid; vcount++; }
            }
            for (int k = 0; k < adj_count[fcur]; k++) {
                int32_t fnext = adj[fcur * 3 + k];
                if (comp[fnext] != -1) { continue; }
                comp[fnext] = cid; queue[qt++] = fnext;
            }
        }
        comp_verts[n_comp++] = vcount;
    }

    size_t w = 0, removed_comps = 0;
    for (size_t c = 0; c < n_comp; c++) {
        if (comp_verts[c] < min_verts) { removed_comps++; }
    }
    for (size_t f = 0; f < nf; f++) {
        if (comp_verts[comp[f]] < min_verts) { continue; }
        faces[w*3+0] = faces[f*3+0];
        faces[w*3+1] = faces[f*3+1];
        faces[w*3+2] = faces[f*3+2];
        w++;
    }
    *out_nf = w;
    *out_removed_comps = removed_comps;
    return nf - w;
}

static void set_env_if_unset(const char *name, const char *val)
{
    if (getenv(name)) return;
#ifdef _MSC_VER
    _putenv_s(name, val);
#else
    setenv(name, val, 1);
#endif
}

static int gw_parse_placed_calibration(const char *json,
                                       double *pitch,
                                       double *umb_y, double *umb_x)
{
    const char *key = NULL, *colon = NULL, *cursor = NULL;
    char *end = NULL;
    double z = 0.0;
    if (json == NULL || pitch == NULL || umb_y == NULL || umb_x == NULL)
        return -1;
    key = strstr(json, "\"pitch\"");
    if (key == NULL || (colon = strchr(key, ':')) == NULL) return -1;
    *pitch = strtod(colon + 1, &end);
    if (end == colon + 1 || !isfinite(*pitch) ||
        *pitch <= 0.0 || *pitch > 100.0)
        return -1;

    key = strstr(json, "\"axis_point_zyx\"");
    if (key == NULL || (colon = strchr(key, ':')) == NULL ||
        (cursor = strchr(colon, '[')) == NULL)
        return -1;
    z = strtod(cursor + 1, &end);
    if (end == cursor + 1 || !isfinite(z)) return -1;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
           *cursor == '\n')
        cursor++;
    if (*cursor++ != ',') return -1;
    *umb_y = strtod(cursor, &end);
    if (end == cursor || !isfinite(*umb_y)) return -1;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
           *cursor == '\n')
        cursor++;
    if (*cursor++ != ',') return -1;
    *umb_x = strtod(cursor, &end);
    if (end == cursor || !isfinite(*umb_x)) return -1;
    return 0;
}

static int gw_arm_placed_calibration(const char *placed_dir)
{
    char path[1024], *buffer = NULL;
    FILE *f = NULL;
    long bytes = 0;
    double pitch = 0.0, umb_y = 0.0, umb_x = 0.0;
    char text[64];
    int need_pitch = getenv("SEAM_WRAP_PITCH") == NULL;
    int need_y = getenv("SEAM_UMBILICUS_Y") == NULL;
    int need_x = getenv("SEAM_UMBILICUS_X") == NULL;
    if (!need_pitch && !need_y && !need_x) return 0;
    if (snprintf(path, sizeof(path), "%s/placed_index.json", placed_dir) < 0)
        return -1;
    f = fopen(path, "rb");
    if (f == NULL || fseek(f, 0, SEEK_END) != 0 ||
        (bytes = ftell(f)) <= 0 || bytes > 16 * 1024 * 1024L ||
        fseek(f, 0, SEEK_SET) != 0)
        goto fail;
    buffer = (char *)malloc((size_t)bytes + 1);
    if (buffer == NULL || fread(buffer, 1, (size_t)bytes, f) != (size_t)bytes)
        goto fail;
    fclose(f); f = NULL;
    buffer[bytes] = '\0';
    if (gw_parse_placed_calibration(
            buffer, &pitch, &umb_y, &umb_x) != 0)
        goto fail;
    if (need_pitch) {
        snprintf(text, sizeof(text), "%.17g", pitch);
        set_env_if_unset("SEAM_WRAP_PITCH", text);
    }
    if (need_y) {
        snprintf(text, sizeof(text), "%.17g", umb_y);
        set_env_if_unset("SEAM_UMBILICUS_Y", text);
    }
    if (need_x) {
        snprintf(text, sizeof(text), "%.17g", umb_x);
        set_env_if_unset("SEAM_UMBILICUS_X", text);
    }
    fprintf(stderr,
            "grid_weld: armed placed winding calibration pitch=%.6g "
            "axis_yx=(%.9g,%.9g) from %s\n",
            atof(getenv("SEAM_WRAP_PITCH")),
            atof(getenv("SEAM_UMBILICUS_Y")),
            atof(getenv("SEAM_UMBILICUS_X")), path);
    free(buffer);
    return 0;
fail:
    if (f != NULL) fclose(f);
    free(buffer);
    return -1;
}

typedef struct {
    uint64_t key;
    int32_t chart;
    int8_t dir;
} GwStreamBoundaryEdge;

static uint64_t gw_stream_edge_key(int32_t a, int32_t b)
{
    uint32_t lo=(uint32_t)(a<b?a:b),hi=(uint32_t)(a<b?b:a);
    return ((uint64_t)lo<<32)|(uint64_t)hi;
}

static int gw_stream_boundary_cmp(const void *pa,const void *pb)
{
    const GwStreamBoundaryEdge *a=(const GwStreamBoundaryEdge*)pa;
    const GwStreamBoundaryEdge *b=(const GwStreamBoundaryEdge*)pb;
    return a->key<b->key?-1:a->key>b->key?1:0;
}

static const GwStreamBoundaryEdge *gw_stream_boundary_find(
    const GwStreamBoundaryEdge *edge,size_t n,uint64_t key)
{
    size_t lo=0,hi=n;
    while(lo<hi){
        size_t mid=lo+(hi-lo)/2;
        if(edge[mid].key<key)lo=mid+1;else hi=mid;
    }
    return lo<n&&edge[lo].key==key?&edge[lo]:NULL;
}

static int gw_stream_prepare_transactions(
    Arena_T arena,const int32_t *vertex_chart,size_t nv,size_t n_charts,
    int32_t *faces,size_t source_nf,size_t *nf,
    ChartZipperTransaction *tx,size_t *ntx,size_t *nbridge)
{
    GwStreamBoundaryEdge *edge;
    size_t ne,w=0,write=source_nf,kept=0,rejected=0;
    if(!arena||!vertex_chart||!faces||!nf||!tx||!ntx||!nbridge||
       source_nf>*nf||source_nf>SIZE_MAX/3)return -1;
    ne=source_nf*3;
    edge=(GwStreamBoundaryEdge*)ARENA_ALLOC(
        arena,(ne?ne:1)*sizeof(*edge));
    for(size_t f=0;f<source_nf;f++){
        const int32_t *tri=&faces[f*3];
        int32_t c;
        for(int k=0;k<3;k++)
            if(tri[k]<0||(size_t)tri[k]>=nv)return -1;
        c=vertex_chart[tri[0]];
        if(c<0||(size_t)c>=n_charts||
           vertex_chart[tri[1]]!=c||vertex_chart[tri[2]]!=c)return -1;
        for(int k=0;k<3;k++){
            int32_t a=tri[k],b=tri[(k+1)%3];
            edge[f*3+(size_t)k].key=gw_stream_edge_key(a,b);
            edge[f*3+(size_t)k].chart=c;
            edge[f*3+(size_t)k].dir=(int8_t)(a<b?1:-1);
        }
    }
    qsort(edge,ne,sizeof(*edge),gw_stream_boundary_cmp);
    for(size_t i=0;i<ne;){
        size_t j=i+1;int sum=edge[i].dir;
        while(j<ne&&edge[j].key==edge[i].key){sum+=edge[j].dir;j++;}
        if(j-i==1)edge[w++]=edge[i];
        else if(j-i!=2||sum!=0)return -1;
        i=j;
    }
    for(size_t t=0;t<*ntx;t++){
        ChartZipperTransaction cur=tx[t];
        int8_t req_a=0,req_b=0;
        int bad=0;
        if(cur.face_first<source_nf||cur.face_first>*nf||
           cur.face_count>*nf-cur.face_first){bad=1;}
        for(size_t f=cur.face_first;!bad&&f<cur.face_first+cur.face_count;f++){
            const int32_t *tri=&faces[f*3];
            for(int k=0;k<3;k++){
                int32_t a=tri[k],b=tri[(k+1)%3];
                const GwStreamBoundaryEdge *se;
                int8_t req,*dst;
                if(a<0||b<0||(size_t)a>=nv||(size_t)b>=nv){bad=1;break;}
                se=gw_stream_boundary_find(edge,w,gw_stream_edge_key(a,b));
                if(!se)continue;
                if(se->chart==cur.chart_a)dst=&req_a;
                else if(se->chart==cur.chart_b)dst=&req_b;
                else{bad=1;break;}
                req=(int8_t)(((a<b?1:-1)==-se->dir)?1:-1);
                if(*dst==0)*dst=req;
                else if(*dst!=req){bad=1;break;}
            }
        }
        if(bad||req_a==0||req_b==0){rejected++;continue;}
        if(write!=cur.face_first)
            memmove(&faces[write*3],&faces[cur.face_first*3],
                    cur.face_count*3*sizeof(*faces));
        cur.face_first=write;cur.orient_a=req_a;cur.orient_b=req_b;
        tx[kept++]=cur;write+=cur.face_count;
    }
    if(rejected)
        fprintf(stderr,"stream seam: rejected %zu winding-inconsistent "
                       "transaction(s)\n",rejected);
    *ntx=kept;*nf=write;*nbridge=write-source_nf;
    return 0;
}

typedef struct {
    const int32_t *face_transaction;
    uint8_t *invalid;
    StreamWeldConflict *conflicts;
    size_t nconflicts,capacity;
    int source_conflict;
    int allocation_failed;
} GwStreamConflictCollector;

static int gw_stream_collect_conflict(size_t face_a,size_t face_b,
                                      int hit_kind,void *context)
{
    GwStreamConflictCollector *c=(GwStreamConflictCollector*)context;
    int32_t ta=c->face_transaction[face_a];
    int32_t tb=c->face_transaction[face_b];
    (void)hit_kind;
    if(ta<0&&tb<0){c->source_conflict=1;return -1;}
    if(ta<0||tb<0){
        c->invalid[(size_t)(ta<0?tb:ta)]=1;
        return 0;
    }
    if(ta==tb){c->invalid[(size_t)ta]=1;return 0;}
    if(c->nconflicts==c->capacity){
        size_t cap=c->capacity?c->capacity*2:64;
        StreamWeldConflict *grown;
        if(cap<c->capacity||cap>SIZE_MAX/sizeof(*grown)){
            c->allocation_failed=1;return -1;
        }
        grown=(StreamWeldConflict*)realloc(c->conflicts,cap*sizeof(*grown));
        if(!grown){c->allocation_failed=1;return -1;}
        c->conflicts=grown;c->capacity=cap;
    }
    if(ta>tb){int32_t q=ta;ta=tb;tb=q;}
    c->conflicts[c->nconflicts++]=(StreamWeldConflict){
        (uint32_t)ta,(uint32_t)tb,STREAM_WELD_CONFLICT_GEOMETRY};
    return 0;
}

static int gw_stream_conflict_cmp(const void *pa,const void *pb)
{
    const StreamWeldConflict *a=(const StreamWeldConflict*)pa;
    const StreamWeldConflict *b=(const StreamWeldConflict*)pb;
    if(a->transaction_a!=b->transaction_a)
        return a->transaction_a<b->transaction_a?-1:1;
    if(a->transaction_b!=b->transaction_b)
        return a->transaction_b<b->transaction_b?-1:1;
    return 0;
}

static int gw_stream_mark_physical_transactions(
    const int32_t *candidate_faces,ChartZipperTransaction *candidate,
    size_t ncandidate,const int32_t *physical_faces,
    const ChartZipperTransaction *physical,size_t nphysical)
{
    size_t marked=0;
    if((ncandidate&&!candidate)||(nphysical&&!physical)||
       !candidate_faces||!physical_faces)return -1;
    for(size_t t=0;t<nphysical;t++){
        const ChartZipperTransaction *want=&physical[t];int found=0;
        for(size_t c=0;c<ncandidate;c++){
            ChartZipperTransaction *have=&candidate[c];
            if(have->flags&CHART_ZIPPER_TRANSACTION_PHYSICAL)continue;
            if(have->chart_a!=want->chart_a||have->chart_b!=want->chart_b||
               have->face_count!=want->face_count||
               have->full_coverage!=want->full_coverage||
               have->span_coverage!=want->span_coverage||
               have->mean_gap!=want->mean_gap||have->support!=want->support||
               have->score!=want->score||
               memcmp(&candidate_faces[have->face_first*3],
                      &physical_faces[want->face_first*3],
                      have->face_count*3*sizeof(*candidate_faces))!=0)
                continue;
            have->flags|=CHART_ZIPPER_TRANSACTION_PHYSICAL;
            marked++;found=1;break;
        }
        if(!found)return -1;
    }
    return marked==nphysical?0:-1;
}

static int gw_stream_classify_transactions(
    const float *verts,size_t nv,int32_t *faces,size_t source_nf,size_t *nf,
    ChartZipperTransaction *tx,size_t *ntx,size_t *nbridge,
    const ChartZipperParams *zp,StreamWeldConflict **out_conflicts,
    size_t *out_nconflicts)
{
    GwStreamConflictCollector c={0};
    IntersectionCleanupParams ip;IntersectionCleanupStats is;
    int32_t *face_tx=NULL,*old_to_new=NULL;
    uint8_t *invalid=NULL;
    size_t expected=source_nf,write=source_nf,kept=0,outn=0,rejected=0;
    int rc=-1;
    if(!verts||!faces||!nf||!tx||!ntx||!nbridge||!zp||
       !out_conflicts||!out_nconflicts||source_nf>*nf||*ntx>UINT32_MAX)
        return -1;
    *out_conflicts=NULL;*out_nconflicts=0;
    face_tx=(int32_t*)malloc((*nf?*nf:1)*sizeof(*face_tx));
    invalid=(uint8_t*)calloc(*ntx?*ntx:1,1);
    old_to_new=(int32_t*)malloc((*ntx?*ntx:1)*sizeof(*old_to_new));
    if(!face_tx||!invalid||!old_to_new)goto done;
    for(size_t f=0;f<*nf;f++)face_tx[f]=-1;
    for(size_t t=0;t<*ntx;t++){
        if(tx[t].face_first!=expected||tx[t].face_count>*nf-expected)
            goto done;
        for(size_t f=expected;f<expected+tx[t].face_count;f++)
            face_tx[f]=(int32_t)t;
        expected+=tx[t].face_count;
    }
    if(expected!=*nf)goto done;
    c.face_transaction=face_tx;c.invalid=invalid;
    IntersectionCleanup_default_params(&ip);
    ip.gap_max=(double)zp->conflict_gap;
    ip.parallel_angle_deg=(double)zp->conflict_parallel_angle_deg;
    ip.include_hinges=1;
    if(IntersectionCleanup_audit_visit(
            verts,nv,faces,*nf,NULL,&ip,NULL,gw_stream_collect_conflict,
            &c,&is)!=0||c.source_conflict||c.allocation_failed)goto done;
    for(size_t t=0;t<*ntx;t++){
        old_to_new[t]=-1;
        if(invalid[t]){rejected++;continue;}
        if(write!=tx[t].face_first)
            memmove(&faces[write*3],&faces[tx[t].face_first*3],
                    tx[t].face_count*3*sizeof(*faces));
        old_to_new[t]=(int32_t)kept;
        tx[kept]=tx[t];tx[kept].face_first=write;
        write+=tx[kept].face_count;kept++;
    }
    for(size_t i=0;i<c.nconflicts;i++){
        int32_t a=old_to_new[c.conflicts[i].transaction_a];
        int32_t b=old_to_new[c.conflicts[i].transaction_b];
        if(a<0||b<0||a==b)continue;
        if(a>b){int32_t q=a;a=b;b=q;}
        c.conflicts[outn]=(StreamWeldConflict){
            (uint32_t)a,(uint32_t)b,c.conflicts[i].reason_mask};
        outn++;
    }
    if(outn>1)
        qsort(c.conflicts,outn,sizeof(*c.conflicts),gw_stream_conflict_cmp);
    if(outn>0){
        size_t w=1;
        for(size_t i=1;i<outn;i++){
            StreamWeldConflict *last=&c.conflicts[w-1];
            if(last->transaction_a==c.conflicts[i].transaction_a&&
               last->transaction_b==c.conflicts[i].transaction_b)
                last->reason_mask|=c.conflicts[i].reason_mask;
            else c.conflicts[w++]=c.conflicts[i];
        }
        outn=w;
    }
    if(rejected)fprintf(stderr,
        "stream seam: rejected %zu source/self-conflicting transaction(s)\n",
        rejected);
    *ntx=kept;*nf=write;*nbridge=write-source_nf;
    *out_conflicts=c.conflicts;*out_nconflicts=outn;c.conflicts=NULL;
    rc=0;
done:
    free(c.conflicts);free(old_to_new);free(invalid);free(face_tx);
    return rc;
}

static int gw_write_chart_matches(
    const char *path,const StreamWeldShard *a,const StreamWeldShard *b,
    const ChartZipperMatch *match,size_t nmatch,
    const ChartZipperMatchSample *sample,size_t nsample)
{
    static const char magic[8]={'G','C','M','A','T','C','2','\n'};
    FILE *f=NULL;uint32_t version=2,flags=0;
    uint64_t nva,nvb,nca,ncb,nmatch64,nsample64;
    char ida[48]={0},idb[48]={0};size_t written_samples=0;int ok=0;
    if(!path||!a||!b||(nmatch&&!match)||(nsample&&!sample)||
       strlen(a->cube_id)>=sizeof(ida)||strlen(b->cube_id)>=sizeof(idb))
        return -1;
    memcpy(ida,a->cube_id,strlen(a->cube_id));
    memcpy(idb,b->cube_id,strlen(b->cube_id));
    nva=(uint64_t)a->nv;nvb=(uint64_t)b->nv;
    nca=(uint64_t)a->n_charts;ncb=(uint64_t)b->n_charts;
    nmatch64=(uint64_t)nmatch;nsample64=(uint64_t)nsample;
    f=fopen(path,"wb");if(!f)return -1;
    ok=fwrite(magic,1,sizeof(magic),f)==sizeof(magic)&&
       fwrite(&version,sizeof(version),1,f)==1&&
       fwrite(&flags,sizeof(flags),1,f)==1&&
       fwrite(ida,1,sizeof(ida),f)==sizeof(ida)&&
       fwrite(idb,1,sizeof(idb),f)==sizeof(idb)&&
       fwrite(&nva,sizeof(nva),1,f)==1&&fwrite(&nvb,sizeof(nvb),1,f)==1&&
       fwrite(&nca,sizeof(nca),1,f)==1&&fwrite(&ncb,sizeof(ncb),1,f)==1&&
       fwrite(&nmatch64,sizeof(nmatch64),1,f)==1&&
       fwrite(&nsample64,sizeof(nsample64),1,f)==1;
    for(size_t m=0;ok&&m<nmatch;m++){
        const ChartZipperMatch *cm=&match[m];int swap=0;
        int32_t ca=-1,cb=-1,pa,pb;
        uint64_t ia0,ia1,ib0,ib1,count=(uint64_t)cm->sample_count;
        double rank[6]={cm->full_coverage,cm->span_coverage,cm->mean_gap,
                        cm->support,cm->score,cm->order_coverage};
        if(cm->chart_a>=0&&(size_t)cm->chart_a<a->n_charts&&
           cm->chart_b>=(int32_t)a->n_charts&&
           (size_t)(cm->chart_b-(int32_t)a->n_charts)<b->n_charts){
            ca=cm->chart_a;cb=cm->chart_b-(int32_t)a->n_charts;
        }else if(cm->chart_b>=0&&(size_t)cm->chart_b<a->n_charts&&
                 cm->chart_a>=(int32_t)a->n_charts&&
                 (size_t)(cm->chart_a-(int32_t)a->n_charts)<b->n_charts){
            swap=1;ca=cm->chart_b;cb=cm->chart_a-(int32_t)a->n_charts;
        }else{ok=0;break;}
        pa=swap?cm->port_b:cm->port_a;
        pb=swap?cm->port_a:cm->port_b;
        ia0=(uint64_t)(swap?cm->interval_b_first:cm->interval_a_first);
        ia1=(uint64_t)(swap?cm->interval_b_last:cm->interval_a_last);
        ib0=(uint64_t)(swap?cm->interval_a_first:cm->interval_b_first);
        ib1=(uint64_t)(swap?cm->interval_a_last:cm->interval_b_last);
        if(cm->sample_first>nsample||
           cm->sample_count>nsample-cm->sample_first){ok=0;break;}
        ok=fwrite(&ca,sizeof(ca),1,f)==1&&fwrite(&cb,sizeof(cb),1,f)==1&&
           fwrite(&pa,sizeof(pa),1,f)==1&&fwrite(&pb,sizeof(pb),1,f)==1&&
           fwrite(&ia0,sizeof(ia0),1,f)==1&&
           fwrite(&ia1,sizeof(ia1),1,f)==1&&
           fwrite(&ib0,sizeof(ib0),1,f)==1&&
           fwrite(&ib1,sizeof(ib1),1,f)==1&&
           fwrite(&count,sizeof(count),1,f)==1&&
            fwrite(rank,sizeof(rank[0]),6,f)==6;
        for(size_t k=cm->sample_first;
            ok&&k<cm->sample_first+cm->sample_count;k++){
            int32_t va=swap?sample[k].vertex_b:sample[k].vertex_a;
            int32_t vb=swap?sample[k].vertex_a:sample[k].vertex_b;
            if(va<0||(size_t)va>=a->nv||vb<(int32_t)a->nv||
               (size_t)(vb-(int32_t)a->nv)>=b->nv){ok=0;break;}
            vb-=(int32_t)a->nv;
            ok=fwrite(&va,sizeof(va),1,f)==1&&
               fwrite(&vb,sizeof(vb),1,f)==1;
            written_samples++;
        }
    }
    if(written_samples!=nsample)ok=0;
    if(ok)ok=fflush(f)==0;
    if(fclose(f)!=0)ok=0;
    return ok?0:-1;
}

static int gw_stream_seam_main(int argc, char **argv)
{
    const char *a_path, *b_path, *out_path;
    const char *placed_dir = NULL, *axis_path = NULL;
    float cube_size = CUBE_SIZE_VOX;
    Arena_T arena = NULL;
    StreamWeldShard a, b;
    GwAxisTable axis;
    float *verts = NULL;
    int32_t *faces = NULL, *chart = NULL, *zipped = NULL;
    int32_t *physical_zipped = NULL;
    size_t nv = 0, nf = 0, ncharts = 0, zipped_nf = 0, nbridge = 0;
    ChartZipperParams zp;
    ChartZipperStats zs;
    ChartZipperStats physical_zs;
    ChartZipperTransaction *tx = NULL;
    ChartZipperTransaction *physical_tx = NULL;
    ChartZipperMatch *matches = NULL;
    ChartZipperMatchSample *match_samples = NULL;
    size_t ntx = 0;
    size_t nmatch = 0, nmatch_samples = 0;
    size_t physical_ntx = 0,physical_nf = 0,physical_nbridge = 0;
    StreamWeldConflict *conflicts = NULL;
    size_t nconflicts = 0;
    BpaBridgeGate gate;
    int64_t az=0,ay=0,ax=0,bz=0,by=0,bx=0;
    int rc = 1;
    int match_only = argc>=2&&!strcmp(argv[1],"--chart-match");
    const char *e;

    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s %s <a.gwshard> <b.gwshard> <%s> "
            "[--placed-calibration DIR] [--axis-table CSV] [--cube-size N]\n",
            argv[0],match_only?"--chart-match":"--stream-seam",
            match_only?"out.gcmatch":"out.gwpatch");
        return 1;
    }
    a_path=argv[2];b_path=argv[3];out_path=argv[4];
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--placed-calibration")&&i+1<argc)
            placed_dir=argv[++i];
        else if(!strcmp(argv[i],"--axis-table")&&i+1<argc)
            axis_path=argv[++i];
        else if(!strcmp(argv[i],"--cube-size")&&i+1<argc){
            double q=atof(argv[++i]);if(q<=0.0)return 1;cube_size=(float)q;
        }else{
            fprintf(stderr,"grid_weld %s: unknown arg %s\n",
                    match_only?"--chart-match":"--stream-seam",argv[i]);
            return 1;
        }
    }
    if(placed_dir&&gw_arm_placed_calibration(placed_dir)!=0){
        fprintf(stderr,"grid_weld --stream-seam: invalid placed calibration\n");
        return 1;
    }
    memset(&axis,0,sizeof(axis));
    if(axis_path&&gw_axis_table_load(axis_path,&axis)!=0){
        fprintf(stderr,"grid_weld --stream-seam: cannot load axis table %s\n",
                axis_path);return 1;
    }
    arena=Arena_new();memset(&a,0,sizeof(a));memset(&b,0,sizeof(b));
    if(StreamWeld_read_shard(arena,a_path,&a)!=0||
       StreamWeld_read_shard(arena,b_path,&b)!=0){
        fprintf(stderr,"grid_weld --stream-seam: invalid shard input\n");goto done;
    }
    if(parse_cube_origin(a.cube_id,&az,&ay,&ax)!=0||
       parse_cube_origin(b.cube_id,&bz,&by,&bx)!=0){
        fprintf(stderr,"grid_weld --stream-seam: invalid cube ids\n");goto done;
    }
    {
        int adjacent=((llabs(az-bz)==(long long)(cube_size+0.5f))&&ay==by&&ax==bx)+
                     ((llabs(ay-by)==(long long)(cube_size+0.5f))&&az==bz&&ax==bx)+
                     ((llabs(ax-bx)==(long long)(cube_size+0.5f))&&az==bz&&ay==by);
        if(adjacent!=1){
            fprintf(stderr,"grid_weld --stream-seam: %s and %s are not one "
                           "cube apart\n",a.cube_id,b.cube_id);goto done;
        }
    }
    if(a.nv>SIZE_MAX-b.nv||a.nf>SIZE_MAX-b.nf||
       a.n_charts>SIZE_MAX-b.n_charts||a.nv+b.nv>(size_t)INT32_MAX||
       a.n_charts+b.n_charts>(size_t)INT32_MAX)goto done;
    nv=a.nv+b.nv;nf=a.nf+b.nf;ncharts=a.n_charts+b.n_charts;
    verts=(float*)ARENA_ALLOC(arena,nv*3*sizeof(*verts));
    faces=(int32_t*)ARENA_ALLOC(arena,nf*3*sizeof(*faces));
    chart=(int32_t*)ARENA_ALLOC(arena,nv*sizeof(*chart));
    memcpy(verts,a.verts,a.nv*3*sizeof(*verts));
    memcpy(&verts[a.nv*3],b.verts,b.nv*3*sizeof(*verts));
    memcpy(faces,a.faces,a.nf*3*sizeof(*faces));
    for(size_t f=0;f<b.nf;f++)for(int k=0;k<3;k++){
        int32_t v=b.faces[f*3+(size_t)k];
        if(v<0||(size_t)v>=b.nv||a.nv>(size_t)INT32_MAX-(size_t)v){
            fprintf(stderr,"grid_weld --stream-seam: invalid face index\n");
            goto done;
        }
        faces[(a.nf+f)*3+(size_t)k]=(int32_t)a.nv+v;
    }
    for(size_t v=0;v<a.nv;v++){
        if(a.vertex_chart[v]>=0&&(size_t)a.vertex_chart[v]>=a.n_charts)goto done;
        chart[v]=a.vertex_chart[v];
    }
    for(size_t v=0;v<b.nv;v++){
        if(b.vertex_chart[v]>=0&&(size_t)b.vertex_chart[v]>=b.n_charts)goto done;
        chart[a.nv+v]=b.vertex_chart[v]<0 ? -1 :
            (int32_t)a.n_charts+b.vertex_chart[v];
    }
    if(!match_only){
        IntersectionCleanupParams ip;IntersectionCleanupStats is;
        IntersectionCleanup_default_params(&ip);ip.gap_max=0.0;ip.include_hinges=1;
        if(IntersectionCleanup_audit(verts,nv,faces,nf,NULL,&ip,NULL,&is)!=0||
           is.conflicts!=0){
            fprintf(stderr,"grid_weld --stream-seam: source pair is not "
                           "embedded (%zu exact conflict(s))\n",is.conflicts);
            goto done;
        }
    }
    memset(&gate,0,sizeof(gate));
    if((e=getenv("SEAM_UMBILICUS_Y")))gate.umb_y=atof(e);
    if((e=getenv("SEAM_UMBILICUS_X")))gate.umb_x=atof(e);
    if((e=getenv("SEAM_WRAP_PITCH"))){double q=atof(e);if(q>0)gate.pitch=q;}
    if(axis.n>=2){
        gate.axis_z=axis.z;gate.axis_y=axis.y;gate.axis_x=axis.x;gate.axis_n=axis.n;
    }
    if(gate.pitch>0.0&&(gate.axis_n>=2||gate.umb_y!=0.0||gate.umb_x!=0.0)){
        gate.tol=SEAM_WIND_TOL_DEFAULT_TURNS;
        gate.hard=SEAM_WIND_HARD_TOL_DEFAULT_TURNS;
        if((e=getenv("SEAM_WIND_TOL"))){double q=atof(e);if(q>0)gate.tol=q;}
        if((e=getenv("SEAM_WIND_HARD_TOL"))){double q=atof(e);if(q>0)gate.hard=q;}
    }
    ChartZipper_default_params(&zp);zp.cube_size=cube_size;
    zp.trace=getenv("SEAM_ZIP_TRACE")!=NULL;
    if((e=getenv("SEAM_BAND"))){double q=atof(e);if(q>0)zp.band=(float)q;}
    if((e=getenv("SEAM_ZIP_MAX_EDGE"))){double q=atof(e);if(q>0)zp.max_cross_edge=(float)q;}
    if((e=getenv("SEAM_ZIP_NORMAL_DOT"))){double q=atof(e);if(q>=0&&q<=1)zp.normal_dot_min=(float)q;}
    if((e=getenv("SEAM_ZIP_MIN_COVERAGE"))){double q=atof(e);if(q>0&&q<=1)zp.min_coverage=(float)q;}
    if((e=getenv("SEAM_ZIP_MIN_SUPPORT"))){double q=atof(e);if(q>0)zp.min_support_length=(float)q;}
    if((e=getenv("SEAM_ZIP_AMBIGUITY"))){double q=atof(e);if(q>0&&q<=1)zp.ambiguity_ratio=(float)q;}
    if(match_only){
        if(ChartZipper_match(
                arena,verts,nv,faces,nf,chart,ncharts,&gate,&zp,
                &matches,&nmatch,&match_samples,&nmatch_samples,&zs)!=0){
            fprintf(stderr,"grid_weld --chart-match: boundary matcher failed\n");
            goto done;
        }
        if(gw_write_chart_matches(
                out_path,&a,&b,matches,nmatch,match_samples,nmatch_samples)!=0){
            fprintf(stderr,"grid_weld --chart-match: cannot write %s\n",out_path);
            goto done;
        }
        fprintf(stderr,
                "chart match %s + %s: %zu scored edge(s), %zu ordered "
                "correspondence sample(s), zero weld faces -> %s\n",
                a.cube_id,b.cube_id,nmatch,nmatch_samples,out_path);
        rc=0;
        goto done;
    }
    if(ChartZipper_process_with_transactions(
            arena,verts,nv,faces,nf,chart,ncharts,&gate,&zp,
            &physical_zipped,&physical_nf,&physical_nbridge,&physical_zs,
            &physical_tx,&physical_ntx)!=0||!physical_zs.embedded_certificate){
        fprintf(stderr,"grid_weld --stream-seam: physical zipper failed\n");
        goto done;
    }
    if(ChartZipper_enumerate_transactions(
            arena,verts,nv,faces,nf,chart,ncharts,&gate,&zp,
            &zipped,&zipped_nf,&nbridge,&zs,&tx,&ntx)!=0){
        fprintf(stderr,"grid_weld --stream-seam: zipper failed\n");goto done;
    }
    if(gw_stream_mark_physical_transactions(
            zipped,tx,ntx,physical_zipped,physical_tx,physical_ntx)!=0){
        fprintf(stderr,
            "grid_weld --stream-seam: cannot embed proven physical subset "
            "in candidate graph\n");
        goto done;
    }
    if(gw_stream_prepare_transactions(
            arena,chart,nv,ncharts,zipped,nf,&zipped_nf,tx,&ntx,&nbridge)!=0||
       gw_stream_classify_transactions(
            verts,nv,zipped,nf,&zipped_nf,tx,&ntx,&nbridge,&zp,
            &conflicts,&nconflicts)!=0)
        goto done;
    if(StreamWeld_write_patch(out_path,&a,&b,zipped,zipped_nf,tx,ntx,
                              conflicts,nconflicts)!=0){
        fprintf(stderr,"grid_weld --stream-seam: cannot write %s\n",out_path);
        goto done;
    }
    fprintf(stderr,"stream seam %s + %s: %zu candidate(s), %zu individually "
                   "certified transaction(s), %zu physical, "
                   "%zu conflict edge(s), "
                   "%zu face(s) -> %s\n",
            a.cube_id,b.cube_id,zs.candidates,ntx,physical_ntx,
            nconflicts,nbridge,out_path);
    rc=0;
done:
    free(conflicts);
    if(arena)Arena_dispose(&arena);gw_axis_table_free(&axis);return rc;
}

static size_t compact_orphan_vertices(Arena_T arena,
                                       float **verts, size_t *nv,
                                       int32_t *faces, size_t nf,
                                       int16_t **cube_idx,
                                       uint8_t **is_weld,
                                       size_t *color_nv,
                                       int32_t **lineage_parent0,
                                       int32_t **lineage_parent1)
{
    size_t old_nv=*nv, kept=0, kept_color=0;
    uint8_t *used;
    int32_t *remap;
    float *new_verts;
    int16_t *new_cube;
    uint8_t *new_weld;
    int32_t *new_parent0=NULL,*new_parent1=NULL;
    if (old_nv==0) return 0;
    used=(uint8_t*)ARENA_CALLOC(arena,old_nv,1L);
    remap=(int32_t*)ARENA_ALLOC(arena,(old_nv*sizeof(*remap)));
    for(size_t f=0;f<nf;f++)for(int k=0;k<3;k++){
        int32_t v=faces[f*3+(size_t)k];
        if(v>=0&&(size_t)v<old_nv)used[v]=1;
    }
    for(size_t v=0;v<old_nv;v++){
        if(used[v]){
            remap[v]=(int32_t)kept++;
            if(v<*color_nv)kept_color++;
        }else remap[v]=-1;
    }
    if(kept==old_nv)return 0;
    new_verts=(float*)ARENA_ALLOC(arena,(kept*3*sizeof(*new_verts)));
    new_cube=(int16_t*)ARENA_ALLOC(arena,(kept*sizeof(*new_cube)));
    new_weld=(uint8_t*)ARENA_CALLOC(arena,kept,1L);
    if(lineage_parent0&&lineage_parent1&&
       *lineage_parent0&&*lineage_parent1){
        new_parent0=(int32_t*)ARENA_ALLOC(
            arena,(kept*sizeof(*new_parent0)));
        new_parent1=(int32_t*)ARENA_ALLOC(
            arena,(kept*sizeof(*new_parent1)));
    }
    for(size_t v=0;v<old_nv;v++)if(remap[v]>=0){
        size_t d=(size_t)remap[v];
        memcpy(&new_verts[d*3],&(*verts)[v*3],3*sizeof(float));
        if(v<*color_nv){
            new_cube[d]=(*cube_idx)[v];
            new_weld[d]=(*is_weld)[v];
        }else new_cube[d]=-1;
        if(new_parent0&&new_parent1){
            int32_t p0=(*lineage_parent0)[v];
            int32_t p1=(*lineage_parent1)[v];
            if(p0<=-2&&p1>=0){
                new_parent0[d]=p0;
                new_parent1[d]=p1;
            }else{
                new_parent0[d]=p0>=0&&((size_t)p0)<old_nv?remap[p0]:-1;
                new_parent1[d]=p1>=0&&((size_t)p1)<old_nv?remap[p1]:-1;
            }
        }
    }
    for(size_t i=0;i<nf*3;i++)faces[i]=remap[faces[i]];
    *verts=new_verts;*nv=kept;
    *cube_idx=new_cube;*is_weld=new_weld;*color_nv=kept_color;
    if(new_parent0&&new_parent1){
        *lineage_parent0=new_parent0;
        *lineage_parent1=new_parent1;
    }
    return old_nv-kept;
}

static size_t audit_source_lineage_positions(
    const char *stage, const float *verts, size_t nv,
    const int32_t *lineage_parent0, const int32_t *lineage_parent1,
    const float *source_reference, size_t source_reference_nv,
    const size_t *cube_voff, const CubeList *cubes)
{
    size_t adjusted=0, violations=0, anchors=0;
    double maximum=0.0;
    const double allowed=
        (double)HOLEFILL_CHART_PINCH_MAX_STEP_VOX+1.0e-3;
    if(!lineage_parent0||!lineage_parent1||!source_reference||
       !cube_voff||!cubes)return 0;
    for(size_t v=0;v<nv;v++){
        int32_t p0=lineage_parent0[v],local=lineage_parent1[v];
        if(p0>-2)continue;
        int64_t cube=-(int64_t)p0-2;
        anchors++;
        if(cube<0||(uint64_t)cube>=(uint64_t)cubes->n||local<0||
           (size_t)local>=cube_voff[(size_t)cube+1]-cube_voff[(size_t)cube]){
            violations++;
            continue;
        }
        size_t source=cube_voff[(size_t)cube]+(size_t)local;
        if(source>=source_reference_nv||
           memcmp(&verts[v*3],&source_reference[source*3],
                  3*sizeof(float))!=0){
            double d[3]={0,0,0};
            if(source<source_reference_nv)for(int axis=0;axis<3;axis++)
                d[axis]=(double)verts[v*3+(size_t)axis]-
                        (double)source_reference[source*3+(size_t)axis];
            double displacement=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
            if(displacement>maximum)maximum=displacement;
            if(displacement<=allowed){
                adjusted++;
            }else{
                if(violations<16)fprintf(stderr,
                    "  source lineage %s VIOLATION: final=%zu cube=%s "
                    "local=%d displacement=%.9g > %.9g\n",stage,v,
                    cube>=0&&(size_t)cube<cubes->n?
                        cubes->ids[(size_t)cube]:"?",
                    local,displacement,allowed);
                violations++;
            }
        }
    }
    fprintf(stderr,
        "  source lineage %s: %zu explicit anchor(s), %zu bounded chart "
        "adjustment(s), %zu violation(s), max displacement %.9g\n",
        stage,anchors,adjusted,violations,maximum);
    return violations;
}

static void set_env_double(const char *name, double value)
{
    char text[64];
    snprintf(text, sizeof(text), "%.12g", value);
#ifdef _MSC_VER
    _putenv_s(name, text);
#else
    setenv(name, text, 1);
#endif
}

static void dump_stage(Arena_T arena, const char *dir, const char *prefix,
                       int idx, const char *name,
                       const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       const int16_t *vert_cube_idx,
                       const int8_t *cube_palette,
                       size_t n_cubes, size_t color_nv)
{
    if (!dir) return;
    static const float GREY[3] = { 0.85f, 0.85f, 0.85f };
    Arena_Mark m = Arena_save(arena);
    float *colors = (float *)ARENA_ALLOC(arena,
                       (nv * 3L * sizeof(float)));
    const char *gw_color_by = getenv("GW_COLOR_BY");
    if (gw_color_by && !strcmp(gw_color_by, "cube")) {
        for (size_t v = 0; v < nv; v++) {
            const float *c = GREY;
            if (v < color_nv) {
                int cube = vert_cube_idx[v];
                int pal = (cube >= 0 && cube < (int)n_cubes) ? cube_palette[cube] : 0;
                c = CUBE_PALETTE[pal];
            }
            colors[v * 3 + 0] = c[0];
            colors[v * 3 + 1] = c[1];
            colors[v * 3 + 2] = c[2];
        }
    } else {
        CCColorOpts opts; CCColor_default_opts(&opts); opts.sat = 0.75;
        CCColor_compute(nv, faces, nf, &opts, colors, NULL);
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s_%02d_%s.obj", dir, prefix, idx, name);
    ves_ensure_parent_dir(path);
    int rc = ObjIO_write_per_vertex_color(path, verts, nv, faces, nf, colors);
    fprintf(stderr, "  [dump-stage %02d] %-11s %zu v %zu f -> %s%s\n",
            idx, name, nv, nf, path, rc == 0 ? "" : "  [WRITE FAILED]");
    fflush(stderr);
    Arena_restore(arena, m);
}

static double gw_phase_prev = -1.0;
static void gw_phase(const char *name)
{
    double now = ves_clock_sec();
    if (gw_phase_prev >= 0.0) {
        fprintf(stderr, "  [time] %-12s %7.2fs\n", name, now - gw_phase_prev);
    }
    gw_phase_prev = now;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && (!strcmp(argv[1], "--stream-seam") ||
                      !strcmp(argv[1], "--chart-match")))
        return gw_stream_seam_main(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return grid_weld_selftest();
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <grid_obj_dir> <output.obj> [--stage <name>] "
            "[--emit-weld-verts <path>] [--emit-lineage <path>] "
            "[--placed-facekeep <dir>] [--stream-shard <path>] [--no-bridge]\n"
            "       %s <grid_obj_dir> <output.obj> --pair <cubeA_id> <compA> <cubeB_id> <compB> [--stage <name>]\n"
            "\n"
            "Reads <grid_obj_dir>/<cube_id>/<cube_id>_<stage>/<cube_id>_<stage>_all.obj\n"
            "for every <cube_id> subdirectory. Plain-concatenates the cubes and\n"
            "welds them by re-running BPA across each cube-boundary seam plane.\n"
            "Writes welded mesh to <output.obj> and a JSON audit report\n"
            "to <output.obj>.weld_report.json.\n"
            "\n"
            "--pair <cubeA_id> <compA> <cubeB_id> <compB>: debug mode -- weld just\n"
            "                two individual components (the _<comp>.obj files, comp\n"
            "                1-based) instead of the whole grid. Auto-dumps the\n"
            "                BPA init front to <output.obj>.front_{tris,edges}.obj.\n"
            "\n"
            "--stage <name>: which dump stage to read (default: step12_final).\n"
            "                e.g. step1_bpa, step7_cc_bpa, step12_final.\n"
            "--no-bridge: assemble the BPA-bounded cube meshes verbatim; do not\n"
            "                synthesize, delete, or repair cross-cube seam faces.\n"
            "--axis-table z,y,x.csv: sampled curved umbilicus for every winding\n"
            "                gate (linear interpolation + endpoint extrapolation).\n"
            "\n"
            "--subgrid z0 z1 y0 y1 x0 x1: weld only cubes whose origin lies in this\n"
            "                inclusive source-voxel bbox -- stitches one rectangular\n"
            "                sub-block of a larger dump dir (e.g. a 4x5x5 tile).\n"
            "\n"
            "--dump-stages <dir>: write each weld stage to its own colored OBJ\n"
            "                <dir>/<prefix>_NN_<stage>.obj (concat..final).\n"
            "                Safe on large regions: per-hole dumps are separate.\n"
            "--dump-holes <dir>: write a per-hole input/result OBJ for every\n"
            "                interior hole considered (very verbose; small regions).\n"
            "--stage-prefix <id>: filename prefix for --dump-stages (default weld).\n"
            "--no-intersection-clean: skip bounded final conflict-graph surgery.\n"
            "--source-disk-cut: normalize source charts to disks with seam cuts\n"
            "                instead of geometric caps (preserves every face).\n"
            "\n"
            "Optional --emit-weld-verts <path> writes a packed binary sidecar:\n"
            "  uint32 nv; uint8 is_weld[nv];\n"
            "Used by the winding diagnostic to compute its headline weld-vert\n"
            "disagreement rate.\n"
            "\n"
            "Optional --emit-lineage <path> writes the exact seam-refinement\n"
            "midpoint parents plus the certified source-face prefix. This lets\n"
            "the winding-safe chart UV follow grid_weld without re-solving it.\n"
            "\n"
            "--placed-facekeep <dir> reads <id>_mesh.vmesh plus <id>_facekeep.u8\n"
            "from scroll_whole's placed directory. Only its winding-certified\n"
            "faces enter refinement/welding; rejected fusion links are never\n"
            "reintroduced. This option is unavailable with --pair.\n"
            "\n"
            "--stream-shard <path> stops after source repair + seam refinement\n"
            "                and writes one binary cube shard. Use repeated\n"
            "                --seam-plane <z|y|x|0|1|2> <coord> to name only\n"
            "                the occupied neighbour planes for that cube.\n"
            "--stream-seam is a separate bounded worker; run with no arguments\n"
            "                after the flag to see its usage.\n",
            argv[0], argv[0]);
        return 1;
    }
    const char *grid_dir = argv[1];
    const char *out_path = argv[2];
    const char *weld_verts_out = NULL;
    const char *lineage_out = NULL;
    const char *placed_facekeep_dir = NULL;
    const char *stream_shard_out = NULL;
    const char *stage = "step12_final";
    const char *dump_stages_dir = NULL;
    const char *dump_holes_dir = NULL;
    const char *stage_prefix = "weld";
    const char *axis_table_path = NULL;
    int allow_unarmed = 0;
    GwAxisTable axis_table;
    memset(&axis_table, 0, sizeof(axis_table));
    float seam_cube = CUBE_SIZE_VOX;
    int pair_mode = 0;
    const char *pair_cube[2] = { NULL, NULL };
    int pair_comp[2] = { 0, 0 };
    int no_bridge = 0;
    int no_pinhole = (getenv("SEAM_NO_PINHOLE") != NULL);
    int no_cleanup = (getenv("SEAM_NO_CLEANUP") != NULL);
    int no_holefill = (getenv("SEAM_NO_HOLEFILL") != NULL);
    int no_intersection_clean =
        (getenv("SEAM_NO_INTERSECTION_CLEAN") != NULL);
    int source_disk_cut = (getenv("SEAM_SOURCE_DISK_CUT") != NULL);
    int legacy_bpa = (getenv("SEAM_LEGACY_BPA") != NULL);
    int subgrid = 0;
    int64_t sg_z0 = 0, sg_z1 = 0, sg_y0 = 0, sg_y1 = 0, sg_x0 = 0, sg_x1 = 0;
    size_t vert_cap_override = 0, face_cap_override = 0;
    SeamPlane explicit_planes[64];
    size_t explicit_nplanes = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--emit-weld-verts") == 0 && i + 1 < argc) {
            weld_verts_out = argv[++i];
        } else if (strcmp(argv[i], "--emit-lineage") == 0 && i + 1 < argc) {
            lineage_out = argv[++i];
        } else if (strcmp(argv[i], "--placed-facekeep") == 0 && i + 1 < argc) {
            placed_facekeep_dir = argv[++i];
        } else if (strcmp(argv[i], "--stream-shard") == 0 && i + 1 < argc) {
            stream_shard_out = argv[++i];
        } else if (strcmp(argv[i], "--seam-plane") == 0 && i + 2 < argc) {
            const char *as = argv[++i];
            int axis = !strcmp(as,"z") || !strcmp(as,"0") ? 0 :
                       !strcmp(as,"y") || !strcmp(as,"1") ? 1 :
                       !strcmp(as,"x") || !strcmp(as,"2") ? 2 : -1;
            double coord = atof(argv[++i]);
            if (axis < 0 || axis > 2 || !isfinite(coord) ||
                explicit_nplanes >= sizeof(explicit_planes)/sizeof(explicit_planes[0])) {
                fprintf(stderr,"grid_weld: invalid --seam-plane %s %.17g\n",as,coord);
                return 1;
            }
            explicit_planes[explicit_nplanes].axis=axis;
            explicit_planes[explicit_nplanes].coord=coord;
            explicit_nplanes++;
        } else if (strcmp(argv[i], "--no-bridge") == 0) {
            no_bridge = 1;
        } else if (strcmp(argv[i], "--no-pinhole") == 0) {
            no_pinhole = 1;
        } else if (strcmp(argv[i], "--no-cleanup") == 0) {
            no_cleanup = 1;
        } else if (strcmp(argv[i], "--no-holefill") == 0) {
            no_holefill = 1;
        } else if (strcmp(argv[i], "--no-intersection-clean") == 0) {
            no_intersection_clean = 1;
        } else if (strcmp(argv[i], "--source-disk-cut") == 0) {
            source_disk_cut = 1;
        } else if (strcmp(argv[i], "--stage") == 0 && i + 1 < argc) {
            stage = argv[++i];
        } else if (strcmp(argv[i], "--axis-table") == 0 && i + 1 < argc) {
            axis_table_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-stages") == 0 && i + 1 < argc) {
            dump_stages_dir = argv[++i];
        } else if (strcmp(argv[i], "--dump-holes") == 0 && i + 1 < argc) {
            dump_holes_dir = argv[++i];
        } else if (strcmp(argv[i], "--stage-prefix") == 0 && i + 1 < argc) {
            stage_prefix = argv[++i];
        } else if (strcmp(argv[i], "--pair") == 0 && i + 4 < argc) {
            pair_mode = 1;
            pair_cube[0] = argv[++i];
            pair_comp[0] = atoi(argv[++i]);
            pair_cube[1] = argv[++i];
            pair_comp[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seam-weld") == 0) {
        } else if (strcmp(argv[i], "--cube-size") == 0 && i + 1 < argc) {
            seam_cube = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--subgrid") == 0 && i + 6 < argc) {
            subgrid = 1;
            sg_z0 = atoll(argv[++i]); sg_z1 = atoll(argv[++i]);
            sg_y0 = atoll(argv[++i]); sg_y1 = atoll(argv[++i]);
            sg_x0 = atoll(argv[++i]); sg_x1 = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--vert-cap") == 0 && i + 1 < argc) {
            vert_cap_override = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--face-cap") == 0 && i + 1 < argc) {
            face_cap_override = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--allow-unarmed") == 0) {
            allow_unarmed = 1;
        } else {
            fprintf(stderr, "grid_weld: unknown arg %s\n", argv[i]);
            return 1;
        }
    }

    if (pair_mode && placed_facekeep_dir != NULL) {
        fprintf(stderr,
                "grid_weld: --placed-facekeep is unavailable with --pair\n");
        return 1;
    }
    if (stream_shard_out != NULL && pair_mode) {
        fprintf(stderr,"grid_weld: --stream-shard is unavailable with --pair\n");
        return 1;
    }
    if (placed_facekeep_dir != NULL &&
        gw_arm_placed_calibration(placed_facekeep_dir) != 0) {
        fprintf(stderr,
            "grid_weld: cannot read a valid winding calibration from "
            "%s/placed_index.json\n", placed_facekeep_dir);
        return 1;
    }

    if (axis_table_path != NULL &&
        gw_axis_table_load(axis_table_path, &axis_table) != 0) {
        fprintf(stderr, "grid_weld: cannot load axis table %s "
                "(need >=2 unique finite z,y,x rows)\n", axis_table_path);
        return 1;
    }

    if (!no_bridge) {
        const char *ep = getenv("SEAM_WRAP_PITCH");
        const char *ey = getenv("SEAM_UMBILICUS_Y");
        const char *ex = getenv("SEAM_UMBILICUS_X");
        double pitch_env = ep != NULL ? atof(ep) : 0.0;
        int axis_ok = axis_table_path != NULL ||
                      (ey != NULL && ex != NULL &&
                       (atof(ey) != 0.0 || atof(ex) != 0.0));
        int armed = pitch_env > 0.0 && axis_ok;
        PitchTable ptab_probe = PitchTable_from_env();
        if (armed) {
            printf("grid_weld: seam winding gate ARMED (umb y=%s x=%s "
                   "pitch=%s axis_table=%s pitch_table_knots=%d)\n",
                   ey != NULL ? ey : "-", ex != NULL ? ex : "-", ep,
                   axis_table_path != NULL ? axis_table_path : "-",
                   PitchTable_knots(ptab_probe));
        } else if (allow_unarmed) {
            printf("grid_weld: seam winding gate OFF (--allow-unarmed)\n");
        } else {
            fprintf(stderr,
                "grid_weld: REFUSING an unarmed primary weld: set "
                "SEAM_UMBILICUS_Y/SEAM_UMBILICUS_X + SEAM_WRAP_PITCH "
                "(or pass --axis-table with SEAM_WRAP_PITCH), or pass "
                "--allow-unarmed for a deliberate gates-off debug weld\n");
            return 2;
        }
    } else {
        printf("grid_weld: assembly-only mode (--no-bridge); seam gates unused\n");
    }

    if (dump_holes_dir) {
        set_env_if_unset("SEAM_HOLE_DUMP_DIR", dump_holes_dir);
        set_env_if_unset("SEAM_HOLE_DUMP_PREFIX", stage_prefix);
    }
    if (!no_bridge) set_env_if_unset("SEAM_SLIVER_MIN_ALT", "0");

    Arena_T arena = Arena_new();
    int ok = 0;
    float *out_verts = NULL;
    size_t out_nv = 0;
    int32_t *lineage_parent0 = NULL, *lineage_parent1 = NULL;
    size_t lineage_initialized_nv = 0;
    float *lineage_source_reference = NULL;
    size_t lineage_source_reference_nv = 0;
    size_t *lineage_cube_voff = NULL;
    size_t lineage_source_faces = 0;
    SortedFace *all_faces = NULL;
    size_t all_nf = 0;
    size_t face_cap = 0;
    ManifoldStats ms = {0, 0, 0, 0};
    size_t n_unique_faces = 0;
    CubeList cubes = {0, 0, 0};

    TRY
        gw_phase("start");
        if (pair_mode) {
            if (pair_comp[0] <= 0 || pair_comp[1] <= 0) {
                fprintf(stderr, "grid_weld: --pair comp ids must be >= 1\n");
                RAISE(IO_Failed);
            }
            cubelist_push(arena, &cubes, pair_cube[0]);
            cubelist_push(arena, &cubes, pair_cube[1]);
            fprintf(stderr,
                "grid_weld: pair mode -- %s comp %03d + %s comp %03d (stage %s)\n",
                pair_cube[0], pair_comp[0], pair_cube[1], pair_comp[1], stage);
        } else {
            if (enumerate_cube_dirs(arena, grid_dir, &cubes) != 0) {
                fprintf(stderr, "grid_weld: cannot enumerate %s\n", grid_dir);
                RAISE(IO_Failed);
            }
            if (cubes.n == 0) {
                fprintf(stderr, "grid_weld: no cube directories under %s\n",
                        grid_dir);
                RAISE(IO_Failed);
            }
            fprintf(stderr, "grid_weld: %zu cubes under %s\n", cubes.n, grid_dir);
            if (subgrid) {
                cubelist_filter_bbox(&cubes, sg_z0, sg_z1, sg_y0, sg_y1, sg_x0, sg_x1);
                fprintf(stderr,
                    "grid_weld: --subgrid z[%lld,%lld] y[%lld,%lld] x[%lld,%lld] -> %zu cubes\n",
                    (long long)sg_z0, (long long)sg_z1, (long long)sg_y0,
                    (long long)sg_y1, (long long)sg_x0, (long long)sg_x1, cubes.n);
                if (cubes.n == 0) {
                    fprintf(stderr, "grid_weld: --subgrid matched no cubes\n");
                    RAISE(IO_Failed);
                }
            }
            if (stream_shard_out != NULL && cubes.n != 1) {
                fprintf(stderr,
                    "grid_weld: --stream-shard requires a subgrid selecting "
                    "exactly one cube (selected %zu)\n", cubes.n);
                RAISE(IO_Failed);
            }
        }

        size_t vert_per_cube_est = 1500000;
        size_t vert_cap = vert_cap_override ? vert_cap_override
                                            : cubes.n * vert_per_cube_est;
        if (vert_cap < (1 << 16)) vert_cap = (1 << 16);
        if (vert_cap > 150000000) vert_cap = 150000000;
        out_verts = (float *)ARENA_ALLOC(arena,
                       (vert_cap * 3L * sizeof(float)));
        int16_t *vert_cube_idx = (int16_t *)ARENA_ALLOC(arena,
                                     vert_cap * sizeof(int16_t));
        uint8_t *vert_is_weld = (uint8_t *)ARENA_CALLOC(arena,
                                    vert_cap, 1L);
        if (lineage_out != NULL || stream_shard_out != NULL) {
            lineage_parent0 = (int32_t *)ARENA_ALLOC(
                arena, (vert_cap * sizeof(*lineage_parent0)));
            lineage_parent1 = (int32_t *)ARENA_ALLOC(
                arena, (vert_cap * sizeof(*lineage_parent1)));
            lineage_cube_voff = (size_t *)ARENA_ALLOC(
                arena, ((cubes.n + 1) * sizeof(*lineage_cube_voff)));
        }
        int8_t *cube_palette = (int8_t *)ARENA_ALLOC(arena,
                                   cubes.n * sizeof(int8_t));
        for (size_t i = 0; i < cubes.n; i++) {
            int64_t vz = 0, vy = 0, vx = 0;
            (void)parse_cube_origin(cubes.ids[i], &vz, &vy, &vx);
            int32_t cz = (int32_t)(vz / 128);
            int32_t cy = (int32_t)(vy / 128);
            int32_t cx = (int32_t)(vx / 128);
            uint32_t idx = (uint32_t)(cz * 7 + cy * 3 + cx);
            cube_palette[i] = (int8_t)(idx & 0xFu);
        }

        face_cap = face_cap_override ? face_cap_override : cubes.n * 2000000;
        if (face_cap < (1 << 16)) face_cap = (1 << 16);
        if (face_cap > 80000000) face_cap = 80000000;
        all_faces = (SortedFace *)ARENA_ALLOC(arena,
                       (face_cap * sizeof(SortedFace)));

        Arena_T scratch = Arena_new();
        size_t total_in_verts = 0;
        size_t total_in_faces = 0;
        size_t total_unique_verts = 0;
        size_t cubes_loaded = 0;
        size_t placed_missing = 0;

        for (size_t ci = 0; ci < cubes.n; ci++) {
            if (lineage_cube_voff) {
                lineage_cube_voff[ci] = out_nv;
                lineage_cube_voff[ci + 1] = out_nv;
            }
            const char *cube_id = cubes.ids[ci];
            int64_t vz = 0, vy = 0, vx = 0;
            if (parse_cube_origin(cube_id, &vz, &vy, &vx) != 0) {
                fprintf(stderr, "grid_weld: bad cube_id %s (skip)\n", cube_id);
                continue;
            }
            char obj_path[1024];
            char facekeep_path[1024] = {0};
            if (placed_facekeep_dir != NULL) {
                snprintf(obj_path, sizeof(obj_path), "%s/%s_mesh.vmesh",
                         placed_facekeep_dir, cube_id);
                snprintf(facekeep_path, sizeof(facekeep_path),
                         "%s/%s_geomkeep.u8", placed_facekeep_dir, cube_id);
                if (!gw_file_exists(facekeep_path))
                    snprintf(facekeep_path, sizeof(facekeep_path),
                             "%s/%s_facekeep.u8", placed_facekeep_dir,
                             cube_id);
            } else if (pair_mode) {
                snprintf(obj_path, sizeof(obj_path),
                    "%s/%s/%s_%s/%s_%s_%03d.obj",
                    grid_dir, cube_id, cube_id, stage, cube_id, stage,
                    pair_comp[ci]);
            } else {
                snprintf(obj_path, sizeof(obj_path),
                    "%s/%s/%s_%s/%s_%s_all.obj",
                    grid_dir, cube_id, cube_id, stage, cube_id, stage);
                if (!gw_file_exists(obj_path)) {
                    snprintf(obj_path, sizeof(obj_path),
                        "%s/%s/%s_%s_all.obj",
                        grid_dir, cube_id, cube_id, stage);
                }
            }

            if (placed_facekeep_dir != NULL) {
                int presence = gw_placed_pair_presence(
                    gw_file_exists(obj_path), gw_file_exists(facekeep_path));
                if (presence == 0) {
                    placed_missing++;
                    fprintf(stderr,
                            "  cube %s: no trusted placed mesh/facekeep pair "
                            "(skip)\n", cube_id);
                    continue;
                }
                if (presence < 0) {
                    fprintf(stderr,
                            "  cube %s: incomplete placed pair (%s, %s) -- "
                            "FATAL\n", cube_id, obj_path, facekeep_path);
                    RAISE(IO_Failed);
                }
            }

            Arena_free(scratch);
            float *vin = NULL;
            int32_t *fin = NULL;
            size_t nv_in = 0, nf_in = 0;
            if (placed_facekeep_dir != NULL) {
                MeshBinData mesh;
                if (MeshBin_read_arena(scratch, obj_path, &mesh) != 0) {
                    fprintf(stderr,
                        "  cube %s: authoritative binary mesh read failed "
                        "(%s); OBJ fallback is disabled -- FATAL\n",
                        cube_id, obj_path);
                    RAISE(IO_Failed);
                }
                vin = mesh.verts; nv_in = mesh.nv;
                fin = mesh.faces; nf_in = mesh.nf;
            } else if (ObjIO_read(scratch, obj_path, &vin, &nv_in,
                                  &fin, &nf_in) != 0) {
                fprintf(stderr, "  cube %s: OBJ read failed (%s) -- %s\n",
                        cube_id, obj_path, pair_mode ? "FATAL" : "skip");
                if (pair_mode) RAISE(IO_Failed);
                continue;
            }
            uint8_t *facekeep = NULL;
            size_t nf_kept = nf_in;
            if (placed_facekeep_dir != NULL) {
                if (gw_read_facekeep(scratch, facekeep_path, nf_in,
                                     &facekeep, &nf_kept) != 0) {
                    fprintf(stderr,
                        "  cube %s: invalid/mismatched facekeep (%s), "
                        "expected %zu bytes -- FATAL\n",
                        cube_id, facekeep_path, nf_in);
                    RAISE(IO_Failed);
                }
            }
            cubes_loaded++;
            total_in_verts += nv_in;
            total_in_faces += nf_kept;

            (void)vz; (void)vy; (void)vx;
            int32_t *remap = (int32_t *)ARENA_ALLOC(scratch,
                                nv_in * sizeof(int32_t));
            if (out_nv + nv_in > vert_cap) {
                fprintf(stderr,
                    "grid_weld: vert capacity %zu exceeded\n", vert_cap);
                RAISE(IO_Failed);
            }
            for (size_t v = 0; v < nv_in; v++) {
                int32_t new_idx = (int32_t)out_nv;
                if (lineage_parent0 && lineage_parent1) {
                    if (ci > (size_t)INT32_MAX - 2 ||
                        v > (size_t)INT32_MAX) {
                        fprintf(stderr,
                                "grid_weld: source provenance exceeds int32 "
                                "range at cube %zu vertex %zu\n", ci, v);
                        RAISE(IO_Failed);
                    }
                    lineage_parent0[out_nv] = -(int32_t)ci - 2;
                    lineage_parent1[out_nv] = (int32_t)v;
                }
                out_verts[out_nv * 3 + 0] = vin[v * 3 + 0];
                out_verts[out_nv * 3 + 1] = vin[v * 3 + 1];
                out_verts[out_nv * 3 + 2] = vin[v * 3 + 2];
                vert_cube_idx[out_nv] = (int16_t)ci;
                out_nv++;
                remap[v] = new_idx;
            }
            total_unique_verts += nv_in;

            for (size_t f = 0; f < nf_in; f++) {
                if (facekeep != NULL && !facekeep[f]) continue;
                int32_t a = remap[fin[f * 3 + 0]];
                int32_t b = remap[fin[f * 3 + 1]];
                int32_t c = remap[fin[f * 3 + 2]];
                if (a == b || b == c || a == c) continue;
                int32_t sorted[3] = { a, b, c };
                sort3i(sorted);
                if (all_nf >= face_cap) {
                    fprintf(stderr,
                        "grid_weld: face capacity %zu exceeded\n", face_cap);
                    RAISE(IO_Failed);
                }
                all_faces[all_nf].a = sorted[0];
                all_faces[all_nf].b = sorted[1];
                all_faces[all_nf].c = sorted[2];
                all_faces[all_nf].orig0 = a;
                all_faces[all_nf].orig1 = b;
                all_faces[all_nf].orig2 = c;
                all_nf++;
            }

            if (facekeep != NULL)
                fprintf(stderr,
                    "  cube %s: %zu v, %zu/%zu winding-certified f "
                    "(dropped %zu before weld)\n",
                    cube_id, nv_in, nf_kept, nf_in, nf_in - nf_kept);
            else
                fprintf(stderr, "  cube %s: %zu v, %zu f (concatenated)\n",
                         cube_id, nv_in, nf_in);
            if(lineage_cube_voff)lineage_cube_voff[ci+1]=out_nv;
        }
        Arena_dispose(&scratch);
        fprintf(stderr,
                "grid_weld: loaded %zu/%zu cube(s); skipped %zu without a "
                "trusted placed pair\n",
                cubes_loaded, cubes.n, placed_missing);
        lineage_initialized_nv = out_nv;
        if(lineage_parent0&&lineage_parent1){
            lineage_source_reference_nv=out_nv;
            lineage_source_reference=(float*)ARENA_ALLOC(
                arena,(out_nv*3*sizeof(*lineage_source_reference)));
            memcpy(lineage_source_reference,out_verts,
                   out_nv*3*sizeof(*lineage_source_reference));
            audit_source_lineage_positions(
                "after-load",out_verts,out_nv,lineage_parent0,lineage_parent1,
                lineage_source_reference,lineage_source_reference_nv,
                lineage_cube_voff,&cubes);
        }

        fprintf(stderr,
            "grid_weld: aggregate: %zu v_in -> %zu unique, %zu f_in pre-dedup\n",
            total_in_verts, out_nv, all_nf);
        gw_phase("load");

        if (pair_mode) {
            for (size_t ci2 = 0; ci2 < cubes.n; ci2++) {
                float lo[3] = { 1e30f, 1e30f, 1e30f };
                float hi[3] = { -1e30f, -1e30f, -1e30f };
                size_t cnt = 0;
                for (size_t v = 0; v < out_nv; v++) {
                    if (vert_cube_idx[v] != (int16_t)ci2) continue;
                    for (int a = 0; a < 3; a++) {
                        float c = out_verts[v * 3 + a];
                        if (c < lo[a]) lo[a] = c;
                        if (c > hi[a]) hi[a] = c;
                    }
                    cnt++;
                }
                fprintf(stderr,
                    "  pair[%zu] %s comp %03d: %zu v, bbox "
                    "z[%.1f,%.1f] y[%.1f,%.1f] x[%.1f,%.1f]\n",
                    ci2, cubes.ids[ci2], pair_comp[ci2], cnt,
                    (double)lo[0], (double)hi[0], (double)lo[1], (double)hi[1],
                    (double)lo[2], (double)hi[2]);
            }
        }

        qsort(all_faces, all_nf, sizeof(SortedFace), cmp_sorted_face);
        size_t write_i = 0;
        for (size_t i = 0; i < all_nf; ) {
            size_t j = i + 1;
            while (j < all_nf &&
                   all_faces[j].a == all_faces[i].a &&
                   all_faces[j].b == all_faces[i].b &&
                   all_faces[j].c == all_faces[i].c) {
                j++;
            }
            if (write_i != i) all_faces[write_i] = all_faces[i];
            write_i++;
            i = j;
        }
        n_unique_faces = write_i;
        fprintf(stderr, "grid_weld: %zu unique faces (deduped from %zu)\n",
                n_unique_faces, all_nf);
        gw_phase("facesort");

        int32_t *flat_faces = (int32_t *)ARENA_ALLOC(arena,
                                  n_unique_faces * 3L * sizeof(int32_t));
        for (size_t f = 0; f < n_unique_faces; f++) {
            flat_faces[f * 3 + 0] = all_faces[f].orig0;
            flat_faces[f * 3 + 1] = all_faces[f].orig1;
            flat_faces[f * 3 + 2] = all_faces[f].orig2;
        }

        size_t color_nv = out_nv;
        dump_stage(arena, dump_stages_dir, stage_prefix, 0, "concat",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);
        size_t recoarsen_collapses = 0, recoarsen_faces_in = 0,
               recoarsen_faces_out = 0;
        size_t bandcvt_accepted = 0, bandcvt_rejected = 0,
               bandcvt_faces_in = 0, bandcvt_faces_out = 0;
        IntersectionCleanupStats intersection_stats;
        IntersectionCleanupStats embedded_stats;
        size_t intersection_faces_in = 0, intersection_masked_faces = 0;
        size_t intersection_planes = 0, intersection_pinch_splits = 0;
        DiskTopologyRepairStats disk_topology_stats;
        size_t disk_topology_components = 0;
        int disk_topology_certificate = 0;
        int embedded_geometry_certificate = 0;
        ChartLineage chart_lineage;
        int have_chart_lineage = 0;
        memset(&intersection_stats, 0, sizeof intersection_stats);
        memset(&embedded_stats, 0, sizeof embedded_stats);
        memset(&disk_topology_stats, 0, sizeof disk_topology_stats);
        memset(&chart_lineage, 0, sizeof chart_lineage);
        if (!no_bridge) {
            if (source_disk_cut) {
                size_t source_closed = 0, source_closed_faces = 0;
                size_t source_closed_verts = 0;
                if (drop_closed_source_bubbles(
                        arena, out_verts, &out_nv,
                        flat_faces, &n_unique_faces,
                        vert_cube_idx, vert_is_weld,
                        lineage_parent0, lineage_parent1,
                        &source_closed, &source_closed_faces,
                        &source_closed_verts) != 0) {
                    fprintf(stderr,
                            "ERROR: source closed-bubble filter failed\n");
                    RAISE(IO_Failed);
                }
                if (source_closed > 0)
                    fprintf(stderr,
                            "  source chart prefilter: removed %zu closed "
                            "bubble(s), %zu faces, %zu vertices\n",
                            source_closed, source_closed_faces,
                            source_closed_verts);
                const size_t source_before_nv = out_nv;
                DiskTopologyRepairStats source_cut_stats;
                memset(&source_cut_stats, 0, sizeof source_cut_stats);
                if (source_disk_repair_with_metadata(
                        arena, &out_verts, &out_nv,
                        &flat_faces, &n_unique_faces, vert_cap,
                        vert_cube_idx, vert_is_weld,
                        lineage_parent0, lineage_parent1,
                        &source_cut_stats) != 0) {
                    fprintf(stderr,
                            "ERROR: source disk-cut transaction failed "
                            "(%zu/%zu components repaired, %zu failures, "
                            "%zu vertices, %zu faces; capacity=%zu)\n",
                            source_cut_stats.repaired_components,
                            source_cut_stats.components,
                            source_cut_stats.failed_components,
                            source_before_nv, n_unique_faces, vert_cap);
                    RAISE(IO_Failed);
                }
                color_nv = out_nv;
                lineage_initialized_nv = out_nv;
                audit_source_lineage_positions(
                    "after-source-disk-cut", out_verts, out_nv,
                    lineage_parent0, lineage_parent1,
                    lineage_source_reference, lineage_source_reference_nv,
                    lineage_cube_voff, &cubes);
                fprintf(stderr,
                        "  source chart disk cut: %zu component(s), %zu already "
                        "disks, %zu repaired, %zu seam edge(s); %zu -> %zu "
                        "vertices, %zu faces preserved\n",
                        source_cut_stats.components,
                        source_cut_stats.already_disks,
                        source_cut_stats.repaired_components,
                        source_cut_stats.seam_edges,
                        source_cut_stats.vertices_in,
                        source_cut_stats.vertices_out,
                        source_cut_stats.faces_out);
                {
                    uint8_t *self_mask = NULL;
                    size_t self_components = 0, self_pairs = 0;
                    IntersectionCleanupParams cleanup_params;
                    IntersectionCleanupStats cleanup_stats;
                    DiskTopologyRepairStats recut_stats;
                    size_t self_nf_before = n_unique_faces;
                    size_t self_pinch_splits = 0;

                    memset(&cleanup_stats, 0, sizeof cleanup_stats);
                    memset(&recut_stats, 0, sizeof recut_stats);
                    if (source_self_conflict_mask(
                            out_verts, out_nv, flat_faces, n_unique_faces,
                            &self_mask, &self_components, &self_pairs) != 0) {
                        free(self_mask);
                        fprintf(stderr,
                                "ERROR: source self-contact audit failed\n");
                        RAISE(IO_Failed);
                    }
                    if (self_components > 0) {
                        IntersectionCleanup_default_params(&cleanup_params);
                        cleanup_params.gap_max = 0.0;
                        cleanup_params.include_hinges = 0;
                        if (IntersectionCleanup_process(
                                out_verts, out_nv, flat_faces,
                                &n_unique_faces, self_mask,
                                &cleanup_params, &cleanup_stats) != 0 ||
                            cleanup_stats.faces_deleted == 0) {
                            free(self_mask);
                            fprintf(stderr,
                                    "ERROR: source self-contact cleanup "
                                    "rejected (%zu chart(s), %zu pair(s), "
                                    "%zu face(s) selected, budget=%zu)\n",
                                    self_components, self_pairs,
                                    cleanup_stats.faces_deleted,
                                    cleanup_stats.delete_budget);
                            RAISE(IO_Failed);
                        }
                        free(self_mask);
                        self_mask = NULL;
                        if (source_depinch_with_metadata(
                                arena, &out_verts, &out_nv,
                                flat_faces, n_unique_faces, vert_cap,
                                vert_cube_idx, vert_is_weld,
                                lineage_parent0, lineage_parent1,
                                &self_pinch_splits) != 0) {
                            fprintf(stderr,
                                    "ERROR: source self-contact depinch "
                                    "transaction failed (capacity=%zu)\n",
                                    vert_cap);
                            RAISE(IO_Failed);
                        }
                        if (source_disk_repair_with_metadata(
                                arena, &out_verts, &out_nv,
                                &flat_faces, &n_unique_faces, vert_cap,
                                vert_cube_idx, vert_is_weld,
                                lineage_parent0, lineage_parent1,
                                &recut_stats) != 0) {
                            fprintf(stderr,
                                    "ERROR: source self-contact disk recut "
                                    "failed (%zu failures, %zu faces)\n",
                                    recut_stats.failed_components,
                                    n_unique_faces);
                            RAISE(IO_Failed);
                        }
                        color_nv = out_nv;
                        lineage_initialized_nv = out_nv;
                        audit_source_lineage_positions(
                            "after-source-self-contact-repair",
                            out_verts, out_nv,
                            lineage_parent0, lineage_parent1,
                            lineage_source_reference,
                            lineage_source_reference_nv,
                            lineage_cube_voff, &cubes);
                        fprintf(stderr,
                                "  source self-contact repair: %zu chart(s), "
                                "%zu exact pair(s), %zu face(s) shelled "
                                "(%zu -> %zu), %zu pinch split(s); "
                                "disk recut repaired %zu "
                                "component(s) with %zu seam edge(s)\n",
                                self_components, self_pairs,
                                cleanup_stats.faces_deleted,
                                self_nf_before, n_unique_faces,
                                self_pinch_splits,
                                recut_stats.repaired_components,
                                recut_stats.seam_edges);
                    } else {
                        free(self_mask);
                    }
                }
                gw_phase("sourcecut");
            }
            if (!no_holefill) {
                size_t source_nv = out_nv;
                size_t source_loops = 0, source_holes = 0, source_filled = 0;
                if (HoleFill_process_ex(
                        arena, &out_verts, &flat_faces, &out_nv,
                        &n_unique_faces, NULL, 2  ,
                        &source_loops, &source_holes,
                        &source_filled) != 0) {
                    fprintf(stderr, "ERROR: source-chart hole fill failed\n");
                    RAISE(IO_Failed);
                }
                audit_source_lineage_positions(
                    "after-primary-holefill",out_verts,out_nv,
                    lineage_parent0,lineage_parent1,
                    lineage_source_reference,lineage_source_reference_nv,
                    lineage_cube_voff,&cubes);
                {
                    ComponentMesh source_mesh;
                    size_t source_closed = 0, source_pinhole = 0;
                    size_t source_added = 0, source_skipped = 0;
                    size_t postclose_loops = 0, postclose_holes = 0;
                    size_t postclose_filled = 0;
                    memset(&source_mesh, 0, sizeof source_mesh);
                    source_mesh.verts = out_verts;
                    source_mesh.faces = flat_faces;
                    source_mesh.nv = out_nv;
                    source_mesh.nf = n_unique_faces;
                    source_mesh.comp_id = 1;
                    source_mesh.self = &source_mesh;
                    if (!source_disk_cut)
                        PinholeFill_close_bowties(arena, &source_mesh, 1,
                                                  &source_closed);
                    if (source_closed > 0 && HoleFill_process_ex(
                            arena, &source_mesh.verts, &source_mesh.faces,
                            &source_mesh.nv, &source_mesh.nf,
                            NULL, 2  ,
                            &postclose_loops, &postclose_holes,
                            &postclose_filled) != 0) {
                        fprintf(stderr,
                                "ERROR: post-bowtie source hole fill failed\n");
                        RAISE(IO_Failed);
                    }
                    if (!source_disk_cut)
                        PinholeFill_fill_small_loops(
                            arena, &source_mesh, 1, 0,
                            &source_pinhole, &source_added, &source_skipped);
                    out_verts = source_mesh.verts;
                    flat_faces = source_mesh.faces;
                    out_nv = source_mesh.nv;
                    n_unique_faces = source_mesh.nf;
                    source_filled += postclose_filled + source_pinhole;
                    if (source_closed > 0)
                        fprintf(stderr, "  source chart: %zu bowtie gap(s) "
                                "closed in place; post-close loops=%zu "
                                "holes=%zu filled=%zu\n", source_closed,
                                postclose_loops, postclose_holes,
                                postclose_filled);
                }
                if (out_nv > vert_cap) {
                    fprintf(stderr,
                            "grid_weld: vert capacity %zu exceeded by "
                            "source-chart fill (%zu)\n", vert_cap, out_nv);
                    RAISE(IO_Failed);
                }
                for (size_t v = source_nv; v < out_nv; v++) {
                    vert_cube_idx[v] = -1;
                    vert_is_weld[v] = 0;
                    if (lineage_parent0 && lineage_parent1)
                        lineage_parent0[v] = lineage_parent1[v] = -1;
                }
                lineage_initialized_nv = out_nv;
                audit_source_lineage_positions(
                    "before-source-certificate",out_verts,out_nv,
                    lineage_parent0,lineage_parent1,
                    lineage_source_reference,lineage_source_reference_nv,
                    lineage_cube_voff,&cubes);
                {
                    size_t source_closed_charts = 0;
                    size_t source_overlap_charts = 0;
                    size_t source_dropped_faces = 0;
                    size_t source_dropped_verts = 0;
                    if (certify_source_charts(
                            arena, out_verts, &out_nv,
                            flat_faces, &n_unique_faces, vert_cap,
                            vert_cube_idx, &cubes, vert_is_weld,
                            lineage_parent0, lineage_parent1,
                            &source_closed_charts,
                            &source_overlap_charts,
                            &source_dropped_faces,
                            &source_dropped_verts) != 0) {
                        fprintf(stderr,
                                "ERROR: source-chart disk certificate failed\n");
                        RAISE(IO_Failed);
                    }
                    if (source_closed_charts > 0 ||
                        source_overlap_charts > 0 ||
                        source_dropped_verts > 0)
                        fprintf(stderr,
                                "  source chart filter: removed %zu closed "
                                "bubble(s) + %zu exact-overlap micro-chart(s), "
                                "%zu faces, %zu unreferenced verts\n",
                                source_closed_charts, source_overlap_charts,
                                source_dropped_faces,
                                source_dropped_verts);
                    color_nv = out_nv;
                    lineage_initialized_nv = out_nv;
                    audit_source_lineage_positions(
                        "after-source-certificate",out_verts,out_nv,
                        lineage_parent0,lineage_parent1,
                        lineage_source_reference,lineage_source_reference_nv,
                        lineage_cube_voff,&cubes);
                }
                fprintf(stderr,
                        "Source-chart topology fill: %zu loops, %zu secondary "
                        "-> %zu filled; V %zu->%zu, F=%zu\n",
                        source_loops, source_holes, source_filled,
                        source_nv, out_nv, n_unique_faces);
                dump_stage(arena, dump_stages_dir, stage_prefix, 0,
                           "chartfill", out_verts, out_nv, flat_faces,
                           n_unique_faces, vert_cube_idx, cube_palette,
                           cubes.n, color_nv);
            }
            float rho = 1.5f, band = 6.0f;
            const char *e;
            if ((e = getenv("SEAM_RHO")))  rho  = (float)atof(e);
            if ((e = getenv("SEAM_BAND"))) band = (float)atof(e);
            BpaBridgeGate seam_gate;
            memset(&seam_gate, 0, sizeof seam_gate);
            if ((e = getenv("SEAM_UMBILICUS_Y"))) seam_gate.umb_y = atof(e);
            if ((e = getenv("SEAM_UMBILICUS_X"))) seam_gate.umb_x = atof(e);
            if ((e = getenv("SEAM_WRAP_PITCH"))) { double v = atof(e); if (v > 0) seam_gate.pitch = v; }
            if (axis_table.n >= 2) {
                double zmin = (double)out_verts[0], zmax = zmin;
                for (size_t v = 1; v < out_nv; v++) {
                    double z = (double)out_verts[v*3];
                    if (z < zmin) zmin = z;
                    if (z > zmax) zmax = z;
                }
                gw_axis_table_eval(&axis_table, 0.5 * (zmin + zmax),
                                   &seam_gate.umb_y, &seam_gate.umb_x);
                seam_gate.axis_z = axis_table.z;
                seam_gate.axis_y = axis_table.y;
                seam_gate.axis_x = axis_table.x;
                seam_gate.axis_n = axis_table.n;
                set_env_double("SEAM_UMBILICUS_Y", seam_gate.umb_y);
                set_env_double("SEAM_UMBILICUS_X", seam_gate.umb_x);
            }
            if (seam_gate.pitch > 0.0 &&
                (seam_gate.axis_n >= 2 || seam_gate.umb_y != 0.0 ||
                 seam_gate.umb_x != 0.0)) {
                seam_gate.tol  = SEAM_WIND_TOL_DEFAULT_TURNS;
                seam_gate.hard = SEAM_WIND_HARD_TOL_DEFAULT_TURNS;
                if ((e = getenv("SEAM_WIND_TOL")))      { double v = atof(e); if (v > 0) seam_gate.tol  = v; }
                if ((e = getenv("SEAM_WIND_HARD_TOL"))) { double v = atof(e); if (v > 0) seam_gate.hard = v; }
                if ((e = getenv("SEAM_WIND_SPAN_TOL"))) { double v = atof(e); if (v > 0) seam_gate.span = v; }
            }
            if (pair_mode) set_env_if_unset("SEAM_DUMP_FRONT", out_path);

            if (!getenv("SEAM_NO_REFINE")) {
                uint8_t *rf_used = (uint8_t *)ARENA_CALLOC(arena, out_nv, 1L);
                for (size_t f = 0; f < n_unique_faces; f++) {
                    rf_used[flat_faces[f*3+0]] = 1;
                    rf_used[flat_faces[f*3+1]] = 1;
                    rf_used[flat_faces[f*3+2]] = 1;
                }
                SeamPlane fplanes[64];
                size_t fnp = 0;
                if (explicit_nplanes > 0) {
                    fnp = explicit_nplanes;
                    memcpy(fplanes, explicit_planes,
                           fnp * sizeof(*fplanes));
                    fprintf(stderr,
                        "Seam refine: using %zu explicit streaming plane(s)\n",
                        fnp);
                } else {
                    fnp = SeamPlanes_detect(out_verts, out_nv, rf_used,
                                            (double)seam_cube, (double)band,
                                            fplanes, 64);
                }
                if (fnp > 0) {
                    SeamRefineParams sp; SeamRefine_default_params(&sp);
                    sp.band = band;
                    if ((e = getenv("SEAM_REFINE_TARGET"))) {
                        double v = atof(e); if (v > 0) sp.target_len = (float)v;
                    }
                    if ((e = getenv("SEAM_REFINE_FLIP_ROUNDS"))) {
                        int v = atoi(e); if (v >= 0) sp.flip_max_rounds = v;
                    }
                    SeamRefineStats sst;
                    float *rf_v = NULL; int32_t *rf_f = NULL;
                    int32_t *rf_src = NULL, *rf_parent1 = NULL;
                    size_t rf_nv = 0, rf_nf = 0, rf_nnew = 0;
                    if (SeamRefine_process_with_parents(
                                           arena, out_verts, out_nv,
                                           flat_faces, n_unique_faces,
                                           fplanes, fnp, &sp,
                                           &rf_v, &rf_nv, &rf_f, &rf_nf,
                                           &rf_src, &rf_parent1,
                                           &rf_nnew, &sst) == 0
                        && rf_nnew > 0) {
                        IntersectionCleanupParams rip;
                        IntersectionCleanupStats ria;
                        IntersectionCleanup_default_params(&rip);
                        rip.gap_max = 0.0;
                        rip.include_hinges = 0;
                        if (IntersectionCleanup_audit(
                                rf_v, rf_nv, rf_f, rf_nf, NULL, &rip,
                                NULL, &ria) != 0) {
                            fprintf(stderr,
                                    "ERROR: seam-refine embedding audit failed\n");
                            RAISE(IO_Failed);
                        }
                        if (ria.conflicts > 0) {
                            SeamRefineParams safe_sp = sp;
                            SeamRefineStats safe_st;
                            float *safe_v = NULL;
                            int32_t *safe_f = NULL, *safe_src = NULL;
                            int32_t *safe_parent1 = NULL;
                            int32_t *safe_face_source = NULL;
                            size_t safe_nv = 0, safe_nf = 0, safe_nnew = 0;
                            size_t rejected_conflicts = ria.conflicts;
                            size_t protected_faces = 0;
                            uint8_t *freeze_source = (uint8_t *)ARENA_CALLOC(
                                arena, n_unique_faces, 1L);
                            size_t *safe_degree = NULL;
                            int rollback_round;
                            safe_sp.flip_max_rounds = 0;
                            if (SeamRefine_process_masked_with_roots(
                                    arena, out_verts, out_nv,
                                    flat_faces, n_unique_faces,
                                    fplanes, fnp, &safe_sp,
                                    NULL, 0,
                                    &safe_v, &safe_nv, &safe_f, &safe_nf,
                                    &safe_src, &safe_parent1,
                                    &safe_face_source,
                                    &safe_nnew, &safe_st) != 0 ||
                                safe_nnew == 0) {
                                fprintf(stderr,
                                        "ERROR: seam-refine ancestry replay "
                                        "failed\n");
                                RAISE(IO_Failed);
                            }
                            safe_degree = (size_t *)ARENA_CALLOC(
                                arena, safe_nf,
                                sizeof(*safe_degree));
                            if (
                                IntersectionCleanup_audit(
                                    safe_v, safe_nv, safe_f, safe_nf, NULL,
                                    &rip, safe_degree, &ria) != 0) {
                                fprintf(stderr,
                                        "ERROR: seam-refine ancestry audit "
                                        "failed\n");
                                RAISE(IO_Failed);
                            }
                            for (rollback_round = 0;
                                 ria.conflicts > 0 && rollback_round < 8;
                                 rollback_round++) {
                                size_t added = 0;
                                for (size_t sf = 0; sf < safe_nf; sf++) {
                                    int32_t root;
                                    if (safe_degree[sf] == 0) continue;
                                    root = safe_face_source[sf];
                                    if (root < 0 ||
                                        (size_t)root >= n_unique_faces)
                                        continue;
                                    if (!freeze_source[(size_t)root]) {
                                        freeze_source[(size_t)root] = 1;
                                        protected_faces++;
                                        added++;
                                    }
                                }
                                if (added == 0) break;
                                if (SeamRefine_process_masked_with_roots(
                                        arena, out_verts, out_nv,
                                        flat_faces, n_unique_faces,
                                        fplanes, fnp, &safe_sp,
                                        freeze_source, n_unique_faces,
                                        &safe_v, &safe_nv,
                                        &safe_f, &safe_nf,
                                        &safe_src, &safe_parent1,
                                        &safe_face_source,
                                        &safe_nnew, &safe_st) != 0) {
                                    fprintf(stderr,
                                            "ERROR: seam-refine protected "
                                            "replay failed\n");
                                    RAISE(IO_Failed);
                                }
                                safe_degree = (size_t *)ARENA_CALLOC(
                                    arena, safe_nf,
                                    sizeof(*safe_degree));
                                if (IntersectionCleanup_audit(
                                        safe_v, safe_nv, safe_f, safe_nf,
                                        NULL, &rip, safe_degree, &ria) != 0) {
                                    fprintf(stderr,
                                            "ERROR: seam-refine protected "
                                            "audit failed\n");
                                    RAISE(IO_Failed);
                                }
                            }
                            if (ria.conflicts != 0) {
                                dump_stage(
                                    arena, dump_stages_dir, stage_prefix, 1,
                                    "refine_rejected", safe_v, safe_nv,
                                    safe_f, safe_nf, vert_cube_idx,
                                    cube_palette, cubes.n, color_nv);
                                fprintf(stderr,
                                        "ERROR: seam-refine chart rollback "
                                        "failed (%zu residual conflict(s))\n",
                                        ria.conflicts);
                                RAISE(IO_Failed);
                            }

                            if (sp.flip_max_rounds > 0) {
                                SeamRefineStats quality_st;
                                float *quality_v = NULL;
                                int32_t *quality_f = NULL;
                                int32_t *quality_src = NULL;
                                int32_t *quality_parent1 = NULL;
                                size_t quality_nv = 0, quality_nf = 0;
                                size_t quality_nnew = 0;
                                if (SeamRefine_process_masked_with_roots(
                                        arena, out_verts, out_nv,
                                        flat_faces, n_unique_faces,
                                        fplanes, fnp, &sp,
                                        freeze_source, n_unique_faces,
                                        &quality_v, &quality_nv,
                                        &quality_f, &quality_nf,
                                        &quality_src, &quality_parent1,
                                        NULL, &quality_nnew,
                                        &quality_st) == 0 &&
                                    quality_nnew > 0 &&
                                    IntersectionCleanup_audit(
                                        quality_v, quality_nv,
                                        quality_f, quality_nf, NULL,
                                        &rip, NULL, &ria) == 0 &&
                                    ria.conflicts == 0) {
                                    safe_v = quality_v;
                                    safe_nv = quality_nv;
                                    safe_f = quality_f;
                                    safe_nf = quality_nf;
                                    safe_src = quality_src;
                                    safe_parent1 = quality_parent1;
                                    safe_nnew = quality_nnew;
                                    safe_st = quality_st;
                                } else {
                                    fprintf(stderr,
                                            "Seam refine: protected quality "
                                            "flips were not embedded; keeping "
                                            "the certified no-flip replay\n");
                                }
                            }
                            fprintf(stderr,
                                    "Seam refine: rolled back %zu complete "
                                    "source face(s) after %zu exact "
                                    "conflict(s); embedded replay accepted\n",
                                    protected_faces, rejected_conflicts);
                            rf_v = safe_v; rf_nv = safe_nv;
                            rf_f = safe_f; rf_nf = safe_nf;
                            rf_src = safe_src; rf_parent1 = safe_parent1;
                            rf_nnew = safe_nnew;
                            sst = safe_st;
                        }
                        if (rf_nv > vert_cap) {
                            fprintf(stderr, "grid_weld: vert capacity %zu "
                                    "exceeded by seam refine (%zu)\n",
                                    vert_cap, rf_nv);
                            RAISE(IO_Failed);
                        }
                        for (size_t v2 = 0; v2 < rf_nnew; v2++)
                            vert_cube_idx[out_nv + v2] =
                                vert_cube_idx[rf_src[v2]];
                        if (lineage_parent0 && lineage_parent1) {
                            for (size_t v2 = 0; v2 < rf_nnew; v2++) {
                                lineage_parent0[out_nv + v2] = rf_src[v2];
                                lineage_parent1[out_nv + v2] = rf_parent1[v2];
                            }
                            lineage_initialized_nv = rf_nv;
                        }
                        out_verts = rf_v; out_nv = rf_nv;
                        flat_faces = rf_f; n_unique_faces = rf_nf;
                        color_nv = out_nv;
                        fprintf(stderr,
                            "Seam refine: %zu bnd + %zu int split(s), %zu "
                            "flip(s), %zu round(s) -> +%zu verts +%zu faces "
                            "(target %.1f vox, %zu plane(s))\n",
                            sst.bnd_splits, sst.int_splits, sst.flips,
                            sst.rounds, sst.verts_added, sst.faces_added,
                            (double)sp.target_len, fnp);
                    }
                }
                gw_phase("refine");
                dump_stage(arena, dump_stages_dir, stage_prefix, 1, "refine",
                           out_verts, out_nv, flat_faces, n_unique_faces,
                           vert_cube_idx, cube_palette, cubes.n, color_nv);
            }

            fprintf(stderr,
                "Seam-weld: cube=%.0f rho=%.2f band=%.2f over %zu faces\n",
                (double)seam_cube, (double)rho, (double)band,
                n_unique_faces);

            int reweld_off = (getenv("SEAM_REWELD") == NULL);
            int32_t *vert_sheet = NULL; size_t n_sheets = 0;
            if (!reweld_off)
                SheetReweld_label(arena, flat_faces, n_unique_faces, out_nv,
                                  &vert_sheet, &n_sheets);

            TopologyAuditReport prebridge_topology;
            memset(&prebridge_topology, 0, sizeof prebridge_topology);
            if (TopologyAudit_analyze(out_verts, out_nv, flat_faces,
                                      n_unique_faces, NULL,
                                      &prebridge_topology) != 0 ||
                !prebridge_topology.all_components_are_disks) {
                fprintf(stderr,
                        "ERROR: pre-bridge chart certificate failed "
                        "(%zu/%zu disks, invalid=%zu, beta1=%lld)\n",
                        prebridge_topology.disk_components,
                        prebridge_topology.face_components,
                        prebridge_topology.invalid_surface_components,
                        (long long)prebridge_topology.beta_1);
                TopologyAudit_dispose(&prebridge_topology);
                RAISE(IO_Failed);
            }

            if (stream_shard_out != NULL) {
                if (cubes_loaded != 1 || cubes.n != 1 ||
                    lineage_parent0 == NULL || lineage_parent1 == NULL ||
                    StreamWeld_write_shard(
                        stream_shard_out, cubes.ids[0], out_verts, out_nv,
                        flat_faces, n_unique_faces,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        lineage_parent0, lineage_parent1) != 0) {
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                        "ERROR: cannot write complete streaming shard %s\n",
                        stream_shard_out);
                    RAISE(IO_Failed);
                }
                fprintf(stderr,
                    "Wrote streaming shard %s (%zu verts, %zu faces, "
                    "%zu certified disk chart(s))\n",
                    stream_shard_out, out_nv, n_unique_faces,
                    prebridge_topology.face_components);
                TopologyAudit_dispose(&prebridge_topology);
                ok = 1;
                goto grid_weld_complete;
            }

            int32_t *sw_faces = NULL; size_t sw_nf = 0, n_bridge = 0;
            size_t bridge_input_nf = 0;
            size_t bridge_support_begin = 0, bridge_support_end = 0;
            if (legacy_bpa) {
            SeamWeld_bridge(arena, out_verts, out_nv, flat_faces,
                            n_unique_faces, seam_cube, rho, 0.0f, band,
                            NULL  ,
                            &seam_gate,
                            &sw_faces, &sw_nf, &n_bridge);
            if (n_bridge > sw_nf) {
                TopologyAudit_dispose(&prebridge_topology);
                fprintf(stderr, "ERROR: seam bridge count exceeds output\n");
                RAISE(IO_Failed);
            }
            bridge_input_nf = sw_nf - n_bridge;
            fprintf(stderr, "  bridge: %zu -> %zu faces (+%zu bridge)\n",
                    n_unique_faces, sw_nf, n_bridge);
            if (n_bridge > 0 &&
                getenv("SEAM_NO_BRIDGE_BOWTIE_CLOSE") == NULL) {
                ComponentMesh bridge_mesh;
                size_t bridge_bowties = 0;
                size_t bridge_before = n_bridge;
                memset(&bridge_mesh, 0, sizeof bridge_mesh);
                bridge_mesh.verts = out_verts;
                bridge_mesh.nv = out_nv;
                bridge_mesh.faces = sw_faces + bridge_input_nf*3;
                bridge_mesh.nf = n_bridge;
                bridge_mesh.comp_id = 1;
                bridge_mesh.self = &bridge_mesh;
                if (PinholeFill_close_bridge_bowties(
                        arena, &bridge_mesh, sw_faces, bridge_input_nf,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        getenv("SEAM_NO_CHART_FOREST")==NULL,
                        &bridge_bowties) != 0) {
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                            "ERROR: raw bridge chart gap closure failed\n");
                    RAISE(IO_Failed);
                }
                if (bridge_mesh.nf != bridge_before) {
                    int32_t *closed_faces = (int32_t *)ARENA_ALLOC(
                        arena, ((bridge_input_nf + bridge_mesh.nf) * 3 *
                                      sizeof(int32_t)));
                    memcpy(closed_faces, sw_faces,
                           bridge_input_nf * 3 * sizeof(int32_t));
                    memcpy(closed_faces + bridge_input_nf*3,
                           bridge_mesh.faces,
                           bridge_mesh.nf * 3 * sizeof(int32_t));
                    sw_faces = closed_faces;
                    n_bridge = bridge_mesh.nf;
                    sw_nf = bridge_input_nf + n_bridge;
                }
                fprintf(stderr,
                        "  raw bridge chart closure: %zu coherent bowtie(s), "
                        "+%zu face(s); no vertex splits\n",
                        bridge_bowties, n_bridge - bridge_before);
                {
                    const char *closed_dump =
                        getenv("SEAM_DUMP_CLOSED_BRIDGE");
                    if (closed_dump != NULL) {
                        FILE *bf = fopen(closed_dump, "w");
                        if (bf != NULL) {
                            for (size_t v = 0; v < out_nv; v++)
                                fprintf(bf, "v %.9g %.9g %.9g\n",
                                        (double)out_verts[v*3],
                                        (double)out_verts[v*3+1],
                                        (double)out_verts[v*3+2]);
                            for (size_t f = 0; f < n_bridge; f++) {
                                const int32_t *tri =
                                    &sw_faces[(bridge_input_nf+f)*3];
                                fprintf(bf, "f %d %d %d\n",
                                        tri[0]+1,tri[1]+1,tri[2]+1);
                            }
                            fclose(bf);
                            fprintf(stderr,
                                    "  [SEAM_DUMP_CLOSED_BRIDGE] wrote %s "
                                    "(%zu bridge faces)\n",
                                    closed_dump,n_bridge);
                        }
                    }
                }
            }
            if(n_bridge>0&&getenv("SEAM_NO_SOURCE_CHART_ORIENT")==NULL){
                ChartBridgeOrientationStats orientation_stats;
                if(ChartBridgeForest_orient_source(
                        arena,out_verts,out_nv,sw_faces,sw_nf,
                        bridge_input_nf,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        &orientation_stats)!=0){
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                            "ERROR: source-chart bridge orientation failed\n");
                    RAISE(IO_Failed);
                }
                fprintf(stderr,
                    "  source-chart bridge orientation: %zu candidate patch(es), "
                    "%zu constraint(s), %zu conflict(s); flipped %zu chart(s), "
                    "%zu source face(s)\n",
                    orientation_stats.candidate_patches,
                    orientation_stats.constraints_used,
                    orientation_stats.constraints_conflicted,
                    orientation_stats.charts_flipped,
                    orientation_stats.faces_flipped);
            }
            if(n_bridge>0&&
               getenv("SEAM_NO_BRIDGE_ATTACHMENT_CLOSE")==NULL){
                ComponentMesh combined_mesh;
                size_t *chart_face_count;
                size_t attachment_closed=0;
                size_t attachment_before=sw_nf;
                bridge_support_begin=n_bridge;
                memset(&combined_mesh,0,sizeof combined_mesh);
                combined_mesh.verts=out_verts;
                combined_mesh.nv=out_nv;
                combined_mesh.faces=sw_faces;
                combined_mesh.nf=sw_nf;
                combined_mesh.comp_id=1;
                combined_mesh.self=&combined_mesh;
                chart_face_count=(size_t*)ARENA_ALLOC(
                    arena,(prebridge_topology.face_components*
                                 sizeof(*chart_face_count)));
                for(size_t c=0;c<prebridge_topology.face_components;c++)
                    chart_face_count[c]=prebridge_topology.component[c].faces;
                if(PinholeFill_close_bridge_attachments(
                        arena,&combined_mesh,bridge_input_nf,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        chart_face_count,
                        getenv("SEAM_NO_CHART_FOREST")==NULL,
                        &attachment_closed)!=0){
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                            "ERROR: raw bridge attachment closure failed\n");
                    RAISE(IO_Failed);
                }
                sw_faces=combined_mesh.faces;
                sw_nf=combined_mesh.nf;
                n_bridge=sw_nf-bridge_input_nf;
                bridge_support_end=n_bridge;
                fprintf(stderr,
                        "  raw bridge attachment closure: %zu coherent "
                        "point contact(s), +%zu face(s); chart-pair exact\n",
                        attachment_closed,sw_nf-attachment_before);
            }
            if(n_bridge>0&&getenv("SEAM_NO_BRIDGE_TRI_CLOSE")==NULL){
                ComponentMesh combined_mesh;
                size_t tri_loops=0,tri_added=0,tri_skipped=0;
                memset(&combined_mesh,0,sizeof combined_mesh);
                combined_mesh.verts=out_verts;
                combined_mesh.nv=out_nv;
                combined_mesh.faces=sw_faces;
                combined_mesh.nf=sw_nf;
                combined_mesh.comp_id=1;
                combined_mesh.self=&combined_mesh;
                if(PinholeFill_fill_small_loops(
                        arena,&combined_mesh,1,0,&tri_loops,&tri_added,
                        &tri_skipped)!=0){
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                            "ERROR: raw bridge triangular closure failed\n");
                    RAISE(IO_Failed);
                }
                sw_faces=combined_mesh.faces;
                sw_nf=combined_mesh.nf;
                n_bridge=sw_nf-bridge_input_nf;
                fprintf(stderr,
                        "  raw bridge triangular closure: %zu exact loop(s), "
                        "+%zu face(s), %zu skipped; no new edges\n",
                        tri_loops,tri_added,tri_skipped);
            }
            if (!getenv("SEAM_NO_CHART_FOREST")) {
                int32_t *forest_faces = NULL;
                size_t forest_nf = 0;
                ChartBridgeForestStats forest_stats;
                if (ChartBridgeForest_filter_with_support(
                        arena, out_verts, out_nv, sw_faces, sw_nf,
                        bridge_input_nf,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        bridge_support_begin,bridge_support_end,
                        &forest_faces, &forest_nf,
                        &forest_stats) != 0) {
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr, "ERROR: chart bridge transaction failed\n");
                    RAISE(IO_Failed);
                }
                sw_faces = forest_faces;
                sw_nf = forest_nf;
                n_bridge = sw_nf - bridge_input_nf;
                fprintf(stderr,
                        "  chart bridge forest: %zu/%zu patch(es), "
                        "%zu/%zu bridge faces (join=%zu grow=%zu hole=%zu); "
                        "reject[self/mixed=%zu patch=%zu attach=%zu "
                        "multipath=%zu edge=%zu orient=%zu]; "
                        "source-shoulder=%zu; "
                        "promotion[rounds=%zu patches=%zu faces=%zu]\n",
                        forest_stats.accepted_patches,
                        forest_stats.patches,
                        forest_stats.bridge_faces_kept,
                        forest_stats.bridge_faces_in,
                        forest_stats.accepted_join_patches,
                        forest_stats.accepted_growth_patches,
                        forest_stats.accepted_hole_patches,
                        forest_stats.rejected_self_or_mixed,
                        forest_stats.rejected_patch_topology,
                        forest_stats.rejected_attachment,
                        forest_stats.rejected_cycle,
                        forest_stats.rejected_edge_conflict,
                        forest_stats.rejected_orientation,
                        forest_stats.accepted_source_shoulder_faces,
                        forest_stats.promotion_rounds,
                        forest_stats.promoted_patches,
                        forest_stats.promoted_faces);
                {
                    TopologyAuditReport forest_topology;
                    memset(&forest_topology, 0, sizeof forest_topology);
                    int forest_genus_zero = 1;
                    if (TopologyAudit_analyze(out_verts, out_nv,
                                              sw_faces, sw_nf, NULL,
                                              &forest_topology) != 0) {
                        forest_genus_zero = 0;
                    } else {
                        for (size_t tc = 0;
                             tc < forest_topology.face_components; tc++) {
                            const TopologyComponentInvariant *ti =
                                &forest_topology.component[tc];
                            if (!ti->surface_valid || !ti->orientable ||
                                ti->orientable_genus != 0) {
                                forest_genus_zero = 0;
                                break;
                            }
                        }
                    }
                    if (!forest_genus_zero) {
                        fprintf(stderr,
                                "ERROR: chart bridge forest violated "
                                "genus-zero surface "
                                "topology (%zu/%zu disks, invalid=%zu, "
                                "beta1=%lld)\n",
                                forest_topology.disk_components,
                                forest_topology.face_components,
                                forest_topology.invalid_surface_components,
                                (long long)forest_topology.beta_1);
                        TopologyAudit_dispose(&forest_topology);
                        TopologyAudit_dispose(&prebridge_topology);
                        RAISE(IO_Failed);
                    }
                    fprintf(stderr,
                            "  chart bridge certificate: PASS (genus zero, "
                            "%zu/%zu already disks, beta1=%lld)\n",
                            forest_topology.disk_components,
                            forest_topology.face_components,
                            (long long)forest_topology.beta_1);
                    TopologyAudit_dispose(&forest_topology);
                }
            }
            TopologyAudit_dispose(&prebridge_topology);
            gw_phase("bridge");

            size_t n_bridge_total = n_bridge;
            if (!reweld_off && vert_sheet && n_sheets >= 2) {
                SheetReweldParams rwp;
                SheetReweld_default_params(&rwp);
                rwp.band = band;
                if ((e = getenv("SHEET_REWELD_RHOMAX"))) {
                    float v = (float)atof(e); if (v > 0.0f) rwp.rho_max = v;
                }
                int32_t *rw_faces = NULL; size_t rw_nf = 0;
                SheetReweldStats rwst;
                SheetReweld_process(arena, out_verts, out_nv, sw_faces, sw_nf,
                                    n_bridge, vert_sheet, n_sheets,
                                    seam_cube, &seam_gate, &rwp,
                                    &rw_faces, &rw_nf, &rwst);
                fprintf(stderr, "  reweld(phase2): %zu sheets, %zu candidate "
                        "pair(s), %zu confirmed, +%zu faces -> %zu\n",
                        rwst.n_sheets, rwst.n_candidates, rwst.n_pairs,
                        rwst.n_faces_added, rw_nf);
                sw_faces = rw_faces; sw_nf = rw_nf;
                n_bridge_total = n_bridge + rwst.n_faces_added;
            }
            if (pair_mode && n_bridge == 0)
                fprintf(stderr,
                    "WARNING: 0 bridge faces -- detect_planes found no shared "
                    "seam plane between the two components. Check the bboxes "
                    "above: they must straddle a multiple of %.0f on one axis "
                    "(see the front_edges.obj dump for the boundary edges).\n",
                    (double)seam_cube);

            for (size_t f = sw_nf - n_bridge_total; f < sw_nf; f++) {
                for (int k = 0; k < 3; k++) {
                    int32_t vv = sw_faces[f * 3 + k];
                    if (vv >= 0 && (size_t)vv < out_nv) vert_is_weld[vv] = 1;
                }
            }

            dump_stage(arena, dump_stages_dir, stage_prefix, 2, "bridge",
                       out_verts, out_nv, sw_faces, sw_nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            size_t sw_comp = 0, sw_flipped = 0, sw_resid = 0;
            OrientMesh_consistent(arena, out_verts, out_nv, NULL,
                                  sw_faces, sw_nf,
                                  &sw_flipped, &sw_comp, &sw_resid);
            fprintf(stderr, "  pre-orient: %zu components, %zu flipped, "
                    "%zu residual same_dir\n", sw_comp, sw_flipped, sw_resid);
            gw_phase("preorient");
            dump_stage(arena, dump_stages_dir, stage_prefix, 3, "preorient",
                       out_verts, out_nv, sw_faces, sw_nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            ComponentMesh cm;
            memset(&cm, 0, sizeof cm);
            cm.verts = out_verts; cm.faces = sw_faces;
            cm.nv = out_nv; cm.nf = sw_nf; cm.comp_id = 1; cm.self = &cm;

            size_t fold_removed = 0;
            FoldCleanup_process(arena, &cm, 1, 8, &fold_removed);
            fprintf(stderr, "  foldcleanup: removed %zu fold faces -> %zu faces\n",
                    fold_removed, cm.nf);
            gw_phase("foldclean");
            dump_stage(arena, dump_stages_dir, stage_prefix, 4, "foldcleanup",
                       cm.verts, cm.nv, cm.faces, cm.nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            if (ChartLineage_capture(arena, cm.verts, cm.nv,
                                     cm.faces, cm.nf,
                                     &chart_lineage) != 0) {
                fprintf(stderr, "ERROR: pre-split chart lineage capture failed\n");
                RAISE(IO_Failed);
            }
            have_chart_lineage = 1;
            fprintf(stderr, "  chart lineage: %zu pre-split chart(s), "
                    "%zu unique source position(s)\n",
                    chart_lineage.n_components, chart_lineage.n_points);

            size_t closed = 0, filled = 0, added = 0, skipped = 0;
            if (!no_pinhole)
                PinholeFill_close_bowties(arena, &cm, 1, &closed);
            fprintf(stderr,
                "  chart gap closure: bowties=%zu filled=%zu tris+=%zu skipped=%zu"
                " -> %zu faces, %zu verts\n",
                closed, filled, added, skipped, cm.nf, cm.nv);
            gw_phase("chartclose");
            dump_stage(arena, dump_stages_dir, stage_prefix, 5, "pinhole",
                       cm.verts, cm.nv, cm.faces, cm.nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            size_t culled_nf = cm.nf, culled_comps = 0;
            size_t culled = cull_tiny_components(arena, cm.faces, cm.nf, 16,
                                                 &culled_nf, &culled_comps);
            cm.nf = culled_nf;
            fprintf(stderr, "  cull: removed %zu faces in %zu tiny comps -> %zu faces\n",
                    culled, culled_comps, cm.nf);
            gw_phase("cull");
            dump_stage(arena, dump_stages_dir, stage_prefix, 6, "cull",
                       cm.verts, cm.nv, cm.faces, cm.nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            if (!no_cleanup) {
                WeldCleanupParams wcp;
                WeldCleanupStats wcs;
                WeldCleanup_default_params(&wcp);
                WeldCleanup_process(arena, &cm, &wcp, &wcs);
                fprintf(stderr,
                    "  cleanup: %zu flips, %zu collapses -> %zu faces"
                    " (sliver/degenerate targets %zu -> %zu)\n",
                    wcs.n_flips, wcs.n_collapses, cm.nf,
                    wcs.targets_in, wcs.targets_out);
            }
            gw_phase("cleanup");

            dump_stage(arena, dump_stages_dir, stage_prefix, 7, "cleanup",
                       cm.verts, cm.nv, cm.faces, cm.nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            if (!no_holefill) {
                size_t hl_loops = 0, hl_interior = 0, hl_filled = 0;
                HoleFill_process_ex(arena, &cm.verts, &cm.faces, &cm.nv, &cm.nf,
                                    NULL, 2  ,
                                    &hl_loops, &hl_interior, &hl_filled);
                fprintf(stderr,
                    "  holefill: %zu loops, %zu interior -> %zu filled"
                    " -> %zu faces, %zu verts\n",
                    hl_loops, hl_interior, hl_filled, cm.nf, cm.nv);
            }
            gw_phase("holefill");

            dump_stage(arena, dump_stages_dir, stage_prefix, 8, "holefill",
                       cm.verts, cm.nv, cm.faces, cm.nf,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);

            flat_faces = cm.faces;
            n_unique_faces = cm.nf;
            out_verts = cm.verts;
            out_nv = cm.nv;
            } else {
                ChartZipperParams zp;
                ChartZipperStats zs;
                TopologyAuditReport zipper_topology;
                size_t zip_comp = 0, zip_flipped = 0, zip_resid = 0;

                ChartZipper_default_params(&zp);
                zp.cube_size = seam_cube;
                zp.band = band;
                zp.trace = (getenv("SEAM_ZIP_TRACE") != NULL);
                if ((e = getenv("SEAM_ZIP_MAX_EDGE"))) {
                    double v = atof(e); if (v > 0.0) zp.max_cross_edge = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_NORMAL_DOT"))) {
                    double v = atof(e); if (v >= 0.0 && v <= 1.0)
                        zp.normal_dot_min = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_MIN_COVERAGE"))) {
                    double v = atof(e); if (v > 0.0 && v <= 1.0)
                        zp.min_coverage = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_MIN_SUPPORT"))) {
                    double v = atof(e); if (v > 0.0)
                        zp.min_support_length = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_AMBIGUITY"))) {
                    double v = atof(e); if (v > 0.0 && v <= 1.0)
                        zp.ambiguity_ratio = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_MIN_ALTITUDE"))) {
                    double v = atof(e); if (v >= 0.0)
                        zp.min_triangle_altitude = (float)v;
                }
                if ((e = getenv("SEAM_ZIP_CONFLICT_GAP"))) {
                    double v = atof(e); if (v >= 0.0)
                        zp.conflict_gap = (float)v;
                }

                if (ChartZipper_process(
                        arena, out_verts, out_nv, flat_faces,
                        n_unique_faces,
                        prebridge_topology.vertex_component,
                        prebridge_topology.face_components,
                        &seam_gate, &zp,
                        &sw_faces, &sw_nf, &n_bridge, &zs) != 0) {
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr, "ERROR: chart zipper transaction failed\n");
                    RAISE(IO_Failed);
                }
                bridge_input_nf = n_unique_faces;
                fprintf(stderr,
                        "  chart zipper: %zu plane(s), %zu seam boundary "
                        "edge(s), %zu open path(s); %zu candidate(s), "
                        "%zu accepted -> +%zu face(s)\n",
                        zs.planes, zs.seam_boundary_edges, zs.open_chains,
                        zs.candidates, zs.accepted_transactions,
                        zs.bridge_faces);
                fprintf(stderr,
                        "    reject: ambiguous=%zu duplicate=%zu cycle=%zu "
                        "overlap=%zu geometry=%zu conflict_tx=%zu "
                        "(audit_pairs=%zu postcompact=%zu/%zu) "
                        "balanced=%zu trimmed=%zu; "
                        "closed=%zu opened=%zu branched=%zu\n",
                        zs.ambiguous_candidates,
                        zs.duplicate_pair_candidates,
                        zs.cycle_candidates, zs.overlap_candidates,
                        zs.geometry_candidates, zs.conflict_transactions,
                        zs.conflict_pairs,
                        zs.postcompact_conflict_transactions,
                        zs.postcompact_conflict_pairs,
                        zs.balanced_candidates,
                        zs.trimmed_candidates,
                        zs.closed_chains_rejected,
                        zs.closed_chains_opened,
                        zs.branched_chains_rejected);
                if (pair_mode && n_bridge == 0)
                    fprintf(stderr,
                            "WARNING: chart zipper found no supported boundary "
                            "interval across a shared seam plane.\n");
                if(!zs.embedded_certificate){
                    TopologyAudit_dispose(&prebridge_topology);
                    fprintf(stderr,
                            "ERROR: chart zipper embedded-geometry "
                            "certificate missing\n");
                    RAISE(IO_Failed);
                }

                for (size_t f = bridge_input_nf; f < sw_nf; f++) {
                    for (int k = 0; k < 3; k++) {
                        int32_t vv = sw_faces[f*3+k];
                        if (vv >= 0 && (size_t)vv < out_nv)
                            vert_is_weld[vv] = 1;
                    }
                }
                dump_stage(arena, dump_stages_dir, stage_prefix, 2,
                           "zipper", out_verts, out_nv, sw_faces, sw_nf,
                           vert_cube_idx, cube_palette, cubes.n, color_nv);

                OrientMesh_consistent(arena, out_verts, out_nv, NULL,
                                      sw_faces, sw_nf,
                                      &zip_flipped, &zip_comp, &zip_resid);

                memset(&zipper_topology, 0, sizeof zipper_topology);
                if (TopologyAudit_analyze(out_verts, out_nv,
                                          sw_faces, sw_nf, NULL,
                                          &zipper_topology) != 0 ||
                    !zipper_topology.all_components_are_disks ||
                    zipper_topology.inconsistent_winding_components != 0 ||
                    zipper_topology.face_components +
                        zs.accepted_transactions !=
                        prebridge_topology.face_components) {
                    fprintf(stderr,
                            "ERROR: chart zipper disk certificate failed "
                            "(cc %zu - %zu != %zu, disks=%zu, invalid=%zu, "
                            "beta1=%lld, winding=%zu)\n",
                            prebridge_topology.face_components,
                            zs.accepted_transactions,
                            zipper_topology.face_components,
                            zipper_topology.disk_components,
                            zipper_topology.invalid_surface_components,
                            (long long)zipper_topology.beta_1,
                            zipper_topology.inconsistent_winding_components);
                    TopologyAudit_dispose(&zipper_topology);
                    TopologyAudit_dispose(&prebridge_topology);
                    RAISE(IO_Failed);
                }
                fprintf(stderr,
                        "  zipper disk certificate: PASS (%zu transaction(s), "
                        "%zu/%zu disks, %zu face flip(s), residual=%zu)\n",
                        zs.accepted_transactions,
                        zipper_topology.disk_components,
                        zipper_topology.face_components,
                        zip_flipped, zip_resid);
                TopologyAudit_dispose(&zipper_topology);
                TopologyAudit_dispose(&prebridge_topology);

                flat_faces = sw_faces;
                n_unique_faces = sw_nf;
                {
                    size_t dust_nf = n_unique_faces, dust_components = 0;
                    size_t dust_faces = cull_tiny_components(
                        arena, flat_faces, n_unique_faces, 4,
                        &dust_nf, &dust_components);
                    n_unique_faces = dust_nf;
                    if (dust_faces <= bridge_input_nf)
                        lineage_source_faces = bridge_input_nf - dust_faces;
                    if (dust_faces > 0)
                        fprintf(stderr,
                                "  chart dust: removed %zu unassimilated "
                                "one-triangle source chart(s) atomically\n",
                                dust_components);
                }
                gw_phase("zipper");
            }
        }

        ms = manifold_audit(arena, flat_faces, n_unique_faces, NULL, NULL);
        {
            MeshManifoldStats vms0 = MeshManifold_audit(arena, out_nv,
                                                        flat_faces, n_unique_faces);
            fprintf(stderr,
                "Pre-repair audit: unpaired=%zu non_manifold=%zu same_dir=%zu "
                "manifold=%zu pinch_verts=%zu\n",
                ms.unpaired, ms.non_manifold, ms.same_dir_pairs,
                ms.manifold_pairs, vms0.nm_verts);
        }
        gw_phase("audit_pre");

        size_t n_components = 0, n_flipped = 0, n_resid = 0;
        OrientMesh_consistent(arena, out_verts, out_nv, NULL,
                              flat_faces, n_unique_faces,
                              &n_flipped, &n_components, &n_resid);
        fprintf(stderr,
            "Winding repair (OrientMesh): %zu components, %zu flipped, "
            "%zu residual same_dir\n", n_components, n_flipped, n_resid);
        gw_phase("orient");

        size_t ow_flipped = 0, ow_radial = 0;
        float ow_axp[3] = { 0.0f, 0.0f, 0.0f };
        float ow_axd[3] = { 1.0f, 0.0f, 0.0f };
        const float *ow_paxp = NULL, *ow_paxd = NULL;
        {
            const char *ey = getenv("SEAM_UMBILICUS_Y");
            const char *ex = getenv("SEAM_UMBILICUS_X");
            if (ey != NULL && ex != NULL) {
                ow_axp[1] = (float)atof(ey);
                ow_axp[2] = (float)atof(ex);
                ow_paxp = ow_axp;
                ow_paxd = ow_axd;
            }
        }
        OrientWeld_components_axis(arena, out_verts, out_nv, flat_faces,
                                   n_unique_faces, 3.0f, ow_paxp, ow_paxd,
                                   &ow_flipped, &ow_radial);
        fprintf(stderr, "Component orientation: %zu backward component(s) flipped"
                " (%zu decided by radial fallback%s)\n", ow_flipped, ow_radial,
                ow_paxp != NULL ? "" : " -- fallback OFF, no umbilicus env");
        gw_phase("orientweld");

        if (legacy_bpa && !no_pinhole) {
            ComponentMesh pcm; memset(&pcm, 0, sizeof pcm);
            pcm.verts = out_verts; pcm.faces = flat_faces;
            pcm.nv = out_nv; pcm.nf = n_unique_faces; pcm.comp_id = 1; pcm.self = &pcm;
            size_t f_closed = 0, f_fl = 0, f_ad = 0, f_sk = 0;
            PinholeFill_close_bowties(arena, &pcm, 1, &f_closed);
            PinholeFill_fill_small_loops(arena, &pcm, 1, 0,
                                         &f_fl, &f_ad, &f_sk);
            fprintf(stderr,
                "Final chart-safe pinhole pass: bowties=%zu filled=%zu "
                "tris+=%zu skipped=%zu -> %zu faces, %zu verts\n",
                f_closed, f_fl, f_ad, f_sk, pcm.nf, pcm.nv);
            flat_faces = pcm.faces; n_unique_faces = pcm.nf;
            out_verts = pcm.verts; out_nv = pcm.nv;
        }
        gw_phase("pinhole2");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "pinhole2",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && !getenv("SEAM_NO_MICROWELD")) {
            size_t w_nf = 0, w_nv = 0;
            size_t w_comp_before = 0, w_comp_after = 0;
            float *w_verts = NULL;
            if (topology_safe_micro_weld(arena, out_verts, out_nv,
                                         flat_faces, n_unique_faces,
                                         have_chart_lineage ? &chart_lineage : NULL,
                                         &w_verts, &w_nv, &w_nf,
                                         &w_comp_before,
                                         &w_comp_after) != 0) {
                fprintf(stderr,
                        "ERROR: topology-scoped micro-weld failed "
                        "(components %zu -> %zu)\n",
                        w_comp_before, w_comp_after);
                RAISE(IO_Failed);
            }
            if (w_nv != out_nv || w_nf != n_unique_faces) {
                fprintf(stderr, "Boundary-arc micro-weld: %zu -> %zu verts, "
                        "%zu -> %zu faces (cc %zu -> %zu)\n",
                        out_nv, w_nv, n_unique_faces, w_nf,
                        w_comp_before, w_comp_after);
            }
            out_verts = w_verts; out_nv = w_nv; n_unique_faces = w_nf;
        }
        gw_phase("microweld");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "microweld",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && !no_holefill && !no_pinhole) {
            for (int round = 1; round <= 3; round++) {
                size_t r_loops = 0, r_int = 0, r_filled = 0;
                HoleFill_process_ex(arena, &out_verts, &flat_faces,
                                    &out_nv, &n_unique_faces,
                                    NULL, 2  ,
                                    &r_loops, &r_int, &r_filled);
                ComponentMesh rcm; memset(&rcm, 0, sizeof rcm);
                rcm.verts = out_verts; rcm.faces = flat_faces;
                rcm.nv = out_nv; rcm.nf = n_unique_faces;
                rcm.comp_id = 1; rcm.self = &rcm;
                size_t r_closed = 0, r_fl = 0, r_ad = 0, r_sk = 0;
                PinholeFill_close_bowties(arena, &rcm, 1, &r_closed);
                PinholeFill_fill_small_loops(arena, &rcm, 1, 0,
                                             &r_fl, &r_ad, &r_sk);
                out_verts = rcm.verts; flat_faces = rcm.faces;
                out_nv = rcm.nv; n_unique_faces = rcm.nf;
                if (r_filled + r_fl + r_closed == 0) break;
                fprintf(stderr, "Fill fixpoint round %d: interior=%zu pinhole=%zu "
                        "bowties=%zu"
                        " -> %zu faces, %zu verts\n",
                        round, r_filled, r_fl, r_closed,
                        n_unique_faces, out_nv);
            }
        }
        gw_phase("fixpoint");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "fixpoint",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && getenv("SEAM_LAMBDA_GATE")) {
            double lmax = 0.05f; float zone = 4.0f;
            { const char *e = getenv("SEAM_LAMBDA_MAX");  if (e) { double v=atof(e); if (v>0) lmax=v; } }
            { const char *e = getenv("SEAM_LAMBDA_ZONE"); if (e) { double v=atof(e); if (v>0) zone=(float)v; } }
            double *lam = (double *)ARENA_ALLOC(arena, (out_nv*sizeof(double)));
            if (Develop_vertex_energy(arena, out_verts, out_nv, flat_faces,
                                      n_unique_faces, lam) == 0) {
                unsigned char *cut = (unsigned char *)ARENA_CALLOC(arena, out_nv, 1L);
                size_t ncut = 0;
                for (size_t v = 0; v < out_nv; v++)
                    if (lam[v] > lmax && near_cube_boundary(&out_verts[v*3], zone)) {
                        cut[v] = 1; ncut++;
                    }
                size_t w = 0, removed = 0;
                for (size_t f = 0; f < n_unique_faces; f++) {
                    int32_t a = flat_faces[f*3+0], b = flat_faces[f*3+1], c = flat_faces[f*3+2];
                    if (cut[a] || cut[b] || cut[c]) { removed++; continue; }
                    flat_faces[w*3+0]=a; flat_faces[w*3+1]=b; flat_faces[w*3+2]=c; w++;
                }
                n_unique_faces = w;
                fprintf(stderr,
                    "Lambda gate: %zu high-lambda seam vert(s) -> %zu weld face(s) "
                    "severed (lambda > %.3f within %.1f vox of a seam)\n",
                    ncut, removed, lmax, (double)zone);
            }
        }

        if (legacy_bpa && getenv("SEAM_PHASE_SEVER")) {
            double ps_umb_y = 0.0, ps_umb_x = 0.0, ps_pitch = 0.0;
            { const char *e = getenv("SEAM_UMBILICUS_Y"); if (e) ps_umb_y = atof(e); }
            { const char *e = getenv("SEAM_UMBILICUS_X"); if (e) ps_umb_x = atof(e); }
            { const char *e = getenv("SEAM_WRAP_PITCH"); if (e) { double v = atof(e); if (v > 0) ps_pitch = v; } }
            double ps_tol = 0.25, ps_lam = 0.05, ps_rmin_p = 2.0;
            { const char *e = getenv("SEAM_PHASE_SEVER_TOL"); if (e) { double v = atof(e); if (v > 0) ps_tol = v; } }
            { const char *e = getenv("SEAM_PHASE_SEVER_LAMBDA"); if (e) { double v = atof(e); if (v > 0) ps_lam = v; } }
            { const char *e = getenv("SEAM_PHASE_SEVER_RMIN_PITCHES"); if (e) { double v = atof(e); if (v > 0) ps_rmin_p = v; } }
            if (ps_pitch <= 0.0 || (ps_umb_y == 0.0 && ps_umb_x == 0.0)) {
                fprintf(stderr, "Phase sever: SKIPPED (needs SEAM_UMBILICUS_Y/X "
                        "+ SEAM_WRAP_PITCH)\n");
            } else {
                double *lam = (double *)ARENA_ALLOC(arena,
                                  (out_nv * sizeof(double)));
                double *ww = (double *)ARENA_ALLOC(arena,
                                  (out_nv * sizeof(double)));
                double *rr = (double *)ARENA_ALLOC(arena,
                                  (out_nv * sizeof(double)));
                if (Develop_vertex_energy(arena, out_verts, out_nv, flat_faces,
                                          n_unique_faces, lam) == 0) {
                    double rmin = ps_rmin_p * ps_pitch;
                    size_t ps_maxc = 2500;
                    { const char *e = getenv("SEAM_PHASE_SEVER_MAXC");
                      if (e) { long v = atol(e); if (v > 0) ps_maxc = (size_t)v; } }
                    for (size_t v = 0; v < out_nv; v++) {
                        double dy = (double)out_verts[v*3+1] - ps_umb_y;
                        double dx = (double)out_verts[v*3+2] - ps_umb_x;
                        rr[v] = hypot(dy, dx);
                        ww[v] = rr[v] / ps_pitch - atan2(dy, dx) / (2.0 * M_PI);
                    }
                    uint8_t *cand = (uint8_t *)ARENA_CALLOC(arena,
                                        n_unique_faces, 1L);
                    for (size_t f = 0; f < n_unique_faces; f++) {
                        int32_t t[3] = { flat_faces[f*3+0], flat_faces[f*3+1],
                                         flat_faces[f*3+2] };
                        if (rr[t[0]] < rmin || rr[t[1]] < rmin || rr[t[2]] < rmin)
                            continue;
                        double span = 0.0;
                        for (int e = 0; e < 3; e++) {
                            int32_t a = t[e], b = t[(e+1)%3];
                            double dth = atan2((double)out_verts[a*3+1] - ps_umb_y,
                                               (double)out_verts[a*3+2] - ps_umb_x)
                                       - atan2((double)out_verts[b*3+1] - ps_umb_y,
                                               (double)out_verts[b*3+2] - ps_umb_x);
                            while (dth >  M_PI) dth -= 2.0*M_PI;
                            while (dth < -M_PI) dth += 2.0*M_PI;
                            double dw = fabs((rr[a] - rr[b]) / ps_pitch
                                             - dth / (2.0 * M_PI));
                            if (dw > span) span = dw;
                        }
                        if (span > ps_tol) cand[f] = 1;
                    }
                    int32_t *fpar = (int32_t *)ARENA_ALLOC(arena,
                                        (n_unique_faces * sizeof(int32_t)));
                    for (size_t f = 0; f < n_unique_faces; f++)
                        fpar[f] = (int32_t)f;
                    int32_t *vf = (int32_t *)ARENA_ALLOC(arena,
                                      (out_nv * sizeof(int32_t)));
                    for (size_t v = 0; v < out_nv; v++) vf[v] = -1;
                    for (size_t f = 0; f < n_unique_faces; f++) {
                        if (!cand[f]) continue;
                        for (int k = 0; k < 3; k++) {
                            int32_t v = flat_faces[f*3+k];
                            if (vf[v] < 0) { vf[v] = (int32_t)f; continue; }
                            int32_t x = (int32_t)f, y = vf[v];
                            while (fpar[x] != x) x = fpar[x] = fpar[fpar[x]];
                            while (fpar[y] != y) y = fpar[y] = fpar[fpar[y]];
                            if (x != y) fpar[x] = y;
                        }
                    }
                    size_t removed = 0, n_clusters = 0, n_cut_clusters = 0;
                    int32_t *croot = (int32_t *)ARENA_ALLOC(arena,
                                         (n_unique_faces * sizeof(int32_t)));
                    size_t *csize = (size_t *)ARENA_CALLOC(arena,
                                        n_unique_faces,
                                        sizeof(size_t));
                    uint8_t *chot = (uint8_t *)ARENA_CALLOC(arena,
                                        n_unique_faces, 1L);
                    for (size_t f = 0; f < n_unique_faces; f++) {
                        if (!cand[f]) { croot[f] = -1; continue; }
                        int32_t x = (int32_t)f;
                        while (fpar[x] != x) x = fpar[x] = fpar[fpar[x]];
                        croot[f] = x;
                        if (csize[x]++ == 0) n_clusters++;
                        for (int k = 0; k < 3; k++)
                            if (lam[flat_faces[f*3+k]] > ps_lam) chot[x] = 1;
                    }
                    size_t w = 0;
                    for (size_t f = 0; f < n_unique_faces; f++) {
                        int cutf = 0;
                        if (croot[f] >= 0) {
                            size_t sz = csize[croot[f]];
                            if (sz <= ps_maxc && chot[croot[f]]) cutf = 1;
                        }
                        if (cutf) { removed++; continue; }
                        flat_faces[w*3+0] = flat_faces[f*3+0];
                        flat_faces[w*3+1] = flat_faces[f*3+1];
                        flat_faces[w*3+2] = flat_faces[f*3+2];
                        w++;
                    }
                    for (size_t f = 0; f < n_unique_faces; f++)
                        if (croot[f] >= 0 && csize[croot[f]] <= ps_maxc
                            && chot[croot[f]] && croot[f] == (int32_t)f)
                            n_cut_clusters++;
                    n_unique_faces = w;
                    fprintf(stderr,
                        "Phase sever: %zu face(s) in %zu cluster(s) cut "
                        "(of %zu candidate clusters; span > %.2f turn, "
                        "cluster <= %zu faces, lambda > %.3f, r > %.0f)\n",
                        removed, n_cut_clusters, n_clusters, ps_tol,
                        ps_maxc, ps_lam, rmin);
                }
            }
        }

        if (legacy_bpa && !getenv("SEAM_NO_SEAMFILL")) {
            ComponentMesh scm; memset(&scm, 0, sizeof scm);
            scm.verts = out_verts; scm.faces = flat_faces;
            scm.nv = out_nv; scm.nf = n_unique_faces; scm.comp_id = 1; scm.self = &scm;
            SeamHoleFillParams shp; SeamHoleFill_default_params(&shp);
            shp.cube = seam_cube;
            const char *e;
            if ((e = getenv("SEAM_UMBILICUS_Y"))) shp.umb_y = atof(e);
            if ((e = getenv("SEAM_UMBILICUS_X"))) shp.umb_x = atof(e);
            if ((e = getenv("SEAM_WRAP_PITCH"))) { double v = atof(e); if (v > 0) shp.pitch = v; }
            if ((e = getenv("SEAM_WIND_TOL"))) { double v = atof(e); if (v > 0) shp.wind_tol_turns = v; }
            if ((e = getenv("SEAM_SEAMFILL_MAXLOOP"))) { int v = atoi(e); if (v >= 3) shp.max_loop = v; }
            if ((e = getenv("SEAM_SEAMFILL_EXTENT"))) { double v = atof(e); if (v > 0) shp.max_extent = v; }
            SeamHoleFillStats shs;
            SeamHoleFill_process(arena, &scm, &shp, &shs);
            fprintf(stderr,
                "Seam-hole fill: %zu candidate(s), %zu filled (+%zu tris); "
                "skip[phase=%zu extent=%zu geom=%zu manifold=%zu bubble=%zu]%s\n",
                shs.seam_candidates, shs.filled, shs.tris_added,
                shs.skip_phase, shs.skip_extent, shs.skip_geom,
                shs.skip_manifold, shs.skip_bubble,
                shp.pitch > 0.0 ? "" : "  (phase gate OFF -- no umbilicus/pitch)");
            flat_faces = scm.faces; n_unique_faces = scm.nf;
            out_verts = scm.verts; out_nv = scm.nv;
        }
        gw_phase("seamfill");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "seamfill",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && !no_cleanup) {
            ComponentMesh wcm; memset(&wcm, 0, sizeof wcm);
            wcm.verts = out_verts; wcm.faces = flat_faces;
            wcm.nv = out_nv; wcm.nf = n_unique_faces; wcm.comp_id = 1; wcm.self = &wcm;
            ManifoldGuardStats mg;
            ManifoldGuard_process_ex(arena, &wcm, 1, 0  ,
                                     0  , &mg);
            fprintf(stderr,
                "Manifold guard: %zu NM-edge(s) (-%zu faces), %zu pinch split(s)\n",
                mg.nm_edges_resolved, mg.faces_deleted, mg.pinch_splits);
            flat_faces = wcm.faces; n_unique_faces = wcm.nf;
            out_verts = wcm.verts; out_nv = wcm.nv;
        }
        gw_phase("guard");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "guard",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && !no_cleanup && !getenv("SEAM_NO_POSTGUARD")) {
            SeamHoleFillParams pgp; SeamHoleFill_default_params(&pgp);
            pgp.cube = seam_cube;
            { const char *e;
              if ((e = getenv("SEAM_UMBILICUS_Y"))) pgp.umb_y = atof(e);
              if ((e = getenv("SEAM_UMBILICUS_X"))) pgp.umb_x = atof(e);
              if ((e = getenv("SEAM_WRAP_PITCH")))  { double v = atof(e); if (v > 0) pgp.pitch = v; }
              if ((e = getenv("SEAM_WIND_TOL")))    { double v = atof(e); if (v > 0) pgp.wind_tol_turns = v; }
              if ((e = getenv("SEAM_SEAMFILL_EXTENT"))) { double v = atof(e); if (v > 0) pgp.max_extent = v; } }
            for (int gr = 1; gr <= 3; gr++) {
                if (!getenv("SEAM_NO_MICROWELD")) {
                    size_t pw_nf = 0, pw_nv = 0;
                    size_t pw_comp_before = 0, pw_comp_after = 0;
                    float *pw_v = NULL;
                    if (topology_safe_micro_weld(arena, out_verts, out_nv,
                                                 flat_faces, n_unique_faces,
                                                 have_chart_lineage ?
                                                     &chart_lineage : NULL,
                                                 &pw_v, &pw_nv, &pw_nf,
                                                 &pw_comp_before,
                                                 &pw_comp_after) != 0) {
                        fprintf(stderr,
                                "ERROR: postguard topology-scoped micro-weld "
                                "failed (components %zu -> %zu)\n",
                                pw_comp_before, pw_comp_after);
                        RAISE(IO_Failed);
                    }
                    if (pw_nv != out_nv) {
                        fprintf(stderr, "  postguard micro-weld: %zu -> %zu verts "
                                "(cc %zu -> %zu)\n",
                                out_nv, pw_nv,
                                pw_comp_before, pw_comp_after);
                    }
                    out_verts = pw_v; out_nv = pw_nv; n_unique_faces = pw_nf;
                }
                ComponentMesh pg; memset(&pg, 0, sizeof pg);
                pg.verts = out_verts; pg.faces = flat_faces;
                pg.nv = out_nv; pg.nf = n_unique_faces; pg.comp_id = 1; pg.self = &pg;
                size_t g_closed=0, g_fl=0, g_ad=0, g_sk=0;
                PinholeFill_close_bowties(arena, &pg, 1, &g_closed);
                PinholeFill_fill_small_loops(arena, &pg, 1, 0,
                                             &g_fl, &g_ad, &g_sk);
                SeamHoleFillStats pgs; SeamHoleFill_process(arena, &pg, &pgp, &pgs);
                ManifoldGuardStats pmg;
                ManifoldGuard_process_ex(arena, &pg, 1, 0,
                                         0  , &pmg);
                out_verts = pg.verts; flat_faces = pg.faces;
                out_nv = pg.nv; n_unique_faces = pg.nf;
                fprintf(stderr,
                    "Post-guard round %d: pinhole=%zu bowties=%zu seamfill=%zu "
                    "guard(-%zu f, "
                    "%zu split) -> %zu faces, %zu verts\n",
                    gr, g_fl, g_closed, pgs.filled,
                    pmg.faces_deleted, pmg.pinch_splits,
                    n_unique_faces, out_nv);
                if (g_fl == 0 && g_closed == 0 && pgs.filled == 0 &&
                    pmg.faces_deleted == 0 && pmg.pinch_splits == 0) break;
            }
        }
        gw_phase("postguard");
        dump_stage(arena, dump_stages_dir, stage_prefix, 8, "postguard",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        if (legacy_bpa && !no_bridge && !no_cleanup &&
            !getenv("SEAM_NO_RECOARSEN")) {
            double r_band = 6.0;
            { const char *rbe = getenv("SEAM_BAND");
              if (rbe) { double v = atof(rbe); if (v > 0) r_band = v; } }
            uint8_t *r_used = (uint8_t *)ARENA_CALLOC(arena, out_nv, 1L);
            for (size_t f = 0; f < n_unique_faces; f++) {
                r_used[flat_faces[f*3+0]] = 1;
                r_used[flat_faces[f*3+1]] = 1;
                r_used[flat_faces[f*3+2]] = 1;
            }
            SeamPlane rplanes[64];
            size_t rnp = SeamPlanes_detect(out_verts, out_nv, r_used,
                                           (double)seam_cube, r_band,
                                           rplanes, 64);
            if (rnp > 0) {
                ComponentMesh rcm; memset(&rcm, 0, sizeof rcm);
                rcm.verts = out_verts; rcm.faces = flat_faces;
                rcm.nv = out_nv; rcm.nf = n_unique_faces;
                rcm.comp_id = 1; rcm.self = &rcm;
                WeldRecoarsenParams rp;
                WeldCleanup_default_recoarsen_params(&rp);
                { const char *re;
                  if ((re = getenv("SEAM_RECOARSEN_BELOW"))) {
                      double v = atof(re); if (v > 0) rp.collapse_below = v; }
                  if ((re = getenv("SEAM_RECOARSEN_BAND"))) {
                      double v = atof(re); if (v > 0) rp.band = v; } }
                WeldRecoarsenStats rs;
                WeldCleanup_recoarsen_seam(arena, &rcm, rplanes, rnp, &rp, &rs);
                flat_faces = rcm.faces; n_unique_faces = rcm.nf;
                recoarsen_collapses = rs.n_collapses;
                recoarsen_faces_in = rs.faces_in;
                recoarsen_faces_out = rs.faces_out;
                fprintf(stderr,
                    "Recoarsen: %zu collapse(s), %zu flip(s), %zu -> %zu faces "
                    "(band %.1f, below %.1f vox, %zu plane(s))\n",
                    rs.n_collapses, rs.n_flips, rs.faces_in, rs.faces_out,
                    rp.band, rp.collapse_below, rnp);
            }
            gw_phase("recoarsen");
            dump_stage(arena, dump_stages_dir, stage_prefix, 9, "recoarsen",
                       out_verts, out_nv, flat_faces, n_unique_faces,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);
        }

        if (legacy_bpa && !no_bridge && !no_cleanup &&
            !getenv("SEAM_NO_BANDCVT")) {
            double bc_band = 6.0;
            { const char *bce = getenv("SEAM_BAND");
              if (bce) { double v = atof(bce); if (v > 0) bc_band = v; } }
            uint8_t *bc_used = (uint8_t *)ARENA_CALLOC(arena, out_nv, 1L);
            for (size_t f = 0; f < n_unique_faces; f++) {
                bc_used[flat_faces[f*3+0]] = 1;
                bc_used[flat_faces[f*3+1]] = 1;
                bc_used[flat_faces[f*3+2]] = 1;
            }
            SeamPlane bplanes[64];
            size_t bnp = SeamPlanes_detect(out_verts, out_nv, bc_used,
                                           (double)seam_cube, bc_band,
                                           bplanes, 64);
            if (bnp > 0) {
                SeamBandCvtParams bp; SeamBandCvt_default_params(&bp);
                { const char *bce;
                  if ((bce = getenv("SEAM_BANDCVT_H"))) {
                      double v = atof(bce); if (v > 0) bp.target_h = v; }
                  if ((bce = getenv("SEAM_BANDCVT_BAND"))) {
                      double v = atof(bce); if (v > 0) bp.band = v; } }
                for (int bc_pass = 0; bc_pass < 2; bc_pass++) {
                    bp.offset_half = bc_pass;
                    SeamBandCvtStats bst;
                    float *bc_v = NULL; int32_t *bc_f = NULL; int32_t *bc_src = NULL;
                    size_t bc_nv = 0, bc_nf = 0, bc_nnew = 0;
                    if (SeamBandCvt_process(arena, out_verts, out_nv,
                                            flat_faces, n_unique_faces,
                                            bplanes, bnp, &bp,
                                            &bc_v, &bc_nv, &bc_f, &bc_nf,
                                            &bc_src, &bc_nnew, &bst) == 0
                        && bst.accepted > 0) {
                        if (bc_nv > vert_cap) {
                            fprintf(stderr, "grid_weld: vert capacity %zu "
                                    "exceeded by band CVT (%zu)\n",
                                    vert_cap, bc_nv);
                            RAISE(IO_Failed);
                        }
                        for (size_t v2 = 0; v2 < bc_nnew; v2++)
                            vert_cube_idx[out_nv + v2] = vert_cube_idx[bc_src[v2]];
                        out_verts = bc_v; out_nv = bc_nv;
                        flat_faces = bc_f; n_unique_faces = bc_nf;
                        color_nv = out_nv;
                        bandcvt_accepted += bst.accepted;
                        bandcvt_rejected += bst.rejected;
                        bandcvt_faces_in += bst.faces_band_in;
                        bandcvt_faces_out += bst.faces_band_out;
                    }
                    fprintf(stderr,
                        "Band CVT pass %c: %zu/%zu patch(es) accepted "
                        "(rej rc=%zu pin=%zu bnd=%zu chi=%zu nm=%zu vtx=%zu comp=%zu; "
                        "%zu small, %zu clean), band %zu -> %zu faces, "
                        "+%zu verts (h %.1f, tile %.0f)\n",
                        bc_pass ? 'B' : 'A', bst.accepted, bst.patches,
                        bst.rej_rc, bst.rej_pin, bst.rej_bnd, bst.rej_chi,
                        bst.rej_manifold, bst.rej_vtx, bst.rej_comp,
                        bst.skipped_small, bst.skipped_clean,
                        bst.faces_band_in, bst.faces_band_out,
                        bst.verts_added, bp.target_h, bp.tile);
                }
                if (bandcvt_accepted > 0) {
                    ComponentMesh gm; memset(&gm, 0, sizeof gm);
                    gm.verts = out_verts; gm.faces = flat_faces;
                    gm.nv = out_nv; gm.nf = n_unique_faces;
                    gm.comp_id = 1; gm.self = &gm;
                    ManifoldGuardStats mg2;
                    ManifoldGuard_process_ex(arena, &gm, 1, 0,
                                             0  , &mg2);
                    out_verts = gm.verts; flat_faces = gm.faces;
                    out_nv = gm.nv; n_unique_faces = gm.nf;
                    if (mg2.nm_edges_resolved || mg2.pinch_splits)
                        fprintf(stderr,
                            "  band-cvt guard: %zu NM edge(s), %zu pinch "
                            "split(s)\n", mg2.nm_edges_resolved,
                            mg2.pinch_splits);
                }
            }
            gw_phase("bandcvt");
            dump_stage(arena, dump_stages_dir, stage_prefix, 10, "bandcvt",
                       out_verts, out_nv, flat_faces, n_unique_faces,
                       vert_cube_idx, cube_palette, cubes.n, color_nv);
        }

        if (legacy_bpa && !no_bridge && !no_cleanup &&
            !no_intersection_clean) {
            double ic_band = 0.0;
            IntersectionCleanupParams ip;
            uint8_t *ic_used = NULL, *ic_mask = NULL;
            SeamPlane ic_planes[64];
            const char *ice = NULL;

            IntersectionCleanup_default_params(&ip);
            if ((ice = getenv("SEAM_INTERSECTION_BAND"))) {
                double v = atof(ice); if (v > 0.0) ic_band = v;
            }
            if ((ice = getenv("SEAM_INTERSECTION_GAP"))) {
                double v = atof(ice); if (v >= 0.0) ip.gap_max = v;
            }
            if ((ice = getenv("SEAM_INTERSECTION_ANGLE"))) {
                double v = atof(ice); if (v >= 0.0 && v < 90.0)
                    ip.parallel_angle_deg = v;
            }
            if ((ice = getenv("SEAM_INTERSECTION_MAX_DELETE"))) {
                double v = atof(ice); if (v >= 0.0)
                    ip.max_delete_fraction = v;
            }
            if ((ice = getenv("SEAM_INTERSECTION_MAX_CONFLICTS"))) {
                size_t v = (size_t)strtoull(ice, NULL, 10);
                if (v > 0) ip.max_conflicts = v;
            }
            ip.include_hinges = 1;

            if (ic_band > 0.0) {
                ic_used = (uint8_t *)ARENA_CALLOC(arena, out_nv, 1L);
                for (size_t f = 0; f < n_unique_faces; f++) {
                    ic_used[flat_faces[f*3+0]] = 1;
                    ic_used[flat_faces[f*3+1]] = 1;
                    ic_used[flat_faces[f*3+2]] = 1;
                }
                intersection_planes =
                    SeamPlanes_detect(out_verts, out_nv, ic_used,
                                      (double)seam_cube, ic_band,
                                      ic_planes, 64);
                ic_mask = (uint8_t *)ARENA_CALLOC(arena,
                                                  n_unique_faces, 1L);
                for (size_t f = 0; f < n_unique_faces; f++) {
                    for (int k = 0; k < 3; k++) {
                        int32_t v = flat_faces[f*3+k];
                        if (SeamPlanes_vert_dist(out_verts, v, ic_planes,
                                                 intersection_planes) <= ic_band) {
                            ic_mask[f] = 1;
                            intersection_masked_faces++;
                            break;
                        }
                    }
                }
            } else {
                intersection_masked_faces = n_unique_faces;
            }

            intersection_faces_in = n_unique_faces;
            if ((ic_band <= 0.0 || intersection_planes > 0) &&
                intersection_masked_faces > 1) {
                if (IntersectionCleanup_process(out_verts, out_nv, flat_faces,
                                                &n_unique_faces, ic_mask,
                                                &ip, &intersection_stats) != 0) {
                    fprintf(stderr,
                            "ERROR: intersection cleanup rejected: conflicts=%zu "
                            "proposed=%zu/%zu effective_budget=%zu nominal=%.4f cap=%zu\n",
                            intersection_stats.conflicts,
                            intersection_stats.faces_deleted,
                            intersection_faces_in, intersection_stats.delete_budget,
                            ip.max_delete_fraction,
                            ip.max_conflicts);
                    RAISE(IO_Failed);
                }
            }
            fprintf(stderr,
                    "Intersection cleanup: %s, %zu/%zu tested faces; "
                    "%zu conflicts [overlap=%zu stab=%zu fold=%zu] -> "
                    "-%zu faces (max degree %zu)\n",
                    ic_band > 0.0 ? "seam-band" : "global",
                    intersection_masked_faces,
                    intersection_faces_in, intersection_stats.conflicts,
                    intersection_stats.overlap_pairs,
                    intersection_stats.stab_pairs,
                    intersection_stats.fold_pairs,
                    intersection_stats.faces_deleted,
                    intersection_stats.max_conflict_degree);

            {
                MeshManifoldStats ims = MeshManifold_audit(
                    arena, out_nv, flat_faces, n_unique_faces);
                if (ims.nm_edges > 0) {
                    fprintf(stderr,
                            "ERROR: intersection cleanup left %zu non-manifold "
                            "edge(s)\n", ims.nm_edges);
                    RAISE(IO_Failed);
                }
                if (ims.nm_verts > 0) {
                    fprintf(stderr,
                            "ERROR: intersection cleanup created %zu chart "
                            "bowtie vertex/vertices; transaction rejected\n",
                            ims.nm_verts);
                    RAISE(IO_Failed);
                }
            }
            fprintf(stderr, "  intersection topology: no chart vertices split\n");
            gw_phase("intersect");
            dump_stage(arena, dump_stages_dir, stage_prefix, 11,
                       "intersectionclean", out_verts, out_nv,
                       flat_faces, n_unique_faces, vert_cube_idx,
                       cube_palette, cubes.n, color_nv);
        }

        if (!no_cleanup) {
            size_t ro_comp = 0, ro_flip = 0, ro_resid = 0;
            OrientMesh_consistent(arena, out_verts, out_nv, NULL,
                                  flat_faces, n_unique_faces,
                                  &ro_flip, &ro_comp, &ro_resid);
            fprintf(stderr, "Post-fill winding repair: %zu components, %zu "
                    "flipped, %zu residual same_dir\n", ro_comp, ro_flip, ro_resid);
            {
                size_t row_flipped = 0, row_radial = 0;
                OrientWeld_components_axis(arena, out_verts, out_nv,
                                           flat_faces, n_unique_faces, 3.0f,
                                           ow_paxp, ow_paxd,
                                           &row_flipped, &row_radial);
                fprintf(stderr,
                        "Post-clean component orientation: %zu flipped "
                        "(%zu radial fallback)\n",
                        row_flipped, row_radial);
            }
        }
        gw_phase("orient2");

        {
            if(lineage_parent0&&lineage_parent1){
                for(size_t v=lineage_initialized_nv;v<out_nv;v++)
                    lineage_parent0[v]=lineage_parent1[v]=-1;
                lineage_initialized_nv=out_nv;
            }
            size_t orphaned=compact_orphan_vertices(
                arena,&out_verts,&out_nv,flat_faces,n_unique_faces,
                &vert_cube_idx,&vert_is_weld,&color_nv,
                &lineage_parent0,&lineage_parent1);
            if(orphaned>0)
                fprintf(stderr,"Orphan compaction: removed %zu unreferenced "
                        "vertex slot(s); faces/topology unchanged\n",orphaned);
        }

        {
            TopologyAuditOptions disk_audit_options;
            TopologyAuditReport disk_audit;
            memset(&disk_audit, 0, sizeof disk_audit);
            TopologyAudit_options_default(&disk_audit_options);
            disk_audit_options.emit_generators = 1;
            disk_audit_options.max_generators_per_component = 8;
            if (TopologyAudit_analyze(
                    out_verts,out_nv,flat_faces,n_unique_faces,
                    &disk_audit_options,&disk_audit)!=0) {
                fprintf(stderr,
                        "ERROR: exact disk topology audit failed: %s\n",
                        disk_audit.error);
                TopologyAudit_dispose(&disk_audit);
                RAISE(IO_Failed);
            }
            disk_topology_components=disk_audit.face_components;
            disk_topology_certificate=
                disk_audit.all_components_are_disks &&
                disk_audit.inconsistent_winding_components==0;
            memset(&disk_topology_stats,0,sizeof disk_topology_stats);
            disk_topology_stats.components=disk_audit.face_components;
            disk_topology_stats.already_disks=disk_audit.disk_components;
            disk_topology_stats.failed_components=disk_audit.nondisk_components;
            disk_topology_stats.vertices_in=out_nv;
            disk_topology_stats.vertices_out=out_nv;
            disk_topology_stats.faces_in=n_unique_faces;
            disk_topology_stats.faces_out=n_unique_faces;
            for (size_t c=0;c<disk_audit.face_components;c++) {
                disk_topology_stats.boundary_loops_before +=
                    disk_audit.component[c].boundary_loops;
                disk_topology_stats.boundary_loops_after +=
                    disk_audit.component[c].boundary_loops;
            }
            fprintf(stderr,
                    "Exact chart disk certificate: %s (%zu/%zu disks, "
                    "beta1=%lld, invalid=%zu, winding=%zu; no vertex cuts)\n",
                    disk_topology_certificate ? "PASS" : "FAIL",
                    disk_audit.disk_components,
                    disk_audit.face_components,
                    (long long)disk_audit.beta_1,
                    disk_audit.invalid_surface_components,
                    disk_audit.inconsistent_winding_components);
            if (!disk_topology_certificate) {
                for (size_t c=0;c<disk_audit.face_components;c++) {
                    const TopologyComponentInvariant *ti=&disk_audit.component[c];
                    if (ti->homeomorphic_to_disk) continue;
                    fprintf(stderr,
                            "  [non-disk] rank=%zu V=%zu E=%zu F=%zu "
                            "loops=%zu genus=%lld crosscap=%lld beta1=%lld "
                            "generators=%zu defects=0x%08x bbox="
                            "[%.3f %.3f %.3f]-[%.3f %.3f %.3f]\n",
                            ti->rank,ti->vertices,ti->edges,ti->faces,
                            ti->boundary_loops,(long long)ti->orientable_genus,
                            (long long)ti->crosscap_number,
                            (long long)ti->beta_1,ti->minimal_generator_rank,
                            (unsigned)ti->defect_mask,
                            ti->bbox_min[0],ti->bbox_min[1],ti->bbox_min[2],
                            ti->bbox_max[0],ti->bbox_max[1],ti->bbox_max[2]);
                    for (size_t g=0;g<disk_audit.emitted_generators;g++) {
                        const TopologyGenerator *gen=&disk_audit.generator[g];
                        if (gen->component != c) continue;
                        fprintf(stderr,
                                "    generator %zu: %zu vertices, length=%.3f "
                                "closing=(%d,%d)\n",
                                gen->ordinal,gen->nvertices,gen->length,
                                gen->closing_edge_a,gen->closing_edge_b);
                    }
                }
                TopologyAudit_dispose(&disk_audit);
                RAISE(IO_Failed);
            }
            TopologyAudit_dispose(&disk_audit);
        }
        gw_phase("disk_topology");
        dump_stage(arena, dump_stages_dir, stage_prefix, 12, "diskcertificate",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        dump_stage(arena, dump_stages_dir, stage_prefix, 12, "final",
                   out_verts, out_nv, flat_faces, n_unique_faces,
                   vert_cube_idx, cube_palette, cubes.n, color_nv);

        char diag_path[1024];
        snprintf(diag_path, sizeof(diag_path), "%s.bad_edges.obj", out_path);
        FILE *diag_fp = fopen(diag_path, "w");
        ms = manifold_audit(arena, flat_faces, n_unique_faces,
                            out_verts, diag_fp);
        if (diag_fp) {
            fclose(diag_fp);
            fprintf(stderr, "Wrote %s\n", diag_path);
        }

        if (ms.non_manifold > 0) {
            char nm_path[1024];
            snprintf(nm_path, sizeof(nm_path), "%s.nonmanifold.obj", out_path);
            emit_nonmanifold_neighborhood(arena, flat_faces, n_unique_faces,
                                          out_verts, out_nv, nm_path);
            fprintf(stderr, "Wrote %s (non-manifold neighbourhood)\n",
                    nm_path);
        }
        MeshManifoldStats vms = MeshManifold_audit(arena, out_nv,
                                                   flat_faces, n_unique_faces);
        size_t pinch_verts = vms.nm_verts;
        fprintf(stderr,
            "Post-repair audit: unpaired=%zu non_manifold=%zu same_dir=%zu "
            "manifold=%zu pinch_verts=%zu\n",
            ms.unpaired, ms.non_manifold, ms.same_dir_pairs, ms.manifold_pairs,
            pinch_verts);
        gw_phase("audit_post");

        {
            IntersectionCleanupParams ep;
            size_t *face_conflict_degree = (size_t *)ARENA_CALLOC(
                arena, n_unique_faces, sizeof(size_t));
            int embedded_audit_rc;
            IntersectionCleanup_default_params(&ep);
            ep.gap_max = 0.0;
            ep.include_hinges = 1;
            embedded_audit_rc = IntersectionCleanup_audit(
                    out_verts,out_nv,flat_faces,n_unique_faces,NULL,
                    &ep,face_conflict_degree,&embedded_stats);
            if (embedded_audit_rc != 0) {
                fprintf(stderr,
                        "ERROR: final embedded-geometry audit failed: "
                        "candidates=%zu conflicts=%zu "
                        "[overlap=%zu stab=%zu fold=%zu]\n",
                        embedded_stats.candidate_pairs,
                        embedded_stats.conflicts,
                        embedded_stats.overlap_pairs,
                        embedded_stats.stab_pairs,
                        embedded_stats.fold_pairs);
                RAISE(IO_Failed);
            }
            if (embedded_stats.conflicts != 0) {
                if (!no_bridge) {
                    fprintf(stderr,
                            "ERROR: final embedded-geometry certificate failed: "
                            "candidates=%zu conflicts=%zu "
                            "[overlap=%zu stab=%zu fold=%zu]\n",
                            embedded_stats.candidate_pairs,
                            embedded_stats.conflicts,
                            embedded_stats.overlap_pairs,
                            embedded_stats.stab_pairs,
                            embedded_stats.fold_pairs);
                    RAISE(IO_Failed);
                } else {
                    char embedded_path[1024];
                    snprintf(embedded_path, sizeof(embedded_path),
                             "%s.embedded_conflicts.obj", out_path);
                    emit_embedded_conflict_neighborhood(
                        arena, flat_faces, n_unique_faces, out_verts, out_nv,
                        face_conflict_degree, embedded_path);
                    fprintf(stderr,
                            "WARNING: assembly-only source geometry has %zu "
                            "exact conflicts [overlap=%zu stab=%zu fold=%zu]; "
                            "preserving all source faces\n",
                            embedded_stats.conflicts,
                            embedded_stats.overlap_pairs,
                            embedded_stats.stab_pairs,
                            embedded_stats.fold_pairs);
                    fprintf(stderr, "Wrote %s (conflict neighbourhood)\n",
                            embedded_path);
                }
            } else {
                embedded_geometry_certificate = 1;
                fprintf(stderr,
                        "Exact embedded-geometry certificate: PASS "
                        "(%zu candidates, 0 conflicts)\n",
                        embedded_stats.candidate_pairs);
            }
        }
        gw_phase("embedded");

        size_t n_weld_verts = 0;
        for (size_t v = 0; v < color_nv; v++) {
            if (vert_is_weld[v]) n_weld_verts++;
        }
        fprintf(stderr, "Welded verts (shared across >=2 cubes): %zu\n",
                n_weld_verts);

        FILE *ofp = fopen(out_path, "w");
        if (!ofp) {
            fprintf(stderr, "grid_weld: cannot open %s\n", out_path);
            RAISE(IO_Failed);
        }
        const char *gw_color_by = getenv("GW_COLOR_BY");
        int by_cube = (gw_color_by && !strcmp(gw_color_by, "cube"));
        float *cc = NULL;
        if (!by_cube) {
            cc = (float *)malloc(out_nv * 3 * sizeof(float));
            if (cc) { CCColorOpts opts; CCColor_default_opts(&opts); opts.sat = 0.75;
                      CCColor_compute(out_nv, flat_faces, n_unique_faces, &opts, cc, NULL); }
        }
        fprintf(ofp, "# grid_weld output: %s vertex colors\n",
                cc ? "connected-component" : "per-cube provenance");
        fprintf(ofp, "# %zu verts (%zu weld), %zu faces\n",
                out_nv, n_weld_verts, n_unique_faces);
        for (size_t v = 0; v < out_nv; v++) {
            float cr, cg, cb;
            if (cc) {
                cr = cc[v*3+0]; cg = cc[v*3+1]; cb = cc[v*3+2];
            } else {
                const float *c;
                int is_weld = (v < color_nv) ? vert_is_weld[v] : 0;
                if (is_weld)            c = SEAM_COLOR;
                else if (v >= color_nv) c = PINHOLE_COLOR;
                else { int cube = vert_cube_idx[v];
                       if (cube < 0) c = PINHOLE_COLOR;
                       else { int pal = (cube < (int)cubes.n) ?
                                            cube_palette[cube] : 0;
                              c = CUBE_PALETTE[pal]; } }
                cr = c[0]; cg = c[1]; cb = c[2];
            }
            fprintf(ofp, "v %.6f %.6f %.6f %.4f %.4f %.4f\n",
                (double)out_verts[v * 3 + 0],
                (double)out_verts[v * 3 + 1],
                (double)out_verts[v * 3 + 2],
                (double)cr, (double)cg, (double)cb);
        }
        free(cc);
        for (size_t f = 0; f < n_unique_faces; f++) {
            fprintf(ofp, "f %d %d %d\n",
                flat_faces[f * 3 + 0] + 1,
                flat_faces[f * 3 + 1] + 1,
                flat_faces[f * 3 + 2] + 1);
        }
        if (fclose(ofp) != 0) RAISE(IO_Failed);
        {
            char binary_path[2048];
            if (MeshBin_companion_path(out_path, binary_path,
                                       sizeof binary_path) != 0 ||
                strcmp(binary_path, out_path) == 0 ||
                MeshBin_write(binary_path, out_verts, out_nv, flat_faces,
                              n_unique_faces, NULL) != 0) {
                fprintf(stderr,
                        "grid_weld: cannot write authoritative binary "
                        "companion for %s\n", out_path);
                RAISE(IO_Failed);
            }
            fprintf(stderr,
                "Wrote %s + %s (%zu verts, %zu green weld verts, %zu faces)\n",
                binary_path, out_path, out_nv, n_weld_verts, n_unique_faces);
        }
        gw_phase("write_obj");

        if (weld_verts_out) {
            FILE *wfp = fopen(weld_verts_out, "wb");
            if (!wfp) {
                fprintf(stderr,
                    "grid_weld: cannot open weld-verts file %s\n",
                    weld_verts_out);
            } else {
                uint32_t nv32 = (uint32_t)color_nv;
                fwrite(&nv32, sizeof(uint32_t), 1, wfp);
                fwrite(vert_is_weld, sizeof(uint8_t), color_nv, wfp);
                fclose(wfp);
                fprintf(stderr,
                    "Wrote %s (%zu weld-vert flags)\n",
                    weld_verts_out, color_nv);
            }
        }

        if (lineage_out) {
            FILE *lfp = fopen(lineage_out, "wb");
            const char magic[8] = {'G','W','L','I','N','2','\r','\n'};
            uint32_t version = 2, reserved = 0;
            uint64_t nv64 = (uint64_t)out_nv;
            uint64_t sf64 = (uint64_t)lineage_source_faces;
            uint64_t nc64 = (uint64_t)cubes.n;
            int lineage_ok = lfp != NULL && lineage_parent0 != NULL &&
                             lineage_parent1 != NULL &&
                             lineage_source_faces <= n_unique_faces &&
                             cubes.n > 0;
            if (lineage_ok)
                lineage_ok = fwrite(magic, 1, sizeof(magic), lfp) == sizeof(magic) &&
                    fwrite(&version, sizeof(version), 1, lfp) == 1 &&
                    fwrite(&reserved, sizeof(reserved), 1, lfp) == 1 &&
                    fwrite(&nv64, sizeof(nv64), 1, lfp) == 1 &&
                    fwrite(&sf64, sizeof(sf64), 1, lfp) == 1 &&
                    fwrite(&nc64, sizeof(nc64), 1, lfp) == 1;
            for (size_t c = 0; lineage_ok && c < cubes.n; c++) {
                char id[48] = {0};
                size_t length = strlen(cubes.ids[c]);
                if (length >= sizeof id) {
                    lineage_ok = 0;
                    break;
                }
                memcpy(id, cubes.ids[c], length);
                lineage_ok = fwrite(id, 1, sizeof id, lfp) == sizeof id;
            }
            if (lineage_ok)
                lineage_ok =
                    fwrite(lineage_parent0, sizeof(*lineage_parent0),
                           out_nv, lfp) == out_nv &&
                    fwrite(lineage_parent1, sizeof(*lineage_parent1),
                           out_nv, lfp) == out_nv;
            if (lfp) fclose(lfp);
            if (!lineage_ok) {
                fprintf(stderr,
                    "grid_weld: cannot write complete lineage file %s\n",
                    lineage_out);
                RAISE(IO_Failed);
            }
            fprintf(stderr,
                "Wrote %s (%zu vertices, %zu certified source faces, "
                "%zu explicit source cubes)\n",
                lineage_out, out_nv, lineage_source_faces, cubes.n);
        }

        char report_path[1024];
        snprintf(report_path, sizeof(report_path), "%s.weld_report.json",
                 out_path);
        FILE *rp = fopen(report_path, "w");
        if (rp) {
            fprintf(rp,
                "{\n"
                "  \"cubes_discovered\": %zu,\n"
                "  \"cubes_processed\": %zu,\n"
                "  \"cubes_skipped_missing_placed\": %zu,\n"
                "  \"total_input_verts\": %zu,\n"
                "  \"total_unique_verts\": %zu,\n"
                "  \"total_input_faces\": %zu,\n"
                "  \"total_unique_faces\": %zu,\n"
                "  \"recoarsen\": {\n"
                "    \"collapses\": %zu,\n"
                "    \"faces_in\":  %zu,\n"
                "    \"faces_out\": %zu\n"
                "  },\n"
                "  \"band_cvt\": {\n"
                "    \"patches_accepted\": %zu,\n"
                "    \"patches_rejected\": %zu,\n"
                "    \"band_faces_in\":  %zu,\n"
                "    \"band_faces_out\": %zu\n"
                "  },\n"
                "  \"intersection_cleanup\": {\n"
                "    \"planes\":          %zu,\n"
                "    \"masked_faces\":    %zu,\n"
                "    \"candidate_pairs\": %zu,\n"
                "    \"conflicts\":       %zu,\n"
                "    \"overlap_pairs\":   %zu,\n"
                "    \"stab_pairs\":      %zu,\n"
                "    \"fold_pairs\":      %zu,\n"
                "    \"faces_deleted\":   %zu,\n"
                "    \"pinch_splits\":    %zu\n"
                "  },\n"
                "  \"disk_topology\": {\n"
                "    \"components\":       %zu,\n"
                "    \"already_disks\":    %zu,\n"
                "    \"repaired\":         %zu,\n"
                "    \"loops_before\":     %zu,\n"
                "    \"loops_after\":      %zu,\n"
                "    \"handles_opened\":   %zu,\n"
                "    \"seam_edges\":       %zu,\n"
                "    \"faces_in\":         %zu,\n"
                "    \"faces_out\":        %zu,\n"
                "    \"face_preserving\":  %s,\n"
                "    \"certificate\":      %s\n"
                "  },\n"
                "  \"embedded_geometry\": {\n"
                "    \"candidate_pairs\": %zu,\n"
                "    \"conflicts\":       %zu,\n"
                "    \"overlap_pairs\":   %zu,\n"
                "    \"stab_pairs\":      %zu,\n"
                "    \"fold_pairs\":      %zu,\n"
                "    \"certificate\":      %s\n"
                "  },\n"
                "  \"manifold_audit\": {\n"
                "    \"unpaired\":     %zu,\n"
                "    \"non_manifold\": %zu,\n"
                "    \"same_dir_pairs\": %zu,\n"
                "    \"manifold_pairs\": %zu,\n"
                "    \"pinch_verts\":   %zu\n"
                "  }\n"
                "}\n",
                cubes.n, cubes_loaded, placed_missing,
                total_in_verts, out_nv, total_in_faces,
                n_unique_faces,
                recoarsen_collapses, recoarsen_faces_in, recoarsen_faces_out,
                bandcvt_accepted, bandcvt_rejected,
                bandcvt_faces_in, bandcvt_faces_out,
                intersection_planes, intersection_masked_faces,
                intersection_stats.candidate_pairs,
                intersection_stats.conflicts,
                intersection_stats.overlap_pairs,
                intersection_stats.stab_pairs,
                intersection_stats.fold_pairs,
                intersection_stats.faces_deleted,
                intersection_pinch_splits,
                disk_topology_components,
                disk_topology_stats.already_disks,
                disk_topology_stats.repaired_components,
                disk_topology_stats.boundary_loops_before,
                disk_topology_stats.boundary_loops_after,
                disk_topology_stats.handles_opened,
                disk_topology_stats.seam_edges,
                disk_topology_stats.faces_in,
                disk_topology_stats.faces_out,
                disk_topology_stats.faces_in == disk_topology_stats.faces_out ?
                    "true" : "false",
                disk_topology_certificate ? "true" : "false",
                embedded_stats.candidate_pairs,
                embedded_stats.conflicts,
                embedded_stats.overlap_pairs,
                embedded_stats.stab_pairs,
                embedded_stats.fold_pairs,
                embedded_geometry_certificate ? "true" : "false",
                ms.unpaired, ms.non_manifold,
                ms.same_dir_pairs, ms.manifold_pairs, pinch_verts);
            fclose(rp);
            fprintf(stderr, "Wrote %s\n", report_path);
        }

        if (!disk_topology_certificate ||
            ms.non_manifold > 0 || pinch_verts > 0 ||
            ms.same_dir_pairs > 0) {
            fprintf(stderr,
                "ERROR: final topology/orientation audit failed (disk=%d, "
                "non_manifold=%zu pinch_verts=%zu same_dir=%zu)\n",
                disk_topology_certificate, ms.non_manifold,
                pinch_verts, ms.same_dir_pairs);
            ok = 0;
        } else {
            ok = 1;
        }

grid_weld_complete:
        ;
    EXCEPT(Arena_Failed)
        fprintf(stderr, "grid_weld: arena OOM\n");
    EXCEPT(IO_Failed)
        fprintf(stderr, "grid_weld: I/O failure\n");
    END_TRY;

    Arena_dispose(&arena);
    gw_axis_table_free(&axis_table);
    return ok ? 0 : 1;
}
