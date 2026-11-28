#include "asm_place.h"
#include "asm_metric.h"
#include "asm_continuity.h"
#include "asm_contacts.h"
#include "asm_field.h"
#include "asm_store.h"
#include "asm_audit.h"
#include "asm_report.h"
#include "../common/uv_guard.h"
#include "../flatten/sparse_solve.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/ves_omp.h"

#include "../common/pipeline_constants.h"
#include "../common/union_find.h"
#include "../common/ves_platform.h"

#define APL_PI 3.14159265358979323846
#define APL_MAX_CLUSTERS 4
#define APL_V_FROM_AXIS 1
#define APL_V_AXIS_TOL 30.0
#define APL_V_AXIS_MAX_SHIFT 100.0
#define APL_V_MODEL_MAX_CORR 0.7
#define APL_V_MODEL_B_MIN 0.95
#define APL_V_MODEL_B_MAX 1.05
#define APL_V_MODEL_MAX_C 0.10
#define APL_TURN_REMEASURE 1
#define APL_T0_REMEASURE 0
#define APL_HOP_TURN_MAX_REL_ERR 0.0
#define APL_CURV_CACHE 1
#define APL_CLUSTER_QSORT 1
#define APL_RIM_SQUARED_CULL 1
#define APL_PROFILE_EVERY 500
#define APL_TICK(field) do { double tn_ = ves_clock_sec(); st->field += tn_ - tq; tq = tn_; } while (0)
#define APL_SEAM_FIRST 1
#define APL_SEAM_FRONTIER_FIRST 0
#define APL_SWITCHED_SEAM_ANCHORS 1
#define APL_SEAM_CLUSTER_VOX 24.0
#define APL_MAX_SEAM_CLUSTERS 2
#define APL_MAX_CLUSTERS_TOTAL (APL_MAX_CLUSTERS + 2 * APL_MAX_SEAM_CLUSTERS)
#define APL_SEAM_MIN_ANCHORS 6
#define APL_SEAM_GATE_MIN 8.0
#define APL_SEAM_GATE_RMS_X 4.0
#define APL_SEAM_V_PIN 0
#define APL_SMALL_HOP_NEEDS_RIM 1
#define APL_SMALL_AREA_VOX2 4.0e4
#define APL_DEFER_RETRY_RIM 1
#define APL_DEFER_ONLY_WITH_SEAMS 0
#define APL_DEFER_LAST_CALL 0
#define APL_OUTLINE_CROP_VOX 8.0
#define APL_TURN_FROM_RADIUS 1
#define APL_TURN_MIN 150.0
#define APL_TURN_MAX 12500.0
#define APL_METRIC_RESOLVE 0
#define APL_METRIC_EVERY 8
#define APL_TURN_MODEL_POOL 0
#define APL_TURN_MODEL_MIN_PAIRS 50
#define APL_TURN_MODEL_MAX_RESID 0.15
#define APL_PAIR_K_FROM_PROBE 1
#define APL_LAYER_KINT 0
#define APL_COORD_CALIBRATE 0
#define APL_TURN_FIELD 0
#define APL_TFIELD_DU 250.0
#define APL_TFIELD_DW 0.6
#define APL_TFIELD_MIN_OBS 5
#define APL_METRIC_MAX_SHIFT 0.05
#define APL_TURN_KHOP 0
#define APL_LAYER_UNKNOWN (-1e30)
#define APL_BORROW_TURNS 0
#define APL_BORROW_REACH 0.35

void AsmPlace_default_opts(AsmPlaceOpts *o)
{
    o->min_anchors = ASM_PLACE_MIN_ANCHORS;
    o->max_rms = ASM_PLACE_MAX_RMS;
    o->rel_rms = 0.04;
    o->max_layers = 3;
    o->layer_tol = 0.35;
    o->min_turn_hits = 3;
    o->max_contra_ratio = 0.25;
    o->max_rot_refine = 0.6;
    o->max_rim_ratio = 0.3;
    o->crosswrap_anchor_veto = ASM_WRAP_VETO_ANCHOR;
    o->stress_band = ASM_STRESS_BAND;
    o->diag_dir = NULL;
    o->axis = NULL;
}

static void apl_pose(const AsmChart *c, double u, double v, double *gx, double *gy)
{
    if (c->flags & ASM_CHART_MIRROR) u = -u;
    double ct = cos(c->pose_theta), st = sin(c->pose_theta);
    *gx = ct * u - st * v + c->pose_x;
    *gy = st * u + ct * v + c->pose_y;
}

static void apl_vertex_point(const AsmChart *c, int32_t v, double *gx, double *gy)
{
    double p[2]; AsmChart_point(c,(size_t)v,p); *gx=p[0]; *gy=p[1];
}

static void apl_face_point(const AsmChart *c, int32_t f, double l0, double l1, double *gx, double *gy)
{
    const int32_t *fv = &c->faces[(size_t)f*3];
    double l2 = 1.0 - l0 - l1;
    if (c->placed_uv) {
        *gx=l0*c->placed_uv[2*(size_t)fv[0]]+l1*c->placed_uv[2*(size_t)fv[1]]+l2*c->placed_uv[2*(size_t)fv[2]];
        *gy=l0*c->placed_uv[2*(size_t)fv[0]+1]+l1*c->placed_uv[2*(size_t)fv[1]+1]+l2*c->placed_uv[2*(size_t)fv[2]+1];
        return;
    }
    double u = l0 * c->uv[(size_t)fv[0]*2] + l1 * c->uv[(size_t)fv[1]*2] + l2 * c->uv[(size_t)fv[2]*2];
    double v = l0 * c->uv[(size_t)fv[0]*2+1] + l1 * c->uv[(size_t)fv[1]*2+1] + l2 * c->uv[(size_t)fv[2]*2+1];
    apl_pose(c, u, v, gx, gy);
}

static void apl_compose(AsmChart *c, double theta, double tx, double ty, int mirror)
{
    double x = c->pose_x, y = c->pose_y;
    if (mirror) x = -x;
    double ct = cos(theta), st = sin(theta);
    c->pose_x = ct * x - st * y + tx;
    c->pose_y = st * x + ct * y + ty;
    c->pose_theta = theta + (mirror ? -c->pose_theta : c->pose_theta);
    if (mirror) c->flags ^= ASM_CHART_MIRROR;
}

static int apl_same_pose(const AsmChart *a, const AsmChart *b, double tolerance)
{
    if (a->id != b->id || a->nv != b->nv || !a->uv || !b->uv || !a->nv) return 0;
    double ct_a = cos(a->pose_theta), sn_a = sin(a->pose_theta);
    double ct_b = cos(b->pose_theta), sn_b = sin(b->pose_theta);
    double ma = a->flags & ASM_CHART_MIRROR ? -1.0 : 1.0;
    double mb = b->flags & ASM_CHART_MIRROR ? -1.0 : 1.0;
    for (size_t v = 0; v < a->nv; v++) {
        double au = ma*a->uv[2*v], av = a->uv[2*v+1];
        double bu = mb*b->uv[2*v], bv = b->uv[2*v+1];
        double dx = ct_a*au-sn_a*av+a->pose_x - (ct_b*bu-sn_b*bv+b->pose_x);
        double dy = sn_a*au+ct_a*av+a->pose_y - (sn_b*bu+ct_b*bv+b->pose_y);
        if (!isfinite(dx) || !isfinite(dy) || dx*dx+dy*dy > tolerance*tolerance) return 0;
    }
    return 1;
}

static int apl_cmp_double(const void *x, const void *y)
{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

static int apl_solve3(double A[9], double b[3], double x[3])
{
    for (int col = 0; col < 3; col++) {
        int piv = col;
        for (int r = col + 1; r < 3; r++) if (fabs(A[r*3+col]) > fabs(A[piv*3+col])) piv = r;
        for (int k = 0; k < 3; k++) { double t = A[col*3+k]; A[col*3+k] = A[piv*3+k]; A[piv*3+k] = t; }
        { double t = b[col]; b[col] = b[piv]; b[piv] = t; }
        if (fabs(A[col*3+col]) < 1e-12) return -1;
        for (int r = 0; r < 3; r++) {
            if (r == col) continue;
            double f = A[r*3+col] / A[col*3+col];
            for (int k = 0; k < 3; k++) A[r*3+k] -= f * A[col*3+k];
            b[r] -= f * b[col];
        }
    }
    for (int k = 0; k < 3; k++) x[k] = b[k] / A[k*3+k];
    return 0;
}

static int apl_chart_quadric(const AsmChart *c, double tu[3], double tv[3], double huu[3])
{
    double M[36] = { 0 }, R[3][6] = { { 0 } };
    double cu = 0.0, cv = 0.0;
    size_t n = c->nv;
    for (size_t i = 0; i < n; i++) { double gx, gy; apl_vertex_point(c, (int32_t)i, &gx, &gy); cu += gx; cv += gy; }
    cu /= (double)n; cv /= (double)n;
    for (size_t i = 0; i < n; i++) {
        double gx, gy;
        apl_vertex_point(c, (int32_t)i, &gx, &gy);
        double u = gx - cu, v = gy - cv;
        double phi[6] = { 1.0, u, v, u*u, u*v, v*v };
        for (int r = 0; r < 6; r++) {
            for (int k = 0; k < 6; k++) M[r*6+k] += phi[r] * phi[k];
            for (int d = 0; d < 3; d++) R[d][r] += phi[r] * c->xyz[i*3 + (size_t)d];
        }
    }
    for (int col = 0; col < 6; col++) {
        int piv = col;
        for (int r = col + 1; r < 6; r++) if (fabs(M[r*6+col]) > fabs(M[piv*6+col])) piv = r;
        for (int k = 0; k < 6; k++) { double t = M[col*6+k]; M[col*6+k] = M[piv*6+k]; M[piv*6+k] = t; }
        for (int d = 0; d < 3; d++) { double t = R[d][col]; R[d][col] = R[d][piv]; R[d][piv] = t; }
        if (fabs(M[col*6+col]) < 1e-9) return -1;
        for (int r = 0; r < 6; r++) {
            if (r == col) continue;
            double f = M[r*6+col] / M[col*6+col];
            for (int k = 0; k < 6; k++) M[r*6+k] -= f * M[col*6+k];
            for (int d = 0; d < 3; d++) R[d][r] -= f * R[d][col];
        }
    }
    for (int d = 0; d < 3; d++) {
        tu[d] = R[d][1] / M[1*6+1];
        tv[d] = R[d][2] / M[2*6+2];
        huu[d] = 2.0 * R[d][3] / M[3*6+3];
    }
    return 0;
}

typedef struct AplTurn { double tx, ty; int n; } AplTurn;

typedef struct AplOrient {
    double theta;
    double grad;
    double bend;
    double turn_u;
    int    has_turn;
    int    mirrored;
} AplOrient;

static void apl_orient_measure(const AsmRun *run, int32_t comp, AplOrient *o)
{
    size_t nc = run->n_charts;
    double sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, sxz = 0, syz = 0, sz = 0, n = 0;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != comp) continue;
        for (size_t k = 0; k < c->nv; k += 2) {
            double gx, gy;
            apl_vertex_point(c, (int32_t)k, &gx, &gy);
            double z = c->xyz[k*3];
            sxx += gx*gx; sxy += gx*gy; syy += gy*gy; sx += gx; sy += gy; sxz += gx*z; syz += gy*z; sz += z; n += 1;
        }
    }
    o->theta = 0.0; o->grad = 0.0;
    if (n < 6) return;
    double A[9] = { sxx, sxy, sx, sxy, syy, sy, sx, sy, n }, b[3] = { sxz, syz, sz }, x[3];
    if (apl_solve3(A, b, x) != 0) return;
    double ga = x[0], gb = x[1], gl = hypot(ga, gb);
    o->grad = gl;
    if (gl > 0.05) o->theta = 0.5 * APL_PI - atan2(gb, ga);
}

static void apl_orient_rotate(AsmRun *run, int32_t comp, AplOrient *o)
{
    apl_orient_measure(run,comp,o);
    size_t nc=run->n_charts;
    for (size_t i = 0; i < nc; i++) {
        AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != comp) continue;
        apl_compose(c, o->theta, 0.0, 0.0, 0);
    }
}

static int8_t apl_axis_inward(const AsmChart *c, const AsmAxis *axis, double inward[3])
{
    if (!axis || !c->xyz || !c->nrm || c->nv < 12) return 0;
    size_t stride=(c->nv+127)/128, samples=0, positive=0, negative=0;
    double sum[3]={0};
    for (size_t v=0; v<c->nv; v+=stride) {
        double p[3],n[3],foot[3],r;
        for (int d=0; d<3; d++) { p[d]=c->xyz[3*v+d]; n[d]=c->nrm[3*v+d]; }
        samples++;
        double nl=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (!(nl>1e-9) || !isfinite(nl)) continue;
        AsmAxis_project(axis,p,NULL,&r,NULL,foot);
        if (!(r>5) || !isfinite(r)) continue;
        double dot=0;
        for (int d=0; d<3; d++) dot+=n[d]*(p[d]-foot[d])/(nl*r);
        if (fabs(dot)<.25 || !isfinite(dot)) continue;
        if (dot>0) positive++; else negative++;
        for (int d=0; d<3; d++) sum[d]-=(p[d]-foot[d])/(r*r);
    }
    size_t known=positive+negative, majority=positive>negative ? positive : negative;
    if (known<6 || 4*known<3*samples || 10*majority<9*known) return 0;
    for (int d=0; d<3; d++) inward[d]=sum[d]/(double)known;
    return positive>negative ? (int8_t)1 : (int8_t)-1;
}

static void apl_component_out_sides(const AsmRun *run, int32_t comp, const AsmAxis *axis, int8_t *out_side)
{
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != comp || c->nv < 12) continue;
        double tu[3], tv[3], huu[3];
        int8_t axial=apl_axis_inward(c,axis,huu);
        if (axial) { out_side[c->id]=axial; continue; }
        if (apl_chart_quadric(c, tu, tv, huu) != 0) continue;
        double hl = sqrt(huu[0]*huu[0] + huu[1]*huu[1] + huu[2]*huu[2]);
        if (hl < 1e-6) continue;
        double mn[3] = { 0.0, 0.0, 0.0 };
        for (size_t k = 0; k < c->nv; k++) { mn[0] += c->nrm[k*3]; mn[1] += c->nrm[k*3+1]; mn[2] += c->nrm[k*3+2]; }
        double inward = mn[0]*huu[0] + mn[1]*huu[1] + mn[2]*huu[2];
        if (fabs(inward) < 1e-9) continue;
        out_side[c->id] = inward > 0.0 ? (int8_t)-1 : (int8_t)1;
    }
}

static void apl_orient_votes(const AsmRun *run, int32_t comp, const AsmAxis *axis,
                             const AplTurn *turn, AplOrient *o, const int8_t *out_side)
{
    size_t nc = run->n_charts;
    double bs = 0.0, bw = 0.0, ts = 0.0, tw = 0.0;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != comp || c->nv < 12) continue;
        double tu[3], tv[3], huu[3];
        if (apl_chart_quadric(c, tu, tv, huu) != 0) continue;
        apl_axis_inward(c,axis,huu);
        double hl = sqrt(huu[0]*huu[0] + huu[1]*huu[1] + huu[2]*huu[2]);
        if (hl < 1e-6) continue;
        double ux = tu[2], uy = tu[1], uz = tu[0], vx = tv[2], vy = tv[1], vz = tv[0];
        double Nx = uy*vz - uz*vy, Ny = uz*vx - ux*vz, Nz = ux*vy - uy*vx;
        double Nl = sqrt(Nx*Nx + Ny*Ny + Nz*Nz);
        if (Nl > 1e-9) {
            double dot = (Nx*huu[2] + Ny*huu[1] + Nz*huu[0]) / (Nl * hl);
            bs += c->area3d * hl * dot; bw += c->area3d * hl;
        }
        int8_t os=out_side[c->id];
        if (!os) continue;
        const AplTurn *t = &turn[(size_t)c->id * 2 + (os > 0 ? 1 : 0)];
        if (t->n <= 0) continue;
        double len = hypot(t->tx, t->ty);
        if (len < 1e-6) continue;
        ts += c->area3d * t->tx / len; tw += c->area3d;
    }
    o->bend = bw > 0.0 ? bs / bw : 0.0;
    o->turn_u = tw > 0.0 ? ts / tw : 0.0;
    o->has_turn = tw > 0.0;
}

static void apl_mirror_component(AsmRun *run, int32_t comp)
{
    for (size_t i = 0; i < run->n_charts; i++) {
        AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != comp) continue;
        apl_compose(c, 0.0, 0.0, 0.0, 1);
    }
}

static int apl_turn_pair_ok(const AsmRun *run, const uint8_t *comp_placed, const AsmLayerPair *lp)
{
    const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
    if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) return 0;
    if (comp_placed[A->component] != 1 || comp_placed[B->component] != 1) return 0;
    if (A->component != B->component) return 0;
    return 1;
}

static void apl_measure_turns(const AsmRun *run, const uint8_t *comp_placed, const int8_t *out_side, const double *lidx, AplTurn *turn, int min_hits)
{
    Arena_Mark mark = Arena_save(run->arena);
    size_t nc = run->n_charts;
    size_t *off = ARENA_CALLOC(run->arena, nc * 2 + 1, sizeof(size_t));
    double *min_d = ARENA_ALLOC(run->arena, (nc * 2 + 1) * sizeof(double));
    for (size_t i = 0; i < nc * 2; i++) { turn[i].tx = 0.0; turn[i].ty = 0.0; turn[i].n = 0; min_d[i] = 1e300; }
    double *bx = NULL, *by = NULL;
    size_t *fill = NULL;
    for (int pass = 0; pass < 3; pass++) {
        for (size_t p = 0; p < run->n_layers; p++) {
            const AsmLayerPair *lp = &run->layers[p];
            if (!apl_turn_pair_ok(run, comp_placed, lp)) continue;
            if (APL_PAIR_K_FROM_PROBE && lp->k > 1) continue;
            int rev = 0, side_b = 0;
            if (out_side[lp->a] != 0 && out_side[lp->b] != 0) {
                side_b = (lp->side == out_side[lp->a]) ? -out_side[lp->b] : out_side[lp->b];
                rev = 1;
            }
            size_t sa = (size_t)lp->a * 2 + (lp->side > 0 ? 1 : 0);
            size_t sb = (size_t)lp->b * 2 + (side_b > 0 ? 1 : 0);
            if (pass == 0) {
                if (lp->d_median < min_d[sa]) min_d[sa] = lp->d_median;
                if (rev && lp->d_median < min_d[sb]) min_d[sb] = lp->d_median;
                continue;
            }
            int fwd = lp->d_median <= 1.35 * min_d[sa];
            if (rev) rev = lp->d_median <= 1.35 * min_d[sb];
            if (!fwd && !rev) continue;
            const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
            double kdiv = 1.0;
            if (APL_TURN_KHOP && lidx != NULL && lidx[lp->a] > APL_LAYER_UNKNOWN && lidx[lp->b] > APL_LAYER_UNKNOWN) {
                double dl = fabs(lidx[lp->b] - lidx[lp->a]);
                int kk = (int)floor(dl + 0.5);
                if (kk < 1) kk = 1;
                if (kk > 4) kk = 4;
                kdiv = (double)kk;
            }
            for (int k = 0; k < lp->hit_count; k++) {
                const AsmLayerHit *h = &run->layer_hits[(size_t)lp->hit_first + (size_t)k];
                if (fabs(h->dist - lp->d_median) > 0.25 * lp->d_median + 1.0) continue;
                if (pass == 1) { if (fwd) off[sa + 1]++; if (rev) off[sb + 1]++; continue; }
                double ox, oy, hx, hy;
                apl_vertex_point(A, h->va, &ox, &oy);
                apl_face_point(B, h->fb, h->l0, h->l1, &hx, &hy);
                if (fwd) { size_t q = off[sa] + fill[sa]++; bx[q] = (hx - ox) / kdiv; by[q] = (hy - oy) / kdiv; }
                if (rev) { size_t q = off[sb] + fill[sb]++; bx[q] = (ox - hx) / kdiv; by[q] = (oy - hy) / kdiv; }
            }
        }
        if (pass == 1) {
            for (size_t i = 0; i < nc * 2; i++) off[i + 1] += off[i];
            size_t total = off[nc * 2];
            bx = ARENA_ALLOC(run->arena, (total ? total : 1) * sizeof(double));
            by = ARENA_ALLOC(run->arena, (total ? total : 1) * sizeof(double));
            fill = ARENA_CALLOC(run->arena, nc * 2 + 1, sizeof(size_t));
        }
    }
    for (size_t sl = 0; sl < nc * 2; sl++) {
        size_t n = off[sl + 1] - off[sl];
        if ((int)n < min_hits) continue;
        qsort(bx + off[sl], n, sizeof(double), apl_cmp_double);
        qsort(by + off[sl], n, sizeof(double), apl_cmp_double);
        double tx = bx[off[sl] + n / 2], ty = by[off[sl] + n / 2], len = hypot(tx, ty);
        if (len < APL_TURN_MIN || len > APL_TURN_MAX) continue;
        turn[sl].tx = tx; turn[sl].ty = ty; turn[sl].n = (int)n;
    }
    Arena_restore(run->arena, mark);
}

static int apl_cmp_double_fwd(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }

static size_t apl_borrow_turns(const AsmRun *run, const uint8_t *comp_placed, const int8_t *out_side, const double *lidx, AplTurn *turn)
{
    (void)lidx;
    size_t nc = run->n_charts, borrowed = 0;
    Arena_Mark mark = Arena_save(run->arena);
    int32_t *meas = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(int32_t));
    double *mx = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double)), *my = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    for (int want_out = 0; want_out < 2; want_out++) {
        size_t nm = 0;
        for (size_t i = 0; i < nc; i++) {
            if (out_side[i] == 0) continue;
            int sd = want_out ? (out_side[i] > 0 ? 1 : 0) : (out_side[i] > 0 ? 0 : 1);
            if (turn[i * 2 + (size_t)sd].n > 0) { apl_pose(&run->charts[i], 0.0, 0.0, &mx[nm], &my[nm]); meas[nm++] = (int32_t)i; }
        }
        if (nm == 0) continue;
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (out_side[i] == 0) continue;
            int side = want_out ? (out_side[i] > 0 ? 1 : 0) : (out_side[i] > 0 ? 0 : 1);
            AplTurn *t = &turn[i * 2 + (size_t)side];
            if (t->n != 0 || !AsmChart_in_layout(c) || c->component < 0 || comp_placed[c->component] != 1) continue;
            double gx, gy;
            apl_pose(c, 0.0, 0.0, &gx, &gy);
            double lens[8], sx = 0.0, sy = 0.0, dist[8];
            size_t nd = 0;
            for (size_t k = 0; k < nm; k++) {
                int32_t d = meas[k];
                int dside = want_out ? (out_side[d] > 0 ? 1 : 0) : (out_side[d] > 0 ? 0 : 1);
                const AplTurn *src = &turn[(size_t)d * 2 + (size_t)dside];
                double len = hypot(src->tx, src->ty);
                double du = fabs(mx[k] - gx), dv = fabs(my[k] - gy);
                if (du > APL_BORROW_REACH * len || dv > 300.0) continue;
                double dd = hypot(du, dv);
                if (nd < 8) { lens[nd] = len; dist[nd] = dd; sx += src->tx / len; sy += src->ty / len; nd++; }
                else {
                    size_t worst = 0;
                    for (size_t q = 1; q < 8; q++) if (dist[q] > dist[worst]) worst = q;
                    if (dd < dist[worst]) { lens[worst] = len; dist[worst] = dd; }
                }
            }
            if (nd < 2) continue;
            qsort(lens, nd, sizeof(double), apl_cmp_double_fwd);
            double med = lens[nd / 2], dl = hypot(sx, sy);
            if (dl < 1e-9) continue;
            t->tx = med * sx / dl; t->ty = med * sy / dl; t->n = -(int)nd;
            borrowed++;
        }
    }
    Arena_restore(run->arena, mark);
    return borrowed;
}

static double apl_turn_t0(const AsmRun *run, const AplTurn *turn, const int8_t *out_side, const int32_t *orig_comp, int32_t primary,
                          int *measured, size_t *ntl_out)
{
    Arena_Mark mt = Arena_save(run->arena);
    size_t nc = run->n_charts;
    double *tl = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    size_t ntl = 0; double ksum = 0.0, kw = 0.0, t0;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || orig_comp[i] != primary || out_side[i] == 0) continue;
        const AplTurn *t = &turn[i * 2 + (out_side[i] > 0 ? 1 : 0)];
        if (t->n > 0) tl[ntl++] = hypot(t->tx, t->ty);
        if (c->nv >= 12) {
            double tu[3], tv[3], huu[3];
            if (apl_chart_quadric(c, tu, tv, huu) == 0) {
                double hl = sqrt(huu[0]*huu[0] + huu[1]*huu[1] + huu[2]*huu[2]);
                if (hl > 1e-4 && hl < 0.2) { ksum += c->area3d * hl; kw += c->area3d; }
            }
        }
    }
    *measured = 0;
    if (ntl >= 3) { qsort(tl, ntl, sizeof(double), apl_cmp_double); t0 = tl[ntl / 2]; *measured = 1; }
    else if (kw > 0.0) t0 = 2.0 * APL_PI / (ksum / kw);
    else t0 = 2.0 * APL_PI * 150.0;
    if (t0 < APL_TURN_MIN) t0 = APL_TURN_MIN;
    if (t0 > APL_TURN_MAX) t0 = APL_TURN_MAX;
    *ntl_out = ntl;
    Arena_restore(run->arena, mt);
    return t0;
}

static size_t apl_measure_growth(const AsmRun *run, const AplTurn *turn, const int8_t *out_side, double *growth)
{
    Arena_Mark mark = Arena_save(run->arena);
    size_t nc = run->n_charts, n = 0;
    double *g = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    for (size_t c = 0; c < nc; c++) {
        if (out_side[c] == 0) continue;
        const AplTurn *to = &turn[c * 2 + (out_side[c] > 0 ? 1 : 0)];
        const AplTurn *ti = &turn[c * 2 + (out_side[c] > 0 ? 0 : 1)];
        if (to->n <= 0 || ti->n <= 0) continue;
        g[n++] = hypot(to->tx, to->ty) - hypot(ti->tx, ti->ty);
    }
    if (n >= 3) {
        qsort(g, n, sizeof(double), apl_cmp_double);
        *growth = g[n/2];
    } else *growth = 2.0 * APL_PI * run->layer_d_global;
    Arena_restore(run->arena, mark);
    return n;
}

static int apl_turn_for(const AplTurn *turn, int32_t c, int side, int8_t out_side, double growth, double *tx, double *ty)
{
    size_t s = (size_t)c * 2 + (side > 0 ? 1 : 0), o = (size_t)c * 2 + (side > 0 ? 0 : 1);
    if (turn[s].n != 0) { *tx = turn[s].tx; *ty = turn[s].ty; return 1; }
    if (turn[o].n != 0 && out_side != 0) {
        double len = hypot(turn[o].tx, turn[o].ty);
        if (len < 1e-6) return 0;
        double target = side == out_side ? len + growth : len - growth;
        if (target < 0.3 * len || target < APL_TURN_MIN || target > APL_TURN_MAX) return 0;
        double f = target / len;
        *tx = -turn[o].tx * f; *ty = -turn[o].ty * f;
        return 2;
    }
    return 0;
}

static int apl_turn_k(const AplTurn *turn, int32_t c, int side, int8_t out_side, double growth, int k, double *tx, double *ty)
{
    double t1x, t1y;
    int src = apl_turn_for(turn, c, side, out_side, growth, &t1x, &t1y);
    if (!src) return 0;
    if (k == 1) { *tx = t1x; *ty = t1y; return src; }
    if (src != 1) return 0;
    if (out_side == 0) return 0;
    double len = hypot(t1x, t1y);
    if (len < 1e-6) return 0;
    double extra = growth * (double)k * (double)(k - 1) * 0.5;
    double total = side == out_side ? (double)k * len + extra : (double)k * len - extra;
    if (total < 0.3 * len) return 0;
    *tx = t1x * total / len; *ty = t1y * total / len;
    return src;
}

enum {
    APL_TSRC_SEAM = 0,
    APL_TSRC_MEASURED,
    APL_TSRC_MEASURED_K,
    APL_TSRC_OPPOSITE,
    APL_TSRC_RADIUS,
    APL_TSRC_LAYER,
    APL_TSRC_STRETCH,
    APL_TSRC_T0,
    APL_TSRC_N
};

typedef struct AplAnchor {
    double sx, sy;
    double tx, ty;
    double w;
    int    k, estimated, reverse;
    int    turn_src;
    int32_t chart_p; int8_t side; double Tx, Ty;
    int32_t chart_c;
    int    cluster;
    double seam_rms;
    int32_t relation, source_run;
    int measured_order, measured_turn, source_flip;
    double measured_x, measured_y;
    double local_u, local_v;
    size_t observation;
    const AsmLayerHit *hit;
} AplAnchor;

static double apl_fit_translation(const AplAnchor *a, const double *w, size_t n, double max_rot,
                                  double *theta, double *tx, double *ty)
{
    double sw = 0.0, cs0 = 0.0, cs1 = 0.0, cd0 = 0.0, cd1 = 0.0;
    for (size_t i = 0; i < n; i++) { sw += w[i]; cs0 += w[i]*a[i].sx; cs1 += w[i]*a[i].sy; cd0 += w[i]*a[i].tx; cd1 += w[i]*a[i].ty; }
    if (!(sw > 0.0)) return 1e300;
    cs0 /= sw; cs1 /= sw; cd0 /= sw; cd1 /= sw;
    double sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sx = a[i].sx - cs0, sy = a[i].sy - cs1, dx = a[i].tx - cd0, dy = a[i].ty - cd1;
        sxx += w[i] * (sx*dx + sy*dy);
        sxy += w[i] * (sx*dy - sy*dx);
    }
    double th = atan2(sxy, sxx);
    if (th > max_rot) th = max_rot;
    if (th < -max_rot) th = -max_rot;
    double c = cos(th), s = sin(th);
    *theta = th;
    *tx = cd0 - (c*cs0 - s*cs1);
    *ty = cd1 - (s*cs0 + c*cs1);
    double e2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double px = c*a[i].sx - s*a[i].sy + *tx, py = s*a[i].sx + c*a[i].sy + *ty;
        e2 += w[i] * ((px-a[i].tx)*(px-a[i].tx) + (py-a[i].ty)*(py-a[i].ty));
    }
    return sqrt(e2 / sw);
}

typedef struct AplAxisFit {
    double mx, my, ax, ay, axis_v, slope, band, variance, dot, cross;
} AplAxisFit;

static double apl_axis_excess(double residual, double band)
{
    return copysign(fmax(0.0, fabs(residual)-band), residual);
}

static double apl_axis_fit_cost(const AplAxisFit *f, double theta)
{
    double c = cos(theta), s = sin(theta);
    double rx = c*f->mx-s*f->my, ry = s*f->mx+c*f->my;
    double perpendicular = apl_axis_excess(f->ay-f->axis_v-ry-f->slope*(f->ax-rx), f->band);
    return f->variance-2*(c*f->dot+s*f->cross)+perpendicular*perpendicular/(1+f->slope*f->slope);
}

static double apl_fit_axis_translation(const AplAnchor *a, const double *w, size_t n, double max_rot,
                                       double cu, double cv, double axis_v, double slope, double band,
                                       double *theta, double *tx, double *ty)
{
    AplAxisFit f; memset(&f, 0, sizeof f); f.axis_v = axis_v; f.slope = slope; f.band = fmax(0.0,band);
    double sw = 0;
    for (size_t i = 0; i < n; i++) if (w[i] > 0) {
        sw += w[i]; f.mx += w[i]*(a[i].sx-cu); f.my += w[i]*(a[i].sy-cv);
        f.ax += w[i]*a[i].tx; f.ay += w[i]*a[i].ty;
    }
    if (!(sw > 0)) return 1e300;
    f.mx /= sw; f.my /= sw; f.ax /= sw; f.ay /= sw;
    for (size_t i = 0; i < n; i++) if (w[i] > 0) {
        double x = a[i].sx-cu-f.mx, y = a[i].sy-cv-f.my;
        double u = a[i].tx-f.ax, v = a[i].ty-f.ay;
        f.variance += w[i]*(x*x+y*y+u*u+v*v);
        f.dot += w[i]*(x*u+y*v); f.cross += w[i]*(x*v-y*u);
    }
    f.variance /= sw; f.dot /= sw; f.cross /= sw;
    double limit = fmin(fmax(max_rot, 0.0), APL_PI);
    double best = fmax(-limit, fmin(limit, atan2(f.cross, f.dot)));
    double best_cost = apl_axis_fit_cost(&f, best), step = 2*limit/128;
    for (int j = 0; j <= 128; j++) {
        double angle = -limit+j*step, cost = apl_axis_fit_cost(&f, angle);
        if (cost < best_cost) { best = angle; best_cost = cost; }
    }
    double lo = fmax(-limit, best-step), hi = fmin(limit, best+step);
    const double golden = 0.6180339887498948482;
    double l = hi-golden*(hi-lo), r = lo+golden*(hi-lo);
    double fl = apl_axis_fit_cost(&f, l), fr = apl_axis_fit_cost(&f, r);
    for (int j = 0; j < 40 && hi-lo > 1e-12; j++) {
        if (fl < fr) { hi = r; r = l; fr = fl; l = hi-golden*(hi-lo); fl = apl_axis_fit_cost(&f, l); }
        else { lo = l; l = r; fl = fr; r = lo+golden*(hi-lo); fr = apl_axis_fit_cost(&f, r); }
    }
    double angle = 0.5*(lo+hi);
    if (apl_axis_fit_cost(&f, angle) < best_cost) best = angle;
    double c = cos(best), s = sin(best), rx = c*f.mx-s*f.my, ry = s*f.mx+c*f.my;
    double excess = apl_axis_excess(f.ay-axis_v-ry-slope*(f.ax-rx), f.band)/(1+slope*slope);
    double centroid_u = f.ax-rx+slope*excess, centroid_v = f.ay-ry-excess;
    *theta = best; *tx = centroid_u-c*cu+s*cv;
    *ty = centroid_v-s*cu-c*cv;
    double e2 = 0;
    for (size_t i = 0; i < n; i++) if (w[i] > 0) {
        double x = c*a[i].sx-s*a[i].sy+*tx-a[i].tx;
        double y = s*a[i].sx+c*a[i].sy+*ty-a[i].ty;
        e2 += w[i]*(x*x+y*y);
    }
    return sqrt(e2/sw);
}

typedef struct AplCompDiag { int status; int attempts; size_t last_anchors; double last_rms, last_gate, last_rot; int last_k1, last_k2, last_clusters;
                             int placed_by; size_t seam_anchors; unsigned rim_logged; } AplCompDiag;
enum { APL_NEVER = 0, APL_PLACED = 1, APL_REFUSED = 2, APL_REVERTED = 3, APL_PRIMARY = 4, APL_FEW_ANCHORS = 5, APL_DEFERRED = 6 };
static const char *apl_status_names[] = { "never", "placed", "refused", "reverted", "primary", "few_anchors", "deferred" };

typedef struct AplRimGrid AplRimGrid;
typedef struct AplStrict AplStrict;
typedef struct AplConsensus { size_t observations, inliers, model_inliers; double clipped_loss, model_loss; } AplConsensus;
typedef struct AplCtx {
    AsmRun *run; const AsmPlaceOpts *o; const AsmConflictOpts *copts;
    uint8_t *comp_placed; AplTurn *turn; int8_t *out_side; int32_t *orig_comp; double growth;
    AplAnchor *anchors; double *wts; size_t hcap;
    int32_t primary; double *comp_area; AplCompDiag *diag; FILE *fanch; int attempt;
    int stretch_ok; double stretch_T, stretch_s, stretch_d, stretch_rel; size_t stretch_n, stretch_pairs;
    double turn_rel;
    double cand_grad;
    AsmConflictGrid *cgrid; AplRimGrid *rgrid;
    const uint8_t *excluded_samples;
    double strip_umin, strip_umax, span_cap;
    const double *lidx;
    double t0;
    double *tfield;
    int t0_measured;
    double *s_axis, *r_axis; int have_radius;
    double *cu, *cv;
    double *tr_top, *tr_bot;
    size_t *off_t, *off_b, *fill_t, *fill_b;
    double vm_a, vm_b, vm_c; int vm_ok;
    double vm_corr;
    double vm_p90;
    int vm_pin;
    int diagnostic_axis_policy;
    int diagnostic_first_step;
    AplStrict *strict;
    double uout_x, uout_y; int uout_ok;
    double turn_c; size_t turn_c_n;
    double turn_slope;
    size_t anchors_radius;
    AsmPlaceStats *st;
    double *curv_hl; uint8_t *curv_ok;
    size_t na_seam;
    int last_silent;
    int last_shape_refused;
    const char *last_reason;
    int source_geometry_conflict;
    size_t na; int cluster, cluster_kind;
    double evidence_gate;
    uint8_t *seam_selected;
    int trial, source_flip; AsmChart *proposal; size_t nproposal;
    Arena_T proposal_uv;
    int uv_trial;
    size_t n_observations;
    AplConsensus consensus;
    FILE *fevidence, *frim, *fcut;
    double fit_cu, fit_cv, fit_s, fit_area;
    size_t cuts_preserved_source;
} AplCtx;

static double apl_axis_band(const AplCtx *x)
{
    return fmax(0.0,1.5*x->vm_p90);
}

static void apl_uv_centroid(const AsmChart *c, double *u, double *v)
{
    *u = *v = 0;
    for (size_t i = 0; i < c->nv; i++) { *u += c->uv[2*i]; *v += c->uv[2*i+1]; }
    if (c->nv) { *u /= c->nv; *v /= c->nv; }
}

static void apl_copy_candidate(AsmChart *out, const AsmChart *in, size_t n,
                                const AsmRun *run, Arena_T *uv_arena)
{
    if (*uv_arena) Arena_free(*uv_arena);
    for (size_t i = 0; i < n; i++) {
        out[i] = in[i];
        if (in[i].uv != run->charts[in[i].id].uv) {
            if (!*uv_arena) *uv_arena = Arena_new();
            out[i].uv = ARENA_ALLOC(*uv_arena,in[i].nv*2*sizeof(float));
            memcpy(out[i].uv,in[i].uv,in[i].nv*2*sizeof(float));
        }
    }
}

static void apl_commit_chart(AplCtx *x, const AsmChart *candidate, float *owned_uv)
{
    AsmChart c = *candidate;
    if (c.uv != owned_uv) {
        memcpy(owned_uv,c.uv,c.nv*2*sizeof(float)); c.uv = owned_uv;
        if (x->cu && x->cv) apl_uv_centroid(&c,&x->cu[c.id],&x->cv[c.id]);
        if (x->curv_ok) x->curv_ok[c.id] = 0;
    }
    x->run->charts[c.id] = c;
}

static void apl_layout_centroid(const AplCtx *x, const AsmChart *c, size_t i, double th, double tx, double ty, double *gx, double *gy)
{
    if (c->placed_uv) {
        double u=0,v=0;
        for(size_t k=0;k<c->nv;k++){u+=c->placed_uv[2*k];v+=c->placed_uv[2*k+1];}
        if(c->nv){u/=c->nv;v/=c->nv;}
        *gx=cos(th)*u-sin(th)*v+tx; *gy=sin(th)*u+cos(th)*v+ty;
        return;
    }
    double u = (c->flags & ASM_CHART_MIRROR) ? -x->cu[i] : x->cu[i], v = x->cv[i];
    double ct = cos(c->pose_theta), st = sin(c->pose_theta);
    double px = ct * u - st * v + c->pose_x, py = st * u + ct * v + c->pose_y;
    double c2 = cos(th), s2 = sin(th);
    *gx = c2 * px - s2 * py + tx; *gy = s2 * px + c2 * py + ty;
}

static int apl_v_model_fit(AplCtx *x, int placed)
{
    AsmRun *run = x->run;
    double A[9] = { 0 }, B[3] = { 0 };
    double n = 0.0, ws = 0.0, ss = 0.0, su = 0.0, sss = 0.0, suu = 0.0, ssu = 0.0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->area3d <= 0.0) continue;
        if (placed ? c->component != x->primary || c->placement_state == ASM_PLACE_NONE : x->orig_comp[i] != x->primary) continue;
        double u, v;
        apl_layout_centroid(x, c, i, 0.0, 0.0, 0.0, &u, &v);
        double w = c->area3d, s = x->s_axis[i];
        double f[3] = { 1.0, s, u };
        for (int r = 0; r < 3; r++) { for (int k = 0; k < 3; k++) A[r*3+k] += w * f[r] * f[k]; B[r] += w * f[r] * v; }
        n += 1.0; ws += w; ss += w * s; su += w * u; sss += w * s * s; suu += w * u * u; ssu += w * s * u;
    }
    x->vm_ok = 0; x->vm_corr = 0.0;
    if (n < 5.0) return -1;
    {
        double vs = sss / ws - (ss / ws) * (ss / ws), vu = suu / ws - (su / ws) * (su / ws), cv = ssu / ws - (ss / ws) * (su / ws);
        x->vm_corr = (vs > 1e-9 && vu > 1e-9) ? cv / sqrt(vs * vu) : 1.0;
        int drop_u = fabs(x->vm_corr) > APL_V_MODEL_MAX_CORR || vu < 250.0 * 250.0;
        if (!drop_u) {
            double M[9], R[3];
            memcpy(M, A, sizeof M); memcpy(R, B, sizeof R);
            int ok = 1;
            for (int col = 0; col < 3 && ok; col++) {
                int piv = col;
                for (int r = col + 1; r < 3; r++) if (fabs(M[r*3+col]) > fabs(M[piv*3+col])) piv = r;
                for (int k = 0; k < 3; k++) { double tt = M[col*3+k]; M[col*3+k] = M[piv*3+k]; M[piv*3+k] = tt; }
                { double tt = R[col]; R[col] = R[piv]; R[piv] = tt; }
                if (fabs(M[col*3+col]) < 1e-9) { ok = 0; break; }
                for (int r = 0; r < 3; r++) {
                    if (r == col) continue;
                    double f = M[r*3+col] / M[col*3+col];
                    for (int k = 0; k < 3; k++) M[r*3+k] -= f * M[col*3+k];
                    R[r] -= f * R[col];
                }
            }
            if (!ok || fabs(R[2] / M[8]) > APL_V_MODEL_MAX_C) drop_u = 1;
        }
        if (drop_u) {
            double det = A[0] * A[4] - A[1] * A[3];
            if (fabs(det) < 1e-9) return -1;
            x->vm_a = (B[0] * A[4] - A[1] * B[1]) / det; x->vm_b = (A[0] * B[1] - A[3] * B[0]) / det; x->vm_c = 0.0;
            x->vm_ok = 1;
            return 0;
        }
    }
    for (int col = 0; col < 3; col++) {
        int piv = col;
        for (int r = col + 1; r < 3; r++) if (fabs(A[r*3+col]) > fabs(A[piv*3+col])) piv = r;
        for (int k = 0; k < 3; k++) { double t = A[col*3+k]; A[col*3+k] = A[piv*3+k]; A[piv*3+k] = t; }
        { double t = B[col]; B[col] = B[piv]; B[piv] = t; }
        if (fabs(A[col*3+col]) < 1e-9) return -1;
        for (int r = 0; r < 3; r++) {
            if (r == col) continue;
            double f = A[r*3+col] / A[col*3+col];
            for (int k = 0; k < 3; k++) A[r*3+k] -= f * A[col*3+k];
            B[r] -= f * B[col];
        }
    }
    x->vm_a = B[0] / A[0]; x->vm_b = B[1] / A[4]; x->vm_c = B[2] / A[8];
    x->vm_ok = 1;
    return 0;
}

static int apl_v_model(AplCtx *x)
{
    return apl_v_model_fit(x,0);
}

typedef struct AplHeightGroup {
    int32_t component;
    double area, residual, squared;
} AplHeightGroup;

static int apl_height_group_cmp(const void *a, const void *b)
{
    int32_t x = ((const AplHeightGroup *)a)->component, y = ((const AplHeightGroup *)b)->component;
    return (x > y)-(x < y);
}

static int apl_v_model_update(AplCtx *x)
{
    if (!APL_V_FROM_AXIS || !x->vm_ok || !x->vm_pin) return 0;
    AplCtx fitted = *x;
    if (apl_v_model_fit(&fitted,1) || !fitted.vm_ok || !isfinite(fitted.vm_a) ||
        !isfinite(fitted.vm_b) || !isfinite(fitted.vm_c) ||
        fitted.vm_b < APL_V_MODEL_B_MIN || fitted.vm_b > APL_V_MODEL_B_MAX ||
        fabs(fitted.vm_c) > APL_V_MODEL_MAX_C) return 0;
    Arena_Mark mark = Arena_save(x->run->arena);
    AplHeightGroup *groups = ARENA_ALLOC(x->run->arena,(x->run->n_charts ? x->run->n_charts : 1)*sizeof *groups);
    size_t count = 0;
    double before = 0, after = 0;
    int ok = 1;
    for (size_t i = 0; i < x->run->n_charts; i++) {
        const AsmChart *c = &x->run->charts[i];
        if (!AsmChart_in_layout(c) || c->area3d <= 0 || c->component != x->primary || c->placement_state == ASM_PLACE_NONE) continue;
        double u,v; apl_layout_centroid(x,c,i,0,0,0,&u,&v);
        double was = x->vm_a+x->vm_b*x->s_axis[i]+x->vm_c*u-v;
        double now = fitted.vm_a+fitted.vm_b*x->s_axis[i]+fitted.vm_c*u-v;
        if (!isfinite(was) || !isfinite(now)) { ok = 0; break; }
        groups[count++] = (AplHeightGroup){x->orig_comp[i],c->area3d,c->area3d*now,c->area3d*now*now};
        before += c->area3d*was*was; after += c->area3d*now*now;
    }
    if (!count || !isfinite(before) || !isfinite(after) || !(after < before-fmax(1e-8,1e-8*before))) ok = 0;
    if (ok) qsort(groups,count,sizeof *groups,apl_height_group_cmp);
    for (size_t i = 0; i < count && ok;) {
        size_t j = i+1; double area = groups[i].area, residual = groups[i].residual, squared = groups[i].squared;
        while (j < count && groups[j].component == groups[i].component) {
            area += groups[j].area; residual += groups[j].residual; squared += groups[j].squared; j++;
        }
        double mean = residual/area, scatter = sqrt(fmax(0.0,squared/area-mean*mean));
        if (fabs(mean) > fmax(APL_V_AXIS_MAX_SHIFT,2*x->vm_p90) ||
            scatter > fmax(APL_V_AXIS_TOL,1.5*x->vm_p90)) ok = 0;
        i = j;
    }
    Arena_restore(x->run->arena,mark);
    if (!ok) return 0;
    x->vm_a = fitted.vm_a; x->vm_b = fitted.vm_b; x->vm_c = fitted.vm_c; x->vm_corr = fitted.vm_corr;
    return 1;
}

static double apl_v_model_p90(AplCtx *x)
{
    AsmRun *run = x->run;
    Arena_Mark mark = Arena_save(run->arena);
    double *r = ARENA_ALLOC(run->arena, (run->n_charts ? run->n_charts : 1) * sizeof(double));
    size_t n = 0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || x->orig_comp[i] != x->primary) continue;
        double gx, gy;
        apl_layout_centroid(x, c, i, 0.0, 0.0, 0.0, &gx, &gy);
        r[n++] = fabs(x->vm_a + x->vm_b * x->s_axis[i] + x->vm_c * gx - gy);
    }
    double p90 = 0.0;
    if (n > 0) { qsort(r, n, sizeof(double), apl_cmp_double); p90 = r[(size_t)((double)(n - 1) * 0.9)]; }
    Arena_restore(run->arena, mark);
    return p90;
}

static void apl_uout(AplCtx *x)
{
    AsmRun *run = x->run;
    double sx = 0.0, sy = 0.0;
    size_t n = 0;
    x->uout_ok = 0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || x->orig_comp[i] != x->primary || x->out_side[i] == 0) continue;
        const AplTurn *t = &x->turn[i * 2 + (x->out_side[i] > 0 ? 1 : 0)];
        if (t->n <= 0) continue;
        double l = hypot(t->tx, t->ty);
        if (l < 1e-6) continue;
        sx += t->tx / l; sy += t->ty / l; n++;
    }
    if (n >= 3 && hypot(sx, sy) > 0.5 * (double)n) { double l = hypot(sx, sy); x->uout_x = sx / l; x->uout_y = sy / l; x->uout_ok = 1; return; }
    if (!x->have_radius) return;
    double su = 0.0, sr = 0.0, suu = 0.0, sur = 0.0, m = 0.0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || x->orig_comp[i] != x->primary) continue;
        double gx, gy;
        apl_layout_centroid(x, c, i, 0.0, 0.0, 0.0, &gx, &gy);
        su += gx; sr += x->r_axis[i]; suu += gx * gx; sur += gx * x->r_axis[i]; m += 1.0;
    }
    if (m < 5.0) return;
    double det = m * suu - su * su;
    if (fabs(det) < 1e-9) return;
    double slope = (m * sur - su * sr) / det;
    x->uout_x = slope >= 0.0 ? 1.0 : -1.0; x->uout_y = 0.0; x->uout_ok = 1;
}

static void apl_fit_turn_radius(AplCtx *x)
{
    const AsmRun *run = x->run;
    x->turn_c = 0.0; x->turn_c_n = 0; x->turn_slope = 0.0;
    if (!x->have_radius || x->r_axis == NULL) return;
    Arena_Mark mark = Arena_save(run->arena);
    size_t nc = run->n_charts, n = 0;
    double *e = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    double *tl = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    double *rr = ARENA_ALLOC(run->arena, (nc ? nc : 1) * sizeof(double));
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || x->out_side[i] == 0) continue;
        const AplTurn *t = &x->turn[i * 2 + (x->out_side[i] > 0 ? 1 : 0)];
        if (t->n <= 0) continue;
        double L = hypot(t->tx, t->ty), r = x->r_axis[i];
        if (!(L > APL_TURN_MIN) || !(r > 10.0)) continue;
        tl[n] = L; rr[n] = r; e[n] = L - 2.0 * APL_PI * r; n++;
    }
    if (n < (size_t)ASM_PLACE_TURN_C_MIN_N) { Arena_restore(run->arena, mark); return; }
    double c = 0.0;
    for (int round = 0; round < 4; round++) {
        size_t m = 0;
        for (size_t i = 0; i < n; i++) {
            if (round > 0) {
                double pred = 2.0 * APL_PI * rr[i] + c;
                if (!(pred > 1e-6)) continue;
                double ratio = tl[i] / pred;
                if (ratio < 0.6 || ratio > 1.45) continue;
            }
            e[m++] = tl[i] - 2.0 * APL_PI * rr[i];
        }
        if (m < (size_t)ASM_PLACE_TURN_C_MIN_N) break;
        qsort(e, m, sizeof(double), apl_cmp_double);
        c = e[m / 2];
        x->turn_c_n = m;
    }
    double rmin = rr[0];
    for (size_t i = 1; i < n; i++) if (rr[i] < rmin) rmin = rr[i];
    if (x->turn_c_n < (size_t)ASM_PLACE_TURN_C_MIN_N ||
        c <= -0.25 * 2.0 * APL_PI * rmin || fabs(c) >= 0.5 * APL_TURN_MAX) { x->turn_c = 0.0; x->turn_c_n = 0; }
    else x->turn_c = c;
    if (n >= 2) {
        double sr = 0.0, st = 0.0, srr = 0.0, srt = 0.0, w = (double)n;
        for (size_t i = 0; i < n; i++) { sr += rr[i]; st += tl[i]; srr += rr[i]*rr[i]; srt += rr[i]*tl[i]; }
        double det = w * srr - sr * sr;
        if (fabs(det) > 1e-9) x->turn_slope = (w * srt - sr * st) / det;
    }
    Arena_restore(run->arena, mark);
}

static size_t apl_growth_for_hops(const AplCtx *x, const AsmRun *run, const AplTurn *turn,
                                  const int8_t *out_side, double *growth)
{
    size_t n = apl_measure_growth(run, turn, out_side, growth);
    if (!ASM_PLACE_GROWTH_PER_WRAP) return n;
    double d = run->layer_d_global > 0.0 ? run->layer_d_global : 9.5;
    double rule = 2.0 * APL_PI * d;
    if (ASM_PLACE_GROWTH_PER_WRAP == 2 && x->turn_slope > 0.0) {
        double g = x->turn_slope * d, lo = 0.8 * rule, hi = 1.25 * rule;
        *growth = g < lo ? lo : (g > hi ? hi : g);
    } else *growth = rule;
    return n;
}

static int apl_turn_radius(const AplCtx *x, int32_t P, int outward, int k, double *tx, double *ty)
{
    if (!APL_TURN_FROM_RADIUS || !x->have_radius || !x->uout_ok) return 0;
    const AsmRun *run = x->run;
    double d = run->chart_layer_d ? run->chart_layer_d[P] : 0.0;
    if (d <= 0.0) d = run->layer_d_global > 0.0 ? run->layer_d_global : 9.5;
    double r = x->r_axis[P] + (outward ? 1.0 : -1.0) * (double)k * d * 0.5;
    if (r < 10.0) r = 10.0;
    double L = 2.0 * APL_PI * (double)k * r + (ASM_PLACE_TURN_INTERCEPT ? (double)k * x->turn_c : 0.0);
    if (L < APL_TURN_MIN || L > APL_TURN_MAX * (double)k) return 0;
    double sgn = outward ? 1.0 : -1.0;
    *tx = sgn * x->uout_x * L; *ty = sgn * x->uout_y * L;
    return 1;
}

static int apl_stretch_turn(AplCtx *x, int32_t best, double *T, double *s_out, double *dbar_out, size_t *nhits,
                            double *rel_out, size_t *npairs_out)
{
    AsmRun *run = x->run;
    double n = 0.0, su = 0.0, sb = 0.0, suu = 0.0, sub = 0.0, sbb = 0.0, dsum = 0.0, umin = 1e300, umax = -1e300;
    size_t npairs = 0;
    double dirvote = 0.0;
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int forward = x->comp_placed[A->component] == 1 && B->component == best;
        int reverse = A->component == best && x->comp_placed[B->component] == 1;
        if (!forward && !reverse) continue;
        if (x->out_side[lp->a] != 0) {
            int cand_outward = forward ? (lp->side == x->out_side[lp->a]) : (lp->side != x->out_side[lp->a]);
            dirvote += cand_outward ? (double)lp->hit_count : -(double)lp->hit_count;
        }
        int pair_used = 0;
        for (int h = 0; h < lp->hit_count; h++) {
            const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
            if (fabs(hit->dist - lp->d_median) > 0.25 * lp->d_median + 1.0) continue;
            double ax, ay, bx, by, up, ub;
            apl_vertex_point(A, hit->va, &ax, &ay);
            apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
            if (forward) { up = ax; ub = bx; } else { up = bx; ub = ax; }
            n += 1.0; su += up; sb += ub; suu += up * up; sub += up * ub; sbb += ub * ub; dsum += lp->d_median;
            pair_used = 1;
            if (up < umin) umin = up;
            if (up > umax) umax = up;
        }
        npairs += (size_t)pair_used;
    }
    *nhits = (size_t)n;
    if (npairs_out) *npairs_out = npairs;
    if (rel_out) *rel_out = -1.0;
    if (n < 12.0 || umax - umin < 100.0) return 0;
    double var = suu - su * su / n, cov = sub - su * sb / n;
    if (var <= 1e-9) return 0;
    double sl = cov / var, dbar = dsum / n;
    *s_out = sl; *dbar_out = dbar;
    {
        double syy = sbb - sb * sb / n;
        double sse = syy - cov * cov / var;
        if (sse < 0.0) sse = 0.0;
        double dof = (double)npairs - 2.0;
        double se = dof >= 1.0 ? sqrt(sse / (dof * var)) : -1.0;
        double delta = fabs(sl - 1.0);
        if (rel_out) *rel_out = se < 0.0 ? 1e300 : (delta > 1e-12 ? se / delta : 1e300);
    }
    if (sl < 0.5 || sl > 2.0) return 0;
    int outward = dirvote != 0.0 ? dirvote > 0.0 : sl > 1.0;
    double kw = 0.0, ksum = 0.0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->nv < 12) continue;
        int on_cand = c->component == best, on_placed = x->comp_placed[c->component] == 1;
        if (outward ? !on_placed : !on_cand) continue;
        double hl;
        if (APL_CURV_CACHE && on_placed && x->curv_ok != NULL && x->curv_ok[i]) { hl = x->curv_hl[i]; if (x->st) x->st->quadric_cached++; }
        else {
            double tu[3], tv[3], huu[3];
            if (x->st) x->st->quadric_calls++;
            hl = apl_chart_quadric(c, tu, tv, huu) != 0 ? -1.0 : sqrt(huu[0]*huu[0] + huu[1]*huu[1] + huu[2]*huu[2]);
            if (APL_CURV_CACHE && on_placed && x->curv_ok != NULL) { x->curv_hl[i] = hl; x->curv_ok[i] = 1; }
        }
        if (hl < 0.0) continue;
        if (hl < 1e-4 || hl > 0.2) continue;
        ksum += c->area3d * hl; kw += c->area3d;
    }
    double t_curv = kw > 0.0 ? 2.0 * APL_PI / (ksum / kw) : 0.0;
    double turn = 0.0;
    if (outward && sl > 1.002) turn = 2.0 * APL_PI * dbar / (sl - 1.0);
    else if (!outward && sl < 0.998) turn = 2.0 * APL_PI * dbar * sl / (1.0 - sl);
    double rel = rel_out ? *rel_out : -1.0;
    (void)rel;
    int unsupported = ASM_STRETCH_MAX_REL_ERR > 0.0
                    ? (rel < 0.0 || rel > ASM_STRETCH_MAX_REL_ERR)
                    : (fabs(sl - 1.0) < 0.02);
    if (t_curv > 0.0 && (turn <= 0.0 || unsupported || turn < 0.4 * t_curv || turn > 2.5 * t_curv)) turn = t_curv;
    if (turn < APL_TURN_MIN || turn > APL_TURN_MAX) return 0;
    *T = outward ? turn : -turn;
    return 1;
}

typedef struct AplLEdge { int32_t a, b; double d, w; } AplLEdge;
#define APL_CUT_MIN_SEP 150.0
#define APL_PLACE_BUDGET_SEC 900.0
#define APL_PLACE_BUDGET_SEC_PER_CUBE 9.0
#define APL_HOP_REPAIR_MIN_SUPPORT 1
#define APL_RELAX_TRIAL_MAX_VERTICES 100000
#define APL_HOP_REPAIR_MAX_VERTICES 50000
static double apl_budget_sec(const AsmRun *run)
{
    double scaled = APL_PLACE_BUDGET_SEC_PER_CUBE * (double)(run ? run->n_cubes : 0);
    return scaled > APL_PLACE_BUDGET_SEC ? scaled : APL_PLACE_BUDGET_SEC;
}
#define APL_SWEEPS_LEGACY_CAP 0
#define APL_HOP_RETRY 0
#define APL_HOP_RETRY_MIN_RATIO 0.5
#define APL_HOP_RETRY_REQUIRE_RIM 1
#define APL_SEAM_ANCHORS 1
#define APL_HOP_ONE_WRAP 0
#define APL_GATE_TURN_FRAC 0.05
#define APL_HOP_LAYER_MODEL 1
#define APL_MAX_CUTS 512
#define APL_SHEET_GRAD 0.7
#define APL_LAYER_SEAM_W 0.0
#define APL_LAYER_SEAM_ZERO_W_CONNECTS 0

static double apl_turn_len_at(const AplTurn *turn, int32_t c, int8_t out_side, double t0, double growth, double layer_guess)
{
    if (turn != NULL && out_side != 0) {
        const AplTurn *to = &turn[(size_t)c * 2 + (out_side > 0 ? 1 : 0)], *ti = &turn[(size_t)c * 2 + (out_side > 0 ? 0 : 1)];
        if (to->n != 0) return hypot(to->tx, to->ty);
        if (ti->n != 0) return hypot(ti->tx, ti->ty) + growth;
    }
    double t = t0 + layer_guess * growth;
    if (t < APL_TURN_MIN) t = APL_TURN_MIN;
    if (t > APL_TURN_MAX) t = APL_TURN_MAX;
    return t;
}

static void apl_layer_index(AsmRun *run, const int8_t *out_side, const AplTurn *turn, double t0, double growth,
                            int32_t anchor_chart, double *lidx, const double *kover,
                            size_t *n_seam_edges, size_t *n_layer_edges, size_t *n_known, double *res_p50, double *res_p95)
{
    Arena_T arena = run->arena;
    Arena_Mark mark = Arena_save(arena);
    size_t nc = run->n_charts;
    for (size_t i = 0; i < nc; i++) lidx[i] = APL_LAYER_UNKNOWN;
    *n_seam_edges = 0; *n_layer_edges = 0; *n_known = 0; *res_p50 = 0.0; *res_p95 = 0.0;
    size_t cap = run->n_rels + run->n_layers + 1, ne = 0;
    AplLEdge *E = ARENA_ALLOC(arena, cap * sizeof(AplLEdge));
    size_t seam_first = 0;
    for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *rl = &run->rels[r];
        if (rl->flags & (ASM_REL_DROPPED | ASM_REL_SWITCHED | ASM_REL_CONTACT | ASM_REL_PLACEMENT_ONLY)) continue;
        const AsmChart *A = &run->charts[(size_t)rl->a], *B = &run->charts[(size_t)rl->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        if (rl->robust_w < 0.5) continue;
        double ax, ay, bx, by;
        apl_pose(A, 0.0, 0.0, &ax, &ay);
        apl_pose(B, 0.0, 0.0, &bx, &by);
        (*n_seam_edges)++;
        if (APL_LAYER_SEAM_W <= 0.0 && !APL_LAYER_SEAM_ZERO_W_CONNECTS) continue;
        E[ne].a = rl->a; E[ne].b = rl->b; E[ne].d = bx - ax; E[ne].w = APL_LAYER_SEAM_W; ne++;
    }
    size_t seam_count = ne;
    double dg = run->layer_d_global > 0.0 ? run->layer_d_global : 10.0;
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        if (!AsmChart_in_layout(&run->charts[(size_t)lp->a]) || !AsmChart_in_layout(&run->charts[(size_t)lp->b])) continue;
        if (out_side[lp->a] == 0 || lp->count <= 0) continue;
        (void)dg;
        int k = APL_PAIR_K_FROM_PROBE ? (lp->k >= 1 ? (lp->k > 3 ? 3 : lp->k) : 1) : 1;
        double sgn = lp->side == out_side[lp->a] ? 1.0 : -1.0;
        E[ne].a = lp->a; E[ne].b = lp->b; E[ne].d = sgn * (double)k;
        if (kover != NULL && kover[p] != 0.0) { E[ne].d = kover[p]; E[ne].w = 4.0 * (lp->count < 20 ? (double)lp->count : 20.0) / 4.0; ne++; (*n_layer_edges)++; continue; }
        E[ne].w = (lp->count < 20 ? (double)lp->count : 20.0) / 4.0; ne++;
        (*n_layer_edges)++;
    }
    UnionFind uf = UF_new(arena, (int32_t)nc);
    for (size_t e = 0; e < ne; e++) uf_union(&uf, E[e].a, E[e].b);
    {
        int32_t pcomp = run->charts[(size_t)anchor_chart].component;
        double *set_area = ARENA_CALLOC(arena, nc, sizeof(double));
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c) || c->component != pcomp) continue;
            set_area[uf_find(&uf, (int32_t)i)] += c->area3d;
        }
        int32_t best_root = -1;
        for (size_t i = 0; i < nc; i++) if (best_root < 0 || set_area[i] > set_area[best_root]) best_root = (int32_t)i;
        if (best_root >= 0 && set_area[best_root] > 0.0 && uf_find(&uf, anchor_chart) != best_root) {
            int32_t pick = -1;
            for (size_t i = 0; i < nc; i++) {
                const AsmChart *c = &run->charts[i];
                if (!AsmChart_in_layout(c) || c->component != pcomp || uf_find(&uf, (int32_t)i) != best_root) continue;
                if (pick < 0 || c->area3d > run->charts[(size_t)pick].area3d) pick = (int32_t)i;
            }
            if (pick >= 0) anchor_chart = pick;
        }
    }
    int32_t root = uf_find(&uf, anchor_chart);
    uint8_t *known = ARENA_CALLOC(arena, nc, 1);
    for (size_t i = 0; i < nc; i++) if (AsmChart_in_layout(&run->charts[i]) && uf_find(&uf, (int32_t)i) == root) { known[i] = 1; (*n_known)++; }
    double *x = ARENA_CALLOC(arena, nc, sizeof(double)), *rhs = ARENA_ALLOC(arena, nc * sizeof(double));
    double *rr = ARENA_ALLOC(arena, nc * sizeof(double)), *pp = ARENA_ALLOC(arena, nc * sizeof(double)), *Ap = ARENA_ALLOC(arena, nc * sizeof(double));
    double *w = ARENA_ALLOC(arena, (ne ? ne : 1) * sizeof(double)), *res = ARENA_ALLOC(arena, (ne ? ne : 1) * sizeof(double));
    double *seam_du = ARENA_ALLOC(arena, (seam_count ? seam_count : 1) * sizeof(double));
    for (size_t e = seam_first; e < seam_count; e++) seam_du[e] = E[e].d;
    for (size_t e = 0; e < ne; e++) w[e] = E[e].w;
    for (int round = 0; round < 8; round++) {
        for (size_t e = seam_first; e < seam_count; e++) {
            double lg = round >= 4 && known[E[e].a] ? x[E[e].a] : 0.0;
            double T = apl_turn_len_at(turn, E[e].a, out_side[E[e].a], t0, growth, lg);
            E[e].d = seam_du[e] / T;
        }
        if (round == 4) for (size_t e = 0; e < ne; e++) w[e] = E[e].w;
        memset(rhs, 0, nc * sizeof(double));
        for (size_t e = 0; e < ne; e++) {
            if (!known[E[e].a] || !known[E[e].b]) continue;
            rhs[E[e].b] += w[e] * E[e].d; rhs[E[e].a] -= w[e] * E[e].d;
        }
        rhs[anchor_chart] = 0.0;
        for (size_t i = 0; i < nc; i++) rr[i] = 0.0;
        for (size_t e = 0; e < ne; e++) {
            if (!known[E[e].a] || !known[E[e].b]) continue;
            double t = w[e] * (x[E[e].b] - x[E[e].a]);
            rr[E[e].b] += t; rr[E[e].a] -= t;
        }
        rr[anchor_chart] = x[anchor_chart];
        for (size_t i = 0; i < nc; i++) rr[i] = known[i] ? rhs[i] - rr[i] : 0.0;
        memcpy(pp, rr, nc * sizeof(double));
        double rs = 0.0;
        for (size_t i = 0; i < nc; i++) rs += rr[i] * rr[i];
        double rs0 = rs;
        size_t maxit = nc + 50 < 4000 ? nc + 50 : 4000;
        for (size_t it = 0; it < maxit && rs > 1e-14 * (rs0 > 0.0 ? rs0 : 1.0); it++) {
            for (size_t i = 0; i < nc; i++) Ap[i] = 0.0;
            for (size_t e = 0; e < ne; e++) {
                if (!known[E[e].a] || !known[E[e].b]) continue;
                double t = w[e] * (pp[E[e].b] - pp[E[e].a]);
                Ap[E[e].b] += t; Ap[E[e].a] -= t;
            }
            Ap[anchor_chart] = pp[anchor_chart];
            double pAp = 0.0;
            for (size_t i = 0; i < nc; i++) if (known[i]) pAp += pp[i] * Ap[i];
            if (!(pAp > 0.0)) break;
            double alpha = rs / pAp;
            for (size_t i = 0; i < nc; i++) if (known[i]) { x[i] += alpha * pp[i]; rr[i] -= alpha * Ap[i]; }
            double rs2 = 0.0;
            for (size_t i = 0; i < nc; i++) rs2 += rr[i] * rr[i];
            double beta = rs2 / rs; rs = rs2;
            for (size_t i = 0; i < nc; i++) pp[i] = known[i] ? rr[i] + beta * pp[i] : 0.0;
        }
        if (APL_LAYER_KINT && round >= 1) {
            for (size_t e = seam_count; e < ne; e++) {
                if (!known[E[e].a] || !known[E[e].b]) continue;
                double dl = fabs(x[E[e].b] - x[E[e].a]);
                int k = (int)floor(dl + 0.5);
                if (k < 1) k = 1;
                if (k > 3) k = 3;
                E[e].d = (E[e].d < 0.0 ? -1.0 : 1.0) * (double)k;
            }
        }
        for (size_t e = 0; e < ne; e++) {
            double r = (known[E[e].a] && known[E[e].b]) ? x[E[e].b] - x[E[e].a] - E[e].d : 0.0;
            res[e] = fabs(r);
            w[e] = E[e].w / (1.0 + (r / 0.5) * (r / 0.5));
        }
    }
    for (size_t i = 0; i < nc; i++) lidx[i] = known[i] ? x[i] - x[anchor_chart] : APL_LAYER_UNKNOWN;
    if (ne > 0) {
        qsort(res, ne, sizeof(double), apl_cmp_double);
        *res_p50 = res[ne / 2]; *res_p95 = res[(ne * 95) / 100];
    }
    Arena_restore(arena, mark);
}

static int apl_placement_seam(const AsmRelation *r, int framed, int veto)
{
    if (r->flags & (ASM_REL_DROPPED | ASM_REL_CONTACT)) return 0;
    if ((r->flags & ASM_REL_CROSSWRAP) &&
        (veto == 1 || (veto == 2 && !framed))) return 0;
    if (r->flags & ASM_REL_PLACEMENT_ONLY) return 1;
    return APL_SWITCHED_SEAM_ANCHORS && (r->flags & ASM_REL_SWITCHED) &&
           r->corr_count >= APL_SEAM_MIN_ANCHORS && r->rms <= 4.0;
}

static int apl_hop_upper(const AplCtx *x, const AsmLayerPair *lp)
{
    int upper = lp->k > 0 ? lp->k : 1;
    double pitch = x->run->layer_d_global;
    if (x->run->chart_layer_d) for (int side = 0; side < 2; side++) {
        double d = x->run->chart_layer_d[side ? lp->b : lp->a];
        if (d > 0 && (!(pitch > 0) || d < pitch)) pitch = d;
    }
    if (pitch > 0 && lp->d_median > 0) {
        double lower_pitch = pitch*fmax(0.5, 1.0-x->o->layer_tol);
        int possible = (int)fmin((double)x->o->max_layers, floor(lp->d_median/lower_pitch));
        if (possible > upper) upper = possible;
    } else upper = x->o->max_layers;
    return upper < x->o->max_layers ? upper : x->o->max_layers;
}

static size_t apl_collect_anchors(AplCtx *x, int32_t best)
{
    AsmRun *run = x->run;
    const AsmPlaceOpts *o = x->o;
    size_t na = 0;
    x->na_seam = 0;
    x->n_observations = 0;
    x->turn_rel = 0.0;
    x->fit_cu = x->fit_cv = x->fit_s = x->fit_area = 0;
    if (APL_V_FROM_AXIS && x->vm_ok && x->vm_pin && x->cu && x->cv && x->s_axis) {
        for (size_t i = 0; i < run->n_charts; i++) {
            const AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c) || c->component != best) continue;
            double u, v; apl_layout_centroid(x, c, i, 0, 0, 0, &u, &v);
            x->fit_area += c->area3d; x->fit_cu += c->area3d*u; x->fit_cv += c->area3d*v;
            x->fit_s += c->area3d*x->s_axis[i];
        }
        if (x->fit_area > 0) { x->fit_cu /= x->fit_area; x->fit_cv /= x->fit_area; x->fit_s /= x->fit_area; }
    }
    if (APL_SEAM_ANCHORS) for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *R = &run->rels[r];
        if (!apl_placement_seam(R, x->have_radius, x->o->crosswrap_anchor_veto)) continue;
        const AsmChart *A = &run->charts[(size_t)R->a], *B = &run->charts[(size_t)R->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int fwd = x->comp_placed[A->component] == 1 && B->component == best;
        int rev = A->component == best && x->comp_placed[B->component] == 1;
        if (!fwd && !rev) continue;
        if (na + (size_t)R->corr_count > x->hcap) break;
        for (int32_t k = 0; k < R->corr_count; k++) {
            const AsmCorr *cr = &run->corr[(size_t)R->corr_first + (size_t)k];
            double target[2];
            if (!AsmContinuity_target(run, R, cr, rev, target)) continue;
            double ax, ay, bx, by;
            apl_vertex_point(A, cr->va, &ax, &ay);
            apl_vertex_point(B, cr->vb, &bx, &by);
            AplAnchor *an = &x->anchors[na++];
            if (fwd) { an->sx = bx; an->sy = by; an->tx = ax; an->ty = ay; an->chart_p = R->a; an->chart_c = R->b; }
            else     { an->sx = ax; an->sy = ay; an->tx = bx; an->ty = by; an->chart_p = R->b; an->chart_c = R->a; }
            an->tx = target[0]; an->ty = target[1];
            an->source_flip = (((A->flags ^ B->flags) & ASM_CHART_MIRROR) != ((R->flags & ASM_REL_PARITY) ? ASM_CHART_MIRROR : 0));
            if (an->source_flip) an->sx = -an->sx;
            an->relation = (int32_t)r; an->source_run = cr->run;
            an->hit = NULL;
            an->measured_order = 0; an->measured_turn = 0;
            an->w = 1.0; an->k = 0; an->estimated = 0; an->reverse = rev; an->side = 0; an->Tx = 0.0; an->Ty = 0.0; an->cluster = -1;
            an->turn_src = APL_TSRC_SEAM;
            an->seam_rms = R->rms; x->na_seam++;
        }
    }
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int forward = x->comp_placed[A->component] == 1 && B->component == best;
        int reverse = A->component == best && x->comp_placed[B->component] == 1;
        if (!forward && !reverse) continue;
        int32_t P = forward ? lp->a : lp->b;
        int side;
        if (forward) side = lp->side;
        else {
            if (x->out_side[lp->a] == 0 || x->out_side[lp->b] == 0) continue;
            int b_outward_of_a = lp->side == x->out_side[lp->a];
            side = b_outward_of_a ? -x->out_side[lp->b] : x->out_side[lp->b];
        }
        size_t observation = x->n_observations;
        x->n_observations += (size_t)lp->hit_count;
        int first_hop = lp->k > 0 ? lp->k : 1, last_hop = apl_hop_upper(x, lp);
        if (x->out_side[P] != 0) {
          for (int kk = first_hop; kk <= last_hop; kk++) {
            int32_t dk = side == x->out_side[P] ? kk : -kk;
            if (kk > o->max_layers || na + (size_t)lp->hit_count > x->hcap) continue;
            int hop_side = side;
            double tx = 0.0, ty = 0.0;
            int tsrc = apl_turn_k(x->turn, P, hop_side, x->out_side[P], x->growth, kk, &tx, &ty);
            int measured_turn = tsrc == 1;
            int tsource = tsrc == 1 ? (kk > 1 ? APL_TSRC_MEASURED_K : APL_TSRC_MEASURED)
                        : tsrc == 2 ? APL_TSRC_OPPOSITE : APL_TSRC_T0;
            double measured_tx = tx, measured_ty = ty;
            if (!tsrc && apl_turn_radius(x, P, dk > 0, kk, &tx, &ty)) { tsrc = 2; tsource = APL_TSRC_RADIUS; x->anchors_radius += (size_t)lp->hit_count; }
            if (!tsrc) {
                int use_layer = APL_HOP_LAYER_MODEL && x->t0_measured && x->lidx != NULL && x->lidx[P] > APL_LAYER_UNKNOWN;
                double t1 = use_layer ? x->t0 + x->lidx[P] * x->growth
                          : (x->stretch_ok ? fabs(x->stretch_T) : x->t0);
                tsource = use_layer ? APL_TSRC_LAYER : (x->stretch_ok ? APL_TSRC_STRETCH : APL_TSRC_T0);
                if (!x->t0_measured && !x->stretch_ok)
                    t1 += 0.5*x->growth*(dk > 0 ? 1.0 : -1.0);
                if (APL_TURN_FIELD && x->tfield != NULL && x->tfield[P] > 0.0) t1 = x->tfield[P];
                {
                    double arel = use_layer ? 0.0
                                : (x->stretch_ok ? (x->stretch_rel >= 0.0 ? x->stretch_rel : 1e300)
                                : (x->t0_measured ? 0.0 : 1e300));
                    if (arel > x->turn_rel) x->turn_rel = arel;
                }
                if (t1 < APL_TURN_MIN) t1 = APL_TURN_MIN;
                if (t1 > APL_TURN_MAX) t1 = APL_TURN_MAX;
                double sum = 0.0;
                for (int j = 0; j < kk; j++) sum += t1 + (double)j * x->growth * (dk > 0 ? 1.0 : -1.0);
                tx = dk > 0 ? sum : -sum; ty = 0.0; tsrc = 2;
            }
            for (int h = 0; h < lp->hit_count; h++) {
                const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
                double ax, ay, bx, by;
                apl_vertex_point(A, hit->va, &ax, &ay);
                apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
                AplAnchor *an = &x->anchors[na++];
                if (forward) { an->sx = bx; an->sy = by; an->tx = ax + tx; an->ty = ay + ty; }
                else         { an->sx = ax; an->sy = ay; an->tx = bx + tx; an->ty = by + ty; }
                an->w = (tsrc == 1 ? 1.0 : 0.5) / (double)kk;
                an->k = kk; an->estimated = tsrc != 1; an->reverse = reverse; an->turn_src = tsource;
                an->chart_p = P; an->side = (int8_t)side; an->Tx = tx; an->Ty = ty; an->cluster = -1;
                an->chart_c = forward ? lp->b : lp->a; an->seam_rms = 0.0;
                an->relation = -1; an->source_run = -1; an->source_flip = 0;
                an->observation = observation+(size_t)h;
                an->hit = hit;
                an->measured_order = hit->order; an->measured_turn = measured_turn;
                an->measured_x = (forward ? ax : bx) + measured_tx;
                an->measured_y = (forward ? ay : by) + measured_ty;
            }
          }
            continue;
        }
        for (int k = first_hop; k <= last_hop && na + (size_t)lp->hit_count <= x->hcap; k++) {
            double tx, ty;
            int tsrc = apl_turn_k(x->turn, P, side, x->out_side[P], x->growth, k, &tx, &ty);
            int measured_turn = tsrc == 1;
            double measured_tx = tsrc ? tx : 0.0, measured_ty = tsrc ? ty : 0.0;
            int tsource = tsrc == 1 ? (k > 1 ? APL_TSRC_MEASURED_K : APL_TSRC_MEASURED)
                        : tsrc == 2 ? APL_TSRC_OPPOSITE : APL_TSRC_T0;
            if (!tsrc && k == 1 && x->stretch_ok) {
                tx = x->stretch_T; ty = 0.0; tsrc = 2; tsource = APL_TSRC_STRETCH;
                double arel = x->stretch_rel >= 0.0 ? x->stretch_rel : 1e300;
                if (arel > x->turn_rel) x->turn_rel = arel;
            }
            if (!tsrc && x->out_side[P] != 0 && apl_turn_radius(x, P, side == x->out_side[P], k, &tx, &ty)) { tsrc = 2; tsource = APL_TSRC_RADIUS; x->anchors_radius += (size_t)lp->hit_count; }
            if (!tsrc) break;
            for (int h = 0; h < lp->hit_count; h++) {
                const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
                double ax, ay, bx, by;
                apl_vertex_point(A, hit->va, &ax, &ay);
                apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
                AplAnchor *an = &x->anchors[na++];
                if (forward) { an->sx = bx; an->sy = by; an->tx = ax + tx; an->ty = ay + ty; }
                else         { an->sx = ax; an->sy = ay; an->tx = bx + tx; an->ty = by + ty; }
                an->w = (tsrc == 1 ? 1.0 : 0.5) / (double)k;
                an->k = k; an->estimated = tsrc == 2; an->reverse = reverse; an->turn_src = tsource;
                an->chart_p = P; an->side = (int8_t)side; an->Tx = tx; an->Ty = ty; an->cluster = -1;
                an->chart_c = forward ? lp->b : lp->a; an->seam_rms = 0.0;
                an->relation = -1; an->source_run = -1; an->source_flip = 0;
                an->observation = observation+(size_t)h;
                an->hit = hit;
                an->measured_order = hit->order; an->measured_turn = measured_turn;
                an->measured_x = (forward ? ax : bx) + measured_tx;
                an->measured_y = (forward ? ay : by) + measured_ty;
            }
        }
    }
    for (size_t i = 0; i < na; i++) {
        AplAnchor *a = &x->anchors[i];
        const AsmChart *c = &run->charts[a->chart_c];
        double sx = a->source_flip ? -a->sx : a->sx;
        double dx = sx-c->pose_x, dy = a->sy-c->pose_y, ct = cos(c->pose_theta), sn = sin(c->pose_theta);
        a->local_u = ct*dx+sn*dy; a->local_v = -sn*dx+ct*dy;
        if (c->flags & ASM_CHART_MIRROR) a->local_u = -a->local_u;
    }
    return na;
}

typedef struct AplDuIdx { double du; size_t idx; } AplDuIdx;
static int apl_cmp_duidx(const void *p, const void *q)
{
    const AplDuIdx *a = p, *b = q;
    if (a->du != b->du) return a->du < b->du ? -1 : 1;
    return (a->idx > b->idx) - (a->idx < b->idx);
}

static void apl_window_add(double *value, double *lazy, size_t node, size_t lo, size_t hi,
                           size_t first, size_t last, double weight)
{
    if (first >= hi || last <= lo) return;
    if (first <= lo && hi <= last) { value[node] += weight; lazy[node] += weight; return; }
    size_t mid = lo+(hi-lo)/2;
    apl_window_add(value, lazy, node*2, lo, mid, first, last, weight);
    apl_window_add(value, lazy, node*2+1, mid, hi, first, last, weight);
    value[node] = lazy[node]+fmax(value[node*2], value[node*2+1]);
}

static size_t apl_window_edge(const double *value, size_t n)
{
    size_t node = 1, lo = 0, hi = n;
    while (hi-lo > 1) {
        size_t mid = lo+(hi-lo)/2;
        if (value[node*2] >= value[node*2+1]) { node *= 2; hi = mid; }
        else { node = node*2+1; lo = mid; }
    }
    return lo;
}

static size_t apl_window_bound(const double *v, size_t n, double key, int upper)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo+(hi-lo)/2;
        if (v[mid] < key || (upper && v[mid] == key)) lo = mid+1; else hi = mid;
    }
    return lo;
}

static int apl_cluster_eligible(const AplAnchor *a, int seam_only)
{
    return a->cluster < 0 && a->w > 0 &&
           (seam_only ? a->k == 0 && a->source_flip == seam_only-1 : a->k > 0);
}

static int apl_cluster_pass(AplCtx *x, size_t na, const size_t *order, const double *du, double wu, double wv, int rounds, int seam_only, int kind,
                            int ncl, double *cl_w, double *cl_u, double *cl_v, int *cl_kind)
{
    Arena_Mark mark = Arena_save(x->run->arena);
    double *v = ARENA_ALLOC(x->run->arena, (na ? na : 1)*sizeof(double));
    size_t nv = 0;
    for (size_t i = 0; i < na; i++) if (apl_cluster_eligible(&x->anchors[i], seam_only))
        v[nv++] = x->anchors[i].ty-x->anchors[i].sy;
    if (!nv) { Arena_restore(x->run->arena, mark); return ncl; }
    qsort(v, nv, sizeof(double), apl_cmp_double);
    size_t unique = 0;
    for (size_t i = 0; i < nv; i++) if (!unique || v[i] != v[unique-1]) v[unique++] = v[i];
    nv = unique;
    size_t *first = ARENA_ALLOC(x->run->arena, na*sizeof(size_t));
    size_t *last = ARENA_ALLOC(x->run->arena, na*sizeof(size_t));
    for (size_t i = 0; i < na; i++) if (apl_cluster_eligible(&x->anchors[i], seam_only)) {
        double y = x->anchors[i].ty-x->anchors[i].sy;
        first[i] = apl_window_bound(v, nv, y-2*wv, 0);
        last[i] = apl_window_bound(v, nv, y, 1);
    }
    double *value = ARENA_ALLOC(x->run->arena, 4*nv*sizeof(double));
    double *lazy = ARENA_ALLOC(x->run->arena, 4*nv*sizeof(double));
    for (int round = 0; round < rounds; round++) {
        memset(value, 0, 4*nv*sizeof(double)); memset(lazy, 0, 4*nv*sizeof(double));
        double best_w = 0.0, bottom = 0.0; size_t bi = 0, bj = 0, j = 0;
        for (size_t i = 0; i < na; i++) {
            while (j < na && du[order[j]]-du[order[i]] <= wu) {
                size_t p = order[j++]; const AplAnchor *a = &x->anchors[p];
                if (apl_cluster_eligible(a, seam_only)) apl_window_add(value, lazy, 1, 0, nv, first[p], last[p], a->w);
            }
            if (value[1] > best_w+1e-9) { best_w = value[1]; bi = i; bj = j; bottom = v[apl_window_edge(value, nv)]; }
            size_t p = order[i]; const AplAnchor *a = &x->anchors[p];
            if (apl_cluster_eligible(a, seam_only)) apl_window_add(value, lazy, 1, 0, nv, first[p], last[p], -a->w);
        }
        if (best_w <= 1e-9) break;
        double w2 = 0.0, su = 0.0, sv = 0.0;
        for (size_t q = bi; q < bj; q++) {
            AplAnchor *a = &x->anchors[order[q]];
            if (!apl_cluster_eligible(a, seam_only)) continue;
            double y = a->ty-a->sy;
            if (y >= bottom && y-bottom <= 2*wv) {
                a->cluster = ncl; w2 += a->w; su += a->w*(a->tx-a->sx); sv += a->w*y;
            }
        }
        if (w2 <= 0.0) break;
        cl_w[ncl] = w2; cl_u[ncl] = su / w2; cl_v[ncl] = sv / w2;
        if (cl_kind) cl_kind[ncl] = kind;
        ncl++;
    }
    Arena_restore(x->run->arena, mark);
    return ncl;
}

static double apl_fit_cluster(AplCtx *x, size_t na, int cl, size_t min_members, double max_rot, double *th, double *tx, double *ty,
                              size_t *kept_out, double *res);

static int apl_cluster_anchors(AplCtx *x, size_t na, double *cl_w, double *cl_u, double *cl_v, int *cl_kind)
{
    Arena_Mark mark = Arena_save(x->run->arena);
    double *du = ARENA_ALLOC(x->run->arena, (na ? na : 1) * sizeof(double));
    size_t *order = ARENA_ALLOC(x->run->arena, (na ? na : 1) * sizeof(size_t));
    double tl = 0.0; size_t ntl = 0;
    for (size_t i = 0; i < na; i++) { du[i] = x->anchors[i].tx - x->anchors[i].sx; if (x->anchors[i].k > 0) { tl += hypot(x->anchors[i].Tx, x->anchors[i].Ty) / (double)x->anchors[i].k; ntl++; } }
    tl = ntl ? tl / (double)ntl : 1000.0;
    double wu = fmax(0.15 * tl, 80.0), wv = 100.0;
    for (size_t i = 0; i < na; i++) order[i] = i;
    if (APL_CLUSTER_QSORT) {
        AplDuIdx *key = ARENA_ALLOC(x->run->arena, (na ? na : 1) * sizeof(AplDuIdx));
        for (size_t i = 0; i < na; i++) { key[i].du = du[i]; key[i].idx = i; }
        if (na > 1) qsort(key, na, sizeof(AplDuIdx), apl_cmp_duidx);
        for (size_t i = 0; i < na; i++) order[i] = key[i].idx;
    } else {
        for (size_t i = 1; i < na; i++) {
            size_t k = order[i]; size_t j = i;
            while (j > 0 && du[order[j-1]] > du[k]) { order[j] = order[j-1]; j--; }
            order[j] = k;
        }
    }
    int ncl = 0;
    if (APL_SEAM_FIRST && x->na_seam > 0) for (int flip = 0; flip < 2; flip++)
        ncl = apl_cluster_pass(x, na, order, du, APL_SEAM_CLUSTER_VOX, APL_SEAM_CLUSTER_VOX, APL_MAX_SEAM_CLUSTERS, flip+1, 1, ncl, cl_w, cl_u, cl_v, cl_kind);
    int first_hop = ncl;
    ncl = apl_cluster_pass(x, na, order, du, wu, wv, APL_MAX_CLUSTERS, 0, 0, ncl, cl_w, cl_u, cl_v, cl_kind);
    if (first_hop < ncl && x->o) {
        AplCtx fit = *x;
        fit.wts = ARENA_ALLOC(x->run->arena,(na ? na : 1)*sizeof(double));
        double *residual = ARENA_ALLOC(x->run->arena,(na ? na : 1)*sizeof(double));
        int *saved = ARENA_ALLOC(x->run->arena,(na ? na : 1)*sizeof(int));
        for (int seed = first_hop; seed < ncl; seed++) {
            double th,tx,ty; size_t kept;
            double rms = apl_fit_cluster(&fit,na,seed,(size_t)x->o->min_anchors,x->o->max_rot_refine,&th,&tx,&ty,&kept,residual);
            if (!(rms <= 16) || !isfinite(th) || fabs(th) < 1e-8) continue;
            double cu = 0, cv = 0, sw = 0;
            for (size_t i = 0; i < na; i++) if (fit.wts[i] > 0) {
                cu += fit.wts[i]*x->anchors[i].sx; cv += fit.wts[i]*x->anchors[i].sy; sw += fit.wts[i];
            }
            if (!(sw > 0)) continue;
            cu /= sw; cv /= sw;
            double ct = cos(th), sn = sin(th);
            double lo[APL_MAX_CLUSTERS_TOTAL][2], hi[APL_MAX_CLUSTERS_TOTAL][2];
            for (int cl = first_hop; cl < ncl; cl++) for (int d = 0; d < 2; d++) { lo[cl][d] = 1e300; hi[cl][d] = -1e300; }
            for (size_t i = 0; i < na; i++) {
                const AplAnchor *a = &x->anchors[i]; int cl = a->cluster;
                if (cl < first_hop || cl >= ncl) continue;
                double u = a->sx-cu, v = a->sy-cv;
                double offset[2] = {a->tx-a->sx+(1-ct)*u+sn*v, a->ty-a->sy-sn*u+(1-ct)*v};
                for (int d = 0; d < 2; d++) { lo[cl][d] = fmin(lo[cl][d],offset[d]); hi[cl][d] = fmax(hi[cl][d],offset[d]); }
            }
            for (int cl = seed+1; cl < ncl; cl++) {
                if (hi[cl][0] < lo[cl][0] ||
                    fmax(hi[seed][0],hi[cl][0])-fmin(lo[seed][0],lo[cl][0]) > wu ||
                    fmax(hi[seed][1],hi[cl][1])-fmin(lo[seed][1],lo[cl][1]) > 2*wv) continue;
                double oth,otx,oty;
                rms = apl_fit_cluster(&fit,na,cl,(size_t)x->o->min_anchors,x->o->max_rot_refine,&oth,&otx,&oty,&kept,residual);
                if (!(rms <= 16)) continue;
                for (size_t i = 0; i < na; i++) {
                    saved[i] = x->anchors[i].cluster;
                    if (saved[i] == cl) x->anchors[i].cluster = seed;
                }
                rms = apl_fit_cluster(&fit,na,seed,(size_t)x->o->min_anchors,x->o->max_rot_refine,&oth,&otx,&oty,&kept,residual);
                if (!(rms <= 16)) {
                    for (size_t i = 0; i < na; i++) x->anchors[i].cluster = saved[i];
                    continue;
                }
                for (int d = 0; d < 2; d++) { lo[seed][d] = fmin(lo[seed][d],lo[cl][d]); hi[seed][d] = fmax(hi[seed][d],hi[cl][d]); }
            }
        }
        int next = first_hop;
        for (int cl = first_hop; cl < ncl; cl++) {
            double sw = 0, su = 0, sv = 0;
            for (size_t i = 0; i < na; i++) if (x->anchors[i].cluster == cl) {
                AplAnchor *a = &x->anchors[i]; a->cluster = next;
                sw += a->w; su += a->w*(a->tx-a->sx); sv += a->w*(a->ty-a->sy);
            }
            if (sw > 0) { cl_w[next] = sw; cl_u[next] = su/sw; cl_v[next] = sv/sw; cl_kind[next] = 0; next++; }
        }
        ncl = next;
    }
    Arena_restore(x->run->arena, mark);
    return ncl;
}

static double apl_fit_cluster_transform(AplCtx *x, size_t na, double rot, double *th, double *tx, double *ty)
{
    int axis = APL_V_FROM_AXIS && x->vm_ok && x->vm_pin && x->fit_area > 0;
    for (size_t i = 0; i < na && axis; i++) if (x->wts[i] > 0 && x->anchors[i].k == 0) axis = 0;
    if (axis) return apl_fit_axis_translation(x->anchors, x->wts, na, rot,
                       x->fit_cu, x->fit_cv, x->vm_a+x->vm_b*x->fit_s, x->vm_c, apl_axis_band(x), th, tx, ty);
    return apl_fit_translation(x->anchors, x->wts, na, rot, th, tx, ty);
}

static double apl_fit_cluster(AplCtx *x, size_t na, int cl, size_t min_members, double max_rot, double *th, double *tx, double *ty,
                              size_t *kept_out, double *res)
{
    size_t members = 0;
    double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
    for (size_t i = 0; i < na; i++) {
        if (x->anchors[i].cluster == cl) {
            const AplAnchor *a = &x->anchors[i]; x->wts[i] = a->w; members++;
            umin = fmin(umin, a->sx); umax = fmax(umax, a->sx);
            vmin = fmin(vmin, a->sy); vmax = fmax(vmax, a->sy);
        }
        else x->wts[i] = 0.0;
        res[i] = 0.0;
    }
    if (members < min_members) { *kept_out = members; return 1e300; }
    double extent = fmax(umax-umin, vmax-vmin);
    double rot = extent >= 300.0 ? max_rot : extent >= 200.0 ? fmin(max_rot, 0.25) : fmin(max_rot, 0.05);
    if (x->cand_grad < APL_SHEET_GRAD && extent >= 200.0) rot = APL_PI;
    double rms = apl_fit_cluster_transform(x, na, rot, th, tx, ty);
    double c = cos(*th), s = sin(*th);
    Arena_Mark mark = Arena_save(x->run->arena);
    double *sorted = ARENA_ALLOC(x->run->arena, members * sizeof(double));
    size_t q = 0;
    for (size_t i = 0; i < na; i++) {
        if (x->wts[i] <= 0.0) continue;
        double px = c*x->anchors[i].sx - s*x->anchors[i].sy + *tx, py = s*x->anchors[i].sx + c*x->anchors[i].sy + *ty;
        res[i] = hypot(px - x->anchors[i].tx, py - x->anchors[i].ty);
        sorted[q++] = res[i];
    }
    qsort(sorted, q, sizeof(double), apl_cmp_double);
    double cut = fmax(3.0 * sorted[q/2], 2.0);
    size_t kept = 0;
    for (size_t i = 0; i < na; i++) if (x->wts[i] > 0.0) { if (res[i] <= cut) kept++; else x->wts[i] = 0.0; }
    if (kept >= min_members) {
        rms = apl_fit_cluster_transform(x, na, rot, th, tx, ty);
        c = cos(*th); s = sin(*th);
        for (size_t i = 0; i < na; i++) {
            if (x->anchors[i].cluster != cl) continue;
            double px = c*x->anchors[i].sx - s*x->anchors[i].sy + *tx, py = s*x->anchors[i].sx + c*x->anchors[i].sy + *ty;
            res[i] = hypot(px - x->anchors[i].tx, py - x->anchors[i].ty);
        }
    } else {
        kept = members;
        for (size_t i = 0; i < na; i++) x->wts[i] = x->anchors[i].cluster == cl ? x->anchors[i].w : 0.0;
    }
    Arena_restore(x->run->arena, mark);
    *kept_out = kept;
    return rms;
}

typedef struct AplRimSample { int64_t key; double u, v; float p[3]; int32_t chart; } AplRimSample;

static int apl_cmp_rim(const void *x, const void *y)
{
    const AplRimSample *a = x, *b = y;
    return (a->key > b->key) - (a->key < b->key);
}

static int64_t apl_rim_key(double u, double v, double cell)
{
    int64_t cu = (int64_t)floor(u / cell) + (1 << 30), cv = (int64_t)floor(v / cell) + (1 << 30);
    return (cu << 32) | (cv & 0xffffffffLL);
}

static double apl_rim_audit(const AsmRun *run, const int32_t *orig_comp, int32_t best, int32_t primary, double reach, size_t *n_rim)
{
    Arena_Mark mark = Arena_save(run->arena);
    size_t nc = run->n_charts, ns = 0, cap = 0;
    const size_t stride_placed = 4, stride_new = 2;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != primary || orig_comp[i] == best) continue;
        cap += c->nv / stride_placed + 1;
    }
    *n_rim = 0;
    if (cap == 0) { Arena_restore(run->arena, mark); return 0.0; }
    AplRimSample *S = ARENA_ALLOC(run->arena, cap * sizeof(AplRimSample));
    double cell = reach;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != primary || orig_comp[i] == best) continue;
        for (size_t k = 0; k < c->nv; k += stride_placed) {
            AplRimSample *s = &S[ns++];
            s->chart = c->id;
            apl_vertex_point(c, (int32_t)k, &s->u, &s->v);
            s->p[0] = c->xyz[k*3]; s->p[1] = c->xyz[k*3+1]; s->p[2] = c->xyz[k*3+2];
            s->key = apl_rim_key(s->u, s->v, cell);
        }
    }
    qsort(S, ns, sizeof(AplRimSample), apl_cmp_rim);
    size_t rim = 0, bad = 0;
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || orig_comp[i] != best) continue;
        double dl = run->chart_layer_d ? run->chart_layer_d[i] : 0.0;
        if (dl <= 0.0) dl = run->layer_d_global;
        double tol = fmax(0.5 * dl, 7.0);
        for (size_t k = 0; k < c->nv; k += stride_new) {
            double u, v;
            apl_vertex_point(c, (int32_t)k, &u, &v);
            const float *p = &c->xyz[k*3];
            int touched = 0, violated = 0;
            for (int du = -1; du <= 1 && !violated; du++) for (int dv = -1; dv <= 1 && !violated; dv++) {
                int64_t key = apl_rim_key(u + du * cell, v + dv * cell, cell);
                size_t lo = 0, hi = ns;
                while (lo < hi) { size_t mid = (lo + hi) / 2; if (S[mid].key < key) lo = mid + 1; else hi = mid; }
                for (size_t q = lo; q < ns && S[q].key == key; q++) {
                    double duv = hypot(S[q].u - u, S[q].v - v);
                    if (duv > reach) continue;
                    touched = 1;
                    double d3 = sqrt((S[q].p[0]-p[0])*(S[q].p[0]-p[0]) + (S[q].p[1]-p[1])*(S[q].p[1]-p[1]) + (S[q].p[2]-p[2])*(S[q].p[2]-p[2]));
                    if (d3 - duv > tol) { violated = 1; break; }
                }
            }
            if (touched) { rim++; bad += (size_t)violated; }
        }
    }
    Arena_restore(run->arena, mark);
    *n_rim = rim;
    return rim ? (double)bad / (double)rim : 0.0;
}

struct AplRimGrid {
    Arena_T arena; double cell;
    int64_t *keys; int32_t *head; size_t cap, count;
    AplRimSample *s; int32_t *next; size_t n, ncap;
};

static size_t aprg_hash(int64_t key, size_t cap) { return (size_t)(((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 20) & (cap - 1); }

static void aprg_rehash(AplRimGrid *g, size_t ncap)
{
    int64_t *nk = ARENA_ALLOC(g->arena, ncap * sizeof(int64_t));
    int32_t *nh = ARENA_ALLOC(g->arena, ncap * sizeof(int32_t));
    for (size_t i = 0; i < ncap; i++) nh[i] = -1;
    for (size_t i = 0; i < g->cap; i++) {
        if (g->head[i] < 0) continue;
        size_t h = aprg_hash(g->keys[i], ncap);
        while (nh[h] >= 0) h = (h + 1) & (ncap - 1);
        nk[h] = g->keys[i]; nh[h] = g->head[i];
    }
    g->keys = nk; g->head = nh; g->cap = ncap;
}

static int32_t *aprg_slot(AplRimGrid *g, int64_t key, int create)
{
    if (create && (g->count + 1) * 2 > g->cap) aprg_rehash(g, g->cap * 2);
    size_t h = aprg_hash(key, g->cap), steps = 0;
    for (;;) {
        if (g->head[h] < 0) { if (!create) return NULL; g->keys[h] = key; g->count++; return &g->head[h]; }
        if (g->keys[h] == key) return &g->head[h];
        h = (h + 1) & (g->cap - 1);
        if (++steps > g->cap) { fprintf(stderr, "[rim grid] slot search exhausted: cap %zu count %zu key %lld\n", g->cap, g->count, (long long)key); return NULL; }
    }
}

static AplRimGrid *aprg_new_sized(Arena_T arena, double cell, size_t reserve)
{
    AplRimGrid *g = ARENA_CALLOC(arena, 1, sizeof *g);
    g->arena = arena; g->cell = cell; g->cap = 16;
    while (g->cap < (1u << 16) && g->cap/2 < reserve) g->cap *= 2;
    g->keys = ARENA_ALLOC(arena, g->cap * sizeof(int64_t));
    g->head = ARENA_ALLOC(arena, g->cap * sizeof(int32_t));
    for (size_t i = 0; i < g->cap; i++) g->head[i] = -1;
    g->ncap = 16;
    while (g->ncap < (1u << 16) && g->ncap < reserve) g->ncap *= 2;
    g->s = ARENA_ALLOC(arena, g->ncap * sizeof(AplRimSample));
    g->next = ARENA_ALLOC(arena, g->ncap * sizeof(int32_t));
    return g;
}

static AplRimGrid *aprg_new(Arena_T arena, double cell)
{
    return aprg_new_sized(arena,cell,1u << 16);
}

static void aprg_insert_chart_stride(AplRimGrid *g, const AsmChart *c, size_t stride)
{
    size_t need = c->nv / stride + 1;
    if (g->n + need > g->ncap) {
        size_t ncap = g->ncap; while (ncap < g->n + need) ncap *= 2;
        AplRimSample *ns = ARENA_ALLOC(g->arena, ncap * sizeof(AplRimSample));
        int32_t *nn = ARENA_ALLOC(g->arena, ncap * sizeof(int32_t));
        memcpy(ns, g->s, g->n * sizeof(AplRimSample)); memcpy(nn, g->next, g->n * sizeof(int32_t));
        g->s = ns; g->next = nn; g->ncap = ncap;
    }
    for (size_t k = 0; k < c->nv; k += stride) {
        AplRimSample *s = &g->s[g->n];
        s->chart = c->id;
        apl_vertex_point(c, (int32_t)k, &s->u, &s->v);
        s->p[0] = c->xyz[k*3]; s->p[1] = c->xyz[k*3+1]; s->p[2] = c->xyz[k*3+2];
        s->key = apl_rim_key(s->u, s->v, g->cell);
        int32_t *slot = aprg_slot(g, s->key, 1);
        if (slot == NULL) break;
        g->next[g->n] = *slot; *slot = (int32_t)g->n; g->n++;
    }
}

static int aprg_pair_test(double dx, double dy, const float *p, const float *q,
                          double reach, double reach2_safe, double reach2_inner, double tol)
{
    double uv2 = dx*dx+dy*dy;
    if (APL_RIM_SQUARED_CULL && uv2 > reach2_safe) return -1;
    double duv = -1;
    if (!(uv2 < reach2_inner)) {
        duv = hypot(dx,dy);
        if (duv > reach) return -1;
    }
    double d3_squared = (q[0]-p[0])*(q[0]-p[0]) + (q[1]-p[1])*(q[1]-p[1]) + (q[2]-p[2])*(q[2]-p[2]);
    if (d3_squared <= (uv2+tol*tol)*(1.0-64.0*DBL_EPSILON)) return 0;
    if (duv < 0) duv = hypot(dx,dy);
    return sqrt(d3_squared)-duv > tol;
}

static void aprg_insert_chart(AplRimGrid *g, const AsmChart *c)
{
    aprg_insert_chart_stride(g,c,4);
}

typedef struct AprgCheckpoint {
    const AsmChart **by_id;
    size_t *first_sample, count, reference_stride;
} AprgCheckpoint;

static double aprg_probe_record(AplRimGrid *g, const AsmRun *run, const AsmChart *const *charts,
                                size_t n, double reach, size_t *n_rim, const uint8_t *excluded,
                                uint8_t *violations, int32_t *partners, int32_t *partner_samples,
                                const AprgCheckpoint *checkpoint, size_t query_stride)
{
    typedef struct AprgRange { const AsmChart *chart; size_t begin, query_base; } AprgRange;
    const size_t block = 512;
    Arena_Mark mark = Arena_save(run->arena);
    size_t nranges = 0;
    for (size_t k = 0; k < n; k++) nranges += (charts[k]->nv+block-1)/block;
    AprgRange *ranges = ARENA_ALLOC(run->arena,(nranges ? nranges : 1)*sizeof *ranges);
    size_t at = 0, query_base = 0;
    for (size_t k = 0; k < n; k++) {
        for (size_t v = 0; v < charts[k]->nv; v += block)
            ranges[at++] = (AprgRange){charts[k],v,query_base};
        query_base += (charts[k]->nv+query_stride-1)/query_stride;
    }
    size_t rim = 0, bad = 0;
    double reach2_safe = reach * reach * (1.0 + 8.0 * DBL_EPSILON);
    double reach2_inner = reach * reach * (1.0 - 64.0 * DBL_EPSILON);
    ptrdiff_t range;
#ifdef _OPENMP
    int threads = omp_get_max_threads();
    if ((size_t)threads > nranges) threads = nranges ? (int)nranges : 1;
#pragma omp parallel for schedule(dynamic,1) reduction(+:rim,bad) num_threads(threads) if(nranges >= 8)
#endif
    for (range = 0; range < (ptrdiff_t)nranges; range++) {
        const AsmChart *c = ranges[range].chart;
        const AsmChart *old = checkpoint && c->id >= 0 && (size_t)c->id < checkpoint->count ? checkpoint->by_id[c->id] : NULL;
        double dl = run->chart_layer_d ? run->chart_layer_d[c->id] : 0.0;
        if (dl <= 0.0) dl = run->layer_d_global;
        double tol = fmax(0.5 * dl, 7.0);
        size_t end = ranges[range].begin+block; if (end > c->nv) end = c->nv;
        for (size_t v = ranges[range].begin; v < end; v += query_stride) {
            double u, w;
            apl_vertex_point(c, (int32_t)v, &u, &w);
            const float *p = &c->xyz[v*3];
            double old_u = 0, old_w = 0;
            int old_valid = old && v < old->nv && old->uv && old->xyz;
            if (old_valid) apl_vertex_point(old,(int32_t)v,&old_u,&old_w);
            int touched = 0, violated = 0, actual_bad = 0; int32_t partner = -1, partner_sample = -1;
            for (int du = -1; du <= 1 && !violated; du++) for (int dv = -1; dv <= 1 && !violated; dv++) {
                int32_t *slot = aprg_slot(g, apl_rim_key(u + du * g->cell, w + dv * g->cell, g->cell), 0);
                if (slot == NULL) continue;
                size_t walk = 0;
                for (int32_t q = *slot; q >= 0; q = g->next[q]) {
                    const AplRimSample *s = &g->s[q];
                    if (++walk > g->n + 1) { fprintf(stderr, "[rim grid] list cycle at node %d (n %zu)\n", q, g->n); break; }
                    if (excluded && excluded[s->chart]) continue;
                    double dx = s->u - u, dy = s->v - w;
                    int result = aprg_pair_test(dx,dy,p,s->p,reach,reach2_safe,reach2_inner,tol);
                    if (result < 0) continue;
                    touched = 1;
                    if (result) {
                        actual_bad = 1;
                        if (checkpoint && old_valid && s->chart >= 0 && (size_t)s->chart < checkpoint->count) {
                            const AsmChart *ref = checkpoint->by_id[s->chart];
                            size_t first = checkpoint->first_sample[s->chart];
                            if (ref && ref->nv && ref->uv && ref->xyz && (size_t)q >= first &&
                                (size_t)q-first <= (ref->nv-1)/checkpoint->reference_stride) {
                                size_t vertex = checkpoint->reference_stride*((size_t)q-first); double ru, rv;
                                apl_vertex_point(ref,(int32_t)vertex,&ru,&rv);
                                if (aprg_pair_test(ru-old_u,rv-old_w,old->xyz+3*v,ref->xyz+3*vertex,
                                                   reach,reach2_safe,reach2_inner,tol) > 0) continue;
                            }
                        }
                        violated = 1; partner = s->chart; partner_sample = q; break;
                    }
                }
            }
            if (violations) violations[ranges[range].query_base+v/query_stride] = (uint8_t)violated;
            if (partners) partners[ranges[range].query_base+v/query_stride] = partner;
            if (partner_samples) partner_samples[ranges[range].query_base+v/query_stride] = partner_sample;
            if (touched) { rim++; bad += (size_t)actual_bad; }
        }
    }
    Arena_restore(run->arena,mark);
    *n_rim = rim;
    return rim ? (double)bad / (double)rim : 0.0;
}

static double aprg_probe_excluding(AplRimGrid *g, const AsmRun *run, const AsmChart *const *charts,
                                    size_t n, double reach, size_t *n_rim, const uint8_t *excluded)
{
    return aprg_probe_record(g,run,charts,n,reach,n_rim,excluded,NULL,NULL,NULL,NULL,2);
}

static double aprg_probe(AplRimGrid *g, const AsmRun *run, const AsmChart *const *charts, size_t n, double reach, size_t *n_rim)
{
    return aprg_probe_excluding(g,run,charts,n,reach,n_rim,NULL);
}

static void apl_write_rim_failure(AplCtx *x, int32_t component, const AsmChart *const *charts, size_t n)
{
    unsigned bit = x->uv_trial ? 2u : 1u;
    if (!x->frim || !x->diag || x->cluster_kind != 1 || (x->diag[component].rim_logged & bit)) return;
    x->diag[component].rim_logged |= bit;
    AplRimGrid *g = x->rgrid;
    size_t rows = 0, queries = 0, examined = 0;
    const double reach = 24, outer = 24*24*(1.0+8.0*DBL_EPSILON), inner = 24*24*(1.0-64.0*DBL_EPSILON);
    for (size_t k = 0; k < n && rows < 32 && queries < 2048 && examined < 32768; k++) {
        const AsmChart *c = charts[k];
        double dl = x->run->chart_layer_d ? x->run->chart_layer_d[c->id] : 0;
        if (dl <= 0) dl = x->run->layer_d_global;
        double tol = fmax(.5*dl,7.0);
        for (size_t v = 0; v < c->nv && rows < 32 && queries < 2048 && examined < 32768; v += 2) {
            queries++;
            double u, w; apl_vertex_point(c,(int32_t)v,&u,&w);
            const float *p = c->xyz+3*v;
            int recorded = 0;
            for (int du = -1; du <= 1 && !recorded && examined < 32768; du++)
            for (int dv = -1; dv <= 1 && !recorded && examined < 32768; dv++) {
                int32_t *slot = aprg_slot(g,apl_rim_key(u+du*g->cell,w+dv*g->cell,g->cell),0);
                if (!slot) continue;
                for (int32_t q = *slot; q >= 0 && examined < 32768; q = g->next[q]) {
                    examined++;
                    const AplRimSample *s = &g->s[q];
                    if (x->excluded_samples && x->excluded_samples[s->chart]) continue;
                    double dx = s->u-u, dy = s->v-w;
                    if (aprg_pair_test(dx,dy,p,s->p,reach,outer,inner,tol) != 1) continue;
                    double physical = sqrt((s->p[0]-p[0])*(s->p[0]-p[0])+(s->p[1]-p[1])*(s->p[1]-p[1])+(s->p[2]-p[2])*(s->p[2]-p[2]));
                    const AsmChart *partner = &x->run->charts[s->chart];
                    fprintf(x->frim,"%d,%d,%d,%d,%d,%zu,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.17g,%.17g,%.17g\n",
                            x->attempt,component,x->cluster,x->uv_trial,c->id,v,s->chart,q,partner->placement_state,
                            u,w,s->u,s->v,p[0],p[1],p[2],s->p[0],s->p[1],s->p[2],hypot(dx,dy),physical,tol);
                    rows++; recorded = 1; break;
                }
            }
        }
    }
}

static int aprg_polish_blocked(AplRimGrid *g, const AsmRun *run, const AsmChart *saved, size_t n,
                               const uint8_t *excluded, double max_ratio, size_t *guaranteed, size_t *queries)
{
    *guaranteed = *queries = 0;
    if (!g || !g->n || !isfinite(max_ratio) || max_ratio < 0 || max_ratio >= 1) return 0;
    for (size_t k = 0; k < n; k++) {
        if (saved[k].id < 0 || (size_t)saved[k].id >= run->n_charts ||
            run->charts[saved[k].id].nv != saved[k].nv || !saved[k].uv ||
            !run->charts[saved[k].id].uv || !run->charts[saved[k].id].xyz) return 0;
        *queries += (saved[k].nv+1)/2;
    }
    for (size_t k = 0; k < n; k++) {
        const AsmChart *c = &run->charts[saved[k].id];
        double dl = run->chart_layer_d ? run->chart_layer_d[c->id] : 0;
        if (dl <= 0) dl = run->layer_d_global;
        double tol = fmax(.5*dl,7.0);
        for (size_t v = 0; v < c->nv; v += 2) {
            double consumed = hypot((double)c->uv[2*v]-saved[k].uv[2*v],
                                    (double)c->uv[2*v+1]-saved[k].uv[2*v+1]);
            double allowance = 8.0+consumed+1e-4, reach = 24.0-allowance;
            if (!(reach > 0)) continue;
            double u, w; apl_vertex_point(c,(int32_t)v,&u,&w);
            const float *p = c->xyz+3*v;
            int blocked = 0;
            for (int du = -1; du <= 1 && !blocked; du++) for (int dv = -1; dv <= 1 && !blocked; dv++) {
                int32_t *slot = aprg_slot(g,apl_rim_key(u+du*g->cell,w+dv*g->cell,g->cell),0);
                if (!slot) continue;
                for (int32_t q = *slot; q >= 0; q = g->next[q]) {
                    const AplRimSample *s = &g->s[q];
                    if (excluded && excluded[s->chart]) continue;
                    if (aprg_pair_test(s->u-u,s->v-w,p,s->p,reach,
                                      reach*reach*(1+8*DBL_EPSILON),reach*reach*(1-64*DBL_EPSILON),tol+allowance) > 0) {
                        blocked = 1; break;
                    }
                }
            }
            *guaranteed += (size_t)blocked;
            if (*guaranteed >= 8 && (double)*guaranteed > max_ratio*(double)*queries) return 1;
        }
    }
    return 0;
}

static size_t apl_internal_record(AsmRun *run, const AsmChart *saved, size_t n, int current,
                                   uint8_t *violations, int32_t *partners, int32_t *partner_vertices, int preserve_pairs)
{
    if (!n) return 0;
    Arena_Mark mark = Arena_save(run->arena);
    const size_t query_stride = preserve_pairs ? 1 : 2, reference_stride = preserve_pairs ? 1 : 4;
    size_t reserve = 0;
    for (size_t k = 0; k < n; k++) {
        size_t need = saved[k].nv/reference_stride+1;
        if (need >= (1u << 16)-reserve) { reserve = 1u << 16; break; }
        reserve += need;
    }
    AplRimGrid *grid = aprg_new_sized(run->arena,24.0,reserve);
    const AsmChart **charts = ARENA_ALLOC(run->arena, n*sizeof(*charts));
    AprgCheckpoint checkpoint = {0};
    if (current && (preserve_pairs || partner_vertices)) {
        checkpoint.count = run->n_charts;
        checkpoint.reference_stride = reference_stride;
        checkpoint.by_id = ARENA_CALLOC(run->arena,run->n_charts,sizeof *checkpoint.by_id);
        checkpoint.first_sample = ARENA_CALLOC(run->arena,run->n_charts,sizeof *checkpoint.first_sample);
    }
    for (size_t k = 0; k < n; k++) {
        charts[k] = current ? &run->charts[saved[k].id] : &saved[k];
        if (checkpoint.by_id) {
            checkpoint.by_id[saved[k].id] = &saved[k];
            checkpoint.first_sample[saved[k].id] = grid->n;
        }
        aprg_insert_chart_stride(grid,charts[k],reference_stride);
    }
    size_t touched = 0;
    double bad = aprg_probe_record(grid,run,charts,n,24.0,&touched,NULL,violations,partners,partner_vertices,
                                  current && preserve_pairs ? &checkpoint : NULL,query_stride);
    if (partner_vertices) for (size_t k = 0, at = 0; k < n; k++) for (size_t v = 0; v < saved[k].nv; v += query_stride, at++) {
        int32_t sample = partner_vertices[at]; partner_vertices[at] = -1;
        if (sample < 0 || (size_t)sample >= grid->n || !checkpoint.by_id) continue;
        int32_t id = grid->s[sample].chart;
        if (id < 0 || (size_t)id >= checkpoint.count || !checkpoint.by_id[id]) continue;
        size_t first = checkpoint.first_sample[id], count = checkpoint.by_id[id]->nv;
        if (count && (size_t)sample >= first && (size_t)sample-first <= (count-1)/reference_stride)
            partner_vertices[at] = (int32_t)(reference_stride*((size_t)sample-first));
    }
    size_t count = (size_t)floor(bad*(double)touched+0.5);
    Arena_restore(run->arena, mark);
    return count;
}

static size_t apl_internal_bad(AsmRun *run, const AsmChart *saved, size_t n, int current)
{
    return apl_internal_record(run,saved,n,current,NULL,NULL,NULL,0);
}

static size_t apl_internal_new(AsmRun *run, const AsmChart *saved, size_t n,
                               AsmContinuityAuditCache *cache, size_t *after_bad)
{
    size_t queries = 0; *after_bad = 0;
    cache->pairs_count = 0;
    int preserve_pairs = cache->guard_charts != NULL;
    const size_t query_stride = preserve_pairs ? 1 : 2;
    if (cache->blocked_charts) {
        if (cache->blocked_charts_count != run->n_charts) return SIZE_MAX;
        memset(cache->blocked_charts,0,run->n_charts);
    }
    if (cache->blocked_vertices) {
        if (cache->blocked_vertices_count != run->n_charts) return SIZE_MAX;
        for (size_t i = 0; i < run->n_charts; i++) if (cache->blocked_vertices[i])
            memset(cache->blocked_vertices[i],0,run->charts[i].nv);
    }
    for (size_t i = 0; i < n; i++) {
        if (saved[i].id < 0 || (size_t)saved[i].id >= run->n_charts || run->charts[saved[i].id].nv != saved[i].nv) return SIZE_MAX;
        queries += (saved[i].nv+query_stride-1)/query_stride;
    }
    if (!cache->baseline_ready) {
        Arena_T scratch = cache->scratch ? cache->scratch : run->arena;
        cache->baseline_violations = ARENA_CALLOC(scratch,queries ? queries : 1,1);
        cache->baseline_queries = queries;
        cache->baseline_bad = apl_internal_record(run,saved,n,0,cache->baseline_violations,NULL,NULL,preserve_pairs);
        cache->baseline_ready = 1;
    }
    if (cache->baseline_queries != queries || !cache->baseline_violations) return SIZE_MAX;
    Arena_Mark mark = Arena_save(run->arena);
    uint8_t *current = ARENA_CALLOC(run->arena,queries ? queries : 1,1);
    int report_pairs = cache->pairs && cache->pairs_capacity;
    int32_t *partners = cache->blocked_charts || cache->blocked_vertices || report_pairs ? ARENA_ALLOC(run->arena,(queries ? queries : 1)*sizeof *partners) : NULL;
    int32_t *vertices = cache->blocked_vertices || report_pairs ? ARENA_ALLOC(run->arena,(queries ? queries : 1)*sizeof *vertices) : NULL;
    *after_bad = apl_internal_record(run,saved,n,1,current,partners,vertices,preserve_pairs);
    size_t introduced = 0;
    for (size_t k = 0, q = 0; k < n; k++) for (size_t v = 0; v < saved[k].nv; v += query_stride, q++) {
        if (!current[q] || (!preserve_pairs && cache->baseline_violations[q])) continue;
        introduced++;
        if (cache->blocked_charts) {
            cache->blocked_charts[saved[k].id] = 1;
            if (partners[q] >= 0 && (size_t)partners[q] < run->n_charts) cache->blocked_charts[partners[q]] = 1;
        }
        if (cache->blocked_vertices) {
            if (cache->blocked_vertices[saved[k].id]) cache->blocked_vertices[saved[k].id][v] = 1;
            int32_t id = partners[q], vertex = vertices[q];
            if (id >= 0 && (size_t)id < run->n_charts && vertex >= 0 && (size_t)vertex < run->charts[id].nv && cache->blocked_vertices[id])
                cache->blocked_vertices[id][vertex] = 1;
        }
        if (report_pairs && cache->pairs_count < cache->pairs_capacity) {
            int32_t id = partners[q], vertex = vertices[q];
            if (id >= 0 && (size_t)id < run->n_charts && vertex >= 0 && (size_t)vertex < run->charts[id].nv)
                cache->pairs[cache->pairs_count++] = (AsmContinuityUvPair){saved[k].id,(int32_t)v,id,vertex};
        }
    }
    Arena_restore(run->arena,mark); return introduced;
}

typedef struct AplSelfContact {
    const AsmChart *saved;
    AsmContinuityAuditCache *cache;
    const AsmContacts *query;
    const double *before;
    size_t introduced;
} AplSelfContact;

static void apl_self_contact(void *context, size_t a, size_t b, double area)
{
    AplSelfContact *x = context;
    const AsmChart *c = x->saved;
    double before=AsmContacts_pair_faces(x->query,x->before,a,b,NULL,NULL);
    if (isfinite(before) && before > 0 && area <= before+fmax(1e-10,1e-10*before)) return;
    x->introduced++;
    if (x->cache->blocked_charts) x->cache->blocked_charts[c->id] = 1;
    if (x->cache->blocked_vertices && x->cache->blocked_vertices[c->id]) {
        for (int k = 0; k < 3; k++) {
            x->cache->blocked_vertices[c->id][c->faces[3*a+k]] = 1;
            x->cache->blocked_vertices[c->id][c->faces[3*b+k]] = 1;
        }
    }
}

static int apl_self_contacts_preserved(AsmRun *run, const AsmChart *saved, size_t n, AsmContinuityAuditCache *cache)
{
    for (size_t i = 0; i < n; i++) {
        const AsmChart *old = &saved[i], *c = &run->charts[old->id];
        if (c->nf < 2) continue;
        if (!old->uv || !c->uv || c->nv != old->nv || c->nf != old->nf || c->faces != old->faces ||
            c->nv > SIZE_MAX/(2*sizeof(double))) return 0;
        if (c->uv == old->uv || !memcmp(c->uv,old->uv,2*c->nv*sizeof(float))) continue;
        Arena_Mark mark = Arena_save(run->arena);
        double *uv = ARENA_ALLOC(run->arena,2*c->nv*sizeof *uv);
        double *before=ARENA_ALLOC(run->arena,2*c->nv*sizeof *before);
        for (size_t v = 0; v < 2*c->nv; v++) uv[v] = c->uv[v];
        for(size_t v=0;v<2*c->nv;v++)before[v]=old->uv[v];
        AsmContacts *query = AsmContacts_new(run->arena,c->nv,c->faces,c->nf,uv,AsmContacts_material_tolerance());
        AsmContactStats stats;
        AplSelfContact context={old,cache,query,before,0};
        int ok = query && !AsmContacts_measure(query,uv,NULL,apl_self_contact,&context,&stats) && !context.introduced;
        Arena_restore(run->arena,mark);
        if (!ok) return 0;
    }
    return 1;
}

static int apl_uv_audit(AsmRun *run, const AsmChart *saved, size_t n, AsmContinuityAuditCache *cache)
{
    size_t after;
    cache->pairs_count = 0;
    if (!AsmContinuity_preserves_incumbent_trial(run,saved,n)) return 0;
    if (!!cache->guard_charts != !!cache->guard_count) return 0;
    if (cache->guard_charts) { saved = cache->guard_charts; n = cache->guard_count; }
    return apl_internal_new(run,saved,n,cache,&after) == 0 && apl_self_contacts_preserved(run,saved,n,cache);
}

static int apl_registration_audit(AsmRun *run, const AsmChart *saved, size_t n, AsmContinuityAuditCache *cache)
{
    if (!AsmContinuity_preserves_incumbent_trial(run,saved,n)) return 0;
    size_t after, introduced = apl_internal_new(run,saved,n,cache,&after);
    if (introduced) fprintf(stderr,"  [registration geometry] component %d, %zu charts: bad rim vertices %zu -> %zu, %zu %s; trial rejected\n",
                            saved[0].component,n,cache->baseline_bad,after,introduced,
                            cache->guard_charts ? "queries with new conflicting pairs" : "previously sound vertices violated");
    return introduced == 0;
}

typedef struct AplCutRow {
    int32_t component; int c1, c2;
    double w1, w2, rms1, rms2, u1, u2, sep, min_sep, t0;
    size_t n1, n2, m1, m2;
    double area1, area2;
    size_t crossing, crossing_sound, sound_max_support;
    double sound_min_rms;
    long first_sound_rel;
} AplCutRow;

static void apl_write_cut(const AplCtx *x, const AplCutRow *r, const char *outcome)
{
    if (!x->fcut) return;
    fprintf(x->fcut, "%d,%d,%s,%d,%d,%.6g,%.6g,%.6g,%.6g,%.2f,%.2f,%.2f,%.2f,%.2f,"
                     "%zu,%zu,%zu,%zu,%.6e,%.6e,%zu,%zu,%.6g,%zu,%ld\n",
            x->attempt, r->component, outcome, r->c1, r->c2, r->w1, r->w2, r->rms1, r->rms2,
            r->u1, r->u2, r->sep, r->min_sep, r->t0,
            r->n1, r->n2, r->m1, r->m2, r->area1, r->area2,
            r->crossing, r->crossing_sound, r->sound_min_rms, r->sound_max_support, r->first_sound_rel);
}

static int32_t apl_cut_component(AplCtx *x, int32_t best, size_t na, int ncl, const double *cl_w, const double *cl_u, const double *cl_rms,
                                 double gate, const int32_t *rel_off, const int32_t *rel_adj, int32_t *ncomp_io, size_t ncomp_cap,
                                 double *comp_area, int32_t *orig_comp, AplOrient *ori, AplCompDiag *diag, int32_t *comp_ring, const double *lidx,
                                 size_t *rels_deferred)
{
    AsmRun *run = x->run;
    size_t nc = run->n_charts;
    AplCutRow row; memset(&row, 0, sizeof row);
    row.component = best; row.c1 = row.c2 = -1; row.first_sound_rel = -1;
    row.sound_min_rms = -1.0;
    if ((size_t)*ncomp_io + 1 >= ncomp_cap) { apl_write_cut(x, &row, "component_cap"); return -1; }
    int c1 = -1, c2 = -1;
    for (int cl = 0; cl < ncl; cl++) {
        if (cl_rms[cl] > gate) continue;
        if (c1 < 0 || cl_w[cl] > cl_w[c1]) { c2 = c1; c1 = cl; }
        else if (c2 < 0 || cl_w[cl] > cl_w[c2]) c2 = cl;
    }
    row.c1 = c1; row.c2 = c2; row.t0 = x->t0;
    if (c1 >= 0) { row.w1 = cl_w[c1]; row.rms1 = cl_rms[c1]; row.u1 = cl_u[c1]; }
    if (c2 >= 0) { row.w2 = cl_w[c2]; row.rms2 = cl_rms[c2]; row.u2 = cl_u[c2]; }
    if (c1 < 0 || c2 < 0) { apl_write_cut(x, &row, "no_two_fitting_clusters"); return -1; }
    double sep = fabs(cl_u[c1] - cl_u[c2]), min_sep = fmax(APL_CUT_MIN_SEP, 0.25 * x->t0);
    row.sep = sep; row.min_sep = min_sep;
    if (sep < min_sep) { apl_write_cut(x, &row, "clusters_too_close"); return -1; }
    Arena_Mark mark = Arena_save(run->arena);
    int8_t *label = ARENA_CALLOC(run->arena, nc, 1);
    for (size_t i = 0; i < na; i++) {
        const AplAnchor *a = &x->anchors[i];
        if (a->cluster != c1 && a->cluster != c2) continue;
        int8_t l = a->cluster == c1 ? 1 : 2;
        int32_t ch = a->chart_c;
        if (ch < 0 || (size_t)ch >= nc || run->charts[(size_t)ch].component != best) continue;
        if (label[ch] == 0) label[ch] = l;
        else if (label[ch] != l) label[ch] = 3;
    }
    size_t n1 = 0, n2 = 0;
    for (size_t i = 0; i < nc; i++) { if (label[i] == 3) label[i] = 0; else if (label[i] == 1) n1++; else if (label[i] == 2) n2++; }
    row.n1 = n1; row.n2 = n2;
    if (n1 == 0 || n2 == 0) { apl_write_cut(x, &row, "no_distinct_charts"); Arena_restore(run->arena, mark); return -1; }
    int32_t *queue = ARENA_ALLOC(run->arena, nc * sizeof(int32_t));
    size_t qh = 0, qt = 0;
    for (size_t i = 0; i < nc; i++) if (label[i] != 0 && run->charts[i].component == best) queue[qt++] = (int32_t)i;
    while (qh < qt) {
        int32_t u = queue[qh++];
        for (int32_t e = rel_off[u]; e < rel_off[u + 1]; e++) {
            const AsmRelation *R = &run->rels[(size_t)rel_adj[e]];
            if (R->flags & (ASM_REL_DROPPED | ASM_REL_CONTACT | ASM_REL_SWITCHED | ASM_REL_PLACEMENT_ONLY)) continue;
            int32_t v = R->a == u ? R->b : R->a;
            if (!AsmChart_in_layout(&run->charts[(size_t)v]) || run->charts[(size_t)v].component != best) continue;
            if (label[v] != 0) continue;
            label[v] = label[u];
            queue[qt++] = v;
        }
    }
    size_t m1 = 0, m2 = 0; double a1 = 0.0, a2 = 0.0;
    for (size_t i = 0; i < nc; i++) {
        if (!AsmChart_in_layout(&run->charts[i]) || run->charts[i].component != best) continue;
        if (label[i] == 0) label[i] = 1;
        if (label[i] == 1) { m1++; a1 += run->charts[i].area3d; } else { m2++; a2 += run->charts[i].area3d; }
    }
    row.m1 = m1; row.m2 = m2; row.area1 = a1; row.area2 = a2;
    if (m1 == 0 || m2 == 0) { apl_write_cut(x, &row, "partition_empty"); Arena_restore(run->arena, mark); return -1; }
    int8_t moved = a2 <= a1 ? 2 : 1;
    size_t crossing = 0, crossing_sound = 0, sound_support = 0;
    double sound_min_rms = -1.0;
    long first_sound = -1;
    for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *R = &run->rels[r];
        if (run->charts[R->a].component != best || run->charts[R->b].component != best ||
            label[R->a] == label[R->b]) continue;
        crossing++;
        double boundary_rms = -1.0; size_t boundary_support = 0;
        if (!((R->continuity & ASM_CONT_INCUMBENT) ||
              AsmContinuity_measure(run,R,&boundary_rms,&boundary_support))) continue;
        crossing_sound++;
        if (first_sound < 0) first_sound = (long)r;
        if (boundary_rms >= 0.0 && (sound_min_rms < 0.0 || boundary_rms < sound_min_rms)) sound_min_rms = boundary_rms;
        if (boundary_support > sound_support) sound_support = boundary_support;
    }
    row.crossing = crossing; row.crossing_sound = crossing_sound;
    row.sound_min_rms = sound_min_rms; row.sound_max_support = sound_support; row.first_sound_rel = first_sound;
    if (crossing_sound) {
        x->cuts_preserved_source++;
        apl_write_cut(x, &row, "source_preserved");
        Arena_restore(run->arena,mark); return -1;
    }
    apl_write_cut(x, &row, "cut");
    int32_t newc = (*ncomp_io)++;
    size_t deferred = 0;
    for (size_t r = 0; r < run->n_rels; r++) {
        AsmRelation *R = &run->rels[r];
        if (R->flags & (ASM_REL_DROPPED | ASM_REL_CONTACT | ASM_REL_PLACEMENT_ONLY)) continue;
        if (run->charts[(size_t)R->a].component != best || run->charts[(size_t)R->b].component != best) continue;
        if (label[R->a] != label[R->b]) {
            double boundary_rms; size_t boundary_support;
            if ((R->continuity & ASM_CONT_VERIFIED) && AsmContinuity_measure(run, R, &boundary_rms, &boundary_support))
                R->continuity |= ASM_CONT_CUT;
            R->flags |= ASM_REL_SWITCHED;
            R->continuity &= ~(uint32_t)ASM_CONT_VERIFIED;
            deferred++;
        }
    }
    comp_area[newc] = 0.0; comp_area[best] = 0.0;
    comp_ring[newc] = 1000000; comp_ring[best] = 1000000;
    for (size_t i = 0; i < nc; i++) {
        AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != best) continue;
        if (label[i] == moved) { c->component = newc; orig_comp[i] = newc; }
        int32_t cc = c->component;
        comp_area[cc] += c->area3d;
        if (lidx[i] > APL_LAYER_UNKNOWN) { int32_t ring = (int32_t)floor(fabs(lidx[i]) + 0.5); if (ring < comp_ring[cc]) comp_ring[cc] = ring; }
    }
    ori[newc] = ori[best];
    memset(&diag[newc], 0, sizeof diag[newc]);
    *rels_deferred += deferred;
    fprintf(stderr, "  [place] CUT component %d: clusters %d (u %.0f, weight %.1f, rms %.1f) and %d (u %.0f, weight %.1f, rms %.1f) disagree by %.0f vox; "
                    "seeds %zu + %zu charts, parts %zu charts / %.3e vox^2 and %zu charts / %.3e vox^2, %zu seam relations deferred -> new component %d\n",
            best, c1 + 1, cl_u[c1], cl_w[c1], cl_rms[c1], c2 + 1, cl_u[c2], cl_w[c2], cl_rms[c2], sep, n1, n2,
            moved == 2 ? m1 : m2, moved == 2 ? a1 : a2, moved == 2 ? m2 : m1, moved == 2 ? a2 : a1, deferred, newc);
    Arena_restore(run->arena, mark);
    return newc;
}

typedef struct AplObs { int32_t ca, cb; double du, dv, w, k, s, wt; } AplObs;

static size_t apl_metric_resolve(AplCtx *x, const uint8_t *comp_placed, const int32_t *orig_comp, int32_t ncomp,
                                 const double *lidx, int32_t primary, Arena_T *garena, const AsmConflictOpts *copts, size_t *n_obs_out,
                                 int model_only)
{
    AsmRun *run = x->run;
    Arena_T arena = run->arena;
    size_t nc = run->n_charts;
    *n_obs_out = 0;
    if (lidx == NULL) return 0;
    Arena_Mark mark = Arena_save(arena);
    AplObs *obs = ARENA_ALLOC(arena, (run->n_layers ? run->n_layers : 1) * sizeof(AplObs));
    size_t no = 0;
    double *tmpu = ARENA_ALLOC(arena, 64 * sizeof(double)), *tmpv = ARENA_ALLOC(arena, 64 * sizeof(double));
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int32_t ca = orig_comp[lp->a], cb = orig_comp[lp->b];
        if (ca < 0 || cb < 0 || comp_placed[ca] != 1 || comp_placed[cb] != 1) continue;
        if (lidx[lp->a] <= APL_LAYER_UNKNOWN || lidx[lp->b] <= APL_LAYER_UNKNOWN) continue;
        double dl = lidx[lp->b] - lidx[lp->a];
        if (fabs(dl) < 0.5) continue;
        int k = (int)floor(fabs(dl) + 0.5);
        if (k > 4) continue;
        size_t nh = 0;
        for (int h = 0; h < lp->hit_count && nh < 64; h++) {
            const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
            double ax, ay, bx, by;
            apl_vertex_point(A, hit->va, &ax, &ay);
            apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
            tmpu[nh] = bx - ax; tmpv[nh] = by - ay; nh++;
        }
        if (nh < 2) continue;
        qsort(tmpu, nh, sizeof(double), apl_cmp_double);
        qsort(tmpv, nh, sizeof(double), apl_cmp_double);
        AplObs *o = &obs[no++];
        o->ca = ca; o->cb = cb; o->du = tmpu[nh / 2]; o->dv = tmpv[nh / 2];
        o->w = 0.5 * (lidx[lp->a] + lidx[lp->b]); o->k = (double)k; o->s = dl < 0.0 ? -1.0 : 1.0;
        o->wt = (double)(lp->hit_count < 16 ? lp->hit_count : 16);
    }
    *n_obs_out = no;
    if (no < 20) { Arena_restore(arena, mark); return 0; }
    double t0 = x->t0, g = x->growth;
    {
        double *wr = ARENA_ALLOC(arena, no * sizeof(double));
        for (size_t i = 0; i < no; i++) wr[i] = 1.0;
        for (int it = 0; it < 6; it++) {
            double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (size_t i = 0; i < no; i++) {
                const AplObs *o = &obs[i];
                if (o->du * o->s <= 0.0) continue;
                double tt = fabs(o->du) / o->k, ww = wr[i] * o->wt;
                sw += ww; sx += ww * o->w; sy += ww * tt; sxx += ww * o->w * o->w; sxy += ww * o->w * tt;
            }
            if (sw <= 0.0) break;
            double det = sw * sxx - sx * sx;
            double gg = fabs(det) > 1e-9 ? (sw * sxy - sx * sy) / det : 0.0;
            if (gg < 0.0) gg = 0.0;
            if (gg > 400.0) gg = 400.0;
            double tt0 = (sy - gg * sx) / sw;
            if (tt0 < APL_TURN_MIN) tt0 = APL_TURN_MIN;
            if (tt0 > APL_TURN_MAX) tt0 = APL_TURN_MAX;
            t0 = tt0; g = gg;
            double scale = 0.15 * t0;
            for (size_t i = 0; i < no; i++) {
                const AplObs *o = &obs[i];
                double r = (o->du * o->s > 0.0) ? fabs(o->du) / o->k - (t0 + g * o->w) : 4.0 * scale;
                wr[i] = 1.0 / (1.0 + (r / scale) * (r / scale));
            }
        }
    }
    if (model_only) {
        double *res = ARENA_ALLOC(arena, no * sizeof(double));
        size_t nr = 0;
        for (size_t i = 0; i < no; i++) {
            const AplObs *o = &obs[i];
            if (o->du * o->s <= 0.0) continue;
            res[nr++] = fabs(fabs(o->du) / o->k - (t0 + g * o->w));
        }
        double med = 0.0;
        if (nr) { qsort(res, nr, sizeof(double), apl_cmp_double); med = res[nr/2]; }
        int sound = nr >= (size_t)APL_TURN_MODEL_MIN_PAIRS && t0 > APL_TURN_MIN && t0 < APL_TURN_MAX &&
                    med <= APL_TURN_MODEL_MAX_RESID * t0;
        if (sound) { x->t0 = t0; x->growth = g; x->t0_measured = 1; }
        fprintf(stderr, "  [place] pooled turn model over %zu placed pairs (%zu sign-agreeing): T(w) = %.0f + w x %.1f vox, residual p50 %.0f vox (%.0f%% of a turn): %s\n",
                no, nr, t0, g, med, t0 > 0.0 ? 100.0 * med / t0 : 0.0, sound ? "adopted" : "refused");
        Arena_restore(arena, mark);
        return 0;
    }
    double *d = ARENA_CALLOC(arena, (size_t)ncomp, sizeof(double));
    double *rhs = ARENA_ALLOC(arena, (size_t)ncomp * sizeof(double)), *rr = ARENA_ALLOC(arena, (size_t)ncomp * sizeof(double));
    double *pp = ARENA_ALLOC(arena, (size_t)ncomp * sizeof(double)), *Ap = ARENA_ALLOC(arena, (size_t)ncomp * sizeof(double));
    double *wgt = ARENA_ALLOC(arena, no * sizeof(double));
    uint8_t *active = ARENA_CALLOC(arena, (size_t)ncomp, 1);
    uint8_t *use = ARENA_CALLOC(arena, no, 1);
    for (size_t i = 0; i < no; i++) {
        wgt[i] = obs[i].wt;
        use[i] = (obs[i].ca != obs[i].cb && obs[i].du * obs[i].s > 0.0) ? 1 : 0;
        if (use[i]) { active[obs[i].ca] = 1; active[obs[i].cb] = 1; }
    }
    active[primary] = 0;
    double tbar = t0 + g * 0.0;
    for (int round = 0; round < 4; round++) {
        memset(rhs, 0, (size_t)ncomp * sizeof(double));
        for (size_t i = 0; i < no; i++) {
            const AplObs *o = &obs[i];
            if (!use[i]) continue;
            double target = o->s * o->k * (t0 + g * o->w) - o->du;
            if (active[o->cb]) rhs[o->cb] += wgt[i] * target;
            if (active[o->ca]) rhs[o->ca] -= wgt[i] * target;
        }
        for (int32_t c = 0; c < ncomp; c++) rr[c] = 0.0;
        for (size_t i = 0; i < no; i++) {
            const AplObs *o = &obs[i];
            if (!use[i]) continue;
            double t = wgt[i] * (d[o->cb] - d[o->ca]);
            if (active[o->cb]) rr[o->cb] += t;
            if (active[o->ca]) rr[o->ca] -= t;
        }
        for (int32_t c = 0; c < ncomp; c++) rr[c] = active[c] ? rhs[c] - rr[c] : 0.0;
        memcpy(pp, rr, (size_t)ncomp * sizeof(double));
        double rs = 0.0;
        for (int32_t c = 0; c < ncomp; c++) rs += rr[c] * rr[c];
        double rs0 = rs;
        for (int it = 0; it < 500 && rs > 1e-12 * (rs0 > 0.0 ? rs0 : 1.0); it++) {
            for (int32_t c = 0; c < ncomp; c++) Ap[c] = 0.0;
            for (size_t i = 0; i < no; i++) {
                const AplObs *o = &obs[i];
                if (!use[i]) continue;
                double t = wgt[i] * (pp[o->cb] - pp[o->ca]);
                if (active[o->cb]) Ap[o->cb] += t;
                if (active[o->ca]) Ap[o->ca] -= t;
            }
            for (int32_t c = 0; c < ncomp; c++) if (active[c]) Ap[c] += 1e-6 * pp[c];
            double pAp = 0.0;
            for (int32_t c = 0; c < ncomp; c++) if (active[c]) pAp += pp[c] * Ap[c];
            if (!(pAp > 0.0)) break;
            double alpha = rs / pAp;
            for (int32_t c = 0; c < ncomp; c++) if (active[c]) { d[c] += alpha * pp[c]; rr[c] -= alpha * Ap[c]; }
            double rs2 = 0.0;
            for (int32_t c = 0; c < ncomp; c++) rs2 += rr[c] * rr[c];
            double beta = rs2 / rs; rs = rs2;
            for (int32_t c = 0; c < ncomp; c++) pp[c] = active[c] ? rr[c] + beta * pp[c] : 0.0;
        }
        double scale = 0.08 * t0;
        for (size_t i = 0; i < no; i++) {
            const AplObs *o = &obs[i];
            double r = (o->du + d[o->cb] - d[o->ca]) - o->s * o->k * (t0 + g * o->w);
            wgt[i] = o->wt / (1.0 + (r / scale) * (r / scale));
        }
    }
    double cap = APL_METRIC_MAX_SHIFT * tbar, dmax = 0.0, dsum = 0.0; size_t moved = 0, nact = 0;
    for (int32_t c = 0; c < ncomp; c++) {
        if (!active[c]) { d[c] = 0.0; continue; }
        if (d[c] > cap) d[c] = cap;
        if (d[c] < -cap) d[c] = -cap;
        nact++; dsum += fabs(d[c]);
        if (fabs(d[c]) > dmax) dmax = fabs(d[c]);
        if (fabs(d[c]) > 1.0) moved++;
    }
    double r0 = 0.0, r1 = 0.0;
    {
        size_t nu = 0;
        double *ra = ARENA_ALLOC(arena, (no ? no : 1) * sizeof(double)), *rb = ARENA_ALLOC(arena, (no ? no : 1) * sizeof(double));
        for (size_t i = 0; i < no; i++) {
            const AplObs *o = &obs[i];
            if (!use[i]) continue;
            double model = o->s * o->k * (t0 + g * o->w);
            ra[nu] = fabs(o->du - model);
            rb[nu] = fabs(o->du + d[o->cb] - d[o->ca] - model);
            nu++;
        }
        if (nu > 0) { qsort(ra, nu, sizeof(double), apl_cmp_double); qsort(rb, nu, sizeof(double), apl_cmp_double); r0 = ra[nu / 2]; r1 = rb[nu / 2]; }
    }
    int applied = r1 + 1e-9 < r0;
    if (applied) {
        for (size_t i = 0; i < nc; i++) {
            AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c)) continue;
            int32_t oc = orig_comp[i];
            if (oc < 0 || oc >= ncomp || !active[oc]) continue;
            c->pose_x += d[oc];
        }
        if (no >= 50) { x->t0 = t0; x->growth = g; x->t0_measured = 1; }
    } else moved = 0;
    fprintf(stderr, "  [place] metric re-solve: %zu placed pairs, turn model T(w) = %.0f + w x %.1f vox, %zu components refit (shift mean %.0f max %.0f vox, %zu moved > 1 vox), residual p50 %.1f -> %.1f: %s\n",
            no, t0, g, nact, nact ? dsum / (double)nact : 0.0, dmax, moved, r0, r1, applied ? "applied" : "not applied");
    Arena_restore(arena, mark);
    if (copts && garena != NULL && *garena != NULL && moved > 0) {
        if (x->cgrid != NULL) AsmConflictGrid_dispose(x->cgrid);
        Arena_dispose(garena);
        *garena = Arena_new();
        x->cgrid = AsmConflictGrid_new(*garena, copts, run);
        x->rgrid = aprg_new(*garena, 24.0);
        x->strip_umin = 1e300; x->strip_umax = -1e300;
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c) || orig_comp[i] < 0 || comp_placed[orig_comp[i]] != 1) continue;
            AsmConflictGrid_insert(x->cgrid, c); aprg_insert_chart(x->rgrid, c);
            if (c->pose_x < x->strip_umin) x->strip_umin = c->pose_x;
            if (c->pose_x > x->strip_umax) x->strip_umax = c->pose_x;
        }
    }
    return moved;
}

static size_t apl_kover_build(AplCtx *x, const uint8_t *comp_placed, const int32_t *orig_comp, const double *lidx, double *kover)
{
    AsmRun *run = x->run;
    Arena_T arena = run->arena;
    size_t n = 0;
    Arena_Mark mark = Arena_save(arena);
    double *tmp = ARENA_ALLOC(arena, 64 * sizeof(double));
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        kover[p] = 0.0;
        const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int32_t ca = orig_comp[lp->a], cb = orig_comp[lp->b];
        if (ca < 0 || cb < 0 || comp_placed[ca] != 1 || comp_placed[cb] != 1) continue;
        size_t nh = 0;
        for (int h = 0; h < lp->hit_count && nh < 64; h++) {
            const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
            double ax, ay, bx, by;
            apl_vertex_point(A, hit->va, &ax, &ay);
            apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
            tmp[nh++] = bx - ax;
        }
        if (nh < 2) continue;
        qsort(tmp, nh, sizeof(double), apl_cmp_double);
        double du = tmp[nh / 2];
        double w = 0.0;
        if (lidx != NULL && lidx[lp->a] > APL_LAYER_UNKNOWN && lidx[lp->b] > APL_LAYER_UNKNOWN) w = fmin(lidx[lp->a], lidx[lp->b]);
        double T = x->t0 + w * x->growth;
        if (T < APL_TURN_MIN) T = APL_TURN_MIN;
        double q = fabs(du) / T;
        if (q < 0.5 || q > 4.5) continue;
        int k = (int)floor(q + 0.5);
        kover[p] = du < 0.0 ? -(double)k : (double)k;
        n++;
    }
    Arena_restore(arena, mark);
    return n;
}

static void apl_comp_ring_update(const AsmRun *run, const double *lidx, const int32_t *orig_comp, int32_t ncomp, int32_t *comp_ring)
{
    for (int32_t c = 0; c < ncomp; c++) comp_ring[c] = 1000000;
    for (size_t i = 0; i < run->n_charts; i++) {
        if (lidx[i] <= APL_LAYER_UNKNOWN || !AsmChart_in_layout(&run->charts[i])) continue;
        int32_t ring = (int32_t)floor(fabs(lidx[i]) + 0.5), oc = orig_comp[i];
        if (oc >= 0 && oc < ncomp && ring < comp_ring[oc]) comp_ring[oc] = ring;
    }
}

typedef struct AplTObs { double u, w, t; } AplTObs;
static int apl_cmp_tobs_cell(const void *x, const void *y)
{
    const AplTObs *a = x, *b = y;
    long ca = (long)floor(a->w / APL_TFIELD_DW), cb = (long)floor(b->w / APL_TFIELD_DW);
    if (ca != cb) return ca < cb ? -1 : 1;
    long ua = (long)floor(a->u / APL_TFIELD_DU), ub = (long)floor(b->u / APL_TFIELD_DU);
    if (ua != ub) return ua < ub ? -1 : 1;
    return (a->t > b->t) - (a->t < b->t);
}

static size_t apl_turn_field_build(AplCtx *x, const uint8_t *comp_placed, const int32_t *orig_comp, const double *lidx)
{
    AsmRun *run = x->run;
    Arena_T arena = run->arena;
    size_t nc = run->n_charts, filled = 0;
    for (size_t i = 0; i < nc; i++) x->tfield[i] = 0.0;
    if (lidx == NULL) return 0;
    Arena_Mark mark = Arena_save(arena);
    AplTObs *obs = ARENA_ALLOC(arena, (run->n_layers ? run->n_layers : 1) * sizeof(AplTObs));
    size_t no = 0;
    double *tmp = ARENA_ALLOC(arena, 64 * sizeof(double));
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *lp = &run->layers[p];
        const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
        int32_t ca = orig_comp[lp->a], cb = orig_comp[lp->b];
        if (ca < 0 || cb < 0 || comp_placed[ca] != 1 || comp_placed[cb] != 1) continue;
        if (lidx[lp->a] <= APL_LAYER_UNKNOWN || lidx[lp->b] <= APL_LAYER_UNKNOWN) continue;
        double dl = lidx[lp->b] - lidx[lp->a];
        if (fabs(dl) < 0.5) continue;
        int k = (int)floor(fabs(dl) + 0.5);
        if (k > 4) continue;
        double sgn = dl < 0.0 ? -1.0 : 1.0;
        size_t nh = 0; double usum = 0.0;
        for (int h = 0; h < lp->hit_count && nh < 64; h++) {
            const AsmLayerHit *hit = &run->layer_hits[(size_t)lp->hit_first + (size_t)h];
            double ax, ay, bx, by;
            apl_vertex_point(A, hit->va, &ax, &ay);
            apl_face_point(B, hit->fb, hit->l0, hit->l1, &bx, &by);
            tmp[nh++] = bx - ax; usum += ax;
        }
        if (nh < 2) continue;
        qsort(tmp, nh, sizeof(double), apl_cmp_double);
        double du = tmp[nh / 2];
        if (du * sgn <= 0.0) continue;
        double t = fabs(du) / (double)k;
        if (t < APL_TURN_MIN || t > APL_TURN_MAX) continue;
        obs[no].u = usum / (double)nh; obs[no].w = 0.5 * (lidx[lp->a] + lidx[lp->b]) - 0.5 * sgn * (double)k;
        obs[no].w = lidx[lp->a] < lidx[lp->b] ? lidx[lp->a] : lidx[lp->b];
        obs[no].t = t; no++;
    }
    if (no < (size_t)APL_TFIELD_MIN_OBS) { Arena_restore(arena, mark); return 0; }
    qsort(obs, no, sizeof(AplTObs), apl_cmp_tobs_cell);
    size_t ncell = 0;
    for (size_t i = 0; i < no; i++) if (i == 0 || apl_cmp_tobs_cell(&obs[i - 1], &obs[i]) != 0 || (long)floor(obs[i-1].w / APL_TFIELD_DW) != (long)floor(obs[i].w / APL_TFIELD_DW) || (long)floor(obs[i-1].u / APL_TFIELD_DU) != (long)floor(obs[i].u / APL_TFIELD_DU)) ncell++;
    size_t *cstart = ARENA_ALLOC(arena, (ncell + 1) * sizeof(size_t));
    long *cw = ARENA_ALLOC(arena, (ncell ? ncell : 1) * sizeof(long)), *cu = ARENA_ALLOC(arena, (ncell ? ncell : 1) * sizeof(long));
    size_t q = 0;
    for (size_t i = 0; i < no; i++) {
        long wi = (long)floor(obs[i].w / APL_TFIELD_DW), ui = (long)floor(obs[i].u / APL_TFIELD_DU);
        if (i == 0 || wi != cw[q - 1] || ui != cu[q - 1]) { cstart[q] = i; cw[q] = wi; cu[q] = ui; q++; }
    }
    cstart[q] = no; ncell = q;
    double *vals = ARENA_ALLOC(arena, (no ? no : 1) * sizeof(double));
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || orig_comp[i] < 0 || comp_placed[orig_comp[i]] != 1 || lidx[i] <= APL_LAYER_UNKNOWN) continue;
        double gx, gy;
        apl_pose(c, 0.0, 0.0, &gx, &gy);
        long wi = (long)floor(lidx[i] / APL_TFIELD_DW), ui = (long)floor(gx / APL_TFIELD_DU);
        size_t nv = 0;
        for (size_t k = 0; k < ncell; k++) {
            if (cw[k] < wi - 1 || cw[k] > wi + 1 || cu[k] < ui - 1 || cu[k] > ui + 1) continue;
            for (size_t j = cstart[k]; j < cstart[k + 1]; j++) vals[nv++] = obs[j].t;
        }
        if (nv < (size_t)APL_TFIELD_MIN_OBS) continue;
        qsort(vals, nv, sizeof(double), apl_cmp_double);
        x->tfield[i] = vals[nv / 2];
        filled++;
    }
    Arena_restore(arena, mark);
    return filled;
}

static int apl_close_seam(AplCtx *x, double *th, double *tx, double *ty)
{
    if (x->cluster_kind != 1) return 1;
    double cx = 0, cy = 0, sw = 0;
    for (size_t i = 0; i < x->na; i++) if (x->wts[i] > 0 && x->anchors[i].k == 0) {
        cx += x->wts[i]*x->anchors[i].sx; cy += x->wts[i]*x->anchors[i].sy; sw += x->wts[i];
    }
    if (sw < APL_SEAM_MIN_ANCHORS) return 0;
    cx /= sw; cy /= sw;
    double gx = cos(*th)*cx-sin(*th)*cy+*tx, gy = sin(*th)*cx+cos(*th)*cy+*ty;
    const int rows[6] = {0,1,1,2,2,2}, cols[6] = {0,0,1,0,1,2};
    for (int it = 0; it < 5; it++) {
        double H[6] = {0}, b[3] = {0}, delta[3] = {0};
        double c = cos(*th), s = sin(*th);
        for (size_t i = 0; i < x->na; i++) {
            const AplAnchor *a = &x->anchors[i];
            if (x->wts[i] <= 0 || a->k != 0) continue;
            double u = a->sx-cx, v = a->sy-cy, px = c*u-s*v, py = s*u+c*v;
            double ex = a->tx-px-gx, ey = a->ty-py-gy, w = x->wts[i];
            H[0] += w; H[2] += w; H[3] -= w*py; H[4] += w*px; H[5] += w*(px*px+py*py);
            b[0] += w*ex; b[1] += w*ey; b[2] += w*(-py*ex+px*ey);
        }
        if (Sparse_solve_sym(3, 6, rows, cols, H, b, delta, SPARSE_SPD) != 0) return 0;
        if (!isfinite(delta[2]) || fabs(*th+delta[2]) > x->o->max_rot_refine) return 0;
        gx += delta[0]; gy += delta[1]; *th += delta[2];
        if (hypot(delta[0],delta[1]) < 1e-6 && fabs(delta[2]) < 1e-8) break;
    }
    *tx = gx-cos(*th)*cx+sin(*th)*cy; *ty = gy-sin(*th)*cx-cos(*th)*cy;
    return 1;
}

static int apl_trial_support(AplCtx *x, int32_t best, const AsmContinuityTarget *targets,
                             size_t n_targets, size_t *support, int32_t *parent, double *support_rms, int report)
{
    AsmRun *run = x->run;
    *support = 0; *parent = -1; *support_rms = 0;
    double e2 = 0;
    if (x->cluster_kind == 0) for (size_t t = 0; t < n_targets; t++) {
        const AsmContinuityTarget *a = &targets[t];
        if (run->charts[a->parent].placement_state == ASM_PLACE_NONE) continue;
        double gx, gy; apl_pose(&run->charts[a->chart], a->u, a->v, &gx, &gy);
        double e = hypot(gx-a->x, gy-a->y);
        if (e <= 16.0) { (*support)++; *parent = a->parent; e2 += e*e; }
    }
    for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *R = &run->rels[r];
        if (!(R->continuity & ASM_CONT_SOURCE) || (R->flags & (ASM_REL_CONTACT | ASM_REL_DROPPED))) continue;
        if (!AsmRel_is_join(R) && !(R->continuity & ASM_CONT_CUT) && (x->cluster_kind != 1 || !x->seam_selected[r])) continue;
        int ca = x->orig_comp[R->a] == best, cb = x->orig_comp[R->b] == best;
        if (!ca && !cb) continue;
        const AsmChart *fixed = &run->charts[ca ? R->b : R->a];
        if (ca != cb && (fixed->placement_state == ASM_PLACE_NONE || fixed->component != x->primary)) continue;
        double rms; size_t n;
        if (!AsmContinuity_measure(run, R, &rms, &n)) {
            x->last_reason = "source_continuation_disagrees";
            if (report) fprintf(stderr, "  [place] source continuation %d-%d disagrees (%.2f vox, %zu pairs); transaction deferred\n", R->a, R->b, rms, n);
            return 0;
        }
        if (ca != cb && x->cluster_kind == 1) { *support += n; *parent = fixed->id; e2 += n*rms*rms; }
    }
    if (*support) *support_rms = sqrt(e2 / *support);
    if (*support < (x->cluster_kind == 1 ? (size_t)APL_SEAM_MIN_ANCHORS : (size_t)x->o->min_anchors))
        { x->last_reason = "trial_support_short"; return 0; }
    return 1;
}

static double apl_candidate_axis(const AplCtx *x, const AsmChart *saved, size_t n, double *rms)
{
    double sw = 0, sr = 0, sr2 = 0;
    *rms = 0;
    if (!APL_V_FROM_AXIS || !x->vm_ok || !x->vm_pin) return 0;
    for (size_t k = 0; k < n; k++) {
        const AsmChart *c = &x->run->charts[saved[k].id];
        double u, v;
        if (c->uv != saved[k].uv) {
            double cu, cv; apl_uv_centroid(c,&cu,&cv); apl_pose(c,cu,cv,&u,&v);
        } else apl_layout_centroid(x, c, c->id, 0, 0, 0, &u, &v);
        double e = x->vm_a + x->vm_b*x->s_axis[c->id] + x->vm_c*u - v;
        sw += c->area3d; sr += c->area3d*e; sr2 += c->area3d*e*e;
    }
    if (!(sw > 0)) return 0;
    double shift = sr/sw;
    *rms = sqrt(fmax(0.0, sr2/sw-shift*shift));
    return shift;
}

static int apl_candidate_shape(AplCtx *x, const AsmChart *saved, size_t n, double axis_pin)
{
    if (!n) return 0;
    if (APL_V_FROM_AXIS && x->vm_ok && x->vm_pin) {
        double vrms, dv = apl_candidate_axis(x, saved, n, &vrms);
        double anchor_shift = 0, anchor_weight = 0;
        if (x->cluster_kind == 0) for (size_t i = 0; i < x->na; i++) if (x->wts[i] > 0 && x->anchors[i].k > 0) {
            const AplAnchor *a = &x->anchors[i]; double u, v;
            apl_pose(&x->run->charts[a->chart_c], a->local_u, a->local_v, &u, &v);
            anchor_shift += x->wts[i]*(a->ty-v); anchor_weight += x->wts[i];
        }
        if (anchor_weight > 0) anchor_shift /= anchor_weight;
        if (!isfinite(vrms) || !isfinite(dv) || !isfinite(anchor_shift)) return 0;
        if (vrms > fmax(APL_V_AXIS_TOL, 1.5*x->vm_p90) ||
            fmax(fmax(fabs(dv), fabs(axis_pin)), fabs(anchor_shift)) > fmax(APL_V_AXIS_MAX_SHIFT, 2*x->vm_p90)) {
            fprintf(stderr, "  [place]   final v-axis gate refuses: scatter %.1f vox rms, residual shift %.1f, applied pin %.1f, anchor shift %.1f vox\n", vrms, dv, axis_pin, anchor_shift);
            if (x->diagnostic_axis_policy == 2 ||
                (x->diagnostic_axis_policy == 1 && x->cluster_kind == 1)) {
                fprintf(stderr, "  [place axis diagnostic] checking the remaining gates, kind %d; no production policy change\n", x->cluster_kind);
            } else {
                if (x->st) x->st->v_refused++;
                return 0;
            }
        }
    }
    double ulo = 1e300, uhi = -1e300, vlo = 1e300, vhi = -1e300, plo = 1e300, phi = -1e300;
    for (size_t k = 0; k < n; k++) {
        const AsmChart *c = &x->run->charts[saved[k].id];
        if (!isfinite(c->pose_x) || !isfinite(c->pose_y) || !isfinite(c->pose_theta)) return 0;
        ulo = fmin(ulo, c->pose_x); uhi = fmax(uhi, c->pose_x);
        vlo = fmin(vlo, c->pose_y); vhi = fmax(vhi, c->pose_y);
        plo = fmin(plo, saved[k].pose_y); phi = fmax(phi, saved[k].pose_y);
    }
    double span = fmax(uhi, x->strip_umax)-fmin(ulo, x->strip_umin);
    if (x->span_cap > 0 && span > x->span_cap) {
        fprintf(stderr, "  [place]   final span gate refuses: %.0f vox (cap %.0f)\n", span, x->span_cap);
        return 0;
    }
    double tv = 0, tw = 0;
    for (size_t i = 0; i < x->na; i++) if (x->wts[i] > 0) { tv += x->wts[i]*x->anchors[i].ty; tw += x->wts[i]; }
    if (tw > 0) {
        double extent = phi-plo+300, tvm = tv/tw;
        if (vlo > tvm+extent || vhi < tvm-extent || vhi-vlo > 2*extent+600) {
            fprintf(stderr, "  [place]   final swing gate refuses: body v %.0f..%.0f, anchors v %.0f, own extent %.0f\n", vlo, vhi, tvm, phi-plo);
            return 0;
        }
    }
    return 1;
}

static AplConsensus apl_candidate_consensus(const AplCtx *x)
{
    AplConsensus result = {0};
    if (!x->n_observations) return result;
    Arena_Mark mark = Arena_save(x->run->arena);
    double *error = ARENA_ALLOC(x->run->arena, x->n_observations*sizeof(double));
    double *model = ARENA_ALLOC(x->run->arena, x->n_observations*sizeof(double));
    size_t *ref = ARENA_ALLOC(x->run->arena, x->n_observations*sizeof(size_t));
    for (size_t i = 0; i < x->n_observations; i++) { error[i] = model[i] = 1e300; ref[i] = SIZE_MAX; }
    for (size_t i = 0; i < x->na; i++) {
        const AplAnchor *a = &x->anchors[i];
        if (a->k <= 0 || a->measured_order <= 0 || a->measured_order > a->k || a->observation >= x->n_observations) continue;
        if (x->run->charts[a->chart_p].placement_state == ASM_PLACE_NONE) continue;
        const AsmChart *c = &x->run->charts[a->chart_c]; double u, v;
        if (x->uv_trial && a->hit) {
            if (a->reverse) apl_vertex_point(c,a->hit->va,&u,&v);
            else apl_face_point(c,a->hit->fb,a->hit->l0,a->hit->l1,&u,&v);
        } else apl_pose(c,a->local_u,a->local_v,&u,&v);
        double e = hypot(u-a->tx, v-a->ty);
        double scale = fmax(16.0, 0.05*hypot(a->Tx,a->Ty)/(double)a->k);
        error[a->observation] = fmin(error[a->observation], e);
        model[a->observation] = fmin(model[a->observation], e/scale);
        ref[a->observation] = i;
    }
    for (size_t i = 0; i < x->n_observations; i++) if (error[i] < 1e299) {
        double e = error[i]/16.0, z = model[i];
        result.observations++; result.inliers += error[i] <= 16.0;
        result.model_inliers += z <= 1.0;
        result.clipped_loss += fmin(1.0, e*e);
        result.model_loss += z <= 1.0 ? 0.5*z*z : z-0.5;
    }
    if (x->fevidence) {
        typedef struct AplEvidence { size_t n, inliers; double loss; } AplEvidence;
        size_t nc = x->run->n_charts;
        AplEvidence *groups = ARENA_CALLOC(x->run->arena, 2*nc, sizeof(AplEvidence));
        int32_t component = -1;
        for (size_t i = 0; i < x->n_observations; i++) if (ref[i] != SIZE_MAX) {
            const AplAnchor *a = &x->anchors[ref[i]];
            component = x->orig_comp[a->chart_c];
            for (int side = 0; side < 2; side++) {
                AplEvidence *g = &groups[(side ? nc : 0)+(size_t)(side ? a->chart_c : a->chart_p)];
                g->n++; g->inliers += error[i] <= 16.0;
                double z = model[i]; g->loss += z <= 1.0 ? 0.5*z*z : z-0.5;
            }
        }
        for (size_t i = 0; i < 2*nc; i++) if (groups[i].n)
            fprintf(x->fevidence, "%d,%d,%d,%s,%zu,%zu,%zu,%.9g\n", x->attempt, component, x->cluster,
                    i < nc ? "parent" : "candidate", i%nc, groups[i].n, groups[i].inliers, groups[i].loss);
    }
    Arena_restore(x->run->arena, mark);
    return result;
}

static int apl_relax_boundary_trial(AplCtx *x, const AsmChart *saved, size_t n)
{
    return x->diagnostic_first_step
        ? AsmContinuity_relax_boundary_step(x->run,saved,n,x->seam_selected,apl_uv_audit,x->o->stress_band)
        : AsmContinuity_relax_boundary_trial(x->run,saved,n,x->seam_selected,apl_uv_audit,x->o->stress_band);
}

static int apl_place_and_audit_once(AplCtx *x, int32_t best, double th, double tx, double ty,
                                    double *out_ratio, int require_rim, int polish, int *did_polish)
{
    *did_polish = 0;
    if (out_ratio) *out_ratio = 0.0;
    x->last_silent = 0; x->last_shape_refused = 0; x->last_reason = "accepted";
    x->uv_trial = 0;
    if (x->proposal_uv) Arena_free(x->proposal_uv);
    int32_t parent = -1;
    size_t support = 0;
    double support_rms = 0;
    size_t min_support = x->cluster_kind == 1 ? APL_SEAM_MIN_ANCHORS : (size_t)x->o->min_anchors;
    AsmRun *run = x->run;
    Arena_T arena = run->arena;
    size_t nc = run->n_charts;
    Arena_Mark m2 = Arena_save(arena);
    size_t nsave = 0;
    for (size_t i = 0; i < nc; i++) if (AsmChart_in_layout(&run->charts[i]) && run->charts[i].component == best) nsave++;
    AsmChart *saved = ARENA_ALLOC(arena, (nsave ? nsave : 1) * sizeof(AsmChart));
    AsmContinuityTarget *targets = ARENA_ALLOC(arena, (x->na ? x->na : 1)*sizeof(AsmContinuityTarget));
    size_t n_targets = 0;
    if (x->cluster_kind == 0) for (size_t i = 0; i < x->na; i++) {
        const AplAnchor *a = &x->anchors[i];
        if (a->cluster != x->cluster || x->wts[i] <= 0 || a->k <= 0 || a->measured_order <= 0 || a->measured_order > a->k) continue;
        targets[n_targets++] = (AsmContinuityTarget){ a->chart_c, a->local_u, a->local_v, a->tx, a->ty, 0.05*x->wts[i], a->chart_p };
    }
    size_t q = 0;
    for (size_t i = 0; i < nc; i++) {
        AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != best) continue;
        saved[q++] = *c;
        apl_compose(c, th, tx, ty, x->source_flip);
        c->component = x->primary;
    }
    double axis_pin = 0;
    if (APL_V_FROM_AXIS && x->vm_ok && x->vm_pin) {
        if (x->cluster_kind == 1 && !APL_SEAM_V_PIN) { if (x->st) x->st->v_seam_kept++; }
        else {
            double vrms;
            axis_pin = apl_axis_excess(apl_candidate_axis(x, saved, nsave, &vrms), apl_axis_band(x));
            for (size_t k = 0; k < nsave; k++) run->charts[saved[k].id].pose_y += axis_pin;
            if (x->st) {
                x->st->v_pinned++;
                x->st->v_shift_max = fmax(x->st->v_shift_max, fabs(axis_pin));
            }
        }
    }
    int ok = apl_trial_support(x, best, targets, n_targets, &support, &parent, &support_rms, 0);
    size_t cand_vertices = 0;
    for (size_t k = 0; k < nsave; k++) cand_vertices += saved[k].nv;
    if (!ok && x->cluster_kind == 0 && support < (size_t)APL_HOP_REPAIR_MIN_SUPPORT && cand_vertices > (size_t)APL_HOP_REPAIR_MAX_VERTICES) {
        fprintf(stderr, "  [place] hop candidate: component %d, initial support %zu/%zu below %d; no joint repair\n", best, support, min_support, APL_HOP_REPAIR_MIN_SUPPORT);
    } else if (!ok) {
        fprintf(stderr, "  [place] joint repair: component %d, initial support %zu/%zu, %zu charts, %zu targets\n", best, support, min_support, nsave, n_targets);
        double *repair_start = ARENA_ALLOC(arena,3*(nsave ? nsave : 1)*sizeof *repair_start);
        for (size_t k = 0; k < nsave; k++) {
            const AsmChart *c = &run->charts[saved[k].id];
            repair_start[3*k] = c->pose_x; repair_start[3*k+1] = c->pose_y; repair_start[3*k+2] = c->pose_theta;
        }
        AsmChart *fit_start = NULL;
        if (x->cluster_kind == 1) {
            fit_start = ARENA_ALLOC(arena, (nsave ? nsave : 1)*sizeof(AsmChart));
            for (size_t k = 0; k < nsave; k++) fit_start[k] = run->charts[saved[k].id];
        }
        int solved = AsmContinuity_close(run, saved, nsave, x->seam_selected, targets, n_targets);
        int repair_bounded = 1;
        for (size_t k = 0; k < nsave && repair_bounded; k++) {
            const AsmChart *c = &run->charts[saved[k].id];
            if (!isfinite(c->pose_x) || !isfinite(c->pose_y) || !isfinite(c->pose_theta) ||
                hypot(c->pose_x-repair_start[3*k],c->pose_y-repair_start[3*k+1]) > 256 ||
                fabs(c->pose_theta-repair_start[3*k+2]) > .25) repair_bounded = 0;
        }
        if (!solved && x->cluster_kind != 1) x->last_reason = "repair_unsolved";
        else if (!repair_bounded) x->last_reason = "repair_unbounded";
        ok = (solved || x->cluster_kind == 1) && repair_bounded &&
             apl_trial_support(x, best, targets, n_targets, &support, &parent, &support_rms, solved);
        if (!solved && ok) fprintf(stderr,"  [place] joint solver stopped for component %d; bounded current SOURCE pose passes support, checking geometry\n",best);
        if (solved) {
            if (!ok && fit_start) {
                for (size_t k = 0; k < nsave; k++) run->charts[saved[k].id] = fit_start[k];
                fprintf(stderr, "  [place] boundary recovery: component %d, %zu charts\n", best, nsave);
                int boundary_solved = AsmContinuity_close_boundaries(run, saved, nsave, x->seam_selected);
                if (boundary_solved)
                    ok = apl_trial_support(x, best, targets, n_targets, &support, &parent, &support_rms, 1);
                if (!ok && boundary_solved) {
                    size_t bytes = 0;
                    for (size_t k = 0; k < nsave; k++) bytes += saved[k].nv*2*sizeof(float);
                    if (bytes <= 268435456 && cand_vertices > (size_t)APL_RELAX_TRIAL_MAX_VERTICES)
                        fprintf(stderr,"  [place] local boundary UV repair skipped: component %d, %zu vertices above %d\n",best,cand_vertices,APL_RELAX_TRIAL_MAX_VERTICES);
                    else if (bytes <= 268435456) {
                        for (size_t k = 0; k < nsave; k++) {
                            AsmChart *c = &run->charts[saved[k].id];
                            c->uv = ARENA_ALLOC(arena,c->nv*2*sizeof(float));
                            memcpy(c->uv,saved[k].uv,c->nv*2*sizeof(float));
                        }
                        AsmChart *previous = ARENA_ALLOC(arena,nsave*sizeof *previous);
                        float *previous_uv = ARENA_ALLOC(arena,bytes);
                        for (int uv_pass = 0; uv_pass < 3; uv_pass++) {
                            size_t offset = 0;
                            for (size_t k = 0; k < nsave; k++) {
                                previous[k] = run->charts[saved[k].id];
                                previous[k].uv = previous_uv+offset;
                                memcpy(previous[k].uv,run->charts[saved[k].id].uv,saved[k].nv*2*sizeof(float));
                                offset += saved[k].nv*2;
                            }
                            if (!apl_relax_boundary_trial(x,saved,nsave)) break;
                            int next_ok = apl_trial_support(x,best,targets,n_targets,&support,&parent,&support_rms,1);
                            int within_bound = 1;
                            if (!next_ok) {
                                within_bound = AsmContinuity_close_boundaries(run,saved,nsave,x->seam_selected);
                                if (within_bound) next_ok = apl_trial_support(x,best,targets,n_targets,&support,&parent,&support_rms,1);
                                for (size_t k = 0; k < nsave && within_bound; k++) {
                                    const AsmChart *c = &run->charts[saved[k].id], *start = &fit_start[k];
                                    if (hypot(c->pose_x-start->pose_x,c->pose_y-start->pose_y) > 256 ||
                                        fabs(c->pose_theta-start->pose_theta) > 0.25) {
                                        fprintf(stderr,"  [place] boundary repair exceeds original movement bound: chart %d\n",c->id);
                                        within_bound = 0;
                                    }
                                }
                            }
                            if (!within_bound || (ok && !next_ok)) {
                                for (size_t k = 0; k < nsave; k++) {
                                    AsmChart *c = &run->charts[saved[k].id]; float *owned_uv = c->uv;
                                    memcpy(owned_uv,previous[k].uv,c->nv*2*sizeof(float));
                                    *c = previous[k]; c->uv = owned_uv;
                                }
                                if (ok) apl_trial_support(x,best,targets,n_targets,&support,&parent,&support_rms,0);
                                break;
                            }
                            x->uv_trial = 1; ok = next_ok;
                            fprintf(stderr,"  [place] local boundary UV repair: component %d, pass %d, source constraints %s\n",best,uv_pass+1,ok ? "pass" : "fail");
                        }
                    }
                }
            }
        }
        if (ok) {
            AsmContinuityAuditCache internal = {0}; internal.scratch = arena;
            size_t internal_after, introduced = apl_internal_new(run,saved,nsave,&internal,&internal_after);
            if (introduced) {
                fprintf(stderr, "  [place] joint repair creates internal contradictions: %zu -> %zu vertices, %zu previously sound vertices violated; transaction deferred\n", internal.baseline_bad, internal_after,introduced);
                ok = 0; x->last_reason = "internal_contradiction";
            }
        }
        if (!ok) x->last_silent = 1;
    }
    if (ok && polish && x->cluster_kind == 1 && nsave && x->copts && x->cgrid) {
        size_t guaranteed, queries;
        if (aprg_polish_blocked(x->rgrid,run,saved,nsave,x->excluded_samples,x->o->max_rim_ratio,&guaranteed,&queries)) {
            polish = 0;
            fprintf(stderr,"  [place] UV polish cannot clear rim: component %d, at least %zu of %zu queries violate throughout the original movement disk; auditing original candidate\n",
                    best,guaranteed,queries);
        }
    }
    if (ok && polish && x->cluster_kind == 1 && nsave) {
        size_t bytes = 0;
        for (size_t k = 0; k < nsave; k++) bytes += saved[k].nv*2*sizeof(float);
        if (bytes <= 268435456 && cand_vertices > (size_t)APL_RELAX_TRIAL_MAX_VERTICES)
            fprintf(stderr,"  [place] supported boundary correction skipped: component %d, %zu vertices above %d\n",best,cand_vertices,APL_RELAX_TRIAL_MAX_VERTICES);
        else if (bytes <= 268435456) {
            AsmChart *incumbent = ARENA_ALLOC(arena,nsave*sizeof *incumbent);
            for (size_t k = 0; k < nsave; k++) {
                AsmChart *c = &run->charts[saved[k].id]; incumbent[k] = *c;
                c->uv = ARENA_ALLOC(arena,c->nv*2*sizeof(float));
                memcpy(c->uv,incumbent[k].uv,c->nv*2*sizeof(float));
            }
            for (int pass = 0; pass < 3; pass++) {
                if (!apl_relax_boundary_trial(x,saved,nsave)) break;
                *did_polish = 1; x->uv_trial = 1;
            }
            if (*did_polish) {
                ok = apl_trial_support(x,best,targets,n_targets,&support,&parent,&support_rms,1);
                fprintf(stderr,"  [place] supported boundary correction: component %d, source constraints %s, RMS %.3f\n",best,ok ? "pass" : "fail",support_rms);
            } else for (size_t k = 0; k < nsave; k++) run->charts[saved[k].id] = incumbent[k];
        }
    }
    if (ok && !AsmContinuity_preserves_incumbent_trial(run,saved,nsave)) { ok = 0; x->last_reason = "incumbent_changed"; }
    if (ok && !apl_candidate_shape(x, saved, nsave, axis_pin)) { ok = 0; x->last_shape_refused = 1; x->last_reason = "shape_refused"; }
    if (ok) {
        x->consensus = apl_candidate_consensus(x);
        if (x->cluster_kind == 0 && x->n_observations &&
            x->consensus.model_inliers <= x->consensus.observations/2) {
            ok = 0; x->last_silent = 1; x->last_reason = "consensus_minority";
            fprintf(stderr,"  [place] component %d: hop model agrees with only %zu/%zu unique incident observations; deferred\n",
                    best,x->consensus.model_inliers,x->consensus.observations);
        }
    }
    int source_supported = ok && x->cluster_kind == 1;
    if (ok && x->copts && x->cgrid != NULL) {
        const AsmChart **cand = ARENA_ALLOC(arena, (nsave ? nsave : 1) * sizeof(AsmChart *));
        for (size_t k = 0; k < nsave; k++) cand[k] = &run->charts[saved[k].id];
        int32_t p1 = -1; double m1 = 0.0;
        double tp0 = ves_clock_sec();
        double m = AsmConflictGrid_probe_excluding(x->cgrid, cand, nsave, x->excluded_samples, &p1, &m1);
        double tp1 = ves_clock_sec();
        double ratio = x->comp_area[best] > 0.0 ? m / x->comp_area[best] : 0.0;
        if (out_ratio) *out_ratio = ratio;
        if (ratio > x->o->max_contra_ratio) {
            ok = 0; x->last_reason = "contradiction_mass";
            fprintf(stderr, "  [place]   audit rejects: contradiction mass %.3e = %.2f of its area; partner chart %d (component %d, %.2e)\n",
                    m, ratio, p1, p1 >= 0 ? x->orig_comp[p1] : -1, m1);
        }
        if (ok) {
            size_t n_rim = 0;
            double bad = aprg_probe_excluding(x->rgrid, run, cand, nsave, 24.0, &n_rim, x->excluded_samples);
            if (n_rim >= 8 && bad > x->o->max_rim_ratio) {
                ok = 0; x->last_reason = "rim_isometry";
                apl_write_rim_failure(x,best,cand,nsave);
                fprintf(stderr, "  [place]   rim audit rejects: %.0f%% of %zu rim vertices are farther in 3-D than in (u,v) by more than half a layer\n", 100.0 * bad, n_rim);
            } else if (require_rim == 1 && (n_rim < 8 || bad > 0.1)) {
                ok = 0; x->last_reason = "rim_unconfirmed";
                fprintf(stderr, "  [place]   rim audit does not confirm the retry: %zu rim vertices, %.0f%% violating\n", n_rim, 100.0 * bad);
            } else if (support < min_support) {
                ok = 0; x->last_silent = 1; x->last_reason = "insufficient_support";
                fprintf(stderr, "  [place]   insufficient positive support: %zu measured pairs, %zu rim vertices; deferred\n", support, n_rim);
            } else if (n_rim >= 8 && bad > 0.0)
                fprintf(stderr, "  [place]   rim audit: %.0f%% of %zu rim vertices violate\n", 100.0 * bad, n_rim);
        }
        double tp2 = ves_clock_sec();
        if (ok) {
            for (size_t k = 0; k < nsave; k++) {
                AsmChart *c = &run->charts[saved[k].id];
                c->placement_state = x->cluster_kind == 1 ? ASM_PLACE_SEAM : ASM_PLACE_LAYER;
                c->placement_parent = parent; c->placement_support = support;
                c->placement_rms = support_rms;
            }
            if (!x->trial) {
                AsmContinuity_confirm(run);
                for (size_t k = 0; k < nsave; k++) { AsmConflictGrid_insert(x->cgrid, cand[k]); aprg_insert_chart(x->rgrid, cand[k]); }
            }
        }
        double tp3 = ves_clock_sec();
        if (x->st) { x->st->t_probe += tp1 - tp0; x->st->t_rim += tp2 - tp1; x->st->t_insert += tp3 - tp2; }
        fprintf(stderr, "  [place]   audit timing: conflict probe %.2f s, rim probe %.2f s, insert %.2f s (grid cells %zu)\n",
                tp1 - tp0, tp2 - tp1, tp3 - tp2, AsmConflictGrid_cells(x->cgrid));
    }
    if (!ok && source_supported) x->source_geometry_conflict = 1;
    if (ok && !x->cgrid) {
        ok = support >= min_support;
        if (!ok) { x->last_silent = 1; x->last_reason = "insufficient_support"; }
        else for (size_t k = 0; k < nsave; k++) {
            AsmChart *c = &run->charts[saved[k].id];
            c->placement_state = x->cluster_kind == 1 ? ASM_PLACE_SEAM : ASM_PLACE_LAYER;
            c->placement_parent = parent; c->placement_support = support;
            c->placement_rms = support_rms;
        }
    }
    if (ok && x->trial) {
        x->nproposal = nsave;
        for (size_t k = 0; k < nsave; k++) {
            x->proposal[k] = run->charts[saved[k].id];
            if (x->proposal[k].uv != saved[k].uv) {
                if (!x->proposal_uv) x->proposal_uv = Arena_new();
                x->proposal[k].uv = ARENA_ALLOC(x->proposal_uv,saved[k].nv*2*sizeof(float));
                memcpy(x->proposal[k].uv,run->charts[saved[k].id].uv,saved[k].nv*2*sizeof(float));
            }
        }
    }
    if (ok && !x->trial) for (size_t k = 0; k < nsave; k++) apl_commit_chart(x,&run->charts[saved[k].id],saved[k].uv);
    if (!ok || x->trial) for (size_t k = 0; k < nsave; k++) run->charts[saved[k].id] = saved[k];
    Arena_restore(arena, m2);
    return ok;
}

static int apl_audit_proposal(AplCtx *x, int32_t best, double th, double tx, double ty, double *out_ratio, int require_rim)
{
    if (x->cluster_kind == 0 && x->source_geometry_conflict) {
        if (out_ratio) *out_ratio = 0;
        x->last_silent = 1; x->last_shape_refused = 0; x->last_reason = "source_geometry_conflict";
        fprintf(stderr,"  [place] component %d: supported boundary fit has an unresolved geometry conflict; alternate hop deferred\n",best);
        return 0;
    }
    int refined = 0;
    int ok = apl_place_and_audit_once(x,best,th,tx,ty,out_ratio,require_rim,1,&refined);
    if (!ok && refined) {
        fprintf(stderr,"  [place] boundary correction failed geometry audit; retrying original supported candidate\n");
        ok = apl_place_and_audit_once(x,best,th,tx,ty,out_ratio,require_rim,0,&refined);
    }
    return ok;
}

static int apl_place_and_audit(AplCtx *x, int32_t best, double th, double tx, double ty, double *out_ratio, int require_rim)
{
    double closed_th = th, closed_tx = tx, closed_ty = ty;
    if (apl_close_seam(x,&closed_th,&closed_tx,&closed_ty)) {
        if (apl_audit_proposal(x,best,closed_th,closed_tx,closed_ty,out_ratio,require_rim)) return 1;
        if (closed_th == th && closed_tx == tx && closed_ty == ty) return 0;
        fprintf(stderr,"  [place] seam closure failed audit for component %d; checking original proposal\n",best);
    } else fprintf(stderr,"  [place] seam closure unavailable for component %d; checking original proposal\n",best);
    return apl_audit_proposal(x,best,th,tx,ty,out_ratio,require_rim);
}

static int apl_reconsider_hop(AplCtx *x, int32_t best, double deadline)
{
    AsmRun *run = x->run;
    if (best == x->primary || x->comp_placed[best] != 1) return 0;
    size_t n = 0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || x->orig_comp[i] != best) continue;
        if (c->component != x->primary || c->placement_state != ASM_PLACE_LAYER) return 0;
        n++;
    }
    if (!n) return 0;
    Arena_Mark mark = Arena_save(run->arena);
    size_t ng = 0;
    for (size_t i = 0; i < run->n_charts; i++) if (x->orig_comp[i] >= 0 && (size_t)x->orig_comp[i] >= ng) ng = (size_t)x->orig_comp[i]+1;
    int32_t *first = ARENA_ALLOC(run->arena,ng*sizeof *first);
    for (size_t i = 0; i < ng; i++) first[i] = -1;
    UnionFind bundle = UF_new(run->arena,(int32_t)run->n_charts);
    for (size_t i = 0; i < run->n_charts; i++) {
        int32_t g = x->orig_comp[i];
        if (g < 0 || !AsmChart_in_layout(&run->charts[i])) continue;
        if (first[g] < 0) first[g] = (int32_t)i; else uf_union(&bundle,first[g],(int32_t)i);
    }
    for (size_t i = 0; i < run->n_rels; i++) {
        const AsmRelation *r = &run->rels[i];
        const AsmChart *a = &run->charts[r->a], *b = &run->charts[r->b];
        if (!AsmChart_in_layout(a) || !AsmChart_in_layout(b) ||
            a->component != x->primary || b->component != x->primary ||
            a->placement_state == ASM_PLACE_NONE || b->placement_state == ASM_PLACE_NONE) continue;
        double rms; size_t support;
        if (AsmContinuity_measure(run,r,&rms,&support)) uf_union(&bundle,r->a,r->b);
    }
    uint8_t *excluded = ARENA_CALLOC(run->arena,run->n_charts,1);
    uint8_t *was_hop = ARENA_CALLOC(run->arena,ng,1);
    int32_t root = uf_find(&bundle,first[best]);
    double bundle_area = 0; n = 0;
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || uf_find(&bundle,(int32_t)i) != root) continue;
        if (c->component != x->primary || c->placement_state == ASM_PLACE_NONE || c->placement_state == ASM_PLACE_ROOT) {
            Arena_restore(run->arena,mark); return 0;
        }
        excluded[i] = 1; n++; bundle_area += c->area3d;
        if (c->placement_state == ASM_PLACE_LAYER) was_hop[x->orig_comp[i]] = 1;
    }
    memset(x->seam_selected,0,run->n_rels);
    size_t broken = 0;
    for (size_t i = 0; i < run->n_rels; i++) {
        const AsmRelation *r = &run->rels[i];
        if (!(r->continuity & ASM_CONT_SOURCE) || (r->flags & (ASM_REL_DROPPED | ASM_REL_CONTACT))) continue;
        int a = excluded[r->a], b = excluded[r->b];
        if (!a && !b) continue;
        const AsmChart *A = &run->charts[r->a], *B = &run->charts[r->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B) ||
            A->placement_state == ASM_PLACE_NONE || B->placement_state == ASM_PLACE_NONE ||
            A->component != x->primary || B->component != x->primary) continue;
        x->seam_selected[i] = 1;
        double rms; size_t support;
        if (a != b && !AsmContinuity_measure(run,r,&rms,&support)) broken++;
    }
    if (!broken) { Arena_restore(run->arena,mark); return 0; }
    AsmChart *saved = ARENA_ALLOC(run->arena,n*sizeof *saved);
    AsmChart *winner = ARENA_ALLOC(run->arena,n*sizeof *winner);
    size_t q = 0;
    for (size_t i = 0; i < run->n_charts; i++) {
        AsmChart *c = &run->charts[i];
        if (excluded[i]) saved[q++] = *c;
    }
    UnionFind uf = UF_new(run->arena,(int32_t)run->n_charts);
    for (size_t i = 0; i < run->n_rels; i++) if (x->seam_selected[i]) uf_union(&uf,run->rels[i].a,run->rels[i].b);
    uint8_t *anchored = ARENA_CALLOC(run->arena,run->n_charts,1);
    for (size_t i = 0; i < run->n_charts; i++) if (!excluded[i] && AsmChart_in_layout(&run->charts[i]) &&
        run->charts[i].component == x->primary && run->charts[i].placement_state != ASM_PLACE_NONE)
        anchored[uf_find(&uf,(int32_t)i)] = 1;
    for (size_t i = 0; i < n; i++) if (!anchored[uf_find(&uf,saved[i].id)]) {
        Arena_restore(run->arena,mark); return 0;
    }
    int32_t *original_groups = x->orig_comp;
    int32_t *trial_groups = ARENA_ALLOC(run->arena,run->n_charts*sizeof *trial_groups);
    memcpy(trial_groups,original_groups,run->n_charts*sizeof *trial_groups);
    double original_area = x->comp_area[best];
    x->orig_comp = trial_groups; x->comp_area[best] = bundle_area;
    for (size_t i = 0; i < n; i++) {
        AsmChart *c = &run->charts[saved[i].id]; c->component = best; c->placement_state = ASM_PLACE_NONE;
        trial_groups[c->id] = best;
    }
    x->comp_placed[best] = 0; x->excluded_samples = excluded;
    x->trial = 1; x->stretch_ok = 0; x->attempt++;
    x->na = apl_collect_anchors(x,best);
    double weight[APL_MAX_CLUSTERS_TOTAL], u[APL_MAX_CLUSTERS_TOTAL], v[APL_MAX_CLUSTERS_TOTAL];
    int kind[APL_MAX_CLUSTERS_TOTAL];
    int ncl = apl_cluster_anchors(x,x->na,weight,u,v,kind);
    double *res = ARENA_ALLOC(run->arena,(x->na ? x->na : 1)*sizeof *res);
    Arena_T winner_uv = NULL;
    size_t winner_n = 0; int ambiguous = 0, exhausted = 0;
    fprintf(stderr,"  [place] reconsider hop: component %d, %zu charts, %zu broken source seams\n",best,n,broken);
    for (int cl = 0; cl < ncl; cl++) {
        if (deadline > 0 && ves_clock_sec() > deadline) { exhausted = 1; break; }
        if (kind[cl] != 1) continue;
        x->source_flip = 0;
        for (size_t i = 0; i < x->na; i++) if (x->anchors[i].cluster == cl && x->anchors[i].k == 0) {
            x->source_flip = x->anchors[i].source_flip; break;
        }
        double th, tx, ty; size_t kept;
        double rms = apl_fit_cluster(x,x->na,cl,APL_SEAM_MIN_ANCHORS,x->o->max_rot_refine,&th,&tx,&ty,&kept,res);
        if (rms >= 1e299) continue;
        x->cluster = cl; x->cluster_kind = 1;
        if (!apl_place_and_audit(x,best,th,tx,ty,NULL,2)) continue;
        if (!winner_n) {
            winner_n = x->nproposal;
            apl_copy_candidate(winner,x->proposal,winner_n,run,&winner_uv);
        } else {
            if (winner_n != x->nproposal) ambiguous = 1;
            else for (size_t k = 0; k < winner_n; k++)
                if (!apl_same_pose(&winner[k],&x->proposal[k],8)) { ambiguous = 1; break; }
        }
    }
    if (deadline > 0 && ves_clock_sec() > deadline) exhausted = 1;
    for (size_t i = 0; i < n; i++) run->charts[saved[i].id] = saved[i];
    x->orig_comp = original_groups; x->comp_area[best] = original_area;
    x->comp_placed[best] = 1; x->excluded_samples = NULL; x->source_flip = 0;
    int accepted = winner_n && !ambiguous && !exhausted;
    if (accepted) {
        for (size_t i = 0; i < winner_n; i++) {
            winner[i].placement_group = run->charts[winner[i].id].placement_group;
            apl_commit_chart(x,&winner[i],run->charts[winner[i].id].uv);
            if (x->curv_ok) x->curv_ok[winner[i].id] = 0;
        }
        AsmContinuity_confirm(run);
        for (size_t g = 0; g < ng; g++) if (was_hop[g]) {
            if (x->diag) x->diag[g].placed_by = 1;
            if (x->st) { x->st->seam_placed++; if (x->st->hop_placed) x->st->hop_placed--; }
        }
        fprintf(stderr,"  [place] source recovery: component %d closes %zu previously broken seams, %zu charts, %zu witnesses at %.3f vox RMS\n",
                best,broken,winner_n,winner[0].placement_support,winner[0].placement_rms);
    } else fprintf(stderr,"  [place] reconsidered hop kept unchanged: component %d%s\n",best,ambiguous ? " (ambiguous)" : "");
    if (exhausted && x->st) x->st->budget_exhausted = 1;
    if (winner_uv) Arena_dispose(&winner_uv);
    Arena_restore(run->arena,mark);
    return accepted;
}

static void apl_rebuild_grids(AplCtx *x, Arena_T *arena)
{
    if (!x->copts || !arena || !*arena) return;
    AsmConflictGrid_dispose(x->cgrid); Arena_dispose(arena); *arena = Arena_new();
    x->cgrid = AsmConflictGrid_new(*arena,x->copts,x->run);
    x->rgrid = aprg_new(*arena,24);
    x->strip_umin = 1e300; x->strip_umax = -1e300;
    for (size_t i = 0; i < x->run->n_charts; i++) {
        const AsmChart *c = &x->run->charts[i];
        if (!AsmChart_in_layout(c) || c->component != x->primary || c->placement_state == ASM_PLACE_NONE) continue;
        AsmConflictGrid_insert(x->cgrid,c); aprg_insert_chart(x->rgrid,c);
        x->strip_umin = fmin(x->strip_umin,c->pose_x); x->strip_umax = fmax(x->strip_umax,c->pose_x);
    }
}

static void apl_profile_line(const AsmPlaceStats *st, size_t sweeps, double total, const char *prefix)
{
    double gates = st->t_clusterloop - st->t_fit - st->t_audit;
    double known = st->t_measure + st->t_growth + st->t_setup + st->t_count + st->t_mirror + st->t_stretch + st->t_collect + st->t_cluster + st->t_clusterloop + st->t_cut;
    fprintf(stderr, "%s: %zu sweeps in %.1f s (%.1f/s) | measure turns %.1f s (%zu re-measurements) growth %.1f setup %.1f candidates %.1f mirror-fit %.1f stretch %.1f (quadric %zu fitted, %zu cached) collect %.1f cluster %.1f fit %.1f gates %.1f audit %.1f (probe %.1f rim %.1f insert %.1f) cut %.1f other %.1f\n",
            prefix, sweeps, total, total > 0.0 ? (double)sweeps / total : 0.0, st->t_measure, st->remeasures, st->t_growth, st->t_setup, st->t_count, st->t_mirror, st->t_stretch,
            st->quadric_calls, st->quadric_cached, st->t_collect, st->t_cluster, st->t_fit, gates < 0.0 ? 0.0 : gates, st->t_audit, st->t_probe, st->t_rim, st->t_insert, st->t_cut, total - known);
}

static int32_t apl_next_component(int32_t ncomp, const uint8_t *placed,
                                  const int32_t *anchors, const int32_t *seams,
                                  const int32_t *tried, const int32_t *ring,
                                  const double *area, int min_anchors)
{
    int32_t best = -1;
    for (int32_t c = 0; c < ncomp; c++) {
        int has_seam = APL_SEAM_FIRST && seams[c] >= APL_SEAM_MIN_ANCHORS;
        if (placed[c] == 1 || (anchors[c] < min_anchors && !has_seam)) continue;
        if (placed[c] == 2 && anchors[c] <= tried[c]) continue;
        if (best < 0) { best = c; continue; }
        int best_seam = APL_SEAM_FIRST && seams[best] >= APL_SEAM_MIN_ANCHORS;
        if (APL_SEAM_FRONTIER_FIRST && (APL_SEAM_FRONTIER_FIRST == 1 || ring[c] == ring[best]) && has_seam != best_seam) {
            if (has_seam) best = c;
            continue;
        }
        if (ring[c] < ring[best] || (ring[c] == ring[best] && area[c] > area[best])) best = c;
    }
    return best;
}

static void apl_record_poses(const AsmRun *run, const AsmPlaceOpts *o, const char *stage, int append)
{
    if (!o->diag_dir) return;
    char path[2048]; snprintf(path, sizeof path, "%s/stage5_poses.csv", o->diag_dir);
    FILE *fp = fopen(path, append ? "ab" : "wb"); if (!fp) return;
    if (!append) fputs("stage,chart,component,u,v,theta,mirror\n", fp);
    for (size_t i = 0; i < run->n_charts; i++) {
        const AsmChart *c = &run->charts[i]; if (!AsmChart_in_layout(c)) continue;
        fprintf(fp, "%s,%zu,%d,%.9g,%.9g,%.12g,%d\n", stage, i, c->component,
                c->pose_x, c->pose_y, c->pose_theta, (c->flags & ASM_CHART_MIRROR) != 0);
    }
    fclose(fp);
    if (!run->continuity_ready) return;
    snprintf(path, sizeof path, "%s/stage5_source_residuals.csv", o->diag_dir);
    int first = strcmp(stage, "registered_rigid") == 0;
    fp = fopen(path, first ? "wb" : "ab"); if (!fp) return;
    if (first) fputs("stage,chart_a,chart_b,component_a,component_b,flags,source,passes,pairs,rms\n", fp);
    for (size_t i = 0; i < run->n_rels; i++) {
        const AsmRelation *r = &run->rels[i];
        if (!AsmChart_in_layout(&run->charts[r->a]) || !AsmChart_in_layout(&run->charts[r->b])) continue;
        double rms; size_t support; int passes = AsmContinuity_measure(run, r, &rms, &support);
        fprintf(fp, "%s,%d,%d,%d,%d,%u,%d,%d,%zu,%.9g\n", stage, r->a, r->b,
                run->charts[r->a].component, run->charts[r->b].component, r->flags,
                (r->continuity & ASM_CONT_SOURCE) != 0, passes, support, rms);
    }
    fclose(fp);
}

#include "asm_place_strict.inc"

static int apl_retry_candidate_qualified(const AplCtx *x,int32_t component)
{
    Arena_T arena=Arena_new();AsmRun trial=*x->run;trial.arena=arena;
    trial.charts=ARENA_ALLOC(arena,trial.n_charts*sizeof(AsmChart));
    memcpy(trial.charts,x->run->charts,trial.n_charts*sizeof(AsmChart));
    for(size_t k=0;k<x->nproposal;k++)trial.charts[x->proposal[k].id]=x->proposal[k];
    AsmField *f=AsmField_new(arena,&trial);int okay=f!=NULL;
    size_t bad_charts=0,failed_seams=0;AsmContactStats contacts={0};
    if(f){
        uint8_t *active=ARENA_CALLOC(arena,f->nf,1);
        for(size_t c=0;c<trial.n_charts;c++)if(x->orig_comp[c]==component){
            if(f->face_offset[c]==SIZE_MAX){bad_charts++;continue;}
            double area=0,good=0;int valid=1;
            for(size_t k=0;k<trial.charts[c].nf;k++){
                size_t t=f->face_offset[c]+k;active[t]=1;double j[4],lo,hi,det;
                AsmField_face(f,t,f->uv,j,&lo,&hi,&det);
                if(!UvGuard_positive(f->uv,f->faces+3*t) || !AsmMetric_within25(lo,hi))valid=0;
                area+=f->metric[t].area;if(AsmMetric_within10(lo,hi))good+=f->metric[t].area;
            }
            bad_charts+=!valid || !(area>0) || good<.95*area-32*DBL_EPSILON*area;
        }
        for(size_t i=0;i<f->n_seams;i++){
            const AsmFieldSeam *s=f->seams+i;const AsmRelation *r=trial.rels+s->relation;
            if((r->flags&ASM_REL_CONTACT) || !(r->continuity&(ASM_CONT_SOURCE|ASM_CONT_CUT)) ||
                (x->orig_comp[s->a]!=component && x->orig_comp[s->b]!=component))continue;
            AsmSeamMeasure m;AsmField_seam(f,i,f->uv,&m);failed_seams+=!m.pass;
        }
        AsmContacts_active_faces(f->contacts,active);
        okay=!AsmContacts_measure(f->contacts,f->uv,NULL,NULL,NULL,&contacts) &&
             !bad_charts && !failed_seams && !contacts.pairs;
    }
    fprintf(stderr,"  [place retry] original field: %zu bad charts, %zu failed source seams, %zu affected overlap pairs; %s\n",
            bad_charts,failed_seams,contacts.pairs,okay?"qualified":"refused");
    Arena_dispose(&arena);return okay;
}

static void apl_write_candidate(FILE *fp, const AplCtx *x, int32_t best, double area, int cl, int kind,
                                int accepted, const char *reason, size_t kept, double rms, double gate,
                                size_t k1, size_t k2, size_t est, double mean_turn, double turn_rel, int measured)
{
    if (!fp) return;
    int chart = -1, mirror = 0;
    double u = 0.0, v = 0.0, srms = 0.0;
    size_t nprop = 0, support = 0;
    if (accepted && x->nproposal) {
        const AsmChart *c = &x->proposal[0];
        double cu, cv;
        apl_uv_centroid(c, &cu, &cv); apl_pose(c, cu, cv, &u, &v);
        chart = c->id; mirror = (c->flags & ASM_CHART_MIRROR) != 0;
        nprop = x->nproposal; support = c->placement_support; srms = c->placement_rms;
    }
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%zu,%zu,%.6g,%zu,%zu,%zu,%.9g,%.9g,%d,%.6f,%.6f,%d,%zu,%.6g,%.6g,%zu,%zu,%zu,%.6g,%.6g,%.6e\n",
            x->attempt, best, cl, kind, accepted, reason, nprop, support, srms,
            measured ? x->consensus.observations : (size_t)0,
            measured ? x->consensus.inliers : (size_t)0,
            measured ? x->consensus.model_inliers : (size_t)0,
            measured ? x->consensus.clipped_loss : 0.0,
            measured ? x->consensus.model_loss : 0.0,
            chart, u, v, mirror, kept, rms, gate, k1, k2, est, mean_turn, turn_rel, area);
}

static int apl_preamble(AsmRun *run, const AsmPlaceOpts *o)
{
    apl_record_poses(run, o, "clean2", 0);
    if (AsmContinuity_prepare(run) != 0) return -1;
    Arena_T registration_arena = Arena_new();
    AsmChart *registration_entry = ARENA_ALLOC(registration_arena,(run->n_charts ? run->n_charts : 1)*sizeof *registration_entry);
    memcpy(registration_entry,run->charts,run->n_charts*sizeof *registration_entry);
    AsmContinuity_register_components(run,apl_registration_audit);
    apl_record_poses(run, o, "registered_rigid", 1);
    AsmContinuity_relax_uv(run, apl_uv_audit, o->stress_band, o->diag_dir);
    AsmContinuity_register_failed(run,registration_entry,apl_registration_audit);
    Arena_dispose(&registration_arena);
    apl_record_poses(run, o, "registered", 1);
    AsmContinuity_components(run);
    return 0;
}

static int apl_run(AsmRun *run, const AsmPlaceOpts *o, const AsmConflictOpts *copts, AsmPlaceStats *st,
                   const uint8_t *retained, int diagnostic_policy, double diagnostic_seconds)
{
    memset(st, 0, sizeof *st);
    if (!retained && apl_preamble(run, o) != 0) return -1;
    double t0 = ves_clock_sec();
    Arena_T arena = run->arena;
    Arena_Mark mark = Arena_save(arena);
    size_t nc = run->n_charts;
    for (size_t i = 0; i < nc; i++) {
        if (retained && retained[i]) continue;
        run->charts[i].placement_group = run->charts[i].component;
        run->charts[i].placement_state = ASM_PLACE_NONE;
        run->charts[i].placement_parent = -1;
        run->charts[i].placement_support = 0;
        run->charts[i].placement_rms = 0;
    }
    int32_t ncomp = 0;
    for (size_t i = 0; i < nc; i++) if (AsmChart_in_layout(&run->charts[i]) && run->charts[i].component + 1 > ncomp) ncomp = run->charts[i].component + 1;
    if (ncomp == 0) { Arena_restore(arena, mark); return 0; }
    size_t ncomp_cap = (size_t)ncomp + APL_MAX_CUTS + 8;
    double *comp_area = ARENA_CALLOC(arena, ncomp_cap, sizeof(double));
    for (size_t i = 0; i < nc; i++) {
        const AsmChart *c = &run->charts[i];
        if (!AsmChart_in_layout(c) || c->component < 0) continue;
        comp_area[c->component] += c->area3d;
        st->area_in += c->area3d;
    }
    int32_t primary = 0;
    if (!retained) for (int32_t c = 1; c < ncomp; c++) if (comp_area[c] > comp_area[primary]) primary = c;
    st->components_in = (size_t)ncomp;

    uint8_t *comp_placed = ARENA_CALLOC(arena, ncomp_cap, sizeof(uint8_t));
    AplTurn *turn = ARENA_ALLOC(arena, nc * 2 * sizeof(AplTurn));
    int8_t *out_side = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(int8_t));
    AplCompDiag *diag = ARENA_CALLOC(arena, ncomp_cap, sizeof(AplCompDiag));
    FILE *fanch = NULL, *fcandidate = NULL, *fevidence = NULL, *frim = NULL, *fcut = NULL;
    if (o->diag_dir) {
        char path[2048];
        snprintf(path, sizeof path, "%s/stage5_anchors.csv", o->diag_dir);
        fanch = fopen(path, "wb");
        if (fanch) fputs("attempt,component,area,cluster,chart_p,chart_c,side,k,turn_source,reverse,Tx,Ty,sx,sy,tx,ty,residual,kept\n", fanch);
        snprintf(path, sizeof path, "%s/stage5_candidates.csv", o->diag_dir);
        fcandidate = fopen(path, "wb");
        if (fcandidate) fputs("attempt,component,cluster,kind,accepted,reason,charts,selected_support,selected_rms,"
                              "observations,all_inliers,model_inliers,clipped_loss,model_loss,chart,u_mean,v_mean,mirror,"
                              "anchors,fit_rms,gate,k1,k2,estimated_turns,mean_turn,turn_rel,area\n", fcandidate);
        snprintf(path, sizeof path, "%s/stage5_evidence.csv", o->diag_dir);
        fevidence = fopen(path, "wb");
        if (fevidence) fputs("attempt,component,cluster,side,chart,observations,inliers,model_loss\n", fevidence);
        snprintf(path, sizeof path, "%s/stage5_cut_proposals.csv", o->diag_dir);
        fcut = fopen(path, "wb");
        if (fcut) fputs("attempt,component,outcome,c1,c2,w1,w2,rms1,rms2,u1,u2,sep,min_sep,t0,"
                        "n1,n2,m1,m2,area1,area2,crossing,crossing_sound,sound_min_rms,sound_max_support,first_sound_rel\n", fcut);
        snprintf(path, sizeof path, "%s/stage5_rim_failures.csv", o->diag_dir);
        frim = fopen(path, "wb");
        if (frim) fputs("attempt,component,cluster,uv_trial,chart,vertex,partner,sample,partner_state,u,v,partner_u,partner_v,z,y,x,partner_z,partner_y,partner_x,uv_distance,physical_distance,tolerance\n",frim);
    }

    AplOrient *ori = ARENA_CALLOC(arena, ncomp_cap, sizeof(AplOrient));
    double chirality = 0.0;
    for (int32_t c = 0; c < ncomp; c++) {
        if (comp_area[c] <= 0.0) continue;
        if(retained && c==primary) apl_orient_measure(run,c,&ori[c]);
        else apl_orient_rotate(run, c, &ori[c]);
        if (!retained && ori[c].grad >= APL_SHEET_GRAD && (ori[primary].grad < APL_SHEET_GRAD || comp_area[c] > comp_area[primary])) primary = c;
        apl_component_out_sides(run, c, o->axis, out_side);
        comp_placed[c] = 1;
        apl_measure_turns(run, comp_placed, out_side, NULL, turn, o->min_turn_hits);
        comp_placed[c] = 0;
        apl_orient_votes(run, c, o->axis, turn, &ori[c], out_side);
        if (ori[c].has_turn) chirality += comp_area[c] * ori[c].bend * ori[c].turn_u;
    }
    for (int32_t c = 0; c < ncomp; c++) {
        if (comp_area[c] <= 0.0) continue;
        if (ori[c].has_turn && fabs(ori[c].turn_u) < 0.6) ori[c].has_turn = 0;
        int mirror = ori[c].has_turn ? (ori[c].turn_u < 0.0) : (chirality != 0.0 && chirality * ori[c].bend < 0.0);
        if(retained && c==primary) mirror=0;
        if (mirror) { apl_mirror_component(run, c); ori[c].mirrored = 1; st->oriented_mirrored++; }
    }
    st->area_primary = comp_area[primary];
    fprintf(stderr, "  [place] primary component %d oriented: rotation %.3f rad, z-gradient %.3f, bend %+.3f, outward turn along u %+.3f%s%s; scroll chirality %+.0f\n",
            primary, ori[primary].theta, ori[primary].grad, ori[primary].bend, ori[primary].turn_u,
            ori[primary].has_turn ? "" : " (unmeasured)", ori[primary].mirrored ? ", mirrored" : "",
            chirality > 0.0 ? 1.0 : chirality < 0.0 ? -1.0 : 0.0);
    {
        size_t n_os = 0;
        for (size_t c = 0; c < nc; c++) n_os += out_side[c] != 0;
        fprintf(stderr, "  [place] outward side known for %zu of %zu charts\n", n_os, nc);
    }

    int32_t *orig_comp = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(int32_t));
    for (size_t i = 0; i < nc; i++) orig_comp[i] = run->charts[i].component;
    if (o->diag_dir) {
        char path[2048];
        snprintf(path, sizeof path, "%s/stage5_layers.csv", o->diag_dir);
        FILE *fl = fopen(path, "wb");
        if (fl) {
            fputs("chart_a,chart_b,comp_a,comp_b,side,out_side_a,count,d_median,hit_count,area_a,area_b,k,nsign\n", fl);
            for (size_t p = 0; p < run->n_layers; p++) {
                const AsmLayerPair *lp = &run->layers[p];
                const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
                fprintf(fl, "%d,%d,%d,%d,%d,%d,%d,%.2f,%d,%.4e,%.4e,%d,%d\n", lp->a, lp->b, A->component, B->component, lp->side, out_side[lp->a],
                        lp->count, lp->d_median, lp->hit_count, A->area3d, B->area3d, (int)lp->k, (int)lp->nsign);
            }
            fclose(fl);
        }
    }

    comp_placed[primary] = 1;
    for (size_t i = 0; i < nc; i++) if ((!retained || !retained[i]) && AsmChart_in_layout(&run->charts[i]) && run->charts[i].component == primary)
        run->charts[i].placement_state = ASM_PLACE_ROOT;
    diag[primary].status = APL_PRIMARY;
    st->placed = 1;
    st->area_placed = comp_area[primary];
    double *rms_list = ARENA_ALLOC(arena, ncomp_cap * sizeof(double));
    size_t n_rms = 0;
    size_t *incident = ARENA_CALLOC(arena, ncomp_cap, sizeof(size_t));
    size_t hop_count = (size_t)(o->max_layers > 0 ? o->max_layers : 1);
    for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *R = &run->rels[r];
        int32_t a = run->charts[R->a].component, b = run->charts[R->b].component;
        if (R->corr_count <= 0) continue;
        if (a >= 0) incident[a] += (size_t)R->corr_count;
        if (b >= 0 && b != a) incident[b] += (size_t)R->corr_count;
    }
    for (size_t p = 0; p < run->n_layers; p++) {
        const AsmLayerPair *L = &run->layers[p];
        int32_t a = run->charts[L->a].component, b = run->charts[L->b].component;
        if (L->hit_count <= 0) continue;
        size_t n = (size_t)L->hit_count * hop_count;
        if (a >= 0) incident[a] += n;
        if (b >= 0 && b != a) incident[b] += n;
    }
    size_t hcap = 1;
    for (int32_t c = 0; c < ncomp; c++) if (c != primary && incident[c] > hcap) hcap = incident[c];
    fprintf(stderr, "  [place] candidate scratch: %zu anchors (%.1f MiB)\n", hcap, (double)hcap*(sizeof(AplAnchor)+2*sizeof(double))/(1024.0*1024.0));
    AplCtx x;
    memset(&x, 0, sizeof x);
    x.diagnostic_axis_policy = diagnostic_policy > 0 && diagnostic_policy <= 2 ? diagnostic_policy : 0;
    x.diagnostic_first_step = diagnostic_policy == 3 || diagnostic_policy == 4;
    if (diagnostic_policy == 4) x.strict = aps_new(arena, run, o->diag_dir);
    x.st = st;
    x.seam_selected = ARENA_CALLOC(arena, run->n_rels ? run->n_rels : 1, 1);
    x.trial = 1; x.proposal = ARENA_ALLOC(arena, (nc ? nc : 1)*sizeof(AsmChart));
    AsmChart *winner = ARENA_ALLOC(arena, (nc ? nc : 1)*sizeof(AsmChart));
    uint8_t *winner_seams = x.strict ? ARENA_CALLOC(arena, run->n_rels ? run->n_rels : 1, 1) : NULL;
    Arena_T winner_uv = NULL;
    x.curv_hl = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(double));
    x.curv_ok = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(uint8_t));
    double *lidx = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(double));
    x.tfield = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(double));
    double *kover = ARENA_CALLOC(arena, run->n_layers ? run->n_layers : 1, sizeof(double));
    int32_t coord_anchor = -1;
    int32_t *comp_ring = ARENA_ALLOC(arena, ncomp_cap * sizeof(int32_t));
    for (size_t i = 0; i < nc; i++) lidx[i] = APL_LAYER_UNKNOWN;
    for (int32_t c = 0; c < ncomp; c++) comp_ring[c] = 1000000;
    x.lidx = lidx;
    Arena_T garena = NULL;
    if (copts) {
        double tg = ves_clock_sec();
        garena = Arena_new();
        x.cgrid = AsmConflictGrid_new(garena, copts, run);
        x.rgrid = aprg_new(garena, 24.0);
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c) || c->component != primary) continue;
            AsmConflictGrid_insert(x.cgrid, c); aprg_insert_chart(x.rgrid, c);
        }
        fprintf(stderr, "  [place] persistent audit grids built from the primary: %zu cells, %.1f s\n", AsmConflictGrid_cells(x.cgrid), ves_clock_sec() - tg);
    }
    {
        float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
        x.strip_umin = 1e300; x.strip_umax = -1e300;
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *ch = &run->charts[i];
            if (!AsmChart_in_layout(ch)) continue;
            for (int d = 0; d < 3; d++) { if (ch->bbox_lo[d] < lo[d]) lo[d] = ch->bbox_lo[d]; if (ch->bbox_hi[d] > hi[d]) hi[d] = ch->bbox_hi[d]; }
            if (ch->component != primary) continue;
            if (ch->pose_x < x.strip_umin) x.strip_umin = ch->pose_x;
            if (ch->pose_x > x.strip_umax) x.strip_umax = ch->pose_x;
        }
        double box_diag = hypot((double)hi[1] - lo[1], (double)hi[2] - lo[2]);
        double dl = run->layer_d_global > 0.0 ? run->layer_d_global : 10.0;
        x.span_cap = (box_diag / dl + 2.0) * APL_TURN_MAX;
        fprintf(stderr, "  [place] strip span cap %.0f vox (box diagonal %.0f / layer %.1f = %.0f wraps x %.0f)\n", x.span_cap, box_diag, dl, box_diag / dl, APL_TURN_MAX);
    }
    x.run = run; x.o = o; x.copts = copts; x.comp_placed = comp_placed; x.turn = turn; x.out_side = out_side; x.orig_comp = orig_comp;
    {
        x.s_axis = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(double));
        x.r_axis = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(double));
        x.have_radius = o->axis != NULL;
        x.cu = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(double));
        x.cv = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(double));
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (c->uv != NULL && c->nv > 0) {
                double su = 0.0, sv = 0.0;
                for (size_t v = 0; v < c->nv; v++) { su += c->uv[v*2]; sv += c->uv[v*2+1]; }
                x.cu[i] = su / (double)c->nv; x.cv[i] = sv / (double)c->nv;
            }
            if (o->axis) {
                double cen[3] = { c->centroid[0], c->centroid[1], c->centroid[2] };
                AsmAxis_project(o->axis, cen, &x.s_axis[i], &x.r_axis[i], NULL, NULL);
            } else { x.s_axis[i] = c->centroid[0]; x.r_axis[i] = 0.0; }
        }
        x.primary = primary;
        x.anchors_radius = 0; x.vm_ok = 0; x.vm_pin = 0; x.uout_ok = 0; x.uout_x = 1.0; x.uout_y = 0.0;
        if (APL_V_FROM_AXIS) {
            apl_v_model(&x);
            x.vm_pin = x.vm_ok && x.vm_b >= APL_V_MODEL_B_MIN && x.vm_b <= APL_V_MODEL_B_MAX;
            x.vm_p90 = x.vm_ok ? apl_v_model_p90(&x) : 0.0;
            if (x.have_radius && x.vm_ok && !x.vm_pin) {
                x.have_radius = 0;
                fprintf(stderr, "  [place] turn length from the radius: DISARMED with the pinning (b %.4f outside [%.2f, %.2f]: the table is not this sheet's axis); the measured turn model places\n", x.vm_b, APL_V_MODEL_B_MIN, APL_V_MODEL_B_MAX);
            }
            char pin_state[128];
            if (x.vm_pin) snprintf(pin_state, sizeof pin_state, "ARMED");
            else snprintf(pin_state, sizeof pin_state, "OFF (b outside [%.2f, %.2f]: the table is not this sheet's axis)", APL_V_MODEL_B_MIN, APL_V_MODEL_B_MAX);
            if (x.vm_ok) fprintf(stderr, "  [place] v from the axis: the primary's v = %.1f + %.4f s + %.4f u (s = axial coordinate; s-u correlation %.2f; the primary's own scatter p90 %.1f vox); tolerance %.0f vox rms, max correction %.0f vox; pinning %s\n", x.vm_a, x.vm_b, x.vm_c, x.vm_corr, x.vm_p90, fmax(APL_V_AXIS_TOL, 1.5 * x.vm_p90), fmax(APL_V_AXIS_MAX_SHIFT, 2.0 * x.vm_p90), pin_state);
            else fprintf(stderr, "  [place] v from the axis: no model (primary too small), placement keeps the anchors' v\n");
        }
        apl_uout(&x);
        if (x.have_radius) {
            double rmin = 1e300, rmax = -1e300;
            for (size_t i = 0; i < nc; i++) if (AsmChart_in_layout(&run->charts[i])) { if (x.r_axis[i] < rmin) rmin = x.r_axis[i]; if (x.r_axis[i] > rmax) rmax = x.r_axis[i]; }
            fprintf(stderr, "  [place] turn length from the radius: axis %s (%zu rows), radii %.0f..%.0f vox (turns %.0f..%.0f), outward direction %s(%.2f, %.2f)\n",
                    o->axis->is_line ? "line" : "table", o->axis->n_rows, rmin, rmax, 2.0 * APL_PI * rmin, 2.0 * APL_PI * rmax, x.uout_ok ? "" : "UNKNOWN ", x.uout_x, x.uout_y);
        }
    }
    x.anchors = ARENA_ALLOC(arena, hcap * sizeof(AplAnchor));
    x.wts = ARENA_ALLOC(arena, hcap * sizeof(double));
    x.hcap = hcap; x.primary = primary; x.comp_area = comp_area; x.diag = diag; x.fanch = fanch; x.fevidence = fevidence; x.frim = frim; x.fcut = fcut;
    double *res = ARENA_ALLOC(arena, hcap * sizeof(double));
    int32_t *anchor_count = ARENA_ALLOC(arena, ncomp_cap * sizeof(int32_t));
    int32_t *tried_at = ARENA_CALLOC(arena, ncomp_cap, sizeof(int32_t));
    size_t *tried_placed = ARENA_CALLOC(arena, ncomp_cap, sizeof(size_t));
    size_t *reconsidered_at = ARENA_CALLOC(arena,ncomp_cap,sizeof(size_t));
    uint8_t *rim_deferred = ARENA_CALLOC(arena, ncomp_cap, sizeof(uint8_t));
    int32_t *seam_count = ARENA_CALLOC(arena, ncomp_cap, sizeof(int32_t));
    uint8_t *has_seams = ARENA_CALLOC(arena, ncomp_cap, sizeof(uint8_t));
    for (size_t r = 0; r < run->n_rels; r++) {
        const AsmRelation *R = &run->rels[r];
        if (R->flags & ASM_REL_CONTACT) continue;
        const AsmChart *A = &run->charts[(size_t)R->a], *B = &run->charts[(size_t)R->b];
        if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B) || A->component < 0 || B->component < 0 || A->component == B->component) continue;
        if (A->component < (int32_t)ncomp_cap) has_seams[A->component] = 1;
        if (B->component < (int32_t)ncomp_cap) has_seams[B->component] = 1;
    }
    int32_t *rel_off = ARENA_CALLOC(arena, nc + 1, sizeof(int32_t));
    int32_t *rel_adj = ARENA_ALLOC(arena, (run->n_rels ? run->n_rels : 1) * 2 * sizeof(int32_t));
    {
        for (size_t r = 0; r < run->n_rels; r++) { rel_off[run->rels[r].a + 1]++; rel_off[run->rels[r].b + 1]++; }
        for (size_t i = 0; i < nc; i++) rel_off[i + 1] += rel_off[i];
        int32_t *fill = ARENA_CALLOC(arena, nc ? nc : 1, sizeof(int32_t));
        for (size_t r = 0; r < run->n_rels; r++) { rel_adj[rel_off[run->rels[r].a] + fill[run->rels[r].a]++] = (int32_t)r; rel_adj[rel_off[run->rels[r].b] + fill[run->rels[r].b]++] = (int32_t)r; }
    }
    size_t n_cuts = 0;
    size_t n_growth = 0;
    x.growth = 2.0 * APL_PI * run->layer_d_global;

    int max_sweeps = APL_SWEEPS_LEGACY_CAP ? 512 : 512 + 16 * ncomp;
    int placed_dirty = 1;
    size_t t0_turns = 0;
    int height_dirty = 0;
    double t_sweep0 = ves_clock_sec();
    const double budget_sec = diagnostic_policy >= 0 && diagnostic_seconds > 0
        ? fmin(apl_budget_sec(run), diagnostic_seconds) : apl_budget_sec(run);
    if (x.strict) x.strict->deadline = t_sweep0 + budget_sec;
    size_t source_recoveries = 0;
    int last_call = 0;
    for (int phase = 0; phase < (APL_DEFER_LAST_CALL ? 2 : 1); phase++) {
    if (phase == 1) {
        size_t nd = 0;
        for (int32_t c = 0; c < ncomp; c++) if (comp_placed[c] != 1 && diag[c].status == APL_DEFERRED) { comp_placed[c] = 0; tried_at[c] = 0; nd++; }
        if (nd == 0) break;
        last_call = 1;
        fprintf(stderr, "  [place] last call: %zu deferred components retried with the strict hop gate (no turn fraction) and the deferral off\n", nd);
    }
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        if (ves_clock_sec() - t_sweep0 > budget_sec) { st->budget_exhausted = 1; fprintf(stderr, "  [place] wall-clock budget %.0f s reached after %d sweeps\n", budget_sec, sweep); break; }
        st->sweeps++;
        double tq = ves_clock_sec();
        if (APL_PROFILE_EVERY > 0 && sweep > 0 && (sweep % APL_PROFILE_EVERY) == 0) apl_profile_line(st, (size_t)sweep, tq - t_sweep0, "  [place] profile");
        int remeasured_now = 0;
        if (!APL_TURN_REMEASURE || placed_dirty) {
            apl_measure_turns(run, comp_placed, out_side, lidx, turn, o->min_turn_hits);
            APL_TICK(t_measure);
            apl_fit_turn_radius(&x);
            n_growth = apl_growth_for_hops(&x, run, turn, out_side, &x.growth);
            APL_TICK(t_growth);
            if (placed_dirty && apl_v_model_update(&x)) {
                height_dirty = 1;
                fprintf(stderr,"  [place] recalibrated height from placed material: b %.6f c %.6f, unchanged uncertainty band %.1f vox\n",
                        x.vm_b,x.vm_c,apl_axis_band(&x));
            }
            placed_dirty = 0; st->remeasures++; remeasured_now = 1;
        }
        if (APL_T0_REMEASURE && sweep > 0 && remeasured_now) {
            size_t ntl2 = 0; int meas2 = 0;
            double t0b = apl_turn_t0(run, turn, out_side, orig_comp, primary, &meas2, &ntl2);
            if (meas2 && ntl2 > t0_turns) {
                fprintf(stderr, "  [place] turn model re-measured at sweep %d: T(w) = %.0f + w x %.1f vox over %zu turns (was %.0f, %s over %zu)\n",
                        sweep, t0b, x.growth, ntl2, x.t0, x.t0_measured ? "measured" : "curvature", t0_turns);
                x.t0 = t0b; x.t0_measured = 1; t0_turns = ntl2;
            }
        }
        size_t n_borrowed = APL_BORROW_TURNS ? apl_borrow_turns(run, comp_placed, out_side, lidx, turn) : 0;
        if (sweep == 0) fprintf(stderr, "  [place] borrowed turns: %zu chart sides took the nearest measured turn\n", n_borrowed);
        if (sweep == 0) {
            fprintf(stderr, "  [place] turn growth per layer %.1f vox measured on %zu charts (offset-curve rule 2 pi d = %.1f) [growth mode %d]\n",
                    x.growth, n_growth, 2.0 * APL_PI * run->layer_d_global, ASM_PLACE_GROWTH_PER_WRAP);
            fprintf(stderr, "  [place] section perimeter excess: T(r) = 2 pi r + %.0f vox over %zu charts, free slope %.3f (2 pi = %.3f, ratio %.3f) [intercept mode %d, %s]\n",
                    x.turn_c, x.turn_c_n, x.turn_slope, 2.0 * APL_PI,
                    x.turn_slope > 0.0 ? x.turn_slope / (2.0 * APL_PI) : 0.0,
                    ASM_PLACE_TURN_INTERCEPT,
                    (ASM_PLACE_TURN_INTERCEPT && x.turn_c_n) ? "APPLIED to the radius turn" : "not applied");
            size_t ntl = 0;
            x.t0 = apl_turn_t0(run, turn, out_side, orig_comp, primary, &x.t0_measured, &ntl);
            t0_turns = x.t0_measured ? ntl : 0;
            fprintf(stderr, "  [place] turn model: T(w) = %.0f + w x %.1f vox (t0 %s over %zu turns)\n", x.t0, x.growth, x.t0_measured ? "measured" : "curvature", ntl);
        {
            int32_t anchor = -1;
            for (size_t i = 0; i < nc; i++) {
                const AsmChart *c = &run->charts[i];
                if (!AsmChart_in_layout(c) || c->component != primary) continue;
                if (anchor < 0 || c->area3d > run->charts[(size_t)anchor].area3d) anchor = (int32_t)i;
            }
            size_t n_seam = 0, n_layer = 0, n_known = 0;
            double p50 = 0.0, p95 = 0.0, lmin = 0.0, lmax = 0.0;
            coord_anchor = anchor;
            if (anchor >= 0) apl_layer_index(run, out_side, turn, x.t0, x.growth, anchor, lidx, NULL, &n_seam, &n_layer, &n_known, &p50, &p95);
            else for (size_t i = 0; i < nc; i++) lidx[i] = APL_LAYER_UNKNOWN;
            for (size_t i = 0; i < nc; i++) {
                if (lidx[i] <= APL_LAYER_UNKNOWN || !AsmChart_in_layout(&run->charts[i])) continue;
                if (lidx[i] < lmin) lmin = lidx[i];
                if (lidx[i] > lmax) lmax = lidx[i];
                int32_t ring = (int32_t)floor(fabs(lidx[i]) + 0.5), oc = orig_comp[i];
                if (ring < comp_ring[oc]) comp_ring[oc] = ring;
            }
            fprintf(stderr, "  [place] layer coordinate: %zu seam edges + %zu layer edges, %zu of %zu charts reached, layers %.1f..%.1f, edge residual p50 %.2f p95 %.2f layers\n",
                    n_seam, n_layer, n_known, nc, lmin, lmax, p50, p95);
            if (APL_COORD_CALIBRATE && coord_anchor >= 0) {
                size_t ncal = apl_kover_build(&x, comp_placed, orig_comp, lidx, kover);
                size_t ns2 = 0, nl2 = 0, nk2 = 0; double q50 = 0.0, q95 = 0.0;
                apl_layer_index(run, out_side, turn, x.t0, x.growth, coord_anchor, lidx, kover, &ns2, &nl2, &nk2, &q50, &q95);
                apl_comp_ring_update(run, lidx, orig_comp, ncomp, comp_ring);
                fprintf(stderr, "  [place] coordinate calibrated on %zu placed pairs (hop counts from the layout): %zu charts reached, edge residual p50 %.2f p95 %.2f layers\n", ncal, nk2, q50, q95);
            }
        }
        if (APL_TURN_KHOP) {
            apl_measure_turns(run, comp_placed, out_side, lidx, turn, o->min_turn_hits);
            apl_fit_turn_radius(&x);
            n_growth = apl_growth_for_hops(&x, run, turn, out_side, &x.growth);
            size_t ntl2 = 0;
            x.t0 = apl_turn_t0(run, turn, out_side, orig_comp, primary, &x.t0_measured, &ntl2);
            fprintf(stderr, "  [place] turn model per wrap: T(w) = %.0f + w x %.1f vox (t0 %s over %zu turns, growth on %zu charts)\n",
                    x.t0, x.growth, x.t0_measured ? "measured" : "curvature", ntl2, n_growth);
        }
        }
        APL_TICK(t_setup);
        memset(anchor_count, 0, (size_t)ncomp * sizeof(int32_t));
        memset(seam_count, 0, (size_t)ncomp * sizeof(int32_t));
        if (APL_SEAM_ANCHORS) for (size_t r = 0; r < run->n_rels; r++) {
            const AsmRelation *R = &run->rels[r];
            if (!apl_placement_seam(R, x.have_radius, o->crosswrap_anchor_veto)) continue;
            const AsmChart *A = &run->charts[(size_t)R->a], *B = &run->charts[(size_t)R->b];
            if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
            if (comp_placed[A->component] == 1 && comp_placed[B->component] != 1) { anchor_count[B->component] += R->corr_count; seam_count[B->component] += R->corr_count; }
            else if (comp_placed[A->component] != 1 && comp_placed[B->component] == 1) { anchor_count[A->component] += R->corr_count; seam_count[A->component] += R->corr_count; }
        }
        for (size_t p = 0; p < run->n_layers; p++) {
            const AsmLayerPair *lp = &run->layers[p];
            const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
            if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
            double tx = 0.0, ty = 0.0;
            if (comp_placed[A->component] == 1 && comp_placed[B->component] != 1) {
                if (lidx[lp->a] > APL_LAYER_UNKNOWN && lidx[lp->b] > APL_LAYER_UNKNOWN && fabs(lidx[lp->b] - lidx[lp->a]) >= 0.5) anchor_count[B->component] += lp->hit_count;
                else if (apl_turn_for(turn, lp->a, lp->side, out_side[lp->a], x.growth, &tx, &ty)) anchor_count[B->component] += lp->hit_count;
                else if (turn[(size_t)lp->a * 2].n == 0 && turn[(size_t)lp->a * 2 + 1].n == 0) anchor_count[B->component] += lp->hit_count;
            } else if (comp_placed[A->component] != 1 && comp_placed[B->component] == 1) {
                if (lidx[lp->a] > APL_LAYER_UNKNOWN && lidx[lp->b] > APL_LAYER_UNKNOWN && fabs(lidx[lp->b] - lidx[lp->a]) >= 0.5) { anchor_count[A->component] += lp->hit_count; continue; }
                if (turn[(size_t)lp->b * 2].n == 0 && turn[(size_t)lp->b * 2 + 1].n == 0) { anchor_count[A->component] += lp->hit_count; continue; }
                if (out_side[lp->a] == 0 || out_side[lp->b] == 0) continue;
                int side = lp->side == out_side[lp->a] ? -out_side[lp->b] : out_side[lp->b];
                if (apl_turn_for(turn, lp->b, side, out_side[lp->b], x.growth, &tx, &ty)) anchor_count[A->component] += lp->hit_count;
            }
        }
        APL_TICK(t_count);
        int32_t best = apl_next_component(ncomp, comp_placed, anchor_count, seam_count,
                                          tried_at, comp_ring, comp_area, o->min_anchors);
        if (best < 0 && height_dirty) {
            size_t retry = 0;
            for (int32_t c = 0; c < ncomp; c++) if (comp_placed[c] == 2) {
                comp_placed[c] = 0; tried_at[c] = 0; retry++;
            }
            height_dirty = 0;
            if (retry) {
                fprintf(stderr,"  [place] height-model retry: %zu failed components after committed calibration changed (all gates retained)\n",retry);
                best = apl_next_component(ncomp, comp_placed, anchor_count, seam_count,
                                           tried_at, comp_ring, comp_area, o->min_anchors);
            }
        }
        if (best < 0 && APL_DEFER_RETRY_RIM && !last_call) {
            size_t retry = 0;
            for (int32_t c = 0; c < ncomp; c++) {
                if (comp_placed[c] != 1 && rim_deferred[c] && tried_placed[c] < st->placed) {
                    comp_placed[c] = 0; tried_at[c] = 0; retry++;
                }
            }
            if (retry > 0) {
                fprintf(stderr, "  [place] rim retry: %zu deferred components after new material was placed (all gates retained)\n", retry);
                best = apl_next_component(ncomp, comp_placed, anchor_count, seam_count,
                                           tried_at, comp_ring, comp_area, o->min_anchors);
            }
        }
        if (best < 0) {
            if(retained) break;
            int recovered = 0;
            size_t epoch = st->placed+source_recoveries+1;
            for (int32_t c = 0; c < ncomp; c++) {
                if (ves_clock_sec()-t_sweep0 > budget_sec) { st->budget_exhausted = 1; break; }
                if (reconsidered_at[c] == epoch) continue;
                reconsidered_at[c] = epoch; x.cand_grad = ori[c].grad;
                if (apl_reconsider_hop(&x,c,t_sweep0+budget_sec)) {
                    source_recoveries++; recovered = 1;
                    apl_rebuild_grids(&x,&garena);
                    for (int32_t j = 0; j < ncomp; j++) if (comp_placed[j] != 1) {
                        comp_placed[j] = 0; tried_at[j] = 0;
                    }
                    placed_dirty = 1; break;
                }
            }
            if (recovered) continue;
            break;
        }
        tried_at[best] = anchor_count[best];
        tried_placed[best] = st->placed;
        rim_deferred[best] = 0;
        x.cand_grad = ori[best].grad;
        x.attempt++;
        diag[best].attempts++;
        if (!ori[best].has_turn) {
            double best_rms[2] = { 1e300, 1e300 };
            for (int state = 0; state < 2; state++) {
                if (state == 1) apl_mirror_component(run, best);
                x.stretch_ok = apl_stretch_turn(&x, best, &x.stretch_T, &x.stretch_s, &x.stretch_d, &x.stretch_n, &x.stretch_rel, &x.stretch_pairs);
                size_t na2 = apl_collect_anchors(&x, best);
                if (na2 >= (size_t)o->min_anchors || (APL_SEAM_FIRST && x.na_seam >= (size_t)APL_SEAM_MIN_ANCHORS)) {
                    double w2[APL_MAX_CLUSTERS_TOTAL], u2[APL_MAX_CLUSTERS_TOTAL], v2[APL_MAX_CLUSTERS_TOTAL]; int kd2[APL_MAX_CLUSTERS_TOTAL];
                    int n2 = apl_cluster_anchors(&x, na2, w2, u2, v2, kd2);
                    for (int cl = 0; cl < n2; cl++) {
                        double th2, tx2, ty2; size_t kept2 = 0;
                        double r2 = apl_fit_cluster(&x, na2, cl, kd2[cl] == 1 ? (size_t)APL_SEAM_MIN_ANCHORS : (size_t)o->min_anchors, o->max_rot_refine, &th2, &tx2, &ty2, &kept2, res);
                        if (r2 < best_rms[state]) best_rms[state] = r2;
                    }
                }
            }
            if (!(best_rms[1] < 0.5 * best_rms[0])) apl_mirror_component(run, best);
            else {
                ori[best].mirrored = !ori[best].mirrored;
                st->mirrored++;
                fprintf(stderr, "  [place] component %d: anchors prefer the mirrored state (rms %.1f vs %.1f)\n", best, best_rms[1], best_rms[0]);
            }
        }
        APL_TICK(t_mirror);
        x.stretch_ok = apl_stretch_turn(&x, best, &x.stretch_T, &x.stretch_s, &x.stretch_d, &x.stretch_n, &x.stretch_rel, &x.stretch_pairs);
        if (x.stretch_ok)
            fprintf(stderr, "  [place] component %d: turn length from the layer stretch: s %.4f +- %.4f over %zu hits on %zu layer pairs, gap %.1f vox -> one turn %.0f vox, (s-1) known to %.0f%% (%s)\n",
                    best, x.stretch_s, x.stretch_rel >= 0.0 ? x.stretch_rel * fabs(x.stretch_s - 1.0) : -1.0,
                    x.stretch_n, x.stretch_pairs, x.stretch_d, fabs(x.stretch_T),
                    x.stretch_rel >= 0.0 ? 100.0 * x.stretch_rel : -1.0,
                    x.stretch_T > 0.0 ? "candidate outward (bending-side vote)" : "candidate inward (bending-side vote)");
        APL_TICK(t_stretch);
        size_t na = apl_collect_anchors(&x, best);
        APL_TICK(t_collect);
        diag[best].seam_anchors = x.na_seam;
        if (na < (size_t)o->min_anchors && !(APL_SEAM_FIRST && x.na_seam >= (size_t)APL_SEAM_MIN_ANCHORS)) { comp_placed[best] = 2; diag[best].status = APL_FEW_ANCHORS; diag[best].last_anchors = na; continue; }
        double cl_w[APL_MAX_CLUSTERS_TOTAL], cl_u[APL_MAX_CLUSTERS_TOTAL], cl_v[APL_MAX_CLUSTERS_TOTAL], cl_rms[APL_MAX_CLUSTERS_TOTAL];
        int cl_kind[APL_MAX_CLUSTERS_TOTAL];
        int ncl = apl_cluster_anchors(&x, na, cl_w, cl_u, cl_v, cl_kind);
        APL_TICK(t_cluster);
        for (int cl = 0; cl < APL_MAX_CLUSTERS_TOTAL; cl++) { cl_rms[cl] = 1e300; if (cl >= ncl) cl_kind[cl] = 0; }
        diag[best].last_clusters = ncl;
        double gate = o->max_rms + o->rel_rms * sqrt(comp_area[best]);
        double mean_turn = 0.0;
        {
            double tsum = 0.0; size_t tn = 0;
            for (size_t i = 0; i < na; i++) { tsum += hypot(x.anchors[i].Tx, x.anchors[i].Ty); tn++; }
            if (tn > 0) mean_turn = tsum / (double)tn;
            if (tn > 0 && APL_GATE_TURN_FRAC > 0.0) gate = fmax(gate, APL_GATE_TURN_FRAC * tsum / (double)tn);
        }
        int placed_now = 0, status = APL_REFUSED;
        double tcl0 = ves_clock_sec();
        size_t winner_n = 0, winner_kept = 0;
        int winner_kind = -1, ambiguous = 0;
        double winner_rms = 0;
        AplCompDiag winner_diag; memset(&winner_diag, 0, sizeof winner_diag);
        x.source_geometry_conflict = 0;
        for (int cl = 0; cl < ncl; cl++) {
            x.source_flip = 0;
            for (size_t i = 0; i < na; i++) if (x.anchors[i].cluster == cl && x.anchors[i].k == 0) { x.source_flip = x.anchors[i].source_flip; break; }
            double th, tx, ty; size_t kept = 0;
            double tf0 = ves_clock_sec();
            double rms = apl_fit_cluster(&x, na, cl, cl_kind[cl] == 1 ? (size_t)APL_SEAM_MIN_ANCHORS : (size_t)o->min_anchors, o->max_rot_refine, &th, &tx, &ty, &kept, res);
            st->t_fit += ves_clock_sec() - tf0;
            cl_rms[cl] = rms;
            double gate_cl = last_call && cl_kind[cl] == 0 ? o->max_rms + o->rel_rms * sqrt(comp_area[best]) : gate;
            if (cl_kind[cl] == 1) {
                double smax = 0.0;
                for (size_t i = 0; i < na; i++) if (x.anchors[i].cluster == cl && x.anchors[i].seam_rms > smax) smax = x.anchors[i].seam_rms;
                gate_cl = fmax(APL_SEAM_GATE_MIN, APL_SEAM_GATE_RMS_X * smax);
            }
            size_t k1 = 0, k2 = 0, est = 0, rev = 0;
            for (size_t i = 0; i < na; i++) { if (x.wts[i] <= 0.0) continue; if (x.anchors[i].k == 1) k1++; else k2++; est += x.anchors[i].estimated; rev += x.anchors[i].reverse; }
            if (rms >= 1e299) {
                fprintf(stderr, "  [place] component %d (area %.3e) cluster %d/%d (weight %.1f, offset %.0f,%.0f): %zu anchors, too few\n",
                        best, comp_area[best], cl + 1, ncl, cl_w[cl], cl_u[cl], cl_v[cl], kept);
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0, "too_few_anchors",
                                    kept, -1.0, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 0);
                continue;
            }
            fprintf(stderr, "  [place] component %d (area %.3e) cluster %d/%d (%s, weight %.1f, offset %.0f,%.0f): %zu anchors (k1 %zu k2+ %zu, estimated-T %zu, reverse %zu) rms %.2f gate %.2f rot %.3f -> %s\n",
                    best, comp_area[best], cl + 1, ncl, cl_kind[cl] == 1 ? "seam" : "hop", cl_w[cl], cl_u[cl], cl_v[cl], kept, k1, k2, est, rev, rms, gate_cl, th,
                    rms > gate_cl ? (cl_kind[cl] == 1 ? "repair" : "refused") : "fit");
            if (fanch) {
                for (size_t i = 0; i < na; i++) {
                    const AplAnchor *a = &x.anchors[i];
                    if (a->cluster != cl) continue;
                    fprintf(fanch, "%d,%d,%.6e,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n", x.attempt, best, comp_area[best], cl,
                            a->chart_p, a->chart_c, a->side, a->k, a->turn_src, a->reverse, a->Tx, a->Ty, a->sx, a->sy, a->tx, a->ty, res[i], x.wts[i] > 0.0);
                }
            }
            diag[best].last_anchors = kept; diag[best].last_rms = rms; diag[best].last_gate = gate_cl; diag[best].last_rot = th;
            diag[best].last_k1 = (int)k1; diag[best].last_k2 = (int)k2;
            if (rms > gate_cl && cl_kind[cl] != 1) {
                status = APL_REFUSED;
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0, "rms_over_gate",
                                    kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 0);
                continue;
            }
            if (APL_HOP_TURN_MAX_REL_ERR > 0.0 && cl_kind[cl] != 1 && est > 0 &&
                x.turn_rel > APL_HOP_TURN_MAX_REL_ERR) {
                status = APL_REFUSED;
                fprintf(stderr, "  [place] component %d cluster %d/%d: the turn its %zu estimated anchors rest on is known to %.0f%%; refused\n",
                        best, cl + 1, ncl, est, 100.0 * x.turn_rel);
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0, "turn_unknown",
                                    kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 0);
                continue;
            }
            double ratio = 0.0;
            double ta0 = ves_clock_sec();
            x.na = na; x.cluster = cl; x.cluster_kind = cl_kind[cl]; x.evidence_gate = gate_cl;
            memset(x.seam_selected, 0, run->n_rels); x.source_flip = 0;
            for (size_t i = 0; i < na; i++) if (x.anchors[i].cluster == cl && x.anchors[i].k == 0 && x.anchors[i].relation >= 0) {
                x.seam_selected[x.anchors[i].relation] = 1;
                x.source_flip = x.anchors[i].source_flip;
            }
            int audit_ok = apl_place_and_audit(&x, best, th, tx, ty, &ratio, 2);
            if(audit_ok && retained && diagnostic_policy < 0 && !apl_retry_candidate_qualified(&x,best))audit_ok=0;
            if (audit_ok && x.strict && !aps_candidate(&x, best)) audit_ok = 0;
            st->t_audit += ves_clock_sec() - ta0;
            if (audit_ok) {
                fprintf(stderr, "  [place] candidate supported: component %d cluster %d, %zu witnesses at %.2f vox rms, %zu fitted anchors\n",
                        best, cl+1, x.proposal[0].placement_support, x.proposal[0].placement_rms, kept);
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 1, "accepted",
                                    kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 1);
                int replace = winner_n == 0 || cl_kind[cl] > winner_kind;
                if (!replace && cl_kind[cl] == winner_kind) {
                    if (winner_n != x.nproposal) ambiguous = 1;
                    else for (size_t k = 0; k < winner_n; k++)
                        if (!apl_same_pose(&winner[k], &x.proposal[k], 8.0)) { ambiguous = 1; break; }
                }
                if (replace) {
                    winner_n = x.nproposal; winner_kind = cl_kind[cl]; ambiguous = 0;
                    apl_copy_candidate(winner,x.proposal,winner_n,run,&winner_uv);
                    if (winner_seams) memcpy(winner_seams, x.seam_selected, run->n_rels);
                    winner_kept = kept; winner_rms = rms; winner_diag = diag[best];
                }
            } else if (x.last_shape_refused) {
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0,
                                    x.last_reason ? x.last_reason : "shape_refused", kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 1);
                status = APL_REFUSED;
            } else if (x.last_silent) {
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0,
                                    x.last_reason ? x.last_reason : "deferred", kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 1);
                status = APL_DEFERRED; st->deferred++; rim_deferred[best] = 1;
            } else {
                apl_write_candidate(fcandidate, &x, best, comp_area[best], cl, cl_kind[cl], 0,
                                    x.last_reason ? x.last_reason : "reverted", kept, rms, gate_cl, k1, k2, est, mean_turn, x.turn_rel, 1);
                status = APL_REVERTED; st->reverted_by_audit++;
                if (APL_HOP_RETRY && ratio > APL_HOP_RETRY_MIN_RATIO) {
                    double hx = 0.0, hy = 0.0, hw = 0.0;
                    for (size_t i = 0; i < na; i++) {
                        const AplAnchor *a = &x.anchors[i];
                        if (x.wts[i] <= 0.0 || a->cluster != cl || a->k < 1) continue;
                        hx += x.wts[i] * a->Tx / (double)a->k; hy += x.wts[i] * a->Ty / (double)a->k; hw += x.wts[i];
                    }
                    if (hw > 0.0 && hypot(hx, hy) > 1.0) {
                        hx /= hw; hy /= hw;
                        for (int dir = -1; dir <= 1 && !placed_now; dir += 2) {
                            double r2 = 0.0;
                            st->hop_retries++;
                            if (apl_place_and_audit(&x, best, th, tx + (double)dir * hx, ty + (double)dir * hy, &r2, APL_HOP_RETRY_REQUIRE_RIM)) {
                                placed_now = 1; st->hop_retry_placed++;
                                st->anchors_used += kept; rms_list[n_rms++] = rms;
                                fprintf(stderr, "  [place]   hop retry %+d ACCEPTED (one turn %.0f,%.0f vox; the fit's landing contradicted %.0f%% of its area)\n", dir, hx, hy, 100.0 * ratio);
                            } else
                                fprintf(stderr, "  [place]   hop retry %+d rejected (contradiction %.0f%%)\n", dir, 100.0 * r2);
                        }
                    }
                }
            }
        }
        if (winner_n && !ambiguous) {
            for (size_t k = 0; k < winner_n; k++) apl_commit_chart(&x,&winner[k],run->charts[winner[k].id].uv);
            if (x.strict) {
                aps_require_selected(run, winner_seams);
                for (size_t k = 0; k < winner_n; k++) if (!aps_box(run->charts + winner[k].id, x.strict->boxes + winner[k].id)) x.strict->valid = 0;
            }
            AsmContinuity_confirm(run);
            if (x.cgrid) for (size_t k = 0; k < winner_n; k++) {
                const AsmChart *c = &run->charts[winner[k].id];
                AsmConflictGrid_insert(x.cgrid, c); aprg_insert_chart(x.rgrid, c);
            }
            placed_now = 1; st->anchors_used += winner_kept; rms_list[n_rms++] = winner_rms;
            diag[best] = winner_diag; diag[best].placed_by = winner_kind == 1 ? 1 : 2;
            if (winner_kind == 1) st->seam_placed++; else st->hop_placed++;
        } else if (ambiguous) {
            fprintf(stderr, "  [place] multiple supported poses; candidate remains deferred\n");
            status = APL_DEFERRED; rim_deferred[best] = 1; st->deferred++;
        }
        x.source_flip = 0;
        st->t_clusterloop += ves_clock_sec() - tcl0;
        if (!retained && !placed_now && ncl >= 2 && n_cuts < APL_MAX_CUTS) {
            double tc0 = ves_clock_sec();
            double cl_rms_hop[APL_MAX_CLUSTERS_TOTAL];
            for (int cl = 0; cl < APL_MAX_CLUSTERS_TOTAL; cl++) cl_rms_hop[cl] = cl_kind[cl] == 1 ? 1e300 : cl_rms[cl];
            int32_t newc = apl_cut_component(&x, best, na, ncl, cl_w, cl_u, cl_rms_hop, gate, rel_off, rel_adj, &ncomp, ncomp_cap,
                                             comp_area, orig_comp, ori, diag, comp_ring, lidx, &st->cut_rels_deferred);
            st->t_cut += ves_clock_sec() - tc0;
            if (newc >= 0) {
                n_cuts++; st->cuts++;
                comp_placed[best] = 0; comp_placed[newc] = 0;
                tried_at[best] = 0; tried_at[newc] = 0;
                diag[best].status = APL_NEVER; diag[best].attempts = 0;
                continue;
            }
        }
        if (!placed_now) { comp_placed[best] = 2; diag[best].status = status; continue; }
        comp_placed[best] = 1;
        placed_dirty = 1;
        diag[best].status = APL_PLACED;
        if (!retained && APL_METRIC_RESOLVE && st->placed > 0 && (st->placed % APL_METRIC_EVERY) == 0) {
            size_t nobs = 0;
            apl_metric_resolve(&x, comp_placed, orig_comp, ncomp, lidx, primary, &garena, copts, &nobs, 0);
        }
        if (!retained && APL_TURN_MODEL_POOL && st->placed > 0 && (st->placed % APL_METRIC_EVERY) == 0) {
            size_t nobs = 0;
            apl_metric_resolve(&x, comp_placed, orig_comp, ncomp, lidx, primary, &garena, copts, &nobs, 1);
        }
        if (APL_COORD_CALIBRATE && coord_anchor >= 0 && st->placed > 0 && (st->placed % APL_METRIC_EVERY) == 0) {
            size_t ncal = apl_kover_build(&x, comp_placed, orig_comp, lidx, kover);
            size_t ns2 = 0, nl2 = 0, nk2 = 0; double q50 = 0.0, q95 = 0.0;
            apl_layer_index(run, out_side, turn, x.t0, x.growth, coord_anchor, lidx, kover, &ns2, &nl2, &nk2, &q50, &q95);
            apl_comp_ring_update(run, lidx, orig_comp, ncomp, comp_ring);
            if ((st->placed % (APL_METRIC_EVERY * 4)) == 0)
                fprintf(stderr, "  [place] coordinate re-calibrated on %zu placed pairs: %zu charts reached, edge residual p50 %.2f p95 %.2f layers\n", ncal, nk2, q50, q95);
        }
        if (APL_TURN_FIELD && (st->placed % APL_METRIC_EVERY) == 0) {
            size_t nf = apl_turn_field_build(&x, comp_placed, orig_comp, lidx);
            if (st->placed == 0 || (st->placed % (APL_METRIC_EVERY * 4)) == 0) fprintf(stderr, "  [place] turn field: %zu placed charts carry a local observed turn\n", nf);
        }
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *ch = &run->charts[i];
            if (!AsmChart_in_layout(ch) || orig_comp[i] != best) continue;
            if (ch->pose_x < x.strip_umin) x.strip_umin = ch->pose_x;
            if (ch->pose_x > x.strip_umax) x.strip_umax = ch->pose_x;
        }
        st->placed++;
        st->area_placed += comp_area[best];
        if (last_call) st->last_call_placed++;
    }
    }
    st->t_loop = ves_clock_sec() - t_sweep0;
    apl_profile_line(st, st->sweeps, st->t_loop, "[assemble place] sweep profile");
    for (int32_t c = 0; c < ncomp; c++) if (comp_placed[c] != 1 && comp_area[c] > 0.0) st->unplaced++;
    {
        int32_t *nch = ARENA_CALLOC(arena, (size_t)(ncomp ? ncomp : 1), sizeof(int32_t));
        uint8_t *evid = ARENA_CALLOC(arena, (size_t)(ncomp ? ncomp : 1), sizeof(uint8_t));
        for (size_t i = 0; i < nc; i++) if (AsmChart_in_layout(&run->charts[i]) && orig_comp[i] >= 0 && orig_comp[i] < ncomp) nch[orig_comp[i]]++;
        for (size_t r = 0; r < run->n_rels; r++) {
            const AsmRelation *R = &run->rels[r];
            if (R->flags & ASM_REL_CONTACT) continue;
            const AsmChart *A = &run->charts[(size_t)R->a], *B = &run->charts[(size_t)R->b];
            if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B)) continue;
            int32_t ca = orig_comp[R->a], cb = orig_comp[R->b];
            if (ca < 0 || cb < 0 || ca >= ncomp || cb >= ncomp || ca == cb) continue;
            int pa = comp_placed[ca] == 1, pb = comp_placed[cb] == 1;
            if (pa == pb) continue;
            uint8_t bit = (R->flags & ASM_REL_SHORT) ? 4 : (R->flags & ASM_REL_WEAK) ? 2 : (R->flags & (ASM_REL_DROPPED | ASM_REL_SWITCHED)) ? 1 : 0;
            if (bit) evid[pa ? cb : ca] |= bit;
        }
        for (int32_t c = 0; c < ncomp; c++) {
            if (comp_area[c] <= 0.0) continue;
            int s = diag[c].status; if (s < 0 || s > 6) s = 0;
            if (nch[c] == 1) { st->singles++; st->singles_area += comp_area[c]; st->singles_by_status[s]++; st->singles_area_by_status[s] += comp_area[c]; }
            if (s == APL_PLACED && diag[c].last_rms > 30.0) { st->placed_rms30++; st->placed_rms30_area += comp_area[c]; }
            if (s == APL_REVERTED) st->reverted_area += comp_area[c];
            if (s == APL_DEFERRED) { st->deferred_comps++; st->deferred_area += comp_area[c]; }
            if (comp_placed[c] != 1 && s != APL_PRIMARY && diag[c].attempts == 0) {
                st->never_attempted++; st->never_attempted_area += comp_area[c];
            }
            if (comp_placed[c] != 1 && s != APL_PRIMARY) {
                if (evid[c]) { st->evid_comps++; st->evid_area += comp_area[c]; if (evid[c] & 1) st->evid_dropped++; if (evid[c] & 2) st->evid_weak++; if (evid[c] & 4) st->evid_short++; }
                else { st->noevid_comps++; st->noevid_area += comp_area[c]; }
            }
        }
    }
    apl_record_poses(run,o,"placed_before_uv",1);
    if(!retained){
        AsmContinuity_relax_placed_uv(run,apl_uv_audit,o->stress_band,o->diag_dir);
        AsmContinuity_confirm(run);
    }
    if(!retained) {
        AplOrient lev;
        apl_orient_rotate(run, primary, &lev);
        st->relevel_theta = lev.theta;
        fprintf(stderr, "  [place] re-levelled the placed strip by %.4f rad (z-gradient %.3f)\n", lev.theta, lev.grad);
    }
    if (n_rms) {
        qsort(rms_list, n_rms, sizeof(double), apl_cmp_double);
        st->rms_p50 = rms_list[n_rms/2]; st->rms_max = rms_list[n_rms-1];
    }
    st->growth_measured = x.growth;
    apl_record_poses(run, o, "placed", 1);
    st->growth_charts = n_growth;
    if (fanch) fclose(fanch);
    if (fcandidate) fclose(fcandidate);
    if (fevidence) fclose(fevidence);
    if (frim) fclose(frim);
    if (fcut) fclose(fcut);
    if (o->diag_dir) {
        char path[2048];
        snprintf(path, sizeof path, "%s/stage5_components.csv", o->diag_dir);
        FILE *fc = fopen(path, "wb");
        if (fc) {
            fputs("component,area,status,attempts,clusters,last_anchors,last_k1,last_k2,last_rms,last_gate,last_rot,mirrored,turn_u,bend,z_gradient,"
                  "cz,cy,cx,z0,z1,y0,y1,x0,x1,u_mean,v_mean,placed_by,seam_anchors\n", fc);
            for (int32_t c = 0; c < ncomp; c++) {
                if (comp_area[c] <= 0.0) continue;
                double cz = 0, cy = 0, cx = 0, um = 0, vm = 0, aw = 0;
                float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
                for (size_t i = 0; i < nc; i++) {
                    const AsmChart *ch = &run->charts[i];
                    if (!AsmChart_in_layout(ch) || orig_comp[i] != c) continue;
                    double gx, gy;
                    apl_pose(ch, 0.0, 0.0, &gx, &gy);
                    cz += ch->area3d * ch->centroid[0]; cy += ch->area3d * ch->centroid[1]; cx += ch->area3d * ch->centroid[2];
                    um += ch->area3d * gx; vm += ch->area3d * gy; aw += ch->area3d;
                    for (int d = 0; d < 3; d++) { if (ch->bbox_lo[d] < lo[d]) lo[d] = ch->bbox_lo[d]; if (ch->bbox_hi[d] > hi[d]) hi[d] = ch->bbox_hi[d]; }
                }
                if (aw > 0.0) { cz /= aw; cy /= aw; cx /= aw; um /= aw; vm /= aw; }
                fprintf(fc, "%d,%.6e,%s,%d,%d,%zu,%d,%d,%.2f,%.2f,%.4f,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.1f,%d,%zu\n",
                        c, comp_area[c], apl_status_names[diag[c].status], diag[c].attempts,
                        diag[c].last_clusters, diag[c].last_anchors, diag[c].last_k1, diag[c].last_k2, diag[c].last_rms, diag[c].last_gate, diag[c].last_rot,
                        ori[c].mirrored, ori[c].turn_u, ori[c].bend, ori[c].grad, cz, cy, cx, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], um, vm,
                        diag[c].placed_by, diag[c].seam_anchors);
            }
            fclose(fc);
        }
        snprintf(path, sizeof path, "%s/stage5_charts.csv", o->diag_dir);
        FILE *fch = fopen(path, "wb");
        if (fch) {
            fputs("chart,cube,orig_component,status,area,nv,cz,cy,cx,u_mean,v_mean,mirror,out_side,layer,s,r,phi\n", fch);
            for (size_t i = 0; i < nc; i++) {
                const AsmChart *ch = &run->charts[i];
                if (!AsmChart_in_layout(ch)) continue;
                double gx, gy;
                apl_pose(ch, 0.0, 0.0, &gx, &gy);
                int32_t oc = orig_comp[i];
                double s_ax = 0.0, r_ax = -1.0, phi_ax = 0.0;
                if (o->axis) {
                    double cen[3] = { ch->centroid[0], ch->centroid[1], ch->centroid[2] }, fr[3];
                    AsmAxis_frame(o->axis, cen, fr);
                    s_ax = fr[0]; r_ax = hypot(fr[1], fr[2]); phi_ax = atan2(fr[2], fr[1]);
                }
                fprintf(fch, "%d,%d,%d,%s,%.4e,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d,%.3f,%.1f,%.1f,%.4f\n", ch->id, ch->cube, oc,
                        oc >= 0 && oc < ncomp ? apl_status_names[diag[oc].status] : "none", ch->area3d, ch->nv,
                        ch->centroid[0], ch->centroid[1], ch->centroid[2], gx, gy, (ch->flags & ASM_CHART_MIRROR) != 0, out_side[i],
                        lidx[i] > APL_LAYER_UNKNOWN ? lidx[i] : -999.0, s_ax, r_ax, phi_ax);
            }
            fclose(fch);
        }
        fprintf(stderr, "  [place] cuts: %zu components split at disagreeing anchor clusters (%zu seam relations deferred), %d components now\n", st->cuts, st->cut_rels_deferred, ncomp);
        fprintf(stderr, "  [place] source continuations preserved against %zu conflicting layer-cut proposals\n",x.cuts_preserved_source);
        int32_t top[10]; size_t ntop = 0;
        for (int32_t c = 0; c < ncomp; c++) {
            if (comp_placed[c] == 1 || comp_area[c] <= 0.0) continue;
            size_t pos = ntop;
            while (pos > 0 && comp_area[top[pos-1]] < comp_area[c]) pos--;
            if (pos >= 10) continue;
            for (size_t k = (ntop < 10 ? ntop : 9); k > pos; k--) top[k] = top[k-1];
            top[pos] = c;
            if (ntop < 10) ntop++;
        }
        for (size_t k = 0; k < ntop; k++) {
            int32_t c = top[k];
            size_t from_placed = 0, with_turn = 0, from_unplaced = 0;
            int32_t src[3] = { -1, -1, -1 }; size_t srcn[3] = { 0, 0, 0 };
            for (size_t p = 0; p < run->n_layers; p++) {
                const AsmLayerPair *lp = &run->layers[p];
                const AsmChart *A = &run->charts[(size_t)lp->a], *B = &run->charts[(size_t)lp->b];
                if (!AsmChart_in_layout(A) || !AsmChart_in_layout(B) || B->component != c) continue;
                if (comp_placed[A->component] != 1) { from_unplaced += (size_t)lp->hit_count; continue; }
                from_placed += (size_t)lp->hit_count;
                double tx = 0.0, ty = 0.0;
                if (apl_turn_for(turn, lp->a, lp->side, out_side[lp->a], x.growth, &tx, &ty)) with_turn += (size_t)lp->hit_count;
                else if (turn[(size_t)lp->a * 2].n == 0 && turn[(size_t)lp->a * 2 + 1].n == 0) with_turn += (size_t)lp->hit_count;
                int32_t oc = orig_comp[lp->a];
                int slot = -1;
                for (int s2 = 0; s2 < 3; s2++) if (src[s2] == oc) slot = s2;
                if (slot < 0) for (int s2 = 0; s2 < 3; s2++) if (src[s2] < 0) { src[s2] = oc; slot = s2; break; }
                if (slot >= 0) srcn[slot] += (size_t)lp->hit_count;
            }
            fprintf(stderr, "  [place] unplaced #%zu: component %d area %.3e %s (attempts %d, clusters %d, last anchors %zu, rms %.1f / gate %.1f) | layer hits from placed %zu (with a turn vector %zu) from unplaced %zu | sources: comp %d x%zu, comp %d x%zu, comp %d x%zu\n",
                    k + 1, c, comp_area[c], apl_status_names[diag[c].status], diag[c].attempts, diag[c].last_clusters, diag[c].last_anchors, diag[c].last_rms, diag[c].last_gate,
                    from_placed, with_turn, from_unplaced, src[0], srcn[0], src[1], srcn[1], src[2], srcn[2]);
        }
    }
    st->anchors_radius = x.anchors_radius;
    st->axis_armed = !(x.vm_ok && !x.vm_pin);
    st->axis_b = x.vm_ok ? x.vm_b : 0.0;
    if (x.vm_ok) apl_v_model(&x);
    if (x.vm_ok) {
        Arena_Mark wm = Arena_save(arena);
        double *res = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(double));
        double *wres = ARENA_ALLOC(arena, (nc ? nc : 1) * sizeof(double));
        size_t np = 0, npl = 0;
        double w30 = 0.0, w100 = 0.0, wpl = 0.0, vmin = 1e300, vmax = -1e300, umin = 1e300, umax = -1e300;
        for (size_t i = 0; i < nc; i++) {
            const AsmChart *c = &run->charts[i];
            if (!AsmChart_in_layout(c) || c->component != primary) continue;
            double gx, gy;
            apl_layout_centroid(&x, c, i, 0.0, 0.0, 0.0, &gx, &gy);
            double r = x.vm_a + x.vm_b * x.s_axis[i] + x.vm_c * gx - gy;
            if (gy < vmin) vmin = gy;
            if (gy > vmax) vmax = gy;
            if (gx < umin) umin = gx;
            if (gx > umax) umax = gx;
            if (orig_comp[i] == primary) res[np++] = fabs(r);
            else { wres[npl] = fabs(r); npl++; wpl += c->area3d; if (fabs(r) > 30.0) w30 += c->area3d; if (fabs(r) > 100.0) w100 += c->area3d; }
        }
        if (np > 0) { qsort(res, np, sizeof(double), apl_cmp_double); st->wobble_primary_p50 = res[np / 2]; st->wobble_primary_p90 = res[(size_t)((double)(np - 1) * 0.9)]; }
        st->placed_area_beyond30 = wpl > 0.0 ? w30 / wpl : 0.0;
        st->placed_area_beyond100 = wpl > 0.0 ? w100 / wpl : 0.0;
        st->v_span = vmax > vmin ? vmax - vmin : 0.0;
        if (umax > umin) {
            size_t nb = (size_t)((umax - umin) / 2000.0) + 1;
            double *steps = ARENA_ALLOC(arena, (nb ? nb : 1) * sizeof(double));
            double prev = 0.0; int have_prev = 0; size_t ns = 0;
            for (size_t b = 0; b < nb; b++) {
                double lo = umin + 2000.0 * (double)b, hi = lo + 2000.0;
                size_t m = 0; double warea = 0.0;
                for (size_t i = 0; i < nc; i++) {
                    const AsmChart *c = &run->charts[i];
                    if (!AsmChart_in_layout(c) || c->component != primary) continue;
                    double gx, gy;
                    apl_layout_centroid(&x, c, i, 0.0, 0.0, 0.0, &gx, &gy);
                    if (gx < lo || gx >= hi) continue;
                    double r = x.vm_a + x.vm_b * x.s_axis[i] + x.vm_c * gx - gy;
                    if (fabs(r) > 100.0) continue;
                    res[m++] = r; warea += c->area3d;
                }
                if (m < 3 || warea < 5.0e4) continue;
                qsort(res, m, sizeof(double), apl_cmp_double);
                double med = res[m / 2];
                if (have_prev) steps[ns++] = fabs(med - prev);
                prev = med; have_prev = 1;
            }
            if (ns > 0) { qsort(steps, ns, sizeof(double), apl_cmp_double); st->spine_step_p50 = steps[ns / 2]; st->spine_step_p90 = steps[(size_t)((double)(ns - 1) * 0.9)]; st->spine_bins = ns + 1; }
        }
        if (umax > umin) {
            size_t nb = (size_t)((umax - umin) / 2000.0) + 1;
            double rsum = 0.0, rw = 0.0, zlo = 1e300, zhi = -1e300;
            for (size_t i = 0; i < nc; i++) {
                const AsmChart *c = &run->charts[i];
                if (!AsmChart_in_layout(c) || c->component != primary || c->uv == NULL || c->xyz == NULL) continue;
                for (size_t k = 0; k < c->nv; k++) {
                    double z = (double)c->xyz[k*3];
                    if (z < zlo) zlo = z;
                    if (z > zhi) zhi = z;
                }
                if (x.have_radius) { rsum += c->area3d * x.r_axis[i]; rw += c->area3d; }
            }
            st->outline_r_mean = rw > 0.0 ? rsum / rw : 0.0;
            if (zhi > zlo + 2.0 * APL_OUTLINE_CROP_VOX) {
                size_t *cnt_t = ARENA_CALLOC(arena, nb, sizeof(size_t)), *cnt_b = ARENA_CALLOC(arena, nb, sizeof(size_t));
                size_t tot_t = 0, tot_b = 0;
                for (int pass = 0; pass < 2; pass++) {
                    for (size_t i = 0; i < nc; i++) {
                        const AsmChart *c = &run->charts[i];
                        if (!AsmChart_in_layout(c) || c->component != primary || c->uv == NULL || c->xyz == NULL) continue;
                        double ct = cos(c->pose_theta), stt = sin(c->pose_theta);
                        double mir = (c->flags & ASM_CHART_MIRROR) ? -1.0 : 1.0;
                        for (size_t k = 0; k < c->nv; k++) {
                            double z = (double)c->xyz[k*3];
                            int on_top = z >= zhi - APL_OUTLINE_CROP_VOX, on_bot = z <= zlo + APL_OUTLINE_CROP_VOX;
                            if (!on_top && !on_bot) continue;
                            double u = mir * c->uv[k*2], v = c->uv[k*2+1];
                            double gx = ct * u - stt * v + c->pose_x, gy = stt * u + ct * v + c->pose_y;
                            if (gx < umin || gx >= umax) continue;
                            size_t b = (size_t)((gx - umin) / 2000.0);
                            if (b >= nb) b = nb - 1;
                            if (pass == 0) { if (on_top) cnt_t[b]++; else cnt_b[b]++; }
                            else if (on_top) x.tr_top[x.off_t[b] + x.fill_t[b]++] = gy;
                            else x.tr_bot[x.off_b[b] + x.fill_b[b]++] = gy;
                        }
                    }
                    if (pass == 0) {
                        x.off_t = ARENA_ALLOC(arena, (nb + 1) * sizeof(size_t)); x.off_b = ARENA_ALLOC(arena, (nb + 1) * sizeof(size_t));
                        x.fill_t = ARENA_CALLOC(arena, nb, sizeof(size_t)); x.fill_b = ARENA_CALLOC(arena, nb, sizeof(size_t));
                        x.off_t[0] = 0; x.off_b[0] = 0;
                        for (size_t b = 0; b < nb; b++) { x.off_t[b+1] = x.off_t[b] + cnt_t[b]; x.off_b[b+1] = x.off_b[b] + cnt_b[b]; }
                        tot_t = x.off_t[nb]; tot_b = x.off_b[nb];
                        x.tr_top = ARENA_ALLOC(arena, (tot_t ? tot_t : 1) * sizeof(double));
                        x.tr_bot = ARENA_ALLOC(arena, (tot_b ? tot_b : 1) * sizeof(double));
                    }
                }
                double tmin = 1e300, tmax = -1e300, bmin = 1e300, bmax = -1e300;
                size_t nbt = 0, nbb = 0;
                for (size_t b = 0; b < nb; b++) {
                    if (cnt_t[b] >= 8) {
                        double *seg = x.tr_top + x.off_t[b];
                        qsort(seg, cnt_t[b], sizeof(double), apl_cmp_double);
                        double med = seg[cnt_t[b] / 2];
                        if (med < tmin) tmin = med;
                        if (med > tmax) tmax = med;
                        nbt++;
                    }
                    if (cnt_b[b] >= 8) {
                        double *seg = x.tr_bot + x.off_b[b];
                        qsort(seg, cnt_b[b], sizeof(double), apl_cmp_double);
                        double med = seg[cnt_b[b] / 2];
                        if (med < bmin) bmin = med;
                        if (med > bmax) bmax = med;
                        nbb++;
                    }
                }
                st->outline_top_p2p = nbt > 1 ? tmax - tmin : 0.0;
                st->outline_bot_p2p = nbb > 1 ? bmax - bmin : 0.0;
                st->outline_bins_top = nbt; st->outline_bins_bot = nbb;
            }
            if (o->axis != NULL && zhi >= zlo) {
                double dir[3] = { 1.0, 0.0, 0.0 };
                AsmAxis_mean_dir(o->axis, zlo, zhi, dir);
                double cz = fabs(dir[0]);
                if (cz > 1.0) cz = 1.0;
                st->axis_tilt_deg = acos(cz) * 180.0 / APL_PI;
                st->outline_pred_p2p = 2.0 * st->outline_r_mean * tan(acos(cz));
            }
        }
        Arena_restore(arena, wm);
    }
    st->v_model[0] = x.vm_ok ? x.vm_a : 0.0; st->v_model[1] = x.vm_ok ? x.vm_b : 0.0; st->v_model[2] = x.vm_ok ? x.vm_c : 0.0;
    st->sec = ves_clock_sec() - t0;
    if (x.cgrid != NULL) AsmConflictGrid_dispose(x.cgrid);
    if (garena != NULL) Arena_dispose(&garena);
    if (x.proposal_uv) Arena_dispose(&x.proposal_uv);
    if (winner_uv) Arena_dispose(&winner_uv);
    int strict_rc = 0;
    if (x.strict) {
        fprintf(stderr, "[place strict] %zu/%zu candidate certificates passed in %.3f s\n", x.strict->accepted, x.strict->checked, x.strict->seconds);
        if (x.strict->log && fclose(x.strict->log)) x.strict->io = 0;
        strict_rc = !x.strict->io || !x.strict->valid;
    }
    Arena_restore(arena, mark);
    return strict_rc ? -1 : 0;
}

int AsmPlace_run(AsmRun *run,const AsmPlaceOpts *o,const AsmConflictOpts *copts,AsmPlaceStats *st)
{return apl_run(run,o,copts,st,NULL,-1,0);}

int AsmPlace_retry_component(AsmRun *run,const uint8_t *core,const AsmPlaceOpts *o,
                             const AsmConflictOpts *copts,AsmPlaceStats *st)
{
    if(!run || !core || !o || !copts || !st || !run->continuity_ready)return -1;
    Arena_T arena=Arena_new();size_t nc=run->n_charts,count=0,fixed=0;
    AsmRun trial=*run;trial.arena=arena;
    trial.charts=ARENA_ALLOC(arena,nc*sizeof(AsmChart));memcpy(trial.charts,run->charts,nc*sizeof(AsmChart));
    trial.rels=ARENA_ALLOC(arena,run->n_rels*sizeof(AsmRelation));memcpy(trial.rels,run->rels,run->n_rels*sizeof(AsmRelation));
    uint8_t *retained=ARENA_CALLOC(arena,nc,1);int result=0;
    for(size_t c=0;c<nc;c++){
        AsmChart *ch=trial.charts+c;
        if(core[c]){
            if(AsmChart_registered(ch) || !AsmChart_in_layout(ch) || ch->placed_uv){result=-1;goto done;}
            ch->component=1;ch->placed=1;ch->placement_state=ASM_PLACE_NONE;count++;
            ch->uv=ARENA_ALLOC(arena,2*ch->nv*sizeof(float));memcpy(ch->uv,run->charts[c].uv,2*ch->nv*sizeof(float));
        }else if(AsmChart_registered(ch)){
            retained[c]=1;fixed++;ch->component=0;
        }else{ch->flags|=ASM_CHART_EXTRAS;ch->component=-1;}
    }
    if(!count || !fixed)goto done;
    for(size_t i=0;i<run->n_rels;i++){
        const AsmRelation *r=run->rels+i;
        if(!(r->flags&ASM_REL_CONTACT) && (r->continuity&(ASM_CONT_SOURCE|ASM_CONT_CUT)) &&
            core[r->a]!=core[r->b])goto done;
    }
    if(apl_run(&trial,o,copts,st,retained,-1,0)){result=-1;goto done;}
    for(size_t c=0;c<nc;c++)if(core[c] && !AsmChart_registered(trial.charts+c))goto done;
    for(size_t c=0;c<nc;c++)if(retained[c]){
        AsmChart after=trial.charts[c];after.component=run->charts[c].component;
        if(memcmp(&after,run->charts+c,sizeof after)){result=-1;goto done;}
    }
    for(size_t c=0;c<nc;c++)if(core[c]){
        AsmChart after=trial.charts[c];
        after.uv=ARENA_ALLOC(run->arena,2*after.nv*sizeof(float));
        memcpy(after.uv,trial.charts[c].uv,2*after.nv*sizeof(float));
        run->charts[c]=after;
    }
    result=1;
done:
    Arena_dispose(&arena);return result;
}

static int apl_registration_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for (int collision = 0; collision < 2; collision++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 12;
        float xyz[2][243], uv[2][162], normals[243]; uint8_t boundary[81]; int32_t faces[384];
        for (int j = 0; j < 9; j++) for (int u = 0; u < 9; u++) {
            int v = 9*j+u; boundary[v] = j == 0 || j == 8 || u == 0 || u == 8;
            normals[3*v] = normals[3*v+1] = 0; normals[3*v+2] = 1;
            for (int side = 0; side < 2; side++) {
                xyz[side][3*v] = (float)(16*j);
                xyz[side][3*v+1] = (float)(5*u+(side && !collision ? 44 : 0));
                xyz[side][3*v+2] = (float)(side && collision ? 12 : 0);
                uv[side][2*v] = (float)(5*u); uv[side][2*v+1] = (float)(16*j);
            }
            if (j < 8 && u < 8) { int32_t f[] = {v,v+1,v+10,v,v+10,v+9}; memcpy(faces+6*(8*j+u),f,sizeof f); }
        }
        for (int side = 0; side < 2; side++) {
            AsmChart c = {0}; c.id = side; c.nv = 81; c.nf = 128; c.area3d = side ? 5120 : 5121;
            c.xyz = xyz[side]; c.uv = uv[side]; c.nrm = normals; c.faces = faces; c.boundary = boundary;
            c.flags = mirror ? ASM_CHART_MIRROR : 0; c.pose_x = side*(mirror ? -200 : 200);
            AsmRun_push_chart(&run,&c);
        }
        AsmRelation rel = {0}; rel.a = 0; rel.b = 1; rel.corr_count = rel.n_corr = 9; rel.rms = 0.1;
        for (int j = 0; j < 9; j++) { AsmCorr c = {9*j+(collision ? 0 : 8),9*j}; AsmRun_push_corr(&run,&c); }
        AsmRun_push_rel(&run,&rel); AsmContinuity_prepare(&run);
        AsmChart saved[2] = {run.charts[0],run.charts[1]};
        AsmRelation saved_rel = run.rels[0]; AsmCorr saved_corr[9]; memcpy(saved_corr,run.corr,sizeof saved_corr);
        size_t before = apl_internal_bad(&run,saved,2,0);
        size_t kept = AsmContinuity_register_components(&run,apl_registration_audit);
        size_t after = apl_internal_bad(&run,saved,2,1);
        int ok = kept == (size_t)!collision && after <= before && !memcmp(&run.charts[0],&saved[0],sizeof saved[0]) &&
            !memcmp(&run.rels[0],&saved_rel,sizeof saved_rel) && !memcmp(run.corr,saved_corr,sizeof saved_corr);
        if (collision) ok = ok && !memcmp(run.charts,saved,sizeof saved);
        else {
            double rms; size_t support;
            ok = ok && fabs(run.charts[1].pose_x-(mirror ? -44 : 44)) < 1e-5 &&
                 AsmContinuity_measure(&run,&run.rels[0],&rms,&support) && rms < 1e-5 && support == 9;
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: registration collision %d mirror %d, kept %zu, bad %zu -> %zu\n",collision,mirror,kept,before,after); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_registration_reject_test(AsmRun *run, const AsmChart *saved, size_t n, AsmContinuityAuditCache *cache)
{
    (void)run; (void)saved; (void)n; (void)cache; return 0;
}

static int apl_registration_planes_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for (int stalled = 0; stalled < 2; stalled++) {
        AsmRun run = {0}; run.arena = Arena_new();
        float uv[4][12] = {{0}}, xyz[4][18] = {{0}}, original_uv[4][12], original_xyz[4][18];
        int32_t faces[] = {0,1,3,0,3,2,2,3,5,2,5,4};
        for (int i = 0; i < 4; i++) {
            for (int v = 0; v < 6; v++) {
                uv[i][2*v] = (float)(2*(v%2)-1); uv[i][2*v+1] = (float)(2*(v/2)-2);
                xyz[i][3*v] = uv[i][2*v]+(i == 3 && !stalled ? 20 : 0); xyz[i][3*v+1] = uv[i][2*v+1];
            }
            AsmChart c = {0}; c.id = i; c.nv = 6; c.nf = 4; c.uv = uv[i]; c.xyz = xyz[i]; c.faces = faces;
            c.area3d = i ? 100 : 101; c.pose_theta = .4; c.flags = mirror ? ASM_CHART_MIRROR : 0;
            if (i == 3 && !stalled) { c.pose_x = (mirror ? -18 : 18)*cos(.4); c.pose_y = (mirror ? -18 : 18)*sin(.4); }
            AsmRun_push_chart(&run,&c);
        }
        for (int i = 0; i < 4; i++) {
            AsmRelation r = {0}; r.a = i == 1 ? 1 : 0; r.b = i == 0 ? 1 : i == 3 ? 3 : 2;
            r.corr_first = (int32_t)run.n_corr; r.corr_count = r.n_corr = 6; r.continuity = ASM_CONT_SOURCE;
            r.flags = i == 2 ? ASM_REL_WEAK : 0;
            for (int v = 0; v < 6; v++) {
                AsmCorr c = {0}; c.va = c.vb = v; c.valid = 3;
                c.gap_a[0] = c.gap_b[0] = i == 2 ? 10 : i == 3 ? 18 : 0;
                if (i == 3 && stalled) {
                    double u = uv[0][2*v], w = uv[0][2*v+1];
                    c.gap_a[0] = c.gap_b[0] = (float)((cos(.3)-1)*u-sin(.3)*w);
                    c.gap_a[1] = c.gap_b[1] = (float)(sin(.3)*u+(cos(.3)-1)*w);
                }
                AsmRun_push_corr(&run,&c);
            }
            AsmRun_push_rel(&run,&r);
        }
        AsmChart saved[4]; AsmRelation rels[4]; AsmCorr corr[24];
        memcpy(saved,run.charts,sizeof saved); memcpy(rels,run.rels,sizeof rels); memcpy(corr,run.corr,sizeof corr);
        memcpy(original_uv,uv,sizeof uv); memcpy(original_xyz,xyz,sizeof xyz);
        AsmContinuityAuditCache cache = {0}; cache.guard_charts = saved; cache.guard_count = 4;
        int ok = apl_uv_audit(&run,saved,4,&cache) && !cache.baseline_bad;
        size_t kept = AsmContinuity_register_failed(&run,saved,apl_registration_audit);
        ok = ok && kept == 1 && apl_uv_audit(&run,saved,4,&cache) &&
             !memcmp(run.charts,saved,sizeof saved[0]) && !memcmp(run.rels,rels,sizeof rels) &&
             !memcmp(run.corr,corr,sizeof corr) && !memcmp(uv,original_uv,sizeof uv) && !memcmp(xyz,original_xyz,sizeof xyz);
        for (int i = 0; i < 4; i++) {
            double rms; size_t support;
            if (!AsmContinuity_measure(&run,&run.rels[i],&rms,&support)) ok = 0;
        }
        for (int i=1; i<4; i++) ok = ok &&
            hypot(run.charts[i].pose_x-saved[i].pose_x,run.charts[i].pose_y-saved[i].pose_y)<=256 &&
            fabs(run.charts[i].pose_theta-saved[i].pose_theta)<=.25;
        if (stalled) {
            memcpy(run.charts,saved,sizeof saved);
            ok = ok && !AsmContinuity_register_failed(&run,saved,apl_registration_reject_test) && !memcmp(run.charts,saved,sizeof saved);
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: constrained registration mirror %d stalled %d, kept %zu\n",mirror,stalled,kept); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_seam_incumbent_selftest(void)
{
    int fails = 0;
    for (int first_step = 0; first_step < 2; first_step++)
    for (int mirror = 0; mirror < 2; mirror++) for (int test = 0; test < 2; test++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 12; run.continuity_ready = 1;
        float uv[3][36], xyz[3][54], normals[54]; int32_t faces[48]; uint8_t boundary[18];
        AsmChart charts[3] = {{0}}, original[3], proposal[1]; AsmCorr corr[18] = {{0}};
        double sign = mirror ? -1 : 1, ct = cos(test ? .08 : .5), sn = sin(test ? .08 : .5);
        AplAnchor anchors[54] = {{0}}; double weights[54] = {0};
        for (int v = 0; v < 18; v++) {
            double u = (test ? 0 : -5)+10.0*v/17, w = (v%2 ? 1 : -1)*(test ? .01 : .2);
            uv[1][2*v] = (float)u; uv[1][2*v+1] = (float)w;
            uv[0][2*v] = (float)(ct*u-sn*w); uv[0][2*v+1] = (float)(sn*u+ct*w);
            uv[2][2*v] = (float)u; uv[2][2*v+1] = (float)(24.05+w);
            for (int side = 0; side < 3; side++) {
                xyz[side][3*v] = (float)(1000+sign*uv[0][2*v]); xyz[side][3*v+1] = uv[0][2*v+1]; xyz[side][3*v+2] = side == 2 ? 100 : 0;
            }
            normals[3*v] = normals[3*v+1] = 0; normals[3*v+2] = 1; boundary[v] = 1;
            corr[v].va = corr[v].vb = v; corr[v].valid = 3; corr[v].run = 0;
            AplAnchor *a = &anchors[v]; a->chart_p = 0; a->chart_c = 1; a->relation = 0;
            a->sx = sign*uv[1][2*v]; a->sy = uv[1][2*v+1];
            a->local_u = uv[1][2*v]; a->local_v = uv[1][2*v+1];
            a->tx = 1000+sign*uv[0][2*v]; a->ty = uv[0][2*v+1]; a->w = weights[v] = 1;
            if (v < 16) { faces[3*v] = v; faces[3*v+1] = v+(v%2 ? 2 : 1); faces[3*v+2] = v+(v%2 ? 1 : 2); }
        }
        for (int i = 0; i < 3; i++) {
            charts[i].id = charts[i].component = i; charts[i].nv = 18; charts[i].nf = 16; charts[i].area3d = 4;
            charts[i].uv = uv[i]; charts[i].xyz = xyz[i]; charts[i].nrm = normals; charts[i].faces = faces; charts[i].boundary = boundary;
            charts[i].flags = mirror ? ASM_CHART_MIRROR : 0;
        }
        charts[0].pose_x = 1000; charts[0].placement_state = ASM_PLACE_ROOT;
        charts[2].pose_x = 1000; charts[2].component = 0; charts[2].placement_state = ASM_PLACE_ROOT;
        memcpy(original,charts,sizeof original);
        AsmRelation rel = {0}; rel.a = 0; rel.b = 1; rel.flags = ASM_REL_WEAK; rel.continuity = ASM_CONT_SOURCE;
        rel.corr_count = rel.n_corr = 18; rel.rms = 1;
        run.charts = charts; run.n_charts = test ? 3 : 2; run.rels = &rel; run.n_rels = 1; run.corr = corr; run.n_corr = 18;
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts); opts.max_rot_refine = .1;
        AsmConflictOpts copts; AsmConflict_default_opts(&copts);
        int32_t groups[] = {0,1,2}; uint8_t selected[] = {1}; double area[] = {4,4,4};
        AplCtx x = {0}; x.run = &run; x.o = &opts; x.copts = &copts; x.orig_comp = groups; x.primary = 0;
        x.trial = 1; x.proposal = proposal; x.cluster_kind = 1; x.anchors = anchors; x.wts = weights; x.na = 18;
        x.seam_selected = selected; x.comp_area = area; x.diagnostic_first_step = first_step;
        Arena_T grid_arena = Arena_new();
        x.cgrid = AsmConflictGrid_new(grid_arena,&copts,&run); x.rgrid = aprg_new(grid_arena,24);
        AsmConflictGrid_insert(x.cgrid,&charts[0]); aprg_insert_chart(x.rgrid,&charts[0]);
        if (test) { AsmConflictGrid_insert(x.cgrid,&charts[2]); aprg_insert_chart(x.rgrid,&charts[2]); }
        charts[1].pose_x = 1000; charts[1].component = 0;
        double rms; size_t support, touched; const AsmChart *query = &charts[1];
        int ok = AsmContinuity_measure(&run,&rel,&rms,&support) && support == 18 && rms < 2 &&
                 aprg_probe(x.rgrid,&run,&query,1,24,&touched) == 0 && touched >= 8;
        charts[1] = original[1];
        double th = 0, tx = 1000, ty = 0;
        int closed = apl_close_seam(&x,&th,&tx,&ty);
        ok = ok && closed == test;
        if (test) {
            apl_compose(&charts[1],th,tx,ty,0); charts[1].component = 0;
            ok = ok && aprg_probe(x.rgrid,&run,&query,1,24,&touched) > opts.max_rim_ratio && touched >= 8;
            charts[1] = original[1];
        }
        int accepted = apl_place_and_audit(&x,1,0,1000,0,NULL,2);
        ok = ok && accepted && x.nproposal == 1 && proposal[0].pose_x == 1000 && proposal[0].pose_theta == 0 &&
             proposal[0].placement_state == ASM_PLACE_SEAM && proposal[0].placement_support == 18 &&
             !memcmp(charts,original,sizeof original);
        for (int v = 0; v < 18; v++) {
            anchors[18+v] = anchors[v]; anchors[18+v].k = anchors[18+v].measured_order = 1;
            anchors[18+v].cluster = 1; anchors[18+v].tx += 2000;
            anchors[18+v].Tx = 1000; anchors[18+v].observation = (size_t)v;
        }
        x.na = 36; x.n_observations = 18;
        int source_precedes = apl_place_and_audit(&x,1,0,1000,0,NULL,2);
        ok = ok && source_precedes && x.consensus.observations == 18 && !x.consensus.model_inliers &&
             proposal[0].placement_state == ASM_PLACE_SEAM && !memcmp(charts,original,sizeof original);
        x.na = 18; x.n_observations = 0;
        if (!test) ok = ok && !apl_place_and_audit(&x,1,0,1100,0,NULL,2) && !memcmp(charts,original,sizeof original);
        for (int v = 0; v < 18; v++) xyz[1][3*v+2] = 1000;
        int rejected = !apl_place_and_audit(&x,1,0,1000,0,NULL,2);
        ok = ok && rejected && !memcmp(charts,original,sizeof original);
        if (test) {
            for (int v = 0; v < 18; v++) {
                xyz[1][3*v+2] = 0;
                uv[2][2*v+1] -= 14;
            }
            apl_rebuild_grids(&x,&grid_arena);
            x.source_geometry_conflict = 0;
            int seam_refused = !apl_place_and_audit(&x,1,0,1000,0,NULL,2);
            ok = ok && seam_refused && x.source_geometry_conflict && !memcmp(charts,original,sizeof original);
            x.cluster_kind = 0; selected[0] = 0;
            for (int v = 0; v < 18; v++) {
                anchors[v].k = anchors[v].measured_order = 1;
                anchors[v].tx = 2000+sign*uv[1][2*v]; anchors[v].ty = uv[1][2*v+1];
                anchors[v].Tx = 1000; anchors[v].observation = (size_t)v;
            }
            x.n_observations = 18;
            int hop_refused = !apl_place_and_audit(&x,1,0,2000,0,NULL,2);
            ok = ok && hop_refused && x.last_silent && !memcmp(charts,original,sizeof original) &&
                 x.last_reason && !strcmp(x.last_reason, "source_geometry_conflict");
            x.source_geometry_conflict = 0;
            int clear_hop = apl_place_and_audit(&x,1,0,2000,0,NULL,2);
            ok = ok && clear_hop && proposal[0].placement_state == ASM_PLACE_LAYER &&
                 proposal[0].placement_support == 18 && x.consensus.observations == 18 &&
                 x.consensus.model_inliers == 18 && !memcmp(charts,original,sizeof original);
            for (int v = 0; v < 18; v++) {
                anchors[18+v] = anchors[v]; anchors[18+v].cluster = 1;
                anchors[18+v].k = 2; anchors[18+v].Tx = 2000; anchors[18+v].tx += 1000;
                anchors[36+v] = anchors[v]; anchors[36+v].cluster = 2;
                anchors[36+v].observation = 18+(size_t)v; anchors[36+v].tx += 1000;
            }
            x.na = 54; x.n_observations = 36;
            int contrary_refused = !apl_place_and_audit(&x,1,0,2000,0,NULL,2);
            ok = ok && contrary_refused && x.last_silent && x.consensus.observations == 36 &&
                 x.consensus.model_inliers == 18 && !memcmp(charts,original,sizeof original) &&
                 x.last_reason && !strcmp(x.last_reason, "consensus_minority");
            size_t cells = AsmConflictGrid_cells(x.cgrid);
            x.trial = 0;
            int direct_refused = !apl_place_and_audit(&x,1,0,2000,0,NULL,2);
            ok = ok && direct_refused && !memcmp(charts,original,sizeof original) &&
                 AsmConflictGrid_cells(x.cgrid) == cells;
            charts[1].pose_x = 2000;
            ok = ok && aprg_probe(x.rgrid,&run,&query,1,24,&touched) == 0 && !touched;
            charts[1] = original[1]; x.trial = 1;
            x.na = 53; x.n_observations = 35;
            int majority_passes = apl_place_and_audit(&x,1,0,2000,0,NULL,2);
            ok = ok && majority_passes && x.consensus.observations == 35 && x.consensus.model_inliers == 18 &&
                 !memcmp(charts,original,sizeof original);
            AsmChart first_pose = proposal[0];
            x.na = 36; x.n_observations = 18; x.cluster = 1;
            for (int v = 0; v < 18; v++) { weights[v] = 0; weights[18+v] = 1; }
            int alternate_passes = apl_place_and_audit(&x,1,0,3000,0,NULL,2);
            ok = ok && alternate_passes && x.consensus.observations == 18 && x.consensus.model_inliers == 18 &&
                 !apl_same_pose(&first_pose,&proposal[0],8) && !memcmp(charts,original,sizeof original);
            if (!contrary_refused || !direct_refused || !majority_passes || !alternate_passes)
                fprintf(stderr,"  asm_place selftest detail: contrary %d direct %d majority %d ambiguous alternative %d\n",
                        contrary_refused,direct_refused,majority_passes,alternate_passes);
            if (!seam_refused || !hop_refused || !clear_hop)
                fprintf(stderr,"  asm_place selftest detail: source conflict %d hop deferred %d clear control %d\n",seam_refused,hop_refused,clear_hop);
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: supported seam incumbent mirror %d test %d, closed %d, accepted %d, unsafe rejected %d\n",mirror,test,closed,accepted,rejected); fails++; }
        AsmConflictGrid_dispose(x.cgrid); Arena_dispose(&grid_arena);
        if (x.proposal_uv) Arena_dispose(&x.proposal_uv);
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_stopped_repair_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for (int total_bound = 0; total_bound < 2; total_bound++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 12; run.continuity_ready = 1;
        AsmChart charts[4] = {{0}}, original[4], proposal[3]; AsmRelation rels[4] = {{0}}; AsmCorr corr[24] = {{0}};
        float uv[4][12], xyz[4][18], normals[18]; uint8_t boundary[6];
        int32_t faces[] = {0,1,3,0,3,2,2,3,5,2,5,4};
        for (int i = 0; i < 4; i++) {
            for (int v = 0; v < 6; v++) {
                uv[i][2*v] = (float)(2*(v%2)-1); uv[i][2*v+1] = (float)(2*(v/2)-2);
                xyz[i][3*v] = uv[i][2*v]; xyz[i][3*v+1] = uv[i][2*v+1]; xyz[i][3*v+2] = 0;
                normals[3*v] = normals[3*v+1] = 0; normals[3*v+2] = 1; boundary[v] = 1;
            }
            AsmChart *c = &charts[i]; c->id = i; c->component = i ? 1 : 0; c->nv = 6; c->nf = 4; c->area3d = 8;
            c->uv = uv[i]; c->xyz = xyz[i]; c->nrm = normals; c->faces = faces; c->boundary = boundary;
            c->pose_theta = .4; c->flags = mirror ? ASM_CHART_MIRROR : 0;
        }
        charts[0].placement_state = ASM_PLACE_ROOT;
        for (int i = 0; i < 4; i++) {
            AsmRelation *r = &rels[i]; r->a = i == 1 ? 1 : 0; r->b = i == 0 ? 1 : i == 3 ? 3 : 2;
            r->corr_first = 6*i; r->corr_count = r->n_corr = 6; r->continuity = ASM_CONT_SOURCE;
            r->flags = i == 2 ? ASM_REL_WEAK : 0;
            for (int v = 0; v < 6; v++) {
                AsmCorr *c = &corr[6*i+v]; c->va = c->vb = v; c->valid = 3;
                c->gap_a[0] = c->gap_b[0] = i == 2 ? 10 : 0;
                if (i == 3) {
                    double angle = total_bound ? .26 : .3, u = uv[0][2*v], w = uv[0][2*v+1];
                    c->gap_a[0] = c->gap_b[0] = (float)((cos(angle)-1)*u-sin(angle)*w);
                    c->gap_a[1] = c->gap_b[1] = (float)(sin(angle)*u+(cos(angle)-1)*w);
                }
            }
        }
        run.charts = charts; run.n_charts = 4; run.rels = rels; run.n_rels = 4; run.corr = corr; run.n_corr = 24;
        memcpy(original,charts,sizeof original);
        int32_t groups[] = {0,1,1,1}; uint8_t selected[] = {1,1,1,1}; double area[] = {8,24};
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts); AsmConflictOpts copts; AsmConflict_default_opts(&copts);
        AplCtx x = {0}; x.run = &run; x.o = &opts; x.copts = &copts; x.orig_comp = groups; x.primary = 0;
        x.trial = 1; x.proposal = proposal; x.cluster_kind = 1; x.seam_selected = selected; x.comp_area = area;
        Arena_T grid_arena = Arena_new(); x.cgrid = AsmConflictGrid_new(grid_arena,&copts,&run); x.rgrid = aprg_new(grid_arena,24);
        AsmConflictGrid_insert(x.cgrid,&charts[0]); aprg_insert_chart(x.rgrid,&charts[0]);
        int solved = AsmContinuity_close(&run,original+1,3,selected,NULL,0);
        int within = 1, seams = 1;
        for (int i = 1; i < 4; i++) if (hypot(charts[i].pose_x-original[i].pose_x,charts[i].pose_y-original[i].pose_y) > 256 ||
                                       fabs(charts[i].pose_theta-original[i].pose_theta) > .25) within = 0;
        for (int i = 0; i < 4; i++) { double rms; size_t support; if (!AsmContinuity_measure(&run,&rels[i],&rms,&support)) seams = 0; }
        int ok = seams && within;
        memcpy(charts,original,sizeof original);
        int polished = 0;
        int accepted = apl_place_and_audit_once(&x,1,0,0,0,NULL,2,0,&polished);
        ok = ok && accepted && !polished && !memcmp(charts,original,sizeof original);
        if (accepted) ok = ok && x.nproposal == 3 && proposal[0].placement_state == ASM_PLACE_SEAM &&
                           proposal[0].placement_support >= 6;
        if (accepted) for (int i=0; i<3; i++) ok = ok &&
            hypot(proposal[i].pose_x-original[i+1].pose_x,proposal[i].pose_y-original[i+1].pose_y)<=256 &&
            fabs(proposal[i].pose_theta-original[i+1].pose_theta)<=.25;
        for (int i = 1; i < 4; i++) for (int v = 0; v < 6; v++) xyz[i][3*v+2] = 1000;
        int rejected = !apl_place_and_audit_once(&x,1,0,0,0,NULL,2,0,&polished);
        ok = ok && rejected && !memcmp(charts,original,sizeof original);
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: bounded repair mirror %d total bound %d, solved %d, seams %d, within %d, accepted %d, unsafe rejected %d\n",
                          mirror,total_bound,solved,seams,within,accepted,rejected); fails++; }
        AsmConflictGrid_dispose(x.cgrid); Arena_dispose(&grid_arena);
        if (x.proposal_uv) Arena_dispose(&x.proposal_uv);
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_reconsider_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for (int test = 0; test < 8; test++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 12;
        float xyz[3][243], uv[3][162], original_uv[3][162], normals[243];
        int32_t faces[384]; uint8_t boundary[81];
        for (int j = 0; j < 9; j++) for (int u = 0; u < 9; u++) {
            int v = j*9+u; boundary[v] = j == 0 || j == 8 || u == 0 || u == 8;
            normals[3*v] = normals[3*v+1] = 0; normals[3*v+2] = 1;
            for (int side = 0; side < 3; side++) {
                xyz[side][3*v] = (float)(j*16);
                xyz[side][3*v+1] = (float)(u*5+((side == 1 || (side == 2 && test == 1)) ? 44 : 0));
                if (side == 2 && test >= 6) xyz[side][3*v+1] = (float)(u*5+88);
                xyz[side][3*v+2] = side == 2 && test == 1 ? 12.0f : 0;
                uv[side][2*v] = (float)(u*5); uv[side][2*v+1] = (float)(j*16);
            }
            if (j < 8 && u < 8) {
                int32_t f[6] = {v,v+1,v+10,v,v+10,v+9}; memcpy(faces+(j*8+u)*6,f,sizeof f);
            }
        }
        double sign = mirror ? -1.0 : 1.0;
        int nch = test == 1 || test == 2 || test >= 5 ? 3 : 2;
        for (int i = 0; i < nch; i++) {
            AsmChart c = {0}; c.id = i; c.component = 0; c.placed = 1;
            c.nv = 81; c.nf = 128; c.xyz = xyz[i]; c.uv = uv[i]; c.faces = faces; c.nrm = normals; c.boundary = boundary;
            c.area3d = 5120; c.flags = mirror ? ASM_CHART_MIRROR : 0;
            c.pose_x = sign*(i == 1 ? 1244 : (i == 2 ? (test == 1 ? 44 : 200) : 0));
            c.placement_state = i == 1 ? (test == 4 ? ASM_PLACE_SEAM : ASM_PLACE_LAYER) : ASM_PLACE_ROOT;
            if (test == 5 && i == 2) c.placement_state = ASM_PLACE_LAYER;
            if (test >= 6 && i == 2) { c.pose_x = sign*1288; c.placement_state = test == 6 ? ASM_PLACE_SEAM : ASM_PLACE_ROOT; }
            c.placement_group = i;
            AsmRun_push_chart(&run,&c);
        }
        for (int i = 0; i < (test == 2 || test >= 6 ? 2 : 1); i++) {
            AsmRelation r = {0}; r.a = i ? 2 : 0; r.b = 1; r.flags = ASM_REL_WEAK; r.rms = 0.1;
            if (i && test >= 6) { r.a = 1; r.b = 2; r.flags = 0; }
            r.corr_first = (int32_t)run.n_corr; r.corr_count = r.n_corr = 9;
            for (int j = 0; j < 9; j++) { AsmCorr c = {j*9+8,j*9}; AsmRun_push_corr(&run,&c); }
            AsmRun_push_rel(&run,&r);
        }
        AsmContinuity_prepare(&run);
        AsmChart saved[3]; memcpy(saved,run.charts,run.n_charts*sizeof *saved);
        AsmRelation rels[2]; memcpy(rels,run.rels,run.n_rels*sizeof *rels);
        memcpy(original_uv,uv,sizeof uv);
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts);
        AsmConflictOpts copts; AsmConflict_default_opts(&copts);
        AsmPlaceStats stats = {0}; AsmChart proposal[3]; AplCompDiag diag[3] = {{0}};
        AplCtx x = {0}; x.run = &run; x.o = &opts; x.copts = &copts; x.st = &stats; x.primary = 0; x.trial = 1; x.proposal = proposal;
        int32_t groups[3] = {0,1,2}; uint8_t placed[3] = {1,1,1}, selected[2] = {0}, curv[3] = {1,1,1};
        double area[3] = {5120,5120,5120}, cu[3], cv[3], weights[18]; AplAnchor anchors[18];
        if (test == 5) { groups[2] = 1; area[1] += area[2]; }
        x.orig_comp = groups; x.comp_placed = placed; x.comp_area = area; x.seam_selected = selected; x.diag = diag;
        x.cu = cu; x.cv = cv; x.curv_ok = curv; x.anchors = anchors; x.wts = weights; x.hcap = 18; x.cand_grad = 1;
        for (int i = 0; i < nch; i++) apl_uv_centroid(&run.charts[i],&cu[i],&cv[i]);
        Arena_T grid_arena = Arena_new(); apl_rebuild_grids(&x,&grid_arena);
        int accepted = apl_reconsider_hop(&x,1,test == 3 ? ves_clock_sec()-1 : 0);
        int ok = accepted == (test == 0 || test == 6) && !x.excluded_samples && placed[1] == 1 && !memcmp(uv,original_uv,sizeof uv);
        ok = ok && x.orig_comp == groups && groups[0] == 0 && groups[1] == 1 && groups[2] == (test == 5 ? 1 : 2) &&
             area[1] == (test == 5 ? 10240 : 5120) && area[2] == 5120;
        if (accepted) {
            double rms; size_t support;
            ok = ok && fabs(run.charts[1].pose_x-sign*44) < 1e-5 &&
                 run.charts[1].placement_state == ASM_PLACE_SEAM && curv[1] == 0 &&
                 AsmContinuity_measure(&run,&run.rels[0],&rms,&support) && support == 9 && rms < 1e-5 &&
                 (run.rels[0].continuity & ASM_CONT_VERIFIED) && !memcmp(&run.charts[0],&saved[0],sizeof *saved);
            if (test == 6) ok = ok && fabs(run.charts[2].pose_x-sign*88) < 1e-5 && curv[2] == 0 &&
                run.charts[2].placement_group == 2 && run.charts[2].placement_state == ASM_PLACE_SEAM &&
                AsmContinuity_measure(&run,&run.rels[1],&rms,&support) && rms < 1e-5 && support == 9;
            apl_rebuild_grids(&x,&grid_arena);
            const AsmChart *old = &saved[1]; size_t touched;
            aprg_probe(x.rgrid,&run,&old,1,24,&touched);
            ok = ok && touched == 0;
        } else {
            ok = ok && !memcmp(run.charts,saved,run.n_charts*sizeof *saved) &&
                 !memcmp(run.rels,rels,run.n_rels*sizeof *rels) && curv[1] == 1;
            if (test == 3) ok = ok && stats.budget_exhausted;
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: reconsider hop test %d mirror %d, accepted %d\n",test,mirror,accepted); fails++; }
        AsmConflictGrid_dispose(x.cgrid); Arena_dispose(&grid_arena);
        if (x.proposal_uv) Arena_dispose(&x.proposal_uv);
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_uv_transaction_selftest(double amplitude)
{
    AsmRun run = {0}; run.arena = Arena_new();
    float xyz[2][243], uv[2][162], original[2][162]; int32_t faces[384]; uint8_t boundary[81];
    for (int j = 0; j < 9; j++) for (int u = 0; u < 9; u++) {
        int v = j*9+u; boundary[v] = j == 0 || j == 8 || u == 0 || u == 8;
        for (int side = 0; side < 2; side++) {
            xyz[side][v*3] = (float)(j*16); xyz[side][v*3+1] = (float)(u*5+side*44); xyz[side][v*3+2] = 0;
            uv[side][v*2] = (float)(u*5); uv[side][v*2+1] = (float)(j*16);
        }
        if (j < 8 && u < 8) {
            int32_t f[6] = {v,v+1,v+10,v,v+10,v+9}; memcpy(faces+(j*8+u)*6,f,sizeof f);
        }
    }
    for (int side = 0; side < 2; side++) {
        AsmChart c = {0}; c.id = c.component = side; c.nv = 81; c.nf = 128;
        c.xyz = xyz[side]; c.uv = uv[side]; c.faces = faces; c.boundary = boundary; c.area3d = 5120;
        c.pose_x = side*44; c.placement_state = side ? ASM_PLACE_NONE : ASM_PLACE_ROOT;
        AsmRun_push_chart(&run,&c);
    }
    AsmRelation r = {0}; r.a = 0; r.b = 1; r.corr_count = r.n_corr = 9; r.rms = 0.1; r.flags = ASM_REL_SWITCHED;
    for (int j = 0; j < 9; j++) { AsmCorr c = {j*9+8,j*9}; AsmRun_push_corr(&run,&c); }
    AsmRun_push_rel(&run,&r); AsmContinuity_prepare(&run);
    for (int j = 0; j < 9; j++) for (int u = 0; u < 9; u++)
        uv[1][(j*9+u)*2] += (float)(amplitude*cos(j*APL_PI/4)*(1-u/8.0));
    double initial_rms; size_t initial_support;
    int initially_passes = AsmContinuity_measure(&run,&run.rels[0],&initial_rms,&initial_support);
    memcpy(original,uv,sizeof uv);
    AsmChart saved[2] = {run.charts[0],run.charts[1]}, proposal[1], winner[1];
    AsmPlaceOpts opts; AsmPlace_default_opts(&opts);
    AplCtx x = {0}; x.run = &run; x.o = &opts; x.primary = 0; x.trial = 1; x.proposal = proposal; x.cluster_kind = 1;
    int32_t groups[2] = {0,1}; uint8_t placed[2] = {1,0}, selected[1] = {1}, curv[2] = {1,1};
    AplAnchor anchors[9]; double weights[9], cu[2], cv[2], old_cu[2], old_cv[2];
    x.orig_comp = groups; x.comp_placed = placed; x.seam_selected = selected;
    x.anchors = anchors; x.wts = weights; x.hcap = 9; x.cu = cu; x.cv = cv; x.curv_ok = curv;
    for (int side = 0; side < 2; side++) apl_uv_centroid(&run.charts[side],&cu[side],&cv[side]);
    memcpy(old_cu,cu,sizeof cu); memcpy(old_cv,cv,sizeof cv);
    int ok = initial_support == 9 && initially_passes == (amplitude <= 8) && apl_collect_anchors(&x,1) == 9; x.na = 9;
    for (int i = 0; i < 9; i++) weights[i] = 1;
    ok = ok && apl_place_and_audit(&x,1,0,0,0,NULL,2) && x.uv_trial && x.nproposal == 1;
    ok = ok && !memcmp(run.charts,saved,sizeof saved) && !memcmp(uv,original,sizeof uv) &&
         !memcmp(cu,old_cu,sizeof cu) && !memcmp(cv,old_cv,sizeof cv) && curv[1] == 1;
    Arena_T winner_uv = NULL;
    if (ok) {
        apl_copy_candidate(winner,proposal,1,&run,&winner_uv);
        float winning_uv[162]; memcpy(winning_uv,winner[0].uv,sizeof winning_uv);
        if (initially_passes) {
            int refined = 0;
            ok = apl_place_and_audit_once(&x,1,0,0,0,NULL,2,0,&refined) && !refined;
            AsmChart base = proposal[0];
            double u, v, bu, bv, pu, pv, axial[2] = {0};
            apl_uv_centroid(&base,&u,&v); apl_pose(&base,u,v,&bu,&bv);
            apl_uv_centroid(&winner[0],&u,&v); apl_pose(&winner[0],u,v,&pu,&pv);
            double change = (pv-0.1*pu)-(bv-0.1*bu);
            ok = ok && fabs(change) > 1e-5;
            x.vm_ok = x.vm_pin = 1; x.vm_c = 0.1; x.s_axis = axial;
            x.vm_a = bv-0.1*bu-copysign(APL_V_AXIS_MAX_SHIFT-0.25*fabs(change),change);
            ok = ok && fabs(x.vm_a+0.1*pu-pv) > APL_V_AXIS_MAX_SHIFT &&
                 apl_place_and_audit_once(&x,1,0,0,0,NULL,2,0,&refined) && !refined &&
                 apl_same_pose(&base,&proposal[0],1e-6) && !memcmp(proposal[0].uv,original[1],sizeof original[1]);
            ok = ok && apl_place_and_audit(&x,1,0,0,0,NULL,2) &&
                 !memcmp(run.charts,saved,sizeof saved) && !memcmp(uv,original,sizeof uv) &&
                 !memcmp(winner[0].uv,winning_uv,sizeof winning_uv);
            if (ok) {
                double au, av; apl_uv_centroid(&proposal[0],&u,&v); apl_pose(&proposal[0],u,v,&au,&av);
                AsmMetricFace metric[128]; AsmMetricStats before, after;
                ok = fabs(x.vm_a+0.1*au-av) <= APL_V_AXIS_MAX_SHIFT && !AsmMetric_prepare(&proposal[0],metric);
                if (ok) {
                    AsmMetric_measure(&proposal[0],metric,original[1],&before);
                    AsmMetric_measure(&proposal[0],metric,proposal[0].uv,&after);
                    ok = AsmMetric_preserved(&before,&after);
                }
            }
            x.vm_ok = x.vm_pin = 0; x.s_axis = NULL;
        }
        x.span_cap = 1;
        ok = ok && !apl_place_and_audit(&x,1,0,0,0,NULL,2) &&
             !memcmp(run.charts,saved,sizeof saved) && !memcmp(uv,original,sizeof uv) &&
             !memcmp(winner[0].uv,winning_uv,sizeof winning_uv);
        apl_commit_chart(&x,&winner[0],uv[1]);
        double rms; size_t support;
        ok = ok && AsmContinuity_measure(&run,&run.rels[0],&rms,&support) && support == 9 && rms < initial_rms &&
             run.charts[1].uv == uv[1] && memcmp(uv[1],original[1],sizeof uv[1]) && curv[1] == 0 &&
             !memcmp(&run.charts[0],&saved[0],sizeof saved[0]) && !memcmp(uv[0],original[0],sizeof uv[0]);
        double mean_u, mean_v; apl_uv_centroid(&run.charts[1],&mean_u,&mean_v);
        ok = ok && cu[1] == mean_u && cv[1] == mean_v;
        for (size_t v = 0; v < 81; v++) {
            ok = ok && hypot((double)uv[1][2*v]-original[1][2*v],
                             (double)uv[1][2*v+1]-original[1][2*v+1]) <= 8;
            if (v%9 == 8) ok = ok && !memcmp(uv[1]+2*v,original[1]+2*v,2*sizeof(float));
        }
    }
    if (!ok) fprintf(stderr,"  asm_place selftest FAIL: local UV trial rollback, winner lifetime or commit (amplitude %.1f)\n",amplitude);
    if (x.proposal_uv) Arena_dispose(&x.proposal_uv);
    if (winner_uv) Arena_dispose(&winner_uv);
    Arena_dispose(&run.arena); return !ok;
}

static int apl_rim_pair_selftest(void)
{
    const double reach = 24;
    const double radii[] = {0,1,7,16,nextafter(24,0),24,nextafter(24,INFINITY),32};
    size_t counts[3] = {0};
    for (int i = 0; i < 65536; i++) {
        double angle = i*0.6180339887498948, r = radii[i%8], tol = 7+(i%5);
        double dx = r*cos(angle), dy = r*sin(angle);
        if (i%7 == 0) { dx = r; dy = 0; }
        float p[3], q[3];
        double physical = i%3 == 0 ? r+tol : (i%3 == 1 ? r : 48.0*(i%101)/100.0);
        for (int k = 0; k < 3; k++) p[k] = i%4 == 0 ? 0 : (float)((k+1)*4096+i%17);
        q[0] = p[0]+(float)(physical*cos(angle));
        q[1] = p[1]+(float)(physical*sin(angle)); q[2] = p[2];
        if (i%11 == 0) { q[0] = p[0]+(float)physical; q[1] = p[1]; }
        if (i%13 == 0) q[0] = nextafterf(q[0],i%2 ? INFINITY : -INFINITY);
        double duv = hypot(dx,dy);
        double d3 = sqrt((q[0]-p[0])*(q[0]-p[0])+(q[1]-p[1])*(q[1]-p[1])+(q[2]-p[2])*(q[2]-p[2]));
        int reference = duv > reach ? -1 : d3-duv > tol;
        int actual = aprg_pair_test(dx,dy,p,q,reach,reach*reach*(1+8*DBL_EPSILON),reach*reach*(1-64*DBL_EPSILON),tol);
        if (reference != actual) {
            fprintf(stderr,"  asm_place selftest FAIL: rim pair %d, UV %.17g physical %.17g tol %.17g: %d vs %d\n",i,duv,d3,tol,actual,reference);
            return 1;
        }
        counts[reference+1]++;
    }
    return !counts[0] || !counts[1] || !counts[2];
}

static int apl_rotated_cluster_selftest(void)
{
    int fails = 0;
    for (int sign = -1; sign <= 1; sign += 2) for (int shifted = 0; shifted < 2; shifted++)
    for (int distinct_rotation = 0; distinct_rotation < 2; distinct_rotation++) {
        AsmRun run = {0}; run.arena = Arena_new();
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts);
        AplAnchor anchors[128] = {{0}}, original[128];
        for (int branch = 0; branch < 2; branch++) for (int v = 0; v < 64; v++) {
            AplAnchor *a = &anchors[64*branch+v];
            double u = 1600.0*(v/2)/31, w = v%2 ? 80 : -80;
            double th = sign*.25*(branch && distinct_rotation ? -1 : 1);
            a->sx = u+(shifted ? 1e6 : 0); a->sy = w-(shifted ? 2e6 : 0);
            a->tx = cos(th)*u-sin(th)*w+1000*(branch+1);
            a->ty = sin(th)*u+cos(th)*w;
            a->w = branch ? .5 : 1; a->k = branch+1; a->measured_order = 1;
            a->Tx = 1000*(branch+1); a->cluster = -1;
        }
        memcpy(original,anchors,sizeof original);
        AplCtx x = {0}; x.run = &run; x.o = &opts; x.anchors = anchors; x.cand_grad = 1;
        double weight[APL_MAX_CLUSTERS_TOTAL], u[APL_MAX_CLUSTERS_TOTAL], v[APL_MAX_CLUSTERS_TOTAL];
        int kind[APL_MAX_CLUSTERS_TOTAL];
        int count = apl_cluster_anchors(&x,128,weight,u,v,kind);
        int ok = distinct_rotation ? count >= 2 : count == 2;
        size_t second = 0;
        for (int i = 0; i < 128; i++) {
            if (i < 64) ok = ok && anchors[i].cluster == 0;
            else {
                ok = ok && anchors[i].cluster != 0;
                if (anchors[i].cluster >= 0) second++;
                if (!distinct_rotation) ok = ok && anchors[i].cluster == 1;
            }
            AplAnchor saved = anchors[i]; saved.cluster = original[i].cluster;
            ok = ok && !memcmp(&saved,&original[i],sizeof saved);
        }
        ok = ok && second >= 12;
        if (!ok) {
            fprintf(stderr,"  asm_place selftest FAIL: rotated clustering sign %d shifted %d distinct %d: %d clusters, second support %zu\n",
                    sign,shifted,distinct_rotation,count,second);
            fails++;
        }
        Arena_dispose(&run.arena);
    }
    for (int sign = -1; sign <= 1; sign += 2) {
        AsmRun run = {0}; run.arena = Arena_new();
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts);
        AplAnchor anchors[96] = {{0}}, original[96];
        for (int i = 0; i < 96; i++) {
            AplAnchor *a = anchors+i;
            double u = 1600.0*((i < 64 ? i : 2*(i-64))/2)/31, v = i%2 ? 80 : -80;
            a->sx = u; a->sy = v; a->w = i < 64 ? 1 : .5;
            a->k = i < 64 ? 1 : 2; a->Tx = 1000*a->k; a->measured_order = 1; a->cluster = -1;
            a->tx = i < 64 ? cos(sign*.25)*u-sin(sign*.25)*v+1000 : u+2500+(i%2 ? 20 : -20);
            a->ty = i < 64 ? sin(sign*.25)*u+cos(sign*.25)*v : v+(i%2 ? -60 : 60);
        }
        memcpy(original,anchors,sizeof original);
        AplCtx x = {0}; x.run = &run; x.o = &opts; x.anchors = anchors; x.cand_grad = 1;
        double w[APL_MAX_CLUSTERS_TOTAL], u[APL_MAX_CLUSTERS_TOTAL], v[APL_MAX_CLUSTERS_TOTAL]; int kind[APL_MAX_CLUSTERS_TOTAL];
        int n = apl_cluster_anchors(&x,96,w,u,v,kind), ok = n == 2;
        for (int i = 0; i < 96; i++) {
            ok = ok && anchors[i].cluster == (i < 64 ? 0 : 1);
            AplAnchor saved = anchors[i]; saved.cluster = original[i].cluster;
            ok = ok && !memcmp(&saved,&original[i],sizeof saved);
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: incoherent alternative regrouped, sign %d (%d clusters)\n",sign,n); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_v_model_update_selftest(void)
{
    AsmRun run = {0}; run.arena = Arena_new();
    double cu[13] = {0}, cv[13] = {0}, axis[13]; int32_t original[13];
    float uv[2] = {0,0};
    for (int i = 0; i < 13; i++) {
        AsmChart c = {0}; c.nv = 1; c.uv = uv; c.area3d = 1;
        c.component = 0; original[i] = i < 8 ? 0 : 1;
        axis[i] = ((i/2)%2) ? 100 : -100;
        c.pose_x = i < 8 ? ((i%2) ? 500 : -500) : ((i%2) ? 2500 : 1500);
        c.pose_y = axis[i]+(i < 8 ? 0 : ((i%2) ? 45 : 35));
        c.placement_state = i < 8 ? ASM_PLACE_ROOT : ASM_PLACE_LAYER;
        if (i == 12) {
            c.area3d = 1e12; c.pose_x = 10000; c.pose_y = -10000;
            c.placement_state = ASM_PLACE_NONE; axis[i] = 0;
        }
        AsmRun_push_chart(&run,&c);
    }
    AplCtx x = {0}; x.run = &run; x.primary = 0; x.orig_comp = original;
    x.cu = cu; x.cv = cv; x.s_axis = axis;
    int ok = !apl_v_model(&x) && x.vm_ok && fabs(x.vm_a) < 1e-9 && fabs(x.vm_b-1) < 1e-9 && fabs(x.vm_c) < 1e-9;
    x.vm_pin = 1; x.vm_p90 = 20;
    AsmChart before[13]; memcpy(before,run.charts,sizeof before);
    ok = ok && apl_v_model_update(&x) && fabs(x.vm_a-100.0/41) < 1e-8 &&
         fabs(x.vm_b-1) < 1e-9 && fabs(x.vm_c-67.0/4100) < 1e-9 && x.vm_p90 == 20 &&
         !memcmp(before,run.charts,sizeof before) && !apl_v_model_update(&x);
    for (int i = 8; i < 12; i++) run.charts[i].pose_y += axis[i];
    double a = x.vm_a, b = x.vm_b, c = x.vm_c;
    ok = ok && !apl_v_model_update(&x) && x.vm_a == a && x.vm_b == b && x.vm_c == c;
    for (int i = 8; i < 12; i++) run.charts[i].pose_y -= axis[i];
    x.vm_a = x.vm_c = 0; x.vm_b = 1;
    run.charts[12].area3d = .001; run.charts[12].pose_y = -99;
    run.charts[12].placement_state = ASM_PLACE_SEAM; original[12] = 2;
    ok = ok && !apl_v_model_update(&x) && x.vm_a == 0 && x.vm_b == 1 && x.vm_c == 0 && x.vm_p90 == 20;
    if (!ok) fprintf(stderr,"  asm_place selftest FAIL: committed height recalibration and incumbent bounds\n");
    Arena_dispose(&run.arena);
    return !ok;
}

static int apl_self_contact_audit_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for(int deep=0;deep<2;deep++) {
        float scale=deep?128.f:1.f;
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float uv[12] = {0,0,2,0,0,2, 1.05f,1.05f,3.05f,1.05f,1.05f,3.05f};
        for(int k=0;k<12;k++)uv[k]*=scale;
        float trial[12], smaller[12], xyz[18];
        int32_t faces[6] = {0,1,2,3,4,5};
        for (int v = 0; v < 6; v++) { xyz[3*v] = uv[2*v]; xyz[3*v+1] = uv[2*v+1]; xyz[3*v+2] = 0; }
        AsmChart chart = {0}; chart.nv = 6; chart.nf = 2; chart.uv = uv; chart.xyz = xyz; chart.faces = faces;
        chart.flags = mirror ? ASM_CHART_MIRROR : 0;
        AsmRun_push_chart(&run,&chart);
        AsmChart saved = run.charts[0];
        run.charts[0].pose_theta = .41; run.charts[0].pose_x = 3000; run.charts[0].pose_y = -4000;
        uint8_t blocked[1] = {0}, vertices[6] = {0}; uint8_t *vertex_rows[1] = {vertices};
        AsmContinuityAuditCache cache = {0};
        cache.blocked_charts = blocked; cache.blocked_charts_count = 1;
        cache.blocked_vertices = vertex_rows; cache.blocked_vertices_count = 1;
        int ok = apl_uv_audit(&run,&saved,1,&cache);
        memcpy(trial,uv,sizeof uv);
        for (int j = 6; j < 12; j++) trial[j] -= .1f*scale;
        run.charts[0].uv = trial;
        ok = ok && apl_internal_bad(&run,&saved,1,1) == 0 && apl_uv_audit(&run,&saved,1,&cache)==!deep && blocked[0]==deep;
        for (int v = 0; v < 6; v++) ok = ok && vertices[v]==deep;
        run.charts[0].uv = uv;
        ok = ok && apl_uv_audit(&run,&saved,1,&cache) && !blocked[0];
        saved.uv = trial;
        memcpy(smaller,uv,sizeof uv);
        for (int j = 6; j < 12; j++) smaller[j] -= .075f*scale;
        run.charts[0].uv = smaller;
        AsmContinuityAuditCache existing = {0};
        ok = ok && apl_uv_audit(&run,&saved,1,&existing);
        for (int j = 6; j < 12; j++) smaller[j] = uv[j]-.15f*scale;
        ok = ok && apl_uv_audit(&run,&saved,1,&existing)==!deep;
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: triangle self-contact preservation mirror %d\n",mirror); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_single_chart_uv_audit_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float xyz[243], uv[162], contracted[162];
        for (int j = 0; j < 9; j++) for (int u = 0; u < 9; u++) {
            int v = 9*j+u;
            xyz[3*v] = uv[2*v] = (float)(4*u);
            xyz[3*v+1] = uv[2*v+1] = (float)(4*j); xyz[3*v+2] = 0;
            contracted[2*v] = .5f*uv[2*v]+8; contracted[2*v+1] = uv[2*v+1];
        }
        AsmChart c = {0}; c.nv = 81; c.xyz = xyz; c.uv = uv;
        c.flags = mirror ? ASM_CHART_MIRROR : 0; AsmRun_push_chart(&run,&c);
        AsmChart saved = run.charts[0];
        run.charts[0].pose_theta = .4; run.charts[0].pose_x = 100; run.charts[0].pose_y = -20;
        AsmContinuityAuditCache cache = {0};
        int ok = apl_uv_audit(&run,&saved,1,&cache);
        for (int v = 0; v < 81; v++)
            ok = ok && hypot(contracted[2*v]-uv[2*v],contracted[2*v+1]-uv[2*v+1]) <= 8;
        run.charts[0].uv = contracted;
        size_t bad = apl_internal_bad(&run,&saved,1,1);
        ok = ok && bad > 0 && !apl_uv_audit(&run,&saved,1,&cache);
        run.charts[0].uv = uv;
        ok = ok && apl_uv_audit(&run,&saved,1,&cache);
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: single-chart UV contraction mirror %d, bad %zu\n",mirror,bad); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_guarded_uv_audit_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float xyz[2][3] = {{0,0,0},{40,0,0}}, uv[2][2] = {{0,0},{40,0}}, trial[2] = {16,0};
        for (int i = 0; i < 2; i++) {
            AsmChart c = {0}; c.id = i; c.nv = 1; c.xyz = xyz[i]; c.uv = uv[i];
            c.flags = mirror ? ASM_CHART_MIRROR : 0; c.pose_theta = .4;
            AsmRun_push_chart(&run,&c);
        }
        AsmChart saved[2] = {run.charts[0],run.charts[1]};
        uint8_t blocked[2] = {0};
        AsmContinuityAuditCache guard = {0}; guard.guard_charts = saved; guard.guard_count = 2;
        guard.blocked_charts = blocked; guard.blocked_charts_count = 2;
        int ok = apl_uv_audit(&run,saved,1,&guard) && guard.baseline_queries == 2 && !guard.baseline_bad;
        run.charts[0].uv = trial;
        AsmContinuityAuditCache alone = {0};
        ok = ok && apl_uv_audit(&run,saved,1,&alone) && !apl_uv_audit(&run,saved,1,&guard) && blocked[0] && blocked[1];
        run.charts[0].uv = uv[0];
        ok = ok && apl_uv_audit(&run,saved,1,&guard) && !blocked[0] && !blocked[1];
        guard.guard_count = 0;
        ok = ok && !apl_uv_audit(&run,saved,1,&guard);
        guard.guard_charts = NULL; guard.guard_count = 2;
        ok = ok && !apl_uv_audit(&run,saved,1,&guard);
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: fixed neighbour of a UV repair, mirror %d\n",mirror); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_guarded_unsampled_vertex_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) for (int query = 0; query < 2; query++) for (int reference = 0; reference < 2; reference++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float uv[2][6], xyz[2][9], trial[6]; int32_t face[3] = {0,1,2};
        for (int i = 0; i < 2; i++) {
            int tip = i ? reference : query;
            for (int v = 0; v < 3; v++) {
                float u = v == tip ? (i ? 25.0f : 0.0f) : (i ? -100.0f : 100.0f);
                float w = v == tip ? 0.0f : (v == 2 ? -100.0f : 100.0f);
                uv[i][2*v] = xyz[i][3*v] = u; uv[i][2*v+1] = xyz[i][3*v+1] = w;
                xyz[i][3*v+2] = (float)(40*i);
            }
            AsmChart c = {0}; c.id = i; c.nv = 3; c.nf = 1; c.xyz = xyz[i]; c.uv = uv[i]; c.faces = face;
            c.flags = mirror ? ASM_CHART_MIRROR : 0; c.pose_theta = .4; c.pose_x = 100; c.pose_y = -50;
            AsmRun_push_chart(&run,&c);
        }
        AsmChart saved[2] = {run.charts[0],run.charts[1]};
        uint8_t blocked[2] = {0}, points[2][5] = {{0xa5,0,0,0,0x5a},{0xa5,0,0,0,0x5a}};
        uint8_t *rows[2] = {points[0]+1,points[1]+1};
        AsmContinuityAuditCache guard = {0}; guard.guard_charts = saved; guard.guard_count = 2;
        guard.blocked_charts = blocked; guard.blocked_charts_count = 2;
        guard.blocked_vertices = rows; guard.blocked_vertices_count = 2;
        AsmContinuityUvPair pairs[3], sentinel;
        memset(&sentinel,0xa5,sizeof sentinel); for (int i = 0; i < 3; i++) pairs[i] = sentinel;
        guard.pairs = pairs+1; guard.pairs_capacity = 1;
        int ok = apl_uv_audit(&run,saved,2,&guard) && !guard.baseline_bad;
        ok = ok && !guard.pairs_count;
        memcpy(trial,uv[0],sizeof trial); trial[2*query] += 2;
        run.charts[0].uv = trial;
        int accepted = apl_uv_audit(&run,saved,2,&guard);
        ok = ok && !accepted && blocked[0] && blocked[1] && rows[0][query] && rows[1][reference];
        ok = ok && guard.pairs_count == 1 && pairs[1].chart_a == 0 && pairs[1].vertex_a == query &&
             pairs[1].chart_b == 1 && pairs[1].vertex_b == reference &&
             !memcmp(pairs,&sentinel,sizeof sentinel) && !memcmp(pairs+2,&sentinel,sizeof sentinel);
        for (int i = 0; i < 2; i++) {
            ok = ok && points[i][0] == 0xa5 && points[i][4] == 0x5a;
            for (int v = 0; v < 3; v++) ok = ok && rows[i][v] == (v == (i ? reference : query));
        }
        guard.blocked_charts = NULL; guard.blocked_vertices = NULL;
        ok = ok && !apl_uv_audit(&run,saved,2,&guard) && guard.pairs_count == 1;
        guard.pairs_capacity = 0; pairs[1] = sentinel;
        ok = ok && !apl_uv_audit(&run,saved,2,&guard) && !guard.pairs_count && !memcmp(pairs+1,&sentinel,sizeof sentinel);
        guard.pairs_capacity = 1; guard.blocked_charts = blocked; guard.blocked_vertices = rows;
        run.charts[0].uv = uv[0];
        ok = ok && apl_uv_audit(&run,saved,2,&guard) && !blocked[0] && !blocked[1];
        ok = ok && !guard.pairs_count;
        if (!ok) {
            fprintf(stderr,"  asm_place selftest FAIL: unsampled UV collision, mirror %d query %d reference %d, accepted %d\n",
                    mirror,query,reference,accepted); fails++;
        }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_guarded_pair_identity_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float uv[2][10] = {{0}}, xyz[2][15] = {{0}}, trial[10];
        for (int i = 0; i < 2; i++) {
            for (int v = 0; v < 5; v++) {
                uv[i][2*v] = xyz[i][3*v] = v == 4 ? 25.0f : 0.0f;
                xyz[i][3*v+2] = (float)(40*i);
            }
            AsmChart c = {0}; c.id = i; c.nv = 5; c.xyz = xyz[i]; c.uv = uv[i];
            c.flags = mirror ? ASM_CHART_MIRROR : 0; c.pose_theta = .4;
            c.pose_x = 100; c.pose_y = -50;
            AsmRun_push_chart(&run,&c);
        }
        AsmChart saved[2] = {run.charts[0],run.charts[1]};
        AsmContinuityAuditCache legacy = {0}, guard = {0};
        uint8_t blocked[4] = {0xa5,0xcc,0xcc,0x5a};
        uint8_t points[2][7]; memset(points,0xcc,sizeof points);
        for (int i = 0; i < 2; i++) { points[i][0] = 0xa5; points[i][6] = 0x5a; }
        uint8_t *point_rows[2] = {points[0]+1,points[1]+1};
        guard.guard_charts = saved; guard.guard_count = 2;
        guard.blocked_charts = blocked+1; guard.blocked_charts_count = 2;
        guard.blocked_vertices = point_rows; guard.blocked_vertices_count = 2;
        size_t before = apl_internal_bad(&run,saved,2,0);
        int ok = before == 6 && apl_uv_audit(&run,saved,2,&guard);
        memcpy(trial,uv[1],sizeof trial); trial[8] = 23;
        run.charts[1].uv = trial;
        size_t after = apl_internal_bad(&run,saved,2,1);
        int old_accept = apl_uv_audit(&run,saved,2,&legacy);
        int guarded_accept = apl_uv_audit(&run,saved,2,&guard);
        ok = ok && after == before && old_accept && !guarded_accept &&
             blocked[0] == 0xa5 && blocked[1] && blocked[2] && blocked[3] == 0x5a;
        for (int i = 0; i < 2; i++) for (int v = 0; v < 5; v++)
            ok = ok && point_rows[i][v] == (i ? v == 4 : v < 4);
        run.charts[1].uv = uv[1];
        ok = ok && apl_uv_audit(&run,saved,2,&guard) && !blocked[1] && !blocked[2];
        for (int i = 0; i < 2; i++) {
            ok = ok && points[i][0] == 0xa5 && points[i][6] == 0x5a;
            for (int v = 0; v < 5; v++) ok = ok && !point_rows[i][v];
        }
        if (!ok) {
            fprintf(stderr,"  asm_place selftest FAIL: additional bad reference pairs at already bad queries, mirror %d, bad %zu -> %zu, accepted legacy %d guarded %d\n",
                    mirror,before,after,old_accept,guarded_accept); fails++;
        }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_polish_bound_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 14;
        float ref_uv[2] = {0,0}, ref_xyz[3] = {0,0,0};
        float uv[34] = {0}, original[34] = {0}, xyz[51] = {0};
        float far_uv[62], far_xyz[93] = {0};
        for (int i = 0; i < 17; i++) xyz[3*i] = 80;
        for (int i = 0; i < 31; i++) { far_uv[2*i] = 1000; far_uv[2*i+1] = 0; }
        for (int i = 0; i < 3; i++) {
            AsmChart c = {0}; c.id = i; c.flags = mirror ? ASM_CHART_MIRROR : 0;
            c.nv = i == 0 ? 1 : i == 1 ? 17 : 31;
            c.uv = i == 0 ? ref_uv : i == 1 ? uv : far_uv;
            c.xyz = i == 0 ? ref_xyz : i == 1 ? xyz : far_xyz;
            c.pose_theta = .43; c.pose_x = 100; c.pose_y = -50;
            AsmRun_push_chart(&run,&c);
        }
        AplRimGrid *grid = aprg_new(run.arena,24);
        aprg_insert_chart(grid,&run.charts[0]);
        AsmChart saved[2] = {run.charts[1],run.charts[2]}; saved[0].uv = original;
        const AsmChart *candidate = &run.charts[1];
        size_t guaranteed, queries, touched;
        int ok = aprg_polish_blocked(grid,&run,saved,1,NULL,.4,&guaranteed,&queries) &&
                 guaranteed == 8 && queries == 9;
        for (int j = 0; j < 64 && ok; j++) {
            double angle = 2*APL_PI*j/64;
            for (int i = 0; i < 17; i++) { uv[2*i] = (float)(8*cos(angle)); uv[2*i+1] = (float)(8*sin(angle)); }
            ok = aprg_probe(grid,&run,&candidate,1,24,&touched) == 1 && touched == 9;
        }
        memset(uv,0,sizeof uv);
        ok = ok && !aprg_polish_blocked(grid,&run,saved,2,NULL,.4,&guaranteed,&queries) && queries == 25;
        uint8_t excluded[3] = {1,0,0};
        ok = ok && !aprg_polish_blocked(grid,&run,saved,1,excluded,.4,&guaranteed,&queries);
        run.charts[1].nv = saved[0].nv = 13;
        ok = ok && !aprg_polish_blocked(grid,&run,saved,1,NULL,.4,&guaranteed,&queries);
        run.charts[1].nv = saved[0].nv = 17;
        for (int i = 0; i < 17; i++) { original[2*i] = 17; uv[2*i] = 9; }
        ok = ok && !aprg_polish_blocked(grid,&run,saved,1,NULL,.4,&guaranteed,&queries);
        for (int i = 0; i < 17; i++) uv[2*i] = 25;
        ok = ok && aprg_probe(grid,&run,&candidate,1,24,&touched) == 0 && !touched;
        memset(uv,0,sizeof uv); memset(original,0,sizeof original);
        for (int i = 0; i < 17; i++) xyz[3*i] = 15;
        ok = ok && !aprg_polish_blocked(grid,&run,saved,1,NULL,.4,&guaranteed,&queries);
        for (int i = 0; i < 17; i++) { original[2*i] = 1; uv[2*i] = 9; }
        ok = ok && aprg_probe(grid,&run,&candidate,1,24,&touched) == 0 && touched == 9;
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: conservative UV polish rim bound, mirror %d\n",mirror); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_internal_overlap_exchange_selftest(void)
{
    int fails = 0;
    for (int mirror = 0; mirror < 2; mirror++) {
        AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 10;
        float uv[50], xyz[4][75];
        for (int v = 0; v < 25; v++) {
            uv[2*v] = (float)(2*(v%5)); uv[2*v+1] = (float)(2*(v/5));
            for (int i = 0; i < 4; i++) {
                xyz[i][3*v] = (float)(12*i);
                xyz[i][3*v+1] = uv[2*v]; xyz[i][3*v+2] = uv[2*v+1];
            }
        }
        for (int i = 0; i < 4; i++) {
            AsmChart c = {0}; c.id = i; c.nv = 25; c.xyz = xyz[i]; c.uv = uv;
            c.pose_x = i < 2 ? 0 : i == 2 ? 100 : 200;
            c.flags = mirror ? ASM_CHART_MIRROR : 0; AsmRun_push_chart(&run,&c);
        }
        AsmChart saved[4]; memcpy(saved,run.charts,sizeof saved);
        size_t before = apl_internal_bad(&run,saved,4,0);
        run.charts[1].pose_x = 50; run.charts[3].pose_x = 100;
        size_t after = apl_internal_bad(&run,saved,4,1);
        AsmContinuityAuditCache uv_cache = {0}, registration_cache = {0};
        uint8_t blocked[] = {0xa5,0xcc,0xcc,0xcc,0xcc,0x5a};
        uv_cache.blocked_charts = blocked+1; uv_cache.blocked_charts_count = 4;
        int uv_accepted = apl_uv_audit(&run,saved,4,&uv_cache);
        int registration_accepted = apl_registration_audit(&run,saved,4,&registration_cache);
        int ok = before > 0 && after == before && !uv_accepted && !registration_accepted &&
                 blocked[0] == 0xa5 && blocked[1] == 0 && blocked[2] == 0 &&
                 blocked[3] == 1 && blocked[4] == 1 && blocked[5] == 0x5a;
        run.charts[3].pose_x = 200;
        ok = ok && !apl_internal_bad(&run,saved,4,1) && apl_uv_audit(&run,saved,4,&uv_cache) &&
             apl_registration_audit(&run,saved,4,&registration_cache);
        for (int i = 1; i <= 4; i++) ok = ok && !blocked[i];
        if (!ok) {
            fprintf(stderr,"  asm_place selftest FAIL: overlap exchanged between regions, mirror %d, bad %zu -> %zu, accepted UV %d registration %d\n",
                    mirror,before,after,uv_accepted,registration_accepted); fails++;
        }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_parallel_rim_selftest(void)
{
    enum { N = 65, NV = N*N };
    AsmRun run = {0}; run.arena = Arena_new(); run.layer_d_global = 12;
    float *uv = ARENA_ALLOC(run.arena,2*NV*sizeof *uv);
    float *xyz[2];
    for (int i = 0; i < 2; i++) xyz[i] = ARENA_ALLOC(run.arena,3*NV*sizeof(float));
    for (int v = 0; v < NV; v++) {
        uv[2*v] = (float)(2*(v%N)); uv[2*v+1] = (float)(2*(v/N));
        for (int i = 0; i < 2; i++) {
            xyz[i][3*v] = uv[2*v]; xyz[i][3*v+1] = uv[2*v+1];
            xyz[i][3*v+2] = (float)(i && v/N >= N/2 ? 12 : 0);
        }
    }
    for (int i = 0; i < 2; i++) {
        AsmChart c = {0}; c.id = c.component = i; c.nv = NV; c.xyz = xyz[i]; c.uv = uv;
        AsmRun_push_chart(&run,&c);
    }
    AplRimGrid *grid = aprg_new(run.arena,24);
    aprg_insert_chart(grid,&run.charts[0]);
    const AsmChart *candidate = &run.charts[1]; int32_t orig[2] = {0,1};
    const double offsets[] = {0,23.999999999999,24,64,300};
    int fails = 0; size_t touched = 0;
    for (size_t i = 0; i < sizeof offsets/sizeof offsets[0]; i++) {
        run.charts[1].pose_x = offsets[i]; size_t ref_n, actual_n;
        double reference = apl_rim_audit(&run,orig,1,0,24,&ref_n);
        double actual = aprg_probe(grid,&run,&candidate,1,24,&actual_n);
        touched += actual_n;
        if (ref_n != actual_n || reference != actual) {
            fprintf(stderr,"  asm_place selftest FAIL: parallel rim at %.17g: %zu/%.17g vs %zu/%.17g\n",
                    offsets[i],actual_n,actual,ref_n,reference); fails++;
        }
    }
    uint8_t excluded[2] = {1,0}; size_t count = 0;
    run.charts[1].pose_x = 0;
    size_t queries = (NV+1)/2, bad_count = 0, recorded_n = 0;
    uint8_t *record = ARENA_ALLOC(run.arena,2*queries+2);
    memset(record,0xcc,2*queries+2); record[0] = 0xa5; record[2*queries+1] = 0x5a;
    int32_t *partners = ARENA_ALLOC(run.arena,(2*queries+2)*sizeof *partners);
    for (size_t i = 0; i < 2*queries+2; i++) partners[i] = -2;
    partners[0] = 12345; partners[2*queries+1] = 54321;
    const AsmChart *both[] = {&run.charts[0],&run.charts[1]};
    double recorded = aprg_probe_record(grid,&run,both,2,24,&recorded_n,NULL,record+1,partners+1,NULL,NULL,2);
    int field_ok = recorded_n == 2*queries && record[0] == 0xa5 && record[2*queries+1] == 0x5a &&
                   partners[0] == 12345 && partners[2*queries+1] == 54321;
    for (int i = 0; i < 2; i++) for (int v = 0; v < NV; v += 2) {
        int expected = i == 1 && v/N >= N/2;
        field_ok = field_ok && record[1+i*queries+(size_t)v/2] == expected;
        field_ok = field_ok && partners[1+i*queries+(size_t)v/2] == (expected ? 0 : -1);
        bad_count += (size_t)expected;
    }
    field_ok = field_ok && recorded == (double)bad_count/(2*queries);
    if (!field_ok) { fprintf(stderr,"  asm_place selftest FAIL: parallel rim violation field or bounds\n"); fails++; }
    if (!touched || aprg_probe_excluding(grid,&run,&candidate,1,24,&count,excluded) != 0 || count != 0) {
        fprintf(stderr,"  asm_place selftest FAIL: parallel rim exclusion or empty support\n"); fails++;
    }
    Arena_dispose(&run.arena); return fails;
}

static int apl_axis_side_selftest(void)
{
    int fails=0;
    for (int world=0; world<2; world++) for (int mirror=0; mirror<2; mirror++) for (int parity=0; parity<2; parity++) {
        AsmRun run={0}; run.arena=Arena_new();
        enum { N=9, NV=N*N };
        float xyz[3*NV],normal[3*NV],uv[2*NV];
        double angle=world ? .4 : 0, ca=cos(angle),sa=sin(angle);
        double origin[3]={world ? 4800 : 0,world ? 3400 : 0,world ? 2900 : 0};
        double dir[3]={ca,0,sa}; int sign=parity ? -1 : 1;
        AsmChart c={0}; c.id=0; c.nv=NV; c.area3d=1600;
        c.xyz=xyz; c.nrm=normal; c.uv=uv; c.flags=mirror ? ASM_CHART_MIRROR : 0;
        for (int j=0; j<N; j++) for (int i=0; i<N; i++) {
            int v=j*N+i; double u=5*(i-4),z=5*(j-4),x=200+.01*u*u;
            double ny=-.02*u,nx=1,nl=hypot(ny,nx);
            double p[3]={ca*z-sa*x,u,sa*z+ca*x};
            double n[3]={-sa*nx,ny,ca*nx};
            for (int d=0; d<3; d++) {
                xyz[3*v+d]=(float)(origin[d]+p[d]); normal[3*v+d]=(float)(sign*n[d]/nl);
                c.centroid[d]+=xyz[3*v+d]/(double)NV;
            }
            uv[2*v]=(float)u; uv[2*v+1]=(float)z;
        }
        AsmAxis *axis=AsmAxis_line(run.arena,origin,dir,origin[0]-500,origin[0]+500);
        run.charts=&c; run.n_charts=1;
        int8_t side[1]={0};
        apl_component_out_sides(&run,0,NULL,side);
        int ok=side[0]==-sign;
        apl_component_out_sides(&run,0,axis,side);
        ok=ok && side[0]==sign;
        AplTurn turn[2]={{200,0,16},{200,0,16}}; AplOrient orientation={0};
        apl_orient_votes(&run,0,axis,turn,&orientation,side);
        ok=ok && side[0]==sign && orientation.has_turn && orientation.turn_u>.99 &&
            (mirror ? orientation.bend>.9 : orientation.bend<-.9);
        for (int v=0; v<NV/2; v++) for (int d=0; d<3; d++) normal[3*v+d]=-normal[3*v+d];
        double inward[3]={0};
        ok=ok && !apl_axis_inward(&c,axis,inward);
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: dented sheet outward side world %d mirror %d normal %d, side %d bend %.6g\n",world,mirror,parity,side[0],orientation.bend); fails++; }
        Arena_dispose(&run.arena);
    }
    return fails;
}

static int apl_stretch_error_selftest(void)
{
    enum { ROWS = 3, COLS = 20, NV = ROWS * COLS, NF = 2 * (ROWS - 1) * (COLS - 1), NH = COLS - 1, PAIRS = 4 };
    int fails = 0;
    double rel_clean = -1.0, rel_noisy = -1.0, rel_dup = -1.0, t_clean = 0.0;
    const double ra = 240.0, d = 12.0, du = 20.0, dv = 20.0;
    const double s_true = (ra + d) / ra;
    for (int cs = 0; cs < 3; cs++) {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[2]; memset(charts, 0, sizeof charts);
        float uv[2][2*NV], xyz[2][3*NV];
        int32_t faces[3*NF];
        for (int k = 0; k < 2; k++) {
            double r = k ? ra + d : ra;
            for (int row = 0; row < ROWS; row++) for (int col = 0; col < COLS; col++) {
                int v = row * COLS + col;
                double arc = du * (double)col * (k ? s_true : 1.0);
                double noise = (k && cs != 0) ? ((col & 1) ? 5.0 : -5.0) : 0.0;
                uv[k][2*v] = (float)(arc + noise); uv[k][2*v+1] = (float)(dv * (double)row);
                double th = arc / r;
                xyz[k][3*v]   = (float)(dv * (double)row);
                xyz[k][3*v+1] = (float)(r * sin(th));
                xyz[k][3*v+2] = (float)(r * cos(th));
            }
        }
        for (int row = 0, f = 0; row < ROWS - 1; row++) for (int col = 0; col < COLS - 1; col++) {
            int a = row * COLS + col, b = a + 1, c = a + COLS, e = c + 1;
            faces[3*f] = a; faces[3*f+1] = b; faces[3*f+2] = c; f++;
            faces[3*f] = b; faces[3*f+1] = e; faces[3*f+2] = c; f++;
        }
        for (int k = 0; k < 2; k++) {
            charts[k].id = k; charts[k].component = k; charts[k].nv = NV; charts[k].nf = NF;
            charts[k].uv = uv[k]; charts[k].xyz = xyz[k]; charts[k].faces = faces; charts[k].area3d = 1000.0;
        }
        int dup = cs == 2;
        AsmLayerPair pairs[PAIRS]; memset(pairs, 0, sizeof pairs);
        AsmLayerHit hits[2*NH]; memset(hits, 0, sizeof hits);
        int nh_written = 0;
        for (int q = 0; q < PAIRS; q++) {
            int lo = q * NH / PAIRS, hi = (q + 1) * NH / PAIRS;
            pairs[q].a = 0; pairs[q].b = 1; pairs[q].side = 1; pairs[q].k = 1; pairs[q].d_median = (float)d;
            pairs[q].hit_first = nh_written;
            for (int i = lo; i < hi; i++) for (int rep = 0; rep < (dup ? 2 : 1); rep++) {
                AsmLayerHit *h = &hits[nh_written++];
                h->a = 0; h->b = 1; h->side = 1; h->order = 1;
                h->va = (int32_t)i; h->fb = (int32_t)(2*i); h->l0 = 1.0f; h->l1 = 0.0f;
                h->dist = (float)d;
            }
            pairs[q].hit_count = nh_written - pairs[q].hit_first;
        }
        run.charts = charts; run.n_charts = 2; run.layers = pairs; run.n_layers = PAIRS;
        run.layer_hits = hits; run.n_layer_hits = (size_t)nh_written; run.layer_d_global = d;
        uint8_t placed[2] = { 1, 0 };
        int8_t sides[2] = { 1, 1 };
        AplCtx x; memset(&x, 0, sizeof x);
        x.run = &run; x.comp_placed = placed; x.out_side = sides;
        double T = 0.0, sl = 0.0, dbar = 0.0, rel = -1.0; size_t nh = 0, np = 0;
        int got = apl_stretch_turn(&x, 1, &T, &sl, &dbar, &nh, &rel, &np);
        if (!got || np != (size_t)PAIRS || nh != (size_t)nh_written) {
            fprintf(stderr, "  asm_place stretch FAIL case %d: got %d, s %.4f, %zu hits on %zu pairs\n", cs, got, sl, nh, np);
            fails++; Arena_dispose(&run.arena); continue;
        }
        if (cs == 0) { rel_clean = rel; t_clean = fabs(T); }
        else if (cs == 1) rel_noisy = rel;
        else rel_dup = rel;
        Arena_dispose(&run.arena);
    }
    if (fabs(t_clean - 2.0 * APL_PI * ra) > 2.0) {
        fprintf(stderr, "  asm_place stretch FAIL: clean turn %.1f, want %.1f\n", t_clean, 2.0 * APL_PI * ra);
        fails++;
    }
    if (!(rel_clean >= 0.0 && rel_clean < 0.01)) {
        fprintf(stderr, "  asm_place stretch FAIL: a noiseless fit must know (s-1) exactly, relative error %.4f\n", rel_clean);
        fails++;
    }
    if (!(rel_noisy > 0.5)) {
        fprintf(stderr, "  asm_place stretch FAIL: a 5-vox sawtooth must leave (s-1) unknown, relative error %.4f\n", rel_noisy);
        fails++;
    }
    if (!(rel_dup >= 0.0 && fabs(rel_dup - rel_noisy) <= 1e-9 * fmax(1.0, rel_noisy))) {
        fprintf(stderr, "  asm_place stretch FAIL: recording every hit twice changed the relative error %.6f -> %.6f\n", rel_noisy, rel_dup);
        fails++;
    }
    if (!fails) fprintf(stderr, "  asm_place stretch error: turn %.0f vox, (s-1) known to %.2f%% clean, %.0f%% noisy, %.0f%% duplicated (ok)\n",
                        t_clean, 100.0*rel_clean, 100.0*rel_noisy, 100.0*rel_dup);
    return fails;
}

static int apl_fit_turn_radius_selftest(void)
{
    enum { NC = 40 };
    const double R0 = 300.0, d = 12.0, C = 600.0;
    int fails = 0;
    for (int cs = 0; cs < 3; cs++) {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[NC]; memset(charts, 0, sizeof charts);
        AplTurn turn[2 * NC]; memset(turn, 0, sizeof turn);
        int8_t sides[NC]; double r_axis[NC];
        float uv[6] = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f };
        size_t nc = cs == 2 ? 4 : NC;
        for (size_t i = 0; i < nc; i++) {
            double r = R0 + (double)(i % 3) * d;
            charts[i].id = (int32_t)i; charts[i].component = 0; charts[i].nv = 3; charts[i].nf = 1;
            charts[i].area3d = 100.0; charts[i].uv = uv;
            sides[i] = 1; r_axis[i] = r;
            double L = 2.0 * APL_PI * r + C;
            if (cs == 1 && (i % 6) == 0) L *= 2.0;
            turn[i * 2 + 1].tx = L; turn[i * 2 + 1].ty = 0.0; turn[i * 2 + 1].n = 4;
        }
        run.charts = charts; run.n_charts = nc; run.layer_d_global = d;
        AplCtx x; memset(&x, 0, sizeof x);
        x.run = &run; x.turn = turn; x.out_side = sides; x.r_axis = r_axis;
        x.have_radius = 1; x.uout_ok = 1; x.uout_x = 1.0; x.uout_y = 0.0;
        apl_fit_turn_radius(&x);
        if (cs == 2) {
            if (x.turn_c != 0.0 || x.turn_c_n != 0) {
                fprintf(stderr, "  asm_place turn-c FAIL case 2: adopted c %.1f over %zu charts on too little support\n", x.turn_c, x.turn_c_n);
                fails++;
            }
            Arena_dispose(&run.arena); continue;
        }
        if (!(fabs(x.turn_c - C) <= 0.05 * C)) {
            fprintf(stderr, "  asm_place turn-c FAIL case %d: c %.1f, want %.1f +/- 5%%\n", cs, x.turn_c, C);
            fails++;
        }
        if (!(x.turn_c_n >= (size_t)ASM_PLACE_TURN_C_MIN_N)) {
            fprintf(stderr, "  asm_place turn-c FAIL case %d: support %zu under the %d bar\n", cs, x.turn_c_n, ASM_PLACE_TURN_C_MIN_N);
            fails++;
        }
        if (cs == 0 && !(fabs(x.turn_slope - 2.0 * APL_PI) <= 0.03 * 2.0 * APL_PI)) {
            fprintf(stderr, "  asm_place turn-c FAIL: free slope %.4f, want %.4f +/- 3%%\n", x.turn_slope, 2.0 * APL_PI);
            fails++;
        }
        if (cs == 0) {
            double tx = 0.0, ty = 0.0;
            int got = apl_turn_radius(&x, 0, 1, 2, &tx, &ty);
            double rbar = r_axis[0] + 2.0 * d * 0.5;
            double want = 2.0 * (2.0 * APL_PI * rbar) + (ASM_PLACE_TURN_INTERCEPT ? 2.0 * C : 0.0);
            if (!got || !(fabs(fabs(tx) - want) <= 0.01 * want) || ty != 0.0) {
                fprintf(stderr, "  asm_place turn-c FAIL: apl_turn_radius k=2 gave %.1f (got %d, ty %.3f), want %.1f at intercept mode %d\n",
                        fabs(tx), got, ty, want, ASM_PLACE_TURN_INTERCEPT);
                fails++;
            }
            double growth = 0.0;
            apl_growth_for_hops(&x, &run, turn, sides, &growth);
            double rule = 2.0 * APL_PI * d;
            int growth_ok = ASM_PLACE_GROWTH_PER_WRAP == 0 ? growth > 0.0
                          : ASM_PLACE_GROWTH_PER_WRAP == 1 ? fabs(growth - rule) < 1e-9
                          : growth >= 0.8 * rule - 1e-9 && growth <= 1.25 * rule + 1e-9;
            if (!growth_ok || growth != growth) {
                fprintf(stderr, "  asm_place turn-c FAIL: growth %.3f at mode %d (2 pi d = %.3f)\n",
                        growth, ASM_PLACE_GROWTH_PER_WRAP, rule);
                fails++;
            }
        }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart one; memset(&one, 0, sizeof one);
        AplTurn t2[2]; memset(t2, 0, sizeof t2);
        int8_t side = 1; double r0 = 0.0;
        float uv1[6] = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f };
        one.id = 0; one.nv = 3; one.nf = 1; one.area3d = 1.0; one.uv = uv1;
        t2[1].tx = 1000.0; t2[1].n = 2;
        run.charts = &one; run.n_charts = 1; run.layer_d_global = d;
        AplCtx x; memset(&x, 0, sizeof x);
        x.run = &run; x.turn = t2; x.out_side = &side; x.r_axis = &r0; x.have_radius = 1; x.uout_ok = 1; x.uout_x = 1.0;
        apl_fit_turn_radius(&x);
        if (x.turn_c != 0.0 || x.turn_c_n != 0 || x.turn_slope != x.turn_slope) {
            fprintf(stderr, "  asm_place turn-c FAIL degenerate: c %.1f n %zu slope %.3f\n", x.turn_c, x.turn_c_n, x.turn_slope);
            fails++;
        }
        x.have_radius = 0;
        apl_fit_turn_radius(&x);
        if (x.turn_c != 0.0) { fprintf(stderr, "  asm_place turn-c FAIL: fitted without a radius model\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    if (!fails) fprintf(stderr, "  asm_place section perimeter excess (intercept mode %d, growth mode %d): ok\n",
                        ASM_PLACE_TURN_INTERCEPT, ASM_PLACE_GROWTH_PER_WRAP);
    return fails;
}

static int apl_crosswrap_evidence_selftest(void)
{
    int fails = 0;
    AsmRelation r; memset(&r, 0, sizeof r); r.corr_count = 32; r.rms = 0.5;
    r.flags = ASM_REL_WEAK;
    if (apl_placement_seam(&r, 0, ASM_WRAP_VETO_ANCHOR) != 1) { fprintf(stderr, "  asm_place crosswrap FAIL: a WEAK seam must anchor\n"); fails++; }
    r.flags = ASM_REL_WEAK | ASM_REL_CROSSWRAP;
    if (apl_placement_seam(&r, 0, ASM_WRAP_VETO_ANCHOR) != (ASM_WRAP_VETO_ANCHOR ? 0 : 1)) {
        fprintf(stderr, "  asm_place crosswrap FAIL: on an unframed pile a CROSSWRAP seam anchors against ASM_WRAP_VETO_ANCHOR %d\n", ASM_WRAP_VETO_ANCHOR);
        fails++;
    }
    if (apl_placement_seam(&r, 1, ASM_WRAP_VETO_ANCHOR) != (ASM_WRAP_VETO_ANCHOR == 1 ? 0 : 1)) {
        fprintf(stderr, "  asm_place crosswrap FAIL: on a framed pile a CROSSWRAP seam does not follow ASM_WRAP_VETO_ANCHOR %d\n", ASM_WRAP_VETO_ANCHOR);
        fails++;
    }
    {
        int want[3][2] = { { 1, 1 }, { 0, 0 }, { 0, 1 } };
        int v = 0, fr = 0;
        for (v = 0; v < 3; v++) for (fr = 0; fr < 2; fr++)
            if (apl_placement_seam(&r, fr, v) != want[v][fr]) {
                fprintf(stderr, "  asm_place crosswrap FAIL: veto %d on a %s pile gave %d, want %d\n",
                        v, fr ? "framed" : "unframed", apl_placement_seam(&r, fr, v), want[v][fr]);
                fails++;
            }
    }
    r.flags = ASM_REL_DROPPED | ASM_REL_CROSSWRAP;
    if (apl_placement_seam(&r, 0, ASM_WRAP_VETO_ANCHOR) != 0) { fprintf(stderr, "  asm_place crosswrap FAIL: a DROPPED seam must never anchor\n"); fails++; }
    return fails;
}

#include "asm_place_strict_test.inc"

int AsmPlace_selftest(void)
{
    int fails = apl_fit_turn_radius_selftest() + apl_crosswrap_evidence_selftest() + apl_stretch_error_selftest() + apl_rotated_cluster_selftest() + apl_v_model_update_selftest() + apl_self_contact_audit_selftest() + apl_axis_side_selftest() + apl_stopped_repair_selftest() + apl_seam_incumbent_selftest() + apl_registration_planes_selftest() + apl_guarded_unsampled_vertex_selftest() + apl_guarded_pair_identity_selftest() + apl_guarded_uv_audit_selftest() + apl_polish_bound_selftest() + apl_internal_overlap_exchange_selftest() + apl_parallel_rim_selftest() + apl_single_chart_uv_audit_selftest() + apl_rim_pair_selftest() + apl_registration_selftest() + apl_reconsider_selftest() + apl_uv_transaction_selftest(10) + apl_uv_transaction_selftest(5);
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[2]; memset(charts, 0, sizeof charts); float uv[12], xyz[18];
        for (int i = 0; i < 6; i++) {
            uv[2*i] = xyz[3*i] = (float)(4*i); uv[2*i+1] = xyz[3*i+1] = xyz[3*i+2] = 0;
        }
        for (int i = 0; i < 2; i++) { charts[i].id = charts[i].component = i; charts[i].uv = uv; charts[i].xyz = xyz; charts[i].nv = 6; charts[i].area3d = 1000; }
        charts[0].placement_state = ASM_PLACE_ROOT; charts[1].pose_y = 12;
        AsmCorr corr[6]; memset(corr, 0, sizeof corr);
        for (int i = 0; i < 6; i++) { corr[i].va = corr[i].vb = i; corr[i].valid = 3; corr[i].run = 0; }
        AsmRelation rel; memset(&rel, 0, sizeof rel); rel.b = 1; rel.corr_count = 6; rel.continuity = ASM_CONT_SOURCE;
        run.charts = charts; run.n_charts = 2; run.rels = &rel; run.n_rels = 1; run.corr = corr; run.n_corr = 6;
        AplAnchor anchors[92]; memset(anchors, 0, sizeof anchors); double weights[92];
        for (int i = 0; i < 92; i++) {
            anchors[i].chart_c = 1; anchors[i].k = anchors[i].measured_order = 1; anchors[i].w = weights[i] = 1;
            anchors[i].local_u = anchors[i].tx = i%12; anchors[i].ty = i < 12 ? 10 : 200;
        }
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts);
        uint8_t selected[1] = {0}; int32_t groups[2] = {0,1}; AsmChart proposal[1];
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.o = &opts; x.anchors = anchors; x.wts = weights; x.na = 92;
        x.orig_comp = groups; x.seam_selected = selected; x.trial = 1; x.proposal = proposal;
        AsmChart before = charts[1];
        int ok = !apl_place_and_audit(&x, 1, 0, 0, 0, NULL, 2) && memcmp(&charts[1], &before, sizeof before) == 0;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: incompatible model repair cannot override source\n"); fails++; }
        x.na = 12; charts[1].pose_y = 28; before = charts[1];
        for (int i = 0; i < 12; i++) anchors[i].local_u = anchors[i].tx = 4*(i%6);
        ok = apl_place_and_audit(&x, 1, 0, 0, 0, NULL, 2) && proposal[0].placement_support == 12 &&
             fabs(proposal[0].pose_y) < 2 && memcmp(&charts[1], &before, sizeof before) == 0;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: unsupported proposal can recover measured agreement\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AplAnchor anchors[16]; memset(anchors, 0, sizeof anchors); double weights[16];
        double cu = 350, cv = -80, slope = 0.02, truth = 0.2;
        double axis_v = sin(truth)*cu+cos(truth)*cv-20-slope*(cos(truth)*cu-sin(truth)*cv+1500);
        for (int i = 0; i < 16; i++) {
            weights[i] = 1; anchors[i].sx = cu+100; anchors[i].sy = cv+4*i-30;
            anchors[i].tx = cos(truth)*anchors[i].sx-sin(truth)*anchors[i].sy+1500;
            anchors[i].ty = sin(truth)*anchors[i].sx+cos(truth)*anchors[i].sy-20;
        }
        double theta, tx, ty;
        double rms = apl_fit_axis_translation(anchors, weights, 16, 0.6, cu, cv, axis_v, slope, 0, &theta, &tx, &ty);
        int ok = rms < 1e-4 && fabs(theta-truth) < 1e-5;
        for (int i = 0; i < 16; i++) anchors[i].ty += 20;
        double free_th, free_tx, free_ty;
        apl_fit_translation(anchors, weights, 16, 0.6, &free_th, &free_tx, &free_ty);
        free_ty = axis_v+slope*(cos(free_th)*cu-sin(free_th)*cv+free_tx)-sin(free_th)*cu-cos(free_th)*cv;
        double old_e2 = 0;
        for (int i = 0; i < 16; i++) {
            double x = cos(free_th)*anchors[i].sx-sin(free_th)*anchors[i].sy+free_tx-anchors[i].tx;
            double y = sin(free_th)*anchors[i].sx+cos(free_th)*anchors[i].sy+free_ty-anchors[i].ty;
            old_e2 += x*x+y*y;
        }
        rms = apl_fit_axis_translation(anchors, weights, 16, 0.6, cu, cv, axis_v, slope, 0, &theta, &tx, &ty);
        double mean_u = cos(theta)*cu-sin(theta)*cv+tx, mean_v = sin(theta)*cu+cos(theta)*cv+ty;
        ok = ok && rms < 0.5*sqrt(old_e2/16) && fabs(mean_v-axis_v-slope*mean_u) < 1e-8 && fabs(theta) <= 0.6;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: rigid fit with the height constraint\n"); fails++; }
        ok = 1;
        for (int shifted = 0; shifted < 2; shifted++) {
            double du = shifted ? 10000 : 0, dv = shifted ? -20000 : 0;
            AplAnchor moved[16]; memcpy(moved,anchors,sizeof moved);
            for (int i = 0; i < 16; i++) { moved[i].sx += du; moved[i].sy += dv; }
            rms = apl_fit_axis_translation(moved,weights,16,0.6,cu+du,cv+dv,axis_v,slope,30,&theta,&tx,&ty);
            double eu = 1500-cos(truth)*du+sin(truth)*dv, ev = -sin(truth)*du-cos(truth)*dv;
            ok = ok && rms < 1e-5 && fabs(theta-truth) < 1e-6 && fabs(tx-eu) < 1e-4 && fabs(ty-ev) < 1e-4;
        }
        for (int i = 0; i < 16; i++) anchors[i].ty += 35;
        rms = apl_fit_axis_translation(anchors,weights,16,0.6,cu,cv,axis_v,slope,30,&theta,&tx,&ty);
        mean_u = cos(theta)*cu-sin(theta)*cv+tx; mean_v = sin(theta)*cu+cos(theta)*cv+ty;
        double plane_rms = apl_fit_axis_translation(anchors,weights,16,0.6,cu,cv,axis_v,slope,0,&free_th,&free_tx,&free_ty);
        ok = ok && rms > 0 && rms < plane_rms && fabs(mean_v-axis_v-slope*mean_u-30) < 1e-8 && fabs(theta) <= 0.6;
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: measured height uncertainty and bounded fit\n"); fails++; }
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AplAnchor anchors[16]; memset(anchors, 0, sizeof anchors);
        double weights[16], residuals[16], th, tx, ty; size_t kept;
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.anchors = anchors; x.wts = weights; x.cand_grad = 1;
        for (int i = 0; i < 16; i++) {
            anchors[i].sx = 100; anchors[i].sy = 40*i; anchors[i].w = 1; anchors[i].k = 1;
            anchors[i].tx = cos(0.2)*anchors[i].sx-sin(0.2)*anchors[i].sy+1500;
            anchors[i].ty = sin(0.2)*anchors[i].sx+cos(0.2)*anchors[i].sy-20;
        }
        double rms = apl_fit_cluster(&x, 16, 0, 12, 0.6, &th, &tx, &ty, &kept, residuals);
        if (rms > 1e-8 || kept != 16 || fabs(th-0.2) > 1e-8 || fabs(tx-1500) > 1e-8 || fabs(ty+20) > 1e-8) {
            fprintf(stderr, "  asm_place selftest FAIL: two-dimensional rotation evidence\n"); fails++;
        }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[2]; memset(charts, 0, sizeof charts);
        float uv[8] = {0,0,20,0,0,20,20,20}, xyz[2][12];
        for (int k = 0; k < 2; k++) {
            charts[k].id = k; charts[k].uv = uv; charts[k].nv = 4; charts[k].xyz = xyz[k]; charts[k].pose_x = 200*k;
            for (int v = 0; v < 4; v++) { xyz[k][3*v] = uv[2*v]+200*k; xyz[k][3*v+1] = uv[2*v+1]; xyz[k][3*v+2] = 0; }
        }
        run.charts = charts; run.n_charts = 2; run.layer_d_global = 10;
        int ok = apl_internal_bad(&run, charts, 2, 0) == 0;
        charts[1].pose_x = 0;
        size_t overlap = apl_internal_bad(&run, charts, 2, 0);
        ok = ok && overlap > 0;
        for (int k = 0; k < 2; k++) apl_compose(&charts[k], 0.7, 300, -100, 1);
        ok = ok && apl_internal_bad(&run, charts, 2, 0) == overlap;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: internal repair collision / rigid invariance\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run = {0}; run.arena = Arena_new();
        AsmChart charts[3] = {0}; float uv[6] = {0,0,10,0,0,10}; int32_t face[3] = {0,1,2};
        for (int i = 0; i < 3; i++) {
            charts[i].id = i; charts[i].uv = uv; charts[i].nv = 3;
            charts[i].faces = face; charts[i].nf = 1; charts[i].pose_x = 1000*i;
        }
        AsmLayerPair pairs[2] = {0}; AsmLayerHit hits[12] = {0};
        for (int i = 0; i < 2; i++) {
            pairs[i].k = (int8_t)(i+1); pairs[i].d_median = i ? 5 : 20;
            pairs[i].hit_first = 6*i; pairs[i].hit_count = 6;
            for (int j = 0; j < 6; j++) {
                AsmLayerHit *h = &hits[6*i+j]; h->va = j%3;
                h->l0 = j%3 == 0; h->l1 = j%3 == 1; h->dist = (float)pairs[i].d_median;
                h->order = (int8_t)(i+1);
            }
        }
        run.charts = charts; run.n_charts = 3; run.layers = pairs; run.n_layers = 2;
        run.layer_hits = hits; run.n_layer_hits = 12;
        uint8_t placed[1] = {1}; int8_t sides[3] = {1,1,1}; AplTurn turns[6]; int ok = 1;
        for (int reverse = 0; reverse < 2; reverse++) {
            for (int i = 0; i < 2; i++) {
                pairs[i].a = reverse ? i+1 : 0; pairs[i].b = reverse ? 0 : i+1;
                pairs[i].side = reverse ? -1 : 1;
            }
            apl_measure_turns(&run,placed,sides,NULL,turns,3);
            ok = ok && turns[1].n == 6 && fabs(turns[1].tx-1000) < 1e-9 && fabs(turns[1].ty) < 1e-9 &&
                 turns[2].n == 6 && fabs(turns[2].tx+1000) < 1e-9 && turns[4].n == 0;
        }
        pairs[0].k = 2;
        apl_measure_turns(&run,placed,sides,NULL,turns,3);
        for (int i = 0; i < 6; i++) ok = ok && turns[i].n == 0;
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: eligible adjacent turn cutoff\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run);
        AsmChart charts[2]; memset(charts, 0, sizeof charts);
        float uv[6] = {0,0,10,0,0,10}; int32_t face[3] = {0,1,2};
        for (int i = 0; i < 2; i++) {
            charts[i].id = charts[i].component = i; charts[i].uv = uv; charts[i].nv = 3;
            charts[i].faces = face; charts[i].nf = 1; charts[i].area3d = 50;
        }
        AsmLayerPair lp; memset(&lp, 0, sizeof lp); lp.b = 1; lp.side = lp.k = 1; lp.hit_count = 1; lp.d_median = 10;
        AsmLayerHit hit; memset(&hit, 0, sizeof hit); hit.l0 = 1; hit.order = 1;
        run.charts = charts; run.n_charts = 2; run.layers = &lp; run.n_layers = 1; run.layer_hits = &hit; run.layer_d_global = 10;
        AsmPlaceOpts opts; AsmPlace_default_opts(&opts); opts.max_layers = 1;
        uint8_t placed[2] = {1,0}; int8_t side[2] = {1,1}; double radius[2] = {100,110};
        AplTurn turns[4]; memset(turns, 0, sizeof turns); turns[1] = (AplTurn){1000,20,12};
        AplAnchor anchors[4]; memset(anchors, 0, sizeof anchors);
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.o = &opts; x.comp_placed = placed;
        x.out_side = side; x.turn = turns; x.anchors = anchors; x.hcap = 4;
        x.have_radius = x.uout_ok = 1; x.uout_x = 1; x.r_axis = radius;
        int ok = 1;
        for (int rev = 0; rev < 2; rev++) {
            lp.a = rev; lp.b = 1-rev; lp.side = rev ? -1 : 1;
            ok = ok && apl_collect_anchors(&x, 1) == 1 && anchors[0].Tx == 1000 && anchors[0].Ty == 20 &&
                 anchors[0].measured_turn && !anchors[0].estimated && anchors[0].reverse == rev;
        }
        memset(turns, 0, sizeof turns);
        ok = ok && apl_collect_anchors(&x, 1) == 1 && fabs(anchors[0].Tx-2*APL_PI*105) < 1e-8 &&
             !anchors[0].measured_turn && anchors[0].estimated;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: measured turn precedence and model provenance\n"); fails++; }
        double perimeter = 0;
        for (int i = 0; i < 4096; i++) {
            double a = 2*APL_PI*((double)i+0.5)/4096;
            perimeter += hypot(240*sin(a),120*cos(a))*2*APL_PI/4096;
        }
        x.growth = 2*APL_PI*run.layer_d_global;
        ok = 1;
        for (int axis = 0; axis < 2; axis++) for (int direction = -1; direction <= 1; direction += 2)
        for (int rotated = 0; rotated < 2; rotated++) for (int rev = 0; rev < 2; rev++) {
            double theta = rotated ? 0.37 : 0;
            double ct = cos(theta), sn = sin(theta);
            double opposite = -direction*(perimeter-direction*x.growth/2);
            double expected = direction*(perimeter+direction*x.growth/2);
            radius[0] = axis ? 240 : 120;
            x.uout_x = ct; x.uout_y = sn;
            memset(turns,0,sizeof turns);
            turns[direction > 0 ? 0 : 1] = (AplTurn){opposite*ct,opposite*sn,12};
            lp.a = rev; lp.b = 1-rev; lp.side = rev ? -direction : direction;
            ok = ok && apl_collect_anchors(&x,1) == 1 &&
                fabs(anchors[0].Tx-expected*ct) < 1e-8 && fabs(anchors[0].Ty-expected*sn) < 1e-8 &&
                anchors[0].estimated && !anchors[0].measured_turn && anchors[0].reverse == rev;
        }
        if (!ok) { fprintf(stderr,"  asm_place selftest FAIL: opposite turn on a noncircular section\n"); fails++; }
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AplAnchor a[24]; double du[24], w[8], u[8], v[8]; size_t order[24]; int kind[8];
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.anchors = a;
        memset(a, 0, sizeof a);
        const double px[4] = {0,7,10,14}, pw[4] = {1,3,3,2};
        for (int i = 0; i < 4; i++) {
            a[i].tx = du[i] = px[i]; a[i].w = pw[i]; a[i].k = 1; a[i].cluster = -1; order[i] = (size_t)i;
        }
        int n = apl_cluster_pass(&x, 4, order, du, 10, 10, 1, 0, 0, 0, w, u, v, kind);
        int ok = n == 1 && w[0] == 8 && a[0].cluster == -1;
        memset(a, 0, sizeof a);
        for (int i = 0; i < 6; i++) {
            a[i].tx = du[i] = i; a[i].ty = i < 3 ? -1000 : 1000;
            a[i].w = 1; a[i].k = 1; a[i].cluster = -1; order[i] = (size_t)i;
        }
        n = apl_cluster_pass(&x, 6, order, du, 10, 10, 2, 0, 0, 0, w, u, v, kind);
        ok = ok && n == 2 && w[0] == 3 && w[1] == 3 && a[0].cluster != a[5].cluster;
        uint32_t seed = 71293;
        for (int trial = 0; trial < 40; trial++) {
            memset(a, 0, sizeof a);
            AplDuIdx key[24];
            for (int i = 0; i < 24; i++) {
                seed = seed*1664525u+1013904223u; a[i].tx = du[i] = (int)(seed%41)-20;
                seed = seed*1664525u+1013904223u; a[i].ty = (int)(seed%101)-50;
                seed = seed*1664525u+1013904223u; a[i].w = 1+seed%7;
                a[i].k = i%5 != 0; a[i].cluster = -1; key[i] = (AplDuIdx){du[i], (size_t)i};
            }
            qsort(key, 24, sizeof key[0], apl_cmp_duidx);
            for (int i = 0; i < 24; i++) order[i] = key[i].idx;
            double expected = 0;
            for (int i = 0; i < 24; i++) for (int j = 0; j < 24; j++) {
                double score = 0;
                for (int k = 0; k < 24; k++) if (a[k].k > 0 && a[k].tx >= a[i].tx &&
                    a[k].tx-a[i].tx <= 10 && a[k].ty >= a[j].ty && a[k].ty-a[j].ty <= 14) score += a[k].w;
                expected = fmax(expected, score);
            }
            n = apl_cluster_pass(&x, 24, order, du, 10, 7, 1, 0, 0, 0, w, u, v, kind);
            ok = ok && n == 1 && w[0] == expected;
        }
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: bounded two-dimensional anchor clusters\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[2]; memset(charts, 0, sizeof charts);
        charts[0].placement_state = ASM_PLACE_ROOT; charts[1].pose_x = 1000;
        run.charts = charts; run.n_charts = 2;
        AplAnchor anchors[3]; memset(anchors, 0, sizeof anchors);
        for (int i = 0; i < 3; i++) {
            anchors[i].k = 1; anchors[i].measured_order = 1; anchors[i].chart_c = 1;
            anchors[i].Tx = 1000; anchors[i].tx = 1000; anchors[i].cluster = i;
        }
        anchors[1].k = 2; anchors[1].tx = anchors[1].Tx = 2000;
        anchors[2].observation = 1; anchors[2].local_u = 10; anchors[2].tx = 1010;
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.anchors = anchors; x.na = 3; x.n_observations = 2;
        AplConsensus good = apl_candidate_consensus(&x);
        charts[1].pose_x += 40;
        AplConsensus uncertain = apl_candidate_consensus(&x);
        charts[1].pose_x += 460;
        AplConsensus bad = apl_candidate_consensus(&x);
        if (good.observations != 2 || good.inliers != 2 || good.model_inliers != 2 || good.model_loss > 1e-10 ||
            uncertain.observations != 2 || uncertain.inliers != 0 || uncertain.model_inliers != 2 ||
            bad.observations != 2 || bad.inliers != 0 || bad.model_inliers != 0 || bad.model_loss <= good.model_loss) {
            fprintf(stderr, "  asm_place selftest FAIL: independent full-evidence consensus\n"); fails++;
        }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        float small[12], long_uv[24];
        for (int i = 0; i < 6; i++) {
            small[2*i] = long_uv[2*i] = (float)(4*i);
            small[2*i+1] = long_uv[2*i+1] = 0;
            long_uv[12+2*i] = (float)(1000+4*i); long_uv[13+2*i] = 0;
        }
        AsmChart charts[3]; memset(charts, 0, sizeof charts);
        for (int i = 0; i < 3; i++) {
            charts[i].id = i; charts[i].component = i ? 1 : 0;
            charts[i].uv = i == 1 ? long_uv : small; charts[i].nv = i == 1 ? 12 : 6;
            charts[i].area3d = 1000;
        }
        charts[0].placement_state = ASM_PLACE_ROOT;
        charts[1].pose_theta = charts[2].pose_theta = -0.2;
        charts[2].pose_x = 1000*cos(-0.2)-4*sin(-0.2);
        charts[2].pose_y = 1000*sin(-0.2)+4*cos(-0.2);
        AsmChart saved[2] = {charts[1], charts[2]}, proposal[2];
        AsmRelation rels[2]; memset(rels, 0, sizeof rels);
        AsmCorr corr[12]; memset(corr, 0, sizeof corr);
        for (int j = 0; j < 2; j++) {
            rels[j].a = j; rels[j].b = j+1; rels[j].corr_first = 6*j; rels[j].corr_count = 6;
            rels[j].continuity = ASM_CONT_SOURCE;
            for (int i = 0; i < 6; i++) {
                AsmCorr *c = &corr[6*j+i]; c->va = 6*j+i; c->vb = i;
                c->valid = 3; c->run = 0; c->gap_a[1] = c->gap_b[1] = 4;
            }
        }
        rels[0].flags = ASM_REL_SWITCHED;
        run.charts = charts; run.n_charts = 3; run.rels = rels; run.n_rels = 2; run.corr = corr; run.n_corr = 12;
        AsmPlaceOpts o; AsmPlace_default_opts(&o);
        AsmPlaceStats stats; memset(&stats, 0, sizeof stats);
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.o = &o; x.st = &stats;
        uint8_t placed[2] = {1,0}, selected[2] = {1,0}; int32_t groups[3] = {0,1,1};
        double cu[3] = {10,510,10}, cv[3] = {0,0,0}, axis[3] = {0,4,8}, weights[6];
        AplAnchor anchors[6]; memset(anchors, 0, sizeof anchors);
        x.comp_placed = placed; x.orig_comp = groups; x.primary = 0; x.trial = 1; x.proposal = proposal;
        x.seam_selected = selected; x.anchors = anchors; x.wts = weights; x.hcap = 6; x.cluster_kind = 1;
        x.cu = cu; x.cv = cv; x.s_axis = axis; x.vm_ok = x.vm_pin = 1; x.vm_b = 1;
        int ok = apl_collect_anchors(&x, 1) == 6;
        x.na = 6;
        for (int i = 0; i < 6; i++) weights[i] = 1;
        ok = ok && !apl_candidate_shape(&x, saved, 2, 0);
        ok = ok && apl_place_and_audit(&x, 1, 0, 0, 0, NULL, 2) && x.nproposal == 2 &&
             fabs(proposal[0].pose_theta) < 1e-6 && fabs(proposal[0].pose_y-4) < 1e-6 &&
             fabs(proposal[1].pose_x-1000) < 1e-5 && fabs(proposal[1].pose_y-8) < 1e-5 &&
             memcmp(&charts[1], &saved[0], sizeof(AsmChart)) == 0 &&
             memcmp(&charts[2], &saved[1], sizeof(AsmChart)) == 0;
        if (ok) {
            charts[1] = proposal[0]; charts[2] = proposal[1];
            charts[2].pose_y += 200;
            ok = !apl_candidate_shape(&x, saved, 2, 0);
            x.diagnostic_axis_policy = 1;
            ok = ok && apl_candidate_shape(&x, saved, 2, 0);
            size_t support = 0; int32_t parent = -1; double error = 0;
            ok = ok && !apl_trial_support(&x, 1, NULL, 0, &support, &parent, &error, 0);
            x.cluster_kind = 0;
            ok = ok && !apl_candidate_shape(&x, saved, 2, 0);
            x.diagnostic_axis_policy = 2;
            ok = ok && apl_candidate_shape(&x, saved, 2, 0);
            double finite_y = charts[2].pose_y;
            charts[2].pose_y = NAN;
            ok = ok && !apl_candidate_shape(&x, saved, 2, 0);
            charts[2].pose_y = finite_y;
            x.diagnostic_axis_policy = 0;
            ok = ok && !apl_candidate_shape(&x, saved, 2, 0);
        }
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: final geometry gates after seam repair\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[3]; memset(charts, 0, sizeof charts);
        float uv[12];
        for (int i = 0; i < 6; i++) { uv[2*i] = (float)(4*i); uv[2*i+1] = 0; }
        for (int i = 0; i < 3; i++) {
            charts[i].id = i; charts[i].component = i ? 1 : 0;
            charts[i].uv = uv; charts[i].nv = 6; charts[i].area3d = 1000;
        }
        charts[2].pose_y = 4;
        AsmCorr corr[6]; memset(corr, 0, sizeof corr);
        for (int i = 0; i < 6; i++) {
            corr[i].va = corr[i].vb = i; corr[i].valid = 3; corr[i].run = 0;
            corr[i].gap_a[1] = corr[i].gap_b[1] = 4;
        }
        AsmRelation rel; memset(&rel, 0, sizeof rel);
        rel.a = 1; rel.b = 2; rel.corr_count = 6;
        rel.continuity = ASM_CONT_SOURCE | ASM_CONT_VERIFIED;
        run.charts = charts; run.n_charts = 3; run.rels = &rel; run.n_rels = 1;
        run.corr = corr; run.n_corr = 6; run.continuity_ready = 1;
        AsmPlaceOpts o; AsmPlace_default_opts(&o);
        AplAnchor anchors[6]; memset(anchors, 0, sizeof anchors);
        anchors[0].chart_c = 1; anchors[0].cluster = 0;
        anchors[1].chart_c = 2; anchors[1].cluster = 1;
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.o = &o; x.anchors = anchors; x.t0 = 1000; x.hcap = 6;
        double weight[2] = {10,9}, offset[2] = {0,1000}, rms[2] = {1,1};
        int32_t rel_off[4] = {0,0,1,2}, rel_adj[2] = {0,0}, groups[3] = {0,1,1};
        int32_t rings[4] = {0,1,0,0}, ncomp = 2;
        double areas[4] = {1000,2000,0,0}, layer[3] = {0,1,1};
        AplOrient ori[4]; memset(ori, 0, sizeof ori);
        AplCompDiag diag[4]; memset(diag, 0, sizeof diag);
        size_t deferred = 0;
        int newc = apl_cut_component(&x, 1, 2, 2, weight, offset, rms, 8,
                                    rel_off, rel_adj, &ncomp, 4, areas, groups,
                                    ori, diag, rings, layer, &deferred);
        int ok = newc == -1 && ncomp == 2 && deferred == 0 && x.cuts_preserved_source == 1 &&
                 charts[1].component == 1 && charts[2].component == 1 &&
                 groups[1] == 1 && groups[2] == 1 && areas[1] == 2000 && areas[2] == 0 &&
                 rel.flags == 0 && rel.continuity == (ASM_CONT_SOURCE | ASM_CONT_VERIFIED);
        charts[2].pose_y += 20;
        newc = apl_cut_component(&x, 1, 2, 2, weight, offset, rms, 8,
                                rel_off, rel_adj, &ncomp, 4, areas, groups,
                                ori, diag, rings, layer, &deferred);
        ok = ok && newc == 2 && ncomp == 3 && deferred == 1 &&
                 !AsmRel_is_join(&rel) && (rel.flags & ASM_REL_SWITCHED) &&
                 !(rel.flags & ASM_REL_DROPPED) && rel.continuity == ASM_CONT_SOURCE;
        charts[2].pose_y -= 20;
        rel.continuity |= ASM_CONT_CUT;
        uint8_t placed[4] = {1,0,0,0}; x.comp_placed = placed;
        charts[1].component = 0; charts[1].placement_state = ASM_PLACE_ROOT;
        ok = ok && apl_collect_anchors(&x, newc) == 6 && x.na_seam == 6;
        x.orig_comp = groups; charts[2].pose_x = 1000;
        AsmContinuityTarget targets[12];
        for (int i = 0; i < 12; i++) targets[i] = (AsmContinuityTarget){2, (double)i, 0, 1000+i, 4, 1, 1};
        size_t support; int32_t parent; double error;
        ok = ok && !apl_trial_support(&x, newc, targets, 12, &support, &parent, &error, 0);
        charts[2].pose_x = 0;
        charts[2].component = 0; charts[2].placement_state = ASM_PLACE_SEAM;
        ok = ok && AsmContinuity_confirm(&run) == 1 && AsmRel_is_join(&rel) &&
             (rel.continuity & ASM_CONT_VERIFIED) && (rel.flags & ASM_REL_READMIT);
        for (int veto = 0; veto < 2; veto++) {
            uint32_t hard = veto ? ASM_REL_CONTACT : ASM_REL_DROPPED;
            rel.flags = ASM_REL_SWITCHED | hard; rel.continuity = ASM_CONT_SOURCE;
            ok = ok && AsmContinuity_confirm(&run) == 0 && (rel.flags & hard) && !AsmRel_is_join(&rel);
        }
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: source-preserving cut / boundary recovery / hard veto\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        float uv[] = {0,0, 1000,0, 0,10};
        AsmChart a; memset(&a, 0, sizeof a); a.uv = uv; a.nv = 3;
        AsmChart b = a; b.pose_theta = 2*APL_PI;
        int ok = apl_same_pose(&a, &b, 8.0);
        b = a; b.flags = ASM_CHART_MIRROR;
        ok = ok && !apl_same_pose(&a, &b, 8.0);
        b = a; b.pose_theta = 0.01;
        ok = ok && !apl_same_pose(&a, &b, 8.0);
        uv[2] = 10; b.pose_theta = 0.1;
        ok = ok && apl_same_pose(&a, &b, 8.0);
        b = a; b.pose_x = 8.0;
        ok = ok && apl_same_pose(&a, &b, 8.0);
        b.pose_x = 8.000001;
        ok = ok && !apl_same_pose(&a, &b, 8.0);
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: material pose equivalence\n"); fails++; }
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.layer_d_global = 10;
        double spacing[2] = {10,10}; run.chart_layer_d = spacing;
        AsmPlaceOpts o; AsmPlace_default_opts(&o);
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run; x.o = &o;
        AsmLayerPair lp; memset(&lp, 0, sizeof lp); lp.a = 0; lp.b = 1; lp.k = 1; lp.d_median = 10;
        int ok = apl_hop_upper(&x, &lp) == 1;
        lp.d_median = 20;
        ok = ok && apl_hop_upper(&x, &lp) >= 2;
        lp.k = 2; lp.d_median = 10;
        ok = ok && apl_hop_upper(&x, &lp) >= 2;
        lp.d_median = 100;
        ok = ok && apl_hop_upper(&x, &lp) == o.max_layers;
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: missing-layer hop hypotheses\n"); fails++; }
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AsmChart charts[2]; memset(charts, 0, sizeof charts);
        float uv[24];
        for (int i = 0; i < 12; i++) { uv[2*i] = (float)(4*i-22); uv[2*i+1] = 0; }
        for (int i = 0; i < 2; i++) { charts[i].id = i; charts[i].component = i; charts[i].uv = uv; charts[i].nv = 12; }
        charts[0].placement_state = ASM_PLACE_ROOT;
        run.charts = charts; run.n_charts = 2;
        AsmChart original = charts[1], proposal[1];
        AplAnchor anchors[12]; memset(anchors, 0, sizeof anchors);
        double weights[12]; int32_t groups[2] = {0,1};
        AsmContinuityTarget targets[12];
        for (int i = 0; i < 12; i++) {
            AplAnchor *a = &anchors[i];
            a->sx = a->local_u = uv[2*i]; a->tx = 1000+cos(0.5)*a->sx; a->ty = sin(0.5)*a->sx;
            a->chart_p = 0; a->chart_c = 1; a->k = a->measured_order = 1; a->w = weights[i] = 1;
            targets[i] = (AsmContinuityTarget){1,a->local_u,0,a->tx,a->ty,0.05,0};
        }
        charts[1].pose_x = 1000;
        int ok = !AsmContinuity_close(&run, &original, 1, NULL, targets, 12);
        charts[1] = original;
        AsmPlaceOpts o; AsmPlace_default_opts(&o);
        AplCtx x; memset(&x, 0, sizeof x);
        x.run = &run; x.o = &o; x.anchors = anchors; x.wts = weights; x.na = 12;
        x.orig_comp = groups; x.primary = 0; x.trial = 1; x.proposal = proposal;
        for (int size = 0; size < 2; size++) {
            charts[1].area3d = original.area3d = size ? 1e6 : 1e3;
            ok = ok && apl_place_and_audit(&x, 1, 0, 1000, 0, NULL, 2);
            ok = ok && x.nproposal == 1 && proposal[0].pose_x == 1000 && proposal[0].pose_theta == 0 &&
                 proposal[0].placement_state == ASM_PLACE_LAYER && proposal[0].placement_support == 12 &&
                 memcmp(&charts[1], &original, sizeof original) == 0;
            for (int i = 0; i < 12; i++) anchors[i].measured_order = 2;
            ok = ok && !apl_place_and_audit(&x, 1, 0, 1000, 0, NULL, 2) && memcmp(&charts[1], &original, sizeof original) == 0;
            for (int i = 0; i < 12; i++) { anchors[i].measured_order = 1; weights[i] = 0; }
            ok = ok && !apl_place_and_audit(&x, 1, 0, 1000, 0, NULL, 2) && memcmp(&charts[1], &original, sizeof original) == 0;
            for (int i = 0; i < 12; i++) weights[i] = 1;
        }
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: supported incumbent / evidence / rollback\n"); fails++; }
        Arena_dispose(&run.arena);
    }
    {
        AsmRelation r; memset(&r, 0, sizeof r);
        r.corr_count = 12; r.rms = 1.0; r.flags = ASM_REL_SWITCHED;
        int ok = apl_placement_seam(&r, 0, 0) == APL_SWITCHED_SEAM_ANCHORS;
        r.flags |= ASM_REL_DROPPED;
        ok = ok && !apl_placement_seam(&r, 0, 0);
        r.flags = ASM_REL_SWITCHED | ASM_REL_CONTACT;
        ok = ok && !apl_placement_seam(&r, 0, 0);
        r.flags = ASM_REL_SWITCHED; r.corr_count = 5;
        ok = ok && !apl_placement_seam(&r, 0, 0);
        r.corr_count = 12; r.rms = 5.0;
        ok = ok && !apl_placement_seam(&r, 0, 0);
        r.flags = 0; r.rms = 1.0;
        ok = ok && !apl_placement_seam(&r, 0, 0);
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: switched-seam evidence vetoes\n"); fails++; }
    }
    {
        uint8_t placed[3] = {1, 0, 0};
        int prior_failures = fails;
        int32_t anchors[3] = {0, 80, 6}, seams[3] = {0, 0, 6};
        int32_t tried[3] = {0, 0, 0}, ring[3] = {0, 1, 2};
        double area[3] = {1e6, 1e5, 1e3};
        int expected = APL_SEAM_FIRST && APL_SEAM_FRONTIER_FIRST == 1 ? 2 : 1;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != expected) fails++;
        ring[2] = 1;
        expected = APL_SEAM_FIRST && APL_SEAM_FRONTIER_FIRST ? 2 : 1;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != expected) fails++;
        placed[2] = 2; tried[2] = 6;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != 1) fails++;
        anchors[2] = 7; seams[2] = 7;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != expected) fails++;
        placed[2] = 0; anchors[2] = 5; seams[2] = 5;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != 1) fails++;
        placed[1] = 1;
        if (apl_next_component(3, placed, anchors, seams, tried, ring, area, 12) != -1) fails++;
        if (fails != prior_failures) fprintf(stderr, "  asm_place selftest FAIL: seam frontier scheduling\n");
    }
    {
        AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
        AplCtx x; memset(&x, 0, sizeof x); x.run = &run;
        AplAnchor an[60]; memset(an, 0, sizeof an);
        for (int i = 0; i < 60; i++) {
            int seam = i < 20;
            an[i].sx = 10.0 * (i % 10); an[i].sy = 5.0 * (i / 10);
            an[i].tx = an[i].sx + (seam ? 0.0 : 100.0); an[i].ty = an[i].sy;
            an[i].w = 1.0; an[i].k = seam ? 0 : 1; an[i].Tx = seam ? 0.0 : 1000.0; an[i].cluster = -1; an[i].seam_rms = seam ? 1.0 : 0.0;
        }
        x.anchors = an; x.na_seam = 20;
        double w[APL_MAX_CLUSTERS_TOTAL], u[APL_MAX_CLUSTERS_TOTAL], v[APL_MAX_CLUSTERS_TOTAL]; int kind[APL_MAX_CLUSTERS_TOTAL];
        int ncl = apl_cluster_anchors(&x, 60, w, u, v, kind);
        int ok = APL_SEAM_FIRST ? (ncl == 2 && kind[0] == 1 && fabs(w[0] - 20.0) < 1e-9 && fabs(u[0]) < 1e-9 && kind[1] == 0 && fabs(w[1] - 40.0) < 1e-9 && fabs(u[1] - 100.0) < 1e-9)
                                : (ncl == 1 && fabs(w[0] - 60.0) < 1e-9);
        if (!ok) { fprintf(stderr, "  asm_place selftest FAIL: seam-first clustering: %d clusters, kind %d w %.1f u %.1f\n", ncl, ncl ? kind[0] : -1, ncl ? w[0] : 0.0, ncl ? u[0] : 0.0); fails++; }
        Arena_dispose(&run.arena);
    }
    AsmRun run; memset(&run, 0, sizeof run); run.arena = Arena_new();
    enum { N = 5 };
    float uv[N*N*2];
    int32_t faces[(N-1)*(N-1)*2*3];
    float xyz[3][N*N*3], nrm[3][N*N*3];
    int nf = 0;
    for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) { uv[(j*N+i)*2] = 10.0f*i; uv[(j*N+i)*2+1] = 10.0f*j; }
    for (int j = 0; j < N-1; j++) for (int i = 0; i < N-1; i++) {
        int a = j*N+i, b = a+1, c = a+N, d = c+1;
        faces[nf*3] = a; faces[nf*3+1] = b; faces[nf*3+2] = d; nf++;
        faces[nf*3] = a; faces[nf*3+1] = d; faces[nf*3+2] = c; nf++;
    }
    for (int L = 0; L < 3; L++) {
        double R = 100.0 + 10.0 * L;
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
            double ang = (10.0 * i) / R;
            float *p = &xyz[L][(j*N+i)*3], *n = &nrm[L][(j*N+i)*3];
            p[0] = (float)(10.0 * j);
            p[1] = (float)(R * sin(ang));
            p[2] = (float)(R * cos(ang));
            n[0] = 0.0f; n[1] = (float)sin(ang); n[2] = (float)cos(ang);
        }
    }
    for (int L = 0; L < 3; L++) {
        AsmChart c; memset(&c, 0, sizeof c);
        c.id = L; c.nv = N*N; c.nf = (size_t)nf; c.uv = uv; c.faces = faces; c.nrm = nrm[L]; c.xyz = xyz[L];
        c.area3d = 1600.0; c.placed = 1;
        c.component = (L < 2) ? 0 : 1;
        if (L < 2) { c.pose_x = 2.0 * APL_PI * (100.0 + 10.0 * L) * L; c.pose_y = 0.0; c.pose_theta = 0.0; }
        else { c.pose_x = 1234.0; c.pose_y = -77.0; c.pose_theta = 0.7; c.flags |= ASM_CHART_MIRROR; }
        AsmRun_push_chart(&run, &c);
    }
    double T0 = run.charts[1].pose_x - run.charts[0].pose_x;
    run.chart_layer_d = ARENA_CALLOC(run.arena, 3, sizeof(double));
    for (int i = 0; i < 3; i++) run.chart_layer_d[i] = 10.0;
    run.layer_d_global = 10.0;
    int spec[3][3] = { {0, 1, 1}, {1, 0, -1}, {1, 2, 1} };
    for (int pi = 0; pi < 3; pi++) {
        AsmLayerPair lp; memset(&lp, 0, sizeof lp);
        lp.a = spec[pi][0]; lp.b = spec[pi][1]; lp.side = (int8_t)spec[pi][2]; lp.count = 16; lp.d_median = 10.0;
        lp.k = 1;
        lp.hit_first = (int32_t)run.n_layer_hits; lp.hit_count = 0;
        for (int j = 0; j < N-1; j++) for (int i = 0; i < N-1; i++) {
            AsmLayerHit h; memset(&h, 0, sizeof h); h.a = lp.a; h.va = j*N+i; h.b = lp.b; h.side = lp.side; h.dist = 10.0f; h.order = 1;
            h.fb = (j*(N-1)+i)*2; h.l0 = 1.0f; h.l1 = 0.0f;
            AsmRun_push_layer_hit(&run, &h);
            lp.hit_count++;
        }
        AsmRun_push_layer(&run, &lp);
    }
    AsmPlaceOpts o; AsmPlace_default_opts(&o);
    o.min_anchors = 8;
    AsmPlaceStats st;
    AsmPlace_run(&run, &o, NULL, &st);
    const AsmChart *c2 = &run.charts[2];
    double gx, gy;
    apl_pose(c2, 0.0, 0.0, &gx, &gy);
    double ex = 2.0*APL_PI*(105.0+115.0);
    if (st.placed != 3 || c2->component != 0 || fabs(gx - ex) > 8.0 || fabs(gy) > 2.0 || fabs(c2->pose_theta) > 0.02 ||
        (c2->flags & ASM_CHART_MIRROR) || st.oriented_mirrored + st.mirrored != 1 ||
        c2->placement_state != ASM_PLACE_LAYER || c2->placement_support < 8 || c2->placement_parent < 0) {
        fprintf(stderr, "  asm_place selftest FAIL: placed %zu comp %d at (%.2f, %.2f) expected (%.2f, 0) theta %.3f mirror %d (mirrored %zu)\n",
                st.placed, c2->component, gx, gy, ex, c2->pose_theta, (c2->flags & ASM_CHART_MIRROR) != 0, st.oriented_mirrored); fails++;
    }
    {
        double u0, v0, u1, v1;
        apl_pose(&run.charts[0], 0.0, 0.0, &u0, &v0);
        apl_pose(&run.charts[1], 0.0, 0.0, &u1, &v1);
        if (!(u1 > u0 + 100.0)) { fprintf(stderr, "  asm_place selftest FAIL: u does not increase outward (%.1f -> %.1f)\n", u0, u1); fails++; }
    }
    {
        double ax, ay, bx, by;
        apl_vertex_point(&run.charts[0], 0, &ax, &ay);
        apl_vertex_point(&run.charts[0], N*(N-1), &bx, &by);
        if (!(by > ay + 30.0) || fabs(bx - ax) > 2.0) { fprintf(stderr, "  asm_place selftest FAIL: v not along z (%.1f,%.1f)->(%.1f,%.1f)\n", ax, ay, bx, by); fails++; }
    }
    {
        const double offsets[] = {-48.0, -24.000000000001, -24.0, -23.999999999999,
                                  0.0, 23.999999999999, 24.0, 24.000000000001, 48.0, 5000.0};
        int32_t orig[3] = {0, 0, 1};
        AsmChart *cand = &run.charts[2];
        const AsmChart *cand_list[1] = {cand};
        AplRimGrid *grid = aprg_new(run.arena, 24.0);
        aprg_insert_chart(grid, &run.charts[0]);
        aprg_insert_chart(grid, &run.charts[1]);
        size_t touched = 0;
        for (size_t j = 0; j < sizeof offsets / sizeof offsets[0]; j++) {
            cand->pose_x = run.charts[1].pose_x + offsets[j];
            size_t ref_n = 0, grid_n = 0;
            double ref_bad = apl_rim_audit(&run, orig, 1, 0, 24.0, &ref_n);
            double grid_bad = aprg_probe(grid, &run, cand_list, 1, 24.0, &grid_n);
            touched += grid_n;
            if (grid_n != ref_n || fabs(grid_bad - ref_bad) > 1e-12) {
                fprintf(stderr, "  asm_place selftest FAIL: rim grid at offset %.12g: %zu/%.6f vs %zu/%.6f\n",
                        offsets[j], grid_n, grid_bad, ref_n, ref_bad); fails++;
            }
        }
        if (touched == 0) { fprintf(stderr, "  asm_place selftest FAIL: rim comparison had no support\n"); fails++; }
    }
    {
        double current[2][2*N*N],saved_current[2][2*N*N];
        for(int c=0;c<2;c++){
            for(int v=0;v<N*N;v++){
                double p[2];AsmChart_point(run.charts+c,(size_t)v,p);
                current[c][2*v]=p[0]+257.125;current[c][2*v+1]=p[1]-91.75;
            }
            run.charts[c].placed_uv=current[c];run.charts[c].pose_x=-5000-100*c;
            run.charts[c].pose_y=8000;run.charts[c].pose_theta=.31;
        }
        memcpy(saved_current,current,sizeof current);
        run.charts[2].placement_state=ASM_PLACE_NONE;run.charts[2].component=1;
        AsmChart saved[3];memcpy(saved,run.charts,sizeof saved);
        uint8_t core[3]={0,0,1};AsmConflictOpts co;AsmConflict_default_opts(&co);
        size_t layers=run.n_layers;run.n_layers=0;
        int refused=AsmPlace_retry_component(&run,core,&o,&co,&st);
        int okay=!refused && !memcmp(saved,run.charts,sizeof saved) && !memcmp(saved_current,current,sizeof current);
        run.n_layers=layers;
        int accepted=AsmPlace_retry_component(&run,core,&o,&co,&st);
        double p[2];AsmChart_point(run.charts+2,0,p);
        okay=okay && accepted==1 && fabs(p[0]-(ex+257.125))<8 && fabs(p[1]+91.75)<2 &&
             !memcmp(saved,run.charts,2*sizeof(AsmChart)) && !memcmp(saved_current,current,sizeof current);
        if(!okay){fprintf(stderr,"  asm_place selftest FAIL: repaired-field retry accepted %d at %.9g,%.9g\n",accepted,p[0],p[1]);fails++;}
    }
    Arena_dispose(&run.arena);
    fails += aps_selftest();
    if (fails == 0) fprintf(stderr, "  asm_place selftest: all passed\n");
    return fails;
}
