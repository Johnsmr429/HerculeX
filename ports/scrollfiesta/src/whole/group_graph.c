#include "group_graph.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/kdtree.h"
#include "../common/maxflow.h"
#include "../common/union_find.h"
#include "cube_register.h"

#define GG_PI 3.14159265358979323846
#define GG_MOVE_SCALE 4096.0

static void *gg_xmalloc(size_t n)
{
    void *p = malloc(n > 0 ? n : 1);
    if (p == NULL) { fprintf(stderr, "group_graph: OOM %zu\n", n); exit(1); }
    return p;
}

static int gg_cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}

static double gg_median(double *v, size_t n)
{
    if (n == 0) return 0.0;
    qsort(v, n, sizeof(double), gg_cmp_dbl);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct { double v, w; } GgWeighted;

static int gg_cmp_weighted(const void *a, const void *b)
{
    double x = ((const GgWeighted *)a)->v;
    double y = ((const GgWeighted *)b)->v;
    return (x < y) ? -1 : (x > y);
}

static double gg_weighted_median(GgWeighted *v, size_t n)
{
    if (n == 0) return 0.0;
    qsort(v, n, sizeof(*v), gg_cmp_weighted);
    double total = 0.0;
    for (size_t i = 0; i < n; i++) total += v[i].w;
    double accum = 0.0;
    for (size_t i = 0; i < n; i++) {
        accum += v[i].w;
        if (2.0 * accum >= total) return v[i].v;
    }
    return v[n - 1].v;
}

static double gg_wrap_pi(double x)
{
    return x - 2.0 * GG_PI * floor(x / (2.0 * GG_PI) + 0.5);
}

void GroupGraphOpts_default(GroupGraphOpts *o)
{
    assert(o);
    memset(o, 0, sizeof(*o));
    o->pair_gate = 3.5;
    o->radius_gate = 0.0;
    o->frac_gate = 0.5;
    o->prior_gate = 0.0;
    o->edge_dr_gate = 1.75;
    o->intra_prior = 0;
    o->intra_frac_max = 0.35;
    o->intra_conf = 128.0;
    o->min_edge_pairs = 3;
    o->conf_n_cap = 256;
    o->conf_mad0 = 0.02;
    o->prior_min_verts = 8;
    o->raw_component_gauge = 0;
    o->raw_du_gauge = 0;
    o->physical_du_gauge = 0;
    o->anchor_weight = 0.0;
    o->anchor_redundancy_ref = 0.0;
    o->do_moves = 1;
    o->max_moves = 40;
    o->du_passes = 4;
    o->verbose = 0;
}

static int32_t gg_node_of(const GroupGraph *g, size_t cube, int32_t gid)
{
    int32_t lo = g->cube_node0[cube], hi = g->cube_node0[cube + 1];
    while (lo < hi) {
        int32_t m = (lo + hi) / 2;
        if (g->nodes[m].gid < gid) lo = m + 1;
        else hi = m;
    }
    if (lo < g->cube_node0[cube + 1] && g->nodes[lo].gid == gid)
        return lo;
    return -1;
}

int32_t GroupGraph_node_of(const GroupGraph *g, size_t cube, int32_t gid)
{
    return gg_node_of(g, cube, gid);
}

typedef struct {
    GGEdge *edges;
    size_t  ne, cap;
    size_t  pairs_used, pairs_rej_radius, pairs_rej_frac;
    size_t  edges_rej_dr, edges_rej_prior;
} GGSeamOut;

static void gg_seam_push(GGSeamOut *res, const GGEdge *e)
{
    if (res->ne == res->cap) {
        size_t ncap = res->cap ? res->cap * 2 : 64;
        GGEdge *nn2 = (GGEdge *)gg_xmalloc(ncap * sizeof(GGEdge));
        if (res->ne > 0) memcpy(nn2, res->edges, res->ne * sizeof(GGEdge));
        free(res->edges);
        res->edges = nn2;
        res->cap = ncap;
    }
    res->edges[res->ne++] = *e;
}

static void gg_seam_edges(Arena_T arena, const GroupGraph *out,
                          const GroupGraphOpts *opts,
                          const float axis_point[3],
                          double spiral_a, double spiral_b,
                          double rgate, double gate2,
                          const SkinVert *skin_c, size_t nskin_c,
                          const SkinVert *skin_j, size_t nskin_j,
                          size_t c, size_t j, GGSeamOut *res)
{
    float *jp = (float *)ARENA_ALLOC(arena, nskin_j * 3 * sizeof(float));
    for (size_t q = 0; q < nskin_j; q++) {
        jp[q * 3 + 0] = skin_j[q].p[0];
        jp[q * 3 + 1] = skin_j[q].p[1];
        jp[q * 3 + 2] = skin_j[q].p[2];
    }
    KDTree_T kd = KDTree_new(arena, jp, nskin_j);

    int32_t ng_i = 0, ng_j = 0;
    for (size_t q = 0; q < nskin_c; q++)
        if (skin_c[q].gid >= ng_i) ng_i = skin_c[q].gid + 1;
    for (size_t q = 0; q < nskin_j; q++)
        if (skin_j[q].gid >= ng_j) ng_j = skin_j[q].gid + 1;
    if (ng_i == 0 || ng_j == 0) return;

    size_t nbuck = (size_t)ng_i * (size_t)ng_j;
    int32_t *bcnt = (int32_t *)ARENA_CALLOC(arena, nbuck, sizeof(int32_t));
    size_t npair_max = nskin_c;
    int32_t *p_j = (int32_t *)ARENA_ALLOC(arena, npair_max * sizeof(int32_t));
    double *p_dr = (double *)ARENA_ALLOC(arena, npair_max * sizeof(double));
    for (size_t q = 0; q < nskin_c; q++) {
        const SkinVert *si = &skin_c[q];
        p_j[q] = -1;
        if (si->gid < 0) continue;
        float d2 = 0.0f;
        size_t hit = KDTree_nearest(kd, si->p, &d2);
        if ((double)d2 > gate2) continue;
        const SkinVert *sj = &skin_j[hit];
        if (sj->gid < 0) continue;
        double dyi = (double)si->p[1] - axis_point[1];
        double dxi = (double)si->p[2] - axis_point[2];
        double dyj = (double)sj->p[1] - axis_point[1];
        double dxj = (double)sj->p[2] - axis_point[2];
        double ri = sqrt(dyi * dyi + dxi * dxi);
        double rj = sqrt(dyj * dyj + dxj * dxj);
        if (fabs(ri - rj) > rgate) {
            res->pairs_rej_radius++;
            continue;
        }
        double dphi = (double)sj->phi - (double)si->phi;
        if (fabs(gg_wrap_pi(dphi)) > opts->frac_gate) {
            res->pairs_rej_frac++;
            continue;
        }
        p_j[q] = (int32_t)hit;
        p_dr[q] = fabs(ri - rj);
        bcnt[(size_t)si->gid * (size_t)ng_j + (size_t)sj->gid]++;
        res->pairs_used++;
    }
    int32_t *boff = (int32_t *)ARENA_CALLOC(arena, nbuck + 1,
                                            sizeof(int32_t));
    for (size_t bq = 0; bq < nbuck; bq++)
        boff[bq + 1] = boff[bq] + bcnt[bq];
    size_t tot = (size_t)boff[nbuck];
    double *bphi = (double *)ARENA_ALLOC(arena, (tot + 1) * sizeof(double));
    double *bu = (double *)ARENA_ALLOC(arena, (tot + 1) * sizeof(double));
    double *bphiraw = (double *)ARENA_ALLOC(arena, (tot + 1)
                                            * sizeof(double));
    double *bdr = (double *)ARENA_ALLOC(arena, (tot + 1) * sizeof(double));
    int32_t *cur = (int32_t *)ARENA_ALLOC(arena, (nbuck + 1)
                                          * sizeof(int32_t));
    memcpy(cur, boff, nbuck * sizeof(int32_t));
    for (size_t q = 0; q < nskin_c; q++) {
        if (p_j[q] < 0) continue;
        const SkinVert *si = &skin_c[q];
        const SkinVert *sj = &skin_j[p_j[q]];
        size_t bq = (size_t)si->gid * (size_t)ng_j + (size_t)sj->gid;
        bphi[cur[bq]] = (double)sj->phi - (double)si->phi;
        bu[cur[bq]] = (double)sj->u - (double)si->u;
        bphiraw[cur[bq]] = (double)si->phi;
        bdr[cur[bq]] = p_dr[q];
        cur[bq]++;
    }
    for (int32_t gi = 0; gi < ng_i; gi++) {
        for (int32_t gj = 0; gj < ng_j; gj++) {
            size_t bq = (size_t)gi * (size_t)ng_j + (size_t)gj;
            int32_t n = bcnt[bq];
            if (n < opts->min_edge_pairs) continue;
            int32_t na = gg_node_of(out, c, gi);
            int32_t nb = gg_node_of(out, j, gj);
            if (na < 0 || nb < 0) continue;
            if (opts->edge_dr_gate > 0.0) {
                double drm = gg_median(&bdr[boff[bq]], (size_t)n);
                if (drm > opts->edge_dr_gate) {
                    res->edges_rej_dr++;
                    continue;
                }
            }
            double *ph = &bphi[boff[bq]];
            double med = gg_median(ph, (size_t)n);
            int32_t obs = (int32_t)lround(med / (2.0 * GG_PI));
            if (opts->prior_gate > 0.0
                && out->nodes[na].n_prior >= opts->prior_min_verts
                && out->nodes[nb].n_prior >= opts->prior_min_verts) {
                double want = out->nodes[na].prior
                            - out->nodes[nb].prior;
                if (fabs((double)obs - want) > opts->prior_gate) {
                    res->edges_rej_prior++;
                    continue;
                }
            }
            double *scr = (double *)ARENA_ALLOC(
                arena, (size_t)n * sizeof(double));
            for (int32_t t = 0; t < n; t++)
                scr[t] = fabs(ph[t] - 2.0 * GG_PI * (double)obs);
            double mad = gg_median(scr, (size_t)n);
            for (int32_t t = 0; t < n; t++)
                scr[t] = bu[boff[bq] + t]
                       - CubeReg_deltaU(spiral_a, spiral_b,
                                        bphiraw[boff[bq] + t], obs);
            double due = gg_median(scr, (size_t)n);
            GGEdge E;
            memset(&E, 0, sizeof(E));
            E.a = na;
            E.b = nb;
            E.obs = obs;
            E.du = due;
            E.mad = mad;
            E.n = n;
            {
                int32_t ncl = n < opts->conf_n_cap ? n : opts->conf_n_cap;
                E.conf = (double)ncl / (1.0 + mad / opts->conf_mad0);
            }
            E.in_tree = 0;
            gg_seam_push(res, &E);
        }
    }
}

int GroupGraph_build(Arena_T arena, const CubeNode *cnodes, size_t n_cubes,
                     SkinVert *const *skins, const size_t *nskin,
                     const float axis_point[3],
                     double spiral_a, double spiral_b, double pitch,
                     const GroupGraphOpts *opts, GroupGraph *out)
{
    assert(arena && cnodes && skins && nskin && opts && out);
    memset(out, 0, sizeof(*out));
    if (n_cubes == 0)
        return -1;
    double rgate = opts->radius_gate > 0.0 ? opts->radius_gate : 0.5 * pitch;
    double gate2 = opts->pair_gate * opts->pair_gate;

    out->n_cubes = n_cubes;
    out->cube_node0 = (int32_t *)ARENA_CALLOC(arena, n_cubes + 1,
                                              sizeof(int32_t));
    size_t total_nodes = 0;
    int32_t max_gid_all = -1;
    for (size_t c = 0; c < n_cubes; c++) {
        int32_t mx = -1;
        for (size_t i = 0; i < nskin[c]; i++)
            if (skins[c][i].gid > mx) mx = skins[c][i].gid;
        if (mx > max_gid_all) max_gid_all = mx;
    }
    uint8_t *present = (uint8_t *)gg_xmalloc((size_t)(max_gid_all + 2));
    for (size_t c = 0; c < n_cubes; c++) {
        memset(present, 0, (size_t)(max_gid_all + 2));
        for (size_t i = 0; i < nskin[c]; i++)
            if (skins[c][i].gid >= 0) present[skins[c][i].gid] = 1;
        int32_t cnt = 0;
        for (int32_t gid = 0; gid <= max_gid_all; gid++)
            if (present[gid]) cnt++;
        out->cube_node0[c + 1] = cnt;
        total_nodes += (size_t)cnt;
    }
    for (size_t c = 0; c < n_cubes; c++)
        out->cube_node0[c + 1] += out->cube_node0[c];
    out->n_nodes = total_nodes;
    out->nodes = (GGNode *)ARENA_CALLOC(arena, total_nodes + 1,
                                        sizeof(GGNode));
    for (size_t c = 0; c < n_cubes; c++) {
        memset(present, 0, (size_t)(max_gid_all + 2));
        for (size_t i = 0; i < nskin[c]; i++)
            if (skins[c][i].gid >= 0) present[skins[c][i].gid] = 1;
        int32_t at = out->cube_node0[c];
        for (int32_t gid = 0; gid <= max_gid_all; gid++) {
            if (!present[gid]) continue;
            out->nodes[at].cube = (int32_t)c;
            out->nodes[at].gid = gid;
            out->nodes[at].comp = -1;
            at++;
        }
        if (spiral_b != 0.0) {
            for (size_t i = 0; i < nskin[c]; i++) {
                const SkinVert *s = &skins[c][i];
                if (s->gid < 0) continue;
                int32_t nd = gg_node_of(out, c, s->gid);
                if (nd < 0) continue;
                double dy = (double)s->p[1] - axis_point[1];
                double dx = (double)s->p[2] - axis_point[2];
                double r = sqrt(dy * dy + dx * dx);
                out->nodes[nd].prior += (r - spiral_a) / spiral_b
                                      - (double)s->phi / (2.0 * GG_PI);
                {
                    double phi = (double)s->phi;
                    double fu = spiral_a * phi
                              + spiral_b * phi * phi / (4.0 * GG_PI);
                    out->nodes[nd].du_prior += fu - (double)s->u;
                }
                out->nodes[nd].n_prior++;
            }
        }
    }
    for (size_t i = 0; i < total_nodes; i++)
        if (out->nodes[i].n_prior > 0) {
            out->nodes[i].prior /= (double)out->nodes[i].n_prior;
            out->nodes[i].du_prior /= (double)out->nodes[i].n_prior;
        }
    free(present);

    size_t n_seams = 0;
    for (size_t c = 0; c < n_cubes; c++) {
        if (nskin[c] == 0) continue;
        for (int e6 = 1; e6 < 6; e6 += 2) {
            int32_t j = cnodes[c].nbr[e6];
            if (j >= 0 && nskin[j] > 0) n_seams++;
        }
    }
    int32_t *seam_c = (int32_t *)ARENA_ALLOC(
        arena, (n_seams + 1) * sizeof(int32_t));
    int32_t *seam_j = (int32_t *)ARENA_ALLOC(
        arena, (n_seams + 1) * sizeof(int32_t));
    {
        size_t at = 0;
        for (size_t c = 0; c < n_cubes; c++) {
            if (nskin[c] == 0) continue;
            for (int e6 = 1; e6 < 6; e6 += 2) {
                int32_t j = cnodes[c].nbr[e6];
                if (j < 0 || nskin[j] == 0) continue;
                seam_c[at] = (int32_t)c;
                seam_j[at] = j;
                at++;
            }
        }
    }
    GGSeamOut *souts = (GGSeamOut *)gg_xmalloc(
        (n_seams ? n_seams : 1) * sizeof(GGSeamOut));
    memset(souts, 0, (n_seams ? n_seams : 1) * sizeof(GGSeamOut));
    {
        int s = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (s = 0; s < (int)n_seams; s++) {
            Arena_T sa = Arena_new();
            gg_seam_edges(sa, out, opts, axis_point, spiral_a, spiral_b,
                          rgate, gate2,
                          skins[seam_c[s]], nskin[seam_c[s]],
                          skins[seam_j[s]], nskin[seam_j[s]],
                          (size_t)seam_c[s], (size_t)seam_j[s], &souts[s]);
            Arena_dispose(&sa);
        }
    }
    size_t cap_e = 1024, ne = 0;
    for (size_t s2 = 0; s2 < n_seams; s2++) cap_e += souts[s2].ne;
    GGEdge *edges = (GGEdge *)gg_xmalloc(cap_e * sizeof(GGEdge));
    for (size_t s2 = 0; s2 < n_seams; s2++) {
        GGSeamOut *so = &souts[s2];
        if (so->ne > 0) {
            memcpy(&edges[ne], so->edges, so->ne * sizeof(GGEdge));
            ne += so->ne;
        }
        free(so->edges);
        out->pairs_used += so->pairs_used;
        out->pairs_rej_radius += so->pairs_rej_radius;
        out->pairs_rej_frac += so->pairs_rej_frac;
        out->edges_rej_dr += so->edges_rej_dr;
        out->edges_rej_prior += so->edges_rej_prior;
    }
    free(souts);
    if (opts->intra_prior) {
        for (size_t c = 0; c < n_cubes; c++) {
            for (int32_t na = out->cube_node0[c];
                 na < out->cube_node0[c + 1]; na++) {
                if (out->nodes[na].n_prior < opts->prior_min_verts) continue;
                for (int32_t nb = na + 1; nb < out->cube_node0[c + 1];
                     nb++) {
                    if (out->nodes[nb].n_prior < opts->prior_min_verts)
                        continue;
                    double d = out->nodes[na].prior - out->nodes[nb].prior;
                    int32_t obs = (int32_t)lround(d);
                    double frac = d - (double)obs;
                    if (fabs(frac) > opts->intra_frac_max) continue;
                    if (ne == cap_e) {
                        cap_e *= 2;
                        GGEdge *nn2 = (GGEdge *)gg_xmalloc(
                            cap_e * sizeof(GGEdge));
                        memcpy(nn2, edges, ne * sizeof(GGEdge));
                        free(edges);
                        edges = nn2;
                    }
                    GGEdge *E = &edges[ne++];
                    E->a = na;
                    E->b = nb;
                    E->obs = obs;
                    E->du = 0.0;
                    E->mad = fabs(frac);
                    E->n = 0;
                    int32_t npmin =
                        out->nodes[na].n_prior < out->nodes[nb].n_prior
                            ? out->nodes[na].n_prior
                            : out->nodes[nb].n_prior;
                    if (npmin > 256) npmin = 256;
                    E->conf = opts->intra_conf * ((double)npmin / 256.0)
                            / (1.0 + fabs(frac) / 0.15);
                    E->in_tree = 0;
                }
            }
        }
    }

    out->edges = (GGEdge *)ARENA_ALLOC(arena, (ne + 1) * sizeof(GGEdge));
    memcpy(out->edges, edges, ne * sizeof(GGEdge));
    free(edges);
    out->n_edges = ne;
    return 0;
}

typedef struct { double conf; int32_t idx; } GgSortEdge;

static int gg_cmp_edge_desc(const void *a, const void *b)
{
    const GgSortEdge *x = (const GgSortEdge *)a, *y = (const GgSortEdge *)b;
    if (x->conf != y->conf) return x->conf > y->conf ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}

static double gg_energy(const GroupGraph *g, size_t *out_frustrated)
{
    double e = 0.0;
    size_t fr = 0;
    for (size_t i = 0; i < g->n_edges; i++) {
        const GGEdge *E = &g->edges[i];
        int32_t r = g->nodes[E->a].k - g->nodes[E->b].k - E->obs;
        if (r != 0) fr++;
        e += E->conf * (double)(r < 0 ? -r : r);
    }
    if (out_frustrated != NULL)
        *out_frustrated = fr;
    return e;
}

static int gg_move(Arena_T arena, GroupGraph *g,
                   const GroupGraphOpts *opts, int delta)
{
    Arena_Mark mark = Arena_save(arena);
    Maxflow_T mf = Maxflow_new(arena, (int32_t)g->n_nodes, g->n_edges);
    int64_t *D = (int64_t *)ARENA_CALLOC(arena, g->n_nodes + 1,
                                         sizeof(int64_t));
    int64_t base = 0;
    for (size_t i = 0; i < g->n_edges; i++) {
        const GGEdge *E = &g->edges[i];
        int64_t c = (int64_t)llround(E->conf * GG_MOVE_SCALE);
        if (c <= 0) continue;
        int64_t r = (int64_t)(g->nodes[E->a].k - g->nodes[E->b].k - E->obs);
        int64_t t00 = c * (r < 0 ? -r : r);
        int64_t rp = r + delta, rm = r - delta;
        int64_t t10 = c * (rp < 0 ? -rp : rp);
        int64_t t01 = c * (rm < 0 ? -rm : rm);
        base += t00;
        D[E->a] += t10 - t00;
        D[E->b] += t00 - t10;
        int64_t beta = t01 + t10 - 2 * t00;
        assert(beta >= 0);
        if (beta > 0)
            Maxflow_add_edge(mf, E->a, E->b, beta, 0);
    }
    if (opts->anchor_weight > 0.0) {
        for (size_t i = 0; i < g->n_nodes; i++) {
            const GGNode *nd = &g->nodes[i];
            int32_t nw = nd->n_prior;
            if (nw <= 0) continue;
            if (nw > opts->conf_n_cap) nw = opts->conf_n_cap;
            int64_t c = (int64_t)llround(opts->anchor_weight
                                         * (double)nw * GG_MOVE_SCALE);
            if (c <= 0) continue;
            int64_t t0 = (int64_t)llround((double)c
                * fabs((double)nd->k - nd->prior));
            int64_t t1 = (int64_t)llround((double)c
                * fabs((double)(nd->k + delta) - nd->prior));
            base += t0;
            D[i] += t1 - t0;
        }
    }
    int64_t constant = base;
    for (size_t i = 0; i < g->n_nodes; i++) {
        if (D[i] > 0)
            Maxflow_add_terminal(mf, (int32_t)i, D[i], 0);
        else if (D[i] < 0) {
            constant += D[i];
            Maxflow_add_terminal(mf, (int32_t)i, 0, -D[i]);
        }
    }
    int64_t cut = Maxflow_solve(mf);
    int64_t new_e = constant + cut;
    int applied = 0;
    if (new_e < base) {
        for (size_t i = 0; i < g->n_nodes; i++)
            if (!Maxflow_in_source_side(mf, (int32_t)i))
                g->nodes[i].k += delta;
        applied = 1;
    }
    Arena_restore(arena, mark);
    return applied;
}

int GroupGraph_solve(Arena_T arena, GroupGraph *g, const GroupGraphOpts *opts)
{
    assert(arena && g && opts);
    size_t nn = g->n_nodes, ne = g->n_edges;
    if (nn == 0)
        return 0;
    Arena_Mark mark = Arena_save(arena);

    GgSortEdge *se = (GgSortEdge *)ARENA_ALLOC(arena, (ne + 1)
                                               * sizeof(GgSortEdge));
    for (size_t i = 0; i < ne; i++) {
        se[i].conf = g->edges[i].conf;
        se[i].idx = (int32_t)i;
    }
    qsort(se, ne, sizeof(GgSortEdge), gg_cmp_edge_desc);
    UnionFind uf = UF_new(arena, (int32_t)nn);
    for (size_t i = 0; i < ne; i++) {
        GGEdge *E = &g->edges[se[i].idx];
        if (uf_find(&uf, E->a) != uf_find(&uf, E->b)) {
            uf_union(&uf, E->a, E->b);
            E->in_tree = 1;
        }
    }

    int32_t *deg = (int32_t *)ARENA_CALLOC(arena, nn + 1, sizeof(int32_t));
    for (size_t i = 0; i < ne; i++) {
        if (!g->edges[i].in_tree) continue;
        deg[g->edges[i].a + 1]++;
        deg[g->edges[i].b + 1]++;
    }
    for (size_t i = 0; i < nn; i++) deg[i + 1] += deg[i];
    int32_t *adj = (int32_t *)ARENA_ALLOC(arena,
                                          ((size_t)deg[nn] + 1)
                                          * sizeof(int32_t));
    {
        int32_t *cur = (int32_t *)ARENA_ALLOC(arena, (nn + 1)
                                              * sizeof(int32_t));
        memcpy(cur, deg, nn * sizeof(int32_t));
        for (size_t i = 0; i < ne; i++) {
            if (!g->edges[i].in_tree) continue;
            adj[cur[g->edges[i].a]++] = (int32_t)i;
            adj[cur[g->edges[i].b]++] = (int32_t)i;
        }
    }

    int32_t *queue = (int32_t *)ARENA_ALLOC(arena, (nn + 1)
                                            * sizeof(int32_t));
    int32_t n_comp = 0;
    for (size_t root = 0; root < nn; root++) {
        if (g->nodes[root].comp >= 0) continue;
        int32_t qh = 0, qt = 0;
        g->nodes[root].comp = n_comp;
        g->nodes[root].k = 0;
        queue[qt++] = (int32_t)root;
        while (qh < qt) {
            int32_t u = queue[qh++];
            for (int32_t e = deg[u]; e < deg[u + 1]; e++) {
                const GGEdge *E = &g->edges[adj[e]];
                int32_t v = E->a == u ? E->b : E->a;
                if (g->nodes[v].comp >= 0) continue;
                g->nodes[v].comp = n_comp;
                g->nodes[v].k = E->a == u ? g->nodes[u].k - E->obs
                                          : g->nodes[u].k + E->obs;
                queue[qt++] = v;
            }
        }
        n_comp++;
    }
    g->n_comp = n_comp;

    if (opts->raw_component_gauge) {
        typedef struct { int32_t v, w; } GgGaugeVote;
        for (int32_t cmp = 0; cmp < n_comp; cmp++) {
            Arena_Mark m2 = Arena_save(arena);
            GgGaugeVote *vote = (GgGaugeVote *)ARENA_ALLOC(
                arena, (nn + 1) * sizeof(GgGaugeVote));
            size_t nv = 0;
            int64_t wtot = 0;
            for (size_t i = 0; i < nn; i++) {
                const GGNode *nd = &g->nodes[i];
                if (nd->comp != cmp) continue;
                int32_t w = nd->n_prior > 0 ? nd->n_prior : 1;
                if (w > opts->conf_n_cap) w = opts->conf_n_cap;
                vote[nv].v = -nd->k;
                vote[nv].w = w;
                wtot += w;
                nv++;
            }
            for (size_t x = 1; x < nv; x++) {
                GgGaugeVote key = vote[x];
                size_t y = x;
                while (y > 0 && vote[y - 1].v > key.v) {
                    vote[y] = vote[y - 1];
                    y--;
                }
                vote[y] = key;
            }
            int64_t acc = 0;
            int32_t shift = 0;
            for (size_t x = 0; x < nv; x++) {
                acc += vote[x].w;
                if (2 * acc >= wtot) {
                    shift = vote[x].v;
                    break;
                }
            }
            for (size_t i = 0; i < nn; i++)
                if (g->nodes[i].comp == cmp)
                    g->nodes[i].k += shift;
            Arena_restore(arena, m2);
        }
    } else {
        double *wsum = (double *)ARENA_CALLOC(arena, (size_t)n_comp,
                                              sizeof(double));
        double *psum = (double *)ARENA_CALLOC(arena, (size_t)n_comp,
                                              sizeof(double));
        for (size_t i = 0; i < nn; i++) {
            const GGNode *nd = &g->nodes[i];
            if (nd->n_prior < opts->prior_min_verts) continue;
            psum[nd->comp] += (double)nd->n_prior
                            * (nd->prior - (double)nd->k);
            wsum[nd->comp] += (double)nd->n_prior;
        }
        for (size_t i = 0; i < nn; i++) {
            GGNode *nd = &g->nodes[i];
            if (wsum[nd->comp] > 0.0)
                nd->k += (int32_t)lround(psum[nd->comp] / wsum[nd->comp]);
            else if (nd->n_prior > 0)
                nd->k += (int32_t)lround(nd->prior - (double)nd->k);
        }
    }

    GroupGraphOpts move_opts = *opts;
    {
        double tree_edges = nn > (size_t)n_comp
                          ? (double)(nn - (size_t)n_comp) : 0.0;
        double cycle_edges = (double)ne - tree_edges;
        if (cycle_edges < 0.0) cycle_edges = 0.0;
        g->edge_redundancy = tree_edges > 0.0
                           ? cycle_edges / tree_edges : 0.0;
        if (move_opts.anchor_redundancy_ref > 0.0 &&
            g->edge_redundancy < move_opts.anchor_redundancy_ref) {
            move_opts.anchor_weight *= sqrt(
                g->edge_redundancy / move_opts.anchor_redundancy_ref);
        }
        g->anchor_weight_effective = move_opts.anchor_weight;
    }
    g->moves_applied = 0;
    if (opts->do_moves && ne > 0) {
        for (int it = 0; it < opts->max_moves; it++) {
            int applied = 0;
            applied |= gg_move(arena, g, &move_opts, +1);
            applied |= gg_move(arena, g, &move_opts, -1);
            if (!applied) break;
            g->moves_applied++;
        }
    }
    g->energy = gg_energy(g, &g->n_frustrated);

    if (opts->raw_du_gauge) {
        for (size_t i = 0; i < nn; i++)
            g->nodes[i].du = 0.0;
    } else {
        uint8_t *seen = (uint8_t *)ARENA_CALLOC(arena, nn + 1, 1);
        for (size_t root = 0; root < nn; root++) {
            if (seen[root]) continue;
            int32_t qh = 0, qt = 0;
            seen[root] = 1;
            g->nodes[root].du = 0.0;
            queue[qt++] = (int32_t)root;
            while (qh < qt) {
                int32_t u = queue[qh++];
                for (int32_t e = deg[u]; e < deg[u + 1]; e++) {
                    const GGEdge *E = &g->edges[adj[e]];
                    int32_t v = E->a == u ? E->b : E->a;
                    if (seen[v]) continue;
                    seen[v] = 1;
                    g->nodes[v].du = E->a == u ? g->nodes[u].du - E->du
                                               : g->nodes[u].du + E->du;
                    queue[qt++] = v;
                }
            }
        }
        int32_t *adeg = (int32_t *)ARENA_CALLOC(arena, nn + 1,
                                                sizeof(int32_t));
        for (size_t i = 0; i < ne; i++) {
            adeg[g->edges[i].a + 1]++;
            adeg[g->edges[i].b + 1]++;
        }
        for (size_t i = 0; i < nn; i++) adeg[i + 1] += adeg[i];
        int32_t *aadj = (int32_t *)ARENA_ALLOC(arena,
                                               ((size_t)adeg[nn] + 1)
                                               * sizeof(int32_t));
        {
            int32_t *cur = (int32_t *)ARENA_ALLOC(arena, (nn + 1)
                                                  * sizeof(int32_t));
            memcpy(cur, adeg, nn * sizeof(int32_t));
            for (size_t i = 0; i < ne; i++) {
                aadj[cur[g->edges[i].a]++] = (int32_t)i;
                aadj[cur[g->edges[i].b]++] = (int32_t)i;
            }
        }
        typedef struct { double v, w; } DuVote;
        for (int pass = 0; pass < opts->du_passes; pass++) {
            for (size_t u = 0; u < nn; u++) {
                int32_t nd = adeg[u + 1] - adeg[u];
                if (nd == 0) continue;
                Arena_Mark m2 = Arena_save(arena);
                DuVote *vt = (DuVote *)ARENA_ALLOC(arena, (size_t)nd
                                                   * sizeof(DuVote));
                int32_t q = 0;
                for (int32_t e = adeg[u]; e < adeg[u + 1]; e++) {
                    const GGEdge *E = &g->edges[aadj[e]];
                    if (E->n == 0) continue;
                    if (E->a == (int32_t)u) {
                        vt[q].v = g->nodes[E->b].du + E->du;
                    } else {
                        vt[q].v = g->nodes[E->a].du - E->du;
                    }
                    vt[q].w = E->conf;
                    q++;
                }
                if (q == 0) continue;
                for (int32_t x = 1; x < q; x++) {
                    DuVote key = vt[x];
                    int32_t y = x - 1;
                    while (y >= 0 && vt[y].v > key.v) {
                        vt[y + 1] = vt[y];
                        y--;
                    }
                    vt[y + 1] = key;
                }
                double wtot = 0.0;
                for (int32_t x = 0; x < q; x++) wtot += vt[x].w;
                double acc = 0.0;
                double med = vt[q - 1].v;
                for (int32_t x = 0; x < q; x++) {
                    acc += vt[x].w;
                    if (acc >= 0.5 * wtot) { med = vt[x].v; break; }
                }
                g->nodes[u].du = med;
                Arena_restore(arena, m2);
            }
        }
        {
            int32_t *coff = (int32_t *)ARENA_CALLOC(
                arena, (size_t)n_comp + 1, sizeof(int32_t));
            for (size_t i = 0; i < nn; i++)
                coff[g->nodes[i].comp + 1]++;
            for (int32_t c = 0; c < n_comp; c++)
                coff[c + 1] += coff[c];
            int32_t *cnode = (int32_t *)ARENA_ALLOC(
                arena, (nn + 1) * sizeof(int32_t));
            int32_t *ccur = (int32_t *)ARENA_ALLOC(
                arena, ((size_t)n_comp + 1) * sizeof(int32_t));
            memcpy(ccur, coff, (size_t)n_comp * sizeof(int32_t));
            for (size_t i = 0; i < nn; i++)
                cnode[ccur[g->nodes[i].comp]++] = (int32_t)i;

            for (int32_t cmp = 0; cmp < n_comp; cmp++) {
                Arena_Mark m2 = Arena_save(arena);
                int32_t cn = coff[cmp + 1] - coff[cmp];
                GgWeighted *vals = (GgWeighted *)ARENA_ALLOC(
                    arena, ((size_t)cn + 1) * sizeof(GgWeighted));
                size_t nv2 = 0;
                for (int32_t q = coff[cmp]; q < coff[cmp + 1]; q++) {
                    GGNode *nd = &g->nodes[cnode[q]];
                    if (nd->n_prior <= 0) continue;
                    vals[nv2].v = opts->physical_du_gauge
                                ? nd->du_prior - nd->du : nd->du;
                    int32_t nw = nd->n_prior;
                    if (nw > opts->conf_n_cap) nw = opts->conf_n_cap;
                    vals[nv2].w = (double)nw;
                    nv2++;
                }
                if (nv2 > 0) {
                    double shift = gg_weighted_median(vals, nv2);
                    if (!opts->physical_du_gauge) shift = -shift;
                    for (int32_t q = coff[cmp]; q < coff[cmp + 1]; q++)
                        g->nodes[cnode[q]].du += shift;
                    if (opts->physical_du_gauge)
                        g->du_gauge_components++;
                }
                Arena_restore(arena, m2);
            }
        }
    }

    {
        double *res = (double *)ARENA_ALLOC(arena, (nn + 1) * sizeof(double));
        size_t nr = 0;
        double rmax = 0.0;
        for (size_t i = 0; i < nn; i++) {
            if (g->nodes[i].n_prior <= 0) continue;
            double r = fabs(g->nodes[i].du - g->nodes[i].du_prior);
            res[nr++] = r;
            if (r > rmax) rmax = r;
        }
        if (nr > 0) {
            qsort(res, nr, sizeof(double), gg_cmp_dbl);
            g->du_prior_resid_median = res[nr / 2];
            g->du_prior_resid_p95 = res[(size_t)(0.95 * (double)(nr - 1))];
            g->du_prior_resid_max = rmax;
        }
    }

    if (opts->verbose)
        fprintf(stderr, "[group_graph] nodes=%zu edges=%zu comps=%d "
                "energy=%.1f frustrated=%zu moves=%d anchor=%.5g "
                "redundancy=%.3f pairs used=%zu "
                "rej r/frac=%zu/%zu edges rej dr/prior=%zu/%zu "
                "du-prior med/p95/max=%.2f/%.2f/%.2f\n", nn, ne,
                g->n_comp, g->energy, g->n_frustrated, g->moves_applied,
                g->anchor_weight_effective, g->edge_redundancy,
                g->pairs_used, g->pairs_rej_radius, g->pairs_rej_frac,
                g->edges_rej_dr, g->edges_rej_prior,
                g->du_prior_resid_median, g->du_prior_resid_p95,
                g->du_prior_resid_max);

    Arena_restore(arena, mark);
    return 0;
}

int GroupGraph_cube_reg(Arena_T arena, const GroupGraph *g, size_t cube,
                        PlacedReg *out)
{
    assert(arena && g && out && cube < g->n_cubes);
    memset(out, 0, sizeof(*out));
    int32_t n0 = g->cube_node0[cube], n1 = g->cube_node0[cube + 1];
    int32_t maxg = -1;
    for (int32_t i = n0; i < n1; i++)
        if (g->nodes[i].gid > maxg) maxg = g->nodes[i].gid;
    {
        int64_t best_w = -1;
        int32_t best_k = 0;
        for (int32_t i = n0; i < n1; i++) {
            int64_t w = 0;
            for (int32_t j2 = n0; j2 < n1; j2++)
                if (g->nodes[j2].k == g->nodes[i].k)
                    w += g->nodes[j2].n_prior > 0 ? g->nodes[j2].n_prior : 1;
            if (w > best_w) { best_w = w; best_k = g->nodes[i].k; }
        }
        out->wk_cube = n1 > n0 ? best_k : 0;
        double dsum = 0.0;
        int32_t dn = 0;
        for (int32_t i = n0; i < n1; i++)
            if (g->nodes[i].k == out->wk_cube) { dsum += g->nodes[i].du; dn++; }
        out->du_cube = dn > 0 ? dsum / (double)dn : 0.0;
    }
    if (maxg < 0)
        return 0;
    int32_t *wk = (int32_t *)ARENA_ALLOC(arena, ((size_t)maxg + 1)
                                         * sizeof(int32_t));
    double *du = (double *)ARENA_ALLOC(arena, ((size_t)maxg + 1)
                                       * sizeof(double));
    for (int32_t gid = 0; gid <= maxg; gid++) {
        wk[gid] = out->wk_cube;
        du[gid] = out->du_cube;
    }
    for (int32_t i = n0; i < n1; i++) {
        wk[g->nodes[i].gid] = g->nodes[i].k;
        du[g->nodes[i].gid] = g->nodes[i].du;
    }
    out->g_wk = wk;
    out->g_du = du;
    out->n_groups = maxg + 1;
    out->g_warp_nk = NULL;
    return 0;
}

static int ggst_check(int cond, const char *what, int *fails)
{
    if (!cond) {
        fprintf(stderr, "[group_graph selftest]   FAIL: %s\n", what);
        (*fails)++;
    }
    return cond;
}

static void ggst_wire(GroupGraph *g, GGNode *nodes, size_t nn,
                      GGEdge *edges, size_t ne, int32_t *cube_node0,
                      size_t n_cubes)
{
    memset(g, 0, sizeof(*g));
    g->nodes = nodes;
    g->n_nodes = nn;
    g->edges = edges;
    g->n_edges = ne;
    g->cube_node0 = cube_node0;
    g->n_cubes = n_cubes;
}

static GGEdge ggst_edge(int32_t a, int32_t b, int32_t obs, double conf,
                        double due)
{
    GGEdge e;
    memset(&e, 0, sizeof(e));
    e.a = a;
    e.b = b;
    e.obs = obs;
    e.conf = conf;
    e.du = due;
    e.n = (int32_t)conf;
    return e;
}

static GGNode ggst_node(int32_t cube, int32_t gid, double prior,
                        int32_t n_prior)
{
    GGNode n;
    memset(&n, 0, sizeof(n));
    n.cube = cube;
    n.gid = gid;
    n.prior = prior;
    n.n_prior = n_prior;
    n.comp = -1;
    return n;
}

int GroupGraph_selftest(void)
{
    int fails = 0;
    Arena_T arena = Arena_new();
    GroupGraphOpts o;
    GroupGraphOpts_default(&o);
    o.verbose = 0;

    {
        GGNode nd[4] = {
            ggst_node(0, 0, 2.1, 50), ggst_node(1, 0, 1.9, 50),
            ggst_node(2, 0, 3.2, 50), ggst_node(3, 0, 2.8, 50)
        };
        GGEdge ed[4] = {
            ggst_edge(0, 1, 0, 1000, 0.0), ggst_edge(1, 2, -1, 800, 0.0),
            ggst_edge(2, 3, 0, 900, 0.0), ggst_edge(3, 0, 1, 700, 0.0)
        };
        int32_t cn0[5] = { 0, 1, 2, 3, 4 };
        GroupGraph g;
        ggst_wire(&g, nd, 4, ed, 4, cn0, 4);
        ggst_check(GroupGraph_solve(arena, &g, &o) == 0, "t1 rc", &fails);
        ggst_check(nd[0].k == 2 && nd[1].k == 2 && nd[2].k == 3
                   && nd[3].k == 3, "t1 consistent cycle exact", &fails);
        ggst_check(g.n_frustrated == 0 && g.energy == 0.0,
                   "t1 zero residual", &fails);
    }

    {
        GGNode nd[4] = {
            ggst_node(0, 0, 0.0, 100), ggst_node(1, 0, 0.0, 100),
            ggst_node(2, 0, 1.0, 100), ggst_node(3, 0, 1.0, 100)
        };
        GGEdge ed[4] = {
            ggst_edge(0, 1, 0, 1000, 0.0), ggst_edge(2, 3, 0, 1000, 0.0),
            ggst_edge(1, 2, -1, 900, 0.0), ggst_edge(0, 3, 0, 5, 0.0)
        };
        int32_t cn0[5] = { 0, 1, 2, 3, 4 };
        GroupGraph g;
        ggst_wire(&g, nd, 4, ed, 4, cn0, 4);
        GroupGraph_solve(arena, &g, &o);
        ggst_check(nd[0].k == 0 && nd[1].k == 0 && nd[2].k == 1
                   && nd[3].k == 1, "t2 tree ignores contaminated edge",
                   &fails);
        ggst_check(g.n_frustrated == 1, "t2 one frustrated (the bad edge)",
                   &fails);
    }

    {
        GGNode nd[4];
        for (int i = 0; i < 4; i++) nd[i] = ggst_node(i, 0, 0.0, 100);
        GGEdge ed[6] = {
            ggst_edge(0, 1, 0, 1000, 0.0), ggst_edge(2, 3, 0, 1000, 0.0),
            ggst_edge(1, 2, 1, 101, 0.0),
            ggst_edge(0, 2, 0, 100, 0.0), ggst_edge(1, 3, 0, 100, 0.0),
            ggst_edge(0, 3, 0, 100, 0.0)
        };
        int32_t cn0[5] = { 0, 1, 2, 3, 4 };
        GroupGraph g;
        ggst_wire(&g, nd, 4, ed, 6, cn0, 4);
        GroupGraph_solve(arena, &g, &o);
        int all_equal = (nd[1].k == nd[0].k && nd[2].k == nd[0].k
                         && nd[3].k == nd[0].k);
        ggst_check(all_equal, "t3 collective shift repairs a bad TREE edge",
                   &fails);
        ggst_check(g.n_frustrated == 1, "t3 only the wrong edge frustrated",
                   &fails);
        GGNode nd2[4];
        for (int i = 0; i < 4; i++) nd2[i] = ggst_node(i, 0, 0.0, 100);
        GGEdge ed2[6];
        memcpy(ed2, ed, sizeof(ed2));
        for (int i = 0; i < 6; i++) ed2[i].in_tree = 0;
        GroupGraph g2;
        ggst_wire(&g2, nd2, 4, ed2, 6, cn0, 4);
        GroupGraphOpts o2 = o;
        o2.do_moves = 0;
        GroupGraph_solve(arena, &g2, &o2);
        int split = (nd2[2].k != nd2[0].k);
        ggst_check(split, "t3b without moves the branch cut persists",
                   &fails);
    }

    {
        GGNode nd[4] = {
            ggst_node(0, 0, 5.1, 100), ggst_node(1, 0, 4.9, 100),
            ggst_node(2, 0, -2.2, 100), ggst_node(3, 0, -1.8, 100)
        };
        GGEdge ed[2] = {
            ggst_edge(0, 1, 0, 500, 0.0), ggst_edge(2, 3, 0, 500, 0.0)
        };
        int32_t cn0[5] = { 0, 1, 2, 3, 4 };
        GroupGraph g;
        ggst_wire(&g, nd, 4, ed, 2, cn0, 4);
        GroupGraph_solve(arena, &g, &o);
        ggst_check(g.n_comp == 2, "t4 two components", &fails);
        ggst_check(nd[0].k == 5 && nd[1].k == 5 && nd[2].k == -2
                   && nd[3].k == -2, "t4 absolute gauge from priors",
                   &fails);
    }

    {
        GGNode nd[3] = {
            ggst_node(0, 0, 0.0, 10), ggst_node(1, 0, 0.0, 10),
            ggst_node(2, 0, 0.0, 10)
        };
        GGEdge ed[2] = {
            ggst_edge(0, 1, 0, 100, 3.0),
            ggst_edge(1, 2, 0, 100, -4.0)
        };
        int32_t cn0[4] = { 0, 1, 2, 3 };
        GroupGraph g;
        ggst_wire(&g, nd, 3, ed, 2, cn0, 3);
        GroupGraph_solve(arena, &g, &o);
        double d01 = nd[0].du - nd[1].du, d12 = nd[1].du - nd[2].du;
        ggst_check(fabs(d01 - 3.0) < 0.1 && fabs(d12 + 4.0) < 0.1,
                   "t5 du differences recovered", &fails);
    }

    {
        enum { NS = 24 };
        SkinVert sa[NS + 1], sb[NS + 1];
        float ap[3] = { 0.0f, 0.0f, 0.0f };
        double pitch = 10.0, a = 1.0, b = -10.0;
        for (int i = 0; i < NS; i++) {
            sa[i].p[0] = (float)(2 * i);
            sa[i].p[1] = 30.0f;
            sa[i].p[2] = 40.0f;
            sa[i].u = 100.0f;
            sa[i].v = (float)(2 * i);
            sa[i].phi = -3.0f;
            sa[i].gid = 0;
            sb[i] = sa[i];
            sb[i].p[0] += 0.3f;
            sb[i].phi = (float)(-3.0 + 2.0 * GG_PI);
            sb[i].u = (float)(100.0
                + CubeReg_deltaU(a, b, -3.0, 1) + 2.5);
            sb[i].gid = 2;
        }
        sa[NS].p[0] = 100.0f;
        sa[NS].p[1] = 30.0f;
        sa[NS].p[2] = 40.0f;
        sa[NS].u = 0.0f;
        sa[NS].v = 0.0f;
        sa[NS].phi = 0.0f;
        sa[NS].gid = 0;
        sb[NS] = sa[NS];
        sb[NS].p[0] += 0.2f;
        sb[NS].phi = 1.0f;
        sb[NS].gid = 2;
        CubeNode cn[2];
        memset(cn, 0, sizeof(cn));
        snprintf(cn[0].id, sizeof(cn[0].id), "a");
        snprintf(cn[1].id, sizeof(cn[1].id), "b");
        for (int i = 0; i < 6; i++) { cn[0].nbr[i] = -1; cn[1].nbr[i] = -1; }
        cn[0].nbr[5] = 1;
        SkinVert *skins[2] = { sa, sb };
        size_t nskin[2] = { NS + 1, NS + 1 };
        GroupGraph g;
        GroupGraphOpts ob = o;
        int rc = GroupGraph_build(arena, cn, 2, skins, nskin, ap,
                                  a, b, pitch, &ob, &g);
        ggst_check(rc == 0, "t6 build rc", &fails);
        ggst_check(g.n_edges == 1, "t6 one edge (contaminant filtered)",
                   &fails);
        if (g.n_edges == 1) {
            ggst_check(g.edges[0].obs == 1, "t6 obs = one turn", &fails);
            ggst_check(g.edges[0].n == NS, "t6 pair count", &fails);
            ggst_check(fabs(g.edges[0].du - 2.5) < 0.05,
                       "t6 du_e recovered", &fails);
            ggst_check(g.pairs_rej_frac >= 1, "t6 frac filter fired",
                       &fails);
        }
        GroupGraph_solve(arena, &g, &o);
        PlacedReg r0, r1;
        GroupGraph_cube_reg(arena, &g, 0, &r0);
        GroupGraph_cube_reg(arena, &g, 1, &r1);
        ggst_check(r0.n_groups >= 1 && r1.n_groups >= 3,
                   "t6 tables sized by max gid", &fails);
        if (r0.n_groups >= 1 && r1.n_groups >= 3)
            ggst_check(r0.g_wk[0] - r1.g_wk[2] == 1,
                       "t6 solved k difference = obs", &fails);
    }

    {
        GGNode nd[2] = {
            ggst_node(0, 0, 10.0, 100), ggst_node(1, 0, 9.0, 100)
        };
        GGEdge ed[1] = { ggst_edge(0, 1, 1, 100, 1234.0) };
        int32_t cn0[3] = { 0, 1, 2 };
        GroupGraph g;
        GroupGraphOpts raw = o;
        raw.raw_component_gauge = 1;
        raw.raw_du_gauge = 1;
        ggst_wire(&g, nd, 2, ed, 1, cn0, 2);
        GroupGraph_solve(arena, &g, &raw);
        ggst_check(nd[0].k == 0 && nd[1].k == -1,
                   "t8 raw component gauge preserves k=0 chart", &fails);
        ggst_check(fabs(nd[0].du) < 1e-12 && fabs(nd[1].du) < 1e-12,
                   "t8 raw du gauge preserves continuous chart", &fails);
        ggst_check(g.n_frustrated == 0,
                   "t8 relative forest constraint remains exact", &fails);
    }

    {
        GGNode nd[3] = {
            ggst_node(0, 0, 0.0, 100), ggst_node(1, 0, 0.0, 100),
            ggst_node(2, 0, 0.0, 100)
        };
        nd[0].du_prior = 7.0;
        nd[1].du_prior = 4.0;
        nd[2].du_prior = 8.0;
        GGEdge ed[2] = {
            ggst_edge(0, 1, 0, 100, 3.0),
            ggst_edge(1, 2, 0, 100, -4.0)
        };
        int32_t cn0[4] = { 0, 1, 2, 3 };
        GroupGraph g;
        GroupGraphOpts physical = o;
        physical.physical_du_gauge = 1;
        ggst_wire(&g, nd, 3, ed, 2, cn0, 3);
        GroupGraph_solve(arena, &g, &physical);
        ggst_check(fabs(nd[0].du - 7.0) < 1e-9
                   && fabs(nd[1].du - 4.0) < 1e-9
                   && fabs(nd[2].du - 8.0) < 1e-9,
                   "t11 physical du gauge fixes absolute translation",
                   &fails);
        ggst_check(fabs((nd[0].du - nd[1].du) - 3.0) < 1e-9
                   && fabs((nd[1].du - nd[2].du) + 4.0) < 1e-9,
                   "t11 physical gauge preserves seam equations", &fails);
    }

    {
        GGNode nd[2] = {
            ggst_node(0, 0, 0.0, 100), ggst_node(1, 0, 0.0, 100)
        };
        GGEdge ed[1] = { ggst_edge(0, 1, 1, 10, 0.0) };
        int32_t cn0[3] = { 0, 1, 2 };
        GroupGraph g;
        GroupGraphOpts anchored = o;
        anchored.anchor_weight = 1.0;
        ggst_wire(&g, nd, 2, ed, 1, cn0, 2);
        GroupGraph_solve(arena, &g, &anchored);
        ggst_check(nd[0].k == 0 && nd[1].k == 0,
                   "t9 anchored cut rejects false seam bridge", &fails);
        ggst_check(g.n_frustrated == 1,
                   "t9 rejected bridge is reported frustrated", &fails);
    }

    {
        GGNode nd[4];
        for (int i = 0; i < 4; i++) nd[i] = ggst_node(i, 0, 0.0, 100);
        GGEdge ed[4] = {
            ggst_edge(0, 1, 0, 100, 0.0), ggst_edge(1, 2, 0, 100, 0.0),
            ggst_edge(2, 3, 0, 100, 0.0), ggst_edge(3, 0, 0, 100, 0.0)
        };
        int32_t cn0[5] = { 0, 1, 2, 3, 4 };
        GroupGraph g;
        GroupGraphOpts sparse = o;
        sparse.anchor_weight = 0.03;
        sparse.anchor_redundancy_ref = 1.0;
        sparse.do_moves = 0;
        ggst_wire(&g, nd, 4, ed, 4, cn0, 4);
        GroupGraph_solve(arena, &g, &sparse);
        ggst_check(fabs(g.edge_redundancy - 1.0 / 3.0) < 1e-12,
                    "t10 cycle redundancy", &fails);
        ggst_check(fabs(g.anchor_weight_effective
                        - 0.03 * sqrt(1.0 / 3.0)) < 1e-12,
                    "t10 sparse anchor scaling", &fails);
    }

    {
        GroupGraph g;
        int32_t cn0[1] = { 0 };
        ggst_wire(&g, NULL, 0, NULL, 0, cn0, 0);
        ggst_check(GroupGraph_solve(arena, &g, &o) == 0, "t7 empty solve",
                   &fails);
    }

    Arena_dispose(&arena);
    fprintf(stderr, "[group_graph selftest] %s (%d failure%s)\n",
            fails == 0 ? "PASSED" : "FAILED", fails, fails == 1 ? "" : "s");
    return fails;
}
