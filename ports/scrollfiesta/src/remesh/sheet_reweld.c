#define _USE_MATH_DEFINES
#include "sheet_reweld.h"

#include "seam_weld.h"
#include "../common/kdtree.h"
#include "../common/union_find.h"
#include "../common/pipeline_constants.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void SheetReweld_default_params(SheetReweldParams *p)
{
    assert(p);
    p->overlap_r   = SHEET_REWELD_OVERLAP_R;
    p->vote_weight = SHEET_REWELD_VOTE_WEIGHT;
    p->min_score   = SHEET_REWELD_MIN_SCORE;
    p->margin_frac = SHEET_REWELD_MARGIN_FRAC;
    p->rho_max     = SHEET_REWELD_RHO_MAX;
    p->band        = SHEET_REWELD_BAND;
}

int SheetReweld_label(Arena_T arena, const int32_t *faces, size_t nf, size_t nv,
                      int32_t **out_vert_sheet, size_t *out_n_sheets)
{
    *out_vert_sheet = NULL; *out_n_sheets = 0;
    if (nv == 0) return 0;
    assert(nv <= (size_t)INT32_MAX);

    UnionFind uf = UF_new(arena, (int32_t)nv);
    for (size_t f = 0; f < nf; f++) {
        int32_t a = faces[f*3+0], b = faces[f*3+1], c = faces[f*3+2];
        uf_union(&uf, a, b);
        uf_union(&uf, b, c);
    }
    uint8_t *used = (uint8_t *)ARENA_CALLOC(arena, (size_t)nv, 1L);
    for (size_t f = 0; f < nf; f++) {
        used[faces[f*3+0]] = 1; used[faces[f*3+1]] = 1; used[faces[f*3+2]] = 1;
    }
    int32_t *root2sheet = (int32_t *)ARENA_ALLOC(arena, (size_t)(nv*sizeof(int32_t)));
    for (size_t v = 0; v < nv; v++) root2sheet[v] = -1;
    int32_t *vert_sheet = (int32_t *)ARENA_ALLOC(arena, (size_t)(nv*sizeof(int32_t)));
    size_t n_sheets = 0;
    for (size_t v = 0; v < nv; v++) {
        if (!used[v]) { vert_sheet[v] = -1; continue; }
        int32_t r = uf_find(&uf, (int32_t)v);
        if (root2sheet[r] < 0) root2sheet[r] = (int32_t)n_sheets++;
        vert_sheet[v] = root2sheet[r];
    }
    *out_vert_sheet = vert_sheet;
    *out_n_sheets = n_sheets;
    return 0;
}

typedef struct { int32_t a, b; double w; } PairW;

static int cmp_pairw(const void *x, const void *y)
{
    const PairW *p = (const PairW *)x, *q = (const PairW *)y;
    if (p->a != q->a) return p->a < q->a ? -1 : 1;
    if (p->b != q->b) return p->b < q->b ? -1 : 1;
    return 0;
}

static void aggregate_pairs(PairW *raw, size_t *n)
{
    if (*n == 0) return;
    qsort(raw, *n, sizeof(PairW), cmp_pairw);
    size_t w = 0;
    for (size_t i = 0; i < *n; ) {
        size_t j = i + 1;
        double sum = raw[i].w;
        while (j < *n && raw[j].a == raw[i].a && raw[j].b == raw[i].b) {
            sum += raw[j].w; j++;
        }
        raw[w].a = raw[i].a; raw[w].b = raw[i].b; raw[w].w = sum; w++;
        i = j;
    }
    *n = w;
}

static int cmp_u64(const void *x, const void *y)
{
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static int near_seam(const float *pos, double cube_size, double band)
{
    for (int a = 0; a < 3; a++) {
        double c = (double)pos[a];
        double k = floor(c / cube_size + 0.5);
        if (fabs(c - k * cube_size) <= band && k >= 1.0) return 1;
    }
    return 0;
}

int SheetReweld_process(Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        size_t n_phase1_bridge,
                        const int32_t *vert_sheet, size_t n_sheets,
                        float cube_size,
                        const BpaBridgeGate *gate,
                        const SheetReweldParams *params,
                        int32_t **out_faces, size_t *out_nf,
                        SheetReweldStats *stats)
{
    (void)gate;
    SheetReweldStats st; memset(&st, 0, sizeof st);
    st.n_sheets = n_sheets;

    {
        size_t cap = (nf ? nf : 1) * 3;
        int32_t *out0 = (int32_t *)ARENA_ALLOC(arena, (size_t)(cap*sizeof(int32_t)));
        memcpy(out0, faces, nf*3*sizeof(int32_t));
        *out_faces = out0; *out_nf = nf;
    }
    if (!params || nf < 1 || n_sheets < 2 || !vert_sheet || cube_size <= 0.0f) {
        if (stats) *stats = st;
        return 0;
    }

    SheetReweldParams p = *params;
    double band = (double)p.band;
    Arena_Mark scratch = Arena_save(arena);

    uint8_t *is_bnd = (uint8_t *)ARENA_CALLOC(arena, (size_t)nv, 1L);
    {
        size_t ne = nf * 3;
        uint64_t *keys = (uint64_t *)ARENA_ALLOC(arena, (size_t)(ne*sizeof(uint64_t)));
        for (size_t f = 0; f < nf; f++)
            for (int e = 0; e < 3; e++) {
                int32_t a = faces[f*3+(size_t)e], b = faces[f*3+(size_t)((e+1)%3)];
                uint64_t lo = (uint64_t)(a < b ? a : b);
                uint64_t hi = (uint64_t)(a < b ? b : a);
                keys[f*3+(size_t)e] = lo * (uint64_t)nv + hi;
            }
        qsort(keys, ne, sizeof(uint64_t), cmp_u64);
        for (size_t i = 0; i < ne; ) {
            size_t j = i + 1;
            while (j < ne && keys[j] == keys[i]) j++;
            if (j - i == 1) {
                uint64_t k = keys[i];
                int32_t lo = (int32_t)(k / (uint64_t)nv);
                int32_t hi = (int32_t)(k % (uint64_t)nv);
                is_bnd[lo] = 1; is_bnd[hi] = 1;
            }
            i = j;
        }
    }

    size_t nbf = (n_phase1_bridge <= nf) ? n_phase1_bridge : nf;

    int32_t *sub = (int32_t *)ARENA_ALLOC(arena, (size_t)((nv?nv:1)*sizeof(int32_t)));
    float   *subpts = (float *)ARENA_ALLOC(arena, (size_t)((nv?nv:1)*3*sizeof(float)));
    size_t   m = 0;
    for (size_t v = 0; v < nv; v++) {
        if (!is_bnd[v] || vert_sheet[v] < 0) continue;
        if (!near_seam(&verts[v*3], (double)cube_size, band)) continue;
        sub[m] = (int32_t)v;
        subpts[m*3+0] = verts[v*3+0];
        subpts[m*3+1] = verts[v*3+1];
        subpts[m*3+2] = verts[v*3+2];
        m++;
    }

    size_t cap_raw = nbf + m * 16 + 1;
    PairW *raw = (PairW *)ARENA_ALLOC(arena, (size_t)(cap_raw*sizeof(PairW)));
    size_t nraw = 0;

    for (size_t k = 0; k < nbf; k++) {
        size_t f = nf - nbf + k;
        int32_t s0 = vert_sheet[faces[f*3+0]];
        int32_t s1 = vert_sheet[faces[f*3+1]];
        int32_t s2 = vert_sheet[faces[f*3+2]];
        int32_t ss[3]; int ns = 0;
        int32_t cand[3] = { s0, s1, s2 };
        for (int t = 0; t < 3; t++) {
            if (cand[t] < 0) continue;
            int seen = 0;
            for (int u = 0; u < ns; u++) if (ss[u] == cand[t]) seen = 1;
            if (!seen) ss[ns++] = cand[t];
        }
        if (ns == 2 && nraw < cap_raw) {
            int32_t a = ss[0] < ss[1] ? ss[0] : ss[1];
            int32_t b = ss[0] < ss[1] ? ss[1] : ss[0];
            raw[nraw].a = a; raw[nraw].b = b; raw[nraw].w = p.vote_weight; nraw++;
        }
    }

    if (m >= 2) {
        KDTree_T kt = KDTree_new(arena, subpts, m);
        float r2 = (float)(p.overlap_r * p.overlap_r);
        int32_t nb[32];
        for (size_t k = 0; k < m; k++) {
            size_t got = KDTree_ball_query(kt, &subpts[k*3], r2, nb, 32);
            if (got > 32) got = 32;
            for (size_t t = 0; t < got; t++) {
                size_t k2 = (size_t)nb[t];
                if (k2 <= k) continue;
                int32_t sa = vert_sheet[sub[k]], sb = vert_sheet[sub[k2]];
                if (sa == sb) continue;
                if (nraw >= cap_raw) break;
                int32_t a = sa < sb ? sa : sb, b = sa < sb ? sb : sa;
                raw[nraw].a = a; raw[nraw].b = b; raw[nraw].w = 1.0; nraw++;
            }
        }
    }

    aggregate_pairs(raw, &nraw);
    st.n_candidates = nraw;

    if (nraw == 0) {
        Arena_restore(arena, scratch);
        if (stats) *stats = st;
        return 0;
    }

    int32_t *best_j   = (int32_t *)ARENA_ALLOC(arena, (size_t)(n_sheets*sizeof(int32_t)));
    double  *best_s   = (double  *)ARENA_ALLOC(arena, (size_t)(n_sheets*sizeof(double)));
    double  *second_s = (double  *)ARENA_ALLOC(arena, (size_t)(n_sheets*sizeof(double)));
    for (size_t s = 0; s < n_sheets; s++) { best_j[s] = -1; best_s[s] = -1.0; second_s[s] = -1.0; }

    for (size_t i = 0; i < nraw; i++) {
        int32_t a = raw[i].a, b = raw[i].b; double sc = raw[i].w;
        if (a < 0 || b < 0 || (size_t)a >= n_sheets || (size_t)b >= n_sheets) continue;
        if (sc > best_s[a]) { second_s[a] = best_s[a]; best_s[a] = sc; best_j[a] = b; }
        else if (sc > second_s[a]) { second_s[a] = sc; }
        if (sc > best_s[b]) { second_s[b] = best_s[b]; best_s[b] = sc; best_j[b] = a; }
        else if (sc > second_s[b]) { second_s[b] = sc; }
    }

    PairW *conf = (PairW *)ARENA_ALLOC(arena, (size_t)(nraw*sizeof(PairW)));
    size_t nconf = 0;
    for (size_t i = 0; i < nraw; i++) {
        int32_t a = raw[i].a, b = raw[i].b; double sc = raw[i].w;
        if (a < 0 || b < 0 || (size_t)a >= n_sheets || (size_t)b >= n_sheets) continue;
        if (best_j[a] != b || best_j[b] != a) continue;
        if (sc < p.min_score) continue;
        if (second_s[a] >= p.margin_frac * sc) continue;
        if (second_s[b] >= p.margin_frac * sc) continue;
        conf[nconf].a = a; conf[nconf].b = b; conf[nconf].w = sc; nconf++;
    }
    st.n_pairs = nconf;

    int32_t *acc = NULL; size_t nacc = 0, cap_acc = 0;
    uint8_t *want = (uint8_t *)ARENA_ALLOC(arena, (size_t)(nv*sizeof(uint8_t)));

    for (size_t i = 0; i < nconf; i++) {
        int32_t sa = conf[i].a, sb = conf[i].b;
        for (size_t v = 0; v < nv; v++)
            want[v] = (vert_sheet[v] == sa || vert_sheet[v] == sb) ? 1 : 0;

        int32_t *pf = NULL; size_t pnf = 0, pnb = 0;
        Arena_Mark pm = Arena_save(arena);
        int rc = SeamWeld_bridge(arena, verts, nv, faces, nf,
                                 cube_size, 1.5f, p.rho_max, p.band,
                                 want, NULL  ,
                                 &pf, &pnf, &pnb);
        if (rc == 0 && pnb > 0 && pf && pnb <= pnf) {
            if (nacc + pnb > cap_acc) {
                cap_acc = (nacc + pnb) * 2 + 16;
                int32_t *na = (int32_t *)realloc(acc, cap_acc*3*sizeof(int32_t));
                if (!na) { Arena_restore(arena, pm); break; }
                acc = na;
            }
            memcpy(acc + nacc*3, pf + (pnf - pnb)*3, pnb*3*sizeof(int32_t));
            nacc += pnb;
        }
        Arena_restore(arena, pm);
    }

    st.n_faces_added = nacc;

    Arena_restore(arena, scratch);
    if (nacc > 0) {
        size_t tot = nf + nacc;
        int32_t *out = (int32_t *)ARENA_ALLOC(arena, (size_t)(tot*3*sizeof(int32_t)));
        memcpy(out, faces, nf*3*sizeof(int32_t));
        memcpy(out + nf*3, acc, nacc*3*sizeof(int32_t));
        *out_faces = out; *out_nf = tot;
    }
    free(acc);

    if (stats) *stats = st;
    return 0;
}

static void sr_add_ribbon(float *V, float *N, int32_t *F, size_t *nv, size_t *nf,
                          float x0, int nx, float z0, float z1, int curve)
{
    size_t base = *nv;
    float zc[2] = { z0, z1 };
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < nx; i++) {
            size_t v = *nv;
            float x = x0 + (float)i;
            float y = curve ? 0.5f * sinf(x) : 0.0f;
            V[v*3+0] = x; V[v*3+1] = y; V[v*3+2] = zc[c];
            if (curve) {
                float dy = 0.5f * cosf(x);
                float nx0 = -dy, ny0 = 1.0f, nz0 = 0.0f;
                float nl = sqrtf(nx0*nx0 + ny0*ny0 + nz0*nz0);
                N[v*3+0] = nx0/nl; N[v*3+1] = ny0/nl; N[v*3+2] = nz0/nl;
            } else { N[v*3+0] = 0.0f; N[v*3+1] = 1.0f; N[v*3+2] = 0.0f; }
            (*nv)++;
        }
    for (int i = 0; i + 1 < nx; i++) {
        int32_t v00 = (int32_t)(base + 0*(size_t)nx + (size_t)i);
        int32_t v01 = (int32_t)(base + 0*(size_t)nx + (size_t)i + 1);
        int32_t v10 = (int32_t)(base + 1*(size_t)nx + (size_t)i);
        int32_t v11 = (int32_t)(base + 1*(size_t)nx + (size_t)i + 1);
        F[*nf*3+0]=v00; F[*nf*3+1]=v10; F[*nf*3+2]=v11; (*nf)++;
        F[*nf*3+0]=v00; F[*nf*3+1]=v11; F[*nf*3+2]=v01; (*nf)++;
    }
}

static void sr_test_params(SheetReweldParams *p)
{
    SheetReweld_default_params(p);
    p->overlap_r = 3.0; p->min_score = 3.0; p->margin_frac = 0.5;
    p->rho_max = 4.0f; p->band = 6.0f;
}

int SheetReweld_selftest(void)
{
    int fails = 0;
    Arena_T A = Arena_new();
    const float CUBE = 128.0f;

    {
        float V[9*3]; int32_t F[3*3];
        for (int t = 0; t < 3; t++) {
            for (int k = 0; k < 3; k++) {
                V[(t*3+k)*3+0]=(float)(t*10+k); V[(t*3+k)*3+1]=0; V[(t*3+k)*3+2]=0;
            }
            F[t*3+0]=t*3+0; F[t*3+1]=t*3+1; F[t*3+2]=t*3+2;
        }
        int32_t *vs = NULL; size_t nsh = 0;
        SheetReweld_label(A, F, 3, 9, &vs, &nsh);
        if (nsh != 3) { printf("  [A] FAIL n_sheets=%zu (want 3)\n", nsh); fails++; }
        else if (vs[0]==vs[3] || vs[3]==vs[6] || vs[0]==vs[6])
            { printf("  [A] FAIL labels not distinct\n"); fails++; }
        else printf("  [A] label 3 components: PASS\n");
    }

    {
        float *V = (float*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(float)));
        float *N = (float*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(float)));
        int32_t *F=(int32_t*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(int32_t)));
        size_t nv=0,nf=0;
        sr_add_ribbon(V,N,F,&nv,&nf, 0.0f, 9, 124.0f, 127.0f, 0);
        sr_add_ribbon(V,N,F,&nv,&nf, 0.0f, 9, 129.0f, 132.0f, 0);
        int32_t *vs=NULL; size_t nsh=0; SheetReweld_label(A,F,nf,nv,&vs,&nsh);
        SheetReweldParams p; sr_test_params(&p);
        int32_t *of=NULL; size_t onf=0; SheetReweldStats s;
        SheetReweld_process(A,V,nv,F,nf,0,vs,nsh,CUBE,NULL,&p,&of,&onf,&s);
        if (s.n_pairs != 1) { printf("  [B1] FAIL n_pairs=%zu (want 1)\n", s.n_pairs); fails++; }
        else printf("  [B1] unambiguous pair confirmed: PASS\n");
    }

    {
        float *V=(float*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(float)));
        float *N=(float*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(float)));
        int32_t *F=(int32_t*)ARENA_ALLOC(A,(size_t)(64*3*sizeof(int32_t)));
        size_t nv=0,nf=0;
        sr_add_ribbon(V,N,F,&nv,&nf, 0.0f, 9, 124.0f, 127.0f, 0);
        sr_add_ribbon(V,N,F,&nv,&nf, 0.0f, 3, 129.0f, 132.0f, 0);
        sr_add_ribbon(V,N,F,&nv,&nf, 6.0f, 3, 129.0f, 132.0f, 0);
        int32_t *vs=NULL; size_t nsh=0; SheetReweld_label(A,F,nf,nv,&vs,&nsh);
        SheetReweldParams p; sr_test_params(&p);
        int32_t *of=NULL; size_t onf=0; SheetReweldStats s;
        SheetReweld_process(A,V,nv,F,nf,0,vs,nsh,CUBE,NULL,&p,&of,&onf,&s);
        if (s.n_pairs != 0) { printf("  [B2] FAIL n_pairs=%zu (want 0, ambiguity)\n", s.n_pairs); fails++; }
        else printf("  [B2] ambiguous sheet rejected: PASS\n");
    }

    {
        const int NR = 5, NC = 3;
        int nvC = 2   * NR * NC;
        float *V=(float*)ARENA_ALLOC(A,(size_t)((size_t)nvC*3*sizeof(float)));
        float *N=(float*)ARENA_ALLOC(A,(size_t)((size_t)nvC*3*sizeof(float)));
        int32_t *F=(int32_t*)ARENA_ALLOC(A,(size_t)((size_t)nvC*4*sizeof(int32_t)));
        size_t nv=0,nf=0;
        float zcol[6] = {124.5f,126.0f,127.5f, 128.5f,130.0f,131.5f};
        for (int sh = 0; sh < 2; sh++) {
            size_t base = nv;
            for (int c = 0; c < NC; c++) {
                int pc = sh*NC + c;
                float pp = 1.1f * (float)pc;
                float x = 0.4f * sinf(pp);
                float dxdp = 0.4f * 1.1f * cosf(pp);
                float nx0=-3.0f, ny0=0.0f, nz0=dxdp;
                float nl=sqrtf(nx0*nx0+ny0*ny0+nz0*nz0);
                for (int r = 0; r < NR; r++) {
                    size_t v = nv;
                    V[v*3+0]=x; V[v*3+1]=(float)r; V[v*3+2]=zcol[pc];
                    N[v*3+0]=nx0/nl; N[v*3+1]=ny0/nl; N[v*3+2]=nz0/nl;
                    nv++;
                }
            }
            for (int c = 0; c + 1 < NC; c++)
                for (int r = 0; r + 1 < NR; r++) {
                    int32_t a=(int32_t)(base+(size_t)c*(size_t)NR+(size_t)r);
                    int32_t b=(int32_t)(base+(size_t)c*(size_t)NR+(size_t)r+1);
                    int32_t d=(int32_t)(base+(size_t)(c+1)*(size_t)NR+(size_t)r);
                    int32_t e=(int32_t)(base+(size_t)(c+1)*(size_t)NR+(size_t)r+1);
                    F[nf*3+0]=a; F[nf*3+1]=d; F[nf*3+2]=e; nf++;
                    F[nf*3+0]=a; F[nf*3+1]=e; F[nf*3+2]=b; nf++;
                }
        }
        int32_t *vs=NULL; size_t nsh=0; SheetReweld_label(A,F,nf,nv,&vs,&nsh);
        SheetReweldParams p; sr_test_params(&p);
        int32_t *of=NULL; size_t onf=0; SheetReweldStats s;
        SheetReweld_process(A,V,nv,F,nf,0,vs,nsh,CUBE,NULL,&p,&of,&onf,&s);
        if (s.n_pairs != 1) { printf("  [C] FAIL n_pairs=%zu (want 1)\n", s.n_pairs); fails++; }
        else if (s.n_faces_added == 0) { printf("  [C] FAIL no bridge faces added\n"); fails++; }
        else if (onf <= nf) { printf("  [C] FAIL output not grown (%zu<=%zu)\n", onf, nf); fails++; }
        else printf("  [C] restricted reweld added %zu faces: PASS\n", s.n_faces_added);
    }

    Arena_dispose(&A);
    if (fails == 0) printf("ALL SHEET_REWELD TESTS PASSED\n");
    else printf("%d SHEET_REWELD TEST(S) FAILED\n", fails);
    return fails;
}
