#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define INF 1e300

typedef struct { double *p; size_t n, cap; } DArr;
typedef struct { int    *p; size_t n, cap; } IArr;
static void darr_push3(DArr *d, double x, double y, double z) {
    if (d->n + 3 > d->cap) { d->cap = d->cap ? d->cap * 2 : 3072;
        d->p = realloc(d->p, d->cap * sizeof(double)); }
    d->p[d->n++] = x; d->p[d->n++] = y; d->p[d->n++] = z;
}
static void iarr_push3(IArr *a, int x, int y, int z) {
    if (a->n + 3 > a->cap) { a->cap = a->cap ? a->cap * 2 : 3072;
        a->p = realloc(a->p, a->cap * sizeof(int)); }
    a->p[a->n++] = x; a->p[a->n++] = y; a->p[a->n++] = z;
}

static int uf_find(int *uf, int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }

typedef struct {
    int    id;
    int    root;
    long   nf;
    long   nv;
    double bbmin[3], bbmax[3], cen[3];
} Comp;
static int comp_cmp_nf(const void *a, const void *b) {
    const Comp *ca = a, *cb = b;
    if (ca->nf != cb->nf) return ca->nf < cb->nf ? 1 : -1;
    return 0;
}

enum { K_SINGLE = 0, K_SEPARATE = 1, K_SEAM = 2, K_PARTIAL = 3, K_DOUBLE = 4 };
static const char *KLASS_TAG[5] = { "SINGLE", "SEPARATE", "SEAM-SPLIT", "PARTIAL-DOUBLE", "DOUBLE" };
typedef struct {
    int  comps;
    double closest_min;
    int  closest_a, closest_b;
    long closest_near;
    double cover_a, cover_b;
    int  klass;
    double crease_deg;
} Result;

static Result analyze(const double *V, size_t nv, const int *F, size_t nf,
                      double gap, double cell, const char *dumppath,
                      const char *seppath, int verbose, int want_crease)
{
    Result R = { 0, INF, -1, -1, 0, 0.0, 0.0, K_SINGLE, -1.0 };
    if (nv == 0 || nf == 0) { if (verbose) printf("  (empty mesh: nv=%zu nf=%zu)\n", nv, nf); return R; }

    int *uf = malloc(nv * sizeof(int));
    for (size_t i = 0; i < nv; i++) uf[i] = (int)i;
    for (size_t fi = 0; fi < nf; fi++) {
        int a = F[fi*3], b = F[fi*3+1], c = F[fi*3+2];
        int ra = uf_find(uf,a), rb = uf_find(uf,b);
        if (ra != rb) uf[rb] = ra;
        ra = uf_find(uf,a); int rc = uf_find(uf,c);
        if (ra != rc) uf[rc] = ra;
    }

    char *ref = calloc(nv, 1);
    for (size_t i = 0; i < nf*3; i++) ref[F[i]] = 1;
    long *rootnf = calloc(nv, sizeof(long));
    long *rootnv = calloc(nv, sizeof(long));
    for (size_t fi = 0; fi < nf; fi++) rootnf[uf_find(uf, F[fi*3])]++;
    for (size_t i = 0; i < nv; i++) if (ref[i]) rootnv[uf_find(uf,(int)i)]++;

    int C = 0;
    for (size_t i = 0; i < nv; i++) if (ref[i] && uf_find(uf,(int)i) == (int)i) C++;
    Comp *comp = malloc((size_t)C * sizeof(Comp));
    int ci = 0;
    for (size_t i = 0; i < nv; i++) if (ref[i] && uf_find(uf,(int)i) == (int)i) {
        comp[ci].root = (int)i; comp[ci].nf = rootnf[i]; comp[ci].nv = rootnv[i];
        for (int k = 0; k < 3; k++) { comp[ci].bbmin[k] = INF; comp[ci].bbmax[k] = -INF; comp[ci].cen[k] = 0; }
        ci++;
    }
    qsort(comp, (size_t)C, sizeof(Comp), comp_cmp_nf);
    for (int k = 0; k < C; k++) comp[k].id = k + 1;

    int *root2id = calloc(nv, sizeof(int));
    for (int k = 0; k < C; k++) root2id[comp[k].root] = comp[k].id;
    int *cov = calloc(nv, sizeof(int));
    for (size_t i = 0; i < nv; i++) if (ref[i]) cov[i] = root2id[uf_find(uf,(int)i)];

    for (size_t i = 0; i < nv; i++) if (ref[i]) {
        Comp *c = &comp[cov[i]-1];
        for (int k = 0; k < 3; k++) { double x = V[i*3+k];
            if (x < c->bbmin[k]) c->bbmin[k] = x;
            if (x > c->bbmax[k]) c->bbmax[k] = x;
            c->cen[k] += x; }
    }
    for (int k = 0; k < C; k++) for (int j = 0; j < 3; j++)
        comp[k].cen[j] /= (double)comp[k].nv;

    if (verbose) {
        printf("  components: %d   (labelled 1..%d by descending face count)\n", C, C);
        printf("    id   faces    verts      bbox z[lo,hi]        y[lo,hi]        x[lo,hi]    extent  centroid(z,y,x)\n");
        for (int k = 0; k < C; k++) {
            Comp *c = &comp[k];
            double ez = c->bbmax[0]-c->bbmin[0], ey = c->bbmax[1]-c->bbmin[1], ex = c->bbmax[2]-c->bbmin[2];
            double emax = ez > ey ? (ez > ex ? ez : ex) : (ey > ex ? ey : ex);
            printf("   %3d %7ld %8ld   [%7.1f,%7.1f] [%7.1f,%7.1f] [%7.1f,%7.1f] %6.1f  (%.1f,%.1f,%.1f)\n",
                   c->id, c->nf, c->nv,
                   c->bbmin[0], c->bbmax[0], c->bbmin[1], c->bbmax[1], c->bbmin[2], c->bbmax[2],
                   emax, c->cen[0], c->cen[1], c->cen[2]);
        }
    }
    R.comps = C;
    if (C < 2) {
        R.klass = K_SINGLE;
        if (verbose) printf("  single component -- nothing to compare.\n");
        free(uf); free(ref); free(rootnf); free(rootnv); free(comp); free(root2id); free(cov);
        return R;
    }

    double gmin[3] = { INF, INF, INF }, gmax[3] = { -INF, -INF, -INF };
    for (size_t i = 0; i < nv; i++) if (ref[i]) for (int k = 0; k < 3; k++) {
        double x = V[i*3+k]; if (x < gmin[k]) gmin[k] = x; if (x > gmax[k]) gmax[k] = x; }
    double cs = cell > 1e-6 ? cell : 1.0;
    int dim[3];
    size_t ncell;
    for (;;) {
        for (int k = 0; k < 3; k++) { dim[k] = (int)((gmax[k]-gmin[k])/cs) + 1; if (dim[k] < 1) dim[k] = 1; }
        ncell = (size_t)dim[0]*(size_t)dim[1]*(size_t)dim[2];
        if (ncell <= 50000000u) break;
        cs *= 2.0;
    }
    int  *cidx = malloc(nv * sizeof(int));
    long *coff = calloc(ncell + 1, sizeof(long));
    for (size_t i = 0; i < nv; i++) {
        if (!ref[i]) { cidx[i] = -1; continue; }
        int g[3];
        for (int k = 0; k < 3; k++) { int q = (int)((V[i*3+k]-gmin[k])/cs);
            if (q < 0) q = 0; if (q >= dim[k]) q = dim[k]-1; g[k] = q; }
        size_t ic = ((size_t)g[0]*(size_t)dim[1] + (size_t)g[1])*(size_t)dim[2] + (size_t)g[2];
        cidx[i] = (int)ic; coff[ic+1]++;
    }
    for (size_t i = 0; i < ncell; i++) coff[i+1] += coff[i];
    int  *cellv = malloc((size_t)coff[ncell] * sizeof(int));
    long *cur = malloc(ncell * sizeof(long));
    for (size_t i = 0; i < ncell; i++) cur[i] = coff[i];
    for (size_t i = 0; i < nv; i++) if (cidx[i] >= 0) cellv[cur[cidx[i]]++] = (int)i;
    free(cur);

    double *pmin2 = malloc((size_t)C*(size_t)C * sizeof(double));
    long   *pcnt  = calloc((size_t)C*(size_t)C, sizeof(long));
    for (size_t i = 0; i < (size_t)C*(size_t)C; i++) pmin2[i] = INF;
    double *nn2 = malloc(nv * sizeof(double));
    int    *nnc = calloc(nv, sizeof(int));
    int    *nn_idx = malloc(nv * sizeof(int));
    for (size_t i = 0; i < nv; i++) { nn2[i] = INF; nn_idx[i] = -1; }
    double gap2 = gap*gap;
    int planeYZ = dim[1]*dim[2];
    for (size_t i = 0; i < nv; i++) {
        if (cidx[i] < 0) continue;
        int ai = cov[i];
        double xi = V[i*3], yi = V[i*3+1], zi = V[i*3+2];
        int g0 = cidx[i] / planeYZ, rem = cidx[i] % planeYZ, g1 = rem / dim[2], g2 = rem % dim[2];
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            int n0 = g0+dz, n1 = g1+dy, n2 = g2+dx;
            if (n0 < 0 || n1 < 0 || n2 < 0 || n0 >= dim[0] || n1 >= dim[1] || n2 >= dim[2]) continue;
            size_t nc = ((size_t)n0*(size_t)dim[1] + (size_t)n1)*(size_t)dim[2] + (size_t)n2;
            for (long t = coff[nc]; t < coff[nc+1]; t++) {
                int w = cellv[t];
                int aw = cov[w];
                if (aw == ai) continue;
                double ddx = V[w*3]-xi, ddy = V[w*3+1]-yi, ddz = V[w*3+2]-zi;
                double d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                if (d2 < nn2[i]) { nn2[i] = d2; nnc[i] = aw; nn_idx[i] = w; }
                if ((size_t)w > i) {
                    int lo = ai < aw ? ai : aw, hi = ai < aw ? aw : ai;
                    size_t pij = (size_t)(lo-1)*(size_t)C + (size_t)(hi-1);
                    if (d2 < pmin2[pij]) pmin2[pij] = d2;
                    if (d2 <= gap2) pcnt[pij]++;
                }
            }
        }
    }

    if (verbose)
        printf("\n  inter-component gaps (exact within search window cell=%.1f vox; near-pair threshold gap=%.2f vox):\n", cs, gap);
    double best = INF; int ba = -1, bb = -1; long bnear = 0;
    for (int a = 0; a < C; a++) for (int b = a+1; b < C; b++) {
        size_t pij = (size_t)a*(size_t)C + (size_t)b;
        if (pmin2[pij] < INF/2) {
            double mn = sqrt(pmin2[pij]);
            if (verbose)
                printf("    comp %d <-> comp %d : min gap %.4f vox,  %ld vertex pair%s within %.2f vox\n",
                       a+1, b+1, mn, pcnt[pij], pcnt[pij]==1?"":"s", gap);
            if (mn < best) { best = mn; ba = a+1; bb = b+1; bnear = pcnt[pij]; }
        } else if (verbose) {
            printf("    comp %d <-> comp %d : gap > %.1f vox (no vertices within window -- SEPARATE)\n", a+1, b+1, cs);
        }
    }
    R.closest_min = best; R.closest_a = ba; R.closest_b = bb; R.closest_near = bnear;
    if (ba < 0) R.klass = K_SEPARATE;

    if (ba > 0) {
        long cov_a = 0, cov_b = 0; double sda = 0, sdb = 0;
        for (size_t i = 0; i < nv; i++) {
            if (!ref[i] || nn2[i] > gap2) continue;
            if (cov[i] == ba && nnc[i] == bb) { cov_a++; sda += sqrt(nn2[i]); }
            else if (cov[i] == bb && nnc[i] == ba) { cov_b++; sdb += sqrt(nn2[i]); }
        }
        long na = comp[ba-1].nv, nb = comp[bb-1].nv;
        double fa = na ? 100.0*(double)cov_a/(double)na : 0.0;
        double fb = nb ? 100.0*(double)cov_b/(double)nb : 0.0;
        R.cover_a = fa; R.cover_b = fb;
        R.klass = (fa > 50.0 && fb > 50.0) ? K_DOUBLE
                : (fa > 50.0 || fb > 50.0) ? K_PARTIAL : K_SEAM;
        if (verbose) {
            printf("\n  closest-pair coincidence (within %.2f vox):\n", gap);
            printf("    comp %d: %ld/%ld verts (%.0f%%) coincide with comp %d  (mean %.3f vox)\n",
                   ba, cov_a, na, fa, bb, cov_a ? sda/(double)cov_a : 0.0);
            printf("    comp %d: %ld/%ld verts (%.0f%%) coincide with comp %d  (mean %.3f vox)\n",
                   bb, cov_b, nb, fb, ba, cov_b ? sdb/(double)cov_b : 0.0);
            printf("    -> %s\n", R.klass == K_DOUBLE
                   ? "high both sides: islands OVERLAP the same region (doubled/shingled sheet)"
                   : R.klass == K_PARTIAL
                     ? "high one side: one island sits atop part of the other (partial double)"
                     : "thin band only: they meet along a SEAM (one split layer)");
        }
    }

    if (want_crease && ba > 0) {
        double *vn = (double *)calloc(nv * 3, sizeof(double));
        for (size_t fi = 0; fi < nf; fi++) {
            int i0 = F[fi*3], i1 = F[fi*3+1], i2 = F[fi*3+2];
            double e1[3], e2[3], fnv[3];
            for (int k = 0; k < 3; k++) { e1[k] = V[i1*3+k]-V[i0*3+k]; e2[k] = V[i2*3+k]-V[i0*3+k]; }
            fnv[0] = e1[1]*e2[2]-e1[2]*e2[1];
            fnv[1] = e1[2]*e2[0]-e1[0]*e2[2];
            fnv[2] = e1[0]*e2[1]-e1[1]*e2[0];
            for (int k = 0; k < 3; k++) { vn[i0*3+k]+=fnv[k]; vn[i1*3+k]+=fnv[k]; vn[i2*3+k]+=fnv[k]; }
        }
        for (size_t i = 0; i < nv; i++) {
            double *p = &vn[i*3], L = sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
            if (L > 1e-12) { p[0]/=L; p[1]/=L; p[2]/=L; }
        }
        static const double DEG = 57.29577951308232;
        static const double ab_edge[7] = { 5,15,30,45,60,75,90 };
        long hist[7] = {0,0,0,0,0,0,0}, ncr = 0;
        double sum_acute = 0, sum_raw = 0;
        double mna[3] = {0,0,0}, mnb[3] = {0,0,0};
        for (size_t i = 0; i < nv; i++) {
            if (!ref[i] || nn2[i] > gap2 || nn_idx[i] < 0) continue;
            int vci = cov[i], vcj = cov[nn_idx[i]];
            int ab = (vci == ba && vcj == bb), ba_ = (vci == bb && vcj == ba);
            if (!ab && !ba_) continue;
            double *ni = &vn[i*3], *nj = &vn[nn_idx[i]*3];
            double dot = ni[0]*nj[0]+ni[1]*nj[1]+ni[2]*nj[2];
            if (dot > 1) dot = 1; if (dot < -1) dot = -1;
            double adot = dot < 0 ? -dot : dot;
            double acute = acos(adot)*DEG, raw = acos(dot)*DEG;
            sum_acute += acute; sum_raw += raw; ncr++;
            int bk = 0; while (bk < 6 && acute >= ab_edge[bk]) bk++; hist[bk]++;
            if (ab) for (int k = 0; k < 3; k++) mna[k] += ni[k];
            else    for (int k = 0; k < 3; k++) mnb[k] += nj[k];
        }
        double mean_acute = ncr ? sum_acute/(double)ncr : -1.0;
        R.crease_deg = mean_acute;
        double la = sqrt(mna[0]*mna[0]+mna[1]*mna[1]+mna[2]*mna[2]);
        double lb = sqrt(mnb[0]*mnb[0]+mnb[1]*mnb[1]+mnb[2]*mnb[2]);
        double bulk = -1.0;
        if (la > 1e-9 && lb > 1e-9) {
            double d = (mna[0]*mnb[0]+mna[1]*mnb[1]+mna[2]*mnb[2])/(la*lb);
            if (d < 0) d = -d; if (d > 1) d = 1;
            bulk = acos(d)*DEG;
        }
        if (verbose) {
            printf("\n  crease/dihedral test (closest pair comp %d <-> comp %d, face-winding normals):\n", ba, bb);
            printf("    facing-normal angle over %ld seam vertex pairs: mean ACUTE = %.1f deg (raw mean = %.1f deg)\n",
                   ncr, mean_acute, ncr ? sum_raw/(double)ncr : 0.0);
            printf("    acute hist [0-5 5-15 15-30 30-45 45-60 60-75 75-90]: %ld %ld %ld %ld %ld %ld %ld\n",
                   hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6]);
            printf("    seam-band mean-normal angle: %.1f deg (acute)\n", bulk);
            printf("    -> %s\n", mean_acute < 10.0
                   ? "COPLANAR continuation (< 10 deg) -- NOT a crease; the split is a sampling/empty-ball stall"
                   : mean_acute < 25.0
                     ? "mild bend (10-25 deg) -- borderline"
                     : "CREASE / fold CONFIRMED (>= 25 deg) -- the seam runs along a sharp surface bend");
        }
        free(vn);
    }

    if (verbose) {
        printf("\n  VERDICT:\n");
        if (ba < 0) {
            printf("    all %d components are > %.1f vox apart -- genuinely separate sheets.\n", C, cs);
        } else {
            Comp *ca = &comp[ba-1], *cb = &comp[bb-1];
            if (best <= gap) {
                printf("    comp %d (%ld f) and comp %d (%ld f) meet at %.4f vox over %ld vertex pairs\n"
                       "    (<= weld distance %.2f vox). They are ONE sheet that BPA failed to stitch:\n"
                       "    the points coincide but no triangle bridges the islands.\n",
                       ba, ca->nf, bb, cb->nf, best, bnear, gap);
            } else if (best <= 2.5) {
                printf("    closest pair comp %d (%ld f) <-> comp %d (%ld f) is %.4f vox apart (just beyond\n"
                       "    weld %.2f vox) -- almost certainly one sheet; the BPA ball (~rho) couldn't bridge it.\n",
                       ba, ca->nf, bb, cb->nf, best, gap);
            } else {
                printf("    closest pair comp %d <-> comp %d is %.4f vox apart.\n", ba, bb, best);
            }
        }
    }

    if (dumppath) {
        static const double PAL[6][3] = {
            {0.75,0.75,0.78}, {0.15,0.85,0.25}, {0.90,0.15,0.15},
            {0.20,0.45,0.95}, {0.95,0.75,0.10}, {0.65,0.25,0.85} };
        FILE *of = fopen(dumppath, "w");
        if (!of) { printf("  ERROR: cannot write dump %s\n", dumppath); }
        else {
            for (size_t i = 0; i < nv; i++) {
                double col[3] = { 0.40, 0.40, 0.40 };
                if (ref[i]) {
                    const double *pc = PAL[(cov[i]-1) % 6];
                    col[0] = pc[0]; col[1] = pc[1]; col[2] = pc[2];
                    if (nn2[i] <= gap2) { col[0] = 1.0; col[1] = 1.0; col[2] = 0.15; }
                }
                fprintf(of, "v %.6f %.6f %.6f %.4f %.4f %.4f\n",
                        V[i*3], V[i*3+1], V[i*3+2], col[0], col[1], col[2]);
            }
            for (size_t fi = 0; fi < nf; fi++)
                fprintf(of, "f %d %d %d\n", F[fi*3]+1, F[fi*3+1]+1, F[fi*3+2]+1);
            fclose(of);
            if (verbose)
                printf("\n  wrote per-component colored OBJ -> %s\n"
                       "   (comp1 gray, comp2 green, comp3 red, comp4 blue, ...; near-seam verts within %.2f vox = bright yellow)\n",
                       dumppath, gap);
        }
    }

    if (seppath) {
        int *remap = malloc(nv * sizeof(int));
        int nwritten = 0;
        for (int k = 0; k < C; k++) {
            int cid = comp[k].id;
            for (size_t i = 0; i < nv; i++) remap[i] = -1;
            int newnv = 0;
            for (size_t i = 0; i < nv; i++) if (ref[i] && cov[i] == cid) remap[i] = newnv++;
            char fpath[1280];
            snprintf(fpath, sizeof(fpath), "%s_c%03d_%ldf.obj", seppath, cid, comp[k].nf);
            FILE *of = fopen(fpath, "w");
            if (!of) { printf("  ERROR: cannot write sep dump %s\n", fpath); continue; }
            for (size_t i = 0; i < nv; i++) if (remap[i] >= 0)
                fprintf(of, "v %.6f %.6f %.6f\n", V[i*3], V[i*3+1], V[i*3+2]);
            for (size_t fi = 0; fi < nf; fi++) {
                int a = F[fi*3];
                if (cov[a] != cid) continue;
                fprintf(of, "f %d %d %d\n",
                        remap[F[fi*3]]+1, remap[F[fi*3+1]]+1, remap[F[fi*3+2]]+1);
            }
            fclose(of);
            nwritten++;
            if (verbose)
                printf("    sep[%d]: comp %d -> %s  (%d verts, %ld faces)\n",
                       nwritten, cid, fpath, newnv, comp[k].nf);
        }
        free(remap);
        if (verbose) printf("  wrote %d per-component OBJ file(s) with prefix '%s'\n", nwritten, seppath);
    }

    free(uf); free(ref); free(rootnf); free(rootnv); free(comp); free(root2id); free(cov);
    free(cidx); free(coff); free(cellv); free(pmin2); free(pcnt); free(nn2); free(nnc); free(nn_idx);
    return R;
}

static int selftest(void)
{
    int fails = 0;
    printf("=== obj_components selftest ===\n");

    {
        double V[] = { 0,0,0,  0,0,1,  0,1,0,  0,1,1 };
        int    F[] = { 0,1,2,  1,3,2 };
        printf("[case1] connected quad (expect comps=1)\n");
        Result r = analyze(V, 4, F, 2, 1.5, 4.0, NULL, NULL, 1,0);
        if (r.comps != 1) { printf("  FAIL: comps=%d != 1\n", r.comps); fails++; }
        else printf("  ok: comps=1\n");
    }
    {
        double V[] = { 0,0,0,  0,0,1,  0,1,0,   10,0,0,  10,0,1,  10,1,0 };
        int    F[] = { 0,1,2,  3,4,5 };
        printf("[case2] two far triangles (expect comps=2, separate)\n");
        Result r = analyze(V, 6, F, 2, 1.5, 4.0, NULL, NULL, 1,0);
        if (!(r.comps == 2 && r.closest_a < 0)) {
            printf("  FAIL: comps=%d closest_a=%d (expect 2, -1)\n", r.comps, r.closest_a); fails++;
        } else printf("  ok: comps=2, correctly separate\n");
    }
    {
        double V[] = { 0,0,0,  0,0,1,  0,1,0,   0.5,0,0,  0.5,0,1,  0.5,1,0 };
        int    F[] = { 0,1,2,  3,4,5 };
        printf("[case3] two near triangles @0.5 vox (expect comps=2, gap~0.5, near>0)\n");
        Result r = analyze(V, 6, F, 2, 1.5, 4.0, NULL, NULL, 1,0);
        if (!(r.comps == 2 && r.closest_a > 0 && r.closest_min < 1.5 && r.closest_near > 0)) {
            printf("  FAIL: comps=%d closest_min=%.3f near=%ld\n", r.comps, r.closest_min, r.closest_near); fails++;
        } else printf("  ok: comps=2 gap=%.3f near=%ld (BPA-miss signature)\n", r.closest_min, r.closest_near);
    }
    {
        double V[] = { 0,0,0,  0,2,0,  0,0,2,    1,0,0.3,  1,2,0.3,  3,0,0.3 };
        int    F[] = { 0,1,2,  3,4,5 };
        printf("[case4] 90-deg crease (expect crease_deg ~90)\n");
        Result r = analyze(V, 6, F, 2, 1.5, 4.0, NULL, NULL, 1,1);
        if (!(r.comps == 2 && r.crease_deg > 80.0 && r.crease_deg <= 90.0001)) {
            printf("  FAIL: comps=%d crease_deg=%.1f (expect ~90)\n", r.comps, r.crease_deg); fails++;
        } else printf("  ok: crease_deg=%.1f (fold confirmed)\n", r.crease_deg);
    }
    {
        double V[] = { 0,0,0,  0,2,0,  0,0,2,    0,0,2.3,  0,2,2.3,  0,0,4 };
        int    F[] = { 0,1,2,  3,4,5 };
        printf("[case5] coplanar continuation (expect crease_deg ~0)\n");
        Result r = analyze(V, 6, F, 2, 1.5, 4.0, NULL, NULL, 1,1);
        if (!(r.comps == 2 && r.crease_deg >= 0.0 && r.crease_deg < 10.0)) {
            printf("  FAIL: comps=%d crease_deg=%.1f (expect ~0)\n", r.comps, r.crease_deg); fails++;
        } else printf("  ok: crease_deg=%.1f (coplanar, not a crease)\n", r.crease_deg);
    }

    {
        double V[] = { 0,0,0,  0,0,1,  0,1,0,   10,0,0,  10,0,1,  10,1,0 };
        int    F[] = { 0,1,2,  3,4,5 };
        const char *pre = "obj_components_selftest_sep";
        printf("[case6] --dump-sep two comps (expect 2 files, 1 face each)\n");
        Result r = analyze(V, 6, F, 2, 1.5, 4.0, NULL, pre, 0, 0);
        int ok = (r.comps == 2);
        for (int cid = 1; cid <= 2 && ok; cid++) {
            char fp[256]; snprintf(fp, sizeof(fp), "%s_c%03d_1f.obj", pre, cid);
            FILE *t = fopen(fp, "r");
            if (!t) { ok = 0; break; }
            int nv6 = 0, nf6 = 0; char ln[256];
            while (fgets(ln, sizeof(ln), t)) {
                if (ln[0]=='v' && ln[1]==' ') nv6++;
                else if (ln[0]=='f' && ln[1]==' ') nf6++;
            }
            fclose(t);
            remove(fp);
            if (!(nv6 == 3 && nf6 == 1)) ok = 0;
        }
        if (!ok) { printf("  FAIL: sep files missing or wrong size\n"); fails++; }
        else printf("  ok: 2 separate OBJs, 3 verts / 1 face each\n");
    }

    printf("=== selftest %s (%d failure%s) ===\n",
           fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 3 : 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <mesh.obj> [--gap <vox=1.5>] [--cell <vox=4.0>] [--dump <out.obj>] [--dump-sep <prefix>] [--brief] [--crease]\n"
            "       %s --selftest\n", argv[0], argv[0]);
        return 2;
    }
    const char *path = argv[1];
    double gap = 1.5, cell = 4.0;
    const char *dump = NULL;
    const char *dumpsep = NULL;
    int brief = 0, want_crease = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--gap")  && i+1 < argc) gap  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cell") && i+1 < argc) cell = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i+1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--dump-sep") && i+1 < argc) dumpsep = argv[++i];
        else if (!strcmp(argv[i], "--brief")) brief = 1;
        else if (!strcmp(argv[i], "--crease")) want_crease = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (gap > cell) { fprintf(stderr, "note: --gap (%.2f) > --cell (%.2f); raising cell to gap\n", gap, cell); cell = gap; }

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 1; }
    DArr V = {0}; IArr Fc = {0};
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            double x = 0, y = 0, z = 0;
            if (sscanf(line+2, "%lf %lf %lf", &x, &y, &z) == 3) darr_push3(&V, x, y, z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            int idx[64]; int ni = 0;
            const char *p = line + 1;
            while (*p && ni < 64) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '\0' || *p == '\n' || *p == '\r') break;
                int vi = (int)strtol(p, (char**)&p, 10);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
                if (vi < 0) vi = (int)(V.n/3) + vi; else vi -= 1;
                idx[ni++] = vi;
            }
            for (int i = 1; i+1 < ni; i++) iarr_push3(&Fc, idx[0], idx[i], idx[i+1]);
        }
    }
    fclose(f);

    size_t nv = V.n/3, nf = Fc.n/3;
    if (!brief) {
        printf("=== obj_components: %s ===\n", path);
        printf("  V=%zu  F=%zu  gap=%.2f vox  cell=%.2f vox\n", nv, nf, gap, cell);
    }
    Result r = analyze(V.p, nv, Fc.p, nf, gap, cell, dump, dumpsep, !brief, want_crease);

    if (brief) {
        const char *base = path;
        for (const char *s = path; *s; s++) if (*s == '/' || *s == '\\') base = s + 1;
        char crease[32] = "";
        if (want_crease && r.crease_deg >= 0.0) snprintf(crease, sizeof(crease), " crease=%.0fdeg", r.crease_deg);
        if (r.comps < 2) {
            printf("%-52s comps=%d  %s\n", base, r.comps, KLASS_TAG[K_SINGLE]);
        } else if (r.closest_a < 0) {
            printf("%-52s comps=%d  %s\n", base, r.comps, KLASS_TAG[r.klass]);
        } else {
            printf("%-52s comps=%d  c%d-c%d gap=%.3f near=%ld cover=%.0f/%.0f%%%s  %s\n",
                   base, r.comps, r.closest_a, r.closest_b, r.closest_min,
                   r.closest_near, r.cover_a, r.cover_b, crease, KLASS_TAG[r.klass]);
        }
    }

    free(V.p); free(Fc.p);
    return 0;
}
