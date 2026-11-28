#include "seam_weld.h"
#include "ball_pivot.h"
#include "seam_planes.h"
#include "../common/pipeline_constants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/pitch_table.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void compute_vertex_normals(const float *verts, size_t nv,
                                   const int32_t *faces, size_t nf,
                                   float *normals)
{
    memset(normals, 0, nv * 3 * sizeof(float));
    for (size_t f = 0; f < nf; f++) {
        int32_t i0 = faces[f*3+0], i1 = faces[f*3+1], i2 = faces[f*3+2];
        const float *p0 = &verts[(size_t)i0*3];
        const float *p1 = &verts[(size_t)i1*3];
        const float *p2 = &verts[(size_t)i2*3];
        double e1[3] = { (double)p1[0]-p0[0], (double)p1[1]-p0[1], (double)p1[2]-p0[2] };
        double e2[3] = { (double)p2[0]-p0[0], (double)p2[1]-p0[1], (double)p2[2]-p0[2] };
        double n0 = e1[1]*e2[2] - e1[2]*e2[1];
        double n1 = e1[2]*e2[0] - e1[0]*e2[2];
        double n2 = e1[0]*e2[1] - e1[1]*e2[0];
        normals[(size_t)i0*3+0] += (float)n0; normals[(size_t)i0*3+1] += (float)n1; normals[(size_t)i0*3+2] += (float)n2;
        normals[(size_t)i1*3+0] += (float)n0; normals[(size_t)i1*3+1] += (float)n1; normals[(size_t)i1*3+2] += (float)n2;
        normals[(size_t)i2*3+0] += (float)n0; normals[(size_t)i2*3+1] += (float)n1; normals[(size_t)i2*3+2] += (float)n2;
    }
    for (size_t v = 0; v < nv; v++) {
        double z = normals[v*3+0], y = normals[v*3+1], x = normals[v*3+2];
        double len = sqrt(z*z + y*y + x*x);
        if (len > 1e-20) {
            normals[v*3+0] = (float)(z/len);
            normals[v*3+1] = (float)(y/len);
            normals[v*3+2] = (float)(x/len);
        }
    }
}

typedef struct {
    int32_t key_u, key_v;
    int32_t va, vb, opp;
} HalfEdge;

static int cmp_he(const void *pa, const void *pb)
{
    const HalfEdge *a = (const HalfEdge *)pa;
    const HalfEdge *b = (const HalfEdge *)pb;
    if (a->key_u != b->key_u) return (a->key_u < b->key_u) ? -1 : 1;
    if (a->key_v != b->key_v) return (a->key_v < b->key_v) ? -1 : 1;
    return 0;
}

static int cmp_double_asc(const void *pa, const void *pb)
{
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a < b) ? -1 : (a > b ? 1 : 0);
}

static double edge_len2(const float *verts, int32_t a, int32_t b)
{
    double dz = (double)verts[(size_t)a*3+0] - (double)verts[(size_t)b*3+0];
    double dy = (double)verts[(size_t)a*3+1] - (double)verts[(size_t)b*3+1];
    double dx = (double)verts[(size_t)a*3+2] - (double)verts[(size_t)b*3+2];
    return dz*dz + dy*dy + dx*dx;
}

typedef struct { int64_t key; int cnt; } ECell;

static size_t ec_pow2(size_t n)
{
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int64_t ec_key(int32_t a, int32_t b)
{
    int32_t lo = (a < b) ? a : b, hi = (a < b) ? b : a;
    return ((int64_t)lo << 32) | (int64_t)(uint32_t)hi;
}

static size_t ec_slot(const ECell *t, size_t mask, int64_t key, int *found)
{
    size_t i = (size_t)((uint64_t)key * 0x9E3779B97F4A7C15ULL) & mask;
    for (;;) {
        if (t[i].key == key) { *found = 1; return i; }
        if (t[i].key == -1)  { *found = 0; return i; }
        i = (i + 1) & mask;
    }
}

static int ec_count(const ECell *t, size_t mask, int64_t key)
{
    int f = 0; size_t i = ec_slot(t, mask, key, &f);
    return f ? t[i].cnt : 0;
}

static void ec_inc(ECell *t, size_t mask, int64_t key)
{
    int f = 0; size_t i = ec_slot(t, mask, key, &f);
    if (!f) { t[i].key = key; t[i].cnt = 0; }
    t[i].cnt++;
}

static void dump_seam_front(const char *prefix,
                            const float *verts, size_t nv,
                            const BpaInitEdge *init, size_t n_init,
                            const BpaInitEdge *excl, size_t n_excl)
{
    char path[1024];

    snprintf(path, sizeof(path), "%s.front_tris.obj", prefix);
    FILE *tf = fopen(path, "w");
    if (tf) {
        for (size_t v = 0; v < nv; v++)
            fprintf(tf, "v %.6f %.6f %.6f\n",
                    (double)verts[v*3+0], (double)verts[v*3+1],
                    (double)verts[v*3+2]);
        for (size_t i = 0; i < n_init; i++)
            fprintf(tf, "f %d %d %d\n",
                    init[i].va + 1, init[i].vb + 1, init[i].v_opp + 1);
        fclose(tf);
        fprintf(stderr, "  [SEAM_DUMP_FRONT] wrote %s (%zu front triangles)\n",
                path, n_init);
    }

    snprintf(path, sizeof(path), "%s.front_edges.obj", prefix);
    FILE *ef = fopen(path, "w");
    if (ef) {
        fprintf(ef, "# seam BPA init front: green = selected (bridged), "
                    "red = excluded (run-1 boundary edge not in a seam plane)\n");
        size_t vidx = 1;
        for (size_t i = 0; i < n_init; i++) {
            const float *a = &verts[(size_t)init[i].va * 3];
            const float *b = &verts[(size_t)init[i].vb * 3];
            fprintf(ef, "v %.6f %.6f %.6f 0.0 1.0 0.0\n",
                    (double)a[0], (double)a[1], (double)a[2]);
            fprintf(ef, "v %.6f %.6f %.6f 0.0 1.0 0.0\n",
                    (double)b[0], (double)b[1], (double)b[2]);
            fprintf(ef, "l %zu %zu\n", vidx, vidx + 1); vidx += 2;
        }
        for (size_t i = 0; i < n_excl; i++) {
            const float *a = &verts[(size_t)excl[i].va * 3];
            const float *b = &verts[(size_t)excl[i].vb * 3];
            fprintf(ef, "v %.6f %.6f %.6f 1.0 0.0 0.0\n",
                    (double)a[0], (double)a[1], (double)a[2]);
            fprintf(ef, "v %.6f %.6f %.6f 1.0 0.0 0.0\n",
                    (double)b[0], (double)b[1], (double)b[2]);
            fprintf(ef, "l %zu %zu\n", vidx, vidx + 1); vidx += 2;
        }
        fclose(ef);
        fprintf(stderr, "  [SEAM_DUMP_FRONT] wrote %s "
                "(%zu selected green, %zu excluded red)\n",
                path, n_init, n_excl);
    }
}

static double tri_min_alt3(const float *p0, const float *p1, const float *p2)
{
    double ax = p1[0]-p0[0], ay = p1[1]-p0[1], az = p1[2]-p0[2];
    double bx = p2[0]-p0[0], by = p2[1]-p0[1], bz = p2[2]-p0[2];
    double cz = ax*by - ay*bx, cy = az*bx - ax*bz, cx = ay*bz - az*by;
    double area2 = sqrt(cx*cx + cy*cy + cz*cz);
    double l0 = sqrt(ax*ax+ay*ay+az*az);
    double l1 = sqrt(bx*bx+by*by+bz*bz);
    double dx = p2[0]-p1[0], dy = p2[1]-p1[1], dz = p2[2]-p1[2];
    double l2 = sqrt(dx*dx+dy*dy+dz*dz);
    double lmax = l0 > l1 ? (l0 > l2 ? l0 : l2) : (l1 > l2 ? l1 : l2);
    return lmax < 1e-12 ? 0.0 : area2 / lmax;
}

typedef struct { int32_t u, v; int32_t face; int32_t le; } FaceEdge;
static int cmp_faceedge(const void *pa, const void *pb)
{
    const FaceEdge *a = (const FaceEdge *)pa, *b = (const FaceEdge *)pb;
    if (a->u != b->u) return a->u < b->u ? -1 : 1;
    if (a->v != b->v) return a->v < b->v ? -1 : 1;
    return 0;
}

static size_t mark_sliver_and_tip_tris(const float *verts,
        const int32_t *faces, size_t nf,
        const SeamPlane *planes, size_t np, double band,
        double min_alt, uint8_t *del)
{
    size_t hn = nf * 3;
    FaceEdge *fe = (FaceEdge *)malloc((hn ? hn : 1) * sizeof(FaceEdge));
    for (size_t f = 0; f < nf; f++) {
        int32_t t[3] = { faces[f*3+0], faces[f*3+1], faces[f*3+2] };
        for (int e = 0; e < 3; e++) {
            int32_t a = t[e], b = t[(e+1)%3];
            FaceEdge *h = &fe[f*3+(size_t)e];
            h->u = (a < b) ? a : b; h->v = (a < b) ? b : a;
            h->face = (int32_t)f; h->le = (int32_t)e;
        }
    }
    qsort(fe, hn, sizeof(FaceEdge), cmp_faceedge);
    uint8_t *bmask = (uint8_t *)calloc(nf ? nf : 1, 1);
    for (size_t i = 0; i < hn; ) {
        size_t j = i + 1;
        while (j < hn && fe[j].u == fe[i].u && fe[j].v == fe[i].v) j++;
        if (j - i == 1) bmask[fe[i].face] |= (uint8_t)(1 << fe[i].le);
        i = j;
    }
    free(fe);

    size_t ndel = 0;
    for (size_t f = 0; f < nf; f++) {
        if (del[f] || bmask[f] == 0) continue;
        int32_t t[3] = { faces[f*3+0], faces[f*3+1], faces[f*3+2] };
        const float *p0 = &verts[(size_t)t[0]*3];
        const float *p1 = &verts[(size_t)t[1]*3];
        const float *p2 = &verts[(size_t)t[2]*3];
        int has_seam_bnd = 0;
        for (int e = 0; e < 3; e++)
            if ((bmask[f] & (1<<e)) &&
                SeamPlanes_edge_in(verts, t[e], t[(e+1)%3], planes, np, band)) { has_seam_bnd = 1; break; }
        if (has_seam_bnd && tri_min_alt3(p0, p1, p2) < min_alt) { del[f] = 1; ndel++; continue; }
        int nb = (bmask[f]&1) + ((bmask[f]>>1)&1) + ((bmask[f]>>2)&1);
        if (nb >= 2) {
            double d0 = SeamPlanes_vert_dist(verts, t[0], planes, np);
            double d1 = SeamPlanes_vert_dist(verts, t[1], planes, np);
            double d2 = SeamPlanes_vert_dist(verts, t[2], planes, np);
            int tip;
            if (nb == 3)                                  tip = (d0<=d1&&d0<=d2)?0:(d1<=d2?1:2);
            else if ((bmask[f]&1) && (bmask[f]&4))        tip = 0;
            else if ((bmask[f]&1) && (bmask[f]&2))        tip = 1;
            else                                          tip = 2;
            double dt = tip==0?d0:(tip==1?d1:d2);
            double dmin_other = 1e30;
            for (int k=0;k<3;k++) if (k!=tip){ double dk=k==0?d0:(k==1?d1:d2); if (dk<dmin_other) dmin_other=dk; }
            if (dt < band && dt < dmin_other) { del[f] = 1; ndel++; continue; }
        }
    }
    free(bmask);
    return ndel;
}

int SeamWeld_bridge(Arena_T arena,
                    const float *verts, size_t nv,
                    const int32_t *faces, size_t nf,
                    float cube_size, float rho, float rho_max_in, float band,
                    const uint8_t *want_mask,
                    const BpaBridgeGate *gate,
                    int32_t **out_faces, size_t *out_nf,
                    size_t *out_n_bridge)
{
    *out_faces = NULL;
    *out_nf = 0;
    if (out_n_bridge) *out_n_bridge = 0;
    if (nf == 0) return 0;

    float *normals = (float *)malloc(nv * 3 * sizeof(float));
    compute_vertex_normals(verts, nv, faces, nf, normals);

    uint8_t *used_any = (uint8_t *)calloc(nv, 1);
    for (size_t f = 0; f < nf; f++) {
        used_any[faces[f*3+0]] = 1;
        used_any[faces[f*3+1]] = 1;
        used_any[faces[f*3+2]] = 1;
    }

    SeamPlane planes[64];
    size_t np = SeamPlanes_detect(verts, nv, used_any, (double)cube_size,
                                  (double)band, planes, 64);
    if (np == 0) {
        int32_t *out = (int32_t *)ARENA_ALLOC(arena, (size_t)(nf*3*sizeof(int32_t)));
        memcpy(out, faces, nf*3*sizeof(int32_t));
        *out_faces = out; *out_nf = nf;
        free(normals); free(used_any);
        return 0;
    }

    double rho_max = (rho_max_in > 0.0f) ? (double)rho_max_in : (double)BRIDGE_RHO_MAX;
    { const char *e = getenv("SEAM_RHO_MAX"); if (e) rho_max = atof(e); }

    int32_t *sliver_owned = NULL;
    {
        double sliver_alt = (double)SEAM_SLIVER_MIN_ALT;
        const char *se = getenv("SEAM_SLIVER_MIN_ALT"); if (se) sliver_alt = atof(se);
        if (sliver_alt > 0.0) {
            size_t nf0 = nf, total_del = 0;
            for (int pass = 0; pass < 8; pass++) {
                uint8_t *del = (uint8_t *)calloc(nf ? nf : 1, 1);
                size_t ndel = mark_sliver_and_tip_tris(verts, faces, nf, planes, np,
                                                       (double)band, sliver_alt, del);
                if (ndel == 0) { free(del); break; }
                size_t nfc = nf - ndel;
                int32_t *fc = (int32_t *)malloc((nfc ? nfc : 1) * 3 * sizeof(int32_t));
                size_t k = 0;
                for (size_t f = 0; f < nf; f++) {
                    if (del[f]) continue;
                    fc[k*3+0] = faces[f*3+0]; fc[k*3+1] = faces[f*3+1]; fc[k*3+2] = faces[f*3+2];
                    k++;
                }
                free(del);
                free(sliver_owned);
                faces = fc; nf = nfc; sliver_owned = fc;
                total_del += ndel;
            }
            if (total_del > 0)
                compute_vertex_normals(verts, nv, faces, nf, normals);
            fprintf(stderr, "  [seam] sliver+tip cull: dropped %zu of %zu faces "
                    "(min_alt < %.2f vox, iterated)\n", total_del, nf0, sliver_alt);
        }
    }

    size_t hn = nf * 3;
    HalfEdge *he = (HalfEdge *)malloc((hn ? hn : 1) * sizeof(HalfEdge));
    for (size_t f = 0; f < nf; f++) {
        int32_t t[3] = { faces[f*3+0], faces[f*3+1], faces[f*3+2] };
        for (int e = 0; e < 3; e++) {
            int32_t a = t[e], b = t[(e+1)%3], o = t[(e+2)%3];
            HalfEdge *h = &he[f*3+(size_t)e];
            h->va = a; h->vb = b; h->opp = o;
            h->key_u = (a < b) ? a : b;
            h->key_v = (a < b) ? b : a;
        }
    }
    qsort(he, hn, sizeof(HalfEdge), cmp_he);

    BpaInitEdge *init = (BpaInitEdge *)malloc((hn ? hn : 1) * sizeof(BpaInitEdge));
    size_t n_init = 0;
    BpaInitEdge *excl = (BpaInitEdge *)malloc((hn ? hn : 1) * sizeof(BpaInitEdge));
    size_t n_excl = 0;
    for (size_t i = 0; i < hn; ) {
        size_t j = i + 1;
        while (j < hn && he[j].key_u == he[i].key_u && he[j].key_v == he[i].key_v) j++;
        if (j - i == 1) {
            int32_t va = he[i].va, vb = he[i].vb, opp = he[i].opp;
            if (SeamPlanes_edge_in(verts, va, vb, planes, np, (double)band)) {
                init[n_init].va = va; init[n_init].vb = vb; init[n_init].v_opp = opp;
                n_init++;
            } else {
                excl[n_excl].va = va; excl[n_excl].vb = vb; excl[n_excl].v_opp = opp;
                n_excl++;
            }
        }
        i = j;
    }

    uint8_t *promoted_vert = NULL;
    const BpaBridgeGate *grz_gate = NULL;
    double grz_pitch = 0.0, grz_tol = 0.0;
    {
        if (gate && gate->pitch > 0.0 &&
            ((gate->axis_z != NULL && gate->axis_y != NULL &&
              gate->axis_x != NULL && gate->axis_n >= 2) ||
             gate->umb_y != 0.0 || gate->umb_x != 0.0)) {
            grz_gate = gate; grz_pitch = gate->pitch;
            grz_tol = (gate->tol > 0.0) ? gate->tol : 0.25;
        }
    }
    if (n_excl > 0 && grz_tol > 0.0 &&
        getenv("SEAM_GRAZING_PROMOTE") != NULL &&
        getenv("SEAM_NO_GRAZING_PROMOTE") == NULL) {
        double reach = 2.0 * rho_max;
        size_t n_all = n_init + n_excl;
        float *mid = (float *)malloc(n_all * 3 * sizeof(float));
        for (size_t i = 0; i < n_all; i++) {
            const BpaInitEdge *e = (i < n_init) ? &init[i] : &excl[i - n_init];
            mid[i*3+0] = 0.5f*(verts[(size_t)e->va*3+0] + verts[(size_t)e->vb*3+0]);
            mid[i*3+1] = 0.5f*(verts[(size_t)e->va*3+1] + verts[(size_t)e->vb*3+1]);
            mid[i*3+2] = 0.5f*(verts[(size_t)e->va*3+2] + verts[(size_t)e->vb*3+2]);
        }
        size_t hcap = ec_pow2(n_all * 2 + 16);
        int64_t *hkey = (int64_t *)malloc(hcap * sizeof(int64_t));
        int32_t *hidx = (int32_t *)malloc(hcap * sizeof(int32_t));
        int32_t *hchain = (int32_t *)malloc(n_all * sizeof(int32_t));
        for (size_t i = 0; i < hcap; i++) { hkey[i] = -1; hidx[i] = -1; }
        size_t hmask = hcap - 1;
        double inv_cell = 1.0 / reach;
        #define GRZ_CELL_KEY(cz, cy, cx) \
            ( (((int64_t)(cz) & 0x1FFFFF) << 42) | (((int64_t)(cy) & 0x1FFFFF) << 21) \
              | ((int64_t)(cx) & 0x1FFFFF) )
        for (size_t i = 0; i < n_all; i++) {
            int64_t cz = (int64_t)floor((double)mid[i*3+0] * inv_cell);
            int64_t cy = (int64_t)floor((double)mid[i*3+1] * inv_cell);
            int64_t cx = (int64_t)floor((double)mid[i*3+2] * inv_cell);
            int64_t key = GRZ_CELL_KEY(cz, cy, cx);
            size_t s = (size_t)key & hmask;
            while (hkey[s] != -1 && hkey[s] != key) s = (s + 1) & hmask;
            hkey[s] = key;
            hchain[i] = hidx[s];
            hidx[s] = (int32_t)i;
        }
        size_t n_promoted = 0;
        for (size_t p = 0; p < np; p++) {
            int ax = planes[p].axis;
            double co = planes[p].coord;
            for (size_t x = 0; x < n_excl; x++) {
                if (excl[x].va < 0) continue;
                size_t i = n_init + x;
                double mc = (double)mid[i*3+(size_t)ax];
                if (fabs(mc - co) >= (double)band) continue;
                int side = (mc > co);
                int found = 0;
                int64_t bz = (int64_t)floor((double)mid[i*3+0] * inv_cell);
                int64_t by = (int64_t)floor((double)mid[i*3+1] * inv_cell);
                int64_t bx = (int64_t)floor((double)mid[i*3+2] * inv_cell);
                for (int64_t dz = -1; dz <= 1 && !found; dz++)
                for (int64_t dy = -1; dy <= 1 && !found; dy++)
                for (int64_t dx = -1; dx <= 1 && !found; dx++) {
                    int64_t key = GRZ_CELL_KEY(bz+dz, by+dy, bx+dx);
                    size_t s = (size_t)key & hmask;
                    while (hkey[s] != -1 && hkey[s] != key) s = (s + 1) & hmask;
                    if (hkey[s] != key) continue;
                    for (int32_t k = hidx[s]; k != -1; k = hchain[k]) {
                        if ((size_t)k == i) continue;
                        double kc = (double)mid[(size_t)k*3+(size_t)ax];
                        if (fabs(kc - co) >= (double)band) continue;
                        if ((kc > co) == side) continue;
                        double ddz = (double)mid[(size_t)k*3+0] - (double)mid[i*3+0];
                        double ddy = (double)mid[(size_t)k*3+1] - (double)mid[i*3+1];
                        double ddx = (double)mid[(size_t)k*3+2] - (double)mid[i*3+2];
                        if (ddz*ddz + ddy*ddy + ddx*ddx > reach*reach) continue;
                        {
                            double iuy, iux, kuy, kux;
                            BpaBridgeGate_axis_at(grz_gate,
                                (double)mid[i*3+0], &iuy, &iux);
                            BpaBridgeGate_axis_at(grz_gate,
                                (double)mid[(size_t)k*3+0], &kuy, &kux);
                            double iy = (double)mid[i*3+1] - iuy;
                            double ix = (double)mid[i*3+2] - iux;
                            double ky = (double)mid[(size_t)k*3+1] - kuy;
                            double kx = (double)mid[(size_t)k*3+2] - kux;
                            double dth = atan2(ky, kx) - atan2(iy, ix);
                            while (dth >  M_PI) dth -= 2.0*M_PI;
                            while (dth < -M_PI) dth += 2.0*M_PI;
                            double dw = PitchTable_dturns(
                                PitchTable_from_env(), hypot(iy, ix),
                                hypot(ky, kx), grz_pitch) - dth/(2.0*M_PI);
                            if (fabs(dw) > grz_tol) continue;
                        }
                        found = 1; break;
                    }
                }
                if (found) {
                    if (!promoted_vert)
                        promoted_vert = (uint8_t *)calloc(nv, 1);
                    promoted_vert[excl[x].va] = 1;
                    promoted_vert[excl[x].vb] = 1;
                    init[n_init + n_promoted] = excl[x];
                    n_promoted++;
                    excl[x].va = -1;
                }
            }
        }
        #undef GRZ_CELL_KEY
        if (n_promoted > 0) {
            n_init += n_promoted;
            size_t w = 0;
            for (size_t x = 0; x < n_excl; x++)
                if (excl[x].va >= 0) excl[w++] = excl[x];
            n_excl = w;
            fprintf(stderr, "  [seam] grazing promote: %zu excluded edge(s) "
                    "with an opposing boundary within %.1f vox joined the front\n",
                    n_promoted, reach);
        }
        free(hchain); free(hidx); free(hkey); free(mid);
    }

    if (want_mask) {
        size_t w = 0;
        for (size_t i = 0; i < n_init; i++) {
            int32_t a = init[i].va, b = init[i].vb, o = init[i].v_opp;
            if (want_mask[a] && want_mask[b] && want_mask[o]) init[w++] = init[i];
        }
        n_init = w;
    }

    {
        const char *fpfx = getenv("SEAM_DUMP_FRONT");
        if (fpfx) dump_seam_front(fpfx, verts, nv, init, n_init, excl, n_excl);
    }

    if (n_init == 0) {
        int32_t *out = (int32_t *)ARENA_ALLOC(arena, (size_t)((nf ? nf : 1)*3*sizeof(int32_t)));
        memcpy(out, faces, nf*3*sizeof(int32_t));
        *out_faces = out; *out_nf = nf;
        free(normals); free(used_any); free(he); free(init); free(excl);
        free(sliver_owned); free(promoted_vert);
        return 0;
    }

    int32_t *g2l = (int32_t *)malloc(nv * sizeof(int32_t));
    for (size_t v = 0; v < nv; v++) g2l[v] = -1;
    uint8_t *want = (uint8_t *)calloc(nv, 1);
    for (size_t i = 0; i < n_init; i++) {
        want[init[i].va] = 1; want[init[i].vb] = 1; want[init[i].v_opp] = 1;
    }
    size_t ln = 0;
    for (size_t v = 0; v < nv; v++) if (want[v]) ln++;

    int32_t *l2g = (int32_t *)malloc((ln ? ln : 1) * sizeof(int32_t));
    float *lv = (float *)malloc((ln ? ln : 1) * 3 * sizeof(float));
    float *lnrm = (float *)malloc((ln ? ln : 1) * 3 * sizeof(float));
    size_t li = 0;
    for (size_t v = 0; v < nv; v++) {
        if (!want[v]) continue;
        g2l[v] = (int32_t)li;
        l2g[li] = (int32_t)v;
        lv[li*3+0] = verts[v*3+0]; lv[li*3+1] = verts[v*3+1]; lv[li*3+2] = verts[v*3+2];
        lnrm[li*3+0] = normals[v*3+0]; lnrm[li*3+1] = normals[v*3+1]; lnrm[li*3+2] = normals[v*3+2];
        li++;
    }
    BpaInitEdge *init_local = (BpaInitEdge *)malloc(n_init * sizeof(BpaInitEdge));
    for (size_t i = 0; i < n_init; i++) {
        init_local[i].va = g2l[init[i].va];
        init_local[i].vb = g2l[init[i].vb];
        init_local[i].v_opp = g2l[init[i].v_opp];
    }

    double *elen = (double *)malloc(n_init * sizeof(double));
    for (size_t i = 0; i < n_init; i++)
        elen[i] = sqrt(edge_len2(verts, init[i].va, init[i].vb));
    qsort(elen, n_init, sizeof(double), cmp_double_asc);
    double s_med = elen[n_init/2];
    free(elen);

    double base = (double)BRIDGE_RHO_K * s_med;
    if (base < (double)BRIDGE_RHO_MIN) base = (double)BRIDGE_RHO_MIN;
    if (base > (double)BRIDGE_RHO_MAX) base = (double)BRIDGE_RHO_MAX;
    if ((double)rho > base) base = (double)rho;
    if (rho_max < base) rho_max = base;

    fprintf(stderr, "  [seam] planes=%zu init=%zu cloud=%zu verts; boundary-edge "
            "median=%.2f -> rho=%.2f..%.2f\n",
            np, n_init, ln, s_med, base, rho_max);

    if (s_med > 1.5 * rho_max) {
        fprintf(stderr,
            "  [seam] WARNING: boundary-edge median %.2f vox >> bridge reach "
            "(rho_max %.2f): front triangles cannot prime, expect a near-zero "
            "bridge. Refine the seam band before bridging (SEAM_NO_REFINE "
            "unset?) or ship finer rims.\n", s_med, rho_max);
    }

    int32_t *bridge_local = NULL;
    size_t n_bridge = 0;
    BallPivot_bridge(arena, lv, lnrm, ln, (float)base, (float)rho_max,
                     init_local, n_init, gate, &bridge_local, &n_bridge);
    fprintf(stderr, "  [seam] BPA produced %zu bridge faces\n", n_bridge);

    double maxedge2 = (2.0 * rho_max) * (2.0 * rho_max);
    size_t ecap = ec_pow2((nf + n_bridge) * 3 * 2 + 16);
    ECell *ec = (ECell *)malloc(ecap * sizeof(ECell));
    for (size_t i = 0; i < ecap; i++) { ec[i].key = -1; ec[i].cnt = 0; }
    size_t emask = ecap - 1;
    for (size_t f = 0; f < nf; f++) {
        ec_inc(ec, emask, ec_key(faces[f*3+0], faces[f*3+1]));
        ec_inc(ec, emask, ec_key(faces[f*3+1], faces[f*3+2]));
        ec_inc(ec, emask, ec_key(faces[f*3+2], faces[f*3+0]));
    }

    size_t total = nf + n_bridge;
    int32_t *out = (int32_t *)ARENA_ALLOC(arena, (size_t)((total ? total : 1) * 3 * sizeof(int32_t)));
    memcpy(out, faces, nf * 3 * sizeof(int32_t));
    size_t outn = nf, accepted = 0, rej_long = 0, rej_fold = 0;
    for (size_t f = 0; f < n_bridge; f++) {
        int32_t g0 = l2g[bridge_local[f*3+0]];
        int32_t g1 = l2g[bridge_local[f*3+1]];
        int32_t g2 = l2g[bridge_local[f*3+2]];
        if (g0 == g1 || g1 == g2 || g0 == g2) continue;
        if (SeamPlanes_face_straddle(verts, g0, g1, g2, planes, np, (double)band) < 0) {
            static int grz_bypass = -1;
            if (grz_bypass < 0)
                grz_bypass = getenv("SEAM_GRAZING_BYPASS") ? 1 : 0;
            int grazing = grz_bypass && promoted_vert &&
                          (promoted_vert[g0] || promoted_vert[g1] ||
                           promoted_vert[g2]);
            if (!grazing) {
                rej_fold++; continue;
            }
        }
        double em = edge_len2(verts, g0, g1);
        double e1 = edge_len2(verts, g1, g2); if (e1 > em) em = e1;
        double e2 = edge_len2(verts, g2, g0); if (e2 > em) em = e2;
        if (em > maxedge2) { rej_long++; continue; }
        int64_t k0 = ec_key(g0, g1), k1 = ec_key(g1, g2), k2 = ec_key(g2, g0);
        if (ec_count(ec, emask, k0) >= 2 ||
            ec_count(ec, emask, k1) >= 2 ||
            ec_count(ec, emask, k2) >= 2) continue;
        ec_inc(ec, emask, k0); ec_inc(ec, emask, k1); ec_inc(ec, emask, k2);
        out[outn*3+0] = g0; out[outn*3+1] = g1; out[outn*3+2] = g2;
        outn++; accepted++;
    }
    free(ec);

    *out_faces = out;
    *out_nf = outn;
    if (out_n_bridge) *out_n_bridge = accepted;
    fprintf(stderr, "  [seam] merged: %zu bridge faces kept (%zu over-long, "
            "%zu fold-back rejected), %zu -> %zu faces\n",
            accepted, rej_long, rej_fold, nf, outn);

    {
        const char *dp = getenv("SEAM_DUMP_BRIDGE");
        if (dp) {
            FILE *bf = fopen(dp, "w");
            if (bf) {
                for (size_t v = 0; v < nv; v++)
                    fprintf(bf, "v %.6f %.6f %.6f\n",
                            (double)verts[v*3+0], (double)verts[v*3+1],
                            (double)verts[v*3+2]);
                for (size_t f = nf; f < outn; f++)
                    fprintf(bf, "f %d %d %d\n",
                            out[f*3+0]+1, out[f*3+1]+1, out[f*3+2]+1);
                fclose(bf);
                fprintf(stderr, "  [SEAM_DUMP_BRIDGE] wrote %s (%zu bridge faces)\n",
                        dp, outn - nf);
            }
        }
    }

    free(normals); free(used_any); free(he); free(init); free(excl); free(promoted_vert);
    free(g2l); free(want); free(l2g); free(lv); free(lnrm); free(init_local);
    free(sliver_owned);
    return 0;
}
