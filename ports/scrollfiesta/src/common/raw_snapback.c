#include "raw_snapback.h"
#include "csr.h"
#include "kdtree.h"
#include "snap_cg.h"
#include "pipeline_constants.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static size_t extract_surface_voxels(Arena_T arena,
                                     const uint8_t *vol,
                                     size_t D, size_t H, size_t W,
                                     float **out_pts)
{
    assert(arena && vol && out_pts);

    size_t HW = H * W;

    size_t count = 0;
    for (size_t z = 0; z < D; z++) {
        for (size_t y = 0; y < H; y++) {
            for (size_t x = 0; x < W; x++) {
                size_t idx = z * HW + y * W + x;
                if (vol[idx] == 0) continue;
                int on_surface = 0;
                if (z == 0 || vol[idx - HW] == 0) on_surface = 1;
                else if (z + 1 >= D || vol[idx + HW] == 0) on_surface = 1;
                else if (y == 0 || vol[idx - W] == 0) on_surface = 1;
                else if (y + 1 >= H || vol[idx + W] == 0) on_surface = 1;
                else if (x == 0 || vol[idx - 1] == 0) on_surface = 1;
                else if (x + 1 >= W || vol[idx + 1] == 0) on_surface = 1;
                if (on_surface) count++;
            }
        }
    }

    if (count == 0) {
        *out_pts = NULL;
        return 0;
    }

    float *pts = (float *)ARENA_ALLOC(arena,
                                       (size_t)count * 3 * sizeof(float));
    size_t wi = 0;
    for (size_t z = 0; z < D; z++) {
        for (size_t y = 0; y < H; y++) {
            for (size_t x = 0; x < W; x++) {
                size_t idx = z * HW + y * W + x;
                if (vol[idx] == 0) continue;
                int on_surface = 0;
                if (z == 0 || vol[idx - HW] == 0) on_surface = 1;
                else if (z + 1 >= D || vol[idx + HW] == 0) on_surface = 1;
                else if (y == 0 || vol[idx - W] == 0) on_surface = 1;
                else if (y + 1 >= H || vol[idx + W] == 0) on_surface = 1;
                else if (x == 0 || vol[idx - 1] == 0) on_surface = 1;
                else if (x + 1 >= W || vol[idx + 1] == 0) on_surface = 1;
                if (on_surface) {
                    pts[wi * 3 + 0] = (float)z;
                    pts[wi * 3 + 1] = (float)y;
                    pts[wi * 3 + 2] = (float)x;
                    wi++;
                }
            }
        }
    }
    assert(wi == count);

    *out_pts = pts;
    return count;
}

int RawSnap_process(Arena_T arena,
                    ComponentMesh *meshes, size_t n_meshes,
                    const uint8_t *input_vol,
                    size_t D, size_t H, size_t W,
                    int iterations, float alpha, float dist_max)
{
    assert(arena);
    if (n_meshes == 0 || !input_vol) return 0;
    if (iterations <= 0) return 0;

    float dist_max_sq = dist_max * dist_max;

    float *surf_pts = NULL;
    size_t n_surf = extract_surface_voxels(arena, input_vol, D, H, W,
                                           &surf_pts);
    if (n_surf == 0) {
        fprintf(stderr, "    RawSnap: no surface voxels found, skipping\n");
        return 0;
    }
    fprintf(stderr, "    RawSnap: %zu surface voxels extracted\n", n_surf);

    KDTree_T surf_tree = KDTree_new(arena, surf_pts, n_surf);

    float alpha_ratio = alpha / (1.0f - alpha);

    for (size_t ci = 0; ci < n_meshes; ci++) {
        ComponentMesh *cm = &meshes[ci];
        assert(cm->self == cm);

        if (cm->nv == 0 || cm->nf == 0) continue;

        Arena_Mark comp_mark = Arena_save(arena);

        {
            uint8_t *vuse = (uint8_t *)ARENA_ALLOC(arena,
                (size_t)cm->nv * sizeof(uint8_t));
            memset(vuse, 0, cm->nv * sizeof(uint8_t));
            for (size_t f = 0; f < cm->nf; f++) {
                vuse[cm->faces[f * 3 + 0]] = 1;
                vuse[cm->faces[f * 3 + 1]] = 1;
                vuse[cm->faces[f * 3 + 2]] = 1;
            }
            size_t nv_used = 0;
            int32_t *remap = (int32_t *)ARENA_ALLOC(arena,
                (size_t)cm->nv * sizeof(int32_t));
            for (size_t i = 0; i < cm->nv; i++) {
                if (vuse[i]) {
                    if (nv_used != i) {
                        cm->verts[nv_used * 3 + 0] = cm->verts[i * 3 + 0];
                        cm->verts[nv_used * 3 + 1] = cm->verts[i * 3 + 1];
                        cm->verts[nv_used * 3 + 2] = cm->verts[i * 3 + 2];
                        if (cm->pin_mask) {
                            cm->pin_mask[nv_used] = cm->pin_mask[i];
                        }
                    }
                    remap[i] = (int32_t)nv_used;
                    nv_used++;
                } else {
                    remap[i] = -1;
                }
            }
            if (nv_used != cm->nv) {
                for (size_t f = 0; f < cm->nf; f++) {
                    cm->faces[f * 3 + 0] = remap[cm->faces[f * 3 + 0]];
                    cm->faces[f * 3 + 1] = remap[cm->faces[f * 3 + 1]];
                    cm->faces[f * 3 + 2] = remap[cm->faces[f * 3 + 2]];
                }
                fprintf(stderr, "    RawSnap: comp %zu: stripped %zu "
                        "dangling verts (%zu -> %zu)\n",
                        ci, cm->nv - nv_used, cm->nv, nv_used);
                cm->nv = nv_used;
            }
        }

        CSR_T adj = CSR_from_faces(arena, cm->faces, cm->nf, cm->nv);
        const int32_t *adj_off = CSR_offset(adj);

        float *targets = (float *)ARENA_ALLOC(arena,
                                               (size_t)cm->nv * 3 * sizeof(float));
        float *w = (float *)ARENA_ALLOC(arena,
                                         (size_t)cm->nv * sizeof(float));

        size_t n_matched = 0;
        double sum_dist_matched = 0.0;

        for (size_t i = 0; i < cm->nv; i++) {
            float dist_sq = 0.0f;
            size_t nn = KDTree_nearest(surf_tree, &cm->verts[i * 3],
                                       &dist_sq);
            targets[i * 3 + 0] = surf_pts[nn * 3 + 0];
            targets[i * 3 + 1] = surf_pts[nn * 3 + 1];
            targets[i * 3 + 2] = surf_pts[nn * 3 + 2];

            if (dist_sq <= dist_max_sq) {
                int32_t degree = adj_off[i + 1] - adj_off[i];
                w[i] = alpha_ratio * (float)degree;
                n_matched++;
                sum_dist_matched += sqrt((double)dist_sq);
            } else {
                w[i] = 0.0f;
            }
        }

        if (n_matched == 0) {
            fprintf(stderr, "    RawSnap: comp %zu: no matched verts, skip\n", ci);
            Arena_restore(arena, comp_mark);
            continue;
        }

        double mean_dist_m = sum_dist_matched / (double)n_matched;
        fprintf(stderr, "    RawSnap: comp %zu: %zu/%zu matched, "
                "mean_dist=%.4f, CG solve...\n",
                ci, n_matched, cm->nv, mean_dist_m);

        float *pinned_save = NULL;
        size_t n_pinned = 0;
        if (cm->pin_mask) {
            pinned_save = (float *)ARENA_ALLOC(arena,
                              (size_t)cm->nv * 3 * sizeof(float));
            for (size_t i = 0; i < cm->nv; i++) {
                if (cm->pin_mask[i]) {
                    pinned_save[i * 3 + 0] = cm->verts[i * 3 + 0];
                    pinned_save[i * 3 + 1] = cm->verts[i * 3 + 1];
                    pinned_save[i * 3 + 2] = cm->verts[i * 3 + 2];
                    n_pinned++;
                }
            }
        }

        SnapCG_solve(arena, adj, w, targets, cm->verts, cm->nv,
                     iterations, SNAP_CG_TOL);

        if (cm->pin_mask && n_pinned > 0) {
            for (size_t i = 0; i < cm->nv; i++) {
                if (cm->pin_mask[i]) {
                    cm->verts[i * 3 + 0] = pinned_save[i * 3 + 0];
                    cm->verts[i * 3 + 1] = pinned_save[i * 3 + 1];
                    cm->verts[i * 3 + 2] = pinned_save[i * 3 + 2];
                }
            }
            fprintf(stderr, "    RawSnap: comp %zu: restored %zu pinned verts\n",
                    ci, n_pinned);
        }

        Arena_restore(arena, comp_mark);
    }

    return 0;
}
