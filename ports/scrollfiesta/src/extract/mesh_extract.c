#include "../common/ves_platform.h"

#include "mesh_extract.h"
#include "pred_reject.h"
#include "../common/arena.h"
#include "../common/tiff_io.h"
#include "../common/halo_loader.h"
#include "../common/union_find.h"
#include "../common/pca.h"
#include "../common/mls_project.h"
#include "../common/vert_weld.h"
#include "../common/obj_io.h"
#include "../common/dump_obj.h"
#include "../common/pipeline_constants.h"
#include "../common/mesh_types.h"
#include "../common/obj_colors.h"
#include "../common/csr.h"
#include "../remesh/ball_pivot.h"
#include "../remesh/orient_mesh.h"
#include "../remesh/normal_orient.h"
#include "../remesh/patch_repair.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static const int NBR6[6][3] = {
    {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}
};

static int comp_info_cmp_desc(const void *a, const void *b)
{
    const CompInfo *ca = (const CompInfo *)a;
    const CompInfo *cb = (const CompInfo *)b;
    if (cb->count > ca->count) return 1;
    if (cb->count < ca->count) return -1;
    return 0;
}

int cc_label_3d(Arena_T arena, const uint8_t *vol,
                int D, int H, int W,
                int32_t *labels,
                CompInfo **out_comps, int32_t *out_n_comps)
{
    size_t vol_size = (size_t)D * (size_t)H * (size_t)W;
    int32_t HW = H * W;

    int32_t *queue = (int32_t *)ARENA_ALLOC(arena,
                                            (size_t)(vol_size * sizeof(int32_t)));
    int32_t max_comps = 4096;
    CompInfo *comps = (CompInfo *)ARENA_ALLOC(arena,
                                              (size_t)((size_t)max_comps * sizeof(CompInfo)));
    int32_t n_comps = 0;

    for (int32_t idx = 0; (size_t)idx < vol_size; idx++) {
        if (vol[idx] == 0 || labels[idx] != 0) continue;

        int32_t comp_label = n_comps + 1;
        int32_t qh = 0, qt = 0;
        queue[qt++] = idx;
        labels[idx] = comp_label;
        int32_t count = 0;

        while (qh < qt) {
            int32_t cur = queue[qh++];
            count++;

            int z = cur / HW;
            int rem = cur - z * HW;
            int y = rem / W;
            int x = rem - y * W;

            for (int n = 0; n < 6; n++) {
                int nz = z + NBR6[n][0];
                int ny = y + NBR6[n][1];
                int nx = x + NBR6[n][2];
                if (nz < 0 || nz >= D || ny < 0 || ny >= H || nx < 0 || nx >= W)
                    continue;
                int32_t ni = (int32_t)((size_t)nz * (size_t)HW +
                                       (size_t)ny * (size_t)W + (size_t)nx);
                if (vol[ni] == 0 || labels[ni] != 0) continue;
                labels[ni] = comp_label;
                queue[qt++] = ni;
            }
        }

        if (n_comps >= max_comps) {
            int32_t new_max = max_comps * 2;
            CompInfo *nc = (CompInfo *)ARENA_ALLOC(arena,
                                                    (size_t)((size_t)new_max * sizeof(CompInfo)));
            memcpy(nc, comps, (size_t)n_comps * sizeof(CompInfo));
            comps = nc;
            max_comps = new_max;
        }
        comps[n_comps].label = comp_label;
        comps[n_comps].count = count;
        n_comps++;
    }

    *out_comps = comps;
    *out_n_comps = n_comps;
    return 0;
}

static size_t propagate_normal_signs(Arena_T arena,
                                     const float *verts, size_t nv,
                                     const int32_t *faces, size_t nf,
                                     float *normals)
{
    if (nv == 0 || nf == 0) return 0;

    Arena_Mark mark = Arena_save(arena);

    CSR_T adj = CSR_from_faces(arena, faces, nf, nv);
    const int32_t *off = CSR_offset(adj);
    const int32_t *tgt = CSR_target(adj);

    uint8_t *visited = (uint8_t *)ARENA_CALLOC(arena, (size_t)nv, 1L);
    int32_t *queue = (int32_t *)ARENA_ALLOC(arena,
                        (size_t)nv * sizeof(int32_t));

    size_t n_flips = 0;

    for (;;) {
        int32_t seed = -1;
        float best_z = 0.0f, best_y = 0.0f, best_x = 0.0f;
        for (size_t v = 0; v < nv; v++) {
            if (visited[v]) continue;
            float nz = normals[v * 3 + 0];
            float ny = normals[v * 3 + 1];
            float nx = normals[v * 3 + 2];
            float nmag2 = nz*nz + ny*ny + nx*nx;
            if (nmag2 < 1e-18f) {
                visited[v] = 1;
                continue;
            }
            float z = verts[v * 3 + 0];
            float y = verts[v * 3 + 1];
            float x = verts[v * 3 + 2];
            if (seed < 0 ||
                z <  best_z ||
                (z == best_z && y <  best_y) ||
                (z == best_z && y == best_y && x < best_x)) {
                seed = (int32_t)v;
                best_z = z;
                best_y = y;
                best_x = x;
            }
        }
        if (seed < 0) break;

        visited[seed] = 1;
        int32_t q_head = 0;
        int32_t q_tail = 0;
        queue[q_tail++] = seed;

        while (q_head < q_tail) {
            int32_t u = queue[q_head++];
            float nuz = normals[u * 3 + 0];
            float nuy = normals[u * 3 + 1];
            float nux = normals[u * 3 + 2];

            for (int32_t j = off[u]; j < off[u + 1]; j++) {
                int32_t w = tgt[j];
                if (visited[w]) continue;

                float nwz = normals[w * 3 + 0];
                float nwy = normals[w * 3 + 1];
                float nwx = normals[w * 3 + 2];
                float wmag2 = nwz*nwz + nwy*nwy + nwx*nwx;
                if (wmag2 < 1e-18f) {
                    visited[w] = 1;
                    continue;
                }

                float d = nuz*nwz + nuy*nwy + nux*nwx;
                if (d < 0.0f) {
                    normals[w * 3 + 0] = -nwz;
                    normals[w * 3 + 1] = -nwy;
                    normals[w * 3 + 2] = -nwx;
                    n_flips++;
                }
                visited[w] = 1;
                queue[q_tail++] = w;
            }
        }
    }

    Arena_restore(arena, mark);
    return n_flips;
}

typedef struct {
    uint64_t key;
    int32_t  face_idx;
    float    cz, cy, cx;
} PairHashEntry;

static int cmp_pair_hash(const void *a, const void *b)
{
    const PairHashEntry *ea = (const PairHashEntry *)a;
    const PairHashEntry *eb = (const PairHashEntry *)b;
    if (ea->key < eb->key) return -1;
    if (ea->key > eb->key) return 1;
    if (ea->cz < eb->cz) return -1;
    if (ea->cz > eb->cz) return 1;
    if (ea->cy < eb->cy) return -1;
    if (ea->cy > eb->cy) return 1;
    if (ea->cx < eb->cx) return -1;
    if (ea->cx > eb->cx) return 1;
    if (ea->face_idx < eb->face_idx) return -1;
    if (ea->face_idx > eb->face_idx) return 1;
    return 0;
}

static inline uint64_t pair_cell_key(int32_t cz, int32_t cy, int32_t cx)
{
    uint64_t uz = (uint64_t)(cz + (1 << 20)) & 0x1FFFFF;
    uint64_t uy = (uint64_t)(cy + (1 << 20)) & 0x1FFFFF;
    uint64_t ux = (uint64_t)(cx + (1 << 20)) & 0x1FFFFF;
    return (uz << 42) | (uy << 21) | ux;
}

static size_t spatial_pairing_cull(Arena_T arena,
                                   const float *verts, size_t nv,
                                   int32_t *faces, size_t nf,
                                   const float cell_origin[3])
{
    if (nv == 0 || nf == 0) return nf;

    Arena_Mark mark = Arena_save(arena);

    float R = PAIRING_RADIUS_VOX;
    float R2 = R * R;
    float inv_R = 1.0f / R;

    float ocz = cell_origin ? cell_origin[0] : 0.0f;
    float ocy = cell_origin ? cell_origin[1] : 0.0f;
    float ocx = cell_origin ? cell_origin[2] : 0.0f;

    float *centroids = (float *)ARENA_ALLOC(arena,
                          (size_t)nf * 3L * sizeof(float));
    float *normals = (float *)ARENA_ALLOC(arena,
                        (size_t)nf * 3L * sizeof(float));
    PairHashEntry *entries = (PairHashEntry *)ARENA_ALLOC(arena,
                              (size_t)nf * sizeof(PairHashEntry));
    uint8_t *keep = (uint8_t *)ARENA_ALLOC(arena, (size_t)nf);

    size_t n_degenerate = 0;
    for (size_t f = 0; f < nf; f++) {
        int32_t a = faces[f*3+0];
        int32_t b = faces[f*3+1];
        int32_t c = faces[f*3+2];
        float az = verts[a*3+0], ay = verts[a*3+1], ax = verts[a*3+2];
        float bz = verts[b*3+0], by = verts[b*3+1], bx = verts[b*3+2];
        float cz = verts[c*3+0], cy = verts[c*3+1], cx = verts[c*3+2];

        float cenZ = (az + bz + cz) * (1.0f / 3.0f);
        float cenY = (ay + by + cy) * (1.0f / 3.0f);
        float cenX = (ax + bx + cx) * (1.0f / 3.0f);
        centroids[f*3+0] = cenZ;
        centroids[f*3+1] = cenY;
        centroids[f*3+2] = cenX;

        float abz_ = bz - az, aby_ = by - ay, abx_ = bx - ax;
        float acz_ = cz - az, acy_ = cy - ay, acx_ = cx - ax;
        float nz = aby_ * acx_ - abx_ * acy_;
        float ny = abx_ * acz_ - abz_ * acx_;
        float nx = abz_ * acy_ - aby_ * acz_;
        float nmag = sqrtf(nz*nz + ny*ny + nx*nx);
        if (nmag > 1e-9f) {
            float inv = 1.0f / nmag;
            normals[f*3+0] = nz * inv;
            normals[f*3+1] = ny * inv;
            normals[f*3+2] = nx * inv;
            keep[f] = 1;
        } else {
            normals[f*3+0] = 0.0f;
            normals[f*3+1] = 0.0f;
            normals[f*3+2] = 0.0f;
            keep[f] = 0;
            n_degenerate++;
        }

        entries[f].cz = cenZ;
        entries[f].cy = cenY;
        entries[f].cx = cenX;
        entries[f].face_idx = (int32_t)f;
        entries[f].key = pair_cell_key(
            (int32_t)floorf((cenZ + ocz) * inv_R),
            (int32_t)floorf((cenY + ocy) * inv_R),
            (int32_t)floorf((cenX + ocx) * inv_R));
    }

    qsort(entries, nf, sizeof(PairHashEntry), cmp_pair_hash);

    uint64_t *keys = (uint64_t *)ARENA_ALLOC(arena,
                        (size_t)nf * sizeof(uint64_t));
    for (size_t i = 0; i < nf; i++) keys[i] = entries[i].key;

    for (size_t f = 0; f < nf; f++) {
        if (!keep[f]) continue;

        float fz = centroids[f*3+0];
        float fy = centroids[f*3+1];
        float fx = centroids[f*3+2];
        float fnz = normals[f*3+0];
        float fny = normals[f*3+1];
        float fnx = normals[f*3+2];

        int32_t ciz = (int32_t)floorf((fz + ocz) * inv_R);
        int32_t ciy = (int32_t)floorf((fy + ocy) * inv_R);
        int32_t cix = (int32_t)floorf((fx + ocx) * inv_R);

        int dropped = 0;
        for (int dz = -1; dz <= 1 && !dropped; dz++) {
            for (int dy = -1; dy <= 1 && !dropped; dy++) {
                for (int dx = -1; dx <= 1 && !dropped; dx++) {
                    uint64_t target = pair_cell_key(ciz+dz, ciy+dy, cix+dx);
                    size_t lo = 0, hi = nf;
                    while (lo < hi) {
                        size_t mid = lo + (hi - lo) / 2;
                        if (keys[mid] < target) lo = mid + 1;
                        else hi = mid;
                    }
                    for (size_t k = lo; k < nf && keys[k] == target; k++) {
                        int32_t f2 = entries[k].face_idx;
                        if ((size_t)f2 == f) continue;
                        if (!keep[f2]) continue;

                        float t2z = entries[k].cz;
                        float t2y = entries[k].cy;
                        float t2x = entries[k].cx;
                        float dz2 = t2z - fz;
                        float dy2 = t2y - fy;
                        float dx2 = t2x - fx;
                        if (dz2*dz2 + dy2*dy2 + dx2*dx2 >= R2) continue;

                        float d = fnz * normals[f2*3+0]
                                + fny * normals[f2*3+1]
                                + fnx * normals[f2*3+2];
                        if (d > PAIRING_DOT_THRESHOLD) continue;

                        int twin_wins = 0;
                        if (t2z > fz)                            twin_wins = 1;
                        else if (t2z == fz && t2y > fy)          twin_wins = 1;
                        else if (t2z == fz && t2y == fy && t2x > fx) twin_wins = 1;
                        else if (t2z == fz && t2y == fy && t2x == fx
                                 && f2 > (int32_t)f)             twin_wins = 1;

                        if (twin_wins) {
                            keep[f] = 0;
                            dropped = 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    size_t kept = 0;
    for (size_t f = 0; f < nf; f++) {
        if (!keep[f]) continue;
        if (kept != f) {
            faces[kept*3+0] = faces[f*3+0];
            faces[kept*3+1] = faces[f*3+1];
            faces[kept*3+2] = faces[f*3+2];
        }
        kept++;
    }

    (void)n_degenerate;

    Arena_restore(arena, mark);
    return kept;
}

static void clean_unreferenced(Arena_T arena,
                               const float *old_verts, size_t old_nv,
                               int32_t *faces, size_t nf,
                               float **new_verts, size_t *new_nv)
{
    int32_t *remap = (int32_t *)ARENA_CALLOC(arena, (size_t)old_nv,
                                              sizeof(int32_t));
    for (size_t i = 0; i < nf * 3; i++) {
        assert(faces[i] >= 0 && (size_t)faces[i] < old_nv);
        remap[faces[i]] = 1;
    }

    int32_t count = 0;
    for (size_t i = 0; i < old_nv; i++) {
        if (remap[i]) {
            remap[i] = count++;
        } else {
            remap[i] = -1;
        }
    }

    float *verts = (float *)ARENA_ALLOC(arena,
                                         (size_t)((size_t)count * 3 * sizeof(float)));
    for (size_t i = 0; i < old_nv; i++) {
        if (remap[i] >= 0) {
            verts[remap[i] * 3 + 0] = old_verts[i * 3 + 0];
            verts[remap[i] * 3 + 1] = old_verts[i * 3 + 1];
            verts[remap[i] * 3 + 2] = old_verts[i * 3 + 2];
        }
    }

    for (size_t i = 0; i < nf * 3; i++) {
        faces[i] = remap[faces[i]];
    }

    *new_verts = verts;
    *new_nv = (size_t)count;
}

static size_t mesh_split_remove_small(Arena_T arena,
                                      size_t nv,
                                      int32_t *faces, size_t nf)
{
    if (nf == 0) return 0;

    UnionFind uf = UF_new(arena, (int32_t)nv);

    for (size_t i = 0; i < nf; i++) {
        int32_t a = faces[i * 3 + 0];
        int32_t b = faces[i * 3 + 1];
        int32_t c = faces[i * 3 + 2];
        uf_union(&uf, a, b);
        uf_union(&uf, b, c);
        uf_union(&uf, a, c);
    }

    int32_t *face_count = (int32_t *)ARENA_CALLOC(arena, (size_t)nv,
                                                    sizeof(int32_t));
    for (size_t i = 0; i < nf; i++) {
        int32_t root = uf_find(&uf, faces[i * 3]);
        face_count[root]++;
    }

    size_t kept = 0;
    for (size_t i = 0; i < nf; i++) {
        int32_t root = uf_find(&uf, faces[i * 3]);
        if (face_count[root] >= MIN_FRAGMENT_FACES) {
            faces[kept * 3 + 0] = faces[i * 3 + 0];
            faces[kept * 3 + 1] = faces[i * 3 + 1];
            faces[kept * 3 + 2] = faces[i * 3 + 2];
            kept++;
        }
    }
    return kept;
}

int MeshExtract_run(Arena_T          arena,
                    const char      *tiff_path,
                    const char      *pred_dir,
                    int              halo_voxels,
                    size_t           cube_D,
                    size_t           cube_H,
                    size_t           cube_W,
                    int              n_threads,
                    const char      *dump_cube_dir,
                    const char      *cube_id,
                    float            trim_inset,
                    const uint8_t   *vol_in,
                    int              p_size_in,
                    const int64_t   *cube_origin_in,
                    PredRectoField  *recto,
                    ComponentMesh  **out_meshes,
                    size_t          *out_n_meshes,
                    MeshResplitCloud **out_clouds)
{
    assert(arena);
    assert(out_meshes && out_n_meshes);
    (void)cube_H; (void)cube_W;
    (void)n_threads;

    *out_meshes = NULL;
    *out_n_meshes = 0;
    if (out_clouds) *out_clouds = NULL;

    if (halo_voxels < 0) {
        fprintf(stderr, "MeshExtract: halo_voxels=%d must be >= 0\n",
                halo_voxels);
        return -1;
    }

#ifdef VESUVIUS_DEBUG
    double t_start, t_phase;
    t_start = ves_clock_sec();
#endif

    uint8_t *vol = NULL;
    int D = 0, H = 0, W = 0;
    float cube_world_origin[3] = {0.0f, 0.0f, 0.0f};
    if (vol_in != NULL) {
        if (p_size_in <= 0 ||
            (size_t)p_size_in != cube_D + 2u * (size_t)halo_voxels) {
            fprintf(stderr, "MeshExtract: p_size_in=%d must equal "
                    "cube_D + 2*halo (%zu)\n",
                    p_size_in, cube_D + 2u * (size_t)halo_voxels);
            return -1;
        }
        {
            size_t pn = (size_t)p_size_in * (size_t)p_size_in
                      * (size_t)p_size_in;
            vol = (uint8_t *)ARENA_ALLOC(arena, (size_t)pn);
            memcpy(vol, vol_in, pn);
        }
        D = H = W = p_size_in;
        if (cube_origin_in) {
            cube_world_origin[0] = (float)cube_origin_in[0];
            cube_world_origin[1] = (float)cube_origin_in[1];
            cube_world_origin[2] = (float)cube_origin_in[2];
        }
    } else if (halo_voxels > 0) {
        if (!pred_dir || !cube_id) {
            fprintf(stderr, "MeshExtract: halo requires pred_dir and cube_id\n");
            return -1;
        }
        int p_size = 0;
        int64_t origin[3] = {0, 0, 0};
        if (HaloLoader_load(arena, pred_dir, cube_id,
                            (int)cube_D, halo_voxels,
                            &vol, &p_size, origin) != 0) {
            fprintf(stderr, "MeshExtract: HaloLoader_load failed for %s\n",
                    cube_id);
            return -1;
        }
        D = H = W = p_size;
        cube_world_origin[0] = (float)(origin[0] + halo_voxels);
        cube_world_origin[1] = (float)(origin[1] + halo_voxels);
        cube_world_origin[2] = (float)(origin[2] + halo_voxels);
    } else {
        assert(tiff_path);
        if (TiffIO_load(arena, tiff_path, &vol, &D, &H, &W) != 0) {
            fprintf(stderr, "MeshExtract: failed to load TIFF: %s\n",
                    tiff_path);
            return -1;
        }
    }

    size_t vol_size = (size_t)D * (size_t)H * (size_t)W;
    if (vol_size == 0) return 0;
    if (recto) {
        recto->vol = vol;
        recto->d = D; recto->h = H; recto->w = W;
        recto->halo = halo_voxels;
    }

    for (size_t i = 0; i < vol_size; i++) {
        vol[i] = (vol[i] > 0) ? 1 : 0;
    }

    uint8_t *ct_raw = NULL;
    {
        const char *arm = getenv("VES_CT_WEIGHTED_LOP");
        if (arm != NULL && arm[0] == '1' && halo_voxels > 0 &&
            pred_dir != NULL && cube_id != NULL) {
            char raw_dir[1024] = {0};
            const char *ovr = getenv("VES_RAW_DIR");
            if (ovr != NULL && ovr[0] != '\0') {
                snprintf(raw_dir, sizeof raw_dir, "%s", ovr);
            } else {
                size_t n = strlen(pred_dir);
                snprintf(raw_dir, sizeof raw_dir, "%s", pred_dir);
                while (n > 0 && (raw_dir[n - 1] == '/' || raw_dir[n - 1] == '\\'))
                    raw_dir[--n] = '\0';
                if (n >= 10 && strcmp(raw_dir + n - 10, "cubes_PRED") == 0)
                    memcpy(raw_dir + n - 10, "cubes_RAW", 10);
                else
                    raw_dir[0] = '\0';
            }
            if (raw_dir[0] != '\0') {
                int p2 = 0;
                int64_t org2[3] = {0, 0, 0};
                if (HaloLoader_load(arena, raw_dir, cube_id, (int)cube_D,
                                    halo_voxels, &ct_raw, &p2, org2) != 0 ||
                    (size_t)p2 != D) {
                    fprintf(stderr, "MeshExtract: CT-weighted LOP armed but the "
                            "raw halo volume is unavailable (%s, p=%d vs %zu); "
                            "unweighted\n", raw_dir, p2, D);
                    ct_raw = NULL;
                } else {
                    fprintf(stderr, "MeshExtract: CT-weighted LOP armed (%s)\n",
                            raw_dir);
                }
            } else {
                fprintf(stderr, "MeshExtract: CT-weighted LOP armed but pred_dir "
                        "is not a cubes_PRED dir; unweighted\n");
            }
        }
    }
    if (!getenv("VESUVIUS_NO_REJECT_GARBAGE")) {
        int off = (halo_voxels > 0) ? halo_voxels : 0;
        int od = (int)cube_D, oh = (int)cube_H, ow = (int)cube_W;
        if (off + od <= D && off + oh <= H && off + ow <= W) {
            Arena_Mark gmark = Arena_save(arena);
            size_t on = (size_t)od * (size_t)oh * (size_t)ow;
            uint8_t *owned = (uint8_t *)ARENA_ALLOC(arena, (size_t)on);
            size_t vol_HW = (size_t)H * (size_t)W;
            for (int oz = 0; oz < od; oz++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++)
                        owned[(size_t)oz * (size_t)oh * (size_t)ow +
                              (size_t)oy * (size_t)ow + (size_t)ox] =
                            vol[(size_t)(oz + off) * vol_HW +
                                (size_t)(oy + off) * (size_t)W + (size_t)(ox + off)];
            PredRejectStats prst;
            int garbage = PredReject_is_garbage(arena, owned, od, oh, ow, &prst);
            Arena_restore(arena, gmark);
            if (garbage) {
                fprintf(stderr, "  Extract: %s rejected as garbage solid-slab "
                        "(fill=%.2f interior=%.2f rect_run=%d) -> no meshes\n",
                        cube_id ? cube_id : "(cube)", prst.fill_frac,
                        prst.interior_frac, prst.max_rect_run);
                return 0;
            }
        }
    }

#ifdef VESUVIUS_DEBUG
    t_phase = ves_clock_sec();
    fprintf(stderr, "  Extract: loaded TIFF %dx%dx%d (%.3f s)\n", D, H, W,
            t_phase - t_start);
#endif

    int32_t *labels = (int32_t *)ARENA_CALLOC(arena, (size_t)vol_size,
                                               sizeof(int32_t));
    CompInfo *comps = NULL;
    int32_t n_raw_comps = 0;
    cc_label_3d(arena, vol, D, H, W, labels, &comps, &n_raw_comps);

    int32_t n_valid = 0;
    for (int32_t i = 0; i < n_raw_comps; i++) {
        if (comps[i].count >= MIN_CC_SIZE) {
            comps[n_valid++] = comps[i];
        }
    }
    qsort(comps, (size_t)n_valid, sizeof(CompInfo), comp_info_cmp_desc);
    if (n_valid > MAX_COMPONENTS) n_valid = MAX_COMPONENTS;

#ifdef VESUVIUS_DEBUG
    {
        double t_cc = ves_clock_sec();
        fprintf(stderr, "  Extract: %d raw CCs, %d after filter (min=%d) (%.3f s)\n",
                n_raw_comps, n_valid, MIN_CC_SIZE,
                t_cc - t_start);
    }
#endif

    if (n_valid == 0) return 0;

    ComponentMesh *meshes = (ComponentMesh *)ARENA_ALLOC(arena,
                                                          (size_t)((size_t)n_valid * sizeof(ComponentMesh)));
    memset(meshes, 0, (size_t)n_valid * sizeof(ComponentMesh));

    MeshResplitCloud *clouds = NULL;
    if (out_clouds) {
        clouds = (MeshResplitCloud *)ARENA_ALLOC(arena,
                     (size_t)((size_t)n_valid * sizeof(MeshResplitCloud)));
        memset(clouds, 0, (size_t)n_valid * sizeof(MeshResplitCloud));
    }

    Arena_T scratch = Arena_new();
    size_t out_count = 0;
    int HW = H * W;

    enum { XT_PREP, XT_CLOUD, XT_MLS, XT_WELDTRIM,
           XT_HOPPE, XT_BPA, XT_PATCH, XT_N };
    double xt[XT_N] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double xt_mark = 0.0;

    int owned_lo = halo_voxels;
    int owned_hi = halo_voxels + (int)cube_D;

    for (int32_t ci = 0; ci < n_valid; ci++) {
        Arena_free(scratch);
        xt_mark = ves_clock_sec();
        int32_t comp_label = comps[ci].label;
        float *snap_orig = NULL, *snap_lop = NULL; size_t snap_n = 0;

#ifdef VESUVIUS_DEBUG
        double t_c0, t_c1;
        t_c0 = ves_clock_sec();
#endif

        int zmin = D, zmax = -1, ymin = H, ymax = -1, xmin = W, xmax = -1;
        for (int z = 0; z < D; z++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    if (labels[(size_t)z * (size_t)HW +
                               (size_t)y * (size_t)W + (size_t)x] == comp_label) {
                        if (z < zmin) zmin = z;
                        if (z > zmax) zmax = z;
                        if (y < ymin) ymin = y;
                        if (y > ymax) ymax = y;
                        if (x < xmin) xmin = x;
                        if (x > xmax) xmax = x;
                    }
                }
            }
        }
        if (zmax < 0) continue;

        if (halo_voxels > 0) {
            int overlap_z = (zmax >= owned_lo) && (zmin < owned_hi);
            int overlap_y = (ymax >= owned_lo) && (ymin < owned_hi);
            int overlap_x = (xmax >= owned_lo) && (xmin < owned_hi);
            if (!overlap_z || !overlap_y || !overlap_x) {
#ifdef VESUVIUS_DEBUG
                fprintf(stderr,
                    "    comp %d: bbox [%d-%d,%d-%d,%d-%d] outside owned"
                    " [%d,%d) -> halo-only, drop\n",
                    comp_label, zmin, zmax, ymin, ymax, xmin, xmax,
                    owned_lo, owned_hi);
#endif
                continue;
            }
        }

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: bbox [%d-%d, %d-%d, %d-%d] (%.3f s)\n",
                comp_label, zmin, zmax, ymin, ymax, xmin, xmax,
                t_c1 - t_c0);
#endif

        int bD = zmax - zmin + 1;
        int bH = ymax - ymin + 1;
        int bW = xmax - xmin + 1;

        int pD = bD + 2;
        int pH = bH + 2;
        int pW = bW + 2;

        size_t pad_size = (size_t)pD * (size_t)pH * (size_t)pW;
        uint8_t *padded = (uint8_t *)ARENA_CALLOC(scratch, (size_t)pad_size, 1L);

        for (int z = zmin - 1; z <= zmax + 1; z++) {
            for (int y = ymin - 1; y <= ymax + 1; y++) {
                for (int x = xmin - 1; x <= xmax + 1; x++) {
                    int iz = z, iy = y, ix = x;
                    if (iz < 0) iz = 0; else if (iz >= D) iz = D - 1;
                    if (iy < 0) iy = 0; else if (iy >= H) iy = H - 1;
                    if (ix < 0) ix = 0; else if (ix >= W) ix = W - 1;
                    if (labels[(size_t)iz * (size_t)HW +
                               (size_t)iy * (size_t)W +
                               (size_t)ix] == comp_label) {
                        size_t pi = (size_t)(z - zmin + 1) * (size_t)pH * (size_t)pW +
                                    (size_t)(y - ymin + 1) * (size_t)pW +
                                    (size_t)(x - xmin + 1);
                        padded[pi] = 1;
                    }
                }
            }
        }

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: subvol %dx%dx%d (%.3f s)\n",
                comp_label, pD, pH, pW,
                t_c1 - t_c0);
#endif

        { double xn = ves_clock_sec(); xt[XT_PREP] += xn - xt_mark; xt_mark = xn; }

        float *surf_verts = NULL;
        int32_t *surf_faces = NULL;
        size_t surf_nv = 0, surf_nf = 0;
        float *surf_w = NULL;
        {
            size_t fg = 0;
            for (size_t pi = 0; pi < pad_size; pi++) if (padded[pi]) fg++;
            surf_nv = fg;
            surf_nf = 0;
            if (fg > 0) {
                surf_verts = (float *)ARENA_ALLOC(scratch,
                                (size_t)(fg * 3 * sizeof(float)));
                if (ct_raw != NULL)
                    surf_w = (float *)ARENA_ALLOC(scratch,
                                (size_t)(fg * sizeof(float)));
                size_t out_idx = 0;
                for (int pz = 0; pz < pD; pz++) {
                    for (int py = 0; py < pH; py++) {
                        for (int px = 0; px < pW; px++) {
                            size_t pi = (size_t)pz * (size_t)pH * (size_t)pW +
                                        (size_t)py * (size_t)pW + (size_t)px;
                            if (!padded[pi]) continue;
                            surf_verts[out_idx * 3 + 0] = (float)pz + 0.5f;
                            surf_verts[out_idx * 3 + 1] = (float)py + 0.5f;
                            surf_verts[out_idx * 3 + 2] = (float)px + 0.5f;
                            if (surf_w != NULL) {
                                long vz = (long)pz + (long)zmin - 1;
                                long vy = (long)py + (long)ymin - 1;
                                long vx = (long)px + (long)xmin - 1;
                                float wgt = EXTRACT_CT_LOP_FLOOR;
                                if (vz >= 0 && vy >= 0 && vx >= 0 &&
                                    (size_t)vz < D && (size_t)vy < H &&
                                    (size_t)vx < W) {
                                    float raw = (float)ct_raw[
                                        (size_t)vz * H * W + (size_t)vy * W +
                                        (size_t)vx];
                                    float t = (raw - EXTRACT_CT_LOP_LO) /
                                              (EXTRACT_CT_LOP_HI - EXTRACT_CT_LOP_LO);
                                    if (t < 0.0f) t = 0.0f;
                                    if (t > 1.0f) t = 1.0f;
                                    wgt = EXTRACT_CT_LOP_FLOOR +
                                          (1.0f - EXTRACT_CT_LOP_FLOOR) *
                                          powf(t, EXTRACT_CT_LOP_GAMMA);
                                }
                                surf_w[out_idx] = wgt;
                            }
                            out_idx++;
                        }
                    }
                }
            }
        }

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: %s -> %zu verts, %zu faces (%.3f s)\n",
                comp_label, "voxel-centers",
                surf_nv, surf_nf, t_c1 - t_c0);
#endif

        if (surf_nv < (size_t)MLS_MIN_NEIGHBOURS) continue;

        for (size_t v = 0; v < surf_nv; v++) {
            surf_verts[v * 3 + 0] += (float)(zmin - 1 - halo_voxels);
            surf_verts[v * 3 + 1] += (float)(ymin - 1 - halo_voxels);
            surf_verts[v * 3 + 2] += (float)(xmin - 1 - halo_voxels);
        }

        if (halo_voxels > 0 && surf_nv > 0) {
            float lo = -(float)halo_voxels;
            float hi_z = (float)cube_D + (float)halo_voxels;
            float hi_y = (float)cube_H + (float)halo_voxels;
            float hi_x = (float)cube_W + (float)halo_voxels;
            int32_t *remap = (int32_t *)ARENA_ALLOC(scratch,
                                (size_t)surf_nv * sizeof(int32_t));
            size_t nv_kept = 0, nv_dropped = 0;
            for (size_t v = 0; v < surf_nv; v++) {
                float zc = surf_verts[v * 3 + 0];
                float yc = surf_verts[v * 3 + 1];
                float xc = surf_verts[v * 3 + 2];
                if (zc <= lo || zc >= hi_z ||
                    yc <= lo || yc >= hi_y ||
                    xc <= lo || xc >= hi_x) {
                    remap[v] = -1;
                    nv_dropped++;
                } else {
                    remap[v] = (int32_t)nv_kept;
                    if (nv_kept != v) {
                        surf_verts[nv_kept * 3 + 0] = zc;
                        surf_verts[nv_kept * 3 + 1] = yc;
                        surf_verts[nv_kept * 3 + 2] = xc;
                        if (surf_w != NULL) surf_w[nv_kept] = surf_w[v];
                    }
                    nv_kept++;
                }
            }
            if (nv_dropped > 0) {
                size_t nf_kept = 0;
                for (size_t f = 0; f < surf_nf; f++) {
                    int32_t v0 = surf_faces[f * 3 + 0];
                    int32_t v1 = surf_faces[f * 3 + 1];
                    int32_t v2 = surf_faces[f * 3 + 2];
                    if (remap[v0] < 0 || remap[v1] < 0 || remap[v2] < 0) continue;
                    surf_faces[nf_kept * 3 + 0] = remap[v0];
                    surf_faces[nf_kept * 3 + 1] = remap[v1];
                    surf_faces[nf_kept * 3 + 2] = remap[v2];
                    nf_kept++;
                }
#ifdef VESUVIUS_DEBUG
                fprintf(stderr,
                        "    comp %d: working-edge filter dropped %zu/%zu v, "
                        "%zu/%zu f\n",
                        comp_label, nv_dropped, surf_nv,
                        surf_nf - nf_kept, surf_nf);
#endif
                surf_nv = nv_kept;
                surf_nf = nf_kept;
                if (surf_nv == 0) continue;
            }
        }

        { double xn = ves_clock_sec(); xt[XT_CLOUD] += xn - xt_mark; xt_mark = xn; }

        const float *diag_color =
            OBJ_COLORS[(comp_label - 1 + OBJ_NUM_COLORS) % OBJ_NUM_COLORS];

        float *mls_verts = (float *)ARENA_ALLOC(scratch,
                              (size_t)(surf_nv * 3 * sizeof(float)));
        float *mls_normals = (float *)ARENA_ALLOC(scratch,
                                (size_t)(surf_nv * 3 * sizeof(float)));
        float *mls_scratch_normals = (float *)ARENA_ALLOC(scratch,
                                        (size_t)(surf_nv * 3 * sizeof(float)));
        float *mls_scratch = (MLS_PROJECT_ITERS >= 2)
            ? (float *)ARENA_ALLOC(scratch,
                                   (size_t)(surf_nv * 3 * sizeof(float)))
            : NULL;
        const float *src = surf_verts;
        float *dst = mls_verts;
        for (int it = 0; it < MLS_PROJECT_ITERS; it++) {
            int is_last = (it == MLS_PROJECT_ITERS - 1);
            float *normals_dst = is_last ? mls_normals : mls_scratch_normals;
            MLS_project_verts(scratch, src, surf_nv,
                              MLS_PROJECT_RADIUS_VOX,
                              cube_world_origin,
                              dst, normals_dst);
            if (!MLS_cubecl_last_call_ok()) {
                fprintf(stderr, "MeshExtract: CubeCL MLS failed: %s\n",
                        MLS_cubecl_last_error());
                return -1;
            }
            src = dst;
            dst = (dst == mls_verts) ? mls_scratch : mls_verts;
        }
        if (surf_w != NULL && surf_nv > 0) {
            float *ct_dst = dst;
            float *ct_n = mls_scratch_normals != NULL ? mls_scratch_normals
                                                      : mls_normals;
            if (ct_dst != NULL && ct_dst != src) {
                MLS_set_point_weights(surf_w);
                MLS_project_verts(scratch, src, surf_nv, MLS_PROJECT_RADIUS_VOX,
                                  cube_world_origin, ct_dst, ct_n);
                MLS_set_point_weights(NULL);
                if (MLS_cubecl_last_call_ok()) {
                    size_t moved = 0;
                    double sum_abs = 0.0;
                    for (size_t v = 0; v < surf_nv; v++) {
                        const float *n = mls_normals + v * 3;
                        float dz = ct_dst[v * 3 + 0] - src[v * 3 + 0];
                        float dy = ct_dst[v * 3 + 1] - src[v * 3 + 1];
                        float dx = ct_dst[v * 3 + 2] - src[v * 3 + 2];
                        float t = dz * n[0] + dy * n[1] + dx * n[2];
                        if (t > 4.0f) t = 4.0f;
                        if (t < -4.0f) t = -4.0f;
                        ct_dst[v * 3 + 0] = src[v * 3 + 0] + t * n[0];
                        ct_dst[v * 3 + 1] = src[v * 3 + 1] + t * n[1];
                        ct_dst[v * 3 + 2] = src[v * 3 + 2] + t * n[2];
                        sum_abs += fabs((double)t);
                        if (fabs((double)t) > 0.25) moved++;
                    }
                    fprintf(stderr, "MeshExtract: CT-weighted normal step: "
                            "%zu of %zu points moved > 0.25 vox, mean |t| %.2f\n",
                            moved, surf_nv, sum_abs / (double)surf_nv);
                    src = ct_dst;
                }
            }
        }
        if (src != mls_verts) {
            memcpy(mls_verts, src, surf_nv * 3 * sizeof(float));
        }
        if (recto) {
            PredRectoStats stats = {0};
            PredRecto_cloud(recto, mls_verts, mls_normals, surf_nv, &stats);
            PredRecto_report("pre-BPA", &stats);
        }

        if (clouds) {
            snap_n = surf_nv;
            snap_orig = (float *)ARENA_ALLOC(arena, (size_t)(surf_nv * 3 * sizeof(float)));
            snap_lop  = (float *)ARENA_ALLOC(arena, (size_t)(surf_nv * 3 * sizeof(float)));
            memcpy(snap_orig, recto ? mls_verts : surf_verts, surf_nv * 3 * sizeof(float));
            memcpy(snap_lop,  mls_verts,  surf_nv * 3 * sizeof(float));
        }

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: MLS-midpoint (%d iter, R=%.1f vox) "
                "-> %zu verts (%.3f s)\n",
                comp_label, MLS_PROJECT_ITERS,
                (double)MLS_PROJECT_RADIUS_VOX, surf_nv, t_c1 - t_c0);
#endif

        { double xn = ves_clock_sec(); xt[XT_MLS] += xn - xt_mark; xt_mark = xn; }

        if (dump_cube_dir && cube_id && getenv("EXTRACT_DIAG")) {
            char sub[1024];
            snprintf(sub, sizeof(sub), "%s/%s_step0_post_lop",
                     dump_cube_dir, cube_id);
            DumpObj_ensure_dir(sub);
            char path[1024];
            snprintf(path, sizeof(path),
                     "%s/%s_post_lop_comp%03zu.obj",
                     sub, cube_id, out_count + 1);
            DumpObj_write_one_world(scratch, path, cube_id,
                                    mls_verts, surf_nv, surf_faces, surf_nf,
                                    diag_color);

            char psub[1024];
            snprintf(psub, sizeof(psub),
                     "%s/%s_step0_post_lop_points",
                     dump_cube_dir, cube_id);
            DumpObj_ensure_dir(psub);
            char ppath[1024];
            snprintf(ppath, sizeof(ppath),
                     "%s/%s_post_lop_points_comp%03zu.obj",
                     psub, cube_id, out_count + 1);
            DumpObj_write_points_world(scratch, ppath, cube_id,
                                       mls_verts, mls_normals, surf_nv);
        }

        {
            size_t welded_nv = 0;
            float *welded_verts = NULL, *welded_normals = NULL;
            int32_t dummy_face = 0;
            size_t dummy_out_nf = 0;
            float weld_eps = MLS_WELD_EPS_VOX;
            const float *src_v = mls_verts;
            const float *src_n = mls_normals;
            size_t src_nv = surf_nv;
            welded_verts = NULL; welded_normals = NULL; welded_nv = src_nv;
            for (int iter = 0; iter < 5; iter++) {
                size_t prev_nv = welded_nv;
                Weld_verts(scratch,
                           src_v, src_nv, src_n,
                           &dummy_face, 0, &dummy_out_nf,
                           weld_eps,  false,
                           &welded_verts, &welded_nv, &welded_normals);
                if (welded_nv == prev_nv && iter > 0) break;
                src_v = welded_verts;
                src_n = welded_normals;
                src_nv = welded_nv;
            }
#ifdef VESUVIUS_DEBUG
            fprintf(stderr,
                "    comp %d: pre-BPA weld %zu -> %zu verts (eps=%.2f, "
                "iterated)\n",
                comp_label, surf_nv, welded_nv, (double)weld_eps);
#endif
            mls_verts   = welded_verts;
            mls_normals = welded_normals;
            surf_nv       = welded_nv;

            if (dump_cube_dir && cube_id && getenv("EXTRACT_DIAG")) {
                char wsub[1024];
                snprintf(wsub, sizeof(wsub), "%s/%s_step0_post_weld",
                         dump_cube_dir, cube_id);
                DumpObj_ensure_dir(wsub);
                char wpath[1024];
                snprintf(wpath, sizeof(wpath),
                         "%s/%s_post_weld_comp%03zu.obj",
                         wsub, cube_id, out_count + 1);
                DumpObj_write_points_world(scratch, wpath, cube_id,
                                           mls_verts, mls_normals, surf_nv);
            }

            if (halo_voxels > 0 && surf_nv > 0) {
                float ins  = trim_inset > -999.0f ? trim_inset
                                                : (float)BPA_OWNED_TRIM_INSET;
                float lo   = ins;
                float hi_z = (float)cube_D - ins;
                float hi_y = (float)cube_H - ins;
                float hi_x = (float)cube_W - ins;
                size_t kept = 0;
                for (size_t v = 0; v < surf_nv; v++) {
                    float zc = mls_verts[v*3+0];
                    float yc = mls_verts[v*3+1];
                    float xc = mls_verts[v*3+2];
                    if (zc < lo || zc > hi_z ||
                        yc < lo || yc > hi_y ||
                        xc < lo || xc > hi_x) continue;
                    if (kept != v) {
                        mls_verts[kept*3+0] = zc;
                        mls_verts[kept*3+1] = yc;
                        mls_verts[kept*3+2] = xc;
                        mls_normals[kept*3+0] = mls_normals[v*3+0];
                        mls_normals[kept*3+1] = mls_normals[v*3+1];
                        mls_normals[kept*3+2] = mls_normals[v*3+2];
                    }
                    kept++;
                }
#ifdef VESUVIUS_DEBUG
                fprintf(stderr,
                    "    comp %d: pre-BPA owned trim (inset=%.2f) %zu -> %zu verts\n",
                    comp_label, (double)ins, surf_nv, kept);
#endif
                surf_nv = kept;
                if (dump_cube_dir && cube_id && getenv("EXTRACT_DIAG")) {
                    char tsub[1024];
                    snprintf(tsub, sizeof(tsub), "%s/%s_step0_post_lop_trimmed",
                             dump_cube_dir, cube_id);
                    DumpObj_ensure_dir(tsub);
                    char tpath[1024];
                    snprintf(tpath, sizeof(tpath),
                             "%s/%s_post_lop_trimmed_comp%03zu.obj",
                             tsub, cube_id, out_count + 1);
                    DumpObj_write_points_world(scratch, tpath, cube_id,
                                               mls_verts, mls_normals, surf_nv);
                }
                if (surf_nv < 3) continue;
            }

            { double xn = ves_clock_sec(); xt[XT_WELDTRIM] += xn - xt_mark; xt_mark = xn; }

            if (!getenv("HOPPE_NO_ORIENT")) {
                float hr = 1.2f;
                const char *he = getenv("HOPPE_RADIUS");
                if (he) { double t = atof(he); if (t > 0.0) hr = (float)t; }
                size_t hf = 0;
                NormalOrient_consistent(scratch, mls_verts, mls_normals,
                                        surf_nv, hr, &hf);
                fprintf(stderr,
                    "    comp %d: Hoppe orient flipped %zu/%zu normals (r=%.2f)\n",
                    comp_label, hf, surf_nv, (double)hr);
            }

            { double xn = ves_clock_sec(); xt[XT_HOPPE] += xn - xt_mark; xt_mark = xn; }

            int32_t *bpa_faces = NULL;
            size_t   bpa_nf = 0;
            BpaReconGate bpa_gate;
            const BpaReconGate *bpa_gate_p =
                BpaReconGate_from_env(&bpa_gate, cube_world_origin) ? &bpa_gate : NULL;
            int rc = BallPivot_reconstruct_gated(
                scratch, mls_verts, mls_normals, surf_nv, BPA_RHO_VOX,
                bpa_gate_p, &bpa_faces, &bpa_nf);
            if (rc != 0 || bpa_nf == 0) {
                fprintf(stderr,
                    "    comp %d: BPA failed (rc=%d, nf=%zu) — skip\n",
                    comp_label, rc, bpa_nf);
                continue;
            }
            surf_faces = bpa_faces;
            surf_nf    = bpa_nf;
            surf_verts = mls_verts;
#ifdef VESUVIUS_DEBUG
            t_c1 = ves_clock_sec();
            fprintf(stderr,
                "    comp %d: BPA rho=%.1f -> %zu faces (%zu verts, %.2f s)\n",
                comp_label, (double)BPA_RHO_VOX, surf_nf, surf_nv, t_c1 - t_c0);
#endif

            { double xn = ves_clock_sec(); xt[XT_BPA] += xn - xt_mark; xt_mark = xn; }

            if (!getenv("PATCH_REPAIR_OFF")) {
                int32_t *pr_faces = NULL;
                size_t pr_nf = 0;
                PatchRepairStats prs;
                if (PatchRepair_repair(scratch, mls_verts, surf_nv,
                                       surf_faces, surf_nf,
                                       &pr_faces, &pr_nf, &prs) == 0 && pr_nf > 0) {
                    fprintf(stderr,
                        "    comp %d: patch_repair %zu/%zu patches (tears=%zu, "
                        "-%zu +%zu faces; skip g%zu r%zu p%zu t%zu)\n",
                        comp_label, prs.n_repaired, prs.n_patches,
                        prs.n_tear_clusters, prs.faces_deleted, prs.faces_added,
                        prs.n_skip_gate, prs.n_skip_ring, prs.n_skip_project,
                        prs.n_skip_tri);
                    surf_faces = pr_faces;
                    surf_nf    = pr_nf;
                }
            }
        }

        { double xn = ves_clock_sec(); xt[XT_PATCH] += xn - xt_mark; xt_mark = xn; }

        size_t n_sign_flips = 0;
        if (getenv("SIGN_PROPAGATE") != NULL) {
            n_sign_flips = propagate_normal_signs(scratch,
                                                  mls_verts, surf_nv,
                                                  surf_faces, surf_nf,
                                                  mls_normals);
            fprintf(stderr,
                    "    comp %d: sign-propagate flipped %zu/%zu vert "
                    "normals\n",
                    comp_label, n_sign_flips, surf_nv);
        }
        (void)n_sign_flips;

        {
            size_t or_flips = 0, or_comps = 0, or_resid = 0;
            OrientMesh_consistent(scratch,
                                  mls_verts, surf_nv, mls_normals,
                                  surf_faces, surf_nf,
                                  &or_flips, &or_comps, &or_resid);
#ifdef VESUVIUS_DEBUG
            fprintf(stderr,
                "    comp %d: orient BFS %zu comps, %zu flips, "
                "%zu residual same-dir edges\n",
                comp_label, or_comps, or_flips, or_resid);
#endif
        }

        surf_verts = mls_verts;

        (void)Weld_verts;
        (void)spatial_pairing_cull;

        if (surf_nf == 0 || surf_nv == 0) continue;

        surf_nf = mesh_split_remove_small(scratch, surf_nv,
                                        surf_faces, surf_nf);
        if (surf_nf == 0) continue;

        float *clean_verts = NULL;
        size_t clean_nv = 0;
        clean_unreferenced(scratch, surf_verts, surf_nv, surf_faces, surf_nf,
                           &clean_verts, &clean_nv);
        surf_verts = clean_verts;
        surf_nv = clean_nv;

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: cleanup -> %zu verts, %zu faces (%.3f s)\n",
                comp_label, surf_nv, surf_nf,
                t_c1 - t_c0);
#endif

#ifdef VESUVIUS_DEBUG
        t_c1 = ves_clock_sec();
        fprintf(stderr, "    comp %d: post-cleanup -> %zu verts, %zu faces (%.3f s)\n",
                comp_label, surf_nv, surf_nf,
                t_c1 - t_c0);
#endif

        if (surf_nf == 0 || surf_nv == 0) continue;

        if (halo_voxels > EXPORT_HALO_VOX && surf_nv > 0) {
            float export_lo = -(float)EXPORT_HALO_VOX;
            float export_hi_z = (float)cube_D + (float)EXPORT_HALO_VOX;
            float export_hi_y = (float)cube_H + (float)EXPORT_HALO_VOX;
            float export_hi_x = (float)cube_W + (float)EXPORT_HALO_VOX;
            int32_t *remap_e = (int32_t *)ARENA_ALLOC(scratch,
                                (size_t)surf_nv * sizeof(int32_t));
            size_t nv_kept_e = 0;
            for (size_t v = 0; v < surf_nv; v++) {
                float zc = surf_verts[v * 3 + 0];
                float yc = surf_verts[v * 3 + 1];
                float xc = surf_verts[v * 3 + 2];
                if (zc < export_lo || zc > export_hi_z ||
                    yc < export_lo || yc > export_hi_y ||
                    xc < export_lo || xc > export_hi_x) {
                    remap_e[v] = -1;
                } else {
                    remap_e[v] = (int32_t)nv_kept_e;
                    if (nv_kept_e != v) {
                        surf_verts[nv_kept_e * 3 + 0] = zc;
                        surf_verts[nv_kept_e * 3 + 1] = yc;
                        surf_verts[nv_kept_e * 3 + 2] = xc;
                    }
                    nv_kept_e++;
                }
            }
            size_t nf_kept_e = 0;
            for (size_t f = 0; f < surf_nf; f++) {
                int32_t v0 = surf_faces[f * 3 + 0];
                int32_t v1 = surf_faces[f * 3 + 1];
                int32_t v2 = surf_faces[f * 3 + 2];
                if (remap_e[v0] < 0 || remap_e[v1] < 0 || remap_e[v2] < 0) continue;
                surf_faces[nf_kept_e * 3 + 0] = remap_e[v0];
                surf_faces[nf_kept_e * 3 + 1] = remap_e[v1];
                surf_faces[nf_kept_e * 3 + 2] = remap_e[v2];
                nf_kept_e++;
            }
#ifdef VESUVIUS_DEBUG
            fprintf(stderr,
                "    comp %d: export-halo trim (read=%d, export=%d) "
                "%zu->%zu v, %zu->%zu f\n",
                comp_label, halo_voxels, EXPORT_HALO_VOX,
                surf_nv, nv_kept_e, surf_nf, nf_kept_e);
#endif
            surf_nv = nv_kept_e;
            surf_nf = nf_kept_e;
            if (surf_nf == 0 || surf_nv == 0) continue;
        }

        if (dump_cube_dir && cube_id && getenv("EXTRACT_DIAG")) {
            char sub[1024];
            snprintf(sub, sizeof(sub), "%s/%s_step0_pre_simplify",
                     dump_cube_dir, cube_id);
            DumpObj_ensure_dir(sub);
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s_pre_simplify_comp%03zu.obj",
                     sub, cube_id, out_count + 1);
            DumpObj_write_one_world(scratch, path, cube_id,
                                    surf_verts, surf_nv, surf_faces, surf_nf,
                                    diag_color);
        }

        float *final_verts = (float *)ARENA_ALLOC(arena,
                                                    (size_t)(surf_nv * 3 * sizeof(float)));
        memcpy(final_verts, surf_verts, surf_nv * 3 * sizeof(float));

        int32_t *final_faces = (int32_t *)ARENA_ALLOC(arena,
                                                        (size_t)(surf_nf * 3 * sizeof(int32_t)));
        memcpy(final_faces, surf_faces, surf_nf * 3 * sizeof(int32_t));

        float *final_normals = NULL;

        float pca_normal[3] = {0, 0, 1};
        float centroid[3] = {0, 0, 0};
        PCA_normal(final_verts, surf_nv, pca_normal, centroid);

        ComponentMesh *cm = &meshes[out_count];
        cm->verts = final_verts;
        cm->faces = final_faces;
        cm->vert_normals = final_normals;
        cm->nv = surf_nv;
        cm->nf = surf_nf;
        cm->comp_id = (int)(out_count + 1);
        cm->pca_normal[0] = pca_normal[0];
        cm->pca_normal[1] = pca_normal[1];
        cm->pca_normal[2] = pca_normal[2];
        cm->centroid[0] = centroid[0];
        cm->centroid[1] = centroid[1];
        cm->centroid[2] = centroid[2];
        cm->self = cm;

        cm->pin_mask = NULL;

        if (clouds) {
            clouds[out_count].orig_pts = snap_orig;
            clouds[out_count].lop_pts  = snap_lop;
            clouds[out_count].cell_origin[0] = cube_world_origin[0];
            clouds[out_count].cell_origin[1] = cube_world_origin[1];
            clouds[out_count].cell_origin[2] = cube_world_origin[2];
            clouds[out_count].n = snap_n;
        }

        assert(ComponentMesh_valid(cm));
        out_count++;

#ifdef VESUVIUS_DEBUG
        fprintf(stderr, "  Extract: comp %d -> %zu verts, %zu faces, "
                "normal=(%.3f,%.3f,%.3f)\n",
                cm->comp_id, cm->nv, cm->nf,
                (double)cm->pca_normal[0], (double)cm->pca_normal[1],
                (double)cm->pca_normal[2]);
#endif
    }

    Arena_dispose(&scratch);

#ifdef VESUVIUS_DEBUG
    {
        double t_end = ves_clock_sec();
        fprintf(stderr, "  Extract: %zu components, total %.3f s\n", out_count,
                t_end - t_start);
    }
#endif

    if (getenv("EXTRACT_TIMING")) {
        fprintf(stderr,
            "  [extract-timing] prep=%.2f cloud=%.2f mls=%.2f weld+trim=%.2f "
            "hoppe=%.2f bpa=%.2f patch=%.2f  (s, summed over %zu comps; "
            "remainder = orient/finalize)\n",
            xt[XT_PREP], xt[XT_CLOUD], xt[XT_MLS], xt[XT_WELDTRIM],
            xt[XT_HOPPE], xt[XT_BPA], xt[XT_PATCH], out_count);
    }

    *out_meshes = meshes;
    *out_n_meshes = out_count;
    if (out_clouds) *out_clouds = clouds;
    return 0;
}
