#include "ves_platform.h"
#include "dump_obj.h"
#include "obj_io.h"
#include "mesh_bin.h"
#include "obj_colors.h"
#include "cc_color.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

#define COMP_PATH_LEN 1024

static int comp_color_idx(const ComponentMesh *cm)
{
    return (cm->comp_id > 0 ? cm->comp_id - 1 : 0) % OBJ_NUM_COLORS;
}

static int parse_cube_origin_zyx(const char *cube_id, float origin[3])
{
    origin[0] = 0.0f;
    origin[1] = 0.0f;
    origin[2] = 0.0f;
    if (!cube_id) return -1;
    int iz = 0, iy = 0, ix = 0;
    if (sscanf(cube_id, "z%d_y%d_x%d", &iz, &iy, &ix) != 3) return -1;
    origin[0] = (float)iz;
    origin[1] = (float)iy;
    origin[2] = (float)ix;
    return 0;
}

static void offset_verts(const float *src, float *dst, size_t nv,
                         const float origin[3])
{
    for (size_t i = 0; i < nv; i++) {
        dst[i * 3 + 0] = src[i * 3 + 0] + origin[0];
        dst[i * 3 + 1] = src[i * 3 + 1] + origin[1];
        dst[i * 3 + 2] = src[i * 3 + 2] + origin[2];
    }
}

static int copy_file(const char *src, const char *dst)
{
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }
    char buf[8192];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        fwrite(buf, 1, n, out);
    fclose(in);
    fclose(out);
    return 0;
}

static void copy_to_all_obj_dir(const char *dir, const char *cube_id,
                                 const char *all_path)
{
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s", dir);
    char *slash_fwd = strrchr(parent, '/');
    char *slash_bck = strrchr(parent, '\\');
    char *slash = slash_fwd;
    if (slash_bck && (!slash || slash_bck > slash)) slash = slash_bck;
    if (!slash) return;
    *slash = '\0';

    char all_dir[1024];
    snprintf(all_dir, sizeof(all_dir), "%.512s/%.256s_all_obj", parent, cube_id);
    if (ves_mkdir(all_dir) != 0 && errno != EEXIST) return;

    const char *base = ves_path_basename(all_path);
    char dst[1024];
    snprintf(dst, sizeof(dst), "%.768s/%.200s", all_dir, base);
    copy_file(all_path, dst);
}

static void write_binary_companion(const char *dir, const char *cube_id,
                                   const char *obj_path,
                                   const float *verts, size_t nv,
                                   const int32_t *faces, size_t nf)
{
    char mesh_path[1024];
    if (MeshBin_companion_path(obj_path, mesh_path, sizeof(mesh_path)) != 0 ||
        MeshBin_write(mesh_path, verts, nv, faces, nf, NULL) != 0) {
        fprintf(stderr, "WARNING: cannot write VMESH companion for %s\n",
                obj_path);
        return;
    }
    copy_to_all_obj_dir(dir, cube_id, mesh_path);
}

void DumpObj_extract_cube_id(const char *input_path, char *out, size_t out_sz)
{
    const char *base = ves_path_basename(input_path);
    size_t len = strlen(base);
    const char *dot = strrchr(base, '.');
    if (dot && dot > base) len = (size_t)(dot - base);
    if (len >= out_sz) len = out_sz - 1;
    memcpy(out, base, len);
    out[len] = '\0';
}

int DumpObj_ensure_dir(const char *path)
{
    if (ves_mkdir(path) != 0 && errno != EEXIST) {
        fprintf(stderr, "WARNING: mkdir(%s): %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

void DumpObj_write_meshes(Arena_T arena, const char *dir,
                          const char *cube_id, const char *step_tag,
                          const ComponentMesh *meshes, size_t count)
{
    if (count == 0) return;

    Arena_Mark mark = Arena_save(arena);

    float origin[3];
    parse_cube_origin_zyx(cube_id, origin);

    static const float FILL_BLUE[3] = {0.10f, 0.45f, 1.00f};

    size_t tot_nv = 0, tot_nf = 0;
    for (size_t i = 0; i < count; i++) { tot_nv += meshes[i].nv; tot_nf += meshes[i].nf; }
    float   *comb_v = (float *)ARENA_ALLOC(arena, (size_t)(tot_nv * 3 * sizeof(float)));
    int32_t *comb_f = (int32_t *)ARENA_ALLOC(arena, (size_t)(tot_nf * 3 * sizeof(int32_t)));
    float   *comb_c = (float *)ARENA_ALLOC(arena, (size_t)(tot_nv * 3 * sizeof(float)));
    size_t vo = 0, fo = 0;
    char comp_path[COMP_PATH_LEN];

    for (size_t i = 0; i < count; i++) {
        snprintf(comp_path, sizeof(comp_path),
                 "%.512s/%.256s_%.128s_%03d.obj",
                 dir, cube_id, step_tag, (int)(i + 1));

        float *shifted = comb_v + vo * 3;
        offset_verts(meshes[i].verts, shifted, meshes[i].nv, origin);

        if (meshes[i].nv_pre_fill > 0) {
            ObjIO_write_twotone(comp_path, shifted, meshes[i].nv,
                                meshes[i].faces, meshes[i].nf,
                                meshes[i].nv_pre_fill,
                                OBJ_COLORS[comp_color_idx(&meshes[i])],
                                FILL_BLUE);
        } else {
            ObjIO_write_colored(comp_path, shifted, meshes[i].nv,
                                meshes[i].faces, meshes[i].nf,
                                OBJ_COLORS[comp_color_idx(&meshes[i])]);
        }
        for (size_t f = 0; f < meshes[i].nf; f++) {
            comb_f[(fo + f) * 3 + 0] = meshes[i].faces[f * 3 + 0] + (int32_t)vo;
            comb_f[(fo + f) * 3 + 1] = meshes[i].faces[f * 3 + 1] + (int32_t)vo;
            comb_f[(fo + f) * 3 + 2] = meshes[i].faces[f * 3 + 2] + (int32_t)vo;
        }
        vo += meshes[i].nv; fo += meshes[i].nf;
    }

    char all_path[1024];
    snprintf(all_path, sizeof(all_path), "%.512s/%.256s_%.128s_all.obj",
             dir, cube_id, step_tag);
    { CCColorOpts opts; CCColor_default_opts(&opts); opts.sat = 0.75;
      CCColor_compute(tot_nv, comb_f, tot_nf, &opts, comb_c, NULL);
      ObjIO_write_per_vertex_color(all_path, comb_v, tot_nv, comb_f, tot_nf, comb_c); }
    write_binary_companion(dir, cube_id, all_path, comb_v, tot_nv,
                           comb_f, tot_nf);
    copy_to_all_obj_dir(dir, cube_id, all_path);
    fprintf(stderr, "  [dump] VMESH + %zu OBJs + CC-coloured combined -> %s\n",
            count, dir);

    Arena_restore(arena, mark);
}

void DumpObj_write_one_world(Arena_T arena, const char *path,
                             const char *cube_id,
                             const float *verts, size_t nv,
                             const int32_t *faces, size_t nf,
                             const float color[3])
{
    if (nv == 0 || !path) return;

    float origin[3];
    parse_cube_origin_zyx(cube_id, origin);

    Arena_Mark mark = Arena_save(arena);
    float *shifted = (float *)ARENA_ALLOC(arena,
        (size_t)(nv * 3 * sizeof(float)));
    offset_verts(verts, shifted, nv, origin);
    ObjIO_write_colored(path, shifted, nv, faces, nf, color);
    Arena_restore(arena, mark);
}

void DumpObj_write_points_world(Arena_T arena, const char *path,
                                const char *cube_id,
                                const float *verts, const float *normals,
                                size_t nv)
{
    if (nv == 0 || !path) return;

    float origin[3];
    parse_cube_origin_zyx(cube_id, origin);

    Arena_Mark mark = Arena_save(arena);
    float *shifted = (float *)ARENA_ALLOC(arena,
        (size_t)(nv * 3 * sizeof(float)));
    offset_verts(verts, shifted, nv, origin);

    FILE *fp = fopen(path, "w");
    if (!fp) { Arena_restore(arena, mark); return; }
    for (size_t i = 0; i < nv; i++) {
        fprintf(fp, "v %.6f %.6f %.6f\n",
                (double)shifted[i * 3 + 0],
                (double)shifted[i * 3 + 1],
                (double)shifted[i * 3 + 2]);
    }
    if (normals) {
        for (size_t i = 0; i < nv; i++) {
            fprintf(fp, "vn %.6f %.6f %.6f\n",
                    (double)normals[i * 3 + 0],
                    (double)normals[i * 3 + 1],
                    (double)normals[i * 3 + 2]);
        }
    }
    fclose(fp);
    Arena_restore(arena, mark);
}

void DumpObj_write_mesh_ptrs(Arena_T arena, const char *dir,
                             const char *cube_id, const char *step_tag,
                             const ComponentMesh **meshes, size_t count)
{
    if (count == 0) return;

    Arena_Mark mark = Arena_save(arena);

    float origin[3];
    parse_cube_origin_zyx(cube_id, origin);

    size_t tot_nv = 0, tot_nf = 0;
    for (size_t i = 0; i < count; i++) { tot_nv += meshes[i]->nv; tot_nf += meshes[i]->nf; }
    float   *comb_v = (float *)ARENA_ALLOC(arena, (size_t)(tot_nv * 3 * sizeof(float)));
    int32_t *comb_f = (int32_t *)ARENA_ALLOC(arena, (size_t)(tot_nf * 3 * sizeof(int32_t)));
    float   *comb_c = (float *)ARENA_ALLOC(arena, (size_t)(tot_nv * 3 * sizeof(float)));
    size_t vo = 0, fo = 0;
    char comp_path[COMP_PATH_LEN];

    for (size_t i = 0; i < count; i++) {
        snprintf(comp_path, sizeof(comp_path),
                 "%.512s/%.256s_%.128s_%03d.obj",
                 dir, cube_id, step_tag, (int)(i + 1));
        float *shifted = comb_v + vo * 3;
        offset_verts(meshes[i]->verts, shifted, meshes[i]->nv, origin);
        ObjIO_write_colored(comp_path, shifted, meshes[i]->nv,
                            meshes[i]->faces, meshes[i]->nf,
                            OBJ_COLORS[comp_color_idx(meshes[i])]);
        for (size_t f = 0; f < meshes[i]->nf; f++) {
            comb_f[(fo + f) * 3 + 0] = meshes[i]->faces[f * 3 + 0] + (int32_t)vo;
            comb_f[(fo + f) * 3 + 1] = meshes[i]->faces[f * 3 + 1] + (int32_t)vo;
            comb_f[(fo + f) * 3 + 2] = meshes[i]->faces[f * 3 + 2] + (int32_t)vo;
        }
        vo += meshes[i]->nv; fo += meshes[i]->nf;
    }

    char all_path[1024];
    snprintf(all_path, sizeof(all_path), "%.512s/%.256s_%.128s_all.obj",
             dir, cube_id, step_tag);
    { CCColorOpts opts; CCColor_default_opts(&opts); opts.sat = 0.75;
      CCColor_compute(tot_nv, comb_f, tot_nf, &opts, comb_c, NULL);
      ObjIO_write_per_vertex_color(all_path, comb_v, tot_nv, comb_f, tot_nf, comb_c); }
    write_binary_companion(dir, cube_id, all_path, comb_v, tot_nv,
                           comb_f, tot_nf);
    copy_to_all_obj_dir(dir, cube_id, all_path);
    fprintf(stderr, "  [dump] VMESH + %zu OBJs + CC-coloured combined -> %s\n",
            count, dir);

    Arena_restore(arena, mark);
}
