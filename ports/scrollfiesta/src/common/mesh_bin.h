#ifndef MESH_BIN_INCLUDED
#define MESH_BIN_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "arena.h"

enum { MESH_BIN_HAS_UV = 1u, MESH_BIN_UV_F64 = 2u };

typedef struct MeshBinData {
    float *verts;
    float *uv;
    int32_t *faces;
    size_t nv, nf;
    double *uv64;
} MeshBinData;

static inline double MeshBin_uv(const MeshBinData *mesh, size_t value)
{
    return mesh->uv64 ? mesh->uv64[value] : (double)mesh->uv[value];
}

int MeshBin_companion_path(const char *path, char *out, size_t out_cap);

int MeshBin_looks_complete(const char *path);

typedef struct MeshBinHeader {
    uint64_t nv, nf, vertices_offset, uv_offset, faces_offset, file_bytes;
    uint32_t flags;
} MeshBinHeader;
int MeshBin_header(const char *path, MeshBinHeader *out);

int MeshBin_write(const char *path, const float *verts, size_t nv,
                  const int32_t *faces, size_t nf, const float *uv);
int MeshBin_write_uv64(const char *path, const float *verts, size_t nv,
                        const int32_t *faces, size_t nf, const double *uv);

typedef struct MeshBinStream_T *MeshBinStream_T;
int MeshBin_stream_open(Arena_T arena, const char *path, size_t nv, size_t nf,
                        int has_uv, MeshBinStream_T *out);
int MeshBin_stream_open_uv64(Arena_T arena, const char *path, size_t nv, size_t nf,
                             MeshBinStream_T *out);
int MeshBin_stream_append(MeshBinStream_T writer, const MeshBinData *chunk);
int MeshBin_stream_append_mapped(MeshBinStream_T writer, const MeshBinData *chunk,
                                 const int32_t *map, size_t nv);
int MeshBin_stream_close(MeshBinStream_T *writer, int publish);

int MeshBin_read_arena(Arena_T arena, const char *path, MeshBinData *out);
int MeshBin_read_malloc(const char *path, MeshBinData *out);
int MeshBin_read_precise_arena(Arena_T arena, const char *path, MeshBinData *out);
int MeshBin_read_precise_malloc(const char *path, MeshBinData *out);
void MeshBin_dispose(MeshBinData *mesh);

int MeshBin_selftest(void);

#endif
