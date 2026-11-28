#ifndef OBJ_IO_INCLUDED
#define OBJ_IO_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

int ObjIO_write(const char *path, const float *verts, size_t nv,
                const int32_t *faces, size_t nf);

int ObjIO_write_colored(const char *path, const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const float color[3]);

int ObjIO_write_twotone(const char *path, const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        size_t nv_split,
                        const float color1[3], const float color2[3]);

int ObjIO_write_per_vertex_color(const char *path,
                                 const float *verts, size_t nv,
                                 const int32_t *faces, size_t nf,
                                 const float *colors);

int ObjIO_read(Arena_T arena, const char *path,
               float **out_verts, size_t *out_nv,
               int32_t **out_faces, size_t *out_nf);

int ObjIO_write_uv(const char *path, const float *verts, size_t nv,
                   const int32_t *faces, size_t nf, const float *uv);

int ObjIO_write_uv_double(const char *path, const float *verts, size_t nv,
                          const int32_t *faces, size_t nf, const double *uv);

int ObjIO_write_uv_masked(const char *path, const float *verts, size_t nv,
                          const int32_t *faces, size_t nf, const float *uv,
                          const uint8_t *keep);

#endif
