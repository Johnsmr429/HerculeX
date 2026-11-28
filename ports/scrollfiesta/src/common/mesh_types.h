#ifndef MESH_TYPES_INCLUDED
#define MESH_TYPES_INCLUDED

#include <stdint.h>
#include <stddef.h>

typedef struct ComponentMesh {
    float   *verts;
    int32_t *faces;
    uint8_t *pin_mask;
    float   *vert_normals;
    size_t   nv;
    size_t   nf;
    int      comp_id;
    float    pca_normal[3];
    float    centroid[3];
    size_t   nv_pre_fill;
    void    *self;
} ComponentMesh;

static inline int ComponentMesh_valid(const ComponentMesh *cm)
{
    return cm && cm->self == (void *)cm && cm->nv > 0 && cm->verts
           && cm->faces && cm->nf > 0;
}

#endif
