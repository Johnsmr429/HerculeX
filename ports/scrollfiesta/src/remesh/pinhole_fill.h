#ifndef PINHOLE_FILL_INCLUDED
#define PINHOLE_FILL_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"

int PinholeFill_process(Arena_T arena,
                        ComponentMesh *meshes, size_t n_meshes,
                        int respect_pins,
                        size_t *out_pinch_splits,
                        size_t *out_loops_filled,
                        size_t *out_tris_added,
                        size_t *out_loops_skipped);

int PinholeFill_split_pinches(Arena_T arena,
                              ComponentMesh *meshes, size_t n_meshes,
                              size_t *out_splits);

int PinholeFill_split_pinches_mapped(Arena_T arena,
                                     ComponentMesh *mesh,
                                     size_t *out_splits,
                                     int32_t **out_vertex_source);

int PinholeFill_close_bowties(Arena_T arena,
                              ComponentMesh *meshes, size_t n_meshes,
                              size_t *out_closed);
int PinholeFill_close_bridge_bowties(Arena_T arena,
                                     ComponentMesh *bridge,
                                     const int32_t *source_faces,
                                     size_t source_nf,
                                     const int32_t *vertex_chart,
                                     size_t n_charts,
                                     int defer_source_orientation,
                                     size_t *out_closed);
int PinholeFill_close_bridge_attachments(Arena_T arena,
                                         ComponentMesh *combined,
                                         size_t source_nf,
                                         const int32_t *vertex_chart,
                                         size_t n_charts,
                                         const size_t *chart_face_count,
                                         int defer_source_orientation,
                                         size_t *out_closed);
int PinholeFill_fill_small_loops(Arena_T arena,
                                 ComponentMesh *meshes, size_t n_meshes,
                                 int respect_pins,
                                 size_t *out_loops_filled,
                                 size_t *out_tris_added,
                                 size_t *out_loops_skipped);

typedef int (*HoleFill_cdt_fn)(Arena_T arena, ComponentMesh *cm, void *user);

int HoleFill_meshes(Arena_T arena,
                    ComponentMesh *meshes, size_t n_meshes,
                    int respect_pins,
                    HoleFill_cdt_fn cdt_fill, void *cdt_user,
                    size_t *out_pinch_splits,
                    size_t *out_loops_filled,
                    size_t *out_tris_added,
                    size_t *out_loops_skipped,
                    size_t *out_cdt_filled);

#define PINHOLE_MAX_LOOP 12

#define PINHOLE_MAX_DIAM_VOX 4.5f

#define PINHOLE_TRI_DIAM_VOX 6.5f

#define HOLEFILL_MIN_ALT_VOX 0.05f

#define BOWTIE_COHERENCE_COS 0.2f
#define BOWTIE_GAP_MAX_VOX   2.0f

#define BRIDGE_MICRO_CHART_MAX_FACES             128
#define BRIDGE_MICRO_ATTACHMENT_COHERENCE_COS   0.08f

#define BOWTIE_GAP_MAX_ARMED_VOX  6.5f

#endif
