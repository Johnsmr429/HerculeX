#ifndef MESH_PILE_INCLUDED
#define MESH_PILE_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

#define MESH_PILE_MAX_PATH 1024

typedef struct {
    char path[MESH_PILE_MAX_PATH];
    char cube_id[24];
    int32_t oz, oy, ox;
    int  has_id;
} MeshPileEntry;

typedef int (*MeshPile_Filter)(void *ctx, long oz, long oy, long ox);

int MeshPile_parse_cube_id(const char *name, char id[24],
                           long *oz, long *oy, long *ox);

int MeshPile_scan(const char *dir, MeshPileEntry *pile, size_t cap,
                  size_t *n, MeshPile_Filter filter, void *ctx);

int MeshPile_scan_arena(Arena_T arena, const char *dir, MeshPileEntry **pile,
                        size_t *n, MeshPile_Filter filter, void *ctx);

void MeshPile_sort(MeshPileEntry *pile, size_t n);

int MeshPile_selftest(void);

#endif
