#ifndef ASM_FIBRE_FRAME_INCLUDED
#define ASM_FIBRE_FRAME_INCLUDED

#include <stddef.h>
#include "asm_types.h"
#include "../common/zarr_u8.h"

typedef struct AsmFibreMoments { double w, m2[2], m4[2]; size_t px; } AsmFibreMoments;

typedef struct AsmFibreStats { size_t charts, px, skipped; size_t chunk_misses, chunk_missing; double seconds; } AsmFibreStats;

int AsmFibre_measure(const AsmRun *run, const char *raw, AsmFibreMoments *out, AsmFibreStats *st);

void AsmFibre_measure_patch(ZarrU8 *volume, Arena_T arena, const AsmChart *chart,
                            AsmFibreMoments *out, size_t *face_samples);

void AsmFibre_transform(AsmFibreMoments *m, int mirror, double cosine, double sine);

typedef struct AsmFibreTurn {
    double coherence, cross_deg, energy_near, energy_far, turn_deg;
    int decided_by;
} AsmFibreTurn;
void AsmFibre_turn(const AsmFibreMoments *m, AsmFibreTurn *out);

int AsmFibre_selftest(void);

#endif
