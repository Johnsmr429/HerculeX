#ifndef ASM_STORE_INCLUDED
#define ASM_STORE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "asm_types.h"

#define ASM_STORE_CODE_VERSION "asm-clean-2026-09-14-chart-area-recovery"

uint64_t AsmStore_fingerprint(const char *text);

int AsmStore_write_cube(const char *dir, const char *cube_id,
                        const char *src_path, uint64_t fingerprint,
                        const AsmChart *charts, size_t n);

int AsmStore_read_cube(Arena_T arena, const char *dir, const char *cube_id,
                       const char *src_path, uint64_t fingerprint,
                       AsmChart **out_charts, size_t *out_n);

int AsmStore_exists(const char *dir, const char *cube_id);

int AsmStore_selftest(void);

int AsmStore_write_run(const char *path, const AsmRun *run);
int AsmStore_read_run(const char *path, AsmRun *run);

#endif
