#ifndef ASM_READING_ORDER_INCLUDED
#define ASM_READING_ORDER_INCLUDED
#include "asm_types.h"
#include "asm_axis.h"

int AsmReadingOrder_write(const AsmRun *run,const AsmAxis *axis,const char *dir);
typedef struct AsmRibbonKey {
    double turn_lo,turn_hi,angular_gradient[2];
    int source_supported;
} AsmRibbonKey;
int AsmReadingOrder_ribbon_keys(const AsmRun *run,const AsmAxis *axis,AsmRibbonKey *keys);
int AsmReadingOrder_selftest(void);
#endif
