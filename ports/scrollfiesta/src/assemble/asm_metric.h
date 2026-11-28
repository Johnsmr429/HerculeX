#ifndef ASM_METRIC_INCLUDED
#define ASM_METRIC_INCLUDED
#include "asm_types.h"
#include "../common/pipeline_constants.h"

static inline int AsmMetric_in_band(double lo,double hi,double lower,double upper)
{ return lo<=hi+ASM_METRIC_ROUNDOFF && lo>=lower-ASM_METRIC_ROUNDOFF && hi<=upper+ASM_METRIC_ROUNDOFF; }
static inline int AsmMetric_within10(double lo,double hi)
{ return AsmMetric_in_band(lo,hi,.90,1.10); }
static inline int AsmMetric_within25(double lo,double hi)
{ return AsmMetric_in_band(lo,hi,.75,1.25); }

typedef struct AsmMetricFace { double length, along, height, area; } AsmMetricFace;
typedef struct AsmMetricStats {
    double area, within10, within25, minimum, maximum;
    size_t invalid;
} AsmMetricStats;

int AsmMetric_prepare(const AsmChart *chart, AsmMetricFace *ref);
void AsmMetric_measure(const AsmChart *chart, const AsmMetricFace *ref,
                       const float *uv, AsmMetricStats *out);
void AsmMetric_measure_encoded(const AsmChart *chart, const AsmMetricFace *ref,
                               const float *uv, AsmMetricStats *out);
void AsmMetric_measure_encoded_uv64(const AsmChart *chart, const AsmMetricFace *ref,
                                    const double *uv, AsmMetricStats *out);
int AsmMetric_preserved(const AsmMetricStats *before, const AsmMetricStats *after);
int AsmMetric_local_face_preserved(const AsmChart *chart, size_t face,
                                  const AsmMetricFace *ref, const float *trial);
int AsmMetric_encoded_faces_preserved(const AsmChart *chart, const AsmMetricFace *ref, const double *encoded);
int AsmMetric_encoded_faces_preserved_parity(const AsmChart *chart, const AsmMetricFace *ref, const double *encoded,
                                             int encoded_mirror);
int AsmMetric_selftest(void);
#endif
