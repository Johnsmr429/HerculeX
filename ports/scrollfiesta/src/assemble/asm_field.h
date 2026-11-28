#ifndef ASM_FIELD_INCLUDED
#define ASM_FIELD_INCLUDED
#include "asm_types.h"
#include "asm_metric.h"
#include "asm_contacts.h"

typedef struct AsmWitness {
    int32_t vertex[2][4];
    double coefficient[2][4];
    double weight;
    int valid, required;
} AsmWitness;
typedef struct AsmFieldSeam {
    size_t relation, first, count;
    int32_t a, b;
    int complete, parity, source_run, required, original_cut;
} AsmFieldSeam;
typedef struct AsmSeamMeasure {
    double mass, support, rms, energy;
    size_t observations, invalid;
    int pass;
} AsmSeamMeasure;
typedef struct AsmField {
    Arena_T arena;
    const AsmRun *run;
    size_t nv, nf, charts;
    size_t *vertex_offset, *face_offset;
    float *xyz;
    int32_t *faces, *face_chart, *vertex_chart;
    int32_t *material_vertex;
    double *uv, *mass;
    AsmMetricFace *metric;
    AsmFieldSeam *seams;
    AsmWitness *witness;
    size_t n_seams, n_witness;
    AsmContacts *contacts;
} AsmField;

int AsmField_threads(int threads);

int AsmField_relation_required(const AsmRelation *relation);
int AsmField_relation_torn(const AsmRelation *relation);

AsmField *AsmField_new(Arena_T arena, const AsmRun *run);
AsmField *AsmField_new_local(Arena_T arena, const AsmRun *run, const uint8_t *member, const AsmField *parent);
int AsmField_build_contacts(AsmField *field);
int AsmField_material_map(Arena_T arena, const AsmRun *run, const size_t *offset,
                          size_t nv, int32_t **material);
int AsmField_commit(AsmField *field, AsmRun *run);
int AsmField_write(const AsmField *field, const char *out_dir);
int AsmField_verify(const AsmField *field, const char *out_dir);
void AsmField_face(const AsmField *field, size_t face, const double *uv,
                    double j[4], double *lo, double *hi, double *det);
void AsmField_seam(const AsmField *field, size_t seam, const double *uv, AsmSeamMeasure *out);
int AsmField_measure_source_seams(const AsmRun *run, AsmSeamMeasure *out);
void AsmField_residual(const AsmWitness *w, int direction, const double *uv, double r[2]);
int AsmField_selftest(void);
#endif
