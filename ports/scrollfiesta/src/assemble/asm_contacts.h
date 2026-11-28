#ifndef ASM_CONTACTS_INCLUDED
#define ASM_CONTACTS_INCLUDED
#include "../common/arena.h"
#include <stddef.h>
#include <stdint.h>

typedef struct AsmContacts AsmContacts;
double AsmContacts_material_tolerance(void);
const char *AsmContacts_material_policy(void);
typedef struct AsmContactStats { size_t pairs; double area; size_t leaf_pairs; int complete; size_t unmeasured; } AsmContactStats;
typedef void (*AsmContactVisit)(void *context, size_t a, size_t b, double area);
size_t AsmContacts_unmeasured_sample(const AsmContacts *query, size_t *ab, size_t cap);
AsmContacts *AsmContacts_new(Arena_T arena, size_t nv, const int32_t *faces,
                             size_t nf, const double *uv, double tolerance);
int AsmContacts_rebuild(AsmContacts *query, const double *uv);
AsmContacts *AsmContacts_new_threads(Arena_T arena, size_t nv, const int32_t *faces,
                                     size_t nf, const double *uv, double tolerance, int threads);
AsmContacts *AsmContacts_new_source(Arena_T arena, size_t nv, const int32_t *faces,
                                    size_t nf, const double *uv, double tolerance, int threads,
                                    const int32_t *material_vertex);
int AsmContacts_measure(AsmContacts *query, const double *uv, double *gradient,
                        AsmContactVisit visit, void *context, AsmContactStats *out);
int AsmContacts_objective(AsmContacts *query,const double *uv,double *gradient,
    AsmContactVisit visit,void *context,AsmContactStats *out,double *raw_area);
typedef double (*AsmContactWeight)(void *context,size_t a,size_t b,double raw_area);
int AsmContacts_weighted_objective(AsmContacts *query,const double *uv,double *gradient,
    AsmContactVisit visit,void *context,AsmContactWeight weight,void *weight_context,
    AsmContactStats *out,double *raw_area);
void AsmContacts_leaf_pair_budget(AsmContacts *query, size_t budget);
int AsmContacts_refit(AsmContacts *query, const double *uv);
int AsmContacts_refit_changed(AsmContacts *query,const double *uv,const uint8_t *changed_faces);
int AsmContacts_probe(AsmContacts *query, const double *uv, size_t n_ext, const double *tri,
                      const uint8_t *exclude_face, AsmContactVisit visit, void *context, AsmContactStats *out);
void AsmContacts_prepare_exclusion(AsmContacts *query,const uint8_t *exclude_face);
int AsmContacts_probe_prepared(AsmContacts *query, size_t n_ext, const double *tri,
                               const uint8_t *exclude_face, AsmContactVisit visit, void *context, AsmContactStats *out);
int AsmContacts_probe_mesh(AsmContacts *query,const double *uv,AsmContacts *source,const double *source_uv,
    size_t n_ext,const size_t *source_face,const uint8_t *exclude_face,AsmContactVisit visit,void *context,AsmContactStats *out);
int AsmContacts_probe_mesh_prepared(AsmContacts *query,AsmContacts *source,const double *source_uv,
    size_t n_ext,const size_t *source_face,const uint8_t *exclude_face,AsmContactVisit visit,void *context,AsmContactStats *out);
size_t AsmContacts_budget(const AsmContacts *query);
size_t AsmContacts_leaves(const AsmContacts *query);
int AsmContacts_touching(AsmContacts *query, const double *uv,
                         AsmContactVisit visit, void *context, AsmContactStats *out);
void AsmContacts_separate_frames(AsmContacts *query, const int32_t *frame_per_face);
void AsmContacts_active_faces(AsmContacts *query, const uint8_t *moving_faces);
void AsmContacts_memo_counts(const AsmContacts *query, size_t *measured, size_t *replayed);

void AsmContacts_threads(AsmContacts *query, int threads);
void AsmContacts_no_memo(AsmContacts *query);
double AsmContacts_pair(const double a[6], const double b[6], double tolerance,
                        double ga[6], double gb[6]);
double AsmContacts_pair_faces(const AsmContacts *query,const double *uv,size_t a,size_t b,double ga[6],double gb[6]);
int AsmContacts_selftest(void);
int AsmContacts_bench(int cells, int reps);
void AsmContacts_pair_counts(long long out[7]);
void AsmContacts_hit_classes(long long out[5]);
void AsmContacts_leaf_counts(long long out[3]);
int AsmContacts_set_charts(AsmContacts *query,const int32_t *face_chart,size_t n_ids);
void AsmContacts_chart_counts(const AsmContacts *query,size_t out[3]);
#endif
