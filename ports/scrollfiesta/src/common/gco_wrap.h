#ifndef GCO_WRAP_INCLUDED
#define GCO_WRAP_INCLUDED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GCO_Opaque *GCO_Handle;

typedef int32_t (*GCO_SmoothCostCallback)(
    int site1, int site2, int label1, int label2, void *userdata);

GCO_Handle GCO_create(int num_sites, int num_labels);

void GCO_set_data_cost(GCO_Handle gc, const int32_t *data);

void GCO_set_smooth_cost(GCO_Handle gc, const int32_t *smooth);

void GCO_set_smooth_cost_callback(
    GCO_Handle gc, GCO_SmoothCostCallback callback, void *userdata);

void GCO_set_neighbor(GCO_Handle gc, int s1, int s2, int weight);

long long GCO_expansion(GCO_Handle gc, int max_iter);

long long GCO_swap(GCO_Handle gc, int max_iter);

int GCO_what_label(GCO_Handle gc, int site);

void GCO_get_labels(GCO_Handle gc, int *out_labels, int num_sites);

void GCO_set_label(GCO_Handle gc, int site, int label);

void GCO_destroy(GCO_Handle gc);

#ifdef __cplusplus
}
#endif

#endif
