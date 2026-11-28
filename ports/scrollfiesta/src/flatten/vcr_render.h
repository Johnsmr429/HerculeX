#ifndef VCR_RENDER_INCLUDED
#define VCR_RENDER_INCLUDED

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *volume;
    const char *segmentation;
    int         composite;
    int         num_slices;
    double      slice_step;
    int         flip_normals;
    int         comp_start, comp_end;
    const char *zarr_output;
    const char *tif_output;
    double      voxel_size;
    const char *voxel_unit;
    int         zarr_shard_cols;
    size_t      gpu_cache_mb;
} VcrRender_job;

int VcrRender_run(const VcrRender_job *job);

int VcrRender_selftest(void);

int VcrRender_storage_probe(const char *input_array, const char *output_group, int shard_cols);

#endif
