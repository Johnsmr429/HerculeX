#ifndef CUDA_DRV_INCLUDED
#define CUDA_DRV_INCLUDED

#include <stddef.h>
#include <stdint.h>

typedef unsigned long long CudaDrv_ptr;

typedef struct CudaDrv *CudaDrv;

CudaDrv CudaDrv_open(const char *source, const char *const *kernel_names, int n_kernels,
                     char *why, size_t why_len);
void    CudaDrv_close(CudaDrv cu);

const char *CudaDrv_device(CudaDrv cu);

int  CudaDrv_alloc(CudaDrv cu, CudaDrv_ptr *p, size_t bytes);
void CudaDrv_free(CudaDrv cu, CudaDrv_ptr p);
int  CudaDrv_upload(CudaDrv cu, CudaDrv_ptr dst, const void *src, size_t bytes);
int  CudaDrv_download(CudaDrv cu, void *dst, CudaDrv_ptr src, size_t bytes);
int  CudaDrv_memset8(CudaDrv cu, CudaDrv_ptr dst, uint8_t value, size_t bytes);

int CudaDrv_launch(CudaDrv cu, int k, unsigned gx, unsigned gy, unsigned bx, unsigned by, void **params);

const char *CudaDrv_error(CudaDrv cu);

int CudaDrv_selftest(void);

#endif
