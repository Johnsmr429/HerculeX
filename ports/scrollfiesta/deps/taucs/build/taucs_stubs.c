
double taucs_available_memory_size(void)
{
    return 8.0e9;
}

void *taucs_amwb_preconditioner_create(void *A, int a, double b, int c)
{
    (void)A; (void)a; (void)b; (void)c;
    return 0;
}

void *taucs_io_create_multifile(char *filename) { (void)filename; return 0; }
void *taucs_io_open_multifile  (char *filename) { (void)filename; return 0; }
int   taucs_io_close (void *f) { (void)f; return -1; }
int   taucs_io_delete(void *f) { (void)f; return -1; }

int taucs_ooc_factor_llt(void *A, void *handle, double memory)
{
    (void)A; (void)handle; (void)memory;
    return -1;
}

int taucs_ooc_solve_llt(void *L, void *x, void *b)
{
    (void)L; (void)x; (void)b;
    return -1;
}
