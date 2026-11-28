#ifndef SIZING_FIELD_INCLUDED
#define SIZING_FIELD_INCLUDED

#include <math.h>

typedef struct {
    double lo, hi;
    double h_seam;
    double h_interior;
    double band_lo;
    double band_hi;
    int    energy_exp;
} CvtField;

static inline double sizing_h(const CvtField *f, double c0, double c1, double c2) {
    double d0 = c0 - f->lo, d1 = f->hi - c0;
    double d2 = c1 - f->lo, d3 = f->hi - c1;
    double d4 = c2 - f->lo, d5 = f->hi - c2;
    double d = d0;
    if (d1 < d) d = d1;
    if (d2 < d) d = d2;
    if (d3 < d) d = d3;
    if (d4 < d) d = d4;
    if (d5 < d) d = d5;
    double w = f->band_hi - f->band_lo;
    double u = (w > 1e-9) ? (d - f->band_lo) / w : (d >= f->band_lo ? 1.0 : 0.0);
    if (u < 0.0) u = 0.0; else if (u > 1.0) u = 1.0;
    double t = u * u * (3.0 - 2.0 * u);
    return f->h_seam + (f->h_interior - f->h_seam) * t;
}

static inline double Rvd_density_rho(const CvtField *f, double c0, double c1, double c2) {
    double h = sizing_h(f, c0, c1, c2);
    if (h < 1e-9) h = 1e-9;
    double h2 = h * h;
    return (f->energy_exp == 2) ? (1.0 / h2) : (1.0 / (h2 * h2));
}

static inline double sizing_seed_rho(const CvtField *f, double c0, double c1, double c2) {
    double h = sizing_h(f, c0, c1, c2);
    if (h < 1e-9) h = 1e-9;
    return 1.0 / (h * h);
}

#endif
