#ifndef CLOSEST_TRI_INCLUDED
#define CLOSEST_TRI_INCLUDED

void ClosestTri_point(const double p[3], const double a[3],
                      const double b[3], const double c[3],
                      double out[3], double *out_u, double *out_v);

double ClosestTri_dist2(const double p[3], const double a[3],
                        const double b[3], const double c[3]);

#endif
