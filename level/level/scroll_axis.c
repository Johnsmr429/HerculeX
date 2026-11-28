#include <math.h>
#include <stdint.h>

typedef struct {
    double z, x, y;
} HxAxisKnot;

static double axis_interpolate(double z, HxAxisKnot a, HxAxisKnot b, int field) {
    double av = field ? a.y : a.x;
    double bv = field ? b.y : b.x;
    if (z == a.z)
        return av;
    double slope = (bv - av) / (b.z - a.z);
    double value = fma(slope, z - a.z, av);
    if (isnan(value)) {
        value = fma(slope, z - b.z, bv);
        if (isnan(value) && av == bv)
            return av;
    }
    return value;
}

void hx_centre_at(const HxAxisKnot *knots, int64_t count, double z, double *x, double *y) {
    int64_t last = count - 1;
    if (last <= 0 || z < knots[0].z) {
        *x = knots[0].x;
        *y = knots[0].y;
        return;
    }
    if (z >= knots[last].z) {
        *x = knots[last].x;
        *y = knots[last].y;
        return;
    }
    int64_t low = 0, high = last;
    while (high - low > 1) {
        int64_t middle = low + (high - low) / 2;
        if (knots[middle].z <= z)
            low = middle;
        else
            high = middle;
    }
    *x = axis_interpolate(z, knots[low], knots[high], 0);
    *y = axis_interpolate(z, knots[low], knots[high], 1);
}
