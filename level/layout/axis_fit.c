#include <stddef.h>

#include <math.h>

#ifndef HX_UNITY
#include "../level/scroll_axis.c"

void hx_centres_at(
    const HxAxisKnot *knots, int64_t count, const double *z, size_t size, double *x, double *y) {
    for (size_t i = 0; i < size; i++) {
        hx_centre_at(knots, count, z[i], x + i, y + i);
    }
}

#endif

static int solve_focus(double a, double b, double d, double u, double v, double centre[2]) {
    double right = b, below = b;
    if (fabs(b) > fabs(a)) {
        below = a;
        a = b;
        right = d;
        d = b;
        double swap = u;
        u = v;
        v = swap;
    }
    if (!a) {
        return 0;
    }
    double factor = below / a;
    double pivot = d - factor * right;
    if (!pivot) {
        return 0;
    }
    double y = (v - factor * u) / pivot;
    double x = (u - right * y) / a;
    centre[0] = x;
    centre[1] = y;
    return isfinite(x) && isfinite(y);
}

static int focus(const double *x, const double *y, const double *ux, const double *uy,
    const double *weight, size_t size, double reach, double scale, double centre[2]) {
    double a = 0, b = 0, d = 0, u = 0, v = 0;
    double inverse_reach = reach > 0 ? 1 / reach : 0;
    double inverse_scale = scale > 0 ? 1 / scale : 0;
    for (size_t i = 0; i < size; i++) {
        double tx = -uy[i], ty = ux[i];
        double w = weight[i];
        if (reach > 0) {
            double dx = x[i] - centre[0], dy = y[i] - centre[1];
            double rx = dx * inverse_reach, ry = dy * inverse_reach, distance = rx * rx + ry * ry;
            double miss = (-dx * tx - dy * ty) * inverse_scale;
            w = w * (1 / (1 + distance * distance)) / (1 + miss * miss);
        }
        double wx = w * tx, wy = w * ty, position = tx * x[i] + ty * y[i];
        a += wx * tx;
        b += wx * ty;
        d += wy * ty;
        u += wx * position;
        v += wy * position;
    }
    return solve_focus(a, b, d, u, v, centre);
}

int hx_fit_centre(const double *x, const double *y, const double *ux, const double *uy,
    const double *weight, size_t size, int robust, double centre[2]) {
    if (!robust) {
        return focus(x, y, ux, uy, weight, size, 0, 0, centre);
    }
    double weight_sum = 0, sx = 0, sy = 0;
    for (size_t i = 0; i < size; i++) {
        weight_sum += weight[i];
        sx += weight[i] * x[i];
        sy += weight[i] * y[i];
    }
    if (!(weight_sum > 0)) {
        return 0;
    }
    centre[0] = sx / weight_sum;
    centre[1] = sy / weight_sum;
    double spread = 0;
    for (size_t i = 0; i < size; i++) {
        double dx = x[i] - centre[0], dy = y[i] - centre[1];
        spread += weight[i] * (dx * dx + dy * dy);
    }
    double radius = sqrt(spread / weight_sum), scale = radius / 2;
    if (!(radius > 0 && isfinite(radius))) {
        return 0;
    }
    for (int step = 0; step < 12; step++) {
        double reach = radius * pow(.75, step);
        for (int retry = 0; retry < 3; retry++) {
            if (!focus(x, y, ux, uy, weight, size, reach, scale, centre)) {
                return 0;
            }
            scale = fmax(scale * .8, 8);
        }
    }
    return 1;
}
