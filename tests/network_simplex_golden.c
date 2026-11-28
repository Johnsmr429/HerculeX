#include <stdio.h>
#include "../level/level/l1_synchronization.c"

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "network simplex golden: %s\n", message);
        exit(1);
    }
}

static double energy(const double *f, int n, const int64_t *a, const int64_t *b, const double *k,
    const double *w, const double *bound) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double residual = f[b[i]] - f[a[i]] - k[i];
        double leniency = residual > 0 ? bound[i] < 0 : bound[i] > 0;
        sum += fabs(residual) * w[i] * (leniency ? .5 : 1.);
    }
    return sum;
}

int main(void) {
    int64_t a[] = {0, 1, 2, 0, 1, 0}, b[] = {1, 2, 3, 2, 3, 3};
    double k[] = {2, -1, 3, 0, 0, 2}, w[] = {1., .75, .5, .25, .625, .875};
    double bound[] = {0., 1., -1., 0., 1., -1.}, f[4], brute[4] = {0};
    require(!hx_wind_solve(WIND_OFFSETS, 4, 6, a, b, k, w, bound, 4, NULL, NULL, NULL, f, NULL),
        "solve a graph with conflicting votes");
    double best = INFINITY;
    for (int x = -10; x <= 10; x++) {
        brute[1] = x;
        for (int y = -10; y <= 10; y++) {
            brute[2] = y;
            for (int z = -10; z <= 10; z++) {
                brute[3] = z;
                best = fmin(best, energy(brute, 6, a, b, k, w, bound));
            }
        }
    }
    require(energy(f, 6, a, b, k, w, bound) == best, "exact objective equals brute force");
    for (int i = 0; i < 4; i++) {
        require(f[i] == nearbyint(f[i]), "winding offsets are integral");
    }
    int64_t term[4];
    k[0] = NAN;
    require(
        hx_wind_solve(WIND_AGGREGATE, 4, 1, a, b, k, w, bound, 4, NULL, NULL, term, f, NULL) == -5,
        "reject invalid winding votes");
    k[0] = INFINITY;
    require(
        hx_wind_solve(WIND_AGGREGATE, 4, 1, a, b, k, w, bound, 4, NULL, NULL, term, f, NULL) == -5,
        "reject infinite winding votes");

    enum { N = 24, EDGES = 2 * N - 3 };
    int64_t left[EDGES], right[EDGES], zone[N];
    double votes[EDGES], weight[EDGES], bounds[EDGES] = {0}, exact[N], window[N];
    int32_t labels[N];
    for (int i = 0; i < N; i++) {
        zone[i] = i;
    }
    for (int i = 0; i < EDGES; i++) {
        int step = i < N - 1 ? 1 : 2;
        left[i] = step == 1 ? i : i - N + 1;
        right[i] = left[i] + step;
        votes[i] = right[i] % 5 - left[i] % 5 + (step == 2 && i % 4 == 0);
        weight[i] = step == 1 ? 1. : .125;
    }
    require(!hx_wind_solve(WIND_OFFSETS, N, EDGES, left, right, votes, weight, bounds, 8, NULL,
                NULL, NULL, exact, NULL) &&
            hx_wind_solve(WIND_SYNCHRONIZE, N, EDGES, left, right, votes, weight, bounds, 8, zone,
                NULL, NULL, window, labels) >= 0 &&
            energy(window, EDGES, left, right, votes, weight, bounds) <=
                1.02 * energy(exact, EDGES, left, right, votes, weight, bounds),
        "windowed solve stays within 2% of the full optimum");
    return 0;
}
