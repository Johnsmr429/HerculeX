#include "lane_turn_mrf.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "winding_mrf.h"

#define LT_TRUST_SIGMA        1.0
#define LT_ABSTAIN_THRESHOLD  0.35
#define LT_MAX_ITERATIONS     6
#define LT_ROUNDS             4
#define LT_WEIGHT_CAP         64.0
#define LT_TOTAL_MAX          (LT_ROUNDS * LANE_TURN_MAX_SHIFT)

int LaneTurn_solve(Arena_T arena,
                   const LaneTurnEdge *edges, size_t nedges,
                   const double *lane_support, size_t nlanes,
                   double trust_weight,
                   int32_t **out_shift,
                   LaneTurnStats *stats)
{
    WindingMRFSite *sites = NULL;
    WindingMRFEdge *medges = NULL;
    LaneTurnEdge *kept = NULL;
    int32_t *total = NULL;
    WindingMRFOptions opt;
    size_t i = 0, ne = 0, anchor = 0;
    int round = 0;
    double best_support = -1.0;
    double energy_first = 0.0, energy_last = 0.0;

    if (out_shift == NULL || nlanes == 0 || nlanes > (size_t)INT32_MAX)
        return -1;
    *out_shift = NULL;
    if (stats != NULL) memset(stats, 0, sizeof(*stats));

    kept = (LaneTurnEdge *)ARENA_ALLOC(
        arena, (size_t)((nedges ? nedges : 1) * sizeof(*kept)));
    for (i = 0; i < nedges; i++) {
        if (edges[i].a == edges[i].b) continue;
        if (edges[i].a < 0 || edges[i].b < 0) continue;
        if ((size_t)edges[i].a >= nlanes || (size_t)edges[i].b >= nlanes)
            continue;
        if (!(edges[i].weight > 0.0) || !isfinite(edges[i].weight)) continue;
        kept[ne++] = edges[i];
    }

    {
        double wmax = 0.0;
        for (i = 0; i < ne; i++) if (kept[i].weight > wmax) wmax = kept[i].weight;
        if (wmax > LT_WEIGHT_CAP) {
            double scale = LT_WEIGHT_CAP / wmax;
            for (i = 0; i < ne; i++) kept[i].weight *= scale;
            trust_weight *= scale;
        }
    }

    total = (int32_t *)ARENA_CALLOC(arena, nlanes, sizeof(*total));
    sites = (WindingMRFSite *)ARENA_ALLOC(
        arena, (size_t)(nlanes * sizeof(*sites)));
    medges = (WindingMRFEdge *)ARENA_ALLOC(
        arena, (size_t)((ne ? ne : 1) * sizeof(*medges)));

    for (i = 0; i < nlanes; i++) {
        if (lane_support != NULL && lane_support[i] > best_support) {
            best_support = lane_support[i];
            anchor = i;
        }
    }

    WindingMRF_default_options(&opt);
    opt.label_min = -LANE_TURN_MAX_SHIFT;
    opt.label_max = LANE_TURN_MAX_SHIFT;
    opt.max_iterations = LT_MAX_ITERATIONS;
    opt.abstain_threshold = LT_ABSTAIN_THRESHOLD;

    for (round = 0; round < LT_ROUNDS; round++) {
        int32_t *labels = NULL;
        float *conf = NULL;
        WindingMRFStats ms;
        size_t changed = 0;

        for (i = 0; i < nlanes; i++) {
            sites[i].center = -(double)total[i];
            sites[i].sigma = LT_TRUST_SIGMA;
            sites[i].weight = trust_weight;
            sites[i].initial_label = 0;
            sites[i].fixed = (i == anchor);
        }
        for (i = 0; i < ne; i++) {
            medges[i].a = kept[i].a;
            medges[i].b = kept[i].b;
            medges[i].target = kept[i].target
                             - (total[kept[i].b] - total[kept[i].a]);
            medges[i].weight = kept[i].weight;
        }
        memset(&ms, 0, sizeof(ms));
        if (WindingMRF_solve(arena, sites, nlanes, medges, ne, &opt,
                             &labels, &conf, &ms) != 0)
            return -1;
        if (round == 0) energy_first = ms.energy_before;
        energy_last = ms.energy_after;
        for (i = 0; i < nlanes; i++) {
            int32_t next = total[i] + labels[i];
            if (next > LT_TOTAL_MAX) next = LT_TOTAL_MAX;
            if (next < -LT_TOTAL_MAX) next = -LT_TOTAL_MAX;
            if (next != total[i]) changed++;
            total[i] = next;
        }
        if (stats != NULL) {
            stats->labels = ms.labels;
            stats->abstained = ms.abstained_sites;
            stats->mean_confidence = ms.mean_confidence;
            stats->rounds = (size_t)round + 1;
            stats->changed += changed;
        }
        if (changed == 0) break;
    }

    if (stats != NULL) {
        stats->sites = nlanes;
        stats->edges = ne;
        stats->energy_before = energy_first;
        stats->energy_after = energy_last;
        stats->shift_min = 0;
        stats->shift_max = 0;
        for (i = 0; i < nlanes; i++) {
            if (total[i] != 0) stats->moved++;
            if (total[i] < stats->shift_min) stats->shift_min = total[i];
            if (total[i] > stats->shift_max) stats->shift_max = total[i];
        }
    }
    *out_shift = total;
    return 0;
}

static int lt_check(int cond, const char *what, int *fails)
{
    fprintf(stderr, "[lane_turn_mrf selftest] %s -> %s\n",
            what, cond ? "ok" : "FAIL");
    if (!cond) (*fails)++;
    return cond;
}

int LaneTurn_selftest(void)
{
    int fails = 0;
    Arena_T arena = Arena_new();

    {
        LaneTurnEdge e[2];
        double sup[3] = { 100.0, 10.0, 10.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 1; e[0].target = 1; e[0].weight = 64.0;
        e[1].a = 1; e[1].b = 2; e[1].target = 1; e[1].weight = 64.0;
        lt_check(LaneTurn_solve(arena, e, 2, sup, 3, 8.0, &shift, &st) == 0,
                 "ladder solves", &fails);
        fprintf(stderr, "  ladder -> {%d,%d,%d} moved=%zu edges=%zu "
                        "E %.3f -> %.3f\n",
                shift ? shift[0] : -99, shift ? shift[1] : -99,
                shift ? shift[2] : -99, st.moved, st.edges,
                st.energy_before, st.energy_after);
        lt_check(shift && shift[0] == 0 && shift[1] == 1 && shift[2] == 2,
                 "ladder lands on 0,1,2", &fails);
        lt_check(st.moved == 2, "two lanes moved", &fails);
        lt_check(st.energy_after <= st.energy_before,
                 "energy did not increase", &fails);
    }

    {
        LaneTurnEdge e[1];
        double sup[2] = { 100.0, 1.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 1; e[0].target = 1; e[0].weight = 1.0;
        lt_check(LaneTurn_solve(arena, e, 1, sup, 2, 32.0, &shift, &st) == 0,
                 "weak-evidence case solves", &fails);
        lt_check(shift && shift[0] == 0 && shift[1] == 0,
                 "weak evidence cannot move a lane", &fails);
        e[0].weight = 64.0;
        lt_check(LaneTurn_solve(arena, e, 1, sup, 2, 32.0, &shift, &st) == 0,
                 "strong-evidence case solves", &fails);
        lt_check(shift && shift[0] == 0 && shift[1] == 1,
                 "strong evidence moves the lane", &fails);
    }

    {
        LaneTurnEdge e[1];
        double sup[2] = { 1.0, 100.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 1; e[0].target = 1; e[0].weight = 64.0;
        lt_check(LaneTurn_solve(arena, e, 1, sup, 2, 8.0, &shift, &st) == 0,
                 "anchored case solves", &fails);
        lt_check(shift && shift[1] == 0 && shift[0] == -1,
                 "the supported lane is pinned; the other moves", &fails);
    }

    {
        LaneTurnEdge e[2];
        double sup[2] = { 100.0, 1.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 1; e[0].target =  1; e[0].weight = 64.0;
        e[1].a = 0; e[1].b = 1; e[1].target = -1; e[1].weight = 64.0;
        lt_check(LaneTurn_solve(arena, e, 2, sup, 2, 8.0, &shift, &st) == 0,
                 "contradictory pair solves", &fails);
        lt_check(shift && shift[0] == 0 && shift[1] == 0,
                 "contradiction leaves the certificate alone", &fails);
    }

    {
        LaneTurnEdge e[3];
        double sup[2] = { 1.0, 1.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 0; e[0].target = 1; e[0].weight = 1e9;
        e[1].a = 0; e[1].b = 9; e[1].target = 1; e[1].weight = 1e9;
        e[2].a = 0; e[2].b = 1; e[2].target = 1; e[2].weight = 0.0;
        lt_check(LaneTurn_solve(arena, e, 3, sup, 2, 8.0, &shift, &st) == 0,
                 "malformed edges solve", &fails);
        lt_check(st.edges == 0, "all three malformed edges are dropped",
                 &fails);
        lt_check(shift && shift[0] == 0 && shift[1] == 0,
                 "no lane moves on no evidence", &fails);
        lt_check(LaneTurn_solve(arena, NULL, 0, NULL, 0, 8.0, &shift, &st) != 0,
                 "zero lanes is refused", &fails);
    }

    {
        LaneTurnEdge e[1];
        double sup[2] = { 100.0, 1.0 };
        int32_t *shift = NULL;
        LaneTurnStats st;
        e[0].a = 0; e[0].b = 1; e[0].target = 20; e[0].weight = 200.0;
        lt_check(LaneTurn_solve(arena, e, 1, sup, 2, 8.0, &shift, &st) == 0,
                 "out-of-window target solves", &fails);
        fprintf(stderr, "  window -> {%d,%d}\n",
                shift ? shift[0] : -99, shift ? shift[1] : -99);
        lt_check(shift && shift[1] == LT_TOTAL_MAX,
                 "the shift clamps to the total reach", &fails);
    }

    Arena_dispose(&arena);
    fprintf(stderr, "[lane_turn_mrf selftest] %s (%d failure%s)\n",
            fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails;
}
