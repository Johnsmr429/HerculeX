#include <stdio.h>
#include "../level/layout/packing.c"

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "layout golden: %s\n", message);
        exit(1);
    }
}

int main(void) {
    uint8_t mask[400];
    memset(mask, 1, sizeof(mask));
    HxLayoutPiece pieces[4] = {0};
    size_t rank[] = {0, 1, 2, 3}, order[4];
    for (int i = 0; i < 4; i++) {
        pieces[i] = (HxLayoutPiece){.mask = mask,
            .rows = 20,
            .columns = 20,
            .row_stride = 20,
            .column_stride = 1,
            .area = 400. - i};
    }
    const double step = 99. * .2;
    HxLayoutJoin joints[] = {
        {0, 1, 20, 1, step, 0}, {1, 2, 20, 1, step, 0}, {2, 3, 20, 1, -step, 0}};
    int groups = hx_layout_mosaics(pieces, 4, rank, joints, 3, order);
    require(groups == 2 && pieces[1].x == step && pieces[2].x == 2 * step &&
            pieces[0].group == pieces[1].group && pieces[0].group == pieces[2].group &&
            pieces[0].group != pieces[3].group,
        "a joint cannot fold the mosaic onto itself");

    const double x[] = {-7, -3, -7}, area[] = {12, 8, 5};
    for (int i = 0; i < 3; i++) {
        pieces[i] = (HxLayoutPiece){.mask = mask,
            .rows = 2,
            .columns = 4,
            .row_stride = 4,
            .column_stride = 1,
            .group = i == 2,
            .area = area[i],
            .group_area = i == 2 ? 5 : 20,
            .x = x[i],
            .y = 3};
    }
    size_t kept;
    require(hx_pack_layout(pieces, 3, 60, &kept) >= 0 && pieces[0].x == 5 && pieces[1].x == 9 &&
            pieces[2].x == 0 && pieces[0].y == 3 && pieces[1].y == 3 && pieces[2].y == 3 &&
            pieces[0].from_joint == 5 && pieces[1].from_joint == 5 && pieces[2].from_joint == 0,
        "packing keeps offsets relative to the shifted sheet origin");

    int turns[4];
    joints[0].weight = 50;
    joints[1].weight = 30;
    joints[1].sign = -1;
    require(hx_layout_turns((double[]){100, 90, 5, 7}, 4, rank, joints, 3, turns) &&
            turns[0] == 1 && turns[1] == 1 && turns[2] == -1 && turns[3] == -1,
        "pieces connected to the same sheet take consistent faces");
    joints[0].sign = -1;
    require(hx_layout_turns((double[]){3, 9}, 2, rank, joints, 1, turns) && turns[0] == -1 &&
            turns[1] == 1,
        "the larger piece keeps its face");
    return 0;
}
