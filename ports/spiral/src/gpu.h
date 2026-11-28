#pragma once

#include "model.h"

#include <string>

namespace sp::gpu {

bool init(std::string &message);
bool enabled();
bool points();
void set_points(bool on);
void upsample(const Model &m, const Field &low, Field &up);
void upsample_adjoint(const Model &m, const Field &grad_up, Field &grad_low);
void adamw(Field &p, const Field &g, Field &first, Field &second, double lr, double weight_decay,
    long step);
void zero(Field &a);
void flow(const Model &m, const Shared &sh, float *y, size_t n, bool inverse, float *inputs);
void flow_backward(const Model &m, const Shared &sh, const float *inputs, float *grad, size_t n,
    bool inverse, Field &acc);

}
