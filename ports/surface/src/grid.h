#pragma once

#include "predict.h"

#include <cstdint>
#include <string>

namespace hx {

int64_t write_fiesta_grid(const std::string &out, const uint8_t *surface, const Box &box,
    const Reader &read, const std::string &raw_source, const std::string &pred_dir,
    const std::string &net, int threshold);

}
