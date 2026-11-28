#pragma once

#include "volume.h"

#include <cstdint>
#include <string>

namespace hx {

int64_t fetch_missing(
    const std::string &folder, const Box &box, const std::string &bucket, int connections);
void create_store(const std::string &folder, const std::string &source, const std::string &bucket);
bool partial_store(const std::string &folder);

}
