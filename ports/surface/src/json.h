#pragma once

#include "../../json.h"

namespace hx {
using Json = herculex::Json;

inline std::string quote(const std::string &s) {
    std::string q = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            q += '\\';
            q += (char)c;
        } else if (c < 0x20) {
            const char *hex = "0123456789abcdef";
            q += "\\u00";
            q += hex[c >> 4];
            q += hex[c & 15];
        } else {
            q += (char)c;
        }
    }
    return q + "\"";
}
}
