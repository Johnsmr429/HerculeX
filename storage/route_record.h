#pragma once

#include <json-c/json.h>

static double route_seconds(json_object *record) {
    json_object *seconds = json_object_object_get(record, "seconds");
    if (!json_object_is_type(seconds, json_type_object))
        return json_object_get_double(seconds);
    double elapsed = 0;
    json_object_object_foreach(seconds, key, part) {
        (void)key;
        elapsed += json_object_get_double(part);
    }
    return elapsed;
}
