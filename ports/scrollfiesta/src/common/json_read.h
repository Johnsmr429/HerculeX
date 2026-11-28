#ifndef JSON_READ_INCLUDED
#define JSON_READ_INCLUDED

#include <stddef.h>
#include "arena.h"

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

const JsonValue *Json_parse(Arena_T arena, const char *text, const char **err);

const JsonValue *Json_parse_file(Arena_T arena, const char *path,
                                 const char **err);

JsonType Json_type(const JsonValue *v);

size_t           Json_array_len(const JsonValue *v);
const JsonValue *Json_array_get(const JsonValue *v, size_t i);

const JsonValue *Json_object_get(const JsonValue *v, const char *key);
size_t           Json_object_len(const JsonValue *v);
const char      *Json_object_key(const JsonValue *v, size_t i);
const JsonValue *Json_object_value_at(const JsonValue *v, size_t i);

double      Json_as_double(const JsonValue *v, double def);
long        Json_as_long(const JsonValue *v, long def);
int         Json_as_bool(const JsonValue *v, int def);
const char *Json_as_string(const JsonValue *v);

double Json_member_double(const JsonValue *obj, const char *key, double def);
long   Json_member_long(const JsonValue *obj, const char *key, long def);

int Json_read_selftest(void);

#endif
