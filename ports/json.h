#pragma once

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace herculex {

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    bool has(const std::string &key) const { return kind == Object && fields.count(key) != 0; }

    const Json &operator[](const std::string &key) const {
        auto it = fields.find(key);
        if (kind != Object || it == fields.end())
            throw std::runtime_error("json: missing key '" + key + "'");
        return it->second;
    }

    const Json &operator[](size_t i) const {
        if (kind != Array || i >= items.size())
            throw std::runtime_error("json: index out of range");
        return items[i];
    }

    size_t size() const { return kind == Array ? items.size() : fields.size(); }

    const std::string &str() const {
        if (kind != String)
            throw std::runtime_error("json: not a string");
        return text;
    }

    double num() const {
        if (kind != Number)
            throw std::runtime_error("json: not a number");
        return number;
    }

    bool truth() const {
        if (kind != Bool)
            throw std::runtime_error("json: not a boolean");
        return boolean;
    }

    static Json parse(const std::string &s) {
        size_t i = 0;
        Json value = parse_value(s, i);
        skip(s, i);
        if (i != s.size())
            throw std::runtime_error("json: trailing characters");
        return value;
    }

  private:
    static void skip(const std::string &s, size_t &i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t'))
            ++i;
    }

    static void expect(const std::string &s, size_t &i, char c) {
        skip(s, i);
        if (i >= s.size() || s[i] != c)
            throw std::runtime_error(std::string("json: expected '") + c + "'");
        ++i;
    }

    static std::string parse_string(const std::string &s, size_t &i) {
        expect(s, i, '"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size())
                break;
            char e = s[i++];
            switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                unsigned code = (unsigned)std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
                i += 4;
                if (code < 0x80)
                    out += (char)code;
                else if (code < 0x800) {
                    out += (char)(0xC0 | (code >> 6));
                    out += (char)(0x80 | (code & 0x3F));
                } else {
                    out += (char)(0xE0 | (code >> 12));
                    out += (char)(0x80 | ((code >> 6) & 0x3F));
                    out += (char)(0x80 | (code & 0x3F));
                }
                break;
            }
            default: out += e;
            }
        }
        if (i >= s.size())
            throw std::runtime_error("json: unterminated string");
        ++i;
        return out;
    }

    static Json parse_value(const std::string &s, size_t &i) {
        skip(s, i);
        if (i >= s.size())
            throw std::runtime_error("json: unexpected end");
        Json v;
        char c = s[i];
        if (c == '{') {
            v.kind = Object;
            ++i;
            skip(s, i);
            if (i < s.size() && s[i] == '}') {
                ++i;
                return v;
            }
            while (true) {
                std::string key = parse_string(s, i);
                expect(s, i, ':');
                v.fields[key] = parse_value(s, i);
                skip(s, i);
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                expect(s, i, '}');
                return v;
            }
        }
        if (c == '[') {
            v.kind = Array;
            ++i;
            skip(s, i);
            if (i < s.size() && s[i] == ']') {
                ++i;
                return v;
            }
            while (true) {
                v.items.push_back(parse_value(s, i));
                skip(s, i);
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                expect(s, i, ']');
                return v;
            }
        }
        if (c == '"') {
            v.kind = String;
            v.text = parse_string(s, i);
            return v;
        }
        if (s.compare(i, 4, "true") == 0) {
            v.kind = Bool;
            v.boolean = true;
            i += 4;
            return v;
        }
        if (s.compare(i, 5, "false") == 0) {
            v.kind = Bool;
            i += 5;
            return v;
        }
        if (s.compare(i, 4, "null") == 0) {
            i += 4;
            return v;
        }
        char *end = nullptr;
        v.kind = Number;
        v.number = std::strtod(s.c_str() + i, &end);
        if (end == s.c_str() + i)
            throw std::runtime_error("json: bad value");
        i = (size_t)(end - s.c_str());
        return v;
    }
};

}
