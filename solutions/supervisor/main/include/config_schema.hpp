#ifndef CONFIG_SCHEMA_HPP
#define CONFIG_SCHEMA_HPP

// Pure config_schema validation helpers (no HTTP / filesystem dependencies so
// the logic is unit-testable on the host).
//
// Schema shape (manifest "config_schema"):
//   { "groups": [ { "key": ..., "items": [ <item>, ... ] }, ... ] }
// Item types:
//   number  : min / max (optional), step is UI-only
//   boolean
//   enum    : options[] (array of allowed values, strings)
//   string  : maxLength (optional, default 256)
//   zone    : value = [[x,y], ...], 3..maxPoints points, coords normalized 0..1
//   line    : value = {"a":[x,y], "b":[x,y], "direction":"ab_in"|"ab_out"},
//             coords normalized 0..1; direction required when "directional".

#include <string>

#include "json.hpp"

namespace config_schema {

using json = nlohmann::json;

// Flatten groups[].items[] into an object: key -> item. Malformed entries are
// skipped (a bad manifest schema must not make every value unvalidatable).
inline json collect_items(const json& schema)
{
    json items = json::object();
    if (!schema.is_object() || !schema.contains("groups") || !schema["groups"].is_array()) {
        return items;
    }
    for (const auto& group : schema["groups"]) {
        if (!group.is_object() || !group.contains("items") || !group["items"].is_array()) {
            continue;
        }
        for (const auto& item : group["items"]) {
            if (!item.is_object() || !item.contains("key") || !item["key"].is_string()) {
                continue;
            }
            items[item["key"].get<std::string>()] = item;
        }
    }
    return items;
}

// Defaults declared in the schema (only items that carry a "default").
inline json defaults_from_schema(const json& schema)
{
    json defaults = json::object();
    // NB: bind to a local first; iterating .items() on a temporary dangles.
    const json items = collect_items(schema);
    for (const auto& [key, item] : items.items()) {
        if (item.contains("default")) {
            defaults[key] = item["default"];
        }
    }
    return defaults;
}

inline bool is_norm_point(const json& p)
{
    if (!p.is_array() || p.size() != 2) {
        return false;
    }
    for (const auto& c : p) {
        if (!c.is_number()) {
            return false;
        }
        double v = c.get<double>();
        if (v < 0.0 || v > 1.0) {
            return false;
        }
    }
    return true;
}

inline bool validate_value(const json& item, const json& value, const std::string& key, std::string& err)
{
    const std::string type = item.value("type", "");

    if (type == "number") {
        if (!value.is_number()) {
            err = "'" + key + "' must be a number";
            return false;
        }
        double v = value.get<double>();
        if (item.contains("min") && item["min"].is_number() && v < item["min"].get<double>()) {
            err = "'" + key + "' is below min " + item["min"].dump();
            return false;
        }
        if (item.contains("max") && item["max"].is_number() && v > item["max"].get<double>()) {
            err = "'" + key + "' is above max " + item["max"].dump();
            return false;
        }
        return true;
    }

    if (type == "boolean") {
        if (!value.is_boolean()) {
            err = "'" + key + "' must be a boolean";
            return false;
        }
        return true;
    }

    if (type == "enum") {
        if (!item.contains("options") || !item["options"].is_array()) {
            err = "'" + key + "': schema enum has no options[]";
            return false;
        }
        for (const auto& opt : item["options"]) {
            if (opt == value) {
                return true;
            }
        }
        err = "'" + key + "' is not one of the allowed options";
        return false;
    }

    if (type == "string") {
        if (!value.is_string()) {
            err = "'" + key + "' must be a string";
            return false;
        }
        size_t max_len = 256;
        if (item.contains("maxLength") && item["maxLength"].is_number_unsigned()) {
            max_len = item["maxLength"].get<size_t>();
        }
        if (value.get<std::string>().size() > max_len) {
            err = "'" + key + "' exceeds maxLength " + std::to_string(max_len);
            return false;
        }
        return true;
    }

    if (type == "zone") {
        // null clears the zone (explicit "no zone configured")
        if (value.is_null()) {
            return true;
        }
        if (!value.is_array()) {
            err = "'" + key + "' must be an array of [x,y] points (or null)";
            return false;
        }
        size_t max_points = 8;
        if (item.contains("maxPoints") && item["maxPoints"].is_number_unsigned()) {
            max_points = item["maxPoints"].get<size_t>();
        }
        if (value.size() < 3 || value.size() > max_points) {
            err = "'" + key + "' needs 3.." + std::to_string(max_points) + " points";
            return false;
        }
        for (const auto& p : value) {
            if (!is_norm_point(p)) {
                err = "'" + key + "' points must be [x,y] with coords in 0..1";
                return false;
            }
        }
        return true;
    }

    if (type == "line") {
        // null clears the line
        if (value.is_null()) {
            return true;
        }
        if (!value.is_object() || !value.contains("a") || !value.contains("b")) {
            err = "'" + key + "' must be {\"a\":[x,y],\"b\":[x,y],...} (or null)";
            return false;
        }
        if (!is_norm_point(value["a"]) || !is_norm_point(value["b"])) {
            err = "'" + key + "' endpoints must be [x,y] with coords in 0..1";
            return false;
        }
        bool directional = item.value("directional", false);
        if (value.contains("direction")) {
            if (!value["direction"].is_string()) {
                err = "'" + key + "'.direction must be a string";
                return false;
            }
            std::string d = value["direction"].get<std::string>();
            if (d != "ab_in" && d != "ab_out") {
                err = "'" + key + "'.direction must be 'ab_in' or 'ab_out'";
                return false;
            }
        } else if (directional) {
            err = "'" + key + "'.direction is required";
            return false;
        }
        // reject unknown keys inside the line object
        for (const auto& [k, v] : value.items()) {
            (void)v;
            if (k != "a" && k != "b" && k != "direction") {
                err = "'" + key + "' has unknown field '" + k + "'";
                return false;
            }
        }
        return true;
    }

    err = "'" + key + "' has unsupported schema type '" + type + "'";
    return false;
}

// Validate a full values{} object against the schema. Unknown keys are
// rejected; each present value is validated per its item type. Keys absent
// from values are simply not set (the app falls back to its defaults).
inline bool validate_values(const json& schema, const json& values, std::string& err)
{
    if (!values.is_object()) {
        err = "values must be an object";
        return false;
    }
    json items = collect_items(schema);
    if (items.empty()) {
        err = "app has no configurable items";
        return false;
    }
    for (const auto& [key, value] : values.items()) {
        if (!items.contains(key)) {
            err = "unknown config key '" + key + "'";
            return false;
        }
        if (!validate_value(items[key], value, key, err)) {
            return false;
        }
    }
    return true;
}

} // namespace config_schema

#endif // CONFIG_SCHEMA_HPP
