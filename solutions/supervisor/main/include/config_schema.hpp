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

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

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

// --- Geometry validation (normalized 0..1 coords) --------------------------
// Shared thresholds: the SpatialEditor frontend enforces the SAME numbers, so
// a shape accepted in the drawing UI is accepted here and vice versa. Keep the
// two in sync when changing either.
static constexpr double GEOM_MIN_LINE_LEN = 0.02;   // min line length (~2% of frame)
static constexpr double GEOM_MIN_ZONE_AREA = 0.005; // min polygon area (normalized)
static constexpr double GEOM_EPS = 1e-9;

inline double geom_point_dist(const json& a, const json& b)
{
    double dx = a[0].get<double>() - b[0].get<double>();
    double dy = a[1].get<double>() - b[1].get<double>();
    return std::sqrt(dx * dx + dy * dy);
}

// Drop consecutive duplicate points, and a closing point that equals the first
// (the polygon auto-closes). Returns the cleaned vertex list.
inline std::vector<std::pair<double, double>> geom_dedup_points(const json& pts)
{
    std::vector<std::pair<double, double>> out;
    for (const auto& p : pts) {
        double x = p[0].get<double>();
        double y = p[1].get<double>();
        if (!out.empty() && std::fabs(out.back().first - x) < GEOM_EPS && std::fabs(out.back().second - y) < GEOM_EPS) {
            continue;
        }
        out.emplace_back(x, y);
    }
    while (out.size() > 1 && std::fabs(out.front().first - out.back().first) < GEOM_EPS && std::fabs(out.front().second - out.back().second) < GEOM_EPS) {
        out.pop_back();
    }
    return out;
}

inline double geom_polygon_area(const std::vector<std::pair<double, double>>& p)
{
    double a = 0.0;
    size_t n = p.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        a += p[i].first * p[j].second - p[j].first * p[i].second;
    }
    return std::fabs(a) * 0.5;
}

inline double geom_cross(double ox, double oy, double ax, double ay, double bx, double by)
{
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

// Assuming collinear, is q on segment p-r?
inline bool geom_on_seg(double px, double py, double qx, double qy, double rx, double ry)
{
    return qx >= std::min(px, rx) - GEOM_EPS && qx <= std::max(px, rx) + GEOM_EPS
        && qy >= std::min(py, ry) - GEOM_EPS && qy <= std::max(py, ry) + GEOM_EPS;
}

inline bool geom_segments_intersect(double p1x, double p1y, double p2x, double p2y,
    double p3x, double p3y, double p4x, double p4y)
{
    double d1 = geom_cross(p3x, p3y, p4x, p4y, p1x, p1y);
    double d2 = geom_cross(p3x, p3y, p4x, p4y, p2x, p2y);
    double d3 = geom_cross(p1x, p1y, p2x, p2y, p3x, p3y);
    double d4 = geom_cross(p1x, p1y, p2x, p2y, p4x, p4y);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }
    if (std::fabs(d1) < GEOM_EPS && geom_on_seg(p3x, p3y, p1x, p1y, p4x, p4y)) return true;
    if (std::fabs(d2) < GEOM_EPS && geom_on_seg(p3x, p3y, p2x, p2y, p4x, p4y)) return true;
    if (std::fabs(d3) < GEOM_EPS && geom_on_seg(p1x, p1y, p3x, p3y, p2x, p2y)) return true;
    if (std::fabs(d4) < GEOM_EPS && geom_on_seg(p1x, p1y, p4x, p4y, p2x, p2y)) return true;
    return false;
}

// A simple polygon self-intersects if any pair of non-adjacent edges cross.
// n <= maxPoints (8) so the O(n^2) sweep is trivially cheap.
inline bool geom_polygon_self_intersects(const std::vector<std::pair<double, double>>& p)
{
    size_t n = p.size();
    if (n < 4) return false; // a triangle cannot self-intersect
    for (size_t i = 0; i < n; ++i) {
        size_t i2 = (i + 1) % n;
        for (size_t j = i + 1; j < n; ++j) {
            size_t j2 = (j + 1) % n;
            // skip the same edge and edges sharing an endpoint (adjacent)
            if (i == j || i2 == j || j2 == i) {
                continue;
            }
            if (geom_segments_intersect(p[i].first, p[i].second, p[i2].first, p[i2].second,
                    p[j].first, p[j].second, p[j2].first, p[j2].second)) {
                return true;
            }
        }
    }
    return false;
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
        // Degenerate-geometry rejection (matches SpatialEditor thresholds):
        // drop duplicate/closing points, then require a real, simple polygon.
        {
            std::vector<std::pair<double, double>> pts = geom_dedup_points(value);
            if (pts.size() < 3) {
                err = "'" + key + "' needs 3 distinct points (duplicates removed)";
                return false;
            }
            if (geom_polygon_area(pts) < GEOM_MIN_ZONE_AREA) {
                err = "'" + key + "' zone area is too small (min " + std::to_string(GEOM_MIN_ZONE_AREA) + ")";
                return false;
            }
            if (geom_polygon_self_intersects(pts)) {
                err = "'" + key + "' zone is self-intersecting";
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
        // Reject a==b / near-zero-length lines (matches SpatialEditor threshold).
        if (geom_point_dist(value["a"], value["b"]) < GEOM_MIN_LINE_LEN) {
            err = "'" + key + "' line is too short (min length " + std::to_string(GEOM_MIN_LINE_LEN) + ")";
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
