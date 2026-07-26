#include "app_config.h"

#include <fstream>

// nlohmann/json single header, provided by the json component this
// solution already links (same header the supervisor uses).
#include "json.hpp"

#include <sscma.h>

#define TAG "AppConfig"

namespace yolo {

using json = nlohmann::json;

static bool parse_norm_point(const json& p, geom::Point& out) {
    if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number()) {
        return false;
    }
    float x = p[0].get<float>();
    float y = p[1].get<float>();
    if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
        return false;
    }
    out = { x, y };
    return true;
}

bool load_app_config(const std::string& path, AppConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false; // no config file: keep default behavior
    }

    json cfg;
    try {
        f >> cfg;
    } catch (const std::exception& e) {
        MA_LOGW(TAG, "Bad config file %s: %s (ignored)", path.c_str(), e.what());
        return false;
    }
    if (!cfg.is_object()) {
        MA_LOGW(TAG, "Config file %s is not a JSON object (ignored)", path.c_str());
        return false;
    }

    // detection.confidence
    if (cfg.contains("confidence") && cfg["confidence"].is_number()) {
        float v = cfg["confidence"].get<float>();
        if (v > 0.0f && v < 1.0f) {
            out.has_confidence = true;
            out.confidence = v;
        } else {
            MA_LOGW(TAG, "confidence %.3f out of (0,1), ignored", v);
        }
    }

    // detection.tracking
    if (cfg.contains("tracking") && cfg["tracking"].is_boolean()) {
        out.has_tracking = true;
        out.tracking = cfg["tracking"].get<bool>();
    }

    // spatial.count_zone: [[x,y], ...] 3+ points
    if (cfg.contains("count_zone") && cfg["count_zone"].is_array()) {
        std::vector<geom::Point> pts;
        bool ok = true;
        for (const auto& p : cfg["count_zone"]) {
            geom::Point pt;
            if (!parse_norm_point(p, pt)) {
                ok = false;
                break;
            }
            pts.push_back(pt);
        }
        if (ok && pts.size() >= 3) {
            out.zone_enabled = true;
            out.zone_points = std::move(pts);
        } else {
            MA_LOGW(TAG, "count_zone malformed (need >=3 [x,y] points in 0..1), ignored");
        }
    }

    // spatial.entry_line: {"a":[x,y], "b":[x,y], "direction":"ab_in"|"ab_out"}
    if (cfg.contains("entry_line") && cfg["entry_line"].is_object()) {
        const auto& line = cfg["entry_line"];
        geom::Point a, b;
        if (line.contains("a") && line.contains("b") &&
            parse_norm_point(line["a"], a) && parse_norm_point(line["b"], b)) {
            std::string dir = line.value("direction", "ab_in");
            if (dir == "ab_in" || dir == "ab_out") {
                out.line_enabled = true;
                out.line_a = a;
                out.line_b = b;
                out.line_ab_in = (dir == "ab_in");
            } else {
                MA_LOGW(TAG, "entry_line.direction '%s' invalid, ignored", dir.c_str());
            }
        } else {
            MA_LOGW(TAG, "entry_line malformed (need a/b [x,y] in 0..1), ignored");
        }
    }

    return true;
}

} // namespace yolo
