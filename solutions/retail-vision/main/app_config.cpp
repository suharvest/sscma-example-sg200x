#include "app_config.h"

#include <fstream>

// nlohmann/json single header, provided by the mongoose component this
// solution already links (same header the supervisor uses).
#include "json.hpp"

#include <sscma.h>

#define TAG "AppConfig"

namespace retail_vision {

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

// Apply a positive-number override, guarded by an (optional) inclusive range.
static bool read_pos_number(const json& cfg, const char* key, float lo, float hi, float& out) {
    if (!cfg.contains(key) || !cfg[key].is_number()) {
        return false;
    }
    float v = cfg[key].get<float>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s %.3f out of [%.2f,%.2f], ignored", key, v, lo, hi);
        return false;
    }
    out = v;
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
    out.has_confidence     = read_pos_number(cfg, "confidence",      0.05f, 0.95f, out.confidence);
    // dwell.*
    out.has_dwell_engaged  = read_pos_number(cfg, "dwell_engaged",   0.1f,  60.0f, out.dwell_engaged);
    out.has_dwell_assist   = read_pos_number(cfg, "dwell_assist",    1.0f, 600.0f, out.dwell_assist);
    out.has_dwell_speed    = read_pos_number(cfg, "dwell_speed",     0.5f, 200.0f, out.dwell_speed);
    // window.*
    out.has_window_duration = read_pos_number(cfg, "window_duration", 5.0f, 3600.0f, out.window_duration);
    out.has_person_height   = read_pos_number(cfg, "person_height",   0.5f,    3.0f, out.person_height);

    // zone.count_zone: [[x,y], ...] 3+ points
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

    // line.entry_line: {"a":[x,y], "b":[x,y], "direction":"ab_in"|"ab_out"}
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

} // namespace retail_vision
