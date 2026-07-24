#ifndef _RETAIL_APP_CONFIG_H_
#define _RETAIL_APP_CONFIG_H_

// Loader for the supervisor-managed per-app config file:
//   /userdata/local/apps/retail-vision.config.json
// written atomically by the supervisor's appMgr/setConfig endpoint after
// validating against the app manifest's config_schema.
//
// Contract: when the file is missing (or unreadable / malformed) the
// application behaves exactly as before this mechanism existed — every
// field below is flagged with a has_* / *_enabled boolean and only applied
// when actually present and well-formed. In particular, count_zone / entry_line
// only take effect when the operator has drawn them; otherwise the app keeps
// its original whole-frame analysis and appearance/disappearance entry counting.

#include <string>
#include <vector>

#include "geometry.h"

namespace retail_vision {

struct AppConfig {
    // detection.confidence (number)
    bool has_confidence = false;
    float confidence = 0.5f;

    // dwell.dwell_engaged (number, seconds)
    bool has_dwell_engaged = false;
    float dwell_engaged = 1.5f;

    // dwell.dwell_assist (number, seconds)
    bool has_dwell_assist = false;
    float dwell_assist = 20.0f;

    // dwell.dwell_speed (number, px/s stationary threshold)
    bool has_dwell_speed = false;
    float dwell_speed = 10.0f;

    // window.window_duration (number, seconds)
    bool has_window_duration = false;
    float window_duration = 60.0f;

    // window.person_height (number, meters)
    bool has_person_height = false;
    float person_height = 1.7f;

    // zone.count_zone (zone): polygon, normalized [0,1] coords
    bool zone_enabled = false;
    std::vector<geom::Point> zone_points;

    // line.entry_line (line): segment a->b + direction, normalized coords
    bool line_enabled = false;
    geom::Point line_a { 0.0f, 0.0f };
    geom::Point line_b { 0.0f, 0.0f };
    bool line_ab_in = true; // direction "ab_in": left->right of a->b counts as "in"
};

// Returns true if the file existed and parsed as a JSON object (individual
// keys are still validated defensively and skipped when malformed).
bool load_app_config(const std::string& path, AppConfig& out);

} // namespace retail_vision

#endif // _RETAIL_APP_CONFIG_H_
