#ifndef _YOLO_APP_CONFIG_H_
#define _YOLO_APP_CONFIG_H_

// Loader for the supervisor-managed per-app config file:
//   /userdata/local/apps/yolo-detector.config.json
// written atomically by the supervisor's appMgr/setConfig endpoint after
// validating against the app manifest's config_schema.
//
// Contract: when the file is missing (or unreadable / malformed) the
// application behaves exactly as before this mechanism existed — every
// field below is flagged with a has_* / *_enabled boolean and only applied
// when actually present and well-formed.

#include <string>
#include <vector>

#include "geometry.h"

namespace yolo {

struct AppConfig {
    // detection.confidence (number)
    bool has_confidence = false;
    float confidence = 0.25f;

    // detection.tracking (boolean)
    bool has_tracking = false;
    bool tracking = true;

    // spatial.count_zone (zone): polygon, normalized [0,1] coords
    bool zone_enabled = false;
    std::vector<geom::Point> zone_points;

    // spatial.entry_line (line): segment a->b + direction, normalized coords
    bool line_enabled = false;
    geom::Point line_a { 0.0f, 0.0f };
    geom::Point line_b { 0.0f, 0.0f };
    bool line_ab_in = true; // direction "ab_in": left->right of a->b counts as "in"
};

// Returns true if the file existed and parsed as a JSON object (individual
// keys are still validated defensively and skipped when malformed).
bool load_app_config(const std::string& path, AppConfig& out);

} // namespace yolo

#endif // _YOLO_APP_CONFIG_H_
