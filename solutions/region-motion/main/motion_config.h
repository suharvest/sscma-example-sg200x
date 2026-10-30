#ifndef REGION_MOTION_CONFIG_H_
#define REGION_MOTION_CONFIG_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "json.hpp"

namespace region_motion {

struct Point {
    float x = 0.0f;
    float y = 0.0f;
};

struct MotionConfig {
    // Algorithmic parameters
    int pixel_threshold = 25;       // Sensitivity threshold (5..100)
    float min_motion_area = 0.015f; // Minimum ratio of motion pixels in ROI (0.001..0.5)
    int debounce_frames = 3;        // Consecutive frames before state change (1..10)
    std::vector<Point> roi_zone;    // Normalized polygon points (0..1)

    // Stream and video settings
    int inference_width = 640;
    int inference_height = 360;
    int inference_fps = 10;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Services
    bool enable_rtsp = true;
    bool enable_debug = true;
    int debug_port = 8001;

    bool enable_mqtt = true;
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/region-motion/results";

    bool verbose = false;

    static MotionConfig load(const std::string& path = "/userdata/local/apps/region-motion.config.json") {
        MotionConfig cfg;
        std::ifstream f(path);
        if (!f.is_open()) {
            return cfg;
        }

        try {
            nlohmann::json j;
            f >> j;
            if (!j.is_object()) {
                return cfg;
            }

            if (j.contains("pixel_threshold") && j["pixel_threshold"].is_number()) {
                int pt = j["pixel_threshold"].get<int>();
                if (pt >= 5 && pt <= 100) {
                    cfg.pixel_threshold = pt;
                }
            }

            if (j.contains("min_motion_area") && j["min_motion_area"].is_number()) {
                float ma = j["min_motion_area"].get<float>();
                if (ma >= 0.001f && ma <= 0.5f) {
                    cfg.min_motion_area = ma;
                }
            }

            if (j.contains("debounce_frames") && j["debounce_frames"].is_number()) {
                int df = j["debounce_frames"].get<int>();
                if (df >= 1 && df <= 10) {
                    cfg.debounce_frames = df;
                }
            }

            if (j.contains("roi_zone") && j["roi_zone"].is_array()) {
                std::vector<Point> pts;
                for (const auto& elem : j["roi_zone"]) {
                    if (elem.is_array() && elem.size() >= 2 && elem[0].is_number() && elem[1].is_number()) {
                        float x = elem[0].get<float>();
                        float y = elem[1].get<float>();
                        x = std::clamp(x, 0.0f, 1.0f);
                        y = std::clamp(y, 0.0f, 1.0f);
                        pts.push_back({x, y});
                    }
                }
                if (pts.size() >= 3) {
                    cfg.roi_zone = std::move(pts);
                }
            }
        } catch (...) {
            // Safe fallback on parse error: keep defaults
        }
        return cfg;
    }
};

}  // namespace region_motion

#endif  // REGION_MOTION_CONFIG_H_
