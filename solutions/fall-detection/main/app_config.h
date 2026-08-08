#ifndef _FALL_DETECTION_APP_CONFIG_H_
#define _FALL_DETECTION_APP_CONFIG_H_

#include <string>
#include <utility>

#include "fall_detector.h"

namespace fall {

struct AppConfig {
    // Person/keypoint thresholds are applied by PoseDetector.
    float confidence = 0.40f;
    float keypoint_confidence = 0.50f;

    // State-machine thresholds. Keep these in the manifest so deployments can
    // tune camera angle/model behaviour without rebuilding the binary.
    FallConfig detector{};
};

bool load_app_config(const std::string& path, AppConfig& out);

class ConfigWatcher {
public:
    explicit ConfigWatcher(std::string path) : path_(std::move(path)) {}

    const AppConfig& loadInitial();
    bool poll();
    const AppConfig& config() const { return config_; }

private:
    static constexpr int kPollFrames = 30;
    std::string path_;
    AppConfig config_;
    long last_mtime_ = 0;
    int frames_since_stat_ = 0;
};

}  // namespace fall

#endif  // _FALL_DETECTION_APP_CONFIG_H_
