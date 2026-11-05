#ifndef _MICRODUCK_APP_CONFIG_H_
#define _MICRODUCK_APP_CONFIG_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace microduck {

struct AppConfig {
    float confidence = 0.60f;
    float keypoint_confidence = 0.35f;
    float gear_tooth_tolerance_deg = 10.0f;
    float bilateral_height_tolerance_norm = 0.040f;
    float spine_tilt_tolerance_deg = 12.0f;
    uint32_t exposure_us = 16666;
    std::vector<float> screen_roi = {0.0f, 0.0f, 1280.0f, 720.0f};
    std::string config_version = "20261105_v1";

    AppConfig createSnapshot() const {
        return *this;
    }
};

bool load_app_config(const std::string& path, AppConfig& out);

class ConfigWatcher {
public:
    explicit ConfigWatcher(std::string path) : path_(std::move(path)) {}

    const AppConfig& loadInitial();
    bool poll();
    AppConfig config() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

    AppConfig createSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_.createSnapshot();
    }

    void setConfig(const AppConfig& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = cfg;
    }

private:
    static constexpr int kPollFrames = 30;

    std::string path_;
    mutable std::mutex mutex_;
    AppConfig config_;
    long last_mtime_ = 0;
    int frames_since_stat_ = 0;
};

}  // namespace microduck

#endif  // _MICRODUCK_APP_CONFIG_H_
