#ifndef _MICRODUCK_APP_CONFIG_H_
#define _MICRODUCK_APP_CONFIG_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace microduck {

// System configuration file. Holds the stand-pose standard (NOMINAL_STAND_*), which is
// specific to one camera placement and is rewritten by the calibrate_stand control method.
constexpr const char* kDefaultConfPath = "/etc/microduck-pose-qa.conf";

// One face-board LED place relative to the face: (u, v) = (p - c) / r, with c, r the centre and
// mid-radius of the lens ring, image axes. Independent of where the camera stands.
// "place" is the middle field of the firmware LED name (face:<place>:<colour>).
struct FaceLedSlotPos {
    std::string place;
    float x = 0.0f;  // u
    float y = 0.0f;  // v
};

// Default: calibrated (lit one LED per place, centroid in ring coordinates) on the simulator's
// station camera with sim/station.json's provisional LED offsets. Recalibrate per product with
// the golden sample (orchestrator --calibrate-face, then set_face_led_layout) or import from CAD.
std::vector<FaceLedSlotPos> default_face_led_layout();
std::string format_face_led_layout(const std::vector<FaceLedSlotPos>& layout);
bool parse_face_led_layout(const std::string& text, std::vector<FaceLedSlotPos>& out);

struct AppConfig {
    float confidence = 0.60f;
    float keypoint_confidence = 0.35f;
    float gear_tooth_tolerance_deg = 10.0f;
    float bilateral_height_tolerance_norm = 0.040f;
    float spine_tilt_tolerance_deg = 12.0f;
    float nominal_stand_theta_l_deg = -20.40f;
    float nominal_stand_theta_r_deg = -55.59f;
    float nominal_stand_disparity_deg = 35.19f;
    float nominal_stand_spine_tilt_deg = 14.82f;
    // Signed (left knee y - right knee y), normalised; the view-dependent baseline of the knee-height check.
    float nominal_stand_knee_diff_norm = 0.0f;
    uint32_t exposure_us = 16666;
    std::vector<float> screen_roi = {0.0f, 0.0f, 1280.0f, 720.0f};
    std::vector<float> face_led_roi = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<FaceLedSlotPos> face_led_layout = default_face_led_layout();  // ring coordinates
    std::string config_version = "20261105_v1";

    AppConfig createSnapshot() const {
        return *this;
    }
};

// Load order: compiled defaults -> JSON app config (path) -> conf_path. The NOMINAL_STAND_*
// keys are always taken from conf_path when present there, so a calibration written by
// save_stand_nominal() wins over older values in the JSON file.
bool load_app_config(const std::string& path, AppConfig& out,
                     const std::string& conf_path = kDefaultConfPath);

// Rewrite the five NOMINAL_STAND_* keys in conf_path (other lines are kept; missing keys are
// appended). Written to a temporary file and renamed, so a reader never sees a partial file.
bool save_stand_nominal(const std::string& conf_path, const AppConfig& cfg, std::string* err = nullptr);

// Rewrite FACE_LED_LAYOUT (relative layout, "place:u,v;...") in conf_path.
bool save_face_led_layout(const std::string& conf_path, const AppConfig& cfg, std::string* err = nullptr);

class ConfigWatcher {
public:
    explicit ConfigWatcher(std::string path, std::string conf_path = kDefaultConfPath)
        : path_(std::move(path)), conf_path_(std::move(conf_path)) {}

    const std::string& confPath() const { return conf_path_; }

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
    std::string conf_path_;
    mutable std::mutex mutex_;
    AppConfig config_;
    long last_mtime_ = 0;
    long last_conf_mtime_ = 0;
    int frames_since_stat_ = 0;
};

}  // namespace microduck

#endif  // _MICRODUCK_APP_CONFIG_H_
