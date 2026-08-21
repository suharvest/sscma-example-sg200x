#include "app_config.h"

#include <sys/stat.h>

#include <fstream>

#include "json.hpp"

#include <sscma.h>

#define TAG "FallConfig"

namespace fall {

using json = nlohmann::json;

namespace {

void read_int(const json& cfg, const char* key, int& out, int lo, int hi) {
    if (!cfg.contains(key) || !cfg[key].is_number_integer()) return;
    const int v = cfg[key].get<int>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %d out of [%d,%d], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

void read_bool(const json& cfg, const char* key, bool& out) {
    if (cfg.contains(key) && cfg[key].is_boolean()) out = cfg[key].get<bool>();
}

void read_float(const json& cfg, const char* key, float& out, float lo, float hi) {
    if (!cfg.contains(key) || !cfg[key].is_number()) return;
    const float v = cfg[key].get<float>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %.3f out of [%.3f,%.3f], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

}  // namespace

bool load_app_config(const std::string& path, AppConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

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

    read_float(cfg, "confidence", out.confidence, 0.05f, 0.95f);
    read_float(cfg, "keypoint_confidence", out.keypoint_confidence, 0.05f, 0.95f);
    read_bool(cfg, "temporal_confirmation_required", out.detector.temporal_confirmation_required);
    read_float(cfg, "hip_drop_speed_threshold", out.detector.hip_drop_speed_threshold, 0.0f, 5.0f);
    read_float(cfg, "hip_drop_distance_threshold", out.detector.hip_drop_distance_threshold, 0.0f, 1.0f);
    read_float(cfg, "motion_window_sec", out.detector.motion_window_sec, 0.0f, 5.0f);
    read_float(cfg, "torso_angle_threshold_deg", out.detector.torso_angle_threshold_deg, 1.0f, 89.0f);
    read_float(cfg, "bbox_aspect_ratio_threshold", out.detector.bbox_aspect_ratio_threshold, 1.0f, 10.0f);
    read_int(cfg, "min_suspected_features", out.detector.min_suspected_features, 1, 3);
    read_float(cfg, "confirmation_sec", out.detector.confirmation_sec, 0.0f, 30.0f);
    read_float(cfg, "suspected_timeout_sec", out.detector.suspected_timeout_sec, 0.1f, 60.0f);
    read_float(cfg, "occlusion_grace_sec", out.detector.occlusion_grace_sec, 0.0f, 5.0f);
    read_float(cfg, "recovery_torso_angle_deg", out.detector.recovery_torso_angle_deg, 1.0f, 89.0f);
    read_float(cfg, "recovery_aspect_ratio", out.detector.recovery_aspect_ratio, 1.0f, 10.0f);
    read_float(cfg, "recovery_window_sec", out.detector.recovery_window_sec, 0.0f, 60.0f);
    read_float(cfg, "cooldown_sec", out.detector.cooldown_sec, 0.0f, 120.0f);
    return true;
}

const AppConfig& ConfigWatcher::loadInitial() {
    struct stat st;
    if (::stat(path_.c_str(), &st) == 0) last_mtime_ = static_cast<long>(st.st_mtime);
    load_app_config(path_, config_);
    return config_;
}

bool ConfigWatcher::poll() {
    if (++frames_since_stat_ < kPollFrames) return false;
    frames_since_stat_ = 0;

    struct stat st;
    if (::stat(path_.c_str(), &st) != 0) return false;
    const long mtime = static_cast<long>(st.st_mtime);
    if (mtime == last_mtime_) return false;

    AppConfig fresh = config_;
    if (!load_app_config(path_, fresh)) return false;
    config_ = fresh;
    last_mtime_ = mtime;
    return true;
}

}  // namespace fall
