#include "app_config.h"

#include <sys/stat.h>

#include <fstream>

// nlohmann/json single header, from the `json` component this solution links
// (the same header the supervisor validates the config with).
#include "json.hpp"

#include <sscma.h>

#include "exercise.h"

#define TAG "AppConfig"

namespace fitness {

using json = nlohmann::json;

namespace {

void read_int(const json& cfg, const char* key, int& out, int lo, int hi) {
    if (!cfg.contains(key) || !cfg[key].is_number()) return;
    const int v = cfg[key].get<int>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %d out of [%d,%d], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

void read_float(const json& cfg, const char* key, float& out, float lo, float hi) {
    if (!cfg.contains(key) || !cfg[key].is_number()) return;
    const float v = cfg[key].get<float>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %.3f out of [%.2f,%.2f], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

}  // namespace

bool load_app_config(const std::string& path, AppConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
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

    if (cfg.contains("mode") && cfg["mode"].is_string()) {
        const std::string mode = cfg["mode"].get<std::string>();
        if (Exercise::known(mode)) {
            out.mode = mode;
        } else {
            // The console validates against the manifest enum, so this can only
            // come from a hand-edited file. Keeping the previous mode is the
            // safe answer: an unrecognised name must not stop the counter.
            MA_LOGW(TAG, "mode '%s' is not a known exercise, keeping '%s'",
                    mode.c_str(), out.mode.c_str());
        }
    }

    read_int(cfg, "target_reps", out.target_reps, 1, 100);
    read_int(cfg, "target_sets", out.target_sets, 1, 20);
    read_int(cfg, "idle_reset_seconds", out.idle_reset_seconds, 0, 3600);
    read_float(cfg, "confidence", out.confidence, 0.05f, 0.95f);
    read_float(cfg, "keypoint_confidence", out.keypoint_confidence, 0.05f, 0.95f);

    return true;
}

const AppConfig& ConfigWatcher::loadInitial() {
    struct stat st;
    if (::stat(path_.c_str(), &st) == 0) {
        last_mtime_ = static_cast<long>(st.st_mtime);
    }
    load_app_config(path_, config_);
    return config_;
}

bool ConfigWatcher::poll() {
    if (++frames_since_stat_ < kPollFrames) {
        return false;
    }
    frames_since_stat_ = 0;

    struct stat st;
    if (::stat(path_.c_str(), &st) != 0) {
        return false;
    }
    const long mtime = static_cast<long>(st.st_mtime);
    if (mtime == last_mtime_) {
        return false;
    }
    last_mtime_ = mtime;

    AppConfig fresh = config_;  // absent keys keep the running value
    if (!load_app_config(path_, fresh)) {
        return false;
    }
    config_ = fresh;
    return true;
}

}  // namespace fitness
