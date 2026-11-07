#include "app_config.h"

#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "json.hpp"

#ifndef MA_LOGW
#define MA_LOGW(tag, fmt, ...) fprintf(stderr, "[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGI(tag, fmt, ...) fprintf(stdout, "[INFO][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGE(tag, fmt, ...) fprintf(stderr, "[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

#define TAG "AppConfig"

namespace microduck {

using json = nlohmann::json;

namespace {

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::unordered_map<std::string, std::string> parse_conf_file(const std::string& path) {
    std::unordered_map<std::string, std::string> kvs;
    std::ifstream infile(path);
    if (!infile.is_open()) {
        return kvs;
    }
    std::string line;
    while (std::getline(infile, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\''))) {
            v = v.substr(1, v.size() - 2);
        }
        kvs[k] = v;
    }
    return kvs;
}

std::string get_conf_or_env(const std::unordered_map<std::string, std::string>& conf_map,
                            const char* key, const char* env_key) {
    if (env_key) {
        const char* ev = std::getenv(env_key);
        if (ev && *ev) return std::string(ev);
    }
    auto it = conf_map.find(key);
    if (it != conf_map.end()) {
        return it->second;
    }
    return "";
}

void read_float(const json& cfg, const char* key, float& out, float lo, float hi) {
    if (!cfg.contains(key) || !cfg[key].is_number()) return;
    const float v = cfg[key].get<float>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %.4f out of [%.4f,%.4f], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

void read_uint32(const json& cfg, const char* key, uint32_t& out, uint32_t lo, uint32_t hi) {
    if (!cfg.contains(key) || !cfg[key].is_number_unsigned()) return;
    const uint32_t v = cfg[key].get<uint32_t>();
    if (v < lo || v > hi) {
        MA_LOGW(TAG, "%s = %u out of [%u,%u], ignored", key, v, lo, hi);
        return;
    }
    out = v;
}

void read_string(const json& cfg, const char* key, std::string& out) {
    if (!cfg.contains(key) || !cfg[key].is_string()) return;
    out = cfg[key].get<std::string>();
}

void read_screen_roi(const json& cfg, const char* key, std::vector<float>& out) {
    if (!cfg.contains(key) || !cfg[key].is_array()) return;
    const auto& arr = cfg[key];
    if (arr.size() == 4) {
        bool all_num = true;
        std::vector<float> tmp;
        for (const auto& item : arr) {
            if (!item.is_number()) { all_num = false; break; }
            tmp.push_back(item.get<float>());
        }
        if (all_num && tmp[2] > 0.0f && tmp[3] > 0.0f) {
            out = std::move(tmp);
        } else {
            MA_LOGW(TAG, "Invalid screen_roi array elements or dimensions");
        }
    }
}

}  // namespace

bool load_app_config(const std::string& path, AppConfig& out) {
    // 1. Parse system conf fallback: /etc/microduck-pose-qa.conf
    auto conf_map = parse_conf_file("/etc/microduck-pose-qa.conf");

    // 2. Try loading JSON file if exists
    std::ifstream f(path);
    if (f.is_open()) {
        json cfg;
        try {
            f >> cfg;
        } catch (const std::exception& e) {
            MA_LOGW(TAG, "Bad config JSON file %s: %s (preserving current config)", path.c_str(), e.what());
            return false;
        }

        if (!cfg.is_object()) {
            MA_LOGW(TAG, "Config file %s is not a JSON object (preserving current config)", path.c_str());
            return false;
        }

        read_float(cfg, "confidence", out.confidence, 0.05f, 0.95f);
        read_float(cfg, "keypoint_confidence", out.keypoint_confidence, 0.05f, 0.95f);
        read_float(cfg, "gear_tooth_tolerance_deg", out.gear_tooth_tolerance_deg, 1.0f, 180.0f);
        read_float(cfg, "bilateral_height_tolerance_norm", out.bilateral_height_tolerance_norm, 0.001f, 0.50f);
        read_float(cfg, "spine_tilt_tolerance_deg", out.spine_tilt_tolerance_deg, 1.0f, 45.0f);
        read_float(cfg, "nominal_stand_theta_l_deg", out.nominal_stand_theta_l_deg, -180.0f, 180.0f);
        read_float(cfg, "nominal_stand_theta_r_deg", out.nominal_stand_theta_r_deg, -180.0f, 180.0f);
        read_float(cfg, "nominal_stand_disparity_deg", out.nominal_stand_disparity_deg, 0.0f, 180.0f);
        read_float(cfg, "nominal_stand_spine_tilt_deg", out.nominal_stand_spine_tilt_deg, 0.0f, 180.0f);
        read_uint32(cfg, "exposure_us", out.exposure_us, 1000, 1000000);
        read_screen_roi(cfg, "screen_roi", out.screen_roi);
        read_screen_roi(cfg, "face_led_roi", out.face_led_roi);
        read_string(cfg, "config_version", out.config_version);

        // Also check nested schema (e.g. tolerences / detection groups if present)
        if (cfg.contains("detection") && cfg["detection"].is_object()) {
            read_float(cfg["detection"], "confidence", out.confidence, 0.05f, 0.95f);
            read_float(cfg["detection"], "keypoint_confidence", out.keypoint_confidence, 0.05f, 0.95f);
        }
        if (cfg.contains("tolerances") && cfg["tolerances"].is_object()) {
            read_float(cfg["tolerances"], "gear_tooth_tolerance_deg", out.gear_tooth_tolerance_deg, 1.0f, 180.0f);
            read_float(cfg["tolerances"], "bilateral_height_tolerance_norm", out.bilateral_height_tolerance_norm, 0.001f, 0.50f);
            read_float(cfg["tolerances"], "spine_tilt_tolerance_deg", out.spine_tilt_tolerance_deg, 1.0f, 45.0f);
            read_float(cfg["tolerances"], "nominal_stand_theta_l_deg", out.nominal_stand_theta_l_deg, -180.0f, 180.0f);
            read_float(cfg["tolerances"], "nominal_stand_theta_r_deg", out.nominal_stand_theta_r_deg, -180.0f, 180.0f);
            read_float(cfg["tolerances"], "nominal_stand_disparity_deg", out.nominal_stand_disparity_deg, 0.0f, 180.0f);
            read_float(cfg["tolerances"], "nominal_stand_spine_tilt_deg", out.nominal_stand_spine_tilt_deg, 0.0f, 180.0f);
        }
        if (cfg.contains("calibration") && cfg["calibration"].is_object()) {
            read_float(cfg["calibration"], "nominal_stand_theta_l_deg", out.nominal_stand_theta_l_deg, -180.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_theta_r_deg", out.nominal_stand_theta_r_deg, -180.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_disparity_deg", out.nominal_stand_disparity_deg, 0.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_spine_tilt_deg", out.nominal_stand_spine_tilt_deg, 0.0f, 180.0f);
        }
    } else {
        // JSON file missing, fallback to /etc/microduck-pose-qa.conf and environment
        auto parse_safe_float = [](const std::string& s, float lo, float hi, float& out, const char* name) {
            if (s.empty()) return;
            try {
                size_t idx = 0;
                float v = std::stof(s, &idx);
                if (idx != s.size()) {
                    MA_LOGW(TAG, "%s = '%s' has trailing characters, ignored", name, s.c_str());
                    return;
                }
                if (v >= lo && v <= hi) {
                    out = v;
                } else {
                    MA_LOGW(TAG, "%s = %s out of [%.4f,%.4f], ignored", name, s.c_str(), lo, hi);
                }
            } catch (const std::exception& e) {
                MA_LOGW(TAG, "Invalid float for %s ('%s'): %s, ignored", name, s.c_str(), e.what());
            }
        };

        auto parse_safe_uint32 = [](const std::string& s, uint32_t lo, uint32_t hi, uint32_t& out, const char* name) {
            if (s.empty()) return;
            try {
                size_t idx = 0;
                unsigned long v = std::stoul(s, &idx);
                if (idx != s.size()) {
                    MA_LOGW(TAG, "%s = '%s' has trailing characters, ignored", name, s.c_str());
                    return;
                }
                if (v >= lo && v <= hi) {
                    out = static_cast<uint32_t>(v);
                } else {
                    MA_LOGW(TAG, "%s = %s out of [%u,%u], ignored", name, s.c_str(), lo, hi);
                }
            } catch (const std::exception& e) {
                MA_LOGW(TAG, "Invalid uint32 for %s ('%s'): %s, ignored", name, s.c_str(), e.what());
            }
        };

        std::string s;
        s = get_conf_or_env(conf_map, "CONFIDENCE", "CONFIDENCE");
        parse_safe_float(s, 0.05f, 0.95f, out.confidence, "CONFIDENCE");

        s = get_conf_or_env(conf_map, "KEYPOINT_CONFIDENCE", "KEYPOINT_CONFIDENCE");
        parse_safe_float(s, 0.05f, 0.95f, out.keypoint_confidence, "KEYPOINT_CONFIDENCE");

        s = get_conf_or_env(conf_map, "GEAR_TOOTH_TOLERANCE_DEG", "GEAR_TOOTH_TOLERANCE_DEG");
        parse_safe_float(s, 1.0f, 180.0f, out.gear_tooth_tolerance_deg, "GEAR_TOOTH_TOLERANCE_DEG");

        s = get_conf_or_env(conf_map, "BILATERAL_HEIGHT_TOLERANCE_NORM", "BILATERAL_HEIGHT_TOLERANCE_NORM");
        parse_safe_float(s, 0.001f, 0.50f, out.bilateral_height_tolerance_norm, "BILATERAL_HEIGHT_TOLERANCE_NORM");

        s = get_conf_or_env(conf_map, "SPINE_TILT_TOLERANCE_DEG", "SPINE_TILT_TOLERANCE_DEG");
        parse_safe_float(s, 1.0f, 45.0f, out.spine_tilt_tolerance_deg, "SPINE_TILT_TOLERANCE_DEG");

        s = get_conf_or_env(conf_map, "NOMINAL_STAND_THETA_L_DEG", "NOMINAL_STAND_THETA_L_DEG");
        if (s.empty()) s = get_conf_or_env(conf_map, "nominal_stand_theta_l_deg", nullptr);
        parse_safe_float(s, -180.0f, 180.0f, out.nominal_stand_theta_l_deg, "NOMINAL_STAND_THETA_L_DEG");

        s = get_conf_or_env(conf_map, "NOMINAL_STAND_THETA_R_DEG", "NOMINAL_STAND_THETA_R_DEG");
        if (s.empty()) s = get_conf_or_env(conf_map, "nominal_stand_theta_r_deg", nullptr);
        parse_safe_float(s, -180.0f, 180.0f, out.nominal_stand_theta_r_deg, "NOMINAL_STAND_THETA_R_DEG");

        s = get_conf_or_env(conf_map, "NOMINAL_STAND_DISPARITY_DEG", "NOMINAL_STAND_DISPARITY_DEG");
        if (s.empty()) s = get_conf_or_env(conf_map, "nominal_stand_disparity_deg", nullptr);
        parse_safe_float(s, 0.0f, 180.0f, out.nominal_stand_disparity_deg, "NOMINAL_STAND_DISPARITY_DEG");

        s = get_conf_or_env(conf_map, "NOMINAL_STAND_SPINE_TILT_DEG", "NOMINAL_STAND_SPINE_TILT_DEG");
        if (s.empty()) s = get_conf_or_env(conf_map, "nominal_stand_spine_tilt_deg", nullptr);
        parse_safe_float(s, 0.0f, 180.0f, out.nominal_stand_spine_tilt_deg, "NOMINAL_STAND_SPINE_TILT_DEG");

        s = get_conf_or_env(conf_map, "EXPOSURE_US", "EXPOSURE_US");
        parse_safe_uint32(s, 1000, 1000000, out.exposure_us, "EXPOSURE_US");
    }

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

    AppConfig fresh;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fresh = config_;
    }
    if (!load_app_config(path_, fresh)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = fresh;
    }
    return true;
}

}  // namespace microduck
