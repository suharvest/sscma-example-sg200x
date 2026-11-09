#include "app_config.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <cerrno>
#include <cstring>

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

void read_face_led_layout(const json& cfg, std::vector<FaceLedSlotPos>& out) {
    // {"face_led_layout": {"rgb": [u, v], "cam": [u, v], ...}} in lens-ring coordinates
    if (!cfg.contains("face_led_layout") || !cfg["face_led_layout"].is_object()) return;
    std::vector<FaceLedSlotPos> tmp;
    for (auto it = cfg["face_led_layout"].begin(); it != cfg["face_led_layout"].end(); ++it) {
        const auto& v = it.value();
        if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number()) {
            MA_LOGW(TAG, "face_led_layout.%s must be [x, y], layout ignored", it.key().c_str());
            return;
        }
        const float x = v[0].get<float>(), y = v[1].get<float>();
        if (std::abs(x) > 10.0f || std::abs(y) > 10.0f) {
            MA_LOGW(TAG, "face_led_layout.%s outside +-10 ring radii, layout ignored", it.key().c_str());
            return;
        }
        tmp.push_back({it.key(), x, y});
    }
    if (!tmp.empty()) out = std::move(tmp);
}

// Nominal stand keys: always read from the conf file (calibrate_stand writes them there).
void read_conf_nominal(const std::unordered_map<std::string, std::string>& conf_map, AppConfig& out) {
    struct Key { const char* name; float* dst; float lo; float hi; };
    const Key keys[] = {
        {"NOMINAL_STAND_THETA_L_DEG", &out.nominal_stand_theta_l_deg, -180.0f, 180.0f},
        {"NOMINAL_STAND_THETA_R_DEG", &out.nominal_stand_theta_r_deg, -180.0f, 180.0f},
        {"NOMINAL_STAND_DISPARITY_DEG", &out.nominal_stand_disparity_deg, 0.0f, 180.0f},
        {"NOMINAL_STAND_SPINE_TILT_DEG", &out.nominal_stand_spine_tilt_deg, 0.0f, 180.0f},
        {"NOMINAL_STAND_KNEE_DIFF_NORM", &out.nominal_stand_knee_diff_norm, -1.0f, 1.0f},
    };
    for (const auto& k : keys) {
        auto it = conf_map.find(k.name);
        if (it == conf_map.end() || it->second.empty()) continue;
        try {
            size_t idx = 0;
            const float v = std::stof(it->second, &idx);
            if (idx != it->second.size() || v < k.lo || v > k.hi) {
                MA_LOGW(TAG, "%s = '%s' invalid or out of [%.1f,%.1f], ignored", k.name, it->second.c_str(), k.lo, k.hi);
                continue;
            }
            *k.dst = v;
        } catch (const std::exception&) {
            MA_LOGW(TAG, "%s = '%s' is not a number, ignored", k.name, it->second.c_str());
        }
    }
}

long file_mtime(const std::string& path) {
    struct stat st;
    return (::stat(path.c_str(), &st) == 0) ? static_cast<long>(st.st_mtime) : 0;
}

}  // namespace

std::vector<FaceLedSlotPos> default_face_led_layout() {
    return {
        {"rgb", -0.2710f, 1.3212f},
        {"cam", -1.3030f, -0.6279f},
        {"gr2", 1.6446f, -0.5057f},
        {"gb", 1.5127f, 0.3584f},
        {"gr1", -2.1333f, 0.4869f},
        {"green2", -2.0369f, -0.1605f},
        {"green1", -1.9708f, -0.8598f},
    };
}

std::string format_face_led_layout(const std::vector<FaceLedSlotPos>& layout) {
    std::string out;
    char buf[96];
    for (const auto& s : layout) {
        std::snprintf(buf, sizeof(buf), "%s%s:%.4f,%.4f", out.empty() ? "" : ";", s.place.c_str(), s.x, s.y);
        out += buf;
    }
    return out;
}

bool parse_face_led_layout(const std::string& text, std::vector<FaceLedSlotPos>& out) {
    std::vector<FaceLedSlotPos> tmp;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ';')) {
        item = trim(item);
        if (item.empty()) continue;
        const size_t c = item.find(':'), k = item.find(',');
        if (c == std::string::npos || k == std::string::npos || k < c) return false;
        try {
            FaceLedSlotPos p;
            p.place = trim(item.substr(0, c));
            p.x = std::stof(item.substr(c + 1, k - c - 1));
            p.y = std::stof(item.substr(k + 1));
            if (p.place.empty() || std::abs(p.x) > 10.0f || std::abs(p.y) > 10.0f) return false;
            tmp.push_back(p);
        } catch (const std::exception&) {
            return false;
        }
    }
    if (tmp.size() < 3) return false;
    out = std::move(tmp);
    return true;
}

static bool rewrite_conf_keys(const std::string& conf_path,
                              const std::vector<std::pair<std::string, std::string>>& kv,
                              const char* header_comment, std::string* err) {
    std::vector<bool> written(kv.size(), false);
    std::vector<std::string> lines;
    {
        std::ifstream in(conf_path);
        std::string line;
        while (std::getline(in, line)) {
            const std::string t = trim(line);
            for (size_t i = 0; i < kv.size(); ++i) {
                const std::string& key = kv[i].first;
                if (t.compare(0, key.size(), key) == 0) {
                    const std::string rest = trim(t.substr(key.size()));
                    if (!rest.empty() && rest[0] == '=') {
                        line = key + "=" + kv[i].second;
                        written[i] = true;
                    }
                }
            }
            lines.push_back(line);
        }
    }
    bool header = false;
    for (size_t i = 0; i < kv.size(); ++i) {
        if (written[i]) continue;
        if (!header) {
            lines.emplace_back("");
            lines.emplace_back(header_comment);
            header = true;
        }
        lines.push_back(kv[i].first + "=" + kv[i].second);
    }
    const std::string tmp = conf_path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) {
        if (err) *err = "cannot open " + tmp + ": " + std::strerror(errno);
        return false;
    }
    for (const auto& l : lines) {
        std::fputs(l.c_str(), f);
        std::fputc('\n', f);
    }
    std::fflush(f);
    ::fsync(fileno(f));
    std::fclose(f);
    if (std::rename(tmp.c_str(), conf_path.c_str()) != 0) {
        if (err) *err = "rename to " + conf_path + " failed: " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool save_face_led_layout(const std::string& conf_path, const AppConfig& cfg, std::string* err) {
    return rewrite_conf_keys(conf_path, {{"FACE_LED_LAYOUT", "\"" + format_face_led_layout(cfg.face_led_layout) + "\""}},
                             "# Face LED layout in lens-ring coordinates (written by set_face_led_layout)", err);
}

bool save_stand_nominal(const std::string& conf_path, const AppConfig& cfg, std::string* err) {
    char buf[5][64];
    std::snprintf(buf[0], sizeof(buf[0]), "%.3f", cfg.nominal_stand_theta_l_deg);
    std::snprintf(buf[1], sizeof(buf[1]), "%.3f", cfg.nominal_stand_theta_r_deg);
    std::snprintf(buf[2], sizeof(buf[2]), "%.3f", cfg.nominal_stand_disparity_deg);
    std::snprintf(buf[3], sizeof(buf[3]), "%.3f", cfg.nominal_stand_spine_tilt_deg);
    std::snprintf(buf[4], sizeof(buf[4]), "%.4f", cfg.nominal_stand_knee_diff_norm);
    return rewrite_conf_keys(conf_path, {
        {"NOMINAL_STAND_THETA_L_DEG", buf[0]},
        {"NOMINAL_STAND_THETA_R_DEG", buf[1]},
        {"NOMINAL_STAND_DISPARITY_DEG", buf[2]},
        {"NOMINAL_STAND_SPINE_TILT_DEG", buf[3]},
        {"NOMINAL_STAND_KNEE_DIFF_NORM", buf[4]},
    }, "# Stand-pose standard (camera specific; written by calibrate_stand)", err);
}

bool load_app_config(const std::string& path, AppConfig& out, const std::string& conf_path) {
    // 1. Parse system conf: /etc/microduck-pose-qa.conf
    auto conf_map = parse_conf_file(conf_path);

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
        read_float(cfg, "nominal_stand_knee_diff_norm", out.nominal_stand_knee_diff_norm, -1.0f, 1.0f);
        read_uint32(cfg, "exposure_us", out.exposure_us, 1000, 1000000);
        read_screen_roi(cfg, "screen_roi", out.screen_roi);
        read_screen_roi(cfg, "face_led_roi", out.face_led_roi);
        read_string(cfg, "config_version", out.config_version);
        read_face_led_layout(cfg, out.face_led_layout);

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
            read_float(cfg["tolerances"], "nominal_stand_knee_diff_norm", out.nominal_stand_knee_diff_norm, -1.0f, 1.0f);
        }
        if (cfg.contains("calibration") && cfg["calibration"].is_object()) {
            read_float(cfg["calibration"], "nominal_stand_theta_l_deg", out.nominal_stand_theta_l_deg, -180.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_theta_r_deg", out.nominal_stand_theta_r_deg, -180.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_disparity_deg", out.nominal_stand_disparity_deg, 0.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_spine_tilt_deg", out.nominal_stand_spine_tilt_deg, 0.0f, 180.0f);
            read_float(cfg["calibration"], "nominal_stand_knee_diff_norm", out.nominal_stand_knee_diff_norm, -1.0f, 1.0f);
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

        s = get_conf_or_env(conf_map, "EXPOSURE_US", "EXPOSURE_US");
        parse_safe_uint32(s, 1000, 1000000, out.exposure_us, "EXPOSURE_US");
    }

    read_conf_nominal(conf_map, out);
    auto lay = conf_map.find("FACE_LED_LAYOUT");
    if (lay != conf_map.end() && !lay->second.empty() && !parse_face_led_layout(lay->second, out.face_led_layout)) {
        MA_LOGW(TAG, "FACE_LED_LAYOUT = '%s' invalid, ignored", lay->second.c_str());
    }
    return true;
}

const AppConfig& ConfigWatcher::loadInitial() {
    last_mtime_ = file_mtime(path_);
    last_conf_mtime_ = file_mtime(conf_path_);
    load_app_config(path_, config_, conf_path_);
    return config_;
}

bool ConfigWatcher::poll() {
    if (++frames_since_stat_ < kPollFrames) {
        return false;
    }
    frames_since_stat_ = 0;

    const long mtime = file_mtime(path_);
    const long conf_mtime = file_mtime(conf_path_);
    if (mtime == last_mtime_ && conf_mtime == last_conf_mtime_) {
        return false;
    }
    last_mtime_ = mtime;
    last_conf_mtime_ = conf_mtime;

    AppConfig fresh;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fresh = config_;
    }
    if (!load_app_config(path_, fresh, conf_path_)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = fresh;
    }
    return true;
}

}  // namespace microduck
