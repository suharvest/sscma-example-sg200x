#include "inference_control.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "json.hpp"

namespace aruco {

static uint64_t get_unix_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

InferenceControl::InferenceControl(int active_fps, bool enabled, OnChange on_change)
    : active_fps_(active_fps), on_change_(std::move(on_change)), continuous_enabled_(enabled), active_triggers_(0) {
    if (on_change_) {
        const bool eff = effective_enabled_locked();
        on_change_(eff, eff ? active_fps_ : kIdleFps);
    }
}

bool InferenceControl::continuous_enabled() const noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    return continuous_enabled_;
}

bool InferenceControl::effective_enabled() const noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    return effective_enabled_locked();
}

int InferenceControl::current_fps() const noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    return effective_enabled_locked() ? active_fps_ : kIdleFps;
}

bool InferenceControl::set_enabled(bool on) {
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (continuous_enabled_ != on) {
            const bool old_eff = effective_enabled_locked();
            continuous_enabled_ = on;
            const bool new_eff = effective_enabled_locked();
            changed = true;
            if (old_eff != new_eff && on_change_) {
                on_change_(new_eff, new_eff ? active_fps_ : kIdleFps);
            }
        }
    }
    if (changed) {
        cv_.notify_all();
    }
    return changed;
}

void InferenceControl::trigger_begin() {
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const bool old_eff = effective_enabled_locked();
        ++active_triggers_;
        const bool new_eff = effective_enabled_locked();
        if (old_eff != new_eff) {
            changed = true;
            if (on_change_) {
                on_change_(new_eff, new_eff ? active_fps_ : kIdleFps);
            }
        }
    }
    if (changed) {
        cv_.notify_all();
    }
}

void InferenceControl::trigger_end() {
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (active_triggers_ > 0) {
            const bool old_eff = effective_enabled_locked();
            --active_triggers_;
            const bool new_eff = effective_enabled_locked();
            if (old_eff != new_eff) {
                changed = true;
                if (on_change_) {
                    on_change_(new_eff, new_eff ? active_fps_ : kIdleFps);
                }
            }
        }
    }
    if (changed) {
        cv_.notify_all();
    }
}

void InferenceControl::publish(const InferenceResult& r) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        latest_ = r;
    }
    cv_.notify_all();
}

bool InferenceControl::latest(bool has_after, uint32_t after, int wait_ms, InferenceResult& out) {
    if (wait_ms < 0) wait_ms = 0;
    if (wait_ms > kMaxWaitMs) wait_ms = kMaxWaitMs;

    std::unique_lock<std::mutex> lk(mu_);
    auto satisfied = [&] { return !has_after || latest_.frame_id > after; };
    if (has_after && wait_ms > 0) {
        cv_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                     [&] { return satisfied() || stopping_ || !effective_enabled_locked(); });
    }
    out = latest_;
    return latest_.frame_id == 0 || !satisfied();
}

void InferenceControl::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------------------

static std::string error_body(const std::string& msg) {
    return nlohmann::json{{"error", msg}}.dump();
}

static std::string state_body(const InferenceControl& ctl) {
    std::ostringstream s;
    s << "{\"enabled\":" << (ctl.continuous_enabled() ? "true" : "false")
      << ",\"fps\":" << ctl.current_fps()
      << ",\"active_fps\":" << ctl.active_fps()
      << ",\"idle_fps\":" << InferenceControl::kIdleFps << "}";
    return s.str();
}

static bool query_param(const std::string& query, const std::string& key, std::string& value) {
    size_t pos = 0;
    while (pos <= query.size()) {
        size_t end = query.find('&', pos);
        if (end == std::string::npos) end = query.size();
        const std::string item = query.substr(pos, end - pos);
        const size_t eq = item.find('=');
        if (item.substr(0, eq) == key) {
            value = eq == std::string::npos ? "" : item.substr(eq + 1);
            return true;
        }
        pos = end + 1;
    }
    return false;
}

static bool parse_u64(const std::string& s, unsigned long long& v) {
    if (s.empty() || s.size() > 19) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    v = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}

static HttpReply latest_reply(InferenceControl& ctl, const std::string& query, uint64_t now_ms) {
    std::string v;
    unsigned long long n = 0;
    bool has_after = false;
    uint32_t after = 0;
    int wait_ms = 0;
    if (query_param(query, "after", v)) {
        if (!parse_u64(v, n) || n > 0xFFFFFFFFull) return {400, error_body("after must be a frame_id (u32)")};
        has_after = true;
        after = static_cast<uint32_t>(n);
    }
    if (query_param(query, "wait_ms", v)) {
        if (!parse_u64(v, n)) return {400, error_body("wait_ms must be a non-negative integer")};
        wait_ms = n > static_cast<unsigned long long>(InferenceControl::kMaxWaitMs)
                      ? InferenceControl::kMaxWaitMs
                      : static_cast<int>(n);
    }

    InferenceResult r;
    const bool timeout = ctl.latest(has_after, after, wait_ms, r);
    const char* enabled = ctl.enabled() ? "true" : "false";
    std::ostringstream s;
    if (r.frame_id == 0) {
        s << "{\"enabled\":" << enabled << ",\"frame_id\":0,\"count\":0,\"markers\":[],\"timeout\":true}";
        return {200, s.str()};
    }
    const uint64_t effective_now = (now_ms == 0) ? get_unix_ms() : now_ms;
    const long long age = static_cast<long long>(effective_now) - static_cast<long long>(r.ts_ms);
    s << std::fixed << "{\"enabled\":" << enabled
      << ",\"frame_id\":" << r.frame_id
      << ",\"ts_ms\":" << r.ts_ms
      << ",\"age_ms\":" << (age < 0 ? 0 : age)
      << ",\"inference_time_ms\":" << std::setprecision(1) << r.inference_ms
      << ",\"count\":" << r.count
      << ",\"undistorted\":" << (r.undistorted ? "true" : "false")
      << ",\"rejected\":" << r.rejected
      << ",\"markers\":" << r.markers_json;
    if (timeout) s << ",\"timeout\":true";
    s << "}";
    return {200, s.str()};
}

static HttpReply trigger_reply(InferenceControl& ctl, const std::string& query, uint64_t now_ms) {
    std::string v;
    unsigned long long n = 0;
    int wait_ms = 2500;  // default 2.5s timeout for single-shot trigger
    if (query_param(query, "wait_ms", v)) {
        if (!parse_u64(v, n)) return {400, error_body("wait_ms must be a non-negative integer")};
        wait_ms = n > static_cast<unsigned long long>(InferenceControl::kMaxWaitMs)
                      ? InferenceControl::kMaxWaitMs
                      : static_cast<int>(n);
    }

    InferenceResult current;
    ctl.latest(false, 0, 0, current);
    const uint32_t baseline_frame_id = current.frame_id;

    InferenceResult r;
    bool timeout = false;
    {
        auto scope = ctl.trigger_scope();
        timeout = ctl.latest(true, baseline_frame_id, wait_ms, r);
    }

    std::ostringstream s;
    if (timeout || r.frame_id <= baseline_frame_id) {
        s << "{\"success\":false,\"timeout\":true,\"count\":0,\"markers\":[]}";
        return {200, s.str()};
    }

    const uint64_t effective_now = (now_ms == 0) ? get_unix_ms() : now_ms;
    const long long age = static_cast<long long>(effective_now) - static_cast<long long>(r.ts_ms);
    s << std::fixed << "{\"success\":true"
      << ",\"frame_id\":" << r.frame_id
      << ",\"ts_ms\":" << r.ts_ms
      << ",\"age_ms\":" << (age < 0 ? 0 : age)
      << ",\"inference_time_ms\":" << std::setprecision(1) << r.inference_ms
      << ",\"count\":" << r.count
      << ",\"undistorted\":" << (r.undistorted ? "true" : "false")
      << ",\"rejected\":" << r.rejected
      << ",\"markers\":" << r.markers_json
      << "}";
    return {200, s.str()};
}

HttpReply handle_control_request(InferenceControl& ctl, const std::string& method,
                                 const std::string& target, const std::string& body,
                                 uint64_t now_ms) {
    const size_t q = target.find('?');
    const std::string path = target.substr(0, q);
    const std::string query = q == std::string::npos ? "" : target.substr(q + 1);

    if (path == "/api/inference") {
        if (method == "GET") return {200, state_body(ctl)};
        if (method == "POST") {
            const nlohmann::json j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.is_object()) return {400, error_body("body must be a JSON object")};
            const auto it = j.find("enabled");
            if (it == j.end() || !it->is_boolean()) return {400, error_body("\"enabled\" must be true or false")};
            ctl.set_enabled(it->get<bool>());
            return {200, state_body(ctl)};
        }
    } else if (path == "/api/results/latest" && method == "GET") {
        return latest_reply(ctl, query, now_ms);
    } else if (path == "/api/trigger" && method == "POST") {
        return trigger_reply(ctl, query, now_ms);
    }
    return {404, error_body("not found")};
}

}  // namespace aruco
