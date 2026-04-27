#include "drowsiness_detector.h"

#include <algorithm>
#include <cmath>

namespace facemesh_reader {

namespace {
inline float secondsBetween(std::chrono::steady_clock::time_point a,
                            std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<float>(a - b).count();
}
}  // namespace

void DrowsinessDetector::reset() {
    is_eyes_closed_ = false;
    continuous_closure_sec_ = 0.f;
    perclos_window_.clear();
    perclos_closed_count_ = 0;
    alert_active_ = false;
    initialized_clock_ = false;
}

DrowsinessState DrowsinessDetector::update(float ear_value,
                                            bool yawn_event_this_frame,
                                            int yawn_count_5min) {
    (void)yawn_event_this_frame;  // event-onset is informational; yawn_count_5min drives flag
    const TimePoint now = Clock::now();
    if (!initialized_clock_) {
        initialized_clock_ = true;
    }

    // ----- 1. Continuous eye closure -----
    const bool closed_now = (ear_value < cfg_.ear_threshold);
    if (closed_now) {
        if (!is_eyes_closed_) {
            eyes_closed_start_ = now;
            is_eyes_closed_ = true;
        }
        continuous_closure_sec_ = secondsBetween(now, eyes_closed_start_);
    } else {
        is_eyes_closed_ = false;
        continuous_closure_sec_ = 0.f;
    }
    const bool drowsy_by_ear =
        is_eyes_closed_ && (continuous_closure_sec_ >= cfg_.ear_continuous_sec);

    // ----- 2. PERCLOS sliding window (O(1) amortized) -----
    perclos_window_.emplace_back(now, closed_now);
    if (closed_now) ++perclos_closed_count_;

    const auto cutoff = now - std::chrono::microseconds(
        (int64_t)(cfg_.perclos_window_sec * 1e6f));
    while (!perclos_window_.empty() && perclos_window_.front().first < cutoff) {
        if (perclos_window_.front().second) --perclos_closed_count_;
        perclos_window_.pop_front();
    }

    float perclos_pct = 0.f;
    if (!perclos_window_.empty()) {
        perclos_pct = 100.f * (float)perclos_closed_count_ /
                      (float)perclos_window_.size();
    }
    const bool drowsy_by_perclos = (perclos_pct >= cfg_.perclos_critical_pct);

    // ----- 3. Yawn-based -----
    const bool drowsy_by_yawn = (yawn_count_5min >= cfg_.yawn_count_threshold);

    // ----- 4. Composite drowsiness level (B-plan: 2-D weighted, no head/gaze) -----
    const float ear_factor =
        std::min(1.0f, continuous_closure_sec_ / std::max(0.001f, cfg_.ear_continuous_sec));
    const float perclos_factor =
        std::min(1.0f, perclos_pct / std::max(0.001f, cfg_.perclos_critical_pct));
    float level = 0.5f * ear_factor + 0.5f * perclos_factor;
    // Yawn bumps level up but only if the eye signal already shows fatigue.
    if (drowsy_by_yawn) level = std::min(1.0f, level + 0.15f);
    if (level < 0.f) level = 0.f;
    if (level > 1.f) level = 1.f;

    // ----- 5. State machine -----
    std::string state_str;
    if (level < 0.3f)       state_str = "Alert";
    else if (level < 0.6f)  state_str = "Tired";
    else if (level < 0.8f)  state_str = "Drowsy";
    else                    state_str = "Danger";

    // ----- 6. Alert with cooldown -----
    constexpr float kAlertOnThreshold = 0.5f;
    if (level >= kAlertOnThreshold) {
        if (!alert_active_) {
            alert_active_ = true;
            alert_start_ = now;
        } else {
            // refresh start so cooldown counts from last "still drowsy" tick
            alert_start_ = now;
        }
    } else {
        if (alert_active_ &&
            secondsBetween(now, alert_start_) > cfg_.alert_cooldown_sec) {
            alert_active_ = false;
        }
    }

    DrowsinessState snap;
    snap.is_eyes_closed         = is_eyes_closed_;
    snap.continuous_closure_sec = continuous_closure_sec_;
    snap.perclos_pct            = perclos_pct;
    snap.perclos_window_samples = (int)perclos_window_.size();
    snap.drowsy_by_ear          = drowsy_by_ear;
    snap.drowsy_by_perclos      = drowsy_by_perclos;
    snap.drowsy_by_yawn         = drowsy_by_yawn;
    snap.drowsiness_level       = level;
    snap.state                  = state_str;
    snap.alert_active           = alert_active_;
    return snap;
}

}  // namespace facemesh_reader
