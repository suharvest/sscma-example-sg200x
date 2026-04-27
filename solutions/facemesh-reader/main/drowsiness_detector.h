#ifndef _DROWSINESS_DETECTOR_H_
#define _DROWSINESS_DETECTOR_H_

#include <chrono>
#include <deque>
#include <string>
#include <utility>

namespace facemesh_reader {

// Snapshot of drowsiness state returned to consumers each frame.
// All durations in seconds, percents in [0, 100], level in [0, 1].
struct DrowsinessState {
    // Continuous closure tracking (current closure run)
    bool  is_eyes_closed = false;
    float continuous_closure_sec = 0.f;

    // PERCLOS sliding window
    float perclos_pct = 0.f;             // 0..100
    int   perclos_window_samples = 0;    // diagnostic

    // Per-dimension drowsy flags
    bool drowsy_by_ear     = false;      // continuous closure >= ear_continuous_sec
    bool drowsy_by_perclos = false;      // PERCLOS >= critical pct
    bool drowsy_by_yawn    = false;      // 5-min yawn count >= 3

    // Composite
    float drowsiness_level = 0.f;        // 0..1
    std::string state = "Alert";         // Alert / Tired / Drowsy / Danger

    // Alert
    bool alert_active = false;
};

class DrowsinessDetector {
public:
    struct Config {
        float ear_threshold        = 0.21f;   // EAR < threshold => closed
        float ear_continuous_sec   = 2.0f;    // continuous closure to trigger drowsy
        float perclos_window_sec   = 60.f;    // sliding window length
        float perclos_warning_pct  = 15.f;
        float perclos_critical_pct = 20.f;
        float alert_cooldown_sec   = 5.f;     // hold alert at least this long after recovery
        int   yawn_count_threshold = 3;       // drowsy_by_yawn if 5-min yawns >= this
    };

    DrowsinessDetector() = default;
    ~DrowsinessDetector() = default;

    void configure(const Config& cfg) { cfg_ = cfg; }
    const Config& config() const { return cfg_; }

    // Update with the current frame's average EAR + whether this frame triggered
    // a fresh yawn event (one-shot, true only on yawn-onset frame). Pass
    // yawn_count_5min so the detector can flag drowsy_by_yawn.
    DrowsinessState update(float ear_value, bool yawn_event_this_frame, int yawn_count_5min);

    void reset();

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    Config cfg_{};

    // Internal state (not exposed; snapshot returned each update)
    bool       is_eyes_closed_ = false;
    TimePoint  eyes_closed_start_{};
    float      continuous_closure_sec_ = 0.f;

    // PERCLOS rolling window of (timestamp, was_closed) samples
    std::deque<std::pair<TimePoint, bool>> perclos_window_;
    int        perclos_closed_count_ = 0;   // running count of closed samples in window

    bool       alert_active_ = false;
    TimePoint  alert_start_{};

    bool       initialized_clock_ = false;
};

}  // namespace facemesh_reader

#endif  // _DROWSINESS_DETECTOR_H_
