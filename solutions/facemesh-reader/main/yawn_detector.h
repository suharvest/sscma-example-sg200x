#ifndef _YAWN_DETECTOR_H_
#define _YAWN_DETECTOR_H_

#include <chrono>
#include <deque>
#include <utility>

namespace facemesh_reader {

struct YawnState {
    bool is_yawning_now  = false;
    int  yawn_count_5min = 0;
};

// Yawn = MAR > threshold for >= consecutive_frames frames in a row.
// Each yawn instance is counted exactly once (event-debounce).
class YawnDetector {
public:
    struct Config {
        float mar_threshold      = 0.65f;
        int   consecutive_frames = 5;       // frames in a row above threshold => yawn onset
        float window_sec         = 300.f;   // 5-min stat window
    };

    YawnDetector() = default;
    ~YawnDetector() = default;

    void configure(const Config& cfg) { cfg_ = cfg; }
    const Config& config() const { return cfg_; }

    // Returns (state_snapshot, this_frame_is_yawn_event).
    // event=true ONLY on the frame the yawn first crosses consecutive_frames
    // (one-shot per yawn instance), used by DrowsinessDetector.
    std::pair<YawnState, bool> update(float mar_value);

    void reset();

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    Config cfg_{};

    int       consecutive_above_ = 0;
    bool      is_yawning_        = false;          // currently inside a yawn instance
    std::deque<TimePoint> yawn_events_;             // 5-min sliding count
};

}  // namespace facemesh_reader

#endif  // _YAWN_DETECTOR_H_
