#include "yawn_detector.h"

namespace facemesh_reader {

void YawnDetector::reset() {
    consecutive_above_ = 0;
    is_yawning_ = false;
    yawn_events_.clear();
}

std::pair<YawnState, bool> YawnDetector::update(float mar_value) {
    const TimePoint now = Clock::now();
    bool event = false;

    if (mar_value > cfg_.mar_threshold) {
        ++consecutive_above_;
        if (!is_yawning_ && consecutive_above_ >= cfg_.consecutive_frames) {
            // Yawn onset (event)
            is_yawning_ = true;
            event = true;
            yawn_events_.push_back(now);
        }
    } else {
        consecutive_above_ = 0;
        is_yawning_ = false;
    }

    // Trim old events outside the window (O(1) amortized via head pop).
    const auto cutoff = now - std::chrono::microseconds(
        (int64_t)(cfg_.window_sec * 1e6f));
    while (!yawn_events_.empty() && yawn_events_.front() < cutoff) {
        yawn_events_.pop_front();
    }

    YawnState s;
    s.is_yawning_now  = is_yawning_;
    s.yawn_count_5min = (int)yawn_events_.size();
    return {s, event};
}

}  // namespace facemesh_reader
