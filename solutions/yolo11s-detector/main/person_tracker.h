#ifndef _PERSON_TRACKER_H_
#define _PERSON_TRACKER_H_

#include <vector>
#include <map>
#include <cmath>

#include "yolo11s_detector.h"

namespace yolo11s {

enum class DwellState {
    TRANSIENT,
    DWELLING,
    ENGAGED,
    ASSISTANCE
};

inline const char* getDwellStateName(DwellState state) {
    switch (state) {
        case DwellState::TRANSIENT:  return "transient";
        case DwellState::DWELLING:   return "dwelling";
        case DwellState::ENGAGED:    return "engaged";
        case DwellState::ASSISTANCE: return "assistance";
        default: return "unknown";
    }
}

struct TrackedPerson {
    int track_id;
    Detection detection;

    float velocity_x = 0.0f;
    float velocity_y = 0.0f;
    float speed_px_s = 0.0f;
    float speed_normalized = 0.0f;

    DwellState dwell_state = DwellState::TRANSIENT;
    float dwell_duration_sec = 0.0f;

    float last_seen_time = 0.0f;
    float dwell_start_time = 0.0f;
    int frames_tracked = 0;
    int stationary_frames = 0;
    int lost_frames = 0;
};

struct TrackerConfig {
    float iou_threshold = 0.3f;
    int max_lost_frames = 30;

    float vel_alpha = 0.08f;
    float vel_alpha_sudden = 0.6f;
    float velocity_zero_threshold = 3.0f;

    float dwell_speed_threshold = 10.0f;
    float dwell_min_duration = 1.5f;
    float dwell_assistance_threshold = 20.0f;
    int dwell_min_frames = 5;

    int frame_width = 640;
    int frame_height = 640;
};

struct StateCount {
    int total = 0;
    int browsing = 0;
    int engaged = 0;
    int assistance = 0;
};

class PersonTracker {
public:
    PersonTracker();
    ~PersonTracker() = default;

    void setConfig(const TrackerConfig& config);

    std::vector<TrackedPerson> update(
        const std::vector<Detection>& detections,
        float current_time_sec
    );

    StateCount getStateCounts() const;
    int getTrackCount() const { return static_cast<int>(tracks_.size()); }

private:
    float computeIoU(const Detection& a, const Detection& b) const;

    std::vector<std::pair<int, int>> matchDetections(
        const std::vector<Detection>& detections
    ) const;

    void updateVelocity(TrackedPerson& track,
                        const Detection& new_det,
                        float dt);

    void updateDwellState(TrackedPerson& track, float current_time);

private:
    TrackerConfig config_;
    std::map<int, TrackedPerson> tracks_;
    int next_track_id_ = 0;
    float last_update_time_ = 0.0f;
};

}  // namespace yolo11s

#endif  // _PERSON_TRACKER_H_
