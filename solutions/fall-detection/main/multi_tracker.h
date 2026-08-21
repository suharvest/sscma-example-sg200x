#ifndef _FALL_MULTI_TRACKER_H_
#define _FALL_MULTI_TRACKER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "fall_detector.h"
#include "pose_detector.h"
#include "temporal_classifier.h"

namespace fall {

// State owned by one associated person.  FallDetector and TemporalClassifier
// are intentionally members rather than process globals: two people in the
// same camera must never splice their motion histories.
struct TrackedPerson {
    std::uint64_t track_id = 0;
    Subject subject;
    bool visible = false;
    int age = 0;
    int missed = 0;
    double last_seen_sec = 0.0;

    FallDetector fall;
    TemporalClassifier temporal;
    FallOutput output;
    FallObservation observation;

    // Internal generation marker used while updating a vector that can grow
    // during a frame. It is kept public only to avoid another allocation.
    bool updated_this_frame = false;
};

struct TrackerConfig {
    float iou_threshold = 0.20f;
    float center_distance_threshold = 0.25f;
    int max_missed_frames = 30;
    float timeout_sec = 3.0f;
    FallConfig fall;
};

// Lightweight greedy multi-person tracker for a single reCamera stream. It
// returns retained tracks (including short-lived occluded tracks) so a fall
// edge emitted during a post-impact gap is still present in the result frame.
// `activeCount()` remains the true number of currently visible detections.
class MultiPersonTracker {
public:
    explicit MultiPersonTracker(TrackerConfig config = {});

    void reset();
    void setFallConfig(const FallConfig& config);
    void setTimeout(float timeout_sec, int max_missed_frames = -1);

    std::vector<TrackedPerson*> update(const std::vector<Subject>& detections,
                                       double timestamp_sec,
                                       int inference_width,
                                       int inference_height);

    int activeCount() const;
    // Monotonic stream-level event sequence. It increments once for every
    // per-track fall_event edge; per-track FallOutput::event_id remains local
    // and is emitted inside persons[].
    std::uint64_t globalEventId() const { return global_event_id_; }
    // Number of frames that carried at least one fall_event edge. This differs
    // from globalEventId() when multiple tracks fall in the same frame.
    std::uint64_t eventEdgeCount() const { return event_edge_count_; }
    const std::vector<std::unique_ptr<TrackedPerson>>& tracks() const { return tracks_; }
    const TrackerConfig& config() const { return config_; }

private:
    void updateVisible(TrackedPerson& track, const Subject& subject,
                       double timestamp_sec, int inference_width,
                       int inference_height);
    void updateOccluded(TrackedPerson& track, double timestamp_sec);

    TrackerConfig config_;
    std::uint64_t next_id_ = 1;
    std::uint64_t global_event_id_ = 0;
    std::uint64_t event_edge_count_ = 0;
    std::vector<std::unique_ptr<TrackedPerson>> tracks_;
};

// Convert a detector subject into the geometric observation consumed by the
// host-testable fall state machine. Coordinates are normalized to the model
// input frame; pose itself remains in input pixels for angle correctness.
FallObservation observationFromSubject(const Subject& subject, double timestamp_sec,
                                       int inference_height);

}  // namespace fall

#endif  // _FALL_MULTI_TRACKER_H_
