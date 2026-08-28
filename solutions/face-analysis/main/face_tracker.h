#ifndef _FACE_TRACKER_H_
#define _FACE_TRACKER_H_

#include <vector>

#include "face_detector.h"

namespace face_analysis {

struct FaceTrackerConfig {
    float iou_threshold   = 0.3f;   // Match threshold
    int   max_lost_frames = 15;     // Drop the identity after this many consecutive misses
};

// Greedy IoU tracker over normalized [0,1] boxes.
//
// Exists because FaceDetector::detect() used to hand out a fresh, monotonically
// increasing number per detection per frame -- so "id" carried no identity at
// all. Everything downstream (attribute evidence accumulation, the MQTT "id"
// field) needs a value that stays put while the same face stays in frame.
//
// Deliberately no velocity prediction and no edge-aware retirement: the frame
// rate here is ~10fps on a single face-sized target, and a Kalman stage would
// add state to tune without changing which detection wins the match.
class FaceTracker {
public:
    explicit FaceTracker(const FaceTrackerConfig& cfg = FaceTrackerConfig());

    // Consumes this frame's detections (normalized coords) and returns track ids
    // aligned by index with `faces`. Every detection either joins an existing
    // track or starts a new one, so the returned vector never contains -1.
    std::vector<int> update(const std::vector<FaceInfo>& faces);

    // Track ids retired by the most recent update(), for evidence reclamation.
    const std::vector<int>& removedIds() const { return removed_ids_; }

    // Frames this track has been matched on. 0 for an unknown id.
    int trackFrames(int track_id) const;

    void setConfig(const FaceTrackerConfig& cfg) { cfg_ = cfg; }

private:
    struct Track {
        int   id             = 0;
        float x = 0.f, y = 0.f, w = 0.f, h = 0.f;
        int   frames_tracked = 0;
        int   lost_frames    = 0;
    };

    FaceTrackerConfig  cfg_;
    std::vector<Track> tracks_;
    std::vector<int>   removed_ids_;
    int                next_id_ = 1;
};

}  // namespace face_analysis

#endif  // _FACE_TRACKER_H_
