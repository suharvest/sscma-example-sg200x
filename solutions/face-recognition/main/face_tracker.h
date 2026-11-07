#ifndef _FACE_RECOGNITION_FACE_TRACKER_H_
#define _FACE_RECOGNITION_FACE_TRACKER_H_

#include <vector>

#include "face_types.h"

namespace face_recognition {

struct FaceTrackerConfig {
    float iou_threshold = 0.3f;  // Match threshold
    int max_lost_frames = 15;    // Drop the identity after this many consecutive misses
};

// Greedy IoU tracker over source-pixel boxes.
//
// Copied from face-analysis (main/face_tracker.{h,cpp}) with the box type
// swapped from normalized FaceInfo to BBox; IoU is scale-free so the arithmetic
// is unchanged.
//
// The track id is what everything downstream keys on: the embedding schedule,
// the "same name twice in a row" stability rule, and the published track_id.
// Without it a face would be re-embedded from scratch every frame.
//
// Deliberately no velocity prediction: at ~10fps on face-sized targets a Kalman
// stage would add state to tune without changing which detection wins.
class FaceTracker {
public:
    explicit FaceTracker(const FaceTrackerConfig& cfg = FaceTrackerConfig());

    // Consumes this frame's detections and returns track ids aligned by index.
    // Every detection either joins an existing track or starts a new one, so the
    // returned vector never contains -1.
    std::vector<int> update(const std::vector<FaceDetection>& faces);

    // Track ids retired by the most recent update(), for state reclamation.
    const std::vector<int>& removedIds() const { return removed_ids_; }

    int trackFrames(int track_id) const;
    void setConfig(const FaceTrackerConfig& cfg) { cfg_ = cfg; }

private:
    struct Track {
        int id = 0;
        BBox box;
        int frames_tracked = 0;
        int lost_frames = 0;
    };

    FaceTrackerConfig cfg_;
    std::vector<Track> tracks_;
    std::vector<int> removed_ids_;
    int next_id_ = 1;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_FACE_TRACKER_H_
