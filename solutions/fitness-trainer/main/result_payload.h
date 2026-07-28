#ifndef _FITNESS_RESULT_PAYLOAD_H_
#define _FITNESS_RESULT_PAYLOAD_H_

#include <cstdint>
#include <string>

#include "exercise.h"
#include "pose.h"

namespace fitness {

struct PayloadContext {
    uint64_t timestamp_ms = 0;
    uint32_t frame_id = 0;
    float inference_time_ms = 0.0f;
    bool person_detected = false;
    const char* exercise_id = "squat";
    int target_reps = 12;
    int target_sets = 3;

    // Frame geometry, needed to map keypoints from the inference channel into
    // the debug video's frame.
    int infer_w = 640;
    int infer_h = 640;
    int stream_w = 1280;
    int stream_h = 720;
};

// The MQTT results document published on recamera/fitness-trainer/results.
// This is the app's external contract -- Home Assistant templates, Node-RED
// flows and anything else downstream read these field names. Add fields
// freely; do not rename or remove them.
std::string buildResultJson(const PayloadContext& ctx, const ExerciseState& st);

// Extra top-level members for the debug /results envelope, which
// debug_stream_build_results() appends verbatim. Returned without the
// enclosing braces and without a leading comma.
//
// Besides the counters this emits two things the console's shared overlay
// renderer already knows how to draw:
//
//   "keypoints"   -> a skeleton (points + edges), instead of a bounding box.
//                    A box tells the athlete nothing; the skeleton is the
//                    thing that shows whether the model is tracking the joints
//                    the counter actually reads.
//   "status_card" -> a card pinned to a corner of the frame. The rep count
//                    belongs somewhere the eye can find it, not on a label
//                    that rides along with the person (and slides off the
//                    bottom of frame when they squat).
//
// Keypoint coordinates are in debug-video pixels, same space and same
// `resolution` as boxes elsewhere in the envelope. Only joints above the
// keypoint confidence threshold are emitted, and `edges` indexes into that
// compacted list -- the renderer draws every point it is given, so an
// invisible joint must not be in the array at all.
std::string buildDebugExtraJson(const PayloadContext& ctx, const ExerciseState& st,
                                const Pose* pose);

}  // namespace fitness

#endif  // _FITNESS_RESULT_PAYLOAD_H_
