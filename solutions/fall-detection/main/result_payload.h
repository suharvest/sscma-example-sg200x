#ifndef _FALL_DETECTION_RESULT_PAYLOAD_H_
#define _FALL_DETECTION_RESULT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "fall_detector.h"
#include "multi_tracker.h"
#include "pose.h"

namespace fall {

struct PayloadContext {
    std::uint64_t timestamp_ms = 0;
    std::uint32_t frame_id = 0;
    float inference_time_ms = 0.0f;
    std::string stream_id = "camera-0";
    bool person_detected = false;
    int person_count = 0;
    int fallen_count = 0;
    int infer_w = 640;
    int infer_h = 640;
    int stream_w = 1280;
    int stream_h = 720;
    // Monotonic stream-level event sequence. Zero is accepted for legacy
    // callers, which then fall back to the maximum per-track event_id.
    std::uint64_t global_event_id = 0;
    bool global_event_id_valid = false;
    // Retained tracks include short occlusions so an event edge is not lost;
    // person_count remains the number of currently visible detections.
    std::vector<const TrackedPerson*> persons;
};

// Build the MQTT payload for all retained tracks. Top-level legacy fields are
// aggregate values; per-track state and pose data live in persons[].
std::string buildResultJson(const PayloadContext& ctx);

// Extra members for the debug /results envelope. Returned without braces so
// debug_stream_build_results() can splice it into the common envelope.
std::string buildDebugExtraJson(const PayloadContext& ctx);

// Legacy scalar overloads retained for offline tools and downstream code that
// still builds a synthetic result directly; the live path always uses persons[].
std::string buildResultJson(const PayloadContext& ctx, const FallOutput& output,
                            const FallObservation& observation, const Pose* pose);
std::string buildDebugExtraJson(const PayloadContext& ctx, const FallOutput& output,
                                const FallObservation& observation, const Pose* pose);

}  // namespace fall

#endif  // _FALL_DETECTION_RESULT_PAYLOAD_H_
