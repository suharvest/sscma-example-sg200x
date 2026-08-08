#ifndef _FALL_DETECTION_RESULT_PAYLOAD_H_
#define _FALL_DETECTION_RESULT_PAYLOAD_H_

#include <cstdint>
#include <string>

#include "fall_detector.h"
#include "pose.h"

namespace fall {

struct PayloadContext {
    std::uint64_t timestamp_ms = 0;
    std::uint32_t frame_id = 0;
    float inference_time_ms = 0.0f;
    bool person_detected = false;
    int person_count = 0;
    int fallen_count = 0;
    int infer_w = 640;
    int infer_h = 640;
    int stream_w = 1280;
    int stream_h = 720;
};

std::string buildResultJson(const PayloadContext& ctx, const FallOutput& output,
                            const FallObservation& observation, const Pose* pose);

// Extra members for the debug /results envelope. Returned without braces so
// debug_stream_build_results() can splice it into the common envelope.
std::string buildDebugExtraJson(const PayloadContext& ctx, const FallOutput& output,
                                const FallObservation& observation, const Pose* pose);

}  // namespace fall

#endif  // _FALL_DETECTION_RESULT_PAYLOAD_H_
