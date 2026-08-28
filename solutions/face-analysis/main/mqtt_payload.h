#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "attribute_analyzer.h"

namespace face_analysis {

// Build the results JSON payload published to the MQTT results topic.
//
// Note for consumers: "id" is now a tracker identity that persists while the
// face stays in frame (it was a per-detection counter), and every
// "*_confidence" is a vote share over accumulated evidence rather than a single
// frame's softmax peak. See the comment block at the top of mqtt_payload.cpp.
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<AnalyzedFace>& faces,
                            float inference_time_ms);

}  // namespace face_analysis

#endif  // _MQTT_PAYLOAD_H_
