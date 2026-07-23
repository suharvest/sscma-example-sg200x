#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

namespace facemesh_reader {

// Forward decl from facemesh_pipeline.h to avoid circular include.
struct AnalyzedFace;

// Build the results JSON payload published to the MQTT results topic.
// include_landmarks: if true, embed all 468 (x,y) pixel-coordinate landmarks per face.
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<AnalyzedFace>& faces,
                            float inference_time_ms,
                            bool include_landmarks = false);

}  // namespace facemesh_reader

#endif  // _MQTT_PAYLOAD_H_
