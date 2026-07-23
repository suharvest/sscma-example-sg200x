#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "attribute_analyzer.h"

namespace face_analysis {

// Build the results JSON payload published to the MQTT results topic.
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<AnalyzedFace>& faces,
                            float inference_time_ms);

}  // namespace face_analysis

#endif  // _MQTT_PAYLOAD_H_
