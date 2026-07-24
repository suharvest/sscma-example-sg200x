#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

namespace weather_classifier {

// Build the classification results JSON published to the MQTT results topic.
//
// EXTERNAL CONTRACT (must not change): field names, order and precision match
// the original weather app's mqtt_publisher.cpp buildResultJson():
//   {"type":"classification","frame":N,"label":"..","class_id":N,
//    "confidence":0.xxxx,"scores":{"<label>":0.xxxx,..},
//    "inference_time_ms":x.xx,"capture_ms":x.xx,"preprocess_ms":x.xx,
//    "total_ms":x.xx}
std::string buildResultJson(uint64_t frame_id,
                            const std::string& label,
                            int class_id,
                            float confidence,
                            const std::vector<std::string>& labels,
                            const std::vector<float>& scores,
                            double inference_time_ms,
                            double capture_ms,
                            double preprocess_ms,
                            double total_ms);

}  // namespace weather_classifier

#endif  // _MQTT_PAYLOAD_H_
