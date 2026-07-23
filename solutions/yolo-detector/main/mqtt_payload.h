#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "detector.h"
#include "person_tracker.h"

namespace yolo {

// Build the basic detection results JSON payload published to the MQTT
// results topic (tracking disabled).
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<Detection>& detections,
                            float inference_time_ms);

// Build the tracking results JSON payload (tracking enabled).
// line_crossing: optional cumulative entry-line counters; when non-null a
//                "line_crossing":{"in":N,"out":N} object is added.
std::string buildTrackingJson(uint64_t timestamp_ms, uint32_t frame_id,
                              const std::vector<TrackedPerson>& persons,
                              const StateCount& counts,
                              float inference_time_ms,
                              const LineCrossingCount* line_crossing = nullptr);

}  // namespace yolo

#endif  // _MQTT_PAYLOAD_H_
