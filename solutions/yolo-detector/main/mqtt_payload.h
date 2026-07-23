#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "detector.h"

namespace yolo {

// Optional tracker summary appended to the results payload when tracking is
// enabled. line in/out counters are included only when has_line is true.
struct TrackingSummary {
    int active_tracks = 0;
    bool has_line = false;
    uint32_t line_in = 0;
    uint32_t line_out = 0;
};

// Build the detection results JSON payload published to the MQTT results
// topic. Single schema for both tracking and non-tracking modes:
//   {"timestamp":..,"frame_id":..,"inference_time_ms":..,
//    "detection_count":N,"detections":[{"class_id":..,"class_name":"..",
//    "confidence":..,"x":..,"y":..,"w":..,"h":..,"track_id":..}, ...],
//    "tracking":{"active_tracks":N}}          <- only when tracking enabled
// Coordinates are normalized [0-1] center x/y plus w/h.
// det_track_ids: optional map of Detection.id -> track id; matched detections
//                get a "track_id" field. tracking: optional summary object.
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<Detection>& detections,
                            float inference_time_ms,
                            const std::map<int, int>* det_track_ids = nullptr,
                            const TrackingSummary* tracking = nullptr);

}  // namespace yolo

#endif  // _MQTT_PAYLOAD_H_
