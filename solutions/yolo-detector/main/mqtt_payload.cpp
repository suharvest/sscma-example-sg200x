#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

namespace yolo {

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<Detection>& detections,
                            float inference_time_ms,
                            const std::map<int, int>* det_track_ids,
                            const TrackingSummary* tracking) {
    std::ostringstream json;
    json << std::fixed;

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms << ",";
    json << "\"detection_count\":" << detections.size() << ",";
    json << "\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];

        if (i > 0) json << ",";

        json << "{";
        json << "\"class_id\":" << det.class_id << ",";
        json << "\"class_name\":\"" << Detector::getClassName(det.class_id) << "\",";
        json << "\"confidence\":" << std::setprecision(3) << det.confidence << ",";

        // Bounding box (normalized center coordinates)
        json << "\"x\":" << std::setprecision(4) << det.x << ",";
        json << "\"y\":" << det.y << ",";
        json << "\"w\":" << det.w << ",";
        json << "\"h\":" << det.h;

        // Track id: only when tracking is on and this detection is tracked
        if (det_track_ids != nullptr) {
            auto it = det_track_ids->find(det.id);
            if (it != det_track_ids->end()) {
                json << ",\"track_id\":" << it->second;
            }
        }

        json << "}";
    }

    json << "]";

    // Optional tracker summary (tracking enabled)
    if (tracking != nullptr) {
        json << ",\"tracking\":{";
        json << "\"active_tracks\":" << tracking->active_tracks;
        if (tracking->has_line) {
            json << ",\"line_in\":" << tracking->line_in;
            json << ",\"line_out\":" << tracking->line_out;
        }
        json << "}";
    }

    json << "}";

    return json.str();
}

}  // namespace yolo
