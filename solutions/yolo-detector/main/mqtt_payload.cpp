#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

namespace yolo {

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<Detection>& detections,
                            float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);

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
        json << "\"id\":" << det.id << ",";
        json << "\"class_id\":" << det.class_id << ",";
        json << "\"class_name\":\"" << Detector::getClassName(det.class_id) << "\",";
        json << "\"confidence\":" << std::setprecision(3) << det.confidence << ",";

        // Bounding box (normalized center coordinates)
        json << "\"bbox\":{";
        json << "\"x\":" << std::setprecision(4) << det.x << ",";
        json << "\"y\":" << det.y << ",";
        json << "\"w\":" << det.w << ",";
        json << "\"h\":" << det.h;
        json << "}";

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

std::string buildTrackingJson(uint64_t timestamp_ms, uint32_t frame_id,
                              const std::vector<TrackedPerson>& persons,
                              const StateCount& counts,
                              float inference_time_ms,
                              const LineCrossingCount* line_crossing) {
    std::ostringstream json;
    json << std::fixed;

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms << ",";

    // Zone occupancy summary
    json << "\"zone_occupancy\":{";
    json << "\"total\":" << counts.total << ",";
    json << "\"browsing\":" << counts.browsing << ",";
    json << "\"engaged\":" << counts.engaged << ",";
    json << "\"assistance\":" << counts.assistance;
    json << "},";

    // Cumulative entry-line crossing counters (only when a line is configured)
    if (line_crossing != nullptr) {
        json << "\"line_crossing\":{";
        json << "\"in\":" << line_crossing->in_count << ",";
        json << "\"out\":" << line_crossing->out_count;
        json << "},";
    }

    // Person array
    json << "\"persons\":[";

    for (size_t i = 0; i < persons.size(); ++i) {
        const auto& person = persons[i];

        if (i > 0) json << ",";

        json << "{";
        json << "\"track_id\":" << person.track_id << ",";
        json << "\"confidence\":" << std::setprecision(3) << person.detection.confidence << ",";

        // Bounding box
        json << "\"bbox\":{";
        json << "\"x\":" << std::setprecision(4) << person.detection.x << ",";
        json << "\"y\":" << person.detection.y << ",";
        json << "\"w\":" << person.detection.w << ",";
        json << "\"h\":" << person.detection.h;
        json << "},";

        // Velocity info
        json << "\"speed_px_s\":" << std::setprecision(1) << person.speed_px_s << ",";
        json << "\"speed_normalized\":" << std::setprecision(1) << person.speed_normalized << ",";

        // Dwell state
        json << "\"state\":\"" << getDwellStateName(person.dwell_state) << "\",";
        json << "\"dwell_duration_sec\":" << std::setprecision(1) << person.dwell_duration_sec;

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

}  // namespace yolo
