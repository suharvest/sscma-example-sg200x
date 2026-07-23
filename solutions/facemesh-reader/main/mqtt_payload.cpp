#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

#include "facemesh_pipeline.h"  // for AnalyzedFace definition

namespace facemesh_reader {

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<AnalyzedFace>& faces,
                            float inference_time_ms,
                            bool include_landmarks) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"face_count\":" << faces.size() << ",";
    json << "\"faces\":[";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& af = faces[i];
        if (i > 0) json << ",";

        json << "{";
        json << "\"id\":" << af.face.id << ",";

        // Bounding box (normalized 0-1)
        json << "\"bbox\":{";
        json << "\"x\":" << af.face.x << ",";
        json << "\"y\":" << af.face.y << ",";
        json << "\"w\":" << af.face.w << ",";
        json << "\"h\":" << af.face.h;
        json << "},";
        json << "\"confidence\":" << af.face.score << ",";

        // EAR / MAR
        json << "\"left_ear\":" << af.metrics.left_ear << ",";
        json << "\"right_ear\":" << af.metrics.right_ear << ",";
        json << "\"ear\":" << af.metrics.avg_ear << ",";
        json << "\"mar\":" << af.metrics.mar << ",";
        json << "\"eyes_closed\":" << (af.metrics.eyes_closed ? "true" : "false") << ",";
        json << "\"mouth_open\":" << (af.metrics.mouth_open ? "true" : "false") << ",";
        json << "\"metrics_valid\":" << (af.metrics.valid ? "true" : "false");

        // ---- Phase 2: edge-autonomous drowsiness conclusion ----
        json << ",\"drowsiness\":{";
        json << "\"level\":" << af.drowsiness.drowsiness_level << ",";
        json << "\"state\":\"" << af.drowsiness.state << "\",";
        json << "\"perclos_pct\":" << af.drowsiness.perclos_pct << ",";
        json << "\"continuous_closure_sec\":" << af.drowsiness.continuous_closure_sec << ",";
        json << "\"alert_active\":" << (af.drowsiness.alert_active ? "true" : "false") << ",";
        json << "\"drowsy_by_ear\":" << (af.drowsiness.drowsy_by_ear ? "true" : "false") << ",";
        json << "\"drowsy_by_perclos\":" << (af.drowsiness.drowsy_by_perclos ? "true" : "false") << ",";
        json << "\"drowsy_by_yawn\":" << (af.drowsiness.drowsy_by_yawn ? "true" : "false");
        json << "},";
        json << "\"yawn\":{";
        json << "\"is_yawning\":" << (af.yawn.is_yawning_now ? "true" : "false") << ",";
        json << "\"yawn_count_5min\":" << af.yawn.yawn_count_5min;
        json << "}";

        if (include_landmarks && !af.landmarks.empty()) {
            json << ",\"landmarks\":[";
            for (size_t k = 0; k < af.landmarks.size(); ++k) {
                if (k > 0) json << ",";
                json << "[" << af.landmarks[k].x << "," << af.landmarks[k].y << "]";
            }
            json << "]";
        }

        json << "}";
    }

    json << "]";
    json << "}";
    return json.str();
}

}  // namespace facemesh_reader
