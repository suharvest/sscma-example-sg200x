#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

namespace weather_classifier {

std::string buildResultJson(uint64_t frame_id,
                            const std::string& label,
                            int class_id,
                            float confidence,
                            const std::vector<std::string>& labels,
                            const std::vector<float>& scores,
                            double inference_time_ms,
                            double capture_ms,
                            double preprocess_ms,
                            double total_ms) {
    std::ostringstream json;
    json << std::fixed;

    json << "{";
    json << "\"type\":\"classification\",";
    json << "\"frame\":" << frame_id << ",";
    json << "\"label\":\"" << label << "\",";
    json << "\"class_id\":" << class_id << ",";
    json << "\"confidence\":" << std::setprecision(4) << confidence << ",";

    json << "\"scores\":{";
    for (size_t i = 0; i < scores.size(); ++i) {
        const std::string name = i < labels.size() ? labels[i] : ("class_" + std::to_string(i));
        if (i > 0) json << ",";
        json << "\"" << name << "\":" << std::setprecision(4) << scores[i];
    }
    json << "},";

    json << "\"inference_time_ms\":" << std::setprecision(2) << inference_time_ms << ",";
    json << "\"capture_ms\":" << std::setprecision(2) << capture_ms << ",";
    json << "\"preprocess_ms\":" << std::setprecision(2) << preprocess_ms << ",";
    json << "\"total_ms\":" << std::setprecision(2) << total_ms;
    json << "}";

    return json.str();
}

}  // namespace weather_classifier
