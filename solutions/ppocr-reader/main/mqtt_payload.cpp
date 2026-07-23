#include "mqtt_payload.h"

#include <cstdio>
#include <iomanip>
#include <sstream>

namespace ppocr {

// Escape special JSON characters in a string
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<OcrResult>& results,
                            const OcrTimings& timings,
                            int frame_width, int frame_height) {
    std::ostringstream json;
    json << std::fixed;

    float inv_w = (frame_width > 0) ? (1.0f / frame_width) : 1.0f;
    float inv_h = (frame_height > 0) ? (1.0f / frame_height) : 1.0f;

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":{";
    json << "\"detection\":" << std::setprecision(1) << timings.detection_ms << ",";
    json << "\"recognition\":" << std::setprecision(1) << timings.recognition_ms << ",";
    json << "\"total\":" << std::setprecision(1) << timings.total_ms;
    json << "},";
    json << "\"text_count\":" << results.size() << ",";
    json << "\"frame_width\":" << frame_width << ",";
    json << "\"frame_height\":" << frame_height << ",";
    json << "\"texts\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        if (i > 0) json << ",";
        json << "{";
        json << "\"id\":" << i << ",";

        // Box as 4 normalized points [[x,y],[x,y],[x,y],[x,y]]
        json << "\"box\":[";
        for (int p = 0; p < 4; ++p) {
            if (p > 0) json << ",";
            json << "[" << std::setprecision(4) << r.box.points[p][0] * inv_w
                 << "," << r.box.points[p][1] * inv_h << "]";
        }
        json << "],";

        json << "\"text\":\"" << jsonEscape(r.text) << "\",";
        json << "\"confidence\":" << std::setprecision(3) << r.rec_confidence << ",";
        json << "\"det_confidence\":" << std::setprecision(3) << r.det_confidence;
        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

}  // namespace ppocr
