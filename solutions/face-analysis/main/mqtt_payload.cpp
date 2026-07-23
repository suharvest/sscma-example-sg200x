#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

namespace face_analysis {

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<AnalyzedFace>& faces,
                            float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"face_count\":" << faces.size() << ",";
    json << "\"faces\":[";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        const auto& attrs = face.attributes;

        if (i > 0) json << ",";

        json << "{";
        json << "\"id\":" << face.face.id << ",";

        // Bounding box (normalized coordinates)
        json << "\"bbox\":{";
        json << "\"x\":" << face.face.x << ",";
        json << "\"y\":" << face.face.y << ",";
        json << "\"w\":" << face.face.w << ",";
        json << "\"h\":" << face.face.h;
        json << "},";

        json << "\"confidence\":" << face.face.score << ",";

        // Age (format depends on model)
        if (attrs.is_fairface) {
            json << "\"age_bin\":" << attrs.age_bin << ",";
        } else {
            json << "\"age\":" << attrs.age_continuous << ",";
        }
        json << "\"age_label\":\"" << attrs.age_label << "\",";
        json << "\"age_confidence\":" << attrs.age_confidence << ",";

        // Gender
        json << "\"gender\":\"" << attrs.gender << "\",";
        json << "\"gender_confidence\":" << attrs.gender_confidence << ",";

        // Race (FairFace only)
        if (attrs.race_bin >= 0) {
            json << "\"race\":\"" << attrs.race_label << "\",";
            json << "\"race_confidence\":" << attrs.race_confidence << ",";
        }

        // Emotion
        json << "\"emotion\":\"" << getEmotionName(attrs.emotion) << "\",";
        json << "\"emotion_confidence\":" << attrs.emotion_confidence << ",";

        // All emotion probabilities (HSEmotion AffectNet 8 classes)
        json << "\"emotion_probs\":{";
        json << "\"angry\":" << attrs.emotion_probs[0] << ",";
        json << "\"contempt\":" << attrs.emotion_probs[1] << ",";
        json << "\"disgust\":" << attrs.emotion_probs[2] << ",";
        json << "\"fear\":" << attrs.emotion_probs[3] << ",";
        json << "\"happy\":" << attrs.emotion_probs[4] << ",";
        json << "\"neutral\":" << attrs.emotion_probs[5] << ",";
        json << "\"sad\":" << attrs.emotion_probs[6] << ",";
        json << "\"surprise\":" << attrs.emotion_probs[7];
        json << "}";

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

}  // namespace face_analysis
