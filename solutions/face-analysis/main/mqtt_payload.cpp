#include "mqtt_payload.h"

#include <iomanip>
#include <sstream>

/*
 * Field semantics changed here, deliberately and visibly:
 *
 * "id" used to be a per-detection counter that incremented on every face on
 * every frame -- it never identified anybody, despite the name. It is now a
 * tracker-assigned identity that holds while the face stays in frame.
 * "track_id" is published alongside with the same value, so a consumer can move
 * onto the unambiguous name and stop reading "id" whenever it likes.
 *
 * Every "*_confidence" is now a VOTE SHARE, not a single frame's softmax peak:
 * the winning label's slice of the probability mass this track has accumulated
 * since it was first seen. It rises as evidence agrees and stays low while the
 * heads disagree, so it is not comparable frame-for-frame with the old value.
 * Read it together with "evidence_frames" and "stable".
 *
 * "gated": the face was too small (short side below --min-face-px in source
 * pixels) for the attribute heads to be worth running, so none of them ran and
 * the attribute fields are defaults.
 */

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
        json << "\"track_id\":" << face.face.id << ",";
        json << "\"gated\":" << (attrs.gated ? "true" : "false") << ",";
        json << "\"stable\":" << (attrs.stable ? "true" : "false") << ",";
        json << "\"evidence_frames\":" << attrs.evidence_frames << ",";

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

        // Emotion. The key stays present for consumers that index it blindly,
        // but carries "" when no verdict exists -- publishing the enum default
        // reported every gated and every just-appeared face as "neutral".
        json << "\"emotion\":\"" << (attrs.has_emotion ? getEmotionName(attrs.emotion) : "") << "\",";
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
