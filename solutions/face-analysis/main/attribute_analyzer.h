#ifndef _ATTRIBUTE_ANALYZER_H_
#define _ATTRIBUTE_ANALYZER_H_

#include <vector>
#include <string>
#include <memory>
#include <array>

#include <sscma.h>
#include "face_detector.h"

namespace face_analysis {

// Emotion types supported by FER+ model
enum class Emotion {
    NEUTRAL = 0,
    HAPPINESS,
    SURPRISE,
    SADNESS,
    ANGER,
    DISGUST,
    FEAR,
    CONTEMPT,
    COUNT
};

// Get emotion name string
inline const char* getEmotionName(Emotion emotion) {
    static const char* names[] = {
        "neutral", "happiness", "surprise", "sadness",
        "anger", "disgust", "fear", "contempt"
    };
    int idx = static_cast<int>(emotion);
    if (idx >= 0 && idx < static_cast<int>(Emotion::COUNT)) {
        return names[idx];
    }
    return "unknown";
}

struct FaceAttributes {
    // Age prediction
    int age;
    float age_confidence;

    // Gender prediction
    std::string gender;  // "male" or "female"
    float gender_confidence;

    // Emotion prediction
    Emotion emotion;
    float emotion_confidence;
    std::array<float, 8> emotion_probs;  // All emotion probabilities
};

struct AnalyzedFace {
    FaceInfo face;
    FaceAttributes attributes;
};

class AttributeAnalyzer {
public:
    AttributeAnalyzer();
    ~AttributeAnalyzer();

    // Initialize with model paths
    // genderage_model: InsightFace GenderAge model
    // emotion_model: FER+ emotion model (optional, pass empty string to disable)
    bool init(const std::string& genderage_model, const std::string& emotion_model = "");

    // Analyze attributes for a single face crop
    FaceAttributes analyze(ma_img_t* face_crop);

    // Analyze attributes for multiple faces from full frame
    // full_frame: The original camera frame
    // faces: Detected face bounding boxes
    std::vector<AnalyzedFace> analyzeAll(ma_img_t* full_frame,
                                          const std::vector<FaceInfo>& faces);

    // Check initialization status
    bool isGenderAgeReady() const { return genderage_ready_; }
    bool isEmotionReady() const { return emotion_ready_; }

private:
    // Crop and resize face region for model input
    bool cropFace(ma_img_t* full_frame, const FaceInfo& face,
                  ma_img_t* output, int target_width, int target_height);

    // Run GenderAge model
    void runGenderAge(ma_img_t* face_crop, FaceAttributes& attrs);

    // Run Emotion model
    void runEmotion(ma_img_t* face_crop, FaceAttributes& attrs);

private:
    // GenderAge model (InsightFace)
    std::unique_ptr<ma::engine::EngineCVI> genderage_engine_;
    int genderage_input_size_;
    bool genderage_ready_;

    // Emotion model (FER+)
    std::unique_ptr<ma::engine::EngineCVI> emotion_engine_;
    int emotion_input_size_;
    bool emotion_ready_;

    // Face crop buffer
    std::vector<uint8_t> crop_buffer_;
};

}  // namespace face_analysis

#endif  // _ATTRIBUTE_ANALYZER_H_
