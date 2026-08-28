#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

namespace face_analysis {

// Class count of the shipped HSEnotion enet_b0_8 head. The array below is sized
// by it; a model emitting fewer classes fills only its own prefix (`n_probs`).
static constexpr int kEmotionClassCount = 8;

struct EmotionResult {
    bool ok = false;
    int emotion = -1;
    float score = 0.f;   // peak probability -- the single-frame argmax's confidence

    // Full softmax distribution, not just the peak. Cross-frame attribute voting
    // (`AttributeEvidence`) accumulates probability MASS per track, so handing it
    // a one-hot vector built from `emotion`/`score` would make every frame vote
    // with the same weight regardless of how uncertain that frame actually was.
    // The softmax is already computed in full inside parseOutputs; this just
    // stops throwing it away.
    float probs[kEmotionClassCount] = {};
    int n_probs = 0;     // classes actually written into `probs`
};

// Emotion inference (SG2002 / CVIMODEL)
// - Input: RGB888 full frame + bbox
// - Output: softmax over the model's classes, plus its argmax
// - The SHIPPED model is HSEmotion enet_b0_8 (AffectNet, 8 classes). Its label
//   order is owned by `attribute_analyzer.h` (`Emotion` enum / getEmotionName):
//   anger, contempt, disgust, fear, happiness, neutral, sadness, surprise.
//   This header used to document a different 7-class order; that comment did not
//   match the deployed model and has been removed rather than corrected in place.
class EmotionRunner {
public:
    EmotionRunner() = default;
    ~EmotionRunner() = default;

    bool init(const std::string& model_path);

    int inputSize() const { return input_size_; }

    void setPreprocess(float mean0, float mean1, float mean2, float scale0, float scale1, float scale2) {
        mean_[0] = mean0;
        mean_[1] = mean1;
        mean_[2] = mean2;
        scale_[0] = scale0;
        scale_[1] = scale1;
        scale_[2] = scale2;
    }

    void setCropScale(float s) { crop_scale_ = s; }

    bool infer(const uint8_t* rgb888, int src_w, int src_h,
               float x1, float y1, float x2, float y2,
               EmotionResult& out);

private:
    static float bf16_to_fp32(uint16_t v);
    static float fp16_to_fp32(uint16_t v);
    static uint16_t fp32_to_bf16(float v);
    static uint16_t fp32_to_fp16(float v);
    static size_t elem_size(ma_tensor_type_t t);
    static size_t shape_numel(const ma_shape_t& s);

    float read_val(const ma_tensor_t& t, int idx) const;

    void alignCropRgb(const uint8_t* src, int src_w, int src_h,
                      float x1, float y1, float x2, float y2,
                      uint8_t* dst, int dst_size) const;

    bool prepareInputTensor();
    void packInput(const uint8_t* rgb_hwc_u8);
    bool parseOutputs(EmotionResult& out);

private:
    int input_size_ = 64;
    float crop_scale_ = 1.3f;

    float mean_[3] = {0.0f, 0.0f, 0.0f};
    float scale_[3] = {1.0f, 1.0f, 1.0f};

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    std::vector<uint8_t> input_rgb_;

    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_NONE;
    bool input_is_chw_ = false;
    int input_c_ = 3;
    int input_h_ = 64;
    int input_w_ = 64;
    size_t input_numel_ = 0;

    std::vector<uint8_t> input_u8_;
    std::vector<int8_t> input_s8_;
    std::vector<uint16_t> input_u16_;
    std::vector<float> input_f32_;

    ma_tensor_t input_tensor_cache_{};
    bool inited_ = false;
};

}  // namespace face_analysis
