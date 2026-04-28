#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

namespace face_analysis {

struct LandmarkResult {
    bool ok = false;
    float pts[10] = {0};  // 5 points (x,y): left_eye, right_eye, nose, left_mouth, right_mouth
    int num_points = 5;   // 5 for PFLD-5point, 98 for WFLD-style (if available)
};

// Facial landmark inference (SG2002 / CVIMODEL)
// Input: RGB888 full frame + bbox
// Output: 5-point or 98-point normalized coordinates [0,1]
// - 5-point: left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner
// - 98-point: WFLW format, select 5 key indices for alignment
class LandmarkRunner {
public:
    LandmarkRunner() = default;
    ~LandmarkRunner() = default;

    bool init(const std::string& model_path);

    int inputSize() const { return input_size_; }

    void setPreprocess(float m0, float m1, float m2, float s0, float s1, float s2) {
        mean_[0] = m0;
        mean_[1] = m1;
        mean_[2] = m2;
        scale_[0] = s0;
        scale_[1] = s1;
        scale_[2] = s2;
    }

    void setCropScale(float s) { crop_scale_ = s; }

    bool infer(const uint8_t* rgb888, int src_w, int src_h,
               float x1, float y1, float x2, float y2,
               LandmarkResult& out);

    bool infer(const uint8_t* rgb888, int src_w, int src_h, int src_stride_bytes,
               float x1, float y1, float x2, float y2,
               LandmarkResult& out);

private:
    static float bf16_to_fp32(uint16_t v);
    static float fp16_to_fp32(uint16_t v);
    static uint16_t fp32_to_bf16(float v);
    static uint16_t fp32_to_fp16(float v);
    static size_t elem_size(ma_tensor_type_t t);
    static size_t shape_numel(const ma_shape_t& s);

    float read_val(const ma_tensor_t& t, int idx) const;

    void alignCropRgb(const uint8_t* src, int src_w, int src_h,
                      int src_stride_bytes,
                      float x1, float y1, float x2, float y2,
                      uint8_t* dst, int dst_size) const;

    bool prepareInputTensor();
    void packInput(const uint8_t* rgb_hwc_u8);
    bool parseOutputs(LandmarkResult& out, float crop_x1, float crop_y1, float crop_scale_factor);

private:
    int input_size_ = 112;
    int output_num_points_ = 5;  // detected from output shape
    float crop_scale_ = 1.5f;

    // PFLD default: mean=127.5, scale=1/127.5 -> [-1,1]
    float mean_[3] = {127.5f, 127.5f, 127.5f};
    float scale_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    std::vector<uint8_t> input_rgb_;

    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_NONE;
    bool input_is_chw_ = false;
    int input_c_ = 3;
    int input_h_ = 112;
    int input_w_ = 112;
    size_t input_numel_ = 0;

    std::vector<uint8_t> input_u8_;
    std::vector<int8_t> input_s8_;
    std::vector<uint16_t> input_u16_;
    std::vector<float> input_f32_;

    ma_tensor_t input_tensor_cache_{};
    bool inited_ = false;
};

}  // namespace face_analysis