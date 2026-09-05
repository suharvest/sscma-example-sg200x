#ifndef _FACE_RECOGNITION_CVI_RUNNER_H_
#define _FACE_RECOGNITION_CVI_RUNNER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

namespace face_recognition {

// One EngineCVI model with its single image input, packed the way the runtime
// tensor actually asks for it.
//
// Why this is shared rather than repeated per model: `setInput` performs an
// unchecked byte copy of `tensor.size`, so a runtime tensor with padded rows
// must be filled at its PHYSICAL row stride, not the logical one. Getting that
// wrong is silent -- the model runs and returns plausible garbage. The three
// models here have three different input types (SCRFD is S8, ArcFace and
// MiniFASNet are float), so the padding logic would otherwise be written three
// times.
//
// The type also decides the normalization:
//   * S8  -> raw passthrough `pixel - 128`. That is the convention the int8
//            converter bakes in (mean 127.5 / scale 1/128) and what
//            ma::model::Detector does for the very same SCRFD cvimodel.
//   * F32/BF16/F16 -> the host applies `(pixel - mean) * scale` itself. The
//            bf16 cvimodels keep `top.Input` outside the TPU subfunction, so
//            nothing on the device performs the normalization.
class CviRunner {
public:
    CviRunner() = default;
    ~CviRunner() = default;

    CviRunner(const CviRunner&) = delete;
    CviRunner& operator=(const CviRunner&) = delete;

    // `mean`/`scale` are applied only on float inputs (see above).
    // `bgr` swaps the channel order while packing (MiniFASNet was converted
    // with pixel_format=bgr).
    bool init(const std::string& model_path,
              const float mean[3], const float scale[3], bool bgr,
              std::string* error = nullptr);

    bool ready() const { return inited_; }
    int inputWidth() const { return input_w_; }
    int inputHeight() const { return input_h_; }
    int inputChannels() const { return input_c_; }
    bool inputIsChw() const { return input_is_chw_; }
    std::string inputDescription() const { return input_desc_; }

    // `rgb_hwc` is inputWidth() x inputHeight() packed 8-bit, 3 channels, in
    // RGB order regardless of the model's own channel order.
    bool run(const uint8_t* rgb_hwc, std::string* error = nullptr);

    int outputCount() const;
    ma_tensor_t output(int index) const;

    // Dequantized read of one element of one output tensor.
    static float readValue(const ma_tensor_t& t, size_t idx);
    static size_t elemSize(ma_tensor_type_t t);
    static size_t shapeNumel(const ma_shape_t& s);
    static size_t tensorNumel(const ma_tensor_t& t);
    static std::string shapeToString(const ma_shape_t& s);

private:
    bool prepareInput(std::string* error);
    void packInput(const uint8_t* rgb_hwc);
    bool submit(std::string* error);

    static float bf16ToFp32(uint16_t v);
    static float fp16ToFp32(uint16_t v);
    static uint16_t fp32ToBf16(float v);
    static uint16_t fp32ToFp16(float v);

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_tensor_{};
    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_NONE;
    bool input_is_chw_ = true;
    int input_c_ = 3;
    int input_h_ = 0;
    int input_w_ = 0;
    size_t input_numel_ = 0;
    std::string input_desc_;

    float mean_[3] = {0.f, 0.f, 0.f};
    float scale_[3] = {1.f, 1.f, 1.f};
    bool bgr_ = false;
    bool raw_passthrough_ = false;

    std::vector<uint8_t> buf_u8_;
    std::vector<int8_t> buf_s8_;
    std::vector<uint16_t> buf_u16_;
    std::vector<float> buf_f32_;
    std::vector<uint8_t> buf_physical_;

    bool inited_ = false;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_CVI_RUNNER_H_
