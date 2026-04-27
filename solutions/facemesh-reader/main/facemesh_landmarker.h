#ifndef _FACEMESH_LANDMARKER_H_
#define _FACEMESH_LANDMARKER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

#include "facial_metrics.h"  // for Point2D

namespace facemesh_reader {

// MediaPipe FaceMesh INT8 cvimodel runner.
// Input:  192x192 RGB packed (uint8). Internally normalized to fp32 = pixel/255.
// Output: 468 (x,y) points in the 192x192 input coordinate system (fp32).
class FacemeshLandmarker {
public:
    FacemeshLandmarker() = default;
    ~FacemeshLandmarker() = default;

    bool init(const std::string& model_path);
    bool isReady() const { return ready_; }

    int inputW() const { return input_w_; }
    int inputH() const { return input_h_; }

    // Run inference on a 192x192 RGB packed buffer (HWC, uint8).
    // Returns 468 Point2D (in 192x192 coordinates) on success, empty on failure.
    std::vector<Point2D> infer(const uint8_t* roi_rgb_192x192);

private:
    // Read tensor element index `idx` as fp32, dispatching on type and quant.
    float readVal(const ma_tensor_t& t, int idx) const;
    static size_t elemSize(ma_tensor_type_t t);
    static size_t shapeNumel(const ma_shape_t& s);

    static float fp16_to_fp32(uint16_t v);
    static float bf16_to_fp32(uint16_t v);
    static uint16_t fp32_to_fp16(float v);
    static uint16_t fp32_to_bf16(float v);

    bool prepareInputTensor();

    int findLandmarkOutputIndex();

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    bool ready_ = false;

    // Cached input tensor metadata
    ma_tensor_t input_tensor_cache_{};
    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_NONE;
    bool input_is_chw_ = false;
    int input_c_ = 3;
    int input_h_ = 192;
    int input_w_ = 192;
    size_t input_numel_ = 0;

    // Per-input-type backing buffers
    std::vector<uint8_t>  input_u8_;
    std::vector<int8_t>   input_s8_;
    std::vector<uint16_t> input_u16_;
    std::vector<float>    input_f32_;

    // Index of the 1404-element landmark output tensor (resolved at init).
    int landmark_output_idx_ = 0;
};

}  // namespace facemesh_reader

#endif  // _FACEMESH_LANDMARKER_H_
