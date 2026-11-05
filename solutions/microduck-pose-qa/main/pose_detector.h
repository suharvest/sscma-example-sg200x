#ifndef _MICRODUCK_POSE_DETECTOR_H_
#define _MICRODUCK_POSE_DETECTOR_H_

#include <memory>
#include <string>
#include <vector>

#include <core/engine/ma_engine_cvi.h>
#include <core/ma_types.h>
#include "norm_box.h"
#include "pose.h"

namespace microduck {

struct Subject {
    geometry::InferBox box;
    Pose pose;
    float score = 0.0f;
};

class PoseDetector {
public:
    PoseDetector() = default;
    ~PoseDetector();

    bool init(const std::string& model_path);
    void setThreshold(float robot_score);
    void setKeypointThreshold(float kpt_score);

    const std::vector<Subject>& detectAll(ma_img_t* img);
    const Subject* detectPrimary(ma_img_t* img);

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    bool initialized() const { return initialized_; }
    int keypointCount() const { return keypoint_count_; }
    bool lastInferenceOk() const { return last_run_ok_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    std::vector<Subject> subjects_;
    Subject primary_;
    bool has_primary_ = false;
    bool last_run_ok_ = true;

    float threshold_ = 0.60f;
    float threshold_nms_ = 0.45f;
    float kpt_threshold_ = 0.45f;
    int input_width_ = 384;
    int input_height_ = 384;
    int input_channels_ = 3;
    bool is_nchw_ = true;
    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_F32;
    int num_records_ = 3024;
    int num_keypoints_ = 17;
    int keypoint_count_ = 0;
    bool initialized_ = false;
    int8_t u8_to_s8_lut_[256] = {0};
};

}  // namespace microduck

#endif  // _MICRODUCK_POSE_DETECTOR_H_
