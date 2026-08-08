#ifndef _FALL_POSE_DETECTOR_H_
#define _FALL_POSE_DETECTOR_H_

#include <memory>
#include <string>

#include <sscma.h>

#include "norm_box.h"
#include "pose.h"

namespace fall {

// One detected person: the box (centre-normalised against the INFERENCE frame,
// which is what YOLO11-Pose emits) plus the keypoints in inference pixels.
struct Subject {
    geometry::InferBox box;
    Pose pose;
    float score = 0.0f;
};

// Thin wrapper over sscma-micro's pose models.
//
// ModelFactory auto-detects the variant from the tensor shapes, so this works
// with YOLO11-Pose (what ships on the device), YOLOv8-Pose and YOLO26-Pose
// without a code change. The device model outputs F32, which the sscma-micro
// postprocessor handles on its own path -- there is no dequantisation to do
// here.
class PoseDetector {
public:
    PoseDetector() = default;
    ~PoseDetector();

    bool init(const std::string& model_path);
    void setThreshold(float person_score);
    void setKeypointThreshold(float kpt_score) { kpt_threshold_ = kpt_score; }

    // Returns one stable subject, or nullptr when nobody passes the threshold.
    // Once acquired, association uses box overlap instead of re-picking the
    // highest score every frame, preventing two nearby people from silently
    // splicing their poses into one temporal sequence.
    const Subject* detectPrimary(ma_img_t* img);

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    bool initialized() const { return initialized_; }
    bool inferenceFailed() const { return inference_failed_; }
    int keypointCount() const { return keypoint_count_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma::model::PoseDetector* model_ = nullptr;
    Subject primary_;
    bool has_primary_ = false;
    geometry::InferBox tracked_box_;
    bool have_tracked_box_ = false;
    int tracking_misses_ = 0;

    float threshold_ = 0.40f;
    float kpt_threshold_ = 0.50f;
    int input_width_ = 640;
    int input_height_ = 640;
    int keypoint_count_ = 0;
    bool initialized_ = false;
    bool inference_failed_ = false;
};

}  // namespace fall

#endif  // _FALL_POSE_DETECTOR_H_
