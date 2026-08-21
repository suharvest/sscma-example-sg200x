#ifndef _FALL_POSE_DETECTOR_H_
#define _FALL_POSE_DETECTOR_H_

#include <memory>
#include <string>
#include <vector>

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

    // Returns every person that passes the confidence threshold.  Association
    // and per-person temporal state deliberately live outside the inference
    // wrapper (see multi_tracker.h), so a second person can never be silently
    // discarded or spliced into the first person's history.
    const std::vector<Subject>& detectAll(ma_img_t* img);

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    bool initialized() const { return initialized_; }
    bool inferenceFailed() const { return inference_failed_; }
    int keypointCount() const { return keypoint_count_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma::model::PoseDetector* model_ = nullptr;
    std::vector<Subject> subjects_;

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
