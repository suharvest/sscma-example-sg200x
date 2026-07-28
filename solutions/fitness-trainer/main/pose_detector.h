#ifndef _FITNESS_POSE_DETECTOR_H_
#define _FITNESS_POSE_DETECTOR_H_

#include <memory>
#include <string>

#include <sscma.h>

#include "norm_box.h"
#include "pose.h"

namespace fitness {

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

    // Returns the single most confident person, or nullptr when nobody passes
    // the score threshold. A fitness app tracks one athlete: picking the
    // largest-scoring subject each frame is both what the user expects and what
    // keeps the rep counter from being driven by a passer-by.
    const Subject* detectPrimary(ma_img_t* img);

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    bool initialized() const { return initialized_; }
    int keypointCount() const { return keypoint_count_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma::model::PoseDetector* model_ = nullptr;
    Subject primary_;
    bool has_primary_ = false;

    float threshold_ = 0.40f;
    float kpt_threshold_ = 0.50f;
    int input_width_ = 640;
    int input_height_ = 640;
    int keypoint_count_ = 0;
    bool initialized_ = false;
};

}  // namespace fitness

#endif  // _FITNESS_POSE_DETECTOR_H_
