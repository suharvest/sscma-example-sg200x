#ifndef REGION_MOTION_DETECTOR_H_
#define REGION_MOTION_DETECTOR_H_

#include <vector>
#include <opencv2/core.hpp>
#include "motion_config.h"
#include <debug_stream.h>

namespace region_motion {

struct MotionResult {
    bool motion_detected = false;          // Debounced state (ON / OFF)
    bool state_changed = false;            // True on state transition
    float motion_ratio = 0.0f;             // Motion pixels in ROI / total ROI pixels
    std::vector<debug_stream_box_t> boxes; // Bounding boxes for debug overlay (inference coords, center-based)
};

class MotionDetector {
public:
    explicit MotionDetector(const MotionConfig& config);
    ~MotionDetector() = default;

    void updateConfig(const MotionConfig& config);
    MotionResult process(const uint8_t* rgb_data, int width, int height);

    const std::vector<Point>& getRoiZone() const { return config_.roi_zone; }
    bool hasRoiZone() const { return config_.roi_zone.size() >= 3; }

private:
    void rebuildRoiMask(int width, int height);

    MotionConfig config_;
    cv::Mat bg_model_;           // CV_32FC1 running average background
    cv::Mat roi_mask_;           // CV_8UC1 binary mask
    int roi_total_pixels_ = 0;
    int last_mask_w_ = 0;
    int last_mask_h_ = 0;

    // Debounce state machine
    bool current_state_ = false;
    int consecutive_motion_frames_ = 0;
    int consecutive_still_frames_ = 0;
    int warmup_frames_ = 5;
};

}  // namespace region_motion

#endif  // REGION_MOTION_DETECTOR_H_
