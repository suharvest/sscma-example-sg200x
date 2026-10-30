#include "motion_detector.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace region_motion {

MotionDetector::MotionDetector(const MotionConfig& config)
    : config_(config) {
}

void MotionDetector::updateConfig(const MotionConfig& config) {
    config_ = config;
    last_mask_w_ = 0;
    last_mask_h_ = 0;
}

void MotionDetector::rebuildRoiMask(int width, int height) {
    roi_mask_ = cv::Mat::zeros(height, width, CV_8UC1);
    last_mask_w_ = width;
    last_mask_h_ = height;

    if (config_.roi_zone.size() >= 3) {
        std::vector<cv::Point> pts;
        pts.reserve(config_.roi_zone.size());
        for (const auto& p : config_.roi_zone) {
            int px = std::clamp(static_cast<int>(std::round(p.x * (width - 1))), 0, width - 1);
            int py = std::clamp(static_cast<int>(std::round(p.y * (height - 1))), 0, height - 1);
            pts.push_back(cv::Point(px, py));
        }
        const cv::Point* ppt[1] = { pts.data() };
        int npt[] = { static_cast<int>(pts.size()) };
        cv::fillPoly(roi_mask_, ppt, npt, 1, cv::Scalar(255));
    } else {
        roi_mask_.setTo(cv::Scalar(255));
    }

    roi_total_pixels_ = cv::countNonZero(roi_mask_);
    if (roi_total_pixels_ <= 0) {
        roi_total_pixels_ = width * height;
    }
}

MotionResult MotionDetector::process(const uint8_t* rgb_data, int width, int height) {
    MotionResult res;
    if (!rgb_data || width <= 0 || height <= 0) {
        return res;
    }

    int proc_w = config_.inference_width;
    int proc_h = config_.inference_height;
    if (proc_w <= 0 || proc_h <= 0) {
        proc_w = width;
        proc_h = height;
    }

    cv::Mat input_mat(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));
    cv::Mat proc_rgb;
    if (width != proc_w || height != proc_h) {
        cv::resize(input_mat, proc_rgb, cv::Size(proc_w, proc_h));
    } else {
        proc_rgb = input_mat;
    }

    cv::Mat gray;
    cv::cvtColor(proc_rgb, gray, cv::COLOR_RGB2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    if (roi_mask_.empty() || last_mask_w_ != proc_w || last_mask_h_ != proc_h) {
        rebuildRoiMask(proc_w, proc_h);
    }

    if (bg_model_.empty() || bg_model_.size() != blurred.size()) {
        blurred.convertTo(bg_model_, CV_32FC1);
    } else {
        cv::accumulateWeighted(blurred, bg_model_, 0.05);
    }

    if (warmup_frames_ > 0) {
        warmup_frames_--;
        res.motion_detected = false;
        res.state_changed = false;
        res.motion_ratio = 0.0f;
        return res;
    }

    cv::Mat bg_gray;
    cv::convertScaleAbs(bg_model_, bg_gray);

    cv::Mat diff;
    cv::absdiff(blurred, bg_gray, diff);

    cv::Mat thresh;
    cv::threshold(diff, thresh, config_.pixel_threshold, 255, cv::THRESH_BINARY);

    cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);

    cv::Mat masked_motion;
    cv::bitwise_and(thresh, roi_mask_, masked_motion);

    int motion_pixels = cv::countNonZero(masked_motion);
    float ratio = 0.0f;
    if (roi_total_pixels_ > 0) {
        ratio = static_cast<float>(motion_pixels) / static_cast<float>(roi_total_pixels_);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(masked_motion.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<debug_stream_box_t> boxes;
    const double min_contour_area = 50.0;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_contour_area) continue;
        cv::Rect r = cv::boundingRect(c);
        debug_stream_box_t box;
        box.x = r.x + r.width * 0.5f;
        box.y = r.y + r.height * 0.5f;
        box.w = static_cast<float>(r.width);
        box.h = static_cast<float>(r.height);
        box.score = 1.0f;
        box.label = "motion";
        boxes.push_back(box);
    }

    bool raw_motion = (ratio >= config_.min_motion_area);
    bool prev_state = current_state_;

    if (raw_motion) {
        consecutive_motion_frames_++;
        consecutive_still_frames_ = 0;
        if (!current_state_ && consecutive_motion_frames_ >= config_.debounce_frames) {
            current_state_ = true;
        }
    } else {
        consecutive_still_frames_++;
        consecutive_motion_frames_ = 0;
        if (current_state_ && consecutive_still_frames_ >= config_.debounce_frames) {
            current_state_ = false;
        }
    }

    res.motion_detected = current_state_;
    res.state_changed = (current_state_ != prev_state);
    res.motion_ratio = ratio;
    res.boxes = std::move(boxes);
    return res;
}

}  // namespace region_motion
