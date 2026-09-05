#include "liveness_runner.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace face_recognition {

namespace {
void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}
}  // namespace

bool LivenessRunner::initOptional(const std::string& model_path, std::string* error) {
    ready_ = false;
    if (model_path.empty()) {
        setError(error, "no liveness model configured");
        return false;
    }
    // Converted with pixel_format=bgr, mean 0, scale 1: the host hands over the
    // raw pixel values, only with the channel order swapped.
    const float mean[3] = {0.f, 0.f, 0.f};
    const float scale[3] = {1.f, 1.f, 1.f};
    if (!runner_.init(model_path, mean, scale, /*bgr=*/true, error)) return false;

    if (runner_.inputWidth() != runner_.inputHeight() || runner_.inputChannels() != 3) {
        setError(error, "liveness input is not square 3-channel");
        return false;
    }
    size_ = runner_.inputWidth();
    size_t total = 0;
    for (int i = 0; i < runner_.outputCount(); ++i) {
        total += CviRunner::tensorNumel(runner_.output(i));
    }
    if (total < 2) {
        setError(error, "liveness model outputs fewer than 2 logits");
        return false;
    }
    crop_.assign(static_cast<size_t>(size_) * size_ * 3u, 0);
    ready_ = true;
    return true;
}

bool LivenessRunner::inferRgbFrame(const uint8_t* rgb, int width, int height,
                                   const BBox& raw_bbox, float& real_probability,
                                   std::string* error) {
    real_probability = 0.f;
    if (!ready_) {
        setError(error, "liveness runner not available");
        return false;
    }
    const float bw = raw_bbox.width();
    const float bh = raw_bbox.height();
    if (!(bw > 0.f) || !(bh > 0.f)) {
        setError(error, "invalid bbox for liveness crop");
        return false;
    }

    float scale = std::min({(height - 1.f) / bh, (width - 1.f) / bw, crop_scale_});
    const float new_w = bw * scale;
    const float new_h = bh * scale;
    const float cx = raw_bbox.x1 + bw * 0.5f;
    const float cy = raw_bbox.y1 + bh * 0.5f;

    float left = cx - new_w * 0.5f;
    float top = cy - new_h * 0.5f;
    float right = cx + new_w * 0.5f;
    float bottom = cy + new_h * 0.5f;

    // Shift back inside the frame rather than truncating (official behaviour).
    if (left < 0.f) { right -= left; left = 0.f; }
    if (top < 0.f) { bottom -= top; top = 0.f; }
    if (right > width - 1.f) { left -= right - (width - 1.f); right = width - 1.f; }
    if (bottom > height - 1.f) { top -= bottom - (height - 1.f); bottom = height - 1.f; }
    left = std::max(0.f, left);
    top = std::max(0.f, top);
    right = std::min(static_cast<float>(width - 1), right);
    bottom = std::min(static_cast<float>(height - 1), bottom);

    const int l = static_cast<int>(left);
    const int t = static_cast<int>(top);
    const int r = static_cast<int>(right);
    const int b = static_cast<int>(bottom);
    if (r < l || b < t) {
        setError(error, "empty liveness crop");
        return false;
    }

    const cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(rgb));
    // Inclusive corners, matching the Python slice `[top:bottom+1, left:right+1]`.
    const cv::Rect roi(l, t, r - l + 1, b - t + 1);
    cv::Mat dst(size_, size_, CV_8UC3, crop_.data());
    cv::resize(frame(roi), dst, cv::Size(size_, size_), 0, 0, cv::INTER_LINEAR);

    if (!runner_.run(crop_.data(), error)) return false;

    std::vector<float> logits;
    for (int i = 0; i < runner_.outputCount(); ++i) {
        const ma_tensor_t tt = runner_.output(i);
        const size_t n = CviRunner::tensorNumel(tt);
        for (size_t k = 0; k < n; ++k) logits.push_back(CviRunner::readValue(tt, k));
    }
    if (logits.size() < 2) {
        setError(error, "liveness produced fewer than 2 logits");
        return false;
    }

    float m = logits[0];
    for (float v : logits) m = std::max(m, v);
    double sum = 0.0;
    for (float v : logits) sum += std::exp(static_cast<double>(v) - m);
    if (!(sum > 0.0)) {
        setError(error, "liveness softmax underflow");
        return false;
    }
    real_probability = static_cast<float>(std::exp(static_cast<double>(logits[1]) - m) / sum);
    return true;
}

}  // namespace face_recognition
