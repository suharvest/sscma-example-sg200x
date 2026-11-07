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

bool LivenessRunner::initHead(Head& head, const std::string& path, float crop_scale,
                              std::string* error) {
    head.ready = false;
    head.crop_scale = crop_scale;
    if (path.empty()) {
        setError(error, "no model path configured");
        return false;
    }
    // Converted with pixel_format=bgr, mean 0, scale 1: the host hands over the
    // raw pixel values, only with the channel order swapped.
    const float mean[3] = {0.f, 0.f, 0.f};
    const float scale[3] = {1.f, 1.f, 1.f};
    if (!head.runner.init(path, mean, scale, /*bgr=*/true, error)) return false;

    if (head.runner.inputWidth() != head.runner.inputHeight() ||
        head.runner.inputChannels() != 3) {
        setError(error, "liveness input is not square 3-channel");
        return false;
    }
    head.size = head.runner.inputWidth();
    size_t total = 0;
    for (int i = 0; i < head.runner.outputCount(); ++i) {
        total += CviRunner::tensorNumel(head.runner.output(i));
    }
    // The MiniFAS family emits exactly three logits. A different count means a
    // different head was converted, and softmax index 1 would then be reading
    // some other model's class.
    if (total != 3) {
        setError(error, "liveness model does not produce exactly 3 logits (got " +
                            std::to_string(total) + ")");
        return false;
    }
    head.crop.assign(static_cast<size_t>(head.size) * head.size * 3u, 0);
    head.ready = true;
    return true;
}

bool LivenessRunner::initOptional(const std::string& v2_path, const std::string& v1se_path,
                                  std::string* error) {
    ready_ = false;
    if (!initHead(v2_, v2_path, 2.7f, error)) return false;
    ready_ = true;

    // V1SE is best-effort. A failure here is reported through `error` but the
    // stage still comes up single-headed, which is why the return value does not
    // depend on it.
    if (!v1se_path.empty()) {
        std::string v1se_err;
        if (!initHead(v1se_, v1se_path, 4.0f, &v1se_err)) {
            setError(error, "V1SE head unavailable (" + v1se_err + "); single-model fallback");
        } else {
            setError(error, "");
        }
    } else {
        setError(error, "no V1SE model configured; single-model fallback");
    }
    return true;
}

std::string LivenessRunner::inputDescription() const {
    std::string s = v2_.runner.inputDescription() + " @2.7x";
    if (v1se_.ready) s += " + " + v1se_.runner.inputDescription() + " @4.0x";
    else s += " (V1SE absent)";
    return s;
}

bool LivenessRunner::inferOne(Head& head, const uint8_t* rgb, int width, int height,
                              const BBox& raw_bbox, float& p_real, std::string* error) {
    p_real = 0.f;
    const float bw = raw_bbox.width();
    const float bh = raw_bbox.height();
    if (!(bw > 0.f) || !(bh > 0.f)) {
        setError(error, "invalid bbox for liveness crop");
        return false;
    }

    float scale = std::min({(height - 1.f) / bh, (width - 1.f) / bw, head.crop_scale});
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
    cv::Mat dst(head.size, head.size, CV_8UC3, head.crop.data());
    cv::resize(frame(roi), dst, cv::Size(head.size, head.size), 0, 0, cv::INTER_LINEAR);

    if (!head.runner.run(head.crop.data(), error)) return false;

    std::vector<float> logits;
    for (int i = 0; i < head.runner.outputCount(); ++i) {
        const ma_tensor_t tt = head.runner.output(i);
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
    p_real = static_cast<float>(std::exp(static_cast<double>(logits[1]) - m) / sum);
    return true;
}

bool LivenessRunner::inferRgbFrame(const uint8_t* rgb, int width, int height,
                                   const BBox& raw_bbox, TextureScores& out,
                                   std::string* error) {
    out = TextureScores{};
    if (!ready_) {
        setError(error, "liveness runner not available");
        return false;
    }
    if (!inferOne(v2_, rgb, width, height, raw_bbox, out.v2, error)) return false;
    out.mean = out.v2;

    if (v1se_.ready) {
        std::string v1se_err;
        if (inferOne(v1se_, rgb, width, height, raw_bbox, out.v1se, &v1se_err)) {
            out.has_v1se = true;
            out.mean = 0.5f * (out.v2 + out.v1se);
        } else {
            // A per-frame V1SE failure must not discard a perfectly good V2
            // reading: keep the V2-only mean and let the caller log.
            setError(error, "V1SE inference failed: " + v1se_err);
        }
    }
    return true;
}

}  // namespace face_recognition
