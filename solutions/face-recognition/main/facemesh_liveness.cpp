#include "facemesh_liveness.h"

#include <algorithm>
#include <cmath>

#include "facial_metrics.h"

namespace face_recognition {

namespace {
void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}
}  // namespace

bool FacemeshLiveness::initOptional(const std::string& model_path, std::string* error) {
    ready_ = false;
    if (model_path.empty()) {
        setError(error, "no facemesh model configured");
        return false;
    }
    if (!landmarker_.init(model_path)) {
        setError(error, "facemesh landmarker init failed for " + model_path);
        return false;
    }
    if (landmarker_.inputW() != landmarker_.inputH()) {
        setError(error, "facemesh input is not square");
        return false;
    }
    size_ = landmarker_.inputW();
    roi_.assign(static_cast<size_t>(size_) * size_ * 3u, 0);
    ready_ = true;
    return true;
}

std::string FacemeshLiveness::inputDescription() const {
    return std::to_string(size_) + "x" + std::to_string(size_) + "x3 RGB";
}

bool FacemeshLiveness::infer(const uint8_t* rgb, int width, int height, const BBox& bbox,
                             float& ear, std::string* error) {
    ear = 0.f;
    if (!ready_) {
        setError(error, "facemesh not available");
        return false;
    }
    if (!rgb || width <= 1 || height <= 1) {
        setError(error, "invalid frame for facemesh");
        return false;
    }
    if (!(bbox.width() > 0.f) || !(bbox.height() > 0.f)) {
        setError(error, "invalid bbox for facemesh");
        return false;
    }

    // Pad, then square around the centre -- FaceMesh expects a square ROI and a
    // non-square resize would shear the eye aspect ratio, which is the one
    // quantity this whole path exists to measure.
    float x1 = bbox.x1, y1 = bbox.y1, x2 = bbox.x2, y2 = bbox.y2;
    const float bw = std::max(1.0f, x2 - x1);
    const float bh = std::max(1.0f, y2 - y1);
    x1 = std::max(0.0f, x1 - bw * bbox_padding_);
    y1 = std::max(0.0f, y1 - bh * bbox_padding_);
    x2 = std::min(static_cast<float>(width - 1), x2 + bw * bbox_padding_);
    y2 = std::min(static_cast<float>(height - 1), y2 + bh * bbox_padding_);

    const float cx = 0.5f * (x1 + x2);
    const float cy = 0.5f * (y1 + y2);
    const float side = std::max(x2 - x1, y2 - y1);
    x1 = cx - 0.5f * side;
    y1 = cy - 0.5f * side;
    x2 = cx + 0.5f * side;
    y2 = cy + 0.5f * side;

    const float src_w = std::max(1.0f, x2 - x1);
    const float src_h = std::max(1.0f, y2 - y1);
    const int dst = size_;

    // Bilinear sample with edge clamping: the square expansion above can push
    // the ROI past the frame, and clamping keeps the mapping affine (a shifted
    // ROI would not).
    auto sample = [&](float sx, float sy, int c) -> uint8_t {
        sx = std::min(std::max(sx, 0.f), static_cast<float>(width - 1));
        sy = std::min(std::max(sy, 0.f), static_cast<float>(height - 1));
        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int x1i = std::min(x0 + 1, width - 1);
        const int y1i = std::min(y0 + 1, height - 1);
        const float ax = sx - x0;
        const float ay = sy - y0;
        const uint8_t* p00 = rgb + (static_cast<size_t>(y0) * width + x0) * 3;
        const uint8_t* p10 = rgb + (static_cast<size_t>(y0) * width + x1i) * 3;
        const uint8_t* p01 = rgb + (static_cast<size_t>(y1i) * width + x0) * 3;
        const uint8_t* p11 = rgb + (static_cast<size_t>(y1i) * width + x1i) * 3;
        const float v0 = p00[c] + (p10[c] - p00[c]) * ax;
        const float v1 = p01[c] + (p11[c] - p01[c]) * ax;
        const int iv = static_cast<int>(std::lround(v0 + (v1 - v0) * ay));
        return static_cast<uint8_t>(std::min(255, std::max(0, iv)));
    };

    for (int dy = 0; dy < dst; ++dy) {
        const float sy = y1 + (dy + 0.5f) * (src_h / dst) - 0.5f;
        for (int dx = 0; dx < dst; ++dx) {
            const float sx = x1 + (dx + 0.5f) * (src_w / dst) - 0.5f;
            uint8_t* op = roi_.data() + (static_cast<size_t>(dy) * dst + dx) * 3;
            op[0] = sample(sx, sy, 0);
            op[1] = sample(sx, sy, 1);
            op[2] = sample(sx, sy, 2);
        }
    }

    const std::vector<facemesh_reader::Point2D> lm = landmarker_.infer(roi_.data());
    if (lm.size() < 468) {
        setError(error, "facemesh returned " + std::to_string(lm.size()) + " landmarks");
        return false;
    }

    // Map back to frame pixels. EAR is a ratio of distances, so a uniform scale
    // cancels out -- but the mapping is kept anyway so the metric is computed in
    // the same units the rest of the pipeline reports.
    const float sx = (x2 - x1) / static_cast<float>(dst);
    const float sy = (y2 - y1) / static_cast<float>(dst);
    mapped_.clear();
    mapped_.reserve(lm.size());
    for (const auto& p : lm) {
        mapped_.push_back(facemesh_reader::Point2D{x1 + p.x * sx, y1 + p.y * sy});
    }

    const facemesh_reader::FaceMetrics m = facemesh_reader::computeMetrics(mapped_);
    if (!m.valid) {
        setError(error, "facemesh metrics invalid");
        return false;
    }
    ear = m.avg_ear;
    return true;
}

}  // namespace face_recognition
