#include "pose.h"

#include <algorithm>

namespace fitness {

Pose::Pose(const std::vector<ma_pt3f_t>& pts, int frame_w, int frame_h, float kpt_threshold)
    : kpt_threshold_(kpt_threshold) {
    pts_.reserve(pts.size());
    conf_.reserve(pts.size());
    for (const auto& p : pts) {
        pts_.push_back({p.x * static_cast<float>(frame_w), p.y * static_cast<float>(frame_h)});
        conf_.push_back(p.z);
    }
}

float Pose::sideScore(std::initializer_list<Joint> joints) const {
    float sum = 0.0f;
    int n = 0;
    for (Joint j : joints) {
        if (!visible(j)) return 0.0f;
        sum += confidence(j);
        ++n;
    }
    return n > 0 ? sum / static_cast<float>(n) : 0.0f;
}

float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c) {
    const float bax = a.x - b.x, bay = a.y - b.y;
    const float bcx = c.x - b.x, bcy = c.y - b.y;

    const float mag_ba = std::sqrt(bax * bax + bay * bay);
    const float mag_bc = std::sqrt(bcx * bcx + bcy * bcy);
    if (mag_ba < 1e-3f || mag_bc < 1e-3f) {
        // Coincident keypoints. The original divided by this unguarded and got
        // a domain error out of acos(); NaN so the caller can skip the frame.
        return std::nanf("");
    }

    float cosine = (bax * bcx + bay * bcy) / (mag_ba * mag_bc);
    cosine = std::max(-1.0f, std::min(1.0f, cosine));  // guard acos() domain
    return std::acos(cosine) * 180.0f / 3.14159265358979323846f;
}

}  // namespace fitness
