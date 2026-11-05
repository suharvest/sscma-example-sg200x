#include "pose.h"
#include <algorithm>

namespace microduck {

Pose::Pose(const std::vector<ma_pt3f_t>& pts, int frame_w, int frame_h, float kpt_threshold)
    : kpt_threshold_(kpt_threshold) {
    pts_.reserve(pts.size());
    conf_.reserve(pts.size());
    for (const auto& p : pts) {
        pts_.push_back({p.x * static_cast<float>(frame_w), p.y * static_cast<float>(frame_h)});
        conf_.push_back(p.z);
    }
}

float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c) {
    const float bax = a.x - b.x, bay = a.y - b.y;
    const float bcx = c.x - b.x, bcy = c.y - b.y;

    const float mag_ba = std::sqrt(bax * bax + bay * bay);
    const float mag_bc = std::sqrt(bcx * bcx + bcy * bcy);
    if (mag_ba < 1e-3f || mag_bc < 1e-3f) {
        return std::nanf("");
    }

    float cosine = (bax * bcx + bay * bcy) / (mag_ba * mag_bc);
    cosine = std::max(-1.0f, std::min(1.0f, cosine));
    return std::acos(cosine) * 180.0f / 3.14159265358979323846f;
}

}  // namespace microduck
