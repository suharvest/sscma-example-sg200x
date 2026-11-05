#ifndef _MICRODUCK_POSE_H_
#define _MICRODUCK_POSE_H_

#include <cmath>
#include <cstddef>
#include <vector>

#include <sscma.h>

namespace microduck {

enum class Joint : int {
    MouthTip      = 0,
    HeadRoll      = 1,
    HeadYaw       = 2,
    HeadPitch     = 3,
    NeckPitch     = 4,
    LeftHipYaw    = 5,
    LeftHipRoll   = 6,
    LeftHipPitch  = 7,
    LeftKnee      = 8,
    LeftAnkle     = 9,
    LeftFoot      = 10,
    RightHipYaw   = 11,
    RightHipRoll  = 12,
    RightHipPitch = 13,
    RightKnee     = 14,
    RightAnkle    = 15,
    RightFoot     = 16,
    Count         = 17,
};

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

struct Bone {
    int a;
    int b;
};

constexpr Bone kMicroduckSkeleton[] = {
    // 头部链
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    // 双髋基座
    {4, 5}, {4, 11}, {5, 11},
    // 左腿 (5->6->7->8->9->10)
    {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10},
    // 右腿 (11->12->13->14->15->16)
    {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
};

class Pose {
public:
    Pose() = default;
    Pose(const std::vector<ma_pt3f_t>& pts, int frame_w, int frame_h, float kpt_threshold);

    bool visible(int idx) const {
        return idx >= 0 && static_cast<size_t>(idx) < conf_.size() && conf_[idx] >= kpt_threshold_;
    }

    bool visible(Joint j) const {
        return visible(static_cast<int>(j));
    }

    Point2f at(int idx) const {
        return (idx >= 0 && static_cast<size_t>(idx) < pts_.size()) ? pts_[idx] : Point2f{};
    }

    Point2f at(Joint j) const {
        return at(static_cast<int>(j));
    }

    float confidence(int idx) const {
        return (idx >= 0 && static_cast<size_t>(idx) < conf_.size()) ? conf_[idx] : 0.0f;
    }

    float confidence(Joint j) const {
        return confidence(static_cast<int>(j));
    }

    int visibleCount() const {
        int cnt = 0;
        for (float c : conf_) {
            if (c >= kpt_threshold_) cnt++;
        }
        return cnt;
    }

    bool empty() const { return pts_.empty(); }

    const std::vector<Point2f>& points() const { return pts_; }
    const std::vector<float>& confidences() const { return conf_; }
    float threshold() const { return kpt_threshold_; }

private:
    std::vector<Point2f> pts_;
    std::vector<float> conf_;
    float kpt_threshold_ = 0.40f;
};

float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c);

}  // namespace microduck

#endif  // _MICRODUCK_POSE_H_
