#ifndef _FALL_POSE_H_
#define _FALL_POSE_H_

// COCO-17 keypoint semantics and geometric helpers used by fall features.
//
// The upstream Python project this was ported from used MediaPipe Pose, whose
// 33 landmarks are numbered differently. Every joint it actually read exists in
// COCO-17, so the port is an index remap -- but the two numberings overlap
// without agreeing (MediaPipe 11 is the LEFT shoulder, COCO 11 is the LEFT
// HIP), which is exactly the kind of silent mistranslation that produces a
// plausible-looking angle for the wrong body part. Hence: no raw indices
// outside this header. Ask for Joint::LeftShoulder.
//
// Keep all keypoint access through Joint names; raw detector indices are easy
// to misread when porting models with a different landmark convention.

#include <cmath>
#include <cstddef>
#include <vector>

#include <sscma.h>

namespace fall {

enum class Joint : int {
    Nose          = 0,
    LeftEye       = 1,
    RightEye      = 2,
    LeftEar       = 3,
    RightEar      = 4,
    LeftShoulder  = 5,
    RightShoulder = 6,
    LeftElbow     = 7,
    RightElbow    = 8,
    LeftWrist     = 9,
    RightWrist    = 10,
    LeftHip       = 11,
    RightHip      = 12,
    LeftKnee      = 13,
    RightKnee     = 14,
    LeftAnkle     = 15,
    RightAnkle    = 16,
    Count         = 17,
};

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

// One person's keypoints, in inference-frame PIXELS.
//
// The model emits coordinates normalised to the inference channel; angles
// computed on normalised coordinates are wrong whenever the channel is not
// square (a 4:3 frame squashes y, tilting every angle). Pose converts once, at
// construction, and everything downstream works in pixels.
class Pose {
public:
    Pose() = default;

    // pts: the model's normalised keypoints (x,y in [0,1], z = per-keypoint
    // confidence). Sizes other than 17 are accepted but leave the missing
    // joints invisible rather than reading out of bounds.
    Pose(const std::vector<ma_pt3f_t>& pts, int frame_w, int frame_h, float kpt_threshold);

    bool visible(Joint j) const {
        auto i = static_cast<size_t>(j);
        return i < conf_.size() && conf_[i] >= kpt_threshold_;
    }

    Point2f at(Joint j) const {
        auto i = static_cast<size_t>(j);
        return i < pts_.size() ? pts_[i] : Point2f{};
    }

    float confidence(Joint j) const {
        auto i = static_cast<size_t>(j);
        return i < conf_.size() ? conf_[i] : 0.0f;
    }

    bool allVisible(std::initializer_list<Joint> joints) const {
        for (Joint j : joints) {
            if (!visible(j)) return false;
        }
        return true;
    }

    // Mean confidence over the listed joints; 0 when any is below threshold.
    // Mean confidence over the listed joints.
    float sideScore(std::initializer_list<Joint> joints) const;

    bool empty() const { return pts_.empty(); }

private:
    std::vector<Point2f> pts_;
    std::vector<float> conf_;
    float kpt_threshold_ = 0.5f;
};

// Interior angle at vertex b, in degrees, range [0,180].
// Returns NaN when either limb has zero length (coincident keypoints), which
// callers must treat as "no reading" rather than as 0 degrees.
float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c);

inline bool isReading(float angle) { return !std::isnan(angle); }

}  // namespace fall

#endif  // _FALL_POSE_H_
