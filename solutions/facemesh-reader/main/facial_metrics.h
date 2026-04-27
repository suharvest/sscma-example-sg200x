#ifndef _FACIAL_METRICS_H_
#define _FACIAL_METRICS_H_

#include <vector>
#include <array>

namespace facemesh_reader {

struct Point2D {
    float x;
    float y;
};

struct FaceMetrics {
    float left_ear = 0.f;
    float right_ear = 0.f;
    float avg_ear = 0.f;
    float mar = 0.f;
    bool eyes_closed = false;   // avg_ear < EAR_THRESHOLD
    bool mouth_open = false;    // mar > MAR_THRESHOLD
    bool valid = false;         // true if landmark count >= 468
};

// EAR / MAR thresholds (kept consistent with DriveSafe Python defaults).
constexpr float kEarThreshold = 0.21f;
constexpr float kMarThreshold = 0.65f;

// MediaPipe FaceMesh 468-point indices (6-point approximation per region).
// Left eye:  outer corner, upper1, upper2, inner corner, lower2, lower1
constexpr int kLeftEyeIdx[6]  = {33, 160, 158, 133, 153, 144};
// Right eye: outer corner, upper1, upper2, inner corner, lower2, lower1
constexpr int kRightEyeIdx[6] = {362, 385, 387, 263, 373, 380};
// Mouth (outer-lip 6-point): left corner, upper-left, upper-mid, upper-right, right corner, lower-mid
constexpr int kMouthIdx[6]    = {61, 39, 0, 269, 291, 17};

// landmarks: 468 (or more) points in any consistent 2D coordinate system.
// Returns a FaceMetrics with `valid=false` if input is too small.
FaceMetrics computeMetrics(const std::vector<Point2D>& landmarks);

}  // namespace facemesh_reader

#endif  // _FACIAL_METRICS_H_
