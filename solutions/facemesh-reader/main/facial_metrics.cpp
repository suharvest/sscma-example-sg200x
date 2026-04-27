#include "facial_metrics.h"

#include <cmath>

namespace facemesh_reader {

static inline float dist2D(const Point2D& a, const Point2D& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 6-point EAR (Soukupová & Čech 2016):
//     EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
static float computeEAR(const std::vector<Point2D>& lm, const int idx[6]) {
    const Point2D& p0 = lm[idx[0]];
    const Point2D& p1 = lm[idx[1]];
    const Point2D& p2 = lm[idx[2]];
    const Point2D& p3 = lm[idx[3]];
    const Point2D& p4 = lm[idx[4]];
    const Point2D& p5 = lm[idx[5]];

    const float horiz = dist2D(p0, p3);
    if (horiz < 1e-6f) return 0.f;
    const float v1 = dist2D(p1, p5);
    const float v2 = dist2D(p2, p4);
    return (v1 + v2) / (2.f * horiz);
}

// 6-point MAR (simple): vertical opening over horizontal mouth width.
//     MAR = |p2-p5| / |p0-p4|
// Indices match kMouthIdx layout: 0=left corner, 4=right corner, 2=upper-mid, 5=lower-mid
static float computeMAR(const std::vector<Point2D>& lm, const int idx[6]) {
    const Point2D& left_corner  = lm[idx[0]];
    const Point2D& upper_mid    = lm[idx[2]];
    const Point2D& right_corner = lm[idx[4]];
    const Point2D& lower_mid    = lm[idx[5]];

    const float horiz = dist2D(left_corner, right_corner);
    if (horiz < 1e-6f) return 0.f;
    const float vert = dist2D(upper_mid, lower_mid);
    return vert / horiz;
}

FaceMetrics computeMetrics(const std::vector<Point2D>& landmarks) {
    FaceMetrics m;
    if (landmarks.size() < 468) {
        return m;  // valid=false
    }

    m.left_ear  = computeEAR(landmarks, kLeftEyeIdx);
    m.right_ear = computeEAR(landmarks, kRightEyeIdx);
    m.avg_ear   = 0.5f * (m.left_ear + m.right_ear);
    m.mar       = computeMAR(landmarks, kMouthIdx);

    m.eyes_closed = (m.avg_ear < kEarThreshold);
    m.mouth_open  = (m.mar > kMarThreshold);
    m.valid       = true;
    return m;
}

}  // namespace facemesh_reader
