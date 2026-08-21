#include "box_tracker.h"

#include <algorithm>
#include <cmath>

namespace fall {

float boxIou(const geometry::InferBox& a, const geometry::InferBox& b) {
    const float left = std::max(a.left(), b.left());
    const float top = std::max(a.top(), b.top());
    const float right = std::min(a.right(), b.right());
    const float bottom = std::min(a.bottom(), b.bottom());
    const float intersection = std::max(0.0f, right - left) *
                               std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
    const float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
    const float union_area = area_a + area_b - intersection;
    return union_area > 1e-8f ? intersection / union_area : 0.0f;
}

float boxCenterDistance(const geometry::InferBox& a, const geometry::InferBox& b) {
    return std::hypot(a.cx - b.cx, a.cy - b.cy);
}

std::vector<int> greedyBoxAssignment(const std::vector<geometry::InferBox>& detections,
                                     const std::vector<geometry::InferBox>& tracks,
                                     float iou_threshold,
                                     float center_distance_threshold) {
    iou_threshold = std::clamp(iou_threshold, 0.0f, 1.0f);
    center_distance_threshold = std::max(0.0f, center_distance_threshold);
    std::vector<int> assigned(detections.size(), -1);
    std::vector<bool> used(tracks.size(), false);

    struct Match {
        float score;
        std::size_t detection;
        std::size_t track;
    };
    std::vector<Match> matches;
    for (std::size_t d = 0; d < detections.size(); ++d) {
        for (std::size_t t = 0; t < tracks.size(); ++t) {
            const float iou = boxIou(detections[d], tracks[t]);
            const float distance = boxCenterDistance(detections[d], tracks[t]);
            if (iou < iou_threshold && distance > center_distance_threshold) continue;
            // IoU is the primary cue; centre distance breaks ties and keeps a
            // briefly occluded box associated after a small detector jitter.
            const float score = iou + (1.0f - std::min(1.0f, distance));
            matches.push_back({score, d, t});
        }
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.detection != b.detection) return a.detection < b.detection;
        return a.track < b.track;
    });
    for (const auto& match : matches) {
        if (assigned[match.detection] >= 0 || used[match.track]) continue;
        assigned[match.detection] = static_cast<int>(match.track);
        used[match.track] = true;
    }
    return assigned;
}

}  // namespace fall
