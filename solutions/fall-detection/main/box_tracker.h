#ifndef _FALL_BOX_TRACKER_H_
#define _FALL_BOX_TRACKER_H_

// Small, device-independent greedy box association helper.  Keeping this
// seam free of SSCMA headers makes the matching policy replayable in host unit
// tests and keeps the embedded tracker intentionally lightweight.

#include <vector>

#include "norm_box.h"

namespace fall {

float boxIou(const geometry::InferBox& a, const geometry::InferBox& b);
float boxCenterDistance(const geometry::InferBox& a, const geometry::InferBox& b);

// Return one track index per detection (-1 when a new track is needed).  A
// detection may match when either IoU or centre distance passes its gate;
// candidates are then greedily consumed in descending combined score.  This
// is deliberately deterministic and avoids a heavyweight dependency such as
// DeepSORT for the small SG2002 scene.
std::vector<int> greedyBoxAssignment(const std::vector<geometry::InferBox>& detections,
                                     const std::vector<geometry::InferBox>& tracks,
                                     float iou_threshold = 0.20f,
                                     float center_distance_threshold = 0.25f);

}  // namespace fall

#endif  // _FALL_BOX_TRACKER_H_
