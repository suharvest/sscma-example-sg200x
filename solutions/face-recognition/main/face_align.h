#ifndef _FACE_RECOGNITION_FACE_ALIGN_H_
#define _FACE_RECOGNITION_FACE_ALIGN_H_

#include <array>
#include <cstdint>

#include "face_types.h"

namespace face_recognition {

// Canonical ArcFace 5-point destination template for a 112x112 chip.
extern const float kArcFaceDst112[10];

// Least-squares similarity transform (rotation + uniform scale + translation)
// from five source points to five destination points. Returns false on a
// degenerate system. Lifted from face-analysis's attribute_analyzer, which
// needed it because cv::estimateAffinePartial2D lives in calib3d and the
// reCamera SDK OpenCV does not ship it.
bool solveSimilarity2D(const float src[10], const float dst[10], double M[6]);

// Warp the face at `landmarks` into the canonical 112x112 ArcFace pose.
//
// Returns false when the landmarks are non-finite or the transform is
// degenerate. The caller must gate such a face rather than fall back to a bbox
// crop: a bbox-aligned chip produces an embedding that is not comparable with
// anything already in the gallery, and the failure would show up as a silent
// accuracy loss instead of an error.
bool alignArcFace112(const uint8_t* rgb, int width, int height,
                     const Landmarks5& landmarks,
                     std::array<uint8_t, 112 * 112 * 3>& aligned_rgb);

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_FACE_ALIGN_H_
