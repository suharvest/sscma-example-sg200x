#ifndef _FACE_RECOGNITION_TYPES_H_
#define _FACE_RECOGNITION_TYPES_H_

#include <array>
#include <cstdint>
#include <string>

namespace face_recognition {

struct Point2f {
    float x = 0.f;
    float y = 0.f;
};

// Source-pixel corners (the frame the detector was handed), never normalized.
// Normalization happens once, in result_payload.cpp, so every consumer of a
// BBox inside the pipeline works in the same units.
struct BBox {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float shortSide() const { return width() < height() ? width() : height(); }
};

using Landmarks5 = std::array<Point2f, 5>;
using Embedding512 = std::array<float, 512>;

struct FaceDetection {
    BBox bbox;
    Landmarks5 landmarks{};
    float score = 0.f;
    int track_id = -1;
};

struct RecognitionResult {
    int track_id = -1;
    BBox bbox;
    float det_score = 0.f;
    std::string name;     // empty = no match
    float score = 0.f;    // best cosine observed, match or not
    bool has_liveness = false;
    bool live = false;
    float liveness_score = 0.f;
    bool stable = false;
    bool gated = false;
    std::string reason;   // "small", "align", "embed", "spoof", "" when fine
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_TYPES_H_
