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

// Liveness v2 verdict for one track at one instant.
//
// It lives here rather than in liveness_temporal.h so RecognitionResult can
// carry it without result_payload.cpp pulling in the temporal machinery, and so
// the host-only test target needs exactly one dependency-free header.
//
// `decision` is "pending" | "live" | "spoof". Every numeric field is nullable: a
// verdict exists before motion has a usable window, and the texture EMA does not
// exist before the first successful MiniFAS inference.
struct LivenessResult {
    std::string decision = "pending";
    bool has_score = false;
    float score = 0.f;
    bool has_texture = false;
    float texture = 0.f;
    bool has_motion = false;
    float motion = 0.f;
    bool blink = false;
    std::string reason = "insufficient_samples";

    bool isLive() const { return decision == "live"; }
    bool isSpoof() const { return decision == "spoof"; }
    bool isPending() const { return decision == "pending"; }
};

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

    // v2: nested verdict. `has_liveness_v2` is false when the v2 stage did not
    // run for this face, and the serializer then emits `"liveness": null`.
    bool has_liveness_v2 = false;
    LivenessResult liveness;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_TYPES_H_
