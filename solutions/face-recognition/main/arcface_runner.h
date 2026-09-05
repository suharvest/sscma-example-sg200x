#ifndef _FACE_RECOGNITION_ARCFACE_RUNNER_H_
#define _FACE_RECOGNITION_ARCFACE_RUNNER_H_

#include <array>
#include <string>

#include "cvi_runner.h"
#include "face_types.h"

namespace face_recognition {

// MobileFaceNet ArcFace embedder, 112x112 RGB in, 512-D L2-normalized out.
//
// The converter recorded mean 127.5 / scale 1/127.5 on `top.Input`, but that op
// stays OUTSIDE the TPU subfunction in the bf16 build -- nothing on the device
// applies it. CviRunner therefore normalizes on the host whenever the runtime
// input tensor is a float type, and falls back to the int8 `pixel-128`
// convention if the model is ever re-converted to INT8.
class ArcFaceRunner {
public:
    bool init(const std::string& model_path, std::string* error = nullptr);
    bool ready() const { return ready_; }
    std::string inputDescription() const { return runner_.inputDescription(); }

    // `aligned_rgb` is a packed 112x112x3 RGB chip from alignArcFace112().
    bool infer(const uint8_t* aligned_rgb, Embedding512& normalized,
               std::string* error = nullptr);

private:
    CviRunner runner_;
    bool ready_ = false;
};

// Cosine of two already-L2-normalized 512-D vectors.
float cosineSimilarity(const Embedding512& a, const Embedding512& b);

// In-place L2 normalization; returns false when the norm is zero.
bool l2Normalize(Embedding512& v);

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_ARCFACE_RUNNER_H_
