#ifndef _FACE_RECOGNITION_LIVENESS_RUNNER_H_
#define _FACE_RECOGNITION_LIVENESS_RUNNER_H_

#include <string>
#include <vector>

#include "cvi_runner.h"
#include "face_types.h"

namespace face_recognition {

// MiniFASNet 2.7_80x80 anti-spoof head. Optional: it is loaded last and a
// failure only disables it -- detection and recognition stay mandatory, because
// the model ION pool is the binding constraint on this device and liveness is
// the one stage the pipeline is still useful without.
//
// The crop is the Minivision `CropImage._get_new_box` contract: expand the raw
// detection box around its centre by 2.7, clamp the effective scale so the
// expansion fits the image, and SHIFT a box that crosses a border back inside
// rather than truncating it. Truncating changes the framing, and this model is
// trained on the framing, not on the face.
class LivenessRunner {
public:
    // Never returns an error the caller must treat as fatal; `error` explains
    // why it is unavailable.
    bool initOptional(const std::string& model_path, std::string* error = nullptr);
    bool available() const { return ready_; }
    std::string inputDescription() const { return runner_.inputDescription(); }

    // `real_probability` is softmax(logits)[1] over the model's three classes.
    bool inferRgbFrame(const uint8_t* rgb, int width, int height,
                       const BBox& raw_bbox, float& real_probability,
                       std::string* error = nullptr);

private:
    CviRunner runner_;
    bool ready_ = false;
    int size_ = 80;
    float crop_scale_ = 2.7f;
    std::vector<uint8_t> crop_;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_LIVENESS_RUNNER_H_
