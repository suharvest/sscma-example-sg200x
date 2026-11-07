#ifndef _FACE_RECOGNITION_LIVENESS_RUNNER_H_
#define _FACE_RECOGNITION_LIVENESS_RUNNER_H_

#include <string>
#include <vector>

#include "cvi_runner.h"
#include "face_types.h"

namespace face_recognition {

// MiniFASNet anti-spoof texture ensemble.
//
// v2 runs TWO heads over the same detection box at different crop scales:
//   * MiniFASNet-V2   at 2.7x  -- tight, mostly face texture.
//   * MiniFASNet-V1SE at 4.0x  -- wide, includes the frame/bezel/hand context
//                                 that betrays a printed sheet or a phone.
// The Minivision reference stack ensembles exactly these two scales, and the two
// crops are independent: neither is a resize of the other.
//
// The V1SE head is optional SEPARATELY from the stage as a whole. If its file is
// absent or fails to load, the runner degrades to the single V2 head, warns, and
// keeps working -- on this device the model ION pool is the binding constraint,
// and a single-head texture stage is worth more than no stage at all.
//
// The crop is the Minivision `CropImage._get_new_box` contract: expand the raw
// detection box around its centre by the scale, clamp the effective scale so the
// expansion fits the image, and SHIFT a box that crosses a border back inside
// rather than truncating it. Truncating changes the framing, and these models
// are trained on the framing, not just on the face.
struct TextureScores {
    float v2 = 0.f;
    float v1se = 0.f;
    bool has_v1se = false;
    float mean = 0.f;   // arithmetic mean over the heads that actually ran
};

class LivenessRunner {
public:
    // Never returns an error the caller must treat as fatal; `error` explains why
    // the stage is unavailable. Succeeds as long as the V2 head loaded.
    bool initOptional(const std::string& v2_path, const std::string& v1se_path,
                      std::string* error = nullptr);

    bool available() const { return ready_; }
    bool ensembleAvailable() const { return ready_ && v1se_.ready; }
    std::string inputDescription() const;

    // Runs every loaded head once, each over its own crop of the same box.
    bool inferRgbFrame(const uint8_t* rgb, int width, int height, const BBox& raw_bbox,
                       TextureScores& out, std::string* error = nullptr);

private:
    struct Head {
        CviRunner runner;
        bool ready = false;
        int size = 80;
        float crop_scale = 2.7f;
        std::vector<uint8_t> crop;
    };

    bool initHead(Head& head, const std::string& path, float crop_scale, std::string* error);
    bool inferOne(Head& head, const uint8_t* rgb, int width, int height, const BBox& raw_bbox,
                  float& p_real, std::string* error);

    Head v2_;
    Head v1se_;
    bool ready_ = false;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_LIVENESS_RUNNER_H_
