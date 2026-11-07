#ifndef _FACE_RECOGNITION_SCRFD_RUNNER_H_
#define _FACE_RECOGNITION_SCRFD_RUNNER_H_

#include <string>
#include <vector>

#include "cvi_runner.h"
#include "face_types.h"

namespace face_recognition {

// SCRFD-500M, nine raw heads, decoded on the CPU.
//
// ModelFactory is not used on purpose: the checked-out sscma-micro factory has
// no SCRFD dispatch case, so going through it would silently pick some other
// post-processor.
//
// Outputs are matched by SHAPE, never by name -- the exporter writes textual
// names that vary between conversions while the shapes are fixed by the
// architecture:
//     rows 12800 / 3200 / 800  -> stride 8 / 16 / 32  (2 anchors per cell)
//     cols 1 / 4 / 10          -> score / bbox delta / 5-point kps delta
class ScrfdRunner {
public:
    bool init(const std::string& model_path, std::string* error = nullptr);
    bool ready() const { return ready_; }

    void setScoreThreshold(float v) { score_threshold_ = v; }
    void setNmsThreshold(float v) { nms_threshold_ = v; }

    int inputSize() const { return input_size_; }
    std::string inputDescription() const { return runner_.inputDescription(); }
    std::string outputDescription() const { return output_desc_; }

    // `rgb` is a packed HWC RGB888 frame of `width` x `height`. It is
    // letterboxed (aspect preserved, centre padded with 128 -- the same grey the
    // fp32 reference pads with) into the model's square input, and every
    // coordinate returned is mapped back into source pixels.
    bool infer(const uint8_t* rgb, int width, int height,
               std::vector<FaceDetection>& out, std::string* error = nullptr);

private:
    struct HeadRef {
        int tensor_index = -1;
        int rows = 0;
        int cols = 0;
        bool nchw = false;   // needs (H,W,anchor,component) unfolding
        int H = 0, W = 0;
    };

    bool mapOutputs(std::string* error);
    // Reads logical element (row, col) of a head, undoing NCHW packing.
    float headValue(const ma_tensor_t& t, const HeadRef& h, int row, int col) const;

    CviRunner runner_;
    bool ready_ = false;
    int input_size_ = 640;
    float score_threshold_ = 0.5f;
    float nms_threshold_ = 0.4f;
    std::string output_desc_;

    // [stride_index][kind] where stride_index 0..2 -> 8/16/32 and
    // kind 0..2 -> scores/bboxes/kps.
    HeadRef heads_[3][3];

    std::vector<uint8_t> letterbox_;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_SCRFD_RUNNER_H_
