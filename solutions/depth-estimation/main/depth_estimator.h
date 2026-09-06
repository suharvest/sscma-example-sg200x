#ifndef _DEPTH_ESTIMATOR_H_
#define _DEPTH_ESTIMATOR_H_

#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

namespace depth {

/* A rectangle in the inference frame, in pixels. */
struct Roi {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

/*
 * Compute the part of `frame` that actually carries sensor content.
 *
 * The camera's VPSS fits the 16:9 sensor picture into every channel preserving
 * aspect (ASPECT_RATIO_AUTO with bEnableBgColor, hard-coded in
 * components/sophgo/video/src/video_paramparse.c:99-104) and pads the rest with
 * grey 0x727272. A channel whose aspect is not 16:9 therefore hands over bars,
 * and those bars are out-of-distribution input for a depth model trained on
 * full-frame photographs: they come back with invented depth that also drags the
 * frame-wide normalisation around. So they are removed before inference, never
 * after.
 *
 * With a 16:9 channel this returns the whole frame and costs nothing.
 */
Roi valid_content_roi(int frame_width, int frame_height);

/*
 * FastDepth (or any single-input, single-output dense depth model) on the CVI
 * TPU.
 *
 * Output values are relative depth in the model's own units: smaller is nearer.
 * Nothing here converts them to a distance, and nothing downstream should.
 */
class DepthEstimator {
public:
    DepthEstimator();
    ~DepthEstimator();

    DepthEstimator(const DepthEstimator&)            = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;

    /* Loads the model, prints the real tensor names/shapes/dtypes and refuses
     * to continue if the shapes are not the 1x3xHxW in / HxW out this
     * application knows how to feed. */
    bool init(const std::string& model_path);

    /* Preprocess `roi` of an RGB888 frame into the input tensor (stretch to the
     * model's input size, RGB, CHW, scaled to [0,1]) and run one forward pass.
     * Returns false on any failure; depth() then keeps its previous contents. */
    bool run(const ma_img_t* frame, const Roi& roi);

    /* Relative depth, row-major, outputWidth() * outputHeight() values. */
    const std::vector<float>& depth() const { return depth_; }

    int inputWidth() const { return input_w_; }
    int inputHeight() const { return input_h_; }
    int outputWidth() const { return output_w_; }
    int outputHeight() const { return output_h_; }
    float lastInferenceMs() const { return last_inference_ms_; }
    /* Breakdown of the same interval, for -v profiling: preprocess (crop +
     * stretch + CHW + scale), the TPU forward pass, and copying the output
     * tensor out. They sum to lastInferenceMs(). */
    float lastPreprocessMs() const { return last_preprocess_ms_; }
    float lastForwardMs() const { return last_forward_ms_; }
    float lastReadoutMs() const { return last_readout_ms_; }
    bool initialized() const { return initialized_; }

private:
    bool preprocess(const ma_img_t* frame, const Roi& roi);
    bool readOutput();

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_{};
    ma_tensor_t output_{};
    std::vector<float> depth_;
    int input_w_           = 0;
    int input_h_           = 0;
    int output_w_          = 0;
    int output_h_          = 0;
    float last_inference_ms_ = 0.0f;
    float last_preprocess_ms_ = 0.0f;
    float last_forward_ms_ = 0.0f;
    float last_readout_ms_ = 0.0f;
    bool initialized_      = false;
};

}  // namespace depth

#endif  // _DEPTH_ESTIMATOR_H_
