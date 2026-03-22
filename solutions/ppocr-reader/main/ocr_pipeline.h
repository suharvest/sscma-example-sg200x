#ifndef _OCR_PIPELINE_H_
#define _OCR_PIPELINE_H_

#include <vector>
#include <string>

#include <sscma.h>

#include "text_detector.h"
#include "text_recognizer.h"

namespace ppocr {

enum class EnhanceMode {
    kNone,      // No enhancement (baseline)
    kClahe,     // CLAHE on LAB L-channel + sharpen (good for solid backgrounds)
    kGray,      // Grayscale → CLAHE → 3-channel + sharpen (good for colored backgrounds)
    kAdaptive,  // Auto-select clahe/gray based on crop saturation
};

struct OcrResult {
    TextBox box;
    std::string text;
    float det_confidence;
    float rec_confidence;
};

struct OcrTimings {
    float detection_ms;
    float recognition_ms;
    float total_ms;
};

class OcrPipeline {
public:
    OcrPipeline();
    ~OcrPipeline();

    bool init(const std::string& det_model_path,
              const std::string& rec_model_path,
              const std::string& dict_path);
    void setMaxBoxes(size_t max_boxes);
    void setEnhanceMode(EnhanceMode mode);

    // Run full OCR pipeline on a camera frame
    std::vector<OcrResult> process(ma_img_t* img, OcrTimings& timings);

private:
    // Crop and perspective-correct text region from image
    bool cropTextRegion(const ma_img_t* img, const TextBox& box,
                        std::vector<uint8_t>& output, int& out_w, int& out_h);

    // Sort boxes top-to-bottom, left-to-right
    void sortBoxes(std::vector<TextBox>& boxes);

    TextDetector detector_;
    TextRecognizer recognizer_;
    std::vector<uint8_t> crop_buffer_;
    size_t max_boxes_;
    EnhanceMode enhance_mode_;

    bool initialized_;
    bool rec_available_;

    // Temporal smoothing: keep previous results for hysteresis
    std::vector<OcrResult> prev_results_;
    int prev_match_count_;  // how many consecutive frames the current text held
};

}  // namespace ppocr

#endif  // _OCR_PIPELINE_H_
