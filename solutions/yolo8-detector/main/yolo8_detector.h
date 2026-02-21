#ifndef _YOLO8_DETECTOR_H_
#define _YOLO8_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <cmath>

#include <sscma.h>

namespace yolo8 {

// Detection result structure
struct Detection {
    float x;            // Normalized center x [0-1]
    float y;            // Normalized center y [0-1]
    float w;            // Normalized width [0-1]
    float h;            // Normalized height [0-1]
    float confidence;   // Detection confidence [0-1]
    int class_id;       // Class index (0-79 for COCO)
    int id;             // Detection ID for tracking
};

// Letterbox info for coordinate transformation
struct LetterboxInfo {
    float scale;        // Scale factor applied to original image
    int pad_left;       // Left padding in pixels
    int pad_top;        // Top padding in pixels
    int new_width;      // Scaled image width (before padding)
    int new_height;     // Scaled image height (before padding)
    int orig_width;     // Original image width
    int orig_height;    // Original image height
};

// COCO class names (80 classes)
extern const char* COCO_CLASSES[80];

/**
 * YOLO8 Detector with DFL (Distribution Focal Loss) post-processing
 *
 * YOLO8 output format:
 *   - bbox tensors: 1x64xHxW (4 coords x 16 DFL bins)
 *   - class tensors: 1x80xHxW (80-class scores)
 *   - 3 scales: 80x80 (stride 8), 40x40 (stride 16), 20x20 (stride 32)
 *
 * DFL decoding:
 *   For each coordinate (left, top, right, bottom):
 *     1. Extract 16 bin values
 *     2. Apply softmax over 16 bins
 *     3. Weighted sum: distance = sum(i * softmax[i]) for i=0..15
 *     4. Convert distances to box coordinates using grid position and stride
 */
class Yolo8Detector {
public:
    Yolo8Detector();
    ~Yolo8Detector();

    // Initialize with model path
    bool init(const std::string& model_path);

    // Set detection parameters
    void setConfThreshold(float threshold);
    void setNmsThreshold(float threshold);

    // Run detection on image
    std::vector<Detection> detect(ma_img_t* img);

    // Get input dimensions
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Get class name
    static const char* getClassName(int class_id);

private:
    // Letterbox preprocessing: resize with aspect ratio preserved, pad to model input size
    void letterboxPreprocess(const ma_img_t* src);

    // Transform detection coordinates from letterbox space to original image space
    void transformCoordinates(std::vector<Detection>& detections);

    // Custom post-processing for YOLO8 with DFL
    void decodeOutputs(std::vector<Detection>& results);

    // Apply Non-Maximum Suppression
    void applyNMS(std::vector<Detection>& detections);

    // Compute DFL: softmax over bins then weighted sum
    float computeDFL(const float* bins) const;

    // Sigmoid activation
    inline float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Inverse sigmoid for threshold pre-computation
    inline float inverseSigmoid(float y) const {
        return -std::log(1.0f / y - 1.0f);
    }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_tensor_;  // Input tensor reference
    ma_img_t img_;              // Internal image buffer for preprocessing
    std::vector<uint8_t> letterbox_buffer_;  // Buffer for letterbox preprocessing
    LetterboxInfo letterbox_info_;           // Current letterbox transformation info
    float conf_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
    bool initialized_;
    int detection_id_counter_;

    // Output tensor count (6 outputs: bbox + cls for 3 scales)
    static constexpr int NUM_OUTPUTS = 6;
    static constexpr int NUM_CLASSES = 80;
    static constexpr int DFL_LEN = 16;        // DFL bin count
    static constexpr int BBOX_CHANNELS = 64;  // 4 coords x 16 bins

    // Grid sizes for each scale
    static constexpr int GRID_SIZES[3] = {80, 40, 20};
    static constexpr int STRIDES[3] = {8, 16, 32};
};

}  // namespace yolo8

#endif  // _YOLO8_DETECTOR_H_
