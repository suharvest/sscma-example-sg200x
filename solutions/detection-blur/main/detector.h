#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include <sscma.h>

namespace detection_blur {

struct DetectionBox {
    float x;        // Normalized center x [0-1]
    float y;        // Normalized center y [0-1]
    float w;        // Normalized width [0-1]
    float h;        // Normalized height [0-1]
    float score;    // Detection confidence [0-1]
    int target;     // Class/target ID
};

class Detector {
public:
    Detector();
    ~Detector();

    // Initialize the detector with model path
    bool init(const std::string& model_path);

    // Set detection threshold
    void setThreshold(float threshold);

    // Run detection on the given image
    std::vector<DetectionBox> detect(ma_img_t* img);

    // Get input dimensions required by the model
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

    bool isInitialized() const { return initialized_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma::model::Detector* detector_;
    float threshold_;
    int input_width_;
    int input_height_;
    bool initialized_;
};

}  // namespace detection_blur

#endif  // _DETECTOR_H_
