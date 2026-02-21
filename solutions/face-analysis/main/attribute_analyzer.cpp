#include "attribute_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>

#define TAG "AttributeAnalyzer"

namespace face_analysis {

AttributeAnalyzer::AttributeAnalyzer()
    : genderage_engine_(nullptr),
      genderage_input_size_(96),
      genderage_ready_(false),
      emotion_engine_(nullptr),
      emotion_input_size_(64),
      emotion_ready_(false) {}

AttributeAnalyzer::~AttributeAnalyzer() {
    // Engines will be cleaned up by unique_ptr
}

bool AttributeAnalyzer::init(const std::string& genderage_model,
                              const std::string& emotion_model) {
    // Initialize GenderAge model
    if (!genderage_model.empty()) {
        MA_LOGI(TAG, "Loading GenderAge model: %s", genderage_model.c_str());

        genderage_engine_ = std::make_unique<ma::engine::EngineCVI>();
        ma_err_t ret = genderage_engine_->init();
        if (ret != MA_OK) {
            MA_LOGE(TAG, "Failed to initialize GenderAge engine");
            return false;
        }

        ret = genderage_engine_->load(genderage_model.c_str());
        if (ret != MA_OK) {
            MA_LOGE(TAG, "Failed to load GenderAge model");
            return false;
        }

        // Get input size from model using getInputShape
        ma_shape_t input_shape = genderage_engine_->getInputShape(0);
        if (input_shape.size >= 3) {
            genderage_input_size_ = input_shape.dims[2];  // Assuming NCHW format
        }

        MA_LOGI(TAG, "GenderAge model loaded, input size: %d", genderage_input_size_);
        genderage_ready_ = true;
    }

    // Initialize Emotion model
    if (!emotion_model.empty()) {
        MA_LOGI(TAG, "Loading Emotion model: %s", emotion_model.c_str());

        emotion_engine_ = std::make_unique<ma::engine::EngineCVI>();
        ma_err_t ret = emotion_engine_->init();
        if (ret != MA_OK) {
            MA_LOGE(TAG, "Failed to initialize Emotion engine");
            // Continue without emotion - it's optional
        } else {
            ret = emotion_engine_->load(emotion_model.c_str());
            if (ret != MA_OK) {
                MA_LOGE(TAG, "Failed to load Emotion model");
            } else {
                // Get input size using getInputShape
                ma_shape_t input_shape = emotion_engine_->getInputShape(0);
                if (input_shape.size >= 3) {
                    emotion_input_size_ = input_shape.dims[2];
                }
                MA_LOGI(TAG, "Emotion model loaded, input size: %d", emotion_input_size_);
                emotion_ready_ = true;
            }
        }
    }

    // Allocate crop buffer (max size for any model)
    int max_size = std::max(genderage_input_size_, emotion_input_size_);
    crop_buffer_.resize(max_size * max_size * 3);  // RGB format

    return genderage_ready_;  // At least GenderAge must be ready
}

bool AttributeAnalyzer::cropFace(ma_img_t* full_frame, const FaceInfo& face,
                                  ma_img_t* output, int target_width, int target_height) {
    if (!full_frame || !output) return false;

    // Calculate absolute coordinates
    int frame_width = full_frame->width;
    int frame_height = full_frame->height;

    int x = static_cast<int>(face.x * frame_width);
    int y = static_cast<int>(face.y * frame_height);
    int w = static_cast<int>(face.w * frame_width);
    int h = static_cast<int>(face.h * frame_height);

    // Add margin (10%)
    int margin_x = w / 10;
    int margin_y = h / 10;
    x = std::max(0, x - margin_x);
    y = std::max(0, y - margin_y);
    w = std::min(frame_width - x, w + 2 * margin_x);
    h = std::min(frame_height - y, h + 2 * margin_y);

    // Use OpenCV for cropping and resizing
    cv::Mat src_mat(frame_height, frame_width, CV_8UC3, full_frame->data);
    cv::Rect roi(x, y, w, h);
    cv::Mat cropped = src_mat(roi);

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(target_width, target_height));

    // Copy to output buffer
    output->data = crop_buffer_.data();
    output->width = target_width;
    output->height = target_height;
    output->size = target_width * target_height * 3;
    output->format = MA_PIXEL_FORMAT_RGB888;

    std::memcpy(output->data, resized.data, output->size);

    return true;
}

void AttributeAnalyzer::runGenderAge(ma_img_t* face_crop, FaceAttributes& attrs) {
    if (!genderage_ready_ || !face_crop) return;

    // Prepare input tensor
    ma_tensor_t input = {
        .size = static_cast<size_t>(face_crop->size),
        .is_physical = false,
        .is_variable = false,
    };
    input.data.data = face_crop->data;

    genderage_engine_->setInput(0, input);
    ma_err_t ret = genderage_engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "GenderAge inference failed");
        return;
    }

    // Get output using getOutput(0)
    ma_tensor_t output = genderage_engine_->getOutput(0);
    if (!output.data.data) return;

    // InsightFace GenderAge output format:
    // Output 0: [gender_female_prob, gender_male_prob] or [gender_score]
    // Output 1: [age] or combined in output 0
    float* data = static_cast<float*>(output.data.data);

    // Parse based on output structure (this may vary by model version)
    // Typical InsightFace format: [gender, age]
    if (output.shape.dims[1] >= 2) {
        // Gender: negative = male, positive = female (or probability based)
        float gender_score = data[0];
        if (gender_score > 0) {
            attrs.gender = "female";
            attrs.gender_confidence = std::min(1.0f, gender_score);
        } else {
            attrs.gender = "male";
            attrs.gender_confidence = std::min(1.0f, -gender_score);
        }

        // Age: typically in range [0-100]
        attrs.age = static_cast<int>(std::round(data[1]));
        attrs.age = std::max(0, std::min(100, attrs.age));
        attrs.age_confidence = 0.8f;  // Approximate confidence
    }
}

void AttributeAnalyzer::runEmotion(ma_img_t* face_crop, FaceAttributes& attrs) {
    if (!emotion_ready_ || !face_crop) return;

    // Note: FER+ typically expects grayscale input
    // This implementation assumes the model handles RGB or preprocessing is done

    ma_tensor_t input = {
        .size = static_cast<size_t>(face_crop->size),
        .is_physical = false,
        .is_variable = false,
    };
    input.data.data = face_crop->data;

    emotion_engine_->setInput(0, input);
    ma_err_t ret = emotion_engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Emotion inference failed");
        return;
    }

    // Get output using getOutput(0)
    ma_tensor_t output = emotion_engine_->getOutput(0);
    if (!output.data.data) return;

    float* probs = static_cast<float*>(output.data.data);
    int num_emotions = std::min(8, static_cast<int>(output.shape.dims[1]));

    // Softmax normalization
    float max_prob = *std::max_element(probs, probs + num_emotions);
    float sum = 0.0f;
    for (int i = 0; i < num_emotions; i++) {
        attrs.emotion_probs[i] = std::exp(probs[i] - max_prob);
        sum += attrs.emotion_probs[i];
    }
    for (int i = 0; i < num_emotions; i++) {
        attrs.emotion_probs[i] /= sum;
    }

    // Find dominant emotion
    int max_idx = 0;
    float max_val = attrs.emotion_probs[0];
    for (int i = 1; i < num_emotions; i++) {
        if (attrs.emotion_probs[i] > max_val) {
            max_val = attrs.emotion_probs[i];
            max_idx = i;
        }
    }

    attrs.emotion = static_cast<Emotion>(max_idx);
    attrs.emotion_confidence = max_val;
}

FaceAttributes AttributeAnalyzer::analyze(ma_img_t* face_crop) {
    FaceAttributes attrs = {};
    attrs.age = -1;
    attrs.gender = "unknown";
    attrs.emotion = Emotion::NEUTRAL;

    if (!face_crop) return attrs;

    // Run GenderAge model
    if (genderage_ready_) {
        ma_img_t resized;
        resized.data = crop_buffer_.data();
        resized.width = genderage_input_size_;
        resized.height = genderage_input_size_;
        resized.size = genderage_input_size_ * genderage_input_size_ * 3;
        resized.format = MA_PIXEL_FORMAT_RGB888;

        // Use OpenCV for resize
        cv::Mat src(face_crop->height, face_crop->width, CV_8UC3, face_crop->data);
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(genderage_input_size_, genderage_input_size_));
        std::memcpy(resized.data, dst.data, resized.size);

        runGenderAge(&resized, attrs);
    }

    // Run Emotion model
    if (emotion_ready_) {
        ma_img_t resized;
        resized.data = crop_buffer_.data();
        resized.width = emotion_input_size_;
        resized.height = emotion_input_size_;
        resized.size = emotion_input_size_ * emotion_input_size_ * 3;
        resized.format = MA_PIXEL_FORMAT_RGB888;

        // Use OpenCV for resize
        cv::Mat src(face_crop->height, face_crop->width, CV_8UC3, face_crop->data);
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(emotion_input_size_, emotion_input_size_));
        std::memcpy(resized.data, dst.data, resized.size);

        runEmotion(&resized, attrs);
    }

    return attrs;
}

std::vector<AnalyzedFace> AttributeAnalyzer::analyzeAll(
    ma_img_t* full_frame,
    const std::vector<FaceInfo>& faces) {

    std::vector<AnalyzedFace> results;
    results.reserve(faces.size());

    for (const auto& face : faces) {
        AnalyzedFace analyzed;
        analyzed.face = face;

        // Crop face from full frame
        ma_img_t face_crop;
        int crop_size = std::max(genderage_input_size_, emotion_input_size_);
        if (cropFace(full_frame, face, &face_crop, crop_size, crop_size)) {
            analyzed.attributes = analyze(&face_crop);
        }

        results.push_back(analyzed);
    }

    return results;
}

}  // namespace face_analysis
