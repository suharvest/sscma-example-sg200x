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

        // Log full tensor info for debugging
        auto ga_input_tensor = genderage_engine_->getInput(0);
        MA_LOGI(TAG, "GenderAge input: ndim=%d dims=[%d,%d,%d,%d] type=%d size=%zu",
                input_shape.size, input_shape.dims[0], input_shape.dims[1],
                input_shape.dims[2], input_shape.dims[3],
                ga_input_tensor.type, ga_input_tensor.size);
        MA_LOGI(TAG, "GenderAge input quant: scale=%.6f zp=%d",
                ga_input_tensor.quant_param.scale, ga_input_tensor.quant_param.zero_point);

        int ga_num_outputs = genderage_engine_->getOutputSize();
        for (int i = 0; i < ga_num_outputs; i++) {
            auto oshape = genderage_engine_->getOutputShape(i);
            auto otensor = genderage_engine_->getOutput(i);
            MA_LOGI(TAG, "GenderAge output[%d]: ndim=%d dims=[%d,%d,%d,%d] type=%d size=%zu",
                    i, oshape.size, oshape.dims[0], oshape.dims[1],
                    oshape.dims[2], oshape.dims[3],
                    otensor.type, otensor.size);
            MA_LOGI(TAG, "GenderAge output[%d] quant: scale=%.6f zp=%d",
                    i, otensor.quant_param.scale, otensor.quant_param.zero_point);
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
                auto em_input_tensor = emotion_engine_->getInput(0);
                MA_LOGI(TAG, "Emotion input: ndim=%d dims=[%d,%d,%d,%d] type=%d size=%zu",
                        input_shape.size, input_shape.dims[0], input_shape.dims[1],
                        input_shape.dims[2], input_shape.dims[3],
                        em_input_tensor.type, em_input_tensor.size);
                MA_LOGI(TAG, "Emotion input quant: scale=%.6f zp=%d",
                        em_input_tensor.quant_param.scale, em_input_tensor.quant_param.zero_point);

                int em_num_outputs = emotion_engine_->getOutputSize();
                for (int j = 0; j < em_num_outputs; j++) {
                    auto oshape = emotion_engine_->getOutputShape(j);
                    auto otensor = emotion_engine_->getOutput(j);
                    MA_LOGI(TAG, "Emotion output[%d]: ndim=%d dims=[%d,%d,%d,%d] type=%d size=%zu",
                            j, oshape.size, oshape.dims[0], oshape.dims[1],
                            oshape.dims[2], oshape.dims[3],
                            otensor.type, otensor.size);
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
    if (!full_frame || !full_frame->data || !output) return false;
    if (full_frame->width <= 0 || full_frame->height <= 0) return false;

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

    // Validate ROI
    if (w <= 0 || h <= 0) return false;

    // Use OpenCV for cropping and resizing
    cv::Mat src_mat(frame_height, frame_width, CV_8UC3, full_frame->data);
    cv::Rect roi(x, y, w, h);
    roi &= cv::Rect(0, 0, frame_width, frame_height);
    if (roi.width <= 0 || roi.height <= 0) return false;
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

    // Get the engine's own input tensor and write directly to its buffer
    // This is how the Detector base class does it - write to engine's internal buffer
    ma_tensor_t engine_input = genderage_engine_->getInput(0);
    if (!engine_input.data.data) {
        MA_LOGE(TAG, "GenderAge engine input buffer is null");
        return;
    }

    // Copy planar RGB data directly into the engine's input buffer
    size_t copy_size = std::min(static_cast<size_t>(face_crop->size), engine_input.size);
    std::memcpy(engine_input.data.data, face_crop->data, copy_size);

    // Log first few bytes of input for diagnostics (only on first call)
    static bool first_call = true;
    if (first_call) {
        uint8_t* buf = static_cast<uint8_t*>(engine_input.data.data);
        MA_LOGD(TAG, "GenderAge input buffer: size=%zu, copy_size=%zu, first 12 bytes: %d %d %d %d %d %d %d %d %d %d %d %d",
                engine_input.size, copy_size,
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5],
                buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]);
        first_call = false;
    }

    ma_err_t ret = genderage_engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "GenderAge inference failed");
        return;
    }

    // Get output using getOutput(0)
    ma_tensor_t output = genderage_engine_->getOutput(0);
    if (!output.data.data) return;

    float* data = static_cast<float*>(output.data.data);

    // Diagnostic: dump output shape and first values
    int total_elements = 1;
    for (int d = 0; d < static_cast<int>(output.shape.size); d++) {
        total_elements *= output.shape.dims[d];
    }
    int dump_count = std::min(total_elements, 10);
    MA_LOGD(TAG, "GenderAge output shape=[%d,%d,%d,%d] type=%d, first %d values:",
            output.shape.dims[0], output.shape.dims[1], output.shape.dims[2], output.shape.dims[3],
            output.type, dump_count);
    for (int i = 0; i < dump_count; i++) {
        MA_LOGD(TAG, "  data[%d] = %.6f", i, data[i]);
    }

    // InsightFace GenderAge model output: fc1 = concat(fullyconnected0, fullyconnected1)
    // fullyconnected0: [2] = gender logits (female, male)
    // fullyconnected1: [1] = age (normalized, multiply by 100 for years)
    // So: data[0] = female logit, data[1] = male logit, data[2] = age/100
    if (total_elements >= 3) {
        float female_logit = data[0];
        float male_logit = data[1];

        // Apply softmax to get probabilities
        float max_logit = std::max(female_logit, male_logit);
        float exp_female = std::exp(female_logit - max_logit);
        float exp_male = std::exp(male_logit - max_logit);
        float sum = exp_female + exp_male;
        float prob_female = exp_female / sum;
        float prob_male = exp_male / sum;

        if (prob_female > prob_male) {
            attrs.gender = "female";
            attrs.gender_confidence = prob_female;
        } else {
            attrs.gender = "male";
            attrs.gender_confidence = prob_male;
        }

        // Age is in data[2], normalized to ~0-1 range, multiply by 100
        float raw_age = data[2];
        attrs.age = static_cast<int>(std::round(raw_age * 100.0f));
        attrs.age = std::max(0, std::min(100, attrs.age));
        attrs.age_confidence = 0.8f;

        MA_LOGD(TAG, "GenderAge interpreted: female_logit=%.3f male_logit=%.3f prob_f=%.3f prob_m=%.3f raw_age=%.4f age=%d",
                female_logit, male_logit, prob_female, prob_male, raw_age, attrs.age);
    }
}

void AttributeAnalyzer::runEmotion(ma_img_t* face_crop, FaceAttributes& attrs) {
    if (!emotion_ready_ || !face_crop) return;

    // Write directly to engine's input buffer (same approach as GenderAge)
    ma_tensor_t engine_input = emotion_engine_->getInput(0);
    if (!engine_input.data.data) {
        MA_LOGE(TAG, "Emotion engine input buffer is null");
        return;
    }

    size_t copy_size = std::min(static_cast<size_t>(face_crop->size), engine_input.size);
    std::memcpy(engine_input.data.data, face_crop->data, copy_size);

    ma_err_t ret = emotion_engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Emotion inference failed");
        return;
    }

    // Get output using getOutput(0)
    ma_tensor_t output = emotion_engine_->getOutput(0);
    if (!output.data.data) return;

    float* probs = static_cast<float*>(output.data.data);
    // Output shape is [1,8,1,1] — 8 emotion classes
    int total_elements = 1;
    for (int d = 0; d < static_cast<int>(output.shape.size); d++) {
        total_elements *= output.shape.dims[d];
    }
    int num_emotions = std::min(8, total_elements);

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

    if (!face_crop || !face_crop->data || face_crop->width <= 0 || face_crop->height <= 0) return attrs;

    // Deep copy the face crop since we'll overwrite crop_buffer_ during format conversion
    cv::Mat face_rgb = cv::Mat(face_crop->height, face_crop->width, CV_8UC3, face_crop->data).clone();

    // Run GenderAge model — input is NCHW [1,3,96,96], needs planar RGB
    if (genderage_ready_) {
        cv::Mat resized;
        cv::resize(face_rgb, resized, cv::Size(genderage_input_size_, genderage_input_size_));

        // Convert interleaved RGB to planar RGB (NCHW layout) in crop_buffer_
        size_t plane_size = genderage_input_size_ * genderage_input_size_;
        uint8_t* buf = crop_buffer_.data();
        for (int y = 0; y < genderage_input_size_; y++) {
            for (int x = 0; x < genderage_input_size_; x++) {
                const uint8_t* pixel = resized.ptr(y) + x * 3;
                buf[0 * plane_size + y * genderage_input_size_ + x] = pixel[0];  // R
                buf[1 * plane_size + y * genderage_input_size_ + x] = pixel[1];  // G
                buf[2 * plane_size + y * genderage_input_size_ + x] = pixel[2];  // B
            }
        }

        ma_img_t ga_input;
        ga_input.data = buf;
        ga_input.width = genderage_input_size_;
        ga_input.height = genderage_input_size_;
        ga_input.size = plane_size * 3;
        ga_input.format = MA_PIXEL_FORMAT_RGB888_PLANAR;

        runGenderAge(&ga_input, attrs);
    }

    // Run Emotion model — input is NCHW [1,1,64,64], needs grayscale
    if (emotion_ready_) {
        cv::Mat gray;
        cv::cvtColor(face_rgb, gray, cv::COLOR_RGB2GRAY);
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(emotion_input_size_, emotion_input_size_));

        // Copy grayscale to crop_buffer_
        size_t gray_size = emotion_input_size_ * emotion_input_size_;
        std::memcpy(crop_buffer_.data(), resized.data, gray_size);

        ma_img_t em_input;
        em_input.data = crop_buffer_.data();
        em_input.width = emotion_input_size_;
        em_input.height = emotion_input_size_;
        em_input.size = gray_size;
        em_input.format = MA_PIXEL_FORMAT_GRAYSCALE;

        runEmotion(&em_input, attrs);
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
