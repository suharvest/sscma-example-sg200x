#include "yolo11s_detector.h"

#include <algorithm>
#include <cstring>
#include <cmath>

#define TAG "Yolo11sDetector"

namespace yolo11s {

// COCO class names
const char* COCO_CLASSES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

constexpr int Yolo11sDetector::GRID_SIZES[3];
constexpr int Yolo11sDetector::STRIDES[3];

Yolo11sDetector::Yolo11sDetector()
    : engine_(nullptr),
      conf_threshold_(0.25f),
      nms_threshold_(0.45f),
      input_width_(640),
      input_height_(640),
      initialized_(false),
      detection_id_counter_(0) {}

Yolo11sDetector::~Yolo11sDetector() {
}

bool Yolo11sDetector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Initializing YOLO11s detector with model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    ma_err_t ret = engine_->init();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize CVI engine");
        return false;
    }

    ret = engine_->load(model_path.c_str());
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    int output_count = engine_->getOutputSize();
    if (output_count != NUM_OUTPUTS) {
        MA_LOGW(TAG, "Expected %d outputs, got %d", NUM_OUTPUTS, output_count);
    }

    input_tensor_ = engine_->getInput(0);
    ma_shape_t input_shape = engine_->getInputShape(0);

    bool is_nhwc = (input_shape.dims[3] == 3 || input_shape.dims[3] == 1);
    if (is_nhwc) {
        input_height_ = input_shape.dims[1];
        input_width_ = input_shape.dims[2];
    } else {
        input_height_ = input_shape.dims[2];
        input_width_ = input_shape.dims[3];
    }

    img_.width = input_width_;
    img_.height = input_height_;
    img_.format = MA_PIXEL_FORMAT_RGB888;
    img_.size = input_width_ * input_height_ * 3;
    img_.data = input_tensor_.data.u8;

    MA_LOGI(TAG, "YOLO11s detector initialized");
    MA_LOGI(TAG, "  Input size: %dx%d", input_width_, input_height_);
    MA_LOGI(TAG, "  Output count: %d", output_count);
    MA_LOGI(TAG, "  Confidence threshold: %.2f", conf_threshold_);
    MA_LOGI(TAG, "  NMS threshold: %.2f", nms_threshold_);
    MA_LOGI(TAG, "  DFL bins: %d, BBox channels: %d", DFL_LEN, BBOX_CHANNELS);

    initialized_ = true;
    return true;
}

void Yolo11sDetector::setConfThreshold(float threshold) {
    conf_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

void Yolo11sDetector::setNmsThreshold(float threshold) {
    nms_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

const char* Yolo11sDetector::getClassName(int class_id) {
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return COCO_CLASSES[class_id];
    }
    return "unknown";
}

float Yolo11sDetector::computeDFL(const float* bins) const {
    float max_val = bins[0];
    for (int i = 1; i < DFL_LEN; i++) {
        if (bins[i] > max_val) max_val = bins[i];
    }

    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < DFL_LEN; i++) {
        float exp_val = std::exp(bins[i] - max_val);
        sum_exp += exp_val;
        weighted_sum += static_cast<float>(i) * exp_val;
    }

    return weighted_sum / sum_exp;
}

void Yolo11sDetector::letterboxPreprocess(const ma_img_t* src) {
    float scale_w = static_cast<float>(input_width_) / src->width;
    float scale_h = static_cast<float>(input_height_) / src->height;
    float scale = std::min(scale_w, scale_h);

    int new_width = static_cast<int>(src->width * scale);
    int new_height = static_cast<int>(src->height * scale);

    int pad_left = (input_width_ - new_width) / 2;
    int pad_top = (input_height_ - new_height) / 2;

    letterbox_info_.scale = scale;
    letterbox_info_.pad_left = pad_left;
    letterbox_info_.pad_top = pad_top;
    letterbox_info_.new_width = new_width;
    letterbox_info_.new_height = new_height;
    letterbox_info_.orig_width = src->width;
    letterbox_info_.orig_height = src->height;

    size_t buffer_size = input_width_ * input_height_ * 3;
    if (letterbox_buffer_.size() != buffer_size) {
        letterbox_buffer_.resize(buffer_size);
    }

    std::memset(letterbox_buffer_.data(), 128, buffer_size);

    uint32_t beta_w = (static_cast<uint32_t>(src->width) << 16) / new_width;
    uint32_t beta_h = (static_cast<uint32_t>(src->height) << 16) / new_height;

    const uint8_t* src_data = src->data;
    uint8_t* dst_data = letterbox_buffer_.data();

    if (src->format == MA_PIXEL_FORMAT_RGB888) {
        for (int i = 0; i < new_height; ++i) {
            int src_y = (i * beta_h) >> 16;
            int dst_row = (i + pad_top) * input_width_;

            for (int j = 0; j < new_width; ++j) {
                int src_x = (j * beta_w) >> 16;
                int src_idx = (src_y * src->width + src_x) * 3;
                int dst_idx = (dst_row + j + pad_left) * 3;

                dst_data[dst_idx + 0] = src_data[src_idx + 0];
                dst_data[dst_idx + 1] = src_data[src_idx + 1];
                dst_data[dst_idx + 2] = src_data[src_idx + 2];
            }
        }
    } else {
        ma_img_t scaled_img;
        scaled_img.width = new_width;
        scaled_img.height = new_height;
        scaled_img.format = MA_PIXEL_FORMAT_RGB888;
        scaled_img.size = new_width * new_height * 3;
        scaled_img.rotate = MA_PIXEL_ROTATE_0;

        std::vector<uint8_t> scaled_buffer(scaled_img.size);
        scaled_img.data = scaled_buffer.data();

        ma::cv::convert(src, &scaled_img);

        for (int i = 0; i < new_height; ++i) {
            int dst_row = (i + pad_top) * input_width_;
            int src_row = i * new_width;

            for (int j = 0; j < new_width; ++j) {
                int src_idx = (src_row + j) * 3;
                int dst_idx = (dst_row + j + pad_left) * 3;

                dst_data[dst_idx + 0] = scaled_buffer[src_idx + 0];
                dst_data[dst_idx + 1] = scaled_buffer[src_idx + 1];
                dst_data[dst_idx + 2] = scaled_buffer[src_idx + 2];
            }
        }
    }

    std::memcpy(input_tensor_.data.u8, letterbox_buffer_.data(), buffer_size);
}

void Yolo11sDetector::transformCoordinates(std::vector<Detection>& detections) {
    for (auto& det : detections) {
        float px = det.x * input_width_;
        float py = det.y * input_height_;
        float pw = det.w * input_width_;
        float ph = det.h * input_height_;

        px -= letterbox_info_.pad_left;
        py -= letterbox_info_.pad_top;

        px /= letterbox_info_.scale;
        py /= letterbox_info_.scale;
        pw /= letterbox_info_.scale;
        ph /= letterbox_info_.scale;

        det.x = px / letterbox_info_.orig_width;
        det.y = py / letterbox_info_.orig_height;
        det.w = pw / letterbox_info_.orig_width;
        det.h = ph / letterbox_info_.orig_height;

        det.x = std::max(0.0f, std::min(1.0f, det.x));
        det.y = std::max(0.0f, std::min(1.0f, det.y));

        float half_w = det.w / 2.0f;
        float half_h = det.h / 2.0f;
        float x1 = std::max(0.0f, det.x - half_w);
        float y1 = std::max(0.0f, det.y - half_h);
        float x2 = std::min(1.0f, det.x + half_w);
        float y2 = std::min(1.0f, det.y + half_h);

        det.x = (x1 + x2) / 2.0f;
        det.y = (y1 + y2) / 2.0f;
        det.w = x2 - x1;
        det.h = y2 - y1;
    }
}

std::vector<Detection> Yolo11sDetector::detect(ma_img_t* img) {
    std::vector<Detection> detections;

    if (!initialized_ || !img) {
        return detections;
    }

    letterboxPreprocess(img);

    if (input_tensor_.type == MA_TENSOR_TYPE_S8) {
        for (size_t i = 0; i < input_tensor_.size; i++) {
            input_tensor_.data.u8[i] -= 128;
        }
    }

    ma_err_t ret = engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Inference failed with error: %d", ret);
        return detections;
    }

    decodeOutputs(detections);
    applyNMS(detections);
    transformCoordinates(detections);

    for (auto& det : detections) {
        det.id = detection_id_counter_++;
    }

    return detections;
}

void Yolo11sDetector::decodeOutputs(std::vector<Detection>& results) {
    results.clear();

    const float score_threshold_logit = inverseSigmoid(conf_threshold_);

    int bbox_indices[3] = {-1, -1, -1};
    int cls_indices[3] = {-1, -1, -1};

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        ma_tensor_t tensor = engine_->getOutput(i);
        int n = (tensor.shape.size > 0) ? tensor.shape.dims[0] : 1;
        int c = (tensor.shape.size > 1) ? tensor.shape.dims[1] : 1;
        int h = (tensor.shape.size > 2) ? tensor.shape.dims[2] : 1;
        int w = (tensor.shape.size > 3) ? tensor.shape.dims[3] : 1;
        (void)n;

        int scale_idx = -1;
        if (h == 80 && w == 80) scale_idx = 0;
        else if (h == 40 && w == 40) scale_idx = 1;
        else if (h == 20 && w == 20) scale_idx = 2;

        if (scale_idx >= 0) {
            if (c == BBOX_CHANNELS) {
                bbox_indices[scale_idx] = i;
            } else if (c == NUM_CLASSES) {
                cls_indices[scale_idx] = i;
            }
        }
    }

    for (int scale = 0; scale < 3; scale++) {
        int bbox_idx = bbox_indices[scale];
        int cls_idx = cls_indices[scale];

        if (bbox_idx < 0 || cls_idx < 0) [[unlikely]] {
            MA_LOGW(TAG, "Missing output for scale %d (bbox=%d, cls=%d)",
                    scale, bbox_idx, cls_idx);
            continue;
        }

        int stride = STRIDES[scale];
        int grid_size = GRID_SIZES[scale];
        int spatial_size = grid_size * grid_size;

        ma_tensor_t bbox_tensor = engine_->getOutput(bbox_idx);
        ma_tensor_t cls_tensor = engine_->getOutput(cls_idx);

        bool is_int8 = (bbox_tensor.type == MA_TENSOR_TYPE_S8);

        float bbox_scale = 1.0f;
        int32_t bbox_zp = 0;
        float cls_scale = 1.0f;
        int32_t cls_zp = 0;

        if (is_int8) {
            bbox_scale = bbox_tensor.quant_param.scale;
            bbox_zp = bbox_tensor.quant_param.zero_point;
            cls_scale = cls_tensor.quant_param.scale;
            cls_zp = cls_tensor.quant_param.zero_point;
        }

        for (int j = 0; j < grid_size; j++) {
            for (int k = 0; k < grid_size; k++) {
                int offset = j * grid_size + k;

                int best_class = -1;
                float max_logit = score_threshold_logit;

                if (is_int8) {
                    int8_t* cls_data_s8 = cls_tensor.data.s8;
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        float logit = (static_cast<float>(cls_data_s8[c * spatial_size + offset]) - cls_zp) * cls_scale;
                        if (logit > max_logit) [[unlikely]] {
                            max_logit = logit;
                            best_class = c;
                        }
                    }
                } else {
                    float* cls_data = cls_tensor.data.f32;
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        float logit = cls_data[c * spatial_size + offset];
                        if (logit > max_logit) [[unlikely]] {
                            max_logit = logit;
                            best_class = c;
                        }
                    }
                }

                if (best_class < 0) [[likely]] {
                    continue;
                }

                float confidence = sigmoid(max_logit);

                float dfl_bins[DFL_LEN];
                float distances[4];

                for (int coord = 0; coord < 4; coord++) {
                    if (is_int8) {
                        int8_t* bbox_data_s8 = bbox_tensor.data.s8;
                        for (int b = 0; b < DFL_LEN; b++) {
                            int ch = coord * DFL_LEN + b;
                            float raw = static_cast<float>(bbox_data_s8[ch * spatial_size + offset]);
                            dfl_bins[b] = (raw - bbox_zp) * bbox_scale;
                        }
                    } else {
                        float* bbox_data = bbox_tensor.data.f32;
                        for (int b = 0; b < DFL_LEN; b++) {
                            int ch = coord * DFL_LEN + b;
                            dfl_bins[b] = bbox_data[ch * spatial_size + offset];
                        }
                    }

                    distances[coord] = computeDFL(dfl_bins);
                }

                float x1 = (static_cast<float>(k) + 0.5f - distances[0]) * stride;
                float y1 = (static_cast<float>(j) + 0.5f - distances[1]) * stride;
                float x2 = (static_cast<float>(k) + 0.5f + distances[2]) * stride;
                float y2 = (static_cast<float>(j) + 0.5f + distances[3]) * stride;

                Detection det;
                det.x = (x1 + x2) * 0.5f / static_cast<float>(input_width_);
                det.y = (y1 + y2) * 0.5f / static_cast<float>(input_height_);
                det.w = (x2 - x1) / static_cast<float>(input_width_);
                det.h = (y2 - y1) / static_cast<float>(input_height_);
                det.confidence = confidence;
                det.class_id = best_class;
                det.id = 0;

                det.x = std::max(0.0f, std::min(1.0f, det.x));
                det.y = std::max(0.0f, std::min(1.0f, det.y));
                det.w = std::max(0.0f, std::min(1.0f, det.w));
                det.h = std::max(0.0f, std::min(1.0f, det.h));

                results.push_back(det);
            }
        }
    }
}

void Yolo11sDetector::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) {
        return;
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    auto computeIoU = [](const Detection& a, const Detection& b) -> float {
        float ax1 = a.x - a.w / 2.0f;
        float ay1 = a.y - a.h / 2.0f;
        float ax2 = a.x + a.w / 2.0f;
        float ay2 = a.y + a.h / 2.0f;

        float bx1 = b.x - b.w / 2.0f;
        float by1 = b.y - b.h / 2.0f;
        float bx2 = b.x + b.w / 2.0f;
        float by2 = b.y + b.h / 2.0f;

        float inter_x1 = std::max(ax1, bx1);
        float inter_y1 = std::max(ay1, by1);
        float inter_x2 = std::min(ax2, bx2);
        float inter_y2 = std::min(ay2, by2);

        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float a_area = a.w * a.h;
        float b_area = b.w * b.h;
        float union_area = a_area + b_area - inter_area;

        return (union_area > 0) ? inter_area / union_area : 0.0f;
    };

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            if (detections[i].class_id != detections[j].class_id) continue;

            float iou = computeIoU(detections[i], detections[j]);
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<Detection> kept;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            kept.push_back(detections[i]);
        }
    }
    detections = std::move(kept);
}

}  // namespace yolo11s
