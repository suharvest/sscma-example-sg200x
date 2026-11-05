#include "pose_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <forward_list>

#include <core/utils/ma_nms.h>

#define TAG "PoseDetector"

namespace microduck {

static const char* tensor_type_name(ma_tensor_type_t type) {
    switch (type) {
        case MA_TENSOR_TYPE_F32:  return "F32";
        case MA_TENSOR_TYPE_BF16: return "BF16";
        case MA_TENSOR_TYPE_F16:  return "F16";
        case MA_TENSOR_TYPE_U8:   return "U8";
        case MA_TENSOR_TYPE_S8:   return "S8";
        case MA_TENSOR_TYPE_U16:  return "U16";
        case MA_TENSOR_TYPE_S16:  return "S16";
        case MA_TENSOR_TYPE_U32:  return "U32";
        case MA_TENSOR_TYPE_S32:  return "S32";
        default:                  return "UNKNOWN";
    }
}

PoseDetector::~PoseDetector() {
    engine_.reset();
}

bool PoseDetector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Loading pose model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize CVI engine");
        return false;
    }
    if (engine_->load(model_path.c_str()) != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    ma_tensor_t in = engine_->getInput(0);
    input_type_ = in.type;
    if (in.shape.size != 4) {
        MA_LOGE(TAG, "Unsupported input tensor shape rank: %d (expected 4)", in.shape.size);
        return false;
    }
    if (in.shape.dims[1] == 3) {
        input_channels_ = 3;
        input_height_ = in.shape.dims[2];
        input_width_ = in.shape.dims[3];
        is_nchw_ = true;
    } else if (in.shape.dims[3] == 3) {
        input_height_ = in.shape.dims[1];
        input_width_ = in.shape.dims[2];
        input_channels_ = 3;
        is_nchw_ = false;
    } else {
        MA_LOGE(TAG, "Only 3-channel (RGB) input models are supported, got dims=[%d,%d,%d,%d]",
                in.shape.dims[0], in.shape.dims[1], in.shape.dims[2], in.shape.dims[3]);
        return false;
    }

    ma_tensor_t out = engine_->getOutput(0);
    num_records_ = out.shape.dims[2];
    num_keypoints_ = (out.shape.dims[1] - 5) / 3;

    // Precompute U8 pixel -> INT8 tensor conversion LUT (symmetric [0, 255] -> [0, 127])
    for (int i = 0; i < 256; ++i) {
        float norm = static_cast<float>(i) / 255.0f;
        int val = static_cast<int>(std::lround(norm * 127.0f));
        u8_to_s8_lut_[i] = static_cast<int8_t>(std::clamp(val, -128, 127));
    }

    MA_LOGI(TAG, "Pose model ready: in=[%d,%d,%d] type=%s(%d) scale=%.6f zp=%d, out=[%d,%d] kpts=%d type=%s(%d)",
            input_channels_, input_height_, input_width_, tensor_type_name(input_type_), input_type_,
            in.quant_param.scale, in.quant_param.zero_point,
            out.shape.dims[1], num_records_, num_keypoints_, tensor_type_name(out.type), out.type);

    initialized_ = true;
    return true;
}

void PoseDetector::setThreshold(float robot_score) {
    threshold_ = std::max(0.0f, std::min(1.0f, robot_score));
}

void PoseDetector::setKeypointThreshold(float kpt_score) {
    kpt_threshold_ = std::max(0.0f, std::min(1.0f, kpt_score));
}

const std::vector<Subject>& PoseDetector::detectAll(ma_img_t* img) {
    subjects_.clear();
    has_primary_ = false;
    if (!initialized_ || engine_ == nullptr || img == nullptr || img->data == nullptr) {
        last_run_ok_ = false;
        return subjects_;
    }
    if (img->width != input_width_ || img->height != input_height_) {
        MA_LOGE(TAG, "Input image dimensions mismatch: expected %dx%d, got %dx%d",
                input_width_, input_height_, img->width, img->height);
        last_run_ok_ = false;
        return subjects_;
    }
    if (img->size < static_cast<size_t>(input_width_ * input_height_ * 3)) {
        MA_LOGE(TAG, "Input image buffer too small: size=%zu, expected >= %d",
                img->size, input_width_ * input_height_ * 3);
        last_run_ok_ = false;
        return subjects_;
    }

    ma_tensor_t in = engine_->getInput(0);
    if (in.data.data == nullptr) {
        MA_LOGE(TAG, "Engine input tensor pointer is null");
        last_run_ok_ = false;
        return subjects_;
    }
    const size_t elem_size = (in.type == MA_TENSOR_TYPE_F32) ? sizeof(float) : 1;
    const size_t req_bytes = static_cast<size_t>(input_width_ * input_height_ * 3) * elem_size;
    if (in.size < req_bytes) {
        MA_LOGE(TAG, "Input tensor capacity too small: tensor_size=%zu, required=%zu", in.size, req_bytes);
        last_run_ok_ = false;
        return subjects_;
    }
    if (in.type != MA_TENSOR_TYPE_F32 && in.type != MA_TENSOR_TYPE_U8 && in.type != MA_TENSOR_TYPE_S8) {
        MA_LOGE(TAG, "Unsupported input tensor type: %d", in.type);
        last_run_ok_ = false;
        return subjects_;
    }
    const int plane_size = input_width_ * input_height_;
    const uint8_t* src = img->data;

    static bool logged_first_frame = false;
    if (!logged_first_frame) {
        logged_first_frame = true;
        MA_LOGI(TAG, "First frame input tensor: type=%s(%d) layout=%s [%d,%d,%d]",
                tensor_type_name(in.type), in.type,
                is_nchw_ ? "NCHW" : "NHWC",
                input_channels_, input_height_, input_width_);
    }

    if (in.type == MA_TENSOR_TYPE_F32) {
        float* dst = in.data.f32;
        if (is_nchw_) {
            float* r = dst;
            float* g = dst + plane_size;
            float* b = dst + 2 * plane_size;
            for (int i = 0; i < plane_size; ++i) {
                r[i] = static_cast<float>(src[3 * i + 0]);
                g[i] = static_cast<float>(src[3 * i + 1]);
                b[i] = static_cast<float>(src[3 * i + 2]);
            }
        } else {
            for (int i = 0; i < plane_size * 3; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        }
    } else if (in.type == MA_TENSOR_TYPE_U8) {
        uint8_t* dst = in.data.u8;
        if (is_nchw_) {
            uint8_t* r = dst;
            uint8_t* g = dst + plane_size;
            uint8_t* b = dst + 2 * plane_size;
            for (int i = 0; i < plane_size; ++i) {
                r[i] = src[3 * i + 0];
                g[i] = src[3 * i + 1];
                b[i] = src[3 * i + 2];
            }
        } else {
            std::memcpy(dst, src, plane_size * 3);
        }
    } else if (in.type == MA_TENSOR_TYPE_S8) {
        int8_t* dst = in.data.s8;
        if (is_nchw_) {
            int8_t* r = dst;
            int8_t* g = dst + plane_size;
            int8_t* b = dst + 2 * plane_size;
            for (int i = 0; i < plane_size; ++i) {
                r[i] = u8_to_s8_lut_[src[3 * i + 0]];
                g[i] = u8_to_s8_lut_[src[3 * i + 1]];
                b[i] = u8_to_s8_lut_[src[3 * i + 2]];
            }
        } else {
            for (int i = 0; i < plane_size * 3; ++i) {
                dst[i] = u8_to_s8_lut_[src[i]];
            }
        }
    }

    last_run_ok_ = (engine_->run() == MA_OK);
    if (!last_run_ok_) {
        MA_LOGE(TAG, "Engine inference run failed");
        return subjects_;
    }

    ma_tensor_t out = engine_->getOutput(0);
    if (out.type != MA_TENSOR_TYPE_F32 || out.data.f32 == nullptr) {
        last_run_ok_ = false;
        MA_LOGE(TAG, "Unsupported or null output tensor: type=%d", out.type);
        return subjects_;
    }

    const float* data = out.data.f32;
    const float inv_w = 1.0f / static_cast<float>(input_width_);
    const float inv_h = 1.0f / static_cast<float>(input_height_);

    std::forward_list<ma_bbox_ext_t> bboxes;
    for (int i = 0; i < num_records_; ++i) {
        const float score = data[i + num_records_ * 4];
        if (score <= threshold_) continue;

        const float cx = data[i] * inv_w;
        const float cy = data[i + num_records_] * inv_h;
        const float w  = data[i + num_records_ * 2] * inv_w;
        const float h  = data[i + num_records_ * 3] * inv_h;

        ma_bbox_ext_t b;
        b.index = i;
        b.x = cx - w * 0.5f;
        b.y = cy - h * 0.5f;
        b.w = w;
        b.h = h;
        b.score = score;
        b.target = 0;
        bboxes.emplace_front(std::move(b));
    }

    ma::utils::nms(bboxes, threshold_nms_, threshold_, false, true);

    keypoint_count_ = 0;
    for (const auto& b : bboxes) {
        std::vector<ma_pt3f_t> pts(num_keypoints_);
        for (int k = 0; k < num_keypoints_; ++k) {
            const int idx = b.index + num_records_ * (5 + k * 3);
            pts[k].x = data[idx] * inv_w;
            pts[k].y = data[idx + num_records_] * inv_h;
            pts[k].z = data[idx + num_records_ * 2];
        }

        Subject subj;
        const float cx = b.x + b.w * 0.5f;
        const float cy = b.y + b.h * 0.5f;
        subj.box = geometry::InferBox::fromCenter(cx, cy, b.w, b.h, b.score);
        subj.score = b.score;
        subj.pose = Pose(pts, input_width_, input_height_, kpt_threshold_);
        keypoint_count_ = std::max(keypoint_count_, static_cast<int>(pts.size()));
        subjects_.push_back(std::move(subj));
    }

    std::sort(subjects_.begin(), subjects_.end(), [](const Subject& a, const Subject& b) {
        return a.score > b.score;
    });

    if (!subjects_.empty()) {
        primary_ = subjects_.front();
        has_primary_ = true;
    }

    return subjects_;
}

const Subject* PoseDetector::detectPrimary(ma_img_t* img) {
    detectAll(img);
    return (has_primary_ && last_run_ok_) ? &primary_ : nullptr;
}

}  // namespace microduck
