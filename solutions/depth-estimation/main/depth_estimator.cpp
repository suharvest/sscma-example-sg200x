#include "depth_estimator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#define TAG "DepthEstimator"

namespace depth {

namespace {

/* The camera's sensor picture is 16:9 (1920x1080 before VPSS scaling; see
 * chn_attr / grp_attr in components/sophgo/video/src/video_paramparse.c). */
constexpr float kSensorAspect = 16.0f / 9.0f;

const char* tensor_type_name(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_U8:  return "u8";
        case MA_TENSOR_TYPE_S8:  return "s8";
        case MA_TENSOR_TYPE_U16: return "u16";
        case MA_TENSOR_TYPE_S16: return "s16";
        case MA_TENSOR_TYPE_F16: return "f16";
        case MA_TENSOR_TYPE_F32: return "f32";
        case MA_TENSOR_TYPE_BF16:return "bf16";
        default: return "other";
    }
}

std::string shape_to_string(const ma_shape_t& s) {
    std::string out = "[";
    for (uint32_t i = 0; i < s.size; i++) {
        if (i) out += ",";
        out += std::to_string(s.dims[i]);
    }
    out += "]";
    return out;
}

/* Bilinear sample of one channel of an RGB888 buffer at (fx, fy) in pixels. */
inline float sample_channel(const uint8_t* src, int stride, int w, int h,
                            float fx, float fy, int c) {
    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 > w - 1) x0 = w - 1;
    if (y0 > h - 1) y0 = h - 1;
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float ax = fx - static_cast<float>(x0);
    const float ay = fy - static_cast<float>(y0);

    const uint8_t* r0 = src + static_cast<size_t>(y0) * stride;
    const uint8_t* r1 = src + static_cast<size_t>(y1) * stride;
    const float v00 = r0[x0 * 3 + c];
    const float v01 = r0[x1 * 3 + c];
    const float v10 = r1[x0 * 3 + c];
    const float v11 = r1[x1 * 3 + c];
    const float top = v00 + (v01 - v00) * ax;
    const float bot = v10 + (v11 - v10) * ax;
    return top + (bot - top) * ay;
}

}  // namespace

Roi valid_content_roi(int frame_width, int frame_height) {
    Roi roi{0, 0, frame_width, frame_height};
    if (frame_width <= 0 || frame_height <= 0) return roi;

    const float aspect = static_cast<float>(frame_width) / static_cast<float>(frame_height);
    if (std::fabs(aspect - kSensorAspect) < 0.01f) {
        return roi;  // channel already 16:9, VPSS added no bars
    }
    if (aspect > kSensorAspect) {
        // Channel wider than the sensor picture: grey bars left and right.
        roi.w = static_cast<int>(std::lround(frame_height * kSensorAspect));
        if (roi.w > frame_width) roi.w = frame_width;
        roi.x = (frame_width - roi.w) / 2;
    } else {
        // Channel taller: grey bars top and bottom. A 224x224 channel gives
        // [0, 49, 224, 126].
        roi.h = static_cast<int>(std::lround(frame_width / kSensorAspect));
        if (roi.h > frame_height) roi.h = frame_height;
        roi.y = (frame_height - roi.h) / 2;
    }
    return roi;
}

DepthEstimator::DepthEstimator()  = default;

DepthEstimator::~DepthEstimator() = default;

bool DepthEstimator::init(const std::string& model_path) {
    MA_LOGI(TAG, "Loading depth model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) {
        MA_LOGE(TAG, "EngineCVI init failed");
        return false;
    }
    if (engine_->load(model_path.c_str()) != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    const int32_t n_in  = engine_->getInputSize();
    const int32_t n_out = engine_->getOutputSize();
    MA_LOGI(TAG, "Model loaded: %d input(s), %d output(s)", n_in, n_out);
    if (n_in != 1 || n_out != 1) {
        MA_LOGE(TAG, "Expected exactly 1 input and 1 output, got %d/%d", n_in, n_out);
        return false;
    }

    input_  = engine_->getInput(0);
    output_ = engine_->getOutput(0);

    const ma_shape_t in_shape  = input_.shape;
    const ma_shape_t out_shape = output_.shape;

    /* Print what the model really carries before judging it: a shape mismatch
     * discovered on the device is only diagnosable if the actual numbers are in
     * the log. */
    MA_LOGI(TAG, "input  '%s' shape=%s dtype=%s scale=%.6f zp=%d bytes=%zu",
            input_.name ? input_.name : "(unnamed)", shape_to_string(in_shape).c_str(),
            tensor_type_name(input_.type), input_.quant_param.scale,
            input_.quant_param.zero_point, input_.size);
    MA_LOGI(TAG, "output '%s' shape=%s dtype=%s scale=%.6f zp=%d bytes=%zu",
            output_.name ? output_.name : "(unnamed)", shape_to_string(out_shape).c_str(),
            tensor_type_name(output_.type), output_.quant_param.scale,
            output_.quant_param.zero_point, output_.size);

    if (in_shape.size != 4 || in_shape.dims[0] != 1 || in_shape.dims[1] != 3) {
        MA_LOGE(TAG, "Unsupported input shape %s: expected 1x3xHxW",
                shape_to_string(in_shape).c_str());
        return false;
    }
    input_h_ = in_shape.dims[2];
    input_w_ = in_shape.dims[3];
    if (input_w_ <= 0 || input_h_ <= 0) {
        MA_LOGE(TAG, "Invalid input spatial size %dx%d", input_w_, input_h_);
        return false;
    }

    /* Dense depth map: accept 1x1xHxW and the squeezed 1xHxW / HxW spellings,
     * because which one a converter emits is not something the application
     * should have an opinion about. Anything else is a different model. */
    int32_t od[2] = {0, 0};
    bool out_ok   = false;
    if (out_shape.size == 4 && out_shape.dims[0] == 1 && out_shape.dims[1] == 1) {
        od[0] = out_shape.dims[2]; od[1] = out_shape.dims[3]; out_ok = true;
    } else if (out_shape.size == 3 && out_shape.dims[0] == 1) {
        od[0] = out_shape.dims[1]; od[1] = out_shape.dims[2]; out_ok = true;
    } else if (out_shape.size == 2) {
        od[0] = out_shape.dims[0]; od[1] = out_shape.dims[1]; out_ok = true;
    }
    if (!out_ok || od[0] <= 0 || od[1] <= 0) {
        MA_LOGE(TAG, "Unsupported output shape %s: expected a dense HxW depth map",
                shape_to_string(out_shape).c_str());
        return false;
    }
    output_h_ = od[0];
    output_w_ = od[1];

    if (input_.type != MA_TENSOR_TYPE_F32 && input_.type != MA_TENSOR_TYPE_S8) {
        MA_LOGE(TAG, "Unsupported input dtype %s (f32 or s8 expected)",
                tensor_type_name(input_.type));
        return false;
    }
    if (output_.type != MA_TENSOR_TYPE_F32 && output_.type != MA_TENSOR_TYPE_S8) {
        MA_LOGE(TAG, "Unsupported output dtype %s (f32 or s8 expected)",
                tensor_type_name(output_.type));
        return false;
    }
    if (input_.data.data == nullptr || output_.data.data == nullptr) {
        MA_LOGE(TAG, "Model tensors carry no buffer");
        return false;
    }

    const size_t in_elems  = static_cast<size_t>(input_w_) * input_h_ * 3;
    const size_t in_bytes  = in_elems * (input_.type == MA_TENSOR_TYPE_F32 ? 4u : 1u);
    if (input_.size < in_bytes) {
        MA_LOGE(TAG, "Input buffer %zu B smaller than the %zu B its shape needs",
                input_.size, in_bytes);
        return false;
    }
    const size_t out_elems = static_cast<size_t>(output_w_) * output_h_;
    const size_t out_bytes = out_elems * (output_.type == MA_TENSOR_TYPE_F32 ? 4u : 1u);
    if (output_.size < out_bytes) {
        MA_LOGE(TAG, "Output buffer %zu B smaller than the %zu B its shape needs",
                output_.size, out_bytes);
        return false;
    }

    depth_.assign(out_elems, 0.0f);
    initialized_ = true;
    MA_LOGI(TAG, "Depth estimator ready (in %dx%d, out %dx%d)",
            input_w_, input_h_, output_w_, output_h_);
    return true;
}

void DepthEstimator::buildXMap(const Roi& roi, int frame_width) {
    if (xmap_valid_ && xmap_roi_x_ == roi.x && xmap_roi_w_ == roi.w &&
        xmap_frame_w_ == frame_width &&
        static_cast<int>(xmap_x0_.size()) == input_w_) {
        return;  // geometry is fixed for the life of the stream
    }
    xmap_x0_.resize(input_w_);
    xmap_x1_.resize(input_w_);
    xmap_ax_.resize(input_w_);
    const float sx = static_cast<float>(roi.w) / static_cast<float>(input_w_);
    for (int x = 0; x < input_w_; x++) {
        const float fx = roi.x + (x + 0.5f) * sx - 0.5f;
        int x0 = static_cast<int>(fx);
        if (x0 < 0) x0 = 0;
        if (x0 > frame_width - 1) x0 = frame_width - 1;
        xmap_x0_[x] = x0;
        xmap_x1_[x] = std::min(x0 + 1, frame_width - 1);
        xmap_ax_[x] = fx - static_cast<float>(x0);
    }
    xmap_roi_x_   = roi.x;
    xmap_roi_w_   = roi.w;
    xmap_frame_w_ = frame_width;
    xmap_valid_   = true;
}

bool DepthEstimator::preprocess(const ma_img_t* frame, const Roi& roi) {
    const uint8_t* src = static_cast<const uint8_t*>(frame->data);
    const int stride   = frame->width * 3;

    /* Stretch, not letterbox. The rest of the pipeline treats the depth map as
     * a straight rescale of the ROI, so nothing anywhere needs to undo a
     * padding offset -- the same convention sscma's own preprocessing uses
     * (rgb888_to_rgb888_planar derives its horizontal and vertical steps
     * independently and pads nothing). */
    const float sx = static_cast<float>(roi.w) / static_cast<float>(input_w_);
    const float sy = static_cast<float>(roi.h) / static_cast<float>(input_h_);
    const size_t plane = static_cast<size_t>(input_w_) * input_h_;

    /* The horizontal map is the same for every row, so resolve it once per
     * frame instead of per pixel, and resolve the row/weight pair once per
     * output pixel instead of once per channel: the three channels share every
     * index and both interpolation weights. */
    buildXMap(roi, frame->width);

    if (input_.type == MA_TENSOR_TYPE_F32) {
        float* dst = input_.data.f32;
        constexpr float kInv255 = 1.0f / 255.0f;
        for (int y = 0; y < input_h_; y++) {
            const float fy = roi.y + (y + 0.5f) * sy - 0.5f;
            int y0 = static_cast<int>(fy);
            if (y0 < 0) y0 = 0;
            if (y0 > frame->height - 1) y0 = frame->height - 1;
            const int y1   = std::min(y0 + 1, frame->height - 1);
            const float ay = fy - static_cast<float>(y0);

            const uint8_t* r0 = src + static_cast<size_t>(y0) * stride;
            const uint8_t* r1 = src + static_cast<size_t>(y1) * stride;
            const size_t row_off = static_cast<size_t>(y) * input_w_;

            for (int x = 0; x < input_w_; x++) {
                const int   x0 = xmap_x0_[x];
                const int   x1 = xmap_x1_[x];
                const float ax = xmap_ax_[x];
                const size_t o = row_off + x;
                const uint8_t* p00 = r0 + x0 * 3;
                const uint8_t* p01 = r0 + x1 * 3;
                const uint8_t* p10 = r1 + x0 * 3;
                const uint8_t* p11 = r1 + x1 * 3;
                for (int c = 0; c < 3; c++) {
                    const float top = p00[c] + (p01[c] - p00[c]) * ax;
                    const float bot = p10[c] + (p11[c] - p10[c]) * ax;
                    dst[c * plane + o] = (top + (bot - top) * ay) * kInv255;
                }
            }
        }
        return true;
    }

    /* Quantised input: the same [0,1] value, expressed in the tensor's own
     * scale. A zero or absent scale would silently produce an all-zero input,
     * so it is a hard error rather than a clamp. */
    const float scale = input_.quant_param.scale;
    if (!(scale > 0.0f)) {
        MA_LOGE(TAG, "Quantised input with non-positive scale %.6f", scale);
        return false;
    }
    const int zp = input_.quant_param.zero_point;
    int8_t* dst  = input_.data.s8;
    for (int y = 0; y < input_h_; y++) {
        const float fy = roi.y + (y + 0.5f) * sy - 0.5f;
        for (int x = 0; x < input_w_; x++) {
            const float fx = roi.x + (x + 0.5f) * sx - 0.5f;
            const size_t o = static_cast<size_t>(y) * input_w_ + x;
            for (int c = 0; c < 3; c++) {
                const float v =
                    sample_channel(src, stride, frame->width, frame->height, fx, fy, c) / 255.0f;
                int q = static_cast<int>(std::lround(v / scale)) + zp;
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                dst[c * plane + o] = static_cast<int8_t>(q);
            }
        }
    }
    return true;
}

bool DepthEstimator::readOutput() {
    const size_t n = depth_.size();
    if (output_.type == MA_TENSOR_TYPE_F32) {
        std::memcpy(depth_.data(), output_.data.f32, n * sizeof(float));
        return true;
    }
    const float scale = output_.quant_param.scale;
    const int zp      = output_.quant_param.zero_point;
    const int8_t* q   = output_.data.s8;
    const float s     = (scale > 0.0f) ? scale : 1.0f;
    for (size_t i = 0; i < n; i++) {
        depth_[i] = (static_cast<float>(q[i]) - static_cast<float>(zp)) * s;
    }
    return true;
}

bool DepthEstimator::run(const ma_img_t* frame, const Roi& roi) {
    if (!initialized_ || frame == nullptr || frame->data == nullptr) return false;
    if (roi.w <= 0 || roi.h <= 0) return false;
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.w > frame->width || roi.y + roi.h > frame->height) {
        MA_LOGW(TAG, "ROI %d,%d %dx%d outside frame %dx%d",
                roi.x, roi.y, roi.w, roi.h, frame->width, frame->height);
        return false;
    }

    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };

    const auto t0 = clk::now();
    if (!preprocess(frame, roi)) return false;
    const auto t1 = clk::now();
    if (engine_->run() != MA_OK) {
        MA_LOGE(TAG, "Forward pass failed");
        return false;
    }
    const auto t2 = clk::now();
    /* Re-read the output descriptor: the runtime may hand back a different
     * buffer pointer after a forward pass. */
    output_ = engine_->getOutput(0);
    if (output_.data.data == nullptr) return false;
    if (!readOutput()) return false;
    const auto t3 = clk::now();

    last_preprocess_ms_ = ms(t0, t1);
    last_forward_ms_    = ms(t1, t2);
    last_readout_ms_    = ms(t2, t3);
    last_inference_ms_  = ms(t0, t3);
    return true;
}

}  // namespace depth
