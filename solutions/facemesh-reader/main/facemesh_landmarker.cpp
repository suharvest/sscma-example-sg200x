#include "facemesh_landmarker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define TAG "FacemeshLandmarker"

namespace facemesh_reader {

// ---------- fp16/bf16 helpers (mirror EmotionRunner) ----------
float FacemeshLandmarker::bf16_to_fp32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float FacemeshLandmarker::fp16_to_fp32(uint16_t v) {
    const uint32_t sign = (uint32_t)(v & 0x8000u) << 16;
    const uint32_t exp = (v & 0x7C00u) >> 10;
    const uint32_t mant = (v & 0x03FFu);

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 0;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                e++;
            }
            m &= 0x03FFu;
            const uint32_t exp32 = (127 - 15 - e) << 23;
            const uint32_t mant32 = m << 13;
            out = sign | exp32 | mant32;
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t exp32 = (exp + (127 - 15)) << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t FacemeshLandmarker::fp32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return (uint16_t)(u >> 16);
}

uint16_t FacemeshLandmarker::fp32_to_fp16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 31) & 1;
    int exp = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15);
        mant |= 0x800000;
        const int shift = 14 - exp;
        return (uint16_t)((sign << 15) | (mant >> shift));
    }
    if (exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    return (uint16_t)((sign << 15) | ((uint16_t)exp << 10) | (uint16_t)(mant >> 13));
}

size_t FacemeshLandmarker::elemSize(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: return 4;
        case MA_TENSOR_TYPE_F16: return 2;
        case MA_TENSOR_TYPE_BF16: return 2;
        case MA_TENSOR_TYPE_S8: return 1;
        case MA_TENSOR_TYPE_U8: return 1;
        default: return 0;
    }
}

size_t FacemeshLandmarker::shapeNumel(const ma_shape_t& s) {
    if (s.size <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < s.size; ++i) {
        if (s.dims[i] <= 0) return 0;
        n *= (size_t)s.dims[i];
    }
    return n;
}

float FacemeshLandmarker::readVal(const ma_tensor_t& t, int idx) const {
    switch (t.type) {
        case MA_TENSOR_TYPE_F32:
            return t.data.f32[idx];
        case MA_TENSOR_TYPE_F16:
            return fp16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_BF16:
            return bf16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_S8: {
            const float scale = t.quant_param.scale;
            const int zp = t.quant_param.zero_point;
            return (t.data.s8[idx] - zp) * scale;
        }
        case MA_TENSOR_TYPE_U8: {
            const float scale = t.quant_param.scale;
            const int zp = t.quant_param.zero_point;
            return ((int)t.data.u8[idx] - zp) * scale;
        }
        default:
            return 0.f;
    }
}

bool FacemeshLandmarker::prepareInputTensor() {
    input_tensor_cache_ = engine_->getInput(0);
    input_type_ = input_tensor_cache_.type;

    const ma_shape_t s = engine_->getInputShape(0);
    if (s.size != 4) {
        MA_LOGE(TAG, "Unexpected input rank: %d (want 4)", s.size);
        return false;
    }

    // FaceMesh standard: NHWC [1,192,192,3] (fp32). Detect layout robustly anyway.
    if (s.dims[3] == 3 || s.dims[3] == 1) {
        input_is_chw_ = false;
        input_h_ = s.dims[1];
        input_w_ = s.dims[2];
        input_c_ = s.dims[3];
    } else if (s.dims[1] == 3 || s.dims[1] == 1) {
        input_is_chw_ = true;
        input_c_ = s.dims[1];
        input_h_ = s.dims[2];
        input_w_ = s.dims[3];
    } else {
        input_is_chw_ = false;
        input_h_ = s.dims[1];
        input_w_ = s.dims[2];
        input_c_ = s.dims[3];
    }

    input_numel_ = shapeNumel(s);
    if (input_numel_ == 0) return false;

    input_u8_.clear();
    input_s8_.clear();
    input_u16_.clear();
    input_f32_.clear();

    switch (input_type_) {
        case MA_TENSOR_TYPE_U8:
            input_u8_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_S8:
            input_s8_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16:
            input_u16_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_F32:
            input_f32_.assign(input_numel_, 0.f);
            break;
        default:
            MA_LOGE(TAG, "Unsupported input dtype: %d", (int)input_type_);
            return false;
    }

    MA_LOGI(TAG, "FaceMesh input: %dx%dx%d (%s), dtype=%d, numel=%zu",
            input_h_, input_w_, input_c_,
            input_is_chw_ ? "CHW" : "HWC",
            (int)input_type_, input_numel_);
    return true;
}

int FacemeshLandmarker::findLandmarkOutputIndex() {
    const int n = engine_->getOutputSize();
    for (int i = 0; i < n; ++i) {
        const ma_tensor_t t = engine_->getOutput(i);
        size_t numel = shapeNumel(t.shape);
        if (numel == 0) {
            const size_t es = elemSize(t.type);
            if (es > 0) numel = t.size / es;
        }
        if (numel == 1404) {
            return i;
        }
    }
    // Fallback: pick the largest tensor.
    int best = 0;
    size_t best_n = 0;
    for (int i = 0; i < n; ++i) {
        const ma_tensor_t t = engine_->getOutput(i);
        size_t numel = shapeNumel(t.shape);
        if (numel == 0) {
            const size_t es = elemSize(t.type);
            if (es > 0) numel = t.size / es;
        }
        if (numel > best_n) {
            best_n = numel;
            best = i;
        }
    }
    return best;
}

bool FacemeshLandmarker::init(const std::string& model_path) {
    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) {
        MA_LOGE(TAG, "engine init failed");
        return false;
    }
    if (engine_->load(model_path) != MA_OK) {
        MA_LOGE(TAG, "engine load failed: %s", model_path.c_str());
        return false;
    }
    if (!prepareInputTensor()) return false;

    landmark_output_idx_ = findLandmarkOutputIndex();
    MA_LOGI(TAG, "Landmark output tensor index: %d", landmark_output_idx_);

    ready_ = true;
    return true;
}

std::vector<Point2D> FacemeshLandmarker::infer(const uint8_t* roi_rgb) {
    std::vector<Point2D> out;
    if (!ready_ || !engine_ || !roi_rgb) return out;

    const int H = input_h_;
    const int W = input_w_;
    const int C = input_c_;

    // Pack input: normalize uint8 RGB -> [0,1] fp32 (or other dtypes if model differs).
    const float qscale = input_tensor_cache_.quant_param.scale;
    const int qzp = input_tensor_cache_.quant_param.zero_point;

    auto store_real = [&](size_t idx, float real) {
        switch (input_type_) {
            case MA_TENSOR_TYPE_F32:
                input_f32_[idx] = real;
                break;
            case MA_TENSOR_TYPE_BF16:
                input_u16_[idx] = fp32_to_bf16(real);
                break;
            case MA_TENSOR_TYPE_F16:
                input_u16_[idx] = fp32_to_fp16(real);
                break;
            case MA_TENSOR_TYPE_S8: {
                const float inv = (qscale > 0.f) ? (1.0f / qscale) : 0.f;
                int q = (int)std::lround(real * inv) + qzp;
                q = std::clamp(q, -128, 127);
                input_s8_[idx] = (int8_t)q;
                break;
            }
            case MA_TENSOR_TYPE_U8: {
                const float inv = (qscale > 0.f) ? (1.0f / qscale) : 0.f;
                int q = (int)std::lround(real * inv) + qzp;
                q = std::clamp(q, 0, 255);
                input_u8_[idx] = (uint8_t)q;
                break;
            }
            default:
                break;
        }
    };

    if (input_is_chw_) {
        const size_t plane = (size_t)H * (size_t)W;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t* p = roi_rgb + ((size_t)y * (size_t)W + (size_t)x) * 3;
                for (int c = 0; c < C; ++c) {
                    const float real = p[c] / 255.0f;
                    store_real((size_t)c * plane + (size_t)y * (size_t)W + (size_t)x, real);
                }
            }
        }
    } else {
        // HWC
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t* p = roi_rgb + ((size_t)y * (size_t)W + (size_t)x) * 3;
                for (int c = 0; c < C; ++c) {
                    const float real = p[c] / 255.0f;
                    store_real(((size_t)y * (size_t)W + (size_t)x) * (size_t)C + (size_t)c, real);
                }
            }
        }
    }

    ma_tensor_t tensor = {
        .size = input_numel_ * elemSize(input_type_),
        .is_physical = false,
        .is_variable = false,
    };
    switch (input_type_) {
        case MA_TENSOR_TYPE_F32:
            tensor.data.data = input_f32_.data();
            break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16:
            tensor.data.data = input_u16_.data();
            break;
        case MA_TENSOR_TYPE_S8:
            tensor.data.data = input_s8_.data();
            break;
        case MA_TENSOR_TYPE_U8:
            tensor.data.data = input_u8_.data();
            break;
        default:
            return out;
    }

    engine_->setInput(0, tensor);
    if (engine_->run() != MA_OK) {
        MA_LOGW(TAG, "engine run failed");
        return out;
    }

    const ma_tensor_t lt = engine_->getOutput(landmark_output_idx_);
    size_t lt_numel = shapeNumel(lt.shape);
    if (lt_numel == 0) {
        const size_t es = elemSize(lt.type);
        if (es > 0) lt_numel = lt.size / es;
    }
    if (lt_numel < 468 * 3) {
        MA_LOGE(TAG, "Landmark tensor too small: %zu (want >= 1404)", lt_numel);
        return out;
    }

    out.reserve(468);
    for (int i = 0; i < 468; ++i) {
        const float x = readVal(lt, i * 3 + 0);
        const float y = readVal(lt, i * 3 + 1);
        // z (i*3+2) ignored for 2D EAR/MAR
        out.push_back(Point2D{x, y});
    }
    return out;
}

}  // namespace facemesh_reader
