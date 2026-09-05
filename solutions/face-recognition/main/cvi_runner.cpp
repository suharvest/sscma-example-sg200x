#include "cvi_runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#define TAG "face-recognition/cvi"

namespace face_recognition {

namespace {
void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}
}  // namespace

float CviRunner::bf16ToFp32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

float CviRunner::fp16ToFp32(uint16_t v) {
    const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
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
            out = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t CviRunner::fp32ToBf16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    // Round-to-nearest-even on the truncated mantissa. Plain truncation biases
    // every value toward zero, which on a 112x112x3 input is a systematic tilt
    // of the whole tensor rather than noise.
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

uint16_t CviRunner::fp32ToFp16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 31) & 1u;
    int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign << 15);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        return static_cast<uint16_t>((sign << 15) | (mant >> shift));
    }
    if (exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint16_t>(exp) << 10) |
                                 static_cast<uint16_t>(mant >> 13));
}

size_t CviRunner::elemSize(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: return 4;
        case MA_TENSOR_TYPE_F16: return 2;
        case MA_TENSOR_TYPE_BF16: return 2;
        case MA_TENSOR_TYPE_S8: return 1;
        case MA_TENSOR_TYPE_U8: return 1;
        case MA_TENSOR_TYPE_S16: return 2;
        case MA_TENSOR_TYPE_S32: return 4;
        default: return 0;
    }
}

size_t CviRunner::shapeNumel(const ma_shape_t& s) {
    if (s.size <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < s.size; ++i) {
        if (s.dims[i] <= 0) return 0;
        n *= static_cast<size_t>(s.dims[i]);
    }
    return n;
}

size_t CviRunner::tensorNumel(const ma_tensor_t& t) {
    const size_t n = shapeNumel(t.shape);
    if (n > 0) return n;
    const size_t es = elemSize(t.type);
    return es > 0 ? (t.size / es) : 0;
}

std::string CviRunner::shapeToString(const ma_shape_t& s) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < s.size; ++i) {
        if (i) oss << ",";
        oss << s.dims[i];
    }
    oss << ")";
    return oss.str();
}

float CviRunner::readValue(const ma_tensor_t& t, size_t idx) {
    switch (t.type) {
        case MA_TENSOR_TYPE_F32: return t.data.f32[idx];
        case MA_TENSOR_TYPE_F16: return fp16ToFp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_BF16: return bf16ToFp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_S8:
            return (static_cast<int>(t.data.s8[idx]) - t.quant_param.zero_point) *
                   t.quant_param.scale;
        case MA_TENSOR_TYPE_U8:
            return (static_cast<int>(t.data.u8[idx]) - t.quant_param.zero_point) *
                   t.quant_param.scale;
        case MA_TENSOR_TYPE_S16:
            return (static_cast<int>(t.data.s16[idx]) - t.quant_param.zero_point) *
                   t.quant_param.scale;
        case MA_TENSOR_TYPE_S32:
            return (static_cast<float>(t.data.s32[idx]) - t.quant_param.zero_point) *
                   t.quant_param.scale;
        default: return 0.f;
    }
}

bool CviRunner::init(const std::string& model_path,
                     const float mean[3], const float scale[3], bool bgr,
                     std::string* error) {
    inited_ = false;
    for (int i = 0; i < 3; ++i) {
        mean_[i] = mean ? mean[i] : 0.f;
        scale_[i] = scale ? scale[i] : 1.f;
    }
    bgr_ = bgr;

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) {
        setError(error, "EngineCVI::init failed");
        return false;
    }
    if (engine_->load(model_path) != MA_OK) {
        setError(error, "cannot load " + model_path);
        engine_.reset();
        return false;
    }
    if (engine_->getInputSize() != 1) {
        setError(error, model_path + ": expected exactly 1 input, got " +
                            std::to_string(engine_->getInputSize()));
        engine_.reset();
        return false;
    }
    if (!prepareInput(error)) {
        engine_.reset();
        return false;
    }
    inited_ = true;
    return true;
}

bool CviRunner::prepareInput(std::string* error) {
    input_tensor_ = engine_->getInput(0);
    input_type_ = input_tensor_.type;

    const ma_shape_t s = engine_->getInputShape(0);
    if (s.size != 4 || s.dims[0] != 1) {
        setError(error, "unsupported input rank/batch " + shapeToString(s));
        return false;
    }
    if (s.dims[1] == 3 || s.dims[1] == 1) {
        input_is_chw_ = true;
        input_c_ = s.dims[1];
        input_h_ = s.dims[2];
        input_w_ = s.dims[3];
    } else if (s.dims[3] == 3 || s.dims[3] == 1) {
        input_is_chw_ = false;
        input_h_ = s.dims[1];
        input_w_ = s.dims[2];
        input_c_ = s.dims[3];
    } else {
        setError(error, "cannot infer layout from input shape " + shapeToString(s));
        return false;
    }

    input_numel_ = shapeNumel(s);
    const size_t es = elemSize(input_type_);
    if (input_numel_ == 0 || es == 0) {
        setError(error, "input tensor has unusable shape/type");
        return false;
    }
    const size_t logical = input_numel_ * es;
    if (input_tensor_.size < logical) {
        setError(error, "input tensor smaller than its own shape");
        return false;
    }

    // A padded CVI tensor must still be a uniform row stride: NHWC has H rows of
    // W*C elements, CHW has C*H rows of W elements.
    if (input_tensor_.size > logical) {
        const size_t rows = input_is_chw_ ? static_cast<size_t>(input_c_) * input_h_
                                          : static_cast<size_t>(input_h_);
        const size_t logical_row = input_is_chw_
                                       ? static_cast<size_t>(input_w_) * es
                                       : static_cast<size_t>(input_w_) * input_c_ * es;
        if (rows == 0 || input_tensor_.size % rows != 0 ||
            input_tensor_.size / rows < logical_row) {
            setError(error, "padded input tensor is not a uniform row stride");
            return false;
        }
        buf_physical_.assign(input_tensor_.size, 0);
    } else {
        buf_physical_.clear();
    }

    buf_u8_.clear();
    buf_s8_.clear();
    buf_u16_.clear();
    buf_f32_.clear();
    switch (input_type_) {
        case MA_TENSOR_TYPE_U8: buf_u8_.assign(input_numel_, 0); break;
        case MA_TENSOR_TYPE_S8: buf_s8_.assign(input_numel_, 0); break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16: buf_u16_.assign(input_numel_, 0); break;
        case MA_TENSOR_TYPE_F32: buf_f32_.assign(input_numel_, 0.f); break;
        default:
            setError(error, "unsupported input tensor type " +
                                std::to_string(static_cast<int>(input_type_)));
            return false;
    }

    // Integer image inputs carry the converter's fused quantization: the value
    // the TPU wants is the raw pixel re-centred, not a host-side normalization.
    raw_passthrough_ = (input_type_ == MA_TENSOR_TYPE_S8 || input_type_ == MA_TENSOR_TYPE_U8);

    const char* type_name = "?";
    switch (input_type_) {
        case MA_TENSOR_TYPE_U8: type_name = "u8"; break;
        case MA_TENSOR_TYPE_S8: type_name = "s8"; break;
        case MA_TENSOR_TYPE_F16: type_name = "f16"; break;
        case MA_TENSOR_TYPE_F32: type_name = "f32"; break;
        case MA_TENSOR_TYPE_BF16: type_name = "bf16"; break;
        default: break;
    }
    std::ostringstream oss;
    oss << shapeToString(s) << " " << (input_is_chw_ ? "NCHW" : "NHWC") << " type="
        << type_name
        << " bytes=" << input_tensor_.size << " (logical " << logical << ")"
        << " norm=" << (raw_passthrough_ ? "pixel-128" : "(x-mean)*scale");
    input_desc_ = oss.str();
    return true;
}

void CviRunner::packInput(const uint8_t* rgb_hwc) {
    const int H = input_h_, W = input_w_, C = input_c_;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    auto store = [&](size_t idx, uint8_t px, int c) {
        if (raw_passthrough_) {
            if (input_type_ == MA_TENSOR_TYPE_U8) {
                buf_u8_[idx] = px;
            } else {
                buf_s8_[idx] = static_cast<int8_t>(
                    std::min(127, std::max(-128, static_cast<int>(px) - 128)));
            }
            return;
        }
        const float real = (static_cast<float>(px) - mean_[c]) * scale_[c];
        switch (input_type_) {
            case MA_TENSOR_TYPE_F32: buf_f32_[idx] = real; break;
            case MA_TENSOR_TYPE_BF16: buf_u16_[idx] = fp32ToBf16(real); break;
            case MA_TENSOR_TYPE_F16: buf_u16_[idx] = fp32ToFp16(real); break;
            default: break;
        }
    };

    for (int y = 0; y < H; ++y) {
        const uint8_t* row = rgb_hwc + static_cast<size_t>(y) * W * 3u;
        for (int x = 0; x < W; ++x) {
            const uint8_t* p = row + static_cast<size_t>(x) * 3u;
            for (int c = 0; c < C; ++c) {
                const int src_c = (C == 3 && bgr_) ? (2 - c) : c;
                const uint8_t px = (C == 1)
                                       ? static_cast<uint8_t>((p[0] * 299 + p[1] * 587 +
                                                               p[2] * 114) / 1000)
                                       : p[src_c];
                const size_t idx =
                    input_is_chw_
                        ? static_cast<size_t>(c) * plane + static_cast<size_t>(y) * W + x
                        : (static_cast<size_t>(y) * W + x) * C + c;
                store(idx, px, c);
            }
        }
    }
}

bool CviRunner::submit(std::string* error) {
    const size_t es = elemSize(input_type_);
    const size_t logical = input_numel_ * es;
    const void* logical_data = nullptr;
    switch (input_type_) {
        case MA_TENSOR_TYPE_F32: logical_data = buf_f32_.data(); break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16: logical_data = buf_u16_.data(); break;
        case MA_TENSOR_TYPE_S8: logical_data = buf_s8_.data(); break;
        case MA_TENSOR_TYPE_U8: logical_data = buf_u8_.data(); break;
        default: setError(error, "unsupported input type at submit"); return false;
    }

    const void* submit_data = logical_data;
    if (input_tensor_.size != logical) {
        const size_t rows = input_is_chw_ ? static_cast<size_t>(input_c_) * input_h_
                                          : static_cast<size_t>(input_h_);
        const size_t logical_row = input_is_chw_
                                       ? static_cast<size_t>(input_w_) * es
                                       : static_cast<size_t>(input_w_) * input_c_ * es;
        const size_t physical_row = input_tensor_.size / rows;
        if (buf_physical_.size() != input_tensor_.size) {
            setError(error, "physical staging buffer size mismatch");
            return false;
        }
        std::fill(buf_physical_.begin(), buf_physical_.end(), 0);
        const uint8_t* src = static_cast<const uint8_t*>(logical_data);
        for (size_t r = 0; r < rows; ++r) {
            memcpy(buf_physical_.data() + r * physical_row, src + r * logical_row,
                   logical_row);
        }
        submit_data = buf_physical_.data();
    }

    ma_tensor_t tensor = {};
    tensor.size = input_tensor_.size;
    tensor.is_physical = false;
    tensor.is_variable = false;
    tensor.data.data = const_cast<void*>(submit_data);

    if (engine_->setInput(0, tensor) != MA_OK) {
        setError(error, "setInput failed");
        return false;
    }
    if (engine_->run() != MA_OK) {
        setError(error, "forward failed");
        return false;
    }
    return true;
}

bool CviRunner::run(const uint8_t* rgb_hwc, std::string* error) {
    if (!inited_ || !engine_ || !rgb_hwc) {
        setError(error, "runner not initialized");
        return false;
    }
    packInput(rgb_hwc);
    return submit(error);
}

int CviRunner::outputCount() const {
    return engine_ ? engine_->getOutputSize() : 0;
}

ma_tensor_t CviRunner::output(int index) const {
    return engine_->getOutput(index);
}

}  // namespace face_recognition
