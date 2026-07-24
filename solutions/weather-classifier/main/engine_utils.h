#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sscma.h>
#include <vector>

namespace weather {

inline float bf16_to_fp32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float fp16_to_fp32(uint16_t v) {
    const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
    const uint32_t exp = (v & 0x7C00u) >> 10;
    const uint32_t mant = v & 0x03FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 0;
            while ((m & 0x0400u) == 0) { m <<= 1; ++e; }
            m &= 0x03FFu;
            out = sign | ((127u - 15u - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

inline uint16_t fp32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

inline uint16_t fp32_to_fp16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 31) & 1u;
    int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign << 15);
        mant |= 0x800000u;
        return static_cast<uint16_t>((sign << 15) | (mant >> (14 - exp)));
    }
    if (exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline size_t shape_numel(const ma_shape_t& s) {
    if (s.size <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < s.size; ++i) {
        if (s.dims[i] <= 0) return 0;
        n *= static_cast<size_t>(s.dims[i]);
    }
    return n;
}

inline size_t tensor_elem_size(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: return 4;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16: return 2;
        case MA_TENSOR_TYPE_S8:
        case MA_TENSOR_TYPE_U8: return 1;
        default: return 0;
    }
}

struct InputBuf {
    std::vector<uint8_t> u8;
    std::vector<int8_t> s8;
    std::vector<uint16_t> u16;
    std::vector<float> f32;

    void resize_for(ma_tensor_type_t t, size_t n) {
        u8.clear(); s8.clear(); u16.clear(); f32.clear();
        switch (t) {
            case MA_TENSOR_TYPE_U8: u8.assign(n, 0); break;
            case MA_TENSOR_TYPE_S8: s8.assign(n, 0); break;
            case MA_TENSOR_TYPE_F16:
            case MA_TENSOR_TYPE_BF16: u16.assign(n, 0); break;
            case MA_TENSOR_TYPE_F32: f32.assign(n, 0.f); break;
            default: break;
        }
    }

    void* data_for(ma_tensor_type_t t) {
        switch (t) {
            case MA_TENSOR_TYPE_U8: return u8.data();
            case MA_TENSOR_TYPE_S8: return s8.data();
            case MA_TENSOR_TYPE_F16:
            case MA_TENSOR_TYPE_BF16: return u16.data();
            case MA_TENSOR_TYPE_F32: return f32.data();
            default: return nullptr;
        }
    }
};

inline void store_val(InputBuf& buf, ma_tensor_type_t t,
                      const ma_quant_param_t& qp, size_t idx, float real) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: buf.f32[idx] = real; break;
        case MA_TENSOR_TYPE_F16: buf.u16[idx] = fp32_to_fp16(real); break;
        case MA_TENSOR_TYPE_BF16: buf.u16[idx] = fp32_to_bf16(real); break;
        case MA_TENSOR_TYPE_S8: {
            const float inv = qp.scale > 0.f ? 1.f / qp.scale : 0.f;
            int q = static_cast<int>(std::lround(real * inv)) + qp.zero_point;
            q = std::max(-128, std::min(127, q));
            buf.s8[idx] = static_cast<int8_t>(q);
            break;
        }
        case MA_TENSOR_TYPE_U8: {
            const float inv = qp.scale > 0.f ? 1.f / qp.scale : 0.f;
            int q = static_cast<int>(std::lround(real * inv)) + qp.zero_point;
            q = std::max(0, std::min(255, q));
            buf.u8[idx] = static_cast<uint8_t>(q);
            break;
        }
        default: break;
    }
}

inline float read_val(const ma_tensor_t& t, size_t idx) {
    switch (t.type) {
        case MA_TENSOR_TYPE_F32: return t.data.f32[idx];
        case MA_TENSOR_TYPE_F16: return fp16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_BF16: return bf16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_S8:
            return (static_cast<int>(t.data.s8[idx]) - t.quant_param.zero_point) * t.quant_param.scale;
        case MA_TENSOR_TYPE_U8:
            return (static_cast<int>(t.data.u8[idx]) - t.quant_param.zero_point) * t.quant_param.scale;
        default: return 0.f;
    }
}

inline ma_tensor_t make_input_tensor(ma_tensor_type_t t, InputBuf& buf, size_t numel) {
    ma_tensor_t tensor{};
    tensor.size = numel * tensor_elem_size(t);
    tensor.is_physical = false;
    tensor.is_variable = false;
    tensor.data.data = buf.data_for(t);
    return tensor;
}

}  // namespace weather
