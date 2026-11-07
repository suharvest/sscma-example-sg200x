/**
 * @file frame.cpp
 * @brief Format tables and the zero-copy crop for af::VideoFrame.
 */
#include "af/frame.h"

#include <algorithm>

namespace af {

uint8_t plane_count(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::RGB888:
        case PixelFormat::BGR888:
        case PixelFormat::Gray8:
            return 1;
        case PixelFormat::NV21:
            return 2;
        case PixelFormat::RGB888Planar:
            return 3;
        default:
            return 0;
    }
}

uint8_t bytes_per_pixel(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::RGB888:
        case PixelFormat::BGR888:
            return 3;
        case PixelFormat::RGB888Planar:
        case PixelFormat::NV21:
        case PixelFormat::Gray8:
            return 1;
        default:
            return 0;
    }
}

const char* to_string(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::RGB888:
            return "RGB888";
        case PixelFormat::BGR888:
            return "BGR888";
        case PixelFormat::RGB888Planar:
            return "RGB888Planar";
        case PixelFormat::NV21:
            return "NV21";
        case PixelFormat::Gray8:
            return "Gray8";
        default:
            return "Unknown";
    }
}

VideoFrame VideoFrame::packed(uint8_t* data, uint16_t width, uint16_t height, PixelFormat fmt, uint32_t stride, uint64_t phy) noexcept {
    VideoFrame f;
    if (plane_count(fmt) != 1) {
        return f;  // invalid: packed() only describes single-plane formats
    }
    f.width = width;
    f.height = height;
    f.format = fmt;
    f.planes[0] = Plane(data, phy, stride != 0 ? stride : static_cast<uint32_t>(width) * bytes_per_pixel(fmt));
    return f;
}

bool VideoFrame::cpu_accessible() const noexcept {
    const uint8_t n = plane_count(format);
    for (uint8_t i = 0; i < n; ++i) {
        if (planes[i].vir == nullptr) {
            return false;
        }
    }
    return n != 0;
}

namespace {

/** Advance a plane by a byte offset, keeping null / zero addresses null / zero. */
Plane offset_plane(const Plane& p, size_t off) noexcept {
    return Plane(p.vir ? p.vir + off : nullptr, p.phy ? p.phy + off : 0, static_cast<uint32_t>(p.stride));
}

}  // namespace

VideoFrame VideoFrame::sub_frame(const Rect& r) const noexcept {
    VideoFrame out;  // invalid until filled in
    if (!valid()) {
        return out;
    }

    // Clip to the frame using 64-bit math so huge rects cannot overflow.
    int64_t x0 = std::max<int64_t>(r.x, 0);
    int64_t y0 = std::max<int64_t>(r.y, 0);
    int64_t x1 = std::min<int64_t>(static_cast<int64_t>(r.x) + r.w, width);
    int64_t y1 = std::min<int64_t>(static_cast<int64_t>(r.y) + r.h, height);

    if (x1 <= x0 || y1 <= y0) {
        return out;  // reject before NV21 growth can turn an empty rect into a 2x2 one
    }
    if (format == PixelFormat::NV21) {
        // Chroma is subsampled 2x2: grow to the smallest even-aligned rect
        // covering the request, without leaving the (even-sized) frame.
        x0 &= ~int64_t{1};
        y0 &= ~int64_t{1};
        x1 = std::min<int64_t>((x1 + 1) & ~int64_t{1}, width & ~1);
        y1 = std::min<int64_t>((y1 + 1) & ~int64_t{1}, height & ~1);
    }
    if (x1 <= x0 || y1 <= y0) {
        return out;
    }

    out = *this;
    out.width = static_cast<uint16_t>(x1 - x0);
    out.height = static_cast<uint16_t>(y1 - y0);

    const size_t bpp = bytes_per_pixel(format);
    const uint8_t n = plane_count(format);
    for (uint8_t i = 0; i < n; ++i) {
        // NV21 VU plane (i == 1): half the rows, same byte offset per column
        // (one V,U pair per 2 pixels == 1 byte per pixel).
        const size_t row = (format == PixelFormat::NV21 && i == 1) ? static_cast<size_t>(y0 / 2) : static_cast<size_t>(y0);
        out.planes[i] = offset_plane(planes[i], row * planes[i].stride + static_cast<size_t>(x0) * bpp);
    }
    return out;
}

}  // namespace af
