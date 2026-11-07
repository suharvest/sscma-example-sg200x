/**
 * @file frame.h
 * @brief VideoFrame: a 64-byte, trivially copyable, non-owning view of one video frame.
 *
 * A VideoFrame never owns pixel memory. It describes memory that somebody else
 * (the camera, a VB pool, a test buffer) keeps alive, so passing it by value
 * through a pipeline costs one cache line and no allocation.
 *
 * Each plane carries both addresses the SG2002 can use:
 *   - @c vir: CPU-visible pointer (nullptr when the frame was retrieved in
 *     physical mode and never mapped);
 *   - @c phy: physical address for TPU / IVE / VPSS (0 when unknown).
 *
 * sub_frame() crops by advancing the plane addresses -- the pixels are not
 * touched -- so a ROI handed to a second model is free.
 *
 * This header has no SDK dependency and is exercised by the host unit tests.
 */
#ifndef AF_FRAME_H_
#define AF_FRAME_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace af {

/** Pixel layouts the framework understands. Values are stable (used in logs). */
enum class PixelFormat : uint8_t {
    Unknown = 0,
    RGB888,        ///< packed R,G,B -- 1 plane, 3 B/px
    BGR888,        ///< packed B,G,R -- 1 plane, 3 B/px
    RGB888Planar,  ///< R / G / B planes -- 3 planes, 1 B/px each
    NV21,          ///< Y plane + interleaved VU plane at half resolution
    Gray8,         ///< 1 plane, 1 B/px
};

/** Number of planes a format uses (0 for Unknown). */
uint8_t plane_count(PixelFormat fmt) noexcept;

/** Bytes per pixel within one plane row (NV21: 1 for both Y and VU rows). */
uint8_t bytes_per_pixel(PixelFormat fmt) noexcept;

/** Human readable format name, for logs. */
const char* to_string(PixelFormat fmt) noexcept;

/** Axis-aligned rectangle in pixels. Negative origins are clipped by sub_frame(). */
struct Rect {
    int32_t x = 0;
    int32_t y = 0;
    int32_t w = 0;
    int32_t h = 0;

    constexpr bool empty() const noexcept { return w <= 0 || h <= 0; }
};

/**
 * One image plane.
 *
 * Packed into 16 bytes so that three planes plus the frame header fit in one
 * 64-byte cache line: physical addresses on SG2002 are well below 2^48 and no
 * supported stride exceeds 65535 bytes (4K RGB888 is 11520).
 */
struct Plane {
    uint8_t* vir = nullptr;  ///< CPU address of pixel (0,0) of this plane, or nullptr
    uint64_t phy : 48;       ///< physical address of pixel (0,0), or 0
    uint64_t stride : 16;    ///< bytes between consecutive rows

    constexpr Plane() noexcept : phy(0), stride(0) {}
    constexpr Plane(uint8_t* v, uint64_t p, uint32_t s) noexcept : vir(v), phy(p), stride(s) {}
};

/**
 * Non-owning view of a frame.
 *
 * Validity: a frame is valid when its format is known and it has a non-zero
 * size. A valid frame may still have vir == nullptr (physical-only frames);
 * check cpu_accessible() before reading pixels.
 */
struct VideoFrame {
    static constexpr int kMaxPlanes = 3;

    Plane planes[kMaxPlanes];                 ///< only the first plane_count(format) are meaningful
    uint16_t width = 0;                       ///< pixels
    uint16_t height = 0;                      ///< pixels
    PixelFormat format = PixelFormat::Unknown;
    uint8_t reserved_ = 0;
    uint16_t channel = 0;                     ///< source VPSS channel, informational
    uint32_t seq = 0;                         ///< monotonically increasing per source
    uint32_t ts_ms = 0;                       ///< capture time, monotonic milliseconds (wraps after ~49 days)

    /**
     * Wrap a single-plane packed buffer (RGB888 / BGR888 / Gray8).
     * @param stride bytes per row; 0 means tightly packed (width * bpp).
     */
    static VideoFrame packed(uint8_t* data, uint16_t width, uint16_t height, PixelFormat fmt, uint32_t stride = 0, uint64_t phy = 0) noexcept;

    bool valid() const noexcept { return format != PixelFormat::Unknown && width != 0 && height != 0; }

    /** True when every used plane has a CPU address. */
    bool cpu_accessible() const noexcept;

    /** Pointer to the first byte of row @p y in plane @p p (no bounds check). */
    uint8_t* row(int y, int p = 0) const noexcept { return planes[p].vir + static_cast<size_t>(y) * planes[p].stride; }

    /** Pointer to pixel (x, y) in a packed single-plane frame (no bounds check). */
    uint8_t* at(int x, int y) const noexcept { return row(y) + static_cast<size_t>(x) * bytes_per_pixel(format); }

    /** Bounds of the whole frame. */
    Rect rect() const noexcept { return Rect{0, 0, width, height}; }

    /**
     * Zero-copy crop.
     *
     * The rectangle is clipped to the frame. For NV21 it is first grown to
     * the smallest rectangle with even origin and even size that covers it,
     * so the chroma plane stays aligned. Returns an invalid frame (valid() == false) when the clipped
     * rectangle is empty or the format is Unknown. seq/ts_ms/channel are kept.
     */
    VideoFrame sub_frame(const Rect& r) const noexcept;
};

static_assert(sizeof(Plane) == 16, "Plane must stay 16 bytes");
static_assert(sizeof(VideoFrame) == 64, "VideoFrame must fit one cache line");
static_assert(std::is_trivially_copyable<VideoFrame>::value, "VideoFrame is passed by value");

}  // namespace af

#endif  // AF_FRAME_H_
