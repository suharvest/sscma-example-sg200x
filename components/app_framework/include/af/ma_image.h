/**
 * @file ma_image.h
 * @brief Bridge from af::VideoFrame to the sscma-micro ma_img_t used by model APIs.
 *
 * Kept out of frame.h so the core headers stay SDK-free (host tests); include
 * it from application code that calls ma::model / sscma APIs.
 */
#ifndef AF_MA_IMAGE_H_
#define AF_MA_IMAGE_H_

#include <sscma.h>

#include "af/frame.h"

namespace af {

/**
 * Describe a CPU-accessible packed RGB888 frame as an ma_img_t. No pixel copy:
 * data points at the frame's buffer, which must outlive the returned struct.
 * The model APIs have no stride field, so the frame must be tightly packed
 * (stride == width * 3), which is what the camera's RGB888 channel delivers.
 */
inline ma_img_t to_ma_img(const VideoFrame& f) noexcept {
    ma_img_t img{};
    img.width = f.width;
    img.height = f.height;
    img.size = static_cast<uint32_t>(f.planes[0].stride) * f.height;
    img.format = f.format == PixelFormat::Gray8 ? MA_PIXEL_FORMAT_GRAYSCALE : MA_PIXEL_FORMAT_RGB888;
    img.rotate = MA_PIXEL_ROTATE_0;
    img.physical = f.planes[0].vir == nullptr;
    img.data = img.physical ? reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(f.planes[0].phy)) : f.planes[0].vir;
    return img;
}

}  // namespace af

#endif  // AF_MA_IMAGE_H_
