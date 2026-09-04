#include "depth_overlay.h"

#include <algorithm>
#include <cstring>

#include <sscma.h>
#include <cvi_region.h>

#define TAG "DepthOverlay"

namespace depth {

namespace {

/* Region handle. Nothing else in this application creates regions (no privacy
 * blur, no OSD), so handle 0 is free. */
constexpr RGN_HANDLE kRgnHandle = 0;

/*
 * Near -> far colour ramp: red, orange, green, cyan, blue.
 *
 * Five stops rather than a continuous hue sweep because the eye reads the
 * boundaries between them as contour lines, which is what makes a depth preview
 * legible at 320x180.
 */
struct Stop { uint8_t r, g, b; };
constexpr Stop kRamp[5] = {
    {0xE5, 0x39, 0x35},  // nearest: red
    {0xFB, 0x8C, 0x00},  // orange
    {0x43, 0xA0, 0x47},  // green
    {0x00, 0xAC, 0xC1},  // cyan
    {0x1E, 0x40, 0xAF},  // farthest: blue
};

/* t in [0,1], 0 = nearest. */
inline uint32_t ramp_argb(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    const float pos = t * 4.0f;
    int i = static_cast<int>(pos);
    if (i > 3) i = 3;
    const float f = pos - static_cast<float>(i);
    const Stop& a = kRamp[i];
    const Stop& b = kRamp[i + 1];
    const uint32_t r = static_cast<uint32_t>(a.r + (b.r - a.r) * f);
    const uint32_t g = static_cast<uint32_t>(a.g + (b.g - a.g) * f);
    const uint32_t bl = static_cast<uint32_t>(a.b + (b.b - a.b) * f);
    return 0xFF000000u | (r << 16) | (g << 8) | bl;
}

}  // namespace

DepthOverlay::~DepthOverlay() { deinit(); }

bool DepthOverlay::init(int stream_w, int stream_h, int pip_w, int pip_h, int margin,
                        int vpss_grp, int vpss_chn) {
    if (ready_) return true;
    if (stream_w <= 0 || stream_h <= 0 || pip_w <= 0 || pip_h <= 0) return false;
    if (pip_w + margin > stream_w || pip_h + margin > stream_h) {
        MA_LOGE(TAG, "PiP %dx%d does not fit in the %dx%d stream",
                pip_w, pip_h, stream_w, stream_h);
        return false;
    }

    pip_w_    = pip_w;
    pip_h_    = pip_h;
    vpss_grp_ = vpss_grp;
    vpss_chn_ = vpss_chn;
    canvas_.assign(static_cast<size_t>(pip_w) * pip_h, 0u);

    RGN_ATTR_S attr;
    memset(&attr, 0, sizeof(attr));
    attr.enType                             = OVERLAY_RGN;
    attr.unAttr.stOverlay.enPixelFormat     = PIXEL_FORMAT_ARGB_8888;
    attr.unAttr.stOverlay.stSize.u32Width   = static_cast<CVI_U32>(pip_w);
    attr.unAttr.stOverlay.stSize.u32Height  = static_cast<CVI_U32>(pip_h);
    /* One canvas: 230 kB. A second would buy tear-free updates on a preview
     * that is redrawn 15 times a second anyway. */
    attr.unAttr.stOverlay.u32CanvasNum      = 1;
    attr.unAttr.stOverlay.u32BgColor        = 0;

    CVI_S32 ret = CVI_RGN_Create(kRgnHandle, &attr);
    if (ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "CVI_RGN_Create failed: 0x%x", ret);
        return false;
    }

    MMF_CHN_S chn;
    chn.enModId  = CVI_ID_VPSS;
    chn.s32DevId = vpss_grp_;
    chn.s32ChnId = vpss_chn_;

    /* Bottom-right, inset by `margin`. X is kept on a 16-pixel grid: the region
     * engine writes in bursts and an unaligned origin is rejected on some
     * pixel formats. */
    int x = stream_w - pip_w - margin;
    int y = stream_h - pip_h - margin;
    x &= ~0xF;
    y &= ~0x1;
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    RGN_CHN_ATTR_S chn_attr;
    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.bShow                                = CVI_FALSE;
    chn_attr.enType                               = OVERLAY_RGN;
    chn_attr.unChnAttr.stOverlayChn.stPoint.s32X  = x;
    chn_attr.unChnAttr.stOverlayChn.stPoint.s32Y  = y;
    chn_attr.unChnAttr.stOverlayChn.u32Layer      = 0;

    ret = CVI_RGN_AttachToChn(kRgnHandle, &chn, &chn_attr);
    if (ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "CVI_RGN_AttachToChn failed: 0x%x", ret);
        CVI_RGN_Destroy(kRgnHandle);
        return false;
    }

    handle_ = kRgnHandle;
    ready_  = true;
    MA_LOGI(TAG, "Depth PiP %dx%d at (%d,%d) on VPSS(%d,%d)",
            pip_w, pip_h, x, y, vpss_grp_, vpss_chn_);
    return true;
}

void DepthOverlay::deinit() {
    if (!ready_) return;

    MMF_CHN_S chn;
    chn.enModId  = CVI_ID_VPSS;
    chn.s32DevId = vpss_grp_;
    chn.s32ChnId = vpss_chn_;
    CVI_RGN_DetachFromChn(handle_, &chn);
    CVI_RGN_Destroy(handle_);

    handle_ = -1;
    ready_  = false;
    shown_  = false;
}

void DepthOverlay::update(const std::vector<float>& depth, int dw, int dh,
                          float p02, float p98) {
    if (!ready_ || dw <= 0 || dh <= 0) return;
    if (depth.size() != static_cast<size_t>(dw) * dh) return;

    const float span = p98 - p02;
    if (span <= 0.0f) return;

    /* Nearest-neighbour from the depth map onto the tile. The depth map already
     * covers exactly the sensor content the stream shows (the grey bars were
     * removed before inference, and preprocessing stretches rather than pads),
     * so this rescale is all the geometry there is: no letterbox to undo. */
    for (int y = 0; y < pip_h_; y++) {
        const int sy = std::min(dh - 1, y * dh / pip_h_);
        const float* srow = depth.data() + static_cast<size_t>(sy) * dw;
        uint32_t* drow    = canvas_.data() + static_cast<size_t>(y) * pip_w_;
        for (int x = 0; x < pip_w_; x++) {
            const int sx = std::min(dw - 1, x * dw / pip_w_);
            float t = (srow[sx] - p02) / span;  // 0 = nearest, 1 = farthest
            drow[x] = ramp_argb(t);
        }
    }

    BITMAP_S bmp;
    memset(&bmp, 0, sizeof(bmp));
    bmp.enPixelFormat = PIXEL_FORMAT_ARGB_8888;
    bmp.u32Width      = static_cast<CVI_U32>(pip_w_);
    bmp.u32Height     = static_cast<CVI_U32>(pip_h_);
    bmp.pData         = canvas_.data();

    const CVI_S32 ret = CVI_RGN_SetBitMap(handle_, &bmp);
    if (ret != CVI_SUCCESS) {
        MA_LOGW(TAG, "CVI_RGN_SetBitMap failed: 0x%x", ret);
        return;
    }

    if (!shown_) {
        MMF_CHN_S chn;
        chn.enModId  = CVI_ID_VPSS;
        chn.s32DevId = vpss_grp_;
        chn.s32ChnId = vpss_chn_;

        RGN_CHN_ATTR_S chn_attr;
        memset(&chn_attr, 0, sizeof(chn_attr));
        if (CVI_RGN_GetDisplayAttr(handle_, &chn, &chn_attr) != CVI_SUCCESS) return;
        chn_attr.bShow = CVI_TRUE;
        if (CVI_RGN_SetDisplayAttr(handle_, &chn, &chn_attr) == CVI_SUCCESS) {
            shown_ = true;
        }
    }
}

}  // namespace depth
