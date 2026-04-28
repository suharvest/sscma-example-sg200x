#include "face_blur.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#include <sscma.h>
#include <opencv2/opencv.hpp>

#define TAG "FaceBlur"

namespace face_analysis {

// ============ KalmanFilter1D ============

void KalmanFilter1D::init(float x0, float pos_var, float vel_var) {
    x   = x0;
    v   = 0.0f;
    p00 = pos_var;
    p01 = 0.0f;
    p11 = vel_var;
}

void KalmanFilter1D::predict(float dt, float q) {
    x += v * dt;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;

    p00 = p00 + 2.0f * p01 * dt + p11 * dt2 + q * dt4 * 0.25f;
    p01 = p01 + p11 * dt + q * dt3 * 0.5f;
    p11 = p11 + q * dt2;
}

void KalmanFilter1D::update(float z, float r) {
    float y  = z - x;
    float s  = p00 + r;
    float k0 = p00 / s;
    float k1 = p01 / s;

    x += k0 * y;
    v += k1 * y;

    float new_p00 = p00 - k0 * p00;
    float new_p01 = p01 - k0 * p01;
    float new_p11 = p11 - k1 * p01;

    p00 = new_p00;
    p01 = new_p01;
    p11 = new_p11;
}

// ============ TrackedRegion ============

void TrackedRegion::init(const FaceInfo& face) {
    kf[0].init(face.x);
    kf[1].init(face.y);
    kf[2].init(face.w, 0.01f, 0.1f);
    kf[3].init(face.h, 0.01f, 0.1f);
    score      = face.score;
    miss_count = 0;
}

void TrackedRegion::predict(float dt, float q) {
    for (int i = 0; i < 4; i++) {
        kf[i].predict(dt, q);
    }
}

void TrackedRegion::update(const FaceInfo& face, float r) {
    kf[0].update(face.x, r);
    kf[1].update(face.y, r);
    kf[2].update(face.w, r * 2.0f);
    kf[3].update(face.h, r * 2.0f);
    score      = face.score;
    miss_count = 0;
}

FaceInfo TrackedRegion::getBox() const {
    FaceInfo f;
    f.x     = kf[0].x;
    f.y     = kf[1].x;
    f.w     = std::max(0.01f, kf[2].x);
    f.h     = std::max(0.01f, kf[3].x);
    f.score = score;
    f.id    = 0;
    return f;
}

// ============ FaceBlur ============

FaceBlur::FaceBlur()
    : max_regions_(kDefaultMaxRegions),
      vpss_grp_(0),
      vpss_chn_(2),
      regions_inited_(false),
      stream_width_(0),
      stream_height_(0),
      predicting_(false),
      process_noise_(5.0f),
      measurement_noise_(0.001f),
      max_miss_(15),
      predict_interval_ms_(33),
      iou_threshold_(0.2f),
      initialized_(false) {}

FaceBlur::~FaceBlur() {
    deinit();
}

void FaceBlur::setMaxRegions(int max_regions) {
    if (max_regions < 1) max_regions = 1;
    if (max_regions > kMaxRegionsLimit) max_regions = kMaxRegionsLimit;
    max_regions_ = max_regions;
}


bool FaceBlur::init(int stream_width, int stream_height, int vpss_grp, int vpss_chn) {
    if (initialized_) return true;

    stream_width_  = stream_width;
    stream_height_ = stream_height;
    vpss_grp_      = vpss_grp;
    vpss_chn_      = vpss_chn;

    if (stream_width_ <= 0 || stream_height_ <= 0) {
        MA_LOGE(TAG, "Invalid stream resolution: %dx%d", stream_width_, stream_height_);
        return false;
    }

    MA_LOGI(TAG, "Initializing face blur: stream %dx%d, vpss(%d,%d), max_regions=%d",
            stream_width_, stream_height_, vpss_grp_, vpss_chn_, max_regions_);

    initRegions();

    if (!regions_inited_) {
        MA_LOGW(TAG, "No RGN regions available, face blur disabled");
        return false;
    }

    predicting_.store(true);
    predict_thread_ = std::thread(&FaceBlur::predictThreadEntry, this);

    initialized_ = true;
    MA_LOGI(TAG, "Face blur initialized with %d regions", (int)slots_.size());
    return true;
}

void FaceBlur::deinit() {
    if (!initialized_) return;

    predicting_.store(false);
    if (predict_thread_.joinable()) {
        predict_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        trackers_.clear();
    }

    deinitRegions();

    initialized_ = false;
    MA_LOGI(TAG, "Face blur deinitialized");
}

// ============ IoU computation ============

float FaceBlur::computeIoU(const FaceInfo& a, const FaceInfo& b) {
    // x,y are top-left, w,h are dimensions (ma_bbox_t convention)
    float a_l = a.x, a_r = a.x + a.w;
    float a_t = a.y, a_b = a.y + a.h;
    float b_l = b.x, b_r = b.x + b.w;
    float b_t = b.y, b_b = b.y + b.h;

    float inter_w = std::max(0.0f, std::min(a_r, b_r) - std::max(a_l, b_l));
    float inter_h = std::max(0.0f, std::min(a_b, b_b) - std::max(a_t, b_t));
    float inter   = inter_w * inter_h;

    float uni = a.w * a.h + b.w * b.h - inter;
    return (uni > 1e-6f) ? inter / uni : 0.0f;
}

// ============ Data association + Kalman update ============

void FaceBlur::associateAndUpdate(const std::vector<FaceInfo>& faces) {
    std::vector<bool> det_matched(faces.size(), false);

    for (auto& tracker : trackers_) {
        FaceInfo predicted = tracker.getBox();
        float best_iou = iou_threshold_;
        int best_idx   = -1;

        for (int d = 0; d < (int)faces.size(); d++) {
            if (det_matched[d]) continue;
            float iou = computeIoU(predicted, faces[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = d;
            }
        }

        if (best_idx >= 0) {
            tracker.update(faces[best_idx], measurement_noise_);
            det_matched[best_idx] = true;
        } else {
            tracker.miss_count++;
        }
    }

    // Remove dead trackers
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [this](const TrackedRegion& t) { return t.miss_count > max_miss_; }),
        trackers_.end());

    // Create new trackers for unmatched detections
    for (int d = 0; d < (int)faces.size(); d++) {
        if (det_matched[d]) continue;

        if ((int)trackers_.size() < max_regions_) {
            TrackedRegion tr;
            tr.init(faces[d]);
            trackers_.push_back(tr);
        } else {
            int worst_idx = -1;
            int worst_miss = 0;
            for (int t = 0; t < (int)trackers_.size(); t++) {
                if (trackers_[t].miss_count > worst_miss) {
                    worst_miss = trackers_[t].miss_count;
                    worst_idx  = t;
                }
            }
            if (worst_idx >= 0 && worst_miss > 0) {
                trackers_[worst_idx].init(faces[d]);
            }
        }
    }
}

// ============ Prediction thread ============

void FaceBlur::predictThreadEntry() {
    MA_LOGI(TAG, "Prediction thread started (%d ms interval)", predict_interval_ms_);

    float dt = (float)predict_interval_ms_ / 1000.0f;

    while (predicting_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(predict_interval_ms_));

        if (!predicting_.load()) break;

        if (!regions_inited_) continue;

        std::vector<FaceInfo> predicted_boxes;
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);

            for (auto& tracker : trackers_) {
                tracker.predict(dt, process_noise_);
            }

            for (const auto& tracker : trackers_) {
                if (tracker.miss_count <= max_miss_) {
                    predicted_boxes.push_back(tracker.getBox());
                }
            }

            std::sort(predicted_boxes.begin(), predicted_boxes.end(),
                [](const FaceInfo& a, const FaceInfo& b) {
                    return a.score > b.score;
                });
        }

        applyRegions(predicted_boxes);
    }

    MA_LOGI(TAG, "Prediction thread stopped");
}

// ============ RGN hardware overlay management ============

void FaceBlur::initRegions() {
    if (regions_inited_) return;

    slots_.clear();

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    // Cap max slots to hardware limit
    int num_slots = std::min(max_regions_, kMaxRegionsLimit);

    for (int i = 0; i < num_slots; i++) {
        RGN_HANDLE hRgn = kRgnHandleBase + i;

        RGN_ATTR_S stRgnAttr;
        memset(&stRgnAttr, 0, sizeof(stRgnAttr));
        stRgnAttr.enType = OVERLAYEX_RGN;
        stRgnAttr.unAttr.stOverlayEx.enPixelFormat = PIXEL_FORMAT_ARGB_8888;
        stRgnAttr.unAttr.stOverlayEx.u32CanvasNum = 2;
        stRgnAttr.unAttr.stOverlayEx.stSize.u32Width = max_bitmap_w_;
        stRgnAttr.unAttr.stOverlayEx.stSize.u32Height = max_bitmap_h_;

        CVI_S32 ret = CVI_RGN_Create(hRgn, &stRgnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_Create(%d, OVERLAYEX) failed: 0x%x", hRgn, ret);
            continue;
        }

        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.bShow  = CVI_FALSE;
        stChnAttr.enType = OVERLAYEX_RGN;
        stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32X = 0;
        stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32Y = 0;
        stChnAttr.unChnAttr.stOverlayExChn.u32Layer    = i;

        ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d, OVERLAYEX) failed: 0x%x", hRgn, ret);
            CVI_RGN_Destroy(hRgn);
            continue;
        }

        Slot slot;
        slot.handle = hRgn;
        slot.last_render_frame = -bitmap_refresh_frames_;  // force first render
        slot.rendered_w = max_bitmap_w_;
        slot.rendered_h = max_bitmap_h_;
        slots_.push_back(slot);
    }

    if (slots_.empty()) {
        MA_LOGE(TAG, "Failed to create any OVERLAYEX regions, disabling face blur");
        return;
    }

    regions_inited_ = true;
    MA_LOGI(TAG, "Initialized %d/%d OVERLAYEX regions on VPSS(%d,%d)",
            (int)slots_.size(), max_regions_, vpss_grp_, vpss_chn_);
}

void FaceBlur::deinitRegions() {
    if (!regions_inited_ && slots_.empty()) return;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (auto& slot : slots_) {
        CVI_RGN_DetachFromChn(slot.handle, &stChn);
        CVI_RGN_Destroy(slot.handle);
    }

    slots_.clear();
    regions_inited_ = false;
    MA_LOGI(TAG, "Deinitialized OVERLAYEX regions");
}

void FaceBlur::applyRegions(const std::vector<FaceInfo>& boxes) {
    // Called from predict thread - only update display POSITION (not bitmap)
    if (!regions_inited_ || stream_width_ <= 0 || stream_height_ <= 0) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    int num_slots = (int)slots_.size();
    int active_count = std::min((int)boxes.size(), num_slots);

    for (int i = 0; i < num_slots; i++) {
        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = OVERLAYEX_RGN;

        if (i < active_count) {
            const auto& box = boxes[i];
            int sx, sy, sw, sh;
            mapNormToStream(box, sx, sy, sw, sh);

            // 8-pixel alignment
            sx = sx & ~7;
            sy = sy & ~7;
            sw = std::max(8, (sw + 7) & ~7);
            sh = std::max(8, (sh + 7) & ~7);

            // Clamp to frame bounds
            sw = std::min(sw, stream_width_ - sx);
            sh = std::min(sh, stream_height_ - sy);
            sw = std::max(8, sw);
            sh = std::max(8, sh);

            stChnAttr.bShow = CVI_TRUE;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32X = sx;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32Y = sy;
            stChnAttr.unChnAttr.stOverlayExChn.u32Layer    = i;
        } else {
            stChnAttr.bShow = CVI_FALSE;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32X = 0;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32Y = 0;
            stChnAttr.unChnAttr.stOverlayExChn.u32Layer    = i;
        }

        std::lock_guard<std::mutex> rgn_lock(rgn_mutexes_[i]);
        CVI_S32 ret = CVI_RGN_SetDisplayAttr(slots_[i].handle, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", slots_[i].handle, ret);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    static int call_count = 0;
    static long long accum_us = 0;
    call_count++;
    accum_us += elapsed_us;

    if (call_count >= 30) {
        MA_LOGI(TAG, "applyRegions: avg per-call %lldus over last 30 calls", (long long)(accum_us / 30));
        call_count = 0;
        accum_us = 0;
    }
}

// ============ Coordinate mapping helper ============

void FaceBlur::mapNormToStream(const FaceInfo& norm, int& sx, int& sy, int& sw, int& sh) {
    // Coordinate mapping: normalized [0,1] -> H264 stream pixel coords
    // Handles letterbox padding for non-square aspect ratios
    float scale_h    = 1.0f;
    float scale_w    = 1.0f;
    int32_t offset_x = 0;
    int32_t offset_y = 0;

    if (stream_width_ > stream_height_) {
        scale_h  = (float)stream_width_ / (float)stream_height_;
        offset_y = (stream_height_ - stream_width_) / 2;
    } else {
        scale_w  = (float)stream_height_ / (float)stream_width_;
        offset_x = (stream_width_ - stream_height_) / 2;
    }

    int target_w = stream_width_ * scale_w;
    int target_h = stream_height_ * scale_h;

    sx = (int)(norm.x * target_w + offset_x);
    sy = (int)(norm.y * target_h + offset_y);
    sw = (int)(norm.w * target_w);
    sh = (int)(norm.h * target_h);

    // Coordinate clamping to prevent out-of-bounds regions
    sx = std::max(0, sx);
    sy = std::max(0, sy);
    if (sx >= stream_width_)  { sx = stream_width_ - 8;  if (sx < 0) sx = 0; }
    if (sy >= stream_height_) { sy = stream_height_ - 8; if (sy < 0) sy = 0; }
    sw = std::max(1, std::min(sw, stream_width_  - sx));
    sh = std::max(1, std::min(sh, stream_height_ - sy));
}

// ============ Mosaic bitmap rendering ============

void FaceBlur::renderMosaicBitmap(const ma_img_t* frame, const FaceInfo& box_norm,
                                   int target_w, int target_h, int block_size,
                                   std::vector<uint8_t>& argb_out, int& out_w, int& out_h) {
    out_w = 0;
    out_h = 0;
    if (!frame || frame->data == nullptr) return;

    // Map normalized box to inference frame pixel coords
    int frame_w = frame->width;
    int frame_h = frame->height;

    // Inference frame may have different aspect ratio - apply letterbox inverse
    float scale_h = 1.0f, scale_w = 1.0f;
    int32_t offset_x = 0, offset_y = 0;

    if (frame_w > frame_h) {
        scale_h = (float)frame_w / (float)frame_h;
        offset_y = (frame_h - frame_w) / 2;
    } else {
        scale_w = (float)frame_h / (float)frame_w;
        offset_x = (frame_w - frame_h) / 2;
    }

    int scaled_w = frame_w * scale_w;
    int scaled_h = frame_h * scale_h;

    int left   = (int)(box_norm.x * scaled_w + offset_x);
    int top    = (int)(box_norm.y * scaled_h + offset_y);
    int roi_w  = (int)(box_norm.w * scaled_w);
    int roi_h  = (int)(box_norm.h * scaled_h);

    // Clamp ROI to frame bounds
    left = std::max(0, std::min(left, frame_w - 1));
    top  = std::max(0, std::min(top, frame_h - 1));
    roi_w = std::max(1, std::min(roi_w, frame_w - left));
    roi_h = std::max(1, std::min(roi_h, frame_h - top));

    // Create cv::Mat from RGB888 frame
    cv::Mat rgb(frame_h, frame_w, CV_8UC3, frame->data);
    cv::Rect roi(left, top, roi_w, roi_h);
    cv::Mat face_roi = rgb(roi);

    // Mosaic: downsample with INTER_AREA, then upsample with INTER_NEAREST
    int down_w = std::max(1, roi_w / block_size);
    int down_h = std::max(1, roi_h / block_size);

    cv::Mat downsampled;
    cv::resize(face_roi, downsampled, cv::Size(down_w, down_h), 0, 0, cv::INTER_AREA);

    cv::Mat mosaic;
    cv::resize(downsampled, mosaic, cv::Size(roi_w, roi_h), 0, 0, cv::INTER_NEAREST);

    // Scale to target bitmap size (fit within max_bitmap_w/h)
    int bmp_w = std::min(target_w, max_bitmap_w_);
    int bmp_h = std::min(target_h, max_bitmap_h_);
    bmp_w = std::max(8, (bmp_w + 7) & ~7);  // 8-pixel align
    bmp_h = std::max(8, (bmp_h + 7) & ~7);

    cv::Mat bitmap_scaled;
    cv::resize(mosaic, bitmap_scaled, cv::Size(bmp_w, bmp_h), 0, 0, cv::INTER_AREA);

    out_w = bmp_w;
    out_h = bmp_h;

    // Convert RGB888 to ARGB8888 (alpha = 0xFF)
    argb_out.resize(bmp_w * bmp_h * 4);
    for (int y = 0; y < bmp_h; y++) {
        for (int x = 0; x < bmp_w; x++) {
            const cv::Vec3b& rgb = bitmap_scaled.at<cv::Vec3b>(y, x);
            int idx = (y * bmp_w + x) * 4;
            // Frame is RGB888 (R,G,B per pixel); CVI ARGB8888 wants memory layout B,G,R,A (LE 0xAARRGGBB)
            argb_out[idx + 0] = rgb[2];  // B (frame blue at byte 2)
            argb_out[idx + 1] = rgb[1];  // G
            argb_out[idx + 2] = rgb[0];  // R (frame red at byte 0)
            argb_out[idx + 3] = 0xFF;    // A
        }
    }
}

// ============ Apply detections with bitmap render ============

// TODO: verify ARGB8888 stride alignment on device — 32-byte align if SetBitMap rejects

void FaceBlur::applyDetections(const std::vector<FaceInfo>& boxes, const ma_img_t* frame, uint32_t frame_id) {
    if (!regions_inited_) return;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    int num_slots = (int)slots_.size();
    int active_count = std::min((int)boxes.size(), num_slots);

    long long render_us = 0;
    long long upload_us = 0;
    int n_renders = 0;
    int n_uploads = 0;

    for (int i = 0; i < num_slots; i++) {
        Slot& slot = slots_[i];

        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = OVERLAYEX_RGN;

        if (i < active_count) {
            const auto& box = boxes[i];
            int sx, sy, sw, sh;
            mapNormToStream(box, sx, sy, sw, sh);

            // 8-pixel alignment
            sx = sx & ~7;
            sy = sy & ~7;
            sw = std::max(8, (sw + 7) & ~7);
            sh = std::max(8, (sh + 7) & ~7);

            // Clamp to frame bounds
            sw = std::min(sw, stream_width_ - sx);
            sh = std::min(sh, stream_height_ - sy);
            sw = std::max(8, sw);
            sh = std::max(8, sh);

            // Render bitmap if cadence expired or IoU too low
            bool need_render = false;
            if (frame_id - slot.last_render_frame >= bitmap_refresh_frames_) {
                need_render = true;
            }
            if (!need_render && computeIoU(slot.last_render_box, box) < 0.7f) {
                need_render = true;
            }

            // Track whether we have a valid bitmap before attempting render
            bool have_valid_bitmap = (slot.last_render_frame >= 0);

            // Render BEFORE acquiring lock (cv::resize is compute-heavy)
            std::vector<uint8_t> argb_data;
            int bmp_w = 0, bmp_h = 0;
            if (need_render && frame && frame->data) {
                auto render_t0 = std::chrono::high_resolution_clock::now();
                renderMosaicBitmap(frame, box, sw, sh, 16, argb_data, bmp_w, bmp_h);
                auto render_t1 = std::chrono::high_resolution_clock::now();
                render_us += std::chrono::duration_cast<std::chrono::microseconds>(render_t1 - render_t0).count();
                n_renders++;
            }

            {
                std::lock_guard<std::mutex> rgn_lock(rgn_mutexes_[i]);

                // Upload bitmap if rendered successfully
                if (bmp_w > 0 && bmp_h > 0 && argb_data.size() == (size_t)(bmp_w * bmp_h * 4)) {
                    BITMAP_S stBitmap;
                    stBitmap.enPixelFormat = PIXEL_FORMAT_ARGB_8888;
                    stBitmap.u32Width      = bmp_w;
                    stBitmap.u32Height     = bmp_h;
                    stBitmap.pData         = argb_data.data();

                    auto upload_t0 = std::chrono::high_resolution_clock::now();
                    CVI_S32 ret = CVI_RGN_SetBitMap(slot.handle, &stBitmap);
                    auto upload_t1 = std::chrono::high_resolution_clock::now();
                    upload_us += std::chrono::duration_cast<std::chrono::microseconds>(upload_t1 - upload_t0).count();
                    n_uploads++;

                    if (ret != CVI_SUCCESS) {
                        MA_LOGW(TAG, "CVI_RGN_SetBitMap(%d) failed: 0x%x", slot.handle, ret);
                    } else {
                        have_valid_bitmap = true;
                        slot.last_render_frame = frame_id;
                        slot.last_render_box   = box;
                        slot.rendered_w        = bmp_w;
                        slot.rendered_h        = bmp_h;
                    }
                }

                // If no valid bitmap exists, force hide to avoid showing stale/empty content at new position
                if (!have_valid_bitmap) {
                    stChnAttr.bShow = CVI_FALSE;
                } else {
                    stChnAttr.bShow = CVI_TRUE;
                    stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32X = sx;
                    stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32Y = sy;
                    stChnAttr.unChnAttr.stOverlayExChn.u32Layer    = i;
                }

                CVI_S32 ret = CVI_RGN_SetDisplayAttr(slot.handle, &stChn, &stChnAttr);
                if (ret != CVI_SUCCESS) {
                    MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", slot.handle, ret);
                }
            }
        } else {
            stChnAttr.bShow = CVI_FALSE;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32X = 0;
            stChnAttr.unChnAttr.stOverlayExChn.stPoint.s32Y = 0;
            stChnAttr.unChnAttr.stOverlayExChn.u32Layer    = i;

            std::lock_guard<std::mutex> rgn_lock(rgn_mutexes_[i]);
            CVI_S32 ret = CVI_RGN_SetDisplayAttr(slot.handle, &stChn, &stChnAttr);
            if (ret != CVI_SUCCESS) {
                MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", slot.handle, ret);
            }
        }
    }

    if (n_uploads > 0) {
        MA_LOGI(TAG, "applyDetections: renders=%d (total %lldus), uploads=%d (total %lldus)",
                n_renders, (long long)render_us, n_uploads, (long long)upload_us);
    }
}

// ============ Detection callbacks ============

void FaceBlur::onDetection(const std::vector<FaceInfo>& faces) {
    onDetection(faces, nullptr, 0);
}

void FaceBlur::onDetection(const std::vector<FaceInfo>& faces, const ma_img_t* frame, uint32_t frame_id) {
    if (!initialized_ || !regions_inited_) return;

    std::vector<FaceInfo> active_boxes;
    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        associateAndUpdate(faces);
        for (const auto& tracker : trackers_) {
            if (tracker.miss_count <= max_miss_) {
                active_boxes.push_back(tracker.getBox());
            }
        }
        std::sort(active_boxes.begin(), active_boxes.end(),
            [](const FaceInfo& a, const FaceInfo& b) { return a.score > b.score; });
    }
    applyDetections(active_boxes, frame, frame_id);
}

}  // namespace face_analysis
