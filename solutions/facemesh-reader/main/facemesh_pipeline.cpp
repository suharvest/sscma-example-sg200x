#include "facemesh_pipeline.h"

#include <algorithm>
#include <cmath>

#define TAG "FacemeshPipeline"

namespace facemesh_reader {

bool FacemeshPipeline::init(const std::string& facemesh_model_path) {
    if (!landmarker_.init(facemesh_model_path)) {
        MA_LOGE(TAG, "Failed to init FaceMesh landmarker");
        return false;
    }
    MA_LOGI(TAG, "Pipeline ready (input %dx%d)",
            landmarker_.inputW(), landmarker_.inputH());
    return true;
}

std::vector<uint8_t> FacemeshPipeline::cropAndResize(
        const uint8_t* frame_rgb, int fw, int fh,
        const FaceInfo& face,
        float& out_x1, float& out_y1,
        float& out_x2, float& out_y2) {

    const int dst = 192;
    std::vector<uint8_t> out((size_t)dst * dst * 3, 0);

    // Convert normalized bbox to pixel coords + apply padding.
    float x1 = face.x * fw;
    float y1 = face.y * fh;
    float x2 = (face.x + face.w) * fw;
    float y2 = (face.y + face.h) * fh;

    const float bw = std::max(1.0f, x2 - x1);
    const float bh = std::max(1.0f, y2 - y1);
    const float pad_x = bw * bbox_padding_;
    const float pad_y = bh * bbox_padding_;

    x1 = std::max(0.0f, x1 - pad_x);
    y1 = std::max(0.0f, y1 - pad_y);
    x2 = std::min((float)(fw - 1), x2 + pad_x);
    y2 = std::min((float)(fh - 1), y2 + pad_y);

    // Make square (FaceMesh expects square input). Expand the smaller side around center.
    float cx = 0.5f * (x1 + x2);
    float cy = 0.5f * (y1 + y2);
    float side = std::max(x2 - x1, y2 - y1);

    x1 = cx - 0.5f * side;
    y1 = cy - 0.5f * side;
    x2 = cx + 0.5f * side;
    y2 = cy + 0.5f * side;

    out_x1 = x1; out_y1 = y1; out_x2 = x2; out_y2 = y2;

    const float src_w = std::max(1.0f, x2 - x1);
    const float src_h = std::max(1.0f, y2 - y1);

    // Bilinear sampler — out-of-bounds reads clamped to edge (mirror-zero would also work).
    auto sample = [&](float sx, float sy, int c) -> uint8_t {
        if (sx < 0.f) sx = 0.f;
        if (sy < 0.f) sy = 0.f;
        if (sx > (float)(fw - 1)) sx = (float)(fw - 1);
        if (sy > (float)(fh - 1)) sy = (float)(fh - 1);
        const int x0 = (int)std::floor(sx);
        const int y0 = (int)std::floor(sy);
        const int x1i = std::min(x0 + 1, fw - 1);
        const int y1i = std::min(y0 + 1, fh - 1);
        const float ax = sx - x0;
        const float ay = sy - y0;
        const uint8_t* p00 = frame_rgb + (y0 * fw + x0) * 3;
        const uint8_t* p10 = frame_rgb + (y0 * fw + x1i) * 3;
        const uint8_t* p01 = frame_rgb + (y1i * fw + x0) * 3;
        const uint8_t* p11 = frame_rgb + (y1i * fw + x1i) * 3;
        const float v00 = (float)p00[c];
        const float v10 = (float)p10[c];
        const float v01 = (float)p01[c];
        const float v11 = (float)p11[c];
        const float v0 = v00 + (v10 - v00) * ax;
        const float v1 = v01 + (v11 - v01) * ax;
        const float v = v0 + (v1 - v0) * ay;
        const int iv = (int)std::lround(v);
        return (uint8_t)std::clamp(iv, 0, 255);
    };

    for (int dy = 0; dy < dst; ++dy) {
        const float sy = y1 + (dy + 0.5f) * (src_h / dst) - 0.5f;
        for (int dx = 0; dx < dst; ++dx) {
            const float sx = x1 + (dx + 0.5f) * (src_w / dst) - 0.5f;
            uint8_t* op = out.data() + ((size_t)dy * dst + dx) * 3;
            op[0] = sample(sx, sy, 0);
            op[1] = sample(sx, sy, 1);
            op[2] = sample(sx, sy, 2);
        }
    }
    return out;
}

std::vector<AnalyzedFace> FacemeshPipeline::processAll(
        ma_img_t* full_frame,
        const std::vector<FaceInfo>& faces) {

    std::vector<AnalyzedFace> results;
    if (!landmarker_.isReady() || !full_frame || !full_frame->data) {
        return results;
    }
    if (full_frame->width <= 0 || full_frame->height <= 0) {
        return results;
    }
    // We only support packed RGB888 input from camera retrieveFrame(MA_PIXEL_FORMAT_RGB888).
    // Other formats (YUV / NV21 / BGR) need explicit conversion not implemented here.

    const uint8_t* frame_ptr = static_cast<const uint8_t*>(full_frame->data);
    const int fw = full_frame->width;
    const int fh = full_frame->height;

    results.reserve(faces.size());

    for (const auto& face : faces) {
        AnalyzedFace af;
        af.face = face;

        float cx1, cy1, cx2, cy2;
        std::vector<uint8_t> roi = cropAndResize(frame_ptr, fw, fh, face,
                                                  cx1, cy1, cx2, cy2);
        if (roi.size() != 192 * 192 * 3) {
            results.push_back(af);
            continue;
        }

        std::vector<Point2D> lm192 = landmarker_.infer(roi.data());
        if (lm192.empty()) {
            results.push_back(af);
            continue;
        }

        // Map 192x192 input coordinates back to full-frame pixel coordinates.
        const float sx = (cx2 - cx1) / 192.0f;
        const float sy = (cy2 - cy1) / 192.0f;
        af.landmarks.reserve(lm192.size());
        for (const auto& p : lm192) {
            af.landmarks.push_back(Point2D{cx1 + p.x * sx, cy1 + p.y * sy});
        }

        af.metrics = computeMetrics(af.landmarks);
        results.push_back(af);
    }

    // Phase 2: drive the drowsiness state machine with the *primary* face.
    // (TODO: when supporting multi-occupant, key detectors by face.id.)
    if (!results.empty() && results.front().metrics.valid) {
        AnalyzedFace& primary = results.front();

        auto yawn_pair = yawn_.update(primary.metrics.mar);
        primary.yawn = yawn_pair.first;

        primary.drowsiness = drowsiness_.update(
            primary.metrics.avg_ear,
            yawn_pair.second,
            primary.yawn.yawn_count_5min);
    } else {
        // No valid face -> still tick yawn/drowsy with neutral inputs so the
        // PERCLOS window keeps shrinking and we don't accumulate stale closures.
        // Use MAR=0 (no yawn) and EAR above threshold (eyes "open").
        auto yawn_pair = yawn_.update(0.f);
        (void)drowsiness_.update(/*ear=*/1.0f, yawn_pair.second,
                                 yawn_pair.first.yawn_count_5min);
    }

    return results;
}

}  // namespace facemesh_reader
