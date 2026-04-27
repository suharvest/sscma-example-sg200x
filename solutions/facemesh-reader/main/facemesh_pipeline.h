#ifndef _FACEMESH_PIPELINE_H_
#define _FACEMESH_PIPELINE_H_

#include <string>
#include <vector>

#include <sscma.h>

#include "face_detector.h"
#include "facemesh_landmarker.h"
#include "facial_metrics.h"
#include "drowsiness_detector.h"
#include "yawn_detector.h"

namespace facemesh_reader {

struct AnalyzedFace {
    FaceInfo face;                   // YOLO-Face bbox + score (normalized 0-1)
    std::vector<Point2D> landmarks;  // 468 landmarks mapped to original frame pixel coords
    FaceMetrics metrics;             // EAR / MAR / per-frame flags
    // Phase 2: edge-autonomous drowsiness detection
    DrowsinessState drowsiness;      // composite state machine output
    YawnState       yawn;             // yawn-specific state
};

class FacemeshPipeline {
public:
    FacemeshPipeline() = default;
    ~FacemeshPipeline() = default;

    bool init(const std::string& facemesh_model_path);
    bool isReady() const { return landmarker_.isReady(); }

    // Process every detected face: ROI crop → 192x192 RGB → landmark → metrics.
    // full_frame must be RGB888 (MA_PIXEL_FORMAT_RGB888).
    // Returned AnalyzedFace.landmarks are in full_frame pixel coordinates.
    std::vector<AnalyzedFace> processAll(ma_img_t* full_frame,
                                          const std::vector<FaceInfo>& faces);

    void setBboxPadding(float pad) { bbox_padding_ = pad; }

    // Phase 2: configure drowsy + yawn thresholds (call after init()).
    void configureDrowsiness(const DrowsinessDetector::Config& cfg) {
        drowsiness_.configure(cfg);
    }
    void configureYawn(const YawnDetector::Config& cfg) {
        yawn_.configure(cfg);
    }

    // Reset per-driver state (e.g. new driver / session start).
    void resetDriverState() {
        drowsiness_.reset();
        yawn_.reset();
    }

private:
    // Crop bbox region (with padding) and resize (bilinear) to 192x192 RGB packed.
    // Returns crop rect (x1,y1,x2,y2) actually used in pixel coords via out_rect.
    std::vector<uint8_t> cropAndResize(const uint8_t* frame_rgb, int fw, int fh,
                                        const FaceInfo& face,
                                        float& out_x1, float& out_y1,
                                        float& out_x2, float& out_y2);

private:
    FacemeshLandmarker landmarker_;
    float bbox_padding_ = 0.10f;  // 10% outward padding around bbox
    // Phase 2: single-driver assumption — one detector instance, applied to the
    // first face only. TODO: per-face tracking by face.id for multi-occupant cabin.
    DrowsinessDetector drowsiness_;
    YawnDetector       yawn_;
};

}  // namespace facemesh_reader

#endif  // _FACEMESH_PIPELINE_H_
