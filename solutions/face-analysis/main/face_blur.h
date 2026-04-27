#ifndef _FACE_BLUR_H_
#define _FACE_BLUR_H_

#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <array>

#include <cvi_region.h>

#include "face_detector.h"

namespace face_analysis {

// Slot for OVERLAYEX RGN region
struct Slot {
    RGN_HANDLE handle;
    bool show;
    int last_render_frame;    // frame_id when bitmap was last rendered
    FaceInfo last_render_box; // the box used at last bitmap render (for IoU check)
    int rendered_w;           // current bitmap width
    int rendered_h;           // current bitmap height
};

// 1D Kalman filter with constant-velocity model
// State: [position, velocity], Observation: [position]
struct KalmanFilter1D {
    float x;    // position estimate
    float v;    // velocity estimate
    float p00;  // var(position)
    float p01;  // cov(position, velocity)
    float p11;  // var(velocity)

    void init(float x0, float pos_var = 0.01f, float vel_var = 1.0f);
    void predict(float dt, float q);
    void update(float z, float r);
};

// Tracked bounding box with 4 independent Kalman filters (x, y, w, h)
struct TrackedRegion {
    KalmanFilter1D kf[4];  // center_x, center_y, width, height
    float score;
    int miss_count;

    void init(const FaceInfo& face);
    void predict(float dt, float q);
    void update(const FaceInfo& face, float r);
    FaceInfo getBox() const;
};

class FaceBlur {
public:
    FaceBlur();
    ~FaceBlur();

    // Initialize with stream resolution and VPSS channel info
    bool init(int stream_width, int stream_height, int vpss_grp = 0, int vpss_chn = 2);

    // Deinitialize and release RGN resources
    void deinit();

    // Set max number of blur regions (1-16)
    void setMaxRegions(int max_regions);

    // Feed detection results to update tracking (backward-compatible, calls 3-arg form)
    void onDetection(const std::vector<FaceInfo>& faces);

    // Feed detection results with frame data for bitmap rendering
    void onDetection(const std::vector<FaceInfo>& faces, const ma_img_t* frame, uint32_t frame_id);

private:
    void initRegions();
    void deinitRegions();
    void applyRegions(const std::vector<FaceInfo>& boxes);
    void applyDetections(const std::vector<FaceInfo>& boxes, const ma_img_t* frame, uint32_t frame_id);
    void associateAndUpdate(const std::vector<FaceInfo>& faces);
    float computeIoU(const FaceInfo& a, const FaceInfo& b);
    void predictThreadEntry();
    void renderMosaicBitmap(const ma_img_t* frame, const FaceInfo& box_norm, int target_w, int target_h, int block_size, std::vector<uint8_t>& argb_out, int& out_w, int& out_h);
    void mapNormToStream(const FaceInfo& norm, int& sx, int& sy, int& sw, int& sh);

private:
    static constexpr int kRgnHandleBase = 100;
    static constexpr int kDefaultMaxRegions = 8;
    static constexpr int kMaxRegionsLimit = 8;  // RGN_MOSAIC_MAX_NUM per channel on CV181x

    // RGN hardware overlay config
    int max_regions_;
    int vpss_grp_;
    int vpss_chn_;
    std::vector<Slot> slots_;
    bool regions_inited_;

    // Bitmap render cadence
    int bitmap_refresh_frames_ = 5;  // re-render bitmap if last_render_frame is older than this
    int max_bitmap_w_ = 256;         // max bitmap allocated per slot
    int max_bitmap_h_ = 256;

    // Stream resolution
    int stream_width_;
    int stream_height_;

    // Kalman prediction tracking
    std::vector<TrackedRegion> trackers_;
    std::mutex tracker_mutex_;
    // Lock ordering: tracker_mutex_ MUST NOT be held when acquiring rgn_mutexes_[i]. rgn_mutexes_ are leaf locks.
    // Serializes SetBitMap + SetDisplayAttr on the same RGN handle between detection thread and predict thread.
    std::array<std::mutex, 8> rgn_mutexes_;
    std::thread predict_thread_;
    std::atomic<bool> predicting_;
    float process_noise_;
    float measurement_noise_;
    int max_miss_;
    int predict_interval_ms_;
    float iou_threshold_;

    bool initialized_;
};

}  // namespace face_analysis

#endif  // _FACE_BLUR_H_
