#ifndef _REGION_BLUR_H_
#define _REGION_BLUR_H_

#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#include <cvi_rgn.h>

#include "detector.h"

namespace detection_blur {

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
    int target;
    float score;
    int miss_count;

    void init(const BBoxInfo& box);
    void predict(float dt, float q);
    void update(const BBoxInfo& box, float r);
    BBoxInfo getBox() const;
};

class RegionBlur {
public:
    RegionBlur();
    ~RegionBlur();

    // Initialize with stream resolution and VPSS channel info
    bool init(int stream_width, int stream_height, int vpss_grp = 0, int vpss_chn = 2);

    // Deinitialize and release RGN resources
    void deinit();

    // Set max number of blur regions (1-16)
    void setMaxRegions(int max_regions);

    // Set cover color (RGB, default black 0x000000)
    void setColor(uint32_t color);

    // Set target class IDs to blur (empty = blur all classes)
    void setTargets(const std::vector<int>& targets);

    // Feed detection results to update tracking
    void onDetection(const std::vector<BBoxInfo>& boxes);

private:
    void initRegions();
    void deinitRegions();
    void applyRegions(const std::vector<BBoxInfo>& boxes);
    void associateAndUpdate(const std::vector<BBoxInfo>& boxes);
    float computeIoU(const BBoxInfo& a, const BBoxInfo& b);
    void predictThreadEntry();

private:
    static constexpr int kRgnHandleBase = 100;
    static constexpr int kDefaultMaxRegions = 12;
    static constexpr int kMaxRegionsLimit = 16;  // SDK RGN_COVEREX_MAX_NUM per channel

    // RGN hardware overlay config
    int max_regions_;
    int vpss_grp_;
    int vpss_chn_;
    uint32_t cover_color_;
    std::vector<int> targets_;
    std::vector<RGN_HANDLE> handles_;
    bool regions_inited_;

    // Stream resolution
    int stream_width_;
    int stream_height_;

    // Kalman prediction tracking
    std::vector<TrackedRegion> trackers_;
    std::mutex tracker_mutex_;
    std::thread predict_thread_;
    std::atomic<bool> predicting_;
    float process_noise_;
    float measurement_noise_;
    int max_miss_;
    int predict_interval_ms_;
    float iou_threshold_;

    bool initialized_;
};

}  // namespace detection_blur

#endif  // _REGION_BLUR_H_
