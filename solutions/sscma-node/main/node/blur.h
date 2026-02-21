#pragma once

#include "node.h"
#include "camera.h"

#include <cvi_rgn.h>
#include <mutex>

namespace ma::node {

class ModelNode;

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

    void init(const ma_bbox_t& box);
    void predict(float dt, float q);
    void update(const ma_bbox_t& box, float r);
    ma_bbox_t getBox() const;
};

class BlurNode : public Node {

public:
    BlurNode(std::string id);
    ~BlurNode();

    ma_err_t onCreate(const json& config) override;
    ma_err_t onStart() override;
    ma_err_t onControl(const std::string& control, const json& data) override;
    ma_err_t onStop() override;
    ma_err_t onDestroy() override;

    // Called from model node's inference thread with detection results
    void onDetection(const std::vector<ma_bbox_t>& boxes);

private:
    void initRegions();
    void deinitRegions();
    void applyRegions(const std::vector<ma_bbox_t>& boxes);
    void associateAndUpdate(const std::vector<ma_bbox_t>& boxes);
    float computeIoU(const ma_bbox_t& a, const ma_bbox_t& b);
    void predictThreadEntry();
    static void predictThreadEntryStub(void* obj);

private:
    static constexpr int kRgnHandleBase = 100;
    static constexpr int kDefaultMaxRegions = 8;

    // RGN hardware overlay config
    int max_regions_;
    int vpss_grp_;
    int vpss_chn_;
    uint32_t cover_color_;
    std::vector<int> targets_;
    std::vector<RGN_HANDLE> handles_;
    bool regions_inited_;

    // Dependencies
    CameraNode* camera_;
    ModelNode* model_;

    // Stream resolution
    int stream_width_;
    int stream_height_;

    // Kalman prediction tracking
    std::vector<TrackedRegion> trackers_;
    std::mutex tracker_mutex_;
    Thread* predict_thread_;
    bool predicting_;
    float process_noise_;
    float measurement_noise_;
    int max_miss_;
    int predict_interval_ms_;
    float iou_threshold_;
};

}  // namespace ma::node
