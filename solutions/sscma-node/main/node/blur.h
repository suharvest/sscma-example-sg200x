#pragma once

#include "node.h"
#include "camera.h"

#include <cvi_rgn.h>

namespace ma::node {

class ModelNode;

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
    void updateRegions(const std::vector<ma_bbox_t>& boxes);

private:
    static constexpr int kRgnHandleBase = 100;  // Base handle ID to avoid conflicts
    static constexpr int kDefaultMaxRegions = 8;

    int max_regions_;
    int vpss_grp_;
    int vpss_chn_;
    uint32_t cover_color_;
    std::vector<int> targets_;  // Target class IDs to blur (empty = all)
    std::vector<RGN_HANDLE> handles_;
    bool regions_inited_;

    CameraNode* camera_;
    ModelNode* model_;

    int stream_width_;
    int stream_height_;
};

}  // namespace ma::node
