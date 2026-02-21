#include "blur.h"
#include "model.h"

namespace ma::node {

static constexpr char TAG[] = "ma::node::blur";

BlurNode::BlurNode(std::string id)
    : Node("blur", id),
      max_regions_(kDefaultMaxRegions),
      vpss_grp_(0),
      vpss_chn_(CHN_H264),
      cover_color_(0x000000),
      regions_inited_(false),
      camera_(nullptr),
      model_(nullptr),
      stream_width_(0),
      stream_height_(0) {}

BlurNode::~BlurNode() {
    onDestroy();
}

void BlurNode::initRegions() {
    if (regions_inited_) {
        return;
    }

    handles_.resize(max_regions_);

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (int i = 0; i < max_regions_; i++) {
        RGN_HANDLE hRgn = kRgnHandleBase + i;

        RGN_ATTR_S stRgnAttr;
        memset(&stRgnAttr, 0, sizeof(stRgnAttr));
        stRgnAttr.enType = COVEREX_RGN;

        CVI_S32 ret = CVI_RGN_Create(hRgn, &stRgnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_Create(%d) failed: 0x%x", hRgn, ret);
            continue;
        }

        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.bShow  = CVI_FALSE;
        stChnAttr.enType = COVEREX_RGN;
        stChnAttr.unChnAttr.stCoverExChn.stRect.s32X     = 0;
        stChnAttr.unChnAttr.stCoverExChn.stRect.s32Y     = 0;
        stChnAttr.unChnAttr.stCoverExChn.stRect.u32Width  = 64;
        stChnAttr.unChnAttr.stCoverExChn.stRect.u32Height = 64;
        stChnAttr.unChnAttr.stCoverExChn.u32Color         = cover_color_;
        stChnAttr.unChnAttr.stCoverExChn.u32Layer         = i;
        stChnAttr.unChnAttr.stCoverExChn.enCoverType      = AREA_RECT;

        ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
            CVI_RGN_Destroy(hRgn);
            continue;
        }

        handles_[i] = hRgn;
        MA_LOGD(TAG, "RGN handle %d created and attached to VPSS(%d,%d)", hRgn, vpss_grp_, vpss_chn_);
    }

    regions_inited_ = true;
    MA_LOGI(TAG, "Initialized %d blur regions on VPSS(%d,%d)", max_regions_, vpss_grp_, vpss_chn_);
}

void BlurNode::deinitRegions() {
    if (!regions_inited_) {
        return;
    }

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (int i = 0; i < (int)handles_.size(); i++) {
        RGN_HANDLE hRgn = handles_[i];
        CVI_RGN_DetachFromChn(hRgn, &stChn);
        CVI_RGN_Destroy(hRgn);
    }

    handles_.clear();
    regions_inited_ = false;
    MA_LOGI(TAG, "Deinitialized blur regions");
}

void BlurNode::updateRegions(const std::vector<ma_bbox_t>& boxes) {
    if (!regions_inited_ || stream_width_ <= 0 || stream_height_ <= 0) {
        return;
    }

    // Coordinate mapping: model normalized coords -> H264 stream pixel coords
    // The model outputs (x, y) as center, (w, h) as dimensions, all in [0,1]
    // These are relative to a letterboxed square input
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

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    // Filter boxes by target classes if specified
    std::vector<ma_bbox_t> filtered;
    for (const auto& box : boxes) {
        if (targets_.empty()) {
            filtered.push_back(box);
        } else {
            for (int t : targets_) {
                if (box.target == t) {
                    filtered.push_back(box);
                    break;
                }
            }
        }
    }

    int active_count = std::min((int)filtered.size(), max_regions_);

    for (int i = 0; i < max_regions_; i++) {
        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = COVEREX_RGN;

        if (i < active_count) {
            const auto& box = filtered[i];

            // Convert center coords to top-left corner
            int left = (int)((box.x - box.w / 2.0f) * target_w + offset_x);
            int top  = (int)((box.y - box.h / 2.0f) * target_h + offset_y);
            int w    = (int)(box.w * target_w);
            int h    = (int)(box.h * target_h);

            // Clamp to frame bounds
            left = std::max(0, left);
            top  = std::max(0, top);
            w    = std::min(w, stream_width_ - left);
            h    = std::min(h, stream_height_ - top);

            // Align to 2 pixels (hardware requirement)
            left = left & ~1;
            top  = top & ~1;
            w    = std::max(4, (w + 1) & ~1);
            h    = std::max(4, (h + 1) & ~1);

            // Re-check bounds after alignment
            if (left + w > stream_width_) w = stream_width_ - left;
            if (top + h > stream_height_) h = stream_height_ - top;
            w = std::max(4, w & ~1);
            h = std::max(4, h & ~1);

            stChnAttr.bShow = CVI_TRUE;
            stChnAttr.unChnAttr.stCoverExChn.stRect.s32X      = left;
            stChnAttr.unChnAttr.stCoverExChn.stRect.s32Y      = top;
            stChnAttr.unChnAttr.stCoverExChn.stRect.u32Width   = w;
            stChnAttr.unChnAttr.stCoverExChn.stRect.u32Height  = h;
            stChnAttr.unChnAttr.stCoverExChn.u32Color          = cover_color_;
            stChnAttr.unChnAttr.stCoverExChn.u32Layer          = i;
            stChnAttr.unChnAttr.stCoverExChn.enCoverType       = AREA_RECT;
        } else {
            // Hide unused regions
            stChnAttr.bShow = CVI_FALSE;
            stChnAttr.unChnAttr.stCoverExChn.stRect.s32X      = 0;
            stChnAttr.unChnAttr.stCoverExChn.stRect.s32Y      = 0;
            stChnAttr.unChnAttr.stCoverExChn.stRect.u32Width   = 64;
            stChnAttr.unChnAttr.stCoverExChn.stRect.u32Height  = 64;
            stChnAttr.unChnAttr.stCoverExChn.u32Color          = cover_color_;
            stChnAttr.unChnAttr.stCoverExChn.u32Layer          = i;
            stChnAttr.unChnAttr.stCoverExChn.enCoverType       = AREA_RECT;
        }

        CVI_S32 ret = CVI_RGN_SetDisplayAttr(handles_[i], &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", handles_[i], ret);
        }
    }
}

void BlurNode::onDetection(const std::vector<ma_bbox_t>& boxes) {
    if (!enabled_ || !regions_inited_) {
        return;
    }
    updateRegions(boxes);
}

ma_err_t BlurNode::onCreate(const json& config) {
    Guard guard(mutex_);

    if (config.contains("max_regions") && config["max_regions"].is_number_integer()) {
        max_regions_ = config["max_regions"].get<int>();
        if (max_regions_ < 1) max_regions_ = 1;
        if (max_regions_ > 8) max_regions_ = 8;
    }

    if (config.contains("vpss_grp") && config["vpss_grp"].is_number_integer()) {
        vpss_grp_ = config["vpss_grp"].get<int>();
    }

    if (config.contains("vpss_chn") && config["vpss_chn"].is_number_integer()) {
        vpss_chn_ = config["vpss_chn"].get<int>();
    }

    if (config.contains("color") && config["color"].is_number_integer()) {
        cover_color_ = config["color"].get<uint32_t>();
    }

    if (config.contains("targets") && config["targets"].is_array()) {
        targets_ = config["targets"].get<std::vector<int>>();
    }

    server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", "create"}, {"code", MA_OK}, {"data", ""}}));
    created_ = true;

    return MA_OK;
}

ma_err_t BlurNode::onStart() {
    Guard guard(mutex_);
    if (started_) {
        return MA_OK;
    }

    // Find camera and model dependencies
    for (auto& dep : dependencies_) {
        if (dep.second->type() == "camera") {
            camera_ = static_cast<CameraNode*>(dep.second);
        } else if (dep.second->type() == "model") {
            model_ = static_cast<ModelNode*>(dep.second);
        }
    }

    if (camera_ == nullptr) {
        MA_THROW(Exception(MA_ENOTSUP, "No camera node found"));
        return MA_ENOTSUP;
    }

    if (model_ == nullptr) {
        MA_THROW(Exception(MA_ENOTSUP, "No model node found"));
        return MA_ENOTSUP;
    }

    // Get H264 stream resolution from camera
    stream_width_  = camera_->getChannelWidth(CHN_H264);
    stream_height_ = camera_->getChannelHeight(CHN_H264);
    if (stream_width_ <= 0 || stream_height_ <= 0) {
        stream_width_  = 1920;
        stream_height_ = 1080;
    }

    MA_LOGI(TAG, "Blur node starting: stream %dx%d, vpss(%d,%d), max_regions=%d",
            stream_width_, stream_height_, vpss_grp_, vpss_chn_, max_regions_);

    // Register blur callback with model node
    model_->setBlurCallback([this](const std::vector<ma_bbox_t>& boxes) {
        this->onDetection(boxes);
    });

    // Initialize RGN hardware overlays
    initRegions();

    started_ = true;

    server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", "enabled"}, {"code", MA_OK}, {"data", enabled_.load()}}));

    return MA_OK;
}

ma_err_t BlurNode::onControl(const std::string& control, const json& data) {
    Guard guard(mutex_);

    if (control == "enabled" && data.is_boolean()) {
        bool enabled = data.get<bool>();
        if (enabled_.load() != enabled) {
            enabled_.store(enabled);
            if (!enabled) {
                // Hide all regions when disabled
                std::vector<ma_bbox_t> empty;
                updateRegions(empty);
            }
        }
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", enabled_.load()}}));
    } else if (control == "targets" && data.is_array()) {
        targets_ = data.get<std::vector<int>>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", targets_}}));
    } else if (control == "color" && data.is_number_integer()) {
        cover_color_ = data.get<uint32_t>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", cover_color_}}));
    } else {
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_ENOTSUP}, {"data", "Not supported"}}));
    }

    return MA_OK;
}

ma_err_t BlurNode::onStop() {
    Guard guard(mutex_);
    if (!started_) {
        return MA_OK;
    }

    started_ = false;

    // Remove blur callback from model
    if (model_ != nullptr) {
        model_->setBlurCallback(nullptr);
    }

    // Clean up RGN hardware overlays
    deinitRegions();

    return MA_OK;
}

ma_err_t BlurNode::onDestroy() {
    Guard guard(mutex_);

    if (!created_) {
        return MA_OK;
    }

    onStop();

    camera_ = nullptr;
    model_  = nullptr;

    created_ = false;

    return MA_OK;
}

REGISTER_NODE("blur", BlurNode);

}  // namespace ma::node
