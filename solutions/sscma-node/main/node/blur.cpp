#include "blur.h"
#include "model.h"

#include <algorithm>
#include <cmath>

namespace ma::node {

static constexpr char TAG[] = "ma::node::blur";

// ============ KalmanFilter1D ============
// Constant-velocity model: F = [[1,dt],[0,1]], H = [1,0]
// Process noise: Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

void KalmanFilter1D::init(float x0, float pos_var, float vel_var) {
    x   = x0;
    v   = 0.0f;
    p00 = pos_var;
    p01 = 0.0f;
    p11 = vel_var;
}

void KalmanFilter1D::predict(float dt, float q) {
    // State prediction
    x += v * dt;
    // v unchanged (constant velocity assumption)

    // Covariance prediction: P = F*P*F^T + Q
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;

    p00 = p00 + 2.0f * p01 * dt + p11 * dt2 + q * dt4 * 0.25f;
    p01 = p01 + p11 * dt + q * dt3 * 0.5f;
    p11 = p11 + q * dt2;
}

void KalmanFilter1D::update(float z, float r) {
    float y  = z - x;       // innovation
    float s  = p00 + r;     // innovation covariance
    float k0 = p00 / s;     // Kalman gain for position
    float k1 = p01 / s;     // Kalman gain for velocity

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

void TrackedRegion::init(const ma_bbox_t& box) {
    kf[0].init(box.x);                    // center x
    kf[1].init(box.y);                    // center y
    kf[2].init(box.w, 0.01f, 0.1f);      // width  (slower dynamics)
    kf[3].init(box.h, 0.01f, 0.1f);      // height (slower dynamics)
    target     = box.target;
    score      = box.score;
    miss_count = 0;
}

void TrackedRegion::predict(float dt, float q) {
    for (int i = 0; i < 4; i++) {
        kf[i].predict(dt, q);
    }
}

void TrackedRegion::update(const ma_bbox_t& box, float r) {
    kf[0].update(box.x, r);
    kf[1].update(box.y, r);
    kf[2].update(box.w, r * 2.0f);   // size measurements slightly less trusted
    kf[3].update(box.h, r * 2.0f);
    target     = box.target;
    score      = box.score;
    miss_count = 0;
}

ma_bbox_t TrackedRegion::getBox() const {
    ma_bbox_t b;
    b.x      = kf[0].x;
    b.y      = kf[1].x;
    b.w      = std::max(0.01f, kf[2].x);
    b.h      = std::max(0.01f, kf[3].x);
    b.target = target;
    b.score  = score;
    return b;
}

// ============ BlurNode ============

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
      stream_height_(0),
      predict_thread_(nullptr),
      predicting_(false),
      process_noise_(5.0f),
      measurement_noise_(0.001f),
      max_miss_(15),
      predict_interval_ms_(33),
      iou_threshold_(0.2f) {}

BlurNode::~BlurNode() {
    onDestroy();
}

// ============ IoU computation ============

float BlurNode::computeIoU(const ma_bbox_t& a, const ma_bbox_t& b) {
    // Boxes are in center format (x, y = center, w, h = dimensions)
    float a_l = a.x - a.w * 0.5f, a_r = a.x + a.w * 0.5f;
    float a_t = a.y - a.h * 0.5f, a_b = a.y + a.h * 0.5f;
    float b_l = b.x - b.w * 0.5f, b_r = b.x + b.w * 0.5f;
    float b_t = b.y - b.h * 0.5f, b_b = b.y + b.h * 0.5f;

    float inter_w = std::max(0.0f, std::min(a_r, b_r) - std::max(a_l, b_l));
    float inter_h = std::max(0.0f, std::min(a_b, b_b) - std::max(a_t, b_t));
    float inter   = inter_w * inter_h;

    float uni = a.w * a.h + b.w * b.h - inter;
    return (uni > 1e-6f) ? inter / uni : 0.0f;
}

// ============ Data association + Kalman update ============

void BlurNode::associateAndUpdate(const std::vector<ma_bbox_t>& boxes) {
    // Filter by target classes
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

    std::vector<bool> det_matched(filtered.size(), false);

    // Greedy IoU matching: for each existing tracker, find best matching detection
    for (auto& tracker : trackers_) {
        ma_bbox_t predicted = tracker.getBox();
        float best_iou = iou_threshold_;
        int best_idx   = -1;

        for (int d = 0; d < (int)filtered.size(); d++) {
            if (det_matched[d]) continue;
            float iou = computeIoU(predicted, filtered[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = d;
            }
        }

        if (best_idx >= 0) {
            tracker.update(filtered[best_idx], measurement_noise_);
            det_matched[best_idx] = true;
        } else {
            tracker.miss_count++;
        }
    }

    // Create new trackers for unmatched detections
    for (int d = 0; d < (int)filtered.size(); d++) {
        if (!det_matched[d] && (int)trackers_.size() < max_regions_) {
            TrackedRegion tr;
            tr.init(filtered[d]);
            trackers_.push_back(tr);
        }
    }

    // Remove dead trackers (too many consecutive misses)
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [this](const TrackedRegion& t) { return t.miss_count > max_miss_; }),
        trackers_.end());
}

// ============ Prediction thread ============

void BlurNode::predictThreadEntryStub(void* obj) {
    reinterpret_cast<BlurNode*>(obj)->predictThreadEntry();
}

void BlurNode::predictThreadEntry() {
    MA_LOGI(TAG, "Prediction thread started (%d ms interval, q=%.3f, r=%.4f)",
            predict_interval_ms_, process_noise_, measurement_noise_);

    float dt = (float)predict_interval_ms_ / 1000.0f;

    while (predicting_) {
        Thread::sleep(Tick::fromMilliseconds(predict_interval_ms_));

        if (!predicting_) break;

        if (!enabled_ || !regions_inited_) {
            if (regions_inited_) {
                std::vector<ma_bbox_t> empty;
                applyRegions(empty);
            }
            continue;
        }

        std::vector<ma_bbox_t> predicted_boxes;
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);

            // Predict all trackers forward by dt
            for (auto& tracker : trackers_) {
                tracker.predict(dt, process_noise_);
            }

            // Collect predicted boxes from active trackers
            for (const auto& tracker : trackers_) {
                if (tracker.miss_count <= max_miss_) {
                    predicted_boxes.push_back(tracker.getBox());
                }
            }
        }

        applyRegions(predicted_boxes);
    }

    MA_LOGI(TAG, "Prediction thread stopped");
}

// ============ Detection callback ============

void BlurNode::onDetection(const std::vector<ma_bbox_t>& boxes) {
    if (!enabled_ || !regions_inited_) return;

    std::lock_guard<std::mutex> lock(tracker_mutex_);
    associateAndUpdate(boxes);
}

// ============ RGN hardware overlay management ============

void BlurNode::initRegions() {
    if (regions_inited_) return;

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
        stChnAttr.unChnAttr.stCoverExChn.stRect.s32X      = 0;
        stChnAttr.unChnAttr.stCoverExChn.stRect.s32Y      = 0;
        stChnAttr.unChnAttr.stCoverExChn.stRect.u32Width   = 64;
        stChnAttr.unChnAttr.stCoverExChn.stRect.u32Height  = 64;
        stChnAttr.unChnAttr.stCoverExChn.u32Color          = cover_color_;
        stChnAttr.unChnAttr.stCoverExChn.u32Layer          = i;
        stChnAttr.unChnAttr.stCoverExChn.enCoverType       = AREA_RECT;

        ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
            CVI_RGN_Destroy(hRgn);
            continue;
        }

        handles_[i] = hRgn;
    }

    regions_inited_ = true;
    MA_LOGI(TAG, "Initialized %d blur regions on VPSS(%d,%d)", max_regions_, vpss_grp_, vpss_chn_);
}

void BlurNode::deinitRegions() {
    if (!regions_inited_) return;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (int i = 0; i < (int)handles_.size(); i++) {
        CVI_RGN_DetachFromChn(handles_[i], &stChn);
        CVI_RGN_Destroy(handles_[i]);
    }

    handles_.clear();
    regions_inited_ = false;
    MA_LOGI(TAG, "Deinitialized blur regions");
}

// Apply predicted/tracked boxes to RGN hardware overlays
void BlurNode::applyRegions(const std::vector<ma_bbox_t>& boxes) {
    if (!regions_inited_ || stream_width_ <= 0 || stream_height_ <= 0) return;

    // Coordinate mapping: normalized [0,1] → H264 stream pixel coords
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

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    int active_count = std::min((int)boxes.size(), max_regions_);

    for (int i = 0; i < max_regions_; i++) {
        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = COVEREX_RGN;

        if (i < active_count) {
            const auto& box = boxes[i];

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

// ============ Node lifecycle ============

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

    if (config.contains("process_noise") && config["process_noise"].is_number()) {
        process_noise_ = config["process_noise"].get<float>();
    }

    if (config.contains("measurement_noise") && config["measurement_noise"].is_number()) {
        measurement_noise_ = config["measurement_noise"].get<float>();
    }

    if (config.contains("max_miss") && config["max_miss"].is_number_integer()) {
        max_miss_ = config["max_miss"].get<int>();
    }

    if (config.contains("predict_fps") && config["predict_fps"].is_number_integer()) {
        int fps = config["predict_fps"].get<int>();
        if (fps > 0) {
            predict_interval_ms_ = 1000 / fps;
        }
    }

    if (config.contains("iou_threshold") && config["iou_threshold"].is_number()) {
        iou_threshold_ = config["iou_threshold"].get<float>();
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

    MA_LOGI(TAG, "Blur node starting: stream %dx%d, vpss(%d,%d), max_regions=%d, predict=%dms",
            stream_width_, stream_height_, vpss_grp_, vpss_chn_, max_regions_, predict_interval_ms_);

    // Register blur callback with model node
    model_->setBlurCallback([this](const std::vector<ma_bbox_t>& boxes) {
        this->onDetection(boxes);
    });

    // Initialize RGN hardware overlays
    initRegions();

    // Start prediction thread for smooth interpolation
    predicting_ = true;
    predict_thread_ = new Thread((type_ + "#" + id_ + "_predict").c_str(),
                                 &BlurNode::predictThreadEntryStub, this);
    predict_thread_->start(this);

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
                // Clear trackers and hide all regions when disabled
                {
                    std::lock_guard<std::mutex> lock(tracker_mutex_);
                    trackers_.clear();
                }
                std::vector<ma_bbox_t> empty;
                applyRegions(empty);
            }
        }
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", enabled_.load()}}));
    } else if (control == "targets" && data.is_array()) {
        targets_ = data.get<std::vector<int>>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", targets_}}));
    } else if (control == "color" && data.is_number_integer()) {
        cover_color_ = data.get<uint32_t>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", cover_color_}}));
    } else if (control == "process_noise" && data.is_number()) {
        process_noise_ = data.get<float>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", process_noise_}}));
    } else if (control == "measurement_noise" && data.is_number()) {
        measurement_noise_ = data.get<float>();
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", control}, {"code", MA_OK}, {"data", measurement_noise_}}));
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

    // Stop prediction thread
    predicting_ = false;
    if (predict_thread_ != nullptr) {
        predict_thread_->join();
        delete predict_thread_;
        predict_thread_ = nullptr;
    }

    // Clear trackers
    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        trackers_.clear();
    }

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
