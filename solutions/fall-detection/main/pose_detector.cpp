#include "pose_detector.h"

#include <algorithm>
#include <vector>

#define TAG "PoseDetector"

namespace fall {
namespace {

float boxIou(const geometry::InferBox& a, const geometry::InferBox& b) {
    const float left = std::max(a.left(), b.left());
    const float top = std::max(a.top(), b.top());
    const float right = std::min(a.right(), b.right());
    const float bottom = std::min(a.bottom(), b.bottom());
    const float intersection = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
    const float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
    const float union_area = area_a + area_b - intersection;
    return union_area > 1e-8f ? intersection / union_area : 0.0f;
}

geometry::InferBox resultBox(const ma_keypoint3f_t& result) {
    return geometry::InferBox::fromCenter(result.box.x, result.box.y,
                                          result.box.w, result.box.h,
                                          result.box.score);
}

}  // namespace

PoseDetector::~PoseDetector() {
    if (model_) {
        ma::ModelFactory::remove(model_);
        model_ = nullptr;
    }
}

bool PoseDetector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Loading pose model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize CVI engine");
        return false;
    }
    if (engine_->load(model_path.c_str()) != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    ma::Model* m = ma::ModelFactory::create(engine_.get());
    if (m == nullptr) {
        MA_LOGE(TAG, "ModelFactory::create failed - model format not recognized");
        return false;
    }
    if (m->getOutputType() != MA_OUTPUT_TYPE_KEYPOINT) {
        // A detection model loads and runs perfectly well; it just never
        // produces a keypoint, so fall analysis would silently see no usable
        // joints. Fail loudly instead.
        MA_LOGE(TAG, "Model '%s' is not a pose model (output type 0x%04X, need KEYPOINT). "
                     "Point --model at a *_pose_*.cvimodel.",
                m->getName(), m->getOutputType());
        ma::ModelFactory::remove(m);
        return false;
    }

    model_ = static_cast<ma::model::PoseDetector*>(m);
    model_->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold_);

    if (const auto* in = static_cast<const ma_img_t*>(model_->getInput())) {
        input_width_ = in->width;
        input_height_ = in->height;
    }

    MA_LOGI(TAG, "Pose model ready (type: %s, input: %dx%d)", m->getName(), input_width_, input_height_);
    initialized_ = true;
    return true;
}

void PoseDetector::setThreshold(float person_score) {
    threshold_ = std::max(0.0f, std::min(1.0f, person_score));
    if (model_) {
        model_->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold_);
    }
}

const Subject* PoseDetector::detectPrimary(ma_img_t* img) {
    has_primary_ = false;
    inference_failed_ = false;
    if (!initialized_ || model_ == nullptr || img == nullptr) {
        return nullptr;
    }

    if (model_->run(img) != MA_OK) {
        inference_failed_ = true;
        MA_LOGE(TAG, "Pose inference failed");
        return nullptr;
    }

    std::vector<const ma_keypoint3f_t*> candidates;
    for (const auto& kp : model_->getResults()) {
        if (kp.box.score < threshold_) continue;
        candidates.push_back(&kp);
    }
    if (candidates.empty()) {
        if (++tracking_misses_ > 5) have_tracked_box_ = false;
        return nullptr;
    }

    const ma_keypoint3f_t* best = nullptr;
    if (have_tracked_box_) {
        float best_iou = 0.0f;
        for (const auto* candidate : candidates) {
            const float iou = boxIou(tracked_box_, resultBox(*candidate));
            if (best == nullptr || iou > best_iou) {
                best = candidate;
                best_iou = iou;
            }
        }
        // A few blank associations are safer than feeding another person's
        // pose into a 3.2-second history. Reacquire after a short grace period.
        if (best_iou < 0.05f && ++tracking_misses_ <= 3) return nullptr;
    }
    if (best == nullptr || tracking_misses_ > 3) {
        best = *std::max_element(candidates.begin(), candidates.end(),
            [](const auto* a, const auto* b) { return a->box.score < b->box.score; });
    }
    tracking_misses_ = 0;

    keypoint_count_ = static_cast<int>(best->pts.size());

    // YOLO pose heads emit box centre coordinates; say so at the only place
    // that knows, so nothing downstream has to guess.
    primary_.box = resultBox(*best);
    tracked_box_ = primary_.box;
    have_tracked_box_ = true;
    primary_.score = best->box.score;
    primary_.pose = Pose(best->pts, input_width_, input_height_, kpt_threshold_);
    has_primary_ = true;
    return &primary_;
}

}  // namespace fall
