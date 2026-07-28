#include "pose_detector.h"

#include <algorithm>

#define TAG "PoseDetector"

namespace fitness {

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
        // produces a keypoint, so the rep counter would sit at zero with no
        // error anywhere. Fail loudly instead.
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
    if (!initialized_ || model_ == nullptr || img == nullptr) {
        return nullptr;
    }

    if (model_->run(img) != MA_OK) {
        MA_LOGE(TAG, "Pose inference failed");
        return nullptr;
    }

    const ma_keypoint3f_t* best = nullptr;
    for (const auto& kp : model_->getResults()) {
        if (kp.box.score < threshold_) continue;
        if (best == nullptr || kp.box.score > best->box.score) {
            best = &kp;
        }
    }
    if (best == nullptr) {
        return nullptr;
    }

    keypoint_count_ = static_cast<int>(best->pts.size());

    // YOLO pose heads emit box centre coordinates; say so at the only place
    // that knows, so nothing downstream has to guess.
    primary_.box = geometry::InferBox::fromCenter(best->box.x, best->box.y,
                                                  best->box.w, best->box.h,
                                                  best->box.score);
    primary_.score = best->box.score;
    primary_.pose = Pose(best->pts, input_width_, input_height_, kpt_threshold_);
    has_primary_ = true;
    return &primary_;
}

}  // namespace fitness
