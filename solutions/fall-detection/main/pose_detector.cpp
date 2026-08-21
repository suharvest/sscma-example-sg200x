#include "pose_detector.h"

#include <algorithm>
#include <vector>

#define TAG "PoseDetector"

namespace fall {
namespace {

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

const std::vector<Subject>& PoseDetector::detectAll(ma_img_t* img) {
    subjects_.clear();
    inference_failed_ = false;
    if (!initialized_ || model_ == nullptr || img == nullptr) {
        return subjects_;
    }

    if (model_->run(img) != MA_OK) {
        inference_failed_ = true;
        MA_LOGE(TAG, "Pose inference failed");
        return subjects_;
    }

    std::vector<const ma_keypoint3f_t*> candidates;
    for (const auto& kp : model_->getResults()) {
        if (kp.box.score < threshold_) continue;
        candidates.push_back(&kp);
    }
    subjects_.reserve(candidates.size());
    keypoint_count_ = 0;
    for (const auto* candidate : candidates) {
        Subject subject;
        // YOLO pose heads emit box centre coordinates; say so at the only
        // place that knows, so nothing downstream has to guess.
        subject.box = resultBox(*candidate);
        subject.score = candidate->box.score;
        subject.pose = Pose(candidate->pts, input_width_, input_height_, kpt_threshold_);
        keypoint_count_ = std::max(keypoint_count_, static_cast<int>(candidate->pts.size()));
        subjects_.push_back(std::move(subject));
    }

    // Keep output deterministic for the MQTT/debug consumers. The tracker
    // still performs a global box assignment, so this ordering is only a
    // presentation detail and never selects a single analysis subject.
    std::sort(subjects_.begin(), subjects_.end(),
              [](const Subject& a, const Subject& b) { return a.score > b.score; });
    return subjects_;
}

}  // namespace fall
