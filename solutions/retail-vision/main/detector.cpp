#include "detector.h"

#include <algorithm>

#define TAG "Detector"

namespace retail_vision {

Detector::Detector()
    : engine_(nullptr),
      detector_(nullptr),
      threshold_(0.5f),
      input_width_(640),
      input_height_(640),
      initialized_(false) {}

Detector::~Detector() {
    if (detector_) {
        ma::ModelFactory::remove(detector_);
        detector_ = nullptr;
    }
}

bool Detector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Initializing detector with model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    ma_err_t ret = engine_->init();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize CVI engine");
        return false;
    }

    ret = engine_->load(model_path.c_str());
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    ma::Model* model = ma::ModelFactory::create(engine_.get());
    if (model == nullptr) {
        MA_LOGE(TAG, "Failed to create model from engine");
        return false;
    }

    if (model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
        MA_LOGE(TAG, "Model output type is not BBOX, got: %d", model->getOutputType());
        ma::ModelFactory::remove(model);
        return false;
    }

    detector_ = static_cast<ma::model::Detector*>(model);

    const ma_img_t* model_input = static_cast<const ma_img_t*>(detector_->getInput());
    if (model_input) {
        input_width_ = model_input->width;
        input_height_ = model_input->height;
    }

    MA_LOGI(TAG, "Detector initialized, input size: %dx%d", input_width_, input_height_);
    initialized_ = true;
    return true;
}

void Detector::setThreshold(float threshold) {
    threshold_ = std::max(0.0f, std::min(1.0f, threshold));
    if (detector_) {
        detector_->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold_);
    }
}

std::vector<DetectionBox> Detector::detect(ma_img_t* img) {
    std::vector<DetectionBox> results;

    if (!initialized_ || detector_ == nullptr || img == nullptr) {
        return results;
    }

    ma_err_t ret = detector_->run(img);
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Detection failed with error: %d", ret);
        return results;
    }

    auto bboxes = detector_->getResults();
    for (const auto& bbox : bboxes) {
        if (bbox.score >= threshold_) {
            DetectionBox det;
            det.x = bbox.x;
            det.y = bbox.y;
            det.w = bbox.w;
            det.h = bbox.h;
            det.score = bbox.score;
            det.target = bbox.target;
            results.push_back(det);
        }
    }

    return results;
}

}  // namespace retail_vision
