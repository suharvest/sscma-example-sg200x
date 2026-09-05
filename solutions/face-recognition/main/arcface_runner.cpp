#include "arcface_runner.h"

#include <cmath>

namespace face_recognition {

namespace {
void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}
}  // namespace

bool l2Normalize(Embedding512& v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    if (!(sum > 0.0) || !std::isfinite(sum)) return false;
    const float inv = static_cast<float>(1.0 / std::sqrt(sum));
    for (auto& x : v) x *= inv;
    return true;
}

float cosineSimilarity(const Embedding512& a, const Embedding512& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);
}

bool ArcFaceRunner::init(const std::string& model_path, std::string* error) {
    ready_ = false;
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float scale[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    if (!runner_.init(model_path, mean, scale, /*bgr=*/false, error)) return false;

    if (runner_.inputWidth() != 112 || runner_.inputHeight() != 112 ||
        runner_.inputChannels() != 3) {
        setError(error, "ArcFace input is not 112x112x3");
        return false;
    }
    if (runner_.outputCount() < 1) {
        setError(error, "ArcFace model has no outputs");
        return false;
    }
    size_t total = 0;
    for (int i = 0; i < runner_.outputCount(); ++i) {
        total += CviRunner::tensorNumel(runner_.output(i));
    }
    if (total < 512) {
        setError(error, "ArcFace outputs " + std::to_string(total) +
                            " values, expected at least 512");
        return false;
    }
    ready_ = true;
    return true;
}

bool ArcFaceRunner::infer(const uint8_t* aligned_rgb, Embedding512& normalized,
                          std::string* error) {
    if (!ready_) {
        setError(error, "ArcFace runner not ready");
        return false;
    }
    if (!runner_.run(aligned_rgb, error)) return false;

    size_t written = 0;
    for (int i = 0; i < runner_.outputCount() && written < normalized.size(); ++i) {
        const ma_tensor_t t = runner_.output(i);
        const size_t n = CviRunner::tensorNumel(t);
        for (size_t k = 0; k < n && written < normalized.size(); ++k) {
            normalized[written++] = CviRunner::readValue(t, k);
        }
    }
    if (written != normalized.size()) {
        setError(error, "ArcFace produced " + std::to_string(written) + " values");
        return false;
    }
    for (float v : normalized) {
        if (!std::isfinite(v)) {
            setError(error, "ArcFace produced a non-finite embedding");
            return false;
        }
    }
    if (!l2Normalize(normalized)) {
        setError(error, "ArcFace embedding has zero norm");
        return false;
    }
    return true;
}

}  // namespace face_recognition
