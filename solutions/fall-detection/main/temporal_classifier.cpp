#include "temporal_classifier.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fall {
namespace {

constexpr int kJoints = 17;
constexpr int kBins = 6;
constexpr int kStride = 3;
static_assert(temporal_weights::kFrameDim == kJoints * 3 + 5, "unexpected frame layout");
static_assert(temporal_weights::kWindow % kBins == 0, "window must divide into bins");
static_assert(temporal_weights::kFeatureDim == temporal_weights::kFrameDim * (kBins + 3),
              "unexpected sequence feature layout");

float clipped(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

}  // namespace

TemporalFrame makeTemporalFrame(const Pose* pose, const FallObservation& observation,
                                int inference_width, int inference_height) {
    TemporalFrame out{};
    if (pose == nullptr || pose->empty() || inference_width <= 0 || inference_height <= 0) {
        return out;
    }

    std::array<float, kJoints> x{};
    std::array<float, kJoints> y{};
    std::array<float, kJoints> confidence{};
    for (int i = 0; i < kJoints; ++i) {
        const Joint joint = static_cast<Joint>(i);
        const Point2f point = pose->at(joint);
        x[i] = point.x / static_cast<float>(inference_width);
        y[i] = point.y / static_cast<float>(inference_height);
        confidence[i] = clipped(pose->confidence(joint), 0.0f, 1.0f);
    }

    auto midpoint = [&](int a, int b, float& mx, float& my, float& mean_confidence) {
        const float weight = confidence[a] + confidence[b];
        if (weight < 0.1f) {
            mx = my = mean_confidence = 0.0f;
            return;
        }
        mx = (x[a] * confidence[a] + x[b] * confidence[b]) / weight;
        my = (y[a] * confidence[a] + y[b] * confidence[b]) / weight;
        mean_confidence = weight * 0.5f;
    };

    float hip_x = 0.0f, hip_y = 0.0f, hip_confidence = 0.0f;
    float shoulder_x = 0.0f, shoulder_y = 0.0f, shoulder_confidence = 0.0f;
    midpoint(11, 12, hip_x, hip_y, hip_confidence);
    midpoint(5, 6, shoulder_x, shoulder_y, shoulder_confidence);

    if (hip_confidence < 0.1f) {
        float weight = 0.0f;
        for (int i = 0; i < kJoints; ++i) {
            if (confidence[i] < 0.1f) continue;
            hip_x += x[i] * confidence[i];
            hip_y += y[i] * confidence[i];
            weight += confidence[i];
        }
        if (weight <= 0.0f) return out;
        hip_x /= weight;
        hip_y /= weight;
    }

    float scale = 0.0f;
    if (shoulder_confidence >= 0.1f) {
        scale = std::hypot(shoulder_x - hip_x, shoulder_y - hip_y);
    }
    if (scale < 0.04f) {
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        bool found = false;
        for (int i = 0; i < kJoints; ++i) {
            if (confidence[i] < 0.1f) continue;
            min_x = std::min(min_x, x[i]);
            min_y = std::min(min_y, y[i]);
            max_x = std::max(max_x, x[i]);
            max_y = std::max(max_y, y[i]);
            found = true;
        }
        if (found) scale = std::max(max_x - min_x, max_y - min_y) * 0.35f;
    }
    scale = std::max(scale, 0.04f);

    for (int i = 0; i < kJoints; ++i) {
        if (confidence[i] >= 0.1f) {
            out[i * 2] = clipped((x[i] - hip_x) / scale, -4.0f, 4.0f);
            out[i * 2 + 1] = clipped((y[i] - hip_y) / scale, -4.0f, 4.0f);
        }
        out[kJoints * 2 + i] = confidence[i];
    }
    out[out.size() - 5] = observation.hip_y;
    out[out.size() - 4] = observation.torso_angle_deg / 90.0f;
    out[out.size() - 3] = std::min(observation.bbox_aspect_ratio, 4.0f) / 4.0f;
    out[out.size() - 2] = observation.person_score;
    out[out.size() - 1] = observation.valid ? 1.0f : 0.0f;
    return out;
}

void TemporalClassifier::reset() {
    frames_.clear();
    last_timestamp_sec_ = -1.0;
    last_evaluation_sec_ = -1.0;
    positive_run_ = 0;
    last_probability_ = 0.0f;
    last_positive_ = false;
}

TemporalPrediction TemporalClassifier::update(const TemporalFrame& frame, double timestamp_sec) {
    if (last_timestamp_sec_ >= 0.0 && timestamp_sec < last_timestamp_sec_) reset();
    last_timestamp_sec_ = timestamp_sec;
    TemporalFrame masked = frame;
    for (std::size_t i = 0; i < masked.size(); ++i) {
        masked[i] *= temporal_weights::kFrameMask[i];
    }
    frames_.push_back({timestamp_sec, masked});
    constexpr double kHistorySec = static_cast<double>(temporal_weights::kWindow - 1) / 15.0;
    const double cutoff = timestamp_sec - kHistorySec - 0.5;
    while (frames_.size() > 1 && frames_[1].timestamp_sec < cutoff) frames_.pop_front();

    TemporalPrediction prediction;
    constexpr double kEvaluationPeriodSec = static_cast<double>(kStride) / 15.0;
    prediction.evaluated = last_evaluation_sec_ < 0.0 ||
        timestamp_sec - last_evaluation_sec_ >= kEvaluationPeriodSec - 1e-6;
    if (prediction.evaluated) {
        last_evaluation_sec_ = timestamp_sec;
        last_probability_ = evaluate(timestamp_sec);
        positive_run_ = last_probability_ >= temporal_weights::kThreshold
            ? positive_run_ + 1 : 0;
        last_positive_ = positive_run_ >= temporal_weights::kConsecutive;
    }
    prediction.probability = last_probability_;
    prediction.positive = last_positive_;
    return prediction;
}

float TemporalClassifier::evaluate(double timestamp_sec) const {
    std::array<TemporalFrame, temporal_weights::kWindow> sequence{};
    std::size_t source = 0;
    for (int i = 0; i < temporal_weights::kWindow; ++i) {
        const double sample_time = timestamp_sec -
            static_cast<double>(temporal_weights::kWindow - 1 - i) / 15.0;
        while (source + 1 < frames_.size() &&
               frames_[source + 1].timestamp_sec <= sample_time + 1e-6) {
            ++source;
        }
        sequence[i] = frames_[source].values;
    }

    std::array<float, temporal_weights::kFeatureDim> feature{};
    constexpr int kBinFrames = temporal_weights::kWindow / kBins;
    for (int bin = 0; bin < kBins; ++bin) {
        for (int d = 0; d < temporal_weights::kFrameDim; ++d) {
            float sum = 0.0f;
            for (int f = 0; f < kBinFrames; ++f) sum += sequence[bin * kBinFrames + f][d];
            feature[bin * temporal_weights::kFrameDim + d] = sum / kBinFrames;
        }
    }
    const int std_offset = kBins * temporal_weights::kFrameDim;
    const int delta_offset = std_offset + temporal_weights::kFrameDim;
    const int span_offset = delta_offset + temporal_weights::kFrameDim;
    for (int d = 0; d < temporal_weights::kFrameDim; ++d) {
        float mean = 0.0f;
        float min_value = sequence[0][d];
        float max_value = sequence[0][d];
        for (const auto& item : sequence) {
            mean += item[d];
            min_value = std::min(min_value, item[d]);
            max_value = std::max(max_value, item[d]);
        }
        mean /= temporal_weights::kWindow;
        float variance = 0.0f;
        for (const auto& item : sequence) {
            const float diff = item[d] - mean;
            variance += diff * diff;
        }
        feature[std_offset + d] = std::sqrt(variance / temporal_weights::kWindow);
        feature[delta_offset + d] = sequence.back()[d] - sequence.front()[d];
        feature[span_offset + d] = max_value - min_value;
    }

    std::array<float, temporal_weights::kFeatureDim> normalized{};
    for (int d = 0; d < temporal_weights::kFeatureDim; ++d) {
        normalized[d] = (feature[d] - temporal_weights::kMean[d]) /
                        std::max(temporal_weights::kScale[d], 1e-12f);
    }
    std::array<float, temporal_weights::kHiddenDim> hidden{};
    for (int h = 0; h < temporal_weights::kHiddenDim; ++h) {
        float value = temporal_weights::kB1[h];
        for (int d = 0; d < temporal_weights::kFeatureDim; ++d) {
            value += normalized[d] * temporal_weights::kW1[d * temporal_weights::kHiddenDim + h];
        }
        hidden[h] = std::max(0.0f, value);
    }
    float logit = temporal_weights::kB2;
    for (int h = 0; h < temporal_weights::kHiddenDim; ++h) {
        logit += hidden[h] * temporal_weights::kW2[h];
    }
    if (logit >= 0.0f) return 1.0f / (1.0f + std::exp(-logit));
    const float e = std::exp(logit);
    return e / (1.0f + e);
}

}  // namespace fall
