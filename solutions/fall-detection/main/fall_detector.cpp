#include "fall_detector.h"

#include <algorithm>
#include <cmath>

namespace fall {

const char* fallStateName(FallState state) {
    switch (state) {
        case FallState::Normal: return "normal";
        case FallState::Suspected: return "suspected";
        case FallState::Fallen: return "fallen";
        case FallState::Recovering: return "recovering";
    }
    return "normal";
}

FallDetector::FallDetector(FallConfig config) : config_(config) {
    setConfig(config);
}

void FallDetector::setConfig(const FallConfig& config) {
    config_ = config;
    config_.hip_drop_speed_threshold = std::max(0.0f, config_.hip_drop_speed_threshold);
    config_.hip_drop_distance_threshold = std::max(0.0f, config_.hip_drop_distance_threshold);
    config_.motion_window_sec = std::max(0.0f, config_.motion_window_sec);
    config_.torso_angle_threshold_deg = std::clamp(config_.torso_angle_threshold_deg, 1.0f, 89.0f);
    config_.bbox_aspect_ratio_threshold = std::max(1.0f, config_.bbox_aspect_ratio_threshold);
    config_.min_suspected_features = std::clamp(config_.min_suspected_features, 1, 3);
    config_.confirmation_sec = std::max(0.0f, config_.confirmation_sec);
    config_.suspected_timeout_sec = std::max(config_.confirmation_sec, config_.suspected_timeout_sec);
    config_.occlusion_grace_sec = std::max(0.0f, config_.occlusion_grace_sec);
    config_.recovery_torso_angle_deg = std::clamp(config_.recovery_torso_angle_deg, 1.0f, 89.0f);
    config_.recovery_aspect_ratio = std::max(1.0f, config_.recovery_aspect_ratio);
    config_.recovery_window_sec = std::max(0.0f, config_.recovery_window_sec);
    config_.cooldown_sec = std::max(0.0f, config_.cooldown_sec);
}

void FallDetector::reset() {
    state_ = FallState::Normal;
    event_id_ = 0;
    initialized_ = false;
    have_previous_ = false;
    previous_hip_y_ = 0.0f;
    previous_timestamp_sec_ = 0.0;
    last_fast_drop_sec_ = -1.0;
    baseline_hip_y_ = 0.0f;
    have_baseline_hip_y_ = false;
    max_drop_distance_ = 0.0f;
    suspected_since_sec_ = -1.0;
    last_strong_evidence_sec_ = -1.0;
    motion_triggered_ = false;
    recovery_since_sec_ = -1.0;
    cooldown_until_sec_ = -1.0;
    diagnostics_ = FallDiagnostics{};
}

bool FallDetector::isLying(const FallObservation& o) const {
    return o.torso_angle_deg >= config_.torso_angle_threshold_deg &&
           o.bbox_aspect_ratio >= config_.bbox_aspect_ratio_threshold;
}

bool FallDetector::isUpright(const FallObservation& o) const {
    return o.torso_angle_deg <= config_.recovery_torso_angle_deg &&
           o.bbox_aspect_ratio <= config_.recovery_aspect_ratio;
}

int FallDetector::featureCount(const FallObservation& o, float hip_speed) const {
    int count = 0;
    if (hip_speed >= config_.hip_drop_speed_threshold) ++count;
    if (o.torso_angle_deg >= config_.torso_angle_threshold_deg) ++count;
    if (o.bbox_aspect_ratio >= config_.bbox_aspect_ratio_threshold) ++count;
    return count;
}

void FallDetector::updateDiagnostics(const FallObservation& o, float hip_speed) {
    diagnostics_.hip_drop_speed = hip_speed;
    diagnostics_.hip_drop_distance = max_drop_distance_;
    diagnostics_.torso_angle_deg = o.torso_angle_deg;
    diagnostics_.bbox_aspect_ratio = o.bbox_aspect_ratio;
    diagnostics_.evidence_features = featureCount(o, hip_speed);
    diagnostics_.evidence_score = static_cast<float>(diagnostics_.evidence_features) / 3.0f;
    diagnostics_.lying_posture = isLying(o);
    diagnostics_.upright_posture = isUpright(o);
    diagnostics_.in_cooldown = o.timestamp_sec < cooldown_until_sec_;
    diagnostics_.temporal_positive = o.temporal_positive;
    diagnostics_.temporal_probability = o.temporal_probability;
    diagnostics_.suspected_for_sec = suspected_since_sec_ >= 0.0
        ? std::max(0.0, o.timestamp_sec - suspected_since_sec_) : 0.0;
    diagnostics_.recovery_for_sec = recovery_since_sec_ >= 0.0
        ? std::max(0.0, o.timestamp_sec - recovery_since_sec_) : 0.0;
}

FallOutput FallDetector::update(const FallObservation& o) {
    FallOutput out;
    out.fall_event = false;

    float hip_speed = 0.0f;
    if (o.valid && have_previous_) {
        const double dt = o.timestamp_sec - previous_timestamp_sec_;
        if (dt > 1e-4 && dt < 10.0) {
            hip_speed = (o.hip_y - previous_hip_y_) / static_cast<float>(dt);
        }
    }

    if (!o.valid) {
        // Missing/invalid pose may retain an already-established state, but it
        // must never originate a new fall event. Live SG2002 testing showed
        // that sparse keypoints and short-lived duplicate tracks can leave a
        // high temporal probability behind after the person disappears. A
        // valid current observation is therefore required at the transition
        // into Fallen; occlusion is continuity, not confirmation evidence.
        diagnostics_.hip_drop_speed = 0.0f;
        diagnostics_.hip_drop_distance = max_drop_distance_;
        diagnostics_.torso_angle_deg = 0.0f;
        diagnostics_.bbox_aspect_ratio = 0.0f;
        diagnostics_.evidence_features = 0;
        diagnostics_.evidence_score = 0.0f;
        diagnostics_.lying_posture = false;
        diagnostics_.upright_posture = false;
        diagnostics_.temporal_positive = o.temporal_positive;
        diagnostics_.temporal_probability = o.temporal_probability;
        if (state_ == FallState::Suspected && suspected_since_sec_ >= 0.0) {
            const double suspected_for = o.timestamp_sec - suspected_since_sec_;
            if (suspected_for > config_.suspected_timeout_sec) {
                state_ = FallState::Normal;
                suspected_since_sec_ = -1.0;
                last_strong_evidence_sec_ = -1.0;
                motion_triggered_ = false;
                last_fast_drop_sec_ = -1.0;
                max_drop_distance_ = 0.0f;
            }
        }
        diagnostics_.in_cooldown = o.timestamp_sec < cooldown_until_sec_;
        diagnostics_.suspected_for_sec = suspected_since_sec_ >= 0.0
            ? std::max(0.0, o.timestamp_sec - suspected_since_sec_) : 0.0;
        diagnostics_.recovery_for_sec = recovery_since_sec_ >= 0.0
            ? std::max(0.0, o.timestamp_sec - recovery_since_sec_) : 0.0;
    } else {
        updateDiagnostics(o, hip_speed);
        const bool horizontal_cue =
            o.torso_angle_deg >= config_.torso_angle_threshold_deg ||
            o.bbox_aspect_ratio >= config_.bbox_aspect_ratio_threshold;
        if (state_ == FallState::Normal && !horizontal_cue) {
            baseline_hip_y_ = o.hip_y;
            have_baseline_hip_y_ = true;
        }
        if (state_ == FallState::Suspected && have_baseline_hip_y_) {
            max_drop_distance_ = std::max(max_drop_distance_, o.hip_y - baseline_hip_y_);
        }
        if (hip_speed >= config_.hip_drop_speed_threshold) {
            last_fast_drop_sec_ = o.timestamp_sec;
        }

        if (!initialized_) {
            // A camera may start while a person is already on the floor. With
            // no preceding history, posture alone is not a fall.
            initialized_ = true;
        } else {
            const int evidence = diagnostics_.evidence_features;
            const bool lying = diagnostics_.lying_posture;
            const bool enough_evidence = evidence >= config_.min_suspected_features;
            const bool cooldown = o.timestamp_sec < cooldown_until_sec_;

            switch (state_) {
                case FallState::Normal:
                    // Horizontal posture alone is not a fall: sleeping,
                    // push-ups, and a deliberate lie-down can look identical.
                    // Arm only on rapid descent plus a horizontal-posture cue.
                    if (!cooldown && last_fast_drop_sec_ >= 0.0 &&
                        o.timestamp_sec - last_fast_drop_sec_ <= config_.motion_window_sec &&
                        (o.torso_angle_deg >= config_.torso_angle_threshold_deg ||
                         o.bbox_aspect_ratio >= config_.bbox_aspect_ratio_threshold)) {
                        state_ = FallState::Suspected;
                        suspected_since_sec_ = last_fast_drop_sec_;
                        last_strong_evidence_sec_ = lying ? o.timestamp_sec : -1.0;
                        motion_triggered_ = true;
                        max_drop_distance_ = have_baseline_hip_y_
                            ? std::max(0.0f, o.hip_y - baseline_hip_y_) : 0.0f;
                    }
                    break;

                case FallState::Suspected:
                    if (!cooldown && o.temporal_available && o.temporal_positive) {
                        state_ = FallState::Fallen;
                        recovery_since_sec_ = -1.0;
                        cooldown_until_sec_ = o.timestamp_sec + config_.cooldown_sec;
                        ++event_id_;
                        out.fall_event = true;
                        break;
                    }
                    if (lying && enough_evidence) {
                        last_strong_evidence_sec_ = o.timestamp_sec;
                    }
                    if (!config_.temporal_confirmation_required && motion_triggered_ && lying && enough_evidence &&
                        max_drop_distance_ >= config_.hip_drop_distance_threshold &&
                        o.timestamp_sec - suspected_since_sec_ >= config_.confirmation_sec) {
                        state_ = FallState::Fallen;
                        recovery_since_sec_ = -1.0;
                        cooldown_until_sec_ = o.timestamp_sec + config_.cooldown_sec;
                        ++event_id_;
                        out.fall_event = true;
                    } else if (diagnostics_.upright_posture ||
                               o.timestamp_sec - suspected_since_sec_ > config_.suspected_timeout_sec) {
                        state_ = FallState::Normal;
                        suspected_since_sec_ = -1.0;
                        last_strong_evidence_sec_ = -1.0;
                        motion_triggered_ = false;
                        last_fast_drop_sec_ = -1.0;
                        max_drop_distance_ = 0.0f;
                    }
                    break;

                case FallState::Fallen:
                    if (diagnostics_.upright_posture) {
                        if (config_.recovery_window_sec <= 0.0f) {
                            state_ = FallState::Normal;
                            recovery_since_sec_ = -1.0;
                            suspected_since_sec_ = -1.0;
                            last_strong_evidence_sec_ = -1.0;
                            motion_triggered_ = false;
                            last_fast_drop_sec_ = -1.0;
                            max_drop_distance_ = 0.0f;
                        } else {
                            state_ = FallState::Recovering;
                            recovery_since_sec_ = o.timestamp_sec;
                        }
                    }
                    break;

                case FallState::Recovering:
                    if (!diagnostics_.upright_posture) {
                        state_ = FallState::Fallen;
                        recovery_since_sec_ = -1.0;
                    } else if (o.timestamp_sec - recovery_since_sec_ >= config_.recovery_window_sec) {
                        state_ = FallState::Normal;
                        recovery_since_sec_ = -1.0;
                        suspected_since_sec_ = -1.0;
                        last_strong_evidence_sec_ = -1.0;
                        motion_triggered_ = false;
                        last_fast_drop_sec_ = -1.0;
                        max_drop_distance_ = 0.0f;
                    }
                    break;
            }
        }

        // Hip velocity is a temporal feature. Keep the last usable reading,
        // even when the state machine remains in Fallen or Recovering.
        previous_hip_y_ = o.hip_y;
        previous_timestamp_sec_ = o.timestamp_sec;
        have_previous_ = true;
    }

    // Populate timers after a transition so the result describes this frame.
    diagnostics_.in_cooldown = o.timestamp_sec < cooldown_until_sec_;
    diagnostics_.suspected_for_sec = suspected_since_sec_ >= 0.0
        ? std::max(0.0, o.timestamp_sec - suspected_since_sec_) : 0.0;
    diagnostics_.recovery_for_sec = recovery_since_sec_ >= 0.0
        ? std::max(0.0, o.timestamp_sec - recovery_since_sec_) : 0.0;
    diagnostics_.hip_drop_distance = max_drop_distance_;

    out.state = state_;
    out.fall_detected = state_ == FallState::Fallen || state_ == FallState::Recovering;
    out.event_id = event_id_;
    out.diagnostics = diagnostics_;
    return out;
}

}  // namespace fall
