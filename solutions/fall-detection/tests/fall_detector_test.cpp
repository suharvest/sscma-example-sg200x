#include "../main/fall_detector.h"

#include <cassert>
#include <cmath>
#include <iostream>

using fall::FallConfig;
using fall::FallDetector;
using fall::FallObservation;
using fall::FallState;

static FallObservation frame(double t, float hip_y, float torso, float aspect) {
    FallObservation o;
    o.valid = true;
    o.timestamp_sec = t;
    o.hip_y = hip_y;
    o.torso_angle_deg = torso;
    o.bbox_aspect_ratio = aspect;
    o.person_score = 0.9f;
    return o;
}

int main() {
    FallConfig cfg;
    cfg.confirmation_sec = 0.6f;
    cfg.suspected_timeout_sec = 1.2f;
    cfg.occlusion_grace_sec = 0.8f;
    cfg.recovery_window_sec = 0.8f;
    cfg.cooldown_sec = 1.0f;
    FallDetector detector(cfg);

    // Upright -> rapid hip drop + torso rotation + wide box -> persistent
    // lying posture. One edge event is emitted only after confirmation.
    auto out = detector.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    assert(out.state == FallState::Normal && !out.fall_event);
    out = detector.update(frame(0.25, 0.72f, 65.0f, 1.45f));
    assert(out.state == FallState::Suspected && !out.fall_event);
    assert(out.diagnostics.evidence_features == 3);
    assert(out.diagnostics.hip_drop_speed > cfg.hip_drop_speed_threshold);
    out = detector.update(frame(0.55, 0.75f, 72.0f, 1.55f));
    assert(out.state == FallState::Suspected && !out.fall_event);
    out = detector.update(frame(1.25, 0.75f, 72.0f, 1.55f));
    assert(out.state == FallState::Fallen && out.fall_event);
    assert(out.event_id == 1 && out.fall_detected);

    // A recent strong lying pose can finish a motion-triggered confirmation
    // through a short post-impact occlusion.
    FallDetector occluded(cfg);
    occluded.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    out = occluded.update(frame(0.20, 0.70f, 68.0f, 1.45f));
    assert(out.state == FallState::Suspected);
    FallObservation missing;
    missing.timestamp_sec = 0.85;
    out = occluded.update(missing);
    assert(out.state == FallState::Fallen && out.fall_event && out.event_id == 1);

    // Bending alone is not enough: one feature must not create a fall.
    FallDetector bending(cfg);
    bending.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    for (int i = 1; i <= 8; ++i) {
        auto bent = bending.update(frame(i * 0.25, 0.52f, 65.0f, 0.80f));
        assert(!bent.fall_event);
        assert(bent.state == FallState::Normal);
    }

    // A slow, deliberate lie-down reaches the same final geometry but lacks
    // the rapid downward transition, so it must not alarm.
    FallDetector lying_down(cfg);
    lying_down.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    for (int i = 1; i <= 12; ++i) {
        const float p = static_cast<float>(i) / 12.0f;
        out = lying_down.update(frame(i * 0.25, 0.50f + 0.18f * p,
                                      10.0f + 65.0f * p, 0.65f + 0.90f * p));
        assert(!out.fall_event);
        assert(out.state == FallState::Normal);
    }

    // If the first frame is already lying down, expose the posture but do not
    // claim a newly observed fall.
    FallDetector initial(cfg);
    out = initial.update(frame(0.0, 0.75f, 72.0f, 1.55f));
    assert(out.state == FallState::Fallen);
    assert(!out.fall_event && out.event_id == 0);

    // Recovery needs an upright posture for the full window. A brief stand-up
    // followed by another lying frame returns to Fallen and does not create a
    // second event. Once recovered, cooldown suppresses an immediate re-alarm.
    out = detector.update(frame(1.35, 0.50f, 12.0f, 0.65f));
    assert(out.state == FallState::Recovering && out.fall_detected);
    out = detector.update(frame(1.70, 0.72f, 65.0f, 1.45f));
    assert(out.state == FallState::Fallen && !out.fall_event && out.event_id == 1);
    out = detector.update(frame(2.00, 0.50f, 12.0f, 0.65f));
    assert(out.state == FallState::Recovering);
    out = detector.update(frame(2.90, 0.50f, 12.0f, 0.65f));
    assert(out.state == FallState::Normal && !out.fall_detected);

    // A new, confirmed fall after cooldown gets a fresh event id.
    detector.update(frame(4.00, 0.50f, 12.0f, 0.65f));
    out = detector.update(frame(4.20, 0.72f, 65.0f, 1.45f));
    assert(out.state == FallState::Suspected && !out.fall_event);
    out = detector.update(frame(4.90, 0.75f, 72.0f, 1.55f));
    assert(out.state == FallState::Fallen && out.fall_event && out.event_id == 2);

    // When the learned temporal gate is available, geometry may expose a
    // Suspected state but cannot emit an alarm on its own. This is the path
    // that rejects deliberate lie-down/push-up transitions in v0.2.
    FallDetector gated(cfg);
    auto upright = frame(0.0, 0.50f, 10.0f, 0.65f);
    upright.temporal_available = true;
    gated.update(upright);
    auto impact = frame(0.20, 0.72f, 70.0f, 1.50f);
    impact.temporal_available = true;
    out = gated.update(impact);
    assert(out.state == FallState::Suspected && !out.fall_event);
    impact.timestamp_sec = 1.00;
    out = gated.update(impact);
    assert(out.state == FallState::Suspected && !out.fall_event);
    impact.timestamp_sec = 1.10;
    impact.temporal_positive = true;
    impact.temporal_probability = 0.91f;
    out = gated.update(impact);
    assert(out.state == FallState::Fallen && out.fall_event && out.event_id == 1);
    assert(std::fabs(out.diagnostics.temporal_probability - 0.91f) < 1e-5f);

    // Post-impact pose loss can still complete a learned transition, while an
    // ordinary disappearance (temporal_positive=false) never creates one.
    FallDetector lost_pose(cfg);
    upright.timestamp_sec = 0.0;
    lost_pose.update(upright);
    FallObservation no_pose;
    no_pose.timestamp_sec = 0.5;
    no_pose.temporal_available = true;
    out = lost_pose.update(no_pose);
    assert(!out.fall_event && out.state == FallState::Normal);
    no_pose.timestamp_sec = 0.7;
    no_pose.temporal_positive = true;
    no_pose.temporal_probability = 0.88f;
    out = lost_pose.update(no_pose);
    assert(out.fall_event && out.state == FallState::Fallen);

    std::cout << "fall_detector_test: all scenarios passed\n";
    return 0;
}
