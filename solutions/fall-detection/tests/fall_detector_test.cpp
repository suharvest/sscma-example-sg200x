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
    cfg.temporal_confirmation_required = false;
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

    // Occlusion retains the candidate but cannot originate an event before
    // confirmation has matured. A valid current observation is required for
    // the final transition. (The gap is kept inside confirmation_sec; no mode
    // confirms on an invalid frame.)
    FallDetector occluded(cfg);
    occluded.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    out = occluded.update(frame(0.20, 0.70f, 68.0f, 1.45f));
    assert(out.state == FallState::Suspected);
    FallObservation missing;
    missing.timestamp_sec = 0.55;
    out = occluded.update(missing);
    assert(out.state == FallState::Suspected && !out.fall_event && out.event_id == 0);
    auto visible_again = frame(0.95, 0.74f, 70.0f, 1.50f);
    visible_again.temporal_available = true;
    visible_again.temporal_positive = true;
    out = occluded.update(visible_again);
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
    assert(out.state == FallState::Normal);
    assert(!out.fall_detected && !out.fall_event && out.event_id == 0);

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
    FallConfig learned_cfg = cfg;
    learned_cfg.temporal_confirmation_required = true;
    FallDetector gated(learned_cfg);
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

    // Default deployment policy requires the learned temporal gate; geometry
    // alone remains suspected. Legacy geometry confirmation is opt-in above.
    FallConfig strict_cfg = cfg;
    strict_cfg.temporal_confirmation_required = true;
    FallDetector strict(strict_cfg);
    strict.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    out = strict.update(frame(0.25, 0.72f, 65.0f, 1.45f));
    assert(out.state == FallState::Suspected && !out.fall_event);
    out = strict.update(frame(1.25, 0.75f, 72.0f, 1.55f));
    assert(out.state == FallState::Suspected && !out.fall_event);

    // Pose loss never completes a learned transition, even if the temporal
    // window remains positive from previous frames.
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
    assert(!out.fall_event && out.state == FallState::Normal);

    // ---- Ported confirmation-gate scenarios (from the reCamera Pro fix) ----

    // Shared learned-gate configuration for the scenarios below.
    FallConfig gate_cfg;
    gate_cfg.confirmation_sec = 0.6f;
    gate_cfg.suspected_timeout_sec = 1.2f;
    gate_cfg.occlusion_grace_sec = 0.8f;
    gate_cfg.recovery_window_sec = 0.8f;
    gate_cfg.cooldown_sec = 3.0f;
    gate_cfg.temporal_confirmation_required = true;
    auto learned = frame(0.0, 0.0f, 0.0f, 0.0f);
    learned.temporal_available = true;
    learned.temporal_positive = true;
    learned.temporal_probability = 0.9f;

    // 1. Upright + temporal positive never leaves Normal.
    {
        FallDetector d(gate_cfg);
        std::uint64_t events = 0;
        for (int i = 0; i <= 24; ++i) {
            auto f = learned;
            f.timestamp_sec = i * 0.25;
            f.hip_y = 0.50f; f.torso_angle_deg = 10.0f; f.bbox_aspect_ratio = 0.65f;
            const auto r = d.update(f);
            events += r.fall_event ? 1u : 0u;
            assert(r.state == FallState::Normal);
        }
        assert(events == 0 && d.eventId() == 0);
    }

    // 2. Sit-down: fast hip drop and a wide box (horizontal cue) but the
    // torso is not lying. Temporal positive must not confirm.
    {
        FallDetector d(gate_cfg);
        std::uint64_t events = 0;
        auto r = d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        for (int i = 1; i <= 24; ++i) {
            auto f = learned;
            f.timestamp_sec = i * 0.25;
            f.hip_y = 0.68f; f.torso_angle_deg = 40.0f; f.bbox_aspect_ratio = 1.45f;
            r = d.update(f);
            events += r.fall_event ? 1u : 0u;
            assert(!r.fall_detected);
        }
        assert(events == 0 && d.eventId() == 0);
        assert(r.state == FallState::Normal);  // expired after the late latch
    }

    // 3. Real fall: fast drop + horizontal + lying + temporal positive ->
    // exactly one event, no re-alarm while still on the floor.
    {
        FallDetector d(gate_cfg);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        auto r = d.update(frame(0.25, 0.72f, 65.0f, 1.45f));
        assert(r.state == FallState::Suspected && !r.fall_event);
        auto lying = learned;
        lying.hip_y = 0.75f; lying.torso_angle_deg = 72.0f; lying.bbox_aspect_ratio = 1.55f;
        lying.timestamp_sec = 0.55;
        r = d.update(lying);
        assert(r.state == FallState::Fallen && r.fall_event && r.event_id == 1);
        for (int i = 1; i <= 12; ++i) {
            lying.timestamp_sec = 0.55 + i * 0.25;
            r = d.update(lying);
            assert(r.state == FallState::Fallen && !r.fall_event && r.event_id == 1);
        }
    }

    // 4. Occluded fall: the drop happens behind occlusion, so hip speed is
    // below threshold; displacement arming against the upright baseline plus
    // lying + temporal positive confirms.
    {
        FallDetector d(gate_cfg);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        FallObservation gap;
        for (double t = 0.30; t <= 0.70; t += 0.20) {
            gap.timestamp_sec = t;
            assert(!d.update(gap).fall_event);
        }
        auto on_floor = learned;
        on_floor.timestamp_sec = 0.90;
        on_floor.hip_y = 0.56f;  // speed across the gap ~0.07 < 0.25
        on_floor.torso_angle_deg = 70.0f; on_floor.bbox_aspect_ratio = 1.50f;
        auto r = d.update(on_floor);
        assert(r.state == FallState::Suspected && !r.fall_event);
        on_floor.timestamp_sec = 1.10;
        r = d.update(on_floor);
        assert(r.state == FallState::Fallen && r.fall_event && r.event_id == 1);
    }

    // 5a. Late temporal positive after suspected_timeout while still lying.
    {
        FallDetector d(gate_cfg);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        d.update(frame(0.25, 0.72f, 65.0f, 1.45f));
        auto lying = frame(0.0, 0.75f, 72.0f, 1.55f);
        for (double t = 0.55; t <= 2.00; t += 0.25) {
            lying.timestamp_sec = t;
            assert(d.update(lying).state == FallState::Suspected);
        }
        auto late = learned;
        late.timestamp_sec = 3.00;
        late.hip_y = 0.75f; late.torso_angle_deg = 72.0f; late.bbox_aspect_ratio = 1.55f;
        const auto r = d.update(late);
        assert(r.state == FallState::Fallen && r.fall_event && r.event_id == 1);
        // The latch is bounded: with no positive by timeout + late window the
        // candidate returns to Normal.
        FallDetector d2(gate_cfg);
        d2.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        d2.update(frame(0.25, 0.72f, 65.0f, 1.45f));
        for (double t = 0.55; t <= 5.00; t += 0.25) {
            lying.timestamp_sec = t;
            const auto q = d2.update(lying);
            if (t > 4.65) assert(q.state == FallState::Normal);
        }
        assert(d2.eventId() == 0);
    }

    // 5b. Stands up before the late positive arrives -> no event.
    {
        FallDetector d(gate_cfg);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        d.update(frame(0.25, 0.72f, 65.0f, 1.45f));
        auto lying = frame(1.00, 0.75f, 72.0f, 1.55f);
        assert(d.update(lying).state == FallState::Suspected);
        auto up = frame(1.50, 0.50f, 12.0f, 0.65f);
        assert(d.update(up).state == FallState::Normal);
        auto late = learned;
        late.timestamp_sec = 3.00;
        late.hip_y = 0.75f; late.torso_angle_deg = 72.0f; late.bbox_aspect_ratio = 1.55f;
        const auto r = d.update(late);
        assert(r.state == FallState::Normal && !r.fall_event && r.event_id == 0);
    }

    // 6. min_suspected_features = 3: a victim lying still (hip speed ~0)
    // still confirms because the arming drop latched the motion feature.
    {
        FallConfig strict3 = gate_cfg;
        strict3.min_suspected_features = 3;
        FallDetector d(strict3);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        d.update(frame(0.25, 0.72f, 65.0f, 1.45f));
        auto still = learned;
        still.timestamp_sec = 0.55;
        still.hip_y = 0.75f; still.torso_angle_deg = 72.0f; still.bbox_aspect_ratio = 1.55f;
        auto r = d.update(still);
        assert(r.state == FallState::Fallen && r.fall_event && r.event_id == 1);
        assert(r.diagnostics.confirmation_features == 3);
        still.timestamp_sec = 0.80;
        r = d.update(still);
        assert(r.diagnostics.confirmation_features == 2);  // posture only once Fallen
    }

    // 7. Temporal positive while Suspected but not lying: no event until the
    // pose actually becomes lying.
    {
        FallDetector d(gate_cfg);
        d.update(frame(0.0, 0.50f, 10.0f, 0.65f));
        auto sitting = learned;
        sitting.timestamp_sec = 0.25;
        sitting.hip_y = 0.70f; sitting.torso_angle_deg = 40.0f; sitting.bbox_aspect_ratio = 1.45f;
        auto r = d.update(sitting);
        assert(r.state == FallState::Suspected && !r.fall_event);
        sitting.timestamp_sec = 0.50;
        r = d.update(sitting);
        assert(r.state == FallState::Suspected && !r.fall_event && r.event_id == 0);
        auto now_lying = sitting;
        now_lying.timestamp_sec = 0.75;
        now_lying.hip_y = 0.74f; now_lying.torso_angle_deg = 70.0f; now_lying.bbox_aspect_ratio = 1.50f;
        r = d.update(now_lying);
        assert(r.state == FallState::Fallen && r.fall_event && r.event_id == 1);
    }

    // 8. Legacy mode: an occluded (invalid) frame never completes a fall,
    // even with a recent lying frame and matured confirmation evidence.
    // Occlusion is continuity, not confirmation evidence.
    {
        FallConfig legacy = gate_cfg;
        legacy.temporal_confirmation_required = false;
        legacy.confirmation_sec = 0.8f;
        legacy.suspected_timeout_sec = 1.5f;
        FallDetector d(legacy);
        const double dt = 1.0 / 15.0;  // 15 fps
        for (double t = 0.0; t < 1.0; t += dt) {
            d.update(frame(t, 0.50f, 10.0f, 0.65f));
        }
        // Fall over 0.4 s (6 frames) to torso 80 / aspect 2.45.
        for (int i = 1; i <= 6; ++i) {
            const double t = 1.0 + i * 0.4 / 6.0;
            const float p = static_cast<float>(i) / 6.0f;
            d.update(frame(t, 0.50f + 0.22f * p, 10.0f + 70.0f * p,
                           0.65f + 1.80f * p));
        }
        // Invalid frames from t=1.5 for 2 s: no event on any of them.
        std::uint64_t events = 0;
        for (int i = 0; i < 30; ++i) {
            FallObservation gap;
            gap.timestamp_sec = 1.5 + i * dt;
            const auto r = d.update(gap);
            events += r.fall_event ? 1u : 0u;
            assert(!r.fall_detected);
        }
        assert(events == 0 && d.eventId() == 0);
    }

    std::cout << "fall_detector_test: all scenarios passed\n";
    return 0;
}
