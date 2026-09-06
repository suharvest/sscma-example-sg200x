#ifndef _FACE_RECOGNITION_LIVENESS_TEMPORAL_H_
#define _FACE_RECOGNITION_LIVENESS_TEMPORAL_H_

#include <cstdint>
#include <deque>
#include <string>

#include "face_types.h"

// Per-track temporal aggregation for liveness v2.
//
// Deliberately free of every device dependency (no <sscma.h>, no OpenCV, no
// CVI): this is the only part of the stage whose behaviour can be pinned down
// without a camera, and tests/liveness_temporal_test.cpp compiles exactly these
// two files plus face_types.h on the host.
//
// Three independent kinds of evidence are combined:
//
//   texture  MiniFAS ensemble mean, smoothed by an EMA over sampled frames.
//   motion   residual of the five detector keypoints AFTER a least-squares
//            similarity (scale + rotation + translation) has been removed, so
//            a face that merely approaches the camera contributes nothing.
//   blink    EAR dips from sampled FaceMesh inferences.
//
// Two asymmetries are intentional and load-bearing:
//   * A blink proves liveness immediately.
//   * The ABSENCE of a blink never proves a spoof. A still, unblinking subject
//     must fall back to texture, not be rejected.

namespace face_recognition {

struct LivenessConfig {
    bool enabled = false;

    float texture_ema_alpha = 0.4f;
    int min_samples = 3;

    float motion_window_sec = 1.75f;
    float timeout_sec = 2.0f;
    float motion_noise_floor = 0.006f;
    float motion_high = 0.020f;
    float correlation_low = 0.15f;
    float correlation_high = 0.65f;

    int facemesh_interval = 3;   // 0 disables FaceMesh sampling
    float ear_threshold = 0.21f;
    int blink_min_samples = 1;
    int blink_max_samples = 3;

    float w_texture = 0.70f;
    float w_motion = 0.30f;
    float w_depth = 0.20f;
    float blink_bonus = 1.0f;
    float t_live = 0.55f;
    float t_spoof = 0.35f;

    bool depth_enabled = false;
    int capture_max_sec = 60;
};

// Result of the five-point motion analysis over the retained window.
struct MotionFeatures {
    bool valid = false;    // false until the window holds enough observations
    float residual = 0.f;  // RMS residual, in units of face short side
    // Mean off-diagonal entry of the temporal correlation matrix: how much
    // consecutive residual vectors agree in direction. ~0 for detector jitter,
    // ~+1 for a deformation that persists across frames.
    float correlation = 0.f;
    float score = 0.f;     // calibrated 0..1
};

// One retained keypoint observation.
struct KeypointSample {
    uint64_t ts_ms = 0;
    float face_px = 1.f;
    Landmarks5 kps{};
};

struct LivenessState {
    // texture
    bool has_texture = false;
    float texture_ema = 0.f;
    int texture_samples = 0;
    float last_v2 = 0.f;
    float last_v1se = 0.f;
    bool last_v1se_valid = false;

    // motion
    std::deque<KeypointSample> keypoints;
    MotionFeatures motion;

    // blink
    bool has_ear = false;
    float ear = 0.f;
    int closed_samples = 0;
    bool closure_overlong = false;
    bool blink_seen = false;
    bool blink_this_frame = false;

    // depth (Pro-only extension point; always absent on SG2002 today)
    bool has_depth = false;
    float depth_score = 0.f;

    // lifetime + last verdict (hysteresis input)
    bool started = false;
    uint64_t first_seen_ms = 0;
    std::string decision = "pending";
    bool has_score = false;
    float score = 0.f;
    std::string reason = "insufficient_samples";
};

// A motion verdict needs at least this many retained observations: three
// adjacent-pair residual frames, which is the minimum for a Pearson correlation
// between two points that is not trivially +/-1.
constexpr int kMinMotionSamples = 4;

// Folds one MiniFAS ensemble sample into the EMA and bumps texture_samples.
void updateTexture(LivenessState& state, float ensemble_mean, const LivenessConfig& cfg);

// Appends one keypoint observation, evicts everything older than the window and
// recomputes the motion features. The returned value is also cached in
// `state.motion`.
MotionFeatures updateMotion(LivenessState& state, uint64_t ts_ms, const Landmarks5& kps,
                            float face_px, const LivenessConfig& cfg);

// One sampled FaceMesh observation. `have_ear=false` means FaceMesh was skipped
// or failed on this frame and the closure counter must NOT advance -- a skipped
// sample is not an open eye.
// Returns true when this call completed a blink.
bool updateBlink(LivenessState& state, bool have_ear, float ear, const LivenessConfig& cfg);

// Combines the available evidence into a verdict, applying the three-sample
// gate, the blink override, the timeout texture-only path and T_live/T_spoof
// hysteresis. Updates `state.decision/score/reason` and returns them.
LivenessResult fuseLiveness(LivenessState& state, uint64_t now_ms, const LivenessConfig& cfg);

// Exposed for the offline calibration tooling and the tests: the piecewise
// linear mapping from (residual, correlation) to a 0..1 motion score.
float normalizedMotion(float residual, float correlation, const LivenessConfig& cfg);

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_LIVENESS_TEMPORAL_H_
