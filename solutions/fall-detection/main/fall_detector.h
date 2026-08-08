#ifndef _FALL_DETECTION_DETECTOR_H_
#define _FALL_DETECTION_DETECTOR_H_

// Host-testable temporal fall detector.  This header deliberately has no
// reCamera/SSCMA includes: the embedded adapter turns YOLO pose keypoints into
// FallObservation and the same state machine can therefore be replayed on a
// laptop with synthetic trajectories.

#include <cstdint>

namespace fall {

enum class FallState {
    Normal,
    Suspected,
    Fallen,
    Recovering,
};

const char* fallStateName(FallState state);

// Features measured for one tracked subject. Coordinates are normalised to
// the inference frame. hip_y increases downwards; torso_angle is degrees away
// from vertical (0 upright, 90 horizontal). bbox_aspect_ratio is width / height.
struct FallObservation {
    bool valid = false;
    double timestamp_sec = 0.0;
    float hip_y = 0.0f;
    float torso_angle_deg = 0.0f;
    float bbox_aspect_ratio = 0.0f;
    float person_score = 0.0f;
    bool temporal_available = false;
    bool temporal_positive = false;
    float temporal_probability = 0.0f;
};

// All temporal and geometric thresholds are configurable through the app
// manifest.  No fall decision is made from a single feature or a single
// frame: evidence is scored and must remain present for confirmation_sec.
struct FallConfig {
    float hip_drop_speed_threshold = 0.25f;  // normalised y units / second
    float hip_drop_distance_threshold = 0.02f;  // from last non-horizontal pose
    float motion_window_sec = 0.75f;         // drop may precede horizontal pose
    float torso_angle_threshold_deg = 55.0f;
    float bbox_aspect_ratio_threshold = 1.25f;
    int min_suspected_features = 2;          // of speed, torso, aspect ratio
    float confirmation_sec = 0.80f;
    float suspected_timeout_sec = 1.50f;
    // Pose can disappear immediately after impact near the floor. A
    // motion-triggered candidate may confirm through this short gap.
    float occlusion_grace_sec = 0.75f;
    float recovery_torso_angle_deg = 35.0f;
    float recovery_aspect_ratio = 1.10f;
    float recovery_window_sec = 2.00f;
    float cooldown_sec = 3.00f;
};

struct FallDiagnostics {
    float hip_drop_speed = 0.0f;
    float hip_drop_distance = 0.0f;
    float torso_angle_deg = 0.0f;
    float bbox_aspect_ratio = 0.0f;
    int evidence_features = 0;
    float evidence_score = 0.0f;
    bool lying_posture = false;
    bool upright_posture = false;
    bool in_cooldown = false;
    bool temporal_positive = false;
    float temporal_probability = 0.0f;
    double suspected_for_sec = 0.0;
    double recovery_for_sec = 0.0;
};

struct FallOutput {
    FallState state = FallState::Normal;
    bool fall_detected = false;  // true while Fallen or Recovering
    bool fall_event = false;     // edge: true only on Normal/Suspected -> Fallen
    std::uint64_t event_id = 0;  // monotonically increasing, 0 before first event
    FallDiagnostics diagnostics;
};

class FallDetector {
public:
    explicit FallDetector(FallConfig config = {});

    void setConfig(const FallConfig& config);
    const FallConfig& config() const { return config_; }
    void reset();

    // Advance one frame. Invalid observations represent no usable pose; they
    // never create an event and allow a pending suspicion to expire.
    FallOutput update(const FallObservation& observation);

    FallState state() const { return state_; }
    std::uint64_t eventId() const { return event_id_; }
    const FallDiagnostics& diagnostics() const { return diagnostics_; }

private:
    bool isLying(const FallObservation& o) const;
    bool isUpright(const FallObservation& o) const;
    int featureCount(const FallObservation& o, float hip_speed) const;
    void updateDiagnostics(const FallObservation& o, float hip_speed);

    FallConfig config_;
    FallState state_ = FallState::Normal;
    std::uint64_t event_id_ = 0;
    bool initialized_ = false;
    bool have_previous_ = false;
    float previous_hip_y_ = 0.0f;
    double previous_timestamp_sec_ = 0.0;
    double last_fast_drop_sec_ = -1.0;
    float baseline_hip_y_ = 0.0f;
    bool have_baseline_hip_y_ = false;
    float max_drop_distance_ = 0.0f;
    double suspected_since_sec_ = -1.0;
    double last_strong_evidence_sec_ = -1.0;
    bool motion_triggered_ = false;
    double recovery_since_sec_ = -1.0;
    double cooldown_until_sec_ = -1.0;
    FallDiagnostics diagnostics_;
};

}  // namespace fall

#endif  // _FALL_DETECTION_DETECTOR_H_
