#ifndef _MICRODUCK_RESULT_PAYLOAD_H_
#define _MICRODUCK_RESULT_PAYLOAD_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "app_config.h"
#include "json.hpp"

#ifndef HOST_CLI
#include "pose.h"
#include "pose_detector.h"
#endif

namespace microduck {

// The stand-pose standard (nominal thigh angles, disparity, spine tilt) lives in AppConfig:
// compiled defaults, overridden by /etc/microduck-pose-qa.conf and by calibrate_stand.

constexpr int SETTLED_WINDOW_FRAMES = 30;
// face_led with params.expected: each frame is registered on its own, a short window suffices
// (the face check costs ~0.1-0.4 s per frame on the SG2002).
constexpr int FACE_LED_EXPECT_FRAMES = 8;
constexpr int MIN_VALID_FRAMES = 15;
constexpr int MIN_DETECTED_FRAMES = 5;
constexpr float OCCLUSION_RATIO_THRESHOLD = 0.20f;
constexpr float KNEE_ASYMMETRY_RATIO_THRESHOLD = 0.30f;
constexpr float SPINE_TILT_RATIO_THRESHOLD = 0.30f;
constexpr int MIN_VISIBLE_JOINTS_PER_FRAME = 10;
constexpr float KPT_CONF_EVAL_THRESHOLD = 0.35f;

struct KeypointRaw {
    int id = 0;
    std::string name;
    float x_norm = 0.0f;
    float y_norm = 0.0f;
    float conf = 0.0f;
};

struct QAMetrics {
    bool has_knee_diff = false;
    float knee_height_diff_norm = 0.0f;
    bool has_spline_offset = false;
    float spline_offset_deg = 0.0f;
    bool has_spine_tilt = false;
    float spine_tilt_deg = 0.0f;
};

struct FrameQAResult {
    std::string qa_status = "NO_ROBOT";  // "PASS", "KNEE_ASYMMETRY", "OCCLUDED", "NO_ROBOT", "SYS_ERROR"
    bool qa_passed = false;
    int visible_joints = 0;
    QAMetrics metrics;
    std::vector<KeypointRaw> kpts_raw_17;
};

struct ActionMetrics {
    float left_knee_height_norm = 0.0f;
    float right_knee_height_norm = 0.0f;
    float knee_height_diff_norm = 0.0f;
    float estimated_spline_offset_deg = 0.0f;
    bool has_spine_tilt = false;
    float spine_tilt_deg = 0.0f;
};

struct FaceLedMetrics {
    int lit_pixels = 0;
    float color_temperature_k = 0.0f;
    std::vector<float> mean_rgb = {0.0f, 0.0f, 0.0f};
    float dark_leakage_lux = 0.0f;
    float uniformity_ratio = 0.0f;
};

// ---- Single-LED face check (start_action face_led with params.expected) ----

// The 12 face LEDs as the firmware names them: face:<place>:<colour>.
extern const char* const kFaceLedNames[12];

// What one frame shows at one LED place after the layout is registered on the frame.
// colour: "" (no lit spot), "red", "green", "blue" or "white" (rgb place only).
struct FaceLedSlotObs {
    std::string place;
    std::string colour;
    float x = 0.0f;  // registered slot position, pixels
    float y = 0.0f;
    int area = 0;    // pixels of the matched lit spot
};

struct FaceLedSlotFrame {
    bool valid = false;          // layout registered on this frame
    std::string reason;          // RING_NOT_FOUND | RESIDUAL_TOO_HIGH when not valid
    bool has_ring = false;
    float ring_x = 0.0f, ring_y = 0.0f, ring_r = 0.0f;  // lens ring, pixels
    float scale_px = 0.0f;       // similarity scale: pixels per ring-radius unit
    float rotation_deg = 0.0f;
    float residual = -1.0f;      // RMS of the LS fit, ring radii
    std::vector<FaceLedSlotObs> slots;
};

struct FaceLedExpectation {
    bool active = false;
    std::string kind;        // "led" | "flashlight"
    std::string led;         // face:<place>:<colour> when kind == "led"
    std::string flashlight;  // red | green | blue | white when kind == "flashlight"
    std::string colour;      // expected colour class at the expected place
    bool has_rgb = false;
    int rgb[3] = {0, 0, 0};
};

// Parse params.expected: {"led": "face:gr2:red", "rgb": [255, 0, 0]} or {"flashlight": "red"}.
bool parseFaceLedExpectation(const nlohmann::json& j, FaceLedExpectation& out, std::string& err);

struct FaceLedLedResult {
    std::string led;
    std::string place;
    std::string colour;
    bool expected_on = false;
    bool observed_on = false;
    int on_frames = 0;
    bool ok = false;
};

struct FaceLedFrameResult {
    bool ok = false;
    FaceLedMetrics metrics;
    FaceLedSlotFrame slots;  // filled when the action carries an expectation
};

// ---- Stand-pose calibration (calibrate_stand) ----

struct StandCalibration {
    bool ok = false;
    std::string reason;
    int total_frames = 0;
    int valid_frames = 0;   // frames with both thighs measured
    int spine_frames = 0;   // frames with the spine measured
    int required_frames = 0;
    float theta_l_deg = 0.0f;
    float theta_r_deg = 0.0f;
    float disparity_deg = 0.0f;
    float spine_tilt_deg = 0.0f;
    float knee_diff_norm = 0.0f;  // signed median of (left knee y - right knee y)
};

struct ActionReport {
    std::string run_id;
    std::string action_id = "stand";
    std::string config_version = "20261105_v1";
    bool overall_passed = false;
    std::string status_code = "PASS";
    bool has_window_frames = false;
    uint32_t window_start_frame = 0;
    uint32_t window_end_frame = 0;
    int total_frames = 0;
    int valid_frames = 0;
    int detected_frames = 0;
    int occluded_frames = 0;
    ActionMetrics metrics;
    FaceLedMetrics optical_metrics;
    std::vector<nlohmann::json> detected_defects;
    // face_led with an expectation
    FaceLedExpectation led_expectation;
    std::vector<FaceLedLedResult> led_results;
    int led_valid_frames = 0;
    std::vector<nlohmann::json> led_registration;  // per frame: ring, scale, rotation, residual
    // calibrate_stand
    StandCalibration calibration;

    nlohmann::json toJson() const;
};

enum class ActionState {
    IDLE,
    COLLECTING,
    REPORT_FROZEN,
    BROADCAST_COMPLETED
};

struct CollectedFrame {
    uint32_t frame_id = 0;
    uint64_t timestamp_ms = 0;
    uint64_t generation = 0;
    bool robot_detected = false;
    int count = 0;
    FrameQAResult qa;
    FaceLedFrameResult optical;
};

class ActionStateMachine {
public:
    ActionStateMachine() = default;

    bool startAction(const std::string& action_id,
                     const std::string& run_id,
                     float timeout_s,
                     bool force,
                     const AppConfig& config_snapshot,
                     const FaceLedExpectation& expectation = FaceLedExpectation{});

    FaceLedExpectation getLedExpectation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return led_expectation_;
    }

    ActionReport getLastReport() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_report_;
    }

    void reset();
    bool stopAction();

    // Check timeout and finalize report if expired. Returns true if state changed to REPORT_FROZEN.
    bool checkTimeout();

    // Add a newly processed frame with generation and timestamp validation
    bool addFrame(uint32_t frame_id,
                  uint64_t frame_time_ms,
                  uint64_t frame_generation,
                  bool robot_detected,
                  int count,
                  const FrameQAResult& frame_qa,
                  const FaceLedFrameResult& optical = FaceLedFrameResult{});

    void setSysError(const std::string& reason, uint64_t gen);

    bool isBroadcastPending(uint64_t& out_gen) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (broadcast_pending_) {
            out_gen = broadcast_pending_generation_;
            return true;
        }
        return false;
    }

    void markBroadcastCompleted(uint64_t gen) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (broadcast_pending_ && broadcast_pending_generation_ == gen) {
            broadcast_pending_ = false;
            if (state_ == ActionState::REPORT_FROZEN) {
                state_ = ActionState::BROADCAST_COMPLETED;
            }
        }
    }

    int getCollectedFrameCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(window_frames_.size());
    }

    ActionState getState() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    std::string getStateString() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stateToString(state_);
    }

    std::string getCurrentRunId() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_run_id_;
    }

    std::string getCurrentActionId() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_action_id_;
    }

    uint64_t getActionGeneration() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return action_generation_;
    }

    AppConfig getActiveConfig() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_snapshot_;
    }

    bool hasReport() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return has_report_;
    }

    nlohmann::json getLastReportJson() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_report_) return nullptr;
        return last_report_.toJson();
    }

    static std::string stateToString(ActionState s) {
        switch (s) {
            case ActionState::IDLE: return "IDLE";
            case ActionState::COLLECTING: return "COLLECTING";
            case ActionState::REPORT_FROZEN: return "REPORT_FROZEN";
            case ActionState::BROADCAST_COMPLETED: return "BROADCAST_COMPLETED";
            default: return "UNKNOWN";
        }
    }

    // Static helper to evaluate an aggregated window of frames (used also by CLI)
    static ActionReport aggregateWindow(const std::string& action_id,
                                        const std::string& run_id,
                                        const AppConfig& config,
                                        const std::vector<CollectedFrame>& frames,
                                        bool timed_out = false,
                                        const FaceLedExpectation& expectation = FaceLedExpectation{});

    // Median thigh angles / disparity / spine tilt over a window of golden-sample stand frames.
    // Rejected (ok == false) with fewer than min_valid measured frames.
    static StandCalibration computeStandCalibration(const AppConfig& config,
                                                    const std::vector<CollectedFrame>& frames,
                                                    int min_valid = MIN_VALID_FRAMES);

    // Per-LED verdicts from per-frame slot observations (majority over registered frames).
    static std::vector<FaceLedLedResult> judgeFaceLeds(const FaceLedExpectation& expectation,
                                                       const std::vector<CollectedFrame>& frames,
                                                       int* valid_frames = nullptr);

private:
    void finalizeReportLocked(bool timed_out);

    mutable std::mutex mutex_;
    ActionState state_ = ActionState::IDLE;
    std::string current_action_id_ = "stand";
    std::string current_run_id_;
    float timeout_s_ = 3.0f;
    uint64_t start_time_ms_ = 0;
    uint64_t action_generation_ = 0;
    AppConfig config_snapshot_;
    FaceLedExpectation led_expectation_;

    std::vector<CollectedFrame> window_frames_;
    bool has_report_ = false;
    bool broadcast_pending_ = false;
    uint64_t broadcast_pending_generation_ = 0;
    ActionReport last_report_;
};

struct PayloadContext {
    uint64_t timestamp_ms = 0;
    uint32_t frame_id = 0;
    float inference_time_ms = 0.0f;
    float fps = 30.0f;
    bool robot_detected = false;
    int duck_count = 0;

    int infer_w = 384;
    int infer_h = 384;
    int stream_w = 1280;
    int stream_h = 720;

    std::string action_id = "stand";
    FaceLedMetrics optical;

    // Snapshot config
    AppConfig config;
};

// Evaluate frame QA from 17 raw keypoints in normalized coordinates [0, 1]
FrameQAResult evaluateFrameFromNormKpts(const std::vector<KeypointRaw>& kpts_raw,
                                       bool robot_detected,
                                       int duck_count,
                                       const AppConfig& config);

#ifndef HOST_CLI
// Evaluate frame QA from detector Pose with camera coordinate transform
FrameQAResult evaluateQA(const PayloadContext& ctx, const Pose& pose);

// Build complete MQTT result JSON
std::string buildResultJson(const PayloadContext& ctx,
                            const std::vector<Subject>& subjects,
                            const FrameQAResult& qa,
                            const ActionStateMachine& sm);

// Build debug extra JSON for WebSocket envelope
std::string buildDebugExtraJson(const PayloadContext& ctx,
                                const std::vector<Subject>& subjects,
                                const FrameQAResult& qa,
                                const ActionStateMachine& sm);
#endif

}  // namespace microduck

#endif  // _MICRODUCK_RESULT_PAYLOAD_H_
