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

constexpr float NOMINAL_STAND_HIP_PITCH_DEG = 26.237f;
constexpr float NOMINAL_STAND_SPINE_TILT_DEG = 14.82f;
constexpr float NOMINAL_STAND_THETA_L_DEG = -20.40f;
constexpr float NOMINAL_STAND_THETA_R_DEG = -55.59f;
constexpr float NOMINAL_STAND_DISPARITY_DEG = 35.19f;

constexpr int SETTLED_WINDOW_FRAMES = 30;
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
    std::vector<nlohmann::json> detected_defects;

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
};

class ActionStateMachine {
public:
    ActionStateMachine() = default;

    bool startAction(const std::string& action_id,
                     const std::string& run_id,
                     float timeout_s,
                     bool force,
                     const AppConfig& config_snapshot);

    void reset();

    // Check timeout and finalize report if expired. Returns true if state changed to REPORT_FROZEN.
    bool checkTimeout();

    // Add a newly processed frame with generation and timestamp validation
    bool addFrame(uint32_t frame_id,
                  uint64_t frame_time_ms,
                  uint64_t frame_generation,
                  bool robot_detected,
                  int count,
                  const FrameQAResult& frame_qa);

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
                                        bool timed_out = false);

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
