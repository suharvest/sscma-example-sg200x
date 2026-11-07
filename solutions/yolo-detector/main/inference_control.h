#ifndef _YOLO_INFERENCE_CONTROL_H_
#define _YOLO_INFERENCE_CONTROL_H_

// On-demand inference state shared by the frame thread and the HTTP control
// server: the enabled switch, the latest result and the long-poll wait.
// No SDK dependencies, so tests/ builds it on the host.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace yolo {

struct InferenceResult {
    uint32_t frame_id = 0;        ///< 0: no result produced yet
    uint64_t ts_ms = 0;           ///< frame capture time, Unix epoch ms
    float inference_ms = 0;
    std::string detections_json = "[]";  ///< JSON array, same objects as the MQTT payload
};

class InferenceControl {
public:
    static constexpr int kIdleFps = 1;
    static constexpr int kMaxWaitMs = 5000;

    /** Called on every state change with the new state and inference channel rate (must not block). */
    using OnChange = std::function<void(bool enabled, int fps)>;

    InferenceControl(int active_fps, bool enabled, OnChange on_change);

    bool enabled() const noexcept { return enabled_.load(std::memory_order_acquire); }
    int active_fps() const noexcept { return active_fps_; }
    /** Inference channel rate for the current state. */
    int current_fps() const noexcept { return enabled() ? active_fps_ : kIdleFps; }

    /** Idempotent. On a change runs on_change (camera rate); always wakes long-poll waiters. Returns true if the state changed. */
    bool set_enabled(bool on);

    /** Frame thread: store a new result and wake waiters. */
    void publish(const InferenceResult& r);

    /**
     * Copy the latest result into @p out. With @p has_after, wait up to
     * @p wait_ms (clamped to 0..kMaxWaitMs) for frame_id > @p after; returns
     * immediately while inference is disabled or after shutdown().
     * @return true when the caller's condition is not met (no result at all, or
     *         has_after and the latest frame_id <= after).
     */
    bool latest(bool has_after, uint32_t after, int wait_ms, InferenceResult& out);

    /** Wake every waiter; later waits return immediately. */
    void shutdown();

private:
    const int active_fps_;
    OnChange on_change_;
    std::atomic<bool> enabled_;
    std::mutex mu_;           ///< guards latest_ and stopping_; serializes set_enabled
    std::condition_variable cv_;
    InferenceResult latest_;
    bool stopping_ = false;
};

struct HttpReply {
    int status = 200;
    std::string body;
};

/**
 * Route one request:
 *   GET  /api/inference
 *   POST /api/inference          {"enabled": bool}
 *   GET  /api/results/latest[?after=<frame_id>&wait_ms=<0..5000>]
 * @param target request target including the query string.
 * @param now_ms device Unix time in ms, for age_ms.
 */
HttpReply handle_control_request(InferenceControl& ctl, const std::string& method,
                                 const std::string& target, const std::string& body,
                                 uint64_t now_ms);

}  // namespace yolo

#endif  // _YOLO_INFERENCE_CONTROL_H_
