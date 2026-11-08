#ifndef _ARUCO_INFERENCE_CONTROL_H_
#define _ARUCO_INFERENCE_CONTROL_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace aruco {

struct InferenceResult {
    uint32_t frame_id = 0;              ///< 0: no result produced yet
    uint64_t ts_ms = 0;                 ///< frame capture time, Unix epoch ms
    float inference_ms = 0;
    int count = 0;
    std::string markers_json = "[]";    ///< JSON array of detected markers
};

class InferenceControl {
public:
    static constexpr int kIdleFps = 1;
    static constexpr int kMaxWaitMs = 5000;

    /** Called on every state change with the new state and inference channel rate. */
    using OnChange = std::function<void(bool enabled, int fps)>;

    InferenceControl(int active_fps, bool enabled, OnChange on_change);

    bool enabled() const noexcept { return enabled_.load(std::memory_order_acquire); }
    int active_fps() const noexcept { return active_fps_; }
    int current_fps() const noexcept { return enabled() ? active_fps_ : kIdleFps; }

    /** Idempotent. On a change runs on_change; wakes waiters. Returns true if changed. */
    bool set_enabled(bool on);

    /** Frame thread: store a new result and wake waiters. */
    void publish(const InferenceResult& r);

    /**
     * Copy the latest result into out. With has_after, wait up to wait_ms for frame_id > after.
     * Returns true if timed out or condition not met.
     */
    bool latest(bool has_after, uint32_t after, int wait_ms, InferenceResult& out);

    /** Wake every waiter; later waits return immediately. */
    void shutdown();

private:
    const int active_fps_;
    OnChange on_change_;
    std::atomic<bool> enabled_;
    std::mutex mu_;
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
 *   POST /api/trigger[?wait_ms=<0..5000>]
 */
HttpReply handle_control_request(InferenceControl& ctl, const std::string& method,
                                 const std::string& target, const std::string& body,
                                 uint64_t now_ms);

}  // namespace aruco

#endif  // _ARUCO_INFERENCE_CONTROL_H_
