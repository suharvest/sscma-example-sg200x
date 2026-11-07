/**
 * @file event_loop.h
 * @brief Single-thread poll() loop for trigger file descriptors.
 *
 * Fixed capacity, no allocation after start(). Watches are registered before
 * start(); stop() wakes the thread through a self-pipe and joins it.
 * Pure POSIX: also built by the host tests.
 */
#ifndef AF_EVENT_LOOP_H_
#define AF_EVENT_LOOP_H_

#include <atomic>
#include <cstddef>
#include <thread>

namespace af {

class EventLoop {
public:
    /** Invoked on the loop thread when @p fd reports @p revents. */
    using Callback = void (*)(int fd, short revents, void* user);

    static constexpr size_t kMaxWatches = 8;

    EventLoop() = default;
    EventLoop(const EventLoop&) = delete;
    EventLoop& operator=(const EventLoop&) = delete;
    ~EventLoop() { stop(); }

    /** Register @p fd. Only valid before start(). Returns false when full, running or fd < 0. */
    bool watch(int fd, short events, Callback cb, void* user) noexcept;

    /** Spawn the loop thread. Returns true when running (also when there is nothing to watch). */
    bool start() noexcept;

    /** Wake and join the loop thread, then forget all watches. Idempotent. */
    void stop() noexcept;

    bool running() const noexcept { return running_.load(std::memory_order_acquire); }
    size_t watch_count() const noexcept { return count_; }

private:
    void loop() noexcept;

    struct Watch {
        int fd;
        short events;
        Callback cb;
        void* user;
    };

    Watch watches_[kMaxWatches] = {};
    size_t count_ = 0;
    int wake_[2] = {-1, -1};
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace af

#endif  // AF_EVENT_LOOP_H_
