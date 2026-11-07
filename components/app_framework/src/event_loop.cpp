/**
 * @file event_loop.cpp
 * @brief poll() loop with a self-pipe for shutdown.
 */
#include "event_loop.h"

#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

namespace af {

bool EventLoop::watch(int fd, short events, Callback cb, void* user) noexcept {
    if (fd < 0 || cb == nullptr || count_ >= kMaxWatches || running()) {
        return false;
    }
    watches_[count_++] = Watch{fd, events, cb, user};
    return true;
}

bool EventLoop::start() noexcept {
    if (running()) {
        return true;
    }
    if (pipe(wake_) != 0) {
        wake_[0] = wake_[1] = -1;
        return false;
    }
    for (int fd : wake_) {
        fcntl(fd, F_SETFD, FD_CLOEXEC);
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    }
    running_.store(true, std::memory_order_release);
    try {
        thread_ = std::thread(&EventLoop::loop, this);
    } catch (...) {
        running_.store(false, std::memory_order_release);
        ::close(wake_[0]);
        ::close(wake_[1]);
        wake_[0] = wake_[1] = -1;
        return false;
    }
    return true;
}

void EventLoop::stop() noexcept {
    if (thread_.joinable()) {
        running_.store(false, std::memory_order_release);
        const char b = 1;
        (void)!write(wake_[1], &b, 1);
        thread_.join();
    }
    for (int& fd : wake_) {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
    count_ = 0;
}

void EventLoop::loop() noexcept {
    // Slot 0 is the wake pipe; watches follow. Built once, reused every poll.
    pollfd fds[kMaxWatches + 1];
    fds[0] = pollfd{wake_[0], POLLIN, 0};
    for (size_t i = 0; i < count_; ++i) {
        fds[i + 1] = pollfd{watches_[i].fd, watches_[i].events, 0};
    }
    const nfds_t n = static_cast<nfds_t>(count_ + 1);

    while (running()) {
        const int rc = poll(fds, n, -1);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (fds[0].revents != 0) {
            break;  // stop() was called
        }
        for (size_t i = 0; i < count_; ++i) {
            const short re = fds[i + 1].revents;
            if (re != 0) {
                watches_[i].cb(watches_[i].fd, re, watches_[i].user);
            }
        }
    }
}

}  // namespace af
