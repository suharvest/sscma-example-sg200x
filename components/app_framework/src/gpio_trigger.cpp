/**
 * @file gpio_trigger.cpp
 * @brief sysfs GPIO edge trigger.
 *
 * sysfs signals an edge as POLLPRI|POLLERR on the value file; the value must
 * then be re-read from offset 0 to re-arm the notification.
 */
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include "af/pipeline.h"
#include "af/trigger.h"

namespace af {

namespace {

bool write_file(const char* path, const char* text) noexcept {
    const int fd = ::open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const size_t len = std::strlen(text);
    const bool ok = ::write(fd, text, len) == static_cast<ssize_t>(len);
    ::close(fd);
    return ok;
}

/** Read the current raw level ('0'/'1') and re-arm the edge notification. */
int read_level(int fd) noexcept {
    char c = '0';
    if (lseek(fd, 0, SEEK_SET) < 0 || ::read(fd, &c, 1) != 1) {
        return -1;
    }
    return c == '1' ? 1 : 0;
}

}  // namespace

int GpioTrigger::open() noexcept {
    if (fd_ >= 0) {
        return fd_;
    }
    if (cfg_.line < 0) {
        return -1;
    }

    char path[64];
    std::snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/value", cfg_.line);
    if (access(path, F_OK) != 0) {
        char num[16];
        std::snprintf(num, sizeof(num), "%d", cfg_.line);
        // EBUSY means somebody exported it in between; the access() below decides.
        write_file("/sys/class/gpio/export", num);
    }

    char attr[64];
    std::snprintf(attr, sizeof(attr), "/sys/class/gpio/gpio%d/direction", cfg_.line);
    if (!write_file(attr, "in")) {
        return -1;
    }

    // The kernel's active_low makes "rising" mean "becoming active", so the
    // configured edge is always in logical terms.
    std::snprintf(attr, sizeof(attr), "/sys/class/gpio/gpio%d/active_low", cfg_.line);
    write_file(attr, cfg_.active_low ? "1" : "0");

    std::snprintf(attr, sizeof(attr), "/sys/class/gpio/gpio%d/edge", cfg_.line);
    const char* edge = cfg_.edge == Edge::Rising ? "rising" : cfg_.edge == Edge::Falling ? "falling" : "both";
    if (!write_file(attr, edge)) {
        return -1;
    }

    fd_ = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        return -1;
    }
    read_level(fd_);  // consume the initial state so the first poll waits for a real edge
    return fd_;
}

short GpioTrigger::poll_events() const noexcept {
    return POLLPRI | POLLERR;
}

bool GpioTrigger::on_ready(short revents, TriggerEvent& out) noexcept {
    if ((revents & (POLLPRI | POLLERR)) == 0) {
        return false;
    }
    const int level = read_level(fd_);
    if (level < 0) {
        return false;
    }

    const uint64_t now = monotonic_us();
    if (last_us_ != 0 && now - last_us_ < static_cast<uint64_t>(cfg_.debounce_ms) * 1000u) {
        return false;  // contact bounce
    }
    // With "both" every edge counts; otherwise drop an edge whose resulting
    // level contradicts the configured direction (a bounce that slipped past
    // the timer and settled the wrong way).
    if (cfg_.edge == Edge::Rising && level != 1) {
        return false;
    }
    if (cfg_.edge == Edge::Falling && level != 0) {
        return false;
    }
    last_us_ = now;

    out = TriggerEvent{};
    out.source = TriggerEvent::Source::Gpio;
    out.level = static_cast<uint8_t>(level);
    out.line = static_cast<uint16_t>(cfg_.line);
    out.seq = ++seq_;
    out.ts_us = now;
    return true;
}

void GpioTrigger::close() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace af
