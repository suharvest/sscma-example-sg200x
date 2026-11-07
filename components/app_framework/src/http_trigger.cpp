/**
 * @file http_trigger.cpp
 * @brief One-request-per-connection HTTP trigger endpoint on a raw socket.
 *
 * Only the request line is parsed: "<METHOD> <path>[?query] HTTP/1.x".
 * Headers and body are ignored. The response is written and the connection
 * closed immediately.
 */
#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "af/pipeline.h"
#include "af/trigger.h"

namespace af {

namespace {

void send_all(int fd, const char* data, size_t len) noexcept {
    while (len > 0) {
        const ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
        if (n <= 0) {
            return;
        }
        data += n;
        len -= static_cast<size_t>(n);
    }
}

void respond(int fd, const char* status, std::string_view body) noexcept {
    char head[160];
    const int n = std::snprintf(head, sizeof(head),
                                "HTTP/1.1 %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                                status, body.size());
    if (n > 0 && static_cast<size_t>(n) < sizeof(head)) {
        send_all(fd, head, static_cast<size_t>(n));
        send_all(fd, body.data(), body.size());
    }
}

/** Copy the value of "tag=" from a query string into @p out (URL-decoding not attempted). */
void extract_tag(const char* query, size_t qlen, char* out, size_t outlen) noexcept {
    const char* p = query;
    const char* end = query + qlen;
    while (p < end) {
        const char* amp = static_cast<const char*>(std::memchr(p, '&', static_cast<size_t>(end - p)));
        const char* stop = amp ? amp : end;
        if (stop - p > 4 && std::memcmp(p, "tag=", 4) == 0) {
            size_t n = static_cast<size_t>(stop - p - 4);
            if (n >= outlen) {
                n = outlen - 1;
            }
            std::memcpy(out, p + 4, n);
            out[n] = '\0';
            return;
        }
        p = stop + 1;
    }
}

}  // namespace

int HttpTrigger::open() noexcept {
    if (fd_ >= 0) {
        return fd_;
    }
    fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd_ < 0) {
        return -1;
    }
    const int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg_.port);
    if (inet_pton(AF_INET, cfg_.bind ? cfg_.bind : "0.0.0.0", &addr.sin_addr) != 1 ||
        ::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(fd_, 4) != 0) {
        close();
        return -1;
    }
    return fd_;
}

short HttpTrigger::poll_events() const noexcept {
    return POLLIN;
}

bool HttpTrigger::on_ready(short revents, TriggerEvent& out) noexcept {
    if ((revents & POLLIN) == 0) {
        return false;
    }
    const int c = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (c < 0) {
        return false;
    }

    timeval tv{};
    tv.tv_sec = cfg_.recv_timeout_ms / 1000;
    tv.tv_usec = static_cast<suseconds_t>((cfg_.recv_timeout_ms % 1000) * 1000);
    setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Read until the end of the request line (or the buffer fills / times out).
    char req[512];
    size_t len = 0;
    while (len < sizeof(req) - 1) {
        const ssize_t n = ::recv(c, req + len, sizeof(req) - 1 - len, 0);
        if (n <= 0) {
            break;
        }
        len += static_cast<size_t>(n);
        if (std::memchr(req, '\n', len) != nullptr) {
            break;
        }
    }
    req[len] = '\0';

    // "<METHOD> <target> ..."
    bool accepted = false;
    char tag[TriggerEvent::kTagLen] = {};
    const char* sp1 = std::strchr(req, ' ');
    const bool method_ok = sp1 && ((sp1 - req == 3 && std::memcmp(req, "GET", 3) == 0) || (sp1 - req == 4 && std::memcmp(req, "POST", 4) == 0));
    if (method_ok) {
        const char* target = sp1 + 1;
        const char* sp2 = std::strchr(target, ' ');
        const size_t tlen = sp2 ? static_cast<size_t>(sp2 - target) : std::strlen(target);
        const char* q = static_cast<const char*>(std::memchr(target, '?', tlen));
        const size_t plen = q ? static_cast<size_t>(q - target) : tlen;
        const char* path = cfg_.path ? cfg_.path : "/trigger";
        if (plen == std::strlen(path) && std::memcmp(target, path, plen) == 0) {
            accepted = true;
            if (q) {
                extract_tag(q + 1, tlen - plen - 1, tag, sizeof(tag));
            }
        }
    }

    if (!accepted) {
        respond(c, "404 Not Found", "{\"accepted\":false}");
        ::close(c);
        return false;
    }

    if (cfg_.sync_response) {
        std::lock_guard<std::mutex> lock(mu_);
        if (held_fd_ >= 0) {
            respond(c, "409 Conflict", "{\"accepted\":false,\"error\":\"busy\"}");
            ::close(c);
            return false;
        }
    }

    out = TriggerEvent{};
    out.source = TriggerEvent::Source::Http;
    out.line = cfg_.port;
    out.seq = ++seq_;
    out.ts_us = monotonic_us();
    std::memcpy(out.tag, tag, sizeof(tag));

    if (cfg_.sync_response) {
        // Keep the connection; complete() answers with the verdict.
        timeval stv{};
        stv.tv_sec = cfg_.send_timeout_ms / 1000;
        stv.tv_usec = static_cast<suseconds_t>((cfg_.send_timeout_ms % 1000) * 1000);
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));
        std::lock_guard<std::mutex> lock(mu_);
        held_fd_ = c;
        held_seq_ = out.seq;
        return true;
    }

    char body[64];
    const int n = std::snprintf(body, sizeof(body), "{\"accepted\":true,\"seq\":%u}", out.seq);
    respond(c, "202 Accepted", std::string_view(body, n > 0 ? static_cast<size_t>(n) : 0));
    ::close(c);
    return true;
}

void HttpTrigger::complete(const TriggerEvent& ev, std::string_view body) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (held_fd_ < 0 || ev.source != TriggerEvent::Source::Http || ev.seq != held_seq_) {
        return;
    }
    char fallback[64];
    if (body.empty()) {
        const int n = std::snprintf(fallback, sizeof(fallback), "{\"accepted\":true,\"seq\":%u}", ev.seq);
        body = std::string_view(fallback, n > 0 ? static_cast<size_t>(n) : 0);
    }
    respond(held_fd_, "200 OK", body);
    ::close(held_fd_);
    held_fd_ = -1;
}

void HttpTrigger::close() noexcept {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (held_fd_ >= 0) {
            respond(held_fd_, "503 Service Unavailable", "{\"accepted\":false,\"error\":\"shutdown\"}");
            ::close(held_fd_);
            held_fd_ = -1;
        }
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

const char* to_string(TriggerEvent::Source s) noexcept {
    switch (s) {
        case TriggerEvent::Source::Gpio:
            return "gpio";
        case TriggerEvent::Source::Http:
            return "http";
        case TriggerEvent::Source::Manual:
            return "manual";
    }
    return "unknown";
}

}  // namespace af
