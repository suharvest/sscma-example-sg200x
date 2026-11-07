#include "http_control.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#if __has_include(<sscma.h>)
#include <sscma.h>
#endif
#ifndef MA_LOGI
#define MA_LOGI(tag, fmt, ...) fprintf(stdout, "[INFO][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGW(tag, fmt, ...) fprintf(stderr, "[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGE(tag, fmt, ...) fprintf(stderr, "[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define TAG "yolo-http"

namespace yolo {

static constexpr size_t kMaxRequestBytes = 8192;
static constexpr int kReadTimeoutMs = 2000;

static uint64_t unix_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

static const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

static void send_reply(int fd, const HttpReply& r) {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << r.status << " " << status_text(r.status) << "\r\n"
       << "Content-Type: application/json\r\n"
       << "Content-Length: " << r.body.size() << "\r\n"
       << "Cache-Control: no-store\r\n"
       << "Connection: close\r\n\r\n"
       << r.body;
    const std::string out = ss.str();
    size_t sent = 0;
    while (sent < out.size()) {
        const ssize_t n = ::send(fd, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

HttpControlServer::~HttpControlServer() {
    stop();
}

bool HttpControlServer::start(int port) {
    if (running_.load()) return true;
    ::signal(SIGPIPE, SIG_IGN);  // a client closing early must not kill the app

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        MA_LOGE(TAG, "socket: %s", std::strerror(errno));
        return false;
    }
    int opt = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(server_fd_, 16) < 0) {
        MA_LOGE(TAG, "bind/listen 0.0.0.0:%d: %s", port, std::strerror(errno));
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    thread_ = std::make_unique<std::thread>(&HttpControlServer::accept_loop, this);
    MA_LOGI(TAG, "Inference control on http://0.0.0.0:%d/api/inference", port);
    return true;
}

void HttpControlServer::stop() {
    if (!running_.exchange(false)) return;
    ctl_.shutdown();  // long-polls return now
    if (thread_ && thread_->joinable()) thread_->join();
    thread_.reset();
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
    // Connection threads are detached and reference *this; wait for them
    // (bounded by the socket timeouts) before the object goes away.
    for (int i = 0; i < 100 && clients_.load() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    MA_LOGI(TAG, "Inference control stopped");
}

void HttpControlServer::accept_loop() {
    while (running_.load()) {
        pollfd pfd{server_fd_, POLLIN, 0};
        const int ret = ::poll(&pfd, 1, 200);
        if (ret < 0 && errno != EINTR) break;
        if (ret <= 0 || !(pfd.revents & POLLIN)) continue;

        const int fd = ::accept(server_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        if (clients_.load() >= kMaxClients) {
            send_reply(fd, {503, "{\"error\":\"too many connections\"}"});
            ::close(fd);
            continue;
        }
        clients_.fetch_add(1);
        try {
            std::thread([this, fd] {
                handle_client(fd);
                ::close(fd);
                clients_.fetch_sub(1);
            }).detach();
        } catch (...) {
            clients_.fetch_sub(1);
            ::close(fd);
        }
    }
}

void HttpControlServer::handle_client(int fd) {
    timeval tv{kReadTimeoutMs / 1000, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    std::string req;
    size_t header_end = std::string::npos;
    size_t content_length = 0;
    char buf[1024];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kReadTimeoutMs);
    while (req.size() < kMaxRequestBytes) {
        if (header_end != std::string::npos && req.size() - (header_end + 4) >= content_length) break;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                              deadline - std::chrono::steady_clock::now())
                              .count();
        if (left <= 0) break;
        pollfd pfd{fd, POLLIN, 0};
        if (::poll(&pfd, 1, static_cast<int>(left)) <= 0) break;
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, static_cast<size_t>(n));
        if (header_end == std::string::npos) {
            header_end = req.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                std::string headers = req.substr(0, header_end);
                for (auto& c : headers) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                const size_t pos = headers.find("\r\ncontent-length:");
                if (pos != std::string::npos) {
                    content_length = std::strtoul(headers.c_str() + pos + 17, nullptr, 10);
                }
            }
        }
    }
    if (header_end == std::string::npos || req.size() - (header_end + 4) < content_length) {
        send_reply(fd, {400, "{\"error\":\"incomplete or oversized request\"}"});
        return;
    }

    std::istringstream line(req.substr(0, req.find("\r\n")));
    std::string method, target;
    line >> method >> target;
    const std::string body = req.substr(header_end + 4, content_length);
    try {
        send_reply(fd, handle_control_request(ctl_, method, target, body, unix_ms()));
    } catch (const std::exception& e) {
        MA_LOGE(TAG, "request %s %s: %s", method.c_str(), target.c_str(), e.what());
        send_reply(fd, {500, "{\"error\":\"internal error\"}"});
    }
}

}  // namespace yolo
