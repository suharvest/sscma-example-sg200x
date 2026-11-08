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

#define TAG "aruco-http"

namespace aruco {

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
       << "Access-Control-Allow-Origin: *\r\n"
       << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
       << "Access-Control-Allow-Headers: Content-Type\r\n"
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
    ::signal(SIGPIPE, SIG_IGN);

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
    ctl_.shutdown();
    if (thread_ && thread_->joinable()) thread_->join();
    thread_.reset();
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
    for (int i = 0; i < 100 && clients_.load() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void HttpControlServer::accept_loop() {
    while (running_.load()) {
        pollfd pfd = {server_fd_, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 200);
        if (pr <= 0) continue;

        sockaddr_in caddr;
        socklen_t clen = sizeof(caddr);
        const int cfd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&caddr), &clen);
        if (cfd < 0) continue;

        if (clients_.load() >= kMaxClients) {
            send_reply(cfd, {503, "{\"error\":\"too many connections\"}"});
            ::close(cfd);
            continue;
        }

        clients_.fetch_add(1, std::memory_order_relaxed);
        std::thread([this, cfd]() {
            handle_client(cfd);
            ::close(cfd);
            clients_.fetch_sub(1, std::memory_order_relaxed);
        }).detach();
    }
}

void HttpControlServer::handle_client(int fd) {
    timeval tv{kReadTimeoutMs / 1000, (kReadTimeoutMs % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string buf;
    buf.reserve(1024);
    char chunk[1024];
    size_t header_end = std::string::npos;
    size_t content_length = 0;
    bool has_content_length = false;

    while (buf.size() < kMaxRequestBytes) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        buf.append(chunk, static_cast<size_t>(n));

        if (header_end == std::string::npos) {
            header_end = buf.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // Parse Content-Length (case-insensitive header lookup)
                const std::string headers = buf.substr(0, header_end);
                std::istringstream hs(headers);
                std::string line;
                while (std::getline(hs, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    const size_t colon = line.find(':');
                    if (colon == std::string::npos) continue;
                    std::string name = line.substr(0, colon);
                    for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    if (name == "content-length") {
                        size_t vstart = colon + 1;
                        while (vstart < line.size() && line[vstart] == ' ') ++vstart;
                        content_length = std::strtoul(line.c_str() + vstart, nullptr, 10);
                        has_content_length = true;
                        break;
                    }
                }
            }
        }
        if (header_end != std::string::npos) {
            const size_t body_have = buf.size() - (header_end + 4);
            if (!has_content_length || body_have >= content_length) break;
        }
    }

    if (header_end == std::string::npos) {
        send_reply(fd, {400, "{\"error\":\"malformed HTTP request\"}"});
        return;
    }

    const std::string line1 = buf.substr(0, buf.find("\r\n"));
    const size_t s1 = line1.find(' ');
    const size_t s2 = s1 == std::string::npos ? std::string::npos : line1.find(' ', s1 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos) {
        send_reply(fd, {400, "{\"error\":\"malformed request line\"}"});
        return;
    }
    const std::string method = line1.substr(0, s1);
    const std::string target = line1.substr(s1 + 1, s2 - s1 - 1);

    if (method == "OPTIONS") {
        send_reply(fd, {200, "{}"});
        return;
    }

    const std::string body = buf.substr(header_end + 4, has_content_length ? content_length : std::string::npos);
    const HttpReply r = handle_control_request(ctl_, method, target, body, unix_ms());
    send_reply(fd, r);
}

}  // namespace aruco
