#include "http_control.h"

#include <arpa/inet.h>
#include <csignal>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#include "json.hpp"

#ifndef MA_LOGW
#define MA_LOGW(tag, fmt, ...) fprintf(stderr, "[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGI(tag, fmt, ...) fprintf(stdout, "[INFO][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define MA_LOGE(tag, fmt, ...) fprintf(stderr, "[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

#define TAG "HttpControl"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace microduck {

using json = nlohmann::json;

HttpControlServer::HttpControlServer() = default;

HttpControlServer::~HttpControlServer() {
    stop();
}

bool HttpControlServer::start(int port, ActionStateMachine* sm, ConfigWatcher* watcher) {
    if (running_.load()) return true;

    // Ignore SIGPIPE so socket writes to closed clients don't kill the daemon
    ::signal(SIGPIPE, SIG_IGN);

    port_ = port;
    sm_ = sm;
    watcher_ = watcher;

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        MA_LOGE(TAG, "Failed to create socket: %s", std::strerror(errno));
        return false;
    }

    int opt = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    // SO_REUSEPORT removed intentionally to prevent port collisions between daemon instances

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        MA_LOGE(TAG, "Failed to bind to 0.0.0.0:%d: %s", port_, std::strerror(errno));
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::listen(server_fd_, 16) < 0) {
        MA_LOGE(TAG, "Failed to listen on socket: %s", std::strerror(errno));
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    thread_ = std::make_unique<std::thread>(&HttpControlServer::serverLoop, this);
    MA_LOGI(TAG, "HTTP control service listening on 0.0.0.0:%d", port_);
    return true;
}

void HttpControlServer::stop() {
    if (!running_.exchange(false)) return;

    if (thread_ && thread_->joinable()) {
        thread_->join();
        thread_.reset();
    }

    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
    MA_LOGI(TAG, "HTTP control service stopped");
}

static void send_http_response(int client_fd, int status_code, const std::string& status_text, const std::string& body) {
    try {
        std::ostringstream ss;
        ss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
        ss << "Content-Type: application/json; charset=utf-8\r\n";
        ss << "Content-Length: " << body.size() << "\r\n";
        ss << "Connection: close\r\n\r\n";
        ss << body;

        const std::string resp = ss.str();
        size_t total_sent = 0;
        while (total_sent < resp.size()) {
            ssize_t n = ::send(client_fd, resp.data() + total_sent, resp.size() - total_sent, MSG_NOSIGNAL);
            if (n <= 0) break;
            total_sent += static_cast<size_t>(n);
        }
    } catch (...) {
        // Suppress any formatting or socket exception during error reporting
    }
}

void HttpControlServer::handleClient(int client_fd) {
    try {
        // Set socket receive and send timeouts to 2.0s to prevent slow client hangs
        struct timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);

        std::string req;
        char buf[2048];
        size_t header_end = std::string::npos;
        size_t content_length = 0;
        bool found_cl = false;

        while (req.size() < 65536) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                break;
            }
            const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            if (remaining_ms <= 0) {
                break;
            }
            struct pollfd pfd;
            pfd.fd = client_fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            int poll_ret = ::poll(&pfd, 1, static_cast<int>(remaining_ms));
            if (poll_ret <= 0) {
                break;  // Timeout or error
            }

            ssize_t n = ::recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            req.append(buf, static_cast<size_t>(n));

            if (header_end == std::string::npos) {
                header_end = req.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    std::string lower_headers = req.substr(0, header_end);
                    for (auto& c : lower_headers) c = std::tolower(c);
                    size_t pos = lower_headers.find("content-length:");
                    if (pos != std::string::npos) {
                        size_t line_end = lower_headers.find("\r\n", pos);
                        std::string cl_str = lower_headers.substr(pos + 15, line_end - (pos + 15));
                        try {
                            content_length = std::stoul(cl_str);
                            found_cl = true;
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }

            if (header_end != std::string::npos) {
                size_t body_len = req.size() - (header_end + 4);
                if (!found_cl || body_len >= content_length) {
                    break;
                }
            }
        }

        if (header_end == std::string::npos) {
            json err = {{"error", "BAD_REQUEST_OR_TIMEOUT"}, {"message", "Incomplete HTTP headers"}};
            send_http_response(client_fd, 400, "Bad Request", err.dump());
            ::close(client_fd);
            return;
        }

        if (found_cl) {
            size_t body_len = req.size() - (header_end + 4);
            if (body_len < content_length) {
                json err = {{"error", "REQUEST_TRUNCATED_OR_TIMEOUT"},
                            {"expected_bytes", content_length},
                            {"received_bytes", body_len}};
                send_http_response(client_fd, 400, "Bad Request", err.dump());
                ::close(client_fd);
                return;
            }
        }

        // Parse method and path from first line
        size_t first_line_end = req.find("\r\n");
        std::string first_line = req.substr(0, first_line_end);
        std::istringstream fl_iss(first_line);
        std::string http_method, uri, http_version;
        fl_iss >> http_method >> uri >> http_version;

        std::string body = req.substr(header_end + 4);
        if (found_cl && body.size() > content_length) {
            body = body.substr(0, content_length);
        }

        // Only handle POST /api/control
        if (http_method != "POST" || uri != "/api/control") {
            json err = {{"error", "NOT_FOUND"}, {"path", uri}};
            send_http_response(client_fd, 404, "Not Found", err.dump());
            ::close(client_fd);
            return;
        }

        json req_json;
        try {
            req_json = json::parse(body);
        } catch (const std::exception& e) {
            json err = {{"error", "INVALID_JSON"}, {"message", e.what()}};
            send_http_response(client_fd, 400, "Bad Request", err.dump());
            ::close(client_fd);
            return;
        }

        if (!req_json.is_object() || !req_json.contains("method") || !req_json["method"].is_string()) {
            json err = {{"error", "INVALID_REQUEST"}, {"message", "Missing or invalid method field"}};
            send_http_response(client_fd, 400, "Bad Request", err.dump());
            ::close(client_fd);
            return;
        }

        const std::string method = req_json["method"].get<std::string>();

        // Strict params validation: if present, must be an object
        if (req_json.contains("params") && !req_json["params"].is_object()) {
            json err = {{"error", "INVALID_PARAMS"}, {"message", "params must be a JSON object"}};
            send_http_response(client_fd, 400, "Bad Request", err.dump());
            ::close(client_fd);
            return;
        }

        json params = req_json.value("params", json::object());

        if (method == "start_action") {
            std::string action_id = "";
            if (params.contains("action_id") && params["action_id"].is_string()) {
                action_id = params["action_id"].get<std::string>();
            } else if (req_json.contains("action_id") && req_json["action_id"].is_string()) {
                action_id = req_json["action_id"].get<std::string>();
            }

            if (action_id.empty()) {
                json err = {{"error", "INVALID_PARAMS"}, {"message", "action_id must be a string"}};
                send_http_response(client_fd, 400, "Bad Request", err.dump());
                ::close(client_fd);
                return;
            }

            if (action_id != "stand" && action_id != "face_led") {
                json err = {
                    {"error", "UNSUPPORTED_ACTION"},
                    {"action_id", action_id},
                    {"supported_actions", json::array({"stand", "face_led"})}
                };
                send_http_response(client_fd, 400, "Bad Request", err.dump());
                ::close(client_fd);
                return;
            }

            std::string run_id = "";
            if (params.contains("run_id") && params["run_id"].is_string()) {
                run_id = params["run_id"].get<std::string>();
            } else if (req_json.contains("run_id") && req_json["run_id"].is_string()) {
                run_id = req_json["run_id"].get<std::string>();
            }

            float timeout_s = 3.0f;
            if (params.contains("timeout_s") && params["timeout_s"].is_number()) {
                timeout_s = params["timeout_s"].get<float>();
            } else if (req_json.contains("timeout_s") && req_json["timeout_s"].is_number()) {
                timeout_s = req_json["timeout_s"].get<float>();
            }
            if (timeout_s < 1.0f || timeout_s > 10.0f) {
                timeout_s = 3.0f;
            }

            bool force = false;
            if (params.contains("force") && params["force"].is_boolean()) {
                force = params["force"].get<bool>();
            } else if (req_json.contains("force") && req_json["force"].is_boolean()) {
                force = req_json["force"].get<bool>();
            }

            if (sm_) {
                if (sm_->getState() == ActionState::COLLECTING && !force) {
                    json err = {
                        {"error", "BUSY"},
                        {"current_run_id", sm_->getCurrentRunId()},
                        {"state", "COLLECTING"}
                    };
                    send_http_response(client_fd, 409, "Conflict", err.dump());
                    ::close(client_fd);
                    return;
                }

                AppConfig snapshot;
                if (watcher_) {
                    snapshot = watcher_->createSnapshot();
                }

                bool ok = sm_->startAction(action_id, run_id, timeout_s, force, snapshot);
                if (ok) {
                    json resp = {
                        {"result", "ok"},
                        {"state", "COLLECTING"},
                        {"run_id", run_id}
                    };
                    send_http_response(client_fd, 200, "OK", resp.dump());
                } else {
                    json err = {
                        {"error", "FAILED_TO_START"},
                        {"state", sm_->getStateString()}
                    };
                    send_http_response(client_fd, 500, "Internal Server Error", err.dump());
                }
            } else {
                json err = {{"error", "STATE_MACHINE_UNAVAILABLE"}};
                send_http_response(client_fd, 503, "Service Unavailable", err.dump());
            }
        } else if (method == "stop_action") {
            if (sm_) {
                sm_->stopAction();
                json resp = {
                    {"result", "ok"},
                    {"state", sm_->getStateString()},
                    {"has_report", sm_->hasReport()}
                };
                send_http_response(client_fd, 200, "OK", resp.dump());
            } else {
                json err = {{"error", "STATE_MACHINE_UNAVAILABLE"}};
                send_http_response(client_fd, 503, "Service Unavailable", err.dump());
            }
        } else if (method == "get_status") {
            if (sm_) {
                sm_->checkTimeout();
                json resp = {
                    {"result", "ok"},
                    {"state", sm_->getStateString()},
                    {"action_id", sm_->getCurrentActionId()},
                    {"run_id", sm_->getCurrentRunId()},
                    {"collected_frames", sm_->getCollectedFrameCount()},
                    {"has_report", sm_->hasReport()}
                };
                send_http_response(client_fd, 200, "OK", resp.dump());
            } else {
                json err = {{"error", "STATE_MACHINE_UNAVAILABLE"}};
                send_http_response(client_fd, 503, "Service Unavailable", err.dump());
            }
        } else if (method == "reset") {
            if (sm_) {
                sm_->reset();
                json resp = {
                    {"result", "ok"},
                    {"state", "IDLE"}
                };
                send_http_response(client_fd, 200, "OK", resp.dump());
            } else {
                json err = {{"error", "STATE_MACHINE_UNAVAILABLE"}};
                send_http_response(client_fd, 503, "Service Unavailable", err.dump());
            }
        } else if (method == "get_last_report") {
            if (sm_) {
                sm_->checkTimeout();
                ActionState cur_state = sm_->getState();
                if (cur_state == ActionState::COLLECTING) {
                    json resp = {
                        {"state", "COLLECTING"},
                        {"action_id", sm_->getCurrentActionId()},
                        {"run_id", sm_->getCurrentRunId()},
                        {"collected_frames", sm_->getCollectedFrameCount()},
                        {"total_required", 30},
                        {"action_report", nullptr}
                    };
                    send_http_response(client_fd, 200, "OK", resp.dump());
                } else if (sm_->hasReport()) {
                    // Return action_report directly per Spec 4.1 contract
                    json resp = sm_->getLastReportJson();
                    resp["state"] = sm_->getStateString();
                    send_http_response(client_fd, 200, "OK", resp.dump());
                } else {
                    json resp = {
                        {"action_report", nullptr},
                        {"state", "IDLE"}
                    };
                    send_http_response(client_fd, 200, "OK", resp.dump());
                }
            } else {
                json err = {{"error", "STATE_MACHINE_UNAVAILABLE"}};
                send_http_response(client_fd, 503, "Service Unavailable", err.dump());
            }
        } else {
            json err = {
                {"error", "UNKNOWN_METHOD"},
                {"method", method},
                {"supported_methods", json::array({"get_status", "start_action", "stop_action", "reset", "get_last_report"})}
            };
            send_http_response(client_fd, 400, "Bad Request", err.dump());
        }

        ::close(client_fd);
    } catch (const std::exception& e) {
        MA_LOGE(TAG, "Unhandled exception in handleClient: %s", e.what());
        try {
            json err = {{"error", "INTERNAL_EXCEPTION"}, {"message", e.what()}};
            send_http_response(client_fd, 500, "Internal Server Error", err.dump());
        } catch (...) {}
        ::close(client_fd);
    } catch (...) {
        MA_LOGE(TAG, "Unknown fatal exception in handleClient");
        ::close(client_fd);
    }
}

void HttpControlServer::serverLoop() {
    while (running_.load()) {
        struct pollfd pfd;
        pfd.fd = server_fd_;
        pfd.events = POLLIN;
        pfd.revents = 0;

        int ret = ::poll(&pfd, 1, 200);  // 200ms timeout
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue;

        if (pfd.revents & POLLIN) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = ::accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
            if (client_fd >= 0) {
                try {
                    handleClient(client_fd);
                } catch (const std::exception& e) {
                    MA_LOGE(TAG, "Exception during client handling in serverLoop: %s", e.what());
                    ::close(client_fd);
                } catch (...) {
                    MA_LOGE(TAG, "Unknown exception during client handling in serverLoop");
                    ::close(client_fd);
                }
            }
        }
    }
}

}  // namespace microduck
