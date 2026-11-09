#include "http_control.h"

#include <arpa/inet.h>
#include <csignal>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <thread>
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

            FaceLedExpectation expectation;
            const json* exp_j = nullptr;
            if (params.contains("expected")) exp_j = &params["expected"];
            else if (req_json.contains("expected")) exp_j = &req_json["expected"];
            if (exp_j && action_id == "face_led") {
                std::string perr;
                if (!parseFaceLedExpectation(*exp_j, expectation, perr)) {
                    json err = {{"error", "INVALID_PARAMS"}, {"message", perr}};
                    send_http_response(client_fd, 400, "Bad Request", err.dump());
                    ::close(client_fd);
                    return;
                }
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

                bool ok = sm_->startAction(action_id, run_id, timeout_s, force, snapshot, expectation);
                if (ok) {
                    json resp = {
                        {"result", "ok"},
                        {"state", "COLLECTING"},
                        {"run_id", run_id}
                    };
                    if (expectation.active) resp["expected_colour"] = expectation.colour;
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
        } else if (method == "calibrate_stand") {
            handleCalibrateStand(client_fd, req_json, params);
            return;
        } else if (method == "set_stand_nominal") {
            handleSetStandNominal(client_fd, params);
            return;
        } else if (method == "set_face_led_layout") {
            // params: {"layout": {"rgb": [u, v], ...}, "persist": true}; ring coordinates
            if (!watcher_) {
                send_http_response(client_fd, 503, "Service Unavailable", json({{"error", "CONFIG_UNAVAILABLE"}}).dump());
            } else {
                AppConfig next = watcher_->config();
                std::vector<FaceLedSlotPos> lay;
                bool ok = params.contains("layout") && params["layout"].is_object();
                if (ok) {
                    for (auto it = params["layout"].begin(); it != params["layout"].end(); ++it) {
                        const auto& v = it.value();
                        if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number() ||
                            std::abs(v[0].get<float>()) > 10.0f || std::abs(v[1].get<float>()) > 10.0f) { ok = false; break; }
                        lay.push_back({it.key(), v[0].get<float>(), v[1].get<float>()});
                    }
                }
                if (!ok || lay.size() < 3) {
                    send_http_response(client_fd, 400, "Bad Request",
                                       json({{"error", "INVALID_PARAMS"}, {"message", "layout: {place: [u, v]} x >= 3, |u|,|v| <= 10"}}).dump());
                } else {
                    next.face_led_layout = lay;
                    std::string err;
                    const bool persist = params.value("persist", true);
                    if (persist && !save_face_led_layout(watcher_->confPath(), next, &err)) {
                        send_http_response(client_fd, 500, "Internal Server Error", json({{"error", "PERSIST_FAILED"}, {"message", err}}).dump());
                    } else {
                        watcher_->setConfig(next);
                        json l = json::object();
                        for (const auto& sl : lay) l[sl.place] = {sl.x, sl.y};
                        send_http_response(client_fd, 200, "OK", json({{"result", "ok"}, {"layout", l}, {"persisted", persist},
                                                                       {"conf_path", watcher_->confPath()}}).dump());
                    }
                }
            }
        } else if (method == "get_config") {
            if (!watcher_) {
                send_http_response(client_fd, 503, "Service Unavailable", json({{"error", "CONFIG_UNAVAILABLE"}}).dump());
            } else {
                const AppConfig c = watcher_->config();
                json layout = json::object();
                for (const auto& sl : c.face_led_layout) layout[sl.place] = {sl.x, sl.y};
                json resp = {
                    {"result", "ok"},
                    {"conf_path", watcher_->confPath()},
                    {"nominal", nominalJson(c)},
                    {"tolerances", {{"gear_tooth_tolerance_deg", c.gear_tooth_tolerance_deg},
                                    {"bilateral_height_tolerance_norm", c.bilateral_height_tolerance_norm},
                                    {"spine_tilt_tolerance_deg", c.spine_tilt_tolerance_deg}}},
                    {"face_led_layout", layout},
                    {"config_version", c.config_version}
                };
                send_http_response(client_fd, 200, "OK", resp.dump());
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
                {"supported_methods", json::array({"get_status", "start_action", "stop_action", "reset", "get_last_report",
                                                    "calibrate_stand", "set_stand_nominal", "set_face_led_layout", "get_config"})}
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

json HttpControlServer::nominalJson(const AppConfig& c) {
    return {
        {"nominal_stand_theta_l_deg", c.nominal_stand_theta_l_deg},
        {"nominal_stand_theta_r_deg", c.nominal_stand_theta_r_deg},
        {"nominal_stand_disparity_deg", c.nominal_stand_disparity_deg},
        {"nominal_stand_spine_tilt_deg", c.nominal_stand_spine_tilt_deg},
        {"nominal_stand_knee_diff_norm", c.nominal_stand_knee_diff_norm}
    };
}

// Apply a new stand standard: write the conf file first (when persisting), then swap the
// in-memory config, so a failed write leaves both unchanged.
bool HttpControlServer::applyNominal(const AppConfig& next, bool persist, std::string& err) {
    if (persist && !save_stand_nominal(watcher_->confPath(), next, &err)) return false;
    watcher_->setConfig(next);
    return true;
}

// calibrate_stand: collect one stand window (same 30-frame window and per-frame evaluation as
// start_action stand) on a golden sample, take the medians as the new standard, persist it and
// answer with the values and frame counts. Blocks this (single) control connection until the
// window closes: at most timeout_s + 2 s.
void HttpControlServer::handleCalibrateStand(int client_fd, const json& req_json, const json& params) {
    auto reply = [&](int code, const char* text, const json& body) {
        send_http_response(client_fd, code, text, body.dump());
        ::close(client_fd);
    };
    if (!sm_ || !watcher_) return reply(503, "Service Unavailable", {{"error", "STATE_MACHINE_UNAVAILABLE"}});

    float timeout_s = 10.0f;
    if (params.contains("timeout_s") && params["timeout_s"].is_number()) timeout_s = params["timeout_s"].get<float>();
    timeout_s = std::clamp(timeout_s, 1.0f, 10.0f);
    const bool persist = params.value("persist", true);
    const bool force = params.value("force", false);
    std::string run_id = params.value("run_id", req_json.value("run_id", std::string("calibrate_stand")));

    if (sm_->getState() == ActionState::COLLECTING && !force) {
        return reply(409, "Conflict", {{"error", "BUSY"}, {"current_run_id", sm_->getCurrentRunId()}});
    }
    const AppConfig before = watcher_->config();
    if (!sm_->startAction("calibrate_stand", run_id, timeout_s, force, before)) {
        return reply(500, "Internal Server Error", {{"error", "FAILED_TO_START"}});
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(static_cast<int>((timeout_s + 2.0f) * 1000));
    while (std::chrono::steady_clock::now() < deadline) {
        sm_->checkTimeout();
        if (sm_->getState() != ActionState::COLLECTING) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (sm_->getState() == ActionState::COLLECTING) sm_->stopAction();

    const ActionReport rep = sm_->getLastReport();
    const StandCalibration& cal = rep.calibration;
    json frames = {{"total", cal.total_frames}, {"valid", cal.valid_frames},
                   {"spine", cal.spine_frames}, {"required", cal.required_frames}};
    if (rep.action_id != "calibrate_stand" || !cal.ok) {
        return reply(422, "Unprocessable Entity",
                     {{"error", "INSUFFICIENT_VALID_FRAMES"},
                      {"reason", cal.reason.empty() ? rep.status_code : cal.reason},
                      {"frames", frames}, {"nominal", nominalJson(before)}});
    }
    AppConfig next = before;
    next.nominal_stand_theta_l_deg = cal.theta_l_deg;
    next.nominal_stand_theta_r_deg = cal.theta_r_deg;
    next.nominal_stand_disparity_deg = cal.disparity_deg;
    next.nominal_stand_spine_tilt_deg = cal.spine_tilt_deg;
    next.nominal_stand_knee_diff_norm = cal.knee_diff_norm;
    std::string err;
    if (!applyNominal(next, persist, err)) {
        return reply(500, "Internal Server Error", {{"error", "PERSIST_FAILED"}, {"message", err},
                                                    {"measured", nominalJson(next)}, {"frames", frames}});
    }
    MA_LOGI(TAG, "Stand standard calibrated from %d frames: L=%.2f R=%.2f disp=%.2f spine=%.2f knee_diff=%.4f (persist=%d)",
            cal.valid_frames, cal.theta_l_deg, cal.theta_r_deg, cal.disparity_deg, cal.spine_tilt_deg, cal.knee_diff_norm, persist);
    reply(200, "OK", {{"result", "ok"}, {"run_id", run_id}, {"nominal", nominalJson(next)},
                      {"previous", nominalJson(before)}, {"frames", frames}, {"persisted", persist},
                      {"conf_path", watcher_->confPath()}});
}

void HttpControlServer::handleSetStandNominal(int client_fd, const json& params) {
    auto reply = [&](int code, const char* text, const json& body) {
        send_http_response(client_fd, code, text, body.dump());
        ::close(client_fd);
    };
    if (!watcher_) return reply(503, "Service Unavailable", {{"error", "CONFIG_UNAVAILABLE"}});
    const AppConfig before = watcher_->config();
    AppConfig next = before;
    struct Key { const char* name; float* dst; float lo; float hi; };
    const Key keys[] = {
        {"nominal_stand_theta_l_deg", &next.nominal_stand_theta_l_deg, -180.0f, 180.0f},
        {"nominal_stand_theta_r_deg", &next.nominal_stand_theta_r_deg, -180.0f, 180.0f},
        {"nominal_stand_disparity_deg", &next.nominal_stand_disparity_deg, 0.0f, 180.0f},
        {"nominal_stand_spine_tilt_deg", &next.nominal_stand_spine_tilt_deg, 0.0f, 180.0f},
        {"nominal_stand_knee_diff_norm", &next.nominal_stand_knee_diff_norm, -1.0f, 1.0f},
    };
    int n = 0;
    for (const auto& k : keys) {
        if (!params.contains(k.name)) continue;
        if (!params[k.name].is_number()) return reply(400, "Bad Request", {{"error", "INVALID_PARAMS"}, {"key", k.name}});
        const float v = params[k.name].get<float>();
        if (v < k.lo || v > k.hi) return reply(400, "Bad Request", {{"error", "OUT_OF_RANGE"}, {"key", k.name}, {"min", k.lo}, {"max", k.hi}});
        *k.dst = v;
        n++;
    }
    if (n == 0) return reply(400, "Bad Request", {{"error", "INVALID_PARAMS"}, {"message", "no nominal_stand_* key given"}});
    const bool persist = params.value("persist", true);
    std::string err;
    if (!applyNominal(next, persist, err)) return reply(500, "Internal Server Error", {{"error", "PERSIST_FAILED"}, {"message", err}});
    reply(200, "OK", {{"result", "ok"}, {"nominal", nominalJson(next)}, {"previous", nominalJson(before)},
                      {"persisted", persist}, {"conf_path", watcher_->confPath()}});
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
