#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "http_control.h"
#include "inference_control.h"
#include "json.hpp"

using namespace aruco;
using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++g_fail;                                                    \
        }                                                                \
    } while (0)

static long ms_since(clk::time_point t) {
    return static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t).count());
}

static InferenceResult make_dummy_result(uint32_t id) {
    InferenceResult r;
    r.frame_id = id;
    r.ts_ms = 1000000 + id;
    r.inference_ms = 4.2f;
    r.count = 1;
    r.rejected = 2;
    r.markers_json = "[{\"id\":0,\"center\":[320.0,180.0],\"pose\":{\"t\":[0.05,0.02,0.45],\"r_deg\":[0.0,0.0,15.0],\"error\":0.001}}]";
    return r;
}

static json get(InferenceControl& c, const std::string& m, const std::string& t, const std::string& b,
                int* status = nullptr) {
    HttpReply r = handle_control_request(c, m, t, b, 1000100);
    if (status) *status = r.status;
    return json::parse(r.body);
}

static void test_toggle() {
    std::vector<std::pair<bool, int>> calls;
    InferenceControl c(25, false, [&](bool on, int fps) { calls.push_back({on, fps}); });

    // Initial state: constructor invoked on_change(false, 1) to sync initial idle camera FPS
    CHECK(calls.size() == 1 && calls[0] == std::make_pair(false, 1));

    json s = get(c, "GET", "/api/inference", "");
    CHECK(s == json::parse(R"({"enabled":false,"fps":1,"active_fps":25,"idle_fps":1})"));

    int st = 0;
    s = get(c, "POST", "/api/inference", R"({"enabled":true})", &st);
    CHECK(st == 200 && s["enabled"] == true && s["fps"] == 25);

    get(c, "POST", "/api/inference", R"({"enabled":true})");  // idempotent
    CHECK(calls.size() == 2 && calls[1] == std::make_pair(true, 25));

    s = get(c, "POST", "/api/inference", R"({"enabled":false})");
    CHECK(s["enabled"] == false && s["fps"] == 1);
    CHECK(calls.size() == 3 && calls[2] == std::make_pair(false, 1));

    for (const char* bad : {"", "nope", "[]", "{}", R"({"enabled":1})", R"({"enabled":"true"})"}) {
        s = get(c, "POST", "/api/inference", bad, &st);
        CHECK(st == 400 && s.contains("error"));
    }

    get(c, "DELETE", "/api/inference", "", &st);
    CHECK(st == 404);
}

static void test_latest_and_wait() {
    InferenceControl c(25, true, nullptr);

    // Initial state: no frames
    json s = get(c, "GET", "/api/results/latest", "");
    CHECK(s["frame_id"] == 0 && s["count"] == 0 && s["timeout"] == true);

    c.publish(make_dummy_result(1));
    s = get(c, "GET", "/api/results/latest", "");
    CHECK(s["frame_id"] == 1 && s["count"] == 1 && !s.contains("timeout"));
    CHECK(s["ts_ms"] == 1000001);
    CHECK(s["rejected"] == 2);
    CHECK(s["age_ms"] == 99);

    // Long poll waiting for frame > 1
    const auto t0 = clk::now();
    std::thread publisher([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        c.publish(make_dummy_result(2));
    });

    s = get(c, "GET", "/api/results/latest?after=1&wait_ms=500", "");
    publisher.join();
    const long dur = ms_since(t0);
    CHECK(dur >= 40 && dur < 400);
    CHECK(s["frame_id"] == 2 && s["count"] == 1);
    CHECK(s["ts_ms"] == 1000002);
    CHECK(s["age_ms"] == 98);
}

static void test_trigger() {
    std::vector<std::pair<bool, int>> calls;
    InferenceControl c(25, false, [&](bool on, int fps) { calls.push_back({on, fps}); });
    CHECK(calls.size() == 1 && calls[0] == std::make_pair(false, 1));

    std::thread publisher([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        c.publish(make_dummy_result(10));
    });

    json s = get(c, "POST", "/api/trigger?wait_ms=1000", "");
    publisher.join();

    CHECK(s["success"] == true);
    CHECK(s["frame_id"] == 10);
    CHECK(s["count"] == 1);
    CHECK(s["ts_ms"] == 1000010);
    CHECK(s["age_ms"] == 90);
    CHECK(c.enabled() == false);  // restored to idle
    CHECK(c.continuous_enabled() == false);
    CHECK(calls.size() == 3);
    CHECK(calls[0] == std::make_pair(false, 1));
    CHECK(calls[1] == std::make_pair(true, 25));
    CHECK(calls[2] == std::make_pair(false, 1));
}

static void test_concurrent_triggers_and_set_enabled() {
    // 1. Two concurrent triggers: only transition to active on first, and to idle on last
    {
        std::vector<std::pair<bool, int>> calls;
        InferenceControl c(25, false, [&](bool on, int fps) { calls.push_back({on, fps}); });
        CHECK(calls.size() == 1 && calls[0] == std::make_pair(false, 1));

        auto s1 = c.trigger_scope();
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 2 && calls[1] == std::make_pair(true, 25));

        auto s2 = c.trigger_scope();
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 2);  // No duplicate notification

        s1.release();
        CHECK(c.enabled() == true);  // s2 is still active!
        CHECK(calls.size() == 2);    // Still at active FPS

        s2.release();
        CHECK(c.enabled() == false);
        CHECK(calls.size() == 3 && calls[2] == std::make_pair(false, 1));
    }

    // 2. Trigger while set_enabled(true) happens
    {
        std::vector<std::pair<bool, int>> calls;
        InferenceControl c(25, false, [&](bool on, int fps) { calls.push_back({on, fps}); });
        CHECK(calls.size() == 1);

        auto s = c.trigger_scope();
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 2 && calls[1] == std::make_pair(true, 25));

        // External set_enabled(true)
        CHECK(c.set_enabled(true) == true);
        CHECK(c.continuous_enabled() == true);
        CHECK(calls.size() == 2);  // Effective was already true

        // Trigger ends: must NOT switch back to idle because continuous is true!
        s.release();
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 2);

        // Now external set_enabled(false) switches to idle
        CHECK(c.set_enabled(false) == true);
        CHECK(c.enabled() == false);
        CHECK(calls.size() == 3 && calls[2] == std::make_pair(false, 1));
    }

    // 3. Continuous enabled, trigger starts, external set_enabled(false) happens during trigger
    {
        std::vector<std::pair<bool, int>> calls;
        InferenceControl c(25, true, [&](bool on, int fps) { calls.push_back({on, fps}); });
        CHECK(calls.size() == 1 && calls[0] == std::make_pair(true, 25));

        auto s = c.trigger_scope();
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 1);  // Already active

        // External set_enabled(false) during trigger
        CHECK(c.set_enabled(false) == true);
        CHECK(c.continuous_enabled() == false);
        // Effective must REMAIN TRUE because trigger is still running!
        CHECK(c.enabled() == true);
        CHECK(calls.size() == 1);  // Not switched to idle yet!

        // Trigger finishes: NOW effective becomes false and switches to idle
        s.release();
        CHECK(c.enabled() == false);
        CHECK(calls.size() == 2 && calls[1] == std::make_pair(false, 1));
    }

    // 4. Multi-threaded stress: concurrent triggers and set_enabled
    {
        std::atomic<bool> stop_flag{false};
        InferenceControl c(25, false, nullptr);
        std::vector<std::thread> workers;

        for (int i = 0; i < 4; ++i) {
            workers.emplace_back([&] {
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    auto s = c.trigger_scope();
                    std::this_thread::yield();
                }
            });
        }
        for (int i = 0; i < 2; ++i) {
            workers.emplace_back([&] {
                bool state = false;
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    c.set_enabled(state);
                    state = !state;
                    std::this_thread::yield();
                }
            });
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        stop_flag.store(true);
        for (auto& w : workers) {
            w.join();
        }

        c.set_enabled(false);
        CHECK(c.continuous_enabled() == false);
        CHECK(c.enabled() == false);
    }
}

static std::string http_roundtrip(int port, const std::string& raw) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a;
    std::memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<uint16_t>(port));

    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) {
        ::close(fd);
        return "";
    }
    ::send(fd, raw.data(), raw.size(), 0);

    std::string out;
    char buf[1024];
    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return out;
}

static void test_http_server() {
    const int port = 18090;
    InferenceControl c(25, false, nullptr);
    HttpControlServer server(c);
    CHECK(server.start(port));

    // GET /api/inference
    std::string resp = http_roundtrip(port, "GET /api/inference HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    CHECK(resp.find("200 OK") != std::string::npos);
    CHECK(resp.find("\"enabled\":false") != std::string::npos);

    // POST /api/inference
    const std::string body = R"({"enabled":true})";
    std::string post_req = "POST /api/inference HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " +
                           std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    resp = http_roundtrip(port, post_req);
    CHECK(resp.find("200 OK") != std::string::npos);
    CHECK(resp.find("\"enabled\":true") != std::string::npos);
    CHECK(c.enabled() == true);

    // No CORS headers on any reply
    CHECK(resp.find("Access-Control") == std::string::npos);

    // POST without / with a non-JSON Content-Type is rejected and does not change state
    const std::string off = R"({"enabled":false})";
    for (const char* ct : {"", "Content-Type: text/plain\r\n", "Content-Type: application/x-www-form-urlencoded\r\n"}) {
        resp = http_roundtrip(port, std::string("POST /api/inference HTTP/1.1\r\nHost: localhost\r\n") + ct +
                                        "Content-Length: " + std::to_string(off.size()) + "\r\nConnection: close\r\n\r\n" + off);
        CHECK(resp.find("415 Unsupported Media Type") != std::string::npos);
        CHECK(resp.find("Access-Control") == std::string::npos);
    }
    resp = http_roundtrip(port, "POST /api/trigger HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    CHECK(resp.find("415") != std::string::npos);
    CHECK(c.enabled() == true);

    // Content-Type with parameters is accepted
    resp = http_roundtrip(port, "POST /api/inference HTTP/1.1\r\nHost: localhost\r\nContent-Type: Application/JSON; charset=utf-8\r\nContent-Length: " +
                                    std::to_string(off.size()) + "\r\nConnection: close\r\n\r\n" + off);
    CHECK(resp.find("200 OK") != std::string::npos);
    CHECK(c.enabled() == false);

    // Preflight is not approved
    resp = http_roundtrip(port, "OPTIONS /api/inference HTTP/1.1\r\nHost: localhost\r\nOrigin: http://evil.example\r\n"
                                "Access-Control-Request-Method: POST\r\nConnection: close\r\n\r\n");
    CHECK(resp.find("405 Method Not Allowed") != std::string::npos);
    CHECK(resp.find("Access-Control") == std::string::npos);

    server.stop();
}

static void test_precheck() {
    CHECK(is_json_content_type("application/json"));
    CHECK(is_json_content_type("  APPLICATION/json ; charset=utf-8"));
    CHECK(!is_json_content_type(""));
    CHECK(!is_json_content_type("text/plain"));
    CHECK(!is_json_content_type("application/jsonx"));
    CHECK(precheck_request("GET", "") == 0);
    CHECK(precheck_request("POST", "application/json") == 0);
    CHECK(precheck_request("POST", "text/plain") == 415);
    CHECK(precheck_request("OPTIONS", "application/json") == 405);
    CHECK(precheck_request("DELETE", "application/json") == 405);
}

static void test_request_deadline() {
    const int port = 18091;
    InferenceControl c(25, false, nullptr);
    HttpControlServer server(c);
    CHECK(server.start(port));

    // Slowloris: one byte every 500 ms never completes the headers; the whole
    // request is cut off near kRequestDeadlineMs even though no single gap
    // exceeds a per-recv timeout.
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a;
    std::memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<uint16_t>(port));
    CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    const auto t0 = clk::now();
    const char* drip = "GET /api/inference HTTP/1.1\r\nX-A: b";
    for (size_t i = 0; i < std::strlen(drip) && ms_since(t0) < 6000; ++i) {
        if (::send(fd, drip + i, 1, MSG_NOSIGNAL) <= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    std::string out;
    char buf[512];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) out.append(buf, static_cast<size_t>(n));
    ::close(fd);
    const long dur = ms_since(t0);
    CHECK(out.find("408 Request Timeout") != std::string::npos);
    CHECK(dur >= HttpControlServer::kRequestDeadlineMs - 100 && dur < HttpControlServer::kRequestDeadlineMs + 1500);

    // A normal request still works afterwards
    std::string resp = http_roundtrip(port, "GET /api/inference HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    CHECK(resp.find("200 OK") != std::string::npos);
    server.stop();
}

static void test_http_server_lifecycle() {
    // A. Rapid start/stop without connections
    {
        InferenceControl c(25, false, nullptr);
        HttpControlServer server(c);
        const int port = 18091;
        for (int i = 0; i < 5; ++i) {
            CHECK(server.start(port));
            server.stop();
        }
    }

    // B. Stop with active long-polling client: server.stop() must interrupt client and return promptly
    {
        InferenceControl c(25, false, nullptr);
        c.set_enabled(true);
        c.publish(make_dummy_result(1));
        HttpControlServer server(c);
        const int port = 18092;
        CHECK(server.start(port));

        std::atomic<int> client_status{0}; // 0: init, 1: sent, -1: error
        std::atomic<bool> client_done{false};

        std::thread client([&] {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a;
            std::memset(&a, 0, sizeof(a));
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            a.sin_port = htons(static_cast<uint16_t>(port));

            if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
                client_status.store(-1);
                ::close(fd);
                client_done.store(true);
                return;
            }

            // Request a 5-second wait
            std::string req = "GET /api/results/latest?after=1&wait_ms=5000 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
            ::send(fd, req.data(), req.size(), 0);
            client_status.store(1);
            char buf[512];
            while (::recv(fd, buf, sizeof(buf), 0) > 0) {}
            ::close(fd);
            client_done.store(true);
        });

        // Wait with deadline until client has sent the request
        const auto wait_start = clk::now();
        while (client_status.load() == 0 && ms_since(wait_start) < 2000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(client_status.load() == 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));

        const auto t0 = clk::now();
        // server.stop() MUST interrupt the long poll and return quickly (< 1000ms), NOT hang for 5s
        server.stop();
        const long stop_dur = ms_since(t0);
        CHECK(stop_dur < 1000);

        client.join();
        CHECK(client_done.load() == true);
    }

    // C. Stop with active trigger client
    {
        InferenceControl c(25, false, nullptr);
        HttpControlServer server(c);
        const int port = 18093;
        CHECK(server.start(port));

        std::atomic<int> client_status{0};
        std::thread client([&] {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a;
            std::memset(&a, 0, sizeof(a));
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            a.sin_port = htons(static_cast<uint16_t>(port));

            if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
                client_status.store(-1);
                ::close(fd);
                return;
            }

            std::string req = "POST /api/trigger?wait_ms=5000 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n";
            ::send(fd, req.data(), req.size(), 0);
            client_status.store(1);
            char buf[512];
            while (::recv(fd, buf, sizeof(buf), 0) > 0) {}
            ::close(fd);
        });

        // Wait until client has sent the request and trigger actually took effect in InferenceControl
        const auto wait_start = clk::now();
        while ((client_status.load() == 0 || !c.effective_enabled()) && ms_since(wait_start) < 2000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(client_status.load() == 1);
        CHECK(c.effective_enabled());

        const auto t0 = clk::now();
        server.stop();
        const long stop_dur = ms_since(t0);
        CHECK(stop_dur < 1000);

        client.join();
    }
}

int main() {
    test_toggle();
    test_latest_and_wait();
    test_trigger();
    test_concurrent_triggers_and_set_enabled();
    test_precheck();
    test_http_server();
    test_request_deadline();
    test_http_server_lifecycle();

    if (g_fail == 0) {
        std::printf("ALL ARUCO INFERENCE CONTROL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TESTS FAILED\n", g_fail);
    return 1;
}
