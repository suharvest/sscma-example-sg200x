#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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

    json s = get(c, "GET", "/api/inference", "");
    CHECK(s == json::parse(R"({"enabled":false,"fps":1,"active_fps":25,"idle_fps":1})"));

    int st = 0;
    s = get(c, "POST", "/api/inference", R"({"enabled":true})", &st);
    CHECK(st == 200 && s["enabled"] == true && s["fps"] == 25);

    get(c, "POST", "/api/inference", R"({"enabled":true})");  // idempotent
    CHECK(calls.size() == 1 && calls[0] == std::make_pair(true, 25));

    s = get(c, "POST", "/api/inference", R"({"enabled":false})");
    CHECK(s["enabled"] == false && s["fps"] == 1);
    CHECK(calls.size() == 2 && calls[1] == std::make_pair(false, 1));

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
}

static void test_trigger() {
    std::vector<std::pair<bool, int>> calls;
    InferenceControl c(25, false, [&](bool on, int fps) { calls.push_back({on, fps}); });

    std::thread publisher([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        c.publish(make_dummy_result(10));
    });

    json s = get(c, "POST", "/api/trigger?wait_ms=1000", "");
    publisher.join();

    CHECK(s["success"] == true);
    CHECK(s["frame_id"] == 10);
    CHECK(s["count"] == 1);
    CHECK(c.enabled() == false);  // restored to idle
    CHECK(calls.size() == 2);
    CHECK(calls[0] == std::make_pair(true, 25));
    CHECK(calls[1] == std::make_pair(false, 1));
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

    server.stop();
}

int main() {
    test_toggle();
    test_latest_and_wait();
    test_trigger();
    test_http_server();

    if (g_fail == 0) {
        std::printf("ALL ARUCO INFERENCE CONTROL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TESTS FAILED\n", g_fail);
    return 1;
}
