// Host test for the on-demand inference control (state, latest store,
// long-poll) and its HTTP server.
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

using namespace yolo;
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

static InferenceResult result(uint32_t id) {
    InferenceResult r;
    r.frame_id = id;
    r.ts_ms = 1000000 + id;
    r.inference_ms = 123.4f;
    r.detections_json = "[{\"class_id\":0,\"class_name\":\"person\",\"confidence\":0.900,\"x\":0.5000,\"y\":0.5000,\"w\":0.1000,\"h\":0.2000}]";
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
    InferenceControl c(8, false, [&](bool on, int fps) { calls.push_back({on, fps}); });
    json s = get(c, "GET", "/api/inference", "");
    CHECK(s == json::parse(R"({"enabled":false,"fps":1,"active_fps":8,"idle_fps":1})"));
    int st = 0;
    s = get(c, "POST", "/api/inference", R"({"enabled":true})", &st);
    CHECK(st == 200 && s["enabled"] == true && s["fps"] == 8);
    get(c, "POST", "/api/inference", R"({"enabled":true})");  // idempotent: no second callback
    CHECK(calls.size() == 1 && calls[0] == std::make_pair(true, 8));
    s = get(c, "POST", "/api/inference", R"({"enabled":false})");
    CHECK(s["enabled"] == false && s["fps"] == 1);
    CHECK(calls.size() == 2 && calls[1] == std::make_pair(false, 1));
    for (const char* bad : {"", "nope", "[]", "{}", R"({"enabled":1})", R"({"enabled":"true"})"}) {
        s = get(c, "POST", "/api/inference", bad, &st);
        CHECK(st == 400 && s.contains("error"));
    }
    CHECK(calls.size() == 2);
    get(c, "DELETE", "/api/inference", "", &st);
    CHECK(st == 404);
    get(c, "GET", "/api/other", "", &st);
    CHECK(st == 404);
    get(c, "POST", "/api/results/latest", "", &st);
    CHECK(st == 404);
    get(c, "GET", "/api/results/latest?after=x", "", &st);
    CHECK(st == 400);
    std::printf("toggle: ok\n");
}

static void test_latest() {
    InferenceControl c(8, true, nullptr);
    json s = get(c, "GET", "/api/results/latest", "");
    CHECK(s == json::parse(R"({"enabled":true,"frame_id":0,"detections":[],"timeout":true})"));
    c.publish(result(5));
    s = get(c, "GET", "/api/results/latest", "");
    CHECK(s["frame_id"] == 5 && s["ts_ms"] == 1000005 && s["age_ms"] == 95);
    CHECK(s["detections"].size() == 1 && s["detections"][0]["class_name"] == "person");
    CHECK(!s.contains("timeout"));
    s = get(c, "GET", "/api/results/latest?after=4", "");
    CHECK(s["frame_id"] == 5 && !s.contains("timeout"));
    s = get(c, "GET", "/api/results/latest?after=5", "");  // wait_ms 0: no wait
    CHECK(s["frame_id"] == 5 && s["timeout"] == true);
    std::printf("latest: ok\n");
}

static void test_wait_wakeup() {
    InferenceControl c(8, true, nullptr);
    c.publish(result(1));
    std::thread t([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        c.publish(result(2));
    });
    auto t0 = clk::now();
    json s = get(c, "GET", "/api/results/latest?after=1&wait_ms=3000", "");
    long el = ms_since(t0);
    t.join();
    CHECK(s["frame_id"] == 2 && !s.contains("timeout"));
    CHECK(el >= 100 && el < 1000);
    std::printf("wait wake-up on new result: ok (%ld ms)\n", el);
}

static void test_timeout() {
    InferenceControl c(8, true, nullptr);
    c.publish(result(3));
    auto t0 = clk::now();
    json s = get(c, "GET", "/api/results/latest?after=3&wait_ms=200", "");
    long el = ms_since(t0);
    CHECK(s["frame_id"] == 3 && s["timeout"] == true);
    CHECK(el >= 190 && el < 600);
    std::printf("timeout: ok (%ld ms)\n", el);
}

static void test_shutdown_wakes() {
    InferenceControl c(8, true, nullptr);
    std::thread t([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        c.shutdown();
    });
    auto t0 = clk::now();
    json s = get(c, "GET", "/api/results/latest?after=0&wait_ms=99999", "");
    long el = ms_since(t0);
    t.join();
    CHECK(s["frame_id"] == 0 && s["timeout"] == true && el < 1000);
    std::printf("shutdown wakes waiter: ok (%ld ms)\n", el);
}

static void test_disable_wakes() {
    InferenceControl c(8, true, nullptr);
    c.publish(result(7));
    std::thread t([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        c.set_enabled(false);
    });
    auto t0 = clk::now();
    json s = get(c, "GET", "/api/results/latest?after=7&wait_ms=5000", "");
    long el = ms_since(t0);
    t.join();
    CHECK(s["enabled"] == false && s["frame_id"] == 7 && el >= 100 && el < 1000);
    // Disabled: no waiting at all.
    t0 = clk::now();
    s = get(c, "GET", "/api/results/latest?after=7&wait_ms=5000", "");
    el = ms_since(t0);
    CHECK(s["enabled"] == false && s["frame_id"] == 7 && el < 50);
    std::printf("disable wakes waiter: ok; disabled answers at once (%ld ms)\n", el);
}

static std::string http(int port, const std::string& req) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
        ::close(fd);
        return "";
    }
    ::send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[1024];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) out.append(buf, static_cast<size_t>(n));
    ::close(fd);
    return out;
}

static void test_server() {
    InferenceControl c(8, false, nullptr);
    HttpControlServer srv(c);
    const int port = 18090;
    CHECK(srv.start(port));
    std::string r = http(port, "GET /api/inference HTTP/1.1\r\nHost: x\r\n\r\n");
    CHECK(r.rfind("HTTP/1.1 200 OK", 0) == 0);
    CHECK(r.find("Content-Type: application/json") != std::string::npos);
    CHECK(r.find("{\"enabled\":false,\"fps\":1,\"active_fps\":8,\"idle_fps\":1}") != std::string::npos);
    const std::string body = "{\"enabled\": true}";
    r = http(port, "POST /api/inference HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n" + body);
    CHECK(r.find("\"enabled\":true") != std::string::npos);
    // A long-poll in flight must not block a toggle on another connection.
    c.publish(result(9));
    std::string poll;
    auto t0 = clk::now();
    std::thread lp([&] { poll = http(port, "GET /api/results/latest?after=9&wait_ms=5000 HTTP/1.1\r\n\r\n"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    r = http(port, "POST /api/inference HTTP/1.1\r\nContent-Length: 17\r\n\r\n{\"enabled\":false}");
    lp.join();
    long el = ms_since(t0);
    CHECK(r.find("\"enabled\":false") != std::string::npos);
    CHECK(poll.find("\"frame_id\":9") != std::string::npos && poll.find("\"enabled\":false") != std::string::npos);
    CHECK(el < 1000);
    r = http(port, "POST /api/inference HTTP/1.1\r\nContent-Length: 3\r\n\r\nbad");
    CHECK(r.rfind("HTTP/1.1 400", 0) == 0);
    r = http(port, "GET /nope HTTP/1.1\r\n\r\n");
    CHECK(r.rfind("HTTP/1.1 404", 0) == 0);
    srv.stop();
    std::printf("http server: ok (long-poll released by toggle on another connection in %ld ms)\n", el);
}

int main() {
    test_toggle();
    test_latest();
    test_wait_wakeup();
    test_timeout();
    test_shutdown_wakes();
    test_disable_wakes();
    test_server();
    std::printf(g_fail ? "FAILED: %d\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
