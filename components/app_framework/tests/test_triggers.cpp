/**
 * @file test_triggers.cpp
 * @brief Host tests for EventLoop + HttpTrigger over loopback (pure POSIX, no device).
 */
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "af/trigger.h"
#include "af_test.h"
#include "event_loop.h"

namespace {

struct Collected {
    std::mutex mu;
    std::vector<af::TriggerEvent> events;
    af::TriggerSource* src = nullptr;
};

void on_fd(int, short revents, void* user) {
    auto* c = static_cast<Collected*>(user);
    af::TriggerEvent ev;
    if (c->src->on_ready(revents, ev)) {
        std::lock_guard<std::mutex> lock(c->mu);
        c->events.push_back(ev);
    }
}

std::string http(uint16_t port, const char* request) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::string out;
    if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
        (void)!send(fd, request, std::strlen(request), 0);
        char buf[512];
        ssize_t n;
        while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) out.append(buf, static_cast<size_t>(n));
    }
    close(fd);
    return out;
}

/** Pick a free port by binding to 0. */
uint16_t free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    close(fd);
    return ntohs(a.sin_port);
}

size_t wait_events(Collected& c, size_t n) {
    for (int i = 0; i < 200; ++i) {
        {
            std::lock_guard<std::mutex> lock(c.mu);
            if (c.events.size() >= n) return c.events.size();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::lock_guard<std::mutex> lock(c.mu);
    return c.events.size();
}

}  // namespace

TEST(event_loop_rejects_bad_watches) {
    af::EventLoop loop;
    CHECK(!loop.watch(-1, 0, on_fd, nullptr));
    CHECK(!loop.watch(0, 0, nullptr, nullptr));
    CHECK(loop.start());  // nothing to watch is still a valid loop
    CHECK(loop.running());
    CHECK(!loop.watch(0, 1, on_fd, nullptr));  // not while running
    loop.stop();
    CHECK(!loop.running());
    loop.stop();  // idempotent
}

TEST(http_trigger_accepts_path_and_tag) {
    af::HttpTrigger::Config cfg;
    cfg.port = free_port();
    cfg.bind = "127.0.0.1";
    af::HttpTrigger trig(cfg);
    Collected col;
    col.src = &trig;

    const int fd = trig.open();
    CHECK(fd >= 0);
    af::EventLoop loop;
    CHECK(loop.watch(fd, trig.poll_events(), on_fd, &col));
    CHECK(loop.start());

    const std::string r1 = http(cfg.port, "POST /trigger?tag=lot42&x=1 HTTP/1.1\r\nHost: a\r\n\r\n");
    CHECK(r1.find("202 Accepted") != std::string::npos);
    CHECK(r1.find("\"seq\":1") != std::string::npos);

    const std::string r2 = http(cfg.port, "GET /other HTTP/1.1\r\n\r\n");
    CHECK(r2.find("404") != std::string::npos);

    const std::string r3 = http(cfg.port, "GET /trigger HTTP/1.0\r\n\r\n");
    CHECK(r3.find("\"seq\":2") != std::string::npos);

    const std::string r4 = http(cfg.port, "DELETE /trigger HTTP/1.1\r\n\r\n");
    CHECK(r4.find("404") != std::string::npos);

    CHECK_EQ(wait_events(col, 2), 2u);
    loop.stop();
    trig.close();

    std::lock_guard<std::mutex> lock(col.mu);
    if (col.events.size() == 2) {
        CHECK(col.events[0].source == af::TriggerEvent::Source::Http);
        CHECK(std::strcmp(col.events[0].tag, "lot42") == 0);
        CHECK_EQ(col.events[0].seq, 1u);
        CHECK_EQ(col.events[0].line, cfg.port);
        CHECK(col.events[0].ts_us > 0);
        CHECK_EQ(col.events[1].tag[0], '\0');
        CHECK_EQ(col.events[1].seq, 2u);
    }
    CHECK(std::strcmp(af::to_string(af::TriggerEvent::Source::Http), "http") == 0);
}

TEST(http_trigger_truncates_long_tag) {
    af::HttpTrigger::Config cfg;
    cfg.port = free_port();
    cfg.bind = "127.0.0.1";
    af::HttpTrigger trig(cfg);
    Collected col;
    col.src = &trig;
    af::EventLoop loop;
    CHECK(loop.watch(trig.open(), trig.poll_events(), on_fd, &col));
    CHECK(loop.start());

    const std::string tag(100, 'x');
    const std::string req = "GET /trigger?tag=" + tag + " HTTP/1.1\r\n\r\n";
    CHECK(http(cfg.port, req.c_str()).find("202") != std::string::npos);
    CHECK_EQ(wait_events(col, 1), 1u);
    loop.stop();

    std::lock_guard<std::mutex> lock(col.mu);
    if (!col.events.empty()) {
        CHECK_EQ(std::strlen(col.events[0].tag), af::TriggerEvent::kTagLen - 1);
    }
}

TEST(http_trigger_port_conflict_fails_open) {
    af::HttpTrigger::Config cfg;
    cfg.port = free_port();
    cfg.bind = "127.0.0.1";
    af::HttpTrigger a(cfg), b(cfg);
    CHECK(a.open() >= 0);
    CHECK_EQ(b.open(), -1);
}

TEST(http_trigger_sync_response_returns_verdict) {
    af::HttpTrigger::Config cfg;
    cfg.port = free_port();
    cfg.bind = "127.0.0.1";
    cfg.sync_response = true;
    af::HttpTrigger trig(cfg);
    Collected col;
    col.src = &trig;
    af::EventLoop loop;
    CHECK(loop.watch(trig.open(), trig.poll_events(), on_fd, &col));
    CHECK(loop.start());

    // The first caller blocks until complete(); run it on its own thread.
    std::string first;
    std::thread caller([&] { first = http(cfg.port, "POST /trigger?tag=A HTTP/1.1\r\n\r\n"); });
    CHECK_EQ(wait_events(col, 1), 1u);

    // A second request while the first is held is refused, not queued.
    const std::string second = http(cfg.port, "POST /trigger HTTP/1.1\r\n\r\n");
    CHECK(second.find("409") != std::string::npos);

    af::TriggerEvent ev;
    {
        std::lock_guard<std::mutex> lock(col.mu);
        ev = col.events[0];
    }
    af::TriggerEvent stale = ev;
    stale.seq = ev.seq + 7;
    trig.complete(stale, "{\"verdict\":\"WRONG\"}");  // ignored: not the held request
    trig.complete(ev, "{\"verdict\":\"OK\"}");
    caller.join();
    CHECK(first.find("200 OK") != std::string::npos);
    CHECK(first.find("{\"verdict\":\"OK\"}") != std::string::npos);
    CHECK(first.find("WRONG") == std::string::npos);
    CHECK(first.find("Content-Length: 16") != std::string::npos);

    // Free again after completion; empty body falls back to the seq document.
    std::string third;
    std::thread caller3([&] { third = http(cfg.port, "GET /trigger HTTP/1.1\r\n\r\n"); });
    CHECK_EQ(wait_events(col, 2), 2u);
    {
        std::lock_guard<std::mutex> lock(col.mu);
        ev = col.events[1];
    }
    trig.complete(ev, "");
    caller3.join();
    CHECK(third.find("\"seq\":2") != std::string::npos);

    // A held request is answered (503) when the trigger closes.
    std::string fourth;
    std::thread caller4([&] { fourth = http(cfg.port, "GET /trigger HTTP/1.1\r\n\r\n"); });
    CHECK_EQ(wait_events(col, 3), 3u);
    loop.stop();
    trig.close();
    caller4.join();
    CHECK(fourth.find("503") != std::string::npos);
}

TEST(http_trigger_large_body_is_sent_whole) {
    af::HttpTrigger::Config cfg;
    cfg.port = free_port();
    cfg.bind = "127.0.0.1";
    cfg.sync_response = true;
    af::HttpTrigger trig(cfg);
    Collected col;
    col.src = &trig;
    af::EventLoop loop;
    CHECK(loop.watch(trig.open(), trig.poll_events(), on_fd, &col));
    CHECK(loop.start());
    std::string resp;
    std::thread caller([&] { resp = http(cfg.port, "POST /trigger HTTP/1.1\r\n\r\n"); });
    CHECK_EQ(wait_events(col, 1), 1u);
    const std::string body = "{\"pad\":\"" + std::string(4000, 'x') + "\"}";
    {
        std::lock_guard<std::mutex> lock(col.mu);
        trig.complete(col.events[0], body);
    }
    caller.join();
    loop.stop();
    CHECK(resp.size() > body.size());
    CHECK(resp.compare(resp.size() - body.size(), body.size(), body) == 0);
}

AF_TEST_MAIN
