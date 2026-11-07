/**
 * @file test_pipeline.cpp
 * @brief Host tests for af::Pipeline: ordering, early stop, timing, capacity, result attach.
 */
#include <cstring>
#include <vector>

#include "af/pipeline.h"
#include "af/sink.h"
#include "af_test.h"

using af::Context;
using af::Flow;
using af::Pipeline;
using af::RunReport;
using af::VideoFrame;

namespace {

// Deterministic clock: every read advances by the amount the running step asked for.
uint64_t g_now = 0;
uint64_t fake_clock() {
    return g_now;
}

struct State {
    std::vector<int> order;
    int detect_hits = 1;
    uint64_t cost_us[4] = {100, 250, 40, 7};
};

Flow s0(Context<State>& c) {
    c.state.order.push_back(0);
    g_now += c.state.cost_us[0];
    return Flow::Next;
}
Flow s1_detect(Context<State>& c) {
    c.state.order.push_back(1);
    g_now += c.state.cost_us[1];
    return c.state.detect_hits ? Flow::Next : c.reject("no target");
}
Flow s2(Context<State>& c) {
    c.state.order.push_back(2);
    g_now += c.state.cost_us[2];
    return Flow::Next;
}
Flow s3_fail(Context<State>& c) {
    c.state.order.push_back(3);
    g_now += c.state.cost_us[3];
    return c.error("tpu timeout");
}

VideoFrame test_frame(std::vector<uint8_t>& buf) {
    buf.assign(32 * 32 * 3, 0);
    return VideoFrame::packed(buf.data(), 32, 32, af::PixelFormat::RGB888);
}

}  // namespace

TEST(runs_all_steps_in_order) {
    Pipeline<State> p;
    p.set_clock(&fake_clock).add("a", s0).add("b", s1_detect).add("c", s2);
    CHECK(p.ok());
    CHECK_EQ(p.size(), 3u);
    CHECK(std::strcmp(p.name(1), "b") == 0);
    CHECK(p.name(3) == nullptr);

    std::vector<uint8_t> buf;
    State st;
    g_now = 1000;
    RunReport r = p.run(test_frame(buf), st);
    CHECK(r.completed());
    CHECK_EQ(static_cast<int>(r.flow), static_cast<int>(Flow::Next));
    CHECK_EQ(r.executed, 3);
    CHECK_EQ(r.stopped_at, 0xFF);
    CHECK(r.reason == nullptr);
    CHECK((st.order == std::vector<int>{0, 1, 2}));
}

TEST(step_us_is_per_step) {
    Pipeline<State> p;
    p.set_clock(&fake_clock).add("a", s0).add("b", s1_detect).add("c", s2);
    std::vector<uint8_t> buf;
    State st;
    g_now = 5;
    RunReport r = p.run(test_frame(buf), st);
    CHECK_EQ(r.step_us[0], 100u);
    CHECK_EQ(r.step_us[1], 250u);
    CHECK_EQ(r.step_us[2], 40u);
    CHECK_EQ(r.step_us[3], 0u);
    CHECK_EQ(r.total_us, 390u);
}

TEST(reject_stops_early) {
    Pipeline<State> p;
    p.set_clock(&fake_clock).add("a", s0).add("detect", s1_detect).add("c", s2);
    std::vector<uint8_t> buf;
    State st;
    st.detect_hits = 0;
    RunReport r = p.run(test_frame(buf), st);
    CHECK(r.stopped());
    CHECK_EQ(static_cast<int>(r.flow), static_cast<int>(Flow::Reject));
    CHECK_EQ(r.stopped_at, 1);
    CHECK_EQ(r.executed, 2);
    CHECK(r.reason && std::strcmp(r.reason, "no target") == 0);
    CHECK((st.order == std::vector<int>{0, 1}));
    CHECK_EQ(r.step_us[2], 0u);
    CHECK_EQ(r.total_us, 350u);
}

TEST(error_stops_early) {
    Pipeline<State> p;
    p.set_clock(&fake_clock).add("a", s0).add("fail", s3_fail).add("c", s2);
    std::vector<uint8_t> buf;
    State st;
    RunReport r = p.run(test_frame(buf), st);
    CHECK_EQ(static_cast<int>(r.flow), static_cast<int>(Flow::Error));
    CHECK_EQ(r.stopped_at, 1);
    CHECK(std::strcmp(r.reason, "tpu timeout") == 0);
    CHECK((st.order == std::vector<int>{0, 3}));
    CHECK(std::strcmp(af::to_string(r.flow), "error") == 0);
}

TEST(empty_pipeline_completes) {
    Pipeline<State> p;
    std::vector<uint8_t> buf;
    State st;
    RunReport r = p.run(test_frame(buf), st);
    CHECK(p.empty());
    CHECK(r.completed());
    CHECK_EQ(r.executed, 0);
    CHECK_EQ(r.total_us, 0u);
}

TEST(capacity_and_null_steps_flag_not_ok) {
    Pipeline<State> p;
    for (size_t i = 0; i < af::kMaxSteps; ++i) p.add("s", s2);
    CHECK(p.ok());
    p.add("overflow", s2);
    CHECK(!p.ok());
    CHECK_EQ(p.size(), af::kMaxSteps);

    Pipeline<State> q;
    q.add("null", nullptr);
    CHECK(!q.ok());
    CHECK(q.empty());
}

TEST(lambdas_and_crop_propagate_through_context) {
    struct S {
        int seen_w = 0;
    };
    Pipeline<S> p;
    p.add("crop", [](Context<S>& c) {
         c.frame = c.frame.sub_frame(af::Rect{8, 8, 10, 6});
         return Flow::Next;
     })
        .add("measure", [](Context<S>& c) {
            c.state.seen_w = c.frame.width;
            return c.frame.height == 6 ? Flow::Next : c.error("bad crop");
        });
    std::vector<uint8_t> buf;
    S st;
    Context<S> ctx(test_frame(buf), st);
    RunReport r = p.run(ctx);
    CHECK(r.completed());
    CHECK_EQ(st.seen_w, 10);
    CHECK(ctx.frame.planes[0].vir == buf.data() + 8 * 96 + 8 * 3);
}

TEST(real_clock_is_monotonic) {
    const uint64_t a = af::monotonic_us();
    const uint64_t b = af::monotonic_us();
    CHECK(b >= a);
    CHECK(a > 0);
}

TEST(result_attach_copies_outcome) {
    RunReport rep;
    rep.flow = Flow::Reject;
    rep.reason = "no target";
    rep.total_us = 12500;
    af::Result res;
    res.attach(rep);
    CHECK_EQ(static_cast<int>(res.flow), static_cast<int>(Flow::Reject));
    CHECK(std::strcmp(res.reason, "no target") == 0);
    CHECK(res.inference_ms > 12.49f && res.inference_ms < 12.51f);

    af::Box boxes[2];
    res.set_boxes(boxes, 2);
    CHECK(res.boxes == boxes);
    CHECK_EQ(res.box_count, 2u);
    CHECK(res.publish);
}

TEST(run_report_is_trivially_copyable) {
    CHECK(std::is_trivially_copyable<RunReport>::value);
}

AF_TEST_MAIN
