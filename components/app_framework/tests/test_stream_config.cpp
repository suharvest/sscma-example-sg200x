/**
 * @file test_stream_config.cpp
 * @brief Host tests for af::StreamConfig: which endpoints start and whether VENC CH2 is configured.
 */
#include <cstring>

#include "af_test.h"
#include "stream_config.h"

using af::RtspOptions;
using af::StreamConfig;
using af::StreamMode;
using af::StreamOptions;
using af::StreamWsOptions;

TEST(defaults_start_nothing) {
    StreamConfig c;
    CHECK(c.mode() == StreamMode::None);
    CHECK(!c.needs_venc());
    CHECK(!c.rtsp_enabled);
    CHECK(!c.ws_enabled);
}

TEST(option_defaults) {
    StreamOptions s;
    CHECK_EQ(s.width, 1280);
    CHECK_EQ(s.height, 720);
    CHECK_EQ(s.fps, 30);
    CHECK_EQ(s.bitrate_kbps, 2500u);
    CHECK_EQ(s.gop, 30u);
    CHECK(s.low_latency);
    StreamWsOptions w;
    CHECK_EQ(w.port, 8001);
    CHECK(w.video && w.results && w.snapshot);
    CHECK_EQ(w.max_clients, 2);
    RtspOptions r;
    CHECK_EQ(r.port, 8554);
    CHECK(r.session_prefix == "live");
    CHECK_EQ(r.width, 0);  // legacy geometry unset
}

TEST(live_view_is_ws_only) {
    StreamConfig c;
    c.set_mode(StreamMode::WsOnly, StreamOptions{}, 9001, 8554);
    CHECK(c.mode() == StreamMode::WsOnly);
    CHECK(c.ws_enabled);
    CHECK(!c.rtsp_enabled);
    CHECK(c.needs_venc());
    CHECK_EQ(c.ws.port, 9001);
    CHECK_EQ(c.stream.fps, 30);
}

TEST(mode_matrix) {
    struct Row {
        StreamMode m;
        bool ws, rtsp, venc;
    } rows[] = {
        {StreamMode::None, false, false, false},
        {StreamMode::WsOnly, true, false, true},
        {StreamMode::RtspOnly, false, true, true},
        {StreamMode::Both, true, true, true},
    };
    for (const Row& r : rows) {
        StreamConfig c;
        c.set_mode(r.m, StreamOptions{}, 8001, 8554);
        CHECK(c.mode() == r.m);
        CHECK(c.ws_enabled == r.ws);
        CHECK(c.rtsp_enabled == r.rtsp);
        CHECK(c.needs_venc() == r.venc);
    }
}

TEST(mode_replaces_earlier_choices) {
    StreamConfig c;
    c.enable_rtsp(RtspOptions{});
    StreamWsOptions w;
    w.snapshot = false;
    c.enable_ws(w);
    CHECK(c.mode() == StreamMode::Both);
    c.set_mode(StreamMode::None, StreamOptions{}, 8001, 8554);
    CHECK(c.mode() == StreamMode::None);
    CHECK(!c.needs_venc());
    c.set_mode(StreamMode::WsOnly, StreamOptions{}, 8002, 8555);
    CHECK(!c.ws.snapshot);  // non-port WS options survive
    CHECK_EQ(c.ws.port, 8002);
    CHECK_EQ(c.rtsp.port, 8555);
}

TEST(ws_without_video_needs_no_venc) {
    StreamConfig c;
    StreamWsOptions w;
    w.video = false;
    c.enable_ws(w);
    CHECK(c.mode() == StreamMode::WsOnly);
    CHECK(!c.needs_venc());
    c.enable_rtsp(RtspOptions{});
    CHECK(c.needs_venc());
}

TEST(legacy_rtsp_geometry_keeps_old_semantics) {
    StreamConfig c;
    RtspOptions r;
    r.width = 640;
    r.height = 360;
    r.fps = 10;
    c.enable_rtsp(r);
    CHECK(c.mode() == StreamMode::RtspOnly);
    CHECK_EQ(c.stream.width, 640);
    CHECK_EQ(c.stream.height, 360);
    CHECK_EQ(c.stream.fps, 10);
    CHECK_EQ(c.stream.gop, 0u);           // ipcam default, as before
    CHECK_EQ(c.stream.bitrate_kbps, 0u);  // ipcam default, as before
    CHECK(!c.stream.low_latency);
}

TEST(legacy_with_stream_does_not_enable) {
    StreamConfig c;
    RtspOptions r;
    r.width = 1920;
    r.height = 1080;
    r.fps = 15;
    r.gop = 30;
    r.bitrate_kbps = 1200;
    r.max_bitrate_kbps = 1500;
    r.port = 9554;
    c.set_stream_legacy(r);
    CHECK(c.mode() == StreamMode::None);
    CHECK_EQ(c.stream.width, 1920);
    CHECK_EQ(c.stream.gop, 30u);
    CHECK_EQ(c.stream.max_bitrate_kbps, 1500u);
    CHECK_EQ(c.rtsp.port, 9554);
}

TEST(protocol_only_rtsp_keeps_stream) {
    StreamConfig c;
    StreamOptions s;
    s.width = 640;
    s.height = 480;
    s.fps = 20;
    c.set_stream(s);
    RtspOptions r;
    r.username = "u";
    c.enable_rtsp(r);
    CHECK_EQ(c.stream.width, 640);
    CHECK_EQ(c.stream.fps, 20);
    CHECK(c.stream.low_latency);
    CHECK(c.rtsp.username == "u");
}

TEST(parse_and_print_modes) {
    StreamMode m = StreamMode::Both;
    CHECK(af::parse_stream_mode("none", m) && m == StreamMode::None);
    CHECK(af::parse_stream_mode("ws", m) && m == StreamMode::WsOnly);
    CHECK(af::parse_stream_mode("rtsp", m) && m == StreamMode::RtspOnly);
    CHECK(af::parse_stream_mode("both", m) && m == StreamMode::Both);
    CHECK(!af::parse_stream_mode("WS", m) && m == StreamMode::Both);
    CHECK(!af::parse_stream_mode("", m));
    CHECK(!af::parse_stream_mode(nullptr, m));
    CHECK(std::strcmp(af::to_string(StreamMode::WsOnly), "ws") == 0);
}

AF_TEST_MAIN
