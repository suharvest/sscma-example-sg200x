// microduck-pose-qa -- Microduck 17-Keypoint Robot Pose Estimation & QA Daemon
//
// Pipeline: camera -> Microduck Pose (SG2002 TPU) -> 17 keypoints ->
// kinematic QA -> MQTT / debug WebSocket / RTSP / HTTP Control (8082)
//
// Camera, streaming, MQTT, signal handling and the frame loop are owned by
// af::App (components/app_framework); this file holds the QA logic only.

#include <getopt.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include <af/app.h>
#include <af/ma_image.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include <sscma.h>

extern "C" {
#include <cvi_comm_isp.h>
#include <cvi_isp.h>
#include <cvi_ae.h>
#include <cvi_awb.h>
}

#include "app_config.h"
#include "face_led_inspector.h"
#include "http_control.h"
#include "pose.h"
#include "pose_detector.h"
#include "result_payload.h"

using namespace ma;
using namespace microduck;

#define TAG "microduck-pose-qa"

static struct {
    std::string model_path;

    std::string mqtt_host = "127.0.0.1";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/microduck-pose-qa/results";

    int inference_width = 384;
    int inference_height = 384;
    int inference_fps = 30;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 30;

    bool enable_debug = true;
    int debug_port = 8001;
    int control_port = 8082;
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;

    float cli_confidence = -1.0f;
    float cli_kpt_confidence = -1.0f;
} g_config;

static const char* MODEL_CANDIDATES[] = {
    "/userdata/local/models/microduck_pose_17kpt_384_int8.cvimodel",
    "/usr/share/supervisor/models/microduck_pose_17kpt_384_int8.cvimodel",
    "/userdata/local/models/microduck_pose_17kpt_384_bf16.cvimodel",
    "/usr/share/supervisor/models/microduck_pose_17kpt_384_bf16.cvimodel",
    "/userdata/local/models/microduck_pose_17kpt_384.cvimodel",
};

static std::string findPoseModel() {
    for (const char* p : MODEL_CANDIDATES) {
        struct stat st;
        if (::stat(p, &st) == 0 && S_ISREG(st.st_mode)) {
            return p;
        }
    }
    return std::string();
}

static const char* APP_CONFIG_PATH = "/userdata/local/apps/microduck-pose-qa.config.json";

static PoseDetector* g_detector = nullptr;
static ConfigWatcher* g_watcher = nullptr;
static ActionStateMachine g_state_machine;
static HttpControlServer g_http_control;
static uint32_t g_frame_id = 0;

// Performance and latency tracking (sliding 100-frame window)
struct LatencyTracker {
    static constexpr size_t kWindowSize = 100;
    std::deque<float> infer_latencies;
    std::deque<float> e2e_latencies;
    std::deque<std::chrono::steady_clock::time_point> frame_times;

    void addSample(float infer_ms, float e2e_ms, std::chrono::steady_clock::time_point tp) {
        infer_latencies.push_back(infer_ms);
        if (infer_latencies.size() > kWindowSize) infer_latencies.pop_front();

        e2e_latencies.push_back(e2e_ms);
        if (e2e_latencies.size() > kWindowSize) e2e_latencies.pop_front();

        frame_times.push_back(tp);
        if (frame_times.size() > kWindowSize) frame_times.pop_front();
    }

    float getFps() const {
        if (frame_times.size() < 2) return 30.0f;
        const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(frame_times.back() - frame_times.front()).count();
        if (dur <= 0) return 30.0f;
        return static_cast<float>(frame_times.size() - 1) * 1000.0f / static_cast<float>(dur);
    }

    float getP95Infer() const {
        if (infer_latencies.empty()) return 0.0f;
        std::vector<float> sorted(infer_latencies.begin(), infer_latencies.end());
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(sorted.size() * 0.95);
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    }

    float getP95E2E() const {
        if (e2e_latencies.empty()) return 0.0f;
        std::vector<float> sorted(e2e_latencies.begin(), e2e_latencies.end());
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(sorted.size() * 0.95);
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    }
};

static LatencyTracker g_perf_tracker;

static void print_usage(const char* prog) {
    printf("Microduck Pose QA for reCamera SG2002 (v1.0.12)\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH        Pose model (default: /userdata/local/models/microduck_pose_17kpt_384_int8.cvimodel)\n");
    printf("  -c, --confidence FLOAT  Robot detection threshold (0.05 - 0.95)\n");
    printf("  -k, --kpt-conf FLOAT    Keypoint detection threshold (0.05 - 0.95)\n");
    printf("  --mqtt-host HOST        MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT        MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC      MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --control-port PORT     HTTP control port (default: %d)\n", g_config.control_port);
    printf("  --enable-rtsp           Enable RTSP streaming (default: disabled)\n");
    printf("  --no-rtsp               Disable RTSP streaming\n");
    printf("  --no-mqtt               Disable MQTT publishing\n");
    printf("  --no-debug              Disable debug WebSocket stream\n");
    printf("  --debug-port PORT       Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose           Verbose logging\n");
    printf("  -h, --help              This message\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"confidence", required_argument, 0, 'c'},
        {"kpt-conf", required_argument, 0, 'k'},
        {"mqtt-host", required_argument, 0, 1},
        {"mqtt-port", required_argument, 0, 2},
        {"mqtt-topic", required_argument, 0, 3},
        {"no-rtsp", no_argument, 0, 4},
        {"no-mqtt", no_argument, 0, 5},
        {"no-debug", no_argument, 0, 6},
        {"debug-port", required_argument, 0, 7},
        {"enable-rtsp", no_argument, 0, 8},
        {"control-port", required_argument, 0, 9},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:k:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 'c': g_config.cli_confidence = std::stof(optarg); break;
            case 'k': g_config.cli_kpt_confidence = std::stof(optarg); break;
            case 1: g_config.mqtt_host = optarg; break;
            case 2: g_config.mqtt_port = std::stoi(optarg); break;
            case 3: g_config.mqtt_topic = optarg; break;
            case 4: g_config.enable_rtsp = false; break;
            case 5: g_config.enable_mqtt = false; break;
            case 6: g_config.enable_debug = false; break;
            case 7: g_config.debug_port = std::stoi(optarg); break;
            case 8: g_config.enable_rtsp = true; break;
            case 9: g_config.control_port = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static ha_mqtt::ClientOptions mqtt_options() {
    ha_mqtt::ClientOptions opts;
    opts.app_id        = "microduck-pose-qa";
    opts.client_id     = "recamera-microduck-pose-qa";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::BinarySensor, "robot_detected", "Robot Detected",
                     "{{ 'ON' if value_json.robot_detected else 'OFF' }}",
                     "occupancy", "", ""},
        EntityConfig{EntityType::Sensor, "duck_count", "Robot Count",
                     "{{ value_json.count }}", "", "", "measurement"},
        EntityConfig{EntityType::Sensor, "qa_status", "QA Status",
                     "{{ value_json.qa_status }}", "", "", ""},
        EntityConfig{EntityType::BinarySensor, "qa_passed", "QA Passed",
                     "{{ 'ON' if value_json.qa_passed else 'OFF' }}",
                     "", "", ""},
        EntityConfig{EntityType::Sensor, "visible_joints", "Visible Joints",
                     "{{ value_json.visible_joints }}", "", "", "measurement"},
    };
    return opts;
}

// Configure ISP exposure and AWB for natural display capture. Needs a running
// ISP, hence App::on_started(); the App then drops warmup_frames so the
// optical parameters settle before the first QA frame.
static void configure_isp() {
    ISP_EXPOSURE_ATTR_S exp_attr;
    std::memset(&exp_attr, 0, sizeof(exp_attr));
    if (CVI_ISP_GetExposureAttr(0, &exp_attr) == CVI_SUCCESS) {
        exp_attr.bByPass = CVI_FALSE;
        exp_attr.enOpType = OP_TYPE_AUTO;
        exp_attr.stAuto.stAntiflicker.bEnable = CVI_TRUE;
        exp_attr.stAuto.stAntiflicker.enFrequency = AE_FREQUENCE_60HZ;
        if (CVI_ISP_SetExposureAttr(0, &exp_attr) == CVI_SUCCESS) {
            MA_LOGI(TAG, "ISP exposure set to OP_TYPE_AUTO (anti-flicker 60Hz)");
        }
    }

    ISP_WB_ATTR_S wb_attr;
    std::memset(&wb_attr, 0, sizeof(wb_attr));
    if (CVI_ISP_GetWBAttr(0, &wb_attr) == CVI_SUCCESS) {
        wb_attr.bByPass = CVI_FALSE;
        wb_attr.enOpType = OP_TYPE_AUTO;
        if (CVI_ISP_SetWBAttr(0, &wb_attr) == CVI_SUCCESS) {
            MA_LOGI(TAG, "ISP AWB set to OP_TYPE_AUTO (dynamic white balance)");
        }
    }
}

// Action generation sampled right before each frame retrieval (App::on_before_grab):
// a frame already in flight when an action starts carries the old generation and
// is rejected by ActionStateMachine::addFrame.
static uint64_t g_pre_gen = 0;

// Payload strings must outlive the handler: the App's sinks publish them after it returns.
static std::string g_mqtt_payload;
static std::string g_debug_extra;

static void on_frame(af::App& app, const af::VideoFrame& frame, af::Result& r) {
    const uint64_t pre_gen = g_pre_gen;

    // Monotonic steady_clock timestamp of frame retrieval (ActionStateMachine compares it with the action start).
    const auto t_frame_in = std::chrono::steady_clock::now();
    const uint64_t frame_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_frame_in.time_since_epoch()).count();

    const ActionState cur_state = g_state_machine.getState();
    const std::string cur_action = g_state_machine.getCurrentActionId();
    const bool is_collecting = (cur_state == ActionState::COLLECTING);
    const bool is_pose_action = (is_collecting && (cur_action == "stand" || cur_action == "calibrate_stand"));
    const bool is_optical_action = (is_collecting && cur_action == "face_led");

    // Lock detector thresholds with snapshot if COLLECTING, or watcher otherwise
    const AppConfig cfg = is_collecting
        ? g_state_machine.getActiveConfig()
        : g_watcher->config();

    bool sys_error = false;
    const Subject* primary = nullptr;
    std::vector<Subject> active_subjects;
    float inference_ms = 0.0f;
    FaceLedFrameResult opt_res;

    if (is_pose_action) {
        // Heavy TPU Pose inference runs ONLY when a POSE action (e.g. action_id == "stand") is active!
        const float effective_conf = (g_config.cli_confidence > 0.0f) ? g_config.cli_confidence : cfg.confidence;
        const float effective_kpt_conf = (g_config.cli_kpt_confidence > 0.0f) ? g_config.cli_kpt_confidence : cfg.keypoint_confidence;
        g_detector->setThreshold(effective_conf);
        g_detector->setKeypointThreshold(effective_kpt_conf);

        ma_img_t img = af::to_ma_img(frame);
        const auto t_infer_start = std::chrono::steady_clock::now();
        try {
            primary = g_detector->detectPrimary(&img);
            if (!g_detector->lastInferenceOk()) {
                sys_error = true;
                g_state_machine.setSysError("TPU_INFERENCE_FATAL_ERROR", pre_gen);
            }
        } catch (const std::exception& e) {
            MA_LOGE(TAG, "TPU inference exception: %s", e.what());
            sys_error = true;
            g_state_machine.setSysError("TPU_INFERENCE_FATAL_ERROR", pre_gen);
        }
        inference_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_infer_start).count();

        if (primary != nullptr) {
            active_subjects.push_back(*primary);
        }
    } else if (is_optical_action) {
        // Face LED Optical Inspection Operator (action_id == "face_led")
        // No TPU inference! Uses OpenCV optical analysis
        const auto t_opt_start = std::chrono::steady_clock::now();
        if (g_state_machine.getLedExpectation().active) {
            // single-LED check only; the legacy whole-ROI metrics are skipped (cost)
            opt_res.slots = inspectFaceLedSlots(frame.planes[0].vir, frame.width, frame.height, cfg);
            opt_res.metrics.lit_pixels = opt_res.slots.valid ? 1 : 0;
        } else {
            opt_res = inspectFaceLed(frame.planes[0].vir, frame.width, frame.height, cfg);
        }
        inference_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_opt_start).count();
    }
    // else: IDLE or non-active state -- no TPU inference; the App keeps frame
    // retrieval and the debug stream alive.

    PayloadContext ctx;
    ctx.timestamp_ms = r.wall_ms;
    ctx.frame_id = g_frame_id;
    ctx.inference_time_ms = inference_ms;
    ctx.fps = g_perf_tracker.getFps();
    ctx.action_id = cur_action;
    ctx.infer_w = g_config.inference_width;
    ctx.infer_h = g_config.inference_height;
    ctx.stream_w = g_config.stream_width;
    ctx.stream_h = g_config.stream_height;
    ctx.config = cfg;

    FrameQAResult qa;

    if (is_pose_action) {
        ctx.robot_detected = (!sys_error && primary != nullptr);
        ctx.duck_count = ctx.robot_detected ? 1 : 0;
        if (sys_error) {
            qa.qa_status = "SYS_ERROR";
            qa.qa_passed = false;
            qa.visible_joints = 0;
        } else if (ctx.robot_detected && primary != nullptr) {
            qa = evaluateQA(ctx, primary->pose);
        } else {
            qa.qa_status = "NO_ROBOT";
            qa.qa_passed = false;
            qa.visible_joints = 0;
        }
    } else if (is_optical_action) {
        ctx.robot_detected = (opt_res.metrics.lit_pixels > 0);
        ctx.duck_count = ctx.robot_detected ? 1 : 0;
        ctx.optical = opt_res.metrics;
        qa.qa_status = opt_res.ok ? "PASS" : "FAIL_OPTICAL";
        qa.qa_passed = opt_res.ok;
        qa.visible_joints = 0;
    } else {
        ctx.robot_detected = false;
        ctx.duck_count = 0;
        qa.qa_status = (cur_state == ActionState::IDLE) ? "IDLE" : "READY";
        qa.qa_passed = false;
        qa.visible_joints = 0;
    }

    // Advance action state machine with generation & monotonic steady_clock timestamp validation
    g_state_machine.checkTimeout();
    if (is_collecting) {
        g_state_machine.addFrame(g_frame_id, frame_time_ms, pre_gen, ctx.robot_detected, ctx.duck_count, qa, opt_res);
    }

    // Debug /results (8001): the App's debug sink publishes the envelope with
    // these fields; no overlay boxes. Built only while a /results client listens.
    r.seq = g_frame_id;
    r.inference_ms = inference_ms;
    g_debug_extra.clear();
    if (debug_stream_results_client_count() > 0) {
        g_debug_extra = buildDebugExtraJson(ctx, active_subjects, qa, g_state_machine);
    }
    r.debug_extra = g_debug_extra;

    // The broadcast is considered delivered once this frame's debug telemetry
    // is built -- the same point as before, independent of client count.
    uint64_t bcast_gen = 0;
    if (g_state_machine.isBroadcastPending(bcast_gen)) {
        g_state_machine.markBroadcastCompleted(bcast_gen);
    }

    // MQTT telemetry, published by the App's MQTT sink.
    if (app.mqtt() != nullptr) {
        g_mqtt_payload = buildResultJson(ctx, active_subjects, qa, g_state_machine);
        r.json = g_mqtt_payload;
    }

    // T_emit: results are complete and handed to the sinks right after return.
    const float e2e_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_frame_in).count();
    g_perf_tracker.addSample(inference_ms, e2e_ms, t_frame_in);

    if (g_frame_id % 150 == 0) {
        MA_LOGI(TAG, "Stats [frame %u]: target=%s, fps=%.1f, inf_p95=%.1fms, e2e_p95=%.1fms, qa=%s (%s), state=%s (action=%s)",
                g_frame_id, ctx.robot_detected ? "Locked" : (is_collecting ? "Searching" : "Idle"),
                g_perf_tracker.getFps(), g_perf_tracker.getP95Infer(), g_perf_tracker.getP95E2E(),
                qa.qa_status.c_str(), qa.qa_passed ? "PASS" : "WARN",
                g_state_machine.getStateString().c_str(), cur_action.c_str());
    } else if (g_config.verbose && ctx.robot_detected && (g_frame_id % 30 == 0)) {
        MA_LOGI(TAG, "Frame %u: locked, score=%.2f, joints=%d, diff=%.4f, qa=%s (%s), inf=%.1fms, e2e=%.1fms",
                g_frame_id, primary ? primary->score : 0.0f, qa.visible_joints,
                qa.metrics.knee_height_diff_norm,
                qa.qa_status.c_str(), qa.qa_passed ? "PASS" : "WARN",
                inference_ms, e2e_ms);
    }

    if (g_watcher->poll()) {
        const AppConfig fresh = g_watcher->config();
        MA_LOGI(TAG, "Config reloaded: conf=%.2f kpt_conf=%.2f gear_tol=%.1f knee_tol=%.4f ver=%s",
                fresh.confidence, fresh.keypoint_confidence, fresh.gear_tooth_tolerance_deg,
                fresh.bilateral_height_tolerance_norm, fresh.config_version.c_str());
        g_detector->setThreshold(fresh.confidence);
        g_detector->setKeypointThreshold(fresh.keypoint_confidence);
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    g_watcher = new ConfigWatcher(APP_CONFIG_PATH);
    const AppConfig& cfg = g_watcher->loadInitial();

    MA_LOGI(TAG, "Starting Microduck Pose QA Service (v1.0.12)");

    auto release = [] {
        delete g_detector;
        g_detector = nullptr;
        delete g_watcher;
        g_watcher = nullptr;
    };

    if (g_config.model_path.empty()) {
        g_config.model_path = findPoseModel();
        if (g_config.model_path.empty()) {
            MA_LOGE(TAG, "No pose model found. Looked in:");
            for (const char* p : MODEL_CANDIDATES) {
                MA_LOGE(TAG, "  %s", p);
            }
            release();
            return 1;
        }
    }
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());

    g_detector = new PoseDetector();
    if (!g_detector->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize pose detector with %s", g_config.model_path.c_str());
        release();
        return 1;
    }

    float conf = (g_config.cli_confidence > 0.0f) ? g_config.cli_confidence : cfg.confidence;
    float kpt_conf = (g_config.cli_kpt_confidence > 0.0f) ? g_config.cli_kpt_confidence : cfg.keypoint_confidence;
    g_detector->setThreshold(conf);
    g_detector->setKeypointThreshold(kpt_conf);

    g_config.inference_width = g_detector->inputWidth();
    g_config.inference_height = g_detector->inputHeight();

    // Independent HTTP control service (8082). Fatal on failure, as before.
    if (!g_http_control.start(g_config.control_port, &g_state_machine, g_watcher)) {
        MA_LOGE(TAG, "Failed to start HTTP control service on port %d", g_config.control_port);
        release();
        return 1;
    }

    af::CameraOptions cam;
    cam.width = static_cast<uint16_t>(g_config.inference_width);
    cam.height = static_cast<uint16_t>(g_config.inference_height);
    cam.fps = static_cast<uint8_t>(g_config.inference_fps);
    cam.warmup_frames = 5;  // drain the ISP pipeline after configure_isp()

    af::RtspOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);
    stream.gop = 30;
    stream.bitrate_kbps = 1200;
    stream.max_bitrate_kbps = 1500;

    af::App app("microduck-pose-qa");
    app.with_camera(cam).with_stream(stream);
    if (g_config.enable_rtsp) app.with_rtsp(stream);
    if (g_config.enable_debug) app.with_debug_ws(g_config.debug_port);
    if (g_config.enable_mqtt) app.with_mqtt(mqtt_options());

    MA_LOGI(TAG, "Camera %dx%d @ %dfps, ISP anti-flicker target exposure=%u us",
            g_config.inference_width, g_config.inference_height, g_config.inference_fps, cfg.exposure_us);
    MA_LOGI(TAG, "HTTP Control: http://<device_ip>:%d/api/control", g_config.control_port);
    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    app.on_started(configure_isp)
       .on_before_grab([] { g_pre_gen = g_state_machine.getActionGeneration(); })
       .on_idle([] { g_state_machine.checkTimeout(); })  // starved: still finalize timed-out actions
       .on_frame([&app](const af::VideoFrame& f, af::Result& r) { on_frame(app, f, r); });
    const int rc = app.run();

    g_http_control.stop();
    release();
    MA_LOGI(TAG, "Microduck Pose QA daemon terminated");
    return rc;
}
