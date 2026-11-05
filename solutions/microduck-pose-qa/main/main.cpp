// microduck-pose-qa -- Microduck 17-Keypoint Robot Pose Estimation & QA Daemon
//
// Pipeline: camera -> Microduck Pose (SG2002 TPU) -> 17 keypoints ->
// kinematic QA -> MQTT / debug WebSocket / RTSP / HTTP Control (8082)

#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "rtsp_server.h"

extern "C" {
#include <cvi_comm_isp.h>
#include <cvi_isp.h>
#include <cvi_ae.h>
#include <cvi_awb.h>
}

#include "app_config.h"
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

static std::atomic<bool> g_running(true);
static PoseDetector* g_detector = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static Camera* g_camera = nullptr;
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

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Microduck Pose QA for reCamera SG2002 (v1.0.6)\n\n");
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

static bool init_camera() {
    Device* device = Device::getInstance();
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() != ma::Sensor::Type::kCamera) continue;

        g_camera = static_cast<Camera*>(sensor);
        g_camera->init(0);

        Camera::CtrlValue value;
        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

        value.i32 = g_config.inference_fps;
        g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

        value.u16s[0] = static_cast<uint16_t>(g_config.inference_width);
        value.u16s[1] = static_cast<uint16_t>(g_config.inference_height);
        g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

        const AppConfig& initial_cfg = g_watcher->config();
        MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps, ISP anti-flicker target exposure=%u us)",
                g_config.inference_width, g_config.inference_height, g_config.inference_fps,
                initial_cfg.exposure_us);
        return true;
    }
    MA_LOGE(TAG, "No camera found");
    return false;
}

static bool init_video_streaming() {
    if (initVideo() != 0) {
        MA_LOGE(TAG, "Failed to initialize video subsystem");
        return false;
    }

    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;

    setupVideo(VIDEO_CH2, &stream_param);

    APP_PARAM_VENC_CTX_S* venc = app_ipcam_Venc_Param_Get();
    if (venc && VIDEO_CH2 < venc->s32VencChnCnt) {
        venc->astVencChnCfg[VIDEO_CH2].u32Gop = 30; // 2 seconds GOP at 15fps
        venc->astVencChnCfg[VIDEO_CH2].u32BitRate = 1200; // 1200 kbps
        venc->astVencChnCfg[VIDEO_CH2].u32MaxBitRate = 1500; // max 1500 kbps
    }

    if (g_config.enable_rtsp) {
        registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
        initRtsp((0x01 << VIDEO_CH2));
        MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps, GOP=%u, bitRate=%u)",
                g_config.stream_width, g_config.stream_height, g_config.stream_fps,
                venc ? venc->astVencChnCfg[VIDEO_CH2].u32Gop : 0,
                venc ? venc->astVencChnCfg[VIDEO_CH2].u32BitRate : 0);
    } else {
        MA_LOGI(TAG, "RTSP streaming disabled (VIDEO_CH2 reserved for debug_stream, GOP=%u, bitRate=%u)",
                venc ? venc->astVencChnCfg[VIDEO_CH2].u32Gop : 0,
                venc ? venc->astVencChnCfg[VIDEO_CH2].u32BitRate : 0);
    }
    return true;
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

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

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    g_http_control.stop();
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    if (g_config.enable_rtsp) { deinitRtsp(); }
    deinitVideo();
    if (g_mqtt) { g_mqtt->deinit(); delete g_mqtt; g_mqtt = nullptr; }
    if (g_detector) { delete g_detector; g_detector = nullptr; }
    if (g_watcher) { delete g_watcher; g_watcher = nullptr; }
    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    // Read generation token BEFORE retrieveFrame
    const uint64_t pre_gen = g_state_machine.getActionGeneration();

    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        // Starved or disconnected: autonomously check and finalize timeout
        g_state_machine.checkTimeout();
        uint64_t starved_gen = 0;
        const bool bcast_pending = g_state_machine.isBroadcastPending(starved_gen);
        static auto s_last_starved_replay = std::chrono::steady_clock::time_point{};
        const auto now_mono = std::chrono::steady_clock::now();
        const bool periodic_replay_due = g_state_machine.hasReport() &&
            (std::chrono::duration_cast<std::chrono::milliseconds>(now_mono - s_last_starved_replay).count() >= 200);

        if ((bcast_pending || periodic_replay_due) && g_config.enable_debug && debug_stream_results_client_count() > 0) {
            const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            PayloadContext ctx;
            ctx.timestamp_ms = static_cast<uint64_t>(wall_ms);
            ctx.frame_id = g_frame_id++;
            ctx.fps = g_perf_tracker.getFps();
            ctx.robot_detected = false;
            ctx.duck_count = 0;
            ctx.config = g_state_machine.getActiveConfig();
            FrameQAResult empty_qa;
            empty_qa.qa_status = "NO_ROBOT";
            empty_qa.qa_passed = false;
            const std::vector<debug_stream_box_t> no_boxes;
            const std::string json_str = debug_stream_build_results(
                ctx.timestamp_ms, ctx.frame_id, 0.0f,
                g_config.stream_width, g_config.stream_height, no_boxes, nullptr,
                buildDebugExtraJson(ctx, {}, empty_qa, g_state_machine));
            if (debug_stream_publish_result(json_str.c_str(), json_str.size()) == 0 &&
                debug_stream_results_client_count() > 0) {
                if (bcast_pending) {
                    g_state_machine.markBroadcastCompleted(starved_gen);
                }
                s_last_starved_replay = now_mono;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    // Capture monotonic steady_clock timestamp immediately upon successful frame retrieval
    const auto t_frame_in = std::chrono::steady_clock::now();
    const uint64_t frame_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_frame_in.time_since_epoch()).count();

    const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Lock detector thresholds with snapshot if COLLECTING, or watcher otherwise
    const AppConfig cfg = (g_state_machine.getState() == ActionState::COLLECTING)
        ? g_state_machine.getActiveConfig()
        : g_watcher->config();
    g_detector->setThreshold(cfg.confidence);
    g_detector->setKeypointThreshold(cfg.keypoint_confidence);

    const auto t_infer_start = std::chrono::steady_clock::now();

    bool sys_error = false;
    const Subject* primary = nullptr;
    try {
        primary = g_detector->detectPrimary(&frame);
        if (!g_detector->lastInferenceOk()) {
            sys_error = true;
            g_state_machine.setSysError("TPU_INFERENCE_FATAL_ERROR", pre_gen);
        }
    } catch (const std::exception& e) {
        MA_LOGE(TAG, "TPU inference exception: %s", e.what());
        sys_error = true;
        g_state_machine.setSysError("TPU_INFERENCE_FATAL_ERROR", pre_gen);
    }

    std::vector<Subject> active_subjects;
    if (primary != nullptr) {
        active_subjects.push_back(*primary);
    }

    debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    g_camera->returnFrame(frame);

    const auto t_infer_end = std::chrono::steady_clock::now();
    const float inference_ms = std::chrono::duration<float, std::milli>(t_infer_end - t_infer_start).count();

    PayloadContext ctx;
    ctx.timestamp_ms = static_cast<uint64_t>(wall_ms);
    ctx.frame_id = g_frame_id;
    ctx.inference_time_ms = inference_ms;
    ctx.fps = g_perf_tracker.getFps();
    ctx.robot_detected = (!sys_error && primary != nullptr);
    ctx.duck_count = ctx.robot_detected ? 1 : 0;
    ctx.infer_w = g_config.inference_width;
    ctx.infer_h = g_config.inference_height;
    ctx.stream_w = g_config.stream_width;
    ctx.stream_h = g_config.stream_height;
    ctx.config = cfg;

    FrameQAResult qa;
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

    // Advance action state machine with generation & monotonic steady_clock timestamp validation
    g_state_machine.checkTimeout();
    g_state_machine.addFrame(g_frame_id, frame_time_ms, pre_gen, ctx.robot_detected, ctx.duck_count, qa);

    uint64_t post_frame_bcast_gen = 0;
    const bool post_frame_bcast_pending = g_state_machine.isBroadcastPending(post_frame_bcast_gen);

    // Publish WebSocket debug telemetry (8001)
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        const std::vector<debug_stream_box_t> no_boxes;
        const std::string json = debug_stream_build_results(
            ctx.timestamp_ms, ctx.frame_id, ctx.inference_time_ms,
            g_config.stream_width, g_config.stream_height, no_boxes, nullptr,
            buildDebugExtraJson(ctx, active_subjects, qa, g_state_machine));
        if (debug_stream_publish_result(json.c_str(), json.size()) == 0 &&
            debug_stream_results_client_count() > 0) {
            if (post_frame_bcast_pending) {
                g_state_machine.markBroadcastCompleted(post_frame_bcast_gen);
            }
        }
    }

    // Publish MQTT telemetry
    if (g_config.enable_mqtt && g_mqtt) {
        const std::string payload = buildResultJson(ctx, active_subjects, qa, g_state_machine);
        g_mqtt->publishResultsJson(payload);
    }

    // T_emit: exactly when results are enqueued
    const auto t_emit = std::chrono::steady_clock::now();
    const float e2e_ms = std::chrono::duration<float, std::milli>(t_emit - t_frame_in).count();
    g_perf_tracker.addSample(inference_ms, e2e_ms, t_frame_in);

    if (g_frame_id % 150 == 0) {
        MA_LOGI(TAG, "Stats [frame %u]: target=%s, fps=%.1f, inf_p95=%.1fms, e2e_p95=%.1fms, qa=%s (%s), state=%s",
                g_frame_id, ctx.robot_detected ? "Locked" : "Searching",
                g_perf_tracker.getFps(), g_perf_tracker.getP95Infer(), g_perf_tracker.getP95E2E(),
                qa.qa_status.c_str(), qa.qa_passed ? "PASS" : "WARN",
                g_state_machine.getStateString().c_str());
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

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    g_watcher = new ConfigWatcher(APP_CONFIG_PATH);
    const AppConfig& cfg = g_watcher->loadInitial();

    MA_LOGI(TAG, "Starting Microduck Pose QA Service (v1.0.6)");

    if (g_config.model_path.empty()) {
        g_config.model_path = findPoseModel();
        if (g_config.model_path.empty()) {
            MA_LOGE(TAG, "No pose model found. Looked in:");
            for (const char* p : MODEL_CANDIDATES) {
                MA_LOGE(TAG, "  %s", p);
            }
            cleanup();
            return 1;
        }
    }
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());

    g_detector = new PoseDetector();
    if (!g_detector->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize pose detector with %s", g_config.model_path.c_str());
        cleanup();
        return 1;
    }

    float conf = (g_config.cli_confidence > 0.0f) ? g_config.cli_confidence : cfg.confidence;
    float kpt_conf = (g_config.cli_kpt_confidence > 0.0f) ? g_config.cli_kpt_confidence : cfg.keypoint_confidence;
    g_detector->setThreshold(conf);
    g_detector->setKeypointThreshold(kpt_conf);

    g_config.inference_width = g_detector->inputWidth();
    g_config.inference_height = g_detector->inputHeight();

    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt()) { cleanup(); return 1; }

    // Start independent HTTP control service on port 8082
    if (!g_http_control.start(g_config.control_port, &g_state_machine, g_watcher)) {
        MA_LOGE(TAG, "Failed to start HTTP control service on port %d", g_config.control_port);
        cleanup();
        return 1;
    }

    // Start camera stream in kRefreshOnReturn mode
    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // Apply strict ISP Exposure and AWB hardware gate checks AFTER startStream()
    const AppConfig& initial_cfg = g_watcher->config();
    ISP_EXPOSURE_ATTR_S exp_attr;
    std::memset(&exp_attr, 0, sizeof(exp_attr));
    CVI_S32 exp_get_ret = CVI_ISP_GetExposureAttr(0, &exp_attr);
    if (exp_get_ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "FATAL: CVI_ISP_GetExposureAttr failed with error 0x%x", exp_get_ret);
        cleanup();
        return 1;
    }
    exp_attr.bByPass = CVI_FALSE;
    exp_attr.enOpType = OP_TYPE_MANUAL;
    exp_attr.stManual.enExpTimeOpType = OP_TYPE_MANUAL;
    exp_attr.stManual.u32ExpTime = initial_cfg.exposure_us;
    CVI_S32 exp_set_ret = CVI_ISP_SetExposureAttr(0, &exp_attr);
    if (exp_set_ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "FATAL: CVI_ISP_SetExposureAttr failed with error 0x%x", exp_set_ret);
        cleanup();
        return 1;
    }

    ISP_EXPOSURE_ATTR_S verify_exp;
    std::memset(&verify_exp, 0, sizeof(verify_exp));
    if (CVI_ISP_GetExposureAttr(0, &verify_exp) != CVI_SUCCESS) {
        MA_LOGE(TAG, "FATAL: CVI_ISP_GetExposureAttr readback failed");
        cleanup();
        return 1;
    }
    const int64_t exp_diff = std::abs(static_cast<int64_t>(verify_exp.stManual.u32ExpTime) - static_cast<int64_t>(initial_cfg.exposure_us));
    if (verify_exp.enOpType != OP_TYPE_MANUAL || exp_diff > 1000) {
        MA_LOGE(TAG, "FATAL: ISP exposure gate readback mismatch: target=%u us, actual=%u us (diff=%ld us > 1000 us line tolerance), opType=%d",
                initial_cfg.exposure_us, verify_exp.stManual.u32ExpTime, static_cast<long>(exp_diff), verify_exp.enOpType);
        cleanup();
        return 1;
    }
    MA_LOGI(TAG, "ISP exposure hardware gate verified: %u us (OP_TYPE_MANUAL, diff=%ld us)",
            verify_exp.stManual.u32ExpTime, static_cast<long>(exp_diff));

    ISP_WB_ATTR_S wb_attr;
    std::memset(&wb_attr, 0, sizeof(wb_attr));
    CVI_S32 wb_get_ret = CVI_ISP_GetWBAttr(0, &wb_attr);
    if (wb_get_ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "FATAL: CVI_ISP_GetWBAttr failed with error 0x%x", wb_get_ret);
        cleanup();
        return 1;
    }
    wb_attr.bByPass = CVI_FALSE;
    wb_attr.enOpType = OP_TYPE_MANUAL;  // Strict AWB lock per spec: fixed unity gains (1024 = 1.0x) disable auto color drift
    wb_attr.stManual.u16Rgain = 1024;
    wb_attr.stManual.u16Grgain = 1024;
    wb_attr.stManual.u16Gbgain = 1024;
    wb_attr.stManual.u16Bgain = 1024;
    CVI_S32 wb_set_ret = CVI_ISP_SetWBAttr(0, &wb_attr);
    if (wb_set_ret != CVI_SUCCESS) {
        MA_LOGE(TAG, "FATAL: CVI_ISP_SetWBAttr failed with error 0x%x", wb_set_ret);
        cleanup();
        return 1;
    }

    ISP_WB_ATTR_S verify_wb;
    std::memset(&verify_wb, 0, sizeof(verify_wb));
    if (CVI_ISP_GetWBAttr(0, &verify_wb) != CVI_SUCCESS ||
        verify_wb.enOpType != OP_TYPE_MANUAL ||
        verify_wb.stManual.u16Rgain != 1024 ||
        verify_wb.stManual.u16Grgain != 1024 ||
        verify_wb.stManual.u16Gbgain != 1024 ||
        verify_wb.stManual.u16Bgain != 1024) {
        MA_LOGE(TAG, "FATAL: ISP AWB lock readback mismatch: opType=%d, gains=(%u,%u,%u,%u, expected 1024)",
                verify_wb.enOpType,
                verify_wb.stManual.u16Rgain, verify_wb.stManual.u16Grgain,
                verify_wb.stManual.u16Gbgain, verify_wb.stManual.u16Bgain);
        cleanup();
        return 1;
    }
    MA_LOGI(TAG, "ISP AWB hardware gate verified: OP_TYPE_MANUAL (locked gains 1024/1024/1024/1024)");

    // Flush initial 5 frames to drain ISP pipeline buffer and ensure optical parameters are settled
    for (int i = 0; i < 5; ++i) {
        ma_img_t warmup_frame;
        if (g_camera->retrieveFrame(warmup_frame, MA_PIXEL_FORMAT_RGB888) == MA_OK) {
            g_camera->returnFrame(warmup_frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    MA_LOGI(TAG, "Microduck Pose QA daemon running");
    if (g_config.enable_rtsp) {
        MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    }
    MA_LOGI(TAG, "Debug WS: ws://<device_ip>:%d/ and /results", g_config.debug_port);
    MA_LOGI(TAG, "HTTP Control: http://<device_ip>:%d/api/control", g_config.control_port);
    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Microduck Pose QA daemon terminated");
    return 0;
}
