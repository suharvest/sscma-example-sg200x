// fall-detection -- temporal multi-feature fall detection on reCamera.
//
// Pipeline: camera -> YOLO11-Pose (TPU) -> temporal geometric features ->
// tiny learned temporal gate -> FallDetector state machine -> MQTT / debug
// WebSocket, with the scene going out over RTSP untouched. One primary subject
// is associated by box overlap so nearby people cannot splice pose histories.

#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "rtsp_server.h"

#include "app_config.h"
#include "fall_detector.h"
#include "pose_detector.h"
#include "result_payload.h"
#include "temporal_classifier.h"

using namespace ma;
using namespace fall;

#define TAG "fall-detection"

static struct {
    // Empty = search the candidate list below. An explicit -m / MODEL_PATH
    // wins and is used verbatim: if an operator names a file, failing loudly
    // beats silently running a different model than the one they asked for.
    std::string model_path;

    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/fall-detection/results";

    int inference_width = 640;
    int inference_height = 640;
    int inference_fps = 15;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    bool enable_debug = true;
    int debug_port = 8001;
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;

    // Offline evaluator: a raw RGB888 file containing contiguous frames.
    std::string offline_rgb_path;
    int offline_width = 640;
    int offline_height = 640;
    int offline_fps = 15;

} g_config;

// Where to look for the pose model, in order.
//
// This app is the only one in the gallery that ships no model of its own: it
// uses the YOLO11n-Pose that comes with the reCamera console, which is the
// official Ultralytics build and would only be duplicated by packaging a
// second copy. The cost of that choice is this list -- a device whose console
// predates the model, or one where it was installed separately, must still
// find it. /userdata is checked too because that is where both the SenseCraft
// deployer and the console's cloud install put model files.
static const char* MODEL_CANDIDATES[] = {
    "/usr/share/supervisor/models/yolo11n_pose_cv181x_int8.cvimodel",
    "/userdata/local/models/yolo11n_pose_cv181x_int8.cvimodel",
};

// First candidate that exists, or empty when none do.
static std::string findPoseModel() {
    for (const char* p : MODEL_CANDIDATES) {
        struct stat st;
        if (::stat(p, &st) == 0 && S_ISREG(st.st_mode)) {
            return p;
        }
    }
    return std::string();
}

static const char* APP_CONFIG_PATH = "/userdata/local/apps/fall-detection.config.json";

static std::atomic<bool> g_running(true);
static PoseDetector* g_pose_detector = nullptr;
static std::unique_ptr<FallDetector> g_detector_state;
static std::unique_ptr<TemporalClassifier> g_temporal_classifier;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static Camera* g_camera = nullptr;
static ConfigWatcher* g_watcher = nullptr;
static uint32_t g_frame_id = 0;
static std::chrono::steady_clock::time_point g_start_time;
static bool g_video_started = false;
static bool g_debug_started = false;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Fall Detection for reCamera -- temporal multi-feature pose detector\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH        Pose model (default: %s)\n", g_config.model_path.c_str());
    printf("  --mqtt-host HOST        MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT        MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC      MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-rtsp               Disable RTSP streaming\n");
    printf("  --no-mqtt               Disable MQTT publishing\n");
    printf("  --no-debug              Disable debug WebSocket stream\n");
    printf("  --debug-port PORT       Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  --offline-rgb PATH      Evaluate contiguous RGB888 frames, no camera/services\n");
    printf("  --offline-width N       Offline frame width (default: %d)\n", g_config.offline_width);
    printf("  --offline-height N      Offline frame height (default: %d)\n", g_config.offline_height);
    printf("  --offline-fps N         Offline frame rate for temporal timing (default: %d)\n", g_config.offline_fps);
    printf("  -v, --verbose           Verbose logging\n");
    printf("  -h, --help              This message\n");
    printf("\nThresholds live in %s\n", APP_CONFIG_PATH);
    printf("RTSP: rtsp://<device_ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"mqtt-host", required_argument, 0, 1},
        {"mqtt-port", required_argument, 0, 2},
        {"mqtt-topic", required_argument, 0, 3},
        {"no-rtsp", no_argument, 0, 4},
        {"no-mqtt", no_argument, 0, 5},
        {"no-debug", no_argument, 0, 6},
        {"debug-port", required_argument, 0, 7},
        {"offline-rgb", required_argument, 0, 8},
        {"offline-width", required_argument, 0, 9},
        {"offline-height", required_argument, 0, 10},
        {"offline-fps", required_argument, 0, 11},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 1: g_config.mqtt_host = optarg; break;
            case 2: g_config.mqtt_port = std::stoi(optarg); break;
            case 3: g_config.mqtt_topic = optarg; break;
            case 4: g_config.enable_rtsp = false; break;
            case 5: g_config.enable_mqtt = false; break;
            case 6: g_config.enable_debug = false; break;
            case 7: g_config.debug_port = std::stoi(optarg); break;
            case 8: g_config.offline_rgb_path = optarg; break;
            case 9: g_config.offline_width = std::stoi(optarg); break;
            case 10: g_config.offline_height = std::stoi(optarg); break;
            case 11: g_config.offline_fps = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    if (!g_config.offline_rgb_path.empty() &&
        (g_config.offline_width <= 0 || g_config.offline_height <= 0 ||
         g_config.offline_width > 65535 || g_config.offline_height > 65535 ||
         g_config.offline_fps <= 0)) {
        fprintf(stderr, "Offline width/height/fps invalid (width/height 1..65535; fps > 0)\n");
        return false;
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

        value.u16s[0] = g_config.inference_width;
        value.u16s[1] = g_config.inference_height;
        g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

        MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps)",
                g_config.inference_width, g_config.inference_height, g_config.inference_fps);
        return true;
    }
    MA_LOGE(TAG, "No camera found");
    return false;
}

static bool init_video_streaming() {
    if (!g_config.enable_rtsp) {
        MA_LOGI(TAG, "RTSP streaming disabled");
        return true;
    }
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
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));
    g_video_started = true;

    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps)",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps);
    return true;
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "fall-detection";
    opts.client_id     = "recamera-fall-detection";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant discovery. Field names must match buildResultJson().
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::BinarySensor, "fall_detected", "Fall Detected",
                     "{{ 'ON' if value_json.fall_detected else 'OFF' }}", "problem", "", ""},
        EntityConfig{EntityType::Sensor, "fall_state", "Fall State",
                     "{{ value_json.state }}", "", "", ""},
        EntityConfig{EntityType::Sensor, "event_id", "Fall Event ID",
                     "{{ value_json.event_id }}", "", "", ""},
        EntityConfig{EntityType::Sensor, "person_count", "Person Count",
                     "{{ value_json.person_count }}", "", "", "measurement"},
        EntityConfig{EntityType::BinarySensor, "person_present", "Person Present",
                     "{{ 'ON' if value_json.person_detected else 'OFF' }}",
                     "occupancy", "", ""},
    };

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) g_camera->stopStream();
    if (g_debug_started) { debug_stream_destroy(); g_debug_started = false; }
    if (g_video_started) { deinitRtsp(); deinitVideo(); g_video_started = false; }
    if (g_mqtt) { g_mqtt->deinit(); delete g_mqtt; g_mqtt = nullptr; }
    g_detector_state.reset();
    g_temporal_classifier.reset();
    if (g_pose_detector) { delete g_pose_detector; g_pose_detector = nullptr; }
    if (g_watcher) { delete g_watcher; g_watcher = nullptr; }
    MA_LOGI(TAG, "Cleanup completed");
}

static bool midpoint(const Pose& pose, Joint a, Joint b, Point2f& out) {
    const bool va = pose.visible(a);
    const bool vb = pose.visible(b);
    if (!va && !vb) return false;
    if (va && vb) {
        out.x = (pose.at(a).x + pose.at(b).x) * 0.5f;
        out.y = (pose.at(a).y + pose.at(b).y) * 0.5f;
    } else {
        out = va ? pose.at(a) : pose.at(b);
    }
    return true;
}

static FallObservation makeObservation(const Subject* subject, double now_sec,
                                       int infer_h) {
    FallObservation observation;
    observation.timestamp_sec = now_sec;
    if (subject == nullptr || subject->pose.empty() || infer_h <= 0 || subject->box.h <= 1e-4f) {
        return observation;
    }

    Point2f hips;
    Point2f shoulders;
    if (!midpoint(subject->pose, Joint::LeftHip, Joint::RightHip, hips) ||
        !midpoint(subject->pose, Joint::LeftShoulder, Joint::RightShoulder, shoulders)) {
        return observation;
    }

    const float dy = hips.y - shoulders.y;
    const float dx = hips.x - shoulders.x;
    const float torso_angle = std::atan2(std::fabs(dx), std::fabs(dy)) * 180.0f / 3.14159265358979323846f;
    if (!std::isfinite(torso_angle)) return observation;

    observation.valid = true;
    observation.hip_y = hips.y / static_cast<float>(infer_h);
    observation.torso_angle_deg = torso_angle;
    observation.bbox_aspect_ratio = subject->box.w / subject->box.h;
    observation.person_score = subject->score;
    return observation;
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const double now_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time).count() / 1000.0;

    const Subject* subject = g_pose_detector->detectPrimary(&frame);

    // Offer the raw frame for /snapshot.jpg before returnFrame() invalidates
    // frame.data. Cheap (one atomic load) unless a client asked recently.
    debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    g_camera->returnFrame(frame);

    const auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    const int feature_height = g_pose_detector->inputHeight() > 0
        ? g_pose_detector->inputHeight() : g_config.inference_height;
    FallObservation observation = makeObservation(subject, now_sec, feature_height);
    const int feature_width = g_pose_detector->inputWidth() > 0
        ? g_pose_detector->inputWidth() : g_config.inference_width;
    const TemporalPrediction temporal = g_temporal_classifier->update(
        makeTemporalFrame(subject ? &subject->pose : nullptr, observation,
                          feature_width, feature_height), now_sec);
    observation.temporal_available = true;
    observation.temporal_positive = temporal.positive;
    observation.temporal_probability = temporal.probability;
    FallOutput output = g_detector_state->update(observation);

    PayloadContext ctx;
    ctx.timestamp_ms = static_cast<uint64_t>(timestamp_ms);
    ctx.frame_id = g_frame_id;
    ctx.inference_time_ms = static_cast<float>(inference_ms);
    ctx.person_detected = (subject != nullptr);
    ctx.person_count = subject != nullptr ? 1 : 0;
    ctx.fallen_count = output.fall_detected ? 1 : 0;
    ctx.infer_w = g_pose_detector->inputWidth() > 0 ? g_pose_detector->inputWidth() : g_config.inference_width;
    ctx.infer_h = feature_height;
    ctx.stream_w = g_config.stream_width;
    ctx.stream_h = g_config.stream_height;

    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        // No boxes: a rectangle round the athlete carries no information the
        // skeleton does not, and the label riding on it can slide off the
        // bottom of frame exactly when it matters. The
        // skeleton and a corner-pinned card go out in the extra members.
        const std::vector<debug_stream_box_t> no_boxes;
        const std::string json = debug_stream_build_results(
            ctx.timestamp_ms, ctx.frame_id, ctx.inference_time_ms,
            g_config.stream_width, g_config.stream_height, no_boxes, nullptr,
            buildDebugExtraJson(ctx, output, observation, subject ? &subject->pose : nullptr));
        debug_stream_publish_result(json.c_str(), json.size());
    }

    if (g_config.enable_mqtt && g_mqtt) {
        const std::string payload = buildResultJson(ctx, output, observation, subject ? &subject->pose : nullptr);
        g_mqtt->publishResultsJson(payload);
    }

    if (output.fall_event) {
        MA_LOGI(TAG, "Fall event id=%llu (temporal=%.3f hip_speed=%.3f torso=%.1f aspect=%.2f)",
                static_cast<unsigned long long>(output.event_id),
                output.diagnostics.temporal_probability, output.diagnostics.hip_drop_speed,
                output.diagnostics.torso_angle_deg,
                output.diagnostics.bbox_aspect_ratio);
    } else if (g_config.verbose) {
        MA_LOGI(TAG, "Frame %u: %s state=%s hip_speed=%.3f torso=%.1f aspect=%.2f inference=%lldms",
                g_frame_id, ctx.person_detected ? "person" : "no person",
                fallStateName(output.state), output.diagnostics.hip_drop_speed,
                output.diagnostics.torso_angle_deg, output.diagnostics.bbox_aspect_ratio,
                inference_ms);
    }

    // Console setConfig restarts the app; this also handles out-of-band edits
    // (SSH/Node-RED) without a restart.
    if (g_watcher->poll()) {
        const AppConfig& fresh = g_watcher->config();
        MA_LOGI(TAG, "Config reloaded: torso=%.1f aspect=%.2f confirm=%.2fs",
                fresh.detector.torso_angle_threshold_deg,
                fresh.detector.bbox_aspect_ratio_threshold,
                fresh.detector.confirmation_sec);
        g_detector_state->setConfig(fresh.detector);
        g_pose_detector->setThreshold(fresh.confidence);
        g_pose_detector->setKeypointThreshold(fresh.keypoint_confidence);
    }

    g_frame_id++;
}

// Evaluate a contiguous RGB888 recording with exactly the same PoseDetector
// and makeObservation/FallDetector path as the live camera. This is intended
// for reproducible public-video or dataset checks on a device with the NPU:
// ffmpeg can produce the input (`-pix_fmt rgb24 -f rawvideo`). No camera,
// RTSP, debug WebSocket, or MQTT service is touched in this mode.
static int run_offline_rgb() {
    const std::uint64_t frame_bytes = static_cast<std::uint64_t>(g_config.offline_width) *
                                      static_cast<std::uint64_t>(g_config.offline_height) * 3u;
    if (frame_bytes == 0 || frame_bytes > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "offline frame size is invalid\n";
        return 2;
    }

    std::ifstream input(g_config.offline_rgb_path, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "cannot open offline RGB file: " << g_config.offline_rgb_path << "\n";
        return 2;
    }

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(frame_bytes));
    std::uint32_t frame_id = 0;
    FallOutput last_output;
    std::size_t frames = 0;
    while (g_running.load()) {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(frame_bytes));
        const std::streamsize got = input.gcount();
        if (got == 0) break;
        if (got != static_cast<std::streamsize>(frame_bytes)) {
            std::cerr << "offline RGB file ends mid-frame at frame " << frame_id << "\n";
            return 2;
        }

        ma_img_t image{};
        image.size = static_cast<std::uint32_t>(frame_bytes);
        image.width = static_cast<std::uint16_t>(g_config.offline_width);
        image.height = static_cast<std::uint16_t>(g_config.offline_height);
        image.format = MA_PIXEL_FORMAT_RGB888;
        image.rotate = MA_PIXEL_ROTATE_0;
        image.data = buffer.data();

        const double now_sec = static_cast<double>(frame_id) / g_config.offline_fps;
        const Subject* subject = g_pose_detector->detectPrimary(&image);
        if (g_pose_detector->inferenceFailed()) {
            std::cerr << "pose inference failed at offline frame " << frame_id << "\n";
            return 3;
        }
        const int feature_height = g_pose_detector->inputHeight() > 0
            ? g_pose_detector->inputHeight() : g_config.offline_height;
        const FallObservation observation = makeObservation(subject, now_sec, feature_height);
        FallObservation classified_observation = observation;
        const int feature_width = g_pose_detector->inputWidth() > 0
            ? g_pose_detector->inputWidth() : g_config.offline_width;
        const TemporalPrediction temporal = g_temporal_classifier->update(
            makeTemporalFrame(subject ? &subject->pose : nullptr, observation,
                              feature_width, feature_height), now_sec);
        classified_observation.temporal_available = true;
        classified_observation.temporal_positive = temporal.positive;
        classified_observation.temporal_probability = temporal.probability;
        last_output = g_detector_state->update(classified_observation);

        PayloadContext ctx;
        ctx.timestamp_ms = static_cast<std::uint64_t>(now_sec * 1000.0 + 0.5);
        ctx.frame_id = frame_id;
        ctx.inference_time_ms = 0.0f;  // offline source has no wall-clock budget
        ctx.person_detected = subject != nullptr;
        ctx.person_count = subject != nullptr ? 1 : 0;
        ctx.fallen_count = last_output.fall_detected ? 1 : 0;
        ctx.infer_w = g_pose_detector->inputWidth() > 0 ? g_pose_detector->inputWidth() : g_config.offline_width;
        ctx.infer_h = feature_height;
        ctx.stream_w = g_config.offline_width;
        ctx.stream_h = g_config.offline_height;

        // JSONL is intentionally one result per input frame. Include the
        // stable COCO-17 feature vector so the exact NPU output can train and
        // replay a temporal classifier without running pose inference again.
        std::cout << buildResultJson(ctx, last_output, classified_observation,
                                     subject ? &subject->pose : nullptr) << '\n';
        ++frames;
        ++frame_id;
    }

    std::cout << "{\"summary\":{\"frames\":" << frames
              << ",\"events\":" << last_output.event_id
              << ",\"last_state\":\"" << fallStateName(last_output.state)
              << "\",\"fall_detected\":" << (last_output.fall_detected ? "true" : "false")
              << "}}\n";
    std::cout.flush();
    return frames > 0 ? 0 : 2;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    g_start_time = std::chrono::steady_clock::now();

    g_watcher = new ConfigWatcher(APP_CONFIG_PATH);
    const AppConfig& cfg = g_watcher->loadInitial();

    MA_LOGI(TAG, "Starting Fall Detection");

    if (g_config.model_path.empty()) {
        g_config.model_path = findPoseModel();
        if (g_config.model_path.empty()) {
            // Name every path tried: "model not found" without the list sends
            // people looking in the wrong directory.
            MA_LOGE(TAG, "No pose model found. Looked in:");
            for (const char* p : MODEL_CANDIDATES) {
                MA_LOGE(TAG, "  %s", p);
            }
            MA_LOGE(TAG, "Install a reCamera console that ships yolo11n-pose, or set "
                         "MODEL_PATH in /etc/fall-detection.conf to a YOLO pose cvimodel.");
            cleanup();
            return 1;
        }
    }
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());

    g_detector_state = std::make_unique<FallDetector>(cfg.detector);
    g_temporal_classifier = std::make_unique<TemporalClassifier>();
    g_pose_detector = new PoseDetector();
    if (!g_pose_detector->init(g_config.model_path)) { cleanup(); return 1; }
    g_pose_detector->setThreshold(cfg.confidence);
    g_pose_detector->setKeypointThreshold(cfg.keypoint_confidence);

    if (!g_config.offline_rgb_path.empty()) {
        const int result = run_offline_rgb();
        cleanup();
        return result;
    }

    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    } else if (g_config.enable_debug) {
        g_debug_started = true;
    }
    if (!init_mqtt()) { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "Fall detection running (stable primary-person tracking + temporal classifier)");
    MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Fall Detection terminated");
    return 0;
}
