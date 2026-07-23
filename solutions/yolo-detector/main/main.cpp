#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>
#include <map>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "rtsp_demo.h"

#include "detector.h"
#include "person_tracker.h"
#include "mqtt_payload.h"
#include "app_config.h"

using namespace ma;
using namespace yolo;

#define TAG "yolo-detector"

// Default configuration
static struct {
    // Model path
    std::string model_path = "/userdata/local/models/yolo11n_cv181x_int8.cvimodel";

    // Detection parameters
    float conf_threshold = 0.25f;

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/yolo/detections";

    // Video configuration
    int inference_width = 640;
    int inference_height = 640;
    int inference_fps = 15;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Tracking configuration
    bool enable_tracking = true;
    float dwell_speed_threshold = 10.0f;
    float dwell_min_duration = 1.5f;
    float dwell_assistance_threshold = 20.0f;

    // Hardware blur (RGN COVEREX)
    bool enable_blur = false;
    int max_blur_regions = 4;
    std::string blur_classes = "";  // comma-separated class IDs to blur, empty = all

    // Debug stream (H.264-over-WS + results JSON for the supervisor console)
    bool enable_debug = true;
    int debug_port = 8001;

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;
} g_config;

// Supervisor-managed per-app configuration (appMgr setConfig writes it,
// validated against the manifest's config_schema; missing file = defaults).
static const char* APP_CONFIG_PATH = "/userdata/local/apps/yolo-detector.config.json";
static AppConfig g_app_config;

// Global state
static std::atomic<bool> g_running(true);
static Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;
static std::chrono::steady_clock::time_point g_start_time;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("YOLO Object Detector for ReCamera\n");
    printf("Supports YOLO11, YOLO26, YOLOv8, YOLOv5 via ModelFactory auto-detection\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH          Model path (default: %s)\n", g_config.model_path.c_str());
    printf("  -c, --conf-threshold F    Confidence threshold (default: %.2f)\n", g_config.conf_threshold);
    printf("  --mqtt-host HOST          MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT          MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC        MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-tracking             Disable person tracking\n");
    printf("  --dwell-speed F           Dwell speed threshold px/s (default: %.1f)\n", g_config.dwell_speed_threshold);
    printf("  --dwell-engaged F         Time for ENGAGED state sec (default: %.1f)\n", g_config.dwell_min_duration);
    printf("  --dwell-assist F          Time for ASSISTANCE state sec (default: %.1f)\n", g_config.dwell_assistance_threshold);
    printf("  --blur                    Enable hardware privacy blur (RGN COVEREX)\n");
    printf("  --blur-classes IDS        Comma-separated class IDs to blur (default: all)\n");
    printf("  --max-blur-regions N      Max blur regions (default: %d, max: 4)\n", g_config.max_blur_regions);
    printf("  --no-rtsp                 Disable RTSP streaming\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  --no-debug                Disable debug WebSocket stream\n");
    printf("  --debug-port PORT         Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nRTSP Stream: rtsp://<device_ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"conf-threshold", required_argument, 0, 'c'},
        {"mqtt-host", required_argument, 0, 1},
        {"mqtt-port", required_argument, 0, 2},
        {"mqtt-topic", required_argument, 0, 3},
        {"no-tracking", no_argument, 0, 4},
        {"dwell-speed", required_argument, 0, 5},
        {"dwell-engaged", required_argument, 0, 6},
        {"dwell-assist", required_argument, 0, 7},
        {"no-rtsp", no_argument, 0, 8},
        {"no-mqtt", no_argument, 0, 9},
        {"blur", no_argument, 0, 10},
        {"blur-classes", required_argument, 0, 11},
        {"max-blur-regions", required_argument, 0, 12},
        {"no-debug", no_argument, 0, 13},
        {"debug-port", required_argument, 0, 14},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 'c': g_config.conf_threshold = std::stof(optarg); break;
            case 1: g_config.mqtt_host = optarg; break;
            case 2: g_config.mqtt_port = std::stoi(optarg); break;
            case 3: g_config.mqtt_topic = optarg; break;
            case 4: g_config.enable_tracking = false; break;
            case 5: g_config.dwell_speed_threshold = std::stof(optarg); break;
            case 6: g_config.dwell_min_duration = std::stof(optarg); break;
            case 7: g_config.dwell_assistance_threshold = std::stof(optarg); break;
            case 8: g_config.enable_rtsp = false; break;
            case 9: g_config.enable_mqtt = false; break;
            case 10: g_config.enable_blur = true; break;
            case 11: g_config.blur_classes = optarg; break;
            case 12: g_config.max_blur_regions = std::min(4, std::stoi(optarg)); break;
            case 13: g_config.enable_debug = false; break;
            case 14: g_config.debug_port = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static bool init_detector() {
    g_detector = new Detector();
    if (!g_detector->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize detector");
        return false;
    }
    g_detector->setThreshold(g_config.conf_threshold);
    return true;
}

// Apply the supervisor-managed app config on top of built-in defaults.
// Called BEFORE parse_args so explicit CLI flags (set by an operator in the
// init script) still win over console-managed configuration.
static void apply_app_config() {
    if (!load_app_config(APP_CONFIG_PATH, g_app_config)) {
        return; // no config file: behavior identical to before
    }
    if (g_app_config.has_confidence) {
        g_config.conf_threshold = g_app_config.confidence;
        MA_LOGI(TAG, "Config: confidence threshold = %.2f", g_config.conf_threshold);
    }
    if (g_app_config.has_tracking) {
        g_config.enable_tracking = g_app_config.tracking;
        MA_LOGI(TAG, "Config: tracking = %s", g_config.enable_tracking ? "on" : "off");
    }
    if (g_app_config.zone_enabled) {
        MA_LOGI(TAG, "Config: counting zone with %zu points", g_app_config.zone_points.size());
    }
    if (g_app_config.line_enabled) {
        MA_LOGI(TAG, "Config: entry line enabled (%s)", g_app_config.line_ab_in ? "ab_in" : "ab_out");
    }
}

static bool init_tracker() {
    if (!g_config.enable_tracking) {
        MA_LOGI(TAG, "Person tracking disabled");
        return true;
    }

    g_tracker = new PersonTracker();
    TrackerConfig tracker_config;
    tracker_config.dwell_speed_threshold = g_config.dwell_speed_threshold;
    tracker_config.dwell_min_duration = g_config.dwell_min_duration;
    tracker_config.dwell_assistance_threshold = g_config.dwell_assistance_threshold;
    tracker_config.frame_width = g_config.inference_width;
    tracker_config.frame_height = g_config.inference_height;
    g_tracker->setConfig(tracker_config);

    if (g_app_config.zone_enabled) {
        g_tracker->setCountZone(g_app_config.zone_points);
    }
    if (g_app_config.line_enabled) {
        g_tracker->setEntryLine(g_app_config.line_a, g_app_config.line_b, g_app_config.line_ab_in);
    }

    MA_LOGI(TAG, "Person tracker initialized (engaged: %.1fs, assist: %.1fs)",
            g_config.dwell_min_duration, g_config.dwell_assistance_threshold);
    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Apply the inference frame rate (else the channel defaults to 30fps
            // and the capture->inference FIFO fills faster than inference drains).
            value.i32 = g_config.inference_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d)", g_config.inference_width, g_config.inference_height);
            return true;
        }
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
    // Debug stream registers its own consumer (idx 1); RTSP owns idx 0.
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));

    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps)",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps);
    return true;
}

// Assemble the debug /results envelope (shared debug_stream builder).
// Detection x/y are normalized center coordinates; box[5] is the
// human-readable class name the console overlay renders verbatim.
static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<Detection>& detections,
                                            float inference_time_ms) {
    std::vector<debug_stream_box_t> boxes;
    boxes.reserve(detections.size());
    for (const auto& det : detections) {
        boxes.push_back({det.x * g_config.inference_width,
                         det.y * g_config.inference_height,
                         det.w * g_config.inference_width,
                         det.h * g_config.inference_height,
                         det.confidence, Detector::getClassName(det.class_id)});
    }
    // The inference channel is letterboxed vs the 16:9 debug video (stream);
    // remap boxes into the stream frame so the overlay aligns with the video.
    debug_stream_letterbox_to_display(boxes, g_config.inference_width, g_config.inference_height,
                                      g_config.stream_width, g_config.stream_height);
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      boxes);
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "yolo-detector";
    opts.client_id     = "recamera-yolo-detector";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant MQTT Discovery entity table (field names must match the
    // results JSON built by mqtt_payload.cpp — a single schema with
    // detection_count/detections regardless of the tracking setting).
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "detection_count", "Detection Count",
                     "{{ value_json.detection_count }}",
                     "", "", "measurement"},
        EntityConfig{EntityType::BinarySensor, "person_occupancy", "Person Detected",
                     "{{ 'ON' if value_json.detections | selectattr('class_name', 'equalto', 'person') | list | length > 0 else 'OFF' }}",
                     "occupancy", "", ""},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    if (g_config.enable_rtsp) { deinitRtsp(); deinitVideo(); }
    if (g_mqtt_publisher) { g_mqtt_publisher->deinit(); delete g_mqtt_publisher; g_mqtt_publisher = nullptr; }
    if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
    if (g_detector) { delete g_detector; g_detector = nullptr; }
    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time
    ).count();
    float current_time_sec = elapsed / 1000.0f;

    // Run detection (ModelFactory auto-selects YOLO variant)
    std::vector<Detection> detections = g_detector->detect(&frame);

    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    // Push the same inference result to debug WS clients (sscma-node format).
    // debug_stream is lazy: skip building JSON when nobody is connected.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        std::string debug_json = build_debug_results_json(timestamp_ms, g_frame_id, detections,
                                                          static_cast<float>(inference_time));
        debug_stream_publish_result(debug_json.c_str(), debug_json.size());
    }

    // Publish results via MQTT (single schema; tracking adds optional fields)
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        std::map<int, int> det_track_ids;
        TrackingSummary tracking;
        bool has_tracking = false;

        if (g_config.enable_tracking && g_tracker) {
            auto tracked_persons = g_tracker->update(detections, current_time_sec);
            for (const auto& person : tracked_persons) {
                det_track_ids[person.detection.id] = person.track_id;
            }
            tracking.active_tracks = g_tracker->getTrackCount();
            if (g_tracker->hasEntryLine()) {
                auto line_crossing = g_tracker->getLineCrossing();
                tracking.has_line = true;
                tracking.line_in = line_crossing.in_count;
                tracking.line_out = line_crossing.out_count;
            }
            has_tracking = true;
        }

        std::string payload = buildResultJson(timestamp_ms, g_frame_id, detections,
                                              static_cast<float>(inference_time),
                                              has_tracking ? &det_track_ids : nullptr,
                                              has_tracking ? &tracking : nullptr);
        g_mqtt_publisher->publishResultsJson(payload);

        if (g_config.verbose || !detections.empty()) {
            if (has_tracking) {
                MA_LOGI(TAG, "Frame %u: %zu detections, %d active tracks, inference=%lldms",
                        g_frame_id, detections.size(), tracking.active_tracks, inference_time);
            } else {
                MA_LOGI(TAG, "Frame %u: %zu detections, inference=%lldms",
                        g_frame_id, detections.size(), inference_time);
            }
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    // Console-managed config first, then CLI flags (CLI wins on conflicts).
    apply_app_config();
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting YOLO Detector (universal)");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence threshold: %.2f", g_config.conf_threshold);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    g_start_time = std::chrono::steady_clock::now();

    if (!init_detector()) { cleanup(); return 1; }
    if (!init_tracker()) { cleanup(); return 1; }
    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    // Debug stream (non-fatal: on failure run without it)
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt()) { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "YOLO detector running...");
    MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());
    MA_LOGI(TAG, "Tracking: %s | Blur: %s",
            g_config.enable_tracking ? "ON" : "OFF",
            g_config.enable_blur ? "ON" : "OFF");

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "YOLO Detector terminated");
    return 0;
}
