#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <getopt.h>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "rtsp_server.h"

#include "motion_config.h"
#include "motion_detector.h"

using namespace ma;
using namespace region_motion;

#define TAG "region-motion"

static MotionConfig g_config;
static std::atomic<bool> g_running(true);
static Camera* g_camera = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static uint32_t g_frame_id = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    std::printf("Region Motion Detection for ReCamera\n");
    std::printf("Usage: %s [options]\n\n", prog);
    std::printf("Options:\n");
    std::printf("  -c, --config PATH         Config file path (default: /userdata/local/apps/region-motion.config.json)\n");
    std::printf("  -m, --mqtt-host HOST      MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    std::printf("  -p, --mqtt-port PORT      MQTT broker port (default: %d)\n", g_config.mqtt_port);
    std::printf("  --mqtt-topic TOPIC        MQTT results topic (default: %s)\n", g_config.mqtt_topic.c_str());
    std::printf("  --pixel-threshold N       Pixel difference sensitivity 5..100 (default: %d)\n", g_config.pixel_threshold);
    std::printf("  --min-motion-area F       Minimum motion ratio 0.001..0.5 (default: %.3f)\n", g_config.min_motion_area);
    std::printf("  --debounce-frames N       Debounce frames 1..10 (default: %d)\n", g_config.debounce_frames);
    std::printf("  --no-rtsp                 Disable RTSP streaming\n");
    std::printf("  --no-mqtt                 Disable MQTT publishing\n");
    std::printf("  --no-debug                Disable debug WebSocket stream\n");
    std::printf("  --debug-port PORT         Debug WebSocket port (default: %d)\n", g_config.debug_port);
    std::printf("  -v, --verbose             Enable verbose logging\n");
    std::printf("  -h, --help                Show this help message\n");
}

static bool parse_args(int argc, char** argv, std::string& config_path) {
    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"mqtt-host", required_argument, 0, 'm'},
        {"mqtt-port", required_argument, 0, 'p'},
        {"mqtt-topic", required_argument, 0, 1},
        {"pixel-threshold", required_argument, 0, 2},
        {"min-motion-area", required_argument, 0, 3},
        {"debounce-frames", required_argument, 0, 4},
        {"no-rtsp", no_argument, 0, 5},
        {"no-mqtt", no_argument, 0, 6},
        {"no-debug", no_argument, 0, 7},
        {"debug-port", required_argument, 0, 8},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'c':
                config_path = optarg;
                break;
            case 'm':
                g_config.mqtt_host = optarg;
                break;
            case 'p':
                g_config.mqtt_port = std::stoi(optarg);
                break;
            case 1:
                g_config.mqtt_topic = optarg;
                break;
            case 2:
                g_config.pixel_threshold = std::stoi(optarg);
                break;
            case 3:
                g_config.min_motion_area = std::stof(optarg);
                break;
            case 4:
                g_config.debounce_frames = std::stoi(optarg);
                break;
            case 5:
                g_config.enable_rtsp = false;
                break;
            case 6:
                g_config.enable_mqtt = false;
                break;
            case 7:
                g_config.enable_debug = false;
                break;
            case 8:
                g_config.debug_port = std::stoi(optarg);
                break;
            case 'v':
                g_config.verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                return false;
        }
    }
    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();
    if (!device) {
        MA_LOGE(TAG, "Device instance is null");
        return false;
    }

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            // Channel 0 for algorithm inference
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Inference frame rate
            value.i32 = g_config.inference_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

            // Inference resolution
            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // CPU accessible RGB frame data
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps for inference)",
                    g_config.inference_width, g_config.inference_height, g_config.inference_fps);
            return true;
        }
    }

    MA_LOGE(TAG, "No camera sensor found");
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

    // Channel 2 for H.264 video stream
    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    // Register RTSP consumer (idx 0)
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);

    // Initialize RTSP server
    initRtsp((0x01 << VIDEO_CH2));

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
    opts.app_id = "region-motion";
    opts.client_id = "recamera-region-motion";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host = g_config.mqtt_host;
    opts.legacy_port = static_cast<uint16_t>(g_config.mqtt_port);

    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::BinarySensor, "motion", "Motion",
                     "{{ value_json.state }}", "motion", "", ""},
    };

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        delete g_mqtt;
        g_mqtt = nullptr;
        return false;
    }

    MA_LOGI(TAG, "MQTT publisher initialized (broker: %s:%d, topic: %s)",
            g_config.mqtt_host.c_str(), g_config.mqtt_port, g_config.mqtt_topic.c_str());
    return true;
}

static void cleanup() {
    if (g_camera) {
        g_camera->stopStream();
    }
    if (g_config.enable_debug) {
        debug_stream_destroy();
    }
    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }
    if (g_mqtt) {
        g_mqtt->deinit();
        delete g_mqtt;
        g_mqtt = nullptr;
    }
    MA_LOGI(TAG, "Cleanup completed");
}

int main(int argc, char** argv) {
    std::string config_path = "/userdata/local/apps/region-motion.config.json";

    // Pre-parse args to see if a custom config file path was given
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "-c" || std::string(argv[i]) == "--config") {
            config_path = argv[i + 1];
            break;
        }
    }

    // Load base configuration from JSON
    g_config = MotionConfig::load(config_path);

    // Allow CLI arguments to override
    if (!parse_args(argc, argv, config_path)) {
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    MA_LOGI(TAG, "Starting Region Motion Detection (sensitivity: %d, min_ratio: %.3f, debounce: %d, roi_pts: %zu)",
            g_config.pixel_threshold, g_config.min_motion_area, g_config.debounce_frames, g_config.roi_zone.size());

    if (!init_mqtt()) {
        MA_LOGW(TAG, "Continuing without MQTT discovery");
    }

    if (!init_camera()) {
        MA_LOGE(TAG, "Failed to initialize camera");
        cleanup();
        return 2;
    }

    if (!init_video_streaming()) {
        MA_LOGE(TAG, "Failed to initialize video streaming");
        cleanup();
        return 3;
    }

    if (g_config.enable_debug) {
        if (debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
            MA_LOGW(TAG, "Debug stream unavailable, disabling");
            g_config.enable_debug = false;
        }
    }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    if (g_config.enable_rtsp) {
        startVideo();
    }

    MotionDetector detector(g_config);
    bool first_mqtt_publish = true;

    MA_LOGI(TAG, "Region motion detector running (inference: %dx%d, stream: %dx%d)",
            g_config.inference_width, g_config.inference_height,
            g_config.stream_width, g_config.stream_height);

    while (g_running.load()) {
        ma_img_t frame;
        if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        g_frame_id++;

        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Run motion detector on RGB frame
        MotionResult res = detector.process(static_cast<const uint8_t*>(frame.data), frame.width, frame.height);

        // Offer unannotated snapshot for debug stream before returning frame
        if (g_config.enable_debug && debug_stream_snapshot_armed()) {
            debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
        }

        // Return frame to camera pipeline
        g_camera->returnFrame(frame);

        auto t1 = std::chrono::high_resolution_clock::now();
        float inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // HA MQTT publication on state change, first start, or periodic heartbeat (every 50 frames ~ 5s)
        if (g_config.enable_mqtt && g_mqtt) {
            if (res.state_changed || first_mqtt_publish || (g_frame_id % (g_config.inference_fps * 5) == 0)) {
                first_mqtt_publish = false;
                nlohmann::json payload;
                payload["state"] = res.motion_detected ? "ON" : "OFF";
                payload["motion"] = res.motion_detected;
                payload["ratio"] = std::round(res.motion_ratio * 1000.0f) / 1000.0f;
                payload["frame_id"] = g_frame_id;
                g_mqtt->publishResultsJson(payload.dump());
            }
        }

        // WebSocket debug results broadcast
        if (g_config.enable_debug && (debug_stream_results_client_count() > 0 || debug_stream_video_client_count() > 0)) {
            // Remap bounding boxes from inference resolution to stream resolution
            debug_stream_letterbox_to_display(res.boxes, g_config.inference_width, g_config.inference_height,
                                              g_config.stream_width, g_config.stream_height);

            std::ostringstream extra;
            extra << std::fixed << std::setprecision(1);

            // polygons: configured roi_zone points
            extra << "\"polygons\":[";
            const auto& roi = detector.getRoiZone();
            if (roi.size() >= 3) {
                extra << "{\"label\":\"ROI\",\"points\":[";
                for (size_t i = 0; i < roi.size(); ++i) {
                    if (i > 0) extra << ",";
                    extra << "[" << std::setprecision(4) << roi[i].x << "," << roi[i].y << std::setprecision(1) << "]";
                }
                extra << "]}";
            }
            extra << "],";

            // status_card: title "Region Motion", tone "warn" or "ok"
            float pct = res.motion_ratio * 100.0f;
            std::ostringstream pct_oss;
            pct_oss << std::fixed << std::setprecision(1) << pct << "%";

            extra << "\"status_card\":{";
            extra << "\"title\":\"Region Motion\",";
            extra << "\"tone\":\"" << (res.motion_detected ? "warn" : "ok") << "\",";
            extra << "\"metrics\":[";
            extra << "{\"k\":\"Status\",\"v\":\"" << (res.motion_detected ? "Motion Detected" : "Clear") << "\"},";
            extra << "{\"k\":\"Motion Area\",\"v\":\"" << pct_oss.str() << "\"}";
            extra << "]}";

            std::string debug_json = debug_stream_build_results(
                timestamp_ms, g_frame_id, inference_ms,
                g_config.stream_width, g_config.stream_height,
                res.boxes, nullptr, extra.str());

            debug_stream_publish_result(debug_json.c_str(), debug_json.length());
        }
    }

    cleanup();
    return 0;
}
