#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>

#include <sscma.h>
#include <video.h>
#include "rtsp_demo.h"

#include "detector.h"
#include "region_blur.h"
#include "mqtt_publisher.h"

using namespace ma;
using namespace detection_blur;

#define TAG "detection-blur"

// Default configuration
static struct {
    // Model path
    std::string model_path = "/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel";

    // Detection parameters
    float threshold = 0.5f;

    // Target classes to blur (empty = blur all)
    std::vector<int> targets;

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/detection-blur/results";

    // Video configuration
    int inference_width = 640;
    int inference_height = 480;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Blur configuration
    int max_regions = 8;

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool enable_blur = true;
    bool verbose = false;
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static Detector* g_detector = nullptr;
static RegionBlur* g_region_blur = nullptr;
static MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Detection Blur for ReCamera\n");
    printf("Hardware-accelerated object blur with Kalman filter tracking\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH          Detection model path (default: %s)\n", g_config.model_path.c_str());
    printf("  -t, --threshold FLOAT     Detection threshold (default: %.2f)\n", g_config.threshold);
    printf("  --targets ID[,ID,...]     Target class IDs to blur (default: all)\n");
    printf("  --mqtt-host HOST          MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT          MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC        MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-rtsp                 Disable RTSP streaming\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  --no-blur                 Disable blur overlay (detection only)\n");
    printf("  --max-regions N           Max mosaic regions (1-8, default: %d)\n", g_config.max_regions);
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
}

static std::vector<int> parse_targets(const char* str) {
    std::vector<int> targets;
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        std::string token = s.substr(pos, comma - pos);
        if (!token.empty()) {
            targets.push_back(std::stoi(token));
        }
        pos = comma + 1;
    }
    return targets;
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"threshold", required_argument, 0, 't'},
        {"targets", required_argument, 0, 1},
        {"mqtt-host", required_argument, 0, 2},
        {"mqtt-port", required_argument, 0, 3},
        {"mqtt-topic", required_argument, 0, 4},
        {"no-rtsp", no_argument, 0, 5},
        {"no-mqtt", no_argument, 0, 6},
        {"no-blur", no_argument, 0, 7},
        {"max-regions", required_argument, 0, 8},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:t:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                g_config.model_path = optarg;
                break;
            case 't':
                g_config.threshold = std::stof(optarg);
                break;
            case 1:
                g_config.targets = parse_targets(optarg);
                break;
            case 2:
                g_config.mqtt_host = optarg;
                break;
            case 3:
                g_config.mqtt_port = std::stoi(optarg);
                break;
            case 4:
                g_config.mqtt_topic = optarg;
                break;
            case 5:
                g_config.enable_rtsp = false;
                break;
            case 6:
                g_config.enable_mqtt = false;
                break;
            case 7:
                g_config.enable_blur = false;
                break;
            case 8:
                g_config.max_regions = std::stoi(optarg);
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

static bool init_detector() {
    g_detector = new Detector();
    if (!g_detector->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize detector");
        return false;
    }
    g_detector->setThreshold(g_config.threshold);
    MA_LOGI(TAG, "Detector initialized (input: %dx%d)",
            g_detector->getInputWidth(), g_detector->getInputHeight());

    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            // Set channel 0 for inference
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Set inference resolution
            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // Disable physical address mode - CPU-based image processing needs virtual addresses
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d for inference)",
                    g_config.inference_width, g_config.inference_height);
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

    // Setup H.264 streaming channel (CH2)
    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    // Register RTSP handler
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

    g_mqtt_publisher = new MqttPublisher();
    MqttConfig mqtt_config;
    mqtt_config.host = g_config.mqtt_host;
    mqtt_config.port = g_config.mqtt_port;
    mqtt_config.topic = g_config.mqtt_topic;

    if (!g_mqtt_publisher->init(mqtt_config)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }

    return true;
}

static bool init_blur() {
    if (!g_config.enable_blur) {
        MA_LOGI(TAG, "Blur overlay disabled");
        return true;
    }

    if (!g_config.enable_rtsp) {
        MA_LOGW(TAG, "Blur requires RTSP streaming, disabling blur");
        g_config.enable_blur = false;
        return true;
    }

    g_region_blur = new RegionBlur();
    g_region_blur->setMaxRegions(g_config.max_regions);
    if (!g_config.targets.empty()) {
        g_region_blur->setTargets(g_config.targets);
    }

    if (!g_region_blur->init(g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "Failed to initialize region blur");
        delete g_region_blur;
        g_region_blur = nullptr;
        return false;
    }

    MA_LOGI(TAG, "Region blur enabled (max_regions=%d)", g_config.max_regions);
    return true;
}

static void cleanup() {
    if (g_region_blur) {
        g_region_blur->deinit();
        delete g_region_blur;
        g_region_blur = nullptr;
    }

    if (g_camera) {
        g_camera->stopStream();
    }

    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
    }

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

    // Step 1: Run object detection
    std::vector<DetectionBox> detections = g_detector->detect(&frame);

    // Return frame to camera
    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    // Step 2: Feed detections to blur overlay
    if (g_config.enable_blur && g_region_blur) {
        g_region_blur->onDetection(detections);
    }

    // Step 3: Publish results via MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        g_mqtt_publisher->publishResults(timestamp_ms, g_frame_id, detections,
                                          static_cast<float>(inference_time));
    }

    // Log results
    if (g_config.verbose || !detections.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu detections, inference=%lldms",
                g_frame_id, detections.size(), inference_time);

        for (const auto& det : detections) {
            MA_LOGI(TAG, "  [target=%d] score=%.2f at (%.3f, %.3f, %.3f, %.3f)",
                    det.target, det.score, det.x, det.y, det.w, det.h);
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) {
        return 1;
    }

    MA_LOGI(TAG, "Starting Detection Blur Application");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Threshold: %.2f", g_config.threshold);
    if (!g_config.targets.empty()) {
        std::string target_str;
        for (size_t i = 0; i < g_config.targets.size(); i++) {
            if (i > 0) target_str += ",";
            target_str += std::to_string(g_config.targets[i]);
        }
        MA_LOGI(TAG, "Target classes: %s", target_str.c_str());
    } else {
        MA_LOGI(TAG, "Target classes: all");
    }

    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize components
    if (!init_detector()) {
        MA_LOGE(TAG, "Detector initialization failed");
        cleanup();
        return 1;
    }

    if (!init_camera()) {
        MA_LOGE(TAG, "Camera initialization failed");
        cleanup();
        return 1;
    }

    if (!init_video_streaming()) {
        MA_LOGE(TAG, "Video streaming initialization failed");
        cleanup();
        return 1;
    }

    if (!init_mqtt()) {
        MA_LOGE(TAG, "MQTT initialization failed");
        cleanup();
        return 1;
    }

    // Start camera streaming
    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // Start video streaming if enabled
    if (g_config.enable_rtsp) {
        startVideo();
    }

    // Initialize blur AFTER video pipeline is started (RGN needs VPSS channel running)
    if (!init_blur()) {
        MA_LOGW(TAG, "Blur initialization failed, continuing without blur");
        g_config.enable_blur = false;
    }

    MA_LOGI(TAG, "Detection blur running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:554/live");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());
    MA_LOGI(TAG, "Mosaic blur: %s (max %d regions)", g_config.enable_blur ? "ENABLED" : "DISABLED", g_config.max_regions);

    // Main processing loop
    while (g_running.load()) {
        process_frame();
    }

    // Cleanup
    cleanup();

    MA_LOGI(TAG, "Detection Blur Application terminated");
    return 0;
}
