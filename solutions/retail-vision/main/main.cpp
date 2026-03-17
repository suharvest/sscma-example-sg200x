#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>

#include <sscma.h>
#include <video.h>
#include "ma_transport_rtsp.h"

#include "detector.h"
#include "person_tracker.h"
#include "zone_metrics.h"
#include "mqtt_publisher.h"

using namespace ma;
using namespace retail_vision;

#define TAG "retail-vision"

static struct {
    std::string model_path = "/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel";

    float conf_threshold = 0.5f;

    // MQTT
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/retail-vision/vision";
    std::string mqtt_user;
    std::string mqtt_pass;

    // RTSP
    int rtsp_port = 8554;
    std::string rtsp_session = "live0";
    std::string rtsp_user;
    std::string rtsp_pass;

    // Video
    int inference_width = 640;
    int inference_height = 640;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Tracking / dwell
    float dwell_speed_threshold = 10.0f;
    float dwell_min_duration = 1.5f;
    float dwell_assistance_threshold = 20.0f;

    // Person height for m/s estimation
    float person_height = 1.7f;

    // Zone metrics window
    float window_duration = 60.0f;

    // Flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;
} g_config;

static std::atomic<bool> g_running(true);
static Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static ZoneMetrics* g_zone_metrics = nullptr;
static MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static TransportRTSP* g_rtsp_transport = nullptr;
static uint32_t g_frame_id = 0;
static std::chrono::steady_clock::time_point g_start_time;

// FPS tracking
static float g_fps = 0.0f;
static int g_fps_frame_count = 0;
static std::chrono::steady_clock::time_point g_fps_last_time;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Retail Vision - People Flow Analytics for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH              Model path (default: %s)\n", g_config.model_path.c_str());
    printf("  -c, --conf-threshold FLOAT    Confidence threshold (default: %.2f)\n", g_config.conf_threshold);
    printf("  --rtsp-port PORT              RTSP server port (default: %d)\n", g_config.rtsp_port);
    printf("  --rtsp-session NAME           RTSP session name (default: %s)\n", g_config.rtsp_session.c_str());
    printf("  --rtsp-user USER              RTSP auth username (default: none)\n");
    printf("  --rtsp-pass PASS              RTSP auth password (default: none)\n");
    printf("  --mqtt-host HOST              MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT              MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC            MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --mqtt-user USER              MQTT auth username (default: none)\n");
    printf("  --mqtt-pass PASS              MQTT auth password (default: none)\n");
    printf("  --person-height FLOAT         Avg person height in meters (default: %.1f)\n", g_config.person_height);
    printf("  --dwell-engaged FLOAT         Engaged threshold sec (default: %.1f)\n", g_config.dwell_min_duration);
    printf("  --dwell-assist FLOAT          Assistance threshold sec (default: %.1f)\n", g_config.dwell_assistance_threshold);
    printf("  --dwell-speed FLOAT           Stationary threshold px/s (default: %.1f)\n", g_config.dwell_speed_threshold);
    printf("  --window-duration FLOAT       Rolling window sec (default: %.1f)\n", g_config.window_duration);
    printf("  --no-rtsp                     Disable RTSP streaming\n");
    printf("  --no-mqtt                     Disable MQTT publishing\n");
    printf("  -v, --verbose                 Verbose logging\n");
    printf("  -h, --help                    Show this help\n");
    printf("\n");
    printf("RTSP Stream: rtsp://<device_ip>:%d/%s\n", g_config.rtsp_port, g_config.rtsp_session.c_str());
    printf("MQTT Topic:  %s\n", g_config.mqtt_topic.c_str());
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model",            required_argument, 0, 'm'},
        {"conf-threshold",   required_argument, 0, 'c'},
        {"rtsp-port",        required_argument, 0, 1},
        {"rtsp-session",     required_argument, 0, 2},
        {"rtsp-user",        required_argument, 0, 3},
        {"rtsp-pass",        required_argument, 0, 4},
        {"mqtt-host",        required_argument, 0, 5},
        {"mqtt-port",        required_argument, 0, 6},
        {"mqtt-topic",       required_argument, 0, 7},
        {"mqtt-user",        required_argument, 0, 15},
        {"mqtt-pass",        required_argument, 0, 16},
        {"person-height",    required_argument, 0, 8},
        {"dwell-engaged",    required_argument, 0, 9},
        {"dwell-assist",     required_argument, 0, 10},
        {"dwell-speed",      required_argument, 0, 11},
        {"window-duration",  required_argument, 0, 12},
        {"no-rtsp",          no_argument,       0, 13},
        {"no-mqtt",          no_argument,       0, 14},
        {"verbose",          no_argument,       0, 'v'},
        {"help",             no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 'c': g_config.conf_threshold = std::stof(optarg); break;
            case 1:   g_config.rtsp_port = std::stoi(optarg); break;
            case 2:   g_config.rtsp_session = optarg; break;
            case 3:   g_config.rtsp_user = optarg; break;
            case 4:   g_config.rtsp_pass = optarg; break;
            case 5:   g_config.mqtt_host = optarg; break;
            case 6:   g_config.mqtt_port = std::stoi(optarg); break;
            case 7:   g_config.mqtt_topic = optarg; break;
            case 15:  g_config.mqtt_user = optarg; break;
            case 16:  g_config.mqtt_pass = optarg; break;
            case 8:   g_config.person_height = std::stof(optarg); break;
            case 9:   g_config.dwell_min_duration = std::stof(optarg); break;
            case 10:  g_config.dwell_assistance_threshold = std::stof(optarg); break;
            case 11:  g_config.dwell_speed_threshold = std::stof(optarg); break;
            case 12:  g_config.window_duration = std::stof(optarg); break;
            case 13:  g_config.enable_rtsp = false; break;
            case 14:  g_config.enable_mqtt = false; break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default:  print_usage(argv[0]); return false;
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
    MA_LOGI(TAG, "Detector initialized (input: %dx%d)", g_detector->getInputWidth(), g_detector->getInputHeight());
    return true;
}

static bool init_tracker() {
    g_tracker = new PersonTracker();

    TrackerConfig cfg;
    cfg.dwell_speed_threshold = g_config.dwell_speed_threshold;
    cfg.dwell_min_duration = g_config.dwell_min_duration;
    cfg.dwell_assistance_threshold = g_config.dwell_assistance_threshold;
    cfg.frame_width = g_config.inference_width;
    cfg.frame_height = g_config.inference_height;
    cfg.avg_person_height_m = g_config.person_height;
    g_tracker->setConfig(cfg);

    g_zone_metrics = new ZoneMetrics();
    g_zone_metrics->setWindowDuration(g_config.window_duration);

    // Wire track removal callback to zone metrics
    g_tracker->setTrackRemovedCallback([](const TrackRecord& record) {
        if (g_zone_metrics) {
            g_zone_metrics->onTrackRemoved(record);
        }
    });

    MA_LOGI(TAG, "Tracker initialized (engaged=%.1fs, assist=%.1fs, speed=%.1fpx/s, height=%.1fm)",
            g_config.dwell_min_duration, g_config.dwell_assistance_threshold,
            g_config.dwell_speed_threshold, g_config.person_height);
    MA_LOGI(TAG, "Zone metrics window: %.0fs", g_config.window_duration);

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

static int rtspFrameCallback(void* pData, void* pArgs, void* pUserData) {
    auto* transport = static_cast<TransportRTSP*>(pUserData);
    if (!transport) return 0;

    VENC_STREAM_S* pstStream = (VENC_STREAM_S*)pData;
    if (!pstStream || pstStream->u32PackCount == 0) return 0;

    for (CVI_U32 i = 0; i < pstStream->u32PackCount; i++) {
        VENC_PACK_S* ppack = &pstStream->pstPack[i];
        transport->send(
            reinterpret_cast<const char*>(ppack->pu8Addr + ppack->u32Offset),
            ppack->u32Len - ppack->u32Offset);
    }

    return 0;
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

    g_rtsp_transport = new TransportRTSP();
    TransportRTSP::Config rtsp_cfg = {
        g_config.rtsp_port,
        MA_PIXEL_FORMAT_H264,
        MA_AUDIO_FORMAT_PCM,
        16000, 1, 16,
        g_config.rtsp_session,
        g_config.rtsp_user,
        g_config.rtsp_pass
    };

    ma_err_t err = g_rtsp_transport->init(&rtsp_cfg);
    if (err != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize RTSP transport: %d", err);
        delete g_rtsp_transport;
        g_rtsp_transport = nullptr;
        return false;
    }

    registerVideoFrameHandler(VIDEO_CH2, 0, rtspFrameCallback, g_rtsp_transport);

    std::string url = "rtsp://";
    if (!g_config.rtsp_user.empty()) {
        url += g_config.rtsp_user + ":" + g_config.rtsp_pass + "@";
    }
    url += "<device_ip>:" + std::to_string(g_config.rtsp_port) + "/" + g_config.rtsp_session;
    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps) %s",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps, url.c_str());
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
    mqtt_config.username = g_config.mqtt_user;
    mqtt_config.password = g_config.mqtt_pass;

    if (!g_mqtt_publisher->init(mqtt_config)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) {
        g_camera->stopStream();
    }

    if (g_rtsp_transport) {
        g_rtsp_transport->deInit();
        delete g_rtsp_transport;
        g_rtsp_transport = nullptr;
    }

    if (g_config.enable_rtsp) {
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    delete g_zone_metrics;
    g_zone_metrics = nullptr;

    delete g_tracker;
    g_tracker = nullptr;

    delete g_detector;
    g_detector = nullptr;

    MA_LOGI(TAG, "Cleanup completed");
}

static void update_fps() {
    g_fps_frame_count++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_fps_last_time).count();
    if (elapsed >= 1000) {
        g_fps = g_fps_frame_count * 1000.0f / elapsed;
        g_fps_frame_count = 0;
        g_fps_last_time = now;
    }
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

    float current_time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time
    ).count() / 1000.0f;

    // Detect
    auto detections = g_detector->detect(&frame);

    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    float inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    ).count() / 1000.0f;

    // Track
    auto tracked_persons = g_tracker->update(detections, current_time_sec);
    auto state_counts = g_tracker->getStateCounts();

    // Update zone metrics
    g_zone_metrics->update(state_counts,
                            g_tracker->getEntryCount(),
                            g_tracker->getExitCount(),
                            current_time_sec);

    // Update FPS
    update_fps();

    // Publish MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        auto zone = g_zone_metrics->getSnapshot();
        g_mqtt_publisher->publishVisionPayload(
            timestamp_ms, g_frame_id, g_fps, inference_time_ms,
            zone, tracked_persons,
            g_config.stream_width, g_config.stream_height,
            g_detector->getInputWidth(), g_detector->getInputHeight());
    }

    // Log
    if (g_config.verbose || !tracked_persons.empty()) {
        auto zone = g_zone_metrics->getSnapshot();
        MA_LOGI(TAG, "F%u: %zu persons, %.1fms, %.1ffps | occ=%d brow=%d eng=%d ast=%d | entry=%d exit=%d",
                g_frame_id, tracked_persons.size(), inference_time_ms, g_fps,
                zone.occupancy_count, zone.browsing_count, zone.engaged_count, zone.assist_count,
                zone.entry_count, zone.exit_count);

        for (const auto& p : tracked_persons) {
            MA_LOGV(TAG, "  [T%d] %s conf=%.0f%% speed=%.2fm/s dwell=%.1fs",
                    p.track_id, getDwellStateName(p.dwell_state),
                    p.detection.score * 100.0f, p.speed_m_s, p.dwell_duration_sec);
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting Retail Vision - People Flow Analytics");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence: %.2f", g_config.conf_threshold);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    g_start_time = std::chrono::steady_clock::now();
    g_fps_last_time = g_start_time;

    if (!init_detector()) { cleanup(); return 1; }
    if (!init_tracker())  { cleanup(); return 1; }
    if (!init_camera())   { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (!init_mqtt())     { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "Retail Vision running...");
    if (g_config.enable_rtsp) MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:%d/%s", g_config.rtsp_port, g_config.rtsp_session.c_str());
    if (g_config.enable_mqtt) MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Retail Vision terminated");
    return 0;
}
