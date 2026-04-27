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

#include "face_detector.h"
#include "facemesh_pipeline.h"
#include "mqtt_publisher.h"
#include "drowsiness_detector.h"
#include "yawn_detector.h"
#include "local_alert.h"

using namespace ma;
using namespace facemesh_reader;

#define TAG "facemesh-reader"

// Default configuration
static struct {
    // Model paths
    std::string face_model     = "/userdata/local/models/yolo-face_mixfp16.cvimodel";
    std::string facemesh_model = "/userdata/local/models/face_landmark_cv181x_bf16.cvimodel";

    // Detection parameters
    float face_threshold = 0.4f;

    // MQTT configuration
    std::string mqtt_host  = "localhost";
    int         mqtt_port  = 1883;
    std::string mqtt_topic = "recamera/facemesh-reader/results";

    // Video configuration
    int inference_width  = 640;
    int inference_height = 480;
    int inference_fps    = 10;
    int stream_width     = 1280;
    int stream_height    = 720;
    int stream_fps       = 15;

    // Runtime flags
    bool enable_rtsp        = true;
    bool enable_mqtt        = true;
    bool include_landmarks  = false;  // include 468 (x,y) per face in MQTT JSON
    bool verbose            = false;

    // Phase 2: drowsiness / yawn thresholds (override defaults via CLI)
    float ear_threshold        = 0.21f;
    float ear_continuous_sec   = 2.0f;
    float mar_threshold        = 0.65f;
    float perclos_warning_pct  = 15.f;
    float perclos_critical_pct = 20.f;
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static FaceDetector*     g_face_detector = nullptr;
static FacemeshPipeline* g_pipeline      = nullptr;
static MqttPublisher*    g_mqtt_publisher = nullptr;
static Camera*           g_camera        = nullptr;
static uint32_t          g_frame_id      = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("FaceMesh Reader for ReCamera (EAR / MAR drowsiness metrics)\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -f, --face-model PATH       Face detection model (default: %s)\n", g_config.face_model.c_str());
    printf("  --facemesh-model PATH       FaceMesh landmark model (default: %s)\n", g_config.facemesh_model.c_str());
    printf("  -t, --threshold FLOAT       Face detection threshold (default: %.2f)\n", g_config.face_threshold);
    printf("  -m, --mqtt-host HOST        MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  -p, --mqtt-port PORT        MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC          MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-rtsp                   Disable RTSP streaming\n");
    printf("  --no-mqtt                   Disable MQTT publishing\n");
    printf("  --include-landmarks         Embed 468 landmarks per face in MQTT JSON\n");
    printf("  --ear-threshold FLOAT       EAR closed-eye threshold (default: %.2f)\n", g_config.ear_threshold);
    printf("  --ear-continuous-sec FLOAT  Continuous closure to trigger drowsy (default: %.1f)\n", g_config.ear_continuous_sec);
    printf("  --mar-threshold FLOAT       MAR yawn threshold (default: %.2f)\n", g_config.mar_threshold);
    printf("  --perclos-warning FLOAT     PERCLOS warning %% (default: %.1f)\n", g_config.perclos_warning_pct);
    printf("  --perclos-critical FLOAT    PERCLOS critical %% (default: %.1f)\n", g_config.perclos_critical_pct);
    printf("  -v, --verbose               Enable verbose logging\n");
    printf("  -h, --help                  Show this help message\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"face-model",        required_argument, 0, 'f'},
        {"facemesh-model",    required_argument, 0,  1 },
        {"threshold",         required_argument, 0, 't'},
        {"mqtt-host",         required_argument, 0, 'm'},
        {"mqtt-port",         required_argument, 0, 'p'},
        {"mqtt-topic",        required_argument, 0,  2 },
        {"no-rtsp",           no_argument,       0,  3 },
        {"no-mqtt",           no_argument,       0,  4 },
        {"include-landmarks", no_argument,       0,  5 },
        {"ear-threshold",     required_argument, 0,  6 },
        {"ear-continuous-sec",required_argument, 0,  7 },
        {"mar-threshold",     required_argument, 0,  8 },
        {"perclos-warning",   required_argument, 0,  9 },
        {"perclos-critical",  required_argument, 0, 10 },
        {"verbose",           no_argument,       0, 'v'},
        {"help",              no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:t:m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f': g_config.face_model     = optarg; break;
            case  1 : g_config.facemesh_model = optarg; break;
            case 't': g_config.face_threshold = std::stof(optarg); break;
            case 'm': g_config.mqtt_host      = optarg; break;
            case 'p': g_config.mqtt_port      = std::stoi(optarg); break;
            case  2 : g_config.mqtt_topic     = optarg; break;
            case  3 : g_config.enable_rtsp    = false; break;
            case  4 : g_config.enable_mqtt    = false; break;
            case  5 : g_config.include_landmarks = true; break;
            case  6 : g_config.ear_threshold        = std::stof(optarg); break;
            case  7 : g_config.ear_continuous_sec   = std::stof(optarg); break;
            case  8 : g_config.mar_threshold        = std::stof(optarg); break;
            case  9 : g_config.perclos_warning_pct  = std::stof(optarg); break;
            case 10 : g_config.perclos_critical_pct = std::stof(optarg); break;
            case 'v': g_config.verbose        = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default:
                print_usage(argv[0]);
                return false;
        }
    }
    return true;
}

static bool init_models() {
    g_face_detector = new FaceDetector();
    if (!g_face_detector->init(g_config.face_model)) {
        MA_LOGE(TAG, "Failed to initialize face detector");
        return false;
    }
    g_face_detector->setThreshold(g_config.face_threshold);
    MA_LOGI(TAG, "Face detector initialized (input: %dx%d)",
            g_face_detector->getInputWidth(), g_face_detector->getInputHeight());

    g_pipeline = new FacemeshPipeline();
    if (!g_pipeline->init(g_config.facemesh_model)) {
        MA_LOGE(TAG, "Failed to initialize FaceMesh pipeline");
        return false;
    }

    // Phase 2: push CLI thresholds into the on-edge state machine.
    DrowsinessDetector::Config dcfg;
    dcfg.ear_threshold        = g_config.ear_threshold;
    dcfg.ear_continuous_sec   = g_config.ear_continuous_sec;
    dcfg.perclos_warning_pct  = g_config.perclos_warning_pct;
    dcfg.perclos_critical_pct = g_config.perclos_critical_pct;
    g_pipeline->configureDrowsiness(dcfg);

    YawnDetector::Config ycfg;
    ycfg.mar_threshold = g_config.mar_threshold;
    g_pipeline->configureYawn(ycfg);

    MA_LOGI(TAG, "FaceMesh pipeline ready (EAR thr=%.2f, %.1fs; MAR thr=%.2f; PERCLOS warn=%.1f%% crit=%.1f%%)",
            dcfg.ear_threshold, dcfg.ear_continuous_sec,
            ycfg.mar_threshold, dcfg.perclos_warning_pct, dcfg.perclos_critical_pct);
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

            // CPU-accessible frames (FaceMesh pipeline reads/writes pixels).
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps for inference)",
                    g_config.inference_width, g_config.inference_height, g_config.inference_fps);
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
    stream_param.width  = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps    = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
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
    mqtt_config.host  = g_config.mqtt_host;
    mqtt_config.port  = g_config.mqtt_port;
    mqtt_config.topic = g_config.mqtt_topic;

    if (!g_mqtt_publisher->init(mqtt_config)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) g_camera->stopStream();

    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    if (g_pipeline) {
        delete g_pipeline;
        g_pipeline = nullptr;
    }

    if (g_face_detector) {
        delete g_face_detector;
        g_face_detector = nullptr;
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

    auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<FaceInfo> faces = g_face_detector->detect(&frame);
    auto detect_end = std::chrono::high_resolution_clock::now();
    auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

    auto mesh_start = std::chrono::high_resolution_clock::now();
    std::vector<AnalyzedFace> analyzed = g_pipeline->processAll(&frame, faces);
    auto mesh_end = std::chrono::high_resolution_clock::now();
    auto mesh_time = std::chrono::duration_cast<std::chrono::milliseconds>(mesh_end - mesh_start).count();

    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Phase 2: edge alert — fire on rising edge of alert_active for the primary face.
    static bool s_prev_alert = false;
    if (!analyzed.empty() && analyzed.front().metrics.valid) {
        const auto& d = analyzed.front().drowsiness;
        if (d.alert_active && !s_prev_alert) {
            MA_LOGW(TAG,
                "DROWSINESS ALERT! state=%s level=%.2f perclos=%.1f%% closure=%.1fs ear=%d perclos_flag=%d yawn_flag=%d",
                d.state.c_str(), d.drowsiness_level, d.perclos_pct, d.continuous_closure_sec,
                d.drowsy_by_ear ? 1 : 0,
                d.drowsy_by_perclos ? 1 : 0,
                d.drowsy_by_yawn ? 1 : 0);
            char reason[256];
            snprintf(reason, sizeof(reason),
                "state=%s level=%.2f perclos=%.1f%% closure=%.1fs",
                d.state.c_str(), d.drowsiness_level, d.perclos_pct, d.continuous_closure_sec);
            fireLocalAlert(reason);
        } else if (!d.alert_active && s_prev_alert) {
            clearLocalAlert();
        }
        s_prev_alert = d.alert_active;
    }

    if (g_config.enable_mqtt && g_mqtt_publisher) {
        g_mqtt_publisher->publishResults(timestamp_ms, g_frame_id, analyzed,
                                          static_cast<float>(total_time),
                                          g_config.include_landmarks);
    }

    if (g_config.verbose || !analyzed.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu faces, detect=%lldms, mesh=%lldms, total=%lldms",
                g_frame_id, analyzed.size(),
                (long long)detect_time, (long long)mesh_time, (long long)total_time);
        for (const auto& af : analyzed) {
            if (af.metrics.valid) {
                MA_LOGI(TAG, "  Face[%d]: ear=%.3f (L=%.3f R=%.3f) mar=%.3f closed=%d open=%d",
                        af.face.id,
                        af.metrics.avg_ear, af.metrics.left_ear, af.metrics.right_ear,
                        af.metrics.mar,
                        af.metrics.eyes_closed ? 1 : 0,
                        af.metrics.mouth_open ? 1 : 0);
            } else {
                MA_LOGI(TAG, "  Face[%d]: landmark inference failed", af.face.id);
            }
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting FaceMesh Reader");
    MA_LOGI(TAG, "Face model:     %s", g_config.face_model.c_str());
    MA_LOGI(TAG, "FaceMesh model: %s", g_config.facemesh_model.c_str());

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!init_models())          { cleanup(); return 1; }
    if (!init_camera())          { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (!init_mqtt())            { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    if (g_config.enable_rtsp) startVideo();

    MA_LOGI(TAG, "FaceMesh reader running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:554/live");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "FaceMesh Reader terminated");
    return 0;
}
