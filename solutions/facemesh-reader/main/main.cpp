#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>
#include <algorithm>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"
#include "onvif_service_bringup.h"
#include "rtsp_server.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ha_mqtt.h>

#include "face_detector.h"
#include "facemesh_pipeline.h"
#include "mqtt_payload.h"
#include "drowsiness_detector.h"
#include "yawn_detector.h"
#include "local_alert.h"

using namespace ma;
using namespace facemesh_reader;

#define TAG "facemesh-reader"

// Default configuration
static struct {
    // Model paths
    std::string face_model     = "/userdata/local/models/yolov8n_face_cv181x_int8.cvimodel";
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
    bool enable_debug       = true;   // H.264-over-WS + results JSON for supervisor console
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
static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static Camera*           g_camera        = nullptr;
static uint32_t          g_frame_id      = 0;

// Alert edge tracking shared between the local alert and the HA snapshot.
static bool g_prev_alert = false;

// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

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

            // Apply the inference frame rate (else the channel defaults high and the
            // capture->inference FIFO fills faster than inference drains).
            value.i32 = g_config.inference_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

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

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "facemesh-reader";
    opts.client_id     = "recamera-facemesh-reader";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant MQTT Discovery entity table (field names must match the
    // results JSON built by mqtt_payload.cpp).
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::BinarySensor, "drowsiness", "Drowsiness Alert",
                     "{{ 'ON' if value_json.face_count > 0 and value_json.faces[0].drowsiness.alert_active else 'OFF' }}",
                     "problem", "", ""},
        EntityConfig{EntityType::BinarySensor, "yawn", "Yawning",
                     "{{ 'ON' if value_json.face_count > 0 and value_json.faces[0].yawn.is_yawning else 'OFF' }}",
                     "", "", ""},
        EntityConfig{EntityType::BinarySensor, "occupancy", "Face Detected",
                     "{{ 'ON' if value_json.face_count > 0 else 'OFF' }}",
                     "occupancy", "", ""},
        EntityConfig{EntityType::Sensor, "perclos", "PERCLOS",
                     "{{ value_json.faces[0].drowsiness.perclos_pct | round(1) if value_json.face_count > 0 else 0 }}",
                     "", "%", "measurement"},
        EntityConfig{EntityType::Sensor, "ear", "Eye Aspect Ratio",
                     "{{ value_json.faces[0].ear | round(3) if value_json.face_count > 0 else 0 }}",
                     "", "", "measurement"},
        EntityConfig{EntityType::Image, "snapshot", "Drowsiness Snapshot",
                     "", "", "", ""},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }

    // Rides the connection above rather than opening a second one; the switch
    // lives in /userdata/local/onvif.conf, written by the console.
    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
    if (g_onvif_meta.enabled()) {
        MA_LOGI(TAG, "ONVIF metadata: %s (every %ums)",
                g_onvif_meta.topic().c_str(), g_onvif_meta.config().interval_ms);
    }

    return true;
}

static void cleanup() {
    if (g_camera) g_camera->stopStream();

    if (g_config.enable_debug) debug_stream_destroy();

    onvif_service_stop();

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

// Assemble the debug /results envelope (shared debug_stream builder).
// Face bbox is normalized (x,y,w,h in 0-1, x/y are top-left) -> scale to
// inference pixels as center-based boxes; label = drowsiness state + EAR.
static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<AnalyzedFace>& faces,
                                            float inference_time_ms) {
    std::vector<debug_stream_box_t> boxes;
    std::vector<std::string> labels;
    boxes.reserve(faces.size());
    labels.reserve(faces.size());
    for (const auto& af : faces) {
        const auto& f = af.face;
        // FaceInfo.x/y is already the box CENTER (face_detector.cpp passes the
        // model's center xy through without converting to top-left), so use it
        // directly — do NOT add w/2 (that would shift the box down-right by
        // half its size). Mirrors yolo-detector, not face-analysis.
        boxes.push_back({f.x * g_config.inference_width,
                         f.y * g_config.inference_height,
                         f.w * g_config.inference_width,
                         f.h * g_config.inference_height,
                         f.score, "face"});
        char lbl[80];
        snprintf(lbl, sizeof(lbl), "%s EAR %.2f",
                 af.drowsiness.state.c_str(), af.metrics.avg_ear);
        labels.push_back(lbl);
    }
    debug_stream_letterbox_to_display(boxes, g_config.inference_width, g_config.inference_height,
                                      g_config.stream_width, g_config.stream_height);
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      boxes, &labels);
}

// HA snapshot: on a drowsiness alert_active edge, publish an annotated JPEG to
// the retained snapshot topic (QoS1, wait for PUBACK) before the state JSON so
// the HA image entity shows the frame that triggered the state change.
// Per-state 10 s cooldown; enabled only in HA mode.
static bool snapshot_cooldown_ok(bool new_state) {
    static std::chrono::steady_clock::time_point s_last[2];  // [0]=clear, [1]=alert
    const auto now = std::chrono::steady_clock::now();
    const int idx = new_state ? 1 : 0;
    if (s_last[idx].time_since_epoch().count() != 0 &&
        std::chrono::duration_cast<std::chrono::seconds>(now - s_last[idx]).count() < 10) {
        return false;
    }
    s_last[idx] = now;
    return true;
}

static void publish_alert_snapshot(const ::cv::Mat& rgb, const AnalyzedFace& af) {
    if (!g_mqtt_publisher) return;

    ::cv::Mat annotated = rgb;  // rgb is already our private copy

    // FaceInfo.x/y is the box CENTER (normalized 0-1) — same rule as
    // build_debug_results_json: convert to top-left only for drawing.
    const auto& f = af.face;
    const float cx = f.x * annotated.cols;
    const float cy = f.y * annotated.rows;
    const float bw = f.w * annotated.cols;
    const float bh = f.h * annotated.rows;
    ::cv::Rect box(static_cast<int>(cx - bw / 2.f), static_cast<int>(cy - bh / 2.f),
                 static_cast<int>(bw), static_cast<int>(bh));
    box &= ::cv::Rect(0, 0, annotated.cols, annotated.rows);
    if (box.area() > 0) {
        ::cv::rectangle(annotated, box, ::cv::Scalar(255, 64, 64), 2);
        char lbl[80];
        snprintf(lbl, sizeof(lbl), "%s EAR %.2f", af.drowsiness.state.c_str(), af.metrics.avg_ear);
        const int ty = std::max(box.y - 6, 14);
        ::cv::putText(annotated, lbl, ::cv::Point(box.x, ty), ::cv::FONT_HERSHEY_SIMPLEX,
                    0.5, ::cv::Scalar(255, 64, 64), 1);
    }

    ::cv::Mat bgr;
    ::cv::cvtColor(annotated, bgr, ::cv::COLOR_RGB2BGR);
    ::cv::Mat resized;
    ::cv::resize(bgr, resized, ::cv::Size(640, 360));

    std::vector<uchar> jpeg;
    if (!::cv::imencode(".jpg", resized, jpeg, {::cv::IMWRITE_JPEG_QUALITY, 80})) {
        MA_LOGW(TAG, "Snapshot JPEG encode failed");
        return;
    }

    // QoS1 retained; block (max 2 s) for PUBACK so the retained image is on the
    // broker before the state JSON that references it goes out.
    if (!g_mqtt_publisher->publishBinary(g_mqtt_publisher->snapshotTopic(),
                                         jpeg.data(), jpeg.size(),
                                         1, true, 2000)) {
        MA_LOGW(TAG, "Snapshot publish failed or PUBACK timed out");
    } else {
        MA_LOGI(TAG, "Snapshot published (%zu bytes) to %s",
                jpeg.size(), g_mqtt_publisher->snapshotTopic().c_str());
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

    auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<FaceInfo> faces = g_face_detector->detect(&frame);
    auto detect_end = std::chrono::high_resolution_clock::now();
    auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

    auto mesh_start = std::chrono::high_resolution_clock::now();
    std::vector<AnalyzedFace> analyzed = g_pipeline->processAll(&frame, faces);
    auto mesh_end = std::chrono::high_resolution_clock::now();
    auto mesh_time = std::chrono::duration_cast<std::chrono::milliseconds>(mesh_end - mesh_start).count();

    // HA snapshot: detect the alert_active edge BEFORE returning the frame so
    // we can copy the RGB pixels for the annotated snapshot (frame data is
    // invalid after returnFrame). Snapshot only in HA mode, per-state cooldown.
    ::cv::Mat snapshot_rgb;
    bool have_snapshot = false;
    if (g_mqtt_publisher && g_mqtt_publisher->haEnabled() &&
        !analyzed.empty() && analyzed.front().metrics.valid) {
        const bool alert_now = analyzed.front().drowsiness.alert_active;
        if (alert_now != g_prev_alert && snapshot_cooldown_ok(alert_now)) {
            ::cv::Mat wrap(frame.height, frame.width, CV_8UC3, frame.data);
            snapshot_rgb = wrap.clone();
            have_snapshot = true;
        }
    // Offer the raw frame for /snapshot.jpg. Returns after one atomic load
    // unless a snapshot client asked recently; must precede returnFrame(),
    // after which frame.data is invalid. Raw, not annotated: the video path
    // is unannotated too and overlays are drawn client-side.
    debug_stream_offer_snapshot(frame.data, frame.width, frame.height);

    }

    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Phase 2: edge alert — fire on rising edge of alert_active for the primary face.
    bool& s_prev_alert = g_prev_alert;
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

    // Push the same result to debug WS clients (sscma-node format).
    // debug_stream is lazy: skip building JSON when nobody is connected.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        std::string dj = build_debug_results_json(timestamp_ms, g_frame_id, analyzed,
                                                  static_cast<float>(total_time));
        debug_stream_publish_result(dj.c_str(), dj.size());
    }

    if (g_config.enable_mqtt && g_mqtt_publisher) {
        // Snapshot first (retained, PUBACK-confirmed) so the HA image entity
        // already holds the triggering frame when the state JSON arrives.
        if (have_snapshot) {
            publish_alert_snapshot(snapshot_rgb, analyzed.front());
        }
        std::string payload = buildResultJson(timestamp_ms, g_frame_id, analyzed,
                                              static_cast<float>(total_time),
                                              g_config.include_landmarks);
        g_mqtt_publisher->publishResultsJson(payload);

        // Additionally publish the same faces in ONVIF's analytics
        // representation, on its own topic and its own rate limit. The
        // recamera/<app>/results contract above is consumed by SenseCraft and
        // does not change. Only boxes and a class here; the drowsiness state
        // has no ONVIF vocabulary and would need a vendor extension.
        if (g_onvif_meta.take(timestamp_ms)) {
            std::vector<onvif_box_t> ob;
            ob.reserve(analyzed.size());
            for (const auto& af : analyzed) {
                const auto& f = af.face;
                // FaceInfo.x/y is already the box centre (see the note in
                // build_debug_results_json), which is what onvif_box_t wants.
                ob.push_back({f.x * g_config.inference_width,
                              f.y * g_config.inference_height,
                              f.w * g_config.inference_width,
                              f.h * g_config.inference_height,
                              f.score, "HumanFace"});
            }
            g_mqtt_publisher->publishText(
                g_onvif_meta.topic(),
                onvif_meta_to_json(onvif_meta_from_boxes(
                    timestamp_ms, "FacemeshReader",
                    g_config.inference_width, g_config.inference_height, ob)));
        }
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
    // Debug stream (non-fatal: on failure run without it)
    if (g_config.enable_debug && debug_stream_start_or_disable(8001, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt())            { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    if (g_config.enable_rtsp) startVideo();

    // ONVIF discovery + Device/Media2 services. After startVideo() on purpose:
    // GetProfiles and GetStreamUri are answered from the running RTSP server's
    // session list, so bringing this up earlier advertises zero profiles and a
    // VMS shows a camera with no video.
    if (onvif_service_bringup(g_onvif_meta.config(), ha_mqtt::readDeviceIdentifier(),
            "reCamera", g_config.enable_debug ? 8001 : 0) == 0 &&
        onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

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
