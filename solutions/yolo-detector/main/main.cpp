#include <chrono>
#include <cstring>
#include <getopt.h>
#include <map>
#include <string>
#include <vector>

#include <af/app.h>
#include <af/ma_image.h>
#include <cvi_ae.h>
#include <cvi_isp.h>
#include <ha_mqtt.h>
#include <sscma.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"

#include "app_config.h"
#include "detector.h"
#include "http_control.h"
#include "inference_control.h"
#include "mqtt_payload.h"
#include "person_tracker.h"

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
    int inference_fps = 8;   // capture->inference channel rate; the SG2002 sustains ~4-5 inferences/s, and every channel frame costs a CPU copy (Thread_Venc0) on the single core
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;    // teleop video; the ISP processes only this many sensor frames (VPSS online_frc), so CPU scales with it: 15 fps ~9% vs 30 fps ~17% for camera+detector
    // VENC CBR target. 1000 kbps is the ipcam default this app ran with before
    // app_framework (whose StreamOptions default is 2500); with MaxQp 35 the
    // encoder already overshoots it during motion, and the stream goes over WiFi.
    int stream_bitrate_kbps = 1000;

    // Tracking configuration
    bool enable_tracking = true;
    float dwell_speed_threshold = 10.0f;
    float dwell_min_duration = 1.5f;
    float dwell_assistance_threshold = 20.0f;

    // Hardware blur (RGN COVEREX). NOTE: parsed and logged only -- no RGN
    // region is configured anywhere in this application yet.
    bool enable_blur = false;
    int max_blur_regions = 4;
    std::string blur_classes = "";  // comma-separated class IDs to blur, empty = all

    // Stream endpoints. Default: H.264 over WebSocket only (lowest latency,
    // RTSP server not started). --no-rtsp / --no-debug remove an endpoint from
    // whatever --stream-mode selected (kept for existing init scripts).
    af::StreamMode stream_mode = af::StreamMode::WsOnly;
    bool enable_debug = true;
    int debug_port = 8001;
    int rtsp_port = 8554;

    // On-demand inference, switched over HTTP (GET/POST /api/inference).
    // Off: detection, tracking and result publishing stop and the inference
    // channel drops to 1 fps; the video stream and /snapshot.jpg keep running.
    bool inference_enabled = false;
    int control_port = 8090;

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
static Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static uint32_t g_frame_id = 1;  // 0 means "no result yet" on /api/results/latest
static InferenceControl* g_inference = nullptr;
static std::chrono::steady_clock::time_point g_start_time;

// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

// COCO class names carry more detail than ONVIF's object taxonomy, so collapse
// them onto the tt:ObjectType spellings a VMS knows how to filter on. Anything
// outside these groups passes through verbatim, which the schema allows:
// tt:ClassDescriptor/Type is a free string, and a client that understands
// "traffic light" is better served by the real label than by a wrong one.
static std::string onvif_class_of(const std::string& label) {
    if (label == "person") return "Human";
    if (label == "bicycle") return "Bicycle";
    if (label == "car" || label == "bus" || label == "truck" ||
        label == "motorcycle" || label == "train" || label == "boat" ||
        label == "airplane") return "Vehicle";
    if (label == "cat" || label == "dog" || label == "bird" ||
        label == "horse" || label == "sheep" || label == "cow" ||
        label == "elephant" || label == "bear" || label == "zebra" ||
        label == "giraffe") return "Animal";
    return label;
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
    printf("  --stream-mode MODE        ws | rtsp | both | none (default: ws -- H.264 over WebSocket, no RTSP)\n");
    printf("  --no-rtsp                 Remove RTSP from the stream mode\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  --no-debug                Remove the WebSocket stream from the stream mode\n");
    printf("  --debug-port PORT         Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  --stream-bitrate KBPS     Stream H.264 CBR target in kbps (default: %d)\n", g_config.stream_bitrate_kbps);
    printf("  --stream-fps N            Stream frame rate, 1..30 (default: %d)\n", g_config.stream_fps);
    printf("  --inference on|off        Run inference at startup (default: off; toggle via HTTP)\n");
    printf("  --control-port PORT       HTTP inference control port (default: %d)\n", g_config.control_port);
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nWebSocket: ws://<device_ip>:%d/   RTSP (rtsp/both): rtsp://<device_ip>:%d/live0\n", g_config.debug_port, g_config.rtsp_port);
    printf("HTTP control: http://<device_ip>:%d/\n", g_config.control_port);
    printf("  GET  /api/inference                                  state\n");
    printf("  POST /api/inference {\"enabled\":true|false}         switch inference on/off\n");
    printf("  GET  /api/results/latest[?after=ID&wait_ms=0..5000]  latest result (long-poll with after)\n");
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
        {"stream-mode", required_argument, 0, 15},
        {"stream-bitrate", required_argument, 0, 16},
        {"inference", required_argument, 0, 17},
        {"control-port", required_argument, 0, 18},
        {"stream-fps", required_argument, 0, 19},
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
            case 15:
                if (!af::parse_stream_mode(optarg, g_config.stream_mode)) {
                    fprintf(stderr, "Invalid --stream-mode '%s' (ws|rtsp|both|none)\n", optarg);
                    return false;
                }
                break;
            case 16: g_config.stream_bitrate_kbps = std::stoi(optarg); break;
            case 17:
                if (std::strcmp(optarg, "on") == 0) {
                    g_config.inference_enabled = true;
                } else if (std::strcmp(optarg, "off") == 0) {
                    g_config.inference_enabled = false;
                } else {
                    fprintf(stderr, "Invalid --inference '%s' (on|off)\n", optarg);
                    return false;
                }
                break;
            case 18: g_config.control_port = std::stoi(optarg); break;
            case 19: g_config.stream_fps = std::max(1, std::min(30, std::stoi(optarg))); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

/** --stream-mode minus the endpoints removed by --no-rtsp / --no-debug. */
static af::StreamMode resolve_stream_mode() {
    const af::StreamMode m = g_config.stream_mode;
    const bool ws = (m == af::StreamMode::WsOnly || m == af::StreamMode::Both) && g_config.enable_debug;
    const bool rtsp = (m == af::StreamMode::RtspOnly || m == af::StreamMode::Both) && g_config.enable_rtsp;
    if (ws && rtsp) return af::StreamMode::Both;
    if (rtsp) return af::StreamMode::RtspOnly;
    if (ws) return af::StreamMode::WsOnly;
    return af::StreamMode::None;
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

// MQTT client options + Home Assistant MQTT Discovery entity table (field
// names must match the results JSON built by mqtt_payload.cpp -- a single
// schema with detection_count/detections regardless of the tracking setting).
static ha_mqtt::ClientOptions mqtt_options() {
    ha_mqtt::ClientOptions opts;
    opts.app_id        = "yolo-detector";
    opts.client_id     = "recamera-yolo-detector";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

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
    return opts;
}

// AE defaults to slow-shutter, which drops the sensor to ~10 fps indoors
// (measured VIFPS 10 with the ISP configured for 30). Teleop needs a steady
// frame rate, so keep it fixed and let AE raise gain instead. Needs a running
// ISP, hence App::on_started().
static void set_ae_fixed_frame_rate() {
    ISP_EXPOSURE_ATTR_S exp_attr;
    std::memset(&exp_attr, 0, sizeof(exp_attr));
    if (CVI_ISP_GetExposureAttr(0, &exp_attr) == CVI_SUCCESS) {
        exp_attr.stAuto.enAEMode = AE_MODE_FIX_FRAME_RATE;
        if (CVI_ISP_SetExposureAttr(0, &exp_attr) == CVI_SUCCESS) {
            MA_LOGI(TAG, "AE mode set to fixed frame rate");
        } else {
            MA_LOGW(TAG, "Failed to set AE fixed frame rate");
        }
    }
}

// Buffers reused across frames so steady state does not reallocate.
static struct {
    std::vector<af::Box> boxes;
    std::vector<onvif_box_t> onvif_boxes;
    std::map<int, int> det_track_ids;
    std::string payload;
} g_scratch;

static void on_frame(af::App& app, const af::VideoFrame& frame, af::Result& r) {
    if (!g_inference->enabled()) {
        r.publish = false;  // no MQTT / debug /results; the snapshot is still offered
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ma_img_t img = af::to_ma_img(frame);
    std::vector<Detection> detections = g_detector->detect(&img);
    const float inference_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const float now_sec =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - g_start_time).count();
    const float W = static_cast<float>(g_config.inference_width);
    const float H = static_cast<float>(g_config.inference_height);

    // Overlay boxes for the debug /results channel (centre pixels, inference frame).
    g_scratch.boxes.clear();
    for (const auto& det : detections) {
        af::Box b;
        b.x = det.x * W;
        b.y = det.y * H;
        b.w = det.w * W;
        b.h = det.h * H;
        b.score = det.confidence;
        b.cls = det.class_id;
        b.label = Detector::getClassName(det.class_id);
        g_scratch.boxes.push_back(b);
    }
    r.inference_ms = inference_ms;
    r.set_boxes(g_scratch.boxes.data(), g_scratch.boxes.size());

    // MQTT results (single schema; tracking adds optional fields). Tracking
    // only feeds this payload, so like before it runs only with MQTT on.
    const std::map<int, int>* track_ids = nullptr;
    ha_mqtt::MqttPublisher* mqtt = app.mqtt();
    if (mqtt != nullptr) {
        g_scratch.det_track_ids.clear();
        TrackingSummary tracking;
        const bool has_tracking = g_tracker != nullptr;
        if (has_tracking) {
            for (const auto& person : g_tracker->update(detections, now_sec)) {
                g_scratch.det_track_ids[person.detection.id] = person.track_id;
            }
            track_ids = &g_scratch.det_track_ids;
            tracking.active_tracks = g_tracker->getTrackCount();
            if (g_tracker->hasEntryLine()) {
                auto line_crossing = g_tracker->getLineCrossing();
                tracking.has_line = true;
                tracking.line_in = line_crossing.in_count;
                tracking.line_out = line_crossing.out_count;
            }
        }
        g_scratch.payload = buildResultJson(r.wall_ms, g_frame_id, detections, inference_ms,
                                            has_tracking ? &g_scratch.det_track_ids : nullptr,
                                            has_tracking ? &tracking : nullptr);
        r.json = g_scratch.payload;  // published by the App's MQTT sink

        // The same detections in ONVIF's analytics representation, on its own
        // topic and rate limit. The recamera/<app>/detections contract above
        // is consumed by SenseCraft and does not change.
        if (g_onvif_meta.take(r.wall_ms)) {
            g_scratch.onvif_boxes.clear();
            for (const auto& b : g_scratch.boxes) {
                g_scratch.onvif_boxes.push_back({b.x, b.y, b.w, b.h, b.score, b.label});
            }
            mqtt->publishText(g_onvif_meta.topic(),
                              onvif_meta_to_json(onvif_meta_from_boxes(
                                  r.wall_ms, "YoloDetector", g_config.inference_width,
                                  g_config.inference_height, g_scratch.onvif_boxes, onvif_class_of)));
        }

        if (g_config.verbose || !detections.empty()) {
            if (has_tracking) {
                MA_LOGI(TAG, "Frame %u: %zu detections, %d active tracks, inference=%.0fms",
                        g_frame_id, detections.size(), tracking.active_tracks, inference_ms);
            } else {
                MA_LOGI(TAG, "Frame %u: %zu detections, inference=%.0fms",
                        g_frame_id, detections.size(), inference_ms);
            }
        }
    }

    InferenceResult latest;
    latest.frame_id = g_frame_id;
    latest.ts_ms = r.wall_ms;
    latest.inference_ms = inference_ms;
    latest.detections_json = buildDetectionsJson(detections, track_ids);
    g_inference->publish(latest);
    g_frame_id++;
}

int main(int argc, char** argv) {
    // Console-managed config first, then CLI flags (CLI wins on conflicts).
    apply_app_config();
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting YOLO Detector (universal)");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence threshold: %.2f", g_config.conf_threshold);
    g_start_time = std::chrono::steady_clock::now();

    if (!init_detector() || !init_tracker()) {
        delete g_tracker;
        delete g_detector;
        return 1;
    }

    af::App app("yolo-detector");
    app.with_camera({static_cast<uint16_t>(g_config.inference_width),
                     static_cast<uint16_t>(g_config.inference_height),
                     static_cast<uint8_t>(g_config.inference_fps), false});
    af::StreamOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);
    stream.bitrate_kbps = static_cast<uint32_t>(g_config.stream_bitrate_kbps);
    app.with_stream_mode(resolve_stream_mode(), stream, g_config.debug_port, g_config.rtsp_port);
    if (g_config.enable_mqtt) {
        const ha_mqtt::ClientOptions opts = mqtt_options();
        app.with_mqtt(opts);
        // The ONVIF switch lives in /userdata/local/onvif.conf, written by the
        // console; metadata rides the MQTT connection above.
        g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
        if (g_onvif_meta.enabled()) {
            MA_LOGI(TAG, "ONVIF metadata: %s (every %ums)",
                    g_onvif_meta.topic().c_str(), g_onvif_meta.config().interval_ms);
        }
    }
    const OnvifMetaConfig& oc = g_onvif_meta.config();
    if (oc.service_enabled && g_config.enable_rtsp && app.stream_mode() == af::StreamMode::WsOnly) {
        // ONVIF switched on in the console: it needs RTSP, so publish both.
        MA_LOGI(TAG, "ONVIF enabled: adding RTSP to the WebSocket stream");
        app.with_stream_mode(af::StreamMode::Both, stream, g_config.debug_port, g_config.rtsp_port);
    }
    if (oc.service_enabled) {
        af::OnvifOptions onvif;
        onvif.port = oc.service_port;
        onvif.serial = ha_mqtt::readDeviceIdentifier();
        onvif.device_name = "reCamera";
        onvif.username = oc.username;
        onvif.password = oc.password;
        onvif.location = oc.location;
        app.with_onvif(onvif);
    }

    InferenceControl inference(g_config.inference_fps, g_config.inference_enabled,
                               [&app](bool on, int fps) {
                                   app.set_camera_fps(static_cast<uint8_t>(fps));
                                   MA_LOGI(TAG, "Inference %s (inference channel %d fps)", on ? "ON" : "OFF", fps);
                               });
    g_inference = &inference;
    if (!g_config.inference_enabled) {
        app.set_camera_fps(InferenceControl::kIdleFps);  // applied before the first frame
    }
    MA_LOGI(TAG, "Inference at startup: %s (%d fps)", g_config.inference_enabled ? "ON" : "OFF",
            inference.current_fps());
    HttpControlServer control(inference);
    if (!control.start(g_config.control_port)) {
        MA_LOGE(TAG, "HTTP control unavailable on port %d; inference stays %s", g_config.control_port,
                g_config.inference_enabled ? "ON" : "OFF");
    }

    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());
    MA_LOGI(TAG, "Tracking: %s | Blur: %s",
            g_config.enable_tracking ? "ON" : "OFF",
            g_config.enable_blur ? "ON" : "OFF");

    app.on_started(set_ae_fixed_frame_rate)
       .on_frame([&app](const af::VideoFrame& f, af::Result& r) { on_frame(app, f, r); });
    const int rc = app.run();
    control.stop();
    g_inference = nullptr;

    delete g_tracker;
    delete g_detector;
    MA_LOGI(TAG, "YOLO Detector terminated");
    return rc;
}
