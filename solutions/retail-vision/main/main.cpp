#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include "rtsp_server.h"

#include <sstream>

#include <ha_mqtt.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"
#include "onvif_service_bringup.h"

#include "detector.h"
#include "person_tracker.h"
#include "zone_metrics.h"
#include "mqtt_payload.h"
#include "app_config.h"
#include "privacy_blur.h"
#include "norm_box.h"

using namespace ma;
using namespace retail_vision;
using privacy_blur::PrivacyBlur;
using privacy_blur::PrivacyBlurConfig;

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
    // A PREFIX, not a full name: rtsp_server names session i "<prefix><i>", so
    // the default "live" still yields "live0" exactly as before. Renamed from
    // rtsp_session when this application moved off TransportRTSP, because
    // silently reinterpreting the old field would have turned "live0" into
    // "live00" for anyone passing it explicitly.
    std::string rtsp_session_prefix = "live";
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

    // Debug stream (H.264-over-WS + results JSON for the supervisor console)
    bool enable_debug = true;
    int debug_port = 8001;

    // Flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;
} g_config;

// Supervisor-managed per-app config (drawn zones/lines + tuned numbers).
// Absent file => defaults => behavior identical to before this mechanism.
static const char* APP_CONFIG_PATH = "/userdata/local/apps/retail-vision.config.json";
static retail_vision::AppConfig g_app_config;

static std::atomic<bool> g_running(true);
static Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static ZoneMetrics* g_zone_metrics = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;
/* Privacy mask over the people this application tracks. Null unless
 * /userdata/local/blur.conf switches it on: analytics is the product here and
 * masking is a deployment decision, so a store that never asked for it gets an
 * unaltered stream. */
static PrivacyBlur* g_privacy_blur = nullptr;
static std::chrono::steady_clock::time_point g_start_time;

// FPS tracking
static float g_fps = 0.0f;
static int g_fps_frame_count = 0;
static std::chrono::steady_clock::time_point g_fps_last_time;

// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

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
    printf("  --rtsp-session PREFIX         RTSP session name prefix; session i is <PREFIX><i> (default: %s)\n", g_config.rtsp_session_prefix.c_str());
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
    printf("  --no-debug                    Disable debug WebSocket stream\n");
    printf("  --debug-port PORT             Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose                 Verbose logging\n");
    printf("  -h, --help                    Show this help\n");
    printf("\n");
    printf("RTSP Stream: rtsp://<device_ip>:%d/%s0\n", g_config.rtsp_port, g_config.rtsp_session_prefix.c_str());
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
        {"no-debug",         no_argument,       0, 17},
        {"debug-port",       required_argument, 0, 18},
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
            case 2:   g_config.rtsp_session_prefix = optarg; break;
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
            case 17:  g_config.enable_debug = false; break;
            case 18:  g_config.debug_port = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default:  print_usage(argv[0]); return false;
        }
    }
    return true;
}

// Load the supervisor-managed config sidecar and fold the numeric overrides
// into g_config (must run before init_detector/init_tracker). Zone/line are
// kept in g_app_config and applied to the tracker in init_tracker().
static void apply_app_config() {
    if (!load_app_config(APP_CONFIG_PATH, g_app_config)) {
        MA_LOGI(TAG, "No app config at %s, using defaults", APP_CONFIG_PATH);
        return;
    }
    if (g_app_config.has_confidence) {
        g_config.conf_threshold = g_app_config.confidence;
    }
    if (g_app_config.has_dwell_engaged) {
        g_config.dwell_min_duration = g_app_config.dwell_engaged;
    }
    if (g_app_config.has_dwell_assist) {
        g_config.dwell_assistance_threshold = g_app_config.dwell_assist;
    }
    if (g_app_config.has_dwell_speed) {
        g_config.dwell_speed_threshold = g_app_config.dwell_speed;
    }
    if (g_app_config.has_window_duration) {
        g_config.window_duration = g_app_config.window_duration;
    }
    if (g_app_config.has_person_height) {
        g_config.person_height = g_app_config.person_height;
    }
    if (g_app_config.zone_enabled) {
        MA_LOGI(TAG, "Config: counting zone with %zu points", g_app_config.zone_points.size());
    }
    if (g_app_config.line_enabled) {
        MA_LOGI(TAG, "Config: entry line enabled (%s)", g_app_config.line_ab_in ? "ab_in" : "ab_out");
    }
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

    // Apply drawn spatial config (no-op when the operator hasn't drawn them).
    if (g_app_config.zone_enabled) {
        g_tracker->setCountZone(g_app_config.zone_points);
    }
    if (g_app_config.line_enabled) {
        g_tracker->setEntryLine(g_app_config.line_a, g_app_config.line_b, g_app_config.line_ab_in);
    }

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

            // Apply a frame rate to the inference channel (else it defaults to
            // 30fps and the capture->inference FIFO fills faster than inference
            // drains). retail has no separate inference_fps; reuse stream_fps.
            value.i32 = g_config.stream_fps;
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

    // rtsp_server rather than sscma-micro's TransportRTSP, which this
    // application used until the streams it produced turned out to be
    // undecodable. The old frame callback forwarded each VENC pack with its own
    // send(), so the SPS, the PPS and the slice of one frame left as three
    // separate RTP frames; a decoder then met a slice referencing a parameter
    // set that was not part of its access unit and reported "non-existing PPS 0
    // referenced" forever. rtsp_server writes all packs of a frame as one
    // CVI_RTSP_DATA, which is what the other seven applications have always
    // done. Moving over also lets ONVIF query the port and session name instead
    // of being told them.
    rtsp_server_config_t rtsp_cfg;
    rtsp_server_config_init(&rtsp_cfg);
    rtsp_cfg.port = g_config.rtsp_port;
    rtsp_cfg.session_prefix = g_config.rtsp_session_prefix.c_str();
    rtsp_cfg.username = g_config.rtsp_user.empty() ? nullptr : g_config.rtsp_user.c_str();
    rtsp_cfg.password = g_config.rtsp_pass.empty() ? nullptr : g_config.rtsp_pass.c_str();
    rtsp_cfg.ch_mask = (0x01 << VIDEO_CH2);
    rtsp_cfg.metadata_enabled = g_onvif_meta.enabled();
    if (rtsp_server_start(&rtsp_cfg) != 0) {
        MA_LOGE(TAG, "Failed to start RTSP server");
        return false;
    }

    // Debug stream registers its own consumer (idx 1); RTSP owns idx 0.
    registerVideoFrameHandler(VIDEO_CH2, 0, rtsp_server_video_handler, nullptr);

    // Asked of the server rather than assembled here. Hand-assembled URLs are
    // how ":554" stayed wrong across eight applications.
    char url[192];
    if (rtsp_server_url(url, sizeof(url), "<device_ip>", 0) < 0) {
        url[0] = '\0';
    }
    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps) %s",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps, url);
    return true;
}

// Assemble the debug /results envelope (shared debug_stream builder).
// DetectionBox x/y are normalized center coordinates; box[5] is
// "T<track_id> <dwell_state>" (rendered verbatim by the console overlay).
// A compact `zone` summary rides along for the console's raw-message panel.
static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<TrackedPerson>& persons,
                                            const ZoneSnapshot& zone,
                                            float inference_time_ms) {
    std::vector<debug_stream_box_t> boxes;
    boxes.reserve(persons.size());
    for (const auto& p : persons) {
        boxes.push_back({p.detection.x * g_config.inference_width,
                         p.detection.y * g_config.inference_height,
                         p.detection.w * g_config.inference_width,
                         p.detection.h * g_config.inference_height,
                         p.detection.score,
                         "T" + std::to_string(p.track_id) + " " + getDwellStateName(p.dwell_state)});
    }
    std::ostringstream zone_json;
    zone_json << "\"zone\":{"
              << "\"occupancy\":" << zone.occupancy_count << ","
              << "\"browsing\":" << zone.browsing_count << ","
              << "\"engaged\":" << zone.engaged_count << ","
              << "\"assistance\":" << zone.assist_count << ","
              << "\"entry\":" << zone.entry_count << ","
              << "\"exit\":" << zone.exit_count << "}";
    // The inference channel is letterboxed vs the 16:9 debug video (stream);
    // remap boxes into the stream frame so the overlay aligns with the video.
    debug_stream_letterbox_to_display(boxes, g_config.inference_width, g_config.inference_height,
                                      g_config.stream_width, g_config.stream_height);
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      boxes, nullptr, zone_json.str());
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "retail-vision";
    opts.client_id     = "recamera-retail-vision";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // ha_mqtt legacy mode has no per-app credential support; broker auth now
    // comes from /userdata/local/ha.conf (HA mode). Warn if the old CLI flags
    // were used so the operator knows they no longer apply.
    if (!g_config.mqtt_user.empty() || !g_config.mqtt_pass.empty()) {
        MA_LOGW(TAG, "--mqtt-user/--mqtt-pass are no longer applied; "
                     "configure broker credentials in /userdata/local/ha.conf");
    }

    // Home Assistant MQTT Discovery entity table (field names must match the
    // results JSON built by mqtt_payload.cpp).
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "people_count", "People Count",
                     "{{ value_json.zone.occupancy_count }}",
                     "", "", "measurement"},
        EntityConfig{EntityType::Sensor, "entered_total", "People Entered Total",
                     "{{ value_json.zone.entry_count }}",
                     "", "", "total_increasing"},
        EntityConfig{EntityType::Sensor, "exited_total", "People Exited Total",
                     "{{ value_json.zone.exit_count }}",
                     "", "", "total_increasing"},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }

    return true;
}

static void init_onvif_config() {
    /* Load independently of MQTT. The RTSP metadata track and SOAP service are
     * useful when no broker is configured, and both must know the switch before
     * init_video_streaming() creates the session SDP. */
    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), "retail-vision");
    if (g_onvif_meta.enabled()) {
        MA_LOGI(TAG, "ONVIF metadata enabled (profile=%s, every %ums)",
                g_onvif_meta.config().profile.c_str(),
                g_onvif_meta.config().interval_ms);
    }
}

static bool init_blur() {
    PrivacyBlurConfig cfg;
    loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, nullptr);
    if (!cfg.enabled) {
        MA_LOGI(TAG, "Privacy blur off (no %s or BLUR_ENABLED=0)",
                privacy_blur::PRIVACY_BLUR_CONFIG_PATH);
        return true;
    }
    if (!g_config.enable_rtsp) {
        // The mask lives in the RGN unit on the encoder's VPSS channel, so with
        // no stream there is nothing to mask.
        MA_LOGW(TAG, "Privacy blur needs RTSP streaming, skipping");
        return true;
    }

    g_privacy_blur = new PrivacyBlur();
    if (!g_privacy_blur->init(cfg, g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "Failed to initialize privacy blur");
        delete g_privacy_blur;
        g_privacy_blur = nullptr;
        return false;
    }

    MA_LOGI(TAG, "Privacy blur enabled (backend=%s, max_regions=%d)",
            cfg.backend.c_str(), cfg.max_regions);
    return true;
}

static void cleanup() {
    if (g_privacy_blur) {
        g_privacy_blur->deinit();
        delete g_privacy_blur;
        g_privacy_blur = nullptr;
    }

    if (g_camera) {
        g_camera->stopStream();
    }

    if (g_config.enable_debug) {
        debug_stream_destroy();
    }

    onvif_service_stop();

    rtsp_server_stop();

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

    // Mask the people before the frame goes back to the camera: the pixelating
    // backend averages the pixels it is about to hide, and frame.data is
    // invalid after returnFrame(). Fed from the raw detections rather than the
    // tracker output so a person is masked on the frame they first appear,
    // without waiting for the track to be confirmed.
    if (g_privacy_blur) {
        /* fromCenter: PersonDetector passes the model's centre xy through,
         * same as the overlay path a few hundred lines up reads it. */
        std::vector<geometry::InferBox> boxes;
        boxes.reserve(detections.size());
        for (const auto& d : detections) {
            boxes.push_back(geometry::InferBox::fromCenter(d.x, d.y, d.w, d.h, d.score));
        }
        g_privacy_blur->onDetection(
            geometry::toStream(boxes, g_config.inference_width, g_config.inference_height,
                               g_config.stream_width, g_config.stream_height),
            &frame);
    }

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

    // Push the same result to debug WS clients (sscma-node format).
    // debug_stream is lazy: skip building JSON when nobody is connected.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        auto zone = g_zone_metrics->getSnapshot();
        std::string debug_json = build_debug_results_json(timestamp_ms, g_frame_id,
                                                          tracked_persons, zone,
                                                          inference_time_ms);
        debug_stream_publish_result(debug_json.c_str(), debug_json.size());
    }

    // Publish the existing application-specific MQTT payload.
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        auto zone = g_zone_metrics->getSnapshot();
        std::string payload = buildVisionJson(
            timestamp_ms, g_frame_id, g_fps, inference_time_ms,
            zone, tracked_persons,
            g_config.stream_width, g_config.stream_height,
            g_detector->getInputWidth(), g_detector->getInputHeight());
        g_mqtt_publisher->publishResultsJson(payload);

    }

    /* Build the standard analytics frame once and fan it out to both ONVIF
     * transports. This is intentionally outside the MQTT gate: a VMS/Frigate
     * RTSP consumer must work on a camera with MQTT disabled. Stable tracker
     * ids become tt:Object/@ObjectId so downstream can follow one person. */
    const bool send_onvif_rtsp = rtsp_server_metadata_enabled();
    const bool send_onvif_mqtt = g_config.enable_mqtt && g_mqtt_publisher &&
                                 g_onvif_meta.take(timestamp_ms);
    if (send_onvif_rtsp || send_onvif_mqtt) {
        onvif_frame_t f;
        f.utc_ms = timestamp_ms;
        f.source = "RetailVision";
        f.frame_w = g_config.stream_width;
        f.frame_h = g_config.stream_height;
        f.objects.reserve(tracked_persons.size());

        std::vector<geometry::InferBox> inference_boxes;
        inference_boxes.reserve(tracked_persons.size());
        for (const auto& p : tracked_persons) {
            inference_boxes.push_back(geometry::InferBox::fromCenter(
                p.detection.x, p.detection.y, p.detection.w, p.detection.h,
                p.detection.score));
        }
        const std::vector<geometry::StreamBox> stream_boxes = geometry::toStream(
            inference_boxes, g_config.inference_width, g_config.inference_height,
            g_config.stream_width, g_config.stream_height);

        for (size_t i = 0; i < tracked_persons.size(); ++i) {
            const auto& p = tracked_persons[i];
            const auto& box = stream_boxes[i];
            onvif_object_t o;
            o.id = p.track_id;
            o.cx = box.cx * g_config.stream_width;
            o.cy = box.cy * g_config.stream_height;
            o.w  = box.w * g_config.stream_width;
            o.h  = box.h * g_config.stream_height;
            o.classes.push_back({"Human", p.detection.score});
            f.objects.push_back(std::move(o));
        }

        if (send_onvif_rtsp) {
            const std::string xml = onvif_meta_to_xml(f);
            if (rtsp_server_write_metadata(xml.data(), xml.size()) != 0) {
                MA_LOGW(TAG, "Failed to queue ONVIF RTSP metadata frame");
            }
        }
        if (send_onvif_mqtt) {
            g_mqtt_publisher->publishText(g_onvif_meta.topic(), onvif_meta_to_json(f));
        }
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

    // CLI args first; the supervisor config sidecar (if present) overrides them.
    apply_app_config();

    MA_LOGI(TAG, "Starting Retail Vision - People Flow Analytics");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence: %.2f", g_config.conf_threshold);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    g_start_time = std::chrono::steady_clock::now();
    g_fps_last_time = g_start_time;

    init_onvif_config();

    if (!init_detector()) { cleanup(); return 1; }
    if (!init_tracker())  { cleanup(); return 1; }
    if (!init_camera())   { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    // Debug stream (non-fatal: on failure run without it)
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt())     { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // Privacy blur after the video pipeline is up: RGN regions attach to a live
    // VPSS channel. Non-fatal, because unmasked analytics still beats no
    // analytics.
    if (!init_blur()) {
        MA_LOGW(TAG, "Privacy blur initialization failed, continuing without it");
    }

    // ONVIF discovery + Device/Media1/Media2 services. After init_video_streaming() on
    // purpose: GetProfiles and GetStreamUri are answered from the running RTSP
    // server, so bringing this up earlier advertises zero profiles and a VMS
    // shows a camera with no video.
    if (onvif_service_bringup(g_onvif_meta.config(), ha_mqtt::readDeviceIdentifier(),
            "reCamera", g_config.enable_debug ? g_config.debug_port : 0) == 0 &&
        onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Retail Vision running...");
    if (g_config.enable_rtsp) MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:%d/%s0", g_config.rtsp_port, g_config.rtsp_session_prefix.c_str());
    if (g_config.enable_mqtt) MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Retail Vision terminated");
    return 0;
}
