#include <chrono>
#include <getopt.h>
#include <string>
#include <vector>

#include <sscma.h>
#include <debug_stream.h>
#include "rtsp_server.h"

#include <sstream>

#include <ha_mqtt.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"
#include "onvif_service.h"

#include "af/app.h"
#include "af/ma_image.h"

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
    // 0 keeps the historical behaviour: one message per processed frame.
    // Set it when several cameras share a broker — the message rate there is
    // people x cameras x frame rate, which is what saturates first.
    int mqtt_interval_ms = 0;
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

static Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static ZoneMetrics* g_zone_metrics = nullptr;
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
    printf("  --mqtt-interval-ms MS         Min ms between result publishes, 0 = every frame (default: %d)\n", g_config.mqtt_interval_ms);
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
        {"mqtt-interval-ms", required_argument, 0, 19},
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
            case 19:  g_config.mqtt_interval_ms = std::stoi(optarg); break;
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

static ha_mqtt::ClientOptions mqtt_options() {
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
    return opts;
}

static void init_onvif_config() {
    /* Load independently of MQTT. The RTSP metadata track and SOAP service are
     * useful when no broker is configured, and both must know the switch before
     * the RTSP options (session SDP) are built. */
    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), "retail-vision");
    if (g_onvif_meta.enabled()) {
        MA_LOGI(TAG, "ONVIF metadata enabled (profile=%s, every %ums)",
                g_onvif_meta.config().profile.c_str(),
                g_onvif_meta.config().interval_ms);
    }
}

// Runs from App::on_started(): RGN regions attach to a live VPSS channel.
static bool init_blur() {
    PrivacyBlurConfig cfg;
    loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, nullptr);
    if (!cfg.present) cfg.enabled = false;
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

    MA_LOGI(TAG, "Privacy blur ready (backend=%s, max_regions=%d, enabled=%d)",
            cfg.backend.c_str(), cfg.max_regions, (int)cfg.enabled);
    return true;
}

// Runs from App::on_stopping(): detach the RGN masks while VPSS still exists.
static void deinit_blur() {
    if (g_privacy_blur) {
        g_privacy_blur->deinit();
        delete g_privacy_blur;
        g_privacy_blur = nullptr;
    }
}

static void on_started() {
    // Non-fatal, because unmasked analytics still beats no analytics.
    if (!init_blur()) {
        MA_LOGW(TAG, "Privacy blur initialization failed, continuing without it");
    }

    if (g_config.enable_rtsp) {
        // Asked of the server rather than assembled here. Hand-assembled URLs are
        // how ":554" stayed wrong across eight applications.
        char url[192];
        if (rtsp_server_url(url, sizeof(url), "<device_ip>", 0) < 0) {
            url[0] = '\0';
        }
        MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps) %s",
                g_config.stream_width, g_config.stream_height, g_config.stream_fps, url);
    } else {
        MA_LOGI(TAG, "RTSP streaming disabled");
    }

    // The App starts ONVIF after RTSP (GetProfiles/GetStreamUri read the live server).
    if (g_onvif_meta.config().service_enabled && onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Retail Vision running...");
    if (g_config.enable_rtsp) MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:%d/%s0", g_config.rtsp_port, g_config.rtsp_session_prefix.c_str());
    if (g_config.enable_mqtt) MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());
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

// Storage behind af::Result (string_views / box pointers), reused across frames.
static struct {
    std::vector<DetectionBox> detections;  // raw detections of the held frame (snapshot mask)
    std::vector<af::Box> boxes;            // debug overlay boxes
    std::vector<std::string> labels;       // "T<track_id> <dwell_state>" per box
    std::string zone_extra;                // debug envelope `zone` member
    std::string payload;                   // MQTT results payload
} g_out;

static void on_frame(af::App& app, const af::VideoFrame& frame, af::Result& r) {
    auto start_time = std::chrono::high_resolution_clock::now();
    const uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    float current_time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time
    ).count() / 1000.0f;

    ma_img_t img = af::to_ma_img(frame);

    // Detect
    g_out.detections = g_detector->detect(&img);
    const auto& detections = g_out.detections;

    // Mask the people while the frame is held: the pixelating backend averages
    // the pixels it is about to hide, and the buffer is invalid once on_frame
    // returns. Fed from the raw detections rather than the tracker output so a
    // person is masked on the frame they first appear, without waiting for the
    // track to be confirmed.
    if (g_privacy_blur) {
        /* fromCenter: PersonDetector passes the model's centre xy through,
         * same as the overlay path reads it. */
        std::vector<geometry::InferBox> boxes;
        boxes.reserve(detections.size());
        for (const auto& d : detections) {
            boxes.push_back(geometry::InferBox::fromCenter(d.x, d.y, d.w, d.h, d.score));
        }
        g_privacy_blur->onDetection(
            geometry::toStream(boxes, g_config.inference_width, g_config.inference_height,
                               g_config.stream_width, g_config.stream_height),
            &img);
    }

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

    r.seq = g_frame_id;
    r.wall_ms = timestamp_ms;
    r.inference_ms = inference_time_ms;

    // Debug /results envelope (published by the App's debug sink, which maps
    // the boxes from the letterboxed inference frame into the stream frame).
    // Box x/y are centre pixels; the label is "T<track_id> <dwell_state>"
    // (rendered verbatim by the console overlay). A compact `zone` summary
    // rides along for the console's raw-message panel. Skipped when nobody is
    // connected, as before.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        g_out.boxes.resize(tracked_persons.size());
        g_out.labels.resize(tracked_persons.size());
        for (size_t i = 0; i < tracked_persons.size(); ++i) {
            const auto& p = tracked_persons[i];
            g_out.labels[i] = "T" + std::to_string(p.track_id) + " " + getDwellStateName(p.dwell_state);
            af::Box& b = g_out.boxes[i];
            b.x = p.detection.x * g_config.inference_width;
            b.y = p.detection.y * g_config.inference_height;
            b.w = p.detection.w * g_config.inference_width;
            b.h = p.detection.h * g_config.inference_height;
            b.score = p.detection.score;
            b.label = g_out.labels[i].c_str();
        }
        r.set_boxes(g_out.boxes.data(), g_out.boxes.size());

        auto zone = g_zone_metrics->getSnapshot();
        std::ostringstream zone_json;
        zone_json << "\"zone\":{"
                  << "\"occupancy\":" << zone.occupancy_count << ","
                  << "\"browsing\":" << zone.browsing_count << ","
                  << "\"engaged\":" << zone.engaged_count << ","
                  << "\"assistance\":" << zone.assist_count << ","
                  << "\"entry\":" << zone.entry_count << ","
                  << "\"exit\":" << zone.exit_count << "}";
        g_out.zone_extra = zone_json.str();
        r.debug_extra = g_out.zone_extra;
    }

    // Application-specific MQTT payload (published by the App's MQTT sink).
    ha_mqtt::MqttPublisher* mqtt = app.mqtt();
    if (mqtt != nullptr) {
        static uint64_t s_last_publish_ms = 0;
        bool due = g_config.mqtt_interval_ms <= 0 ||
                   timestamp_ms - s_last_publish_ms >=
                       static_cast<uint64_t>(g_config.mqtt_interval_ms);
        if (due) {
            s_last_publish_ms = timestamp_ms;
            auto zone = g_zone_metrics->getSnapshot();
            g_out.payload = buildVisionJson(
                timestamp_ms, g_frame_id, g_fps, inference_time_ms,
                zone, tracked_persons,
                g_config.stream_width, g_config.stream_height,
                g_detector->getInputWidth(), g_detector->getInputHeight());
            r.json = g_out.payload;
        }
    }

    /* Build the standard analytics frame once and fan it out to both ONVIF
     * transports. This is intentionally outside the MQTT gate: a VMS/Frigate
     * RTSP consumer must work on a camera with MQTT disabled. Stable tracker
     * ids become tt:Object/@ObjectId so downstream can follow one person. */
    const bool send_onvif_rtsp = rtsp_server_metadata_enabled();
    const bool send_onvif_mqtt = mqtt != nullptr && g_onvif_meta.take(timestamp_ms);
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
            mqtt->publishText(g_onvif_meta.topic(), onvif_meta_to_json(f));
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

// App::on_snapshot(): /snapshot.jpg is encoded from the inference frame, which
// the RGN mask (VPSS->VENC path) never touches. Mask the same people the stream
// masks; detections are normalised against this buffer, so no letterbox mapping.
static void mask_snapshot(const af::VideoFrame& frame) {
    if (g_privacy_blur == nullptr || !g_privacy_blur->enabled() || g_out.detections.empty()) {
        return;
    }
    std::vector<geometry::InferBox> snap_boxes;
    snap_boxes.reserve(g_out.detections.size());
    for (const auto& d : g_out.detections) {
        snap_boxes.push_back(geometry::InferBox::fromCenter(d.x, d.y, d.w, d.h, d.score));
    }
    privacy_blur::pixelateRgb888(frame.planes[0].vir, frame.width, frame.height, snap_boxes,
                                 g_privacy_blur->blockPx());
}

static void release_models() {
    delete g_zone_metrics;
    g_zone_metrics = nullptr;
    delete g_tracker;
    g_tracker = nullptr;
    delete g_detector;
    g_detector = nullptr;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    // CLI args first; the supervisor config sidecar (if present) overrides them.
    apply_app_config();

    MA_LOGI(TAG, "Starting Retail Vision - People Flow Analytics");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence: %.2f", g_config.conf_threshold);

    g_start_time = std::chrono::steady_clock::now();
    g_fps_last_time = g_start_time;

    // Before the RTSP options: the metadata track is part of the session SDP.
    init_onvif_config();

    if (!init_detector() || !init_tracker()) {
        release_models();
        return 1;
    }

    af::CameraOptions cam;
    cam.width = static_cast<uint16_t>(g_config.inference_width);
    cam.height = static_cast<uint16_t>(g_config.inference_height);
    // Apply a frame rate to the inference channel (else it defaults to 30fps
    // and the capture->inference FIFO fills faster than inference drains).
    // retail has no separate inference_fps; reuse stream_fps.
    cam.fps = static_cast<uint8_t>(g_config.stream_fps);

    af::RtspOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);
    stream.port = g_config.rtsp_port;
    stream.session_prefix = g_config.rtsp_session_prefix;
    stream.username = g_config.rtsp_user;
    stream.password = g_config.rtsp_pass;
    stream.metadata = g_onvif_meta.enabled();

    af::App app("retail-vision");
    app.with_camera(cam).with_stream(stream);
    if (g_config.enable_rtsp) app.with_rtsp(stream);
    if (g_config.enable_debug) app.with_debug_ws(g_config.debug_port);
    if (g_config.enable_mqtt) {
        app.with_mqtt(mqtt_options());
    } else {
        MA_LOGI(TAG, "MQTT publishing disabled");
    }
    const OnvifMetaConfig& oc = g_onvif_meta.config();
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

    app.on_started(on_started)
       .on_stopping(deinit_blur)
       .on_snapshot(mask_snapshot)
       .on_frame([&app](const af::VideoFrame& f, af::Result& r) { on_frame(app, f, r); });
    const int rc = app.run();

    deinit_blur();  // no-op unless run() failed before on_stopping
    release_models();
    MA_LOGI(TAG, "Cleanup completed");
    MA_LOGI(TAG, "Retail Vision terminated");
    return rc;
}
