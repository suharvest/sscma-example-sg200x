#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <getopt.h>

#include <sscma.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include <af/app.h>

#include "motion_config.h"
#include "motion_detector.h"

using namespace ma;
using namespace region_motion;

#define TAG "region-motion"

static MotionConfig g_config;
static uint32_t g_frame_id = 0;

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

static ha_mqtt::ClientOptions mqtt_options() {
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
    return opts;
}

// Result points at these after on_frame returns (sinks publish after the
// frame is released), so they live outside the handler and are reused.
static struct {
    MotionResult res;
    std::vector<af::Box> boxes;
    std::string payload;
    std::string debug_extra;
} g_scratch;

static void on_frame(af::App& app, MotionDetector& detector, bool& first_mqtt_publish,
                     const af::VideoFrame& frame, af::Result& r) {
    g_frame_id++;
    r.seq = g_frame_id;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Run motion detector on RGB frame
    g_scratch.res = detector.process(frame.planes[0].vir, frame.width, frame.height);
    const MotionResult& res = g_scratch.res;

    auto t1 = std::chrono::high_resolution_clock::now();
    r.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // HA MQTT publication on state change, first start, or periodic heartbeat (every 50 frames ~ 5s)
    if (g_config.enable_mqtt && app.mqtt()) {
        if (res.state_changed || first_mqtt_publish || (g_frame_id % (g_config.inference_fps * 5) == 0)) {
            first_mqtt_publish = false;
            nlohmann::json payload;
            payload["state"] = res.motion_detected ? "ON" : "OFF";
            payload["motion"] = res.motion_detected;
            payload["ratio"] = std::round(res.motion_ratio * 1000.0f) / 1000.0f;
            payload["frame_id"] = g_frame_id;
            g_scratch.payload = payload.dump();
            r.json = g_scratch.payload;  // published by the App's MQTT sink
        }
    }

    // WebSocket debug results: the App's debug sink maps the boxes from
    // inference to stream resolution and builds the envelope.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        g_scratch.boxes.clear();
        for (const auto& db : res.boxes) {
            af::Box b;
            b.x = db.x;
            b.y = db.y;
            b.w = db.w;
            b.h = db.h;
            b.score = db.score;
            b.label = db.label.c_str();
            g_scratch.boxes.push_back(b);
        }
        r.set_boxes(g_scratch.boxes.data(), g_scratch.boxes.size());

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

        g_scratch.debug_extra = extra.str();
        r.debug_extra = g_scratch.debug_extra;
    }
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

    MA_LOGI(TAG, "Starting Region Motion Detection (sensitivity: %d, min_ratio: %.3f, debounce: %d, roi_pts: %zu)",
            g_config.pixel_threshold, g_config.min_motion_area, g_config.debounce_frames, g_config.roi_zone.size());

    af::App app("region-motion");
    app.with_camera({static_cast<uint16_t>(g_config.inference_width),
                     static_cast<uint16_t>(g_config.inference_height),
                     static_cast<uint8_t>(g_config.inference_fps), false});

    af::RtspOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);
    app.with_stream(stream);
    if (g_config.enable_rtsp) {
        app.with_rtsp(stream);
    } else {
        MA_LOGI(TAG, "RTSP streaming disabled");
    }
    if (g_config.enable_debug) {
        app.with_debug_ws(g_config.debug_port);
    }
    if (g_config.enable_mqtt) {
        app.with_mqtt(mqtt_options());
    } else {
        MA_LOGI(TAG, "MQTT publishing disabled");
    }

    MotionDetector detector(g_config);
    bool first_mqtt_publish = true;

    MA_LOGI(TAG, "Region motion detector running (inference: %dx%d, stream: %dx%d)",
            g_config.inference_width, g_config.inference_height,
            g_config.stream_width, g_config.stream_height);

    app.on_frame([&](const af::VideoFrame& f, af::Result& r) { on_frame(app, detector, first_mqtt_publish, f, r); });
    const int rc = app.run();

    MA_LOGI(TAG, "Cleanup completed");
    return rc;
}
