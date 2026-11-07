/*
 * Monocular depth estimation for reCamera.
 *
 * Runs a dense depth model (FastDepth BF16 by default) on the camera's
 * inference channel, draws the result as a small colour preview in the corner
 * of the RTSP stream, and reports relative proximity over MQTT.
 *
 * Relative, throughout. A single camera with no scale reference cannot measure
 * distance, so no part of this application -- payload, log line or document --
 * expresses a result in metres.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <getopt.h>
#include <string>

#include <af/app.h>
#include <af/ma_image.h>
#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "onvif_meta_gate.h"
#include "onvif_service.h"

#include "depth_estimator.h"
#include "depth_overlay.h"
#include "depth_payload.h"

using namespace ma;

#define TAG "depth-estimation"

static struct {
    std::string model_path = "/userdata/local/models/fastdepth_224_bf16.cvimodel";

    std::string mqtt_host  = "localhost";
    int mqtt_port          = 1883;
    std::string mqtt_topic = "recamera/depth-estimation/results";
    /* 2 Hz. The preview updates every inference frame; the broker does not
     * need 15 documents a second to drive an automation. */
    int mqtt_interval_ms   = 500;

    /* 16:9 inference channel so VPSS adds no grey bars in the first place --
     * see valid_content_roi() in depth_estimator.cpp. */
    int inference_width  = 320;
    int inference_height = 180;
    int inference_fps    = 15;
    int stream_width     = 1280;
    int stream_height    = 720;
    int stream_fps       = 15;

    bool enable_pip = true;
    int pip_width   = 320;
    int pip_height  = 180;
    int pip_margin  = 16;

    /* proximity at which a pixel counts as near, and the fraction of such
     * pixels that makes near_present true. */
    float near_threshold       = 0.75f;
    float near_ratio_threshold = 0.05f;

    bool enable_debug = true;
    int debug_port    = 8001;

    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose     = false;
} g_config;

static depth::DepthEstimator* g_estimator = nullptr;
static depth::DepthOverlay* g_overlay     = nullptr;
static uint32_t g_frame_id                = 0;
static int64_t g_last_mqtt_ms             = 0;
static bool g_roi_logged                  = false;

/* Result::json / debug_extra point here; they must outlive the handler. */
static std::string g_mqtt_payload;
static std::string g_debug_extra;

/* ONVIF analytics metadata is deliberately not published: a depth map has no
 * objects, and inventing bounding boxes for one would put fiction on a VMS
 * timeline. The gate is still read so the Device/Media2 services get the
 * console's port, credentials and location. */
static OnvifMetaGate g_onvif_meta;

static void print_usage(const char* prog) {
    printf("Monocular Depth Estimation for ReCamera\n");
    printf("Relative depth only -- results carry no absolute distance.\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH        Model path (default: %s)\n", g_config.model_path.c_str());
    printf("      --mqtt-host HOST    MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("      --mqtt-port PORT    MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("      --mqtt-topic TOPIC  MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("      --mqtt-interval MS  Minimum gap between results (default: %d)\n", g_config.mqtt_interval_ms);
    printf("      --near-threshold F  Proximity counted as near, 0..1 (default: %.2f)\n", g_config.near_threshold);
    printf("      --near-ratio F      Near pixel fraction for near_present (default: %.2f)\n", g_config.near_ratio_threshold);
    printf("      --no-pip            Disable the depth preview overlay\n");
    printf("      --pip-size WxH      Preview size (default: %dx%d)\n", g_config.pip_width, g_config.pip_height);
    printf("      --no-rtsp           Disable RTSP streaming\n");
    printf("      --no-mqtt           Disable MQTT publishing\n");
    printf("      --no-debug          Disable the debug WebSocket stream\n");
    printf("      --debug-port PORT   Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose           Enable verbose logging\n");
    printf("  -h, --help              Show this help message\n");
    printf("\nRTSP Stream: rtsp://<device_ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"mqtt-host", required_argument, 0, 1},
        {"mqtt-port", required_argument, 0, 2},
        {"mqtt-topic", required_argument, 0, 3},
        {"mqtt-interval", required_argument, 0, 4},
        {"near-threshold", required_argument, 0, 5},
        {"near-ratio", required_argument, 0, 6},
        {"no-pip", no_argument, 0, 7},
        {"pip-size", required_argument, 0, 8},
        {"no-rtsp", no_argument, 0, 9},
        {"no-mqtt", no_argument, 0, 10},
        {"no-debug", no_argument, 0, 11},
        {"debug-port", required_argument, 0, 12},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "m:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 1: g_config.mqtt_host = optarg; break;
            case 2: g_config.mqtt_port = std::stoi(optarg); break;
            case 3: g_config.mqtt_topic = optarg; break;
            case 4: g_config.mqtt_interval_ms = std::max(0, std::stoi(optarg)); break;
            case 5: g_config.near_threshold = std::stof(optarg); break;
            case 6: g_config.near_ratio_threshold = std::stof(optarg); break;
            case 7: g_config.enable_pip = false; break;
            case 8: {
                int w = 0, h = 0;
                if (sscanf(optarg, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                    g_config.pip_width  = w;
                    g_config.pip_height = h;
                } else {
                    fprintf(stderr, "Invalid --pip-size '%s' (expected WxH)\n", optarg);
                    return false;
                }
                break;
            }
            case 9: g_config.enable_rtsp = false; break;
            case 10: g_config.enable_mqtt = false; break;
            case 11: g_config.enable_debug = false; break;
            case 12: g_config.debug_port = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static bool init_estimator() {
    g_estimator = new depth::DepthEstimator();
    if (!g_estimator->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize the depth estimator");
        return false;
    }
    return true;
}

static ha_mqtt::ClientOptions mqtt_options() {
    ha_mqtt::ClientOptions opts;
    opts.app_id        = "depth-estimation";
    opts.client_id     = "recamera-depth-estimation";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    /* Three entities, all relative. "Nearest" values are proximity in [0,1]:
     * 1 is the nearest content in the frame, not a distance. */
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "near_area", "Near Area",
                     "{{ (value_json.depth.near_ratio * 100) | round(1) }}",
                     "", "%", "measurement"},
        EntityConfig{EntityType::BinarySensor, "near_presence", "Near Object",
                     "{{ 'ON' if value_json.depth.near_present else 'OFF' }}",
                     "occupancy", "", ""},
        EntityConfig{EntityType::Sensor, "center_nearest", "Center Nearest (relative)",
                     "{{ value_json.depth.zones[4] }}",
                     "", "", "measurement"},
    };
    return opts;
}

/* The overlay region attaches to the stream channel's VPSS, which only exists
 * once the stream is running -- hence App::on_started, never before. A failure
 * here is not fatal: the depth report is the product, the preview is a
 * convenience. */
static void on_started() {
    if (g_config.enable_pip && g_config.enable_rtsp) {
        g_overlay = new depth::DepthOverlay();
        if (!g_overlay->init(g_config.stream_width, g_config.stream_height,
                             g_config.pip_width, g_config.pip_height,
                             g_config.pip_margin, 0, VIDEO_CH2)) {
            MA_LOGW(TAG, "Depth preview unavailable; continuing without it");
            delete g_overlay;
            g_overlay = nullptr;
        }
    }

    if (onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Depth estimation running...");
    MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT: %s (every %dms)", g_config.mqtt_topic.c_str(), g_config.mqtt_interval_ms);
}

/* RGN regions must be detached while the VPSS channel still exists. */
static void on_stopping() {
    if (g_overlay) { g_overlay->deinit(); delete g_overlay; g_overlay = nullptr; }
}

static void on_frame(af::App& app, const af::VideoFrame& f, af::Result& r) {
    const auto t_got_frame = std::chrono::steady_clock::now();
    const int64_t timestamp_ms = static_cast<int64_t>(r.wall_ms);

    ma_img_t frame = af::to_ma_img(f);

    /* Strip the VPSS letterbox before inference, not after: grey bars are
     * out-of-distribution input for a depth model and corrupt both the depth
     * they land on and the frame-wide p02/p98 normalisation. */
    const depth::Roi roi = depth::valid_content_roi(frame.width, frame.height);
    if (!g_roi_logged) {
        MA_LOGI(TAG, "Inference frame %dx%d, valid ROI [x=%d,y=%d,w=%d,h=%d]%s",
                frame.width, frame.height, roi.x, roi.y, roi.w, roi.h,
                (roi.w == frame.width && roi.h == frame.height)
                    ? " (16:9 channel, no letterbox)"
                    : " (letterboxed channel, bars cropped)");
        g_roi_logged = true;
    }

    if (!g_estimator->run(&frame, roi)) {
        r.publish = false;  // nothing to report for this frame
        return;
    }
    const int src_w = frame.width;
    const int src_h = frame.height;

    const auto t_infer_done = std::chrono::steady_clock::now();

    const depth::DepthStats stats = depth::computeStats(
        g_estimator->depth(), g_estimator->outputWidth(), g_estimator->outputHeight(),
        roi, src_w, src_h, g_config.near_threshold, g_config.near_ratio_threshold);

    const auto t_stats_done = std::chrono::steady_clock::now();

    /* Preview on every inference frame: the whole point of it is to show what
     * the model is seeing right now. */
    if (g_overlay && g_overlay->ready()) {
        g_overlay->update(g_estimator->depth(), g_estimator->outputWidth(),
                          g_estimator->outputHeight(), stats.p02, stats.p98);
    }

    const auto t_overlay_done = std::chrono::steady_clock::now();

    const float inference_ms = g_estimator->lastInferenceMs();

    /* -v profiling: where the per-frame budget actually goes. lastInferenceMs()
     * covers preprocess+forward+readout, so the three sum to it; stats and
     * overlay sit outside it and are what separates it from the frame period. */
    if (g_config.verbose) {
        const auto ms = [](std::chrono::steady_clock::time_point a,
                           std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        MA_LOGI(TAG,
                "profile pre=%.1f fwd=%.1f read=%.1f | stats=%.1f overlay=%.1f "
                "| infer=%.1f total=%.1f",
                g_estimator->lastPreprocessMs(), g_estimator->lastForwardMs(),
                g_estimator->lastReadoutMs(), ms(t_infer_done, t_stats_done),
                ms(t_stats_done, t_overlay_done), inference_ms,
                ms(t_got_frame, t_overlay_done));
    }

    /* Debug console (framework sink, stream-dims envelope): no boxes -- there
     * are no objects here -- and the depth object as an extra top-level member. */
    r.seq          = g_frame_id;
    r.inference_ms = inference_ms;
    r.set_boxes(nullptr, 0);
    r.debug_extra = {};
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        g_debug_extra = "\"depth\":" + depth::buildDepthObject(stats);
        r.debug_extra = g_debug_extra;
    }

    /* MQTT is rate-limited: only frames past the interval carry a payload. */
    r.json = {};
    if (g_config.enable_mqtt && app.mqtt() &&
        (timestamp_ms - g_last_mqtt_ms) >= g_config.mqtt_interval_ms) {
        g_last_mqtt_ms = timestamp_ms;
        g_mqtt_payload = depth::buildResultJson(timestamp_ms, g_frame_id, inference_ms, stats);
        r.json = g_mqtt_payload;
    }

    if (g_config.verbose) {
        MA_LOGI(TAG,
                "Frame %u: near_ratio=%.3f near=%s p02=%.3f p50=%.3f p98=%.3f center=%.3f "
                "inference=%.1fms",
                g_frame_id, stats.near_ratio, stats.near_present ? "yes" : "no",
                stats.p02, stats.p50, stats.p98, stats.zones[4], inference_ms);
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting Monocular Depth Estimation");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Output is relative depth (smaller = nearer); no absolute distance");

    if (!init_estimator()) { delete g_estimator; return 1; }

    af::App app("depth-estimation");
    app.with_camera({static_cast<uint16_t>(g_config.inference_width),
                     static_cast<uint16_t>(g_config.inference_height),
                     static_cast<uint8_t>(g_config.inference_fps), false});

    af::RtspOptions stream;
    stream.width  = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps    = static_cast<uint8_t>(g_config.stream_fps);
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
        const ha_mqtt::ClientOptions opts = mqtt_options();
        app.with_mqtt(opts);
        g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
    } else {
        MA_LOGI(TAG, "MQTT publishing disabled");
    }
    const OnvifMetaConfig& oc = g_onvif_meta.config();
    if (oc.service_enabled) {
        af::OnvifOptions onvif;
        onvif.port        = oc.service_port;
        onvif.serial      = ha_mqtt::readDeviceIdentifier();
        onvif.device_name = "reCamera";
        onvif.username    = oc.username;
        onvif.password    = oc.password;
        onvif.location    = oc.location;
        app.with_onvif(onvif);
    }

    app.on_started(on_started)
       .on_stopping(on_stopping)
       .on_frame([&app](const af::VideoFrame& f, af::Result& r) { on_frame(app, f, r); });
    const int rc = app.run();

    delete g_estimator;
    g_estimator = nullptr;
    MA_LOGI(TAG, "Monocular Depth Estimation terminated");
    return rc;
}
