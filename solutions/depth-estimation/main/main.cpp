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
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <getopt.h>
#include <string>
#include <thread>
#include <unistd.h>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "onvif_meta_gate.h"
#include "onvif_service_bringup.h"
#include "rtsp_server.h"

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

static std::atomic<bool> g_running(true);
static depth::DepthEstimator* g_estimator = nullptr;
static depth::DepthOverlay* g_overlay     = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt     = nullptr;
static Camera* g_camera                   = nullptr;
static uint32_t g_frame_id                = 0;
static int64_t g_last_mqtt_ms             = 0;
static bool g_roi_logged                  = false;

/* ONVIF analytics metadata is deliberately not published: a depth map has no
 * objects, and inventing bounding boxes for one would put fiction on a VMS
 * timeline. The gate is still read so the Device/Media2 services below get the
 * console's port, credentials and location. */
static OnvifMetaGate g_onvif_meta;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

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

static bool init_camera() {
    Device* device = Device::getInstance();
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() != ma::Sensor::Type::kCamera) continue;

        g_camera = static_cast<Camera*>(sensor);
        g_camera->init(0);

        Camera::CtrlValue value;
        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

        value.i32 = g_config.inference_fps;
        g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

        value.u16s[0] = g_config.inference_width;
        value.u16s[1] = g_config.inference_height;
        g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

        MA_LOGI(TAG, "Camera initialized (requested %dx%d @ %dfps)",
                g_config.inference_width, g_config.inference_height, g_config.inference_fps);
        return true;
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

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }

    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
    return true;
}

static void cleanup() {
    if (g_overlay) { g_overlay->deinit(); delete g_overlay; g_overlay = nullptr; }
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    onvif_service_stop();
    if (g_config.enable_rtsp) { deinitRtsp(); deinitVideo(); }
    if (g_mqtt) { g_mqtt->deinit(); delete g_mqtt; g_mqtt = nullptr; }
    if (g_estimator) { delete g_estimator; g_estimator = nullptr; }
    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return;
    }

    const auto t_got_frame = std::chrono::steady_clock::now();

    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();

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

    const bool ok = g_estimator->run(&frame, roi);

    /* Offer the raw frame for /snapshot.jpg before returnFrame() invalidates
     * frame.data. Cheap unless a snapshot client asked recently. */
    debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    const int src_w = frame.width;
    const int src_h = frame.height;
    g_camera->returnFrame(frame);

    if (!ok) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return;
    }

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

    /* Debug console: no boxes -- there are no objects here -- and the depth
     * object as an extra top-level member. */
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        const std::vector<debug_stream_box_t> no_boxes;
        const std::string extra = "\"depth\":" + depth::buildDepthObject(stats);
        const std::string json  = debug_stream_build_results(
            timestamp_ms, g_frame_id, inference_ms,
            g_config.stream_width, g_config.stream_height, no_boxes, nullptr, extra);
        debug_stream_publish_result(json.c_str(), json.size());
    }

    if (g_config.enable_mqtt && g_mqtt &&
        (timestamp_ms - g_last_mqtt_ms) >= g_config.mqtt_interval_ms) {
        g_last_mqtt_ms = timestamp_ms;
        const std::string payload =
            depth::buildResultJson(timestamp_ms, g_frame_id, inference_ms, stats);
        g_mqtt->publishResultsJson(payload);
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

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!init_estimator()) { cleanup(); return 1; }
    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt()) { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    /* The overlay region attaches to the stream channel's VPSS, which only
     * exists once the stream is running -- hence after startStream(), never
     * before. A failure here is not fatal: the depth report is the product,
     * the preview is a convenience. */
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

    /* After RTSP: GetProfiles and GetStreamUri are answered from the running
     * server's session list (onvif_service_bringup.h:23-35). */
    if (onvif_service_bringup(g_onvif_meta.config(), ha_mqtt::readDeviceIdentifier(),
                              "reCamera",
                              g_config.enable_debug ? g_config.debug_port : 0) == 0 &&
        onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Depth estimation running...");
    MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT: %s (every %dms)", g_config.mqtt_topic.c_str(), g_config.mqtt_interval_ms);

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Monocular Depth Estimation terminated");
    return 0;
}
