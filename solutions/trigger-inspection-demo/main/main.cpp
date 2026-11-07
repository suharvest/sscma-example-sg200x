// trigger-inspection-demo -- trigger-only station inspection on reCamera.
//
// The factory-station pattern on top of af::App:
//   standby  the frame thread sleeps on the trigger queue; the camera channel
//            runs at --idle-fps (default 1) so the callback copies one frame/s
//   trigger  HTTP (/trigger, answered synchronously with the verdict) or a
//            GPIO edge (PLC output, light barrier, foot switch)
//   inspect  one fresh frame -> af::Pipeline: validate -> roi -> measure -> judge
//   verdict  OK / NG / ERROR as JSON: HTTP response, MQTT, debug /results
//
// The pipeline is a deliberately simple optical check (ROI brightness and
// contrast) so the demo runs without a model; replace the measure/judge steps
// with a detector or classifier for a real station.

#include <getopt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <af/app.h>
#include <ha_mqtt.h>
#include <sscma.h>

#define TAG "trigger-inspection-demo"

static struct {
    // Camera: inference channel and its standby rate
    int width = 640;
    int height = 480;
    int active_fps = 15;
    int idle_fps = 1;

    // Preview (H.264 for the console debug page / RTSP)
    int stream_width = 640;
    int stream_height = 360;
    int stream_fps = 10;
    bool enable_rtsp = false;
    bool enable_debug = false;  // StreamMode::None by default: no VENC, no WebSocket, no RTSP; --debug / --rtsp opt in
    int debug_port = 8001;

    // Triggers
    int http_port = 8090;  // 0 disables
    int gpio_line = -1;    // sysfs GPIO number; -1 disables
    bool gpio_active_low = false;
    af::GpioTrigger::Edge gpio_edge = af::GpioTrigger::Edge::Rising;
    int gpio_debounce_ms = 50;

    // MQTT
    bool enable_mqtt = true;
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/trigger-inspection-demo/results";

    // Inspection: ROI as fractions of the frame, luma limits
    float roi[4] = {0.25f, 0.25f, 0.5f, 0.5f};
    float mean_min = 40.f;
    float mean_max = 220.f;
    float std_min = 12.f;

    bool verbose = false;
} g_config;

// ---------------------------------------------------------------------------
// Inspection pipeline
// ---------------------------------------------------------------------------

struct Inspection {
    af::Rect roi;            // pixels, set by the roi step
    float mean = 0.f;        // ROI luma mean (0..255)
    float stddev = 0.f;      // ROI luma standard deviation
    uint32_t samples = 0;
};

static af::Flow step_validate(af::Context<Inspection>& c) {
    if (!c.frame.valid() || c.frame.format != af::PixelFormat::RGB888) return c.error("unsupported_frame");
    if (!c.frame.cpu_accessible()) return c.error("frame_not_cpu_accessible");
    return af::Flow::Next;
}

static af::Flow step_roi(af::Context<Inspection>& c) {
    const float W = c.frame.width, H = c.frame.height;
    c.state.roi = af::Rect{static_cast<int32_t>(g_config.roi[0] * W), static_cast<int32_t>(g_config.roi[1] * H),
                           static_cast<int32_t>(g_config.roi[2] * W), static_cast<int32_t>(g_config.roi[3] * H)};
    c.frame = c.frame.sub_frame(c.state.roi);  // zero-copy crop for the steps below
    return c.frame.valid() ? af::Flow::Next : c.error("roi_empty");
}

static af::Flow step_measure(af::Context<Inspection>& c) {
    // Every second pixel of every second row: 4x fewer reads, plenty for a mean/stddev.
    uint64_t sum = 0, sum_sq = 0;
    uint32_t n = 0;
    for (int y = 0; y < c.frame.height; y += 2) {
        const uint8_t* p = c.frame.row(y);
        for (int x = 0; x < c.frame.width; x += 2, p += 6) {
            const uint32_t l = (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;  // BT.601 luma
            sum += l;
            sum_sq += l * l;
            ++n;
        }
    }
    if (n == 0) return c.error("roi_empty");
    const double mean = static_cast<double>(sum) / n;
    c.state.mean = static_cast<float>(mean);
    c.state.stddev = static_cast<float>(std::sqrt(std::max(0.0, static_cast<double>(sum_sq) / n - mean * mean)));
    c.state.samples = n;
    return af::Flow::Next;
}

static af::Flow step_judge(af::Context<Inspection>& c) {
    if (c.state.mean < g_config.mean_min) return c.reject("too_dark");
    if (c.state.mean > g_config.mean_max) return c.reject("too_bright");
    if (c.state.stddev < g_config.std_min) return c.reject("low_contrast");
    return af::Flow::Next;
}

// ---------------------------------------------------------------------------
// Result document
// ---------------------------------------------------------------------------

/** Append @p s as a JSON string literal (the HTTP tag is caller-controlled). */
static void append_json_string(std::string& out, const char* s) {
    out += '"';
    for (; *s; ++s) {
        const unsigned char ch = static_cast<unsigned char>(*s);
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += static_cast<char>(ch);
        } else if (ch < 0x20) {
            char esc[8];
            std::snprintf(esc, sizeof(esc), "\\u%04x", ch);
            out += esc;
        } else {
            out += static_cast<char>(ch);
        }
    }
    out += '"';
}

static const char* verdict_of(af::Flow f) {
    return f == af::Flow::Next ? "OK" : f == af::Flow::Reject ? "NG" : "ERROR";
}

static void build_result(std::string& out, const af::TriggerEvent& ev, const af::Result& r, const Inspection& st,
                         const af::Pipeline<Inspection>& p, const af::RunReport& rep) {
    char buf[256];
    out.clear();
    std::snprintf(buf, sizeof(buf), "{\"timestamp\":%llu,\"frame_id\":%u,\"verdict\":\"%s\",\"reason\":",
                  static_cast<unsigned long long>(r.wall_ms), r.seq, verdict_of(rep.flow));
    out += buf;
    if (rep.reason) {
        append_json_string(out, rep.reason);
    } else {
        out += "null";
    }
    std::snprintf(buf, sizeof(buf), ",\"trigger\":{\"source\":\"%s\",\"seq\":%u,\"tag\":", af::to_string(ev.source), ev.seq);
    out += buf;
    append_json_string(out, ev.tag);
    std::snprintf(buf, sizeof(buf),
                  "},\"metrics\":{\"mean\":%.1f,\"stddev\":%.1f,\"samples\":%u},\"roi\":[%d,%d,%d,%d],\"resolution\":[%u,%u],\"timing_us\":{",
                  st.mean, st.stddev, st.samples, st.roi.x, st.roi.y, st.roi.w, st.roi.h, r.width, r.height);
    out += buf;
    for (size_t i = 0; i < p.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "\"%s\":%u,", p.name(i), rep.step_us[i]);
        out += buf;
    }
    std::snprintf(buf, sizeof(buf), "\"total\":%u}}", rep.total_us);
    out += buf;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    printf("Trigger-only inspection demo for reCamera\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Triggers:\n");
    printf("  --http-port PORT       HTTP trigger port, 0 = off (default: %d); GET/POST /trigger[?tag=ID]\n", g_config.http_port);
    printf("  --gpio N               sysfs GPIO number of the trigger input (default: off)\n");
    printf("  --gpio-edge E          rising | falling | both (default: rising)\n");
    printf("  --gpio-active-low      Input is active low (button to GND with pull-up)\n");
    printf("  --gpio-debounce MS     Debounce interval (default: %d)\n", g_config.gpio_debounce_ms);
    printf("Camera:\n");
    printf("  --size WxH             Inspection frame size (default: %dx%d)\n", g_config.width, g_config.height);
    printf("  --active-fps N         Channel rate while capturing (default: %d)\n", g_config.active_fps);
    printf("  --idle-fps N           Channel rate in standby, 1..active (default: %d)\n", g_config.idle_fps);
    printf("Inspection:\n");
    printf("  --roi x,y,w,h          ROI as frame fractions (default: %.2f,%.2f,%.2f,%.2f)\n", g_config.roi[0], g_config.roi[1],
           g_config.roi[2], g_config.roi[3]);
    printf("  --mean-min F / --mean-max F   ROI luma mean limits (default: %.0f / %.0f)\n", g_config.mean_min, g_config.mean_max);
    printf("  --std-min F            Minimum ROI luma stddev (default: %.0f)\n", g_config.std_min);
    printf("Outputs:\n");
    printf("  --mqtt-host HOST / --mqtt-port PORT / --mqtt-topic TOPIC\n");
    printf("  --no-mqtt              Disable MQTT\n");
    printf("  --rtsp                 Publish the preview over RTSP (default: off)\n");
    printf("  --debug                Serve the WebSocket preview + /snapshot.jpg (default: off)\n");
    printf("  --no-debug             No WebSocket preview (default)\n");
    printf("  --debug-port PORT      Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose          Verbose logging\n");
    printf("  -h, --help             This message\n");
}

static bool parse_args(int argc, char** argv) {
    enum {
        kHttpPort = 1, kGpio, kGpioEdge, kGpioActiveLow, kGpioDebounce, kSize, kActiveFps, kIdleFps, kRoi, kMeanMin, kMeanMax,
        kStdMin, kMqttHost, kMqttPort, kMqttTopic, kNoMqtt, kRtsp, kNoDebug, kDebugPort, kDebug
    };
    static struct option long_options[] = {
        {"http-port", required_argument, 0, kHttpPort},   {"gpio", required_argument, 0, kGpio},
        {"gpio-edge", required_argument, 0, kGpioEdge},   {"gpio-active-low", no_argument, 0, kGpioActiveLow},
        {"gpio-debounce", required_argument, 0, kGpioDebounce}, {"size", required_argument, 0, kSize},
        {"active-fps", required_argument, 0, kActiveFps}, {"idle-fps", required_argument, 0, kIdleFps},
        {"roi", required_argument, 0, kRoi},              {"mean-min", required_argument, 0, kMeanMin},
        {"mean-max", required_argument, 0, kMeanMax},     {"std-min", required_argument, 0, kStdMin},
        {"mqtt-host", required_argument, 0, kMqttHost},   {"mqtt-port", required_argument, 0, kMqttPort},
        {"mqtt-topic", required_argument, 0, kMqttTopic}, {"no-mqtt", no_argument, 0, kNoMqtt},
        {"rtsp", no_argument, 0, kRtsp},                  {"no-debug", no_argument, 0, kNoDebug},
        {"debug-port", required_argument, 0, kDebugPort}, {"verbose", no_argument, 0, 'v'},
        {"debug", no_argument, 0, kDebug},                {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case kHttpPort: g_config.http_port = std::atoi(optarg); break;
            case kGpio: g_config.gpio_line = std::atoi(optarg); break;
            case kGpioEdge:
                if (!std::strcmp(optarg, "rising")) g_config.gpio_edge = af::GpioTrigger::Edge::Rising;
                else if (!std::strcmp(optarg, "falling")) g_config.gpio_edge = af::GpioTrigger::Edge::Falling;
                else if (!std::strcmp(optarg, "both")) g_config.gpio_edge = af::GpioTrigger::Edge::Both;
                else { fprintf(stderr, "Invalid --gpio-edge: %s\n", optarg); return false; }
                break;
            case kGpioActiveLow: g_config.gpio_active_low = true; break;
            case kGpioDebounce: g_config.gpio_debounce_ms = std::atoi(optarg); break;
            case kSize:
                if (std::sscanf(optarg, "%dx%d", &g_config.width, &g_config.height) != 2) {
                    fprintf(stderr, "Invalid --size: %s\n", optarg);
                    return false;
                }
                break;
            case kActiveFps: g_config.active_fps = std::atoi(optarg); break;
            case kIdleFps: g_config.idle_fps = std::atoi(optarg); break;
            case kRoi:
                if (std::sscanf(optarg, "%f,%f,%f,%f", &g_config.roi[0], &g_config.roi[1], &g_config.roi[2], &g_config.roi[3]) != 4) {
                    fprintf(stderr, "Invalid --roi: %s\n", optarg);
                    return false;
                }
                break;
            case kMeanMin: g_config.mean_min = std::strtof(optarg, nullptr); break;
            case kMeanMax: g_config.mean_max = std::strtof(optarg, nullptr); break;
            case kStdMin: g_config.std_min = std::strtof(optarg, nullptr); break;
            case kMqttHost: g_config.mqtt_host = optarg; break;
            case kMqttPort: g_config.mqtt_port = std::atoi(optarg); break;
            case kMqttTopic: g_config.mqtt_topic = optarg; break;
            case kNoMqtt: g_config.enable_mqtt = false; break;
            case kRtsp: g_config.enable_rtsp = true; break;
            case kNoDebug: g_config.enable_debug = false; break;
            case kDebug: g_config.enable_debug = true; break;
            case kDebugPort: g_config.debug_port = std::atoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    // The driver treats fps 0 as "no rate limit", so clamp rather than pass it through.
    g_config.active_fps = std::min(std::max(g_config.active_fps, 1), 30);
    g_config.idle_fps = std::min(std::max(g_config.idle_fps, 1), g_config.active_fps);
    return true;
}

static ha_mqtt::ClientOptions mqtt_options() {
    ha_mqtt::ClientOptions opts;
    opts.app_id = "trigger-inspection-demo";
    opts.client_id = "recamera-trigger-inspection-demo";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host = g_config.mqtt_host;
    opts.legacy_port = static_cast<uint16_t>(g_config.mqtt_port);
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "verdict", "Inspection Verdict", "{{ value_json.verdict }}", "", "", ""},
        EntityConfig{EntityType::BinarySensor, "inspection_ok", "Inspection OK", "{{ 'ON' if value_json.verdict == 'OK' else 'OFF' }}", "", "", ""},
        EntityConfig{EntityType::Sensor, "roi_mean", "ROI Brightness", "{{ value_json.metrics.mean }}", "", "", "measurement"},
    };
    return opts;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    af::Pipeline<Inspection> pipeline;
    pipeline.add("validate", step_validate).add("roi", step_roi).add("measure", step_measure).add("judge", step_judge);

    af::CameraOptions cam;
    cam.width = static_cast<uint16_t>(g_config.width);
    cam.height = static_cast<uint16_t>(g_config.height);
    cam.fps = static_cast<uint8_t>(g_config.active_fps);
    cam.idle_fps = static_cast<uint8_t>(g_config.idle_fps);

    af::StreamOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);
    stream.bitrate_kbps = 0;  // preview only: ipcam encoder defaults
    stream.gop = 0;
    const af::StreamMode stream_mode = g_config.enable_rtsp ? (g_config.enable_debug ? af::StreamMode::Both : af::StreamMode::RtspOnly)
                                                            : (g_config.enable_debug ? af::StreamMode::WsOnly : af::StreamMode::None);

    af::HttpTrigger::Config http_cfg;
    http_cfg.port = static_cast<uint16_t>(g_config.http_port);
    http_cfg.sync_response = true;  // the caller gets the verdict in the HTTP response
    http_cfg.send_timeout_ms = 2000;
    af::HttpTrigger http(http_cfg);

    af::GpioTrigger::Config gpio_cfg;
    gpio_cfg.line = g_config.gpio_line;
    gpio_cfg.edge = g_config.gpio_edge;
    gpio_cfg.active_low = g_config.gpio_active_low;
    gpio_cfg.debounce_ms = static_cast<uint32_t>(g_config.gpio_debounce_ms);
    af::GpioTrigger gpio(gpio_cfg);

    af::App app(TAG);
    // Trigger station: None unless --debug/--rtsp, so standby costs only the CH0 idle copies.
    app.with_camera(cam).with_stream_mode(stream_mode, stream, g_config.debug_port);
    if (g_config.enable_mqtt) app.with_mqtt(mqtt_options());
    if (g_config.http_port > 0) app.with_trigger(http);
    if (g_config.gpio_line >= 0) app.with_trigger(gpio);
    if (g_config.http_port <= 0 && g_config.gpio_line < 0) {
        MA_LOGW(TAG, "No trigger source configured (--http-port / --gpio); the app will only idle");
    }

    MA_LOGI(TAG, "Inspection %dx%d, active %d fps, standby %d fps, ROI %.2f,%.2f,%.2f,%.2f, mean %.0f..%.0f, stddev >= %.0f",
            g_config.width, g_config.height, g_config.active_fps, g_config.idle_fps, g_config.roi[0], g_config.roi[1], g_config.roi[2],
            g_config.roi[3], g_config.mean_min, g_config.mean_max, g_config.std_min);
    if (g_config.http_port > 0) MA_LOGI(TAG, "HTTP trigger: http://<device_ip>:%d/trigger[?tag=ID]", g_config.http_port);
    if (g_config.gpio_line >= 0) MA_LOGI(TAG, "GPIO trigger: gpio%d", g_config.gpio_line);

    // Strings the sinks read after the handler returns.
    static std::string payload;
    static std::string debug_extra;
    static af::Box roi_box;

    // No on_frame(): trigger-only mode.
    app.on_trigger([&](const af::TriggerEvent& ev, const af::VideoFrame& frame, af::Result& r) {
        Inspection st;
        const af::RunReport rep = pipeline.run(frame, st);
        r.attach(rep);
        build_result(payload, ev, r, st, pipeline, rep);
        r.json = payload;  // MQTT + synchronous HTTP response

        // Debug overlay: the ROI, labelled with the verdict.
        roi_box = af::Box{};
        roi_box.x = st.roi.x + st.roi.w * 0.5f;
        roi_box.y = st.roi.y + st.roi.h * 0.5f;
        roi_box.w = static_cast<float>(st.roi.w);
        roi_box.h = static_cast<float>(st.roi.h);
        roi_box.score = 1.f;
        roi_box.label = verdict_of(rep.flow);
        r.set_boxes(&roi_box, 1);
        debug_extra = "\"inspection\":" + payload;
        r.debug_extra = debug_extra;

        MA_LOGI(TAG, "Trigger %s #%u%s%s: %s%s%s (mean=%.1f stddev=%.1f, %u us)", af::to_string(ev.source), ev.seq, ev.tag[0] ? " tag=" : "",
                ev.tag, verdict_of(rep.flow), rep.reason ? " " : "", rep.reason ? rep.reason : "", st.mean, st.stddev, rep.total_us);
    });

    return app.run();
}
