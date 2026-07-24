#include <getopt.h>
#include <signal.h>
#include <syslog.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <quirc.h>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>

#include "mqtt_payload.h"
#include "rtsp_demo.h"

using namespace ma;

using Clock = std::chrono::steady_clock;

#define TAG "qrcode-reader"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static struct {
    // Inference (CH0) frame — QR detection runs on this RGB888 buffer.
    int camera_w   = 640;
    int camera_h   = 640;
    int camera_fps = 10;

    std::string mqtt_host  = "localhost";
    int mqtt_port          = 1883;
    std::string mqtt_topic = "recamera/qrcode-reader/results";
    bool enable_mqtt       = true;

    bool enable_rtsp  = true;
    int stream_width  = 1280;
    int stream_height = 720;
    int stream_fps    = 30;

    bool enable_debug = true;  // H.264-over-WS + results JSON for supervisor console
    bool verbose      = false;

    int print_interval = 30;
} g_config;

static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static Camera*                 g_camera         = nullptr;
static std::atomic<bool>       g_running{true};

// QR decoding state reused across frames (decode runs only on the main loop
// thread now; the mutex only guards against an unexpected reentry).
static struct quirc* g_qr = nullptr;
static std::mutex g_qr_mutex;
static std::atomic<uint64_t> g_frame_id{0};

// ---------------------------------------------------------------------------
// Letterbox mapping
// ---------------------------------------------------------------------------

// Map a point in inference-frame pixels to normalized [0,1] display-frame
// coords, replicating debug_stream_letterbox_to_display()'s content-fit math
// (the box helper only transforms boxes, so QR corners get an equivalent
// per-point mapping). The sensor content (display aspect) is fit into the
// inference frame preserving aspect and centered; we recover that sub-rect and
// scale into the display frame, then normalize.
static void map_point_norm(float px, float py,
                           int inf_w, int inf_h, int disp_w, int disp_h,
                           float& nx, float& ny) {
    float content_w = static_cast<float>(inf_w);
    float content_h = static_cast<float>(inf_h);
    float x_off = 0.f, y_off = 0.f;
    const long long disp_vs_inf =
        static_cast<long long>(disp_w) * inf_h - static_cast<long long>(disp_h) * inf_w;
    if (disp_vs_inf > 0) {
        // Display wider than inference frame -> bars top/bottom in inference.
        content_h = static_cast<float>(inf_w) * disp_h / disp_w;
        y_off     = (inf_h - content_h) * 0.5f;
    } else if (disp_vs_inf < 0) {
        // Display taller -> bars left/right.
        content_w = static_cast<float>(inf_h) * disp_w / disp_h;
        x_off     = (inf_w - content_w) * 0.5f;
    }
    nx = (px - x_off) / content_w;  // ((px-x_off)*sx)/disp_w with sx=disp_w/content_w
    ny = (py - y_off) / content_h;
    if (nx < 0.f) nx = 0.f;
    if (nx > 1.f) nx = 1.f;
    if (ny < 0.f) ny = 0.f;
    if (ny > 1.f) ny = 1.f;
}

// ---------------------------------------------------------------------------
// Cleanup / signal handling
// ---------------------------------------------------------------------------

static void cleanup() {
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    deinitVideo();
    deinitRtsp();
    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }
    if (g_qr) {
        quirc_destroy(g_qr);
        g_qr = nullptr;
    }
}

static void app_ipcam_ExitSig_handle(int signo) {
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);

    if ((SIGINT == signo) || (SIGTERM == signo)) {
        std::printf("[INFO] received signal %d, shutting down\n", signo);
        g_running.store(false);
    }
}

// ---------------------------------------------------------------------------
// Debug /results envelope: QR detection has no boxes — the overlay box array is
// empty and the decoded codes ride as an extra top-level "qrcodes" member.
// ---------------------------------------------------------------------------

static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            double detect_cost_ms,
                                            const std::vector<qrcode_reader::QrCode>& codes) {
    const std::string extra = qrcode_reader::buildQrcodesExtra(codes);
    const std::vector<debug_stream_box_t> no_boxes;
    return debug_stream_build_results(timestamp_ms, frame_id,
                                      static_cast<float>(detect_cost_ms),
                                      g_config.stream_width, g_config.stream_height,
                                      no_boxes, nullptr, extra);
}

// ---------------------------------------------------------------------------
// QR detection — runs on the normal-priority main loop (NOT the CH0 realtime
// VENC consumer thread). Heavy RGB->gray + quirc work here no longer preempts
// the RTSP/libevent server thread, so :8554 keeps accepting clients.
// ---------------------------------------------------------------------------

static void processQRCode(const ma_img_t& frame) {
    std::lock_guard<std::mutex> lk(g_qr_mutex);

    const int W = static_cast<int>(frame.width);
    const int H = static_cast<int>(frame.height);
    const uint8_t* pu8VirAddr = frame.data;
    if (!pu8VirAddr) return;

    // Lazily (re)create the quirc context sized to the frame.
    if (!g_qr) {
        g_qr = quirc_new();
        if (g_qr && quirc_resize(g_qr, W, H) < 0) {
            quirc_destroy(g_qr);
            g_qr = nullptr;
        }
    }
    if (!g_qr) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "Failed to initialize Quirc\n");
        return;
    }

    const auto t0 = Clock::now();

    // RGB888 -> grayscale straight into the quirc buffer.
    uint8_t* buffer = quirc_begin(g_qr, nullptr, nullptr);
    const uint32_t npix = frame.width * frame.height;
    for (uint32_t i = 0; i < npix; ++i) {
        const uint8_t r = pu8VirAddr[i * 3];
        const uint8_t g = pu8VirAddr[i * 3 + 1];
        const uint8_t b = pu8VirAddr[i * 3 + 2];
        buffer[i]       = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }
    quirc_end(g_qr);

    std::vector<qrcode_reader::QrCode> codes;
    const int count = quirc_count(g_qr);
    for (int j = 0; j < count; ++j) {
        struct quirc_code code;
        struct quirc_data data;
        quirc_extract(g_qr, j, &code);
        if (quirc_decode(&code, &data) != QUIRC_SUCCESS) continue;

        qrcode_reader::QrCode qc;
        qc.text.assign(reinterpret_cast<char*>(data.payload),
                       static_cast<size_t>(data.payload_len));
        for (int p = 0; p < 4; ++p) {
            float nx = 0.f, ny = 0.f;
            map_point_norm(static_cast<float>(code.corners[p].x),
                           static_cast<float>(code.corners[p].y),
                           W, H, g_config.stream_width, g_config.stream_height, nx, ny);
            qc.points[p][0] = nx;
            qc.points[p][1] = ny;
        }
        codes.push_back(std::move(qc));
    }

    const auto t1              = Clock::now();
    const double detect_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const bool qr_found        = !codes.empty();
    const uint64_t frame_id    = g_frame_id.fetch_add(1) + 1;

    // Publish every frame (empty codes too: clears the Console overlay and the
    // HA qr_count on the frame after a code leaves the view).
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        const std::string payload =
            qrcode_reader::buildResultJson(frame_id, qr_found, detect_ms, codes);
        g_mqtt_publisher->publishResultsJson(payload);
    }

    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        const uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
        const std::string dj =
            build_debug_results_json(ts_ms, static_cast<uint32_t>(frame_id), detect_ms, codes);
        debug_stream_publish_result(dj.c_str(), dj.size());
    }

    if (g_config.verbose || (frame_id % static_cast<uint64_t>(g_config.print_interval)) == 0) {
        if (qr_found) {
            for (const auto& c : codes) {
                std::printf("[RESULT] frame=%llu qr=\"%s\" detect=%.2fms\n",
                            static_cast<unsigned long long>(frame_id), c.text.c_str(), detect_ms);
            }
        } else {
            std::printf("[RESULT] frame=%llu no QR code (detect=%.2fms)\n",
                        static_cast<unsigned long long>(frame_id), detect_ms);
        }
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::printf("QR Code Reader for ReCamera\n");
    std::printf("Usage: %s [options]\n\n", prog);
    std::printf("Options:\n");
    std::printf("  --camera-width N      Inference frame width (default: %d)\n", g_config.camera_w);
    std::printf("  --camera-height N     Inference frame height (default: %d)\n", g_config.camera_h);
    std::printf("  --camera-fps N        Inference frame rate (default: %d)\n", g_config.camera_fps);
    std::printf("  -m, --mqtt-host HOST  MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    std::printf("  -p, --mqtt-port PORT  MQTT broker port (default: %d)\n", g_config.mqtt_port);
    std::printf("  --mqtt-topic TOPIC    MQTT publish topic (default: %s)\n", g_config.mqtt_topic.c_str());
    std::printf("  --no-mqtt             Disable MQTT publishing\n");
    std::printf("  --no-rtsp             Disable RTSP streaming\n");
    std::printf("  --stream-width N      RTSP encode width (default: %d)\n", g_config.stream_width);
    std::printf("  --stream-height N     RTSP encode height (default: %d)\n", g_config.stream_height);
    std::printf("  --stream-fps N        RTSP encode fps (default: %d)\n", g_config.stream_fps);
    std::printf("  --print-interval N    Print every N frames (default: %d)\n", g_config.print_interval);
    std::printf("  -v, --verbose         Enable verbose logging\n");
    std::printf("  -h, --help            Show this help message\n\n");
    std::printf("RTSP stream: rtsp://<device-ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"camera-width",   required_argument, 0,  1 },
        {"camera-height",  required_argument, 0,  2 },
        {"camera-fps",     required_argument, 0,  3 },
        {"mqtt-host",      required_argument, 0, 'm'},
        {"mqtt-port",      required_argument, 0, 'p'},
        {"mqtt-topic",     required_argument, 0,  4 },
        {"no-mqtt",        no_argument,       0,  5 },
        {"no-rtsp",        no_argument,       0,  6 },
        {"stream-width",   required_argument, 0,  7 },
        {"stream-height",  required_argument, 0,  8 },
        {"stream-fps",     required_argument, 0,  9 },
        {"print-interval", required_argument, 0, 10 },
        {"verbose",        no_argument,       0, 'v'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case  1 : g_config.camera_w       = std::atoi(optarg); break;
            case  2 : g_config.camera_h       = std::atoi(optarg); break;
            case  3 : g_config.camera_fps     = std::atoi(optarg); break;
            case 'm': g_config.mqtt_host      = optarg; break;
            case 'p': g_config.mqtt_port      = std::atoi(optarg); break;
            case  4 : g_config.mqtt_topic     = optarg; break;
            case  5 : g_config.enable_mqtt    = false; break;
            case  6 : g_config.enable_rtsp    = false; break;
            case  7 : g_config.stream_width   = std::atoi(optarg); break;
            case  8 : g_config.stream_height  = std::atoi(optarg); break;
            case  9 : g_config.stream_fps     = std::atoi(optarg); break;
            case 10 : g_config.print_interval = std::max(1, std::atoi(optarg)); break;
            case 'v': g_config.verbose        = true; break;
            case 'h': print_usage(argv[0]); std::exit(0);
            default : print_usage(argv[0]); return false;
        }
    }
    return true;
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        std::printf("[INFO] MQTT publishing disabled\n");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "qrcode-reader";
    opts.client_id     = "recamera-qrcode-reader";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "qr_last_text", "QR Last Text",
                     "{{ value_json.codes[0].text if value_json.codes | length > 0 else '' }}",
                     "", "", ""},
        EntityConfig{EntityType::Sensor, "qr_count", "QR Count",
                     "{{ value_json.codes | length }}",
                     "", "", "measurement"},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        std::fprintf(stderr, "[ERROR] MQTT publisher init failed\n");
        return false;
    }
    std::printf("[OK] MQTT publishing topic=%s\n", g_config.mqtt_topic.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Camera (CH0 inference frames) — same pattern as facemesh-reader: the frame
// is pulled on the normal-priority main loop via retrieveFrame/returnFrame,
// keeping QR decoding off the realtime VENC consumer thread.
// ---------------------------------------------------------------------------

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Apply the inference frame rate (else the channel defaults high and
            // the capture->inference FIFO fills faster than inference drains).
            value.i32 = g_config.camera_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

            value.u16s[0] = g_config.camera_w;
            value.u16s[1] = g_config.camera_h;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // CPU-accessible frames (quirc reads pixels).
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            std::printf("[OK] camera initialized (%dx%d @ %dfps for inference)\n",
                        g_config.camera_w, g_config.camera_h, g_config.camera_fps);
            return true;
        }
    }

    std::fprintf(stderr, "[ERROR] no camera found\n");
    return false;
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
    processQRCode(frame);
    g_camera->returnFrame(frame);
}

int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) return 1;

    std::printf("[INFO] starting qrcode reader\n");

    signal(SIGINT, app_ipcam_ExitSig_handle);
    signal(SIGTERM, app_ipcam_ExitSig_handle);

    if (!init_mqtt()) return 2;

    // CH0 inference frames come through the Camera abstraction (main-loop
    // retrieveFrame), so no CH0 setupVideo / realtime frame handler here.
    if (!init_camera()) return 3;

    if (initVideo()) {
        std::fprintf(stderr, "[ERROR] initVideo failed\n");
        return 3;
    }

    video_ch_param_t param{};

    // CH2: H.264 with RTSP (+ debug_stream as consumer index 1).
    if (g_config.enable_rtsp) {
        param.format = VIDEO_FORMAT_H264;
        param.width  = static_cast<uint32_t>(g_config.stream_width);
        param.height = static_cast<uint32_t>(g_config.stream_height);
        param.fps    = static_cast<uint8_t>(g_config.stream_fps);
        setupVideo(VIDEO_CH2, &param);
        registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
        initRtsp((0x01 << VIDEO_CH2));
        std::printf("[OK] RTSP streaming rtsp://<device-ip>:8554/live0 (%dx%d@%dfps)\n",
                    g_config.stream_width, g_config.stream_height, g_config.stream_fps);
    }

    // Debug stream (non-fatal): registers debug_stream_video_handler as VENC
    // consumer index 1 on CH2 (RTSP owns index 0).
    if (g_config.enable_debug && debug_stream_start_or_disable(8001, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    startVideo();

    std::printf("[OK] qrcode reader running (inference %dx%d RGB888)\n",
                g_config.camera_w, g_config.camera_h);

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    std::printf("[INFO] qrcode reader terminated\n");
    return 0;
}
