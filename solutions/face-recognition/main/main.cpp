#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <map>
#include <signal.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"
#include "onvif_service_bringup.h"
#include "rtsp_server.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "arcface_runner.h"
#include "command_handler.h"
#include "face_align.h"
#include "face_tracker.h"
#include "face_types.h"
#include "gallery.h"
#include "liveness_runner.h"
#include "privacy_blur.h"
#include "result_payload.h"
#include "scrfd_runner.h"

using namespace ma;
using namespace face_recognition;
using privacy_blur::PrivacyBlur;
using privacy_blur::PrivacyBlurConfig;

#define TAG "face-recognition"

// The gallery is keyed on this. Both models are baked into it because an
// embedding produced by a different detector/aligner pair is not comparable
// either -- the crop is part of the contract, not just the weights.
static const char* kModelTag = "cvi:scrfd500m+mbf512@bf16";

static struct {
    std::string detect_model = "/userdata/local/models/scrfd_500m_int8.cvimodel";
    std::string embed_model = "/userdata/local/models/arcface_mbf_bf16.cvimodel";
    std::string liveness_model = "/userdata/local/models/liveness_minifasnet_bf16.cvimodel";
    std::string gallery_path =
        "/userdata/local/face-gallery/cvi_scrfd500m_mbf512_bf16.json";

    float det_threshold = 0.5f;
    float nms_threshold = 0.4f;
    float match_threshold = 0.40f;
    float min_face_px = 64.f;
    int embed_interval = 5;
    bool liveness_enabled = false;
    float liveness_threshold = 0.5f;

    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/face-recognition/results";
    std::string mqtt_cmd_topic = "recamera/face-recognition/cmd";
    std::string mqtt_ack_topic = "recamera/face-recognition/cmd/ack";

    int inference_width = 640;
    int inference_height = 480;
    int inference_fps = 10;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    int max_regions = 12;
    int track_max_lost = 15;

    bool enable_debug = true;
    int debug_port = 8001;
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool enable_blur = true;
    bool verbose = false;
} g_config;

static std::atomic<bool> g_running(true);
static ScrfdRunner* g_detector = nullptr;
static ArcFaceRunner* g_embedder = nullptr;
static LivenessRunner* g_liveness = nullptr;
static Gallery* g_gallery = nullptr;
static FaceTracker* g_tracker = nullptr;
static CommandQueue* g_commands = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static PrivacyBlur* g_blur = nullptr;
static Camera* g_camera = nullptr;
static OnvifMetaGate g_onvif_meta;
static uint32_t g_frame_id = 0;

// Per-track recognition state. Keyed on the tracker identity so a face is
// embedded on appearance and then only every embed_interval frames -- ArcFace
// on this NPU costs more than detection does, and a face that has not moved
// does not produce a new answer.
struct TrackState {
    uint32_t last_embed_frame = 0;
    bool embedded = false;
    std::string candidate_name;
    int consecutive_matches = 0;
    RecognitionResult cached;
};
static std::map<int, TrackState> g_track_state;

// A camera-source enrollment consumes frames from the loop, so it lives across
// iterations rather than blocking inside the command drain.
struct PendingEnroll {
    bool active = false;
    std::string request_id;
    std::string name;
    int frames_wanted = 1;
    int frames_left = 0;
    std::vector<Embedding512> collected;
};
static PendingEnroll g_pending_enroll;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Face Recognition for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -d, --detect-model PATH   SCRFD face detector (default: %s)\n", g_config.detect_model.c_str());
    printf("  -e, --embed-model PATH    ArcFace embedder (default: %s)\n", g_config.embed_model.c_str());
    printf("  -L, --liveness-model PATH MiniFASNet anti-spoof (default: %s)\n", g_config.liveness_model.c_str());
    printf("  -G, --gallery PATH        Gallery JSON (default: %s)\n", g_config.gallery_path.c_str());
    printf("  -t, --threshold FLOAT     Detection threshold (default: %.2f)\n", g_config.det_threshold);
    printf("      --nms FLOAT           Detection NMS IoU (default: %.2f)\n", g_config.nms_threshold);
    printf("      --match FLOAT         Cosine match threshold (default: %.2f)\n", g_config.match_threshold);
    printf("      --min-face-px N       Gate faces below this short side (default: %.0f)\n", g_config.min_face_px);
    printf("      --embed-interval N    Re-embed a track every N frames (default: %d)\n", g_config.embed_interval);
    printf("      --liveness            Enable the anti-spoof stage\n");
    printf("      --liveness-threshold F  P(real) above which a face is live (default: %.2f)\n", g_config.liveness_threshold);
    printf("  -m, --mqtt-host HOST      MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  -p, --mqtt-port PORT      MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("      --no-rtsp             Disable RTSP streaming\n");
    printf("      --no-mqtt             Disable MQTT (also disables enrollment)\n");
    printf("      --no-blur             Disable privacy blur on the stream\n");
    printf("      --no-debug            Disable the debug WebSocket stream\n");
    printf("      --debug-port PORT     Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("      --max-regions N       Max blur regions (default: %d)\n", g_config.max_regions);
    printf("      --track-max-lost N    Frames a track survives unmatched (default: %d)\n", g_config.track_max_lost);
    printf("  -v, --verbose             Verbose logging\n");
    printf("  -h, --help                Show this help\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"detect-model", required_argument, 0, 'd'},
        {"embed-model", required_argument, 0, 'e'},
        {"liveness-model", required_argument, 0, 'L'},
        {"gallery", required_argument, 0, 'G'},
        {"threshold", required_argument, 0, 't'},
        {"mqtt-host", required_argument, 0, 'm'},
        {"mqtt-port", required_argument, 0, 'p'},
        {"no-rtsp", no_argument, 0, 1},
        {"no-mqtt", no_argument, 0, 2},
        {"no-blur", no_argument, 0, 3},
        {"no-debug", no_argument, 0, 4},
        {"debug-port", required_argument, 0, 5},
        {"max-regions", required_argument, 0, 6},
        {"min-face-px", required_argument, 0, 7},
        {"embed-interval", required_argument, 0, 8},
        {"match", required_argument, 0, 9},
        {"nms", required_argument, 0, 10},
        {"liveness", no_argument, 0, 11},
        {"liveness-threshold", required_argument, 0, 12},
        {"track-max-lost", required_argument, 0, 13},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "d:e:L:G:t:m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': g_config.detect_model = optarg; break;
            case 'e': g_config.embed_model = optarg; break;
            case 'L': g_config.liveness_model = optarg; break;
            case 'G': g_config.gallery_path = optarg; break;
            case 't': g_config.det_threshold = std::stof(optarg); break;
            case 'm': g_config.mqtt_host = optarg; break;
            case 'p': g_config.mqtt_port = std::stoi(optarg); break;
            case 1: g_config.enable_rtsp = false; break;
            case 2: g_config.enable_mqtt = false; break;
            case 3: g_config.enable_blur = false; break;
            case 4: g_config.enable_debug = false; break;
            case 5: g_config.debug_port = std::stoi(optarg); break;
            case 6: g_config.max_regions = std::stoi(optarg); break;
            case 7: g_config.min_face_px = std::stof(optarg); break;
            case 8: g_config.embed_interval = std::max(1, std::stoi(optarg)); break;
            case 9: g_config.match_threshold = std::stof(optarg); break;
            case 10: g_config.nms_threshold = std::stof(optarg); break;
            case 11: g_config.liveness_enabled = true; break;
            case 12: g_config.liveness_threshold = std::stof(optarg); break;
            case 13: g_config.track_max_lost = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static bool init_models() {
    std::string err;

    g_detector = new ScrfdRunner();
    if (!g_detector->init(g_config.detect_model, &err)) {
        MA_LOGE(TAG, "Face detector: %s", err.c_str());
        return false;
    }
    g_detector->setScoreThreshold(g_config.det_threshold);
    g_detector->setNmsThreshold(g_config.nms_threshold);
    MA_LOGI(TAG, "SCRFD input  %s", g_detector->inputDescription().c_str());
    MA_LOGI(TAG, "SCRFD heads  %s", g_detector->outputDescription().c_str());

    g_embedder = new ArcFaceRunner();
    if (!g_embedder->init(g_config.embed_model, &err)) {
        MA_LOGE(TAG, "Face embedder: %s", err.c_str());
        return false;
    }
    MA_LOGI(TAG, "ArcFace input %s", g_embedder->inputDescription().c_str());

    // Liveness last, and never fatal: the model ION pool is the binding
    // constraint on this device, and recognition is still useful without an
    // anti-spoof stage while the reverse is not true.
    g_liveness = new LivenessRunner();
    if (g_config.liveness_enabled) {
        if (!g_liveness->initOptional(g_config.liveness_model, &err)) {
            MA_LOGW(TAG, "Liveness disabled: %s", err.c_str());
            g_config.liveness_enabled = false;
        } else {
            MA_LOGI(TAG, "Liveness input %s", g_liveness->inputDescription().c_str());
        }
    } else {
        MA_LOGI(TAG, "Liveness stage disabled by configuration");
    }

    g_gallery = new Gallery();
    if (!g_gallery->load(g_config.gallery_path, kModelTag, g_config.match_threshold, &err)) {
        MA_LOGE(TAG, "Gallery: %s", err.c_str());
        return false;
    }
    MA_LOGI(TAG, "Gallery %s: %zu user(s), model_tag=%s, match>=%.2f",
            g_config.gallery_path.c_str(), g_gallery->size(), kModelTag,
            g_config.match_threshold);

    FaceTrackerConfig tcfg;
    tcfg.max_lost_frames = g_config.track_max_lost;
    g_tracker = new FaceTracker(tcfg);
    g_commands = new CommandQueue();
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
        // Recognition needs CPU access to the pixels (letterbox, warp, crop).
        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

        MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps for inference)",
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
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));
    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps)", g_config.stream_width,
            g_config.stream_height, g_config.stream_fps);
    return true;
}

static void on_mqtt_command(const std::string& topic, const std::string& payload);

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT disabled -- enrollment is unreachable in this mode");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id = "face-recognition";
    opts.client_id = "recamera-face-recognition";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host = g_config.mqtt_host;
    opts.legacy_port = static_cast<uint16_t>(g_config.mqtt_port);
    opts.subscribe_topics = {g_config.mqtt_cmd_topic};
    opts.message_handler = on_mqtt_command;

    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "face_count", "Face Count",
                     "{{ value_json.face_count }}", "", "", "measurement"},
        EntityConfig{EntityType::BinarySensor, "occupancy", "Face Detected",
                     "{{ 'ON' if value_json.face_count > 0 else 'OFF' }}", "occupancy", "", ""},
    };

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    MA_LOGI(TAG, "Command topic: %s -> ack %s", g_config.mqtt_cmd_topic.c_str(),
            g_config.mqtt_ack_topic.c_str());

    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
    if (g_onvif_meta.enabled()) {
        MA_LOGI(TAG, "ONVIF metadata: %s (every %ums)", g_onvif_meta.topic().c_str(),
                g_onvif_meta.config().interval_ms);
    }
    return true;
}

static bool init_blur() {
    if (!g_config.enable_blur) return true;
    if (!g_config.enable_rtsp) {
        MA_LOGW(TAG, "Privacy blur requires RTSP streaming, disabling it");
        g_config.enable_blur = false;
        return true;
    }
    PrivacyBlurConfig cfg;
    loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, nullptr);
    // No device-wide config means stay off: a recognition app that silently
    // masked the faces it recognises would be a surprising default.
    if (!cfg.present) cfg.enabled = false;
    cfg.max_regions = g_config.max_regions;

    g_blur = new PrivacyBlur();
    if (!g_blur->init(cfg, g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "Failed to initialize privacy blur");
        delete g_blur;
        g_blur = nullptr;
        return false;
    }
    MA_LOGI(TAG, "Privacy blur ready (backend=%s, max_regions=%d, enabled=%d)",
            cfg.backend.c_str(), cfg.max_regions, static_cast<int>(cfg.enabled));
    return true;
}

static void cleanup() {
    if (g_blur) {
        g_blur->deinit();
        delete g_blur;
        g_blur = nullptr;
    }
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    onvif_service_stop();
    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }
    if (g_mqtt) {
        g_mqtt->deinit();
        delete g_mqtt;
        g_mqtt = nullptr;
    }
    delete g_commands;   g_commands = nullptr;
    delete g_tracker;    g_tracker = nullptr;
    delete g_gallery;    g_gallery = nullptr;
    delete g_liveness;   g_liveness = nullptr;
    delete g_embedder;   g_embedder = nullptr;
    delete g_detector;   g_detector = nullptr;
    MA_LOGI(TAG, "Cleanup completed");
}

// ---------------------------------------------------------------------------
// MQTT command channel
// ---------------------------------------------------------------------------

static void publish_ack(const std::string& request_id, const std::string& op, bool ok,
                        const std::string& err, const AckData& data = AckData()) {
    const size_t users = g_gallery ? g_gallery->size() : 0;
    const std::string payload =
        buildAckJson(request_id, op, ok, err, kModelTag, users, data);
    if (g_mqtt) g_mqtt->publishText(g_config.mqtt_ack_topic, payload, 1, false);
    if (!ok) {
        MA_LOGW(TAG, "cmd %s failed: %s", op.c_str(), err.c_str());
    } else {
        MA_LOGI(TAG, "cmd %s ok", op.c_str());
    }
}

// Runs on the Mosquitto network thread: parse and enqueue only.
static void on_mqtt_command(const std::string& topic, const std::string& payload) {
    if (topic != g_config.mqtt_cmd_topic || !g_commands) return;
    std::string err, request_id, op;
    if (!g_commands->push(payload, &err, &request_id, &op)) {
        publish_ack(request_id, op, false, err);
    }
}

// Detects, aligns and embeds a single decoded still. Used by image enrollment.
//
// `::cv` everywhere below: `using namespace ma` above brings ma::cv into scope,
// which makes a bare `cv::` ambiguous against OpenCV's own namespace.
static bool embed_still(const ::cv::Mat& rgb, Embedding512& out, std::string* err) {
    std::vector<FaceDetection> dets;
    if (!g_detector->infer(rgb.data, rgb.cols, rgb.rows, dets, err)) return false;
    if (dets.empty()) {
        *err = "no face detected in the supplied image";
        return false;
    }
    // Score-descending already; take the strongest.
    const FaceDetection* best = &dets[0];
    for (const auto& d : dets) {
        if (d.score > best->score) best = &d;
    }
    std::array<uint8_t, 112 * 112 * 3> chip;
    if (!alignArcFace112(rgb.data, rgb.cols, rgb.rows, best->landmarks, chip)) {
        *err = "landmark alignment failed";
        return false;
    }
    return g_embedder->infer(chip.data(), out, err);
}

static void handle_image_enroll(const Command& cmd) {
    std::string err;
    const ::cv::Mat encoded(1, static_cast<int>(cmd.encoded_image.size()), CV_8UC1,
                            const_cast<uint8_t*>(cmd.encoded_image.data()));
    ::cv::Mat bgr = ::cv::imdecode(encoded, ::cv::IMREAD_COLOR);
    if (bgr.empty()) {
        publish_ack(cmd.request_id, "enroll", false, "cannot decode image_b64");
        return;
    }
    ::cv::Mat rgb;
    ::cv::cvtColor(bgr, rgb, ::cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    Embedding512 emb;
    if (!embed_still(rgb, emb, &err)) {
        publish_ack(cmd.request_id, "enroll", false, err);
        return;
    }
    if (!g_gallery->enroll(cmd.name, {emb}, &err)) {
        publish_ack(cmd.request_id, "enroll", false, err);
        return;
    }
    AckData data;
    data.addString("name", cmd.name);
    data.addUInt("n", 1);
    data.addString("source", "image");
    publish_ack(cmd.request_id, "enroll", true, "", data);
}

// Drains the command queue. Called from the frame loop with the current frame
// still valid, because camera enrollment needs exactly that.
static void drain_commands() {
    if (!g_commands) return;
    Command cmd;
    while (g_commands->pop(cmd)) {
        std::string err;
        switch (cmd.op) {
            case CommandOp::List: {
                AckData data;
                data.addRaw("names", buildNamesJson(g_gallery->list()));
                publish_ack(cmd.request_id, "list", true, "", data);
                break;
            }
            case CommandOp::Reload: {
                if (!g_gallery->reload(&err)) {
                    publish_ack(cmd.request_id, "reload", false, err);
                } else {
                    AckData data;
                    data.addUInt("reloaded", g_gallery->size());
                    publish_ack(cmd.request_id, "reload", true, "", data);
                }
                break;
            }
            case CommandOp::Remove: {
                if (!g_gallery->remove(cmd.name, &err)) {
                    publish_ack(cmd.request_id, "remove", false, err);
                } else {
                    AckData data;
                    data.addString("name", cmd.name);
                    publish_ack(cmd.request_id, "remove", true, "", data);
                }
                break;
            }
            case CommandOp::Enroll: {
                if (cmd.source == EnrollSource::Image) {
                    handle_image_enroll(cmd);
                    break;
                }
                if (g_pending_enroll.active) {
                    publish_ack(cmd.request_id, "enroll", false,
                                "another camera enrollment is already in progress");
                    break;
                }
                g_pending_enroll.active = true;
                g_pending_enroll.request_id = cmd.request_id;
                g_pending_enroll.name = cmd.name;
                g_pending_enroll.frames_wanted = cmd.frames;
                // Budget: 40 frames at ~10fps is about four seconds of looking
                // for a single clean face before giving up.
                g_pending_enroll.frames_left = 40;
                g_pending_enroll.collected.clear();
                MA_LOGI(TAG, "Camera enrollment armed for '%s' (%d frame(s))",
                        cmd.name.c_str(), cmd.frames);
                break;
            }
        }
    }
}

// Feeds one frame to an armed camera enrollment. `dets` must be this frame's
// detections and `rgb` must still be the live camera buffer.
static void step_camera_enroll(const uint8_t* rgb, int w, int h,
                               const std::vector<FaceDetection>& dets) {
    if (!g_pending_enroll.active) return;

    // Exactly one eligible face, or this frame does not count.
    std::vector<const FaceDetection*> eligible;
    for (const auto& d : dets) {
        if (d.bbox.shortSide() >= g_config.min_face_px) eligible.push_back(&d);
    }
    if (eligible.size() == 1) {
        std::array<uint8_t, 112 * 112 * 3> chip;
        Embedding512 emb;
        std::string err;
        if (alignArcFace112(rgb, w, h, eligible[0]->landmarks, chip) &&
            g_embedder->infer(chip.data(), emb, &err)) {
            bool live = true;
            if (g_config.liveness_enabled && g_liveness->available()) {
                float p = 0.f;
                if (g_liveness->inferRgbFrame(rgb, w, h, eligible[0]->bbox, p, &err)) {
                    live = p >= g_config.liveness_threshold;
                }
            }
            if (live) g_pending_enroll.collected.push_back(emb);
        }
    }

    if (static_cast<int>(g_pending_enroll.collected.size()) >= g_pending_enroll.frames_wanted) {
        std::string err;
        const bool ok = g_gallery->enroll(g_pending_enroll.name,
                                          g_pending_enroll.collected, &err);
        AckData data;
        data.addString("name", g_pending_enroll.name);
        data.addUInt("n", g_pending_enroll.collected.size());
        data.addString("source", "camera");
        publish_ack(g_pending_enroll.request_id, "enroll", ok, err, data);
        g_pending_enroll = PendingEnroll{};
        return;
    }

    if (--g_pending_enroll.frames_left <= 0) {
        publish_ack(g_pending_enroll.request_id, "enroll", false,
                    "timed out waiting for a single usable live face");
        g_pending_enroll = PendingEnroll{};
    }
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<RecognitionResult>& faces,
                                            float inference_time_ms) {
    std::vector<debug_stream_box_t> boxes;
    std::vector<std::string> labels;
    boxes.reserve(faces.size());
    labels.reserve(faces.size());
    for (const auto& f : faces) {
        boxes.push_back({(f.bbox.x1 + f.bbox.x2) * 0.5f, (f.bbox.y1 + f.bbox.y2) * 0.5f,
                         f.bbox.width(), f.bbox.height(), f.det_score, "face"});
        if (!f.name.empty()) {
            char buf[96];
            snprintf(buf, sizeof(buf), "%s %.2f", f.name.c_str(), f.score);
            labels.push_back(buf);
        } else if (f.gated) {
            labels.push_back("face");
        } else if (!f.reason.empty()) {
            labels.push_back(f.reason);
        } else {
            labels.push_back("unknown");
        }
    }
    debug_stream_letterbox_to_display(boxes, g_config.inference_width,
                                      g_config.inference_height, g_config.stream_width,
                                      g_config.stream_height);
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      boxes, &labels);
}

static void recognize(const uint8_t* rgb, int w, int h,
                      const std::vector<FaceDetection>& dets,
                      std::vector<RecognitionResult>& out) {
    out.clear();
    out.reserve(dets.size());

    for (const auto& d : dets) {
        TrackState& st = g_track_state[d.track_id];
        RecognitionResult r = st.cached;
        r.track_id = d.track_id;
        r.bbox = d.bbox;
        r.det_score = d.score;
        r.reason.clear();

        if (d.bbox.shortSide() < g_config.min_face_px) {
            // Too small to embed: below this size the 112x112 chip is mostly
            // interpolation and the cosine it produces is noise that would be
            // written into the track's identity.
            r.gated = true;
            r.reason = "small";
            r.name.clear();
            r.score = 0.f;
            r.stable = false;
            r.has_liveness = false;
            st.cached = r;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
            out.push_back(r);
            continue;
        }
        r.gated = false;

        const bool due = !st.embedded ||
                         (g_frame_id - st.last_embed_frame) >= static_cast<uint32_t>(g_config.embed_interval);
        if (!due) {
            out.push_back(r);
            continue;
        }

        st.last_embed_frame = g_frame_id;
        st.embedded = true;

        std::string err;
        std::array<uint8_t, 112 * 112 * 3> chip;
        if (!alignArcFace112(rgb, w, h, d.landmarks, chip)) {
            r.gated = true;
            r.reason = "align";
            r.name.clear();
            r.stable = false;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
            st.cached = r;
            out.push_back(r);
            continue;
        }

        Embedding512 emb;
        if (!g_embedder->infer(chip.data(), emb, &err)) {
            r.gated = true;
            r.reason = "embed";
            r.name.clear();
            r.stable = false;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
            st.cached = r;
            out.push_back(r);
            if (g_config.verbose) MA_LOGW(TAG, "embed failed: %s", err.c_str());
            continue;
        }

        if (g_config.liveness_enabled && g_liveness->available()) {
            float p = 0.f;
            if (g_liveness->inferRgbFrame(rgb, w, h, d.bbox, p, &err)) {
                r.has_liveness = true;
                r.liveness_score = p;
                r.live = p >= g_config.liveness_threshold;
            } else {
                r.has_liveness = false;
                if (g_config.verbose) MA_LOGW(TAG, "liveness failed: %s", err.c_str());
            }
        } else {
            r.has_liveness = false;
        }

        const GalleryMatch m = g_gallery->match(emb);
        r.score = m.score;

        if (r.has_liveness && !r.live) {
            // A spoof is not an identity. The score is still reported so the
            // near-match is visible in the log.
            r.name.clear();
            r.reason = "spoof";
            r.stable = false;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
        } else if (m.matched) {
            r.name = m.name;
            if (st.candidate_name == m.name) {
                st.consecutive_matches++;
            } else {
                st.candidate_name = m.name;
                st.consecutive_matches = 1;
            }
            // Two consecutive embedding runs agreeing is what promotes a name
            // from "this frame said so" to something a consumer should act on.
            r.stable = st.consecutive_matches >= 2;
        } else {
            r.name.clear();
            r.stable = false;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
        }

        st.cached = r;
        out.push_back(r);
    }
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();

    const int fw = frame.width;
    const int fh = frame.height;
    const uint8_t* pixels = frame.data;

    // Every pixel-dependent stage must finish before returnFrame(): the camera
    // owns this buffer and reuses it the moment it is handed back.
    const auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<FaceDetection> dets;
    std::string err;
    if (!g_detector->infer(pixels, fw, fh, dets, &err)) {
        MA_LOGW(TAG, "detect failed: %s", err.c_str());
        dets.clear();
    }
    const auto detect_end = std::chrono::high_resolution_clock::now();
    const auto detect_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

    const std::vector<int> track_ids = g_tracker->update(dets);
    for (size_t i = 0; i < dets.size() && i < track_ids.size(); ++i) {
        dets[i].track_id = track_ids[i];
    }
    for (int id : g_tracker->removedIds()) g_track_state.erase(id);

    if (g_config.enable_blur && g_blur) {
        // Same letterbox conversion the overlay uses: the detections are
        // normalized against the 640x480 inference frame while the mask is
        // applied on the 16:9 stream frame, and the two crop differently.
        std::vector<geometry::InferBox> boxes;
        boxes.reserve(dets.size());
        for (const auto& d : dets) {
            boxes.push_back(geometry::InferBox::fromCorner(
                d.bbox.x1 / fw, d.bbox.y1 / fh, d.bbox.width() / fw, d.bbox.height() / fh,
                d.score));
        }
        const auto stream_boxes =
            geometry::toStream(boxes, g_config.inference_width, g_config.inference_height,
                               g_config.stream_width, g_config.stream_height);
        g_blur->onDetection(stream_boxes, &frame);
    }

    const auto recog_start = std::chrono::high_resolution_clock::now();
    std::vector<RecognitionResult> faces;
    recognize(pixels, fw, fh, dets, faces);
    const auto recog_end = std::chrono::high_resolution_clock::now();
    const auto recog_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(recog_end - recog_start).count();

    // A camera enrollment consumes THIS frame, so it has to run inside the
    // frame's lifetime. The rest of the command handling does not, and is
    // drained from the main loop instead -- see the comment there.
    step_camera_enroll(pixels, fw, fh, dets);

    if (debug_stream_snapshot_armed()) {
        if (g_config.enable_blur && g_blur && g_blur->enabled() && !dets.empty()) {
            std::vector<geometry::InferBox> snap_boxes;
            snap_boxes.reserve(dets.size());
            for (const auto& d : dets) {
                snap_boxes.push_back(geometry::InferBox::fromCorner(
                    d.bbox.x1 / fw, d.bbox.y1 / fh, d.bbox.width() / fw,
                    d.bbox.height() / fh, d.score));
            }
            privacy_blur::pixelateRgb888(frame.data, frame.width, frame.height, snap_boxes,
                                         g_blur->blockPx());
        }
        debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    }

    g_camera->returnFrame(frame);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (g_config.enable_mqtt && g_mqtt) {
        const std::string payload =
            buildResultsJson(static_cast<uint64_t>(timestamp_ms), g_frame_id,
                             static_cast<float>(total_time), fw, fh, kModelTag, faces);
        g_mqtt->publishResultsJson(payload);

        if (g_onvif_meta.take(timestamp_ms)) {
            std::vector<onvif_box_t> ob;
            ob.reserve(faces.size());
            for (const auto& f : faces) {
                ob.push_back({(f.bbox.x1 + f.bbox.x2) * 0.5f, (f.bbox.y1 + f.bbox.y2) * 0.5f,
                              f.bbox.width(), f.bbox.height(), f.det_score, "HumanFace"});
            }
            g_mqtt->publishText(g_onvif_meta.topic(),
                                onvif_meta_to_json(onvif_meta_from_boxes(
                                    timestamp_ms, "FaceRecognition", fw, fh, ob)));
        }
    }

    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        const std::string debug_json = build_debug_results_json(
            static_cast<uint64_t>(timestamp_ms), g_frame_id, faces,
            static_cast<float>(total_time));
        debug_stream_publish_result(debug_json.c_str(), debug_json.size());
    }

    if (g_config.verbose || !faces.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu faces, detect=%lldms, recognize=%lldms, total=%lldms",
                g_frame_id, faces.size(), static_cast<long long>(detect_time),
                static_cast<long long>(recog_time), static_cast<long long>(total_time));
        for (const auto& f : faces) {
            MA_LOGI(TAG, "  Face[%d]: name=%s score=%.3f det=%.2f live=%s stable=%d gated=%d%s%s",
                    f.track_id, f.name.empty() ? "-" : f.name.c_str(), f.score, f.det_score,
                    f.has_liveness ? (f.live ? "yes" : "no") : "n/a",
                    static_cast<int>(f.stable), static_cast<int>(f.gated),
                    f.reason.empty() ? "" : " reason=", f.reason.c_str());
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    MA_LOGI(TAG, "Starting Face Recognition Application");
    MA_LOGI(TAG, "Detector: %s", g_config.detect_model.c_str());
    MA_LOGI(TAG, "Embedder: %s", g_config.embed_model.c_str());
    MA_LOGI(TAG, "Gallery : %s", g_config.gallery_path.c_str());

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!init_models()) {
        MA_LOGE(TAG, "Model initialization failed");
        cleanup();
        return 1;
    }
    if (!init_camera()) {
        MA_LOGE(TAG, "Camera initialization failed");
        cleanup();
        return 1;
    }
    if (!init_video_streaming()) {
        MA_LOGE(TAG, "Video streaming initialization failed");
        cleanup();
        return 1;
    }
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt()) {
        MA_LOGE(TAG, "MQTT initialization failed");
        cleanup();
        return 1;
    }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    if (g_config.enable_rtsp) startVideo();

    if (!init_blur()) {
        MA_LOGW(TAG, "Privacy blur initialization failed, continuing without it");
        g_config.enable_blur = false;
    }

    if (onvif_service_bringup(g_onvif_meta.config(), ha_mqtt::readDeviceIdentifier(),
                              "reCamera",
                              g_config.enable_debug ? g_config.debug_port : 0) == 0 &&
        onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Face recognition running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:554/live");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    // Commands are drained OUTSIDE process_frame() on purpose. process_frame()
    // returns early when the camera hands back no frame, and a camera that has
    // stopped delivering is exactly when an operator most needs `list` and
    // `remove` to still answer -- draining inside it made the command channel
    // go silent along with the video.
    //
    // Still on the main thread, so gallery mutation and CVI inference never race
    // the Mosquitto network thread.
    while (g_running.load()) {
        drain_commands();
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Face Recognition Application terminated");
    return 0;
}
