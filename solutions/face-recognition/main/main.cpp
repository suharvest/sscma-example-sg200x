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
#include "facemesh_liveness.h"
#include "gallery.h"
#include "liveness_capture_writer.h"
#include "liveness_runner.h"
#include "liveness_temporal.h"
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
    std::string liveness_v1se_model =
        "/userdata/local/models/liveness_minifasnet_v1se_bf16.cvimodel";
    std::string facemesh_model = "/userdata/local/models/face_landmark_cv181x_bf16.cvimodel";
    std::string gallery_path =
        "/userdata/local/face-gallery/cvi_scrfd500m_mbf512_bf16.json";
    // /usr/local/bin is read-only on this device, so the capture file lives with
    // the gallery -- the one directory the package guarantees is writable.
    std::string liveness_capture_path =
        "/userdata/local/face-gallery/liveness_capture.jsonl";

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

// Liveness v2 tuning. `liveness_threshold` above stays as the single-shot legacy
// knob; everything temporal lives here.
static LivenessConfig g_lcfg;

static std::atomic<bool> g_running(true);
static ScrfdRunner* g_detector = nullptr;
static ArcFaceRunner* g_embedder = nullptr;
static LivenessRunner* g_liveness = nullptr;
static FacemeshLiveness* g_facemesh = nullptr;
static Gallery* g_gallery = nullptr;
static FaceTracker* g_tracker = nullptr;
static CommandQueue* g_commands = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static PrivacyBlur* g_blur = nullptr;
static Camera* g_camera = nullptr;
static OnvifMetaGate g_onvif_meta;
static uint32_t g_frame_id = 0;

// Monotonic milliseconds. Every temporal liveness decision and the capture
// deadline use this rather than the wall clock, so an NTP step mid-session
// cannot fabricate a two-second timeout or truncate a capture.
static uint64_t steady_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Wall-clock seconds, for the `ts` field of a capture row -- that one is meant
// to be correlated with external recordings, so it must not be monotonic.
static double wall_seconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
               .count() / 1000.0;
}

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
    // v2: temporal liveness evidence, retired with the track.
    LivenessState liveness;
};
static std::map<int, TrackState> g_track_state;

// Calibration capture. Armed by an MQTT command, drained and stopped on the main
// thread only, so the Mosquitto thread never touches the FILE*.
static LivenessCaptureWriter g_capture;
static std::string g_capture_request_id;

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
    printf("      --liveness-v1se-model PATH  MiniFASNet-V1SE 4.0x head (default: %s)\n", g_config.liveness_v1se_model.c_str());
    printf("      --facemesh-model PATH   FaceMesh landmarks for blink (default: %s)\n", g_config.facemesh_model.c_str());
    printf("      --liveness            Enable the anti-spoof stage\n");
    printf("      --liveness-threshold F  P(real) above which a face is live (default: %.2f)\n", g_config.liveness_threshold);
    printf("  Liveness v2 (temporal fusion):\n");
    printf("      --liveness-texture-ema-alpha F  (default: %.2f)\n", g_lcfg.texture_ema_alpha);
    printf("      --liveness-min-samples N        Texture samples before a verdict (default: %d)\n", g_lcfg.min_samples);
    printf("      --liveness-motion-window-sec F  (default: %.2f)\n", g_lcfg.motion_window_sec);
    printf("      --liveness-timeout-sec F        (default: %.2f)\n", g_lcfg.timeout_sec);
    printf("      --liveness-motion-noise-floor F (default: %.4f)\n", g_lcfg.motion_noise_floor);
    printf("      --liveness-motion-high F        (default: %.4f)\n", g_lcfg.motion_high);
    printf("      --liveness-correlation-low F    (default: %.2f)\n", g_lcfg.correlation_low);
    printf("      --liveness-correlation-high F   (default: %.2f)\n", g_lcfg.correlation_high);
    printf("      --liveness-facemesh-interval N  0 disables blink (default: %d)\n", g_lcfg.facemesh_interval);
    printf("      --liveness-ear-threshold F      (default: %.2f)\n", g_lcfg.ear_threshold);
    printf("      --liveness-blink-min-samples N  (default: %d)\n", g_lcfg.blink_min_samples);
    printf("      --liveness-blink-max-samples N  (default: %d)\n", g_lcfg.blink_max_samples);
    printf("      --liveness-w-texture F          (default: %.2f)\n", g_lcfg.w_texture);
    printf("      --liveness-w-motion F           (default: %.2f)\n", g_lcfg.w_motion);
    printf("      --liveness-w-depth F            (default: %.2f)\n", g_lcfg.w_depth);
    printf("      --liveness-blink-bonus F        (default: %.2f)\n", g_lcfg.blink_bonus);
    printf("      --liveness-t-live F             (default: %.2f)\n", g_lcfg.t_live);
    printf("      --liveness-t-spoof F            (default: %.2f)\n", g_lcfg.t_spoof);
    printf("      --liveness-depth-enabled N      Pro-only stub (default: 0)\n");
    printf("      --liveness-capture-max-sec N    (default: %d)\n", g_lcfg.capture_max_sec);
    printf("      --liveness-capture-path PATH    (default: %s)\n", g_config.liveness_capture_path.c_str());
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
        {"liveness-v1se-model", required_argument, 0, 20},
        {"facemesh-model", required_argument, 0, 21},
        {"liveness-texture-ema-alpha", required_argument, 0, 22},
        {"liveness-min-samples", required_argument, 0, 23},
        {"liveness-motion-window-sec", required_argument, 0, 24},
        {"liveness-timeout-sec", required_argument, 0, 25},
        {"liveness-motion-noise-floor", required_argument, 0, 26},
        {"liveness-motion-high", required_argument, 0, 27},
        {"liveness-correlation-low", required_argument, 0, 28},
        {"liveness-correlation-high", required_argument, 0, 29},
        {"liveness-facemesh-interval", required_argument, 0, 30},
        {"liveness-ear-threshold", required_argument, 0, 31},
        {"liveness-blink-min-samples", required_argument, 0, 32},
        {"liveness-blink-max-samples", required_argument, 0, 33},
        {"liveness-w-texture", required_argument, 0, 34},
        {"liveness-w-motion", required_argument, 0, 35},
        {"liveness-w-depth", required_argument, 0, 36},
        {"liveness-blink-bonus", required_argument, 0, 37},
        {"liveness-t-live", required_argument, 0, 38},
        {"liveness-t-spoof", required_argument, 0, 39},
        {"liveness-depth-enabled", required_argument, 0, 40},
        {"liveness-capture-max-sec", required_argument, 0, 41},
        {"liveness-capture-path", required_argument, 0, 42},
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
            case 20: g_config.liveness_v1se_model = optarg; break;
            case 21: g_config.facemesh_model = optarg; break;
            case 22: g_lcfg.texture_ema_alpha = std::stof(optarg); break;
            case 23: g_lcfg.min_samples = std::max(1, std::stoi(optarg)); break;
            case 24: g_lcfg.motion_window_sec = std::stof(optarg); break;
            case 25: g_lcfg.timeout_sec = std::stof(optarg); break;
            case 26: g_lcfg.motion_noise_floor = std::stof(optarg); break;
            case 27: g_lcfg.motion_high = std::stof(optarg); break;
            case 28: g_lcfg.correlation_low = std::stof(optarg); break;
            case 29: g_lcfg.correlation_high = std::stof(optarg); break;
            case 30: g_lcfg.facemesh_interval = std::max(0, std::stoi(optarg)); break;
            case 31: g_lcfg.ear_threshold = std::stof(optarg); break;
            case 32: g_lcfg.blink_min_samples = std::max(1, std::stoi(optarg)); break;
            case 33: g_lcfg.blink_max_samples = std::max(1, std::stoi(optarg)); break;
            case 34: g_lcfg.w_texture = std::stof(optarg); break;
            case 35: g_lcfg.w_motion = std::stof(optarg); break;
            case 36: g_lcfg.w_depth = std::stof(optarg); break;
            case 37: g_lcfg.blink_bonus = std::stof(optarg); break;
            case 38: g_lcfg.t_live = std::stof(optarg); break;
            case 39: g_lcfg.t_spoof = std::stof(optarg); break;
            case 40: g_lcfg.depth_enabled = std::stoi(optarg) != 0; break;
            case 41: g_lcfg.capture_max_sec = std::max(1, std::stoi(optarg)); break;
            case 42: g_config.liveness_capture_path = optarg; break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

// Model ION is the binding constraint on this device: 180 MB of system RAM and
// a carveout pool that face-analysis measured OOMing around 47 MB of aggregate
// model allocation. Disk size does not predict it, so the only honest way to
// know whether V2 + V1SE + FaceMesh fit under the 40 MB operational ceiling is
// to print the numbers on either side of every load and subtract.
//
// MemFree/MemAvailable come from /proc/meminfo (the same source `free -m`
// reads). The ION line is best-effort: the CVI carveout exporter is not at a
// stable path across kernels, so a miss prints "n/a" rather than failing.
static void log_mem(const char* stage) {
    long free_kb = -1, avail_kb = -1;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            long v = 0;
            if (sscanf(line, "MemFree: %ld kB", &v) == 1) free_kb = v;
            else if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) avail_kb = v;
        }
        fclose(f);
    }
    static const char* kIonPaths[] = {
        "/proc/cvitek/ion",
        "/sys/kernel/debug/ion/cvi_carveout_heap",
        "/proc/ion/cvi_carveout_heap",
    };
    std::string ion = "n/a";
    for (const char* p : kIonPaths) {
        FILE* g = fopen(p, "r");
        if (!g) continue;
        char line[512];
        while (fgets(line, sizeof(line), g)) {
            if (strstr(line, "Mem") || strstr(line, "mem") || strstr(line, "total")) {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                if (!s.empty()) ion = s;   // keep the last summary line
            }
        }
        fclose(g);
        break;
    }
    MA_LOGI(TAG, "MEM[%s] free=%ldMB available=%ldMB ion=%s", stage,
            free_kb >= 0 ? free_kb / 1024 : -1, avail_kb >= 0 ? avail_kb / 1024 : -1,
            ion.c_str());
}

static bool init_models() {
    std::string err;

    log_mem("before-scrfd");
    g_detector = new ScrfdRunner();
    if (!g_detector->init(g_config.detect_model, &err)) {
        MA_LOGE(TAG, "Face detector: %s", err.c_str());
        return false;
    }
    g_detector->setScoreThreshold(g_config.det_threshold);
    g_detector->setNmsThreshold(g_config.nms_threshold);
    MA_LOGI(TAG, "SCRFD input  %s", g_detector->inputDescription().c_str());
    MA_LOGI(TAG, "SCRFD heads  %s", g_detector->outputDescription().c_str());
    log_mem("after-scrfd");

    g_embedder = new ArcFaceRunner();
    if (!g_embedder->init(g_config.embed_model, &err)) {
        MA_LOGE(TAG, "Face embedder: %s", err.c_str());
        return false;
    }
    MA_LOGI(TAG, "ArcFace input %s", g_embedder->inputDescription().c_str());
    log_mem("after-arcface");

    // Liveness last, and never fatal: the model ION pool is the binding
    // constraint on this device, and recognition is still useful without an
    // anti-spoof stage while the reverse is not true.
    //
    // Three separate degradations are possible and all of them are survivable:
    //   V1SE missing   -> single-head texture ensemble.
    //   FaceMesh fails -> no blink evidence; texture + motion still fuse.
    //   V2 fails       -> the whole stage is disabled.
    g_liveness = new LivenessRunner();
    g_facemesh = new FacemeshLiveness();
    g_lcfg.enabled = g_config.liveness_enabled;
    if (g_config.liveness_enabled) {
        log_mem("before-liveness");
        if (!g_liveness->initOptional(g_config.liveness_model, g_config.liveness_v1se_model,
                                      &err)) {
            MA_LOGW(TAG, "Liveness disabled: %s", err.c_str());
            g_config.liveness_enabled = false;
            g_lcfg.enabled = false;
        } else {
            if (!err.empty()) MA_LOGW(TAG, "Liveness ensemble: %s", err.c_str());
            MA_LOGI(TAG, "Liveness input %s (ensemble=%d)",
                    g_liveness->inputDescription().c_str(),
                    static_cast<int>(g_liveness->ensembleAvailable()));
        }
        log_mem("after-liveness");

        if (g_config.liveness_enabled && g_lcfg.facemesh_interval > 0) {
            log_mem("before-facemesh");
            std::string fm_err;
            if (!g_facemesh->initOptional(g_config.facemesh_model, &fm_err)) {
                MA_LOGW(TAG, "FaceMesh blink evidence disabled: %s", fm_err.c_str());
                g_lcfg.facemesh_interval = 0;
            } else {
                MA_LOGI(TAG, "FaceMesh input %s (every %d frames)",
                        g_facemesh->inputDescription().c_str(), g_lcfg.facemesh_interval);
            }
            log_mem("after-facemesh");
        }
        MA_LOGI(TAG,
                "Liveness v2: min_samples=%d ema=%.2f window=%.2fs timeout=%.2fs "
                "floor=%.4f high=%.4f corr=[%.2f,%.2f] ear=%.2f blink=[%d,%d] "
                "w=[t %.2f m %.2f d %.2f] bonus=%.2f T=[live %.2f spoof %.2f]",
                g_lcfg.min_samples, g_lcfg.texture_ema_alpha, g_lcfg.motion_window_sec,
                g_lcfg.timeout_sec, g_lcfg.motion_noise_floor, g_lcfg.motion_high,
                g_lcfg.correlation_low, g_lcfg.correlation_high, g_lcfg.ear_threshold,
                g_lcfg.blink_min_samples, g_lcfg.blink_max_samples, g_lcfg.w_texture,
                g_lcfg.w_motion, g_lcfg.w_depth, g_lcfg.blink_bonus, g_lcfg.t_live,
                g_lcfg.t_spoof);
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
    g_capture.stop();
    delete g_commands;   g_commands = nullptr;
    delete g_tracker;    g_tracker = nullptr;
    delete g_gallery;    g_gallery = nullptr;
    delete g_facemesh;   g_facemesh = nullptr;
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
            case CommandOp::LivenessCapture: {
                if (!g_config.liveness_enabled || !g_liveness->available()) {
                    publish_ack(cmd.request_id, "liveness_capture", false,
                                "liveness stage is disabled; nothing to capture");
                    break;
                }
                const uint64_t now_ms = steady_ms();
                if (!g_capture.start(g_config.liveness_capture_path, cmd.capture_label,
                                     cmd.capture_seconds, g_lcfg.capture_max_sec, now_ms,
                                     &err)) {
                    publish_ack(cmd.request_id, "liveness_capture", false, err);
                    break;
                }
                g_capture_request_id = cmd.request_id;
                AckData data;
                data.addString("state", "started");
                data.addString("label", g_capture.label());
                data.addString("path", g_capture.path());
                data.addUInt("seconds", static_cast<uint64_t>(g_capture.seconds()));
                publish_ack(cmd.request_id, "liveness_capture", true, "", data);
                MA_LOGI(TAG, "Liveness capture armed: label=%s %ds -> %s",
                        g_capture.label().c_str(), g_capture.seconds(),
                        g_capture.path().c_str());
                break;
            }
        }
    }
}

// Publishes the closing acknowledgement for a capture that has run out its
// deadline, then releases the file.
static void finish_capture_if_expired(uint64_t now_ms) {
    if (!g_capture.active() || !g_capture.expired(now_ms)) return;
    AckData data;
    data.addString("state", "stopped");
    data.addString("label", g_capture.label());
    data.addString("path", g_capture.path());
    data.addUInt("rows", g_capture.rows());
    data.addUInt("seconds", static_cast<uint64_t>(g_capture.seconds()));
    MA_LOGI(TAG, "Liveness capture finished: %llu row(s) -> %s",
            static_cast<unsigned long long>(g_capture.rows()), g_capture.path().c_str());
    g_capture.stop();
    publish_ack(g_capture_request_id, "liveness_capture", true, "", data);
    g_capture_request_id.clear();
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
                // Enrollment is a single-shot decision on one frame, so it uses
                // the ensemble mean against the legacy threshold rather than the
                // temporal fusion, which needs a track that persists.
                TextureScores tex;
                if (g_liveness->inferRgbFrame(rgb, w, h, eligible[0]->bbox, tex, &err)) {
                    live = tex.mean >= g_config.liveness_threshold;
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

// Applies a v2 verdict to a published row.
//
// Identity leakage rule (spec pitfall 3): a name is actionable only once the
// track is `live`. While `pending` the name is withheld -- not because the match
// is wrong, but because a consumer that acts on it has skipped the anti-spoof
// stage entirely. On `spoof` the name is withheld AND the accumulated identity
// evidence on the track is discarded.
static void apply_liveness(RecognitionResult& r, TrackState& st, const LivenessResult& lr) {
    r.has_liveness_v2 = true;
    r.liveness = lr;
    // Legacy contract: null while pending, so an old consumer can still tell
    // "not checked" from "checked and failed".
    r.has_liveness = !lr.isPending();
    r.live = lr.isLive();
    r.liveness_score = lr.has_texture ? lr.texture : 0.f;

    if (lr.isSpoof()) {
        r.name.clear();
        r.stable = false;
        r.reason = "spoof";
        st.candidate_name.clear();
        st.consecutive_matches = 0;
        st.cached.name.clear();
        st.cached.stable = false;
    } else if (lr.isPending()) {
        r.name.clear();
        r.stable = false;
        if (r.reason.empty()) r.reason = "liveness_pending";
    }
}

// Runs every liveness sampler that is due for this face and returns the fused
// verdict. Called BEFORE the embedding-due early return, because motion needs a
// sample on every frame the face is visible and FaceMesh runs on its own
// cadence -- the pre-v2 code only ever reached liveness on ArcFace frames.
static bool sample_liveness(const uint8_t* rgb, int w, int h, const FaceDetection& d,
                            TrackState& st, uint64_t now_ms, bool texture_due,
                            LivenessResult& out) {
    if (!g_config.liveness_enabled || !g_liveness || !g_liveness->available()) return false;

    LivenessState& ls = st.liveness;
    if (!ls.started) {
        ls.started = true;
        ls.first_seen_ms = now_ms;
    }

    updateMotion(ls, now_ms, d.landmarks, d.bbox.shortSide(), g_lcfg);

    bool have_ear = false;
    float ear = 0.f;
    if (g_facemesh && g_facemesh->available() && g_lcfg.facemesh_interval > 0 &&
        (g_frame_id % static_cast<uint32_t>(g_lcfg.facemesh_interval)) == 0) {
        std::string fm_err;
        if (g_facemesh->infer(rgb, w, h, d.bbox, ear, &fm_err)) {
            have_ear = true;
        } else if (g_config.verbose) {
            MA_LOGW(TAG, "facemesh failed: %s", fm_err.c_str());
        }
    }
    updateBlink(ls, have_ear, ear, g_lcfg);

    TextureScores tex;
    bool have_tex = false;
    if (texture_due) {
        std::string tex_err;
        if (g_liveness->inferRgbFrame(rgb, w, h, d.bbox, tex, &tex_err)) {
            have_tex = true;
            ls.last_v2 = tex.v2;
            ls.last_v1se = tex.v1se;
            ls.last_v1se_valid = tex.has_v1se;
            updateTexture(ls, tex.mean, g_lcfg);
        } else if (g_config.verbose) {
            MA_LOGW(TAG, "liveness failed: %s", tex_err.c_str());
        }
    }

    out = fuseLiveness(ls, now_ms, g_lcfg);

    // One capture row per eligible face per captured frame, written after every
    // available feature has been updated. Features only, never pixels.
    if (g_capture.active()) {
        CaptureRow row;
        row.has_v2 = have_tex;
        row.p_tex_v2 = tex.v2;
        row.has_v1se = have_tex && tex.has_v1se;
        row.p_tex_v1se = tex.v1se;
        row.has_motion = ls.motion.valid;
        row.motion_residual = ls.motion.residual;
        row.correlation = ls.motion.correlation;
        row.has_ear = have_ear;
        row.ear = ear;
        row.blink = ls.blink_this_frame;
        row.face_px_size = d.bbox.shortSide();
        row.track_id = d.track_id;
        row.ts = wall_seconds();
        std::string cap_err;
        if (!g_capture.append(row, &cap_err)) {
            MA_LOGW(TAG, "capture append failed: %s", cap_err.c_str());
        }
    }
    return true;
}

// Wall time spent inside sample_liveness() for the frame currently being
// processed, so the cost of the v2 stage is visible in the per-frame log rather
// than hidden inside "recognize".
static long long g_liveness_ms = 0;

static void recognize(const uint8_t* rgb, int w, int h,
                      const std::vector<FaceDetection>& dets,
                      std::vector<RecognitionResult>& out) {
    out.clear();
    out.reserve(dets.size());
    const uint64_t now_ms = steady_ms();
    g_liveness_ms = 0;

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
            r.has_liveness_v2 = false;
            st.cached = r;
            st.candidate_name.clear();
            st.consecutive_matches = 0;
            out.push_back(r);
            continue;
        }
        r.gated = false;

        const bool due = !st.embedded ||
                         (g_frame_id - st.last_embed_frame) >= static_cast<uint32_t>(g_config.embed_interval);

        // Liveness sampling happens for EVERY eligible face on EVERY frame, and
        // deliberately before the "not due for a new embedding" early return
        // below: motion needs a per-frame keypoint sample, FaceMesh runs on its
        // own interval, and neither has anything to do with the ArcFace cadence.
        // Texture follows embed_interval to stay affordable, except during a
        // calibration capture, which forces both heads every frame.
        LivenessResult lr;
        const auto lv_start = std::chrono::high_resolution_clock::now();
        const bool has_lr =
            sample_liveness(rgb, w, h, d, st, now_ms, due || g_capture.active(), lr);
        g_liveness_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::high_resolution_clock::now() - lv_start)
                             .count();

        if (!due) {
            if (has_lr) apply_liveness(r, st, lr);
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
            if (has_lr) apply_liveness(r, st, lr);
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
            if (has_lr) apply_liveness(r, st, lr);
            out.push_back(r);
            if (g_config.verbose) MA_LOGW(TAG, "embed failed: %s", err.c_str());
            continue;
        }

        // Liveness already ran above, on every frame, ahead of the embedding
        // cadence. Nothing to do here but match.
        const GalleryMatch m = g_gallery->match(emb);
        r.score = m.score;

        if (m.matched) {
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

        // The identity is cached BEFORE the liveness gate, so a track that is
        // still `pending` keeps accumulating stability and publishes the name
        // the instant the verdict turns live. apply_liveness() wipes both the
        // published row and this cache when the verdict is `spoof`.
        st.cached = r;
        if (has_lr) apply_liveness(r, st, lr);
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
        MA_LOGI(TAG,
                "Frame %u: %zu faces, detect=%lldms, recognize=%lldms, liveness=%lldms, "
                "total=%lldms",
                g_frame_id, faces.size(), static_cast<long long>(detect_time),
                static_cast<long long>(recog_time), g_liveness_ms,
                static_cast<long long>(total_time));
        for (const auto& f : faces) {
            MA_LOGI(TAG, "  Face[%d]: name=%s score=%.3f det=%.2f live=%s stable=%d gated=%d%s%s",
                    f.track_id, f.name.empty() ? "-" : f.name.c_str(), f.score, f.det_score,
                    f.has_liveness ? (f.live ? "yes" : "no") : "n/a",
                    static_cast<int>(f.stable), static_cast<int>(f.gated),
                    f.reason.empty() ? "" : " reason=", f.reason.c_str());
            if (f.has_liveness_v2) {
                MA_LOGI(TAG,
                        "    liveness: decision=%s reason=%s score=%s texture=%s motion=%s "
                        "blink=%d",
                        f.liveness.decision.c_str(), f.liveness.reason.c_str(),
                        f.liveness.has_score ? std::to_string(f.liveness.score).c_str() : "null",
                        f.liveness.has_texture ? std::to_string(f.liveness.texture).c_str()
                                               : "null",
                        f.liveness.has_motion ? std::to_string(f.liveness.motion).c_str()
                                              : "null",
                        static_cast<int>(f.liveness.blink));
            }
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
        // Capture bookkeeping outside process_frame() for the same reason the
        // command drain is: a stalled camera must not strand an open file or
        // withhold the closing acknowledgement.
        const uint64_t now_ms = steady_ms();
        g_capture.tick(now_ms);
        finish_capture_if_expired(now_ms);
    }

    cleanup();
    MA_LOGI(TAG, "Face Recognition Application terminated");
    return 0;
}
