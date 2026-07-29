// fitness-trainer -- rep counting and form feedback on reCamera.
//
// Pipeline: camera -> YOLO11-Pose (TPU) -> joint angles -> exercise state
// machine -> MQTT / debug WebSocket, with the scene going out over RTSP
// untouched.
//
// The exercise is selected from the console (or by editing the config file),
// see app_config.h. Adding an exercise is a change to exercise.cpp and the
// manifest's enum; this file does not know the list.

#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "rtsp_server.h"

#include "app_config.h"
#include "exercise.h"
#include "pose_detector.h"
#include "result_payload.h"

using namespace ma;
using namespace fitness;

#define TAG "fitness-trainer"

static struct {
    // Empty = search the candidate list below. An explicit -m / MODEL_PATH
    // wins and is used verbatim: if an operator names a file, failing loudly
    // beats silently running a different model than the one they asked for.
    std::string model_path;

    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/fitness-trainer/results";

    int inference_width = 640;
    int inference_height = 640;
    int inference_fps = 15;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    bool enable_debug = true;
    int debug_port = 8001;
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;

    // CLI overrides for the config file (empty / <=0 means "not set").
    std::string cli_mode;
    int cli_target_reps = 0;
    int cli_target_sets = 0;
} g_config;

// Where to look for the pose model, in order.
//
// This app is the only one in the gallery that ships no model of its own: it
// uses the YOLO11n-Pose that comes with the reCamera console, which is the
// official Ultralytics build and would only be duplicated by packaging a
// second copy. The cost of that choice is this list -- a device whose console
// predates the model, or one where it was installed separately, must still
// find it. /userdata is checked too because that is where both the SenseCraft
// deployer and the console's cloud install put model files.
static const char* MODEL_CANDIDATES[] = {
    "/usr/share/supervisor/models/yolo11n_pose_cv181x_int8.cvimodel",
    "/userdata/local/models/yolo11n_pose_cv181x_int8.cvimodel",
};

// First candidate that exists, or empty when none do.
static std::string findPoseModel() {
    for (const char* p : MODEL_CANDIDATES) {
        struct stat st;
        if (::stat(p, &st) == 0 && S_ISREG(st.st_mode)) {
            return p;
        }
    }
    return std::string();
}

static const char* APP_CONFIG_PATH = "/userdata/local/apps/fitness-trainer.config.json";

static std::atomic<bool> g_running(true);
static PoseDetector* g_detector = nullptr;
static std::unique_ptr<Exercise> g_exercise;
static ha_mqtt::MqttPublisher* g_mqtt = nullptr;
static Camera* g_camera = nullptr;
static ConfigWatcher* g_watcher = nullptr;
static uint32_t g_frame_id = 0;
static std::chrono::steady_clock::time_point g_start_time;
static double g_last_person_seen = -1.0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Fitness Trainer for reCamera -- pose-based rep counting\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH        Pose model (default: %s)\n", g_config.model_path.c_str());
    printf("  --mode NAME             Exercise: squat | push_up | hammer_curl\n");
    printf("                          (overrides the console config for this run)\n");
    printf("  --reps N                Target reps per set\n");
    printf("  --sets N                Target sets\n");
    printf("  --mqtt-host HOST        MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT        MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC      MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-rtsp               Disable RTSP streaming\n");
    printf("  --no-mqtt               Disable MQTT publishing\n");
    printf("  --no-debug              Disable debug WebSocket stream\n");
    printf("  --debug-port PORT       Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  -v, --verbose           Verbose logging\n");
    printf("  -h, --help              This message\n");
    printf("\nExercise selection lives in %s\n", APP_CONFIG_PATH);
    printf("RTSP: rtsp://<device_ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"mode", required_argument, 0, 1},
        {"reps", required_argument, 0, 2},
        {"sets", required_argument, 0, 3},
        {"mqtt-host", required_argument, 0, 4},
        {"mqtt-port", required_argument, 0, 5},
        {"mqtt-topic", required_argument, 0, 6},
        {"no-rtsp", no_argument, 0, 7},
        {"no-mqtt", no_argument, 0, 8},
        {"no-debug", no_argument, 0, 9},
        {"debug-port", required_argument, 0, 10},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': g_config.model_path = optarg; break;
            case 1: g_config.cli_mode = optarg; break;
            case 2: g_config.cli_target_reps = std::stoi(optarg); break;
            case 3: g_config.cli_target_sets = std::stoi(optarg); break;
            case 4: g_config.mqtt_host = optarg; break;
            case 5: g_config.mqtt_port = std::stoi(optarg); break;
            case 6: g_config.mqtt_topic = optarg; break;
            case 7: g_config.enable_rtsp = false; break;
            case 8: g_config.enable_mqtt = false; break;
            case 9: g_config.enable_debug = false; break;
            case 10: g_config.debug_port = std::stoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }

    if (!g_config.cli_mode.empty() && !Exercise::known(g_config.cli_mode)) {
        fprintf(stderr, "Unknown exercise '%s'. Known: ", g_config.cli_mode.c_str());
        for (const auto& id : Exercise::ids()) fprintf(stderr, "%s ", id.c_str());
        fprintf(stderr, "\n");
        return false;
    }
    return true;
}

// Build (or rebuild) the exercise from the current config. Called at startup
// and whenever the config file changes.
static bool applyExercise(const AppConfig& cfg, bool announce) {
    const std::string mode = g_config.cli_mode.empty() ? cfg.mode : g_config.cli_mode;

    if (g_exercise && g_exercise->id() == mode) {
        // Same exercise, possibly new targets: keep the athlete's progress.
        g_exercise->setTargets(g_config.cli_target_reps > 0 ? g_config.cli_target_reps : cfg.target_reps,
                               g_config.cli_target_sets > 0 ? g_config.cli_target_sets : cfg.target_sets);
        return true;
    }

    auto next = Exercise::create(mode);
    if (!next) {
        MA_LOGE(TAG, "Unknown exercise '%s'", mode.c_str());
        return false;
    }
    next->setTargets(g_config.cli_target_reps > 0 ? g_config.cli_target_reps : cfg.target_reps,
                     g_config.cli_target_sets > 0 ? g_config.cli_target_sets : cfg.target_sets);
    g_exercise = std::move(next);

    if (announce) {
        MA_LOGI(TAG, "Exercise switched to %s (%d reps x %d sets)",
                g_exercise->displayName(), g_exercise->targetReps(), g_exercise->targetSets());
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

        MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps)",
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
    opts.app_id        = "fitness-trainer";
    opts.client_id     = "recamera-fitness-trainer";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant discovery. Field names must match buildResultJson().
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "reps", "Reps",
                     "{{ value_json.reps }}", "", "", "measurement"},
        EntityConfig{EntityType::Sensor, "set", "Set",
                     "{{ value_json.set }}", "", "", "measurement"},
        EntityConfig{EntityType::Sensor, "exercise", "Exercise",
                     "{{ value_json.exercise }}", "", "", ""},
        EntityConfig{EntityType::Sensor, "stage", "Stage",
                     "{{ value_json.stage }}", "", "", ""},
        EntityConfig{EntityType::BinarySensor, "athlete_present", "Athlete Present",
                     "{{ 'ON' if value_json.person_detected else 'OFF' }}",
                     "occupancy", "", ""},
        EntityConfig{EntityType::BinarySensor, "workout_complete", "Workout Complete",
                     "{{ 'ON' if value_json.workout_complete else 'OFF' }}",
                     "", "", ""},
    };

    g_mqtt = new ha_mqtt::MqttPublisher();
    if (!g_mqtt->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) g_camera->stopStream();
    if (g_config.enable_debug) debug_stream_destroy();
    if (g_config.enable_rtsp) { deinitRtsp(); deinitVideo(); }
    if (g_mqtt) { g_mqtt->deinit(); delete g_mqtt; g_mqtt = nullptr; }
    g_exercise.reset();
    if (g_detector) { delete g_detector; g_detector = nullptr; }
    if (g_watcher) { delete g_watcher; g_watcher = nullptr; }
    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const double now_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time).count() / 1000.0;

    const Subject* subject = g_detector->detectPrimary(&frame);

    // Offer the raw frame for /snapshot.jpg before returnFrame() invalidates
    // frame.data. Cheap (one atomic load) unless a client asked recently.
    debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    g_camera->returnFrame(frame);

    const auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    g_exercise->update(subject ? &subject->pose : nullptr, now_sec);
    const ExerciseState& st = g_exercise->state();

    // Idle reset: a workout that ended half an hour ago should not still be
    // showing 7/12 when the next person walks in.
    const AppConfig& cfg = g_watcher->config();
    if (subject) {
        g_last_person_seen = now_sec;
    } else if (cfg.idle_reset_seconds > 0 && g_last_person_seen >= 0.0 &&
               now_sec - g_last_person_seen > cfg.idle_reset_seconds &&
               (st.reps > 0 || st.set > 1 || st.workout_complete)) {
        MA_LOGI(TAG, "No athlete for %ds, resetting workout", cfg.idle_reset_seconds);
        g_exercise->reset();
        g_last_person_seen = now_sec;
    }

    PayloadContext ctx;
    ctx.timestamp_ms = static_cast<uint64_t>(timestamp_ms);
    ctx.frame_id = g_frame_id;
    ctx.inference_time_ms = static_cast<float>(inference_ms);
    ctx.person_detected = (subject != nullptr);
    ctx.exercise_id = g_exercise->id();
    ctx.target_reps = g_exercise->targetReps();
    ctx.target_sets = g_exercise->targetSets();
    ctx.infer_w = g_config.inference_width;
    ctx.infer_h = g_config.inference_height;
    ctx.stream_w = g_config.stream_width;
    ctx.stream_h = g_config.stream_height;

    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        // No boxes: a rectangle round the athlete carries no information the
        // skeleton does not, and the label riding on it slid off the bottom of
        // frame exactly when it mattered (at the bottom of a squat). The
        // skeleton and a corner-pinned card go out in the extra members.
        const std::vector<debug_stream_box_t> no_boxes;
        const std::string json = debug_stream_build_results(
            ctx.timestamp_ms, ctx.frame_id, ctx.inference_time_ms,
            g_config.stream_width, g_config.stream_height, no_boxes, nullptr,
            buildDebugExtraJson(ctx, st, subject ? &subject->pose : nullptr));
        debug_stream_publish_result(json.c_str(), json.size());
    }

    if (g_config.enable_mqtt && g_mqtt) {
        const std::string payload = buildResultJson(ctx, st, subject ? &subject->pose : nullptr);
        g_mqtt->publishResultsJson(payload);
    }

    if (st.rep_completed) {
        MA_LOGI(TAG, "%s rep %d/%d (set %d/%d)%s%s",
                g_exercise->id(), st.reps, ctx.target_reps, st.set, ctx.target_sets,
                st.form_warning.empty() ? "" : " -- ", st.form_warning.c_str());
    } else if (g_config.verbose) {
        MA_LOGI(TAG, "Frame %u: %s angle=%.1f stage=%s reps=%d inference=%lldms",
                g_frame_id, ctx.person_detected ? "person" : "no person",
                st.has_angle ? st.angle : 0.0f, st.stage.c_str(), st.reps, inference_ms);
    }

    // Console setConfig restarts the app, so this only fires for out-of-band
    // edits (SSH, Node-RED). Switching the exercise resets the count; changing
    // only the targets keeps it.
    if (g_watcher->poll()) {
        const AppConfig& fresh = g_watcher->config();
        MA_LOGI(TAG, "Config reloaded: mode=%s reps=%d sets=%d",
                fresh.mode.c_str(), fresh.target_reps, fresh.target_sets);
        applyExercise(fresh, true);
        g_detector->setThreshold(fresh.confidence);
        g_detector->setKeypointThreshold(fresh.keypoint_confidence);
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    g_start_time = std::chrono::steady_clock::now();

    g_watcher = new ConfigWatcher(APP_CONFIG_PATH);
    const AppConfig& cfg = g_watcher->loadInitial();

    MA_LOGI(TAG, "Starting Fitness Trainer");

    if (g_config.model_path.empty()) {
        g_config.model_path = findPoseModel();
        if (g_config.model_path.empty()) {
            // Name every path tried: "model not found" without the list sends
            // people looking in the wrong directory.
            MA_LOGE(TAG, "No pose model found. Looked in:");
            for (const char* p : MODEL_CANDIDATES) {
                MA_LOGE(TAG, "  %s", p);
            }
            MA_LOGE(TAG, "Install a reCamera console that ships yolo11n-pose, or set "
                         "MODEL_PATH in /etc/fitness-trainer.conf to a YOLO pose cvimodel.");
            cleanup();
            return 1;
        }
    }
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());

    if (!applyExercise(cfg, false)) { cleanup(); return 1; }
    MA_LOGI(TAG, "Exercise: %s (%d reps x %d sets)",
            g_exercise->displayName(), g_exercise->targetReps(), g_exercise->targetSets());

    g_detector = new PoseDetector();
    if (!g_detector->init(g_config.model_path)) { cleanup(); return 1; }
    g_detector->setThreshold(cfg.confidence);
    g_detector->setKeypointThreshold(cfg.keypoint_confidence);

    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }
    if (!init_mqtt()) { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "Fitness trainer running");
    MA_LOGI(TAG, "RTSP: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
    }

    cleanup();
    MA_LOGI(TAG, "Fitness Trainer terminated");
    return 0;
}
