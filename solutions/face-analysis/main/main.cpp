#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>

#include <sscma.h>
#include <video.h>
#include <debug_stream.h>
#include <ha_mqtt.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"
#include "onvif_service_bringup.h"
#include "rtsp_server.h"

#include "face_detector.h"
#include "attribute_analyzer.h"
#include "mqtt_payload.h"
#include "privacy_blur.h"

using namespace ma;
using namespace face_analysis;
using privacy_blur::PrivacyBlur;
using privacy_blur::PrivacyBlurConfig;

#define TAG "face-analysis"

// Default configuration
static struct {
    // Model paths (auto-detects FairFace vs InsightFace format)
    std::string face_model = "/userdata/local/models/yolov8n_face_cv181x_int8.cvimodel";
    std::string genderage_model = "/userdata/local/models/fairface_int8.cvimodel";
    std::string emotion_model = "/userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel";
    // PFLD landmark is optional — only useful if AGR is InsightFace (alignment-sensitive).
    // FairFace is bin-classification and tolerates loose bbox crops, so default off.
    std::string landmark_model = "";

    // Detection parameters
    float face_threshold = 0.4f;

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/face-analysis/results";

    // Video configuration
    int inference_width = 640;
    int inference_height = 480;
    int inference_fps = 10;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Blur configuration
    int max_regions = 12;

    // Emotion runs every N frames (1 = every frame, 2 = every 2 frames, ...)
    int emotion_interval = 2;

    // Debug stream (H.264-over-WS + results JSON for the supervisor console)
    bool enable_debug = true;
    int debug_port = 8001;

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool enable_blur = true;
    bool verbose = false;
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static FaceDetector* g_face_detector = nullptr;
static AttributeAnalyzer* g_attribute_analyzer = nullptr;
static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static PrivacyBlur* g_face_blur = nullptr;
/* Temporary placement diagnostics, off unless BLUR_TRACE is set in the
 * environment. Behind a switch rather than always-on because it prints once per
 * face per frame, which would bury the log it is meant to be read in. */
static const bool g_blur_trace = (getenv("BLUR_TRACE") != nullptr);
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Face Analysis for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -f, --face-model PATH     Face detection model (default: %s)\n", g_config.face_model.c_str());
    printf("  -g, --genderage-model PATH GenderAge model (default: %s)\n", g_config.genderage_model.c_str());
    printf("  -e, --emotion-model PATH  Emotion model (default: %s)\n", g_config.emotion_model.c_str());
    printf("  -l, --landmark-model PATH Landmark model (default: %s)\n", g_config.landmark_model.c_str());
    printf("  -t, --threshold FLOAT     Face detection threshold (default: %.2f)\n", g_config.face_threshold);
    printf("  -m, --mqtt-host HOST      MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  -p, --mqtt-port PORT      MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --no-rtsp                 Disable RTSP streaming\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  --no-blur                 Disable face blur on RTSP stream\n");
    printf("  --no-debug                Disable debug WebSocket stream\n");
    printf("  --debug-port PORT         Debug WebSocket port (default: %d)\n", g_config.debug_port);
    printf("  --max-regions N           Max blur regions (1-8, default: %d)\n", g_config.max_regions);
    printf("  --emotion-interval N      Run emotion every N frames (default: %d)\n", g_config.emotion_interval);
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"face-model", required_argument, 0, 'f'},
        {"genderage-model", required_argument, 0, 'g'},
        {"emotion-model", required_argument, 0, 'e'},
        {"landmark-model", required_argument, 0, 'l'},
        {"threshold", required_argument, 0, 't'},
        {"mqtt-host", required_argument, 0, 'm'},
        {"mqtt-port", required_argument, 0, 'p'},
        {"no-rtsp", no_argument, 0, 1},
        {"no-mqtt", no_argument, 0, 2},
        {"no-blur", no_argument, 0, 3},
        {"max-regions", required_argument, 0, 4},
        {"emotion-interval", required_argument, 0, 5},
        {"no-debug", no_argument, 0, 6},
        {"debug-port", required_argument, 0, 7},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:g:e:l:t:m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                g_config.face_model = optarg;
                break;
            case 'g':
                g_config.genderage_model = optarg;
                break;
            case 'e':
                g_config.emotion_model = optarg;
                break;
            case 'l':
                g_config.landmark_model = optarg;
                break;
            case 't':
                g_config.face_threshold = std::stof(optarg);
                break;
            case 'm':
                g_config.mqtt_host = optarg;
                break;
            case 'p':
                g_config.mqtt_port = std::stoi(optarg);
                break;
            case 1:
                g_config.enable_rtsp = false;
                break;
            case 2:
                g_config.enable_mqtt = false;
                break;
            case 3:
                g_config.enable_blur = false;
                break;
            case 4:
                g_config.max_regions = std::stoi(optarg);
                break;
            case 5:
                g_config.emotion_interval = std::stoi(optarg);
                break;
            case 6:
                g_config.enable_debug = false;
                break;
            case 7:
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

static bool init_models() {
    // Initialize face detector
    g_face_detector = new FaceDetector();
    if (!g_face_detector->init(g_config.face_model)) {
        MA_LOGE(TAG, "Failed to initialize face detector");
        return false;
    }
    g_face_detector->setThreshold(g_config.face_threshold);
    MA_LOGI(TAG, "Face detector initialized (input: %dx%d)",
            g_face_detector->getInputWidth(), g_face_detector->getInputHeight());

    // Initialize attribute analyzer
    g_attribute_analyzer = new AttributeAnalyzer();
    if (!g_attribute_analyzer->init(g_config.genderage_model, g_config.emotion_model, g_config.landmark_model)) {
        MA_LOGE(TAG, "Failed to initialize attribute analyzer");
        return false;
    }
    g_attribute_analyzer->setEmotionInterval(g_config.emotion_interval);
    MA_LOGI(TAG, "Attribute analyzer initialized (GenderAge: %s, Emotion: %s)",
            g_attribute_analyzer->isGenderAgeReady() ? "yes" : "no",
            g_attribute_analyzer->isEmotionReady() ? "yes" : "no");

    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            // Set channel 0 for inference
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Apply the configured inference frame rate to the channel. Without
            // this the channel keeps its 30fps default, so the capture->inference
            // FIFO is sized/filled at 30fps while inference consumes far slower —
            // wasting CPU copying frames that get dropped.
            value.i32 = g_config.inference_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

            // Set inference resolution
            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // Disable physical address mode - attribute analysis needs CPU access to frame data
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps for inference)",
                    g_config.inference_width, g_config.inference_height, g_config.inference_fps);
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

    // Setup H.264 streaming channel (CH2)
    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    // Register RTSP handler (debug_stream registers its own consumer, idx 1)
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);

    // Initialize RTSP server
    initRtsp((0x01 << VIDEO_CH2));

    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps)",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps);

    return true;
}

// Assemble the debug /results envelope (shared debug_stream builder).
// FaceInfo x/y are normalized top-left; convert to center-based pixels.
// box[5] is "face" (aligned with the other apps' class-name labels); the
// parallel labels[] carries the per-face attributes.
// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<AnalyzedFace>& faces,
                                            float inference_time_ms) {
    std::vector<debug_stream_box_t> boxes;
    std::vector<std::string> labels;
    boxes.reserve(faces.size());
    labels.reserve(faces.size());
    for (const auto& af : faces) {
        const auto& f = af.face;
        boxes.push_back({(f.x + f.w * 0.5f) * g_config.inference_width,
                         (f.y + f.h * 0.5f) * g_config.inference_height,
                         f.w * g_config.inference_width,
                         f.h * g_config.inference_height,
                         f.score, "face"});
        const auto& attr = af.attributes;
        // gender · age · race · emotion. race_label is empty for InsightFace
        // (no race head) — skip it there so that path's label is unchanged.
        std::string label = attr.gender + " " + attr.age_label;
        if (!attr.race_label.empty()) label += " " + attr.race_label;
        label += std::string(" ") + getEmotionName(attr.emotion);
        labels.push_back(label);
    }
    // The inference channel is letterboxed vs the 16:9 debug video (stream);
    // remap boxes into the stream frame so the overlay aligns with the video.
    debug_stream_letterbox_to_display(boxes, g_config.inference_width, g_config.inference_height,
                                      g_config.stream_width, g_config.stream_height);
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      boxes, &labels);
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "face-analysis";
    opts.client_id     = "recamera-face-analysis";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant MQTT Discovery entity table (field names must match the
    // results JSON built by mqtt_payload.cpp).
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "face_count", "Face Count",
                     "{{ value_json.face_count }}",
                     "", "", "measurement"},
        EntityConfig{EntityType::BinarySensor, "occupancy", "Face Detected",
                     "{{ 'ON' if value_json.face_count > 0 else 'OFF' }}",
                     "occupancy", "", ""},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }

    // Rides the connection above rather than opening a second one; the switch
    // lives in /userdata/local/onvif.conf, written by the console.
    g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
    if (g_onvif_meta.enabled()) {
        MA_LOGI(TAG, "ONVIF metadata: %s (every %ums)",
                g_onvif_meta.topic().c_str(), g_onvif_meta.config().interval_ms);
    }

    return true;
}

static bool init_blur() {
    if (!g_config.enable_blur) {
        return true;
    }

    if (!g_config.enable_rtsp) {
        MA_LOGW(TAG, "Face blur requires RTSP streaming, ignoring --blur");
        g_config.enable_blur = false;
        return true;
    }

    PrivacyBlurConfig cfg;
    loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, nullptr);

    /* No device-wide config: stay off, which is what this application has
     * always shipped as (its own conf carried BLUR_ENABLED=0). Turning masking
     * on for existing devices merely because the setting moved would be a
     * surprising change to what their stream looks like. A blur.conf written
     * by the console is an explicit decision and is honoured in both
     * directions. */
    if (!cfg.present) cfg.enabled = false;
    cfg.max_regions = g_config.max_regions;

    g_face_blur = new PrivacyBlur();
    if (!g_face_blur->init(cfg, g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "Failed to initialize face blur");
        delete g_face_blur;
        g_face_blur = nullptr;
        return false;
    }

    MA_LOGI(TAG, "Face blur ready (backend=%s, max_regions=%d, enabled=%d)",
            cfg.backend.c_str(), cfg.max_regions, (int)cfg.enabled);
    return true;
}

static void cleanup() {
    if (g_face_blur) {
        g_face_blur->deinit();
        delete g_face_blur;
        g_face_blur = nullptr;
    }

    if (g_camera) {
        g_camera->stopStream();
    }

    if (g_config.enable_debug) {
        debug_stream_destroy();
    }

    onvif_service_stop();

    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    if (g_attribute_analyzer) {
        delete g_attribute_analyzer;
        g_attribute_analyzer = nullptr;
    }

    if (g_face_detector) {
        delete g_face_detector;
        g_face_detector = nullptr;
    }

    MA_LOGI(TAG, "Cleanup completed");
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

    // Step 1: Face detection
    auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<FaceInfo> faces = g_face_detector->detect(&frame);
    auto detect_end = std::chrono::high_resolution_clock::now();
    auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

    // Step 2: Feed face detections to the privacy mask. Before returnFrame()
    // below, because the pixelating backend averages the pixels it hides and
    // frame.data is invalid once the frame goes back to the camera.
    if (g_config.enable_blur && g_face_blur) {
        /*
         * The mask has to travel the same road as the overlay boxes, and for
         * the same reason.
         *
         * FaceInfo is normalised against the *inference* frame (640x480 here),
         * the mask is applied on the *stream* frame (1280x720), and the two
         * have different aspect ratios. The camera fits the sensor content into
         * each channel preserving aspect, so the inference frame carries
         * letterbox bars the stream does not: a normalised coordinate means a
         * different place in each. Handing the inference-normalised value
         * straight to a component initialised with the stream dimensions treats
         * one as the other and puts the mask somewhere the face is not.
         *
         * debug_stream_letterbox_to_display() already does this conversion for
         * the console overlay a few lines up. Reusing it -- rather than
         * repeating the arithmetic -- is what keeps the drawn box and the mask
         * from drifting apart later: they now derive from one implementation.
         */
        std::vector<debug_stream_box_t> px;
        px.reserve(faces.size());
        for (const auto& f : faces) {
            /* FaceInfo carries the TOP-LEFT corner -- FaceDetector normalises
             * every model to that because the attribute analyzer crops from it
             * -- while BlurBox is centre-based like every other detector in the
             * tree. Converting here rather than changing either convention: the
             * cropping code downstream depends on top-left, and the component
             * is shared with applications whose boxes are already centred. */
            px.push_back({(f.x + f.w * 0.5f) * g_config.inference_width,
                          (f.y + f.h * 0.5f) * g_config.inference_height,
                          f.w * g_config.inference_width,
                          f.h * g_config.inference_height,
                          f.score, std::string()});
        }
        debug_stream_letterbox_to_display(px, g_config.inference_width, g_config.inference_height,
                                          g_config.stream_width, g_config.stream_height);

        std::vector<privacy_blur::BlurBox> blur_boxes;
        blur_boxes.reserve(px.size());
        for (const auto& b : px) {
            blur_boxes.push_back({b.x / g_config.stream_width, b.y / g_config.stream_height,
                                  b.w / g_config.stream_width, b.h / g_config.stream_height,
                                  b.score});
        }

        /* Temporary: prints the raw detection next to what the mask is actually
         * asked to cover, so a misplaced mask can be read off the log instead of
         * inferred from a screenshot. Remove once the placement is settled. */
        if (g_blur_trace && !faces.empty()) {
            for (size_t i = 0; i < faces.size(); ++i) {
                MA_LOGI(TAG,
                        "blur-trace face[%zu] inf_norm(tl)=(%.3f,%.3f,%.3f,%.3f) -> "
                        "disp_px(c)=(%.1f,%.1f,%.1f,%.1f) -> stream_norm(c)=(%.3f,%.3f,%.3f,%.3f)",
                        i, faces[i].x, faces[i].y, faces[i].w, faces[i].h,
                        px[i].x, px[i].y, px[i].w, px[i].h,
                        blur_boxes[i].x, blur_boxes[i].y, blur_boxes[i].w, blur_boxes[i].h);
            }
        }

        g_face_blur->onDetection(blur_boxes, &frame);
    }

    // Step 3: Attribute analysis for each face
    auto analyze_start = std::chrono::high_resolution_clock::now();
    std::vector<AnalyzedFace> analyzed_faces = g_attribute_analyzer->analyzeAll(&frame, faces);
    auto analyze_end = std::chrono::high_resolution_clock::now();
    auto analyze_time = std::chrono::duration_cast<std::chrono::milliseconds>(analyze_end - analyze_start).count();

    // Offer the frame for /snapshot.jpg. Returns after one atomic load unless a
    // snapshot client asked recently; must precede returnFrame(), after which
    // frame.data is invalid. Unannotated: the video path is unannotated too and
    // overlays are drawn client-side.
    //
    // The privacy mask, though, has to be applied here in software. The RGN
    // mask lives in the VPSS->VENC path and so covers RTSP and the console's
    // debug video only; this buffer is the inference frame and goes to the JPEG
    // encoder untouched by it. Left as it was, the device would serve a masked
    // video stream and an unmasked still of the same scene -- from the very URL
    // ONVIF advertises as GetSnapshotUri, so a client honouring the mask on the
    // stream could pull the faces it hides from the same device.
    //
    // The detections are used directly, without the letterbox conversion the
    // mask path needs: FaceInfo is normalised against this exact frame, so
    // converting to stream coordinates and back would only add a way to be
    // wrong.
    if (debug_stream_snapshot_armed()) {
        if (g_config.enable_blur && g_face_blur != nullptr && g_face_blur->enabled() &&
            !faces.empty()) {
            std::vector<privacy_blur::BlurBox> snap_boxes;
            snap_boxes.reserve(faces.size());
            for (const auto& f : faces) {
                snap_boxes.push_back({f.x + f.w * 0.5f, f.y + f.h * 0.5f, f.w, f.h, f.score});
            }
            privacy_blur::pixelateRgb888(frame.data, frame.width, frame.height, snap_boxes,
                                         g_face_blur->blockPx());
        }
        debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
    }

    // Return frame to camera
    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Step 4: Publish results via MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        std::string payload = buildResultJson(timestamp_ms, g_frame_id, analyzed_faces, static_cast<float>(total_time));
        g_mqtt_publisher->publishResultsJson(payload);

        // Additionally publish the same detections in ONVIF's analytics
        // representation, on its own topic and its own rate limit. The
        // recamera/<app>/results contract above is consumed by SenseCraft and
        // does not change. Only boxes and a class for now; face attributes map
        // onto tt:HumanFace and can follow once this path has proven itself.
        if (g_onvif_meta.take(timestamp_ms)) {
            std::vector<onvif_box_t> ob;
            ob.reserve(analyzed_faces.size());
            for (const auto& af : analyzed_faces) {
                const auto& f = af.face;
                ob.push_back({(f.x + f.w * 0.5f) * g_config.inference_width,
                              (f.y + f.h * 0.5f) * g_config.inference_height,
                              f.w * g_config.inference_width,
                              f.h * g_config.inference_height,
                              f.score, "HumanFace"});
            }
            g_mqtt_publisher->publishText(
                g_onvif_meta.topic(),
                onvif_meta_to_json(onvif_meta_from_boxes(
                    timestamp_ms, "FaceAnalysis",
                    g_config.inference_width, g_config.inference_height, ob)));
        }
    }

    // Step 4.5: Push the same inference result to debug WS clients (sscma-node
    // format). debug_stream is lazy: skip building JSON when nobody listens.
    if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
        std::string debug_json = build_debug_results_json(timestamp_ms, g_frame_id, analyzed_faces,
                                                          static_cast<float>(total_time));
        debug_stream_publish_result(debug_json.c_str(), debug_json.size());
    }

    // Log results
    if (g_config.verbose || !analyzed_faces.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu faces, detect=%lldms, analyze=%lldms, total=%lldms",
                g_frame_id, analyzed_faces.size(), detect_time, analyze_time, total_time);

        for (const auto& face : analyzed_faces) {
            if (face.attributes.is_fairface) {
                MA_LOGI(TAG, "  Face[%d]: age=%s, gender=%s(%.2f), race=%s(%.2f), emotion=%s(%.2f)",
                        face.face.id,
                        face.attributes.age_label.c_str(),
                        face.attributes.gender.c_str(),
                        face.attributes.gender_confidence,
                        face.attributes.race_label.c_str(),
                        face.attributes.race_confidence,
                        getEmotionName(face.attributes.emotion),
                        face.attributes.emotion_confidence);
            } else {
                MA_LOGI(TAG, "  Face[%d]: age=%s, gender=%s(%.2f), emotion=%s(%.2f)",
                        face.face.id,
                        face.attributes.age_label.c_str(),
                        face.attributes.gender.c_str(),
                        face.attributes.gender_confidence,
                        getEmotionName(face.attributes.emotion),
                        face.attributes.emotion_confidence);
            }
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (!parse_args(argc, argv)) {
        return 1;
    }

    MA_LOGI(TAG, "Starting Face Analysis Application");
    MA_LOGI(TAG, "Face model: %s", g_config.face_model.c_str());
    MA_LOGI(TAG, "GenderAge model: %s", g_config.genderage_model.c_str());
    MA_LOGI(TAG, "Emotion model: %s", g_config.emotion_model.c_str());

    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize components
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

    // Debug stream (non-fatal: on failure run without it)
    if (g_config.enable_debug && debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }

    if (!init_mqtt()) {
        MA_LOGE(TAG, "MQTT initialization failed");
        cleanup();
        return 1;
    }

    // Start camera streaming
    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // Start video streaming if enabled
    if (g_config.enable_rtsp) {
        startVideo();
    }

    // Initialize blur AFTER video pipeline is started (RGN needs VPSS channel running)
    if (!init_blur()) {
        MA_LOGW(TAG, "Face blur initialization failed, continuing without blur");
        g_config.enable_blur = false;
    }

    // ONVIF discovery + Device/Media2 services. After startVideo() on purpose:
    // GetProfiles and GetStreamUri are answered from the running RTSP server's
    // session list, so bringing this up earlier advertises zero profiles and a
    // VMS shows a camera with no video.
    if (onvif_service_bringup(g_onvif_meta.config(), ha_mqtt::readDeviceIdentifier(),
            "reCamera", g_config.enable_debug ? g_config.debug_port : 0) == 0 &&
        onvif_service_soap_running()) {
        MA_LOGI(TAG, "ONVIF service on port %d", g_onvif_meta.config().service_port);
    }

    MA_LOGI(TAG, "Face analysis running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:554/live");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    // Main processing loop
    while (g_running.load()) {
        process_frame();
    }

    // Cleanup
    cleanup();

    MA_LOGI(TAG, "Face Analysis Application terminated");
    return 0;
}
