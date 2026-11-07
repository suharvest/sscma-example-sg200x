#include <chrono>
#include <getopt.h>
#include <string>
#include <vector>

#include <af/app.h>
#include <af/ma_image.h>
#include <ha_mqtt.h>
#include <sscma.h>
#include "onvif_meta.h"
#include "onvif_meta_gate.h"

#include "attribute_analyzer.h"
#include "face_detector.h"
#include "face_tracker.h"
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

    // Attribute accuracy: quality gate + temporal evidence + tracking
    float min_face_px = 64.f;     // Face box short side (source pixels) below which stage 2/3 is skipped
    int min_track_frames = 3;     // Gated-through frames before a verdict is reported as stable
    int track_max_lost = 15;      // Frames a track survives without a match
    float evidence_decay = 1.0f;  // Per-frame decay on accumulated evidence; 1.0 = plain sum

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
static FaceDetector* g_face_detector = nullptr;
static FaceTracker* g_face_tracker = nullptr;
static AttributeAnalyzer* g_attribute_analyzer = nullptr;
static PrivacyBlur* g_face_blur = nullptr;
/* Temporary placement diagnostics, off unless BLUR_TRACE is set in the
 * environment. Behind a switch rather than always-on because it prints once per
 * face per frame, which would bury the log it is meant to be read in. */
static const bool g_blur_trace = (getenv("BLUR_TRACE") != nullptr);
static uint32_t g_frame_id = 0;

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
    printf("  --min-face-px N           Skip attribute inference below this face short side in px (default: %.0f)\n", g_config.min_face_px);
    printf("  --min-track-frames N      Frames of evidence before a verdict is stable (default: %d)\n", g_config.min_track_frames);
    printf("  --track-max-lost N        Frames a track survives unmatched (default: %d)\n", g_config.track_max_lost);
    printf("  --evidence-decay F        Per-frame decay on accumulated evidence (default: %.2f)\n", g_config.evidence_decay);
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
        {"min-face-px", required_argument, 0, 8},
        {"min-track-frames", required_argument, 0, 9},
        {"track-max-lost", required_argument, 0, 10},
        {"evidence-decay", required_argument, 0, 11},
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
            case 8:
                g_config.min_face_px = std::stof(optarg);
                break;
            case 9:
                g_config.min_track_frames = std::stoi(optarg);
                break;
            case 10:
                g_config.track_max_lost = std::stoi(optarg);
                break;
            case 11:
                g_config.evidence_decay = std::stof(optarg);
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

    // Tracking + evidence: face.id becomes a real identity, and attributes are
    // reported off accumulated per-track evidence instead of a single frame.
    FaceTrackerConfig track_cfg;
    track_cfg.max_lost_frames = g_config.track_max_lost;
    g_face_tracker = new FaceTracker(track_cfg);

    EvidenceConfig ev_cfg;
    ev_cfg.min_face_px = g_config.min_face_px;
    ev_cfg.min_track_frames = g_config.min_track_frames;
    ev_cfg.decay = g_config.evidence_decay;
    g_attribute_analyzer->setEvidenceConfig(ev_cfg);
    MA_LOGI(TAG, "Tracking/evidence: min_face_px=%.0f min_track_frames=%d max_lost=%d decay=%.2f",
            ev_cfg.min_face_px, ev_cfg.min_track_frames, track_cfg.max_lost_frames, ev_cfg.decay);
    MA_LOGI(TAG, "Attribute analyzer initialized (GenderAge: %s, Emotion: %s)",
            g_attribute_analyzer->isGenderAgeReady() ? "yes" : "no",
            g_attribute_analyzer->isEmotionReady() ? "yes" : "no");

    return true;
}

// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

static ha_mqtt::ClientOptions mqtt_options() {
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
    return opts;
}

// Runs from App::on_started(): RGN needs the VPSS channel running.
static void init_blur() {
    if (!g_config.enable_blur) {
        return;
    }
    if (!g_config.enable_rtsp) {
        MA_LOGW(TAG, "Face blur requires RTSP streaming, ignoring --blur");
        g_config.enable_blur = false;
        return;
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
        MA_LOGW(TAG, "Face blur initialization failed, continuing without blur");
        delete g_face_blur;
        g_face_blur = nullptr;
        g_config.enable_blur = false;
        return;
    }
    MA_LOGI(TAG, "Face blur ready (backend=%s, max_regions=%d, enabled=%d)",
            cfg.backend.c_str(), cfg.max_regions, (int)cfg.enabled);
}

// Runs from App::on_stopping(): detach the RGN masks while VPSS still exists.
static void deinit_blur() {
    if (g_face_blur) {
        g_face_blur->deinit();
        delete g_face_blur;
        g_face_blur = nullptr;
    }
}

// Per-frame state that outlives the handler: the App's sinks read the boxes
// and JSON after on_frame returns, and the snapshot hook reads the detections.
static struct {
    std::vector<FaceInfo> faces;          // this frame's detections (top-left normalised)
    std::vector<af::Box> boxes;           // debug overlay boxes
    std::vector<std::string> labels;      // debug labels[] (attributes)
    std::string payload;                  // MQTT results JSON
} g_out;

static const char* attribute_label(const AnalyzedFace& af, std::string& out) {
    const auto& attr = af.attributes;
    // A gated face ran no attribute stage at all, so every attribute field
    // is still its default -- and the default emotion is NEUTRAL, which is
    // how a too-small face used to render as a bare "neutral" label. Label
    // it as a plain detection instead: the box and the track are real, the
    // attributes are not.
    if (attr.gated) {
        out = "face";
        return out.c_str();
    }
    // gender · age · race · emotion. race_label is empty for InsightFace
    // (no race head) — skip it there so that path's label is unchanged.
    out = attr.gender + " " + attr.age_label;
    if (!attr.race_label.empty()) out += " " + attr.race_label;
    if (attr.has_emotion) out += std::string(" ") + getEmotionName(attr.emotion);
    return out.c_str();
}

static void on_frame(af::App& app, const af::VideoFrame& frame, af::Result& r) {
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point t) {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t).count());
    };
    const auto start_time = clock::now();
    const uint64_t timestamp_ms = r.wall_ms;
    ma_img_t img = af::to_ma_img(frame);

    // Step 1: Face detection
    const auto detect_start = clock::now();
    g_out.faces = g_face_detector->detect(&img);
    std::vector<FaceInfo>& faces = g_out.faces;
    const long long detect_time = ms_since(detect_start);

    // Step 1.5: Resolve identities. FaceDetector hands out a fresh number per
    // detection per frame, so face.id carries no identity until this runs; from
    // here on it is the track id, which is what the evidence accumulator keys on
    // and what the MQTT "id" field publishes.
    if (g_face_tracker) {
        const std::vector<int> track_ids = g_face_tracker->update(faces);
        for (size_t i = 0; i < faces.size() && i < track_ids.size(); ++i) {
            faces[i].id = track_ids[i];
        }
    }

    // Step 2: Feed face detections to the privacy mask while the frame is held
    // (the pixelating backend averages the pixels it hides).
    if (g_config.enable_blur && g_face_blur) {
        /* FaceInfo is normalised against the inference frame, the mask is
         * applied on the stream frame, and the two have different aspect
         * ratios (letterbox). geometry::toStream() is the one conversion both
         * the mask and the overlay derive from. fromCorner: this app's
         * FaceDetector normalises to the TOP-LEFT corner (the attribute
         * analyzer crops from it), unlike the centre-based detectors elsewhere. */
        std::vector<geometry::InferBox> boxes;
        boxes.reserve(faces.size());
        for (const auto& f : faces) {
            boxes.push_back(geometry::InferBox::fromCorner(f.x, f.y, f.w, f.h, f.score));
        }
        const auto stream_boxes =
            geometry::toStream(boxes, g_config.inference_width, g_config.inference_height,
                               g_config.stream_width, g_config.stream_height);

        /* Temporary: prints the raw detection next to what the mask is actually
         * asked to cover. Remove once the placement is settled. */
        if (g_blur_trace && !faces.empty()) {
            for (size_t i = 0; i < faces.size(); ++i) {
                MA_LOGI(TAG,
                        "blur-trace face[%zu] inf_norm(tl)=(%.3f,%.3f,%.3f,%.3f) -> "
                        "stream_norm(c)=(%.3f,%.3f,%.3f,%.3f)",
                        i, faces[i].x, faces[i].y, faces[i].w, faces[i].h,
                        stream_boxes[i].cx, stream_boxes[i].cy,
                        stream_boxes[i].w, stream_boxes[i].h);
            }
        }
        g_face_blur->onDetection(stream_boxes, &img);
    }

    // Step 3: Attribute analysis for each face
    const auto analyze_start = clock::now();
    std::vector<AnalyzedFace> analyzed_faces = g_attribute_analyzer->analyzeAll(&img, faces);
    if (g_face_tracker) {
        g_attribute_analyzer->sweepEvidence(g_face_tracker->removedIds());
    }
    const long long analyze_time = ms_since(analyze_start);
    const long long total_time = ms_since(start_time);

    // Step 4: debug overlay (published by the App's debug sink when a /results
    // client listens). FaceInfo x/y are top-left normalised -> centre pixels;
    // box label "face", labels[] carries the per-face attributes.
    const float W = static_cast<float>(g_config.inference_width);
    const float H = static_cast<float>(g_config.inference_height);
    g_out.boxes.resize(analyzed_faces.size());
    g_out.labels.resize(analyzed_faces.size());
    for (size_t i = 0; i < analyzed_faces.size(); ++i) {
        const auto& f = analyzed_faces[i].face;
        af::Box& b = g_out.boxes[i];
        b.x = (f.x + f.w * 0.5f) * W;
        b.y = (f.y + f.h * 0.5f) * H;
        b.w = f.w * W;
        b.h = f.h * H;
        b.score = f.score;
        b.label = "face";
        b.detail = attribute_label(analyzed_faces[i], g_out.labels[i]);
    }
    r.seq = g_frame_id;
    r.inference_ms = static_cast<float>(total_time);
    r.set_boxes(g_out.boxes.data(), g_out.boxes.size());

    // Step 5: MQTT results (published by the App's MQTT sink).
    ha_mqtt::MqttPublisher* mqtt = app.mqtt();
    if (mqtt != nullptr) {
        g_out.payload = buildResultJson(timestamp_ms, g_frame_id, analyzed_faces, static_cast<float>(total_time));
        r.json = g_out.payload;

        // Additionally publish the same detections in ONVIF's analytics
        // representation, on its own topic and its own rate limit. The
        // recamera/<app>/results contract above is consumed by SenseCraft and
        // does not change. Only boxes and a class for now.
        if (g_onvif_meta.take(timestamp_ms)) {
            std::vector<onvif_box_t> ob;
            ob.reserve(g_out.boxes.size());
            for (const auto& b : g_out.boxes) {
                ob.push_back({b.x, b.y, b.w, b.h, b.score, "HumanFace"});
            }
            mqtt->publishText(
                g_onvif_meta.topic(),
                onvif_meta_to_json(onvif_meta_from_boxes(
                    timestamp_ms, "FaceAnalysis",
                    g_config.inference_width, g_config.inference_height, ob)));
        }
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

// App::on_snapshot(): /snapshot.jpg is encoded from this inference frame,
// which the RGN mask (VPSS->VENC path) never touches. Without this the device
// would serve a masked stream and an unmasked still of the same scene -- from
// the URL ONVIF advertises as GetSnapshotUri. Detections are already
// normalised against this buffer, so no letterbox conversion.
static void mask_snapshot(const af::VideoFrame& frame) {
    if (!g_config.enable_blur || g_face_blur == nullptr || !g_face_blur->enabled() || g_out.faces.empty()) {
        return;
    }
    std::vector<geometry::InferBox> snap_boxes;
    snap_boxes.reserve(g_out.faces.size());
    for (const auto& f : g_out.faces) {
        snap_boxes.push_back(geometry::InferBox::fromCorner(f.x, f.y, f.w, f.h, f.score));
    }
    privacy_blur::pixelateRgb888(frame.planes[0].vir, frame.width, frame.height, snap_boxes,
                                 g_face_blur->blockPx());
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) {
        return 1;
    }

    MA_LOGI(TAG, "Starting Face Analysis Application");
    MA_LOGI(TAG, "Face model: %s", g_config.face_model.c_str());
    MA_LOGI(TAG, "GenderAge model: %s", g_config.genderage_model.c_str());
    MA_LOGI(TAG, "Emotion model: %s", g_config.emotion_model.c_str());

    auto release_models = [] {
        delete g_attribute_analyzer;
        g_attribute_analyzer = nullptr;
        delete g_face_tracker;
        g_face_tracker = nullptr;
        delete g_face_detector;
        g_face_detector = nullptr;
    };
    if (!init_models()) {
        MA_LOGE(TAG, "Model initialization failed");
        release_models();
        return 1;
    }

    af::RtspOptions stream;
    stream.width = static_cast<uint16_t>(g_config.stream_width);
    stream.height = static_cast<uint16_t>(g_config.stream_height);
    stream.fps = static_cast<uint8_t>(g_config.stream_fps);

    af::CameraOptions cam;
    cam.width = static_cast<uint16_t>(g_config.inference_width);
    cam.height = static_cast<uint16_t>(g_config.inference_height);
    cam.fps = static_cast<uint8_t>(g_config.inference_fps);

    af::App app("face-analysis");
    app.with_camera(cam).with_stream(stream);
    if (g_config.enable_rtsp) app.with_rtsp(stream);
    if (g_config.enable_debug) app.with_debug_ws(g_config.debug_port);
    if (g_config.enable_mqtt) {
        const ha_mqtt::ClientOptions opts = mqtt_options();
        app.with_mqtt(opts);
        // The ONVIF switch lives in /userdata/local/onvif.conf, written by the console.
        g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), opts.app_id);
        if (g_onvif_meta.enabled()) {
            MA_LOGI(TAG, "ONVIF metadata: %s (every %ums)",
                    g_onvif_meta.topic().c_str(), g_onvif_meta.config().interval_ms);
        }
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
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    app.on_started(init_blur)
       .on_stopping(deinit_blur)
       .on_snapshot(mask_snapshot)
       .on_frame([&app](const af::VideoFrame& f, af::Result& r) { on_frame(app, f, r); });
    const int rc = app.run();

    deinit_blur();  // no-op unless run() failed before on_stopping
    release_models();
    MA_LOGI(TAG, "Face Analysis Application terminated");
    return rc;
}
