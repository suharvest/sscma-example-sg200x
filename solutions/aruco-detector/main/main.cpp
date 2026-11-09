#include <getopt.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// components/app_framework
#include <af/app.h>

#include <time.h>

// AprilTag 3 & ArUco dictionaries
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag16h5.h"
#include "tag36h11.h"
#include "aruco/tagAruco4x4_50.h"
#include "aruco/tagAruco4x4_100.h"
#include "aruco/tagAruco4x4_250.h"
#include "aruco/tagAruco5x5_50.h"
#include "aruco/tagAruco6x6_250.h"

// Platform logging & MQTT
#include <ha_mqtt.h>
#include <sscma.h>

#include "camera_model.h"
#include "capture_time.h"
#include "http_control.h"
#include "inference_control.h"
#include "marker_filter.h"

#define TAG "aruco-detector"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    // Inference camera geometry (CH0 RGB888, converted via integer fixpoint to gray)
    int camera_w = 640;
    int camera_h = 360;
    int camera_fps = 25;

    // Stream geometry (CH2 H.264 live preview)
    bool enable_rtsp = true;
    int stream_w = 1280;
    int stream_h = 720;
    int stream_fps = 30;

    // Dictionary family: aruco4x4_50, aruco4x4_100, aruco4x4_250, aruco5x5_50, aruco6x6_250, tag36h11, tag16h5
    std::string family_name = "aruco4x4_50";

    // Detector parameters
    double quad_decimate = 2.0;  // 2x decimate for 30fps embedded speed on SG2002
    double quad_sigma = 0.0;
    bool refine_edges = true;

    // Metric calibration & 3D pose estimation (meters / pixels)
    double marker_size = 0.05;  // 50mm default marker side
    double fx = 600.0;          // focal length x
    double fy = 600.0;          // focal length y
    double cx = 320.0;          // principal point x
    double cy = 180.0;          // principal point y
    // OpenCV Brown-Conrady distortion (k1, k2, p1, p2, k3); all zero = no undistortion
    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0;

    // MQTT
    bool enable_mqtt = true;
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/aruco-detector/results";

    // Debug WebSocket (Web Console)
    bool enable_debug = true;
    int debug_port = 8001;
    int rtsp_port = 8554;
    bool verbose = false;

    // On-demand inference control (HTTP 8090)
    bool inference_enabled = false;
    int control_port = 8090;

    // False-positive filter (applied before pose estimation)
    aruco::MarkerFilter filter;  // defaults: min_margin 30, min_side_px 12, all ids
} g_config;

static aruco::InferenceControl* g_inference = nullptr;

// ---------------------------------------------------------------------------
// Helpers: Euler angles from 3x3 rotation matrix
// ---------------------------------------------------------------------------

static void rot_to_euler(const matd_t* R, double& roll, double& pitch, double& yaw) {
    double r20 = MATD_EL(R, 2, 0);
    double r21 = MATD_EL(R, 2, 1);
    double r22 = MATD_EL(R, 2, 2);
    double r00 = MATD_EL(R, 0, 0);
    double r10 = MATD_EL(R, 1, 0);

    pitch = std::asin(-std::clamp(r20, -1.0, 1.0));
    if (std::abs(r20) < 0.999999) {
        roll = std::atan2(r21, r22);
        yaw  = std::atan2(r10, r00);
    } else {
        roll = std::atan2(-MATD_EL(R, 1, 2), MATD_EL(R, 1, 1));
        yaw  = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Family lifecycle management
// ---------------------------------------------------------------------------

struct FamilyDeleter {
    std::string name;
    void operator()(apriltag_family_t* tf) const {
        if (!tf) return;
        if (name == "aruco4x4_50") tagAruco4x4_50_destroy(tf);
        else if (name == "aruco4x4_100") tagAruco4x4_100_destroy(tf);
        else if (name == "aruco4x4_250") tagAruco4x4_250_destroy(tf);
        else if (name == "aruco5x5_50") tagAruco5x5_50_destroy(tf);
        else if (name == "aruco6x6_250") tagAruco6x6_250_destroy(tf);
        else if (name == "tag36h11") tag36h11_destroy(tf);
        else if (name == "tag16h5") tag16h5_destroy(tf);
    }
};

static apriltag_family_t* create_family(const std::string& name) {
    if (name == "aruco4x4_50") return tagAruco4x4_50_create();
    if (name == "aruco4x4_100") return tagAruco4x4_100_create();
    if (name == "aruco4x4_250") return tagAruco4x4_250_create();
    if (name == "aruco5x5_50") return tagAruco5x5_50_create();
    if (name == "aruco6x6_250") return tagAruco6x6_250_create();
    if (name == "tag36h11") return tag36h11_create();
    if (name == "tag16h5") return tag16h5_create();
    return nullptr;
}

// ---------------------------------------------------------------------------
// CLI Options
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -f, --family <name>       Marker family (aruco4x4_50|100|250, aruco5x5_50, aruco6x6_250, tag36h11, tag16h5) [default: aruco4x4_50]\n");
    printf("  -s, --marker-size <m>     Physical marker side in meters [default: 0.05]\n");
    printf("      --fx <pixels>         Camera focal length X [default: 600.0]\n");
    printf("      --fy <pixels>         Camera focal length Y [default: 600.0]\n");
    printf("      --cx <pixels>         Principal point X [default: 320.0]\n");
    printf("      --cy <pixels>         Principal point Y [default: 180.0]\n");
    printf("      --k1 --k2 --p1 --p2 --k3 <v>  Lens distortion (OpenCV order, at inference resolution) [default: 0]\n");
    printf("      --decimate <factor>   Quad decimation factor (1.0, 1.5, 2.0) [default: 2.0]\n");
    printf("      --camera-width <w>    Inference width [default: 640]\n");
    printf("      --camera-height <h>   Inference height [default: 360]\n");
    printf("      --camera-fps <fps>    Inference fps [default: 25]\n");
    printf("      --stream-width <w>    RTSP/WebSocket stream width [default: 1280]\n");
    printf("      --stream-height <h>   RTSP/WebSocket stream height [default: 720]\n");
    printf("      --stream-fps <fps>    RTSP/WebSocket stream fps [default: 30]\n");
    printf("  -m, --mqtt-host <host>    MQTT broker host [default: localhost]\n");
    printf("  -p, --mqtt-port <port>    MQTT broker port [default: 1883]\n");
    printf("      --mqtt-topic <topic>  MQTT topic [default: recamera/aruco-detector/results]\n");
    printf("      --no-mqtt             Disable MQTT publishing\n");
    printf("      --no-rtsp             Disable RTSP server\n");
    printf("      --debug-port <port>   WebSocket debug port [default: 8001]\n");
    printf("      --control-port <port> HTTP control port [default: 8090]\n");
    printf("      --min-margin <v>      Drop detections with decision_margin below v (0 = off) [default: 30]\n");
    printf("      --min-side <px>       Drop detections whose shortest quad edge is below px (0 = off) [default: 12]\n");
    printf("      --ids <list>          Comma-separated marker id allowlist, e.g. 3,7,12 [default: empty = all]\n");
    printf("      --inference-on        Start with inference enabled [default: standby/off]\n");
    printf("  -v, --verbose             Verbose logging\n");
    printf("  -h, --help                Show this help\n");
}

static void parse_args(int argc, char** argv) {
    static struct option long_opts[] = {
        {"family", required_argument, nullptr, 'f'},
        {"marker-size", required_argument, nullptr, 's'},
        {"fx", required_argument, nullptr, 1001},
        {"fy", required_argument, nullptr, 1002},
        {"cx", required_argument, nullptr, 1003},
        {"cy", required_argument, nullptr, 1004},
        {"decimate", required_argument, nullptr, 1005},
        {"camera-width", required_argument, nullptr, 1006},
        {"camera-height", required_argument, nullptr, 1007},
        {"camera-fps", required_argument, nullptr, 1008},
        {"mqtt-host", required_argument, nullptr, 'm'},
        {"mqtt-port", required_argument, nullptr, 'p'},
        {"mqtt-topic", required_argument, nullptr, 1009},
        {"no-mqtt", no_argument, nullptr, 1010},
        {"no-rtsp", no_argument, nullptr, 1011},
        {"debug-port", required_argument, nullptr, 1012},
        {"control-port", required_argument, nullptr, 1013},
        {"inference-on", no_argument, nullptr, 1014},
        {"k1", required_argument, nullptr, 1015},
        {"k2", required_argument, nullptr, 1016},
        {"p1", required_argument, nullptr, 1017},
        {"p2", required_argument, nullptr, 1018},
        {"k3", required_argument, nullptr, 1019},
        {"min-margin", required_argument, nullptr, 1020},
        {"min-side", required_argument, nullptr, 1021},
        {"ids", required_argument, nullptr, 1022},
        {"stream-width", required_argument, nullptr, 1023},
        {"stream-height", required_argument, nullptr, 1024},
        {"stream-fps", required_argument, nullptr, 1025},
        {"verbose", no_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:s:m:p:vh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'f': g_config.family_name = optarg; break;
            case 's': g_config.marker_size = std::atof(optarg); break;
            case 1001: g_config.fx = std::atof(optarg); break;
            case 1002: g_config.fy = std::atof(optarg); break;
            case 1003: g_config.cx = std::atof(optarg); break;
            case 1004: g_config.cy = std::atof(optarg); break;
            case 1005: g_config.quad_decimate = std::atof(optarg); break;
            case 1006: g_config.camera_w = std::atoi(optarg); break;
            case 1007: g_config.camera_h = std::atoi(optarg); break;
            case 1008: g_config.camera_fps = std::atoi(optarg); break;
            case 'm': g_config.mqtt_host = optarg; break;
            case 'p': g_config.mqtt_port = std::atoi(optarg); break;
            case 1009: g_config.mqtt_topic = optarg; break;
            case 1010: g_config.enable_mqtt = false; break;
            case 1011: g_config.enable_rtsp = false; break;
            case 1012: g_config.debug_port = std::atoi(optarg); break;
            case 1013: g_config.control_port = std::atoi(optarg); break;
            case 1014: g_config.inference_enabled = true; break;
            case 1015: g_config.k1 = std::atof(optarg); break;
            case 1016: g_config.k2 = std::atof(optarg); break;
            case 1017: g_config.p1 = std::atof(optarg); break;
            case 1018: g_config.p2 = std::atof(optarg); break;
            case 1019: g_config.k3 = std::atof(optarg); break;
            case 1020:
            case 1021: {
                char* end = nullptr;
                const double v = std::strtod(optarg, &end);
                if (end == optarg || *end != '\0' || !std::isfinite(v) || v < 0.0) {
                    fprintf(stderr, "Invalid value for --%s: '%s' (expected number >= 0)\n",
                            opt == 1020 ? "min-margin" : "min-side", optarg);
                    exit(1);
                }
                (opt == 1020 ? g_config.filter.min_margin : g_config.filter.min_side_px) = static_cast<float>(v);
                break;
            }
            case 1022: {
                std::string err;
                if (!aruco::parse_id_list(optarg, g_config.filter.ids, &err)) {
                    fprintf(stderr, "Invalid --ids '%s': %s\n", optarg, err.c_str());
                    exit(1);
                }
                break;
            }
            case 1023: g_config.stream_w = std::atoi(optarg); break;
            case 1024: g_config.stream_h = std::atoi(optarg); break;
            case 1025: g_config.stream_fps = std::atoi(optarg); break;
            case 'v': g_config.verbose = true; break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Main Entry: app_framework Fluent API
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    parse_args(argc, argv);

    MA_LOGI(TAG, "Initializing ArUco & AprilTag detector (Family: %s, MarkerSize: %.3fm)",
            g_config.family_name.c_str(), g_config.marker_size);

    aruco::CameraModel cam_model;
    cam_model.fx = g_config.fx;
    cam_model.fy = g_config.fy;
    cam_model.cx = g_config.cx;
    cam_model.cy = g_config.cy;
    cam_model.k1 = g_config.k1;
    cam_model.k2 = g_config.k2;
    cam_model.p1 = g_config.p1;
    cam_model.p2 = g_config.p2;
    cam_model.k3 = g_config.k3;
    const bool undistort = cam_model.has_distortion();
    MA_LOGI(TAG, "Intrinsics @%dx%d: fx=%.3f fy=%.3f cx=%.3f cy=%.3f | dist k1=%.6f k2=%.6f p1=%.6f p2=%.6f k3=%.6f (undistort %s)",
            g_config.camera_w, g_config.camera_h, cam_model.fx, cam_model.fy, cam_model.cx, cam_model.cy,
            cam_model.k1, cam_model.k2, cam_model.p1, cam_model.p2, cam_model.k3, undistort ? "on" : "off");

    MA_LOGI(TAG, "Marker filter: min_margin=%.1f min_side=%.1fpx ids=%s",
            g_config.filter.min_margin, g_config.filter.min_side_px,
            aruco::format_id_list(g_config.filter.ids).c_str());

    // Initialize AprilTag family and detector
    apriltag_family_t* tf = create_family(g_config.family_name);
    if (!tf) {
        MA_LOGE(TAG, "Unsupported marker family: %s", g_config.family_name.c_str());
        return 1;
    }

    apriltag_detector_t* td = apriltag_detector_create();
    if (!td) {
        MA_LOGE(TAG, "Failed to create apriltag detector");
        return 1;
    }
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = static_cast<float>(g_config.quad_decimate);
    td->quad_sigma = static_cast<float>(g_config.quad_sigma);
    td->refine_edges = g_config.refine_edges ? 1 : 0;
    td->nthreads = 1;

    // Fluent App assembly
    af::App app("aruco-detector");

    // Camera channel: CH0 (640x360 @ 25fps, physical=false for lifecycle safety)
    af::CameraOptions cam_opt;
    cam_opt.width = static_cast<uint16_t>(g_config.camera_w);
    cam_opt.height = static_cast<uint16_t>(g_config.camera_h);
    cam_opt.fps = static_cast<uint8_t>(g_config.camera_fps);
    cam_opt.physical = false;
    cam_opt.idle_fps = aruco::InferenceControl::kIdleFps;

    // Stream channel: CH2 (1280x720 @ 30fps H.264)
    af::StreamOptions stream_opt;
    stream_opt.width = static_cast<uint16_t>(g_config.stream_w);
    stream_opt.height = static_cast<uint16_t>(g_config.stream_h);
    stream_opt.fps = static_cast<uint8_t>(g_config.stream_fps);
    stream_opt.low_latency = true;

    app.with_camera(cam_opt)
       .with_stream(stream_opt)
       .with_debug_ws(g_config.debug_port);

    if (g_config.enable_rtsp) {
        af::RtspOptions rtsp_opt;
        rtsp_opt.port = g_config.rtsp_port;
        app.with_rtsp(rtsp_opt);
    }

    if (g_config.enable_mqtt) {
        ha_mqtt::ClientOptions mqtt_opt;
        mqtt_opt.app_id = "aruco-detector";
        mqtt_opt.client_id = "aruco-detector-" + std::to_string(getpid());
        mqtt_opt.legacy_host = g_config.mqtt_host;
        mqtt_opt.legacy_port = static_cast<uint16_t>(g_config.mqtt_port);
        mqtt_opt.results_topic = g_config.mqtt_topic;
        app.with_mqtt(mqtt_opt);
    }

    // On-demand inference and HTTP control
    aruco::InferenceControl inference(
        g_config.camera_fps, g_config.inference_enabled,
        [&](bool, int fps) { app.set_camera_fps(static_cast<uint8_t>(fps)); });
    g_inference = &inference;

    aruco::HttpControlServer http(inference);
    if (!http.start(g_config.control_port)) {
        MA_LOGW(TAG, "Failed to start HTTP control on port %d", g_config.control_port);
    }

    // Preallocated buffers for zero-heap-allocation per-frame processing
    std::vector<uint8_t> gray_buf(g_config.camera_w * g_config.camera_h);
    std::vector<af::Box> boxes;
    std::vector<std::string> label_storage;
    std::vector<std::string> detail_storage;
    std::string json_payload;
    std::string markers_payload;
    json_payload.reserve(2048);
    markers_payload.reserve(2048);

    uint64_t frame_seq = 0;
    auto last_stat_time = std::chrono::steady_clock::now();
    uint32_t processed_frames = 0;

    app.on_frame([&](const af::VideoFrame& frame, af::Result& result) {
        // Capture time of this frame (not processing time): wall clock minus the
        // frame's age, where frame.ts_ms is CLOCK_MONOTONIC ms stamped at grab.
        timespec mono_ts{};
        clock_gettime(CLOCK_MONOTONIC, &mono_ts);
        const uint32_t now_mono_ms = static_cast<uint32_t>(
            (static_cast<uint64_t>(mono_ts.tv_sec) * 1000u + static_cast<uint64_t>(mono_ts.tv_nsec) / 1000000u));
        const uint64_t now_unix_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        const uint64_t capture_unix_ms = aruco::capture_unix_ms(now_unix_ms, now_mono_ms, frame.ts_ms);
        frame_seq++;

        if (!g_inference->enabled()) {
            result.publish = false;
            return;
        }

        processed_frames++;

        if (!frame.valid() || !frame.cpu_accessible()) {
            return;
        }

        const uint16_t width = frame.width;
        const uint16_t height = frame.height;
        const size_t total_px = width * height;
        if (gray_buf.size() < total_px) {
            gray_buf.resize(total_px);
        }

        auto start_proc = std::chrono::steady_clock::now();

        // Convert RGB888 to 8-bit Grayscale using fixed-point integer arithmetic (0.4ms)
        const uint8_t* rgb = frame.planes[0].vir;
        uint8_t* gray = gray_buf.data();
        for (size_t i = 0; i < total_px; ++i) {
            uint32_t r = rgb[3 * i];
            uint32_t g = rgb[3 * i + 1];
            uint32_t b = rgb[3 * i + 2];
            gray[i] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
        }

        image_u8_t im = {
            .width = static_cast<int32_t>(width),
            .height = static_cast<int32_t>(height),
            .stride = static_cast<int32_t>(width),
            .buf = gray
        };

        // Detect fiducial markers
        zarray_t* detections = apriltag_detector_detect(td, &im);
        const int n_det = zarray_size(detections);
        int count = 0;     // accepted
        int rejected = 0;  // dropped by filter

        boxes.clear();
        label_storage.clear();
        detail_storage.clear();
        boxes.reserve(n_det);
        label_storage.reserve(n_det);
        detail_storage.reserve(n_det);

        markers_payload.clear();
        markers_payload.push_back('[');

        for (int i = 0; i < n_det; ++i) {
            apriltag_detection_t* det = nullptr;
            zarray_get(detections, i, &det);
            if (!det) continue;

            // False-positive filter, before pose estimation
            const aruco::FilterVerdict verdict =
                aruco::classify_marker(g_config.filter, det->id, det->decision_margin, det->p);
            if (verdict != aruco::FilterVerdict::Accept) {
                ++rejected;
                if (g_config.verbose) {
                    MA_LOGI(TAG, "Rejected id=%d margin=%.1f side=%.1fpx (%s)", det->id, det->decision_margin,
                            aruco::min_quad_side(det->p),
                            verdict == aruco::FilterVerdict::LowMargin   ? "margin"
                            : verdict == aruco::FilterVerdict::SmallSide ? "side"
                                                                         : "id");
                }
                continue;
            }

            // Bounding box in inference pixel space
            double min_x = det->p[0][0], max_x = det->p[0][0];
            double min_y = det->p[0][1], max_y = det->p[0][1];
            for (int p = 1; p < 4; ++p) {
                min_x = std::min(min_x, det->p[p][0]);
                max_x = std::max(max_x, det->p[p][0]);
                min_y = std::min(min_y, det->p[p][1]);
                max_y = std::max(max_y, det->p[p][1]);
            }

            // Estimate 6-DoF Metric Pose
            apriltag_detection_info_t info;
            // Pose is solved on ideal (undistorted) pixel coordinates; JSON keeps raw corners.
            apriltag_detection_t det_u;
            matd_t* H_u = nullptr;
            info.det = det;
            if (undistort) {
                det_u = *det;
                for (int p = 0; p < 4; ++p) {
                    cam_model.undistort(det->p[p][0], det->p[p][1], det_u.p[p][0], det_u.p[p][1]);
                }
                cam_model.undistort(det->c[0], det->c[1], det_u.c[0], det_u.c[1]);
                double h[9];
                if (aruco::homography_from_corners(det_u.p, h)) {
                    H_u = matd_create_data(3, 3, h);
                    det_u.H = H_u;
                    info.det = &det_u;
                }
            }
            info.tagsize = g_config.marker_size;
            info.fx = g_config.fx;
            info.fy = g_config.fy;
            info.cx = g_config.cx;
            info.cy = g_config.cy;

            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);
            double roll = 0.0, pitch = 0.0, yaw = 0.0;
            rot_to_euler(pose.R, roll, pitch, yaw);

            // Construct af::Box for Debug WebSocket display overlay
            label_storage.push_back("ID:" + std::to_string(det->id));
            char det_buf[96];
            snprintf(det_buf, sizeof(det_buf), "x:%.2fm y:%.2fm z:%.2fm",
                     MATD_EL(pose.t, 0, 0), MATD_EL(pose.t, 1, 0), MATD_EL(pose.t, 2, 0));
            detail_storage.push_back(det_buf);

            af::Box b;
            b.x = static_cast<float>(det->c[0]);
            b.y = static_cast<float>(det->c[1]);
            b.w = static_cast<float>(max_x - min_x);
            b.h = static_cast<float>(max_y - min_y);
            b.score = 1.0f;
            b.cls = det->id;
            b.label = label_storage.back().c_str();
            b.detail = detail_storage.back().c_str();
            boxes.push_back(b);

            // Append marker object to JSON payload
            if (count > 0) markers_payload.push_back(',');
            ++count;
            char item[420];
            snprintf(item, sizeof(item),
                     "{\"id\":%d,\"margin\":%.1f,\"center\":[%.1f,%.1f],\"corners\":[[%.1f,%.1f],[%.1f,%.1f],[%.1f,%.1f],[%.1f,%.1f]],"
                     "\"pose\":{\"t\":[%.4f,%.4f,%.4f],\"r_deg\":[%.2f,%.2f,%.2f],\"error\":%.4f}}",
                     det->id, det->decision_margin,
                     det->c[0], det->c[1],
                     det->p[0][0], det->p[0][1],
                     det->p[1][0], det->p[1][1],
                     det->p[2][0], det->p[2][1],
                     det->p[3][0], det->p[3][1],
                     MATD_EL(pose.t, 0, 0), MATD_EL(pose.t, 1, 0), MATD_EL(pose.t, 2, 0),
                     roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI,
                     err);
            markers_payload.append(item);

            matd_destroy(pose.R);
            matd_destroy(pose.t);
            if (H_u) matd_destroy(H_u);
        }

        markers_payload.push_back(']');
        apriltag_detections_destroy(detections);

        auto end_proc = std::chrono::steady_clock::now();
        float proc_ms = std::chrono::duration<float, std::milli>(end_proc - start_proc).count();

        json_payload.clear();
        json_payload.append("{\"frame_id\":");
        json_payload.append(std::to_string(frame_seq));
        json_payload.append(",\"timestamp\":");
        json_payload.append(std::to_string(capture_unix_ms));
        json_payload.append(",\"family\":\"");
        json_payload.append(g_config.family_name);
        json_payload.append("\",\"count\":");
        json_payload.append(std::to_string(count));
        json_payload.append(undistort ? ",\"undistorted\":true" : ",\"undistorted\":false");
        json_payload.append(",\"rejected\":");
        json_payload.append(std::to_string(rejected));
        json_payload.append(",\"markers\":");
        json_payload.append(markers_payload);
        json_payload.append("}");

        // Publish to InferenceControl
        aruco::InferenceResult ir;
        ir.frame_id = static_cast<uint32_t>(frame_seq);
        ir.ts_ms = capture_unix_ms;
        ir.inference_ms = proc_ms;
        ir.count = count;
        ir.rejected = rejected;
        ir.markers_json = markers_payload;
        ir.undistorted = undistort;
        g_inference->publish(ir);

        // Feed results into af::Result sink fan-out
        result.width = width;
        result.height = height;
        result.inference_ms = proc_ms;
        result.set_boxes(boxes.data(), boxes.size());
        result.json = json_payload;

        // Periodic log output
        if (g_config.verbose && processed_frames % 25 == 0) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - last_stat_time).count();
            float fps = static_cast<float>(processed_frames) / elapsed;
            MA_LOGI(TAG, "Pipeline: %.1f FPS (Proc: %.1fms), Detected: %d markers",
                    fps, proc_ms, count);
            processed_frames = 0;
            last_stat_time = now;
        }
    });

    int exit_code = app.run();

    http.stop();
    g_inference = nullptr;

    // Clean up
    apriltag_detector_destroy(td);
    FamilyDeleter{g_config.family_name}(tf);

    return exit_code;
}
