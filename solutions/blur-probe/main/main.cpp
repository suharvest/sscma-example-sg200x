/*
 * blur-probe -- exercise the privacy mask on the real video path with boxes
 * whose position is known in advance.
 *
 * Why this exists
 * ---------------
 * Verifying the mask against a live detector conflates two questions: whether
 * the mask lands where it was asked to, and whether the detector asked for the
 * right place. When a face came out unmasked there was no way to tell which had
 * failed, and every attempt needed a person to stand in front of the camera --
 * so each iteration cost a round trip and produced a screenshot rather than a
 * number.
 *
 * This replaces only the detector. Camera, VPSS, RGN, VENC and RTSP are the
 * same code the real solutions run: nothing here is a simulation of the video
 * path, and a mask that works here is a mask that works in an application.
 * Because the boxes are supplied on the command line, "did it land correctly"
 * becomes a measurement -- the box is at a known place, so any offset can be
 * read straight off the frame in pixels.
 *
 * Coordinates
 * -----------
 * Boxes are given in STREAM-normalised centre form, the coordinate system
 * PrivacyBlur itself consumes. The letterbox conversion an application needs
 * (its detector works on a differently-shaped inference frame) is deliberately
 * NOT applied here. That keeps two independent suspects apart: if a mask is
 * misplaced under this tool the fault is in the mask path, and if it is only
 * misplaced in an application the fault is in that application's conversion.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <signal.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <sscma.h>
#include <video.h>

#include <debug_stream.h>

#include "privacy_blur.h"
#include "rtsp_server.h"

using namespace ma;
using privacy_blur::BlurBox;
using privacy_blur::PrivacyBlur;
using privacy_blur::PrivacyBlurConfig;

#define TAG "blur-probe"

namespace {

struct Config {
    int inference_width  = 640;
    int inference_height = 480;
    int inference_fps    = 10;
    int stream_width     = 1280;
    int stream_height    = 720;
    int stream_fps       = 30;

    /* "one", "two", "grid", "sweep" -- see makeBoxes(). */
    std::string pattern = "one";
    /* Overrides the pattern when non-empty. Repeatable, because measuring the
     * mask needs more than one box: a lone rectangle gives no gap, and the gap
     * between two is the one edge in the picture whose position is unambiguous
     * (it renders as pure black, which the scene never produces). */
    std::vector<BlurBox> explicit_boxes;

    int  debug_port  = 8001;
    bool mask_snapshot = true;

    int  max_regions = 8;
    int  alpha       = -1; /* -1: leave blur.conf / driver default alone */
    bool log_boxes   = true;
};

Config     g_config;
Camera*    g_camera = nullptr;
PrivacyBlur* g_blur = nullptr;
std::atomic<bool> g_running{true};

void onSignal(int) { g_running.store(false); }

/*
 * The synthetic detections.
 *
 * "two" is the interesting one: two boxes with a deliberate gap between them is
 * the arrangement that exposed the band of black cells, because the cells in
 * the gap belong to neither rectangle and so are nobody's to fill. A single box
 * cannot show that failure at all.
 */
std::vector<BlurBox> makeBoxes(unsigned frame) {
    std::vector<BlurBox> v;

    if (!g_config.explicit_boxes.empty()) {
        return g_config.explicit_boxes;
    }

    const std::string& p = g_config.pattern;
    if (p == "one") {
        v.push_back({0.5f, 0.5f, 0.25f, 0.25f, 1.0f});
    } else if (p == "two") {
        v.push_back({0.30f, 0.5f, 0.18f, 0.25f, 1.0f});
        v.push_back({0.70f, 0.5f, 0.18f, 0.25f, 1.0f});
    } else if (p == "grid") {
        /* Four corners plus the centre: catches an offset that only shows up
         * away from the middle, where a scale error and a correct mapping agree. */
        v.push_back({0.20f, 0.25f, 0.14f, 0.18f, 1.0f});
        v.push_back({0.80f, 0.25f, 0.14f, 0.18f, 1.0f});
        v.push_back({0.20f, 0.75f, 0.14f, 0.18f, 1.0f});
        v.push_back({0.80f, 0.75f, 0.14f, 0.18f, 1.0f});
        v.push_back({0.50f, 0.50f, 0.14f, 0.18f, 1.0f});
    } else if (p == "sweep") {
        /* Moves so the tracker and the prediction thread are exercised: a mask
         * that is correct only while nothing moves is not much of a mask. */
        const float t  = (frame % 120) / 120.0f;
        const float cx = 0.15f + 0.70f * t;
        v.push_back({cx, 0.5f, 0.18f, 0.25f, 1.0f});
    }
    return v;
}

bool initCamera() {
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
        /* CPU access required: the pixelating backend averages the pixels it
         * hides, so it has to be able to read them. */
        value.i32 = 0;
        g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

        MA_LOGI(TAG, "camera %dx%d @ %dfps", g_config.inference_width,
                g_config.inference_height, g_config.inference_fps);
        return true;
    }
    MA_LOGE(TAG, "no camera found");
    return false;
}

bool initVideoStreaming() {
    if (initVideo() != 0) {
        MA_LOGE(TAG, "initVideo failed");
        return false;
    }
    video_ch_param_t param;
    memset(&param, 0, sizeof(param));
    param.format = VIDEO_FORMAT_H264;
    param.width  = g_config.stream_width;
    param.height = g_config.stream_height;
    param.fps    = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &param);
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));
    MA_LOGI(TAG, "RTSP %dx%d @ %dfps", g_config.stream_width, g_config.stream_height,
            g_config.stream_fps);
    return true;
}

bool initBlur() {
    PrivacyBlurConfig cfg;
    loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, nullptr);

    /* Unlike an application, the probe masks whether or not the device-wide
     * switch is on. It exists to look at the mask; starting it and finding
     * nothing because a config file said so would waste the run. */
    cfg.enabled     = true;
    cfg.max_regions = g_config.max_regions;
    if (g_config.alpha >= 0) cfg.alpha = g_config.alpha;

    g_blur = new PrivacyBlur();
    if (!g_blur->init(cfg, g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "privacy blur init failed");
        delete g_blur;
        g_blur = nullptr;
        return false;
    }
    return true;
}

void usage(const char* argv0) {
    printf(
        "Usage: %s [options]\n"
        "  --pattern <one|two|grid|sweep>  synthetic box layout (default: one)\n"
        "  --box cx,cy,w,h                 explicit box, stream-normalised; repeatable\n"
        "  --max-regions <n>               regions offered to the mask (default 8)\n"
        "  --alpha <0-255>                 override mask opacity\n"
        "  --quiet                         do not log the boxes each second\n"
        "  --raw-snapshot                  serve /snapshot.jpg unmasked (shows the bypass)\n"
        "  -h, --help                      this text\n"
        "\nBoxes are stream-normalised centre form; no letterbox conversion is\n"
        "applied, so a mask that is misplaced here is misplaced in the mask path.\n",
        argv0);
}

bool parseArgs(int argc, char** argv) {
    static struct option opts[] = {
        {"pattern", required_argument, nullptr, 'p'},
        {"box", required_argument, nullptr, 'b'},
        {"max-regions", required_argument, nullptr, 'r'},
        {"alpha", required_argument, nullptr, 'a'},
        {"quiet", no_argument, nullptr, 'q'},
        {"raw-snapshot", no_argument, nullptr, 'R'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "p:b:r:a:qRh", opts, nullptr)) != -1) {
        switch (opt) {
        case 'p':
            g_config.pattern = optarg;
            break;
        case 'b': {
            BlurBox b;
            if (sscanf(optarg, "%f,%f,%f,%f", &b.x, &b.y, &b.w, &b.h) != 4) {
                fprintf(stderr, "--box wants cx,cy,w,h\n");
                return false;
            }
            b.score = 1.0f;
            g_config.explicit_boxes.push_back(b);
            break;
        }
        case 'r':
            g_config.max_regions = atoi(optarg);
            break;
        case 'a':
            g_config.alpha = atoi(optarg);
            break;
        case 'q':
            g_config.log_boxes = false;
            break;
        case 'R':
            /* Serves the snapshot unmasked, to show the gap this closes. */
            g_config.mask_snapshot = false;
            break;
        case 'h':
        default:
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (!parseArgs(argc, argv)) return 1;

    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    if (!initCamera()) return 1;
    if (!initVideoStreaming()) return 1;

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    /* After the video pipeline, like the applications do: RGN attaches to a
     * running VPSS channel and silently does nothing if it is not up yet. */
    if (!initBlur()) return 1;

    if (g_config.explicit_boxes.empty()) {
        if (debug_stream_start_or_disable(g_config.debug_port, VIDEO_CH2) == 0) {
        MA_LOGI(TAG, "debug stream + snapshot on port %d", g_config.debug_port);
    }
    MA_LOGI(TAG, "probe running, pattern=%s", g_config.pattern.c_str());
    } else {
        MA_LOGI(TAG, "probe running, %zu explicit box(es)", g_config.explicit_boxes.size());
    }

    unsigned frame_id = 0;
    while (g_running.load()) {
        ma_img_t frame;
        if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::vector<BlurBox> boxes = makeBoxes(frame_id);
        g_blur->onDetection(boxes, &frame);

        /* Once a second, and only the first box: enough to confirm what was
         * asked for without burying the log the mask errors appear in. */
        if (g_config.log_boxes && !boxes.empty() &&
            (frame_id % (unsigned)(g_config.inference_fps > 0 ? g_config.inference_fps : 10)) == 0) {
            const BlurBox& b = boxes[0];
            const int left = (int)((b.x - b.w / 2.0f) * g_config.stream_width);
            const int top  = (int)((b.y - b.h / 2.0f) * g_config.stream_height);
            const int w    = (int)(b.w * g_config.stream_width);
            const int h    = (int)(b.h * g_config.stream_height);
            MA_LOGI(TAG, "frame %u: %zu box(es); box[0] norm(c)=(%.3f,%.3f,%.3f,%.3f) "
                         "expect px rect=(%d,%d %dx%d)",
                    frame_id, boxes.size(), b.x, b.y, b.w, b.h, left, top, w, h);
        }

        /*
         * The snapshot is encoded from this buffer, which the RGN mask never
         * touches -- it lives further down the pipeline, in VPSS->VENC. Without
         * masking here the device would serve a masked video stream and an
         * unmasked still of the same scene, from the URL ONVIF advertises as
         * GetSnapshotUri.
         *
         * Only when a snapshot is actually due: offer_snapshot is a cheap
         * atomic load when nobody has asked, and pixelating a frame nobody
         * will look at would put real work on every iteration.
         */
        if (debug_stream_snapshot_armed()) {
            if (g_config.mask_snapshot && !boxes.empty()) {
                /* boxes are stream-normalised; this buffer is the inference
                 * frame, a different shape. Convert with the same arithmetic
                 * the overlay uses, in reverse. */
                std::vector<debug_stream_box_t> px;
                px.reserve(boxes.size());
                for (const auto& b : boxes) {
                    px.push_back({b.x * g_config.stream_width, b.y * g_config.stream_height,
                                  b.w * g_config.stream_width, b.h * g_config.stream_height,
                                  b.score, std::string()});
                }
                debug_stream_display_to_letterbox(px, frame.width, frame.height,
                                                  g_config.stream_width, g_config.stream_height);
                std::vector<BlurBox> local;
                local.reserve(px.size());
                for (const auto& b : px) {
                    local.push_back({b.x / frame.width, b.y / frame.height,
                                     b.w / frame.width, b.h / frame.height, b.score});
                }
                privacy_blur::pixelateRgb888(frame.data, frame.width, frame.height, local, 16);
            }
            debug_stream_offer_snapshot(frame.data, frame.width, frame.height);
        }

        g_camera->returnFrame(frame);
        frame_id++;
    }

    MA_LOGI(TAG, "stopping");
    if (g_blur) {
        g_blur->deinit();
        delete g_blur;
        g_blur = nullptr;
    }
    if (g_camera) g_camera->stopStream();
    deinitVideo();
    return 0;
}
