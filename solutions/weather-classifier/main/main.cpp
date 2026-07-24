#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <memory>
#include <numeric>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sscma.h>
#include <video.h>

#include <debug_stream.h>
#include <ha_mqtt.h>

#include "engine_utils.h"
#include "mqtt_payload.h"
#include "rtsp_demo.h"

using Clock = std::chrono::steady_clock;
using namespace ma;

#define TAG "weather-classifier"

static std::atomic<bool> g_running{true};

static double ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty())
            labels.push_back(line);
    }
    return labels;
}

static const char* dtype_name(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32:
            return "F32";
        case MA_TENSOR_TYPE_F16:
            return "F16";
        case MA_TENSOR_TYPE_BF16:
            return "BF16";
        case MA_TENSOR_TYPE_S8:
            return "S8";
        case MA_TENSOR_TYPE_U8:
            return "U8";
        default:
            return "UNKNOWN";
    }
}

struct Classifier {
    std::unique_ptr<ma::engine::EngineCVI> engine;
    ma_tensor_t input_desc{};
    ma_tensor_type_t input_type{};
    ma_quant_param_t input_qp{};
    weather::InputBuf input_buf;
    size_t input_numel = 0;
    int input_w        = 0;
    int input_h        = 0;
    bool nchw          = true;

    bool init(const std::string& model_path) {
        engine = std::make_unique<ma::engine::EngineCVI>();
        if (engine->init() != MA_OK) {
            std::fprintf(stderr, "[ERROR] EngineCVI::init failed\n");
            return false;
        }
        if (engine->load(model_path) != MA_OK) {
            std::fprintf(stderr, "[ERROR] load model failed: %s\n", model_path.c_str());
            return false;
        }

        if (engine->getInputSize() != 1) {
            std::fprintf(stderr, "[ERROR] expected one input, got %d\n", engine->getInputSize());
            return false;
        }

        input_desc             = engine->getInput(0);
        input_type             = input_desc.type;
        input_qp               = input_desc.quant_param;
        const ma_shape_t shape = engine->getInputShape(0);
        if (shape.size != 4) {
            std::fprintf(stderr, "[ERROR] expected 4-D image input\n");
            return false;
        }

        if (shape.dims[1] == 3) {
            nchw    = true;
            input_h = shape.dims[2];
            input_w = shape.dims[3];
        } else if (shape.dims[3] == 3) {
            nchw    = false;
            input_h = shape.dims[1];
            input_w = shape.dims[2];
        } else {
            std::fprintf(stderr, "[ERROR] cannot determine NCHW/NHWC layout\n");
            return false;
        }

        input_numel = weather::shape_numel(shape);
        input_buf.resize_for(input_type, input_numel);

        std::printf("[MODEL] input name=%s type=%s layout=%s shape=(", input_desc.name ? input_desc.name : "null", dtype_name(input_type), nchw ? "NCHW" : "NHWC");
        for (int i = 0; i < shape.size; ++i) {
            std::printf("%d%s", shape.dims[i], i + 1 < shape.size ? "," : "");
        }
        std::printf(") quant(scale=%g,zp=%d)\n", input_qp.scale, input_qp.zero_point);

        for (int i = 0; i < engine->getOutputSize(); ++i) {
            const ma_tensor_t t = engine->getOutput(i);
            const ma_shape_t s  = engine->getOutputShape(i);
            std::printf("[MODEL] output[%d] name=%s type=%s shape=(", i, t.name ? t.name : "null", dtype_name(t.type));
            for (int j = 0; j < s.size; ++j) {
                std::printf("%d%s", s.dims[j], j + 1 < s.size ? "," : "");
            }
            std::printf(")\n");
        }
        return true;
    }

    // 默认对齐 torchvision ImageNet 预处理：
    // RGB -> resize(input_w,input_h) -> /255 -> (x-mean)/std
    // 若你的 export_onnx.py 已把 Normalize 写进模型，请把 mean 改为 0、std 改为 1。
    void preprocess(const ::cv::Mat& rgb) {
        ::cv::Mat resized;
        ::cv::resize(rgb, resized, ::cv::Size(input_w, input_h), 0, 0, ::cv::INTER_LINEAR);

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};

        for (int y = 0; y < input_h; ++y) {
            const uint8_t* row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < input_w; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const float real = (row[x * 3 + c] / 255.0f - mean[c]) / stdv[c];
                    size_t idx;
                    if (nchw) {
                        idx = static_cast<size_t>(c) * input_h * input_w + static_cast<size_t>(y) * input_w + x;
                    } else {
                        idx = (static_cast<size_t>(y) * input_w + x) * 3 + c;
                    }
                    weather::store_val(input_buf, input_type, input_qp, idx, real);
                }
            }
        }
    }

    bool infer(std::vector<float>& output, double& run_ms) {
        ma_tensor_t input = weather::make_input_tensor(input_type, input_buf, input_numel);
        input.type        = input_type;
        input.quant_param = input_qp;

        if (engine->setInput(0, input) != MA_OK) {
            std::fprintf(stderr, "[ERROR] setInput failed\n");
            return false;
        }

        const auto t0 = Clock::now();
        const int ret = engine->run();
        const auto t1 = Clock::now();
        run_ms        = ms_between(t0, t1);
        if (ret != MA_OK) {
            std::fprintf(stderr, "[ERROR] engine run failed: %d\n", ret);
            return false;
        }

        const ma_tensor_t out = engine->getOutput(0);
        const size_t n        = weather::shape_numel(engine->getOutputShape(0));
        output.resize(n);
        for (size_t i = 0; i < n; ++i)
            output[i] = weather::read_val(out, i);
        return true;
    }
};

static std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty())
        return {};
    const float maxv = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxv);
        sum += probs[i];
    }
    if (sum > 0) {
        for (float& p : probs)
            p = static_cast<float>(p / sum);
    }
    return probs;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static struct {
    std::string model_path  = "/userdata/local/models/weather_mobilenetv3_small_bf16.cvimodel";
    std::string labels_path = "/usr/share/weather-classifier/labels.txt";

    int print_interval = 30;
    int camera_w       = 640;
    int camera_h       = 480;
    int camera_fps     = 10;

    std::string mqtt_host  = "localhost";
    int mqtt_port          = 1883;
    std::string mqtt_topic = "recamera/weather/results";
    bool enable_mqtt       = true;

    bool enable_rtsp  = true;
    int stream_width  = 1280;
    int stream_height = 720;
    int stream_fps    = 15;

    bool enable_debug = true;  // H.264-over-WS + results JSON for supervisor console
    bool verbose      = false;
} g_config;

static ha_mqtt::MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera                         = nullptr;

static void signal_handler(int sig) {
    std::printf("[INFO] received signal %d, shutting down...\n", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    std::printf("Weather Classifier for ReCamera\n");
    std::printf("Usage: %s [options]\n\n", prog);
    std::printf("Options:\n");
    std::printf("  --model PATH           Classification model (default: %s)\n", g_config.model_path.c_str());
    std::printf("  --labels PATH          Labels file (default: %s)\n", g_config.labels_path.c_str());
    std::printf("  --print-interval N     Print every N frames (default: %d)\n", g_config.print_interval);
    std::printf("  --camera-width N       Inference frame width (default: %d)\n", g_config.camera_w);
    std::printf("  --camera-height N      Inference frame height (default: %d)\n", g_config.camera_h);
    std::printf("  --camera-fps N         Inference frame rate (default: %d)\n", g_config.camera_fps);
    std::printf("  -m, --mqtt-host HOST   MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    std::printf("  -p, --mqtt-port PORT   MQTT broker port (default: %d)\n", g_config.mqtt_port);
    std::printf("  --mqtt-topic TOPIC     MQTT publish topic (default: %s)\n", g_config.mqtt_topic.c_str());
    std::printf("  --no-mqtt              Disable MQTT publishing\n");
    std::printf("  --no-rtsp              Disable RTSP streaming\n");
    std::printf("  --stream-width N       RTSP encode width (default: %d)\n", g_config.stream_width);
    std::printf("  --stream-height N      RTSP encode height (default: %d)\n", g_config.stream_height);
    std::printf("  --stream-fps N         RTSP encode fps (default: %d)\n", g_config.stream_fps);
    std::printf("  -v, --verbose          Enable verbose logging\n");
    std::printf("  -h, --help             Show this help message\n\n");
    std::printf("RTSP stream: rtsp://<device-ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model",          required_argument, 0,  1 },
        {"labels",         required_argument, 0,  2 },
        {"print-interval", required_argument, 0,  3 },
        {"camera-width",   required_argument, 0,  4 },
        {"camera-height",  required_argument, 0,  5 },
        {"camera-fps",     required_argument, 0,  6 },
        {"mqtt-host",      required_argument, 0, 'm'},
        {"mqtt-port",      required_argument, 0, 'p'},
        {"mqtt-topic",     required_argument, 0,  7 },
        {"no-mqtt",        no_argument,       0,  8 },
        {"no-rtsp",        no_argument,       0,  9 },
        {"stream-width",   required_argument, 0, 10 },
        {"stream-height",  required_argument, 0, 11 },
        {"stream-fps",     required_argument, 0, 12 },
        {"verbose",        no_argument,       0, 'v'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case  1 : g_config.model_path     = optarg; break;
            case  2 : g_config.labels_path    = optarg; break;
            case  3 : g_config.print_interval = std::max(1, std::atoi(optarg)); break;
            case  4 : g_config.camera_w       = std::atoi(optarg); break;
            case  5 : g_config.camera_h       = std::atoi(optarg); break;
            case  6 : g_config.camera_fps     = std::atoi(optarg); break;
            case 'm': g_config.mqtt_host      = optarg; break;
            case 'p': g_config.mqtt_port      = std::atoi(optarg); break;
            case  7 : g_config.mqtt_topic     = optarg; break;
            case  8 : g_config.enable_mqtt    = false; break;
            case  9 : g_config.enable_rtsp    = false; break;
            case 10 : g_config.stream_width   = std::atoi(optarg); break;
            case 11 : g_config.stream_height  = std::atoi(optarg); break;
            case 12 : g_config.stream_fps     = std::atoi(optarg); break;
            case 'v': g_config.verbose        = true; break;
            case 'h': print_usage(argv[0]); std::exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Camera / RTSP / debug / MQTT bring-up (facemesh-reader pattern)
// ---------------------------------------------------------------------------

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            if (g_camera->init(0) != MA_OK) {
                std::fprintf(stderr, "[ERROR] camera init failed\n");
                return false;
            }

            Camera::CtrlValue value;

            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Cap the inference channel frame rate (else the capture FIFO fills
            // faster than a whole-frame classification drains it).
            value.i32 = g_config.camera_fps;
            g_camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, value);

            value.u16s[0] = static_cast<uint16_t>(g_config.camera_w);
            value.u16s[1] = static_cast<uint16_t>(g_config.camera_h);
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // CPU-accessible frames: retrieveFrame() hands back a virtual
            // address so the preprocess loop can read pixels directly (no
            // manual CVI_SYS_Mmap of a physical address).
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

static bool init_video_streaming() {
    if (!g_config.enable_rtsp) {
        std::printf("[INFO] RTSP streaming disabled\n");
        return true;
    }

    if (initVideo() != 0) {
        std::fprintf(stderr, "[ERROR] initVideo failed\n");
        return false;
    }

    video_ch_param_t stream_param{};
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width  = static_cast<uint32_t>(g_config.stream_width);
    stream_param.height = static_cast<uint32_t>(g_config.stream_height);
    stream_param.fps    = static_cast<uint8_t>(g_config.stream_fps);
    setupVideo(VIDEO_CH2, &stream_param);

    // RTSP owns VENC consumer index 0; debug_stream registers itself as index 1.
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));

    std::printf("[OK] RTSP streaming rtsp://<device-ip>:8554/live0 (%dx%d@%dfps)\n",
                g_config.stream_width, g_config.stream_height, g_config.stream_fps);
    return true;
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        std::printf("[INFO] MQTT publishing disabled\n");
        return true;
    }

    ha_mqtt::ClientOptions opts;
    opts.app_id        = "weather-classifier";
    opts.client_id     = "recamera-weather-classifier";
    opts.results_topic = g_config.mqtt_topic;
    opts.legacy_host   = g_config.mqtt_host;
    opts.legacy_port   = static_cast<uint16_t>(g_config.mqtt_port);

    // Home Assistant MQTT Discovery entity table (field names must match the
    // results JSON built by mqtt_payload.cpp).
    using ha_mqtt::EntityConfig;
    using ha_mqtt::EntityType;
    opts.entities = {
        EntityConfig{EntityType::Sensor, "weather_condition", "Weather Condition",
                     "{{ value_json.label }}",
                     "", "", ""},
        EntityConfig{EntityType::Sensor, "weather_confidence", "Weather Confidence",
                     "{{ (value_json.confidence * 100) | round(1) }}",
                     "", "%", "measurement"},
    };

    g_mqtt_publisher = new ha_mqtt::MqttPublisher();
    if (!g_mqtt_publisher->init(opts)) {
        std::fprintf(stderr, "[ERROR] MQTT publisher init failed\n");
        return false;
    }
    std::printf("[OK] MQTT publishing topic=%s\n", g_config.mqtt_topic.c_str());
    return true;
}

static void cleanup() {
    if (g_camera) {
        g_camera->stopStream();
        g_camera->deInit();
        g_camera = nullptr;
    }

    if (g_config.enable_debug) debug_stream_destroy();

    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    std::printf("[INFO] cleanup completed\n");
}

// Debug /results envelope: classification has no boxes — the overlay box array
// is empty and the classification result rides as extra top-level members.
static std::string build_debug_results_json(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::string& label, int class_id,
                                            float confidence,
                                            const std::vector<std::string>& labels,
                                            const std::vector<float>& probs,
                                            float inference_time_ms) {
    std::ostringstream extra;
    extra << std::fixed << std::setprecision(4);
    extra << "\"classification\":{";
    extra << "\"label\":\"" << label << "\",";
    extra << "\"class_id\":" << class_id << ",";
    extra << "\"confidence\":" << confidence << ",";
    extra << "\"scores\":{";
    for (size_t i = 0; i < probs.size(); ++i) {
        const std::string name = i < labels.size() ? labels[i] : ("class_" + std::to_string(i));
        if (i > 0) extra << ",";
        extra << "\"" << name << "\":" << probs[i];
    }
    extra << "}}";

    const std::vector<debug_stream_box_t> no_boxes;
    return debug_stream_build_results(timestamp_ms, frame_id, inference_time_ms,
                                      g_config.stream_width, g_config.stream_height,
                                      no_boxes, nullptr, extra.str());
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    std::printf("[INFO] starting weather classifier\n");
    std::printf("[INFO] model:  %s\n", g_config.model_path.c_str());
    std::printf("[INFO] labels: %s\n", g_config.labels_path.c_str());

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    const auto labels = load_labels(g_config.labels_path);
    if (labels.empty()) {
        std::fprintf(stderr, "[WARN] labels file empty or unreadable: %s\n", g_config.labels_path.c_str());
    }

    Classifier classifier;
    if (!classifier.init(g_config.model_path)) return 2;

    if (!init_camera())          { cleanup(); return 3; }
    if (!init_video_streaming()) { cleanup(); return 4; }

    // Debug stream (non-fatal: on failure run without it)
    if (g_config.enable_debug && debug_stream_start_or_disable(8001, VIDEO_CH2) != 0) {
        g_config.enable_debug = false;
    }

    if (!init_mqtt())            { cleanup(); return 5; }

    if (g_camera->startStream(Camera::StreamMode::kRefreshOnReturn) != MA_OK) {
        std::fprintf(stderr, "[ERROR] camera startStream failed\n");
        cleanup();
        return 6;
    }

    if (g_config.enable_rtsp) startVideo();

    std::printf("[OK] weather classifier running (camera %dx%d RGB888)\n",
                g_config.camera_w, g_config.camera_h);
    std::printf("[INFO] first 10 inference runs are warm-up and excluded from averages\n");

    uint64_t frame_id  = 0;
    uint64_t measured  = 0;
    double sum_capture = 0.0, sum_pre = 0.0, sum_run = 0.0, sum_total = 0.0;
    double min_run = 1e9, max_run = 0.0;

    while (g_running.load()) {
        const auto total0 = Clock::now();
        ma_img_t frame{};

        const auto cap0 = Clock::now();
        if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto cap1 = Clock::now();

        if (!frame.data || frame.width == 0 || frame.height == 0) {
            g_camera->returnFrame(frame);
            continue;
        }

        const int width  = static_cast<int>(frame.width);
        const int height = static_cast<int>(frame.height);
        int stride       = width * 3;
        if (height > 0 && frame.size > 0) {
            const size_t guessed = frame.size / static_cast<size_t>(height);
            if (guessed >= static_cast<size_t>(width * 3))
                stride = static_cast<int>(guessed);
        }

        // kPhysical=0: frame.data is a CPU-visible virtual address.
        ::cv::Mat rgb(height, width, CV_8UC3, frame.data, static_cast<size_t>(stride));

        const auto pre0 = Clock::now();
        classifier.preprocess(rgb);
        const auto pre1 = Clock::now();

        std::vector<float> logits;
        double run_ms = 0.0;
        const bool ok = classifier.infer(logits, run_ms);

        g_camera->returnFrame(frame);

        if (!ok || logits.empty())
            continue;

        // 若模型输出本身已经是概率，softmax 仍会改变结果。
        // 可通过观察输出和与 ONNX 对比确认；默认按 logits 处理。
        const std::vector<float> probs = softmax(logits);
        const auto best_it             = std::max_element(probs.begin(), probs.end());
        const size_t best              = static_cast<size_t>(std::distance(probs.begin(), best_it));
        const float score              = *best_it;

        const auto total1       = Clock::now();
        const double capture_ms = ms_between(cap0, cap1);
        const double pre_ms     = ms_between(pre0, pre1);
        const double total_ms   = ms_between(total0, total1);

        ++frame_id;
        if (frame_id > 10) {
            ++measured;
            sum_capture += capture_ms;
            sum_pre += pre_ms;
            sum_run += run_ms;
            sum_total += total_ms;
            min_run = std::min(min_run, run_ms);
            max_run = std::max(max_run, run_ms);
        }

        const std::string name = best < labels.size() ? labels[best] : ("class_" + std::to_string(best));

        // Push the same result to debug WS clients (sscma-node envelope).
        // debug_stream is lazy: skip building JSON when nobody is connected.
        if (g_config.enable_debug && debug_stream_results_client_count() > 0) {
            const uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::string dj = build_debug_results_json(timestamp_ms, static_cast<uint32_t>(frame_id),
                                                      name, static_cast<int>(best), score,
                                                      labels, probs, static_cast<float>(total_ms));
            debug_stream_publish_result(dj.c_str(), dj.size());
        }

        // Publish every classified frame over MQTT (independent of the stdout
        // print_interval throttle below).
        if (g_config.enable_mqtt && g_mqtt_publisher) {
            const std::string payload = weather_classifier::buildResultJson(
                frame_id, name, static_cast<int>(best), score, labels, probs,
                run_ms, capture_ms, pre_ms, total_ms);
            g_mqtt_publisher->publishResultsJson(payload);
        }

        if (g_config.verbose || frame_id % static_cast<uint64_t>(g_config.print_interval) == 0) {
            std::printf(
                "[RESULT] frame=%llu class=%zu(%s) score=%.4f | capture=%.2fms pre=%.2fms "
                "TPU_run=%.2fms total=%.2fms",
                static_cast<unsigned long long>(frame_id),
                best,
                name.c_str(),
                score,
                capture_ms,
                pre_ms,
                run_ms,
                total_ms);
            if (measured > 0) {
                std::printf(" | avg_run=%.2fms min=%.2fms max=%.2fms avg_total=%.2fms", sum_run / measured, min_run, max_run, sum_total / measured);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    cleanup();

    if (measured > 0) {
        std::printf("\n[SUMMARY] samples=%llu\n", static_cast<unsigned long long>(measured));
        std::printf("  avg capture : %.3f ms\n", sum_capture / measured);
        std::printf("  avg preprocess: %.3f ms\n", sum_pre / measured);
        std::printf("  avg TPU_run : %.3f ms (min %.3f, max %.3f)\n", sum_run / measured, min_run, max_run);
        std::printf("  avg total   : %.3f ms\n", sum_total / measured);
        std::printf("  inference FPS (TPU only): %.2f\n", 1000.0 / (sum_run / measured));
        std::printf("  end-to-end FPS: %.2f\n", 1000.0 / (sum_total / measured));
    }
    return 0;
}
