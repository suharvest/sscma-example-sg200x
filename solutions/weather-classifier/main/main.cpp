#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sscma.h>

#include <af/app.h>
#include <af/frame.h>
#include <debug_stream.h>
#include <ha_mqtt.h>

#include "engine_utils.h"
#include "mqtt_payload.h"
#include "onvif_meta.h"
#include "onvif_meta_gate.h"

using Clock = std::chrono::steady_clock;
using namespace ma;

#define TAG "weather-classifier"

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
    int rtsp_port     = 8554;

    bool enable_debug = true;  // H.264-over-WS + results JSON for supervisor console
    int debug_port    = 8001;
    bool verbose      = false;
} g_config;

// ONVIF analytics metadata, off unless switched on in the console. The gate
// owns the switch and the rate limit; when disabled it costs one bool test.
static OnvifMetaGate g_onvif_meta;

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
    std::printf("  --no-debug             Disable debug stream (WebSocket)\n");
    std::printf("  --debug-port PORT      Debug stream WebSocket port (default: %d)\n", g_config.debug_port);
    std::printf("  --rtsp-port PORT       RTSP stream port (default: %d)\n", g_config.rtsp_port);
    std::printf("  --stream-width N       Stream encode width (default: %d)\n", g_config.stream_width);
    std::printf("  --stream-height N      Stream encode height (default: %d)\n", g_config.stream_height);
    std::printf("  --stream-fps N         Stream encode fps (default: %d)\n", g_config.stream_fps);
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
        {"no-debug",       no_argument,       0, 13 },
        {"debug-port",     required_argument, 0, 14 },
        {"rtsp-port",      required_argument, 0, 15 },
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
            case 13 : g_config.enable_debug   = false; break;
            case 14 : g_config.debug_port     = std::atoi(optarg); break;
            case 15 : g_config.rtsp_port      = std::atoi(optarg); break;
            case 'v': g_config.verbose        = true; break;
            case 'h': print_usage(argv[0]); std::exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static ha_mqtt::ClientOptions mqtt_options() {
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
    return opts;
}

static af::StreamMode resolve_stream_mode() {
    const bool ws   = g_config.enable_debug;
    const bool rtsp = g_config.enable_rtsp;
    if (ws && rtsp) return af::StreamMode::Both;
    if (rtsp)       return af::StreamMode::RtspOnly;
    if (ws)         return af::StreamMode::WsOnly;
    return af::StreamMode::None;
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

    const auto labels = load_labels(g_config.labels_path);
    if (labels.empty()) {
        std::fprintf(stderr, "[WARN] labels file empty or unreadable: %s\n", g_config.labels_path.c_str());
    }

    Classifier classifier;
    if (!classifier.init(g_config.model_path)) return 2;

    af::App app("weather-classifier");
    app.with_camera({static_cast<uint16_t>(g_config.camera_w),
                     static_cast<uint16_t>(g_config.camera_h),
                     static_cast<uint8_t>(g_config.camera_fps)});

    af::StreamOptions stream_opt;
    stream_opt.width  = static_cast<uint16_t>(g_config.stream_width);
    stream_opt.height = static_cast<uint16_t>(g_config.stream_height);
    stream_opt.fps    = static_cast<uint8_t>(g_config.stream_fps);

    const af::StreamMode stream_mode = resolve_stream_mode();
    app.with_stream_mode(stream_mode, stream_opt, g_config.debug_port, g_config.rtsp_port);

    if (g_config.enable_mqtt) {
        const ha_mqtt::ClientOptions mqtt_opts = mqtt_options();
        app.with_mqtt(mqtt_opts);
        g_onvif_meta.reload(ha_mqtt::readDeviceIdentifier(), mqtt_opts.app_id);
        if (g_onvif_meta.enabled()) {
            std::printf("[OK] ONVIF metadata topic=%s every %ums\n",
                        g_onvif_meta.topic().c_str(), g_onvif_meta.config().interval_ms);
        }
    } else {
        std::printf("[INFO] MQTT publishing disabled\n");
    }

    const OnvifMetaConfig& oc = g_onvif_meta.config();
    if (oc.service_enabled && g_config.enable_rtsp && app.stream_mode() == af::StreamMode::WsOnly) {
        app.with_stream_mode(af::StreamMode::Both, stream_opt, g_config.debug_port, g_config.rtsp_port);
    }
    if (oc.service_enabled) {
        af::OnvifOptions onvif;
        onvif.port        = oc.service_port;
        onvif.serial      = ha_mqtt::readDeviceIdentifier();
        onvif.device_name = "reCamera";
        onvif.username    = oc.username;
        onvif.password    = oc.password;
        onvif.location    = oc.location;
        app.with_onvif(onvif);
    }

    std::printf("[OK] weather classifier running (camera %dx%d RGB888)\n",
                g_config.camera_w, g_config.camera_h);
    std::printf("[INFO] first 10 inference runs are warm-up and excluded from averages\n");

    uint64_t frame_id   = 0;
    uint64_t measured   = 0;
    double sum_capture  = 0.0, sum_pre = 0.0, sum_run = 0.0, sum_total = 0.0;
    double min_run      = 1e9, max_run = 0.0;
    Clock::time_point t_before_grab = Clock::now();

    app.on_before_grab([&] {
        t_before_grab = Clock::now();
    });

    app.on_frame([&](const af::VideoFrame& frame, af::Result& r) {
        r.publish = false;
        if (!frame.valid() || !frame.cpu_accessible()) {
            return;
        }

        const auto t_got_frame = Clock::now();
        const double capture_ms = ms_between(t_before_grab, t_got_frame);

        const int width     = static_cast<int>(frame.width);
        const int height    = static_cast<int>(frame.height);
        const size_t stride = static_cast<size_t>(frame.stride());

        // Wrap RGB mat from frame.data() and frame.stride()
        ::cv::Mat rgb(height, width, CV_8UC3, frame.data(), stride);

        const auto pre0 = Clock::now();
        classifier.preprocess(rgb);
        const auto pre1 = Clock::now();

        std::vector<float> logits;
        double run_ms = 0.0;
        const bool ok = classifier.infer(logits, run_ms);
        if (!ok || logits.empty()) {
            return;
        }

        // Calculate softmax and find top prediction
        const std::vector<float> probs = softmax(logits);
        const auto best_it             = std::max_element(probs.begin(), probs.end());
        const size_t best              = static_cast<size_t>(std::distance(probs.begin(), best_it));
        const float score              = *best_it;

        const auto total1     = Clock::now();
        const double pre_ms   = ms_between(pre0, pre1);
        const double total_ms = ms_between(t_before_grab, total1);

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

        // When app.debug_clients_count() > 0: build and send the classification debug result
        if (app.debug_clients_count() > 0) {
            const uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const std::string dj = build_debug_results_json(timestamp_ms, static_cast<uint32_t>(frame_id),
                                                            name, static_cast<int>(best), score,
                                                            labels, probs, static_cast<float>(total_ms));
            debug_stream_publish_result(dj.c_str(), dj.size());
        }

        // Publish MQTT result via app.mqtt()->publishResultsJson(payload)
        if (g_config.enable_mqtt && app.mqtt()) {
            const std::string payload = weather_classifier::buildResultJson(
                frame_id, name, static_cast<int>(best), score, labels, probs,
                run_ms, capture_ms, pre_ms, total_ms);
            app.mqtt()->publishResultsJson(payload);

            // Publish ONVIF verdict if enabled
            const uint64_t onvif_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (g_onvif_meta.take(onvif_ts)) {
                onvif_frame_t f;
                f.utc_ms  = onvif_ts;
                f.source  = "WeatherClassifier";
                f.frame_w = width;
                f.frame_h = height;
                onvif_object_t o;
                o.id = 1;
                o.cx = width * 0.5f;
                o.cy = height * 0.5f;
                o.w  = static_cast<float>(width);
                o.h  = static_cast<float>(height);
                o.classes.push_back({name, score});
                f.objects.push_back(std::move(o));
                app.mqtt()->publishText(g_onvif_meta.topic(), onvif_meta_to_json(f));
            }
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
                std::printf(" | avg_run=%.2fms min=%.2fms max=%.2fms avg_total=%.2fms",
                            sum_run / measured, min_run, max_run, sum_total / measured);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    });

    const int rc = app.run();

    if (measured > 0) {
        std::printf("\n[SUMMARY] samples=%llu\n", static_cast<unsigned long long>(measured));
        std::printf("  avg capture : %.3f ms\n", sum_capture / measured);
        std::printf("  avg preprocess: %.3f ms\n", sum_pre / measured);
        std::printf("  avg TPU_run : %.3f ms (min %.3f, max %.3f)\n", sum_run / measured, min_run, max_run);
        std::printf("  avg total   : %.3f ms\n", sum_total / measured);
        std::printf("  inference FPS (TPU only): %.2f\n", 1000.0 / (sum_run / measured));
        std::printf("  end-to-end FPS: %.2f\n", 1000.0 / (sum_total / measured));
    }

    return rc;
}
