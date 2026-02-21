/**
 * ppocr-debug: Standalone debug tool for PP-OCRv3 on reCamera
 *
 * Loads a static image, runs detection + recognition, and prints
 * detailed diagnostic info at every stage.
 *
 * Usage:
 *   ppocr-debug <image_path> [det_model] [rec_model] [dict]
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

#include <sscma.h>
#include <opencv2/opencv.hpp>

using namespace ma;
using namespace ma::engine;

// ============================================================
// Helpers
// ============================================================

static void dump_tensor_info(const char* label, const ma_tensor_t& t) {
    const char* type_str = "unknown";
    switch (t.type) {
        case MA_TENSOR_TYPE_U8:  type_str = "uint8"; break;
        case MA_TENSOR_TYPE_S8:  type_str = "int8";  break;
        case MA_TENSOR_TYPE_U16: type_str = "uint16"; break;
        case MA_TENSOR_TYPE_S16: type_str = "int16"; break;
        case MA_TENSOR_TYPE_U32: type_str = "uint32"; break;
        case MA_TENSOR_TYPE_S32: type_str = "int32"; break;
        case MA_TENSOR_TYPE_F32: type_str = "float32"; break;
        case MA_TENSOR_TYPE_F64: type_str = "float64"; break;
        default: break;
    }

    printf("  [%s] type=%s, size=%zu, shape=[", label, type_str, t.size);
    for (int i = 0; i < t.shape.size; i++) {
        if (i > 0) printf(",");
        printf("%d", t.shape.dims[i]);
    }
    printf("], quant=(scale=%.6f, zp=%d)\n", t.quant_param.scale, t.quant_param.zero_point);
}

static void dump_float_stats(const char* label, const float* data, int count) {
    float min_val = data[0], max_val = data[0], sum = 0;
    int nonzero = 0;
    int above_01 = 0, above_03 = 0, above_05 = 0, above_08 = 0;

    for (int i = 0; i < count; i++) {
        float v = data[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
        if (v != 0.0f) nonzero++;
        if (v > 0.1f) above_01++;
        if (v > 0.3f) above_03++;
        if (v > 0.5f) above_05++;
        if (v > 0.8f) above_08++;
    }

    printf("  [%s] count=%d, min=%.6f, max=%.6f, mean=%.6f\n",
           label, count, min_val, max_val, sum / count);
    printf("  [%s] nonzero=%d, >0.1=%d, >0.3=%d, >0.5=%d, >0.8=%d\n",
           label, nonzero, above_01, above_03, above_05, above_08);
}

static void dump_u8_stats(const char* label, const uint8_t* data, size_t count) {
    int hist[256] = {};
    for (size_t i = 0; i < count; i++) hist[data[i]]++;
    int min_val = 255, max_val = 0;
    long long sum = 0;
    for (size_t i = 0; i < count; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }
    printf("  [%s] count=%zu, min=%d, max=%d, mean=%.1f\n",
           label, count, min_val, max_val, (double)sum / count);
    printf("  [%s] hist[0]=%d, hist[128]=%d, hist[255]=%d\n",
           label, hist[0], hist[128], hist[255]);
}

// ============================================================
// Letterbox preprocess (same as ppocr-reader)
// ============================================================

struct LetterboxInfo {
    float scale;
    int pad_left, pad_top;
    int orig_w, orig_h;
    int new_w, new_h;
};

static std::vector<uint8_t> letterbox_resize(const ::cv::Mat& img, int target_w, int target_h, LetterboxInfo& info) {
    info.orig_w = img.cols;
    info.orig_h = img.rows;

    float scale_w = (float)target_w / img.cols;
    float scale_h = (float)target_h / img.rows;
    info.scale = std::min(scale_w, scale_h);

    info.new_w = (int)(img.cols * info.scale);
    info.new_h = (int)(img.rows * info.scale);
    info.pad_left = (target_w - info.new_w) / 2;
    info.pad_top = (target_h - info.new_h) / 2;

    // Resize
    ::cv::Mat resized;
    ::cv::resize(img, resized, ::cv::Size(info.new_w, info.new_h), 0, 0, ::cv::INTER_LINEAR);

    // Create letterbox canvas (gray 128)
    size_t buf_size = target_w * target_h * 3;
    std::vector<uint8_t> buf(buf_size, 128);

    // Copy resized image into canvas
    for (int y = 0; y < info.new_h; y++) {
        const uint8_t* src_row = resized.ptr<uint8_t>(y);
        uint8_t* dst_row = buf.data() + ((y + info.pad_top) * target_w + info.pad_left) * 3;
        memcpy(dst_row, src_row, info.new_w * 3);
    }

    return buf;
}

// ============================================================
// Load dictionary
// ============================================================

static std::vector<std::string> load_dict(const std::string& path) {
    std::vector<std::string> dict;
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("ERROR: Cannot open dictionary: %s\n", path.c_str());
        return dict;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        dict.push_back(line);
    }
    // PP-OCRv3 class layout: [blank, char1, ..., charN, space]
    // Insert blank at the beginning (index 0), space at the end
    dict.insert(dict.begin(), "");  // index 0: CTC blank
    dict.push_back(" ");            // last index: space
    printf("  Dictionary loaded: %zu classes (blank + %zu chars + space, from %s)\n",
           dict.size(), dict.size() - 2, path.c_str());
    return dict;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ppocr-debug <image_path> [det_model] [rec_model] [dict]\n");
        printf("\nDefaults:\n");
        printf("  det_model: /userdata/local/models/ppocr_det_cv181x_int8.cvimodel\n");
        printf("  rec_model: /userdata/local/models/ppocr_rec_cv181x_int8.cvimodel\n");
        printf("  dict:      /userdata/local/dict/ppocr_keys_v1.txt\n");
        return 1;
    }

    const char* image_path = argv[1];
    const char* det_model = argc > 2 ? argv[2] : "/userdata/local/models/ppocr_det_cv181x_int8.cvimodel";
    const char* rec_model = argc > 3 ? argv[3] : "/userdata/local/models/ppocr_rec_cv181x_int8.cvimodel";
    const char* dict_path = argc > 4 ? argv[4] : "/userdata/local/dict/ppocr_keys_v1.txt";

    printf("========================================\n");
    printf("PP-OCR Debug Tool\n");
    printf("========================================\n");
    printf("Image:     %s\n", image_path);
    printf("Det model: %s\n", det_model);
    printf("Rec model: %s\n", rec_model);
    printf("Dict:      %s\n", dict_path);
    printf("\n");

    // ---- Load image ----
    printf("[1] Loading image...\n");
    ::cv::Mat img_bgr = ::cv::imread(image_path, ::cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        printf("ERROR: Cannot read image: %s\n", image_path);
        return 1;
    }
    printf("  Image loaded: %dx%d, channels=%d\n", img_bgr.cols, img_bgr.rows, img_bgr.channels());

    // Convert BGR -> RGB
    ::cv::Mat img_rgb;
    ::cv::cvtColor(img_bgr, img_rgb, ::cv::COLOR_BGR2RGB);

    // ---- Load detection model ----
    printf("\n[2] Loading detection model...\n");
    EngineCVI det_engine;
    if (det_engine.init() != MA_OK) {
        printf("ERROR: Failed to init det engine\n");
        return 1;
    }
    if (det_engine.load(det_model) != MA_OK) {
        printf("ERROR: Failed to load det model: %s\n", det_model);
        return 1;
    }

    int det_in_count = det_engine.getInputSize();
    int det_out_count = det_engine.getOutputSize();
    printf("  Det model loaded: %d inputs, %d outputs\n", det_in_count, det_out_count);

    ma_tensor_t det_input = det_engine.getInput(0);
    dump_tensor_info("det_input", det_input);

    for (int i = 0; i < det_out_count; i++) {
        ma_tensor_t out = det_engine.getOutput(i);
        char label[32];
        snprintf(label, sizeof(label), "det_output[%d]", i);
        dump_tensor_info(label, out);
    }

    // Determine input dimensions (NHWC)
    ma_shape_t in_shape = det_engine.getInputShape(0);
    int in_h, in_w;
    bool is_nhwc = (in_shape.dims[3] == 3 || in_shape.dims[3] == 1);
    if (is_nhwc) {
        in_h = in_shape.dims[1];
        in_w = in_shape.dims[2];
    } else {
        in_h = in_shape.dims[2];
        in_w = in_shape.dims[3];
    }
    printf("  Det input resolution: %dx%d (is_nhwc=%d)\n", in_w, in_h, is_nhwc);

    // ---- Preprocess: letterbox ----
    printf("\n[3] Preprocessing (letterbox %dx%d -> %dx%d)...\n", img_rgb.cols, img_rgb.rows, in_w, in_h);
    LetterboxInfo lb;
    std::vector<uint8_t> lb_buf = letterbox_resize(img_rgb, in_w, in_h, lb);
    printf("  Letterbox: scale=%.4f, new_size=%dx%d, pad=(%d,%d)\n",
           lb.scale, lb.new_w, lb.new_h, lb.pad_left, lb.pad_top);

    dump_u8_stats("letterbox_buf", lb_buf.data(), lb_buf.size());

    // Save letterbox image for visual verification
    {
        ::cv::Mat lb_mat(in_h, in_w, CV_8UC3, lb_buf.data());
        ::cv::Mat lb_bgr;
        ::cv::cvtColor(lb_mat, lb_bgr, ::cv::COLOR_RGB2BGR);
        ::cv::imwrite("/tmp/debug_letterbox.jpg", lb_bgr);
        printf("  Saved letterbox image: /tmp/debug_letterbox.jpg\n");
    }

    // Copy to model input
    printf("\n[4] Copying to model input tensor...\n");
    size_t copy_size = lb_buf.size();
    printf("  Buffer size: %zu, Tensor size: %zu\n", copy_size, det_input.size);

    if (copy_size > det_input.size) {
        printf("  WARNING: buffer (%zu) > tensor (%zu), truncating!\n", copy_size, det_input.size);
        copy_size = det_input.size;
    }
    memcpy(det_input.data.u8, lb_buf.data(), copy_size);

    // Check if INT8 offset is needed
    printf("  Input tensor type: %d (U8=%d, S8=%d)\n",
           det_input.type, MA_TENSOR_TYPE_U8, MA_TENSOR_TYPE_S8);

    if (det_input.type == MA_TENSOR_TYPE_S8) {
        printf("  *** APPLYING INT8 OFFSET (-128) ***\n");
        for (size_t i = 0; i < det_input.size; i++) {
            det_input.data.u8[i] -= 128;
        }
        dump_u8_stats("input_after_offset", det_input.data.u8, std::min(det_input.size, copy_size));
    } else {
        printf("  No INT8 offset needed (type is not S8)\n");
    }

    // ---- Run detection ----
    printf("\n[5] Running detection inference...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    ma_err_t ret = det_engine.run();
    auto t1 = std::chrono::high_resolution_clock::now();
    float det_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (ret != MA_OK) {
        printf("ERROR: Detection inference failed: %d\n", ret);
        return 1;
    }
    printf("  Detection inference: %.1f ms (ret=%d)\n", det_ms, ret);

    // ---- Analyze detection output ----
    printf("\n[6] Analyzing detection output...\n");
    ma_tensor_t det_output = det_engine.getOutput(0);
    dump_tensor_info("det_output", det_output);

    int out_h = 0, out_w = 0;
    if (det_output.shape.size == 4) {
        out_h = det_output.shape.dims[2];
        out_w = det_output.shape.dims[3];
    } else if (det_output.shape.size == 3) {
        out_h = det_output.shape.dims[1];
        out_w = det_output.shape.dims[2];
    }
    printf("  Output map: %dx%d\n", out_w, out_h);

    int map_size = out_h * out_w;
    std::vector<float> prob_map(map_size);

    if (det_output.type == MA_TENSOR_TYPE_S8) {
        printf("  Output is INT8, dequantizing...\n");
        float scale = det_output.quant_param.scale;
        int32_t zp = det_output.quant_param.zero_point;
        printf("  scale=%.8f, zp=%d\n", scale, zp);
        for (int i = 0; i < map_size; i++) {
            prob_map[i] = (float)(det_output.data.s8[i] - zp) * scale;
        }
    } else if (det_output.type == MA_TENSOR_TYPE_F32) {
        printf("  Output is F32, direct copy\n");
        memcpy(prob_map.data(), det_output.data.f32, map_size * sizeof(float));
    } else {
        printf("  WARNING: Unexpected output type %d\n", det_output.type);
        // Try reading as f32 anyway
        memcpy(prob_map.data(), det_output.data.f32, map_size * sizeof(float));
    }

    dump_float_stats("prob_map", prob_map.data(), map_size);

    // Save probability map as image
    {
        ::cv::Mat prob_mat(out_h, out_w, CV_32F, prob_map.data());
        ::cv::Mat prob_vis;
        prob_mat.convertTo(prob_vis, CV_8UC1, 255.0);
        ::cv::imwrite("/tmp/debug_prob_map.jpg", prob_vis);
        printf("  Saved probability map: /tmp/debug_prob_map.jpg\n");
    }

    // ---- DBNet postprocess ----
    printf("\n[7] DBNet postprocess...\n");
    float det_threshold = 0.3f;
    float box_threshold = 0.5f;
    float unclip_ratio = 1.6f;
    int min_box_size = 3;

    ::cv::Mat prob_mat(out_h, out_w, CV_32F, prob_map.data());
    ::cv::Mat binary_mat;
    ::cv::threshold(prob_mat, binary_mat, det_threshold, 1.0f, ::cv::THRESH_BINARY);

    ::cv::Mat binary_u8;
    binary_mat.convertTo(binary_u8, CV_8UC1, 255);

    int white_pixels = ::cv::countNonZero(binary_u8);
    printf("  Threshold=%.2f: %d/%d pixels above threshold (%.2f%%)\n",
           det_threshold, white_pixels, map_size, 100.0f * white_pixels / map_size);

    // Save binary map
    ::cv::imwrite("/tmp/debug_binary_map.jpg", binary_u8);
    printf("  Saved binary map: /tmp/debug_binary_map.jpg\n");

    // Also try lower thresholds
    for (float th : {0.2f, 0.1f, 0.05f, 0.01f}) {
        ::cv::Mat tmp;
        ::cv::threshold(prob_mat, tmp, th, 1.0f, ::cv::THRESH_BINARY);
        ::cv::Mat tmp8;
        tmp.convertTo(tmp8, CV_8UC1, 255);
        int cnt = ::cv::countNonZero(tmp8);
        printf("  Threshold=%.2f: %d pixels (%.2f%%)\n", th, cnt, 100.0f * cnt / map_size);
    }

    // Find contours
    std::vector<std::vector<::cv::Point>> contours;
    ::cv::findContours(binary_u8, contours, ::cv::RETR_LIST, ::cv::CHAIN_APPROX_SIMPLE);
    printf("  Contours found: %zu\n", contours.size());

    // Process contours into boxes
    struct DetBox {
        float points[4][2];
        float score;
        float width, height;
    };
    std::vector<DetBox> det_boxes;

    for (size_t ci = 0; ci < contours.size(); ci++) {
        const auto& contour = contours[ci];
        if (contour.size() < 3) continue;

        ::cv::RotatedRect rect = ::cv::minAreaRect(contour);
        float w = std::min(rect.size.width, rect.size.height);
        float h = std::max(rect.size.width, rect.size.height);

        // Calculate score
        ::cv::Mat mask = ::cv::Mat::zeros(out_h, out_w, CV_8UC1);
        std::vector<std::vector<::cv::Point>> single = {contour};
        ::cv::fillPoly(mask, single, ::cv::Scalar(255));
        float score = ::cv::mean(prob_mat, mask)[0];

        printf("  Contour[%zu]: area=%.0f, w=%.1f, h=%.1f, score=%.4f",
               ci, ::cv::contourArea(contour), w, h, score);

        if (w < min_box_size || h < min_box_size) {
            printf(" -> SKIP (too small)\n");
            continue;
        }
        if (score < box_threshold) {
            printf(" -> SKIP (score < %.2f)\n", box_threshold);
            continue;
        }
        printf(" -> ACCEPT\n");

        // Get corners and map to original image space
        ::cv::Point2f corners[4];
        rect.points(corners);

        DetBox box;
        box.score = score;
        float scale_x = (float)in_w / out_w;
        float scale_y = (float)in_h / out_h;

        for (int i = 0; i < 4; i++) {
            float x = corners[i].x * scale_x;
            float y = corners[i].y * scale_y;
            // Remove letterbox offset -> original image coords
            x = (x - lb.pad_left) / lb.scale;
            y = (y - lb.pad_top) / lb.scale;
            x = std::max(0.0f, std::min((float)lb.orig_w, x));
            y = std::max(0.0f, std::min((float)lb.orig_h, y));
            box.points[i][0] = x;
            box.points[i][1] = y;
        }
        box.width = w * scale_x / lb.scale;
        box.height = h * scale_y / lb.scale;
        det_boxes.push_back(box);
    }

    printf("\n  Detected text boxes: %zu\n", det_boxes.size());
    for (size_t i = 0; i < det_boxes.size(); i++) {
        const auto& b = det_boxes[i];
        printf("  Box[%zu]: score=%.4f, size=%.0fx%.0f, corners=[(%.0f,%.0f),(%.0f,%.0f),(%.0f,%.0f),(%.0f,%.0f)]\n",
               i, b.score, b.width, b.height,
               b.points[0][0], b.points[0][1], b.points[1][0], b.points[1][1],
               b.points[2][0], b.points[2][1], b.points[3][0], b.points[3][1]);
    }

    // ---- Recognition (if boxes found) ----
    if (det_boxes.empty()) {
        printf("\n[8] No text boxes detected - skipping recognition.\n");
        printf("\n*** DIAGNOSIS: Detection model output has max=%.6f ***\n",
               *std::max_element(prob_map.begin(), prob_map.end()));
        if (*std::max_element(prob_map.begin(), prob_map.end()) < 0.01f) {
            printf("*** All output values near zero - model may not be working correctly.\n");
            printf("*** Check: preprocessing, model conversion, calibration data.\n");
        } else if (*std::max_element(prob_map.begin(), prob_map.end()) < det_threshold) {
            printf("*** Max value below threshold (%.2f) - try lowering det_threshold.\n", det_threshold);
        }

        // Try with very low threshold for diagnosis
        printf("\n  Trying threshold=0.01 for diagnostic...\n");
        ::cv::Mat low_binary;
        ::cv::threshold(prob_mat, low_binary, 0.01f, 1.0f, ::cv::THRESH_BINARY);
        ::cv::Mat low_u8;
        low_binary.convertTo(low_u8, CV_8UC1, 255);
        std::vector<std::vector<::cv::Point>> low_contours;
        ::cv::findContours(low_u8, low_contours, ::cv::RETR_LIST, ::cv::CHAIN_APPROX_SIMPLE);
        printf("  Low-threshold contours: %zu\n", low_contours.size());
        for (size_t ci = 0; ci < std::min(low_contours.size(), (size_t)5); ci++) {
            ::cv::RotatedRect r = ::cv::minAreaRect(low_contours[ci]);
            ::cv::Mat m = ::cv::Mat::zeros(out_h, out_w, CV_8UC1);
            std::vector<std::vector<::cv::Point>> s = {low_contours[ci]};
            ::cv::fillPoly(m, s, ::cv::Scalar(255));
            float sc = ::cv::mean(prob_mat, m)[0];
            printf("  LowContour[%zu]: area=%.0f, rect=%.0fx%.0f, score=%.6f\n",
                   ci, ::cv::contourArea(low_contours[ci]), r.size.width, r.size.height, sc);
        }
    } else {
        printf("\n[8] Loading recognition model...\n");
        EngineCVI rec_engine;
        if (rec_engine.init() != MA_OK || rec_engine.load(rec_model) != MA_OK) {
            printf("ERROR: Failed to load rec model\n");
            return 1;
        }

        ma_tensor_t rec_input = rec_engine.getInput(0);
        dump_tensor_info("rec_input", rec_input);

        ma_shape_t rec_shape = rec_engine.getInputShape(0);
        int rec_h, rec_w;
        bool rec_nhwc = (rec_shape.dims[3] == 3);
        if (rec_nhwc) {
            rec_h = rec_shape.dims[1];
            rec_w = rec_shape.dims[2];
        } else {
            rec_h = rec_shape.dims[2];
            rec_w = rec_shape.dims[3];
        }
        printf("  Rec input: %dx%d\n", rec_w, rec_h);

        // Load dictionary
        auto dict = load_dict(dict_path);

        // Process each box
        for (size_t bi = 0; bi < det_boxes.size(); bi++) {
            const auto& box = det_boxes[bi];
            printf("\n  --- Recognition Box[%zu] ---\n", bi);

            // Get bounding rect of the 4 points
            float min_x = box.points[0][0], max_x = box.points[0][0];
            float min_y = box.points[0][1], max_y = box.points[0][1];
            for (int i = 1; i < 4; i++) {
                min_x = std::min(min_x, box.points[i][0]);
                max_x = std::max(max_x, box.points[i][0]);
                min_y = std::min(min_y, box.points[i][1]);
                max_y = std::max(max_y, box.points[i][1]);
            }

            int x1 = std::max(0, (int)min_x);
            int y1 = std::max(0, (int)min_y);
            int x2 = std::min(img_rgb.cols, (int)max_x + 1);
            int y2 = std::min(img_rgb.rows, (int)max_y + 1);

            if (x2 - x1 < 2 || y2 - y1 < 2) {
                printf("  Crop too small: %dx%d\n", x2 - x1, y2 - y1);
                continue;
            }

            // Add vertical padding (50% of height on each side)
            int crop_h = y2 - y1;
            int pad = (int)(crop_h * 0.8f);
            int y1_padded = std::max(0, y1 - pad);
            int y2_padded = std::min(img_rgb.rows, y2 + pad);
            printf("  Original crop: (%d,%d)-(%d,%d) = %dx%d\n", x1, y1, x2, y2, x2-x1, y2-y1);
            printf("  Padded crop:   (%d,%d)-(%d,%d) = %dx%d (pad=%d)\n", x1, y1_padded, x2, y2_padded, x2-x1, y2_padded-y1_padded, pad);
            y1 = y1_padded;
            y2 = y2_padded;

            // Crop
            ::cv::Mat crop_rgb = img_rgb(::cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
            printf("  Crop: (%d,%d)-(%d,%d) = %dx%d\n", x1, y1, x2, y2, crop_rgb.cols, crop_rgb.rows);

            // Resize to rec input (keep aspect ratio, pad right)
            float ratio = (float)rec_h / crop_rgb.rows;
            int resize_w = std::min((int)(crop_rgb.cols * ratio), rec_w);
            ::cv::Mat resized;
            ::cv::resize(crop_rgb, resized, ::cv::Size(resize_w, rec_h), 0, 0, ::cv::INTER_LINEAR);

            // Create padded buffer
            std::vector<uint8_t> rec_buf(rec_w * rec_h * 3, 0);
            for (int y = 0; y < rec_h; y++) {
                const uint8_t* src = resized.ptr<uint8_t>(y);
                memcpy(rec_buf.data() + y * rec_w * 3, src, resize_w * 3);
            }

            // Save crop for verification
            char crop_path[64];
            snprintf(crop_path, sizeof(crop_path), "/tmp/debug_crop_%zu.jpg", bi);
            ::cv::Mat crop_bgr;
            ::cv::cvtColor(crop_rgb, crop_bgr, ::cv::COLOR_RGB2BGR);
            ::cv::imwrite(crop_path, crop_bgr);
            printf("  Saved crop: %s\n", crop_path);

            // Copy to rec input tensor
            size_t rec_copy = std::min(rec_buf.size(), rec_input.size);
            memcpy(rec_input.data.u8, rec_buf.data(), rec_copy);

            if (rec_input.type == MA_TENSOR_TYPE_S8) {
                printf("  Applying INT8 offset to rec input\n");
                for (size_t i = 0; i < rec_input.size; i++) {
                    rec_input.data.u8[i] -= 128;
                }
            }

            // Run recognition
            auto rt0 = std::chrono::high_resolution_clock::now();
            ret = rec_engine.run();
            auto rt1 = std::chrono::high_resolution_clock::now();
            float rec_ms = std::chrono::duration<float, std::milli>(rt1 - rt0).count();

            if (ret != MA_OK) {
                printf("  ERROR: Recognition failed: %d\n", ret);
                continue;
            }
            printf("  Recognition inference: %.1f ms\n", rec_ms);

            // Get output
            ma_tensor_t rec_output = rec_engine.getOutput(0);
            dump_tensor_info("rec_output", rec_output);

            // Decode: output shape should be [1, T, C] where T=timesteps, C=classes
            int timesteps = 0, num_classes = 0;
            if (rec_output.shape.size == 4) {
                // [1, T, C, 1]
                timesteps = rec_output.shape.dims[1];
                num_classes = rec_output.shape.dims[2];
            } else if (rec_output.shape.size == 3) {
                timesteps = rec_output.shape.dims[1];
                num_classes = rec_output.shape.dims[2];
            }
            printf("  Output: timesteps=%d, num_classes=%d\n", timesteps, num_classes);

            // CTC greedy decode with detailed diagnostics
            if (rec_output.type == MA_TENSOR_TYPE_F32 && timesteps > 0 && num_classes > 0) {
                const float* output_data = rec_output.data.f32;

                // First: analyze raw output statistics
                printf("  --- Raw output analysis ---\n");
                {
                    float gmin = output_data[0], gmax = output_data[0];
                    double gsum = 0;
                    int total = timesteps * num_classes;
                    for (int i = 0; i < total; i++) {
                        if (output_data[i] < gmin) gmin = output_data[i];
                        if (output_data[i] > gmax) gmax = output_data[i];
                        gsum += output_data[i];
                    }
                    printf("  Global: min=%.6f, max=%.6f, mean=%.6f\n", gmin, gmax, gsum / total);
                }

                // Print first 3 timesteps raw top-5 values
                printf("  --- First 5 timesteps raw top-5 ---\n");
                for (int t = 0; t < std::min(5, timesteps); t++) {
                    const float* row = output_data + t * num_classes;
                    // Find top 5
                    int top5_idx[5] = {0,0,0,0,0};
                    float top5_val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                    for (int c = 0; c < num_classes; c++) {
                        for (int k = 0; k < 5; k++) {
                            if (row[c] > top5_val[k]) {
                                for (int j = 4; j > k; j--) {
                                    top5_val[j] = top5_val[j-1];
                                    top5_idx[j] = top5_idx[j-1];
                                }
                                top5_val[k] = row[c];
                                top5_idx[k] = c;
                                break;
                            }
                        }
                    }
                    printf("    t=%2d:", t);
                    for (int k = 0; k < 5; k++) {
                        const char* ch = (top5_idx[k] < (int)dict.size()) ? dict[top5_idx[k]].c_str() : "?";
                        if (top5_idx[k] == 0) ch = "<blank>";
                        printf(" [%d]='%s'(%.4f)", top5_idx[k], ch, top5_val[k]);
                    }
                    printf("\n");

                    // Also print row stats
                    float rmin = row[0], rmax = row[0];
                    double rsum = 0;
                    for (int c = 0; c < num_classes; c++) {
                        if (row[c] < rmin) rmin = row[c];
                        if (row[c] > rmax) rmax = row[c];
                        rsum += row[c];
                    }
                    printf("          row: min=%.6f, max=%.6f, mean=%.8f, sum=%.4f\n", rmin, rmax, rsum / num_classes, (float)rsum);
                }

                // Check if output looks like logits (large range) or softmax (0-1, sums to ~1)
                {
                    const float* row0 = output_data;
                    float sum0 = 0;
                    for (int c = 0; c < num_classes; c++) sum0 += row0[c];
                    bool is_softmax = (sum0 > 0.9f && sum0 < 1.1f);
                    printf("  Row0 sum = %.6f -> %s\n", sum0, is_softmax ? "looks like SOFTMAX" : "looks like LOGITS (need softmax)");
                }

                // Apply softmax per timestep, then CTC decode
                printf("  --- CTC decode with softmax ---\n");
                std::vector<float> softmax_buf(num_classes);
                std::string text;
                float total_conf = 0;
                int char_count = 0;
                int prev_idx = -1;
                int blank_idx = 0;

                for (int t = 0; t < timesteps; t++) {
                    const float* row = output_data + t * num_classes;

                    // Softmax
                    float max_logit = row[0];
                    for (int c = 1; c < num_classes; c++)
                        if (row[c] > max_logit) max_logit = row[c];

                    float exp_sum = 0;
                    for (int c = 0; c < num_classes; c++) {
                        softmax_buf[c] = expf(row[c] - max_logit);
                        exp_sum += softmax_buf[c];
                    }
                    for (int c = 0; c < num_classes; c++)
                        softmax_buf[c] /= exp_sum;

                    // Argmax on softmax
                    int max_idx = 0;
                    float max_val = softmax_buf[0];
                    for (int c = 1; c < num_classes; c++) {
                        if (softmax_buf[c] > max_val) {
                            max_val = softmax_buf[c];
                            max_idx = c;
                        }
                    }

                    if (max_idx != blank_idx && max_idx != prev_idx) {
                        if (max_idx < (int)dict.size()) {
                            text += dict[max_idx];
                            total_conf += max_val;
                            char_count++;
                            printf("    t=%2d: idx=%d, softmax_conf=%.4f, char='%s'\n",
                                   t, max_idx, max_val, dict[max_idx].c_str());
                        }
                    }
                    prev_idx = max_idx;
                }

                float avg_conf = char_count > 0 ? total_conf / char_count : 0;
                printf("  RESULT: \"%s\" (confidence=%.4f, chars=%d)\n", text.c_str(), avg_conf, char_count);

                // Also try without softmax (raw argmax) for comparison
                printf("  --- CTC decode WITHOUT softmax (raw argmax) ---\n");
                {
                    std::string text2;
                    float total_conf2 = 0;
                    int char_count2 = 0;
                    int prev_idx2 = -1;
                    for (int t = 0; t < timesteps; t++) {
                        const float* row = output_data + t * num_classes;
                        int max_idx = 0;
                        float max_val = row[0];
                        for (int c = 1; c < num_classes; c++) {
                            if (row[c] > max_val) {
                                max_val = row[c];
                                max_idx = c;
                            }
                        }
                        if (max_idx != blank_idx && max_idx != prev_idx2) {
                            if (max_idx < (int)dict.size()) {
                                text2 += dict[max_idx];
                                total_conf2 += max_val;
                                char_count2++;
                            }
                        }
                        prev_idx2 = max_idx;
                    }
                    float avg_conf2 = char_count2 > 0 ? total_conf2 / char_count2 : 0;
                    printf("  RAW RESULT: \"%s\" (confidence=%.4f, chars=%d)\n", text2.c_str(), avg_conf2, char_count2);
                }
            }
        }
    }

    printf("\n========================================\n");
    printf("Debug complete. Check /tmp/debug_*.jpg\n");
    printf("========================================\n");

    return 0;
}
