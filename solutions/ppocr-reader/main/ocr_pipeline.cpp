#include "ocr_pipeline.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>

#define TAG "OcrPipeline"

namespace ppocr {

OcrPipeline::OcrPipeline()
    : max_boxes_(5), enhance_mode_(EnhanceMode::kAdaptive), initialized_(false), rec_available_(false), dbg_dump_frame_(0), prev_match_count_(0) {}

OcrPipeline::~OcrPipeline() {}

bool OcrPipeline::init(const std::string& det_model_path,
                        const std::string& rec_model_path,
                        const std::string& dict_path) {
    if (!detector_.init(det_model_path)) {
        MA_LOGE(TAG, "Failed to initialize text detector");
        return false;
    }

    if (!recognizer_.init(rec_model_path, dict_path)) {
        MA_LOGW(TAG, "Text recognizer init failed - detection only mode");
        rec_available_ = false;
    } else {
        rec_available_ = true;
    }

    // Pre-allocate crop buffer (max reasonable text region at 640x480 input)
    crop_buffer_.resize(640 * 480 * 3);

    initialized_ = true;
    MA_LOGI(TAG, "OCR pipeline initialized (recognition: %s)",
            rec_available_ ? "enabled" : "disabled");
    return true;
}

void OcrPipeline::setMaxBoxes(size_t max_boxes) {
    max_boxes_ = max_boxes;
}

void OcrPipeline::setEnhanceMode(EnhanceMode mode) {
    enhance_mode_ = mode;
    const char* names[] = {"none", "clahe", "gray", "adaptive"};
    MA_LOGI(TAG, "Enhance mode: %s", names[static_cast<int>(mode)]);
}

void OcrPipeline::sortBoxes(std::vector<TextBox>& boxes) {
    // Sort boxes top-to-bottom, left-to-right
    // Group by approximate row (within 20px vertical tolerance)
    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) {
        float ay = std::min({a.points[0][1], a.points[1][1], a.points[2][1], a.points[3][1]});
        float by = std::min({b.points[0][1], b.points[1][1], b.points[2][1], b.points[3][1]});

        if (std::abs(ay - by) < 20.0f) {
            float ax = std::min({a.points[0][0], a.points[1][0], a.points[2][0], a.points[3][0]});
            float bx = std::min({b.points[0][0], b.points[1][0], b.points[2][0], b.points[3][0]});
            return ax < bx;
        }
        return ay < by;
    });
}

bool OcrPipeline::cropTextRegion(const ma_img_t* img, const TextBox& box,
                                  std::vector<uint8_t>& output, int& out_w, int& out_h) {
    // Source points (the detected text polygon)
    // Points are ordered: [0]=TL, [1]=TR, [2]=BR, [3]=BL
    cv::Point2f src_pts[4];
    for (int i = 0; i < 4; ++i) {
        src_pts[i] = cv::Point2f(box.points[i][0], box.points[i][1]);
    }

    // Calculate destination size from polygon dimensions
    float width_top = std::sqrt(
        std::pow(src_pts[1].x - src_pts[0].x, 2) + std::pow(src_pts[1].y - src_pts[0].y, 2));
    float width_bot = std::sqrt(
        std::pow(src_pts[2].x - src_pts[3].x, 2) + std::pow(src_pts[2].y - src_pts[3].y, 2));
    float height_left = std::sqrt(
        std::pow(src_pts[3].x - src_pts[0].x, 2) + std::pow(src_pts[3].y - src_pts[0].y, 2));
    float height_right = std::sqrt(
        std::pow(src_pts[2].x - src_pts[1].x, 2) + std::pow(src_pts[2].y - src_pts[1].y, 2));

    out_w = static_cast<int>(std::max(width_top, width_bot));
    out_h = static_cast<int>(std::max(height_left, height_right));

    if (out_w < 2 || out_h < 2) return false;

    // Pad the source quad to give the recognizer context around text edges.
    // Use the text height axis (mid_top → mid_bottom) for vertical padding,
    // and the text direction (TL→TR) for horizontal padding.
    float text_h = std::max(height_left, height_right);
    float pad_v = text_h * 0.12f;   // 12% vertical padding each side
    float pad_h = std::max(width_top, width_bot) * 0.06f;  // 6% horizontal padding each side

    // Vertical axis: midpoint of top edge → midpoint of bottom edge
    cv::Point2f mid_top((src_pts[0].x + src_pts[1].x) * 0.5f, (src_pts[0].y + src_pts[1].y) * 0.5f);
    cv::Point2f mid_bot((src_pts[2].x + src_pts[3].x) * 0.5f, (src_pts[2].y + src_pts[3].y) * 0.5f);
    cv::Point2f v_axis(mid_bot.x - mid_top.x, mid_bot.y - mid_top.y);
    float v_len = std::sqrt(v_axis.x * v_axis.x + v_axis.y * v_axis.y);
    if (v_len > 1e-6f) { v_axis.x /= v_len; v_axis.y /= v_len; }

    // Horizontal axis: TL → TR direction
    cv::Point2f h_axis(src_pts[1].x - src_pts[0].x, src_pts[1].y - src_pts[0].y);
    float h_len = std::sqrt(h_axis.x * h_axis.x + h_axis.y * h_axis.y);
    if (h_len > 1e-6f) { h_axis.x /= h_len; h_axis.y /= h_len; }

    // Expand: move top points up (against v_axis), bottom points down (along v_axis)
    // and left points left (against h_axis), right points right (along h_axis)
    src_pts[0].x += -v_axis.x * pad_v - h_axis.x * pad_h;  // TL: up + left
    src_pts[0].y += -v_axis.y * pad_v - h_axis.y * pad_h;
    src_pts[1].x += -v_axis.x * pad_v + h_axis.x * pad_h;  // TR: up + right
    src_pts[1].y += -v_axis.y * pad_v + h_axis.y * pad_h;
    src_pts[2].x += v_axis.x * pad_v + h_axis.x * pad_h;   // BR: down + right
    src_pts[2].y += v_axis.y * pad_v + h_axis.y * pad_h;
    src_pts[3].x += v_axis.x * pad_v - h_axis.x * pad_h;   // BL: down + left
    src_pts[3].y += v_axis.y * pad_v - h_axis.y * pad_h;

    // Clip source points to image bounds
    for (int i = 0; i < 4; ++i) {
        src_pts[i].x = std::max(0.0f, std::min(static_cast<float>(img->width - 1), src_pts[i].x));
        src_pts[i].y = std::max(0.0f, std::min(static_cast<float>(img->height - 1), src_pts[i].y));
    }

    // Update output dimensions to include padding
    out_h = static_cast<int>(out_h + 2 * pad_v);
    out_w = static_cast<int>(out_w + 2 * pad_h);

    // Destination points (rectangle)
    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(out_w), 0),
        cv::Point2f(static_cast<float>(out_w), static_cast<float>(out_h)),
        cv::Point2f(0, static_cast<float>(out_h))
    };

    // Optimization: crop a bounding-box ROI from the full image first,
    // then run warpPerspective on the smaller ROI (much faster on RISC-V).
    float min_x = src_pts[0].x, max_x = src_pts[0].x;
    float min_y = src_pts[0].y, max_y = src_pts[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, src_pts[i].x);
        max_x = std::max(max_x, src_pts[i].x);
        min_y = std::min(min_y, src_pts[i].y);
        max_y = std::max(max_y, src_pts[i].y);
    }

    int roi_x = static_cast<int>(std::max(0.0f, std::floor(min_x)));
    int roi_y = static_cast<int>(std::max(0.0f, std::floor(min_y)));
    int roi_x2 = static_cast<int>(std::min(static_cast<float>(img->width), std::ceil(max_x)));
    int roi_y2 = static_cast<int>(std::min(static_cast<float>(img->height), std::ceil(max_y)));
    int roi_w = roi_x2 - roi_x;
    int roi_h = roi_y2 - roi_y;

    if (roi_w < 2 || roi_h < 2) return false;

    // Adjust source points to be relative to ROI
    cv::Point2f roi_pts[4];
    for (int i = 0; i < 4; ++i) {
        roi_pts[i] = cv::Point2f(src_pts[i].x - roi_x, src_pts[i].y - roi_y);
    }

    cv::Mat full_mat(img->height, img->width, CV_8UC3, img->data);
    cv::Mat roi_mat = full_mat(cv::Rect(roi_x, roi_y, roi_w, roi_h));

    cv::Mat M = cv::getPerspectiveTransform(roi_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(roi_mat, warped, M, cv::Size(out_w, out_h),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // If text is taller than wide, it's likely vertical - rotate
    if (static_cast<float>(out_h) > static_cast<float>(out_w) * 1.5f) {
        cv::Mat rotated;
        cv::rotate(warped, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        warped = rotated;
        std::swap(out_w, out_h);
    }

    // Enhance text region for better recognition
    EnhanceMode mode = enhance_mode_;

    // Adaptive: pick strategy based on crop color saturation
    if (mode == EnhanceMode::kAdaptive) {
        cv::Mat hsv;
        cv::cvtColor(warped, hsv, cv::COLOR_RGB2HSV);
        cv::Scalar mean_hsv = cv::mean(hsv);
        // High saturation (>30) = colored background → use grayscale
        mode = (mean_hsv[1] > 30.0) ? EnhanceMode::kGray : EnhanceMode::kClahe;
    }

    if (mode == EnhanceMode::kGray) {
        // Grayscale CLAHE: remove color noise, then enhance contrast
        cv::Mat gray;
        cv::cvtColor(warped, gray, cv::COLOR_RGB2GRAY);
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, gray);
        cv::cvtColor(gray, warped, cv::COLOR_GRAY2RGB);
    } else if (mode == EnhanceMode::kClahe) {
        // LAB CLAHE: enhance luminance while preserving color
        cv::Mat lab;
        cv::cvtColor(warped, lab, cv::COLOR_RGB2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, warped, cv::COLOR_Lab2RGB);
    }
    // kNone: skip enhancement

    // Unsharp mask (skip for kNone)
    if (mode != EnhanceMode::kNone) {
        cv::Mat blurred;
        cv::GaussianBlur(warped, blurred, cv::Size(0, 0), 1.5);
        cv::addWeighted(warped, 1.5, blurred, -0.5, 0, warped);
    }

    // Copy to output buffer
    size_t required = out_w * out_h * 3;
    if (output.size() < required) {
        output.resize(required);
    }
    std::memcpy(output.data(), warped.data, required);

    return true;
}

std::vector<OcrResult> OcrPipeline::process(ma_img_t* img, OcrTimings& timings) {
    std::vector<OcrResult> results;

    if (!initialized_ || !img) return results;

    // Detection
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<TextBox> boxes = detector_.detect(img);

    auto t1 = std::chrono::high_resolution_clock::now();
    timings.detection_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (boxes.empty()) {
        timings.recognition_ms = 0;
        timings.total_ms = timings.detection_ms;
        return results;
    }

    // Limit to top-N boxes by detection confidence to cap recognition time
    if (max_boxes_ > 0 && boxes.size() > max_boxes_) {
        std::partial_sort(boxes.begin(), boxes.begin() + max_boxes_, boxes.end(),
                          [](const TextBox& a, const TextBox& b) { return a.score > b.score; });
        boxes.resize(max_boxes_);
    }

    // Sort boxes for reading order
    sortBoxes(boxes);

    // Recognition for each detected text region
    auto t2 = std::chrono::high_resolution_clock::now();

    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        const auto& box = boxes[bi];

        if (!rec_available_) {
            // Detection-only mode: return boxes without text
            OcrResult ocr;
            ocr.box = box;
            ocr.det_confidence = box.score;
            ocr.rec_confidence = 0.0f;
            results.push_back(std::move(ocr));
            continue;
        }

        int crop_w = 0, crop_h = 0;
        if (!cropTextRegion(img, box, crop_buffer_, crop_w, crop_h)) {
            continue;
        }

        // Debug: save crop images for first 2 frames
        if (dbg_dump_frame_ < 2) {
            char path[128];
            snprintf(path, sizeof(path), "/tmp/ppocr_crop_f%d_b%zu.ppm", dbg_dump_frame_, bi);
            FILE* fp = fopen(path, "wb");
            if (fp) {
                fprintf(fp, "P6\n%d %d\n255\n", crop_w, crop_h);
                fwrite(crop_buffer_.data(), 1, crop_w * crop_h * 3, fp);
                fclose(fp);
            }
        }

        RecognitionResult rec = recognizer_.recognize(crop_buffer_.data(), crop_w, crop_h);

        // Low-confidence retry: try inverted image for dark/colored backgrounds
        static constexpr float kRetryThreshold = 0.45f;
        if (rec.confidence < kRetryThreshold && crop_w > 0 && crop_h > 0) {
            size_t crop_size = crop_w * crop_h * 3;
            std::vector<uint8_t> inv_buffer(crop_size);
            for (size_t j = 0; j < crop_size; ++j) {
                inv_buffer[j] = 255 - crop_buffer_[j];
            }
            RecognitionResult rec_inv = recognizer_.recognize(inv_buffer.data(), crop_w, crop_h);
            if (rec_inv.confidence > rec.confidence) {
                MA_LOGD(TAG, "Box[%zu] invert improved: '%s'(%.3f) -> '%s'(%.3f)",
                        bi, rec.text.c_str(), rec.confidence,
                        rec_inv.text.c_str(), rec_inv.confidence);
                rec = rec_inv;
            }
        }

        OcrResult ocr;
        ocr.box = box;
        ocr.det_confidence = box.score;

        // Always include the detection box; attach text only if confidence is sufficient
        static constexpr float kMinRecConfidence = 0.25f;
        if (!rec.text.empty() && rec.confidence >= kMinRecConfidence) {
            ocr.text = rec.text;
            ocr.rec_confidence = rec.confidence;
        } else {
            MA_LOGD(TAG, "Box[%zu] rec dropped: text='%s' conf=%.4f (threshold=%.2f)",
                    bi, rec.text.c_str(), rec.confidence, kMinRecConfidence);
            ocr.rec_confidence = rec.confidence;
        }
        results.push_back(std::move(ocr));
    }

    if (dbg_dump_frame_ < 2) dbg_dump_frame_++;

    // Text hysteresis: keep previous text unless new result wins for 2 consecutive frames
    // or has significantly higher confidence. This stabilizes flickering output.
    if (!prev_results_.empty() && results.size() == prev_results_.size()) {
        for (size_t i = 0; i < results.size(); ++i) {
            auto& cur = results[i];
            const auto& prev = prev_results_[i];
            if (!prev.text.empty() && !cur.text.empty() && cur.text != prev.text) {
                // New text differs from previous — only accept if significantly better
                if (cur.rec_confidence < prev.rec_confidence + 0.1f) {
                    cur.text = prev.text;
                    cur.rec_confidence = prev.rec_confidence;
                }
            } else if (!prev.text.empty() && cur.text.empty()) {
                // Previous had text, current doesn't — keep previous if it was decent
                if (prev.rec_confidence >= 0.3f) {
                    cur.text = prev.text;
                    cur.rec_confidence = prev.rec_confidence * 0.95f;  // slight decay
                }
            }
        }
    }
    prev_results_ = results;

    auto t3 = std::chrono::high_resolution_clock::now();
    timings.recognition_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();
    timings.total_ms = std::chrono::duration<float, std::milli>(t3 - t0).count();

    return results;
}

}  // namespace ppocr
