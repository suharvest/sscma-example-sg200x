#include "face_led_inspector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace microduck {

float calculateColorTemperature(float r, float g, float b) {
    if (r <= 1.0f && g <= 1.0f && b <= 1.0f) return 0.0f;

    // sRGB to CIE XYZ (assuming D65 white point reference)
    const float rf = r / 255.0f;
    const float gf = g / 255.0f;
    const float bf = b / 255.0f;

    const float X = 0.4124564f * rf + 0.3575761f * gf + 0.1804375f * bf;
    const float Y = 0.2126729f * rf + 0.7151522f * gf + 0.0721750f * bf;
    const float Z = 0.0193339f * rf + 0.1191920f * gf + 0.9503041f * bf;

    const float sum = X + Y + Z;
    if (sum < 1e-5f) return 0.0f;

    const float x = X / sum;
    const float y = Y / sum;

    // McCamy's approximation formula for Correlated Color Temperature (CCT)
    const float denom = 0.1858f - y;
    if (std::abs(denom) < 1e-5f) return 6500.0f;

    const float n = (x - 0.3320f) / denom;
    float cct = 449.0f * (n * n * n) + 3525.0f * (n * n) + 6823.3f * n + 5520.33f;

    if (cct < 1000.0f) cct = 1000.0f;
    if (cct > 25000.0f) cct = 25000.0f;
    return cct;
}

FaceLedFrameResult inspectFaceLed(const uint8_t* rgb_data, int width, int height, const AppConfig& cfg) {
    FaceLedFrameResult res;
    res.ok = false;
    res.metrics.lit_pixels = 0;
    res.metrics.color_temperature_k = 0.0f;
    res.metrics.mean_rgb = {0.0f, 0.0f, 0.0f};
    res.metrics.dark_leakage_lux = 0.0f;
    res.metrics.uniformity_ratio = 0.0f;

    if (!rgb_data || width <= 0 || height <= 0) {
        return res;
    }

    // Wrap RGB888 buffer
    cv::Mat full_img(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));

    // Determine ROI: if cfg.face_led_roi is configured with non-zero dimensions, use it;
    // otherwise operate on the full incoming frame.
    cv::Mat roi;
    if (cfg.face_led_roi.size() >= 4 && cfg.face_led_roi[2] > 0.0f && cfg.face_led_roi[3] > 0.0f) {
        int rx = 0, ry = 0, rw = width, rh = height;
        if (cfg.face_led_roi[2] <= 1.0f && cfg.face_led_roi[3] <= 1.0f) {
            rx = std::clamp(static_cast<int>(cfg.face_led_roi[0] * width), 0, width - 1);
            ry = std::clamp(static_cast<int>(cfg.face_led_roi[1] * height), 0, height - 1);
            rw = std::clamp(static_cast<int>(cfg.face_led_roi[2] * width), 1, width - rx);
            rh = std::clamp(static_cast<int>(cfg.face_led_roi[3] * height), 1, height - ry);
        } else {
            rx = std::clamp(static_cast<int>(cfg.face_led_roi[0]), 0, width - 1);
            ry = std::clamp(static_cast<int>(cfg.face_led_roi[1]), 0, height - 1);
            rw = std::clamp(static_cast<int>(cfg.face_led_roi[2]), 1, width - rx);
            rh = std::clamp(static_cast<int>(cfg.face_led_roi[3]), 1, height - ry);
        }
        roi = full_img(cv::Rect(rx, ry, rw, rh));
    } else {
        roi = full_img;
    }

    if (roi.empty()) {
        return res;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_RGB2GRAY);

    // Threshold candidate lit regions (intensity threshold = 45.0)
    constexpr double kLitThreshold = 45.0;
    constexpr double kMinPeakVal = 50.0;

    cv::Mat bin;
    cv::threshold(gray, bin, kLitThreshold, 255.0, cv::THRESH_BINARY);

    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);

    std::unordered_set<int> valid_labels;
    std::vector<float> spot_intensities;
    const int roi_area = roi.rows * roi.cols;
    const int max_allowed_area = static_cast<int>(roi_area * 0.35);

    cv::Mat valid_lit_mask = cv::Mat::zeros(roi.size(), CV_8UC1);

    for (int i = 1; i < n_labels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (area < 1 || area > max_allowed_area) {
            continue;
        }

        // Find peak and mean intensity inside component i
        double max_val = 0.0;
        double sum_val = 0.0;

        for (int r = top; r < top + h; ++r) {
            const int32_t* ptr_lbl = labels.ptr<int32_t>(r);
            const uint8_t* ptr_gray = gray.ptr<uint8_t>(r);

            for (int c = left; c < left + w; ++c) {
                if (ptr_lbl[c] == i) {
                    const uint8_t g = ptr_gray[c];
                    sum_val += g;
                    if (g > max_val) {
                        max_val = g;
                    }
                }
            }
        }

        if (max_val < kMinPeakVal) {
            continue;
        }

        valid_labels.insert(i);

        // Mark valid lit pixels on mask
        for (int r = top; r < top + h; ++r) {
            const int32_t* ptr_lbl = labels.ptr<int32_t>(r);
            uint8_t* ptr_mask = valid_lit_mask.ptr<uint8_t>(r);
            for (int c = left; c < left + w; ++c) {
                if (ptr_lbl[c] == i) {
                    ptr_mask[c] = 255;
                }
            }
        }

        const float mean_intensity = static_cast<float>(sum_val / area);
        spot_intensities.push_back(mean_intensity);
    }

    const int lit_pixels = static_cast<int>(spot_intensities.size());
    res.metrics.lit_pixels = lit_pixels;

    // 1. Calculate mean_rgb across all lit pixels
    if (lit_pixels > 0 && !valid_labels.empty()) {
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        int64_t total_lit_px = 0;

        for (int r = 0; r < roi.rows; ++r) {
            const uint8_t* ptr_rgb = roi.ptr<uint8_t>(r);
            const uint8_t* ptr_mask = valid_lit_mask.ptr<uint8_t>(r);

            for (int c = 0; c < roi.cols; ++c) {
                if (ptr_mask[c] > 0) {
                    sum_r += ptr_rgb[c * 3 + 0];
                    sum_g += ptr_rgb[c * 3 + 1];
                    sum_b += ptr_rgb[c * 3 + 2];
                    total_lit_px++;
                }
            }
        }

        if (total_lit_px > 0) {
            const float mean_r = static_cast<float>(sum_r / total_lit_px);
            const float mean_g = static_cast<float>(sum_g / total_lit_px);
            const float mean_b = static_cast<float>(sum_b / total_lit_px);
            res.metrics.mean_rgb = {mean_r, mean_g, mean_b};
            res.metrics.color_temperature_k = calculateColorTemperature(mean_r, mean_g, mean_b);
        }
    }

    // 2. Calculate dark_leakage_lux: max residual intensity outside lit blobs
    // (or across entire ROI if all LEDs are off)
    if (lit_pixels == 0) {
        double max_val = 0.0;
        cv::minMaxLoc(gray, nullptr, &max_val);
        res.metrics.dark_leakage_lux = static_cast<float>(max_val);
    } else {
        // Dilate lit mask by 3x3 to avoid optical edge flare right on spot boundaries
        cv::Mat dilated_lit;
        cv::dilate(valid_lit_mask, dilated_lit, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        cv::Mat bg_mask = ~dilated_lit;

        if (cv::countNonZero(bg_mask) > 0) {
            double max_val = 0.0;
            cv::minMaxLoc(gray, nullptr, &max_val, nullptr, nullptr, bg_mask);
            res.metrics.dark_leakage_lux = static_cast<float>(max_val);
        } else {
            res.metrics.dark_leakage_lux = 0.0f;
        }
    }

    // 3. Calculate uniformity_ratio: min/max intensity across lit blobs
    if (lit_pixels > 0 && !spot_intensities.empty()) {
        float min_intensity = spot_intensities.front();
        float max_intensity = spot_intensities.front();

        for (float val : spot_intensities) {
            if (val < min_intensity) min_intensity = val;
            if (val > max_intensity) max_intensity = val;
        }

        if (max_intensity > 0.001f) {
            res.metrics.uniformity_ratio = min_intensity / max_intensity;
        } else {
            res.metrics.uniformity_ratio = 0.0f;
        }
    } else {
        res.metrics.uniformity_ratio = 0.0f;
    }

    // Pass condition per specification: (lit_pixels >= 10 && dark_leakage_lux < 25.0)
    res.ok = (res.metrics.lit_pixels >= 10 && res.metrics.dark_leakage_lux < 25.0f);
    return res;
}

}  // namespace microduck
