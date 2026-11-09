#include "face_led_inspector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

namespace {

// Constants: identical to tools/sim_qa/face_led.py
constexpr int kRingHLo = 6, kRingHHi = 40, kRingSMin = 60, kRingVMin = 70;
constexpr float kRingLensRatio = 0.9f;  // lens (hole) luma < this x ring band luma
constexpr float kSpotR = 0.28f;
constexpr float kSearchT = 0.30f;
constexpr int kSearchScaleSteps = 3;
constexpr float kSearchScaleStep = 0.05f;
constexpr int kSearchRot = 15;
constexpr float kMaxResidual = 0.35f;
constexpr int kLitRise = 20, kLitLead = 40, kLitMin = 120, kRedRatioPct = 48;
constexpr int kWhiteMin = 160, kWhiteSpreadPct = 30;  // all channels >= 160 and spread <= 30 % of max

inline int ri(double x) { return x >= 0 ? static_cast<int>(std::floor(x + 0.5)) : -static_cast<int>(std::floor(-x + 0.5)); }

bool placeAllows(const std::string& place, const std::string& colour) {
    if (place == "rgb") return colour == "red" || colour == "green" || colour == "blue" || colour == "white";
    for (const char* n : kFaceLedNames) {
        std::string s(n);
        const size_t a = s.find(':'), b = s.find(':', a + 1);
        if (s.substr(a + 1, b - a - 1) == place && s.substr(b + 1) == colour) return true;
    }
    return false;
}

[[maybe_unused]] bool warm(const cv::Vec3b& c) {
    const int r = c[0], g = c[1], b = c[2];
    const int mx = std::max({r, g, b}), mn = std::min({r, g, b});
    (void)g; (void)b;
    return mx == r && mx >= 70 && (mx - mn) * 10 >= 3 * mx;
}

float sampleF(const cv::Mat& m, double x, double y) {
    const int xi = ri(x), yi = ri(y);
    if (xi < 0 || yi < 0 || xi >= m.cols || yi >= m.rows) return -1e6f;
    return m.at<float>(yi, xi);
}

}  // namespace

bool findLensRing(const uint8_t* rgb_data, int width, int height, float& out_cx, float& out_cy, float& out_r) {
    const cv::Mat img(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));
    cv::Mat hsv, gray;
    cv::cvtColor(img, hsv, cv::COLOR_RGB2HSV);
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    cv::Mat mask(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* p = hsv.ptr<cv::Vec3b>(y);
        uint8_t* m = mask.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            m[x] = (p[x][0] >= kRingHLo && p[x][0] <= kRingHHi && p[x][1] >= kRingSMin && p[x][2] >= kRingVMin) ? 1 : 0;
        }
    }
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::Mat pad;
    cv::copyMakeBorder(1 - mask, pad, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(1));
    cv::Mat ffmask = cv::Mat::zeros(height + 4, width + 4, CV_8UC1);
    cv::floodFill(pad, ffmask, cv::Point(0, 0), cv::Scalar(2));
    cv::Mat holes = (pad(cv::Rect(1, 1, width, height)) == 1) / 255;
    cv::Mat lab, st, cen;
    const int n = cv::connectedComponentsWithStats(holes, lab, st, cen, 4, CV_32S);
    int best_area = 0;
    bool found = false;
    for (int i = 1; i < n; ++i) {
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH), hh = st.at<int>(i, cv::CC_STAT_HEIGHT);
        const int area = st.at<int>(i, cv::CC_STAT_AREA);
        const double ar = static_cast<double>(w) / hh;
        if (area < 4 || ar < 0.5 || ar > 2.0 || area <= best_area) continue;
        const double cx = cen.at<double>(i, 0), cy = cen.at<double>(i, 1);
        const double r_in = std::sqrt(area / M_PI);
        const double reach = 2.5 * r_in + 2;
        std::vector<double> dists;
        int hits = 0;
        for (int k = 0; k < 16; ++k) {
            const double a = 2 * M_PI * k / 16;
            for (double t = 0.5; t < reach; t += 0.5) {
                const int xx = ri(cx + t * std::cos(a)), yy = ri(cy + t * std::sin(a));
                if (xx < 0 || yy < 0 || xx >= width || yy >= height) break;
                if (mask.at<uint8_t>(yy, xx)) {
                    hits++;
                    double t2 = t;
                    while (t2 < reach * 1.6) {
                        const int x2 = ri(cx + t2 * std::cos(a)), y2 = ri(cy + t2 * std::sin(a));
                        if (x2 < 0 || y2 < 0 || x2 >= width || y2 >= height || !mask.at<uint8_t>(y2, x2)) break;
                        t2 += 0.5;
                    }
                    dists.push_back(0.5 * (t + t2 - 0.5));
                    break;
                }
            }
        }
        if (hits < 13) continue;
        // the lens inside the ring is dark: hole mean luma < kRingLensRatio x ring band mean luma
        {
            const cv::Rect hb(st.at<int>(i, cv::CC_STAT_LEFT), st.at<int>(i, cv::CC_STAT_TOP), w, hh);
            const cv::Scalar hole_mean = cv::mean(gray(hb), lab(hb) == i);
            const int yy0 = std::max(0, static_cast<int>(cy - reach)), yy1 = std::min(height, static_cast<int>(cy + reach) + 1);
            const int xx0 = std::max(0, static_cast<int>(cx - reach)), xx1 = std::min(width, static_cast<int>(cx + reach) + 1);
            const cv::Rect box(xx0, yy0, xx1 - xx0, yy1 - yy0);
            const cv::Mat band = mask(box) > 0;
            if (cv::countNonZero(band) == 0) continue;
            const cv::Scalar band_mean = cv::mean(gray(box), band);
            if (hole_mean[0] >= kRingLensRatio * band_mean[0]) continue;
        }
        std::sort(dists.begin(), dists.end());
        const size_t m = dists.size();
        const double med = (m % 2) ? dists[m / 2] : 0.5 * (dists[m / 2 - 1] + dists[m / 2]);
        out_cx = static_cast<float>(cx);
        out_cy = static_cast<float>(cy);
        out_r = static_cast<float>(med);
        best_area = area;
        found = true;
    }
    return found;
}

FaceLedSlotFrame inspectFaceLedSlots(const uint8_t* rgb_data, int width, int height, const AppConfig& cfg) {
    FaceLedSlotFrame out;
    if (!rgb_data || width <= 0 || height <= 0 || cfg.face_led_layout.size() < 3) return out;
    float cx, cy, r;
    if (!findLensRing(rgb_data, width, height, cx, cy, r)) {
        out.reason = "RING_NOT_FOUND";
        return out;
    }
    out.has_ring = true;
    out.ring_x = cx; out.ring_y = cy; out.ring_r = r;
    const cv::Mat img(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));
    // canonical place order (rgb, cam, gr2, gb, gr1, green2, green1) as in face_led.py
    static const char* kPlaces[] = {"rgb", "cam", "gr2", "gb", "gr1", "green2", "green1"};
    std::vector<std::string> names;
    std::vector<cv::Point2d> tpl;
    for (const char* pl : kPlaces) {
        for (const auto& s : cfg.face_led_layout) {
            if (s.place == pl) { names.push_back(s.place); tpl.emplace_back(s.x, s.y); }
        }
    }
    cv::Mat gray, fs, fl, resp;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(gray, CV_32F);
    const double spot_px = kSpotR * r;
    cv::GaussianBlur(gray, fs, cv::Size(0, 0), std::max(0.6, 0.5 * spot_px));
    cv::GaussianBlur(gray, fl, cv::Size(0, 0), 2.0 * spot_px);
    resp = cv::abs(fs - fl);
    cv::dilate(resp, resp, cv::Mat::ones(3, 3, CV_8U));

    const int nt = static_cast<int>(std::ceil(kSearchT * r));
    auto score = [&](double rs, int deg, double ix, double iy) {
        const double a = deg * M_PI / 180.0, c = std::cos(a), sn = std::sin(a);
        double sc = 0;
        for (const auto& t : tpl) sc += sampleF(resp, cx + rs * (c * t.x - sn * t.y) + ix, cy + rs * (sn * t.x + c * t.y) + iy);
        return sc;
    };
    // coarse: scale 1 +- 0.05 k, rotation 3 deg steps, translation 2 px steps
    double best = -1.0, bdx = 0, bdy = 0, brs = r;
    int bdeg = 0;
    for (int k = -kSearchScaleSteps; k <= kSearchScaleSteps; ++k) {
        const double rs = r * (1.0 + kSearchScaleStep * k);
        for (int deg = -kSearchRot; deg <= kSearchRot; deg += 3)
            for (int iy = -nt; iy <= nt; iy += 2)
                for (int ix = -nt; ix <= nt; ix += 2) {
                    const double sc = score(rs, deg, ix, iy);
                    if (sc > best) { best = sc; bdeg = deg; bdx = ix; bdy = iy; brs = rs; }
                }
    }
    // fine: scale x(1 +- 0.025), rotation +-2 deg, translation +-1 px around the coarse best
    {
        const int d0 = bdeg;
        const double x0 = bdx, y0 = bdy, r0 = brs;
        for (int j = -1; j <= 1; ++j) {
            const double rs = r0 * (1.0 + 0.5 * kSearchScaleStep * j);
            for (int deg = d0 - 2; deg <= d0 + 2; ++deg)
                for (int jy = -1; jy <= 1; ++jy)
                    for (int jx = -1; jx <= 1; ++jx) {
                        const double sc = score(rs, deg, x0 + jx, y0 + jy);
                        if (sc > best) { best = sc; bdeg = deg; bdx = x0 + jx; bdy = y0 + jy; brs = rs; }
                    }
        }
    }
    const double a0 = bdeg * M_PI / 180.0, c0 = std::cos(a0), s0 = std::sin(a0);
    const int rad = std::max(1, ri(kSpotR * r));
    std::vector<cv::Point2d> src, dst;
    std::vector<double> wts;
    for (size_t i = 0; i < tpl.size(); ++i) {
        const double x = cx + bdx + brs * (c0 * tpl[i].x - s0 * tpl[i].y);
        const double y = cy + bdy + brs * (s0 * tpl[i].x + c0 * tpl[i].y);
        const int xi = ri(x), yi = ri(y);
        const int y0 = std::max(0, yi - rad), y1 = std::min(resp.rows, yi + rad + 1);
        const int x0 = std::max(0, xi - rad), x1 = std::min(resp.cols, xi + rad + 1);
        float bv = -1e9f;
        int bx = xi, by = yi;
        for (int yy = y0; yy < y1; ++yy) {
            for (int xx = x0; xx < x1; ++xx) {
                const float v = resp.at<float>(yy, xx);
                if (v > bv) { bv = v; bx = xx; by = yy; }
            }
        }
        src.push_back(tpl[i]);
        dst.emplace_back(bx, by);
        wts.push_back(1.0);
    }
    src.emplace_back(0.0, 0.0);
    dst.emplace_back(cx, cy);
    wts.push_back(3.0);
    // weighted Umeyama similarity
    double wsum = 0, msx = 0, msy = 0, mdx = 0, mdy = 0;
    for (size_t i = 0; i < src.size(); ++i) wsum += wts[i];
    for (size_t i = 0; i < src.size(); ++i) {
        const double w = wts[i] / wsum;
        msx += w * src[i].x; msy += w * src[i].y; mdx += w * dst[i].x; mdy += w * dst[i].y;
    }
    double sxx = 0, sxy = 0, var = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double w = wts[i] / wsum;
        const double ax = src[i].x - msx, ay = src[i].y - msy, bx = dst[i].x - mdx, by = dst[i].y - mdy;
        sxx += w * (ax * bx + ay * by);
        sxy += w * (ax * by - ay * bx);
        var += w * (ax * ax + ay * ay);
    }
    const double ang = std::atan2(sxy, sxx);
    const double s = var > 0 ? std::hypot(sxx, sxy) / var : 1.0;
    const double c = std::cos(ang), sn = std::sin(ang);
    const double tx = mdx - s * (c * msx - sn * msy), ty = mdy - s * (sn * msx + c * msy);
    double se = 0;
    for (size_t i = 0; i + 1 < src.size(); ++i) {
        const double fx = s * (c * src[i].x - sn * src[i].y) + tx, fy = s * (sn * src[i].x + c * src[i].y) + ty;
        se += (fx - dst[i].x) * (fx - dst[i].x) + (fy - dst[i].y) * (fy - dst[i].y);
        FaceLedSlotObs o;
        o.place = names[i];
        o.x = static_cast<float>(fx);
        o.y = static_cast<float>(fy);
        out.slots.push_back(o);
    }
    out.scale_px = static_cast<float>(s);
    out.rotation_deg = static_cast<float>(ang * 180.0 / M_PI);
    out.residual = static_cast<float>(std::sqrt(se / std::max<size_t>(1, src.size() - 1)) / s);
    if (out.residual > kMaxResidual) {
        out.reason = "RESIDUAL_TOO_HIGH";
        out.slots.clear();
        return out;
    }
    out.valid = true;

    // colour per place
    const int rb = std::max(2, ri(0.75 * s));
    for (auto& o : out.slots) {
        // background = median of the 8 least saturated samples (the panel the LED sits on, not the
        // lens ring / red strip next to some places nor dark off LEDs)
        std::vector<cv::Vec3b> ring;
        for (int k = 0; k < 16; ++k) {
            const double a = 2 * M_PI * k / 16;
            const int bx = ri(o.x + rb * std::cos(a)), by = ri(o.y + rb * std::sin(a));
            if (bx >= 0 && by >= 0 && bx < width && by < height) ring.push_back(img.at<cv::Vec3b>(by, bx));
        }
        if (ring.empty()) continue;
        std::vector<double> sat(ring.size());
        for (size_t i = 0; i < ring.size(); ++i) {
            const int mx = std::max({ring[i][0], ring[i][1], ring[i][2]}), mn = std::min({ring[i][0], ring[i][1], ring[i][2]});
            sat[i] = mx > 0 ? static_cast<double>(mx - mn) / mx : 0.0;
        }
        std::vector<size_t> order(ring.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sat[a] < sat[b]; });
        order.resize(std::min<size_t>(8, order.size()));
        std::vector<cv::Vec3b> use;
        for (size_t i : order) use.push_back(ring[i]);
        double bg[3];
        for (int ch = 0; ch < 3; ++ch) {
            std::vector<int> v;
            for (const auto& p : use) v.push_back(p[ch]);
            std::sort(v.begin(), v.end());
            const size_t m = v.size();
            bg[ch] = (m % 2) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
        }
        const int xi = ri(o.x), yi = ri(o.y);
        double best_sc = -1e9;
        std::string best_col;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const int x = xi + ox, y = yi + oy;
                if (x < 0 || y < 0 || x >= width || y >= height) continue;
                const cv::Vec3b pix = img.at<cv::Vec3b>(y, x);
                const double d[3] = {pix[0] - bg[0], pix[1] - bg[1], pix[2] - bg[2]};
                const double dmin = std::min({d[0], d[1], d[2]}), dmax = std::max({d[0], d[1], d[2]});
                const int pmin = std::min({pix[0], pix[1], pix[2]}), pmax = std::max({pix[0], pix[1], pix[2]});
                std::string col;
                if ((pmin >= kWhiteMin && (pmax - pmin) * 100 <= kWhiteSpreadPct * pmax) || (dmin >= 35 && (dmax - dmin) * 10 <= 6 * dmax)) {
                    col = "white";
                } else {
                    int cc = 0;
                    for (int ch = 1; ch < 3; ++ch) if (d[ch] > d[cc]) cc = ch;  // first max, as np.argmax
                    double others = -1e9;
                    for (int ch = 0; ch < 3; ++ch) if (ch != cc) others = std::max(others, d[ch]);
                    const bool red_ok = cc != 0 || std::max(pix[1], pix[2]) * 100 <= kRedRatioPct * pix[0];
                    if (pix[cc] >= kLitMin && d[cc] >= kLitRise && d[cc] - others >= kLitLead && red_ok) {
                        static const char* names3[] = {"red", "green", "blue"};
                        col = names3[cc];
                    }
                }
                if (!col.empty() && placeAllows(o.place, col) && dmax > best_sc) {
                    best_sc = dmax;
                    best_col = col;
                }
            }
        }
        o.colour = best_col;
    }
    return out;
}

}  // namespace microduck
