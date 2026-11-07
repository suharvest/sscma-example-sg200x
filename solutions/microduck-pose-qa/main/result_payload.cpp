#include "result_payload.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace microduck {

using json = nlohmann::json;

static const char* kMicroduckJointNames[] = {
    "mouth_tip",       // 0
    "head_roll",       // 1
    "head_yaw",        // 2
    "head_pitch",      // 3
    "neck_pitch",      // 4
    "left_hip_yaw",    // 5
    "left_hip_roll",   // 6
    "left_hip_pitch",  // 7
    "left_knee",       // 8
    "left_ankle",      // 9
    "left_foot",       // 10
    "right_hip_yaw",   // 11
    "right_hip_roll",  // 12
    "right_hip_pitch", // 13
    "right_knee",      // 14
    "right_ankle",     // 15
    "right_foot"       // 16
};

static void letterbox_to_display_pt(float& x, float& y,
                                    int inference_w, int inference_h,
                                    int display_w, int display_h) {
    if (inference_w <= 0 || inference_h <= 0 || display_w <= 0 || display_h <= 0) return;
    float content_w = static_cast<float>(inference_w);
    float content_h = static_cast<float>(inference_h);
    float x_off = 0.0f, y_off = 0.0f;
    const long long disp_vs_inf =
        static_cast<long long>(display_w) * inference_h - static_cast<long long>(display_h) * inference_w;
    if (disp_vs_inf > 0) {
        content_h = static_cast<float>(inference_w) * display_h / display_w;
        y_off     = (inference_h - content_h) * 0.5f;
    } else if (disp_vs_inf < 0) {
        content_w = static_cast<float>(inference_h) * display_w / display_h;
        x_off     = (inference_w - content_w) * 0.5f;
    }
    const float sx = static_cast<float>(display_w) / content_w;
    const float sy = static_cast<float>(display_h) / content_h;
    x = (x - x_off) * sx;
    y = (y - y_off) * sy;
}

json ActionReport::toJson() const {
    json j;
    j["run_id"] = run_id;
    j["action_id"] = action_id;
    j["config_version"] = config_version;
    j["overall_passed"] = overall_passed;
    j["status_code"] = status_code;
    if (has_window_frames) {
        j["window_start_frame"] = window_start_frame;
        j["window_end_frame"] = window_end_frame;
    } else {
        j["window_start_frame"] = nullptr;
        j["window_end_frame"] = nullptr;
    }
    j["total_frames"] = total_frames;
    j["valid_frames"] = valid_frames;
    j["detected_frames"] = detected_frames;
    j["occluded_frames"] = occluded_frames;

    json met;
    if (action_id == "face_led") {
        met["lit_pixels"] = optical_metrics.lit_pixels;
        met["color_temperature_k"] = std::round(optical_metrics.color_temperature_k * 10.0f) / 10.0f;
        json rgb = json::array();
        for (float c : optical_metrics.mean_rgb) {
            rgb.push_back(std::round(c * 10.0f) / 10.0f);
        }
        met["mean_rgb"] = std::move(rgb);
        met["dark_leakage_lux"] = std::round(optical_metrics.dark_leakage_lux * 10.0f) / 10.0f;
        met["uniformity_ratio"] = std::round(optical_metrics.uniformity_ratio * 1000.0f) / 1000.0f;
        met["detected_leds"] = optical_metrics.lit_pixels;
        met["dead_pixels_count"] = std::max(0, 12 - optical_metrics.lit_pixels);
    } else {
        met["left_knee_height_norm"] = std::round(metrics.left_knee_height_norm * 10000.0f) / 10000.0f;
        met["right_knee_height_norm"] = std::round(metrics.right_knee_height_norm * 10000.0f) / 10000.0f;
        met["knee_height_diff_norm"] = std::round(metrics.knee_height_diff_norm * 10000.0f) / 10000.0f;
        met["estimated_spline_offset_deg"] = std::round(metrics.estimated_spline_offset_deg * 10.0f) / 10.0f;
        if (metrics.has_spine_tilt) {
            met["spine_tilt_deg"] = std::round(metrics.spine_tilt_deg * 100.0f) / 100.0f;
        } else {
            met["spine_tilt_deg"] = nullptr;
        }
    }
    j["metrics"] = met;
    if (action_id == "face_led") {
        j["optical_metrics"] = met;
    }
    j["detected_defects"] = detected_defects;
    return j;
}

FrameQAResult evaluateFrameFromNormKpts(const std::vector<KeypointRaw>& kpts_raw,
                                       bool robot_detected,
                                       int duck_count,
                                       const AppConfig& config) {
    FrameQAResult res;
    res.kpts_raw_17 = kpts_raw;

    // 1. NO_ROBOT check
    if (!robot_detected || duck_count == 0 || kpts_raw.empty()) {
        res.qa_status = "NO_ROBOT";
        res.qa_passed = false;
        res.visible_joints = 0;
        return res;
    }

    const float eval_thresh = (config.keypoint_confidence > 0.0f) ? config.keypoint_confidence : KPT_CONF_EVAL_THRESHOLD;

    // Count visible joints with raw confidence >= eval_thresh
    int vis = 0;
    for (const auto& kp : kpts_raw) {
        if (kp.conf >= eval_thresh) {
            vis++;
        }
    }
    res.visible_joints = vis;

    // 2. OCCLUDED check (< 10 visible joints)
    if (vis < MIN_VISIBLE_JOINTS_PER_FRAME) {
        res.qa_status = "OCCLUDED";
        res.qa_passed = false;
        return res;
    }

    if (kpts_raw.size() < 17) {
        res.qa_status = "OCCLUDED";
        res.qa_passed = false;
        return res;
    }

    // Trunk spine tilt calculation: neck (4), left_hip (5), right_hip (11)
    const float c4 = kpts_raw[4].conf;
    const float c5 = kpts_raw[5].conf;
    const float c11 = kpts_raw[11].conf;
    if (c4 >= eval_thresh && c5 >= eval_thresh && c11 >= eval_thresh) {
        const float neck_x = kpts_raw[4].x_norm;
        const float neck_y = kpts_raw[4].y_norm;
        const float mid_hip_x = (kpts_raw[5].x_norm + kpts_raw[11].x_norm) * 0.5f;
        const float mid_hip_y = (kpts_raw[5].y_norm + kpts_raw[11].y_norm) * 0.5f;
        const float dx_px = (neck_x - mid_hip_x) * 1280.0f;
        const float dy_px = (neck_y - mid_hip_y) * 720.0f;
        const float spine_tilt = std::abs(static_cast<float>(std::atan2(dx_px, -dy_px) * 180.0 / M_PI));
        res.metrics.has_spine_tilt = true;
        // Keep raw float precision throughout evaluation
        res.metrics.spine_tilt_deg = spine_tilt;
    }

    // Essential standing keypoints visibility & Knee asymmetry check
    // Left knee = 8, Right knee = 14
    const float l_knee_conf = kpts_raw[8].conf;
    const float r_knee_conf = kpts_raw[14].conf;
    if (l_knee_conf < eval_thresh || r_knee_conf < eval_thresh) {
        res.qa_status = "OCCLUDED";
        res.qa_passed = false;
        return res;
    }

    const float l_knee_y = kpts_raw[8].y_norm;
    const float r_knee_y = kpts_raw[14].y_norm;
    const float knee_height_diff = std::abs(l_knee_y - r_knee_y);
    res.metrics.has_knee_diff = true;
    // Keep raw float precision throughout evaluation
    res.metrics.knee_height_diff_norm = knee_height_diff;

    // Frame-level thigh spline calculation
    const float c7 = kpts_raw[7].conf;
    const float c8 = kpts_raw[8].conf;
    const float c13 = kpts_raw[13].conf;
    const float c14 = kpts_raw[14].conf;
    if (c7 >= eval_thresh && c8 >= eval_thresh &&
        c13 >= eval_thresh && c14 >= eval_thresh) {
        const float x7 = kpts_raw[7].x_norm, y7 = kpts_raw[7].y_norm;
        const float x8 = kpts_raw[8].x_norm, y8 = kpts_raw[8].y_norm;
        const float x13 = kpts_raw[13].x_norm, y13 = kpts_raw[13].y_norm;
        const float x14 = kpts_raw[14].x_norm, y14 = kpts_raw[14].y_norm;

        const float theta_L = static_cast<float>(std::atan2((x8 - x7) * 1280.0f, (y8 - y7) * 720.0f) * 180.0 / M_PI);
        const float theta_R = static_cast<float>(std::atan2((x14 - x13) * 1280.0f, (y14 - y13) * 720.0f) * 180.0 / M_PI);
        const float disparity = std::abs(static_cast<float>(std::remainder(theta_L - theta_R, 360.0)));

#ifndef HOST_CLI
        const float dx_l_px = (x8 - x7) * 1280.0f;
        const float dy_l_px = (y8 - y7) * 720.0f;
        const float dev_l = (std::abs(dy_l_px) < 4.0f || std::abs(dx_l_px) < 2.0f)
            ? 0.0f
            : std::abs(static_cast<float>(std::remainder(theta_L - config.nominal_stand_theta_l_deg, 360.0)));
        const float dx_r_px = (x14 - x13) * 1280.0f;
        const float dy_r_px = (y14 - y13) * 720.0f;
        const float dev_r = (std::abs(dy_r_px) < 4.0f || std::abs(dx_r_px) < 2.0f)
            ? 0.0f
            : std::abs(static_cast<float>(std::remainder(theta_R - config.nominal_stand_theta_r_deg, 360.0)));
#else
        const float dev_l = (x8 == x7 && y8 == y7) ? 0.0f : std::abs(static_cast<float>(std::remainder(theta_L - config.nominal_stand_theta_l_deg, 360.0)));
        const float dev_r = (x14 == x13 && y14 == y13) ? 0.0f : std::abs(static_cast<float>(std::remainder(theta_R - config.nominal_stand_theta_r_deg, 360.0)));
#endif
        const float max_dev = std::max(dev_l, dev_r);
        const float disp_dev = std::abs(static_cast<float>(std::remainder(disparity - config.nominal_stand_disparity_deg, 360.0)));

        float frame_spline = 0.0f;
        if (max_dev >= config.gear_tooth_tolerance_deg) {
            frame_spline = max_dev;
        } else if (knee_height_diff <= config.bilateral_height_tolerance_norm) {
            frame_spline = disp_dev;
        } else {
            frame_spline = max_dev;
        }
        res.metrics.has_spline_offset = true;
        // Keep raw float precision throughout evaluation
        res.metrics.spline_offset_deg = frame_spline;
    }

    if (knee_height_diff > config.bilateral_height_tolerance_norm) {
        res.qa_status = "KNEE_ASYMMETRY";
        res.qa_passed = false;
        return res;
    }

    res.qa_status = "PASS";
    res.qa_passed = true;
    return res;
}

#ifndef HOST_CLI
FrameQAResult evaluateQA(const PayloadContext& ctx, const Pose& pose) {
    if (!ctx.robot_detected || pose.empty()) {
        FrameQAResult empty_res;
        empty_res.qa_status = "NO_ROBOT";
        empty_res.qa_passed = false;
        empty_res.visible_joints = 0;
        return empty_res;
    }

    std::vector<KeypointRaw> kpts_raw;
    kpts_raw.reserve(17);

    const auto& roi = ctx.config.screen_roi;
    const float roi_x0 = (roi.size() >= 4) ? roi[0] : 0.0f;
    const float roi_y0 = (roi.size() >= 4) ? roi[1] : 0.0f;
    const float roi_w  = (roi.size() >= 4 && roi[2] > 0.0f) ? roi[2] : static_cast<float>(ctx.stream_w);
    const float roi_h  = (roi.size() >= 4 && roi[3] > 0.0f) ? roi[3] : static_cast<float>(ctx.stream_h);

    constexpr int kJointCount = static_cast<int>(Joint::Count);
    for (int i = 0; i < kJointCount; ++i) {
        Point2f pt = pose.at(i);
        float x_disp = pt.x;
        float y_disp = pt.y;

        // Undo letterbox mapping from inference frame to camera display frame
        letterbox_to_display_pt(x_disp, y_disp, ctx.infer_w, ctx.infer_h, ctx.stream_w, ctx.stream_h);
        x_disp = std::clamp(x_disp, 0.0f, static_cast<float>(ctx.stream_w));
        y_disp = std::clamp(y_disp, 0.0f, static_cast<float>(ctx.stream_h));

        // Map from camera display coords to screen ROI normalized coordinates [0, 1]
        float x_norm = std::clamp((x_disp - roi_x0) / roi_w, 0.0f, 1.0f);
        float y_norm = std::clamp((y_disp - roi_y0) / roi_h, 0.0f, 1.0f);
        float conf = pose.confidence(i);

        KeypointRaw raw;
        raw.id = i;
        raw.name = (i >= 0 && i < 17) ? kMicroduckJointNames[i] : "";
        // Keep raw float precision (no premature rounding)
        raw.x_norm = x_norm;
        raw.y_norm = y_norm;
        raw.conf = conf;
        kpts_raw.push_back(std::move(raw));
    }

    return evaluateFrameFromNormKpts(kpts_raw, ctx.robot_detected, ctx.duck_count, ctx.config);
}
#endif

ActionReport ActionStateMachine::aggregateWindow(const std::string& action_id,
                                                 const std::string& run_id,
                                                 const AppConfig& config,
                                                 const std::vector<CollectedFrame>& frames,
                                                 bool timed_out) {
    ActionReport rep;
    rep.action_id = action_id;
    rep.run_id = run_id;
    rep.config_version = config.config_version;
    rep.total_frames = static_cast<int>(frames.size());

    if (!frames.empty()) {
        rep.has_window_frames = true;
        rep.window_start_frame = frames.front().frame_id;
        rep.window_end_frame = frames.back().frame_id;
    } else {
        rep.has_window_frames = false;
        rep.window_start_frame = 0;
        rep.window_end_frame = 0;
    }

    if (action_id == "face_led") {
        if (timed_out && frames.empty()) {
            rep.status_code = "FAIL_MOTION_TIMEOUT";
            rep.overall_passed = false;
            rep.detected_defects.push_back({
                {"type", "MOTION_TIMEOUT"},
                {"collected_frames", rep.total_frames},
                {"required_frames", SETTLED_WINDOW_FRAMES}
            });
            return rep;
        }

        std::vector<int> lit_counts;
        std::vector<float> ccts;
        std::vector<float> r_vals, g_vals, b_vals;
        std::vector<float> leakages;
        std::vector<float> uniformities;
        int valid_optical_frames = 0;

        for (const auto& item : frames) {
            const auto& m = item.optical.metrics;
            lit_counts.push_back(m.lit_pixels);
            leakages.push_back(m.dark_leakage_lux);
            uniformities.push_back(m.uniformity_ratio);
            if (m.lit_pixels > 0) {
                valid_optical_frames++;
                ccts.push_back(m.color_temperature_k);
                if (m.mean_rgb.size() >= 3) {
                    r_vals.push_back(m.mean_rgb[0]);
                    g_vals.push_back(m.mean_rgb[1]);
                    b_vals.push_back(m.mean_rgb[2]);
                }
            }
        }

        rep.valid_frames = valid_optical_frames;
        rep.detected_frames = valid_optical_frames;

        // Aggregate lit_pixels (median across frames)
        int agg_lit = 0;
        if (!lit_counts.empty()) {
            std::sort(lit_counts.begin(), lit_counts.end());
            agg_lit = lit_counts[lit_counts.size() / 2];
        }

        // Aggregate CCT (mean of frames with lit spots)
        float agg_cct = 0.0f;
        if (!ccts.empty()) {
            agg_cct = std::accumulate(ccts.begin(), ccts.end(), 0.0f) / ccts.size();
        }

        // Aggregate mean_rgb
        std::vector<float> agg_rgb = {0.0f, 0.0f, 0.0f};
        if (!r_vals.empty()) {
            agg_rgb[0] = std::accumulate(r_vals.begin(), r_vals.end(), 0.0f) / r_vals.size();
            agg_rgb[1] = std::accumulate(g_vals.begin(), g_vals.end(), 0.0f) / g_vals.size();
            agg_rgb[2] = std::accumulate(b_vals.begin(), b_vals.end(), 0.0f) / b_vals.size();
        }

        // Aggregate dark leakage (max across frames)
        float agg_leak = 0.0f;
        if (!leakages.empty()) {
            agg_leak = *std::max_element(leakages.begin(), leakages.end());
        }

        // Aggregate uniformity (mean)
        float agg_unif = 0.0f;
        if (!uniformities.empty()) {
            agg_unif = std::accumulate(uniformities.begin(), uniformities.end(), 0.0f) / uniformities.size();
        }

        rep.optical_metrics.lit_pixels = agg_lit;
        rep.optical_metrics.color_temperature_k = agg_cct;
        rep.optical_metrics.mean_rgb = std::move(agg_rgb);
        rep.optical_metrics.dark_leakage_lux = agg_leak;
        rep.optical_metrics.uniformity_ratio = agg_unif;

        if (timed_out && frames.size() < static_cast<size_t>(MIN_VALID_FRAMES)) {
            rep.status_code = "FAIL_MOTION_TIMEOUT";
            rep.overall_passed = false;
            rep.detected_defects.push_back({
                {"type", "MOTION_TIMEOUT"},
                {"collected_frames", rep.total_frames},
                {"required_frames", SETTLED_WINDOW_FRAMES}
            });
            return rep;
        }

        rep.overall_passed = (agg_lit >= 10 && agg_leak < 25.0f);
        if (rep.overall_passed) {
            rep.status_code = "PASS";
        } else {
            if (agg_lit < 10) {
                rep.status_code = "FAIL_DEAD_PIXELS";
                rep.detected_defects.push_back({
                    {"type", "DEAD_PIXELS"},
                    {"lit_pixels", agg_lit},
                    {"required_min", 10}
                });
            }
            if (agg_leak >= 25.0f) {
                if (rep.status_code == "PASS") {
                    rep.status_code = "FAIL_DARK_LEAKAGE";
                }
                rep.detected_defects.push_back({
                    {"type", "DARK_LEAKAGE"},
                    {"measured_lux", std::round(agg_leak * 10.0f) / 10.0f},
                    {"threshold_lux", 25.0}
                });
            }
        }
        return rep;
    }

    if (action_id != "stand") {
        rep.overall_passed = false;
        rep.status_code = "FAIL_UNSUPPORTED_ACTION";
        rep.detected_defects.push_back({
            {"type", "UNSUPPORTED_ACTION"},
            {"action_id", action_id}
        });
        return rep;
    }

    int detected_count = 0;
    int valid_frames = 0;
    int occluded_count = 0;
    int knee_asymmetry_count = 0;

    std::vector<float> l_knee_heights;
    std::vector<float> r_knee_heights;
    std::vector<float> knee_diffs;
    std::vector<float> spline_offsets;
    std::vector<std::string> faulted_joints;
    std::vector<float> spine_tilts;

    for (const auto& item : frames) {
        const auto& qa = item.qa;
        if (item.robot_detected && item.count > 0) {
            detected_count++;
        }

        if (qa.qa_status != "NO_ROBOT") {
            if (qa.qa_status == "OCCLUDED") {
                occluded_count++;
            } else {
                valid_frames++;
                if (qa.qa_status == "KNEE_ASYMMETRY") {
                    knee_asymmetry_count++;
                }

                const float eval_thresh = (config.keypoint_confidence > 0.0f) ? config.keypoint_confidence : KPT_CONF_EVAL_THRESHOLD;
                // Spline disparity estimation on valid frame
                const auto& kpts = qa.kpts_raw_17;
                if (kpts.size() >= 17) {
                    const float c7 = kpts[7].conf, c8 = kpts[8].conf;
                    const float c13 = kpts[13].conf, c14 = kpts[14].conf;
                    if (c7 >= eval_thresh && c8 >= eval_thresh &&
                        c13 >= eval_thresh && c14 >= eval_thresh) {
                        const float x7 = kpts[7].x_norm, y7 = kpts[7].y_norm;
                        const float x8 = kpts[8].x_norm, y8 = kpts[8].y_norm;
                        const float x13 = kpts[13].x_norm, y13 = kpts[13].y_norm;
                        const float x14 = kpts[14].x_norm, y14 = kpts[14].y_norm;

                        const float theta_L = static_cast<float>(std::atan2((x8 - x7) * 1280.0f, (y8 - y7) * 720.0f) * 180.0 / M_PI);
                        const float theta_R = static_cast<float>(std::atan2((x14 - x13) * 1280.0f, (y14 - y13) * 720.0f) * 180.0 / M_PI);
                        const float disparity = std::abs(static_cast<float>(std::remainder(theta_L - theta_R, 360.0)));

                        const float ly = kpts[8].y_norm;
                        const float ry = kpts[14].y_norm;
                        const float knee_diff = std::abs(ly - ry);
#ifndef HOST_CLI
                        const float dx_l_px = (x8 - x7) * 1280.0f;
                        const float dy_l_px = (y8 - y7) * 720.0f;
                        const float dev_l = (std::abs(dy_l_px) < 4.0f || std::abs(dx_l_px) < 2.0f)
                            ? 0.0f
                            : std::abs(static_cast<float>(std::remainder(theta_L - config.nominal_stand_theta_l_deg, 360.0)));
                        const float dx_r_px = (x14 - x13) * 1280.0f;
                        const float dy_r_px = (y14 - y13) * 720.0f;
                        const float dev_r = (std::abs(dy_r_px) < 4.0f || std::abs(dx_r_px) < 2.0f)
                            ? 0.0f
                            : std::abs(static_cast<float>(std::remainder(theta_R - config.nominal_stand_theta_r_deg, 360.0)));
#else
                        const float dev_l = (x8 == x7 && y8 == y7) ? 0.0f : std::abs(static_cast<float>(std::remainder(theta_L - config.nominal_stand_theta_l_deg, 360.0)));
                        const float dev_r = (x14 == x13 && y14 == y13) ? 0.0f : std::abs(static_cast<float>(std::remainder(theta_R - config.nominal_stand_theta_r_deg, 360.0)));
#endif
                        const float max_dev = std::max(dev_l, dev_r);
                        const float disp_dev = std::abs(static_cast<float>(std::remainder(disparity - config.nominal_stand_disparity_deg, 360.0)));

                        float frame_spline = 0.0f;
                        if (max_dev >= config.gear_tooth_tolerance_deg) {
                            frame_spline = max_dev;
                        } else if (knee_diff <= config.bilateral_height_tolerance_norm) {
                            frame_spline = disp_dev;
                        } else {
                            frame_spline = max_dev;
                        }
                        spline_offsets.push_back(frame_spline);
                        faulted_joints.push_back(dev_l > dev_r ? "left_hip_pitch" : "right_hip_pitch");
                    }

                    if (kpts[8].conf >= eval_thresh && kpts[14].conf >= eval_thresh) {
                        const float ly = kpts[8].y_norm;
                        const float ry = kpts[14].y_norm;
                        l_knee_heights.push_back(ly);
                        r_knee_heights.push_back(ry);
                        knee_diffs.push_back(std::abs(ly - ry));
                    }
                }

                if (qa.metrics.has_spine_tilt) {
                    spine_tilts.push_back(qa.metrics.spine_tilt_deg);
                }
            }
        }
    }

    rep.detected_frames = detected_count;
    rep.valid_frames = valid_frames;
    rep.occluded_frames = occluded_count;

    // Metrics aggregation (always preserves real valid measurements)
    const float avg_l_knee = !l_knee_heights.empty()
        ? std::accumulate(l_knee_heights.begin(), l_knee_heights.end(), 0.0f) / l_knee_heights.size() : 0.0f;
    const float avg_r_knee = !r_knee_heights.empty()
        ? std::accumulate(r_knee_heights.begin(), r_knee_heights.end(), 0.0f) / r_knee_heights.size() : 0.0f;
    const float avg_knee_diff = !knee_diffs.empty()
        ? std::accumulate(knee_diffs.begin(), knee_diffs.end(), 0.0f) / knee_diffs.size() : 0.0f;
    float max_spline_offset = 0.0f;
    if (!spline_offsets.empty()) {
        float sum_splines = std::accumulate(spline_offsets.begin(), spline_offsets.end(), 0.0f);
        max_spline_offset = sum_splines / spline_offsets.size();
    }
    const std::string faulted_joint = !faulted_joints.empty() ? faulted_joints.front() : "right_hip_pitch";

    rep.metrics.left_knee_height_norm = avg_l_knee;
    rep.metrics.right_knee_height_norm = avg_r_knee;
    rep.metrics.knee_height_diff_norm = avg_knee_diff;
    rep.metrics.estimated_spline_offset_deg = max_spline_offset;

    if (!spine_tilts.empty()) {
        rep.metrics.has_spine_tilt = true;
        rep.metrics.spine_tilt_deg = std::accumulate(spine_tilts.begin(), spine_tilts.end(), 0.0f) / spine_tilts.size();
    } else {
        rep.metrics.has_spine_tilt = false;
        rep.metrics.spine_tilt_deg = 0.0f;
    }

    // P-TIMEOUT arbitration
    if (timed_out) {
        rep.status_code = "FAIL_MOTION_TIMEOUT";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "MOTION_TIMEOUT"},
            {"collected_frames", rep.total_frames},
            {"required_frames", SETTLED_WINDOW_FRAMES}
        });
        return rep;
    }

    const int total_non_no_robot = valid_frames + occluded_count;

    // P0: FAIL_NO_ROBOT (< 5 detected frames)
    if (detected_count < MIN_DETECTED_FRAMES) {
        rep.status_code = "FAIL_NO_ROBOT";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "NO_ROBOT"},
            {"detected_frames", detected_count}
        });
    }
    // P1: FAIL_OCCLUDED (>= 20% occluded of total non-no-robot frames)
    else if (total_non_no_robot > 0 &&
             (static_cast<float>(occluded_count) / static_cast<float>(total_non_no_robot)) >= OCCLUSION_RATIO_THRESHOLD) {
        rep.status_code = "FAIL_OCCLUDED";
        rep.overall_passed = false;
        const float occ_ratio = static_cast<float>(occluded_count) / static_cast<float>(total_non_no_robot);
        rep.detected_defects.push_back({
            {"type", "OCCLUDED"},
            {"occluded_ratio", std::round(occ_ratio * 100.0f) / 100.0f}
        });
    }
    // P2: FAIL_SPLINE_OFFSET (>= tolerance)
    else if (max_spline_offset >= config.gear_tooth_tolerance_deg) {
        rep.status_code = "FAIL_SPLINE_OFFSET";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "SPLINE_OFFSET"},
            {"joint", faulted_joint},
            {"measured_offset_deg", std::round(max_spline_offset * 10.0f) / 10.0f},
            {"tolerance_deg", config.gear_tooth_tolerance_deg}
        });
    }
    // P3: FAIL_KNEE_ASYMMETRY (> 30% valid frames exceeding height tolerance)
    else if (valid_frames > 0 &&
             (static_cast<float>(knee_asymmetry_count) / static_cast<float>(valid_frames)) > KNEE_ASYMMETRY_RATIO_THRESHOLD) {
        rep.status_code = "FAIL_KNEE_ASYMMETRY";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "KNEE_ASYMMETRY"},
            {"measured_diff_norm", std::round(avg_knee_diff * 10000.0f) / 10000.0f},
            {"tolerance_norm", config.bilateral_height_tolerance_norm}
        });
    }
    // P4: FAIL_SPINE_TILT (|avg_tilt - 14.82| > tolerance with >= 15 valid frames)
    else if (rep.metrics.has_spine_tilt &&
             spine_tilts.size() >= static_cast<size_t>(MIN_VALID_FRAMES) &&
             std::abs(rep.metrics.spine_tilt_deg - config.nominal_stand_spine_tilt_deg) > config.spine_tilt_tolerance_deg) {
        rep.status_code = "FAIL_SPINE_TILT";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "SPINE_TILT"},
            {"measured_tilt_deg", std::round(rep.metrics.spine_tilt_deg * 100.0f) / 100.0f},
            {"tolerance_deg", config.spine_tilt_tolerance_deg}
        });
    }
    // P5: FAIL_INSUFFICIENT_FRAMES (insufficient posture measurements)
    else if (spline_offsets.size() < static_cast<size_t>(MIN_VALID_FRAMES) ||
             knee_diffs.size() < static_cast<size_t>(MIN_VALID_FRAMES) ||
             spine_tilts.size() < static_cast<size_t>(MIN_VALID_FRAMES)) {
        rep.status_code = "FAIL_INSUFFICIENT_FRAMES";
        rep.overall_passed = false;
        rep.detected_defects.push_back({
            {"type", "INSUFFICIENT_MEASUREMENTS"},
            {"valid_spline_frames", static_cast<int>(spline_offsets.size())},
            {"valid_knee_frames", static_cast<int>(knee_diffs.size())},
            {"valid_spine_frames", static_cast<int>(spine_tilts.size())},
            {"required_frames", MIN_VALID_FRAMES}
        });
    }
    // P6: PASS check (>= 15 valid frames)
    else if (valid_frames < MIN_VALID_FRAMES) {
        rep.status_code = "FAIL_INSUFFICIENT_FRAMES";
        rep.overall_passed = false;
    } else {
        rep.status_code = "PASS";
        rep.overall_passed = true;
    }

    return rep;
}

bool ActionStateMachine::startAction(const std::string& action_id,
                                     const std::string& run_id,
                                     float timeout_s,
                                     bool force,
                                     const AppConfig& config_snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == ActionState::COLLECTING && !force) {
        return false;
    }

    current_action_id_ = action_id;
    current_run_id_ = run_id;
    timeout_s_ = (timeout_s >= 1.0f && timeout_s <= 10.0f) ? timeout_s : 3.0f;
    config_snapshot_ = config_snapshot;
    window_frames_.clear();
    window_frames_.reserve(SETTLED_WINDOW_FRAMES);

    action_generation_++;
    const auto now = std::chrono::steady_clock::now();
    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    state_ = ActionState::COLLECTING;
    has_report_ = false;
    broadcast_pending_ = false;
    return true;
}

void ActionStateMachine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = ActionState::IDLE;
    window_frames_.clear();
    has_report_ = false;
    broadcast_pending_ = false;
    current_run_id_.clear();
}

bool ActionStateMachine::stopAction() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == ActionState::COLLECTING) {
        finalizeReportLocked(false);
        return true;
    }
    return true;
}

void ActionStateMachine::finalizeReportLocked(bool timed_out) {
    last_report_ = aggregateWindow(current_action_id_, current_run_id_, config_snapshot_, window_frames_, timed_out);
    has_report_ = true;
    state_ = ActionState::REPORT_FROZEN;
    broadcast_pending_ = true;
    broadcast_pending_generation_ = action_generation_;
}

bool ActionStateMachine::checkTimeout() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != ActionState::COLLECTING) return false;

    const auto now = std::chrono::steady_clock::now();
    const uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const float elapsed_s = static_cast<float>(now_ms - start_time_ms_) / 1000.0f;

    if (elapsed_s >= timeout_s_) {
        finalizeReportLocked(true);
        return true;
    }
    return false;
}

bool ActionStateMachine::addFrame(uint32_t frame_id,
                                  uint64_t frame_time_ms,
                                  uint64_t frame_generation,
                                  bool robot_detected,
                                  int count,
                                  const FrameQAResult& frame_qa,
                                  const FaceLedFrameResult& optical) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != ActionState::COLLECTING) return false;

    // Reject stale generation or frame timestamped before action began
    if (frame_generation != action_generation_) return false;
    if (frame_time_ms > 0 && frame_time_ms < start_time_ms_) return false;

    // Check timeout
    const auto now = std::chrono::steady_clock::now();
    const uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const float elapsed_s = static_cast<float>(now_ms - start_time_ms_) / 1000.0f;
    if (elapsed_s >= timeout_s_) {
        finalizeReportLocked(true);
        return true;
    }

    CollectedFrame cf;
    cf.frame_id = frame_id;
    cf.timestamp_ms = frame_time_ms;
    cf.generation = frame_generation;
    cf.robot_detected = robot_detected;
    cf.count = count;
    cf.qa = frame_qa;
    cf.optical = optical;
    window_frames_.push_back(std::move(cf));

    if (window_frames_.size() >= static_cast<size_t>(SETTLED_WINDOW_FRAMES)) {
        finalizeReportLocked(false);
        return true;
    }
    return false;
}

void ActionStateMachine::setSysError(const std::string& reason, uint64_t gen) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Strictly require COLLECTING state and exact generation match; never overwrite frozen reports
    if (state_ != ActionState::COLLECTING || gen != action_generation_) {
        return;
    }

    // In COLLECTING: retain existing valid frame measurement averages
    ActionReport rep = aggregateWindow(current_action_id_, current_run_id_, config_snapshot_, window_frames_, false);
    rep.overall_passed = false;
    rep.status_code = "FAIL_SYS_ERROR";
    rep.detected_defects = {
        {
            {"type", "SYS_ERROR"},
            {"reason", reason}
        }
    };
    last_report_ = std::move(rep);
    has_report_ = true;
    state_ = ActionState::REPORT_FROZEN;
    broadcast_pending_ = true;
    broadcast_pending_generation_ = action_generation_;
}

#ifndef HOST_CLI
json skeletonJson(const PayloadContext& ctx, const std::vector<Subject>& subjects) {
    json groups = json::array();

    for (const auto& subj : subjects) {
        const auto& pose = subj.pose;
        if (pose.empty()) continue;

        json pts = json::array();
        constexpr int kJointCount = static_cast<int>(Joint::Count);
        for (int i = 0; i < kJointCount; ++i) {
            Point2f pt = pose.at(i);
            float x_disp = pt.x, y_disp = pt.y;
            letterbox_to_display_pt(x_disp, y_disp, ctx.infer_w, ctx.infer_h, ctx.stream_w, ctx.stream_h);
            x_disp = std::clamp(x_disp, 0.0f, static_cast<float>(ctx.stream_w));
            y_disp = std::clamp(y_disp, 0.0f, static_cast<float>(ctx.stream_h));
            pts.push_back(json::array({std::round(x_disp * 10.0f) / 10.0f, std::round(y_disp * 10.0f) / 10.0f}));
        }

        json edges = json::array();
        for (const auto& b : kMicroduckSkeleton) {
            edges.push_back(json::array({b.a, b.b}));
        }

        json group;
        group["resolution"] = json::array({ctx.stream_w, ctx.stream_h});
        group["points"] = std::move(pts);
        group["edges"] = std::move(edges);
        group["color"] = "#00ffcc";
        groups.push_back(std::move(group));
    }

    return groups;
}

static json statusCardJson(const PayloadContext& ctx,
                           const FrameQAResult& qa,
                           const ActionStateMachine& sm) {
    json card;
    card["title"] = "Microduck Pose QA";
    card["action_state"] = sm.getStateString();
    card["current_run_id"] = sm.getCurrentRunId();
    card["current_action_id"] = sm.getCurrentActionId();
    card["qa_status"] = qa.qa_status;
    card["qa_passed"] = qa.qa_passed;

    if (sm.getCurrentActionId() == "face_led") {
        card["lit_pixels"] = ctx.optical.lit_pixels;
        card["dark_leakage_lux"] = std::round(ctx.optical.dark_leakage_lux * 10.0f) / 10.0f;
        card["uniformity_ratio"] = std::round(ctx.optical.uniformity_ratio * 1000.0f) / 1000.0f;
    } else {
        if (qa.metrics.has_knee_diff) {
            card["knee_diff_norm"] = std::round(qa.metrics.knee_height_diff_norm * 10000.0f) / 10000.0f;
        }
        if (qa.metrics.has_spline_offset) {
            card["spline_offset_deg"] = std::round(qa.metrics.spline_offset_deg * 10.0f) / 10.0f;
        }
        if (qa.metrics.has_spine_tilt) {
            card["spine_tilt_deg"] = std::round(qa.metrics.spine_tilt_deg * 100.0f) / 100.0f;
        }
    }

    if (sm.hasReport()) {
        card["last_action_report"] = sm.getLastReportJson();
    }
    return card;
}

std::string buildResultJson(const PayloadContext& ctx,
                            const std::vector<Subject>& subjects,
                            const FrameQAResult& qa,
                            const ActionStateMachine& sm) {
    json j;
    j["timestamp"] = ctx.timestamp_ms;
    j["frame_id"] = ctx.frame_id;
    j["inference_time_ms"] = std::round(ctx.inference_time_ms * 10.0f) / 10.0f;
    j["fps"] = std::round(ctx.fps * 10.0f) / 10.0f;
    j["robot_detected"] = ctx.robot_detected;
    j["count"] = ctx.duck_count;
    j["resolution"] = json::array({ctx.stream_w, ctx.stream_h});
    j["qa_status"] = qa.qa_status;
    j["qa_passed"] = qa.qa_passed;

    // Backward-compatible skeleton points and edges
    if (!subjects.empty()) {
        json skel = skeletonJson(ctx, subjects);
        if (!skel.empty()) {
            j["keypoints"] = std::move(skel);
        }
    }

    // 17 normalized raw keypoints
    json kpts_arr = json::array();
    for (const auto& kp : qa.kpts_raw_17) {
        json k;
        k["id"] = kp.id;
        k["name"] = kp.name;
        k["x_norm"] = std::round(kp.x_norm * 10000.0f) / 10000.0f;
        k["y_norm"] = std::round(kp.y_norm * 10000.0f) / 10000.0f;
        k["conf"] = std::round(kp.conf * 100.0f) / 100.0f;
        kpts_arr.push_back(std::move(k));
    }
    j["kpts_raw_17"] = std::move(kpts_arr);

    // qa_metrics
    json qm;
    if (ctx.action_id == "face_led") {
        qm["lit_pixels"] = ctx.optical.lit_pixels;
        qm["color_temperature_k"] = std::round(ctx.optical.color_temperature_k * 10.0f) / 10.0f;
        json rgb = json::array();
        for (float c : ctx.optical.mean_rgb) {
            rgb.push_back(std::round(c * 10.0f) / 10.0f);
        }
        qm["mean_rgb"] = std::move(rgb);
        qm["dark_leakage_lux"] = std::round(ctx.optical.dark_leakage_lux * 10.0f) / 10.0f;
        qm["uniformity_ratio"] = std::round(ctx.optical.uniformity_ratio * 1000.0f) / 1000.0f;
        j["optical_metrics"] = qm;
    } else {
        if (qa.metrics.has_knee_diff) {
            qm["knee_height_diff_norm"] = std::round(qa.metrics.knee_height_diff_norm * 10000.0f) / 10000.0f;
        } else {
            qm["knee_height_diff_norm"] = nullptr;
        }
        if (qa.metrics.has_spline_offset) {
            qm["spline_offset_deg"] = std::round(qa.metrics.spline_offset_deg * 10.0f) / 10.0f;
        } else {
            qm["spline_offset_deg"] = nullptr;
        }
        if (qa.metrics.has_spine_tilt) {
            qm["spine_tilt_deg"] = std::round(qa.metrics.spine_tilt_deg * 100.0f) / 100.0f;
        } else {
            qm["spine_tilt_deg"] = nullptr;
        }
    }
    j["qa_metrics"] = std::move(qm);

    // action_report
    if (sm.hasReport()) {
        j["action_report"] = sm.getLastReportJson();
    } else {
        j["action_report"] = nullptr;
    }

    return j.dump();
}

std::string buildDebugExtraJson(const PayloadContext& ctx,
                                const std::vector<Subject>& subjects,
                                const FrameQAResult& qa,
                                const ActionStateMachine& sm) {
    json j;
    // timestamp, frame_id, inference_time_ms, resolution are provided by outer debug_stream.cpp:590
    j["fps"] = std::round(ctx.fps * 10.0f) / 10.0f;
    j["robot_detected"] = ctx.robot_detected;
    j["count"] = ctx.duck_count;
    j["qa_status"] = qa.qa_status;
    j["qa_passed"] = qa.qa_passed;

    if (!subjects.empty()) {
        json skel = skeletonJson(ctx, subjects);
        if (!skel.empty()) {
            j["keypoints"] = std::move(skel);
        }
    }

    json kpts_arr = json::array();
    for (const auto& kp : qa.kpts_raw_17) {
        json k;
        k["id"] = kp.id;
        k["name"] = kp.name;
        k["x_norm"] = std::round(kp.x_norm * 10000.0f) / 10000.0f;
        k["y_norm"] = std::round(kp.y_norm * 10000.0f) / 10000.0f;
        k["conf"] = std::round(kp.conf * 100.0f) / 100.0f;
        kpts_arr.push_back(std::move(k));
    }
    j["kpts_raw_17"] = std::move(kpts_arr);

    json qm;
    if (ctx.action_id == "face_led") {
        qm["lit_pixels"] = ctx.optical.lit_pixels;
        qm["color_temperature_k"] = std::round(ctx.optical.color_temperature_k * 10.0f) / 10.0f;
        json rgb = json::array();
        for (float c : ctx.optical.mean_rgb) {
            rgb.push_back(std::round(c * 10.0f) / 10.0f);
        }
        qm["mean_rgb"] = std::move(rgb);
        qm["dark_leakage_lux"] = std::round(ctx.optical.dark_leakage_lux * 10.0f) / 10.0f;
        qm["uniformity_ratio"] = std::round(ctx.optical.uniformity_ratio * 1000.0f) / 1000.0f;
        j["optical_metrics"] = qm;
    } else {
        if (qa.metrics.has_knee_diff) {
            qm["knee_height_diff_norm"] = std::round(qa.metrics.knee_height_diff_norm * 10000.0f) / 10000.0f;
        } else {
            qm["knee_height_diff_norm"] = nullptr;
        }
        if (qa.metrics.has_spline_offset) {
            qm["spline_offset_deg"] = std::round(qa.metrics.spline_offset_deg * 10.0f) / 10.0f;
        } else {
            qm["spline_offset_deg"] = nullptr;
        }
        if (qa.metrics.has_spine_tilt) {
            qm["spine_tilt_deg"] = std::round(qa.metrics.spine_tilt_deg * 100.0f) / 100.0f;
        } else {
            qm["spine_tilt_deg"] = nullptr;
        }
    }
    j["qa_metrics"] = std::move(qm);

    if (sm.hasReport()) {
        j["action_report"] = sm.getLastReportJson();
    } else {
        j["action_report"] = nullptr;
    }

    j["status_card"] = statusCardJson(ctx, qa, sm);

    std::string s = j.dump();
    if (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
        return s.substr(1, s.size() - 2);
    }
    return std::string();
}
#endif

}  // namespace microduck
