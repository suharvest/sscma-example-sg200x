#include "result_payload.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <debug_stream.h>

#include "json.hpp"

namespace fall {

using json = nlohmann::json;

namespace {

struct Bone {
    Joint a;
    Joint b;
};

constexpr Bone kSkeleton[] = {
    {Joint::Nose, Joint::LeftEye}, {Joint::Nose, Joint::RightEye},
    {Joint::LeftEye, Joint::LeftEar}, {Joint::RightEye, Joint::RightEar},
    {Joint::LeftShoulder, Joint::RightShoulder},
    {Joint::LeftShoulder, Joint::LeftElbow}, {Joint::LeftElbow, Joint::LeftWrist},
    {Joint::RightShoulder, Joint::RightElbow}, {Joint::RightElbow, Joint::RightWrist},
    {Joint::LeftShoulder, Joint::LeftHip}, {Joint::RightShoulder, Joint::RightHip},
    {Joint::LeftHip, Joint::RightHip},
    {Joint::LeftHip, Joint::LeftKnee}, {Joint::LeftKnee, Joint::LeftAnkle},
    {Joint::RightHip, Joint::RightKnee}, {Joint::RightKnee, Joint::RightAnkle},
};

json skeletonJson(const PayloadContext& ctx, const Pose& pose, bool normalize) {
    constexpr int kJointCount = static_cast<int>(Joint::Count);
    std::vector<debug_stream_box_t> points_px;
    std::vector<int> compact(kJointCount, -1);
    points_px.reserve(kJointCount);

    for (int i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        if (!pose.visible(j)) continue;
        const Point2f p = pose.at(j);
        compact[i] = static_cast<int>(points_px.size());
        points_px.push_back({p.x, p.y, 0.0f, 0.0f, pose.confidence(j), std::string()});
    }
    if (points_px.empty()) return json();

    debug_stream_letterbox_to_display(points_px, ctx.infer_w, ctx.infer_h,
                                      ctx.stream_w, ctx.stream_h);
    json points = json::array();
    for (const auto& p : points_px) {
        if (normalize) {
            points.push_back({std::round(p.x / ctx.stream_w * 10000.0f) / 10000.0f,
                              std::round(p.y / ctx.stream_h * 10000.0f) / 10000.0f});
        } else {
            points.push_back({std::round(p.x * 10.0f) / 10.0f,
                              std::round(p.y * 10.0f) / 10.0f});
        }
    }

    json edges = json::array();
    for (const Bone& bone : kSkeleton) {
        const int a = compact[static_cast<int>(bone.a)];
        const int b = compact[static_cast<int>(bone.b)];
        if (a >= 0 && b >= 0) edges.push_back({a, b});
    }
    json group;
    group["points"] = std::move(points);
    group["edges"] = std::move(edges);
    return json::array({std::move(group)});
}

// Stable COCO-17 feature layout for offline training and downstream temporal
// models. Unlike the compact drawing payload, indices never shift when a
// joint is missing: each row is [x/infer_w, y/infer_h, confidence].
json pose17Json(const PayloadContext& ctx, const Pose& pose) {
    json points = json::array();
    const float w = static_cast<float>(std::max(1, ctx.infer_w));
    const float h = static_cast<float>(std::max(1, ctx.infer_h));
    for (int i = 0; i < static_cast<int>(Joint::Count); ++i) {
        const Joint joint = static_cast<Joint>(i);
        const Point2f p = pose.at(joint);
        points.push_back({
            std::round(p.x / w * 10000.0f) / 10000.0f,
            std::round(p.y / h * 10000.0f) / 10000.0f,
            std::round(pose.confidence(joint) * 10000.0f) / 10000.0f,
        });
    }
    return points;
}

json featureJson(const FallObservation& o, const FallDiagnostics& d) {
    json f;
    f["hip_y"] = std::round(o.hip_y * 10000.0f) / 10000.0f;
    f["person_score"] = std::round(o.person_score * 100.0f) / 100.0f;
    f["hip_drop_speed"] = std::round(d.hip_drop_speed * 1000.0f) / 1000.0f;
    f["hip_drop_distance"] = std::round(d.hip_drop_distance * 1000.0f) / 1000.0f;
    f["torso_angle_deg"] = std::round(d.torso_angle_deg * 10.0f) / 10.0f;
    f["bbox_aspect_ratio"] = std::round(d.bbox_aspect_ratio * 100.0f) / 100.0f;
    f["evidence_features"] = d.evidence_features;
    f["evidence_score"] = std::round(d.evidence_score * 100.0f) / 100.0f;
    f["lying_posture"] = d.lying_posture;
    f["upright_posture"] = d.upright_posture;
    f["in_cooldown"] = d.in_cooldown;
    f["temporal_probability"] = std::round(d.temporal_probability * 10000.0f) / 10000.0f;
    f["temporal_positive"] = d.temporal_positive;
    f["suspected_for_sec"] = std::round(d.suspected_for_sec * 100.0) / 100.0;
    f["recovery_for_sec"] = std::round(d.recovery_for_sec * 100.0) / 100.0;
    return f;
}

json coreFields(const PayloadContext& ctx, const FallOutput& output,
                const FallObservation& observation) {
    json j;
    j["fall_detected"] = output.fall_detected;
    j["fall_event"] = output.fall_event;
    j["event_id"] = output.event_id;
    j["state"] = fallStateName(output.state);
    j["person_detected"] = ctx.person_detected;
    j["person_count"] = ctx.person_count;
    j["fallen_count"] = ctx.fallen_count;
    j["tracking"] = observation.valid;
    j["features"] = featureJson(observation, output.diagnostics);
    return j;
}

json statusCardJson(const PayloadContext& ctx, const FallOutput& output,
                   const FallObservation& observation) {
    json metrics = json::array();
    metrics.push_back({{"k", "State"}, {"v", fallStateName(output.state)}});
    metrics.push_back({{"k", "People"}, {"v", std::to_string(ctx.person_count)}});
    metrics.push_back({{"k", "Evidence"}, {"v", std::to_string(output.diagnostics.evidence_features) + "/3"}});
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f%%", output.diagnostics.temporal_probability * 100.0f);
        metrics.push_back({{"k", "Temporal"}, {"v", std::string(buf)}});
    }
    if (observation.valid) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f°", output.diagnostics.torso_angle_deg);
        metrics.push_back({{"k", "Torso"}, {"v", std::string(buf)}});
    }
    json card;
    card["title"] = output.fall_detected ? "FALL DETECTED" : fallStateName(output.state);
    card["tone"] = output.fall_detected ? "alert"
                  : (output.state == FallState::Suspected ? "warn" : "ok");
    card["metrics"] = std::move(metrics);
    return card;
}

}  // namespace

std::string buildResultJson(const PayloadContext& ctx, const FallOutput& output,
                            const FallObservation& observation, const Pose* pose) {
    json j = coreFields(ctx, output, observation);
    j["timestamp"] = ctx.timestamp_ms;
    j["frame_id"] = ctx.frame_id;
    j["inference_time_ms"] = static_cast<int>(ctx.inference_time_ms + 0.5f);
    if (pose != nullptr && !pose->empty()) {
        json skeleton = skeletonJson(ctx, *pose, true);
        if (!skeleton.is_null()) j["keypoints"] = std::move(skeleton);
        j["pose17"] = pose17Json(ctx, *pose);
    }
    return j.dump();
}

std::string buildDebugExtraJson(const PayloadContext& ctx, const FallOutput& output,
                                const FallObservation& observation, const Pose* pose) {
    json j = coreFields(ctx, output, observation);
    if (pose != nullptr && !pose->empty()) {
        json skeleton = skeletonJson(ctx, *pose, false);
        if (!skeleton.is_null()) j["keypoints"] = std::move(skeleton);
    }
    j["status_card"] = statusCardJson(ctx, output, observation);
    const std::string s = j.dump();
    if (s.size() >= 2 && s.front() == '{' && s.back() == '}') return s.substr(1, s.size() - 2);
    return std::string();
}

}  // namespace fall
