#include "result_payload.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <debug_stream.h>

#include "json.hpp"
#include "payload_aggregate.h"

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
            points.push_back({std::round(p.x / std::max(1, ctx.stream_w) * 10000.0f) / 10000.0f,
                              std::round(p.y / std::max(1, ctx.stream_h) * 10000.0f) / 10000.0f});
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
    f["valid"] = o.valid;
    return f;
}

struct Aggregate {
    bool fall_detected = false;
    bool fall_event = false;
    bool person_detected = false;
    int person_count = 0;
    int fallen_count = 0;
    std::uint64_t event_id = 0;
    FallState state = FallState::Normal;
    const TrackedPerson* primary = nullptr;
};

Aggregate aggregate(const PayloadContext& ctx) {
    Aggregate result;
    std::vector<PayloadPersonSummary> summaries;
    summaries.reserve(ctx.persons.size());
    int visible_count = 0;
    for (const auto* person : ctx.persons) {
        if (person == nullptr) continue;
        if (person->visible) ++visible_count;
        if (result.primary == nullptr ||
            (person->visible && !result.primary->visible) ||
            (person->visible == result.primary->visible &&
             person->subject.score > result.primary->subject.score)) {
            result.primary = person;
        }
        summaries.push_back({person->output.event_id, person->output.state,
                             person->output.fall_detected, person->output.fall_event});
    }
    const int person_count = ctx.persons.empty() ? ctx.person_count : visible_count;
    const PayloadAggregate summary = aggregatePayload(summaries, person_count,
                                                       ctx.global_event_id,
                                                       ctx.global_event_id_valid);
    result.fall_detected = summary.fall_detected;
    result.fall_event = summary.fall_event;
    result.person_count = summary.person_count;
    result.fallen_count = summary.fallen_count;
    result.event_id = summary.event_id;
    result.state = summary.state;
    result.person_detected = result.person_count > 0;
    return result;
}

json coreFields(const PayloadContext& ctx, const Aggregate& a) {
    json j;
    j["fall_detected"] = a.fall_detected;
    j["fall_event"] = a.fall_event;
    j["event_id"] = a.event_id;
    // The live tracker supplies a monotonic stream-global sequence. Legacy
    // callers with no tracker context fall back to max(per-track event_id),
    // and the scope field makes that distinction explicit to consumers.
    j["event_id_scope"] = ctx.global_event_id_valid
        ? "stream_global_event_id" : "max_per_track_event_id";
    if (ctx.global_event_id_valid) j["global_event_id"] = ctx.global_event_id;
    j["state"] = fallStateName(a.state);
    j["person_detected"] = a.person_detected;
    j["person_count"] = a.person_count;
    j["fallen_count"] = a.fallen_count;
    j["tracking"] = !ctx.persons.empty();
    if (a.primary != nullptr) {
        j["features"] = featureJson(a.primary->observation, a.primary->output.diagnostics);
    } else {
        j["features"] = featureJson(FallObservation{}, FallDiagnostics{});
    }
    return j;
}

json personJson(const PayloadContext& ctx, const TrackedPerson& person) {
    json p;
    p["track_id"] = person.track_id;
    p["person_detected"] = person.visible;
    p["person_score"] = std::round(person.subject.score * 10000.0f) / 10000.0f;
    p["fall_detected"] = person.output.fall_detected;
    p["fall_event"] = person.output.fall_event;
    p["event_id"] = person.output.event_id;
    p["state"] = fallStateName(person.output.state);
    p["tracking"] = person.visible;
    p["missed_frames"] = person.missed;
    p["bbox"] = {
        std::round(person.subject.box.cx * 10000.0f) / 10000.0f,
        std::round(person.subject.box.cy * 10000.0f) / 10000.0f,
        std::round(person.subject.box.w * 10000.0f) / 10000.0f,
        std::round(person.subject.box.h * 10000.0f) / 10000.0f,
    };
    p["features"] = featureJson(person.observation, person.output.diagnostics);
    p["keypoints"] = json::array();
    p["pose17"] = json::array();
    if (person.visible && !person.subject.pose.empty()) {
        json skeleton = skeletonJson(ctx, person.subject.pose, true);
        if (!skeleton.is_null()) p["keypoints"] = std::move(skeleton);
        p["pose17"] = pose17Json(ctx, person.subject.pose);
    }
    return p;
}

json aggregateSkeletonJson(const PayloadContext& ctx, bool normalize) {
    json groups = json::array();
    for (const auto* person : ctx.persons) {
        if (person == nullptr || !person->visible || person->subject.pose.empty()) continue;
        json skeleton = skeletonJson(ctx, person->subject.pose, normalize);
        if (!skeleton.is_array()) continue;
        for (auto& group : skeleton) groups.push_back(std::move(group));
    }
    return groups;
}

json personsJson(const PayloadContext& ctx) {
    json persons = json::array();
    for (const auto* person : ctx.persons) {
        if (person != nullptr) persons.push_back(personJson(ctx, *person));
    }
    return persons;
}

json statusCardJson(const Aggregate& a) {
    json metrics = json::array();
    metrics.push_back({{"k", "State"}, {"v", fallStateName(a.state)}});
    metrics.push_back({{"k", "People"}, {"v", std::to_string(a.person_count)}});
    metrics.push_back({{"k", "Fallen"}, {"v", std::to_string(a.fallen_count)}});
    if (a.primary != nullptr) {
        metrics.push_back({{"k", "Evidence"},
                           {"v", std::to_string(a.primary->output.diagnostics.evidence_features) + "/3"}});
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f%%",
                      a.primary->output.diagnostics.temporal_probability * 100.0f);
        metrics.push_back({{"k", "Temporal"}, {"v", std::string(buf)}});
        if (a.primary->observation.valid) {
            std::snprintf(buf, sizeof(buf), "%.0f°", a.primary->output.diagnostics.torso_angle_deg);
            metrics.push_back({{"k", "Torso"}, {"v", std::string(buf)}});
        }
    }
    json card;
    card["title"] = a.fall_detected ? "FALL DETECTED" : fallStateName(a.state);
    card["tone"] = a.fall_detected ? "alert"
                  : (a.state == FallState::Suspected ? "warn" : "ok");
    card["metrics"] = std::move(metrics);
    return card;
}

std::string buildPayloadJson(const PayloadContext& ctx, bool debug) {
    const Aggregate a = aggregate(ctx);
    json j = coreFields(ctx, a);
    j["timestamp"] = ctx.timestamp_ms;
    j["frame_id"] = ctx.frame_id;
    j["inference_time_ms"] = static_cast<int>(ctx.inference_time_ms + 0.5f);
    j["stream_id"] = ctx.stream_id;
    j["persons"] = personsJson(ctx);
    const json skeleton = aggregateSkeletonJson(ctx, !debug);
    j["keypoints"] = skeleton;
    j["pose17"] = json::array();
    if (a.primary != nullptr && a.primary->visible && !a.primary->subject.pose.empty()) {
        j["pose17"] = pose17Json(ctx, a.primary->subject.pose);
    }
    if (debug) j["status_card"] = statusCardJson(a);
    return j.dump();
}

}  // namespace

std::string buildResultJson(const PayloadContext& ctx) {
    return buildPayloadJson(ctx, false);
}

std::string buildDebugExtraJson(const PayloadContext& ctx) {
    const Aggregate a = aggregate(ctx);
    json j = coreFields(ctx, a);
    j["stream_id"] = ctx.stream_id;
    j["persons"] = personsJson(ctx);
    const json skeleton = aggregateSkeletonJson(ctx, false);
    j["keypoints"] = skeleton;
    j["pose17"] = json::array();
    if (a.primary != nullptr && a.primary->visible && !a.primary->subject.pose.empty()) {
        j["pose17"] = pose17Json(ctx, a.primary->subject.pose);
    }
    j["status_card"] = statusCardJson(a);
    const std::string document = j.dump();
    if (document.size() >= 2 && document.front() == '{' && document.back() == '}') {
        return document.substr(1, document.size() - 2);
    }
    return std::string();
}

std::string buildResultJson(const PayloadContext& ctx, const FallOutput& output,
                            const FallObservation& observation, const Pose* pose) {
    TrackedPerson person;
    person.track_id = 0;
    person.visible = pose != nullptr && !pose->empty();
    person.subject.score = observation.person_score;
    if (pose != nullptr) person.subject.pose = *pose;
    person.output = output;
    person.observation = observation;
    PayloadContext copy = ctx;
    copy.persons = {&person};
    return buildResultJson(copy);
}

std::string buildDebugExtraJson(const PayloadContext& ctx, const FallOutput& output,
                                const FallObservation& observation, const Pose* pose) {
    TrackedPerson person;
    person.track_id = 0;
    person.visible = pose != nullptr && !pose->empty();
    person.subject.score = observation.person_score;
    if (pose != nullptr) person.subject.pose = *pose;
    person.output = output;
    person.observation = observation;
    PayloadContext copy = ctx;
    copy.persons = {&person};
    return buildDebugExtraJson(copy);
}

}  // namespace fall
