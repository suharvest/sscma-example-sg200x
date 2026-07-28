#include "result_payload.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <debug_stream.h>

#include "json.hpp"

namespace fitness {

using json = nlohmann::json;

namespace {

json coreFields(const PayloadContext& ctx, const ExerciseState& st) {
    json j;
    j["exercise"] = ctx.exercise_id;
    j["stage"] = st.stage;
    j["reps"] = st.reps;
    j["target_reps"] = ctx.target_reps;
    j["set"] = st.set;
    j["target_sets"] = ctx.target_sets;
    j["workout_complete"] = st.workout_complete;
    j["person_detected"] = ctx.person_detected;
    j["tracking"] = st.tracking;

    if (st.has_angle) {
        // One decimal is already past the model's angular resolution; it is
        // there so a chart of the value does not look like a staircase.
        j["angle"] = std::round(st.angle * 10.0f) / 10.0f;
    }
    if (st.two_sided) {
        j["reps_left"] = st.reps_left;
        j["reps_right"] = st.reps_right;
    }
    if (!st.form_warning.empty()) {
        j["form_warning"] = st.form_warning;
    }
    return j;
}

// COCO-17 skeleton, in Joint terms rather than raw indices.
struct Bone {
    Joint a;
    Joint b;
};

constexpr Bone kSkeleton[] = {
    // Head
    {Joint::Nose, Joint::LeftEye},        {Joint::Nose, Joint::RightEye},
    {Joint::LeftEye, Joint::LeftEar},     {Joint::RightEye, Joint::RightEar},
    // Shoulders and arms
    {Joint::LeftShoulder, Joint::RightShoulder},
    {Joint::LeftShoulder, Joint::LeftElbow},   {Joint::LeftElbow, Joint::LeftWrist},
    {Joint::RightShoulder, Joint::RightElbow}, {Joint::RightElbow, Joint::RightWrist},
    // Torso
    {Joint::LeftShoulder, Joint::LeftHip},     {Joint::RightShoulder, Joint::RightHip},
    {Joint::LeftHip, Joint::RightHip},
    // Legs
    {Joint::LeftHip, Joint::LeftKnee},         {Joint::LeftKnee, Joint::LeftAnkle},
    {Joint::RightHip, Joint::RightKnee},       {Joint::RightKnee, Joint::RightAnkle},
};

// Build the "keypoints" layer payload: visible joints only, compacted, with
// edges reindexed into the compacted array.
json skeletonJson(const PayloadContext& ctx, const Pose& pose) {
    constexpr int kJointCount = static_cast<int>(Joint::Count);

    // Map inference-frame pixels into debug-video pixels by running each joint
    // through the same letterbox transform the boxes use. Reusing the shared
    // helper rather than reimplementing it is deliberate: two hand-written
    // copies of this mapping is exactly how the overlay and the privacy mask
    // once ended up disagreeing about where a face was.
    std::vector<debug_stream_box_t> pts;
    std::vector<int> compact_index(kJointCount, -1);
    pts.reserve(kJointCount);

    for (int i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        if (!pose.visible(j)) continue;
        const Point2f p = pose.at(j);
        compact_index[i] = static_cast<int>(pts.size());
        pts.push_back({p.x, p.y, 0.0f, 0.0f, pose.confidence(j), std::string()});
    }
    if (pts.empty()) {
        return json();
    }
    debug_stream_letterbox_to_display(pts, ctx.infer_w, ctx.infer_h, ctx.stream_w, ctx.stream_h);

    json points = json::array();
    for (const auto& p : pts) {
        points.push_back({std::round(p.x * 10.0f) / 10.0f, std::round(p.y * 10.0f) / 10.0f});
    }

    json edges = json::array();
    for (const Bone& bone : kSkeleton) {
        const int a = compact_index[static_cast<int>(bone.a)];
        const int b = compact_index[static_cast<int>(bone.b)];
        if (a < 0 || b < 0) continue;  // a bone needs both ends visible
        edges.push_back({a, b});
    }

    json group;
    group["points"] = std::move(points);
    group["edges"] = std::move(edges);
    return json::array({std::move(group)});
}

json statusCardJson(const PayloadContext& ctx, const ExerciseState& st) {
    json metrics = json::array();
    metrics.push_back({{"k", "Exercise"}, {"v", ctx.exercise_id}});
    metrics.push_back({{"k", "Set"}, {"v", std::to_string(st.set) + " / " + std::to_string(ctx.target_sets)}});
    metrics.push_back({{"k", "Stage"}, {"v", st.stage}});
    if (st.has_angle) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.0f", st.angle);
        metrics.push_back({{"k", "Angle"}, {"v", std::string(buf) + "\xc2\xb0"}});
    }
    if (st.two_sided) {
        metrics.push_back({{"k", "L / R"},
                           {"v", std::to_string(st.reps_left) + " / " + std::to_string(st.reps_right)}});
    }

    json card;
    card["title"] = st.workout_complete
                        ? std::string("DONE")
                        : (std::to_string(st.reps) + " / " + std::to_string(ctx.target_reps));
    // Green while a rep is being counted, amber when nobody is tracked -- the
    // colour answers "is it seeing me?" without reading the rows.
    card["tone"] = st.workout_complete ? "ok" : (st.tracking ? "ok" : "warn");
    card["metrics"] = std::move(metrics);
    if (!st.form_warning.empty()) {
        card["banner"] = st.form_warning;
    }
    return card;
}

}  // namespace

std::string buildResultJson(const PayloadContext& ctx, const ExerciseState& st) {
    json j = coreFields(ctx, st);
    j["timestamp"] = ctx.timestamp_ms;
    j["frame_id"] = ctx.frame_id;
    j["inference_time_ms"] = static_cast<int>(ctx.inference_time_ms + 0.5f);
    // Edge events, so a subscriber can react to a rep without diffing counts.
    j["rep_completed"] = st.rep_completed;
    j["set_completed"] = st.set_completed;
    return j.dump();
}

std::string buildDebugExtraJson(const PayloadContext& ctx, const ExerciseState& st,
                                const Pose* pose) {
    json j = coreFields(ctx, st);

    if (pose != nullptr && !pose->empty()) {
        json skel = skeletonJson(ctx, *pose);
        if (!skel.is_null()) {
            j["keypoints"] = std::move(skel);
        }
    }
    j["status_card"] = statusCardJson(ctx, st);

    std::string s = j.dump();
    // Strip the outer braces: debug_stream_build_results splices the result in
    // as additional top-level members of its own envelope.
    if (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
        return s.substr(1, s.size() - 2);
    }
    return std::string();
}

}  // namespace fitness
