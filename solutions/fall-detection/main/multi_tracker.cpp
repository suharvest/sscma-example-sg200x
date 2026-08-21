#include "multi_tracker.h"

#include "box_tracker.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace fall {
namespace {

bool midpoint(const Pose& pose, Joint a, Joint b, Point2f& out) {
    const bool va = pose.visible(a);
    const bool vb = pose.visible(b);
    if (!va && !vb) return false;
    if (va && vb) {
        out.x = (pose.at(a).x + pose.at(b).x) * 0.5f;
        out.y = (pose.at(a).y + pose.at(b).y) * 0.5f;
    } else {
        out = va ? pose.at(a) : pose.at(b);
    }
    return true;
}

}  // namespace

FallObservation observationFromSubject(const Subject& subject, double timestamp_sec,
                                       int inference_height) {
    FallObservation observation;
    observation.timestamp_sec = timestamp_sec;
    observation.person_score = subject.score;
    if (subject.pose.empty() || inference_height <= 0 || subject.box.h <= 1e-4f) {
        return observation;
    }

    Point2f hips;
    Point2f shoulders;
    if (!midpoint(subject.pose, Joint::LeftHip, Joint::RightHip, hips) ||
        !midpoint(subject.pose, Joint::LeftShoulder, Joint::RightShoulder, shoulders)) {
        return observation;
    }

    const float dy = hips.y - shoulders.y;
    const float dx = hips.x - shoulders.x;
    const float torso_angle = std::atan2(std::fabs(dx), std::fabs(dy)) *
                              180.0f / 3.14159265358979323846f;
    if (!std::isfinite(torso_angle)) return observation;

    observation.valid = true;
    observation.hip_y = hips.y / static_cast<float>(inference_height);
    observation.torso_angle_deg = torso_angle;
    observation.bbox_aspect_ratio = subject.box.w / subject.box.h;
    return observation;
}

MultiPersonTracker::MultiPersonTracker(TrackerConfig config) : config_(std::move(config)) {
    config_.iou_threshold = std::clamp(config_.iou_threshold, 0.0f, 1.0f);
    config_.center_distance_threshold = std::max(0.0f, config_.center_distance_threshold);
    config_.max_missed_frames = std::max(1, config_.max_missed_frames);
    config_.timeout_sec = std::max(0.0f, config_.timeout_sec);
}

void MultiPersonTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    global_event_id_ = 0;
    event_edge_count_ = 0;
}

void MultiPersonTracker::setFallConfig(const FallConfig& config) {
    config_.fall = config;
    for (auto& track : tracks_) track->fall.setConfig(config);
}

void MultiPersonTracker::setTimeout(float timeout_sec, int max_missed_frames) {
    config_.timeout_sec = std::max(0.0f, timeout_sec);
    if (max_missed_frames >= 0) config_.max_missed_frames = std::max(1, max_missed_frames);
}

void MultiPersonTracker::updateVisible(TrackedPerson& track, const Subject& subject,
                                       double timestamp_sec, int inference_width,
                                       int inference_height) {
    track.subject = subject;
    track.visible = true;
    track.missed = 0;
    ++track.age;
    track.last_seen_sec = timestamp_sec;
    track.observation = observationFromSubject(track.subject, timestamp_sec, inference_height);
    const TemporalPrediction temporal = track.temporal.update(
        makeTemporalFrame(&track.subject.pose, track.observation,
                          inference_width, inference_height), timestamp_sec);
    track.observation.temporal_available = true;
    track.observation.temporal_positive = temporal.positive;
    track.observation.temporal_probability = temporal.probability;
    track.output = track.fall.update(track.observation);
    track.updated_this_frame = true;
}

void MultiPersonTracker::updateOccluded(TrackedPerson& track, double timestamp_sec) {
    track.visible = false;
    ++track.missed;
    track.observation = FallObservation{};
    track.observation.timestamp_sec = timestamp_sec;
    const TemporalPrediction temporal = track.temporal.update(TemporalFrame{}, timestamp_sec);
    track.observation.temporal_available = true;
    track.observation.temporal_positive = temporal.positive;
    track.observation.temporal_probability = temporal.probability;
    track.output = track.fall.update(track.observation);
    track.updated_this_frame = true;
}

std::vector<TrackedPerson*> MultiPersonTracker::update(
    const std::vector<Subject>& detections, double timestamp_sec,
    int inference_width, int inference_height) {
    // Retire tracks before matching as well as after advancing their state.
    // Without this guard, a detector result arriving after a long camera gap
    // could match an old box and silently inherit its event history instead of
    // receiving a fresh track_id.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const auto& track) {
        const double elapsed = timestamp_sec - track->last_seen_sec;
        return track->missed > config_.max_missed_frames ||
               elapsed > config_.timeout_sec;
    }), tracks_.end());

    for (auto& track : tracks_) track->updated_this_frame = false;

    std::vector<geometry::InferBox> detection_boxes;
    detection_boxes.reserve(detections.size());
    for (const auto& detection : detections) detection_boxes.push_back(detection.box);

    std::vector<geometry::InferBox> track_boxes;
    track_boxes.reserve(tracks_.size());
    for (const auto& track : tracks_) track_boxes.push_back(track->subject.box);

    const std::vector<int> assignment = greedyBoxAssignment(
        detection_boxes, track_boxes, config_.iou_threshold,
        config_.center_distance_threshold);

    for (std::size_t d = 0; d < detections.size(); ++d) {
        TrackedPerson* track = nullptr;
        if (assignment[d] >= 0) {
            track = tracks_[static_cast<std::size_t>(assignment[d])].get();
        } else {
            auto fresh = std::make_unique<TrackedPerson>();
            fresh->track_id = next_id_++;
            fresh->fall.setConfig(config_.fall);
            tracks_.push_back(std::move(fresh));
            track = tracks_.back().get();
        }
        updateVisible(*track, detections[d], timestamp_sec,
                      inference_width, inference_height);
    }

    // Keep each track's state machine advancing through a short detector miss.
    // FallDetector treats the invalid observation as an occlusion, while the
    // temporal window receives an explicit blank frame rather than stale pose.
    for (auto& track : tracks_) {
        if (!track->updated_this_frame) updateOccluded(*track, timestamp_sec);
    }

    // FallDetector emits a one-frame edge. Assign a stream-level sequence
    // before any stale track can be retired so aggregate MQTT event_id never
    // collides when two independent tracks both report their first event.
    bool frame_edge = false;
    for (const auto& track : tracks_) {
        if (!track->output.fall_event) continue;
        ++global_event_id_;
        frame_edge = true;
    }
    if (frame_edge) ++event_edge_count_;

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const auto& track) {
        const double elapsed = timestamp_sec - track->last_seen_sec;
        const bool too_old = elapsed > config_.timeout_sec;
        // Keep an edge-bearing frame visible to the publisher even if the
        // timeout boundary and the fall confirmation happen together.
        return (track->missed > config_.max_missed_frames || too_old) &&
               !track->output.fall_event;
    }), tracks_.end());

    std::vector<TrackedPerson*> retained;
    retained.reserve(tracks_.size());
    for (auto& track : tracks_) retained.push_back(track.get());
    return retained;
}

int MultiPersonTracker::activeCount() const {
    int count = 0;
    for (const auto& track : tracks_) {
        if (track->visible) ++count;
    }
    return count;
}

}  // namespace fall
