#include "location_cooldown.h"

#include <algorithm>
#include <cmath>

namespace fall {

bool boxesNear(const geometry::InferBox& a, const geometry::InferBox& b) {
    const float side = std::max({a.w, a.h, b.w, b.h, 0.0f});
    return std::hypot(a.cx - b.cx, a.cy - b.cy) <= 0.75f * side;
}

void LocationCooldown::setWindow(float window_sec) {
    if (!std::isfinite(window_sec)) window_sec = 0.0f;
    window_sec_ = std::clamp(window_sec, 0.0f, kMaxWindowSec);
    if (window_sec_ <= 0.0f) recent_.clear();
}

bool LocationCooldown::hit(std::uint64_t track_id, const geometry::InferBox& box,
                           double now_sec,
                           const std::vector<std::uint64_t>& visible_track_ids) {
    if (window_sec_ <= 0.0f) {
        recent_.clear();
        return false;
    }
    recent_.erase(std::remove_if(recent_.begin(), recent_.end(), [&](const Event& e) {
        const double age = now_sec - e.timestamp_sec;
        return age < 0.0 || age > window_sec_;
    }), recent_.end());
    for (const auto& e : recent_) {
        if (e.timestamp_sec >= now_sec || e.track_id == track_id) continue;
        if (std::find(visible_track_ids.begin(), visible_track_ids.end(), e.track_id) !=
            visible_track_ids.end()) continue;
        if (boxesNear(e.box, box)) return true;
    }
    return false;
}

void LocationCooldown::remember(std::uint64_t track_id, const geometry::InferBox& box,
                                double now_sec) {
    if (window_sec_ <= 0.0f) return;
    recent_.push_back({now_sec, box, track_id});
    if (recent_.size() > kMaxRemembered) {
        recent_.erase(recent_.begin(), recent_.end() - kMaxRemembered);
    }
}

void LocationCooldown::apply(std::uint64_t track_id, const geometry::InferBox& box,
                             double now_sec,
                             const std::vector<std::uint64_t>& visible_track_ids,
                             FallOutput& out, CooldownTrackState& state) {
    state.suppressed_edge = false;
    if (out.fall_event) {
        if (hit(track_id, box, now_sec, visible_track_ids)) {
            state.suppressed_event_id = out.event_id;
            state.suppressed_edge = true;
            out.fall_event = false;
        } else {
            state.emitted_event_id = out.event_id;
            remember(track_id, box, now_sec);
        }
    }
    state.fall_suppressed = out.fall_detected && state.suppressed_event_id != 0 &&
                            state.suppressed_event_id == out.event_id;
    if (state.fall_suppressed) out.fall_detected = false;
    out.event_id = state.emitted_event_id;
}

}  // namespace fall
