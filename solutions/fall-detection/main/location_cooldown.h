#ifndef _FALL_LOCATION_COOLDOWN_H_
#define _FALL_LOCATION_COOLDOWN_H_

// Same-location alarm cooldown, ported from the reCamera Pro reference
// (apps/fall-detection/app.py, _location_cooldown_hit). Free of SSCMA headers
// so it builds in host tests.

#include <cstdint>
#include <vector>

#include "fall_detector.h"
#include "norm_box.h"

namespace fall {

// Centre distance <= 0.75 x the largest side of either box. Boxes are in the
// normalized inference frame (square model input).
bool boxesNear(const geometry::InferBox& a, const geometry::InferBox& b);

// The per-track FallDetector cooldown dies with its track, and the tracker can
// reissue ids for a person still on the floor (partial boxes, short misses).
// A fall edge is suppressed when an event EMITTED within window_sec lies near
// it and that event's track is not visible in the current frame. An event
// from a still-visible track, or from this same frame, belongs to a different
// person. Suppressed edges are not remembered, so they do not extend the
// window.
// Per-track publishing state. Published fields follow what was emitted, not
// the detector's raw counter.
struct CooldownTrackState {
    std::uint64_t emitted_event_id = 0;     // last event actually emitted
    std::uint64_t suppressed_event_id = 0;  // last edge hidden by the cooldown
    bool fall_suppressed = false;           // detector is in a suppressed episode
    bool suppressed_edge = false;           // an edge was suppressed this frame
};

class LocationCooldown {
public:
    static constexpr float kMaxWindowSec = 120.0f;
    static constexpr std::size_t kMaxRemembered = 32;

    explicit LocationCooldown(float window_sec = 10.0f) { setWindow(window_sec); }

    // Clamped to [0, kMaxWindowSec]; 0 disables and forgets remembered events.
    void setWindow(float window_sec);
    float window() const { return window_sec_; }
    void reset() { recent_.clear(); }

    bool hit(std::uint64_t track_id, const geometry::InferBox& box, double now_sec,
             const std::vector<std::uint64_t>& visible_track_ids);
    void remember(std::uint64_t track_id, const geometry::InferBox& box, double now_sec);

    // Apply the cooldown to one track's fresh detector output, in place. A
    // suppressed edge clears fall_event; while the detector stays in that
    // episode fall_detected is reported false and state.fall_suppressed true.
    // out.event_id is replaced by the track's last emitted id (0 if none), so
    // an unemitted event id is never published. The detector itself keeps its
    // Fallen state and its own cooldown.
    void apply(std::uint64_t track_id, const geometry::InferBox& box, double now_sec,
               const std::vector<std::uint64_t>& visible_track_ids,
               FallOutput& out, CooldownTrackState& state);

private:
    struct Event {
        double timestamp_sec;
        geometry::InferBox box;
        std::uint64_t track_id;
    };
    float window_sec_ = 10.0f;
    std::vector<Event> recent_;
};

}  // namespace fall

#endif  // _FALL_LOCATION_COOLDOWN_H_
