#include "../main/location_cooldown.h"

#include <cassert>
#include <iostream>
#include <vector>

using fall::CooldownTrackState;
using fall::FallOutput;
using fall::FallState;
using fall::LocationCooldown;
using geometry::InferBox;

// Detector output for the frame on which a track confirms event `id`.
static FallOutput edge(std::uint64_t id) {
    FallOutput o;
    o.state = FallState::Fallen;
    o.fall_detected = true;
    o.fall_event = true;
    o.event_id = id;
    return o;
}

static FallOutput fallen(std::uint64_t id) {
    FallOutput o = edge(id);
    o.fall_event = false;
    return o;
}

int main() {
    const InferBox spot = InferBox::fromCenter(0.40f, 0.70f, 0.30f, 0.20f);
    const InferBox near_spot = InferBox::fromCenter(0.48f, 0.72f, 0.28f, 0.20f);
    const InferBox far_spot = InferBox::fromCenter(0.90f, 0.20f, 0.10f, 0.25f);
    assert(fall::boxesNear(spot, near_spot));
    assert(!fall::boxesNear(spot, far_spot));

    // 1. Track 1 emits at t=1. Track 1 is gone; track 2 confirms near the same
    // spot at t=5: suppressed. Published output hides the unemitted event.
    {
        LocationCooldown lc(10.0f);
        CooldownTrackState s1, s2;
        FallOutput o = edge(1);
        lc.apply(1, spot, 1.0, {1}, o, s1);
        assert(o.fall_event && o.fall_detected && o.event_id == 1 && !s1.fall_suppressed);

        o = edge(1);  // track 2's own detector counter starts at 1
        lc.apply(2, near_spot, 5.0, {2}, o, s2);
        assert(!o.fall_event && !o.fall_detected && o.event_id == 0);
        assert(s2.fall_suppressed && s2.suppressed_edge && s2.suppressed_event_id == 1);
        assert(o.state == FallState::Fallen);  // detector state is kept

        // Following frames of the same episode stay suppressed.
        o = fallen(1);
        lc.apply(2, near_spot, 5.1, {2}, o, s2);
        assert(!o.fall_event && !o.fall_detected && o.event_id == 0);
        assert(s2.fall_suppressed && !s2.suppressed_edge);

        // A later, genuinely new event on track 2 (outside the window) is
        // emitted and clears the suppressed flag.
        o = edge(2);
        lc.apply(2, near_spot, 12.0, {2}, o, s2);
        assert(o.fall_event && o.fall_detected && o.event_id == 2 && !s2.fall_suppressed);
    }

    // 2. Not suppressed while the first event's track is still visible.
    {
        LocationCooldown lc(10.0f);
        CooldownTrackState s1, s2;
        FallOutput o = edge(1);
        lc.apply(1, spot, 1.0, {1}, o, s1);
        o = edge(1);
        lc.apply(2, near_spot, 2.0, {1, 2}, o, s2);
        assert(o.fall_event && o.fall_detected && o.event_id == 1 && !s2.fall_suppressed);
    }

    // 3. Same frame: two tracks fall side by side, both emitted.
    {
        LocationCooldown lc(10.0f);
        CooldownTrackState s1, s2;
        FallOutput a = edge(1), b = edge(1);
        lc.apply(1, spot, 3.0, {}, a, s1);
        lc.apply(2, near_spot, 3.0, {}, b, s2);
        assert(a.fall_event && b.fall_event);
    }

    // 4. Not self-extending: a suppressed edge at t=9 does not restart the
    // window; at t=11.5 (> 10 s after the emitted t=1 event) a new track at the
    // same spot is emitted.
    {
        LocationCooldown lc(10.0f);
        CooldownTrackState s1, s2, s3;
        FallOutput o = edge(1);
        lc.apply(1, spot, 1.0, {1}, o, s1);
        o = edge(1);
        lc.apply(2, near_spot, 9.0, {2}, o, s2);
        assert(!o.fall_event && s2.fall_suppressed);
        o = edge(1);
        lc.apply(3, near_spot, 11.5, {3}, o, s3);
        assert(o.fall_event && o.fall_detected && o.event_id == 1 && !s3.fall_suppressed);
    }

    // 5. Far away is never suppressed; 0 disables; window clamps to 0..120.
    {
        LocationCooldown lc(10.0f);
        CooldownTrackState s1, s2;
        FallOutput o = edge(1);
        lc.apply(1, spot, 1.0, {1}, o, s1);
        o = edge(1);
        lc.apply(2, far_spot, 2.0, {2}, o, s2);
        assert(o.fall_event);

        LocationCooldown off(0.0f);
        CooldownTrackState t1, t2;
        o = edge(1);
        off.apply(1, spot, 1.0, {1}, o, t1);
        o = edge(1);
        off.apply(2, near_spot, 2.0, {2}, o, t2);
        assert(o.fall_event && !t2.fall_suppressed);

        LocationCooldown big(500.0f), neg(-3.0f);
        assert(big.window() == 120.0f && neg.window() == 0.0f);
    }

    std::cout << "location_cooldown_test: all scenarios passed\n";
    return 0;
}
