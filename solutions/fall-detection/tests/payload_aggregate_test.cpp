#include "../main/payload_aggregate.h"

#include <cassert>
#include <iostream>
#include <vector>

using fall::FallState;
using fall::PayloadPersonSummary;
using fall::aggregatePayload;

int main() {
    // Active people are counted from this frame, while fallen_count and
    // severity include retained occluded tracks. The stream-global sequence
    // wins over colliding per-track event counters.
    const std::vector<PayloadPersonSummary> persons = {
        {1, FallState::Fallen, true, false},
        {1, FallState::Suspected, false, true},
        {0, FallState::Normal, false, false},
    };
    const auto aggregate = aggregatePayload(persons, 2, 7, true);
    assert(aggregate.person_count == 2);
    assert(aggregate.fallen_count == 1);
    assert(aggregate.fall_detected);
    assert(aggregate.fall_event);
    assert(aggregate.state == FallState::Fallen);
    assert(aggregate.event_id == 7);

    // Legacy callers without a tracker context retain max(per-track ID)
    // semantics and still aggregate the most severe state.
    const auto legacy = aggregatePayload(persons, 1, 0);
    assert(legacy.event_id == 1);
    assert(legacy.state == FallState::Fallen);

    // A live stream with no event yet still has a valid global sequence (0),
    // which must not silently switch the field back to the legacy max rule.
    const auto first_frame = aggregatePayload(persons, 2, 0, true);
    assert(first_frame.event_id == 0);

    std::cout << "payload_aggregate_test: all scenarios passed\n";
    return 0;
}
