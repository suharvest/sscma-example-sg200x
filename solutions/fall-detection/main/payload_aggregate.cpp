#include "payload_aggregate.h"

#include <algorithm>

namespace fall {
namespace {

int severity(FallState state) {
    switch (state) {
        case FallState::Fallen: return 3;
        case FallState::Recovering: return 2;
        case FallState::Suspected: return 1;
        case FallState::Normal: return 0;
    }
    return 0;
}

}  // namespace

PayloadAggregate aggregatePayload(const std::vector<PayloadPersonSummary>& persons,
                                  int active_person_count,
                                  std::uint64_t global_event_id,
                                  bool global_event_id_valid) {
    PayloadAggregate result;
    result.person_count = std::max(0, active_person_count);
    for (const auto& person : persons) {
        result.fall_detected = result.fall_detected || person.fall_detected;
        result.fall_event = result.fall_event || person.fall_event;
        if (person.fall_detected) ++result.fallen_count;
        result.event_id = std::max(result.event_id, person.event_id);
        if (severity(person.state) > severity(result.state)) result.state = person.state;
    }
    if (global_event_id_valid) result.event_id = global_event_id;
    return result;
}

}  // namespace fall
