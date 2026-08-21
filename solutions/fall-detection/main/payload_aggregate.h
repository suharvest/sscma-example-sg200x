#ifndef _FALL_PAYLOAD_AGGREGATE_H_
#define _FALL_PAYLOAD_AGGREGATE_H_

#include <cstdint>
#include <vector>

#include "fall_detector.h"

namespace fall {

// Device-independent projection of one retained track used to verify the
// legacy top-level MQTT aggregate without pulling in SSCMA/pose headers.
struct PayloadPersonSummary {
    std::uint64_t event_id = 0;
    FallState state = FallState::Normal;
    bool fall_detected = false;
    bool fall_event = false;
};

struct PayloadAggregate {
    bool fall_detected = false;
    bool fall_event = false;
    int person_count = 0;
    int fallen_count = 0;
    std::uint64_t event_id = 0;
    FallState state = FallState::Normal;
};

PayloadAggregate aggregatePayload(const std::vector<PayloadPersonSummary>& persons,
                                  int active_person_count,
                                  std::uint64_t global_event_id = 0,
                                  bool global_event_id_valid = false);

}  // namespace fall

#endif  // _FALL_PAYLOAD_AGGREGATE_H_
