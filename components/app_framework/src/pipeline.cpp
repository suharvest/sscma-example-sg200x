/**
 * @file pipeline.cpp
 * @brief Non-template parts of the pipeline: names and the default clock.
 */
#include "af/pipeline.h"

#include <ctime>

namespace af {

const char* to_string(Flow f) noexcept {
    switch (f) {
        case Flow::Next:
            return "next";
        case Flow::Reject:
            return "reject";
        case Flow::Error:
            return "error";
    }
    return "unknown";
}

uint64_t monotonic_us() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000ull;
}

}  // namespace af
