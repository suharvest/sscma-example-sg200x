/**
 * @file sink.h
 * @brief Result aggregation and the ResultSink fan-out interface.
 *
 * A frame handler fills one Result per processed frame. The App then hands
 * the same Result to every registered sink (debug WebSocket, MQTT, custom).
 * Result only points at memory owned by the handler (boxes, JSON text); it is
 * valid for the duration of the publish call, so sinks that need to keep data
 * must copy it.
 *
 * Sinks report whether anyone is listening (active()); when no sink is active
 * the App can tell the handler so it skips building JSON at all
 * (App::any_sink_active()).
 */
#ifndef AF_SINK_H_
#define AF_SINK_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "af/pipeline.h"

namespace af {

/** One detection, centre-based pixels in the inference frame. */
struct Box {
    float x = 0;  ///< centre x
    float y = 0;  ///< centre y
    float w = 0;
    float h = 0;
    float score = 0;
    int32_t cls = -1;              ///< class id, -1 when not applicable
    const char* label = nullptr;   ///< display text; nullptr falls back to cls
    const char* detail = nullptr;  ///< debug /results labels[] entry (e.g. attributes); nullptr mirrors label
};

/** Everything a sink may need about one processed frame. */
struct Result {
    uint32_t seq = 0;        ///< frame sequence number
    uint32_t ts_ms = 0;      ///< frame capture time (monotonic ms)
    uint64_t wall_ms = 0;    ///< frame capture time (Unix epoch ms), for external consumers
    uint16_t width = 0;      ///< inference frame width (box coordinate space)
    uint16_t height = 0;     ///< inference frame height
    bool publish = true;     ///< handler may clear this to suppress fan-out for this frame

    Flow flow = Flow::Next;       ///< pipeline outcome
    const char* reason = nullptr; ///< pipeline stop reason
    float inference_ms = 0;       ///< reported in the debug /results envelope

    const Box* boxes = nullptr;   ///< overlay boxes (debug stream)
    size_t box_count = 0;

    /** Application payload, published verbatim on the MQTT results topic (external contract). */
    std::string_view json;

    /** Extra top-level members appended to the debug /results envelope, e.g. "\"state\":\"ok\"". */
    std::string_view debug_extra;

    /** Copy outcome and timing from a pipeline run. */
    void attach(const RunReport& r) noexcept {
        flow = r.flow;
        reason = r.reason;
        inference_ms = static_cast<float>(r.total_us) / 1000.0f;
    }

    /** Point at a caller-owned box array. */
    void set_boxes(const Box* b, size_t n) noexcept {
        boxes = b;
        box_count = n;
    }
};

/** Destination for Results. Implementations must not block for long: publish() runs on the frame thread. */
class ResultSink {
public:
    virtual ~ResultSink() = default;

    /** Short name for logs. */
    virtual const char* name() const noexcept = 0;

    /** Whether publishing would reach anyone right now (cheap, called every frame). */
    virtual bool active() const noexcept { return true; }

    /** Deliver one result. */
    virtual void publish(const Result& r) noexcept = 0;
};

}  // namespace af

#endif  // AF_SINK_H_
