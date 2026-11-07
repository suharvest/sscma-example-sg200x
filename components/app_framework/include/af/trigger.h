/**
 * @file trigger.h
 * @brief External triggers (GPIO edge, HTTP request) as poll()-able sources.
 *
 * A TriggerSource exposes one file descriptor. The App's event-loop thread
 * polls it and calls on_ready() when it is readable; the source turns that
 * into zero or one TriggerEvent. Sources never spawn threads and never call
 * user code directly -- the App queues the event and hands it to the
 * on_trigger() handler on the frame thread together with the next frame.
 *
 * Writing a new source (MQTT command, UART, timer...) means implementing the
 * four virtuals below; the App needs no change.
 */
#ifndef AF_TRIGGER_H_
#define AF_TRIGGER_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace af {

/** One trigger occurrence. Trivially copyable, fixed size. */
struct TriggerEvent {
    enum class Source : uint8_t { Gpio, Http, Manual };

    static constexpr size_t kTagLen = 32;

    Source source = Source::Manual;
    uint8_t level = 0;      ///< GPIO: logical level after the edge (1 = active)
    uint16_t line = 0;      ///< GPIO: line number; HTTP: listening port
    uint32_t seq = 0;       ///< per-source counter, assigned by the source
    uint64_t ts_us = 0;     ///< monotonic microseconds when the trigger was observed
    char tag[kTagLen] = {}; ///< free-form NUL-terminated label (HTTP: "?tag=" value)
};

/** Name of a trigger source kind ("gpio", "http", "manual"). */
const char* to_string(TriggerEvent::Source s) noexcept;

/** Interface implemented by every trigger. Not copyable. */
class TriggerSource {
public:
    TriggerSource() = default;
    TriggerSource(const TriggerSource&) = delete;
    TriggerSource& operator=(const TriggerSource&) = delete;
    virtual ~TriggerSource() = default;

    /** Short name for logs. */
    virtual const char* name() const noexcept = 0;

    /**
     * Acquire the underlying resource.
     * @return file descriptor to poll, or -1 on failure (the App logs and
     *         continues without this trigger).
     */
    virtual int open() noexcept = 0;

    /** poll() events to wait for (POLLIN, POLLPRI...). */
    virtual short poll_events() const noexcept = 0;

    /**
     * Called on the event-loop thread when the fd is ready.
     * @param revents poll() result bits.
     * @param out     filled when a trigger fired.
     * @return true when @p out holds a new event (debounced / accepted).
     */
    virtual bool on_ready(short revents, TriggerEvent& out) noexcept = 0;

    /**
     * The App finished handling @p ev; @p body is the Result::json (may be
     * empty) or a small error document. Called from the frame thread or, when
     * the queue is full, the event-loop thread. Default: nothing to answer.
     */
    virtual void complete(const TriggerEvent& ev, std::string_view body) noexcept {
        (void)ev;
        (void)body;
    }

    /** Release the resource. Safe to call when not open. */
    virtual void close() noexcept = 0;
};

/**
 * Edge-triggered GPIO input via the sysfs GPIO interface
 * (/sys/class/gpio/gpioN/{direction,edge,value}).
 *
 * The line is exported on open() if needed and left exported on close() so a
 * concurrent user is not disturbed.
 */
class GpioTrigger final : public TriggerSource {
public:
    enum class Edge : uint8_t { Rising, Falling, Both };

    struct Config {
        int line = -1;               ///< global sysfs GPIO number
        Edge edge = Edge::Rising;    ///< edge reported as a trigger (in logical terms)
        bool active_low = false;     ///< invert the level (button to GND with pull-up)
        uint32_t debounce_ms = 50;   ///< ignore edges closer than this to the last accepted one
    };

    explicit GpioTrigger(const Config& cfg) noexcept : cfg_(cfg) {}
    ~GpioTrigger() override { close(); }

    const char* name() const noexcept override { return "gpio"; }
    int open() noexcept override;
    short poll_events() const noexcept override;
    bool on_ready(short revents, TriggerEvent& out) noexcept override;
    void close() noexcept override;

private:
    Config cfg_;
    int fd_ = -1;
    uint32_t seq_ = 0;
    uint64_t last_us_ = 0;
};

/**
 * Minimal HTTP trigger endpoint: `GET|POST <path>[?tag=<label>]` on its own
 * port. Default: answers 202 `{"accepted":true,"seq":N}` at once. With
 * sync_response the connection is held until the App calls complete() and the
 * caller gets 200 with the verdict JSON (one request in flight; a second gets
 * 409). Everything else gets 404.
 *
 * One request per connection, handled synchronously on the event-loop thread
 * with a short receive timeout, which is enough for a PLC / Node-RED / curl
 * caller and avoids pulling an HTTP library into the trigger path.
 */
class HttpTrigger final : public TriggerSource {
public:
    struct Config {
        uint16_t port = 8090;           ///< TCP port to listen on
        const char* path = "/trigger";  ///< request path (static string)
        const char* bind = "0.0.0.0";   ///< IPv4 bind address (static string)
        uint32_t recv_timeout_ms = 200; ///< per-connection read timeout
        bool sync_response = false;     ///< hold the connection and answer with the result
        uint32_t send_timeout_ms = 1000;///< socket send timeout for the held connection
    };

    explicit HttpTrigger(const Config& cfg) noexcept : cfg_(cfg) {}
    ~HttpTrigger() override { close(); }

    const char* name() const noexcept override { return "http"; }
    int open() noexcept override;
    short poll_events() const noexcept override;
    bool on_ready(short revents, TriggerEvent& out) noexcept override;
    void complete(const TriggerEvent& ev, std::string_view body) noexcept override;
    void close() noexcept override;

private:
    Config cfg_;
    int fd_ = -1;
    uint32_t seq_ = 0;
    std::mutex mu_;             ///< guards the held connection (event-loop vs frame thread)
    int held_fd_ = -1;          ///< sync_response: connection awaiting complete()
    uint32_t held_seq_ = 0;
};

}  // namespace af

#endif  // AF_TRIGGER_H_
