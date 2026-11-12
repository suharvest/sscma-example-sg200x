/**
 * @file app.cpp
 * @brief App assembly: service bring-up, the frame loop, trigger queue and built-in sinks.
 */
#include "af/app.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <string>
#include <vector>

#include <debug_stream.h>
#include <ha_mqtt.h>
#include <onvif_service.h>
#include <sscma.h>

#include "camera_hub.h"
#include "event_loop.h"

namespace af {

namespace {

constexpr const char* TAG = "af.app";
constexpr size_t kMaxTriggers = EventLoop::kMaxWatches - 1;
constexpr size_t kMaxExtraSinks = 6;
constexpr size_t kPendingTriggers = 8;
constexpr int kGrabRetrySleepMs = 10;
/** Upper bound for getting a post-trigger frame (covers the fps switch plus one frame period). */
constexpr uint32_t kTriggerGrabTimeoutMs = 2000;
constexpr const char kNoFrameBody[] = "{\"accepted\":true,\"error\":\"no_frame\"}";
constexpr const char kBusyBody[] = "{\"accepted\":false,\"error\":\"busy\"}";

/** The App whose run() is active; the signal handler stops it. */
std::atomic<App*> g_running_app{nullptr};

void on_signal(int) {
    if (App* app = g_running_app.load()) {
        app->stop();
    }
}

/** Result pre-filled with the frame identity; handlers fill the rest. */
Result make_result(const VideoFrame& f) noexcept {
    Result r;
    r.seq = f.seq;
    r.ts_ms = f.ts_ms;
    r.width = f.width;
    r.height = f.height;
    r.wall_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    return r;
}

/** Debug WebSocket /results channel. Builds the envelope only when a client is connected. */
class DebugWsSink final : public ResultSink {
public:
    DebugWsSink(int display_w, int display_h) : display_w_(display_w), display_h_(display_h) {}

    const char* name() const noexcept override { return "debug_ws"; }

    /** Only /results subscribers consume this; video-only viewers do not need the JSON. */
    bool active() const noexcept override { return debug_stream_results_client_count() > 0; }

    void publish(const Result& r) noexcept override {
        try {
            // Allocation here is accepted: it only happens while a debug client watches.
            boxes_.clear();
            labels_.clear();
            bool has_detail = false;
            for (size_t i = 0; i < r.box_count; ++i) {
                const Box& b = r.boxes[i];
                boxes_.push_back(debug_stream_box_t{b.x, b.y, b.w, b.h, b.score, b.label ? std::string(b.label) : std::to_string(b.cls)});
                has_detail = has_detail || b.detail != nullptr;
            }
            if (has_detail) {
                for (size_t i = 0; i < r.box_count; ++i) {
                    labels_.push_back(r.boxes[i].detail ? std::string(r.boxes[i].detail) : boxes_[i].label);
                }
            }
            int res_w = r.width;
            int res_h = r.height;
            if (display_w_ > 0 && display_h_ > 0) {
                debug_stream_letterbox_to_display(boxes_, r.width, r.height, display_w_, display_h_);
                res_w = display_w_;
                res_h = display_h_;
            }
            const std::string json =
                debug_stream_build_results(r.wall_ms, r.seq, r.inference_ms, res_w, res_h, boxes_, has_detail ? &labels_ : nullptr,
                                           std::string(r.debug_extra));
            debug_stream_publish_result(json.data(), json.size());
        } catch (...) {
            // Out of memory while building debug output must not take the app down.
        }
    }

private:
    int display_w_;
    int display_h_;
    std::vector<debug_stream_box_t> boxes_;
    std::vector<std::string> labels_;
};

/** MQTT results topic. Publishes Result::json verbatim (external contract). */
class MqttSink final : public ResultSink {
public:
    explicit MqttSink(ha_mqtt::MqttPublisher& pub) : pub_(pub) {}

    const char* name() const noexcept override { return "mqtt"; }

    bool active() const noexcept override { return pub_.isConnected(); }

    void publish(const Result& r) noexcept override {
        if (r.json.empty()) {
            return;
        }
        try {
            pub_.publishResultsJson(std::string(r.json));
        } catch (...) {
        }
    }

private:
    ha_mqtt::MqttPublisher& pub_;
};

}  // namespace

struct App::Impl {
    explicit Impl(const char* n) : name(n) {}

    const char* name;
    HubConfig hub_cfg;
    bool camera_set = false;

    std::unique_ptr<ha_mqtt::ClientOptions> mqtt_opts;
    std::unique_ptr<ha_mqtt::MqttPublisher> mqtt;
    std::unique_ptr<OnvifOptions> onvif_opts;
    bool onvif_running = false;

    TriggerSource* triggers[kMaxTriggers] = {};
    size_t trigger_count = 0;

    ResultSink* sinks[kMaxExtraSinks + 2] = {};  // + debug ws + mqtt
    size_t sink_count = 0;
    std::unique_ptr<DebugWsSink> debug_sink;
    std::unique_ptr<MqttSink> mqtt_sink;

    std::function<void()> on_started;
    std::function<void()> on_stopping;
    std::function<void()> on_idle;
    std::function<void()> on_before_grab;
    std::function<void(const VideoFrame&)> on_snapshot;
    FrameHandler on_frame;
    TriggerHandler on_trigger;

    CameraHub hub;
    EventLoop loop;

    std::atomic<bool> stop{false};
    std::atomic<uint32_t> frames{0};
    std::atomic<uint8_t> fps_request{0};  ///< 0: no change requested

    /** A queued trigger and the source to answer (nullptr for App::fire()). */
    struct Pending {
        TriggerEvent ev;
        TriggerSource* src = nullptr;
    };

    // Pending trigger ring, filled by the event-loop thread and fire(), drained by the frame thread.
    std::mutex pending_mu;
    std::condition_variable pending_cv;
    Pending pending[kPendingTriggers];
    size_t pending_head = 0;
    size_t pending_size = 0;

    bool push_pending(const Pending& p) noexcept {
        {
            std::lock_guard<std::mutex> lock(pending_mu);
            if (pending_size == kPendingTriggers) {
                return false;
            }
            pending[(pending_head + pending_size) % kPendingTriggers] = p;
            ++pending_size;
        }
        pending_cv.notify_one();
        return true;
    }

    bool pop_pending_locked(Pending& p) noexcept {
        if (pending_size == 0) {
            return false;
        }
        p = pending[pending_head];
        pending_head = (pending_head + 1) % kPendingTriggers;
        --pending_size;
        return true;
    }

    bool pop_pending(Pending& p) noexcept {
        std::lock_guard<std::mutex> lock(pending_mu);
        return pop_pending_locked(p);
    }

    /** Block up to @p timeout_ms for a trigger. Not woken by stop(); the timeout bounds the shutdown delay. */
    bool wait_pending(Pending& p, uint32_t timeout_ms) {
        std::unique_lock<std::mutex> lock(pending_mu);
        pending_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return pending_size != 0; });
        return pop_pending_locked(p);
    }

    /** Answer the source that raised @p p (synchronous HTTP verdicts). */
    static void complete(const Pending& p, std::string_view body) noexcept {
        if (p.src != nullptr) {
            p.src->complete(p.ev, body);
        }
    }

    /** Event-loop callback: let the source decode the readiness, queue what it reports. */
    static void on_trigger_fd(int, short revents, void* user) {
        auto* ctx = static_cast<std::pair<Impl*, TriggerSource*>*>(user);
        Pending p;
        p.src = ctx->second;
        if (ctx->second->on_ready(revents, p.ev) && !ctx->first->push_pending(p)) {
            MA_LOGW(TAG, "Trigger queue full, dropped %s #%u", ctx->second->name(), p.ev.seq);
            complete(p, kBusyBody);
        }
    }
    std::pair<Impl*, TriggerSource*> trigger_ctx[kMaxTriggers];

    /** Apply a set_camera_fps() request on the frame thread. Returns the new fps or 0. */
    uint8_t apply_fps_request() noexcept {
        const uint8_t fps = fps_request.exchange(0);
        if (fps != 0) {
            hub.set_fps(fps);
        }
        return fps;
    }

    /** Masking hook + snapshot offer; must run while the frame is still held. */
    void offer_snapshot(const VideoFrame& frame) {
        if (!hub.snapshot_enabled() || !debug_stream_snapshot_armed()) {
            return;
        }
        const uint8_t* rgb = hub.held_rgb();
        if (rgb == nullptr) {
            return;
        }
        if (on_snapshot) {
            on_snapshot(frame);  // e.g. privacy masking, in place, before the JPEG encode
        }
        debug_stream_offer_snapshot(rgb, frame.width, frame.height);
    }

    void run_continuous();
    void run_triggered();
    bool start_mqtt();
    void start_onvif();
    void start_triggers();
    void publish(const Result& r) noexcept;
    void shutdown();
};

bool App::Impl::start_mqtt() {
    if (!mqtt_opts) {
        return true;
    }
    mqtt = std::make_unique<ha_mqtt::MqttPublisher>();
    if (!mqtt->init(*mqtt_opts)) {
        MA_LOGW(TAG, "MQTT init failed, continuing without it");
        mqtt.reset();
        return false;
    }
    mqtt_sink = std::make_unique<MqttSink>(*mqtt);
    sinks[sink_count++] = mqtt_sink.get();
    return true;
}

void App::Impl::start_onvif() {
    if (!onvif_opts) {
        return;
    }
    if (!hub.rtsp_running()) {
        MA_LOGW(TAG, "ONVIF requires RTSP; skipped");
        return;
    }
    onvif_service_config sc;
    onvif_service_config_init(&sc);
    sc.service_port = onvif_opts->port;
    sc.serial = onvif_opts->serial;
    sc.username = onvif_opts->username;
    sc.password = onvif_opts->password;
    sc.location = onvif_opts->location;
    if (!onvif_opts->device_name.empty()) {
        sc.device_name = onvif_opts->device_name;
    }
    sc.snapshot_port = hub.snapshot_enabled() ? hub.debug_port() : 0;  // 0 -> GetSnapshotUri answers "not supported"
    onvif_running = onvif_service_start(&sc) == 0;
    if (!onvif_running) {
        MA_LOGW(TAG, "ONVIF start failed, continuing without it");
    }
}

void App::Impl::start_triggers() {
    for (size_t i = 0; i < trigger_count; ++i) {
        TriggerSource* t = triggers[i];
        const int fd = t->open();
        trigger_ctx[i] = {this, t};
        if (fd < 0 || !loop.watch(fd, t->poll_events(), &Impl::on_trigger_fd, &trigger_ctx[i])) {
            MA_LOGW(TAG, "Trigger %s unavailable", t->name());
            t->close();
            continue;
        }
        MA_LOGI(TAG, "Trigger %s armed (fd %d)", t->name(), fd);
    }
    if (loop.watch_count() > 0 && !loop.start()) {
        MA_LOGW(TAG, "Trigger event loop failed to start");
    }
}

void App::Impl::publish(const Result& r) noexcept {
    if (!r.publish) {
        return;
    }
    for (size_t i = 0; i < sink_count; ++i) {
        if (sinks[i]->active()) {
            sinks[i]->publish(r);
        }
    }
}

void App::Impl::shutdown() {
    loop.stop();
    if (on_stopping && hub.streaming()) {
        on_stopping();  // before the video pipeline goes away (e.g. RGN detach)
    }
    for (size_t i = 0; i < trigger_count; ++i) {
        triggers[i]->close();
    }
    if (onvif_running) {
        onvif_service_stop();
        onvif_running = false;
    }
    hub.close();
    if (mqtt) {
        mqtt->deinit();
        mqtt.reset();
    }
    // Drop built-in sinks; extra sinks stay registered for a later run().
    size_t kept = 0;
    for (size_t i = 0; i < sink_count; ++i) {
        if (sinks[i] != debug_sink.get() && sinks[i] != mqtt_sink.get()) {
            sinks[kept++] = sinks[i];
        }
    }
    sink_count = kept;
    debug_sink.reset();
    mqtt_sink.reset();
    MA_LOGI(TAG, "%s stopped after %u frames", name, frames.load());
}

void App::Impl::run_continuous() {
    VideoFrame frame;
    while (!stop.load(std::memory_order_relaxed)) {
        apply_fps_request();

        // Pop before grabbing so the frame handed to on_trigger is retrieved after the event.
        Pending p;
        const bool triggered = on_trigger && pop_pending(p);

        if (on_before_grab) {
            on_before_grab();
        }
        if (!hub.grab(frame)) {
            if (triggered) {
                push_pending(p);  // retry with the next frame
            }
            if (on_idle) {
                on_idle();
            }
            ma::Thread::sleep(ma::Tick::fromMilliseconds(kGrabRetrySleepMs));
            continue;
        }
        frames.fetch_add(1, std::memory_order_relaxed);

        Result frame_result = make_result(frame);
        Result trigger_result = frame_result;
        on_frame(frame, frame_result);
        if (triggered) {
            on_trigger(p.ev, frame, trigger_result);
        }

        offer_snapshot(frame);
        // Return the buffer before fan-out: sinks only read Result, and
        // publishing (JSON, sockets) should not hold a camera buffer.
        hub.release();

        publish(frame_result);
        if (triggered) {
            publish(trigger_result);
            complete(p, trigger_result.json);
        }
    }
}

/*
 * Trigger-only mode (no on_frame handler).
 *
 * Standby: the frame thread sleeps on the trigger queue. The camera channel is
 * dropped to idle_fps, and once per idle period one frame is retrieved and
 * returned -- the camera callback blocks on a full queue while holding a VPSS
 * buffer, so the queue must not be left to fill. That same frame serves an
 * armed /snapshot.jpg. Cost in standby: idle_fps copies of the inference frame
 * per second in the camera callback; nothing else runs.
 *
 * Trigger: switch to the active fps, wait for a frame whose capture tick is not
 * older than the trigger, run on_trigger once, publish, answer the source,
 * return to idle_fps.
 */
void App::Impl::run_triggered() {
    const uint8_t active_fps = hub_cfg.camera.fps;
    uint8_t idle_fps = hub_cfg.camera.idle_fps;
    if (idle_fps != 0 && idle_fps != active_fps) {
        hub.set_fps(idle_fps);
    }
    VideoFrame frame;
    while (!stop.load(std::memory_order_relaxed)) {
        if (const uint8_t f = apply_fps_request()) {
            idle_fps = f;  // in this mode set_camera_fps() moves the standby rate
        }
        const uint8_t standby_fps = idle_fps != 0 ? idle_fps : active_fps;

        Pending p;
        if (!wait_pending(p, 1000u / standby_fps)) {
            if (hub.grab(frame)) {
                offer_snapshot(frame);
                hub.release();
            } else if (on_idle) {
                on_idle();
            }
            continue;
        }

        if (standby_fps != active_fps) {
            hub.set_fps(active_fps);
        }
        const uint32_t mark = hub.mark();
        if (on_before_grab) {
            on_before_grab();
        }
        if (!hub.grab_after(mark, frame, kTriggerGrabTimeoutMs)) {
            MA_LOGW(TAG, "Trigger %s #%u: no frame within %ums", to_string(p.ev.source), p.ev.seq, kTriggerGrabTimeoutMs);
            complete(p, kNoFrameBody);
        } else {
            frames.fetch_add(1, std::memory_order_relaxed);
            Result r = make_result(frame);
            on_trigger(p.ev, frame, r);
            offer_snapshot(frame);
            hub.release();
            publish(r);
            complete(p, r.json);
        }
        if (standby_fps != active_fps) {
            hub.set_fps(standby_fps);
        }
    }
}

App::App(const char* name) : impl_(std::make_unique<Impl>(name)) {}

App::~App() = default;

App& App::with_camera(const CameraOptions& opt) {
    impl_->hub_cfg.camera = opt;
    impl_->camera_set = true;
    return *this;
}

App& App::with_raw_channel(const RawChannelOptions& opt, RawFrameHandler fn) {
    impl_->hub_cfg.raw = opt;
    impl_->hub_cfg.raw_fn = std::move(fn);
    return *this;
}

App& App::with_live_view(int port) {
    return with_stream_mode(StreamMode::WsOnly, StreamOptions{}, port);
}

App& App::with_stream(const StreamOptions& opt) {
    impl_->hub_cfg.streams.set_stream(opt);
    return *this;
}

App& App::with_stream_ws(const StreamWsOptions& opt) {
    impl_->hub_cfg.streams.enable_ws(opt);
    return *this;
}

App& App::with_stream_mode(StreamMode mode, const StreamOptions& stream_opt, int ws_port, int rtsp_port) {
    impl_->hub_cfg.streams.set_mode(mode, stream_opt, ws_port, rtsp_port);
    return *this;
}

App& App::with_rtsp(const RtspOptions& opt) {
    impl_->hub_cfg.streams.enable_rtsp(opt);
    return *this;
}

App& App::with_stream(const RtspOptions& opt) {
    impl_->hub_cfg.streams.set_stream_legacy(opt);
    return *this;
}

App& App::with_debug_ws(int port) {
    StreamWsOptions ws;
    ws.port = port;
    return with_stream_ws(ws);
}

StreamMode App::stream_mode() const noexcept {
    return impl_->hub_cfg.streams.mode();
}

App& App::with_mqtt(const ha_mqtt::ClientOptions& opt) {
    impl_->mqtt_opts = std::make_unique<ha_mqtt::ClientOptions>(opt);
    return *this;
}

App& App::with_onvif(const OnvifOptions& opt) {
    impl_->onvif_opts = std::make_unique<OnvifOptions>(opt);
    return *this;
}

App& App::with_trigger(TriggerSource& src) {
    if (impl_->trigger_count < kMaxTriggers) {
        impl_->triggers[impl_->trigger_count++] = &src;
    } else {
        MA_LOGW(TAG, "Too many triggers, %s ignored", src.name());
    }
    return *this;
}

App& App::with_sink(ResultSink& sink) {
    if (impl_->sink_count < kMaxExtraSinks) {
        impl_->sinks[impl_->sink_count++] = &sink;
    } else {
        MA_LOGW(TAG, "Too many sinks, %s ignored", sink.name());
    }
    return *this;
}

App& App::on_started(std::function<void()> fn) {
    impl_->on_started = std::move(fn);
    return *this;
}

App& App::on_stopping(std::function<void()> fn) {
    impl_->on_stopping = std::move(fn);
    return *this;
}

App& App::on_snapshot(std::function<void(const VideoFrame&)> fn) {
    impl_->on_snapshot = std::move(fn);
    return *this;
}

App& App::on_before_grab(std::function<void()> fn) {
    impl_->on_before_grab = std::move(fn);
    return *this;
}

App& App::on_idle(std::function<void()> fn) {
    impl_->on_idle = std::move(fn);
    return *this;
}

App& App::on_frame(FrameHandler fn) {
    impl_->on_frame = std::move(fn);
    return *this;
}

App& App::on_trigger(TriggerHandler fn) {
    impl_->on_trigger = std::move(fn);
    return *this;
}

void App::stop() noexcept {
    impl_->stop.store(true);
}

bool App::fire(const TriggerEvent& ev) noexcept {
    Impl::Pending p;
    p.ev = ev;
    return impl_->push_pending(p);
}

void App::set_camera_fps(uint8_t fps) noexcept {
    if (fps != 0) {
        impl_->fps_request.store(fps);
    }
}

bool App::any_sink_active() const noexcept {
    for (size_t i = 0; i < impl_->sink_count; ++i) {
        if (impl_->sinks[i]->active()) {
            return true;
        }
    }
    return false;
}

int App::debug_clients_count() const noexcept {
    return debug_stream_results_client_count();
}

ha_mqtt::MqttPublisher* App::mqtt() noexcept {
    return impl_->mqtt.get();
}

uint32_t App::frames() const noexcept {
    return impl_->frames.load();
}

int App::run() {
    Impl& s = *impl_;
    if (!s.on_frame && !s.on_trigger) {
        MA_LOGE(TAG, "%s: neither on_frame nor on_trigger set", s.name);
        return kExitNoHandler;
    }
    if (!s.camera_set) {
        MA_LOGI(TAG, "%s: with_camera() not called, using defaults", s.name);
    }

    s.stop.store(false);
    g_running_app.store(this);
    auto prev_int = std::signal(SIGINT, on_signal);
    auto prev_term = std::signal(SIGTERM, on_signal);

    // MQTT first: independent of the camera, and HA availability should go
    // out even if the video pipeline later fails.
    s.start_mqtt();

    int rc = s.hub.open(s.hub_cfg);
    if (rc == kExitOk) {
        if (s.hub.results_enabled()) {
            s.debug_sink = std::make_unique<DebugWsSink>(s.hub_cfg.streams.stream.width, s.hub_cfg.streams.stream.height);
            s.sinks[s.sink_count++] = s.debug_sink.get();
        }
        s.start_onvif();  // after RTSP: GetProfiles reads the live session list
        s.start_triggers();
        if (!s.hub.start()) {
            rc = kExitCamera;
        }
    }

    if (rc == kExitOk) {
        if (s.on_started) {
            s.on_started();
        }
        VideoFrame warm;
        for (uint8_t i = 0; i < s.hub_cfg.camera.warmup_frames; ++i) {
            if (s.hub.grab(warm)) {
                s.hub.release();
                ma::Thread::sleep(ma::Tick::fromMilliseconds(kGrabRetrySleepMs));
            }
        }
        if (s.on_frame) {
            MA_LOGI(TAG, "%s running (continuous)", s.name);
            s.run_continuous();
        } else {
            MA_LOGI(TAG, "%s running (trigger-only)", s.name);
            s.run_triggered();
        }
    } else {
        MA_LOGE(TAG, "%s: start-up failed (%d)", s.name, rc);
    }

    s.shutdown();
    std::signal(SIGINT, prev_int);
    std::signal(SIGTERM, prev_term);
    g_running_app.store(nullptr);
    return rc;
}

}  // namespace af
