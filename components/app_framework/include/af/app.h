/**
 * @file app.h
 * @brief App: fluent assembler that owns start-up order, the frame loop and teardown.
 *
 * Replaces the ~400 lines every solution repeats (camera ctrl sequence,
 * RTSP/debug-stream/MQTT/ONVIF bring-up in the right order, signal handling,
 * retrieve/return loop, snapshot offer, result fan-out, cleanup):
 *
 * @code
 *   af::App app("region-motion");
 *   app.with_camera({640, 640, 10})
 *      .with_live_view()                           // 720p@30 H.264 over WS :8001, no RTSP
 *   //  or .with_stream_mode(af::StreamMode::Both, {1280, 720, 15})
 *      .with_mqtt(mqtt_opts)                       // ha_mqtt::ClientOptions
 *      .on_frame([&](const af::VideoFrame& f, af::Result& r) {
 *          af::RunReport rep = pipeline.run(f, state);
 *          r.attach(rep);
 *          r.set_boxes(state.boxes, state.n);
 *          r.json = state.payload;                 // MQTT contract, built by the app
 *      });
 *   return app.run();
 * @endcode
 *
 * Threads: run() executes on the caller's thread and calls every handler and
 * sink there. Triggers are polled on one internal event-loop thread and only
 * queued from it.
 *
 * Modes: with an on_frame() handler the loop processes every frame and hands
 * queued triggers to on_trigger() alongside. With on_trigger() only, the loop
 * sleeps until a trigger arrives, captures one fresh frame and returns to
 * standby (see CameraOptions::idle_fps). Sources that answer their caller
 * (HttpTrigger with sync_response) receive Result::json via complete().
 *
 * Exit codes from run(): 0 normal stop, 2 camera init failed, 3 video/RTSP
 * init failed, 4 no handler configured (same numbering the solutions used).
 */
#ifndef AF_APP_H_
#define AF_APP_H_

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

#include "af/frame.h"
#include "af/pipeline.h"
#include "af/sink.h"
#include "af/trigger.h"

namespace ha_mqtt {
struct ClientOptions;
class MqttPublisher;
}  // namespace ha_mqtt

namespace af {

/** Inference channel (VPSS CH0, CPU/TPU readable RGB888). */
struct CameraOptions {
    uint16_t width = 640;
    uint16_t height = 640;
    uint8_t fps = 10;
    /** true: frames carry only the physical address (vir == nullptr), no memcpy in the camera callback. */
    bool physical = false;
    /** Frames retrieved and dropped after on_started(), e.g. to let AE/AWB settle. */
    uint8_t warmup_frames = 0;
    /**
     * Trigger-only apps: channel rate while waiting for a trigger (1..fps; 0 =
     * stay at fps). The camera callback copies every accepted frame, so this is
     * what standby costs. Raised to fps for the triggered capture.
     */
    uint8_t idle_fps = 0;
};

/**
 * Optional raw channel (VPSS CH1, NV21, not encoded) next to the inference
 * channel, e.g. full-resolution stills while CH0 runs at the model size. It
 * shows the same sensor picture as CH0 (both fitted with ASPECT_RATIO_AUTO).
 * Costs one VB pool of `blocks` x width*height*3/2 bytes (1920x1080: 3.1 MB
 * per block) and VPSS bandwidth at `fps`.
 */
struct RawChannelOptions {
    uint16_t width = 1920;
    uint16_t height = 1080;
    uint8_t fps = 5;
    uint8_t blocks = 3;
};

/**
 * Raw channel frame: NV21, planes 0 (Y) and 1 (VU) with physical address and
 * stride only (vir == nullptr; map with CVI_SYS_Mmap*). Called on the
 * channel's VPSS thread for every frame; the buffer goes back to VPSS when the
 * handler returns, so keep it short.
 */
using RawFrameHandler = std::function<void(const VideoFrame&)>;

/**
 * H.264 stream channel (VENC CH2) geometry and encoder settings, shared by
 * every network endpoint (RTSP, WebSocket). Configuring it does not start any
 * endpoint; the channel is only set up when at least one endpoint that needs
 * video is enabled (see StreamMode).
 */
struct StreamOptions {
    uint16_t width = 1280;
    uint16_t height = 720;
    uint8_t fps = 30;
    uint32_t bitrate_kbps = 2500;   ///< VENC target bitrate; 0 keeps the ipcam parameter default
    uint32_t gop = 30;              ///< VENC GOP in frames; 0 keeps the ipcam parameter default
    uint32_t max_bitrate_kbps = 0;  ///< VENC max bitrate (VBR modes); 0 keeps the default
    /** Force a P-only GOP (no B-frame reordering) with CBR rate control. */
    bool low_latency = true;
};

/** Which network endpoints publish the stream channel. */
enum class StreamMode : uint8_t {
    None,      ///< no VENC channel, no WebSocket, no RTSP
    WsOnly,    ///< H.264 over WebSocket (+ /results, /snapshot.jpg); RTSP server never started
    RtspOnly,  ///< RTSP only
    Both,
};

/** "none" / "ws" / "rtsp" / "both". */
inline const char* to_string(StreamMode m) noexcept {
    switch (m) {
        case StreamMode::WsOnly:
            return "ws";
        case StreamMode::RtspOnly:
            return "rtsp";
        case StreamMode::Both:
            return "both";
        case StreamMode::None:
        default:
            return "none";
    }
}

/** Parse "none" / "ws" / "rtsp" / "both" (command-line --stream-mode). Returns false and leaves @p out unchanged otherwise. */
inline bool parse_stream_mode(const char* s, StreamMode& out) noexcept {
    if (s == nullptr) return false;
    const StreamMode all[] = {StreamMode::None, StreamMode::WsOnly, StreamMode::RtspOnly, StreamMode::Both};
    for (StreamMode m : all) {
        if (std::strcmp(s, to_string(m)) == 0) {
            out = m;
            return true;
        }
    }
    return false;
}

/** WebSocket endpoint: H.264 video on "/", result JSON on "/results", JPEG on "/snapshot.jpg". */
struct StreamWsOptions {
    int port = 8001;
    bool video = true;    ///< false: "/" rejects clients; no VENC consumer (and no VENC at all without RTSP)
    bool results = true;  ///< false: "/results" rejects clients, no result envelopes are built
    bool snapshot = true; ///< false: frames are never offered for /snapshot.jpg (GET answers 503)
    int max_clients = 2;  ///< per path (video and results each)
};

/**
 * RTSP protocol settings. Stream geometry/encoder settings come from
 * StreamOptions (with_stream / with_stream_mode).
 */
struct RtspOptions {
    int port = 8554;
    std::string session_prefix = "live";  ///< session i is "<prefix><i>"
    bool metadata = false;                ///< ONVIF analytics metadata track; feed it with rtsp_server_write_metadata()
    std::string username;  ///< empty: no authentication
    std::string password;

    // Legacy (pre-StreamOptions) channel settings, kept so existing callers
    // compile unchanged. width == 0 (the default) means "not set": the stream
    // channel keeps what with_stream() configured. When width/height are set,
    // with_rtsp()/with_stream(RtspOptions) copy all six fields into the stream
    // channel with the old semantics (fps 0 keeps the current value; gop and
    // bitrates 0 keep the ipcam defaults; low_latency off). New code: use StreamOptions.
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t fps = 0;
    uint32_t gop = 0;
    uint32_t bitrate_kbps = 0;
    uint32_t max_bitrate_kbps = 0;
};

/** ONVIF discovery + Device/Media2 SOAP. Requires RTSP (with_rtsp() or StreamMode::RtspOnly/Both). */
struct OnvifOptions {
    int port = 8000;
    std::string serial;       ///< device serial / uuid seed
    std::string device_name;  ///< empty: component default
    std::string username;
    std::string password;
    std::string location;
};

/** Exit codes returned by App::run(). */
enum ExitCode : int {
    kExitOk = 0,
    kExitCamera = 2,
    kExitVideo = 3,
    kExitNoHandler = 4,
};

class App {
public:
    /** Called for every retrieved frame. The frame is valid only during the call; Result is published after the frame is returned. */
    using FrameHandler = std::function<void(const VideoFrame&, Result&)>;
    /** Called with the first frame retrieved after a trigger was observed. */
    using TriggerHandler = std::function<void(const TriggerEvent&, const VideoFrame&, Result&)>;

    /** @param name application id used in logs (static string). */
    explicit App(const char* name);
    ~App();
    App(const App&) = delete;
    App& operator=(const App&) = delete;

    App& with_camera(const CameraOptions& opt = CameraOptions{});
    /** Open the raw channel CH1 (see RawChannelOptions); @p fn gets its frames. */
    App& with_raw_channel(const RawChannelOptions& opt, RawFrameHandler fn);

    /**
     * Live view for teleoperation: 720p@30 H.264 over WebSocket on @p port,
     * RTSP not started. Same as with_stream_mode(StreamMode::WsOnly, StreamOptions{}, port).
     */
    App& with_live_view(int port = 8001);
    /** Configure the stream channel (geometry, encoder). Starts no endpoint by itself. */
    App& with_stream(const StreamOptions& opt = StreamOptions{});
    /** Enable the WebSocket endpoint. Configures the stream channel only if opt.video (or RTSP) needs it. */
    App& with_stream_ws(const StreamWsOptions& opt = StreamWsOptions{});
    /**
     * Set the stream channel and exactly which endpoints run, replacing earlier
     * with_rtsp()/with_stream_ws() choices. Ports replace the configured ones;
     * other RTSP/WS options set earlier are kept.
     */
    App& with_stream_mode(StreamMode mode, const StreamOptions& stream_opt = StreamOptions{}, int ws_port = 8001, int rtsp_port = 8554);

    /** Enable RTSP. Legacy geometry fields in @p opt, when set, also configure the stream channel. */
    App& with_rtsp(const RtspOptions& opt = RtspOptions{});
    /** Legacy: configure the stream channel from RtspOptions' legacy fields (and store its RTSP settings) without enabling RTSP. */
    App& with_stream(const RtspOptions& opt);
    /** Legacy: with_stream_ws() with default options on @p port. */
    App& with_debug_ws(int port = 8001);

    /** Endpoints currently configured. */
    StreamMode stream_mode() const noexcept;
    /** MQTT results + HA discovery. A failed connect is logged and the app keeps running. */
    App& with_mqtt(const ha_mqtt::ClientOptions& opt);
    App& with_onvif(const OnvifOptions& opt);
    /** Register a trigger source (not owned; must outlive run()). Up to 7 sources. */
    App& with_trigger(TriggerSource& src);
    /** Register an extra result sink (not owned; must outlive run()). Up to 6 extra sinks. */
    App& with_sink(ResultSink& sink);

    /**
     * Called once on the run() thread after the camera is streaming and before
     * the first frame -- the place for ISP tweaks that need a running pipeline.
     */
    App& on_started(std::function<void()> fn);

    /**
     * Called on the run() thread right before each frame retrieval. Lets an
     * application snapshot state (e.g. an action generation counter) so that
     * frames already in flight when the state changes can be told apart.
     */
    App& on_before_grab(std::function<void()> fn);

    /**
     * Called on the run() thread with the held frame right before it is offered
     * for /snapshot.jpg (only when a snapshot client is waiting). May modify the
     * pixels in place -- the place for privacy masking, since hardware masks on
     * the encoder path do not reach the snapshot.
     */
    App& on_snapshot(std::function<void(const VideoFrame&)> fn);

    /** Called once while the camera is still streaming, before teardown (e.g. detach RGN overlays). */
    App& on_stopping(std::function<void()> fn);

    /** Called on the run() thread whenever no frame could be retrieved (timeouts, watchdogs). */
    App& on_idle(std::function<void()> fn);

    App& on_frame(FrameHandler fn);
    App& on_trigger(TriggerHandler fn);

    /** Bring everything up, loop until stop() / SIGINT / SIGTERM, tear down. */
    int run();

    /** Request the loop to end. Async-signal-safe. */
    void stop() noexcept;

    /**
     * Change the inference channel rate at runtime (any thread; applied by the
     * frame thread before its next retrieval). Lets an app drop to 1-5 fps while
     * idle: the camera callback copies every accepted frame on the single core,
     * so this is the main standby cost. 0 is ignored (the driver treats 0 as
     * "no rate limit"). In trigger-only mode it sets the standby rate.
     */
    void set_camera_fps(uint8_t fps) noexcept;

    /**
     * Queue a trigger from any thread (e.g. an MQTT command callback).
     * @return false when the pending queue is full (event dropped).
     */
    bool fire(const TriggerEvent& ev) noexcept;

    /** Whether any sink would deliver a result now; lets handlers skip building JSON. */
    bool any_sink_active() const noexcept;

    /** Connected WebSocket debug /results clients count. */
    int debug_clients_count() const noexcept;

    /** MQTT publisher for extra topics; nullptr before run() or without with_mqtt(). */
    ha_mqtt::MqttPublisher* mqtt() noexcept;

    /** Inference frames processed so far. */
    uint32_t frames() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace af

#endif  // AF_APP_H_
