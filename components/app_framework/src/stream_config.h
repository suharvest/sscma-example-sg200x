/**
 * @file stream_config.h
 * @brief Stream channel + endpoint selection, separate from the hardware bring-up.
 *
 * SDK-free (only af/app.h types) so the decision of what gets started is unit
 * tested on the host; CameraHub only executes it:
 *
 *   needs_venc()  -> initVideo + setupVideo(CH2): RTSP, or WS with video
 *   rtsp_enabled  -> RTSP consumer + rtsp_server_start (nothing RTSP otherwise)
 *   ws_enabled    -> debug_stream server; VENC consumer only when ws.video
 */
#ifndef AF_STREAM_CONFIG_H_
#define AF_STREAM_CONFIG_H_

#include "af/app.h"

namespace af {

struct StreamConfig {
    StreamOptions stream;
    RtspOptions rtsp;
    StreamWsOptions ws;
    bool rtsp_enabled = false;
    bool ws_enabled = false;

    void set_stream(const StreamOptions& opt) { stream = opt; }

    /** Copy the legacy channel fields of @p opt into the stream channel when they are set (width/height != 0). */
    void apply_legacy(const RtspOptions& opt) {
        if (opt.width == 0 || opt.height == 0) {
            return;
        }
        stream.width = opt.width;
        stream.height = opt.height;
        if (opt.fps != 0) {
            stream.fps = opt.fps;
        }
        // Old semantics: 0 keeps the ipcam default, encoder mode untouched.
        stream.gop = opt.gop;
        stream.bitrate_kbps = opt.bitrate_kbps;
        stream.max_bitrate_kbps = opt.max_bitrate_kbps;
        stream.low_latency = false;
    }

    /** Legacy with_stream(RtspOptions): channel from the legacy fields, RTSP settings stored, RTSP not enabled. */
    void set_stream_legacy(const RtspOptions& opt) {
        apply_legacy(opt);
        rtsp = opt;
    }

    void enable_rtsp(const RtspOptions& opt) {
        apply_legacy(opt);
        rtsp = opt;
        rtsp_enabled = true;
    }

    void enable_ws(const StreamWsOptions& opt) {
        ws = opt;
        ws_enabled = true;
    }

    void set_mode(StreamMode mode, const StreamOptions& opt, int ws_port, int rtsp_port) {
        stream = opt;
        ws.port = ws_port;
        rtsp.port = rtsp_port;
        ws_enabled = mode == StreamMode::WsOnly || mode == StreamMode::Both;
        rtsp_enabled = mode == StreamMode::RtspOnly || mode == StreamMode::Both;
    }

    StreamMode mode() const noexcept {
        if (rtsp_enabled && ws_enabled) return StreamMode::Both;
        if (rtsp_enabled) return StreamMode::RtspOnly;
        if (ws_enabled) return StreamMode::WsOnly;
        return StreamMode::None;
    }

    /** Whether the H.264 channel (VENC CH2) has a consumer and must be configured. */
    bool needs_venc() const noexcept { return rtsp_enabled || (ws_enabled && ws.video); }
};

}  // namespace af

#endif  // AF_STREAM_CONFIG_H_
