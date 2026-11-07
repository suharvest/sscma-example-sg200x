/**
 * @file camera_hub.h
 * @brief Owns the SG200X camera and the streaming services that hang off the video pipeline.
 *
 * The bring-up order is not arbitrary; each step depends on the previous one:
 *
 *   1. CameraSG200X::init(0) + the four CH0 ctrls (kChannel, kFps, kWindow, kPhysical)
 *      -- only records the configuration, no hardware touched yet.
 *   2. initVideo()                   -- loads the ipcam parameter set.
 *   3. setupVideo(CH2, H.264)        -- stream channel, before any consumer is registered.
 *                                       Steps 2-3 only when an endpoint consumes video
 *                                       (StreamConfig::needs_venc()).
 *   4. RTSP: consumer index 0 on CH2 + rtsp_server_start(). Only when RTSP is
 *      enabled; otherwise the RTSP server is never initialised.
 *   5. debug_stream (WebSocket): server, plus consumer index 1 on CH2 when it
 *      carries video. The VENC callback hands H.264 straight to it.
 *   6. Camera::startStream()         -- sets up CH0, registers the camera callback, startVideo().
 *   7. startVideo()                  -- idempotent guard; kept for parity with the solutions.
 *
 * ONVIF must come after step 4 (GetProfiles answers from the running RTSP
 * server); the App handles that between open() and start().
 *
 * Teardown mirrors the shipped solutions: stopStream (which also deinits the
 * video pipeline), debug_stream, RTSP, deinitVideo.
 */
#ifndef AF_CAMERA_HUB_H_
#define AF_CAMERA_HUB_H_

#include <cstdint>

#include <sscma.h>

#include "af/app.h"
#include "af/frame.h"
#include "stream_config.h"

namespace af {

struct HubConfig {
    CameraOptions camera;
    StreamConfig streams;  ///< H.264 channel + which endpoints publish it
};

class CameraHub {
public:
    CameraHub() = default;
    CameraHub(const CameraHub&) = delete;
    CameraHub& operator=(const CameraHub&) = delete;
    ~CameraHub() { close(); }

    /** Steps 1-5. Returns kExitOk, kExitCamera or kExitVideo. The WebSocket endpoint degrades silently. */
    int open(const HubConfig& cfg);

    /** Steps 6-7. */
    bool start();

    /**
     * Retrieve the newest inference frame and describe it as a VideoFrame.
     * The underlying buffer stays valid until release(); grab() releases a
     * previously held frame first.
     */
    bool grab(VideoFrame& out);

    /**
     * Like grab(), but skip frames captured before @p mark (from mark()) and
     * keep retrieving until a newer one arrives or @p timeout_ms passes.
     */
    bool grab_after(uint32_t mark, VideoFrame& out, uint32_t timeout_ms);

    /** Capture-tick reference for grab_after(); same clock as the camera's frame timestamps. */
    uint32_t mark() const noexcept;

    /** Change the CH0 rate while streaming (1..255). The camera callback reads it per frame. */
    bool set_fps(uint8_t fps);

    bool streaming() const noexcept { return streaming_; }

    /** Return the held frame to the camera. No-op when nothing is held. */
    void release();

    /** Stop streaming and tear down every service opened by open(). Idempotent. */
    void close();

    bool rtsp_running() const noexcept { return rtsp_; }
    bool debug_running() const noexcept { return debug_; }
    int debug_port() const noexcept { return debug_ ? cfg_.streams.ws.port : 0; }
    /** /snapshot.jpg is served and fed: WS running with snapshot enabled. */
    bool snapshot_enabled() const noexcept { return debug_ && cfg_.streams.ws.snapshot; }
    /** /results is served: WS running with results enabled. */
    bool results_enabled() const noexcept { return debug_ && cfg_.streams.ws.results; }
    const HubConfig& config() const noexcept { return cfg_; }

    /** CPU pointer of the held RGB888 frame, for debug_stream_offer_snapshot(); nullptr if none. */
    const uint8_t* held_rgb() const noexcept;

private:
    bool open_camera();
    bool open_venc();
    bool open_rtsp();
    bool open_ws();

    HubConfig cfg_;
    ma::Camera* camera_ = nullptr;
    ma_img_t held_{};
    bool holding_ = false;
    bool video_init_ = false;
    bool rtsp_ = false;
    bool debug_ = false;
    bool streaming_ = false;
    uint32_t seq_ = 0;
};

}  // namespace af

#endif  // AF_CAMERA_HUB_H_
