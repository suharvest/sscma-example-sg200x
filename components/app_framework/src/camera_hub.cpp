/**
 * @file camera_hub.cpp
 * @brief Camera + VENC + RTSP + WebSocket bring-up in the order documented in camera_hub.h.
 */
#include "camera_hub.h"

#include <debug_stream.h>
#include <rtsp_server.h>
#include <video.h>

namespace af {

namespace {

constexpr const char* TAG = "af.hub";

bool ctrl(ma::Camera* cam, ma::Camera::CtrlType type, ma::Camera::CtrlValue v) {
    return cam->commandCtrl(type, ma::Camera::CtrlMode::kWrite, v) == MA_OK;
}

}  // namespace

int CameraHub::open(const HubConfig& cfg) {
    cfg_ = cfg;
    if (!open_camera()) {
        return kExitCamera;
    }
    const StreamConfig& sc = cfg_.streams;
    MA_LOGI(TAG, "Stream mode: %s (VENC CH2 %s)", to_string(sc.mode()), sc.needs_venc() ? "on" : "off");
    if (sc.needs_venc() && !open_venc()) {
        return kExitVideo;
    }
    if (sc.rtsp_enabled && !open_rtsp()) {
        return kExitVideo;
    }
    if (sc.ws_enabled) {
        debug_ = open_ws();
        if (!debug_) {
            MA_LOGW(TAG, "WebSocket stream unavailable, continuing without it");
        }
    }
    if (cfg_.raw_fn) {
        open_raw();
    }
    return kExitOk;
}

void CameraHub::open_raw() {
    const RawChannelOptions& ro = cfg_.raw;
    video_ch_param_t p{};
    p.format = VIDEO_FORMAT_NV21;
    p.width = ro.width;
    p.height = ro.height;
    p.fps = ro.fps;
    p.blkcnt = ro.blocks;
    setupVideo(VIDEO_CH1, &p);
    registerVideoFrameHandler(VIDEO_CH1, 0, raw_stub, this);
    MA_LOGI(TAG, "Raw CH1 NV21 %ux%u @ %ufps, %u VB blocks", ro.width, ro.height, ro.fps, ro.blocks);
}

int CameraHub::raw_stub(void* pData, void* /*pArgs*/, void* pUserData) {
    auto* self = static_cast<CameraHub*>(pUserData);
    const auto* info = static_cast<const VIDEO_FRAME_INFO_S*>(pData);
    if (self == nullptr || info == nullptr || !self->cfg_.raw_fn) {
        return 0;
    }
    const VIDEO_FRAME_S& f = info->stVFrame;
    VideoFrame v;
    v.width = static_cast<uint16_t>(f.u32Width);
    v.height = static_cast<uint16_t>(f.u32Height);
    v.format = PixelFormat::NV21;
    v.planes[0] = Plane(nullptr, f.u64PhyAddr[0], f.u32Stride[0]);
    v.planes[1] = Plane(nullptr, f.u64PhyAddr[1], f.u32Stride[1]);
    v.channel = 1;
    v.seq = ++self->raw_seq_;
    v.ts_ms = static_cast<uint32_t>(monotonic_us() / 1000u);
    self->cfg_.raw_fn(v);
    return 0;
}

bool CameraHub::open_camera() {
    ma::Device* device = ma::Device::getInstance();
    if (device == nullptr) {
        MA_LOGE(TAG, "Device instance is null");
        return false;
    }
    for (auto* sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera_ = static_cast<ma::Camera*>(sensor);
            break;
        }
    }
    if (camera_ == nullptr) {
        MA_LOGE(TAG, "No camera sensor found");
        return false;
    }
    if (camera_->init(0) != MA_OK) {
        MA_LOGE(TAG, "Camera init failed");
        camera_ = nullptr;
        return false;
    }

    // kChannel must come first: the other ctrls apply to the selected channel.
    ma::Camera::CtrlValue v{};
    v.i32 = 0;
    bool ok = ctrl(camera_, ma::Camera::CtrlType::kChannel, v);
    v.i32 = cfg_.camera.fps;
    ok = ok && ctrl(camera_, ma::Camera::CtrlType::kFps, v);
    v.u16s[0] = cfg_.camera.width;
    v.u16s[1] = cfg_.camera.height;
    ok = ok && ctrl(camera_, ma::Camera::CtrlType::kWindow, v);
    v.i32 = cfg_.camera.physical ? 1 : 0;
    ok = ok && ctrl(camera_, ma::Camera::CtrlType::kPhysical, v);
    if (!ok) {
        MA_LOGE(TAG, "Camera ctrl sequence failed");
        return false;
    }
    MA_LOGI(TAG, "Camera CH0 %ux%u @ %ufps (%s)", cfg_.camera.width, cfg_.camera.height, cfg_.camera.fps, cfg_.camera.physical ? "physical" : "virtual");
    return true;
}

bool CameraHub::open_venc() {
    const StreamOptions& so = cfg_.streams.stream;
    if (initVideo() != 0) {
        MA_LOGE(TAG, "initVideo failed");
        return false;
    }
    video_init_ = true;

    video_ch_param_t p{};
    p.format = VIDEO_FORMAT_H264;
    p.width = so.width;
    p.height = so.height;
    p.fps = so.fps;
    if (setupVideo(VIDEO_CH2, &p) != 0) {
        MA_LOGE(TAG, "setupVideo(CH2) failed");
        return false;
    }

    // Encoder overrides go into the ipcam parameter set, which VENC reads when
    // startVideo() brings the channel up -- so this must happen before start().
    APP_PARAM_VENC_CTX_S* venc = app_ipcam_Venc_Param_Get();
    if (venc != nullptr && VIDEO_CH2 < venc->s32VencChnCnt) {
        APP_VENC_CHN_CFG_S& ch = venc->astVencChnCfg[VIDEO_CH2];
        if (so.gop != 0) ch.u32Gop = so.gop;
        if (so.bitrate_kbps != 0) {
            ch.u32BitRate = so.bitrate_kbps;
            // The ipcam defaults cap QP at 35 (video_paramparse.c h264/h265_stRcParam).
            // The driver RC clips every frame's QP to that ceiling (cvi_enc_rc.c
            // CLIP3(minQp, maxQp, qp)), so once QP 35 already costs more than the
            // target the stream overshoots it (measured 9 Mbps for a 1.2 Mbps CBR
            // target at 720p30). Lift the ceiling to the SDK default (DEF_264_MAXQP
            // = 51) so the requested bitrate is what the encoder converges to.
            ch.stRcParam.u32MaxQp = 51;
            ch.stRcParam.u32MaxIQp = 51;
        }
        if (so.max_bitrate_kbps != 0) ch.u32MaxBitRate = so.max_bitrate_kbps;
        if (so.low_latency) {
            // No B-frames: every encoded frame can be sent as soon as it is produced.
            ch.enGopMode = VENC_GOPMODE_NORMALP;
            ch.enRcMode = VENC_RC_MODE_H264CBR;
        }
        MA_LOGI(TAG, "Stream CH2 %ux%u @ %ufps, GOP=%u, bitrate=%u/%u kbps, QP %u-%u (I %u-%u)%s", so.width, so.height, so.fps, ch.u32Gop, ch.u32BitRate,
                ch.u32MaxBitRate, ch.stRcParam.u32MinQp, ch.stRcParam.u32MaxQp, ch.stRcParam.u32MinIQp, ch.stRcParam.u32MaxIQp,
                so.low_latency ? ", low-latency (P-only, CBR)" : "");
    }
    return true;
}

bool CameraHub::open_rtsp() {
    const RtspOptions& ro = cfg_.streams.rtsp;
    // Consumer index 0 is RTSP's by convention; the WebSocket takes index 1.
    registerVideoFrameHandler(VIDEO_CH2, 0, rtsp_server_video_handler, nullptr);
    rtsp_server_config_t rc;
    rtsp_server_config_init(&rc);
    rc.port = ro.port;
    rc.session_prefix = ro.session_prefix.c_str();
    rc.metadata_enabled = ro.metadata;
    rc.ch_mask = static_cast<uint8_t>(1u << VIDEO_CH2);
    rc.username = ro.username.empty() ? nullptr : ro.username.c_str();
    rc.password = ro.password.empty() ? nullptr : ro.password.c_str();
    if (rtsp_server_start(&rc) != 0) {
        MA_LOGE(TAG, "rtsp_server_start failed");
        return false;
    }
    rtsp_ = true;
    MA_LOGI(TAG, "RTSP rtsp://<ip>:%d/%s", rtsp_server_port(), rtsp_server_session_name(0));
    return true;
}

bool CameraHub::open_ws() {
    const StreamWsOptions& wo = cfg_.streams.ws;
    debug_stream_config_t dc;
    debug_stream_config_init(&dc);
    dc.port = wo.port;
    dc.video_ch = VIDEO_CH2;
    // 0 makes debug_stream reject the path.
    dc.max_video_clients = wo.video ? wo.max_clients : 0;
    dc.max_results_clients = wo.results ? wo.max_clients : 0;
    if (debug_stream_create(&dc) != 0) {
        return false;
    }
    if (wo.video) {
        // VENC callback -> debug_stream queue -> WS event thread; no RTSP involved.
        registerVideoFrameHandler(VIDEO_CH2, 1, debug_stream_video_handler, nullptr);
    }
    MA_LOGI(TAG, "WebSocket :%d video=%s results=%s snapshot=%s (max %d clients/path)", wo.port, wo.video ? "on" : "off", wo.results ? "on" : "off",
            wo.snapshot ? "on" : "off", wo.max_clients);
    return true;
}

bool CameraHub::start() {
    if (camera_ == nullptr) {
        return false;
    }
    if (camera_->startStream(ma::Camera::StreamMode::kRefreshOnReturn) != MA_OK) {
        MA_LOGE(TAG, "startStream failed");
        return false;
    }
    streaming_ = true;
    if (video_init_) {
        startVideo();  // no-op when startStream already started it
    }
    return true;
}

bool CameraHub::grab(VideoFrame& out) {
    release();
    if (!streaming_ || camera_->retrieveFrame(held_, MA_PIXEL_FORMAT_RGB888) != MA_OK || held_.data == nullptr) {
        return false;
    }
    holding_ = true;

    // The VPSS copy may pad rows; derive the stride from the delivered size
    // when it divides evenly, otherwise assume tight packing.
    uint32_t stride = static_cast<uint32_t>(held_.width) * 3u;
    if (held_.height != 0 && held_.size % held_.height == 0 && held_.size / held_.height >= stride) {
        stride = held_.size / held_.height;
    }

    if (held_.physical) {
        out = VideoFrame::packed(nullptr, held_.width, held_.height, PixelFormat::RGB888, stride, reinterpret_cast<uintptr_t>(held_.data));
    } else {
        out = VideoFrame::packed(held_.data, held_.width, held_.height, PixelFormat::RGB888, stride);
    }
    out.channel = 0;
    out.seq = ++seq_;
    out.ts_ms = static_cast<uint32_t>(monotonic_us() / 1000u);
    return true;
}

uint32_t CameraHub::mark() const noexcept {
    // ma_img_t::timestamp is Tick::current() (ns) truncated to 32 bits, so it
    // wraps every ~4.3 s; grab_after() compares with a wrap-aware difference,
    // valid while frame and mark are within ~2.1 s of each other.
    return static_cast<uint32_t>(ma::Tick::current());
}

bool CameraHub::grab_after(uint32_t mark, VideoFrame& out, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_us() + static_cast<uint64_t>(timeout_ms) * 1000u;
    while (monotonic_us() < deadline) {
        if (!grab(out)) {
            // retrieveFrame normally waited a frame period already; this only
            // keeps a non-streaming camera (immediate error) from spinning.
            ma::Thread::sleep(ma::Tick::fromMilliseconds(5));
            continue;
        }
        if (static_cast<int32_t>(held_.timestamp - mark) >= 0) {
            return true;
        }
        release();  // captured before the trigger
    }
    return false;
}

bool CameraHub::set_fps(uint8_t fps) {
    if (camera_ == nullptr || fps == 0) {
        return false;
    }
    // CH0 stays the selected ctrl channel since open_camera().
    ma::Camera::CtrlValue v{};
    v.i32 = fps;
    const bool ok = ctrl(camera_, ma::Camera::CtrlType::kFps, v);
    MA_LOGI(TAG, "Camera CH0 fps -> %u%s", fps, ok ? "" : " (failed)");
    return ok;
}

void CameraHub::release() {
    if (holding_) {
        camera_->returnFrame(held_);
        held_ = ma_img_t{};
        holding_ = false;
    }
}

const uint8_t* CameraHub::held_rgb() const noexcept {
    return holding_ && !held_.physical ? held_.data : nullptr;
}

void CameraHub::close() {
    release();
    if (camera_ != nullptr && streaming_) {
        camera_->stopStream();
    }
    streaming_ = false;
    if (debug_) {
        debug_stream_destroy();
        debug_ = false;
    }
    if (rtsp_) {
        rtsp_server_stop();
        rtsp_ = false;
    }
    if (video_init_) {
        deinitVideo();
        video_init_ = false;
    }
    camera_ = nullptr;
}

}  // namespace af
