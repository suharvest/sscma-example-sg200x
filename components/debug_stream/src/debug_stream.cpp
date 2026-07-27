#include "debug_stream.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ws_transport.h"

#include <video.h>            // requestVideoIDR, video_ch_index_t
#include "app_ipcam_venc.h"   // VENC_STREAM_S / VENC_PACK_S

#define DS_TAG "debug_stream"

// Bounded queue depths / limits
#define DS_VIDEO_QUEUE_MAX   4        // pending encoded frames (drop oldest)
#define DS_RESULT_QUEUE_MAX  16       // pending result JSON documents
#define DS_CLIENT_BACKLOG_MAX (1024 * 1024)  // per-connection send buffer cap

// Connection kind, kept in ws_conn tag slot 0. Must be non-zero: the transport
// treats tag slot 0 == 0 as "never completed an upgrade".
#define DS_CONN_VIDEO   'V'
#define DS_CONN_RESULT  'R'

// Per-connection resync flag kept in ws_conn tag slot 1 (video conns only).
// Set when we had to drop frames for a slow client; while set, that client is
// mid-GOP and must wait for the next keyframe (SPS/IDR) before it can decode
// again. Relationship to the global awaiting_idr:
//   - awaiting_idr (global, producer side): a *new* client joined and the
//     producer must not even queue P/B frames until one keyframe exists.
//   - needs_idr (per connection, broadcast side): *this already-running*
//     client fell behind and dropped frames, so it must skip forward to the
//     next keyframe. Independent because other clients keep streaming fine.
#define DS_NEEDS_IDR    'I'   // data[1] == DS_NEEDS_IDR -> awaiting keyframe
#define DS_SYNCED       '\0'  // data[1] == DS_SYNCED    -> normal delivery

namespace {

struct DebugStreamState {
    // Owns the listener and the event thread; see ws_transport.h for the
    // threading contract (ws_transport_wake is the only cross-thread call).
    struct ws_transport* transport = nullptr;
    std::atomic<bool> running{false};

    // Config (paths copied so caller-owned strings may go away)
    int port = 8001;
    std::string video_path = "/";
    std::string results_path = "/results";
    int video_ch = 2;  // VIDEO_CH2
    int max_video_clients = 2;
    int max_results_clients = 2;
    std::string snapshot_path = "/snapshot.jpg";

    // Client accounting. Written on the event thread, read by producers.
    std::atomic<int> video_clients{0};
    std::atomic<int> results_clients{0};

    // Keyframe gating: set when a video client joins, cleared once a frame
    // starting with SPS/PPS+IDR has been queued.
    std::atomic<bool> awaiting_idr{false};

    // Cached parameter sets (guarded by ps_mutex, touched on producer thread)
    std::mutex ps_mutex;
    std::vector<uint8_t> sps;  // includes Annex-B start code
    std::vector<uint8_t> pps;

    // Snapshot: latest encoded JPEG plus the arming state that keeps the
    // encode off the hot path when nobody is pulling snapshots.
    std::mutex snap_mutex;
    std::vector<uint8_t> snap_jpeg;      // guarded by snap_mutex
    std::atomic<uint64_t> snap_asked_ms{0};    // last GET
    std::atomic<uint64_t> snap_encoded_ms{0};  // last successful encode

    // Producer -> event thread queues (guarded by q_mutex)
    std::mutex q_mutex;
    std::deque<std::vector<uint8_t>> video_q;
    std::deque<std::string> results_q;
};

DebugStreamState g_ds;

// ---------------------------------------------------------------------------
// Annex-B helpers
// ---------------------------------------------------------------------------

// Return the H.264 NAL unit type of a VENC pack payload (packs are single
// NALs prefixed with a 3- or 4-byte Annex-B start code), or -1 if unknown.
static int nal_type_of(const uint8_t* p, size_t len) {
    size_t off = 0;
    if (len >= 4 && p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 1) {
        off = 4;
    } else if (len >= 3 && p[0] == 0 && p[1] == 0 && p[2] == 1) {
        off = 3;
    } else {
        return -1;
    }
    if (off >= len) return -1;
    return p[off] & 0x1F;
}

static uint64_t unix_ms_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

static void append_ts_le(std::vector<uint8_t>& buf, uint64_t ms) {
    for (int i = 0; i < 8; i++) {
        buf.push_back((uint8_t)((ms >> (8 * i)) & 0xFF));  // little-endian tail
    }
}

// ---------------------------------------------------------------------------
// Event-thread side. Everything below runs on the ws_transport event thread;
// producers only ever call ws_transport_wake(). See ws_transport.h.
// ---------------------------------------------------------------------------

// What one drain pass hands to each connection.
struct DrainBatch {
    const std::deque<std::vector<uint8_t>>* video;
    const std::deque<std::string>* results;
};

static void ds_feed_conn(struct ws_conn* c, void* ctx) {
    const DrainBatch* batch = static_cast<const DrainBatch*>(ctx);
    const uint8_t kind = ws_conn_tag(c, 0);

    if (kind == DS_CONN_VIDEO) {
        // Slow client: if its backlog is already large, skip frames instead of
        // buffering without bound. Mark the connection so it re-syncs on the
        // next keyframe rather than resuming mid-GOP (which decodes as garbage
        // until the next natural IDR), and nudge the encoder for a keyframe
        // (requestVideoIDR coalesces).
        if (ws_conn_backlog(c) > DS_CLIENT_BACKLOG_MAX) {
            if (ws_conn_tag(c, 1) != DS_NEEDS_IDR) {
                ws_conn_set_tag(c, 1, DS_NEEDS_IDR);
                requestVideoIDR((video_ch_index_t)g_ds.video_ch);
            }
            return;
        }
        for (const auto& frame : *batch->video) {
            // A connection recovering from a drop must wait for the next
            // keyframe (SPS/IDR) before it can decode; skip P/B frames until
            // then. Keyframe AUs start with SPS (7) or IDR (5).
            if (ws_conn_tag(c, 1) == DS_NEEDS_IDR) {
                int nt = nal_type_of(frame.data(), frame.size());
                if (nt != 7 && nt != 5) continue;
                ws_conn_set_tag(c, 1, DS_SYNCED);  // keyframe reached, resume
            }
            ws_conn_send(c, frame.data(), frame.size(), WS_OP_BINARY);
        }
    } else if (kind == DS_CONN_RESULT) {
        if (ws_conn_backlog(c) > DS_CLIENT_BACKLOG_MAX) return;
        for (const auto& doc : *batch->results) {
            ws_conn_send(c, doc.data(), doc.size(), WS_OP_TEXT);
        }
    }
}

// ws_transport_callbacks_t::on_drain
static void ds_on_drain(void* user) {
    (void)user;
    std::deque<std::vector<uint8_t>> vq;
    std::deque<std::string> rq;
    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        vq.swap(g_ds.video_q);
        rq.swap(g_ds.results_q);
    }
    if (vq.empty() && rq.empty()) return;  // coalesced/spurious wake

    DrainBatch batch{&vq, &rq};
    ws_transport_for_each(g_ds.transport, ds_feed_conn, &batch);
}

// ws_transport_callbacks_t::on_http — plain GET, tried before on_upgrade.
// Serves the JPEG snapshot; everything else falls through.
// How long a cold-start snapshot request waits for the first encoded frame,
// and how often it re-checks. The producer encodes on the very next frame after
// arming, so at 10fps the image is usually there by the second or third poll;
// the deadline is generous enough to cover a slow first frame and short enough
// that a client with no picture is told so rather than left hanging.
#define DS_SNAPSHOT_POLL_MS 80
#define DS_SNAPSHOT_MAX_ATTEMPTS 25  /* ~2s */

static ws_http_result_t ds_on_http(void* user, const char* path, int attempt,
                                   int* status, const char** content_type,
                                   const void** body, size_t* len,
                                   int* retry_ms) {
    (void)user;
    if (g_ds.snapshot_path != path) return WS_HTTP_PASS;

    // Arm first: even when this request cannot be served yet, it is what causes
    // the producer to start encoding.
    g_ds.snap_asked_ms.store(unix_ms_now(), std::memory_order_release);

    // Copy under the lock into a buffer only this thread touches, and hand out
    // *that*. The transport copies the body after we return, by which point
    // snap_mutex is released and the producer is free to reallocate snap_jpeg
    // underneath us -- returning snap_jpeg.data() directly would be a race.
    // ds_on_http only ever runs on the event thread, so snap_serve needs no
    // lock of its own.
    static std::vector<uint8_t> snap_serve;
    {
        std::lock_guard<std::mutex> lock(g_ds.snap_mutex);
        snap_serve = g_ds.snap_jpeg;
    }

    if (snap_serve.empty()) {
        // Cold start: this request armed the encoder and the first JPEG is
        // milliseconds away, so ask the transport to come back rather than
        // answering with an error. The 503 used to be returned immediately,
        // which made the snapshot URL that ONVIF advertises fail for any client
        // that fetches once and gives up -- adding a camera to a VMS is exactly
        // that shape of request.
        if (attempt < DS_SNAPSHOT_MAX_ATTEMPTS) {
            *retry_ms = DS_SNAPSHOT_POLL_MS;
            return WS_HTTP_RETRY;
        }
        // Waited and still nothing: the video pipeline is not producing. That
        // is a real failure and 503 is the honest answer.
        static const char kWarming[] = "snapshot unavailable\n";
        *status = 503;
        *content_type = "text/plain";
        *body = kWarming;
        *len = sizeof(kWarming) - 1;
        return WS_HTTP_DONE;
    }
    *status = 200;
    *content_type = "image/jpeg";
    *body = snap_serve.data();
    *len = snap_serve.size();
    return WS_HTTP_DONE;
}

// ws_transport_callbacks_t::on_upgrade
static bool ds_on_upgrade(void* user, const char* path, uint8_t* tag0,
                          int* status, const char** body) {
    (void)user;
    if (g_ds.video_path == path) {
        if (g_ds.video_clients.load(std::memory_order_relaxed) >= g_ds.max_video_clients) {
            *status = 503;
            *body = "debug video client limit reached\n";
            return false;
        }
        *tag0 = DS_CONN_VIDEO;
        return true;
    }
    if (g_ds.results_path == path) {
        if (g_ds.results_clients.load(std::memory_order_relaxed) >= g_ds.max_results_clients) {
            *status = 503;
            *body = "debug results client limit reached\n";
            return false;
        }
        *tag0 = DS_CONN_RESULT;
        return true;
    }
    return false;  // transport replies with the 404 default
}

// ws_transport_callbacks_t::on_open
static void ds_on_open(void* user, struct ws_conn* c, uint8_t tag0) {
    (void)user;
    if (tag0 == DS_CONN_VIDEO) {
        ws_conn_set_tag(c, 1, DS_SYNCED);  // per-conn resync flag starts clear
        g_ds.video_clients.fetch_add(1, std::memory_order_release);
        // New decoder needs SPS/PPS + IDR before any P/B frame.
        g_ds.awaiting_idr.store(true, std::memory_order_release);
        requestVideoIDR((video_ch_index_t)g_ds.video_ch);
    } else if (tag0 == DS_CONN_RESULT) {
        g_ds.results_clients.fetch_add(1, std::memory_order_release);
    }
}

// ws_transport_callbacks_t::on_close
static void ds_on_close(void* user, struct ws_conn* c, uint8_t tag0) {
    (void)user;
    (void)c;
    if (tag0 == DS_CONN_VIDEO) {
        g_ds.video_clients.fetch_sub(1, std::memory_order_release);
    } else if (tag0 == DS_CONN_RESULT) {
        g_ds.results_clients.fetch_sub(1, std::memory_order_release);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

void debug_stream_config_init(debug_stream_config_t* cfg) {
    if (cfg == NULL) return;
    cfg->port = 8001;
    cfg->video_path = "/";
    cfg->results_path = "/results";
    cfg->snapshot_path = "/snapshot.jpg";
    cfg->video_ch = 2;  // VIDEO_CH2
    cfg->max_video_clients = 2;
    cfg->max_results_clients = 2;
}

int debug_stream_create(const debug_stream_config_t* cfg) {
    if (g_ds.running.load()) {
        return -1;  // already running
    }

    debug_stream_config_t def;
    debug_stream_config_init(&def);
    if (cfg == NULL) cfg = &def;

    g_ds.port = cfg->port > 0 ? cfg->port : def.port;
    g_ds.video_path = (cfg->video_path && cfg->video_path[0]) ? cfg->video_path : def.video_path;
    g_ds.results_path = (cfg->results_path && cfg->results_path[0]) ? cfg->results_path : def.results_path;
    g_ds.snapshot_path = (cfg->snapshot_path && cfg->snapshot_path[0]) ? cfg->snapshot_path : def.snapshot_path;
    g_ds.video_ch = cfg->video_ch >= 0 ? cfg->video_ch : def.video_ch;
    g_ds.max_video_clients = cfg->max_video_clients > 0 ? cfg->max_video_clients : def.max_video_clients;
    g_ds.max_results_clients = cfg->max_results_clients > 0 ? cfg->max_results_clients : def.max_results_clients;

    g_ds.video_clients.store(0);
    g_ds.results_clients.store(0);
    g_ds.awaiting_idr.store(false);
    {
        std::lock_guard<std::mutex> lock(g_ds.ps_mutex);
        g_ds.sps.clear();
        g_ds.pps.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        g_ds.video_q.clear();
        g_ds.results_q.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ds.snap_mutex);
        g_ds.snap_jpeg.clear();
    }
    g_ds.snap_asked_ms.store(0);
    g_ds.snap_encoded_ms.store(0);

    ws_transport_config_t tcfg;
    tcfg.port = g_ds.port;

    ws_transport_callbacks_t tcb;
    tcb.user = nullptr;
    tcb.on_http = ds_on_http;
    tcb.on_upgrade = ds_on_upgrade;
    tcb.on_open = ds_on_open;
    tcb.on_close = ds_on_close;
    tcb.on_drain = ds_on_drain;

    // running must be visible before the event thread can call back into us.
    g_ds.running.store(true, std::memory_order_release);
    g_ds.transport = ws_transport_create(&tcfg, &tcb);
    if (g_ds.transport == nullptr) {
        g_ds.running.store(false, std::memory_order_release);
        fprintf(stderr, "[%s] failed to start transport on port %d\n", DS_TAG, g_ds.port);
        return -1;
    }

    fprintf(stderr, "[%s] listening on http://0.0.0.0:%d (video: %s, results: %s)\n",
            DS_TAG, g_ds.port, g_ds.video_path.c_str(), g_ds.results_path.c_str());
    return 0;
}

void debug_stream_destroy(void) {
    if (!g_ds.running.load()) {
        return;
    }
    // Clear running first: producers stop enqueueing/waking, then tearing the
    // transport down joins the event thread and closes every connection.
    g_ds.running.store(false, std::memory_order_release);
    ws_transport_destroy(g_ds.transport);
    g_ds.transport = nullptr;

    g_ds.video_clients.store(0);
    g_ds.results_clients.store(0);
    g_ds.awaiting_idr.store(false);
    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        g_ds.video_q.clear();
        g_ds.results_q.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ds.ps_mutex);
        g_ds.sps.clear();
        g_ds.pps.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ds.snap_mutex);
        g_ds.snap_jpeg.clear();
    }
    g_ds.snap_asked_ms.store(0);
    g_ds.snap_encoded_ms.store(0);
}

int debug_stream_video_handler(void* pData, void* pCtx, void* pUserData) {
    (void)pCtx;
    (void)pUserData;

    if (!g_ds.running.load(std::memory_order_acquire)) return 0;
    // Lazy path: no video clients -> single atomic load, no locks, no copies.
    if (g_ds.video_clients.load(std::memory_order_relaxed) == 0) return 0;

    VENC_STREAM_S* stream = (VENC_STREAM_S*)pData;
    if (stream == NULL || stream->u32PackCount == 0 || stream->pstPack == NULL) return 0;

    // First pass: classify NALs, cache SPS/PPS, measure total payload size.
    bool has_idr = false, has_sps = false, has_pps = false;
    size_t total = 0;
    for (CVI_U32 i = 0; i < stream->u32PackCount; i++) {
        VENC_PACK_S* pk = &stream->pstPack[i];
        if (pk->pu8Addr == NULL || pk->u32Len <= pk->u32Offset) continue;
        const uint8_t* d = pk->pu8Addr + pk->u32Offset;
        size_t l = pk->u32Len - pk->u32Offset;
        switch (nal_type_of(d, l)) {
            case 7: {  // SPS: cache latest (with start code)
                std::lock_guard<std::mutex> lock(g_ds.ps_mutex);
                g_ds.sps.assign(d, d + l);
                has_sps = true;
                break;
            }
            case 8: {  // PPS: cache latest (with start code)
                std::lock_guard<std::mutex> lock(g_ds.ps_mutex);
                g_ds.pps.assign(d, d + l);
                has_pps = true;
                break;
            }
            case 5:  // IDR slice
                has_idr = true;
                break;
            default:
                break;
        }
        total += l;
    }

    const bool awaiting = g_ds.awaiting_idr.load(std::memory_order_acquire);
    if (awaiting && !has_idr) {
        // A fresh client cannot decode P/B frames: drop until the keyframe.
        return 0;
    }

    // Assemble one WS message: [SPS+PPS (only if the IDR AU lacks them)]
    // [all packs][8-byte LE unix-ms timestamp].
    std::vector<uint8_t> buf;
    if (has_idr && (!has_sps || !has_pps)) {
        std::lock_guard<std::mutex> lock(g_ds.ps_mutex);
        buf.reserve(total + g_ds.sps.size() + g_ds.pps.size() + 8);
        if (!has_sps) buf.insert(buf.end(), g_ds.sps.begin(), g_ds.sps.end());
        if (!has_pps) buf.insert(buf.end(), g_ds.pps.begin(), g_ds.pps.end());
    } else {
        buf.reserve(total + 8);
    }
    for (CVI_U32 i = 0; i < stream->u32PackCount; i++) {
        VENC_PACK_S* pk = &stream->pstPack[i];
        if (pk->pu8Addr == NULL || pk->u32Len <= pk->u32Offset) continue;
        buf.insert(buf.end(), pk->pu8Addr + pk->u32Offset, pk->pu8Addr + pk->u32Len);
    }
    if (buf.empty()) return 0;
    append_ts_le(buf, unix_ms_now());

    if (awaiting && has_idr) {
        g_ds.awaiting_idr.store(false, std::memory_order_release);
    }

    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        if (g_ds.video_q.size() >= DS_VIDEO_QUEUE_MAX) {
            g_ds.video_q.pop_front();  // drop oldest frame under backpressure
        }
        g_ds.video_q.push_back(std::move(buf));
    }
    ws_transport_wake(g_ds.transport);
    return 0;
}

int debug_stream_publish_result(const char* json, size_t len) {
    if (!g_ds.running.load(std::memory_order_acquire)) return -1;
    if (json == NULL || len == 0) return -1;
    // Lazy path: nobody listening -> skip the copy entirely.
    if (g_ds.results_clients.load(std::memory_order_relaxed) == 0) return 0;

    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        if (g_ds.results_q.size() >= DS_RESULT_QUEUE_MAX) {
            g_ds.results_q.pop_front();
        }
        g_ds.results_q.emplace_back(json, len);
    }
    ws_transport_wake(g_ds.transport);
    return 0;
}

bool debug_stream_snapshot_armed(void) {
    const uint64_t asked = g_ds.snap_asked_ms.load(std::memory_order_acquire);
    if (asked == 0) return false;
    const uint64_t now = unix_ms_now();
    // now < asked means the wall clock jumped backwards (this device boots at
    // 1970 and gets stepped by NTP). Treat the request as still recent rather
    // than letting an unsigned wrap disarm us until the next GET.
    return now < asked || (now - asked) <= DEBUG_STREAM_SNAPSHOT_ARM_MS;
}

int debug_stream_offer_snapshot(const void* rgb888, int width, int height) {
    if (!g_ds.running.load(std::memory_order_acquire)) return -1;
    if (rgb888 == nullptr || width <= 0 || height <= 0) return -1;

    // Lazy path: nobody has asked recently -> one atomic load and out.
    if (!debug_stream_snapshot_armed()) return 0;

    const uint64_t now = unix_ms_now();
    const uint64_t last = g_ds.snap_encoded_ms.load(std::memory_order_acquire);
    if (last != 0 && now >= last && (now - last) < DEBUG_STREAM_SNAPSHOT_MIN_INTERVAL_MS) {
        return 0;  // throttled
    }

    // Encoding runs here, on the caller's thread, by design: the transport's
    // event thread must never do cv::imencode (it would stall the video
    // WebSocket), and the VENC callback must never do it either (real-time
    // priority; starving it is what broke RTSP once already).
    ::cv::Mat rgb(height, width, CV_8UC3, const_cast<void*>(rgb888));
    ::cv::Mat bgr;
    ::cv::cvtColor(rgb, bgr, ::cv::COLOR_RGB2BGR);

    std::vector<uint8_t> jpeg;
    if (!::cv::imencode(".jpg", bgr, jpeg, {::cv::IMWRITE_JPEG_QUALITY, 80})) {
        fprintf(stderr, "[%s] snapshot JPEG encode failed\n", DS_TAG);
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_ds.snap_mutex);
        g_ds.snap_jpeg.swap(jpeg);
    }
    g_ds.snap_encoded_ms.store(now, std::memory_order_release);
    return 0;
}

int debug_stream_video_client_count(void) {
    return g_ds.video_clients.load(std::memory_order_relaxed);
}

int debug_stream_results_client_count(void) {
    return g_ds.results_clients.load(std::memory_order_relaxed);
}

int debug_stream_start_or_disable(int port, int video_ch) {
    debug_stream_config_t cfg;
    debug_stream_config_init(&cfg);
    cfg.port = port;
    cfg.video_ch = video_ch;

    if (debug_stream_create(&cfg) != 0) {
        fprintf(stderr, "[%s] failed to start debug stream on port %d, continuing without it\n",
                DS_TAG, port);
        return -1;
    }

    fprintf(stderr, "[%s] debug stream: ws://<device_ip>:%d/ (video), ws://<device_ip>:%d/results\n",
            DS_TAG, port, port);
    // Debug stream shares the VENC output (consumer index 1, RTSP owns 0).
    registerVideoFrameHandler((video_ch_index_t)video_ch, 1, debug_stream_video_handler, NULL);
    return 0;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// C++-only: /results envelope builder shared by the gallery applications
// ---------------------------------------------------------------------------

std::string debug_stream_build_results(uint64_t timestamp_ms, uint32_t frame_id,
                                       float inference_time_ms, int res_w, int res_h,
                                       const std::vector<debug_stream_box_t>& boxes,
                                       const std::vector<std::string>* labels,
                                       const std::string& extra_json) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(1);
    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"resolution\":[" << res_w << "," << res_h << "],";
    json << "\"boxes\":[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& b = boxes[i];
        if (i > 0) json << ",";
        json << "[" << b.x << "," << b.y << "," << b.w << "," << b.h << ","
             << std::setprecision(3) << b.score << std::setprecision(1) << ","
             << "\"" << b.label << "\"]";
    }
    json << "],";
    json << "\"labels\":[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (i > 0) json << ",";
        json << "\"" << (labels ? (*labels)[i] : boxes[i].label) << "\"";
    }
    json << "]";
    if (!extra_json.empty()) {
        json << "," << extra_json;
    }
    json << "}";
    return json.str();
}

void debug_stream_letterbox_to_display(std::vector<debug_stream_box_t>& boxes,
                                       int inference_w, int inference_h,
                                       int display_w, int display_h) {
    if (inference_w <= 0 || inference_h <= 0 || display_w <= 0 || display_h <= 0) {
        return;
    }
    // The sensor content (display aspect) is fit into the inference frame
    // preserving aspect and centered; find that content sub-rect in inference
    // pixels. Compare aspects via cross-multiply to avoid float division.
    float content_w = static_cast<float>(inference_w);
    float content_h = static_cast<float>(inference_h);
    float x_off = 0.f, y_off = 0.f;
    const long long disp_vs_inf =
        static_cast<long long>(display_w) * inference_h - static_cast<long long>(display_h) * inference_w;
    if (disp_vs_inf > 0) {
        // Display is wider than the inference frame -> bars top/bottom.
        content_h = static_cast<float>(inference_w) * display_h / display_w;
        y_off     = (inference_h - content_h) * 0.5f;
    } else if (disp_vs_inf < 0) {
        // Display is taller -> bars left/right.
        content_w = static_cast<float>(inference_h) * display_w / display_h;
        x_off     = (inference_w - content_w) * 0.5f;
    }
    // else: aspects match -> content fills the frame (pure scale below).
    const float sx = display_w / content_w;
    const float sy = display_h / content_h;
    for (auto& b : boxes) {
        b.x = (b.x - x_off) * sx;
        b.y = (b.y - y_off) * sy;
        b.w = b.w * sx;
        b.h = b.h * sy;
    }
}
