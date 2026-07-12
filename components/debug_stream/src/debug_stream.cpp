#include "debug_stream.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mongoose.h"

#include <video.h>            // requestVideoIDR, video_ch_index_t
#include "app_ipcam_venc.h"   // VENC_STREAM_S / VENC_PACK_S

#define DS_TAG "debug_stream"

// Bounded queue depths / limits
#define DS_VIDEO_QUEUE_MAX   4        // pending encoded frames (drop oldest)
#define DS_RESULT_QUEUE_MAX  16       // pending result JSON documents
#define DS_CLIENT_BACKLOG_MAX (1024 * 1024)  // per-connection send buffer cap

// Connection tags kept in mg_connection::data[0]
#define DS_CONN_VIDEO   'V'
#define DS_CONN_RESULT  'R'

// Per-connection resync flag kept in mg_connection::data[1] (video conns only).
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
    struct mg_mgr mgr;
    std::thread poll_thread;
    std::atomic<bool> running{false};

    // Config (paths copied so caller-owned strings may go away)
    int port = 8001;
    std::string video_path = "/";
    std::string results_path = "/results";
    int video_ch = 2;  // VIDEO_CH2
    int max_video_clients = 2;
    int max_results_clients = 2;

    unsigned long listener_id = 0;

    // Client accounting. Written on the poll thread, read by producers.
    std::atomic<int> video_clients{0};
    std::atomic<int> results_clients{0};

    // Keyframe gating: set when a video client joins, cleared once a frame
    // starting with SPS/PPS+IDR has been queued.
    std::atomic<bool> awaiting_idr{false};

    // Cached parameter sets (guarded by ps_mutex, touched on producer thread)
    std::mutex ps_mutex;
    std::vector<uint8_t> sps;  // includes Annex-B start code
    std::vector<uint8_t> pps;

    // Producer -> poll thread queues (guarded by q_mutex)
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
// Poll-thread side: mongoose event handling (poll thread owns mg_mgr)
// ---------------------------------------------------------------------------

static void ds_broadcast(struct mg_mgr* mgr) {
    std::deque<std::vector<uint8_t>> vq;
    std::deque<std::string> rq;
    {
        std::lock_guard<std::mutex> lock(g_ds.q_mutex);
        vq.swap(g_ds.video_q);
        rq.swap(g_ds.results_q);
    }
    if (vq.empty() && rq.empty()) return;

    for (struct mg_connection* t = mgr->conns; t != NULL; t = t->next) {
        if (!t->is_websocket) continue;
        if (t->data[0] == DS_CONN_VIDEO) {
            // Slow client: if its send buffer is already large, skip frames
            // instead of buffering without bound. Mark the connection so it
            // re-syncs on the next keyframe rather than resuming mid-GOP (which
            // decodes as garbage until the next natural IDR), and nudge the
            // encoder for a keyframe (requestVideoIDR coalesces).
            if (t->send.len > DS_CLIENT_BACKLOG_MAX) {
                if (t->data[1] != DS_NEEDS_IDR) {
                    t->data[1] = DS_NEEDS_IDR;
                    requestVideoIDR((video_ch_index_t)g_ds.video_ch);
                }
                continue;
            }
            for (const auto& frame : vq) {
                // A connection recovering from a drop must wait for the next
                // keyframe (SPS/IDR) before it can decode; skip P/B frames
                // until then. Keyframe AUs start with SPS (7) or IDR (5).
                if (t->data[1] == DS_NEEDS_IDR) {
                    int nt = nal_type_of(frame.data(), frame.size());
                    if (nt != 7 && nt != 5) continue;
                    t->data[1] = DS_SYNCED;  // keyframe reached, resume delivery
                }
                mg_ws_send(t, frame.data(), frame.size(), WEBSOCKET_OP_BINARY);
            }
        } else if (t->data[0] == DS_CONN_RESULT) {
            if (t->send.len > DS_CLIENT_BACKLOG_MAX) continue;
            for (const auto& doc : rq) {
                mg_ws_send(t, doc.data(), doc.size(), WEBSOCKET_OP_TEXT);
            }
        }
    }
}

static void ds_ev_handler(struct mg_connection* c, int ev, void* ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;
        if (mg_strcmp(hm->uri, mg_str(g_ds.video_path.c_str())) == 0) {
            if (g_ds.video_clients.load(std::memory_order_relaxed) >= g_ds.max_video_clients) {
                mg_http_reply(c, 503, "", "debug video client limit reached\n");
                return;
            }
            mg_ws_upgrade(c, hm, NULL);
            c->data[0] = DS_CONN_VIDEO;
            c->data[1] = DS_SYNCED;  // per-conn resync flag starts clear
            g_ds.video_clients.fetch_add(1, std::memory_order_release);
            // New decoder needs SPS/PPS + IDR before any P/B frame.
            g_ds.awaiting_idr.store(true, std::memory_order_release);
            requestVideoIDR((video_ch_index_t)g_ds.video_ch);
        } else if (mg_strcmp(hm->uri, mg_str(g_ds.results_path.c_str())) == 0) {
            if (g_ds.results_clients.load(std::memory_order_relaxed) >= g_ds.max_results_clients) {
                mg_http_reply(c, 503, "", "debug results client limit reached\n");
                return;
            }
            mg_ws_upgrade(c, hm, NULL);
            c->data[0] = DS_CONN_RESULT;
            g_ds.results_clients.fetch_add(1, std::memory_order_release);
        } else {
            mg_http_reply(c, 404, "", "not found\n");
        }
    } else if (ev == MG_EV_CLOSE) {
        if (c->data[0] == DS_CONN_VIDEO) {
            g_ds.video_clients.fetch_sub(1, std::memory_order_release);
        } else if (c->data[0] == DS_CONN_RESULT) {
            g_ds.results_clients.fetch_sub(1, std::memory_order_release);
        }
    } else if (ev == MG_EV_WAKEUP) {
        // Producers queued data and kicked us; fan it out.
        ds_broadcast(c->mgr);
    }
}

static void ds_poll_loop(void) {
    while (g_ds.running.load(std::memory_order_acquire)) {
        mg_mgr_poll(&g_ds.mgr, 100);
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

    mg_mgr_init(&g_ds.mgr);
    if (!mg_wakeup_init(&g_ds.mgr)) {
        fprintf(stderr, "[%s] mg_wakeup_init failed\n", DS_TAG);
        mg_mgr_free(&g_ds.mgr);
        return -1;
    }

    char url[64];
    snprintf(url, sizeof(url), "http://0.0.0.0:%d", g_ds.port);
    struct mg_connection* lc = mg_http_listen(&g_ds.mgr, url, ds_ev_handler, NULL);
    if (lc == NULL) {
        fprintf(stderr, "[%s] failed to listen on %s\n", DS_TAG, url);
        mg_mgr_free(&g_ds.mgr);
        return -1;
    }
    g_ds.listener_id = lc->id;

    g_ds.running.store(true, std::memory_order_release);
    g_ds.poll_thread = std::thread(ds_poll_loop);

    fprintf(stderr, "[%s] listening on %s (video: %s, results: %s)\n",
            DS_TAG, url, g_ds.video_path.c_str(), g_ds.results_path.c_str());
    return 0;
}

void debug_stream_destroy(void) {
    if (!g_ds.running.load()) {
        return;
    }
    g_ds.running.store(false, std::memory_order_release);
    if (g_ds.poll_thread.joinable()) {
        g_ds.poll_thread.join();
    }
    mg_mgr_free(&g_ds.mgr);

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
    mg_wakeup(&g_ds.mgr, g_ds.listener_id, "v", 1);
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
    mg_wakeup(&g_ds.mgr, g_ds.listener_id, "r", 1);
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
