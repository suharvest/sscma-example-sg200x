/*
 * ws_transport backed by libwebsockets.
 *
 * Sibling of ws_transport_mongoose.cpp; exactly one of the two is compiled in
 * (see CMakeLists.txt). debug_stream.cpp does not change between them, which
 * is the point of the seam.
 *
 * The interface was shaped after lws rather than mongoose precisely so this
 * file could exist without renegotiating the contract. Where mongoose was
 * happy to be told "send this now", lws is not, and the differences are all in
 * one direction:
 *
 *   - lws does not buffer for the application. mg_ws_send() copied into the
 *     connection's send buffer synchronously; here a message is queued and the
 *     actual write happens later in LWS_CALLBACK_SERVER_WRITEABLE, after
 *     asking for the slot with lws_callback_on_writable().
 *   - There is no "peek at the socket send buffer". Backlog is the depth of
 *     the queue this file owns, which is what ws_conn_backlog() was defined as
 *     rather than as send.len.
 *   - lws has no public way to iterate every wsi, so the connection list is
 *     maintained here.
 *   - Every outgoing buffer needs LWS_PRE bytes of headroom for lws to write
 *     the frame header in place. Payloads are copied into a padded buffer on
 *     the way into the queue; the copy is unavoidable anyway since the caller
 *     only guarantees its data for the duration of the call.
 *
 * Threading: lws_cancel_service() is the only API safe from another thread,
 * which is exactly ws_transport_wake(). Everything else runs on the event
 * thread. Calling lws_callback_on_writable() from a producer thread is the
 * classic way to corrupt an lws context, and it fails intermittently rather
 * than immediately -- hence the contract in ws_transport.h.
 */

#include "ws_transport.h"

#include <libwebsockets.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define WST_TAG "ws_transport"

namespace {

struct OutMsg {
    std::vector<uint8_t> buf;  /* LWS_PRE bytes of headroom, then payload */
    size_t len = 0;            /* payload length, excluding the headroom */
    ws_op_t op = WS_OP_BINARY;
};

/* Per-connection state. Lives in lws's per_session_data, constructed on
 * ESTABLISHED and destroyed on CLOSED. */
struct LwsConn {
    struct lws* wsi = nullptr;
    uint8_t tags[WS_CONN_TAG_SLOTS] = { 0 };
    std::deque<OutMsg> outq;
    size_t outq_bytes = 0;
};

/* An HTTP reply staged by on_http, written from HTTP_WRITEABLE. lws cannot
 * write a body synchronously the way mongoose could. */
struct HttpReply {
    std::vector<uint8_t> body;
    size_t sent = 0;
    bool active = false;
};

struct PerSession {
    LwsConn* conn = nullptr;   /* websocket connections only */
    HttpReply* http = nullptr; /* plain HTTP requests only */
    /* WS_HTTP_RETRY answers given for the request in flight. Lives here rather
     * than in the application because the application sees one call at a time
     * and has nowhere to hang per-request state; the transport already has a
     * per-connection slot. */
    int http_attempt = 0;
};

} // namespace

struct ws_transport {
    struct lws_context* ctx = nullptr;
    std::thread event_thread;
    std::atomic<bool> running { false };
    ws_transport_callbacks_t cb {};
    std::vector<LwsConn*> conns;  /* event thread only */

    /*
     * Connection kind decided at upgrade time, consumed at ESTABLISHED.
     *
     * The obvious lws_set_opaque_user_data()/lws_get_opaque_user_data() pair
     * does not survive that transition reliably, and the failure is quiet: the
     * results connection came back tagged as a video connection, so it counted
     * against the video client limit and the second real video client was
     * refused with 503. An explicit map is boring and correct. Event thread
     * only, so no lock.
     */
    std::unordered_map<struct lws*, uint8_t> pending_tags;
};

/* Single instance: debug_stream creates one transport, and lws's protocol
 * table needs a stable pointer back to it from the callback. */
static struct ws_transport* g_t = nullptr;

// ---------------------------------------------------------------------------
// Per-connection accessors
// ---------------------------------------------------------------------------

static inline LwsConn* as_conn(struct ws_conn* c)
{
    return reinterpret_cast<LwsConn*>(c);
}
static inline const LwsConn* as_conn(const struct ws_conn* c)
{
    return reinterpret_cast<const LwsConn*>(c);
}
static inline struct ws_conn* as_ws(LwsConn* c)
{
    return reinterpret_cast<struct ws_conn*>(c);
}

uint8_t ws_conn_tag(const struct ws_conn* c, int slot)
{
    if (c == NULL || slot < 0 || slot >= WS_CONN_TAG_SLOTS) return 0;
    return as_conn(c)->tags[slot];
}

void ws_conn_set_tag(struct ws_conn* c, int slot, uint8_t value)
{
    if (c == NULL || slot < 0 || slot >= WS_CONN_TAG_SLOTS) return;
    as_conn(c)->tags[slot] = value;
}

size_t ws_conn_backlog(const struct ws_conn* c)
{
    if (c == NULL) return 0;
    /* Bytes accepted from the application but not yet written to the socket --
     * the same quantity mongoose exposed as send.len, owned here instead. */
    return as_conn(c)->outq_bytes;
}

void ws_conn_send(struct ws_conn* c, const void* data, size_t len, ws_op_t op)
{
    if (c == NULL || data == NULL || len == 0) return;
    LwsConn* lc = as_conn(c);
    if (lc->wsi == nullptr) return;

    OutMsg m;
    m.buf.resize(LWS_PRE + len);
    memcpy(m.buf.data() + LWS_PRE, data, len);
    m.len = len;
    m.op = op;

    lc->outq_bytes += len;
    lc->outq.push_back(std::move(m));

    /* Event thread only; see the threading note at the top. */
    lws_callback_on_writable(lc->wsi);
}

// ---------------------------------------------------------------------------
// Protocol callback
// ---------------------------------------------------------------------------

static std::string uri_of(struct lws* wsi)
{
    char buf[256];
    /* GET_URI is the path for both plain HTTP and the upgrade request. */
    int n = lws_hdr_copy(wsi, buf, sizeof(buf), WSI_TOKEN_GET_URI);
    if (n <= 0) return "";
    return std::string(buf, static_cast<size_t>(n));
}

static int wst_callback(struct lws* wsi, enum lws_callback_reasons reason,
    void* user, void* in, size_t len)
{
    struct ws_transport* t = g_t;
    PerSession* ps = static_cast<PerSession*>(user);
    if (t == nullptr) return 0;

#ifdef WST_LWS_TRACE
    switch (reason) {
    case LWS_CALLBACK_HTTP:
    case LWS_CALLBACK_HTTP_WRITEABLE:
    case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
    case LWS_CALLBACK_ESTABLISHED:
    case LWS_CALLBACK_CLOSED:
    case LWS_CALLBACK_CLOSED_HTTP:
    case LWS_CALLBACK_HTTP_DROP_PROTOCOL:
    case LWS_CALLBACK_EVENT_WAIT_CANCELLED:
    case LWS_CALLBACK_SERVER_WRITEABLE:
        fprintf(stderr, "[wst-trace] reason=%d wsi=%p uri=%s\n",
            (int)reason, (void*)wsi,
            (reason == LWS_CALLBACK_HTTP || reason == LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION)
                ? uri_of(wsi).c_str() : "-");
        break;
    default: break;
    }
#endif

    switch (reason) {

    // ---- plain HTTP -------------------------------------------------------
    case LWS_CALLBACK_TIMER:
    case LWS_CALLBACK_HTTP: {
        const std::string path = uri_of(wsi);

        int status = 404;
        const char* ctype = "text/plain";
        const void* body = nullptr;
        size_t blen = 0;
        int retry_ms = 0;
        ws_http_result_t r = WS_HTTP_PASS;

        if (t->cb.on_http != nullptr) {
            r = t->cb.on_http(t->cb.user, path.c_str(),
                ps != nullptr ? ps->http_attempt : 0,
                &status, &ctype, &body, &blen, &retry_ms);
        }

        /* Ask again later without answering now. lws's own timer is used
         * rather than a thread: this must not block the event loop, which is
         * also servicing every WebSocket video client. */
        if (r == WS_HTTP_RETRY) {
            if (ps != nullptr) ps->http_attempt++;
            if (retry_ms < 10) retry_ms = 10;
            lws_set_timer_usecs(wsi, static_cast<lws_usec_t>(retry_ms) * 1000);
            return 0;
        }
        if (ps != nullptr) ps->http_attempt = 0;

        if (r == WS_HTTP_PASS) {
            static const char kNotFound[] = "not found\n";
            status = 404;
            ctype = "text/plain";
            body = kNotFound;
            blen = sizeof(kNotFound) - 1;
        }

        /* Copy now: the contract says the body is only valid during the
         * callback, and lws will write it after we return. */
        if (ps != nullptr) {
            ps->http = new (std::nothrow) HttpReply();
            if (ps->http == nullptr) return -1;
            ps->http->body.assign(static_cast<const uint8_t*>(body),
                static_cast<const uint8_t*>(body) + blen);
            ps->http->active = true;
        }

        uint8_t hdr[LWS_PRE + 512];
        uint8_t* p = hdr + LWS_PRE;
        uint8_t* end = hdr + sizeof(hdr) - 1;
        if (lws_add_http_common_headers(wsi, static_cast<unsigned int>(status),
                ctype, static_cast<lws_filepos_t>(blen), &p, end))
            return 1;
        /* Matches the mongoose backend's reply headers. */
        if (lws_add_http_header_by_name(wsi,
                reinterpret_cast<const unsigned char*>("access-control-allow-origin:"),
                reinterpret_cast<const unsigned char*>("*"), 1, &p, end))
            return 1;
        if (lws_add_http_header_by_name(wsi,
                reinterpret_cast<const unsigned char*>("cache-control:"),
                reinterpret_cast<const unsigned char*>("no-store"), 8, &p, end))
            return 1;
        if (lws_finalize_write_http_header(wsi, hdr + LWS_PRE, &p, end))
            return 1;

        lws_callback_on_writable(wsi);
        return 0;
    }

    case LWS_CALLBACK_HTTP_WRITEABLE: {
        if (ps == nullptr || ps->http == nullptr || !ps->http->active) return 0;
        HttpReply* r = ps->http;
        if (r->sent >= r->body.size()) {
            return lws_http_transaction_completed(wsi) ? -1 : 0;
        }
        /* Chunk it: a snapshot JPEG is tens of kilobytes and lws_write wants a
         * buffer with LWS_PRE headroom. */
        const size_t chunk = 4096;
        const size_t n = std::min(chunk, r->body.size() - r->sent);
        std::vector<uint8_t> out(LWS_PRE + n);
        memcpy(out.data() + LWS_PRE, r->body.data() + r->sent, n);
        const int w = lws_write(wsi, out.data() + LWS_PRE, n,
            (r->sent + n >= r->body.size()) ? LWS_WRITE_HTTP_FINAL : LWS_WRITE_HTTP);
        if (w < static_cast<int>(n)) return -1;
        r->sent += n;
        if (r->sent >= r->body.size()) {
            return lws_http_transaction_completed(wsi) ? -1 : 0;
        }
        lws_callback_on_writable(wsi);
        return 0;
    }

    // ---- websocket upgrade ------------------------------------------------
    case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION: {
        const std::string path = uri_of(wsi);
        uint8_t tag0 = 0;
        int status = 404;
        const char* body = "not found\n";
        bool accept = false;
        if (t->cb.on_upgrade != nullptr) {
            accept = t->cb.on_upgrade(t->cb.user, path.c_str(), &tag0, &status, &body);
        }
        if (!accept) {
            /* Refuse with the application's status and body rather than a bare
             * close, so a client hitting the limit is told why -- matching the
             * mongoose backend, which replied 503 "…limit reached". */
            t->pending_tags.erase(wsi);
            lws_return_http_status(wsi, static_cast<unsigned int>(status),
                body != nullptr ? body : "");
            return -1;
        }
        /* Stash the kind until ESTABLISHED, where per_session_data exists. */
        t->pending_tags[wsi] = tag0;
#ifdef WST_LWS_TRACE
        fprintf(stderr, "[wst-trace] FILTER accept uri=%s tag0=%c wsi=%p\n",
            path.c_str(), tag0 ? (char)tag0 : '?', (void*)wsi);
#endif
        return 0;
    }

    case LWS_CALLBACK_ESTABLISHED: {
        if (ps == nullptr) return -1;
        LwsConn* lc = new (std::nothrow) LwsConn();
        if (lc == nullptr) return -1;
        lc->wsi = wsi;
        uint8_t tag0 = 0;
        auto it = t->pending_tags.find(wsi);
        if (it != t->pending_tags.end()) {
            tag0 = it->second;
            t->pending_tags.erase(it);
        }
        lc->tags[0] = tag0;
#ifdef WST_LWS_TRACE
        fprintf(stderr, "[wst-trace] ESTABLISHED tag0=%c wsi=%p (map %s)\n",
            tag0 ? (char)tag0 : '?', (void*)wsi,
            it != t->pending_tags.end() ? "hit" : "MISS");
#endif
        ps->conn = lc;
        t->conns.push_back(lc);
        if (t->cb.on_open != nullptr) {
            t->cb.on_open(t->cb.user, as_ws(lc), tag0);
        }
        return 0;
    }

    case LWS_CALLBACK_SERVER_WRITEABLE: {
        if (ps == nullptr || ps->conn == nullptr) return 0;
        LwsConn* lc = ps->conn;
        if (lc->outq.empty()) return 0;

        OutMsg& m = lc->outq.front();
        const int w = lws_write(wsi, m.buf.data() + LWS_PRE, m.len,
            m.op == WS_OP_BINARY ? LWS_WRITE_BINARY : LWS_WRITE_TEXT);
        if (w < static_cast<int>(m.len)) {
            /* Partial or failed write: drop the connection rather than send a
             * torn WebSocket frame, which a decoder cannot recover from. */
            return -1;
        }
        lc->outq_bytes -= m.len;
        lc->outq.pop_front();
        if (!lc->outq.empty()) {
            lws_callback_on_writable(wsi);
        }
        return 0;
    }

    case LWS_CALLBACK_CLOSED: {
        if (ps == nullptr || ps->conn == nullptr) return 0;
        LwsConn* lc = ps->conn;
        if (t->cb.on_close != nullptr) {
            t->cb.on_close(t->cb.user, as_ws(lc), lc->tags[0]);
        }
        for (size_t i = 0; i < t->conns.size(); ++i) {
            if (t->conns[i] == lc) {
                t->conns.erase(t->conns.begin() + static_cast<long>(i));
                break;
            }
        }
        delete lc;
        ps->conn = nullptr;
        return 0;
    }

    case LWS_CALLBACK_HTTP_DROP_PROTOCOL:
    case LWS_CALLBACK_CLOSED_HTTP: {
        if (ps != nullptr && ps->http != nullptr) {
            delete ps->http;
            ps->http = nullptr;
        }
        return 0;
    }

    // ---- cross-thread wake ------------------------------------------------
    case LWS_CALLBACK_EVENT_WAIT_CANCELLED: {
        if (t->cb.on_drain != nullptr) {
            t->cb.on_drain(t->cb.user);
        }
        return 0;
    }

    default:
        break;
    }
    return 0;
}

static struct lws_protocols g_protocols[] = {
    { "ds", wst_callback, sizeof(PerSession), 0, 0, nullptr, 0 },
    /* Explicit terminator rather than LWS_PROTOCOL_LIST_TERM: that macro uses
     * designated initialisers and does not compile as C++. */
    { nullptr, nullptr, 0, 0, 0, nullptr, 0 }
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static void wst_event_loop(struct ws_transport* t)
{
    while (t->running.load(std::memory_order_acquire)) {
        /* Bounded so the stop flag is observed even with no traffic. */
        lws_service(t->ctx, 0);
    }
}

struct ws_transport* ws_transport_create(const ws_transport_config_t* cfg,
    const ws_transport_callbacks_t* cb)
{
    if (cfg == nullptr || cb == nullptr || g_t != nullptr) return nullptr;

    struct ws_transport* t = new (std::nothrow) ws_transport();
    if (t == nullptr) return nullptr;
    t->cb = *cb;
    g_t = t;

    /*
     * Without a mount, plain HTTP never reaches the protocol callback: lws
     * handles it internally and answers with its own HTML error page, so
     * /snapshot.jpg 404'd with markup instead of the documented body. Mounting
     * "/" onto the callback protocol routes every non-upgrade request here.
     */
    static struct lws_http_mount mount;
    memset(&mount, 0, sizeof(mount));
    mount.mountpoint = "/";
    mount.mountpoint_len = 1;
    mount.origin_protocol = LWSMPRO_CALLBACK;
    /* Both origin and protocol name the callback protocol. Leaving origin NULL
     * compiles and then dereferences null inside lws during context creation --
     * a load fault at boot, not a diagnostic. lws's own examples set both. */
    mount.origin = "ds";
    mount.protocol = "ds";

    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    info.port = cfg->port;
    info.protocols = g_protocols;
    info.mounts = &mount;
    /* The SDK's lws is built LWS_WITHOUT_EXTENSIONS and warns if this is not
     * explicitly null. */
    info.extensions = nullptr;
    info.gid = -1;
    info.uid = -1;
    /* Without this lws sends a keepalive-less close on idle HTTP; the console
     * polls /snapshot.jpg, so keep transactions cheap. */
    info.keepalive_timeout = 30;

    t->ctx = lws_create_context(&info);
    if (t->ctx == nullptr) {
        fprintf(stderr, "[%s] lws_create_context failed on port %d\n",
            WST_TAG, cfg->port);
        g_t = nullptr;
        delete t;
        return nullptr;
    }

    t->running.store(true, std::memory_order_release);
    t->event_thread = std::thread(wst_event_loop, t);
    return t;
}

void ws_transport_destroy(struct ws_transport* t)
{
    if (t == nullptr) return;
    t->running.store(false, std::memory_order_release);
    /* Break lws_service() out of its wait so the thread notices the flag. */
    if (t->ctx != nullptr) lws_cancel_service(t->ctx);
    if (t->event_thread.joinable()) t->event_thread.join();
    if (t->ctx != nullptr) lws_context_destroy(t->ctx);
    if (g_t == t) g_t = nullptr;
    delete t;
}

void ws_transport_wake(struct ws_transport* t)
{
    if (t == nullptr || t->ctx == nullptr) return;
    /* The one API lws documents as safe from another thread. Coalescing is
     * expected and fine: on_drain re-reads the application queues wholesale. */
    lws_cancel_service(t->ctx);
}

void ws_transport_for_each(struct ws_transport* t,
    void (*fn)(struct ws_conn* c, void* ctx), void* ctx)
{
    if (t == nullptr || fn == nullptr) return;
    /* Snapshot: fn may call ws_conn_send, which does not mutate the list, but
     * a future callback that closes a connection would. */
    std::vector<LwsConn*> snapshot = t->conns;
    for (LwsConn* c : snapshot) {
        fn(as_ws(c), ctx);
    }
}
