/*
 * ws_transport backed by mongoose.
 *
 * The one and only file in this component that touches mongoose. Swapping the
 * HTTP/WebSocket library means adding a sibling ws_transport_<lib>.cpp and
 * selecting it in CMakeLists.txt; debug_stream.cpp does not change.
 *
 * Mapping notes for whoever writes the libwebsockets backend:
 *
 *   ws_transport_wake()   mg_wakeup()            -> lws_cancel_service()
 *   on_drain              MG_EV_WAKEUP           -> LWS_CALLBACK_EVENT_WAIT_CANCELLED
 *   ws_conn_backlog()     c->send.len            -> depth of the backend's own
 *                                                   per-connection send queue
 *   ws_conn_send()        mg_ws_send()           -> append to that queue, then
 *                                                   lws_callback_on_writable(wsi)
 *                                                   (event thread: safe)
 *   ws_transport_for_each mgr->conns walk        -> the backend's own list
 *   tag slots             c->data[0..1]          -> per_session_data
 *
 * mongoose buffers unboundedly inside the connection, so backlog is simply
 * send.len. lws does not buffer for you, so that backend must keep an explicit
 * per-connection queue and report its byte count here. Either way the policy in
 * debug_stream.cpp -- "backlog too deep, drop and resync on the next keyframe"
 * -- is unchanged.
 */

#include "ws_transport.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#include "mongoose.h"

#define WST_TAG "ws_transport"

struct ws_transport {
    struct mg_mgr mgr;
    std::thread event_thread;
    std::atomic<bool> running{false};
    unsigned long listener_id = 0;
    ws_transport_callbacks_t cb{};
};

/*
 * ws_conn is mg_connection. Kept as a cast rather than a wrapper so the
 * mongoose backend adds no per-connection allocation; the lws backend will
 * need a real struct (it has to own the send queue).
 */
static inline struct mg_connection* as_mg(struct ws_conn* c) {
    return reinterpret_cast<struct mg_connection*>(c);
}
static inline const struct mg_connection* as_mg(const struct ws_conn* c) {
    return reinterpret_cast<const struct mg_connection*>(c);
}
static inline struct ws_conn* as_ws(struct mg_connection* c) {
    return reinterpret_cast<struct ws_conn*>(c);
}

// ---------------------------------------------------------------------------
// Per-connection accessors
// ---------------------------------------------------------------------------

uint8_t ws_conn_tag(const struct ws_conn* c, int slot) {
    if (c == NULL || slot < 0 || slot >= WS_CONN_TAG_SLOTS) return 0;
    return (uint8_t)as_mg(c)->data[slot];
}

void ws_conn_set_tag(struct ws_conn* c, int slot, uint8_t value) {
    if (c == NULL || slot < 0 || slot >= WS_CONN_TAG_SLOTS) return;
    as_mg(c)->data[slot] = (char)value;
}

size_t ws_conn_backlog(const struct ws_conn* c) {
    if (c == NULL) return 0;
    return as_mg(c)->send.len;
}

void ws_conn_send(struct ws_conn* c, const void* data, size_t len, ws_op_t op) {
    if (c == NULL || data == NULL || len == 0) return;
    mg_ws_send(as_mg(c), data, len,
               op == WS_OP_BINARY ? WEBSOCKET_OP_BINARY : WEBSOCKET_OP_TEXT);
}

// ---------------------------------------------------------------------------
// Event thread
// ---------------------------------------------------------------------------

static void wst_ev_handler(struct mg_connection* c, int ev, void* ev_data) {
    struct ws_transport* t = (struct ws_transport*)c->fn_data;
    if (t == NULL) return;

    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;

        // hm->uri is not NUL-terminated; copy out a C string for the callback.
        char path[128];
        size_t n = hm->uri.len < sizeof(path) - 1 ? hm->uri.len : sizeof(path) - 1;
        memcpy(path, hm->uri.buf, n);
        path[n] = '\0';

        uint8_t tag0 = 0;
        int status = 404;
        const char* body = "not found\n";
        bool accept = false;
        if (t->cb.on_upgrade != NULL) {
            accept = t->cb.on_upgrade(t->cb.user, path, &tag0, &status, &body);
        }
        if (!accept) {
            mg_http_reply(c, status, "", "%s", body != NULL ? body : "");
            return;
        }

        mg_ws_upgrade(c, hm, NULL);
        c->data[0] = (char)tag0;
        for (int i = 1; i < WS_CONN_TAG_SLOTS; i++) c->data[i] = 0;
        if (t->cb.on_open != NULL) {
            t->cb.on_open(t->cb.user, as_ws(c), tag0);
        }
    } else if (ev == MG_EV_CLOSE) {
        uint8_t tag0 = (uint8_t)c->data[0];
        if (tag0 != 0 && t->cb.on_close != NULL) {
            t->cb.on_close(t->cb.user, as_ws(c), tag0);
        }
    } else if (ev == MG_EV_WAKEUP) {
        if (t->cb.on_drain != NULL) {
            t->cb.on_drain(t->cb.user);
        }
    }
}

static void wst_event_loop(struct ws_transport* t) {
    while (t->running.load(std::memory_order_acquire)) {
        mg_mgr_poll(&t->mgr, 100);
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

struct ws_transport* ws_transport_create(const ws_transport_config_t* cfg,
                                         const ws_transport_callbacks_t* cb) {
    if (cfg == NULL || cb == NULL) return NULL;

    struct ws_transport* t = new (std::nothrow) ws_transport();
    if (t == NULL) return NULL;
    t->cb = *cb;

    mg_mgr_init(&t->mgr);
    if (!mg_wakeup_init(&t->mgr)) {
        fprintf(stderr, "[%s] mg_wakeup_init failed\n", WST_TAG);
        mg_mgr_free(&t->mgr);
        delete t;
        return NULL;
    }

    char url[64];
    snprintf(url, sizeof(url), "http://0.0.0.0:%d", cfg->port);
    struct mg_connection* lc = mg_http_listen(&t->mgr, url, wst_ev_handler, t);
    if (lc == NULL) {
        fprintf(stderr, "[%s] failed to listen on %s\n", WST_TAG, url);
        mg_mgr_free(&t->mgr);
        delete t;
        return NULL;
    }
    t->listener_id = lc->id;

    t->running.store(true, std::memory_order_release);
    t->event_thread = std::thread(wst_event_loop, t);
    return t;
}

void ws_transport_destroy(struct ws_transport* t) {
    if (t == NULL) return;
    t->running.store(false, std::memory_order_release);
    if (t->event_thread.joinable()) {
        t->event_thread.join();
    }
    mg_mgr_free(&t->mgr);
    delete t;
}

void ws_transport_wake(struct ws_transport* t) {
    if (t == NULL) return;
    // Payload is unused: on_drain re-reads the application queues wholesale,
    // so several wakes coalescing into one drain is correct by construction.
    mg_wakeup(&t->mgr, t->listener_id, "w", 1);
}

void ws_transport_for_each(struct ws_transport* t,
                           void (*fn)(struct ws_conn* c, void* ctx), void* ctx) {
    if (t == NULL || fn == NULL) return;
    for (struct mg_connection* c = t->mgr.conns; c != NULL; c = c->next) {
        if (!c->is_websocket) continue;
        fn(as_ws(c), ctx);
    }
}
