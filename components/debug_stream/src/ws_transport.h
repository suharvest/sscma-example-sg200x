#ifndef _WS_TRANSPORT_H_
#define _WS_TRANSPORT_H_

#include <stddef.h>
#include <stdint.h>

/*
 * ws_transport: the seam between debug_stream's streaming *policy* (queueing,
 * keyframe gating, drop-under-backpressure) and the HTTP/WebSocket library
 * underneath it.
 *
 * Component-internal: not installed, not part of debug_stream.h. Only
 * debug_stream.cpp and the ws_transport_*.cpp backends include this.
 *
 * Why this exists
 * ---------------
 * mongoose is GPL-2.0-only, which is incompatible with this repository's
 * Apache-2.0 license, so the HTTP/WS layer has to be replaceable. This header
 * is deliberately shaped after libwebsockets rather than mongoose, because lws
 * is the more constrained of the two:
 *
 *   - lws forbids touching a connection from any thread but the event thread;
 *     the single exception is lws_cancel_service(). mongoose is equally happy
 *     with that discipline, so the interface enforces it for both.
 *   - lws has no "peek at the socket send buffer" primitive, so backlog is
 *     defined as bytes accepted from the application but not yet written,
 *     which a backend can always account for itself.
 *
 * Emulating this interface on mongoose is trivial; emulating a
 * mongoose-shaped interface on lws is not. Hence the direction.
 *
 * Threading contract (both backends, and the reason this seam exists)
 * ------------------------------------------------------------------
 *   ws_transport_wake()   MAY be called from any thread. It is the ONLY call
 *                         that may. Producers (VENC callback, inference loop)
 *                         use nothing else.
 *   everything else       Event thread only, i.e. from inside one of the
 *                         ws_transport_callbacks.
 *
 * Violating this is the classic lws bug: lws_callback_on_writable() is not
 * thread-safe, and misuse fails intermittently rather than immediately.
 */

struct ws_transport;
struct ws_conn;

typedef enum {
    WS_OP_TEXT,
    WS_OP_BINARY,
} ws_op_t;

/* Number of per-connection tag slots (see ws_conn_tag). */
#define WS_CONN_TAG_SLOTS 2

/*
 * Per-connection application tags. debug_stream uses:
 *   slot 0 -> connection kind (video / results)
 *   slot 1 -> resync state (awaiting keyframe / synced)
 * Event thread only.
 */
uint8_t ws_conn_tag(const struct ws_conn* c, int slot);
void ws_conn_set_tag(struct ws_conn* c, int slot, uint8_t value);

/*
 * Bytes accepted from the application for this connection but not yet handed
 * to the kernel. This is the backpressure signal: a large value means the peer
 * is not draining and the application should drop rather than buffer without
 * bound. Event thread only.
 */
size_t ws_conn_backlog(const struct ws_conn* c);

/*
 * Queue one WebSocket message. Never blocks; the backend owns the buffering.
 * Event thread only.
 */
void ws_conn_send(struct ws_conn* c, const void* data, size_t len, ws_op_t op);

typedef struct {
    void* user;

    /*
     * An HTTP upgrade request arrived at `path`. Return true to accept the
     * upgrade, storing the connection kind in *tag0 (it lands in tag slot 0).
     * Return false to refuse, setting *status and *body for the HTTP error
     * reply (both have sane defaults, 404 / "not found\n").
     */
    bool (*on_upgrade)(void* user, const char* path, uint8_t* tag0,
                       int* status, const char** body);

    /* WebSocket established, after on_upgrade returned true. */
    void (*on_open)(void* user, struct ws_conn* c, uint8_t tag0);

    /* Connection gone. Not called for requests refused by on_upgrade. */
    void (*on_close)(void* user, struct ws_conn* c, uint8_t tag0);

    /*
     * ws_transport_wake() was called. Drain the application queues now,
     * reaching connections with ws_transport_for_each(). Runs on the event
     * thread. Coalescing is expected: several wakes may collapse into one
     * on_drain, and a spurious on_drain with nothing queued is legal.
     */
    void (*on_drain)(void* user);
} ws_transport_callbacks_t;

typedef struct {
    int port;
} ws_transport_config_t;

/*
 * Create the listener and start the event thread. Returns NULL on failure
 * (nothing is left running). The callbacks struct is copied.
 */
struct ws_transport* ws_transport_create(const ws_transport_config_t* cfg,
                                         const ws_transport_callbacks_t* cb);

/* Stop the event thread, close every connection, release resources. */
void ws_transport_destroy(struct ws_transport* t);

/*
 * Kick the event thread so it runs on_drain. Safe from any thread; see the
 * threading contract above. Cheap enough to call per encoded frame.
 */
void ws_transport_wake(struct ws_transport* t);

/*
 * Visit every live WebSocket connection. Event thread only, normally from
 * on_drain. `fn` may call ws_conn_send / ws_conn_backlog / ws_conn_*_tag.
 */
void ws_transport_for_each(struct ws_transport* t,
                           void (*fn)(struct ws_conn* c, void* ctx), void* ctx);

#endif /* _WS_TRANSPORT_H_ */
