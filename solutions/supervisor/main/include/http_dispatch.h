#ifndef HTTP_DISPATCH_H
#define HTTP_DISPATCH_H

#include <string>

/*
 * http_dispatch: the seam between async_exec's job/gate policy and the HTTP
 * library underneath it.
 *
 * Why this exists
 * ---------------
 * mongoose is GPL-2.0-only, which is incompatible with this repository's
 * Apache-2.0 license, so the HTTP layer has to be replaceable. This interface
 * is shaped after libwebsockets rather than mongoose, because lws is the more
 * constrained of the two:
 *
 *   - lws forbids touching a connection from any thread but the event thread.
 *     The single exception is lws_cancel_service(), which is exactly wake().
 *     mongoose is equally happy with that discipline, so it is enforced here
 *     for both.
 *   - Connections are addressed by a stable, monotonic, never-reused id rather
 *     than by pointer. Both libraries can supply one, and it is what makes a
 *     disconnected client unable to strand a job (the lookup simply fails and
 *     the reply is dropped, while commit/finalize still run).
 *
 * Threading contract
 * ------------------
 *   wake()          MAY be called from any thread. It is the ONLY one that may.
 *   everything else Event thread only.
 *
 * Mirrors components/debug_stream/src/ws_transport.h, which does the same job
 * for the WebSocket side. Kept separate on purpose: this one is about
 * request/response on an already-routed connection, that one is about
 * broadcast with backpressure. Merging them would only produce a union of two
 * unrelated shapes.
 */
class http_dispatch {
public:
    virtual ~http_dispatch() = default;

    /*
     * Poke the event thread so it drains finished jobs. Safe from a worker
     * thread; see the threading contract above.
     *
     * Best-effort by design: an implementation may drop the notification (the
     * mongoose backend rides a non-blocking UDP socketpair and mongoose 7.17
     * ignores send failures). Callers must therefore also drain unconditionally
     * once per event-loop cycle, otherwise a dropped poke would strand a
     * finished job forever with its endpoint gate wedged.
     */
    virtual void wake() = 0;

    /*
     * Replies, event thread only. Each returns false when conn_id no longer
     * refers to a live connection, i.e. the client went away and the reply was
     * dropped. That is an expected outcome, not an error: the caller still has
     * to run its finalize() to release the endpoint gate.
     */

    /* 200 + application/json + the CORS header set used across the API. */
    virtual bool reply_json(unsigned long conn_id, const std::string& body) = 0;

    /* 200 + caller-chosen content type + raw bytes (audioRecord's WAV). */
    virtual bool reply_bytes(unsigned long conn_id,
                             const std::string& content_type,
                             const std::string& body) = 0;

    /* Arbitrary status with a plain body (401 Unauthorized, 500, ...). */
    virtual bool reply_status(unsigned long conn_id, int http_status,
                              const char* content_type, const char* body) = 0;
};

#endif // HTTP_DISPATCH_H
