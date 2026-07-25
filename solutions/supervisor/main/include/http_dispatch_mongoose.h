#ifndef HTTP_DISPATCH_MONGOOSE_H
#define HTTP_DISPATCH_MONGOOSE_H

#include <mutex>
#include <string>

#include "http_dispatch.h"
#include "mongoose.h"

/*
 * http_dispatch backed by mongoose.
 *
 * Together with http_server.h this is the only place in the supervisor that
 * touches mongoose from the async path. Swapping the HTTP library means adding
 * a sibling http_dispatch_<lib>.h and changing the one construction site in
 * http_server.h; async_exec.h does not change.
 *
 * Mapping notes for whoever writes the libwebsockets backend:
 *
 *   wake()          mg_wakeup()          -> lws_cancel_service()
 *   (drain trigger) MG_EV_WAKEUP         -> LWS_CALLBACK_EVENT_WAIT_CANCELLED
 *   find_conn()     mgr->conns walk      -> the backend's own wsi table keyed
 *                                           by the same monotonic id
 *   reply_*()       mg_http_reply /      -> lws_add_http_header_* +
 *                   mg_printf + mg_send     lws_write, from a WRITEABLE callback
 *
 * Note the lws backend cannot write a response synchronously the way mongoose
 * does; it has to stage the body and ask for LWS_CALLBACK_HTTP_WRITEABLE. That
 * is invisible above this interface: reply_*() returning true means "accepted
 * for delivery", never "already on the wire".
 */
class http_dispatch_mongoose : public http_dispatch {
public:
    /*
     * mgr and listener_id are owned by http_server and outlive this object.
     * Wakeups are routed to the LISTENER connection (always alive) rather than
     * a client connection: mongoose delivers MG_EV_WAKEUP only to the
     * connection whose id matches the datagram's first 8 bytes, so routing to
     * the listener guarantees the event fires even when the client that
     * started the job has already disconnected.
     */
    http_dispatch_mongoose(struct mg_mgr* mgr, unsigned long listener_id)
        : _mgr(mgr)
        , _listener_id(listener_id)
    {
    }

    void wake() override
    {
        // Serialize sends so datagrams from multiple workers never interleave
        // at the syscall boundary. The payload carries no information -- the
        // completion itself is already parked in async_exec's _done queue and
        // the drain always scans it in full -- so one dummy byte suffices
        // (mg_wakeup prepends the routing conn id itself).
        std::lock_guard<std::mutex> lk(_wake_mutex);
        if (_mgr != nullptr) {
            static const uint8_t poke = 1;
            mg_wakeup(_mgr, _listener_id, &poke, sizeof(poke));
        }
    }

    bool reply_json(unsigned long conn_id, const std::string& body) override
    {
        struct mg_connection* c = find_conn(conn_id);
        if (c == nullptr) {
            return false;
        }
        mg_http_reply(c, 200,
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type\r\n",
            "%s", body.c_str());
        return true;
    }

    bool reply_bytes(unsigned long conn_id, const std::string& content_type,
        const std::string& body) override
    {
        struct mg_connection* c = find_conn(conn_id);
        if (c == nullptr) {
            return false;
        }
        // Mirror mg_http_reply's framing but binary-safe (mg_send instead of a
        // printf format, which would stop at the first NUL).
        mg_printf(c,
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: %s\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: %lu\r\n\r\n",
            content_type.c_str(), (unsigned long)body.size());
        mg_send(c, body.data(), body.size());
        c->is_resp = 0;
        return true;
    }

    bool reply_status(unsigned long conn_id, int http_status,
        const char* content_type, const char* body) override
    {
        struct mg_connection* c = find_conn(conn_id);
        if (c == nullptr) {
            return false;
        }
        std::string hdr = std::string("Content-Type: ")
            + (content_type != nullptr ? content_type : "text/plain") + "\r\n";
        mg_http_reply(c, http_status, hdr.c_str(), "%s", body != nullptr ? body : "");
        return true;
    }

private:
    struct mg_connection* find_conn(unsigned long id)
    {
        if (_mgr == nullptr || id == 0) {
            return nullptr;
        }
        for (struct mg_connection* c = _mgr->conns; c != nullptr; c = c->next) {
            if (c->id == id) {
                return c;
            }
        }
        return nullptr;
    }

    struct mg_mgr* _mgr = nullptr;
    unsigned long _listener_id = 0;
    std::mutex _wake_mutex; // serialize mg_wakeup() datagram sends
};

#endif // HTTP_DISPATCH_MONGOOSE_H
