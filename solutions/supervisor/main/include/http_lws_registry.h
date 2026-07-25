#ifndef _HTTP_LWS_REGISTRY_H_
#define _HTTP_LWS_REGISTRY_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <libwebsockets.h>

/*
 * Connection registry shared by the libwebsockets HTTP server and the
 * http_dispatch backend that replies on its connections.
 *
 * Two things mongoose gave for free and lws does not:
 *
 *   A stable connection id. async_exec addresses a client by a monotonic,
 *   never-reused id so a worker's reply can find its connection later, or
 *   discover it is gone. mongoose had mg_connection::id; lws has no equivalent,
 *   so ids are assigned here.
 *
 *   Synchronous writes. mg_http_reply() could be called at any time; lws only
 *   writes from LWS_CALLBACK_HTTP_WRITEABLE, so a reply is staged here and the
 *   connection is asked for a writable slot.
 *
 * Event thread only. The one cross-thread entry point in the whole design is
 * lws_cancel_service(), which http_dispatch_lws::wake() calls.
 */
namespace http_lws {

struct PendingReply {
    int status = 200;
    std::string content_type;
    std::string extra_headers; /* CRLF-terminated lines, may be empty */
    std::string body;
    size_t sent = 0;
    /* Distinct from sent == 0: the headers go out on the first writable slot
     * and the body on later ones, so "nothing sent yet" and "headers not sent
     * yet" are different questions. Conflating them re-emits the header block
     * into the body stream, which presents as a response whose body starts
     * with "HTTP/1.1 200 OK". */
    bool headers_sent = false;
    bool staged = false;
};

struct Conn {
    struct lws* wsi = nullptr;
    unsigned long id = 0;
    std::string body;      /* accumulated request body, see multipart note */
    PendingReply reply;
};

/*
 * Registry singleton. A singleton rather than an instance because
 * lws_protocols callbacks carry no user pointer of ours, the supervisor runs
 * exactly one HTTP server, and async_exec is itself a singleton -- inventing
 * plumbing to pass an instance around would be ceremony without a second user.
 */
class Registry {
public:
    static Registry& instance()
    {
        static Registry r;
        return r;
    }

    Conn* attach(struct lws* wsi)
    {
        Conn* c = &conns_[wsi];
        c->wsi = wsi;
        if (c->id == 0) {
            c->id = ++next_id_;
            by_id_[c->id] = wsi;
        }
        return c;
    }

    void detach(struct lws* wsi)
    {
        auto it = conns_.find(wsi);
        if (it == conns_.end()) return;
        by_id_.erase(it->second.id);
        conns_.erase(it);
    }

    Conn* find(struct lws* wsi)
    {
        auto it = conns_.find(wsi);
        return it == conns_.end() ? nullptr : &it->second;
    }

    /* Null when the client has gone. Callers treat that as "reply dropped",
     * which is expected rather than an error. */
    Conn* find_by_id(unsigned long id)
    {
        auto it = by_id_.find(id);
        if (it == by_id_.end()) return nullptr;
        return find(it->second);
    }

    void set_context(struct lws_context* ctx) { ctx_ = ctx; }
    struct lws_context* context() const { return ctx_; }

private:
    Registry() = default;
    std::unordered_map<struct lws*, Conn> conns_;
    std::unordered_map<unsigned long, struct lws*> by_id_;
    unsigned long next_id_ = 0;
    struct lws_context* ctx_ = nullptr;
};

} // namespace http_lws

#endif // _HTTP_LWS_REGISTRY_H_
