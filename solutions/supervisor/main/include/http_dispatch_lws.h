#ifndef _HTTP_DISPATCH_LWS_H_
#define _HTTP_DISPATCH_LWS_H_

#include <string>

#include <libwebsockets.h>

#include "http_dispatch.h"
#include "http_lws_registry.h"

/*
 * http_dispatch backed by libwebsockets.
 *
 * The difference that shapes this file: mongoose could write a response the
 * moment async_exec asked, so reply_json() called mg_http_reply() and was
 * done. lws only writes from LWS_CALLBACK_HTTP_WRITEABLE, so a reply is staged
 * on the connection and a writable slot requested; http_server_lws drains it.
 *
 * That is invisible above the interface, which is why http_dispatch.h says
 * reply_*() returning true means "accepted for delivery", never "already on
 * the wire". async_exec is unchanged.
 */
class http_dispatch_lws : public http_dispatch {
public:
    void wake() override
    {
        /* The only lws call documented as safe from another thread, and the
         * only place in this class reached off the event thread. Coalescing is
         * fine: drain_completions() scans the whole queue. */
        struct lws_context* ctx = http_lws::Registry::instance().context();
        if (ctx != nullptr) {
            lws_cancel_service(ctx);
        }
    }

    bool reply_json(unsigned long conn_id, const std::string& body) override
    {
        return stage(conn_id, 200, "application/json", body,
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type\r\n");
    }

    bool reply_bytes(unsigned long conn_id, const std::string& content_type,
        const std::string& body) override
    {
        return stage(conn_id, 200, content_type, body,
            "Access-Control-Allow-Origin: *\r\n");
    }

    bool reply_status(unsigned long conn_id, int http_status,
        const char* content_type, const char* body) override
    {
        return stage(conn_id, http_status,
            content_type != nullptr ? content_type : "text/plain",
            body != nullptr ? body : "", "");
    }

private:
    static bool stage(unsigned long conn_id, int status,
        const std::string& ctype, const std::string& body,
        const char* extra)
    {
        http_lws::Conn* c = http_lws::Registry::instance().find_by_id(conn_id);
        if (c == nullptr || c->wsi == nullptr) {
            /* Client went away. Not an error: async_exec logs it and still
             * runs finalize(), so the endpoint gate is released either way. */
            return false;
        }
        c->reply.status = status;
        c->reply.content_type = ctype;
        c->reply.extra_headers = extra != nullptr ? extra : "";
        c->reply.body = body;
        c->reply.sent = 0;
        c->reply.headers_sent = false;
        c->reply.staged = true;
        lws_callback_on_writable(c->wsi);
        return true;
    }
};

#endif // _HTTP_DISPATCH_LWS_H_
