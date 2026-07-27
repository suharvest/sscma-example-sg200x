#ifndef _HTTP_SERVER_LWS_H_
#define _HTTP_SERVER_LWS_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <libwebsockets.h>

#include "api_app.h"
#include "api_audio.h"
#include "api_base.h"
#include "api_device.h"
#include "api_file.h"
#include "api_halow.h"
#include "api_led.h"
#include "api_user.h"
#include "api_wifi.h"
#include "http_dispatch_lws.h"
#include "http_lws_registry.h"
#include "http_request_lws.h"
#include "logger.hpp"

/*
 * The supervisor's HTTP server on libwebsockets.
 *
 * Same responsibilities as the mongoose version: route /api/* through the API
 * layer, serve the console's static build, keep the async worker pool fed, and
 * redirect stray hostnames to the captive-portal address. The API layer itself
 * is untouched -- it speaks http_request and http_dispatch, both of which have
 * lws backends now.
 *
 * Three structural differences from mongoose, all forced by lws:
 *
 *   Request bodies arrive in chunks (LWS_CALLBACK_HTTP_BODY) and the request
 *   is only dispatched at HTTP_BODY_COMPLETION. mongoose handed over a
 *   complete mg_http_message. The accumulated body lives in the registry.
 *
 *   Responses cannot be written when the handler decides on them; they are
 *   staged and written from HTTP_WRITEABLE. That is why http_dispatch's
 *   contract says "accepted for delivery" rather than "sent".
 *
 *   Static files come from a mount rather than a call, so the ordering between
 *   /api and / is expressed in the mount list instead of in an if-chain.
 */
class http_server_lws {
public:
    http_server_lws(const std::string& root_dir = "www", bool gallery_mode = false,
        const std::string& cert = "", const std::string& key = "")
        : _root_dir(root_dir)
    {
        (void)cert;
        (void)key; /* TLS is compiled out today; see http_server.h history. */
        _apis.emplace_back(std::make_unique<api_base>());
        _apis.emplace_back(std::make_unique<api_device>(gallery_mode));
        _apis.emplace_back(std::make_unique<api_audio>());
        _apis.emplace_back(std::make_unique<api_app>());
        _apis.emplace_back(std::make_unique<api_file>());
        _apis.emplace_back(std::make_unique<api_led>());
        _apis.emplace_back(std::make_unique<api_user>());
        _apis.emplace_back(std::make_unique<api_wifi>());
        _apis.emplace_back(std::make_unique<api_halow>());
        s_self = this;
    }

    ~http_server_lws() { stop(); }

    bool start(const std::string& http_port = "80", const std::string& https_port = "")
    {
        (void)https_port;

        /* Longest mountpoint wins in lws, so /api is matched before /. */
        static struct lws_http_mount mount_api;
        static struct lws_http_mount mount_files;
        static std::string root_copy;
        root_copy = _root_dir;

        memset(&mount_files, 0, sizeof(mount_files));
        mount_files.mountpoint = "/";
        mount_files.mountpoint_len = 1;
        mount_files.origin = root_copy.c_str();
        mount_files.origin_protocol = LWSMPRO_FILE;
        mount_files.def = "index.html";

        memset(&mount_api, 0, sizeof(mount_api));
        mount_api.mountpoint = "/api";
        mount_api.mountpoint_len = 4;
        /* Both fields name the protocol; leaving origin null faults inside
         * lws_create_context rather than failing cleanly. */
        mount_api.origin = "sv";
        mount_api.protocol = "sv";
        mount_api.origin_protocol = LWSMPRO_CALLBACK;
        mount_api.mount_next = &mount_files;

        struct lws_context_creation_info info;
        memset(&info, 0, sizeof(info));
        info.port = std::atoi(http_port.c_str());
        info.protocols = protocols();
        info.mounts = &mount_api;
        info.extensions = nullptr;
        info.gid = -1;
        info.uid = -1;
        info.keepalive_timeout = 30;
        /* Firmware and model uploads are tens of megabytes; the default cap is
         * far below that. Matches what mongoose accepted. */
        info.pt_serv_buf_size = 64 * 1024;

        _ctx = lws_create_context(&info);
        if (_ctx == nullptr) {
            LOGE("lws_create_context failed on port %s", http_port.c_str());
            return false;
        }
        http_lws::Registry::instance().set_context(_ctx);

        _dispatch = std::make_unique<http_dispatch_lws>();
        async_exec::instance().init(_dispatch.get());

        _running = true;
        _worker = std::thread([this] {
            while (_running) {
                lws_service(_ctx, 0);
                /* Backstop, exactly as the mongoose loop had: a dropped wake
                 * must not strand a finished job with its gate held. */
                async_exec::instance().drain_completions();
            }
            LOGV("poll_loop exit");
        });
        LOGV("HTTP server started on %s", http_port.c_str());
        return true;
    }

    void stop()
    {
        if (!_running) return;
        _running = false;
        if (_ctx != nullptr) lws_cancel_service(_ctx);
        if (_worker.joinable()) _worker.join();
        async_exec::instance().shutdown();
        if (_ctx != nullptr) {
            lws_context_destroy(_ctx);
            _ctx = nullptr;
        }
        LOGV("Server stopped");
    }

private:
    static struct lws_protocols* protocols()
    {
        static struct lws_protocols p[] = {
            { "sv", callback, 0, 0, 0, nullptr, 0 },
            { nullptr, nullptr, 0, 0, 0, nullptr, 0 }
        };
        return p;
    }

    static int callback(struct lws* wsi, enum lws_callback_reasons reason,
        void* user, void* in, size_t len)
    {
        (void)user;
        auto& reg = http_lws::Registry::instance();

        switch (reason) {
        case LWS_CALLBACK_HTTP: {
            http_lws::Conn* c = reg.attach(wsi);
            c->body.clear();
            c->reply.staged = false;
            /* A GET has no body and is dispatched now; a POST waits for
             * HTTP_BODY_COMPLETION. PUT is not used by this API and its token
             * is not compiled into this lws build. */
            if (lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI) == 0) {
                return dispatch(wsi, c);
            }
            return 0;
        }

        case LWS_CALLBACK_HTTP_BODY: {
            http_lws::Conn* c = reg.find(wsi);
            if (c == nullptr) return -1;
            c->body.append(static_cast<const char*>(in), len);
            return 0;
        }

        case LWS_CALLBACK_HTTP_BODY_COMPLETION: {
            http_lws::Conn* c = reg.find(wsi);
            if (c == nullptr) return -1;
            return dispatch(wsi, c);
        }

        case LWS_CALLBACK_HTTP_WRITEABLE: {
            http_lws::Conn* c = reg.find(wsi);
            if (c == nullptr || !c->reply.staged) return 0;
            return write_reply(wsi, c);
        }

        case LWS_CALLBACK_CLOSED_HTTP:
        case LWS_CALLBACK_HTTP_DROP_PROTOCOL: {
            http_lws::Conn* c = reg.find(wsi);
            if (c != nullptr) {
                /* Mark any in-flight async job so its reply is dropped; the
                 * worker still finishes and the gate is still released. */
                async_exec::instance().on_conn_close(c->id);
                reg.detach(wsi);
            }
            return 0;
        }

        case LWS_CALLBACK_EVENT_WAIT_CANCELLED:
            async_exec::instance().drain_completions();
            return 0;

        default:
            break;
        }
        return 0;
    }

    /* Runs the API layer for one complete request. */
    static int dispatch(struct lws* wsi, http_lws::Conn* c)
    {
        http_lws::RequestCtx ctx;
        ctx.wsi = wsi;
        ctx.conn = c;
        ctx.uri = http_lws::hdr_value(wsi, WSI_TOKEN_GET_URI);
        if (ctx.uri.empty()) ctx.uri = http_lws::hdr_value(wsi, WSI_TOKEN_POST_URI);
        ctx.query = http_lws::hdr_value(wsi, WSI_TOKEN_HTTP_URI_ARGS);

        http_request req = http_lws::make_request(ctx);

        /* Which connection an async handler must reply to later. */
        api_base::set_conn_id(c->id);

        json res;
        api_status_t status = api_base::api_handler(&req, res);

        if (status == API_STATUS_ASYNC) {
            /* Keep the connection open; the reply is staged by http_dispatch
             * when the worker finishes. */
            return 0;
        }
        if (status == API_STATUS_OK) {
            stage(c, 200, "application/json", res.dump(),
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Authorization, Content-Type\r\n");
            lws_callback_on_writable(wsi);
            return 0;
        }
        if (status == API_STATUS_UNAUTHORIZED) {
            stage(c, 401, "text/plain", "Unauthorized", "");
            lws_callback_on_writable(wsi);
            return 0;
        }
        if (status == API_STATUS_REPLY_FILE) {
            std::string fname;
            try {
                fname = res["data"]["file"].get<std::string>();
            } catch (const json::exception& e) {
                LOGE("json error: %s", e.what());
            }
            if (fname.empty()) {
                stage(c, 400, "text/plain", "Bad Request", "");
                lws_callback_on_writable(wsi);
                return 0;
            }
            /* lws serves the file itself; nothing is staged. */
            return lws_serve_http_file(wsi, fname.c_str(), nullptr, nullptr, 0) < 0 ? -1 : 0;
        }
        if (status != API_STATUS_NEXT) {
            stage(c, 500, "text/plain", "Internal Server Error", "");
            lws_callback_on_writable(wsi);
            return 0;
        }

        /* API_STATUS_NEXT under the /api mount means no handler matched. The
         * mongoose build fell through to the captive-portal redirect here;
         * static assets never reach this path because they are served by the
         * file mount. */
        return redirect(wsi, c, ctx);
    }

    /* Captive-portal behaviour, preserved from the mongoose implementation:
     * a Host that is not a bare IP is bounced to the AP address. */
    static int redirect(struct lws* wsi, http_lws::Conn* c,
        const http_lws::RequestCtx& ctx)
    {
        std::string host = http_lws::hdr_value(wsi, WSI_TOKEN_HOST);
        std::stringstream ss(host);
        std::string segment;
        std::string target = host;
        while (std::getline(ss, segment, '.')) {
            if (segment.empty() || segment.find(':') != std::string::npos) continue;
            if (!std::all_of(segment.begin(), segment.end(), ::isdigit)) {
                target = "192.168.16.1";
                break;
            }
        }
        LOGD("redirect============>%s", target.c_str());
        stage(c, 307, "text/plain", "",
            ("Location: http://" + target + "/\r\n").c_str());
        lws_callback_on_writable(wsi);
        return 0;
    }

    static void stage(http_lws::Conn* c, int status, const std::string& ctype,
        const std::string& body, const char* extra)
    {
        c->reply.status = status;
        c->reply.content_type = ctype;
        c->reply.extra_headers = extra != nullptr ? extra : "";
        c->reply.body = body;
        c->reply.sent = 0;
        c->reply.headers_sent = false;
        c->reply.staged = true;
    }

    static int write_reply(struct lws* wsi, http_lws::Conn* c)
    {
        http_lws::PendingReply& r = c->reply;

        if (!r.headers_sent) {
            uint8_t hdr[LWS_PRE + 1024];
            uint8_t* p = hdr + LWS_PRE;
            uint8_t* end = hdr + sizeof(hdr) - 1;
            if (lws_add_http_common_headers(wsi,
                    static_cast<unsigned int>(r.status), r.content_type.c_str(),
                    static_cast<lws_filepos_t>(r.body.size()), &p, end))
                return -1;
            /* extra_headers arrives as raw "Name: value\r\n" lines, the shape
             * the mongoose code used; split and add them individually. */
            size_t off = 0;
            while (off < r.extra_headers.size()) {
                const size_t nl = r.extra_headers.find("\r\n", off);
                if (nl == std::string::npos) break;
                const std::string line = r.extra_headers.substr(off, nl - off);
                off = nl + 2;
                const size_t colon = line.find(':');
                if (colon == std::string::npos) continue;
                std::string name = line.substr(0, colon + 1);
                std::string value = line.substr(colon + 1);
                while (!value.empty() && value.front() == ' ') value.erase(0, 1);
                if (lws_add_http_header_by_name(wsi,
                        reinterpret_cast<const unsigned char*>(name.c_str()),
                        reinterpret_cast<const unsigned char*>(value.c_str()),
                        static_cast<int>(value.size()), &p, end))
                    return -1;
            }
            if (lws_finalize_write_http_header(wsi, hdr + LWS_PRE, &p, end))
                return -1;
            r.headers_sent = true;
            if (r.body.empty()) {
                r.staged = false;
                return lws_http_transaction_completed(wsi) ? -1 : 0;
            }
            lws_callback_on_writable(wsi);
            return 0;
        }

        const size_t chunk = 8192;
        const size_t n = std::min(chunk, r.body.size() - r.sent);
        std::vector<uint8_t> out(LWS_PRE + n);
        memcpy(out.data() + LWS_PRE, r.body.data() + r.sent, n);
        const bool last = (r.sent + n >= r.body.size());
        if (lws_write(wsi, out.data() + LWS_PRE, n,
                last ? LWS_WRITE_HTTP_FINAL : LWS_WRITE_HTTP) < static_cast<int>(n))
            return -1;
        r.sent += n;
        if (!last) {
            lws_callback_on_writable(wsi);
            return 0;
        }
        r.staged = false;
        return lws_http_transaction_completed(wsi) ? -1 : 0;
    }

    std::string _root_dir;
    struct lws_context* _ctx = nullptr;
    std::atomic<bool> _running { false };
    std::thread _worker;
    std::vector<std::unique_ptr<api_base>> _apis;
    std::unique_ptr<http_dispatch_lws> _dispatch;
    static inline http_server_lws* s_self = nullptr;
};

#endif // _HTTP_SERVER_LWS_H_
