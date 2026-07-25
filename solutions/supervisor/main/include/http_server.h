#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

/*
 * Backend selection. mongoose is GPL-2.0-only against this Apache-2.0
 * repository (docs/onvif-implementation-spec.md 0.5-B), so it is being
 * replaced by libwebsockets. Both implementations coexist during the
 * transition -- lws uses an lws_ prefix, so nothing collides -- and
 * SUPERVISOR_HTTP_BACKEND_LWS picks the new one. main.cpp is unaffected: it
 * constructs `http_server` and calls start(), which both provide.
 */
#ifdef SUPERVISOR_HTTP_BACKEND_LWS
#include "http_server_lws.h"
using http_server = http_server_lws;
#else

#include "logger.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "api_app.h"
#include "api_audio.h"
#include "api_base.h"
#include "api_device.h"
#include "api_file.h"
#include "api_led.h"
#include "api_user.h"
#include "api_wifi.h"
#include "api_halow.h"
#include "http_dispatch_mongoose.h"
#include "http_request_mongoose.h"

class http_server {
public:
    http_server(const std::string& root_dir = "www", bool gallery_mode = false,
        const std::string& cert = "", const std::string& key = "")
        : _cert(cert.c_str())
        , _key(key.c_str())
        , _root_dir(root_dir.c_str())
    {
        _apis.emplace_back(std::make_unique<api_base>());
        _apis.emplace_back(std::make_unique<api_device>(gallery_mode));
        _apis.emplace_back(std::make_unique<api_audio>());
        _apis.emplace_back(std::make_unique<api_app>());
        _apis.emplace_back(std::make_unique<api_file>());
        _apis.emplace_back(std::make_unique<api_led>());
        _apis.emplace_back(std::make_unique<api_user>());
        _apis.emplace_back(std::make_unique<api_wifi>());
        _apis.emplace_back(std::make_unique<api_halow>());
        mg_mgr_init(&mgr);
    }

    ~http_server()
    {
        stop();
    }

    bool start(const std::string& http_port = "80", const std::string& https_port = "")
    {
        if (!http_port.empty()) {
            http_conn = mg_http_listen(&mgr, std::string(":" + http_port).c_str(),
                event_handler, this);
            if (!http_conn)
                return false;
            LOGV("HTTP server started on %s", http_port.c_str());
        }
        if (!https_port.empty()) {
            https_conn = mg_http_listen(&mgr, std::string(":" + https_port).c_str(),
                https_event_handler, this);
            if (!https_conn)
                return false;
            LOGV("HTTPS server started on %s", https_port.c_str());
        }

        if (!http_conn && !https_conn) {
            LOGV("Error: At least one valid port required");
            return false;
        }

        // #14: enable mg_wakeup() so worker threads can poke the poll thread,
        // and start the async worker pool. Wakeups are routed to the listener
        // connection (always alive), so pick whichever listener exists.
        if (!mg_wakeup_init(&mgr)) {
            LOGE("mg_wakeup_init failed");
            return false;
        }
        unsigned long listener_id = http_conn ? http_conn->id : https_conn->id;
        // The async path reaches mongoose only through this adapter; see
        // http_dispatch.h. Constructed here so it outlives the worker pool
        // (shutdown() joins the pool before this object dies).
        _dispatch = std::make_unique<http_dispatch_mongoose>(&mgr, listener_id);
        async_exec::instance().init(_dispatch.get());

        worker = std::thread([this]() {
            running = true;
            signal(SIGUSR1, [](int sig) {
            });
            while (running) {
                // #14 HIGH#2: bounded poll (was -1/block-forever) + an
                // unconditional drain each cycle as a backstop. mg_wakeup() is
                // only a best-effort low-latency edge poke — mongoose 7.17
                // ignores non-blocking UDP send failures and returns true — so
                // a dropped wakeup datagram could otherwise strand a finished
                // job in _done forever (its commit/finalize never run, gate
                // wedged). drain_completions() is lock-guarded and a no-op when
                // nothing is pending, so this only adds up to ~250ms latency in
                // the rare dropped-wakeup case; the fast path still gets
                // immediate MG_EV_WAKEUP delivery.
                mg_mgr_poll(&mgr, 250);
                async_exec::instance().drain_completions();
            }
            LOGV("poll_loop exit");
        });

        return true;
    }

    void stop()
    {
        LOGV("");
        if (running) {
            running = false;
            pthread_kill(worker.native_handle(), SIGUSR1);
            if (worker.joinable()) {
                worker.join();
            }
            async_exec::instance().shutdown();
            mg_mgr_free(&mgr);
        }
        LOGV("Server stopped");
    }

private:
    const char* _cert;
    const char* _key;
    const char* _root_dir;
    // const std::string _ssi_pattern = "#.html";

    mg_mgr mgr;
    mg_connection* http_conn = nullptr;
    mg_connection* https_conn = nullptr;
    std::unique_ptr<http_dispatch_mongoose> _dispatch;

    std::atomic<bool> running { false };
    std::thread worker;
    std::vector<std::unique_ptr<api_base>> _apis;

    static void event_handler(mg_connection* c, int ev, void* ev_data)
    {
        // #14: a worker finished a long operation. Drain every completed job
        // (commit state, reply on the original connection, release the gate).
        // Routed to the listener connection, so this fires here regardless of
        // which client connection the job belonged to.
        if (ev == MG_EV_WAKEUP) {
            async_exec::instance().drain_completions();
            return;
        }
        // #14: a client connection closed. Mark any in-flight job on it
        // cancelled so its reply is dropped (its worker still finishes and its
        // gate is still released from MG_EV_WAKEUP).
        if (ev == MG_EV_CLOSE) {
            async_exec::instance().on_conn_close(c->id);
            return;
        }
        if (ev == MG_EV_HTTP_MSG) {
            http_server* server = static_cast<http_server*>(c->fn_data);
            mg_http_message* hm = (mg_http_message*)ev_data;

            // #14: remember which connection this request belongs to so an
            // async handler can reply to it later from MG_EV_WAKEUP.
            api_base::set_conn_id(c->id);

            LOGV(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            LOGV("---> uri=%s", std::string(hm->uri.buf, hm->uri.len).c_str());
            LOGV("---> query=%s", std::string(hm->query.buf, hm->query.len).c_str());
            LOGV("---> head=%s", std::string(hm->head.buf, hm->head.len).c_str());
            // LOGV("---> body=%s", std::string(hm->body.buf, hm->body.len).c_str());
            // LOGV(std::string(hm->message.buf, hm->message.len).c_str());
            LOGV("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n");

            // Wrap the library's message in the neutral view the API layer
            // speaks (http_request.h). Zero-copy: it points into hm, and lives
            // exactly as long as this dispatch.
            http_request req = http_request_from_mg(hm);

            json res;
            api_status_t status = api_base::api_handler(&req, res);
            if (status == API_STATUS_ASYNC) {
                // Long operation queued: keep the connection open, the reply
                // is sent from MG_EV_WAKEUP when the worker finishes.
                return;
            }
            if (status == API_STATUS_OK) {
                mg_http_reply(c, 200, "Content-Type: application/json\r\n"
                                      "Access-Control-Allow-Origin: *\r\n"
                                      "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                                      "Access-Control-Allow-Headers: Authorization, Content-Type\r\n",
                    "%s", res.dump().c_str());
                return;
            } else if (status == API_STATUS_UNAUTHORIZED) {
                mg_http_reply(c, 401, "Content-Type: text/plain\r\n", "Unauthorized");
                return;
            } else if (status == API_STATUS_REPLY_FILE) {
                std::string fname("");
                try {
                    LOGV("Reply file: %s", res.dump().c_str());
                    fname = res["data"]["file"].get<std::string>();
                } catch (const json::exception& e) {
                    LOGE("json error: %s", e.what());
                }
                if (fname.empty()) {
                    mg_http_reply(c, 400, "Content-Type: text/plain\r\n", "Bad Request");
                    return;
                }
                struct mg_http_serve_opts _opts = { .root_dir = NULL };
                mg_http_serve_file(c, hm, fname.c_str(), &_opts);
                return;
            } else if (status != API_STATUS_NEXT) {
                mg_http_reply(c, 500, "Content-Type: text/plain\r\n", "Internal Server Error");
                return;
            }

            // redirection
            std::string uri(hm->uri.buf, hm->uri.len);
            if (!(uri == "/"
                    || uri.find(".png") != std::string::npos
                    || uri.find(".svg") != std::string::npos
                    || uri.find(".html") != std::string::npos
                    || uri.find("assets") != std::string::npos
                    || uri.find("js") != std::string::npos)) {
                mg_str* host = mg_http_get_header(hm, "Host");
                std::string redirect(host->buf, host->len);

                // is ip address ?
                std::stringstream ss(redirect);
                std::string segment;
                while (std::getline(ss, segment, '.')) {
                    if (segment.empty() || segment.find(":") != std::string::npos)
                        continue;
                    if (!std::all_of(segment.begin(), segment.end(), ::isdigit)) {
                        redirect = "192.168.16.1";
                        break;
                    }
                }

                redirect = "Location: http://" + redirect + "/\r\n";
                LOGD("redirect============>%s", redirect.c_str());
                mg_http_reply(c, 307, redirect.c_str(), "");
                return;
            }

            // Serve web root directory
            struct mg_http_serve_opts opts = { 0 };
            opts.root_dir = server->_root_dir;
            // opts.ssi_pattern = server->_ssi_pattern;
            mg_http_serve_dir(c, hm, &opts);
        }
    }

    static void https_event_handler(mg_connection* c, int ev, void* ev_data)
    {
        http_server* server = static_cast<http_server*>(c->fn_data);
        if (ev == MG_EV_ACCEPT) {
            struct mg_tls_opts opts;
            memset(&opts, 0, sizeof(opts));
            opts.cert = mg_str(server->_cert);
            opts.key = mg_str(server->_key);
            mg_tls_init(c, &opts);
        } else {
            event_handler(c, ev, ev_data);
        }
    }
};
#endif // SUPERVISOR_HTTP_BACKEND_LWS

#endif // HTTP_SERVER_H
