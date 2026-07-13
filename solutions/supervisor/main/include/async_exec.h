#ifndef ASYNC_EXEC_H
#define ASYNC_EXEC_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "http_interface.h" // api_status_t
#include "json.hpp"
#include "logger.hpp"
#include "mongoose.h"

using json = nlohmann::json;

// async_exec — worker pool for the supervisor's long/blocking HTTP operations
// (#14). The mongoose server runs a single poll thread; before this change a
// long handler (opkg install ~130s, node-red mode switch ~40s, ...) froze the
// entire management plane while it ran. This manager keeps the poll thread
// free by moving ONLY the blocking part off-thread.
//
// Threading contract (see the #14 spec's five hard constraints):
//   - poll thread  : routing, auth, param parsing, ALL process-internal state
//                    mutation and the final HTTP reply.
//   - worker pool  : ONLY script_timeout()/popen()/sleep(). Never touches the
//                    token map or process-internal state.
//   - handoff      : a finished worker pushes its job id onto a completion
//                    queue, then pokes the poll thread with mg_wakeup(). The
//                    poll thread drains the queue in MG_EV_WAKEUP, runs the
//                    job's commit() (state mutation + response), replies on the
//                    original client connection, then releases the endpoint's
//                    busy gate via finalize().
//
// Why wakeups are routed to the LISTENER connection (always alive) instead of
// the client connection, and why the payload is ignored: mgr->pipe is a UDP
// socketpair (SOCK_DGRAM, see mg_socketpair), so datagrams never coalesce or
// tear, but mongoose's wufn() delivers MG_EV_WAKEUP only to the connection
// whose id matches the first 8 bytes. Routing to the listener guarantees the
// event always fires even if the client disconnected mid-operation; the
// handler then drains the whole completion queue and looks each client up by
// its monotonic (never-reused) id, dropping the reply cleanly if it is gone.
// This makes a closed client connection unable to strand a job or its gate.
class async_exec {
public:
    struct job {
        uint64_t id = 0;
        unsigned long conn_id = 0;
        bool cancelled = false;    // set on MG_EV_CLOSE (poll thread only)
        bool worker_threw = false; // work() threw -> poll replies 500

        std::function<void()> work;                     // worker thread (blocking)
        std::function<api_status_t(job&)> commit;       // poll thread: mutate state, fill res/bytes
        std::function<void()> finalize;                 // poll thread: release gate (ALWAYS)

        json res = json::object();

        // Optional raw-bytes reply (audioRecord streams a WAV instead of JSON).
        bool reply_bytes = false;
        std::string bytes_body;
        std::string bytes_content_type;
    };

    static async_exec& instance()
    {
        static async_exec inst;
        return inst;
    }

    // Poll thread, once, before the mgr loop starts.
    void init(struct mg_mgr* mgr, unsigned long listener_id, int workers = 4)
    {
        _mgr = mgr;
        _listener_id = listener_id;
        _running = true;
        for (int i = 0; i < workers; ++i) {
            _pool.emplace_back([this] { worker_loop(); });
        }
        LOGI("async_exec: %d workers, listener id=%lu", workers, listener_id);
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lk(_q_mutex);
            _running = false;
        }
        _q_cv.notify_all();
        for (auto& t : _pool) {
            if (t.joinable()) {
                t.join();
            }
        }
        _pool.clear();
    }

    static constexpr size_t MAX_PENDING = 32;

    // Poll thread. Enqueue a long operation. On success returns
    // API_STATUS_ASYNC (the caller MUST propagate it so the dispatcher keeps
    // the connection open). On pool saturation, fills res with a busy(-2)
    // body, runs finalize() (release the caller's gate) and returns
    // API_STATUS_OK so the busy reply is sent immediately.
    api_status_t submit(unsigned long conn_id,
        std::function<void()> work,
        std::function<api_status_t(job&)> commit,
        std::function<void()> finalize,
        json& res)
    {
        auto j = std::make_shared<job>();
        j->id = ++_next_id;
        j->conn_id = conn_id;
        j->work = std::move(work);
        j->commit = std::move(commit);
        j->finalize = std::move(finalize);

        {
            std::lock_guard<std::mutex> lk(_q_mutex);
            if (!_running || _queue.size() >= MAX_PENDING) {
                res["code"] = -2;
                res["msg"] = "busy: server is processing too many operations";
                res["data"] = json::object();
                if (j->finalize) {
                    j->finalize();
                }
                return API_STATUS_OK;
            }
            _jobs[j->id] = j;
            _queue.push_back(j->id);
        }
        _q_cv.notify_one();
        return API_STATUS_ASYNC;
    }

    // Poll thread (MG_EV_WAKEUP): commit + reply + finalize every finished job.
    void drain_completions()
    {
        std::vector<uint64_t> done;
        {
            std::lock_guard<std::mutex> lk(_done_mutex);
            done.swap(_done);
        }
        for (uint64_t id : done) {
            std::shared_ptr<job> j;
            {
                std::lock_guard<std::mutex> lk(_q_mutex);
                auto it = _jobs.find(id);
                if (it == _jobs.end()) {
                    continue;
                }
                j = it->second;
                _jobs.erase(it);
            }
            finish_job(j);
        }
    }

    // Poll thread (MG_EV_CLOSE): the client for this connection went away.
    void on_conn_close(unsigned long conn_id)
    {
        std::lock_guard<std::mutex> lk(_q_mutex);
        for (auto& kv : _jobs) {
            if (kv.second->conn_id == conn_id) {
                kv.second->cancelled = true;
            }
        }
    }

private:
    async_exec() = default;
    ~async_exec() { shutdown(); }
    async_exec(const async_exec&) = delete;
    async_exec& operator=(const async_exec&) = delete;

    void finish_job(std::shared_ptr<job> j)
    {
        api_status_t status = API_STATUS_OK;
        if (j->worker_threw) {
            status = API_STATUS_ERROR;
        } else if (j->commit) {
            try {
                status = j->commit(*j);
            } catch (const std::exception& e) {
                LOGE("async commit threw: %s", e.what());
                status = API_STATUS_ERROR;
            } catch (...) {
                LOGE("async commit threw (unknown)");
                status = API_STATUS_ERROR;
            }
        }

        struct mg_connection* c = find_conn(j->conn_id);
        if (c != nullptr && !j->cancelled) {
            reply(c, j, status);
        } else {
            LOGW("async job %llu: client connection gone, reply dropped",
                (unsigned long long)j->id);
        }

        // Release the endpoint busy gate no matter what (even if the client
        // disconnected), otherwise the endpoint would stay wedged at busy(-2).
        if (j->finalize) {
            j->finalize();
        }
    }

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

    void reply(struct mg_connection* c, std::shared_ptr<job> j, api_status_t status)
    {
        if (status == API_STATUS_OK && j->reply_bytes) {
            // Binary reply (audioRecord WAV). Mirror mg_http_reply's framing
            // but binary-safe (mg_send instead of a printf format).
            mg_printf(c,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: %s\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Content-Length: %lu\r\n\r\n",
                j->bytes_content_type.c_str(),
                (unsigned long)j->bytes_body.size());
            mg_send(c, j->bytes_body.data(), j->bytes_body.size());
            c->is_resp = 0;
            return;
        }
        if (status == API_STATUS_OK) {
            mg_http_reply(c, 200,
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Authorization, Content-Type\r\n",
                "%s", j->res.dump().c_str());
        } else if (status == API_STATUS_UNAUTHORIZED) {
            mg_http_reply(c, 401, "Content-Type: text/plain\r\n", "Unauthorized");
        } else {
            mg_http_reply(c, 500, "Content-Type: text/plain\r\n", "Internal Server Error");
        }
    }

    void worker_loop()
    {
        for (;;) {
            uint64_t id = 0;
            std::shared_ptr<job> j;
            {
                std::unique_lock<std::mutex> lk(_q_mutex);
                _q_cv.wait(lk, [this] { return !_running || !_queue.empty(); });
                if (!_running && _queue.empty()) {
                    return;
                }
                id = _queue.front();
                _queue.pop_front();
                auto it = _jobs.find(id);
                if (it == _jobs.end()) {
                    continue; // cancelled/removed before we picked it up
                }
                j = it->second;
            }

            try {
                if (j->work) {
                    j->work();
                }
            } catch (const std::exception& e) {
                LOGE("async worker threw: %s", e.what());
                j->worker_threw = true;
            } catch (...) {
                LOGE("async worker threw (unknown)");
                j->worker_threw = true;
            }

            {
                std::lock_guard<std::mutex> lk(_done_mutex);
                _done.push_back(id);
            }
            // Poke the poll thread. The socket carries ONLY an edge
            // notification ("some job finished"); the real completion is
            // already parked in _done and drain_completions() always scans it
            // in full, so the payload is irrelevant — a single dummy byte
            // suffices (the routing conn_id is prepended by mg_wakeup itself).
            // This is a best-effort poke: a dropped datagram is covered by the
            // unconditional per-cycle drain in http_server's poll loop.
            // Serialize sends so datagrams from multiple workers never
            // interleave at the syscall boundary.
            {
                std::lock_guard<std::mutex> lk(_wake_mutex);
                if (_mgr != nullptr) {
                    static const uint8_t poke = 1;
                    mg_wakeup(_mgr, _listener_id, &poke, sizeof(poke));
                }
            }
        }
    }

    struct mg_mgr* _mgr = nullptr;
    unsigned long _listener_id = 0;
    std::atomic<bool> _running { false };
    std::atomic<uint64_t> _next_id { 0 };

    std::vector<std::thread> _pool;
    std::mutex _q_mutex;
    std::condition_variable _q_cv;
    std::deque<uint64_t> _queue;                    // job ids waiting for a worker
    std::map<uint64_t, std::shared_ptr<job>> _jobs; // all live jobs

    std::mutex _done_mutex;
    std::vector<uint64_t> _done; // finished job ids awaiting drain

    std::mutex _wake_mutex; // serialize mg_wakeup() datagram sends
};

#endif // ASYNC_EXEC_H
