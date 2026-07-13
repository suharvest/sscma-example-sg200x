#ifndef API_BASE_H
#define API_BASE_H

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "logger.hpp"

#include "async_exec.h"
#include "http_interface.h"
#include "version.h"

#define REG_API_FULL(__uri, __handler, __no_auth) \
    register_api(__uri, __handler, __no_auth)
#define REG_API(__handler) \
    REG_API_FULL(#__handler, __handler, false)
#define REG_API_NO_AUTH(__handler) \
    REG_API_FULL(#__handler, __handler, true)

class rest_api {
public:
    rest_api(const api_handler_t handler, bool no_auth = false)
        : _handler(handler)
        , _no_auth(no_auth)
    {
    }

    virtual ~rest_api() = default;

    api_status_t operator()(request_t req, response_t res) { return _handler(req, res); }

    inline bool no_auth() const { return _no_auth; }

private:
    const api_handler_t _handler;
    const bool _no_auth;
};

class api_base : public http_interface {
public:
    api_base(std::string group = "")
        : _group(group)
    {
        if (_group.empty()) {
            REG_API_NO_AUTH(version);
        }
    }

    virtual ~api_base() = default;

    static void set_force_no_auth(bool no_auth) { _force_no_auth = no_auth; }
    static void set_script(std::string dir) { _script = dir; }

    void register_api(std::string uri, api_handler_t handler, bool no_auth = false)
    {
        _api_map[_group.empty() ? uri : _group + "/" + uri] = std::make_unique<rest_api>(handler, no_auth);
    }

    static api_status_t api_handler(request_t req, response_t res)
    {
        std::string uri = get_uri(req);
        auto pos = uri.find("/api/");
        if (pos == std::string::npos) {
            return API_STATUS_NEXT;
        }
        uri = uri.substr(pos + 5);

        auto api = _api_map.find(uri);
        if (api == _api_map.end()) {
            for (auto& [key, value] : _api_map) {
                if (uri.find(key) == 0) {
                    return (*value)(req, res);
                }
            }
            LOGE("API not implemented: %s", uri.c_str());
            return API_STATUS_NEXT;
        }
        if (!_force_no_auth && !api->second->no_auth()) {
            std::string token = get_param(req, "Authorization");
            if (token.empty() || !check_token(token)) {
                LOGE("Unauthorized: %s", uri.c_str());
                return API_STATUS_UNAUTHORIZED;
            }
        }
        return (*api->second)(req, res);
    }

    template <typename... Args>
    static std::string script(const std::string& cmd, Args&&... args)
    {
        return script_timeout(30, cmd, std::forward<Args>(args)...);
    }

    // Same as script() but with an explicit read timeout. Needed for slow
    // operations like `app_install` (opkg has a 120s budget in main.sh; the
    // caller passes a slightly larger value so main.sh times out first and
    // its "EXIT:124" marker is still captured here).
    template <typename... Args>
    static std::string script_timeout(int timeout_sec, const std::string& cmd, Args&&... args)
    {
        // #14 HIGH#1: fork + execv (was popen) so the timeout is a HARD upper
        // bound, not merely a read-loop break followed by an unbounded
        // pclose()/wait(). The whole management plane's recovery guarantee
        // (setRunMode / forceConsole must finish within budget so the mode
        // gate is always released) rests on this primitive never blocking
        // forever.
        //
        // Each arg is passed as an INDEPENDENT argv element (never spliced into
        // a shell string), which removes the shell-injection surface entirely;
        // main.sh still sees them as $1/$2/... exactly as before. We exec the
        // script directly so its shebang (#!/bin/bash) is honored — identical
        // to the old popen("sh -c \"main.sh ...\"") which also ran main.sh
        // under bash. (main.sh relies on bash-only features, so it must NOT be
        // forced under /bin/sh = busybox.)
        std::vector<std::string> argv_str;
        argv_str.reserve(2 + sizeof...(args));
        argv_str.push_back(_script);
        argv_str.push_back(cmd);
        auto push_arg = [&argv_str](auto&& a) {
            std::stringstream s;
            s << a;
            argv_str.push_back(s.str());
        };
        (push_arg(std::forward<Args>(args)), ...);

        std::vector<char*> argv;
        argv.reserve(argv_str.size() + 1);
        for (auto& s : argv_str) {
            argv.push_back(const_cast<char*>(s.c_str()));
        }
        argv.push_back(nullptr);

        // For logging parity with the old popen() path.
        std::string full_cmd;
        for (const auto& s : argv_str) {
            full_cmd += s;
            full_cmd += ' ';
        }
        LOGV("Executing: %s", full_cmd.c_str());

        int pipefd[2];
        if (pipe(pipefd) != 0) {
            LOGE("pipe() failed: %s, errno=%d, strerror=%s",
                full_cmd.c_str(), errno, strerror(errno));
            return "";
        }

        pid_t pid = fork();
        if (pid < 0) {
            LOGE("fork() failed: %s, errno=%d, strerror=%s",
                full_cmd.c_str(), errno, strerror(errno));
            close(pipefd[0]);
            close(pipefd[1]);
            return "";
        }

        if (pid == 0) {
            // Child: only async-signal-safe calls before execv().
            setpgid(0, 0); // own process group (pgid == child pid)
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);
            execv(argv[0], argv.data()); // honors main.sh's #!/bin/bash shebang
            _exit(127);                  // execv failed
        }

        // Parent.
        close(pipefd[1]);    // parent only reads
        setpgid(pid, pid);   // race-free vs child (harmless EACCES/ESRCH if it lost)

        // non-blocking read
        int fd = pipefd[0];
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        std::vector<char> buffer(512);
        std::string result = "";
        time_t start_time = time(nullptr);
        bool timed_out = false;

        while (true) {
            if (time(nullptr) - start_time > timeout_sec) {
                LOGE("Command timeout after %d seconds: %s", timeout_sec, full_cmd.c_str());
                timed_out = true;
                break;
            }

            ssize_t bytes_read = read(fd, buffer.data(), buffer.size());
            if (bytes_read > 0) {
                result.append(buffer.data(), bytes_read);
            } else if (bytes_read == 0) {
                break; // EOF: child closed stdout
            } else {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    LOGE("read() error: %s, errno=%d", strerror(errno), errno);
                    break;
                }
                usleep(1000); // 1ms
            }
        }
        close(fd);

        // HARD deadline reap — never an unbounded wait. Normal completion
        // reaps immediately; a wedged child (possibly ignoring SIGTERM) is
        // escalated on its WHOLE process group: SIGTERM -> bounded 3s grace ->
        // SIGKILL -> bounded final reap. killpg() targets pgid == child pid
        // (set above), so descendants the child spawned are torn down too.
        int status = 0;
        if (!reap_bounded(pid, status, timed_out ? 0 : 200)) {
            killpg(pid, SIGTERM);
            if (!reap_bounded(pid, status, 3000)) {
                LOGE("Command ignored SIGTERM, sending SIGKILL: %s", full_cmd.c_str());
                killpg(pid, SIGKILL);
                if (!reap_bounded(pid, status, 2000)) {
                    LOGE("Command unreapable after SIGKILL (D-state?): %s", full_cmd.c_str());
                }
            }
        }

        if (!timed_out && !WIFEXITED(status)) {
            LOGE("Command terminated abnormally: %s, status=%d", full_cmd.c_str(), status);
        }

        // Strip trailing newlines
        if (!result.empty()) {
            size_t end = result.find_last_not_of("\r\n");
            if (end != std::string::npos) {
                result = result.substr(0, end + 1);
            } else {
                result.clear();
            }
        }

        LOGV("Completed: [%d] %s", result.size(), result.c_str());
        return result;
    }

    // #14: enqueue the blocking part of a long operation on the worker pool.
    // Must be called on the poll thread from inside a handler that has already
    // passed auth, acquired its busy gate and parsed every parameter it needs
    // into owning storage captured by the lambdas.
    //   work     : worker thread; only script_timeout()/popen()/sleep() and
    //              writing into captured owning storage. Never state/_tokens.
    //   commit   : poll thread; mutate process-internal state and fill res,
    //              return the reply status.
    //   finalize : poll thread; release the endpoint busy gate (runs even if
    //              the client disconnected).
    // Returns API_STATUS_ASYNC when queued (propagate it); on saturation fills
    // res with busy(-2), releases the gate and returns API_STATUS_OK.
    static api_status_t submit_async(std::function<void()> work,
        std::function<api_status_t(json&)> commit,
        std::function<void()> finalize,
        response_t res)
    {
        auto commit_res = std::move(commit);
        return async_exec::instance().submit(conn_id(),
            std::move(work),
            [commit_res](async_exec::job& j) -> api_status_t { return commit_res(j.res); },
            std::move(finalize), res);
    }

    // Raw variant: commit receives the async_exec::job so it can also produce a
    // binary reply (job.reply_bytes/bytes_body/bytes_content_type), used by
    // audioRecord to stream a WAV. Same threading contract as submit_async.
    static api_status_t submit_async_raw(std::function<void()> work,
        std::function<api_status_t(async_exec::job&)> commit,
        std::function<void()> finalize,
        response_t res)
    {
        return async_exec::instance().submit(conn_id(),
            std::move(work), std::move(commit), std::move(finalize), res);
    }

    template <typename T>
    static json parse_result(T&& result) { return to_json(result); }

    using vvstr_t = std::vector<std::vector<std::string>>;
    static vvstr_t parse_result(std::string file, char delimiter, bool skip_header = false)
    {
        vvstr_t result;
        std::ifstream f(file);
        if (!f.is_open()) {
            LOGE("Failed to open file: %s", file.c_str());
            return result;
        }

        std::string line;
        if (skip_header)
            std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty())
                continue;
            std::stringstream ss(std::move(line));
            std::string field;
            std::vector<std::string> fields;
            while (std::getline(ss, field, delimiter)) {
                fields.push_back(field);
            }
            if (fields.size() < 1)
                continue;
            result.push_back(fields);
        }
        return result;
    }

    // utils
    static uint64_t uptime(void)
    {
        uint64_t uptime = 0;
        std::ifstream uptime_file("/proc/uptime");
        if (uptime_file.is_open()) {
            double uptime_seconds;
            uptime_file >> uptime_seconds;
            uptime = static_cast<uint64_t>(uptime_seconds * 1000);
            uptime_file.close();
        }
        return uptime;
    }

    static auto timestamp()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto timestamp = duration_cast<seconds>(now.time_since_epoch());
        return timestamp.count();
    }

private:
    // Poll waitpid(WNOHANG) for up to budget_ms. Returns true if the child was
    // reaped (status filled) or is already gone (ECHILD); false if it is still
    // alive after the budget. budget_ms == 0 does a single non-blocking check.
    static bool reap_bounded(pid_t pid, int& status, int budget_ms)
    {
        for (int i = 0;; ++i) {
            pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) {
                return true;
            }
            if (r < 0) {
                return true; // ECHILD: nothing left to reap
            }
            if (i >= budget_ms) {
                return false;
            }
            usleep(1000); // 1ms granularity
        }
    }

    const std::string _group;

    static inline bool _force_no_auth = false;
    static inline std::string _script;
    static inline std::unordered_map<std::string, std::unique_ptr<rest_api>> _api_map;

    static api_status_t version(request_t req, response_t res)
    {
        res["uptime"] = uptime();
        res["timestamp"] = timestamp();
        response(res, 0, STR_OK, PROJECT_VERSION);
        return API_STATUS_OK;
    }
};
#endif // API_BASE_H
