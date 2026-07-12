#ifndef API_APP_H
#define API_APP_H

#include <atomic>
#include <mutex>
#include <string>

#include "api_base.h"

// appMgr: application gallery manager (gallery mode).
//
// Manages "applications" (face-analysis, yolo-detector, external firmware...)
// described by JSON manifests:
//   - built-in:  /usr/share/supervisor/apps/<id>.json
//   - user:      /userdata/local/apps/<id>.json  (same id overrides built-in)
//
// Exactly one application may own the camera at a time. Switching is:
//   stop current (10s, TERM only) -> sleep 2 (VPSS release) -> start target (15s)
// The active selection is persisted to /userdata/local/apps/state.json
// (atomic tmp+fsync+rename) and restored at boot via `main.sh app_restore`.
class api_app : public api_base {
public:
    api_app();
    ~api_app() = default;

private:
    enum class app_state {
        STOPPED,
        STOPPING,
        WAIT_RELEASE,
        STARTING,
        RUNNING,
        ERROR,
    };

    // REST handlers (all token-protected, see constructor)
    static api_status_t list(request_t req, response_t res);
    static api_status_t current(request_t req, response_t res);
    static api_status_t switchApp(request_t req, response_t res); // uri: "switch"
    static api_status_t stop(request_t req, response_t res);
    static api_status_t setModel(request_t req, response_t res);
    static api_status_t getConfig(request_t req, response_t res);
    static api_status_t setConfig(request_t req, response_t res);
    static api_status_t getIntegrationDoc(request_t req, response_t res);
    static api_status_t installApp(request_t req, response_t res);

    // helpers
    static bool valid_app_id(const std::string& id);
    // Hardware capability gating (P5-A): manifest requires[] vs the cached
    // api_device::capabilities() set.
    static bool valid_capability_key(const std::string& key);
    static bool capability_present(const json& caps, const std::string& key);
    static json missing_capabilities(const json& manifest);
    static bool valid_init_script_path(const std::string& path);
    static bool check_init_script_fs(const std::string& path, std::string& err);
    static std::string jstr(const json& j, const std::string& key);
    static json load_manifest_file(const std::string& path);
    static json load_manifests(); // object: id -> manifest
    static json read_state();
    static bool write_file_atomic(const std::string& dst, const std::string& tmp, const std::string& data);
    static bool write_state(json& state);
    static bool write_model_override(const std::string& app_id, const std::string& model_path);
    static json read_config_file(const std::string& app_id);
    static bool write_config_file(const std::string& app_id, const json& values);
    static const char* state_str(app_state s);

    // Concurrency guard shared by every mutating appMgr handler (switch / stop /
    // setModel / setConfig / installApp). #14: an atomic busy gate instead of a
    // std::mutex, because an async operation acquires it on the poll thread
    // (before enqueue) and releases it on the poll thread (in the wakeup
    // finalize) — a std::mutex cannot be unlocked by a different lock/scope.
    // The gate covers the WHOLE job lifecycle (enqueue -> worker -> commit) so
    // the busy(-2) contract still holds while a long op runs on a worker.
    // atomic_flag is the one type guaranteed lock-free everywhere (no libatomic
    // needed on this riscv musl toolchain); test_and_set returns the previous
    // value, so acquire succeeds only when the flag was clear.
    static bool op_try_acquire() // poll thread
    {
        return !_op_busy.test_and_set(std::memory_order_acquire);
    }
    static void op_release() { _op_busy.clear(std::memory_order_release); } // poll thread
    // RAII release for handlers that stay fully synchronous.
    struct op_guard {
        bool owns = false;
        ~op_guard()
        {
            if (owns) {
                op_release();
            }
        }
    };
    // Try to acquire for a synchronous handler; on contention writes the
    // busy(-2) response and returns false.
    static bool acquire_op_or_busy(response_t res, op_guard& g)
    {
        if (!op_try_acquire()) {
            response(res, -2, "busy: another app operation is in progress");
            return false;
        }
        g.owns = true;
        return true;
    }

    // #14: worker-thread camera hand-over outcome. The worker fills only these
    // shell results; the poll-thread commit maps them to _state / _last_error /
    // state.json and the HTTP reply.
    struct app_op {
        bool stop_ok = true;   // no stop needed OR app_stop succeeded
        std::string stop_err;  // set when stop_ok == false
        bool start_ok = false;
        std::string start_err; // set when start_ok == false
        std::string probe = "unknown";
    };
    // Pure worker-thread shell steps (no process state, no HTTP response).
    static void sh_stop(const std::string& script_path, app_op& o);
    static void sh_start(const std::string& script_path, app_op& o);

    // #14: shared by setModel/setConfig. If app_id is NOT the active app,
    // replies success synchronously (restarted=false, g releases the op gate).
    // If it IS active, restarts it (stop -> start) on a worker and replies from
    // the wakeup (restarted=true); ownership of the op gate transfers to the
    // async finalize. success_data is merged into the success reply.
    static api_status_t restart_after_change(const json& app, const std::string& app_id,
        json success_data, op_guard& g, response_t res);

    static constexpr const char* BUILTIN_APPS_DIR = "/usr/share/supervisor/apps";
    static constexpr const char* USER_APPS_DIR = "/userdata/local/apps";
    static constexpr const char* STATE_FILE = "/userdata/local/apps/state.json";
    static constexpr const char* INIT_DIR_PREFIX = "/etc/init.d/";

    // #14: atomic busy gate (see op_try_acquire above), replaces the old
    // std::timed_mutex so the gate can span an async job's whole lifecycle.
    static inline std::atomic_flag _op_busy = ATOMIC_FLAG_INIT;
    static inline std::atomic<app_state> _state { app_state::STOPPED };
    static inline std::string _last_error;
};

#endif // API_APP_H
