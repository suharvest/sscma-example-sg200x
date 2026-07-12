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
    static bool acquire_op_lock_or_busy(response_t res, std::unique_lock<std::timed_mutex>& lk);
    static bool stop_current_locked(const std::string& script_path, response_t res);
    static bool start_target_locked(const std::string& script_path, response_t res);
    static bool restart_if_active_locked(const json& app, const std::string& app_id, response_t res, bool& restarted);

    static constexpr const char* BUILTIN_APPS_DIR = "/usr/share/supervisor/apps";
    static constexpr const char* USER_APPS_DIR = "/userdata/local/apps";
    static constexpr const char* STATE_FILE = "/userdata/local/apps/state.json";
    static constexpr const char* INIT_DIR_PREFIX = "/etc/init.d/";

    // Concurrency guard for switch/stop/setModel: try-lock only, no queueing.
    static inline std::timed_mutex _op_mutex;
    static inline std::atomic<app_state> _state { app_state::STOPPED };
    static inline std::string _last_error;
};

#endif // API_APP_H
