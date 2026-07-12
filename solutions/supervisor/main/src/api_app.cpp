#include "api_app.h"

#include "config_schema.hpp"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits.h>
#include <sys/stat.h>
#include <thread>

namespace fs = std::filesystem;

api_app::api_app()
    : api_base("appMgr")
{
    // Sync in-memory state with persisted selection. app_restore (init script)
    // starts the persisted app asynchronously at boot, assume RUNNING; the
    // per-request status probe (app_status) reports the live state.
    json state = read_state();
    if (!jstr(state, "active_app").empty()) {
        _state = app_state::RUNNING;
    }

    // Security: every appMgr endpoint requires a valid token.
    REG_API(list);
    REG_API(current);
    REG_API_FULL("switch", switchApp, false); // "switch" is a C++ keyword
    REG_API(stop);
    REG_API(setModel);
    REG_API(getConfig);
    REG_API(setConfig);
    REG_API(getIntegrationDoc);
}

const char* api_app::state_str(app_state s)
{
    switch (s) {
    case app_state::STOPPED:
        return "stopped";
    case app_state::STOPPING:
        return "stopping";
    case app_state::WAIT_RELEASE:
        return "wait_release";
    case app_state::STARTING:
        return "starting";
    case app_state::RUNNING:
        return "running";
    case app_state::ERROR:
        return "error";
    }
    return "unknown";
}

std::string api_app::jstr(const json& j, const std::string& key)
{
    if (j.is_object() && j.contains(key) && j[key].is_string()) {
        return j[key].get<std::string>();
    }
    return "";
}

// app id whitelist: [a-z0-9-], 1..64 chars
bool api_app::valid_app_id(const std::string& id)
{
    if (id.empty() || id.size() > 64) {
        return false;
    }
    for (char c : id) {
        if (!(std::islower((unsigned char)c) || std::isdigit((unsigned char)c) || c == '-')) {
            return false;
        }
    }
    return true;
}

// init_script whitelist: must be /etc/init.d/[SK][0-9]... with a safe charset,
// no subdirectories (so no traversal is possible).
bool api_app::valid_init_script_path(const std::string& path)
{
    const std::string prefix = INIT_DIR_PREFIX;
    if (path.rfind(prefix, 0) != 0) {
        return false;
    }
    std::string base = path.substr(prefix.size());
    if (base.size() < 2 || base.size() > 64) {
        return false;
    }
    if (base[0] != 'S' && base[0] != 'K') {
        return false;
    }
    if (!std::isdigit((unsigned char)base[1])) {
        return false;
    }
    for (char c : base) {
        if (!(std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.')) {
            return false;
        }
    }
    return true;
}

// Filesystem-level checks done right before executing an init script:
// regular file, not a symlink, root-owned, canonical path unchanged by realpath.
bool api_app::check_init_script_fs(const std::string& path, std::string& err)
{
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        err = "init script not found: " + path;
        return false;
    }
    if (S_ISLNK(st.st_mode)) {
        err = "init script is a symlink (rejected): " + path;
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        err = "init script is not a regular file: " + path;
        return false;
    }
    if (st.st_uid != 0) {
        err = "init script is not root-owned (rejected): " + path;
        return false;
    }
    char resolved[PATH_MAX] = { 0 };
    if (realpath(path.c_str(), resolved) == nullptr) {
        err = "realpath() failed for: " + path;
        return false;
    }
    if (std::string(resolved) != path) {
        err = "init script path is not canonical (rejected): " + path;
        return false;
    }
    return true;
}

json api_app::load_manifest_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        return json();
    }
    json m;
    try {
        f >> m;
    } catch (const std::exception& e) {
        LOGW("Bad app manifest %s: %s", path.c_str(), e.what());
        return json();
    }
    if (!m.is_object()) {
        LOGW("Bad app manifest %s: not an object", path.c_str());
        return json();
    }

    std::string id = jstr(m, "id");
    std::string type = jstr(m, "type");
    if (!valid_app_id(id)) {
        LOGW("App manifest %s: invalid id '%s'", path.c_str(), id.c_str());
        return json();
    }
    if (type != "native" && type != "external-firmware") {
        LOGW("App manifest %s: invalid type '%s'", path.c_str(), type.c_str());
        return json();
    }
    if (jstr(m, "name").empty()) {
        LOGW("App manifest %s: missing name", path.c_str());
        return json();
    }
    if (!valid_init_script_path(jstr(m, "init_script"))) {
        LOGW("App manifest %s: init_script rejected by whitelist", path.c_str());
        return json();
    }
    if (!m.contains("models") || !m["models"].is_array()) {
        m["models"] = json::array();
    }
    // Optional fixed multi-model pipeline (display/integration only, not
    // switchable). Normalize to an array so the frontend can rely on it.
    if (m.contains("pipeline") && !m["pipeline"].is_array()) {
        m.erase("pipeline");
    }
    return m;
}

json api_app::load_manifests()
{
    json apps = json::object();
    // Built-in first, user dir second: same id in /userdata overrides built-in.
    for (const char* dir : { BUILTIN_APPS_DIR, USER_APPS_DIR }) {
        std::error_code ec;
        fs::directory_iterator it(dir, ec);
        if (ec) {
            continue;
        }
        for (const auto& entry : it) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const auto& p = entry.path();
            if (p.extension() != ".json" || p.filename() == "state.json") {
                continue;
            }
            // <id>.config.json files are per-app config values, not manifests.
            const std::string fname = p.filename().string();
            const std::string cfg_suffix = ".config.json";
            if (fname.size() > cfg_suffix.size() && fname.compare(fname.size() - cfg_suffix.size(), cfg_suffix.size(), cfg_suffix) == 0) {
                continue;
            }
            json m = load_manifest_file(p.string());
            if (!m.is_object()) {
                continue;
            }
            apps[m["id"].get<std::string>()] = m;
        }
    }
    return apps;
}

json api_app::read_state()
{
    json st = json::object();
    std::ifstream f(STATE_FILE);
    if (f.is_open()) {
        try {
            f >> st;
        } catch (const std::exception& e) {
            LOGW("Bad state file %s: %s", STATE_FILE, e.what());
            st = json::object();
        }
    }
    if (!st.is_object()) {
        st = json::object();
    }
    if (!st.contains("models") || !st["models"].is_object()) {
        st["models"] = json::object();
    }
    return st;
}

// Atomic persist: write tmp -> fsync -> rename over state.json.
bool api_app::write_state(json& state)
{
    state["updated_at"] = timestamp();

    std::error_code ec;
    fs::create_directories(USER_APPS_DIR, ec);

    std::string tmp = std::string(USER_APPS_DIR) + "/.state.json.tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
    std::string data = state.dump(2);
    ssize_t n = ::write(fd, data.data(), data.size());
    ::fsync(fd);
    ::close(fd);
    if (n != (ssize_t)data.size()) {
        LOGE("short write to %s", tmp.c_str());
        ::unlink(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), STATE_FILE) != 0) {
        LOGE("rename(%s -> %s) failed: %s", tmp.c_str(), STATE_FILE, strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

// Atomic write of /userdata/local/apps/<app_id>.model (one line: model path).
// Same tmp + fsync + rename pattern as write_state().
bool api_app::write_model_override(const std::string& app_id, const std::string& model_path)
{
    std::error_code ec;
    fs::create_directories(USER_APPS_DIR, ec);

    std::string dst = std::string(USER_APPS_DIR) + "/" + app_id + ".model";
    std::string tmp = std::string(USER_APPS_DIR) + "/." + app_id + ".model.tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
    std::string data = model_path + "\n";
    ssize_t n = ::write(fd, data.data(), data.size());
    ::fsync(fd);
    ::close(fd);
    if (n != (ssize_t)data.size()) {
        LOGE("short write to %s", tmp.c_str());
        ::unlink(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), dst.c_str()) != 0) {
        LOGE("rename(%s -> %s) failed: %s", tmp.c_str(), dst.c_str(), strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

// Current config values of an app: /userdata/local/apps/<id>.config.json.
// Missing or unparsable file -> empty object (never an error).
json api_app::read_config_file(const std::string& app_id)
{
    json values = json::object();
    std::ifstream f(std::string(USER_APPS_DIR) + "/" + app_id + ".config.json");
    if (f.is_open()) {
        try {
            f >> values;
        } catch (const std::exception& e) {
            LOGW("Bad config file for app %s: %s", app_id.c_str(), e.what());
            values = json::object();
        }
    }
    if (!values.is_object()) {
        values = json::object();
    }
    return values;
}

// Atomic write of /userdata/local/apps/<id>.config.json.
// Same tmp + fsync + rename pattern as write_state().
bool api_app::write_config_file(const std::string& app_id, const json& values)
{
    std::error_code ec;
    fs::create_directories(USER_APPS_DIR, ec);

    std::string dst = std::string(USER_APPS_DIR) + "/" + app_id + ".config.json";
    std::string tmp = std::string(USER_APPS_DIR) + "/." + app_id + ".config.json.tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
    std::string data = values.dump(2) + "\n";
    ssize_t n = ::write(fd, data.data(), data.size());
    ::fsync(fd);
    ::close(fd);
    if (n != (ssize_t)data.size()) {
        LOGE("short write to %s", tmp.c_str());
        ::unlink(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), dst.c_str()) != 0) {
        LOGE("rename(%s -> %s) failed: %s", tmp.c_str(), dst.c_str(), strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

// --- switch/stop building blocks (caller must hold _op_mutex) ---

bool api_app::stop_current_locked(const std::string& script_path, response_t res)
{
    std::string err;
    if (!valid_init_script_path(script_path) || !check_init_script_fs(script_path, err)) {
        // Script vanished or is invalid: treat as already stopped, just log.
        LOGW("stop: skipping invalid current script '%s': %s", script_path.c_str(), err.c_str());
        return true;
    }
    _state = app_state::STOPPING;
    std::string r = script("app_stop", script_path); // main.sh: 10s timeout, TERM only
    if (r != STR_OK) {
        _state = app_state::ERROR;
        _last_error = "stop failed (" + r + "): " + script_path;
        LOGE("%s", _last_error.c_str());
        response(res, -1, _last_error, { { "state", state_str(_state) } });
        return false;
    }
    // VPSS/camera release grace period (kernel driver is fragile, do not rush).
    _state = app_state::WAIT_RELEASE;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return true;
}

bool api_app::start_target_locked(const std::string& script_path, response_t res)
{
    std::string err;
    if (!check_init_script_fs(script_path, err)) {
        _state = app_state::ERROR;
        _last_error = err;
        response(res, -1, _last_error, { { "state", state_str(_state) } });
        return false;
    }
    _state = app_state::STARTING;
    std::string r = script("app_start", script_path); // main.sh: 15s timeout, TERM only
    if (r != STR_OK) {
        _state = app_state::ERROR;
        _last_error = "start failed (" + r + "): " + script_path;
        LOGE("%s", _last_error.c_str());
        // Deliberately no retry loop here: VPSS failures need a human decision.
        response(res, -1, _last_error, { { "state", state_str(_state) } });
        return false;
    }
    return true;
}

// --- REST handlers ---

api_status_t api_app::list(request_t req, response_t res)
{
    json manifests = load_manifests();
    json state = read_state();
    std::string active = jstr(state, "active_app");

    json arr = json::array();
    for (auto& [id, m] : manifests.items()) {
        json item = m;
        bool is_active = (id == active);
        item["active"] = is_active;
        item["status"] = is_active ? state_str(_state) : "stopped";
        if (state["models"].contains(id) && state["models"][id].is_string()) {
            item["current_model"] = state["models"][id];
        } else {
            item["current_model"] = m.value("default_model", "");
        }
        arr.push_back(item);
    }

    json data = json::object();
    data["apps"] = arr;
    data["current"] = active.empty() ? json(nullptr) : json(active);
    data["state"] = state_str(_state);
    response(res, 0, STR_OK, data);
    return API_STATUS_OK;
}

api_status_t api_app::current(request_t req, response_t res)
{
    json state = read_state();
    std::string active = jstr(state, "active_app");

    json data = json::object();
    data["state"] = state_str(_state);
    data["lastError"] = _last_error;

    if (active.empty()) {
        data["app"] = nullptr;
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    json manifests = load_manifests();
    if (!manifests.contains(active)) {
        data["app"] = nullptr;
        data["orphan"] = active; // state points at a manifest that no longer exists
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    json app = manifests[active];
    // Substitute {host} in rtsp_url with the host the client used to reach us.
    std::string rtsp = jstr(app, "rtsp_url");
    const std::string ph = "{host}";
    size_t pos;
    while ((pos = rtsp.find(ph)) != std::string::npos) {
        rtsp.replace(pos, ph.size(), get_host(req));
    }
    app["rtsp_url"] = rtsp;
    if (state["models"].contains(active) && state["models"][active].is_string()) {
        app["current_model"] = state["models"][active];
    } else {
        app["current_model"] = app.value("default_model", "");
    }

    // Live status probe via init script (5s timeout in main.sh).
    json probe = parse_result(script("app_status", jstr(app, "init_script")));
    data["probe"] = probe.value("status", "unknown");
    data["app"] = app;
    response(res, 0, STR_OK, data);
    return API_STATUS_OK;
}

api_status_t api_app::switchApp(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    std::string app_id = body.value("app_id", "");
    if (!valid_app_id(app_id)) {
        response(res, -1, "Invalid app_id");
        return API_STATUS_OK;
    }

    if (!_op_mutex.try_lock()) {
        response(res, -2, "busy: another app operation is in progress");
        return API_STATUS_OK;
    }
    std::unique_lock<std::timed_mutex> lk(_op_mutex, std::adopt_lock);

    json manifests = load_manifests();
    if (!manifests.contains(app_id)) {
        response(res, -1, "Unknown app: " + app_id);
        return API_STATUS_OK;
    }
    std::string target_script = jstr(manifests[app_id], "init_script");

    json state = read_state();
    std::string active = jstr(state, "active_app");
    std::string active_script = jstr(state, "active_script");

    if (active == app_id && _state == app_state::RUNNING) {
        response(res, 0, STR_OK, { { "current", app_id }, { "state", state_str(_state) }, { "note", "already running" } });
        return API_STATUS_OK;
    }

    // stop current -> sleep 2 (VPSS release)
    if (!active.empty() && !active_script.empty()) {
        if (!stop_current_locked(active_script, res)) {
            return API_STATUS_OK;
        }
    }

    // start target
    if (!start_target_locked(target_script, res)) {
        // Camera is now free but nothing is running; persist that fact.
        state["active_app"] = nullptr;
        state["active_script"] = nullptr;
        write_state(state);
        return API_STATUS_OK;
    }

    // status probe (5s timeout in main.sh), informational only
    json probe = parse_result(script("app_status", target_script));

    state["active_app"] = app_id;
    state["active_script"] = target_script;
    if (!write_state(state)) {
        LOGE("failed to persist app state (app still switched)");
    }
    _state = app_state::RUNNING;
    _last_error.clear();

    response(res, 0, STR_OK,
        { { "current", app_id },
            { "state", state_str(_state) },
            { "probe", probe.value("status", "unknown") } });
    return API_STATUS_OK;
}

api_status_t api_app::stop(request_t req, response_t res)
{
    if (!_op_mutex.try_lock()) {
        response(res, -2, "busy: another app operation is in progress");
        return API_STATUS_OK;
    }
    std::unique_lock<std::timed_mutex> lk(_op_mutex, std::adopt_lock);

    json state = read_state();
    std::string active = jstr(state, "active_app");
    std::string active_script = jstr(state, "active_script");

    if (active.empty() || active_script.empty()) {
        _state = app_state::STOPPED;
        response(res, 0, STR_OK, { { "state", state_str(_state) }, { "note", "no active app" } });
        return API_STATUS_OK;
    }

    if (!stop_current_locked(active_script, res)) {
        return API_STATUS_OK;
    }

    _state = app_state::STOPPED;
    _last_error.clear();
    state["active_app"] = nullptr;
    state["active_script"] = nullptr;
    write_state(state);

    response(res, 0, STR_OK, { { "state", state_str(_state) } });
    return API_STATUS_OK;
}

api_status_t api_app::setModel(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    std::string app_id = body.value("app_id", "");
    std::string model = body.value("model", "");
    if (!valid_app_id(app_id) || model.empty() || model.size() > 128) {
        response(res, -1, "Invalid app_id or model");
        return API_STATUS_OK;
    }

    if (!_op_mutex.try_lock()) {
        response(res, -2, "busy: another app operation is in progress");
        return API_STATUS_OK;
    }
    std::unique_lock<std::timed_mutex> lk(_op_mutex, std::adopt_lock);

    json manifests = load_manifests();
    if (!manifests.contains(app_id)) {
        response(res, -1, "Unknown app: " + app_id);
        return API_STATUS_OK;
    }
    const json& app = manifests[app_id];

    // Apps whose models[] is empty are fixed single/multi-model pipelines:
    // there is nothing to switch (pipeline[] components are not switchable).
    if (app["models"].empty()) {
        response(res, -1, "app has no switchable models");
        return API_STATUS_OK;
    }

    // model must be declared in the manifest's models[] (match by name)
    std::string model_path;
    bool found = false;
    for (const auto& m : app["models"]) {
        if (m.is_object() && jstr(m, "name") == model) {
            model_path = jstr(m, "path");
            found = true;
            break;
        }
    }
    if (!found) {
        response(res, -1, "Model '" + model + "' is not declared by app '" + app_id + "'");
        return API_STATUS_OK;
    }
    if (model_path.empty()) {
        response(res, -1, "Model '" + model + "' has no path in the manifest of app '" + app_id + "'");
        return API_STATUS_OK;
    }

    json state = read_state();
    state["models"][app_id] = model;
    if (!write_state(state)) {
        response(res, -1, "Failed to persist model selection");
        return API_STATUS_OK;
    }

    // Model override file consumed by the app's init script:
    // /userdata/local/apps/<app_id>.model = one line, absolute model path.
    // Written atomically (tmp + fsync + rename), same pattern as state.json.
    if (!write_model_override(app_id, model_path)) {
        response(res, -1, "Failed to write model override file");
        return API_STATUS_OK;
    }

    // MVP semantics: persist + restart the app if it is the active one.
    // (In-app hot model switching is Phase 2.)
    bool restarted = false;
    if (jstr(state, "active_app") == app_id) {
        std::string script_path = jstr(app, "init_script");
        if (!stop_current_locked(script_path, res)) {
            return API_STATUS_OK;
        }
        if (!start_target_locked(script_path, res)) {
            json st2 = read_state();
            st2["active_app"] = nullptr;
            st2["active_script"] = nullptr;
            write_state(st2);
            return API_STATUS_OK;
        }
        _state = app_state::RUNNING;
        restarted = true;
    }

    response(res, 0, STR_OK, { { "app_id", app_id }, { "model", model }, { "restarted", restarted } });
    return API_STATUS_OK;
}

// GET/POST /api/appMgr/getConfig?app_id=<id>
// Returns {schema, values, defaults}:
//   schema   : the manifest's config_schema (or null if the app has none)
//   values   : current /userdata/local/apps/<id>.config.json content ({} if unset)
//   defaults : per-key defaults declared in the schema
api_status_t api_app::getConfig(request_t req, response_t res)
{
    std::string app_id = get_param(req, "app_id");
    if (app_id.empty()) {
        app_id = parse_body(req).value("app_id", "");
    }
    if (!valid_app_id(app_id)) {
        response(res, -1, "Invalid app_id");
        return API_STATUS_OK;
    }

    json manifests = load_manifests();
    if (!manifests.contains(app_id)) {
        response(res, -1, "Unknown app: " + app_id);
        return API_STATUS_OK;
    }
    const json& app = manifests[app_id];

    json data = json::object();
    if (app.contains("config_schema") && app["config_schema"].is_object()) {
        data["schema"] = app["config_schema"];
        data["defaults"] = config_schema::defaults_from_schema(app["config_schema"]);
    } else {
        data["schema"] = nullptr;
        data["defaults"] = json::object();
    }
    data["app_id"] = app_id;
    data["values"] = read_config_file(app_id);
    response(res, 0, STR_OK, data);
    return API_STATUS_OK;
}

// POST /api/appMgr/setConfig  body: {app_id, values:{...}}
// Every key in values is validated against the manifest's config_schema
// (unknown keys rejected, per-type range/shape checks), then the whole
// object is written atomically to /userdata/local/apps/<id>.config.json.
// If the app is the active one it is restarted so the new config applies
// (same stop/start building blocks and busy semantics as setModel).
api_status_t api_app::setConfig(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    std::string app_id = body.value("app_id", "");
    if (!valid_app_id(app_id)) {
        response(res, -1, "Invalid app_id");
        return API_STATUS_OK;
    }
    if (!body.contains("values") || !body["values"].is_object()) {
        response(res, -1, "Missing values object");
        return API_STATUS_OK;
    }
    const json& values = body["values"];

    if (!_op_mutex.try_lock()) {
        response(res, -2, "busy: another app operation is in progress");
        return API_STATUS_OK;
    }
    std::unique_lock<std::timed_mutex> lk(_op_mutex, std::adopt_lock);

    json manifests = load_manifests();
    if (!manifests.contains(app_id)) {
        response(res, -1, "Unknown app: " + app_id);
        return API_STATUS_OK;
    }
    const json& app = manifests[app_id];

    if (!app.contains("config_schema") || !app["config_schema"].is_object()) {
        response(res, -1, "app '" + app_id + "' has no config_schema");
        return API_STATUS_OK;
    }

    std::string err;
    if (!config_schema::validate_values(app["config_schema"], values, err)) {
        response(res, -1, "Invalid config: " + err);
        return API_STATUS_OK;
    }

    if (!write_config_file(app_id, values)) {
        response(res, -1, "Failed to persist config");
        return API_STATUS_OK;
    }

    // Restart the app if it is the active one so the config takes effect.
    bool restarted = false;
    json state = read_state();
    if (jstr(state, "active_app") == app_id) {
        std::string script_path = jstr(app, "init_script");
        if (!stop_current_locked(script_path, res)) {
            return API_STATUS_OK;
        }
        if (!start_target_locked(script_path, res)) {
            json st2 = read_state();
            st2["active_app"] = nullptr;
            st2["active_script"] = nullptr;
            write_state(st2);
            return API_STATUS_OK;
        }
        _state = app_state::RUNNING;
        restarted = true;
    }

    response(res, 0, STR_OK, { { "app_id", app_id }, { "values", values }, { "restarted", restarted } });
    return API_STATUS_OK;
}

// GET/POST /api/appMgr/getIntegrationDoc?app_id=<id>[&lang=zh|en]
// Returns the integration/output-format markdown for an application.
// Candidates, first match wins (user dir overrides built-in; zh variants
// are preferred when lang=zh, with backend fallback to English):
//   lang=zh: /userdata/local/apps/<id>.zh.md, /usr/share/supervisor/apps/<id>.zh.md,
//            /userdata/local/apps/<id>.md,    /usr/share/supervisor/apps/<id>.md
//   default: the two <id>.md paths only
// The response carries "lang_served" ("zh" only when a .zh.md was served).
// Missing file is NOT an error: code 0 + data.content = "" (frontend hides
// the section). app_id goes through the [a-z0-9-] whitelist, so the joined
// path cannot escape the apps directories (no '/', '.' or '..' possible).
api_status_t api_app::getIntegrationDoc(request_t req, response_t res)
{
    static constexpr size_t MAX_DOC_SIZE = 32 * 1024; // spec: md files < 32KB

    std::string app_id = get_param(req, "app_id");
    std::string lang = get_param(req, "lang");
    if (app_id.empty()) {
        auto&& body = parse_body(req);
        app_id = body.value("app_id", "");
        if (lang.empty()) {
            lang = body.value("lang", "");
        }
    }
    if (!valid_app_id(app_id)) {
        response(res, -1, "Invalid app_id");
        return API_STATUS_OK;
    }
    // Anything that is not exactly "zh" is served English (backend fallback).
    bool want_zh = (lang == "zh");

    json data = json::object();
    data["app_id"] = app_id;
    data["content"] = "";
    data["source"] = "";
    data["lang_served"] = "en";

    // Candidate list: zh variants first when requested, user dir over builtin.
    struct doc_candidate {
        std::string path;
        const char* source;
        const char* lang;
    };
    std::vector<doc_candidate> candidates;
    if (want_zh) {
        candidates.push_back({ std::string(USER_APPS_DIR) + "/" + app_id + ".zh.md", "user", "zh" });
        candidates.push_back({ std::string(BUILTIN_APPS_DIR) + "/" + app_id + ".zh.md", "builtin", "zh" });
    }
    candidates.push_back({ std::string(USER_APPS_DIR) + "/" + app_id + ".md", "user", "en" });
    candidates.push_back({ std::string(BUILTIN_APPS_DIR) + "/" + app_id + ".md", "builtin", "en" });

    for (const auto& c : candidates) {
        // Defense in depth: the whitelist already prevents traversal, but
        // verify the canonical path is still inside an apps directory.
        char resolved[PATH_MAX] = { 0 };
        if (realpath(c.path.c_str(), resolved) == nullptr) {
            continue; // does not exist (or unreadable) -> try next candidate
        }
        std::string resolved_s(resolved);
        if (resolved_s.rfind(std::string(USER_APPS_DIR) + "/", 0) != 0
            && resolved_s.rfind(std::string(BUILTIN_APPS_DIR) + "/", 0) != 0) {
            LOGW("integration doc path escapes apps dir (rejected): %s", resolved);
            continue;
        }

        std::ifstream f(resolved, std::ios::binary);
        if (!f.is_open()) {
            continue;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();
        if (content.size() > MAX_DOC_SIZE) {
            content.resize(MAX_DOC_SIZE);
        }
        data["content"] = content;
        data["source"] = c.source;
        data["lang_served"] = c.lang;
        break;
    }

    response(res, 0, STR_OK, data);
    return API_STATUS_OK;
}
