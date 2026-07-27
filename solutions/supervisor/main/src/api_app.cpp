#include "api_app.h"

#include "api_device.h"
#include "blur_config.h"
#include "camera_config.h"
#include "config_schema.hpp"
#include "ha_config.h"
#include "onvif_config.h"

#include <arpa/inet.h>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits.h>
#include <mosquitto.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>

namespace fs = std::filesystem;

namespace {

// Node-RED mode gate: while Node-RED owns the camera the C++ app stack is
// parked, so app lifecycle operations (switch / setModel / setConfig /
// installApp) must not stop/start apps. They are refused with code -3 (the
// frontend recognizes it); config-only writes (setHaConfig) skip the restart
// instead. Read from the persisted mode file — same source getRunMode uses.
constexpr int CODE_NODERED_MODE = -3;
constexpr const char* MSG_NODERED_MODE = "Not available in Node-RED mode";

bool in_nodered_mode()
{
    return api_device::read_run_mode_file() == "nodered";
}

// Wedge-defense (B): true if a VPSS group with GrpID 0 is still present in the
// driver. Parses /proc/cvitek/vpss and looks for a data row (leading "# N")
// under the "VPSS GRP ATTR" table whose GrpID is exactly 0. When no app owns
// the camera the GRP ATTR table has no data rows, so this returns false.
bool vpss_grp0_present()
{
    std::ifstream f("/proc/cvitek/vpss");
    if (!f.is_open()) {
        return false; // driver not exposing the file -> nothing to wait on
    }
    std::string line;
    bool in_grp_attr = false;
    while (std::getline(f, line)) {
        if (line.find("VPSS GRP ATTR") != std::string::npos) {
            in_grp_attr = true;
            continue;
        }
        if (!in_grp_attr) {
            continue;
        }
        if (line.find("-----") != std::string::npos) {
            break; // reached the next section without finding grp 0
        }
        size_t h = line.find('#');
        if (h == std::string::npos) {
            continue; // column-header row ("GrpID ...") has no '#'
        }
        size_t p = h + 1;
        while (p < line.size() && line[p] == ' ') {
            ++p;
        }
        // GrpID token is exactly "0" (not 10, 20, ...)
        if (p < line.size() && line[p] == '0' &&
            (p + 1 >= line.size() || !std::isdigit(static_cast<unsigned char>(line[p + 1])))) {
            return true;
        }
    }
    return false;
}

// Poll up to max_ms for Grp(0) to be released after an app stop. Returns true
// if released in time, false on timeout (caller logs and proceeds). Cheap
// fallback: if the proc file is unreadable, vpss_grp0_present() returns false
// so this returns immediately.
bool wait_vpss_grp0_released(int max_ms)
{
    const int step_ms = 200;
    int waited = 0;
    while (waited < max_ms) {
        if (!vpss_grp0_present()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        waited += step_ms;
    }
    return !vpss_grp0_present();
}

} // namespace

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
    REG_API(installApp);
    REG_API(getHaConfig);
    REG_API(setHaConfig);
    REG_API(testHaConnection);
    REG_API(getOnvifConfig);
    REG_API(setOnvifConfig);
    REG_API(getBlurConfig);
    REG_API(setBlurConfig);
    REG_API(getCameraConfig);
    REG_API(setCameraConfig);
    REG_API(getFocusValue);
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
    // Filename must be "<id>.json". This lets lookups resolve a single app by
    // path (USER_APPS_DIR/<id>.json → builtin) instead of scanning every
    // manifest, and keeps the per-app sidecar files (<id>.config.json /
    // <id>.model / <id>.md) addressable by the same id. A vendor that ships a
    // mismatched filename is rejected here rather than silently shadowing or
    // orphaning another app's config.
    if (fs::path(path).stem().string() != id) {
        LOGW("App manifest %s: filename does not match id '%s' (expected %s.json)",
            path.c_str(), id.c_str(), id.c_str());
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
    // Does this app apply the device-wide privacy mask? The masking setting
    // itself is device-wide, but the shortcut for it only belongs on apps that
    // honour it -- a switch offered next to a stream it cannot change is worse
    // than no switch at all. Normalized to a bool, defaulting to false: an app
    // that does not say so does not mask.
    m["privacy_blur"] = m.contains("privacy_blur") && m["privacy_blur"].is_boolean()
                            ? m["privacy_blur"].get<bool>()
                            : false;
    // Optional hardware dependencies (P5-A): requires = ["gimbal", ...].
    // Keys are validated against the capability whitelist; unknown keys are
    // dropped with a warning (forward compatibility: an app written for a
    // future firmware must not be bricked by one unknown key). Normalized
    // to an array of unique valid strings.
    {
        json requires_norm = json::array();
        if (m.contains("requires")) {
            if (m["requires"].is_array()) {
                for (const auto& r : m["requires"]) {
                    if (!r.is_string()) {
                        LOGW("App manifest %s: non-string requires entry ignored", path.c_str());
                        continue;
                    }
                    const std::string key = r.get<std::string>();
                    if (!valid_capability_key(key)) {
                        LOGW("App manifest %s: unknown capability '%s' in requires[] ignored", path.c_str(), key.c_str());
                        continue;
                    }
                    if (std::find(requires_norm.begin(), requires_norm.end(), json(key)) == requires_norm.end()) {
                        requires_norm.push_back(key);
                    }
                }
            } else {
                LOGW("App manifest %s: requires is not an array, ignored", path.c_str());
            }
        }
        m["requires"] = requires_norm;
    }
    return m;
}

// --- hardware capability gating (P5-A) --------------------------------------

// Whitelisted requires[] keys. Must stay in sync with the JSON shape of
// api_device::probe_capabilities().
bool api_app::valid_capability_key(const std::string& key)
{
    static const char* whitelist[] = { "gimbal", "hdr", "halow", "can", "sd", "battery", "audio" };
    for (const char* k : whitelist) {
        if (key == k) {
            return true;
        }
    }
    return false;
}

// Is one capability key present on this device? (caps = cached
// api_device::capabilities() JSON.)
bool api_app::capability_present(const json& caps, const std::string& key)
{
    if (!caps.is_object() || !caps.contains(key)) {
        return false;
    }
    const json& v = caps[key];
    if (v.is_boolean()) {
        return v.get<bool>(); // battery / sd / halow / can
    }
    if (v.is_object()) {
        if (key == "audio") { // mic OR speaker: any audio path counts
            return v.value("mic", false) || v.value("speaker", false);
        }
        return v.value("present", false); // gimbal / hdr
    }
    return false;
}

// Capabilities from the manifest's (already normalized) requires[] that the
// device does not have. Empty array = app is hardware-supported.
json api_app::missing_capabilities(const json& manifest)
{
    json missing = json::array();
    if (!manifest.contains("requires") || !manifest["requires"].is_array()) {
        return missing;
    }
    const json& caps = api_device::capabilities();
    for (const auto& r : manifest["requires"]) {
        if (r.is_string() && !capability_present(caps, r.get<std::string>())) {
            missing.push_back(r);
        }
    }
    return missing;
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

// Atomic write into USER_APPS_DIR: write tmp -> fsync -> rename over dst.
// On any failure the tmp file is removed, the error logged, false returned.
bool api_app::write_file_atomic(const std::string& dst, const std::string& tmp, const std::string& data)
{
    std::error_code ec;
    fs::create_directories(USER_APPS_DIR, ec);

    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
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

// Atomic persist of state.json.
bool api_app::write_state(json& state)
{
    state["updated_at"] = timestamp();
    return write_file_atomic(STATE_FILE, std::string(USER_APPS_DIR) + "/.state.json.tmp", state.dump(2));
}

// Atomic write of /userdata/local/apps/<app_id>.model (one line: model path).
bool api_app::write_model_override(const std::string& app_id, const std::string& model_path)
{
    return write_file_atomic(std::string(USER_APPS_DIR) + "/" + app_id + ".model",
        std::string(USER_APPS_DIR) + "/." + app_id + ".model.tmp",
        model_path + "\n");
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
bool api_app::write_config_file(const std::string& app_id, const json& values)
{
    return write_file_atomic(std::string(USER_APPS_DIR) + "/" + app_id + ".config.json",
        std::string(USER_APPS_DIR) + "/." + app_id + ".config.json.tmp",
        values.dump(2) + "\n");
}

// --- switch/stop building blocks ---
//
// #14: the camera hand-over (app_stop -> sleep 2 -> app_start -> app_status) is
// the blocking part and now runs on a worker. The worker fills an app_op with
// nothing but shell outcomes; ALL process-internal state (_state / _last_error)
// and state.json commits stay on the poll thread (in the async commit). These
// two helpers are the pure worker-side shell steps (no state, no response).

// worker thread: stop script_path if it is a live, valid init script (invalid
// or vanished -> treat as already stopped). On success sleeps 2s for VPSS
// release. Returns via app_op fields.
void api_app::sh_stop(const std::string& script_path, app_op& o)
{
    std::string err;
    if (!valid_init_script_path(script_path) || !check_init_script_fs(script_path, err)) {
        LOGW("stop: skipping invalid script '%s': %s", script_path.c_str(), err.c_str());
        o.stop_ok = true;
        return;
    }
    std::string r = script("app_stop", script_path); // main.sh: 10s timeout, TERM only
    if (r != STR_OK) {
        o.stop_ok = false;
        o.stop_err = "stop failed (" + r + "): " + script_path;
        LOGE("%s", o.stop_err.c_str());
        return;
    }
    // Wedge-defense (B): instead of a blind 2s sleep, actively confirm the VPSS
    // group the old app owned is released before we let the next app start.
    // Starting while Grp(0) is still alive is exactly what produces
    // "Grp(0) is occupied" / "get chn frame fail". Poll up to 5s; if it still
    // has not released, log loudly but proceed (the driver may be wedged — the
    // next start will fail cleanly rather than us hanging forever).
    if (wait_vpss_grp0_released(5000)) {
        LOGD("VPSS Grp(0) released after stopping %s", script_path.c_str());
    } else {
        LOGW("VPSS Grp(0) still present 5s after stopping %s - starting next app anyway", script_path.c_str());
    }
    // Small settle margin for the VI/SYS side after the group is gone.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

// worker thread: start script_path (already fs-validated on the poll thread).
// Runs app_start then the informational app_status probe. Returns via app_op.
void api_app::sh_start(const std::string& script_path, app_op& o)
{
    std::string r = script("app_start", script_path); // main.sh: 15s timeout, TERM only
    if (r != STR_OK) {
        o.start_ok = false;
        o.start_err = "start failed (" + r + "): " + script_path;
        LOGE("%s", o.start_err.c_str());
        return;
    }
    o.start_ok = true;
    o.probe = parse_result(script("app_status", script_path)).value("status", "unknown");
}

// Restart app_id if it is the active app so a persisted change (model/config)
// takes effect. See header. success_data gains "restarted".
api_status_t api_app::restart_after_change(const json& app, const std::string& app_id,
    json success_data, op_guard& g, response_t res)
{
    if (jstr(read_state(), "active_app") != app_id) {
        success_data["restarted"] = false;
        response(res, 0, STR_OK, success_data);
        return API_STATUS_OK; // g releases the op gate on scope exit
    }

    std::string script_path = jstr(app, "init_script");
    std::string err;
    if (!check_init_script_fs(script_path, err)) {
        _state = app_state::ERROR;
        _last_error = err;
        response(res, -1, _last_error, { { "state", state_str(_state) } });
        return API_STATUS_OK;
    }

    // _state is committed only in the wakeup (see switchApp note).
    auto op = std::make_shared<app_op>();
    g.owns = false; // op gate ownership transfers to the async finalize
    return submit_async(
        [op, script_path]() {
            sh_stop(script_path, *op);
            if (op->stop_ok) {
                sh_start(script_path, *op);
            }
        },
        [op, success_data](json& res) -> api_status_t {
            if (!op->stop_ok) {
                _state = app_state::ERROR;
                _last_error = op->stop_err;
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            if (!op->start_ok) {
                _state = app_state::ERROR;
                _last_error = op->start_err;
                json st2 = read_state();
                st2["active_app"] = nullptr;
                st2["active_script"] = nullptr;
                write_state(st2);
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            _state = app_state::RUNNING;
            _last_error.clear();
            json d = success_data;
            d["restarted"] = true;
            response(res, 0, STR_OK, d);
            return API_STATUS_OK;
        },
        []() { op_release(); },
        res);
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
        // P5-A: hardware dependency check against the cached capability set.
        json missing = missing_capabilities(m);
        item["hw_supported"] = missing.empty();
        item["missing_capabilities"] = missing;
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
    // get_host(req) MUST be read here on the poll thread (req is only valid for
    // this event); the resolved manifest is then owned by the job.
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

    // #14: the live status probe (app_status, 5s in main.sh) is the only
    // blocking part; run it on a worker. `data` and `app` are owning copies.
    std::string init_script = jstr(app, "init_script");
    auto probe = std::make_shared<json>();
    return submit_async(
        [probe, init_script]() { *probe = parse_result(script("app_status", init_script)); },
        [data, app, probe](json& res) -> api_status_t {
            json d = data;
            d["probe"] = probe->value("status", "unknown");
            d["app"] = app;
            response(res, 0, STR_OK, d);
            return API_STATUS_OK;
        },
        []() {},
        res);
}

api_status_t api_app::switchApp(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    std::string app_id = body.value("app_id", "");
    if (!valid_app_id(app_id)) {
        response(res, -1, "Invalid app_id");
        return API_STATUS_OK;
    }

    if (in_nodered_mode()) {
        response(res, CODE_NODERED_MODE, MSG_NODERED_MODE);
        return API_STATUS_OK;
    }

    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    json manifests = load_manifests();
    if (!manifests.contains(app_id)) {
        response(res, -1, "Unknown app: " + app_id);
        return API_STATUS_OK;
    }

    // P5-A: never start an app whose declared hardware is absent (the error
    // would otherwise surface as an opaque init-script failure).
    json missing = missing_capabilities(manifests[app_id]);
    if (!missing.empty()) {
        std::string keys;
        for (const auto& k : missing) {
            if (!keys.empty()) {
                keys += ", ";
            }
            keys += k.get<std::string>();
        }
        response(res, -1, "missing hardware capability: " + keys, { { "missing_capabilities", missing } });
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

    // Validate the target init script on the poll thread (fast fs stat) so an
    // invalid target fails fast/synchronously before we go async.
    std::string err;
    if (!check_init_script_fs(target_script, err)) {
        _state = app_state::ERROR;
        _last_error = err;
        response(res, -1, _last_error, { { "state", state_str(_state) } });
        return API_STATUS_OK;
    }

    // #14: the camera hand-over (stop -> sleep 2 -> start -> probe) is blocking;
    // run it on a worker. State/state.json commit happen on the poll thread.
    // _state is left untouched until commit: in the old sync design the
    // intermediate STOPPING/STARTING values were never observable to a
    // concurrent reader (the poll thread was blocked), and deferring the write
    // to commit also avoids leaking a transitional state if the pool saturates.
    bool had_active = !active.empty() && !active_script.empty();
    auto op = std::make_shared<app_op>();
    g.owns = false; // ownership of the op gate transfers to the async finalize
    return submit_async(
        // worker thread: only shell + sleep.
        [op, had_active, active_script, target_script]() {
            if (had_active) {
                sh_stop(active_script, *op);
                if (!op->stop_ok) {
                    return;
                }
            }
            sh_start(target_script, *op);
        },
        // poll thread: map shell outcome to state/state.json + reply.
        [op, app_id, target_script](json& res) -> api_status_t {
            if (!op->stop_ok) {
                _state = app_state::ERROR;
                _last_error = op->stop_err;
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            if (!op->start_ok) {
                // Camera is now free but nothing is running; persist that fact.
                _state = app_state::ERROR;
                _last_error = op->start_err;
                json st = read_state();
                st["active_app"] = nullptr;
                st["active_script"] = nullptr;
                write_state(st);
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            json st = read_state();
            st["active_app"] = app_id;
            st["active_script"] = target_script;
            if (!write_state(st)) {
                // Persisting the selection failed (read-only/full fs). The new
                // app is running but state.json does not record it; roll it back
                // so process reality matches persisted state. (Rare error path;
                // this app_stop briefly runs on the poll thread.)
                LOGE("failed to persist app state; rolling back '%s'", target_script.c_str());
                script("app_stop", target_script);
                _state = app_state::ERROR;
                _last_error = "failed to persist app state (filesystem read-only or full)";
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            _state = app_state::RUNNING;
            _last_error.clear();
            response(res, 0, STR_OK,
                { { "current", app_id },
                    { "state", state_str(_state) },
                    { "probe", op->probe } });
            return API_STATUS_OK;
        },
        []() { op_release(); },
        res);
}

api_status_t api_app::stop(request_t req, response_t res)
{
    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    json state = read_state();
    std::string active = jstr(state, "active_app");
    std::string active_script = jstr(state, "active_script");

    if (active.empty() || active_script.empty()) {
        _state = app_state::STOPPED;
        response(res, 0, STR_OK, { { "state", state_str(_state) }, { "note", "no active app" } });
        return API_STATUS_OK;
    }

    // #14: app_stop (+2s VPSS release) is blocking; run it on a worker.
    // _state is committed only in the wakeup (see switchApp note).
    auto op = std::make_shared<app_op>();
    g.owns = false; // op gate ownership transfers to the async finalize
    return submit_async(
        [op, active_script]() { sh_stop(active_script, *op); },
        [op](json& res) -> api_status_t {
            if (!op->stop_ok) {
                _state = app_state::ERROR;
                _last_error = op->stop_err;
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            _state = app_state::STOPPED;
            json st = read_state();
            st["active_app"] = nullptr;
            st["active_script"] = nullptr;
            if (!write_state(st)) {
                // Stopped but the cleared selection did not persist: a reboot
                // would resurrect the app from the stale state.json. Camera is
                // free, so not fatal, but report the inconsistent fs state.
                LOGE("stop: failed to persist cleared app state");
                _last_error = "app stopped but failed to persist state (filesystem read-only or full)";
                response(res, -1, _last_error, { { "state", state_str(_state) } });
                return API_STATUS_OK;
            }
            _last_error.clear();
            response(res, 0, STR_OK, { { "state", state_str(_state) } });
            return API_STATUS_OK;
        },
        []() { op_release(); },
        res);
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

    if (in_nodered_mode()) {
        response(res, CODE_NODERED_MODE, MSG_NODERED_MODE);
        return API_STATUS_OK;
    }

    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

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
    // (In-app hot model switching is Phase 2.) The restart, when needed, runs
    // on a worker; the sync/no-restart path replies immediately.
    return restart_after_change(app, app_id, { { "app_id", app_id }, { "model", model } }, g, res);
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

    if (in_nodered_mode()) {
        response(res, CODE_NODERED_MODE, MSG_NODERED_MODE);
        return API_STATUS_OK;
    }

    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

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

    // Restart the app if it is the active one so the config takes effect (async
    // when a restart is needed; immediate reply otherwise).
    return restart_after_change(app, app_id, { { "app_id", app_id }, { "values", values } }, g, res);
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

// POST /api/appMgr/installApp  body: {path}
// Installs an application package (.deb) that was previously uploaded via
// fileMgr/upload — which always lands under /userdata — using
// `main.sh app_install` (opkg install --force-reinstall, 120s budget).
//
// Trust boundary (documented on purpose): opkg installs an arbitrary package
// as root. That is the same trust level the rest of this token-protected LAN
// console already grants (init-script control, file upload, shell via ttyd),
// so no additional sandboxing is attempted here. The path validation below is
// about preventing directory traversal / shell injection and accidental
// non-package installs, not about containing a malicious package.
//
// Response data: {path, exit_code, output (opkg tail, <=2KB), apps_count}.
// apps_count is the number of manifests visible after the install so the
// frontend knows to refresh its gallery. code: 0 ok, -1 failed, -2 busy.
api_status_t api_app::installApp(request_t req, response_t res)
{
    static constexpr uintmax_t MAX_DEB_SIZE = 200ull * 1024 * 1024; // 200MB
    static constexpr size_t MAX_OUTPUT_TAIL = 2048; // opkg output tail limit

    auto&& body = parse_body(req);
    std::string path = body.value("path", "");
    if (path.empty() || path.size() >= PATH_MAX) {
        response(res, -1, "Missing or oversized path");
        return API_STATUS_OK;
    }

    if (in_nodered_mode()) {
        response(res, CODE_NODERED_MODE, MSG_NODERED_MODE);
        return API_STATUS_OK;
    }

    // Whitelist: must resolve (realpath) to a regular .deb file under
    // /userdata/. realpath collapses any ../ and symlink tricks first.
    char resolved[PATH_MAX] = { 0 };
    if (realpath(path.c_str(), resolved) == nullptr) {
        response(res, -1, "Package not found: " + path);
        return API_STATUS_OK;
    }
    std::string deb(resolved);
    if (deb.rfind("/userdata/", 0) != 0) {
        response(res, -1, "Package must live under /userdata (upload it via the file manager first)");
        return API_STATUS_OK;
    }
    const std::string suffix = ".deb";
    if (deb.size() <= suffix.size() || deb.compare(deb.size() - suffix.size(), suffix.size(), suffix) != 0) {
        response(res, -1, "Package must be a .deb file");
        return API_STATUS_OK;
    }
    // The resolved path is passed to main.sh through a single-quoted shell
    // argument (api_base::script). Restrict it to a safe charset so it can
    // never break out of the quoting, whatever the uploaded filename was.
    for (char c : deb) {
        if (!(std::isalnum((unsigned char)c) || c == '/' || c == '.' || c == '-' || c == '_' || c == '+')) {
            response(res, -1, "Package path contains unsupported characters");
            return API_STATUS_OK;
        }
    }
    struct stat st;
    if (stat(deb.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        response(res, -1, "Package is not a regular file");
        return API_STATUS_OK;
    }
    if ((uintmax_t)st.st_size >= MAX_DEB_SIZE) {
        response(res, -1, "Package exceeds the 200MB limit");
        return API_STATUS_OK;
    }

    // Installing while another operation switches/stops apps would race the
    // camera hand-over; same busy semantics as switch/setModel. Acquired here
    // on the poll thread, released in the async finalize (op_release).
    if (!op_try_acquire()) {
        response(res, -2, "busy: another app operation is in progress");
        return API_STATUS_OK;
    }

    // #14: opkg install has a 130s budget and used to freeze the whole
    // management plane. Run it on a worker; commit + reply on the poll thread.
    // `deb` is an owning std::string, captured by value into the worker.
    auto out = std::make_shared<std::string>();
    return submit_async(
        // worker thread: only the blocking shell call.
        [out, deb]() {
            // main.sh output protocol: first line "EXIT:<code>" (124 = opkg
            // timed out after 120s), then the tail of the opkg output. 130s
            // here so the shell-side timeout always fires first.
            *out = script_timeout(130, "app_install", deb);
        },
        // poll thread: parse, rescan manifests, build the exact same response.
        [out, deb](json& res) -> api_status_t {
            int exit_code = -1;
            std::string opkg_out = *out;
            if (out->rfind("EXIT:", 0) == 0) {
                size_t nl = out->find('\n');
                std::string code_str = out->substr(5, (nl == std::string::npos ? out->size() : nl) - 5);
                try {
                    exit_code = std::stoi(code_str);
                } catch (const std::exception&) {
                    exit_code = -1;
                }
                opkg_out = (nl == std::string::npos) ? "" : out->substr(nl + 1);
            }
            if (opkg_out.size() > MAX_OUTPUT_TAIL) {
                opkg_out = opkg_out.substr(opkg_out.size() - MAX_OUTPUT_TAIL);
            }

            // Rescan manifests: a well-behaved package drops its manifest into
            // /userdata/local/apps/ (or ships a built-in one), so the count
            // tells the frontend the gallery content may have changed.
            int apps_count = (int)load_manifests().size();

            json data = json::object();
            data["path"] = deb;
            data["exit_code"] = exit_code;
            data["output"] = opkg_out;
            data["apps_count"] = apps_count;

            if (exit_code == 0) {
                response(res, 0, STR_OK, data);
            } else if (exit_code == 124) {
                response(res, -1, "opkg install timed out (120s)", data);
            } else {
                response(res, -1, "opkg install failed (exit " + std::to_string(exit_code) + ")", data);
            }
            return API_STATUS_OK;
        },
        // poll thread: release the op gate whatever happened.
        []() { op_release(); },
        res);
}

// --- Home Assistant MQTT integration ----------------------------------------
//
// Config lives in /userdata/local/ha.conf (see ha_config.h). The active app's
// init script / runtime is expected to source it; changing it therefore
// restarts the active app (same building blocks as setModel/setConfig).

namespace {

// libmosquitto global init, once per process.
std::once_flag g_mosq_init_flag;
void ensure_mosq_init()
{
    std::call_once(g_mosq_init_flag, []() { mosquitto_lib_init(); });
}

struct ha_test_result {
    bool ok = false;
    int mosq_rc = 0; // mosquitto rc or CONNACK code (see message for context)
    std::string message;
};

// Bounded TCP connect (the OS default connect timeout can be minutes; the
// endpoint promises ~5s). Returns false with err filled on failure.
bool tcp_probe(const std::string& host, int port, int timeout_ms, std::string& err)
{
    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    int grc = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (grc != 0 || res == nullptr) {
        err = std::string("cannot resolve host: ") + gai_strerror(grc);
        return false;
    }
    bool connected = false;
    for (struct addrinfo* ai = res; ai != nullptr && !connected; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_NONBLOCK, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        int c = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (c == 0) {
            connected = true;
        } else if (errno == EINPROGRESS) {
            struct pollfd pfd = { fd, POLLOUT, 0 };
            int pr = ::poll(&pfd, 1, timeout_ms);
            if (pr == 1) {
                int so_err = 0;
                socklen_t len = sizeof(so_err);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) == 0 && so_err == 0) {
                    connected = true;
                } else {
                    err = std::string("connect failed: ") + strerror(so_err ? so_err : ECONNREFUSED);
                }
            } else {
                err = "connect timed out";
            }
        } else {
            err = std::string("connect failed: ") + strerror(errno);
        }
        ::close(fd);
    }
    freeaddrinfo(res);
    if (!connected && err.empty()) {
        err = "connect failed";
    }
    return connected;
}

// Worker-thread MQTT connectivity probe: TCP precheck (bounded) then a real
// MQTT CONNECT waiting for the CONNACK, all within ~5s.
ha_test_result mqtt_probe(const std::string& host, int port,
    const std::string& username, const std::string& password)
{
    static constexpr int TIMEOUT_MS = 5000;
    ha_test_result r;

    std::string tcp_err;
    if (!tcp_probe(host, port, TIMEOUT_MS, tcp_err)) {
        r.mosq_rc = MOSQ_ERR_NO_CONN;
        r.message = tcp_err;
        return r;
    }

    ensure_mosq_init();
    // connack: -1 = not received yet; >=0 = CONNACK code (0 accepted).
    auto connack = std::make_shared<std::atomic<int>>(-1);
    struct mosquitto* m = mosquitto_new(nullptr, true, connack.get());
    if (m == nullptr) {
        r.mosq_rc = MOSQ_ERR_NOMEM;
        r.message = "mosquitto_new failed";
        return r;
    }
    mosquitto_connect_callback_set(m, [](struct mosquitto*, void* ud, int rc) {
        static_cast<std::atomic<int>*>(ud)->store(rc);
    });
    if (!username.empty()) {
        mosquitto_username_pw_set(m, username.c_str(), password.c_str());
    }

    int rc = mosquitto_connect(m, host.c_str(), port, 10);
    if (rc != MOSQ_ERR_SUCCESS) {
        r.mosq_rc = rc;
        r.message = mosquitto_strerror(rc);
        mosquitto_destroy(m);
        return r;
    }

    // Pump the network loop until the CONNACK arrives or the deadline hits.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(TIMEOUT_MS);
    while (connack->load() < 0 && std::chrono::steady_clock::now() < deadline) {
        rc = mosquitto_loop(m, 200, 1);
        if (rc != MOSQ_ERR_SUCCESS) {
            break;
        }
    }

    int ack = connack->load();
    if (ack == 0) {
        r.ok = true;
        r.mosq_rc = 0;
        r.message = "connection accepted";
    } else if (ack > 0) {
        r.mosq_rc = ack;
        r.message = std::string("broker refused connection: ") + mosquitto_connack_string(ack);
    } else if (rc != MOSQ_ERR_SUCCESS) {
        r.mosq_rc = rc;
        r.message = std::string("connection error: ") + mosquitto_strerror(rc);
    } else {
        r.mosq_rc = MOSQ_ERR_CONN_PENDING;
        r.message = "timed out waiting for broker response (5s)";
    }
    mosquitto_disconnect(m);
    mosquitto_destroy(m);
    return r;
}

// Public (non-secret) view of the config for API replies.
json ha_conf_to_json(const ha_config::conf& c)
{
    json data = json::object();
    data["enabled"] = c.enabled;
    data["broker_host"] = c.broker_host;
    data["broker_port"] = c.broker_port;
    data["username"] = c.username;
    data["discovery_prefix"] = c.discovery_prefix;
    data["password_set"] = !c.password.empty();
    return data;
}

} // namespace

// GET/POST /api/appMgr/getHaConfig
// Returns every field except the password itself, plus password_set.
api_status_t api_app::getHaConfig(request_t req, response_t res)
{
    response(res, 0, STR_OK, ha_conf_to_json(ha_config::load()));
    return API_STATUS_OK;
}

// POST /api/appMgr/setHaConfig
// body: {enabled, broker_host, broker_port, username?, password?, discovery_prefix?}
// password absent -> the stored password is kept. Shares the app op busy gate
// (same level as switchApp); after the atomic conf write the active app (if
// any) is restarted so the new config takes effect. Reply data carries the
// saved (password-less) config + restarted bool.
api_status_t api_app::setHaConfig(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    if (!body.is_object() || !body.contains("enabled") || !body["enabled"].is_boolean()) {
        response(res, -1, "Missing or invalid 'enabled' (boolean required)");
        return API_STATUS_OK;
    }

    ha_config::conf cur = ha_config::load();
    ha_config::conf next = cur; // password retained unless provided

    next.enabled = body["enabled"].get<bool>();

    if (body.contains("broker_host")) {
        if (!body["broker_host"].is_string()) {
            response(res, -1, "Invalid broker_host");
            return API_STATUS_OK;
        }
        next.broker_host = body["broker_host"].get<std::string>();
    }
    if (!ha_config::valid_value(next.broker_host) || next.broker_host.find(' ') != std::string::npos) {
        response(res, -1, "Invalid broker_host");
        return API_STATUS_OK;
    }
    if (next.enabled && next.broker_host.empty()) {
        response(res, -1, "broker_host is required when enabled");
        return API_STATUS_OK;
    }

    if (body.contains("broker_port")) {
        if (!body["broker_port"].is_number_integer()) {
            response(res, -1, "Invalid broker_port (integer 1-65535 required)");
            return API_STATUS_OK;
        }
        next.broker_port = body["broker_port"].get<int>();
    }
    if (next.broker_port < 1 || next.broker_port > 65535) {
        response(res, -1, "Invalid broker_port (integer 1-65535 required)");
        return API_STATUS_OK;
    }

    if (body.contains("username")) {
        if (!body["username"].is_string() || !ha_config::valid_value(body["username"].get<std::string>())) {
            response(res, -1, "Invalid username");
            return API_STATUS_OK;
        }
        next.username = body["username"].get<std::string>();
    }
    if (body.contains("password")) {
        if (!body["password"].is_string() || !ha_config::valid_value(body["password"].get<std::string>())) {
            response(res, -1, "Invalid password");
            return API_STATUS_OK;
        }
        next.password = body["password"].get<std::string>();
    }
    if (body.contains("discovery_prefix")) {
        if (!body["discovery_prefix"].is_string() || !ha_config::valid_value(body["discovery_prefix"].get<std::string>())) {
            response(res, -1, "Invalid discovery_prefix");
            return API_STATUS_OK;
        }
        next.discovery_prefix = body["discovery_prefix"].get<std::string>();
    }
    if (next.discovery_prefix.empty()) {
        next.discovery_prefix = "homeassistant";
    }

    // Same busy level as switchApp: writing the conf and restarting the active
    // app must not race a concurrent app operation.
    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    if (!ha_config::save(next)) {
        response(res, -1, "Failed to persist Home Assistant config");
        return API_STATUS_OK;
    }

    json data = ha_conf_to_json(next);

    // Node-RED mode: the conf is persisted, but the C++ app stack is parked —
    // never restart it (that would start a camera app under Node-RED's feet).
    // The saved config takes effect when the device returns to console mode.
    if (in_nodered_mode()) {
        data["restarted"] = false;
        data["note"] = "nodered_mode";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    // Restart the active app (if any) so the new config takes effect.
    json state = read_state();
    std::string active = jstr(state, "active_app");
    if (active.empty()) {
        data["restarted"] = false;
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    json manifests = load_manifests();
    if (!manifests.contains(active)) {
        data["restarted"] = false;
        data["note"] = "active app manifest missing, not restarted";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    return restart_after_change(manifests[active], active, data, g, res);
}

// POST /api/appMgr/testHaConnection
// body: {broker_host, broker_port, username?, password?, use_saved_password?}
// Never touches ha.conf. Runs the (up to ~5s) probe on a worker.
// code 0 ok, -1 failure (data.mosquitto_rc + message), -2 concurrent test.
api_status_t api_app::testHaConnection(request_t req, response_t res)
{
    auto&& body = parse_body(req);

    std::string host = body.value("broker_host", "");
    if (host.empty() || !ha_config::valid_value(host) || host.find(' ') != std::string::npos) {
        response(res, -1, "Missing or invalid broker_host");
        return API_STATUS_OK;
    }
    int port = 1883;
    if (body.contains("broker_port")) {
        if (!body["broker_port"].is_number_integer()) {
            response(res, -1, "Invalid broker_port (integer 1-65535 required)");
            return API_STATUS_OK;
        }
        port = body["broker_port"].get<int>();
    }
    if (port < 1 || port > 65535) {
        response(res, -1, "Invalid broker_port (integer 1-65535 required)");
        return API_STATUS_OK;
    }
    std::string username = body.value("username", "");
    std::string password = body.value("password", "");
    if (!ha_config::valid_value(username) || !ha_config::valid_value(password)) {
        response(res, -1, "Invalid username or password");
        return API_STATUS_OK;
    }
    // use_saved_password: reuse the stored secret so the frontend can test
    // edited settings without forcing the user to re-type the password.
    if (body.value("use_saved_password", false)) {
        password = ha_config::load().password;
    }

    if (_ha_test_busy.test_and_set(std::memory_order_acquire)) {
        response(res, -2, "busy: a connection test is already running");
        return API_STATUS_OK;
    }

    auto result = std::make_shared<ha_test_result>();
    return submit_async(
        // worker thread: bounded network probe only.
        [result, host, port, username, password]() {
            *result = mqtt_probe(host, port, username, password);
        },
        // poll thread: map the probe outcome to the reply.
        [result, host, port](json& res) -> api_status_t {
            json data = json::object();
            data["broker_host"] = host;
            data["broker_port"] = port;
            data["mosquitto_rc"] = result->mosq_rc;
            if (result->ok) {
                response(res, 0, STR_OK, data);
            } else {
                response(res, -1, result->message, data);
            }
            return API_STATUS_OK;
        },
        []() { _ha_test_busy.clear(std::memory_order_release); },
        res);
}

// --- ONVIF integration ---

namespace {

// Public (non-secret) view of the config for API replies.
json onvif_conf_to_json(const onvif_config::conf& c)
{
    json data = json::object();
    data["meta_enabled"] = c.meta_enabled;
    data["meta_interval_ms"] = c.meta_interval_ms;
    data["meta_profile"] = c.meta_profile;
    data["meta_prefix"] = c.meta_prefix;
    data["service_enabled"] = c.service_enabled;
    data["service_port"] = c.service_port;
    data["username"] = c.username;
    data["password_set"] = !c.password.empty();
    data["location"] = c.location;
    return data;
}

} // namespace

// GET/POST /api/appMgr/getOnvifConfig
// Returns every field except the password itself, plus password_set.
api_status_t api_app::getOnvifConfig(request_t req, response_t res)
{
    response(res, 0, STR_OK, onvif_conf_to_json(onvif_config::load()));
    return API_STATUS_OK;
}

// POST /api/appMgr/setOnvifConfig
// body: any subset of {meta_enabled, meta_interval_ms, meta_profile,
//                      meta_prefix, service_enabled, service_port,
//                      username, location} plus an optional plaintext password.
// Every omitted field keeps its stored value. Shares the app op busy gate (same
// level as switchApp); after the atomic conf write the active app (if any) is
// restarted so the new config takes effect. Reply data carries restarted + note.
api_status_t api_app::setOnvifConfig(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    if (!body.is_object()) {
        response(res, -1, "Invalid body (JSON object required)");
        return API_STATUS_OK;
    }

    onvif_config::conf cur = onvif_config::load();
    onvif_config::conf next = cur; // password retained unless provided

    if (body.contains("meta_enabled")) {
        if (!body["meta_enabled"].is_boolean()) {
            response(res, -1, "Invalid meta_enabled (boolean required)");
            return API_STATUS_OK;
        }
        next.meta_enabled = body["meta_enabled"].get<bool>();
    }
    if (body.contains("service_enabled")) {
        if (!body["service_enabled"].is_boolean()) {
            response(res, -1, "Invalid service_enabled (boolean required)");
            return API_STATUS_OK;
        }
        next.service_enabled = body["service_enabled"].get<bool>();
    }

    if (body.contains("meta_interval_ms")) {
        if (!body["meta_interval_ms"].is_number_integer()) {
            response(res, -1, "Invalid meta_interval_ms (integer 20-60000 required)");
            return API_STATUS_OK;
        }
        next.meta_interval_ms = body["meta_interval_ms"].get<int>();
    }
    // Rejected rather than clamped: the reader clamps silently, and a console
    // that echoed back a value the device did not actually use would be lying
    // to the user about the publish rate they configured.
    if (next.meta_interval_ms < onvif_config::META_INTERVAL_MIN_MS
        || next.meta_interval_ms > onvif_config::META_INTERVAL_MAX_MS) {
        response(res, -1, "Invalid meta_interval_ms (integer 20-60000 required)");
        return API_STATUS_OK;
    }

    if (body.contains("service_port")) {
        if (!body["service_port"].is_number_integer()) {
            response(res, -1, "Invalid service_port (integer 1025-65535 required)");
            return API_STATUS_OK;
        }
        next.service_port = body["service_port"].get<int>();
    }
    // Same reasoning as the interval, with a sharper failure mode: applications
    // do not run as root, so a privileged port would present to the user as
    // "ONVIF is on but nothing answers".
    if (next.service_port <= onvif_config::SERVICE_PORT_MIN_EXCLUSIVE
        || next.service_port >= onvif_config::SERVICE_PORT_MAX_EXCLUSIVE) {
        response(res, -1, "Invalid service_port (integer 1025-65535 required)");
        return API_STATUS_OK;
    }

    if (body.contains("meta_profile")) {
        if (!body["meta_profile"].is_string() || !onvif_config::valid_value(body["meta_profile"].get<std::string>())) {
            response(res, -1, "Invalid meta_profile");
            return API_STATUS_OK;
        }
        next.meta_profile = body["meta_profile"].get<std::string>();
    }
    // The profile name becomes part of the metadata topic, so an empty one
    // would produce a malformed topic; fall back to the documented default.
    if (next.meta_profile.empty()) {
        next.meta_profile = "live0";
    }

    if (body.contains("meta_prefix")) {
        if (!body["meta_prefix"].is_string() || !onvif_config::valid_value(body["meta_prefix"].get<std::string>())) {
            response(res, -1, "Invalid meta_prefix");
            return API_STATUS_OK;
        }
        // Empty is meaningful here, unlike the profile: it tells the reader to
        // substitute the device identifier.
        next.meta_prefix = body["meta_prefix"].get<std::string>();
    }

    if (body.contains("username")) {
        if (!body["username"].is_string() || !onvif_config::valid_value(body["username"].get<std::string>())) {
            response(res, -1, "Invalid username");
            return API_STATUS_OK;
        }
        next.username = body["username"].get<std::string>();
    }
    if (body.contains("location")) {
        if (!body["location"].is_string() || !onvif_config::valid_value(body["location"].get<std::string>())) {
            response(res, -1, "Invalid location");
            return API_STATUS_OK;
        }
        next.location = body["location"].get<std::string>();
    }

    // Absent password and empty password mean different things, and the console
    // depends on the difference: the form never receives the stored password
    // back (only password_set), so it omits the key to leave it alone, and
    // sends "" when the user explicitly clears the credential.
    if (body.contains("password")) {
        if (!body["password"].is_string() || !onvif_config::valid_value(body["password"].get<std::string>())) {
            response(res, -1, "Invalid password");
            return API_STATUS_OK;
        }
        next.password = body["password"].get<std::string>();
    }

    // Same busy level as switchApp: writing the conf and restarting the active
    // app must not race a concurrent app operation.
    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    if (!onvif_config::save(next)) {
        response(res, -1, "Failed to persist ONVIF config");
        return API_STATUS_OK;
    }

    json data = json::object();
    data["restarted"] = true;
    data["note"] = "";

    // Node-RED mode: the conf is persisted, but the C++ app stack is parked --
    // never restart it (that would start a camera app under Node-RED's feet).
    // The saved config takes effect when the device returns to console mode.
    if (in_nodered_mode()) {
        data["restarted"] = false;
        data["note"] = "nodered_mode";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    // Restart the active app (if any) so the new config takes effect.
    json state = read_state();
    std::string active = jstr(state, "active_app");
    if (active.empty()) {
        data["restarted"] = false;
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    json manifests = load_manifests();
    if (!manifests.contains(active)) {
        data["restarted"] = false;
        data["note"] = "active app manifest missing, not restarted";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    return restart_after_change(manifests[active], active, data, g, res);
}

// --- Privacy blur ---

namespace {

json blur_conf_to_json(const blur_config::conf& c)
{
    json data = json::object();
    data["enabled"] = c.enabled;
    data["backend"] = c.backend;
    data["block_px"] = c.block_px;
    data["max_regions"] = c.max_regions;
    data["alpha"] = c.alpha;
    return data;
}

} // namespace

// GET/POST /api/appMgr/getBlurConfig
api_status_t api_app::getBlurConfig(request_t req, response_t res)
{
    response(res, 0, STR_OK, blur_conf_to_json(blur_config::load()));
    return API_STATUS_OK;
}

// POST /api/appMgr/setBlurConfig
// body: any subset of {enabled, backend, block_px, max_regions, alpha}. Every omitted
// field keeps its stored value. Shares the app op busy gate (same level as
// switchApp); after the atomic conf write the active app (if any) is restarted
// so the new config takes effect. Reply data carries restarted + note.
api_status_t api_app::setBlurConfig(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    if (!body.is_object()) {
        response(res, -1, "Invalid body (JSON object required)");
        return API_STATUS_OK;
    }

    blur_config::conf next = blur_config::load();

    if (body.contains("enabled")) {
        if (!body["enabled"].is_boolean()) {
            response(res, -1, "Invalid enabled (boolean required)");
            return API_STATUS_OK;
        }
        next.enabled = body["enabled"].get<bool>();
    }

    if (body.contains("backend")) {
        if (!body["backend"].is_string() || !blur_config::valid_backend(body["backend"].get<std::string>())) {
            response(res, -1, "Invalid backend (mosaic, coverex or pixelate required)");
            return API_STATUS_OK;
        }
        next.backend = body["backend"].get<std::string>();
    }

    if (body.contains("block_px")) {
        if (!body["block_px"].is_number_integer()) {
            response(res, -1, "Invalid block_px (integer 8 or 16 required)");
            return API_STATUS_OK;
        }
        next.block_px = body["block_px"].get<int>();
    }
    // Rejected rather than snapped to the nearest supported size: a console
    // that echoed back a block size the hardware does not actually use would
    // be lying to the user about how coarse their masking is.
    if (!blur_config::valid_block_px(next.block_px)) {
        response(res, -1, "Invalid block_px (integer 8 or 16 required)");
        return API_STATUS_OK;
    }

    if (body.contains("max_regions")) {
        if (!body["max_regions"].is_number_integer()) {
            response(res, -1, "Invalid max_regions (integer 1-8 required)");
            return API_STATUS_OK;
        }
        next.max_regions = body["max_regions"].get<int>();
    }
    // Same reasoning as the block size, with a privacy failure mode: silently
    // lowering the ceiling would leave subjects unmasked while the console
    // claimed they were covered.
    if (!blur_config::valid_max_regions(next.max_regions)) {
        response(res, -1, "Invalid max_regions (integer 1-8 required)");
        return API_STATUS_OK;
    }

    if (body.contains("alpha")) {
        if (!body["alpha"].is_number_integer()) {
            response(res, -1, "Invalid alpha (integer 0-255 required)");
            return API_STATUS_OK;
        }
        next.alpha = body["alpha"].get<int>();
    }
    // Out-of-range alpha is rejected instead of clamped, for the same reason a
    // bad block size is: the console must never report a mask strength the
    // device is not actually applying. Clamping would be especially misleading
    // here because the direction matters -- a caller asking for 300 and
    // silently getting 255 is harmless, but one asking for -1 and getting 0
    // would end up with an invisible mask it believes is opaque.
    if (!blur_config::valid_alpha(next.alpha)) {
        response(res, -1, "Invalid alpha (integer 0-255 required)");
        return API_STATUS_OK;
    }

    // Same busy level as switchApp: writing the conf and restarting the active
    // app must not race a concurrent app operation.
    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    // Which fields changed decides whether anything has to be restarted.
    // Read before the write, obviously, and compared field by field rather than
    // wholesale: an operator adjusting opacity while watching the stream should
    // not have the stream taken away, and that is the only way they can judge
    // whether the value is right.
    const blur_config::conf prev = blur_config::load();
    const bool cold_changed = prev.backend != next.backend ||
                              prev.block_px != next.block_px ||
                              prev.max_regions != next.max_regions;

    if (!blur_config::save(next)) {
        response(res, -1, "Failed to persist privacy blur config");
        return API_STATUS_OK;
    }

    // Opacity is a kernel parameter the mask unit re-reads on every region
    // update, so pushing it here makes it effective immediately -- even when no
    // application is running to notice the file changed.
    if (prev.alpha != next.alpha) {
        FILE* f = ::fopen("/sys/module/cv181x_vpss/parameters/mask_alpha", "w");
        if (f != nullptr) {
            ::fprintf(f, "%d\n", next.alpha);
            ::fclose(f);
        }
        // No error if absent: a stock kernel simply has no such parameter, and
        // its mask is always fully opaque, which is the safe end to fail at.
    }

    json data = json::object();
    data["restarted"] = true;
    data["note"] = "";

    // The application re-reads the switch, the opacity and the block cap from
    // the conf file about once a second, so those need no restart at all.
    if (!cold_changed) {
        data["restarted"] = false;
        data["note"] = "applied_live";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    // Node-RED mode: the conf is persisted, but the C++ app stack is parked --
    // never restart it (that would start a camera app under Node-RED's feet).
    // The saved config takes effect when the device returns to console mode.
    if (in_nodered_mode()) {
        data["restarted"] = false;
        data["note"] = "nodered_mode";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    // Restart the active app (if any) so the new config takes effect.
    json state = read_state();
    std::string active = jstr(state, "active_app");
    if (active.empty()) {
        data["restarted"] = false;
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    json manifests = load_manifests();
    if (!manifests.contains(active)) {
        data["restarted"] = false;
        data["note"] = "active app manifest missing, not restarted";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    return restart_after_change(manifests[active], active, data, g, res);
}

// --- Camera picture orientation + focus assist ---

namespace {

json camera_conf_to_json(const camera_config::conf& c)
{
    json data = json::object();
    data["mirror"] = c.mirror;
    data["flip"] = c.flip;
    data["rotation"] = c.rotation;
    return data;
}

} // namespace

// GET/POST /api/appMgr/getCameraConfig
// -> {mirror:bool, flip:bool, rotation:0|180}
api_status_t api_app::getCameraConfig(request_t req, response_t res)
{
    response(res, 0, STR_OK, camera_conf_to_json(camera_config::load()));
    return API_STATUS_OK;
}

// POST /api/appMgr/setCameraConfig
// body: {mirror:bool, flip:bool, rotation:0|180} (each field optional, the
// stored value is kept when absent). Shares the app op busy gate (same level
// as switchApp/setHaConfig). camera.conf orientation is folded into VI only at
// startup (applyCameraConf runs before VI init); applying it mid-stream races
// the frame path and hangs VPSS, so after the atomic conf write the active app
// (if any) is restarted so the new orientation takes effect. Reply data carries
// the saved config, restarted:bool and applied:"restart".
api_status_t api_app::setCameraConfig(request_t req, response_t res)
{
    auto&& body = parse_body(req);
    if (!body.is_object()) {
        response(res, -1, "Invalid request body (JSON object required)");
        return API_STATUS_OK;
    }

    camera_config::conf next = camera_config::load();

    if (body.contains("mirror")) {
        if (!body["mirror"].is_boolean()) {
            response(res, -1, "Invalid mirror (boolean required)");
            return API_STATUS_OK;
        }
        next.mirror = body["mirror"].get<bool>();
    }
    if (body.contains("flip")) {
        if (!body["flip"].is_boolean()) {
            response(res, -1, "Invalid flip (boolean required)");
            return API_STATUS_OK;
        }
        next.flip = body["flip"].get<bool>();
    }
    if (body.contains("rotation")) {
        if (!body["rotation"].is_number_integer()) {
            response(res, -1, "Invalid rotation (0 or 180 required)");
            return API_STATUS_OK;
        }
        int r = body["rotation"].get<int>();
        if (r != 0 && r != 180) {
            response(res, -1, "Invalid rotation (0 or 180 required)");
            return API_STATUS_OK;
        }
        next.rotation = r;
    }

    // Same busy level as switchApp: conservative — writing the conf while an
    // app operation is in flight is harmless, but keep the gate anyway.
    op_guard g;
    if (!acquire_op_or_busy(res, g)) {
        return API_STATUS_OK;
    }

    if (!camera_config::save(next)) {
        response(res, -1, "Failed to persist camera config");
        return API_STATUS_OK;
    }

    json data = camera_conf_to_json(next);
    data["applied"] = "restart"; // orientation only takes effect on app restart

    // Node-RED mode: the conf is persisted, but the C++ app stack is parked —
    // never restart it (that would start a camera app under Node-RED's feet).
    // The saved orientation takes effect when the device returns to console mode.
    if (in_nodered_mode()) {
        data["restarted"] = false;
        data["note"] = "nodered_mode";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    // Restart the active app (if any) so the new orientation is folded into VI.
    json state = read_state();
    std::string active = jstr(state, "active_app");
    if (active.empty()) {
        data["restarted"] = false;
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    json manifests = load_manifests();
    if (!manifests.contains(active)) {
        data["restarted"] = false;
        data["note"] = "active app manifest missing, not restarted";
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    return restart_after_change(manifests[active], active, data, g, res);
}

// GET/POST /api/appMgr/getFocusValue
// Reads /tmp/camera_fv.json ({"fv":N,"ts":M}, refreshed ~200ms by the active
// app's video component) and echoes it plus available:bool. Freshness is
// judged by the file's mtime (the embedded ts is device-monotonic, not
// comparable to wall time): missing file or mtime older than 2s ->
// available:false. Lightweight read-only poll target: no lock, no busy gate.
api_status_t api_app::getFocusValue(request_t req, response_t res)
{
    static constexpr const char* FV_FILE = "/tmp/camera_fv.json";
    static constexpr double FV_STALE_SECONDS = 2.0;

    json data = json::object();
    data["available"] = false;

    struct stat st {};
    if (::stat(FV_FILE, &st) != 0) {
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    std::ifstream f(FV_FILE);
    if (!f.is_open()) {
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }
    json fv = json::parse(f, nullptr, false);
    if (fv.is_discarded() || !fv.is_object() || !fv.contains("fv") || !fv["fv"].is_number()) {
        response(res, 0, STR_OK, data);
        return API_STATUS_OK;
    }

    data["fv"] = fv["fv"];
    if (fv.contains("ts")) {
        data["ts"] = fv["ts"];
    }

    // Freshness: writer updates every ~200ms while the camera runs. A stale
    // mtime means the camera stopped without removing the file.
    struct timespec now {};
    ::clock_gettime(CLOCK_REALTIME, &now);
    double age = (double)(now.tv_sec - st.st_mtim.tv_sec) + (double)(now.tv_nsec - st.st_mtim.tv_nsec) / 1e9;
    data["available"] = (age >= 0.0 && age <= FV_STALE_SECONDS);

    response(res, 0, STR_OK, data);
    return API_STATUS_OK;
}
