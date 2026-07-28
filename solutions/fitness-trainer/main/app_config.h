#ifndef _FITNESS_APP_CONFIG_H_
#define _FITNESS_APP_CONFIG_H_

// Loader + watcher for the supervisor-managed per-app config file:
//   /userdata/local/apps/fitness-trainer.config.json
// written atomically by the console's appMgr/setConfig after validating
// against this app's manifest config_schema.
//
// The console restarts the app after a config change, so a plain load at
// startup would already be enough for the exercise selector to work. The
// watcher exists for the other path: editing the file over SSH, which is how
// the mode gets switched from a phone, a shell script, or Node-RED without a
// console round-trip. Re-reading costs one stat() every kPollFrames frames.
//
// Contract: a missing, unreadable or malformed file leaves every field at its
// default. Individual malformed keys are skipped with a warning, never fatal.

#include <string>

namespace fitness {

struct AppConfig {
    std::string mode = "squat";     // one of Exercise::ids()
    int target_reps = 12;
    int target_sets = 3;
    float confidence = 0.40f;       // person detection score
    float keypoint_confidence = 0.50f;
    int idle_reset_seconds = 60;    // reset the workout after this long with
                                    // nobody in frame; 0 disables
};

// Parse `path` into `out`. Returns true when the file existed and parsed as a
// JSON object. Fields absent from the file keep the value already in `out`.
bool load_app_config(const std::string& path, AppConfig& out);

// Reloads `path` when its mtime changes.
class ConfigWatcher {
public:
    explicit ConfigWatcher(std::string path) : path_(std::move(path)) {}

    // Load once at startup. Returns the config regardless of whether a file
    // was found.
    const AppConfig& loadInitial();

    // Cheap poll. Returns true when the file changed AND re-parsed, in which
    // case config() holds the new values. Call once per frame; it stats the
    // file only every kPollFrames calls.
    bool poll();

    const AppConfig& config() const { return config_; }

private:
    static constexpr int kPollFrames = 30;

    std::string path_;
    AppConfig config_;
    long last_mtime_ = 0;
    int frames_since_stat_ = 0;
};

}  // namespace fitness

#endif  // _FITNESS_APP_CONFIG_H_
