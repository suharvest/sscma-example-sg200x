#ifndef API_DEVICE_H
#define API_DEVICE_H

#include <memory>
#include <stdexcept>
#include <mutex>
#include <deque>
#include <thread>
#include <atomic>
#include <condition_variable>

#include "api_base.h"
#include "serviced.h"

class api_device : public api_base {
private:
    // Battery voltage cache structure
    struct BatteryVoltageData {
        long voltage_mv;       // Voltage in millivolts
        long raw_value;        // Raw ADC value
        double scale;          // Scale factor
        bool valid;            // Data validity flag
    };

    static inline std::unique_ptr<serviced> _serviced;

    // Battery monitoring variables
    static const int VOLTAGE_QUEUE_SIZE = 10;
    static const int VOLTAGE_THRESHOLD_MV = 100;  // 100mV threshold for filtering
    static inline std::deque<BatteryVoltageData> _voltage_queue;
    static inline std::mutex _voltage_queue_mutex;
    static inline std::thread _battery_collector_thread;
    static inline std::atomic<bool> _battery_collector_running{false};
    static inline BatteryVoltageData _cached_voltage_data{0, 0, 0.0, false};
    static inline bool _battery_collector_initialized = false;
    static inline std::condition_variable _battery_cv;
    static inline std::mutex _battery_init_mutex;
    static inline bool _adc_available = false;

    static api_status_t getCameraWebsocketUrl(request_t req, response_t res);

    static api_status_t getDeviceInfo(request_t req, response_t res);
    static api_status_t getDeviceList(request_t req, response_t res);
    static api_status_t updateDeviceName(request_t req, response_t res);
    static api_status_t queryDeviceInfo(request_t req, response_t res);
    static api_status_t getSystemStatus(request_t req, response_t res);
    static api_status_t queryServiceStatus(request_t req, response_t res);
    static api_status_t setPower(request_t req, response_t res);

    static api_status_t getAppInfo(request_t req, response_t res);
    static api_status_t uploadApp(request_t req, response_t res);

    static api_status_t getModelFile(request_t req, response_t res);
    static api_status_t getModelInfo(request_t req, response_t res);
    static api_status_t getModelList(request_t req, response_t res);
    static api_status_t uploadModel(request_t req, response_t res);

    static api_status_t getPlatformInfo(request_t req, response_t res);
    static api_status_t savePlatformInfo(request_t req, response_t res);

    static api_status_t updateChannel(request_t req, response_t res);
    static api_status_t cancelUpdate(request_t req, response_t res);
    static api_status_t getSystemUpdateVersion(request_t req, response_t res);
    static api_status_t getUpdateProgress(request_t req, response_t res);
    static api_status_t updateSystem(request_t req, response_t res);

    // Timestamp APIs
    static api_status_t setTimestamp(request_t req, response_t res);
    static api_status_t getTimestamp(request_t req, response_t res);
    static api_status_t setTimezone(request_t req, response_t res);
    static api_status_t getTimezone(request_t req, response_t res);
    static api_status_t getTimezoneList(request_t req, response_t res);
    static api_status_t syncTime(request_t req, response_t res);

    static api_status_t queryBatteryInfo(request_t req, response_t res);
    static api_status_t getSensorStatus(request_t req, response_t res);

    // Hardware capabilities (P5-A): shell probe, cached for the process
    // lifetime (hardware does not change while running); ?refresh=1 reprobes.
    static api_status_t getCapabilities(request_t req, response_t res);
    // The "type" string reported by queryDeviceInfo (shared builder so
    // capabilities.device_type is byte-identical).
    static std::string build_device_type();
    // Run main.sh queryCapabilities and normalize the result (all keys
    // present with sane defaults even if the script fails).
    static json probe_capabilities();

    // Runtime mode (P4-D): console (gallery apps) <-> Node-RED.
    static api_status_t getRunMode(request_t req, response_t res);
    static api_status_t setRunMode(request_t req, response_t res);
    // Hard escape hatch (#20): force console without the graceful hand-over,
    // independent of Node-RED health. Used to recover a wedged device.
    static api_status_t forceConsole(request_t req, response_t res);
    // Reads /userdata/local/apps/mode; anything but "nodered" -> "console".
    static std::string read_run_mode_file();

    // Battery collector thread function
    static void battery_collector_thread();
    static BatteryVoltageData read_battery_voltage();
    static BatteryVoltageData process_voltage_queue();
    static bool check_adc_available();

public:
    // Cached capability set for other API groups (appMgr requires[] gating).
    // Probes on first use; all handlers run on the single mongoose poll
    // thread (see setRunMode notes), so no locking is needed.
    static const json& capabilities()
    {
        if (_capabilities.empty()) {
            _capabilities = probe_capabilities();
        }
        return _capabilities;
    }

    explicit api_device(bool gallery_mode = false)
        : api_base("deviceMgr")
    {
        // P4-D: the persisted run-mode file overrides the -g flag at startup.
        // S93sscma-supervisor always passes -g; when the user selected
        // Node-RED mode we downgrade here (serviced watchdog on, galleryMode
        // reported false) instead of requiring a different launch command.
        // Without -g (legacy manual launches) the flag semantics are kept.
        if (gallery_mode && read_run_mode_file() == "nodered") {
            LOGI("Run-mode file says 'nodered': overriding -g at startup");
            gallery_mode = false;
        }
        _gallery_mode = gallery_mode;
        if (!_gallery_mode) {
            _serviced = std::make_unique<serviced>();
            if (_serviced == nullptr) {
                throw std::runtime_error("Failed to create serviced");
                return;
            }
        } else {
            LOGI("Gallery mode: serviced watchdog disabled");
        }

        _dev_info = parse_result(script(__func__));
        LOGD("%s", _dev_info.dump().c_str());

        try {
            _model_dir = _dev_info["model"]["preset"];
            _model_suffix = _dev_info["model"]["file"];
            _model_suffix = _model_suffix.substr(_model_suffix.find_last_of("."));
        } catch (const json::exception& e) {
            LOGE("Invalid model preset.");
            _model_dir = "";
            _model_suffix = "";
        }

        REG_API(getCameraWebsocketUrl);

        REG_API(getDeviceInfo);
        REG_API_NO_AUTH(getDeviceList);
        REG_API(updateDeviceName);
        REG_API_NO_AUTH(queryDeviceInfo);
        REG_API(getSystemStatus);
        REG_API_NO_AUTH(queryServiceStatus);
        REG_API(setPower);

        REG_API(getAppInfo);
        REG_API(uploadApp);

        REG_API_NO_AUTH(getModelFile);
        REG_API_NO_AUTH(getModelInfo);
        REG_API_NO_AUTH(getModelList);
        REG_API(uploadModel); // security: token required for model upload

        REG_API_NO_AUTH(getPlatformInfo);
        REG_API(savePlatformInfo);

        REG_API(updateChannel);
        REG_API(cancelUpdate);
        REG_API(getSystemUpdateVersion);
        REG_API(getUpdateProgress);
        REG_API(updateSystem);

        REG_API(setTimestamp);
        REG_API(getTimestamp);
        REG_API(setTimezone);
        REG_API(getTimezone);
        REG_API(getTimezoneList);
        REG_API(syncTime);

        REG_API_NO_AUTH(queryBatteryInfo);
        REG_API(getSensorStatus);

        REG_API_NO_AUTH(getCapabilities); // read-only hardware facts, same trust level as queryDeviceInfo

        REG_API_NO_AUTH(getRunMode); // read-only, same trust level as queryDeviceInfo
        REG_API(setRunMode); // security: token required (starts/stops services)
        REG_API(forceConsole); // security: token required (hard mode reset)

        // Check ADC availability before starting battery collector
        _adc_available = check_adc_available();
        if (_adc_available) {
            _battery_collector_running = true;
            std::thread collector(battery_collector_thread);
            _battery_collector_thread = std::move(collector);
            LOGI("Battery collector thread started");
        } else {
            LOGI("Battery collector thread NOT started (ADC unavailable)");
        }
    }

    ~api_device()
    {
        // Stop battery collector thread only if it was started
        if (_adc_available) {
            _battery_collector_running = false;
            if (_battery_collector_thread.joinable()) {
                _battery_cv.notify_all();
                _battery_collector_thread.join();
            }
        }
        LOGV("");
    }

private:
    // json::object() (not the default null json): build_device_type() must
    // stay safe even before the constructor filled it (json::value() throws
    // on null).
    static inline json _dev_info = json::object();
    static inline json _capabilities = json::object(); // P5-A cache, see capabilities()
    static inline bool _gallery_mode = false;
    static inline std::string _model_dir = "";
    static inline std::string _model_suffix = "";
    static inline std::mutex _battery_mutex;
};

#endif // API_DEVICE_H
