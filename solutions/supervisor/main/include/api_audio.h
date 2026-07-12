#ifndef API_AUDIO_H
#define API_AUDIO_H

#include <cctype>
#include <fstream>
#include <mutex>
#include <string>

#include "api_base.h"

// Audio probe endpoints (microphone / speaker / mixer), P3-E.
//
// Registered under the existing "deviceMgr" group: api_base instances share
// one static API map, so the group name is only a URL prefix. This keeps the
// frontend on a single /api/deviceMgr/* prefix while the audio logic and its
// mutual exclusion stay self-contained in this header.
//
// Hardware (verified on device): card 0 cv182xa_adc = microphone (capture,
// hw:0,0), card 1 cv182xa_dac = speaker (playback, hw:1,0), 16 kHz mono
// S16_LE smoke-tested. Actual arecord/aplay/amixer invocations live in
// main.sh (audioRecord / audioPlayTest / audioVolumeGet / audioVolumeSet).
class api_audio : public api_base {
private:
    // One capture/playback operation at a time: arecord and aplay share the
    // codec path and overlapping invocations must not stack (two clicks on
    // "record" must not start two arecords). Same busy contract as
    // api_app::_op_mutex: try_lock, code -2 when busy. Volume get/set is
    // instant and does not take this lock.
    static inline std::mutex _audio_mutex;

    static constexpr const char* PROBE_WAV = "/tmp/audio_probe.wav";

    // Parameter from URL query or JSON body (numbers must be sent as strings
    // from the frontend, same convention as setTimestamp).
    static std::string getParam(request_t req, const std::string& name) {
        try {
            std::string val = get_param(req, name);
            if (val.empty()) {
                auto params = parse_body(req);
                val         = params.value(name, "");
            }
            return val;
        } catch (const std::exception& e) {
            return "";
        }
    }

    // duration: strictly an integer 1..10 — anything else is rejected
    // (empty, signs, whitespace, floats, hex, "03"-style padding is fine but
    // ">2 chars" and non-digits are not).
    static bool parseDuration(const std::string& s, int& out) {
        if (s.empty() || s.size() > 2)
            return false;
        for (char c : s) {
            if (!std::isdigit(static_cast<unsigned char>(c)))
                return false;
        }
        int v = std::stoi(s);  // safe: 1..2 digits
        if (v < 1 || v > 10)
            return false;
        out = v;
        return true;
    }

    // percent: strictly an integer 0..100.
    static bool parsePercent(const std::string& s, int& out) {
        if (s.empty() || s.size() > 3)
            return false;
        for (char c : s) {
            if (!std::isdigit(static_cast<unsigned char>(c)))
                return false;
        }
        int v = std::stoi(s);  // safe: 1..3 digits
        if (v > 100)
            return false;
        out = v;
        return true;
    }

    // amixer simple-control name: conservative whitelist so the value passes
    // safely through script()'s single-quote wrapping (no quotes, no shell
    // metacharacters).
    static bool validControlName(const std::string& s) {
        if (s.empty() || s.size() > 64)
            return false;
        for (char c : s) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '.' || c == '_' || c == '-'))
                return false;
        }
        return true;
    }

    // GET/POST /api/deviceMgr/audioRecord?duration=N
    // Records N seconds (1..10) of 16 kHz mono S16_LE from hw:0,0 into
    // /tmp/audio_probe.wav (main.sh wraps arecord in _app_run_timeout) and
    // streams the WAV back via API_STATUS_REPLY_FILE (mongoose's mime table
    // maps .wav to audio/wav). The probe file may be overwritten by the next
    // record request; single-operator usage is assumed.
    static api_status_t audioRecord(request_t req, response_t res) {
        int secs = 0;
        if (!parseDuration(getParam(req, "duration"), secs)) {
            response(res, -1, "Invalid duration: integer 1..10 required");
            return API_STATUS_OK;
        }

        if (!_audio_mutex.try_lock()) {
            response(res, -2, "busy: another audio operation is in progress");
            return API_STATUS_OK;
        }
        std::lock_guard<std::mutex> lk(_audio_mutex, std::adopt_lock);

        std::string result = script(__func__, secs);
        if (result != STR_OK) {
            response(res, -1, "Recording failed");
            return API_STATUS_OK;
        }
        std::ifstream f(PROBE_WAV, std::ios::binary);
        if (!f.is_open()) {
            response(res, -1, "Recording produced no output");
            return API_STATUS_OK;
        }
        response(res, 0, STR_OK, { { "file", PROBE_WAV } });
        return API_STATUS_REPLY_FILE;
    }

    // POST /api/deviceMgr/audioPlayTest
    // Plays the packaged test tone (/usr/share/supervisor/sounds/
    // test_tone.wav) on hw:1,0. The user confirms audibility at the device.
    static api_status_t audioPlayTest(request_t req, response_t res) {
        if (!_audio_mutex.try_lock()) {
            response(res, -2, "busy: another audio operation is in progress");
            return API_STATUS_OK;
        }
        std::lock_guard<std::mutex> lk(_audio_mutex, std::adopt_lock);

        std::string result = script(__func__);
        if (result != STR_OK) {
            response(res, -1, "Playback failed");
            return API_STATUS_OK;
        }
        response(res, 0, STR_OK);
        return API_STATUS_OK;
    }

    // GET  /api/deviceMgr/audioVolume
    //        -> { supported: bool, controls: [{name, percent}] }
    // POST /api/deviceMgr/audioVolume  {control, percent}
    //        -> set one simple control (percent "0".."100", sent as string)
    // Both paths are backed by defensive amixer parsing in main.sh; when
    // amixer is missing or exposes no volume control, get reports
    // supported=false and set fails cleanly.
    static api_status_t audioVolume(request_t req, response_t res) {
        std::string percentStr = getParam(req, "percent");
        if (percentStr.empty()) {
            // get path
            json out = parse_result(script("audioVolumeGet"));
            if (!out.is_object() || !out.contains("supported")) {
                out = { { "supported", false }, { "controls", json::array() } };
            }
            response(res, 0, STR_OK, out);
            return API_STATUS_OK;
        }

        // set path
        int pct = 0;
        if (!parsePercent(percentStr, pct)) {
            response(res, -1, "Invalid percent: integer 0..100 required");
            return API_STATUS_OK;
        }
        std::string control = getParam(req, "control");
        if (!validControlName(control)) {
            response(res, -1, "Invalid control name");
            return API_STATUS_OK;
        }

        std::string result = script("audioVolumeSet", control, pct);
        if (result != STR_OK) {
            response(res, -1, "Failed to set volume (control not found or mixer unavailable)");
            return API_STATUS_OK;
        }
        response(res, 0, STR_OK, { { "control", control }, { "percent", pct } });
        return API_STATUS_OK;
    }

public:
    api_audio() : api_base("deviceMgr") {
        REG_API(audioRecord);
        REG_API(audioPlayTest);
        REG_API(audioVolume);
    }

    ~api_audio() = default;
};

#endif  // API_AUDIO_H
