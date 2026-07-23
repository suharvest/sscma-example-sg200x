#ifndef HA_CONFIG_H
#define HA_CONFIG_H

#include <string>

// Home Assistant MQTT integration configuration.
//
// Persisted to /userdata/local/ha.conf as KEY='value' lines (shell-sourceable:
// values are single-quoted, an embedded single quote is written as '\''). The
// file is chmod 0600 (it may contain the broker password) and written
// atomically (tmp + fsync + rename, same pattern as api_app state.json).
//
// Keys: HA_ENABLED (0/1), HA_BROKER_HOST, HA_BROKER_PORT, HA_USERNAME,
// HA_PASSWORD, HA_DISCOVERY_PREFIX.
class ha_config {
public:
    struct conf {
        bool enabled = false;
        std::string broker_host;
        int broker_port = 1883;
        std::string username;
        std::string password;
        std::string discovery_prefix = "homeassistant";
    };

    static constexpr const char* CONF_FILE = "/userdata/local/ha.conf";

    // Read CONF_FILE. Missing/unreadable file -> defaults (never an error).
    static conf load();

    // Atomic write of CONF_FILE (0600). Returns false on I/O failure.
    static bool save(const conf& c);

    // A value is storable if it contains no NUL/CR/LF (they would break the
    // line-oriented KEY='value' format). Max 256 bytes.
    static bool valid_value(const std::string& v);

private:
    static std::string quote(const std::string& v);
    static bool unquote(const std::string& in, std::string& out);
};

#endif // HA_CONFIG_H
