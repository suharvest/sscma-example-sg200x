#ifndef ONVIF_CONFIG_H
#define ONVIF_CONFIG_H

#include <string>

// ONVIF integration configuration (metadata publishing + Device/Media2 SOAP
// services).
//
// Persisted to /userdata/local/onvif.conf as KEY='value' lines, exactly the
// convention ha_config uses (shell-sourceable: values are single-quoted, an
// embedded single quote is written as '\''). The file is chmod 0600 because it
// carries the ONVIF Digest password, and it is written atomically
// (tmp + fsync + rename) so that a power cut can never leave a half-written
// file behind -- an application that reads a truncated conf would come up with
// silently wrong settings.
//
// The authoritative definition of the keys and of their accepted ranges lives
// on the consumer side, in components/onvif_meta/include/onvif_meta_config.h;
// this class is only the writer. Keys: ONVIF_META_ENABLED (0/1),
// ONVIF_META_INTERVAL_MS, ONVIF_META_PROFILE, ONVIF_META_PREFIX,
// ONVIF_SERVICE_ENABLED (0/1), ONVIF_SERVICE_PORT, ONVIF_USERNAME,
// ONVIF_PASSWORD, ONVIF_LOCATION.
class onvif_config {
public:
    struct conf {
        bool meta_enabled = false;
        int meta_interval_ms = 200;
        std::string meta_profile = "live0";
        std::string meta_prefix; // empty -> the reader substitutes the device id
        bool service_enabled = false;
        int service_port = 8000;
        std::string username;
        std::string password;
        std::string location;
    };

    static constexpr const char* CONF_FILE = "/userdata/local/onvif.conf";

    // Bounds mirrored from the parser in onvif_meta_config.cpp. They are
    // enforced (not clamped) by the API so that an out-of-range request is
    // reported instead of being silently rewritten into something else.
    static constexpr int META_INTERVAL_MIN_MS = 20;
    static constexpr int META_INTERVAL_MAX_MS = 60000;
    // Strictly above 1024 because applications do not run as root, and below
    // 65536 because that is the top of the port space.
    static constexpr int SERVICE_PORT_MIN_EXCLUSIVE = 1024;
    static constexpr int SERVICE_PORT_MAX_EXCLUSIVE = 65536;

    // Read CONF_FILE. Missing/unreadable file -> defaults (never an error):
    // a device on which ONVIF was never switched on simply has no file.
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

#endif // ONVIF_CONFIG_H
