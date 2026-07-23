#include "ha_config.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "logger.hpp"

namespace fs = std::filesystem;

// Reject NUL/CR/LF: they would break the line-oriented KEY='value' format
// (and a NUL would truncate the C string on the consumer side).
bool ha_config::valid_value(const std::string& v)
{
    if (v.size() > 256) {
        return false;
    }
    for (char c : v) {
        if (c == '\0' || c == '\r' || c == '\n') {
            return false;
        }
    }
    return true;
}

// Shell-style single quoting: wrap in '...', embedded ' becomes '\''.
std::string ha_config::quote(const std::string& v)
{
    std::string out = "'";
    for (char c : v) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Inverse of quote(). Accepts only the exact format quote() emits:
// leading ', trailing ', embedded quotes as '\''. Returns false on anything
// else (the value is then ignored and the default kept).
bool ha_config::unquote(const std::string& in, std::string& out)
{
    if (in.size() < 2 || in.front() != '\'') {
        return false;
    }
    out.clear();
    size_t i = 1;
    while (i < in.size()) {
        if (in[i] == '\'') {
            if (i == in.size() - 1) {
                return true; // closing quote, nothing may follow
            }
            // must be the 4-char '\'' escape encoding one single quote
            if (in.compare(i, 4, "'\\''") == 0) {
                out += '\'';
                i += 4;
                continue;
            }
            return false;
        }
        out += in[i];
        ++i;
    }
    return false; // no closing quote
}

ha_config::conf ha_config::load()
{
    conf c;
    std::ifstream f(CONF_FILE);
    if (!f.is_open()) {
        return c;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, eq);
        std::string val;
        if (!unquote(line.substr(eq + 1), val)) {
            LOGW("ha.conf: malformed value for %s ignored", key.c_str());
            continue;
        }
        if (key == "HA_ENABLED") {
            c.enabled = (val == "1" || val == "true");
        } else if (key == "HA_BROKER_HOST") {
            c.broker_host = val;
        } else if (key == "HA_BROKER_PORT") {
            try {
                int p = std::stoi(val);
                if (p >= 1 && p <= 65535) {
                    c.broker_port = p;
                }
            } catch (const std::exception&) {
                // keep default
            }
        } else if (key == "HA_USERNAME") {
            c.username = val;
        } else if (key == "HA_PASSWORD") {
            c.password = val;
        } else if (key == "HA_DISCOVERY_PREFIX") {
            if (!val.empty()) {
                c.discovery_prefix = val;
            }
        }
    }
    return c;
}

// Atomic write: tmp (0600) -> fsync -> rename over CONF_FILE. The file holds
// the broker password, so it is never world-readable, not even transiently.
bool ha_config::save(const conf& c)
{
    std::ostringstream ss;
    ss << "# Home Assistant MQTT integration (written by supervisor)\n";
    ss << "HA_ENABLED=" << quote(c.enabled ? "1" : "0") << "\n";
    ss << "HA_BROKER_HOST=" << quote(c.broker_host) << "\n";
    ss << "HA_BROKER_PORT=" << quote(std::to_string(c.broker_port)) << "\n";
    ss << "HA_USERNAME=" << quote(c.username) << "\n";
    ss << "HA_PASSWORD=" << quote(c.password) << "\n";
    ss << "HA_DISCOVERY_PREFIX=" << quote(c.discovery_prefix) << "\n";
    const std::string data = ss.str();

    std::error_code ec;
    fs::create_directories(fs::path(CONF_FILE).parent_path(), ec);

    const std::string tmp = std::string(CONF_FILE) + ".tmp";
    ::unlink(tmp.c_str()); // ensure O_CREAT applies our 0600 mode
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
    ::fchmod(fd, 0600); // belt and braces vs umask surprises
    ssize_t n = ::write(fd, data.data(), data.size());
    ::fsync(fd);
    ::close(fd);
    if (n != (ssize_t)data.size()) {
        LOGE("short write to %s", tmp.c_str());
        ::unlink(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), CONF_FILE) != 0) {
        LOGE("rename(%s -> %s) failed: %s", tmp.c_str(), CONF_FILE, strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}
