#include "onvif_config.h"

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
bool onvif_config::valid_value(const std::string& v)
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

// Shell-style single quoting: wrap in '...', embedded ' becomes '\''. This is
// what makes the file safe to `source` even when a password contains quotes or
// spaces.
std::string onvif_config::quote(const std::string& v)
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

// Inverse of quote(), but deliberately more permissive than quote() is strict.
//
// An unquoted value is accepted verbatim. That is not laxity: the file is
// documented as shell-sourceable KEY=value lines, plain KEY=value is valid
// shell, and the application-side reader in
// components/onvif_meta/src/onvif_meta_config.cpp has always accepted it. When
// this function rejected unquoted lines instead, load() fell back to defaults
// for a hand-written file while the application honoured it -- so the console
// reported ONVIF as switched off while the service was demonstrably listening
// on its port. Two readers of one file disagreeing about its format is a worse
// failure than any value they could disagree about.
//
// Returns false only for something that opens with a quote and does not close
// it properly, which is genuinely mangled rather than merely unquoted.
bool onvif_config::unquote(const std::string& in, std::string& out)
{
    if (in.empty() || in.front() != '\'') {
        out = in;
        return true;
    }
    if (in.size() < 2) {
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

onvif_config::conf onvif_config::load()
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
            LOGW("onvif.conf: malformed value for %s ignored", key.c_str());
            continue;
        }
        if (key == "ONVIF_META_ENABLED") {
            c.meta_enabled = (val == "1" || val == "true");
        } else if (key == "ONVIF_META_INTERVAL_MS") {
            try {
                int v = std::stoi(val);
                if (v >= META_INTERVAL_MIN_MS && v <= META_INTERVAL_MAX_MS) {
                    c.meta_interval_ms = v;
                }
            } catch (const std::exception&) {
                // keep default
            }
        } else if (key == "ONVIF_META_PROFILE") {
            if (!val.empty()) {
                c.meta_profile = val;
            }
        } else if (key == "ONVIF_META_PREFIX") {
            c.meta_prefix = val;
        } else if (key == "ONVIF_SERVICE_ENABLED") {
            c.service_enabled = (val == "1" || val == "true");
        } else if (key == "ONVIF_SERVICE_PORT") {
            try {
                int v = std::stoi(val);
                if (v > SERVICE_PORT_MIN_EXCLUSIVE && v < SERVICE_PORT_MAX_EXCLUSIVE) {
                    c.service_port = v;
                }
            } catch (const std::exception&) {
                // keep default
            }
        } else if (key == "ONVIF_USERNAME") {
            c.username = val;
        } else if (key == "ONVIF_PASSWORD") {
            c.password = val;
        } else if (key == "ONVIF_LOCATION") {
            c.location = val;
        }
        // Unknown keys are ignored on purpose: this file gains keys over time
        // and an older reader must not choke on newer ones.
    }
    return c;
}

// Atomic write: tmp (0600) -> fsync -> rename over CONF_FILE. The file holds
// the ONVIF password, so it is never world-readable, not even transiently, and
// the rename means a reader either sees the whole old file or the whole new
// one -- never a truncated one after a power cut.
bool onvif_config::save(const conf& c)
{
    std::ostringstream ss;
    ss << "# ONVIF integration (written by supervisor)\n";
    ss << "ONVIF_META_ENABLED=" << quote(c.meta_enabled ? "1" : "0") << "\n";
    ss << "ONVIF_META_INTERVAL_MS=" << quote(std::to_string(c.meta_interval_ms)) << "\n";
    ss << "ONVIF_META_PROFILE=" << quote(c.meta_profile) << "\n";
    ss << "ONVIF_META_PREFIX=" << quote(c.meta_prefix) << "\n";
    ss << "ONVIF_SERVICE_ENABLED=" << quote(c.service_enabled ? "1" : "0") << "\n";
    ss << "ONVIF_SERVICE_PORT=" << quote(std::to_string(c.service_port)) << "\n";
    ss << "ONVIF_USERNAME=" << quote(c.username) << "\n";
    ss << "ONVIF_PASSWORD=" << quote(c.password) << "\n";
    ss << "ONVIF_LOCATION=" << quote(c.location) << "\n";
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
