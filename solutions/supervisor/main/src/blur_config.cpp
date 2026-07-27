#include "blur_config.h"

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

bool blur_config::valid_backend(const std::string& v)
{
    return v == "mosaic" || v == "coverex" || v == "pixelate";
}

bool blur_config::valid_block_px(int v)
{
    return v == BLOCK_PX_SMALL || v == BLOCK_PX_LARGE;
}

bool blur_config::valid_max_regions(int v)
{
    return v >= MAX_REGIONS_MIN && v <= MAX_REGIONS_MAX;
}

bool blur_config::valid_alpha(int v)
{
    return v >= ALPHA_MIN && v <= ALPHA_MAX;
}

// Shell-style single quoting: wrap in '...', embedded ' becomes '\''. None of
// the current values can contain a quote, but writing them the same way as the
// other conf files keeps one single format for everything that is sourced.
std::string blur_config::quote(const std::string& v)
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
// shell, and the application side reads it with `source`, which accepts both.
// The ONVIF conf hit exactly this: while its reader rejected unquoted lines,
// load() fell back to defaults for a hand-written file that the application
// honoured, so the console reported the feature as switched off while the
// service was demonstrably running. Two readers of one file disagreeing about
// its format is a worse failure than any value they could disagree about --
// and for privacy blur the same bug would tell a user their faces are masked
// when they are not.
//
// Returns false only for something that opens with a quote and does not close
// it properly, which is genuinely mangled rather than merely unquoted.
bool blur_config::unquote(const std::string& in, std::string& out)
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

blur_config::conf blur_config::load()
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
            LOGW("blur.conf: malformed value for %s ignored", key.c_str());
            continue;
        }
        if (key == "BLUR_ENABLED") {
            c.enabled = (val == "1" || val == "true");
        } else if (key == "BLUR_BACKEND") {
            // An unrecognised backend keeps the default rather than being
            // propagated: the API is the only thing that should ever be able
            // to produce a value the applications cannot render.
            if (valid_backend(val)) {
                c.backend = val;
            }
        } else if (key == "BLUR_BLOCK_PX") {
            try {
                int v = std::stoi(val);
                if (valid_block_px(v)) {
                    c.block_px = v;
                }
            } catch (const std::exception&) {
                // keep default
            }
        } else if (key == "BLUR_MAX_REGIONS") {
            try {
                int v = std::stoi(val);
                if (valid_max_regions(v)) {
                    c.max_regions = v;
                }
            } catch (const std::exception&) {
                // keep default
            }
        } else if (key == "BLUR_ALPHA") {
            try {
                int v = std::stoi(val);
                if (valid_alpha(v)) {
                    c.alpha = v;
                }
            } catch (const std::exception&) {
                // An unparseable alpha falls back to the opaque default rather
                // than to anything see-through, so a mangled file can never
                // quietly make the mask transparent.
            }
        }
        // Unknown keys are ignored on purpose: this file gains keys over time
        // and an older reader must not choke on newer ones.
    }
    return c;
}

// Atomic write: tmp -> fsync -> rename over CONF_FILE. The rename means a
// reader either sees the whole old file or the whole new one, never a
// truncated one after a power cut. Mode is 0644 (unlike onvif.conf's 0600)
// because nothing here is secret and the camera applications that read it do
// not run as root.
bool blur_config::save(const conf& c)
{
    std::ostringstream ss;
    ss << "# Privacy blur (written by supervisor)\n";
    ss << "BLUR_ENABLED=" << quote(c.enabled ? "1" : "0") << "\n";
    ss << "BLUR_BACKEND=" << quote(c.backend) << "\n";
    ss << "BLUR_BLOCK_PX=" << quote(std::to_string(c.block_px)) << "\n";
    ss << "BLUR_MAX_REGIONS=" << quote(std::to_string(c.max_regions)) << "\n";
    ss << "BLUR_ALPHA=" << quote(std::to_string(c.alpha)) << "\n";
    const std::string data = ss.str();

    std::error_code ec;
    fs::create_directories(fs::path(CONF_FILE).parent_path(), ec);

    const std::string tmp = std::string(CONF_FILE) + ".tmp";
    ::unlink(tmp.c_str()); // ensure O_CREAT applies our mode
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", tmp.c_str(), strerror(errno));
        return false;
    }
    ::fchmod(fd, 0644); // belt and braces vs umask surprises
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
