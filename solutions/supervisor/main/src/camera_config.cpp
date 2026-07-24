#include "camera_config.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "logger.hpp"

namespace fs = std::filesystem;

camera_config::conf camera_config::load()
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
        std::string val = line.substr(eq + 1);
        // Tolerate quoted values ('1' / "1") from hand-edited files.
        if (val.size() >= 2 && (val.front() == '\'' || val.front() == '"') && val.back() == val.front()) {
            val = val.substr(1, val.size() - 2);
        }
        if (key == "CAM_MIRROR") {
            c.mirror = (val == "1" || val == "true");
        } else if (key == "CAM_FLIP") {
            c.flip = (val == "1" || val == "true");
        } else if (key == "CAM_ROTATION") {
            // Only 0/180 are supported; anything else keeps the default.
            if (val == "180") {
                c.rotation = 180;
            } else if (val == "0") {
                c.rotation = 0;
            } else {
                LOGW("camera.conf: unsupported CAM_ROTATION '%s' ignored", val.c_str());
            }
        }
    }
    return c;
}

// Atomic write: tmp -> fsync -> rename over CONF_FILE. No secrets in this
// file, so it is world-readable (0644) — the consumer app may run unprivileged.
bool camera_config::save(const conf& c)
{
    std::ostringstream ss;
    ss << "# Camera picture orientation (written by supervisor)\n";
    ss << "CAM_MIRROR=" << (c.mirror ? "1" : "0") << "\n";
    ss << "CAM_FLIP=" << (c.flip ? "1" : "0") << "\n";
    ss << "CAM_ROTATION=" << (c.rotation == 180 ? "180" : "0") << "\n";
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
