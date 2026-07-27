#include "blur_driver.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <openssl/md5.h>

#include "logger.hpp"

namespace fs = std::filesystem;

const std::vector<std::string>& blur_driver::modules()
{
    static const std::vector<std::string> m = { "cv181x_rgn.ko", "cv181x_vpss.ko" };
    return m;
}

namespace {

/*
 * A string that appears in the patched cv181x_vpss and in no stock build: the
 * module parameter the mask patch adds. It is the discriminator for "is this
 * module patched?", asked two ways -- of a file on disk (does the image contain
 * the name?) and of the running kernel (did the module export the parameter?).
 *
 * If the patch ever renames this parameter, driver state silently reports the
 * patched module as stock. That is the cost of using a feature of the artefact
 * as its own identity; the alternative, srcversion, is not built into these
 * modules (checked on the device: /sys/module/cv181x_vpss/srcversion does not
 * exist and the .ko carries no srcversion string).
 */
constexpr const char* PATCH_MARKER = "mask_force_alpha";
constexpr const char* PATCH_MARKER_SYSFS = "/sys/module/cv181x_vpss/parameters/mask_force_alpha";
constexpr const char* LOADED_VPSS_MODULE = "cv181x_vpss.ko";

/* Does this module image contain the marker? Substring search over the file --
 * the modules are under a megabyte and this runs only when the console asks for
 * driver state. */
bool image_is_patched(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    const std::string marker(PATCH_MARKER);
    std::string window;
    std::vector<char> buf(64 * 1024);
    while (f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        const std::streamsize n = f.gcount();
        if (n <= 0) {
            break;
        }
        window.append(buf.data(), (size_t)n);
        if (window.find(marker) != std::string::npos) {
            return true;
        }
        /* Keep just enough tail that a marker split across two reads is still
         * found, and do not grow the buffer without bound. */
        if (window.size() > marker.size()) {
            window.erase(0, window.size() - (marker.size() - 1));
        }
    }
    return false;
}

}  // namespace

bool blur_driver::file_md5(const std::string& path, std::string& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    MD5_CTX ctx;
    MD5_Init(&ctx);
    std::vector<char> buf(64 * 1024);
    while (f) {
        f.read(buf.data(), buf.size());
        std::streamsize n = f.gcount();
        if (n > 0) {
            MD5_Update(&ctx, buf.data(), (size_t)n);
        }
    }
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5_Final(digest, &ctx);
    char hex[2 * MD5_DIGEST_LENGTH + 1];
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    }
    out.assign(hex, 2 * MD5_DIGEST_LENGTH);
    return true;
}

// The vermagic lives in the .modinfo section as a NUL-terminated
// "vermagic=<release> <flags>" entry. Scanning the raw file for that literal
// avoids depending on modinfo(8), which busybox does not provide, and on any
// ELF parsing that would have to track section headers for a foreign
// architecture.
bool blur_driver::read_vermagic(const std::string& path, std::string& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    static const std::string key = "vermagic=";
    size_t pos = data.find(key);
    if (pos == std::string::npos) {
        return false;
    }
    size_t start = pos + key.size();
    size_t end = data.find('\0', start);
    if (end == std::string::npos) {
        end = data.size();
    }
    out = data.substr(start, end - start);
    return !out.empty();
}

std::string blur_driver::kernel_release()
{
    struct utsname u;
    if (uname(&u) != 0) {
        return "";
    }
    return std::string(u.release);
}

int blur_driver::run(const std::vector<std::string>& argv)
{
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& s : argv) {
        cargv.push_back(const_cast<char*>(s.c_str()));
    }
    cargv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        LOGE("fork() failed: %s", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        execv(cargv[0], cargv.data());
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

bool blur_driver::remount_root(bool writable, std::string& err)
{
    const std::string opt = writable ? "remount,rw" : "remount,ro";
    int rc = run({ "/bin/mount", "-o", opt, "/" });
    if (rc != 0) {
        err = "failed to remount / " + std::string(writable ? "read-write" : "read-only")
            + " (mount exit " + std::to_string(rc) + ")";
        return false;
    }
    return true;
}

bool blur_driver::copy_file(const std::string& src, const std::string& dst, std::string& err)
{
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
        err = "cannot read " + src;
        return false;
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.empty()) {
        err = src + " is empty";
        return false;
    }

    const std::string tmp = dst + ".tmp";
    ::unlink(tmp.c_str());
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        err = "cannot create " + tmp + ": " + strerror(errno);
        return false;
    }
    ssize_t n = ::write(fd, data.data(), data.size());
    // fsync before the rename: the point of this whole operation is to survive
    // the reboot that follows it, and an unflushed module would come back as a
    // zero-length file the kernel refuses to load.
    ::fsync(fd);
    ::close(fd);
    if (n != (ssize_t)data.size()) {
        err = "short write to " + tmp;
        ::unlink(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), dst.c_str()) != 0) {
        err = "cannot replace " + dst + ": " + strerror(errno);
        ::unlink(tmp.c_str());
        return false;
    }
    // Flush the directory entry too, otherwise the rename itself can be lost.
    int dfd = ::open(fs::path(dst).parent_path().c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
    return true;
}

blur_driver::status blur_driver::probe()
{
    status st;
    st.kernel_release = kernel_release();

    bool all_packaged = true;
    bool all_match = true;
    bool all_installed = true;
    bool all_backed_up = true;

    for (const auto& m : modules()) {
        const std::string pkg = std::string(PKG_DIR) + "/" + m;
        const std::string sys = std::string(SYS_DIR) + "/" + m;
        const std::string bak = std::string(BACKUP_DIR) + "/" + m;

        std::error_code ec;
        if (!fs::exists(pkg, ec)) {
            all_packaged = false;
            all_installed = false;
            continue;
        }
        std::string vermagic;
        if (!read_vermagic(pkg, vermagic)) {
            all_match = false;
        } else {
            if (st.packaged_vermagic.empty()) {
                st.packaged_vermagic = vermagic;
            }
            // The vermagic begins with the kernel release and continues with
            // build flags; matching the release prefix on a token boundary is
            // what insmod effectively requires and avoids rejecting a module
            // over an irrelevant flag ordering difference.
            if (vermagic.compare(0, st.kernel_release.size(), st.kernel_release) != 0
                || (vermagic.size() > st.kernel_release.size()
                    && vermagic[st.kernel_release.size()] != ' ')) {
                all_match = false;
            }
        }

        std::string pkg_md5, sys_md5;
        if (!file_md5(pkg, pkg_md5) || !file_md5(sys, sys_md5) || pkg_md5 != sys_md5) {
            all_installed = false;
        }
        if (!fs::exists(bak, ec)) {
            all_backed_up = false;
        }
    }

    st.available = all_packaged && all_match && !st.kernel_release.empty();
    st.installed = all_packaged && all_installed;
    st.backup_present = all_backed_up;
    if (!all_packaged) {
        st.reason = "not_packaged";
    } else if (!all_match) {
        st.reason = "vermagic_mismatch";
    }

    /*
     * Does the module on disk differ from the one the kernel is running?
     *
     * Asked of the artefacts themselves -- the marker in the image on disk
     * against the same marker in /sys/module -- and not of any clock.
     *
     * This used to compare a marker file's mtime against boot time, on the
     * reasoning that the flag would then clear itself after a reboot. That
     * reasoning assumed a monotonic wall clock. This device has no RTC battery:
     * every boot restarts wall time at epoch 0, so a marker written late in the
     * previous session (mtime 16606) always looks newer than this session's
     * boot (time 230 - uptime 230 = 0). The check therefore reported "reboot
     * required" forever, and the longer the previous session had run the more
     * certain it looked. Observed on the device after a restore that had in
     * fact already taken effect.
     */
    const std::string live_vpss = std::string(SYS_DIR) + "/" + LOADED_VPSS_MODULE;
    const bool disk_patched   = image_is_patched(live_vpss);
    std::error_code ec2;
    const bool loaded_patched = fs::exists(PATCH_MARKER_SYSFS, ec2);
    st.reboot_required = disk_patched != loaded_patched;
    return st;
}

bool blur_driver::deploy(const std::string& src_dir, const std::string& dst_dir, std::string& err)
{
    if (!remount_root(true, err)) {
        return false;
    }

    bool ok = true;
    for (const auto& m : modules()) {
        const std::string src = src_dir + "/" + m;
        const std::string dst = dst_dir + "/" + m;
        if (!copy_file(src, dst, err)) {
            ok = false;
            break;
        }
        ::sync();
        // Re-hash instead of trusting the copy: a read-only filesystem, a full
        // one, or a partially flushed write can all make the copy look like it
        // succeeded, and reporting a deployment that did not happen is the one
        // outcome this whole endpoint exists to prevent.
        std::string src_md5, dst_md5;
        if (!file_md5(src, src_md5) || !file_md5(dst, dst_md5) || src_md5 != dst_md5) {
            err = "verification failed for " + m + " (the file on disk does not match the source)";
            ok = false;
            break;
        }
        LOGI("blur driver: %s -> %s verified (md5 %s)", src.c_str(), dst.c_str(), dst_md5.c_str());
    }

    // Always restore the read-only mount, including on the failure paths: a
    // root left writable would silently drop the protection the device relies
    // on for the rest of its uptime.
    std::string ro_err;
    if (!remount_root(false, ro_err)) {
        LOGE("blur driver: %s", ro_err.c_str());
        if (ok) {
            err = ro_err;
            ok = false;
        }
    }
    return ok;
}

bool blur_driver::install(std::string& err)
{
    status st = probe();
    if (!st.available) {
        if (st.reason == "vermagic_mismatch") {
            err = "the packaged modules were built for kernel '" + st.packaged_vermagic
                + "' but this device runs '" + st.kernel_release
                + "'; installing them would leave the camera unusable after the next reboot";
        } else {
            err = "no patched modules are shipped with this supervisor build";
        }
        return false;
    }

    // Back up first, and only what is not backed up yet. A second install must
    // not overwrite the backup, because by then /mnt/system/ko holds the
    // patched modules and capturing them as "the stock ones" would destroy the
    // only copy of the originals -- restore would then be unable to undo
    // anything, permanently.
    std::error_code ec;
    fs::create_directories(BACKUP_DIR, ec);
    if (ec) {
        err = "cannot create " + std::string(BACKUP_DIR) + ": " + ec.message();
        return false;
    }
    for (const auto& m : modules()) {
        const std::string sys = std::string(SYS_DIR) + "/" + m;
        const std::string bak = std::string(BACKUP_DIR) + "/" + m;
        if (fs::exists(bak, ec)) {
            LOGI("blur driver: backup of %s already present, kept", m.c_str());
            continue;
        }
        if (!fs::exists(sys, ec)) {
            err = "missing " + sys + "; refusing to install without a stock module to fall back to";
            return false;
        }
        if (!copy_file(sys, bak, err)) {
            err = "backup failed: " + err;
            return false;
        }
        std::string a, b;
        if (!file_md5(sys, a) || !file_md5(bak, b) || a != b) {
            err = "backup verification failed for " + m;
            return false;
        }
        LOGI("blur driver: backed up %s (md5 %s)", m.c_str(), b.c_str());
    }

    if (!deploy(PKG_DIR, SYS_DIR, err)) {
        return false;
    }
    /* No "reboot pending" flag to write: probe() answers that by comparing the
     * module on disk with the one the kernel is running, which cannot go stale
     * and needs no clock. */
    return true;
}

bool blur_driver::restore(std::string& err)
{
    std::error_code ec;
    for (const auto& m : modules()) {
        if (!fs::exists(std::string(BACKUP_DIR) + "/" + m, ec)) {
            err = "no backup of the stock modules exists on this device; nothing to restore";
            return false;
        }
    }
    if (!deploy(BACKUP_DIR, SYS_DIR, err)) {
        return false;
    }
    return true;
}
