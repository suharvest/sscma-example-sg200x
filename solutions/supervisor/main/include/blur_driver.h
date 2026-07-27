#ifndef BLUR_DRIVER_H
#define BLUR_DRIVER_H

#include <string>
#include <vector>

// Deployment of the patched hardware-masking kernel modules.
//
// The privacy blur works on any device: without the patched modules the
// application composites the mask in software, which produces the same picture
// but costs roughly 38 ms of CPU per frame. Replacing cv181x_rgn.ko and
// cv181x_vpss.ko in /mnt/system/ko with the patched builds shipped inside this
// package moves that work into the hardware compositor. It is therefore an
// optional acceleration, never a prerequisite -- which is why nothing here
// runs automatically and why every failure path leaves the device exactly as
// it was.
//
// Two properties of the target make this more delicate than a file copy:
//
//   * / is a read-only ext4. A copy attempted without remounting it read-write
//     simply fails, and a naive script that ignored the return value would
//     report success while the stock modules were still in place. Every step
//     below is checked, and the copied files are re-hashed afterwards so
//     "installed" always means the bytes are really on disk.
//   * A kernel module whose vermagic does not match the running kernel cannot
//     be loaded. Installing one would leave a device that boots without a
//     camera and can only be recovered over a serial console, so the vermagic
//     is verified before anything is written, not after.
class blur_driver {
public:
    // The two modules always move together: the patched rgn module draws the
    // mask, the patched vpss module is what lets it reach the encoder, and a
    // device holding one of each would render nothing.
    static const std::vector<std::string>& modules();

    // Shipped inside the supervisor package; absent in builds made without the
    // patched kernel tree, in which case the feature reports itself
    // unavailable rather than failing at deploy time.
    static constexpr const char* PKG_DIR = "/usr/share/supervisor/ko";
    static constexpr const char* SYS_DIR = "/mnt/system/ko";
    // On /userdata, which is writable and survives a package upgrade, so the
    // stock modules remain recoverable even after the supervisor is replaced.
    static constexpr const char* BACKUP_DIR = "/userdata/ko-backup";
    // Touched by install/restore; its mtime is compared against the boot time
    // to decide whether the pending change has been picked up yet.
    static constexpr const char* PENDING_MARKER = "/userdata/ko-backup/.reboot-pending";

    struct status {
        bool available = false;       // packaged modules present and loadable on this kernel
        bool installed = false;       // the patched modules are the ones in SYS_DIR
        bool backup_present = false;  // the stock modules can be put back
        bool reboot_required = false; // the on-disk modules differ from the loaded ones
        // Empty when available; otherwise a machine-readable cause:
        // "not_packaged" or "vermagic_mismatch".
        std::string reason;
        std::string packaged_vermagic; // "" when not packaged
        std::string kernel_release;    // uname -r of the running kernel
    };

    // Read-only inspection: hashes files and reads vermagic strings, never
    // mounts or writes anything. Safe to call at any time.
    static status probe();

    // Back up the stock modules (once), then replace them with the packaged
    // ones. Refuses to touch the filesystem unless probe() reports available.
    // On failure `err` explains why and nothing has been changed.
    static bool install(std::string& err);

    // Put the backed-up stock modules back. Fails when no backup exists,
    // because overwriting the patched modules with nothing to restore from
    // would be worse than leaving them alone.
    static bool restore(std::string& err);

private:
    static bool file_md5(const std::string& path, std::string& out);
    static bool read_vermagic(const std::string& path, std::string& out);
    static std::string kernel_release();
    // Whole-file copy through a temporary + rename, so a reader never sees a
    // half-written module.
    static bool copy_file(const std::string& src, const std::string& dst, std::string& err);
    // fork + execv (no shell): returns the child's exit status, or -1.
    static int run(const std::vector<std::string>& argv);
    static bool remount_root(bool writable, std::string& err);
    // Copy every module from `src_dir` to `dst_dir` under a read-write root,
    // verifying each result by hash. Shared by install and restore because the
    // two differ only in direction.
    static bool deploy(const std::string& src_dir, const std::string& dst_dir, std::string& err);
    static void mark_reboot_pending();
};

#endif // BLUR_DRIVER_H
