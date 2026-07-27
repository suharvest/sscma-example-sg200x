#ifndef BLUR_CONFIG_H
#define BLUR_CONFIG_H

#include <string>

// Privacy blur configuration (on-device masking of detected subjects).
//
// Persisted to /userdata/local/blur.conf as shell-sourceable KEY='value'
// lines, the same convention ha_config and onvif_config use. The file is
// written atomically (tmp + fsync + rename) so a power cut can never leave a
// half-written file behind: an application that read a truncated conf would
// come up with silently wrong masking settings, which for a privacy feature
// means faces going unmasked.
//
// The authoritative definition of the keys lives on the consumer side (the
// camera applications read this file directly); this class is only the writer
// plus the console's reader. Keys: BLUR_ENABLED (0/1), BLUR_BACKEND
// (mosaic|coverex|pixelate), BLUR_BLOCK_PX (8 or 16), BLUR_MAX_REGIONS (1-8),
// BLUR_ALPHA (0-255).
class blur_config {
public:
    struct conf {
        bool enabled = false;
        // "pixelate" is the default because it is the only backend that both
        // looks like real pixelation and costs nothing at runtime; the other
        // two exist for hardware/compatibility fallbacks.
        std::string backend = "pixelate";
        int block_px = 16;
        int max_regions = 8;
        // Fully opaque by default. Anything below 255 lets the original
        // picture show through the mask, and whatever shows through is
        // recognisable again -- so the safe value is the one that hides
        // everything, and lowering it is a deliberate act by the user.
        int alpha = 255;
    };

    static constexpr const char* CONF_FILE = "/userdata/local/blur.conf";

    // The hardware mosaic grid is built around these two cell sizes only, so
    // anything else would be silently rounded by the driver.
    static constexpr int BLOCK_PX_SMALL = 8;
    static constexpr int BLOCK_PX_LARGE = 16;
    // Region count ceiling imposed by the number of overlay handles the
    // hardware compositor can bind to one video channel.
    static constexpr int MAX_REGIONS_MIN = 1;
    static constexpr int MAX_REGIONS_MAX = 8;
    // 8-bit per-pixel alpha as the hardware overlay expresses it: 0 is an
    // invisible mask, 255 an opaque one.
    static constexpr int ALPHA_MIN = 0;
    static constexpr int ALPHA_MAX = 255;

    static bool valid_backend(const std::string& v);
    static bool valid_block_px(int v);
    static bool valid_max_regions(int v);
    static bool valid_alpha(int v);

    // Read CONF_FILE. Missing/unreadable file -> defaults (never an error):
    // a device on which blur was never switched on simply has no file.
    static conf load();

    // Atomic write of CONF_FILE. Returns false on I/O failure.
    static bool save(const conf& c);

private:
    static std::string quote(const std::string& v);
    static bool unquote(const std::string& in, std::string& out);
};

#endif // BLUR_CONFIG_H
