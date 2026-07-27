#include "privacy_blur.h"

#include <cstdlib>
#include <fstream>
#include <string>

namespace privacy_blur {

const char* const PRIVACY_BLUR_CONFIG_PATH = "/userdata/local/blur.conf";

namespace {

/* Strip shell-style single quotes, mirroring how the console writes values:
 * KEY='value', with an embedded quote escaped as '\''. Reading unquoted values
 * too is intentional -- the file is meant to survive being hand-edited over
 * SSH, and a missing pair of quotes should not silently disable privacy. */
std::string unquote(const std::string& in)
{
    if (in.size() >= 2 && in.front() == '\'' && in.back() == '\'') {
        std::string body = in.substr(1, in.size() - 2);
        std::string out;
        out.reserve(body.size());
        for (size_t i = 0; i < body.size(); ++i) {
            if (body.compare(i, 4, "'\\''") == 0) {
                out += '\'';
                i += 3;
            } else {
                out += body[i];
            }
        }
        return out;
    }
    return in;
}

std::string trim(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace

bool loadPrivacyBlurConfig(const std::string& path, PrivacyBlurConfig& out,
    std::string* error)
{
    out = PrivacyBlurConfig {};

    std::ifstream f(path);
    if (!f.is_open()) {
        // Absence is the "off" state, not a failure: a device that has never
        // had privacy blur switched on simply has no file.
        return true;
    }
    out.present = true;

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key = trim(line.substr(0, eq));
        const std::string val = unquote(trim(line.substr(eq + 1)));

        if (key == "BLUR_ENABLED") {
            out.enabled = (val == "1" || val == "true" || val == "yes");
        } else if (key == "BLUR_BACKEND") {
            // Only the three known names are accepted. An unrecognised backend
            // leaves the default in place rather than disabling masking,
            // because a typo must not be a way to accidentally publish faces.
            if (val == "mosaic" || val == "coverex" || val == "pixelate") {
                out.backend = val;
            }
        } else if (key == "BLUR_BLOCK_PX") {
            // The hardware privacy-mask unit only implements 8x8 and 16x16
            // grids, so anything else is snapped to 16 -- the coarser of the
            // two, which is the safer thing to guess when the intent is to
            // conceal.
            const long v = strtol(val.c_str(), nullptr, 10);
            out.block_px = (v == 8) ? 8 : 16;
        } else if (key == "BLUR_MAX_REGIONS") {
            const long v = strtol(val.c_str(), nullptr, 10);
            // Clamped, not rejected: the per-backend ceiling is applied later
            // in init(), where the chosen backend is known.
            if (v >= 1 && v <= 8) out.max_regions = static_cast<int>(v);
        } else if (key == "BLUR_BLOCKS_PER_TARGET") {
            char* end = nullptr;
            const long v = strtol(val.c_str(), &end, 10);
            /* 0 disables adaptive sizing. An absurd cap would make the blocks
             * so small that the mask stops concealing, so clamp rather than
             * accept: a hand-edited file must not be able to produce a mask
             * that looks applied and hides nothing. */
            out.blocks_per_target =
                (end != nullptr && *end == '\0' && end != val.c_str() && v >= 0 && v <= 64)
                    ? (int)v
                    : out.blocks_per_target;
        } else if (key == "BLUR_ALPHA") {
            // Anything that is not a number in 0..255 falls back to fully
            // opaque rather than to some middle value, because every other
            // outcome of a malformed line would make the mask more transparent
            // than the person who wrote the file could have intended, and a
            // mask that accidentally shows the face through is the one failure
            // this feature exists to prevent.
            char* end = nullptr;
            const long v = strtol(val.c_str(), &end, 10);
            out.alpha = (end != nullptr && *end == '\0' && end != val.c_str() && v >= 0 && v <= 255)
                ? static_cast<int>(v)
                : 255;
        }
        // Unknown keys are ignored on purpose: this file gains keys over time
        // and an older application binary must not choke on newer ones.
    }

    if (error) error->clear();
    return true;
}

}  // namespace privacy_blur
