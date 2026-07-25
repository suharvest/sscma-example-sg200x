#include "onvif_meta_config.h"
#include "onvif_meta_gate.h"
#include "onvif_meta.h"

#include <cstdlib>
#include <fstream>
#include <string>

const char* const ONVIF_META_CONFIG_PATH = "/userdata/local/onvif.conf";

namespace {

/* Strip shell-style single quotes, mirroring how ha_config writes values:
 * KEY='value', with an embedded quote escaped as '\''. */
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

bool loadOnvifMetaConfig(const std::string& path, OnvifMetaConfig& out,
    std::string* error)
{
    out = OnvifMetaConfig {};

    std::ifstream f(path);
    if (!f.is_open()) {
        // Absence is the "off" state, not a failure: a device that has never
        // had ONVIF switched on simply has no file.
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

        if (key == "ONVIF_META_ENABLED") {
            out.enabled = (val == "1" || val == "true" || val == "yes");
        } else if (key == "ONVIF_META_INTERVAL_MS") {
            const long v = strtol(val.c_str(), nullptr, 10);
            // Clamp rather than reject: a hand-edited 0 should not turn this
            // into a per-frame firehose on a shared MQTT broker.
            if (v >= 20 && v <= 60000) out.interval_ms = static_cast<uint32_t>(v);
        } else if (key == "ONVIF_META_PROFILE") {
            if (!val.empty()) out.profile = val;
        } else if (key == "ONVIF_META_PREFIX") {
            out.topic_prefix = val;
        }
        // Unknown keys are ignored on purpose: the ONVIF service settings will
        // land in this same file later, and an older application binary must
        // not choke on them.
    }

    if (error) error->clear();
    return true;
}

// ---------------------------------------------------------------------------

void OnvifMetaGate::reload(const std::string& device_id, const std::string& module,
    const std::string& path)
{
    loadOnvifMetaConfig(path, cfg_, nullptr);
    const std::string prefix = cfg_.topic_prefix.empty() ? device_id : cfg_.topic_prefix;
    topic_ = onvif_meta_topic(prefix, cfg_.profile, module);
    last_ms_ = 0;
}

bool OnvifMetaGate::take(uint64_t now_ms)
{
    if (!cfg_.enabled) return false;
    if (last_ms_ != 0 && now_ms >= last_ms_ &&
        (now_ms - last_ms_) < cfg_.interval_ms) {
        return false;
    }
    last_ms_ = now_ms;
    return true;
}

// ---------------------------------------------------------------------------

void onvif_meta_add_box(onvif_frame_t& frame, int id,
    float cx, float cy, float w, float h,
    const std::string& type, float likelihood)
{
    onvif_object_t o;
    o.id = id;
    o.cx = cx;
    o.cy = cy;
    o.w = w;
    o.h = h;
    if (!type.empty()) {
        o.classes.push_back(onvif_class_t { type, likelihood });
    }
    frame.objects.push_back(std::move(o));
}

onvif_frame_t onvif_meta_from_boxes(uint64_t utc_ms, const std::string& source,
    int frame_w, int frame_h, const std::vector<onvif_box_t>& boxes,
    std::string (*classify)(const std::string&))
{
    onvif_frame_t f;
    f.utc_ms = utc_ms;
    f.source = source;
    f.frame_w = frame_w;
    f.frame_h = frame_h;
    f.objects.reserve(boxes.size());
    int id = 1;
    for (const onvif_box_t& b : boxes) {
        // Ids are positional here. A tracker-backed application should fill
        // onvif_object_t directly so ONVIF object ids stay stable across
        // frames, which is what makes a VMS able to follow one object.
        onvif_meta_add_box(f, id++, b.x, b.y, b.w, b.h,
            classify ? classify(b.label) : b.label, b.score);
    }
    return f;
}
