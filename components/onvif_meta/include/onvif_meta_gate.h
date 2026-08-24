#ifndef _ONVIF_META_GATE_H_
#define _ONVIF_META_GATE_H_

#include <cstdint>
#include <string>

#include "onvif_meta_config.h"

/*
 * The switch and the rate limit, in one object, so no application
 * re-implements either.
 *
 * Usage in an inference loop:
 *
 *     if (g_onvif.enabled()) {                    // RTSP at inference cadence
 *         onvif_frame_t f;
 *         ... fill from the boxes already built for the console ...
 *         const std::string xml = onvif_meta_to_xml(f);
 *         rtsp_server_write_metadata(xml.data(), xml.size());
 *         if (g_onvif.take(now_ms))               // MQTT is rate limited
 *             mqtt->publishText(g_onvif.topic(), onvif_meta_to_json(f));
 *     }
 *
 * Nothing here touches a transport: the caller owns MQTT and RTSP. Keeping the
 * gate transport-neutral lets both serialisers share one data model.
 */
class OnvifMetaGate {
public:
    /*
     * Read /userdata/local/onvif.conf (or `path`) and compute the topic.
     * `device_id` is the fallback topic prefix when the config leaves it
     * empty -- pass readDeviceIdentifier(), which is the serial number.
     * `module` is the analytics module name, conventionally the app id.
     *
     * Safe to call again to pick up a configuration change; supervisor
     * restarts the application after writing, so in practice this runs once at
     * startup.
     */
    void reload(const std::string& device_id, const std::string& module,
                const std::string& path = ONVIF_META_CONFIG_PATH);

    /* Off unless the config says otherwise. */
    bool enabled() const { return cfg_.enabled; }

    /*
     * True when publishing is enabled and the rate limit allows another one
     * now; claims the slot, so call it once per frame and publish if it
     * returns true.
     *
     * A clock that moves backwards (this device boots at 1970 and gets stepped
     * by NTP or a first ONVIF client) is treated as "due" rather than blocking
     * publishes until the old timestamp is passed again.
     */
    bool take(uint64_t now_ms);

    const std::string& topic() const { return topic_; }
    const OnvifMetaConfig& config() const { return cfg_; }

private:
    OnvifMetaConfig cfg_;
    std::string topic_;
    uint64_t last_ms_ = 0;
};

#endif /* _ONVIF_META_GATE_H_ */
