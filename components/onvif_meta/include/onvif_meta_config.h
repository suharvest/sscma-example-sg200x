#ifndef _ONVIF_META_CONFIG_H_
#define _ONVIF_META_CONFIG_H_

#include <cstdint>
#include <string>

/*
 * Device-wide ONVIF settings, written by supervisor and read in-process by
 * whichever application is running. Same mechanism as /userdata/local/ha.conf:
 * one switch in the console, no per-application configuration, no init-script
 * or command-line plumbing to touch.
 *
 * File: /userdata/local/onvif.conf, shell-sourceable KEY='value' lines.
 *
 *   ONVIF_META_ENABLED=0|1        publish analytics metadata over MQTT
 *   ONVIF_META_INTERVAL_MS=200    minimum gap between metadata publishes
 *   ONVIF_META_PROFILE=live0      media profile name used in the topic
 *   ONVIF_META_PREFIX=            topic prefix; empty -> device identifier
 *
 * Deliberately its own file rather than more keys in ha.conf, because it will
 * grow: when the ONVIF service itself lands, ONVIF_ENABLED / ONVIF_PORT /
 * ONVIF_USERNAME join it here and the console shows one ONVIF card with
 * several switches, instead of ONVIF settings living in two places.
 *
 * The MQTT broker is deliberately NOT configured here. Applications already
 * hold an MQTT connection unconditionally (--mqtt-host, default localhost);
 * Home Assistant mode merely adds discovery and an LWT on top of it. Metadata
 * rides that same connection rather than opening a second one.
 */
struct OnvifMetaConfig {
    bool present = false;  /* file existed and parsed */
    bool enabled = false;

    /* Rate limit, independent of the Home Assistant results topic on purpose.
     * Those two have consumers with very different appetites: Home Assistant
     * writes every state change to its recorder database, so a per-frame feed
     * is hostile to it, whereas ONVIF metadata consumers expect a per-frame
     * scene description. Frame-accurate metadata is the RTSP metadata track's
     * job, not this one's, so a few hertz is the right default here. */
    uint32_t interval_ms = 200;

    std::string profile = "live0";
    std::string topic_prefix; /* empty -> caller substitutes the device id */
};

/* Parse the config file. A missing file is not an error: `out` comes back with
 * present=false, enabled=false and the documented defaults, which is exactly
 * the "ONVIF off" state. Returns false only on a genuine read/parse failure. */
bool loadOnvifMetaConfig(const std::string& path, OnvifMetaConfig& out,
                         std::string* error = nullptr);

/* Default location, matching what supervisor writes. */
extern const char* const ONVIF_META_CONFIG_PATH;

#endif /* _ONVIF_META_CONFIG_H_ */
