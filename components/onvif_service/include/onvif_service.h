#ifndef _ONVIF_SERVICE_H_
#define _ONVIF_SERVICE_H_

#include <string>

/*
 * onvif_service: the device-level half of ONVIF -- discovery, and later the
 * Device and Media2 SOAP services.
 *
 * Split from components/onvif_meta on purpose. Everything here answers
 * questions about the *device*: serial number, RTSP endpoint, snapshot URL,
 * network interfaces, clock. None of it depends on which application is
 * running, so an application gets all of it by linking the component and
 * calling start() -- no per-application code, unlike the analytics payload
 * where "what is this detection called in ONVIF's vocabulary" has no generic
 * answer.
 *
 * The endpoints it advertises come from the components that own them rather
 * than being duplicated here:
 *   GetStreamUri / GetProfiles -> rtsp_server_url(), rtsp_server_session_name()
 *   GetSnapshotUri             -> debug_stream's /snapshot.jpg
 *   GetDeviceInformation       -> the serial number
 * That is why rtsp_server was given self-description; without it every one of
 * those would be a hardcoded string somewhere, which is exactly how the
 * ":554" bug survived for so long.
 */

struct onvif_service_config {
    /*
     * Discovery defaults to ON. A camera whose discovery ships disabled is a
     * camera nobody finds, which is the opposite of the point. The switch
     * exists for networks that forbid unapproved service advertisement, and
     * for sites already running an ONVIF gateway where a second advertiser
     * shows up as a duplicate device.
     */
    bool discovery_enabled = true;

    /* Where the SOAP services live; advertised as XAddrs in ProbeMatches. */
    int service_port = 8000;
    std::string service_path = "/onvif/device_service";

    /* Shown by clients in device lists. */
    std::string device_name = "reCamera";
    std::string manufacturer = "Seeed Studio";
    std::string model = "reCamera";
    std::string firmware;      /* empty -> read from the OS */
    std::string serial;        /* empty -> readDeviceIdentifier() */
    std::string hardware = "SG2002";

    /*
     * Free-form location scope, e.g. "country/CN" or "city/Shenzhen".
     * Empty means none is advertised.
     *
     * NOTE: profile scopes (onvif://www.onvif.org/Profile/...) are
     * deliberately NOT configurable. Advertising one is a conformance claim,
     * and ONVIF asks unconformant products to stop making them. Only the
     * factual type scope is sent. See the note in onvif_discovery.cpp.
     */
    std::string location;
};

void onvif_service_config_init(onvif_service_config* cfg);

/*
 * Start the discovery responder (and, later, the SOAP listener). Returns 0 on
 * success. Spawns its own thread; safe to call once.
 *
 * A failure to join the multicast group is not fatal and not silent: the rest
 * of the service still runs and the reason is logged, because a device that
 * merely cannot be auto-discovered is far better than one that refuses to
 * start.
 */
int onvif_service_start(const onvif_service_config* cfg);

/* Send the Bye announcement, stop the thread, release resources. */
void onvif_service_stop(void);

/* Whether the discovery responder is currently running. */
bool onvif_service_discovery_running(void);

/* Number of Probe messages answered since start; for diagnostics, since the
 * usual failure mode is "the VMS cannot see it" with nothing else to look at. */
unsigned long onvif_service_probe_count(void);

#endif /* _ONVIF_SERVICE_H_ */
