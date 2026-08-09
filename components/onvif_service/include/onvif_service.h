#ifndef _ONVIF_SERVICE_H_
#define _ONVIF_SERVICE_H_

#include <string>

/*
 * onvif_service: the device-level half of ONVIF -- discovery, and later the
 * Device plus Media1/Media2 SOAP services.
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

    /*
     * The SOAP listener itself. Separate from discovery_enabled because the two
     * fail independently and for different reasons: discovery can be forbidden
     * by network policy while the services stay reachable at a known address,
     * and the listener can lose its port to another process while discovery
     * keeps answering (and would then advertise an address that refuses
     * connections -- which is why start() logs loudly if this happens).
     */
    bool soap_enabled = true;

    /*
     * Snapshot endpoint, answered by GetSnapshotUri. Port 0 means the device
     * advertises no snapshot capability, which is honest for an application
     * that runs without debug_stream.
     *
     * Passed in rather than queried because onvif_service deliberately does not
     * depend on debug_stream -- see docs/onvif-implementation-spec.md 14.9.1 D1
     * and D2. The application owns both components and is the only place that
     * knows whether the stream is running and on which port.
     */
    int snapshot_port = 0;
    std::string snapshot_path = "/snapshot.jpg";

    /*
     * HTTP Digest credentials for the SOAP services. Both empty -> anonymous,
     * which is the default and matches the RTSP server's default.
     *
     * GetSystemDateAndTime and GetCapabilities stay anonymous even when these
     * are set: the first because a client cannot compute a Digest response
     * before it knows the device clock, the second because most VMS probe with
     * it and read a 401 as "device unusable" rather than "needs credentials".
     */
    std::string username;
    std::string password;

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

/* Whether the SOAP listener is up. False with discovery running is the
 * interesting state: the device is discoverable but advertises an address that
 * refuses connections. */
bool onvif_service_soap_running(void);

/* SOAP requests served, and requests rejected as unknown operations. The second
 * is what to look at when a VMS finds the device and then shows it as offline:
 * it is asking for something not implemented. */
unsigned long onvif_service_soap_count(void);
unsigned long onvif_service_soap_unknown_count(void);

/* ---------------------------------------------------------------------------
 * Internal, shared between onvif_discovery.cpp and onvif_soap.cpp.
 * Not part of the component's contract; do not call from applications.
 * --------------------------------------------------------------------------- */

/* Start/stop the SOAP listener. Called by onvif_service_start/stop. */
int onvif_soap_start(const onvif_service_config* cfg, const std::string& uuid);
void onvif_soap_stop(void);

#endif /* _ONVIF_SERVICE_H_ */
