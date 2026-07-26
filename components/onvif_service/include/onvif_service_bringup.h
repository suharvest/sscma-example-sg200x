#ifndef _ONVIF_SERVICE_BRINGUP_H_
#define _ONVIF_SERVICE_BRINGUP_H_

#include <string>

#include "onvif_meta_config.h"
#include "onvif_service.h"

/*
 * One call to bring ONVIF up in an application.
 *
 * Without this every application would repeat the same eight lines of mapping
 * from the config file onto onvif_service_config. Seven copies of a mapping is
 * how the ":554" string stayed wrong across eight applications for as long as
 * it did, and this file exists so that does not happen again.
 *
 * Header-only, and it includes onvif_meta_config.h rather than onvif_service
 * gaining a link dependency on onvif_meta: the two components stay independent
 * (see docs/onvif-implementation-spec.md 14.9.1 D1), they merely agree on a
 * config file. An application that links onvif_service alone still builds; it
 * just fills onvif_service_config itself.
 *
 * Typical use, right after the RTSP server is up and the debug stream port is
 * known:
 *
 *     onvif_service_bringup(cfg, deviceId, "reCamera", debug_port);
 *
 * Ordering matters: call it AFTER rtsp_server_start(), because GetProfiles and
 * GetStreamUri are answered from the running server's session list. Called
 * before, the device advertises zero profiles and a VMS shows it as a camera
 * with no video.
 */
inline int onvif_service_bringup(const OnvifMetaConfig& cfg,
    const std::string& serial, const std::string& device_name,
    int snapshot_port)
{
    if (!cfg.service_enabled) return 0;

    onvif_service_config sc;
    onvif_service_config_init(&sc);
    sc.discovery_enabled = true;
    sc.soap_enabled = true;
    sc.service_port = cfg.service_port;
    sc.serial = serial;
    sc.username = cfg.username;
    sc.password = cfg.password;
    sc.location = cfg.location;
    if (!device_name.empty()) sc.device_name = device_name;
    /* Zero when the application runs without a debug stream, which makes
     * GetSnapshotUri answer ActionNotSupported rather than hand out a URL that
     * refuses connections. */
    sc.snapshot_port = snapshot_port;

    return onvif_service_start(&sc);
}

#endif /* _ONVIF_SERVICE_BRINGUP_H_ */
