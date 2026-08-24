/**
 * appMgr ONVIF integration API types.
 *
 * Two independent features share one config blob:
 *  - `service_*`  : the ONVIF device service (discovery + stream URI for a VMS)
 *  - `meta_*`     : ONVIF analytics over the RTSP metadata track and MQTT
 *
 * The password is write-only — the backend never returns it, only `password_set`.
 */

export interface IOnvifConfig {
  /** Expose ONVIF analytics over RTSP and, when connected, MQTT. */
  meta_enabled: boolean;
  /** MQTT metadata publish interval, 20-60000 ms; RTSP follows inference. */
  meta_interval_ms: number;
  /** Profile token the metadata is attributed to, default "live0". */
  meta_profile: string;
  /** Topic prefix; empty means the device serial number is used. */
  meta_prefix: string;
  /** Run the ONVIF device service so a VMS can discover and pull the stream. */
  service_enabled: boolean;
  /** ONVIF device service port, default 8000. */
  service_port: number;
  /** ONVIF user; empty means anonymous access. */
  username: string;
  /** True when a password is stored on the device. */
  password_set: boolean;
  /** Free-form location string reported by the device, e.g. "city/Shenzhen". */
  location: string;
}

/** getOnvifConfig -> data */
export type IGetOnvifConfigResult = IOnvifConfig;

/** setOnvifConfig -> data */
export interface ISetOnvifConfigResult {
  /** False in Node-RED mode (note: "nodered_mode") — config stored, not applied. */
  restarted: boolean;
  note?: string;
}

/**
 * Any subset of the config fields; omitted fields keep their stored value.
 * `password` omitted keeps the stored password, `""` clears it.
 */
export interface ISetOnvifConfigParams {
  meta_enabled?: boolean;
  meta_interval_ms?: number;
  meta_profile?: string;
  meta_prefix?: string;
  service_enabled?: boolean;
  service_port?: number;
  username?: string;
  password?: string;
  location?: string;
}
