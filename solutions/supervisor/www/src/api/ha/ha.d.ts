/**
 * appMgr Home Assistant integration API types.
 * Config is persisted on-device at /userdata/local/ha.conf; the password is
 * write-only (never returned — only password_set).
 */

export interface IHaConfig {
  enabled: boolean;
  broker_host: string;
  broker_port: number;
  username: string;
  discovery_prefix: string;
  /** True when a password is stored on the device. */
  password_set: boolean;
}

/** getHaConfig -> data */
export type IGetHaConfigResult = IHaConfig;

/** setHaConfig -> data (saved config + whether the active app was restarted) */
export interface ISetHaConfigResult extends IHaConfig {
  restarted: boolean;
  note?: string;
}

export interface ISetHaConfigParams {
  enabled: boolean;
  broker_host: string;
  broker_port: number;
  username?: string;
  /** Omit to keep the stored password unchanged. */
  password?: string;
  discovery_prefix?: string;
}

/** testHaConnection -> data (code 0 ok, -1 failed, -2 busy) */
export interface ITestHaConnectionResult {
  broker_host: string;
  broker_port: number;
  /** libmosquitto rc / CONNACK code, context in msg. */
  mosquitto_rc: number;
}

export interface ITestHaConnectionParams {
  broker_host: string;
  broker_port: number;
  username?: string;
  password?: string;
  /** Use the password already stored on the device instead of `password`. */
  use_saved_password?: boolean;
}
