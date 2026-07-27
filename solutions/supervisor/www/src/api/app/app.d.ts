/**
 * appMgr API types — mirrors the app manifest format described in the
 * Phase 1 spec (/usr/share/supervisor/apps/<id>.json overridden by
 * /userdata/local/apps/<id>.json).
 */

export type AppType = "native" | "external-firmware";

/** Runtime state reported by the backend app state machine. */
export type AppRunState =
  | "RUNNING"
  | "STOPPED"
  | "STOPPING"
  | "WAIT_RELEASE"
  | "STARTING"
  | "ERROR"
  | string; // be tolerant of backend additions

export interface IAppModel {
  name: string;
  path: string;
  task?: string; // detect | classify | pose | segment | ...
}

export interface IDebugWsInfo {
  port: number;
  video_path?: string; // default "/"
  results_path?: string; // default "/results"
}

export interface IAppManifest {
  id: string;
  name: string;
  name_zh?: string;
  scene?: string;
  scene_zh?: string;
  description?: string;
  description_zh?: string;
  /**
   * Optional pre-joined scene-tag line shown under the app name
   * (e.g. "Perimeter security · Parking"). Frontend falls back to a
   * built-in per-app-id wordlist (utils/appTags.ts) when absent.
   */
  tags?: string;
  tags_zh?: string;
  type: AppType;
  init_script?: string;
  /** Optional preview image URL (e.g. "/apps/<id>.svg"), shown on the card. */
  image?: string;
  rtsp_url?: string; // may contain "{host}" placeholder
  mqtt_topic?: string;
  debug_ws?: IDebugWsInfo;
  /**
   * Does this app apply the device-wide privacy mask? Gates the masking
   * shortcut on the debug page — the setting is device-wide, but offering it
   * beside a stream that ignores it makes the switch look broken. The backend
   * normalizes this to a bool (absent = false).
   */
  privacy_blur?: boolean;
  /** Switchable alternative models (mutually exclusive, setModel target). */
  models?: IAppModel[];
  /**
   * Fixed multi-model pipeline components (display / integration doc only,
   * NOT switchable — apps with a pipeline have models = []).
   */
  pipeline?: IAppModel[];
  default_model?: string;
  version?: string;
  author?: string;
  /**
   * Optional hardware dependencies (P5): capability keys the device must
   * have ("gimbal" | "hdr" | "halow" | "can" | "sd" | "battery" | "audio").
   * The backend normalizes this to an array of valid keys.
   */
  requires?: string[];
}

/** Manifest + runtime status, as returned by /api/appMgr/list. */
export interface IAppInfo extends IAppManifest {
  status?: AppRunState;
  running?: boolean;
  current_model?: string;
  /** P5: false when the device lacks required hardware (switch is refused). */
  hw_supported?: boolean;
  /** P5: requires[] entries the device does not have. */
  missing_capabilities?: string[];
}

export interface IAppListResult {
  apps: IAppInfo[];
  /** Active app id; backend field is `current`. */
  current?: string | null;
  /** State machine string (lowercase, e.g. "running"). */
  state?: AppRunState;
}

/** Result of /api/appMgr/getIntegrationDoc. */
export interface IIntegrationDocResult {
  app_id: string;
  /** Raw markdown; empty string when no doc is installed for this app. */
  content: string;
  /** "user" | "builtin" | "" */
  source?: string;
}

/* ------------------------------------------------------------------ */
/* App configuration (manifest config_schema, Phase 3)                 */
/* ------------------------------------------------------------------ */

export type ConfigItemType =
  | "number"
  | "boolean"
  | "enum"
  | "string"
  | "zone"
  | "line";

/** Normalized [x, y] point, both coords in 0..1 (resolution independent). */
export type NormPoint = [number, number];

/** zone value: polygon vertices (3..maxPoints); null = explicitly cleared. */
export type ZoneValue = NormPoint[] | null;

export type LineDirection = "ab_in" | "ab_out";

/** line value: two endpoints; direction required when item.directional. */
export interface ILineValue {
  a: NormPoint;
  b: NormPoint;
  direction?: LineDirection;
}

export type ConfigValue =
  | number
  | boolean
  | string
  | NormPoint[]
  | ILineValue
  | null;

export interface IConfigItem {
  key: string;
  type: ConfigItemType;
  title?: string;
  title_zh?: string;
  /** number */
  min?: number;
  max?: number;
  /** number — UI-only granularity */
  step?: number;
  /** enum */
  options?: string[];
  /** string (backend default 256) */
  maxLength?: number;
  /** zone (backend default 8) */
  maxPoints?: number;
  /** line — when true, value.direction is required */
  directional?: boolean;
  default?: ConfigValue;
}

export interface IConfigGroup {
  key: string;
  title?: string;
  title_zh?: string;
  items: IConfigItem[];
}

export interface IConfigSchema {
  groups: IConfigGroup[];
}

export type IConfigValues = Record<string, ConfigValue>;

/** Result of /api/appMgr/getConfig. schema === null -> app not configurable. */
export interface IAppConfigResult {
  app_id: string;
  schema: IConfigSchema | null;
  /** Current /userdata/local/apps/<id>.config.json content ({} if unset). */
  values: IConfigValues;
  /** Per-key defaults declared in the schema. */
  defaults: IConfigValues;
}

/** Result of /api/appMgr/setConfig (code 0). */
export interface ISetConfigResult {
  app_id: string;
  values: IConfigValues;
  /** True when the active app was restarted to apply the config. */
  restarted: boolean;
}

/**
 * Result of /api/appMgr/installApp (present on code 0 and code -1; code -2
 * means "busy" and carries no data).
 */
export interface IInstallAppResult {
  /** Canonical (realpath) package path that was installed. */
  path: string;
  /** opkg exit code; 124 = timed out after 120s. */
  exit_code: number;
  /** Tail (up to 2KB) of the opkg install output. */
  output: string;
  /** Number of app manifests visible after the install (refresh hint). */
  apps_count: number;
}

export interface IAppCurrentResult {
  app?: IAppInfo | null;
  /** State machine string (lowercase). */
  state?: AppRunState;
  /** Live init-script probe: "running" | "stopped" | "unknown". */
  probe?: string;
  lastError?: string;
  current_model?: string;
}
