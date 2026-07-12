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
  description?: string;
  type: AppType;
  init_script?: string;
  rtsp_url?: string; // may contain "{host}" placeholder
  mqtt_topic?: string;
  debug_ws?: IDebugWsInfo;
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
}

/** Manifest + runtime status, as returned by /api/appMgr/list. */
export interface IAppInfo extends IAppManifest {
  status?: AppRunState;
  running?: boolean;
  active?: boolean;
  current_model?: string;
}

export interface IAppListResult {
  apps: IAppInfo[];
  /** Active app id; backend field is `current`. */
  current?: string | null;
  active_app?: string | null;
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

export interface IAppCurrentResult {
  app?: IAppInfo | null;
  active_app?: string | null;
  /** State machine string (lowercase). */
  state?: AppRunState;
  /** Live init-script probe: "running" | "stopped" | "unknown". */
  probe?: string;
  lastError?: string;
  status?: AppRunState;
  current_model?: string;
}
