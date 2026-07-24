/**
 * appMgr camera picture / focus assist API types.
 * Orientation is persisted on-device at /userdata/local/camera.conf and read
 * by the active application at startup (saving restarts the active app).
 */

export interface ICameraConfig {
  mirror: boolean;
  flip: boolean;
  /** Only 0 and 180 are supported by the sensor pipeline. */
  rotation: 0 | 180;
}

/** getCameraConfig -> data */
export type IGetCameraConfigResult = ICameraConfig;

/** setCameraConfig -> data (saved config + whether the active app was restarted) */
export interface ISetCameraConfigResult extends ICameraConfig {
  restarted: boolean;
  note?: string;
}

export interface ISetCameraConfigParams {
  mirror?: boolean;
  flip?: boolean;
  rotation?: 0 | 180;
}

/** getFocusValue -> data. fv/ts come straight from /tmp/camera_fv.json;
 *  available=false when the camera is not running (file missing or stale). */
export interface IGetFocusValueResult {
  available: boolean;
  /** Focus sharpness score — relative, higher is sharper. */
  fv?: number;
  /** Device-monotonic timestamp of the sample (not wall time). */
  ts?: number;
}
