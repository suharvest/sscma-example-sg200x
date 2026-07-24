import { supervisorRequest } from "@/utils/request";
import {
  IGetCameraConfigResult,
  ISetCameraConfigResult,
  ISetCameraConfigParams,
  IGetFocusValueResult,
} from "./camera";

// Current camera picture orientation (mirror / flip / rotation).
export const getCameraConfigApi = async () =>
  supervisorRequest<IGetCameraConfigResult>(
    {
      url: "api/appMgr/getCameraConfig",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Persist the camera picture orientation. The backend restarts the active
// app so the new orientation takes effect — same stop/start path as
// setHaConfig, hence the long timeout.
// code 0 ok, -1 invalid/failed, -2 another app operation holds the lock.
export const setCameraConfigApi = async (data: ISetCameraConfigParams) =>
  supervisorRequest<ISetCameraConfigResult>(
    {
      url: "api/appMgr/setCameraConfig",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );

// Read-only focus assist: current focus sharpness score from the running
// camera app (available=false when the camera is not running). Cheap enough
// to poll at 1s.
export const getFocusValueApi = async () =>
  supervisorRequest<IGetFocusValueResult>(
    {
      url: "api/appMgr/getFocusValue",
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );
