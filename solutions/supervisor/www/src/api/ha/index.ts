import { supervisorRequest } from "@/utils/request";
import {
  IGetHaConfigResult,
  ISetHaConfigResult,
  ISetHaConfigParams,
  ITestHaConnectionResult,
  ITestHaConnectionParams,
} from "./ha";

// Current Home Assistant MQTT config (password never returned, only
// password_set).
export const getHaConfigApi = async () =>
  supervisorRequest<IGetHaConfigResult>(
    {
      url: "api/appMgr/getHaConfig",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Persist the Home Assistant MQTT config. Omitting `password` keeps the
// stored one. The backend restarts the active app so the config takes
// effect — same stop/start path as setConfig, hence the long timeout.
// code 0 ok, -1 invalid/failed, -2 another app operation holds the lock.
export const setHaConfigApi = async (data: ISetHaConfigParams) =>
  supervisorRequest<ISetHaConfigResult>(
    {
      url: "api/appMgr/setHaConfig",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );

// Probe the broker without persisting anything (TCP + MQTT CONNECT, ~5s
// budget on-device). Pass use_saved_password to test with the stored
// password without re-typing it. code 0 ok, -1 failed (data.mosquitto_rc,
// human message in msg), -2 a test is already running.
export const testHaConnectionApi = async (data: ITestHaConnectionParams) =>
  supervisorRequest<ITestHaConnectionResult>(
    {
      url: "api/appMgr/testHaConnection",
      method: "post",
      data,
      timeout: 15000,
    },
    {
      catchs: true,
    }
  );
