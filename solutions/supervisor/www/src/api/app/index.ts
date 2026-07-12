import { supervisorRequest } from "@/utils/request";
import {
  IAppListResult,
  IAppCurrentResult,
  IIntegrationDocResult,
  IAppConfigResult,
  ISetConfigResult,
  IConfigValues,
} from "./app";

// List all registered applications + active app id + run status
export const getAppListApi = async () =>
  supervisorRequest<IAppListResult>(
    {
      url: "api/appMgr/list",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Current (active) application details + run status + stream info
export const getCurrentAppApi = async () =>
  supervisorRequest<IAppCurrentResult>(
    {
      url: "api/appMgr/current",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Switch active application (stops current app, starts target app).
// Long timeout: stop(10s) + VPSS release(2s) + start(15s) + status(5s).
export const switchAppApi = async (data: { app_id: string }) =>
  supervisorRequest(
    {
      url: "api/appMgr/switch",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );

// Stop all applications (field troubleshooting)
export const stopAppApi = async () =>
  supervisorRequest(
    {
      url: "api/appMgr/stop",
      method: "post",
      timeout: 20000,
    },
    {
      catchs: true,
    }
  );

// Integration / output-format doc (markdown) for an app.
// Empty content means "no doc installed" — hide the section.
// `lang` selects the doc variant ("zh" -> <id>.zh.md); the BACKEND falls
// back to the English doc when the variant is missing — no retry here.
export const getIntegrationDocApi = async (
  app_id: string,
  lang?: "zh" | "en"
) =>
  supervisorRequest<IIntegrationDocResult>(
    {
      url: "api/appMgr/getIntegrationDoc",
      method: "get",
      params: lang ? { app_id, lang } : { app_id },
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// App configuration: manifest config_schema + current values + defaults.
// data.schema === null means the app has no configuration capability
// (the UI hides the configuration card entirely).
export const getConfigApi = async (app_id: string) =>
  supervisorRequest<IAppConfigResult>(
    {
      url: "api/appMgr/getConfig",
      method: "get",
      params: { app_id },
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Persist app configuration. Backend validates every key against the
// schema (code -1 with a specific message on failure, code -2 when another
// app operation holds the lock) and restarts the app when it is active —
// hence the long timeout (same stop/start path as setModel).
export const setConfigApi = async (data: {
  app_id: string;
  values: IConfigValues;
}) =>
  supervisorRequest<ISetConfigResult>(
    {
      url: "api/appMgr/setConfig",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );

// Set model for an app (writes state + restarts the app)
export const setAppModelApi = async (data: { app_id: string; model: string }) =>
  supervisorRequest(
    {
      url: "api/appMgr/setModel",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );
