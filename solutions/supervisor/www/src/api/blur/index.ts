import { supervisorRequest } from "@/utils/request";
import {
  IGetBlurConfigResult,
  ISetBlurConfigResult,
  ISetBlurConfigParams,
} from "./blur";

// Current privacy blur config.
export const getBlurConfigApi = async () =>
  supervisorRequest<IGetBlurConfigResult>(
    {
      url: "api/appMgr/getBlurConfig",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Persist the privacy blur config. The body is a partial patch — only the
// fields sent are changed. The backend restarts the active app to apply it
// (same stop/start path as setConfig), hence the long timeout.
export const setBlurConfigApi = async (data: ISetBlurConfigParams) =>
  supervisorRequest<ISetBlurConfigResult>(
    {
      url: "api/appMgr/setBlurConfig",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );
