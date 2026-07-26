import { supervisorRequest } from "@/utils/request";
import { IGetOnvifConfigResult, ISetOnvifConfigResult, ISetOnvifConfigParams } from "./onvif";

// Current ONVIF config (password never returned, only password_set).
export const getOnvifConfigApi = async () =>
  supervisorRequest<IGetOnvifConfigResult>(
    {
      url: "api/appMgr/getOnvifConfig",
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// Persist the ONVIF config. The body is a partial patch — only the fields
// sent are changed. The backend restarts the active app to apply it (same
// stop/start path as setConfig), hence the long timeout.
export const setOnvifConfigApi = async (data: ISetOnvifConfigParams) =>
  supervisorRequest<ISetOnvifConfigResult>(
    {
      url: "api/appMgr/setOnvifConfig",
      method: "post",
      data,
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );
