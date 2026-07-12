import axios from "axios";
import { supervisorRequest } from "@/utils/request";
import { baseIP } from "@/utils/supervisorRequest";
import { getToken } from "@/store/user";
import { PowerMode, DeviceChannleMode, SystemUpdateStatus } from "@/enum";
import {
  IDeviceInfo,
  IChannelParams,
  IServiceStatus,
  IIPDevice,
  IBatteryInfo,
  ISensorStatus,
  IAudioVolume,
  ICapabilities,
} from "./device";

// 获取设备信息
export const queryDeviceInfoApi = async () =>
  supervisorRequest<IDeviceInfo>(
    {
      url: "api/deviceMgr/queryDeviceInfo",
      method: "get",
    },
    {
      catchs: true,
    }
  );

export const getDeviceListApi = async () =>
  supervisorRequest<{
    deviceList: IIPDevice[];
  }>(
    {
      url: "api/deviceMgr/getDeviceList",
      method: "get",
    },
    {
      catchs: true,
    }
  );

// 获取设备运行状态
export const queryServiceStatusApi = async () =>
  supervisorRequest<IServiceStatus>(
    {
      url: "api/deviceMgr/queryServiceStatus",
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );

// 修改设备信息
export const updateDeviceInfoApi = async (data: { deviceName: string }) =>
  supervisorRequest<IDeviceInfo>({
    url: "api/deviceMgr/updateDeviceName",
    method: "post",
    data,
  });

// 修改渠道信息
export const changeChannleApi = async (data: IChannelParams) =>
  supervisorRequest({
    url: "api/deviceMgr/updateChannel",
    method: "post",
    data,
  });
// 设备重启与关机
export const setDevicePowerApi = async (data: { mode: PowerMode }) =>
  supervisorRequest({
    url: "api/deviceMgr/setPower",
    method: "post",
    data,
  });
// 更新设备系统
export const updateSystemApi = async () =>
  supervisorRequest({
    url: "api/deviceMgr/updateSystem",
    method: "post",
  });
// 获取设备更新进度
export const getUpdateSystemProgressApi = async () =>
  supervisorRequest<{
    progress: number;
  }>({
    url: "api/deviceMgr/getUpdateProgress",
    method: "get",
  });
// 更新设备系统
export const cancelUpdateApi = async () =>
  supervisorRequest({
    url: "api/deviceMgr/cancelUpdate",
    method: "post",
  });

// 获取设备更新版本信息
export const getSystemUpdateVesionInfoApi = async (data: {
  url: string;
  channel?: DeviceChannleMode;
}) =>
  supervisorRequest<{
    osName: string;
    osVersion: string;
    status: SystemUpdateStatus;
  }>(
    {
      url: "api/deviceMgr/getSystemUpdateVersion",
      method: "post",
      data,
    },
    {
      catchs: true,
    }
  );

// 获取模型信息
export const getModelInfoApi = async () =>
  supervisorRequest<IDeviceInfo>(
    {
      url: "api/deviceMgr/getModelInfo",
      method: "get",
    },
    {
      catchs: true,
    }
  );

// 上传模型（支持分片上传）
export const uploadModelApi = async (
  data: FormData,
  onProgress?: (progress: number) => void,
  signal?: AbortSignal
) => {
  const CHUNK_SIZE = 512 * 1024; // 512KB
  
  // 提取文件和其他数据
  let modelFile: File | Blob | null = null;
  let modelInfo: string | null = null;
  
  for (const [key, value] of data.entries()) {
    if (key === "model_file" && (value instanceof File || value instanceof Blob)) {
      modelFile = value;
    } else if (key === "model_info" && typeof value === "string") {
      modelInfo = value;
    }
  }
  
  // 如果没有模型文件或文件小于阈值，使用传统上传
  if (!modelFile || modelFile.size <= CHUNK_SIZE) {
    return supervisorRequest<IDeviceInfo>({
      url: "api/deviceMgr/uploadModel",
      method: "post",
      data,
      signal,
    });
  }
  
  // 分片上传大文件
  const totalSize = modelFile.size;
  let offset = 0;
  
  while (offset < totalSize) {
    const chunk = modelFile.slice(offset, offset + CHUNK_SIZE);
    const chunkFormData = new FormData();
    
    chunkFormData.append("model_file", chunk);
    chunkFormData.append("offset", offset.toString());
    chunkFormData.append("size", totalSize.toString());
    
    // 只在最后一个分片附加 model_info
    if (offset + CHUNK_SIZE >= totalSize && modelInfo) {
      chunkFormData.append("model_info", modelInfo);
    }
    
    const response = await supervisorRequest<IDeviceInfo>({
      url: "api/deviceMgr/uploadModel",
      method: "post",
      data: chunkFormData,
      signal,
    });
    
    offset += CHUNK_SIZE;
    
    // 报告进度
    if (onProgress) {
      const progress = Math.min((offset / totalSize) * 100, 100);
      onProgress(progress);
    }
    
    // 如果这是最后一个分片，返回响应
    if (offset >= totalSize) {
      return response;
    }
  }
  
  // 理论上不会到达这里，但为了类型安全
  throw new Error("Upload failed");
};

// 保存平台信息
export const savePlatformInfoApi = async (data: { platform_info: string }) =>
  supervisorRequest({
    url: "api/deviceMgr/savePlatformInfo",
    method: "post",
    data,
  });

// 获取平台信息
export const getPlatformInfoApi = async () =>
  supervisorRequest<{
    platform_info: string;
  }>(
    {
      url: "api/deviceMgr/getPlatformInfo",
      method: "get",
    },
    {
      catchs: true,
    }
  );

// 获取电池信息
export const queryBatteryInfoApi = async () =>
  supervisorRequest<IBatteryInfo>(
    {
      url: "api/deviceMgr/queryBatteryInfo",
      method: "get",
    },
    {
      catchs: true,
    }
  );

// 获取温度/存储传感器状态
export const getSensorStatusApi = async () =>
  supervisorRequest<ISensorStatus>(
    {
      url: "api/deviceMgr/getSensorStatus",
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );

// 获取设备当前时间戳（秒）
export const getTimestampApi = async () =>
  supervisorRequest<{ timestamp: number }>(
    {
      url: "api/deviceMgr/getTimestamp",
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );

// 设置设备时间戳（秒）。后端 setTimestamp 从 JSON body 里按字符串取
// "timestamp"（body.value<string>），所以这里必须传字符串。
export const setTimestampApi = async (timestamp: number) =>
  supervisorRequest<{ timestamp: number }>(
    {
      url: "api/deviceMgr/setTimestamp",
      method: "post",
      data: { timestamp: String(Math.floor(timestamp)) },
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// 设置设备时区。后端校验 /usr/share/zoneinfo/<timezone> 是否存在，
// 直接传 IANA 名称（如 "Asia/Shanghai"）。
export const setTimezoneApi = async (timezone: string) =>
  supervisorRequest<{ timezone: string }>(
    {
      url: "api/deviceMgr/setTimezone",
      method: "post",
      data: { timezone },
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );

// NTP 时间同步（P3-B 后端：依次尝试多个 NTP 服务器，单个 10s 超时）。
// 成功返回 data.timestamp（秒）。设备无外网时会失败。
export const syncTimeNtpApi = async () =>
  supervisorRequest<{ timestamp: number }>(
    {
      url: "api/deviceMgr/syncTime",
      method: "post",
      timeout: 45000,
    },
    {
      catchs: true,
    }
  );

// 麦克风录音探测（P3-E）。成功时后端以 audio/wav 二进制回传录音，所以
// 不能走 supervisorRequest（那是 JSON 封装）；失败时后端返回 JSON
// {code, msg}（code === -2 表示音频设备忙），这里解析后作为 reject
// 原因抛出。duration 后端严格校验整数 1..10（按 setTimestamp 约定传
// 字符串）。timeout 20s 覆盖最长 10s 录音 + 回传。
export const audioRecordApi = async (duration: number): Promise<Blob> => {
  const res = await axios.get(`${baseIP}/api/deviceMgr/audioRecord`, {
    params: { duration: String(duration) },
    responseType: "blob",
    timeout: 20000,
    headers: { Authorization: getToken() },
  });
  const blob = res.data as Blob;
  if (blob.type.includes("application/json")) {
    throw JSON.parse(await blob.text());
  }
  return blob;
};

// 扬声器测试音（P3-E）：设备端 aplay 播放随 deb 安装的 2s 提示音，
// 成功仅代表播放命令执行完成，需要人在设备旁确认出声。
export const audioPlayTestApi = async () =>
  supervisorRequest(
    {
      url: "api/deviceMgr/audioPlayTest",
      method: "post",
      timeout: 15000,
    },
    {
      catchs: true,
    }
  );

// 获取音量控制项（P3-E）。supported=false 表示设备没有可用 mixer，
// 前端不渲染音量滑条。
export const getAudioVolumeApi = async () =>
  supervisorRequest<IAudioVolume>(
    {
      url: "api/deviceMgr/audioVolume",
      method: "get",
      timeout: 8000,
    },
    {
      catchs: true,
    }
  );

// 设置音量（P3-E）。percent 按 setTimestamp 约定传字符串（后端从
// JSON body 按字符串取值）。
export const setAudioVolumeApi = async (control: string, percent: number) =>
  supervisorRequest(
    {
      url: "api/deviceMgr/audioVolume",
      method: "post",
      data: { control, percent: String(Math.round(percent)) },
      timeout: 8000,
    },
    {
      catchs: true,
    }
  );

// 切换运行模式（P4-D）：console = C++ 应用画廊（生产），nodered = Node-RED
// 流程编排（调试）。后端同步完成：写模式文件 → 停/起相关服务（→nodered 最坏
// 约 40s）→ 进程内翻转 galleryMode 与看门狗，**不重启 supervisor**。
// 成功返回时 galleryMode 已翻转，前端再拉一次 queryDeviceInfo 确认后
// location.reload() 即可（见 useRunModeSwitch）。后端 shell 预算 60s，
// 这里 90s 兜底（含 →console 时看门狗线程 join 的等待）。
export const setRunModeApi = async (mode: "console" | "nodered") =>
  supervisorRequest<{ mode: string; galleryMode: boolean }>(
    {
      url: "api/deviceMgr/setRunMode",
      method: "post",
      data: { mode },
      timeout: 90000,
    },
    {
      catchs: true,
    }
  );

// 获取设备时区
export const getTimezoneApi = async () =>
  supervisorRequest<{ timezone: string }>(
    {
      url: "api/deviceMgr/getTimezone",
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );

// 硬件能力探测（P5）。首次调用后端会跑 shell 探测并缓存进程级结果；
// refresh=true 强制重探。REG_API_NO_AUTH，登录前也可拉取。
export const getCapabilitiesApi = async (refresh = false) =>
  supervisorRequest<ICapabilities>(
    {
      url: `api/deviceMgr/getCapabilities${refresh ? "?refresh=1" : ""}`,
      method: "get",
      timeout: 10000,
    },
    {
      catchs: true,
    }
  );
