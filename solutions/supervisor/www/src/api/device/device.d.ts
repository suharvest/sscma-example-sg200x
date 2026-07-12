import { DeviceChannleMode, DeviceNeedRestart, ServiceStatus } from "@/enum";

interface IDeviceInfo {
  appName: string;
  deviceName: string;
  isReCamera: boolean;
  ip: string;
  sn: string;
  wifiIp: string;
  mask: string;
  gateway: string;
  dns: string;
  channel: DeviceChannleMode;
  serverUrl: string;
  officialUrl: string;
  cpu: string;
  ram: string;
  npu: string;
  osVersion: string;
  osName: string;
  osUpdateTime: string;
  needRestart: DeviceNeedRestart;
  type: string; //Gimbal with Wifi - 8G
  // Gallery (solution console) mode flag. Optional: older firmware omits it.
  galleryMode?: boolean;
  [prop: string]: string;
}

interface IChannelParams {
  channel?: DeviceChannleMode;
  serverUrl?: string;
}

interface IServiceStatus {
  sscmaNode: ServiceStatus;
  nodeRed: ServiceStatus;
  system: ServiceStatus;
  uptime: number;
}

interface IIPDevice {
  device: string;
  info: { sn: string; lastSix: string } | string;
  ip: string;
  type: string;
}

interface IBatteryInfo {
  voltage?: number;         // Battery voltage (mV), undefined when unavailable
}

interface ISensorStorage {
  total?: number;     // bytes (or KB, backend-defined) total of /userdata
  used?: number;
  available?: number;
}

interface ISensorStatus {
  temperature_c?: number;   // SoC temperature in Celsius
  storage?: ISensorStorage; // /userdata usage
}

interface IAudioControl {
  name: string;    // amixer simple control name
  percent: number; // current volume 0..100
}

interface IAudioVolume {
  supported: boolean;        // false: no amixer / no volume controls
  controls: IAudioControl[];
}
