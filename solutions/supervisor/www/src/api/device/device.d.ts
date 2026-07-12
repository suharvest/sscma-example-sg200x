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

/* Hardware capability set — deviceMgr/getCapabilities (Phase 5).
 * Shape mirrors api_device::probe_capabilities(); the backend guarantees
 * every key is present (missing hardware = false / "" / []). */
interface ICapabilityGimbal {
  present: boolean;
  /** CAN interface name when present (e.g. "can0"), "" otherwise. */
  bus: string;
}

interface ICapabilityHdr {
  present: boolean;
  /** Sensor model ("OV5647" | "GC2053" | "unknown"). */
  sensor: string;
}

interface ICapabilityAudio {
  mic: boolean;
  speaker: boolean;
}

interface ICapabilities {
  /** Same string as queryDeviceInfo `type` (e.g. "Basic WiFi 8G (OV5647)"). */
  device_type: string;
  gimbal: ICapabilityGimbal;
  hdr: ICapabilityHdr;
  /** User-facing LED names under /sys/class/leds (mmc* triggers filtered). */
  leds: string[];
  audio: ICapabilityAudio;
  battery: boolean;
  sd: boolean;
  halow: boolean;
  can: boolean;
  /** Unix seconds of the probe. */
  probed_at?: number;
}

interface IAudioControl {
  name: string;    // amixer simple control name
  percent: number; // current volume 0..100
}

interface IAudioVolume {
  supported: boolean;        // false: no amixer / no volume controls
  controls: IAudioControl[];
}
