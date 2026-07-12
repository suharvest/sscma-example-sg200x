import { useEffect, useState } from "react";
import { Button, Modal, Progress, Switch, message } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  PoweroffOutlined,
} from "@ant-design/icons";
import {
  queryBatteryInfoApi,
  getSensorStatusApi,
  getTimestampApi,
  getTimezoneApi,
  setDevicePowerApi,
} from "@/api/device/index";
import { ISensorStatus } from "@/api/device/device";
import { setLedApi } from "@/api/led";
import { PowerMode } from "@/enum";

/** Common LED names on reCamera (/sys/class/leds/<name>). Toggling an
 *  absent LED returns an error and the row is marked unavailable. */
const LED_NAMES = ["white", "red", "blue"];

type LedState = "unknown" | "on" | "off" | "unavailable";

function formatStorage(value?: number): string {
  if (value === undefined || value === null || isNaN(value)) return "-";
  // Heuristic: df reports 1K blocks; anything below 100M is treated as KB.
  const bytes = value < 1e8 ? value * 1024 : value;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

const DeviceTools = () => {
  const [ledStates, setLedStates] = useState<Record<string, LedState>>(
    Object.fromEntries(LED_NAMES.map((n) => [n, "unknown"]))
  );
  const [battery, setBattery] = useState<number | null | undefined>(undefined); // undefined=loading, null=unavailable
  const [sensor, setSensor] = useState<ISensorStatus | null | undefined>(
    undefined
  );
  const [deviceTime, setDeviceTime] = useState<number | null>(null);
  const [timezone, setTimezone] = useState<string>("");
  const [powerLoading, setPowerLoading] = useState(false);

  const fetchAll = async () => {
    setBattery(undefined);
    setSensor(undefined);
    // Battery
    queryBatteryInfoApi()
      .then((res) => {
        if ((res.code === 0 || res.code === "0") && res.data?.voltage) {
          setBattery(res.data.voltage);
        } else {
          setBattery(null);
        }
      })
      .catch(() => setBattery(null));
    // Temperature + storage (new endpoint, may not exist on older firmware)
    getSensorStatusApi()
      .then((res) => {
        if ((res.code === 0 || res.code === "0") && res.data) {
          setSensor(res.data);
        } else {
          setSensor(null);
        }
      })
      .catch(() => setSensor(null));
    // Time / timezone
    getTimestampApi()
      .then((res) => {
        if ((res.code === 0 || res.code === "0") && res.data?.timestamp) {
          setDeviceTime(res.data.timestamp * 1000);
        }
      })
      .catch(() => setDeviceTime(null));
    getTimezoneApi()
      .then((res) => {
        if (res.code === 0 || res.code === "0") {
          setTimezone(res.data?.timezone || "");
        }
      })
      .catch(() => setTimezone(""));
  };

  useEffect(() => {
    fetchAll();
  }, []);

  const onToggleLed = async (name: string, on: boolean) => {
    const prev = ledStates[name];
    setLedStates((s) => ({ ...s, [name]: on ? "on" : "off" }));
    try {
      const res = await setLedApi(name, on);
      if (res.code !== 0 && res.code !== "0") {
        setLedStates((s) => ({ ...s, [name]: "unavailable" }));
        message.warning(`LED "${name}" is not available on this device`);
      }
    } catch (e) {
      setLedStates((s) => ({
        ...s,
        [name]: prev === "unknown" ? "unavailable" : prev,
      }));
      message.error(`Failed to set LED "${name}"`);
    }
  };

  const onPower = (mode: PowerMode) => {
    const isReboot = mode === PowerMode.Restart;
    Modal.confirm({
      title: isReboot ? "Reboot device?" : "Shut down device?",
      icon: <ExclamationCircleOutlined />,
      content: isReboot
        ? "The device will restart. All running applications and streams will be interrupted."
        : "The device will power off. You will need physical access to turn it back on.",
      okText: isReboot ? "Reboot" : "Shut down",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        setPowerLoading(true);
        try {
          await setDevicePowerApi({ mode });
          message.success(isReboot ? "Rebooting…" : "Shutting down…");
        } catch (e) {
          message.error("Operation failed");
        } finally {
          setPowerLoading(false);
        }
      },
    });
  };

  const storage = sensor?.storage;
  const storagePercent =
    storage?.total && storage?.used
      ? Math.min(Math.round((storage.used / storage.total) * 100), 100)
      : null;

  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">Solution Console</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            Device
          </h1>
          <p className="text-muted text-13 mt-4 mb-0">
            Low-level device controls and health.
          </p>
        </div>
        <Button icon={<ReloadOutlined />} onClick={fetchAll}>
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-16 mt-24">
        {/* LED control */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">LED Control</div>
          <div className="flex flex-col">
            {LED_NAMES.map((name, i) => (
              <div
                key={name}
                className={`flex items-center justify-between py-10 ${
                  i ? "border-t border-line" : ""
                }`}
              >
                <div>
                  <span className="text-14 font-medium capitalize">
                    {name} LED
                  </span>
                  {ledStates[name] === "unavailable" && (
                    <span className="text-12 text-muted ml-8">
                      not available on this device
                    </span>
                  )}
                </div>
                <Switch
                  size="small"
                  checked={ledStates[name] === "on"}
                  disabled={ledStates[name] === "unavailable"}
                  onChange={(v) => onToggleLed(name, v)}
                />
              </div>
            ))}
          </div>
          <div className="text-12 text-muted mt-8">
            Writes /sys/class/leds/&lt;name&gt;/brightness directly.
          </div>
        </div>

        {/* Battery */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">Battery</div>
          {battery === undefined ? (
            <div className="text-13 text-muted">Reading…</div>
          ) : battery === null ? (
            <div className="text-13 text-muted">
              No battery detected on this device.
            </div>
          ) : (
            <div>
              <div className="rc-kpi-label">Voltage</div>
              <div className="rc-kpi-value">
                {(battery / 1000).toFixed(2)}
                <span className="text-14 font-medium text-muted ml-4">V</span>
              </div>
              <div className="rc-mono text-11 text-muted mt-2">
                {battery} mV raw
              </div>
            </div>
          )}
        </div>

        {/* Temperature + storage */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">Health</div>
          {sensor === undefined ? (
            <div className="text-13 text-muted">Reading…</div>
          ) : sensor === null ? (
            <div className="text-13 text-muted">
              Sensor endpoint unavailable on this firmware.
            </div>
          ) : (
            <div className="flex flex-col gap-16">
              <div>
                <div className="rc-kpi-label">SoC Temperature</div>
                <div className="rc-kpi-value">
                  {sensor.temperature_c !== undefined
                    ? sensor.temperature_c.toFixed(1)
                    : "-"}
                  <span className="text-14 font-medium text-muted ml-4">
                    °C
                  </span>
                </div>
              </div>
              <div>
                <div className="rc-kpi-label">Storage (/userdata)</div>
                {storagePercent !== null ? (
                  <>
                    <Progress
                      percent={storagePercent}
                      size="small"
                      strokeColor="#8fc31f"
                    />
                    <div className="rc-mono text-11 text-muted">
                      {formatStorage(storage?.used)} used /{" "}
                      {formatStorage(storage?.total)} ·{" "}
                      {formatStorage(storage?.available)} free
                    </div>
                  </>
                ) : (
                  <div className="text-13 text-muted mt-4">-</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Time */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">Time &amp; Timezone</div>
          <div className="rc-kpi-label">Device time</div>
          <div className="rc-kpi-value" style={{ fontSize: 20 }}>
            {deviceTime ? new Date(deviceTime).toLocaleString() : "-"}
          </div>
          <div className="rc-mono text-11 text-muted mt-6">
            timezone: {timezone || "-"}
          </div>
          <div className="text-12 text-muted mt-10">
            Change the timezone in System settings.
          </div>
        </div>

        {/* Power */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">Power</div>
          <div className="flex flex-col gap-10">
            <Button
              danger
              icon={<ReloadOutlined />}
              loading={powerLoading}
              onClick={() => onPower(PowerMode.Restart)}
            >
              Reboot
            </Button>
            <Button
              danger
              icon={<PoweroffOutlined />}
              loading={powerLoading}
              onClick={() => onPower(PowerMode.Shutdown)}
            >
              Shutdown
            </Button>
          </div>
          <div className="text-12 text-muted mt-10">
            Both actions interrupt all running applications.
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeviceTools;
