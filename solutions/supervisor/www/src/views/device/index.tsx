import { useEffect, useRef, useState } from "react";
import { Button, Dropdown, Modal, Progress, Segmented, Slider, Switch, Tooltip, message } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  PoweroffOutlined,
  ClockCircleOutlined,
  GlobalOutlined,
  AudioOutlined,
  SoundOutlined,
  DownloadOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import {
  queryBatteryInfoApi,
  getSensorStatusApi,
  getTimestampApi,
  getTimezoneApi,
  setTimestampApi,
  setTimezoneApi,
  syncTimeNtpApi,
  setDevicePowerApi,
  audioRecordApi,
  audioPlayTestApi,
  audioPlayRecordingApi,
  getAudioVolumeApi,
  setAudioVolumeApi,
} from "@/api/device/index";
import { ISensorStatus, IAudioVolume } from "@/api/device/device";
import { getCameraConfigApi, setCameraConfigApi } from "@/api/camera";
import { ICameraConfig } from "@/api/camera/camera";
import { isOk, isBusy } from "@/utils/api";
import { setLedApi } from "@/api/led";
import { PowerMode } from "@/enum";
import useCapabilitiesStore from "@/store/capabilities";

/** Legacy fallback LED list, used only while the capability probe has not
 *  answered (older firmware). With capabilities present the card renders
 *  the enumerated /sys/class/leds names instead. */
const LED_NAMES = ["white", "red", "blue"];

/** Optional-hardware badge order for the Hardware card. */
const HW_BADGE_KEYS = ["gimbal", "hdr", "halow", "sd", "can"] as const;

type LedState = "unknown" | "on" | "off" | "unavailable";

/** Device clock is considered "not set" below this year (no RTC battery
 *  -> boots at 1970). */
const CLOCK_SANE_YEAR = 2020;

function formatStorage(value?: number): string {
  if (value === undefined || value === null || isNaN(value)) return "-";
  // Heuristic: df reports 1K blocks; anything below 100M is treated as KB.
  const bytes = value < 1e8 ? value * 1024 : value;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

const WAVE_BARS = 56;

/** Decode a recorded WAV blob and reduce channel 0 to WAVE_BARS peak bars
 *  (normalized to the loudest bar so quiet speech stays visible). `silent` is
 *  true when the loudest sample sits at the noise floor — a dead/absent mic. */
async function analyzeWaveform(
  blob: Blob
): Promise<{ peaks: number[]; silent: boolean }> {
  const AC =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;
  const arrayBuf = await blob.arrayBuffer();
  const ctx = new AC();
  try {
    const audio = await ctx.decodeAudioData(arrayBuf);
    const ch = audio.getChannelData(0);
    const block = Math.floor(ch.length / WAVE_BARS) || 1;
    const peaks: number[] = [];
    let globalMax = 0;
    for (let i = 0; i < WAVE_BARS; i++) {
      let max = 0;
      const start = i * block;
      for (let j = 0; j < block && start + j < ch.length; j++) {
        const v = Math.abs(ch[start + j]);
        if (v > max) max = v;
      }
      peaks.push(max);
      if (max > globalMax) globalMax = max;
    }
    const norm = globalMax > 0 ? peaks.map((p) => p / globalMax) : peaks;
    return { peaks: norm, silent: globalMax < 0.01 };
  } finally {
    ctx.close();
  }
}

const DeviceTools = () => {
  const { t } = useTranslation();
  // P5 capability-driven rendering. undefined = probe not answered (older
  // firmware): keep every card visible with its legacy behavior.
  const capabilities = useCapabilitiesStore((s) => s.capabilities);
  const ledNames = capabilities ? capabilities.leds : LED_NAMES;
  const showLedCard = !capabilities || capabilities.leds.length > 0;
  const showBatteryCard = !capabilities || capabilities.battery;
  const showMic = !capabilities || capabilities.audio.mic;
  const showSpeaker = !capabilities || capabilities.audio.speaker;
  const showAudioCard = showMic || showSpeaker;
  const hwBadges = capabilities
    ? HW_BADGE_KEYS.filter((k) =>
        k === "gimbal" || k === "hdr"
          ? capabilities[k].present
          : capabilities[k]
      )
    : [];

  const [ledStates, setLedStates] = useState<Record<string, LedState>>({});
  const [battery, setBattery] = useState<number | null | undefined>(undefined); // undefined=loading, null=unavailable
  const [sensor, setSensor] = useState<ISensorStatus | null | undefined>(
    undefined
  );
  const [deviceTime, setDeviceTime] = useState<number | null>(null);
  const [timezone, setTimezone] = useState<string>("");
  const [powerLoading, setPowerLoading] = useState(false);
  const [browserSyncing, setBrowserSyncing] = useState(false);
  const [ntpSyncing, setNtpSyncing] = useState(false);
  // Audio card (P3-E). recordUrl: object URL of the last mic capture.
  const [recording, setRecording] = useState(false);
  const [recordUrl, setRecordUrl] = useState<string | null>(null);
  // Waveform of the last capture: normalized peak bars + silent flag.
  const [wavePeaks, setWavePeaks] = useState<number[] | null>(null);
  const [waveSilent, setWaveSilent] = useState(false);
  const [playTesting, setPlayTesting] = useState(false);
  const [playingRecording, setPlayingRecording] = useState(false);
  // undefined=loading, null=unsupported/unavailable
  const [audioVolume, setAudioVolume] = useState<
    IAudioVolume | null | undefined
  >(undefined);
  const recordUrlRef = useRef<string | null>(null);
  const volumeTimer = useRef<ReturnType<typeof setTimeout>>();
  // Camera picture orientation (camera.conf). undefined=loading,
  // null=endpoint unavailable (older firmware).
  const [cameraConf, setCameraConf] = useState<
    ICameraConfig | null | undefined
  >(undefined);
  const [cameraSaved, setCameraSaved] = useState<ICameraConfig | null>(null);
  const [cameraSaving, setCameraSaving] = useState(false);
  const cameraDirty =
    !!cameraConf &&
    !!cameraSaved &&
    (cameraConf.mirror !== cameraSaved.mirror ||
      cameraConf.flip !== cameraSaved.flip ||
      cameraConf.rotation !== cameraSaved.rotation);

  const fetchTime = () => {
    getTimestampApi()
      .then((res) => {
        if (isOk(res) && res.data?.timestamp) {
          setDeviceTime(res.data.timestamp * 1000);
        }
      })
      .catch(() => setDeviceTime(null));
    getTimezoneApi()
      .then((res) => {
        if (isOk(res)) {
          setTimezone(res.data?.timezone || "");
        }
      })
      .catch(() => setTimezone(""));
  };

  const fetchAll = async () => {
    setBattery(undefined);
    setSensor(undefined);
    // Battery
    queryBatteryInfoApi()
      .then((res) => {
        if (isOk(res) && res.data?.voltage) {
          setBattery(res.data.voltage);
        } else {
          setBattery(null);
        }
      })
      .catch(() => setBattery(null));
    // Temperature + storage (new endpoint, may not exist on older firmware)
    getSensorStatusApi()
      .then((res) => {
        if (isOk(res) && res.data) {
          setSensor(res.data);
        } else {
          setSensor(null);
        }
      })
      .catch(() => setSensor(null));
    fetchTime();
    // Camera picture orientation (older firmware returns 404)
    setCameraConf(undefined);
    getCameraConfigApi()
      .then((res) => {
        if (isOk(res) && res.data && typeof res.data.mirror === "boolean") {
          setCameraConf(res.data);
          setCameraSaved(res.data);
        } else {
          setCameraConf(null);
        }
      })
      .catch(() => setCameraConf(null));
    // Audio volume controls (P3-E endpoint; older firmware returns 404)
    setAudioVolume(undefined);
    getAudioVolumeApi()
      .then((res) => {
        if (isOk(res) && res.data?.supported && res.data.controls?.length) {
          setAudioVolume(res.data);
        } else {
          setAudioVolume(null);
        }
      })
      .catch(() => setAudioVolume(null));
  };

  useEffect(() => {
    fetchAll();
    // Release the last recording's object URL on unmount.
    return () => {
      if (recordUrlRef.current) URL.revokeObjectURL(recordUrlRef.current);
    };
  }, []);

  /** Record 3 s from the on-board mic, get the WAV back as a blob and feed
   *  it to an <audio> element. Backend enforces one audio op at a time
   *  (code -2 = busy). */
  const onRecord = async () => {
    setRecording(true);
    setWavePeaks(null);
    try {
      const blob = await audioRecordApi(3);
      if (recordUrlRef.current) URL.revokeObjectURL(recordUrlRef.current);
      const url = URL.createObjectURL(blob);
      recordUrlRef.current = url;
      setRecordUrl(url);
      try {
        const { peaks, silent } = await analyzeWaveform(blob);
        setWavePeaks(peaks);
        setWaveSilent(silent);
      } catch {
        setWavePeaks(null); // undecodable clip -> just skip the waveform
      }
    } catch (e) {
      if (isBusy((e as { code?: number | string }) ?? {})) {
        message.warning(t("audio.busy"));
      } else {
        message.error(t("audio.recordFailed"));
      }
    } finally {
      setRecording(false);
    }
  };

  /** Play the packaged test tone on the device speaker; the user confirms
   *  audibility at the device. */
  const onPlayTest = async () => {
    setPlayTesting(true);
    try {
      const res = await audioPlayTestApi();
      if (isOk(res)) {
        message.success(t("audio.playConfirm"));
      } else if (isBusy(res)) {
        message.warning(t("audio.busy"));
      } else {
        message.error(t("audio.playFailed"));
      }
    } catch (e) {
      message.error(t("audio.playFailed"));
    } finally {
      setPlayTesting(false);
    }
  };

  /** Play the last mic capture back on the device speaker (mic->speaker
   *  loopback check); the user confirms audibility at the device. Only enabled
   *  after a successful recording. */
  const onPlayRecording = async () => {
    setPlayingRecording(true);
    try {
      const res = await audioPlayRecordingApi();
      if (isOk(res)) {
        message.success(t("audio.playConfirm"));
      } else if (isBusy(res)) {
        message.warning(t("audio.busy"));
      } else {
        message.error(t("audio.playRecordingFailed"));
      }
    } catch (e) {
      message.error(t("audio.playRecordingFailed"));
    } finally {
      setPlayingRecording(false);
    }
  };

  /** Slider onChange: update UI immediately, debounce the backend set. */
  const onVolumeChange = (name: string, pct: number) => {
    setAudioVolume((v) =>
      v
        ? {
            ...v,
            controls: v.controls.map((c) =>
              c.name === name ? { ...c, percent: pct } : c
            ),
          }
        : v
    );
    if (volumeTimer.current) clearTimeout(volumeTimer.current);
    volumeTimer.current = setTimeout(async () => {
      try {
        const res = await setAudioVolumeApi(name, pct);
        if (!isOk(res)) {
          message.error(t("audio.volumeFailed"));
        }
      } catch (e) {
        message.error(t("audio.volumeFailed"));
      }
    }, 400);
  };

  /** Persist the camera orientation. The backend restarts the active app so
   *  the setting takes effect (same busy semantics as the HA config). */
  const onSaveCamera = async () => {
    if (!cameraConf) return;
    setCameraSaving(true);
    try {
      const res = await setCameraConfigApi({
        mirror: cameraConf.mirror,
        flip: cameraConf.flip,
        rotation: cameraConf.rotation,
      });
      if (isOk(res)) {
        const saved: ICameraConfig = {
          mirror: !!res.data?.mirror,
          flip: !!res.data?.flip,
          rotation: res.data?.rotation === 180 ? 180 : 0,
        };
        setCameraConf(saved);
        setCameraSaved(saved);
        message.success(
          res.data?.restarted
            ? `${t("camera.saved")} ${t("camera.restarted")}`
            : t("camera.saved")
        );
      } else if (isBusy(res)) {
        message.warning(t("camera.busy"));
      } else {
        message.error(res.msg || t("camera.saveFailed"));
      }
    } catch (e) {
      message.error(t("camera.saveFailed"));
    } finally {
      setCameraSaving(false);
    }
  };

  const onToggleLed = async (name: string, on: boolean) => {
    const prev = ledStates[name] ?? "unknown";
    setLedStates((s) => ({ ...s, [name]: on ? "on" : "off" }));
    try {
      const res = await setLedApi(name, on);
      if (!isOk(res)) {
        setLedStates((s) => ({ ...s, [name]: "unavailable" }));
        message.warning(t("device.ledUnavailableMsg", { name }));
      }
    } catch (e) {
      setLedStates((s) => ({
        ...s,
        [name]: prev === "unknown" ? "unavailable" : prev,
      }));
      message.error(t("device.ledSetFailed", { name }));
    }
  };

  /** Field-deployment path: copy the (correct) browser clock + timezone to
   *  the device. Works fully offline. Backend expects seconds. */
  const onSyncFromBrowser = async () => {
    setBrowserSyncing(true);
    try {
      const res = await setTimestampApi(Math.floor(Date.now() / 1000));
      if (!isOk(res)) {
        throw new Error(res.msg || "setTimestamp failed");
      }
      // IANA name, validated by the backend against /usr/share/zoneinfo.
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      if (tz) {
        try {
          const tzRes = await setTimezoneApi(tz);
          if (!isOk(tzRes)) {
            message.warning(t("device.timezoneNotApplied", { tz }));
          }
        } catch {
          message.warning(t("device.timezoneNotApplied", { tz }));
        }
      }
      message.success(t("device.syncSuccess"));
    } catch (e) {
      message.error(t("device.syncFailed"));
    } finally {
      setBrowserSyncing(false);
      fetchTime();
    }
  };

  /** NTP path (P3-B backend endpoint). Requires internet on the device;
   *  failure/404 must not block the page. */
  const onSyncNtp = async () => {
    setNtpSyncing(true);
    try {
      const res = await syncTimeNtpApi();
      if (isOk(res) && res.data?.timestamp) {
        message.success(t("device.syncSuccess"));
      } else {
        message.error(t("device.ntpFailed"));
      }
    } catch (e) {
      message.error(t("device.ntpFailed"));
    } finally {
      setNtpSyncing(false);
      fetchTime();
    }
  };

  const onPower = (mode: PowerMode) => {
    const isReboot = mode === PowerMode.Restart;
    Modal.confirm({
      title: isReboot ? t("device.rebootTitle") : t("device.shutdownTitle"),
      icon: <ExclamationCircleOutlined />,
      content: isReboot
        ? t("device.rebootContent")
        : t("device.shutdownContent"),
      okText: isReboot ? t("device.reboot") : t("device.shutdownOk"),
      okButtonProps: { danger: true },
      cancelText: t("common.cancel"),
      onOk: async () => {
        setPowerLoading(true);
        try {
          await setDevicePowerApi({ mode });
          message.success(
            isReboot ? t("device.rebooting") : t("device.shuttingDown")
          );
        } catch (e) {
          message.error(t("device.opFailed"));
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

  const clockNotSet =
    deviceTime !== null &&
    new Date(deviceTime).getFullYear() < CLOCK_SANE_YEAR;

  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">{t("common.console")}</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            {t("device.title")}
          </h1>
          <p className="text-muted text-13 mt-4 mb-0 rc-prose">
            {t("device.subtitle")}
          </p>
        </div>
        <div className="flex gap-8">
          <Button icon={<ReloadOutlined />} onClick={fetchAll}>
            {t("common.refresh")}
          </Button>
          <Dropdown
            menu={{
              items: [
                {
                  key: "reboot",
                  icon: <ReloadOutlined />,
                  label: t("device.reboot"),
                },
                {
                  key: "shutdown",
                  icon: <PoweroffOutlined />,
                  label: t("device.shutdown"),
                  danger: true,
                },
              ],
              onClick: ({ key }) =>
                onPower(
                  key === "reboot" ? PowerMode.Restart : PowerMode.Shutdown
                ),
            }}
            trigger={["click"]}
          >
            <Tooltip title={t("device.power")}>
              <Button icon={<PoweroffOutlined />} loading={powerLoading} />
            </Tooltip>
          </Dropdown>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-16 mt-24">
        {/* Hardware variant (P5): device_type + capability badges */}
        {capabilities && (
          <div className="rc-card p-20">
            <div className="rc-section-label mb-12">
              {t("capabilities.card")}
            </div>
            <div className="rc-kpi-label">{t("capabilities.deviceType")}</div>
            <div className="font-display font-semibold text-16 mt-2">
              {capabilities.device_type || "-"}
            </div>
            <div className="rc-mono text-11 text-muted mt-4">
              {t("capabilities.sensor", {
                sensor: capabilities.hdr.sensor || "unknown",
              })}
            </div>
            <div className="rc-kpi-label mt-14">
              {t("capabilities.optional")}
            </div>
            {hwBadges.length ? (
              <div className="flex flex-wrap gap-6 mt-6">
                {hwBadges.map((k) => (
                  <span key={k} className="rc-badge accent">
                    <span className="dot" />
                    {t(`capabilities.keys.${k}`)}
                  </span>
                ))}
              </div>
            ) : (
              <div className="text-13 text-muted mt-6">
                {t("capabilities.none")}
              </div>
            )}
            <div className="text-12 text-muted mt-10">
              {t("capabilities.hint")}
            </div>
          </div>
        )}

        {/* LED control — enumerated from capabilities.leds when available */}
        {showLedCard && (
          <div className="rc-card p-20">
            <div className="rc-section-label mb-12">
              {t("device.ledControl")}
            </div>
            <div className="flex flex-col">
              {ledNames.map((name, i) => (
                <div
                  key={name}
                  className={`flex items-center justify-between py-10 ${
                    i ? "border-t border-line" : ""
                  }`}
                >
                  <div>
                    <span className="text-14 font-medium">
                      {t(`device.led.${name}`, {
                        defaultValue: `${name} LED`,
                      })}
                    </span>
                    {ledStates[name] === "unavailable" && (
                      <span className="text-12 text-muted ml-8">
                        {t("device.ledNotAvailable")}
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
            <div className="text-12 text-muted mt-8">{t("device.ledHint")}</div>
          </div>
        )}

        {/* Battery — hidden entirely when the probe says no battery ADC */}
        {showBatteryCard && (
          <div className="rc-card p-20">
            <div className="rc-section-label mb-12">{t("device.battery")}</div>
            {battery === undefined ? (
              <div className="text-13 text-muted">{t("common.reading")}</div>
            ) : battery === null ? (
              <div className="text-13 text-muted">{t("device.noBattery")}</div>
            ) : (
              <div>
                <div className="rc-kpi-label">{t("device.voltage")}</div>
                <div className="rc-kpi-value">
                  {(battery / 1000).toFixed(2)}
                  <span className="text-14 font-medium text-muted ml-4">V</span>
                </div>
                <div className="rc-mono text-11 text-muted mt-2">
                  {t("device.rawMv", { mv: battery })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Temperature + storage */}
        <div className="rc-card p-20">
          <div className="rc-section-label mb-12">{t("device.health")}</div>
          {sensor === undefined ? (
            <div className="text-13 text-muted">{t("common.reading")}</div>
          ) : sensor === null ? (
            <div className="text-13 text-muted">
              {t("device.sensorUnavailable")}
            </div>
          ) : (
            <div className="flex flex-col gap-16">
              <div>
                <div className="rc-kpi-label">{t("device.socTemp")}</div>
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
                <div className="rc-kpi-label">{t("device.storageLabel")}</div>
                {storagePercent !== null ? (
                  <>
                    <Progress
                      percent={storagePercent}
                      size="small"
                      strokeColor="#8fc31f"
                    />
                    <div className="rc-mono text-11 text-muted">
                      {t("device.storageDetail", {
                        used: formatStorage(storage?.used),
                        total: formatStorage(storage?.total),
                        free: formatStorage(storage?.available),
                      })}
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
          <div className="flex items-center justify-between gap-8 mb-12">
            <div className="rc-section-label">{t("device.timeCard")}</div>
            {clockNotSet && (
              <span className="rc-badge accent">
                <span className="dot" />
                {t("device.clockNotSet")}
              </span>
            )}
          </div>
          <div className="rc-kpi-label">{t("device.deviceTime")}</div>
          <div className="rc-kpi-value" style={{ fontSize: 20 }}>
            {deviceTime ? new Date(deviceTime).toLocaleString() : "-"}
          </div>
          <div className="rc-mono text-11 text-muted mt-6">
            {t("device.timezone", { tz: timezone || "-" })}
          </div>
          <div className="flex gap-8 mt-14 flex-wrap">
            <Button
              size="small"
              type={clockNotSet ? "primary" : "default"}
              icon={<ClockCircleOutlined />}
              loading={browserSyncing}
              disabled={ntpSyncing}
              onClick={onSyncFromBrowser}
            >
              {t("device.syncFromBrowser")}
            </Button>
            <Button
              size="small"
              icon={<GlobalOutlined />}
              loading={ntpSyncing}
              disabled={browserSyncing}
              onClick={onSyncNtp}
            >
              {t("device.syncViaNtp")}
            </Button>
          </div>
          <div className="text-12 text-muted mt-10">
            {t("device.timeHint")}
          </div>
        </div>

        {/* Audio (P3-E, P5-gated): hidden without mic AND speaker; a single
            present side renders only its own half. */}
        {showAudioCard && (
          <div className="rc-card p-20">
            <div className="rc-section-label mb-12">{t("audio.card")}</div>

            {showMic && (
              <>
                <div className="rc-kpi-label">{t("audio.microphone")}</div>
                <div className="flex gap-8 mt-6 flex-wrap items-center">
                  <Button
                    size="small"
                    icon={<AudioOutlined />}
                    loading={recording}
                    disabled={playTesting}
                    onClick={onRecord}
                  >
                    {recording ? t("audio.recording") : t("audio.recordBtn")}
                  </Button>
                  {recordUrl && (
                    <Button
                      size="small"
                      icon={<DownloadOutlined />}
                      href={recordUrl}
                      download="audio_probe.wav"
                    >
                      {t("audio.download")}
                    </Button>
                  )}
                </div>
                {wavePeaks && (
                  <div className="mt-10">
                    <svg
                      viewBox={`0 0 ${wavePeaks.length * 4} 40`}
                      preserveAspectRatio="none"
                      className="w-full"
                      style={{ height: 40 }}
                      role="img"
                      aria-label={
                        waveSilent ? t("audio.silent") : t("audio.captured")
                      }
                    >
                      {wavePeaks.map((p, i) => {
                        const h = Math.max(2, p * 38);
                        return (
                          <rect
                            key={i}
                            x={i * 4}
                            y={(40 - h) / 2}
                            width={2.4}
                            height={h}
                            rx={1}
                            fill={waveSilent ? "#d48806" : "#52c41a"}
                          />
                        );
                      })}
                    </svg>
                    <div
                      className="text-11 mt-4 text-muted"
                      style={waveSilent ? { color: "#d48806" } : undefined}
                    >
                      {waveSilent ? t("audio.silent") : t("audio.captured")}
                    </div>
                  </div>
                )}
                {recordUrl && (
                  <audio
                    controls
                    src={recordUrl}
                    className="w-full mt-10"
                    style={{ height: 32 }}
                  />
                )}
              </>
            )}

            {showSpeaker && (
              <>
                <div className={`rc-kpi-label ${showMic ? "mt-16" : ""}`}>
                  {t("audio.speaker")}
                </div>
                <div className="mt-6 flex gap-8">
                  <Button
                    size="small"
                    icon={<SoundOutlined />}
                    loading={playTesting}
                    disabled={recording || playingRecording}
                    onClick={onPlayTest}
                  >
                    {t("audio.playBtn")}
                  </Button>
                  {showMic && (
                    <Button
                      size="small"
                      icon={<SoundOutlined />}
                      loading={playingRecording}
                      disabled={recording || playTesting || !recordUrl}
                      onClick={onPlayRecording}
                    >
                      {t("audio.playRecording")}
                    </Button>
                  )}
                </div>
                {showMic && (
                  <div className="text-11 text-muted mt-6">
                    {t("audio.playHint")}
                  </div>
                )}
              </>
            )}

            {audioVolume && audioVolume.controls.length > 0 && (
              <>
                <div className="rc-kpi-label mt-16">{t("audio.volume")}</div>
                {audioVolume.controls.map((c) => (
                  <div key={c.name} className="mt-4">
                    <div className="rc-mono text-11 text-muted">
                      {c.name} · {c.percent}%
                    </div>
                    <Slider
                      min={0}
                      max={100}
                      value={c.percent}
                      onChange={(v) => onVolumeChange(c.name, v)}
                    />
                  </div>
                ))}
              </>
            )}

            <div className="text-12 text-muted mt-10">
              {showMic ? t("audio.micHint") : t("audio.speakerOnlyHint")}
            </div>
          </div>
        )}

        {/* Camera picture orientation — hidden when the endpoint is missing
            (older firmware). Saving restarts the active application. */}
        {cameraConf !== null && (
          <div className="rc-card p-20">
            <div className="rc-section-label mb-12">{t("camera.card")}</div>
            {cameraConf === undefined ? (
              <div className="text-13 text-muted">{t("common.reading")}</div>
            ) : (
              <>
                <div className="flex items-center justify-between py-10">
                  <div>
                    <span className="text-14 font-medium">
                      {t("camera.mirror")}
                    </span>
                    <div className="text-12 text-muted mt-2">
                      {t("camera.mirrorHint")}
                    </div>
                  </div>
                  <Switch
                    size="small"
                    checked={cameraConf.mirror}
                    onChange={(v) =>
                      setCameraConf({ ...cameraConf, mirror: v })
                    }
                  />
                </div>
                <div className="flex items-center justify-between py-10 border-t border-line">
                  <div>
                    <span className="text-14 font-medium">
                      {t("camera.flip")}
                    </span>
                    <div className="text-12 text-muted mt-2">
                      {t("camera.flipHint")}
                    </div>
                  </div>
                  <Switch
                    size="small"
                    checked={cameraConf.flip}
                    onChange={(v) => setCameraConf({ ...cameraConf, flip: v })}
                  />
                </div>
                <div className="flex items-center justify-between py-10 border-t border-line">
                  <span className="text-14 font-medium">
                    {t("camera.rotation")}
                  </span>
                  <Segmented
                    size="small"
                    value={cameraConf.rotation}
                    onChange={(v) =>
                      setCameraConf({
                        ...cameraConf,
                        rotation: v === 180 ? 180 : 0,
                      })
                    }
                    options={[
                      { label: "0°", value: 0 },
                      { label: "180°", value: 180 },
                    ]}
                  />
                </div>
                <Button
                  size="small"
                  type="primary"
                  icon={<VideoCameraOutlined />}
                  className="mt-10"
                  loading={cameraSaving}
                  disabled={!cameraDirty}
                  onClick={onSaveCamera}
                >
                  {t("camera.save")}
                </Button>
                <div className="text-12 text-muted mt-10">
                  {t("camera.hint")}
                </div>
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
};

export default DeviceTools;
