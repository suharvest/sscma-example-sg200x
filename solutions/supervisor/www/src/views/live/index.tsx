import { useEffect, useMemo, useRef, useState } from "react";
import { App, Button, Collapse, Segmented, Select, Spin, Switch } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { getCurrentAppApi, setAppModelApi } from "@/api/app";
import {
  getFocusValueApi,
  getCameraConfigApi,
  setCameraConfigApi,
} from "@/api/camera";
import { ICameraConfig, IGetFocusValueResult } from "@/api/camera/camera";
import { IAppInfo, IConfigItem } from "@/api/app/app";
import { SchemaForm, SpatialEditor, useAppConfig } from "@/components/app-config";
import useDebugStream, { IOverlayFrame } from "@/hooks/useDebugStream";
import { isOk, isBusy } from "@/utils/api";
import { copyText } from "@/utils/clipboard";
import {
  resolveRtspUrl,
  resolveDebugVideoUrl,
  resolveDebugResultsUrl,
} from "@/utils/appStream";
import { pickLocalized, pickLocalizedAlt } from "@/utils/appLocale";
import IntegrationDoc from "@/components/integration-doc";
import useCapabilitiesStore from "@/store/capabilities";

interface ContentRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Bounding-box overlay. Coordinates follow the SSCMA convention:
 *  x,y = box center, relative to the inference resolution (pixels);
 *  values <= 1 are treated as already-normalized. */
function BoxOverlay({
  frame,
  rect,
}: {
  frame: IOverlayFrame;
  rect: ContentRect;
}) {
  if (!rect.width || !rect.height) return null;
  const { resW, resH, boxes } = frame;
  return (
    <svg
      className="absolute pointer-events-none"
      style={{
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
      }}
      viewBox={`0 0 ${resW} ${resH}`}
      preserveAspectRatio="none"
    >
      {boxes.map((b, i) => {
        const normalized = b.x <= 1 && b.y <= 1 && b.w <= 1 && b.h <= 1;
        const x = normalized ? b.x * resW : b.x;
        const y = normalized ? b.y * resH : b.y;
        const w = normalized ? b.w * resW : b.w;
        const h = normalized ? b.h * resH : b.h;
        const left = x - w / 2;
        const top = y - h / 2;
        const pct = (b.score <= 1 ? b.score * 100 : b.score).toFixed(0);
        // Prefer the app's rich label (e.g. "male 20-29 neutral") over the
        // bare box target ("face"); append the detection score.
        const label = b.label
          ? `${b.label} ${pct}%`
          : b.target !== undefined && b.target !== null
          ? `${b.target} ${pct}%`
          : "";
        return (
          <g key={i}>
            <rect
              x={left}
              y={top}
              width={w}
              height={h}
              fill="none"
              stroke="#8fc31f"
              strokeWidth={Math.max(2, resW / 320)}
            />
            {label && (
              <text
                x={left + 4}
                y={Math.max(top - 6, 12)}
                fill="#8fc31f"
                fontSize={Math.max(12, resW / 40)}
                fontFamily="SFMono-Regular, Menlo, monospace"
                paintOrder="stroke"
                stroke="rgba(255,255,255,0.85)"
                strokeWidth={3}
              >
                {label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

const Live = () => {
  const { t } = useTranslation();
  const { modal, message } = App.useApp();
  // P5: gimbal variants get a placeholder control card (actual motor
  // control lands in Phase 6, gated on gimbal hardware for development).
  const gimbalPresent = useCapabilitiesStore(
    (s) => s.capabilities?.gimbal.present ?? false
  );
  const [loading, setLoading] = useState(true);
  const [app, setApp] = useState<IAppInfo | null>(null);
  const [appStatus, setAppStatus] = useState<string>("");
  const [currentModel, setCurrentModel] = useState<string | undefined>();
  const [debugOn, setDebugOn] = useState(false);
  const [overlayOn, setOverlayOn] = useState(true);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [contentRect, setContentRect] = useState<ContentRect>({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Focus assist: while enabled, poll getFocusValue at 1s and draw a bar of
  // the current sharpness score with the session peak marked. null = no
  // sample yet.
  const [focusOn, setFocusOn] = useState(false);
  const [focus, setFocus] = useState<IGetFocusValueResult | null>(null);
  const [focusPeak, setFocusPeak] = useState(0);
  useEffect(() => {
    if (!focusOn) {
      setFocus(null);
      setFocusPeak(0);
      return;
    }
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await getFocusValueApi();
        if (cancelled) return;
        if (isOk(res) && res.data) {
          setFocus(res.data);
          if (res.data.available && typeof res.data.fv === "number") {
            const v = res.data.fv;
            setFocusPeak((p) => (v > p ? v : p));
          }
        } else {
          setFocus({ available: false });
        }
      } catch (e) {
        if (!cancelled) setFocus({ available: false });
      }
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [focusOn]);

  const focusValue =
    focus?.available && typeof focus.fv === "number" ? focus.fv : null;
  // Relative score with no fixed range: scale against the session peak so
  // the bar always reads "how close to the sharpest seen so far".
  const focusScale = Math.max(focusPeak, focusValue ?? 0, 1e-6);
  const focusPct =
    focusValue !== null
      ? Math.min((focusValue / focusScale) * 100, 100)
      : 0;
  const focusPeakPct =
    focusPeak > 0 ? Math.min((focusPeak / focusScale) * 100, 100) : 0;

  // Camera picture orientation (shared with the Device page; each page pulls
  // its own copy on mount). undefined=loading, null=endpoint unavailable
  // (older firmware) — the panel is hidden in that case.
  const [camConf, setCamConf] = useState<ICameraConfig | null | undefined>(
    undefined
  );
  const [camSaved, setCamSaved] = useState<ICameraConfig | null>(null);
  const [camSaving, setCamSaving] = useState(false);
  const camDirty =
    !!camConf &&
    !!camSaved &&
    (camConf.mirror !== camSaved.mirror ||
      camConf.flip !== camSaved.flip ||
      camConf.rotation !== camSaved.rotation);
  useEffect(() => {
    getCameraConfigApi()
      .then((res) => {
        if (isOk(res) && res.data && typeof res.data.mirror === "boolean") {
          setCamConf(res.data);
          setCamSaved(res.data);
        } else {
          setCamConf(null);
        }
      })
      .catch(() => setCamConf(null));
  }, []);

  /** Persist the camera orientation. The change is hot-applied by the video
   *  pipeline within ~1-2 s — no app restart, so the debug stream stays up. */
  const onSaveCamera = async () => {
    if (!camConf) return;
    setCamSaving(true);
    try {
      const res = await setCameraConfigApi({
        mirror: camConf.mirror,
        flip: camConf.flip,
        rotation: camConf.rotation,
      });
      if (isOk(res)) {
        const saved: ICameraConfig = {
          mirror: !!res.data?.mirror,
          flip: !!res.data?.flip,
          rotation: res.data?.rotation === 180 ? 180 : 0,
        };
        setCamConf(saved);
        setCamSaved(saved);
        message.success(
          res.data?.restarted
            ? `${t("camera.saved")} ${t("camera.restarted")}`
            : t("camera.saved")
        );
        fetchCurrent();
      } else if (isBusy(res)) {
        message.warning(t("camera.busy"));
      } else {
        message.error(res.msg || t("camera.saveFailed"));
      }
    } catch (e) {
      message.error(t("camera.saveFailed"));
    } finally {
      setCamSaving(false);
    }
  };

  // App configuration (manifest config_schema) — schema null hides the card.
  const appConfig = useAppConfig(app?.id);
  // Spatial item (zone/line) currently being edited on the video, if any.
  const [editingItem, setEditingItem] = useState<IConfigItem | null>(null);
  useEffect(() => {
    setEditingItem(null);
  }, [app?.id]);

  const copy = (text: string) =>
    copyText(text, t("common.copied"), t("common.copyFailed"));

  const videoUrl = useMemo(() => resolveDebugVideoUrl(app), [app]);
  const resultsUrl = useMemo(() => resolveDebugResultsUrl(app), [app]);
  const rtspUrl = useMemo(() => resolveRtspUrl(app), [app]);
  const hasDebugWs = !!videoUrl;

  const { status, messages, overlay, lastFrameDelay } = useDebugStream({
    enabled: debugOn && hasDebugWs,
    wsUrl: videoUrl,
    resultsUrl,
    videoRef,
  });

  const fetchCurrent = async () => {
    setLoading(true);
    try {
      const res = await getCurrentAppApi();
      const current = (isOk(res) && (res.data?.app as IAppInfo)) || null;
      if (current?.id) {
        setApp(current);
        // Backend /current returns `probe` (live init-script status) and
        // `state` (state machine); prefer the live probe.
        setAppStatus(String(res.data.probe || res.data.state || ""));
        setCurrentModel(
          res.data.current_model ||
            current.current_model ||
            current.default_model
        );
      } else {
        setApp(null);
      }
    } catch (e) {
      setApp(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCurrent();
  }, []);

  // Track the actual displayed video content area (object-fit: contain).
  useEffect(() => {
    const video = videoRef.current;
    const container = containerRef.current;
    if (!video || !container) return;

    const update = () => {
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!cw || !ch) return;
      if (!vw || !vh) {
        setContentRect({ left: 0, top: 0, width: cw, height: ch });
        return;
      }
      const scale = Math.min(cw / vw, ch / vh);
      const width = vw * scale;
      const height = vh * scale;
      setContentRect({
        left: (cw - width) / 2,
        top: (ch - height) / 2,
        width,
        height,
      });
    };

    update();
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
      observer.disconnect();
    };
  }, [debugOn]);

  const onModelChange = (model: string) => {
    if (!app || model === currentModel) return;
    modal.confirm({
      title: t("live.switchModelTitle", { model }),
      icon: <ExclamationCircleOutlined />,
      content: t("live.switchModelContent"),
      okText: t("common.confirm"),
      cancelText: t("common.cancel"),
      onOk: async () => {
        setModelSwitching(true);
        setDebugOn(false);
        try {
          const res = await setAppModelApi({ app_id: app.id, model });
          if (isOk(res)) {
            setCurrentModel(model);
            message.success(t("live.modelUpdated"));
          } else {
            message.error(res.msg || t("live.switchModelFailed"));
          }
        } catch (e) {
          message.error(t("live.switchModelFailed"));
        } finally {
          setModelSwitching(false);
          fetchCurrent();
        }
      },
    });
  };

  // Save the config draft: confirm (the active app restarts to apply it),
  // drop the debug stream first (the restart would kill it mid-frame), then
  // refresh the app status on success.
  const onSaveConfig = () => {
    modal.confirm({
      title: t("config.saveTitle"),
      icon: <ExclamationCircleOutlined />,
      content: t("config.saveContent"),
      okText: t("common.confirm"),
      cancelText: t("common.cancel"),
      onOk: async () => {
        setEditingItem(null);
        setDebugOn(false);
        const ok = await appConfig.save();
        if (ok) fetchCurrent();
      },
    });
  };

  const running = ["RUNNING", "STARTING"].includes(
    (appStatus || "").toUpperCase()
  );

  const rtspFallbackCard = (
    <div className="rc-card-surface p-24 max-w-[440px] text-left">
      <div className="rc-section-label mb-8">{t("live.fallbackTitle")}</div>
      <div className="text-13 text-muted mb-12">
        {hasDebugWs
          ? t("live.fallbackDebugUnavailable")
          : t("live.fallbackNoDebug")}
      </div>
      {rtspUrl ? (
        <div className="bg-white border border-line rounded-8 px-12 py-8 flex items-center justify-between gap-8">
          <span className="rc-mono text-12 break-all">{rtspUrl}</span>
          <Button size="small" onClick={() => copy(rtspUrl)}>
            {t("common.copy")}
          </Button>
        </div>
      ) : (
        <div className="text-13 text-muted">{t("live.fallbackNoRtsp")}</div>
      )}
      <div className="text-12 text-muted mt-12">{t("live.vlcHint")}</div>
    </div>
  );

  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">{t("common.console")}</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            {t("live.title")}
          </h1>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchCurrent}
          loading={loading}
        >
          {t("common.refresh")}
        </Button>
      </div>

      <Spin spinning={loading}>
        {!app ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <VideoCameraOutlined style={{ fontSize: 28, color: "#666" }} />
            <div className="mt-12 font-medium">{t("live.noActive")}</div>
            <div className="text-muted text-13 mt-4 mb-16">
              {t("live.noActiveHint")}
            </div>
            <Button type="primary" onClick={() => navigate("/")}>
              {t("live.goToApps")}
            </Button>
          </div>
        ) : (
          <>
            {/* Current app info bar */}
            <div className="rc-card mt-24 px-20 py-14 flex items-center justify-between gap-12 flex-wrap">
              <div className="flex items-center gap-12 flex-wrap">
                <span className="font-display font-semibold text-15">
                  {pickLocalized(app, "name") || app.id}
                </span>
                {pickLocalizedAlt(app, "name") && (
                  <span className="text-muted text-13">
                    {pickLocalizedAlt(app, "name")}
                  </span>
                )}
                <span className={`rc-badge ${running ? "accent" : ""}`}>
                  <span
                    className="dot"
                    style={{ background: running ? "#8fc31f" : "#bbbbbb" }}
                  />
                  {appStatus
                    ? t(`apps.status.${appStatus.toLowerCase()}`, {
                        defaultValue: appStatus.toLowerCase(),
                      })
                    : t("common.unknown")}
                </span>
              </div>
              {currentModel && (
                <span className="rc-section-label">
                  {t("live.modelLabel", { model: currentModel })}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-16 mt-16">
              {/* Player */}
              <div className="lg:col-span-3">
                <div
                  ref={containerRef}
                  className="relative w-full bg-black rounded-12 overflow-hidden"
                  style={{ aspectRatio: "16 / 9" }}
                >
                  {debugOn && hasDebugWs && status !== "error" ? (
                    <>
                      <video
                        ref={videoRef}
                        className="absolute inset-0 w-full h-full object-contain"
                        autoPlay
                        muted
                        playsInline
                      />
                      {overlayOn && overlay && (
                        <BoxOverlay frame={overlay} rect={contentRect} />
                      )}
                      {status === "connecting" && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Spin />
                          <span className="text-white text-13 ml-12 opacity-80">
                            {t("live.connecting")}
                          </span>
                        </div>
                      )}
                      {status === "streaming" && lastFrameDelay !== null && (
                        <div className="absolute right-8 top-8 rc-mono text-11 text-white bg-black bg-opacity-50 px-8 py-2 rounded-6">
                          delay ~{Math.max(lastFrameDelay, 0)}ms
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center p-16">
                      {(!hasDebugWs || status === "error") && debugOn ? (
                        rtspFallbackCard
                      ) : !debugOn ? (
                        <div className="text-center">
                          <VideoCameraOutlined
                            style={{ fontSize: 32, color: "#555" }}
                          />
                          <div className="text-white opacity-60 text-13 mt-8">
                            {t("live.debugOffHint")}
                          </div>
                        </div>
                      ) : (
                        rtspFallbackCard
                      )}
                    </div>
                  )}

                  {/* Zone/line spatial editor overlay (config_schema) */}
                  {editingItem && (
                    <SpatialEditor
                      key={editingItem.key}
                      item={editingItem}
                      value={
                        Object.prototype.hasOwnProperty.call(
                          appConfig.draft,
                          editingItem.key
                        )
                          ? appConfig.draft[editingItem.key]
                          : appConfig.defaults[editingItem.key]
                      }
                      contentRect={contentRect}
                      streaming={
                        debugOn && hasDebugWs && status === "streaming"
                      }
                      onDone={(v) => {
                        appConfig.setValue(editingItem.key, v);
                        setEditingItem(null);
                      }}
                      onCancel={() => setEditingItem(null)}
                    />
                  )}
                </div>
              </div>

              {/* Control panel — collapsible accordion so the right column
                  stays short and the video reads as the primary element. */}
              <div>
                <Collapse
                  className="rc-live-panel"
                  defaultActiveKey={["debug"]}
                  expandIconPosition="end"
                  items={[
                    {
                      key: "debug",
                      label: (
                        <span className="rc-section-label">
                          {t("live.debugControls")}
                        </span>
                      ),
                      children: (
                        <>
                          <div className="flex items-center justify-between gap-12">
                            <div>
                              <div className="text-14 font-medium">
                                {t("live.debugStream")}
                              </div>
                              <div className="text-12 text-muted mt-2">
                                {t("live.debugStreamHint")}
                              </div>
                            </div>
                            <Switch
                              checked={debugOn}
                              disabled={!app}
                              onChange={setDebugOn}
                            />
                          </div>
                          <div className="flex items-center justify-between gap-12 mt-16 pt-16 border-t border-line">
                            <div>
                              <div className="text-14 font-medium">
                                {t("live.resultOverlay")}
                              </div>
                              <div className="text-12 text-muted mt-2">
                                {t("live.resultOverlayHint")}
                              </div>
                            </div>
                            <Switch
                              checked={overlayOn}
                              disabled={!debugOn}
                              onChange={setOverlayOn}
                            />
                          </div>
                          {!!app.models && app.models.length >= 2 && (
                            <div className="mt-16 pt-16 border-t border-line">
                              <div className="text-14 font-medium mb-8">
                                {t("live.model")}
                              </div>
                              <Select
                                className="w-full"
                                value={currentModel}
                                loading={modelSwitching}
                                disabled={modelSwitching}
                                onChange={onModelChange}
                                options={app.models.map((m) => ({
                                  value: m.name,
                                  label: m.task
                                    ? `${m.name} (${m.task})`
                                    : m.name,
                                }))}
                              />
                              <div className="text-12 text-muted mt-6">
                                {t("live.switchRestarts")}
                              </div>
                            </div>
                          )}
                        </>
                      ),
                    },
                    {
                      key: "focus",
                      label: (
                        <span className="rc-section-label">
                          {t("focus.title")}
                        </span>
                      ),
                      children: (
                        <>
                          <div className="flex items-center justify-between gap-12">
                            <div>
                              <div className="text-14 font-medium">
                                {t("focus.enable")}
                              </div>
                              <div className="text-12 text-muted mt-2">
                                {t("focus.enableHint")}
                              </div>
                            </div>
                            <Switch checked={focusOn} onChange={setFocusOn} />
                          </div>
                          {focusOn && (
                            <div className="mt-16 pt-16 border-t border-line">
                              {focus === null ? (
                                <div className="text-13 text-muted">
                                  {t("focus.waiting")}
                                </div>
                              ) : !focus.available ? (
                                <div className="text-13 text-muted">
                                  {t("focus.notRunning")}
                                </div>
                              ) : (
                                <>
                                  <div className="flex items-baseline justify-between gap-8">
                                    <span className="text-12 text-muted">
                                      {t("focus.score")}
                                    </span>
                                    <span className="rc-mono text-12">
                                      {focusValue !== null
                                        ? Math.round(focusValue)
                                        : "-"}
                                    </span>
                                  </div>
                                  {/* Score bar + session-peak marker */}
                                  <div
                                    className="relative w-full mt-8 rounded-6 overflow-hidden"
                                    style={{
                                      height: 10,
                                      background: "rgba(0,0,0,0.08)",
                                    }}
                                  >
                                    <div
                                      className="absolute left-0 top-0 h-full rounded-6"
                                      style={{
                                        width: `${focusPct}%`,
                                        background: "#8fc31f",
                                        transition: "width 0.4s ease",
                                      }}
                                    />
                                    {focusPeak > 0 && (
                                      <div
                                        className="absolute top-0 h-full"
                                        style={{
                                          left: `calc(${focusPeakPct}% - 1px)`,
                                          width: 2,
                                          background: "#4a6510",
                                        }}
                                      />
                                    )}
                                  </div>
                                  <div className="rc-mono text-11 text-muted mt-6">
                                    {t("focus.peak", {
                                      peak: Math.round(focusPeak),
                                    })}
                                  </div>
                                  <div className="text-12 text-muted mt-8">
                                    {t("focus.hint")}
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                        </>
                      ),
                    },
                    // Camera picture orientation — hidden when the endpoint
                    // is missing (older firmware). Compact version of the
                    // Device page card; saving restarts the active app so the
                    // debug stream is dropped first.
                    ...(camConf !== null
                      ? [
                          {
                            key: "camera",
                            label: (
                              <span className="flex items-center gap-8">
                                <span className="rc-section-label">
                                  {t("camera.card")}
                                </span>
                                {camDirty && (
                                  <span className="rc-badge accent">
                                    {t("config.unsaved")}
                                  </span>
                                )}
                              </span>
                            ),
                            children:
                              camConf === undefined ? (
                                <div className="text-13 text-muted">
                                  {t("common.reading")}
                                </div>
                              ) : (
                                <>
                                  <div className="flex items-center justify-between gap-12">
                                    <div className="text-14 font-medium">
                                      {t("camera.mirror")}
                                    </div>
                                    <Switch
                                      size="small"
                                      checked={camConf.mirror}
                                      onChange={(v) =>
                                        setCamConf({ ...camConf, mirror: v })
                                      }
                                    />
                                  </div>
                                  <div className="flex items-center justify-between gap-12 mt-12 pt-12 border-t border-line">
                                    <div className="text-14 font-medium">
                                      {t("camera.flip")}
                                    </div>
                                    <Switch
                                      size="small"
                                      checked={camConf.flip}
                                      onChange={(v) =>
                                        setCamConf({ ...camConf, flip: v })
                                      }
                                    />
                                  </div>
                                  <div className="flex items-center justify-between gap-12 mt-12 pt-12 border-t border-line">
                                    <div className="text-14 font-medium">
                                      {t("camera.rotation")}
                                    </div>
                                    <Segmented
                                      size="small"
                                      value={camConf.rotation}
                                      onChange={(v) =>
                                        setCamConf({
                                          ...camConf,
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
                                    className="mt-12"
                                    loading={camSaving}
                                    disabled={!camDirty}
                                    onClick={onSaveCamera}
                                  >
                                    {t("camera.save")}
                                  </Button>
                                  <div className="text-12 text-muted mt-8">
                                    {t("camera.liveHint")}
                                  </div>
                                </>
                              ),
                          },
                        ]
                      : []),
                    ...(appConfig.schema
                      ? [
                          {
                            key: "config",
                            label: (
                              <span className="flex items-center gap-8">
                                <span className="rc-section-label">
                                  {t("config.title")}
                                </span>
                                {appConfig.dirty && (
                                  <span className="rc-badge accent">
                                    {t("config.unsaved")}
                                  </span>
                                )}
                              </span>
                            ),
                            children: (
                              <SchemaForm
                                embedded
                                schema={appConfig.schema}
                                draft={appConfig.draft}
                                defaults={appConfig.defaults}
                                dirty={appConfig.dirty}
                                saving={appConfig.saving}
                                editingKey={editingItem?.key ?? null}
                                onChange={appConfig.setValue}
                                onEditSpatial={(item) => setEditingItem(item)}
                                onSave={onSaveConfig}
                                onReset={() => {
                                  setEditingItem(null);
                                  appConfig.reset();
                                }}
                              />
                            ),
                          },
                        ]
                      : []),
                    {
                      key: "endpoints",
                      label: (
                        <span className="rc-section-label">
                          {t("live.endpoints")}
                        </span>
                      ),
                      children: (
                        <>
                          <div className="mb-12">
                            <div className="text-12 text-muted mb-4">
                              {t("live.rtspStream")}
                            </div>
                            {rtspUrl ? (
                              <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                                <span className="rc-mono text-12 break-all">
                                  {rtspUrl}
                                </span>
                                <Button
                                  size="small"
                                  onClick={() => copy(rtspUrl)}
                                >
                                  {t("common.copy")}
                                </Button>
                              </div>
                            ) : (
                              <div className="text-12 text-muted">
                                {t("common.notDeclared")}
                              </div>
                            )}
                          </div>
                          <div>
                            <div className="text-12 text-muted mb-4">
                              {t("live.mqttTopic")}
                            </div>
                            {app.mqtt_topic ? (
                              <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                                <span className="rc-mono text-12 break-all">
                                  {app.mqtt_topic}
                                </span>
                                <Button
                                  size="small"
                                  onClick={() => copy(app.mqtt_topic || "")}
                                >
                                  {t("common.copy")}
                                </Button>
                              </div>
                            ) : (
                              <div className="text-12 text-muted">
                                {t("common.notDeclared")}
                              </div>
                            )}
                          </div>
                        </>
                      ),
                    },
                    ...(app.pipeline?.length
                      ? [
                          {
                            key: "pipeline",
                            label: (
                              <span className="rc-section-label">
                                {t("live.pipelineFixed")}
                              </span>
                            ),
                            children: (
                              <>
                                <div className="flex flex-col gap-6">
                                  {app.pipeline.map((m, idx) => (
                                    <div
                                      key={m.name}
                                      className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8"
                                    >
                                      <span className="rc-mono text-12">
                                        {idx + 1}. {m.name}
                                      </span>
                                      <span className="rc-badge">
                                        {m.task || t("common.model")}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                                <div className="text-12 text-muted mt-8">
                                  {t("live.pipelineHint")}
                                </div>
                              </>
                            ),
                          },
                        ]
                      : []),
                    ...(gimbalPresent
                      ? [
                          {
                            key: "gimbal",
                            label: (
                              <span className="flex items-center gap-8">
                                <span className="rc-section-label">
                                  {t("live.gimbalTitle")}
                                </span>
                                <span className="rc-badge">
                                  {t("live.gimbalSoon")}
                                </span>
                              </span>
                            ),
                            children: (
                              <div className="text-13 text-muted">
                                {t("live.gimbalPlaceholder")}
                              </div>
                            ),
                          },
                        ]
                      : []),
                  ]}
                />
              </div>
            </div>

            {/* Recent messages */}
            <div className="rc-card mt-16">
              <div className="flex items-baseline justify-between px-20 py-14 border-b border-line">
                <span className="rc-section-label">
                  {t("live.recentMessages")}
                </span>
                <span className="rc-mono text-11 text-muted">
                  {messages.length
                    ? `${messages.length} / 100`
                    : t("live.liveTag")}
                </span>
              </div>
              {messages.length ? (
                <div className="max-h-[320px] overflow-y-auto">
                  {messages.map((m, i) => (
                    <div
                      key={`${m.receivedAt}-${i}`}
                      className="px-20 py-8 border-t border-line first:border-t-0 flex gap-12 items-baseline"
                    >
                      <span className="rc-mono text-11 text-muted flex-none">
                        {m.receivedAt}
                      </span>
                      <span className="rc-mono text-11 break-all whitespace-pre-wrap">
                        {m.raw.length > 500 ? `${m.raw.slice(0, 500)}…` : m.raw}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="px-20 py-32 text-center text-13 text-muted">
                  {debugOn
                    ? t("live.noMessages")
                    : t("live.enableDebugHint")}
                </div>
              )}
            </div>

            {/* Integration / output-format documentation */}
            <IntegrationDoc appId={app.id} className="mt-16" />
          </>
        )}
      </Spin>
    </div>
  );
};

export default Live;
