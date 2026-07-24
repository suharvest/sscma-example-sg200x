import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { App, Button, Collapse, Select, Spin, Switch, Tooltip } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  VideoCameraOutlined,
  SwapOutlined,
  SyncOutlined,
  AimOutlined,
  WarningOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { getCurrentAppApi, setAppModelApi } from "@/api/app";
import {
  getFocusValueApi,
  getCameraConfigApi,
  setCameraConfigApi,
} from "@/api/camera";
import {
  ICameraConfig,
  IGetFocusValueResult,
} from "@/api/camera/camera";
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
import { pickLocalized } from "@/utils/appLocale";
import { getAppTags } from "@/utils/appTags";
import IntegrationDoc from "@/components/integration-doc";
import useCapabilitiesStore from "@/store/capabilities";
// Shared, app-agnostic overlay renderer — single source of truth in
// app_collaboration/frontend/src/overlay/renderer.js, vendored here (see that
// file's header for the sync procedure). Same renderer the Tauri preview and
// the sensecraft draw_*.js adapters use, so boxes/cards look identical
// everywhere.
import { render as renderOverlay } from "@/vendor/recamera-overlay";

interface ContentRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Map a debug-stream overlay frame to a shared-renderer `model`.
 *  - boxes: SSCMA center coords (inference px, or already-normalized when <=1)
 *    → normalized TOP-LEFT bbox the contract expects.
 *  - classification → a `card:classification` layer.
 *  Covers every box app uniformly (yolo / face / retail / ocr / facemesh —
 *  facemesh's debug envelope is center boxes with a "state EAR" label). */
function adaptDebugFrame(frame: IOverlayFrame) {
  const layers: Array<Record<string, unknown>> = [];
  const { resW, resH, boxes, classification, qrcodes } = frame;
  if (boxes && boxes.length) {
    const items = boxes.map((b) => {
      const normalized = b.x <= 1 && b.y <= 1 && b.w <= 1 && b.h <= 1;
      const cx = normalized ? b.x : b.x / resW;
      const cy = normalized ? b.y : b.y / resH;
      const w = normalized ? b.w : b.w / resW;
      const h = normalized ? b.h : b.h / resH;
      const label =
        b.label ??
        (b.target !== undefined && b.target !== null
          ? String(b.target)
          : undefined);
      return { bbox: { x: cx - w / 2, y: cy - h / 2, w, h }, label, confidence: b.score };
    });
    layers.push({ type: "boxes", items });
  }
  if (classification) {
    const scores: Record<string, number> = {};
    for (const s of classification.scores) scores[s.name] = s.score;
    layers.push({
      type: "card",
      variant: "classification",
      anchor: "tl",
      data: {
        label: classification.label,
        confidence: classification.confidence,
        scores,
      },
    });
  }
  // QR codes: each decoded code is a closed polygon (4 normalized corners) with
  // its payload as the label, fed to the shared renderer's `polygons` layer.
  if (qrcodes && qrcodes.length) {
    layers.push({
      type: "polygons",
      items: qrcodes.map((q) => ({ points: q.points, label: q.text })),
    });
  }
  return { layers };
}

/** Canvas overlay driven by the shared RecameraOverlay renderer. Replaces the
 *  old SVG BoxOverlay + DOM ClassificationOverlay with one <canvas> so the
 *  device Console draws through the exact same code path as the app previews. */
function OverlayCanvas({
  frame,
  rect,
}: {
  frame: IOverlayFrame;
  rect: ContentRect;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(rect.width);
    const H = Math.round(rect.height);
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    renderOverlay(ctx, adaptDebugFrame(frame), { width: W, height: H });
  }, [frame, rect.width, rect.height]);
  if (!rect.width || !rect.height) return null;
  return (
    <canvas
      ref={ref}
      className="absolute pointer-events-none"
      style={{
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
      }}
    />
  );
}

/** Icon button used by the in-video camera toolbar. Active state fills the
 *  chip with the theme green; disabled dims it. */
function CamToolButton({
  tip,
  active,
  disabled,
  loading,
  onClick,
  children,
}: {
  tip: string;
  active?: boolean;
  disabled?: boolean;
  loading?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Tooltip title={tip} placement="top">
      <button
        type="button"
        disabled={disabled}
        onClick={onClick}
        style={{
          width: 30,
          height: 30,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          border: "none",
          borderRadius: 6,
          cursor: disabled ? "not-allowed" : "pointer",
          background: active ? "#8fc31f" : "transparent",
          color: active ? "#1a1a1a" : "#ffffff",
          opacity: disabled ? 0.35 : loading ? 0.6 : 1,
          fontSize: 15,
          transition: "background 0.15s ease, color 0.15s ease",
        }}
      >
        {children}
      </button>
    </Tooltip>
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
  // sample yet. (Polling effect lives below useDebugStream — it also probes
  // for legacy app packages while the debug stream is playing.)
  const [focusOn, setFocusOn] = useState(false);
  const [focus, setFocus] = useState<IGetFocusValueResult | null>(null);
  const [focusPeak, setFocusPeak] = useState(0);

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
  // (older supervisor) — the overlay toolbar is hidden entirely in that case.
  // camConf = the orientation persisted on-device (saved). pending = the local
  // draft edited by the toolbar toggles; it diverges from camConf until Apply.
  // undefined=loading, null=endpoint unavailable (older supervisor) — toolbar
  // hidden entirely in that case.
  const [camConf, setCamConf] = useState<ICameraConfig | null | undefined>(
    undefined
  );
  const [pending, setPending] = useState<ICameraConfig | null>(null);
  const [camBusy, setCamBusy] = useState(false);
  useEffect(() => {
    getCameraConfigApi()
      .then((res) => {
        if (isOk(res) && res.data && typeof res.data.mirror === "boolean") {
          const saved: ICameraConfig = {
            mirror: !!res.data.mirror,
            flip: !!res.data.flip,
            rotation: res.data.rotation === 180 ? 180 : 0,
          };
          setCamConf(saved);
          setPending(saved);
        } else {
          setCamConf(null);
        }
      })
      .catch(() => setCamConf(null));
  }, []);

  // Orientation only takes effect on app restart (applyCameraConf runs before
  // VI init; mid-stream re-apply hangs VPSS), so the toggles edit a local draft
  // and a single Apply persists all three at once and restarts the active app.
  const camDirty =
    !!camConf &&
    !!pending &&
    (pending.mirror !== camConf.mirror ||
      pending.flip !== camConf.flip ||
      pending.rotation !== camConf.rotation);

  /** Toggle an orientation field in the local draft only — no API call. */
  const onTogglePending = (patch: Partial<ICameraConfig>) => {
    if (!pending || camBusy) return;
    setPending({ ...pending, ...patch });
  };

  /** Persist the whole draft in one shot. The backend restarts the active app
   *  to fold the new orientation into VI, so the debug stream drops and then
   *  reconnects on its own; keep the toolbar disabled through that window. */
  const onApplyCamera = async () => {
    if (!camConf || !pending || camBusy || !camDirty) return;
    setCamBusy(true);
    try {
      const res = await setCameraConfigApi({
        mirror: pending.mirror,
        flip: pending.flip,
        rotation: pending.rotation,
      });
      if (isOk(res)) {
        const saved: ICameraConfig = {
          mirror: !!res.data?.mirror,
          flip: !!res.data?.flip,
          rotation: res.data?.rotation === 180 ? 180 : 0,
        };
        setCamConf(saved);
        setPending(saved);
        message.info(t("camera.applying"));
        // App is restarting — keep the toolbar locked through the restart
        // window; the debug stream reconnects automatically once it is back.
        setTimeout(() => setCamBusy(false), 8000);
        return;
      }
      if (isBusy(res)) {
        message.warning(t("camera.busy"));
      } else {
        message.error(res.msg || t("camera.saveFailed"));
      }
    } catch (e) {
      message.error(t("camera.saveFailed"));
    }
    setCamBusy(false);
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

  // Legacy app-package probe: while the debug stream is actually playing the
  // camera app is running, so getFocusValue returning available:false for
  // >= 3 consecutive samples means the installed app predates the focus/
  // hot-orientation support. Cleared as soon as a sample comes back available.
  const [legacyApp, setLegacyApp] = useState(false);
  const legacyMissRef = useRef(0);
  const streamingNow = debugOn && hasDebugWs && status === "streaming";

  // Poll getFocusValue at 1s while the focus bar is shown, or (in the
  // background) while the stream is playing so the legacy probe can run.
  useEffect(() => {
    if (!focusOn) {
      setFocus(null);
      setFocusPeak(0);
    }
    if (!focusOn && !streamingNow) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await getFocusValueApi();
        if (cancelled) return;
        const data =
          isOk(res) && res.data ? res.data : ({ available: false } as const);
        if (focusOn) {
          setFocus(data);
          if (data.available && typeof data.fv === "number") {
            const v = data.fv;
            setFocusPeak((p) => (v > p ? v : p));
          }
        }
        if (data.available) {
          legacyMissRef.current = 0;
          setLegacyApp(false);
        } else if (streamingNow) {
          legacyMissRef.current += 1;
          if (legacyMissRef.current >= 3) setLegacyApp(true);
        }
      } catch (e) {
        if (!cancelled && focusOn) setFocus({ available: false });
      }
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [focusOn, streamingNow]);

  // Reset the miss counter whenever the stream stops so a later session
  // starts a fresh 3-sample probe.
  useEffect(() => {
    if (!streamingNow) legacyMissRef.current = 0;
  }, [streamingNow]);

  // A legacy app cannot report focus values — force the assist bar off.
  useEffect(() => {
    if (legacyApp && focusOn) setFocusOn(false);
  }, [legacyApp, focusOn]);

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
                {getAppTags(app) && (
                  <span className="text-muted text-12">{getAppTags(app)}</span>
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

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(280px,320px)] gap-16 mt-16">
              {/* Player */}
              <div className="min-w-0">
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
                        <OverlayCanvas frame={overlay} rect={contentRect} />
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

                  {/* Focus assist bar — compact sharpness readout pinned to
                      the top of the video area while the toolbar's focus
                      button is on. */}
                  {focusOn && (
                    <div
                      className="absolute left-8 right-8 top-8 rounded-8 px-12 py-6"
                      style={{ background: "rgba(0,0,0,0.6)" }}
                    >
                      {focus === null ? (
                        <div className="text-12 text-white opacity-70">
                          {t("focus.waiting")}
                        </div>
                      ) : !focus.available ? (
                        <div className="text-12 text-white opacity-70">
                          {t("focus.notRunning")}
                        </div>
                      ) : (
                        <div className="flex items-center gap-12">
                          <span className="text-11 text-white opacity-70 flex-none">
                            {t("focus.score")}
                          </span>
                          <span className="rc-mono text-12 text-white flex-none">
                            {focusValue !== null ? Math.round(focusValue) : "-"}
                          </span>
                          <div
                            className="relative flex-1 rounded-6 overflow-hidden"
                            style={{
                              height: 8,
                              background: "rgba(255,255,255,0.2)",
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
                                  background: "#ffffff",
                                }}
                              />
                            )}
                          </div>
                          <span className="rc-mono text-11 text-white opacity-70 flex-none">
                            {t("focus.peak", { peak: Math.round(focusPeak) })}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Camera toolbar — orientation toggles + focus assist,
                      overlaid bottom-right. Hidden entirely when the camera
                      config endpoint is unavailable (older supervisor). Shown
                      even while the debug stream is off: orientation writes
                      do not depend on the stream. */}
                  {!!camConf && (
                    <div
                      className="rc-cam-toolbar absolute right-8 bottom-8 flex items-center gap-4 rounded-8 px-6 py-4"
                      style={{ background: "rgba(0,0,0,0.6)" }}
                    >
                      {legacyApp && (
                        <Tooltip
                          title={t("camera.legacyAppHint")}
                          placement="top"
                        >
                          <span
                            className="flex items-center justify-center"
                            style={{
                              width: 30,
                              height: 30,
                              color: "#faad14",
                              fontSize: 15,
                            }}
                          >
                            <WarningOutlined />
                          </span>
                        </Tooltip>
                      )}
                      <CamToolButton
                        tip={t("camera.mirror")}
                        active={pending?.mirror}
                        disabled={camBusy}
                        onClick={() =>
                          onTogglePending({ mirror: !pending?.mirror })
                        }
                      >
                        <SwapOutlined />
                      </CamToolButton>
                      <CamToolButton
                        tip={t("camera.flip")}
                        active={pending?.flip}
                        disabled={camBusy}
                        onClick={() =>
                          onTogglePending({ flip: !pending?.flip })
                        }
                      >
                        <SwapOutlined rotate={90} />
                      </CamToolButton>
                      <CamToolButton
                        tip={t("camera.rotation")}
                        active={pending?.rotation === 180}
                        disabled={camBusy}
                        onClick={() =>
                          onTogglePending({
                            rotation: pending?.rotation === 180 ? 0 : 180,
                          })
                        }
                      >
                        <SyncOutlined />
                      </CamToolButton>
                      {camDirty && (
                        <CamToolButton
                          tip={t("camera.apply")}
                          active
                          loading={camBusy}
                          disabled={camBusy}
                          onClick={onApplyCamera}
                        >
                          <CheckOutlined />
                        </CamToolButton>
                      )}
                      <span
                        className="flex-none"
                        style={{
                          width: 1,
                          height: 18,
                          background: "rgba(255,255,255,0.25)",
                          margin: "0 2px",
                        }}
                      />
                      <CamToolButton
                        tip={t("focus.title")}
                        active={focusOn}
                        disabled={legacyApp}
                        onClick={() => setFocusOn((v) => !v)}
                      >
                        <AimOutlined />
                      </CamToolButton>
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
