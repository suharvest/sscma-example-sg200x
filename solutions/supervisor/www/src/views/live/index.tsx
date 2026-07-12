import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Modal, Select, Spin, Switch, message } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { getCurrentAppApi, getAppListApi, setAppModelApi } from "@/api/app";
import { IAppInfo } from "@/api/app/app";
import useDebugStream, { IOverlayFrame } from "@/hooks/useDebugStream";
import {
  resolveRtspUrl,
  resolveDebugVideoUrl,
  resolveDebugResultsUrl,
} from "@/utils/appStream";
import IntegrationDoc from "@/components/integration-doc";

function copyText(text: string) {
  navigator.clipboard
    ?.writeText(text)
    .then(() => message.success("Copied"))
    .catch(() => message.error("Copy failed"));
}

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
        const label =
          b.target !== undefined && b.target !== null
            ? `${b.target} ${(b.score <= 1 ? b.score * 100 : b.score).toFixed(
                0
              )}%`
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
      if ((res.code === 0 || res.code === "0") && res.data) {
        const data = res.data;
        // `current` may come as { app: {...} } or as the manifest itself.
        const inline = data as unknown as IAppInfo;
        const current: IAppInfo | null =
          (data.app as IAppInfo) || (inline?.id ? inline : null);
        if (current?.id) {
          setApp(current);
          // Backend /current returns `probe` (live init-script status) and
          // `state` (state machine); prefer the live probe.
          setAppStatus(
            String(data.probe || data.state || data.status || current.status || "")
          );
          setCurrentModel(
            data.current_model || current.current_model || current.default_model
          );
          return;
        }
      }
      // Fallback: derive from list
      const listRes = await getAppListApi();
      if (listRes.code === 0 || listRes.code === "0") {
        const apps = Array.isArray(listRes.data)
          ? (listRes.data as unknown as IAppInfo[])
          : listRes.data?.apps ?? [];
        const activeId = !Array.isArray(listRes.data)
          ? listRes.data?.current || listRes.data?.active_app
          : null;
        const active =
          apps.find((a) => a.id === activeId) ||
          apps.find((a) => a.active) ||
          null;
        setApp(active);
        setAppStatus(String(active?.status || ""));
        setCurrentModel(active?.current_model || active?.default_model);
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
    Modal.confirm({
      title: `Switch model to "${model}"?`,
      icon: <ExclamationCircleOutlined />,
      content:
        "The application will restart to load the new model. Streams will be briefly interrupted.",
      okText: "Switch",
      cancelText: "Cancel",
      onOk: async () => {
        setModelSwitching(true);
        setDebugOn(false);
        try {
          const res = await setAppModelApi({ app_id: app.id, model });
          if (res.code === 0 || res.code === "0") {
            setCurrentModel(model);
            message.success(
              "Model updated. The application is restarting with the new model."
            );
          } else {
            message.error(res.msg || "Failed to switch model");
          }
        } catch (e) {
          message.error("Failed to switch model");
        } finally {
          setModelSwitching(false);
          fetchCurrent();
        }
      },
    });
  };

  const running = ["RUNNING", "STARTING"].includes(
    (appStatus || "").toUpperCase()
  );

  const rtspFallbackCard = (
    <div className="rc-card-surface p-24 max-w-[440px] text-left">
      <div className="rc-section-label mb-8">RTSP direct access</div>
      <div className="text-13 text-muted mb-12">
        {hasDebugWs
          ? "The debug stream is unavailable right now. You can still open the application's RTSP stream directly:"
          : "This application does not provide a browser debug stream. Open its RTSP stream with a player such as VLC:"}
      </div>
      {rtspUrl ? (
        <div className="bg-white border border-line rounded-8 px-12 py-8 flex items-center justify-between gap-8">
          <span className="rc-mono text-12 break-all">{rtspUrl}</span>
          <Button size="small" onClick={() => copyText(rtspUrl)}>
            Copy
          </Button>
        </div>
      ) : (
        <div className="text-13 text-muted">
          No RTSP address declared by this application.
        </div>
      )}
      <div className="text-12 text-muted mt-12">
        VLC: Media → Open Network Stream → paste the address above.
      </div>
    </div>
  );

  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">Solution Console</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            Live Debug
          </h1>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchCurrent}
          loading={loading}
        >
          Refresh
        </Button>
      </div>

      <Spin spinning={loading}>
        {!app ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <VideoCameraOutlined style={{ fontSize: 28, color: "#666" }} />
            <div className="mt-12 font-medium">No active application</div>
            <div className="text-muted text-13 mt-4 mb-16">
              Activate an application to view and debug its output.
            </div>
            <Button type="primary" onClick={() => navigate("/")}>
              Go to Applications
            </Button>
          </div>
        ) : (
          <>
            {/* Current app info bar */}
            <div className="rc-card mt-24 px-20 py-14 flex items-center justify-between gap-12 flex-wrap">
              <div className="flex items-center gap-12 flex-wrap">
                <span className="font-display font-semibold text-15">
                  {app.name || app.id}
                </span>
                {app.name_zh && (
                  <span className="text-muted text-13">{app.name_zh}</span>
                )}
                <span className={`rc-badge ${running ? "accent" : ""}`}>
                  <span
                    className="dot"
                    style={{ background: running ? "#8fc31f" : "#bbbbbb" }}
                  />
                  {appStatus ? appStatus.toLowerCase() : "unknown"}
                </span>
              </div>
              {currentModel && (
                <span className="rc-section-label">
                  model: {currentModel}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-16 mt-16">
              {/* Player */}
              <div className="lg:col-span-2">
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
                            Connecting to debug stream…
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
                            Debug stream is off. Toggle it on to preview the
                            camera in the browser.
                          </div>
                        </div>
                      ) : (
                        rtspFallbackCard
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Control panel */}
              <div className="flex flex-col gap-16">
                <div className="rc-card p-20">
                  <div className="rc-section-label mb-12">Debug Controls</div>
                  <div className="flex items-center justify-between gap-12">
                    <div>
                      <div className="text-14 font-medium">Debug stream</div>
                      <div className="text-12 text-muted mt-2">
                        Taps H.264 from the device only while enabled — zero
                        overhead when off.
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
                        Result overlay
                      </div>
                      <div className="text-12 text-muted mt-2">
                        Draw inference boxes over the video.
                      </div>
                    </div>
                    <Switch
                      checked={overlayOn}
                      disabled={!debugOn}
                      onChange={setOverlayOn}
                    />
                  </div>
                  {(app.models?.length ?? 0) >= 2 && !!app.models && (
                    <div className="mt-16 pt-16 border-t border-line">
                      <div className="text-14 font-medium mb-8">Model</div>
                      <Select
                        className="w-full"
                        value={currentModel}
                        loading={modelSwitching}
                        disabled={modelSwitching}
                        onChange={onModelChange}
                        options={app.models.map((m) => ({
                          value: m.name,
                          label: m.task ? `${m.name} (${m.task})` : m.name,
                        }))}
                      />
                      <div className="text-12 text-muted mt-6">
                        Switching restarts the application.
                      </div>
                    </div>
                  )}
                </div>

                <div className="rc-card p-20">
                  <div className="rc-section-label mb-12">Endpoints</div>
                  <div className="mb-12">
                    <div className="text-12 text-muted mb-4">RTSP stream</div>
                    {rtspUrl ? (
                      <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                        <span className="rc-mono text-12 break-all">
                          {rtspUrl}
                        </span>
                        <Button size="small" onClick={() => copyText(rtspUrl)}>
                          Copy
                        </Button>
                      </div>
                    ) : (
                      <div className="text-12 text-muted">Not declared.</div>
                    )}
                  </div>
                  <div>
                    <div className="text-12 text-muted mb-4">MQTT topic</div>
                    {app.mqtt_topic ? (
                      <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                        <span className="rc-mono text-12 break-all">
                          {app.mqtt_topic}
                        </span>
                        <Button
                          size="small"
                          onClick={() => copyText(app.mqtt_topic || "")}
                        >
                          Copy
                        </Button>
                      </div>
                    ) : (
                      <div className="text-12 text-muted">Not declared.</div>
                    )}
                  </div>
                </div>

                {!!app.pipeline?.length && (
                  <div className="rc-card p-20">
                    <div className="rc-section-label mb-12">
                      Pipeline (fixed)
                    </div>
                    <div className="flex flex-col gap-6">
                      {app.pipeline.map((m, idx) => (
                        <div
                          key={m.name}
                          className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8"
                        >
                          <span className="rc-mono text-12">
                            {idx + 1}. {m.name}
                          </span>
                          <span className="rc-badge">{m.task || "model"}</span>
                        </div>
                      ))}
                    </div>
                    <div className="text-12 text-muted mt-8">
                      This application runs a fixed multi-model pipeline —
                      its components cannot be switched.
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Recent messages */}
            <div className="rc-card mt-16">
              <div className="flex items-baseline justify-between px-20 py-14 border-b border-line">
                <span className="rc-section-label">
                  Recent Inference Messages
                </span>
                <span className="rc-mono text-11 text-muted">
                  {messages.length ? `${messages.length} / 100` : "live"}
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
                        {new Date(m.receivedAt).toLocaleTimeString()}
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
                    ? "No messages received yet — waiting for inference results."
                    : "Enable the debug stream to see live inference results here."}
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
