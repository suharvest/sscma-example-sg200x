import { useEffect, useMemo, useState } from "react";
import { Button, Drawer, Modal, Spin, message } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  AppstoreOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { getAppListApi, switchAppApi, stopAppApi } from "@/api/app";
import { IAppInfo } from "@/api/app/app";
import { resolveRtspUrl } from "@/utils/appStream";
import IntegrationDoc from "@/components/integration-doc";

const RUNNING_STATES = ["RUNNING", "STARTING"];

function statusColor(app: IAppInfo, isActive: boolean): string {
  const s = (app.status || "").toUpperCase();
  if (s === "ERROR") return "#D54941";
  if (RUNNING_STATES.includes(s) || app.running || isActive) return "#8fc31f";
  return "#bbbbbb";
}

function statusText(app: IAppInfo, isActive: boolean): string {
  if (app.status) return String(app.status).toLowerCase();
  if (app.running) return "running";
  return isActive ? "active" : "stopped";
}

function copyText(text: string) {
  navigator.clipboard
    ?.writeText(text)
    .then(() => message.success("Copied"))
    .catch(() => message.error("Copy failed"));
}

const Applications = () => {
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const [apps, setApps] = useState<IAppInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detailApp, setDetailApp] = useState<IAppInfo | null>(null);
  const [loadError, setLoadError] = useState(false);
  const navigate = useNavigate();

  const fetchList = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const res = await getAppListApi();
      if (res.code === 0 || res.code === "0") {
        const data = res.data;
        const list = Array.isArray(data)
          ? (data as unknown as IAppInfo[])
          : data?.apps ?? [];
        setApps(list);
        const active =
          (!Array.isArray(data) && (data?.current || data?.active_app)) ||
          list.find((a) => a.active)?.id ||
          null;
        setActiveId(active || null);
        setLoadError(false);
      } else {
        setLoadError(true);
      }
    } catch (e) {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchList();
  }, []);

  const onActivate = (app: IAppInfo) => {
    Modal.confirm({
      title: `Activate "${app.name}"?`,
      icon: <ExclamationCircleOutlined />,
      content:
        "Switching applications will stop the currently running application and interrupt its RTSP/MQTT output. The camera will be handed over to the new application.",
      okText: "Activate",
      cancelText: "Cancel",
      onOk: async () => {
        setSwitching(app.id);
        try {
          const res = await switchAppApi({ app_id: app.id });
          if (res.code === 0 || res.code === "0") {
            message.success(`"${app.name}" activated`);
          } else {
            message.error(res.msg || "Failed to switch application");
          }
        } catch (e) {
          message.error("Failed to switch application");
        } finally {
          setSwitching(null);
          fetchList(true);
        }
      },
    });
  };

  const onStop = (app: IAppInfo) => {
    Modal.confirm({
      title: `Stop "${app.name}"?`,
      icon: <ExclamationCircleOutlined />,
      content:
        "The application and its streams will be stopped. You can activate it again at any time.",
      okText: "Stop",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        setSwitching(app.id);
        try {
          const res = await stopAppApi();
          if (res.code === 0 || res.code === "0") {
            message.success("Application stopped");
          } else {
            message.error(res.msg || "Failed to stop application");
          }
        } catch (e) {
          message.error("Failed to stop application");
        } finally {
          setSwitching(null);
          fetchList(true);
        }
      },
    });
  };

  const sortedApps = useMemo(() => {
    return [...apps].sort((a, b) => {
      if (a.id === activeId) return -1;
      if (b.id === activeId) return 1;
      return (a.name || a.id).localeCompare(b.name || b.id);
    });
  }, [apps, activeId]);

  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">Solution Console</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            Applications
          </h1>
          <p className="text-muted text-13 mt-4 mb-0 max-w-[56ch]">
            Each application is a complete scene solution. Only one application
            can hold the camera at a time.
          </p>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={() => fetchList()}
          loading={loading}
        >
          Refresh
        </Button>
      </div>

      <Spin spinning={loading}>
        {loadError && !apps.length ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <div className="text-muted text-13">
              Could not load the application list. The appMgr service may be
              unavailable on this firmware.
            </div>
            <Button className="mt-16" onClick={() => fetchList()}>
              Retry
            </Button>
          </div>
        ) : !apps.length && !loading ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <AppstoreOutlined style={{ fontSize: 28, color: "#666" }} />
            <div className="mt-12 font-medium">No applications installed</div>
            <div className="text-muted text-13 mt-4">
              Install a solution package (.deb) to register applications here.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-16 mt-24">
            {sortedApps.map((app) => {
              const isActive = app.id === activeId;
              return (
                <div
                  key={app.id}
                  className="rc-card p-20 flex flex-col"
                  style={
                    isActive
                      ? {
                          borderColor: "#8fc31f",
                          boxShadow: "0 0 0 1px #8fc31f inset",
                        }
                      : undefined
                  }
                >
                  <div className="flex items-center justify-between gap-8">
                    <span className="rc-badge">
                      {app.scene || (app.type === "external-firmware"
                        ? "Firmware"
                        : "General")}
                    </span>
                    <span className={`rc-badge ${isActive ? "accent" : ""}`}>
                      <span
                        className="dot"
                        style={{ background: statusColor(app, isActive) }}
                      />
                      {isActive ? "Active" : statusText(app, isActive)}
                    </span>
                  </div>

                  <div className="mt-14">
                    <div className="font-display font-semibold text-16 leading-snug">
                      {app.name || app.id}
                    </div>
                    {app.name_zh && (
                      <div className="text-muted text-13 mt-2">
                        {app.name_zh}
                      </div>
                    )}
                  </div>

                  <p className="text-muted text-13 leading-relaxed mt-8 mb-0 flex-1 min-h-36">
                    {app.description || "No description provided."}
                  </p>

                  <div className="rc-section-label mt-14">
                    {app.models?.length
                      ? `${app.models.length} model${
                          app.models.length > 1 ? "s" : ""
                        }`
                      : app.pipeline?.length
                      ? `${app.pipeline.length}-model pipeline`
                      : app.type === "external-firmware"
                      ? "external firmware"
                      : "no switchable models"}
                    {app.version ? ` · v${app.version}` : ""}
                  </div>

                  <div className="flex gap-8 mt-14 pt-14 border-t border-line">
                    {isActive ? (
                      <>
                        <Button
                          type="primary"
                          size="small"
                          onClick={() => navigate("/live")}
                        >
                          Live View
                        </Button>
                        <Button
                          size="small"
                          danger
                          loading={switching === app.id}
                          onClick={() => onStop(app)}
                        >
                          Stop
                        </Button>
                      </>
                    ) : (
                      <Button
                        type="primary"
                        size="small"
                        loading={switching === app.id}
                        disabled={switching !== null && switching !== app.id}
                        onClick={() => onActivate(app)}
                      >
                        Activate
                      </Button>
                    )}
                    <Button size="small" onClick={() => setDetailApp(app)}>
                      Details
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Spin>

      <Drawer
        title={detailApp?.name || detailApp?.id}
        open={!!detailApp}
        onClose={() => setDetailApp(null)}
        width={Math.min(520, window.innerWidth - 24)}
      >
        {detailApp && (
          <div className="flex flex-col gap-16 text-13">
            {detailApp.name_zh && (
              <div className="text-muted -mt-8">{detailApp.name_zh}</div>
            )}
            <div>
              <div className="rc-section-label mb-4">Description</div>
              <div>{detailApp.description || "-"}</div>
            </div>
            <div className="grid grid-cols-2 gap-12">
              <div>
                <div className="rc-section-label mb-4">Type</div>
                <div>{detailApp.type || "native"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">Version</div>
                <div>{detailApp.version || "-"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">Scene</div>
                <div>{detailApp.scene || "-"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">Author</div>
                <div>{detailApp.author || "-"}</div>
              </div>
            </div>
            {resolveRtspUrl(detailApp) && (
              <div>
                <div className="rc-section-label mb-4">RTSP Stream</div>
                <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                  <span className="rc-mono text-12 break-all">
                    {resolveRtspUrl(detailApp)}
                  </span>
                  <Button
                    size="small"
                    onClick={() => copyText(resolveRtspUrl(detailApp))}
                  >
                    Copy
                  </Button>
                </div>
              </div>
            )}
            {detailApp.mqtt_topic && (
              <div>
                <div className="rc-section-label mb-4">MQTT Topic</div>
                <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                  <span className="rc-mono text-12 break-all">
                    {detailApp.mqtt_topic}
                  </span>
                  <Button
                    size="small"
                    onClick={() => copyText(detailApp.mqtt_topic || "")}
                  >
                    Copy
                  </Button>
                </div>
              </div>
            )}
            {!!detailApp.models?.length && (
              <div>
                <div className="rc-section-label mb-4">Models</div>
                <div className="flex flex-col gap-6">
                  {detailApp.models.map((m) => (
                    <div
                      key={m.name}
                      className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8"
                    >
                      <span className="rc-mono text-12">{m.name}</span>
                      <span className="rc-badge">{m.task || "model"}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!!detailApp.pipeline?.length && (
              <div>
                <div className="rc-section-label mb-4">
                  Pipeline (fixed, not switchable)
                </div>
                <div className="flex flex-col gap-6">
                  {detailApp.pipeline.map((m, idx) => (
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
              </div>
            )}
            <IntegrationDoc appId={detailApp.id} />
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default Applications;
