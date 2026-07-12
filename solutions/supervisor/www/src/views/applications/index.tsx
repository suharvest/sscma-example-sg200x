import { useEffect, useMemo, useState } from "react";
import { Button, Drawer, Modal, Spin, message } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  AppstoreOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { getAppListApi, switchAppApi, stopAppApi } from "@/api/app";
import { IAppInfo } from "@/api/app/app";
import { resolveRtspUrl } from "@/utils/appStream";
import { pickLocalized, pickLocalizedAlt } from "@/utils/appLocale";
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

const Applications = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const [apps, setApps] = useState<IAppInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detailApp, setDetailApp] = useState<IAppInfo | null>(null);
  const [loadError, setLoadError] = useState(false);
  const navigate = useNavigate();

  const copyText = (text: string) => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => message.success(t("common.copied")))
      .catch(() => message.error(t("common.copyFailed")));
  };

  /** Localized status label; unknown backend states fall back verbatim. */
  const statusLabel = (app: IAppInfo, isActive: boolean) => {
    const raw = statusText(app, isActive);
    return t(`apps.status.${raw}`, { defaultValue: raw });
  };

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
    const name = pickLocalized(app, "name") || app.id;
    Modal.confirm({
      title: t("apps.activateTitle", { name }),
      icon: <ExclamationCircleOutlined />,
      content: t("apps.activateContent"),
      okText: t("common.activate"),
      cancelText: t("common.cancel"),
      onOk: async () => {
        setSwitching(app.id);
        try {
          const res = await switchAppApi({ app_id: app.id });
          if (res.code === 0 || res.code === "0") {
            message.success(t("apps.activated", { name }));
          } else {
            message.error(res.msg || t("apps.activateFailed"));
          }
        } catch (e) {
          message.error(t("apps.activateFailed"));
        } finally {
          setSwitching(null);
          fetchList(true);
        }
      },
    });
  };

  const onStop = (app: IAppInfo) => {
    const name = pickLocalized(app, "name") || app.id;
    Modal.confirm({
      title: t("apps.stopTitle", { name }),
      icon: <ExclamationCircleOutlined />,
      content: t("apps.stopContent"),
      okText: t("common.stop"),
      okButtonProps: { danger: true },
      cancelText: t("common.cancel"),
      onOk: async () => {
        setSwitching(app.id);
        try {
          const res = await stopAppApi();
          if (res.code === 0 || res.code === "0") {
            message.success(t("apps.stopSuccess"));
          } else {
            message.error(res.msg || t("apps.stopFailed"));
          }
        } catch (e) {
          message.error(t("apps.stopFailed"));
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
          <div className="rc-eyebrow mb-4">{t("common.console")}</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            {t("apps.title")}
          </h1>
          <p className="text-muted text-13 mt-4 mb-0 max-w-[56ch]">
            {t("apps.subtitle")}
          </p>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={() => fetchList()}
          loading={loading}
        >
          {t("common.refresh")}
        </Button>
      </div>

      <Spin spinning={loading}>
        {loadError && !apps.length ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <div className="text-muted text-13">{t("apps.loadError")}</div>
            <Button className="mt-16" onClick={() => fetchList()}>
              {t("common.retry")}
            </Button>
          </div>
        ) : !apps.length && !loading ? (
          <div className="rc-card-surface mt-24 p-32 text-center">
            <AppstoreOutlined style={{ fontSize: 28, color: "#666" }} />
            <div className="mt-12 font-medium">{t("apps.noApps")}</div>
            <div className="text-muted text-13 mt-4">
              {t("apps.noAppsHint")}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-16 mt-24">
            {sortedApps.map((app) => {
              const isActive = app.id === activeId;
              const displayName = pickLocalized(app, "name") || app.id;
              const altName = pickLocalizedAlt(app, "name");
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
                      {pickLocalized(app, "scene") ||
                        (app.type === "external-firmware"
                          ? t("apps.sceneFirmware")
                          : t("apps.sceneGeneral"))}
                    </span>
                    <span className={`rc-badge ${isActive ? "accent" : ""}`}>
                      <span
                        className="dot"
                        style={{ background: statusColor(app, isActive) }}
                      />
                      {isActive ? t("apps.active") : statusLabel(app, isActive)}
                    </span>
                  </div>

                  <div className="mt-14">
                    <div className="font-display font-semibold text-16 leading-snug">
                      {displayName}
                    </div>
                    {altName && (
                      <div className="text-muted text-13 mt-2">{altName}</div>
                    )}
                  </div>

                  <p className="text-muted text-13 leading-relaxed mt-8 mb-0 flex-1 min-h-36">
                    {pickLocalized(app, "description") ||
                      t("apps.noDescription")}
                  </p>

                  <div className="rc-section-label mt-14">
                    {app.models?.length
                      ? t("apps.modelsCount", { count: app.models.length })
                      : app.pipeline?.length
                      ? t("apps.pipelineCount", { count: app.pipeline.length })
                      : app.type === "external-firmware"
                      ? t("apps.externalFirmware")
                      : t("apps.noSwitchableModels")}
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
                          {t("apps.liveView")}
                        </Button>
                        <Button
                          size="small"
                          danger
                          loading={switching === app.id}
                          onClick={() => onStop(app)}
                        >
                          {t("common.stop")}
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
                        {t("common.activate")}
                      </Button>
                    )}
                    <Button size="small" onClick={() => setDetailApp(app)}>
                      {t("common.details")}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Spin>

      <Drawer
        title={
          detailApp ? pickLocalized(detailApp, "name") || detailApp.id : ""
        }
        open={!!detailApp}
        onClose={() => setDetailApp(null)}
        width={Math.min(520, window.innerWidth - 24)}
      >
        {detailApp && (
          <div className="flex flex-col gap-16 text-13">
            {pickLocalizedAlt(detailApp, "name") && (
              <div className="text-muted -mt-8">
                {pickLocalizedAlt(detailApp, "name")}
              </div>
            )}
            <div>
              <div className="rc-section-label mb-4">
                {t("apps.drawer.description")}
              </div>
              <div>{pickLocalized(detailApp, "description") || "-"}</div>
            </div>
            <div className="grid grid-cols-2 gap-12">
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.type")}
                </div>
                <div>{detailApp.type || "native"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.version")}
                </div>
                <div>{detailApp.version || "-"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.scene")}
                </div>
                <div>{pickLocalized(detailApp, "scene") || "-"}</div>
              </div>
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.author")}
                </div>
                <div>{detailApp.author || "-"}</div>
              </div>
            </div>
            {resolveRtspUrl(detailApp) && (
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.rtsp")}
                </div>
                <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                  <span className="rc-mono text-12 break-all">
                    {resolveRtspUrl(detailApp)}
                  </span>
                  <Button
                    size="small"
                    onClick={() => copyText(resolveRtspUrl(detailApp))}
                  >
                    {t("common.copy")}
                  </Button>
                </div>
              </div>
            )}
            {detailApp.mqtt_topic && (
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.mqtt")}
                </div>
                <div className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8">
                  <span className="rc-mono text-12 break-all">
                    {detailApp.mqtt_topic}
                  </span>
                  <Button
                    size="small"
                    onClick={() => copyText(detailApp.mqtt_topic || "")}
                  >
                    {t("common.copy")}
                  </Button>
                </div>
              </div>
            )}
            {!!detailApp.models?.length && (
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.models")}
                </div>
                <div className="flex flex-col gap-6">
                  {detailApp.models.map((m) => (
                    <div
                      key={m.name}
                      className="rc-card-surface px-12 py-8 flex items-center justify-between gap-8"
                    >
                      <span className="rc-mono text-12">{m.name}</span>
                      <span className="rc-badge">
                        {m.task || t("common.model")}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!!detailApp.pipeline?.length && (
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.pipeline")}
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
                      <span className="rc-badge">
                        {m.task || t("common.model")}
                      </span>
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
