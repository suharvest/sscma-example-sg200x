import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Drawer, Modal, Progress, Spin, Tooltip, Upload, message } from "antd";
import type { UploadFile, UploadProps } from "antd";
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  AppstoreOutlined,
  InboxOutlined,
  DownloadOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  getAppListApi,
  switchAppApi,
  stopAppApi,
  installAppApi,
} from "@/api/app";
import { IAppInfo, IInstallAppResult } from "@/api/app/app";
import { uploadFiles, ensureDirectory } from "@/api/files";
import { resolveRtspUrl } from "@/utils/appStream";
import { pickLocalized, pickLocalizedAlt } from "@/utils/appLocale";
import IntegrationDoc from "@/components/integration-doc";
import useConfigStore from "@/store/config";
import useRunModeSwitch from "@/hooks/useRunModeSwitch";

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

/* ------------------------------------------------------------------ */
/* Install App modal: chunked upload (fileMgr) -> appMgr/installApp    */
/* ------------------------------------------------------------------ */

/** Upload destination, relative to the fileMgr "local" storage (/userdata). */
const INSTALL_UPLOAD_DIR = "apps_upload";
const MAX_DEB_SIZE = 200 * 1024 * 1024; // keep in sync with backend installApp
/** Backend passes the path through a quoted shell arg — same whitelist. */
const SAFE_NAME_RE = /^[A-Za-z0-9._+-]+$/;

type InstallStage = "idle" | "uploading" | "installing" | "success" | "failed";

const InstallAppModal = ({
  open,
  onClose,
  onInstalled,
}: {
  open: boolean;
  onClose: () => void;
  onInstalled: () => void;
}) => {
  const { t } = useTranslation();
  const [stage, setStage] = useState<InstallStage>("idle");
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [output, setOutput] = useState("");

  const busy = stage === "uploading" || stage === "installing";

  const reset = () => {
    setStage("idle");
    setFileList([]);
    setUploadPercent(0);
    setErrorMsg("");
    setOutput("");
  };

  const close = () => {
    if (busy) return; // never abandon a running install silently
    reset();
    onClose();
  };

  const beforeUpload: UploadProps["beforeUpload"] = (file) => {
    if (!file.name.toLowerCase().endsWith(".deb")) {
      message.error(t("apps.install.onlyDeb"));
      return Upload.LIST_IGNORE;
    }
    if (!SAFE_NAME_RE.test(file.name)) {
      message.error(t("apps.install.badChars"));
      return Upload.LIST_IGNORE;
    }
    if (file.size >= MAX_DEB_SIZE) {
      message.error(t("apps.install.tooLarge"));
      return Upload.LIST_IGNORE;
    }
    return false; // valid: keep in the list, upload manually on Install
  };

  const handleInstall = async () => {
    const raw = fileList[0]?.originFileObj;
    if (!raw) return;

    // Stage 1: chunked upload to local:/apps_upload/ (existing fileMgr
    // protocol — 1MB offset chunks — reused as-is via uploadFiles()).
    setStage("uploading");
    setUploadPercent(0);
    setErrorMsg("");
    setOutput("");
    try {
      await ensureDirectory("local", INSTALL_UPLOAD_DIR);
      const dt = new DataTransfer();
      dt.items.add(raw);
      await uploadFiles("local", INSTALL_UPLOAD_DIR, dt.files, (info) => {
        setUploadPercent(info.currentFileProgress);
      });
    } catch (e) {
      setStage("failed");
      setErrorMsg(t("apps.install.uploadFailed"));
      return;
    }

    // Stage 2: opkg install on the device (up to ~2 minutes).
    setStage("installing");
    try {
      const res = await installAppApi({
        path: `/userdata/${INSTALL_UPLOAD_DIR}/${raw.name}`,
      });
      const data = res.data as IInstallAppResult | undefined;
      setOutput(data?.output || "");
      if (res.code === 0 || res.code === "0") {
        setStage("success");
        onInstalled();
      } else if (res.code === -2 || res.code === "-2") {
        setStage("failed");
        setErrorMsg(t("apps.install.busy"));
      } else {
        setStage("failed");
        setErrorMsg(res.msg || t("apps.install.failed"));
      }
    } catch (e) {
      setStage("failed");
      setErrorMsg(t("apps.install.failed"));
    }
  };

  return (
    <Modal
      title={t("apps.install.title")}
      open={open}
      onCancel={close}
      maskClosable={false}
      closable={!busy}
      keyboard={!busy}
      footer={
        stage === "idle" ? (
          <>
            <Button onClick={close}>{t("common.cancel")}</Button>
            <Button
              type="primary"
              disabled={!fileList.length}
              onClick={handleInstall}
            >
              {t("apps.install.start")}
            </Button>
          </>
        ) : busy ? null : (
          <>
            <Button onClick={reset}>{t("apps.install.installAnother")}</Button>
            <Button type="primary" onClick={close}>
              {t("apps.install.close")}
            </Button>
          </>
        )
      }
    >
      {stage === "idle" && (
        <div className="flex flex-col gap-12">
          <p className="text-muted text-13 m-0">{t("apps.install.hint")}</p>
          <Upload.Dragger
            accept=".deb"
            maxCount={1}
            fileList={fileList}
            beforeUpload={beforeUpload}
            onRemove={() => setFileList([])}
            onChange={({ fileList: fl }) => setFileList(fl.slice(-1))}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">{t("apps.install.selectFile")}</p>
          </Upload.Dragger>
          <Alert type="warning" showIcon message={t("apps.install.trust")} />
        </div>
      )}

      {stage === "uploading" && (
        <div className="py-8">
          <div className="text-13 mb-8">
            {t("apps.install.uploading", { percent: uploadPercent })}
          </div>
          <Progress percent={uploadPercent} status="active" />
        </div>
      )}

      {stage === "installing" && (
        <div className="py-8">
          <div className="text-13 mb-8">{t("apps.install.installing")}</div>
          <Progress percent={100} status="active" showInfo={false} />
        </div>
      )}

      {(stage === "success" || stage === "failed") && (
        <div className="flex flex-col gap-12">
          <Alert
            type={stage === "success" ? "success" : "error"}
            showIcon
            message={
              stage === "success" ? t("apps.install.success") : errorMsg
            }
          />
          {output && (
            <div>
              <div className="rc-section-label mb-4">
                {t("apps.install.output")}
              </div>
              <pre className="rc-mono text-12 rc-card-surface p-12 m-0 max-h-[240px] overflow-auto whitespace-pre-wrap break-all">
                {output}
              </pre>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
};

const Applications = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const [apps, setApps] = useState<IAppInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detailApp, setDetailApp] = useState<IAppInfo | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [installOpen, setInstallOpen] = useState(false);
  const navigate = useNavigate();

  // P4-D: in Node-RED mode the C++ app stack is stopped and this page is
  // removed from the menu — a direct link still lands here, so show a hint
  // banner with a switch-back action instead of a broken gallery.
  const { galleryMode, deviceInfo } = useConfigStore();
  const modeKnown = Boolean(deviceInfo?.appName);
  const noderedMode = modeKnown && !galleryMode;
  // renamed: `switching` is taken by the app switch/stop state above
  const { switching: modeSwitching, requestSwitch } = useRunModeSwitch();

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
        <div className="flex gap-8">
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => setInstallOpen(true)}
          >
            {t("apps.install.button")}
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => fetchList()}
            loading={loading}
          >
            {t("common.refresh")}
          </Button>
        </div>
      </div>

      {noderedMode && (
        <Alert
          className="mt-16"
          type="warning"
          showIcon
          message={t("runtimeMode.noderedBanner")}
          action={
            <Button
              size="small"
              type="primary"
              loading={modeSwitching === "console"}
              onClick={() => requestSwitch("console")}
            >
              {t("runtimeMode.switchBack")}
            </Button>
          }
        />
      )}

      <InstallAppModal
        open={installOpen}
        onClose={() => setInstallOpen(false)}
        onInstalled={() => fetchList(true)}
      />

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
              // P5: hardware dependency gating (backend appMgr/list fields).
              const hwUnsupported = app.hw_supported === false;
              const missingList = (app.missing_capabilities || [])
                .map((k) => t(`capabilities.keys.${k}`, { defaultValue: k }))
                .join(", ");
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
                    <div className="flex items-center gap-6 flex-wrap">
                      <span className="rc-badge">
                        {pickLocalized(app, "scene") ||
                          (app.type === "external-firmware"
                            ? t("apps.sceneFirmware")
                            : t("apps.sceneGeneral"))}
                      </span>
                      {hwUnsupported && (
                        <Tooltip
                          title={t("apps.hwMissingTooltip", {
                            list: missingList,
                          })}
                        >
                          <span
                            className="rc-badge"
                            style={{ borderColor: "#D54941", color: "#D54941" }}
                          >
                            {t("apps.hwNotSupported")}
                          </span>
                        </Tooltip>
                      )}
                    </div>
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
                    ) : hwUnsupported ? (
                      <Tooltip
                        title={t("apps.hwMissingTooltip", { list: missingList })}
                      >
                        {/* span wrapper: antd Tooltip needs a non-disabled DOM target */}
                        <span>
                          <Button type="primary" size="small" disabled>
                            {t("common.activate")}
                          </Button>
                        </span>
                      </Tooltip>
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
