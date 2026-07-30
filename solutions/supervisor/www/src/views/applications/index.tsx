import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  App,
  Button,
  Modal,
  Progress,
  Spin,
  Tabs,
  Tooltip,
  Upload,
} from "antd";
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
  uninstallAppApi,
} from "@/api/app";
import { IAppInfo, IInstallAppResult } from "@/api/app/app";
import {
  appDownloadSize,
  downloadToFile,
  fetchCatalog,
  sha256Hex,
  type ICatalogApp,
} from "@/api/app/catalog";
import {
  uploadFiles,
  ensureDirectory,
  getStorageInfo,
  removeEntry,
} from "@/api/files";
import { isOk, isBusy } from "@/utils/api";
import { copyText } from "@/utils/clipboard";
import { resolveRtspUrl } from "@/utils/appStream";
import { pickLocalized, pickLocalizedText } from "@/utils/appLocale";
import { getAppTags } from "@/utils/appTags";
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
/* plus "install from cloud": browser fetches the package on the device's */
/* behalf, then the same upload -> install tail runs unchanged.          */
/* ------------------------------------------------------------------ */

/** Upload destination, relative to the fileMgr "local" storage (/userdata). */
const INSTALL_UPLOAD_DIR = "apps_upload";
const MAX_DEB_SIZE = 200 * 1024 * 1024; // keep in sync with backend installApp
/** Backend passes the path through a quoted shell arg — same whitelist. */
const SAFE_NAME_RE = /^[A-Za-z0-9._+-]+$/;

type InstallStage =
  | "idle"
  | "downloading"
  | "uploading"
  | "installing"
  | "success"
  | "failed";

const InstallAppModal = ({
  open,
  onClose,
  onInstalled,
  installedIds,
}: {
  open: boolean;
  onClose: () => void;
  onInstalled: () => void;
  /** Package ids already on the device — catalog rows for these are disabled. */
  installedIds: Set<string>;
}) => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [stage, setStage] = useState<InstallStage>("idle");
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [output, setOutput] = useState("");

  // --- Install from cloud -------------------------------------------------
  // The device has no route to the internet over USB, so the browser fetches
  // the package and pushes it over the existing upload API. Everything after
  // "we have a File" is the same code path as a manual .deb upload.
  const [tab, setTab] = useState<"cloud" | "upload">("cloud");
  const [catalog, setCatalog] = useState<ICatalogApp[] | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [downloadLabel, setDownloadLabel] = useState("");
  const [downloadPercent, setDownloadPercent] = useState(0);

  const busy =
    stage === "downloading" ||
    stage === "uploading" ||
    stage === "installing";

  const reset = () => {
    setStage("idle");
    setFileList([]);
    setUploadPercent(0);
    setErrorMsg("");
    setOutput("");
    setDownloadLabel("");
    setDownloadPercent(0);
  };

  // Load the catalog when the modal opens. A failure is expected and not an
  // error state: a machine without internet simply gets the upload tab.
  //
  // The "have we started" flag is a ref, not state, and the effect depends on
  // `open` alone. With `catalogLoading` in the dependency array this effect
  // aborted its own request: setCatalogLoading(true) changed a dependency, the
  // cleanup ran ac.abort(), the fetch rejected with AbortError, the catch
  // switched to the upload tab, `finally` cleared the flag, and the whole
  // thing started over — an endless loop that also yanked the tab back every
  // time the user clicked "install from cloud".
  const catalogRequested = useRef(false);
  useEffect(() => {
    if (!open || catalogRequested.current) return;
    catalogRequested.current = true;
    const ac = new AbortController();
    setCatalogLoading(true);
    fetchCatalog(ac.signal)
      .then((c) => {
        setCatalog(c.apps);
        setCatalogError("");
      })
      .catch((e: Error) => {
        // An abort is our own doing (the modal closed); do not report it as a
        // catalog failure or the upload tab gets forced on the way out.
        if (e?.name === "AbortError") return;
        setCatalogError(e?.message || "unknown error");
        setTab("upload");
        catalogRequested.current = false; // allow a retry on reopen
      })
      .finally(() => setCatalogLoading(false));
    return () => ac.abort();
  }, [open]);

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

  /**
   * Push an already-in-memory File to the device, then opkg install it.
   * Shared by the upload tab and the cloud tab — the only difference between
   * them is where the bytes came from.
   */
  const uploadAndInstall = async (file: File): Promise<boolean> => {
    setStage("uploading");
    setUploadPercent(0);
    try {
      await ensureDirectory("local", INSTALL_UPLOAD_DIR);
      const dt = new DataTransfer();
      dt.items.add(file);
      await uploadFiles("local", INSTALL_UPLOAD_DIR, dt.files, (info) => {
        setUploadPercent(info.currentFileProgress);
      });
    } catch (e) {
      setStage("failed");
      setErrorMsg(t("apps.install.uploadFailed"));
      return false;
    }

    setStage("installing");
    try {
      const res = await installAppApi({
        path: `/userdata/${INSTALL_UPLOAD_DIR}/${file.name}`,
      });
      const data = res.data as IInstallAppResult | undefined;
      setOutput(data?.output || "");
      if (isOk(res)) {
        // opkg has unpacked it; the staged copy is dead weight from here on.
        // Only on success — a failed install keeps the file so a retry does
        // not re-download it, and so it is still there to inspect.
        await removeEntry(
          "local",
          `${INSTALL_UPLOAD_DIR}/${file.name}`
        ).catch(() => undefined);
        return true;
      }
      setStage("failed");
      setErrorMsg(
        isBusy(res) ? t("apps.install.busy") : res.msg || t("apps.install.failed")
      );
      return false;
    } catch (e) {
      setStage("failed");
      setErrorMsg(t("apps.install.failed"));
      return false;
    }
  };

  const handleCloudInstall = async (app: ICatalogApp) => {
    setErrorMsg("");
    setOutput("");

    // Refuse before downloading rather than after. Everything an app installs
    // lands on the same storage as the staged package (the /usr and /etc
    // overlays keep their upperdir under /userdata), so one number covers
    // both. The margin is deliberately generous: opkg unpacks on top of the
    // copy we upload, so both exist at once, and the .deb is compressed.
    const needed = appDownloadSize(app) * 3 + 8 * 1024 * 1024;
    const info = await getStorageInfo("local");
    // A null reply means an older supervisor without the endpoint; skip the
    // check rather than block the install on a missing feature.
    if (info && info.free > 0 && info.free < needed) {
      setStage("failed");
      setErrorMsg(
        t("apps.install.noSpace", {
          need: (needed / 1e6).toFixed(0),
          free: (info.free / 1e6).toFixed(0),
        })
      );
      return;
    }

    // Models first, then the package. The .deb's postinst may look for the
    // files it needs, and the app is startable the moment install returns —
    // arriving with half its models is a worse failure than a slow install.
    const files = [...app.models, app.package];
    const downloaded: { file: File; target: string | null }[] = [];

    setStage("downloading");
    for (let i = 0; i < files.length; i++) {
      const entry = files[i];
      setDownloadLabel(`${entry.filename} (${i + 1}/${files.length})`);
      setDownloadPercent(0);
      try {
        const file = await downloadToFile(entry, (loaded, total) => {
          setDownloadPercent(total ? Math.round((loaded / total) * 100) : 0);
        });
        // Verify before anything reaches the device: a truncated download
        // would otherwise be installed and fail somewhere far less obvious.
        // An empty digest means WebCrypto is unavailable (the console is
        // served over plain HTTP, where crypto.subtle exists on localhost
        // only) — that is "cannot verify", not "mismatch".
        if (entry.sha256) {
          const got = await sha256Hex(file);
          if (got && got !== entry.sha256) {
            setStage("failed");
            setErrorMsg(t("apps.install.checksumFailed", { file: entry.filename }));
            return;
          }
        }
        downloaded.push({ file, target: entry.target_path || null });
      } catch (e) {
        setStage("failed");
        setErrorMsg(t("apps.install.downloadFailed", { file: entry.filename }));
        return;
      }
    }

    // Models go straight to their directory under /userdata (gigabytes free),
    // not to the upload staging dir on the cramped root partition.
    for (const item of downloaded) {
      if (!item.target) continue;
      setStage("uploading");
      setUploadPercent(0);
      const rel = item.target.replace(/^\/userdata\/?/, "");
      try {
        await ensureDirectory("local", rel);
        const dt = new DataTransfer();
        dt.items.add(item.file);
        await uploadFiles("local", rel, dt.files, (info) => {
          setUploadPercent(info.currentFileProgress);
        });
      } catch (e) {
        setStage("failed");
        setErrorMsg(t("apps.install.uploadFailed"));
        return;
      }
    }

    const pkg = downloaded.find((d) => !d.target);
    if (!pkg) {
      setStage("failed");
      setErrorMsg(t("apps.install.failed"));
      return;
    }
    if (await uploadAndInstall(pkg.file)) {
      setStage("success");
      onInstalled();
    }
  };

  const handleInstall = async () => {
    const raw = fileList[0]?.originFileObj;
    if (!raw) return;
    setErrorMsg("");
    setOutput("");
    // Same upload -> install tail the cloud tab uses; the only difference is
    // that these bytes came from the user's disk instead of the CDN.
    if (await uploadAndInstall(raw as File)) {
      setStage("success");
      onInstalled();
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
            {tab === "upload" && (
              <Button
                type="primary"
                disabled={!fileList.length}
                onClick={handleInstall}
              >
                {t("apps.install.start")}
              </Button>
            )}
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
        <Tabs
          activeKey={tab}
          onChange={(k) => setTab(k as "cloud" | "upload")}
          items={[
            {
              key: "cloud",
              label: t("apps.install.fromCloud"),
              children: catalogLoading ? (
                <div className="py-24 text-center">
                  <Spin />
                </div>
              ) : catalogError ? (
                <Alert
                  type="info"
                  showIcon
                  message={t("apps.install.catalogUnavailable")}
                  description={
                    <>
                      <div>{t("apps.install.catalogUnavailableHint")}</div>
                      <div className="rc-mono text-12 text-muted mt-8 break-all">
                        {catalogError}
                      </div>
                    </>
                  }
                />
              ) : (
                <div className="flex flex-col gap-8 max-h-[340px] overflow-auto">
                  {(catalog || []).map((app) => {
                    const installed = installedIds.has(app.id);
                    const mb = appDownloadSize(app) / 1e6;
                    return (
                      <div
                        key={app.id}
                        className="rc-card-surface p-12 flex items-start gap-12"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-14">
                            {pickLocalizedText(app.name, app.name_zh)}
                          </div>
                          <div className="text-muted text-12 mt-2 line-clamp-2">
                            {pickLocalizedText(
                              app.description,
                              app.description_zh
                            )}
                          </div>
                          <div className="text-muted text-12 mt-4 rc-mono">
                            {mb > 0 ? `${mb.toFixed(1)} MB` : ""}
                            {app.models.length
                              ? ` · ${t("apps.install.withModels", {
                                  count: app.models.length,
                                })}`
                              : ""}
                          </div>
                        </div>
                        <Button
                          size="small"
                          type={installed ? "default" : "primary"}
                          disabled={installed}
                          onClick={() => handleCloudInstall(app)}
                        >
                          {installed
                            ? t("apps.install.installed")
                            : t("apps.install.installAction")}
                        </Button>
                      </div>
                    );
                  })}
                </div>
              ),
            },
            {
              key: "upload",
              label: t("apps.install.fromFile"),
              children: (
                <div className="flex flex-col gap-12">
                  <p className="text-muted text-13 m-0">
                    {t("apps.install.hint")}
                  </p>
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
                    <p className="ant-upload-text">
                      {t("apps.install.selectFile")}
                    </p>
                  </Upload.Dragger>
                  <Alert
                    type="warning"
                    showIcon
                    message={t("apps.install.trust")}
                  />
                </div>
              ),
            },
          ]}
        />
      )}

      {stage === "downloading" && (
        <div className="py-8">
          <div className="text-13 mb-8">
            {t("apps.install.downloading", { file: downloadLabel })}
          </div>
          <Progress
            percent={downloadPercent}
            status="active"
            showInfo={downloadPercent > 0}
          />
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
  const { modal, message } = App.useApp();
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const [apps, setApps] = useState<IAppInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detailApp, setDetailApp] = useState<IAppInfo | null>(null);
  const [detailDocHasContent, setDetailDocHasContent] = useState(false);
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

  const copy = (text: string) =>
    copyText(text, t("common.copied"), t("common.copyFailed"));

  /** Localized status label; unknown backend states fall back verbatim. */
  const statusLabel = (app: IAppInfo, isActive: boolean) => {
    const raw = statusText(app, isActive);
    return t(`apps.status.${raw}`, { defaultValue: raw });
  };

  const fetchList = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const res = await getAppListApi();
      if (isOk(res)) {
        setApps(res.data.apps);
        setActiveId(res.data.current || null);
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
    const doActivate = async () => {
      setSwitching(app.id);
      try {
        const res = await switchAppApi({ app_id: app.id });
        if (isOk(res)) {
          message.success(t("apps.activated", { name }));
          // Jump to the Live Debug page so the just-activated app's stream
          // and results are front and center.
          navigate("/live");
        } else if (Number(res.code) === -3) {
          // Backend Node-RED mode gate (P6): app operations are refused.
          message.warning(t("runtimeMode.actionUnavailable"));
        } else {
          message.error(res.msg || t("apps.activateFailed"));
        }
      } catch (e) {
        message.error(t("apps.activateFailed"));
      } finally {
        setSwitching(null);
        fetchList(true);
      }
    };
    // Only confirm when another app is actually running and would be
    // interrupted (the warning is about stopping the current app + its
    // RTSP/MQTT). With nothing running there is nothing to interrupt, so
    // activate directly.
    if (!activeId) {
      doActivate();
      return;
    }
    modal.confirm({
      title: t("apps.activateTitle", { name }),
      icon: <ExclamationCircleOutlined />,
      content: t("apps.activateContent"),
      okText: t("common.activate"),
      cancelText: t("common.cancel"),
      onOk: doActivate,
    });
  };

  const onUninstall = (app: IAppInfo) => {
    const label = pickLocalized(app, "name") || app.id;
    modal.confirm({
      title: t("apps.uninstall.title", { name: label }),
      icon: <ExclamationCircleOutlined />,
      content: t("apps.uninstall.content"),
      okText: t("apps.uninstall.confirm"),
      okButtonProps: { danger: true },
      cancelText: t("common.cancel"),
      onOk: async () => {
        setSwitching(app.id);
        try {
          const res = await uninstallAppApi({ app_id: app.id });
          if (isOk(res)) {
            message.success(t("apps.uninstall.done", { name: label }));
          } else if (isBusy(res)) {
            message.error(t("apps.install.busy"));
          } else {
            // opkg's own words are far more useful than a generic failure —
            // "depends on" and "not installed" both land here.
            const out = (res.data as IInstallAppResult | undefined)?.output;
            message.error(out?.trim() || res.msg || t("apps.uninstall.failed"));
          }
        } catch (e) {
          message.error(t("apps.uninstall.failed"));
        } finally {
          setSwitching(null);
          fetchList(true);
        }
      },
    });
  };

  const onStop = (app: IAppInfo) => {
    const name = pickLocalized(app, "name") || app.id;
    modal.confirm({
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
          if (isOk(res)) {
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

  // Catalog rows for apps already on the device are shown as installed rather
  // than hidden — a missing row reads as "not offered", which is what the
  // built-in-manifest problem looked like from the user's side.
  const installedIds = useMemo(
    () => new Set(apps.map((a) => a.id)),
    [apps]
  );

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
        installedIds={installedIds}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-16 mt-24">
            {sortedApps.map((app) => {
              const isActive = app.id === activeId;
              const displayName = pickLocalized(app, "name") || app.id;
              const tagsLine = getAppTags(app);
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
                  {app.image && (
                    <div className="-mx-20 -mt-20 mb-16 w-auto aspect-[2/1] overflow-hidden rounded-t-[12px] rc-card-surface">
                      <img
                        src={app.image}
                        alt=""
                        loading="lazy"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          const p = e.currentTarget.parentElement;
                          if (p) p.style.display = "none";
                        }}
                      />
                    </div>
                  )}
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
                    {tagsLine && (
                      <div className="text-muted text-12 mt-2">{tagsLine}</div>
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
                    ) : noderedMode ? (
                      <Tooltip title={t("runtimeMode.actionUnavailable")}>
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
                    {/* Built-in apps have no package to remove; only the ones
                        that came from a .deb can be uninstalled. */}
                    {!noderedMode && (
                      <Button
                        size="small"
                        danger
                        className="ml-auto"
                        loading={switching === app.id}
                        disabled={switching !== null && switching !== app.id}
                        onClick={() => onUninstall(app)}
                      >
                        {t("apps.uninstall.action")}
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Spin>

      <Modal
        title={
          detailApp ? (
            <div className="flex items-baseline gap-8 flex-wrap pr-24">
              <span>{pickLocalized(detailApp, "name") || detailApp.id}</span>
              <span className="text-12 text-muted font-normal">
                {getAppTags(detailApp)}
              </span>
            </div>
          ) : (
            ""
          )
        }
        open={!!detailApp}
        onCancel={() => {
          setDetailApp(null);
          setDetailDocHasContent(false);
        }}
        footer={null}
        centered
        width={Math.min(920, window.innerWidth - 32)}
        styles={{ body: { maxHeight: "72vh", overflowY: "auto" } }}
      >
        {detailApp && (
          <div
            className={`grid grid-cols-1 gap-x-20 gap-y-16 text-13 items-start ${
              detailDocHasContent ? "md:grid-cols-2" : ""
            }`}
          >
            <div className="flex flex-col gap-16 min-w-0">
              <div>
                <div className="rc-section-label mb-4">
                  {t("apps.drawer.description")}
                </div>
                <div className="rc-prose">
                  {pickLocalized(detailApp, "description") || "-"}
                </div>
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
                    onClick={() => copy(resolveRtspUrl(detailApp))}
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
                    onClick={() => copy(detailApp.mqtt_topic || "")}
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
            </div>
            <div className={detailDocHasContent ? "min-w-0" : "hidden"}>
              <IntegrationDoc
                appId={detailApp.id}
                onHasContent={setDetailDocHasContent}
              />
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Applications;
