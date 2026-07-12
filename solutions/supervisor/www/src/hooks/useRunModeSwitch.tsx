import { useRef, useState } from "react";
import { Modal, message } from "antd";
import { ExclamationCircleOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import {
  queryDeviceInfoApi,
  setRunModeApi,
  forceConsoleApi,
} from "@/api/device/index";
import { isOk, isBusy } from "@/utils/api";

export type RunMode = "console" | "nodered";

/**
 * P4-D: Console <-> Node-RED runtime mode switch, shared by the System page
 * card and the Applications page banner.
 *
 * The backend endpoint is synchronous and does NOT restart the supervisor:
 * when it returns code 0 the services have been moved over and galleryMode
 * has already flipped in-process. We still re-query the public
 * queryDeviceInfo endpoint to confirm the flip (defense against a stale
 * response), then window.location.reload() so menu/pages re-render from a
 * clean store. The call itself can take 10-40s (service stop/start), so the
 * caller should bind `switching` to its button loading state.
 */
export function useRunModeSwitch() {
  const { t } = useTranslation();
  // target mode while a switch is in flight, null when idle
  const [switching, setSwitching] = useState<RunMode | null>(null);
  const busyRef = useRef(false); // sync guard: state updates are async

  const doSwitch = async (target: RunMode) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setSwitching(target);
    try {
      const res = await setRunModeApi(target);
      if (!isOk(res)) {
        if (isBusy(res)) {
          message.error(t("runtimeMode.busy"));
        } else {
          message.error(res.msg || t("runtimeMode.failed"));
        }
        return;
      }
      // Confirm the flip via the public queryDeviceInfo endpoint, then
      // reload. A few retries cover a transiently dropped request; the
      // flip itself already happened before the response above.
      const wantGallery = target === "console";
      for (let i = 0; i < 5; i++) {
        try {
          const info = await queryDeviceInfoApi();
          if (isOk(info) && Boolean(info.data?.galleryMode) === wantGallery) {
            window.location.reload();
            return;
          }
        } catch (e) {
          // transient — retry below
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      // Switch reported OK but we could not confirm the flip: reload anyway,
      // the app boots from queryDeviceInfo and will render the real state.
      window.location.reload();
    } catch (e) {
      message.error(t("runtimeMode.failed"));
    } finally {
      busyRef.current = false;
      setSwitching(null);
    }
  };

  /** Confirm dialog + switch. No-op when a switch is already in flight. */
  const requestSwitch = (target: RunMode) => {
    if (busyRef.current) return;
    Modal.confirm({
      title: t(
        target === "console"
          ? "runtimeMode.confirmTitleConsole"
          : "runtimeMode.confirmTitleNodered"
      ),
      icon: <ExclamationCircleOutlined />,
      content: t("runtimeMode.confirmContent"),
      okText: t("common.confirm"),
      cancelText: t("common.cancel"),
      onOk: () => doSwitch(target),
    });
  };

  // #20 hard escape hatch: force console without the graceful hand-over. Used
  // when Node-RED mode has wedged the device; the supervisor HTTP path is
  // protected from the OOM killer (#19) so it stays reachable.
  const [forcing, setForcing] = useState(false);

  const doForceConsole = async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setForcing(true);
    try {
      const res = await forceConsoleApi();
      if (!isOk(res)) {
        message.error(res.msg || t("runtimeMode.forceConsoleFailed"));
        return;
      }
      message.success(t("runtimeMode.forceConsoleDone"));
      // The flip already happened in-process; reload so menu/pages re-render
      // from a clean store reflecting console mode.
      setTimeout(() => window.location.reload(), 800);
    } catch (e) {
      message.error(t("runtimeMode.forceConsoleFailed"));
    } finally {
      busyRef.current = false;
      setForcing(false);
    }
  };

  /** Confirm dialog + force console. No-op when an operation is in flight. */
  const requestForceConsole = () => {
    if (busyRef.current) return;
    Modal.confirm({
      title: t("runtimeMode.forceConsoleConfirmTitle"),
      icon: <ExclamationCircleOutlined />,
      content: t("runtimeMode.forceConsoleConfirmContent"),
      okText: t("common.confirm"),
      cancelText: t("common.cancel"),
      okButtonProps: { danger: true },
      onOk: () => doForceConsole(),
    });
  };

  return { switching, requestSwitch, forcing, requestForceConsole };
}

export default useRunModeSwitch;
