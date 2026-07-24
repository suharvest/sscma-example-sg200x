import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import useConfigStore from "@/store/config";

function WebShell() {
  const { t } = useTranslation();
  const { deviceInfo } = useConfigStore();
  const [iframeUrl, setIframeUrl] = useState("");
  useEffect(() => {
    if (deviceInfo) {
      const url =
        import.meta.env.MODE === "development"
          ? "http://192.168.120.99"
          : window.location.origin;
      setIframeUrl(`${url}:${deviceInfo.terminalPort}`);
    }
  }, [deviceInfo]);
  return (
    <div className="py-24 pb-40">
      <div className="flex items-start justify-between gap-12 flex-wrap">
        <div>
          <div className="rc-eyebrow mb-4">{t("terminal.eyebrow")}</div>
          <h1 className="font-display font-bold text-24 m-0 tracking-tight">
            {t("terminal.title")}
          </h1>
          <p className="text-muted text-13 mt-4 mb-0 rc-prose">
            {t("terminal.subtitle")}
          </p>
        </div>
      </div>
      <div
        className="mt-24 rounded-12 overflow-hidden border border-line bg-black"
        style={{ height: "calc(100vh - 280px)", minHeight: 420 }}
      >
        <iframe
          src={iframeUrl}
          style={{ width: "100%", height: "100%", border: "none" }}
        ></iframe>
      </div>
    </div>
  );
}

export default WebShell;
