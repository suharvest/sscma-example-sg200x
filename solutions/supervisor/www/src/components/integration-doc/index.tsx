import { useEffect, useState } from "react";
import { Button } from "antd";
import { CopyOutlined, DownloadOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useTranslation } from "react-i18next";
import { getIntegrationDocApi } from "@/api/app";
import { isOk } from "@/utils/api";
import { copyText } from "@/utils/clipboard";
import { apiLang } from "@/i18n";

/**
 * Integration / output-format documentation block for an application.
 *
 * Fetches the markdown served by /api/appMgr/getIntegrationDoc and renders
 * it in the rc-card design language. Renders nothing while loading or when
 * the app has no doc installed (empty content).
 *
 * The current UI language is passed as `lang` (zh|en); the backend serves
 * the <id>.zh.md variant when present and falls back to English itself —
 * no second request from here.
 */
const IntegrationDoc = ({
  appId,
  className = "",
  onHasContent,
}: {
  appId?: string | null;
  className?: string;
  onHasContent?: (has: boolean) => void;
}) => {
  const { t, i18n } = useTranslation();
  const [content, setContent] = useState("");

  useEffect(() => {
    let cancelled = false;
    setContent("");
    onHasContent?.(false);
    if (!appId) return;
    getIntegrationDocApi(appId, apiLang())
      .then((res) => {
        if (cancelled) return;
        if (isOk(res) && res.data?.content) {
          // Strip the frontmatter HTML comment (<!-- app: ... -->) and any
          // other HTML comments so they never leak as visible text in the
          // rendered markdown.
          setContent(
            res.data.content.replace(/<!--[\s\S]*?-->/g, "").replace(/^\s+/, "")
          );
          onHasContent?.(true);
        } else {
          onHasContent?.(false);
        }
      })
      .catch(() => {
        // No doc / endpoint unavailable -> keep the section hidden.
        onHasContent?.(false);
      });
    return () => {
      cancelled = true;
    };
  }, [appId, i18n.language]);

  if (!appId || !content) return null;

  const onCopy = () =>
    copyText(content, t("doc.copied"), t("common.copyFailed"));

  const onExport = () => {
    const blob = new Blob([content], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${appId}-integration.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`rc-card ${className}`}>
      <div className="flex items-center justify-between gap-8 flex-wrap px-20 py-14 border-b border-line">
        <span className="rc-section-label">{t("doc.integration")}</span>
        <div className="flex gap-8">
          <Button size="small" icon={<CopyOutlined />} onClick={onCopy}>
            {t("common.copy")}
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={onExport}
          >
            {t("doc.exportMd")}
          </Button>
        </div>
      </div>
      <div className="px-20 py-16">
        <div className="rc-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div className="rc-markdown-table-wrap">
                  <table>{children}</table>
                </div>
              ),
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default IntegrationDoc;
