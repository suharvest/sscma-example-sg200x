import { useEffect, useState } from "react";
import { Button, message } from "antd";
import { CopyOutlined, DownloadOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getIntegrationDocApi } from "@/api/app";

/**
 * Integration / output-format documentation block for an application.
 *
 * Fetches the markdown served by /api/appMgr/getIntegrationDoc and renders
 * it in the rc-card design language. Renders nothing while loading or when
 * the app has no doc installed (empty content).
 */
const IntegrationDoc = ({
  appId,
  className = "",
}: {
  appId?: string | null;
  className?: string;
}) => {
  const [content, setContent] = useState("");

  useEffect(() => {
    let cancelled = false;
    setContent("");
    if (!appId) return;
    getIntegrationDocApi(appId)
      .then((res) => {
        if (cancelled) return;
        if ((res.code === 0 || res.code === "0") && res.data?.content) {
          setContent(res.data.content);
        }
      })
      .catch(() => {
        // No doc / endpoint unavailable -> keep the section hidden.
      });
    return () => {
      cancelled = true;
    };
  }, [appId]);

  if (!appId || !content) return null;

  const onCopy = () => {
    navigator.clipboard
      ?.writeText(content)
      .then(() => message.success("Integration doc copied"))
      .catch(() => message.error("Copy failed"));
  };

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
        <span className="rc-section-label">Integration</span>
        <div className="flex gap-8">
          <Button size="small" icon={<CopyOutlined />} onClick={onCopy}>
            Copy
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={onExport}
          >
            Export .md
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
