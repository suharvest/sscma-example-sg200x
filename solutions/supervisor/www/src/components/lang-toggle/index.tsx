import { useTranslation } from "react-i18next";
import { setLanguage } from "@/i18n";

/**
 * Minimal EN / 中 language toggle, styled after the rc-mono badge system.
 * Persists the choice to localStorage('rc-lang') via setLanguage().
 */
const LangToggle = ({ className = "" }: { className?: string }) => {
  const { i18n } = useTranslation();
  const isZh = (i18n.language || "").toLowerCase().startsWith("zh");

  const item = (active: boolean) =>
    `cursor-pointer px-4 leading-none ${
      active ? "text-fg font-semibold" : "text-muted"
    }`;

  return (
    <span
      className={`rc-mono text-11 inline-flex items-center select-none ${className}`}
      title="Language / 语言"
    >
      <span className={item(!isZh)} onClick={() => setLanguage("en-US")}>
        EN
      </span>
      <span className="text-muted opacity-50">/</span>
      <span className={item(isZh)} onClick={() => setLanguage("zh-CN")}>
        中
      </span>
    </span>
  );
};

export default LangToggle;
