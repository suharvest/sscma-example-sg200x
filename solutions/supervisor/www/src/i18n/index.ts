/**
 * i18n bootstrap (react-i18next, bundled locally — no CDN).
 *
 * Detection order: localStorage('rc-lang') > navigator.language
 * (zh* -> zh-CN, everything else -> en-US).
 */
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import enUS from "@/locales/en-US.json";
import zhCN from "@/locales/zh-CN.json";

export const LANG_STORAGE_KEY = "rc-lang";

export type AppLanguage = "en-US" | "zh-CN";

export function detectLanguage(): AppLanguage {
  try {
    const stored = localStorage.getItem(LANG_STORAGE_KEY);
    if (stored === "en-US" || stored === "zh-CN") return stored;
  } catch {
    // localStorage unavailable (e.g. privacy mode) — fall through.
  }
  const nav = (navigator.language || "").toLowerCase();
  return nav.startsWith("zh") ? "zh-CN" : "en-US";
}

i18n.use(initReactI18next).init({
  resources: {
    "en-US": { translation: enUS },
    "zh-CN": { translation: zhCN },
  },
  lng: detectLanguage(),
  fallbackLng: "en-US",
  interpolation: {
    // React already escapes rendered strings.
    escapeValue: false,
  },
});

export function setLanguage(lng: AppLanguage) {
  i18n.changeLanguage(lng);
  try {
    localStorage.setItem(LANG_STORAGE_KEY, lng);
  } catch {
    // Non-fatal: language just won't persist.
  }
}

/** True when the active UI language is Chinese. */
export function isZhLang(): boolean {
  return (i18n.language || "").toLowerCase().startsWith("zh");
}

/** Two-letter language code sent to backend APIs (getIntegrationDoc etc.). */
export function apiLang(): "zh" | "en" {
  return isZhLang() ? "zh" : "en";
}

export default i18n;
