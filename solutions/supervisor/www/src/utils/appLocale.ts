/**
 * Bilingual app-manifest field helpers.
 *
 * Manifests may carry optional `*_zh` variants (name_zh / description_zh /
 * scene_zh). In Chinese UI the *_zh field wins; missing *_zh falls back to
 * the English field. All display sites should go through pickLocalized so
 * the fallback rule lives in one place.
 */
import { isZhLang } from "@/i18n";

export type LocalizableField = "name" | "description" | "scene";

interface LocalizableApp {
  name?: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  scene?: string;
  scene_zh?: string;
}

export function pickLocalized(
  app: LocalizableApp | null | undefined,
  field: LocalizableField
): string {
  if (!app) return "";
  const en = app[field] || "";
  const zh = app[`${field}_zh` as const] || "";
  return isZhLang() && zh ? zh : en;
}

/**
 * Generic bilingual pick for ad-hoc title/title_zh pairs (config_schema
 * groups/items etc.) — same fallback rule as pickLocalized: in Chinese UI
 * the zh variant wins when present, otherwise the English text.
 */
export function pickLocalizedText(en?: string, zh?: string): string {
  return isZhLang() && zh ? zh : en || "";
}

/**
 * The "other language" variant, for secondary display next to the primary
 * name (e.g. English name under the Chinese one). Empty when there is no
 * distinct alternative.
 */
export function pickLocalizedAlt(
  app: LocalizableApp | null | undefined,
  field: LocalizableField
): string {
  if (!app) return "";
  const primary = pickLocalized(app, field);
  const en = app[field] || "";
  const zh = app[`${field}_zh` as const] || "";
  const alt = primary === zh ? en : zh;
  return alt && alt !== primary ? alt : "";
}
