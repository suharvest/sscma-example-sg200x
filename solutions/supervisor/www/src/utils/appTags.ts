/**
 * Scene-tag line shown under an application's name (card / live info bar).
 *
 * Source of truth is the manifest's optional `tags` / `tags_zh` fields
 * (single pre-joined string, e.g. "Perimeter security · Parking"). Apps
 * without manifest tags fall back to this built-in per-app-id wordlist;
 * unknown apps get an empty string (the row is simply not rendered).
 */
import { isZhLang } from "@/i18n";

interface TaggableApp {
  id?: string;
  tags?: string;
  tags_zh?: string;
}

const BUILTIN_TAGS: Record<string, { en: string; zh: string }> = {
  "yolo-detector": {
    en: "Perimeter security · Parking · Deliveries · Asset checks",
    zh: "周界安防 · 车位监控 · 快递到件 · 资产巡检",
  },
  "face-analysis": {
    en: "Store audience profiling · Showroom analytics · Campaign feedback",
    zh: "门店客群画像 · 展厅观众分析 · 活动效果评估",
  },
  "retail-vision": {
    en: "Footfall counting · Dwell hotspots · Staff dispatch",
    zh: "客流统计 · 停留热区 · 导购调度",
  },
  "ppocr-reader": {
    en: "Meter reading · Logistics labels · Station tags",
    zh: "仪表抄读 · 物流面单 · 工位标签",
  },
  "facemesh-reader": {
    en: "Long-haul driving · Duty posts · Heavy machinery",
    zh: "长途驾驶 · 值守岗位 · 工程机械",
  },
  "weather-classifier": {
    en: "Greenhouse control · Yard automation · Outdoor protection",
    zh: "温室联动 · 庭院自动化 · 户外设备保护",
  },
};

/** The scene-tag line for an app in the current UI language ("" = none). */
export function getAppTags(app: TaggableApp | null | undefined): string {
  if (!app) return "";
  const zh = isZhLang();
  const fromManifest = zh ? app.tags_zh || app.tags : app.tags;
  if (fromManifest) return fromManifest;
  const builtin = app.id ? BUILTIN_TAGS[app.id] : undefined;
  if (!builtin) return "";
  return zh ? builtin.zh : builtin.en;
}
