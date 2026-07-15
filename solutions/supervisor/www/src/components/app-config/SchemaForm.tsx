import { Button, Input, InputNumber, Select, Slider, Switch } from "antd";
import { AimOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import {
  IConfigSchema,
  IConfigItem,
  IConfigValues,
  ConfigValue,
  ILineValue,
  NormPoint,
} from "@/api/app/app";
import { pickLocalizedText } from "@/utils/appLocale";

/**
 * Schema-driven configuration form (manifest config_schema).
 *
 * Scalar types are edited inline; spatial types (zone/line) show a value
 * summary plus an "Edit on video" button — the actual drawing happens in
 * <SpatialEditor> overlaid on the live player (state lives in the page).
 * Dirty tracking / persistence live in useAppConfig; this component is
 * purely presentational.
 */

interface SchemaFormProps {
  schema: IConfigSchema;
  draft: IConfigValues;
  defaults: IConfigValues;
  dirty: boolean;
  saving: boolean;
  /** Key of the spatial item currently being edited on the video (if any). */
  editingKey?: string | null;
  onChange: (key: string, value: ConfigValue) => void;
  onEditSpatial: (item: IConfigItem) => void;
  onSave: () => void;
  onReset: () => void;
  /** When true, drop the rc-card wrapper + title header so the form can be
   *  embedded inside a Collapse panel (the panel provides card + label). */
  embedded?: boolean;
}

/** Effective value shown in the form: explicit draft wins, else default. */
function effective(
  item: IConfigItem,
  draft: IConfigValues,
  defaults: IConfigValues
): ConfigValue | undefined {
  if (Object.prototype.hasOwnProperty.call(draft, item.key)) {
    return draft[item.key];
  }
  return defaults[item.key];
}

function formatDefault(v: ConfigValue | undefined): string {
  if (v === undefined || v === null) return "";
  if (typeof v === "object") return "";
  return String(v);
}

const SchemaForm = ({
  schema,
  draft,
  defaults,
  dirty,
  saving,
  editingKey,
  onChange,
  onEditSpatial,
  onSave,
  onReset,
  embedded = false,
}: SchemaFormProps) => {
  const { t } = useTranslation();

  const renderScalar = (item: IConfigItem) => {
    const value = effective(item, draft, defaults);
    switch (item.type) {
      case "number": {
        const num = typeof value === "number" ? value : undefined;
        const hasRange =
          typeof item.min === "number" && typeof item.max === "number";
        return (
          <div className="flex items-center gap-12">
            {hasRange && (
              <Slider
                className="flex-1 m-0"
                min={item.min}
                max={item.max}
                step={item.step}
                value={num}
                onChange={(v) => onChange(item.key, v)}
              />
            )}
            <InputNumber
              size="small"
              className={hasRange ? "w-[88px] flex-none" : "w-full"}
              min={item.min}
              max={item.max}
              step={item.step}
              value={num}
              onChange={(v) => {
                if (typeof v === "number") onChange(item.key, v);
              }}
            />
          </div>
        );
      }
      case "boolean":
        return (
          <Switch
            checked={value === true}
            onChange={(checked) => onChange(item.key, checked)}
          />
        );
      case "enum":
        return (
          <Select
            size="small"
            className="w-full"
            value={typeof value === "string" ? value : undefined}
            onChange={(v) => onChange(item.key, v)}
            options={(item.options || []).map((o) => ({
              value: o,
              label: o,
            }))}
          />
        );
      case "string":
        return (
          <Input
            size="small"
            maxLength={item.maxLength ?? 256}
            value={typeof value === "string" ? value : ""}
            onChange={(e) => onChange(item.key, e.target.value)}
          />
        );
      default:
        return null;
    }
  };

  const renderSpatial = (item: IConfigItem) => {
    const value = effective(item, draft, defaults);
    let summary: string;
    if (item.type === "zone") {
      summary = Array.isArray(value)
        ? t("config.zoneVertices", { count: (value as NormPoint[]).length })
        : t("config.notSet");
    } else {
      const line = value as ILineValue | null | undefined;
      summary =
        line && typeof line === "object" && !Array.isArray(line)
          ? t("config.lineSet")
          : t("config.notSet");
    }
    const set = summary !== t("config.notSet");
    const line =
      item.type === "line" && set ? (value as ILineValue) : null;
    return (
      <div className="flex items-center justify-between gap-8 flex-wrap">
        <span className={`text-12 ${set ? "" : "text-muted"}`}>
          {summary}
          {line?.direction && (
            <span className="rc-mono text-11 text-muted"> · {line.direction}</span>
          )}
        </span>
        <div className="flex gap-8">
          {set && (
            <Button size="small" onClick={() => onChange(item.key, null)}>
              {t("config.clear")}
            </Button>
          )}
          <Button
            size="small"
            icon={<AimOutlined />}
            type={editingKey === item.key ? "primary" : "default"}
            onClick={() => onEditSpatial(item)}
          >
            {t("config.editOnVideo")}
          </Button>
        </div>
      </div>
    );
  };

  const body = (
    <>
      {!embedded && (
        <div className="flex items-center justify-between gap-8 mb-4">
          <span className="rc-section-label">{t("config.title")}</span>
          {dirty && (
            <span className="rc-badge accent">{t("config.unsaved")}</span>
          )}
        </div>
      )}

      {schema.groups.map((group, gi) => (
        <div
          key={group.key}
          className={
            embedded && gi === 0 ? "" : "mt-12 pt-12 border-t border-line"
          }
        >
          <div className="rc-section-label mb-10 opacity-70">
            {pickLocalizedText(group.title, group.title_zh) || group.key}
          </div>
          <div className="flex flex-col gap-14">
            {group.items.map((item) => {
              const isSpatial = item.type === "zone" || item.type === "line";
              const defaultText = formatDefault(item.default);
              return (
                <div key={item.key}>
                  <div className="flex items-center justify-between gap-8 mb-6">
                    <span className="text-13 font-medium">
                      {pickLocalizedText(item.title, item.title_zh) || item.key}
                    </span>
                    {item.type === "boolean" && renderScalar(item)}
                  </div>
                  {item.type !== "boolean" &&
                    (isSpatial ? renderSpatial(item) : renderScalar(item))}
                  {defaultText && (
                    <div className="text-11 text-muted mt-4">
                      {t("config.defaultHint", { value: defaultText })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <div className="flex justify-end gap-8 mt-16 pt-16 border-t border-line">
        <Button size="small" disabled={!dirty || saving} onClick={onReset}>
          {t("config.reset")}
        </Button>
        <Button
          size="small"
          type="primary"
          disabled={!dirty}
          loading={saving}
          onClick={onSave}
        >
          {t("common.save")}
        </Button>
      </div>
    </>
  );

  return embedded ? body : <div className="rc-card p-20">{body}</div>;
};

export default SchemaForm;
