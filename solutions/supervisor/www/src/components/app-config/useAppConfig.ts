import { useCallback, useEffect, useMemo, useState } from "react";
import { message } from "antd";
import { useTranslation } from "react-i18next";
import { getConfigApi, setConfigApi } from "@/api/app";
import { IConfigSchema, IConfigValues, ConfigValue } from "@/api/app/app";
import { isOk, isBusy } from "@/utils/api";

/**
 * App configuration state (schema-driven form backing store).
 *
 * - `schema === null` -> the app declares no config_schema; hide the UI.
 * - `draft` starts as the saved values from the device; edits only touch
 *   the keys the user changed (unset keys keep falling back to `defaults`
 *   on the device side, so we do NOT bake defaults into the payload).
 * - `save()` POSTs the whole draft; backend validates per schema and
 *   restarts the app when it is the active one (code 0 + restarted flag).
 *   code -2 = another app operation holds the lock (busy).
 *   code -1 = validation/persist error with a specific message.
 */
export default function useAppConfig(appId?: string | null) {
  const { t } = useTranslation();
  const [schema, setSchema] = useState<IConfigSchema | null>(null);
  const [defaults, setDefaults] = useState<IConfigValues>({});
  const [saved, setSaved] = useState<IConfigValues>({});
  const [draft, setDraft] = useState<IConfigValues>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSchema(null);
    setDefaults({});
    setSaved({});
    setDraft({});
    if (!appId) return;
    getConfigApi(appId)
      .then((res) => {
        if (cancelled) return;
        if (isOk(res) && res.data?.schema) {
          setSchema(res.data.schema);
          setDefaults(res.data.defaults || {});
          setSaved(res.data.values || {});
          setDraft(res.data.values || {});
        }
      })
      .catch(() => {
        // Endpoint unavailable (older firmware) -> keep the card hidden.
      });
    return () => {
      cancelled = true;
    };
  }, [appId]);

  const setValue = useCallback((key: string, value: ConfigValue) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }, []);

  const reset = useCallback(() => {
    setDraft(saved);
  }, [saved]);

  const dirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(saved),
    [draft, saved]
  );

  /** Returns true on success so the caller can refresh app status. */
  const save = useCallback(async (): Promise<boolean> => {
    if (!appId) return false;
    setSaving(true);
    try {
      const res = await setConfigApi({ app_id: appId, values: draft });
      if (isOk(res)) {
        setSaved(draft);
        message.success(
          res.data?.restarted ? t("config.savedRestarting") : t("config.saved")
        );
        return true;
      }
      if (isBusy(res)) {
        message.warning(t("config.busy"));
      } else {
        message.error(res.msg || t("config.saveFailed"));
      }
      return false;
    } catch (e) {
      message.error(t("config.saveFailed"));
      return false;
    } finally {
      setSaving(false);
    }
  }, [appId, draft, t]);

  return {
    schema,
    defaults,
    draft,
    saving,
    dirty,
    setValue,
    reset,
    save,
  };
}
