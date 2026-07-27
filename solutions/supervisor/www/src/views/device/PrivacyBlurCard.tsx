import { useEffect, useState } from "react";
import {
  Alert,
  App,
  Button,
  Collapse,
  Form,
  InputNumber,
  Select,
  Slider,
  Switch,
} from "antd";
import { useTranslation } from "react-i18next";
import { getBlurConfigApi, setBlurConfigApi } from "@/api/blur";
import { IBlurConfig, ISetBlurConfigParams } from "@/api/blur/blur";
import useConfigStore from "@/store/config";

interface IBlurFormValues {
  enabled: boolean;
  backend: IBlurConfig["backend"];
  block_px: number;
  max_regions: number;
  alpha: number;
}

/*
 * Lives on the Device page, next to the driver card, because the two are halves
 * of one feature: this decides whether and how hard to mask, that decides
 * whether the masking is done by the camera hardware or by the CPU. Splitting
 * them across two pages -- masking under Integrations, the driver under Device
 * -- meant an operator who had just deployed the driver had nowhere obvious to
 * turn masking on.
 */

const BLUR_DEFAULT_BACKEND: IBlurConfig["backend"] = "pixelate";
const BLUR_DEFAULT_BLOCK_PX = 16;
const BLUR_DEFAULT_MAX_REGIONS = 8;
// Fully opaque. The mask is only a privacy measure as long as it actually
// hides the subject, so the default is the value that hides all of it.
const BLUR_DEFAULT_ALPHA = 255;
// Below this the subject starts showing through enough to be recognised again;
// the card warns from here down rather than silently letting it happen.
// The solid-colour backend can only bind four overlay regions, so a higher
// ceiling would quietly go unhonoured for it.
const COVEREX_MAX_REGIONS = 4;

/**
 * Privacy blur card.
 *
 * One master switch, because that is the only decision most users have: mask
 * the people the camera detects, or do not. The three backends differ enough
 * in appearance and in region budget that they cannot be auto-selected, but
 * they belong behind the advanced fold rather than in the user's face.
 */
const PrivacyBlurCard = () => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [form] = Form.useForm<IBlurFormValues>();

  // Same caveat as the other cards: masking is done by the Console
  // application, which is parked while Node-RED owns the camera.
  const { galleryMode, deviceInfo } = useConfigStore();
  const modeKnown = Boolean(deviceInfo?.appName);
  const noderedMode = modeKnown && !galleryMode;

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const backend = Form.useWatch("backend", form);
  const maxRegions = Form.useWatch("max_regions", form);

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const res = await getBlurConfigApi();
      if (res.code === 0 || res.code === "0") {
        const d = res.data;
        form.setFieldsValue({
          enabled: Boolean(d.enabled),
          backend: d.backend || BLUR_DEFAULT_BACKEND,
          block_px: d.block_px || BLUR_DEFAULT_BLOCK_PX,
          max_regions: d.max_regions || BLUR_DEFAULT_MAX_REGIONS,
          // `?? ` rather than `||`: alpha 0 is a legal (if unwise) stored
          // value and must not be rewritten to the opaque default here, or
          // the card would show a mask strength the device is not applying.
          alpha: d.alpha ?? BLUR_DEFAULT_ALPHA,
        });
      }
    } catch (e) {
      // request layer already surfaced the error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const buildParams = (values: IBlurFormValues): ISetBlurConfigParams => ({
    enabled: Boolean(values.enabled),
    backend: values.backend || BLUR_DEFAULT_BACKEND,
    block_px: Number(values.block_px) || BLUR_DEFAULT_BLOCK_PX,
    max_regions: Number(values.max_regions) || BLUR_DEFAULT_MAX_REGIONS,
    alpha: Number.isFinite(Number(values.alpha))
      ? Number(values.alpha)
      : BLUR_DEFAULT_ALPHA,
  });

  const onSave = async () => {
    let values: IBlurFormValues;
    try {
      values = await form.validateFields();
    } catch (e) {
      return;
    }
    setSaving(true);
    try {
      const res = await setBlurConfigApi(buildParams(values));
      if (res.code === 0 || res.code === "0") {
        message.success(
          res.data?.restarted
            ? `${t("blur.saved")} ${t("blur.restarted")}`
            : res.data?.note === "applied_live"
              ? `${t("blur.saved")} ${t("blur.appliedLive")}`
              : t("blur.saved")
        );
        // Re-read so the form always shows what the device actually stored.
        fetchConfig();
      } else if (res.code === -2 || res.code === "-2") {
        message.warning(t("blur.busy"));
      } else {
        message.error(res.msg || t("blur.saveFailed"));
      }
    } catch (e) {
      message.error(t("blur.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    /* Device-page card styling, not the Integrations page's: this card moved
       here and kept the old classes, so it rendered with no frame at all while
       every card beside it had one. */
    <div className="rc-card p-20">
      <div className="rc-section-label mb-12">{t("blur.title")}</div>
      <div className="text-black opacity-60 mb-16 text-13">
        {t("blur.subtitle")}
      </div>

      {noderedMode && (
        <Alert
          className="mb-16"
          type="info"
          showIcon
          message={t("blur.noderedNotice")}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        disabled={loading}
        initialValues={{
          enabled: false,
          backend: BLUR_DEFAULT_BACKEND,
          block_px: BLUR_DEFAULT_BLOCK_PX,
          max_regions: BLUR_DEFAULT_MAX_REGIONS,
          alpha: BLUR_DEFAULT_ALPHA,
        }}
      >
        <div className="flex justify-between items-center">
          <span className="font-bold text-16">{t("blur.enabled")}</span>
          <Form.Item name="enabled" valuePropName="checked" noStyle>
            <Switch />
          </Form.Item>
        </div>
        <div className="text-black opacity-60 mt-4 mb-8 text-13">
          {t("blur.enabledHint")}
        </div>
        <div className="text-black opacity-60 mb-8 text-13">
          {t("blur.regionsNote", {
            regions: maxRegions || BLUR_DEFAULT_MAX_REGIONS,
          })}
        </div>

        <Collapse
          ghost
          items={[
            {
              key: "advanced",
              label: t("blur.advanced"),
              children: (
                <>
                  <Form.Item
                    name="backend"
                    label={t("blur.backend")}
                    extra={t(`blur.backendHint.${backend || BLUR_DEFAULT_BACKEND}`)}
                  >
                    <Select
                      options={[
                        {
                          value: "pixelate",
                          label: t("blur.backendLabel.pixelate"),
                        },
                        {
                          value: "coverex",
                          label: t("blur.backendLabel.coverex"),
                        },
                        {
                          value: "mosaic",
                          label: t("blur.backendLabel.mosaic"),
                        },
                      ]}
                    />
                  </Form.Item>

                  <Form.Item
                    name="block_px"
                    label={t("blur.blockPx")}
                    extra={t("blur.blockPxHint")}
                  >
                    <Select
                      options={[
                        { value: 8, label: t("blur.blockPxFine") },
                        { value: 16, label: t("blur.blockPxCoarse") },
                      ]}
                    />
                  </Form.Item>

                  <Form.Item
                    name="max_regions"
                    label={t("blur.maxRegions")}
                    extra={t("blur.maxRegionsHint")}
                    rules={[
                      {
                        required: true,
                        type: "number",
                        min: 1,
                        max: 8,
                        message: t("blur.invalidMaxRegions"),
                      },
                    ]}
                  >
                    <InputNumber
                      min={1}
                      max={8}
                      precision={0}
                      className="w-full"
                      placeholder={String(BLUR_DEFAULT_MAX_REGIONS)}
                    />
                  </Form.Item>

                  <Form.Item
                    name="alpha"
                    label={t("blur.alpha")}
                    extra={t("blur.alphaHint")}
                    rules={[
                      {
                        required: true,
                        type: "number",
                        min: 0,
                        max: 255,
                        message: t("blur.invalidAlpha"),
                      },
                    ]}
                  >
                    <Slider
                      min={0}
                      max={255}
                      step={5}
                      marks={{
                        0: t("blur.alphaTransparent"),
                        255: t("blur.alphaOpaque"),
                      }}
                    />
                  </Form.Item>

                  {backend === "coverex" &&
                    Number(maxRegions) > COVEREX_MAX_REGIONS && (
                      <Alert
                        className="mb-16"
                        type="warning"
                        showIcon
                        message={t("blur.coverexLimitNotice")}
                      />
                    )}
                </>
              ),
            },
          ]}
        />

        <div className="flex justify-center mt-16">
          <Button
            type="primary"
            onClick={onSave}
            loading={saving}
            disabled={loading}
          >
            {t("blur.save")}
          </Button>
        </div>
      </Form>
    </div>
  );
};

export default PrivacyBlurCard;
