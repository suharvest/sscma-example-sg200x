import { useEffect, useState } from "react";
import {
  Alert,
  App,
  Button,
  Collapse,
  Form,
  Input,
  InputNumber,
  Switch,
} from "antd";
import { CopyOutlined, CheckOutlined, DownOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { getHaConfigApi, setHaConfigApi, testHaConnectionApi } from "@/api/ha";
import { ISetHaConfigParams } from "@/api/ha/ha";
import { getOnvifConfigApi, setOnvifConfigApi } from "@/api/onvif";
import { ISetOnvifConfigParams } from "@/api/onvif/onvif";
import { getDeviceHost } from "@/utils/appStream";
import useConfigStore from "@/store/config";

interface IHaFormValues {
  enabled: boolean;
  broker_host: string;
  broker_port: number;
  username?: string;
  password?: string;
  discovery_prefix?: string;
}

/** Small copy-to-clipboard button with transient "copied" feedback. */
const CopyButton = ({ text }: { text: string }) => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      // clipboard API unavailable (http origin): fallback textarea trick
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(true);
    message.success(t("ha.copied"));
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      size="small"
      icon={copied ? <CheckOutlined /> : <CopyOutlined />}
      onClick={onCopy}
    >
      {t("ha.copy")}
    </Button>
  );
};

const CodeBlock = ({ text }: { text: string }) => (
  <div className="relative bg-black bg-opacity-5 rounded-8 p-12 mt-8 mb-16">
    <pre className="m-0 text-12 whitespace-pre-wrap break-all">{text}</pre>
    <div className="absolute" style={{ top: 8, right: 8 }}>
      <CopyButton text={text} />
    </div>
  </div>
);


/**
 * A collapsible integration section, collapsed by default.
 *
 * These sections are reference material -- stream URLs and YAML to paste into
 * somebody else's configuration -- and a page that opens with three screens of
 * it buries the switches that actually do something. Collapsed, the page reads
 * as a list of what this device can be integrated with, and the details are one
 * click away for whoever is actually wiring one up.
 */
const IntegrationSection = ({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-16 bg-white p-30 mt-12 mb-24">
      <div
        className="flex items-start justify-between cursor-pointer"
        onClick={() => setOpen((v) => !v)}
      >
        <div>
          <div className="font-bold text-18">{title}</div>
          <div className="text-black opacity-60 mt-4 text-13">{subtitle}</div>
        </div>
        <DownOutlined
          className="mt-6 ml-16 transition-transform"
          style={{
            transform: open ? "rotate(180deg)" : "none",
            opacity: 0.45,
            flexShrink: 0,
          }}
        />
      </div>
      {open && <div className="mt-16">{children}</div>}
    </div>
  );
};

/** Home Assistant integration card: MQTT config + RTSP stream how-to. */
const HomeAssistantCard = () => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [form] = Form.useForm<IHaFormValues>();

  // Node-RED mode hint: HA publishing is done by the Console application,
  // which is parked while Node-RED owns the camera. Config saved here is
  // persisted but only takes effect back in Console mode.
  const { galleryMode, deviceInfo } = useConfigStore();
  const modeKnown = Boolean(deviceInfo?.appName);
  const noderedMode = modeKnown && !galleryMode;

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [passwordSet, setPasswordSet] = useState(false);
  const enabled = Form.useWatch("enabled", form);

  const host = getDeviceHost();
  const rtspUrl = `rtsp://${host}:8554/live0`;
  const genericCameraExample = [
    "# Settings -> Devices & Services -> Add Integration -> Generic Camera",
    `stream_source: ${rtspUrl}`,
    "rtsp_transport: tcp",
    "verify_ssl: false",
  ].join("\n");
  const go2rtcExample = [
    "# go2rtc.yaml (or the go2rtc add-on configuration)",
    "streams:",
    `  recamera: ${rtspUrl}`,
  ].join("\n");

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const res = await getHaConfigApi();
      if (res.code === 0 || res.code === "0") {
        const d = res.data;
        setPasswordSet(Boolean(d.password_set));
        form.setFieldsValue({
          enabled: Boolean(d.enabled),
          broker_host: d.broker_host || "",
          broker_port: d.broker_port || 1883,
          username: d.username || "",
          password: "",
          discovery_prefix: d.discovery_prefix || "homeassistant",
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

  const buildParams = (values: IHaFormValues): ISetHaConfigParams => {
    const params: ISetHaConfigParams = {
      enabled: Boolean(values.enabled),
      broker_host: (values.broker_host || "").trim(),
      broker_port: Number(values.broker_port) || 1883,
      username: (values.username || "").trim(),
      discovery_prefix:
        (values.discovery_prefix || "").trim() || "homeassistant",
    };
    // Empty password box = keep the stored one (omit the field entirely).
    if (values.password) {
      params.password = values.password;
    }
    return params;
  };

  const onSave = async () => {
    let values: IHaFormValues;
    try {
      values = await form.validateFields();
    } catch (e) {
      return;
    }
    setSaving(true);
    try {
      const res = await setHaConfigApi(buildParams(values));
      if (res.code === 0 || res.code === "0") {
        setPasswordSet(Boolean(res.data?.password_set));
        form.setFieldValue("password", "");
        message.success(
          res.data?.restarted
            ? `${t("ha.saved")} ${t("ha.restarted")}`
            : t("ha.saved")
        );
      } else if (res.code === -2 || res.code === "-2") {
        message.warning(t("ha.busy"));
      } else {
        message.error(res.msg || t("ha.saveFailed"));
      }
    } catch (e) {
      message.error(t("ha.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  const onTest = async () => {
    let values: IHaFormValues;
    try {
      values = await form.validateFields(["broker_host", "broker_port"]);
      values = { ...form.getFieldsValue(), ...values };
    } catch (e) {
      return;
    }
    setTesting(true);
    try {
      const res = await testHaConnectionApi({
        broker_host: (values.broker_host || "").trim(),
        broker_port: Number(values.broker_port) || 1883,
        username: (values.username || "").trim(),
        // No password typed but one is stored -> test with the stored one.
        ...(values.password
          ? { password: values.password }
          : passwordSet
          ? { use_saved_password: true }
          : {}),
      });
      if (res.code === 0 || res.code === "0") {
        message.success(t("ha.testSuccess"));
      } else if (res.code === -2 || res.code === "-2") {
        message.warning(t("ha.busy"));
      } else {
        message.error(
          `${t("ha.testFailed")}${res.msg ? `: ${res.msg}` : ""}`
        );
      }
    } catch (e) {
      message.error(t("ha.testFailed"));
    } finally {
      setTesting(false);
    }
  };

  return (
    <IntegrationSection title={t("ha.title")} subtitle={t("ha.subtitle")}>

      {noderedMode && (
        <Alert
          className="mb-16"
          type="info"
          showIcon
          message={t("ha.noderedNotice")}
        />
      )}

      <div>
        <Form
          form={form}
          layout="vertical"
          disabled={loading}
          initialValues={{
            enabled: false,
            broker_port: 1883,
            discovery_prefix: "homeassistant",
          }}
        >
          <div className="flex justify-between items-center mb-16">
            <span className="font-bold text-16">{t("ha.enabled")}</span>
            <Form.Item name="enabled" valuePropName="checked" noStyle>
              <Switch />
            </Form.Item>
          </div>

          <Form.Item
            name="broker_host"
            label={t("ha.brokerHost")}
            rules={[
              {
                required: Boolean(enabled),
                whitespace: true,
                message: t("ha.invalidHost"),
              },
            ]}
          >
            <Input
              placeholder="homeassistant.local / 192.168.1.10"
              allowClear
              maxLength={253}
            />
          </Form.Item>

          <Form.Item
            name="broker_port"
            label={t("ha.brokerPort")}
            rules={[
              {
                required: true,
                type: "number",
                min: 1,
                max: 65535,
                message: t("ha.invalidPort"),
              },
            ]}
          >
            <InputNumber
              min={1}
              max={65535}
              precision={0}
              className="w-full"
              placeholder="1883"
            />
          </Form.Item>

          <Form.Item name="username" label={t("ha.username")}>
            <Input placeholder="" allowClear maxLength={128} autoComplete="off" />
          </Form.Item>

          <Form.Item
            name="password"
            label={t("ha.password")}
            extra={passwordSet ? t("ha.passwordUnchanged") : undefined}
          >
            <Input.Password
              placeholder={passwordSet ? t("ha.passwordUnchanged") : ""}
              allowClear
              maxLength={128}
              autoComplete="new-password"
            />
          </Form.Item>

          <Collapse
            ghost
            items={[
              {
                key: "advanced",
                label: t("ha.advanced"),
                children: (
                  <Form.Item
                    name="discovery_prefix"
                    label={t("ha.discoveryPrefix")}
                    extra={t("ha.discoveryPrefixHint")}
                  >
                    <Input placeholder="homeassistant" allowClear maxLength={64} />
                  </Form.Item>
                ),
              },
            ]}
          />

          <div className="flex justify-center mt-16" style={{ gap: 12 }}>
            <Button onClick={onTest} loading={testing} disabled={loading}>
              {testing ? t("ha.testing") : t("ha.test")}
            </Button>
            <Button
              type="primary"
              onClick={onSave}
              loading={saving}
              disabled={loading}
            >
              {t("ha.save")}
            </Button>
          </div>
        </Form>
      </div>

      <div className="font-bold text-16 mt-24">{t("ha.discoveryTitle")}</div>
      <div className="mt-12">
        <Alert
          type="info"
          showIcon
          message={t("ha.discoveryHint")}
        />
      </div>

      {/* The bare stream URL has its own section: it is not a Home Assistant
          thing, and burying it here made it look like one. What stays are the
          two recipes that only make sense inside Home Assistant. */}
      <div className="font-bold text-16 mt-24">{t("ha.gcTitle")}</div>
      <div className="text-black opacity-60 text-13">{t("ha.gcHint")}</div>
      <CodeBlock text={genericCameraExample} />

      <div className="font-bold text-16 mt-24">{t("ha.go2rtcTitle")}</div>
      <div className="text-black opacity-60 text-13">{t("ha.go2rtcHint")}</div>
      <CodeBlock text={go2rtcExample} />
    </IntegrationSection>
  );
};

/**
 * The RTSP stream on its own, ahead of the integrations that consume it.
 *
 * Anything that speaks RTSP -- VLC, a VMS, ffmpeg, a Home Assistant Generic
 * Camera -- needs exactly this one line, so it is the first thing on the page
 * and it belongs to no particular integration.
 */
const RtspCard = () => {
  const { t } = useTranslation();
  const rtspUrl = `rtsp://${getDeviceHost()}:8554/live0`;
  return (
    <IntegrationSection title={t("rtsp.title")} subtitle={t("rtsp.subtitle")}>
      <div className="flex items-center justify-between bg-black bg-opacity-5 rounded-8 p-12">
        <code className="text-13 break-all mr-12">{rtspUrl}</code>
        <CopyButton text={rtspUrl} />
      </div>
      <div className="text-black opacity-60 mt-8 text-13">{t("rtsp.hint")}</div>
    </IntegrationSection>
  );
};

interface IOnvifFormValues {
  service_enabled: boolean;
  service_port: number;
  username?: string;
  password?: string;
  location?: string;
  meta_enabled: boolean;
  meta_interval_ms: number;
  meta_profile?: string;
  meta_prefix?: string;
}

const ONVIF_DEFAULT_PORT = 8000;
const ONVIF_DEFAULT_INTERVAL = 200;
const ONVIF_DEFAULT_PROFILE = "live0";

/**
 * ONVIF integration card.
 *
 * Two switches that are deliberately independent: the device service (so a VMS
 * can discover the camera and pull the stream) and the analytics metadata
 * stream (inference results shaped like ONVIF, on their own MQTT topic).
 * Users routinely want one without the other, so they are rendered as two
 * separate blocks rather than one master toggle with sub-options.
 */
const OnvifCard = () => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [form] = Form.useForm<IOnvifFormValues>();

  // Same caveat as HA: the ONVIF service lives in the Console application, so
  // it is parked while Node-RED owns the camera.
  const { galleryMode, deviceInfo } = useConfigStore();
  const modeKnown = Boolean(deviceInfo?.appName);
  const noderedMode = modeKnown && !galleryMode;

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [passwordSet, setPasswordSet] = useState(false);
  const serviceEnabled = Form.useWatch("service_enabled", form);
  const servicePort = Form.useWatch("service_port", form);

  const host = getDeviceHost();
  // The VMS needs the device service endpoint; keep it in sync with whatever
  // port is currently in the form so the user can copy it before saving.
  const deviceServiceUrl = `http://${host}:${
    servicePort || ONVIF_DEFAULT_PORT
  }/onvif/device_service`;

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const res = await getOnvifConfigApi();
      if (res.code === 0 || res.code === "0") {
        const d = res.data;
        setPasswordSet(Boolean(d.password_set));
        form.setFieldsValue({
          service_enabled: Boolean(d.service_enabled),
          service_port: d.service_port || ONVIF_DEFAULT_PORT,
          username: d.username || "",
          password: "",
          location: d.location || "",
          meta_enabled: Boolean(d.meta_enabled),
          meta_interval_ms: d.meta_interval_ms || ONVIF_DEFAULT_INTERVAL,
          meta_profile: d.meta_profile || ONVIF_DEFAULT_PROFILE,
          meta_prefix: d.meta_prefix || "",
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

  const buildParams = (values: IOnvifFormValues): ISetOnvifConfigParams => {
    const params: ISetOnvifConfigParams = {
      service_enabled: Boolean(values.service_enabled),
      service_port: Number(values.service_port) || ONVIF_DEFAULT_PORT,
      username: (values.username || "").trim(),
      location: (values.location || "").trim(),
      meta_enabled: Boolean(values.meta_enabled),
      meta_interval_ms:
        Number(values.meta_interval_ms) || ONVIF_DEFAULT_INTERVAL,
      meta_profile: (values.meta_profile || "").trim() || ONVIF_DEFAULT_PROFILE,
      meta_prefix: (values.meta_prefix || "").trim(),
    };
    // Empty password box = keep the stored one (omit the field entirely).
    // Clearing a stored password is done from the "clear" link below.
    if (values.password) {
      params.password = values.password;
    }
    return params;
  };

  const submit = async (params: ISetOnvifConfigParams) => {
    setSaving(true);
    try {
      const res = await setOnvifConfigApi(params);
      if (res.code === 0 || res.code === "0") {
        form.setFieldValue("password", "");
        message.success(
          res.data?.restarted
            ? `${t("onvif.saved")} ${t("onvif.restarted")}`
            : t("onvif.saved")
        );
        // Re-read so password_set / server-side clamping is reflected.
        fetchConfig();
      } else if (res.code === -2 || res.code === "-2") {
        message.warning(t("onvif.busy"));
      } else {
        message.error(res.msg || t("onvif.saveFailed"));
      }
    } catch (e) {
      message.error(t("onvif.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  const onSave = async () => {
    let values: IOnvifFormValues;
    try {
      values = await form.validateFields();
    } catch (e) {
      return;
    }
    submit(buildParams(values));
  };

  // Explicit password removal: the API distinguishes "omitted" (keep) from
  // "" (clear), which an empty text box alone cannot express.
  const onClearPassword = async () => {
    let values: IOnvifFormValues;
    try {
      values = await form.validateFields();
    } catch (e) {
      return;
    }
    submit({ ...buildParams(values), password: "" });
  };

  return (
    <IntegrationSection title={t("onvif.title")} subtitle={t("onvif.subtitle")}>

      {noderedMode && (
        <Alert
          className="mb-16"
          type="info"
          showIcon
          message={t("onvif.noderedNotice")}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        disabled={loading}
        initialValues={{
          service_enabled: false,
          service_port: ONVIF_DEFAULT_PORT,
          meta_enabled: false,
          meta_interval_ms: ONVIF_DEFAULT_INTERVAL,
          meta_profile: ONVIF_DEFAULT_PROFILE,
        }}
      >
        {/* ---- 1. ONVIF device service ---- */}
        <div className="flex justify-between items-center">
          <span className="font-bold text-16">{t("onvif.serviceEnabled")}</span>
          <Form.Item name="service_enabled" valuePropName="checked" noStyle>
            <Switch />
          </Form.Item>
        </div>
        <div className="text-black opacity-60 mt-4 mb-8 text-13">
          {t("onvif.serviceHint")}
        </div>

        {serviceEnabled && (
          <div className="mb-8">
            <div className="text-black opacity-60 mb-8 text-13">
              {t("onvif.deviceServiceHint")}
            </div>
            <div className="flex items-center justify-between bg-black bg-opacity-5 rounded-8 p-12">
              <code className="text-13 break-all mr-12">
                {deviceServiceUrl}
              </code>
              <CopyButton text={deviceServiceUrl} />
            </div>
          </div>
        )}

        <Collapse
          ghost
          items={[
            {
              key: "service",
              label: t("onvif.serviceAdvanced"),
              children: (
                <>
                  <Form.Item
                    name="service_port"
                    label={t("onvif.servicePort")}
                    rules={[
                      {
                        required: true,
                        type: "number",
                        min: 1025,
                        max: 65535,
                        message: t("onvif.invalidPort"),
                      },
                    ]}
                  >
                    <InputNumber
                      min={1025}
                      max={65535}
                      precision={0}
                      className="w-full"
                      placeholder={String(ONVIF_DEFAULT_PORT)}
                    />
                  </Form.Item>

                  <Form.Item
                    name="username"
                    label={t("onvif.username")}
                    extra={t("onvif.credentialsHint")}
                  >
                    <Input
                      placeholder=""
                      allowClear
                      maxLength={128}
                      autoComplete="off"
                    />
                  </Form.Item>

                  <Form.Item
                    name="password"
                    label={t("onvif.password")}
                    extra={
                      passwordSet ? t("onvif.passwordUnchanged") : undefined
                    }
                  >
                    <Input.Password
                      placeholder={
                        passwordSet ? t("onvif.passwordUnchanged") : ""
                      }
                      allowClear
                      maxLength={128}
                      autoComplete="new-password"
                    />
                  </Form.Item>

                  {passwordSet && (
                    <div className="mb-16" style={{ marginTop: -16 }}>
                      <Button
                        type="link"
                        size="small"
                        className="p-0"
                        onClick={onClearPassword}
                        disabled={loading || saving}
                      >
                        {t("onvif.clearPassword")}
                      </Button>
                    </div>
                  )}

                  <Form.Item
                    name="location"
                    label={t("onvif.location")}
                    extra={t("onvif.locationHint")}
                  >
                    <Input
                      placeholder="city/Shenzhen"
                      allowClear
                      maxLength={128}
                    />
                  </Form.Item>
                </>
              ),
            },
          ]}
        />

        {/* ---- 2. ONVIF analytics metadata (independent of the service) ---- */}
        <div className="flex justify-between items-center mt-16">
          <span className="font-bold text-16">{t("onvif.metaEnabled")}</span>
          <Form.Item name="meta_enabled" valuePropName="checked" noStyle>
            <Switch />
          </Form.Item>
        </div>
        <div className="text-black opacity-60 mt-4 mb-8 text-13">
          {t("onvif.metaHint")}
        </div>
        <div className="text-black opacity-60 mb-8 text-13">
          {t("onvif.metaExtraTopicHint")}
        </div>

        <Collapse
          ghost
          items={[
            {
              key: "meta",
              label: t("onvif.metaAdvanced"),
              children: (
                <>
                  <Form.Item
                    name="meta_interval_ms"
                    label={t("onvif.metaInterval")}
                    extra={t("onvif.metaIntervalHint")}
                    rules={[
                      {
                        required: true,
                        type: "number",
                        min: 20,
                        max: 60000,
                        message: t("onvif.invalidInterval"),
                      },
                    ]}
                  >
                    <InputNumber
                      min={20}
                      max={60000}
                      precision={0}
                      className="w-full"
                      placeholder={String(ONVIF_DEFAULT_INTERVAL)}
                    />
                  </Form.Item>

                  <Form.Item
                    name="meta_profile"
                    label={t("onvif.metaProfile")}
                    extra={t("onvif.metaProfileHint")}
                  >
                    <Input
                      placeholder={ONVIF_DEFAULT_PROFILE}
                      allowClear
                      maxLength={64}
                    />
                  </Form.Item>

                  <Form.Item
                    name="meta_prefix"
                    label={t("onvif.metaPrefix")}
                    extra={t("onvif.metaPrefixHint")}
                  >
                    <Input placeholder="" allowClear maxLength={128} />
                  </Form.Item>
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
            {t("onvif.save")}
          </Button>
        </div>
      </Form>

      <div className="mt-24">
        {/* "ONVIF-compatible", never "conformant": we ship no certification. */}
        <Alert type="info" showIcon message={t("onvif.complianceNote")} />
      </div>
    </IntegrationSection>
  );
};




/**
 * Integration cards shown on the page, in order.
 * Add a new integration by appending a { key, Card } entry here.
 */
const integrationCards: { key: string; Card: () => JSX.Element }[] = [
  { key: "rtsp", Card: RtspCard },
  { key: "home-assistant", Card: HomeAssistantCard },
  { key: "onvif", Card: OnvifCard },
];

const Integrations = () => {
  const { t } = useTranslation();

  return (
    <div className="rc-page-narrow my-8 p-16">
      <div className="font-bold text-18">{t("integrations.title")}</div>
      <div className="text-black opacity-60 mt-4 mb-12 text-13">
        {t("integrations.subtitle")}
      </div>

      {integrationCards.map(({ key, Card }) => (
        <Card key={key} />
      ))}
    </div>
  );
};

export default Integrations;
