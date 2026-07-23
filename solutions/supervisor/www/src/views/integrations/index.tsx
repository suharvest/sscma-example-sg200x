import { useEffect, useState } from "react";
import {
  App,
  Button,
  Collapse,
  Form,
  Input,
  InputNumber,
  Switch,
} from "antd";
import { CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { getHaConfigApi, setHaConfigApi, testHaConnectionApi } from "@/api/ha";
import { ISetHaConfigParams } from "@/api/ha/ha";
import { getDeviceHost } from "@/utils/appStream";

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

/** Home Assistant integration card: MQTT config + RTSP stream how-to. */
const HomeAssistantCard = () => {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [form] = Form.useForm<IHaFormValues>();

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
    <div className="rounded-16 bg-white p-30 mt-12 mb-24">
      <div className="font-bold text-18">{t("ha.title")}</div>
      <div className="text-black opacity-60 mt-4 mb-16 text-13">
        {t("ha.subtitle")}
      </div>

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

      <div className="font-bold text-16 mt-24">{t("ha.rtspTitle")}</div>
      <div className="mt-12">
        <div className="text-black opacity-60 mb-8 text-13">
          {t("ha.rtspHint")}
        </div>
        <div className="flex items-center justify-between bg-black bg-opacity-5 rounded-8 p-12 mb-16">
          <code className="text-13 break-all mr-12">{rtspUrl}</code>
          <CopyButton text={rtspUrl} />
        </div>

        <div className="font-bold mb-4">{t("ha.gcTitle")}</div>
        <div className="text-black opacity-60 text-13">{t("ha.gcHint")}</div>
        <CodeBlock text={genericCameraExample} />

        <div className="font-bold mb-4">{t("ha.go2rtcTitle")}</div>
        <div className="text-black opacity-60 text-13">
          {t("ha.go2rtcHint")}
        </div>
        <CodeBlock text={go2rtcExample} />
      </div>
    </div>
  );
};

/**
 * Integration cards shown on the page, in order.
 * Add a new integration by appending a { key, Card } entry here.
 */
const integrationCards: { key: string; Card: () => JSX.Element }[] = [
  { key: "home-assistant", Card: HomeAssistantCard },
];

const Integrations = () => {
  const { t } = useTranslation();

  return (
    <div className="my-8 p-16">
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
