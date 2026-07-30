import {
  LockOutlined,
  UserOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import { Button, Form, Input, Modal, message, Checkbox } from "antd";
import { useState } from "react";
import recameraLogo from "@/assets/images/recamera.png";
import useUserStore from "@/store/user";
import { encryptPassword } from "@/utils";
import { requiredTrimValidate, passwordRules } from "@/utils/validate";
import { loginApi, updateUserPasswordApi } from "@/api/user";
import { useTranslation } from "react-i18next";
import LangToggle from "@/components/lang-toggle";

const Login = () => {
  const { t } = useTranslation();
  const { firstLogin, updateFirstLogin, updateUserInfo } = useUserStore();

  const [form] = Form.useForm();
  const [messageApi, messageContextHolder] = message.useMessage();
  const [agreed, setAgreed] = useState(false);
  const [showAgreement, setShowAgreement] = useState(false);
  const [passwordErrorMsg, setPasswordErrorMsg] = useState<string | null>(null);
  const [changingPassword, setChangingPassword] = useState(false);

  const handleChangePassword = async () => {
    let fieldsValue: {
      oldpassword: string;
      newpassword: string;
      confirmpassword: string;
    };
    try {
      // surfaces the per-field rules instead of silently doing nothing
      fieldsValue = await form.validateFields([
        "oldpassword",
        "newpassword",
        "confirmpassword",
      ]);
    } catch {
      return;
    }

    setChangingPassword(true);
    try {
      const response = await updateUserPasswordApi(
        {
          oldPassword: encryptPassword(fieldsValue.oldpassword),
          newPassword: encryptPassword(fieldsValue.newpassword),
        },
        true
      );
      if (response.code == 0) {
        messageApi.success(t("login.changeSuccess"));
        form.resetFields(["oldpassword", "newpassword", "confirmpassword"]);
        updateFirstLogin(false);
      } else {
        messageApi.error(response.msg || t("login.changeFailed"));
      }
    } catch {
      // supervisorRequest rejects (and stays silent on 401) -- without this the
      // rejection escapes as an unhandled promise and the click looks like a no-op
      messageApi.error(t("login.changeFailed"));
    } finally {
      setChangingPassword(false);
    }
  };

  const loginAction = async (userName: string, password: string) => {
    try {
      const encryptedPassword = encryptPassword(password);
      const response = await loginApi({
        userName,
        password: encryptedPassword,
      });
      const code = response.code;
      const data = response.data;
      if (code === 0) {
        updateUserInfo({
          userName,
          password: encryptedPassword,
          token: data.token,
        });
        return { success: true };
      }
      // 统一错误信息
      let errorMsg = response.msg || t("login.loginFailed");
      if (code === -1 && data && typeof data.retryCount !== "undefined") {
        errorMsg =
          data.retryCount > 0
            ? t("login.attemptsRemaining", { count: data.retryCount })
            : t("login.accountLocked");
        return { success: false, errorMsg, usePasswordErrorMsg: true };
      }
      return { success: false, errorMsg, usePasswordErrorMsg: false };
    } catch (error) {
      return {
        success: false,
        errorMsg: t("login.loginFailed"),
        usePasswordErrorMsg: false,
      };
    }
  };

  const onFinish = async (values: { username: string; password: string }) => {
    const userName = values.username;
    const password = values.password;
    const result = await loginAction(userName, password);
    if (!result.success) {
      if (result.usePasswordErrorMsg) {
        setPasswordErrorMsg(result.errorMsg);
      } else {
        setPasswordErrorMsg(null);
        messageApi.error(result.errorMsg);
      }
    } else {
      setPasswordErrorMsg(null);
    }
  };

  const handleAcknowledge = () => {
    setShowAgreement(false);
    setAgreed(true);
  };

  const agreementTitle = (
    <div className="flex items-center">
      <InfoCircleOutlined className="text-primary text-24 mr-12" />
      <span>{t("login.agreementTitle")}</span>
    </div>
  );

  return (
    <div className="h-full flex flex-col justify-center items-center text-18 relative">
      <div className="absolute right-16 top-16">
        <LangToggle />
      </div>
      <img src={recameraLogo} className="w-300" />
      <div className="text-16 w-500 mt-40 mb-40">{t("login.welcome")}</div>
      <Form
        className="w-400"
        name="login"
        labelCol={{ span: 6 }}
        wrapperCol={{ span: 18 }}
        initialValues={{ username: "recamera" }}
        onFinish={onFinish}
      >
        <Form.Item
          name="username"
          label={t("login.username")}
          rules={[
            {
              required: true,
              message: t("login.usernameRequired"),
              whitespace: true,
            },
          ]}
        >
          <Input prefix={<UserOutlined />} placeholder={t("login.username")} />
        </Form.Item>
        <Form.Item
          name="password"
          label={t("login.password")}
          rules={[requiredTrimValidate()]}
          validateStatus={passwordErrorMsg ? "error" : undefined}
          help={passwordErrorMsg}
          extra={
            !passwordErrorMsg && (
              <>
                {t("login.firstTimeHintPrefix")}&nbsp;
                <span className="font-bold text-primary">"recamera"</span>
              </>
            )
          }
        >
          <Input.Password
            prefix={<LockOutlined />}
            placeholder={t("login.password")}
            visibilityToggle
            minLength={8}
            maxLength={32}
          />
        </Form.Item>
        <Form.Item className="w-full" noStyle>
          <Button block type="primary" htmlType="submit" disabled={!agreed}>
            {t("login.loginBtn")}
          </Button>
        </Form.Item>
      </Form>

      <div className="mt-16 w-750 flex items-center">
        <Checkbox
          checked={agreed}
          onChange={(e) => setAgreed(e.target.checked)}
          className="mt-1"
        />
        <div className="ml-10 text-14">
          {t("login.agreePrefix")}{" "}
          <span
            className="text-primary cursor-pointer underline"
            onClick={() => setShowAgreement(true)}
          >
            {t("login.agreementTitle")}
          </span>
          {t("login.agreeSuffix")}
        </div>
      </div>

      <Modal
        title={agreementTitle}
        open={showAgreement}
        closable={false}
        onCancel={() => setShowAgreement(false)}
        footer={
          <Button type="primary" onClick={handleAcknowledge}>
            {t("login.acknowledge")}
          </Button>
        }
        width={1000}
      >
        <div className="text-left text-14 leading-relaxed ml-36">
          <p className="text-15 font-medium mb-16">
            {t("login.agreement.intro")}
          </p>

          <h3 className="text-15 font-semibold mb-8 mt-24">
            {t("login.agreement.localTitle")}
          </h3>
          <p className="mb-4">{t("login.agreement.localIntro")}</p>
          <ul className="list-none p-0 mt-4">
            <li className="mb-4">• {t("login.agreement.local1")}</li>
            <li className="mb-4">• {t("login.agreement.local2")}</li>
            <li className="mb-4">• {t("login.agreement.local3")}</li>
          </ul>

          <h3 className="text-15 font-semibold mb-8 mt-24">
            {t("login.agreement.netTitle")}
          </h3>
          <p className="mb-4">{t("login.agreement.netIntro")}</p>
          <ul className="list-none p-0 mt-4">
            <li className="mb-4">• {t("login.agreement.net1")}</li>
            <li className="mb-4">• {t("login.agreement.net2")}</li>
            <li className="mb-4">• {t("login.agreement.net3")}</li>
            <li className="mb-4">• {t("login.agreement.net4")}</li>
          </ul>

          <h3 className="text-15 font-semibold mb-8 mt-24">
            {t("login.agreement.dataTitle")}
          </h3>
          <p className="mb-4">{t("login.agreement.dataIntro")}</p>
          <ul className="list-none p-0 mt-4">
            <li className="mb-4">• {t("login.agreement.data1")}</li>
            <li className="mb-4">• {t("login.agreement.data2")}</li>
            <li className="mb-4">• {t("login.agreement.data3")}</li>
          </ul>

          <h3 className="text-15 font-semibold mb-8 mt-24">
            {t("login.agreement.respTitle")}
          </h3>
          <p className="mb-4">{t("login.agreement.respIntro")}</p>
          <ul className="list-none p-0 mt-4">
            <li className="mb-4">• {t("login.agreement.resp1")}</li>
            <li className="mb-4">• {t("login.agreement.resp2")}</li>
            <li className="mb-4">• {t("login.agreement.resp3")}</li>
          </ul>
        </div>
      </Modal>

      <Modal
        title={t("login.changePasswordTitle")}
        open={firstLogin}
        closable={false}
        footer={
          <Button
            className="w-1/2 m-auto block"
            type="primary"
            loading={changingPassword}
            onClick={handleChangePassword}
          >
            {t("common.confirm")}
          </Button>
        }
      >
        <Form
          form={form}
          name="dependencies"
          autoComplete="off"
          style={{ maxWidth: 600 }}
          layout="vertical"
        >
          <Form.Item
            name="oldpassword"
            label={t("login.oldPassword")}
            rules={[requiredTrimValidate()]}
            extra={
              <>
                {t("login.oldPasswordExtraPrefix")}&nbsp;
                <span className="font-bold text-primary">"recamera"</span>
              </>
            }
          >
            <Input.Password placeholder="recamera" visibilityToggle minLength={8} maxLength={32} />
          </Form.Item>
          <Form.Item
            name="newpassword"
            label={t("login.newPassword")}
            rules={passwordRules}
            extra={t("login.newPasswordExtra")}
          >
            <Input.Password
              placeholder={t("login.newPasswordPlaceholder")}
              visibilityToggle
              minLength={8}
              maxLength={32}
            />
          </Form.Item>

          <Form.Item
            name="confirmpassword"
            label={t("login.confirmPassword")}
            dependencies={["newpassword"]}
            rules={[
              {
                required: true,
              },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("newpassword") === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(
                    new Error(t("login.passwordMismatch"))
                  );
                },
              }),
            ]}
          >
            <Input.Password
              placeholder={t("login.confirmPasswordPlaceholder")}
              visibilityToggle
              minLength={8}
              maxLength={32}
            />
          </Form.Item>
        </Form>
      </Modal>
      {messageContextHolder}
    </div>
  );
};

export default Login;
