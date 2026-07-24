import CommonPopup from "@/components/common-popup";
import { Button, Form, Input, Switch } from "antd";
import KeyImg from "@/assets/images/svg/key.svg";
import { DeleteOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { useData, IFormTypeEnum } from "./hook";
import moment from "moment";
import {
  requiredTrimValidate,
  publicKeyValidate,
  passwordRules,
} from "@/utils/validate";

const titleObj = {
  [IFormTypeEnum.Key]: "security.addKeyTitle",
  [IFormTypeEnum.Username]: "security.editUsernameTitle",
  [IFormTypeEnum.Password]: "security.changePasswordTitle",
  [IFormTypeEnum.DelKey]: "security.removeKeyTitle",
};

const Security = () => {
  const { t } = useTranslation();
  const {
    state,
    formRef,
    passwordFormRef,
    usernameFormRef,
    onCancel,
    onEdit,
    addSshKey,
    onPasswordFinish,
    onUsernameFinish,
    onDelete,
    onAddSshFinish,
    onDeleteFinish,
    setSShStatus,
  } = useData();

  const handleSShStatusChange = (checked: boolean) => {
    setSShStatus(checked);
  };

  return (
    <div className="rc-page-narrow my-8 p-16">
      <div className="font-bold text-18 ">{t("security.user")}</div>
      <div className="rounded-16 bg-white p-30 mt-12 mb-24">
        <div className="flex justify-between mb-16">
          <span className="text-3d">{t("security.username")}</span>
          <div className="flex">
            <span className="text-3d">{state.username}</span>
          </div>
        </div>
        <div className="flex justify-center">
          <Button
            color="primary"
            variant="outlined"
            onClick={() => onEdit(IFormTypeEnum.Password)}
          >
            {t("security.changePassword")}
          </Button>
        </div>
      </div>
      <div className="font-bold text-18 flex justify-between items-center">
        <span>{t("security.ssh")}</span>
        <Switch checked={state.sshEnabled} onChange={handleSShStatusChange} />
      </div>
      {state.sshEnabled && (
        <div className="rounded-16 bg-white p-30 mt-12 mb-24">
          {state.sshkeyList?.length ? (
            state.sshkeyList.map((item, index) => {
              return (
                <div
                  className="bg-white border rounded-20 p-16 flex mb-16"
                  key={item.id || index}
                >
                  <div className="mr-20">
                    <img className="w-44 h-44 mb-16" src={KeyImg} alt="" />
                    <div className="border px-5 py-3 rounded-2">
                      {t("security.ssh")}
                    </div>
                  </div>
                  <div className="flex-1 truncate">
                    <div className="text-16 flex justify-between">
                      <span className="">{item.name}</span>
                      <DeleteOutlined
                        className="font-bold cursor-pointer text-error"
                        onClick={() => onDelete(item)}
                      />
                    </div>
                    <div className="text-black opacity-60 mt-12 mb-8 text-wrap break-words">
                      {item.value}
                    </div>
                    <div className="text-black opacity-60">
                      <div>
                        {t("security.addedOn", {
                          date: item.addTime
                            ? moment(item.addTime).format(
                                t("security.dateFormat")
                              )
                            : "",
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-14 text-text">
              {t("security.noKeys")}
            </div>
          )}
          <div className="flex justify-center mt-24">
            <Button color="primary" variant="outlined" onClick={addSshKey}>
              {t("security.addNewKey")}
            </Button>
          </div>
        </div>
      )}
      <CommonPopup
        visible={state.visible}
        title={t(titleObj[state.formType])}
        onCancel={onCancel}
      >
        {state.formType == IFormTypeEnum.Key && (
          <Form
            form={formRef}
            className="border-b-0"
            onFinish={onAddSshFinish}
            labelCol={{ style: { width: 55 } }}
          >
            <Form.Item
              name="sshName"
              label={t("security.keyName")}
              rules={[requiredTrimValidate()]}
            >
              <Input
                placeholder={t("security.keyNamePlaceholder")}
                allowClear
                maxLength={32}
              />
            </Form.Item>
            <Form.Item
              name="sshKey"
              label={t("security.key")}
              trigger="onChange"
              rules={[publicKeyValidate()]}
            >
              <Input.TextArea
                rows={8}
                placeholder={t("security.keyPlaceholder")}
              />
            </Form.Item>
            <Form.Item>
              <Button type="primary" block htmlType="submit">
                {t("security.addSshKey")}
              </Button>
            </Form.Item>
          </Form>
        )}
        {state.formType == IFormTypeEnum.Password && (
          <Form
            form={passwordFormRef}
            className="border-b-0"
            onFinish={onPasswordFinish}
            labelCol={{ style: { width: 120 } }}
          >
            <Form.Item
              name="oldPassword"
              label={t("security.oldPassword")}
              rules={[requiredTrimValidate()]}
            >
              <Input.Password placeholder="" allowClear minLength={8} maxLength={32} />
            </Form.Item>
            <Form.Item
              name="newPassword"
              label={t("security.newPassword")}
              rules={passwordRules}
              extra={t("security.passwordExtra")}
            >
              <Input.Password placeholder="" allowClear minLength={8} maxLength={32} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" block htmlType="submit">
                {t("common.confirm")}
              </Button>
            </Form.Item>
          </Form>
        )}
        {state.formType == IFormTypeEnum.Username && (
          <Form
            form={usernameFormRef}
            onFinish={onUsernameFinish}
            initialValues={{
              username: state.username,
            }}
          >
            <Form.Item
              name="username"
              label={t("security.username")}
              rules={[requiredTrimValidate()]}
            >
              <Input placeholder="" allowClear maxLength={32} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" block htmlType="submit">
                {t("common.confirm")}
              </Button>
            </Form.Item>
          </Form>
        )}
        {state.formType == IFormTypeEnum.DelKey && (
          <div>
            <div className="text-3d text-16 mb-20">
              {t("security.removeConfirm")}
            </div>
            <Button type="primary" block danger onClick={onDeleteFinish}>
              {t("common.confirm")}
            </Button>
          </div>
        )}
      </CommonPopup>
    </div>
  );
};

export default Security;
