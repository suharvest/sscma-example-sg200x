import React, { useState } from "react";
import { Form, Input, Modal } from "antd";
import {
  AppstoreOutlined,
  PlaySquareOutlined,
  ControlOutlined,
  ApiOutlined,
} from "@ant-design/icons";
import useConfigStore from "@/store/config";
import EditImg from "@/assets/images/svg/edit.svg";
import OverviewImg from "@/assets/images/svg/overview.svg";
import SecurityImg from "@/assets/images/svg/security.svg";
import NetworkImg from "@/assets/images/svg/network.svg";
import TerminalImg from "@/assets/images/svg/terminal.svg";
import SystemImg from "@/assets/images/svg/system.svg";
import PowerImg from "@/assets/images/svg/power.svg";
import FilesImg from "@/assets/images/svg/files.svg";
import ApplicationImg from "@/assets/images/svg/application.svg";
import DashboardImg from "@/assets/images/svg/dashboard.svg";
import { updateDeviceInfoApi, queryDeviceInfoApi } from "@/api/device/index";
import { hostnameValidate } from "@/utils/validate";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { getMenuSections } from "@/layout/menu";
import LangToggle from "@/components/lang-toggle";

interface Props {
  children: React.ReactNode;
}

const iconMap: Record<string, React.ReactNode> = {
  applications: <AppstoreOutlined style={{ fontSize: 16 }} />,
  live: <PlaySquareOutlined style={{ fontSize: 16 }} />,
  device: <ControlOutlined style={{ fontSize: 16 }} />,
  integrations: <ApiOutlined style={{ fontSize: 16 }} />,
  overview: <img className="w-16 h-16" src={OverviewImg} alt="" />,
  dashboard: <img className="w-16 h-16" src={DashboardImg} alt="" />,
  workspace: <img className="w-16 h-16" src={ApplicationImg} alt="" />,
  files: <img className="w-16 h-16" src={FilesImg} alt="" />,
  security: <img className="w-16 h-16" src={SecurityImg} alt="" />,
  network: <img className="w-16 h-16" src={NetworkImg} alt="" />,
  terminal: <img className="w-16 h-16" src={TerminalImg} alt="" />,
  system: <img className="w-16 h-16" src={SystemImg} alt="" />,
  power: <img className="w-16 h-16" src={PowerImg} alt="" />,
};

const PCLayout: React.FC<Props> = ({ children }) => {
  const { deviceInfo, galleryMode, updateDeviceInfo } = useConfigStore();
  const { t } = useTranslation();
  const [isEditNameModalOpen, setIsEditNameModalOpen] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const location = useLocation();
  const [form] = Form.useForm();
  const currentRoute = location.pathname;
  const navigate = useNavigate();

  const menuSections = getMenuSections({
    galleryMode,
    // queryDeviceInfo always sets appName; before it lands the mode is unknown
    modeKnown: Boolean(deviceInfo.appName),
  });

  const isActive = (route: string) =>
    currentRoute === route ||
    (route === "/" && currentRoute === "/applications");

  const onQueryDeviceInfo = async () => {
    const res = await queryDeviceInfoApi();
    updateDeviceInfo(res.data);
  };

  const handleEditNameOk = async () => {
    try {
      const values = await form.validateFields();
      const deviceName = (values.deviceName || "").trim();
      setConfirmLoading(true);
      await updateDeviceInfoApi({ deviceName });
      setIsEditNameModalOpen(false);
      form.resetFields();
      await onQueryDeviceInfo();
    } catch (error) {
      console.log(error);
    } finally {
      setConfirmLoading(false);
    }
  };

  const handleEditNameCancel = () => {
    setIsEditNameModalOpen(false);
    form.resetFields();
  };

  return (
    <>
      <div className="bg-white text-center py-14 border-b border-line sticky top-0 z-30 backdrop-blur">
        <div className="font-display font-semibold text-16 relative flex justify-center px-40 pl-50 tracking-tight">
          <div className="absolute left-0 -mt-4 "></div>
          <div className="truncate">{deviceInfo?.deviceName}</div>
          <img
            className="w-20 h-20 ml-4 self-center cursor-pointer"
            onClick={() => {
              setIsEditNameModalOpen(true);
            }}
            src={EditImg}
            alt=""
          />
          <div
            className="absolute right-16"
            style={{ top: "50%", transform: "translateY(-50%)" }}
          >
            <LangToggle />
          </div>
        </div>
        <div className="mt-2 rc-mono text-11 text-muted">{deviceInfo?.ip}</div>
      </div>

      <div className="flex flex-1">
        <div
          className="h-full w-[228px] flex-none border-r border-line px-14 py-18"
          style={{ background: "var(--rc-surface)" }}
        >
          {menuSections.map((section) => (
            <div key={section.title}>
              <div className="rc-section-label px-10 pt-14 pb-6">
                {t(section.title)}
              </div>
              {section.items.map((item) => (
                <div
                  key={item.route}
                  className={`rc-side-link ${
                    isActive(item.route) ? "active" : ""
                  }`}
                  onClick={() => {
                    navigate(item.route);
                  }}
                >
                  {iconMap[item.key]}
                  <span>{t(item.label)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="flex-1 min-w-0 px-32 bg-white">
          <div className="w-full max-w-[1520px] mx-auto">{children}</div>
        </div>
      </div>

      <Modal
        title={t("layout.editDeviceName")}
        open={isEditNameModalOpen}
        confirmLoading={confirmLoading}
        onOk={handleEditNameOk}
        onCancel={handleEditNameCancel}
      >
        <Form
          form={form}
          name="dependencies"
          autoComplete="off"
          style={{ maxWidth: 600 }}
          layout="vertical"
        >
          <Form.Item
            name="deviceName"
            label={t("layout.name")}
            rules={[hostnameValidate(32)]}
          >
            <Input placeholder="recamera-132456" maxLength={32} allowClear />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export default PCLayout;
