import { useState } from "react";
import { Popup } from "antd-mobile";
import {
  AppstoreOutlined,
  PlaySquareOutlined,
  ControlOutlined,
  MenuOutlined,
} from "@ant-design/icons";
import OverviewImg from "@/assets/images/svg/overview.svg";
import SecurityImg from "@/assets/images/svg/security.svg";
import NetworkImg from "@/assets/images/svg/network.svg";
import TerminalImg from "@/assets/images/svg/terminal.svg";
import SystemImg from "@/assets/images/svg/system.svg";
import PowerImg from "@/assets/images/svg/power.svg";
import ApplicationImg from "@/assets/images/svg/application.svg";
import DashboardImg from "@/assets/images/svg/dashboard.svg";
import FilesImg from "@/assets/images/svg/files.svg";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import useConfigStore from "@/store/config";
import { getMenuSections } from "@/layout/menu";
import LangToggle from "@/components/lang-toggle";

const iconMap: Record<string, React.ReactNode> = {
  applications: <AppstoreOutlined style={{ fontSize: 16 }} />,
  live: <PlaySquareOutlined style={{ fontSize: 16 }} />,
  device: <ControlOutlined style={{ fontSize: 16 }} />,
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

function Sidebar() {
  const location = useLocation();
  const currentRoute = location.pathname;
  const [visible, setVisible] = useState(false);
  const navigate = useNavigate();
  const { deviceInfo, galleryMode } = useConfigStore();
  const { t } = useTranslation();

  const menuSections = getMenuSections({
    galleryMode,
    isReCamera: Boolean(deviceInfo.isReCamera),
    // queryDeviceInfo always sets appName; before it lands the mode is unknown
    modeKnown: Boolean(deviceInfo.appName),
  });

  const isActive = (route: string) =>
    currentRoute === route ||
    (route === "/" && currentRoute === "/applications");

  return (
    <div className="inline-block">
      <div
        className="flex p-16 cursor-pointer"
        onClick={() => {
          setVisible(true);
        }}
      >
        <MenuOutlined />
      </div>

      <Popup
        visible={visible}
        onMaskClick={() => {
          setVisible(false);
        }}
        position="left"
        bodyStyle={{ width: "280px", background: "var(--rc-surface)" }}
      >
        <div className="pt-60 px-12 pb-24 h-full overflow-y-auto">
          <div className="font-display font-bold text-20 px-10 pb-16 truncate">
            {deviceInfo.deviceName}
          </div>
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
                    setVisible(false);
                  }}
                >
                  {iconMap[item.key]}
                  <span>{t(item.label)}</span>
                </div>
              ))}
            </div>
          ))}
          <div className="px-10 pt-16 mt-16 border-t border-line">
            <LangToggle />
          </div>
        </div>
      </Popup>
    </div>
  );
}

export default Sidebar;
