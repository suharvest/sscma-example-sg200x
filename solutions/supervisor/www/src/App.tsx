import { useEffect, useMemo } from "react";
import { ConfigProvider, App as AntdApp } from "antd";
import { createHashRouter, RouterProvider } from "react-router-dom";
import Routes from "@/router";
import Login from "@/views/login";
import { queryDeviceInfoApi } from "@/api/device/index";
import { getUserInfoApi } from "@/api/user";
import useUserStore from "@/store/user";
import useConfigStore from "@/store/config";
import useCapabilitiesStore from "@/store/capabilities";
import { Version } from "@/utils";

const router = createHashRouter(Routes);

const App = () => {
  const {
    currentSn,
    usersBySn,
    setCurrentSn,
    updateFirstLogin,
    clearCurrentUserInfo,
  } = useUserStore();

  const { updateDeviceInfo } = useConfigStore();
  const { fetchCapabilities } = useCapabilitiesStore();

  useEffect(() => {
    console.log(`%cVersion: ${Version}`, "font-weight: bold");
    initUserData();
    // P5: pull the hardware capability set once (no-auth endpoint; failure
    // keeps the store undefined and the UI on its legacy fallbacks).
    fetchCapabilities();
  }, []);

  const token = useMemo(() => {
    return currentSn ? usersBySn[currentSn]?.token : null;
  }, [usersBySn, currentSn]);

  const initUserData = async () => {
    try {
      const response = await queryDeviceInfoApi();
      const deviceInfo = response.data;
      // 查询设备信息，获取sn
      updateDeviceInfo(deviceInfo);
      const sn = deviceInfo.sn;
      setCurrentSn(sn);
      if (sn) {
        // 查询设备是否第一次登录，第一次登录，直接进登录页
        const response = await getUserInfoApi();
        if (response.code == 0) {
          const data = response.data;
          const firstLogin = data.firstLogin;
          updateFirstLogin(firstLogin);
          if (firstLogin) {
            clearCurrentUserInfo();
          }
        }
      }
    } catch (error) {
      // 不能清用户信息，很可能是服务没起来超时
    }
  };

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#8fc31f",
          colorPrimaryHover: "#81ae1b",
          colorText: "#0a0a0a",
          colorTextSecondary: "#666666",
          colorBorder: "#e6e6e6",
          colorBorderSecondary: "#e6e6e6",
          borderRadius: 8,
          fontFamily:
            'Montserrat, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
        },
      }}
    >
      <AntdApp className="w-full h-full">
        <div className="w-full h-full">
          {token ? <RouterProvider router={router} /> : <Login />}
        </div>
      </AntdApp>
    </ConfigProvider>
  );
};

export default App;
