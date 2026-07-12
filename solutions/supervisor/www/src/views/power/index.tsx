import { message, Button } from "antd";
import { useTranslation } from "react-i18next";
import { setDevicePowerApi } from "@/api/device/index";
import { PowerMode } from "@/enum";

function Power() {
  const { t } = useTranslation();
  const onOperateDevice = async (mode: PowerMode) => {
    await setDevicePowerApi({ mode });
    message.success(t("power.operateSuccess"));
  };

  return (
    <div className="flex h-full justify-center p-32 flex-col text-18">
      <Button
        className="mb-12"
        color="danger"
        variant="outlined"
        onClick={() => onOperateDevice(PowerMode.Restart)}
      >
        {t("power.reboot")}
      </Button>
      <Button
        color="danger"
        variant="outlined"
        onClick={() => onOperateDevice(PowerMode.Shutdown)}
      >
        {t("power.shutdown")}
      </Button>
    </div>
  );
}

export default Power;
