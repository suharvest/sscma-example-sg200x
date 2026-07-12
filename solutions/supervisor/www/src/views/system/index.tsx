import { useEffect, useMemo, useState } from "react";
import EditBlackImg from "@/assets/images/svg/editBlack.svg";
import ArrowImg from "@/assets/images/svg/downArrow.svg";
import CommonPopup from "@/components/common-popup";
import { Form, Input, Picker, ProgressBar, Mask } from "antd-mobile";
import {
  PickerValue,
  PickerValueExtend,
} from "antd-mobile/es/components/picker";
import { Button } from "antd";
import moment from "moment";
import { useTranslation } from "react-i18next";
import { useData } from "./hook";
import { DeviceChannleMode, UpdateStatus, PowerSourceMode } from "@/enum";
import { requiredTrimValidate } from "@/utils/validate";
import { parseUrlParam } from "@/utils";
import useConfigStore from "@/store/config";
import useRunModeSwitch, { RunMode } from "@/hooks/useRunModeSwitch";

// P4-D runtime mode selector entries (labels/descriptions are i18n keys)
const runModeList: { mode: RunMode; label: string; desc: string }[] = [
  {
    mode: "console",
    label: "runtimeMode.console",
    desc: "runtimeMode.consoleDesc",
  },
  {
    mode: "nodered",
    label: "runtimeMode.nodered",
    desc: "runtimeMode.noderedDesc",
  },
];

const channelList = [
  { label: "system.channelSelfHost", value: DeviceChannleMode.Self },
  { label: "system.channelOfficial", value: DeviceChannleMode.Official },
];
const infoList = [
  { label: "system.infoCpu", key: "cpu" },
  { label: "system.infoRam", key: "ram" },
  { label: "system.infoNpu", key: "npu" },
  { label: "system.infoOs", key: "osVersion" },
  { label: "system.infoDevice", key: "type" },
  { label: "system.infoBattery", key: "powerSource", isPowerSource: true },
];

function System() {
  const { t } = useTranslation();
  const {
    deviceInfo,
    batteryInfo,
    addressFormRef,
    onEditServerAddress,
    onCancel,
    onFinish,
    onUpdateCancel,
    onUpdateApply,
    onConfirm,
    onChannelChange,
    onUpdateRestart,
    onUpdateCheck,
    onPowerSourceChange,
  } = useData();

  const { systemUpdateState, setSystemUpdateState, galleryMode } =
    useConfigStore();

  // Runtime mode (P4-D). galleryMode mirrors the live process state; the
  // mode is only trustworthy once queryDeviceInfo has populated the store.
  const modeKnown = Boolean(deviceInfo?.appName);
  const currentMode: RunMode = galleryMode ? "console" : "nodered";
  const { switching, requestSwitch, forcing, requestForceConsole } =
    useRunModeSwitch();

  const [isDashboard, setIsDashboard] = useState(false);
  useEffect(() => {
    const param = parseUrlParam(window.location.href);
    const dashboard = param.dashboard || param.disablelayout;
    setIsDashboard(dashboard == 1);
  }, []);

  const channelLable = useMemo(() => {
    const index = channelList.findIndex(
      (item) => item.value === systemUpdateState.channel
    );
    return index > -1 && t(channelList[index].label);
  }, [systemUpdateState.channel, t]);

  const displayInfoList = useMemo(() => {
    // Only show Battery row if battery hardware is detected as available
    return infoList.filter(
      (item) => !(item as any).isPowerSource || systemUpdateState.batteryAvailable === true
    );
  }, [systemUpdateState.batteryAvailable]);

  return (
    <div className="my-8 p-16">
      {!isDashboard && (
        <>
          <div className="font-bold text-18 mb-14">
            {t("runtimeMode.title")}
          </div>
          <div className="bg-white rounded-20 px-24 mb-24">
            {runModeList.map((item, index) => {
              const isCurrent = modeKnown && currentMode === item.mode;
              return (
                <div
                  key={item.mode}
                  className={`flex justify-between items-center py-20 ${
                    index ? "border-t" : ""
                  }`}
                >
                  <div className="flex-1 mr-16">
                    <div className="flex items-center flex-wrap">
                      <span className="font-bold">{t(item.label)}</span>
                      {isCurrent && (
                        <span
                          className="ml-8 text-12 px-8 rounded-full"
                          style={{
                            background: "#f0f7e0",
                            color: "#6a9316",
                            lineHeight: "20px",
                          }}
                        >
                          {t("runtimeMode.current")}
                        </span>
                      )}
                    </div>
                    <div className="text-12 opacity-60 mt-4">
                      {t(item.desc)}
                    </div>
                  </div>
                  {!isCurrent && (
                    <Button
                      type="primary"
                      loading={switching === item.mode}
                      disabled={
                        !modeKnown ||
                        (switching !== null && switching !== item.mode)
                      }
                      onClick={() => requestSwitch(item.mode)}
                    >
                      {switching === item.mode
                        ? t("runtimeMode.switchingBtn")
                        : t("runtimeMode.switchTo")}
                    </Button>
                  )}
                </div>
              );
            })}
            {/* #20 hard escape hatch: unobtrusive danger link to force console
                mode when Node-RED has wedged the device. */}
            <div className="flex justify-between items-center py-16 border-t">
              <div className="flex-1 mr-16">
                <div className="text-12 opacity-60">
                  {t("runtimeMode.forceConsoleDesc")}
                </div>
              </div>
              <Button
                type="link"
                danger
                size="small"
                loading={forcing}
                disabled={!modeKnown || switching !== null}
                onClick={requestForceConsole}
              >
                {t("runtimeMode.forceConsole")}
              </Button>
            </div>
          </div>
        </>
      )}

      <div className="font-bold text-18 mb-14">{t("system.update")}</div>
      <div className="bg-white rounded-20 px-24">
        <div className="flex justify-between pt-24">
          <span className="opacity-60 self-center mr-20">
            {t("system.softwareUpdate")}
          </span>
          <div className="flex-1 text-right justify-end flex">
            {systemUpdateState.status == UpdateStatus.NoNeedUpdate && (
              <span className="self-center ml-12">{t("system.upToDate")}</span>
            )}
            {systemUpdateState.status == UpdateStatus.Check && (
              <Button type="primary" onClick={() => onUpdateCheck(true)}>
                {t("system.checkUpdate")}
              </Button>
            )}
            {systemUpdateState.status == UpdateStatus.NeedUpdate && (
              <Button type="primary" onClick={onUpdateApply}>
                {t("system.updateBtn")}
              </Button>
            )}
            {systemUpdateState.status == UpdateStatus.Updating && (
              <Button onClick={onUpdateCancel}>{t("common.cancel")}</Button>
            )}
            {systemUpdateState.status == UpdateStatus.UpdateDone && (
              <Button type="primary" onClick={onUpdateRestart}>
                {t("system.reboot")}
              </Button>
            )}
          </div>
        </div>
        <div className="flex justify-between py-12 text-3d ">
          {systemUpdateState.status == UpdateStatus.NoNeedUpdate && (
            <span className="text-12">
              {t("system.upToDateHint")}
            </span>
          )}
          {systemUpdateState.status == UpdateStatus.UpdateDone && (
            <span className="text-12">
              {t("system.rebootHint")}
            </span>
          )}
          {systemUpdateState.status == UpdateStatus.Updating && (
            <div className="w-full mb-8">
              <div className="flex justify-between mb-4">
                <span>{systemUpdateState.percent}%</span>
                <span>{moment().fromNow()}</span>
              </div>
              <div>
                <ProgressBar
                  className="w-full"
                  rounded={false}
                  percent={systemUpdateState.percent}
                />
              </div>
              <div className="mt-8">
                {t("system.updatingHint")}
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="font-bold text-18 mb-14 my-24">
        {" "}
        {t("system.betaParticipation")}
      </div>

      <div className="bg-white rounded-20 px-24">
        <div className="flex justify-between py-24">
          <span className="opacity-60 mr-20">{t("system.channel")}</span>
          <div
            className="flex-1 text-right justify-end flex"
            onClick={onChannelChange}
          >
            <span>{channelLable}</span>
            <span className="self-center ml-12">
              <img
                className={`w-24 h-24 ml-6 self-center ${
                  systemUpdateState.channelVisible && "rotate-180 "
                }`}
                src={ArrowImg}
                alt=""
              />
            </span>
          </div>
        </div>
        {systemUpdateState.channel == DeviceChannleMode.Self && (
          <div className="flex justify-between py-24 w-full border-t">
            <span className="opacity-60 mr-20">
              {t("system.serverAddress")}
            </span>
            <div
              className="flex-1 text-right justify-end flex truncate"
              onClick={onEditServerAddress}
            >
              <span className="truncate ">{systemUpdateState.address}</span>
              <img
                className="w-24 h-24 ml-6 self-center"
                src={EditBlackImg}
                alt=""
              />
            </div>
          </div>
        )}
      </div>
      <Picker
        columns={[
          channelList.map((item) => ({
            label: t(item.label),
            value: item.value,
          })),
        ]}
        visible={systemUpdateState.channelVisible}
        onClose={() => {
          setSystemUpdateState({
            channelVisible: false,
          });
        }}
        value={[systemUpdateState.channel] as PickerValue[]}
        onConfirm={
          onConfirm as (value: PickerValue[], extend: PickerValueExtend) => void
        }
      />

      {!isDashboard && (
        <div>
          <div className="font-bold text-18 mb-14 my-24">
            {" "}
            {t("system.systemInfo")}
          </div>
          <div className="bg-white rounded-20 px-24">
            {displayInfoList.map((item, index) => {
              const isPowerSource = (item as any).isPowerSource;
              return (
                <div key={item.key} className={isPowerSource && systemUpdateState.powerSourceMode === PowerSourceMode.Battery ? 'pb-20' : ''}>
                  <div
                    className={`flex justify-between items-center py-24 ${
                      index && "border-t"
                    }`}
                  >
                    <span className="opacity-60 text-black mr-20">
                      {t(item.label)}
                    </span>
                    <div className="flex-1 truncate text-right flex items-center justify-end">
                      {isPowerSource ? (
                        <div
                          className={`cursor-pointer select-none w-28 h-15 rounded-full relative transition-colors shadow-sm ${
                            systemUpdateState.powerSourceMode === PowerSourceMode.Battery
                              ? 'bg-green-500'
                              : 'bg-gray-300'
                          }`}
                          onClick={() => onPowerSourceChange(PowerSourceMode.Battery)}
                        >
                          <div
                            className={`w-13 h-13 bg-white rounded-full absolute top-1 shadow-md transition-all duration-300 ${
                              systemUpdateState.powerSourceMode === PowerSourceMode.Battery
                                ? 'right-[1px]'
                                : 'left-[1px]'
                            }`}
                          />
                        </div>
                      ) : item.key == "osVersion" ? (
                        `${deviceInfo.osName} ${deviceInfo[item.key]}`
                      ) : (
                        deviceInfo[item.key]
                      )}
                    </div>
                  </div>
                  {isPowerSource && systemUpdateState.powerSourceMode === PowerSourceMode.Battery && (
                    <div className="flex items-center text-14 text-gray-600 relative h-10">
                      {/* L-shaped arrow from Battery label */}
                      <svg className="absolute left-8 -top-6 w-16 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 60 40">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 5 L5 35 L40 35" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M35 30 L40 35 L35 40" />
                      </svg>
                      <span className="ml-20 px-2" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        🔋{batteryInfo?.voltage ? (batteryInfo.voltage / 1000).toFixed(1) : '0.0'} V
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <CommonPopup
        visible={systemUpdateState.visible}
        title={t("system.serverAddress")}
        onCancel={onCancel}
      >
        <Form
          ref={addressFormRef}
          className="border-b-0"
          requiredMarkStyle="none"
          onFinish={onFinish}
          initialValues={{
            serverUrl: systemUpdateState.address,
          }}
          footer={
            <Button block htmlType="submit" type="primary">
              {t("common.confirm")}
            </Button>
          }
        >
          <Form.Item name="serverUrl" label="" rules={[requiredTrimValidate()]}>
            <Input className="border rounded-6 p-10" placeholder="" clearable />
          </Form.Item>
        </Form>
      </CommonPopup>
      <Mask
        visible={systemUpdateState.updateInfoVisible}
        onMaskClick={() =>
          setSystemUpdateState({
            updateInfoVisible: false,
          })
        }
      >
        <div className="px-30 pt-100 pb-100 h-full" style={{ height: "100vh" }}>
          <div className="text-3d bg-white  rounded-16 p-20 h-full flex-1  flex flex-col justify-between">
            <div className="font-bold text-16">{t("system.osUpdateTitle")}</div>
            <div className="flex justify-between opacity-60 font-bold mt-6 mb-10">
              <span>Version 15.4</span>
              <span>24/06/2024</span>
            </div>
            <div className="flex-1 overflow-y-auto">
              <div className="text-12">
                channel, and included all the changes have been tested in
                preview
              </div>
            </div>
            <div className="flex mt-20">
              <Button className="flex-1 mr-28" onClick={onUpdateCancel}>
                {t("common.cancel")}
              </Button>
              <Button type="primary" className="flex-1" onClick={onUpdateApply}>
                {t("system.apply")}
              </Button>
            </div>
          </div>
        </div>
      </Mask>
    </div>
  );
}

export default System;
