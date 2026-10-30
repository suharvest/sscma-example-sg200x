<!-- app: region-motion | version: 0.1.0 | doc-format: recamera-integration/v1 -->

# 选区运动检测 — 集成指南

## 概述

在 CPU 上实时监测画面指定多边形区域内的运动变化，无需 NPU 模型。采用自适应背景建模与轮廓分析算法，在控制台画面上实时绘制选区与运动包围盒，原生支持 Home Assistant 自动发现与告警联动。

输出通道：

| 通道 | 协议/地址 | 用途 |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 实时视频流 |
| MQTT | `<device-ip>:1883`，指定主题 | 运动状态与面积占比 (JSON) |
| 调试 WebSocket | `ws://<device-ip>:8001/` 与 `ws://<device-ip>:8001/results` | 网页端低延迟画面与标注图层 |

## RTSP 输出

- **URL**: `rtsp://<device-ip>:8554/live0`
- **编码**: H.264，1280x720 @ 15 fps

## MQTT 与 Home Assistant 集成

### 连接配置

- **Broker**: 设备端 mosquitto 1883 端口（或通过 `/userdata/local/ha.conf` 配置的外部 broker）
- **Topic**: `recamera/region-motion/results`

### Home Assistant 自动发现

启动后自动注册 `BinarySensor` 实体：
- **实体 ID**: `motion`
- **设备类型**: `motion`
- **状态主题**: `recamera/region-motion/results`
- **状态取值**: `{{ value_json.state }}`

### 消息格式

```json
{
  "state": "ON",
  "motion": true,
  "ratio": 0.038,
  "frame_id": 1024
}
```

## 配置项说明

配置文件路径为 `/userdata/local/apps/region-motion.config.json`，可直接在控制台应用页面修改：

- `pixel_threshold`: 像素变化灵敏度（5..100，默认 25）
- `min_motion_area`: 触发告警的最小运动面积占比（0.001..0.5，默认 0.015）
- `debounce_frames`: 防抖确认帧数（1..10，默认 3）
- `roi_zone`: 划定的多边形选区顶点坐标数组 `[[x, y], ...]`
