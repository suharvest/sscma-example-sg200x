<!-- app: facemesh-reader | version: 0.1.1 | doc-format: recamera-integration/v1 -->

# 困倦检测 — 集成指南

## 概述

**典型场景**：长途驾驶舱、值守岗位、工程机械操作位——疲劳征兆出现在事故之前，这个应用负责把那几分钟找出来。

完全在设备端运行（reCamera SG2002，RISC-V + TPU）。yolov8n-face 检测人脸，468 点 FaceMesh 取出眼睛与嘴部轮廓，据此每帧计算：

| 指标 | 含义 |
|---|---|
| `EAR`（Eye Aspect Ratio） | 眼睛张开程度。持续低于阈值即为闭眼 |
| `MAR`（Mouth Aspect Ratio） | 嘴巴张开程度，用于识别哈欠 |
| `PERCLOS` | 滚动窗口内闭眼时间占比。疲劳研究中最常用的客观指标 |

三条判据（持续闭眼 / PERCLOS 超标 / 打哈欠）任一触发即进入告警，状态取值 `Alert` → `Drowsy` → `Danger`。

输出通道：

| 通道 | 传输 | 用途 |
|---|---|---|
| RTSP | `rtsp://<设备IP>:8554/live0` | H.264 视频，接 NVR / VMS / VLC |
| MQTT | `<设备IP>:1883`，主题 `recamera/facemesh-reader/results` | 逐帧指标与告警状态（JSON） |
| 调试 WebSocket | `ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results` | 浏览器实时调试（console UI） |

## RTSP 输出

- **URL**：`rtsp://<设备IP>:8554/live0`
- **编码**：H.264，1280x720 @ 15 fps（默认值）
- **隐私**：开启设备级打码后，检测到的人脸会在编码之前被遮挡，因此 RTSP 流、console 预览、`/snapshot.jpg` 拿到的都已经是打码后的画面。默认关闭，开关与透明度在 console 的**设备**页（调试页有快捷开关），改动即时生效、不重启应用。
- 建议同一时间只有一个 RTSP 消费者；摄像头链路由本应用独占。

## MQTT 输出

### 连接

| 项 | 值 |
|---|---|
| Broker | `<设备IP>:1883` |
| 主题 | `recamera/facemesh-reader/results`（`--mqtt-topic` 可改） |
| QoS | 0 |
| 频率 | 每个推理帧一条 |

### 消息格式

```json
{
  "timestamp": 1737964800000,
  "frame_id": 1024,
  "inference_time_ms": 97.0,
  "face_count": 1,
  "faces": [
    {
      "id": 0,
      "bbox": { "x": 0.5341, "y": 0.1938, "w": 0.2184, "h": 0.3782 },
      "confidence": 0.8290,
      "left_ear": 0.1402,
      "right_ear": 0.1385,
      "ear": 0.1394,
      "mar": 0.3120,
      "eyes_closed": true,
      "mouth_open": false,
      "metrics_valid": true,
      "drowsiness": {
        "level": 0.7350,
        "state": "Danger",
        "perclos_pct": 42.8571,
        "continuous_closure_sec": 1.9000,
        "alert_active": true,
        "drowsy_by_ear": true,
        "drowsy_by_perclos": true,
        "drowsy_by_yawn": false
      },
      "yawn": { "is_yawning": false, "yawn_count_5min": 2 }
    }
  ]
}
```

### 字段说明

| 字段 | 说明 |
|---|---|
| `drowsiness.state` | `Alert`（清醒）/ `Drowsy`（困倦）/ `Danger`（危险） |
| `drowsiness.level` | 0–1，由 EAR 与 PERCLOS 各占一半合成 |
| `drowsiness.alert_active` | 告警是否处于激活状态。**接入告警系统时用这个**，不要自己对 `level` 设阈值 |
| `drowsiness.drowsy_by_*` | 三条判据分别是否成立，便于定位是哪一条触发的 |
| `metrics_valid` | 该帧的关键点是否可用。为 `false` 时上面的 EAR/MAR 不可信 |
| `yawn.yawn_count_5min` | 滚动 5 分钟内的哈欠次数 |

### 坐标约定

`bbox` 是**归一化的中心坐标**（`x`/`y` 为中心点，不是左上角），范围 0–1，相对推理帧。

> 注意推理通道（4:3）与出流（16:9）形状不同，摄像头按比例装载画面。直接把这里的坐标画到 1280x720 的视频上会偏——需要先做 letterbox 换算，`components/geometry` 里的 `toStream()` 就是干这个的。

### 468 点关键点

默认**不发**（每帧近千个浮点数，会把 broker 打满）。用 `--include-landmarks` 启用后，每张脸多一个 `landmarks` 数组，元素为 `[x, y]` 归一化坐标。

## 调试 WebSocket（浏览器实时预览）

### `ws://<设备IP>:8001/` — H.264 视频

二进制帧，Annex-B。console 的调试页用它做实时预览。

### `ws://<设备IP>:8001/results` — 推理结果

文本消息，与 MQTT 同源但坐标已换算到出流像素，可直接叠加在视频上。

### `http://<设备IP>:8001/snapshot.jpg`

当前帧的 JPEG。**打码开启时这里也是打码后的画面**——ONVIF 的 `GetSnapshotUri` 指向的就是它。

## 可调参数

阈值都可在启动参数里调，写进 `/etc/facemesh-reader.conf` 后随服务生效：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--ear-threshold` | 0.21 | 低于此值判定为闭眼 |
| `--ear-continuous-sec` | 1.5 | 连续闭眼多久算困倦 |
| `--mar-threshold` | 0.6 | 高于此值判定为张嘴/哈欠 |
| `--perclos-warning` | 15 | PERCLOS 警告线（%） |
| `--perclos-critical` | 30 | PERCLOS 危险线（%） |
| `--threshold` | 0.5 | 人脸检测置信度 |

> 这些阈值是通用起点，不是标定值。不同摄像头角度、佩戴眼镜与否都会影响 EAR 的绝对值，**部署前应在实际机位上对着 `ear` 字段观察一段时间再定**。
