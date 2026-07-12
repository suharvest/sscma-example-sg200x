<!-- app: retail-vision | version: 1.1.0 | doc-format: recamera-integration/v1 -->

# 零售人流分析 — 集成指南

## 概述

完全在设备端运行的零售人流分析（reCamera SG2002，RISC-V + TPU）。YOLO11n 检测人员，跟踪器跨帧追踪，并为每个人划分驻留状态：

| 状态 | 含义 |
|---|---|
| `transient` | 路过，未停留 |
| `dwelling` | 刚停下（未达 engaged 阈值，默认 < 1.5 秒） |
| `engaged` | 停留 1.5–20 秒（正在看货架/陈列） |
| `assistance` | 停留 > 20 秒（可能需要店员协助） |

整个画面即为分析区域。滚动窗口（默认 60 秒）的区域指标与逐人数据随每个推理帧发布一次。

输出通道：

| 通道 | 传输 | 用途 |
|---|---|---|
| RTSP | `rtsp://<设备IP>:8554/live0` | H.264 视频，接 NVR / VMS / VLC |
| MQTT | `<设备IP>:1883`，主题见下文 | 区域指标 + 逐人分析（JSON） |
| 调试 WebSocket | `ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results` | 浏览器实时调试（控制台 UI） |

## RTSP 输出

- **URL**：`rtsp://<设备IP>:8554/live0`
- **编码**：H.264，1280x720 @ 15 fps（默认值，启动参数可调）
- 可选 RTSP 鉴权：启动参数 `--rtsp-user` / `--rtsp-pass`。

## MQTT 输出

### 连接

- **Broker**：设备上的 mosquitto，端口 **1883**（明文 TCP，无 TLS）。默认只监听 `localhost`；如需从其他主机订阅，请在 `/etc/mosquitto/mosquitto.conf` 中开启外部监听（`listener 1883 0.0.0.0`、`allow_anonymous true`）或桥接到自有 broker。
- **主题**：`recamera/retail-vision/vision`（manifest 默认值；可在 `/etc/retail-vision.conf` 中用 `MQTT_TOPIC` 覆盖）。**以控制台 Live 页显示为准。**
- **QoS**：0，**retain**：false，**客户端 ID**：`recamera-retail-vision`
- 每个推理帧发布一条消息（640x640，最高约 10–15 fps）。

### 消息格式

顶层字段：

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `timestamp` | integer | Unix 毫秒 | 帧采集/发布时间 |
| `frame_id` | integer | — | 单调递增帧计数（应用启动时归零） |
| `frame_width` / `frame_height` | integer | 像素 | `bbox` 坐标参照的显示（RTSP）分辨率（默认 1280x720） |
| `fps` | number | 帧/秒 | 当前处理帧率 |
| `inference_time_ms` | number | 毫秒 | 检测推理耗时 |
| `zone` | object | — | 滚动窗口区域指标（见下） |
| `persons` | array | — | 当前每个被跟踪人员一个对象 |

`zone` 对象（滚动窗口，默认 60 秒）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `occupancy_count` | integer | 平滑后的当前在场人数 |
| `browsing_count` | integer | `transient`/`dwelling` 状态人数 |
| `engaged_count` | integer | `engaged` 状态人数 |
| `assist_count` | integer | `assistance` 状态人数 |
| `peak_customer` | integer | 窗口内峰值人数 |
| `avg_dwell_time` | number（秒） | 窗口内结束轨迹的平均驻留总时长 |
| `avg_engagement_time` | number（秒） | 窗口内结束轨迹的平均 engaged+ 时长 |
| `avg_velocity` | number（m/s） | 窗口内结束轨迹的平均移动速度 |
| `entry_count` / `exit_count` | integer | 自应用启动起累计的进/出人次（重启归零） |

`persons[]` 每个元素：

| 字段 | 类型 | 单位/范围 | 说明 |
|---|---|---|---|
| `track_id` | integer | — | 稳定的人员轨迹 ID |
| `confidence` | number | 0–1 | 检测置信度 |
| `bbox.x` / `bbox.y` | number | 归一化 0–1 | 框**左上角**坐标，参照 `frame_width`x`frame_height`（letterbox 已校正） |
| `bbox.w` / `bbox.h` | number | 归一化 0–1 | 框宽高，同一参照系 |
| `velocity.vx` / `velocity.vy` | number | 归一化/秒 | 速度分量 |
| `velocity.speed_m_s` | number | m/s | 估算的真实移动速度（按 1.7 m 身高假设） |
| `state` | string | 见概述 | 驻留状态 |
| `dwell_duration` | number | 秒 | 当前驻留片段的静止时长 |

示例：

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "frame_width": 1280,
  "frame_height": 720,
  "fps": 9.8,
  "inference_time_ms": 71.0,
  "zone": {
    "occupancy_count": 2, "browsing_count": 1, "engaged_count": 1,
    "assist_count": 0, "peak_customer": 3, "avg_dwell_time": 4.2,
    "avg_engagement_time": 2.1, "avg_velocity": 0.42,
    "entry_count": 15, "exit_count": 13
  },
  "persons": [
    {
      "track_id": 12,
      "confidence": 0.87,
      "bbox": { "x": 0.4413, "y": 0.375, "w": 0.142, "h": 0.4522 },
      "velocity": { "vx": 0.01, "vy": 0.0, "speed_m_s": 0.05 },
      "state": "engaged",
      "dwell_duration": 4.7
    }
  ]
}
```

### 坐标约定（MQTT）

`bbox` 为**左上角基准、归一化到 [0, 1]**，参照*显示*分辨率（`frame_width` x `frame_height`，默认 1280x720）——16:9 视频流与方形模型输入之间的 letterbox 畸变已被校正，可直接叠画在 RTSP 视频上：

```text
px_left = bbox.x * frame_width
px_top  = bbox.y * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

注意：这与下述调试 WebSocket 通道不同（后者是中心点基准、推理分辨率像素）。

## 调试 WebSocket（浏览器实时预览）

供 reCamera 控制台 Live 页使用；任何 WebSocket 客户端也可接入。惰性推送：无客户端连接时零拷贝开销。每条路径最多 2 个客户端。`--no-debug` 关闭，`--debug-port` 改端口（默认 8001）。

### `ws://<设备IP>:8001/` — H.264 视频

二进制消息：每条为一个 H.264 Annex-B 访问单元，尾部附加 **8 字节小端 `uint64`** Unix 毫秒时间戳（解码前去掉最后 8 字节）。连接后以 SPS/PPS + IDR 开始。

### `ws://<设备IP>:8001/results` — 推理结果（JSON 文本消息）

sscma-node 兼容格式，每个推理帧一条：

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "inference_time_ms": 71.0,
  "resolution": [640, 640],
  "boxes": [[320.0, 400.5, 90.9, 289.4, 0.874, "T12 engaged"]],
  "labels": ["T12 engaged"],
  "zone": { "occupancy": 1, "browsing": 0, "engaged": 1, "assistance": 0, "entry": 15, "exit": 13 }
}
```

- `boxes` 元素为 `[cx, cy, w, h, score, label]`：`cx, cy` 是推理分辨率（`resolution`）下的**框中心像素坐标**，`w, h` 为像素宽高，`score` 为 0–1，第 6 个元素是显示标签字符串 `"T<track_id> <状态>"`（控制台叠加框直接渲染）。
- `labels[i]` 与 `boxes[i][5]` 一一对应，供程序化消费。
- `zone` 是 MQTT 区域计数器的精简副本。

## 模型

| 名称 | 任务 | 文件 |
|---|---|---|
| `yolo11n`（默认） | detect | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` |

可通过向 `/userdata/local/apps/retail-vision.model` 首行写入绝对路径覆盖模型（控制台管理）；删除该文件即回退默认。

## 配置

运行时调参通过 `/etc/retail-vision.conf`（由 init 脚本 `/etc/init.d/K92retail-vision` 加载）中的启动参数完成：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--conf-threshold` | 0.5 | 检测置信度阈值 |
| `--dwell-engaged` | 1.5 秒 | 进入 `engaged` 的静止时长 |
| `--dwell-assist` | 20 秒 | 进入 `assistance` 的静止时长 |
| `--dwell-speed` | 10 px/s | 低于该速度视为静止 |
| `--window-duration` | 60 秒 | 区域指标滚动窗口 |
| `--person-height` | 1.7 m | 用于 m/s 估算的身高假设 |

本版本不提供控制台 `config_schema` 配置；分析区域始终为整个画面。

## 快速上手（集成清单）

1. 在 reCamera 控制台（Applications 页）激活本应用——同一时间只能有一个应用占用摄像头。
2. 用 NVR/VLC 打开 `rtsp://<设备IP>:8554/live0`。
3. 订阅：`mosquitto_sub -h <设备IP> -t "recamera/retail-vision/vision" -v`（需按上文开启 broker 外部监听）。
4. 将 `zone` 计数器接入 BI 大屏；用 `persons[]` 做逐客动线/热区分析，或对 `state == "assistance"` 触发店员协助告警。
