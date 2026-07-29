<!-- app: yolo-detector | version: 1.1.0 | doc-format: recamera-integration/v1 | lang: zh -->

# 目标检测（YOLO）— 集成指南

## 概述

**典型场景**：出入口/周界无人值守（出现人员或车辆即刻通知）、快递到件检测、车位占用、货架/资产在位巡检。事件可接入 Home Assistant 或任意 MQTT 系统做告警与联动。

通用 YOLO 目标检测（80 类 COCO），完全在设备端运行（reCamera SG2002，RISC-V + TPU）。支持切换模型（YOLO11 / YOLO26 系列，detect 或 pose 任务）。默认还会运行**人员追踪器**，在原始检测之上叠加逐人驻留状态分析。

输出通道：

| 通道 | 传输方式 | 用途 |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 视频，供 NVR / VMS / VLC 使用 |
| MQTT | `<device-ip>:1883`，主题见下文 | 结构化检测/追踪结果（JSON） |
| 调试 WebSocket | `ws://<device-ip>:8001/` 与 `ws://<device-ip>:8001/results` | 浏览器实时调试（控制台 UI） |

## RTSP 输出

- **URL**：`rtsp://<device-ip>:8554/live0`
- **编码**：H.264（Annex-B），1280x720 @ 15 fps（默认值，可在启动时配置）
- 建议同一时间只有一个 RTSP 消费端；摄像头管线由本应用独占。

## MQTT 输出

### 连接

- **Broker**：设备上运行的 mosquitto，端口 **1883**（明文 TCP，无 TLS）。
  mosquitto 默认只监听 `localhost`；如需从其他主机消费结果，请在 `/etc/mosquitto/mosquitto.conf` 中开启外部监听（`listener 1883 0.0.0.0`、`allow_anonymous true`），或桥接到你自己的 broker。
- **主题**：以应用 manifest 及控制台 Live 页显示为准 —— 控制台托管部署下为 `recamera/yolo-detector/results`。二进制内置默认值是 `recamera/yolo/detections`（可用 `--mqtt-topic` 覆盖）。**请务必以控制台 Live 页显示的主题为准。**
- **QoS**：0，**retain**：false
- **Client ID**（发布端）：`recamera-yolo-detector`
- 每帧推理发布一条消息（默认 640x640，最高 15 fps）。

### 载荷格式 A —— 追踪模式（默认）

人员追踪**默认开启**（`--no-tracking` 关闭）。此模式只发布被追踪的**人**（COCO `person` 类），并附带运动/驻留分析：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `timestamp` | integer | Unix 纪元**毫秒** | 该帧的采集/发布时间 |
| `frame_id` | integer | — | 单调递增帧计数（应用启动时从 0 开始） |
| `inference_time_ms` | number | 毫秒 | 该帧的检测推理耗时 |
| `zone_occupancy.total` | integer | — | 画面内被追踪人员总数 |
| `zone_occupancy.browsing` | integer | — | 处于 `transient` 或 `dwelling` 状态的人数 |
| `zone_occupancy.engaged` | integer | — | 处于 `engaged` 状态的人数 |
| `zone_occupancy.assistance` | integer | — | 处于 `assistance` 状态的人数 |
| `line_crossing.in` | integer | — | **仅在配置了进出计数线时出现。** 自应用启动以来累计的 "in" 跨线次数（重启清零） |
| `line_crossing.out` | integer | — | **仅在配置了进出计数线时出现。** 自应用启动以来累计的 "out" 跨线次数（重启清零） |
| `persons` | array | — | 每个被追踪人员一个对象（可能为空） |

注意：配置了**统计区域**（见下方「配置」）时，`zone_occupancy` 只统计 bbox 中心落在多边形内的人员；`persons[]` 仍包含画面内所有被追踪人员。

`persons[]` 的每个元素：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `track_id` | integer | — | 稳定的人员追踪 ID（跨帧保持） |
| `confidence` | number | 0–1 | 对应检测框的置信度 |
| `bbox.x` | number | 归一化 0–1 | **框中心** X（相对画面宽度） |
| `bbox.y` | number | 归一化 0–1 | **框中心** Y（相对画面高度） |
| `bbox.w` | number | 归一化 0–1 | 框宽度（相对画面宽度） |
| `bbox.h` | number | 归一化 0–1 | 框高度（相对画面高度） |
| `speed_px_s` | number | 像素/秒 | 在推理画面（默认 640x640）中的移动速度 |
| `speed_normalized` | number | %/秒 | 以自身身高百分比表示的速度 |
| `state` | string | 见下 | 驻留状态：`transient`（路过）、`dwelling`（停留 < 1.5 s）、`engaged`（停留 1.5–20 s）、`assistance`（停留 > 20 s） |
| `dwell_duration_sec` | number | 秒 | 当前驻留片段中保持静止的时长 |

示例：

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "inference_time_ms": 71.0,
  "zone_occupancy": { "total": 2, "browsing": 1, "engaged": 1, "assistance": 0 },
  "persons": [
    {
      "track_id": 12,
      "confidence": 0.874,
      "bbox": { "x": 0.5123, "y": 0.6011, "w": 0.1420, "h": 0.4522 },
      "speed_px_s": 3.4,
      "speed_normalized": 1.2,
      "state": "engaged",
      "dwell_duration_sec": 4.7
    }
  ]
}
```

### 载荷格式 B —— 原始检测（`--no-tracking`）

发布所有检测类别，不含追踪分析：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `timestamp` | integer | Unix 纪元**毫秒** | 该帧的采集/发布时间 |
| `frame_id` | integer | — | 单调递增帧计数 |
| `inference_time_ms` | number | 毫秒 | 该帧推理耗时 |
| `detection_count` | integer | — | `detections` 数组元素个数 |
| `detections` | array | — | 每个检测一个对象 |

`detections[]` 的每个元素：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `id` | integer | — | 检测 ID |
| `class_id` | integer | 0–79 | COCO 类别索引（0 = person） |
| `class_name` | string | — | COCO 类别名（`person`、`bicycle`、`car`…） |
| `confidence` | number | 0–1 | 检测置信度 |
| `bbox.x` / `bbox.y` | number | 归一化 0–1 | **框中心**（相对画面宽/高） |
| `bbox.w` / `bbox.h` | number | 归一化 0–1 | 框宽/高（相对画面） |

### 坐标约定（MQTT）

`bbox` 为**中心点表示、归一化到 [0, 1]**，参考系是推理画面（默认 640x640）。换算为任意分辨率（如 1280x720 RTSP 输出）上的左上角像素坐标：

```text
px_left = (bbox.x - bbox.w / 2) * frame_width
px_top  = (bbox.y - bbox.h / 2) * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

## 调试 WebSocket（浏览器实时预览）

供 reCamera 控制台 Live 页使用；任何 WebSocket 客户端均可连接。惰性推送：无客户端连接时不拷贝视频。**每路径最多 2 个客户端。**

### `ws://<device-ip>:8001/` —— H.264 视频

- 二进制消息。每条消息是一个 **Annex-B** 格式的 H.264 访问单元，尾部附 **8 字节小端 `uint64`** Unix 毫秒时间戳：

```text
[ Annex-B H.264 字节 ...... ][ uint64 unix_ms，小端 ]
                              ^ 消息的最后 8 字节
```

- 连接建立后流以 SPS + PPS + IDR 帧开始，解码器（如 JMuxer）可立即起播。

### `ws://<device-ip>:8001/results` —— 推理结果（JSON 文本消息）

sscma-node 兼容格式，每帧一条：

```json
{
  "boxes": [[cx, cy, w, h, score, target]],
  "labels": ["person", "bicycle", "car"],
  "resolution": [640, 640]
}
```

- `boxes` 元素为 `[cx, cy, w, h, score, target]`，其中 `cx, cy` 是推理分辨率（`resolution`）下的**像素中心点**，`w, h` 为像素宽高，`score` 为 0–1（或 0–100），`target` 是类别名称字符串（如 `"person"`）；同时提供 `labels` 数组供按索引消费的客户端使用。
- 与 MQTT 的区别：`/results` 使用**像素**坐标；MQTT `bbox` 使用**归一化**坐标。二者均为中心点表示。

## 模型（可切换）

可在控制台（Live 页 → Model）切换当前模型。切换会重启应用，流会短暂中断。

| 名称 | 任务 | 文件 |
|---|---|---|
| `yolo11n`（默认） | detect | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` |
| `yolo11n-pose` | pose | `/userdata/local/models/yolo11n_pose_cv181x_int8.cvimodel` |
| `yolo26n` | detect | `/userdata/local/models/yolo26n_cv181x_int8.cvimodel` |

机制：控制台把所选模型的绝对路径写入 `/userdata/local/apps/yolo-detector.model`（单行）；init 脚本在下次启动时通过 `-m` 传给二进制。删除该文件即回退到内置默认。

注意：pose 模型只检测人；发布的载荷仍保持相同的检测/追踪 JSON 结构（关键点不经 MQTT 发布）。

## 配置（控制台托管）

控制台（Applications 页 → Configure）把经过校验的设置写入 `/userdata/local/apps/yolo-detector.config.json`；应用在启动时读取该文件，且当应用处于激活状态时保存配置会自动重启应用。删除该文件即恢复内置默认值。init 脚本中的显式 CLI 参数优先级仍高于配置文件。

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `confidence` | number 0.05–0.95 | 0.25 | 检测置信度阈值 |
| `tracking` | boolean | `true` | 启用人员追踪（载荷格式 A）。`false` 切换为原始检测（格式 B） |
| `count_zone` | 多边形（3–8 个点） | 未设置（全画面） | 统计区域。`zone_occupancy` 只统计 bbox **中心**在多边形内的人员 |
| `entry_line` | 线段 + 方向 | 未设置 | 进出计数线；启用载荷中的 `line_crossing` 字段 |

所有空间坐标均**归一化到 [0, 1]**（与分辨率无关），例如：

```json
{
  "confidence": 0.35,
  "count_zone": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
  "entry_line": { "a": [0.5, 0.0], "b": [0.5, 1.0], "direction": "ab_in" }
}
```

### 跨线判定语义

- 当被追踪人员的 bbox 中心在相邻两次追踪位置之间从线段 `a -> b` 的一侧移动到另一侧时计一次跨线（按线段-线段相交判定，从线段端点外绕过不计数）。
- **方向约定**：沿 `a` 指向 `b` 的方向看，`"direction": "ab_in"` 时**从左侧穿到右侧**计为 `in`（右到左计为 `out`）；`"ab_out"` 则相反。
- 计数为应用启动以来的累计值，每次重启（包括保存配置、切换模型）都会清零。需要持久总量的消费端应在外部累加增量。

## 快速上手（集成清单）

1. 在 reCamera 控制台（Applications 页）激活本应用 —— 同一时间只有一个应用占用摄像头。
2. 将 NVR/VLC 指向 `rtsp://<device-ip>:8554/live0`。
3. 订阅：`mosquitto_sub -h <device-ip> -t "<控制台 Live 页显示的主题>" -v`（需先按上文开启 broker 外部监听）。
4. 选择追踪或原始模式：默认追踪（仅人员）。如需全类别原始检测，在 init 脚本参数中加 `--no-tracking`。
