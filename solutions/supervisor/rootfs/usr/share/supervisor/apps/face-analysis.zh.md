<!-- app: face-analysis | version: 1.0.0 | doc-format: recamera-integration/v1 -->

# 人脸分析 — 集成指南

## 概述

**典型场景**：门店客群画像（分时段统计到店顾客性别/年龄构成）、活动效果的情绪反馈评估、展厅观众分析。全部分析在设备端完成，人脸原始影像不出设备。

实时人脸检测 + 逐人脸属性分析（年龄、性别、种族、情绪），完全在设备端运行（reCamera SG2002，RISC-V + TPU）。本应用是固定的多模型流水线：

1. **人脸检测** — YOLOv8n-face，每一推理帧都跑。
2. **年龄 / 性别 / 种族** — FairFace（或 InsightFace 变体），对每个检测到的人脸跑。
3. **情绪** — HSEmotion（AffectNet，8 类），对每个检测到的人脸跑。

输出：

| 通道 | 传输 | 用途 |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 视频，供 NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`，主题 `recamera/face-analysis/results` | 结构化分析结果（JSON） |
| 调试 WebSocket | `ws://<device-ip>:8001/` 与 `ws://<device-ip>:8001/results` | 浏览器实时调试（控制台界面） |

## RTSP 输出

- **URL**：`rtsp://<device-ip>:8554/live0`
- **编码**：H.264（Annex-B），默认 1280x720 @ 15 fps（启动时可配）
- **隐私**：RTSP 流上人脸区域**默认打码模糊**（`--no-blur` 关闭，最多 16 个区域）。
- 建议同一时间只有一个 RTSP 消费者；摄像头链路由本应用独占。

## MQTT 输出

### 连接

- **Broker**：设备上运行的 mosquitto，端口 **1883**（明文 TCP，无 TLS）。
  mosquitto 默认只监听 `localhost`；要从其他主机消费结果，需在 `/etc/mosquitto/mosquitto.conf` 开外部监听（`listener 1883 0.0.0.0`、`allow_anonymous true`），或桥接到你自己的 broker。
- **主题**：`recamera/face-analysis/results`
- **QoS**：0，**retain**：false
- **Client ID**（发布方）：`recamera-face-analysis`
- 每分析一帧发一条消息（默认推理速率：640x480 @ 最高 10 fps）。

### JSON schema

顶层对象：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `timestamp` | integer | Unix 纪元**毫秒** | 该帧的采集/发布时间 |
| `frame_id` | integer | — | 单调递增的帧计数器（应用启动时从 0 开始） |
| `inference_time_ms` | number | 毫秒 | 该帧整条流水线的推理耗时 |
| `face_count` | integer | — | `faces` 中的条目数 |
| `faces` | array | — | 每个检测到的人脸一个对象（可能为空） |

`faces[]` 每个元素：

| 字段 | 类型 | 单位 / 范围 | 说明 |
|---|---|---|---|
| `id` | integer | — | 人脸跟踪 ID（跟踪期间跨帧稳定） |
| `bbox.x` | number | 归一化 0–1 | 人脸框**左上角** X（占帧宽的比例） |
| `bbox.y` | number | 归一化 0–1 | 人脸框**左上角** Y（占帧高的比例） |
| `bbox.w` | number | 归一化 0–1 | 框宽（占帧宽的比例） |
| `bbox.h` | number | 归一化 0–1 | 框高（占帧高的比例） |
| `confidence` | number | 0–1 | 人脸检测得分 |
| `age_bin` | integer | 0–8 | **仅 FairFace 模型。** 年龄区间索引 |
| `age` | integer | 0–100 岁 | **仅 InsightFace 模型。** 连续年龄估计 |
| `age_label` | string | — | 可读年龄：FairFace 区间标签（`"0-2"`、`"3-9"`、`"10-19"`、`"20-29"`、`"30-39"`、`"40-49"`、`"50-59"`、`"60-69"`、`"70+"`）或 InsightFace 年龄的字符串形式 |
| `age_confidence` | number | 0–1 | 年龄预测置信度 |
| `gender` | string | `"male"` \| `"female"` | 性别预测 |
| `gender_confidence` | number | 0–1 | 性别预测置信度 |
| `race` | string | 见下 | **仅 FairFace**（否则不存在）。取值之一：`White`、`Black`、`Latino_Hispanic`、`East_Asian`、`Southeast_Asian`、`Indian`、`Middle_Eastern` |
| `race_confidence` | number | 0–1 | **仅 FairFace。** 种族预测置信度 |
| `emotion` | string | 见下 | 主导情绪：`angry`、`contempt`、`disgust`、`fear`、`happy`、`neutral`、`sad`、`surprise` |
| `emotion_confidence` | number | 0–1 | 主导情绪的置信度 |
| `emotion_probs` | object | 各 0–1 | 完整 8 类概率表，键：`angry`、`contempt`、`disgust`、`fear`、`happy`、`neutral`、`sad`、`surprise` |

说明：

- `age_bin` / `age` 二者恰有其一，取决于部署了哪个属性模型变体（默认部署 FairFace，故应为 `age_bin`）。
- 数值序列化保留 3 位小数。

### 坐标约定（MQTT）

`bbox` 是**基于左上角、归一化到 [0, 1]**，相对于推理帧（默认 640x480）。换算到流的任意分辨率（如 1280x720 的 RTSP 输出）的像素：

```text
px_left = bbox.x * frame_width
px_top  = bbox.y * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

（这与调试 WebSocket `/results` 通道不同——后者用基于中心点的像素坐标，见下。）

### 示例载荷

```json
{
  "timestamp": 1720771200123,
  "frame_id": 1502,
  "inference_time_ms": 84.250,
  "face_count": 1,
  "faces": [
    {
      "id": 7,
      "bbox": { "x": 0.412, "y": 0.238, "w": 0.144, "h": 0.221 },
      "confidence": 0.912,
      "age_bin": 3,
      "age_label": "20-29",
      "age_confidence": 0.671,
      "gender": "female",
      "gender_confidence": 0.983,
      "race": "East_Asian",
      "race_confidence": 0.542,
      "emotion": "happy",
      "emotion_confidence": 0.877,
      "emotion_probs": {
        "angry": 0.004, "contempt": 0.011, "disgust": 0.002, "fear": 0.006,
        "happy": 0.877, "neutral": 0.082, "sad": 0.009, "surprise": 0.009
      }
    }
  ]
}
```

## 调试 WebSocket（浏览器实时预览）

面向 reCamera 控制台的实时页；任何 WebSocket 客户端均可用。惰性：无客户端连接时不拷贝任何视频。**每路径客户端上限：2。**

### `ws://<device-ip>:8001/` — H.264 视频

- 二进制消息。每条消息是一个 **Annex-B** 字节流格式的 H.264 访问单元，尾部追加 **8 字节小端 `uint64`** 的 Unix **毫秒**时间戳：

```text
[ Annex-B H.264 字节 ...... ][ uint64 unix_ms，小端 ]
                              ^ 消息的最后 8 字节
```

- 连接建立时，流以 SPS + PPS 起始、随后一个 IDR 帧，解码器（如 JMuxer）可立即开始解码。

### `ws://<device-ip>:8001/results` — 推理结果（JSON，文本消息）

sscma-node 兼容格式，每推理帧一条消息：

```json
{
  "boxes": [[cx, cy, w, h, score, target]],
  "labels": ["face"],
  "resolution": [640, 480]
}
```

- `boxes` 元素为 `[cx, cy, w, h, score, target]`，其中 `cx, cy` 是推理分辨率（`resolution`）下的**像素中心点**，`w, h` 为像素宽高，`score` 为 0–1（或 0–100），`target` 是 `labels` 的类别索引。
- 注意与 MQTT 的区别：`/results` 是**基于中心点的像素**坐标；MQTT `bbox` 是**基于左上角的归一化**坐标。

## 流水线（固定，不可切换）

本应用**没有可切换模型**（`setModel` 不适用）。它运行一条固定级联：

| 级 | 模型 | 文件 | 作用 |
|---|---|---|---|
| 1 | `yolov8n-face` | `/userdata/local/models/yolov8n_face_cv181x_int8.cvimodel` | 人脸检测（INT8）——产出 `bbox`、`confidence`、跟踪 `id` |
| 2 | `fairface` | `/userdata/local/models/fairface_int8.cvimodel` | 逐人脸裁剪的年龄区间（9）、性别（2）、种族（7）（INT8） |
| 3 | `hsemotion` | `/userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel` | 情绪，逐人脸裁剪的 AffectNet 8 类（BF16） |

情绪推理默认每隔一帧跑一次；其间复用缓存结果。

## 快速上手（集成清单）

1. 从 reCamera 控制台（应用页）激活本应用——同一时间只有一个应用占用摄像头。
2. 把你的 NVR/VLC 指向 `rtsp://<device-ip>:8554/live0`。
3. 订阅：`mosquitto_sub -h <device-ip> -t "recamera/face-analysis/results" -v`（需上文的外部监听 broker 配置）。
4. 按 schema 解析；如需与视频对齐，用 `timestamp` + `frame_id`。
