<!-- app: ppocr-reader | version: 0.4.0 | doc-format: recamera-integration/v1 -->

# PP-OCR 文字识别 — 集成指南

## 概述

完全在设备端运行的场景文字识别（reCamera SG2002，RISC-V + TPU），固定两模型 PP-OCRv3 流水线：

1. **文字检测** — DBNet（MobileNetV3 + RSE-FPN head，480x480 输入，INT8/BF16 混合量化）输出四点多边形文字区域。
2. **文字识别** — 每个区域经裁剪与透视校正后送入 SVTR-LCNet（48x320 输入，BF16），CTC 解码，字典含 6623 字符（简体中文、英文、数字、标点）。

区域按从上到下、从左到右排序；每帧最多识别 `--kmax` 个区域（默认 5）。相机以 640x480 采集；端到端 OCR 约 1–3 fps（检测约 65 ms，每个识别区域另加约 50 ms），RTSP 视频保持实时。

输出通道：

| 通道 | 传输 | 用途 |
|---|---|---|
| RTSP | `rtsp://<设备IP>:8554/live0` | H.264 视频，接 NVR / VMS / VLC |
| MQTT | `<设备IP>:1883`，主题见下文 | 每帧识别文本 + 多边形（JSON） |
| 调试 WebSocket | `ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results` | 浏览器实时调试（控制台 UI） |

## RTSP 输出

- **URL**：`rtsp://<设备IP>:8554/live0`
- **编码**：H.264，640x480 @ 15 fps（默认值）
- 启动参数 `--no-rtsp` 可关闭。

## MQTT 输出

### 连接

- **Broker**：设备上的 mosquitto，端口 **1883**（明文 TCP，无 TLS）。默认只监听 `localhost`；如需从其他主机订阅，请在 `/etc/mosquitto/mosquitto.conf` 中开启外部监听（`listener 1883 0.0.0.0`、`allow_anonymous true`）或桥接到自有 broker。
- **主题**：`recamera/ppocr/texts`（manifest 默认值；可在 `/etc/ppocr-reader.conf` 中用 `MQTT_TOPIC` 覆盖）。**以控制台 Live 页显示为准。**
- **QoS**：0，**retain**：false，**客户端 ID**：`recamera-ppocr-reader`
- 每个处理帧发布一条消息（画面无文字时 `texts` 为空数组 `[]`）。

### 消息格式

顶层字段：

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `timestamp` | integer | Unix 毫秒 | 帧采集/发布时间 |
| `frame_id` | integer | — | 单调递增帧计数（应用启动时归零） |
| `inference_time_ms.detection` | number | 毫秒 | 检测推理耗时 |
| `inference_time_ms.recognition` | number | 毫秒 | 识别耗时（所有区域累计） |
| `inference_time_ms.total` | number | 毫秒 | 流水线总耗时 |
| `text_count` | integer | — | `texts[]` 数组长度 |
| `frame_width` / `frame_height` | integer | 像素 | `box` 坐标归一化参照的推理帧尺寸（默认 640x480） |
| `texts` | array | — | 每个检测到的文字区域一个对象 |

`texts[]` 每个元素：

| 字段 | 类型 | 单位/范围 | 说明 |
|---|---|---|---|
| `id` | integer | — | 区域在当前帧内的序号 |
| `box` | array | 归一化 0–1 | 四点多边形 `[[x,y],[x,y],[x,y],[x,y]]`，左上起顺时针，参照 `frame_width`x`frame_height` |
| `text` | string | UTF-8 | 识别出的文字（未识别的区域为空字符串，例如超出 `--kmax`） |
| `confidence` | number | 0–1 | 识别（CTC）置信度 |
| `det_confidence` | number | 0–1 | 检测置信度 |

示例：

```json
{
  "timestamp": 1768969602957,
  "frame_id": 42,
  "inference_time_ms": { "detection": 65.2, "recognition": 48.3, "total": 113.5 },
  "text_count": 2,
  "frame_width": 640,
  "frame_height": 480,
  "texts": [
    {
      "id": 0,
      "box": [[0.0156,0.0417],[0.3125,0.0417],[0.3125,0.1042],[0.0156,0.1042]],
      "text": "Hello World",
      "confidence": 0.95,
      "det_confidence": 0.89
    },
    {
      "id": 1,
      "box": [[0.0156,0.1250],[0.2344,0.1250],[0.2344,0.2083],[0.0156,0.2083]],
      "text": "你好世界",
      "confidence": 0.88,
      "det_confidence": 0.91
    }
  ]
}
```

### 坐标约定（MQTT）

`box` 为**四点多边形，归一化到 [0, 1]**，参照推理帧（`frame_width` x `frame_height`，默认 640x480）。叠画到 RTSP 视频（默认同为 640x480）：

```text
px = point[0] * frame_width
py = point[1] * frame_height
```

注意：这与下述调试 WebSocket 通道不同（后者是轴对齐、中心点基准的像素框）。

## 调试 WebSocket（浏览器实时预览）

供 reCamera 控制台 Live 页使用；任何 WebSocket 客户端也可接入。惰性推送：无客户端连接时零拷贝开销。每条路径最多 2 个客户端。`--no-debug` 关闭，`--debug-port` 改端口（默认 8001）。

### `ws://<设备IP>:8001/` — H.264 视频

二进制消息：每条为一个 H.264 Annex-B 访问单元，尾部附加 **8 字节小端 `uint64`** Unix 毫秒时间戳（解码前去掉最后 8 字节）。连接后以 SPS/PPS + IDR 开始。

### `ws://<设备IP>:8001/results` — 推理结果（JSON 文本消息）

sscma-node 兼容格式，每个处理帧一条：

```json
{
  "timestamp": 1768969602957,
  "frame_id": 42,
  "inference_time_ms": 113.5,
  "resolution": [640, 480],
  "boxes": [[105.0, 35.0, 190.0, 30.0, 0.890, "Hello World"]],
  "labels": ["Hello World"],
  "texts": ["Hello World"]
}
```

- `boxes` 元素为 `[cx, cy, w, h, score, label]`：OCR 多边形被压缩为**轴对齐外接矩形**，`cx, cy` 是推理分辨率（`resolution`）下的框中心像素坐标，`w, h` 为像素宽高，`score` 为检测置信度（0–1），第 6 个元素是识别文本（截断至 32 字节；未识别区域为占位符 `"text"`），控制台叠加框直接渲染。
- `labels[i]` 与 `boxes[i][5]` 一一对应，供程序化消费。
- `texts` 携带**完整未截断**的识别文本（与 `boxes` 顺序一致）。
- `inference_time_ms` 为流水线总耗时（检测 + 识别）。

## 流水线模型与字典

固定两模型流水线（无可切换的 `models[]`；manifest 在 `pipeline[]` 中列出）。首次启动前需部署以下三个文件：

| 组件 | 任务 | 设备路径 |
|---|---|---|
| `ppocr-det` | text-detect | `/userdata/local/models/ppocr_det_cv181x_mix.cvimodel` |
| `ppocr-rec` | text-recognize | `/userdata/local/models/ppocr_rec_cv181x_bf16.cvimodel` |
| 字典 | CTC 解码 | `/userdata/local/dict/ppocr_keys_v1.txt` |

推荐转换产物（来自 `model_conversion/recamera_ppocr`）：检测用**混合精度**模型（`ppocr_det_cv181x_mix.cvimodel` — sigmoid/attention 层 BF16，其余 INT8），识别用 **BF16** 模型（`ppocr_rec_cv181x_bf16.cvimodel`）。纯 INT8 版本存在但识别精度较低；若部署 INT8，请在 `/etc/ppocr-reader.conf` 中显式覆盖路径。

另有英文专用识别器：`ppocr_rec_en_cv181x_bf16.cvimodel` + `en_dict.txt`（在 `/etc/ppocr-reader.conf` 中设置 `REC_MODEL` / `DICT_FILE`）。

## 配置

运行时调参通过 `/etc/ppocr-reader.conf`（由 init 脚本 `/etc/init.d/K92ppocr-reader` 加载）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DET_MODEL` | `/userdata/local/models/ppocr_det_cv181x_mix.cvimodel` | 检测模型路径 |
| `REC_MODEL` | `/userdata/local/models/ppocr_rec_cv181x_bf16.cvimodel` | 识别模型路径 |
| `DICT_FILE` | `/userdata/local/dict/ppocr_keys_v1.txt` | 字符字典 |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_TOPIC` | `localhost` / `1883` / `recamera/ppocr/texts` | MQTT broker 设置 |
| `DEBUG_ENABLED` / `DEBUG_PORT` | `1` / `8001` | 调试 WebSocket 流 |
| `KMAX` | 5 | 每帧最多识别区域数（0 = 不限） |
| `ENHANCE_MODE` | `none` | 裁剪增强：`none`、`clahe`、`gray`、`adaptive` |
| `VERBOSE` | 0 | 详细日志 |

本版本不提供控制台 `config_schema` 配置，也不支持模型切换。

## 快速上手（集成清单）

1. 将两个模型与字典复制到上表所列设备路径。
2. 在 reCamera 控制台（Applications 页）激活本应用——同一时间只能有一个应用占用摄像头。
3. 用 NVR/VLC 打开 `rtsp://<设备IP>:8554/live0`。
4. 订阅：`mosquitto_sub -h <设备IP> -t "recamera/ppocr/texts" -v`（需按上文开启 broker 外部监听）。
5. 在业务侧消费 `texts[]`：与预期字符串比对（标签、仪表读数、车牌等），按 `confidence` 过滤，用 `box` 多边形做定位。
