# 单目深度估计

用 reCamera 的单颗摄像头输出稠密相对深度。模型在 CVI TPU 上逐帧推理，结果以彩色
预览画在视频流角落，同时把一份精简的近距离指标通过 MQTT 发布，供 Home Assistant
或其它自动化使用。

## 这里的"相对深度"指什么

模型对每个像素输出一个数值，**数值越小表示越近**。这些数值没有单位、没有绝对尺度：
单目摄像头在没有参照物的情况下无法测量距离，本应用也不会把输出换算成距离。同一场景
在不同光照下，原始数值范围可能不同。

稳定的是**同一帧内部的远近排序**——画面哪部分比哪部分更近。所有派生字段
（`proximity`、`near_ratio`、`zones`）都在各自帧内归一化，因此即使原始深度不可跨帧
比较，这些字段可以。

室内场景精度最好：默认模型基于室内照片训练。室外，以及大面积无纹理表面（白墙、
晴空）下深度图会退化——远近排序大体仍对，细节结构会错。

## 真机实测

reCamera (SG2002 / CV181x)，BF16 模型，2026-09-05：

| | |
|---|---|
| 推理通道 | VPSS 接受 320x180，有效区域 `[0,0,320,180]`，无灰边 |
| 推理延迟 | p50 31.8 ms，p95 46.7 ms，max 76.2 ms |
| 稳定性（干净重启后 60 秒） | 带 PiP 590 帧 / 0 次挂死；`--no-pip` 562 帧 / 0 次挂死 |
| 深度 PiP | 渲染在出流右下角 (944,524) |

MQTT 里的 `zones` 数值与 PiP 上画出来的颜色一致。

延迟高于该模型基准报告的约 18 ms（那是纯 INT8 前向）。这里记录的是整条
逐帧路径上量的，包含预处理和伪彩映射。

**深度质量尚未验证。** 手头的测试画面是正对天花板的大面积无纹理场景，
恰是该模型已知的失效条件。判断远近结构是否正确，需要对准一个物体距离
差异明显的场景。

若画面卡住且日志里出现 `get chn frame fail`，是 VPSS 驱动挂死；这个状态
重启应用无法恢复，需要重启设备。本次是由「模型文件缺失时启动过一次」
触发的，与深度叠加无关。

## 模型

| | |
|---|---|
| 默认路径 | `/userdata/local/models/fastdepth_224_bf16.cvimodel` |
| 输入 | 1x3x224x224，RGB，CHW，归一化到 `[0,1]` |
| 输出 | 稠密 HxW 相对深度图 |
| 实测 | 约 19 ms/次 |

### 选 BF16 还是 INT8

同一个网络有两个版本，**默认用 BF16，没有特别理由就别换。**

| | BF16（默认） | INT8 |
|---|---|---|
| 校准集 | 不需要 | 约 200-500 张，且需域匹配 |
| `.cvimodel` | 2.9 MB | 1.4 MB |
| ION | 6.69 MB | 3.91 MB |
| cosine（vs float32） | 0.999998 | 0.999502 |
| SQNR（vs float32） | 39.98 dB | 16.38 dB |
| 推理 | 约 19 ms | 约 18 ms |

BF16 不需要校准表，也就没有校准集要采、没有采错的可能。INT8 换来 1.5 MB 和约 6%
的速度——在 66.7 ms 的帧周期和约 60 MB 的 ION 预算下，这两样都不构成约束。

**真正该看的是 SQNR 那一行。** 深度模型的输出是连续标量场，量化噪声会直接表现为
深度值抖动和局部远近翻转；同样的噪声在检测模型里会被 argmax 吃掉，在这里不会。
若确实要用 INT8，请用**你自己的 reCamera** 在**实际部署场景**下拍的图来校准——
通用图库代表不了这个网络真实遇到的激活分布。

模型**不随本 deb 一起安装**，启用前需自行拷贝到 `/userdata/local/models/`。控制台的
模型覆盖文件 `/userdata/local/apps/depth-estimation.model`（首行为绝对路径）存在时
优先于默认路径。

启动时应用会打印实际读到的 tensor 名称、shape、dtype 与量化参数；若不是单输入
1x3xHxW 加单输出稠密 HxW，直接报错退出，而不是继续跑出一个看起来合理的错误结果。

## 命令行参数

```
depth-estimation [options]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-m`, `--model PATH` | `/userdata/local/models/fastdepth_224_bf16.cvimodel` | 模型路径 |
| `--mqtt-host HOST` | `localhost` | broker 地址（legacy 模式） |
| `--mqtt-port PORT` | `1883` | broker 端口（legacy 模式） |
| `--mqtt-topic TOPIC` | `recamera/depth-estimation/results` | 结果主题 |
| `--mqtt-interval MS` | `500` | 两次发布之间的最小间隔（2 Hz） |
| `--near-threshold F` | `0.75` | 达到该 proximity 的像素计为"近" |
| `--near-ratio F` | `0.05` | 近像素占比达到该值时 `near_present` 为 true |
| `--no-pip` | 预览开启 | 关闭深度预览叠加 |
| `--pip-size WxH` | `320x180` | 预览尺寸 |
| `--no-rtsp` | RTSP 开启 | 关闭 RTSP 推流 |
| `--no-mqtt` | MQTT 开启 | 关闭 MQTT 发布 |
| `--no-debug` | 调试流开启 | 关闭调试 WebSocket |
| `--debug-port PORT` | `8001` | 调试 WebSocket 端口 |
| `-v`, `--verbose` | 关闭 | 每帧打一行日志 |
| `-h`, `--help` | | 帮助 |

init 脚本 `/etc/init.d/K92depth-estimation` 先读 `/etc/recamera.conf`，再读
`/etc/depth-estimation.conf`，把 `MQTT_HOST`、`MQTT_PORT`、`MQTT_TOPIC`、
`DEBUG_ENABLED`、`DEBUG_PORT`、`PIP_ENABLED`、`NEAR_THRESHOLD`、`NEAR_RATIO`
映射到上表参数。

## 视频

| | |
|---|---|
| RTSP | `rtsp://<device_ip>:8554/live0` —— 1280x720 @ 15 fps |
| 调试视频 | `ws://<device_ip>:8001/` |
| 调试结果 | `ws://<device_ip>:8001/results` |
| 快照 | `http://<device_ip>:8001/snapshot.jpg` |
| 推理通道 | 320x180 @ 15 fps |

每一帧都参与推理：15 fps 下 66.7 ms 的帧周期远大于约 18 ms 的前向耗时，不做跳帧。
MQTT 单独限流（见 `--mqtt-interval`）；预览每个推理帧都更新。

### 深度预览（画中画）

编码流**右下角**一块 320x180 的图块，距两边各内缩 16 像素，由一个硬件叠加区域绘制。
配色由近到远为红 → 橙 → 绿 → 青 → 蓝。

预览覆盖的正是主画面显示的传感器内容，图块中某个位置与主画面同一相对位置对应，
不存在需要换算的 letterbox。

本应用的预览与设备级隐私打码不能同时开启：同一 VPSS 通道内两个普通叠加区域不允许
相交。因此 manifest 声明 `"privacy_blur": false`，应用也从不创建打码区域。

## MQTT

主题：`recamera/depth-estimation/results`（默认 2 Hz）。

```json
{
  "timestamp": 1751000000000,
  "frame_id": 412,
  "inference_time_ms": 18.4,
  "depth": {
    "unit": "relative",
    "smaller_is_nearer": true,
    "source_size": [320, 180],
    "valid_roi": [0, 0, 320, 180],
    "min": 0.1120, "max": 0.9840, "mean": 0.5231,
    "p02": 0.1450, "p50": 0.5010, "p98": 0.9120,
    "near_ratio": 0.1834,
    "near_present": true,
    "zones": [0.21, 0.35, 0.19, 0.44, 0.87, 0.41, 0.62, 0.91, 0.58]
  }
}
```

| 字段 | 含义 |
|---|---|
| `unit` | 恒为 `"relative"`。这里没有米制读数。 |
| `smaller_is_nearer` | 恒为 `true`——原始尺度的方向。 |
| `source_size` | 计算深度所用推理帧的 `[w, h]`。 |
| `valid_roi` | 该帧中有效传感器内容的 `[x, y, w, h]`（见下）。 |
| `min`/`max`/`mean` | 整张深度图的原始模型单位值。 |
| `p02`/`p50`/`p98` | 同一批原始值的分位数。 |
| `near_ratio` | proximity 达到 `--near-threshold` 的像素占比 `[0,1]`。 |
| `near_present` | `near_ratio >= --near-ratio`。 |
| `zones` | 3x3 九宫格，行优先（索引 4 为中心）。每格取该格深度的 5% 分位并换算成 **proximity**——即该格最近内容有多近，`0` 远，`1` 为全帧最近。 |

proximity 按帧定义为

```
proximity = clamp((p98 - d) / (p98 - p02), 0, 1)
```

用 p02/p98 而不是 min/max：单个极端像素会把整帧重新缩放，让两帧看起来一样的画面
报出跳变的数值。

### Home Assistant

配置 `/userdata/local/ha.conf`（`HA_ENABLED=1`）后，通过 MQTT Discovery 发布三个实体：

| 实体 | 类型 | 取值 |
|---|---|---|
| Near Area | sensor，`%` | `near_ratio * 100` |
| Near Object | binary_sensor，`occupancy` | `near_present` |
| Center Nearest (relative) | sensor | `zones[4]`，`0..1` |

没有 `ha.conf` 时退回普通模式，连接 `--mqtt-host`/`--mqtt-port` 并只发布结果主题。

## 灰边，以及为什么在推理前就裁掉

摄像头 VPSS 会把 16:9 的传感器画面按比例装进每个通道，多出来的部分用灰色填充。
基于整幅照片训练的深度模型没见过这些灰边：它会给灰边编造深度，而这些编造值还会带偏
全帧 p02/p98 归一化，污染的不只是灰边区域，而是所有上报数值。

所以灰边在**推理之前**就被裁掉。推理通道正是为此配置成 16:9（320x180），让 VPSS
根本不产生灰边；若设备返回的帧形状不同，应用会在运行时算出内接的 16:9 矩形、裁剪，
并把它写进 `valid_roi`。首帧会打印实际帧尺寸和推导出的 ROI：

```
Inference frame 320x180, valid ROI [x=0,y=0,w=320,h=180] (16:9 channel, no letterbox)
```

随后预处理把该 ROI **拉伸**到模型输入尺寸，不做 letterbox——与 SSCMA 流水线其余部分
的约定一致——因此深度图线性对应回画面，下游无需反算任何偏移。

## ONVIF

提供 Device 与 Media2 服务及 snapshot URI，VMS 可以发现设备并拉流，开关由控制台的
ONVIF 设置统一管理，与其它应用一致。

ONVIF analytics metadata **不发布**。深度图里没有目标对象，由它伪造检测框等于往 VMS
时间线上写虚构内容。

## 不包含的能力

不做目标检测、不做逐目标深度采样、不做跟踪、不输出绝对距离。产出就是每帧一张相对
深度图和上述汇总指标。
