# 跌倒检测 — 集成说明

应用在 reCamera TPU 上运行 YOLO11n-Pose，稳定关联一个目标，把 3.2 秒的
COCO-17 姿态历史送入轻量时序分类器。髋部速度、躯干角度和人体框宽高比仍用于
可解释的 `suspected` / 恢复状态，最终告警由跨帧分类器确认；推理完全在设备本地完成，
不依赖云服务。

状态为 `normal`、`suspected`、`fallen`、`recovering`：

- `normal`：没有跌倒证据，或恢复窗口已经完成；
- `suspected`：多特征证据正在累计；
- `fallen`：确认跌倒，只有刚进入该状态的那一帧 `fall_event=true`，`event_id`
  加一；
- `recovering`：检测到站立，但必须持续满足恢复窗口才回到正常。

如果应用启动首帧已经是躺卧姿态，会报告 `fallen`，但不产生
`fall_event`（启动前没有可观察到的转变）。

## 配置

console 写入 `/userdata/local/apps/fall-detection.config.json`，也可通过 SSH
编辑，应用运行中会轮询加载。所有阈值都可配置：

| 键 | 默认值 | 含义 |
|---|---:|---|
| `confidence` | 0.40 | 人体置信度阈值 |
| `keypoint_confidence` | 0.50 | COCO-17 关键点阈值 |
| `hip_drop_speed_threshold` | 0.25 | 归一化画面中的髋部向下速度/秒 |
| `hip_drop_distance_threshold` | 0.02 | 相对最近非水平姿态的髋部净下坠距离 |
| `motion_window_sec` | 0.75 | 快速下坠到身体转横的最大间隔 |
| `torso_angle_threshold_deg` | 55 | 躯干偏离竖直达到躺卧的角度 |
| `bbox_aspect_ratio_threshold` | 1.25 | 躺卧框宽/高比 |
| `min_suspected_features` | 2 | 速度、躯干角、宽高比中至少命中的数量 |
| `confirmation_sec` | 0.80 | 证据持续确认时间 |
| `suspected_timeout_sec` | 1.50 | 疑似状态最长等待时间 |
| `occlusion_grace_sec` | 0.75 | 落地后关键点短暂丢失的确认宽限 |
| `recovery_torso_angle_deg` | 35 | 站立躯干角上限 |
| `recovery_aspect_ratio` | 1.10 | 站立框宽高比上限 |
| `recovery_window_sec` | 2.00 | 站立持续恢复时间 |
| `cooldown_sec` | 3.00 | 事件后的再次报警抑制时间 |

## MQTT / Home Assistant

主题：`recamera/fall-detection/results`，每处理一帧发布一条 JSON。自动发现
实体包括 **Fall Detected**、**Fall State**、**Fall Event ID**、**Person Count**
和 **Person Present**。`fall_event` 是边沿标志，`event_id` 可用于去重；
`fall_detected` 在 `fallen` 和 `recovering` 期间保持 true。`features` 中每帧
输出诊断值，便于离线调阈值。

## 视频、相机和多人说明

- RTSP：`rtsp://<设备IP>:8554/live0`
- 预览：`ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results`
- 快照：`http://<设备IP>:8001/snapshot.jpg`

请让整个人体以及双肩、双髋保持在画面中。本版本明确只分析一个人：首次取最高
置信度目标，后续用人体框重叠稳定关联，不会每帧按最高分在两个人之间切换。
`person_count` / `fallen_count` 因而只有 0 或 1；没有宣称完整的多人跌倒分析。

初始化脚本为 `K92fall-detection`（K 前缀），由 supervisor 管理启动，不会在开机
时抢占摄像头。启动前会停止常见的 sscma-node、Node-RED、检测/OCR 服务，等待进程
退出，并设置 CVI 所需完整 `LD_LIBRARY_PATH`。同时运行多个相机应用会造成 VPSS 或
RTSP 端口冲突。

## 纯 C++ 测试

`tests/fall_detector_test.cpp` 不依赖设备或 SSCMA，除几何、恢复和冷却外，还覆盖
时序模型确认门控及落地后姿态丢失路径：

```bash
c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/fall_detector.cpp \
  solutions/fall-detection/tests/fall_detector_test.cpp -o /tmp/fall_detector_test
/tmp/fall_detector_test
```

如需使用开源视频或公开数据集做真实 NPU 端到端检查，可用离线 RGB 模式。它加载同一
个 cvimodel、`PoseDetector`、特征提取和 `FallDetector`，跳过摄像头/RTSP、debug 和
MQTT。输入是不带头的连续 RGB888 帧：

```bash
ffmpeg -i public-fall-video.mp4 \
  -vf 'fps=15,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black' \
  -pix_fmt rgb24 -f rawvideo /tmp/video.rgb
fall-detection --model /path/to/yolo11n_pose_cv181x_int8.cvimodel \
  --offline-rgb /tmp/video.rgb --offline-width 640 --offline-height 640 --offline-fps 15 \
  > results.jsonl
```

每帧输出一行 JSON，最后一行 `summary` 包含 `frames`、`events`、`last_state` 和
`fall_detected`。返回码 0 表示所有完整帧均处理成功；2 表示文件不存在、为空或在
帧中间结束；1 表示模型初始化失败，便于 CI 或带标注的公开视频回归检查。
若 NPU 在某帧推理失败则返回 3。
