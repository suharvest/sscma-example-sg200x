# 跌倒检测 — 集成说明

应用在 reCamera TPU 上运行 YOLO11n-Pose，稳定关联多个人体轨道，把每人的 3.2 秒
COCO-17 姿态历史送入轻量时序分类器。髋部速度、躯干角度和人体框宽高比仍用于
可解释的 `suspected` / 恢复状态，最终告警由跨帧分类器确认；推理完全在设备本地完成，
不依赖云服务。

状态为 `normal`、`suspected`、`fallen`、`recovering`：

- `normal`：没有跌倒证据，或恢复窗口已经完成；
- `suspected`：多特征证据正在累计；
- `fallen`：确认跌倒，只有刚进入该状态的那一帧 `fall_event=true`，`event_id`
  加一；
- `recovering`：检测到站立，但必须持续满足恢复窗口才回到正常。

如果应用启动首帧已经是躺卧姿态，仍保持 `normal`：启动前没有可观察到的转变，
不能仅凭静态躺卧姿态判定跌倒。

## 配置

console 写入 `/userdata/local/apps/fall-detection.config.json`，也可通过 SSH
编辑，应用运行中会轮询加载。所有阈值都可配置：

| 键 | 默认值 | 含义 |
|---|---:|---|
| `confidence` | 0.40 | 人体置信度阈值 |
| `keypoint_confidence` | 0.50 | COCO-17 关键点阈值 |
| `temporal_confirmation_required` | true | 必须由有效当前姿态和学习时序模型共同确认；false 才启用旧版纯几何确认 |
| `hip_drop_speed_threshold` | 0.25 | 归一化画面中的髋部向下速度/秒 |
| `hip_drop_distance_threshold` | 0.02 | 相对最近非水平姿态的髋部净下坠距离 |
| `motion_window_sec` | 0.75 | 快速下坠到身体转横的最大间隔 |
| `torso_angle_threshold_deg` | 55 | 躯干偏离竖直达到躺卧的角度 |
| `bbox_aspect_ratio_threshold` | 1.25 | 躺卧框宽/高比 |
| `min_suspected_features` | 2 | 速度、躯干角、宽高比中至少命中的数量 |
| `confirmation_sec` | 0.80 | 证据持续确认时间 |
| `suspected_timeout_sec` | 1.50 | 疑似状态最长等待时间 |
| `occlusion_grace_sec` | 0.75 | 关键点短暂丢失时保留轨迹/状态；不用于确认新事件 |
| `recovery_torso_angle_deg` | 35 | 站立躯干角上限 |
| `recovery_aspect_ratio` | 1.10 | 站立框宽高比上限 |
| `recovery_window_sec` | 2.00 | 站立持续恢复时间 |
| `cooldown_sec` | 3.00 | 事件后的再次报警抑制时间 |

## MQTT / Home Assistant

主题：`recamera/fall-detection/results`，每处理一帧发布一条 JSON。自动发现
实体包括 **Fall Detected**、**Fall State**、**Fall Event ID**、**Person Count**、
**Fallen Count** 和 **Person Present**。`fall_event` 是边沿标志，`event_id` 可用于去重；
`fall_detected` 在 `fallen` 和 `recovering` 期间保持 true。`features` 中每帧
输出诊断值，便于离线调阈值。

## 视频、相机和多人说明

- RTSP：`rtsp://<设备IP>:8554/live0`
- 预览：`ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results`
- 快照：`http://<设备IP>:8001/snapshot.jpg`

请让整个人体以及双肩、双髋保持在画面中。应用会返回画面中的所有姿态人体，使用
轻量 IoU/中心距离跟踪器关联人体框。每个 `track_id` 都有独立的时序历史和跌倒状态机，
不会把附近两个人的历史拼接在一起。检测短暂丢失时，`persons[]` 会保留
`person_detected:false` 的轨道，支持落地后的遮挡确认；轨道在超时后清理。
顶层旧字段是聚合值：`fall_detected` / `fall_event` 为所有保留轨道的 OR，
`person_count` 是当前可见人数，`fallen_count` 统计保留轨道中的跌倒状态，`state` 是最严重状态。
顶层 `event_id` 是流级递增序号（`event_id_scope:"stream_global_event_id"`），
`persons[]` 内的 `event_id` 仍是各轨道自己的序号。

每个 `persons[]` 项包含 `track_id`、`state`、`event_id`、`features`、`keypoints` 和稳定的
COCO-17 `pose17`；MQTT 仍保留顶层旧字段以兼容现有消费者。debug `/results` 会为每个可见人体
输出一个骨架分组。

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

人体关联和聚合契约也可在主机上离线测试，不依赖设备：

```bash
c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/box_tracker.cpp \
  components/geometry/src/norm_box.cpp \
  solutions/fall-detection/tests/multi_person_tracker_test.cpp \
  -I components/geometry/include -o /tmp/multi_person_tracker_test
/tmp/multi_person_tracker_test

c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/fall_detector.cpp \
  solutions/fall-detection/main/payload_aggregate.cpp \
  solutions/fall-detection/tests/payload_aggregate_test.cpp \
  -o /tmp/payload_aggregate_test
/tmp/payload_aggregate_test
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

每帧输出一行 JSON，最后一行 `summary` 包含 `frames`、流级 `events`、带边沿的
`event_edges`、`last_state` 和 `fall_detected`。返回码 0 表示所有完整帧均处理成功；2 表示文件不存在、为空或在
帧中间结束；1 表示模型初始化失败，便于 CI 或带标注的公开视频回归检查。
若 NPU 在某帧推理失败则返回 3。
