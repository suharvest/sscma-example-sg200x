# 健身教练 — 集成说明

仅凭摄像头统计指定动作的次数，并把实时计数通过 MQTT 发出。推理全部在设备本地完成，画面不出设备。

检测用 YOLO11n-Pose（COCO 17 关键点）跑在 TPU 上。计数器读取每个动作对应的一个关节角度，驱动一个带迟滞的状态机。

## 选择运动模式

目前内置三个动作：

| `mode` | 动作 | 跟踪的角度 |
|---|---|---|
| `squat` | 深蹲 | 膝关节屈曲 — 髋 / 膝 / 踝 |
| `push_up` | 俯卧撑 | 肘关节屈曲 — 肩 / 肘 / 腕 |
| `hammer_curl` | 哑铃弯举 | 肘关节屈曲，左右手分别计数 |

在 console 里本应用的页面上设置（**Exercise / 运动模式**），同时设定每组次数和组数。保存会重启应用，计数清零。

同样的设置也可以直接写文件——手机、脚本或 Node-RED 就是这样绕过 console 切换模式的：

```bash
cat > /userdata/local/apps/fitness-trainer.config.json <<'EOF'
{"mode":"push_up","target_reps":15,"target_sets":4}
EOF
```

应用会轮询这个文件，约两秒内生效，**不需要重启**。切换动作会清零计数；只改次数/组数则保留计数。

如果设备专用于某一个动作，可以在 `/etc/fitness-trainer.conf` 里钉死（`EXERCISE_MODE=squat`），它会覆盖 console 的设置。

## 配置项

| 键 | 默认值 | 含义 |
|---|---|---|
| `mode` | `squat` | 运动模式，见上表 |
| `target_reps` | 12 | 每组次数 |
| `target_sets` | 3 | 训练总组数 |
| `idle_reset_seconds` | 60 | 画面中无人超过这么久后清零计数（0 = 不清零） |
| `confidence` | 0.40 | 人体检测置信度阈值 |
| `keypoint_confidence` | 0.50 | 单个关键点的阈值，低于它的关节视为不可见 |

## MQTT 输出

主题：`recamera/fitness-trainer/results`，每处理一帧发一条。

```json
{
  "timestamp": 1753689600123,
  "frame_id": 4821,
  "inference_time_ms": 96,
  "exercise": "squat",
  "stage": "down",
  "angle": 92.4,
  "reps": 7,
  "target_reps": 12,
  "set": 2,
  "target_sets": 3,
  "workout_complete": false,
  "person_detected": true,
  "tracking": true,
  "rep_completed": false,
  "set_completed": false
}
```

- `timestamp` —— 设备墙上时钟的毫秒数。reCamera 出厂默认没有加载 RTC 模块、也没有后备电池，所以从未同步过时间的设备上，这个值是从开机算起的，不是 Unix 纪元。需要真实时间戳请同步 NTP（或 `insmod cv181x_rtc.ko`）；只需要排序的话用 `frame_id`。
- `stage` —— 深蹲和俯卧撑是 `up` / `down`，弯举是 `curl` / `extend`；没跟上人时是 `idle`，检测到人但这个动作需要的关节不可见时是 `out of frame`。
- `angle` —— 跟踪的关节角度（度），已平滑。没有读数时该字段不出现。
- `rep_completed` / `set_completed` —— 只在事件发生的那一条为 true，订阅方不必自己比对计数差值。
- `reps_left` / `reps_right` —— 仅弯举。两只手都可见时，组计数按较慢的那只手推进。
- `form_warning` —— 只在动作有问题时出现，例如 `"Partial rep - squat deeper"`（幅度不够）或 `"Left elbow drifting - keep it at your side"`（左肘外翻）。

`reps` 是**当前这一组**的次数；一组完成后归零、`set` 加一。最后一组会停在 `target_reps`，同时 `workout_complete` 置 true。

### Home Assistant

通过 MQTT 自动发现自动创建实体：**Reps**（次数）、**Set**（组数）、**Exercise**（动作）、**Stage**（阶段）、**Athlete Present**（有人）、**Workout Complete**（训练完成）。不需要写 YAML，在 Home Assistant 里打开 MQTT 发现即可看到设备。

## 视频

- RTSP：`rtsp://<设备IP>:8554/live0` —— 干净画面，不带任何叠加。
- Console 预览：`ws://<设备IP>:8001/`（H.264）与 `ws://<设备IP>:8001/results`（JSON）。叠加层会在人身上画一个框，标注 `squat  7/12  set 2/3  down`。
- 快照：`http://<设备IP>:8001/snapshot.jpg`。

## 相机摆位

不同动作对取景的要求不一样：

- **深蹲** —— 侧面或 45° 拍全身。**踝关节必须入镜**；在膝盖处截断的画面拿不到读数，状态会一直是 `out of frame`。
- **俯卧撑** —— 贴近地面高度的侧面视角。正对着拍的话肘角几乎不变，次数会漏计。
- **哑铃弯举** —— 正面或侧面，腰以上入镜。想两只手都计数，就要让双肩、双肘、双腕都在画面里。

## 关于精度，实话实说

计数器读的是量化模型的关键点，所以做了相应的平滑与去抖：快于 0.4–0.5 秒的"一次"会被当作抖动丢弃，角度在迟滞区间里来回晃也不会计数。代价是真正爆发式的快速组会被低估；换来的是不会把一次抽搐算成一次。

**次数在"起来"的时候计**，也就是动作完成时，而不是在最低点。所以蹲下去不起来算 0 次而不是 1 次；蹲得浅但完整地做完，会计数并附带 `form_warning`。

动作纠正的范围是刻意收窄的：深蹲和俯卧撑只看幅度，弯举只看肘部外翻。它不是康复师，不检查膝盖内扣、脊柱角度或节奏。

## 没有做的部分

- **隐私打码**。刻意没接：这个应用的全部意义就是看清画面里的人。console 里的设备级打码开关对本应用无效，所以本应用页面上也不提供这个开关。
- **ONVIF**。没有 WS-Discovery，也不发分析元数据：VMS 不会自动发现这台相机，但手动填 RTSP 地址可以拉流。计数只走 MQTT。
