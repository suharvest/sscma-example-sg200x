# Microduck 关节姿态质检 — 集成说明

Microduck 关节姿态质检应用在 reCamera SG2002 TPU 上运行专用的 17 关键点轻量姿态估计模型。在 384×384 分辨率下实时提取 14 个舵机驱动关节，自动化执行 Microduck 整机制造测试规范中的 T-16 首件与联测质检，识别 14.4° 舵机花键错齿并校验左右腿动作对称性。

## 17 关键点拓扑映射

| 序号 | 标识 | 物理映射 |
|---:|---|---|
| 0 | `snout` | 嘴尖 |
| 1 | `head_top` | 头顶 / 头翻滚轴 |
| 2 | `neck` | 颈部 / 头偏航轴 |
| 3 | `body_center` | 机身中心 / 颈俯仰轴 |
| 4 | `tail` | 鸭尾 |
| 5 | `fl_shoulder` | 左前肩 / 左髋偏航 |
| 6 | `fl_knee` | 左前膝 / 左膝关节 |
| 7 | `fl_foot` | 左前脚掌 / 左脚底板 |
| 8 | `fr_shoulder` | 右前肩 / 右髋偏航 |
| 9 | `fr_knee` | 右前膝 / 右膝关节 |
| 10 | `fr_foot` | 右前脚掌 / 右脚底板 |
| 11 | `hl_hip` | 左后髋 |
| 12 | `hl_knee` | 左后膝 |
| 13 | `hl_foot` | 左后足 |
| 14 | `hr_hip` | 右后髋 |
| 15 | `hr_knee` | 右后膝 |
| 16 | `hr_foot` | 右后足 |

## 核心算法判据

1. **花键错齿检测（14.4° 齿差）**：
   - Microduck XL330 舵机输出轴为 25T 花键，每错装 1 格对应 $360^\circ / 25 = 14.4^\circ$ 偏差；
   - 提取大腿与小腿向量夹角 $\alpha_{knee}$，若与标称值偏差 $|\alpha_{knee} - \alpha_{nominal}| \ge 7.0^\circ$ 触发报警。
2. **左右腿运动对称性**：
   - 在坐立与深蹲过程中，动态监测双膝高度差 $\Delta h(t) = |y_{P8}(t) - y_{P14}(t)|$；
   - 偏差超过 12 像素判定单侧传动阻滞或力矩衰减。
3. **回位漂移容差**：
   - 比对动作执行前后的站立姿态坐标差；
   - 漂移超过 3.0mm 判定固定螺丝松动或结构形变。

## 服务与接口

- **配置文件**：`/etc/microduck-pose-qa.conf`
- **初始化脚本**：`/etc/init.d/K92microduck-pose-qa`（K 前缀，由 supervisor 统一调度，开机不自启）
- **模型文件**：`/userdata/local/models/microduck_pose_17kpt_384_bf16.cvimodel`
- **RTSP 取流**：`rtsp://<设备IP>:8554/live0`
- **WebSocket 预览**：`ws://<设备IP>:8001/` 与 `ws://<设备IP>:8001/results`
- **MQTT 结果上报**：`recamera/microduck-pose-qa/results` 与 `factory/microduck/qa/alarm`

## 站姿标准值标定与单灯判定（1.0.8，灯位表 1.0.9）

站姿标准值（两侧大腿二维角度、两者夹角、躯干倾角）随相机位置变化。守护进程从 `/etc/microduck-pose-qa.conf` 的 `NOMINAL_STAND_*` 读取（该文件登记为 conffile，升级时保留），通过 `POST :8082/api/control` 修改：

| 方法 | 参数 | 结果 |
|---|---|---|
| `calibrate_stand` | `timeout_s`（1-10，默认 10）、`persist`（默认 true） | 对金样采一个 30 帧站姿窗口，取中位数作为标准值并写回配置文件；有效帧少于 15 时返回 422 |
| `set_stand_nominal` | `nominal_stand_theta_l_deg`、`nominal_stand_theta_r_deg`、`nominal_stand_disparity_deg`、`nominal_stand_spine_tilt_deg` 任选；`persist` | 直接设置 |
| `get_config` | - | 当前标准值、容差、脸部灯位布局 |

`start_action` 的 `action_id: "face_led"` 接受 `params.expected`：`{"led": "face:gr2:red", "rgb": [255, 0, 0]}` 或 `{"flashlight": "red|green|blue|white"}`。灯位表以脸部为参照系保存：每个灯位在镜头圆环坐标系下的 (u, v)（圆心为原点、半径为单位长度），存于配置文件的 `FACE_LED_LAYOUT`，由 `set_face_led_layout`（{"layout": {灯位: [u, v]}, "persist": true}）写入。每帧先找镜头圆环，再用圆环和灯点做最小二乘相似变换（缩放、旋转、平移），期望灯位须出现期望颜色，其他灯位不得点亮；找不到圆环或残差大于 0.35 个圆环半径时判 `FAIL_FACE_NOT_REGISTERED`。`get_last_report.registration` 给出逐帧的缩放、旋转和残差（1.0.9）。`get_last_report` 返回 `led_results`（12 项：`expected_on`、`observed_on`、`on_frames`、`ok`）。默认布局取自仿真工位暂定布局的投影，接真机脸板和相机后需重新标定。

