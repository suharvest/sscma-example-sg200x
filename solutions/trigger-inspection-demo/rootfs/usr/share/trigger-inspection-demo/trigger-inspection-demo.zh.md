# 触发式工位质检示例

每次触发抓拍一帧并返回 OK/NG 判定。两次触发之间，相机通道以待机采样率运行（`--idle-fps`，默认 1 fps）。

## 触发方式

| 触发 | 方法 | 响应 |
|---|---|---|
| HTTP | `GET` 或 `POST http://<设备IP>:8090/trigger?tag=<编号>` | 检测完成后返回 `200` 和结果 JSON；已有 HTTP 触发在处理时返回 `409` |
| GPIO | sysfs GPIO 输入上的边沿（配置文件中的 `GPIO_LINE`） | 结果仅通过 MQTT 发布 |

```bash
curl -X POST 'http://<设备IP>:8090/trigger?tag=part-001'
```

## 结果

发布到 `recamera/trigger-inspection-demo/results`，同时作为 HTTP 响应返回：

```json
{"timestamp":1730956800000,"frame_id":12,"verdict":"NG","reason":"low_contrast",
 "trigger":{"source":"http","seq":3,"tag":"part-001"},
 "metrics":{"mean":118.4,"stddev":6.2,"samples":19200},"roi":[160,120,320,240],"resolution":[640,480],
 "timing_us":{"validate":3,"roi":2,"measure":5210,"judge":1,"total":5216}}
```

`verdict` 取值：`OK`；`NG`（某项检查未通过，`reason` 为 `too_dark`、`too_bright` 或 `low_contrast`）；`ERROR`（该帧无法检测）。HTTP 触发后 2 秒内未取到帧时，响应为 `{"accepted":true,"error":"no_frame"}`。

## 配置

`/etc/trigger-inspection-demo.conf`（shell 变量，由 init 脚本读取）：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `HTTP_PORT` | `8090` | HTTP 触发端口，`0` 关闭 |
| `GPIO_LINE` | 未设置 | 触发输入的 sysfs GPIO 编号 |
| `GPIO_EDGE` | `rising` | `rising`、`falling` 或 `both` |
| `GPIO_ACTIVE_LOW` | `0` | 输入低电平有效时设为 `1` |
| `IDLE_FPS` | `1` | 待机时相机通道采样率（1 到 15） |
| `EXTRA_OPTS` | 空 | 其他参数，例如 `--roi 0.3,0.3,0.4,0.4 --mean-min 60 --std-min 20 --rtsp` |
| `MQTT_HOST`、`MQTT_PORT`、`MQTT_TOPIC` | `localhost`、`1883`、见上 | MQTT 服务器与主题 |
| `DEBUG_ENABLED`、`DEBUG_PORT` | `0`、`8001` | 控制台预览（WebSocket 视频 + `/snapshot.jpg`）；设为 `1` 开启 |

完整参数列表：`trigger-inspection-demo --help`。

## 限制

- 示例检测仅为单个区域的亮度与对比度；实际质检需替换 `main.cpp` 中的 `measure` / `judge` 步骤。
- HTTP 触发同一时间只处理一个；GPIO 与 HTTP 触发共用长度为 8 的队列。
- 待机时相机仍每秒拷贝 `IDLE_FPS` 帧；预览流（调试 WebSocket、RTSP）默认关闭；开启后由视频硬件按各自帧率编码。
