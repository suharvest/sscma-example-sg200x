# 机械臂视觉伴侣原子服务 (reCamera SG2002 RISC-V)

面向 Seeed reCamera（算能 SG2002 单核 1.0 GHz RISC-V 平台）专为机械臂视觉伺服、手眼标定和工件精准抓取提供的高性能原子视觉能力服务。

## 架构边界原则
**reCamera 只提供底层高性能通用原子能力，不承担复杂的业务编排与重型 Python 逻辑：**
- 复杂的业务决策、多模态 VLM 理解或轨迹编排，交由上位机/云端 Agent 远程调用。
- 业务流程确定后，可直接将固化好的轻量程序运行在工控机或本地客户端上。

## 核心原子能力与接口规范

| 协议 / 端口 | 接口路径 | 模式 | 功能说明 |
|---|---|---|---|
| **HTTP :8090** | `GET /api/inference` | 状态查询 | 返回 `{"enabled":false,"fps":1,"active_fps":25,"idle_fps":1}` |
| **HTTP :8090** | `POST /api/inference` | 状态切换 | 传入 `{"enabled":true}` 或 `{"enabled":false}`，控制待机降频 |
| **HTTP :8090** | `POST /api/trigger` | **单次原子触发** | 触发单帧抓拍解算，返回 6-DoF 位姿 JSON，完成后自动保持待机 |
| **HTTP :8090** | `GET /api/results/latest` | 长轮询获取 | `?after=<id>&wait_ms=1000`，阻塞等待最新帧结果 |
| **HTTP :80** | `GET /snapshot.jpg` | 高清图像快照 | 供远程 VLM / Agent 多模态理解的高清快照抓取 |
| **WS :8001** | `ws://{host}:8001/` | H.264 视频流 | 720P 极低时延视频预览流（前端通过 JMuxer 直接硬解） |
| **WS :8001** | `ws://{host}:8001/results`| 实时坐标流 | 推送中心点像素坐标与 6-DoF 姿态数据包 |
| **MQTT :1883** | `recamera/aruco-detector/results` | 广播上报 | 适配 Home Assistant / 工业 MQTT 消息总线 |

## 单次抓拍返回数据示例 (`POST /api/trigger`)
```json
{
  "success": true,
  "frame_id": 142,
  "ts_ms": 1726001234567,
  "inference_time_ms": 4.1,
  "count": 1,
  "markers": [
    {
      "id": 0,
      "center": [320.5, 180.2],
      "corners": [[300.0, 160.0], [341.0, 160.0], [341.0, 200.4], [300.0, 200.4]],
      "pose": {
        "t": [0.0452, -0.0120, 0.3541],
        "r_deg": [-1.25, 3.40, 45.10],
        "error": 0.0008
      }
    }
  ]
}
```

## 待机低功耗指标
- **待机状态 (Inference Disabled)**：底层相机动态降至 1 FPS，跳过 AprilTag 解算，SG2002 CPU 占用 < 3%，机身常温保持 38℃。
- **触发工作 (Triggered / Enabled)**：瞬间恢复至 25 FPS 工作帧率，单帧解算时间约 4ms。
