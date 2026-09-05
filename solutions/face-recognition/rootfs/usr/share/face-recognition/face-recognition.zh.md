# 人脸识别

reCamera（SG2002 / CV181x）端侧人脸识别。

## 流水线

| 阶段 | 模型 | 输入 | 输出 |
|---|---|---|---|
| 检测 | SCRFD-500M，INT8 | 640x640 RGB | 人脸框 + 5 点关键点 |
| 对齐 | 相似变换 | 5 点关键点 | 112x112 人脸图 |
| 编码 | MobileFaceNet ArcFace，BF16 | 112x112 RGB | 512 维，L2 归一化 |
| 活体（可选） | MiniFASNet 2.7_80x80，BF16 | 80x80 BGR | P(real) |
| 比对 | 余弦线性扫描 | 512 维 | 姓名 + 分数 |

检测每帧执行。编码与活体在轨迹首次出现时执行，之后每 `EMBED_INTERVAL` 帧执行一次，
其余帧复用缓存结果。连续两次编码得到同一姓名后，该姓名标记为 `stable`。

## 结果主题

`recamera/face-recognition/results`

```json
{"timestamp": 1757000000000, "frame_id": 42, "inference_time_ms": 96.0,
 "model_tag": "cvi:scrfd500m+mbf512@bf16", "face_count": 1,
 "faces": [{"track_id": 3, "bbox": [0.31, 0.22, 0.55, 0.61], "det_score": 0.87,
            "name": "alice", "score": 0.62, "live": null, "liveness_score": null,
            "stable": true, "gated": false, "reason": null}]}
```

`bbox` 是相对推理帧归一化的 `[x1, y1, x2, y2]`。活体阶段未运行时
`live` / `liveness_score` 为 `null`，以便消费方区分"未检测"与"检测未通过"。

## 命令

发布到 `recamera/face-recognition/cmd`，在 `recamera/face-recognition/cmd/ack`
读取应答。

```json
{"op": "enroll", "name": "alice", "source": "camera", "frames": 5, "request_id": "r1"}
{"op": "enroll", "name": "alice", "source": "image", "image_b64": "...", "request_id": "r2"}
{"op": "remove", "name": "alice", "request_id": "r3"}
{"op": "list", "request_id": "r4"}
{"op": "reload", "request_id": "r5"}
```

camera 方式的注册只采纳画面中恰好有一张合格人脸的帧，约 4 秒后超时。每条应答包含
`{"request_id", "op", "ok", "model_tag", "users"}`，`ok` 为 false 时附带 `"err"`。

## 人脸库

`/userdata/local/face-gallery/cvi_scrfd500m_mbf512_bf16.json`

文件名与其中的 `_meta.model_tag` 标明特征由哪组模型产生。不同后端的 ArcFace 权重不同，
在别的加速器上注册的人脸库在这里的余弦接近 0——这不会报错，只会谁都认不出来。
因此 tag 不一致时应用拒绝加载。

## 配置

`/etc/face-recognition.conf`。`MATCH_THRESHOLD`（默认 0.40）是判定为库内身份的余弦阈值；
`MIN_FACE_PX`（默认 64）是短边像素下限，低于该值的人脸以 `gated` 发布且不做编码。

## 设备实测（2026-09-06）

- 仅检测的帧：**61 ms**（int8 输入 SCRFD；F32 输入旧模型为 283 ms）。
- 有一张脸、每 5 帧做一次 embedding：**81–105 ms**。
- 已录入人员的实时识别：score 0.85–0.87，`stable=true`。
- 同一张图设备 BF16 embedding 与 fp32 参考的 cosine：0.99。
- 内存：运行中占用约 27 MB，空闲 >100 MB。详见 `solutions/face-recognition/EVALUATION.md`。
