<!-- app: yolo-detector | version: 1.0.0 | doc-format: recamera-integration/v1 -->

# Object Detection (YOLO) — Integration Guide

## Overview

General-purpose YOLO object detection (80 COCO classes) running fully on-device (reCamera SG2002, RISC-V + TPU). Supports switchable models (YOLO11 / YOLO26 family, detect or pose task). By default it also runs a **person tracker** that adds per-person dwell-state analytics on top of raw detections.

Outputs:

| Channel | Transport | Purpose |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video for NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`, topic see below | Structured detection/tracking results (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Browser live debug (console UI) |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264 (Annex-B), 1280x720 @ 15 fps (defaults; configurable at launch)
- One RTSP consumer at a time is recommended; the camera pipeline is owned exclusively by this application.

## MQTT Output

### Connection

- **Broker**: mosquitto running on the device, port **1883** (plain TCP, no TLS).
  By default mosquitto only listens on `localhost`; to consume results from another host, enable an external listener in `/etc/mosquitto/mosquitto.conf` (`listener 1883 0.0.0.0`, `allow_anonymous true`) or bridge to your own broker.
- **Topic**: as declared by the app manifest and shown on the console Live page — `recamera/yolo-detector/results` in console-managed deployments. The binary's built-in default is `recamera/yolo/detections` (override with `--mqtt-topic`). **Always verify the topic on the console Live page.**
- **QoS**: 0, **retain**: false
- **Client ID** (publisher): `recamera-yolo-detector`
- One message per inference frame (default: 640x640 @ up to 15 fps).

### Payload format A — tracking mode (default)

Person tracking is **enabled by default** (`--no-tracking` disables it). In this mode only tracked **persons** (COCO class `person`) are published, enriched with motion/dwell analytics:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `timestamp` | integer | Unix epoch **milliseconds** | Capture/publish time of the frame |
| `frame_id` | integer | — | Monotonic frame counter (starts at 0 on app start) |
| `inference_time_ms` | number | milliseconds | Detection inference time for this frame |
| `zone_occupancy.total` | integer | — | Total tracked persons in frame |
| `zone_occupancy.browsing` | integer | — | Persons in `transient` or `dwelling` state |
| `zone_occupancy.engaged` | integer | — | Persons in `engaged` state |
| `zone_occupancy.assistance` | integer | — | Persons in `assistance` state |
| `persons` | array | — | One object per tracked person (may be empty) |

Each element of `persons[]`:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `track_id` | integer | — | Stable person track ID (survives across frames) |
| `confidence` | number | 0–1 | Detection score of the underlying box |
| `bbox.x` | number | normalized 0–1 | **Box center** X (fraction of frame width) |
| `bbox.y` | number | normalized 0–1 | **Box center** Y (fraction of frame height) |
| `bbox.w` | number | normalized 0–1 | Box width (fraction of frame width) |
| `bbox.h` | number | normalized 0–1 | Box height (fraction of frame height) |
| `speed_px_s` | number | pixels/second | Movement speed in inference-frame pixels (640x640 by default) |
| `speed_normalized` | number | %/second | Speed as percent of the person's body height per second |
| `state` | string | see below | Dwell state: `transient` (passing by), `dwelling` (stopped < 1.5 s), `engaged` (stopped 1.5–20 s), `assistance` (stopped > 20 s) |
| `dwell_duration_sec` | number | seconds | Time spent stationary in the current dwell episode |

Example:

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "inference_time_ms": 71.0,
  "zone_occupancy": { "total": 2, "browsing": 1, "engaged": 1, "assistance": 0 },
  "persons": [
    {
      "track_id": 12,
      "confidence": 0.874,
      "bbox": { "x": 0.5123, "y": 0.6011, "w": 0.1420, "h": 0.4522 },
      "speed_px_s": 3.4,
      "speed_normalized": 1.2,
      "state": "engaged",
      "dwell_duration_sec": 4.7
    },
    {
      "track_id": 15,
      "confidence": 0.791,
      "bbox": { "x": 0.2210, "y": 0.5533, "w": 0.1101, "h": 0.4123 },
      "speed_px_s": 96.5,
      "speed_normalized": 36.6,
      "state": "transient",
      "dwell_duration_sec": 0.0
    }
  ]
}
```

### Payload format B — raw detections (`--no-tracking`)

All detected classes are published without tracking analytics:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `timestamp` | integer | Unix epoch **milliseconds** | Capture/publish time of the frame |
| `frame_id` | integer | — | Monotonic frame counter |
| `inference_time_ms` | number | milliseconds | Inference time for this frame |
| `detection_count` | integer | — | Number of entries in `detections` |
| `detections` | array | — | One object per detection |

Each element of `detections[]`:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `id` | integer | — | Detection ID |
| `class_id` | integer | 0–79 | COCO class index (0 = person) |
| `class_name` | string | — | COCO class name (`person`, `bicycle`, `car`, ...) |
| `confidence` | number | 0–1 | Detection score |
| `bbox.x` / `bbox.y` | number | normalized 0–1 | **Box center** (fraction of frame width/height) |
| `bbox.w` / `bbox.h` | number | normalized 0–1 | Box width/height (fraction of frame) |

Example:

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "inference_time_ms": 68.0,
  "detection_count": 1,
  "detections": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.874,
      "bbox": { "x": 0.5123, "y": 0.6011, "w": 0.1420, "h": 0.4522 }
    }
  ]
}
```

### Coordinate convention (MQTT)

`bbox` is **center-based and normalized to [0, 1]** relative to the inference frame (default 640x640). To convert to top-left pixel coordinates on any rendition of the stream (e.g. the 1280x720 RTSP output):

```text
px_left = (bbox.x - bbox.w / 2) * frame_width
px_top  = (bbox.y - bbox.h / 2) * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

## Debug WebSocket (browser live view)

Intended for the reCamera console's Live page; any WebSocket client can use it. Lazy: no video is copied while no client is connected. **Client limit: 2 per path.**

### `ws://<device-ip>:8001/` — H.264 video

- Binary messages. Each message is one H.264 access unit in **Annex-B** byte-stream format, followed by a trailing **8-byte little-endian `uint64`** Unix timestamp in **milliseconds**:

```text
[ Annex-B H.264 bytes ...... ][ uint64 unix_ms, little-endian ]
                                ^ last 8 bytes of the message
```

- On connect, the stream starts with SPS + PPS followed by an IDR frame, so decoders (e.g. JMuxer) can start immediately.

### `ws://<device-ip>:8001/results` — inference results (JSON, text messages)

sscma-node compatible format, one message per inference frame:

```json
{
  "boxes": [[cx, cy, w, h, score, target]],
  "labels": ["person", "bicycle", "car"],
  "resolution": [640, 640]
}
```

- `boxes` entries are `[cx, cy, w, h, score, target]` where `cx, cy` is the **box center in pixels** of the inference resolution (`resolution`), `w, h` are pixel width/height, `score` is 0–1 (or 0–100), `target` is the class index into `labels`.
- Note the difference vs MQTT: `/results` uses **pixel** coordinates; MQTT `bbox` uses **normalized** coordinates. Both are center-based.

## Models (switchable)

The active model can be switched from the console (Live page → Model). Switching restarts the application; streams are briefly interrupted.

| Name | Task | File |
|---|---|---|
| `yolo11n` (default) | detect | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` |
| `yolo11n-pose` | pose | `/userdata/local/models/yolo11n_pose_cv181x_int8.cvimodel` |
| `yolo26n` | detect | `/userdata/local/models/yolo26n_cv181x_int8.cvimodel` |

Mechanics: the console writes the selected model's absolute path to `/userdata/local/apps/yolo-detector.model` (one line); the init script passes it to the binary via `-m` on the next start. Delete the file to fall back to the built-in default.

Note: the pose model detects persons only; the published payload keeps the same detection/tracking JSON schema (keypoints are not published over MQTT).

## Quick start (integrator checklist)

1. Activate the app from the reCamera console (Applications page) — only one app owns the camera at a time.
2. Point your NVR/VLC at `rtsp://<device-ip>:8554/live0`.
3. Subscribe: `mosquitto_sub -h <device-ip> -t "<topic from console Live page>" -v` (requires the external-listener broker config above).
4. Decide tracking vs raw mode: default is tracking (persons only). Use `--no-tracking` in the init script's options for all-class raw detections.
