<!-- app: retail-vision | version: 1.1.0 | doc-format: recamera-integration/v1 -->

# Retail People Counting — Integration Guide

## Overview

**Typical uses**: store entry/exit counting for staffing decisions, dwell hot-spot discovery for display placement, assistance-needed detection for on-floor service.

People flow analytics running fully on-device (reCamera SG2002, RISC-V + TPU). A YOLO11n detector finds persons, a tracker follows them across frames, and every person is classified into a dwell state:

| State | Meaning |
|---|---|
| `transient` | Passing through, not stopped |
| `dwelling` | Just stopped (below the engaged threshold, default < 1.5 s) |
| `engaged` | Stationary 1.5–20 s (looking at a shelf / display) |
| `assistance` | Stationary > 20 s (may need staff assistance) |

The whole camera frame is the analysis zone. Rolling zone metrics (default 60 s window) plus per-person data are published once per inference frame.

Outputs:

| Channel | Transport | Purpose |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video for NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`, topic below | Zone metrics + per-person analytics (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Browser live debug (console UI) |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264, 1280x720 @ 15 fps (defaults; configurable at launch)
- Optional RTSP auth via `--rtsp-user` / `--rtsp-pass` launch flags.

## MQTT Output

### Connection

- **Broker**: mosquitto on the device, port **1883** (plain TCP, no TLS). By default mosquitto only listens on `localhost`; to consume from another host enable an external listener (`listener 1883 0.0.0.0`, `allow_anonymous true` in `/etc/mosquitto/mosquitto.conf`) or bridge to your own broker.
- **Topic**: `recamera/retail-vision/vision` (manifest default; `/etc/retail-vision.conf` `MQTT_TOPIC` overrides). **Verify on the console Live page.**
- **QoS**: 0, **retain**: false, **client ID**: `recamera-retail-vision`
- One message per inference frame (640x640 @ up to ~10–15 fps).

### Payload

Top-level fields:

| Field | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | integer | Unix epoch ms | Frame capture/publish time |
| `frame_id` | integer | — | Monotonic frame counter (resets on app start) |
| `frame_width` / `frame_height` | integer | pixels | Display (RTSP) resolution the `bbox` coordinates refer to (default 1280x720) |
| `fps` | number | frames/s | Current processing rate |
| `inference_time_ms` | number | ms | Detection inference time |
| `zone` | object | — | Rolling-window zone metrics (see below) |
| `persons` | array | — | One object per currently tracked person |

`zone` object (rolling window, default 60 s):

| Field | Type | Description |
|---|---|---|
| `occupancy_count` | integer | Smoothed number of persons currently in frame |
| `browsing_count` | integer | Persons in `transient`/`dwelling` state |
| `engaged_count` | integer | Persons in `engaged` state |
| `assist_count` | integer | Persons in `assistance` state |
| `peak_customer` | integer | Peak occupancy within the window |
| `avg_dwell_time` | number (s) | Average total stationary time of tracks that ended within the window |
| `avg_engagement_time` | number (s) | Average time in `engaged`+ state of ended tracks |
| `avg_velocity` | number (m/s) | Average movement speed of ended tracks |
| `entry_count` / `exit_count` | integer | Cumulative entries/exits since app start (reset on restart) |

Each element of `persons[]`:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `track_id` | integer | — | Stable person track ID |
| `confidence` | number | 0–1 | Detection score |
| `bbox.x` / `bbox.y` | number | normalized 0–1 | **Top-left corner** of the box, relative to `frame_width`x`frame_height` (letterbox already corrected) |
| `bbox.w` / `bbox.h` | number | normalized 0–1 | Box size, same reference |
| `velocity.vx` / `velocity.vy` | number | normalized/s | Velocity components |
| `velocity.speed_m_s` | number | m/s | Estimated real-world speed (assumes 1.7 m person height) |
| `state` | string | see Overview | Dwell state |
| `dwell_duration` | number | seconds | Time stationary in the current dwell episode |

Example:

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "frame_width": 1280,
  "frame_height": 720,
  "fps": 9.8,
  "inference_time_ms": 71.0,
  "zone": {
    "occupancy_count": 2, "browsing_count": 1, "engaged_count": 1,
    "assist_count": 0, "peak_customer": 3, "avg_dwell_time": 4.2,
    "avg_engagement_time": 2.1, "avg_velocity": 0.42,
    "entry_count": 15, "exit_count": 13
  },
  "persons": [
    {
      "track_id": 12,
      "confidence": 0.87,
      "bbox": { "x": 0.4413, "y": 0.375, "w": 0.142, "h": 0.4522 },
      "velocity": { "vx": 0.01, "vy": 0.0, "speed_m_s": 0.05 },
      "state": "engaged",
      "dwell_duration": 4.7
    }
  ]
}
```

### Coordinate convention (MQTT)

`bbox` is **top-left based and normalized to [0, 1]** relative to the *display* resolution (`frame_width` x `frame_height`, default 1280x720) — letterbox distortion between the 16:9 stream and the square model input is already corrected, so the boxes can be drawn directly on the RTSP stream:

```text
px_left = bbox.x * frame_width
px_top  = bbox.y * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

Note this differs from the debug WebSocket channel below (center-based, inference-resolution pixels).

## Debug WebSocket (browser live view)

Intended for the reCamera console's Live page; any WebSocket client can use it. Lazy: nothing is copied while no client is connected. Client limit: 2 per path. Disable with `--no-debug`, change the port with `--debug-port` (default 8001).

### `ws://<device-ip>:8001/` — H.264 video

Binary messages: one H.264 Annex-B access unit plus a trailing **8-byte little-endian `uint64`** Unix-ms timestamp (strip the last 8 bytes before decoding). Starts with SPS/PPS + IDR on connect.

### `ws://<device-ip>:8001/results` — inference results (JSON text messages)

sscma-node compatible format, one message per inference frame:

```json
{
  "timestamp": 1720771201456,
  "frame_id": 3021,
  "inference_time_ms": 71.0,
  "resolution": [640, 640],
  "boxes": [[320.0, 400.5, 90.9, 289.4, 0.874, "T12 engaged"]],
  "labels": ["T12 engaged"],
  "zone": { "occupancy": 1, "browsing": 0, "engaged": 1, "assistance": 0, "entry": 15, "exit": 13 }
}
```

- `boxes` entries are `[cx, cy, w, h, score, label]`: `cx, cy` is the **box center in pixels** of the inference resolution (`resolution`), `w, h` are pixel sizes, `score` is 0–1, and the 6th element is the display label string `"T<track_id> <state>"` (rendered directly by the console overlay).
- `labels[i]` mirrors `boxes[i][5]` for programmatic consumers.
- `zone` is a compact copy of the MQTT zone counters.

## Model

| Name | Task | File |
|---|---|---|
| `yolo11n` (default) | detect | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` |

The model path can be overridden by writing an absolute path to the first line of `/userdata/local/apps/retail-vision.model` (console-managed); delete the file to fall back to the default.

## Configuration

Runtime tuning is done via launch flags in `/etc/retail-vision.conf` (sourced by the init script `/etc/init.d/K92retail-vision`):

| Flag | Default | Description |
|---|---|---|
| `--conf-threshold` | 0.5 | Detection confidence threshold |
| `--dwell-engaged` | 1.5 s | Stationary time before `engaged` |
| `--dwell-assist` | 20 s | Stationary time before `assistance` |
| `--dwell-speed` | 10 px/s | Speed below which a person counts as stationary |
| `--window-duration` | 60 s | Rolling window for zone metrics |
| `--person-height` | 1.7 m | Assumed person height for m/s estimation |

There is no console-managed `config_schema` in this version; the analysis zone is always the full frame.

## Quick start (integrator checklist)

1. Activate the app from the reCamera console (Applications page) — only one app owns the camera at a time.
2. Point your NVR/VLC at `rtsp://<device-ip>:8554/live0`.
3. Subscribe: `mosquitto_sub -h <device-ip> -t "recamera/retail-vision/vision" -v` (requires the external-listener broker config above).
4. Feed `zone` counters into your BI dashboard; use `persons[]` for per-shopper heatmaps or staff-assist alerts (`state == "assistance"`).
