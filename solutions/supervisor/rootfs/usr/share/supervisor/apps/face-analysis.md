<!-- app: face-analysis | version: 1.0.0 | doc-format: recamera-integration/v1 -->

# Face Analysis — Integration Guide

## Overview

Real-time face detection with per-face attribute analysis (age, gender, race, emotion) running fully on-device (reCamera SG2002, RISC-V + TPU). The application is a fixed multi-model pipeline:

1. **Face detection** — YOLOv8n-face, run on every inference frame.
2. **Age / gender / race** — FairFace (or InsightFace variant), run per detected face.
3. **Emotion** — HSEmotion (AffectNet, 8 classes), run per detected face.

Outputs:

| Channel | Transport | Purpose |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video for NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`, topic `recamera/face-analysis/results` | Structured analysis results (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Browser live debug (console UI) |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264 (Annex-B), 1280x720 @ 15 fps (defaults; configurable at launch)
- **Privacy**: face regions are **blurred by default** on the RTSP stream (`--no-blur` disables, max 16 regions).
- One RTSP consumer at a time is recommended; the camera pipeline is owned exclusively by this application.

## MQTT Output

### Connection

- **Broker**: mosquitto running on the device, port **1883** (plain TCP, no TLS).
  By default mosquitto only listens on `localhost`; to consume results from another host, enable an external listener in `/etc/mosquitto/mosquitto.conf` (`listener 1883 0.0.0.0`, `allow_anonymous true`) or bridge to your own broker.
- **Topic**: `recamera/face-analysis/results`
- **QoS**: 0, **retain**: false
- **Client ID** (publisher): `recamera-face-analysis`
- One message per analyzed frame (default inference rate: 640x480 @ up to 10 fps).

### JSON schema

Top-level object:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `timestamp` | integer | Unix epoch **milliseconds** | Capture/publish time of the frame |
| `frame_id` | integer | — | Monotonic frame counter (starts at 0 on app start) |
| `inference_time_ms` | number | milliseconds | Total pipeline inference time for this frame |
| `face_count` | integer | — | Number of entries in `faces` |
| `faces` | array | — | One object per detected face (may be empty) |

Each element of `faces[]`:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `id` | integer | — | Face tracking ID (stable across frames while tracked) |
| `bbox.x` | number | normalized 0–1 | **Top-left** X of the face box (fraction of frame width) |
| `bbox.y` | number | normalized 0–1 | **Top-left** Y of the face box (fraction of frame height) |
| `bbox.w` | number | normalized 0–1 | Box width (fraction of frame width) |
| `bbox.h` | number | normalized 0–1 | Box height (fraction of frame height) |
| `confidence` | number | 0–1 | Face detection score |
| `age_bin` | integer | 0–8 | **FairFace model only.** Age bin index |
| `age` | integer | 0–100 years | **InsightFace model only.** Continuous age estimate |
| `age_label` | string | — | Human-readable age: FairFace bin label (`"0-2"`, `"3-9"`, `"10-19"`, `"20-29"`, `"30-39"`, `"40-49"`, `"50-59"`, `"60-69"`, `"70+"`) or InsightFace age as string |
| `age_confidence` | number | 0–1 | Age prediction confidence |
| `gender` | string | `"male"` \| `"female"` | Gender prediction |
| `gender_confidence` | number | 0–1 | Gender prediction confidence |
| `race` | string | see below | **FairFace only** (absent otherwise). One of `White`, `Black`, `Latino_Hispanic`, `East_Asian`, `Southeast_Asian`, `Indian`, `Middle_Eastern` |
| `race_confidence` | number | 0–1 | **FairFace only.** Race prediction confidence |
| `emotion` | string | see below | Dominant emotion: `angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |
| `emotion_confidence` | number | 0–1 | Confidence of the dominant emotion |
| `emotion_probs` | object | 0–1 each | Full 8-class probability map, keys: `angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |

Notes:

- Exactly one of `age_bin` / `age` is present, depending on which attribute model variant is deployed (default deployment ships FairFace, so expect `age_bin`).
- Numbers are serialized with 3 decimal places.

### Coordinate convention (MQTT)

`bbox` is **top-left based and normalized to [0, 1]** relative to the inference frame (default 640x480). To convert to pixels on any rendition of the stream (e.g. the 1280x720 RTSP output):

```text
px_left = bbox.x * frame_width
px_top  = bbox.y * frame_height
px_w    = bbox.w * frame_width
px_h    = bbox.h * frame_height
```

(This differs from the Debug WebSocket `/results` channel, which uses center-based pixel coordinates — see below.)

### Example payload

```json
{
  "timestamp": 1720771200123,
  "frame_id": 1502,
  "inference_time_ms": 84.250,
  "face_count": 1,
  "faces": [
    {
      "id": 7,
      "bbox": { "x": 0.412, "y": 0.238, "w": 0.144, "h": 0.221 },
      "confidence": 0.912,
      "age_bin": 3,
      "age_label": "20-29",
      "age_confidence": 0.671,
      "gender": "female",
      "gender_confidence": 0.983,
      "race": "East_Asian",
      "race_confidence": 0.542,
      "emotion": "happy",
      "emotion_confidence": 0.877,
      "emotion_probs": {
        "angry": 0.004, "contempt": 0.011, "disgust": 0.002, "fear": 0.006,
        "happy": 0.877, "neutral": 0.082, "sad": 0.009, "surprise": 0.009
      }
    }
  ]
}
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
  "labels": ["face"],
  "resolution": [640, 480]
}
```

- `boxes` entries are `[cx, cy, w, h, score, target]` where `cx, cy` is the **box center in pixels** of the inference resolution (`resolution`), `w, h` are pixel width/height, `score` is 0–1 (or 0–100), `target` is the class index into `labels`.
- Note the difference vs MQTT: `/results` is **center-based, pixel** coordinates; MQTT `bbox` is **top-left, normalized**.

## Pipeline (fixed, not switchable)

This application has **no switchable models** (`setModel` is not applicable). It runs a fixed cascade:

| Stage | Model | File | Role |
|---|---|---|---|
| 1 | `yolov8n-face` | `/userdata/local/models/yolov8n_face_cv181x_int8.cvimodel` | Face detection (INT8) — produces `bbox`, `confidence`, tracking `id` |
| 2 | `fairface` | `/userdata/local/models/fairface_int8.cvimodel` | Age bin (9), gender (2), race (7) per face crop (INT8) |
| 3 | `hsemotion` | `/userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel` | Emotion, AffectNet 8-class per face crop (BF16) |

Emotion inference runs every 2nd frame by default; cached results are reused in between.

## Quick start (integrator checklist)

1. Activate the app from the reCamera console (Applications page) — only one app owns the camera at a time.
2. Point your NVR/VLC at `rtsp://<device-ip>:8554/live0`.
3. Subscribe: `mosquitto_sub -h <device-ip> -t "recamera/face-analysis/results" -v` (requires the external-listener broker config above).
4. Parse per the schema; use `timestamp` + `frame_id` to align with video if needed.
