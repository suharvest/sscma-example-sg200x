<!-- app: facemesh-reader | version: 0.1.1 | doc-format: recamera-integration/v1 -->

# Drowsiness Detection — Integration Guide

## Overview

**Typical uses**: long-haul cabs, monitoring posts, heavy-machinery operator seats — fatigue shows up minutes before an incident does, and this application's job is to find those minutes.

Runs entirely on the device (reCamera SG2002, RISC-V + TPU). yolov8n-face finds the face, a 468-point FaceMesh traces the eye and mouth contours, and from those it computes, every frame:

| Metric | Meaning |
|---|---|
| `EAR` (Eye Aspect Ratio) | How open the eyes are. Sustained below threshold means closed |
| `MAR` (Mouth Aspect Ratio) | How open the mouth is; used to spot yawns |
| `PERCLOS` | Share of a rolling window spent with eyes closed. The most established objective fatigue measure |

Three independent triggers (sustained closure, PERCLOS over threshold, yawning) each raise the alert; state runs `Alert` → `Drowsy` → `Danger`.

Output channels:

| Channel | Transport | Use |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video for NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`, topic `recamera/facemesh-reader/results` | Per-frame metrics and alert state (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Live browser debugging (console UI) |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264, 1280x720 @ 15 fps (defaults)
- **Privacy**: when device-wide masking is on, every detected face is concealed before the frame is encoded, so the RTSP stream, the console preview and `/snapshot.jpg` all carry the mask. Off by default; the switch and its opacity live on the console's **Device** page (there is a shortcut on the debug page). Applies immediately without restarting the application.
- One RTSP consumer at a time is recommended; the camera pipeline is owned exclusively by this application.

## MQTT Output

### Connection

| Item | Value |
|---|---|
| Broker | `<device-ip>:1883` |
| Topic | `recamera/facemesh-reader/results` (`--mqtt-topic` to change) |
| QoS | 0 |
| Rate | One message per inference frame |

### Message format

```json
{
  "timestamp": 1737964800000,
  "frame_id": 1024,
  "inference_time_ms": 97.0,
  "face_count": 1,
  "faces": [
    {
      "id": 0,
      "bbox": { "x": 0.5341, "y": 0.1938, "w": 0.2184, "h": 0.3782 },
      "confidence": 0.8290,
      "left_ear": 0.1402,
      "right_ear": 0.1385,
      "ear": 0.1394,
      "mar": 0.3120,
      "eyes_closed": true,
      "mouth_open": false,
      "metrics_valid": true,
      "drowsiness": {
        "level": 0.7350,
        "state": "Danger",
        "perclos_pct": 42.8571,
        "continuous_closure_sec": 1.9000,
        "alert_active": true,
        "drowsy_by_ear": true,
        "drowsy_by_perclos": true,
        "drowsy_by_yawn": false
      },
      "yawn": { "is_yawning": false, "yawn_count_5min": 2 }
    }
  ]
}
```

### Field notes

| Field | Notes |
|---|---|
| `drowsiness.state` | `Alert` / `Drowsy` / `Danger` |
| `drowsiness.level` | 0–1, half from EAR and half from PERCLOS |
| `drowsiness.alert_active` | Whether the alert is currently raised. **Drive alerting from this**, not from your own threshold on `level` |
| `drowsiness.drowsy_by_*` | Which of the three triggers fired, so a raised alert can be explained |
| `metrics_valid` | Whether this frame's landmarks were usable. When `false`, the EAR/MAR above mean nothing |
| `yawn.yawn_count_5min` | Yawns in the trailing five minutes |

### Coordinate convention

`bbox` is **normalised and centre-based** (`x`/`y` are the centre, not the top-left corner), 0–1, against the inference frame.

> The inference channel (4:3) and the video stream (16:9) are different shapes, and the camera fits the scene into each preserving aspect. Drawing these coordinates straight onto the 1280x720 video puts them in the wrong place — convert first; `toStream()` in `components/geometry` is exactly that conversion.

### The 468 landmarks

**Not published by default** — nearly a thousand floats per frame would swamp the broker. `--include-landmarks` adds a `landmarks` array per face, each element an `[x, y]` normalised pair.

## Debug WebSocket (live browser preview)

### `ws://<device-ip>:8001/` — H.264 video

Binary frames, Annex-B. This is what the console's debug page previews.

### `ws://<device-ip>:8001/results` — inference results

Text messages, same source as MQTT but with coordinates already converted to stream pixels, ready to overlay on the video.

### `http://<device-ip>:8001/snapshot.jpg`

The current frame as JPEG. **Masked too when masking is on** — this is the URL ONVIF advertises as `GetSnapshotUri`.

## Tunables

Thresholds are launch flags; put them in `/etc/facemesh-reader.conf` to have the service pick them up:

| Flag | Default | Meaning |
|---|---|---|
| `--ear-threshold` | 0.21 | Below this the eye counts as closed |
| `--ear-continuous-sec` | 1.5 | How long closure must persist to count as drowsy |
| `--mar-threshold` | 0.6 | Above this the mouth counts as open / yawning |
| `--perclos-warning` | 15 | PERCLOS warning line (%) |
| `--perclos-critical` | 30 | PERCLOS danger line (%) |
| `--threshold` | 0.5 | Face detection confidence |

> These are a generic starting point, not a calibration. Camera angle and whether the subject wears glasses both shift the absolute EAR, so **watch the `ear` field from the actual mounting position for a while before settling on numbers**.
