<!-- app: region-motion | version: 0.1.0 | doc-format: recamera-integration/v1 -->

# Region Motion Detection — Integration Guide

## Overview

Monitors user-defined polygon regions for motion on CPU without NPU models. Uses OpenCV background subtraction and contour analysis, rendering real-time overlay in the web console and supporting native Home Assistant MQTT auto-discovery.

Outputs:

| Channel | Transport | Purpose |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video stream |
| MQTT | `<device-ip>:1883`, topic below | Motion status and ratio (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Browser live video and overlay |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264, 1280x720 @ 15 fps

## MQTT Output

### Connection

- **Broker**: mosquitto on device, port 1883 (or external broker configured via `/userdata/local/ha.conf`)
- **Topic**: `recamera/region-motion/results`

### Home Assistant Auto-Discovery

Automatically registers a `BinarySensor` entity:
- **Object ID**: `motion`
- **Device Class**: `motion`
- **State Topic**: `recamera/region-motion/results`
- **Value Template**: `{{ value_json.state }}`

### Payload

```json
{
  "state": "ON",
  "motion": true,
  "ratio": 0.038,
  "frame_id": 1024
}
```

## Configuration

Settings are stored in `/userdata/local/apps/region-motion.config.json` and configurable directly via the console Applications page:

- `pixel_threshold`: Pixel sensitivity (5..100, default: 25)
- `min_motion_area`: Minimum motion area ratio to trigger alarm (0.001..0.5, default: 0.015)
- `debounce_frames`: Debounce frame count (1..10, default: 3)
- `roi_zone`: Custom polygon coordinates `[[x, y], ...]`
