# QR Code Reader — Integration Guide

Decodes every QR code in view and publishes the payloads over MQTT. Detection
runs on the CPU with [quirc](https://github.com/dlbeer/quirc) — no model, no
TPU, nothing to download.

Multiple codes in one frame are decoded in the same pass, which is what makes
it usable as a fixed scanner over a conveyor or a counter rather than a
one-code-at-a-time phone app.

## MQTT output

Topic: `recamera/qrcode-reader/results`, one message per processed frame.

```json
{
  "type": "qrcode",
  "frame": 4821,
  "qr_found": true,
  "detect_cost_ms": 34.10,
  "codes": [
    {
      "text": "https://www.seeedstudio.com",
      "points": [[0.31, 0.28], [0.55, 0.29], [0.54, 0.53], [0.30, 0.52]]
    }
  ]
}
```

- `codes[].text` — the decoded payload, verbatim. JSON-escaped, so binary and
  multi-line payloads survive intact.
- `codes[].points` — the four corners, **normalized 0–1 against the video
  stream frame**, in the order quirc returns them (top-left, top-right,
  bottom-right, bottom-left for an upright code). Rotated codes keep their
  winding, so the quad can be drawn directly without sorting.
- `qr_found` — false with an empty `codes` array when nothing decoded. Messages
  are published either way, so a subscriber can tell "nothing in view" from
  "the app stopped".
- `detect_cost_ms` — decode time for this frame, excluding capture.

## Video

- RTSP: `rtsp://<device_ip>:8554/live0` — the clean scene, no overlay.
- Console preview: `ws://<device_ip>:8001/` (H.264) and
  `ws://<device_ip>:8001/results` (JSON). The overlay outlines each decoded
  code and prints its payload.
- Snapshot: `http://<device_ip>:8001/snapshot.jpg`.

## Configuration

`/etc/qrcode-reader.conf`, read by the init script:

| Key | Default | Meaning |
|---|---|---|
| `MQTT_HOST` / `MQTT_PORT` | `localhost` / `1883` | Broker to publish to |
| `MQTT_TOPIC` | `recamera/qrcode-reader/results` | Result topic |
| `RTSP_ENABLED` | `1` | `0` disables the RTSP stream |
| `STREAM_WIDTH` / `STREAM_HEIGHT` / `STREAM_FPS` | `1280` / `720` / `30` | Encoded stream |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` / `CAMERA_FPS` | `1280` / `720` / `10` | Detection frame |

**The detection frame is deliberately large.** Most detectors here run at
640×640; quirc needs enough pixels per QR module, and a code usually fills only
part of the frame. At 640×640 real-world codes — phone screens especially —
came back undecodable. 1280×720 matches the sensor aspect (so the finder
patterns are not squished) and gives ~2.25× the area.

## Camera placement

- Fill more of the frame with the code than feels necessary. A code smaller
  than roughly 1/6 of the frame width is at the edge of what decodes reliably.
- Even, diffuse light. Glare on a phone screen or a laminated label kills the
  finder patterns faster than low light does.
- Codes may be rotated or moderately tilted; steep angles lose the modules on
  the far edge.

## Not included

- **No model.** This is a classical CPU decoder, so nothing needs downloading
  and it does not compete for the TPU.
- **Barcodes (1D) are not read** — QR only.
- **Privacy blur is not wired in**: the frame is being read for content, and
  masking it would defeat the purpose.
