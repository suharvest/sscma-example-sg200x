<!-- app: ppocr-reader | version: 0.4.0 | doc-format: recamera-integration/v1 -->

# PP-OCR Text Reader — Integration Guide

## Overview

Scene text reading running fully on-device (reCamera SG2002, RISC-V + TPU) with a fixed two-model PP-OCRv3 pipeline:

1. **Text detection** — DBNet (MobileNetV3 + RSE-FPN head, 480x480 input, mixed INT8/BF16 quantization) finds text regions as 4-point polygons.
2. **Text recognition** — each region is cropped, perspective-corrected and fed to SVTR-LCNet (48x320 input, BF16) with CTC decoding against a 6623-character dictionary (Simplified Chinese, English, digits, punctuation).

Regions are ordered top-to-bottom, left-to-right; at most `--kmax` regions (default 5) are recognized per frame. Camera frames are captured at 640x480; end-to-end OCR runs at roughly 1–3 fps (detection ~65 ms, plus ~50 ms per recognized region), while the RTSP video stays real-time.

Outputs:

| Channel | Transport | Purpose |
|---|---|---|
| RTSP | `rtsp://<device-ip>:8554/live0` | H.264 video for NVR / VMS / VLC |
| MQTT | `<device-ip>:1883`, topic below | Recognized texts + polygons per frame (JSON) |
| Debug WebSocket | `ws://<device-ip>:8001/` and `ws://<device-ip>:8001/results` | Browser live debug (console UI) |

## RTSP Output

- **URL**: `rtsp://<device-ip>:8554/live0`
- **Codec**: H.264, 640x480 @ 15 fps (defaults)
- Disable with the `--no-rtsp` launch flag.

## MQTT Output

### Connection

- **Broker**: mosquitto on the device, port **1883** (plain TCP, no TLS). By default mosquitto only listens on `localhost`; to consume from another host enable an external listener (`listener 1883 0.0.0.0`, `allow_anonymous true` in `/etc/mosquitto/mosquitto.conf`) or bridge to your own broker.
- **Topic**: `recamera/ppocr/texts` (manifest default; `/etc/ppocr-reader.conf` `MQTT_TOPIC` overrides). **Verify on the console Live page.**
- **QoS**: 0, **retain**: false, **client ID**: `recamera-ppocr-reader`
- One message per processed frame (also when no text is found — `texts` is then `[]`).

### Payload

Top-level fields:

| Field | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | integer | Unix epoch ms | Frame capture/publish time |
| `frame_id` | integer | — | Monotonic frame counter (resets on app start) |
| `inference_time_ms.detection` | number | ms | Detection inference time |
| `inference_time_ms.recognition` | number | ms | Recognition time (sum over all regions) |
| `inference_time_ms.total` | number | ms | Total pipeline time |
| `text_count` | integer | — | Length of `texts[]` |
| `frame_width` / `frame_height` | integer | pixels | Inference frame size the `box` coordinates are normalized against (default 640x480) |
| `texts` | array | — | One object per detected text region |

Each element of `texts[]`:

| Field | Type | Unit / Range | Description |
|---|---|---|---|
| `id` | integer | — | Index of the region within this frame |
| `box` | array | normalized 0–1 | 4-point polygon `[[x,y],[x,y],[x,y],[x,y]]`, clockwise from top-left, relative to `frame_width`x`frame_height` |
| `text` | string | UTF-8 | Recognized text (empty string if the region was not recognized, e.g. beyond `--kmax`) |
| `confidence` | number | 0–1 | Recognition (CTC) confidence |
| `det_confidence` | number | 0–1 | Detection confidence |

Example:

```json
{
  "timestamp": 1768969602957,
  "frame_id": 42,
  "inference_time_ms": { "detection": 65.2, "recognition": 48.3, "total": 113.5 },
  "text_count": 2,
  "frame_width": 640,
  "frame_height": 480,
  "texts": [
    {
      "id": 0,
      "box": [[0.0156,0.0417],[0.3125,0.0417],[0.3125,0.1042],[0.0156,0.1042]],
      "text": "Hello World",
      "confidence": 0.95,
      "det_confidence": 0.89
    },
    {
      "id": 1,
      "box": [[0.0156,0.1250],[0.2344,0.1250],[0.2344,0.2083],[0.0156,0.2083]],
      "text": "你好世界",
      "confidence": 0.88,
      "det_confidence": 0.91
    }
  ]
}
```

### Coordinate convention (MQTT)

`box` is a **4-point polygon, normalized to [0, 1]** relative to the inference frame (`frame_width` x `frame_height`, default 640x480). To draw on the RTSP stream (same 640x480 by default):

```text
px = point[0] * frame_width
py = point[1] * frame_height
```

Note this differs from the debug WebSocket channel below (axis-aligned, center-based pixel boxes).

## Debug WebSocket (browser live view)

Intended for the reCamera console's Live page; any WebSocket client can use it. Lazy: nothing is copied while no client is connected. Client limit: 2 per path. Disable with `--no-debug`, change the port with `--debug-port` (default 8001).

### `ws://<device-ip>:8001/` — H.264 video

Binary messages: one H.264 Annex-B access unit plus a trailing **8-byte little-endian `uint64`** Unix-ms timestamp (strip the last 8 bytes before decoding). Starts with SPS/PPS + IDR on connect.

### `ws://<device-ip>:8001/results` — inference results (JSON text messages)

sscma-node compatible format, one message per processed frame:

```json
{
  "timestamp": 1768969602957,
  "frame_id": 42,
  "inference_time_ms": 113.5,
  "resolution": [640, 480],
  "boxes": [[105.0, 35.0, 190.0, 30.0, 0.890, "Hello World"]],
  "labels": ["Hello World"],
  "texts": ["Hello World"]
}
```

- `boxes` entries are `[cx, cy, w, h, score, label]`: the OCR polygon is reduced to its **axis-aligned bounding rectangle**, `cx, cy` is the box center in pixels of the inference resolution (`resolution`), `w, h` are pixel sizes, `score` is the detection confidence (0–1), and the 6th element is the recognized text truncated to 32 bytes (`"text"` placeholder when the region was not recognized). The console overlay renders it verbatim.
- `labels[i]` mirrors `boxes[i][5]` for programmatic consumers.
- `texts` carries the **full, untruncated** recognized strings (same order as `boxes`).
- `inference_time_ms` is the total pipeline time (detection + recognition).

## Pipeline models & dictionary

Fixed two-model pipeline (no switchable `models[]`; the manifest lists them under `pipeline[]`). Deploy all three files before first start:

| Component | Task | Device path |
|---|---|---|
| `ppocr-det` | text-detect | `/userdata/local/models/ppocr_det_cv181x_mix.cvimodel` |
| `ppocr-rec` | text-recognize | `/userdata/local/models/ppocr_rec_cv181x_bf16.cvimodel` |
| dictionary | CTC decode | `/userdata/local/dict/ppocr_keys_v1.txt` |

Recommended conversion artifacts (from `model_conversion/recamera_ppocr`): the **mixed-precision** detection model (`ppocr_det_cv181x_mix.cvimodel` — sigmoid/attention layers in BF16, rest INT8) and the **BF16** recognition model (`ppocr_rec_cv181x_bf16.cvimodel`). Pure INT8 variants exist but reduce recognition accuracy; if you deploy them, override the paths in `/etc/ppocr-reader.conf`.

An English-only recognizer is also available: `ppocr_rec_en_cv181x_bf16.cvimodel` + `en_dict.txt` (set `REC_MODEL` / `DICT_FILE` in `/etc/ppocr-reader.conf`).

## Configuration

Runtime tuning via `/etc/ppocr-reader.conf` (sourced by the init script `/etc/init.d/K92ppocr-reader`):

| Variable | Default | Description |
|---|---|---|
| `DET_MODEL` | `/userdata/local/models/ppocr_det_cv181x_mix.cvimodel` | Detection model path |
| `REC_MODEL` | `/userdata/local/models/ppocr_rec_cv181x_bf16.cvimodel` | Recognition model path |
| `DICT_FILE` | `/userdata/local/dict/ppocr_keys_v1.txt` | Character dictionary |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_TOPIC` | `localhost` / `1883` / `recamera/ppocr/texts` | MQTT broker settings |
| `DEBUG_ENABLED` / `DEBUG_PORT` | `1` / `8001` | Debug WebSocket stream |
| `KMAX` | 5 | Max regions recognized per frame (0 = unlimited) |
| `ENHANCE_MODE` | `none` | Crop enhancement: `none`, `clahe`, `gray`, `adaptive` |
| `VERBOSE` | 0 | Verbose logging |

There is no console-managed `config_schema` or model switching in this version.

## Quick start (integrator checklist)

1. Copy the two models and the dictionary to the device paths listed above.
2. Activate the app from the reCamera console (Applications page) — only one app owns the camera at a time.
3. Point your NVR/VLC at `rtsp://<device-ip>:8554/live0`.
4. Subscribe: `mosquitto_sub -h <device-ip> -t "recamera/ppocr/texts" -v` (requires the external-listener broker config above).
5. Consume `texts[]` in your application: match against expected strings (labels, meter readings, license plates), filter by `confidence`, and use `box` polygons for localization.
