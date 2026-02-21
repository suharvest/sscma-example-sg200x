# Face Analysis Debugger

A web-based debugging tool for the ReCamera Face Analysis application. This tool runs locally on your computer (not on the ReCamera device) and connects to the device over the network.

## Features

- Real-time video stream display with face detection overlays
- MQTT subscription for inference results
- Face attribute display (age, gender, emotion)
- Performance statistics (FPS, inference time)
- Connection status monitoring

## Prerequisites

- ReCamera running the face-analysis application
- MQTT broker with WebSocket support (e.g., Mosquitto with WebSocket listener)
- Video streaming proxy (recommended: MediaMTX for WebRTC/WHEP support)

## Setup

### 1. MQTT WebSocket Configuration

The ReCamera's Mosquitto broker needs WebSocket support. Add to `/etc/mosquitto/mosquitto.conf`:

```conf
listener 1883
listener 9001
protocol websockets
```

### 2. Video Streaming

The RTSP stream needs to be converted to a web-compatible format. Options:

**Option A: MediaMTX (Recommended)**
```bash
# On a server or your local machine
docker run --rm -p 8889:8889 -p 8890:8890 aler9/mediamtx
```

Configure MediaMTX to pull from ReCamera:
```yaml
paths:
  live:
    source: rtsp://192.168.42.1:554/live
```

Then use `http://localhost:8889/live/whep` as the video URL.

**Option B: go2rtc**
```bash
docker run --rm -p 1984:1984 alexxit/go2rtc
```

### 3. Run the Debugger

Simply open `index.html` in a web browser:

```bash
# On macOS
open index.html

# On Linux
xdg-open index.html

# Or use a local HTTP server
python3 -m http.server 8080
# Then open http://localhost:8080
```

## Usage

1. **MQTT WebSocket URL**: Enter the WebSocket URL for the MQTT broker
   - Default: `ws://192.168.42.1:9001`

2. **MQTT Topic**: The topic where face analysis results are published
   - Default: `recamera/face-analysis/results`

3. **RTSP URL**: The URL for the video stream proxy
   - For MediaMTX WHEP: `http://192.168.42.1:8889/live/whep`
   - For HLS: `http://host:port/live.m3u8`

4. Click **Connect** to start receiving data

## MQTT Message Format

The face-analysis application publishes JSON messages:

```json
{
  "timestamp": 1703750400000,
  "frame_id": 123,
  "inference_time_ms": 45.5,
  "face_count": 2,
  "faces": [
    {
      "id": 0,
      "bbox": {"x": 0.1, "y": 0.2, "w": 0.15, "h": 0.2},
      "confidence": 0.95,
      "age": 28,
      "age_confidence": 0.85,
      "gender": "male",
      "gender_confidence": 0.92,
      "emotion": "happiness",
      "emotion_confidence": 0.78,
      "emotion_probs": {
        "neutral": 0.1,
        "happiness": 0.78,
        "surprise": 0.05,
        "sadness": 0.02,
        "anger": 0.01,
        "disgust": 0.01,
        "fear": 0.02,
        "contempt": 0.01
      }
    }
  ]
}
```

## Troubleshooting

### MQTT not connecting
- Check if the WebSocket listener is enabled on Mosquitto
- Verify the port number (default: 9001 for WebSocket)
- Check firewall settings

### Video not showing
- Ensure the RTSP stream is active: `ffplay rtsp://192.168.42.1:554/live`
- Check if the video proxy is running
- Try a different browser (Chrome recommended)

### Overlays not aligned with faces
- This can happen if the video aspect ratio doesn't match
- The overlay coordinates are normalized (0-1), ensure video is displaying correctly
