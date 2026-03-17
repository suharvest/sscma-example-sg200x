# Retail Vision - People Flow Analytics

Real-time people detection, tracking, and dwell-state analytics on ReCamera (SG2002 RISC-V). Outputs RTSP video stream (with optional authentication) and publishes per-frame analytics via MQTT.

## Features

- **YOLOv11n INT8** person detection (~49ms, ~15 FPS)
- **Two-pass tracker**: IoU matching with velocity prediction + center-distance fallback
- **Dwell state machine**: TRANSIENT → DWELLING → ENGAGED → ASSISTANCE
- **Entry/exit counting** with velocity-aware edge classification
- **Zone metrics**: rolling-window occupancy, peak customer, avg dwell/engagement/velocity
- **Letterbox-corrected coordinates**: absolute pixel coords matching display resolution
- **RTSP** H.264 1280x720 stream with optional Basic auth
- **MQTT** VisionPayload JSON published every frame

## Quick Start

### Prerequisites

- Docker container `ubuntu_dev_x86` running (see top-level CLAUDE.md)
- `sshpass` installed: `brew install hudochenkov/sshpass/sshpass`
- Detection model at `/userdata/local/models/` on device

### Build & Deploy

```bash
cd solutions/retail-vision

# Build + deploy (default device: 192.168.42.1)
./deploy.sh

# Deploy to a specific host
./deploy.sh --host 192.168.10.158 --password recamera.1

# Deploy-only (skip Docker build)
./deploy.sh --skip-build --host 192.168.10.158 --password recamera.1
```

### Deploy Model

```bash
scp yolo11n_cv181x_int8.cvimodel recamera@<device_ip>:/tmp/
ssh recamera@<device_ip> "echo '<password>' | sudo -S mkdir -p /userdata/local/models && sudo mv /tmp/yolo11n_cv181x_int8.cvimodel /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel"
```

Any SSCMA-compatible YOLO detection cvimodel works (YOLOv8/v11/v26) via ModelFactory auto-detection.

## Configuration

The init script loads `/etc/retail-vision.conf` on the device. A default config is installed with the deb package; it will **not** be overwritten on upgrade.

```bash
# /etc/retail-vision.conf
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel"
```

### Common Config Examples

```bash
# Change MQTT broker address
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel --mqtt-host 192.168.1.100 --mqtt-port 1883"

# Add RTSP authentication
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel --rtsp-user recamera --rtsp-pass recamera.1"

# Full example with remote MQTT and RTSP auth
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel --mqtt-host 192.168.1.100 --mqtt-port 1883 --rtsp-user recamera --rtsp-pass recamera.1"
```

After editing, restart the service:

```bash
sudo /etc/init.d/S92retail-vision restart
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model PATH` | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` | Model path |
| `-c, --conf-threshold` | `0.5` | Detection confidence threshold |
| `--rtsp-port` | `8554` | RTSP server port |
| `--rtsp-session` | `live0` | RTSP session name |
| `--rtsp-user` | *(none)* | RTSP Basic auth username |
| `--rtsp-pass` | *(none)* | RTSP Basic auth password |
| `--mqtt-host` | `localhost` | MQTT broker host |
| `--mqtt-port` | `1883` | MQTT broker port |
| `--mqtt-topic` | `recamera/retail-vision/vision` | MQTT publish topic |
| `--person-height` | `1.7` | Average person height in meters (for speed estimation) |
| `--dwell-engaged` | `1.5` | Seconds stationary before ENGAGED state |
| `--dwell-assist` | `20.0` | Seconds stationary before ASSISTANCE state |
| `--dwell-speed` | `10.0` | Speed threshold (px/s) below which = stationary |
| `--window-duration` | `60.0` | Rolling window for zone metrics (seconds) |
| `--no-rtsp` | | Disable RTSP streaming |
| `--no-mqtt` | | Disable MQTT publishing |
| `-v, --verbose` | | Verbose logging |

## Outputs

### RTSP Stream

```
# Without auth
rtsp://<device_ip>:8554/live0

# With auth
rtsp://recamera:recamera.1@<device_ip>:8554/live0
```

Open with VLC, ffplay, or any RTSP client.

### MQTT Payload

Topic: `recamera/retail-vision/vision`

```json
{
  "timestamp": 1709500000000,
  "frame_id": 12345,
  "frame_width": 1280,
  "frame_height": 720,
  "fps": 14.8,
  "inference_time_ms": 49.0,
  "zone": {
    "occupancy_count": 3,
    "browsing_count": 1,
    "engaged_count": 1,
    "assist_count": 1,
    "peak_customer": 5,
    "avg_dwell_time": 8.5,
    "avg_engagement_time": 4.2,
    "avg_velocity": 0.45,
    "entry_count": 12,
    "exit_count": 10
  },
  "persons": [
    {
      "track_id": 7,
      "confidence": 0.85,
      "bbox": {"x": 120, "y": 85, "width": 210, "height": 480},
      "velocity": {"vx": 0.15, "vy": -0.02, "speed_m_s": 0.42},
      "state": "engaged",
      "dwell_duration": 5.2
    }
  ]
}
```

**Coordinate format**: `bbox` uses absolute pixel coordinates (top-left x, y + width, height) referenced to `frame_width` x `frame_height`. Letterbox correction is applied internally to compensate for the square model input (640x640) vs 16:9 display output.

#### Zone Metrics

| Field | Description |
|-------|-------------|
| `occupancy_count` | Current persons in view (median-smoothed over 5 frames) |
| `browsing_count` | Persons in TRANSIENT or DWELLING state |
| `engaged_count` | Persons in ENGAGED state (stationary > 1.5s) |
| `assist_count` | Persons in ASSISTANCE state (stationary > 20s) |
| `peak_customer` | Max occupancy in rolling window |
| `avg_dwell_time` | Average dwell duration of removed tracks in window |
| `avg_engagement_time` | Average engagement duration of removed tracks |
| `avg_velocity` | Average speed (m/s) of removed tracks |
| `entry_count` | Cumulative entries since start |
| `exit_count` | Cumulative exits since start |

#### Dwell States

| State | Condition |
|-------|-----------|
| `transient` | Person is moving |
| `dwelling` | Person stopped < 1.5s |
| `engaged` | Person stationary 1.5-20s |
| `assistance` | Person stationary > 20s |

## Architecture

```
Camera → YOLOv11n detect → PersonTracker.update → ZoneMetrics.update → MQTT publish
                                                                     → RTSP H.264 stream
```

### Tracking Logic (aligned with UA VisionStream)

- **Two-pass matching**: IoU with velocity-predicted position, then center-distance fallback for recently lost tracks
- **Entry**: counted when a new track is created
- **Exit**: counted when a lost track expires near frame edge (velocity-aware classification)
- **Edge loss** (near edge + moving toward edge): short buffer (0.5s), then exit
- **Center loss** (mid-frame disappearance): long buffer (3s), no exit count (likely occlusion)
- **Velocity**: EMA-smoothed (alpha=0.08) with adaptive sudden-stop detection (alpha=0.6)
- **Speed m/s**: `(speed_px_s / bbox_height_px) * avg_person_height_m`
- **Stationary frame decay**: gradual decay tolerates brief gestures without resetting dwell timer

### Letterbox Coordinate Correction

The CVI VPSS letterboxes the 16:9 sensor image into the 640x640 model input (640x360 content area with 140px top/bottom padding). The MQTT publisher corrects for this when converting model-normalized coordinates to absolute display pixels.

## Debug Page

A browser-based debug tool is included at `debug/index.html`:

```bash
cd solutions/retail-vision/debug
python3 server.py --rtsp "rtsp://recamera:recamera.1@<device_ip>:8554/live0"
# Open http://localhost:8080/index.html
```

Features:
- MJPEG proxy of RTSP stream (low-latency ffmpeg transcoding)
- MQTT overlay: bounding boxes, track IDs, dwell states
- Zone metrics panel, person list, raw JSON view

Requires: `mosquitto` WebSocket listener on port 9001 (configure in `/etc/mosquitto/mosquitto.conf` or local mosquitto).

## Debugging

```bash
# SSH to device
ssh recamera@<device_ip>

# View live log
sudo tail -f /var/log/retail-vision.log

# Service control
sudo /etc/init.d/S92retail-vision status
sudo /etc/init.d/S92retail-vision restart
sudo /etc/init.d/S92retail-vision stop

# Manual run with custom args
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib/3rd:/lib:/usr/lib
retail-vision -v --mqtt-host 192.168.10.195 --rtsp-user admin --rtsp-pass secret

# Subscribe to MQTT from host
mosquitto_sub -h <device_ip_or_host> -p 1883 -t "recamera/retail-vision/#" -v
```
