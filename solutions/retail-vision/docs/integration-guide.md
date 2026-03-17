# Retail Vision — Integration Guide

ReCamera running **Retail Vision** provides two output interfaces: an **RTSP** video stream and an **MQTT** data stream with real-time people-flow analytics.

## RTSP Video Stream

| Property | Value |
|----------|-------|
| URL | `rtsp://recamera:recamera.1@<device_ip>:8554/live0` |
| Codec | H.264 |
| Resolution | 1280 x 720 |
| Frame Rate | 15 fps |
| Auth | Basic (username: `recamera`, password: `recamera.1`) |

The stream can be opened with VLC, ffplay, or any RTSP-compatible client.

```bash
# Example
ffplay rtsp://recamera:recamera.1@192.168.10.158:8554/live0
```

## MQTT Data Stream

| Property | Value |
|----------|-------|
| Broker | `192.168.20.38:1883` (configurable) |
| Topic | `recamera/retail-vision/vision` |
| QoS | 0 |
| Retain | false |
| Publish Rate | Every inference frame (~9 fps) |
| Payload Format | JSON (UTF-8) |

### Subscribing

```bash
mosquitto_sub -h 192.168.20.38 -p 1883 -t "recamera/retail-vision/vision" -v
```

### Payload Schema

Each message is a single JSON object:

```json
{
  "timestamp": 1709500000000,
  "frame_id": 12345,
  "frame_width": 1280,
  "frame_height": 720,
  "fps": 8.9,
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
      "bbox": {
        "x": 120,
        "y": 85,
        "width": 210,
        "height": 480
      },
      "velocity": {
        "vx": 0.15,
        "vy": -0.02,
        "speed_m_s": 0.42
      },
      "state": "engaged",
      "dwell_duration": 5.2
    }
  ]
}
```

### Field Reference

#### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | integer | Unix timestamp in milliseconds |
| `frame_id` | integer | Monotonically increasing frame counter (resets on restart) |
| `frame_width` | integer | Display frame width in pixels (1280) |
| `frame_height` | integer | Display frame height in pixels (720) |
| `fps` | float | Current processing frame rate |
| `inference_time_ms` | float | Model inference time in milliseconds |

#### `zone` — Aggregate Zone Metrics

| Field | Type | Description |
|-------|------|-------------|
| `occupancy_count` | integer | Current number of persons in view |
| `browsing_count` | integer | Persons in `transient` or `dwelling` state |
| `engaged_count` | integer | Persons in `engaged` state (stationary 1.5–20s) |
| `assist_count` | integer | Persons in `assistance` state (stationary > 20s) |
| `peak_customer` | integer | Maximum occupancy observed in the rolling window (default 60s) |
| `avg_dwell_time` | float | Average dwell duration (seconds) of completed tracks in window |
| `avg_engagement_time` | float | Average engagement duration (seconds) of completed tracks |
| `avg_velocity` | float | Average walking speed (m/s) of completed tracks |
| `entry_count` | integer | Cumulative entry count since service start |
| `exit_count` | integer | Cumulative exit count since service start |

#### `persons[]` — Per-Person Tracking Data

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | integer | Unique track identifier (monotonically increasing) |
| `confidence` | float | Detection confidence (0–1) |
| `bbox.x` | integer | Bounding box top-left X in pixels |
| `bbox.y` | integer | Bounding box top-left Y in pixels |
| `bbox.width` | integer | Bounding box width in pixels |
| `bbox.height` | integer | Bounding box height in pixels |
| `velocity.vx` | float | Horizontal velocity (normalized per frame width, per second) |
| `velocity.vy` | float | Vertical velocity (normalized per frame height, per second) |
| `velocity.speed_m_s` | float | Estimated walking speed in meters per second |
| `state` | string | Dwell state (see below) |
| `dwell_duration` | float | Time in current dwell state (seconds) |

**Coordinate system**: `bbox` values are absolute pixel coordinates relative to `frame_width` x `frame_height` (1280 x 720). Origin is top-left corner. Letterbox correction is applied internally so coordinates map directly to the RTSP video frame.

#### Dwell States

| State | Condition | Description |
|-------|-----------|-------------|
| `transient` | Person is moving | Just passing through |
| `dwelling` | Stationary < 1.5s | Briefly stopped |
| `engaged` | Stationary 1.5–20s | Actively looking at something |
| `assistance` | Stationary > 20s | May need staff assistance |

The state machine transitions: `transient` → `dwelling` → `engaged` → `assistance`. Movement resets the state back to `transient`.

#### Entry / Exit Counting

- **Entry**: incremented when a new person track is created.
- **Exit**: incremented when a track expires near the frame edge with velocity pointing outward. Tracks that disappear mid-frame (likely occlusion) do not increment exit count.
- Both counters are cumulative since service start and do not reset within the rolling window.

## Configuration

Device config file: `/etc/retail-vision.conf`

```bash
# Default
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel"

# With remote MQTT broker
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel --mqtt-host 192.168.20.38 --mqtt-port 1883"

# With RTSP authentication
DAEMON_OPTS="-v -m /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel --rtsp-user recamera --rtsp-pass recamera.1"
```

After editing, restart the service:

```bash
sudo /etc/init.d/S92retail-vision restart
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `yolo11n_detection_cv181x_int8.cvimodel` | Detection model path |
| `-c, --conf-threshold` | `0.5` | Detection confidence threshold |
| `--mqtt-host` | `localhost` | MQTT broker address |
| `--mqtt-port` | `1883` | MQTT broker port |
| `--mqtt-topic` | `recamera/retail-vision/vision` | MQTT topic |
| `--rtsp-port` | `8554` | RTSP server port |
| `--rtsp-session` | `live0` | RTSP session name |
| `--rtsp-user` | *(none)* | RTSP auth username |
| `--rtsp-pass` | *(none)* | RTSP auth password |
| `--person-height` | `1.7` | Average person height in meters |
| `--dwell-engaged` | `1.5` | Seconds before ENGAGED state |
| `--dwell-assist` | `20.0` | Seconds before ASSISTANCE state |
| `--dwell-speed` | `10.0` | Stationary speed threshold (px/s) |
| `--window-duration` | `60.0` | Rolling window duration (seconds) |
| `--no-rtsp` | | Disable RTSP output |
| `--no-mqtt` | | Disable MQTT output |
| `-v, --verbose` | | Enable verbose logging |
