# Fall Detection — integration guide

Fall Detection runs YOLO11n-Pose on the reCamera TPU, associates one stable
subject, and feeds a 3.2-second COCO-17 pose history to a tiny learned temporal
classifier. Geometric hip, torso and box features remain visible and drive the
explainable `suspected`/recovery states; the classifier confirms the alarm.
Inference is fully local and requires no cloud service.

The states are `normal`, `suspected`, `fallen`, and `recovering`:

- `normal` — no fall evidence (or the subject has completed recovery).
- `suspected` — multi-feature evidence is accumulating.
- `fallen` — a confirmed fall; `fall_event` is true only on the transition
  into this state and `event_id` increments once.
- `recovering` — an upright posture has been seen, but it must persist for the
  recovery window before returning to `normal`.

If the first frame already shows a person lying down, the app reports state
`fallen` but intentionally emits no `fall_event`; there was no observable
transition after startup.

## Configuration

The console writes `/userdata/local/apps/fall-detection.config.json`; it can
also be edited over SSH and is reloaded while the app runs. All thresholds are
configurable (the manifest lists ranges and defaults):

| Key | Default | Meaning |
|---|---:|---|
| `confidence` | 0.40 | YOLO person score threshold |
| `keypoint_confidence` | 0.50 | COCO-17 keypoint visibility threshold |
| `hip_drop_speed_threshold` | 0.25 | Downward hip velocity in normalised frame units/s |
| `hip_drop_distance_threshold` | 0.02 | Net hip descent from the last non-horizontal pose |
| `motion_window_sec` | 0.75 | Maximum delay from rapid drop to horizontal posture |
| `torso_angle_threshold_deg` | 55 | Angle away from vertical considered lying |
| `bbox_aspect_ratio_threshold` | 1.25 | Width/height considered lying |
| `min_suspected_features` | 2 | Required active features (speed, torso, aspect) |
| `confirmation_sec` | 0.80 | Evidence persistence before event |
| `suspected_timeout_sec` | 1.50 | How long a pending suspicion may wait |
| `occlusion_grace_sec` | 0.75 | Confirm through a short post-impact pose occlusion |
| `recovery_torso_angle_deg` | 35 | Upright torso-angle limit |
| `recovery_aspect_ratio` | 1.10 | Upright width/height limit |
| `recovery_window_sec` | 2.00 | Upright persistence before normal |
| `cooldown_sec` | 3.00 | Suppress immediate re-alarms after an event |

Example:

```json
{"confirmation_sec":1.0,"cooldown_sec":5.0,"torso_angle_threshold_deg":60}
```

## MQTT and Home Assistant

One JSON document is published per processed frame on
`recamera/fall-detection/results`. Discovery entities include **Fall Detected**,
**Fall State**, **Fall Event ID**, **Person Count**, and **Person Present**.

```json
{
  "timestamp": 1753689600123,
  "frame_id": 4821,
  "fall_detected": true,
  "fall_event": true,
  "event_id": 7,
  "state": "fallen",
  "person_detected": true,
  "person_count": 1,
  "fallen_count": 1,
  "tracking": true,
  "features": {
    "hip_y": 0.74,
    "hip_drop_speed": 0.81,
    "torso_angle_deg": 72.0,
    "bbox_aspect_ratio": 1.55,
    "evidence_features": 3,
    "lying_posture": true,
    "temporal_probability": 0.91,
    "temporal_positive": true,
    "in_cooldown": true
  }
}
```

`fall_event` is an edge flag; use `event_id` for de-duplication. `fall_detected`
remains true throughout `fallen` and `recovering`. Diagnostic features are
included on every frame so threshold tuning can be done from recorded MQTT
without rerunning inference.

## Video and camera placement

- RTSP: `rtsp://<device_ip>:8554/live0` (clean scene)
- Console preview: `ws://<device_ip>:8001/` and `ws://<device_ip>:8001/results`
- Snapshot: `http://<device_ip>:8001/snapshot.jpg`

Use a fixed view that keeps the whole person and both shoulders/hips in frame.
The app intentionally tracks one subject. It acquires the highest-confidence
person, then associates subsequent boxes by overlap instead of switching to
the highest score every frame. `person_count` is consequently 0 or 1 and
`fallen_count` is 0 or 1. Full multi-person fall analysis is not claimed.

The init script is `K92fall-detection` (K-prefix). The supervisor owns camera
start/stop and app selection, so the script does not autostart at boot. Before
starting it stops common camera consumers (sscma-node, Node-RED, detector and
OCR services), waits for their processes to exit, and exports the complete
`LD_LIBRARY_PATH` required by the CVI runtime. Keep only one camera app active;
otherwise VPSS/RTSP port conflicts are expected.

## Testing

`tests/fall_detector_test.cpp` is a pure C++ replay test and has no device or
SSCMA dependency. It also verifies the learned gate and post-impact pose-loss
paths in addition to geometry, recovery and cooldown:

```bash
c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/fall_detector.cpp \
  solutions/fall-detection/tests/fall_detector_test.cpp -o /tmp/fall_detector_test
/tmp/fall_detector_test
```

For NPU end-to-end checks with an open video or public dataset, use the
offline RGB evaluator. It loads the same cvimodel, `PoseDetector`, feature
extraction, and `FallDetector` as the live path, while skipping camera/RTSP,
debug, and MQTT services. Input is contiguous RGB888 frames (no header):

```bash
ffmpeg -i public-fall-video.mp4 \
  -vf 'fps=15,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black' \
  -pix_fmt rgb24 -f rawvideo /tmp/video.rgb
fall-detection --model /path/to/yolo11n_pose_cv181x_int8.cvimodel \
  --offline-rgb /tmp/video.rgb --offline-width 640 --offline-height 640 --offline-fps 15 \
  > results.jsonl
```

Each frame produces one JSON line; the final `summary` line contains `frames`,
`events`, `last_state`, and `fall_detected`. Exit code 0 means all complete
frames were processed; 2 means the file was missing, empty, or ended mid-frame;
1 indicates model initialisation failure and 3 indicates an NPU inference error. This makes the command suitable for
CI or a labeled public clip without claiming that a single clip is a benchmark.
