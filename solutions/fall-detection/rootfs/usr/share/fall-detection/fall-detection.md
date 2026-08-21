# Fall Detection — integration guide

Fall Detection runs YOLO11n-Pose on the reCamera TPU, associates stable
multi-person tracks, and feeds each 3.2-second COCO-17 pose history to a tiny learned temporal
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

If the first frame already shows a person lying down, the app remains `normal`:
there is no observable transition history, so posture alone is not a fall.

## Configuration

The console writes `/userdata/local/apps/fall-detection.config.json`; it can
also be edited over SSH and is reloaded while the app runs. All thresholds are
configurable (the manifest lists ranges and defaults):

| Key | Default | Meaning |
|---|---:|---|
| `confidence` | 0.40 | YOLO person score threshold |
| `keypoint_confidence` | 0.50 | COCO-17 keypoint visibility threshold |
| `temporal_confirmation_required` | true | Require valid current pose plus learned temporal confirmation; false enables legacy geometry-only confirmation |
| `hip_drop_speed_threshold` | 0.25 | Downward hip velocity in normalised frame units/s |
| `hip_drop_distance_threshold` | 0.02 | Net hip descent from the last non-horizontal pose |
| `motion_window_sec` | 0.75 | Maximum delay from rapid drop to horizontal posture |
| `torso_angle_threshold_deg` | 55 | Angle away from vertical considered lying |
| `bbox_aspect_ratio_threshold` | 1.25 | Width/height considered lying |
| `min_suspected_features` | 2 | Required active features (speed, torso, aspect) |
| `confirmation_sec` | 0.80 | Evidence persistence before event |
| `suspected_timeout_sec` | 1.50 | How long a pending suspicion may wait |
| `occlusion_grace_sec` | 0.75 | Retain track/state through a short pose gap; never confirms a new event |
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
**Fall State**, **Fall Event ID**, **Person Count**, **Fallen Count**, and
**Person Present**.

```json
{
  "timestamp": 1753689600123,
  "frame_id": 4821,
  "fall_detected": true,
  "fall_event": true,
  "event_id": 7,
  "event_id_scope": "stream_global_event_id",
  "state": "fallen",
  "person_detected": true,
  "person_count": 1,
  "fallen_count": 1,
  "tracking": true,
  "persons": [
    {
      "track_id": 3,
      "person_detected": true,
      "state": "fallen",
      "event_id": 2,
      "features": {"temporal_probability": 0.91},
      "keypoints": [{"points": [[0.5, 0.2]], "edges": []}],
      "pose17": [[0.5, 0.2, 0.99]]
    }
  ],
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
The app returns every pose person and associates boxes with a lightweight
IoU/centre-distance tracker. Each `track_id` owns an independent temporal
history and fall state machine, so nearby people cannot splice their histories.
Short detector gaps remain in `persons[]` as `person_detected:false` tracks,
allowing post-impact occlusion confirmation; tracks are retired after the
configured timeout. Legacy top-level fields are aggregates: `fall_detected`
and `fall_event` are OR across retained tracks, `person_count` is the number of
currently visible people, `fallen_count` counts retained fallen states, and
`state` is the most severe state. Top-level `event_id` is a stream-global
sequence (`event_id_scope:"stream_global_event_id"`); each `persons[]` item has
its own per-track `event_id`.

Each `persons[]` item includes `track_id`, `state`, `event_id`, `features`,
`keypoints` and stable COCO-17 `pose17` data. The MQTT payload also keeps the
legacy top-level fields for existing consumers. The debug `/results` payload
contains one skeleton group per visible person.

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

The association and aggregate contracts are also host-testable without a
device:

```bash
c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/box_tracker.cpp \
  components/geometry/src/norm_box.cpp \
  solutions/fall-detection/tests/multi_person_tracker_test.cpp \
  -I components/geometry/include -o /tmp/multi_person_tracker_test
/tmp/multi_person_tracker_test

c++ -std=c++17 -I solutions/fall-detection/main \
  solutions/fall-detection/main/fall_detector.cpp \
  solutions/fall-detection/main/payload_aggregate.cpp \
  solutions/fall-detection/tests/payload_aggregate_test.cpp \
  -o /tmp/payload_aggregate_test
/tmp/payload_aggregate_test
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
the stream-global `events`, edge-bearing `event_edges`, `last_state`, and
`fall_detected`. Exit code 0 means all complete
frames were processed; 2 means the file was missing, empty, or ended mid-frame;
1 indicates model initialisation failure and 3 indicates an NPU inference error. This makes the command suitable for
CI or a labeled public clip without claiming that a single clip is a benchmark.
