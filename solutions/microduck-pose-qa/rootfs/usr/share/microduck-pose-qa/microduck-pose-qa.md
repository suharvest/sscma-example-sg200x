# Microduck Pose QA — Integration Guide

Microduck Pose QA executes on-device 17-keypoint robot pose estimation on the Seeed reCamera SG2002 TPU. It tracks 14 servo drive joints in real time at 384x384 resolution, automates kinematic quality assurance (T-16 in the Microduck production test suite), detects 14.4° gear spline assembly errors, and verifies bilateral leg symmetry.

## 17 Keypoints Topology

| Index | Name | Anatomical Mapping |
|---:|---|---|
| 0 | `snout` | Mouth tip |
| 1 | `head_top` | Head roll axis |
| 2 | `neck` | Head yaw axis |
| 3 | `body_center` | Neck pitch / Torso center |
| 4 | `tail` | Tail end |
| 5 | `fl_shoulder` | Front left hip yaw |
| 6 | `fl_knee` | Front left knee |
| 7 | `fl_foot` | Front left foot sole |
| 8 | `fr_shoulder` | Front right hip yaw |
| 9 | `fr_knee` | Front right knee |
| 10 | `fr_foot` | Front right foot sole |
| 11 | `hl_hip` | Hind left hip |
| 12 | `hl_knee` | Hind left knee |
| 13 | `hl_foot` | Hind left foot |
| 14 | `hr_hip` | Hind right hip |
| 15 | `hr_knee` | Hind right knee |
| 16 | `hr_foot` | Hind right foot |

## Kinematic Verification Algorithms

1. **Gear Spline Misalignment (14.4° Error)**:
   - Microduck XL330 servos use 25T output splines ($360^\circ / 25 = 14.4^\circ$).
   - Computes knee angle $\alpha_{knee}$ from vector $\vec{v}_{thigh}$ and $\vec{v}_{shank}$.
   - Alerts if $|\alpha_{knee} - \alpha_{nominal}| \ge 7.0^\circ$.
2. **Bilateral Knee Symmetry**:
   - Computes $\Delta h(t) = |y_{knee,left}(t) - y_{knee,right}(t)|$ during sit-to-stand transitions.
   - Flags motor torque imbalance or mechanical friction when $\Delta h > 12\text{ px}$.
3. **Stand Posture Drift**:
   - Assesses keypoint displacement between initial and post-motion standing states.
   - Flags loose fasteners or structural deformation if drift exceeds $3.0\text{ mm}$.

## Service & Configuration

- **Configuration File**: `/etc/microduck-pose-qa.conf`
- **Init Script**: `/etc/init.d/K92microduck-pose-qa` (managed by supervisor, non-autostart)
- **Model Path**: `/userdata/local/models/microduck_pose_17kpt_384_bf16.cvimodel`
- **RTSP Stream**: `rtsp://<device_ip>:8554/live0`
- **Websocket Preview**: `ws://<device_ip>:8001/` and `ws://<device_ip>:8001/results`
- **MQTT Output**: `recamera/microduck-pose-qa/results`

## Stand Standard Calibration and Single-LED Check (1.0.8, layout 1.0.9)

The stand standard (nominal 2-D thigh angles, their disparity and the trunk tilt) depends on the camera placement. It is read from `NOMINAL_STAND_*` in `/etc/microduck-pose-qa.conf` (a conffile, kept across upgrades) and changed through `POST :8082/api/control`:

| method | params | result |
|---|---|---|
| `calibrate_stand` | `timeout_s` (1-10, default 10), `persist` (default true) | one 30-frame stand window on a golden sample; medians become the standard and are written to the conf file; 422 with fewer than 15 measured frames |
| `set_stand_nominal` | any of `nominal_stand_theta_l_deg`, `nominal_stand_theta_r_deg`, `nominal_stand_disparity_deg`, `nominal_stand_spine_tilt_deg`; `persist` | sets the values directly |
| `get_config` | - | current standard, tolerances, face LED layout |

`start_action` with `action_id: "face_led"` accepts `params.expected`: `{"led": "face:gr2:red", "rgb": [255, 0, 0]}` or `{"flashlight": "red|green|blue|white"}`. The LED layout is stored relative to the face: per LED place (u, v) in lens-ring coordinates (ring centre = origin, ring radius = 1), `FACE_LED_LAYOUT` in the conf file, written by `set_face_led_layout` ({"layout": {place: [u, v]}, "persist": true}). Each frame the ring is found, a similarity (scale, rotation, translation) is fitted to the ring and the LED dots by least squares, and the expected place must show the expected colour while no other place is lit; no ring or residual > 0.35 ring radii gives `FAIL_FACE_NOT_REGISTERED`. `get_last_report.registration` lists the per-frame scale, rotation and residual (1.0.9). `get_last_report` returns `led_results` (12 entries: `expected_on`, `observed_on`, `on_frames`, `ok`). The default layout is a projection of the simulator station's provisional layout and must be recalibrated for a real face board and camera.

