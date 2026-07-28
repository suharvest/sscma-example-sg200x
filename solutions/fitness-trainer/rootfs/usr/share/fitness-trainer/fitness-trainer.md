# Fitness Trainer — Integration Guide

Counts repetitions of a chosen exercise from the camera alone and publishes the
running count over MQTT. All inference runs on the device; no video leaves it.

Detection is YOLO11n-Pose (COCO 17 keypoints) on the TPU. The rep counter reads
one joint angle per exercise and advances a hysteresis state machine.

## Choosing the exercise

Three movements ship today:

| `mode` | Movement | Angle tracked |
|---|---|---|
| `squat` | Squat | knee flexion — hip / knee / ankle |
| `push_up` | Push-up | elbow flexion — shoulder / elbow / wrist |
| `hammer_curl` | Hammer curl | elbow flexion, each arm independently |

Set it on this app's page in the console (**Exercise**), together with reps per
set and number of sets. Saving restarts the app, which resets the count.

The same settings can be written directly, which is how a phone, a shell script
or a Node-RED flow switches modes without the console:

```bash
cat > /userdata/local/apps/fitness-trainer.config.json <<'EOF'
{"mode":"push_up","target_reps":15,"target_sets":4}
EOF
```

The app polls that file and picks the change up within about two seconds — no
restart. Switching the exercise resets the count; changing only the targets
keeps it.

A device dedicated to one movement can pin it in `/etc/fitness-trainer.conf`
(`EXERCISE_MODE=squat`), which overrides the console for every run.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `mode` | `squat` | Exercise, see table above |
| `target_reps` | 12 | Reps per set |
| `target_sets` | 3 | Sets before the workout is complete |
| `idle_reset_seconds` | 60 | Reset the count after this long with nobody in frame (0 = never) |
| `confidence` | 0.40 | Person detection score threshold |
| `keypoint_confidence` | 0.50 | Per-keypoint threshold; joints below it are treated as not visible |

## MQTT output

Topic: `recamera/fitness-trainer/results`, one message per processed frame.

```json
{
  "timestamp": 1753689600123,
  "frame_id": 4821,
  "inference_time_ms": 96,
  "exercise": "squat",
  "stage": "down",
  "angle": 92.4,
  "reps": 7,
  "target_reps": 12,
  "set": 2,
  "target_sets": 3,
  "workout_complete": false,
  "person_detected": true,
  "tracking": true,
  "rep_completed": false,
  "set_completed": false
}
```

- `timestamp` — milliseconds from the device's wall clock. reCamera ships with
  its RTC module unloaded and has no backup battery, so on a device that has
  never synced time this counts from boot rather than from the epoch. Sync NTP
  (or load `cv181x_rtc.ko`) if you need real timestamps; use `frame_id` if you
  only need ordering.
- `stage` — `up` / `down` for squat and push-up, `curl` / `extend` for hammer
  curl, `idle` when nobody is tracked, `out of frame` when the person is
  detected but the joints this exercise needs are not visible.
- `angle` — the tracked joint angle in degrees, smoothed. Absent when there is
  no reading.
- `rep_completed` / `set_completed` — true only on the message where the event
  happened, so a subscriber can react without diffing counts.
- `reps_left` / `reps_right` — hammer curl only. The set counter advances at the
  pace of the slower arm when both are visible.
- `form_warning` — present only when something is off, e.g.
  `"Partial rep - squat deeper"` or `"Left elbow drifting - keep it at your side"`.

`reps` is the count **within the current set**; it returns to 0 when a set
completes and `set` increments. On the final set it stops at `target_reps` and
`workout_complete` goes true.

### Home Assistant

Entities are published automatically via MQTT discovery: **Reps**, **Set**,
**Exercise**, **Stage**, **Athlete Present**, **Workout Complete**. No YAML
needed — enable MQTT discovery in Home Assistant and the device appears.

## Video

- RTSP: `rtsp://<device_ip>:8554/live0` — the clean scene, no overlay.
- Console preview: `ws://<device_ip>:8001/` (H.264) and
  `ws://<device_ip>:8001/results` (JSON). The overlay draws one box around the
  athlete labelled `squat  7/12  set 2/3  down`.
- Snapshot: `http://<device_ip>:8001/snapshot.jpg`.

## Camera placement

The exercise dictates what has to be in frame:

- **Squat** — the whole body from the side or at 45°. The ankle must be visible;
  a camera that crops at the knee gives no reading and the stage stays
  `out of frame`.
- **Push-up** — a side view at roughly floor height. Head-on, the elbow angle
  barely changes and reps go uncounted.
- **Hammer curl** — a front or side view from the waist up. Both shoulders,
  elbows and wrists in frame if you want both arms counted.

## Accuracy, honestly

The counter reads a quantised keypoint model, and it is smoothed and debounced
accordingly: a rep faster than 0.4–0.5 s is rejected as jitter, and an angle
rattling inside the hysteresis band does not count. What this costs you is
genuinely explosive reps; what it buys is not counting a twitch as a rep.

Reps are counted on the way **back up**, when the movement completes — not at
the bottom. Half a squat therefore counts as nothing rather than as one, and a
shallow-but-complete rep counts with a `form_warning`.

Form feedback is deliberately narrow: rep depth for squat and push-up, elbow
drift for hammer curl. It is not a physiotherapist and does not check knee
tracking, spine angle or tempo.

## Not included

- **Privacy blur.** Deliberately not wired in — masking the athlete is at odds
  with what this app does. The device-level blur switch in the console has no
  effect on this app, which is why its page does not offer one.
- **ONVIF.** No WS-Discovery and no analytics metadata: a VMS will not find this
  camera on its own, though it will play the RTSP URL if you enter it by hand.
  Rep counts go out over MQTT only.
