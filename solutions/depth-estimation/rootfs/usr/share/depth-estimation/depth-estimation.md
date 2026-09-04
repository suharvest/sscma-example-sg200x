# Monocular Depth Estimation

Dense relative depth from the single reCamera sensor. The model runs on the CVI
TPU at every inference frame, the result is drawn as a colour preview in the
corner of the video stream, and a compact proximity summary is published over
MQTT for Home Assistant or any other automation.

## What "relative depth" means here

The model returns one number per pixel. **Smaller means nearer.** The numbers
have no unit and no absolute scale: a single camera with no reference object
cannot measure how far away something is, and nothing in this application
converts its output into a distance. Two frames of the same scene under
different lighting can produce different raw ranges.

What is stable, and what the reported fields are built on, is the *ordering*
within one frame — which parts of the picture are nearer than which. Every
derived value (`proximity`, `near_ratio`, `zones`) is normalised inside its own
frame, so it is comparable over time even though the raw depth is not.

Accuracy is best on indoor scenes: the default model is trained on indoor
photographs. Outdoors, and on large uniform surfaces (a blank wall, a clear
sky), the depth map degrades — expect the ordering to stay roughly right and the
fine structure to be wrong.

## Model

| | |
|---|---|
| Default path | `/userdata/local/models/fastdepth_224_bf16.cvimodel` |
| Input | 1x3x224x224, RGB, CHW, scaled to `[0,1]` |
| Output | dense HxW relative depth map |
| Measured | ~19 ms per inference |

### BF16 or INT8

Two builds of the same network are available. **BF16 is the default and the one
to use unless you have a reason not to.**

| | BF16 (default) | INT8 |
|---|---|---|
| Calibration set | none needed | ~200-500 images, domain-matched |
| `.cvimodel` | 2.9 MB | 1.4 MB |
| ION | 6.69 MB | 3.91 MB |
| Cosine vs float32 | 0.999998 | 0.999502 |
| SQNR vs float32 | 39.98 dB | 16.38 dB |
| Inference | ~19 ms | ~18 ms |

BF16 needs no calibration table, so there is nothing to collect and nothing to
get wrong. INT8 buys 1.5 MB and roughly 6% — neither matters here, since the
inference budget is a 66.7 ms frame period and the ION budget is around 60 MB.

The SQNR gap is the number that matters. Depth output is a continuous scalar
field, so quantisation noise shows up directly as depth jitter and local
near/far inversions; a detector would hide the same noise behind its argmax.
If you do want INT8, calibrate it on images captured with **your** reCamera in
**your** deployment scene — a general-purpose image set will not represent the
activation ranges this network actually sees.

The model does **not** ship inside this package; copy it to
`/userdata/local/models/` before enabling the app. The console's model override
file `/userdata/local/apps/depth-estimation.model` (first line = absolute path)
takes precedence over the default when present.

At start-up the application logs the real tensor names, shapes, dtypes and
quantisation parameters it found, and refuses to run if the shapes are not a
single 1x3xHxW input and a single dense HxW output. A model that does not match
produces a clear error and an exit, never a plausible-looking wrong answer.

## Command-line options

```
depth-estimation [options]
```

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--model PATH` | `/userdata/local/models/fastdepth_224_bf16.cvimodel` | Model to load |
| `--mqtt-host HOST` | `localhost` | Broker host (legacy mode only) |
| `--mqtt-port PORT` | `1883` | Broker port (legacy mode only) |
| `--mqtt-topic TOPIC` | `recamera/depth-estimation/results` | Results topic |
| `--mqtt-interval MS` | `500` | Minimum gap between published results (2 Hz) |
| `--near-threshold F` | `0.75` | Proximity at or above which a pixel counts as near |
| `--near-ratio F` | `0.05` | Fraction of near pixels that makes `near_present` true |
| `--no-pip` | preview on | Disable the depth preview overlay |
| `--pip-size WxH` | `320x180` | Preview size |
| `--no-rtsp` | RTSP on | Disable RTSP streaming |
| `--no-mqtt` | MQTT on | Disable MQTT publishing |
| `--no-debug` | debug on | Disable the debug WebSocket stream |
| `--debug-port PORT` | `8001` | Debug WebSocket port |
| `-v`, `--verbose` | off | Per-frame log line |
| `-h`, `--help` | | Usage |

The init script `/etc/init.d/K92depth-estimation` reads
`/etc/recamera.conf` and then `/etc/depth-estimation.conf`; it maps
`MQTT_HOST`, `MQTT_PORT`, `MQTT_TOPIC`, `DEBUG_ENABLED`, `DEBUG_PORT`,
`PIP_ENABLED`, `NEAR_THRESHOLD` and `NEAR_RATIO` onto the options above.

## Video

| | |
|---|---|
| RTSP | `rtsp://<device_ip>:8554/live0` — 1280x720 @ 15 fps |
| Debug video | `ws://<device_ip>:8001/` |
| Debug results | `ws://<device_ip>:8001/results` |
| Snapshot | `http://<device_ip>:8001/snapshot.jpg` |
| Inference channel | 320x180 @ 15 fps |

Inference runs on every captured frame; at 15 fps the ~18 ms forward pass leaves
the 66.7 ms frame period comfortably clear, so no frames are skipped. MQTT is
rate-limited separately (see `--mqtt-interval`); the preview updates on every
inference frame.

### Depth preview (picture-in-picture)

A 320x180 tile in the **bottom-right corner** of the encoded stream, inset 16
pixels from both edges, drawn as one hardware overlay region. Colours run from
red (nearest in this frame) through orange, green and cyan to blue (farthest).

The preview covers exactly the sensor content the stream shows, so a feature in
the tile sits at the same relative position as in the main picture — there is no
letterbox to reason about.

The preview and the device-wide privacy blur cannot both be active on this
application: two ordinary overlay regions on one VPSS channel are not allowed to
intersect. The app manifest therefore declares `"privacy_blur": false` and the
app never creates a mask.

## MQTT

Topic: `recamera/depth-estimation/results` (2 Hz by default).

```json
{
  "timestamp": 1751000000000,
  "frame_id": 412,
  "inference_time_ms": 18.4,
  "depth": {
    "unit": "relative",
    "smaller_is_nearer": true,
    "source_size": [320, 180],
    "valid_roi": [0, 0, 320, 180],
    "min": 0.1120, "max": 0.9840, "mean": 0.5231,
    "p02": 0.1450, "p50": 0.5010, "p98": 0.9120,
    "near_ratio": 0.1834,
    "near_present": true,
    "zones": [0.21, 0.35, 0.19, 0.44, 0.87, 0.41, 0.62, 0.91, 0.58]
  }
}
```

| Field | Meaning |
|---|---|
| `unit` | Always `"relative"`. There is no metric reading here. |
| `smaller_is_nearer` | Always `true` — the direction of the raw scale. |
| `source_size` | `[w, h]` of the inference frame the depth was computed from. |
| `valid_roi` | `[x, y, w, h]` of the sensor content inside that frame (see below). |
| `min`/`max`/`mean` | Raw model units over the whole depth map. |
| `p02`/`p50`/`p98` | Percentiles of the same raw values. |
| `near_ratio` | Fraction `[0,1]` of pixels whose proximity is at or above `--near-threshold`. |
| `near_present` | `near_ratio >= --near-ratio`. |
| `zones` | 3x3 grid, row-major (index 4 = centre). Each entry is the **proximity** of that cell's 5th-percentile depth — how near its nearest content is, `0` = far, `1` = nearest in frame. |

Proximity is defined per frame as

```
proximity = clamp((p98 - d) / (p98 - p02), 0, 1)
```

p02/p98 rather than min/max: one hot pixel at either extreme would otherwise
rescale the entire frame and make the reported numbers jump between two frames
that look identical.

### Home Assistant

With `/userdata/local/ha.conf` configured (`HA_ENABLED=1`), three entities are
published by MQTT Discovery:

| Entity | Type | Value |
|---|---|---|
| Near Area | sensor, `%` | `near_ratio * 100` |
| Near Object | binary_sensor, `occupancy` | `near_present` |
| Center Nearest (relative) | sensor | `zones[4]`, `0..1` |

Without `ha.conf` the app falls back to a plain connection to
`--mqtt-host`/`--mqtt-port` and publishes only the results topic.

## Grey bars, and why they are cropped before inference

The camera's VPSS fits the 16:9 sensor picture into every channel preserving
aspect ratio and pads the remainder with grey. A depth model trained on
full-frame photographs has never seen those bars: it invents depth for them, and
that invented depth also drags the frame-wide p02/p98 normalisation around,
corrupting every reported number — not just the bars.

So the bars are removed *before* inference. The inference channel is configured
16:9 (320x180) precisely so VPSS produces none in the first place; if a device
hands back a differently shaped frame anyway, the application computes the inner
16:9 rectangle at runtime, crops to it, and reports it in `valid_roi`. The first
frame logs the actual frame size and the ROI it derived:

```
Inference frame 320x180, valid ROI [x=0,y=0,w=320,h=180] (16:9 channel, no letterbox)
```

Preprocessing then stretches that ROI to the model's input size. It does not
letterbox — the same convention the rest of the SSCMA pipeline uses — so the
depth map maps back to the picture linearly and nothing downstream has to undo
an offset.

## ONVIF

Device and Media2 services plus the snapshot URI are available so a VMS can
discover the camera and pull the stream, controlled by the console's ONVIF
settings as with every other application.

ONVIF analytics metadata is deliberately **not** published. A depth map contains
no objects, and manufacturing bounding boxes from it would put fiction on a VMS
timeline.

## Not included

No object detection, no per-object depth sampling, no tracking, and no absolute
distance output. The result is one relative depth field per frame and the
summary above.
