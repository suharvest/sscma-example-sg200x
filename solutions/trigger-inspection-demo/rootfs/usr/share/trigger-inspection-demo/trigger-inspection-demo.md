# Trigger Inspection Demo

Captures one frame per trigger and returns an OK/NG verdict. Between triggers the camera channel runs at the standby rate (`--idle-fps`, default 1 fps).

## Triggers

| Trigger | How | Response |
|---|---|---|
| HTTP | `GET` or `POST http://<device_ip>:8090/trigger?tag=<id>` | `200` with the result JSON once the frame is inspected; `409` while another HTTP trigger is in progress |
| GPIO | Edge on a sysfs GPIO input (`GPIO_LINE` in the config file) | Result on MQTT only |

```bash
curl -X POST 'http://<device_ip>:8090/trigger?tag=part-001'
```

## Result

Published on `recamera/trigger-inspection-demo/results` and returned in the HTTP response:

```json
{"timestamp":1730956800000,"frame_id":12,"verdict":"NG","reason":"low_contrast",
 "trigger":{"source":"http","seq":3,"tag":"part-001"},
 "metrics":{"mean":118.4,"stddev":6.2,"samples":19200},"roi":[160,120,320,240],"resolution":[640,480],
 "timing_us":{"validate":3,"roi":2,"measure":5210,"judge":1,"total":5216}}
```

`verdict` is `OK`, `NG` (a check failed; `reason` is `too_dark`, `too_bright` or `low_contrast`) or `ERROR` (the frame could not be inspected). If no frame arrives within 2 s of an HTTP trigger, the response is `{"accepted":true,"error":"no_frame"}`.

## Configuration

`/etc/trigger-inspection-demo.conf` (shell variables, read by the init script):

| Variable | Default | Meaning |
|---|---|---|
| `HTTP_PORT` | `8090` | HTTP trigger port, `0` disables |
| `GPIO_LINE` | unset | sysfs GPIO number of the trigger input |
| `GPIO_EDGE` | `rising` | `rising`, `falling` or `both` |
| `GPIO_ACTIVE_LOW` | `0` | `1` for an input that is active low |
| `IDLE_FPS` | `1` | Camera channel rate in standby (1 to 15) |
| `EXTRA_OPTS` | empty | Further options, e.g. `--roi 0.3,0.3,0.4,0.4 --mean-min 60 --std-min 20 --rtsp` |
| `MQTT_HOST`, `MQTT_PORT`, `MQTT_TOPIC` | `localhost`, `1883`, see above | MQTT broker and topic |
| `DEBUG_ENABLED`, `DEBUG_PORT` | `0`, `8001` | Console preview (WebSocket video + `/snapshot.jpg`); `1` turns it on |

Run `trigger-inspection-demo --help` for the full option list.

## Limits

- The sample check is brightness and contrast in one region; replace the `measure` / `judge` steps in `main.cpp` for a real inspection.
- One HTTP trigger is processed at a time; GPIO and HTTP triggers share a queue of 8.
- In standby the camera still copies `IDLE_FPS` frames per second; the preview stream (debug WebSocket, RTSP) is off by default; when enabled it is encoded by the video hardware at its own rate.
