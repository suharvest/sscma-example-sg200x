# face-recognition — on-device evaluation (reCamera SG2002)

Date: 2026-09-05/06. Device: `recamera-one` (SG2002, 180 MB RAM, reCamera OS), camera 1280×720, inference 640×640.
Models: `scrfd_500m_int8_qi.cvimodel` (INT8, int8 input), `arcface_mbf_bf16.cvimodel`, `liveness_minifasnet_bf16.cvimodel` (disabled by default).
model_tag `cvi:scrfd500m+mbf512@bf16`.

## Latency (`inference_time_ms` from MQTT results, steady state)

| Condition | Detector input | pack | forward | per-frame total |
|---|---|---|---|---|
| no face | F32 (old `scrfd_500m_int8.cvimodel`) | 94 ms | 189 ms | 283 ms (10-frame mean; 600–880 ms in the first minute) |
| no face | int8 (`--quant_input`) | 46 ms | 10 ms | **61 ms** (58–61 steady) |
| 1 face, embedding every 5th frame | int8 | — | — | **81–105 ms** |

Stage timings are logged every 30 calls (`scrfd timing:` / `cvi timing:` in `/var/log/face-recognition.log`).
The remaining host cost is the NCHW int8 pack loop (46 ms of the 61); a per-plane memcpy would remove most of it.

## Accuracy

| Check | Result |
|---|---|
| Same probe image, device BF16 embedding vs fp32 onnxruntime (same decoder/alignment code) | cosine **0.9897** |
| Live camera, person enrolled from 5 frames then recognised | score **0.854–0.868**, `stable=true`, det_score 0.57–0.79 at ~400 px face |
| Cross identity (enrolled person vs probe) | −0.04 |
| `model_deploy` INT8 SCRFD vs fp32 (9 heads) | min cosine 0.903 / euclid 0.508 |

## Resources

- RAM: 27 MB used / 103–107 MB free with the app running (`free -m`); ION not measured.
- `/userdata`: models 8.7 MB + gallery ≈ 5 KB per user (512 float32 as JSON text).
- dmesg: 0 `vpss … err` / `vi err` over the session after the power cycle.

## Known issues

- `S93sscma-supervisor` holds the camera; `K92face-recognition start` stops node-red/sscma-node/other solutions but not the supervisor — stop it first or start via the app gallery.
- After a VPSS wedge (`vpss_get_chn_frame … fail` loop) `reboot` does not recover; power cycle. Reproduced with unmodified face-analysis, so it is not app-specific.
- ArcFace INT8 (MS1M-calibrated) failed the euclid gate (0.762 < 0.8); BF16 is shipped.
