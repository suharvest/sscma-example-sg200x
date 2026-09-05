# Face Recognition

On-device face recognition for reCamera (SG2002 / CV181x).

## Pipeline

| Stage | Model | Input | Output |
|---|---|---|---|
| Detect | SCRFD-500M, INT8 | 640x640 RGB | boxes + 5 landmarks |
| Align | similarity transform | 5 landmarks | 112x112 chip |
| Embed | MobileFaceNet ArcFace, BF16 | 112x112 RGB | 512-D, L2-normalized |
| Liveness (optional) | MiniFASNet 2.7_80x80, BF16 | 80x80 BGR | P(real) |
| Match | cosine linear scan | 512-D | name + score |

Detection runs every frame. Embedding and liveness run when a track first
appears and then every `EMBED_INTERVAL` frames; the cached verdict is reused in
between. A name is reported as `stable` once two consecutive embedding runs
agree on it.

## Results topic

`recamera/face-recognition/results`

```json
{"timestamp": 1757000000000, "frame_id": 42, "inference_time_ms": 96.0,
 "model_tag": "cvi:scrfd500m+mbf512@bf16", "face_count": 1,
 "faces": [{"track_id": 3, "bbox": [0.31, 0.22, 0.55, 0.61], "det_score": 0.87,
            "name": "alice", "score": 0.62, "live": null, "liveness_score": null,
            "stable": true, "gated": false, "reason": null}]}
```

`bbox` is normalized `[x1, y1, x2, y2]` against the inference frame.
`live` / `liveness_score` are `null` when the anti-spoof stage did not run, so a
consumer can tell "not checked" from "checked and failed".

## Commands

Publish to `recamera/face-recognition/cmd`, read the reply on
`recamera/face-recognition/cmd/ack`.

```json
{"op": "enroll", "name": "alice", "source": "camera", "frames": 5, "request_id": "r1"}
{"op": "enroll", "name": "alice", "source": "image", "image_b64": "...", "request_id": "r2"}
{"op": "remove", "name": "alice", "request_id": "r3"}
{"op": "list", "request_id": "r4"}
{"op": "reload", "request_id": "r5"}
```

A camera enrollment consumes frames containing exactly one eligible face and
gives up after about four seconds. Every ack carries
`{"request_id", "op", "ok", "model_tag", "users"}` plus `"err"` when `ok` is
false.

## Gallery

`/userdata/local/face-gallery/cvi_scrfd500m_mbf512_bf16.json`

The filename and the `_meta.model_tag` inside it encode which models produced
the templates. ArcFace weights differ per backend, so a gallery enrolled on
another accelerator scores cosine near zero here -- that would not fail, it
would silently recognise nobody. The application refuses to load a gallery whose
tag disagrees.

## Configuration

`/etc/face-recognition.conf`. `MATCH_THRESHOLD` (default 0.40) is the cosine
above which an embedding is accepted as a gallery identity; `MIN_FACE_PX`
(default 64) is the short side below which a face is published as `gated` and
never embedded.
