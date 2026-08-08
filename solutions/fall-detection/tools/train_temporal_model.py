#!/usr/bin/env python3
"""Train/export the tiny pose-sequence classifier used by fall-detection.

The split is intentionally subject-disjoint: subjects 1-2 train, subject 3
selects the configuration, and 27 clean subject-4 clips are touched only by
the final test. Ten earlier pipeline-smoke clips are permanently discarded.
The script expects JSONL traces produced by extract_pose_features.sh and the
public dataset CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


FPS = 15.0
WINDOW = 48
STRIDE = 3
JOINTS = 17
FRAME_DIM = JOINTS * 3 + 5
BINS = 6
FEATURE_DIM = FRAME_DIM * (BINS + 3)
FRAME_MASKS = {
    # Candidate selection is performed only inside the development subjects.
    "all": np.ones(FRAME_DIM, dtype=np.float32),
    "pose": np.asarray([1] * (JOINTS * 3) + [0, 0, 0, 0, 1], dtype=np.float32),
    # Pelvis-centred geometry is comparatively insensitive to clothing,
    # illumination and confidence calibration shifts between people.
    "shape": np.asarray([1] * (JOINTS * 2) + [0] * JOINTS + [0, 1, 1, 0, 1], dtype=np.float32),
}
SMOKE_TEST_CLIPS = {
    ("ADL", "01"), ("ADL", "05"), ("ADL", "10"), ("ADL", "15"), ("ADL", "20"),
    ("Fall", "01"), ("Fall", "05"), ("Fall", "09"), ("Fall", "13"), ("Fall", "17"),
}
EXPECTED_SUBJECT_CLIPS = {1: 32, 2: 48, 3: 43, 4: 37}


@dataclass
class Clip:
    path: Path
    subject: int
    label: int
    onset: float
    frames: np.ndarray
    heuristic_trigger_sec: float


def load_onsets(dataset: Path) -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    pattern = re.compile(r"fall(?:ing)?[^\[]*\[\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    for subject in range(1, 5):
        path = dataset / f"subject-{subject}" / "Fall.csv"
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[1:]:
            name = raw.split(",", 1)[0].strip()
            if not re.fullmatch(r"\d{2}\.mp4", name):
                continue
            starts = [float(m.group(1)) for m in pattern.finditer(raw)]
            if starts:
                result[(subject, name)] = min(starts)
    return result


def canonical_frame(row: dict) -> np.ndarray:
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    raw = row.get("pose17")
    if not isinstance(raw, list) or len(raw) != JOINTS:
        return out
    pose = np.asarray(raw, dtype=np.float32)
    if pose.shape != (JOINTS, 3) or not np.isfinite(pose).all():
        return out
    conf = np.clip(pose[:, 2], 0.0, 1.0)

    def midpoint(a: int, b: int) -> tuple[np.ndarray, float]:
        ca, cb = float(conf[a]), float(conf[b])
        weight = ca + cb
        if weight < 0.1:
            return np.zeros(2, dtype=np.float32), 0.0
        return (pose[a, :2] * ca + pose[b, :2] * cb) / weight, weight * 0.5

    hips, hip_conf = midpoint(11, 12)
    shoulders, shoulder_conf = midpoint(5, 6)
    visible = conf >= 0.1
    if hip_conf < 0.1:
        if not visible.any():
            return out
        hips = np.average(pose[visible, :2], axis=0, weights=conf[visible])

    torso = float(np.linalg.norm(shoulders - hips)) if shoulder_conf >= 0.1 else 0.0
    if torso < 0.04:
        pts = pose[visible, :2]
        torso = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.35) if len(pts) else 0.0
    scale = max(torso, 0.04)
    xy = np.clip((pose[:, :2] - hips) / scale, -4.0, 4.0)
    xy[~visible] = 0.0
    out[: JOINTS * 2] = xy.reshape(-1)
    out[JOINTS * 2 : JOINTS * 3] = conf

    f = row.get("features") or {}
    out[-5:] = [
        float(f.get("hip_y", 0.0)),
        float(f.get("torso_angle_deg", 0.0)) / 90.0,
        min(float(f.get("bbox_aspect_ratio", 0.0)), 4.0) / 4.0,
        float(f.get("person_score", 0.0)),
        1.0 if row.get("tracking") else 0.0,
    ]
    return out


def sequence_feature(frames: np.ndarray, end: int, frame_mask: np.ndarray | None = None) -> np.ndarray:
    start = end - WINDOW + 1
    if start < 0:
        prefix = np.repeat(frames[[0]], -start, axis=0)
        seq = np.concatenate([prefix, frames[: end + 1]], axis=0)
    else:
        seq = frames[start : end + 1]
    assert seq.shape == (WINDOW, FRAME_DIM)
    if frame_mask is not None:
        seq = seq * frame_mask
    means = seq.reshape(BINS, WINDOW // BINS, FRAME_DIM).mean(axis=1).reshape(-1)
    std = seq.std(axis=0)
    delta = seq[-1] - seq[0]
    span = seq.max(axis=0) - seq.min(axis=0)
    return np.concatenate([means, std, delta, span]).astype(np.float32)


def read_trace(path: Path) -> tuple[np.ndarray, float]:
    frames: list[np.ndarray] = []
    heuristic_trigger_sec = math.inf
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{") or '"summary"' in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames.append(canonical_frame(row))
        if row.get("fall_event") and not math.isfinite(heuristic_trigger_sec):
            heuristic_trigger_sec = float(row.get("timestamp", (len(frames) - 1) / FPS * 1000)) / 1000.0
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames), heuristic_trigger_sec


def load_clips(traces: Path, dataset: Path,
               allowed_subjects: set[int] | None = None) -> list[Clip]:
    onsets = load_onsets(dataset)
    clips: list[Clip] = []
    for path in sorted(traces.glob("subject-*/*/*.jsonl")):
        m = re.search(r"subject-(\d+)/(ADL|Fall)/(\d{2})\.jsonl$", path.as_posix())
        if not m:
            continue
        subject, cls, number = int(m.group(1)), m.group(2), m.group(3)
        if allowed_subjects is not None and subject not in allowed_subjects:
            continue
        frames, heuristic_trigger_sec = read_trace(path)
        duration = len(frames) / FPS
        label = int(cls == "Fall")
        onset = onsets.get((subject, f"{number}.mp4"), duration * 0.35) if label else math.inf
        clips.append(Clip(path, subject, label, onset, frames, heuristic_trigger_sec))
    return clips


def load_external_clips(traces: Path, manifest_path: Path) -> list[Clip]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips: list[Clip] = []
    for row in manifest:
        if row.get("upstream_subset") != "testing":
            continue
        stem = Path(row["path"]).stem
        trace = traces / "external" / "Fall" / f"{stem}.jsonl"
        frames, heuristic_trigger_sec = read_trace(trace)
        clips.append(Clip(
            trace, 5, 1, float(row["onset_sec"]), frames, heuristic_trigger_sec))
    if len(clips) != 34:
        raise SystemExit(f"expected 34 RealBiomFall test clips, got {len(clips)}")
    return clips


def clip_windows(clip: Clip, for_training: bool, frame_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for end in range(0, len(clip.frames), STRIDE):
        t = end / FPS
        if clip.label:
            if t < clip.onset - 0.25:
                y = 0
            elif t >= clip.onset + 0.30:
                y = 1
            elif for_training:
                continue
            else:
                y = 0
        else:
            y = 0
        xs.append(sequence_feature(clip.frames, end, frame_mask))
        ys.append(y)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def training_matrix(clips: list[Clip], subjects: set[int], seed: int,
                    frame_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = [clip_windows(c, True, frame_mask) for c in clips if c.subject in subjects]
    x = np.concatenate([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    # Keep broad ADL coverage while preventing the much longer negative clips
    # from overwhelming transition examples.
    keep_neg = rng.choice(neg, size=min(len(neg), max(len(pos) * 2, 1)), replace=False)
    keep = np.concatenate([pos, keep_neg])
    rng.shuffle(keep)
    return x[keep], y[keep]


def fit_model(clips: list[Clip], subjects: set[int], hidden: int, alpha: float,
              seed: int, frame_mask: np.ndarray):
    x, y = training_matrix(clips, subjects, seed, frame_mask)
    scaler = StandardScaler().fit(x)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
        alpha=alpha, batch_size=128, learning_rate_init=1e-3,
        max_iter=400, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=seed,
    ).fit(scaler.transform(x), y)
    return scaler, model


def clip_probability(clip: Clip, scaler: StandardScaler, model: MLPClassifier,
                     frame_mask: np.ndarray) -> np.ndarray:
    x, _ = clip_windows(clip, False, frame_mask)
    return model.predict_proba(scaler.transform(x))[:, 1]


def predict_clip(probs: np.ndarray, threshold: float, consecutive: int) -> int:
    run = 0
    for p in probs:
        run = run + 1 if p >= threshold else 0
        if run >= consecutive:
            return 1
    return 0


def trigger_time(probs: np.ndarray, threshold: float, consecutive: int) -> float:
    run = 0
    for i, p in enumerate(probs):
        run = run + 1 if p >= threshold else 0
        if run >= consecutive:
            return i * STRIDE / FPS
    return math.inf


def trigger_metrics(clips: list[Clip], triggers: dict[Path, float]) -> dict:
    # Alerts more than 0.5 s before the annotated fall onset are counted as an
    # early false alarm, not as a successful fall prediction.
    truth = [c.label for c in clips]
    pred: list[int] = []
    latencies: list[float] = []
    early_fall_alerts = 0
    for clip in clips:
        when = triggers[clip.path]
        if clip.label:
            early = math.isfinite(when) and when < clip.onset - 0.5
            if early:
                early_fall_alerts += 1
            detected = math.isfinite(when) and not early
            pred.append(int(detected))
            if detected:
                latencies.append(when - clip.onset)
        else:
            pred.append(int(math.isfinite(when)))
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / max(len(truth), 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    def display_path(path: Path) -> str:
        parts = path.parts
        for index, part in enumerate(parts):
            if part.startswith("subject-") or part == "external":
                return Path(*parts[index:]).as_posix()
        return path.name

    return {
        "n": len(truth), "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
        "accuracy": accuracy, "recall": recall, "specificity": specificity,
        "precision": precision, "f1": f1, "early_fall_alerts": early_fall_alerts,
        "mean_detection_latency_sec": float(np.mean(latencies)) if latencies else None,
        "median_detection_latency_sec": float(np.median(latencies)) if latencies else None,
        "misclassified": [display_path(c.path) for c, y in zip(clips, pred) if y != c.label],
    }


def metrics(clips: list[Clip], probs: dict[Path, np.ndarray], threshold: float, consecutive: int) -> dict:
    triggers = {c.path: trigger_time(probs[c.path], threshold, consecutive) for c in clips}
    return trigger_metrics(clips, triggers)


def export_header(path: Path, scaler: StandardScaler, model: MLPClassifier,
                  threshold: float, consecutive: int, frame_mask: np.ndarray) -> None:
    w1 = model.coefs_[0].astype(np.float32)
    b1 = model.intercepts_[0].astype(np.float32)
    w2 = model.coefs_[1].reshape(-1).astype(np.float32)
    b2 = float(model.intercepts_[1][0])

    def literal(value: float) -> str:
        rendered = f"{float(value):.9g}"
        if "." not in rendered and "e" not in rendered.lower():
            rendered += ".0"
        return rendered + "f"

    def array(name: str, values: np.ndarray) -> str:
        flat = values.reshape(-1)
        body = ",\n    ".join(literal(v) for v in flat)
        return f"inline constexpr float {name}[{len(flat)}] = {{\n    {body}\n}};\n"

    content = """// Generated by tools/train_temporal_model.py. Do not hand edit.\n#pragma once\n\nnamespace fall::temporal_weights {\n"""
    content += f"inline constexpr int kWindow = {WINDOW};\n"
    content += f"inline constexpr int kFrameDim = {FRAME_DIM};\n"
    content += f"inline constexpr int kFeatureDim = {FEATURE_DIM};\n"
    content += f"inline constexpr int kHiddenDim = {w1.shape[1]};\n"
    content += f"inline constexpr float kThreshold = {literal(threshold)};\n"
    content += f"inline constexpr int kConsecutive = {consecutive};\n"
    content += array("kFrameMask", frame_mask.astype(np.float32))
    content += array("kMean", scaler.mean_.astype(np.float32))
    content += array("kScale", scaler.scale_.astype(np.float32))
    content += array("kW1", w1)
    content += array("kB1", b1)
    content += array("kW2", w2)
    content += f"inline constexpr float kB2 = {literal(b2)};\n"
    content += "}  // namespace fall::temporal_weights\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--external-traces", type=Path)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--freeze-only", action="store_true",
                        help="select on subjects 1-3 and do not read subject 4")
    parser.add_argument("--frozen-report", type=Path,
                        help="reuse a previously frozen configuration; never reselect on test data")
    args = parser.parse_args()

    clips = load_clips(
        args.traces, args.dataset, {1, 2, 3} if args.freeze_only else None)
    counts = {s: sum(c.subject == s for c in clips) for s in range(1, 5)}
    required_subjects = (1, 2, 3) if args.freeze_only else (1, 2, 3, 4)
    if any(counts[s] != EXPECTED_SUBJECT_CLIPS[s] for s in required_subjects):
        raise SystemExit(f"required subject splits are incomplete, got {counts}")

    training = [c for c in clips if c.subject in {1, 2}]
    validation = [c for c in clips if c.subject == 3]
    if args.frozen_report is not None:
        best = json.loads(args.frozen_report.read_text(encoding="utf-8"))["best"]
    else:
        candidates: list[dict] = []
        for variant, frame_mask in FRAME_MASKS.items():
            for hidden in (16, 32):
                for alpha in (1e-3, 1e-2):
                    scaler, model = fit_model(
                        clips, {1, 2}, hidden, alpha, 2026, frame_mask)
                    validation_probs = {
                        c.path: clip_probability(c, scaler, model, frame_mask)
                        for c in validation
                    }
                    for threshold in np.arange(0.30, 0.81, 0.05):
                        for consecutive in (1, 2, 3):
                            validation_metrics = metrics(
                                validation, validation_probs, float(threshold), consecutive)
                            candidates.append({
                                "variant": variant, "hidden": hidden, "alpha": alpha,
                                "threshold": float(threshold), "consecutive": consecutive,
                                "validation_f1": validation_metrics["f1"],
                                "validation_balanced_accuracy": (
                                    validation_metrics["recall"] +
                                    validation_metrics["specificity"]
                                ) * 0.5,
                                "validation": validation_metrics,
                            })
        best = max(candidates, key=lambda c: (
            c["validation_f1"], c["validation_balanced_accuracy"],
            c["consecutive"], c["threshold"], -c["hidden"], c["alpha"],
        ))
    best_mask = FRAME_MASKS[best["variant"]]
    scaler, model = fit_model(clips, {1, 2, 3}, best["hidden"], best["alpha"], 2026, best_mask)
    heuristic_validation = trigger_metrics(
        validation, {c.path: c.heuristic_trigger_sec for c in validation})
    holdout: list[Clip] = []
    holdout_metrics = None
    heuristic_holdout = None
    if not args.freeze_only:
        # Ten subject-4 clips were used only for an early end-to-end smoke run
        # and are excluded permanently. The remaining 27 are the clean test.
        holdout = [
            c for c in clips if c.subject == 4 and
            (c.path.parent.name, c.path.stem) not in SMOKE_TEST_CLIPS
        ]
        if len(holdout) != 27:
            raise SystemExit(f"expected 27 clean subject-4 test clips, got {len(holdout)}")
        test_probs = {
            c.path: clip_probability(c, scaler, model, best_mask) for c in holdout
        }
        holdout_metrics = metrics(
            holdout, test_probs, best["threshold"], best["consecutive"])
        heuristic_holdout = trigger_metrics(
            holdout, {c.path: c.heuristic_trigger_sec for c in holdout})
    external_metrics = None
    heuristic_external = None
    if args.external_traces is not None or args.external_manifest is not None:
        if args.external_traces is None or args.external_manifest is None:
            raise SystemExit("--external-traces and --external-manifest must be used together")
        external = load_external_clips(args.external_traces, args.external_manifest)
        external_probs = {
            c.path: clip_probability(c, scaler, model, best_mask) for c in external
        }
        external_metrics = metrics(
            external, external_probs, best["threshold"], best["consecutive"])
        heuristic_external = trigger_metrics(
            external, {c.path: c.heuristic_trigger_sec for c in external})

    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_header(args.header, scaler, model, best["threshold"], best["consecutive"], best_mask)
    report = {
        "protocol": (
            "train subjects 1-2; validate subject 3; refit subjects 1-3 after configuration "
            "freeze; test on 27 untouched subject-4 clips (10 prior smoke clips excluded)"
        ),
        "clips_by_subject": counts,
        "phase": "configuration_freeze" if args.freeze_only else "frozen_test",
        "split_sizes": {
            "training": len(training), "validation": len(validation), "test": len(holdout),
            "discarded_subject4_smoke": len(SMOKE_TEST_CLIPS),
        },
        "feature_dim": FEATURE_DIM,
        "window_frames": WINDOW,
        "stride_frames": STRIDE,
        "best": best,
        "validation": best["validation"],
        "subject4_clean_test": holdout_metrics,
        "heuristic_validation": heuristic_validation,
        "heuristic_subject4_clean_test": heuristic_holdout,
        "realbiomfall_external_test": external_metrics,
        "heuristic_realbiomfall_external_test": heuristic_external,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "best": best, "validation": best["validation"],
        "subject4_clean_test": holdout_metrics,
        "heuristic_validation": heuristic_validation,
        "heuristic_subject4_clean_test": heuristic_holdout,
        "realbiomfall_external_test": external_metrics,
        "heuristic_realbiomfall_external_test": heuristic_external,
    }, indent=2))


if __name__ == "__main__":
    main()
