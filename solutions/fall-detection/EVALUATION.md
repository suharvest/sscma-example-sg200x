# Fall Detection 0.2.0 evaluation

This is a reproducible device-level benchmark, not a medical-safety
certification. It measures staged/public videos and cannot establish the false
alarm rate of a particular home, camera angle, population, or care workflow.

## Device path

- Physical SG2002 reCamera at `192.168.42.1`.
- CV181x INT8 YOLO11n-Pose on the TPU; the same decoder and COCO-17 features
  used by the live application.
- FFmpeg input at 15 FPS, aspect-preserving 640x640 letterbox, RGB888 streamed
  to the application's offline mode.
- One JSONL result per frame. A clip is positive when the classifier reaches
  the frozen threshold for three consecutive 0.2-second evaluations.
- An alert more than 0.5 seconds before the published fall onset is counted as
  an early false alarm, not a true positive.

All 160 GMDCSA-24 v2.1 videos were decoded and run on the device. The public
dataset has an uneven per-subject distribution: 32 / 48 / 43 / 37 clips.

## Strict split

| Purpose | Clips | Use |
|---|---:|---|
| Train | Subjects 1–2, 80 | Fit scaler and MLP weights |
| Validation | Subject 3, 43 | Select feature mask, hidden size, regularization, threshold, consecutive count |
| Discarded | Subject 4, 10 | Earlier pipeline smoke clips; never used for fitting or final metrics |
| Clean test | Remaining Subject 4, 27 | Read once after configuration freeze |

`tools/train_temporal_model.py --freeze-only` does not load Subject 4. The
test phase requires the resulting frozen report and cannot reselect a
configuration. The frozen model uses pelvis-centred pose/confidence features,
a 48-frame (3.2-second) window, one 16-unit hidden layer, threshold 0.8 and
three consecutive positives. The generated weight header SHA-256 is recorded
in the release artifacts.

## Results

| Set / method | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation, v0.2 learned | 21 | 0 | 20 | 2 | 95.3% | 100.0% | 90.9% | 91.3% | 95.5% | 0.97 s |
| Validation, v0.1 geometry | 7 | 14 | 22 | 0 | 67.4% | 33.3% | 100.0% | 100.0% | 50.0% | 1.39 s |
| Clean test, v0.2 learned | 10 | 2 | 10 | 5 | 74.1% | 83.3% | 66.7% | 66.7% | 74.1% | 1.75 s |
| Clean test, v0.1 geometry | 6 | 6 | 14 | 1 | 74.1% | 50.0% | 93.3% | 85.7% | 63.2% | 2.04 s |
| RealBiomFall external, v0.2 learned | 20 | 14 | — | — | — | 58.8% | — | — | — | 1.18 s |
| RealBiomFall external, v0.1 geometry | 4 | 30 | — | — | — | 11.8% | — | — | — | 2.17 s |

The learned model improves fall recall by 33.3 percentage points and F1 by
10.9 points on the clean test, but reduces specificity by 26.6 points. Overall
accuracy is unchanged because the additional true positives and false
positives balance. Treat 0.2.0 as a high-recall beta, not as an across-the-board
accuracy increase.

The clean-test false positives are Subject 4 ADL clips 06, 07, 08, 16 and 17.
The two missed/early falls are clips 04 and 16. These filenames are disclosed
for future hard-negative collection; they were not used to retune this release.

## External distribution check

RealBiomFall (CC BY 4.0, DOI `10.5281/zenodo.11620083`) supplies 100 realistic
web-video fall clips. Its upstream `testing` subset contains 34 fall-only clips,
so it measures cross-domain recall and latency but **cannot** measure
specificity or accuracy. The same physical reCamera path is used. Results are
added only after the GMDCSA configuration and test report are frozen. The
frozen v0.2 model detects 20/34 falls (58.8% recall; mean latency 1.18 seconds),
versus 4/34 (11.8%; 2.17 seconds) for the v0.1 geometry rules. Nine v0.2 alerts
occur more than 0.5 seconds before the annotated onset and are kept separate
from the 20 true positives. This improvement is substantial, but 58.8% recall
is not sufficient for a safety-critical deployment. Long shots, occlusion and
low pose-tracking coverage are the main next data/model targets.

## Sources

- GMDCSA-24 repository: https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
- GMDCSA-24 paper: https://doi.org/10.1016/j.dib.2024.110892
- RealBiomFall: https://doi.org/10.5281/zenodo.11620083
- Ultralytics pose task: https://docs.ultralytics.com/tasks/pose/
