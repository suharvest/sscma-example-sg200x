# reCamera Monocular Depth Model Quantization and Deployment Evaluation

## Background

The current reCamera pipeline already runs an object detection model. The existing YOLOv11n INT8 implementation takes approximately 49 ms per inference and the complete video pipeline runs at approximately 15 FPS.

We want to add monocular depth estimation for:

- Depth heatmap overlays on the video stream
- Relative near/far estimation of detected objects
- Representative depth sampling inside detection boxes

This phase targets **relative depth**. Accurate metric depth in meters is out of scope because it requires additional camera calibration and target-domain training.

## Objective

Convert, quantize, deploy, and benchmark lightweight monocular depth models on the reCamera SG2002/CV181x platform. The evaluation must determine:

1. Whether the model provides acceptable frame rate when used with the existing detector.
2. Whether it can be reliably converted and quantized with the CV181x TPU-MLIR toolchain.
3. Whether its depth quality remains acceptable after quantization.

The final outcome must recommend either ZipDepth or FastDepth for product integration, or document why neither model is suitable.

## Models to Evaluate

### Primary candidate: ZipDepth NPU

- Repository: <https://github.com/fabiotosi92/ZipDepth>
- Use the official NPU-friendly checkpoint and export path.
- Evaluate fixed input sizes of 384x384 and 320x320.
- Test BF16, full INT8, and INT8/BF16 mixed quantization.
- If full INT8 causes unsupported operations or unacceptable accuracy loss, keep attention, Softmax, Sigmoid, and the final depth head in BF16 through a quantization table.

### Performance baseline: FastDepth

- Repository: <https://github.com/dwofk/fast-depth>
- Use a fixed input size of 224x224.
- Prioritize full INT8 deployment.
- Use this model as the minimum-compute and maximum-frame-rate baseline.

Depth Anything V2 may be used on a development machine as a qualitative reference or teacher model, but it is not part of the initial on-device deployment scope.

## Calibration Dataset

Prepare approximately 500 calibration images:

| Source | Image count | Share |
|---|---:|---:|
| Frames captured with the target reCamera | 350 | 70% |
| DIODE indoor validation images | 75 | 15% |
| DIODE outdoor validation images | 75 | 15% |

DIODE dataset: <https://diode-dataset.org/>

Calibration images do not require depth labels. The dataset must include:

- Daylight, low-light, nighttime, and backlit scenes
- Indoor and outdoor scenes matching the intended deployment
- Nearby people and objects as well as distant backgrounds
- Low-texture regions such as walls, floors, and sky
- Complex textures, occlusion, and moderate motion blur

Do not fill the calibration set with consecutive or nearly identical video frames.

The images must use exactly the same preprocessing as deployment, including:

- RGB/BGR channel order
- Resize, crop, or letterbox behavior
- Mean and scale values
- Input resolution
- Pixel value range

## Implementation Tasks

1. Export both models to ONNX with fixed input dimensions.
2. Verify the ONNX graphs and identify operations unsupported by the current CV181x TPU-MLIR toolchain.
3. Convert each model to BF16 first and validate preprocessing, output orientation, and numerical behavior.
4. Generate an INT8 calibration table using the same calibration dataset for all comparable model variants.
5. Build the following models:
   - FastDepth INT8
   - ZipDepth BF16
   - ZipDepth INT8
   - ZipDepth mixed INT8/BF16 if full INT8 is not acceptable
6. Benchmark each model independently on the target reCamera device.
7. Load the depth model together with the existing YOLO model and measure combined memory usage and pipeline performance.
8. Implement basic depth post-processing:
   - Depth normalization
   - Pseudocolor mapping
   - Resize and letterbox reversal
   - Alpha-blended video overlay
9. Implement representative depth sampling for each detection. Prefer the median value from the lower-center region of the bounding box to reduce background and outlier contamination.
10. Evaluate reduced-frequency scheduling by running depth inference once every two or three frames and reusing the most recent depth map between inferences.

## Benchmark Procedure

For every model variant:

1. Warm up the model for at least 20 inferences.
2. Run at least 300 measured inferences.
3. Record mean, P50, and P95 inference latency.
4. Record the `.cvimodel` file size.
5. Record the model's `Need ION` value.
6. Measure standalone depth inference FPS.
7. Measure complete pipeline FPS with YOLO and depth enabled.
8. Run the combined pipeline continuously for at least 30 minutes and monitor crashes, allocation failures, memory stability, and visual artifacts.

All timing results must state the input size, precision, preprocessing path, and depth scheduling interval.

## Acceptance Criteria

### General requirements

- The model converts successfully to `.cvimodel`.
- The model loads and runs reliably on the target reCamera.
- Combined YOLO and depth model ION should remain at or below 35 MB and must not exceed approximately 40 MB.
- The depth map aligns correctly with the source video after resize, crop, or letterbox reversal.
- Quantized depth accuracy degradation relative to BF16 should not exceed 5% on the selected evaluation metric.
- The output must not exhibit constant maps, widespread depth reversal, severe banding, or major structural discontinuities introduced by quantization.

### FastDepth targets

- Full INT8 conversion succeeds.
- Target depth inference latency: no more than 35 ms.
- Target complete pipeline performance: at least 12 FPS.
- Preserve the results as the performance baseline even if its depth quality is insufficient for final use.

### ZipDepth targets

- BF16 conversion succeeds before INT8 optimization begins.
- Full INT8 should be attempted first; mixed INT8/BF16 is acceptable when necessary.
- Maximum target depth inference latency: 100 ms.
- Preferred depth inference latency: 80 ms or less.
- Minimum complete pipeline performance: 6 FPS.
- Preferred complete pipeline performance: at least 8 FPS.
- Depth boundaries and cross-domain behavior should be visibly better than FastDepth.

These latency and FPS values are engineering targets, not previously measured SG2002 results.

## Accuracy Evaluation

Use the following evaluation data:

- ZipDepth: DIODE validation set, reported separately for indoor and outdoor subsets.
- FastDepth: NYU Depth V2 for in-domain indoor accuracy, plus DIODE for cross-domain behavior.
- reCamera captures: qualitative and temporal-stability evaluation.

Compare:

- Original PyTorch or ONNX output
- BF16 `.cvimodel` output
- INT8 or mixed-precision `.cvimodel` output

Record at least:

- Absolute Relative Error (AbsRel)
- Delta accuracy at 1.25 (`delta1`)
- Depth boundary preservation
- Correct near/far ordering
- Temporal flicker in video
- Failure cases involving glass, mirrors, sky, low texture, and strong backlighting

Relative-depth models must be evaluated with the same scale-and-shift alignment procedure. Their raw output values must not be interpreted as meters.

For quantization regression, compare the quantized output against the BF16 or ONNX reference using identical inputs and preprocessing. Ground-truth depth is required only for measuring actual model accuracy, not for generating calibration tables.

## Expected Deliverables

1. Fixed-shape ONNX files for ZipDepth and FastDepth.
2. Successfully generated BF16, INT8, and applicable mixed-precision `.cvimodel` files.
3. Calibration image manifest and generated calibration tables.
4. Reproducible export, conversion, and deployment commands or scripts.
5. reCamera depth inference and pseudocolor overlay implementation.
6. A benchmark table covering latency, FPS, model size, and ION usage.
7. BF16-versus-INT8 comparison images and representative test videos.
8. A documented issue list covering unsupported operations, quantization degradation, ION pressure, and runtime failures.
9. A final recommendation with supporting measurements.

## Decision Rules

- Select ZipDepth if it reaches at least 8 FPS in the combined pipeline, fits within the memory budget, and retains stable quantized depth quality.
- Select FastDepth if ZipDepth cannot be converted reliably, exceeds the memory budget, or runs below 6 FPS.
- If the product requires at least 12 FPS, prefer FastDepth or schedule ZipDepth once every three to five frames.
- If neither model meets the required quality, proceed to a separate phase for target-domain fine-tuning or knowledge distillation into a lightweight CNN student.

## Out of Scope

- Accurate metric depth in meters
- Camera-specific metric depth calibration
- Training a new depth model from scratch
- Deployment of Depth Anything V2 on SG2002
- Multi-camera or stereo depth estimation
