# Face Analysis Models

This directory contains the ONNX models and conversion scripts for the face-analysis solution.

## Downloaded Models

| Model | File | Size | Input | Output | Source |
|-------|------|------|-------|--------|--------|
| Face Detection | `onnx/det_500m.onnx` | 2.4MB | 1x3x640x640 (RGB) | Bboxes + keypoints | [InsightFace SCRFD](https://github.com/deepinsight/insightface) |
| Gender + Age | `onnx/genderage.onnx` | 1.3MB | 1x3x96x96 (RGB) | [gender, age] | [InsightFace](https://github.com/deepinsight/insightface) |
| Emotion | `onnx/emotion-ferplus-8.onnx` | 33MB | 1x1x64x64 (Gray) | 8 emotion scores | [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/emotion-ferplus-8) |

## Emotion Classes (FER+)

0. neutral
1. happiness
2. surprise
3. sadness
4. anger
5. disgust
6. fear
7. contempt

## Model Conversion to cvimodel

Follow the [ReCamera Model Conversion Guide](https://wiki.seeedstudio.com/recamera_model_conversion/).

### Prerequisites

```bash
# Install tpu_mlir (use Docker recommended)
docker pull sophgo/tpuc_dev:latest
docker run -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### Calibration Dataset

Use MS1M-ArcFace dataset for calibration:
```
/Users/harvest/project/datasets/ms1m-arcface
```

### Convert SCRFD Face Detection

```bash
# ONNX -> MLIR
model_transform \
  --model_name scrfd_500m \
  --model_def onnx/det_500m.onnx \
  --input_shapes "[[1,3,640,640]]" \
  --mean "127.5,127.5,127.5" \
  --scale "0.0078125,0.0078125,0.0078125" \
  --pixel_format rgb \
  --mlir mlir/scrfd_500m.mlir

# Calibration
run_calibration mlir/scrfd_500m.mlir \
  --dataset /path/to/calibration_images \
  --input_num 200 \
  -o calib/scrfd_500m_calib_table

# MLIR -> INT8 cvimodel
model_deploy \
  --mlir mlir/scrfd_500m.mlir \
  --quantize INT8 \
  --calibration_table calib/scrfd_500m_calib_table \
  --processor cv181x \
  --model cvimodel/scrfd_500m_int8.cvimodel
```

### Convert GenderAge

```bash
# ONNX -> MLIR
model_transform \
  --model_name genderage \
  --model_def onnx/genderage.onnx \
  --input_shapes "[[1,3,96,96]]" \
  --mean "127.5,127.5,127.5" \
  --scale "0.0078125,0.0078125,0.0078125" \
  --pixel_format rgb \
  --mlir mlir/genderage.mlir

# Calibration
run_calibration mlir/genderage.mlir \
  --dataset /path/to/face_crops \
  --input_num 200 \
  -o calib/genderage_calib_table

# MLIR -> INT8 cvimodel
model_deploy \
  --mlir mlir/genderage.mlir \
  --quantize INT8 \
  --calibration_table calib/genderage_calib_table \
  --processor cv181x \
  --model cvimodel/genderage_int8.cvimodel
```

### Convert Emotion (FER+)

```bash
# ONNX -> MLIR (note: grayscale input)
model_transform \
  --model_name emotion \
  --model_def onnx/emotion-ferplus-8.onnx \
  --input_shapes "[[1,1,64,64]]" \
  --pixel_format gray \
  --mlir mlir/emotion.mlir

# Calibration (requires grayscale face images 64x64)
run_calibration mlir/emotion.mlir \
  --dataset /path/to/gray_face_crops \
  --input_num 200 \
  -o calib/emotion_calib_table

# MLIR -> INT8 cvimodel
model_deploy \
  --mlir mlir/emotion.mlir \
  --quantize INT8 \
  --calibration_table calib/emotion_calib_table \
  --processor cv181x \
  --model cvimodel/emotion_int8.cvimodel
```

## Expected INT8 Model Sizes

After quantization, approximate sizes:
- `scrfd_500m_int8.cvimodel`: ~1MB
- `genderage_int8.cvimodel`: ~0.5MB
- `emotion_int8.cvimodel`: ~8MB

**Total: ~10MB** (fits comfortably in 256MB device)

## Alternative Smaller Emotion Models

If memory is tight, consider these alternatives:
- MobileNetV2-based emotion model (~3MB INT8)
- MiniTR (Mini Transformer) - only 69K parameters
- Custom distilled model from FER+

## References

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [ReCamera Model Conversion](https://wiki.seeedstudio.com/recamera_model_conversion/)
- [yakhyo/facial-analysis](https://github.com/yakhyo/facial-analysis) - Pre-converted ONNX models
