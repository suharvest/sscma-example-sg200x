#!/bin/bash
#
# Convert FastDepth ONNX model to cvimodel for reCamera (cv181x)
#
# Model: dwofk/fast-depth, mobilenet-nnconv5dw-skipadd-pruned variant
# (ONNX exported via PINTO0309/PINTO_model_zoo #146, opset-bumped to 17
#  and re-simplified by scripts/export_fastdepth.py -- see README).
#
# This script should be run inside the sophgo/tpuc_dev:v3.1 Docker container,
# with this project directory mounted at /workspace.
#
# Usage (inside container):
#   cd /workspace
#   ./scripts/convert_to_cvimodel.sh --input-num 500
#   ./scripts/convert_to_cvimodel.sh --input-num 200   # if calibration OOMs, see README
#
# Options:
#   --model NAME       Model name (default: fastdepth_224)
#   --input-num N       Number of calibration images to use (default: 500)
#   --skip-setup        Skip tpu_mlir pip install (if already done)

set -e

MODEL_NAME="fastdepth_224"
INPUT_NUM=500
SKIP_SETUP=false
WORKSPACE_DIR="/workspace"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --input-num)
            INPUT_NUM="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "FastDepth to cvimodel Conversion"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Calibration input-num: ${INPUT_NUM}"
echo ""

# Step 1: Install tpu_mlir if not already installed
if ! command -v model_transform &> /dev/null; then
    echo "[Step 1] Installing tpu_mlir..."
    pip install tpu_mlir[all]==1.7
else
    echo "[Step 1] tpu_mlir already installed"
fi

# Step 2: Create model workspace
MODEL_WORKSPACE="${WORKSPACE_DIR}/model_workspace"
mkdir -p ${MODEL_WORKSPACE}
cd ${MODEL_WORKSPACE}

ONNX_PATH="${WORKSPACE_DIR}/${MODEL_NAME}.onnx"
if [ ! -f "${ONNX_PATH}" ]; then
    echo "Error: ONNX file not found: ${ONNX_PATH}"
    echo "Run scripts/export_fastdepth.py first."
    exit 1
fi
cp ${ONNX_PATH} .

CALIB_DIR="${WORKSPACE_DIR}/calib_set"
if [ ! -d "${CALIB_DIR}" ]; then
    echo "Error: calibration set not found: ${CALIB_DIR}"
    echo "Run: uv run python scripts/download_coco_calib.py --count 500"
    exit 1
fi

TEST_IMG=$(ls ${CALIB_DIR}/*.jpg | head -1)
echo "Using test image: ${TEST_IMG}"

# Step 3: Convert ONNX to MLIR
echo "[Step 3] Converting ONNX to MLIR..."
model_transform \
    --model_name ${MODEL_NAME} \
    --model_def ${MODEL_NAME}.onnx \
    --input_shapes "[[1,3,224,224]]" \
    --mean "0.0,0.0,0.0" \
    --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
    --pixel_format rgb \
    --test_input ${TEST_IMG} \
    --test_result ${MODEL_NAME}_top_outputs.npz \
    --mlir ${MODEL_NAME}.mlir

echo "MLIR conversion complete: ${MODEL_NAME}.mlir"

# Step 4: Calibration
echo "[Step 4] Running calibration for INT8 quantization (input_num=${INPUT_NUM})..."
run_calibration \
    ${MODEL_NAME}.mlir \
    --dataset ${CALIB_DIR} \
    --input_num ${INPUT_NUM} \
    -o ${MODEL_NAME}_calib_table

echo "Calibration complete: ${MODEL_NAME}_calib_table"

# Step 5: Deploy (MLIR -> cvimodel, INT8)
echo "[Step 5] Converting MLIR to cvimodel (INT8)..."
model_deploy \
    --mlir ${MODEL_NAME}.mlir \
    --quantize INT8 \
    --calibration_table ${MODEL_NAME}_calib_table \
    --processor cv181x \
    --test_input ${MODEL_NAME}_in_f32.npz \
    --test_reference ${MODEL_NAME}_top_outputs.npz \
    --tolerance 0.85,0.45 \
    --model ${MODEL_NAME}_int8.cvimodel

OUTPUT_MODEL="${MODEL_NAME}_int8.cvimodel"

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo "Output: ${MODEL_WORKSPACE}/${OUTPUT_MODEL}"
echo ""

# Step 6: Verify model
echo "[Step 6] Verifying model..."
model_tool --info ${OUTPUT_MODEL}

# Copy to workspace root
cp ${OUTPUT_MODEL} ${WORKSPACE_DIR}/
echo ""
echo "Model copied to: ${WORKSPACE_DIR}/${OUTPUT_MODEL}"
echo ""
echo "Deploy to reCamera:"
echo "  scp ${WORKSPACE_DIR}/${OUTPUT_MODEL} recamera@192.168.42.1:/userdata/local/models/"
