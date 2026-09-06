#!/usr/bin/env python3
"""
Export FastDepth (dwofk/fast-depth, mobilenet-nnconv5dw-skipadd-pruned)
to a clean, fixed-shape 224x224 ONNX file ready for tpu_mlir.

Weight source
-------------
The official pretrained checkpoint (mobilenet-nnconv5dw-skipadd-pruned.pth.tar)
is hosted at http://datasets.lids.mit.edu/fastdepth/results/, which was
unreachable from this network (connection timeout, both direct and via the
local HTTP proxy -- see README "模型来源" section).

Instead this script starts from the already-exported ONNX graph published by
PINTO0309/PINTO_model_zoo (entry #146_FastDepth, sourced from the same
dwofk/fast-depth checkpoint via openvino2tensorflow), fetched as:

    https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/146_FastDepth/resources.tar.gz
    -> saved_model_224x224/fast_depth_224x224.onnx

That file already has fixed 1x3x224x224 input / opset 11 / IR 6, and its op
set is already exactly {Conv, Clip, Relu, Resize, Add} (verified below) --
identical to the reference recipe's expectation. This script:
  1. runs onnxsim to fully constant-fold/clean the graph
  2. upgrades the opset to 17 (tpu_mlir / cv181x requires opset<=17; PINTO's
     export is opset 11, which is below the ">=13" floor asked for)
  3. runs onnxsim again on the upgraded graph
  4. verifies the final op set is still Conv/Clip/Relu/Resize/Add only

Usage:
    uv run python scripts/export_fastdepth.py \\
        --input pretrained/fast_depth_224x224_raw.onnx \\
        --output fastdepth_224.onnx
"""

import argparse
from pathlib import Path

import onnx
from onnx import version_converter
from onnxsim import simplify


ALLOWED_OPS = {"Conv", "Clip", "Relu", "Resize", "Add", "Constant"}


def op_histogram(model: onnx.ModelProto) -> dict:
    hist: dict[str, int] = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return hist


def export(input_path: Path, output_path: Path, target_opset: int = 17) -> None:
    print(f"Loading {input_path} ...")
    model = onnx.load(str(input_path))
    print(f"  IR version: {model.ir_version}, opset: {[o.version for o in model.opset_import]}")
    print(f"  ops: {op_histogram(model)}")

    print("Running onnxsim (pass 1) ...")
    model_sim, ok = simplify(model)
    assert ok, "onnxsim (pass 1) failed"

    cur_opset = next((o.version for o in model_sim.opset_import if o.domain in ("", "ai.onnx")), 0)
    if cur_opset < target_opset:
        print(f"Upgrading opset {cur_opset} -> {target_opset} ...")
        model_sim = version_converter.convert_version(model_sim, target_opset)
        onnx.checker.check_model(model_sim)

    print("Running onnxsim (pass 2, post opset upgrade) ...")
    model_sim, ok = simplify(model_sim)
    assert ok, "onnxsim (pass 2) failed"

    hist = op_histogram(model_sim)
    print(f"Final ops: {hist}")
    unexpected = set(hist) - ALLOWED_OPS
    if unexpected:
        raise SystemExit(
            f"ERROR: unexpected op types after simplification: {unexpected}. "
            f"Expected only {ALLOWED_OPS}."
        )

    out_shapes = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in model_sim.graph.output]
    in_shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in model_sim.graph.input]
    print(f"Input shapes: {in_shapes}")
    print(f"Output shapes: {out_shapes}")

    onnx.save(model_sim, str(output_path))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export FastDepth to clean 224x224 ONNX")
    parser.add_argument("--input", type=Path, default=Path("pretrained/fast_depth_224x224_raw.onnx"))
    parser.add_argument("--output", type=Path, default=Path("fastdepth_224.onnx"))
    parser.add_argument("--target-opset", type=int, default=17)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input not found: {args.input}\n"
            "Download it first, e.g.:\n"
            "  curl -sL -o pretrained/resources.tar.gz \\\n"
            "    https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/146_FastDepth/resources.tar.gz\n"
            "  tar -xzf pretrained/resources.tar.gz saved_model_224x224/fast_depth_224x224.onnx\n"
            "  mv saved_model_224x224/fast_depth_224x224.onnx pretrained/fast_depth_224x224_raw.onnx"
        )

    export(args.input, args.output, args.target_opset)


if __name__ == "__main__":
    main()
