#!/usr/bin/env python3
"""
Check and optionally downgrade ONNX model version for tpu_mlir compatibility.

reCamera (cv181x) requirements:
- IR version: <= 8
- Opset version: <= 17
"""

import argparse
import subprocess
import sys
from pathlib import Path

import onnx


def check_onnx_version(model_path: Path) -> tuple[int, int]:
    """Check the IR and Opset versions of an ONNX model."""
    model = onnx.load(str(model_path))

    ir_version = model.ir_version

    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    return ir_version, opset_version


def print_model_info(model_path: Path) -> tuple[int, int]:
    """Print ONNX model version information."""
    ir_version, opset_version = check_onnx_version(model_path)

    print(f"Model: {model_path}")
    print(f"  IR Version: {ir_version}")
    print(f"  Opset Version: {opset_version}")

    target_ir = 8
    target_opset = 17

    if ir_version <= target_ir and opset_version <= target_opset:
        print(f"  Status: Compatible with tpu_mlir (IR<={target_ir}, Opset<={target_opset})")
    else:
        print(f"  Status: INCOMPATIBLE - needs downgrade to IR<={target_ir}, Opset<={target_opset}")
        if ir_version > target_ir:
            print(f"    - IR version {ir_version} > {target_ir}")
        if opset_version > target_opset:
            print(f"    - Opset version {opset_version} > {target_opset}")

    return ir_version, opset_version


def downgrade_onnx(
    input_path: Path,
    output_path: Path,
    target_ir: int = 8,
    target_opset: int = 17,
) -> bool:
    """Downgrade ONNX model using ONNX_Downgrade tool."""
    downgrade_repo = Path("ONNX_Downgrade")
    downgrade_script = downgrade_repo / "downgrade_onnx.py"

    if not downgrade_script.exists():
        print("ONNX_Downgrade tool not found. Cloning repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/jjjadand/ONNX_Downgrade.git"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to clone ONNX_Downgrade: {result.stderr}")
            return False

    print(f"Downgrading {input_path} to IR={target_ir}, Opset={target_opset}...")
    result = subprocess.run(
        [
            sys.executable,
            str(downgrade_script),
            str(input_path),
            str(output_path),
            "--target_ir_version",
            str(target_ir),
            "--target_opset_version",
            str(target_opset),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Downgrade failed: {result.stderr}")
        return False

    print(f"Downgraded model saved to: {output_path}")

    print("\nVerifying downgraded model:")
    print_model_info(output_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check and downgrade ONNX model version for tpu_mlir"
    )
    parser.add_argument("model", type=Path, help="Path to ONNX model file")
    parser.add_argument("--downgrade", action="store_true", help="Downgrade model if incompatible")
    parser.add_argument("--output", type=Path, help="Output path for downgraded model")
    parser.add_argument("--target-ir", type=int, default=8, help="Target IR version (default: 8)")
    parser.add_argument("--target-opset", type=int, default=17, help="Target opset version (default: 17)")

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    ir_version, opset_version = print_model_info(args.model)

    needs_downgrade = ir_version > args.target_ir or opset_version > args.target_opset

    if args.downgrade and needs_downgrade:
        output_path = args.output or args.model.with_stem(f"{args.model.stem}_v{args.target_ir}")
        success = downgrade_onnx(args.model, output_path, args.target_ir, args.target_opset)
        sys.exit(0 if success else 1)
    elif args.downgrade and not needs_downgrade:
        print("\nModel already compatible, no downgrade needed.")
    elif needs_downgrade:
        print("\nRun with --downgrade to convert to compatible version.")
        sys.exit(1)


if __name__ == "__main__":
    main()
