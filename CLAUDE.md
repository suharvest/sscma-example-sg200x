# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSCMA Example for SG200X is a compilation framework for developing applications on the **ReCamera** platform - an embedded camera device using RISC-V SG2002 SoC. Applications are cross-compiled on x86_64 hosts and deployed as .deb packages to the device.

## Build Commands

### Docker Build (Recommended)

Use the `sophgo/tpuc_dev:v3.1` Docker image with the local toolchain and SDK mounted. A running container named `recamera` already exists.

```bash
# Build a solution using Docker (one-liner)
docker run --rm \
  -v /Users/harvest/project/recamera:/recamera \
  sophgo/tpuc_dev:v3.1 \
  bash -c "
    export PATH=/recamera/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
    export SG200X_SDK_PATH=/recamera/sg2002_recamera_emmc
    cd /recamera/sscma-example-sg200x/solutions/<solution_name>
    rm -rf build && mkdir build && cd build
    cmake .. && make -j\$(nproc)
  "
```

Key paths on this machine:
- **Toolchain**: `/Users/harvest/project/recamera/host-tools/gcc/riscv64-linux-musl-x86_64/bin/`
- **SDK**: `/Users/harvest/project/recamera/sg2002_recamera_emmc`
- **Project**: `/Users/harvest/project/recamera/sscma-example-sg200x`

### Native Build (if toolchain installed on host)

```bash
# Set up environment
export SG200X_SDK_PATH=/Users/harvest/project/recamera/sg2002_recamera_emmc
export PATH=/Users/harvest/project/recamera/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
```

### Build a Solution
```bash
cd solutions/<solution_name>
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Package for Deployment
```bash
cd build && cpack
# Creates: <name>-<version>.deb
```

### Deploy to Device
```bash
scp build/<package>.deb recamera@192.168.42.1:/tmp/
ssh recamera@192.168.42.1 "sudo opkg install /tmp/<package>.deb"
# Default sudo password: recamera
```

## Architecture

### Build System
- **cmake/toolchain-riscv64-linux-musl-x86_64.cmake**: Cross-compilation toolchain for RISC-V (c906fdv CPU, rv64gcv0p7_zfh_xthead arch)
- **cmake/project.cmake**: Includes components based on `REQUIREDS` list from solution's main/CMakeLists.txt
- **cmake/macro.cmake**: Defines `component_register()` function for registering components
- **cmake/package.cmake**: CPack configuration for .deb package generation

### Solution Structure
Each solution in `solutions/` follows this pattern:
```
solutions/<name>/
├── CMakeLists.txt       # Includes toolchain and project.cmake
├── main/
│   ├── CMakeLists.txt   # Uses component_register(COMPONENT_NAME main SRCS ... REQUIREDS ...)
│   └── main.cpp
├── control/             # Optional: preinst, postinst, prerm, postrm scripts
└── rootfs/              # Optional: Files installed to device root filesystem
```

### Components
Components in `components/` are reusable libraries:

- **sophgo**: SG200X platform HAL (video/audio interfaces, ISP integration)
- **sscma-micro**: SSCMA ML framework porting with CVINN engine support
- **mongoose**: HTTP server library
- **quirc**: QR code detection library

Components declare dependencies via `REQUIREDS` and `PRIVATE_REQUIREDS` in their CMakeLists.txt.

### Key Solutions
- **supervisor**: HTTP service and Web UI for device management
- **sscma-node**: Node-RED backend service using MQTT protocol (see docs/sscma-node-protocol.md)
- **sscma-model**: Example for model inference with SSCMA-Micro

## Code Style

Uses clang-format with these key settings:
- IndentWidth: 4, TabWidth: 4, UseTab: Never
- ColumnLimit: 200
- PointerAlignment: Left
- BraceWrapping: Attach style
- C++17 standard

## Platform Details

- Target: RISC-V 64-bit (riscv64-unknown-linux-musl)
- CPU: c906fdv
- SDK provides: cviruntime (TPU), cvi_rtsp, mosquitto, OpenSSL, libhv
- Device IP (USB): 192.168.42.1

## API Notes

### Tensor Quantization (ma_quant_param_t)
The quant param struct has scalar fields (not arrays):
```c
typedef struct {
    float scale;
    int32_t zero_point;
} ma_quant_param_t;
```
Usage: `tensor.quant_param.scale` and `tensor.quant_param.zero_point` (direct access, no indexing).
